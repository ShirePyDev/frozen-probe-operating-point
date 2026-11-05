"""Why does CKA fail to predict frozen-probe damage? A mechanistic test.

Hypothesis
----------
CKA is invariant to orthogonal transformation and to isotropic rescaling of a
representation. A linear probe with FROZEN weights is invariant to neither: if
the representation is rotated, the probe's decision hyperplane no longer points
along the direction it was fitted to, and its outputs degrade - while CKA, by
construction, does not move at all.

If that is the mechanism, then the change induced by an update should be well
approximated by an orthogonal map plus a rescaling: the part of the change that
CKA is blind to should be large, and the part it can see should be small.

What is measured, per layer
---------------------------
  cka                    the rotation-invariant similarity (for reference)
  procrustes_r2          fraction of B's variance explained by an OPTIMAL
                         orthogonal map applied to A. High = the change is
                         mostly a rotation/reflection, which CKA cannot see.
  rotation_angle_deg     mean principal angle of that optimal orthogonal map.
                         Large angle + high procrustes_r2 = "same geometry,
                         moved basis", precisely the regime that breaks a frozen
                         probe while leaving CKA near 1.
  scale_ratio            ||B|| / ||A|| in Frobenius norm. A pure isotropic
                         rescale is also invisible to CKA but shifts every score
                         a frozen THRESHOLD depends on.
  identity_r2            fraction of B's variance explained by the IDENTITY map,
                         i.e. how well A predicts B with no alignment at all.
                         The gap between procrustes_r2 and identity_r2 is the
                         part of the change that is pure basis movement.

Interpretation key
------------------
  procrustes_r2 high AND identity_r2 low  -> change is mostly basis movement.
     CKA stays near 1, a frozen probe breaks, a REFIT probe recovers fully.
  both low                                -> genuine loss of the information
     the probe needs; refitting will not fully recover it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repdist import linear_cka  # noqa: E402
from provenance import (  # noqa: E402
    canonical_hash,
    ensure_utf8_stdout,
    provenance_block,
    sha256_file,
    write_results,
)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "hidden"


def load(repo: str, rev: str, axis: str, part: str):
    p = CACHE / f"{repo.replace('/', '__')}@{rev[:12]}__{axis}__{part}__mean.npz"
    if not p.exists():
        raise FileNotFoundError(f"missing cache {p.name}")
    return np.load(p, allow_pickle=False), p


def procrustes(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Optimal orthogonal map R minimising ||A R - B||_F, plus diagnostics.

    Both inputs are column-centred first, so this measures the change in shape
    and orientation rather than a shift of the mean.
    """
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))

    # R = U V^T from the SVD of A^T B  (classic orthogonal Procrustes)
    m = a.T @ b
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    r = u @ vt

    resid_rot = float(np.linalg.norm(a @ r - b) ** 2)
    resid_id = float(np.linalg.norm(a - b) ** 2)
    denom = float(np.linalg.norm(b) ** 2)

    # Mean principal angle of R, from the eigenvalue phases of an orthogonal
    # matrix: trace(R) / dim = mean cosine of the rotation angles.
    dim = r.shape[0]
    mean_cos = float(np.trace(r) / dim)
    mean_cos = max(-1.0, min(1.0, mean_cos))
    angle = float(np.degrees(np.arccos(mean_cos)))

    return {
        "procrustes_r2": 1.0 - resid_rot / denom if denom > 0 else None,
        "identity_r2": 1.0 - resid_id / denom if denom > 0 else None,
        "rotation_angle_deg": angle,
        "scale_ratio": norm_b / norm_a if norm_a > 0 else None,
        "dim": int(dim),
    }


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Rotation/scale decomposition of update-induced change")
    ap.add_argument("--pre-repo", required=True)
    ap.add_argument("--pre-rev", required=True)
    ap.add_argument("--post-repo", required=True)
    ap.add_argument("--post-rev", required=True)
    ap.add_argument("--update-kind", required=True)
    ap.add_argument("--axis", default="safety")
    ap.add_argument("--part", default="eval")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    A, pa = load(args.pre_repo, args.pre_rev, args.axis, args.part)
    B, pb = load(args.post_repo, args.post_rev, args.axis, args.part)
    if not np.array_equal(A["item_ids"], B["item_ids"]):
        raise RuntimeError("item_ids differ between caches")

    bad = A["nonfinite"].astype(bool) | B["nonfinite"].astype(bool)
    keep = ~bad
    if bad.any():
        print(f"dropping {int(bad.sum())} non-finite item(s)")

    keys = sorted([k for k in A.files if k.startswith("layer_") and k in B.files],
                  key=lambda s: int(s.split("_")[1]))
    n_max = max(int(k.split("_")[1]) for k in keys)

    rows: list[dict[str, Any]] = []
    print(f"  {'layer':>5} {'depth':>6} {'CKA':>8} {'procR2':>8} {'identR2':>8} "
          f"{'gap':>8} {'angle_deg':>10} {'scale':>7}")
    print("  " + "-" * 68)
    for k in keys:
        li = int(k.split("_")[1])
        x = A[k][keep].astype(np.float64)
        z = B[k][keep].astype(np.float64)
        if x.shape[1] != z.shape[1]:
            print(f"  layer {li}: width mismatch, skipped")
            continue
        pr = procrustes(x, z)
        row = {"layer": li, "relative_depth": li / n_max if n_max else None,
               "cka": linear_cka(x, z), **pr}
        row["basis_movement_gap"] = (
            row["procrustes_r2"] - row["identity_r2"]
            if row["procrustes_r2"] is not None and row["identity_r2"] is not None else None
        )
        rows.append(row)
        print(f"  {li:>5} {row['relative_depth']:>6.2f} {row['cka']:>8.5f} "
              f"{row['procrustes_r2']:>8.4f} {row['identity_r2']:>8.4f} "
              f"{row['basis_movement_gap']:>8.4f} {row['rotation_angle_deg']:>10.2f} "
              f"{row['scale_ratio']:>7.4f}")

    payload = {
        **provenance_block("rotation_test"),
        "config": vars(args),
        "config_hash": canonical_hash(vars(args)),
        "update_kind": args.update_kind,
        "input_cache_sha256": {"pre": sha256_file(pa), "post": sha256_file(pb)},
        "n_items": int(keep.sum()),
        "rows": rows,
    }
    digest = write_results(args.out, payload)
    print(f"\nWritten: {args.out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
