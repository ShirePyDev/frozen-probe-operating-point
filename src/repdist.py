"""Representational distance between two checkpoints, measured on frozen items.

Why this exists
---------------
E6 returned a null for H1 on Mistral v0.1 -> v0.2, and the diagnosis was not
"beta is stable" but "the treatment was tiny": pooled hidden states between those
two checkpoints have mean cosine similarity 0.9976, and per-layer AUROC moved by
0.0004. The weights are genuinely different (0/5 sampled tensors bit-identical),
so it IS a real retrain - just a representationally small one.

That turns the binary pre/post design into the wrong instrument. A VERSION LABEL
does not tell you how much the representation moved, so it cannot tell an operator
whether to revalidate. What is needed is a continuous treatment variable:

    D  = representational distance between checkpoints  (measured here)
    dB = change in stack coupling                       (measured by pilot_e6)

and then the dose-response D -> dB. That is both a stronger design and a more
useful output: it yields a revalidation TRIGGER rather than a yes/no claim.

Measures
--------
linear CKA      rotation- and isotropic-scaling-invariant similarity of two
                representation matrices. This is the one that matters: a frozen
                probe can break from a pure basis rotation, which is a
                maintenance artefact, whereas CKA only moves when the
                representational GEOMETRY changes.
mean cosine     per-item cosine between the two checkpoints' pooled vectors.
                Not rotation-invariant, so a gap between this and CKA is itself
                diagnostic: high CKA with low cosine means "same geometry,
                rotated basis".
relative L2     per-item ||a-b|| / ||a||.
probe transfer  fit a probe on checkpoint A, evaluate on checkpoint B's states.
                A FUNCTIONAL distance: it asks whether the decision boundary
                still means anything, which is the quantity an operator with a
                frozen probe actually cares about.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402
from provenance import (  # noqa: E402
    ensure_utf8_stdout,
    provenance_block,
    sha256_file,
    write_results,
)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "hidden"


def cache_file(repo: str, revision: str, axis: str, part: str, pooling: str = "mean") -> Path:
    return CACHE / f"{repo.replace('/', '__')}@{revision[:12]}__{axis}__{part}__{pooling}.npz"


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between two (n_items, dim) representation matrices.

    Invariant to orthogonal transformation and isotropic scaling of either side,
    which is precisely what distinguishes a genuine representational change from
    a rotation of the same geometry. Computed via the Gram-matrix form with
    column-centred inputs; dims may differ between x and y.
    """
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    # ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    xty = x.T @ y
    num = float((xty**2).sum())
    den = float(np.sqrt(((x.T @ x) ** 2).sum()) * np.sqrt(((y.T @ y) ** 2).sum()))
    if den <= 0:
        return float("nan")
    return num / den


def probe_transfer(
    xa_tr: np.ndarray, ya_tr: np.ndarray, xa_ev: np.ndarray, xb_ev: np.ndarray,
    y_ev: np.ndarray, seed: int = 0,
) -> dict[str, Any]:
    """Fit a probe on A's training states; score A's and B's eval states with it.

    The AUROC drop from A to B is a functional measure of how much a FROZEN probe
    degrades - the no-maintenance operator condition (D-002 secondary arm).
    """
    from sklearn.linear_model import LogisticRegression

    mu, sd = xa_tr.mean(0), xa_tr.std(0)
    sd[sd < 1e-8] = 1.0
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit((xa_tr - mu) / sd, ya_tr)
    sa = clf.decision_function((xa_ev - mu) / sd)
    sb = clf.decision_function((xb_ev - mu) / sd)
    au_a = M.auroc(sa[y_ev], sa[~y_ev]).value
    au_b = M.auroc(sb[y_ev], sb[~y_ev]).value
    return {
        "auroc_on_source": au_a,
        "auroc_transferred": au_b,
        "auroc_drop": (au_a - au_b) if (au_a is not None and au_b is not None) else None,
        "score_pearson": float(np.corrcoef(sa, sb)[0, 1]),
    }


def compare(
    a_repo: str, a_rev: str, b_repo: str, b_rev: str, axis: str,
    layers: list[int] | None, b_layers: list[int] | None = None,
) -> dict[str, Any]:
    """Compare two checkpoints.

    `b_layers`, when given, is paired POSITIONALLY with `layers`. This is required
    whenever the two checkpoints have different depths - Qwen2.5-7B has 28 layers
    and Qwen3-8B-Base has 36, so "layer 16" is not the same relative depth on both
    sides. Comparing absolute indices across different depths would measure depth
    mismatch, not representational drift. CKA tolerates differing widths, so only
    the depth alignment has to be chosen deliberately.
    """
    ap_ = cache_file(a_repo, a_rev, axis, "eval")
    bp_ = cache_file(b_repo, b_rev, axis, "eval")
    atr = cache_file(a_repo, a_rev, axis, "probe_train")
    btr = cache_file(b_repo, b_rev, axis, "probe_train")
    for p in (ap_, bp_):
        if not p.exists():
            raise FileNotFoundError(f"missing cache: {p.name}")
    A, B = np.load(ap_), np.load(bp_)
    if not np.array_equal(A["item_ids"], B["item_ids"]):
        raise RuntimeError("item_ids differ; representational distance would be meaningless")
    y = A["labels"].astype(bool)

    keys = sorted([k for k in A.files if k.startswith("layer_")],
                  key=lambda s: int(s.split("_")[1]))
    if layers:
        want = {f"layer_{li}" for li in layers}
        keys = [k for k in keys if k in want]
    if b_layers:
        if not layers or len(b_layers) != len(layers):
            raise ValueError("--b-layers must be given alongside --layers and match in length")
        key_pairs = [(f"layer_{a}", f"layer_{b}") for a, b in zip(layers, b_layers)]
    else:
        key_pairs = [(k, k) for k in keys]

    Atr = np.load(atr) if atr.exists() else None
    Btr = np.load(btr) if btr.exists() else None

    out: dict[str, Any] = {
        "pair": f"{a_repo}@{a_rev[:12]} -> {b_repo}@{b_rev[:12]}",
        "axis": axis,
        "n_items": int(len(y)),
        "input_sha256": {"a_eval": sha256_file(ap_), "b_eval": sha256_file(bp_)},
        "layers": {},
    }
    for ka, kb in key_pairs:
        if ka not in A.files or kb not in B.files:
            out["layers"][f"{ka}->{kb}"] = {"error": "layer missing in one cache"}
            continue
        k = ka if ka == kb else f"{ka}->{kb}"
        x, z = A[ka].astype(np.float64), B[kb].astype(np.float64)
        same_width = x.shape[1] == z.shape[1]
        entry: dict[str, Any] = {
            "cka": linear_cka(x, z),
            "dim_a": int(x.shape[1]),
            "dim_b": int(z.shape[1]),
            "bit_identical": bool(same_width and np.array_equal(A[ka], B[kb])),
        }
        # Per-item cosine and relative L2 are only defined when the two
        # representations live in the same space. Across different widths only
        # CKA is meaningful, which is exactly why it is the primary distance.
        if same_width:
            xn = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
            zn = z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-12, None)
            cos = (xn * zn).sum(axis=1)
            rel = np.linalg.norm(x - z, axis=1) / np.clip(np.linalg.norm(x, axis=1), 1e-12, None)
            entry["cos_mean"] = float(cos.mean())
            entry["cos_p05"] = float(np.percentile(cos, 5))
            entry["rel_l2_mean"] = float(rel.mean())
        else:
            entry["cos_mean"] = entry["cos_p05"] = entry["rel_l2_mean"] = None
        # Probe transfer requires matching widths: a probe fit on A cannot be
        # applied to B's states if the hidden sizes differ.
        if (Atr is not None and Btr is not None and ka in Atr.files
                and x.shape[1] == z.shape[1]):
            entry["probe_transfer"] = probe_transfer(
                Atr[ka].astype(np.float64), Atr["labels"], x, z, y
            )
        elif x.shape[1] != z.shape[1]:
            entry["probe_transfer_skipped"] = f"width mismatch {x.shape[1]} vs {z.shape[1]}"
        out["layers"][k] = entry
    return out


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Representational distance between checkpoints")
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="each as arepo:arev:brepo:brev (revs may be 12-char prefixes)")
    ap.add_argument("--axis", default="safety")
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    ap.add_argument("--b-layers", nargs="*", type=int, default=None,
                    help="layer indices on the B side, paired positionally with --layers; "
                         "use when the two checkpoints have different depths")
    ap.add_argument("--out", default=str(ROOT / "results" / "repdist.json"))
    args = ap.parse_args()

    results: dict[str, Any] = {**provenance_block("repdist"), "axis": args.axis, "pairs": []}
    for spec in args.pairs:
        parts = spec.split(":")
        if len(parts) != 4:
            raise ValueError(f"bad pair spec {spec!r}; want arepo:arev:brepo:brev")
        res = compare(parts[0], parts[1], parts[2], parts[3], args.axis,
                      args.layers, args.b_layers)
        results["pairs"].append(res)
        print(f"\n=== {res['pair']}  ({res['n_items']} items) ===")
        print(f"  {'layer':<18} {'CKA':>8} {'cos_mean':>9} {'cos_p05':>9} {'relL2':>8} "
              f"{'probeAUROC_src':>15} {'transferred':>12} {'drop':>8}")
        for k, v in res["layers"].items():
            pt = v.get("probe_transfer") or {}
            src = f"{pt.get('auroc_on_source'):.4f}" if pt.get("auroc_on_source") is not None else "-"
            tr = f"{pt.get('auroc_transferred'):.4f}" if pt.get("auroc_transferred") is not None else "-"
            dr = f"{pt.get('auroc_drop'):+.4f}" if pt.get("auroc_drop") is not None else "-"
            f2 = lambda z: f"{z:>9.5f}" if z is not None else f"{'n/a':>9}"  # noqa: E731
            rl = f"{v['rel_l2_mean']:>8.5f}" if v['rel_l2_mean'] is not None else f"{'n/a':>8}"
            print(f"  {k:<18} {v['cka']:>8.5f} {f2(v['cos_mean'])} {f2(v['cos_p05'])} "
                  f"{rl} {src:>15} {tr:>12} {dr:>8}")

    digest = write_results(args.out, results)
    print(f"\nWritten: {args.out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
