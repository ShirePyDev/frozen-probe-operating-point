"""Is the damage a BIGGER representation shift, or a BETTER-AIMED one? (D-104)

The frozen probe scores x as w . (x - mu_pre)/sd_pre + b, so the mean score shift an
update causes is exactly

    w . d      with   d = (mu_post - mu_pre) / sd_pre

and that product factorises into ||w|| * ||d|| * cos(theta). At a given layer and seed
||w|| is a property of the probe, identical across update arms, so any difference in
damage between two updates must come from how FAR the representation moved (||d||) or
how well the move aligns with what the probe reads (cos(theta)).

This distinguishes "a safety fine-tune is simply a larger update" from "a safety
fine-tune moves the representation along the very axis the safety probe reads".

Reference scale: for a random direction in d = 4096 dimensions, |cos| is about
1/sqrt(d) = 0.0156, so |cos| below ~0.05 is not distinguishable from an unaligned shift.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import (  # noqa: E402
    ensure_utf8_stdout,
    provenance_block,
    resolve_cache,
    sha256_file,
    write_results,
)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "hidden"


def load(repo: str, rev: str, axis: str, part: str):
    p = resolve_cache(f"{repo.replace('/', '__')}@{rev[:12]}__{axis}__{part}__mean.npz", CACHE)
    if not p.exists():
        raise FileNotFoundError(f"missing cache {p.name}")
    return np.load(p, allow_pickle=False), p


def fit_probe(x, y, seed):
    """Identical protocol to repair_test / moment_decomposition."""
    from sklearn.linear_model import LogisticRegression

    mu, sd = x.mean(0), x.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit((x - mu) / sd, y)
    return clf.coef_.ravel(), mu, sd


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Shift magnitude vs probe alignment (D-104)")
    ap.add_argument("--pre-repo", required=True)
    ap.add_argument("--pre-rev", required=True)
    ap.add_argument("--post-repo", required=True)
    ap.add_argument("--post-rev", required=True)
    ap.add_argument("--update-kind", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--axis", default="safety")
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    hashes, data = {}, {}
    for side, repo, rev in (("pre", args.pre_repo, args.pre_rev), ("post", args.post_repo, args.post_rev)):
        data[side] = {}
        for part in ("probe_train", "eval"):
            z, p = load(repo, rev, args.axis, part)
            data[side][part] = {k: z[k] for k in z.files}
            hashes[f"{side}/{part}"] = sha256_file(p)
    for part in ("probe_train", "eval"):
        bad = (data["pre"][part]["nonfinite"].astype(bool) | data["post"][part]["nonfinite"].astype(bool))
        if bad.any():
            for side in ("pre", "post"):
                for k, arr in list(data[side][part].items()):
                    if hasattr(arr, "shape") and arr.shape and arr.shape[0] == bad.shape[0]:
                        data[side][part][k] = arr[~bad]

    P, Q = data["pre"], data["post"]
    common = sorted([k for k in P["probe_train"] if k.startswith("layer_") and k in Q["probe_train"]],
                    key=lambda s: int(s.split("_")[1]))
    if args.layers:
        want = {f"layer_{i}" for i in args.layers}
        common = [k for k in common if k in want]
    y_tr = P["probe_train"]["labels"].astype(bool)
    n_tr = len(y_tr)

    rows: list[dict[str, Any]] = []
    for k in common:
        li = int(k.split("_")[1])
        if P["probe_train"][k].shape[1] != Q["probe_train"][k].shape[1]:
            continue
        xp, xq = P["eval"][k].astype(np.float64), Q["eval"][k].astype(np.float64)
        per_seed = []
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n_tr, size=int(args.subsample * n_tr), replace=False)
            w, mu, sd = fit_probe(P["probe_train"][k][idx], y_tr[idx], seed)
            d = (xq.mean(0) - xp.mean(0)) / sd          # standardised representation shift
            nd, nw = float(np.linalg.norm(d)), float(np.linalg.norm(w))
            cos = float(d @ w / (nd * nw)) if nd > 0 and nw > 0 else 0.0
            per_seed.append({"shift_norm": nd, "w_norm": nw, "cos": cos,
                             "score_shift": float(d @ w),
                             "aligned_component": float(nd * abs(cos))})
        agg = {q: float(np.mean([s[q] for s in per_seed])) for q in per_seed[0]}
        agg["cos_abs"] = float(np.mean([abs(s["cos"]) for s in per_seed]))
        agg["layer"] = li
        rows.append(agg)
        print(f"  layer {li:>3}  ||d|| {agg['shift_norm']:>8.3f}  cos {agg['cos']:>+7.4f}  "
              f"|cos| {agg['cos_abs']:.4f}  score shift {agg['score_shift']:>+9.3f}", flush=True)

    summary = {q: float(np.median([r[q] for r in rows]))
               for q in ("shift_norm", "cos", "cos_abs", "score_shift", "aligned_component", "w_norm")}
    summary["n_layers"] = len(rows)
    summary["random_cos_reference"] = float(1.0 / np.sqrt(P["eval"][common[0]].shape[1]))
    print(f"\n  MEDIAN over {len(rows)} layers: ||d|| {summary['shift_norm']:.3f}  "
          f"cos {summary['cos']:+.4f}  |cos| {summary['cos_abs']:.4f}  "
          f"|score shift| {abs(summary['score_shift']):.3f}  "
          f"(random |cos| ~ {summary['random_cos_reference']:.4f})")

    payload = {**provenance_block("shift_alignment"), "preregistration": "D-104",
               "label": args.label, "update_kind": args.update_kind, "config": vars(args),
               "input_cache_sha256": hashes, "summary": summary, "rows": rows}
    digest = write_results(args.out, payload)
    print(f"\nWritten: {args.out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
