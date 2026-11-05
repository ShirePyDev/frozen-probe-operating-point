"""Does the label-free repair extend to a NON-PARAMETRIC guardrail layer?

Background
----------
D-046/D-047 established, for a parametric probe, that post-update calibration
collapse is caused by stale standardisation statistics and is repaired without
labels by re-estimating mu/sd on the new model.

D-039 recorded a different failure for the similarity layer: its frozen
threshold went from flagging 2.60% of harmful items to 0.05% after a backbone
change, a 52x collapse, while its AUROC moved only 0.611 -> 0.597.

I previously described that layer as having "no mu/sd to re-estimate". That was
imprecise and is corrected here. The layer has no FEATURE statistics, but it
does have two adaptable things:

  1. the exemplar embeddings - and in any real retrieval guardrail the exemplar
     TEXTS are what is stored, so they are re-embedded by the new model as a
     matter of course. That is already the condition measured below; it is not
     a repair, it is the default.
  2. the SCORE distribution - which is where the repair has to act.

So for a non-parametric layer the repair moves from feature level to score
level. The arms test exactly that.

Arms (all keep the exemplar SET frozen; none uses labels on the new model)
-------------------------------------------------------------------------
  frozen_thr    re-embed exemplars with the new model, keep the old threshold.
                Current practice, and the D-039 failure condition.
  quantile_thr  move the threshold to the same QUANTILE of the new score
                distribution that it occupied pre-update.
  zscore_thr    standardise scores by the new model's own score mean/sd and
                apply the threshold in standardised space. This is the direct
                score-level analogue of the feature-level restandardise repair.
  oracle_recal  recalibrate on LABELLED post-update data (upper bound only).

Reported against the target FPR the threshold was originally calibrated to.
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


def sim_scores(x: np.ndarray, ex: np.ndarray) -> np.ndarray:
    """Max cosine similarity to any exemplar."""
    xn = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    en = ex / np.clip(np.linalg.norm(ex, axis=1, keepdims=True), 1e-12, None)
    return (xn @ en.T).max(axis=1)


# The frozen-threshold rule is defined once, in src/paired.py (2026-09-25). This file's
# local copy was verified to give identical thresholds on 12,000 random and tie-heavy
# cases before it was replaced.
from paired import thr_at_fpr  # noqa: E402


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Label-free repair for a non-parametric layer")
    ap.add_argument("--pre-repo", required=True)
    ap.add_argument("--pre-rev", required=True)
    ap.add_argument("--post-repo", required=True)
    ap.add_argument("--post-rev", required=True)
    ap.add_argument("--update-kind", required=True)
    ap.add_argument("--axis", default="safety")
    ap.add_argument("--layers", nargs="*", type=int, default=None,
                    help="layer indices to evaluate; default = all common layers")
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--n-exemplars", type=int, default=256)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    hashes, data = {}, {}
    for side, repo, rev in (("pre", args.pre_repo, args.pre_rev),
                            ("post", args.post_repo, args.post_rev)):
        data[side] = {}
        for part in ("probe_train", "threshold_calib", "eval"):
            z, p = load(repo, rev, args.axis, part)
            data[side][part] = {k: z[k] for k in z.files}
            hashes[f"{side}/{part}"] = sha256_file(p)

    for part in ("probe_train", "threshold_calib", "eval"):
        if not np.array_equal(data["pre"][part]["item_ids"], data["post"][part]["item_ids"]):
            raise RuntimeError(f"item_ids differ for {part}")
        bad = (data["pre"][part]["nonfinite"].astype(bool)
               | data["post"][part]["nonfinite"].astype(bool))
        if bad.any():
            print(f"dropping {int(bad.sum())} non-finite item(s) from {part}")
            for side in ("pre", "post"):
                for k, arr in list(data[side][part].items()):
                    if hasattr(arr, "shape") and arr.shape and arr.shape[0] == bad.shape[0]:
                        data[side][part][k] = arr[~bad]

    common = sorted([k for k in data["pre"]["probe_train"]
                     if k.startswith("layer_") and k in data["post"]["probe_train"]],
                    key=lambda s: int(s.split("_")[1]))
    if args.layers:
        want = {f"layer_{i}" for i in args.layers}
        common = [k for k in common if k in want]
    n_max = max(int(k.split("_")[1]) for k in common)

    y_tr = data["pre"]["probe_train"]["labels"].astype(bool)
    y_cal = data["pre"]["threshold_calib"]["labels"].astype(bool)
    y_ev = data["pre"]["eval"]["labels"].astype(bool)
    pos_idx = np.flatnonzero(y_tr)

    arms = ("frozen_thr", "quantile_thr", "zscore_thr", "oracle_recal")
    print(f"  {'layer':>5} {'depth':>6} | " + " ".join(f"{a:>12}" for a in arms)
          + " | flag_pos pre->post (frozen)")
    print("  " + "-" * 96)
    rows: list[dict[str, Any]] = []
    for k in common:
        li = int(k.split("_")[1])
        acc: dict[str, list[float]] = {a: [] for a in arms}
        flag_pre, flag_post, au_pre, au_post = [], [], [], []
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            ex_idx = np.sort(rng.choice(pos_idx, size=min(args.n_exemplars, pos_idx.size),
                                        replace=False))
            # Exemplar TEXTS are frozen; each side embeds them with its own model,
            # which is what a deployed retrieval guardrail does automatically.
            ex_pre = data["pre"]["probe_train"][k][ex_idx]
            ex_post = data["post"]["probe_train"][k][ex_idx]

            cal_pre = sim_scores(data["pre"]["threshold_calib"][k], ex_pre)
            cal_post = sim_scores(data["post"]["threshold_calib"][k], ex_post)
            ev_pre = sim_scores(data["pre"]["eval"][k], ex_pre)
            ev_post = sim_scores(data["post"]["eval"][k], ex_post)

            t = thr_at_fpr(cal_pre[~y_cal], args.target_fpr)
            acc["frozen_thr"].append(float((ev_post[~y_ev] >= t).mean()))

            q = float((cal_pre < t).mean())
            t_q = float(np.quantile(cal_post, q))
            acc["quantile_thr"].append(float((ev_post[~y_ev] >= t_q).mean()))

            # Score-level analogue of restandardise: z-score using the NEW
            # model's own unlabelled score distribution.
            mu_a, sd_a = cal_pre.mean(), cal_pre.std() or 1.0
            mu_b, sd_b = cal_post.mean(), cal_post.std() or 1.0
            tz = (t - mu_a) / sd_a
            acc["zscore_thr"].append(float((((ev_post[~y_ev] - mu_b) / sd_b) >= tz).mean()))

            t_o = thr_at_fpr(cal_post[~y_cal], args.target_fpr)
            acc["oracle_recal"].append(float((ev_post[~y_ev] >= t_o).mean()))

            flag_pre.append(float((ev_pre[y_ev] >= t).mean()))
            flag_post.append(float((ev_post[y_ev] >= t).mean()))
            au_pre.append(M.auroc(ev_pre[y_ev], ev_pre[~y_ev]).value)
            au_post.append(M.auroc(ev_post[y_ev], ev_post[~y_ev]).value)

        row = {"layer": li, "relative_depth": li / n_max if n_max else None,
               "flag_rate_harmful_pre": float(np.mean(flag_pre)),
               "flag_rate_harmful_post_frozen": float(np.mean(flag_post)),
               "auroc_pre": float(np.mean(au_pre)), "auroc_post": float(np.mean(au_post))}
        for a in arms:
            row[f"{a}_fpr_mean"] = float(np.mean(acc[a]))
            row[f"{a}_fpr_sd"] = float(np.std(acc[a], ddof=1)) if len(acc[a]) > 1 else 0.0
        rows.append(row)
        print(f"  {li:>5} {row['relative_depth']:>6.2f} | "
              + " ".join(f"{row[a + '_fpr_mean']:>12.4f}" for a in arms)
              + f" | {row['flag_rate_harmful_pre']:.4f} -> {row['flag_rate_harmful_post_frozen']:.4f}")

    tgt = args.target_fpr
    summary = {a: float(np.mean([abs(np.log((r[f"{a}_fpr_mean"] or 1e-9) / tgt)) for r in rows]))
               for a in arms}
    payload = {
        **provenance_block("repair_nonparametric"),
        "config": vars(args), "config_hash": canonical_hash(vars(args)),
        "update_kind": args.update_kind, "input_cache_sha256": hashes,
        "mean_abs_log_fpr_error": summary, "rows": rows,
    }
    digest = write_results(args.out, payload)
    print("\n  mean |log(realised FPR / target)| across layers, lower is better:")
    for a, v in sorted(summary.items(), key=lambda kv: kv[1]):
        print(f"    {a:<14} {v:.4f}")
    print(f"\nWritten: {args.out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
