"""Which part of the post-update representation change actually breaks a frozen probe?

Why this script exists - a self-correction (D-076)
--------------------------------------------------
repair_test.py reports pearson(predicted score shift from mean drift, observed shift)
= 0.9996-0.9999, and earlier decisions cited it as proof of a mean-shift mechanism.
For a LINEAR probe that is near an identity: mean score shift = w . dmean / sd
exactly. This script checks that identity numerically and replaces it with
interventions, which are not tautological.

Arms - probe WEIGHTS and decision THRESHOLD stay frozen unless stated
---------------------------------------------------------------------
  frozen          stale mu_pre, sd_pre                            (current practice)
  mean_only       mu re-estimated on the NEW model, sd kept       (mean alone)
  restandardise   mu and sd re-estimated on the NEW model          (D-047 repair)
  score_affine    1-D: match post score mean/std to pre, unlabelled calib pools
  paired_align    ridge affine map post -> pre activations from PAIRED unlabelled
                  items, frozen probe as-is. OUR REIMPLEMENTATION of the repair
                  class arXiv:2606.15980 describes; their exact form is unspecified
  align_restd     paired_align, then restandardise using the ALIGNED activations'
                  own mean/sd (tests whether variance shrinkage is what hurts)
  align_affine    paired_align, then the 1-D score_affine correction
  self_align_PRE  CONTROL, no update at all: the same ridge map fitted pre -> pre
                  and applied to PRE eval. If AUROC rises and FPR collapses here
                  too, those effects come from ridge shrinkage, not from the update
  refit_oracle    probe refit AND threshold recalibrated with labels (upper bound)

Diagnostics: feature-std ratio and score-std ratio of aligned vs true pre
activations. Least-squares predictions have variance R^2 * var(target), so a ratio
below 1 is the signature of regression-to-the-mean shrinkage.
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
    resolve_cache,
    canonical_hash,
    ensure_utf8_stdout,
    provenance_block,
    sha256_file,
    write_results,
)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "hidden"
ARMS = ("frozen", "mean_only", "restandardise", "score_affine", "paired_align",
        "align_restd", "align_affine", "self_align_PRE", "refit_oracle")


def load(repo: str, rev: str, axis: str, part: str):
    p = resolve_cache(f"{repo.replace('/', '__')}@{rev[:12]}__{axis}__{part}__mean.npz", CACHE)
    if not p.exists():
        raise FileNotFoundError(f"missing cache {p.name}")
    return np.load(p, allow_pickle=False), p


def fit_probe(x, y, seed):
    from sklearn.linear_model import LogisticRegression

    mu, sd = x.mean(0), x.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit((x - mu) / sd, y)
    return clf.coef_.ravel(), float(clf.intercept_[0]), mu, sd


# The frozen-threshold rule is defined once, in src/paired.py (2026-09-25). This file's
# local copy was verified to give identical thresholds on 12,000 random and tie-heavy
# cases before it was replaced.
from paired import thr_at_fpr  # noqa: E402


def fit_alignment(x_src, x_tgt, xv_src, xv_tgt):
    """Ridge affine map src -> tgt from paired unlabelled items.

    n < d, so an unregularised map is underdetermined. The ridge strength is picked
    by reconstruction MSE on a SEPARATE paired unlabelled pool; no labels used.
    """
    ms, mt = x_src.mean(0), x_tgt.mean(0)
    xs, xt = x_src - ms, x_tgt - mt
    gram, cross = xs.T @ xs, xs.T @ xt
    scale = float(np.trace(gram)) / gram.shape[0]
    best, tried = None, []
    for lam_rel in (1e-3, 1e-2, 1e-1, 1.0):
        a = np.linalg.solve(gram + lam_rel * scale * np.eye(gram.shape[0]), cross)
        mse = float((((xv_src - ms) @ a + mt - xv_tgt) ** 2).mean())
        tried.append({"lambda_rel": lam_rel, "val_mse": mse})
        if best is None or mse < best[0]:
            best = (mse, lam_rel, a)
    mse, lam_rel, a = best
    return (lambda x: (x - ms) @ a + mt), {
        "lambda_rel_chosen": lam_rel, "grid": tried,
        "val_mse_identity": float(((xv_src - xv_tgt) ** 2).mean()), "val_mse_aligned": mse}


def affine_to(s, s_cal, ref_cal):
    return (s - s_cal.mean()) / (s_cal.std() or 1.0) * (ref_cal.std() or 1.0) + ref_cal.mean()


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Intervention-based decomposition of probe staleness")
    ap.add_argument("--pre-repo", required=True)
    ap.add_argument("--pre-rev", required=True)
    ap.add_argument("--post-repo", required=True)
    ap.add_argument("--post-rev", required=True)
    ap.add_argument("--update-kind", required=True)
    ap.add_argument("--axis", default="safety")
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--subsample", type=float, default=0.8)
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
        bad = data["pre"][part]["nonfinite"].astype(bool) | data["post"][part]["nonfinite"].astype(bool)
        if bad.any():
            print(f"dropping {int(bad.sum())} non-finite from {part}")
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
    P, Q = data["pre"], data["post"]
    y_tr = P["probe_train"]["labels"].astype(bool)
    y_cal = P["threshold_calib"]["labels"].astype(bool)
    y_ev = P["eval"]["labels"].astype(bool)
    n_tr, tgt = len(y_tr), args.target_fpr

    rows: list[dict[str, Any]] = []
    identity_resid, pred_train, observed = [], [], []
    for k in common:
        li = int(k.split("_")[1])
        if P["probe_train"][k].shape[1] != Q["probe_train"][k].shape[1]:
            print(f"  layer {li}: width mismatch, skipped")
            continue
        f64 = lambda a: a.astype(np.float64)  # noqa: E731
        align, align_info = fit_alignment(f64(Q["probe_train"][k]), f64(P["probe_train"][k]),
                                          f64(Q["threshold_calib"][k]), f64(P["threshold_calib"][k]))
        selfal, self_info = fit_alignment(f64(P["probe_train"][k]), f64(P["probe_train"][k]),
                                          f64(P["threshold_calib"][k]), f64(P["threshold_calib"][k]))
        AQ = {part: align(f64(Q[part][k])) for part in ("probe_train", "threshold_calib", "eval")}
        SP_eval, SP_cal = selfal(f64(P["eval"][k])), selfal(f64(P["threshold_calib"][k]))
        feat_ratio = float(np.median(AQ["eval"].std(0) / np.maximum(P["eval"][k].std(0), 1e-8)))
        self_feat_ratio = float(np.median(SP_eval.std(0) / np.maximum(P["eval"][k].std(0), 1e-8)))

        acc = {a: {"fpr": [], "auroc": [], "tpr": []} for a in ARMS}
        auroc_pre, score_ratio = [], []
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n_tr, size=int(args.subsample * n_tr), replace=False)
            w, b, mu_a, sd_a = fit_probe(P["probe_train"][k][idx], y_tr[idx], seed)
            sc = lambda x, mu, sd: (x - mu) / sd @ w + b  # noqa: E731

            cal_pre = sc(P["threshold_calib"][k], mu_a, sd_a)
            t = thr_at_fpr(cal_pre[~y_cal], tgt)
            ev_pre = sc(P["eval"][k], mu_a, sd_a)
            auroc_pre.append(M.auroc(ev_pre[y_ev], ev_pre[~y_ev]).value)

            ev_frozen = sc(Q["eval"][k], mu_a, sd_a)
            obs = float(ev_frozen.mean() - ev_pre.mean())
            identity_resid.append(abs(obs - float(w @ ((Q["eval"][k].mean(0) - P["eval"][k].mean(0)) / sd_a))))
            xb_tr = Q["probe_train"][k][idx]
            mu_b, sd_b = xb_tr.mean(0), np.where(xb_tr.std(0) < 1e-8, 1.0, xb_tr.std(0))
            pred_train.append(float(w @ ((mu_b - mu_a) / sd_a)))
            observed.append(obs)

            cal_frozen = sc(Q["threshold_calib"][k], mu_a, sd_a)
            al_ev, al_cal = sc(AQ["eval"], mu_a, sd_a), sc(AQ["threshold_calib"], mu_a, sd_a)
            atr = AQ["probe_train"][idx]
            mu_al, sd_al = atr.mean(0), np.where(atr.std(0) < 1e-8, 1.0, atr.std(0))
            score_ratio.append(float(al_ev.std() / (ev_pre.std() or 1.0)))

            post = {
                "frozen": ev_frozen,
                "mean_only": sc(Q["eval"][k], mu_b, sd_a),
                "restandardise": sc(Q["eval"][k], mu_b, sd_b),
                "score_affine": affine_to(ev_frozen, cal_frozen, cal_pre),
                "paired_align": al_ev,
                "align_restd": sc(AQ["eval"], mu_al, sd_al),
                "align_affine": affine_to(al_ev, al_cal, cal_pre),
            }
            for arm, s in post.items():
                acc[arm]["fpr"].append(float((s[~y_ev] >= t).mean()))
                acc[arm]["tpr"].append(float((s[y_ev] >= t).mean()))
                acc[arm]["auroc"].append(M.auroc(s[y_ev], s[~y_ev]).value)

            # CONTROL on the PRE model: does ridge alignment alone move AUROC / FPR?
            s_self = sc(SP_eval, mu_a, sd_a)
            acc["self_align_PRE"]["fpr"].append(float((s_self[~y_ev] >= t).mean()))
            acc["self_align_PRE"]["tpr"].append(float((s_self[y_ev] >= t).mean()))
            acc["self_align_PRE"]["auroc"].append(M.auroc(s_self[y_ev], s_self[~y_ev]).value)

            w2, b2, mu2, sd2 = fit_probe(xb_tr, y_tr[idx], seed)
            s2c = (Q["threshold_calib"][k] - mu2) / sd2 @ w2 + b2
            t2 = thr_at_fpr(s2c[~y_cal], tgt)
            s2 = (Q["eval"][k] - mu2) / sd2 @ w2 + b2
            acc["refit_oracle"]["fpr"].append(float((s2[~y_ev] >= t2).mean()))
            acc["refit_oracle"]["tpr"].append(float((s2[y_ev] >= t2).mean()))
            acc["refit_oracle"]["auroc"].append(M.auroc(s2[y_ev], s2[~y_ev]).value)

        row: dict[str, Any] = {
            "layer": li, "auroc_pre": float(np.mean(auroc_pre)),
            "alignment_post_to_pre": align_info, "alignment_self_pre": self_info,
            "aligned_feature_std_ratio_median": feat_ratio,
            "self_aligned_feature_std_ratio_median": self_feat_ratio,
            "aligned_score_std_ratio_mean": float(np.mean(score_ratio)), "arms": {}}
        for arm in ARMS:
            f = np.asarray(acc[arm]["fpr"])
            row["arms"][arm] = {
                "fpr_mean": float(f.mean()), "fpr_sd": float(f.std(ddof=1)) if f.size > 1 else 0.0,
                "log_fpr_err": float(abs(np.log(max(f.mean(), 1e-9) / tgt))),
                "auroc_mean": float(np.mean(acc[arm]["auroc"])),
                "tpr_mean": float(np.mean(acc[arm]["tpr"]))}
        rows.append(row)
        print(f"  layer {li:>3}  preAUROC {row['auroc_pre']:.4f}  aligned feat-std ratio "
              f"{feat_ratio:.3f} (self {self_feat_ratio:.3f})  score-std ratio "
              f"{row['aligned_score_std_ratio_mean']:.3f}")
        for arm in ARMS:
            r = row["arms"][arm]
            print(f"      {arm:<15} FPR {r['fpr_mean']:.4f}  AUROC {r['auroc_mean']:.4f}  TPR {r['tpr_mean']:.4f}")

    # F-034: report censored cells (realised FPR of exactly 0, floored at 1e-9 by the
    # original metric) and a resolution-floored variant beside the original, which is
    # kept unchanged so earlier results stay comparable.
    n_neg_ev = int((~y_ev).sum())
    def _resfloor(r, a):
        f = r["arms"][a]["fpr_mean"]
        return abs(np.log(max(f, 1.0 / n_neg_ev) / tgt))

    summary = {a: {"mean_log_fpr_err": float(np.mean([r["arms"][a]["log_fpr_err"] for r in rows])),
                   "mean_log_fpr_err_resolution_floored": float(np.mean([_resfloor(r, a) for r in rows])),
                   "n_censored_zero_fpr_cells": int(sum(1 for r in rows if r["arms"][a]["fpr_mean"] == 0)),
                   "mean_auroc_drop_vs_pre": float(np.mean([r["auroc_pre"] - r["arms"][a]["auroc_mean"]
                                                            for r in rows]))} for a in ARMS}
    ef, em, er = (summary[x]["mean_log_fpr_err"] for x in ("frozen", "mean_only", "restandardise"))
    share_mean = (ef - em) / (ef - er) if ef != er else None
    ident = {"max_abs_identity_residual": float(max(identity_resid)),
             "pearson_train_predicted_vs_observed": float(np.corrcoef(pred_train, observed)[0, 1]),
             "note": "residual ~0: the mean score shift IS w.dmean/sd; a high pearson is expected"}

    print("\n  per arm: mean |log(FPR/target)|, mean AUROC drop vs pre (negative = higher than pre)")
    for a in ARMS:
        print(f"    {a:<15} calib err {summary[a]['mean_log_fpr_err']:.4f}   "
              f"AUROC drop {summary[a]['mean_auroc_drop_vs_pre']:+.4f}")
    print(f"  share of restandardise repair from the MEAN alone: {share_mean:.1%}")
    print(f"  identity self-audit residual {ident['max_abs_identity_residual']:.2e}, "
          f"pearson {ident['pearson_train_predicted_vs_observed']:.4f}")

    payload = {**provenance_block("moment_decomposition"), "config": vars(args),
               "config_hash": canonical_hash(vars(args)), "update_kind": args.update_kind,
               "input_cache_sha256": hashes, "summary": summary,
               "share_of_repair_from_mean_alone": share_mean,
               "identity_self_audit": ident, "rows": rows}
    digest = write_results(args.out, payload)
    print(f"\nWritten: {args.out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
