"""Mechanism and repair for post-update calibration collapse.

Where this came from
--------------------
D-041/D-042 measured the phenomenon: after instruction tuning, a probe whose
weights AND threshold are frozen realises anywhere from 0.72% to 59.2% FPR
against a 1% target, depending only on which layer it sits at, while its AUROC
barely moves.

D-043 found CKA does not predict that damage. My first hypothesis was that the
update rotates the representation - CKA is rotation-invariant, a frozen probe is
not. results/rotation_mistral_base2instruct.json REFUTES it: layer 2 has the
worst damage and the SMALLEST basis-movement gap (0.0249), layer 32 the largest
gap (0.1947) and the best calibration. Orthogonal Procrustes centres both
matrices, so it is blind to a shift of the mean - which is the one thing it
should have been looking at.

The mechanism this script tests
-------------------------------
A probe scores s(x) = w . ((x - mu_fit) / sd_fit) + b, where mu_fit and sd_fit
are standardisation statistics estimated when the probe was FIT. After an update
the new representation has its own mean and scale. Holding mu_fit/sd_fit fixed,
every score picks up an approximately constant offset

    delta ~= w . (mu_post - mu_pre) / sd_pre

which slides the whole score distribution past a frozen threshold. That is a
CALIBRATION failure produced by stale normalisation statistics, and it is
invisible to AUROC because adding a constant to every score cannot change any
ranking.

If that is right, the repair needs NO LABELS: re-estimate mu and sd on the new
model from unlabelled data, keep the probe weights and the threshold frozen.

Arms compared (all keep the probe WEIGHTS frozen - no labels on the new model)
-----------------------------------------------------------------------------
  frozen_all      stale mu/sd, stale threshold            (current practice)
  restandardise   mu/sd re-estimated on post, UNLABELLED; threshold unchanged
  quantile_thr    stale mu/sd; threshold moved to the same QUANTILE of the post
                  score distribution that it occupied pre-update (unlabelled)
  refit_probe     probe refit with labels on post (upper bound, needs labels)
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
    return clf, mu, sd


# The frozen-threshold rule is defined once, in src/paired.py (2026-09-25). This file's
# local copy was verified to give identical thresholds on 12,000 random and tie-heavy
# cases before it was replaced.
from paired import thr_at_fpr  # noqa: E402


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Label-free repair of post-update calibration")
    ap.add_argument("--pre-repo", required=True)
    ap.add_argument("--pre-rev", required=True)
    ap.add_argument("--post-repo", required=True)
    ap.add_argument("--post-rev", required=True)
    ap.add_argument("--update-kind", required=True)
    ap.add_argument("--axis", default="safety")
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
        bad = (data["pre"][part]["nonfinite"].astype(bool)
               | data["post"][part]["nonfinite"].astype(bool))
        if bad.any():
            for side in ("pre", "post"):
                for k, arr in list(data[side][part].items()):
                    if hasattr(arr, "shape") and arr.shape and arr.shape[0] == bad.shape[0]:
                        data[side][part][k] = arr[~bad]

    keys = sorted([k for k in data["pre"]["probe_train"]
                   if k.startswith("layer_") and k in data["post"]["probe_train"]],
                  key=lambda s: int(s.split("_")[1]))
    n_max = max(int(k.split("_")[1]) for k in keys)

    y_tr = data["pre"]["probe_train"]["labels"].astype(bool)
    y_cal = data["pre"]["threshold_calib"]["labels"].astype(bool)
    y_ev = data["pre"]["eval"]["labels"].astype(bool)
    n_tr = len(y_tr)

    print(f"  {'layer':>5} {'depth':>6} | {'frozen_all':>10} {'restand':>9} {'quantile':>9} "
          f"{'refit':>8} | {'predicted':>10} {'observed':>9}")
    print("  " + "-" * 82)
    rows: list[dict[str, Any]] = []
    for k in keys:
        li = int(k.split("_")[1])
        acc: dict[str, list[float]] = {a: [] for a in
                                       ("frozen_all", "restandardise", "quantile_thr", "refit_probe")}
        shift_pred, shift_obs, aurocs = [], [], []
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n_tr, size=int(args.subsample * n_tr), replace=False)
            xa_tr = data["pre"]["probe_train"][k][idx]
            xb_tr = data["post"]["probe_train"][k][idx]
            clf, mu_a, sd_a = fit_probe(xa_tr, y_tr[idx], seed)
            w, b = clf.coef_.ravel(), float(clf.intercept_[0])

            # Post-model standardisation statistics, estimated WITHOUT labels.
            mu_b, sd_b = xb_tr.mean(0), xb_tr.std(0)
            sd_b = np.where(sd_b < 1e-8, 1.0, sd_b)

            def s_stale(x):
                return (x - mu_a) / sd_a @ w + b

            def s_restd(x):
                return (x - mu_b) / sd_b @ w + b

            cal_pre = s_stale(data["pre"]["threshold_calib"][k])
            t = thr_at_fpr(cal_pre[~y_cal], args.target_fpr)

            ev_post_stale = s_stale(data["post"]["eval"][k])
            ev_post_restd = s_restd(data["post"]["eval"][k])
            ev_pre_stale = s_stale(data["pre"]["eval"][k])

            acc["frozen_all"].append(float((ev_post_stale[~y_ev] >= t).mean()))
            acc["restandardise"].append(float((ev_post_restd[~y_ev] >= t).mean()))

            # Quantile transfer: keep the threshold at the same rank position in
            # the post-update score distribution that it held pre-update. Uses
            # the unlabelled calibration pool only.
            cal_post_stale = s_stale(data["post"]["threshold_calib"][k])
            q = float((cal_pre < t).mean())
            t_q = float(np.quantile(cal_post_stale, q))
            acc["quantile_thr"].append(float((ev_post_stale[~y_ev] >= t_q).mean()))

            clf2, mu2, sd2 = fit_probe(xb_tr, y_tr[idx], seed)
            cal2 = (data["post"]["threshold_calib"][k] - mu2) / sd2 @ clf2.coef_.ravel() + float(clf2.intercept_[0])
            t2 = thr_at_fpr(cal2[~y_cal], args.target_fpr)
            ev2 = (data["post"]["eval"][k] - mu2) / sd2 @ clf2.coef_.ravel() + float(clf2.intercept_[0])
            acc["refit_probe"].append(float((ev2[~y_ev] >= t2).mean()))

            # Is the score shift explained by the mean shift alone?
            shift_pred.append(float(w @ ((xb_tr.mean(0) - mu_a) / sd_a)))
            shift_obs.append(float(ev_post_stale.mean() - ev_pre_stale.mean()))
            aurocs.append(M.auroc(ev_post_stale[y_ev], ev_post_stale[~y_ev]).value)

        row = {"layer": li, "relative_depth": li / n_max if n_max else None,
               "auroc_post_frozen_mean": float(np.mean(aurocs)),
               "predicted_shift_from_mean": float(np.mean(shift_pred)),
               "observed_shift": float(np.mean(shift_obs))}
        for a, v in acc.items():
            row[f"{a}_fpr_mean"] = float(np.mean(v))
            row[f"{a}_fpr_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        rows.append(row)
        print(f"  {li:>5} {row['relative_depth']:>6.2f} | {row['frozen_all_fpr_mean']:>10.4f} "
              f"{row['restandardise_fpr_mean']:>9.4f} {row['quantile_thr_fpr_mean']:>9.4f} "
              f"{row['refit_probe_fpr_mean']:>8.4f} | {row['predicted_shift_from_mean']:>10.3f} "
              f"{row['observed_shift']:>9.3f}")

    tgt = args.target_fpr
    def err(key):
        return float(np.mean([abs(np.log((r[key] or 1e-9) / tgt)) for r in rows]))

    # F-034: an FPR of exactly 0 is CENSORED at the eval resolution, not 1e-9. The
    # original metric is kept for comparability; these two fields make the censoring
    # visible and give the resolution-floored value beside it.
    n_neg_ev = int((~y_ev).sum())
    def err_resfloor(key):
        return float(np.mean([abs(np.log(max(r[key], 1.0 / n_neg_ev) / tgt)) for r in rows]))

    def n_censored(key):
        return int(sum(1 for r in rows if r[key] == 0))

    arms_ = ("frozen_all", "restandardise", "quantile_thr", "refit_probe")
    summary = {a: err(f"{a}_fpr_mean") for a in arms_}
    summary_resfloor = {a: err_resfloor(f"{a}_fpr_mean") for a in arms_}
    censored = {a: n_censored(f"{a}_fpr_mean") for a in arms_}
    corr = float(np.corrcoef([r["predicted_shift_from_mean"] for r in rows],
                             [r["observed_shift"] for r in rows])[0, 1])

    payload = {
        **provenance_block("repair_test"),
        "config": vars(args), "config_hash": canonical_hash(vars(args)),
        "update_kind": args.update_kind, "input_cache_sha256": hashes,
        "mean_abs_log_fpr_error": summary,
        "mean_abs_log_fpr_error_resolution_floored": summary_resfloor,
        "n_censored_zero_fpr_cells": censored,
        "n_neg_eval": n_neg_ev,
        "pearson_predicted_vs_observed_shift": corr,
        "rows": rows,
    }
    digest = write_results(args.out, payload)
    print("\n  mean |log(realised FPR / target)| across layers, lower is better:")
    for a, v in sorted(summary.items(), key=lambda kv: kv[1]):
        print(f"    {a:<16} {v:.4f}")
    print(f"\n  pearson(predicted shift from mean-drift, observed shift) = {corr:+.4f}")
    print(f"\nWritten: {args.out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
