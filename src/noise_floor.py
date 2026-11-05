"""No-update noise floor of the calibration-error metric (D-083).

Every repair and damage claim uses mean |log(realised FPR / target)|: per layer,
the realised eval FPR at a threshold set on the calibration split, averaged over
seeds, logged against the target, then averaged over layers. None of those claims
was judged against the value the metric takes when NOTHING was updated.

This script recomputes that value from the PRE model only, with the settings of
the result it audits (seeds, subsample, probe, threshold rule, non-finite mask and
layer set, all read from that result's config), in two forms:

  as_realised    pre probe, pre threshold, pre eval - the same split realisation
                 every arm of the audited result was measured on
  resplit null   calibration and eval NEGATIVES pooled and re-partitioned at the
                 original sizes, N times; one partition is shared by all seeds and
                 layers, as the frozen split is. Only negatives enter an FPR, so
                 positives do not need re-partitioning.

Precondition, for moment_* results: the pre-model probe refit here must reproduce
the stored per-layer auroc_pre, or the audit is not measuring the same probes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402
from provenance import (  # noqa: E402
    ensure_utf8_stdout,
    provenance_block,
    resolve_cache,
    sha256_file,
    write_results,
)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "hidden"
PARTS = ("probe_train", "threshold_calib", "eval")


def cache(repo: str, rev: str, axis: str, part: str) -> Path:
    p = resolve_cache(f"{repo.replace('/', '__')}@{rev[:12]}__{axis}__{part}__mean.npz", CACHE)
    if not p.exists():
        raise FileNotFoundError(f"missing cache {p.name}")
    return p


def fit_probe(x, y, seed):
    """Identical to repair_test.fit_probe / moment_decomposition.fit_probe."""
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


def thr_fast(neg, target):
    """Same rule in O(n log n). count(neg >= t) <= k with k = floor(target * n)
    holds iff t > the (k+1)-th largest score; take the smallest unique score above it."""
    n = neg.size
    k = int(np.floor(target * n + 1e-12))
    if k >= n:
        return float(np.min(neg))
    kth = np.sort(neg)[::-1][k]
    u = np.unique(neg)
    j = np.searchsorted(u, kth, side="right")
    return float(u[j]) if j < u.size else float(np.inf)


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="No-update noise floor for a repair_/moment_ result")
    ap.add_argument("--result", required=True, help="results/repair_*.json or results/moment_*.json")
    ap.add_argument("--n-resplits", type=int, default=1000)
    ap.add_argument("--resplit-seed", type=int, default=12345)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rp = Path(args.result)
    rp = rp if rp.is_absolute() else ROOT / rp
    res = json.load(open(rp, encoding="utf-8"))
    cfg = res["config"]
    kind = "moment" if "summary" in res else "repair"
    tgt, seeds, sub, axis = cfg["target_fpr"], cfg["seeds"], cfg["subsample"], cfg["axis"]

    pre, hashes, bad = {}, {}, {}
    for part in PARTS:
        p = cache(cfg["pre_repo"], cfg["pre_rev"], axis, part)
        q = cache(cfg["post_repo"], cfg["post_rev"], axis, part)
        z = np.load(p, allow_pickle=False)
        with np.load(q, allow_pickle=False) as zq:
            post_layers = {k for k in zq.files if k.startswith("layer_")}
            nf_post = zq["nonfinite"].astype(bool)
            assert np.array_equal(z["item_ids"], zq["item_ids"]), f"item_ids differ for {part}"
        bad[part] = z["nonfinite"].astype(bool) | nf_post
        pre[part] = {"z": z, "keep": ~bad[part], "post_layers": post_layers}
        hashes[f"pre/{part}"] = sha256_file(p)
        hashes[f"post/{part}"] = sha256_file(q)

    common = sorted([k for k in pre["probe_train"]["z"].files
                     if k.startswith("layer_") and k in pre["probe_train"]["post_layers"]],
                    key=lambda s: int(s.split("_")[1]))
    if cfg.get("layers"):
        want = {f"layer_{i}" for i in cfg["layers"]}
        common = [k for k in common if k in want]

    lab = {part: pre[part]["z"]["labels"][pre[part]["keep"]].astype(bool) for part in PARTS}
    n_tr = int(lab["probe_train"].size)
    neg_cal_n, neg_ev_n = int((~lab["threshold_calib"]).sum()), int((~lab["eval"]).sum())
    print(f"{rp.name}: kind {kind}, axis {axis}, {len(common)} layers, seeds {seeds}, "
          f"negatives calib {neg_cal_n} / eval {neg_ev_n}, non-finite dropped "
          f"{ {p: int(bad[p].sum()) for p in PARTS} }")

    # Self-test of the fast threshold rule against the reference rule, with ties.
    trng = np.random.default_rng(0)
    for _ in range(300):
        v = np.round(trng.normal(size=int(trng.integers(5, 1500))), int(trng.integers(0, 3)))
        assert thr_fast(v, tgt) == thr_at_fpr(v, tgt), "thr_fast disagrees with the reference rule"

    # Negative scores per layer x seed on the pooled (calib, eval) negatives.
    neg_scores = np.empty((len(common), len(seeds), neg_cal_n + neg_ev_n))
    auroc_pre, asis_fpr = [], np.empty((len(common), len(seeds)))
    for i, k in enumerate(common):
        x_tr = pre["probe_train"]["z"][k][pre["probe_train"]["keep"]]
        x_cal = pre["threshold_calib"]["z"][k][pre["threshold_calib"]["keep"]]
        x_ev = pre["eval"]["z"][k][pre["eval"]["keep"]]
        au = []
        for j, seed in enumerate(seeds):
            rng = np.random.default_rng(seed)
            idx = rng.choice(n_tr, size=int(sub * n_tr), replace=False)
            w, b, mu, sd = fit_probe(x_tr[idx], lab["probe_train"][idx], seed)
            s_cal, s_ev = (x_cal - mu) / sd @ w + b, (x_ev - mu) / sd @ w + b
            t = thr_at_fpr(s_cal[~lab["threshold_calib"]], tgt)
            asis_fpr[i, j] = float((s_ev[~lab["eval"]] >= t).mean())
            neg_scores[i, j] = np.concatenate([s_cal[~lab["threshold_calib"]], s_ev[~lab["eval"]]])
            au.append(M.auroc(s_ev[lab["eval"]], s_ev[~lab["eval"]]).value)
        auroc_pre.append(float(np.mean(au)))
        print(f"  {k:<9} as-realised FPR {asis_fpr[i].mean():.4f}  AUROC_pre {auroc_pre[-1]:.4f}", flush=True)

    def metric(fpr_ls: np.ndarray) -> float:  # layers x seeds -> the scripts' summary metric
        # A censored cell (seed-mean FPR of exactly 0) is floored at the measurement
        # resolution 1/n_neg_eval, not at 1e-9 (F-034): 1e-9 enters the average as 16.1
        # and can dominate a mean over seventeen layers. No cell in any stored floor file
        # is censored -- the smallest seed-mean FPR across all 24 is 0.0028, against a
        # resolution of 0.0005 -- so this changes none of the reported numbers, but the
        # old substitution contradicted what the Method says and would have fired
        # silently on a future run.
        floor = 1.0 / neg_ev_n
        return float(np.mean([abs(np.log(max(m, floor) / tgt)) for m in fpr_ls.mean(1)]))

    asis = metric(asis_fpr)

    precondition: dict[str, Any] = {"checked": False}
    if kind == "moment":
        stored = {f"layer_{r['layer']}": r["auroc_pre"] for r in res["rows"]}
        diffs = [abs(stored[k] - a) for k, a in zip(common, auroc_pre) if k in stored]
        precondition = {"checked": True, "n_layers": len(diffs),
                        "max_abs_auroc_pre_diff": float(max(diffs)),
                        # 1e-4, fixed before the first run: BLAS thread count can move
                        # LBFGS in the last bits (pilot_e6 re-run, <= 2.5e-7), which can
                        # flip single score pairs; a different probe moves AUROC far more.
                        "pass": bool(len(diffs) == len(common) and max(diffs) <= 1e-4)}
        print(f"  precondition (auroc_pre reproduces stored): {precondition}")
        if not precondition["pass"]:
            print("PRECONDITION FAILED - not the same probes; refusing to report a floor")
            return 1

    rrng = np.random.default_rng(args.resplit_seed)
    null = np.empty(args.n_resplits)
    fpr_buf = np.empty((len(common), len(seeds)))
    for r in range(args.n_resplits):
        perm = rrng.permutation(neg_cal_n + neg_ev_n)
        ci, ei = perm[:neg_cal_n], perm[neg_cal_n:]
        for i in range(len(common)):
            for j in range(len(seeds)):
                v = neg_scores[i, j]
                fpr_buf[i, j] = float((v[ei] >= thr_fast(v[ci], tgt)).mean())
        null[r] = metric(fpr_buf)
    q = {f"q{int(p)}": float(np.quantile(null, p / 100)) for p in (5, 50, 90, 95, 99)}

    if kind == "moment":
        arms = {a: v["mean_log_fpr_err"] for a, v in res["summary"].items()}
    else:
        arms = dict(res["mean_abs_log_fpr_error"])
    verdict = {a: {"error": e, "at_floor_q95": bool(e <= q["q95"]),
                   "null_fraction_ge": float((null >= e).mean())} for a, e in arms.items()}

    print(f"\n  no-update floor: as-realised {asis:.4f}   resplit null "
          + "  ".join(f"{k} {v:.4f}" for k, v in q.items()))
    for a, v in verdict.items():
        print(f"    {a:<15} error {v['error']:.4f}  at floor (<= q95): {v['at_floor_q95']}  "
              f"P(null >= error) {v['null_fraction_ge']:.3f}")

    payload = {**provenance_block("noise_floor"), "preregistration": "D-083",
               "audited_result": rp.name, "audited_result_sha256": sha256_file(rp),
               "kind": kind, "config": vars(args), "input_cache_sha256": hashes,
               "n_neg_calib": neg_cal_n, "n_neg_eval": neg_ev_n,
               "layers": [int(k.split("_")[1]) for k in common],
               "auroc_pre_by_layer": auroc_pre, "precondition": precondition,
               "floor_as_realised": asis, "asis_fpr_seed_mean_by_layer": asis_fpr.mean(1).tolist(),
               "null_quantiles": q, "null_mean": float(null.mean()), "arms": verdict}
    digest = write_results(args.out, payload)
    print(f"\nWritten: {args.out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
