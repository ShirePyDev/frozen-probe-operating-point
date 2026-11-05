"""Does the safety-versus-general damage gap survive removing contaminated eval items? (D-110)

D-109 measured, for every frozen safety prompt, the maximum Jaccard similarity to the
6400 examples each run actually trained on. The residual contamination is small (2.17%
of safety items for the HH arm, 0.96% for PKU, 0.00% for alpaca; under 1% of BENIGN
items, which are the ones a false-positive rate is computed on) - but small is an
argument, not a measurement.

This recomputes the frozen-threshold damage twice: on all evaluation items, and on the
subset whose nearest trained example is below the similarity threshold. If the damage
is contamination-driven it should fall when those items are dropped.

Probe protocol, threshold rule and metric are identical to repair_test.py, and the
metric floors a censored FPR at the eval resolution (F-034).
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
RES = ROOT / "results"


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


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Damage with contaminated eval items removed (D-110)")
    ap.add_argument("--result", required=True, help="a repair_*.json naming the pair")
    ap.add_argument("--arm", required=True, help="training-record name, key into the per-item map")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    res = json.load(open(ROOT / args.result, encoding="utf-8"))
    cfg = res["config"]
    tgt, seeds, sub, axis = cfg["target_fpr"], cfg["seeds"], cfg["subsample"], cfg["axis"]
    jac = json.load(open(RES / "paraphrase_per_item_safety.json", encoding="utf-8"))[args.arm]

    data, hashes = {}, {}
    for side, repo, rev in (("pre", cfg["pre_repo"], cfg["pre_rev"]),
                            ("post", cfg["post_repo"], cfg["post_rev"])):
        data[side] = {}
        for part in ("probe_train", "threshold_calib", "eval"):
            z, p = load(repo, rev, axis, part)
            data[side][part] = {k: z[k] for k in z.files}
            hashes[f"{side}/{part}"] = sha256_file(p)
    for part in ("probe_train", "threshold_calib", "eval"):
        bad = (data["pre"][part]["nonfinite"].astype(bool) | data["post"][part]["nonfinite"].astype(bool))
        if bad.any():
            for side in ("pre", "post"):
                for k, arr in list(data[side][part].items()):
                    if hasattr(arr, "shape") and arr.shape and arr.shape[0] == bad.shape[0]:
                        data[side][part][k] = arr[~bad]

    P, Q = data["pre"], data["post"]
    keys = sorted([k for k in P["probe_train"] if k.startswith("layer_") and k in Q["probe_train"]],
                  key=lambda s: int(s.split("_")[1]))
    y_tr = P["probe_train"]["labels"].astype(bool)
    y_cal = P["threshold_calib"]["labels"].astype(bool)
    y_ev = P["eval"]["labels"].astype(bool)
    ev_ids = [str(i) for i in P["eval"]["item_ids"]]
    sim = np.array([jac.get(i, 0.0) for i in ev_ids])
    clean = sim < args.threshold
    n_tr = len(y_tr)
    neg_all = ~y_ev
    neg_clean = neg_all & clean
    print(f"{args.arm}: eval items {len(ev_ids)}, benign {int(neg_all.sum())}, "
          f"benign contaminated at J>={args.threshold}: {int((neg_all & ~clean).sum())} "
          f"({(neg_all & ~clean).sum() / max(neg_all.sum(), 1):.2%})")

    def metric(fprs: list[float], n_neg: int) -> float:
        return float(np.mean([abs(np.log(max(f, 1.0 / max(n_neg, 1)) / tgt)) for f in fprs]))

    out: dict[str, Any] = {}
    for arm_name, subset in (("all_eval_items", neg_all), ("contamination_free", neg_clean)):
        per_layer = {"frozen": [], "restandardise": [], "refit": []}
        for k in keys:
            acc = {a: [] for a in per_layer}
            for seed in seeds:
                rng = np.random.default_rng(seed)
                idx = rng.choice(n_tr, size=int(sub * n_tr), replace=False)
                w, b, mu_a, sd_a = fit_probe(P["probe_train"][k][idx], y_tr[idx], seed)
                sc = lambda x, mu, sd: (x - mu) / sd @ w + b  # noqa: E731
                t = thr_at_fpr(sc(P["threshold_calib"][k], mu_a, sd_a)[~y_cal], tgt)
                xb = Q["probe_train"][k][idx]
                mu_b, sd_b = xb.mean(0), np.where(xb.std(0) < 1e-8, 1.0, xb.std(0))
                acc["frozen"].append(float((sc(Q["eval"][k], mu_a, sd_a)[subset] >= t).mean()))
                acc["restandardise"].append(float((sc(Q["eval"][k], mu_b, sd_b)[subset] >= t).mean()))
                w2, b2, mu2, sd2 = fit_probe(xb, y_tr[idx], seed)
                s2c = (Q["threshold_calib"][k] - mu2) / sd2 @ w2 + b2
                t2 = thr_at_fpr(s2c[~y_cal], tgt)
                s2 = (Q["eval"][k] - mu2) / sd2 @ w2 + b2
                acc["refit"].append(float((s2[subset] >= t2).mean()))
            for a in per_layer:
                per_layer[a].append(float(np.mean(acc[a])))
        n_neg = int(subset.sum())
        out[arm_name] = {"n_benign_items": n_neg,
                         **{a: metric(v, n_neg) for a, v in per_layer.items()}}
        print(f"  {arm_name:<20} n={n_neg:<5} frozen {out[arm_name]['frozen']:.4f}  "
              f"restandardise {out[arm_name]['restandardise']:.4f}  refit {out[arm_name]['refit']:.4f}")

    ch = out["contamination_free"]["frozen"] - out["all_eval_items"]["frozen"]
    rel = ch / out["all_eval_items"]["frozen"] if out["all_eval_items"]["frozen"] else 0.0
    print(f"  frozen damage change when contaminated items are dropped: {ch:+.4f} ({rel:+.1%})")

    payload = {**provenance_block("contamination_robustness"), "preregistration": "D-110",
               "arm": args.arm, "audited_result": args.result, "threshold": args.threshold,
               "input_cache_sha256": hashes, "results": out,
               "frozen_change_absolute": ch, "frozen_change_relative": rel}
    digest = write_results(args.out, payload)
    print(f"\nWritten: {args.out}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
