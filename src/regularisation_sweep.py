"""Does the frozen probe's regularisation change the story? Sweep C, and pick honestly.

F-045: at every real model x axis cell the probe reaches AUROC 1.0000 on its own fitting
set and FPR 0.0000 there. With 3200 fitting items in 4096 dimensions that is
interpolation, not skill, so C=1.0 is effectively unregularised. If the damage and the
repair only hold for an interpolating probe, that is a scope limit a reviewer will find.

This sweeps C over {1e-4, 1e-3, 1e-2, 1e-1, 1} and adds a mean-difference probe -- the
simplest possible readout, w = mu_pos - mu_neg on standardised features, no fitting at
all -- as a floor to compare against.

SELECTION RULE, and the reason it matters: C and the deployer's layer are chosen per
model on the SEED'S HELD-OUT 20% of probe_train (ho20), never on eval. An earlier table
in this project used one column name, "held-out", for two different populations (the
20% and eval), which is how the ambiguity was caught. Here ho20 is the only selection
signal; eval numbers are computed after selection and are labelled as reporting only.

Reads frozen caches, writes one results file. CPU only.

D-165 (2026-09-25): regenerated under committed, hashed code with the SAME protocol (grid, arms,
split, standardisation, fits, argmax on ho20, reporting TPR). The only changes are where inputs
come from and what is recorded: pairs, revisions and expected cache bytes from
src/input_registry.py (extraction records and committed manifests), never from a superseded
results file; a paired.code_block at start and end; a new output that is never overwritten.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import (  # noqa: E402
    ensure_utf8_stdout,
    provenance_block,
    write_results,
)
import input_registry as REG  # noqa: E402
import paired as P  # noqa: E402
import populations as POP  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CODE_FILES = ["src/regularisation_sweep.py", "src/input_registry.py", "src/paired.py", "src/populations.py",
              "src/provenance.py", "src/superseded_provenance.py"]
# Pre-registered in D-117, written before the full sweep ran.
C_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0]


def auroc(scores, y) -> float:
    """Rank-based AUROC; ties get average ranks."""
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    s = np.asarray(scores)
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_logistic(x, y, seed, C):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=2000, C=C, random_state=seed)
    clf.fit(x, y)
    return clf.coef_.ravel(), float(clf.intercept_[0])


def fit_meandiff(x, y):
    """No fitting: the difference of class means.

    x is ALREADY standardised by the caller, with the same mu and sd used for the
    logistic probe at that layer and seed, so this is a difference of means on
    standardised features and not on raw ones. That distinction matters: on raw
    features this readout is dominated by whichever dimensions happen to have the
    largest variance, which would make a best layer of 0 an artefact of scaling
    rather than a fact about the representation.

    The intercept only centres the score; AUROC is invariant to it.
    """
    w = x[y].mean(0) - x[~y].mean(0)
    return w, -float((x.mean(0) @ w))


# The frozen-threshold rule is defined once, in src/paired.py (2026-09-25). This file's
# local copy was verified to give identical thresholds on 12,000 random and tie-heavy
# cases before it was replaced.
from paired import thr_at_fpr  # noqa: E402


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Regularisation sweep with honest selection")
    ap.add_argument("--pairs", nargs="*", default=POP.UPDATE_PAIRS_REPAIR)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--holdout", type=float, default=0.20)
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--out", default=str(RESULTS / "selection_sweep_d165.json"))
    args = ap.parse_args()
    out_path = Path(args.out).resolve()
    if out_path.exists():
        raise SystemExit(f"REFUSING: {out_path} exists; results are never overwritten (D-152 P5)")
    code0 = P.code_block(CODE_FILES)

    models: list[dict[str, Any]] = []
    input_sha: dict[str, str] = {}
    input_source: dict[str, str] = {}
    input_recs: dict[str, dict[str, Any]] = {}
    runs: dict[str, dict[str, str]] = {}

    for stem in args.pairs:
        cfg = REG.run(stem)            # refuses an undeclared run; no superseded file is read
        runs[stem] = cfg
        axis = cfg["axis"]
        # threshold_calib is loaded for REPORTING only. It sets the 1% threshold at which the
        # already-selected probe's ho20 TPR is measured. It never enters the selection (D-117).
        z_tr, r_tr = REG.load(cfg["pre_repo"], axis, "probe_train")
        z_cal, r_cal = REG.load(cfg["pre_repo"], axis, "threshold_calib")
        for part, r in (("probe_train", r_tr), ("threshold_calib", r_cal)):
            input_sha[f"{stem}:pre/{part}"] = r["sha256"]
            input_source[f"{stem}:pre/{part}"] = f"{r['file']} (expected bytes from {r['expected_from']})"
            input_recs[f"{stem}:pre/{part}"] = r

        y_all = z_tr["labels"].astype(bool)
        n = len(y_all)
        keys = sorted((k for k in z_tr.files if k.startswith("layer_")),
                      key=lambda s: int(s.split("_")[1]))
        print(f"\n{stem}  [{axis}]  {n} probe_train items, {len(keys)} layers")
        print(f"  selection signal: held-out {int(args.holdout * 100)}% of probe_train "
              f"({int(args.holdout * n)} items), per seed. eval is NOT consulted.")

        # arm -> layer -> list over seeds of (auroc_fit, auroc_ho20)
        grid: dict[str, dict[int, list[tuple[float, float]]]] = {}
        for k in keys:
            li = int(k.split("_")[1])
            X = z_tr[k]
            for seed in args.seeds:
                rng = np.random.default_rng(seed)
                perm = rng.permutation(n)
                n_ho = int(args.holdout * n)
                ho, fit = perm[:n_ho], perm[n_ho:]
                xf, yf = X[fit], y_all[fit]
                mu, sd = xf.mean(0), xf.std(0)
                sd = np.where(sd < 1e-8, 1.0, sd)
                xf_s = (xf - mu) / sd
                xh_s = (X[ho] - mu) / sd
                yh = y_all[ho]
                for C in C_GRID:
                    w, b = fit_logistic(xf_s, yf, seed, C)
                    grid.setdefault(f"C={C:g}", {}).setdefault(li, []).append(
                        (auroc(xf_s @ w + b, yf), auroc(xh_s @ w + b, yh)))
                w, b = fit_meandiff(xf_s, yf)
                grid.setdefault("mean_diff", {}).setdefault(li, []).append(
                    (auroc(xf_s @ w + b, yf), auroc(xh_s @ w + b, yh)))

        # Mean over seeds, then the honest argmax over (arm, layer) on ho20 only.
        table = {}
        best = None
        for arm, per_layer in grid.items():
            table[arm] = {}
            for li, vals in per_layer.items():
                f = float(np.mean([v[0] for v in vals]))
                h = float(np.mean([v[1] for v in vals]))
                table[arm][li] = {"auroc_fit": f, "auroc_ho20": h, "gap": f - h}
                if best is None or h > best["auroc_ho20"]:
                    best = {"arm": arm, "layer": li, "auroc_fit": f, "auroc_ho20": h,
                            "gap": f - h}

        print(f"  {'arm':<12}{'best layer':>11}{'AUROC fit':>11}{'AUROC ho20':>12}{'gap':>8}")
        for arm in [f"C={C:g}" for C in C_GRID] + ["mean_diff"]:
            if arm not in table:
                continue
            bl = max(table[arm], key=lambda li: table[arm][li]["auroc_ho20"])
            e = table[arm][bl]
            print(f"  {arm:<12}{bl:>11}{e['auroc_fit']:>11.4f}{e['auroc_ho20']:>12.4f}"
                  f"{e['gap']:>8.4f}")
        print(f"  SELECTED on ho20: {best['arm']} at layer {best['layer']}, "
              f"ho20 AUROC {best['auroc_ho20']:.4f} (fit {best['auroc_fit']:.4f}, "
              f"gap {best['gap']:.4f})")

        # mean_diff profile across depth, so a best layer of 0 can be read in context
        md = table.get("mean_diff", {})
        if md:
            prof = " ".join(f"{li}:{md[li]['auroc_ho20']:.3f}" for li in sorted(md))
            print(f"  mean_diff ho20 AUROC by layer (standardised features): {prof}")

        # REPORTING ONLY: ho20 TPR of the selected probe at the 1% threshold taken
        # from threshold_calib. Not part of selection (D-117).
        k = f"layer_{best['layer']}"
        y_cal = z_cal["labels"].astype(bool)
        tprs, thrs = [], []
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(n)
            n_ho = int(args.holdout * n)
            ho, fit = perm[:n_ho], perm[n_ho:]
            xf = z_tr[k][fit]
            mu, sd = xf.mean(0), xf.std(0)
            sd = np.where(sd < 1e-8, 1.0, sd)
            xf_s = (xf - mu) / sd
            if best["arm"] == "mean_diff":
                w, b = fit_meandiff(xf_s, y_all[fit])
            else:
                w, b = fit_logistic(xf_s, y_all[fit], seed, float(best["arm"].split("=")[1]))
            t = thr_at_fpr(((z_cal[k] - mu) / sd @ w + b)[~y_cal], args.target_fpr)
            s_ho = (z_tr[k][ho] - mu) / sd @ w + b
            yh = y_all[ho]
            tprs.append(float((s_ho[yh] >= t).mean()))
            thrs.append(t)
        best["ho20_tpr_at_1pct"] = float(np.mean(tprs))
        best["ho20_tpr_range"] = [float(np.min(tprs)), float(np.max(tprs))]
        print(f"  ho20 TPR at the 1% threshold set on threshold_calib: "
              f"{np.mean(tprs):.4f}  (seed range {np.min(tprs):.4f}-{np.max(tprs):.4f})")

        models.append({"pair": stem, "axis": axis, "n_probe_train": n,
                       "holdout_fraction": args.holdout, "selected": best,
                       "grid": {a: {str(li): v for li, v in d.items()}
                                for a, d in table.items()}})

    if models:
        print("\n" + "=" * 78)
        print(f"  {'pair':<34}{'axis':<18}{'selected':<14}{'layer':>6}{'ho20':>9}")
        for m in models:
            s = m["selected"]
            print(f"  {m['pair'][:33]:<34}{m['axis']:<18}{s['arm']:<14}{s['layer']:>6}"
                  f"{s['auroc_ho20']:>9.4f}")
        by_axis: dict[str, list[float]] = {}
        for m in models:
            by_axis.setdefault(m["axis"], []).append(m["selected"]["auroc_ho20"])
        print("\n  best ho20 AUROC per dataset:")
        for ax, v in sorted(by_axis.items()):
            print(f"    {ax:<18} n={len(v)}  best {max(v):.4f}  mean {np.mean(v):.4f}")
        print("=" * 78)

        code1 = P.code_block(CODE_FILES)
        if code1["sha256"] != code0["sha256"]:
            raise SystemExit("REFUSING to write: source changed during the run")
        digest = write_results(out_path, {
            **provenance_block("regularisation_sweep"),
            "preregistration": "D-117 (protocol), D-165 (regenerated under committed code)",
            "code": code0,
            "config": {"C_grid": C_GRID, "seeds": args.seeds,
                       "holdout": args.holdout, "target_fpr": args.target_fpr,
                       "selection_population": "held-out 20% of probe_train, per seed",
                       "eval_used_for_selection": False,
                       "pairs": [m["pair"] for m in models],
                       "runs": runs, "inputs_from": "src/input_registry.py"},
            "input_verification": REG.verification(input_recs),
            "input_cache_sha256": input_sha,
            "input_cache_source": input_source,
            "models": models,
        }, exclusive=True)
        print(f"\nWritten: {out_path}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
