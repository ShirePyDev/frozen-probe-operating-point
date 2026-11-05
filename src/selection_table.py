"""The deployer's-probe selection table, per seed, with the C=1 column at both layers.

Answers three things the sweep summary left implicit:

  * the per-SEED ho20 AUROC and ho20 TPR behind each selection, not just their mean,
    so the spread is visible and a selection resting on one lucky seed would show;
  * which layer the C=1 comparison is taken at -- reported BOTH at C=1's own best
    layer and at the selected probe's layer, because those are different questions
    and quoting whichever is more favourable would be cherry-picking;
  * that selection touched the pre-update model only.

Selection protocol is pre-registered as D-117. This script does not select anything;
it re-derives the per-seed detail of a selection already made, from the same caches.

Reads frozen caches and the sweep result. CPU only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import (  # noqa: E402
    ensure_utf8_stdout, provenance_block, resolve_cache, sha256_file, write_results,
)
from regularisation_sweep import auroc, fit_logistic, fit_meandiff, thr_at_fpr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "hidden"
RESULTS = ROOT / "results"


def load(repo: str, rev: str, axis: str, part: str):
    p = resolve_cache(f"{repo.replace('/', '__')}@{rev[:12]}__{axis}__{part}__mean.npz",
                      CACHE)
    if not p.exists():
        raise FileNotFoundError(p.name)
    return np.load(p, allow_pickle=False), p


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Per-seed selection table")
    ap.add_argument("--sweep", default=str(RESULTS / "regularisation_sweep.json"))
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--holdout", type=float, default=0.20)
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--out", default=str(RESULTS / "selection_table.json"))
    args = ap.parse_args()

    sweep = json.load(open(args.sweep, encoding="utf-8"))
    cfgp = sweep.get("config", {})
    print("SELECTION PROTOCOL")
    print("  decision-log entry : D-117 (written before the sweep ran)")
    print(f"  grid               : C in {cfgp.get('C_grid')}")
    print(f"  selection data     : {cfgp.get('selection_population')}")
    print(f"  eval used to select: {cfgp.get('eval_used_for_selection')}")
    print("  pre-update only    : the sweep loads cfg['pre_repo'] / cfg['pre_rev'] and")
    print("                       never the post side; threshold_calib is loaded only to")
    print("                       set the 1% threshold for REPORTING TPR, never to choose")
    print("                       C or the layer.")
    print()

    rows, inputs = [], {}
    for m in sweep["models"]:
        stem, axis = m["pair"], m["axis"]
        sel = m["selected"]
        C = float(sel["arm"].split("=")[1]) if sel["arm"].startswith("C=") else None
        L = int(sel["layer"])
        rp = json.load(open(RESULTS / f"{stem}.json", encoding="utf-8"))["config"]
        try:
            z_tr, p_tr = load(rp["pre_repo"], rp["pre_rev"], axis, "probe_train")
            z_cal, p_cal = load(rp["pre_repo"], rp["pre_rev"], axis, "threshold_calib")
        except FileNotFoundError as e:
            print(f"  BLOCKED {stem}: {e}")
            continue
        inputs[f"{stem}:pre/probe_train"] = sha256_file(p_tr)
        inputs[f"{stem}:pre/threshold_calib"] = sha256_file(p_cal)

        y = z_tr["labels"].astype(bool)
        y_cal = z_cal["labels"].astype(bool)
        n = len(y)
        k = f"layer_{L}"
        per_seed = []
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(n)
            n_ho = int(args.holdout * n)
            ho, fit = perm[:n_ho], perm[n_ho:]
            xf = z_tr[k][fit]
            mu, sd = xf.mean(0), xf.std(0)
            sd = np.where(sd < 1e-8, 1.0, sd)
            xf_s = (xf - mu) / sd
            if C is None:
                w, b = fit_meandiff(xf_s, y[fit])
            else:
                w, b = fit_logistic(xf_s, y[fit], seed, C)
            s_ho = (z_tr[k][ho] - mu) / sd @ w + b
            t = thr_at_fpr(((z_cal[k] - mu) / sd @ w + b)[~y_cal], args.target_fpr)
            per_seed.append({"seed": seed,
                             "auroc_ho20": auroc(s_ho, y[ho]),
                             "tpr_ho20_at_1pct": float((s_ho[y[ho]] >= t).mean())})

        g = m["grid"]
        c1 = g.get("C=1", {})
        c1_best_layer = max(c1, key=lambda kk: c1[kk]["auroc_ho20"]) if c1 else None
        rows.append({
            "pair": stem, "axis": axis, "selected_C": C, "selected_layer": L,
            "per_seed": per_seed,
            "auroc_ho20_mean": float(np.mean([p["auroc_ho20"] for p in per_seed])),
            "tpr_ho20_mean": float(np.mean([p["tpr_ho20_at_1pct"] for p in per_seed])),
            "c1_at_selected_layer": c1.get(str(L)),
            "c1_best_layer": int(c1_best_layer) if c1_best_layer is not None else None,
            "c1_at_own_best_layer": c1.get(c1_best_layer) if c1_best_layer else None,
        })

    print("PER-PAIR SELECTION, ho20 AUROC BY SEED  (ho20 = the seed's 20% of "
          "probe_train, pre-update model)")
    hdr = "  {:<30}{:>6}{:>4}".format("pair", "C", "L")
    hdr += "".join(f"{'s' + str(s):>8}" for s in args.seeds) + f"{'mean':>8}{'TPR@1%':>8}"
    print(hdr)
    for r in rows:
        line = f"  {r['pair'][:28]:<30}{r['selected_C']:>6.3f}{r['selected_layer']:>4}"
        line += "".join(f"{p['auroc_ho20']:>8.4f}" for p in r["per_seed"])
        line += f"{r['auroc_ho20_mean']:>8.4f}{r['tpr_ho20_mean']:>8.4f}"
        print(line)

    print()
    print("  ho20 TPR at the 1% threshold set on threshold_calib, by seed:")
    for r in rows:
        vals = " ".join(f"{p['tpr_ho20_at_1pct']:.4f}" for p in r["per_seed"])
        print(f"  {r['pair'][:28]:<30}{vals}")

    print()
    print("THE C=1 COLUMN, AT BOTH LAYERS  (C=1 is the scikit-learn / deployed default)")
    print(f"  {'pair':<30}{'sel L':>6}{'C=1 @ sel L':>13}{'C=1 own best L':>16}"
          f"{'C=1 @ own best':>15}")
    for r in rows:
        a = r["c1_at_selected_layer"]
        b = r["c1_at_own_best_layer"]
        print(f"  {r['pair'][:28]:<30}{r['selected_layer']:>6}"
              f"{(a['auroc_ho20'] if a else float('nan')):>13.4f}"
              f"{str(r['c1_best_layer']):>16}"
              f"{(b['auroc_ho20'] if b else float('nan')):>15.4f}")

    write_results(Path(args.out), {
        **provenance_block("selection_table"),
        "protocol_decision_id": "D-117",
        "config": {"seeds": args.seeds, "holdout": args.holdout,
                   "target_fpr": args.target_fpr,
                   "selection_population": cfgp.get("selection_population"),
                   "eval_used_for_selection": cfgp.get("eval_used_for_selection"),
                   "pre_update_model_only": True},
        "input_cache_sha256": inputs,
        "rows": rows,
    })
    print(f"\nWritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
