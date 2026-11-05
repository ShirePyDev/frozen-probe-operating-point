"""F-045 regenerated under D-156 (D-169 item 15, D-179 (2); D-239): does the probe interpolate its training set?

For each of the eight cells of the superseded overfit_* results (the pre models of the nine released repair pairs,
per axis: Mistral v0.1, Llama-3, Llama-3.1, Qwen2.5, OLMo-2 and Gemma-2 on Aegis; Mistral v0.1 and Llama-3 on
WildGuardMix), at every layer of the grid, the PRE model only (no update is involved), with the D-117 probes: the
selected C (the deployer's probe, fitted on the D-117 80% split in permutation order) and C=1 (file order), seeds
0-4, exactly as depth_d156 and repair_d156 fit them.
  THRESHOLD-FREE, per seed and their mean: AUROC on the seed's own fitting items (train), on its D-117 holdout
    (the 20% of probe_train the selection held out; never fitted), and on the eval split; train minus holdout.
  AT THE FROZEN THRESHOLD, under D-156 (item level; the D-164 cluster level beside): T_d, each seed's 1% threshold
    on the PRE scores of the draw's calibration part cal_d; then, seed means of raw rates: FPR on cal_d itself,
    on the seed's own training negatives, on its holdout negatives, and on the draw's eval part; the order-statistic
    expectation fpr_calib * n/(n+1); the share of draws where the train FPR is 0 in every seed. D-156 mean, 95% band,
    frozen split. Per probe set, the per-draw mean over layers of each rate, with its band.
  ANCHOR: the seed mean of floored eval FPRs at T_d must equal the matching depth_d156 file's fpr_pre exactly, at
    every layer, probe set and level, so the probes, thresholds and draws are the ones every D-156 item used.
Writes results/overfit_d156_<cell>.json (never overwritten); a test writes outside results/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_d156 as D  # noqa: E402
import input_registry as REG  # noqa: E402
import metrics as M  # noqa: E402
import paired as P  # noqa: E402
import populations as POP  # noqa: E402
from provenance import (  # noqa: E402
    ensure_utf8_stdout, provenance_block, resolve_cache, sha256_bytes, sha256_file, write_results,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CODE_FILES = ["src/overfit_d156.py", *D.CODE_FILES]
LEVELS, PROBE_SETS, REGISTERED = D.LEVELS, D.PROBE_SETS, D.REGISTERED
RATES = ("fpr_calib", "fpr_train", "fpr_holdout", "fpr_eval", "expected_fpr_order_stat")
NULL_SEED = 20260959                                  # D-259 (4): the holdout-FPR re-partition null (no other stream uses it)


def cells() -> dict[str, dict[str, Any]]:
    """The eight cells, named as the superseded overfit_* files: <pre repo basename without dots>__<axis>."""
    out: dict[str, dict[str, Any]] = {}
    for stem in POP.UPDATE_PAIRS_REPAIR:
        c = REG.run(stem)
        name = f"{c['pre_repo'].split('/')[-1].replace('.', '')}__{c['axis']}"
        out.setdefault(name, {"repo": c["pre_repo"], "rev": c["pre_rev"], "axis": c["axis"], "pairs": []})["pairs"].append(stem)
    return out


def thresholds(S: np.ndarray, target: float) -> np.ndarray:
    return np.array([P.thr_at_fpr(S[:, s], target) for s in range(S.shape[1])])


def main() -> int:
    ensure_utf8_stdout()
    CELLS = cells()
    ap = argparse.ArgumentParser(description="F-045 under D-156 (D-169 item 15, D-239)")
    ap.add_argument("--cell", required=True, choices=sorted(CELLS))
    ap.add_argument("--n-draws", type=int, default=REGISTERED["n_draws"])
    ap.add_argument("--target-fpr", type=float, default=REGISTERED["target_fpr"])
    ap.add_argument("--seeds", type=int, nargs="*", default=REGISTERED["seeds"])
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="TESTING: a subset of layers")
    ap.add_argument("--out", default=None)
    ap.add_argument("--null", action="store_true",
                    help="D-256 (9), D-259 (4): add the re-partition null of the holdout FPR and write "
                         "overfit_d156_<cell>_null.json (the same probes, draws and anchors)")
    args = ap.parse_args()
    cell = CELLS[args.cell]
    # the anchor holds whenever the draws, target and seeds are the registered ones, so a layer-subset test checks it too
    anchor_on = (args.n_draws == REGISTERED["n_draws"] and args.target_fpr == REGISTERED["target_fpr"]
                 and list(args.seeds) == REGISTERED["seeds"])
    registered = anchor_on and args.layers is None
    final_path = (RESULTS / f"overfit_d156_{args.cell}{'_null' if args.null else ''}.json").resolve()
    out_path = Path(args.out).resolve() if args.out else final_path
    final = out_path == final_path
    if out_path.exists():
        raise SystemExit(f"REFUSING: {out_path} exists; results are never overwritten")
    if not final and out_path.is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: a test run (--out) must not write under results/")
    threads = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    if final and (not registered or any(t != "6" for t in threads.values())):
        raise SystemExit(f"REFUSING the final name: registered parameters {registered}, threads {threads}")
    code0 = P.code_block(CODE_FILES)
    sel_path = D.SELECTION_OF_RECORD
    sel_sha, cl_sha = sha256_file(sel_path), sha256_file(D.CLUSTERS)
    sel_doc = json.load(open(sel_path, encoding="utf-8"))
    if list(sel_doc["config"]["seeds"]) != list(args.seeds) or float(sel_doc["config"]["holdout"]) != P.HOLDOUT:
        raise SystemExit("REFUSING: selection seeds/holdout differ from this run's")
    sels = {m["pair"]: m["selected"] for m in sel_doc["models"] if m["pair"] in cell["pairs"]}
    if set(sels) != set(cell["pairs"]) or len({(s["arm"], s["layer"]) for s in sels.values()}) != 1:
        raise SystemExit(f"REFUSING: the pairs of {args.cell} do not share one D-117 selection: {sels}")
    arm, L_sel = next(iter(sels.values()))["arm"], int(next(iter(sels.values()))["layer"])
    if not arm.startswith("C="):
        raise SystemExit(f"REFUSING: {args.cell}'s selection is not a logistic C ({arm})")
    C_sel = float(arm.split("=")[1])
    # the anchor: the depth file of the cell's first pair (same pre caches, probes, draws)
    depth_rel = f"results/depth_d156_{cell['pairs'][0][len('repair_'):]}.json"
    depth_sha = sha256_file(ROOT / depth_rel)
    ddoc = json.load(open(ROOT / depth_rel, encoding="utf-8"))
    if (ddoc["config"]["selected_C"], ddoc["config"]["selected_layer"], ddoc["config"]["selection_sha256"]) != (C_sel, L_sel, sel_sha):
        raise SystemExit(f"REFUSING: {depth_rel} was run on another selection")
    dcells = {(c["layer"], c["probe_set"]): c for c in ddoc["cells"]}
    # the selection sweep's own AUROCs at the selected cell (same split, standardisation and fit): a second anchor
    sel_cell = next(m for m in sel_doc["models"] if m["pair"] == cell["pairs"][0])["grid"][arm][str(L_sel)]
    selection_check: dict[str, Any] = {}

    axis = cell["axis"]
    data: dict[str, Any] = {}
    input_sha: dict[str, str] = {}
    input_recs: dict[str, dict[str, Any]] = {}
    for part in ("probe_train", "threshold_calib", "eval"):
        z, r = REG.load(cell["repo"], axis, part)
        data[part] = z
        input_sha[f"pre/{part}"], input_recs[f"pre/{part}"] = r["sha256"], r
    reg_sources = REG.sources_sha256()
    if D.sources_now() != reg_sources:
        raise SystemExit("REFUSING: the extraction records or pins on disk are not the ones the registry read")
    if any(ddoc["input_cache_sha256"].get(k) != h for k, h in input_sha.items()):
        raise SystemExit(f"REFUSING: {depth_rel} read other pre caches")
    verification = REG.verification(input_recs)
    for z in data.values():
        if "nonfinite" in z.files and z["nonfinite"].any():
            raise SystemExit("REFUSING: non-finite items")
    grids = [sorted(int(k.split("_")[1]) for k in z.files if k.startswith("layer_")) for z in data.values()]
    if any(g != grids[0] for g in grids):
        raise SystemExit("REFUSING: the input caches have different layer grids")
    layers = [li for li in grids[0] if args.layers is None or li in args.layers]

    y_tr = data["probe_train"]["labels"].astype(bool)
    y_cal = data["threshold_calib"]["labels"].astype(bool)
    y_ev = data["eval"]["labels"].astype(bool)
    ncal, neg, pos = ~y_cal, ~y_ev, y_ev
    pooled_ids = [str(i) for i in data["threshold_calib"]["item_ids"][ncal]] + [str(i) for i in data["eval"]["item_ids"][neg]]
    n_cal, n_pool = int(ncal.sum()), len(pooled_ids)
    cl = json.load(open(D.CLUSTERS, encoding="utf-8"))["axes"][axis]
    if cl["n_pooled_negatives"] != n_pool or cl["n_calibration"] != n_cal or \
            cl["pool_item_ids_sha256"] != sha256_bytes("\n".join(pooled_ids).encode("utf-8")):
        raise SystemExit(f"REFUSING: manifests/negative_clusters.json does not describe the {axis} pool")
    tr_ids = {str(i) for i in data["probe_train"]["item_ids"]}
    if tr_ids & set(pooled_ids):
        raise SystemExit("REFUSING: a probe_train item is in the pooled negatives (F-045)")
    frozen_part = (np.arange(n_cal), np.arange(n_cal, n_pool))
    draws = {LEVELS[0]: P.repartition_draws(n_cal, n_pool - n_cal, args.n_draws, P.AXIS_DRAW_SEED[axis]),
             LEVELS[1]: P.cluster_repartition_draws(P.cluster_units(pooled_ids, cl["clusters"]), n_cal,
                                                    args.n_draws, P.AXIS_CLUSTER_DRAW_SEED[axis])}
    parts = {lvl: [frozen_part] + [(d[:n_cal], d[n_cal:]) for d in dr] for lvl, dr in draws.items()}
    print(f"overfit {args.cell} ({cell['repo']} @ {cell['rev'][:12]}, {axis}; pairs {cell['pairs']}): layers {layers}; "
          f"selected C {C_sel:g} (D-117 layer {L_sel}); pooled {n_cal} + {n_pool - n_cal}; D-170 flag {verification['flag']}; "
          f"registered {registered}", flush=True)

    out_cells: list[dict[str, Any]] = []
    per_layer: dict[tuple[str, str, str], list[np.ndarray]] = {}
    null_subs: dict[tuple[int, int, int], np.ndarray] = {}   # D-259 (4): (level, draw, seed) -> the null subset
    n_anchor = 0
    for li in layers:
        k = f"layer_{li}"
        Xtr, Xcal, Xev = data["probe_train"][k], data["threshold_calib"][k], data["eval"][k]
        for pset in PROBE_SETS:
            probes, fit_rows, ho_rows = [], [], []
            for seed in args.seeds:
                ho, fit = P.d117_split(len(y_tr), seed)
                rows, C, order = (fit, C_sel, "d117_permutation") if pset == "selected" else (np.sort(fit), 1.0, "file_order")
                probes.append(P.fit_probe(Xtr[rows], y_tr[rows], seed, C, order))
                fit_rows.append(rows)
                ho_rows.append(ho)
            pool = np.column_stack([np.concatenate([q.score(Xcal[ncal]), q.score(Xev[neg])]) for q in probes])
            pos_ev = np.column_stack([q.score(Xev[pos]) for q in probes])
            tr_s = [q.score(Xtr[r]) for q, r in zip(probes, fit_rows)]
            ho_s = [q.score(Xtr[r]) for q, r in zip(probes, ho_rows)]
            tr_neg = [s[~y_tr[r]] for s, r in zip(tr_s, fit_rows)]
            ho_neg = [s[~y_tr[r]] for s, r in zip(ho_s, ho_rows)]
            au = {"train": [M.auroc(s[y_tr[r]], s[~y_tr[r]]).value for s, r in zip(tr_s, fit_rows)],
                  "holdout": [M.auroc(s[y_tr[r]], s[~y_tr[r]]).value for s, r in zip(ho_s, ho_rows)],
                  "eval": [M.auroc(pos_ev[:, i], pool[n_cal:, i]).value for i in range(len(probes))]}
            gap = [a - b for a, b in zip(au["train"], au["holdout"])]
            dc = dcells[(li, pset)]
            if anchor_on and dc["auroc"]["pre"]["per_seed"] != au["eval"]:
                raise SystemExit(f"REFUSING: L{li} {pset}: the eval AUROC per seed is not {depth_rel}'s")
            if li == L_sel and pset == "selected":
                d_fit, d_ho = abs(float(np.mean(au["train"])) - sel_cell["auroc_fit"]), abs(float(np.mean(au["holdout"])) - sel_cell["auroc_ho20"])
                selection_check = {"layer": li, "arm": arm, "selection_auroc_fit": sel_cell["auroc_fit"],
                                   "selection_auroc_ho20": sel_cell["auroc_ho20"], "abs_diff_train": d_fit,
                                   "abs_diff_holdout": d_ho, "tolerance": 1e-6}
                if anchor_on and max(d_fit, d_ho) > 1e-6:
                    raise SystemExit(f"REFUSING: at the selected cell the train/holdout AUROC differ from the selection's: {selection_check}")
            cblk: dict[str, Any] = {
                "layer": li, "probe_set": pset, "probe_C": C_sel if pset == "selected" else 1.0,
                "input_verification": verification,
                "auroc": {kk: {"per_seed": v, "mean": float(np.mean(v))} for kk, v in au.items()},
                "auroc_train_minus_holdout": {"per_seed": gap, "mean": float(np.mean(gap))},
                "n_fit_items_per_seed": int(len(fit_rows[0])), "n_features": int(Xtr.shape[1]),
                "n_train_negatives_per_seed": [int(len(t)) for t in tr_neg],
                "n_holdout_negatives_per_seed": [int(len(h)) for h in ho_neg]}
            for lvl_i, lvl in enumerate(LEVELS):
                vals: dict[str, list[float]] = {q: [] for q in RATES}
                zero_all, fpr_pre, null_fpr = [], [], []
                for d_i, (c_, e_) in enumerate(parts[lvl]):
                    T = thresholds(pool[c_], args.target_fpr)
                    F_ev = pool[e_] >= T
                    fc = float((pool[c_] >= T).mean(axis=0).mean())
                    ftr = [float((tn >= T[i]).mean()) for i, tn in enumerate(tr_neg)]
                    vals["fpr_calib"].append(fc)
                    vals["fpr_train"].append(float(np.mean(ftr)))
                    vals["fpr_holdout"].append(float(np.mean([(hn >= T[i]).mean() for i, hn in enumerate(ho_neg)])))
                    vals["fpr_eval"].append(float(F_ev.mean(axis=0).mean()))
                    vals["expected_fpr_order_stat"].append(fc * n_cal / (n_cal + 1))
                    zero_all.append(all(f == 0.0 for f in ftr))
                    fpr_pre.append(float(P.floored_fpr(F_ev).mean()))
                    if args.null:
                        # D-259 (4): per seed, a random subset of the draw's EVALUATION part (never used to set T_d) of
                        # the size of that seed's holdout negatives; the same subset at every layer and probe set
                        nf = []
                        for i, hn in enumerate(ho_neg):
                            key_ = (lvl_i, d_i, args.seeds[i])
                            if key_ not in null_subs:          # drawn once; the same subset at every layer and probe set
                                null_subs[key_] = np.random.default_rng([NULL_SEED, lvl_i, d_i, args.seeds[i]]).choice(
                                    len(e_), size=len(hn), replace=False)
                            nf.append(float(F_ev[null_subs[key_], i].mean()))
                        null_fpr.append(float(np.mean(nf)))
                # the anchor: depth_d156's pre-side FPR at every draw, exactly (its band summary)
                want = dc["operating_point"]["fpr_pre"] if lvl == LEVELS[0] else dc[LEVELS[1]]["operating_point"]["fpr_pre"]
                got = json.loads(json.dumps(P.band_summary(np.array(fpr_pre[1:]), fpr_pre[0])))
                if anchor_on and got != want:
                    raise SystemExit(f"REFUSING: L{li} {pset} {lvl}: the eval FPR at T_d is not {depth_rel}'s fpr_pre")
                n_anchor += int(anchor_on)
                cblk[lvl] = {q: P.band_summary(np.array(v[1:]), v[0]) for q, v in vals.items()}
                cblk[lvl]["share_of_draws_train_fpr_zero_in_every_seed"] = float(np.mean(zero_all[1:]))
                cblk[lvl]["frozen_split_train_fpr_zero_in_every_seed"] = bool(zero_all[0])
                for q in ("fpr_train", "fpr_holdout", "fpr_eval"):
                    per_layer.setdefault((pset, lvl, q), []).append(np.array(vals[q]))
                if args.null:
                    nu, dd = np.array(null_fpr), np.array(vals["fpr_holdout"]) - np.array(null_fpr)
                    cblk[lvl]["fpr_holdout_null"] = P.band_summary(nu[1:], nu[0])
                    cblk[lvl]["holdout_minus_null"] = P.band_summary(dd[1:], dd[0])
                    per_layer.setdefault((pset, lvl, "fpr_holdout_null"), []).append(nu)
                    per_layer.setdefault((pset, lvl, "holdout_minus_null"), []).append(dd)
            out_cells.append(cblk)
        print(f"  layer {li:>2} done", flush=True)

    summary: dict[str, Any] = {}
    for pset in PROBE_SETS:
        cs = [c for c in out_cells if c["probe_set"] == pset]
        summary[pset] = {"auroc_train_mean_over_layers": float(np.mean([c["auroc"]["train"]["mean"] for c in cs])),
                         "auroc_holdout_mean_over_layers": float(np.mean([c["auroc"]["holdout"]["mean"] for c in cs])),
                         "auroc_train_minus_holdout_mean_over_layers": float(np.mean([c["auroc_train_minus_holdout"]["mean"] for c in cs])),
                         "n_layers_auroc_train_1_every_seed": int(sum(all(v == 1.0 for v in c["auroc"]["train"]["per_seed"]) for c in cs))}
        for lvl in LEVELS:
            summary[pset][lvl] = {}
            for q in ("fpr_train", "fpr_holdout", "fpr_eval", *(("fpr_holdout_null", "holdout_minus_null") if args.null else ())):
                mq = np.mean(np.vstack(per_layer[(pset, lvl, q)]), axis=0)
                summary[pset][lvl][f"{q}_mean_over_layers"] = P.band_summary(mq[1:], mq[0])
            if args.null:
                b_ = summary[pset][lvl]["holdout_minus_null_mean_over_layers"]["band95"]
                judged = (pset, lvl) == ("selected", LEVELS[0])
                summary[pset][lvl]["excess_over_null" if judged else "excess_over_null_not_judged"] = bool(b_[0] > 0)
    if anchor_on and n_anchor != len(layers) * len(PROBE_SETS) * len(LEVELS):
        raise SystemExit(f"REFUSING to write: {n_anchor} anchor blocks")
    if registered and not selection_check:
        raise SystemExit("REFUSING to write: the selected cell was never checked against the selection sweep")
    companion = None
    if args.null:
        # D-259 (4): the null record re-derives the D-250 record's numbers; tie it to that record (named by its manifest)
        # and, for a registered run, refuse unless every non-null number equals it (input_verification aside: D-168 may
        # change the flag, not the numbers)
        comp_rel, comp_man = f"results/overfit_d156_{args.cell}.json", ROOT / "logs" / "overfit_d156_20260927.sha256"
        cb = (ROOT / comp_rel).read_bytes()
        hits = [ln.split(maxsplit=1)[0] for ln in comp_man.read_text(encoding="utf-8").splitlines()
                if ln.strip() and ln.split(maxsplit=1)[1].strip().lstrip("*") == comp_rel]
        if hits != [sha256_bytes(cb)]:
            raise SystemExit(f"REFUSING: {comp_rel} is not the file {comp_man.name} names (once)")
        companion = {"file": comp_rel, "sha256": sha256_bytes(cb), "manifest": "logs/overfit_d156_20260927.sha256",
                     "non_null_numbers_identical": None}
        if registered:
            cdoc = json.loads(cb.decode("utf-8"))
            nulls = ("fpr_holdout_null", "holdout_minus_null", "fpr_holdout_null_mean_over_layers",
                     "holdout_minus_null_mean_over_layers", "excess_over_null", "excess_over_null_not_judged")
            strip = lambda x: ({k: strip(v) for k, v in x.items() if k not in nulls and k != "input_verification"}  # noqa: E731
                               if isinstance(x, dict) else [strip(v) for v in x] if isinstance(x, list) else x)
            canon = lambda x: json.dumps(json.loads(json.dumps(strip(x), default=str)), sort_keys=True)  # noqa: E731
            same = canon(out_cells) == canon(cdoc["cells"]) and canon(summary) == canon(cdoc["summary"])
            if not same:
                raise SystemExit(f"REFUSING to write: the null run's non-null numbers are not {comp_rel}'s (D-259 (4))")
            companion["non_null_numbers_identical"] = True
    code1 = P.code_block(CODE_FILES)
    changed = [n for n, a, b in (("code", code1["sha256"], code0["sha256"]), ("selection", sha256_file(sel_path), sel_sha),
                                 ("cluster manifest", sha256_file(D.CLUSTERS), cl_sha), (depth_rel, sha256_file(ROOT / depth_rel), depth_sha),
                                 ("registry sources", D.sources_now(), reg_sources)) if a != b]
    changed += [f"cache {r['file']}" for r in input_recs.values() if sha256_file(resolve_cache(r["file"], REG.CACHE)) != r["sha256"]]
    if changed:
        raise SystemExit(f"REFUSING to write: changed during the run: {changed}")
    payload = {**provenance_block("overfit_d156"),
               "preregistration": "D-169 item 15, D-179 (2), D-239" + ("; D-256 (9), D-259 (4) (the re-partition null)" if args.null else ""),
               **({"null": {"seed": NULL_SEED, "rng": "numpy default_rng([seed, level index, draw index, probe seed])",
                            "levels": list(LEVELS),
                            "method": "per seed, a random subset without replacement of the draw's evaluation part, of the "
                                      "size of that seed's holdout negatives; the same subset at every layer and probe set; "
                                      "its raw FPR at T_d is the null. holdout_minus_null = observed holdout FPR - null, per "
                                      "draw. JUDGED: selected C, item level, the mean over layers; excess = band wholly above 0"}}
                  if args.null else {}),
               "code": code0,
               "registered_parameters": registered, "blas_threads": threads, "input_verification": verification,
               "config": {"cell": args.cell, "repo": cell["repo"], "rev": cell["rev"], "axis": axis, "pairs": cell["pairs"],
                          "selection_file": str(sel_path.relative_to(ROOT)).replace("\\", "/"), "selection_sha256": sel_sha,
                          "selected_C": C_sel, "selected_layer": L_sel, "layers": layers, "seeds": args.seeds,
                          "target_fpr": args.target_fpr, "n_draws": args.n_draws, "draw_seed": P.AXIS_DRAW_SEED[axis],
                          "cluster_draw_seed": P.AXIS_CLUSTER_DRAW_SEED[axis],
                          "cluster_manifest": {"file": "manifests/negative_clusters.json", "sha256": cl_sha},
                          "input_registry_sources_sha256": reg_sources},
               "anchor": {"file": depth_rel, "sha256": depth_sha, "n_level_blocks_identical": n_anchor,
                          "kind": "the seed mean of floored eval FPRs at T_d equals the depth file's fpr_pre band summary; "
                                  "the eval AUROC per seed equals its auroc.pre.per_seed"},
               "selection_check": selection_check,
               "rates_note": "raw rates (not floored): a train FPR of exactly 0 is the F-045 finding itself. The holdout is "
                             "the D-117 selection's 20% of probe_train (never fitted, but used to CHOOSE C and layer), so at "
                             "the selected layer and C its AUROC and FPR are optimistic and the train-minus-holdout gap is "
                             "understated there. A holdout FPR rests on about 20% of probe_train's negatives (counts per "
                             "seed in each cell). The order-statistic expectation k/(n+1) assumes exchangeable items: exact at "
                             "the item level, approximate at the cluster level, where whole clusters move together",
               "superseded": None if args.null else f"results/overfit_{args.cell}.json (D-161)",
               **({"companion": companion} if args.null else {}),
               "input_cache_sha256": input_sha,
               "input_cache_source": {kk: {"file": r["file"], "expected_from": r["expected_from"], "pin_only": r["pin_only"]}
                                      for kk, r in input_recs.items()},
               "summary": summary, "cells": out_cells}
    digest = write_results(out_path, payload, exclusive=True)
    for pset in PROBE_SETS:
        s = summary[pset]
        ft = s[LEVELS[0]]["fpr_train_mean_over_layers"]
        print(f"{pset:8s} AUROC train {s['auroc_train_mean_over_layers']:.4f} holdout {s['auroc_holdout_mean_over_layers']:.4f} "
              f"(gap {s['auroc_train_minus_holdout_mean_over_layers']:+.4f}; train = 1 at {s['n_layers_auroc_train_1_every_seed']} "
              f"of {len(layers)} layers); train FPR at T_d {ft['mean']:.4f} [{ft['band95'][0]:.4f}, {ft['band95'][1]:.4f}]", flush=True)
    if args.null:
        s_ = summary["selected"][LEVELS[0]]
        dm = s_["holdout_minus_null_mean_over_layers"]
        print(f"null (selected C, item level, mean over layers): holdout - null {dm['mean']:+.4f} "
              f"[{dm['band95'][0]:+.4f}, {dm['band95'][1]:+.4f}]; excess over the null: {s_['excess_over_null']}", flush=True)
    print(f"Written: {out_path}\nsha256:  {digest}\nanchor blocks identical: {n_anchor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
