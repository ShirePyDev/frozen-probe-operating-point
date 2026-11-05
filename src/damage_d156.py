"""The nine-pair table and 4a under the D-156 re-partition estimator (second run: D-165, D-164).

D-156 (author's decision, 2026-09-25, after D-155/D-157 showed the Aegis frozen partition
is a ~1st-percentile draw of its own resplit null): every operating-point metric is
reported as the mean and 95% band over 1000 re-partitions of the pooled negatives
(threshold_calib + eval negatives, 1000 + 2000, their pre- AND post-update scores), with
the eval positives fixed; the frozen split is reported alongside as one draw, with its
percentile. One partition per draw is shared by all seeds, pre and post, and every cell on
the same axis. Frozen-split significance carries the minimum achievable McNemar p.

Scope (D-156): (A) the nine released update pairs of UPDATE_PAIRS_REPAIR at their D-117
selected layer; (B) 4a's nine LoRA runs (LORA_ARMS) at the A100 base's selected layer,
with the between-arm damage and TPR gaps formed within each draw. Both at the selected C
(D-117 split, permutation order) and at the C=1 sensitivity row (file order, library-default
tolerance, F-059). Probe = D-152 P1, five seeds.

Second run (D-165, D-169 item 2): the selection comes from the regenerated sweeps
(results/selection_sweep_d165.json, results/selection_sweep_lora_a100_d165.json); pairs,
revisions and expected cache bytes from src/input_registry.py, never from a superseded run
record. D-164: every cell and every between-arm gap is also reported under CLUSTER-LEVEL
re-partitions, in which no near-duplicate cluster of manifests/negative_clusters.json straddles
calibration and evaluation (paired.cluster_repartition_draws, its own per-axis seeds), beside
the item-level numbers. At the cluster level the 'frozen' value is still the frozen split, which
splits clusters and so is not a cluster-level draw (said in the output). Every cell and
between-arm row carries its D-170 input_verification (paper eligibility is decided per cell,
D-172); the selection files, the cluster manifest and the registry's sources are hashed at start
and, the first two, re-checked at the end. Writes results/damage_d156_r2.json; only the
registered parameters may write that name; results/damage_d156.json (the first run) is never
touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import input_registry as REG  # noqa: E402
import paired as P  # noqa: E402
import populations as POP  # noqa: E402
from provenance import ensure_utf8_stdout, provenance_block, sha256_bytes, sha256_file, write_results  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CODE_FILES = ["src/damage_d156.py", "src/input_registry.py", "src/paired.py", "src/populations.py",
              "src/provenance.py", "src/superseded_provenance.py"]
SELECTION = {"nine-pair table": RESULTS / "selection_sweep_d165.json",
             "4a": RESULTS / "selection_sweep_lora_a100_d165.json"}
CLUSTERS = ROOT / "manifests" / "negative_clusters.json"
DOSES = (25, 100, 400)
LEVELS = ("item_level_D156", "cluster_level_D164")
FINAL_1 = RESULTS / "damage_d156_r2.json"
FINAL_5 = RESULTS / "damage_d156_r2_fpr5.json"              # 4e (D-152, D-248): the 5% sensitivity arm
DAMAGE_1_SHA256 = "9c6cd0d6ee2700f60626adca034eae45ace6c18f336fcd2cffc306b59e3919f6"   # the 1% record of record
TARGET_5 = 0.05


def selection(doc: dict, pair: str) -> tuple[float, int, dict[str, str]]:
    m = next((x for x in doc["models"] if x["pair"] == pair), None)
    if m is None or not m["selected"]["arm"].startswith("C="):
        raise SystemExit(f"REFUSING: no logistic-C selection for {pair}")
    ins = {k.split(":", 1)[1]: v for k, v in doc["input_cache_sha256"].items() if k.startswith(pair + ":")}
    return float(m["selected"]["arm"].split("=")[1]), int(m["selected"]["layer"]), ins


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Nine-pair table and 4a under D-156 (second run)")
    ap.add_argument("--n-draws", type=int, default=P.N_DRAWS)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--target-fpr", type=float, default=None)
    ap.add_argument("--fpr5", action="store_true",
                    help="4e (D-152, D-248): the registered 5%% arm; every cell is also computed at 1%% and must equal "
                         "results/damage_d156_r2.json exactly (the self-anchor), then the 5%% numbers are written")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.fpr5 and args.target_fpr not in (None, TARGET_5):
        raise SystemExit("REFUSING: --fpr5 fixes the target at 0.05")
    args.target_fpr = TARGET_5 if args.fpr5 else (P.TARGET_FPR if args.target_fpr is None else args.target_fpr)
    final = FINAL_5 if args.fpr5 else FINAL_1
    out_path = Path(args.out).resolve() if args.out else final.resolve()
    if out_path.exists():
        raise SystemExit(f"REFUSING: {out_path} exists; results are never overwritten")
    registered = (args.n_draws == P.N_DRAWS and list(args.seeds) == [0, 1, 2, 3, 4]
                  and args.target_fpr == (TARGET_5 if args.fpr5 else P.TARGET_FPR))
    if out_path in (FINAL_1.resolve(), FINAL_5.resolve()) and (not registered or out_path != final.resolve()):
        raise SystemExit("REFUSING: only the registered parameters of this mode may write its final name")
    record_1: dict[tuple[str, str, str], dict[str, Any]] = {}
    if args.fpr5:
        rb = FINAL_1.read_bytes()
        if sha256_bytes(rb) != DAMAGE_1_SHA256:
            raise SystemExit(f"REFUSING: {FINAL_1.name} is not the 1% record of record")
        record_1 = {(c["scope"], c["cell"], c["probe_set"]): c for c in json.loads(rb.decode("utf-8"))["cells"]}
    n_self_anchor = 0
    code0 = P.code_block(CODE_FILES)
    fixed_inputs = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in [*SELECTION.values(), CLUSTERS]}
    sel_docs = {k: json.load(open(p, encoding="utf-8")) for k, p in SELECTION.items()}
    for k, d in sel_docs.items():
        if list(d["config"]["seeds"]) != list(args.seeds) or float(d["config"]["holdout"]) != P.HOLDOUT:
            raise SystemExit(f"REFUSING: {SELECTION[k].name} seeds/holdout differ from this run's")
    clusters = json.load(open(CLUSTERS, encoding="utf-8"))["axes"]

    # --- cell specifications ----------------------------------------------------------
    specs: list[dict[str, Any]] = []
    for stem in POP.UPDATE_PAIRS_REPAIR:
        C, L, sel_in = selection(sel_docs["nine-pair table"], stem)
        specs.append({"scope": "nine-pair table", "cell": stem, "run": REG.run(stem), "C": C, "L": L,
                      "sel_in": sel_in, "selection_file": "results/selection_sweep_d165.json"})
    C4, L4, sel4 = selection(sel_docs["4a"], "repair_loraA100_mistral_s25")
    for arm, stems in POP.LORA_ARMS.items():
        for run, dose in zip(stems, DOSES):
            specs.append({"scope": "4a", "cell": run, "arm": arm, "dose": dose, "run": REG.run(f"repair_{run}"),
                          "C": C4, "L": L4, "sel_in": sel4, "selection_file": "results/selection_sweep_lora_a100_d165.json"})

    input_sha: dict[str, str] = {}
    input_source: dict[str, str] = {}
    all_recs: dict[str, dict[str, Any]] = {}
    cell_recs: dict[str, dict[str, dict[str, Any]]] = {}
    probe_cache: dict[tuple, dict[str, list[P.Probe]]] = {}
    axis_ids: dict[str, tuple] = {}
    draws: dict[str, dict[str, np.ndarray]] = {}
    cells: list[dict[str, Any]] = []
    arrays: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    frozen_vals: dict[tuple[str, str, str], dict[str, float]] = {}

    for s in specs:
        cfg = s["run"]
        axis, k = cfg["axis"], f"layer_{s['L']}"
        z, recs = {}, {}
        for side in ("pre", "post"):
            for part in ("probe_train", "threshold_calib", "eval"):
                if side == "post" and part == "probe_train":
                    continue
                z[(side, part)], r = REG.load(cfg[f"{side}_repo"], axis, part)   # bytes checked there
                recs[f"{side}/{part}"] = r
                all_recs[f"{s['cell']}:{side}/{part}"] = r
                input_sha[f"{s['cell']}:{side}/{part}"] = r["sha256"]
                input_source[f"{s['cell']}:{side}/{part}"] = f"{r['file']} ({r['expected_from']})"
        cell_recs[s["cell"]] = recs
        bad = [f"selection {key}" for key in ("pre/probe_train", "pre/threshold_calib")
               if s["sel_in"].get(key) != input_sha[f"{s['cell']}:{key}"]]
        if bad:
            raise SystemExit(f"REFUSING: {s['cell']}: the selection was made on other bytes: {bad}")
        for part in ("threshold_calib", "eval"):
            a, b = z[("pre", part)], z[("post", part)]
            if not np.array_equal(a["item_ids"], b["item_ids"]) or not np.array_equal(a["labels"], b["labels"]):
                raise SystemExit(f"REFUSING: {s['cell']} {part} items or labels differ pre/post")
        for zz in z.values():
            if "nonfinite" in zz.files and zz["nonfinite"].any():
                raise SystemExit(f"REFUSING: {s['cell']} has a non-finite item in an input cache")
        y_tr = z[("pre", "probe_train")]["labels"].astype(bool)
        ncal = ~z[("pre", "threshold_calib")]["labels"].astype(bool)
        y_ev = z[("pre", "eval")]["labels"].astype(bool)
        ids = (tuple(map(str, z[("pre", "threshold_calib")]["item_ids"][ncal])), tuple(map(str, z[("pre", "eval")]["item_ids"][~y_ev])))
        if axis not in axis_ids:
            axis_ids[axis] = ids
            pool = list(ids[0]) + list(ids[1])
            cl = clusters[axis]
            if cl["n_pooled_negatives"] != len(pool) or cl["n_calibration"] != len(ids[0]) or \
                    cl["pool_item_ids_sha256"] != sha256_bytes("\n".join(pool).encode("utf-8")):
                raise SystemExit(f"REFUSING: manifests/negative_clusters.json does not describe the {axis} pool "
                                 f"(counts or ordered item-id hash differ)")
            unit = P.cluster_units(pool, cl["clusters"])
            draws[axis] = {LEVELS[0]: P.repartition_draws(len(ids[0]), len(ids[1]), args.n_draws, P.AXIS_DRAW_SEED[axis]),
                           LEVELS[1]: P.cluster_repartition_draws(unit, len(ids[0]), args.n_draws,
                                                                  P.AXIS_CLUSTER_DRAW_SEED[axis])}
        elif axis_ids[axis] != ids:
            raise SystemExit(f"REFUSING: {s['cell']} pooled negatives differ from other cells on {axis}")

        pkey = (cfg["pre_repo"], cfg["pre_rev"], axis, s["L"], s["C"])
        if pkey not in probe_cache:
            X_tr = z[("pre", "probe_train")][k]
            sets: dict[str, list[P.Probe]] = {"selected": [], "C=1": []}
            for seed in args.seeds:
                _, fit = P.d117_split(len(y_tr), seed)
                sets["selected"].append(P.fit_probe(X_tr[fit], y_tr[fit], seed, s["C"], "d117_permutation"))
                fo = np.sort(fit)
                sets["C=1"].append(P.fit_probe(X_tr[fo], y_tr[fo], seed, 1.0, "file_order"))
            probe_cache[pkey] = sets
        X = {(side, part): z[(side, part)][k] for side in ("pre", "post") for part in ("threshold_calib", "eval")}
        for pset, probes in probe_cache[pkey].items():
            sc = lambda side, part, m: np.column_stack([q.score(X[(side, part)][m]) for q in probes])  # noqa: E731
            scores = (sc("pre", "threshold_calib", ncal), sc("pre", "eval", ~y_ev), sc("post", "threshold_calib", ncal),
                      sc("post", "eval", ~y_ev), sc("pre", "eval", y_ev), sc("post", "eval", y_ev))
            reps = {}
            for lvl in LEVELS:
                reps[lvl], arr = P.repartition_estimate(*scores, draws[axis][lvl], args.target_fpr)
                arrays[(s["cell"], pset, lvl)] = arr
                frozen_vals[(s["cell"], pset, lvl)] = {m: reps[lvl][m]["frozen"] for m in ("paired_ratio", "delta_tpr")}
            if args.fpr5 and registered:
                # D-248 self-anchor: the same cell at 1% must be the 1% record's, exactly, at both levels
                rec = record_1.get((s["scope"], s["cell"], pset))
                if rec is None:
                    raise SystemExit(f"REFUSING: {FINAL_1.name} has no cell {s['scope']}/{s['cell']}/{pset}")
                for lvl in LEVELS:
                    rep1, _ = P.repartition_estimate(*scores, draws[axis][lvl], P.TARGET_FPR)
                    rep1 = json.loads(json.dumps(rep1, default=str))
                    want = {m: rec.get(m) for m in rep1} if lvl == LEVELS[0] else rec[LEVELS[1]]
                    # canonical text, so a NaN compares equal to itself
                    if json.dumps(rep1, sort_keys=True) != json.dumps(json.loads(json.dumps(want, default=str)), sort_keys=True):
                        raise SystemExit(f"REFUSING: {s['cell']} {pset} {lvl}: the 1% twin is not {FINAL_1.name}'s (D-248)")
                n_self_anchor += 1
            cells.append({"scope": s["scope"], "cell": s["cell"], "arm": s.get("arm"), "dose": s.get("dose"),
                          "axis": axis, "layer": s["L"], "probe_set": pset,
                          "probe_C": s["C"] if pset == "selected" else 1.0,
                          "fit_row_order": "d117_permutation" if pset == "selected" else "file_order",
                          "selection_file": s["selection_file"], "input_verification": REG.verification(recs),
                          **reps[LEVELS[0]], LEVELS[1]: reps[LEVELS[1]],
                          "probe_fit_diagnostics": [q.diag for q in probes]})
            r, rc = reps[LEVELS[0]]["paired_ratio"], reps[LEVELS[1]]["paired_ratio"]
            print(f"  [{s['scope']:<15}] {s['cell']:<32} {pset:<8} paired ratio {r['mean']:7.3f} "
                  f"[{r['band95'][0]:.3f}, {r['band95'][1]:.3f}] frozen {r['frozen']:7.3f} (pctl {r['frozen_percentile']:5.1f}) "
                  f"| cluster-level {rc['mean']:7.3f} [{rc['band95'][0]:.3f}, {rc['band95'][1]:.3f}]", flush=True)

    # --- 4a between-arm gaps, formed within each draw, at both levels -------------------
    between = {lvl: [] for lvl in LEVELS}
    for lvl in LEVELS:
        for pset in ("selected", "C=1"):
            for arm, stems in POP.LORA_ARMS.items():
                if arm.startswith("general"):
                    continue
                for run, dose in zip(stems, DOSES):
                    g = POP.LORA_ARMS["general (alpaca)"][DOSES.index(dose)]
                    both = {f"{c}:{k}": r for c in (run, g) for k, r in cell_recs[c].items()}
                    row = {"probe_set": pset, "safety_arm": arm, "safety_run": run, "general_run": g, "dose": dose,
                           "input_verification": REG.verification(both)}
                    for key, arr_key, f in (("log_damage_gap", "paired_ratio", np.log), ("delta_tpr_gap", "delta_tpr", lambda v: v)):
                        per_draw = f(arrays[(run, pset, lvl)][arr_key]) - f(arrays[(g, pset, lvl)][arr_key])
                        fz = f(np.array(frozen_vals[(run, pset, lvl)][arr_key])) - f(np.array(frozen_vals[(g, pset, lvl)][arr_key]))
                        row[key] = {**P.band_summary(per_draw, float(fz)),
                                    "share_of_draws_above_0": float((per_draw > 0).mean())}
                    between[lvl].append(row)

    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"] or any(sha256_file(ROOT / f) != h for f, h in fixed_inputs.items()):
        raise SystemExit("REFUSING to write: source, a selection file or the cluster manifest changed during the run")
    if args.fpr5 and registered and n_self_anchor != len(cells):
        raise SystemExit(f"REFUSING to write: {n_self_anchor} self-anchored cells of {len(cells)}")
    payload = {**provenance_block("damage_d156"),
               "preregistration": "D-156, D-164, D-165" + ("; 4e: D-152, D-248 (the 5% sensitivity arm)" if args.fpr5 else ""),
               **({"fpr5_self_anchor": {"file": str(FINAL_1.relative_to(ROOT)).replace("\\", "/"), "sha256": DAMAGE_1_SHA256,
                                        "n_cells_identical_at_1pct": n_self_anchor,
                                        "kind": "every cell recomputed at 1% equals the 1% record at both levels (D-248)"}}
                  if args.fpr5 else {}),
               "code": code0,
               "config": {"n_draws": args.n_draws, "axis_draw_seeds": P.AXIS_DRAW_SEED,
                          "axis_cluster_draw_seeds": P.AXIS_CLUSTER_DRAW_SEED, "seeds": args.seeds,
                          "target_fpr": args.target_fpr, "split": f"D-117 (holdout {P.HOLDOUT})",
                          "pooled_negatives": {ax: {"n_calibration": len(v[0]), "n_evaluation": len(v[1])} for ax, v in axis_ids.items()},
                          "positives": "eval positives, fixed",
                          "levels": {LEVELS[0]: "item-level re-partitions (D-156)",
                                     LEVELS[1]: "cluster-level re-partitions, no near-duplicate cluster straddles (D-164). "
                                                "Its 'frozen' value is the frozen split, which is NOT a cluster-level draw "
                                                "(it splits 36 of 67 Aegis and 7 of 11 WildGuardMix clusters), so its "
                                                "frozen_percentile places a split this distribution cannot produce; the "
                                                "two levels use independent seeds, so their means differ by Monte Carlo "
                                                "noise of about sqrt(2) * sd_over_draws / sqrt(n_draws)"},
                          "paper_eligibility": "decided per cell: a cell (or between-arm row) whose input_verification "
                                               "carries the D-170 flag waits for D-168; the file-level flag only says "
                                               "that some cell does (D-172)",
                          **{f"{c}:{side}_{kk}": sp["run"][f"{side}_{kk}"] for sp in specs for c in (sp["cell"],)
                             for side in ("pre", "post") for kk in ("repo", "rev")},
                          "runs": {sp["cell"]: sp["run"] for sp in specs}, "inputs_from": "src/input_registry.py",
                          "input_registry_sources_sha256": REG.sources_sha256(),
                          "fixed_inputs_sha256": fixed_inputs},
               "population": {"declared_in": "src/populations.py",
                              "groups": {"nine-pair table": "UPDATE_PAIRS_REPAIR", "4a": "LORA_ARMS"}},
               "supersedes": [] if args.fpr5 else ["results/damage_d156.json (first run: superseded selection and inputs, D-165)"],
               "input_verification": REG.verification(all_recs),
               "input_cache_sha256": input_sha, "input_cache_source": input_source, "cells": cells,
               "between_arms_4a": between[LEVELS[0]], "between_arms_4a_cluster_level_D164": between[LEVELS[1]]}
    digest = write_results(out_path, payload, exclusive=True)
    rel = str(out_path.relative_to(ROOT)).replace("\\", "/") if out_path.is_relative_to(ROOT) else out_path.name
    print(f"\nWritten: {rel}\nsha256:  {digest}")

    fm = lambda d, f="{:.3f}": f"{f.format(d['mean'])} [{f.format(d['band95'][0])}, {f.format(d['band95'][1])}] fz {f.format(d['frozen'])} ({d['frozen_percentile']:.0f})"  # noqa: E731
    print("\nD-156 TABLE: mean [95% band over re-partitions], fz = frozen split (its percentile); "
          "McNemar on the frozen split, max p over seeds (minimum achievable p); share of draws p<0.05; "
          "cluster-level (D-164) mean [band] beside")
    for c in cells:
        sg, cl = c["frozen_split_significance"], c[LEVELS[1]]
        print(f"  [{c['scope']:<15}] {c['cell'][:30]:<31} {c['probe_set']:<8} L{c['layer']:<3} ratio {fm(c['paired_ratio'])} | "
              f"cluster {cl['paired_ratio']['mean']:.3f} [{cl['paired_ratio']['band95'][0]:.3f}, {cl['paired_ratio']['band95'][1]:.3f}] | "
              f"to target {fm(c['ratio_to_target'])} | floor {fm(c['floor_ratio_to_target'])} | "
              f"TPR {c['tpr_pre']['mean']:.3f}->{c['tpr_post']['mean']:.3f} dTPR {fm(c['delta_tpr'], '{:+.4f}')} | "
              f"p_neg {sg['mcnemar_neg_p_max']:.1e} (min {sg['mcnemar_neg_min_achievable_p']:.1e}) share {c['share_of_draws_mcnemar_neg_p_max_below_0_05']:.2f} | "
              f"p_pos {sg['mcnemar_pos_p_max']:.1e} (min {sg['mcnemar_pos_min_achievable_p']:.1e})")
    for lvl in LEVELS:
        print(f"\n4a BETWEEN ARMS (safety minus general, matched dose), per draw, {lvl}:")
        for b in between[lvl]:
            print(f"  [{b['probe_set']:<8}] {b['safety_run']:<24} s{b['dose']:<4} log damage gap {fm(b['log_damage_gap'])} "
                  f"share>0 {b['log_damage_gap']['share_of_draws_above_0']:.3f} | dTPR gap {fm(b['delta_tpr_gap'], '{:+.4f}')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
