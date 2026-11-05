"""D-282 (3), Table 1's TPR columns: the combined item-bootstrap x re-partition interval (D-281 (1)) for TPR pre,
TPR post and delta TPR, beside the D-156 band, at every layer where Table 1 quotes them.

Table 1 is results/depth_d156_summary.json's table_1 (D-180, the file of record D-190), whose rows are
populations.UPDATE_PAIRS_DEPTH: the nine released pairs and the dense Mistral pair. Each row quotes TPR at its
headline layer (the D-117 selected layer) and at the layers of its other D-180 rows (sweep maximum, lowest layer,
two-sided), per probe set, read here from the depth record's own pair_summary. D-284 (4) promised advstrat's pooled
operating point its combined interval "with Table 1", so the two WildGuardMix rows also cover D-209's layers at the
selected C. For each row, probe set and layer, the script re-does depth_d156's work at that layer only: the same
caches through the input registry, the same selection of record, the same D-117 fits, and the same item-level D-156
draws. Then:
  - ANCHOR: the item-level operating point must equal the depth record's cell as canonical JSON text, and the
    per-draw ratio_to_target array must match the cell's per_draw_ratio_to_target_sha256. The draw ORDER is pinned
    too, which the pairing of draw d_r with counts row r needs;
  - ONES CHECK: counts of all ones must give back the D-156 per-draw TPR arrays bit for bit (D-282 (4));
  - SHARED PRE SIDE: rows that share the pre model, axis, C, layer and probe set (the dense pair and Mistral;
    llama3_to_31 and llama3) must give identical TPR-pre replicates.
paired.combined_tpr_replicates then adds the combined interval. Replicate r uses draw d_r = 1 + ((r - 1) mod 1000)
and row r of paired.counts_matrix(n positives, 5000, depth_d156.ITEM_BOOT_SEED["eval_positives"]), the matrix D-282
(2) names, which is a function of n positives. The eval positives must be the same items in the same order in every
row on an axis (checked by an ordered item-id hash), so the matrix pairs by item across rows. The cluster level keeps
its D-156 band only (D-282 (2)). --fpr5 does the same at 5% against the depth _fpr5 records (D-248). No 5% summary
exists, so their own pair_summary gives the layers there, and their fpr5_self_anchor and stage manifest are checked.
It writes results/table1_tpr_d282[_fpr5].json, never overwriting; the depth records are untouched. A test writes
outside results/.
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
import paired as P  # noqa: E402
import populations as POP  # noqa: E402
from provenance import (  # noqa: E402
    ensure_utf8_stdout, provenance_block, reissue_payload, resolve_cache, sha256_bytes, sha256_file, write_results,
)
# D-300: the record --r2 re-issues (it names no repo ID or revision for its cache inputs, D-297 (1))
FIRST_RECORD = ("results/table1_tpr_d282.json", "c0d5cc49c22570dc0f5fe9756b4fe3b7bdb648c7d54f8df6d28a6baae1b6df66")

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CODE_FILES = ["src/table1_tpr_d282.py", *D.CODE_FILES]
SUMMARY = RESULTS / "depth_d156_summary.json"
SUMMARY_OF_RECORD_SHA256 = "efabb2181946617d944c10e9deec3251073da28b84476b98df3c7a4bfb3de3b7"   # D-190
FPR5_DEPTH_MANIFEST = ROOT / "logs" / "fpr5_stage_depth_20260928.sha256"                         # D-277
QUANT = ("tpr_pre", "tpr_post", "delta_tpr")
D209_LAYERS = {"depth_WG_mistral_base2instruct": (32,), "depth_WG_llama3_base2instruct": (24, 26, 28, 30)}
NOTE = ("D-282 (D-281 (1)): combined 95% interval, the 2.5-97.5 percentiles over n_boot replicates. Replicate r is "
        "evaluated at item-level draw d_r = 1 + ((r - 1) mod n_draws) with row r of the axis's item-bootstrap counts "
        "over the eval positives, paired within item (the same counts weight pre and post and every seed). The D-156 "
        "band (partition variation only), its mean and the frozen split's value are copied beside it from the depth "
        "record's cell")


def canon(o: Any) -> str:
    return json.dumps(json.loads(json.dumps(o, default=str)), sort_keys=True)


def rel_path(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/") if p.is_relative_to(ROOT) else p.name


def table1_layers(pair_summary: dict[str, Any], pset: str) -> list[int]:
    """Every layer at which the pair's D-180 rows (item level) quote TPR."""
    out: set[int] = set()

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            if "layer" in o and all(q in o for q in QUANT):
                out.add(int(o["layer"]))
            for v in o.values():
                walk(v)
    walk(pair_summary[pset]["item_level_D156"])
    return sorted(out)


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Table 1's TPR columns with D-282's combined interval")
    ap.add_argument("--fpr5", action="store_true", help="the 5%% Table 1 (D-248), against the depth _fpr5 records")
    ap.add_argument("--pairs", nargs="+", default=None, help="a subset of the rows (a test)")
    ap.add_argument("--n-draws", type=int, default=D.REGISTERED["n_draws"])
    ap.add_argument("--n-boot", type=int, default=D.REGISTERED["n_boot"])
    ap.add_argument("--r2", action="store_true",
                    help="re-issue table1_tpr_d282.json naming each cache input's repo ID and revision, checked equal (D-300)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.r2 and args.fpr5:
        raise SystemExit("REFUSING: --r2 re-issues the 1% record only (D-300)")
    target = D.TARGET_5 if args.fpr5 else D.REGISTERED["target_fpr"]
    seeds = D.REGISTERED["seeds"]
    rows = list(POP.UPDATE_PAIRS_DEPTH)
    summ_bytes = SUMMARY.read_bytes()                                   # the bytes hashed are the bytes used
    summ_sha = sha256_bytes(summ_bytes)
    summ = json.loads(summ_bytes.decode("utf-8"))
    if list(summ["table_1"]) != rows:
        raise SystemExit(f"REFUSING: {SUMMARY.name}'s table_1 rows are not populations.UPDATE_PAIRS_DEPTH")
    run_rows = rows if args.pairs is None else list(args.pairs)
    if not set(run_rows) <= set(rows) or len(set(run_rows)) != len(run_rows):
        raise SystemExit(f"REFUSING: {run_rows} are not distinct Table 1 rows")
    registered = (args.n_draws == D.REGISTERED["n_draws"] and args.n_boot == D.REGISTERED["n_boot"] and run_rows == rows)
    final_path = (RESULTS / f"table1_tpr_d282{'_fpr5' if args.fpr5 else ''}{'_r2' if args.r2 else ''}.json").resolve()
    out_path = Path(args.out).resolve() if args.out else final_path
    final = out_path == final_path
    if out_path.exists():
        raise SystemExit(f"REFUSING: {rel_path(out_path)} exists; results are never overwritten")
    if not final and out_path.is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: a test run (--out) must not write under results/")
    threads = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    if final and (not registered or any(t != "6" for t in threads.values()) or summ_sha != SUMMARY_OF_RECORD_SHA256):
        raise SystemExit(f"REFUSING the final name: registered {registered}, threads {threads}, summary {summ_sha[:16]} "
                         f"(D-190: {SUMMARY_OF_RECORD_SHA256[:16]})")
    fpr5_manifest: dict[str, str] = {}
    if args.fpr5:
        if not FPR5_DEPTH_MANIFEST.is_file():
            raise SystemExit(f"REFUSING: {rel_path(FPR5_DEPTH_MANIFEST)} is not on disk")
        fpr5_manifest = {ln.split()[1]: ln.split()[0] for ln in FPR5_DEPTH_MANIFEST.read_text(encoding="utf-8").splitlines() if ln.strip()}
    code0 = P.code_block(CODE_FILES)
    sel_path = D.SELECTION_OF_RECORD
    sel_sha = sha256_file(sel_path)
    sel_doc = json.loads(sel_path.read_bytes().decode("utf-8"))
    if list(sel_doc["config"]["seeds"]) != seeds or float(sel_doc["config"]["holdout"]) != P.HOLDOUT:
        raise SystemExit("REFUSING: the selection's seeds or holdout differ from the registered ones")
    reg_sources = REG.sources_sha256()
    if D.sources_now() != reg_sources:
        raise SystemExit("REFUSING: the extraction records or pins on disk are not the ones the registry read")

    counts: dict[int, np.ndarray] = {}
    positives_id: dict[str, str] = {}
    shared_pre: dict[tuple, tuple[str, str]] = {}
    depth_files: dict[str, str] = {}
    input_recs: dict[str, dict[str, Any]] = {}
    out_rows: dict[str, Any] = {}
    n_anchor = n_ones = n_shared = 0
    for stem in run_rows:
        reg_stem = D.registry_stem(stem)
        sel_stem = D.SELECTION_STEM.get(stem, reg_stem)
        sel = next((m for m in sel_doc["models"] if m["pair"] == sel_stem), None)
        if sel is None or not sel["selected"]["arm"].startswith("C="):
            raise SystemExit(f"REFUSING: no logistic-C selection for {sel_stem}")
        C_sel, L_sel = float(sel["selected"]["arm"].split("=")[1]), int(sel["selected"]["layer"])
        rel = f"results/{D.out_name(stem)}{'_fpr5' if args.fpr5 else ''}.json"
        if not (ROOT / rel).is_file():
            raise SystemExit(f"REFUSING: {rel} is not on disk")
        rb = (ROOT / rel).read_bytes()
        depth_files[rel] = sha256_bytes(rb)
        ddoc = json.loads(rb.decode("utf-8"))
        dcfg = ddoc["config"]
        if (dcfg["selected_C"], dcfg["selected_layer"], dcfg["selection_sha256"], dcfg["target_fpr"]) != (C_sel, L_sel, sel_sha, target):
            raise SystemExit(f"REFUSING: {rel} was run on another selection or target")
        if args.fpr5:
            sa = ddoc.get("fpr5_self_anchor") or {}
            rel1 = f"results/{D.out_name(stem)}.json"
            if fpr5_manifest.get(rel) != depth_files[rel] or sa.get("n_cells_identical_at_1pct") != len(ddoc["cells"]) or \
                    sa.get("file") != rel1 or sa.get("sha256") != summ["input_files_sha256"].get(rel1):
                raise SystemExit(f"REFUSING: {rel} is not a verified, self-anchored 5% depth record (D-248, D-277)")
        else:
            srow = summ["table_1"][stem]
            # the summary's row is the pair_summary plus its own 'role' label
            if summ["input_files_sha256"].get(rel) != depth_files[rel] or srow["selected_layer"] != L_sel or \
                    any(canon(srow[p].get(kk)) != canon(v) for p in D.PROBE_SETS for kk, v in ddoc["pair_summary"][p].items()):
                raise SystemExit(f"REFUSING: {rel} is not the record {SUMMARY.name}'s Table 1 row was built from")
        dcells = {(c["layer"], c["probe_set"]): c for c in ddoc["cells"]}
        cfg = REG.run(reg_stem)
        axis = cfg["axis"]
        z: dict[tuple[str, str], Any] = {}
        row_recs: dict[str, dict[str, Any]] = {}
        for side in ("pre", "post"):
            for part in ("probe_train", "threshold_calib", "eval"):
                z[(side, part)], r = REG.load(cfg[f"{side}_repo"], axis, part)
                row_recs[f"{side}/{part}"] = r
                input_recs[f"{D.out_name(stem)}:{side}/{part}"] = r
                if ddoc["input_cache_sha256"].get(f"{side}/{part}") != r["sha256"]:
                    raise SystemExit(f"REFUSING: {rel} read other bytes for {side}/{part}")
        verification = REG.verification(row_recs)
        if verification["flag"] != ddoc["input_verification"]["flag"]:
            raise SystemExit(f"REFUSING: {stem}'s D-170 flag differs from {rel}'s")
        y_tr = z[("pre", "probe_train")]["labels"].astype(bool)
        ncal = ~z[("pre", "threshold_calib")]["labels"].astype(bool)
        y_ev = z[("pre", "eval")]["labels"].astype(bool)
        neg, pos = ~y_ev, y_ev
        n_cal, n_ev, n_pos = int(ncal.sum()), int(neg.sum()), int(pos.sum())
        pid = sha256_bytes("\n".join(str(i) for i in z[("pre", "eval")]["item_ids"][pos]).encode("utf-8"))
        if positives_id.setdefault(axis, pid) != pid:
            raise SystemExit(f"REFUSING: {stem}'s eval positives are not the other {axis} rows' (the counts would not pair by item)")
        if n_pos not in counts:
            counts[n_pos] = P.counts_matrix(n_pos, args.n_boot, D.ITEM_BOOT_SEED["eval_positives"])
        CNT = counts[n_pos]
        draws = P.repartition_draws(n_cal, n_ev, args.n_draws, P.AXIS_DRAW_SEED[axis])
        layer_sets = {pset: sorted(set(table1_layers(ddoc["pair_summary"], pset)) | {L_sel} |
                                   (set(D209_LAYERS.get(stem, ())) if pset == "selected" else set()))
                      for pset in D.PROBE_SETS}
        out_rows[stem] = {"registry_pair": reg_stem, "selection_of_pair": sel_stem, "axis": axis, "headline_layer": L_sel,
                          "selected_C": C_sel, "depth_record": rel, "n_positives": n_pos, "eval_positives_item_ids_sha256": pid,
                          **{f"{side}_{kk}": cfg[f"{side}_{kk}"] for side in ("pre", "post") for kk in ("repo", "rev")},
                          "input_verification": verification,
                          "input_cache_source": {k: {"file": r["file"], "expected_from": r["expected_from"], "pin_only": r["pin_only"]}
                                                 for k, r in row_recs.items()},
                          "d209_layers": list(D209_LAYERS.get(stem, ())), "probe_sets": {}}
        for li in sorted(set().union(*layer_sets.values())):
            k = f"layer_{li}"
            X = {(s, p): z[(s, p)][k] for s in ("pre", "post") for p in ("probe_train", "threshold_calib", "eval")}
            for pset in D.PROBE_SETS:
                if li not in layer_sets[pset]:
                    continue
                C, rows_of, order = (C_sel, lambda f: f, "d117_permutation") if pset == "selected" else (1.0, np.sort, "file_order")
                probes = []
                for seed in seeds:
                    _, fit = P.d117_split(len(y_tr), seed)
                    idx = rows_of(fit)
                    probes.append(P.fit_probe(X[("pre", "probe_train")][idx], y_tr[idx], seed, C, order))
                S = lambda side, part, m: np.column_stack([q.score(X[(side, part)][m]) for q in probes])  # noqa: E731
                sc = {"cal_pre": S("pre", "threshold_calib", ncal), "ev_pre": S("pre", "eval", neg),
                      "cal_post": S("post", "threshold_calib", ncal), "ev_post": S("post", "eval", neg),
                      "pos_pre": S("pre", "eval", pos), "pos_post": S("post", "eval", pos)}
                report, arr = P.repartition_estimate(sc["cal_pre"], sc["ev_pre"], sc["cal_post"], sc["ev_post"],
                                                     sc["pos_pre"], sc["pos_post"], draws, target)
                cell = dcells.get((li, pset))
                rt = np.concatenate([[report["ratio_to_target"]["frozen"]], arr["ratio_to_target"]]).astype("<f8")
                if cell is None or canon(report) != canon(cell["operating_point"]) or \
                        sha256_bytes(rt.tobytes()) != cell["frozen_curve_abs_log_ratio_to_target"]["per_draw_ratio_to_target_sha256"]:
                    raise SystemExit(f"REFUSING: {stem} {pset} L{li}: the operating point or its per-draw order is not {rel}'s")
                n_anchor += 1
                pool_pre = np.vstack([sc["cal_pre"], sc["ev_pre"]])
                T_dr = np.array([[P.thr_at_fpr(pool_pre[d[:n_cal], s], target) for s in range(len(seeds))] for d in draws])
                ones = P.combined_tpr_replicates(T_dr, sc["pos_pre"], sc["pos_post"], np.ones((len(draws), n_pos), dtype=np.float32))
                if any(not np.array_equal(ones[q], arr[q]) for q in QUANT):
                    raise SystemExit(f"REFUSING: {stem} {pset} L{li}: counts of ones do not give back the D-156 draws (D-282)")
                n_ones += 1
                reps = P.combined_tpr_replicates(T_dr, sc["pos_pre"], sc["pos_post"], CNT)
                hashes = {q: sha256_bytes(reps[q].astype("<f8").tobytes()) for q in QUANT}
                key = (cfg["pre_repo"], cfg["pre_rev"], axis, C, li, pset)
                if key in shared_pre:
                    if shared_pre[key][1] != hashes["tpr_pre"]:
                        raise SystemExit(f"REFUSING: {stem} {pset} L{li}: TPR pre differs from {shared_pre[key][0]}'s, which "
                                         f"shares its pre side")
                    n_shared += 1
                else:
                    shared_pre[key] = (stem, hashes["tpr_pre"])
                op = cell["operating_point"]
                out_rows[stem]["probe_sets"].setdefault(pset, {"probe_C": C, "layers": {}})["layers"][str(li)] = {
                    "is_headline": li == L_sel, "is_d209_layer": li in D209_LAYERS.get(stem, ()) and pset == "selected",
                    **{q: {"frozen": op[q]["frozen"], "mean_D156": op[q]["mean"], "band95_D156": op[q]["band95"],
                           "combined": P.ci(reps[q]), "replicates_sha256": hashes[q]} for q in QUANT}}
                b = out_rows[stem]["probe_sets"][pset]["layers"][str(li)]["delta_tpr"]
                print(f"  {stem:<32} {pset:<8} L{li:<3} delta TPR frozen {b['frozen']:+.4f}; D-156 band "
                      f"[{b['band95_D156'][0]:+.4f}, {b['band95_D156'][1]:+.4f}]; combined "
                      f"[{b['combined']['ci95'][0]:+.4f}, {b['combined']['ci95'][1]:+.4f}]", flush=True)

    want = sum(len(set(table1_layers(json.loads((ROOT / f).read_bytes().decode("utf-8"))["pair_summary"], p))
                    | {out_rows[s]["headline_layer"]} | (set(D209_LAYERS.get(s, ())) if p == "selected" else set()))
               for s, f in ((s, out_rows[s]["depth_record"]) for s in run_rows) for p in D.PROBE_SETS)
    if n_anchor != want or n_ones != n_anchor:
        raise SystemExit(f"REFUSING to write: {n_anchor} anchored and {n_ones} ones-checked blocks of {want}")
    changed = [n for n, a, b in (("code", P.code_block(CODE_FILES)["sha256"], code0["sha256"]),
                                 ("selection", sha256_file(sel_path), sel_sha), ("summary", sha256_file(SUMMARY), summ_sha),
                                 ("registry sources", D.sources_now(), reg_sources)) if a != b]
    changed += [f for f, h in depth_files.items() if sha256_file(ROOT / f) != h]
    caches = {r["file"]: r["sha256"] for r in input_recs.values()}
    changed += [f"cache {f}" for f, h in caches.items() if sha256_file(resolve_cache(f, REG.CACHE)) != h]
    if changed:
        raise SystemExit(f"REFUSING to write: changed during the run: {changed}")
    payload = {**provenance_block("table1_tpr_d282"),
               "preregistration": "D-281 (1), D-282 (3); D-180 (Table 1), D-190; D-284 (4) (the D-209 cells)" +
                                  ("; D-248 (5%)" if args.fpr5 else ""),
               "code": code0, "registered_parameters": registered, "blas_threads": threads, "note": NOTE,
               "config": {"target_fpr": target, "seeds": seeds, "n_draws": args.n_draws, "n_boot": args.n_boot,
                          "level": "item_level_D156 (the cluster level keeps its D-156 band only, D-282 (2))",
                          "layers": "every layer where the row's D-180 rows quote TPR (the depth record's pair_summary, "
                                    "item level), the headline layer, and for the WildGuardMix rows D-209's layers at "
                                    "the selected C",
                          "draw_seeds": P.AXIS_DRAW_SEED, "draw_of_replicate": "d_r = 1 + ((r - 1) mod n_draws)",
                          "counts": "paired.counts_matrix(n positives, n_boot, depth_d156.ITEM_BOOT_SEED['eval_positives']) "
                                    "(depth_d156's CNT_P, a function of n positives); every row on an axis has the same "
                                    "eval positives in the same order, so the matrix pairs by item across rows",
                          "counts_seed": D.ITEM_BOOT_SEED["eval_positives"],
                          "eval_positives_item_ids_sha256_per_axis": positives_id,
                          "replicates_sha256_of": "float64 little-endian bytes, r = 1..n_boot in order",
                          "selection_file": rel_path(sel_path), "selection_sha256": sel_sha,
                          "table_1": {"file": rel_path(SUMMARY), "sha256": summ_sha, "rows": rows,
                                      "role": "the rows and, at 1%, the check that each depth record is the one Table 1 read"
                                              + ("; no 5% summary exists, so at 5% the layers come from each _fpr5 "
                                                 "record's own pair_summary" if args.fpr5 else "")},
                          **({"fpr5_depth_manifest": {"file": rel_path(FPR5_DEPTH_MANIFEST),
                                                      "sha256": sha256_file(FPR5_DEPTH_MANIFEST)}} if args.fpr5 else {}),
                          "input_registry_sources_sha256": reg_sources},
               "anchor": {"depth_records_sha256": depth_files, "n_blocks_identical": n_anchor,
                          "kind": "each block's item-level operating point equals the depth record's cell as canonical JSON "
                                  "text, and its per-draw ratio_to_target array hashes to the cell's "
                                  "per_draw_ratio_to_target_sha256 (the draw order)"},
               "ones_check": {"n_blocks": n_ones, "kind": "counts of all ones give back the D-156 per-draw TPR arrays bit for bit"},
               "shared_pre_check": {"n_blocks": n_shared, "kind": "rows sharing the pre model, axis, C, layer and probe set "
                                                                  "give identical TPR-pre replicates"},
               "population": {"declared_in": "src/populations.py",
                              "group": "DEPTH_D156_D282_FPR5" if args.fpr5 else "DEPTH_D156_D282", "replaces": None},
               "input_verification": {**REG.verification(input_recs),
                                      "n_rows_flagged": sum(r["input_verification"]["flag"] is not None for r in out_rows.values())},
               "input_cache_sha256": {k: r["sha256"] for k, r in input_recs.items()}, "rows": out_rows}
    if args.r2:
        payload = reissue_payload(payload, *FIRST_RECORD)
    digest = write_results(out_path, payload, exclusive=True)
    print(f"Written: {rel_path(out_path)}\nsha256:  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
