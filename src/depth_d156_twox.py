"""Depth under D-156, addition 2 of D-191: the null for the 2x criterion under no update, and the
realised fractions beside it.

Per D-156 draw (item level; D-164 cluster level beside), the fraction of layer cells whose
no-update ratio FPR_pre / target (the floor) is >= 2, the fraction <= 0.5, and the fraction outside
(0.5, 2) (D-085's calibration-failure definition, their union); the same three fractions of the
realised ratio to target FPR_post / target under the update. Mean and 95% band over the draws, the
frozen split's fraction beside; per pair, pooled over the nine pairs of UPDATE_PAIRS_REPAIR, pooled
over all ten, and pooled over the pairs with no flagged cell; selected C and C=1.

DATA GAP (D-191): the depth files keep each cell's floor only as a band summary, and the 4b draw
arrays keep log FPR_post per draw but not FPR_pre. So the per-draw floor is RECONSTRUCTED here,
recomputing ONLY the pre side through the shared code path: the D-117 split, paired.fit_probe, the
per-seed threshold on each draw's calibration part, FPR_pre on its evaluation part (floored at
1/n, seed mean) - the floor_ratio_to_target of paired.operating_point. Every cell must reproduce its
stored floor band summary (mean, band, frozen value, percentile) in results/depth_d156_<pair>.json
exactly, or this refuses: a reconstruction of stored numbers, not a new estimate.
The realised side (FPR_post at the same thresholds) is reconstructed the same way for every pair and
must reproduce the stored ratio-to-target band exactly. For the nine pairs it must ALSO equal, draw
by draw, the integer flag count the 4b arrays carry (rint(exp(log_fpr_post) x n x seeds), from
results/twomoment_<pair>_draws.npz, hash-checked against manifests/draw_arrays.sha256): D-191's
named source, read as the exact integer it encodes rather than through exp(log x) / target, which
misses the stored ratio by a bit in most cells (review 2026-09-26; D-195).
The criteria are judged on exact integers: the seed sum of floored flag counts K against
2 x target x n x seeds (200 here) and 0.5 x ... (50), so no float lands on the wrong side. Each row
records how many cell-draws sit exactly on a cut and how many of them a float comparison of the
stored ratio would put on the other side.
Writes results/depth_d156_twox.json and results/depth_d156_twox_counts.npz (the per-draw K of every
cell, so any other cut or union can be derived later); neither is ever overwritten.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import input_registry as REG  # noqa: E402
import paired as P  # noqa: E402
import populations as POP  # noqa: E402
from provenance import (  # noqa: E402
    commit_staged, ensure_utf8_stdout, provenance_block, reissue_payload, resolve_cache, sha256_bytes, sha256_file,
    stage_bytes, write_results,
)
# D-300: the record --r2 re-issues (it names no repo ID or revision for its cache inputs, D-297 (1)), and its counts file
FIRST_RECORD = ("results/depth_d156_twox.json", "1902f53c42ce27cadceea3deca3d302395269ac767bb4087229e94bc9496471f")
FIRST_COUNTS = ("results/depth_d156_twox_counts.npz", "34a04a71776aa85c5e78e6bc7f766d563f6aba673d7646b96efb760805565646")

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CODE_FILES = ["src/depth_d156_twox.py", "src/input_registry.py", "src/paired.py", "src/populations.py",
              "src/provenance.py", "src/superseded_provenance.py"]
CLUSTERS = ROOT / "manifests" / "negative_clusters.json"
DRAW_MANIFEST = ROOT / "manifests" / "draw_arrays.sha256"
SUMMARY = RESULTS / "depth_d156_summary.json"
LEVELS = ("item_level_D156", "cluster_level_D164")
LEVEL_SUFFIX_4B = {LEVELS[0]: "", LEVELS[1]: "__cluster"}
PROBE_SETS = ("selected", "C=1")
PSET_KEY = {"selected": "selected", "C=1": "C1"}
DENSE = "depth_mistral_v01_v02_dense"
NINE = [f"depth_{s[len('repair_'):]}" for s in POP.UPDATE_PAIRS_REPAIR]
HI, LO = 2.0, 0.5
NAMES = ("null_no_update_share_ge_2x", "null_no_update_share_le_0_5x", "null_no_update_share_outside_0_5x_2x",
         "realised_share_ge_2x", "realised_share_le_0_5x", "realised_share_outside_0_5x_2x")


def draw_manifest() -> dict[str, str]:
    out = {}
    for line in DRAW_MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, f = line.split(maxsplit=1)
            out[f.strip().lstrip("*")] = h
    return out


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Depth addition 2: the no-update null for the 2x criterion (D-191)")
    ap.add_argument("--pairs", nargs="*", default=NINE + [DENSE], choices=NINE + [DENSE])
    ap.add_argument("--out", default=None, help="default results/depth_d156_twox.json (_r2 with --r2)")
    ap.add_argument("--r2", action="store_true",
                    help="re-issue naming each cache input's repo ID and revision; the counts must equal the first "
                         "record's counts file array by array, which the re-issue names instead of writing a new one (D-300)")
    args = ap.parse_args()
    if args.out is None:
        args.out = str(RESULTS / f"depth_d156_twox{'_r2' if args.r2 else ''}.json")
    if len(set(args.pairs)) != len(args.pairs):
        raise SystemExit("REFUSING: a pair is named twice (its cells would count twice)")
    out = Path(args.out)
    npz_out = out.with_name(out.stem + "_counts.npz")
    for p in ((out,) if args.r2 else (out, npz_out)):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists; results are never overwritten")
    final = out.resolve() == (RESULTS / f"depth_d156_twox{'_r2' if args.r2 else ''}.json").resolve()
    threads = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    if final and (sorted(args.pairs) != sorted(NINE + [DENSE]) or any(t != "6" for t in threads.values())):
        raise SystemExit(f"REFUSING the final name: it needs all ten pairs at the 6-thread pin (the stored numbers' pin); {threads}")
    if not final and out.resolve().is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: a test output must not be under results/")
    code0 = P.code_block(CODE_FILES)
    summ_sha = sha256_file(SUMMARY)
    summ_inputs = json.loads(SUMMARY.read_text(encoding="utf-8"))["input_files_sha256"]  # listed (D-297): summ_sha
    cl_bytes = CLUSTERS.read_bytes()
    cl_sha = sha256_bytes(cl_bytes)
    cl_doc = json.loads(cl_bytes.decode("utf-8"))  # not a result (D-297): manifests/negative_clusters.json, hashed
    dm_sha = sha256_file(DRAW_MANIFEST)
    dman = draw_manifest()
    per_pair: dict[str, Any] = {}
    # (pair, pset, level) -> per cell: (layer, K_pre, K_post, floor float, realised float, flagged), arrays over 1 + n_draws
    cellrows: dict[tuple[str, str, str], list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]]] = {}
    shas: dict[str, str] = {}
    caches: dict[str, str] = {}
    const: dict[str, Any] = {}
    for stem in args.pairs:
        dpath = RESULTS / f"depth_d156_{stem[len('depth_'):]}.json"
        dbytes = dpath.read_bytes()
        rel_d = f"results/{dpath.name}"
        shas[rel_d] = sha256_bytes(dbytes)
        if summ_inputs.get(rel_d) != shas[rel_d]:
            raise SystemExit(f"REFUSING: {dpath.name} is not the file depth_d156_summary.json read")
        d = json.loads(dbytes.decode("utf-8"))  # listed (D-297): these bytes' sha256 is shas[rel_d]
        cfg = d["config"]
        if cfg["cluster_manifest"]["sha256"] != cl_sha:
            raise SystemExit(f"REFUSING: {stem}: the cluster manifest is not the one the depth file used")
        if code0["libraries"] != d["code"]["libraries"]:
            raise SystemExit(f"REFUSING: {stem}: library versions {code0['libraries']} differ from the depth run's "
                             f"{d['code']['libraries']}; the reconstruction would not be bitwise")
        reg = REG.run(cfg["registry_pair"])
        axis = cfg["axis"]
        if (reg["pre_repo"], reg["pre_rev"]) != (cfg["pre_repo"], cfg["pre_rev"]):
            raise SystemExit(f"REFUSING: {stem}: the registry's pre model is not the depth file's")
        if (cfg["draw_seed"], cfg["cluster_draw_seed"]) != (P.AXIS_DRAW_SEED[axis], P.AXIS_CLUSTER_DRAW_SEED[axis]):
            raise SystemExit(f"REFUSING: {stem}: the depth file's draw seeds are not paired.py's for axis {axis}")
        data = {part: REG.load(reg["pre_repo"], axis, part) for part in ("probe_train", "threshold_calib", "eval")}
        post = {part: REG.load(reg["post_repo"], axis, part) for part in ("threshold_calib", "eval")}
        for side, recs in (("pre", data), ("post", post)):
            for part, (_, r) in recs.items():
                if r["sha256"] != d["input_cache_sha256"][f"{side}/{part}"]:
                    raise SystemExit(f"REFUSING: {stem} {side}/{part} is not the bytes the depth file read")
                caches[r["file"]] = r["sha256"]
        # the 4b per-draw arrays (nine pairs): the realised side's per-draw anchor
        z4b = None
        if stem in NINE:
            rel4b = f"results/twomoment_repair_{stem[len('depth_'):]}_draws.npz"
            h4b = sha256_file(ROOT / rel4b)
            if dman.get(rel4b) != h4b:
                raise SystemExit(f"REFUSING: {rel4b} is not the file manifests/draw_arrays.sha256 names")
            shas[rel4b] = h4b
            z4b = np.load(ROOT / rel4b, allow_pickle=False)
            idx4b = {(int(li), str(ps)): i for i, (li, ps) in enumerate(zip(z4b["cell_layer"], z4b["cell_probe_set"]))}
            lf4b = {lvl: z4b["log_fpr_post" + LEVEL_SUFFIX_4B[lvl]] for lvl in LEVELS}
        z_tr, z_cal, z_ev = (data[p][0] for p in ("probe_train", "threshold_calib", "eval"))
        y_tr = z_tr["labels"].astype(bool)
        ncal = ~z_cal["labels"].astype(bool)
        neg = ~z_ev["labels"].astype(bool)
        pooled_ids = [str(i) for i in z_cal["item_ids"][ncal]] + [str(i) for i in z_ev["item_ids"][neg]]
        n_cal, n_pool = int(ncal.sum()), len(pooled_ids)
        if sha256_bytes("\n".join(pooled_ids).encode()) != cfg["pooled_negatives"]["item_ids_sha256"]:
            raise SystemExit(f"REFUSING: {stem}: pooled negatives differ from the depth file's")
        cl = cl_doc["axes"][axis]
        draws = {LEVELS[0]: P.repartition_draws(n_cal, n_pool - n_cal, cfg["n_draws"], P.AXIS_DRAW_SEED[axis]),
                 LEVELS[1]: P.cluster_repartition_draws(P.cluster_units(pooled_ids, cl["clusters"]), n_cal, cfg["n_draws"],
                                                        P.AXIS_CLUSTER_DRAW_SEED[axis])}
        parts = {lvl: [(np.arange(n_cal), np.arange(n_cal, n_pool))] + [(dr[:n_cal], dr[n_cal:]) for dr in draws[lvl]] for lvl in LEVELS}
        target = cfg["target_fpr"]
        n_seeds, n_ev = len(cfg["seeds"]), n_pool - n_cal
        hi_k, lo_k = HI * target * n_ev * n_seeds, LO * target * n_ev * n_seeds     # ratio >= 2 <=> K >= hi_k
        if hi_k != round(hi_k) or lo_k != round(lo_k):
            raise SystemExit("REFUSING: the 2x / 0.5x cut is not an integer flag count")
        c_now = {"target_fpr": target, "n_eval_negatives": n_ev, "n_seeds": n_seeds, "hi_k": int(round(hi_k)),
                 "lo_k": int(round(lo_k)), "n_draws": cfg["n_draws"]}
        if const and const != c_now:
            raise SystemExit(f"REFUSING: {stem}: {c_now} differs from the other pairs' {const}; one cut cannot serve both")
        const = c_now
        cells = {(c["layer"], c["probe_set"]): c for c in d["cells"]}
        n_matched = n_anchor = n_exp_log_flips = 0
        for li in cfg["layers"]:
            k = f"layer_{li}"
            # each array read once per layer (an NpzFile access decompresses the whole array again)
            X_tr, X_cal, X_ev = z_tr[k], z_cal[k][ncal], z_ev[k][neg]
            X_cal_post, X_ev_post = post["threshold_calib"][0][k][ncal], post["eval"][0][k][neg]
            for pset in PROBE_SETS:
                probes = []
                for seed in cfg["seeds"]:
                    _, fitrows = P.d117_split(len(y_tr), seed)
                    rows, C, order = (fitrows, cfg["selected_C"], "d117_permutation") if pset == "selected" \
                        else (np.sort(fitrows), 1.0, "file_order")
                    probes.append(P.fit_probe(X_tr[rows], y_tr[rows], seed, C, order))
                S = lambda x: np.column_stack([q.score(x) for q in probes])  # noqa: E731
                pool_pre = np.vstack([S(X_cal), S(X_ev)])
                pool_post = np.vstack([S(X_cal_post), S(X_ev_post)])
                c = cells[(li, pset)]
                for lvl in LEVELS:
                    floor, realised, kpre, kpost = [], [], [], []
                    for cal, ev in parts[lvl]:
                        T = np.array([P.thr_at_fpr(pool_pre[cal, s], target) for s in range(n_seeds)])
                        F_pre, F_post = pool_pre[ev] >= T, pool_post[ev] >= T
                        # exactly as paired.operating_point: floored per-seed FPR, seed mean, / target
                        floor.append(float(P.floored_fpr(F_pre).mean() / target))
                        realised.append(float(P.floored_fpr(F_post).mean() / target))
                        # the criterion on exact integers: the seed sum of floored flag counts
                        kpre.append(int(np.maximum(F_pre.sum(axis=0), 1).sum()))
                        kpost.append(int(np.maximum(F_post.sum(axis=0), 1).sum()))
                    floor, realised = np.array(floor), np.array(realised)
                    kpre, kpost = np.array(kpre, dtype=np.int64), np.array(kpost, dtype=np.int64)
                    stored = (c if lvl == LEVELS[0] else c[lvl])["operating_point"]
                    if json.loads(json.dumps(P.band_summary(floor[1:], floor[0]))) != stored["floor_ratio_to_target"]:  # not a result (D-297)
                        raise SystemExit(f"REFUSING: {stem} L{li} {pset} {lvl}: the reconstructed floor is not the stored band")
                    if json.loads(json.dumps(P.band_summary(realised[1:], realised[0]))) != stored["ratio_to_target"]:  # not a result (D-297)
                        raise SystemExit(f"REFUSING: {stem} L{li} {pset} {lvl}: the reconstructed ratio to target is not the stored band")
                    n_matched += 1
                    if z4b is not None:
                        lf = lf4b[lvl][idx4b[(li, pset)]]
                        k_real = np.exp(lf) * n_ev * n_seeds
                        k4b = np.rint(k_real)
                        if np.max(np.abs(k_real - k4b)) > 1e-6:
                            raise SystemExit(f"REFUSING: {stem} L{li} {pset} {lvl}: the 4b log_fpr_post is not an integer count")
                        if not np.array_equal(k4b.astype(np.int64), kpost):
                            raise SystemExit(f"REFUSING: {stem} L{li} {pset} {lvl}: the reconstructed realised count differs "
                                             f"from the 4b array in {int(np.sum(k4b != kpost))} draw(s)")
                        n_anchor += 1
                        rt = np.exp(lf) / target
                        n_exp_log_flips += int(np.sum((rt >= HI) != (kpost >= hi_k)) + np.sum((rt <= LO) != (kpost <= lo_k)))
                    cellrows.setdefault((stem, pset, lvl), []).append(
                        (li, kpre, kpost, floor, realised, c["input_verification"]["flag"] is not None))
        if z4b is not None:
            z4b.close()
        flagged_cells = sum(c["input_verification"]["flag"] is not None for c in d["cells"])
        per_pair[stem] = {"n_layers": len(cfg["layers"]), "n_cell_levels_reproduced_exactly": n_matched,
                          "n_cells_flagged": flagged_cells, "n_cells": len(d["cells"]),
                          "flag": d["input_verification"]["flag"],
                          "realised_from": "reconstructed at the same thresholds; stored band reproduced exactly"
                                           + ("; per-draw count equal to the 4b array's in every draw" if z4b is not None else
                                              "; no 4b array (not in UPDATE_PAIRS_REPAIR): band check only"),
                          "n_cell_levels_anchored_to_4b_per_draw": n_anchor,
                          "realised_cell_draws_a_4b_exp_log_float_route_would_misclassify": n_exp_log_flips if z4b is not None else None}
        print(f"{stem}: {n_matched} cell x level floors and ratios reproduce their stored bands exactly; "
              f"{n_anchor} anchored per draw to the 4b counts", flush=True)

    unflagged = [s for s in args.pairs if per_pair[s]["n_cells_flagged"] == 0]

    def summarise(stems: list[str], pset: str, lvl: str) -> dict[str, Any]:
        rows = [r for s in stems for r in cellrows[(s, pset, lvl)]]
        Kpre = np.stack([r[1] for r in rows])                                 # (cells, 1 + n_draws)
        Kpost = np.stack([r[2] for r in rows])
        Fl, Re = np.stack([r[3] for r in rows]), np.stack([r[4] for r in rows])
        hk, lk = const["hi_k"], const["lo_k"]
        ind = np.stack([Kpre >= hk, Kpre <= lk, (Kpre >= hk) | (Kpre <= lk),
                        Kpost >= hk, Kpost <= lk, (Kpost >= hk) | (Kpost <= lk)])  # (6, cells, 1 + n_draws)
        f = ind.mean(axis=1)                                                   # fraction of cells, per draw
        boundary = {side: {"cell_draws_on_a_cut": int(np.sum((K == hk) | (K == lk))),
                           "of_which_a_float_comparison_of_the_stored_ratio_misclassifies":
                               int(np.sum((K == hk) & ~(R >= HI)) + np.sum((K == lk) & ~(R <= LO))),
                           "float_vs_integer_disagreements_anywhere":
                               int(np.sum((R >= HI) != (K >= hk)) + np.sum((R <= LO) != (K <= lk)))}
                    for side, K, R in (("null_no_update", Kpre, Fl), ("realised", Kpost, Re))}
        return {"n_cells": int(Kpre.shape[0]), "n_pairs": len(stems),
                "n_cells_flagged": int(sum(r[5] for r in rows)),
                "pairs_with_flagged_cells": sorted(s for s in stems if per_pair[s]["n_cells_flagged"]),
                **{n: P.band_summary(f[i, 1:], f[i, 0]) for i, n in enumerate(NAMES)},
                "boundary_cell_draws": boundary}

    table: dict[str, Any] = {}
    for pset in PROBE_SETS:
        table[pset] = {}
        for lvl in LEVELS:
            table[pset][lvl] = {"per_pair": {s: summarise([s], pset, lvl) for s in args.pairs}}
            nine = [s for s in args.pairs if s in NINE]
            if nine:
                table[pset][lvl]["pooled_nine_pairs"] = summarise(nine, pset, lvl)
            table[pset][lvl]["pooled_all_pairs_run"] = summarise(list(args.pairs), pset, lvl)
            if unflagged:
                table[pset][lvl]["pooled_unflagged_pairs"] = summarise(unflagged, pset, lvl)

    # the per-draw counts of every cell: any other cut or union can be derived from these later
    arrays: dict[str, np.ndarray] = {}
    for pset in PROBE_SETS:
        for lvl in LEVELS:
            rows = [(s, r) for s in args.pairs for r in cellrows[(s, pset, lvl)]]
            key = f"{PSET_KEY[pset]}__{lvl}"
            arrays[f"k_pre__{key}"] = np.stack([r[1] for _, r in rows]).astype(np.int32)
            arrays[f"k_post__{key}"] = np.stack([r[2] for _, r in rows]).astype(np.int32)
            pair_col, layer_col = np.array([s for s, _ in rows]), np.array([r[0] for _, r in rows], dtype=np.int64)
            if "cell_pair" in arrays and not (np.array_equal(arrays["cell_pair"], pair_col)
                                              and np.array_equal(arrays["cell_layer"], layer_col)):
                raise SystemExit("REFUSING: the cell order differs between probe sets or levels")
            arrays["cell_pair"], arrays["cell_layer"] = pair_col, layer_col
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    npz_bytes = buf.getvalue()

    code1 = P.code_block(CODE_FILES)
    changed = [n for n, a, b in (("code", code1["sha256"], code0["sha256"]),
                                 ("cluster manifest", sha256_file(CLUSTERS), cl_sha),
                                 ("draw-array manifest", sha256_file(DRAW_MANIFEST), dm_sha),
                                 ("depth_d156_summary.json", sha256_file(SUMMARY), summ_sha)) if a != b]
    changed += [f for f, h in shas.items() if sha256_file(ROOT / f) != h]
    changed += [f"cache {f}" for f, h in caches.items() if sha256_file(resolve_cache(f, REG.CACHE)) != h]
    if changed:
        raise SystemExit(f"REFUSING to write: changed during the run: {changed}")
    rel = lambda p: str(p.resolve().relative_to(ROOT.resolve())).replace("\\", "/") if p.resolve().is_relative_to(ROOT.resolve()) else str(p)  # noqa: E731
    payload = {**provenance_block("depth_d156_twox"), "preregistration": "D-191 (2); D-195 (choices after review)", "code": code0,
               "criterion": {"high": HI, "low": LO, "quantity": "ratio to target: FPR (seed mean, floored at 1/n) / target",
                             "judged_on": "the integer K = seed sum of max(flag count, 1): ratio >= 2 <=> K >= hi_k, "
                                          "ratio <= 0.5 <=> K <= lo_k (exact; target read as 1/100)",
                             "outside_0_5x_2x": "D-085's calibration-failure definition (the union of the two)", **const},
               "thread_pin": threads,
               "input_files_sha256": shas | {"results/depth_d156_summary.json": summ_sha,
                                             "manifests/negative_clusters.json": cl_sha,
                                             "manifests/draw_arrays.sha256": dm_sha},
               "input_cache_sha256": caches,
               "per_pair": per_pair,
               "input_verification": {"flag": "input verified by pin only, pending D-168",
                                      "n_pairs_flagged": sum(v["flag"] is not None for v in per_pair.values()),
                                      "rule": "rows containing flagged pairs enter no paper table before D-168 (D-170); "
                                              "pooled_unflagged_pairs carries none"},
               "counts_file": {"file": rel(npz_out), "sha256": sha256_bytes(npz_bytes), "bytes": len(npz_bytes),
                               "column_0": "the frozen split; columns 1..n_draws: the re-partitions in draw order",
                               "rows": "cells: cell_pair, cell_layer", "keys": sorted(arrays)},
               "fractions": table}
    if args.r2:
        c_rel, c_sha = FIRST_COUNTS
        if sha256_file(ROOT / c_rel) != c_sha:
            raise SystemExit(f"REFUSING: {c_rel} is not the first record's counts file")
        with np.load(ROOT / c_rel) as z0:
            same = sorted(z0.files) == sorted(arrays) and all(
                z0[k].shape == np.asarray(arrays[k]).shape and z0[k].dtype == np.asarray(arrays[k]).dtype and np.array_equal(
                    z0[k], np.asarray(arrays[k]), equal_nan=z0[k].dtype.kind == "f") for k in arrays)
        if not same:
            raise SystemExit(f"REFUSING to write: the counts differ from {c_rel} (D-300)")
        payload["counts_file"] = {**payload["counts_file"], "file": c_rel, "sha256": c_sha,
                                  "bytes": (ROOT / c_rel).stat().st_size}
        payload = reissue_payload(payload, *FIRST_RECORD)
        payload["reissue_of"]["counts_check"] = {"file": c_rel, "sha256": c_sha, "arrays_equal": sorted(arrays),
                                                 "note": "recomputed array by array; no new counts file is written"}
        digest = write_results(out, payload, exclusive=True)
        print(f"Written: {out}\nsha256:  {digest}")
        return 0
    json_bytes = json.dumps(payload, indent=2, sort_keys=False, default=str).encode("utf-8")
    json.loads(json_bytes.decode("utf-8"))  # not a result (D-297): the payload's own round trip
    # Stage BOTH outputs (written, fsynced, verified) before either final name exists; then link.
    staged: list[Path] = []
    try:
        t_npz, h_npz = stage_bytes(npz_out, npz_bytes)
        staged.append(t_npz)
        t_json, h_json = stage_bytes(out, json_bytes)
        staged.append(t_json)
        commit_staged(t_npz, npz_out, h_npz)
        commit_staged(t_json, out, h_json)
    finally:
        for t in staged:
            if t.exists():
                t.unlink()
    print(f"Written: {npz_out}\nsha256:  {h_npz}\nWritten: {out}\nsha256:  {h_json}")
    for pset in PROBE_SETS:
        r = table[pset][LEVELS[0]].get("pooled_nine_pairs")
        if r:
            print(f"[{pset}] nine pairs, {r['n_cells']} cells: null >=2x {r['null_no_update_share_ge_2x']['mean']:.4f} "
                  f"{r['null_no_update_share_ge_2x']['band95']}; realised >=2x {r['realised_share_ge_2x']['mean']:.4f} "
                  f"{r['realised_share_ge_2x']['band95']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
