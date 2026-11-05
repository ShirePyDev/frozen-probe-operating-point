"""4c, the label-free health check, under D-156 (D-152 4c; D-169 item 12; D-241), one pair per run.

For every layer cell of the pair's grid, at the D-117 selected C (C=1 as the sensitivity), five seeds, the probes
fitted exactly as depth_d156 fits them, and every D-156 draw (item level; the D-164 cluster level beside):
  REFERENCE SET  the draw's calibration part cal_d: the prompts that set T_d, scored by the pre model and by the
                 post model (D-241 (1)).
  STATISTICS     per seed, then the mean over seeds: z_mean = (m_post - m_pre)/sd_pre; log(sd_post/sd_pre);
                 the GAUSSIAN-predicted ratio g_post/g_pre, g = 1 - Phi((T_d - m)/sd) with that model's cal_d
                 moments (log-survival, so no underflow); the DIRECT ratio r_post/r_pre, r = the share of cal_d
                 scores at or above T_d, floored at 1/n (D-241 (2)).
  REALISED       paired.repartition_estimate on the draw's evaluation part; its report must equal the pair's
                 depth_d156 operating point exactly at every layer, probe set and level (the anchor).
  CLASSES        realised paired ratio >= 2 or <= 0.5 (primary), >= 5 or <= 0.2, and >= 2 only (the high side).
  DETECTORS      a health ratio crossing the same cut as the class it is scored against (Gaussian, direct), and the
                 AUC rule (dAUC <= -0.05 or post AUROC < 0.70) read from the cell's depth_d156 AUROCs (D-241 (4)).
REPORTED per cell: each statistic's D-156 mean and band with the frozen split; the share of draws in each class;
the share of draws each detector flags. Per pair: per draw and probe set and level, the confusion counts of every
detector against every class, pooled over the pair's cells (their D-156 means for pooling across pairs, D-177 (b)).
Per-draw arrays go to results/healthcheck_d156_<pair>_draws.npz, whose sha256 and size the JSON records (D-163 (6)).
The pooled scoring across pairs is src/healthcheck_summary_d156.py. Writes results/healthcheck_d156_<pair>.json
(never overwritten); a test writes outside results/.
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
from scipy.special import logsumexp
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_d156 as D  # noqa: E402
import input_registry as REG  # noqa: E402
import paired as P  # noqa: E402
import populations as POP  # noqa: E402
from provenance import (  # noqa: E402
    commit_staged, ensure_utf8_stdout, provenance_block, resolve_cache, sha256_bytes, sha256_file, stage_bytes,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CODE_FILES = ["src/healthcheck_d156.py", *D.CODE_FILES]
LEVELS, PROBE_SETS, REGISTERED = D.LEVELS, D.PROBE_SETS, D.REGISTERED
CONTROL = "depth_hw_v100_a100"                                  # D-241 (5): the negative check
PAIRS = [*POP.UPDATE_PAIRS_REPAIR, CONTROL, *D.LORA_DEPTH]   # D-278: the nine LoRA runs, by their depth stems
CLASSES = {"two_sided_2x": (2.0, 0.5), "two_sided_5x": (5.0, 0.2), "high_side_2x": (2.0, None)}
DETECTORS = ("gaussian", "direct", "auc_rule")
STATS = ("z_mean", "log_sd_ratio", "gaussian_ratio", "log_gaussian_ratio", "direct_ratio", "realised_paired_ratio")


def core_of(stem: str) -> str:
    return stem[len("repair_"):] if stem.startswith("repair_") else stem[len("depth_"):]


def crosses(x: np.ndarray, cut: tuple[float, float | None]) -> np.ndarray:
    hi, lo = cut
    return (x >= hi) | (x <= lo) if lo is not None else (x >= hi)


def thresholds(S: np.ndarray, target: float) -> np.ndarray:
    return np.array([P.thr_at_fpr(S[:, s], target) for s in range(S.shape[1])])


def health(pre_c: np.ndarray, post_c: np.ndarray, T: np.ndarray) -> dict[str, float]:
    """The reference-set statistics of one partition, per seed then the mean over seeds (D-241 (2))."""
    m0, s0 = pre_c.mean(axis=0), pre_c.std(axis=0)
    m1, s1 = post_c.mean(axis=0), post_c.std(axis=0)
    n = pre_c.shape[0]
    lg = norm.logsf((T - m1) / s1) - norm.logsf((T - m0) / s0)
    r0 = np.maximum((pre_c >= T).mean(axis=0), 1.0 / n)
    r1 = np.maximum((post_c >= T).mean(axis=0), 1.0 / n)
    return {"z_mean": float(((m1 - m0) / s0).mean()), "log_sd_ratio": float(np.log(s1 / s0).mean()),
            "gaussian_ratio": float(np.exp(lg).mean()),
            "log_gaussian_ratio": float(logsumexp(lg) - np.log(lg.size)),     # log of the same mean, never log(0)
            "direct_ratio": float((r1 / r0).mean())}


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="4c under D-156 (D-152 4c, D-241)")
    ap.add_argument("--pair", required=True, choices=PAIRS)
    ap.add_argument("--n-draws", type=int, default=REGISTERED["n_draws"])
    ap.add_argument("--target-fpr", type=float, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=REGISTERED["seeds"])
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="TESTING: a subset of layers")
    ap.add_argument("--fpr5", action="store_true",
                    help="4e (D-152, D-248): the registered 5%% arm, written as healthcheck_d156_<pair>_fpr5; anchored to "
                         "the 5%% depth file, and every cell's 1%% twin must equal the 1%% depth record exactly")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.fpr5 and args.target_fpr not in (None, 0.05):
        raise SystemExit("REFUSING: --fpr5 fixes the target at 0.05")
    args.target_fpr = 0.05 if args.fpr5 else (REGISTERED["target_fpr"] if args.target_fpr is None else args.target_fpr)
    core = core_of(args.pair)
    reg_stem = D.registry_stem(f"depth_{core}")
    lora = f"depth_{core}" in D.LORA_DEPTH
    if lora and args.fpr5:
        raise SystemExit("REFUSING: 4e has no 5% arm for the LoRA runs (D-278)")
    # D-224, D-278: the LoRA runs take the LoRA A100 selection, as their depth_d156 files did
    sel_path = D.SELECTION_OF_RECORD_LORA if lora else D.SELECTION_OF_RECORD
    sel_stem = D.SELECTION_STEM.get(f"depth_{core}", reg_stem)
    anchor_on = (args.n_draws == REGISTERED["n_draws"] and args.target_fpr == (0.05 if args.fpr5 else REGISTERED["target_fpr"])
                 and list(args.seeds) == REGISTERED["seeds"])
    registered = anchor_on and args.layers is None
    sfx = "_fpr5" if args.fpr5 else ""
    final_path = (RESULTS / f"healthcheck_d156_{core}{sfx}.json").resolve()
    out_path = Path(args.out).resolve() if args.out else final_path
    npz_path = out_path.with_name(out_path.stem + "_draws.npz")
    final = out_path == final_path
    for p in (out_path, npz_path):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists; results are never overwritten")
    if not final and out_path.is_relative_to(RESULTS.resolve()):
        raise SystemExit("REFUSING: a test run (--out) must not write under results/")
    threads = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    if final and (not registered or any(t != "6" for t in threads.values())):
        raise SystemExit(f"REFUSING the final name: registered parameters {registered}, threads {threads}")
    code0 = P.code_block(CODE_FILES)
    sel_sha, cl_sha = sha256_file(sel_path), sha256_file(D.CLUSTERS)
    sel_doc = json.load(open(sel_path, encoding="utf-8"))
    sel = next((m for m in sel_doc["models"] if m["pair"] == sel_stem), None)
    if sel is None or not sel["selected"]["arm"].startswith("C="):
        raise SystemExit(f"REFUSING: no logistic-C selection for {sel_stem}")
    C_sel, L_sel = float(sel["selected"]["arm"].split("=")[1]), int(sel["selected"]["layer"])
    depth_rel = f"results/depth_d156_{core}{sfx}.json"
    depth_sha = sha256_file(ROOT / depth_rel)
    ddoc = json.load(open(ROOT / depth_rel, encoding="utf-8"))
    dcfg = ddoc["config"]
    dcells1: dict[tuple[int, str], dict[str, Any]] = {}
    depth1_rel, depth1_sha = f"results/depth_d156_{core}.json", None
    if args.fpr5:
        # D-248: the 5% depth file is complete and self-anchored; the 1% depth record anchors the 1% twin
        if (ddoc.get("fpr5_self_anchor") or {}).get("n_cells_identical_at_1pct") != len(ddoc["cells"]):
            raise SystemExit(f"REFUSING: {depth_rel} is not a complete, self-anchored 5% depth file (D-248)")
        d1b = (ROOT / depth1_rel).read_bytes()
        depth1_sha = sha256_bytes(d1b)
        if depth1_sha != ddoc["fpr5_self_anchor"]["sha256"]:
            raise SystemExit(f"REFUSING: {depth1_rel} is not the 1% record {depth_rel} was anchored to")
        dcells1 = {(c["layer"], c["probe_set"]): c for c in json.loads(d1b.decode("utf-8"))["cells"]}
    n_self_anchor = 0
    if (dcfg["selected_C"], dcfg["selected_layer"], dcfg["selection_sha256"]) != (C_sel, L_sel, sel_sha):
        raise SystemExit(f"REFUSING: {depth_rel} was run on another selection")
    for k_, want_ in (("registry_pair", reg_stem), ("selection_of_pair", sel_stem),
                      ("selection_file", str(sel_path.relative_to(ROOT)).replace("\\", "/"))):
        if k_ in dcfg and dcfg[k_] != want_:
            raise SystemExit(f"REFUSING: {depth_rel} records {k_} {dcfg[k_]!r}, this run resolved {want_!r}")
    dcells = {(c["layer"], c["probe_set"]): c for c in ddoc["cells"]}

    cfg = REG.run(reg_stem)
    axis = cfg["axis"]
    data: dict[str, dict[str, Any]] = {"pre": {}, "post": {}}
    input_sha: dict[str, str] = {}
    input_recs: dict[str, dict[str, Any]] = {}
    for side in data:
        for part in ("probe_train", "threshold_calib", "eval"):
            z, r = REG.load(cfg[f"{side}_repo"], axis, part)
            data[side][part] = z
            input_sha[f"{side}/{part}"], input_recs[f"{side}/{part}"] = r["sha256"], r
    reg_sources = REG.sources_sha256()
    if D.sources_now() != reg_sources:
        raise SystemExit("REFUSING: the extraction records or pins on disk are not the ones the registry read")
    if any(ddoc["input_cache_sha256"].get(k) != h for k, h in input_sha.items()):
        raise SystemExit(f"REFUSING: {depth_rel} read other caches")
    verification = REG.verification(input_recs)
    for part in ("probe_train", "threshold_calib", "eval"):
        a, b = data["pre"][part], data["post"][part]
        if not np.array_equal(a["item_ids"], b["item_ids"]) or not np.array_equal(a["labels"], b["labels"]):
            raise SystemExit(f"REFUSING: {part} items or labels differ between pre and post")
        for z in (a, b):
            if "nonfinite" in z.files and z["nonfinite"].any():
                raise SystemExit(f"REFUSING: {part} has non-finite items")
    grids = [sorted(int(k.split("_")[1]) for k in z.files if k.startswith("layer_")) for s in data for z in data[s].values()]
    if any(g != grids[0] for g in grids):
        raise SystemExit("REFUSING: the input caches have different layer grids")
    layers = [li for li in grids[0] if args.layers is None or li in args.layers]

    y_tr = data["pre"]["probe_train"]["labels"].astype(bool)
    y_cal = data["pre"]["threshold_calib"]["labels"].astype(bool)
    y_ev = data["pre"]["eval"]["labels"].astype(bool)
    ncal, neg, pos = ~y_cal, ~y_ev, y_ev
    pooled_ids = [str(i) for i in data["pre"]["threshold_calib"]["item_ids"][ncal]] + \
                 [str(i) for i in data["pre"]["eval"]["item_ids"][neg]]
    n_cal, n_pool = int(ncal.sum()), len(pooled_ids)
    cl = json.load(open(D.CLUSTERS, encoding="utf-8"))["axes"][axis]
    if cl["n_pooled_negatives"] != n_pool or cl["n_calibration"] != n_cal or \
            cl["pool_item_ids_sha256"] != sha256_bytes("\n".join(pooled_ids).encode("utf-8")):
        raise SystemExit(f"REFUSING: manifests/negative_clusters.json does not describe the {axis} pool")
    if {str(i) for i in data["pre"]["probe_train"]["item_ids"]} & set(pooled_ids):
        raise SystemExit("REFUSING: a probe_train item is in the pooled negatives (F-045)")
    frozen_part = (np.arange(n_cal), np.arange(n_cal, n_pool))
    draws = {LEVELS[0]: P.repartition_draws(n_cal, n_pool - n_cal, args.n_draws, P.AXIS_DRAW_SEED[axis]),
             LEVELS[1]: P.cluster_repartition_draws(P.cluster_units(pooled_ids, cl["clusters"]), n_cal,
                                                    args.n_draws, P.AXIS_CLUSTER_DRAW_SEED[axis])}
    parts = {lvl: [frozen_part] + [(d[:n_cal], d[n_cal:]) for d in dr] for lvl, dr in draws.items()}
    print(f"4c {args.pair} ({cfg['pre_repo']} -> {cfg['post_repo']}, {axis}): layers {layers}; selected C {C_sel:g}; "
          f"control {args.pair == CONTROL}; D-170 flag {verification['flag']}; registered {registered}", flush=True)

    cells: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    confusion: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    n_anchor = 0
    for li in layers:
        k = f"layer_{li}"
        X = {s: {p: data[s][p][k] for p in ("probe_train", "threshold_calib", "eval")} for s in data}
        for pset in PROBE_SETS:
            probes = []
            for seed in args.seeds:
                _, fit = P.d117_split(len(y_tr), seed)
                rows, C, order = (fit, C_sel, "d117_permutation") if pset == "selected" else (np.sort(fit), 1.0, "file_order")
                probes.append(P.fit_probe(X["pre"]["probe_train"][rows], y_tr[rows], seed, C, order))
            S = lambda x: np.column_stack([q.score(x) for q in probes])  # noqa: E731
            pre_pool = np.vstack([S(X["pre"]["threshold_calib"][ncal]), S(X["pre"]["eval"][neg])])
            post_pool = np.vstack([S(X["post"]["threshold_calib"][ncal]), S(X["post"]["eval"][neg])])
            pos_pre, pos_post = S(X["pre"]["eval"][pos]), S(X["post"]["eval"][pos])
            dc = dcells[(li, pset)]
            au = dc["auroc"]
            d_auc = au["frozen"]["mean_over_seeds"] - au["pre"]["mean_over_seeds"]
            auc_flag = bool(d_auc <= -0.05 or au["frozen"]["mean_over_seeds"] < 0.70)
            cblk: dict[str, Any] = {"layer": li, "probe_set": pset, "probe_C": C_sel if pset == "selected" else 1.0,
                                    "is_selected_layer": li == L_sel, "input_verification": verification,
                                    "auc_rule": {"auroc_pre": au["pre"]["mean_over_seeds"], "auroc_post_frozen": au["frozen"]["mean_over_seeds"],
                                                 "d_auroc": d_auc, "flag": auc_flag, "source": f"{depth_rel} (frozen eval split)"}}
            for lvl in LEVELS:
                rep, arr = P.repartition_estimate(pre_pool[:n_cal], pre_pool[n_cal:], post_pool[:n_cal], post_pool[n_cal:],
                                                  pos_pre, pos_post, draws[lvl], args.target_fpr)
                want = dc["operating_point"] if lvl == LEVELS[0] else dc[LEVELS[1]]["operating_point"]
                if anchor_on and json.loads(json.dumps(rep, default=str)) != want:
                    raise SystemExit(f"REFUSING: L{li} {pset} {lvl}: the realised operating point is not {depth_rel}'s")
                n_anchor += int(anchor_on)
                if args.fpr5 and anchor_on:
                    # D-248 self-anchor: the same cell at 1% equals the 1% depth record's operating point
                    rep1, _ = P.repartition_estimate(pre_pool[:n_cal], pre_pool[n_cal:], post_pool[:n_cal], post_pool[n_cal:],
                                                     pos_pre, pos_post, draws[lvl], P.TARGET_FPR)
                    d1 = dcells1[(li, pset)]
                    want1 = d1["operating_point"] if lvl == LEVELS[0] else d1[LEVELS[1]]["operating_point"]
                    if json.dumps(json.loads(json.dumps(rep1, default=str)), sort_keys=True) != json.dumps(want1, sort_keys=True):
                        raise SystemExit(f"REFUSING: L{li} {pset} {lvl}: the 1% twin is not {depth1_rel}'s (D-248)")
                    n_self_anchor += 1
                per = {q: [] for q in STATS if q != "realised_paired_ratio"}
                mine = []
                for c_, e_ in parts[lvl]:
                    T = thresholds(pre_pool[c_], args.target_fpr)
                    h = health(pre_pool[c_], post_pool[c_], T)
                    for q in per:
                        per[q].append(h[q])
                    mine.append(P.ratio_from_flags(pre_pool[e_] >= T, post_pool[e_] >= T))
                v = {q: np.asarray(x, dtype="<f8") for q, x in per.items()}
                v["realised_paired_ratio"] = np.concatenate([[rep["paired_ratio"]["frozen"]], arr["paired_ratio"]]).astype("<f8")
                # the health thresholds are the realised ones: the same T_d gives the same paired ratio, draw by draw
                if not np.array_equal(np.asarray(mine, dtype="<f8"), v["realised_paired_ratio"]):
                    raise SystemExit(f"REFUSING: L{li} {pset} {lvl}: the partitions are not repartition_estimate's")
                blk: dict[str, Any] = {q: P.band_summary(x[1:], x[0]) for q, x in v.items()}
                rr = v["realised_paired_ratio"]
                flags = {"gaussian": v["gaussian_ratio"], "direct": v["direct_ratio"]}
                blk["share_of_draws_in_class"] = {c: float(crosses(rr[1:], cut).mean()) for c, cut in CLASSES.items()}
                blk["share_of_draws_flagged"] = {f"{d}|{c}": float(crosses(x[1:], cut).mean()) for d, x in flags.items()
                                                 for c, cut in CLASSES.items()}
                blk["share_of_draws_flagged"].update({f"auc_rule|{c}": float(auc_flag) for c in CLASSES})
                cblk[lvl] = blk
                for q, x in v.items():
                    arrays[f"L{li}|{pset}|{lvl}|{q}"] = x
                for c, cut in CLASSES.items():
                    y = crosses(rr, cut)
                    for dname in DETECTORS:
                        f = np.full(rr.shape, auc_flag) if dname == "auc_rule" else crosses(flags[dname], cut)
                        cc = confusion.setdefault((pset, lvl), {})
                        for key, val in (("tp", y & f), ("fn", y & ~f), ("fp", ~y & f), ("tn", ~y & ~f)):
                            cc[f"{dname}|{c}|{key}"] = cc.get(f"{dname}|{c}|{key}", np.zeros(rr.shape, dtype=np.int64)) + val
            cells.append(cblk)
        print(f"  layer {li:>2} done", flush=True)

    if args.fpr5 and anchor_on and n_self_anchor != len(layers) * len(PROBE_SETS) * len(LEVELS):
        raise SystemExit(f"REFUSING to write: {n_self_anchor} self-anchored blocks (D-248)")
    if anchor_on and n_anchor != len(layers) * len(PROBE_SETS) * len(LEVELS):
        raise SystemExit(f"REFUSING to write: {n_anchor} anchor blocks")
    counts: dict[str, Any] = {}
    for (pset, lvl), cc in confusion.items():
        blk = counts.setdefault(pset, {}).setdefault(lvl, {})
        for key, x in cc.items():
            arrays[f"counts|{pset}|{lvl}|{key}"] = x
            blk[key] = {"d156_mean": float(x[1:].mean()), "frozen_split": int(x[0])}
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    npz_bytes = buf.getvalue()
    code1 = P.code_block(CODE_FILES)
    changed = [n for n, a, b in (("code", code1["sha256"], code0["sha256"]), ("selection", sha256_file(sel_path), sel_sha),
                                 ("cluster manifest", sha256_file(D.CLUSTERS), cl_sha), (depth_rel, sha256_file(ROOT / depth_rel), depth_sha),
                                 ("registry sources", D.sources_now(), reg_sources)) if a != b]
    changed += [f"cache {r['file']}" for r in input_recs.values() if sha256_file(resolve_cache(r["file"], REG.CACHE)) != r["sha256"]]
    if changed:
        raise SystemExit(f"REFUSING to write: changed during the run: {changed}")
    payload = {**provenance_block("healthcheck_d156"),
               "preregistration": "D-152 4c, D-241 (D-169 item 12)" + ("; 4e: D-152, D-248 (the 5% arm)" if args.fpr5 else ""),
               **({"fpr5_self_anchor": {"file": depth1_rel, "sha256": depth1_sha, "n_level_blocks_identical_at_1pct": n_self_anchor,
                                        "kind": "every cell's operating point recomputed at 1% equals the 1% depth record "
                                                "(D-248)"}} if args.fpr5 else {}),
               "code": code0,
               "registered_parameters": registered, "blas_threads": threads, "input_verification": verification,
               "config": {"pair": args.pair, "registry_pair": reg_stem, "control": args.pair == CONTROL, "axis": axis,
                          "selection_file": str(sel_path.relative_to(ROOT)).replace("\\", "/"), "selection_pair": sel_stem,
                          "selection_sha256": sel_sha, "selected_C": C_sel, "selected_layer": L_sel, "layers": layers,
                          "seeds": args.seeds, "target_fpr": args.target_fpr, "n_draws": args.n_draws,
                          "draw_seed": P.AXIS_DRAW_SEED[axis], "cluster_draw_seed": P.AXIS_CLUSTER_DRAW_SEED[axis],
                          "cluster_manifest": {"file": "manifests/negative_clusters.json", "sha256": cl_sha},
                          "classes": {c: list(cut) for c, cut in CLASSES.items()}, "detectors": list(DETECTORS),
                          "pre_repo": cfg["pre_repo"], "pre_rev": cfg["pre_rev"], "post_repo": cfg["post_repo"],
                          "post_rev": cfg["post_rev"], "input_registry_sources_sha256": reg_sources},
               "anchor": {"file": depth_rel, "sha256": depth_sha, "n_level_blocks_identical": n_anchor,
                          "kind": "the realised operating point (paired.repartition_estimate's report) at every layer, "
                                  "probe set and level; the AUC rule reads the same file's AUROCs"},
               "population": {"declared_in": "src/populations.py", "group": "HEALTHCHECK_D156_LORA" if lora else "HEALTHCHECK_D156",
                              "replaces": [] if args.fpr5 else (["auc_criterion_audit_lora (its LoRA part, D-278)"] if lora
                                                                  else list(POP.AUC_AUDITS.values())),
                              "role": ("negative check" if args.pair == CONTROL else
                                       "LoRA run (D-278; all nine share one pre model)" if lora else "released pair") +
                                      ("; 4e sensitivity arm at 5% (D-248)" if args.fpr5 else "")},
               "input_cache_sha256": input_sha,
               "input_cache_source": {kk: {"file": r["file"], "expected_from": r["expected_from"], "pin_only": r["pin_only"]}
                                      for kk, r in input_recs.items()},
               "draws_npz": {"file": str(npz_path.relative_to(ROOT)).replace("\\", "/") if npz_path.is_relative_to(ROOT) else str(npz_path),
                             "sha256": sha256_bytes(npz_bytes), "size": len(npz_bytes),
                             "keys": "L<layer>|<probe set>|<level>|<statistic> and counts|<probe set>|<level>|<detector>|<class>|tp/fn/fp/tn; "
                                     "index 0 the frozen split"},
               "confusion_counts_pooled_over_cells": counts, "cells": cells}
    json_bytes = json.dumps(payload, indent=2, sort_keys=False, default=str).encode("utf-8")
    staged: list[tuple[Path, Path, str]] = []
    try:
        for p, b in ((npz_path, npz_bytes), (out_path, json_bytes)):
            t, h = stage_bytes(p, b)
            staged.append((t, p, h))
        for t, p, h in staged:
            commit_staged(t, p, h)
    finally:
        for t, _, _ in staged:
            if t.exists():
                t.unlink()
    for t, p, h in staged:
        print(f"Written: {p}\nsha256:  {h}")
    print(f"anchor blocks identical: {n_anchor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
