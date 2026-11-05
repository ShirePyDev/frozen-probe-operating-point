"""Over-refusal under D-156 / D-167 (D-169 item 9; pre-registered in D-187): does a backbone update
make a frozen safety probe flag more of XSTest's safe-but-scary prompts?

Replaces the D-161-superseded overrefusal_* results (src/over_refusal_test.py). As before, the probe
is fitted and its threshold calibrated on the SAFETY axis (Aegis) and then applied unchanged to
XSTest: 250 safe prompts built to look unsafe (label 0) and 200 genuinely unsafe contrast prompts
(label 1).

  D-167       XSTest, a fixed external set, is NEVER re-partitioned. Each D-156 draw sets every
              seed's threshold on that draw's calibration part of the pooled Aegis negatives (the
              same partitions as 4b and depth: paired.AXIS_DRAW_SEED; the D-164 cluster level
              beside), and all 450 XSTest prompts are scored at it. Reported per draw: the refusal
              rate on the safe prompts pre and post and its change; the paired refusal ratio (mean
              over seeds of floored post/pre rates, floor 1/250); the mean number of safe prompts
              flagged pre and post (the paper's item counts); the flag rate on the contrast prompts
              pre, post and change; exact McNemar on the safe prompts (maximum p over seeds, with the
              minimum achievable p). Each as mean and 95% band over draws, the frozen split as one draw
              with its percentile and its per-seed counts k/250.
  PROBES      D-117 selected C (primary) and C=1 (continuity; the superseded runs used C=1), at every
              layer present in the Aegis cache and in both XSTest caches (see LAYERS; D-189).
              HEADLINE: the mid layer declared in populations.OVER_REFUSAL_PAIRS (D-187).
  AUROC       contrast vs safe on XSTest, pre and post and change: threshold-free, so the D-152 P5
              item bootstrap (B = 5000; safe and contrast prompts resampled separately; D-187 seeds).
  INPUTS      only what the item reads: the PRE model's Aegis probe_train, threshold_calib and eval
              (probe, pooled negatives, and the pre-side check) and both models' XSTest eval, all
              through src/input_registry.py (D-165). The D-170 flag follows those bytes.
  CHECKS      the pre-side operating point at every layer, probe set and level (FPR and TPR pre,
              the floor, the pre counts) must equal twomoment_<pair>.json's exactly (for Mistral
              v0.1 -> v0.2, the Mistral pair's: the same pre bytes); that file is the one the D-174
              fit read. The thresholds XSTest is scored at are tied to the shared estimator draw by
              draw: at them, the Aegis pre FPR on each draw's evaluation part equals its per-draw
              value exactly (D-189).
  LAYERS      every layer present in the Aegis grid and in BOTH XSTest grids (D-189; Qwen2.5 and
              Mistral v0.1 -> v0.2 have pre and post XSTest caches on different grids).
Only registered parameters at the 6-thread pin write results/overrefusal_d156_<pair>.json.
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
import input_registry as REG  # noqa: E402
import metrics as M  # noqa: E402
import paired as P  # noqa: E402
import populations as POP  # noqa: E402
from depth_d156 import weighted_auroc  # noqa: E402
from provenance import (  # noqa: E402
    ensure_utf8_stdout, provenance_block, resolve_cache, sha256_bytes, sha256_file, write_results,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CODE_FILES = ["src/overrefusal_d156.py", "src/depth_d156.py", "src/input_registry.py", "src/paired.py",
              "src/populations.py", "src/provenance.py", "src/superseded_provenance.py", "src/metrics.py",
              "src/repdist.py"]
CLUSTERS = ROOT / "manifests" / "negative_clusters.json"
SELECTION_OF_RECORD = RESULTS / "selection_sweep_d165.json"
TWOMOMENT_FIT = RESULTS / "twomoment_fit.json"
TWOMOMENT_FIT_SHA256 = "515faab949465d0852cd61dbc1c7805cf3d800c5e2222029e4622261727a3b96"   # D-174
LEVELS = ("item_level_D156", "cluster_level_D164")
PROBE_SETS = ("selected", "C=1")
REGISTERED = {"n_draws": P.N_DRAWS, "target_fpr": P.TARGET_FPR, "seeds": [0, 1, 2, 3, 4], "n_boot": 5000}
ITEM_BOOT_SEED = {"xstest_safe": 20260933, "xstest_contrast": 20260934}        # D-187
# over-refusal stem -> (registry pair, the pair whose selection and twomoment file apply)
PAIRS = {"overrefusal_mistral": ("repair_mistral_base2instruct", "repair_mistral_base2instruct"),
         "overrefusal_gemma2": ("repair_gemma2_base2instruct", "repair_gemma2_base2instruct"),
         "overrefusal_llama3": ("repair_llama3_base2instruct", "repair_llama3_base2instruct"),
         "overrefusal_llama31": ("repair_llama31_base2instruct", "repair_llama31_base2instruct"),
         "overrefusal_qwen": ("repair_qwen_base2instruct", "repair_qwen_base2instruct"),
         "overrefusal_mistral_v01_v02": ("depth_mistral_v01_v02_dense", "repair_mistral_base2instruct"),
         "overrefusal_llama3_to_31": ("repair_llama3_to_31", "repair_llama3_to_31")}
PRE_ONLY_METRICS = ("n_draws", "fpr_pre", "tpr_pre", "floor_ratio_to_target", "floor_abs_log")
PRE_ONLY_COUNTS = ("n_flagged_pre", "hits_pre", "n_eval_negatives", "n_positives")
QUANT = ("refusal_pre", "refusal_post", "refusal_change", "paired_refusal_ratio", "safe_flagged_pre_mean",
         "safe_flagged_post_mean", "contrast_flag_pre", "contrast_flag_post", "contrast_flag_change",
         "mcnemar_safe_p_max")
AUROC_NOTE = ("XSTest contrast vs safe, pre and post: threshold-free, never re-partitioned (D-167); 95% interval by "
              "the D-152 P5 item bootstrap (B = 5000, safe and contrast prompts resampled separately)")


def sources_now() -> dict[str, str]:
    """The registry's sources re-read from disk: every ORIGINAL extraction record (reruns excluded,
    as the registry excludes them) and every committed pin manifest, with their sha256."""
    out = {f"results/{f.name}": sha256_file(f) for f in sorted(RESULTS.glob("extract__*.json")) if "__rerun_" not in f.name}
    for m in REG.MANIFESTS:
        if m.exists():
            out[str(m.relative_to(ROOT)).replace("\\", "/")] = sha256_file(m)
    return out


def xstest_point(pool_pre: np.ndarray, cal: np.ndarray, ev: np.ndarray, safe_pre: np.ndarray, safe_post: np.ndarray,
                 con_pre: np.ndarray, con_post: np.ndarray, target: float) -> dict[str, Any]:
    """XSTest at ONE partition's per-seed thresholds (the thresholds paired.operating_point sets
    from the same calibration scores). Arrays (items, seeds). `aegis_fpr_pre` (the Aegis pre FPR on
    the partition's evaluation part at these thresholds) exists only to tie the thresholds to the
    shared estimator draw by draw; XSTest itself is never re-partitioned (D-167)."""
    T = np.array([P.thr_at_fpr(pool_pre[cal, s], target) for s in range(pool_pre.shape[1])])
    S_pre, S_post = safe_pre >= T, safe_post >= T
    C_pre, C_post = con_pre >= T, con_post >= T
    b = (S_pre & ~S_post).sum(axis=0)
    c = (~S_pre & S_post).sum(axis=0)
    bc, cc = (C_pre & ~C_post).sum(axis=0), (~C_pre & C_post).sum(axis=0)
    zero = (S_pre.sum(axis=0) == 0) | (S_post.sum(axis=0) == 0)       # the 1/250 floor bites
    return {"aegis_fpr_pre": float(P.floored_fpr(pool_pre[ev] >= T).mean()),
            "floored": bool(zero.any()), "n_floored_seeds": int(zero.sum()),
            "refusal_pre": float(P.rate(S_pre).mean()), "refusal_post": float(P.rate(S_post).mean()),
            "refusal_change": float((P.rate(S_post) - P.rate(S_pre)).mean()),
            "paired_refusal_ratio": P.ratio_from_flags(S_pre, S_post),
            "safe_flagged_pre_mean": float(S_pre.sum(axis=0).mean()), "safe_flagged_post_mean": float(S_post.sum(axis=0).mean()),
            "contrast_flag_pre": float(P.rate(C_pre).mean()), "contrast_flag_post": float(P.rate(C_post).mean()),
            "contrast_flag_change": P.delta_tpr_from_flags(C_pre, C_post),
            "mcnemar_safe_p_max": float(max(P.mcnemar_exact(int(x), int(y)) for x, y in zip(b, c))),
            "_frozen": {"safe_flagged_pre": S_pre.sum(axis=0).tolist(), "safe_flagged_post": S_post.sum(axis=0).tolist(),
                        "safe_discordant_b_c": [[int(x), int(y)] for x, y in zip(b, c)],
                        "contrast_discordant_b_c": [[int(x), int(y)] for x, y in zip(bc, cc)],
                        "n_floored_seeds": int(zero.sum()),
                        "mcnemar_min_achievable_p": float(max(P.min_mcnemar_p(int(x + y)) for x, y in zip(b, c))),
                        "contrast_flagged_pre": C_pre.sum(axis=0).tolist(), "contrast_flagged_post": C_post.sum(axis=0).tolist(),
                        "n_safe": int(S_pre.shape[0]), "n_contrast": int(C_pre.shape[0])}}


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Over-refusal under D-156/D-167 (D-169 item 9, D-187)")
    ap.add_argument("--pair", required=True, choices=list(PAIRS))
    ap.add_argument("--seeds", type=int, nargs="*", default=REGISTERED["seeds"])
    ap.add_argument("--n-draws", type=int, default=REGISTERED["n_draws"])
    ap.add_argument("--target-fpr", type=float, default=REGISTERED["target_fpr"])
    ap.add_argument("--n-boot", type=int, default=REGISTERED["n_boot"])
    ap.add_argument("--selection", default=str(SELECTION_OF_RECORD))
    ap.add_argument("--xstest-cache-tag", default=None, choices=["xsfull"],
                    help="D-206: read the XSTest cache of a side from its tagged full-grid extraction where one "
                         "exists (Qwen2.5-7B, Mistral-7B-v0.2); writes overrefusal_d156_<pair>_<tag>.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    stem = args.pair
    reg_stem, sel_stem = PAIRS[stem]
    mid = POP.OVER_REFUSAL_PAIRS[stem][2]
    registered = (args.n_draws == REGISTERED["n_draws"] and args.target_fpr == REGISTERED["target_fpr"]
                  and list(args.seeds) == REGISTERED["seeds"] and args.n_boot == REGISTERED["n_boot"])
    suffix = f"_{args.xstest_cache_tag}" if args.xstest_cache_tag else ""
    final_path = (RESULTS / f"overrefusal_d156_{stem[len('overrefusal_'):]}{suffix}.json").resolve()
    out_path = Path(args.out).resolve() if args.out else final_path
    final = out_path == final_path
    if out_path.exists():
        raise SystemExit(f"REFUSING: {out_path} exists; results are never overwritten (D-152 P5)")
    if out_path.is_relative_to(RESULTS.resolve()) and not final:
        raise SystemExit("REFUSING: a test run (--out) must not write under results/")
    sel_path = Path(args.selection).resolve()
    fit_sha = sha256_file(TWOMOMENT_FIT)
    if final:
        threads = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
        problems = ([] if registered else ["non-registered parameters"]) + \
                   ([] if sel_path == SELECTION_OF_RECORD.resolve() else ["not the selection of record"]) + \
                   ([] if all(t == "6" for t in threads.values()) else [f"no 6-thread BLAS pin: {threads}"]) + \
                   ([] if fit_sha == TWOMOMENT_FIT_SHA256 else ["twomoment_fit.json is not the D-174 fit"])
        if problems:
            raise SystemExit(f"REFUSING the final name: {problems}; use --out outside results/ for a test")
    sel_sha, cl_sha = sha256_file(sel_path), sha256_file(CLUSTERS)
    code0 = P.code_block(CODE_FILES)

    sel_doc = json.load(open(sel_path, encoding="utf-8"))
    if list(sel_doc["config"]["seeds"]) != list(args.seeds) or float(sel_doc["config"]["holdout"]) != P.HOLDOUT:
        raise SystemExit("REFUSING: selection seeds/holdout differ from this run's")
    sel = next((m for m in sel_doc["models"] if m["pair"] == sel_stem), None)
    if sel is None or not sel["selected"]["arm"].startswith("C="):
        raise SystemExit(f"REFUSING: no logistic-C selection for {sel_stem}")
    C_sel, L_sel = float(sel["selected"]["arm"].split("=")[1]), int(sel["selected"]["layer"])

    tm_rel = f"results/twomoment_{sel_stem}.json"
    fit_doc = json.load(open(TWOMOMENT_FIT, encoding="utf-8"))
    if not (ROOT / tm_rel).exists() or sha256_file(ROOT / tm_rel) != fit_doc["input_files_sha256"].get(tm_rel):
        raise SystemExit(f"REFUSING: {tm_rel} is missing or not the file twomoment_fit.json read")
    tm_sha = sha256_file(ROOT / tm_rel)
    tm_doc = json.load(open(ROOT / tm_rel, encoding="utf-8"))
    tm_cfg = tm_doc["config"]
    if (tm_cfg["selected_C"], tm_cfg["selected_layer"], tm_cfg["selection_sha256"], list(tm_cfg["seeds"])) != \
            (C_sel, L_sel, sel_sha, list(args.seeds)):
        raise SystemExit(f"REFUSING: {tm_rel} was run on another selection or seeds")
    tm_cells = {(c["layer"], c["probe_set"]): c for c in tm_doc["cells"]}

    cfg = REG.run(reg_stem)
    axis = cfg["axis"]
    if axis != "safety":
        raise SystemExit("REFUSING: the over-refusal protocol trains and calibrates on the Aegis (safety) axis")
    need = [("pre", axis, "probe_train"), ("pre", axis, "threshold_calib"), ("pre", axis, "eval"),
            ("pre", "over_refusal", "eval"), ("post", "over_refusal", "eval")]
    data: dict[str, Any] = {}
    input_sha: dict[str, str] = {}
    input_recs: dict[str, dict[str, Any]] = {}
    xs_repo: dict[str, str] = {}
    for side, ax, part in need:
        repo = cfg[f"{side}_repo"]
        if ax == "over_refusal" and args.xstest_cache_tag:
            # D-206: the tagged full-grid XSTest extraction where one exists (its own records name it)
            tagged = f"{repo}__{args.xstest_cache_tag}"
            if any(r["repo"] == tagged for r in REG._load_records()):
                if REG.revision(tagged) != cfg[f"{side}_rev"]:        # refuses on conflicting tagged records too
                    raise SystemExit(f"REFUSING: {tagged}'s revision is not {repo}'s pinned revision")
                repo = tagged
            xs_repo[side] = repo
        z, r = REG.load(repo, ax, part)
        key = f"{side}/{ax}/{part}"
        data[key], input_sha[key], input_recs[key] = z, r["sha256"], r
    if args.xstest_cache_tag and not any(v.endswith(f"__{args.xstest_cache_tag}") for v in xs_repo.values()):
        raise SystemExit(f"REFUSING: --xstest-cache-tag {args.xstest_cache_tag}: neither side of {stem} has a tagged XSTest cache")
    reg_sources = REG.sources_sha256()
    # the registry caches its records' hashes at first load, so a change or a new record during the
    # run is caught by re-reading them here, not by calling it again (review, D-189)
    if sources_now() != reg_sources:
        raise SystemExit("REFUSING: the extraction records or pins on disk are not the ones the registry read")
    if any(sel_doc["input_cache_sha256"].get(f"{sel_stem}:pre/{p}") != input_sha[f"pre/{axis}/{p}"]
           for p in ("probe_train", "threshold_calib")):
        raise SystemExit("REFUSING: the selection was made on other bytes")
    if any(tm_doc["input_cache_sha256"][f"pre/{p}"] != input_sha[f"pre/{axis}/{p}"] for p in ("threshold_calib", "eval")):
        raise SystemExit(f"REFUSING: {tm_rel} read other pre-update caches")
    verification = REG.verification(input_recs)
    xa, xb = data["pre/over_refusal/eval"], data["post/over_refusal/eval"]
    if not np.array_equal(xa["item_ids"], xb["item_ids"]) or not np.array_equal(xa["labels"], xb["labels"]):
        raise SystemExit("REFUSING: XSTest items or labels differ between pre and post")
    for key, z in data.items():
        if "nonfinite" in z.files and z["nonfinite"].any():
            raise SystemExit(f"REFUSING: {key} has non-finite items")
    if all(k in xa.files and k in xb.files for k in ("n_tokens", "truncated")):
        tok = {"n_items_n_tokens_differ": int((xa["n_tokens"] != xb["n_tokens"]).sum()),
               "n_items_truncated_differ": int((xa["truncated"] != xb["truncated"]).sum())}
    elif final:
        raise SystemExit("REFUSING: an XSTest cache lacks n_tokens/truncated; tokenisation cannot be compared (D-180)")
    else:
        tok = "not in these caches (test inputs)"
    grid = lambda z: sorted(int(k.split("_")[1]) for k in z.files if k.startswith("layer_"))  # noqa: E731
    g_safety = [grid(data[f"pre/{axis}/{p}"]) for p in ("probe_train", "threshold_calib", "eval")]
    if any(g != g_safety[0] for g in g_safety):
        raise SystemExit("REFUSING: the Aegis caches have different layer grids")
    # the pre and post XSTest caches may hold different grids (Qwen2.5: [7,14,21,28] vs even 0..28;
    # Mistral v0.1 even 0..32 vs v0.2 [8,16,24,32]): every layer present in all three (D-189)
    grids = {"aegis": g_safety[0], "xstest_pre": grid(xa), "xstest_post": grid(xb)}
    layers = sorted(set(grids["aegis"]) & set(grids["xstest_pre"]) & set(grids["xstest_post"]))
    if mid not in layers:
        raise SystemExit(f"REFUSING: the declared mid layer {mid} is not in every grid {layers}")
    if args.xstest_cache_tag and L_sel not in layers:
        raise SystemExit(f"REFUSING: the D-117 layer {L_sel} is not in every grid {layers} even with the tagged caches (D-206)")
    if not set(layers) <= {li for li, _ in tm_cells}:
        raise SystemExit(f"REFUSING: {tm_rel} lacks layers {sorted(set(layers) - {li for li, _ in tm_cells})}")

    y_tr = data[f"pre/{axis}/probe_train"]["labels"].astype(bool)
    ncal = ~data[f"pre/{axis}/threshold_calib"]["labels"].astype(bool)
    y_ev = data[f"pre/{axis}/eval"]["labels"].astype(bool)
    neg, pos = ~y_ev, y_ev
    x_con = xa["labels"].astype(bool)          # XSTest: 1 = contrast (unsafe), 0 = safe-but-scary
    x_safe = ~x_con
    pooled_ids = [str(i) for i in data[f"pre/{axis}/threshold_calib"]["item_ids"][ncal]] + \
                 [str(i) for i in data[f"pre/{axis}/eval"]["item_ids"][neg]]
    n_cal, n_pool = int(ncal.sum()), len(pooled_ids)
    if sha256_bytes("\n".join(pooled_ids).encode()) != tm_cfg["pooled_negatives"]["item_ids_sha256"]:
        raise SystemExit(f"REFUSING: the pooled negatives differ from {tm_rel}'s")
    frozen_part = (np.arange(n_cal), np.arange(n_cal, n_pool))
    draws = {LEVELS[0]: P.repartition_draws(n_cal, n_pool - n_cal, args.n_draws, P.AXIS_DRAW_SEED[axis])}
    cl = json.load(open(CLUSTERS, encoding="utf-8"))["axes"][axis]
    if cl["n_pooled_negatives"] != n_pool or cl["n_calibration"] != n_cal or \
            cl["pool_item_ids_sha256"] != sha256_bytes("\n".join(pooled_ids).encode("utf-8")):
        raise SystemExit(f"REFUSING: manifests/negative_clusters.json does not describe the {axis} pool")
    draws[LEVELS[1]] = P.cluster_repartition_draws(P.cluster_units(pooled_ids, cl["clusters"]), n_cal,
                                                   args.n_draws, P.AXIS_CLUSTER_DRAW_SEED[axis])
    parts = {lvl: [frozen_part] + [(d[:n_cal], d[n_cal:]) for d in dr] for lvl, dr in draws.items()}
    n_safe, n_con = int(x_safe.sum()), int(x_con.sum())
    if final and (n_safe, n_con) != (250, 200):
        raise SystemExit(f"REFUSING: XSTest holds {n_safe} safe and {n_con} contrast prompts, not 250 and 200 (D-187)")
    CNT_S = P.counts_matrix(n_safe, args.n_boot, ITEM_BOOT_SEED["xstest_safe"])
    CNT_C = P.counts_matrix(n_con, args.n_boot, ITEM_BOOT_SEED["xstest_contrast"])
    print(f"{stem} via {reg_stem} (selection and 4b check of {sel_stem}): C={C_sel:g}; headline mid layer {mid}; "
          f"layers {layers}; XSTest {n_safe} safe + {n_con} contrast; {args.n_draws} draws per level; "
          f"D-170 flag: {verification['flag']}; registered: {registered}", flush=True)

    cells = []
    n_checked = n_draw_ties = 0
    for li in [mid] + [x for x in layers if x != mid]:
        k = f"layer_{li}"
        Xtr, Xcal, Xev = (data[f"pre/{axis}/{p}"][k] for p in ("probe_train", "threshold_calib", "eval"))
        Xa, Xb = xa[k], xb[k]
        for pset in PROBE_SETS:
            probes = []
            for seed in args.seeds:
                _, fit = P.d117_split(len(y_tr), seed)
                rows, C, order = (fit, C_sel, "d117_permutation") if pset == "selected" else (np.sort(fit), 1.0, "file_order")
                probes.append(P.fit_probe(Xtr[rows], y_tr[rows], seed, C, order))
            S = lambda x: np.column_stack([q.score(x) for q in probes])  # noqa: E731
            cal_pre, ev_pre, pos_pre = S(Xcal[ncal]), S(Xev[neg]), S(Xev[pos])
            pool_pre = np.vstack([cal_pre, ev_pre])
            sa, sb = S(Xa), S(Xb)
            cell: dict[str, Any] = {"pair": stem, "layer": li, "probe_set": pset,
                                    "probe_C": C_sel if pset == "selected" else 1.0, "is_headline_mid_layer": li == mid,
                                    "input_verification": verification}
            tm = tm_cells[(li, pset)]
            for lvl in LEVELS:
                # the pre side through the shared estimator (post = pre): must equal 4b's pre-only numbers
                pre_report, pre_arr = P.repartition_estimate(cal_pre, ev_pre, cal_pre, ev_pre, pos_pre, pos_pre,
                                                             draws[lvl], args.target_fpr)
                mine = json.loads(json.dumps(pre_report, default=str))
                want = tm["realised"] if lvl == LEVELS[0] else tm[LEVELS[1]]["realised"]
                diff = [q for q in PRE_ONLY_METRICS if mine[q] != want[q]] + \
                       [q for q in PRE_ONLY_COUNTS if mine["frozen_split_counts"][q] != want["frozen_split_counts"][q]]
                if diff:
                    raise SystemExit(f"REFUSING: layer {li} {pset} {lvl}: the pre side differs from {tm_rel} in {diff}")
                n_checked += 1
                per = [xstest_point(pool_pre, c, e, sa[x_safe], sb[x_safe], sa[x_con], sb[x_con], args.target_fpr)
                       for c, e in parts[lvl]]
                # the thresholds XSTest is scored at are the shared estimator's, draw by draw (D-189)
                afp = np.array([p["aegis_fpr_pre"] for p in per])
                if afp[0] != pre_report["fpr_pre"]["frozen"] or not np.array_equal(afp[1:], pre_arr["fpr_pre"]):
                    raise SystemExit(f"REFUSING: layer {li} {pset} {lvl}: the XSTest thresholds are not the shared estimator's")
                n_draw_ties += 1
                blk: dict[str, Any] = {q: P.band_summary(np.array([p[q] for p in per[1:]]), per[0][q]) for q in QUANT}
                blk["share_of_draws_mcnemar_safe_p_max_below_0_05"] = float(np.mean([p["mcnemar_safe_p_max"] < 0.05 for p in per[1:]]))
                # D-163 (2): the paired refusal ratio is floored at 1/250; the rates themselves are not
                blk["paired_refusal_ratio_share_of_draws_floored"] = float(np.mean([p["floored"] for p in per[1:]]))
                blk["frozen_split"] = per[0]["_frozen"]
                if lvl == LEVELS[1]:
                    blk["note"] = ("cluster-level re-partitions of the Aegis negatives (D-164); XSTest itself is never "
                                   "re-partitioned; 'frozen' is the frozen split, not a cluster-level draw")
                cell[lvl] = blk
            au: dict[str, Any] = {"note": AUROC_NOTE}
            reps, pts = {}, {}
            for name, sc in (("pre", sa), ("post", sb)):
                pt = np.array([M.auroc(sc[x_con, s], sc[x_safe, s]).value for s in range(len(args.seeds))])
                unit = np.array([weighted_auroc(sc[x_con, s], sc[x_safe, s], np.ones((1, n_con)), np.ones((1, n_safe)))[0]
                                 for s in range(len(args.seeds))])
                if not np.array_equal(pt, unit):
                    raise SystemExit(f"REFUSING: layer {li} {pset} {name}: weighted AUROC is not metrics.auroc")
                pts[name] = pt
                reps[name] = np.mean([weighted_auroc(sc[x_con, s], sc[x_safe, s], CNT_C, CNT_S)
                                      for s in range(len(args.seeds))], axis=0)
            pts["change"], reps["change"] = pts["post"] - pts["pre"], reps["post"] - reps["pre"]
            for name in ("pre", "post", "change"):
                au[name] = {"mean_over_seeds": float(pts[name].mean()), "per_seed": pts[name].tolist(), **P.ci(reps[name])}
            cell["xstest_auroc"] = au
            cell["probe_fit_diagnostics"] = [q.diag for q in probes]
            cells.append(cell)
        h = next(x for x in cells if x["layer"] == li and x["probe_set"] == "selected")[LEVELS[0]]
        print(f"  layer {li:>2} [selected C]: safe prompts flagged pre {h['safe_flagged_pre_mean']['mean']:.1f} -> post "
              f"{h['safe_flagged_post_mean']['mean']:.1f} of {n_safe}; refusal change {h['refusal_change']['mean']:+.4f} "
              f"[{h['refusal_change']['band95'][0]:+.4f}, {h['refusal_change']['band95'][1]:+.4f}]", flush=True)

    headline = {pset: next(x for x in cells if x["layer"] == mid and x["probe_set"] == pset) for pset in PROBE_SETS}
    # D-206: the D-117 selected layer is the PRIMARY headline of a tagged re-run; the mid layer stays beside it
    headline_d117 = ({pset: next(x for x in cells if x["layer"] == L_sel and x["probe_set"] == pset) for pset in PROBE_SETS}
                     if L_sel in layers else None)
    code1 = P.code_block(CODE_FILES)
    changed = [n for n, a, b in (("code", code1["sha256"], code0["sha256"]), ("selection", sha256_file(sel_path), sel_sha),
                                 ("cluster manifest", sha256_file(CLUSTERS), cl_sha), ("twomoment file", sha256_file(ROOT / tm_rel), tm_sha),
                                 ("twomoment_fit.json", sha256_file(TWOMOMENT_FIT), fit_sha),
                                 ("registry sources (re-read from disk)", sources_now(), reg_sources)) if a != b]
    changed += [f"cache {r['file']}" for r in input_recs.values() if sha256_file(resolve_cache(r["file"], REG.CACHE)) != r["sha256"]]
    if changed:
        raise SystemExit(f"REFUSING to write: changed during the run: {changed}")
    rel = lambda p: str(p.relative_to(ROOT)).replace("\\", "/") if p.is_relative_to(ROOT) else p.name  # noqa: E731
    payload = {**provenance_block("overrefusal_d156"), "preregistration": "D-169 item 9, D-152 P5, D-156, D-164, D-165, D-167, D-187",
               "code": code0, "registered_parameters": registered, "input_verification": verification,
               "config": {"pair": stem, "registry_pair": reg_stem, "selection_of_pair": sel_stem, "axis": axis,
                          "external_set": "XSTest (over_refusal/eval), never re-partitioned (D-167)",
                          "headline_layer": {"layer": mid, "rule": "the mid layer declared in populations.OVER_REFUSAL_PAIRS (D-187)"},
                          "selected_C": C_sel, "selected_layer": L_sel, "layers": layers, "seeds": args.seeds,
                          "target_fpr": args.target_fpr, "selection_file": rel(sel_path), "selection_sha256": sel_sha,
                          "pre_repo": cfg["pre_repo"], "pre_rev": cfg["pre_rev"], "post_repo": cfg["post_repo"],
                          "post_rev": cfg["post_rev"], "input_registry_sources_sha256": reg_sources,
                          "cluster_manifest": {"file": "manifests/negative_clusters.json", "sha256": cl_sha},
                          "n_draws": args.n_draws, "draw_seed": P.AXIS_DRAW_SEED[axis],
                          "cluster_draw_seed": P.AXIS_CLUSTER_DRAW_SEED[axis],
                          "item_bootstrap": {"n_boot": args.n_boot, "seeds": ITEM_BOOT_SEED},
                          "n_safe": n_safe, "n_contrast": n_con},
               "tokenisation_xstest_pre_vs_post": tok,
               "twomoment_check": {"file": tm_rel, "sha256": tm_sha, "twomoment_fit_sha256": fit_sha,
                                   "kind": "pre-side operating point (FPR/TPR pre, floor, pre counts); the XSTest "
                                           "thresholds tied to the shared estimator in every draw",
                                   "n_reports_identical": n_checked,
                                   "n_level_blocks_xstest_thresholds_tied_every_draw": n_draw_ties},
               "layer_grids": grids,
               "population": {"declared_in": "src/populations.py", "group": "OVERREFUSAL_D156", "replaces": f"results/{stem}.json"},
               "input_cache_sha256": input_sha,
               "input_cache_source": {k: {"file": r["file"], "expected_from": r["expected_from"], "pin_only": r["pin_only"]}
                                      for k, r in input_recs.items()},
               "xstest_cache_tag": args.xstest_cache_tag, "xstest_cache_repos": xs_repo or None,
               "headline_d117_layer": headline_d117,
               "headline_mid_layer": headline, "cells": sorted(cells, key=lambda c: (c["layer"], c["probe_set"]))}
    digest = write_results(out_path, payload, exclusive=True)
    print(f"Written: {rel(out_path)}\nsha256:  {digest}\npre-side reports identical to {tm_rel}: {n_checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
