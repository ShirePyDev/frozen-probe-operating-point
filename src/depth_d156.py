"""Depth curves under D-156 (D-169 item 4; pre-registered in D-180): per layer, how a frozen probe
and its frozen threshold fare across a backbone update, for the ten pairs of UPDATE_PAIRS_DEPTH.

Replaces the depth_* curves: seven are D-161 superseded (code provenance unverifiable); three
(Mistral base->instruct, Qwen2.5, Mistral v0.1->v0.2) recorded their input hashes but no per-file
code hashes and are regenerated like the rest (D-169 (8)). What changed from src/depth_curve.py:

  PROBES      the shared machinery of src/paired.py (D-152 P3): per seed the D-117 80% split; the
              D-117 SELECTED C in permutation order (primary) and C=1 in file order (the continuity
              and sensitivity row, as in 4b). depth_curve.py fitted its own C=1 probe on a random 80%
              subsample. The post-refit probe is fitted on the post model's probe_train with the
              same rows, C and row order as the pre probe it is compared with.
  OPERATING   D-156 (and the D-164 cluster level beside it): the 1000 threshold_calib and 2000 eval
  POINTS      negatives pooled and re-partitioned 1000 times per axis with the fixed seeds; eval
              positives fixed; the frozen split is one more draw. paired.repartition_estimate,
              unchanged: FPR pre/post at the frozen threshold (floored at 1/2000), paired ratio,
              ratio to target, TPR pre/post, McNemar on the frozen split; plus the share of draws in
              which any seed's realised pre or post FPR was 0 before flooring (twomoment's
              realised_censoring definition). depth_curve.py measured the post-update FPR on the
              calibration negatives the threshold was set on (in-sample, unfloored).
  AUROC       threshold-free, so NOT re-partitioned (D-156 keeps the D-152 P5 item bootstrap for
              such quantities): on the frozen eval split, per seed and mean over seeds, pre (pre
              probe, pre scores), frozen (pre probe, post scores), post-refit, and drop_frozen =
              pre - frozen; 95% interval by the item bootstrap of D-152 P1/P5 (B = 5000; eval
              negatives and eval positives resampled separately, as src/intent_selected.py does;
              all five seeds recomputed on every resample; the same resamples at every layer, so
              the pair-level mean and maximum over layers have intervals too). Weighted
              Mann-Whitney in exact integer arithmetic; with unit weights it equals metrics.auroc
              bit for bit, which is checked per seed.
  CKA         linear CKA of pre and post representations on the frozen eval split (all items),
              unchanged; a point value, outside D-156 (no threshold, no label).
  FROZEN      per cell and level, the per-draw band of |log(FPR_post / target)| (the per-layer frozen
  CURVE       curve; 4d's Table 2 frozen column must equal it at shared cells, D-180).
  TABLE 1     (D-180) per pair, probe set and level: HEADLINE = the D-117 selected layer's paired
              ratio (D-156 mean and band) with ratio to target and TPR pre/post/delta beside it;
              SWEEP MAXIMUM = the layer with the highest per-layer mean, for the paired ratio and
              the ratio to target, each with its band and its own no-update null (D-156 floor: 1
              for the paired ratio; the maximum over layers of the mean FPR_pre / target for the
              ratio to target); the lowest and the two-sided largest (|log| of the mean) layers;
              the per-draw maximum over layers as a SENSITIVITY only (grows with the layer count;
              ties counted). AUROC drop: mean and maximum over layers with item-bootstrap
              intervals. Minimum CKA.
  INPUTS      D-165: pairs, revisions and expected cache bytes from src/input_registry.py; the
              selection of record results/selection_sweep_d165.json (D-171). The Mistral v0.1 ->
              v0.2 pair takes the repair_mistral_base2instruct selection: its pre-update caches are
              the bytes that selection was made on (checked; D-180). D-170 flag per cell and file.
              Pre/post tokenisation (n_tokens, truncated) compared and recorded per split.
  CHECKS      (1) nine pairs: at every layer and probe set, the operating-point report and the
              floored share at BOTH levels must equal results/twomoment_<pair>.json's exactly (same
              probes, partitions and code path; that file must be the one twomoment_fit.json read).
              (2) Mistral v0.1 -> v0.2: the pre-only quantities must equal the Mistral pair's.
              The cross-check always runs, so a test run needs a twomoment file made with the same
              parameters (the sandbox does this with 50 draws).

Only the registered parameters (1000 draws, 1% target, seeds 0-4, B = 5000) at the 6-thread BLAS
pin may write the final names results/depth_d156_<pair>.json; a test writes outside results/.

D-224 (D-169 items 10 and 11, the depth parts): the same code path for
  LoRA ARMS   the nine depth_<run> of populations.LORA_ARMS: the selection of record is the LoRA A100
              one (results/selection_sweep_lora_a100_d165.json, D-171; its one model, the shared A100
              base). CHECK: at the selected layer, both probe sets and both levels, the operating-point
              report must equal the run's 4a cell in results/damage_d156_r2.json exactly (same probes,
              pools and draws); there is no 4b file for these runs, so other layers have no reference.
  GPU CONTROL depth_hw_v100_a100 (populations.CONTROLS): Mistral-7B-v0.1 on the V100 against the same
              model on the A100, a no-update hardware null. It takes the Mistral pair's selection (the
              same pre bytes) and the pre-only check against that pair's 4b file, as the dense pair does.
"""

from __future__ import annotations

import argparse
import hashlib
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
from provenance import (  # noqa: E402
    ensure_utf8_stdout, provenance_block, resolve_cache, sha256_bytes, sha256_file, write_results,
)
from repdist import linear_cka  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CODE_FILES = ["src/depth_d156.py", "src/input_registry.py", "src/paired.py", "src/populations.py",
              "src/provenance.py", "src/superseded_provenance.py", "src/metrics.py", "src/repdist.py"]
CLUSTERS = ROOT / "manifests" / "negative_clusters.json"
SELECTION_OF_RECORD = RESULTS / "selection_sweep_d165.json"      # D-171
TWOMOMENT_FIT = RESULTS / "twomoment_fit.json"                    # names the twomoment files of record
TWOMOMENT_FIT_SHA256 = "515faab949465d0852cd61dbc1c7805cf3d800c5e2222029e4622261727a3b96"   # D-174; final names
LEVELS = ("item_level_D156", "cluster_level_D164")
REGISTERED = {"n_draws": P.N_DRAWS, "target_fpr": P.TARGET_FPR, "seeds": [0, 1, 2, 3, 4], "n_boot": 5000}
ITEM_BOOT_SEED = {"eval_negatives": 20260929, "eval_positives": 20260930}     # D-180
PROBE_SETS = ("selected", "C=1")
SELECTION_OF_RECORD_LORA = RESULTS / "selection_sweep_lora_a100_d165.json"     # D-171
LORA_SELECTION_PAIR = "repair_loraA100_mistral_s25"      # its one model: the A100 base every LoRA arm shares
LORA_DEPTH = [f"depth_{r}" for runs in POP.LORA_ARMS.values() for r in runs]   # D-224
CONTROL_DEPTH = ["depth_hw_v100_a100"]                                         # D-224; populations.CONTROLS
DAMAGE_OF_RECORD = RESULTS / "damage_d156_r2.json"                            # the LoRA arms' 4a cells (D-173)
DAMAGE_OF_RECORD_SHA256 = "9c6cd0d6ee2700f60626adca034eae45ace6c18f336fcd2cffc306b59e3919f6"
# 4e (D-152, D-248): the 5% arm reads the 5% upstream files and self-anchors on the 1% depth record
DAMAGE_FPR5 = RESULTS / "damage_d156_r2_fpr5.json"
TWOMOMENT_FIT_FPR5 = RESULTS / "twomoment_fit_fpr5.json"
TARGET_5 = 0.05
# D-180: the dense Mistral v0.1 -> v0.2 pair takes the selection made on its (identical) pre bytes; so does
# the GPU control (D-224). The LoRA arms take the LoRA A100 selection (D-224).
SELECTION_STEM = {"depth_mistral_v01_v02_dense": "repair_mistral_base2instruct",
                  "depth_hw_v100_a100": "repair_mistral_base2instruct",
                  **{s: LORA_SELECTION_PAIR for s in LORA_DEPTH}}
PRE_ONLY_METRICS = ("n_draws", "fpr_pre", "tpr_pre", "floor_ratio_to_target", "floor_abs_log")
PRE_ONLY_COUNTS = ("n_flagged_pre", "hits_pre", "n_eval_negatives", "n_positives")     # of frozen_split_counts
CLUSTER_NOTE = ("cluster-level re-partitions (D-164), reported beside the item level, never instead; its "
                "'frozen' value is the frozen split, which is not a cluster-level draw")
AUROC_NOTE = ("frozen eval split; 95% interval by the D-152 P5 item bootstrap (B = 5000, eval negatives and "
              "positives resampled separately, all seeds recomputed per resample); not re-partitioned (D-156 "
              "keeps the item bootstrap for threshold-free quantities)")
TIE_RULE = ("argmax/argmin over layers: values within rtol 1e-12 of the extreme count as tied (sums equal in exact "
            "arithmetic can differ in the last float bit); a tie goes to the lowest layer; n_layers_tied / n_draws_tied "
            "count them (D-183)")
FROZEN_CURVE_NOTE = ("per draw |log(FPR_post / target)|, FPR_post the seed-mean of floored per-seed FPRs (as in "
                     "ratio_to_target; the floor_abs_log definition applied to post); 4d's Table 2 frozen column must "
                     "equal it bit for bit at shared cells (D-180)")
PER_DRAW_MAX_NOTE = ("SENSITIVITY: per draw, the maximum over the pair's layers, then its D-156 band; a maximum "
                     "grows with the number of layers (15, 17 or 22 by grid), so it is not comparable in size across "
                     "pairs; ties as TIE_RULE")


def weighted_auroc(pos: np.ndarray, neg: np.ndarray, Wp: np.ndarray, Wn: np.ndarray) -> np.ndarray:
    """(R,) AUROC of positive scores `pos` against negative scores `neg` with per-replicate item
    multiplicities Wp (R, n_pos) and Wn (R, n_neg): the Mann-Whitney U with ties counted half,
    over the resampled multisets. Every sum is of integers (exact in float64), so with unit
    weights the result equals metrics.auroc bit for bit (one correctly rounded division)."""
    order = np.argsort(pos, kind="mergesort")
    ps = pos[order]
    C = np.zeros((Wp.shape[0], ps.size + 1))
    C[:, 1:] = np.cumsum(Wp[:, order], axis=1, dtype=np.float64)
    lo = np.searchsorted(ps, neg, side="left")
    hi = np.searchsorted(ps, neg, side="right")
    tot = C[:, -1]
    E = 2.0 * (tot[:, None] - C[:, hi]) + (C[:, hi] - C[:, lo])       # 2 x (above) + (tied), per negative
    return (Wn * E).sum(axis=1) / (2.0 * tot * Wn.sum(axis=1))


def floored_draws(pool_pre: np.ndarray, pool_post: np.ndarray, parts: list[tuple[np.ndarray, np.ndarray]],
                  target: float) -> np.ndarray:
    """(1 + n_draws,) bool: in that partition, did any seed's realised pre or post FPR equal 0 before
    flooring (twomoment_cells' realised_censoring definition; thresholds as paired.operating_point)."""
    out = []
    for cal, ev in parts:
        T = np.array([P.thr_at_fpr(pool_pre[cal, s], target) for s in range(pool_pre.shape[1])])
        z = ((pool_pre[ev] >= T).sum(axis=0) == 0) | ((pool_post[ev] >= T).sum(axis=0) == 0)
        out.append(bool(z.any()))
    return np.array(out)


def sources_now() -> dict[str, str]:
    """The registry's sources re-read from disk (D-189): every ORIGINAL extraction record (reruns
    excluded, as the registry excludes them) and every committed pin manifest, with their sha256."""
    out = {f"results/{f.name}": sha256_file(f) for f in sorted(RESULTS.glob("extract__*.json")) if "__rerun_" not in f.name}
    for m in REG.MANIFESTS:
        if m.exists():
            out[str(m.relative_to(ROOT)).replace("\\", "/")] = sha256_file(m)
    return out


def registry_stem(depth_stem: str) -> str:
    return depth_stem if depth_stem in REG.DEPTH_ONLY else "repair_" + depth_stem[len("depth_"):]


def out_name(depth_stem: str) -> str:
    return f"depth_d156_{depth_stem[len('depth_'):]}"


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Depth curves under D-156 (D-169 item 4, D-180)")
    ap.add_argument("--pair", required=True, choices=POP.UPDATE_PAIRS_DEPTH + LORA_DEPTH + CONTROL_DEPTH)
    ap.add_argument("--selection", default=None, help="default: the pair's selection of record (D-171, D-224)")
    ap.add_argument("--seeds", type=int, nargs="*", default=REGISTERED["seeds"])
    ap.add_argument("--n-draws", type=int, default=REGISTERED["n_draws"])
    ap.add_argument("--target-fpr", type=float, default=None)
    ap.add_argument("--n-boot", type=int, default=REGISTERED["n_boot"])
    ap.add_argument("--fpr5", action="store_true",
                    help="4e (D-152, D-248): the registered 5%% arm, written as depth_d156_<pair>_fpr5; anchored to the "
                         "5%% upstream files, and every cell's 1%% twin must equal the 1%% depth record exactly")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.fpr5 and args.target_fpr not in (None, TARGET_5):
        raise SystemExit("REFUSING: --fpr5 fixes the target at 0.05")
    args.target_fpr = TARGET_5 if args.fpr5 else (REGISTERED["target_fpr"] if args.target_fpr is None else args.target_fpr)
    stem = args.pair
    reg_stem = registry_stem(stem)
    sel_stem = SELECTION_STEM.get(stem, reg_stem)
    lora = stem in LORA_DEPTH
    sel_of_record = SELECTION_OF_RECORD_LORA if lora else SELECTION_OF_RECORD
    if args.selection is None:
        args.selection = str(sel_of_record)
    registered = (args.n_draws == REGISTERED["n_draws"] and list(args.seeds) == REGISTERED["seeds"]
                  and args.n_boot == REGISTERED["n_boot"]
                  and args.target_fpr == (TARGET_5 if args.fpr5 else REGISTERED["target_fpr"]))
    final_path = (RESULTS / f"{out_name(stem)}{'_fpr5' if args.fpr5 else ''}.json").resolve()
    rec1: dict[tuple[int, str], dict[str, Any]] = {}
    rec1_info: dict[str, Any] = {}
    if args.fpr5:
        rel1 = f"results/{out_name(stem)}.json"
        if not (ROOT / rel1).is_file():
            raise SystemExit(f"REFUSING: {rel1}, the 1% record the 5% run anchors on, is not on disk (D-248)")
        rb = (ROOT / rel1).read_bytes()
        rec1 = {(c["layer"], c["probe_set"]): c for c in json.loads(rb.decode("utf-8"))["cells"]}
        rec1_info = {"file": rel1, "sha256": sha256_bytes(rb)}
    n_self_anchor = 0
    out_path = Path(args.out).resolve() if args.out else final_path
    final = out_path == final_path
    if out_path.exists():
        raise SystemExit(f"REFUSING: {out_path} exists; results are never overwritten (D-152 P5)")
    if out_path.is_relative_to(RESULTS.resolve()) and not final:        # any depth under results/ (D-184)
        raise SystemExit("REFUSING: a test run (--out) must not write under results/; only the pair's own final name may")
    sel_path = Path(args.selection).resolve()
    if final:
        threads = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
        problems = []
        if not registered:
            problems.append("non-registered --n-draws/--target-fpr/--seeds/--n-boot")
        if sel_path != sel_of_record.resolve():
            problems.append(f"selection {sel_path.name} is not the selection of record (D-171)")
        if any(t != "6" for t in threads.values()):
            problems.append(f"no 6-thread BLAS pin (D-152 P6): {threads}")
        if problems:
            raise SystemExit(f"REFUSING the final name: {problems}; use --out outside results/ for a test")
    sel_sha, cl_sha = sha256_file(sel_path), sha256_file(CLUSTERS)
    code0 = P.code_block(CODE_FILES)

    sel_doc = json.load(open(sel_path, encoding="utf-8"))
    if list(sel_doc["config"]["seeds"]) != list(args.seeds) or float(sel_doc["config"]["holdout"]) != P.HOLDOUT:
        raise SystemExit("REFUSING: selection seeds/holdout differ from this run's")
    sel = next((m for m in sel_doc["models"] if m["pair"] == sel_stem), None)
    if sel is None or not sel["selected"]["arm"].startswith("C="):
        raise SystemExit(f"REFUSING: no logistic-C selection for {sel_stem} in {sel_path.name}")
    C_sel, L_sel = float(sel["selected"]["arm"].split("=")[1]), int(sel["selected"]["layer"])

    fit_path = TWOMOMENT_FIT_FPR5 if args.fpr5 else TWOMOMENT_FIT       # D-248: the 5% fit names the 5% 4b files
    dmg_path = DAMAGE_FPR5 if args.fpr5 else DAMAGE_OF_RECORD
    fit_sha = sha256_file(fit_path) if fit_path.exists() else None
    if lora:
        # D-224: the LoRA arms have no 4b file; the anchor is their 4a cell in damage_d156_r2 at the selected
        # layer (same probes, pools and draws), both probe sets, both levels (at 5%: the 5% damage file, D-248)
        run = stem[len("depth_"):]
        tm_rel = str(dmg_path.relative_to(ROOT)).replace("\\", "/")
        tm_sha = sha256_file(dmg_path)
        if final and not args.fpr5 and tm_sha != DAMAGE_OF_RECORD_SHA256:
            raise SystemExit(f"REFUSING: {tm_rel} is not the damage file of record ({tm_sha[:16]})")
        tm_doc = json.load(open(dmg_path, encoding="utf-8"))
        if args.fpr5 and ((tm_doc.get("fpr5_self_anchor") or {}).get("n_cells_identical_at_1pct") != len(tm_doc["cells"])
                          or tm_doc["config"]["target_fpr"] != TARGET_5):
            raise SystemExit(f"REFUSING: {tm_rel} is not a complete, self-anchored 5% damage file (D-248)")
        # the damage run's selection bytes (its config.fixed_inputs_sha256), compared by hash only
        if sel_sha not in (tm_doc["config"].get("fixed_inputs_sha256") or {}).values():
            raise SystemExit(f"REFUSING: {tm_rel} was made on another selection than {sel_path.name}")
        dcells = {c["probe_set"]: c for c in tm_doc["cells"] if c["scope"] == "4a" and c["cell"] == run}
        if set(dcells) != set(PROBE_SETS) or any(c["layer"] != L_sel for c in dcells.values()) or \
                dcells["selected"]["probe_C"] != C_sel:
            raise SystemExit(f"REFUSING: {tm_rel} has no 4a cells of {run} at the selected layer {L_sel} and C {C_sel:g}")
        tm_cells = {(L_sel, p): c for p, c in dcells.items()}
        tm_cfg = None
        full_check = True
        check_kind = ("the run's 4a operating point in damage_d156_r2 at the selected layer, both probe sets and "
                      "levels (D-224); no reference at other layers (no 4b file for the LoRA arms)")
    else:
        # The twomoment file of record for the cross-check: this pair's own, or (the dense pair, the GPU
        # control) the Mistral pair's whose pre side it shares. Its bytes must be those the D-174 fit read.
        tm_rel = f"results/twomoment_{sel_stem}{'_fpr5' if args.fpr5 else ''}.json"
        if fit_sha is None:
            raise SystemExit(f"REFUSING: {fit_path.name}, which names the 4b files of record, is not on disk")
        if final and not args.fpr5 and fit_sha != TWOMOMENT_FIT_SHA256:
            raise SystemExit(f"REFUSING: results/twomoment_fit.json is not the D-174 fit ({fit_sha[:16]})")
        fit_doc = json.load(open(fit_path, encoding="utf-8"))
        if args.fpr5 and ("D-248" not in str(fit_doc.get("preregistration")) or (fit_doc.get("config") or {}).get("allow_unregistered")):
            raise SystemExit(f"REFUSING: {fit_path.name} is not a registered D-248 fit")
        if not (ROOT / tm_rel).exists():
            raise SystemExit(f"REFUSING: {tm_rel}, the cross-check's file of record, is not on disk")
        tm_sha = sha256_file(ROOT / tm_rel)
        if tm_sha != fit_doc["input_files_sha256"].get(tm_rel):
            raise SystemExit(f"REFUSING: {tm_rel} is not the file {fit_path.name} read")
        tm_doc = json.load(open(ROOT / tm_rel, encoding="utf-8"))
        if args.fpr5 and (tm_doc.get("fpr5_self_anchor") or {}).get("n_cells_identical_at_1pct") != len(tm_doc["cells"]):
            raise SystemExit(f"REFUSING: {tm_rel} is not a complete, self-anchored 5% 4b record (D-248)")
        tm_cfg = tm_doc["config"]
        if (tm_cfg["selected_C"], tm_cfg["selected_layer"], tm_cfg["selection_sha256"], list(tm_cfg["seeds"])) != \
                (C_sel, L_sel, sel_sha, list(args.seeds)):
            raise SystemExit(f"REFUSING: {tm_rel} was run on another selection or seeds")
        tm_cells = {(c["layer"], c["probe_set"]): c for c in tm_doc["cells"]}
        full_check = sel_stem == reg_stem
        check_kind = "full operating-point report and floored share" if full_check else \
            "pre-only quantities (the shared pre side)"

    cfg = REG.run(reg_stem)
    axis = cfg["axis"]
    if axis not in P.AXIS_DRAW_SEED:
        raise SystemExit(f"REFUSING: no D-156 partition seed for axis {axis}")
    data: dict[str, dict[str, Any]] = {"pre": {}, "post": {}}
    input_sha: dict[str, str] = {}
    input_recs: dict[str, dict[str, Any]] = {}
    for side in data:
        for part in ("probe_train", "threshold_calib", "eval"):
            z, r = REG.load(cfg[f"{side}_repo"], axis, part)       # expected bytes checked there
            data[side][part] = z
            input_sha[f"{side}/{part}"] = r["sha256"]
            input_recs[f"{side}/{part}"] = r
    reg_sources = REG.sources_sha256()          # the records and pins the expectations above came from
    if sources_now() != reg_sources:            # re-read from disk: the registry caches its hashes (D-189)
        raise SystemExit("REFUSING: the extraction records or pins on disk are not the ones the registry read")
    # The selection was made on these pre bytes (for the dense pair: on the Mistral pair's, D-180).
    bad = [f"{k}: have {input_sha[k][:16]}, selection read {(sel_doc['input_cache_sha256'].get(f'{sel_stem}:{k}') or 'NOT RECORDED')[:16]}"
           for k in ("pre/probe_train", "pre/threshold_calib")
           if input_sha[k] != sel_doc["input_cache_sha256"].get(f"{sel_stem}:{k}")]
    if bad:
        raise SystemExit("REFUSING: the selection was made on other bytes:\n  " + "\n  ".join(bad))
    anchor_key = (lambda k: f"{run}:{k}") if lora else (lambda k: k)      # damage_d156_r2 prefixes its keys
    if any(tm_doc["input_cache_sha256"].get(anchor_key(k)) != input_sha[k] for k in ("pre/threshold_calib", "pre/eval")):
        raise SystemExit(f"REFUSING: {tm_rel} read other pre-update caches")
    if full_check and any(tm_doc["input_cache_sha256"].get(anchor_key(k)) != input_sha[k]
                          for k in ("post/threshold_calib", "post/eval")):
        raise SystemExit(f"REFUSING: {tm_rel} read other post-update caches")
    verification = REG.verification(input_recs)
    tokenisation: dict[str, Any] = {}
    for part in ("probe_train", "threshold_calib", "eval"):
        a, b = data["pre"][part], data["post"][part]
        if not np.array_equal(a["item_ids"], b["item_ids"]) or not np.array_equal(a["labels"], b["labels"]):
            raise SystemExit(f"REFUSING: {part} items or labels differ between pre and post")
        if all(k in a.files and k in b.files for k in ("n_tokens", "truncated")):
            tokenisation[part] = {"n_items_n_tokens_differ": int((a["n_tokens"] != b["n_tokens"]).sum()),
                                  "n_items_truncated_differ": int((a["truncated"] != b["truncated"]).sum()),
                                  "max_n_tokens": {"pre": int(a["n_tokens"].max()), "post": int(b["n_tokens"].max())},
                                  "n_truncated": {"pre": int(a["truncated"].sum()), "post": int(b["truncated"].sum())}}
        elif final:
            raise SystemExit(f"REFUSING: {part} caches lack n_tokens/truncated; tokenisation cannot be compared")
        else:
            tokenisation[part] = "not in these caches (test inputs)"
    for side in data:
        for part, z in data[side].items():
            if "nonfinite" in z.files and z["nonfinite"].any():
                raise SystemExit(f"REFUSING: {side}/{part} has non-finite items")
    grids = [sorted(int(k.split("_")[1]) for k in z.files if k.startswith("layer_"))
             for side in data for z in data[side].values()]
    if any(g != grids[0] for g in grids):
        raise SystemExit(f"REFUSING: the input caches have different layer grids: {grids}")
    layers = grids[0]
    if L_sel not in layers:
        raise SystemExit(f"REFUSING: selected layer {L_sel} is not in the caches' grid")
    if not lora and sorted({li for li, _ in tm_cells}) != layers:
        raise SystemExit(f"REFUSING: {tm_rel} covers other layers than this pair's grid")

    y_tr = data["pre"]["probe_train"]["labels"].astype(bool)
    ncal = ~data["pre"]["threshold_calib"]["labels"].astype(bool)
    y_ev = data["pre"]["eval"]["labels"].astype(bool)
    neg, pos = ~y_ev, y_ev
    n_neg, n_pos = int(neg.sum()), int(pos.sum())
    pooled_ids = [str(i) for i in data["pre"]["threshold_calib"]["item_ids"][ncal]] + \
                 [str(i) for i in data["pre"]["eval"]["item_ids"][neg]]
    n_cal, n_pool = int(ncal.sum()), len(pooled_ids)
    pooled_sha = hashlib.sha256("\n".join(pooled_ids).encode()).hexdigest()
    # the pool's identity: against the 4b file's; for the LoRA arms (no 4b file), the cluster manifest's
    # ordered item-id hash below is the check, the one damage_d156_r2 itself passed (D-224)
    if not lora and pooled_sha != tm_cfg["pooled_negatives"]["item_ids_sha256"]:
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
    # D-152 P5 item bootstrap for the threshold-free AUROC: the same resamples at every layer.
    CNT_N = P.counts_matrix(n_neg, args.n_boot, ITEM_BOOT_SEED["eval_negatives"])
    CNT_P = P.counts_matrix(n_pos, args.n_boot, ITEM_BOOT_SEED["eval_positives"])
    ONE_N, ONE_P = np.ones((1, n_neg)), np.ones((1, n_pos))
    max_layer = max(layers)
    print(f"{stem} [{axis}] via {reg_stem}: selection of {sel_stem} (C={C_sel:g}, layer {L_sel}); {len(layers)} layers; "
          f"pooled negatives {n_cal} + {n_pool - n_cal}, {n_pos} positives; {args.n_draws} draws per level; "
          f"item bootstrap B={args.n_boot}; cross-check vs {tm_rel}: {check_kind}; "
          f"D-170 flag: {verification['flag']}; registered: {registered}", flush=True)

    done: dict[tuple[int, str], dict[str, Any]] = {}
    per_draw: dict[tuple[str, str, str], dict[int, np.ndarray]] = {}   # (pset, level, quantity) -> {layer: (1 + n_draws,)}
    drop_reps: dict[str, dict[int, np.ndarray]] = {s: {} for s in PROBE_SETS}   # item-bootstrap replicates
    n_checked = 0
    n_auroc_checked = 0
    for li in [L_sel] + [x for x in layers if x != L_sel]:
        k = f"layer_{li}"
        # each array read once per layer (an NpzFile access decompresses the whole array again)
        X = {s: {p: data[s][p][k] for p in ("probe_train", "threshold_calib", "eval")} for s in data}
        cka = float(linear_cka(X["pre"]["eval"].astype(np.float64), X["post"]["eval"].astype(np.float64)))
        fits: dict[str, list[P.Probe]] = {s: [] for s in PROBE_SETS}
        refits: dict[str, list[P.Probe]] = {s: [] for s in PROBE_SETS}
        for seed in args.seeds:
            _, fit = P.d117_split(len(y_tr), seed)
            fo = np.sort(fit)
            for pset, rows, C, order in (("selected", fit, C_sel, "d117_permutation"), ("C=1", fo, 1.0, "file_order")):
                fits[pset].append(P.fit_probe(X["pre"]["probe_train"][rows], y_tr[rows], seed, C, order))
                refits[pset].append(P.fit_probe(X["post"]["probe_train"][rows], y_tr[rows], seed, C, order))

        def scores(probes: list[P.Probe]) -> dict[str, np.ndarray]:
            S = lambda side, part, m: np.column_stack([q.score(X[side][part][m]) for q in probes])  # noqa: E731
            return {"cal_pre": S("pre", "threshold_calib", ncal), "ev_pre": S("pre", "eval", neg),
                    "cal_post": S("post", "threshold_calib", ncal), "ev_post": S("post", "eval", neg),
                    "pos_pre": S("pre", "eval", pos), "pos_post": S("post", "eval", pos)}

        for pset in PROBE_SETS:
            sc = scores(fits[pset])
            # the post-refit probe is only ever applied to the post model's eval scores
            RS = lambda m: np.column_stack([q.score(X["post"]["eval"][m]) for q in refits[pset]])  # noqa: E731
            rf = {"ev_post": RS(neg), "pos_post": RS(pos)}
            pool_pre, pool_post = np.vstack([sc["cal_pre"], sc["ev_pre"]]), np.vstack([sc["cal_post"], sc["ev_post"]])
            reports = {lvl: P.repartition_estimate(sc["cal_pre"], sc["ev_pre"], sc["cal_post"], sc["ev_post"],
                                                   sc["pos_pre"], sc["pos_post"], draws[lvl], args.target_fpr)
                       for lvl in LEVELS}
            floored = {lvl: floored_draws(pool_pre, pool_post, parts[lvl], args.target_fpr) for lvl in LEVELS}
            if args.fpr5 and registered:
                # D-248 self-anchor: this cell's operating point at 1%, both levels, equals the 1% depth record's
                r1c = rec1.get((li, pset))
                if r1c is None:
                    raise SystemExit(f"REFUSING: {rec1_info['file']} has no cell L{li} {pset}")
                for lvl, want in ((LEVELS[0], r1c["operating_point"]), (LEVELS[1], r1c[LEVELS[1]]["operating_point"])):
                    rep1, _ = P.repartition_estimate(sc["cal_pre"], sc["ev_pre"], sc["cal_post"], sc["ev_post"],
                                                     sc["pos_pre"], sc["pos_post"], draws[lvl], P.TARGET_FPR)
                    if json.dumps(json.loads(json.dumps(rep1, default=str)), sort_keys=True) != json.dumps(want, sort_keys=True):
                        raise SystemExit(f"REFUSING: layer {li} {pset} {lvl}: the 1% twin is not {rec1_info['file']}'s (D-248)")
                n_self_anchor += 1
            if lora:
                # D-224: the damage 4a cell holds the item-level report at its top level and the cluster
                # level beside it; it exists at the selected layer only, and records no floored share
                tm = tm_cells.get((li, pset))
                ref = None if tm is None else {LEVELS[0]: (tm, None), LEVELS[1]: (tm[LEVELS[1]], None)}
            else:
                tm = tm_cells[(li, pset)]
                ref = {LEVELS[0]: (tm["realised"], tm["realised_censoring"]["share_of_draws_floored"]),
                       LEVELS[1]: (tm[LEVELS[1]]["realised"], tm[LEVELS[1]]["realised_censoring"]["share_of_draws_floored"])}
            for lvl in (LEVELS if ref else ()):
                mine = json.loads(json.dumps(reports[lvl][0], default=str))
                want, want_fl = ref[lvl]
                if lora:
                    want = {key: want[key] for key in mine if key in want}
                if full_check:
                    diff = sorted(set(mine) ^ set(want)) + [key for key in mine if key in want and mine[key] != want[key]]
                    if want_fl is not None and float(floored[lvl][1:].mean()) != want_fl:
                        diff.append("realised_censoring.share_of_draws_floored")
                else:          # the dense pair: only what depends on the pre side alone
                    diff = [key for key in PRE_ONLY_METRICS if mine[key] != want[key]] + \
                           [key for key in PRE_ONLY_COUNTS
                            if mine["frozen_split_counts"][key] != want["frozen_split_counts"][key]]
                if diff:
                    raise SystemExit(f"REFUSING: layer {li} {pset} {lvl}: differs from {tm_rel} in {diff}")
                n_checked += 1

            # AUROC on the frozen eval split, item bootstrap (D-152 P5); unit weights = metrics.auroc
            au: dict[str, Any] = {"note": AUROC_NOTE}
            reps: dict[str, np.ndarray] = {}
            points: dict[str, np.ndarray] = {}
            for name, (pk, nk, src) in {"pre": ("pos_pre", "ev_pre", sc), "frozen": ("pos_post", "ev_post", sc),
                                        "post_refit": ("pos_post", "ev_post", rf)}.items():
                pt = np.array([M.auroc(src[pk][:, s], src[nk][:, s]).value for s in range(len(args.seeds))])
                unit = np.array([weighted_auroc(src[pk][:, s], src[nk][:, s], ONE_P, ONE_N)[0]
                                 for s in range(len(args.seeds))])
                if not np.array_equal(pt, unit):
                    raise SystemExit(f"REFUSING: layer {li} {pset} {name}: the weighted AUROC is not metrics.auroc "
                                     f"bit for bit ({np.max(np.abs(pt - unit)):.2e})")
                n_auroc_checked += len(args.seeds)
                points[name] = pt
                reps[name] = np.mean([weighted_auroc(src[pk][:, s], src[nk][:, s], CNT_P, CNT_N)
                                      for s in range(len(args.seeds))], axis=0)
            points["drop_frozen"] = points["pre"] - points["frozen"]
            reps["drop_frozen"] = reps["pre"] - reps["frozen"]
            for name in ("pre", "frozen", "post_refit", "drop_frozen"):
                au[name] = {"mean_over_seeds": float(points[name].mean()), "per_seed": points[name].tolist(),
                            "range_over_seeds": [float(points[name].min()), float(points[name].max())],
                            **P.ci(reps[name])}
            drop_reps[pset][li] = np.concatenate([[points["drop_frozen"].mean()], reps["drop_frozen"]])

            cell: dict[str, Any] = {"pair": stem, "registry_pair": reg_stem, "axis": axis, "layer": li,
                                    "relative_depth": li / max_layer if max_layer else None, "probe_set": pset,
                                    "probe_C": C_sel if pset == "selected" else 1.0,
                                    "fit_row_order": "d117_permutation" if pset == "selected" else "file_order",
                                    "is_selected_layer": li == L_sel, "input_verification": verification,
                                    "cka_frozen_eval_split": cka, "auroc": au}
            for lvl in LEVELS:
                report, arr = reports[lvl]
                # D-180 frozen curve: per draw |log(FPR_post / target)|, FPR_post the seed-mean of floored
                # per-seed FPRs (ratio_to_target); 4d's Table 2 frozen column must equal it at shared cells
                rt = np.concatenate([[report["ratio_to_target"]["frozen"]], arr["ratio_to_target"]]).astype("<f8")
                fc = P.abs_log(rt).astype("<f8")
                block = {"operating_point": report,
                         "frozen_curve_abs_log_ratio_to_target": {
                             **P.band_summary(fc[1:], fc[0]), "note": FROZEN_CURVE_NOTE,
                             # fingerprints for 4d's bit-for-bit check (frozen split first, then draws in order)
                             "per_draw_sha256": sha256_bytes(fc.tobytes()),
                             "per_draw_ratio_to_target_sha256": sha256_bytes(rt.tobytes())},
                         "realised_censoring": {"share_of_draws_floored": float(floored[lvl][1:].mean()),
                                                "frozen_split_floored": bool(floored[lvl][0])}}
                if lvl == LEVELS[0]:
                    cell.update(block)
                else:
                    cell[lvl] = {"note": CLUSTER_NOTE, **block}
                for q in ("ratio_to_target", "paired_ratio"):
                    per_draw.setdefault((pset, lvl, q), {})[li] = np.concatenate([[report[q]["frozen"]], arr[q]])
            cell["probe_fit_diagnostics"] = [q.diag for q in fits[pset]]
            cell["post_refit_fit_diagnostics"] = [q.diag for q in refits[pset]]
            done[(li, pset)] = cell
        c = done[(li, "selected")]
        pr, dr = c["operating_point"]["paired_ratio"], c["auroc"]["drop_frozen"]
        print(f"  layer {li:>2} [selected C]: paired ratio {pr['mean']:8.3f} [{pr['band95'][0]:.3f}, {pr['band95'][1]:.3f}] "
              f"| AUROC drop {dr['mean_over_seeds']:+.4f} {dr['ci95']} | CKA {cka:.5f}", flush=True)

    def at(li: int, pset: str, lvl: str) -> dict[str, Any]:
        return done[(li, pset)] if lvl == LEVELS[0] else done[(li, pset)][lvl]

    def layer_row(pset: str, lvl: str, li: int) -> dict[str, Any]:
        """One layer's Table 1 entries: paired ratio, ratio to target, TPR pre/post/delta, the floor
        (the no-update value at that layer, D-156), floored share."""
        op = at(li, pset, lvl)["operating_point"]
        return {"layer": li, **{q: op[q] for q in ("paired_ratio", "ratio_to_target", "tpr_pre", "tpr_post", "delta_tpr",
                                                   "floor_ratio_to_target", "floor_abs_log")},
                "share_of_draws_floored": at(li, pset, lvl)["realised_censoring"]["share_of_draws_floored"]}

    def pick(m: np.ndarray, how: str) -> tuple[int, int]:
        """(layer, number of layers tied with it) for the argmax/argmin of the per-layer means m;
        ties (within rtol 1e-12) go to the lowest layer (TIE_RULE)."""
        target = m.max() if how == "max" else m.min()
        tied = np.isclose(m, target, rtol=1e-12, atol=0.0)
        return int(layers[int(np.argmax(tied))]), int(tied.sum())

    def per_draw_max(Q: np.ndarray) -> dict[str, Any]:
        """Q: (layers ascending, 1 + n_draws). Per draw the maximum over layers (sensitivity)."""
        worst = Q.max(axis=0)
        near = np.isclose(Q, worst[None, :], rtol=1e-12, atol=0.0)     # sums equal in exact arithmetic
        arg = np.array(layers)[near.argmax(axis=0)]
        tied = near.sum(axis=0) > 1
        vals, counts = np.unique(arg[1:], return_counts=True)
        return {"note": PER_DRAW_MAX_NOTE, "band": P.band_summary(worst[1:], worst[0]),
                "layer_at_frozen_split": int(arg[0]), "tied_at_frozen_split": bool(tied[0]),
                "modal_layer_over_draws": int(vals[counts.argmax()]),
                "share_of_draws_at_modal_layer": float(counts.max() / counts.sum()), "n_draws_tied": int(tied[1:].sum())}

    summary: dict[str, Any] = {"table_1_rule": "D-180 (author 2)"}
    for pset in PROBE_SETS:
        summary[pset] = {}
        for lvl in LEVELS:
            means = {q: np.array([at(li, pset, lvl)["operating_point"][q]["mean"] for li in layers])
                     for q in ("paired_ratio", "ratio_to_target")}
            floor_means = np.array([at(li, pset, lvl)["operating_point"]["floor_ratio_to_target"]["mean"] for li in layers])
            (l_pr, t_pr), (l_rt, t_rt), (l_fl, t_fl) = pick(means["paired_ratio"], "max"), \
                pick(means["ratio_to_target"], "max"), pick(floor_means, "max")
            row = lambda li, n_tied: {**layer_row(pset, lvl, li), "n_layers_tied": n_tied}  # noqa: E731
            blk: dict[str, Any] = {
                "n_layers": len(layers), "tie_rule": TIE_RULE,
                # HEADLINE ROW: the D-117 selected layer; paired ratio, with ratio to target and TPR beside it
                "headline_selected_layer": layer_row(pset, lvl, L_sel),
                # SECOND ROW: the sweep maximum, with its own no-update null (D-156 "the floor"); each
                # quantity takes its own maximising layer (D-183)
                "sweep_maximum": {
                    "paired_ratio": {**row(l_pr, t_pr),
                                     "null_no_update": {"value": 1.0, "layer": None,
                                                        "rule": "post = pre: the paired ratio is 1 at every layer and "
                                                                "draw, so its maximum over layers is 1 (D-156 floor, D-180)"}},
                    "ratio_to_target": {**row(l_rt, t_rt),
                                        "null_no_update": {"value": float(floor_means[layers.index(l_fl)]), "layer": l_fl,
                                                           "n_layers_tied": t_fl,
                                                           "rule": "the maximum over layers of the per-layer D-156 mean of "
                                                                   "FPR_pre / target (D-156 floor, D-180)"}}},
                "lowest_layer_by_mean": {q: row(*pick(m, "min")) for q, m in means.items()},
                "two_sided_largest_abs_log_mean": {q: row(*pick(np.abs(np.log(m)), "max")) for q, m in means.items()},
                "per_draw_max_over_layers_SENSITIVITY": {
                    q: {"n_layers": len(layers), **per_draw_max(np.stack([per_draw[(pset, lvl, q)][li] for li in layers]))}
                    for q in ("paired_ratio", "ratio_to_target")}}
            if lvl == LEVELS[1]:
                blk["note"] = CLUSTER_NOTE
            summary[pset][lvl] = blk
        D = np.stack([drop_reps[pset][li] for li in layers])         # (layers, 1 + B): point, then replicates
        summary[pset]["auroc_drop_frozen"] = {
            "note": AUROC_NOTE,
            "mean_over_layers": {"point": float(D[:, 0].mean()), **P.ci(D[:, 1:].mean(axis=0))},
            "max_over_layers": {"point": float(D[:, 0].max()), "layer": int(layers[int(np.argmax(D[:, 0]))]),
                                **P.ci(D[:, 1:].max(axis=0))}}
        summary[pset]["min_cka_over_layers"] = min(done[(li, pset)]["cka_frozen_eval_split"] for li in layers)

    # D-224: the LoRA anchor must actually have been compared: both probe sets at both levels at L_sel
    if lora and n_checked != len(PROBE_SETS) * len(LEVELS):
        raise SystemExit(f"REFUSING to write: {n_checked} anchor comparisons with {tm_rel}, not "
                         f"{len(PROBE_SETS) * len(LEVELS)} (both probe sets, both levels, L{L_sel})")

    # Nothing the run read may have changed while it ran: code, selection, cluster manifest, the
    # twomoment file and fit, the registry's records and pins, and the input caches themselves.
    code1 = P.code_block(CODE_FILES)
    changed = [n for n, a, b in (("code", code1["sha256"], code0["sha256"]),
                                 ("selection", sha256_file(sel_path), sel_sha),
                                 ("cluster manifest", sha256_file(CLUSTERS), cl_sha),
                                 ("twomoment file", sha256_file(ROOT / tm_rel), tm_sha),
                                 (fit_path.name, sha256_file(fit_path) if fit_path.exists() else None, fit_sha),
                                 ("registry sources (re-read from disk)", sources_now(), reg_sources)) if a != b]
    changed += [f"cache {r['file']}" for r in input_recs.values()
                if sha256_file(resolve_cache(r["file"], REG.CACHE)) != r["sha256"]]
    if changed:
        raise SystemExit(f"REFUSING to write: changed during the run: {changed}")
    if args.fpr5 and registered and n_self_anchor != len(layers) * len(PROBE_SETS):
        raise SystemExit(f"REFUSING to write: {n_self_anchor} self-anchored cells of {len(layers) * len(PROBE_SETS)} (D-248)")
    cells = [done[(li, pset)] for li in layers for pset in PROBE_SETS]
    rel = lambda p: str(p.relative_to(ROOT)).replace("\\", "/") if p.is_relative_to(ROOT) else p.name  # noqa: E731
    payload = {**provenance_block("depth_d156"),
               "preregistration": "D-169 item 4, D-152 P5, D-156, D-164, D-165, D-180" + ("; 4e: D-152, D-248 (the 5% arm)" if args.fpr5 else ""),
               **({"fpr5_self_anchor": {**rec1_info, "n_cells_identical_at_1pct": n_self_anchor,
                                        "kind": "every cell's operating point recomputed at 1% equals the 1% depth record "
                                                "at both levels (D-248)"}} if args.fpr5 else {}),
               "code": code0, "registered_parameters": registered, "input_verification": verification,
               "config": {"pair": stem, "registry_pair": reg_stem, "axis": axis,
                          "selection_file": rel(sel_path), "selection_sha256": sel_sha, "selection_of_pair": sel_stem,
                          "selected_C": C_sel, "selected_layer": L_sel, "layers": layers, "seeds": args.seeds,
                          "split": f"D-117 (holdout {P.HOLDOUT})", "target_fpr": args.target_fpr,
                          "probe_sets": {"selected": "D-117 selected C, permutation order (primary)",
                                         "C=1": "C=1, file order (continuity and sensitivity row)"},
                          "post_refit": "refitted on the post model's probe_train with the same rows, C and row order",
                          "pre_repo": cfg["pre_repo"], "pre_rev": cfg["pre_rev"],
                          "post_repo": cfg["post_repo"], "post_rev": cfg["post_rev"],
                          "inputs_from": "src/input_registry.py (D-165)", "input_registry_sources_sha256": reg_sources,
                          "cluster_manifest": {"file": "manifests/negative_clusters.json", "sha256": cl_sha},
                          "n_draws": args.n_draws, "draw_seed": P.AXIS_DRAW_SEED[axis],
                          "cluster_draw_seed": P.AXIS_CLUSTER_DRAW_SEED[axis],
                          "item_bootstrap": {"n_boot": args.n_boot, "seeds": ITEM_BOOT_SEED,
                                             "resampled": "eval negatives and eval positives separately",
                                             "applies_to": "AUROC (threshold-free, D-156)"},
                          "pooled_negatives": {"n_calibration": n_cal, "n_evaluation": n_pool - n_cal,
                                               "item_ids_sha256": pooled_sha,
                                               "order": "threshold_calib negatives then eval negatives, cache order"},
                          "n_eval_positives": n_pos, "positives": "eval positives, fixed",
                          "cka": "linear CKA, pre vs post, frozen eval split (all items), partition-free"},
               "tokenisation_pre_vs_post": tokenisation,
               "twomoment_check": {"file": tm_rel, "sha256": tm_sha, "twomoment_fit_sha256": fit_sha, "kind": check_kind,
                                   "n_reports_identical": n_checked, "levels": list(LEVELS),
                                   "n_auroc_values_equal_to_metrics_auroc": n_auroc_checked},
               "population": {"declared_in": "src/populations.py",
                              "group": "DEPTH_D156_LORA" if lora else "DEPTH_D156_CONTROLS" if stem in CONTROL_DEPTH
                              else "DEPTH_D156_PAIRS",
                              "replaces": None if args.fpr5 else
                                          {"file": f"results/{stem}.json",
                                           "status": POP.SUPERSEDED.get(stem) or
                                           "primary until this regeneration is registered: input hashes recorded, "
                                           "no per-file code hashes (D-169 (8)); superseded then"},
                              **({"role": "4e sensitivity arm at 5% (D-248); replaces nothing"} if args.fpr5 else {})},
               "input_cache_sha256": input_sha,
               "input_cache_source": {k: {"file": r["file"], "expected_from": r["expected_from"],
                                          "pin_only": r["pin_only"]} for k, r in input_recs.items()},
               "pair_summary": summary, "cells": cells}
    digest = write_results(out_path, payload, exclusive=True)
    print(f"Written: {rel(out_path)}\nsha256:  {digest}")
    for pset in PROBE_SETS:
        for lvl in LEVELS:
            b = lambda s: f"{s['mean']:.3f} {[round(v, 3) for v in s['band95']]}"  # noqa: E731
            h, sm = summary[pset][lvl]["headline_selected_layer"], summary[pset][lvl]["sweep_maximum"]
            nl = sm["ratio_to_target"]["null_no_update"]
            print(f"[{pset} {lvl}] headline (selected layer {h['layer']}): paired ratio {b(h['paired_ratio'])}, "
                  f"ratio to target {b(h['ratio_to_target'])}, delta TPR {b(h['delta_tpr'])} | sweep max: paired ratio "
                  f"layer {sm['paired_ratio']['layer']} {b(sm['paired_ratio']['paired_ratio'])} (null 1); ratio to target "
                  f"layer {sm['ratio_to_target']['layer']} {b(sm['ratio_to_target']['ratio_to_target'])} "
                  f"(null {nl['value']:.3f} at layer {nl['layer']})")
        a = summary[pset]["auroc_drop_frozen"]
        print(f"[{pset}] AUROC drop: mean over layers {a['mean_over_layers']['point']:+.4f} {a['mean_over_layers']['ci95']}, "
              f"max {a['max_over_layers']['point']:+.4f} at layer {a['max_over_layers']['layer']} | "
              f"min CKA {summary[pset]['min_cka_over_layers']:.5f}")
    print(f"cross-check vs {tm_rel}: {n_checked} reports identical ({check_kind}); "
          f"{n_auroc_checked} AUROC values equal to metrics.auroc bit for bit; tokenisation {tokenisation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
