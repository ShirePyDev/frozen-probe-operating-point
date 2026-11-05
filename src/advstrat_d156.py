"""Adversarial stratification under D-156 (D-169 item 8; pre-registered in D-186): does a backbone
update hurt a frozen probe more on ADVERSARIAL prompts than on vanilla ones?

Replaces the D-161-superseded advstrat_* results (src/adversarial_strat.py). WildGuardMix is the
axis whose items carry an `adversarial` flag (frozen splits, D-051). As before, ONE global
threshold per seed is set on the calibration negatives (all strata): an operator does not run a
threshold per prompt type. The stratification applies to the measurement only.

  PROBES      D-117 selected C in permutation order (primary) and C=1 in file order (continuity),
              every layer of the pair's grid (D-180 pattern); inputs from src/input_registry.py and
              the selection of record (D-165, D-171).
  OPERATING   D-156 at the item level and the D-164 cluster level beside it: per draw the global
  POINTS      per-seed thresholds on the draw's calibration part; then, PER STRATUM, on the draw's
              evaluation-part negatives of that stratum, FPR pre/post (floored at 1/n of that
              stratum in that draw), paired ratio, ratio to target, the floor |log(FPR_pre/target)|,
              the frozen curve |log(FPR_post/target)|, and on the fixed eval positives of that
              stratum, TPR pre/post and delta. The CONTRAST adversarial minus vanilla is formed within
              each draw (D-156: between-arm gaps within a draw): log paired ratio, log ratio to
              target, delta TPR.
  AUROC       per stratum on the frozen eval split (threshold-free; D-156 keeps the D-152 P5 item
              bootstrap): pre, frozen (post) and drop, with the adversarial minus vanilla drop, 95%
              intervals from ONE item bootstrap (B = 5000; eval negatives and positives resampled
              separately; seeds of D-186), so the contrast's interval is paired.
  CHECKS      the pooled (all strata) operating point at every layer, probe set and level must
              equal results/twomoment_<pair>.json's realised report exactly (the D-174 file of
              record); the per-stratum flagged counts must sum to the pooled counts on the frozen
              split; the stratum labels come from the frozen splits, whose bytes must equal their
              committed manifests.
  D-282       (D-281 (1), the author's correction to D-156) at the item level, every TPR quantity per stratum (TPR
              pre, TPR post, delta TPR) and the adversarial-minus-vanilla delta TPR get a COMBINED interval beside
              the D-156 band. Replicate r = 1..B is evaluated at draw d_r = 1 + ((r - 1) mod n_draws) with row r of
              an item-bootstrap counts matrix over that stratum's positives. There is one matrix per stratum, from
              the D-186 eval_positives seed offset by the stratum's index in STRATA, and each matrix weights the pre
              and post flags and every seed (paired.combined_tpr_replicates). The cluster level keeps its D-156 band
              only. A run first checks that counts of all ones give back the D-156 draws bit for bit.
  RE-ISSUE    the output is results/advstrat_d156_<pair>[_fpr5]_r2.json. A registered run compares every key of
              the first record (REISSUE_OF, hash checked) with its own payload, after removing the D-282 blocks, and
              refuses to write if any number differs. Only the run's header (utc, git, env, code), the
              preregistration and the population block may differ; config's snapshot of the registry sources may
              have grown, but every source the first record listed must be there with the same hash. The first
              record is superseded, never deleted.
Only registered parameters at the 6-thread pin write the final name.
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
from depth_d156 import ITEM_BOOT_SEED as DEPTH_ITEM_BOOT_SEED, weighted_auroc  # noqa: E402
from provenance import (  # noqa: E402
    ensure_utf8_stdout, provenance_block, read_jsonl, resolve_cache, sha256_bytes, sha256_file, write_results,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FROZEN = ROOT / "data" / "frozen"
SPLIT_MANIFESTS = ROOT / "manifests" / "splits"
CODE_FILES = ["src/advstrat_d156.py", "src/depth_d156.py", "src/input_registry.py", "src/paired.py",
              "src/populations.py", "src/provenance.py", "src/superseded_provenance.py", "src/metrics.py",
              "src/repdist.py"]
CLUSTERS = ROOT / "manifests" / "negative_clusters.json"
SELECTION_OF_RECORD = RESULTS / "selection_sweep_d165.json"
TWOMOMENT_FIT = RESULTS / "twomoment_fit.json"
TWOMOMENT_FIT_SHA256 = "515faab949465d0852cd61dbc1c7805cf3d800c5e2222029e4622261727a3b96"   # D-174
TWOMOMENT_FIT_FPR5 = RESULTS / "twomoment_fit_fpr5.json"             # 4e (D-248): names the 5% 4b records
# D-209: exactly the five fail-open cells of D-208, at the selected C, at 5%; no other cell and no re-selection
D209_LAYERS = {"advstrat_mistral": (32,), "advstrat_llama3": (24, 26, 28, 30)}
LEVELS = ("item_level_D156", "cluster_level_D164")
PROBE_SETS = ("selected", "C=1")
REGISTERED = {"n_draws": P.N_DRAWS, "target_fpr": P.TARGET_FPR, "seeds": [0, 1, 2, 3, 4], "n_boot": 5000}
ITEM_BOOT_SEED = {"eval_negatives": 20260931, "eval_positives": 20260932}      # D-186
PAIRS = {"advstrat_mistral": "repair_WG_mistral_base2instruct",
         "advstrat_llama3": "repair_WG_llama3_base2instruct"}
FIELD = "adversarial"
STRATA = {"True": "adversarial", "False": "vanilla"}
QUANT = ("fpr_pre", "fpr_post", "paired_ratio", "ratio_to_target", "floor_abs_log", "frozen_abs_log",
         "tpr_pre", "tpr_post", "delta_tpr", "mcnemar_neg_p_max", "mcnemar_pos_p_max")
CONTRASTS = {"log_paired_ratio": lambda a, v: np.log(a["paired_ratio"]) - np.log(v["paired_ratio"]),
             "log_ratio_to_target": lambda a, v: np.log(a["ratio_to_target"]) - np.log(v["ratio_to_target"]),
             # D-156 floor of the contrast above (D-188): with one global threshold the strata differ before
             # any update, and that gap is the no-update value of log ratio to target's contrast
             "log_floor_ratio_to_target_NULL": lambda a, v: np.log(a["fpr_pre"]) - np.log(v["fpr_pre"]),
             "delta_tpr": lambda a, v: a["delta_tpr"] - v["delta_tpr"]}
EXPECTED_STRATA = {"adversarial": {"pooled_negatives": 1513, "eval_positives": 896},     # D-186
                   "vanilla": {"pooled_negatives": 1487, "eval_positives": 1104}}
CLUSTER_NOTE = ("cluster-level re-partitions (D-164), reported beside the item level, never instead; its "
                "'frozen' value is the frozen split, which is not a cluster-level draw")
AUROC_NOTE = ("frozen eval split, per stratum; 95% interval by the D-152 P5 item bootstrap (B = 5000, eval "
              "negatives and positives resampled separately, all seeds per resample); the same resamples give "
              "the paired adversarial-minus-vanilla interval")
# D-282: the TPR quantities that get the combined interval, and the first records the _r2 files re-issue
D282_QUANT = ("tpr_pre", "tpr_post", "delta_tpr")
D282_KEY = "item_bootstrap_D282"
D282_NOTE = ("D-282 (D-281 (1)): combined 95% interval, the 2.5-97.5 percentiles over B replicates. Replicate r is "
             "evaluated at re-partition draw d_r = 1 + ((r - 1) mod n_draws) with row r of the stratum's item-bootstrap "
             "counts over its positives, paired within item (the same counts weight pre and post and every seed). "
             "band95_D156 (partition variation only) and the frozen split's value are copied beside it")
REISSUE_OF = {("advstrat_mistral", True): ("results/advstrat_d156_mistral_fpr5.json",
                                           "b1b51aecdd65d3a14883dbbc2504b2771aeb47f8eb606904c5afe5ca4b9ab565"),
              ("advstrat_llama3", True): ("results/advstrat_d156_llama3_fpr5.json",
                                          "601054b1f29989d7c91d7a1990687b2cf01fcd5b2a1f5b93867ee85f7d620adb"),
              ("advstrat_mistral", False): ("results/advstrat_d156_mistral.json",
                                            "707867278116536d62d1e32878f75b8590bc3553f3dc596438c63dc6a236effe"),
              ("advstrat_llama3", False): ("results/advstrat_d156_llama3.json",
                                           "74d2d7883da840542de33bbbe1815bccfb7cdf2f2c9628b247baa3c3ba7c5b9e")}
REISSUE_MAY_DIFFER = ("utc", "git", "env", "code", "preregistration", "population")     # the run's header, not numbers
REISSUE_ADDS = ("supersedes", "d282", "reissue_of")
REG_SOURCES_KEY = "input_registry_sources_sha256"


def sources_now() -> dict[str, str]:
    """The registry's sources re-read from disk (D-189): every ORIGINAL extraction record (reruns
    excluded, as the registry excludes them) and every committed pin manifest, with their sha256."""
    out = {f"results/{f.name}": sha256_file(f) for f in sorted(RESULTS.glob("extract__*.json")) if "__rerun_" not in f.name}
    for m in REG.MANIFESTS:
        if m.exists():
            out[str(m.relative_to(ROOT)).replace("\\", "/")] = sha256_file(m)
    return out


def stratum_labels(part: str, item_ids: np.ndarray) -> tuple[np.ndarray, str]:
    """The adversarial/vanilla label of each cache item, from the frozen split whose bytes must
    equal its committed manifest; (labels in cache order, split sha256)."""
    f = FROZEN / f"safety_wildguard__{part}.jsonl"
    man = json.loads((SPLIT_MANIFESTS / f"safety_wildguard__{part}__manifest.json").read_text(encoding="utf-8"))
    sha = sha256_file(f)
    if sha != man["sha256_of_original"]:
        raise SystemExit(f"REFUSING: {f.name} is not the bytes of its committed manifest")
    flag = {r["item_id"]: str(r.get(FIELD)) for r in read_jsonl(f)}     # the sanctioned reader (F-017)
    lab = np.array([STRATA.get(flag.get(str(i), "MISSING"), "UNKNOWN") for i in item_ids])
    if (lab == "UNKNOWN").any():
        raise SystemExit(f"REFUSING: {int((lab == 'UNKNOWN').sum())} {part} items have no {FIELD} flag")
    return lab, sha


def draw_thresholds(pool_pre: np.ndarray, cal: np.ndarray, target: float) -> np.ndarray:
    """The global per-seed thresholds of one partition, on its calibration part (as paired.operating_point sets them)."""
    return np.array([P.thr_at_fpr(pool_pre[cal, s], target) for s in range(pool_pre.shape[1])])


def without_d282(o: Any) -> Any:
    """A payload with every D-282 block removed, for the comparison with the first record."""
    if isinstance(o, dict):
        return {k: without_d282(v) for k, v in o.items() if k != D282_KEY}
    if isinstance(o, list):
        return [without_d282(v) for v in o]
    return o


def reissue_check(payload: dict[str, Any], r1b: bytes, r1_rel: str, r1_sha: str) -> dict[str, Any]:
    """D-282 (3): the re-issue must reproduce every number of the first record (its bytes r1b, hash checked). Returns
    the record's reissue_of block, or refuses. Only the run's header (REISSUE_MAY_DIFFER) may differ."""
    if sha256_bytes(r1b) != r1_sha:
        raise SystemExit(f"REFUSING to write: {r1_rel} is not the first record ({r1_sha[:8]})")
    r1 = json.loads(r1b.decode("utf-8"))
    mine = without_d282(json.loads(json.dumps(payload, default=str)))
    if set(mine) - set(REISSUE_ADDS) != set(r1):
        raise SystemExit(f"REFUSING to write: top-level keys differ from {r1_rel}: "
                         f"{sorted((set(mine) - set(REISSUE_ADDS)) ^ set(r1))}")
    # the registry's snapshot of the extraction records on disk grows as records are added (two on 2026-09-28);
    # every source the first record listed must still be there with the same hash, and the bytes this run read
    # are pinned by input_cache_sha256, which is compared
    old_src, new_src = r1["config"][REG_SOURCES_KEY], mine["config"][REG_SOURCES_KEY]
    lost = sorted(k for k, h in old_src.items() if new_src.get(k) != h)
    if lost:
        raise SystemExit(f"REFUSING to write: registry sources of {r1_rel} missing or changed: {lost}")
    norm = lambda k, o: {x: y for x, y in o.items() if x != REG_SOURCES_KEY} if k == "config" else o  # noqa: E731
    compared = [k for k in r1 if k not in REISSUE_MAY_DIFFER]
    differ = [k for k in compared
              if json.dumps(norm(k, mine[k]), sort_keys=True) != json.dumps(norm(k, r1[k]), sort_keys=True)]
    if differ:
        sub = {k: sorted(x for x in set(norm(k, mine[k])) | set(norm(k, r1[k]))
                         if json.dumps(norm(k, mine[k]).get(x), sort_keys=True) != json.dumps(norm(k, r1[k]).get(x), sort_keys=True))
               for k in differ if isinstance(mine[k], dict) and isinstance(r1[k], dict)}
        raise SystemExit(f"REFUSING to write: {differ} differ from {r1_rel} (D-282 (3)); differing sub-keys: {sub}")
    return {"file": r1_rel, "sha256": r1_sha, "compared": True, "keys_compared": compared,
            "keys_allowed_to_differ": list(REISSUE_MAY_DIFFER), "n_cells": len(r1["cells"]),
            "method": "the payload serialised as it is written (json.dumps, default=str) and parsed back; each compared "
                      "key must give the same json.dumps(sort_keys=True) text as the first record's",
            "removed_before_comparison": [f"{D282_KEY} (at every depth)", *(f"top-level {k}" for k in REISSUE_ADDS)],
            "config_registry_sources": {
                "compared_as": f"config.{REG_SOURCES_KEY}: every entry of the first record present with the same hash",
                "n_first_record": len(old_src), "n_now": len(new_src), "added_since": sorted(set(new_src) - set(old_src))},
            "result": "every compared key equals the first record exactly, with the D-282 blocks removed"}


def stratum_point(pool_pre: np.ndarray, pool_post: np.ndarray, cal: np.ndarray, ev: np.ndarray,
                  s_pool: np.ndarray, pos_pre: np.ndarray, pos_post: np.ndarray, s_pos: np.ndarray,
                  target: float) -> dict[str, dict[str, float]]:
    """Per stratum, the operating point of ONE partition at the global per-seed thresholds that
    paired.operating_point sets (same function on the same calibration scores)."""
    T = draw_thresholds(pool_pre, cal, target)
    out: dict[str, dict[str, float]] = {}
    for st in STRATA.values():
        e = ev[s_pool[ev] == st]
        F_pre, F_post = pool_pre[e] >= T, pool_post[e] >= T
        q = s_pos == st
        Q_pre, Q_post = pos_pre[q] >= T, pos_post[q] >= T
        f_pre, f_post = P.floored_fpr(F_pre), P.floored_fpr(F_post)
        # D-156 significance in every table (D-188), as paired.operating_point computes it
        b, c = (F_pre & ~F_post).sum(axis=0), (~F_pre & F_post).sum(axis=0)
        bp, cp = (Q_pre & ~Q_post).sum(axis=0), (~Q_pre & Q_post).sum(axis=0)
        out[st] = {"n_neg": int(len(e)), "fpr_pre": float(f_pre.mean()), "fpr_post": float(f_post.mean()),
                   "mcnemar_neg_p_max": float(max(P.mcnemar_exact(int(x), int(y)) for x, y in zip(b, c))),
                   "mcnemar_pos_p_max": float(max(P.mcnemar_exact(int(x), int(y)) for x, y in zip(bp, cp))),
                   "mcnemar_neg_min_achievable_p": float(max(P.min_mcnemar_p(int(x + y)) for x, y in zip(b, c))),
                   "mcnemar_pos_min_achievable_p": float(max(P.min_mcnemar_p(int(x + y)) for x, y in zip(bp, cp))),
                   "neg_discordant_b_c": [[int(x), int(y)] for x, y in zip(b, c)],
                   "pos_discordant_b_c": [[int(x), int(y)] for x, y in zip(bp, cp)],
                   "hits_pre": Q_pre.sum(axis=0).tolist(), "hits_post": Q_post.sum(axis=0).tolist(),
                   "paired_ratio": P.ratio_from_flags(F_pre, F_post),
                   "ratio_to_target": float(f_post.mean() / target),
                   "floor_abs_log": float(P.abs_log(f_pre.mean() / target)),
                   "frozen_abs_log": float(P.abs_log(f_post.mean() / target)),
                   "tpr_pre": float(P.rate(Q_pre).mean()), "tpr_post": float(P.rate(Q_post).mean()),
                   "delta_tpr": P.delta_tpr_from_flags(Q_pre, Q_post),
                   "n_flagged_pre": F_pre.sum(axis=0).tolist(), "n_flagged_post": F_post.sum(axis=0).tolist(),
                   "floored": bool(((P.rate(F_pre) == 0) | (P.rate(F_post) == 0)).any())}
    return out


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Adversarial stratification under D-156 (D-169 item 8, D-186)")
    ap.add_argument("--pair", required=True, choices=list(PAIRS))
    ap.add_argument("--seeds", type=int, nargs="*", default=REGISTERED["seeds"])
    ap.add_argument("--n-draws", type=int, default=REGISTERED["n_draws"])
    ap.add_argument("--target-fpr", type=float, default=None)
    ap.add_argument("--n-boot", type=int, default=REGISTERED["n_boot"])
    ap.add_argument("--selection", default=str(SELECTION_OF_RECORD))
    ap.add_argument("--fpr5", action="store_true",
                    help="D-209 (4e, D-248): the five fail-open cells at 5%%, selected C only, written as "
                         "advstrat_d156_<pair>_fpr5; anchored to the 5%% 4b record, and each cell's 1%% twin to the 1%% one")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.fpr5 and args.target_fpr not in (None, 0.05):
        raise SystemExit("REFUSING: --fpr5 fixes the target at 0.05 (D-209)")
    args.target_fpr = 0.05 if args.fpr5 else (REGISTERED["target_fpr"] if args.target_fpr is None else args.target_fpr)
    stem, reg_stem = args.pair, PAIRS[args.pair]
    if sorted(PAIRS) != sorted(POP.ADVERSARIAL):
        raise SystemExit(f"REFUSING: this script's pairs {sorted(PAIRS)} are not populations.ADVERSARIAL")
    registered = (args.n_draws == REGISTERED["n_draws"] and list(args.seeds) == REGISTERED["seeds"]
                  and args.n_boot == REGISTERED["n_boot"]
                  and args.target_fpr == (0.05 if args.fpr5 else REGISTERED["target_fpr"]))
    final_path = (RESULTS / f"advstrat_d156_{stem[len('advstrat_'):]}{'_fpr5' if args.fpr5 else ''}_r2.json").resolve()
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
                   ([] if fit_sha == TWOMOMENT_FIT_SHA256 else ["twomoment_fit.json is not the D-174 fit"]) + \
                   ([] if not args.fpr5 or TWOMOMENT_FIT_FPR5.exists() else ["no twomoment_fit_fpr5.json (D-248)"])
        if problems:
            raise SystemExit(f"REFUSING the final name: {problems}; use --out outside results/ for a test")
    sel_sha, cl_sha = sha256_file(sel_path), sha256_file(CLUSTERS)
    code0 = P.code_block(CODE_FILES)

    sel_doc = json.load(open(sel_path, encoding="utf-8"))
    if list(sel_doc["config"]["seeds"]) != list(args.seeds) or float(sel_doc["config"]["holdout"]) != P.HOLDOUT:
        raise SystemExit("REFUSING: selection seeds/holdout differ from this run's")
    sel = next((m for m in sel_doc["models"] if m["pair"] == reg_stem), None)
    if sel is None or not sel["selected"]["arm"].startswith("C="):
        raise SystemExit(f"REFUSING: no logistic-C selection for {reg_stem}")
    C_sel, L_sel = float(sel["selected"]["arm"].split("=")[1]), int(sel["selected"]["layer"])

    tm_rel = f"results/twomoment_{reg_stem}.json"
    fit_doc = json.load(open(TWOMOMENT_FIT, encoding="utf-8"))
    if not (ROOT / tm_rel).exists() or sha256_file(ROOT / tm_rel) != fit_doc["input_files_sha256"].get(tm_rel):
        raise SystemExit(f"REFUSING: {tm_rel} is missing or not the file twomoment_fit.json read")
    tm1_cells: dict[tuple[int, str], dict[str, Any]] = {}
    tm1_info: dict[str, Any] = {}
    fit5_sha = None
    if args.fpr5:
        # D-248: the 1% 4b record (named by the fit of record) anchors each cell's 1% twin; the 5% 4b record,
        # named by the 5% fit, is the anchor of the 5% report. Each file read once: the bytes hashed are the bytes used.
        t1b = (ROOT / tm_rel).read_bytes()
        tm1_info = {"file": tm_rel, "sha256": sha256_bytes(t1b)}
        tm1_cells = {(c["layer"], c["probe_set"]): c for c in json.loads(t1b.decode("utf-8"))["cells"]}
        tm_rel = f"results/twomoment_{reg_stem}_fpr5.json"
        f5b = TWOMOMENT_FIT_FPR5.read_bytes()
        fit5_sha = sha256_bytes(f5b)
        fit_doc = json.loads(f5b.decode("utf-8"))
        if "D-248" not in str(fit_doc.get("preregistration")):
            raise SystemExit(f"REFUSING: {TWOMOMENT_FIT_FPR5.name} is not a registered D-248 fit")
        if not (ROOT / tm_rel).exists() or sha256_file(ROOT / tm_rel) != fit_doc["input_files_sha256"].get(tm_rel):
            raise SystemExit(f"REFUSING: {tm_rel} is missing or not the file twomoment_fit_fpr5.json read")
    tm_sha = sha256_file(ROOT / tm_rel)
    tm_doc = json.load(open(ROOT / tm_rel, encoding="utf-8"))
    tm_cfg = tm_doc["config"]
    if args.fpr5 and (tm_doc.get("fpr5_self_anchor") or {}).get("n_cells_identical_at_1pct") != len(tm_doc["cells"]):
        raise SystemExit(f"REFUSING: {tm_rel} is not a complete, self-anchored 5% 4b record (D-248)")
    if (tm_cfg["selected_C"], tm_cfg["selected_layer"], tm_cfg["selection_sha256"], list(tm_cfg["seeds"])) != \
            (C_sel, L_sel, sel_sha, list(args.seeds)):
        raise SystemExit(f"REFUSING: {tm_rel} was run on another selection or seeds")
    tm_cells = {(c["layer"], c["probe_set"]): c for c in tm_doc["cells"]}

    cfg = REG.run(reg_stem)
    axis = cfg["axis"]
    if axis != "safety_wildguard":
        raise SystemExit("REFUSING: the adversarial flag exists on the WildGuardMix axis only")
    data: dict[str, dict[str, Any]] = {"pre": {}, "post": {}}
    input_sha: dict[str, str] = {}
    input_recs: dict[str, dict[str, Any]] = {}
    for side in data:
        for part in ("probe_train", "threshold_calib", "eval"):
            z, r = REG.load(cfg[f"{side}_repo"], axis, part)
            data[side][part] = z
            input_sha[f"{side}/{part}"] = r["sha256"]
            input_recs[f"{side}/{part}"] = r
    reg_sources = REG.sources_sha256()
    if sources_now() != reg_sources:
        raise SystemExit("REFUSING: the extraction records or pins on disk are not the ones the registry read")
    if any(sel_doc["input_cache_sha256"].get(f"{reg_stem}:{k}") != input_sha[k] for k in ("pre/probe_train", "pre/threshold_calib")):
        raise SystemExit("REFUSING: the selection was made on other bytes")
    if any(tm_doc["input_cache_sha256"][k] != input_sha[k] for k in ("pre/threshold_calib", "pre/eval", "post/threshold_calib", "post/eval")):
        raise SystemExit(f"REFUSING: {tm_rel} read other caches")
    verification = REG.verification(input_recs)
    for part in ("probe_train", "threshold_calib", "eval"):
        a, b = data["pre"][part], data["post"][part]
        if not np.array_equal(a["item_ids"], b["item_ids"]) or not np.array_equal(a["labels"], b["labels"]):
            raise SystemExit(f"REFUSING: {part} items or labels differ between pre and post")
        if "nonfinite" in a.files and (a["nonfinite"].any() or b["nonfinite"].any()):
            raise SystemExit(f"REFUSING: {part} has non-finite items")
    grids = [sorted(int(k.split("_")[1]) for k in z.files if k.startswith("layer_")) for side in data for z in data[side].values()]
    if any(g != grids[0] for g in grids) or L_sel not in grids[0] or sorted({li for li, _ in tm_cells}) != grids[0]:
        raise SystemExit("REFUSING: layer grids differ between the caches, the selection and the twomoment file")
    layers = grids[0]

    y_tr = data["pre"]["probe_train"]["labels"].astype(bool)
    ncal = ~data["pre"]["threshold_calib"]["labels"].astype(bool)
    y_ev = data["pre"]["eval"]["labels"].astype(bool)
    neg, pos = ~y_ev, y_ev
    n_neg, n_pos = int(neg.sum()), int(pos.sum())
    lab_cal, sha_cal = stratum_labels("threshold_calib", data["pre"]["threshold_calib"]["item_ids"])
    lab_ev, sha_ev = stratum_labels("eval", data["pre"]["eval"]["item_ids"])
    s_pool = np.concatenate([lab_cal[ncal], lab_ev[neg]])
    s_pos = lab_ev[pos]
    pooled_ids = [str(i) for i in data["pre"]["threshold_calib"]["item_ids"][ncal]] + \
                 [str(i) for i in data["pre"]["eval"]["item_ids"][neg]]
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
    CNT_N = P.counts_matrix(n_neg, args.n_boot, ITEM_BOOT_SEED["eval_negatives"])
    CNT_P = P.counts_matrix(n_pos, args.n_boot, ITEM_BOOT_SEED["eval_positives"])
    # D-282: one counts matrix per stratum over its own positives, the D-186 stream offset by the stratum's index
    D282_SEED = {st: ITEM_BOOT_SEED["eval_positives"] + i for i, st in enumerate(STRATA.values())}
    CNT_ST = {st: P.counts_matrix(int((lab_ev[pos] == st).sum()), args.n_boot, D282_SEED[st]) for st in STRATA.values()}
    n_d282_ones_checked = 0
    s_ev_neg = lab_ev[neg]
    strata_sizes = {st: {"pooled_negatives": int((s_pool == st).sum()), "eval_positives": int((s_pos == st).sum())}
                    for st in STRATA.values()}
    split_files = {f"data/frozen/safety_wildguard__{p}.jsonl": s for p, s in (("threshold_calib", sha_cal), ("eval", sha_ev))}
    man_sha = {f"manifests/splits/safety_wildguard__{p}__manifest.json":
               sha256_file(SPLIT_MANIFESTS / f"safety_wildguard__{p}__manifest.json") for p in ("threshold_calib", "eval")}
    if final and (strata_sizes != EXPECTED_STRATA or verification["flag"] is not None):
        raise SystemExit(f"REFUSING the final name: strata {strata_sizes} (D-186: {EXPECTED_STRATA}) or pin-only inputs "
                         f"{verification['pin_only_inputs']} (D-186: none)")
    print(f"{stem} via {reg_stem}: C={C_sel:g}, selected layer {L_sel}; {len(layers)} layers; strata {strata_sizes}; "
          f"{args.n_draws} draws per level; D-170 flag: {verification['flag']}; registered: {registered}", flush=True)

    cells = []
    n_checked = n_strata_checked = n_auroc_checked = n_self_anchor = 0
    if args.fpr5 and not set(D209_LAYERS[stem]) <= set(layers):
        raise SystemExit(f"REFUSING: D-209's layers {D209_LAYERS[stem]} are not all in the grid")
    run_layers = list(D209_LAYERS[stem]) if args.fpr5 else [L_sel] + [x for x in layers if x != L_sel]
    run_psets = ("selected",) if args.fpr5 else PROBE_SETS          # D-209: the selected C only
    for li in run_layers:
        k = f"layer_{li}"
        X = {s: {p: data[s][p][k] for p in ("probe_train", "threshold_calib", "eval")} for s in data}
        probes = {pset: [] for pset in PROBE_SETS}
        for seed in args.seeds:
            _, fit = P.d117_split(len(y_tr), seed)
            fo = np.sort(fit)
            probes["selected"].append(P.fit_probe(X["pre"]["probe_train"][fit], y_tr[fit], seed, C_sel, "d117_permutation"))
            probes["C=1"].append(P.fit_probe(X["pre"]["probe_train"][fo], y_tr[fo], seed, 1.0, "file_order"))
        for pset in run_psets:
            S = lambda side, part, m: np.column_stack([q.score(X[side][part][m]) for q in probes[pset]])  # noqa: E731
            sc = {"cal_pre": S("pre", "threshold_calib", ncal), "ev_pre": S("pre", "eval", neg),
                  "cal_post": S("post", "threshold_calib", ncal), "ev_post": S("post", "eval", neg),
                  "pos_pre": S("pre", "eval", pos), "pos_post": S("post", "eval", pos)}
            pool_pre, pool_post = np.vstack([sc["cal_pre"], sc["ev_pre"]]), np.vstack([sc["cal_post"], sc["ev_post"]])
            tm = tm_cells[(li, pset)]
            cell: dict[str, Any] = {"pair": stem, "layer": li, "probe_set": pset,
                                    "probe_C": C_sel if pset == "selected" else 1.0,
                                    "is_selected_layer": li == L_sel, "input_verification": verification}
            for lvl in LEVELS:
                report, arrp = P.repartition_estimate(sc["cal_pre"], sc["ev_pre"], sc["cal_post"], sc["ev_post"],
                                                      sc["pos_pre"], sc["pos_post"], draws[lvl], args.target_fpr)
                want = tm["realised"] if lvl == LEVELS[0] else tm[LEVELS[1]]["realised"]
                mine = json.loads(json.dumps(report, default=str))
                diff = sorted(set(mine) ^ set(want)) + [q for q in mine if q in want and mine[q] != want[q]]
                if diff:
                    raise SystemExit(f"REFUSING: layer {li} {pset} {lvl}: pooled report differs from {tm_rel} in {diff}")
                n_checked += 1
                if args.fpr5:
                    # D-248 self-anchor: the same cell at 1% equals the 1% 4b record's realised report
                    rep1, _ = P.repartition_estimate(sc["cal_pre"], sc["ev_pre"], sc["cal_post"], sc["ev_post"],
                                                     sc["pos_pre"], sc["pos_post"], draws[lvl], P.TARGET_FPR)
                    t1 = tm1_cells[(li, pset)]
                    want1 = t1["realised"] if lvl == LEVELS[0] else t1[LEVELS[1]]["realised"]
                    if json.dumps(json.loads(json.dumps(rep1, default=str)), sort_keys=True) != json.dumps(want1, sort_keys=True):
                        raise SystemExit(f"REFUSING: layer {li} {pset} {lvl}: the 1% twin is not {tm1_info['file']}'s (D-248)")
                    n_self_anchor += 1
                per = [stratum_point(pool_pre, pool_post, c, e, s_pool, sc["pos_pre"], sc["pos_post"], s_pos,
                                     args.target_fpr) for c, e in parts[lvl]]
                # the strata partition the pooled negatives and the positives: in EVERY draw, the strata's
                # flagged counts rebuild the shared estimator's pooled FPR exactly, and on the frozen split
                # the counts and hits sum to the pooled ones (D-188)
                n_ev = n_pool - n_cal
                for q in ("pre", "post"):
                    k_seed = [np.sum([p[st][f"n_flagged_{q}"] for st in STRATA.values()], axis=0) for p in per]
                    rebuilt = np.array([np.maximum(np.asarray(ks) / n_ev, 1.0 / n_ev).mean() for ks in k_seed])
                    if rebuilt[0] != report[f"fpr_{q}"]["frozen"] or not np.array_equal(rebuilt[1:], arrp[f"fpr_{q}"]):
                        raise SystemExit(f"REFUSING: layer {li} {pset} {lvl}: strata do not rebuild the pooled FPR {q}")
                fz = report["frozen_split_counts"]
                for key in ("n_flagged_pre", "n_flagged_post", "hits_pre", "hits_post"):
                    if [int(sum(v)) for v in zip(*(per[0][st][key] for st in STRATA.values()))] != fz[key]:
                        raise SystemExit(f"REFUSING: layer {li} {pset}: strata {key} do not sum to the pooled counts")
                n_strata_checked += 1
                blk: dict[str, Any] = {"note": CLUSTER_NOTE} if lvl == LEVELS[1] else {}
                for st in STRATA.values():
                    arr = {q: np.array([p[st][q] for p in per]) for q in QUANT}
                    blk[st] = {q: P.band_summary(arr[q][1:], arr[q][0]) for q in QUANT}
                    blk[st]["n_negatives_in_evaluation_part"] = {"frozen": per[0][st]["n_neg"],
                                                                "min": int(min(p[st]["n_neg"] for p in per[1:])),
                                                                "max": int(max(p[st]["n_neg"] for p in per[1:]))}
                    blk[st]["share_of_draws_floored"] = float(np.mean([p[st]["floored"] for p in per[1:]]))
                    blk[st]["frozen_split_floored"] = bool(per[0][st]["floored"])
                    for tail in ("neg", "pos"):
                        blk[st][f"share_of_draws_mcnemar_{tail}_p_max_below_0_05"] = float(np.mean(
                            [p[st][f"mcnemar_{tail}_p_max"] < 0.05 for p in per[1:]]))
                    blk[st]["frozen_split_counts"] = {key: per[0][st][key] for key in (
                        "n_flagged_pre", "n_flagged_post", "hits_pre", "hits_post", "neg_discordant_b_c", "pos_discordant_b_c",
                        "mcnemar_neg_min_achievable_p", "mcnemar_pos_min_achievable_p")} | {"n_neg": per[0][st]["n_neg"],
                                                                                            "n_pos": int((s_pos == st).sum())}
                a = {q: np.array([p["adversarial"][q] for p in per]) for q in QUANT}
                v = {q: np.array([p["vanilla"][q] for p in per]) for q in QUANT}
                blk["contrast_adversarial_minus_vanilla"] = {name: P.band_summary(f(a, v)[1:], f(a, v)[0])
                                                             for name, f in CONTRASTS.items()}
                blk["pooled_operating_point"] = report
                if lvl == LEVELS[0]:
                    # D-282: the combined interval of each TPR quantity beside its D-156 band (item level only)
                    T_dr = np.array([draw_thresholds(pool_pre, c, args.target_fpr) for c, _ in parts[lvl][1:]])
                    ones: dict[str, dict[str, np.ndarray]] = {}
                    reps_st: dict[str, dict[str, np.ndarray]] = {}
                    d282: dict[str, Any] = {"note": D282_NOTE}
                    for st in STRATA.values():
                        qm = s_pos == st
                        ones[st] = P.combined_tpr_replicates(T_dr, sc["pos_pre"][qm], sc["pos_post"][qm],
                                                             np.ones((len(T_dr), int(qm.sum())), dtype=np.float32))
                        if any(not np.array_equal(ones[st][name], np.array([p[st][name] for p in per[1:]])) for name in D282_QUANT):
                            raise SystemExit(f"REFUSING: layer {li} {pset} {st}: counts of ones do not give back the D-156 draws (D-282)")
                        n_d282_ones_checked += 1
                        reps_st[st] = P.combined_tpr_replicates(T_dr, sc["pos_pre"][qm], sc["pos_post"][qm], CNT_ST[st])
                        d282[st] = {name: {"frozen": blk[st][name]["frozen"], "band95_D156": blk[st][name]["band95"],
                                           "combined": P.ci(reps_st[st][name]),
                                           "replicates_sha256": sha256_bytes(reps_st[st][name].astype("<f8").tobytes())}
                                    for name in D282_QUANT}
                    if not np.array_equal(ones["adversarial"]["delta_tpr"] - ones["vanilla"]["delta_tpr"],
                                          CONTRASTS["delta_tpr"](a, v)[1:]):
                        raise SystemExit(f"REFUSING: layer {li} {pset}: counts of ones do not give back the D-156 contrast (D-282)")
                    cr = reps_st["adversarial"]["delta_tpr"] - reps_st["vanilla"]["delta_tpr"]
                    cb = blk["contrast_adversarial_minus_vanilla"]["delta_tpr"]
                    d282["contrast_adversarial_minus_vanilla"] = {"delta_tpr": {
                        "frozen": cb["frozen"], "band95_D156": cb["band95"], "combined": P.ci(cr),
                        "replicates_sha256": sha256_bytes(cr.astype("<f8").tobytes())}}
                    d282["counts_all_ones_give_back_the_D156_draws"] = True
                    blk[D282_KEY] = d282
                cell[lvl] = blk
            # AUROC per stratum on the frozen eval split, one item bootstrap for both strata
            au: dict[str, Any] = {"note": AUROC_NOTE}
            reps: dict[str, dict[str, np.ndarray]] = {}
            for st in STRATA.values():
                mn, mp = s_ev_neg == st, s_pos == st
                reps[st] = {}
                pts = {}
                for name, (pk, nk) in {"pre": ("pos_pre", "ev_pre"), "frozen": ("pos_post", "ev_post")}.items():
                    pt = np.array([M.auroc(sc[pk][mp, s], sc[nk][mn, s]).value for s in range(len(args.seeds))])
                    unit = np.array([weighted_auroc(sc[pk][mp, s], sc[nk][mn, s], np.ones((1, int(mp.sum()))),
                                                    np.ones((1, int(mn.sum()))))[0] for s in range(len(args.seeds))])
                    if not np.array_equal(pt, unit):
                        raise SystemExit(f"REFUSING: layer {li} {pset} {st} {name}: weighted AUROC is not metrics.auroc")
                    n_auroc_checked += len(pt)
                    pts[name] = pt
                    reps[st][name] = np.mean([weighted_auroc(sc[pk][mp, s], sc[nk][mn, s], CNT_P[:, mp], CNT_N[:, mn])
                                              for s in range(len(args.seeds))], axis=0)
                pts["drop"] = pts["pre"] - pts["frozen"]
                reps[st]["drop"] = reps[st]["pre"] - reps[st]["frozen"]
                au[st] = {name: {"mean_over_seeds": float(pts[name].mean()), "per_seed": pts[name].tolist(),
                                 "range_over_seeds": [float(pts[name].min()), float(pts[name].max())],
                                 **P.ci(reps[st][name])} for name in ("pre", "frozen", "drop")}
            d_pt = au["adversarial"]["drop"]["mean_over_seeds"] - au["vanilla"]["drop"]["mean_over_seeds"]
            au["drop_adversarial_minus_vanilla"] = {"point": d_pt, **P.ci(reps["adversarial"]["drop"] - reps["vanilla"]["drop"])}
            cell["auroc"] = au
            cell["probe_fit_diagnostics"] = [q.diag for q in probes[pset]]
            cells.append(cell)
        c = next(x for x in cells if x["layer"] == li and x["probe_set"] == "selected")[LEVELS[0]]
        ca = c["contrast_adversarial_minus_vanilla"]["log_paired_ratio"]
        print(f"  layer {li:>2} [selected C]: paired ratio adversarial {c['adversarial']['paired_ratio']['mean']:.3f}, "
              f"vanilla {c['vanilla']['paired_ratio']['mean']:.3f}; log-ratio contrast {ca['mean']:+.3f} "
              f"[{ca['band95'][0]:+.3f}, {ca['band95'][1]:+.3f}]", flush=True)
        cd = c[D282_KEY]["contrast_adversarial_minus_vanilla"]["delta_tpr"]
        print(f"            delta TPR contrast (D-282): frozen {cd['frozen']}, D-156 band {cd['band95_D156']}, "
              f"combined {cd['combined']['ci95']}", flush=True)

    headline = None if args.fpr5 else \
        {pset: {**{lvl: next(x for x in cells if x["layer"] == L_sel and x["probe_set"] == pset)[lvl] for lvl in LEVELS},
                "auroc": next(x for x in cells if x["layer"] == L_sel and x["probe_set"] == pset)["auroc"]}
         for pset in PROBE_SETS}
    if args.fpr5 and registered and n_self_anchor != len(run_layers) * len(run_psets) * len(LEVELS):
        raise SystemExit(f"REFUSING to write: {n_self_anchor} self-anchored blocks (D-248)")
    if n_d282_ones_checked != len(run_layers) * len(run_psets) * len(STRATA):
        raise SystemExit(f"REFUSING to write: {n_d282_ones_checked} D-282 blocks checked against the D-156 draws")
    code1 = P.code_block(CODE_FILES)
    changed = [n for n, a, b in (("code", code1["sha256"], code0["sha256"]), ("selection", sha256_file(sel_path), sel_sha),
                                 ("cluster manifest", sha256_file(CLUSTERS), cl_sha), ("twomoment file", sha256_file(ROOT / tm_rel), tm_sha),
                                 ("twomoment_fit.json", sha256_file(TWOMOMENT_FIT), fit_sha),
                                 *((("twomoment_fit_fpr5.json", sha256_file(TWOMOMENT_FIT_FPR5), fit5_sha),
                                    (tm1_info["file"], sha256_file(ROOT / tm1_info["file"]), tm1_info["sha256"])) if args.fpr5 else ()),
                                 ("registry sources (re-read from disk)", sources_now(), reg_sources)) if a != b]
    changed += [f"cache {r['file']}" for r in input_recs.values() if sha256_file(resolve_cache(r["file"], REG.CACHE)) != r["sha256"]]
    changed += [f for f, h in {**split_files, **man_sha}.items() if sha256_file(ROOT / f) != h]
    if changed:
        raise SystemExit(f"REFUSING to write: changed during the run: {changed}")
    rel = lambda p: str(p.relative_to(ROOT)).replace("\\", "/") if p.is_relative_to(ROOT) else p.name  # noqa: E731
    r1_rel, r1_sha = REISSUE_OF[(stem, args.fpr5)]
    payload = {**provenance_block("advstrat_d156"),
               "preregistration": "D-169 item 8, D-152 P5, D-156, D-164, D-165, D-186" +
                                  ("; D-209 (the five fail-open cells at 5%), 4e: D-152, D-248" if args.fpr5 else "") +
                                  f"; D-281 (1), D-282 (the combined interval for TPR quantities; re-issues {r1_rel})",
               **({"fpr5": {"cells": [{"layer": li, "probe_set": "selected"} for li in run_layers],
                            "self_anchor": {**tm1_info, "n_level_blocks_identical_at_1pct": n_self_anchor,
                                            "kind": "each cell's pooled report recomputed at 1% equals the 1% 4b record (D-248)"}}}
                  if args.fpr5 else {}),
               "code": code0, "registered_parameters": registered, "input_verification": verification,
               "config": {"pair": stem, "registry_pair": reg_stem, "axis": axis, "field": FIELD, "strata": STRATA,
                          "strata_sizes": strata_sizes, "selection_file": rel(sel_path), "selection_sha256": sel_sha,
                          "selected_C": C_sel, "selected_layer": L_sel, "layers": layers, "seeds": args.seeds,
                          "target_fpr": args.target_fpr, "threshold": "ONE global per-seed threshold on the calibration part",
                          "pre_repo": cfg["pre_repo"], "pre_rev": cfg["pre_rev"], "post_repo": cfg["post_repo"],
                          "post_rev": cfg["post_rev"], "input_registry_sources_sha256": reg_sources,
                          "frozen_splits_sha256": split_files, "split_manifests_sha256": man_sha,
                          "cluster_manifest": {"file": "manifests/negative_clusters.json", "sha256": cl_sha},
                          "n_draws": args.n_draws, "draw_seed": P.AXIS_DRAW_SEED[axis],
                          "cluster_draw_seed": P.AXIS_CLUSTER_DRAW_SEED[axis],
                          "item_bootstrap": {"n_boot": args.n_boot, "seeds": ITEM_BOOT_SEED}},
               "twomoment_check": {"file": tm_rel, "sha256": tm_sha, "twomoment_fit_sha256": fit5_sha if args.fpr5 else fit_sha,
                                   "n_pooled_reports_identical": n_checked,
                                   "n_level_blocks_strata_rebuild_pooled_fpr_every_draw": n_strata_checked,
                                   "n_auroc_values_equal_to_metrics_auroc": n_auroc_checked},
               "population": {"declared_in": "src/populations.py", "group": "ADVSTRAT_D156_FPR5" if args.fpr5 else "ADVSTRAT_D156",
                              "replaces": None if args.fpr5 else f"results/{stem}.json",
                              **({"role": "D-209, the 4e sensitivity arm at 5%; replaces nothing"} if args.fpr5 else {})},
               "input_cache_sha256": input_sha,
               "input_cache_source": {k: {"file": r["file"], "expected_from": r["expected_from"], "pin_only": r["pin_only"]}
                                      for k, r in input_recs.items()},
               "headline_selected_layer": headline, "cells": sorted(cells, key=lambda c: (c["layer"], c["probe_set"]))}
    payload["supersedes"] = [f"{r1_rel} (D-282: the same numbers plus the combined intervals for TPR; never deleted)"]
    payload["d282"] = {"preregistration": "D-281 (1), D-282", "level": LEVELS[0], "quantities_per_stratum": list(D282_QUANT),
                       "contrast": "adversarial minus vanilla, delta_tpr", "n_boot": args.n_boot, "n_draws": args.n_draws,
                       "draw_of_replicate": "d_r = 1 + ((r - 1) mod n_draws); the frozen split is not a draw",
                       "counts": "paired.counts_matrix(n positives of the stratum, n_boot, the stratum's seed), row r - 1 "
                                 "for replicate r",
                       "counts_seed_per_stratum": D282_SEED,
                       "counts_items_per_stratum": {st: int(CNT_ST[st].shape[1]) for st in STRATA.values()},
                       "seed_note": (f"D-282 (2) names depth_d156's eval_positives seed ({DEPTH_ITEM_BOOT_SEED['eval_positives']}) "
                                     f"as the matrix D-152 P5 already uses; for this script that matrix is its own D-186 "
                                     f"stream ({ITEM_BOOT_SEED['eval_positives']}), used here offset by the stratum's index "
                                     f"in STRATA (D-181, the closest entry D-186). The literal reading would put the vanilla "
                                     f"stratum on {DEPTH_ITEM_BOOT_SEED['eval_positives'] + 1}, this script's eval_negatives "
                                     f"seed. The AUROC bootstrap is unchanged"),
                       "replicates_sha256_of": "the float64 little-endian bytes of the replicate vector, r = 1..n_boot in "
                                               "order, length n_boot",
                       "cluster_level": "keeps its D-156 band only (D-282 (2))",
                       "pooled_operating_point": "unchanged, a copy of the 4b record's realised report; its combined "
                                                 "interval comes with Table 1 (D-282 (3))",
                       "n_blocks_counts_of_ones_checked": n_d282_ones_checked}
    if registered:
        payload["reissue_of"] = reissue_check(payload, (ROOT / r1_rel).read_bytes(), r1_rel, r1_sha)
    else:
        payload["reissue_of"] = {"file": r1_rel, "sha256": r1_sha, "compared": False,
                                 "why": "non-registered parameters (a test run); the numbers cannot equal the record's"}
    digest = write_results(out_path, payload, exclusive=True)
    print(f"Written: {rel(out_path)}\nsha256:  {digest}\npooled reports identical to {tm_rel}: {n_checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
