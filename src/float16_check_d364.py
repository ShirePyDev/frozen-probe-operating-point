"""D-364: the float16 one-cell check. It decides whether the LoRA user-span-mean subsets may be stored in float16 (the
author's D-318 (7) item 5, D-357 (B)(3), (B)(5)). A storage-validation check, not an analysis that enters the paper
(D-364 (4)): its A values are Phase 2 evaluation numbers of one cell, reported only as part of this check.

THE CELL (D-364 (1)), fixed here; nothing about it is a command-line option: the Phase 2 primary pair
Mistral-7B-Instruct-v0.1 -> v0.2, user-span mean, the pre side's selection of record (results/phase2/selection_p2__
mistralai__Mistral-7B-Instruct-v0.1@ec5deb64f2c6__usermean.json: its arm, L*, C, seeds and holdout), item level, split
v2 (D-362) read by D-320 (4)'s join rule from the durable copies only (D-307 (1)), and the D-156 estimator as D-361 (2):
the calibration and evaluation negatives pooled (5,000 + 6,000 at v2's sizes), pre and post scores of the same items,
re-partitioned 1000 times, the frozen split as one more draw, the evaluation positives fixed, five seeds (one probe per
seed), per-seed thresholds at the target on the calibration part of the PRE scores, FPR floored at 1/|evaluation
part|, the paired ratio the mean over seeds of the per-seed floored ratios, TPR at the frozen threshold; all through
paired.repartition_estimate, unchanged. 1% primary, 5% beside.

TWO ARMS (D-364 (2)), both sides of every comparison in the same dtype, with the SAME probes and the SAME draws:
  A  every feature as stored (float32);
  B  every SCORED feature (pre and post; calibration and evaluation; negatives and positives) cast to float16 and back
     to float32 before scoring.
In both arms the probe is the one fitted on the stored float32 probe_train: per seed, selection_tpr_v2.seed_probes (the
selection's reporting block: the same permutation, split, standardisation and fit), fitted once.

CRITERION (D-364 (3), exactly; both inequalities strict): float16 storage is ALLOWED only if, at 1%, for every
operating-point number - the paired ratio, FPR/target pre (floor_ratio_to_target) and FPR/target post
(ratio_to_target), each at the frozen split and as the D-156 mean - |log(B / A)| < 0.008, and for every TPR number -
pre, post and delta, frozen and D-156 mean - |B - A| < 0.001; else NOT_ALLOWED (float16 is not used and every LoRA
subset stays float32). A ratio number that is not finite and positive in both arms fails. A TPR is a count over
(positives x seeds), its D-156 mean a count over (positives x seeds x draws): |B - A| is compared in exact rational
arithmetic from those counts, since a float difference of exactly 0.001 can round to either side. The 5% numbers are
computed, differenced and judged by the same rule BESIDE (D-364 (1) '5% beside'; D-361 (4): beside, never a verdict).

STEPS (steps 1 and 2 read records only, so a missing or mismatched record refuses before any cache is read)
  1. SELECTION  selection_tpr_v2.read_selection (unchanged): the selection record names this model, revision, pooling
     and axis, its select record as it is now, this run's BLAS threads (all three "6") and libraries, and
     src/selection_p2.py, src/regularisation_sweep.py and src/paired.py at their present sha256.
  2. RECORDS  the v2 record and its calibration file (selection_tpr_v2.read_v2, unchanged) and its evaluation file
     (the same checks here: the literal path, the record's bytes, counts, unique ids, labels 0/1, no excluded item, no
     item also in v2 calibration, and the per-v1-split counts the record's plan implies); the post side's select record
     (ingest yes, documented-tag form); both models' durable rest subsets (selection_tpr_v2.read_subset, unchanged);
     the pre subset's layer plan (selection_tpr_v2.check_l_star) and the post subset's (role 'post', this pair, this
     selection record and L*); D-320 (4): every cache read is keyed p2wg/<v1 split> and indexes the v1 file the v2
     record names (a cache naming a v2 split file under a v1 key refuses: it is never read as v2); every subset entry
     by selection_tpr_v2.subset_entry.
  3. CACHES (D-307 (1): data/hidden only, regular file, mode 444, in the record's ingest index with the record's
     sha256, hashing to it, item for item the frozen split): the pre side's probe_train and v1 calibration and the post
     side's v1 calibration by selection_p2.load_ingested; both sides' subset eval and pool4d by
     selection_tpr_v2.load_subset_cache. The pre side's two must be the caches the selection read. Pre and post must
     hold the same items with the same labels. Every array read must be float32 (arm A is float32 storage).
  4. ANCHOR  the per-seed probes, thresholded on v1 calibration, reproduce the selection record's ho20 TPRs bit for bit
     (selection_tpr_v2.anchor_diffs); otherwise nothing is written ("ANCHOR FAILED").
  5. v2 GATHER  v2 calibration and v2 evaluation at L*, each row BY item_id from the cache of the split v1 put it in
     (selection_tpr_v2.gather; positives through the same function with each label read as 1 - label, after every
     item's v1 label is checked against its v2 label), for each side; the counts must be the v2 record's plan.
  6. ARMS  A and B scored with the same probes; paired.repartition_estimate on the same draws at 1% and 5%.
  7. RECORD  each number's A, B and difference, the maximum of each kind, the verdict; the D-156 reports of both arms;
     fingerprints (sha256) of the probes, of every scored matrix and of every score matrix; the cast's effect on every
     scored matrix; the provenance (every result read with its sha256, the split files, the ingest indexes, the caches'
     sha256, the code, config and config_hash).

OUTPUT results/phase2/float16_check_d364.json, exclusive (never overwritten). The final name requires the registered
parameters (1000 draws, seeds 0-4, the 1% target), the 6-thread BLAS pin (selection_p2.check_threads; every run
requires it, since the selection record ran under it) and the selection record and v2 split files at their D-364 /
D-362 sha256 (PINNED). A test writes elsewhere with --out, never under results/.

  python3 -u src/float16_check_d364.py                                  (through scripts/project_env.sh; lane C)
  python3 -u src/float16_check_d364.py --out <path> [--n-draws N]       (a test)

CPU only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import (  # noqa: E402
    _RESULT_READS,
    canonical_hash,
    ensure_utf8_stdout,
    provenance_block,
    read_result,
    sha256_bytes,
    sha256_file,
    write_results,
)
import paired as P  # noqa: E402
import provenance as PV  # noqa: E402
import selection_p2 as SP  # noqa: E402  (imported, not changed: its loaders and checks are used as they are)
import selection_tpr_v2 as T  # noqa: E402  (imported, not changed: its v2 gather and record checks)

CODE_FILES = ["src/float16_check_d364.py", *T.CODE_FILES]
PRE_REPO = "mistralai/Mistral-7B-Instruct-v0.1"      # D-364 (1): the Phase 2 primary pair; its post side is the LoRA base
POST_REPO = "mistralai/Mistral-7B-Instruct-v0.2"
POOLING = "usermean"                                  # D-364 (1): the only pooling float16 would shrink
AXIS = SP.AXIS
SUB_TAG = T.SUB_TAG
V1_SPLITS = T.V1_SPLITS
OUT_NAME = "float16_check_d364.json"
PRIMARY_TARGET = 0.01                                 # D-364 (1): 1% primary
BESIDE_TARGET = 0.05                                  # 5% beside, never the verdict (D-361 (4))
TARGETS = (PRIMARY_TARGET, BESIDE_TARGET)
REGISTERED = {"n_draws": P.N_DRAWS, "seeds": [0, 1, 2, 3, 4], "target_fpr": PRIMARY_TARGET}
# paired.AXIS_DRAW_SEED holds no Phase 2 axis (D-156 fixes one partition seed per axis; D-361's code fixes p2wg's).
# This check's draws are shared by both arms and both targets, so A - B does not depend on the seed; fixed here before
# any run (D-181, by D-156).
DRAW_SEED = 20261006
TOL_LOG = 0.008                                       # D-364 (3): |log(B / A)| < 0.008
TOL_TPR = Fraction(1, 1000)                           # D-364 (3): |B - A| < 0.001, exact
V2_DIR = "data/frozen_p2_v2/"
V2_EVAL_FILE = "data/frozen_p2_v2/p2wg_v2__eval.jsonl"
PINNED = {                                            # the final name only
    "selection_record": "4966000b2d75c3534ccedd8eab464e7650e1397c68938a82ce2a57cbb4b1aede",   # the file at review
    "v2/calibration": "98bc80c966349bc5e03c5cdeab97d035d40447b34c5337d72e2f9062841393e5",   # D-362 (1)
    "v2/eval": "e0fcf8a30de263e502d191c7df87e11d330362602670247bc7de8620cdd17a4c",          # D-362 (1)
}
SIDES = ("pre", "post")
MATRICES = ("calibration", "eval_neg", "eval_pos")    # every scored split: v2 calibration, v2 eval negatives, positives
RATIO_NUMBERS = ("paired_ratio", "floor_ratio_to_target", "ratio_to_target")
TPR_NUMBERS = ("tpr_pre", "tpr_post", "delta_tpr")
STATS = ("frozen", "mean")
LABELS = {"paired_ratio": "paired ratio FPR_post / FPR_pre", "floor_ratio_to_target": "FPR / target, pre",
          "ratio_to_target": "FPR / target, post", "tpr_pre": "TPR pre", "tpr_post": "TPR post",
          "delta_tpr": "delta TPR (post - pre)"}
STAT_LABELS = {"frozen": "the frozen split", "mean": "the D-156 mean over the draws"}
FLOAT16_MAX = float(np.finfo(np.float16).max)
FLOAT16_TINY = float(np.finfo(np.float16).tiny)       # the smallest normal float16
CRITERION = ("D-364 (3): float16 storage is ALLOWED only if, for every operating-point number (paired ratio, FPR/target "
             "pre and post; frozen split and D-156 mean), |log(B / A)| < 0.008, and for every TPR number (pre, post, "
             "delta; frozen and D-156 mean) |B - A| < 0.001; else float16 is not used and every LoRA subset stays "
             "float32. Judged on the 1% numbers (1% primary, D-364 (1)); the 5% numbers by the same rule beside, never "
             "the verdict (D-361 (4)). Strict inequalities; TPR differences in exact rational arithmetic from the hit "
             "counts; a ratio number not finite and positive in both arms fails.")

refuse = SP.refuse
rel = SP.rel


# --------------------------------------------------------------------------- the arms and the criterion

def f16(X: np.ndarray) -> np.ndarray:
    """Arm B's cast (D-364 (2)): float16 and back to float32."""
    return X.astype(np.float16).astype(np.float32)


def arm_inputs(mats: dict[str, dict[str, np.ndarray]], arm: str) -> dict[str, dict[str, np.ndarray]]:
    """The scored features of one arm: A as stored; B every matrix of both sides cast (none is left out)."""
    if arm == "A":
        return mats
    if arm == "B":
        return {side: {m: f16(X) for m, X in d.items()} for side, d in mats.items()}
    raise ValueError(f"unknown arm {arm!r}")


def scores(probes: list[dict[str, Any]], X: np.ndarray) -> np.ndarray:
    """(n, seeds): each seed's probe on X, in selection_tpr_v2.tprs_at's expression."""
    return np.column_stack([(X - p["mu"]) / p["sd"] @ p["w"] + p["b"] for p in probes])


def judge_ratio(a: Any, b: Any) -> dict[str, Any]:
    """|log(B / A)| < TOL_LOG, strict; a number that is not finite and positive in both arms fails."""
    ok = all(isinstance(x, float) and math.isfinite(x) and x > 0 for x in (a, b))
    lg = math.log(b / a) if ok else None
    d = abs(lg) if lg is not None else None
    return {"A": a, "B": b, "log_B_over_A": lg, "abs_log_B_over_A": d, "tolerance": TOL_LOG,
            "within": bool(d is not None and math.isfinite(d) and d < TOL_LOG)}


def judge_tpr(a: Any, b: Any, ea: Fraction, eb: Fraction) -> dict[str, Any]:
    """|B - A| < 1/1000, strict, on the exact values ea, eb (a and b are the reported floats)."""
    d = abs(eb - ea)
    return {"A": a, "B": b, "A_exact": f"{ea.numerator}/{ea.denominator}", "B_exact": f"{eb.numerator}/{eb.denominator}",
            "B_minus_A": float(eb - ea), "abs_B_minus_A": float(d), "abs_B_minus_A_exact": f"{d.numerator}/{d.denominator}",
            "tolerance": float(TOL_TPR), "within": bool(d < TOL_TPR)}


def exact_tprs(report: dict[str, Any], arrays: dict[str, np.ndarray], n_seeds: int) -> dict[str, dict[str, Fraction]]:
    """The TPR numbers as exact fractions: the frozen split's from its hit counts, the D-156 means from the per-draw
    values, each a count over (positives x seeds) recovered by rounding (refused unless within 1e-6 of an integer).
    Each must reproduce the reported float (within 1e-9)."""
    c = report["frozen_split_counts"]
    n_pos = int(c["n_positives"])
    unit = n_pos * n_seeds
    if n_pos < 1 or len(c["hits_pre"]) != n_seeds or len(c["hits_post"]) != n_seeds:
        refuse(f"the frozen split's counts do not hold one hit count per seed ({n_seeds}) over {n_pos} positives")
    pre, post = Fraction(int(sum(c["hits_pre"])), unit), Fraction(int(sum(c["hits_post"])), unit)
    out: dict[str, dict[str, Fraction]] = {"frozen": {"tpr_pre": pre, "tpr_post": post, "delta_tpr": post - pre},
                                           "mean": {}}
    for q in TPR_NUMBERS:
        v = np.asarray(arrays[q], dtype=np.float64) * unit
        k = np.rint(v)
        if v.size == 0 or not bool(np.all(np.isfinite(v))) or float(np.max(np.abs(v - k))) > 1e-6:
            refuse(f"the per-draw {q} values are not counts over {n_pos} positives x {n_seeds} seeds")
        out["mean"][q] = Fraction(int(k.astype(np.int64).sum()), unit * int(v.size))
    for stat in STATS:
        for q in TPR_NUMBERS:
            got = report[q][stat]
            if not isinstance(got, float) or abs(float(out[stat][q]) - got) > 1e-9:
                refuse(f"the exact {q} ({stat}) {float(out[stat][q])!r} does not reproduce the reported {got!r}")
    return out


def compare(rep_a: dict[str, Any], arr_a: dict[str, np.ndarray], rep_b: dict[str, Any], arr_b: dict[str, np.ndarray],
            n_seeds: int) -> dict[str, Any]:
    """Every number of D-364 (3) at one target: A, B, the difference, within or not; the maximum of each kind."""
    ex_a, ex_b = exact_tprs(rep_a, arr_a, n_seeds), exact_tprs(rep_b, arr_b, n_seeds)
    rows: list[dict[str, Any]] = []
    for q in RATIO_NUMBERS:
        for stat in STATS:
            rows.append({"number": q, "statistic": stat, "label": f"{LABELS[q]}, {STAT_LABELS[stat]}",
                         "kind": "ratio", **judge_ratio(rep_a[q][stat], rep_b[q][stat])})
    for q in TPR_NUMBERS:
        for stat in STATS:
            rows.append({"number": q, "statistic": stat, "label": f"{LABELS[q]}, {STAT_LABELS[stat]}",
                         "kind": "tpr", **judge_tpr(rep_a[q][stat], rep_b[q][stat], ex_a[stat][q], ex_b[stat][q])})
    rat = [r for r in rows if r["kind"] == "ratio"]
    undefined = [f"{r['number']} ({r['statistic']})" for r in rat if r["abs_log_B_over_A"] is None]
    worst_r = max((r for r in rat if r["abs_log_B_over_A"] is not None), key=lambda r: r["abs_log_B_over_A"],
                  default=None)
    tpr = [r for r in rows if r["kind"] == "tpr"]
    worst_t = max(tpr, key=lambda r: Fraction(r["abs_B_minus_A_exact"]))
    return {"numbers": rows,
            "max_abs_log_B_over_A": {"value": None if worst_r is None else worst_r["abs_log_B_over_A"],
                                     "at": None if worst_r is None else f"{worst_r['number']} ({worst_r['statistic']})",
                                     "undefined_numbers": undefined, "tolerance": TOL_LOG},
            "max_abs_tpr_difference": {"value": worst_t["abs_B_minus_A"], "exact": worst_t["abs_B_minus_A_exact"],
                                       "at": f"{worst_t['number']} ({worst_t['statistic']})",
                                       "tolerance": float(TOL_TPR)},
            "n_numbers": len(rows), "n_within": sum(r["within"] for r in rows),
            "criterion_met": all(r["within"] for r in rows)}


def cast_effect(X: np.ndarray) -> dict[str, Any]:
    """What arm B's cast does to one stored matrix (recorded; the verdict is the criterion alone)."""
    Y = f16(X)
    fin = np.isfinite(Y)
    ax = np.abs(X)
    ch = np.abs(Y - X)[fin] if bool(fin.any()) else np.zeros(0, np.float32)
    nz = (X != 0) & fin
    rel_ch = (np.abs(Y[nz] - X[nz]) / ax[nz]) if bool(nz.any()) else np.zeros(0, np.float32)
    return {"dtype_stored": str(X.dtype), "shape": [int(s) for s in X.shape], "n_values": int(X.size),
            "max_abs_stored": float(ax.max()) if X.size else None,
            "n_values_changed_by_cast": int((Y != X).sum()),
            "n_above_float16_max": int((ax > FLOAT16_MAX).sum()),
            "n_nonfinite_after_cast": int((~fin).sum()),
            "n_float16_subnormal_or_zeroed": int(((ax < FLOAT16_TINY) & (X != 0)).sum()),
            "max_abs_change": float(ch.max()) if ch.size else None,
            "max_relative_change": float(rel_ch.max()) if rel_ch.size else None}


def fingerprint(X: np.ndarray) -> dict[str, Any]:
    return {"sha256": sha256_bytes(np.ascontiguousarray(X).tobytes()), "dtype": str(X.dtype),
            "shape": [int(s) for s in X.shape]}


# --------------------------------------------------------------------------- records (no cache is read)

def read_select(repo: str) -> dict[str, Any]:
    """A model's select record (selection_p2.select_record), checked as selection_tpr_v2.read_selection checks the pre
    side's: this repo, set select, ingest yes (D-307 (1)), the documented-tag form."""
    p = SP.select_record(repo)
    r = rel(p)
    rec = read_result(p)
    if rec.get("repo") != repo or rec.get("set_name") != "select":
        refuse(f"{r} names repo {rec.get('repo')!r}, set {rec.get('set_name')!r}")
    if rec.get("ingest") != "yes":
        refuse(f"{r} says ingest: {rec.get('ingest')!r}; a result reads only ingested caches (D-307 (1))")
    rev = rec.get("revision")
    if not isinstance(rev, str) or len(rev) != 40 or rec.get("cache_revision") != rev \
            or rec.get("cache_repo") != f"{repo}__{SP.TAG}":
        refuse(f"{r}: revision, cache_revision or cache_repo is not the documented-tag form")
    return {"path": p, "rel": r, "rec": rec, "rev": rev}


def check_post_plan(sb: dict[str, Any], sr: dict[str, Any], pooling: str) -> dict[str, Any]:
    """The post side's subset layer plan for the pooling names this pair with role 'post', this selection record, its
    sha256 and L* (a FULL_GRID plan keeps every layer and names none; recorded as such)."""
    pp = ((sb["rec"].get("layer_plan") or {}).get("per_pooling") or {}).get(pooling)
    if not isinstance(pp, dict):
        refuse(f"{sb['rel']} has no layer_plan for {pooling}")
    if pp.get("full_grid_reason"):
        return {"full_grid_reason": pp["full_grid_reason"], "post_entries": 0, "pair": None}
    ent = [e for e in (pp.get("l_star") or []) if isinstance(e, dict) and e.get("role") == "post"
           and list(e.get("pair") or []) == [PRE_REPO, POST_REPO]]
    want = (sr["sel_rel"], _RESULT_READS[sr["sel_rel"]], sr["layer"])
    got = [(e.get("selection_record"), e.get("selection_record_sha256"), e.get("layer")) for e in ent]
    if len(ent) != 1 or got[0] != want:
        refuse(f"{sb['rel']}'s layer plan for {pooling} does not name, once, the pair {PRE_REPO} -> {POST_REPO} as its "
               f"post side with {sr['sel_rel']} ({want[1][:16]}) at layer {sr['layer']}: {got}")
    return {"full_grid_reason": None, "post_entries": 1, "pair": [PRE_REPO, POST_REPO]}


def read_v2_eval(v2: dict[str, Any]) -> dict[str, Any]:
    """The v2 evaluation partition and file, checked as selection_tpr_v2.read_v2 checks calibration, plus labels 0/1,
    the record's counts, no item in v2 calibration, and the per-v1-split counts its plan implies (moved negatives by
    plan.items_from_v1_to_v2; positives and their clusters' negatives stay in v1 eval: plan.fixed_*)."""
    rec, v2_rel = v2["rec"], v2["rel"]
    part = ((((rec.get("axes") or {}).get(AXIS) or {}).get("partitions")) or {}).get("eval") or {}
    if part.get("file") != V2_EVAL_FILE:
        refuse(f"{v2_rel} names {part.get('file')!r} for v2 eval, not {V2_EVAL_FILE} (a v1 file is never read as v2, "
               "D-320 (4))")
    if not isinstance(part.get("sha256"), str) or len(part["sha256"]) != 64:
        refuse(f"{v2_rel}: malformed v2 eval sha256 {part.get('sha256')!r}")
    n, n_pos, n_neg = part.get("n"), part.get("n_pos"), part.get("n_neg")
    if not all(isinstance(x, int) and not isinstance(x, bool) and x >= 1 for x in (n, n_pos, n_neg)) \
            or n != n_pos + n_neg:
        refuse(f"{v2_rel}: v2 eval n {n!r}, n_pos {n_pos!r}, n_neg {n_neg!r} (both classes, n = n_pos + n_neg)")
    f = SP.SPLIT_ROOT / V2_EVAL_FILE
    if f.is_symlink() or not f.is_file():
        refuse(f"no v2 eval file {V2_EVAL_FILE} (or not a regular file)")
    b = f.read_bytes()
    if sha256_bytes(b) != part["sha256"]:
        refuse(f"{V2_EVAL_FILE} is not the bytes {v2_rel} names ({sha256_bytes(b)[:16]} vs {part['sha256'][:16]})")
    rows = [json.loads(line) for line in b.decode("utf-8").split("\n") if line.strip()]  # not a result (D-297)
    if len(rows) != n:
        refuse(f"{V2_EVAL_FILE} holds {len(rows)} rows, {v2_rel} says {n}")
    ids = [r.get("item_id") for r in rows]
    if not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != len(ids):
        refuse(f"{V2_EVAL_FILE}: an item_id is missing or repeats")
    labels = [r.get("label") for r in rows]
    if any(isinstance(x, bool) or x not in (0, 1) for x in labels):
        refuse(f"{V2_EVAL_FILE}: a label that is not 0 or 1")
    if (sum(labels), len(labels) - sum(labels)) != (n_pos, n_neg):
        refuse(f"{V2_EVAL_FILE} holds {sum(labels)} positives and {len(labels) - sum(labels)} negatives, {v2_rel} says "
               f"{n_pos} and {n_neg}")
    held = sorted(set(ids) & set(v2["excluded_items"]))
    if held:
        refuse(f"{V2_EVAL_FILE} holds excluded items {held} (config.excluded_items of {v2_rel})")
    both = sorted(set(ids) & set(v2["ids"]))
    if both:
        refuse(f"{len(both)} items are in both v2 eval and v2 calibration (first {both[0]})")
    plan = rec.get("plan") or {}
    moves, fneg, fpos = plan.get("items_from_v1_to_v2"), plan.get("fixed_negatives"), plan.get("fixed_positives")
    if not all(isinstance(x, dict) for x in (moves, fneg, fpos)):
        refuse(f"{v2_rel}: plan.items_from_v1_to_v2, fixed_negatives or fixed_positives is missing")
    expect_neg = {sp: int(moves.get(f"{sp}->eval", 0)) + (int(fneg.get("eval", 0)) if sp == "eval" else 0)
                  for sp in V1_SPLITS}
    expect_pos = {sp: int(fpos.get("eval", 0)) if sp == "eval" else 0 for sp in V1_SPLITS}
    if sum(expect_neg.values()) != n_neg or sum(expect_pos.values()) != n_pos:
        refuse(f"{v2_rel}'s plan implies {sum(expect_neg.values())} negatives and {sum(expect_pos.values())} positives "
               f"in v2 eval, its partition says {n_neg} and {n_pos}")
    return {"ids": ids, "labels": [int(x) for x in labels], "n": n, "n_pos": n_pos, "n_neg": n_neg,
            "sha256": part["sha256"], "expect_neg": expect_neg, "expect_pos": expect_pos}


def check_v1_keyed(entries: dict[str, dict[str, Any]], v2: dict[str, Any]) -> dict[str, Any]:
    """D-320 (4): every cache read is keyed p2wg/<v1 split> and is a v1 cache: its split file is the v1 file (and
    sha256) the v2 record names for that split, never a v2 split file. A v1 cache is read only through the join rule
    (gathered by item_id), never as a v2 split."""
    out = {}
    for key, e in entries.items():
        parts = key.split("/")
        split = parts[-2]
        if parts[-3] != AXIS or split not in (*V1_SPLITS, "probe_train"):
            refuse(f"{key} is not a cache keyed {AXIS}/<v1 split>")
        f = e.get("split_file")
        if not isinstance(f, str) or f.startswith(V2_DIR) or Path(f).name.startswith(f"{AXIS}_v2"):
            refuse(f"{key}: a cache keyed {AXIS}/{split} names the v2 split file {f!r}; a cache keyed {AXIS}/<split> is "
                   "v1 and is never read as v2 (D-320 (4))")
        want = v2["probe_train"] if split == "probe_train" else v2["v1_files"][split]
        if {"file": f, "sha256": e.get("split_sha256")} != want:
            refuse(f"{key} indexes {f!r} ({str(e.get('split_sha256'))[:16]}), not the v1 {split} file {v2['rel']} "
                   f"names {want}")
        out[key] = {"file": f, "sha256": e.get("split_sha256")}
    return out


# --------------------------------------------------------------------------- the v2 gather

def gather_split(name: str, ids: list[str], labels: list[int],
                 sources: dict[str, tuple[np.ndarray, list[str], list[int]]]) -> tuple[np.ndarray, np.ndarray,
                                                                                       dict[str, int], dict[str, int]]:
    """One v2 split's L* rows under the join rule, in the v2 file's order within each class: the negatives through
    selection_tpr_v2.gather as it is; the positives through the same function with every source label read as
    1 - label (so its 'label 0' check reads 'label 1'), after every item's v1 label is checked against its v2 label
    here. Returns (negatives, positives, counts of negatives per v1 split, counts of positives per v1 split)."""
    where: dict[str, list[tuple[str, int]]] = {}
    for s, (_, sids, slabs) in sources.items():
        for iid, lab in zip(sids, slabs):
            where.setdefault(iid, []).append((s, lab))
    for iid, lab in zip(ids, labels):
        hits = where.get(iid) or []
        if len(hits) == 1 and hits[0][1] != lab:
            refuse(f"v2 {name} item {iid} has label {lab} in the v2 file but {hits[0][1]} in the v1 {hits[0][0]} cache")
    neg = [i for i, lab in zip(ids, labels) if lab == 0]
    pos = [i for i, lab in zip(ids, labels) if lab == 1]
    Xn, cn = T.gather({"n": len(neg), "ids": neg}, sources)
    if pos:
        flipped = {s: (X, sids, [1 - lab for lab in slabs]) for s, (X, sids, slabs) in sources.items()}
        Xp, cp = T.gather({"n": len(pos), "ids": pos}, flipped)
    else:
        Xp, cp = Xn[:0], {s: 0 for s in sources}
    return Xn, Xp, cn, cp


# --------------------------------------------------------------------------- the run

def run(n_draws: int = P.N_DRAWS, out: str | Path | None = None) -> Path:
    threads = SP.check_threads()
    code0 = P.code_block(CODE_FILES)
    final_path = (SP.P2 / OUT_NAME).resolve()
    out_path = final_path if out is None else Path(out).resolve()
    final = out_path == final_path
    if out_path.exists() or out_path.is_symlink():
        refuse(f"{out_path} exists; results are never overwritten (D-152 P5)")
    if not final and out_path.is_relative_to(SP.RESULTS.resolve()):
        refuse("a test run (--out) must not write under results/; only the final name may")
    if not isinstance(n_draws, int) or n_draws < 1:
        refuse(f"n_draws {n_draws!r}")

    # step 1: the selection record (the pre side)
    sr = T.read_selection(PRE_REPO, POOLING, threads, code0)
    arm, layer, seeds, holdout = sr["arm"], sr["layer"], sr["seeds"], sr["holdout"]
    if sr["target_fpr"] != PRIMARY_TARGET:
        refuse(f"{sr['sel_rel']}'s target FPR is {sr['target_fpr']!r}, not the primary {PRIMARY_TARGET}")
    registered = n_draws == REGISTERED["n_draws"] and list(seeds) == REGISTERED["seeds"]
    if final and not registered:
        refuse(f"the final name needs the registered parameters {REGISTERED} (n_draws {n_draws}, seeds {seeds}); use "
               "--out outside results/ for a test")

    # step 2: the records, before any cache is read
    v2 = T.read_v2(code0)
    ev2 = read_v2_eval(v2)
    if final:
        have = {"selection_record": _RESULT_READS[sr["sel_rel"]], "v2/calibration": v2["sha256"], "v2/eval": ev2["sha256"]}
        bad = {k: have[k][:16] for k in PINNED if have[k] != PINNED[k]}
        if bad:
            refuse(f"the final name needs the selection record as it was at review and the v2 split files of D-362: "
                   f"{bad} differ from "
                   f"{ {k: PINNED[k][:16] for k in bad} }")
    sb = {"pre": T.read_subset(PRE_REPO, sr["rev"])}
    l_star = {"pre": T.check_l_star(sb["pre"], sr, POOLING)}
    post = read_select(POST_REPO)
    sb["post"] = T.read_subset(POST_REPO, post["rev"])
    l_star["post"] = check_post_plan(sb["post"], sr, POOLING)
    sel_entry = {"pre": sr["rec"].get("splits") or {}, "post": post["rec"].get("splits") or {}}
    entries: dict[str, dict[str, Any]] = {}
    for side, parts in (("pre", ("probe_train", "calibration")), ("post", ("calibration",))):
        for part in parts:
            e = sel_entry[side].get(f"{AXIS}/{part}/{POOLING}")
            if not isinstance(e, dict):
                refuse(f"{sr['rec_rel'] if side == 'pre' else post['rel']} has no split {AXIS}/{part}/{POOLING}")
            entries[f"{side}/{AXIS}/{part}/{POOLING}"] = e
        for part in ("eval", "pool4d"):
            e = (sb[side]["rec"].get("splits") or {}).get(f"{AXIS}/{part}/{POOLING}")
            if not isinstance(e, dict):
                refuse(f"{sb[side]['rel']} has no split {AXIS}/{part}/{POOLING}")
            entries[f"{side}/{SUB_TAG}/{AXIS}/{part}/{POOLING}"] = e
    v1_keyed = check_v1_keyed(entries, v2)
    s_sub = {side: {part: T.subset_entry(sb[side], part, POOLING, layer) for part in ("eval", "pool4d")}
             for side in SIDES}
    if layer not in (post["rec"].get("layers_resolved") or []):
        refuse(f"L* {layer} is not a layer of {post['rel']}")
    out_rel = rel(out_path)
    print(f"D-364 float16 check: {PRE_REPO} -> {POST_REPO} [{AXIS}, {POOLING}], {arm} at layer {layer} (from "
          f"{sr['sel_rel']}); seeds {seeds}; split v2: calibration {v2['n']} negatives, eval {ev2['n_neg']} negatives + "
          f"{ev2['n_pos']} positives; {n_draws} draws (seed {DRAW_SEED}); targets {list(TARGETS)} (1% the verdict); "
          f"output {out_rel}; registered: {registered}", flush=True)

    # step 3: the durable caches (D-307 (1))
    index = {"pre": SP.ingest_index(sr["rec_path"]), "post": SP.ingest_index(post["path"])}
    z_tr, r_tr, _ = SP.load_ingested(sr["rec"], sr["rec_rel"], index["pre"][1], "probe_train", POOLING)
    z_cal = {}
    r_cal = {}
    z_cal["pre"], r_cal["pre"], _ = SP.load_ingested(sr["rec"], sr["rec_rel"], index["pre"][1], "calibration", POOLING)
    z_cal["post"], r_cal["post"], _ = SP.load_ingested(post["rec"], post["rel"], index["post"][1], "calibration",
                                                       POOLING)
    recorded = sr["sel"].get("input_cache_sha256") if isinstance(sr["sel"].get("input_cache_sha256"), dict) else {}
    for part, r in (("probe_train", r_tr), ("calibration", r_cal["pre"])):
        k = f"pre/{AXIS}/{part}/{POOLING}"
        if recorded.get(k) != r["sha256"]:
            refuse(f"{r['file']} ({r['sha256'][:16]}) is not the cache {sr['sel_rel']} read for {k} "
                   f"({str(recorded.get(k))[:16]})")
    key = f"layer_{layer}"
    y_tr = z_tr["labels"].astype(bool)
    if sr["sel"]["model"].get("n_probe_train") != len(y_tr):
        refuse(f"probe_train holds {len(y_tr)} items, {sr['sel_rel']} says {sr['sel']['model'].get('n_probe_train')}")
    X_tr = z_tr[key]
    z_tr.close()
    sources: dict[str, dict[str, tuple[np.ndarray, list[str], list[int]]]] = {}
    for side in SIDES:
        z = z_cal[side]
        sources[side] = {"calibration": (z[key], [str(x) for x in z["item_ids"]], [int(x) for x in z["labels"]])}
        z.close()
    r_sub: dict[str, dict[str, dict[str, Any]]] = {}
    for side in SIDES:
        r_sub[side] = {}
        for part in ("eval", "pool4d"):
            X, ids, labs, r = T.load_subset_cache(sb[side], s_sub[side][part], part, POOLING, layer)
            sources[side][part] = (X, ids, labs)
            r_sub[side][part] = r
    for sp in V1_SPLITS:
        if sources["pre"][sp][1:] != sources["post"][sp][1:]:
            refuse(f"v1 {sp}: the pre and post caches hold other items or labels")
    stored = {"pre/probe_train": X_tr.dtype, **{f"{side}/{sp}": sources[side][sp][0].dtype
                                                for side in SIDES for sp in V1_SPLITS}}
    not32 = {k: str(v) for k, v in stored.items() if v != np.float32}
    if not32:
        refuse(f"arm A is float32 storage (D-364 (2)), but these arrays are stored as {not32}")

    # step 4: the anchor (the probes are the selection's)
    y_cal_v1 = np.asarray(sources["pre"]["calibration"][2]) == 1
    if y_cal_v1.all():
        refuse("v1 calibration holds no negative")
    probes = T.seed_probes(X_tr, y_tr, arm, seeds, holdout)
    v1_rows = T.tprs_at(probes, sources["pre"]["calibration"][0], ~y_cal_v1, sr["target_fpr"])
    tprs_v1 = [r["tpr"] for r in v1_rows]
    diffs, n_cmp = T.anchor_diffs(tprs_v1, sr["selected"])
    if diffs:
        for d in diffs:
            print(f"  DIFFERENT {d}")
        raise SystemExit(f"ANCHOR FAILED: {len(diffs)} of {n_cmp} values are not equal to {sr['sel_rel']}; nothing "
                         "written")
    print(f"  ANCHOR OK: {n_cmp} of {n_cmp} values equal, bit for bit (the probes are the selection's)", flush=True)
    del X_tr

    # step 5: v2 calibration and v2 evaluation, gathered by item_id under the join rule
    mats: dict[str, dict[str, np.ndarray]] = {}
    gathered: dict[str, Any] = {}
    for side in SIDES:
        Xc, Xc_pos, cc, ccp = gather_split("calibration", v2["ids"], [0] * v2["n"], sources[side])
        Xn, Xp, cn, cp = gather_split("eval", ev2["ids"], ev2["labels"], sources[side])
        if Xc_pos.shape[0] or cc != v2["plan_counts"] or cn != ev2["expect_neg"] or cp != ev2["expect_pos"]:
            refuse(f"{side}: v2 items per v1 split (calibration {cc}, eval negatives {cn}, eval positives {cp}) are not "
                   f"the v2 record's plan (calibration {v2['plan_counts']}, eval negatives {ev2['expect_neg']}, eval "
                   f"positives {ev2['expect_pos']})")
        mats[side] = {"calibration": Xc, "eval_neg": Xn, "eval_pos": Xp}
        gathered[side] = {"calibration": cc, "eval_negatives": cn, "eval_positives": cp}
    del sources
    neg_ids = [i for i, lab in zip(ev2["ids"], ev2["labels"]) if lab == 0]
    pool_ids = list(v2["ids"]) + neg_ids
    n_cal, n_ev, n_pos = v2["n"], ev2["n_neg"], ev2["n_pos"]

    # step 6: the two arms, the same probes and draws
    draws = P.repartition_draws(n_cal, n_ev, n_draws, DRAW_SEED)
    reports: dict[str, dict[str, Any]] = {}
    arrays: dict[str, dict[float, dict[str, np.ndarray]]] = {}
    scored: dict[str, Any] = {}
    score_sha: dict[str, Any] = {}
    for a in ("A", "B"):
        X = arm_inputs(mats, a)
        scored[a] = {side: {m: fingerprint(X[side][m]) for m in MATRICES} for side in SIDES}
        S = {side: {m: scores(probes, X[side][m]) for m in MATRICES} for side in SIDES}
        score_sha[a] = {side: {m: sha256_bytes(S[side][m].astype("<f8").tobytes()) for m in MATRICES} for side in SIDES}
        reports[a], arrays[a] = {}, {}
        for t in TARGETS:
            rep, arr = P.repartition_estimate(S["pre"]["calibration"], S["pre"]["eval_neg"], S["post"]["calibration"],
                                              S["post"]["eval_neg"], S["pre"]["eval_pos"], S["post"]["eval_pos"],
                                              draws, t)
            reports[a][f"target_{t:g}"] = rep
            arrays[a][t] = arr
        del X, S
    cast = {side: {m: cast_effect(mats[side][m]) for m in MATRICES} for side in SIDES}
    comparison = {f"target_{t:g}": compare(reports["A"][f"target_{t:g}"], arrays["A"][t], reports["B"][f"target_{t:g}"],
                                           arrays["B"][t], len(seeds)) for t in TARGETS}
    prim, beside = comparison[f"target_{PRIMARY_TARGET:g}"], comparison[f"target_{BESIDE_TARGET:g}"]
    prim["role"] = "PRIMARY: the verdict (D-364 (1), (3))"
    beside["role"] = "BESIDE: the same numbers and rule at 5%, never the verdict (D-364 (1); D-361 (4))"
    verdict = "ALLOWED" if prim["criterion_met"] else "NOT_ALLOWED"

    # step 7: nothing read may have changed; the record
    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"]:
        refuse("source changed during the run")
    files_read = {sr["sel_rel"]: sr["sel_path"], sr["rec_rel"]: sr["rec_path"], post["rel"]: post["path"],
                  v2["rel"]: v2["path"], T.V2_CAL_FILE: SP.SPLIT_ROOT / T.V2_CAL_FILE,
                  V2_EVAL_FILE: SP.SPLIT_ROOT / V2_EVAL_FILE,
                  rel(index["pre"][0]): index["pre"][0], rel(index["post"][0]): index["post"][0]}
    for side in SIDES:
        files_read.update({sb[side]["rel"]: sb[side]["path"], sb[side]["rest_rel"]: sb[side]["rest_path"],
                           rel(sb[side]["index_path"]): sb[side]["index_path"]})
    files_sha = {k: sha256_file(p) for k, p in files_read.items()}
    want_sha = {**dict(_RESULT_READS), T.V2_CAL_FILE: v2["sha256"], V2_EVAL_FILE: ev2["sha256"]}
    moved = sorted(k for k, h in files_sha.items() if k in want_sha and h != want_sha[k])
    if moved:
        refuse(f"changed during the run: {moved}")
    caches = {f"pre/{AXIS}/probe_train/{POOLING}": r_tr, f"pre/{AXIS}/calibration/{POOLING}": r_cal["pre"],
              f"post/{AXIS}/calibration/{POOLING}": r_cal["post"]}
    for side in SIDES:
        for part in ("eval", "pool4d"):
            caches[f"{side}/{SUB_TAG}/{AXIS}/{part}/{POOLING}"] = r_sub[side][part]
    C = None if arm == "mean_diff" else float(arm.split("=")[1])
    config = {"cell": "D-364 (1)", "pre_repo": PRE_REPO, "pre_rev": sr["rev"], "post_repo": POST_REPO,
              "post_rev": post["rev"], "axis": AXIS, "pooling": POOLING, "split_version": 2,
              "level": "item (D-156); the D-164 cluster level is not part of D-364",
              "arm": arm, "layer": layer, "C": C, "seeds": seeds, "holdout": holdout,
              "selection_record": sr["sel_rel"],
              "settings_from": "the selection record of D-364 (1): arm, layer, C, seeds, holdout (none is a command-line "
                               "option)",
              "targets": {"primary": PRIMARY_TARGET, "beside": [BESIDE_TARGET]}, "n_draws": n_draws,
              "draw_seed": DRAW_SEED,
              "draw_seed_note": "paired.AXIS_DRAW_SEED holds no Phase 2 axis; this check's draws are shared by both arms "
                                "and both targets, so A - B does not depend on the seed (D-181, by D-156)",
              "estimator": "D-156 as D-361 (2): paired.repartition_estimate, unchanged; v2 calibration and v2 eval "
                           "negatives pooled (calibration first, then eval negatives, each in its v2 file's order), "
                           "re-partitioned at the v2 sizes; the frozen split as one more draw; the v2 eval positives "
                           "fixed; per-seed thresholds on the calibration part of the PRE scores; FPR floored at "
                           "1/|evaluation part|",
              "arms": {"A": "every feature as stored (float32)",
                       "B": "every scored feature (pre and post; v2 calibration, v2 eval negatives and positives) cast "
                            "to float16 and back to float32 before scoring"},
              "probe": "per seed, selection_tpr_v2.seed_probes on the stored float32 probe_train (the selection's "
                       "reporting block), fitted once and used in both arms; anchored bit for bit on the selection "
                       "record's ho20 TPRs",
              "criterion": {"rule": CRITERION, "log_tolerance": TOL_LOG, "tpr_tolerance": "1/1000", "strict": True,
                            "verdict_on": f"the {prim['n_numbers']} numbers at {PRIMARY_TARGET:g}"},
              "join_rule": v2["join_rule"], "blas_threads": threads,
              "inputs_from": "the selection record; both models' select records and durable rest subsets (ingested "
                             "caches in data/hidden only); the v2 record and its calibration and eval files"}
    host = {"this_run": {k: PV.env_state().get(k) for k in SP.HOST_KEYS},
            "selection_record": {k: (sr["sel"].get("env") or {}).get(k) for k in SP.HOST_KEYS}}
    host["equal"] = host["this_run"] == host["selection_record"]
    probe_fp = [{"seed": p["seed"], "mu_sha256": sha256_bytes(np.ascontiguousarray(p["mu"]).tobytes()),
                 "sd_sha256": sha256_bytes(np.ascontiguousarray(p["sd"]).tobytes()),
                 "w_sha256": sha256_bytes(np.ascontiguousarray(p["w"]).tobytes()), "b": float(p["b"]),
                 "dtypes": {"mu": str(p["mu"].dtype), "sd": str(p["sd"].dtype), "w": str(np.asarray(p["w"]).dtype)}}
                for p in probes]
    digest = write_results(out_path, {
        **provenance_block("float16_check_d364"),
        "decision": "D-364 (pre-registration: the float16 one-cell check; the author's D-318 (7) item 5, D-357 (B)(3), "
                    "(B)(5)); D-361 (2) (the D-156 estimator on Phase 2, split v2); D-320 (4) (the join rule); D-307 (1) "
                    "(durable caches only); D-117 (the selection of record); D-156",
        "role": "storage validation (D-364 (4)): decides whether the LoRA user-span-mean subsets may be stored in "
                "float16; not an analysis that enters the paper. The A values are Phase 2 evaluation numbers of one "
                "cell, reported only as part of this check",
        "verdict": {"verdict": verdict, "criterion_met_at_1pct": prim["criterion_met"],
                    "n_numbers_within_at_1pct": f"{prim['n_within']} of {prim['n_numbers']}",
                    "max_abs_log_B_over_A_at_1pct": prim["max_abs_log_B_over_A"],
                    "max_abs_tpr_difference_at_1pct": prim["max_abs_tpr_difference"],
                    "beside_5pct_criterion_met": beside["criterion_met"],
                    "beside_5pct_n_within": f"{beside['n_within']} of {beside['n_numbers']}",
                    "rule": CRITERION,
                    "consequence": "ALLOWED: the LoRA user-span-mean subsets may be stored in float16; NOT_ALLOWED: "
                                   "float16 is not used and every LoRA subset stays float32 (D-364 (3))"},
        "code": code0,
        "registered_parameters": registered,
        "config": config,
        "config_hash": canonical_hash(config),
        "inputs": dict(_RESULT_READS),
        "input_files_sha256": files_sha,
        "input_cache_sha256": {k: r["sha256"] for k, r in caches.items()},
        "input_cache_source": {k: f"data/hidden/{r['file']} (expected bytes from {r['expected_from']})"
                               for k, r in caches.items()},
        "durable_inputs": "every cache read from data/hidden: a regular file, mode 444, in its record's ingest index "
                          "with the record's sha256, hashing to it, item for item its frozen split (D-307 (1))",
        "selection_record": {"file": sr["sel_rel"], "sha256": _RESULT_READS[sr["sel_rel"]]},
        "records": {"pre_select": {"file": sr["rec_rel"], "sha256": _RESULT_READS[sr["rec_rel"]]},
                    "post_select": {"file": post["rel"], "sha256": _RESULT_READS[post["rel"]]},
                    **{f"{side}_subset": {"file": sb[side]["rel"], "sha256": _RESULT_READS[sb[side]["rel"]]}
                       for side in SIDES},
                    **{f"{side}_rest": {"file": sb[side]["rest_rel"], "sha256": _RESULT_READS[sb[side]["rest_rel"]]}
                       for side in SIDES},
                    "v2": {"file": v2["rel"], "sha256": _RESULT_READS[v2["rel"]],
                           "written_by": {T.V2_CODE_FILE: v2["code_sha256"], "equal_to_this_tree": True},
                           "excluded_items": v2["excluded_items"]}},
        "pinned": {"checked": final, "values": PINNED},
        "subset_layer_plans": l_star,
        "v1_keyed_caches": {"rule": "D-320 (4): every cache read is a v1 cache keyed p2wg/<v1 split>, its split file the "
                                    "v1 file the v2 record names; read only through the join rule", "split_files": v1_keyed},
        "host": host,
        "split_files": {"v1/probe_train": r_tr["split"], "v1/calibration": r_cal["pre"]["split"],
                        "v1/eval": r_sub["pre"]["eval"]["split"], "v1/pool4d": r_sub["pre"]["pool4d"]["split"],
                        "v2/calibration": {"file": T.V2_CAL_FILE, "sha256": v2["sha256"]},
                        "v2/eval": {"file": V2_EVAL_FILE, "sha256": ev2["sha256"]}},
        "subset_caches": {side: {part: {k2: r_sub[side][part][k2] for k2 in ("layers_resolved", "source_cache")}
                                 for part in ("eval", "pool4d")} for side in SIDES},
        "code_binding": {"files": {f: code0["sha256"][f] for f in T.BOUND_FILES}, "equal_to_the_selection_record": True},
        "selected": {"arm": arm, "layer": layer, "C": C},
        "anchor": {"ho20_tpr_per_seed_recorded": sr["selected"]["ho20_tpr_per_seed"],
                   "ho20_tpr_per_seed_recomputed": tprs_v1, "values_compared": n_cmp, "values_differing": 0,
                   "exact_equal": True, "threshold_population": "v1 calibration's negatives (the selection's)"},
        "v2_gather": {"per_side": gathered,
                      "expected_from_the_v2_plan": {"calibration": v2["plan_counts"], "eval_negatives": ev2["expect_neg"],
                                                    "eval_positives": ev2["expect_pos"]},
                      "pre_and_post_same_items": True},
        "pooled_negatives": {"n_calibration": n_cal, "n_evaluation": n_ev,
                             "item_ids_sha256": sha256_bytes("\n".join(pool_ids).encode("utf-8")),
                             "order": "v2 calibration (its file's order), then v2 eval negatives (its file's order)"},
        "n_eval_positives": n_pos,
        "probes": {"fitted_on": f"the stored float32 probe_train ({len(y_tr)} items) at layer {layer}, in both arms",
                   "per_seed": probe_fp},
        "scored_inputs": scored,
        "scores_sha256": score_sha,
        "cast_effect": cast,
        "operating_points": reports,
        "comparison": comparison,
    }, exclusive=True)
    m1, m2 = prim["max_abs_log_B_over_A"], prim["max_abs_tpr_difference"]
    print(f"VERDICT {verdict} (D-364 (3), 1%): {prim['n_within']} of {prim['n_numbers']} numbers within; max |log(B/A)| "
          f"{m1['value']!r} at {m1['at']} (tolerance {TOL_LOG}); max |dTPR| {m2['value']!r} at {m2['at']} (tolerance "
          f"0.001); 5% beside: {beside['n_within']} of {beside['n_numbers']} within")
    print(f"Written: {out_rel}\nsha256:  {digest}")
    return out_path


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="D-364: the float16 one-cell check (Mistral-7B-Instruct v0.1 -> v0.2, "
                                             "user-span mean, L*, split v2); the cell is fixed, not an option")
    ap.add_argument("--out", default=None, help="a test: write here (never under results/); default the final name")
    ap.add_argument("--n-draws", type=int, default=P.N_DRAWS, help="a test only; the final name needs 1000")
    args = ap.parse_args()
    run(args.n_draws, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
