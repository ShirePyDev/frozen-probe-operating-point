"""Phase 3 attacker A1: the by-c heavy-tail check DECISIONS.md D-370 pre-registers ((3) the measures, (4) the reading;
quoted whole in PREREG_TEXT and in the record). Do P-A1.3's misses and P-A1.4's false alarms share a heavier-than-Gaussian
upper tail of the benign post-update scores? No A1 verdict changes (D-353 stands).

INPUTS, bound as src/phase3_a1_pooled.py binds them and by its functions (discover, load, record_problems, shared_code,
prediction_problems): the nine white-box A1 records results/phase3_a1_<pair>.json and their _draws.npz, the predictions
records they name and their npz (36 files), and the D-353 pooled record results/phase3_a1_pooled.json (its sha256 pinned,
D353_POOLED_SHA256, when the records are results/'s; it must have bound the same 36 files by sha256 and been made by this
tree's src/phase3_a1_pooled.py and src/healthcheck_summary_d156.py, PINNED_SHA256). The records' A1 code block must be
this tree's, file for file (the tail statistic re-scores the caches through it), with src/phase3_a1.py and
src/phase3_a1_predictions.py the reviewed blobs (phase3_a1_pooled.REVIEWED_SHA256). For the tail statistic, each pair's
Phase 1 caches through phase3_a1_predictions.setup, unchanged (the same draws, seeds, L*, selected C and E).

MEASURES (D-370 (3)), white-box, per pair and c != 0:
  false alarms   per draw, the cell's seed-mean flag (the 4c Gaussian health flag at 2 / 0.5, and the direct detector's,
                 D-264) against the realised class (realised paired ratio >= 2 or <= 0.5), by phase3_a1.confusion on
                 phase3_a1_pooled.flag_arrays of one c at a time; ANCHORS: per c, the flags read give the record's cell
                 at that c (its shares of draws flagged and in the realised class, and the frozen split's flag and
                 class, phase3_a1.summaries), so no c is read for another; summed over c, every pair's per-draw counts
                 equal its record's P-A1.4 arrays (every detector), and pooled by healthcheck_summary_d156.pooled_counts
                 with the pair-cluster index they reproduce the pooled record's white-box P-A1.4 blocks exactly; else
                 it refuses. Per pair and c: the D-156 mean counts (draws 1..), the false-alarm share fp / (fp + tn)
                 and the sensitivity tp / (tp + fn) of those means (choice (a)); pooled by c, over c < 0, c > 0 and
                 every c != 0 by the same estimator (point only).
  log error      per cell the verdict line's realised and predicted FPR/target (phase3_a1_pooled.cell_rows, white-box,
                 unfloored) and log(realised / predicted), signed; ANCHOR: the 90 rows equal the pooled record's
                 white-box rows and phase3_a1.accuracy over them gives its verdict line (n cells, share within x1.5,
                 median |log|) exactly; else it refuses. D-177's accuracy line per c, per pair over c < 0 and c > 0,
                 and per c set; the floored prediction's mean log error beside (choice (k)).
  tail          tau = (q99 - m) / s - Phi^-1(0.99) of the post scores at c = 0 of the draw's E, per draw and seed (m, s
                 ddof 0 as D-346 (2)(b), formed exactly as phase3_a1_predictions.build forms them and required equal to
                 its m and s bit for bit; q99 numpy.quantile's default linear interpolation, choice (d)); per draw the
                 seed mean; the D-156 mean over the draws. ANCHORS: the A1 record's own input_cache_sha256 equals the
                 re-scored caches'; phase3_a1.check_predictions_against, unchanged: the
                 draws (index hash), the caches, the variants and every predictions array (T, sigma, m, s, every shift
                 and predicted FPR) equal the predictions record's npz bit for bit; else it refuses. A test at fewer
                 draws (--n-reps) checks the arrays' leading rows instead (the draws are a prefix of the registered
                 ones: poolstudy_d156.partitions and draw_pools draw each repetition from its own stream or a fixed list).
READING (D-370 (4)), each part's value recorded: (a) the Gaussian detector's pooled false alarms (sum over pairs and c of
  the D-156 mean fp) at c < 0 are at least 80% of those over every c != 0; (b) the direct detector's pooled false-alarm
  rate at c < 0 is below the Gaussian detector's; (c) tau > 0 in at least 7 of the 9 pairs; (d) the Spearman correlation
  over the nine pairs between tau and the pair's mean log(realised/predicted) over c < 0 is positive (n = 9; no
  significance claimed, no p-value computed). SUPPORTED only if all four hold; otherwise NOT SUPPORTED. Parts (a) and
  (b) are compared exactly on integer totals (choice (h)); an undefined part carries undefined_reason and does not hold.
CHOICES BY THE CLOSEST D-ENTRY (D-181): CHOICES below, written into the record.
Writes results/phase3_a1_byc_d370.json and results/phase3_a1_byc_d370_draws.npz (per pair, q99 and tau per draw and seed;
D-163 (6)) once (exclusive), then prints the anchors and the reading. Only the records in results/, every draw, the
tail statistic and the 6-thread BLAS pin write the final names. A test reads --records-dir and writes --out, outside
results/; --n-reps and --skip-tail are for tests and the sandbox only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_d156 as D  # noqa: E402
import healthcheck_summary_d156 as HS  # noqa: E402
import input_registry as REG  # noqa: E402
import paired as P  # noqa: E402
import phase3_a1 as A1  # noqa: E402
import phase3_a1_pooled as PL  # noqa: E402
import phase3_a1_predictions as A1P  # noqa: E402
from provenance import (  # noqa: E402
    canonical_hash, ensure_utf8_stdout, provenance_block, read_result, resolve_cache, sha256_bytes, sha256_file,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FINAL_OUT = RESULTS / "phase3_a1_byc_d370.json"
POOLED_NAME = "phase3_a1_pooled.json"
THIS_FILE = "src/phase3_a1_byc_d370.py"
CODE_FILES = list(dict.fromkeys([THIS_FILE, *PL.CODE_FILES]))
PAIRS = PL.PAIRS
PRIMARY = PL.PRIMARY                                                           # white-box (D-346 (2)(g), D-370 (2))
NZ = PL.NZ                                                                     # c != 0
NEG = [A1P.ckey(c) for c in A1P.C_GRID if c < 0]
POS = [A1P.ckey(c) for c in A1P.C_GRID if c > 0]
DETS = ("gaussian", "direct")                                                  # D-370 (3)
# the D-353 pooled record (DECISIONS.md D-353 (1)) and the code whose functions this script calls: the A1 blobs
# (phase3_a1_pooled.REVIEWED_SHA256) and the pooled scorer's two files as that record names them
D353_POOLED_SHA256 = "0518bf06c69e8756f44bc2be3a16036d6ff83d86d6153793ef942e2645e738b5"
PINNED_SHA256 = {**PL.REVIEWED_SHA256,
                 "src/phase3_a1_pooled.py": "4e7fa8d183b18f3f0aa689c8bfef5a2189085c761df3b0d6cb8b69245e3ca423",
                 "src/healthcheck_summary_d156.py": "b149b66181fe1c2d953d255355e44e2d15651bff9ccf8a456819092db526ccf5"}
POOLED_CODE = ("src/phase3_a1_pooled.py", "src/healthcheck_summary_d156.py")
FA_SHARE_MIN = Fraction(4, 5)                                                  # D-370 (4)(a): 80%, compared exactly
TAU_PAIRS_MIN = 7                                                              # D-370 (4)(c)
Q_TAIL = 0.99
Z99 = float(stats.norm.ppf(Q_TAIL))                                            # Phi^-1(0.99)
# beside only (minor of the review): the mean tau of the same estimator on standard-normal samples of |E| = 1000
GAUSS_NULL = {"n": PL.N_E, "samples": 20000, "seed": 20261006}
HEALTH_SHARES = (("health|gaussian_flag", "share_of_draws_gaussian_flag_2x"),
                 ("health|direct_flag", "share_of_draws_direct_flag_2x"),
                 ("realised_class_2x", "share_of_draws_realised_class_2x"))     # phase3_a1.summaries' cell shares
FROZEN_FLAGS = (("health|gaussian_flag", "gaussian_flag"), ("realised_class_2x", "realised_class_2x"))
SAME_AS_RECORD = ("pair", "registry_pair", "axis", "layer", "selected_C", "probe_set", "selection_file",
                  "selection_sha256", "seeds", "target_fpr", "pool_seed", "design", "evaluation_half", "traffic_pool",
                  "c_grid", "variants", "pre_repo", "pre_rev", "post_repo", "post_rev", "cluster_manifest")
# the registry's sources (extraction records, pins) are compared beside, not required: the caches themselves are
# required equal by sha256 (check_predictions_against, as phase3_a1 checks them), and a later extraction record of
# another model must not refuse this check
SAME_AS_RECORD_REGISTERED = ("n_reps", "draws", "draws_index_sha256")
PREREG = "D-370 (3)-(4) (the A1 by-c heavy-tail check); D-327; D-346 (2)(b); D-351 (2); D-353; D-156; D-264"
PREREG_TEXT = (
    "(3) MEASURES, per pair and c != 0 (D-156 means over draws; seed-mean flags per draw, as D-351 (2)): the Gaussian "
    "detector's false-alarm share fp / (fp + tn) and sensitivity; the direct detector's (D-264, no Gaussian assumption) "
    "the same; P-A1.3's log(realised/predicted) (unfloored, as the verdict line). TAIL STATISTIC per pair (independent "
    "of any attack): tau = (q99(E post scores) - m) / s - Phi^-1(0.99), m and s the mean and SD (ddof 0, as D-346 (2)(b)) "
    "of the same scores, per draw and seed, D-156 mean; tau > 0 means an upper tail heavier than Gaussian at the 1% "
    "level. (4) READING, fixed now: the shared-heavy-tail account is SUPPORTED only if all four hold: (a) at least 80% "
    "of the Gaussian detector's pooled false alarms (sum over pairs and c of the D-156 mean fp) fall at c < 0; (b) the "
    "direct detector's pooled false-alarm rate at c < 0 is below the Gaussian detector's; (c) tau > 0 in at least 7 of "
    "the 9 pairs; (d) across the nine pairs the Spearman correlation between tau and the pair's mean "
    "log(realised/predicted) over c < 0 is positive (reported with n = 9; no significance claimed). Otherwise NOT "
    "SUPPORTED, with each part's value. Either way no A1 verdict changes (D-353 stands); the result informs how C2, C5 "
    "and C6 are worded.")
CONFIG = {
    "pairs": PAIRS, "variant": PRIMARY, "c_grid": list(A1P.C_GRID), "c_scored": NZ, "c_negative": NEG, "c_positive": POS,
    "detectors": list(DETS), "class_and_flag_cut": list(A1.CUT), "per_draw_unit": "the seed-mean flag (D-351 (2))",
    "reading": {"a_min_share_of_gaussian_false_alarms_at_c_negative": float(FA_SHARE_MIN),
                "b": "the direct detector's pooled false-alarm rate at c < 0 strictly below the Gaussian detector's",
                "a_b_comparison": "exact, as fractions of integer totals (fp and tn summed over the pairs, the c set and "
                                  "draws 1..; every pair has the same draws, so these are the ratios of the summed "
                                  "D-156 mean counts)",
                "c_min_pairs_with_tau_above_0": TAU_PAIRS_MIN, "c_tau_cut": "tau > 0 (strict), on the D-156 mean",
                "d": "Spearman correlation (scipy.stats.spearmanr, average ranks) > 0 (strict), n = 9, no p-value",
                "n_pairs": len(PAIRS)},
    "tail": {"scores": "the post scores at c = 0 of the draw's E (1000 negatives), per draw and seed", "quantile": Q_TAIL,
             "quantile_method": "numpy.quantile default ('linear'), float64", "moments_ddof": 0, "z": Z99,
             "z_source": "scipy.stats.norm.ppf(0.99)", "summary": "per draw the seed mean; the D-156 mean over draws 1..",
             "gaussian_null_beside": GAUSS_NULL},
    "pooled_record": {"name": f"results/{POOLED_NAME}", "pinned_sha256": D353_POOLED_SHA256, "decision": "D-353"},
    "bootstrap_of_the_anchor": {"B": HS.B, "seed": HS.SEED,
                                "role": "only to reproduce the pooled record's P-A1.4 blocks; no by-c interval"},
}
CHOICES = [
    "(a) Per pair and c, the false-alarm share is the D-156 mean fp over the D-156 mean fp + tn (draws 1.., the frozen "
    "split beside), and the sensitivity the D-156 mean tp over tp + fn: the ratio of the means, D-353's 'D-156 mean fp of "
    "fp + tn per draw' and D-255's pooled form (healthcheck_summary_d156.pooled_counts, pooled_from_mean_counts). With one "
    "cell per pair, c and draw it equals the share of the draws in the realised negative class that are flagged. "
    "Undefined (None) when no draw has the cell in that class.",
    "(b) Part (a): the share is the sum over the nine pairs and the five c < 0 of the Gaussian D-156 mean fp over the same "
    "sum over every c != 0 (the latter is the P-A1.4 verdict line's mean fp, reproduced from the pooled record). With no "
    "false alarm at all the share is undefined and the part does not hold.",
    "(c) Part (b): each detector's pooled false-alarm rate at c < 0 is fp / (fp + tn) of the D-156 mean counts summed over "
    "the nine pairs and the five c < 0 (D-255's pooled form, as P-A1.4's specificity is formed); 'below' is strict; an "
    "undefined rate (no negative-class cell at c < 0) makes the part not hold.",
    "(d) The tail statistic: per draw d and seed s, x the post scores at c = 0 of E_d (phase3_a1_predictions.setup's "
    "scores, unchanged: SC['post']['pool'][E]); m = x.mean(axis=0), s = x.std(axis=0) (ddof 0) exactly as "
    "phase3_a1_predictions.build forms them and required equal to its m and s bit for bit; q99 = numpy.quantile(x, 0.99, "
    "axis=0) in float64 with numpy's default linear interpolation (the implementer's choice, recorded); tau = (q99 - m) / "
    "s - scipy.stats.norm.ppf(0.99). Per draw the seed mean, then the D-156 mean over draws 1.. (band and frozen split "
    "beside). Part (c) compares the D-156 mean with 0 strictly; a pair with no finite tau does not count.",
    "(e) Part (d): the pair's mean of the signed log(realised/predicted) over its five c < 0 cells, realised and predicted "
    "the verdict line's (phase3_a1_pooled.cell_rows, white-box, unfloored). A zero prediction gives +inf (a miss, D-322 "
    "(4)), ranked highest (recorded as None with the count; D-353: no cell was predicted zero). The Spearman correlation "
    "is scipy.stats.spearmanr (average ranks for ties); 'positive' is > 0; an undefined correlation (a constant vector, "
    "or a pair with no finite tau) is not positive. No p-value is computed (D-370 (4): no significance claimed).",
    "(f) White-box only (D-370 (2): the white-box A1 records; D-346 (2)(g)); c != 0 scored, c = 0 not used for the false "
    "alarms; the per-draw unit is the seed-mean flag (D-351 (2)); the transfer variant is not read.",
    "(g) Binding: the records' A1 code block must equal this tree's file for file (the tail statistic re-scores the "
    "caches through it, so 'unchanged' is checked, not assumed), src/phase3_a1.py and src/phase3_a1_predictions.py the "
    "reviewed blobs, src/phase3_a1_pooled.py and src/healthcheck_summary_d156.py the D-353 record's (PINNED_SHA256); the "
    "pooled record is D-353's by sha256 when the records are results/'s.",
    "(h) FIX ROUND (three reviews of 94e7dc6/b6b7594; rules (a)-(g) unchanged): parts (a) and (b) are compared exactly, "
    "as fractions of integer totals (fp and tn summed over the nine pairs, the c set and draws 1..): (a) 5 j >= 4 k with "
    "j, k the Gaussian fp totals at c < 0 and at every c != 0; (b) the direct fp / (fp + tn) below the Gaussian's as "
    "fractions. Every pair has the same draws, so these are exactly the ratios of the summed D-156 mean counts of (b) "
    "and (c) above, without their floating-point sums.",
    "(i) The q99 estimator's offset, recorded, not corrected (D-370 (3) fixed tau): numpy's linear q99 of |E| = 1000 "
    "scores, standardised by the sample m and s, runs low under a Gaussian null (the review's figure: about 0.02), so "
    "'tau > 0' means an upper tail heavier than Gaussian by about that much. The record's "
    "tail_statistic_gaussian_null_beside gives the same estimator's mean tau over standard-normal samples of 1000 "
    "(GAUSS_NULL: count and seed) with its Monte Carlo standard error, beside only.",
    "(j) Dtypes, recorded per pair (tail_statistic.per_pair.<pair>.dtypes): m and s are computed in the post scores' "
    "own dtype exactly as build computes them (numpy's mean and std of that array); q99 on the same scores cast to "
    "float64; tau in float64. No computation changes with the dtype.",
    "(k) Beside part (d), not in the reading: the pair's mean log(realised/predicted) over c < 0 with the prediction "
    "floored per seed at 1/|E| (the realised side's floor; the pooled record's floored prediction, D-346 (2)(e)); and "
    "D-177's accuracy line (phase3_a1.accuracy, unfloored: n cells, share within x1.5, median |log|) per c, per pair "
    "over c < 0 and over c > 0, and pooled over each c set.",
]


def refuse(msg: str) -> None:
    raise SystemExit(f"REFUSING: {msg}")


def jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, default=str))


# ---- the inputs --------------------------------------------------------------------------------------------------------

def bind(rdir: Path, scorer_sha: dict[str, str], pin: bool) -> dict[str, Any]:
    """The nine A1 records and their predictions records, bound and checked by phase3_a1_pooled's functions (every
    refusal of its pool()), the records' A1 code block equal to this tree's, and the D-353 pooled record."""
    paths = PL.discover(rdir)
    files: dict[str, tuple[Path, str]] = {}
    prs = {c: PL.load(c, p, files) for c, p in paths.items()}
    bad = {c: b for c, b in ((c, PL.record_problems(c, pr["doc"])) for c, pr in prs.items()) if b}
    if bad:
        refuse(f"not every record can enter: {bad}")
    params = {c: {k: pr["doc"]["config"].get(k) for k in ("n_reps", "seeds", "target_fpr")} for c, pr in prs.items()}
    if any(v != params[PAIRS[0]] for v in params.values()):
        refuse(f"the nine records were run at different draws, seeds or target: {params}")
    off = [f for f, h in PINNED_SHA256.items() if scorer_sha.get(f) != h]
    if off:
        refuse(f"this tree's {off} are not the pinned blobs (PINNED_SHA256)")
    code = PL.shared_code(prs, scorer_sha)
    if code["a1_files_differing_from_this_tree"]:
        refuse(f"the records' A1 code block differs from this tree's: {code['a1_files_differing_from_this_tree']} "
               f"(the tail statistic re-scores through it unchanged)")
    pbad = {c: b for c, b in ((c, PL.prediction_problems(c, pr, code["predictions_code_sha256"])) for c, pr in prs.items()) if b}
    if pbad:
        refuse(f"not every predictions record is its A1 record's: {pbad}")
    pp = rdir / POOLED_NAME
    if not pp.is_file():
        refuse(f"{POOLED_NAME} (the D-353 pooled record) is missing from {rdir}")
    pdoc = read_result(pp)                                  # a registered read (D-297 (1)), as phase3_a1_pooled.load
    psha = sha256_file(pp)
    if pin and psha != D353_POOLED_SHA256:
        refuse(f"{POOLED_NAME} is not the D-353 record ({psha[:12]} vs {D353_POOLED_SHA256[:12]})")
    if pdoc.get("run_kind") != "phase3_a1_pooled":
        refuse(f"{POOLED_NAME} is not a pooled A1 record (run_kind {pdoc.get('run_kind')!r})")
    if pdoc.get("input_files_sha256") != {n: h for n, (_, h) in files.items()}:
        refuse(f"{POOLED_NAME} did not bind these 36 inputs by sha256")
    pcode = (pdoc.get("code") or {}).get("sha256") or {}
    off = [f for f in POOLED_CODE if pcode.get(f) != scorer_sha.get(f)]
    if off:
        refuse(f"{POOLED_NAME} was made by other {off} than this tree's")
    files[A1P.rel(pp)] = (pp, psha)
    return {"prs": prs, "files": files, "code": code, "params": params[PAIRS[0]],
            "pooled": {"doc": pdoc, "file": A1P.rel(pp), "sha256": psha, "pinned_to_d353": bool(pin)}}


# ---- false alarms by c ---------------------------------------------------------------------------------------------------

def counts_by_c(prs: dict[str, dict[str, Any]]) -> dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]:
    """core -> c -> detector -> count -> per-draw array (one cell per draw: 0 or 1), by phase3_a1.confusion on one c
    at a time. ANCHORS: per c, the flags read are the record's cell at that c (the shares of draws with the Gaussian
    flag, the direct flag and the realised class, and the frozen split's Gaussian flag and class, as
    phase3_a1.summaries writes them), so no c is read for another; summed over c != 0, each pair's per-draw counts are
    its record's P-A1.4 arrays, every detector."""
    out: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
    for core in PAIRS:
        z = prs[core]["z"]
        cells = (prs[core]["doc"].get("cells") or {}).get(PRIMARY) or {}
        per = {}
        for kc in NZ:
            fa = PL.flag_arrays(z, PRIMARY, [kc])
            h = (cells.get(kc) or {}).get("health") or {}
            off = [name for key, name in HEALTH_SHARES if float(fa[kc][key][1:].mean()) != h.get(name)]
            fz = h.get("frozen_split") or {}
            off += [f"frozen_split.{name}" for key, name in FROZEN_FLAGS if bool(fa[kc][key][0]) != fz.get(name)]
            if off:
                refuse(f"{core} {kc}: the flags read for this c are not the record's cell at {kc}: {off}")
            per[kc] = A1.confusion(fa, [kc])
        out[core] = {kc: {det: per[kc][det]["_arrays"] for det in HS.DETECTORS} for kc in NZ}
        for det in HS.DETECTORS:
            for q in HS.COUNTS:
                key = f"P-A1.4|{PRIMARY}|{det}|{q}"
                tot = np.sum([per[kc][det]["_arrays"][q] for kc in NZ], axis=0)
                if key not in z or not np.array_equal(tot, z[key]):
                    refuse(f"{core}: the per-c counts summed over c are not the record's {key}")
    return out


def rates(mc: dict[str, float]) -> dict[str, Any]:
    """Choice (a): the false-alarm share and the sensitivity of D-156 mean counts (None with no cell of the class)."""
    neg, pos = mc["fp"] + mc["tn"], mc["tp"] + mc["fn"]
    return {"false_alarm_share": float(mc["fp"] / neg) if neg > 0 else None,
            "sensitivity": float(mc["tp"] / pos) if pos > 0 else None}


def pooled_line(cnt: dict, names: list[str], cs: list[str], det: str, idx: np.ndarray | None = None) -> dict[str, Any]:
    """D-255's estimator (healthcheck_summary_d156.pooled_counts) over the pairs `names` and the cells `cs`."""
    return HS.pooled_counts({q: [np.sum([cnt[n][kc][det][q] for kc in cs], axis=0) for n in names] for q in HS.COUNTS}, idx)


def int_totals(cnt: dict, names: list[str], cs: list[str], det: str) -> dict[str, int]:
    """Choice (h): each count summed over the pairs `names`, the cells `cs` and draws 1.. (the frozen split, row 0,
    left out, as the D-156 means leave it out), as Python integers."""
    return {q: int(sum(int(np.asarray(cnt[n][kc][det][q])[1:].sum()) for n in names for kc in cs)) for q in HS.COUNTS}


def anchor_p14(cnt: dict, pooled_doc: dict[str, Any]) -> dict[str, Any]:
    """ANCHOR: pooled over the nine pairs and every c != 0 with the pair-cluster index, the per-c counts give the
    pooled record's white-box P-A1.4 block of every detector exactly."""
    idx = HS.pair_cluster_index(len(PAIRS))
    rec = (pooled_doc.get("P-A1.4") or {}).get(PRIMARY) or {}
    off = [det for det in HS.DETECTORS if jsonable(pooled_line(cnt, PAIRS, NZ, det, idx)) != rec.get(det)]
    if off:
        refuse(f"the per-c counts pooled over c do not reproduce the pooled record's P-A1.4 {PRIMARY} block: {off}")
    return {"checked": True, "identical": True, "detectors": list(HS.DETECTORS),
            "what": "per pair, the per-draw counts summed over c != 0 equal the record's P-A1.4 arrays (every detector); "
                    "pooled by healthcheck_summary_d156.pooled_counts with the pair-cluster index (B 5000, seed "
                    "520260925) they equal the pooled record's P-A1.4.white_box block of every detector (JSON-exact)"}


def false_alarms(cnt: dict) -> dict[str, Any]:
    """Per pair and c, per pair over c < 0 / c > 0 / c != 0, and pooled over the pairs by c and by c set."""
    per_pair: dict[str, Any] = {}
    for core in PAIRS:
        per_pair[core] = {}
        for kc in NZ:
            blk = {}
            for det in DETS:
                a = cnt[core][kc][det]
                mc = {q: float(a[q][1:].mean()) for q in HS.COUNTS}
                blk[det] = {"d156_mean_counts": mc, "frozen_split_counts": {q: int(a[q][0]) for q in HS.COUNTS},
                            **rates(mc)}
            per_pair[core][kc] = blk
        for name, cs in (("c_negative", NEG), ("c_positive", POS), ("c_nonzero", NZ)):
            per_pair[core][name] = {}
            for det in DETS:
                mc = pooled_line(cnt, [core], cs, det)["mean_counts_over_draws"]
                per_pair[core][name][det] = {"d156_mean_counts": mc, **rates(mc)}
    pooled: dict[str, Any] = {}
    for name, cs in [(kc, [kc]) for kc in NZ] + [("c_negative", NEG), ("c_positive", POS), ("c_nonzero", NZ)]:
        pooled[name] = {}
        for det in DETS:
            pc = pooled_line(cnt, PAIRS, cs, det)
            pooled[name][det] = {"d156_mean_counts": pc["mean_counts_over_draws"], **rates(pc["mean_counts_over_draws"]),
                                 "integer_totals_draws_1_on": int_totals(cnt, PAIRS, cs, det),
                                 "per_draw_sensitivity": pc["per_draw_sensitivity"],
                                 "per_draw_specificity": pc["per_draw_specificity"]}
    return {"unit": "per draw one cell per pair and c: the seed-mean flag against the realised class",
            "per_pair": per_pair, "pooled_over_pairs": pooled}


# ---- P-A1.3's log error by c -------------------------------------------------------------------------------------------

def anchor_p13(rows: list[dict[str, Any]], pooled_doc: dict[str, Any]) -> dict[str, Any]:
    """ANCHOR: the 90 white-box rows are the pooled record's, and phase3_a1.accuracy over them is its verdict line."""
    p13 = pooled_doc.get("P-A1.3") or {}
    rec_rows = [r for r in p13.get("rows") or [] if r.get("variant") == PRIMARY]
    if jsonable(rows) != rec_rows:
        refuse(f"the {len(rows)} white-box P-A1.3 rows are not the pooled record's")
    acc = A1.accuracy(rows, "predicted_ratio_to_target")
    sc = p13.get("scored") or {}
    keys = ("n_cells", "n_cells_predicted_zero", "share_within_factor_1_5", "median_abs_log_realised_over_predicted")
    off = [k for k in keys if acc[k] != sc.get(k)]
    if off or sc.get("line") != f"lines.{PRIMARY}.primary_unfloored":
        refuse(f"the 90 cells do not reproduce the pooled record's P-A1.3 verdict line: {off}")
    return {"checked": True, "identical": True, "n_cells": acc["n_cells"],
            "what": "the white-box rows (phase3_a1_pooled.cell_rows) equal the pooled record's, and phase3_a1.accuracy "
                    "over them gives its verdict line's n cells, predicted-zero count, share within x1.5 and median "
                    "|log(realised/predicted)| exactly"}


def signed_log(r: dict[str, Any]) -> float:
    """log(realised / predicted), unfloored; +inf for a zero prediction (a miss, D-322 (4))."""
    p = r["predicted_ratio_to_target"]
    return float(np.log(r["realised_ratio_to_target"] / p)) if p > 0 else float("inf")


def fin(x: float | None) -> float | None:
    return float(x) if x is not None and np.isfinite(x) else None


def signed_log_floored(r: dict[str, Any]) -> float:
    """Beside (choice (k)): log(realised / predicted floored per seed at 1/|E|), the pooled record's floored prediction."""
    return float(np.log(r["realised_ratio_to_target"] / r["predicted_ratio_to_target_floored"]))


def accuracy_line(rs: list[dict[str, Any]]) -> dict[str, Any]:
    """D-177's accuracy line (phase3_a1.accuracy, as phase3_a1_pooled forms it) on the unfloored prediction."""
    return A1.accuracy(rs, "predicted_ratio_to_target")


def log_errors(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per cell the signed log error; per pair its mean over c < 0 (part (d)) and c > 0 with D-177's accuracy line and
    the floored mean beside; per c and per c set over the pairs, with D-177's accuracy line."""
    cells = [{"pair": r["pair"], "c": r["c"], "realised_ratio_to_target": r["realised_ratio_to_target"],
              "predicted_ratio_to_target": r["predicted_ratio_to_target"], "log_realised_over_predicted": fin(signed_log(r)),
              "predicted_ratio_to_target_floored_beside": r["predicted_ratio_to_target_floored"],
              "log_realised_over_predicted_floored_beside": signed_log_floored(r),
              "predicted_zero": not r["predicted_ratio_to_target"] > 0} for r in rows]
    per_pair, means, means_fl = {}, {}, {}
    for core in PAIRS:
        blk = {}
        for name, cs in (("c_negative", NEG), ("c_positive", POS)):
            rs = [r for r in rows if r["pair"] == core and A1P.ckey(r["c"]) in cs]
            if len(rs) != len(cs):
                refuse(f"{core}: {len(rs)} P-A1.3 cells at {name}, not {len(cs)}")
            e = [signed_log(r) for r in rs]
            m, mf = float(np.mean(e)), float(np.mean([signed_log_floored(r) for r in rs]))
            blk[name] = {"mean_log_realised_over_predicted": fin(m), "n_cells": len(e),
                         "n_predicted_zero": int(sum(not np.isfinite(x) for x in e)),
                         "mean_log_realised_over_predicted_floored_beside": mf,
                         "accuracy_line": accuracy_line(rs)}
            if name == "c_negative":
                means[core], means_fl[core] = m, mf
        per_pair[core] = blk
    by_c = {}
    for kc in NZ:
        rs = [r for r in rows if A1P.ckey(r["c"]) == kc]
        e = np.array([signed_log(r) for r in rs])
        by_c[kc] = {"n_pairs": int(e.size), "mean_log": fin(float(e.mean())), "median_log": fin(float(np.median(e))),
                    "n_within_factor_1_5": int((np.abs(e) <= np.log(A1.FACTOR)).sum()),
                    "n_realised_above_predicted": int((e > 0).sum()), "accuracy_line": accuracy_line(rs)}
    by_set = {name: accuracy_line([r for r in rows if A1P.ckey(r["c"]) in cs])
              for name, cs in (("c_negative", NEG), ("c_positive", POS), ("c_nonzero", NZ))}
    return {"cells": cells, "per_pair": per_pair, "by_c": by_c, "accuracy_line_by_c_set": by_set,
            "accuracy_line_rule": "D-177 (phase3_a1.accuracy): n cells, share within x1.5, median "
                                  "|log(realised/predicted)|, unfloored; a zero prediction a miss (D-322 (4))",
            "_means_c_negative": means, "_means_c_negative_floored": means_fl}


# ---- the tail statistic ----------------------------------------------------------------------------------------------------

def tail_anchor(S: dict[str, Any], pdoc: dict[str, Any], parr: dict[str, np.ndarray], adoc: dict[str, Any]) -> dict[str, Any]:
    """The re-scored pair is the A1 record's: its config (pair, L*, selected C, seeds, target, selection, inputs), its
    caches by sha256 (the A1 record's own input_cache_sha256) and, at every draw, phase3_a1.check_predictions_against
    unchanged (draws, caches, variants, every array bit for bit); at fewer draws (a test) the arrays' leading rows."""
    mine = A1P.input_block(S)
    n_rec = int(adoc["config"]["n_reps"])
    full = S["n_reps"] == n_rec
    keys = SAME_AS_RECORD + (SAME_AS_RECORD_REGISTERED if full else ())
    off = [k for k in keys if jsonable(mine.get(k)) != adoc["config"].get(k)]
    if off:
        refuse(f"{S['pair']}: the re-scored setup is not the A1 record's: {off}")
    if adoc.get("input_cache_sha256") != S["input_sha"]:                  # the A1 record's own caches, every branch
        refuse(f"{S['pair']}: the re-scored caches are not the A1 record's (input_cache_sha256)")
    beside = {"registry_sources_as_the_record": jsonable(mine.get("input_registry_sources_sha256"))
              == adoc["config"].get("input_registry_sources_sha256")}
    if full:
        A1.check_predictions_against(S, pdoc, parr)
        return {"checked": True, "kind": "phase3_a1.check_predictions_against (every draw)", "n_draws": S["n_reps"],
                **beside}
    if S["n_reps"] > n_rec or pdoc.get("input_cache_sha256") != S["input_sha"] or pdoc["config"].get("variants") != S["variants"]:
        refuse(f"{S['pair']}: the predictions record names other inputs or variants, or fewer draws")
    arr, n = A1P.prediction_arrays(S), S["n_reps"] + 1
    bad = sorted(set(arr) ^ set(parr)) + [k for k in arr if k in parr and not (
        arr[k].dtype == parr[k].dtype and np.array_equal(arr[k], parr[k][:n]))]
    if bad:
        refuse(f"{S['pair']}: the predictions arrays' first {n} rows are not this setup's: {bad[:6]}")
    return {"checked": True, "kind": f"the predictions arrays' first {n} rows bit for bit (a test at fewer draws)",
            "n_draws": S["n_reps"], **beside}


def tail_arrays(S: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Per draw and seed: q99 and tau of the post scores at c = 0 of E (choice (d)); m and s formed as
    phase3_a1_predictions.build forms them and required equal to its m and s; the dtypes used (choice (j))."""
    nD, nS = len(S["draws"]), len(S["seeds"])
    q99, tau = np.empty((nD, nS)), np.empty((nD, nS))
    dt = set()
    for d, dr in enumerate(S["draws"]):
        post_E = S["SC"]["post"]["pool"][dr["E"]]
        m, s = post_E.mean(axis=0), post_E.std(axis=0)                    # as build(): ddof 0
        dt.add((str(post_E.dtype), str(m.dtype)))
        if not (np.array_equal(m, S["m"][d]) and np.array_equal(s, S["s"][d])):
            refuse(f"{S['pair']} draw {d}: m and s are not phase3_a1_predictions.build's")
        q99[d] = np.quantile(np.asarray(post_E, dtype=np.float64), Q_TAIL, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            tau[d] = (q99[d] - S["m"][d]) / S["s"][d] - Z99
    if len(dt) != 1:
        refuse(f"{S['pair']}: the post scores' dtype differs between draws: {sorted(dt)}")
    (sc_dt, ms_dt), = dt
    return q99, tau, {"post_scores": sc_dt, "m_and_s_computed_in": ms_dt, "m_and_s_held_as": str(S["m"].dtype),
                      "q99_computed_in": str(q99.dtype), "tau": str(tau.dtype)}


def gaussian_null(n: int = GAUSS_NULL["n"], samples: int = GAUSS_NULL["samples"],
                  seed: int = GAUSS_NULL["seed"]) -> dict[str, Any]:
    """Beside (choice (i)): the same estimator (numpy's linear q99, m and s with ddof 0) on standard-normal samples of
    n scores: the mean tau and its Monte Carlo standard error. Not used by the reading."""
    x = np.random.default_rng(seed).standard_normal((n, samples))
    t = (np.quantile(x, Q_TAIL, axis=0) - x.mean(axis=0)) / x.std(axis=0) - Z99
    return {**GAUSS_NULL, "mean_tau": float(t.mean()), "mc_standard_error": float(t.std(ddof=1) / np.sqrt(samples)),
            "role": "beside only: the estimator's offset under a Gaussian null (choice (i)); tau and its cut unchanged"}


def tail_pair(setup_fn: Callable[..., dict[str, Any]], core: str, pr: dict[str, Any], n_reps: int) -> dict[str, Any]:
    """One pair: the caches through setup_fn (phase3_a1_predictions.setup), the anchor, then tau."""
    t0 = time.time()
    cfg = pr["doc"]["config"]
    S = setup_fn(cfg["pair"], n_reps, list(cfg["seeds"]), cfg["target_fpr"])
    anchor = tail_anchor(S, pr["pred"]["doc"], pr["pred"]["z"], pr["doc"])
    q99, tau, dtypes = tail_arrays(S)
    sm = tau.mean(axis=1)
    out = {"tau": P.band_summary(sm[1:], sm[0]), "anchor": anchor, "dtypes": dtypes, "n_draws": S["n_reps"],
           "seeds": S["seeds"],
           "inputs": A1P.input_block(S), "depth_record": {"file": S["depth_rel"], "sha256": S["depth_sha"]},
           "input_verification": S["verification"],
           "input_cache_source": {k: {"file": r["file"], "expected_from": r["expected_from"], "pin_only": r["pin_only"]}
                                  for k, r in S["input_recs"].items()},
           "runtime_seconds": round(time.time() - t0, 1)}
    keep = {k: S[k] for k in ("pair", "input_sha", "input_recs", "cfg", "sel_path", "sel_sha", "cl_sha", "depth_rel",
                              "depth_sha", "reg_sources")}
    return {"block": out, "arrays": {f"{core}|q99": q99, f"{core}|tau": tau}, "inputs": keep}


def tail_inputs_changed(kept: dict[str, Any]) -> list[str]:
    """What of one pair's tail inputs changed during the run (phase3_a1_predictions.main's list)."""
    ch = [n for n, a, b in (("selection", sha256_file(kept["sel_path"]), kept["sel_sha"]),
                            ("cluster manifest", sha256_file(D.CLUSTERS), kept["cl_sha"]),
                            (kept["depth_rel"], sha256_file(ROOT / kept["depth_rel"]), kept["depth_sha"]),
                            ("registry sources", D.sources_now(), kept["reg_sources"])) if a != b]
    ch += [f"cache {r['file']}" for r in kept["input_recs"].values()
           if sha256_file(resolve_cache(r["file"], REG.CACHE)) != r["sha256"]]
    return [f"{kept['pair']}: {x}" for x in ch]


# ---- the reading -----------------------------------------------------------------------------------------------------------

def spearman(x: list[float], y: list[float]) -> float | None:
    """scipy.stats.spearmanr (average ranks); None when undefined (a constant vector or a non-finite tau)."""
    if not all(np.isfinite(v) for v in x) or any(np.isnan(v) for v in y):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = stats.spearmanr(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    rho = float(getattr(r, "statistic", r[0]))
    return rho if np.isfinite(rho) else None


def frac(num: int, den: int) -> Fraction | None:
    return Fraction(int(num), int(den)) if den > 0 else None


def reading(g_neg: dict[str, int], g_all: dict[str, int], d_neg: dict[str, int],
            taus: dict[str, float | None] | None, mlogs: dict[str, float],
            mlogs_floored: dict[str, float] | None = None) -> dict[str, Any]:
    """D-370 (4). g_neg, g_all, d_neg: integer count totals over the pairs and draws 1.. (Gaussian at c < 0, Gaussian
    at every c != 0, direct at c < 0), compared exactly as fractions (choice (h)); taus: the per-pair D-156 mean tau
    (None: the tail statistic not computed, a test); mlogs: the per-pair mean log(realised/predicted) over c < 0;
    mlogs_floored: the same with the floored prediction, beside only. Every undefined part carries undefined_reason
    (as phase3_a1_pooled.scored) and does not hold."""
    share = frac(g_neg["fp"], g_all["fp"])
    a = {"value": None if share is None else float(share), "gaussian_fp_total_c_negative": int(g_neg["fp"]),
         "gaussian_fp_total_c_nonzero": int(g_all["fp"]), "threshold": f">= {float(FA_SHARE_MIN)} (exactly 5 j >= 4 k)",
         "holds": bool(share is not None and share >= FA_SHARE_MIN),
         "undefined_reason": None if share is not None else "undefined: no Gaussian false alarm at any c != 0"}
    rg, rd = frac(g_neg["fp"], g_neg["fp"] + g_neg["tn"]), frac(d_neg["fp"], d_neg["fp"] + d_neg["tn"])
    b = {"direct_rate_c_negative": None if rd is None else float(rd),
         "gaussian_rate_c_negative": None if rg is None else float(rg),
         "direct_totals_c_negative": {q: int(d_neg[q]) for q in ("fp", "tn")},
         "gaussian_totals_c_negative": {q: int(g_neg[q]) for q in ("fp", "tn")},
         "rule": "direct < gaussian (strict), compared exactly as fractions",
         "holds": bool(rg is not None and rd is not None and rd < rg),
         "undefined_reason": None if rg is not None and rd is not None else "undefined: no negative-class cell at c < 0"}
    if taus is None:
        why = "not computed: a test run without the tail statistic (--skip-tail)"
        c = {"value": None, "holds": None, "undefined_reason": why}
        d = {"value": None, "holds": None, "undefined_reason": why}
    else:
        tv = {k: (float(v) if v is not None else float("nan")) for k, v in taus.items()}
        above = [k for k in PAIRS if np.isfinite(tv[k]) and tv[k] > 0]
        undef = [k for k in PAIRS if not np.isfinite(tv[k])]
        c = {"value": len(above), "pairs_with_tau_above_0": above, "pairs_with_undefined_tau": undef,
             "n_pairs": len(PAIRS), "threshold": f">= {TAU_PAIRS_MIN} of {len(PAIRS)}",
             "holds": bool(len(above) >= TAU_PAIRS_MIN), "undefined_reason": None}
        rho = spearman([tv[k] for k in PAIRS], [mlogs[k] for k in PAIRS])
        if rho is not None:
            reason = None
        elif undef:
            reason = f"undefined: no finite tau for {undef}"
        else:
            reason = "undefined: a constant vector (no rank variation in tau or in the mean log error)"
        d = {"value": rho, "n": len(PAIRS), "rule": "Spearman > 0 (strict); no significance claimed",
             "tau": {k: fin(tv[k]) for k in PAIRS}, "mean_log_realised_over_predicted_c_negative": {k: fin(mlogs[k]) for k in PAIRS},
             "holds": bool(rho is not None and rho > 0), "undefined_reason": reason}
        if mlogs_floored is not None:
            d["beside_mean_log_floored_prediction_c_negative"] = {k: fin(mlogs_floored[k]) for k in PAIRS}
    parts = {"a": a, "b": b, "c": c, "d": d}
    if any(p["holds"] is None for p in parts.values()):
        supported, verdict = None, "NOT COMPUTED (a test run without the tail statistic)"
    else:
        supported = all(p["holds"] for p in parts.values())
        verdict = "SUPPORTED" if supported else "NOT SUPPORTED"
    return {**parts, "supported": supported, "verdict": verdict,
            "rule": "SUPPORTED only if (a)-(d) all hold (D-370 (4)); no A1 verdict changes (D-353)"}


# ---- the run ---------------------------------------------------------------------------------------------------------------

def compute(rdir: Path, scorer_sha: dict[str, str], setup_fn: Callable[..., dict[str, Any]], n_reps: int | None,
            with_tail: bool, pin: bool, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Every refusal and anchor, then the measures and the reading."""
    B = bind(rdir, scorer_sha, pin)
    prs, pdoc, params = B["prs"], B["pooled"]["doc"], B["params"]
    cnt = counts_by_c(prs)
    anchors = {"P-A1.4": anchor_p14(cnt, pdoc)}
    log("  anchor P-A1.4: per-c counts summed over c = the nine records' arrays; pooled = the pooled record's blocks: PASS")
    rows = [r for core in PAIRS for r in PL.cell_rows(prs[core]) if r["variant"] == PRIMARY]
    anchors["P-A1.3"] = anchor_p13(rows, pdoc)
    log(f"  anchor P-A1.3: {len(rows)} cells = the pooled record's rows and verdict line: PASS")
    fa = false_alarms(cnt)
    le = log_errors(rows)
    means, means_fl = le.pop("_means_c_negative"), le.pop("_means_c_negative_floored")
    n_tail = int(params["n_reps"] if n_reps is None else n_reps)
    if with_tail and not 1 <= n_tail <= int(params["n_reps"]):
        refuse(f"--n-reps {n_tail}: the tail statistic takes 1 to {params['n_reps']} draws")
    tail, arrays, kept = None, {}, []
    if with_tail:
        tail = {}
        for core in PAIRS:
            t = tail_pair(setup_fn, core, prs[core], n_tail)
            tail[core], kept = t["block"], kept + [t["inputs"]]
            arrays.update(t["arrays"])
            log(f"  {core}: tail anchor PASS ({t['block']['anchor']['kind']}); {t['block']['runtime_seconds']:.0f} s")
    rd = reading(int_totals(cnt, PAIRS, NEG, "gaussian"), int_totals(cnt, PAIRS, NZ, "gaussian"),
                 int_totals(cnt, PAIRS, NEG, "direct"),
                 None if tail is None else {k: tail[k]["tau"]["mean"] for k in PAIRS}, means, means_fl)
    registered = bool(with_tail and n_tail == int(params["n_reps"])
                      and {k: params[k] for k in A1P.REGISTERED} == A1P.REGISTERED)
    records = {c: {"file": pr["file"], "sha256": pr["sha256"], "draws_npz": pr["draws_npz"],
                   "predictions_record": {"file": pr["pred"]["file"], "sha256": pr["pred"]["sha256"],
                                          "draws_npz": pr["pred"]["draws_npz"]},
                   "d170_flag": (pr["doc"].get("input_verification") or {}).get("flag")} for c, pr in prs.items()}
    body = {"parameters": params, "tail_n_draws": n_tail if with_tail else None, "registered_parameters": registered,
            "a1_code": B["code"], "pooled_record": {k: v for k, v in B["pooled"].items() if k != "doc"},
            "records": records,
            "d170": PL.d170_status(prs, PAIRS, {n: len(NZ) for n in PAIRS}, "cells per draw"),
            "anchors": anchors, "false_alarms": fa, "log_error": le,
            "tail_statistic": {"z": Z99, "per_pair": tail} if tail is not None else None,
            "tail_statistic_gaussian_null_beside": gaussian_null(),
            "reading": rd}
    cache_sha = {f"{k['pair'][len('repair_'):]}:{kk}": h for k in kept for kk, h in k["input_sha"].items()}
    ident = {}
    for k in kept:
        for kk in k["input_sha"]:
            side = kk.split("/")[0]
            src = "pre" if side == "surrogate" else side             # the surrogate is the pre model's other axis
            ident[f"{k['pair'][len('repair_'):]}:{kk}"] = {"repo": k["cfg"][f"{src}_repo"], "rev": k["cfg"][f"{src}_rev"]}
    return {"body": body, "files": B["files"], "arrays": arrays, "kept": kept, "input_cache_sha256": cache_sha,
            "input_identity": ident}


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Phase 3 A1: the by-c heavy-tail check (D-370 (3)-(4))")
    ap.add_argument("--records-dir", default=None, help="TESTING: a directory of A1 and pooled records (default results/)")
    ap.add_argument("--out", default=None, help="TESTING: a JSON path outside results/")
    ap.add_argument("--n-reps", type=int, default=None, help="TESTING: the tail statistic on the first n draws only")
    ap.add_argument("--skip-tail", action="store_true", help="TESTING: no cache read; parts (c) and (d) not computed")
    args = ap.parse_args()
    t0 = time.time()
    rdir = Path(args.records_dir).resolve() if args.records_dir else RESULTS.resolve()
    out = Path(args.out).resolve() if args.out else FINAL_OUT.resolve()
    npz_path = out.with_name(out.stem + "_draws.npz")
    final = out == FINAL_OUT.resolve()
    for p in (out, npz_path):
        if p.exists():
            refuse(f"{p} exists; results are never overwritten")
    if not final and out.is_relative_to(RESULTS.resolve()):
        refuse("a test output (--out) must not be under results/")
    threads = {v: os.environ.get(v) for v in A1.THREAD_VARS}
    if final and (rdir != RESULTS.resolve() or args.n_reps is not None or args.skip_tail
                  or any(t != "6" for t in threads.values())):
        refuse(f"the final name: records from results/ ({rdir == RESULTS.resolve()}), every draw (--n-reps "
               f"{args.n_reps}), the tail statistic (--skip-tail {args.skip_tail}), the 6-thread pin {threads}")
    code0 = P.code_block(CODE_FILES)
    res = compute(rdir, code0["sha256"], A1P.setup, args.n_reps, not args.skip_tail, pin=rdir == RESULTS.resolve(),
                  log=lambda m: print(m, flush=True))
    body = res["body"]
    if final and not body["registered_parameters"]:
        refuse("the final name needs the registered parameters (200 draws, seeds 0-4, 1% target)")
    code1 = P.code_block(CODE_FILES)
    changed = [n for n, (p, h) in res["files"].items() if sha256_file(p) != h]
    changed += [x for k in res["kept"] for x in tail_inputs_changed(k)]
    if code1["sha256"] != code0["sha256"] or changed:
        refuse(f"to write: the code or an input changed during the run: {changed}")
    nb = A1P.npz_bytes(res["arrays"])
    payload = {**provenance_block("phase3_a1_byc_d370"), "preregistration": PREREG,
               "preregistration_text": {"entry": "DECISIONS.md D-370 (3)-(4)", "text": PREREG_TEXT},
               "code": code0, "pinned_sha256": PINNED_SHA256, "blas_threads": threads,
               "config": CONFIG, "config_hash": canonical_hash(CONFIG),
               "input_files_sha256": {n: h for n, (_, h) in res["files"].items()},
               "input_cache_sha256": res["input_cache_sha256"], "input_identity": res["input_identity"],
               "draws_npz": {"file": A1P.rel(npz_path), "sha256": sha256_bytes(nb), "size": len(nb),
                             "keys": sorted(res["arrays"]),
                             "rows": "<pair>|q99 and <pair>|tau: draw index (0 the frozen split) x seeds in config order"},
               "choices": CHOICES, **body, "runtime_seconds": round(time.time() - t0, 1)}
    for p, h in A1P.write_pair(out, payload, npz_path, nb):
        print(f"Written: {p}\nsha256:  {h}")
    rd = body["reading"]                                              # printed only once the record is written
    for k in "abcd":
        print(f"  D-370 (4)({k}): value {rd[k].get('value')}; holds {rd[k]['holds']}")
    print(f"  READING: {rd['verdict']}\nruntime {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
