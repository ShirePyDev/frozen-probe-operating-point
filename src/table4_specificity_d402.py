"""Table IV's safety-run specificity drop, cell by cell (D-402 as amended by D-405; the author's D-401 (2d)).

Pre-registered in D-402 and amended by D-405 before any of this code's numbers existed (details by the closest
D-entries, D-181: D-255, D-241, D-363, D-380, D-156, D-164, D-177, D-322). QUESTION: Table IV's LoRA safety rows
lose Gaussian specificity on the D-363 reduced items (0.987 -> 0.784; direct 0.993 -> 0.884, D-380's unexplained
finding). Is that the DETECTOR changing its mind, or the class moving under it?

POPULATION (D-402 (1)): the six safety runs of 4c's "LoRA, safety" group (hh and pku at steps 25, 100, 400), every
layer of the sweep, selected C, item level, class two_sided_2x, in BOTH readings - the records of record
(results/healthcheck_d156_<run>.json with their _draws.npz, named by logs/healthcheck_d156_lora.sha256) and the
D-363 sensitivity (results/hh_sensitivity_d363_<run>.json with their _draws.npz, named by
logs/hh_sensitivity_d363.sha256). BESIDE: the three general (alpaca) runs the same way, and the D-164 cluster level.
The direct detector is reported beside the Gaussian one (D-241 (4)). NO cache is read, nothing is refitted and no
draw is made: the two record sets and their arrays are all this module reads.

UNIT (D-402 (2)): one (run, layer, draw) triple. Truth and flag are D-255's, recomputed here from the stored
per-draw arrays with healthcheck_summary_d156.crosses and its CLASSES, never redefined: truth is the draw's
realised_paired_ratio crossing the class cut (2x or 0.5x), the Gaussian flag is gaussian_ratio crossing the same
cut, the direct flag is direct_ratio's. Pooling is D-255's own estimator, healthcheck_summary_d156.pooled_counts
with pair_cluster_index, imported UNCHANGED (B = 5000, seed 520260925, each run its own cluster); the record binds
that file's sha256.

DECOMPOSITION (D-402 (3) as amended by D-405 (1)-(3), fixed there before any number of it existed): every
specificity triple of the record (truth negative) falls in exactly one bucket - (a) TRUTH MOVED, its sensitivity
truth is positive, so it leaves the specificity denominator; (b) DETECTOR MOVED, truth negative in both, flag
negative in the record and positive in the sensitivity: a new false positive; (c) BOTH; (d) UNCHANGED. Those four
cannot sum to the drop, because the specificity DENOMINATOR itself moves (D-405 (1)), so D-405 (2) adds three terms,
EACH ITS OWN NAMED TERM here and in the report, never folded into a bucket's share:
  (e) entering               a triple whose truth moved IN (positive in the record, negative in the sensitivity), so
                             the sensitivity's denominator gains it (entering_tn and entering_fp);
  (f) a_truth_moved_record_fp a record FALSE positive whose truth moved out of the denominator (D-402 (3)'s bucket
                             (a) says "a true negative", so this is not bucket (a));
  (g) d_unchanged_cleared    a record false positive whose flag cleared (negative in the sensitivity).
D-405 (3) fixes the ORDER, and STEPS is exactly it: (a) truth moved, (b) detector moved, (c) both, then (f), (g) and
(e). The running sum must land EXACTLY on the sensitivity's own pooled counts or the module refuses, so the drop is
written without a residual. Each term's own contribution is reported applied ALONE as well, and P-T4.1 is scored on
the alone share with the sequential share beside (D-405 (3), CHOICE 4). FLAGGED (CLAUDE.md rule 5): the three added
terms are the author's to read; none of them is a bucket of D-402 (3).

PER-CELL TABLE (D-402 (4)): the (run, layer) cells ranked by their contribution to bucket (b), each with the reduced
sizes after R (probe_train, threshold_calib, evaluation negatives, from the sensitivity record's population block),
the record's and the sensitivity's per-draw mean and SD of realised_paired_ratio, gaussian_ratio, z_mean and
log_sd_ratio, and the record's distance from the class boundary on the log scale, |log(realised ratio)| - log 2.

PREDICTIONS (D-402 (5)): written by their own job to results/table4_specificity_d402_predictions.json BEFORE any
number of the diagnostic exists (`predictions`), then bound by sha256 and scored HELD or FAILED by `run`, on the
registered primary cell only (safety, selected C, item level, Gaussian, two_sided_2x); every other block is beside
and never a verdict.
  P-T4.1  bucket (b) explains at least half the drop (the detector's own change, not reclassification).
  P-T4.2  bucket (b) cells sit closer to the class boundary than the unchanged negatives.
  P-T4.3  the per-draw SD of the Gaussian ratio rises from record to sensitivity in at least 80% of bucket (b) cells.
  P-T4.4  the general runs' pooled specificity change is smaller in absolute value than the safety runs'.

ACCURACY LINE (D-402 (6), D-177 (3), D-322 (4)): each run's sensitivity specificity reconstructed from its record by
applying bucket (b) ALONE; that predicted-against-realised table carries the share within a factor of 1.5 and the
median |log(realised/predicted)|, with n. The (run, layer) cell is the beside unit. A unit whose side is MISSING (an
empty specificity denominator) is not a unit and stays out; a unit whose two sides are defined and one of them is
zero is a MISS - an infinite error, outside x1.5, inside the median - as src/phase3_a1.py's `accuracy` and
src/mlp_probe_d383.py's `accuracy_line` score a zero side. The missing-side rule is THIS module's own choice: neither
of those two has an empty-denominator case (mlp_probe_d383 scores a None side as a miss, phase3_a1 has no missing-side
path at all), and an undefined specificity is not a prediction. Both conventions are reported (D-322 (4), CHOICE 17).

ANCHOR, before anything is written: the per-draw confusion counts recomputed here, summed over a run's cells, must
equal that run's own npz counts arrays in BOTH readings (every level, both detectors, the class), and the pooled
record and sensitivity specificity must equal results/healthcheck_d156_summary_lora.json's lora_safety /
lora_general row and results/hh_sensitivity_d363.json's table_iv_lora_rows_all rows, whose two safety readings must
round to D-402's quoted pair. Otherwise the run refuses and writes nothing.

POPULATION GROUP (CHOICE 12, D-380's after-pull steps): the group TABLE4_SPECIFICITY_D402 - the record and its
_draws.npz - is declared in src/populations.py AFTER the pull, outside primary_result_stems, exactly as D-380's
after-pull steps declared HH_SENSITIVITY_D363; it is not declared before the run, because nothing of it exists yet.
The record names the group and where it will be declared (POPULATION below), and the lane list's header repeats the
step.

REUSED BYTES (D-402 (7)): every subcommand refuses unless REUSED_SHA256 is on disk at the reviewed bytes - the
final name, a sandbox name and the lane list alike. There is no loose path: a reused function at other bytes is a
different estimator, and nothing this module writes would mean what it says (CHOICE 18).

OUTPUTS (written once: staged, hard-linked to the final name, read back, then mode 444; never overwritten)
  predictions   results/table4_specificity_d402_predictions.json   the registered statements and their rules, no number
  run           results/table4_specificity_d402.json and its _draws.npz (D-163 (6)): the per-cell per-draw truth,
                flag, bucket code and boundary distance arrays, from which every aggregate is recomputable
  queue --write queues/laneA_<date>_t4diag.tsv                     the lane A list (D-402 (8)), labels checked unique
A test writes outside results/, logs/ and queues/.
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
import healthcheck_summary_d156 as HS  # noqa: E402   D-255's estimator and class/flag rule, reused UNCHANGED
import paired as P  # noqa: E402
import populations as POP  # noqa: E402
import safe_outputs as SO  # noqa: E402
from provenance import (  # noqa: E402
    canonical_hash, ensure_utf8_stdout, input_identity_problems, provenance_block, sha256_bytes, sha256_file,
    utc_now, write_results,
)

ROOT = Path(__file__).resolve().parent.parent
CODE_ROOT = Path(__file__).resolve().parent.parent      # where the source files are hashed (ROOT may be a test tree)
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
QUEUES = ROOT / "queues"
OWN = "src/table4_specificity_d402.py"
RUN_KIND = "table4_specificity_d402"
STEM = "table4_specificity_d402"
CODE_FILES = list(dict.fromkeys([OWN, *HS.CODE_FILES, "src/safe_outputs.py"]))
PREREG = ("D-402 as amended by D-405 (the author's D-401 (2d), 2026-10-08; both written before any code or number "
          "of them existed); details by the closest D-entries (D-181): D-255, D-241, D-363, D-380, D-156, D-164, "
          "D-177, D-322")

# the reviewed files whose functions and per-draw definitions this module reuses, at the bytes it was written and
# anchored against; a final name refuses other bytes (D-363 (4), D-402 (7): reused WITHOUT being changed)
REUSED_SHA256 = {
    "src/healthcheck_summary_d156.py": "b149b66181fe1c2d953d255355e44e2d15651bff9ccf8a456819092db526ccf5",
    "src/healthcheck_d156.py": "0ec2687ecdd887a28d4b8fb7a4db4b9081b69ea3fcf2b175c345279b4ed74de7",
}
# the two summaries bound by a pinned sha256 because no logs/*.sha256 manifest names them (healthcheck_baselines_
# d381's pin for the 4c LoRA summary; D-380's ledger line for the D-363 summary), and the two INPUT MANIFESTS pinned
# too: a manifest names its records, so a swapped manifest with swapped records would otherwise pass every check
# (CLAUDE.md rule 7, D-152 P5). Both values are the sha256 of the files committed at 436754f (D-380's pull).
PINNED = {"SUMMARY_4C_LORA": "f697569d57c82d07678d8c8e79c0a953f9b6b7ffff8e4d6dffb802f7b16fe500",
          "SUMMARY_D363": "030f6c4ccc639231c0f7d083b43319077d59a0a024cbb4ace763679ed56ccdc0",
          "MANIFEST_RECORD": "f91f72ecbc60f730a8c9891f6b3303108439b4c825fadc1a9becc3c9c71c5f0d",
          "MANIFEST_SENSITIVITY": "f4b5de20cc5d5c2c326da100a4050cff8d2bbfbf16c4631e39078d6cdedc639d"}
SUMMARY_4C_LORA = "results/healthcheck_d156_summary_lora.json"
SUMMARY_D363 = "results/hh_sensitivity_d363.json"
MANIFEST_RECORD = "logs/healthcheck_d156_lora.sha256"
MANIFEST_SENSITIVITY = "logs/hh_sensitivity_d363.sha256"

PROBE_SET = "selected"                                  # D-402 (1): selected C only
CLASS = "two_sided_2x"                                  # D-255's primary cell
LEVELS = HS.LEVELS                                      # item level primary, the D-164 cluster level beside
DETECTORS = ("gaussian", "direct")                      # D-241 (4): Gaussian primary, direct beside
DETECTOR_STAT = {"gaussian": "gaussian_ratio", "direct": "direct_ratio"}
CELL_STATS = ("realised_paired_ratio", "gaussian_ratio", "z_mean", "log_sd_ratio")      # D-402 (4)
GROUPS = {"safety": [r for arm in ("safety (hh-harmless)", "safety (pku-saferlhf)") for r in POP.LORA_ARMS[arm]],
          "general": list(POP.LORA_ARMS["general (alpaca)"])}
GROUP_ROW = {"safety": "lora_safety", "general": "lora_general"}        # the rows of the two summaries of record
PRIMARY = {"group": "safety", "probe_set": PROBE_SET, "level": LEVELS[0], "class": CLASS, "detector": "gaussian"}
# D-380 / D-402's quoted safety-run readings, to three decimals; the run refuses if the record's own rows do not
# round to them (so a wrong row or a wrong class can never be scored)
QUOTED = {"gaussian": (0.987, 0.784), "direct": (0.993, 0.884)}

BUCKETS = ("a_truth_moved", "b_detector_moved", "c_both", "d_unchanged")
# the exact terms: the four buckets' sub-counts and the two the sensitivity's denominator gains
TERMS = ("a_truth_moved_record_tn", "a_truth_moved_record_fp", "b_detector_moved", "c_both",
         "d_unchanged_tn", "d_unchanged_fp", "d_unchanged_cleared", "entering_tn", "entering_fp")
UNNAMED_BY_D402_3 = ("a_truth_moved_record_fp", "d_unchanged_cleared", "entering_tn", "entering_fp")
# D-405 (3)'s fixed order, written out: (a) truth moved, (b) detector moved, (c) both, then (f), (g) and (e). Each
# step is ONE named term - (f), (g) and (e) are never folded into a bucket's step (D-405 (2)) - and the increments
# sum to the drop exactly, landing on the sensitivity's own counts. Step (a) is bucket (a) AS D-402 (3) WORDS IT,
# "a true negative of the record whose sensitivity truth is positive"; its false-positive counterpart is (f).
STEPS = ("a_truth_moved_record_tn", "b_detector_moved", "c_both", "f_truth_moved_record_fp",
         "g_unchanged_cleared", "e_entering")
STEP_D405 = {"a_truth_moved_record_tn": "(a) truth moved (the record's true negative leaves the denominator)",
             "b_detector_moved": "(b) detector moved (a new false positive)",
             "c_both": "(c) both",
             "f_truth_moved_record_fp": "(f) a record FALSE positive whose truth moved out of the denominator",
             "g_unchanged_cleared": "(g) a record false positive whose flag cleared",
             "e_entering": "(e) entering: the sensitivity's denominator gains the triple"}
STEP_TERMS = {"a_truth_moved_record_tn": ("a_truth_moved_record_tn",), "b_detector_moved": ("b_detector_moved",),
              "c_both": ("c_both",), "f_truth_moved_record_fp": ("a_truth_moved_record_fp",),
              "g_unchanged_cleared": ("d_unchanged_cleared",), "e_entering": ("entering_tn", "entering_fp")}
ALONE = STEPS                                           # the same terms, each applied to the record on its own
BUCKET_CODE = {"not_a_record_specificity_triple": 0, "a_truth_moved_record_tn": 1, "a_truth_moved_record_fp": 2,
               "b_detector_moved": 3, "c_both": 4, "d_unchanged_tn": 5, "d_unchanged_fp": 6,
               "d_unchanged_cleared": 7, "entering_tn": 8, "entering_fp": 9}

N_DRAWS = 1000                                          # plus the frozen split at index 0 (D-156)
B, SEED, MIN_PAIRS = HS.B, HS.SEED, HS.MIN_PAIRS        # D-255's pair-cluster bootstrap, unchanged
COUNTS = HS.COUNTS
LOG15 = float(np.log(1.5))                              # D-177 (3): the factor-1.5 band
LOG2 = float(np.log(2.0))                               # the class cut on the log scale
B_CELL_CUT = 0.0                                        # CHOICE 6: a bucket (b) cell has mean count above this
B_CELL_CUT_BESIDE = 0.5
PRED_IDS = ("P-T4.1", "P-T4.2", "P-T4.3", "P-T4.4")
PRED_HALF, PRED_SHARE = 0.5, 0.8                        # D-402 (5): "at least half", "at least 80%"
QUEUE_LANE, QUEUE_TAG = "A", "t4diag"

POPULATION = {"declared_in": "src/populations.py", "group": "TABLE4_SPECIFICITY_D402",
              "declared_when": "after the pull, as D-380's after-pull steps declared HH_SENSITIVITY_D363 (CHOICE 12)",
              "role": "a cell-level diagnostic of Table IV's safety-run specificity drop on the D-363 reduced items; "
                      "beside the two readings of record, never primary and in no table (D-402 (9), D-401 (d))"}

CHOICES = [
    {"n": 1, "choice": "the pooled decomposition runs over draws 1..1000; index 0, the frozen split, is reported "
                       "beside and never enters the mean counts",
     "settled_by": "D-156 (the mean over the 1000 draws, the frozen split beside as one draw) as "
                   "src/healthcheck_summary_d156.py's pooled_counts implements it (a[1:].mean()); D-255 is the "
                   "summary of record that uses it"},
    {"n": 2, "choice": "the sequential decomposition applies ONE named term per step, in D-405 (3)'s fixed order: "
                       "(a) truth moved (the record's true negative), (b) detector moved, (c) both, then (f) a "
                       "record false positive whose truth moved out, (g) a record false positive whose flag cleared "
                       "and (e) entering - STEPS is that order written out, and no added term is folded into a "
                       "bucket's step",
     "settled_by": "D-405 (3) fixes the order and the exact landing on the sensitivity's own counts; D-405 (2) "
                   "makes each added term its own term (it supersedes D-402 (3)'s four-bucket order). D-181"},
    {"n": 3, "choice": "a term's own contribution ('alone') is the pooled specificity after applying that term's "
                       "moves to the record alone, minus the record's",
     "settled_by": "D-402 (6) registers exactly that counterfactual for bucket (b)"},
    {"n": 4, "choice": "P-T4.1's share is the bucket-(b)-alone contribution divided by the pooled drop; the "
                       "sequential share is reported beside",
     "settled_by": "D-405 (3) ('P-T4.1 is scored on the alone share with the sequential share beside'), D-402 (6) "
                   "(the registered reconstruction is bucket (b) applied alone), D-402 (5)"},
    {"n": 5, "choice": "the three terms added by D-405 (2) - (f) a record false positive whose truth moved out, "
                       "(g) a record false positive whose flag cleared, and (e) the triples entering the "
                       "denominator - are each their own named term and step, never folded into a bucket's share",
     "settled_by": "D-405 (2) ('each its own named term in the record and in the report, never folded into a "
                   "bucket's share'), D-405 (1) (the four buckets cannot sum to the drop); CLAUDE.md rule 5, which "
                   "raised the gap, keeps it FLAGGED for the author"},
    {"n": 6, "choice": f"a (run, layer) cell is a bucket (b) cell when its mean-over-draws bucket (b) count is above "
                       f"{B_CELL_CUT:g} (a beside row uses {B_CELL_CUT_BESIDE:g}, the bucket in at least half the "
                       f"draws); P-T4.2's 'unchanged negatives' are the cells whose record specificity triples are "
                       f"ALL in bucket (d) (bucket (d) is the one D-402 (3) names UNCHANGED), with the looser set "
                       f"(no bucket (b) triple) beside. THE VERDICT LINE of both P-T4.2 and P-T4.3 is the strict "
                       f"cut, mean bucket (b) count above {B_CELL_CUT:g}; the {B_CELL_CUT_BESIDE:g} cut is beside "
                       f"and carries no verdict",
     "settled_by": "D-402 (4) makes the (run, layer) cell the unit of the table; D-402 (3) names bucket (d) "
                   "UNCHANGED; D-402 (5) words P-T4.2 and P-T4.3 over 'bucket (b) cells' without a count cut, so "
                   "the cut is this module's choice; D-181"},
    {"n": 7, "choice": "a cell's distance from the class boundary is the mean over the draws of "
                       "|log(realised paired ratio)| - log 2 (its absolute value for P-T4.2)",
     "settled_by": "D-156: every operating-point quantity is reported as the mean over the draws"},
    {"n": 8, "choice": "the accuracy line's unit is the RUN (six rows); the (run, layer) cell is reported beside",
     "settled_by": "D-402 (6) ('each run's sensitivity specificity'), D-177 (3)"},
    {"n": 9, "choice": "the per-draw mean and SD of D-402 (4) are paired.band_summary's 'mean' and 'sd_over_draws'",
     "settled_by": "D-402 (4) asks for the per-draw mean and SD; D-156 reports every per-draw quantity as the mean "
                   "over the 1000 draws with its band through src/paired.py's band_summary, which the records of "
                   "record themselves use, so its 'sd_over_draws' is taken from there unchanged, not recomputed"},
    {"n": 10, "choice": "the predictions are written by their own job, before the diagnostic job exists; the run "
                        "binds that file's sha256 and refuses statements that are not this module's",
     "settled_by": "D-402 (5): on disk before any number of the diagnostic is read"},
    {"n": 11, "choice": "the final name does not require the 6-thread BLAS pin (the threads are recorded): this is a "
                        "records-only summary that fits nothing, as healthcheck_summary_d156 is",
     "settled_by": "the CODE of D-255's summary of record, src/healthcheck_summary_d156.py, which gates on no "
                   "thread pin (D-255's text says nothing about threads; the code is the precedent followed here), "
                   "and D-402 (8), which puts this on lane A, 'records and arrays only'; the threads are recorded "
                   "either way, so D-152 (P6)'s pin stays checkable after the fact"},
    {"n": 12, "choice": "the population group TABLE4_SPECIFICITY_D402 is named in the record and declared in "
                        "src/populations.py after the pull, not before the run",
     "settled_by": "D-380's AFTER-PULL STEPS, which declared the group HH_SENSITIVITY_D363 in src/populations.py "
                   "and checked it with validate() after that pull: the same step, repeated in this lane list's "
                   "header"},
    {"n": 13, "choice": "the per-cell arrays go to results/table4_specificity_d402_draws.npz",
     "settled_by": "D-163 (6): per-draw arrays are evidence, tracked under results/*_draws.npz and hashed in the "
                   "record and in the dated draw-array manifest"},
    {"n": 14, "choice": "the anchor is byte-level on the records' own npz counts arrays and value-level on the two "
                        "summaries of record; nothing is written if any comparison fails",
     "settled_by": "D-363 (4) ('an anchor first'), D-255 (the records of record), D-402 (2) (truth and flag exactly "
                   "as D-255 computes them)"},
    {"n": 15, "choice": "verdicts are produced for the registered primary cell only (safety, selected C, item level, "
                        "Gaussian, two_sided_2x); the general group, the cluster level and the direct detector are "
                        "reported beside and carry no verdict",
     "settled_by": "D-402 (1) (the primary cell and what is 'beside'), D-402 (5) (the four statements are scored on "
                   "that cell), D-241 (4) (the direct detector is reported beside the Gaussian one)"},
    {"n": 16, "choice": "a prediction whose rule cannot be evaluated scores FAILED with the reason named",
     "settled_by": "D-402 (5): the only two verdicts are HELD and FAILED"},
    {"n": 17, "choice": "in the accuracy line a MISSING side (None or non-finite - an empty specificity "
                        "denominator) makes the row not a unit and it stays out of n and of the share, while a "
                        "DEFINED zero side is a MISS: an infinite error, outside x1.5, inside the median. Both "
                        "conventions are reported - the line over every unit and, beside it, the line over the "
                        "units with a finite error",
     "settled_by": "D-177 (3) (the standing accuracy line), D-322 (4) (a zero or sign-flipped side counted as a "
                   "miss; 'both conventions are reported from now on'), as src/phase3_a1.py's `accuracy` and "
                   "src/mlp_probe_d383.py's `accuracy_line` already implement it"},
    {"n": 18, "choice": "the REUSED_SHA256 refusal applies on every path - predictions, run and the lane list, "
                        "the final name and a sandbox name alike; there is no loose path",
     "settled_by": "D-402 (7) (the reviewed functions are reused WITHOUT being changed), D-363 (4) (their sha256 "
                   "bound by the record): at other bytes the estimator is a different one, so nothing this module "
                   "writes would mean what it says"},
    {"n": 19, "choice": "the intermediate pooled calls of the sequential and alone contributions zero tp and fn, so "
                        "only their SPECIFICITY (and its interval) is defined; no sensitivity is computed or "
                        "reported from them, and each recorded step says so",
     "settled_by": "D-402 (3), D-405 (1): only the specificity's numerator and denominator move in the "
                   "decomposition; D-241 (6)'s pooled block is reused unchanged, which is why the field exists at "
                   "all"},
]


# ------------------------------------------------------------------------------------------------------- inputs

def rel_of(path: Path) -> str:
    return str(Path(path).relative_to(ROOT)).replace("\\", "/")


def parse_manifest(rel: str, pin_key: str, inputs: dict[str, str]) -> dict[str, str]:
    """A logs/*.sha256 manifest as {repo-relative file: sha256}; its own bytes are PINNED and listed as an input.

    The pin matters: a manifest names its records, so a swapped manifest with swapped records would otherwise
    satisfy every other check in this module (CLAUDE.md rule 7)."""
    p = ROOT / rel
    if not p.is_file():
        raise SystemExit(f"REFUSING: no manifest {rel} (the records are named by it, D-255, D-380)")
    b = p.read_bytes()
    have = sha256_bytes(b)
    if have != PINNED[pin_key]:
        raise SystemExit(f"REFUSING: {rel} is not the manifest this module pins ({have[:12]} vs "
                         f"{PINNED[pin_key][:12]}); a swapped manifest would name swapped records")
    inputs[rel] = have
    man: dict[str, str] = {}
    for line in b.decode("utf-8").splitlines():
        if line.strip():
            h, f = line.split(maxsplit=1)
            man[f.strip().lstrip("*")] = h
    return man


def load_json_bound(rel: str, want: str, inputs: dict[str, str], what: str) -> dict[str, Any]:
    """Read a results JSON whose sha256 must be `want`, and list it with that sha256."""
    p = ROOT / rel
    if not p.is_file():
        raise SystemExit(f"REFUSING: {rel} is missing ({what})")
    b = p.read_bytes()
    have = sha256_bytes(b)
    if have != want:
        raise SystemExit(f"REFUSING: {rel} is not the file {what} names ({have[:12]} vs {want[:12]})")
    inputs[rel] = have
    return json.loads(b.decode("utf-8"))            # listed (D-297)


def load_sensitivity(run: str, man: dict[str, str], inputs: dict[str, str]) -> dict[str, Any]:
    """One D-363 per-run record and its npz, both named by logs/hh_sensitivity_d363.sha256 and by the record itself.

    The same shape as healthcheck_summary_d156.load's return, for the sensitivity reading (whose file name that
    reviewed function does not build; it is imported and used unchanged for the record reading)."""
    rel = f"results/hh_sensitivity_d363_{run}.json"
    p = ROOT / rel
    if not p.is_file():
        raise SystemExit(f"REFUSING: {rel} is missing (the D-363 reading of record, D-380)")
    b = p.read_bytes()
    if man.get(rel) != sha256_bytes(b):
        raise SystemExit(f"REFUSING: {rel} is not the file {MANIFEST_SENSITIVITY} names")
    d = json.loads(b.decode("utf-8"))               # listed (D-297)
    if d.get("registered_parameters") is not True:
        raise SystemExit(f"REFUSING: {rel} is not a registered run")
    nrel = d["draws_npz"]["file"]
    nb = (ROOT / nrel).read_bytes()
    if man.get(nrel) != sha256_bytes(nb) or sha256_bytes(nb) != d["draws_npz"]["sha256"]:
        raise SystemExit(f"REFUSING: {nrel} is not the npz {rel} and {MANIFEST_SENSITIVITY} name")
    inputs[rel], inputs[nrel] = sha256_bytes(b), sha256_bytes(nb)
    z = np.load(ROOT / nrel, allow_pickle=False)
    return {"doc": d, "z": {k: z[k] for k in z.files}, "flag": d["input_verification"]["flag"]}


def array_of(side: dict[str, Any], run: str, reading: str, key: str, want: int | None = None) -> np.ndarray:
    """One per-draw array, or a refusal naming the missing key (a missing array is never a zero).

    `want` is the draw count the cell was established at (n_draws plus the frozen split). EVERY array read with a
    `want` must carry exactly that many draws: a shorter one would broadcast silently against the others (review 2,
    should-fix a). The two realised_paired_ratio arrays that ESTABLISH the count are read with want=None and
    checked against each other and against n_draws by cells_of_run, which names the two sides in its refusal."""
    z = side["z"]
    if key not in z:
        raise SystemExit(f"REFUSING: the {reading} npz of {run} has no array {key!r}")
    a = np.asarray(z[key])
    if a.ndim != 1:
        raise SystemExit(f"REFUSING: the {reading} array {key!r} of {run} is not one-dimensional ({a.shape})")
    if want is not None and a.size != want:
        raise SystemExit(f"REFUSING: the {reading} array {key!r} of {run} carries {a.size} draws, not {want}")
    return a


def layers_of(side: dict[str, Any], level: str) -> list[int]:
    return sorted({int(k[1:].split("|")[0]) for k in side["z"]
                   if k.startswith("L") and k.endswith(f"|{PROBE_SET}|{level}|realised_paired_ratio")})


# ------------------------------------------------------------------------------------------------ the estimator

def pooled(per_run: dict[str, dict[str, np.ndarray]], runs: list[str], idx: np.ndarray | None) -> dict[str, Any]:
    """D-255's estimator, healthcheck_summary_d156.pooled_counts, on per-run per-draw count arrays (unchanged)."""
    return HS.pooled_counts({q: [per_run[r][q] for r in runs] for q in COUNTS}, idx)


def spec_of(block: dict[str, Any]) -> float | None:
    return block["pooled_from_mean_counts"]["specificity"]


def zeros_like_counts(n: int) -> dict[str, np.ndarray]:
    return {q: np.zeros(n, dtype=np.int64) for q in COUNTS}


def accuracy_line(realised: list[float | None], predicted: list[float | None], unit: str) -> dict[str, Any]:
    """D-177 (3)'s accuracy line: the share of units predicted within a factor of 1.5 and the median
    |log(realised/predicted)|, with n.

    CHOICE 17: the ZERO-side rule follows src/phase3_a1.py's `accuracy` and src/mlp_probe_d383.py's
    `accuracy_line` (D-177 (3), D-322 (4)). The MISSING-side rule is this module's own, cited to neither: a side that
    is MISSING - None or non-finite, i.e. an empty specificity denominator - makes the row NOT A UNIT, and it stays
    out of n and out of the share (an undefined specificity is not a prediction; mlp_probe_d383 scores a None side as
    a miss, phase3_a1 has no such path, so this is a choice for the author, not a precedent); a unit whose two sides are both DEFINED but one of
    them is zero is a MISS, an infinite error that is outside x1.5 and inside the median (so the median is None
    with median_is_a_miss when it falls on one). D-322 (4) asks for both conventions, so the line over the units
    with a finite error is reported beside."""
    if len(realised) != len(predicted):
        raise SystemExit(f"REFUSING: the accuracy line of {unit} has {len(realised)} realised values and "
                         f"{len(predicted)} predicted ones")
    def defined(x: Any) -> float | None:
        return None if x is None or not np.isfinite(float(x)) else float(x)
    units = [(a, b) for a, b in ((defined(x), defined(y)) for x, y in zip(realised, predicted))
             if a is not None and b is not None]
    e = np.array([abs(np.log(a / b)) if a > 0 and b > 0 else np.inf for a, b in units], dtype=float)
    med = float(np.median(e)) if e.size else None
    fin = e[np.isfinite(e)]
    return {"unit": unit, "n": int(e.size),
            "n_misses_by_a_zero_side": int((~np.isfinite(e)).sum()),
            "n_not_a_unit_missing_side": len(realised) - len(units),
            "share_within_1.5x": float((e <= LOG15).mean()) if e.size else None,
            "median_abs_log_error": med if med is not None and np.isfinite(med) else None,
            "median_is_a_miss": bool(med is not None and not np.isfinite(med)),
            "beside_finite_errors_only": {
                "n": int(fin.size),
                "share_within_1.5x": float((fin <= LOG15).mean()) if fin.size else None,
                "median_abs_log_error": float(np.median(fin)) if fin.size else None},
            "rule": "D-177 (3), D-322 (4), CHOICE 17: a missing side (an empty denominator) is not a unit; a "
                    "DEFINED zero side is a MISS - an infinite error, outside x1.5, inside the median. Both "
                    "conventions are reported (beside_finite_errors_only leaves the misses out)"}


def median_or_none(v: list[float]) -> float | None:
    a = np.asarray([x for x in v if x is not None], dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else None


def bucket_b_cells(rows: list[dict[str, Any]], cut: float) -> list[dict[str, Any]]:
    """CHOICE 6: the cells whose mean-over-draws bucket (b) count is above `cut`."""
    return [r for r in rows if r["mean_count_bucket_b"] > cut]


def unchanged_cells(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """CHOICE 6: the UNCHANGED cells of P-T4.2, those whose record specificity triples are all in bucket (d)."""
    return [r for r in rows if r["record_specificity_denominator_mean"] > 0
            and all(r["triple_counts_mean_over_draws"][q] == 0 for q in
                    ("a_truth_moved_record_tn", "a_truth_moved_record_fp", "b_detector_moved", "c_both"))]


# ----------------------------------------------------------------------------------- the per-cell decomposition

def cell_terms(truth_rec: np.ndarray, flag_rec: np.ndarray, truth_sen: np.ndarray,
               flag_sen: np.ndarray) -> dict[str, np.ndarray]:
    """D-402 (3)'s buckets for one cell, as boolean per-draw arrays, plus the terms its wording does not name.

    Every draw of the cell is one triple. The record's specificity triples (truth negative) are partitioned by
    (truth moved out?, flag negative in the record and positive in the sensitivity?) into (a), (b), (c), (d); (a)
    and (d) are split by the record's flag, and the triples whose truth moved IN are the entering term."""
    den_rec = ~truth_rec
    moved = den_rec & truth_sen                                     # leaves the specificity denominator
    new_fp = den_rec & ~flag_rec & flag_sen                         # the detector's own change, upwards
    a = moved & ~new_fp
    b = den_rec & ~truth_sen & new_fp
    c = moved & new_fp
    d = den_rec & ~truth_sen & ~new_fp
    enter = truth_rec & ~truth_sen
    t = {"a_truth_moved_record_tn": a & ~flag_rec, "a_truth_moved_record_fp": a & flag_rec,
         "b_detector_moved": b, "c_both": c,
         "d_unchanged_tn": d & ~flag_rec & ~flag_sen, "d_unchanged_fp": d & flag_rec & flag_sen,
         "d_unchanged_cleared": d & flag_rec & ~flag_sen,
         "entering_tn": enter & ~flag_sen, "entering_fp": enter & flag_sen}
    # the partition is exact, draw by draw: every record specificity triple in one bucket, nothing counted twice
    stack = np.vstack([t[q] for q in TERMS]).sum(axis=0)
    pairs = ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d))
    if not np.array_equal(a | b | c | d, den_rec) or any((x & y).any() for x, y in pairs):
        raise SystemExit("REFUSING: the bucket partition of a cell is not a partition of its record negatives")
    if not np.array_equal(stack, (den_rec | enter).astype(np.int64)):
        raise SystemExit("REFUSING: the terms of a cell do not cover its triples exactly once")
    t["buckets_a"] = a
    t["buckets_b"] = b
    t["buckets_c"] = c
    t["buckets_d"] = d
    return t


def bucket_code(t: dict[str, np.ndarray]) -> np.ndarray:
    """One int8 per draw: which term this cell's triple is (BUCKET_CODE), 0 when it is not in either denominator."""
    out = np.zeros(t["b_detector_moved"].shape, dtype=np.int8)
    for name in TERMS:
        out[t[name]] = BUCKET_CODE[name]
    return out


def counts_from_flags(truth: np.ndarray, flag: np.ndarray) -> dict[str, np.ndarray]:
    return {"tp": (truth & flag).astype(np.int64), "fn": (truth & ~flag).astype(np.int64),
            "fp": (~truth & flag).astype(np.int64), "tn": (~truth & ~flag).astype(np.int64)}


def apply_step(tn: np.ndarray, fp: np.ndarray, step: str, m: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """ONE named term's move on a (tn, fp) pair of per-draw counts (D-405 (2), (3); CHOICE 2's order is STEPS).

    Each branch applies exactly one term of TERMS (STEP_TERMS says which), so no added term of D-405 (2) is folded
    into a bucket's step: (f) is its own step, not part of (a)."""
    if step == "a_truth_moved_record_tn":       # (a) a record TRUE negative leaves the specificity denominator
        return tn - m["a_truth_moved_record_tn"], fp
    if step == "b_detector_moved":              # (b) a new false positive: the numerator loses it, the denominator keeps it
        return tn - m["b_detector_moved"], fp + m["b_detector_moved"]
    if step == "c_both":                        # (c) a record true negative that leaves the denominator
        return tn - m["c_both"], fp
    if step == "f_truth_moved_record_fp":       # (f) a record FALSE positive leaves the denominator (D-405 (2))
        return tn, fp - m["a_truth_moved_record_fp"]
    if step == "g_unchanged_cleared":           # (g) a record false positive the sensitivity does not flag
        return tn + m["d_unchanged_cleared"], fp - m["d_unchanged_cleared"]
    if step == "e_entering":                    # (e) a triple whose truth moved IN: the denominator gains it
        return tn + m["entering_tn"], fp + m["entering_fp"]
    raise SystemExit(f"REFUSING: unknown decomposition step {step!r}")


# --------------------------------------------------------------------------------------------- the per-run pass

def read_sides(runs: list[str], inputs: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Both readings of every run, each bound to its manifest and its own record, and the manifests listed."""
    man_r = parse_manifest(MANIFEST_RECORD, "MANIFEST_RECORD", inputs)
    man_s = parse_manifest(MANIFEST_SENSITIVITY, "MANIFEST_SENSITIVITY", inputs)
    rec = {r: HS.load(r, man_r, inputs, "", Path(MANIFEST_RECORD).name) for r in runs}
    sen = {r: load_sensitivity(r, man_s, inputs) for r in runs}
    bad = [r for r in runs if rec[r]["doc"]["config"]["control"]
           or rec[r]["doc"]["config"]["selection_file"] != "results/selection_sweep_lora_a100_d165.json"
           or rec[r]["doc"]["config"]["target_fpr"] != P.TARGET_FPR]
    if bad:
        raise SystemExit(f"REFUSING: not the LoRA 4c records of record (control, LoRA A100 selection, 1%): {bad}")
    bad = [r for r in runs if sen[r]["doc"]["config"]["run"] != r
           or sen[r]["doc"]["config"]["target_fpr"] != P.TARGET_FPR]
    if bad:
        raise SystemExit(f"REFUSING: not the D-363 sensitivity records of those runs (run name, 1%): {bad}")
    return rec, sen, {"record": MANIFEST_RECORD, "sensitivity": MANIFEST_SENSITIVITY}


def cells_of_run(run: str, rec: dict[str, Any], sen: dict[str, Any], level: str,
                 arrays: dict[str, np.ndarray], n_draws: int) -> list[dict[str, Any]]:
    """Every (run, layer) cell at one level: the two readings' truth and flags, the terms, and the per-cell arrays.

    The two readings must carry the same layers and the same draw count; a mismatch is a refusal, never a trim.
    `n_draws` is --n-draws (N_DRAWS under the final name) and it is ENFORCED, on the two arrays that establish the
    count here and, through array_of's `want`, on every other array read of the cell."""
    lr, ls = layers_of(rec, level), layers_of(sen, level)
    if lr != ls or not lr:
        raise SystemExit(f"REFUSING: {run} {level}: the two readings' layer grids differ ({lr} vs {ls})")
    cut = HS.CLASSES[CLASS]
    want = n_draws + 1
    out: list[dict[str, Any]] = []
    for li in lr:
        base = f"L{li}|{PROBE_SET}|{level}"
        rr = {"record": array_of(rec, run, "record", f"{base}|realised_paired_ratio"),
              "sensitivity": array_of(sen, run, "sensitivity", f"{base}|realised_paired_ratio")}
        if rr["record"].size != rr["sensitivity"].size:
            raise SystemExit(f"REFUSING: {run} L{li} {level}: the record has {rr['record'].size} draws and the "
                             f"sensitivity {rr['sensitivity'].size}")
        if rr["record"].size != want:
            raise SystemExit(f"REFUSING: {run} L{li} {level}: {rr['record'].size} draws, not {n_draws} plus the "
                             f"frozen split (D-156)")
        truth = {k: HS.crosses(v, cut) for k, v in rr.items()}
        with np.errstate(divide="ignore", invalid="ignore"):
            dist = np.abs(np.log(rr["record"])) - LOG2
        cell: dict[str, Any] = {"run": run, "layer": li, "level": level, "n_draws": int(rr["record"].size - 1),
                                "truth": truth, "distance_log": dist, "stats": {}, "terms": {}, "counts": {}}
        for reading, side in (("record", rec), ("sensitivity", sen)):
            cell["stats"][reading] = {q: array_of(side, run, reading, f"{base}|{q}", want) for q in CELL_STATS}
        for det in DETECTORS:
            key = f"{base}|{DETECTOR_STAT[det]}"
            flag = {"record": HS.crosses(array_of(rec, run, "record", key, want), cut),
                    "sensitivity": HS.crosses(array_of(sen, run, "sensitivity", key, want), cut)}
            cell["terms"][det] = cell_terms(truth["record"], flag["record"], truth["sensitivity"],
                                            flag["sensitivity"])
            cell["counts"][det] = {k: counts_from_flags(truth[k], flag[k]) for k in flag}
            arrays[f"{run}|L{li}|{PROBE_SET}|{level}|{det}|flag_record"] = flag["record"].astype(np.uint8)
            arrays[f"{run}|L{li}|{PROBE_SET}|{level}|{det}|flag_sensitivity"] = flag["sensitivity"].astype(np.uint8)
            arrays[f"{run}|L{li}|{PROBE_SET}|{level}|{det}|bucket_code"] = bucket_code(cell["terms"][det])
        arrays[f"{run}|L{li}|{PROBE_SET}|{level}|truth_record"] = truth["record"].astype(np.uint8)
        arrays[f"{run}|L{li}|{PROBE_SET}|{level}|truth_sensitivity"] = truth["sensitivity"].astype(np.uint8)
        arrays[f"{run}|L{li}|{PROBE_SET}|{level}|distance_log_record"] = dist.astype("<f8")
        out.append(cell)
    return out


def anchor_counts(run: str, cells: list[dict[str, Any]], rec: dict[str, Any], sen: dict[str, Any],
                  level: str, n_draws: int) -> int:
    """The recomputed per-draw confusion counts, summed over the run's cells, against the two records' own counts
    arrays (D-402 (2): truth and flag exactly as D-255 computes them). Any difference is a refusal.

    The anchor runs over EVERY layer of the sweep, whatever --layers selects for the decomposition."""
    n = 0
    for det in DETECTORS:
        for reading, side in (("record", rec), ("sensitivity", sen)):
            for q in COUNTS:
                mine = np.sum([c["counts"][det][reading][q] for c in cells], axis=0)
                theirs = array_of(side, run, reading, f"counts|{PROBE_SET}|{level}|{det}|{CLASS}|{q}", n_draws + 1)
                if not np.array_equal(mine.astype(np.int64), theirs.astype(np.int64)):
                    raise SystemExit(f"REFUSING: {run} {level} {det} {q}: the recomputed counts are not the "
                                     f"{reading} npz's (first differing draw "
                                     f"{int(np.flatnonzero(mine != theirs)[0])})")
                n += 1
    return n


# ------------------------------------------------------------------------------------------------- the scoring

def mean_over_draws(a: np.ndarray) -> float:
    return float(np.asarray(a, dtype=float)[1:].mean())


def decompose_group(group: str, runs: list[str], level: str, det: str,
                    cells: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """One (group, level, detector) block: the two pooled readings, the terms' counts, the sequential and
    alone contributions with their shares, the per-cell table and the accuracy line of D-402 (6)."""
    idx = HS.pair_cluster_index(len(runs))
    n = cells[runs[0]][0]["truth"]["record"].size
    per_run = {reading: {} for reading in ("record", "sensitivity")}
    per_run_terms: dict[str, dict[str, np.ndarray]] = {}
    for r in runs:
        for reading in per_run:
            per_run[reading][r] = {q: np.sum([c["counts"][det][reading][q] for c in cells[r]], axis=0)
                                   for q in COUNTS}
        per_run_terms[r] = {q: np.sum([c["terms"][det][q].astype(np.int64) for c in cells[r]], axis=0)
                            for q in TERMS}
    blocks = {reading: pooled(per_run[reading], runs, idx) for reading in per_run}
    spec_rec, spec_sen = spec_of(blocks["record"]), spec_of(blocks["sensitivity"])
    if spec_rec is None or spec_sen is None:
        raise SystemExit(f"REFUSING: {group} {level} {det}: a pooled specificity is undefined (an empty denominator)")
    drop = spec_sen - spec_rec

    # The sequential decomposition: the terms applied in D-405 (3)'s order (CHOICE 2) must land on the sensitivity's
    # own counts. Every intermediate pooled call below is made with tp and fn ZEROED (zeros_like_counts), because
    # only the specificity denominator moves here; such a block's sensitivity is therefore not computed and is NOT
    # reported - only its specificity and that specificity's interval are read out of it (CHOICE 19).
    state = {r: (per_run["record"][r]["tn"].copy(), per_run["record"][r]["fp"].copy()) for r in runs}
    seq: dict[str, Any] = {}
    prev = spec_rec
    for step in STEPS:
        for r in runs:
            state[r] = apply_step(*state[r], step, per_run_terms[r])
        blk = pooled({r: {**zeros_like_counts(n), "tn": state[r][0], "fp": state[r][1]} for r in runs}, runs, idx)
        now = spec_of(blk)
        seq[step] = {"d405_term": STEP_D405[step], "terms_applied": list(STEP_TERMS[step]),
                     "specificity_after": now, "contribution": None if now is None else now - prev,
                     "share_of_drop": None if now is None or drop == 0 else (now - prev) / drop,
                     "specificity_ci95_after": blk["pooled_from_mean_counts"].get("specificity_ci", {}).get("ci95"),
                     "only_the_specificity_is_defined": "tp and fn are zeroed in this intermediate pooled call; no "
                                                        "sensitivity is computed or reported (CHOICE 19)"}
        prev = now if now is not None else prev
    for r in runs:
        for q, got in (("tn", state[r][0]), ("fp", state[r][1])):
            if not np.array_equal(got, per_run["sensitivity"][r][q]):
                raise SystemExit(f"REFUSING: {group} {level} {det} {r}: the decomposition does not land on the "
                                 f"sensitivity's own {q} counts")
    # 1e-9: the counts are integers and every specificity is a ratio of sums of their means, so the only error in
    # total_seq - drop is float64 rounding over the six steps - of order 1e-16 relative, i.e. about 1e-16 absolute
    # on a specificity in [0, 1]. 1e-9 is seven orders above that and seven below the smallest drop this diagnostic
    # could report (one triple out of the ~11 mean negatives of D-405 (1) moves the specificity by about 1e-2), so
    # the check cannot pass on a real residual and cannot fail on rounding.
    total_seq = sum(v["contribution"] for v in seq.values() if v["contribution"] is not None)
    if abs(total_seq - drop) > 1e-9:
        raise SystemExit(f"REFUSING: {group} {level} {det}: the sequential contributions sum to {total_seq!r}, not "
                         f"the drop {drop!r}")

    # a term applied ALONE to the record (D-402 (6)'s reconstruction, CHOICE 3)
    alone: dict[str, Any] = {}
    for step in ALONE:
        st = {r: apply_step(per_run["record"][r]["tn"].copy(), per_run["record"][r]["fp"].copy(), step,
                            per_run_terms[r]) for r in runs}
        blk = pooled({r: {**zeros_like_counts(n), "tn": st[r][0], "fp": st[r][1]} for r in runs}, runs, idx)
        now = spec_of(blk)
        alone[step] = {"d405_term": STEP_D405[step], "terms_applied": list(STEP_TERMS[step]),
                       "specificity": now, "contribution": None if now is None else now - spec_rec,
                       "share_of_drop": None if now is None or drop == 0 else (now - spec_rec) / drop,
                       "specificity_ci95": blk["pooled_from_mean_counts"].get("specificity_ci", {}).get("ci95"),
                       "only_the_specificity_is_defined": "tp and fn are zeroed in this intermediate pooled call; "
                                                          "no sensitivity is computed or reported (CHOICE 19)"}

    terms_total = {q: float(sum(mean_over_draws(per_run_terms[r][q]) for r in runs)) for q in TERMS}
    buckets_total = {"a_truth_moved": terms_total["a_truth_moved_record_tn"] + terms_total["a_truth_moved_record_fp"],
                     "b_detector_moved": terms_total["b_detector_moved"], "c_both": terms_total["c_both"],
                     "d_unchanged": (terms_total["d_unchanged_tn"] + terms_total["d_unchanged_fp"]
                                     + terms_total["d_unchanged_cleared"])}
    table, per_cell_pred, per_cell_real = per_cell_table(runs, det, cells)

    # D-402 (6): each RUN's sensitivity specificity reconstructed by bucket (b) alone
    run_rows = []
    for r in runs:
        tn, fp = mean_over_draws(per_run["record"][r]["tn"]), mean_over_draws(per_run["record"][r]["fp"])
        b_mean = mean_over_draws(per_run_terms[r]["b_detector_moved"])
        tn_s, fp_s = mean_over_draws(per_run["sensitivity"][r]["tn"]), mean_over_draws(per_run["sensitivity"][r]["fp"])
        pred = (tn - b_mean) / (tn + fp) if (tn + fp) > 0 else float("nan")
        real = tn_s / (tn_s + fp_s) if (tn_s + fp_s) > 0 else float("nan")
        run_rows.append({"run": r, "specificity_record": tn / (tn + fp) if (tn + fp) > 0 else None,
                         "mean_counts_record": {"tn": tn, "fp": fp},
                         "mean_count_bucket_b": b_mean,
                         "specificity_predicted_bucket_b_alone": None if not np.isfinite(pred) else pred,
                         "specificity_realised_sensitivity": None if not np.isfinite(real) else real,
                         "log_error": None if not (np.isfinite(pred) and np.isfinite(real) and pred > 0 and real > 0)
                         else float(np.log(real / pred))})
    return {"group": group, "runs": runs, "n_runs": len(runs), "level": level, "detector": det,
            "probe_set": PROBE_SET, "class": CLASS,
            "interval": f"pair-cluster bootstrap B={B}, seed {SEED}" if idx is not None
                        else f"point only (fewer than {MIN_PAIRS} runs)",
            "n_cells": sum(len(cells[r]) for r in runs),
            "pooled_record": blocks["record"], "pooled_sensitivity": blocks["sensitivity"],
            "specificity": {"record": spec_rec, "sensitivity": spec_sen, "drop": drop},
            "triple_counts_mean_over_draws": {"buckets": buckets_total, "terms": terms_total,
                                              "record_denominator": buckets_total["a_truth_moved"]
                                              + terms_total["b_detector_moved"] + terms_total["c_both"]
                                              + buckets_total["d_unchanged"],
                                              "terms_not_named_by_D402_3": list(UNNAMED_BY_D402_3),
                                              "terms_added_by_D405_2": list(UNNAMED_BY_D402_3)},
            "decomposition_order": list(STEPS), "decomposition_order_d405": [STEP_D405[s] for s in STEPS],
            "contributions_sequential": seq, "contributions_alone": alone,
            "sequential_sum_minus_drop": float(total_seq - drop),        # checked above; written, not asserted
            "accuracy_line_run": accuracy_line([r["specificity_realised_sensitivity"] for r in run_rows],
                                               [r["specificity_predicted_bucket_b_alone"] for r in run_rows], "run"),
            "accuracy_line_cell_beside": accuracy_line(per_cell_real, per_cell_pred, "(run, layer) cell"),
            "reconstruction_by_run": run_rows, "cells_ranked_by_bucket_b": table}


def per_cell_table(runs: list[str], det: str,
                   cells: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[float], list[float]]:
    """D-402 (4): the (run, layer) cells ranked by their contribution to bucket (b), with the reduced sizes, the two
    readings' per-draw mean and SD of the four statistics and the record's distance from the class boundary."""
    rows: list[dict[str, Any]] = []
    pred: list[float] = []
    real: list[float] = []
    for r in runs:
        for c in cells[r]:
            t = c["terms"][det]
            mean_t = {q: mean_over_draws(t[q].astype(np.int64)) for q in TERMS}
            den_rec = mean_t["a_truth_moved_record_tn"] + mean_t["a_truth_moved_record_fp"] + \
                mean_t["b_detector_moved"] + mean_t["c_both"] + mean_t["d_unchanged_tn"] + \
                mean_t["d_unchanged_fp"] + mean_t["d_unchanged_cleared"]
            tn_rec = mean_over_draws(c["counts"][det]["record"]["tn"])
            fp_rec = mean_over_draws(c["counts"][det]["record"]["fp"])
            tn_sen = mean_over_draws(c["counts"][det]["sensitivity"]["tn"])
            fp_sen = mean_over_draws(c["counts"][det]["sensitivity"]["fp"])
            p_cell = (tn_rec - mean_t["b_detector_moved"]) / (tn_rec + fp_rec) if (tn_rec + fp_rec) > 0 else float("nan")
            r_cell = tn_sen / (tn_sen + fp_sen) if (tn_sen + fp_sen) > 0 else float("nan")
            pred.append(p_cell)
            real.append(r_cell)
            d = np.asarray(c["distance_log"], dtype=float)
            stats = {}
            for reading, v in c["stats"].items():
                stats[reading] = {}
                for q in CELL_STATS:
                    bs = P.band_summary(v[q][1:], v[q][0])        # D-156's own report; CHOICE 9 takes two fields
                    stats[reading][q] = {"mean": bs["mean"], "sd_over_draws": bs["sd_over_draws"],
                                         "frozen": bs["frozen"], "n_nonfinite_draws": bs["n_nonfinite_draws"]}
            rows.append({
                "run": r, "layer": c["layer"], "level": c["level"], "detector": det,
                "mean_count_bucket_b": mean_t["b_detector_moved"],
                "share_of_draws_in_bucket_b": float(t["b_detector_moved"][1:].mean()),
                "frozen_split_bucket_code": int(bucket_code(t)[0]),
                "triple_counts_mean_over_draws": mean_t,
                "record_specificity_denominator_mean": den_rec,
                "specificity_record": tn_rec / (tn_rec + fp_rec) if (tn_rec + fp_rec) > 0 else None,
                "specificity_predicted_bucket_b_alone": None if not np.isfinite(p_cell) else p_cell,
                "specificity_realised_sensitivity": None if not np.isfinite(r_cell) else r_cell,
                "distance_from_class_boundary_log": {
                    "mean_signed": float(np.mean(d[1:][np.isfinite(d[1:])])) if np.isfinite(d[1:]).any() else None,
                    "mean_abs": float(np.mean(np.abs(d[1:][np.isfinite(d[1:])]))) if np.isfinite(d[1:]).any() else None,
                    "frozen_signed": float(d[0]) if np.isfinite(d[0]) else None,
                    "n_nonfinite_draws": int((~np.isfinite(d[1:])).sum()),
                    "definition": "|log(realised paired ratio)| - log 2, per draw, then the mean over the draws "
                                  "(D-402 (4), CHOICE 7)"},
                "per_draw": stats})
    rows.sort(key=lambda row: (-row["mean_count_bucket_b"], row["run"], row["layer"]))
    return rows, pred, real


# ------------------------------------------------------------------------------------------------- predictions

def predictions_registered() -> dict[str, Any]:
    """D-402 (5)'s four statements and the rule each is scored by. No number of the diagnostic appears here.

    D-405 (4) leaves P-T4.1 to P-T4.4 unchanged and still on disk before any number is read; D-405 (3) fixes that
    P-T4.1 is scored on the ALONE share, with the sequential share beside."""
    return {
        "preregistration": PREREG, "preregistration_entries": ["D-402", "D-405"],
        "amended_by": "D-405 (1)-(4), before any number of the diagnostic existed; the four statements themselves "
                      "are unchanged (D-405 (4))",
        "decomposition_order": list(STEPS), "decomposition_order_d405": [STEP_D405[s] for s in STEPS],
        "written_before": "any number of the diagnostic (D-402 (5))",
        "scored_on": {**PRIMARY, "note": "the registered primary cell only; every other block is beside and carries "
                                         "no verdict (CHOICE 15)"},
        "verdicts": ["HELD", "FAILED"],
        "undefined_rule": "a prediction whose rule cannot be evaluated scores FAILED with the reason named (CHOICE 16)",
        "predictions": {
            "P-T4.1": {"statement": "bucket (b) explains at least half the drop (the detector's own change, not "
                                    "reclassification)",
                       "rule": "the bucket-(b)-alone contribution to the pooled specificity, divided by the pooled "
                               "drop (sensitivity minus record), is at least 0.5; D-405 (3) fixes the ALONE share "
                               "as the verdict line, with the sequential share beside (CHOICE 4)",
                       "threshold": PRED_HALF, "direction": "at least"},
            "P-T4.2": {"statement": "bucket (b) cells sit closer to the class boundary than the unchanged negatives "
                                    "(a smaller median of ||log ratio| - log 2| in the record)",
                       "rule": "the median over bucket-(b) cells of the record's mean ||log(realised ratio)| - log 2| "
                               "is smaller than the median over the UNCHANGED cells, those whose record specificity "
                               "triples are all in bucket (d); the looser set (no bucket (b) triple) beside "
                               "(CHOICE 6, CHOICE 7)",
                       "threshold": None, "direction": "smaller"},
            "P-T4.3": {"statement": "the per-draw SD of the Gaussian ratio rises from record to sensitivity in at "
                                    "least 80% of bucket (b) cells (fewer items, a noisier estimate, not a "
                                    "systematic shift)",
                       "rule": f"the share of bucket-(b) cells whose sensitivity SD of gaussian_ratio exceeds the "
                               f"record's is at least 0.8. THE VERDICT LINE is the strict cut, a mean bucket (b) "
                               f"count above {B_CELL_CUT:g}; the {B_CELL_CUT_BESIDE:g} cut (the bucket in at least "
                               f"half the draws) is reported beside and carries no verdict (CHOICE 6)",
                       "threshold": PRED_SHARE, "direction": "at least"},
            "P-T4.4": {"statement": "the general runs' pooled specificity change is smaller in absolute value than "
                                    "the safety runs'",
                       "rule": "|specificity(sensitivity) - specificity(record)| for the three general (alpaca) runs "
                               "is smaller than the same quantity for the six safety runs, at the primary cell",
                       "threshold": None, "direction": "smaller"},
        },
    }


def score_predictions(primary: dict[str, Any], general: dict[str, Any]) -> dict[str, Any]:
    """D-402 (5): each prediction HELD or FAILED, with the value its rule read and where it came from."""
    out: dict[str, Any] = {}
    alone_b = primary["contributions_alone"]["b_detector_moved"]
    share = alone_b["share_of_drop"]
    out["P-T4.1"] = {"value": share, "threshold": PRED_HALF,
                     "beside": {"share_sequential": primary["contributions_sequential"]["b_detector_moved"]
                                ["share_of_drop"],
                                "contribution_alone": alone_b["contribution"],
                                "drop": primary["specificity"]["drop"]},
                     "verdict": "HELD" if share is not None and share >= PRED_HALF else "FAILED",
                     "note": None if share is not None else "the pooled drop is zero, so the share is undefined"}

    rows = primary["cells_ranked_by_bucket_b"]
    b_cells = bucket_b_cells(rows, B_CELL_CUT)
    u_cells = unchanged_cells(rows)
    loose = [r for r in rows if r["mean_count_bucket_b"] <= B_CELL_CUT
             and r["record_specificity_denominator_mean"] > 0]
    med_b = median_or_none([r["distance_from_class_boundary_log"]["mean_abs"] for r in b_cells])
    med_u = median_or_none([r["distance_from_class_boundary_log"]["mean_abs"] for r in u_cells])
    b_beside = bucket_b_cells(rows, B_CELL_CUT_BESIDE)
    out["P-T4.2"] = {"median_abs_distance_bucket_b_cells": med_b, "median_abs_distance_unchanged_cells": med_u,
                     "n_bucket_b_cells": len(b_cells), "n_unchanged_cells": len(u_cells),
                     "beside": {"cut": B_CELL_CUT_BESIDE, "n_bucket_b_cells": len(b_beside),
                                "median_abs_distance_bucket_b_cells": median_or_none(
                                    [r["distance_from_class_boundary_log"]["mean_abs"] for r in b_beside]),
                                "n_cells_with_no_bucket_b_triple": len(loose),
                                "median_abs_distance_no_bucket_b_cells": median_or_none(
                                    [r["distance_from_class_boundary_log"]["mean_abs"] for r in loose])},
                     "verdict": "HELD" if med_b is not None and med_u is not None and med_b < med_u else "FAILED",
                     "note": None if med_b is not None and med_u is not None
                     else "one of the two cell sets is empty, so the medians cannot be compared"}

    def sd_rose(cs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [r for r in cs
                if r["per_draw"]["sensitivity"]["gaussian_ratio"]["sd_over_draws"] is not None
                and r["per_draw"]["record"]["gaussian_ratio"]["sd_over_draws"] is not None
                and r["per_draw"]["sensitivity"]["gaussian_ratio"]["sd_over_draws"]
                > r["per_draw"]["record"]["gaussian_ratio"]["sd_over_draws"]]

    rose = sd_rose(b_cells)
    sh = len(rose) / len(b_cells) if b_cells else None
    rose_beside = sd_rose(b_beside)
    out["P-T4.3"] = {"value": sh, "threshold": PRED_SHARE, "n_bucket_b_cells": len(b_cells),
                     "n_cells_with_a_higher_sensitivity_sd": len(rose),
                     # CHOICE 6: the verdict line is the strict cut; the 0.5 cut is beside and carries no verdict
                     "verdict_line": f"the bucket (b) cells at the strict cut, mean bucket (b) count above "
                                     f"{B_CELL_CUT:g}",
                     "beside": {"cut": B_CELL_CUT_BESIDE, "n_bucket_b_cells": len(b_beside),
                                "n_cells_with_a_higher_sensitivity_sd": len(rose_beside),
                                "value": len(rose_beside) / len(b_beside) if b_beside else None},
                     "verdict": "HELD" if sh is not None and sh >= PRED_SHARE else "FAILED",
                     "note": None if sh is not None else "no bucket (b) cell, so the share is undefined"}

    ds, dg = primary["specificity"]["drop"], general["specificity"]["drop"]
    ok = ds is not None and dg is not None and abs(dg) < abs(ds)
    out["P-T4.4"] = {"change_safety": ds, "change_general": dg,
                     "abs_change_safety": None if ds is None else abs(ds),
                     "abs_change_general": None if dg is None else abs(dg),
                     "verdict": "HELD" if ok else "FAILED",
                     "note": None if ds is not None and dg is not None else "a pooled change is undefined"}
    for k in PRED_IDS:
        if out[k]["verdict"] not in ("HELD", "FAILED"):
            raise SystemExit(f"REFUSING: {k} has no HELD/FAILED verdict")
    return out


# -------------------------------------------------------------------------------------------------- the anchor

def reused_problems() -> list[str]:
    out = []
    for f, want in REUSED_SHA256.items():
        have = sha256_file(CODE_ROOT / f) if (CODE_ROOT / f).is_file() else None
        if have != want:
            out.append(f"{f} is not the reviewed bytes this module reuses ({(have or 'missing')[:12]} vs {want[:12]})")
    return out


def refuse_unless_reused_bytes(what: str) -> None:
    """D-402 (7), CHOICE 18: every subcommand refuses unless the reused files are at their reviewed bytes - the
    final name, a sandbox name and the lane list alike. There is no loose path: at other bytes `crosses`,
    `pooled_counts` and `pair_cluster_index` are a different estimator, so nothing written would mean what it says."""
    problems = reused_problems()
    if problems:
        raise SystemExit(f"REFUSING ({what}, D-402 (7), CHOICE 18): {problems}")


def anchor_summaries(blocks: dict[str, dict[str, Any]], s4c: dict[str, Any], s363: dict[str, Any]) -> dict[str, Any]:
    """The pooled record and sensitivity specificity recomputed here against the two summaries of record, and the
    safety readings against D-402's quoted pair. Any difference is a refusal (D-402 (2), CHOICE 14)."""
    key = f"{PROBE_SET}|{LEVELS[0]}"
    checked: dict[str, Any] = {"comparisons": [], "n_identical": 0}
    rows363 = s363["table_iv_lora_rows_all"]
    for group in GROUPS:
        row = GROUP_ROW[group]
        for det in DETECTORS:
            blk = blocks[f"{group}|{LEVELS[0]}|{det}"]
            cellk = f"{det}|{CLASS}"
            want_rec = s4c["rows"][key][row][cellk]["pooled_from_mean_counts"]["specificity"]
            want_sen = rows363["sensitivity"][key][row][cellk]["pooled_from_mean_counts"]["specificity"]
            want_rec2 = rows363["record"][key][row][cellk]["pooled_from_mean_counts"]["specificity"]
            for what, mine, theirs, src in (("record", blk["specificity"]["record"], want_rec, SUMMARY_4C_LORA),
                                            ("record", blk["specificity"]["record"], want_rec2, SUMMARY_D363),
                                            ("sensitivity", blk["specificity"]["sensitivity"], want_sen,
                                             SUMMARY_D363)):
                if theirs is None or mine is None or abs(mine - theirs) > 1e-12:
                    raise SystemExit(f"REFUSING: {group} {det} pooled {what} specificity {mine!r} is not {src}'s "
                                     f"{theirs!r}")
                checked["comparisons"].append({"group": group, "detector": det, "reading": what, "source": src,
                                               "value": theirs})
                checked["n_identical"] += 1
    for det, (q_rec, q_sen) in QUOTED.items():
        blk = blocks[f"safety|{LEVELS[0]}|{det}"]
        got = (round(blk["specificity"]["record"], 3), round(blk["specificity"]["sensitivity"], 3))
        if got != (q_rec, q_sen):
            raise SystemExit(f"REFUSING: the safety {det} readings round to {got}, not D-402's {(q_rec, q_sen)}")
    checked["quoted_by_d402"] = {d: list(v) for d, v in QUOTED.items()}
    checked["kind"] = ("the pooled record and sensitivity specificity of every group and detector at the item level, "
                       "against healthcheck_d156_summary_lora.json and hh_sensitivity_d363.json's "
                       "table_iv_lora_rows_all; the safety readings against D-402's quoted pair")
    return checked


# --------------------------------------------------------------------------------------------------- the writer

def write_outputs(out_path: Path, npz_path: Path, payload: dict[str, Any], npz_bytes: bytes) -> dict[Path, str]:
    problems = input_identity_problems(payload)
    if problems:
        raise SystemExit(f"REFUSING to write {out_path.name} (D-297 (1) results-writer rule): {problems}")
    json_bytes = json.dumps(payload, indent=2, sort_keys=False, default=str).encode("utf-8")
    digests = SO.write_all([(npz_path, npz_bytes), (out_path, json_bytes)])
    for p in digests:
        os.chmod(p, 0o444)
    return digests


def refuse_if_exists(*paths: Path) -> None:
    for p in paths:
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists; results are never overwritten (D-152 P5)")


def not_under_results(out: Path, *more: Path) -> None:
    """A non-final output must land outside results/ of BOTH trees: ROOT may be a test tree, but a test must never
    write under the repo's own results/ either (CODE_ROOT is where the source lives)."""
    for p in (out, *more):
        for base in {(ROOT / "results").resolve(), (CODE_ROOT / "results").resolve()}:
            if p.resolve().is_relative_to(base):
                raise SystemExit(f"REFUSING: a test output must not be under results/ ({base})")


# ------------------------------------------------------------------------------------------------- subcommands

def cmd_predictions(args: argparse.Namespace) -> int:
    refuse_unless_reused_bytes("the predictions job")
    final = (RESULTS / f"{STEM}_predictions.json").resolve()
    out = Path(args.out).resolve() if args.out else final
    if out != final:
        not_under_results(out)
    refuse_if_exists(out)
    reg = predictions_registered()
    payload = {**provenance_block(f"{RUN_KIND}_predictions"), "code": P.code_block(CODE_FILES),
               "preregistration": PREREG, "registered_parameters": True,
               "reused_files_sha256": {f: sha256_file(CODE_ROOT / f) for f in REUSED_SHA256},
               "registered": reg, "registered_hash": canonical_hash(reg),
               "population": POPULATION, "implementation_choices": CHOICES,
               "note": "D-402 (5): written before any number of the diagnostic exists. "
                       f"src/{Path(OWN).name}'s run binds this file's sha256 and refuses statements that are not "
                       "these; it scores each one HELD or FAILED in its own record."}
    digest = write_results(out, payload, exclusive=True)
    os.chmod(out, 0o444)
    print(f"Written: {out}\nsha256:  {digest}")
    for k in PRED_IDS:
        print(f"  {k}: {reg['predictions'][k]['statement']}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    refuse_unless_reused_bytes("the diagnostic")
    final = (RESULTS / f"{STEM}.json").resolve()
    out = Path(args.out).resolve() if args.out else final
    npz = out.with_name(out.stem + "_draws.npz")
    is_final = out == final
    if not is_final:
        not_under_results(out, npz)
    refuse_if_exists(out, npz)
    threads = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    registered = (args.layers is None and args.n_draws == N_DRAWS
                  and sorted(GROUPS) == sorted(("general", "safety")) and len(GROUPS["safety"]) == 6
                  and len(GROUPS["general"]) == 3 and (B, SEED) == (5000, 520260925))
    if is_final and not registered:
        raise SystemExit(f"REFUSING the final name: registered parameters {registered}")
    if args.n_draws < 1:
        raise SystemExit(f"REFUSING: --n-draws {args.n_draws} is not a draw count")
    # --layers is REFUSED on every path, never silently ignored and never recorded as if it had run: a narrowed
    # layer set cannot meet CHOICE 14's value-level anchor, which compares the pooled specificity of the WHOLE
    # population against the two summaries of record, so there is no path on which it could be applied honestly.
    if args.layers is not None:
        raise SystemExit(f"REFUSING: --layers {args.layers} is not applicable on any path: the anchor of CHOICE 14 "
                         f"compares the pooled specificity over EVERY layer of each run's sweep against "
                         f"{SUMMARY_4C_LORA} and {SUMMARY_D363}, so a narrowed layer set could not be anchored and "
                         f"a config naming it would not be what ran")
    code0 = P.code_block(CODE_FILES)

    inputs: dict[str, str] = {}
    pred_rel = f"results/{STEM}_predictions.json"
    pred_path = Path(args.predictions).resolve() if args.predictions else (ROOT / pred_rel)
    if not pred_path.is_file():
        raise SystemExit(f"REFUSING: no predictions file {pred_path}; D-402 (5) puts the predictions on disk before "
                         f"any number of the diagnostic is read (run `predictions` first)")
    pb = pred_path.read_bytes()
    pdoc = json.loads(pb.decode("utf-8"))                   # listed (D-297)
    if canonical_hash(pdoc.get("registered")) != canonical_hash(predictions_registered()):
        raise SystemExit(f"REFUSING: {pred_path} does not carry this module's registered predictions (D-402 (5))")
    inputs[rel_of(pred_path) if pred_path.is_relative_to(ROOT) else pred_path.name] = sha256_bytes(pb)

    runs = [r for g in GROUPS.values() for r in g]
    if len(runs) != len(set(runs)):
        raise SystemExit(f"REFUSING: the groups do not name distinct runs: {runs}")
    rec, sen, manifests = read_sides(runs, inputs)
    s4c = load_json_bound(SUMMARY_4C_LORA, PINNED["SUMMARY_4C_LORA"], inputs, "the pinned 4c LoRA summary (D-278)")
    s363 = load_json_bound(SUMMARY_D363, PINNED["SUMMARY_D363"], inputs, "the pinned D-363 summary (D-380)")

    arrays: dict[str, np.ndarray] = {}
    cells: dict[str, dict[str, list[dict[str, Any]]]] = {lvl: {} for lvl in LEVELS}
    anchored = 0
    for r in runs:
        for lvl in LEVELS:
            cells[lvl][r] = cells_of_run(r, rec[r], sen[r], lvl, arrays, args.n_draws)
            anchored += anchor_counts(r, cells[lvl][r], rec[r], sen[r], lvl, args.n_draws)
        print(f"  {r}: {len(cells[LEVELS[0]][r])} cells per level, counts anchored", flush=True)
    want_anchor = len(runs) * len(LEVELS) * len(DETECTORS) * 2 * len(COUNTS)
    if anchored != want_anchor:
        raise SystemExit(f"REFUSING: {anchored} anchored count arrays, not {want_anchor}")

    blocks: dict[str, dict[str, Any]] = {}
    for group, grs in GROUPS.items():
        for lvl in LEVELS:
            for det in DETECTORS:
                blocks[f"{group}|{lvl}|{det}"] = decompose_group(group, grs, lvl, det, cells[lvl])
    anchor = anchor_summaries(blocks, s4c, s363)
    primary_key = f"{PRIMARY['group']}|{PRIMARY['level']}|{PRIMARY['detector']}"
    scored = score_predictions(blocks[primary_key], blocks[f"general|{PRIMARY['level']}|{PRIMARY['detector']}"])

    sizes = {r: {"reduced_after_R": {"probe_train": sen[r]["doc"]["population"]["n_probe_train"],
                                     "threshold_calib_negatives": sen[r]["doc"]["population"]["n_calibration_negatives"],
                                     "evaluation_negatives": sen[r]["doc"]["population"]["n_evaluation_negatives"],
                                     "pooled_negatives": sen[r]["doc"]["population"]["n_pooled_negatives"],
                                     "evaluation_positives": sen[r]["doc"]["population"]["n_evaluation_positives"]},
                 "removed_from_this_run": sen[r]["doc"]["removal_set"]["removed_from_this_run"],
                 "d170_flag": {"record": rec[r]["flag"], "sensitivity": sen[r]["flag"]}} for r in runs}

    config = {"preregistration": "D-402 as amended by D-405", "preregistration_entries": ["D-402", "D-405"],
              "groups": GROUPS, "runs": runs, "probe_set": PROBE_SET, "class": CLASS,
              "class_cut": list(HS.CLASSES[CLASS]), "levels": list(LEVELS), "detectors": list(DETECTORS),
              "primary": PRIMARY, "n_draws": args.n_draws, "frozen_split_index": 0,
              "n_boot": B, "bootstrap_seed": SEED, "min_runs_for_an_interval": MIN_PAIRS,
              # the config is what RAN: --n-draws is applied (it is the draw count every per-draw array read must
              # carry) and --layers is refused on every path, so the layers are always the whole sweep
              "layers": "every layer of each run's sweep",
              "n_draws_applied_to": "every per-draw array read: an array that does not carry n_draws plus the "
                                    "frozen split is a refusal, never a broadcast",
              "layers_option": "refused on every path: a narrowed layer set cannot meet CHOICE 14's value-level "
                               "anchor over the whole population",
              "estimator": "D-255's pooled estimator (healthcheck_summary_d156.pooled_counts with "
                           "pair_cluster_index), imported unchanged; each run its own cluster",
              "reads": "the two record sets and their _draws.npz only; no cache read, no refit, no new draw "
                       "(D-402 (2))", "reads_caches": False, "refits": False, "makes_draws": False,
              "manifests": manifests, "pinned_records": {k: PINNED[k] for k in PINNED},
              "decomposition_order": list(STEPS), "decomposition_order_d405": [STEP_D405[s] for s in STEPS],
              "decomposition_order_settled_by": "D-405 (3)", "step_terms": {s: list(STEP_TERMS[s]) for s in STEPS},
              "terms": list(TERMS),
              "terms_not_named_by_D402_3": list(UNNAMED_BY_D402_3),
              "terms_added_by_D405_2": list(UNNAMED_BY_D402_3), "bucket_code": BUCKET_CODE,
              "predictions_file": rel_of(pred_path) if pred_path.is_relative_to(ROOT) else str(pred_path),
              "predictions_sha256": sha256_bytes(pb)}
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    npz_bytes = buf.getvalue()
    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"] or any(sha256_file(ROOT / f) != h for f, h in inputs.items()
                                                 if (ROOT / f).is_file()):
        raise SystemExit("REFUSING to write: code or an input changed during the run")
    payload = {**provenance_block(RUN_KIND), "code": code0, "preregistration": PREREG,
               "status": "FLAGGED (D-402 (9), D-401 (d)): a diagnostic reported BESIDE the two readings of record, "
                         "never in place of either and in no table; every number here is read back from this file",
               "under_check": "D-402 (9), D-401 (d): until the author scores this diagnostic, BOTH readings of "
                              "Table IV's safety-run specificity cell are reported wherever it appears (STATE, "
                              "docs/claims.md, docs/phase1_exit.md and any report), the record's first and the "
                              "sensitivity's beside; Table IV's LoRA-row marks stay (D-401 (c))",
               "reading": {"rule": "D-402 (3) as amended by D-405 (3): the pooled drop (sensitivity minus record) is "
                                   "the sum of the six terms' sequential contributions in the order (a), (b), (c), "
                                   "(f), (g), (e), which land on the sensitivity's own counts; a term's "
                                   "own share is its ALONE contribution (D-402 (6), D-405 (3), CHOICE 3). "
                                   "The primary cell is "
                                   "the safety group, selected C, item level, Gaussian, class two_sided_2x; the "
                                   "general group, the cluster level and the direct detector are beside and carry "
                                   "no verdict (CHOICE 15).",
                           "primary_block": f"{PRIMARY['group']}|{PRIMARY['level']}|{PRIMARY['detector']}"},
               "registered_parameters": registered, "blas_threads": threads,
               "reused_files_sha256": {f: sha256_file(CODE_ROOT / f) for f in REUSED_SHA256},
               "config": config, "config_hash": canonical_hash(config),
               "input_files_sha256": inputs, "anchor": {"counts_arrays_identical": anchored, **anchor},
               "population": POPULATION, "implementation_choices": CHOICES,
               "flagged": [
                   "D-402 (3)'s four buckets partition the RECORD's specificity triples only, so they cannot sum to "
                   "the drop: the specificity DENOMINATOR itself moves (D-405 (1)). The gap was raised by this "
                   "implementation under CLAUDE.md rule 5 and the author amended the pre-registration in D-405 (2), "
                   "(3): the three added terms - (e) entering, (f) a record false positive whose truth moved out, "
                   "(g) a record false positive whose flag cleared - are each their own named term and their own "
                   "step here (terms_added_by_D405_2), applied in D-405 (3)'s fixed order (a), (b), (c), (f), (g), "
                   "(e), landing exactly on the sensitivity's own counts; none is folded into a bucket's share. The "
                   "author reads the three terms' own contributions.",
                   "D-401 (d), D-402 (9): until this diagnostic is scored by the author, both readings of the Table IV "
                   "cell are reported side by side wherever it appears, the record's first and the sensitivity's "
                   "beside. Nothing here replaces either reading.",
                   "D-255's own FLAGGED note stands: the reference set and the evaluation part come from one pool of "
                   "negatives re-partitioned per draw, so every specificity here is an upper bound for a deployer "
                   "whose reference prompts differ from its traffic.",
               ],
               "reduced_sizes_per_run": sizes,
               "predictions": {"file": config["predictions_file"], "sha256": config["predictions_sha256"],
                               "registered": pdoc.get("registered"), "scored": scored},
               "blocks": blocks,
               "draws_npz": {"file": rel_of(npz) if npz.is_relative_to(ROOT) else str(npz),
                             "sha256": sha256_bytes(npz_bytes), "size": len(npz_bytes),
                             "keys": "<run>|L<layer>|<probe set>|<level>|truth_record / truth_sensitivity / "
                                     "distance_log_record and <run>|L<layer>|<probe set>|<level>|<detector>|"
                                     "flag_record / flag_sensitivity / bucket_code (BUCKET_CODE); index 0 the frozen "
                                     "split. Every aggregate in this record is recomputable from them."}}
    digests = write_outputs(out, npz, payload, npz_bytes)
    for p, h in digests.items():
        print(f"Written: {p}\nsha256:  {h}")
    print(f"anchored count arrays: {anchored}; summary comparisons: {anchor['n_identical']}")
    for k in PRED_IDS:
        print(f"  {k}: {scored[k]['verdict']}")
    return 0


# ------------------------------------------------------------------------------------------------- the lane list

def queue_rows() -> list[tuple[str, str, str]]:
    """(label, declared outputs, command): exactly what each job writes (D-402 (8), lane A, CPU)."""
    mod = f"src/{Path(OWN).name}"
    man = f"logs/{STEM}.sha256"
    files = f"results/{STEM}_predictions.json results/{STEM}.json results/{STEM}_draws.npz"
    return [
        (f"{QUEUE_TAG}_predictions", f"results/{STEM}_predictions.json",
         f"python3 -u {mod} predictions"),
        (f"{QUEUE_TAG}_run", f"results/{STEM}.json results/{STEM}_draws.npz",
         f"python3 -u {mod} run"),
        (f"manifest_{QUEUE_TAG}", man,
         f'[ ! -e {man} ] || {{ echo "REFUSING: {man} exists; never overwritten (D-152 P5)"; exit 1; }}; set -C; '
         f'n=$(ls {files} | wc -l); [ "$n" = "3" ] && sha256sum {files} > {man} && sync'),
    ]


def queue_header() -> list[str]:
    return [
        f"# Lane {QUEUE_LANE} (CPU only, no network; D-402 as amended by D-405, the author's D-401 (2d)): the "
        f"cell-level diagnostic of "
        f"Table IV's safety-run specificity drop on the D-363 reduced items (Gaussian 0.987 -> 0.784, direct 0.993 -> "
        f"0.884, D-380's unexplained finding), by src/{Path(OWN).name}. Every (run, layer, draw) triple of the six "
        f"safety runs is put in exactly one of D-402 (3)'s buckets in BOTH readings, and the pooled drop is written as "
        f"the sum of SIX terms applied in D-405 (3)'s fixed order - (a) truth moved, (b) detector moved, (c) both, "
        f"then (f) a record false positive whose truth moved out, (g) a record false positive whose flag cleared and "
        f"(e) entering - each its own named term, never folded into a bucket's share (D-405 (2)); the three general "
        f"runs and the D-164 cluster level beside. Reads the "
        f"records of record (results/healthcheck_d156_lora*.json, named by {MANIFEST_RECORD}), the D-363 sensitivity "
        f"(results/hh_sensitivity_d363_lora*.json, named by {MANIFEST_SENSITIVITY}) and their _draws.npz, plus the two "
        f"pinned summaries ({SUMMARY_4C_LORA}, {SUMMARY_D363}). NO cache read, NO refit, NO new draw, no NAS write "
        f"beyond its own outputs (D-402 (8)).",
        f"# DO NOT LOAD before the three reviews of src/{Path(OWN).name} against the AMENDED text (D-405 (7): the "
        f"reviews run against D-402 as D-405 amends it; new analysis code, D-402 (7), D-357 (A)(3)) "
        f"and the push that carries it and this list.",
        "# ORDER: (1) t4diag_predictions writes the four pre-registered statements of D-402 (5) and their scoring "
        "rules, with no number of the diagnostic in it; (2) t4diag_run REFUSES unless that file exists and carries "
        "this module's statements, binds its sha256, and scores each one HELD or FAILED; (3) the manifest. A failed "
        "job stops the list (queue runner); nothing downstream runs on a partial set.",
        "# ANCHOR, before anything is written: the per-draw confusion counts recomputed from the stored arrays with "
        "healthcheck_summary_d156.crosses, summed over a run's cells, must equal that run's own npz counts arrays in "
        "both readings (every level, both detectors, class two_sided_2x), and the pooled record and sensitivity "
        "specificity must equal the two summaries of record, the safety readings rounding to D-402's quoted pair. "
        "REFUSALS: a missing per-draw array; a draw-count or layer-grid mismatch between the two readings; a record "
        "that is not the one its manifest or pin names, or not a registered run; a manifest whose own sha256 is not "
        "the pinned one (a swapped manifest would name swapped records); a per-draw array that does not carry 1000 "
        "draws plus the frozen split; a reused file whose sha256 is not "
        "the reviewed bytes (REUSED_SHA256: healthcheck_summary_d156.py, healthcheck_d156.py, neither changed; "
        "EVERY subcommand refuses on it, CHOICE 18); an "
        "output that already exists (results are never overwritten); a decomposition that does not land on the "
        "sensitivity's own counts or whose contributions do not sum to the drop.",
        "# Final names need the registered parameters (1000 draws plus the frozen split, B = 5000, seed 520260925, "
        "every layer, the six safety and three general runs), the pinned records and manifests and the reused files "
        "at their reviewed bytes. No BLAS pin is required (a records-only summary that fits nothing, CHOICE 11; the "
        "threads are recorded). Runs from the repo root through scripts/detach.sh.",
        "# CPU TIME: an estimate, not a measurement of these jobs (D-402 (8): minutes to about an hour). It reads 18 "
        "records and 18 npz (about 100 MB in all), recomputes 306 cells x 1001 draws of boolean arrays and runs the "
        "pair-cluster bootstrap (B = 5000) over 8 blocks x 11 pooled calls; the predictions job is seconds and the "
        "manifest seconds. One job at a time on this lane.",
        "# After the pull: verify against logs/" + STEM + ".sha256; add results/" + STEM + "_draws.npz to the dated "
        "draw-array manifest (D-163 (6), src/write_draw_manifest.py); declare the population group "
        "TABLE4_SPECIFICITY_D402 in src/populations.py, outside primary_result_stems (D-380's after-pull steps, "
        "CHOICE 12); then the author reads the four verdicts and the three terms D-405 (2) added - (e) entering, "
        "(f) and (g) - each reported on its own. Until the author scores it, BOTH readings of the Table IV cell are "
        "reported side by side (D-401 (d), D-402 (9)).",
        f"# Labels checked unique against queues/lane*.tsv by the writer of this list (src/{Path(OWN).name} "
        f"queue --write); check them against the node's logs/queue/{QUEUE_LANE}/ before loading.",
    ]


def cmd_queue(args: argparse.Namespace) -> int:
    refuse_unless_reused_bytes("the lane list")
    if not args.write:
        for label, outs, cmd in queue_rows():
            print(f"{label}\t{outs}\t{cmd}")
        return 0
    name = f"lane{QUEUE_LANE}_{args.date}_{QUEUE_TAG}.tsv"
    out = Path(args.out).resolve() if args.out else (QUEUES / name)
    if out.exists():
        raise SystemExit(f"REFUSING: {out} exists; a loaded list is never rewritten")
    rows = queue_rows()
    labels = [r[0] for r in rows]
    if len(labels) != len(set(labels)):
        raise SystemExit(f"REFUSING: this list's own labels are not unique: {labels}")
    taken: dict[str, str] = {}
    for q in sorted(QUEUES.glob("lane*.tsv")):
        if q.resolve() == out.resolve():
            continue
        for line in q.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                taken.setdefault(line.split("\t")[0], q.name)
    clash = {lb: taken[lb] for lb in labels if lb in taken}
    if clash:
        raise SystemExit(f"REFUSING: these labels are already used by another list: {clash}")
    text = "\n".join([*queue_header(), *[f"{lb}\t{o}\t{c}" for lb, o, c in rows]]) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(f"Written: {out}\n{len(rows)} jobs, labels {labels}")
    return 0


def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Table IV's safety-run specificity drop, cell by cell (D-402, D-405)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("predictions", help="D-402 (5): write the four statements before any number exists")
    pp.add_argument("--out", default=None)
    rp = sub.add_parser("run", help="the diagnostic: the decomposition, the per-cell table and the verdicts")
    rp.add_argument("--out", default=None)
    rp.add_argument("--predictions", default=None, help="TESTING: the predictions file to bind")
    rp.add_argument("--n-draws", type=int, default=N_DRAWS, dest="n_draws",
                    help="TESTING: the draw count every per-draw array must carry; never a final name")
    rp.add_argument("--layers", type=int, nargs="*", default=None,
                    help="REFUSED on every path (kept so an old call fails loudly instead of silently running the "
                         "whole sweep): CHOICE 14's anchor needs every layer")
    qp = sub.add_parser("queue", help="the lane A list of D-402 (8)")
    qp.add_argument("--write", action="store_true")
    qp.add_argument("--date", default=utc_now()[:10].replace("-", ""))
    qp.add_argument("--out", default=None)
    args = ap.parse_args()
    return {"predictions": cmd_predictions, "run": cmd_run, "queue": cmd_queue}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
