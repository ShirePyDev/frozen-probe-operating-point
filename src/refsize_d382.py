"""D-382: the reference-size trade-off. How many reference prompts do the Gaussian health ratio and the direct
reference ratio need, with and without A1's score translations?

Pre-registered in DECISIONS.md D-382 before this code or any number of it existed (the author's message of 2026-10-07,
step 4 item 2; details by the closest D-entries, D-181: D-241 (2) the two ratios and the direct floor, D-255 the 4c
estimator and its pooled scoring, D-327 A1's c grid, D-353 A1's results, D-361 (2) the Phase 2 cells, D-156, D-164,
D-177 (b)). NO reviewed module is edited. The reviewed Phase 2 scoring of src/twomoment_p2_d361.py is imported and
called (passing_anchor, durable_select, check_pair_plan, plan_grid, open_sub_cache, p2_cluster_units, fit_probes,
score_matrices, make_parts, bos_check, float16_cross_check, probe_fingerprints, accuracy_line, final_problems,
read_manifest, read_listed, write_outputs, end_hash_problems), together with the reviewed 4c functions
(healthcheck_d156.thresholds, health, crosses, core_of, CLASSES, LEVELS) and 4c's pooled estimator
(healthcheck_summary_d156.pooled_counts, pair_cluster_index, rate, COUNTS), unchanged; every one of those files is
bound by sha256 in every record this module writes (REUSED_SHA256), and a final name refuses other bytes. Where a
reviewed script computes inside its own main(), the lines are repeated here and pinned by an anchor (below).

WHAT IS MEASURED (D-382 (1), (2), (3), (4))
  n          the reference size, n in {200, 500, 1000, 5000} (N_GRID); a population supplies only the n it can hold
             WITHOUT replacement (D-382 rejects a reference drawn with replacement), so Phase 1 runs 200, 500, 1000.
  PRIMARY    the nine Phase 2 headline pairs of D-361 (2) (twomoment_p2_d361.HEADLINE_PAIRS), the only population that
             holds 5,000 reference negatives: split v2 (calibration 5,000 negatives, evaluation 6,000, positives
             4,000), D-361's cells (GRID6 plus the pre side's L*), user-span mean pooling, the selected C of record,
             item level, 1%, its D-156 estimator.
  BESIDE     the nine Phase 1 released pairs (populations.UPDATE_PAIRS_REPAIR) at n = 200, 500, 1000; n = 1000 is 4c's
             record and is the ANCHOR (below), reproduced bit for bit.
  PER DRAW   T_d from the FULL calibration part's pre scores, as D-361 and 4c set it (healthcheck_d156.thresholds), NOT
             from the n-subset. The reference is the first n items of ONE SEEDED PERMUTATION of that draw's calibration
             part (REF_PERM_SEED per axis, one permutation matrix shared by every cell of the axis, as D-156 shares one
             partition per draw), so the reference sets are nested in n and n is not confounded with which items. The
             Gaussian ratio and the direct ratio are healthcheck_d156.health's, so exactly D-241 (2)'s, and the direct
             floor is 1/n with n the REFERENCE size (health's own n); beside them, per draw, the share of SEEDS whose
             reference share is AT that floor, per side and either side (CHOICES (q), D-353 (3)'s count). The class is
             the realised paired ratio on the draw's evaluation part (paired.ratio_from_flags on the same T_d), which
             does not depend on n.
  A1 SHIFTS  (D-382 (3), D-327 (2)) the POST scores of every item, reference and evaluation alike, translated by
             c * sd_pre, sd_pre = the SD of the draw's PRE calibration-negative scores per seed (D-327's sigma: the
             exact effect of A1's white-box feature-space delta on a linear probe), for c in D-327's grid without 0
             (C_GRID: -3, -2, -1, -0.5, -0.25, 0.25, 0.5, 1, 2, 3). The class is recomputed on the shifted realised
             ratio. The unshifted setting ("none") is c = 0 by construction and is the no-shift row of D-382 (2); its
             post scores are passed through untouched, so the anchor is bit for bit exact.
  METRICS    4c's summary (D-241 (6), D-255): per detector (gaussian, direct) and class (healthcheck_d156.CLASSES) the
             per-draw confusion counts summed over a pair's cells, pooled sensitivity and specificity from the per-pair
             D-156 mean counts with the pair-cluster bootstrap (healthcheck_summary_d156.pooled_counts and
             pair_cluster_index: B = 5000, seed 520260925, D-177 (b)); and per draw the ROC AUC across a row's cells of
             |log Gaussian ratio| and |log direct ratio| (D-156 mean and band).

MODES (each output written once: staged, hard-linked to its final name, read back; never overwritten)
  predictions  results/refsize_d382_predictions.json: D-382 (5)'s P-N.1, P-N.2 and P-N.3 as registered text with their
               scoring rules and verdict NOT YET RUN, written BEFORE any number of this item exists. Every later job
               refuses unless it exists and was written by this exact code; the summary scores each one HELD or FAILED
               from its own rows and refuses unless the registered text is unchanged.
  p1 --pair S  one Phase 1 released pair: results/refsize_d382_p1_<core>.json and its _draws.npz (D-163 (6)).
  p2 --pair K  one Phase 2 headline pair (user-span mean): results/refsize_d382_p2_<K>.json and its _draws.npz.
  preflight --lane A|D   every record check of that lane's jobs up to the first cache read, then "RECORDS OK"; no cache
               is read, no evaluation number is computed and nothing is written.
  summary      results/refsize_d382_summary.json from the 18 records and their npz, each bound to its line of
               logs/refsize_d382.sha256 (the manifest job): the rows, the three verdicts and D-177's accuracy line on
               every predicted-against-realised table.
  queue --write   writes the two lane lists (queues/laneA_<date>_refsize.tsv, queues/laneD_<date>_refsize.tsv).

THE ANCHORS (nothing is written before they pass)
  PHASE 1   at n = |cal_d| = 1000 and shift "none" the reference IS the draw's calibration part in its own order, so
            every one of healthcheck_d156.health's statistics and the realised paired ratio must equal the pair's 4c
            npz (results/healthcheck_d156_<core>_draws.npz, bound by its record and by logs/healthcheck_d156.sha256)
            BYTE FOR BYTE at every layer, and the realised report must be paired.repartition_estimate's on the same
            partitions (4c's own check, healthcheck_d156.main's line).
  PHASE 2   twomoment_p2_d361.passing_anchor (the D-361 anchor record: the Phase 2 path reproduced a Phase 1 pair's
            record, written by D-361's exact code); the probes at L* equal the selection's own fit bit for bit and
            reproduce its ho20 TPRs (selection_tpr_v2.seed_probes, anchor_diffs); the realised operating point at 1%,
            item level, equals the pair's D-361 record (results/twomoment_p2_d361_<K>_usermean.json, bound through
            logs/twomoment_p2_d361.sha256) at every layer, and the unshifted realised per-draw array is
            paired.repartition_estimate's; on D-364's cell the float16 cross-check of twomoment_p2_d361 CHOICES (o).

POPULATION GROUP: these outputs deliberately sit OUTSIDE every group of src/populations.py for now and are declared as
REFSIZE_D382 after the pull, exactly as D-381's records are, because populations.validate() requires every declared
file to exist on disk and no output of this item exists until the lane has run.

Final names need the registered parameters (1000 draws, seeds 0-4, every layer, the registered n and c grids), the
6-thread BLAS pin (scripts/project_env.sh), GUARDRAIL_CACHE_DIRS unset, the predictions record, and the reused files at
their reviewed bytes (REUSED_SHA256). A test writes outside results/ (--out / --out-dir).

CHOICES (D-181: every choice D-382 did not fix, each with the D-entry that settles it; written into every record)
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_d156 as D  # noqa: E402  (imported, not changed: LEVELS, PROBE_SETS, CLUSTERS, the selections of record)
import float16_check_d364 as F16  # noqa: E402  (imported, not changed: read_v2_eval, gather_split, check_v1_keyed)
import healthcheck_d156 as HC  # noqa: E402  (imported, not changed: thresholds, health, crosses, core_of, CLASSES)
import healthcheck_summary_d156 as HS  # noqa: E402  (imported, not changed: pooled_counts, pair_cluster_index, rate)
import input_registry as REG  # noqa: E402
import metrics as M  # noqa: E402
import paired as P  # noqa: E402
import populations as POP  # noqa: E402
import selection_p2 as SP  # noqa: E402  (imported, not changed)
import selection_tpr_v2 as T  # noqa: E402  (imported, not changed)
import subset_p2 as SUB  # noqa: E402  (imported, not changed: grid6)
import twomoment_fit as TF  # noqa: E402  (imported, not changed: d156_estimates)
import twomoment_p2_d361 as TP2  # noqa: E402  (imported, not changed: the reviewed Phase 2 scoring)
from provenance import (  # noqa: E402
    _RESULT_READS, _repo_rel, canonical_hash, ensure_utf8_stdout, input_identity_problems, provenance_block,
    read_result, resolve_cache, sha256_bytes, sha256_file,
)

ROOT = Path(__file__).resolve().parent.parent
CODE_ROOT = Path(__file__).resolve().parent.parent      # where the source files are hashed (ROOT may be a test tree)
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
QUEUES = ROOT / "queues"
OWN = "src/refsize_d382.py"
CODE_FILES = list(dict.fromkeys([OWN, *TP2.CODE_FILES, *HC.CODE_FILES, *HS.CODE_FILES]))
# the reviewed files whose functions this module calls, at the bytes it was written and tested against; a final name
# refuses other bytes, so an edit of a reused file cannot pass unnoticed (twomoment_p2_d361's own rule)
REUSED_SHA256 = {
    "src/twomoment_p2_d361.py": "f4741f8f1d8d601ce630a9e63c917933e7b5f5533ed729680fdb36ffa58d8d5f",
    "src/healthcheck_d156.py": "0ec2687ecdd887a28d4b8fb7a4db4b9081b69ea3fcf2b175c345279b4ed74de7",
    "src/healthcheck_summary_d156.py": "b149b66181fe1c2d953d255355e44e2d15651bff9ccf8a456819092db526ccf5",
    "src/depth_d156.py": "a1f7b50ce70395f290b681f336e5a5e95e1428a2ca7c5915818644ad70f26734",
    "src/paired.py": "80429fa57f0267236a2af1670c74ef327802598257ab75834fddf64a70d97195",
    "src/twomoment_fit.py": "1a1c57668068c4e86dfb08bc61c141d9c921d92a04da3b620ec7781ae9713b26",
    "src/selection_tpr_v2.py": "e928a44e969e8fc1ec87a2e9bb93f3c9ddd11f06778dd3066b1d9d0905c3746d",
    "src/selection_p2.py": "a79bb1b8a635c15190bad181e29c1ecd885ef01a71fda26e1101339f3bc59e97",
    "src/float16_check_d364.py": "6027c597844e4c684a530c15af3fd40d06bbd218275ee50ee7f00e4a2fa356b7",
    "src/subset_p2.py": "fc81a30170b0047c4d4606cd5f700001b2f7e2afb45b1e7d130f8269155576c5",
    "src/input_registry.py": "e05d1fba11a4db8c46d0e7231fa7b804fbd8101017fda1c0889070c193684fbf",
    "src/metrics.py": "f7aac316793d69097939ecbeb1de0719534d0bb0cdff04534a2d7de0a454b9ed",
}
RUN_KIND = {"predictions": "refsize_d382_predictions", "p1": "refsize_d382_p1", "p2": "refsize_d382_p2",
            "summary": "refsize_d382_summary"}
PREREG = ("D-382 (the author's message of 2026-10-07, step 4 item 2; written before any code or number of it existed); "
          "details by the closest D-entries (D-181): D-241 (2) (the Gaussian and direct ratios, the direct floor 1/n), "
          "D-255 (the 4c estimator, classes and pooled scoring), D-327 (A1's c grid and white-box translation), D-353 "
          "(A1's results), D-361 (2) (the Phase 2 cells, split v2 and the D-156 estimator), D-156, D-164, D-177 (b)")

# ----------------------------------------------------------------------------- the registered grids and seeds
N_GRID = (200, 500, 1000, 5000)                         # D-382 (1)
C_GRID = (-3.0, -2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0, 3.0)      # D-327 (2)'s grid without 0 (D-382 (3))
HEALTH = ("z_mean", "log_sd_ratio", "gaussian_ratio", "log_gaussian_ratio", "direct_ratio")  # healthcheck_d156.health
# the per-seed share AT the direct floor 1/n, per side and either side (CHOICES (q); health() returns no per-seed share)
FLOORED = ("direct_floored_pre", "direct_floored_post", "direct_floored_either")
REALISED = "realised_paired_ratio"
FOURC_STATS = (*HEALTH, REALISED)
DETECTORS = ("gaussian", "direct")                      # D-382 (4); the AUC rule does not depend on n (CHOICES (d))
CLASSES = HC.CLASSES
PRIMARY_CLASS = "two_sided_2x"
LEVEL = HC.LEVELS[0]                                    # item_level_D156 (D-382 (1): item level)
PSET = "selected"                                       # the selected C of record (D-382 (1))
COUNTS = HS.COUNTS
POOLED_C = "pooled_over_c"                              # the 10 shifted settings summed (P-N.3)
SEEDS = [0, 1, 2, 3, 4]
REGISTERED = {"n_draws": P.N_DRAWS, "seeds": SEEDS, "n_grid": list(N_GRID), "c_grid": list(C_GRID)}
TARGET = P.TARGET_FPR                                   # 1% (D-382 (1))
# D-156: one reference permutation per draw on an axis, the same for every cell of the axis. Fixed here before any run,
# distinct from every partition seed in the repo (paired.AXIS_DRAW_SEED, AXIS_CLUSTER_DRAW_SEED,
# twomoment_p2_d361.P2_DRAW_SEED 20261006 and P2_CLUSTER_DRAW_SEED 20261007).
REF_PERM_SEED = {"safety": 20261012, "safety_wildguard": 20261013, SP.AXIS: 20261014}
# three axes, three distinct seeds: an axis-name collision (SP.AXIS equal to a Phase 1 axis name) would silently
# overwrite a seed and give two populations the same reference permutation
assert len(REF_PERM_SEED) == len(set(REF_PERM_SEED.values())) == 3, REF_PERM_SEED
P2_POOLING = TP2.PRIMARY_POOLING                        # usermean (D-361 (2))
P1_PAIRS = list(POP.UPDATE_PAIRS_REPAIR)                # the nine released pairs (4b/4c/4d population)
P2_PAIRS = list(TP2.HEADLINE_PAIRS)                     # the nine Phase 2 headline pairs (D-361 (2))
MANIFEST_4C = "logs/healthcheck_d156.sha256"            # D-255: the ten verified 4c records
MANIFEST_P2MECH = TP2.RUN_MANIFEST                      # logs/twomoment_p2_d361.sha256 (D-361's list manifest job)
MANIFEST_OWN = "logs/refsize_d382.sha256"               # this list's manifest job (the summary reads it)
PRED_REL = "results/refsize_d382_predictions.json"
FACTOR = TP2.FACTOR                                     # D-177: within a factor of 1.5

# ----------------------------------------------------------------------------- D-382 (5): the three predictions
REGISTERED_PREDICTIONS = {
    "P-N.1": {
        "text": "without shifts the direct ratio's ROC AUC (D-156 mean) is non-decreasing over n = 200, 500, 1000, 5000",
        "row": "the primary population (the nine Phase 2 headline pairs of D-361 (2)), shift none, the primary class "
               "(realised paired ratio >= 2 or <= 0.5), selected C, item level, 1%",
        "quantity": "the D-156 mean of the per-draw ROC AUC of |log direct ratio| across the row's cells, per n",
        "rule": "HELD iff every value is defined and mean(200) <= mean(500) <= mean(1000) <= mean(5000); FAILED if any "
                "step decreases; UNDEFINED (not a verdict) if a value is undefined",
    },
    "P-N.2": {
        "text": "without shifts, at n = 200, the Gaussian ratio's ROC AUC exceeds the direct ratio's",
        "row": "the primary population, shift none, n = 200, the primary class, selected C, item level, 1%",
        "quantity": "the D-156 means of the per-draw ROC AUC of |log Gaussian ratio| and of |log direct ratio| across "
                    "the row's cells",
        "rule": "HELD iff both are defined and the Gaussian mean is STRICTLY larger than the direct mean (a tie is "
                "FAILED)",
    },
    "P-N.3": {
        "text": "with the A1 shifts pooled over c, the direct ratio's pooled specificity at n = 5000 is at least its "
                "pooled specificity at n = 200",
        "row": "the primary population, the 10 shifted settings of D-327's grid without 0 pooled (their confusion "
               "counts summed per pair, draw and cell), the primary class, selected C, item level, 1%",
        "quantity": "the pooled specificity from the per-pair D-156 mean counts (healthcheck_summary_d156.pooled_counts)"
                    " at n = 5000 and at n = 200",
        "rule": "HELD iff both are defined and specificity(5000) >= specificity(200) (at or above, so a tie is HELD)",
    },
}
BESIDE_NOTE = ("every other row is BESIDE and never a verdict (D-382 (5)): the Phase 1 released pairs at n = 200, 500, "
               "1000; the Gaussian detector's rows for P-N.1 and P-N.3; every shifted setting by c; the per-draw "
               "sensitivity and specificity bands; the accuracy lines")

CHOICES = [
    "(a) The reference rows are the SET given by the first n of the draw's permutation, indexed in the calibration "
    "part's OWN order (np.sort of that prefix). The selection itself is the literal prefix, so the sets are nested in n "
    "(D-382 (2): 'nested sets, so n is not confounded with which items'), and at n = |cal_d| the reference is the "
    "calibration part exactly as 4c reads it, which is what makes the Phase 1 anchor bit for bit (D-382 (1): 'n = 1000 "
    "is 4c's record and is the anchor, reproduced bit for bit'). Summation order, not membership, is what this fixes.",
    "(b) The permutation seed: one fixed seed per axis (REF_PERM_SEED), drawn once per job as a (n_draws + 1, n_cal) "
    "matrix shared by every cell, probe and shift of the axis, so reference r is the same subset everywhere on the axis "
    "(D-156's rule for partitions, applied to the reference draw); row 0 is the frozen split's. NO D-entry registers a "
    "REFERENCE permutation seed: these three seeds are this module's own choice (D-181), fixed in this file before any "
    "run, distinct from every partition seed in the repo, and follow the form D-164's implementation spec sets for the "
    "partition seeds it adds ('permuted with a fixed per-axis seed distinct from the item-level one', 'one partition "
    "per draw shared by every cell, seed, and pre/post').",
    "(c) sd_pre is the SD of the draw's PRE calibration-negative scores PER SEED over the FULL calibration part, the "
    "population D-327 (2) names ('sigma, the SD of the pre-update scores of cal_d's negatives for that draw and seed'); "
    "it does not depend on n, so the shift is the same whatever the reference size. numpy's population SD (ddof 0), as "
    "healthcheck_d156.health computes its own moments.",
    "(d) Detectors: the Gaussian health ratio and the direct reference ratio only (D-382 (2), (4)). The AUC rule is "
    "threshold-free and its flag is the same in every draw (D-241 (4)), so it cannot depend on the reference size and "
    "is outside this item; its value is in 4c's record (D-255).",
    "(e) Level and probe set: the item level (D-156) and the selected C, which D-382 (1) names through D-361 (2); the "
    "D-164 cluster level and C=1 are 4c's and D-361's sensitivity rows and are not recomputed here (no new estimand).",
    "(f) The unshifted setting 'none' passes the post scores through untouched (no addition of a zero array), so it is "
    "c = 0 arithmetically and bit for bit, which is what the Phase 1 anchor compares (D-382 (2), (3)).",
    "(g) The class is recomputed per shift from the shifted realised paired ratio (D-382 (3)) with the SAME T_d (set on "
    "the pre scores, never on the post), as D-327 (2) attacks only the post side and D-156 freezes the threshold.",
    "(h) The Phase 1 cells are every layer of the pair's grid at the selected C (4c's population, D-241 (5)); the Phase "
    "2 cells are every layer both sides hold, which must include GRID6 plus the pre side's L* (D-361 (2), D-325 (2); "
    "twomoment_p2_d361 CHOICES (f), whose check is called here).",
    "(i) A population supplies only the n it can hold without replacement: Phase 1's calibration part holds 1,000, so "
    "n = 5000 is REFUSED there (D-382's rejected alternative 'a reference drawn with replacement to reach 5,000 on "
    "Phase 1 (no new information)').",
    "(j) The accuracy line (D-177 (3), D-255 OPEN): the Gaussian and direct ratios ARE predictions of the realised "
    "paired ratio, so every predicted-against-realised table here carries twomoment_p2_d361.accuracy_line (imported "
    "unchanged): the share of cells within x1.5 and the median |log(realised/predicted)| with n cells. Counting a ZERO "
    "or UNDEFINED prediction as a MISS is that reviewed function's own convention, which extends D-322 (4)'s rule for "
    "the median (D-322 (4) counts the cells where predicted and measured disagree in SIGN as misses rather than "
    "dropping them, and requires both conventions to be reported); Phase 1's defined-cells-only convention is reported "
    "beside, as D-322 (4) asks. Per-cell values are the D-156 estimates of twomoment_fit.d156_estimates (log of the "
    "mean over the draws where both are defined), as D-174 and D-177 (2a) form them.",
    "(k) P-N.3's 'pooled over c' sums the per-draw confusion counts of the 10 shifted settings over the settings and "
    "over the pair's cells before pooling over pairs, so a cell contributes one count per shift and draw (D-241 (6)'s "
    "pooling, extended over the c grid D-382 (3) registers; the per-c rows are beside).",
    "(l) Pooled intervals are healthcheck_summary_d156's unchanged (pair-cluster bootstrap B = 5000, seed 520260925, "
    "pairs resampled, point only below 3 pairs; D-177 (b)); per-draw bands cover partitions only and pooled intervals "
    "pairs only (D-166).",
    "(m) The ROC AUC per draw across a row's cells of |log predicted ratio| is D-241 (6)'s row, and repeats "
    "healthcheck_summary_d156.scoring's lines (metrics.auroc, paired.band_summary) for |log Gaussian ratio| "
    "(healthcheck_d156.health's own log of the seed mean, never log 0, D-241 (2)) and |log direct ratio| (the log of "
    "the "
    "floored ratio of D-241 (2)); a draw with no cell in a class or none outside it has an undefined AUC, counted.",
    "(n) The Phase 2 side binds the D-361 per-pair record of record and compares the realised operating point at 1%, "
    "item level, at every layer (canonical JSON), and the unshifted realised per-draw array against "
    "paired.repartition_estimate's; D-382 (6) orders this item after the D-361 run, so that record exists.",
    "(o) The predictions record is written by its own job before any number of this item exists (D-382 (5), the D-327 "
    "(6) pattern), and every later job refuses without it; the summary refuses unless its registered text is exactly "
    "REGISTERED_PREDICTIONS.",
    "(p) The two lane lists hold the per-pair jobs only; the manifest and the summary must run after BOTH lanes and are "
    "loaded by the next review session. D-382 (6) orders this item and no D-entry places its manifest or summary job: "
    "the split itself follows D-401 (g) (the author's item 3: the MLP list 'priced from the node sandbox and split over "
    "lanes C and D'), which settles the SPLIT only, and leaving the manifest and the summary out of both lists is this "
    "module's own choice (D-181), because each reads the outputs of both lanes (D-382 (4), (5)).",
    "(q) AT THE DIRECT FLOOR is counted PER SEED and per side: a side is at the floor when no reference item of that "
    "seed is at or above T_d, so its share is exactly the 1/n floor of D-241 (2) (healthcheck_d156.health returns the "
    "mean over seeds only, so the per-seed indicator is computed here beside it, from the same rows and T_d). Reported "
    "as the mean over seeds per draw (the D-156 mean over draws in the record) for the pre side, the post side and "
    "either side, with the share of draws in which any seed is floored. D-353 (3) quotes the analogous count ('7 misses "
    "have the realised ratio at the floor') and D-174 a floored-cell sensitivity, so the count is the quantity of "
    "record; a direct ratio of exactly 1 is NOT that count and is reported under its own name.",
]


def choice(letter: str) -> str:
    return next(c for c in CHOICES if c.startswith(f"({letter}) "))


# ----------------------------------------------------------------------------- small helpers
def refuse(msg: str) -> None:
    raise SystemExit(f"REFUSING: {msg}")


def rel(p: Path | str) -> str:
    q = Path(p).resolve()
    return str(q.relative_to(ROOT.resolve())).replace("\\", "/") if q.is_relative_to(ROOT.resolve()) else q.name


def canon(x: Any) -> str:
    """Canonical JSON text (json.dumps default=str, parsed back, keys sorted); not a result (D-297)."""
    return json.dumps(json.loads(json.dumps(x, default=str)), sort_keys=True)


def blas_threads() -> dict[str, str | None]:
    return {v: os.environ.get(v) for v in SP.THREAD_VARS}


def reused_now() -> dict[str, str | None]:
    return {f: (sha256_file(CODE_ROOT / f) if (CODE_ROOT / f).is_file() else None) for f in REUSED_SHA256}


def ckey(c: float) -> str:
    """The shift key of c; 'none' is the unshifted setting (CHOICES (f))."""
    return "none" if c == 0.0 else f"c={c:g}"


SHIFTS: tuple[str, ...] = ("none", *(ckey(c) for c in C_GRID))
SHIFT_C: dict[str, float] = {"none": 0.0, **{ckey(c): float(c) for c in C_GRID}}
SHIFTED: tuple[str, ...] = tuple(s for s in SHIFTS if s != "none")


def ns_for(n_cal: int) -> list[int]:
    """The registered n a calibration part of n_cal items can hold without replacement (CHOICES (i))."""
    return [n for n in N_GRID if n <= n_cal]


def check_ns(ns: list[int], n_cal: int) -> None:
    if not ns:
        refuse(f"a calibration part of {n_cal} items holds no registered reference size {list(N_GRID)}")
    bad = [n for n in ns if n > n_cal or n < 1]
    if bad:
        refuse(f"reference sizes {bad} do not fit a calibration part of {n_cal} items without replacement "
               f"(D-382's rejected alternative; {choice('i')})")


def check_anchor_size(ns: list[int], n_cal: int) -> None:
    """|cal_d| must itself be one of the registered reference sizes, so that the LARGEST reference of the run IS the
    whole calibration part in its own order (CHOICES (a)) and n = |cal_d| is a row of the grid: D-382 (1) registers
    1000 for Phase 1 and 5000 for Phase 2 (split v2's calibration part). On the PHASE 1 side that row is also what the
    anchor compares byte for byte with the pair's 4c npz (D-382 (1)); the Phase 2 anchor does not compare a reference at
    all (it is the D-361 anchor record, the selection's own probe fit and the D-361 pair record's realised operating
    point, CHOICES (n)), so there this is a registration check only."""
    if n_cal not in ns:
        refuse(f"the calibration part holds {n_cal} items, which is not a registered reference size {list(N_GRID)}: "
               "the largest reference of a run must be the whole calibration part (D-382 (1))")


def ref_selection(perm_row: np.ndarray, n: int) -> np.ndarray:
    """D-382 (2): the reference SELECTION of one draw at size n, the first n of its permutation (a literal prefix, so
    the selections are nested in n)."""
    return np.asarray(perm_row)[:n]


def ref_rows(perm_row: np.ndarray, n: int) -> np.ndarray:
    """The rows of the calibration part the reference is scored on: the same set, in the calibration part's own order
    (CHOICES (a))."""
    return np.sort(ref_selection(perm_row, n))


def ref_permutations(n_cal: int, n_draws: int, seed: int) -> np.ndarray:
    """(n_draws + 1, n_cal) permutations of the calibration part's positions, row 0 the frozen split's (CHOICES (b))."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(n_cal) for _ in range(n_draws + 1)])


def axis_perm_seed(axis: str) -> int:
    if axis not in REF_PERM_SEED:
        refuse(f"no registered reference permutation seed for axis {axis!r} ({choice('b')})")
    return REF_PERM_SEED[axis]


def final_problems(registered: bool) -> list[str]:
    """Why a final name may not be written (empty when it may)."""
    out: list[str] = []
    if not registered:
        out.append(f"non-registered parameters (the final names need {REGISTERED})")
    th = blas_threads()
    if any(v != SP.THREADS for v in th.values()):
        out.append(f"no 6-thread BLAS pin (D-152 P6): {th}")
    if os.environ.get("GUARDRAIL_CACHE_DIRS"):
        out.append("GUARDRAIL_CACHE_DIRS is set; the caches must come from their recorded directories")
    for f, want in REUSED_SHA256.items():
        have = reused_now()[f]
        if have is None:
            out.append(f"{f} is missing: this module reuses its reviewed functions")
        elif want is not None and have != want:
            out.append(f"{f} is not the reviewed bytes this module reuses ({have[:12]} vs {want[:12]})")
    return out


def check_final_gates(final: bool, registered: bool) -> None:
    """The final-name gates on EVERY entry point, before anything is read or written: the predictions job and the
    summary write a final name too, and the predictions record licenses every other job, so an edited reused file, a
    missing BLAS pin, a set GUARDRAIL_CACHE_DIRS or unregistered parameters must stop each of them, not only the two
    per-pair jobs (D-152 P5, D-382 (5), CHOICES (o))."""
    if final:
        bad = final_problems(registered)
        if bad:
            refuse(f"the final names need: {bad}")


def read_manifest_map(path: Path, pinned: str | None = None) -> tuple[dict[str, str], str]:
    """{repo-relative file: sha256} of a sha256sum manifest and the manifest's own sha256
    (twomoment_p2_d361.read_manifest, unchanged)."""
    return TP2.read_manifest(path, pinned)


def read_bound(rel_path: str, listed: dict[str, str] | None, man_name: str) -> tuple[dict[str, Any], str]:
    """A results JSON read through read_result (D-297 (1)), refused unless `listed` names it with these bytes."""
    p = ROOT / rel_path
    if p.is_symlink() or not p.is_file():
        refuse(f"{rel_path} is missing")
    doc = read_result(p)
    h = _RESULT_READS[_repo_rel(p)]
    if listed is not None and listed.get(rel_path) != h:
        refuse(f"{rel_path} is not the file {man_name} names ({h[:12]} vs {str(listed.get(rel_path))[:12]})")
    if not isinstance(doc, dict):
        refuse(f"{rel_path} is not a JSON object")
    return doc, h


def npz_bound(rel_path: str, listed: dict[str, str] | None, want: str,
              man_name: str) -> tuple[dict[str, np.ndarray], str]:
    """A per-draw npz read once from its bytes, which must be its record's (want) and its manifest's, when given."""
    p = ROOT / rel_path
    if p.is_symlink() or not p.is_file():
        refuse(f"{rel_path} is missing")
    b = p.read_bytes()
    h = sha256_bytes(b)
    if h != want or (listed is not None and listed.get(rel_path) != h):
        refuse(f"{rel_path} is not the npz its record and {man_name} name ({h[:12]} vs {str(want)[:12]})")
    with np.load(io.BytesIO(b), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}, h


def same_arrays(mine: np.ndarray, rec: np.ndarray) -> bool:
    return mine.dtype == rec.dtype and mine.shape == rec.shape and mine.tobytes() == rec.tobytes()


def anchor_arrays(mine: dict[str, np.ndarray], rec: dict[str, np.ndarray], keys: list[tuple[str, str]],
                  what: str) -> int:
    """Refuses unless every (my key, record key) pair holds byte-identical arrays; returns the number compared."""
    missing = [rk for _, rk in keys if rk not in rec]
    if missing:
        refuse(f"the anchor: {what} has no arrays {missing[:3]}; nothing written")
    bad = [rk for mk, rk in keys if not same_arrays(mine[mk], rec[rk])]
    if bad:
        refuse(f"the anchor: {len(bad)} of {len(keys)} arrays differ from {what} (first {bad[:3]}); nothing written")
    return len(keys)


def passing_predictions(code0: dict[str, Any], records_dir: Path | None = None) -> dict[str, Any]:
    """D-382 (5), CHOICES (o): the predictions record, read as a registered input. It must exist, be this module's
    predictions record, carry the registered text unchanged and have been written by this exact code."""
    p = (records_dir or RESULTS) / Path(PRED_REL).name
    if p.is_symlink() or not p.is_file():
        refuse(f"no predictions record {p.name}: run `predictions` first; D-382 (5) writes the predictions BEFORE any "
               "number of this item exists")
    doc = read_result(p)
    problems = []
    if doc.get("run_kind") != RUN_KIND["predictions"]:
        problems.append("not a predictions record")
    if canon(doc.get("predictions")) != canon(REGISTERED_PREDICTIONS):
        problems.append("its registered prediction text is not this module's REGISTERED_PREDICTIONS")
    if (doc.get("code") or {}).get("sha256") != code0["sha256"]:
        problems.append("it was written by other code")
    if canon(doc.get("reused_files_sha256")) != canon(reused_now()):
        problems.append("it was written against other bytes of the reused reviewed files (REUSED_SHA256)")
    if canon(doc.get("grids")) != canon({"n_grid": list(N_GRID), "c_grid": list(C_GRID), "shifts": list(SHIFTS)}):
        problems.append("it registered other n or c grids")
    if problems:
        refuse(f"the predictions record {p.name} cannot license this run: {problems}")
    return {"file": rel(p), "sha256": _RESULT_READS[_repo_rel(p)], "labels": sorted(REGISTERED_PREDICTIONS),
            "reused_files_sha256_equal": True}


def status_block(registered: bool, flags: list[str], note: str | None = None) -> dict[str, Any]:
    """The record's STATUS, as the evidence standard asks of every record: FINAL, FLAGGED (with the D-entry that flags
    it) or PROVISIONAL (not a run of record: non-registered parameters, a subset of layers or a test output)."""
    status = "PROVISIONAL" if not registered else ("FLAGGED" if flags else "FINAL")
    out: dict[str, Any] = {
        "status": status, "d_entries": list(flags),
        "rule": "CLAUDE.md evidence standard: FINAL, FLAGGED (with the D-entry) or PROVISIONAL. PROVISIONAL here means "
                f"the run did not use the registered parameters {REGISTERED} at every layer, so it is not a record of "
                "record; FLAGGED means an input of this record carries a D-entry's flag, named in d_entries."}
    if note:
        out["note"] = note
    return out


def seeds_block(axis: str, draw_seed: int) -> dict[str, Any]:
    """Every seed this item uses, in one place (the evidence standard): the probe seeds, the D-156 partition seed of
    the axis, the reference permutation seed (CHOICES (b)) and the pair-cluster bootstrap's (D-177 (b))."""
    return {"probe_seeds": list(SEEDS), "d156_partition_seed": draw_seed,
            "reference_permutation_seed": axis_perm_seed(axis),
            "pair_cluster_bootstrap": {"n_boot": HS.B, "seed": HS.SEED, "min_pairs": HS.MIN_PAIRS,
                                       "rule": "healthcheck_summary_d156.pair_cluster_index, unchanged (D-177 (b))"}}


def tp2_code_block(code0: dict[str, Any]) -> dict[str, Any]:
    """twomoment_p2_d361's OWN code block, taken as the restriction of this run's to its CODE_FILES. It cannot be
    computed by calling paired.code_block(TP2.CODE_FILES): D-152 P5 refuses a block that omits a loaded src module,
    and this module and the 4c modules are loaded. The restriction is what twomoment_p2_d361 itself would compute, so
    its anchor record's code block compares value for value."""
    keys = [str(Path(f)).replace("\\", "/") for f in TP2.CODE_FILES]
    missing = [k for k in keys if k not in code0["sha256"]]
    if missing:
        refuse(f"refsize_d382.CODE_FILES does not cover twomoment_p2_d361's: {missing}")
    return {**code0, "sha256": {k: code0["sha256"][k] for k in keys}}


def write_outputs(items: list[tuple[Path, bytes]]) -> None:
    """twomoment_p2_d361.write_outputs, unchanged: stage every output, then name it (never overwritten)."""
    TP2.write_outputs(items)


def json_bytes_of(payload: dict[str, Any]) -> bytes:
    b = json.dumps(payload, indent=2, sort_keys=False, default=str).encode("utf-8")
    json.loads(b.decode("utf-8"))      # not a result (D-297): this run's own bytes, parsed before staging
    return b


# ----------------------------------------------------------------------------- the per-partition arithmetic
def partition_arrays(pool_pre: np.ndarray, pool_post: np.ndarray, parts: list[tuple[np.ndarray, np.ndarray]],
                     perm: np.ndarray, ns: list[int], target: float = TARGET
                     ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """One cell's per-partition arrays (index 0 the frozen split), as D-382 (2) and (3) register them:
      T_d        healthcheck_d156.thresholds on the FULL calibration part's pre scores (never the n-subset);
      sd_pre     the per-seed SD of those same pre scores (D-327's sigma, CHOICES (c));
      reference  the first n of this draw's permutation, in the calibration part's own order (CHOICES (a));
      per shift  the post scores of the reference AND of the evaluation part translated by c * sd_pre; the unshifted
                 setting passes them through untouched (CHOICES (f));
      per (n, shift)  healthcheck_d156.health's five statistics (the Gaussian and direct ratios among them, direct
                 floor 1/n with n the reference size), and the per-seed AT-THE-FLOOR indicators of CHOICES (q);
      per shift  the realised paired ratio on the evaluation part at the same T_d (paired.ratio_from_flags,
                 healthcheck_d156.main's line), which does not depend on n.
    The pools must be one (n_pool, n_seeds) matrix each and every partition must index all n_pool rows exactly once, so
    a short array refuses instead of broadcasting over a partition (a length-1 pool is not a population).
    Returns ({'n<n>|<shift>|<stat>' and '<shift>|realised_paired_ratio': (n_parts,) '<f8'}, sd_pre (n_parts, S))."""
    n_cal = int(np.asarray(parts[0][0]).size)
    check_ns(ns, n_cal)
    if perm.shape != (len(parts), n_cal):
        refuse(f"the reference permutations are {perm.shape}, not {(len(parts), n_cal)}: a draw-count mismatch "
               f"({choice('b')})")
    if int(perm.min()) < 0 or int(perm.max()) >= n_cal:
        refuse(f"the reference permutations index outside the calibration part (0..{n_cal - 1}): {choice('a')}")
    pre_a, post_a = np.asarray(pool_pre), np.asarray(pool_post)
    if pre_a.ndim != 2 or post_a.shape != pre_a.shape or pre_a.shape[1] < 1:
        refuse(f"the score pools are {pre_a.shape} (pre) and {post_a.shape} (post): each must be the SAME "
               "(n_pooled_negatives, n_seeds) matrix, so that no shift, threshold or statistic broadcasts a short "
               "array over a partition")
    n_pool = int(pre_a.shape[0])
    if n_pool < n_cal:
        refuse(f"the score pools hold {n_pool} rows, fewer than the calibration part's {n_cal}")
    out: dict[str, list[float]] = {f"n{n}|{sk}|{q}": [] for n in ns for sk in SHIFTS for q in (*HEALTH, *FLOORED)}
    out.update({f"{sk}|{REALISED}": [] for sk in SHIFTS})
    sd_rows: list[np.ndarray] = []
    for i, (c_, e_) in enumerate(parts):
        ci, ei = np.asarray(c_), np.asarray(e_)
        if ci.ndim != 1 or ei.ndim != 1 or ci.dtype.kind not in "iu" or ei.dtype.kind not in "iu":
            refuse(f"partition {i} is not a pair of 1-D integer index arrays into the pools")
        if int(ci.size) != n_cal:
            refuse("a calibration part is not of the frozen size (D-156, D-164)")
        idx = np.concatenate([ci, ei])
        seen = None if (int(idx.size) != n_pool or int(idx.min()) < 0) else np.bincount(idx, minlength=n_pool)
        if seen is None or int(seen.size) != n_pool or int(seen.min()) != 1:
            refuse(f"partition {i} does not index each of the {n_pool} pooled negatives exactly once (calibration "
                   f"{ci.size}, evaluation {ei.size}): the pools and the partitions are not the same population "
                   "(D-156: one partition of the pool per draw)")
        cp, cq = pre_a[ci], post_a[ci]
        ep, eq = pre_a[ei], post_a[ei]
        T = HC.thresholds(cp, target)                       # D-382 (2): the FULL calibration part
        sd = cp.std(axis=0)                                 # D-327 (2): sigma per seed (CHOICES (c))
        sd_rows.append(np.asarray(sd, dtype="<f8"))
        F_pre_e = ep >= T
        rows = {n: ref_rows(perm[i], n) for n in ns}
        k_pre = {n: (cp[rows[n]] >= T).sum(axis=0) for n in ns}      # per seed; the pre side never moves (CHOICES (g))
        for sk in SHIFTS:
            c = SHIFT_C[sk]
            if c == 0.0:                                    # CHOICES (f): untouched, so bit for bit 4c's
                eq_s, cq_s = eq, cq
            else:
                d = c * sd
                eq_s, cq_s = eq + d, cq + d
            out[f"{sk}|{REALISED}"].append(P.ratio_from_flags(F_pre_e, eq_s >= T))
            for n in ns:
                r = rows[n]
                h = HC.health(cp[r], cq_s[r], T)
                for q in HEALTH:
                    out[f"n{n}|{sk}|{q}"].append(h[q])
                # CHOICES (q): the share of SEEDS whose reference share is AT the 1/n floor, i.e. no reference item at
                # or above T_d on that side (health() returns the seed mean only, so it is counted here)
                f0, f1 = k_pre[n] == 0, (cq_s[r] >= T).sum(axis=0) == 0
                for q, x in zip(FLOORED, (f0, f1, f0 | f1)):
                    out[f"n{n}|{sk}|{q}"].append(float(np.asarray(x).mean()))
    return ({k: np.asarray(v, dtype="<f8") for k, v in out.items()},
            np.asarray(sd_rows, dtype="<f8"))


def detector_ratios(v: dict[str, np.ndarray], n: int, sk: str) -> dict[str, np.ndarray]:
    return {"gaussian": v[f"n{n}|{sk}|gaussian_ratio"], "direct": v[f"n{n}|{sk}|direct_ratio"]}


def cell_block(v: dict[str, np.ndarray], ns: list[int], sd: np.ndarray) -> dict[str, Any]:
    """One cell's level block: healthcheck_d156.main's per-cell lines, per reference size and shift."""
    blk: dict[str, Any] = {"n_grid": list(ns), "shifts": list(SHIFTS), "by_shift": {}, "by_n_and_shift": {},
                           "sd_pre_per_seed_mean_over_draws": [float(x) for x in sd[1:].mean(axis=0)],
                           "sd_pre_frozen_per_seed": [float(x) for x in sd[0]]}
    for sk in SHIFTS:
        rr = v[f"{sk}|{REALISED}"]
        blk["by_shift"][sk] = {
            "c": SHIFT_C[sk], REALISED: P.band_summary(rr[1:], rr[0]),
            "share_of_draws_in_class": {c: float(HC.crosses(rr[1:], cut).mean()) for c, cut in CLASSES.items()}}
    for n in ns:
        for sk in SHIFTS:
            rr = v[f"{sk}|{REALISED}"]
            ratios = detector_ratios(v, n, sk)
            b: dict[str, Any] = {"n_reference": n, "c": SHIFT_C[sk], "direct_floor": 1.0 / n,
                                 "gaussian_ratio": P.band_summary(v[f"n{n}|{sk}|gaussian_ratio"][1:],
                                                                  v[f"n{n}|{sk}|gaussian_ratio"][0]),
                                 "direct_ratio": P.band_summary(v[f"n{n}|{sk}|direct_ratio"][1:],
                                                                v[f"n{n}|{sk}|direct_ratio"][0]),
                                 "share_of_draws_flagged": {f"{d}|{c}": float(HC.crosses(x[1:], cut).mean())
                                                            for d, x in ratios.items()
                                                            for c, cut in CLASSES.items()},
                                 # CHOICES (q): AT THE FLOOR means no reference item at or above T_d on that side, so
                                 # that side's share is the 1/n floor; it is a per-seed event, counted over seeds
                                 "at_the_direct_floor": {
                                     **{f"share_of_seed_draws|{q}": float(v[f"n{n}|{sk}|{q}"][1:].mean())
                                        for q in FLOORED},
                                     "share_of_draws_with_any_seed_floored_either":
                                         float((v[f"n{n}|{sk}|direct_floored_either"][1:] > 0.0).mean()),
                                     "frozen_split_share_of_seeds_floored_either":
                                         float(v[f"n{n}|{sk}|direct_floored_either"][0]),
                                     "rule": choice("q")},
                                 # NOT the floored share: the draws whose seed-mean direct ratio is exactly 1
                                 "share_of_draws_with_direct_ratio_exactly_1":
                                     float((v[f"n{n}|{sk}|direct_ratio"][1:] == 1.0).mean())}
            if sk == "none":
                for q in ("z_mean", "log_sd_ratio", "log_gaussian_ratio"):
                    b[q] = P.band_summary(v[f"n{n}|{sk}|{q}"][1:], v[f"n{n}|{sk}|{q}"][0])
            blk["by_n_and_shift"][f"n{n}|{sk}"] = b
    return blk


def add_counts(confusion: dict[str, np.ndarray], v: dict[str, np.ndarray], ns: list[int]) -> None:
    """healthcheck_d156.main's confusion lines per reference size and shift, summed over a pair's cells; plus the
    'pooled over c' sums of the 10 shifted settings (CHOICES (k))."""
    for n in ns:
        for sk in SHIFTS:
            rr = v[f"{sk}|{REALISED}"]
            ratios = detector_ratios(v, n, sk)
            for c, cut in CLASSES.items():
                y = HC.crosses(rr, cut)
                for dname in DETECTORS:
                    f = HC.crosses(ratios[dname], cut)
                    for key, val in (("tp", y & f), ("fn", y & ~f), ("fp", ~y & f), ("tn", ~y & ~f)):
                        for tag in ((sk,) if sk == "none" else (sk, POOLED_C)):
                            k_ = f"n{n}|{tag}|{dname}|{c}|{key}"
                            confusion[k_] = confusion.get(k_, np.zeros(rr.shape, dtype=np.int64)) + val


def expected_count_keys(ns: list[int]) -> list[str]:
    return [f"n{n}|{tag}|{d}|{c}|{q}" for n in ns for tag in (*SHIFTS, POOLED_C) for d in DETECTORS for c in CLASSES
            for q in COUNTS]


# ----------------------------------------------------------------------------- the Phase 1 job (beside)
def p1_bind(pair: str, n_draws: int, seeds: list[int], final: bool, records_only: bool,
            code0: dict[str, Any]) -> dict[str, Any]:
    """Every record this Phase 1 job reads, before any cache: the predictions record, the pair's 4c record of record and
    its npz (bound through logs/healthcheck_d156.sha256), its selection of record, the cluster manifest and the input
    registry's declared run. Refuses a draw-count or parameter mismatch with the 4c record."""
    if pair not in P1_PAIRS:
        refuse(f"{pair!r} is not one of the nine Phase 1 released pairs: {P1_PAIRS}")
    core = HC.core_of(pair)
    preds = None if records_only else passing_predictions(code0)
    listed, man_sha = read_manifest_map(LOGS / Path(MANIFEST_4C).name)
    rec_rel = f"results/healthcheck_d156_{core}.json"
    rec_doc, rec_sha = read_bound(rec_rel, listed, MANIFEST_4C)
    rcfg = rec_doc.get("config") or {}
    if (final and rec_doc.get("registered_parameters") is not True) or rcfg.get("control"):
        refuse(f"{rec_rel} is not a registered 4c run of a released pair")
    if rcfg.get("n_draws") != n_draws:
        refuse(f"{rec_rel} ran {rcfg.get('n_draws')} draws, this run {n_draws}: a draw-count mismatch; the anchor "
               "compares per-draw arrays draw for draw")
    if list(rcfg.get("seeds") or []) != list(seeds) or rcfg.get("target_fpr") != TARGET:
        refuse(f"{rec_rel} ran seeds {rcfg.get('seeds')} at {rcfg.get('target_fpr')}, this run {seeds} at {TARGET}")
    npz_rel = (rec_doc.get("draws_npz") or {}).get("file")
    rec_z, npz_sha = npz_bound(str(npz_rel), listed, (rec_doc.get("draws_npz") or {}).get("sha256"), MANIFEST_4C)
    reg_stem = D.registry_stem(f"depth_{core}")
    sel_path = D.SELECTION_OF_RECORD
    sel_stem = D.SELECTION_STEM.get(f"depth_{core}", reg_stem)
    sel_doc = read_result(sel_path)
    sel_sha = _RESULT_READS[_repo_rel(sel_path)]
    sel = next((m for m in sel_doc.get("models") or [] if m.get("pair") == sel_stem), None)
    if sel is None or not str((sel.get("selected") or {}).get("arm", "")).startswith("C="):
        refuse(f"no logistic-C selection for {sel_stem} in {rel(sel_path)}")
    C_sel, L_sel = float(sel["selected"]["arm"].split("=")[1]), int(sel["selected"]["layer"])
    if (rcfg.get("selected_C"), rcfg.get("selected_layer"), rcfg.get("selection_sha256")) != (C_sel, L_sel, sel_sha):
        refuse(f"{rec_rel} was run on another selection of record")
    cl_bytes = (D.CLUSTERS).read_bytes()
    cl_sha = sha256_bytes(cl_bytes)
    if rcfg.get("cluster_manifest", {}).get("sha256") not in (None, cl_sha):
        refuse("manifests/negative_clusters.json is not the cluster manifest the 4c record read")
    cfg = REG.run(reg_stem)
    out: dict[str, Any] = {"core": core, "predictions": preds, "rec_rel": rec_rel, "rec_doc": rec_doc,
                           "rec_sha": rec_sha, "npz_rel": str(npz_rel), "rec_z": rec_z, "npz_sha": npz_sha,
                           "reg_stem": reg_stem, "sel_path": sel_path, "sel_stem": sel_stem, "sel_sha": sel_sha,
                           "C_sel": C_sel, "L_sel": L_sel, "cl_bytes": cl_bytes, "cl_sha": cl_sha, "cfg": cfg,
                           "axis": cfg["axis"], "manifests": {MANIFEST_4C: man_sha}}
    if final:
        bad = final_problems(n_draws == REGISTERED["n_draws"] and list(seeds) == REGISTERED["seeds"])
        if bad:
            refuse(f"the final names need: {bad}")
    return out


def run_p1(pair: str, n_draws: int = P.N_DRAWS, seeds: list[int] | None = None, out: str | Path | None = None,
           layers: list[int] | None = None, records_only: bool = False) -> Path:
    """One Phase 1 released pair (BESIDE, D-382 (1)): n = 200, 500, 1000 at every layer of the pair's grid, the selected
    C, item level, 1%; the n = 1000 unshifted arrays anchored on the pair's 4c npz byte for byte."""
    t0 = time.time()
    seeds = list(SEEDS if seeds is None else seeds)
    core = HC.core_of(pair)
    registered = n_draws == REGISTERED["n_draws"] and seeds == REGISTERED["seeds"] and layers is None
    final_path = (RESULTS / f"refsize_d382_p1_{core}.json").resolve()
    out_path = Path(out).resolve() if out else final_path
    npz_path = out_path.with_name(out_path.stem + "_draws.npz")
    final = out_path == final_path
    if not final and out_path.is_relative_to(RESULTS.resolve()):
        refuse("a test run (--out) must not write under results/; only this job's own final name may")
    check_final_gates(final, registered)
    if npz_path.exists() and not out_path.exists():
        refuse(f"{npz_path} exists without {out_path.name}: the orphan of a failed write. Move it aside, never delete "
               f"it, to {npz_path.stem}.orphan-<utc>.npz, and rerun")
    for p in (out_path, npz_path):
        if p.exists() or p.is_symlink():
            refuse(f"{p} exists; results are never overwritten (D-152 P5)")
    code0 = P.code_block(CODE_FILES)
    b = p1_bind(pair, n_draws, seeds, final, records_only, code0)
    cfg, axis, C_sel, L_sel = b["cfg"], b["axis"], b["C_sel"], b["L_sel"]

    # the caches through the input registry (healthcheck_d156.main's lines)
    parts_in = ("probe_train", "threshold_calib", "eval")
    if records_only:
        print(f"RECORDS OK: p1 {pair} (records only; no cache read, nothing written)")
        return out_path
    data: dict[str, dict[str, Any]] = {"pre": {}, "post": {}}
    input_sha: dict[str, str] = {}
    input_recs: dict[str, dict[str, Any]] = {}
    for side in data:
        for part in parts_in:
            z, r = REG.load(cfg[f"{side}_repo"], axis, part)
            data[side][part] = z
            input_sha[f"{side}/{part}"], input_recs[f"{side}/{part}"] = r["sha256"], r
    reg_sources = REG.sources_sha256()
    if D.sources_now() != reg_sources:
        refuse("the extraction records or pins on disk are not the ones the registry read")
    if any((b["rec_doc"].get("input_cache_sha256") or {}).get(k) != h for k, h in input_sha.items()):
        refuse(f"{b['rec_rel']} read other caches than this run")
    verification = REG.verification(input_recs)
    for part in parts_in:
        a_, b_ = data["pre"][part], data["post"][part]
        if not np.array_equal(a_["item_ids"], b_["item_ids"]) or not np.array_equal(a_["labels"], b_["labels"]):
            refuse(f"{part} items or labels differ between pre and post")
        for zz in (a_, b_):
            if "nonfinite" in zz.files and zz["nonfinite"].any():
                refuse(f"{part} has non-finite items")
    grids = [sorted(int(k.split("_")[1]) for k in zz.files if k.startswith("layer_"))
             for s in data for zz in data[s].values()]
    if any(g != grids[0] for g in grids) or grids[0] != list((b["rec_doc"]["config"] or {}).get("layers") or []):
        refuse(f"the caches' layer grids differ from each other or from {b['rec_rel']}'s")
    if layers is not None and (not layers or set(layers) - set(grids[0])):
        refuse(f"layers {layers} are not a subset of the grid {grids[0]}")
    use_layers = [li for li in grids[0] if layers is None or li in layers]

    y_tr = data["pre"]["probe_train"]["labels"].astype(bool)
    y_cal = data["pre"]["threshold_calib"]["labels"].astype(bool)
    y_ev = data["pre"]["eval"]["labels"].astype(bool)
    ncal, neg, pos = ~y_cal, ~y_ev, y_ev
    pooled_ids = [str(i) for i in data["pre"]["threshold_calib"]["item_ids"][ncal]] + \
                 [str(i) for i in data["pre"]["eval"]["item_ids"][neg]]
    n_cal, n_pool = int(ncal.sum()), len(pooled_ids)
    cl = json.loads(b["cl_bytes"].decode("utf-8"))["axes"][axis]
    if cl["n_pooled_negatives"] != n_pool or cl["n_calibration"] != n_cal or \
            cl["pool_item_ids_sha256"] != sha256_bytes("\n".join(pooled_ids).encode("utf-8")):
        refuse(f"manifests/negative_clusters.json does not describe the {axis} pool")
    if {str(i) for i in data["pre"]["probe_train"]["item_ids"]} & set(pooled_ids):
        refuse("a probe_train item is in the pooled negatives (F-045)")
    ns = ns_for(n_cal)
    check_ns(ns, n_cal)
    check_anchor_size(ns, n_cal)
    draws = P.repartition_draws(n_cal, n_pool - n_cal, n_draws, P.AXIS_DRAW_SEED[axis])
    parts = [(np.arange(n_cal), np.arange(n_cal, n_pool))] + [(d[:n_cal], d[n_cal:]) for d in draws]
    perm = ref_permutations(n_cal, n_draws, axis_perm_seed(axis))
    print(f"D-382 p1 {pair} ({cfg['pre_repo']} -> {cfg['post_repo']}, {axis}): layers {use_layers}; selected C "
          f"{C_sel:g}; n_cal {n_cal}; reference sizes {ns}; shifts {len(SHIFTS)}; anchor {b['npz_rel']}; registered "
          f"{registered}", flush=True)

    cells: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    confusion: dict[str, np.ndarray] = {}
    n_anchor = {"arrays": 0, "operating_points": 0}
    for li in use_layers:
        k = f"layer_{li}"
        X = {s: {p_: data[s][p_][k] for p_ in parts_in} for s in data}
        probes = []
        for seed in seeds:
            _, fit = P.d117_split(len(y_tr), seed)
            probes.append(P.fit_probe(X["pre"]["probe_train"][fit], y_tr[fit], seed, C_sel, "d117_permutation"))
        S = (lambda x: np.column_stack([q.score(x) for q in probes]))  # noqa: E731
        pre_pool = np.vstack([S(X["pre"]["threshold_calib"][ncal]), S(X["pre"]["eval"][neg])])
        post_pool = np.vstack([S(X["post"]["threshold_calib"][ncal]), S(X["post"]["eval"][neg])])
        pos_pre, pos_post = S(X["pre"]["eval"][pos]), S(X["post"]["eval"][pos])
        v, sd = partition_arrays(pre_pool, post_pool, parts, perm, ns)
        # ANCHOR: the realised report is paired.repartition_estimate's (4c's line), and the n = |cal_d| unshifted
        # arrays are the 4c npz's byte for byte
        report, arr = P.repartition_estimate(pre_pool[:n_cal], pre_pool[n_cal:], post_pool[:n_cal], post_pool[n_cal:],
                                             pos_pre, pos_post, draws, TARGET)
        pr = np.concatenate([[report["paired_ratio"]["frozen"]], arr["paired_ratio"]]).astype("<f8")
        if not same_arrays(v[f"none|{REALISED}"], pr):
            refuse(f"the anchor: L{li}: the partitions are not paired.repartition_estimate's; nothing written")
        n_anchor["operating_points"] += 1
        base = f"L{li}|{PSET}|{LEVEL}"
        pairs_of_keys = [(f"n{n_cal}|none|{q}", f"{base}|{q}") for q in HEALTH]
        pairs_of_keys += [(f"none|{REALISED}", f"{base}|{REALISED}")]
        n_anchor["arrays"] += anchor_arrays(v, b["rec_z"], pairs_of_keys, f"{b['npz_rel']} (n = {n_cal}, no shift)")
        cells.append({"layer": li, "probe_set": PSET, "probe_C": C_sel, "is_selected_layer": li == L_sel,
                      "input_verification": verification, "realised_operating_point": report,
                      LEVEL: cell_block(v, ns, sd)})
        for kk, x in v.items():
            arrays[f"L{li}|{kk}"] = x
        arrays[f"L{li}|sd_pre_per_seed"] = sd
        add_counts(confusion, v, ns)
        print(f"  layer {li:>2} done ({len(ns)} reference sizes x {len(SHIFTS)} shifts)", flush=True)

    if n_anchor["arrays"] != len(use_layers) * (len(HEALTH) + 1) or n_anchor["operating_points"] != len(use_layers):
        refuse(f"to write: {n_anchor} anchored over {len(use_layers)} layers")
    counts_block = finish_counts(arrays, confusion, ns)
    for n in ns:
        arrays[f"ref_rows_frozen|n{n}"] = ref_rows(perm[0], n).astype(np.int64)
        arrays[f"ref_selection_frozen|n{n}"] = ref_selection(perm[0], n).astype(np.int64)
    npz_bytes = savez_bytes(arrays)
    code1 = P.code_block(CODE_FILES)
    changed = [w for w, x, y in (("code", code1["sha256"], code0["sha256"]),
                                 ("registry sources", D.sources_now(), reg_sources),
                                 ("cluster manifest", sha256_file(D.CLUSTERS), b["cl_sha"]),
                                 (b["npz_rel"], sha256_file(ROOT / b["npz_rel"]), b["npz_sha"])) if x != y]
    changed += [r_ for r_, h in _RESULT_READS.items() if sha256_file(ROOT / r_) != h]
    changed += [m for m, h in b["manifests"].items() if sha256_file(ROOT / m) != h]
    changed += [f"cache {r['file']}" for r in input_recs.values()
                if sha256_file(resolve_cache(r["file"], REG.CACHE)) != r["sha256"]]
    if changed:
        refuse(f"to write: changed during the run: {changed}")
    config = {"line": "beside (D-382 (1): the nine Phase 1 released pairs)", "population": "phase1_released",
              "pair": pair, "core": core, "registry_pair": b["reg_stem"], "axis": axis,
              "selection_file": rel(b["sel_path"]), "selection_pair": b["sel_stem"], "selection_sha256": b["sel_sha"],
              "selected_C": C_sel, "selected_layer": L_sel, "probe_set": PSET, "layers": use_layers, "seeds": seeds,
              "target_fpr": TARGET, "level": LEVEL, "n_draws": n_draws, "draw_seed": P.AXIS_DRAW_SEED[axis],
              "seeds_block": seeds_block(axis, P.AXIS_DRAW_SEED[axis]),
              "reference_permutation_seed": axis_perm_seed(axis),
              "reference_permutations_sha256": sha256_bytes(np.ascontiguousarray(perm).tobytes()),
              "n_grid": ns, "n_grid_registered": list(N_GRID), "c_grid": list(C_GRID), "shifts": list(SHIFTS),
              "n_calibration_negatives": n_cal, "n_pooled_negatives": n_pool,
              "cluster_manifest": {"file": rel(D.CLUSTERS), "sha256": b["cl_sha"]},
              "classes": {c: list(cut) for c, cut in CLASSES.items()}, "detectors": list(DETECTORS),
              "pre_repo": cfg["pre_repo"], "pre_rev": cfg["pre_rev"], "post_repo": cfg["post_repo"],
              "post_rev": cfg["post_rev"], "inputs_from": "src/input_registry.py (D-165)",
              "input_registry_sources_sha256": reg_sources, "blas_threads": blas_threads()}
    flags = [f"D-170 ({w}: {v!r}); it stands until D-168 is final (D-172)" for w, v in
             (("this run's own input verification", (verification or {}).get("flag")),
              (f"the pair's 4c record of record {b['rec_rel']}",
               (b["rec_doc"].get("input_verification") or {}).get("flag")))
             if v not in (None, "None", "")]
    payload = {**provenance_block(RUN_KIND["p1"]), "preregistration": PREREG, "code": code0,
               "reused_files_sha256": reused_now(), "registered_parameters": registered,
               "status": status_block(registered, flags,
                                      note="BESIDE (D-382 (1), (5)): no verdict of this item is read off a Phase 1 row"),
               "elapsed_s": round(time.time() - t0, 1), "implementation_choices": CHOICES,
               "predictions_record": b["predictions"], "input_verification": verification,
               "config": config, "config_hash": canonical_hash(config),
               "anchor": {"kind": "4c npz at n = |cal_d|, no shift (D-382 (1))", "identical": True, "counts": n_anchor,
                          "rule": "every healthcheck_d156.health statistic and the realised paired ratio at n = |cal_d| "
                                  "and shift 'none' equals the pair's 4c npz byte for byte at every layer, and the "
                                  "realised report is paired.repartition_estimate's on the same partitions",
                          "records": {"healthcheck_4c": {"file": b["rec_rel"], "sha256": b["rec_sha"],
                                                         "d170_flag": (b["rec_doc"].get("input_verification")
                                                                       or {}).get("flag")},
                                      "healthcheck_4c_npz": {"file": b["npz_rel"], "sha256": b["npz_sha"]},
                                      "manifests": b["manifests"]}},
               "population_group": {"declared_in": None,
                                    "note": "outside every src/populations.py group until the pull; declared as "
                                            "REFSIZE_D382 then (validate() requires every declared file on disk)"},
               "input_files_sha256": {**dict(_RESULT_READS), b["npz_rel"]: b["npz_sha"],
                                      rel(D.CLUSTERS): b["cl_sha"], **b["manifests"]},
               "input_cache_sha256": input_sha,
               "input_cache_source": {kk: {"file": r["file"], "expected_from": r["expected_from"],
                                           "pin_only": r["pin_only"]} for kk, r in input_recs.items()},
               "draws_npz": {"file": rel(npz_path), "sha256": sha256_bytes(npz_bytes), "size": len(npz_bytes),
                             "keys": NPZ_KEYS},
               "confusion_counts_pooled_over_cells": counts_block, "cells": cells}
    problems = input_identity_problems(payload)
    if problems:
        refuse(f"to write (D-297 (1) results-writer rule): {problems}")
    write_outputs([(npz_path, npz_bytes), (out_path, json_bytes_of(payload))])
    print(f"anchor identical: {n_anchor}")
    return out_path


NPZ_KEYS = ("L<layer>|n<n>|<shift>|<statistic> (healthcheck_d156.health's five, then the three at-the-floor shares "
            "over seeds of CHOICES (q): direct_floored_pre, direct_floored_post, direct_floored_either), "
            "L<layer>|<shift>|"
            "realised_paired_ratio, L<layer>|sd_pre_per_seed (n_parts x seeds), counts|n<n>|<shift or pooled_over_c>|"
            "<detector>|<class>|tp/fn/fp/tn summed over the pair's cells, ref_rows_frozen|n<n> and "
            "ref_selection_frozen|n<n>; index 0 of every per-draw array is the frozen split")


def savez_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def finish_counts(arrays: dict[str, np.ndarray], confusion: dict[str, np.ndarray], ns: list[int]) -> dict[str, Any]:
    """The per-draw confusion arrays into the npz and their D-156 means into the record; refuses a missing key."""
    want = expected_count_keys(ns)
    missing = [k for k in want if k not in confusion]
    if missing:
        refuse(f"the confusion counts are incomplete: {len(missing)} keys missing (first {missing[:3]})")
    out: dict[str, Any] = {}
    for k in want:
        x = confusion[k]
        arrays[f"counts|{k}"] = x
        out[k] = {"d156_mean": float(x[1:].mean()), "frozen_split": int(x[0])}
    return out


# ----------------------------------------------------------------------------- the Phase 2 job (primary)
def p2_bind(key: str, n_draws: int, final: bool, records_only: bool, code0: dict[str, Any]) -> dict[str, Any]:
    """Every record the Phase 2 job reads before any cache, by twomoment_p2_d361's own steps 1-2 (its functions called
    unchanged), plus the predictions record and the pair's D-361 record of record (CHOICES (n))."""
    if key not in P2_PAIRS:
        refuse(f"{key!r} is not one of the nine D-361 headline pairs: {P2_PAIRS}")
    pre_repo, post_repo, group, note = TP2.PAIRS[key]
    pooling = P2_POOLING
    threads = SP.check_threads()
    preds = None if records_only else passing_predictions(code0)
    # the D-361 anchor record: the Phase 2 path was proven on a Phase 1 pair by D-361's own code
    tp2_code = tp2_code_block(code0)
    anchor = None if records_only else TP2.passing_anchor(tp2_code, final=final)
    sr = T.read_selection(pre_repo, pooling, threads, tp2_code)
    arm, L_sel, seeds, holdout = sr["arm"], sr["layer"], list(sr["seeds"]), sr["holdout"]
    if not arm.startswith("C="):
        refuse(f"{sr['sel_rel']} selected {arm}; D-361 (2) names the selected C (a logistic arm)")
    if sr["target_fpr"] != TARGET or holdout != P.HOLDOUT:
        refuse(f"{sr['sel_rel']}: target {sr['target_fpr']!r}, holdout {holdout!r} (want {TARGET}, {P.HOLDOUT})")
    C_sel = float(arm.split("=")[1])
    v2 = T.read_v2(tp2_code)
    ev2 = F16.read_v2_eval(v2)
    sb = {"pre": T.read_subset(pre_repo, sr["rev"])}
    plans = {"pre_all_pre_entries": T.check_l_star(sb["pre"], sr, pooling),
             "pre_this_pair": TP2.check_pair_plan(sb["pre"], sr, pooling, [pre_repo, post_repo], "pre")}
    post_sel = TP2.durable_select(post_repo)
    bos = TP2.bos_check({"pre": (pre_repo, sr["rev"]), "post": (post_repo, post_sel["rev"])})
    sb["post"] = T.read_subset(post_repo, post_sel["rev"])
    plans["post_rest"] = TP2.check_pair_plan(sb["post"], sr, pooling, [pre_repo, post_repo], "post")
    if post_sel["kind"] == "p2sub":
        plans["post_select"] = TP2.check_pair_plan(post_sel, sr, pooling, [pre_repo, post_repo], "post")
    nl, g6 = TP2.plan_grid(sb["pre"])
    if TP2.plan_grid(sb["post"]) != (nl, g6):
        refuse(f"the two sides' layer plans name other layer counts or GRID6: {TP2.plan_grid(sb['post'])} vs {(nl, g6)}")
    required = sorted(set(g6) | {L_sel})
    index_pre = SP.ingest_index(sr["rec_path"])
    index_read = {rel(SP.INGEST_INDEX / f"{p.stem}.sha256"): SP.INGEST_INDEX / f"{p.stem}.sha256"
                  for p in (sr["rec_path"], sb["pre"]["path"], sb["post"]["path"], post_sel["path"])}
    index_sha0 = {k: sha256_file(p) for k, p in index_read.items()}
    entries: dict[str, dict[str, Any]] = {}
    s_sub: dict[str, dict[str, dict[str, Any]]] = {}
    for part in ("probe_train", "calibration"):
        e = (sr["rec"].get("splits") or {}).get(f"{SP.AXIS}/{part}/{pooling}")
        if not isinstance(e, dict):
            refuse(f"{sr['rec_rel']} has no split {SP.AXIS}/{part}/{pooling}")
        entries[f"pre/{SP.AXIS}/{part}/{pooling}"] = e
    for side in ("pre", "post"):
        s_sub[side] = {part: T.subset_entry(sb[side], part, pooling, L_sel) for part in ("eval", "pool4d")}
        for part in ("eval", "pool4d"):
            entries[f"{side}/{T.SUB_TAG}/{SP.AXIS}/{part}/{pooling}"] = s_sub[side][part]
    if post_sel["kind"] == "p2tpl":
        e = (post_sel["rec"].get("splits") or {}).get(f"{SP.AXIS}/calibration/{pooling}")
        if not isinstance(e, dict):
            refuse(f"{post_sel['rel']} has no split {SP.AXIS}/calibration/{pooling}")
        entries[f"post/{SP.AXIS}/calibration/{pooling}"] = e
        post_cal_layers = list(post_sel["rec"].get("layers_resolved") or [])
        post_cal_entry, post_cal_key = None, f"post/{SP.AXIS}/calibration/{pooling}"
    else:
        sel_sb = {**post_sel, "rest": post_sel["source"]["rec"], "rest_rel": post_sel["source"]["rel"]}
        e = T.subset_entry(sel_sb, "calibration", pooling, L_sel)
        post_cal_key = f"post/{T.SUB_TAG}/{SP.AXIS}/calibration/{pooling}"
        entries[post_cal_key] = e
        post_cal_layers = list(e.get("layers_resolved") or [])
        post_cal_entry = e
    v1_keyed = F16.check_v1_keyed(entries, v2)
    held = set(sr["rec"].get("layers_resolved") or []) & set(post_cal_layers)
    for side in ("pre", "post"):
        for part in ("eval", "pool4d"):
            held &= set(s_sub[side][part].get("layers_resolved") or [])
    if not set(required) <= held:
        refuse(f"the layers both sides hold {sorted(held)} do not include GRID6 + L* {required} (missing "
               f"{sorted(set(required) - held)}; D-361 (2), D-325 (2), {choice('h')})")
    use_layers = sorted(held)
    beyond = sorted(held - set(required))
    neg_ids = [i for i, lab in zip(ev2["ids"], ev2["labels"]) if lab == 0]
    pool_ids = list(v2["ids"]) + neg_ids
    n_cal, n_ev, n_pos = v2["n"], ev2["n_neg"], ev2["n_pos"]
    if final and (n_cal, n_ev, n_pos) != TP2.V2_SIZES:
        refuse(f"split v2 holds {(n_cal, n_ev, n_pos)}, not D-361 (2)'s {TP2.V2_SIZES}")
    unit, units_block = TP2.p2_cluster_units(pool_ids, n_cal, v2)
    # the D-361 record of record for this pair (CHOICES (n)), bound through the list's manifest
    d361_rel = f"results/twomoment_p2_d361_{key}_{pooling}.json"
    listed, man_sha = read_manifest_map(LOGS / Path(MANIFEST_P2MECH).name)
    d361, d361_sha = read_bound(d361_rel, listed, MANIFEST_P2MECH)
    dcfg = d361.get("config") or {}
    if d361.get("run_kind") != TP2.RUN_KIND["pair"] or (final and d361.get("registered_parameters") is not True):
        refuse(f"{d361_rel} is not a registered D-361 pair record")
    if dcfg.get("n_draws") != n_draws:
        refuse(f"{d361_rel} ran {dcfg.get('n_draws')} draws, this run {n_draws}: a draw-count mismatch")
    if (dcfg.get("selected_C"), dcfg.get("selected_layer"), dcfg.get("pooling")) != (C_sel, L_sel, pooling) or \
            list(dcfg.get("layers") or []) != use_layers or list(dcfg.get("seeds") or []) != seeds:
        refuse(f"{d361_rel} ran another cell set: C {dcfg.get('selected_C')}, L* {dcfg.get('selected_layer')}, layers "
               f"{dcfg.get('layers')}, seeds {dcfg.get('seeds')}")
    d361_cells = {int(c["layer"]): c for c in d361.get("cells") or []}
    if sorted(d361_cells) != use_layers:
        refuse(f"{d361_rel} holds cells for {sorted(d361_cells)}, this run's layers are {use_layers}")
    pinned_keys: tuple[str, ...] = (("results/phase2/float16_check_d364.json",)
                                    if (key, pooling) == TP2.FLOAT16_CELL else ())
    if final:
        registered = n_draws == REGISTERED["n_draws"] and seeds == REGISTERED["seeds"]
        bad = final_problems(registered) + TP2.final_problems(registered, pinned_keys)
        have = {"v2/calibration": v2["sha256"], "v2/eval": ev2["sha256"]}
        bad += [f"{k} is not D-362's ({have[k][:16]})" for k in have if have[k] != TP2.PINNED[k]]
        if bad:
            refuse(f"the final names need: {bad}")
    return {"pre_repo": pre_repo, "post_repo": post_repo, "group": group, "note": note, "pooling": pooling,
            "threads": threads, "predictions": preds, "anchor": anchor, "tp2_code": tp2_code, "sr": sr, "arm": arm,
            "C_sel": C_sel, "L_sel": L_sel, "seeds": seeds, "holdout": holdout, "v2": v2, "ev2": ev2, "sb": sb,
            "plans": plans, "post_sel": post_sel, "bos": bos, "nl": nl, "g6": g6, "required": required,
            "index_pre": index_pre, "index_read": index_read, "index_sha0": index_sha0, "entries": entries,
            "s_sub": s_sub, "post_cal_key": post_cal_key, "post_cal_entry": post_cal_entry, "v1_keyed": v1_keyed,
            "layers": use_layers, "beyond": beyond, "pool_ids": pool_ids, "n_cal": n_cal, "n_ev": n_ev,
            "n_pos": n_pos, "unit": unit, "units_block": units_block, "d361_rel": d361_rel, "d361": d361,
            "d361_sha": d361_sha, "d361_cells": d361_cells, "manifests": {MANIFEST_P2MECH: man_sha},
            "pinned_keys": pinned_keys}


def run_p2(key: str, n_draws: int = P.N_DRAWS, out_dir: str | Path | None = None,
           records_only: bool = False) -> Path:
    """One Phase 2 headline pair (PRIMARY, D-382 (1)): n = 200, 500, 1000, 5000 at every layer both sides hold, user-span
    mean, the selected C of record, item level, 1%; the reviewed Phase 2 scoring of twomoment_p2_d361, its anchor record
    and the pair's D-361 record of record."""
    t0 = time.time()
    final_dir = RESULTS.resolve()
    odir = final_dir if out_dir is None else Path(out_dir).resolve()
    final = odir == final_dir
    out_path = odir / f"refsize_d382_p2_{key}.json"
    npz_path = out_path.with_name(out_path.stem + "_draws.npz")
    if not final and odir.is_relative_to(RESULTS.resolve()):
        refuse("a test run (--out-dir) must not write under results/; only the final names may")
    check_final_gates(final, n_draws == REGISTERED["n_draws"])   # the seeds are the selection's; p2_bind re-checks
    if npz_path.exists() and not out_path.exists():
        refuse(f"{npz_path} exists without {out_path.name}: the orphan of a failed write. Move it aside, never delete "
               f"it, to {npz_path.stem}.orphan-<utc>.npz, and rerun")
    for p in (out_path, npz_path):
        if p.exists() or p.is_symlink():
            refuse(f"{p} exists; results are never overwritten (D-152 P5)")
    code0 = P.code_block(CODE_FILES)
    b = p2_bind(key, n_draws, final, records_only, code0)
    if records_only:
        print(f"RECORDS OK: p2 {key} (records only; no cache read, nothing written)")
        return out_path
    pooling, sr, seeds, C_sel, L_sel = b["pooling"], b["sr"], b["seeds"], b["C_sel"], b["L_sel"]
    v2, ev2, sb, post_sel, layers = b["v2"], b["ev2"], b["sb"], b["post_sel"], b["layers"]
    n_cal, n_ev, n_pos = b["n_cal"], b["n_ev"], b["n_pos"]
    registered = n_draws == REGISTERED["n_draws"] and seeds == REGISTERED["seeds"]

    # the durable caches, by twomoment_p2_d361's step 3 (its loaders called unchanged)
    z_tr, r_tr, _ = SP.load_ingested(sr["rec"], sr["rec_rel"], b["index_pre"][1], "probe_train", pooling)
    z_cal: dict[str, Any] = {}
    r_cal: dict[str, Any] = {}
    z_cal["pre"], r_cal["pre"], _ = SP.load_ingested(sr["rec"], sr["rec_rel"], b["index_pre"][1], "calibration", pooling)
    recorded = sr["sel"].get("input_cache_sha256") if isinstance(sr["sel"].get("input_cache_sha256"), dict) else {}
    for part, r in (("probe_train", r_tr), ("calibration", r_cal["pre"])):
        k_ = f"pre/{SP.AXIS}/{part}/{pooling}"
        if recorded.get(k_) != r["sha256"]:
            refuse(f"{r['file']} ({r['sha256'][:16]}) is not the cache {sr['sel_rel']} read for {k_}")
    if post_sel["kind"] == "p2tpl":
        z_cal["post"], r_cal["post"], _ = SP.load_ingested(post_sel["rec"], post_sel["rel"], post_sel["index"],
                                                           "calibration", pooling)
    else:
        z_cal["post"], _, _, r_cal["post"] = TP2.open_sub_cache(post_sel, b["post_cal_entry"], "calibration", pooling,
                                                                layers)
    z_sub: dict[str, dict[str, Any]] = {}
    r_sub: dict[str, dict[str, dict[str, Any]]] = {}
    meta_sub: dict[str, dict[str, tuple[list[str], list[int]]]] = {}
    for side in ("pre", "post"):
        z_sub[side], r_sub[side], meta_sub[side] = {}, {}, {}
        for part in ("eval", "pool4d"):
            zz, ids, labs, r = TP2.open_sub_cache(sb[side], b["s_sub"][side][part], part, pooling, layers)
            z_sub[side][part], r_sub[side][part], meta_sub[side][part] = zz, r, (ids, labs)
    cal_meta = {side: ([str(x) for x in z_cal[side]["item_ids"]], [int(x) for x in z_cal[side]["labels"]])
                for side in ("pre", "post")}
    if cal_meta["pre"] != cal_meta["post"] or any(meta_sub["pre"][p_] != meta_sub["post"][p_]
                                                  for p_ in ("eval", "pool4d")):
        refuse("v1 calibration, eval or pool4d: the pre and post caches hold other items or labels")
    y_tr = z_tr["labels"].astype(bool)
    if sr["sel"]["model"].get("n_probe_train") != len(y_tr):
        refuse(f"probe_train holds {len(y_tr)} items, {sr['sel_rel']} says {sr['sel']['model'].get('n_probe_train')}")
    y_cal_v1 = np.asarray(cal_meta["pre"][1]) == 1
    if y_cal_v1.all():
        refuse("v1 calibration holds no negative")

    draws = {LEVEL: P.repartition_draws(n_cal, n_ev, n_draws, TP2.P2_DRAW_SEED)}
    parts = TP2.make_parts(draws, n_cal, n_cal + n_ev)[LEVEL]
    ns = ns_for(n_cal)
    check_ns(ns, n_cal)
    check_anchor_size(ns, n_cal)
    perm = ref_permutations(n_cal, n_draws, axis_perm_seed(SP.AXIS))
    f16 = None
    if (key, pooling) == TP2.FLOAT16_CELL:
        p16 = SP.P2 / F16.OUT_NAME
        if p16.is_symlink() or not p16.is_file():
            refuse(f"no {rel(p16)}: the cross-check of twomoment_p2_d361 CHOICES (o) needs D-364's record")
        f16 = read_result(p16)
    print(f"D-382 p2 {key} [{SP.AXIS}, {pooling}]: {b['pre_repo']} -> {b['post_repo']} ({b['group']}); {b['arm']} at "
          f"L* {L_sel}; layers {layers}; v2 {n_cal} + {n_ev} negatives, {n_pos} positives; reference sizes {ns}; "
          f"shifts {len(SHIFTS)}; anchor {b['anchor']['file']}; registered {registered}", flush=True)

    cells: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    confusion: dict[str, np.ndarray] = {}
    n_anchor = {"operating_points": 0, "paired_ratio_arrays": 0, "probe_blocks": 0}
    probe_anchor: dict[str, Any] = {}
    cross: dict[str, Any] = {"applies": False, "cell": list(TP2.FLOAT16_CELL)}
    gathered: dict[str, Any] = {}
    for li in [L_sel] + [x for x in layers if x != L_sel]:
        k = f"layer_{li}"
        probes = TP2.fit_probes(z_tr[k], y_tr, seeds, C_sel)
        if li == L_sel:                                       # twomoment_p2_d361's step 4 (its CHOICES (b))
            sel_probes = T.seed_probes(z_tr[k], y_tr, b["arm"], seeds, b["holdout"])
            off = [q.diag.get("C") for q, s in zip(probes, sel_probes)
                   if not (q.mu.dtype == s["mu"].dtype and q.mu.tobytes() == s["mu"].tobytes()
                           and q.sd.dtype == s["sd"].dtype and q.sd.tobytes() == s["sd"].tobytes()
                           and np.asarray(q.w).dtype == np.asarray(s["w"]).dtype
                           and np.asarray(q.w).tobytes() == np.asarray(s["w"]).tobytes() and q.b == s["b"])]
            if off or len(sel_probes) != len(probes):
                raise SystemExit(f"ANCHOR FAILED: {len(off)} of {len(probes)} probes are not the selection's own fit "
                                 "bit for bit (selection_tpr_v2.seed_probes); nothing written")
            tprs = [r["tpr"] for r in T.tprs_at(sel_probes, z_cal["pre"][k], ~y_cal_v1, sr["target_fpr"])]
            diffs, n_cmp = T.anchor_diffs(tprs, sr["selected"])
            if diffs:
                raise SystemExit(f"ANCHOR FAILED: {len(diffs)} of {n_cmp} values are not equal to {sr['sel_rel']}; "
                                 "nothing written")
            probe_anchor = {"selection_record": {"file": sr["sel_rel"], "sha256": _RESULT_READS[sr["sel_rel"]]},
                            "layer": li, "probes_equal_to_selection_fit_bit_for_bit": True,
                            "ho20_tpr_per_seed_recorded": sr["selected"]["ho20_tpr_per_seed"],
                            "ho20_tpr_per_seed_recomputed": tprs, "values_compared": n_cmp, "values_differing": 0,
                            "exact_equal": True}
            n_anchor["probe_blocks"] += 1
            print(f"  ANCHOR OK at L* {li}: probes bit for bit the selection's; {n_cmp} of {n_cmp} ho20 values equal",
                  flush=True)
        mats: dict[str, dict[str, np.ndarray]] = {}
        for side in ("pre", "post"):
            sources = {"calibration": (z_cal[side][k], *cal_meta[side]),
                       "eval": (z_sub[side]["eval"][k], *meta_sub[side]["eval"]),
                       "pool4d": (z_sub[side]["pool4d"][k], *meta_sub[side]["pool4d"])}
            Xc, Xc_pos, cc, _ = F16.gather_split("calibration", v2["ids"], [0] * v2["n"], sources)
            Xn, Xp, cn, cp = F16.gather_split("eval", ev2["ids"], ev2["labels"], sources)
            if Xc_pos.shape[0] or cc != v2["plan_counts"] or cn != ev2["expect_neg"] or cp != ev2["expect_pos"]:
                refuse(f"{side} layer {li}: v2 items per v1 split (calibration {cc}, eval negatives {cn}, eval "
                       f"positives {cp}) are not the v2 record's plan")
            mats[side] = {"cal": Xc, "evn": Xn, "pos": Xp}
            gathered[side] = {"calibration": cc, "eval_negatives": cn, "eval_positives": cp}
            del sources
        sc = TP2.score_matrices(probes, {"cal_pre": mats["pre"]["cal"], "ev_pre": mats["pre"]["evn"],
                                         "cal_post": mats["post"]["cal"], "ev_post": mats["post"]["evn"],
                                         "pos_pre": mats["pre"]["pos"], "pos_post": mats["post"]["pos"]})
        del mats
        pool_pre = np.vstack([sc["cal_pre"], sc["ev_pre"]])
        pool_post = np.vstack([sc["cal_post"], sc["ev_post"]])
        v, sd = partition_arrays(pool_pre, pool_post, parts, perm, ns)
        report, arr = P.repartition_estimate(sc["cal_pre"], sc["ev_pre"], sc["cal_post"], sc["ev_post"], sc["pos_pre"],
                                             sc["pos_post"], draws[LEVEL], TARGET)
        want = (b["d361_cells"][li].get("realised"))
        if canon(report) != canon(want):
            refuse(f"the anchor: L{li}: the realised operating point at 1% is not {b['d361_rel']}'s; nothing written")
        n_anchor["operating_points"] += 1
        pr = np.concatenate([[report["paired_ratio"]["frozen"]], arr["paired_ratio"]]).astype("<f8")
        if not same_arrays(v[f"none|{REALISED}"], pr):
            refuse(f"the anchor: L{li}: the partitions are not paired.repartition_estimate's; nothing written")
        n_anchor["paired_ratio_arrays"] += 1
        if f16 is not None and li == L_sel:
            mine = {"pre_repo": b["pre_repo"], "post_repo": b["post_repo"], "pooling": pooling, "layer": L_sel,
                    "arm": b["arm"], "seeds": list(seeds), "n_draws": n_draws, "draw_seed": TP2.P2_DRAW_SEED,
                    "pooled_item_ids_sha256": sha256_bytes("\n".join(b["pool_ids"]).encode("utf-8")),
                    "n_eval_positives": n_pos}
            d = TP2.float16_cross_check(f16, probes, list(seeds), sc, {TARGET: report}, mine)
            if d:
                refuse(f"the D-364 cell differs from {F16.OUT_NAME}'s arm A: {d}")
            cross = {"applies": True, "cell": list(TP2.FLOAT16_CELL), "file": rel(SP.P2 / F16.OUT_NAME),
                     "sha256": _RESULT_READS[rel(SP.P2 / F16.OUT_NAME)], "identical": True,
                     "compared": "twomoment_p2_d361.float16_cross_check at 1% (its CHOICES (o))"}
            print("  FLOAT16 CROSS-CHECK OK: probes, scores and the realised report equal D-364's arm A", flush=True)
        cells.append({"layer": li, "probe_set": PSET, "probe_C": C_sel, "is_selected_layer": li == L_sel,
                      "input_verification": TP2.D170_P2, "realised_operating_point": report,
                      LEVEL: cell_block(v, ns, sd)})
        for kk, x in v.items():
            arrays[f"L{li}|{kk}"] = x
        arrays[f"L{li}|sd_pre_per_seed"] = sd
        add_counts(confusion, v, ns)
        del sc, pool_pre, pool_post
        print(f"  layer {li:>2} done ({len(ns)} reference sizes x {len(SHIFTS)} shifts)", flush=True)
    for zz in (z_tr, *z_cal.values(), *(z for d in z_sub.values() for z in d.values())):
        zz.close()
    if n_anchor["operating_points"] != len(layers) or n_anchor["paired_ratio_arrays"] != len(layers) or \
            n_anchor["probe_blocks"] != 1:
        refuse(f"to write: {n_anchor} anchored over {len(layers)} layers")
    cells.sort(key=lambda c: c["layer"])
    counts_block = finish_counts(arrays, confusion, ns)
    for n in ns:
        arrays[f"ref_rows_frozen|n{n}"] = ref_rows(perm[0], n).astype(np.int64)
        arrays[f"ref_selection_frozen|n{n}"] = ref_selection(perm[0], n).astype(np.int64)
    npz_bytes = savez_bytes(arrays)
    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"]:
        refuse("source changed during the run")
    caches = {f"pre/{SP.AXIS}/probe_train/{pooling}": r_tr, f"pre/{SP.AXIS}/calibration/{pooling}": r_cal["pre"],
              b["post_cal_key"]: r_cal["post"]}
    for side in ("pre", "post"):
        for part in ("eval", "pool4d"):
            caches[f"{side}/{T.SUB_TAG}/{SP.AXIS}/{part}/{pooling}"] = r_sub[side][part]
    files_sha = {T.V2_CAL_FILE: sha256_file(SP.SPLIT_ROOT / T.V2_CAL_FILE),
                 F16.V2_EVAL_FILE: sha256_file(SP.SPLIT_ROOT / F16.V2_EVAL_FILE)}
    files_sha.update({k: sha256_file(p) for k, p in b["index_read"].items()})
    moved = [f"{k} (the v2 record names {w[:16]})" for k, w in ((T.V2_CAL_FILE, v2["sha256"]),
                                                                (F16.V2_EVAL_FILE, ev2["sha256"]))
             if files_sha[k] != w]
    moved += [k for k, h in b["index_sha0"].items() if files_sha[k] != h]
    moved += TP2.end_hash_problems({f"data/hidden/{r['file']}": (SP.CACHE / r["file"], r["sha256"])
                                    for r in caches.values()})
    moved += [m for m, h in b["manifests"].items() if sha256_file(ROOT / m) != h]
    if final:
        moved += TP2.end_hash_problems({k: (TP2.resolve(k), h) for k, h in _RESULT_READS.items()})
    if moved:
        refuse(f"changed during the run: {moved}")
    files_sha.update({k: h for k, h in _RESULT_READS.items()})
    config = {"line": "PRIMARY (D-382 (1): the nine Phase 2 headline pairs of D-361 (2))",
              "population": "phase2_headline", "pair": key, "pair_group": b["group"], "pair_note": b["note"],
              "pre_repo": b["pre_repo"], "pre_rev": sr["rev"], "post_repo": b["post_repo"],
              "post_rev": post_sel["rev"], "pooling": pooling, "axis": SP.AXIS, "split_version": 2,
              "selection_record": {"file": sr["sel_rel"], "sha256": _RESULT_READS[sr["sel_rel"]]},
              "selected_arm": b["arm"], "selected_C": C_sel, "selected_layer": L_sel, "probe_set": PSET,
              "n_model_layers": b["nl"], "grid6": b["g6"], "grid6_plus_pre_lstar": b["required"], "layers": layers,
              "layers_beyond_grid6_plus_pre_lstar": b["beyond"], "layer_rule": choice("h"),
              "seeds": seeds, "holdout": b["holdout"], "target_fpr": TARGET, "level": LEVEL, "n_draws": n_draws,
              "draw_seed": TP2.P2_DRAW_SEED, "seeds_block": seeds_block(SP.AXIS, TP2.P2_DRAW_SEED),
              "reference_permutation_seed": axis_perm_seed(SP.AXIS),
              "reference_permutations_sha256": sha256_bytes(np.ascontiguousarray(perm).tobytes()),
              "n_grid": ns, "n_grid_registered": list(N_GRID), "c_grid": list(C_GRID), "shifts": list(SHIFTS),
              "n_eval_positives": n_pos,
              "pooled_negatives": {"n_calibration": n_cal, "n_evaluation": n_ev,
                                   "item_ids_sha256": sha256_bytes("\n".join(b["pool_ids"]).encode("utf-8")),
                                   "order": "v2 calibration (its file's order), then v2 evaluation negatives"},
              "classes": {c: list(cut) for c, cut in CLASSES.items()}, "detectors": list(DETECTORS),
              "estimator": "D-156 as D-361 (2), through twomoment_p2_d361's reviewed path",
              "join_rule": v2["join_rule"], "post_select_set": post_sel["kind"], "blas_threads": b["threads"]}
    p2_flag = (TP2.D170_P2 or {}).get("flag")
    payload = {**provenance_block(RUN_KIND["p2"]), "preregistration": PREREG, "code": code0,
               "reused_files_sha256": reused_now(), "registered_parameters": registered,
               "status": status_block(
                   registered,
                   [] if p2_flag in (None, "None", "") else [f"D-170 (twomoment_p2_d361.D170_P2: {p2_flag!r})"],
                   note="PRIMARY (D-382 (1), (5)): the three verdicts are read off these rows. The 4c FLAGGED caveat of "
                        "D-255 carries over by design (the reference set and the evaluation part are one pool of "
                        "negatives re-partitioned per draw), as D-382 (2) registers; it is a property of the estimand, "
                        "not of an input, and is stated in the summary"),
               "elapsed_s": round(time.time() - t0, 1), "implementation_choices": CHOICES,
               "predictions_record": b["predictions"], "input_verification": TP2.D170_P2,
               "config": config, "config_hash": canonical_hash(config),
               "anchor": {"kind": "the D-361 anchor record, the selection's own probe fit, and the D-361 pair record's "
                                  "realised operating point (CHOICES (n))",
                          "identical": True, "counts": n_anchor,
                          "d361_anchor_record": b["anchor"],
                          "d361_pair_record": {"file": b["d361_rel"], "sha256": b["d361_sha"]},
                          "manifests": b["manifests"]},
               "probe_anchor": probe_anchor, "float16_cross_check": cross, "bos_check": b["bos"],
               "layer_plans": b["plans"], "cluster_units": b["units_block"],
               "v1_keyed_caches": {"rule": "D-320 (4)", "split_files": b["v1_keyed"]},
               "split_files": {"v2/calibration": {"file": T.V2_CAL_FILE, "sha256": v2["sha256"]},
                               "v2/eval": {"file": F16.V2_EVAL_FILE, "sha256": ev2["sha256"]}},
               "v2_gather": {"per_side_at_every_layer": gathered,
                             "expected_from_the_v2_plan": {"calibration": v2["plan_counts"],
                                                           "eval_negatives": ev2["expect_neg"],
                                                           "eval_positives": ev2["expect_pos"]}},
               "population_group": {"declared_in": None,
                                    "note": "outside every src/populations.py group until the pull; declared as "
                                            "REFSIZE_D382 then (validate() requires every declared file on disk)"},
               "inputs": dict(_RESULT_READS), "input_files_sha256": files_sha,
               "ingest_indexes_sha256": {"at_read": b["index_sha0"], "unchanged_at_end": True},
               "input_cache_sha256": {k: r["sha256"] for k, r in caches.items()},
               "input_cache_source": {k: f"data/hidden/{r['file']} (expected bytes from {r['expected_from']})"
                                      for k, r in caches.items()},
               "caches_rehashed_at_end": True,
               "draws_npz": {"file": rel(npz_path), "sha256": sha256_bytes(npz_bytes), "size": len(npz_bytes),
                             "keys": NPZ_KEYS},
               "confusion_counts_pooled_over_cells": counts_block, "cells": cells}
    problems = input_identity_problems(payload)
    if problems:
        refuse(f"to write (D-297 (1) results-writer rule): {problems}")
    write_outputs([(npz_path, npz_bytes), (out_path, json_bytes_of(payload))])
    print(f"anchor identical: {n_anchor}")
    return out_path


# ----------------------------------------------------------------------------- the predictions job (D-382 (5))
def run_predictions(out: str | Path | None = None) -> Path:
    """Writes the predictions record BEFORE any number of this item exists (D-382 (5), CHOICES (o)). It carries no
    quantity of the analysis: only the registered statements, their rules and the registered grids."""
    final_path = (RESULTS / Path(PRED_REL).name).resolve()
    out_path = Path(out).resolve() if out else final_path
    final = out_path == final_path
    if not final and out_path.is_relative_to(RESULTS.resolve()):
        refuse("a test run (--out) must not write under results/")
    check_final_gates(final, True)      # the predictions job registers the grids themselves; nothing else to register
    if out_path.exists() or out_path.is_symlink():
        refuse(f"{out_path} exists; results are never overwritten (D-152 P5)")
    code0 = P.code_block(CODE_FILES)
    config = {"preregistration": PREREG, "n_grid": list(N_GRID), "c_grid": list(C_GRID), "shifts": list(SHIFTS),
              "primary_population": "phase2_headline (the nine D-361 (2) headline pairs, user-span mean, selected C, "
                                    "item level, 1%)",
              "beside_population": "phase1_released (the nine Phase 1 released pairs at n = 200, 500, 1000)",
              "primary_class": PRIMARY_CLASS, "classes": {c: list(cut) for c, cut in CLASSES.items()},
              "detectors": list(DETECTORS), "level": LEVEL, "probe_set": PSET, "target_fpr": TARGET,
              "n_draws": REGISTERED["n_draws"], "seeds": REGISTERED["seeds"],
              "seeds_block": {"probe_seeds": list(SEEDS),
                              "reference_permutation_seed_by_axis": dict(REF_PERM_SEED),
                              "pair_cluster_bootstrap": {"n_boot": HS.B, "seed": HS.SEED}},
              "pooled_interval": f"pair-cluster bootstrap B={HS.B}, seed {HS.SEED} (D-177 (b))",
              "accuracy_line": "D-177 (3) on every predicted-against-realised table (CHOICES (j))"}
    payload = {**provenance_block(RUN_KIND["predictions"]), "preregistration": PREREG, "code": code0,
               "reused_files_sha256": reused_now(), "implementation_choices": CHOICES,
               "status": {"status": "NOT YET RUN", "d_entries": ["D-382 (5)"],
                          "rule": "the evidence standard's status of a record that carries no number: this file holds "
                                  "only the registered statements and their rules, so it is neither FINAL, FLAGGED nor "
                                  "PROVISIONAL; every verdict below is NOT YET RUN"},
               "written_before_any_number": True,
               "note": "D-382 (5): written before any number of this item exists. No quantity of the analysis appears "
                       "in this file; src/refsize_d382.py summary scores each prediction HELD or FAILED from its own "
                       "rows and binds this record by sha256.",
               "grids": {"n_grid": list(N_GRID), "c_grid": list(C_GRID), "shifts": list(SHIFTS)},
               "predictions": REGISTERED_PREDICTIONS,
               "verdicts": {k: "NOT YET RUN" for k in REGISTERED_PREDICTIONS},
               "beside": BESIDE_NOTE, "config": config, "config_hash": canonical_hash(config)}
    write_outputs([(out_path, json_bytes_of(payload))])
    return out_path


# ----------------------------------------------------------------------------- the summary
def roc_auc(stat: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """healthcheck_summary_d156.scoring's ROC lines, repeated (CHOICES (m)): per draw (column) the AUC of `stat` across
    the row's cells for the class `y`; index 0 is the frozen split."""
    a = np.array([M.auroc(stat[y[:, d], d], stat[~y[:, d], d]).value if 0 < y[:, d].sum() < y.shape[0] else np.nan
                  for d in range(y.shape[1])])
    return P.band_summary(a[1:], a[0])


def row_cells(z_of: dict[str, dict[str, np.ndarray]], names: list[str]) -> list[tuple[str, int]]:
    """(pair, layer) of every cell of a row, in the records' order."""
    out: list[tuple[str, int]] = []
    for nm in names:
        lays = sorted({int(k.split("|")[0][1:]) for k in z_of[nm] if re.fullmatch(r"L\d+\|none\|" + REALISED,
                                                                                 k) is not None})
        out += [(nm, li) for li in lays]
    return out


def row_block(z_of: dict[str, dict[str, np.ndarray]], names: list[str], n: int, tag: str) -> dict[str, Any]:
    """One (reference size, shift) row: 4c's pooled scoring, the ROC AUC lines and D-177's accuracy line."""
    k = len(names)
    out: dict[str, Any] = {"n_reference": n, "shift": tag, "c": SHIFT_C.get(tag), "pairs": names, "n_pairs": k,
                           "direct_floor": 1.0 / n}
    idx = HS.pair_cluster_index(k)
    out["interval"] = (f"pair-cluster bootstrap B={HS.B}, seed {HS.SEED}" if idx is not None
                       else f"point only (fewer than {HS.MIN_PAIRS} pairs)")
    for c in CLASSES:
        for dname in DETECTORS:
            out[f"{dname}|{c}"] = HS.pooled_counts(
                {q: [z_of[nm][f"counts|n{n}|{tag}|{dname}|{c}|{q}"] for nm in names] for q in COUNTS}, idx)
    if tag == POOLED_C:
        out["note"] = choice("k")
        return out
    cells = row_cells(z_of, names)
    out["n_cells"] = len(cells)
    rr = np.vstack([z_of[nm][f"L{li}|{tag}|{REALISED}"] for nm, li in cells])
    stats = {"abs_log_gaussian_ratio": np.abs(np.vstack([z_of[nm][f"L{li}|n{n}|{tag}|log_gaussian_ratio"]
                                                         for nm, li in cells])),
             "abs_log_direct_ratio": np.abs(np.log(np.vstack([z_of[nm][f"L{li}|n{n}|{tag}|direct_ratio"]
                                                              for nm, li in cells])))}
    for c, cut in CLASSES.items():
        y = HC.crosses(rr, cut)
        for sname, sv in stats.items():
            out[f"roc_auc|{sname}|{c}"] = roc_auc(sv, y)
    # D-177 (3), CHOICES (j): the Gaussian and the direct ratio ARE predictions of the realised paired ratio
    log_real = np.log(rr)
    acc: dict[str, Any] = {}
    for pname, logpred in (("gaussian_ratio", np.vstack([z_of[nm][f"L{li}|n{n}|{tag}|log_gaussian_ratio"]
                                                         for nm, li in cells])),
                           ("direct_ratio", np.log(np.vstack([z_of[nm][f"L{li}|n{n}|{tag}|direct_ratio"]
                                                              for nm, li in cells])))):
        x, y_, ncom = TF.d156_estimates(logpred, log_real)
        acc[pname] = {**TP2.accuracy_line(x, y_), "n_common_draws_min": int(ncom.min()) if ncom.size else None}
    out["accuracy_line"] = acc
    return out


def verdict_block(rows: dict[str, Any], ns: list[int]) -> dict[str, Any]:
    """D-382 (5): P-N.1, P-N.2 and P-N.3 scored HELD or FAILED (UNDEFINED when a quantity is undefined)."""
    word = {True: "HELD", False: "FAILED"}

    def roc(n: int, stat: str) -> float | None:
        v = rows.get(f"n{n}|none", {}).get(f"roc_auc|{stat}|{PRIMARY_CLASS}") or {}
        m = v.get("mean")
        return float(m) if isinstance(m, (int, float)) and math.isfinite(m) else None

    want = list(N_GRID)
    seq = {n: roc(n, "abs_log_direct_ratio") for n in want}
    missing = [n for n in want if n not in ns or seq[n] is None]
    steps = [] if missing else [seq[a] <= seq[b] for a, b in zip(want, want[1:])]
    p1: dict[str, Any] = {"verdict": "UNDEFINED" if missing else word[all(steps)],
                          "text": REGISTERED_PREDICTIONS["P-N.1"]["text"],
                          "rule": REGISTERED_PREDICTIONS["P-N.1"]["rule"],
                          "roc_auc_direct_by_n": {f"n{n}": seq[n] for n in want},
                          "non_decreasing_steps": steps,
                          "reason": (f"undefined at n {missing}" if missing else None)}
    g200, d200 = roc(200, "abs_log_gaussian_ratio"), roc(200, "abs_log_direct_ratio")
    p2: dict[str, Any] = {"verdict": "UNDEFINED" if (g200 is None or d200 is None) else word[g200 > d200],
                          "text": REGISTERED_PREDICTIONS["P-N.2"]["text"],
                          "rule": REGISTERED_PREDICTIONS["P-N.2"]["rule"],
                          "roc_auc_gaussian_n200": g200, "roc_auc_direct_n200": d200,
                          "reason": ("a quantity is undefined" if g200 is None or d200 is None else None)}

    def spec(n: int) -> float | None:
        v = ((rows.get(f"n{n}|{POOLED_C}", {}).get(f"direct|{PRIMARY_CLASS}") or {})
             .get("pooled_from_mean_counts") or {}).get("specificity")
        return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None

    s5, s2 = spec(5000), spec(200)
    p3: dict[str, Any] = {"verdict": "UNDEFINED" if (s5 is None or s2 is None) else word[s5 >= s2],
                          "text": REGISTERED_PREDICTIONS["P-N.3"]["text"],
                          "rule": REGISTERED_PREDICTIONS["P-N.3"]["rule"],
                          "pooled_specificity_direct_n5000": s5, "pooled_specificity_direct_n200": s2,
                          "reason": ("a quantity is undefined" if s5 is None or s2 is None else None)}
    return {"P-N.1": p1, "P-N.2": p2, "P-N.3": p3}


def summary_inputs() -> list[str]:
    """The 18 records and their npz the manifest job hashes, repo-relative."""
    names = [f"refsize_d382_p2_{k}" for k in P2_PAIRS] + [f"refsize_d382_p1_{HC.core_of(s)}" for s in P1_PAIRS]
    return [PRED_REL] + [f"results/{n}{s}" for n in names for s in (".json", "_draws.npz")]


def run_summary(out: str | Path | None = None, records_dir: str | Path | None = None,
                manifest: str | Path | None = None) -> Path:
    """The pooled scoring (D-382 (4), (5)) from the 18 records and their npz, each bound to its line of
    logs/refsize_d382.sha256; the predictions record bound and its text checked; nothing else is read."""
    t0 = time.time()
    final_path = (RESULTS / "refsize_d382_summary.json").resolve()
    out_path = Path(out).resolve() if out else final_path
    final = out_path == final_path
    if not final and out_path.is_relative_to(RESULTS.resolve()):
        refuse("a test run (--out) must not write under results/")
    check_final_gates(final, True)      # the summary's own parameters are the records'; each is re-checked below
    if out_path.exists() or out_path.is_symlink():
        refuse(f"{out_path} exists; results are never overwritten (D-152 P5)")
    code0 = P.code_block(CODE_FILES)
    rdir = Path(records_dir).resolve() if records_dir else RESULTS
    man_path = Path(manifest).resolve() if manifest else (LOGS / Path(MANIFEST_OWN).name)
    listed, man_sha = read_manifest_map(man_path)
    want = summary_inputs()
    if sorted(listed) != sorted(want):
        refuse(f"{rel(man_path)} lists {len(listed)} files, not exactly this list's {len(want)} "
               f"(missing {sorted(set(want) - set(listed))[:3]}, extra {sorted(set(listed) - set(want))[:3]})")
    preds = passing_predictions(code0, records_dir=rdir)
    if listed.get(PRED_REL) != preds["sha256"]:
        refuse(f"{PRED_REL} is not the predictions record {rel(man_path)} names")
    docs: dict[str, dict[str, Any]] = {}
    z_of: dict[str, dict[str, np.ndarray]] = {}
    per_record: dict[str, Any] = {}
    inputs: dict[str, str] = {rel(man_path): man_sha, PRED_REL: preds["sha256"]}
    members = {"phase2_headline": [f"refsize_d382_p2_{k}" for k in P2_PAIRS],
               "phase1_released": [f"refsize_d382_p1_{HC.core_of(s)}" for s in P1_PAIRS]}
    kind_of = {"phase2_headline": RUN_KIND["p2"], "phase1_released": RUN_KIND["p1"]}
    n_draws_seen: set[int] = set()
    for pop, names in members.items():
        for nm in names:
            jrel = f"results/{nm}.json"
            doc, h = read_bound(jrel, listed, rel(man_path))
            if doc.get("run_kind") != kind_of[pop] or (final and doc.get("registered_parameters") is not True):
                refuse(f"{jrel} is not a registered {kind_of[pop]} record")
            if (doc.get("code") or {}).get("sha256") != code0["sha256"]:
                refuse(f"{jrel} was written by other code than this summary")
            if (doc.get("predictions_record") or {}).get("sha256") != preds["sha256"]:
                refuse(f"{jrel} was not licensed by the predictions record this summary reads")
            cfg = doc.get("config") or {}
            if cfg.get("population") != pop or cfg.get("level") != LEVEL or cfg.get("probe_set") != PSET:
                refuse(f"{jrel} is not a {pop}, {LEVEL}, {PSET} record")
            if list(cfg.get("c_grid") or []) != list(C_GRID) or list(cfg.get("shifts") or []) != list(SHIFTS):
                refuse(f"{jrel} ran another c grid")
            n_draws_seen.add(int(cfg.get("n_draws") or -1))
            nrel = (doc.get("draws_npz") or {}).get("file")
            zz, nh = npz_bound(str(nrel), listed, (doc.get("draws_npz") or {}).get("sha256"), rel(man_path))
            docs[nm], z_of[nm] = doc, zz
            inputs[jrel], inputs[str(nrel)] = h, nh
            per_record[nm] = {"file": jrel, "sha256": h, "npz": {"file": str(nrel), "sha256": nh},
                              "commit": (doc.get("git") or {}).get("commit"), "config_hash": doc.get("config_hash"),
                              "n_grid": list(cfg.get("n_grid") or []), "layers": list(cfg.get("layers") or []),
                              "anchor_counts": (doc.get("anchor") or {}).get("counts"),
                              "d170_flag": (doc.get("input_verification") or {}).get("flag"),
                              "elapsed_s": doc.get("elapsed_s")}
    if len(n_draws_seen) != 1:
        refuse(f"the records ran different draw counts {sorted(n_draws_seen)}: a draw-count mismatch")
    n_draws = n_draws_seen.pop()
    flagged = [nm for nm, r in per_record.items() if r["d170_flag"] is not None]
    rows: dict[str, dict[str, Any]] = {}
    for pop, names in members.items():
        ns = sorted({n for nm in names for n in (docs[nm]["config"].get("n_grid") or [])})
        if any(sorted(docs[nm]["config"].get("n_grid") or []) != ns for nm in names):
            refuse(f"the {pop} records ran different reference sizes")
        rows[pop] = {f"n{n}|{tag}": row_block(z_of, names, n, tag) for n in ns for tag in (*SHIFTS, POOLED_C)}
        rows[pop]["n_grid"] = ns
    verdicts = verdict_block(rows["phase2_headline"], rows["phase2_headline"]["n_grid"])
    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"] or any(sha256_file(ROOT / f) != h for f, h in inputs.items()
                                                 if (ROOT / f).is_file()):
        refuse("to write: code or an input changed during the run")
    for lab, blk in verdicts.items():
        print(f"{lab}: {blk['verdict']}", flush=True)
    config = {"preregistration": PREREG, "manifest": {"file": rel(man_path), "sha256": man_sha},
              "predictions_record": preds, "n_draws": n_draws, "seeds": REGISTERED["seeds"],
              "n_grid_registered": list(N_GRID), "c_grid": list(C_GRID), "shifts": list(SHIFTS),
              "seeds_block": {"probe_seeds": REGISTERED["seeds"],
                              "reference_permutation_seed_by_axis": dict(REF_PERM_SEED),
                              "pair_cluster_bootstrap": {"n_boot": HS.B, "seed": HS.SEED}},
              "primary_row": "phase2_headline", "primary_class": PRIMARY_CLASS, "level": LEVEL, "probe_set": PSET,
              "target_fpr": TARGET, "detectors": list(DETECTORS),
              "pooled_interval": f"pair-cluster bootstrap B={HS.B}, seed {HS.SEED} (D-177 (b))",
              "band_note": "per-draw bands cover partitions only; pooled intervals cover pairs only (D-166)",
              "d170": f"D-170: {len(flagged)} of {len(per_record)} records carry a flag",
              "no_unflagged_pairs_subrow":
                  "this summary has no 'unflagged pairs' sub-row (4c's D-255 summary has one) because the PRIMARY "
                  "population is Phase 2, where no record carries a D-170 flag (twomoment_p2_d361.D170_P2), and D-170 "
                  "does not interact with the reference size, which is this item's estimand; for the beside Phase 1 "
                  "rows each record's own flag is recorded beside it (records[*].d170_flag and its status block), "
                  "so a flagged-pair restriction can be read off the records without a re-run",
              "blas_threads": blas_threads()}
    payload = {**provenance_block(RUN_KIND["summary"]), "preregistration": PREREG, "code": code0,
               "reused_files_sha256": reused_now(), "implementation_choices": CHOICES, "elapsed_s": round(time.time() - t0, 1),
               "status": status_block(
                   all(bool(docs[nm].get("registered_parameters")) for nm in per_record),
                   [f"D-170 ({len(flagged)} of {len(per_record)} records carry the flag: {sorted(flagged)[:3]}...)"]
                   if flagged else [],
                   note="FLAGGED by design as 4c is (D-255's FLAGGED paragraph): the reference set and the evaluation "
                        "part are one pool of negatives re-partitioned per draw (D-382 (2)), so every accuracy here is "
                        "an upper bound for a deployer whose reference prompts differ from its traffic"),
               "config": config, "config_hash": canonical_hash(config),
               "verdicts": verdicts, "beside": BESIDE_NOTE,
               "accuracy_line_rule": choice("j"), "rows": rows, "records": per_record,
               "population_group": {"declared_in": None,
                                    "note": "outside every src/populations.py group until the pull; declared as "
                                            "REFSIZE_D382 then"},
               "input_files_sha256": inputs}
    problems = input_identity_problems(payload)
    if problems:
        refuse(f"to write (D-297 (1) results-writer rule): {problems}")
    write_outputs([(out_path, json_bytes_of(payload))])
    return out_path


# ----------------------------------------------------------------------------- the preflight
LANE_JOBS = {"A": [("p2", k) for k in P2_PAIRS], "D": [("p1", s) for s in P1_PAIRS]}


def run_preflight(lane: str, n_draws: int = P.N_DRAWS, final: bool = True) -> int:
    """Every record check of a lane's jobs up to the first cache read (records only); nothing is written and no
    evaluation number is computed. final (the default, what the lane list runs) also applies the final-name gates, so
    a missing BLAS pin, a set GUARDRAIL_CACHE_DIRS, an unregistered record of record or an edited reused file stops the
    list before it is loaded; a test passes final=False."""
    if lane not in LANE_JOBS:
        refuse(f"lane must be one of {sorted(LANE_JOBS)}")
    check_final_gates(final, n_draws == REGISTERED["n_draws"])
    code0 = P.code_block(CODE_FILES)
    if lane != "A":
        # lane A's own second job writes the predictions record, so only lane A's preflight may run without it; every
        # other lane's jobs refuse without it, so its preflight refuses too and the list is not loaded too early
        passing_predictions(code0)
    bad: list[str] = []
    for kind, name in LANE_JOBS[lane]:
        try:
            if kind == "p1":
                p1_bind(name, n_draws, list(SEEDS), final=final, records_only=True, code0=code0)
            else:
                p2_bind(name, n_draws, final=final, records_only=True, code0=code0)
            print(f"RECORDS OK: {kind} {name}")
        except SystemExit as e:
            bad.append(f"{kind} {name}: {e}")
            print(f"REFUSED: {kind} {name}: {e}")
    print(f"PREFLIGHT lane {lane}: {len(LANE_JOBS[lane]) - len(bad)} of {len(LANE_JOBS[lane])} RECORDS OK")
    if bad:
        refuse(f"{len(bad)} job(s) of lane {lane} would refuse; the list must not be loaded")
    print("RECORDS OK")
    return 0


# ----------------------------------------------------------------------------- the two lane lists
LANE_LETTER = {"A": "A", "D": "D"}
PRICING = (
    "CPU TIME. The measured evidence used; every figure read back from a file in this repo except (3) and (5), which "
    "say "
    "so in their own words (CLAUDE.md rule 2): "
    "(1) the ten 4c jobs of D-255 on the node, one lane, sequential: the completion times stored in "
    "results/healthcheck_d156_<pair>.json run 2026-09-28T00:53:42Z to 02:14:01Z, and D-255 records the lane's start at "
    "00:43:13Z, so the NINE released pairs' own gaps are 6.9 to 11.9 min, median 8.7 (Mistral 10.5, Gemma-2 11.9 at 22 "
    "layers, Llama-3 10.3, Llama-3.1 8.4, Qwen2.5 6.9 at 15 layers, OLMo-2 7.6, Llama-3 -> 3.1 8.7, WG Mistral 8.5, WG "
    "Llama-3 8.8); with the GPU-change control the ten gaps have median 8.8 and mean 9.1 (D-255's 'about 9 minutes "
    "each'). Each covers 17-22 layers x 2 probe sets x 2 levels of 1001 partitions on 1,000 + 2,000 negatives, 170 "
    "probe fits and six caches hashed twice. "
    "(2) the workstation measurement recorded in the header of queues/laneA_20261007_p2mech.tsv (D-361's own price): "
    "one Phase 2-sized cell (5,000 + 6,000 negatives, 1000 draws, BOTH LEVELS) about 64 s per target, so about 32 s for "
    "ONE level at Phase 2 size; a Phase 1-sized cell about 22 s for both levels, so about 11 s for one. Its per-pair "
    "job "
    "was priced 20-30 min, of which 7 layers x 2 targets x 64 s = about 15 min is that arithmetic. "
    "(3) THIS module's own refsize_d382.partition_arrays, timed on the workstation with 6 BLAS threads on SYNTHETIC "
    "scores of the real shapes (21 partitions scaled to 1001, the upper of three runs, 2026-10-08, after the fix "
    "round): "
    "44.0 s per Phase 2 cell (5,000 + 6,000 negatives, 4 reference sizes x 11 shift settings) and 21.3 s per Phase 1 "
    "cell (1,000 + 2,000, 3 x 11). THESE TWO NUMBERS ARE READ BACK FROM NO FILE IN THIS REPO: they are an UNRECORDED "
    "SANDBOX TIMING of this code on synthetic data, kept here for lane time only (CLAUDE.md rule 2); no number of this "
    "item rests on them, no record quotes them, and the RECIPE below replaces them with a node measurement. The "
    "arithmetic of this item is 4c's health() called |n| x 11 times per partition instead of once, plus the per-seed "
    "at-the-floor count of CHOICES (q), so the per-cell arithmetic is the dominant NEW term and (3) prices it directly, "
    "while the probe fits, the gather and the cache hashing are 4c's and D-361's and (1) and (2) price them. "
    "(4) the D-383 list was priced at 22.5 h by a workstation sandbox and ran in about 1 h 20 m on the node "
    "(docs/STATE.md 2026-10-08), so a workstation price taken on a loaded host is PESSIMISTIC for the node: the "
    "figures below are upper bounds, not forecasts. Layer counts are read from the records: 7 per Phase 2 pair (GRID6 "
    "+ L*, exactly 7 for all 20 D-361 pair jobs by that list's header) and 17 per Phase 1 pair except Gemma-2 (22) and "
    "Qwen2.5 (15), 156 cells in all, from results/healthcheck_d156_<pair>.json's config.layers. "
    "(5) NAS BYTES, so there is a bound to check against the headroom before loading: from the nine released pairs' 4c "
    "npz on disk under results/ (2.48-3.65 MB each, 25.9 MB in all, 552-672 arrays of 1001 float64 or int64) and, for "
    "this module's own shapes, from np.savez_compressed on arrays of exactly the keys, dtypes and value scales it "
    "writes "
    "at 1001 draws (the same unrecorded sandbox, 2026-10-08, read back from no file either): each Phase 2 npz about 14 "
    "MB (3,708 arrays) and each Phase 1 npz about 1.5 MB per layer cell (25.6 MB at 17 layers, 33 MB at Gemma-2's 22, "
    "about 23 MB at Qwen2.5's 15) - about 9x a 4c npz, because this module writes 5 health statistics plus 3 "
    "at-the-floor shares per (reference size, shift) where 4c writes 6 per (probe set, level); each record about 1.0 MB "
    "(Phase 2) or 0.1 MB per layer cell (Phase 1). The per-lane totals are in the price of this lane below."
)
# The commands the recipe names, as argv (the test parses each with build_parser(), so every flag it names exists).
PRICE_COMMANDS: dict[str, tuple[tuple[str, ...], ...]] = {
    "A": (("preflight", "--lane", "A"),
          ("p2", "--pair", "qwen2_to_25", "--out-dir", "$HOME/.cache/gccf_scratch/refsize_price")),
    "D": (("preflight", "--lane", "D"),
          ("p1", "--pair", "repair_qwen_base2instruct", "--out",
           "$HOME/.cache/gccf_scratch/refsize_price/refsize_d382_p1_qwen_base2instruct.json"),
          ("p1", "--pair", "repair_qwen_base2instruct", "--layers", "20", "--out",
           "$HOME/.cache/gccf_scratch/refsize_price/refsize_d382_p1_qwen_one_layer.json")),
}


def price_recipe(lane: str) -> str:
    """The recipe that replaces (3) with a node measurement, written so that every command it gives RUNS: on the node,
    from the repo root, with scripts/project_env.sh sourced (the 6-thread pin) and GUARDRAIL_CACHE_DIRS unset."""
    cmds = [" ".join(("python3", "-u", "src/refsize_d382.py", *c)) for c in PRICE_COMMANDS[lane]]
    end = ("Lane A has no per-layer flag (p2 takes --pair, --n-draws, --records-only and --out-dir only), so for a "
           "cheaper probe use lane D's one-layer run below and (2)'s ratio between the two sizes."
           if lane == "A" else
           f"CHEAPER, one layer cell only (then multiply by the pair's layer count, 15 for Qwen2.5): `{cmds[2]}` "
           "(--layers makes the run unregistered, which only a final name would refuse).")
    return (
        "RECIPE to replace (3) with a node measurement. On the NODE (not the workstation: see the PREFLIGHT line "
        "above), from the repo root, scripts/project_env.sh sourced and GUARDRAIL_CACHE_DIRS unset: "
        f"(i) `{cmds[0]}` - records only, writes nothing, must print RECORDS OK for every job. "
        f"(ii) AFTER lane A's job refsize_predictions reports DONE (results/refsize_d382_predictions.json must exist at "
        f"its final name: every job that reads a cache refuses without it, D-382 (5)), and before this list is loaded, "
        f"ONE job of this lane into a scratch directory OUTSIDE results/: `{cmds[1]}`. Read ONLY the record's "
        "elapsed_s: that job is one of this lane's own jobs at full size, so the number IS its price (per layer: "
        "elapsed_s / the record's config.layers), not an extrapolation. "
        "NEVER pass --n-draws: p1_bind and p2_bind refuse any draw count other than the record of record's 1000 (a "
        "draw-count mismatch), because the anchor compares per-draw arrays draw for draw. Nothing under results/ is "
        "written (both jobs refuse an output path under results/), and the scratch record is not a record of record. "
        + end)


def lane_list_text(lane: str, date: str, prices: dict[str, str]) -> str:
    """One lane list: the header (what it is, when it may load, the order, the price) and its job lines."""
    letter = LANE_LETTER[lane]
    other = "D" if lane == "A" else "A"
    body = []
    body.append(
        f"# Lane {letter} (CPU only, no network; D-382, the author's step-4 item 2, pre-registered before any of this "
        f"code or its numbers existed): the reference-size trade-off, by src/refsize_d382.py. How many reference "
        f"prompts do the Gaussian health ratio and the direct reference ratio need, with and without A1's score "
        f"translations? n in {list(N_GRID)}; per draw T_d from the FULL calibration part's pre scores (D-382 (2)), the "
        f"reference the first n items of one seeded permutation of that part (nested sets), the two ratios exactly "
        f"D-241 (2)'s with the direct floor 1/n, the class from the realised paired ratio on the evaluation part; A1's "
        f"shifts translate the POST scores of every item by c * sd_pre for c in D-327's grid without 0 "
        f"({list(C_GRID)}), the class recomputed on the shifted realised ratio; metrics as 4c's summary (pooled "
        f"sensitivity and specificity with the pair-cluster bootstrap, ROC AUC per draw of |log ratio|). Item level, "
        f"the selected C of record, 1%, 1000 draws plus the frozen split, seeds 0-4. No reviewed module is edited: the "
        f"reviewed Phase 2 scoring of src/twomoment_p2_d361.py and the reviewed 4c functions of "
        f"src/healthcheck_d156.py and src/healthcheck_summary_d156.py are imported and called, and every one is bound "
        f"by sha256 in every record. Writes each declared output once (staged, hard-linked, never overwritten).")
    body.append(
        "# DO NOT LOAD before the THREE REVIEWS of src/refsize_d382.py (new analysis code, D-357 (A)(3)) and its fix "
        "round, and before the push that carries the module, its tests and both lane lists. No real-data run of this "
        "module has happened: its tests use synthetic worlds only.")
    if lane == "A":
        body.append(
            f"# THIS LIST is one of two (lane {letter} and lane {other}), split so neither idle CPU lane waits: lane A "
            f"carries the PRIMARY population (the nine Phase 2 headline pairs of D-361 (2), the only population that "
            f"holds 5,000 reference negatives) and lane D the BESIDE population (the nine Phase 1 released pairs at "
            f"n = 200, 500, 1000). The manifest and the summary are in NEITHER list: they must run after every job of "
            f"BOTH lanes, so the next review session loads them after the pull (minutes). Load lane A FIRST: its "
            f"second job writes the predictions record, which every job of both lanes requires.")
    else:
        body.append(
            f"# THIS LIST is one of two (lane {other} and lane {letter}), split so neither idle CPU lane waits: lane A "
            f"carries the PRIMARY population (the nine Phase 2 headline pairs) and this lane the BESIDE population "
            f"(the nine Phase 1 released pairs at n = 200, 500, 1000; n = 1000 is 4c's record and is the anchor). The "
            f"manifest and the summary are in NEITHER list (they must run after both lanes) and are loaded by the next "
            f"review session after the pull. LOAD THIS LIST AFTER lane A's job refsize_predictions reports DONE: it "
            f"writes results/refsize_d382_predictions.json in seconds and every job here refuses without it (D-382 "
            f"(5): the predictions are written before any number). This list's own preflight refuses, writing nothing, "
            f"if it is loaded too early.")
    body.append(
        "# ORDER: (1) the PREFLIGHT job first: every record check of this lane's jobs up to the first cache read (the "
        "predictions record, the records of record, the selections, the manifests, the layer plans, the ingest "
        "indexes, the BOS check); it reads no cache, computes no evaluation number and writes nothing, and the list "
        "must not be loaded unless it prints RECORDS OK for every job. "
        + ("(2) refsize_predictions: the three registered predictions P-N.1 to P-N.3 of D-382 (5) written BEFORE any "
           "number of this item exists; every later job of both lanes refuses unless it exists and was written by this "
           "exact code. (3) the nine Phase 2 pair jobs, each refusing unless the D-361 anchor record exists, passed "
           "and was written by D-361's own code, unless its probes equal its selection's fit bit for bit and "
           "reproduce its ho20 TPRs (ANCHOR FAILED otherwise, nothing written), and unless the realised operating "
           "point at 1% equals the pair's D-361 record of record at every layer; the Mistral v0.1 -> v0.2 job also "
           "refuses unless its L* cell equals results/phase2/float16_check_d364.json's arm A."
           if lane == "A" else
           "(2) the nine Phase 1 pair jobs, each refusing unless the n = 1000 unshifted arrays equal the pair's 4c npz "
           "(results/healthcheck_d156_<core>_draws.npz, bound by its record and by logs/healthcheck_d156.sha256) BYTE "
           "FOR BYTE at every layer and unless the realised report is paired.repartition_estimate's on the same "
           "partitions; nothing is written otherwise.")
        + " A failed job stops the list (queue runner); nothing downstream runs on a partial set.")
    body.append(
        "# AFTER BOTH LANES (the next review session, not in either list): the manifest job "
        "`sha256sum " + " ".join(summary_inputs()) + " > " + MANIFEST_OWN + "` (exactly those "
        + str(len(summary_inputs())) + " files), then `python3 -u src/refsize_d382.py summary`, which reads only those "
        "files through the manifest's lines, refuses unless the manifest lists exactly them and every record was "
        "written by its own code and licensed by the predictions record, and scores P-N.1 to P-N.3 HELD or FAILED with "
        "D-177's accuracy line on every predicted-against-realised table.")
    body.append(
        "# Final names need the registered parameters (1000 draws, seeds 0-4, every layer, n " + str(list(N_GRID))
        + ", c " + str(list(C_GRID)) + "), the 6-thread BLAS pin (scripts/project_env.sh), GUARDRAIL_CACHE_DIRS unset, "
        "the predictions record and the reused files at their reviewed bytes (REUSED_SHA256). Runs from the repo root "
        "through scripts/detach.sh.")
    body.append(
        "# PREFLIGHT HOST: the preflight and every job of BOTH lanes must run on the NODE. A WORKSTATION preflight "
        "CANNOT PASS: lane A's p2 jobs read the Phase 2 selection record of record through "
        "selection_tpr_v2.read_selection, which refuses unless this run's numpy, scipy and scikit-learn are the ones "
        "that record ran with (selection_p2.LIBS) - the records hold numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.2 (the "
        "node's), the workstation has numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, so it refuses, writing nothing. "
        "Lane D's p1 jobs have no library gate, but their caches and the 6-thread BLAS pin that every final name needs "
        "are the node's too, so lane D's preflight is run on the node as well; a workstation run of it proves nothing "
        "about the lane.")
    body.append("# " + PRICING)
    body.append(f"# PRICE OF THIS LANE: {prices[lane]}")
    body.append("# " + price_recipe(lane))
    body.append(
        "# After the pull: verify against logs/refsize_d382.sha256; add the 18 _draws.npz to the draw-array manifest "
        "(D-163 (6), src/write_draw_manifest.py); declare the group REFSIZE_D382 in src/populations.py (the records "
        "deliberately sit outside it until then, as D-381's do); report P-N.1 to P-N.3 against D-382 with the beside "
        "rows from results/refsize_d382_summary.json (D-382 (5): a FAILED prediction is a finding). Labels checked "
        f"unique against queues/lane*_*.tsv ({date}); the node's logs/queue/{letter}/ cannot be checked from the "
        f"workstation: check the labels against it before loading.")
    lines = list(body)
    for label, outputs, cmd in lane_jobs(lane):
        lines.append(f"{label}\t{outputs}\t{cmd}")
    return "\n".join(lines) + "\n"


def lane_jobs(lane: str) -> list[tuple[str, str, str]]:
    """(label, declared outputs, command) of one lane list; the outputs are exactly what each job writes."""
    out: list[tuple[str, str, str]] = [
        (f"refsize_preflight_{lane.lower()}", "-", f"python3 -u src/refsize_d382.py preflight --lane {lane}")]
    if lane == "A":
        out.append(("refsize_predictions", PRED_REL, "python3 -u src/refsize_d382.py predictions"))
        for k in P2_PAIRS:
            out.append((f"refsize_p2_{k}",
                        f"results/refsize_d382_p2_{k}.json results/refsize_d382_p2_{k}_draws.npz",
                        f"python3 -u src/refsize_d382.py p2 --pair {k}"))
    else:
        for s in P1_PAIRS:
            core = HC.core_of(s)
            out.append((f"refsize_p1_{core}",
                        f"results/refsize_d382_p1_{core}.json results/refsize_d382_p1_{core}_draws.npz",
                        f"python3 -u src/refsize_d382.py p1 --pair {s}"))
    return out


def label_conflicts(texts: dict[str, str], root: Path = ROOT) -> list[str]:
    """Every label of these lists that is already a label of another list of the same lane in queues/ or repeated here
    (lora_jobs_p2.check_labels' rule, repeated so no list is written with a label the queue runner could confuse)."""
    def lane_of(name: str) -> str:
        m = re.fullmatch(r"lane([A-Za-z0-9]+)_[A-Za-z0-9_.-]+\.tsv", name)
        if not m:
            refuse(f"{name} is not a queues/lane<L>_<...>.tsv list name")
        return m.group(1)

    seen: dict[tuple[str, str], str] = {}
    for p in sorted((Path(root) / "queues").glob("lane*_*.tsv")):
        if p.name in texts:
            continue
        for ln in p.read_text(encoding="utf-8").split("\n"):
            if ln.strip() and not ln.startswith("#"):
                seen.setdefault((lane_of(p.name), ln.split("\t")[0]), p.name)
    bad: list[str] = []
    for name, text in texts.items():
        lane = lane_of(name)
        for ln in text.split("\n"):
            if not ln.strip() or ln.startswith("#"):
                continue
            label = ln.split("\t")[0]
            if (lane, label) in seen:
                bad.append(f"label {label} of {name} is also a label of {seen[(lane, label)]} in lane {lane}")
            seen[(lane, label)] = name
    return bad


def lane_prices() -> dict[str, str]:
    """The two lanes' prices, from PRICING's measured evidence; the arithmetic is stated so a reviewer can redo it."""
    return {
        "A": "nine Phase 2 jobs, 7 layers each = 63 cells. NEW arithmetic, from (3): 63 x 44.0 s = 46 min. One "
             "paired.repartition_estimate per layer (realised + the fixed positives) at Phase 2 size and ONE level, "
             "bounded by half of (2)'s 64 s (its 64 s covers both levels per target; 22 s is a PHASE 1-sized cell and "
             "is not this bound): 63 x 32 s = 34 min. D-361's own non-arithmetic work per pair (the probe fits at L* "
             "and every layer, the v2 gather, six Phase 2 caches hashed at read and again at the end), from (2)'s 20-30 "
             "min per pair less its about 15 min of arithmetic, both figures from the same host: 5-15 min per pair = "
             "45-135 min. TOTAL about 2.1-3.6 h on this lane (plus the preflight and the predictions job, seconds "
             "each). At 2x the workstation rate, under 7.5 h; (4) says the node is faster, not slower. NAS BYTES, from "
             "(5): nine npz about 14 MB each = about 126 MB, nine records about 1.0 MB each = about 9 MB, so about 135 "
             "MB under results/ on this lane.",
        "D": "nine Phase 1 jobs, 156 layer cells in all (17 each, 22 Gemma-2, 15 Qwen2.5). NEW arithmetic, from (3): "
             "156 x 21.3 s = 55 min. One paired.repartition_estimate per layer at Phase 1 size, bounded by half of "
             "(2)'s 22 s (one level, one target): 156 x 11 s = 29 min. 4c's own non-arithmetic work per pair (the probe "
             "fits, six caches hashed twice, the realised report) is bounded ABOVE by (1)'s WHOLE per-pair node time, "
             "6.9-11.9 min (median 8.7), because that time already contains 4c's own arithmetic: this run fits half of "
             "4c's probes (the selected C only) and scores one level, so the bound is loose but sound, and subtracting "
             "(2)'s workstation seconds from (1)'s node minutes would not be: 9 x 6.9-11.9 min = 62-107 min. TOTAL "
             "about 2.4-3.2 h on this lane (plus the preflight, seconds). Gemma-2 (22 layers) is the longest job, "
             "Qwen2.5 (15) the shortest. The two lanes' ranges overlap (A 2.1-3.6 h, D 2.4-3.2 h; midpoints within "
             "about 0.1 h), so neither idles long before the manifest and the summary are loaded. NAS BYTES, from (5): "
             "156 cells x about 1.5 MB = about 235 MB of npz (23-33 MB per pair) and about 16 MB of records, so about "
             "250 MB under results/ on this lane; both lanes together under 0.45 GB, the summary a few MB.",
    }


def run_queue(write: bool = False, date: str | None = None, root: Path | None = None) -> dict[str, str]:
    """The two lane lists. --write writes them under queues/ (never overwriting); without it they are printed."""
    rt = Path(root).resolve() if root else ROOT
    day = date or datetime.now(timezone.utc).strftime("%Y%m%d")
    prices = lane_prices()
    texts = {f"lane{LANE_LETTER[lane]}_{day}_refsize.tsv": lane_list_text(lane, day, prices) for lane in ("A", "D")}
    bad = label_conflicts(texts, root=rt)
    if bad:
        refuse(f"the labels are not unique: {bad}")
    for name, text in texts.items():
        for ln in text.split("\n"):
            if ln.strip() and not ln.startswith("#") and len(ln.split("\t")) != 3:
                refuse(f"{name}: a job line without three TAB fields: {ln[:80]!r}")
    if write:
        for name, text in texts.items():
            p = rt / "queues" / name
            if p.exists() or p.is_symlink():
                refuse(f"{rel(p)} exists; a queue list is never overwritten")
            p.write_text(text, encoding="utf-8", newline="\n")
            print(f"Written: {rel(p)}\nsha256:  {sha256_bytes(text.encode('utf-8'))}")
    else:
        for name, text in texts.items():
            print(f"--- {name} ---\n{text}")
    return texts


# ----------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    """The CLI, built here so a test can check that every flag the PRICING RECIPE names exists. allow_abbrev=False
    everywhere: a flag must be spelled as it is, so `p2 --out <dir>` is an ERROR rather than an abbreviation of
    --out-dir, and a lane list that names the wrong one of the two fails loudly."""
    ap = argparse.ArgumentParser(description="D-382: the reference-size trade-off", allow_abbrev=False)
    sub = ap.add_subparsers(dest="mode", required=True, parser_class=lambda **kw: argparse.ArgumentParser(
        allow_abbrev=False, **kw))
    sp = sub.add_parser("predictions", help="write the predictions record (D-382 (5)); before any number")
    sp.add_argument("--out", default=None)
    sp = sub.add_parser("p1", help="one Phase 1 released pair (beside)")
    sp.add_argument("--pair", required=True, choices=P1_PAIRS)
    sp.add_argument("--n-draws", type=int, default=P.N_DRAWS)
    sp.add_argument("--seeds", type=int, nargs="*", default=None)
    sp.add_argument("--layers", type=int, nargs="*", default=None, help="TESTING: a subset of layers")
    sp.add_argument("--records-only", action="store_true")
    sp.add_argument("--out", default=None)
    sp = sub.add_parser("p2", help="one Phase 2 headline pair (primary)")
    sp.add_argument("--pair", required=True, choices=P2_PAIRS)
    sp.add_argument("--n-draws", type=int, default=P.N_DRAWS)
    sp.add_argument("--records-only", action="store_true")
    sp.add_argument("--out-dir", default=None)
    sp = sub.add_parser("preflight", help="every record check of a lane's jobs; writes nothing")
    sp.add_argument("--lane", required=True, choices=sorted(LANE_JOBS))
    sp.add_argument("--n-draws", type=int, default=P.N_DRAWS)
    sp.add_argument("--no-final-gates", action="store_true",
                    help="TESTING: skip the final-name gates (the BLAS pin, GUARDRAIL_CACHE_DIRS, the registered "
                         "records of record and the reused files' bytes)")
    sp = sub.add_parser("summary", help="the pooled scoring and the three verdicts")
    sp.add_argument("--out", default=None)
    sp.add_argument("--records-dir", default=None)
    sp.add_argument("--manifest", default=None)
    sp = sub.add_parser("queue", help="the two lane lists")
    sp.add_argument("--write", action="store_true")
    sp.add_argument("--date", default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdout()
    args = build_parser().parse_args(argv)
    if args.mode == "predictions":
        run_predictions(args.out)
    elif args.mode == "p1":
        run_p1(args.pair, args.n_draws, args.seeds, args.out, args.layers, args.records_only)
    elif args.mode == "p2":
        run_p2(args.pair, args.n_draws, args.out_dir, args.records_only)
    elif args.mode == "preflight":
        return run_preflight(args.lane, args.n_draws, final=not args.no_final_gates)
    elif args.mode == "summary":
        run_summary(args.out, args.records_dir, args.manifest)
    elif args.mode == "queue":
        run_queue(args.write, args.date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
