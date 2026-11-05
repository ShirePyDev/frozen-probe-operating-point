"""D-433: RECALIBRATION NOISE ON 200 TRUSTED PROMPTS (the author's item 6 of 2026-10-10, following D-430 (9), C5's
operational case). PRE-REGISTERED, DESCRIPTIVE (no prediction, no verdict), CPU only, from EXISTING caches only (no GPU
work, no extraction).

THE QUESTION. D-430 (9): no record holds the FPR ratio after recalibrating on 200 trusted prompts for the updates whose
realised ratio stayed within x1.5. Per such cell this module gives the frozen threshold's realised paired ratio (as the
records hold it) beside the 5th, 50th and 95th percentile over the re-partition draws of the realised paired ratio
after the threshold is recalibrated on 200 trusted benign prompts, by each threshold recalibration the paper reports.

CELLS (CHOICES (a)): D-430 (9)'s selection, reproduced exactly (docs/reports_d430/item9_opcase.py's rule): a cell of a
refsize_d382 record (D-382; Phase 1 the nine released pairs, every layer of the pair's grid; Phase 2 the nine D-361
headline pairs, GRID6 plus L*, user-span mean) is selected when the D-156 mean over draws 1..1000 of its unshifted
realised paired ratio lies in [1/1.5, 1.5]: 36 of 156 Phase 1 cells (9 pairs) and 41 of 63 Phase 2 cells (8 pairs;
llama3_to_31 has none and gets no job), as D-425 (1)(a) counts them. Selected C, item level, seeds 0-4, 1%.

PER DRAW d (index 0 the frozen split, then the 1000 D-156 item-level draws of the refsize record; CHOICES (b)-(f)):
  T_d        each seed's 1% threshold on the PRE scores of the draw's FULL calibration part cal_d (the frozen
             threshold; healthcheck_d156.thresholds), as D-156 and D-382 (2) set it.
  REFERENCE  the 200 trusted benign prompts: the first 200 of D-382's seeded permutation of cal_d (refsize_d382.ref_rows,
             REF_PERM_SEED of the axis), i.e. exactly the 200 prompts D-382's check reads at n = 200. cal_d is the
             draw's stored benign calibration part (Table II's R_same pool), disjoint from the draw's evaluation part
             by construction (a D-156 partition; checked per draw). Scored by the post model, and by the pre model for
             the two transfer methods.
  METHODS    (CHOICES (e); each per seed, on the reference only)
               reset              the 1% threshold on the reference's POST scores (paired.thr_at_fpr through
                                  healthcheck_d156.thresholds): the paper's threshold re-set, R_fresh / R_same
                                  (paper/tables_d156.tex:89, :179, :223; poolstudy_d156.py:416-418 as run).
               quantile_transfer  T_d's rank among the reference's PRE scores carried to its POST scores
                                  (paired.quantile_transfer, F-070): the paper's quantile transfer on the stored benign
                                  pool (paper/method.tex:114-116; repair_d156.py:283-287, Table II's R_same arm as run;
                                  on the full reference it coincides with the re-set, on 200 it does not).
               zscore_transfer    the two-moment transfer: post scores standardised by the reference's post mean and SD,
                                  against T_d standardised by its pre mean and SD (repair_d156.py:288-293, lines repeated):
                                  the paper's z-score transfer (paper/method.tex:126-128, results.tex:199-204), which the
                                  paper reports on the retrieval layer only.
  RATIO      paired.ratio_from_flags(F_pre, F_post): per seed the floored post FPR at the method's threshold on the
             draw's evaluation part over the floored pre FPR at T_d on the same items (the frozen ratio's own
             denominator, D-156), then the mean over the seeds. The frozen ratio is the same with F_post at T_d.
REPORTED per cell: the frozen ratio's D-156 mean, band and frozen split (and its 5th/50th/95th percentiles) beside each
method's 5th, 50th and 95th percentile over draws 1..N, D-156 mean and band, frozen split, and the shares of draws in
the two-fold class and outside x1.5; FPR/target beside. Per phase (summary): the median over cells (min, max) of each
percentile. Per-draw, per-seed values go to the job's npz (CHOICES (j)).

ANCHOR (nothing is written unless it passes; CHOICES (i)): per cell, the unshifted realised paired ratio per draw and
healthcheck_d156.health's five statistics at n = 200 equal the refsize record's npz byte for byte (draws 0..N), so the
draws, T_d, the 200 reference rows, the probes and the scores are D-382's; the refsize records are bound through
logs/refsize_d382.sha256. Phase 2 also runs D-361's anchor record check and the L* probe anchor (p2auroc_d424).

MODES (each output written once: staged, then named, never overwritten; the lane runner sets mode 444)
  preflight [--no-hash] [--phase P]  the refsize records bound by their manifest; the selection reproduces 36 + 41 cells
                    and D-430's cell list; per job the records it binds (refsize_d382.p1_bind / p2_bind, records only)
                    and every cache it reads present (and, without --no-hash, its sha256 the record's); the NAS bytes the
                    jobs will write (an upper bound from the array shapes). Reads no cache array; writes nothing.
  job --phase {1,2} [--pair P] [--out PATH] [--n-draws N]
                    results/recalnoise_d433_p<phase>_<core|key>.json and its _draws.npz (without --pair: every pair of
                    the phase that has a selected cell, one record each). --n-draws N < 1000 (the sandbox) computes
                    draws 0..N, a prefix of the registered draws, and may only write outside results/ (PROVISIONAL).
  summary [--records-dir D] [--out PATH]
                    results/recalnoise_d433_summary.json from the 17 job records and their npz: every ratio recomputed
                    from the stored counts, every percentile from the stored arrays, the frozen percentiles' medians
                    checked against docs/reports_d430/item9_opcase.json; per phase the medians over cells.
  price [--n-draws N]  times the per-cell arithmetic on SYNTHETIC arrays of the two phases' shapes (writes nothing).
  queue [--write] [--date YYYYMMDD]  queues/laneD_<date>_recalnoise_d433.tsv (sandbox, preflight, 17 jobs, summary).

Final names need n_draws 1000, seeds 0-4, the 6-thread BLAS pin (scripts/project_env.sh), GUARDRAIL_CACHE_DIRS unset and
the reused files at the bytes this module was written and tested against (REUSED_SHA256). src/refsize_d382.py is pinned
(D-418, D-423, D-424): imported, never edited. A test writes outside results/.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_d156 as D  # noqa: E402  (imported, not changed: registry_stem, CLUSTERS, sources_now)
import float16_check_d364 as F16  # noqa: E402  (imported, not changed: V2_EVAL_FILE)
import healthcheck_d156 as HC  # noqa: E402  (imported, not changed: thresholds, health, crosses, CLASSES, LEVELS)
import input_registry as REG  # noqa: E402
import p2auroc_d424 as P2A  # noqa: E402  (imported, not changed: load_scores, cache_plan, presence, p2_final_problems)
import paired as P  # noqa: E402
import refsize_d382 as RS  # noqa: E402  (PINNED, imported, never edited: p1_bind, p2_bind, ref_rows, ref_permutations)
import selection_p2 as SP  # noqa: E402  (imported, not changed: AXIS, CACHE, THREADS)
import selection_tpr_v2 as T  # noqa: E402  (imported, not changed: V2_CAL_FILE)
import twomoment_p2_d361 as TP2  # noqa: E402  (imported, not changed: make_parts, passing_anchor, write_outputs)
from provenance import (  # noqa: E402
    _RESULT_READS, _repo_rel, canonical_hash, ensure_utf8_stdout, input_identity_problems, provenance_block,
    read_result, resolve_cache, sha256_bytes, sha256_file,
)

ROOT = Path(__file__).resolve().parent.parent
CODE_ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
QUEUES = ROOT / "queues"
OWN = "src/recalnoise_d433.py"
# repair_d156.py and poolstudy_d156.py are hashed, not imported: their lines are repeated or cited (CHOICES (e))
CODE_FILES = list(dict.fromkeys([OWN, *P2A.CODE_FILES, *HC.CODE_FILES, "src/repair_d156.py", "src/poolstudy_d156.py"]))
# the reviewed files whose functions (or, for repair_d156 and poolstudy_d156, whose lines) this module reuses, at the
# bytes it was written and tested against (master 4efb9dd, 2026-10-10); a final name refuses other bytes
REUSED_SHA256 = {
    "src/refsize_d382.py": "36d1a7709904b21cfdc522aa1feb61f151a0d0d10b14a57b99bbfb3ec507e81a",
    "src/p2auroc_d424.py": "edc7346487c7d9369526602b0cd54176db04ffb9f319c4276ad4ae4f7d1c106f",
    "src/twomoment_p2_d361.py": "843a6c13316545b550c3c1850840707a45a409bb8cdb63d5b8d4b863339ddea1",
    "src/healthcheck_d156.py": "0ec2687ecdd887a28d4b8fb7a4db4b9081b69ea3fcf2b175c345279b4ed74de7",
    "src/repair_d156.py": "01211e1ec9b69adf2bcc7b738e80a06f07c229c283f105ced11f05083cbe358e",
    "src/poolstudy_d156.py": "e39614ed8f7e86f9f5251aaad25f44329c56f4fa3b70c3ffb77f8ce2ffb64908",
    "src/depth_d156.py": "a1f7b50ce70395f290b681f336e5a5e95e1428a2ca7c5915818644ad70f26734",
    "src/paired.py": "80429fa57f0267236a2af1670c74ef327802598257ab75834fddf64a70d97195",
    "src/input_registry.py": "e05d1fba11a4db8c46d0e7231fa7b804fbd8101017fda1c0889070c193684fbf",
    "src/selection_tpr_v2.py": "e928a44e969e8fc1ec87a2e9bb93f3c9ddd11f06778dd3066b1d9d0905c3746d",
    "src/selection_p2.py": "a79bb1b8a635c15190bad181e29c1ecd885ef01a71fda26e1101339f3bc59e97",
    "src/float16_check_d364.py": "6027c597844e4c684a530c15af3fd40d06bbd218275ee50ee7f00e4a2fa356b7",
}
RUN_KIND = {1: "recalnoise_d433_p1", 2: "recalnoise_d433_p2", "summary": "recalnoise_d433_summary"}
PREREG = ("D-433 (the author's item 6 of 2026-10-10: recalibration noise on 200 trusted prompts, following D-430 (9); "
          "PRE-REGISTERED, DESCRIPTIVE, CPU, existing caches); details by the closest D-entries (D-181): D-430 (9) and "
          "D-425 (1)(a) (the cells), D-382 (2) (T_d on the full calibration part, the reference = the first n of one "
          "seeded permutation of it per draw), D-156 (the paired ratio and its estimator), D-244 (2) and D-330 (1) "
          "(threshold recalibration on the stored calibration negatives, R_same), D-384 CHOICES 1 (R_same as the "
          "quantile transfer), D-256 (1) (the z-score transfer of the non-parametric layer), D-418 / D-423 (anchor on "
          "the record of the same pair), D-412 (1) and D-424 (descriptive: no D-177 line)")

# ----------------------------------------------------------------------------- registered parameters
SEEDS = [0, 1, 2, 3, 4]
N_DRAWS = P.N_DRAWS                                     # 1000 D-156 draws, the refsize records' (D-156, D-382 (1))
N_REF = 200                                             # the author's 200 trusted prompts; D-382's n = 200 reference
TARGET = P.TARGET_FPR                                   # 1%: the paper reports no recalibration at 5% (CHOICES (h))
WITHIN = TP2.FACTOR                                     # 1.5: D-430 (9)'s 'within x1.5' (D-177's factor)
METHODS = ("reset", "quantile_transfer", "zscore_transfer")
PERCENTILES = (5.0, 50.0, 95.0)
PRIMARY_CLASS = "two_sided_2x"                          # healthcheck_d156.CLASSES: ratio >= 2 or <= 0.5 (beside)
LEVEL = HC.LEVELS[0]                                    # item_level_D156
REGISTERED = {"n_draws": N_DRAWS, "seeds": SEEDS, "n_reference": N_REF, "target_fpr": TARGET}
PHASE_PAIRS = {1: list(RS.P1_PAIRS), 2: list(RS.P2_PAIRS)}
POPULATION = {1: "phase1_released", 2: "phase2_headline"}
EXPECTED_CELLS = {1: (36, 9), 2: (41, 8)}               # D-430 (9) / D-425 (1)(a): (cells, pairs) within x1.5
ITEM9_REL = "docs/reports_d430/item9_opcase.json"       # D-430 (9)'s committed report (the selection and frozen range)
ITEM9_POP = {1: "phase1_released", 2: "phase2_headline"}
MANIFEST_REFSIZE = RS.MANIFEST_OWN                      # logs/refsize_d382.sha256 (D-382's manifest job)
HEALTH = tuple(RS.HEALTH)                               # healthcheck_d156.health's five statistics (the anchor)
NODE_HOME = os.environ.get("GUARDRAIL_NODE_HOME") or Path.home().as_posix()  # the node account's home (the lane lists' scratch paths)
SANDBOX_P1, SANDBOX_P2 = "repair_llama3_to_31", "mistral_v01_v02"   # two selected cells each; L* not selected in P2
SANDBOX_P1_WG = "repair_WG_mistral_base2instruct"                    # review 2: the safety_wildguard axis
SANDBOX_DRAWS = 3

METHOD_TEXT = {
    "reset": "the 1% threshold on the reference's POST scores, per seed (paired.thr_at_fpr through "
             "healthcheck_d156.thresholds): the paper's threshold re-set (paper/tables_d156.tex:89's R_same / R_fresh "
             "caption, :179 and :223 R_fresh at n = 200, 500, 1000; poolstudy_d156.py:416-418 'recal' as run; the same "
             "arithmetic as repair_d156.py:277-279, whose arms there are the oracles)",
    "quantile_transfer": "T_d's rank among the reference's PRE scores carried to its POST scores, per seed "
                         "(paired.quantile_transfer, F-070): the paper's quantile transfer on the stored benign pool "
                         "(paper/method.tex:114-116; repair_d156.py:283-287 'quantile_thr_benign', Table II's R_same arm "
                         "as run, D-384 CHOICES 1 - on the FULL stored reference T is its own 1% quantile, so this "
                         "coincides with the re-set tables_d156.tex:89's caption names; on 200 of them the two differ; "
                         "tables_d156.tex:89 and :223's live-traffic transfer is another pool)",
    "zscore_transfer": "post scores standardised by the reference's post mean and SD (ddof 0) against T_d standardised "
                       "by the reference's pre mean and SD, a zero SD replaced by 1 (repair_d156.py:288-293 "
                       "'zscore_thr_benign', lines repeated): the paper's z-score transfer (paper/method.tex:126-128, "
                       "results.tex:199-204), reported in the paper on the retrieval layer only",
}
TRUSTED_POOL = ("the draw's calibration part cal_d (benign items, label 0, of the axis's pooled negatives; Phase 1: the "
                "frozen split's threshold_calib and eval negatives, 1,000 + 2,000, re-partitioned 1,000 / 2,000 per "
                "draw; Phase 2: split v2's calibration and evaluation negatives, 5,000 + 6,000, re-partitioned 5,000 / "
                "6,000), Table II's R_same pool (stored calibration negatives); the reference is the first 200 of the "
                "draw's D-382 permutation of cal_d (refsize_d382.ref_rows, the check's n = 200 reference), scored by "
                "the POST model (and the pre model for the transfers)")
EVAL_ITEMS = ("the draw's evaluation part (Phase 1: 2,000; Phase 2: 6,000 benign negatives), the complement of cal_d in "
              "the pooled negatives, so disjoint from every reference prompt (checked per draw: every partition "
              "indexes each pooled negative exactly once and the reference rows lie inside cal_d)")
DENOMINATOR = ("the frozen paired ratio's own denominator (D-156, paired.ratio_from_flags): per seed the pre model's FPR "
               "at T_d on the draw's evaluation part, floored at 1/n_eval; the per-seed ratio's numerator is the post "
               "FPR at the method's threshold on the same items, floored at 1/n_eval; the draw's value is the mean of "
               "the five per-seed ratios")

CHOICES = [
    "(a) CELLS: D-430 (9)'s selection exactly (docs/reports_d430/item9_opcase.py): a refsize_d382 record's cell (Phase 1 "
    "every layer of the pair's grid, Phase 2 GRID6 plus L*; selected C, item level, 1%) is selected when "
    "numpy.nanmean over draws 1..1000 of its unshifted realised paired ratio (npz key L<l>|none|realised_paired_ratio) "
    "lies in [1/1.5, 1.5], both ends included; 36 of 156 Phase 1 cells (9 pairs) and 41 of 63 Phase 2 cells (8 pairs), "
    "D-425 (1)(a)'s counts; the preflight and the summary refuse any other count or cell list. A pair with no selected "
    "cell (Phase 2 llama3_to_31) has no job. Only the selected layers are scored (no other cell is computed).",
    "(b) DRAWS: the refsize record's own: index 0 the frozen split, then the 1000 D-156 item-level draws (Phase 1 "
    "paired.repartition_draws at paired.AXIS_DRAW_SEED[axis]; Phase 2 at twomoment_p2_d361.P2_DRAW_SEED through "
    "make_parts), so the frozen ratio per draw IS the recorded one (CHOICES (i)). Item level only (D-382 (1)); no "
    "cluster-level row (no new estimand). Percentiles are over draws 1..N; the frozen split is reported beside as one "
    "draw (D-156), as item9_opcase reads the records.",
    "(c) THE 200 TRUSTED PROMPTS: per draw the first 200 of D-382's seeded permutation of the draw's calibration part "
    "(refsize_d382.ref_permutations at REF_PERM_SEED[axis], refsize_d382.ref_rows: the set in cal_d's own order), i.e. "
    "the 200 prompts on which D-382's check at n = 200 is computed (D-382 (2): 'the reference = the first n items of "
    "one seeded permutation of the calibration part per draw'; D-430 (9): 'the check at n = 200'). So the author's "
    "'resample 200 trusted benign prompts per draw' is D-382's per-draw resample, and the check and the "
    "recalibration read the same 200 prompts in every draw. The pool is cal_d, the stored benign calibration "
    "negatives (Table II's R_same, 'stored calibration data', D-330 (1); D-244 (2)); it is disjoint from the evaluation "
    "part by construction and the job checks it per draw. NOT R_fresh's 'held out of evaluation' pool: drawing the 200 "
    "from the evaluation part would change the evaluation items and so the frozen ratio would not be the recorded one.",
    "(d) T_d: each seed's 1% threshold on the PRE scores of the FULL calibration part (healthcheck_d156.thresholds), "
    "never the 200 (D-156; D-382 (2) 'T_d from the full calibration part'). The 200 are a subset of the items that set "
    "T_d, as D-382's check reads them; for the two transfers the reference's pre scores therefore share items with "
    "T_d's calibration part (a property of the operational case, stated, not corrected).",
    "(e) METHODS: every threshold-level recalibration the paper reports, applied per seed to the 200 reference prompts "
    "only: reset (paper/tables_d156.tex:89 'Threshold (R_same): re-set at the 1% upper tail of the post-update scores' "
    "and 'Threshold (R_fresh): re-set at 1% FPR on the post scores of 1000 benign prompts held out of evaluation'; "
    ":179 and :223, R_fresh re-set at n = 200, 500, 1000), quantile_transfer (paper/method.tex:114-116; tables_d156.tex"
    ":89 'quantile transfer of T from the pool's pre to its post scores' and :223) and zscore_transfer (paper/method.tex"
    ":126-128 'we compare frozen, quantile and z-score transfers against a labelled recalibration'; results.tex:199-204; "
    "reported on the retrieval layer only, applied here to the probe's scores unchanged). Their code: reset is "
    "healthcheck_d156.thresholds (paired.thr_at_fpr per seed; poolstudy_d156.py:416-418 and repair_d156.py:277-279 "
    "compute exactly this); quantile_transfer is paired.quantile_transfer per seed (repair_d156.py:283-287); "
    "zscore_transfer repeats repair_d156.py:288-293 (no function holds them), checked bit for bit in the tests against "
    "repair_d156.partition. NOT included: restandardisation (a feature-level repair that keeps T), the labelled refit "
    "(needs labels), and a Gaussian 'mean + z_0.99 SD' threshold (the paper reports none; listed for the author). "
    "Each method is applied as its function defines it, edge cases included: paired.quantile_transfer is +inf when T_d "
    "lies above all 200 reference pre scores (F-070's rule; T_d has at most 1% of cal_d at or above it, so a 200 subset "
    "can hold none of those items), which flags nothing and gives that seed the floored post FPR; such draws are kept "
    "and counted per cell (nonfinite_thresholds), never dropped or redefined.",
    "(f) THE RATIO: paired.ratio_from_flags(F_pre, F_post) with F_pre the pre scores of the draw's evaluation part at "
    "T_d and F_post the post scores of the same items at the method's threshold: per seed the floored post FPR over "
    "the floored pre FPR (floor 1/n_eval), then the mean over the five seeds per draw (D-156's paired ratio with only "
    "the post side's threshold changed, as poolstudy_d156's put() forms its recalibration arms, D-244). SEEDS: each seed "
    "is recalibrated on its own probe's scores and the seed mean is taken per draw, then percentiles over draws (D-156: "
    "'the paired ratio (mean over seeds of per-seed floored post/pre FPR)'); no per-seed percentile is reported.",
    "(g) REPORTED per cell, for the frozen ratio and each method: numpy.nanpercentile (linear) at 5, 50, 95 over draws "
    "1..N (item9_opcase's estimator); paired.band_summary (D-156 mean, 2.5-97.5 band, the frozen split and its "
    "percentile); the shares of draws with the ratio >= 2 or <= 0.5 (healthcheck_d156.CLASSES['two_sided_2x']) and "
    "outside [1/1.5, 1.5]; beside, FPR/target (the seed-mean floored post FPR over 1%). Per phase (the summary): the "
    "median over cells, with min and max, of each percentile. Descriptive: no prediction, no verdict and so no D-177 "
    "accuracy line (D-412 (1), D-424).",
    "(h) TARGET 1% only: the paper reports no recalibration at 5% (paper/tables_d156.tex has none; D-384 CHOICES 6: no "
    "5% repair record) and the refsize records are at 1% only, so no 5% row.",
    "(i) ANCHOR, per cell before anything is written: the unshifted realised paired ratio per draw and "
    "healthcheck_d156.health's five statistics at n = 200 equal the refsize record's npz byte for byte (draws 0..N); "
    "the reference permutation matrix hashes to the record's reference_permutations_sha256; the record's C, L*, seeds, "
    "axis, draw seed and permutation seed are the binding's. So draws, T_d, the 200 rows, probes and scores are "
    "D-382's (realign_d423's rule: anchor on the record of the same pair, which was itself anchored on 4c / D-361). The "
    "refsize records and npz are bound through logs/refsize_d382.sha256 (D-382's manifest). Phase 2 also passes D-361's "
    "anchor record (twomoment_p2_d361.passing_anchor) and the L* probe anchor (p2auroc_d424.load_scores).",
    "(j) EVIDENCE: the npz holds per cell, per draw and per seed, T_d, the count of evaluation negatives flagged by the "
    "pre model at T_d, and per arm (frozen and each method) the count flagged by the post model, the threshold "
    "(zscore_transfer: the equivalent mp + sp * tz, beside; its flags come from the standardised form) and the "
    "seed-mean ratio; every stored ratio is recomputed from the counts before writing and again by the summary, and the "
    "frozen split's 200 reference rows are stored. Per-item flags are not stored (up to 6,000 items x 5 seeds x 4 arms x "
    "1001 draws per cell); they are reproducible from the caches, probes and recorded seeds (refsize_d382's and "
    "realign_d418's per-draw standard).",
    "(k) LOADING: Phase 1 repeats refsize_d382.run_p1's lines (the registry caches, the probes fitted exactly so; the 4c "
    "record, the selection and the registry run bound by refsize_d382.p1_bind, records only) for the selected layers; "
    "Phase 2 calls p2auroc_d424.load_scores (refsize_d382.run_p2's loading section, its loaders unchanged) under "
    "refsize_d382.p2_bind (records only), as p2auroc_d424 and realign_d418 call it. refsize_d382's own final gates and "
    "predictions record are D-382's, not this item's; this module's final gates and pins apply (REUSED_SHA256).",
    "(l) SANDBOX DRAWS: --n-draws N < 1000 computes draws 0..N, a prefix of the registered draws (the D-156 partitions "
    "and the reference permutations are drawn in sequence from one generator, so the prefix is the same draws; the "
    "full matrices are generated and hashed, then sliced); such a record is PROVISIONAL and may not be written under "
    "results/.",
    "(m) LANE: one CPU list on lane D (the brief's choice, chained behind D-423's lane D list; the author's 'first free "
    "CPU lane' is for the session that loads it): a sandbox job first (tests, one real job per phase at --n-draws 3, the "
    "preflight without hashing and the summary over the three sandbox records, into a fresh scratch dir; a failure stops "
    "the lane), then the hashing preflight, the 9 Phase 1 and 8 Phase 2 jobs and the summary. No manifest job: the "
    "summary binds each job record by the sha256 it reads and each npz by the sha256 its record names.",
    "(n) POPULATION GROUP: none until the pull (src/populations.py validate() needs every declared file on disk; "
    "REFSIZE_D382's and p2auroc_d424's precedent, D-411 (4)).",
]


def choice(letter: str) -> str:
    return next(c for c in CHOICES if c.startswith(f"({letter}) "))


# ----------------------------------------------------------------------------- small helpers
def refuse(msg: str) -> None:
    raise SystemExit(f"REFUSING: {msg}")


def rel(p: Path | str) -> str:
    return _repo_rel(Path(p))


def canon(x: Any) -> str:
    """Canonical JSON text (json.dumps default=str, parsed back, keys sorted); not a result (D-297)."""
    return json.dumps(json.loads(json.dumps(x, default=str)), sort_keys=True)


def blas_threads() -> dict[str, str | None]:
    return {v: os.environ.get(v) for v in SP.THREAD_VARS}


def reused_now() -> dict[str, str | None]:
    return {f: (sha256_file(CODE_ROOT / f) if (CODE_ROOT / f).is_file() else None) for f in REUSED_SHA256}


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
    now = reused_now()
    for f, want in REUSED_SHA256.items():
        if now[f] is None:
            out.append(f"{f} is missing: this module reuses its reviewed functions")
        elif now[f] != want:
            out.append(f"{f} is not the reviewed bytes this module reuses ({now[f][:12]} vs {want[:12]})")
    return out


def check_final_gates(final: bool, registered: bool) -> None:
    if final:
        bad = final_problems(registered)
        if bad:
            refuse(f"the final names need: {bad}")


def json_bytes_of(payload: dict[str, Any]) -> bytes:
    b = json.dumps(payload, indent=2, sort_keys=False, default=str).encode("utf-8")
    json.loads(b.decode("utf-8"))      # not a result (D-297): this run's own bytes, parsed before staging
    return b


def savez_bytes(arrays: dict[str, np.ndarray], compressed: bool = True) -> bytes:
    buf = io.BytesIO()
    # np.ascontiguousarray would turn a 0-d array (n_eval) into shape (1,); a 0-d array is contiguous already
    (np.savez_compressed if compressed else np.savez)(
        buf, **{k: (np.asarray(v) if np.ndim(v) == 0 else np.ascontiguousarray(v)) for k, v in arrays.items()})
    return buf.getvalue()


def status_block(registered: bool, flags: list[str]) -> dict[str, Any]:
    return {"status": "PROVISIONAL" if not registered else ("FLAGGED" if flags else "FINAL"), "d_entries": flags,
            "rule": "CLAUDE.md evidence standard: FINAL, FLAGGED (with the D-entry) or PROVISIONAL (not the registered "
                    f"parameters {REGISTERED}, a test or sandbox output)"}


def stem_of(phase: int, pair: str) -> str:
    """The record stem of a pair: Phase 1 the core of its registry name (repair_<core>), Phase 2 the key itself."""
    if phase not in PHASE_PAIRS:
        refuse(f"phase must be 1 or 2, not {phase!r}")
    if pair not in PHASE_PAIRS[phase]:
        refuse(f"{pair!r} is not one of the Phase {phase} pairs {PHASE_PAIRS[phase]}")
    return HC.core_of(pair) if phase == 1 else pair


def out_paths(phase: int, stem: str, out: str | Path | None) -> tuple[Path, Path, bool]:
    final_path = (RESULTS / f"recalnoise_d433_p{phase}_{stem}.json").resolve()
    out_path = Path(out).resolve() if out else final_path
    final = out_path == final_path
    if not final and out_path.is_relative_to(RESULTS.resolve()):
        refuse("a test run (--out) must not write under results/; only this job's own final name may")
    if out_path.suffix != ".json":
        refuse(f"--out must name a .json file, not {out_path.name}")
    npz_path = out_path.with_name(out_path.stem + "_draws.npz")
    if npz_path.exists() and not out_path.exists():
        refuse(f"{npz_path} exists without {out_path.name}: the orphan of a failed write; move it aside (never delete "
               "it) and rerun")
    for p in (out_path, npz_path):
        if p.exists() or p.is_symlink():
            refuse(f"{p} exists; results are never overwritten (D-152 P5)")
    return out_path, npz_path, final


# ----------------------------------------------------------------------------- the threshold methods (CHOICES (e))
def recalibrate(method: str, pre_ref: np.ndarray, post_ref: np.ndarray, T: np.ndarray, x: np.ndarray,
                target: float = TARGET) -> tuple[np.ndarray, np.ndarray]:
    """One method on one draw: (flags of the post scores x (n, S) at the recalibrated threshold, the per-seed threshold
    on the post score scale (S,)). pre_ref / post_ref: (n_ref, S) scores of the reference prompts; T: (S,) T_d."""
    if method == "reset":
        t = HC.thresholds(post_ref, target)                     # paired.thr_at_fpr per seed (poolstudy's recal)
        return x >= t, t
    if method == "quantile_transfer":
        # repair_d156.py:283-287 (F-070: the exact rank transfer; returns T_d with no update)
        t = np.array([P.quantile_transfer(pre_ref[:, s], post_ref[:, s], T[s]) for s in range(post_ref.shape[1])])
        return x >= t, t
    if method == "zscore_transfer":
        # repair_d156.py:288-293, repeated (no function holds these lines)
        mq, sq = pre_ref.mean(axis=0), pre_ref.std(axis=0)
        mp, sp = post_ref.mean(axis=0), post_ref.std(axis=0)
        tz = (T - mq) / np.where(sq > 0, sq, 1.0)
        sp = np.where(sp > 0, sp, 1.0)
        return (x - mp) / sp >= tz, mp + sp * tz
    refuse(f"unknown method {method!r}; the registered methods are {METHODS}")
    raise AssertionError("unreachable")


def ratio_from_counts(k_pre: np.ndarray, k_post: np.ndarray, n: int) -> np.ndarray:
    """The per-draw paired ratio recomputed from per-seed flag counts (draws, S): paired.floored_fpr's and
    per_seed_ratio's arithmetic on k / n, then the mean over the seeds (paired.ratio_from_flags)."""
    kp, kq = np.asarray(k_pre, dtype=np.int64), np.asarray(k_post, dtype=np.int64)
    if kp.ndim != 2 or kp.shape != kq.shape or n < 1:
        refuse(f"flag counts {kp.shape} and {kq.shape} over n = {n} cannot give a ratio")
    out = np.empty(kp.shape[0], dtype="<f8")
    for i in range(kp.shape[0]):
        f0 = np.maximum(kp[i] / n, 1.0 / n)
        f1 = np.maximum(kq[i] / n, 1.0 / n)
        out[i] = float((f1 / f0).mean(axis=-1))
    return out


def cell_arrays(pool_pre: np.ndarray, pool_post: np.ndarray, parts: list[tuple[np.ndarray, np.ndarray]],
                perm: np.ndarray, n_ref: int = N_REF, target: float = TARGET,
                methods: tuple[str, ...] = METHODS) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """One cell's per-draw arrays (index 0 the frozen split). pool_*: (n_pool, S) scores of the pooled negatives (the
    calibration part's rows first in partition 0); parts: the partitions; perm: (n_parts, n_cal) reference permutations.
    Returns (arrays for the npz, the health statistics at n_ref for the anchor). Refuses a partition that does not index
    every pooled negative exactly once, a short array, a reference outside the calibration part, an unknown method."""
    pre_a, post_a = np.asarray(pool_pre), np.asarray(pool_post)
    if pre_a.ndim != 2 or post_a.shape != pre_a.shape or pre_a.shape[1] < 1:
        refuse(f"the score pools are {pre_a.shape} (pre) and {post_a.shape} (post): each must be the SAME "
               "(n_pooled_negatives, n_seeds) matrix")
    bad = [m for m in methods if m not in METHODS]
    if bad or not methods:
        refuse(f"unknown methods {bad}; the registered methods are {METHODS}")
    if not parts:
        refuse("no partition")
    n_pool, S = (int(x) for x in pre_a.shape)
    n_cal = int(np.asarray(parts[0][0]).size)
    if not 1 <= n_ref <= n_cal:
        refuse(f"a reference of {n_ref} does not fit a calibration part of {n_cal} without replacement")
    perm = np.asarray(perm)
    if perm.shape != (len(parts), n_cal):
        refuse(f"the reference permutations are {perm.shape}, not {(len(parts), n_cal)}")
    if perm.size and (int(perm.min()) < 0 or int(perm.max()) >= n_cal):
        refuse(f"the reference permutations index outside the calibration part (0..{n_cal - 1})")
    n_ev = n_pool - n_cal
    if n_ev < 1:
        refuse(f"the pools hold {n_pool} rows, no evaluation part beside a calibration part of {n_cal}")
    acc: dict[str, list[Any]] = {"T_d": [], "k_pre": [], "frozen|k_post": [], "frozen|ratio": []}
    for m in methods:
        acc.update({f"{m}|thr": [], f"{m}|k_post": [], f"{m}|ratio": []})
    health: dict[str, list[float]] = {q: [] for q in HEALTH}
    for i, (c_, e_) in enumerate(parts):
        ci, ei = np.asarray(c_), np.asarray(e_)
        if ci.ndim != 1 or ei.ndim != 1 or ci.dtype.kind not in "iu" or ei.dtype.kind not in "iu":
            refuse(f"partition {i} is not a pair of 1-D integer index arrays into the pools")
        if int(ci.size) != n_cal or int(ei.size) != n_ev:
            refuse(f"partition {i} has {ci.size} + {ei.size} items, not the frozen sizes {n_cal} + {n_ev} (D-156)")
        idx = np.concatenate([ci, ei])
        seen = None if int(idx.min()) < 0 or int(idx.max()) >= n_pool else np.bincount(idx, minlength=n_pool)
        if seen is None or int(seen.min()) != 1 or int(seen.max()) != 1:
            refuse(f"partition {i} does not index each of the {n_pool} pooled negatives exactly once: the reference "
                   "and the evaluation items would not be disjoint (D-156)")
        cp, cq, ep, eq = pre_a[ci], post_a[ci], pre_a[ei], post_a[ei]
        T_d = HC.thresholds(cp, target)                      # the FULL calibration part (CHOICES (d))
        F_pre = ep >= T_d
        rows = RS.ref_rows(perm[i], n_ref)                   # positions inside cal_d (CHOICES (c))
        pre_ref, post_ref = cp[rows], cq[rows]
        h = HC.health(pre_ref, post_ref, T_d)
        for q in HEALTH:
            health[q].append(h[q])
        F_fz = eq >= T_d
        acc["T_d"].append(np.asarray(T_d, dtype="<f8"))
        acc["k_pre"].append(F_pre.sum(axis=0))
        acc["frozen|k_post"].append(F_fz.sum(axis=0))
        acc["frozen|ratio"].append(P.ratio_from_flags(F_pre, F_fz))
        for m in methods:
            F_m, thr = recalibrate(m, pre_ref, post_ref, T_d, eq, target)
            acc[f"{m}|thr"].append(np.asarray(thr, dtype="<f8"))
            acc[f"{m}|k_post"].append(F_m.sum(axis=0))
            acc[f"{m}|ratio"].append(P.ratio_from_flags(F_pre, F_m))
    out: dict[str, np.ndarray] = {}
    for k, v in acc.items():
        if k.endswith("|ratio"):
            out[k] = np.asarray(v, dtype="<f8")
        elif k == "T_d" or k.endswith("|thr"):
            out[k] = np.asarray(v, dtype="<f8").reshape(len(parts), S)
        else:
            out[k] = np.asarray(v, dtype="<i4").reshape(len(parts), S)
    out["n_eval"] = np.asarray(n_ev, dtype="<i8")
    for k in ["frozen", *methods]:                           # CHOICES (j): the stored counts give the stored ratio
        if not RS.same_arrays(ratio_from_counts(out["k_pre"], out[f"{k}|k_post"], n_ev), out[f"{k}|ratio"]):
            refuse(f"{k}: the ratio recomputed from the per-seed counts is not paired.ratio_from_flags' bit for bit")
    return out, {q: np.asarray(v, dtype="<f8") for q, v in health.items()}


# ----------------------------------------------------------------------------- the cell selection (CHOICES (a))
def select_layers(z: dict[str, np.ndarray]) -> list[tuple[int, float]]:
    """D-430 (9)'s rule on one refsize npz: (layer, D-156 mean) of the cells whose mean over draws 1.. of the unshifted
    realised paired ratio lies in [1/1.5, 1.5] (docs/reports_d430/item9_opcase.py's lines)."""
    layers = sorted({int(k.split("|")[0][1:]) for k in z if k.startswith("L") and k.endswith("|none|realised_paired_ratio")})
    out = []
    for li in layers:
        rr = np.asarray(z[f"L{li}|none|realised_paired_ratio"])
        mean = float(np.nanmean(rr[1:]))
        if 1 / WITHIN <= mean <= WITHIN:
            out.append((li, mean))
    return out


def npz_needed(rel_path: str, listed: dict[str, str] | None, want: str, man_name: str) -> tuple[dict[str, np.ndarray], str]:
    """A refsize npz read once from its bytes (its record's and its manifest's sha256), keeping only the arrays this
    item reads: the unshifted realised paired ratio and the n = 200 unshifted health statistics of every layer."""
    p = ROOT / rel_path
    if p.is_symlink() or not p.is_file():
        refuse(f"{rel_path} is missing")
    b = p.read_bytes()
    h = sha256_bytes(b)
    if h != want or (listed is not None and listed.get(rel_path) != h):
        refuse(f"{rel_path} is not the npz its record and {man_name} name ({h[:12]} vs {str(want)[:12]})")
    keep = {}
    with np.load(io.BytesIO(b), allow_pickle=False) as z:
        for k in z.files:
            if k.endswith("|none|realised_paired_ratio") or any(k.endswith(f"|n{N_REF}|none|{q}") for q in HEALTH) \
                    or k == f"ref_rows_frozen|n{N_REF}":
                keep[k] = z[k]
    return keep, h


def refsize_manifest() -> tuple[dict[str, str], str]:
    return RS.read_manifest_map(LOGS / Path(MANIFEST_REFSIZE).name)


def bind_refsize(phase: int, stem: str, listed: dict[str, str]) -> dict[str, Any]:
    """The pair's refsize_d382 record and npz, bound through logs/refsize_d382.sha256: a registered record of this
    phase, 1000 draws, seeds 0-4, 1%."""
    rec_rel = f"results/refsize_d382_p{phase}_{stem}.json"
    doc, sha = RS.read_bound(rec_rel, listed, MANIFEST_REFSIZE)
    cfg = doc.get("config") or {}
    problems = []
    if doc.get("run_kind") != RS.RUN_KIND[f"p{phase}"]:
        problems.append(f"run_kind {doc.get('run_kind')!r}")
    if doc.get("registered_parameters") is not True:
        problems.append("not a run of D-382's registered parameters")
    if (doc.get("status") or {}).get("status") not in ("FINAL", "FLAGGED"):
        problems.append(f"status {(doc.get('status') or {}).get('status')!r}")
    if cfg.get("n_draws") != N_DRAWS or list(cfg.get("seeds") or []) != SEEDS or cfg.get("target_fpr") != TARGET:
        problems.append(f"n_draws {cfg.get('n_draws')}, seeds {cfg.get('seeds')}, target {cfg.get('target_fpr')}")
    if N_REF not in (cfg.get("n_grid") or []) or "none" not in (cfg.get("shifts") or []):
        problems.append(f"n_grid {cfg.get('n_grid')} lacks {N_REF} or its shifts lack 'none'")
    if problems:
        refuse(f"{rec_rel} cannot anchor this item: {problems}")
    dz = doc.get("draws_npz") or {}
    z, npz_sha = npz_needed(str(dz.get("file")), listed, str(dz.get("sha256")), MANIFEST_REFSIZE)
    return {"rel": rec_rel, "doc": doc, "sha": sha, "npz_rel": str(dz.get("file")), "z": z, "npz_sha": npz_sha,
            "selected": select_layers(z)}


def phase_selection(phase: int, listed: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Every pair's refsize binding and selected cells of one phase (the pair's registry name as key)."""
    return {pair: bind_refsize(phase, stem_of(phase, pair), listed) for pair in PHASE_PAIRS[phase]}


def check_against_item9(phase: int, sel: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The selection must be D-430 (9)'s: the expected counts and the committed report's cell list, cell for cell."""
    cells = [(pair, li) for pair, r in sel.items() for li, _ in r["selected"]]
    n_cells, n_pairs = len(cells), len({p for p, _ in cells})
    if (n_cells, n_pairs) != EXPECTED_CELLS[phase]:
        refuse(f"Phase {phase}: {n_cells} cells of {n_pairs} pairs within x{WITHIN:g}, not D-430 (9)'s "
               f"{EXPECTED_CELLS[phase]}")
    if (ROOT / ITEM9_REL).is_symlink() or not (ROOT / ITEM9_REL).is_file():
        refuse(f"{ITEM9_REL} (D-430 (9)'s committed report) is not in this tree")
    doc = read_result(ROOT / ITEM9_REL)
    rep = [(c["pair"], int(c["layer"])) for c in ((doc.get("populations") or {}).get(ITEM9_POP[phase]) or {}).get("cells", [])]
    if sorted(rep) != sorted(cells):
        refuse(f"Phase {phase}: the selected cells are not {ITEM9_REL}'s ({len(rep)} listed there)")
    return {"file": ITEM9_REL, "sha256": _RESULT_READS[ITEM9_REL], "n_cells": n_cells, "n_pairs": n_pairs,
            "cells_equal": True}


# ----------------------------------------------------------------------------- the anchor (CHOICES (i))
def anchor_cell(arr: dict[str, np.ndarray], health: dict[str, np.ndarray], z: dict[str, np.ndarray], li: int,
                where: str) -> int:
    """Refuses unless this cell's frozen ratio and n = 200 health statistics equal the refsize npz's draws 0..N byte for
    byte; returns the number of arrays compared."""
    n = int(arr["frozen|ratio"].size)
    keys = [(arr["frozen|ratio"], f"L{li}|none|{RS.REALISED}")] + [(health[q], f"L{li}|n{N_REF}|none|{q}") for q in HEALTH]
    missing = [rk for _, rk in keys if rk not in z]
    if missing:
        refuse(f"the anchor: {where} has no arrays {missing[:3]}; nothing written")
    bad = [rk for mine, rk in keys if int(np.asarray(z[rk]).size) < n
           or not RS.same_arrays(np.ascontiguousarray(mine, dtype="<f8"), np.ascontiguousarray(np.asarray(z[rk])[:n]))]
    if bad:
        refuse(f"the anchor: {len(bad)} of {len(keys)} arrays differ from {where} (first {bad[:3]}); nothing written")
    return len(keys)


# ----------------------------------------------------------------------------- the per-cell report (CHOICES (g))
def dist_block(x: np.ndarray) -> dict[str, Any]:
    a = np.asarray(x, dtype=float)
    draws = a[1:]
    fin = draws[np.isfinite(draws)]
    pct = [float(np.nanpercentile(draws, q)) for q in PERCENTILES] if fin.size else None
    return {"p5_p50_p95": pct, "d156": P.band_summary(draws, float(a[0])), "frozen_split": float(a[0]),
            "share_of_draws_two_sided_2x": float(HC.crosses(fin, HC.CLASSES[PRIMARY_CLASS]).mean()) if fin.size else None,
            "share_of_draws_outside_x1.5": float(((fin > WITHIN) | (fin < 1 / WITHIN)).mean()) if fin.size else None,
            "n_draws": int(draws.size), "n_nonfinite_draws": int(draws.size - fin.size)}


def fpr_over_target(k_post: np.ndarray, n: int) -> np.ndarray:
    """The seed-mean floored post FPR over the target, per draw (Table II's FPR/target), from the counts."""
    k = np.asarray(k_post, dtype=np.int64)
    return np.asarray([float(np.maximum(k[i] / n, 1.0 / n).mean()) / TARGET for i in range(k.shape[0])], dtype="<f8")


def cell_block(arr: dict[str, np.ndarray], li: int, mean_sel: float, meta: dict[str, Any]) -> dict[str, Any]:
    n_ev = int(arr["n_eval"])
    blk: dict[str, Any] = {
        "layer": li, "is_selected_layer": li == meta["L_sel"], "probe_set": "selected", "probe_C": meta["C"],
        "selection": {"d156_mean_realised_paired_ratio": mean_sel, "within": [1 / WITHIN, WITHIN], "rule": choice("a")},
        "n_reference": N_REF, "n_eval_negatives": n_ev,
        "frozen": {"paired_ratio": dist_block(arr["frozen|ratio"]),
                   "fpr_over_target": dist_block(fpr_over_target(arr["frozen|k_post"], n_ev))},
        "recalibrated": {}}
    for m in METHODS:
        nf = ~np.isfinite(np.asarray(arr[f"{m}|thr"], dtype=float))
        blk["recalibrated"][m] = {
            "paired_ratio": dist_block(arr[f"{m}|ratio"]),
            "fpr_over_target": dist_block(fpr_over_target(arr[f"{m}|k_post"], n_ev)),
            "nonfinite_thresholds": {"seed_draws": int(nf[1:].sum()),
                                     "share_of_draws_with_any_seed": float(nf[1:].any(axis=1).mean()) if nf.shape[0] > 1
                                     else None,
                                     "frozen_split_seeds": int(nf[0].sum()),
                                     "rule": "a +inf threshold flags nothing, so that seed's post FPR is the floor "
                                             "1/n_eval; paired.quantile_transfer returns +inf when T_d lies above "
                                             "every reference pre score (F-070), and paired.thr_at_fpr when no "
                                             "reference score meets the target"},
            "method": METHOD_TEXT[m]}
    return blk


NPZ_KEYS = ("L<layer>|T_d (draws x seeds, <f8): each seed's 1% threshold on the pre scores of the draw's full "
            "calibration part; L<layer>|k_pre (draws x seeds, <i4): evaluation negatives the pre model flags at T_d; "
            "L<layer>|<arm>|k_post (<i4), |thr (<f8), |ratio (draws, <f8) for arm in frozen (thr absent: it is T_d), "
            "reset, quantile_transfer, zscore_transfer (zscore thr = the equivalent mp + sp * tz); L<layer>|n_eval; "
            "layers; ref_rows_frozen (the frozen split's 200 reference rows, positions in its calibration part); index "
            "0 of every per-draw array is the frozen split")


# ----------------------------------------------------------------------------- Phase 1 world (CHOICES (k))
def check_refsize_config(rcfg: dict[str, Any], want: dict[str, Any], rec_rel: str) -> None:
    bad = {k: (rcfg.get(k), v) for k, v in want.items() if rcfg.get(k) != v}
    if bad:
        refuse(f"{rec_rel} was not run on this binding: {bad}")


def world_p1(pair: str, rs: dict[str, Any], layers: list[int], n_draws: int, records_only: bool,
             code0: dict[str, Any], log: Callable[[str], None]) -> dict[str, Any]:
    """Phase 1: refsize_d382.p1_bind (records only), then refsize_d382.run_p1's loading and scoring lines for the
    selected layers (the probes fitted exactly as there). Returns the pools, partitions, permutations and records."""
    b = RS.p1_bind(pair, N_DRAWS, list(SEEDS), False, True, code0)
    cfg, axis, C_sel, L_sel = b["cfg"], b["axis"], b["C_sel"], b["L_sel"]
    rcfg = rs["doc"]["config"]
    check_refsize_config(rcfg, {"pair": pair, "axis": axis, "selected_C": C_sel, "selected_layer": L_sel,
                                "seeds": list(SEEDS), "draw_seed": P.AXIS_DRAW_SEED[axis],
                                "reference_permutation_seed": RS.axis_perm_seed(axis), "level": LEVEL,
                                "pre_repo": cfg["pre_repo"], "pre_rev": cfg["pre_rev"], "post_repo": cfg["post_repo"],
                                "post_rev": cfg["post_rev"], "selection_sha256": b["sel_sha"]}, rs["rel"])
    if set(layers) - set(rcfg.get("layers") or []):
        refuse(f"layers {layers} are not all in {rs['rel']}'s grid {rcfg.get('layers')}")
    base = {"b": b, "axis": axis, "C_sel": C_sel, "L_sel": L_sel, "pre_repo": cfg["pre_repo"], "pre_rev": cfg["pre_rev"],
            "post_repo": cfg["post_repo"], "post_rev": cfg["post_rev"], "draw_seed": P.AXIS_DRAW_SEED[axis],
            "perm_seed": RS.axis_perm_seed(axis)}
    if records_only:
        return base
    parts_in = ("probe_train", "threshold_calib", "eval")
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
    if any((rs["doc"].get("input_cache_sha256") or {}).get(k) != h for k, h in input_sha.items()):
        refuse(f"{rs['rel']} read other caches than this run")
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
    if any(g != grids[0] for g in grids) or grids[0] != list(rcfg.get("layers") or []):
        refuse(f"the caches' layer grids differ from each other or from {rs['rel']}'s")
    y_tr = data["pre"]["probe_train"]["labels"].astype(bool)
    y_cal = data["pre"]["threshold_calib"]["labels"].astype(bool)
    y_ev = data["pre"]["eval"]["labels"].astype(bool)
    ncal, neg = ~y_cal, ~y_ev
    pooled_ids = [str(i) for i in data["pre"]["threshold_calib"]["item_ids"][ncal]] + \
                 [str(i) for i in data["pre"]["eval"]["item_ids"][neg]]
    n_cal, n_pool = int(ncal.sum()), len(pooled_ids)
    cl = json.loads(b["cl_bytes"].decode("utf-8"))["axes"][axis]
    pool_sha = sha256_bytes("\n".join(pooled_ids).encode("utf-8"))
    if cl["n_pooled_negatives"] != n_pool or cl["n_calibration"] != n_cal or cl["pool_item_ids_sha256"] != pool_sha:
        refuse(f"manifests/negative_clusters.json does not describe the {axis} pool")
    if (rcfg.get("n_calibration_negatives"), rcfg.get("n_pooled_negatives")) != (n_cal, n_pool):
        refuse(f"{rs['rel']} pooled {rcfg.get('n_calibration_negatives')} + ... = {rcfg.get('n_pooled_negatives')}, "
               f"the caches {n_cal} of {n_pool}")
    if {str(i) for i in data["pre"]["probe_train"]["item_ids"]} & set(pooled_ids):
        refuse("a probe_train item is in the pooled negatives (F-045)")
    draws = P.repartition_draws(n_cal, n_pool - n_cal, N_DRAWS, P.AXIS_DRAW_SEED[axis])
    parts = [(np.arange(n_cal), np.arange(n_cal, n_pool))] + [(d[:n_cal], d[n_cal:]) for d in draws]
    perm = RS.ref_permutations(n_cal, N_DRAWS, RS.axis_perm_seed(axis))
    perm_sha = sha256_bytes(np.ascontiguousarray(perm).tobytes())
    if perm_sha != rcfg.get("reference_permutations_sha256"):
        refuse(f"the reference permutations are not {rs['rel']}'s ({perm_sha[:12]})")
    pools: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    diag: dict[int, list[dict[str, Any]]] = {}
    for li in layers:
        k = f"layer_{li}"
        Xtr = data["pre"]["probe_train"][k]
        probes = []
        for seed in SEEDS:                                   # refsize_d382.run_p1's lines (and 4c's)
            _, fit = P.d117_split(len(y_tr), seed)
            probes.append(P.fit_probe(Xtr[fit], y_tr[fit], seed, C_sel, "d117_permutation"))
        S_ = (lambda x: np.column_stack([q.score(x) for q in probes]))  # noqa: E731
        pre_pool = np.vstack([S_(data["pre"]["threshold_calib"][k][ncal]), S_(data["pre"]["eval"][k][neg])])
        post_pool = np.vstack([S_(data["post"]["threshold_calib"][k][ncal]), S_(data["post"]["eval"][k][neg])])
        pools[li] = (pre_pool, post_pool)
        diag[li] = [q.diag for q in probes]
        log(f"  layer {li:>2} scored")
    for side in data:
        for zz in data[side].values():
            zz.close()

    def end_checks() -> list[str]:
        moved = [w for w, x, y in (("registry sources", D.sources_now(), reg_sources),
                                   ("cluster manifest", sha256_file(D.CLUSTERS), b["cl_sha"]),
                                   (b["npz_rel"], sha256_file(ROOT / b["npz_rel"]), b["npz_sha"])) if x != y]
        moved += [m for m, h in b["manifests"].items() if sha256_file(ROOT / m) != h]
        moved += [f"cache {r['file']}" for r in input_recs.values()
                  if sha256_file(resolve_cache(r["file"], REG.CACHE)) != r["sha256"]]
        return moved

    flags = [f"D-170 ({w}: {v!r}); it stands until D-168 is final (D-172)" for w, v in
             (("this run's own input verification", (verification or {}).get("flag")),
              (f"the refsize record {rs['rel']}", ((rs["doc"].get("input_verification") or {}).get("flag"))))
             if v not in (None, "None", "")]
    return {**base, "pools": pools, "parts": parts[:n_draws + 1], "perm": perm[:n_draws + 1], "perm_sha": perm_sha,
            "n_cal": n_cal, "n_ev": n_pool - n_cal, "n_pool": n_pool, "pool_sha": pool_sha,
            "input_cache_sha256": input_sha,
            "input_cache_source": {kk: {"file": r["file"], "expected_from": r["expected_from"], "pin_only": r["pin_only"]}
                                   for kk, r in input_recs.items()},
            "input_verification": verification, "probe_diagnostics": diag, "end_checks": end_checks, "flags": flags,
            "extra_files": {b["npz_rel"]: b["npz_sha"], rel(D.CLUSTERS): b["cl_sha"], **b["manifests"]},
            "anchor_extra": {"healthcheck_4c": {"file": b["rec_rel"], "sha256": b["rec_sha"]},
                             "selection": {"file": rel(b["sel_path"]), "sha256": b["sel_sha"]}},
            "input_registry_sources_sha256": reg_sources}


# ----------------------------------------------------------------------------- Phase 2 world (CHOICES (k))
def world_p2(key: str, rs: dict[str, Any], layers: list[int], n_draws: int, final: bool, records_only: bool,
             code0: dict[str, Any], log: Callable[[str], None]) -> dict[str, Any]:
    """Phase 2: refsize_d382.p2_bind (records only), D-361's anchor record, the cache plan and its presence, then
    p2auroc_d424.load_scores for the selected layers (refsize_d382.run_p2's loading section; the L* probe anchor)."""
    b = RS.p2_bind(key, N_DRAWS, False, True, code0)
    registered = n_draws == N_DRAWS and list(b["seeds"]) == SEEDS
    rcfg = rs["doc"]["config"]
    pool_sha = sha256_bytes("\n".join(b["pool_ids"]).encode("utf-8"))
    check_refsize_config(rcfg, {"pair": key, "pooling": TP2.PRIMARY_POOLING, "axis": SP.AXIS,
                                "selected_C": b["C_sel"], "selected_layer": b["L_sel"], "layers": list(b["layers"]),
                                "seeds": list(SEEDS), "draw_seed": TP2.P2_DRAW_SEED,
                                "reference_permutation_seed": RS.axis_perm_seed(SP.AXIS), "level": LEVEL,
                                "pre_repo": b["pre_repo"], "pre_rev": b["sr"]["rev"], "post_repo": b["post_repo"],
                                "post_rev": b["post_sel"]["rev"]}, rs["rel"])
    if (rcfg.get("pooled_negatives") or {}).get("item_ids_sha256") != pool_sha:
        refuse(f"{rs['rel']} pooled other negatives than the binding's v2 split")
    if set(layers) - set(b["layers"]):
        refuse(f"layers {layers} are not all in the pair's cell set {b['layers']}")
    if final:
        bad = P2A.p2_final_problems(b, registered)
        if (b.get("d361") or {}).get("registered_parameters") is not True:
            bad.append(f"{b.get('d361_rel')} was not run with D-361's registered parameters")
        if bad:
            refuse(f"the final names need: {bad}")
    tp2_code = RS.tp2_code_block(code0)
    anchor_rec = None if records_only else TP2.passing_anchor(tp2_code, final=final)
    plan = P2A.cache_plan(b)
    pres, missing = P2A.presence(plan, list(layers))
    if missing:
        refuse(f"{key}: caches absent or short of a layer: {sorted({m['cache_key'] for m in missing})} (no imputation)")
    base = {"b": b, "axis": SP.AXIS, "C_sel": b["C_sel"], "L_sel": b["L_sel"], "pre_repo": b["pre_repo"],
            "pre_rev": b["sr"]["rev"], "post_repo": b["post_repo"], "post_rev": b["post_sel"]["rev"],
            "draw_seed": TP2.P2_DRAW_SEED, "perm_seed": RS.axis_perm_seed(SP.AXIS), "plan": plan, "presence": pres}
    if records_only:
        return base
    loaded = P2A.load_scores(b, list(layers), log)
    if sorted(loaded["sc"]) != sorted(layers):
        refuse(f"the loaders scored layers {sorted(loaded['sc'])}, not {sorted(layers)}")
    n_cal, n_ev = b["n_cal"], b["n_ev"]
    draws = P.repartition_draws(n_cal, n_ev, N_DRAWS, TP2.P2_DRAW_SEED)
    parts = TP2.make_parts({LEVEL: draws}, n_cal, n_cal + n_ev)[LEVEL]
    perm = RS.ref_permutations(n_cal, N_DRAWS, RS.axis_perm_seed(SP.AXIS))
    perm_sha = sha256_bytes(np.ascontiguousarray(perm).tobytes())
    if perm_sha != rcfg.get("reference_permutations_sha256"):
        refuse(f"the reference permutations are not {rs['rel']}'s ({perm_sha[:12]})")
    pools = {li: (np.vstack([sc["cal_pre"], sc["ev_pre"]]), np.vstack([sc["cal_post"], sc["ev_post"]]))
             for li, sc in loaded["sc"].items()}
    caches = loaded["caches"]

    def end_checks() -> list[str]:
        files_sha = {T.V2_CAL_FILE: sha256_file(SP.SPLIT_ROOT / T.V2_CAL_FILE),
                     F16.V2_EVAL_FILE: sha256_file(SP.SPLIT_ROOT / F16.V2_EVAL_FILE)}
        files_sha.update({k: sha256_file(p) for k, p in b["index_read"].items()})
        moved = [f"{k} (the v2 record names {w[:16]})" for k, w in ((T.V2_CAL_FILE, b["v2"]["sha256"]),
                                                                    (F16.V2_EVAL_FILE, b["ev2"]["sha256"]))
                 if files_sha[k] != w]
        moved += [k for k, h in b["index_sha0"].items() if files_sha[k] != h]
        moved += TP2.end_hash_problems({f"data/hidden/{r['file']}": (SP.CACHE / r["file"], r["sha256"])
                                        for r in caches.values()})
        moved += [m for m, h in b["manifests"].items() if sha256_file(RS.ROOT / m) != h]
        return moved

    return {**base, "pools": pools, "parts": parts[:n_draws + 1], "perm": perm[:n_draws + 1], "perm_sha": perm_sha,
            "n_cal": n_cal, "n_ev": n_ev, "n_pool": n_cal + n_ev, "pool_sha": pool_sha,
            "input_cache_sha256": {k: r["sha256"] for k, r in caches.items()},
            "input_cache_source": {k: f"data/hidden/{r['file']} (expected bytes from {r['expected_from']})"
                                   for k, r in caches.items()},
            "input_verification": TP2.D170_P2, "probe_diagnostics": loaded["probe_diagnostics"],
            "end_checks": end_checks, "flags": [],
            "extra_files": {T.V2_CAL_FILE: b["v2"]["sha256"], F16.V2_EVAL_FILE: b["ev2"]["sha256"],
                            **{k: h for k, h in b["index_sha0"].items()}, **b["manifests"]},
            "anchor_extra": {"d361_anchor_record": anchor_rec, "d361_pair_record": {"file": b["d361_rel"],
                                                                                    "sha256": b["d361_sha"]},
                             "probe_anchor_at_L_star": loaded["probe_anchor"]},
            "v2_gather": loaded["gathered"]}


# ----------------------------------------------------------------------------- the job
def run_job(phase: int, pair: str, n_draws: int = N_DRAWS, out: str | Path | None = None, records_only: bool = False,
            log: Callable[[str], None] = print) -> Path:
    """One pair of one phase: its selected cells (module docstring)."""
    t0 = time.time()
    stem = stem_of(phase, pair)
    if not 1 <= int(n_draws) <= N_DRAWS:
        refuse(f"--n-draws must lie in 1..{N_DRAWS}, not {n_draws}")
    registered = int(n_draws) == N_DRAWS
    out_path, npz_path, final = out_paths(phase, stem, out)
    check_final_gates(final, registered)
    code0 = P.code_block(CODE_FILES)
    listed, man_sha = refsize_manifest()
    rs = bind_refsize(phase, stem, listed)
    sel = rs["selected"]
    if not sel:
        refuse(f"{pair}: no cell of {rs['rel']} lies within x{WITHIN:g} (D-430 (9)); this pair has no job")
    layers = [li for li, _ in sel]
    if phase == 1:
        W = world_p1(pair, rs, layers, n_draws, records_only, code0, log)
    else:
        W = world_p2(pair, rs, layers, n_draws, final, records_only, code0, log)
    if records_only:
        log(f"RECORDS OK: Phase {phase} {pair}: {len(layers)} selected cells {layers} (records only; no cache read, "
            "nothing written)")
        return out_path
    meta = {"L_sel": W["L_sel"], "C": W["C_sel"]}
    log(f"D-433 Phase {phase} {pair}: {W['pre_repo']} -> {W['post_repo']}; C {W['C_sel']:g}, L* {W['L_sel']}; cells "
        f"{layers}; {len(W['parts']) - 1} draws + the frozen split; n_cal {W['n_cal']}, n_eval {W['n_ev']}, reference "
        f"{N_REF}; registered {registered}")
    cells: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    n_anchor = 0
    for li, mean_sel in sel:
        pre_pool, post_pool = W["pools"][li]
        arr, health = cell_arrays(pre_pool, post_pool, W["parts"], W["perm"], N_REF, TARGET, METHODS)
        n_anchor += anchor_cell(arr, health, rs["z"], li, f"{rs['npz_rel']} L{li}")
        blk = cell_block(arr, li, mean_sel, meta)
        blk["probe_fit_diagnostics"] = W["probe_diagnostics"].get(li)
        cells.append(blk)
        for k, v in arr.items():
            arrays[f"L{li}|{k}"] = v
        fz = blk["frozen"]["paired_ratio"]["p5_p50_p95"]
        log(f"  L{li:>2}: anchor OK; frozen p5/p50/p95 {fz}; " + "; ".join(
            f"{m} {blk['recalibrated'][m]['paired_ratio']['p5_p50_p95']}" for m in METHODS))
    arrays["layers"] = np.asarray(layers, dtype="<i8")
    ref0 = RS.ref_rows(W["perm"][0], N_REF).astype("<i8")
    rec_ref0 = rs["z"].get(f"ref_rows_frozen|n{N_REF}")
    if rec_ref0 is not None and not RS.same_arrays(ref0, np.asarray(rec_ref0, dtype="<i8")):
        refuse(f"the frozen split's {N_REF} reference rows are not {rs['npz_rel']}'s")
    arrays["ref_rows_frozen"] = ref0
    if P.code_block(CODE_FILES)["sha256"] != code0["sha256"]:
        refuse("source changed during the run")
    moved = W["end_checks"]()
    moved += [r_ for r_, h in _RESULT_READS.items() if not (ROOT / r_).is_file() or sha256_file(ROOT / r_) != h]
    moved += [w for w, p_, h in (("refsize npz", ROOT / rs["npz_rel"], rs["npz_sha"]),
                                 (MANIFEST_REFSIZE, ROOT / MANIFEST_REFSIZE, man_sha)) if sha256_file(p_) != h]
    if moved:
        refuse(f"to write: changed during the run: {moved}")
    config = {"item": "D-433 (the author's item 6 of 2026-10-10): recalibration noise on 200 trusted prompts, "
                      "descriptive", "phase": phase, "population": POPULATION[phase], "pair": pair, "stem": stem,
              "axis": W["axis"], "pre_repo": W["pre_repo"], "pre_rev": W["pre_rev"], "post_repo": W["post_repo"],
              "post_rev": W["post_rev"], "selected_C": W["C_sel"], "selected_layer": W["L_sel"],
              "probe_set": "selected", "seeds": list(SEEDS), "target_fpr": TARGET, "level": LEVEL,
              "n_draws": int(n_draws), "n_draws_registered": N_DRAWS, "draw_seed": W["draw_seed"],
              "reference_permutation_seed": W["perm_seed"], "reference_permutations_sha256": W["perm_sha"],
              "n_reference": N_REF, "methods": dict(METHOD_TEXT), "trusted_pool": TRUSTED_POOL,
              "evaluation_items": EVAL_ITEMS, "denominator": DENOMINATOR,
              "n_calibration_negatives": W["n_cal"], "n_evaluation_negatives": W["n_ev"],
              "n_pooled_negatives": W["n_pool"], "pooled_item_ids_sha256": W["pool_sha"],
              "disjoint_reference_and_evaluation": {"checked_partitions": len(W["parts"]),
                                                    "rule": "every partition indexes each pooled negative exactly once "
                                                            "and the reference rows are positions inside cal_d"},
              "selection": {"rule": choice("a"), "layers_selected": layers,
                            "layers_in_record": list(rs["doc"]["config"].get("layers") or []),
                            "d156_mean_realised_paired_ratio": {str(li): m for li, m in sel}},
              "refsize_record": {"file": rs["rel"], "sha256": rs["sha"]},
              "refsize_npz": {"file": rs["npz_rel"], "sha256": rs["npz_sha"]},
              "percentiles": list(PERCENTILES), "percentile_method": "numpy.nanpercentile (linear) over draws 1..N",
              "blas_threads": blas_threads()}
    if phase == 1:
        config["input_registry_sources_sha256"] = W["input_registry_sources_sha256"]
    npz_bytes = savez_bytes(arrays)
    flags = list(W["flags"])
    rs_status = rs["doc"].get("status") or {}
    if rs_status.get("status") == "FLAGGED":
        flags += [f"inherited from {rs['rel']}: {d}" for d in rs_status.get("d_entries") or []]
    flags = list(dict.fromkeys(flags))
    payload = {**provenance_block(RUN_KIND[phase]), "preregistration": PREREG, "code": code0,
               "reused_files_sha256": reused_now(), "registered_parameters": registered,
               "status": status_block(registered, flags), "elapsed_s": round(time.time() - t0, 1),
               "implementation_choices": CHOICES, "config": config, "config_hash": canonical_hash(config),
               "anchor": {"kind": choice("i"), "arrays_compared": n_anchor, "identical": True,
                          "refsize_manifest": {"file": MANIFEST_REFSIZE, "sha256": man_sha}, **W["anchor_extra"]},
               "input_verification": W["input_verification"],
               "inputs": dict(_RESULT_READS),
               "input_files_sha256": {**dict(_RESULT_READS), rs["npz_rel"]: rs["npz_sha"], MANIFEST_REFSIZE: man_sha,
                                      **W["extra_files"]},
               "input_cache_sha256": W["input_cache_sha256"], "input_cache_source": W["input_cache_source"],
               "caches_rehashed_at_end": True,
               "draws_npz": {"file": rel(npz_path), "sha256": sha256_bytes(npz_bytes), "size": len(npz_bytes),
                             "keys": NPZ_KEYS},
               "population_group": {"declared_in": None, "note": choice("n")},
               "cells": cells}
    if phase == 2:
        payload["v2_gather"] = W["v2_gather"]
    problems = input_identity_problems(payload)
    if problems:
        refuse(f"to write (D-297 (1) results-writer rule): {problems}")
    TP2.write_outputs([(npz_path, npz_bytes), (out_path, json_bytes_of(payload))])
    log(f"elapsed {payload['elapsed_s']} s; {len(cells)} cells; anchor arrays identical: {n_anchor}")
    return out_path


# ----------------------------------------------------------------------------- the NAS bytes (preflight)
def npz_upper_bound(n_cells: int, n_parts: int = N_DRAWS + 1, n_seeds: int = len(SEEDS)) -> int:
    """The bytes of one job's npz WITHOUT compression for n_cells cells (np.savez of zero arrays of the job's exact
    shapes and dtypes; savez_compressed writes at most this plus a few bytes per member)."""
    arrays: dict[str, np.ndarray] = {}
    for c in range(n_cells):
        L = f"L{c}|"
        arrays[L + "T_d"] = np.zeros((n_parts, n_seeds), "<f8")
        arrays[L + "k_pre"] = np.zeros((n_parts, n_seeds), "<i4")
        arrays[L + "frozen|k_post"] = np.zeros((n_parts, n_seeds), "<i4")
        arrays[L + "frozen|ratio"] = np.zeros(n_parts, "<f8")
        for m in METHODS:
            arrays[L + f"{m}|thr"] = np.zeros((n_parts, n_seeds), "<f8")
            arrays[L + f"{m}|k_post"] = np.zeros((n_parts, n_seeds), "<i4")
            arrays[L + f"{m}|ratio"] = np.zeros(n_parts, "<f8")
        arrays[L + "n_eval"] = np.asarray(0, "<i8")
    arrays["layers"] = np.zeros(n_cells, "<i8")
    arrays["ref_rows_frozen"] = np.zeros(N_REF, "<i8")
    return len(savez_bytes(arrays, compressed=False))


# ----------------------------------------------------------------------------- preflight
def cache_files_p1(b: dict[str, Any]) -> list[tuple[str, Path, str]]:
    """(registry cache name, path, expected sha256) of the six Phase 1 caches a job reads (input_registry.load's)."""
    out = []
    for side in ("pre", "post"):
        for part in ("probe_train", "threshold_calib", "eval"):
            name = REG.cache_name(b["cfg"][f"{side}_repo"], b["axis"], part)
            want, _ = REG.expected_sha256(name)
            out.append((name, resolve_cache(name, REG.CACHE), want))
    return out


def cache_files_p2(b: dict[str, Any], plan: dict[str, dict[str, Any]]) -> list[tuple[str, Path, str]]:
    """(cache key, path, expected sha256 from the binding's record entry) of the seven Phase 2 caches a job reads."""
    out = []
    for k, e in plan.items():
        ent = b["entries"].get(k) or {}
        want = ent.get("cache_sha256")
        if ent.get("cache_file") != e["file"] or not isinstance(want, str) or len(want) != 64:
            refuse(f"{k}: the binding's entry names no sha256 for {e['file']}")
        out.append((k, SP.CACHE / e["file"], want))
    return out


def run_preflight(hash_caches: bool = True, phases: tuple[int, ...] = (1, 2), log: Callable[[str], None] = print) -> int:
    """Records, selection, presence (and sha256) of every cache, NAS bytes. Reads no cache array; writes nothing."""
    code0 = P.code_block(CODE_FILES)
    bad: list[str] = [f"final-name gate: {x}" for x in final_problems(True)]
    for x in bad:
        log(x)
    listed, man_sha = refsize_manifest()
    log(f"refsize manifest {MANIFEST_REFSIZE} ({man_sha[:12]}): {len(listed)} files")
    total_bytes = 0
    hashed: dict[str, bool] = {}
    if 2 in phases:
        # review 2 (D-433): D-361's anchor record as a final name reads it (a record and host check; once, not per pair)
        try:
            TP2.passing_anchor(RS.tp2_code_block(code0), final=True)
            log("Phase 2: D-361's anchor record passes for a final name")
        except SystemExit as e:
            bad.append(f"Phase 2 anchor: {e}")
            log(f"Phase 2 anchor: FAILED: {e}")
    for phase in phases:
        sel = phase_selection(phase, listed)
        chk = check_against_item9(phase, sel)
        log(f"Phase {phase}: {chk['n_cells']} cells of {chk['n_pairs']} pairs within x{WITHIN:g}, equal to {ITEM9_REL}")
        for pair, r in sel.items():
            layers = [li for li, _ in r["selected"]]
            if not layers:
                log(f"  {pair}: no selected cell (no job)")
                continue
            try:
                if phase == 1:
                    W = world_p1(pair, r, layers, N_DRAWS, True, code0, log)
                    files = cache_files_p1(W["b"])
                else:
                    # review 2 (D-433): the FINAL-name record checks (p2auroc_d424.p2_final_problems, D-361's
                    # registered parameters) run here too, records only, so a refusal stops the lane at job 2 and not
                    # after nine Phase 1 jobs
                    W = world_p2(pair, r, layers, N_DRAWS, True, True, code0, log)
                    files = cache_files_p2(W["b"], W["plan"])
            except SystemExit as e:
                bad.append(f"Phase {phase} {pair}: {e}")
                log(f"  {pair}: FAILED: {e}")
                continue
            nb = npz_upper_bound(len(layers))
            total_bytes += nb
            msgs = []
            n_ok = 0
            for name, p, want in files:
                if p.is_symlink() or not p.is_file():
                    bad.append(f"{pair}: {name} absent ({p})")
                    msgs.append(f"{name}: ABSENT")
                    continue
                size = p.stat().st_size
                if hash_caches:
                    if str(p) not in hashed:
                        hashed[str(p)] = sha256_file(p) == want
                    if not hashed[str(p)]:
                        bad.append(f"{pair}: {name} is not the bytes its record names")
                        msgs.append(f"{p.name} {size} OTHER BYTES")
                        continue
                n_ok += 1
                msgs.append(f"{p.name} {size}")
            log(f"  {pair}: cells {layers}; {n_ok} of {len(files)} caches present"
                f"{' and hash-verified' if hash_caches else ''}; npz upper bound {nb} bytes")
            for m in msgs:
                log(f"      {m}")
    log(f"NAS BYTES: the {sum(1 for _ in phases)} phase(s)' npz at most {total_bytes} bytes uncompressed (from the array "
        "shapes; savez_compressed writes less), plus one JSON record per job and the summary")
    if bad:
        log(f"PREFLIGHT FAILED ({len(bad)}):")
        for x in bad:
            log(f"  {x}")
        return 1
    log("PREFLIGHT OK" + ("" if hash_caches else " (presence only; caches not hashed)"))
    return 0


# ----------------------------------------------------------------------------- the summary
def read_npz_of(doc: dict[str, Any], rd: Path, owner: str) -> tuple[dict[str, np.ndarray], str]:
    dz = doc.get("draws_npz") or {}
    p = rd / Path(str(dz.get("file"))).name
    if p.is_symlink() or not p.is_file():
        refuse(f"{p.name}, named by {owner}, is missing")
    b = p.read_bytes()
    h = sha256_bytes(b)
    if h != dz.get("sha256") or len(b) != dz.get("size"):
        refuse(f"{p.name} is not the npz {owner} names ({h[:12]} vs {str(dz.get('sha256'))[:12]})")
    with np.load(io.BytesIO(b), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}, h


def verify_record(doc: dict[str, Any], z: dict[str, np.ndarray], owner: str) -> int:
    """Every stored ratio recomputed from the stored counts, and every cell block recomputed from the stored arrays;
    refuses a difference. Returns the number of cells verified."""
    n = 0
    for c in doc.get("cells") or []:
        li = int(c["layer"])
        L = f"L{li}|"
        n_ev = int(z[L + "n_eval"])
        for k in ["frozen", *METHODS]:
            if not RS.same_arrays(ratio_from_counts(z[L + "k_pre"], z[L + f"{k}|k_post"], n_ev), z[L + f"{k}|ratio"]):
                refuse(f"{owner} L{li} {k}: the stored ratio is not the one its counts give")
        arr = {k[len(L):]: v for k, v in z.items() if k.startswith(L)}
        mine = cell_block(arr, li, c["selection"]["d156_mean_realised_paired_ratio"],
                          {"L_sel": doc["config"]["selected_layer"], "C": c["probe_C"]})
        mine["probe_fit_diagnostics"] = c.get("probe_fit_diagnostics")
        if canon(mine) != canon(c):
            refuse(f"{owner} L{li}: the cell block is not the one its npz gives")
        n += 1
    return n


def agg(vals: list[float]) -> list[float] | None:
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    return [float(v.min()), float(np.median(v)), float(v.max())] if v.size else None


def summary_phase(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_cells": len(rows), "n_pairs": len({r["pair"] for r in rows}),
                           "form": "[min, median, max] over cells of each cell's percentile over its draws"}
    for arm in ["frozen", *METHODS]:
        out[arm] = {f"p{int(q)}": agg([r[arm]["p5_p50_p95"][i] if r[arm]["p5_p50_p95"] else None for r in rows])
                    for i, q in enumerate(PERCENTILES)}
        out[arm]["d156_mean"] = agg([r[arm]["d156_mean"] for r in rows])
        out[arm]["share_of_draws_two_sided_2x_mean_over_cells"] = float(np.mean(
            [r[arm]["share_of_draws_two_sided_2x"] for r in rows])) if rows else None
        out[arm]["share_of_draws_outside_x1.5_mean_over_cells"] = float(np.mean(
            [r[arm]["share_of_draws_outside_x1.5"] for r in rows])) if rows else None
    return out


def run_summary(out: str | Path | None = None, records_dir: str | Path | None = None,
                log: Callable[[str], None] = print) -> Path:
    t0 = time.time()
    final_path = (RESULTS / "recalnoise_d433_summary.json").resolve()
    out_path = Path(out).resolve() if out else final_path
    final = out_path == final_path
    rd = Path(records_dir).resolve() if records_dir else RESULTS.resolve()
    if not final and out_path.is_relative_to(RESULTS.resolve()):
        refuse("a test or sandbox summary (--out) must not write under results/")
    if final and rd != RESULTS.resolve():
        refuse("the final summary reads the records under results/ only")
    if out_path.exists() or out_path.is_symlink():
        refuse(f"{out_path} exists; results are never overwritten (D-152 P5)")
    check_final_gates(final, True)
    code0 = P.code_block(CODE_FILES)
    listed, man_sha = refsize_manifest()
    phases: dict[str, Any] = {}
    records: dict[str, dict[str, str]] = {}
    missing: list[str] = []
    all_registered = True
    flags: list[str] = []
    for phase in (1, 2):
        sel = phase_selection(phase, listed)
        chk = check_against_item9(phase, sel)
        rows: list[dict[str, Any]] = []
        for pair, r in sel.items():
            layers = [li for li, _ in r["selected"]]
            if not layers:
                continue
            stem = stem_of(phase, pair)
            p = rd / f"recalnoise_d433_p{phase}_{stem}.json"
            if p.is_symlink() or not p.is_file():
                missing.append(p.name)
                continue
            doc = read_result(p)
            owner = p.name
            cfg = doc.get("config") or {}
            probs = []
            if doc.get("run_kind") != RUN_KIND[phase] or cfg.get("pair") != pair:
                probs.append("not this module's record of this pair")
            if (doc.get("code") or {}).get("sha256") != code0["sha256"]:
                probs.append("it was written by other code")
            if [c.get("layer") for c in doc.get("cells") or []] != layers:
                probs.append(f"its cells are not the selected layers {layers}")
            if (cfg.get("refsize_record") or {}).get("sha256") != r["sha"]:
                probs.append("it anchored on another refsize record")
            if final and doc.get("registered_parameters") is not True:
                probs.append("not a run of the registered parameters")
            if probs:
                refuse(f"{owner}: {probs}")
            all_registered &= doc.get("registered_parameters") is True
            flags += [f"{owner}: {d}" for d in (doc.get("status") or {}).get("d_entries") or []]
            z, zsha = read_npz_of(doc, rd, owner)
            verify_record(doc, z, owner)
            records[owner] = {"file": rel(p), "sha256": _RESULT_READS[_repo_rel(p)],
                              "npz": (doc.get("draws_npz") or {}).get("file"), "npz_sha256": zsha}
            for c in doc["cells"]:
                row = {"pair": pair, "layer": c["layer"], "is_selected_layer": c["is_selected_layer"],
                       "d156_mean_realised_paired_ratio": c["selection"]["d156_mean_realised_paired_ratio"]}
                for arm in ["frozen", *METHODS]:
                    blk = c["frozen"] if arm == "frozen" else c["recalibrated"][arm]
                    pr = blk["paired_ratio"]
                    row[arm] = {"p5_p50_p95": pr["p5_p50_p95"], "d156_mean": pr["d156"]["mean"],
                                "frozen_split": pr["frozen_split"],
                                "share_of_draws_two_sided_2x": pr["share_of_draws_two_sided_2x"],
                                "share_of_draws_outside_x1.5": pr["share_of_draws_outside_x1.5"]}
                rows.append(row)
        agg_ = summary_phase(rows)
        # the frozen percentiles' medians over cells must be D-430 (9)'s (the same arrays, the same estimator)
        item9 = ((read_result(ROOT / ITEM9_REL).get("populations") or {}).get(ITEM9_POP[phase]) or {}).get("summary") or {}
        cross = None
        if rows and len(rows) == chk["n_cells"]:
            want = {"p5": item9.get("frozen_p5_over_cells_min_median_max"),
                    "p95": item9.get("frozen_p95_over_cells_min_median_max")}
            have = {"p5": agg_["frozen"]["p5"], "p95": agg_["frozen"]["p95"]}
            if canon(want) != canon(have):
                refuse(f"Phase {phase}: the frozen ratio's percentiles over cells are not {ITEM9_REL}'s: {have} vs {want}")
            cross = {"file": ITEM9_REL, "frozen_p5_and_p95_min_median_max_equal": True}
        phases[POPULATION[phase]] = {"selection": chk, "summary": agg_, "cross_check_item9": cross, "cells": rows}
    if missing and final:
        refuse(f"records missing: {missing}")
    if P.code_block(CODE_FILES)["sha256"] != code0["sha256"]:
        refuse("source changed during the run")
    registered = all_registered and not missing
    payload = {**provenance_block(RUN_KIND["summary"]), "preregistration": PREREG, "code": code0,
               "reused_files_sha256": reused_now(), "registered_parameters": registered,
               "status": status_block(registered, list(dict.fromkeys(flags))),
               "elapsed_s": round(time.time() - t0, 1), "implementation_choices": CHOICES,
               "config": {"methods": dict(METHOD_TEXT), "n_reference": N_REF, "target_fpr": TARGET,
                          "percentiles": list(PERCENTILES), "records_dir": str(rd) if not final else "results",
                          "refsize_manifest": {"file": MANIFEST_REFSIZE, "sha256": man_sha}},
               "records": records, "records_missing": missing, "inputs": dict(_RESULT_READS),
               "input_files_sha256": {**dict(_RESULT_READS), MANIFEST_REFSIZE: man_sha},
               "phases": phases}
    problems = input_identity_problems(payload)
    if problems:
        refuse(f"to write (D-297 (1) results-writer rule): {problems}")
    TP2.write_outputs([(out_path, json_bytes_of(payload))])
    for pop, ph in phases.items():
        s = ph["summary"]
        log(f"{pop}: {s['n_cells']} cells of {s['n_pairs']} pairs; frozen p5/p95 medians {s['frozen']['p5']} "
            f"{s['frozen']['p95']}")
    return out_path


# ----------------------------------------------------------------------------- price (synthetic, writes nothing)
def run_price(n_draws: int = 50, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Times cell_arrays on SYNTHETIC scores of each phase's shapes (Phase 1: 1000 + 2000 pooled negatives; Phase 2:
    5000 + 6000), five seeds, n_draws draws plus the frozen split; extrapolates per cell to 1000 draws. Arithmetic only:
    the probe fits and cache reads of a job are not in it."""
    rng = np.random.default_rng(20261010)
    out: dict[str, Any] = {}
    for name, (n_cal, n_ev) in {"phase1": (1000, 2000), "phase2": (5000, 6000)}.items():
        pre = rng.standard_normal((n_cal + n_ev, len(SEEDS)))
        post = 1.1 * pre + 0.2 + 0.1 * rng.standard_normal(pre.shape)
        draws = P.repartition_draws(n_cal, n_ev, n_draws, 1)
        parts = [(np.arange(n_cal), np.arange(n_cal, n_cal + n_ev))] + [(d[:n_cal], d[n_cal:]) for d in draws]
        perm = RS.ref_permutations(n_cal, n_draws, 2)
        t = time.time()
        cell_arrays(pre, post, parts, perm)
        dt = time.time() - t
        out[name] = {"seconds": round(dt, 3), "draws": n_draws + 1,
                     "seconds_per_cell_at_1001_draws": round(dt * (N_DRAWS + 1) / (n_draws + 1), 1)}
        log(f"{name}: {dt:.2f} s for {n_draws + 1} partitions; about {out[name]['seconds_per_cell_at_1001_draws']} s a "
            "cell at 1001 (arithmetic only, this machine)")
    return out


# ----------------------------------------------------------------------------- the lane list
def job_cells() -> dict[int, dict[str, list[int]]]:
    """The selected layers of every pair with a selected cell, per phase, from the committed refsize records (bound by
    their manifest)."""
    listed, _ = refsize_manifest()
    return {ph: {pair: [li for li, _ in r["selected"]] for pair, r in phase_selection(ph, listed).items() if r["selected"]}
            for ph in (1, 2)}


def job_pairs() -> dict[int, list[str]]:
    return {ph: list(c) for ph, c in job_cells().items()}


# the cache bytes the jobs read, MEASURED by a read-only stat of data/hidden on the node (2026-10-10, about 08:00Z;
# the D-433 session's check, the file names from input_registry.cache_name and p2auroc_d424.cache_plan)
CACHE_BYTES_NODE = {"phase1": {"read_by_the_jobs": 42873905364, "unique_files": 48, "unique_bytes": 38118725049},
                    "phase2": {"read_by_the_jobs": 58872394355, "unique_files": 47, "unique_bytes": 48380150029}}
# the per-pair elapsed seconds of the records whose load-and-fit path these jobs repeat (their 'elapsed_s')
ELAPSED_OF_RECORD_S = {"refsize_d382_p1": (395.2, 561.2), "refsize_d382_p2": (418.0, 473.8),
                       "p2auroc_d424": (414.4, 505.8)}
# a read-only timing on the node in a scratch dir (2026-10-10 08:12Z, load average 25.4 with four lanes running):
# sha256 of one 0.95 GB Phase 1 cache, and the five selected-C probe fits of one Phase 1 layer (llama3_to_31, L2)
NODE_TIMING = {"sha256_951190620_bytes_s": 2.0, "five_probe_fits_one_p1_layer_s": 28.7}


def job_outputs(phase: int, pair: str) -> str:
    stem = stem_of(phase, pair)
    return f"results/recalnoise_d433_p{phase}_{stem}.json results/recalnoise_d433_p{phase}_{stem}_draws.npz"


def sandbox_dir(date: str) -> str:
    return f"{NODE_HOME}/.cache/gccf_scratch/recalnoise433_{date}"


def lane_jobs(date: str, pairs: dict[int, list[str]]) -> list[tuple[str, str, str]]:
    jobs = [("recalnoise_d433_sandbox", "-", f"bash scripts/sandbox_d433.sh {sandbox_dir(date)}"),
            ("recalnoise_d433_preflight", "-", "python3 -u src/recalnoise_d433.py preflight")]
    for ph in (1, 2):
        for pair in pairs[ph]:
            jobs.append((f"recalnoise_d433_p{ph}_{stem_of(ph, pair)}", job_outputs(ph, pair),
                         f"python3 -u src/recalnoise_d433.py job --phase {ph} --pair {pair}"))
    jobs.append(("recalnoise_d433_summary", "results/recalnoise_d433_summary.json",
                 "python3 -u src/recalnoise_d433.py summary"))
    return jobs


def lane_list_text(date: str, cells: dict[int, dict[str, list[int]]]) -> str:
    pairs = {ph: list(c) for ph, c in cells.items()}
    n1, n2 = len(pairs[1]), len(pairs[2])
    nb = sum(npz_upper_bound(len(lay)) for c in cells.values() for lay in c.values())
    n_cells = sum(len(lay) for c in cells.values() for lay in c.values())
    c1, c2 = CACHE_BYTES_NODE["phase1"], CACHE_BYTES_NODE["phase2"]
    e1, e2, e3 = (ELAPSED_OF_RECORD_S[k] for k in ("refsize_d382_p1", "refsize_d382_p2", "p2auroc_d424"))
    head = [
        "# Lane D (CPU only, no network, no GPU; D-433, the author's item 6 of 2026-10-10, PRE-REGISTERED, DESCRIPTIVE): "
        "RECALIBRATION NOISE on 200 trusted prompts by src/recalnoise_d433.py. On D-430 (9)'s cells (the frozen "
        "threshold's realised paired ratio within x1.5: 36 Phase 1 cells of 9 pairs, 41 Phase 2 cells of 8 pairs), per "
        "D-156 draw of the refsize_d382 records: the threshold recalibrated on the draw's 200 D-382 reference prompts by "
        "each method the paper reports (reset, quantile transfer, z-score transfer); the 5-95% range of the realised "
        "paired ratio beside the frozen one. From existing caches only.",
        "# DO NOT LOAD before: the D-433 entry committed, the review and its fix round, and the push that carries the "
        "module, its tests, scripts/sandbox_d433.sh and this list. ORDER: (0) the SANDBOX (tests; one real job per "
        f"phase, {SANDBOX_P1} and {SANDBOX_P2}, plus {SANDBOX_P1_WG} for the WildGuard axis's load path, at --n-draws "
        f"{SANDBOX_DRAWS} into a fresh scratch dir; the preflight "
        "without hashing; the summary over the three sandbox records; outputs discarded; a failure stops the lane: F-081/"
        "F-082, D-400), (1) the preflight (records, the 36 + 41 cells against docs/reports_d430/item9_opcase.json, every "
        f"cache present and hash-verified, the NAS bytes), (2) {n1} Phase 1 jobs, (3) {n2} Phase 2 jobs, (4) the summary "
        "(results/recalnoise_d433_summary.json). No edit to src/recalnoise_d433.py or its code files between the jobs "
        "and the summary (the summary refuses records written by other code).",
        "# Final names need 1000 draws, seeds 0-4, 1%, the 6-thread BLAS pin (scripts/project_env.sh), GUARDRAIL_CACHE_DIRS "
        "unset and the reused files at their pinned bytes (REUSED_SHA256; src/refsize_d382.py is pinned, never edited). "
        "Runs from the repo root through scripts/detach.sh, on the NODE; chained behind D-423's lane D list (or on the "
        "first CPU lane that frees, as the loading session decides; the list is lane-agnostic but for its name).",
        f"# PRICE (an ESTIMATE, not a measurement): the caches are on the node under data/hidden (stat of 2026-10-10): "
        f"the {n1} Phase 1 jobs read {c1['read_by_the_jobs'] / 1e9:.1f} GB (six caches a job; {c1['unique_files']} "
        f"unique files, {c1['unique_bytes'] / 1e9:.1f} GB), the {n2} Phase 2 jobs {c2['read_by_the_jobs'] / 1e9:.1f} GB "
        f"(seven a job; {c2['unique_files']} unique, {c2['unique_bytes'] / 1e9:.1f} GB); each job hashes its caches at "
        "load and again at the end, and the preflight hashes every unique file once. Per job at most about the "
        f"per-pair time of the records whose load-and-fit path it repeats (refsize_d382 Phase 1 {e1[0]:.0f}-{e1[1]:.0f} "
        f"s, Phase 2 {e2[0]:.0f}-{e2[1]:.0f} s, p2auroc_d424 {e3[0]:.0f}-{e3[1]:.0f} s, every layer of a pair; these "
        "jobs fit the selected layers only, and the per-cell arithmetic is seconds). A node timing at load 25 "
        f"(2026-10-10): five probe fits of one Phase 1 layer {NODE_TIMING['five_probe_fits_one_p1_layer_s']} s, sha256 of "
        f"a 0.95 GB cache {NODE_TIMING['sha256_951190620_bytes_s']} s. So about {n_cells} cells (plus {n2} L* anchor "
        "fits in Phase 2) at about half a minute each, the hashing and the sandbox: about 1-2 h for the whole list "
        "on a quiet node, about 2-4 h under the F-084 contention of three other lanes (review 2: F-084 measured about "
        "3.5x). NAS: the npz at "
        f"most {nb} bytes uncompressed in all (from the array shapes; written compressed) plus {n1 + n2 + 1} JSON "
        "records.",
    ]
    lines = head + ["\t".join(j) for j in lane_jobs(date, pairs)]
    return "\n".join(lines) + "\n"


def run_queue(write: bool = False, date: str | None = None, root: Path | None = None) -> str:
    date = date or datetime.now(timezone.utc).strftime("%Y%m%d")
    if len(date) != 8 or not date.isdigit():
        refuse(f"--date must be YYYYMMDD, not {date!r}")
    text = lane_list_text(date, job_cells())
    labels = [ln.split("\t")[0] for ln in text.splitlines() if ln and not ln.startswith("#")]
    if len(labels) != len(set(labels)):
        refuse("duplicate labels in the list")
    for ln in text.splitlines():
        if ln and not ln.startswith("#") and (len(ln.split("\t")) != 3 or any(not f.strip() for f in ln.split("\t"))):
            refuse(f"a list line is not three non-empty TAB-separated fields: {ln[:80]!r}")
    p = (root or ROOT) / "queues" / f"laneD_{date}_recalnoise_d433.tsv"
    if write:
        if p.exists():
            refuse(f"{p} exists; a list is never overwritten")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode("utf-8"))
        print(f"Written: {rel(p)} ({len(labels)} jobs)")
    else:
        sys.stdout.write(text)
    return text


# ----------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="D-433: recalibration noise on 200 trusted prompts (descriptive)")
    sub = ap.add_subparsers(dest="mode", required=True)
    pf = sub.add_parser("preflight", help="records, selection, caches present (and hashed), NAS bytes; writes nothing")
    pf.add_argument("--no-hash", action="store_true", help="presence and sizes only (the sandbox's preflight)")
    pf.add_argument("--phase", type=int, choices=(1, 2), default=None)
    jb = sub.add_parser("job", help="one pair's selected cells (or every pair of the phase)")
    jb.add_argument("--phase", type=int, choices=(1, 2), required=True)
    jb.add_argument("--pair", default=None)
    jb.add_argument("--out", default=None, help="TESTING / SANDBOX: a .json path outside results/")
    jb.add_argument("--n-draws", type=int, default=N_DRAWS, help="TESTING / SANDBOX: draws 0..N (a prefix)")
    jb.add_argument("--records-only", action="store_true", help="bind the records and stop (no cache read)")
    sm = sub.add_parser("summary", help="results/recalnoise_d433_summary.json from the job records")
    sm.add_argument("--records-dir", default=None)
    sm.add_argument("--out", default=None)
    pr = sub.add_parser("price", help="time the per-cell arithmetic on synthetic arrays (writes nothing)")
    pr.add_argument("--n-draws", type=int, default=50)
    qu = sub.add_parser("queue", help="the lane list")
    qu.add_argument("--write", action="store_true")
    qu.add_argument("--date", default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdout()
    a = build_parser().parse_args(argv)
    if a.mode == "preflight":
        return run_preflight(hash_caches=not a.no_hash, phases=(a.phase,) if a.phase else (1, 2))
    if a.mode == "job":
        if a.pair is None:
            if a.out is not None:
                refuse("--out needs --pair (one record per pair)")
            for pair in job_pairs()[a.phase]:
                run_job(a.phase, pair, a.n_draws, None, a.records_only)
            return 0
        run_job(a.phase, a.pair, a.n_draws, a.out, a.records_only)
        return 0
    if a.mode == "summary":
        run_summary(a.out, a.records_dir)
        return 0
    if a.mode == "price":
        run_price(a.n_draws)
        return 0
    if a.mode == "queue":
        run_queue(a.write, a.date)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
