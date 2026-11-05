"""C5 health-check baselines (D-381, amended by D-392): KS and ATC beside 4c's Gaussian ratio, BBSE beside as a ranking
line; one pair (or LoRA run) and target per job, and their pooled scoring (--summary).

Pre-registered in D-381 before any of this code or its numbers existed (the author's step-4 item 1; details by the
closest D-entries, D-181: D-152 4c, D-241, D-248, D-255, D-264, D-278, D-156, D-164, D-177 (b)). Question: do standard
label-free shift detectors find the cells whose frozen threshold fails, as 4c's Gaussian health ratio does? No reviewed
module is edited: 4c's functions are called (healthcheck_d156.thresholds, health, crosses, core_of; paired.d117_split,
fit_probe, repartition_draws, cluster_units, cluster_repartition_draws, ratio_from_flags, repartition_estimate,
band_summary) and 4c's pooling estimator (healthcheck_summary_d156.pooled_counts, pair_cluster_index, unchanged); where
4c computes inside its main() the lines are repeated here and pinned by the anchor (3).

UNITS (D-381 (1)): the nine released pairs, the GPU-change control (its own row) and the nine LoRA runs (D-278), each at
1% (primary) and 5% (beside; new for the LoRA runs, D-248): 38 jobs. Every layer cell of the pair's grid; selected C
(D-117, permutation order) primary, C=1 (file order) beside; the item level (D-156) primary, the D-164 cluster level
beside; 1000 draws plus the frozen split (index 0); seeds 0-4; probes fitted exactly as 4c fits them.
  REFERENCE   each draw's calibration part cal_d (1000 negatives at both levels, D-164 fills it exactly), scored by the
              pre model and by the post model through the pre model's frozen probes (4c's pre_pool and post_pool rows
              of cal_d, D-241 (1)).
  LABELLED    ATC's fit set: cal_d's negatives and the 1000 threshold_calib positives (fixed in every draw: D-156
              re-partitions the negatives only), scored by the PRE model (D-381 (2)(b)).
  KS (2)(a)   per seed the two-sample KS statistic D between cal_d's pre and post scores (arXiv:1810.11953, BBSDs +
              KS, on the probe score); the cell's score is the mean of D over seeds (ROC score); the flag is
              mean D > 1.358 sqrt((n + m) / (n m)), n = m = |cal_d| (alpha 0.05, asymptotic). One flag per draw,
              scored against every class as the AUC rule's is. Stated, not corrected (D-381): the two samples are the
              same prompts, which the test's independence premise does not cover.
  ATC (2)(b)  (arXiv:2201.04234) at the deployed operating point, per seed: confidence max(p, 1 - p), p = sigmoid(s - T_d);
              t fitted on the pre labelled calibration part so that the share of its items with confidence below t
              equals its pre error rate at T_d (negatives at or above T_d, positives below it); a model's estimate =
              the share of cal_d below t under that model, floored at 1/n; ratio = post / pre per seed, then the mean
              over seeds (D-241 (2)); flag when the ratio crosses the class's cut (healthcheck_d156.crosses, as the
              Gaussian ratio does); ROC score |log ratio|.
  BBSE (2)(c) (arXiv:1802.03916), as amended by D-392 (1): computed BESIDE as a ranking line, no flag (no registered
              cut), not in P-C5.3. Per cell, draw and seed the estimated harmful share on the benign reference
              w = (r_post - r_pre) / (TPR_pre - r_pre): r = the share of cal_d at or above T_d under a model, floored at
              1/n (4c's direct ratio's r); TPR_pre = the share of the 1000 calibration positives at or above T_d under
              the pre model (unfloored, on the calibration positives: ATC's fit set, BBSE's labelled source confusion
              matrix); undefined when TPR_pre <= r_pre; the cell's smallest defined TPR_pre - r_pre is stored
              (bbse_min_denominator); the cell's value is the
              mean over the seeds with a defined estimate, the undefined seeds counted, and a cell with none is left
              out of that draw's ROC AUC (CHOICE 16); the summary's ROC AUC of |w| across a row's cells (D-156 mean and
              band). Within one cell, draw and seed w is increasing affine in r_post (bbse_harmful_share,
              bbse_matching_ratio_cut); across a row's cells it is not a function of the direct ratio (TPR_pre differs by
              cell, the direct ratio is a mean over seeds), so its ranking can differ (D-392 (1)). Its premise, p(x | y)
              unchanged, is what an update breaks (BBSE_TEXT).
  CLASSES     4c's (healthcheck_d156.CLASSES), from the realised paired ratio on the draw's evaluation part.
  ANCHOR (3)  per pair and target the job recomputes 4c's per-draw arrays (z_mean, log_sd_ratio, gaussian_ratio,
              log_gaussian_ratio, direct_ratio, realised_paired_ratio) with healthcheck_d156's functions and refuses,
              before anything is written, unless every one equals the pair's 4c npz byte for byte
              (results/healthcheck_d156_<pair>[_fpr5]_draws.npz, bound by its manifest and by its record's sha256),
              and, with every layer, unless the Gaussian and direct confusion counts summed over cells equal the npz's.
              LoRA at 5% (no 4c record exists, D-278): CHOICE 9.
OUTPUTS (written once: staged, hard-linked to the final name, read back, then mode 444; never overwritten):
  --pair <stem> [--fpr5]   results/healthcheck_baselines_d381_<core>[_fpr5].json and its _draws.npz (D-163 (6)): per
                           cell and level the baselines' D-156 blocks (BBSE's beside), flag shares and class shares; per probe set and
                           level the per-draw confusion counts of KS, ATC, Gaussian and direct against the three classes
                           summed over the pair's cells (NPZ_KEYS).
  --summary                results/healthcheck_baselines_d381_summary.json from the 38 records named by
                           logs/healthcheck_baselines_d381.sha256: per target, probe set and level the rows of
                           healthcheck_summary_d156 (released pairs, unflagged pairs as the 4c summary lists them, GPU
                           control; LoRA all, general, safety) for KS and ATC, BBSE's ROC AUC beside, the 4c rows beside,
                           P-C5.1-3 scored; a per-record block (file, sha256, commit, config hash, anchor counts, the
                           record's D-170 flag beside its 4c record's) in its config.
Final names need the registered parameters (1000 draws, seeds 0-4, every layer), the 6-thread BLAS pin
(scripts/project_env.sh), GUARDRAIL_CACHE_DIRS unset, registered 4c records and the reused files at their reviewed bytes
(REUSED_SHA256); a test writes outside results/.

PREDICTIONS (D-381 (4); scored HELD or FAILED on the released pairs, 1%, selected C, item level, the primary class,
pooled point estimates; every other row beside, never a verdict):
  P-C5.1  KS's pooled specificity is lower than the Gaussian ratio's.
  P-C5.2  ATC's ROC AUC (D-156 mean) is lower than the Gaussian ratio's.
  P-C5.3  no baseline has both pooled sensitivity and pooled specificity at or above the Gaussian ratio's.

CHOICES (D-181; each by the closest D-entry, none changes an estimand or what is primary). Recorded in D-392: 1-3, 5, 6,
8, 9, 12, 14 and CHOICE 4's fit set (BBSE itself, computed beside as a ranking line, is D-392 (1)). Implementation
choices recorded in the review-outcome D-entry of this fix round: 4 (the floor and the per-seed mean of the ATC ratio),
7, 10, 11, 13, 15, 16 and 17.
  1  (D-392 (3)) ATC in margin space: max(p, 1 - p) = sigmoid(|s - T_d|) is strictly increasing in the margin |s - T_d|, so
     "confidence below t" is evaluated as "margin below t"; the same counts, without float64's saturation of the
     sigmoid at 1.0 (|s - T_d| > ~37), which would create ties the margins do not have (D-241 (2): 4c's log-survival,
     "so no underflow").
  2  (D-392 (3)) ATC's t is the (k+1)-th smallest labelled margin (k = the number of pre errors at T_d; +inf when every item is an
     error), so exactly k items lie below it unless a tie sits at t; such ties are counted per cell (atc_fit_ties),
     never resolved silently (paired.thr_at_fpr: the threshold is an observed score with a one-sided count rule).
  3  (D-392 (3)) An error at T_d is a negative scored at or above T_d or a positive below it: 4c's flag rule s >= T_d (D-241 (2)).
  4  (an implementation choice recorded in the review-outcome D-entry of this fix round, following D-241 (2)'s per-seed-
     mean convention; the fit set is D-392 (3)) ATC's floor 1/n uses n = |cal_d| = 1000 at both levels, and the ATC
     ratio is the mean over seeds of per-seed post / pre ratios, as 4c's direct ratio is (D-241 (2), D-164); the fit set
     is the labelled calibration part as registered (1000 negatives, 1000 positives), so the pre error rate is driven by
     the positives missed at T_d (D-392 (3)).
  5  (D-392 (4)) KS's D is the two-sided two-sample statistic max |ECDF_pre - ECDF_post| over the pooled scores, computed exactly
     (integer counts, one rounding; scipy.stats.ks_2samp agrees to float rounding, tested), per seed then the mean
     over seeds (D-241 (2)); n = m = |cal_d|; "exceeds" is strict. D does not depend on T_d: KS's flags at 5% equal
     those at 1% in the same draw, only the classes differ.
  6  (D-392 (4)) KS's single per-draw flag is scored against all three classes, as the AUC rule's single flag is (D-241 (4)).
  7  Per-cell blocks and confusion counts: healthcheck_d156.main's lines (the class and flag shares, the tp/fn/fp/tn
     sums over cells, index 0 the frozen split) repeated with the detectors ks and atc; gaussian and direct by the same
     lines, anchored on the 4c npz's counts (D-255).
  8  (D-392 (5)) The anchor covers all six 4c statistics, a superset of the Gaussian and direct arrays D-381 (3) names: the
     realised paired ratio defines the classes the baselines are scored against (D-241 (3)).
  9  (D-392 (2)) LoRA at 5%: 4c has no 5% LoRA arm (D-278), so no 4c npz exists. The anchor is then 4c's two anchors for a 5% arm:
     (i) D-248's self-anchor: the same probes and partitions recomputed at 1% give every 4c statistic equal, byte for
     byte, to the run's 1% 4c npz; (ii) D-241 (3), D-278 (1): the 5% realised operating point (paired.
     repartition_estimate's report) equals results/depth_d156_<run>_fpr5.json's at every layer, probe set and level,
     the per-draw paired ratios equal its arrays, and that depth file (named by logs/fpr5_stage_depth_20260928.sha256)
     is complete, self-anchored and anchored on the 1% depth record the 1% 4c record anchored on. These records also
     carry the 4c statistics at 5% (no other file holds them); the summary's 4c rows for LoRA at 5% are computed from
     them and labelled so.
  10 Summary rows as healthcheck_summary_d156 builds them, per target. The unflagged row is built on the 4c summary's
     own pair list (its unflagged_pairs 'pairs'), so the KS and ATC row and the 4c row beside it cover the same pairs;
     the summary refuses unless that list is well-formed (distinct released pairs in their order, the same in every row
     of that summary, n_pairs its length) and equals the released pairs whose 4c record stored no D-170 flag (each
     record carries that flag, anchor.records.healthcheck_4c.d170_flag). Each record's current flag (its own
     input_verification, which D-331 may have cleared) is carried beside the flag its 4c record stored (the row's
     d170_flags and the per-record block). The row is beside.
  11 ROC AUC per draw across a row's cells: healthcheck_summary_d156.scoring's lines repeated (roc_auc) for the scores
     mean D, |log ATC ratio| and |BBSE w|; the tests show the repeated lines equal scoring's on the same records. A
     cell whose score is not finite in a draw (BBSE with no defined seed) is left out of that draw's AUC, which is the
     AUC over the row's other cells, and the row reports the cells left out (CHOICE 16); a draw then left with no cell
     in a class or none outside it has an undefined AUC, counted (P.band_summary's rule).
  12 (D-392 (5)) The 4c rows beside are read from results/healthcheck_d156_summary{,_fpr5,_lora}.json, pinned (D-255: read, not
     recomputed); the summary refuses unless its own pooling of the records' (anchored) Gaussian and direct counts
     reproduces those files' released-pair, unflagged-pair, GPU-control and LoRA blocks exactly (CHOICE 17 adds the
     ROC lines and the cell counts).
  13 P-C5.1 and P-C5.2 strict ("lower"), P-C5.3 at or above (>=), over KS and ATC (BBSE has no flag and does not
     enter P-C5.3, D-392 (1)); a pooled rate with no item of its class makes holds None with the reason (D-346 (4)).
  14 (D-392 (6)) No D-177 accuracy line: the KS, ATC and BBSE rows are detection rows (flags, ROC), not predictions of the realised ratio
     (D-255 OPEN).
  15 Final-name gates as src/hh_sensitivity_d363.py's (D-363): pinned reused files, GUARDRAIL_CACHE_DIRS unset; and every
     cache's resolved path under the input registry's directory (caches_outside_registry).
  16 (an implementation choice recorded in the review-outcome D-entry of this fix round, following D-156's non-finite
     convention; BBSE itself, beside as a ranking line, is D-392 (1)) BBSE's r and its floor are 4c's direct ratio's
     (D-241 (2)); TPR_pre is unfloored, on the calibration positives at T_d (ATC's fit set, BBSE's labelled source
     confusion matrix); a seed with TPR_pre <= r_pre has no estimate (the 2x2 confusion matrix is not invertible), and
     each cell stores its smallest defined TPR_pre - r_pre (bbse_min_denominator, the estimate's largest slope 1 /
     (TPR_pre - r_pre)). Each row reports the share of cell-draws whose BBSE mean uses fewer than the five seeds
     (bbse_mean_over_fewer_seeds, from bbse_n_undefined_seeds). Per cell and draw the mean over seeds
     is taken over the seeds with a defined estimate and the number of undefined seeds is stored (bbse_n_undefined_seeds
     per draw, bbse_undefined per cell); a cell with no defined seed is NaN in that draw (band_summary counts it) and is
     left out of that draw's ROC AUC; each row reports per draw the number of cells left out (D-156 mean and maximum)
     and the share of draws with any left out (cells_left_out); a draw whose row then has no cell in a class or none
     outside it has an undefined AUC, counted, as for every ROC line.
  17 (an implementation choice recorded in the review-outcome D-entry of this fix round; D-392 (4), (5)) The summary
     also refuses unless: each unit's ks_mean_D arrays at 1% and 5% are byte-identical (D-392 (4): D does not depend on
     T_d); this module's ROC path (roc_auc) applied to the released pairs' 4c scores (their 4c npz, the bytes each record
     anchored on, bound to the 4c manifest) gives the 4c summaries' ROC AUC blocks and cell counts of the released and
     unflagged rows exactly, at both targets (so P-C5.2 compares ATC's ROC AUC with a Gaussian ROC AUC this code
     reproduces); every row's KS and ATC cell count equals the 4c row's (and, before P-C5.2, the primary row's); the
     records' control flags are as expected (healthcheck_summary_d156's check). A final run also refuses a cache whose
     resolved path is not under the input registry's directory (CHOICE 15).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_d156 as D  # noqa: E402
import healthcheck_d156 as HC  # noqa: E402
import healthcheck_summary_d156 as HS  # noqa: E402
import input_registry as REG  # noqa: E402
import metrics as M  # noqa: E402
import paired as P  # noqa: E402
import populations as POP  # noqa: E402
from provenance import (  # noqa: E402
    _RESULT_READS, _repo_rel, canonical_hash, commit_staged, ensure_utf8_stdout, input_identity_problems,
    provenance_block, read_result, resolve_cache, sha256_bytes, sha256_file, stage_bytes, write_results,
)

ROOT = Path(__file__).resolve().parent.parent
CODE_ROOT = Path(__file__).resolve().parent.parent      # where the source files are hashed (ROOT may be a test tree)
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
OWN = "src/healthcheck_baselines_d381.py"
CODE_FILES = list(dict.fromkeys([OWN, *HC.CODE_FILES, *HS.CODE_FILES]))
# the reviewed files whose functions are reused, at the bytes this module was written and anchored against (master
# fb329e0); a final name refuses other bytes
REUSED_SHA256 = {
    "src/healthcheck_d156.py": "0ec2687ecdd887a28d4b8fb7a4db4b9081b69ea3fcf2b175c345279b4ed74de7",
    "src/healthcheck_summary_d156.py": "b149b66181fe1c2d953d255355e44e2d15651bff9ccf8a456819092db526ccf5",
    "src/depth_d156.py": "a1f7b50ce70395f290b681f336e5a5e95e1428a2ca7c5915818644ad70f26734",
}
RUN_KIND = "healthcheck_baselines_d381"
PREREG = ("D-381 (the author's step-4 item 1, 2026-10-07); details by the closest D-entries (D-181): D-152 4c, D-241, "
          "D-248, D-255, D-264, D-278, D-156, D-164, D-177 (b)")
LEVELS, PROBE_SETS, CLASSES = HC.LEVELS, HC.PROBE_SETS, HC.CLASSES
COUNTS = HS.COUNTS
PARTS = ("probe_train", "threshold_calib", "eval")
REGISTERED = {"n_draws": P.N_DRAWS, "seeds": [0, 1, 2, 3, 4]}
TARGET_1, TARGET_5 = P.TARGET_FPR, 0.05
HEALTH = ("z_mean", "log_sd_ratio", "gaussian_ratio", "log_gaussian_ratio", "direct_ratio")    # healthcheck_d156.health
REALISED = "realised_paired_ratio"
FOURC_STATS = (*HEALTH, REALISED)                       # == healthcheck_d156.STATS, the anchored arrays
BASE_ARRAYS = ("ks_mean_D", "atc_ratio", "atc_est_pre", "atc_est_post", "atc_error_rate_pre", "bbse_harmful_share",
               "bbse_tpr_pre", "bbse_n_undefined_seeds")
BASELINES = ("ks", "atc")
FOURC_DETECTORS = ("gaussian", "direct")                # recomputed by 4c's lines, anchored on its counts
DETECTORS = (*BASELINES, *FOURC_DETECTORS)
KS_ALPHA, KS_C_ALPHA = 0.05, 1.358                      # D-381 (2)(a): the asymptotic two-sample value at alpha 0.05
PRIMARY = {"target_fpr": TARGET_1, "probe_set": "selected", "level": "item_level_D156", "class": "two_sided_2x",
           "row": "released_pairs"}
CONTROL = HC.CONTROL
CONTROL_CORE = HC.core_of(CONTROL)
# the summary's rows, by core name as healthcheck_summary_d156 names them (rows of HS.main and HS.lora_summary)
RELEASED_CORES = [HC.core_of(s) for s in POP.UPDATE_PAIRS_REPAIR]
LORA_GROUPS = {"lora_general": list(POP.LORA_ARMS["general (alpaca)"]),
               "lora_safety": [r for arm in ("safety (hh-harmless)", "safety (pku-saferlhf)") for r in POP.LORA_ARMS[arm]]}
LORA_CORES = [*LORA_GROUPS["lora_general"], *LORA_GROUPS["lora_safety"]]
# manifests and records of record (repo-relative names; paths are built from ROOT at call time, so a test tree works)
MANIFEST_4C = {"released": "logs/healthcheck_d156.sha256", "released_fpr5": "logs/healthcheck_d156_fpr5.sha256",
               "lora": "logs/healthcheck_d156_lora.sha256"}
DEPTH_FPR5_MANIFEST = "logs/fpr5_stage_depth_20260928.sha256"
RUN_MANIFEST = "logs/healthcheck_baselines_d381.sha256"           # written by the pooled list's manifest job
SUMMARY_NAME = "results/healthcheck_baselines_d381_summary.json"
SUMMARY_4C = {"target_0.01": "results/healthcheck_d156_summary.json", "target_0.05": "results/healthcheck_d156_summary_fpr5.json",
              "lora_target_0.01": "results/healthcheck_d156_summary_lora.json"}
SUMMARY_4C_SHA256 = {   # D-255, the 4e summary (D-248) and D-295's LoRA summary, as pulled and verified
    "results/healthcheck_d156_summary.json": "8ef049f304b97ea9cf30a151ac90956d6097cb492c51b1d7f03b27b43df6047e",
    "results/healthcheck_d156_summary_fpr5.json": "ec8d6a8fb54b084789037ed2eab868c2b267f7040241734f2a96994939322af8",
    "results/healthcheck_d156_summary_lora.json": "f697569d57c82d07678d8c8e79c0a953f9b6b7ffff8e4d6dffb802f7b16fe500"}
NPZ_KEYS = ("L<layer>|<probe set>|<level>|<array> for the arrays ks_mean_D, atc_ratio, atc_est_pre, atc_est_post, "
            "atc_error_rate_pre, bbse_harmful_share, bbse_tpr_pre, bbse_n_undefined_seeds and realised_paired_ratio (and, LoRA at 5% only, healthcheck_d156's z_mean, log_sd_ratio, "
            "gaussian_ratio, log_gaussian_ratio, direct_ratio); counts|<probe set>|<level>|<detector>|<class>|tp/fn/fp/tn "
            "for the detectors ks, atc, gaussian, direct, summed over the cells; index 0 the frozen split, then the draws")
BBSE_TEXT = (
    "D-381 (2)(c) as amended by D-392 (1): BBSE (arXiv:1802.03916) is computed BESIDE as a ranking line, with no flag (no "
    "registered cut), and does not enter P-C5.3. Per cell, draw and seed the estimated harmful share on the benign "
    "reference w = (r_post - r_pre) / (TPR_pre - r_pre): r the share of cal_d at or above T_d under a model, floored at "
    "1/n (4c's direct ratio's r); TPR_pre the share of the calibration positives at or above T_d under the pre model, "
    "unfloored; undefined when TPR_pre <= r_pre (counted); the cell's value is the mean over the seeds with a defined "
    "estimate, and a cell with none is left out of that draw's ROC AUC (counted per row); ROC score |w|. Within "
    "one cell, draw and seed (r_pre, TPR_pre fixed) w is an increasing affine function of r_post, and w >= w_c if and only "
    "if r_post / r_pre >= 1 + w_c (TPR_pre - r_pre) / r_pre; across a row's cells it is not a function of the direct "
    "ratio (TPR_pre differs by cell, the direct ratio is a mean over seeds), so its ranking can differ. Its premise, "
    "p(x | y) unchanged across the shift, is what a backbone update breaks: the scores of both classes move.")
KS_TEXT = (
    "D-381 (2)(a): the two-sample KS test (arXiv:1810.11953, BBSDs + KS) on the probe score of cal_d, pre against post, "
    "per seed; the flag is the mean of D over seeds > 1.358 sqrt((n + m) / (n m)) (alpha 0.05, asymptotic). Stated, not "
    "corrected: the two samples are the same prompts, which the test's independence premise does not cover; it is "
    "applied as a deployer would apply it.")


def refuse(msg: str) -> None:
    raise SystemExit(f"REFUSING: {msg}")


def rel(p: Path | str) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(ROOT.resolve())).replace("\\", "/") if p.is_relative_to(ROOT.resolve()) else str(p)


def blas_threads() -> dict[str, str | None]:
    return {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}


def canon(x: Any) -> str:
    return json.dumps(json.loads(json.dumps(x, default=str)), sort_keys=True)  # not a result (D-297)


# ------------------------------------------------------------------------------------------------ the baselines
def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """The two-sample KS statistic (two-sided): max |F_a - F_b| over the pooled values, both empirical CDFs
    right-continuous, computed exactly as max |i m - j n| / (n m) in integers with one rounding (CHOICE 5);
    scipy.stats.ks_2samp's statistic agrees to float rounding (tested)."""
    a, b = np.sort(np.asarray(a, dtype=np.float64)), np.sort(np.asarray(b, dtype=np.float64))
    n, m = a.size, b.size
    allv = np.concatenate([a, b])
    i = np.searchsorted(a, allv, side="right").astype(np.int64)
    j = np.searchsorted(b, allv, side="right").astype(np.int64)
    return float(np.max(np.abs(i * m - j * n)) / (n * m))


def ks_per_seed(pre_c: np.ndarray, post_c: np.ndarray) -> np.ndarray:
    """(S,) KS statistics between the pre and the post scores of cal_d, one per seed (column)."""
    return np.array([ks_statistic(pre_c[:, s], post_c[:, s]) for s in range(pre_c.shape[1])])


def ks_critical(n: int, m: int, c_alpha: float = KS_C_ALPHA) -> float:
    """D-381 (2)(a): the asymptotic two-sample critical value c(alpha) sqrt((n + m) / (n m)), alpha 0.05."""
    return float(c_alpha * np.sqrt((n + m) / (n * m)))


def ks_flag(mean_d: np.ndarray, n: int, m: int) -> np.ndarray:
    """The KS flag: the mean of D over seeds exceeds (strictly) the critical value."""
    return np.asarray(mean_d) > ks_critical(n, m)


def atc_confidence(s: np.ndarray, T: np.ndarray) -> np.ndarray:
    """ATC's confidence as D-381 (2)(b) states it: max(p, 1 - p), p = sigmoid(s - T_d) (for the tests; the job counts
    in margin space, CHOICE 1)."""
    from scipy.special import expit
    p = expit(np.asarray(s, dtype=np.float64) - T)
    return np.maximum(p, 1.0 - p)


def atc_fit(margins: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per seed (column), the ATC threshold in margin space and the number of labelled items below it. margins (N, S):
    |s - T_d| of the labelled calibration items; k (S,): their number of errors at T_d. t is the (k+1)-th smallest
    margin (+inf when k = N), so exactly k items lie below t unless a tie sits at t (CHOICE 2)."""
    N, S = margins.shape
    srt = np.sort(margins, axis=0)
    k = np.asarray(k, dtype=np.int64)
    t = np.where(k < N, srt[np.minimum(k, N - 1), np.arange(S)], np.inf)
    return t, (margins < t).sum(axis=0)


def atc_per_seed(pre_c: np.ndarray, post_c: np.ndarray, pos_pre: np.ndarray, T: np.ndarray) -> dict[str, Any]:
    """D-381 (2)(b) for one partition. pre_c, post_c (n, S): cal_d's scores under the pre and the post model; pos_pre
    (n_pos, S): the calibration positives' pre scores; T (S,): the draw's frozen thresholds. Per seed: the fit on the
    pre labelled set (cal_d's negatives and the positives), the estimates on cal_d floored at 1/n, their ratio."""
    n = pre_c.shape[0]
    k = (pre_c >= T).sum(axis=0) + (pos_pre < T).sum(axis=0)          # errors at T_d: flagged negatives, missed positives
    lab = np.vstack([np.abs(pre_c - T), np.abs(pos_pre - T)])
    t, below = atc_fit(lab, k)
    est_pre = np.maximum((np.abs(pre_c - T) < t).mean(axis=0), 1.0 / n)
    est_post = np.maximum((np.abs(post_c - T) < t).mean(axis=0), 1.0 / n)
    return {"t_margin": t, "error_rate_pre": k / lab.shape[0], "est_pre": est_pre, "est_post": est_post,
            "ratio": est_post / est_pre, "ties": int((below != k).sum())}


def bbse_harmful_share(r_pre: Any, r_post: Any, tpr_pre: Any) -> Any:
    """BBSE's harmful-share estimate on a benign-only reference (D-381 (2)(c), D-392 (1)); no flag."""
    return (np.asarray(r_post, dtype=np.float64) - r_pre) / (np.asarray(tpr_pre, dtype=np.float64) - r_pre)


def bbse_matching_ratio_cut(w_cut: Any, r_pre: Any, tpr_pre: Any) -> Any:
    """Within ONE cell, draw and seed (r_pre, TPR_pre fixed, TPR_pre > r_pre > 0): the direct-ratio cut at which r_post /
    r_pre flags exactly what BBSE flags at w >= w_cut. Across cells the matching cut differs (D-392 (1))."""
    return 1.0 + np.asarray(w_cut, dtype=np.float64) * (np.asarray(tpr_pre, dtype=np.float64) - r_pre) / r_pre


def bbse_per_seed(pre_c: np.ndarray, post_c: np.ndarray, pos_pre: np.ndarray, T: np.ndarray) -> dict[str, np.ndarray]:
    """D-392 (1) for one partition, per seed: r_pre and r_post the shares of cal_d at or above T_d, floored at 1/n as in
    healthcheck_d156.health's direct ratio; TPR_pre the calibration positives' pre share at or above T_d (unfloored);
    w = (r_post - r_pre) / (TPR_pre - r_pre), NaN where TPR_pre <= r_pre (CHOICE 16)."""
    n = pre_c.shape[0]
    r_pre = np.maximum((pre_c >= T).mean(axis=0), 1.0 / n)
    r_post = np.maximum((post_c >= T).mean(axis=0), 1.0 / n)
    tpr = (pos_pre >= T).mean(axis=0)
    ok = tpr > r_pre
    w = np.where(ok, bbse_harmful_share(r_pre, r_post, np.where(ok, tpr, r_pre + 1.0)), np.nan)
    return {"r_pre": r_pre, "r_post": r_post, "tpr_pre": tpr, "w": w, "undefined": int((~ok).sum())}


# ------------------------------------------------------------------------------------------------ one cell and level
def partition_arrays(pre_pool: np.ndarray, post_pool: np.ndarray, cal_pos_pre: np.ndarray,
                     parts: list[tuple[np.ndarray, np.ndarray]], target: float,
                     twin_target: float | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Per partition (index 0 the frozen split): 4c's statistics at `target` (healthcheck_d156.thresholds and health on
    cal_d; the realised paired ratio by paired.ratio_from_flags on the evaluation part: healthcheck_d156.main's lines)
    and the two baselines on cal_d. With twin_target, 4c's statistics also at that target ('twin|<statistic>', D-248).
    Returns the '<f8' arrays and the tally: the counts of ATC fits with a tie at t and of seeds without a BBSE estimate,
    and the smallest defined BBSE denominator TPR_pre - r_pre over the partitions and seeds (None if none, CHOICE 16)."""
    keys = [*FOURC_STATS, *BASE_ARRAYS] + ([f"twin|{q}" for q in FOURC_STATS] if twin_target is not None else [])
    out: dict[str, list[float]] = {q: [] for q in keys}
    n_cal = parts[0][0].size
    ties = undefined = 0
    min_den = np.inf
    for c_, e_ in parts:
        if c_.size != n_cal:
            refuse("a calibration part is not of the frozen size (D-156, D-164)")
        cp, cq = pre_pool[c_], post_pool[c_]
        T = HC.thresholds(cp, target)
        h = HC.health(cp, cq, T)
        for q in HEALTH:
            out[q].append(h[q])
        out[REALISED].append(P.ratio_from_flags(pre_pool[e_] >= T, post_pool[e_] >= T))
        out["ks_mean_D"].append(float(ks_per_seed(cp, cq).mean()))
        a = atc_per_seed(cp, cq, cal_pos_pre, T)
        out["atc_ratio"].append(float(a["ratio"].mean()))
        out["atc_est_pre"].append(float(a["est_pre"].mean()))
        out["atc_est_post"].append(float(a["est_post"].mean()))
        out["atc_error_rate_pre"].append(float(a["error_rate_pre"].mean()))
        ties += a["ties"]
        bb = bbse_per_seed(cp, cq, cal_pos_pre, T)
        w_def = bb["w"][np.isfinite(bb["w"])]                              # the seeds with a defined estimate
        out["bbse_harmful_share"].append(float(w_def.mean()) if w_def.size else np.nan)
        out["bbse_n_undefined_seeds"].append(float(bb["undefined"]))
        out["bbse_tpr_pre"].append(float(bb["tpr_pre"].mean()))
        undefined += bb["undefined"]
        den = bb["tpr_pre"] - bb["r_pre"]
        if (den > 0).any():
            min_den = min(min_den, float(den[den > 0].min()))
        if twin_target is not None:
            T1 = HC.thresholds(cp, twin_target)
            h1 = HC.health(cp, cq, T1)
            for q in HEALTH:
                out[f"twin|{q}"].append(h1[q])
            out[f"twin|{REALISED}"].append(P.ratio_from_flags(pre_pool[e_] >= T1, post_pool[e_] >= T1))
    return ({q: np.asarray(x, dtype="<f8") for q, x in out.items()},
            {"atc_fit_ties": ties, "bbse_undefined": undefined,
             "bbse_min_denominator": float(min_den) if np.isfinite(min_den) else None})


def detector_flags(v: dict[str, np.ndarray], n_cal: int) -> dict[str, dict[str, np.ndarray]]:
    """Per detector and class, the per-draw flag: KS one flag for every class (CHOICE 6); ATC, Gaussian and direct
    when their ratio crosses the class's cut."""
    ksf = ks_flag(v["ks_mean_D"], n_cal, n_cal)
    ratio = {"atc": v["atc_ratio"], "gaussian": v["gaussian_ratio"], "direct": v["direct_ratio"]}
    out = {"ks": {c: ksf for c in CLASSES}}
    out.update({d: {c: HC.crosses(x, cut) for c, cut in CLASSES.items()} for d, x in ratio.items()})
    return out


def level_block(v: dict[str, np.ndarray], flags: dict[str, dict[str, np.ndarray]], tally: dict[str, int], n_cal: int,
                with_4c_stats: bool) -> dict[str, Any]:
    """One cell's level block: healthcheck_d156.main's per-cell lines with the detectors of this module."""
    blk: dict[str, Any] = {q: P.band_summary(v[q][1:], v[q][0]) for q in BASE_ARRAYS}
    if with_4c_stats:
        blk.update({q: P.band_summary(v[q][1:], v[q][0]) for q in FOURC_STATS})
    rr = v[REALISED]
    blk["ks_critical_value"] = ks_critical(n_cal, n_cal)
    blk.update(tally)                    # atc_fit_ties, bbse_undefined (seeds over the partitions), bbse_min_denominator
    blk["bbse_note"] = "beside, a ranking line with no flag (D-392 (1))"
    blk["share_of_draws_in_class"] = {c: float(HC.crosses(rr[1:], cut).mean()) for c, cut in CLASSES.items()}
    blk["share_of_draws_flagged"] = {f"{d}|{c}": float(flags[d][c][1:].mean()) for d in DETECTORS for c in CLASSES}
    return blk


def add_counts(confusion: dict[str, np.ndarray], rr: np.ndarray, flags: dict[str, dict[str, np.ndarray]]) -> None:
    """healthcheck_d156.main's confusion lines: per detector and class, tp/fn/fp/tn per draw, summed over cells."""
    for c, cut in CLASSES.items():
        y = HC.crosses(rr, cut)
        for dname in DETECTORS:
            f = flags[dname][c]
            for key, val in (("tp", y & f), ("fn", y & ~f), ("fp", ~y & f), ("tn", ~y & ~f)):
                k_ = f"{dname}|{c}|{key}"
                confusion[k_] = confusion.get(k_, np.zeros(rr.shape, dtype=np.int64)) + val


# ------------------------------------------------------------------------------------------------ the anchor (3)
def anchor_plan(pair: str, fpr5: bool) -> dict[str, Any]:
    """What a unit is anchored on: the 4c record and npz of the same pair and target, or (LoRA at 5%, CHOICE 9) the
    1% 4c record as D-248's twin and the 5% depth record as the realised operating point's anchor."""
    core = HC.core_of(pair)
    lora = pair in D.LORA_DEPTH
    if lora and fpr5:
        return {"kind": "lora_fpr5: D-248 twin at 1% on the 4c npz, and the 5% operating point on depth_d156 (CHOICE 9)",
                "twin": {"record": f"results/healthcheck_d156_{core}.json", "manifest": MANIFEST_4C["lora"],
                         "target_fpr": TARGET_1},
                "depth": {"record": f"results/depth_d156_{core}_fpr5.json", "manifest": DEPTH_FPR5_MANIFEST}}
    man = MANIFEST_4C["lora"] if lora else MANIFEST_4C["released_fpr5" if fpr5 else "released"]
    return {"kind": "4c npz", "record": f"results/healthcheck_d156_{core}{'_fpr5' if fpr5 else ''}.json",
            "manifest": man, "target_fpr": TARGET_5 if fpr5 else TARGET_1}


def same_arrays(mine: np.ndarray, rec: np.ndarray) -> bool:
    return mine.dtype == rec.dtype and mine.shape == rec.shape and mine.tobytes() == rec.tobytes()


def anchor_arrays(mine: dict[str, np.ndarray], rec: dict[str, np.ndarray], pairs_of_keys: list[tuple[str, str]],
                  what: str) -> int:
    """Refuses unless every (my key, record key) pair holds byte-identical arrays; returns the number compared."""
    missing = [rk for _, rk in pairs_of_keys if rk not in rec]
    if missing:
        refuse(f"the anchor: {what} has no arrays {missing[:3]}; nothing written")
    bad = [rk for mk, rk in pairs_of_keys if not same_arrays(mine[mk], rec[rk])]
    if bad:
        refuse(f"the anchor: {len(bad)} of {len(pairs_of_keys)} arrays differ from {what} (first {bad[:3]}); nothing written")
    return len(pairs_of_keys)


def manifest_map(path: Path) -> tuple[dict[str, str], str]:
    """{repo-relative file: sha256} of a sha256sum manifest, and the manifest's own sha256."""
    if not path.is_file():
        refuse(f"no manifest {rel(path)}")
    b = path.read_bytes()
    out: dict[str, str] = {}
    for line in b.decode("utf-8").splitlines():
        if line.strip():
            h, f = line.split(maxsplit=1)
            f = f.strip().lstrip("*")
            if f in out:
                refuse(f"{rel(path)} names {f} twice")
            out[f] = h
    return out, sha256_bytes(b)


def read_bound(rel_path: str, listed: dict[str, str] | None = None, want: str | None = None) -> tuple[dict[str, Any], str]:
    """A results JSON read through read_result (D-297 (1)); refuses unless `listed` (a manifest's mapping) names it with
    these bytes, and unless they are `want`, when given."""
    p = ROOT / rel_path
    if not p.is_file():
        refuse(f"{rel_path} is missing")
    doc = read_result(p)
    h = _RESULT_READS[_repo_rel(p)]
    if listed is not None and listed.get(rel_path) != h:
        refuse(f"{rel_path} is not the file its manifest names")
    if want is not None and want != h:
        refuse(f"{rel_path} is not the pinned file ({h[:12]} vs {want[:12]})")
    return doc, h


def npz_bound(rel_path: str, listed: dict[str, str] | None, want: str) -> tuple[dict[str, np.ndarray], str]:
    """A per-draw npz read once from its bytes, which must be its record's (want) and its manifest's, when given."""
    p = ROOT / rel_path
    if not p.is_file():
        refuse(f"{rel_path} is missing")
    b = p.read_bytes()
    h = sha256_bytes(b)
    if want != h or (listed is not None and listed.get(rel_path) != h):
        refuse(f"{rel_path} is not the npz its record and manifest name")
    with np.load(io.BytesIO(b), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}, h


def final_problems(registered: bool, threads: dict[str, str | None]) -> list[str]:
    """Why a final name may not be written (empty when it may)."""
    out = []
    if not registered:
        out.append("non-registered parameters (1000 draws, seeds 0-4, every layer)")
    if any(t != "6" for t in threads.values()):
        out.append(f"no 6-thread BLAS pin (D-152 P6): {threads}")
    if os.environ.get("GUARDRAIL_CACHE_DIRS"):
        out.append("GUARDRAIL_CACHE_DIRS is set; the caches must come from the input registry's directory")
    for f, want in REUSED_SHA256.items():
        have = sha256_file(CODE_ROOT / f) if (CODE_ROOT / f).is_file() else None
        if have != want:
            out.append(f"{f} is not the reviewed bytes this module reuses ({(have or 'missing')[:12]} vs {want[:12]})")
    return out


def caches_outside_registry(input_recs: dict[str, dict[str, Any]]) -> list[str]:
    """The caches read whose resolved path is not under the input registry's directory REG.CACHE (CHOICE 15: a final
    run reads its caches from there only; GUARDRAIL_CACHE_DIRS or GUARDRAIL_CACHE_WRITE_DIR could point elsewhere)."""
    root = Path(REG.CACHE).resolve()
    return [r["file"] for r in input_recs.values()
            if not resolve_cache(r["file"], REG.CACHE).resolve().is_relative_to(root)]


def write_pair(out_path: Path, npz_path: Path, payload: dict[str, Any], npz_bytes: bytes) -> list[tuple[Path, str]]:
    """Stage both files, then name them (hard links: an existing name is never overwritten), then make them 444. If a
    later name fails, the names this call already committed are removed (no orphan npz without its record)."""
    json_bytes = json.dumps(payload, indent=2, sort_keys=False, default=str).encode("utf-8")
    staged: list[tuple[Path, Path, str]] = []
    committed: list[Path] = []
    try:
        for p, b in ((npz_path, npz_bytes), (out_path, json_bytes)):
            t, h = stage_bytes(p, b)
            staged.append((t, p, h))
        for t, p, h in staged:
            commit_staged(t, p, h)
            committed.append(p)
    except BaseException:
        for p in committed:                      # only names this call created (commit_staged returned for them)
            p.unlink()
        raise
    finally:
        for t, _, _ in staged:
            if t.exists():
                t.unlink()
    for _, p, _ in staged:
        os.chmod(p, 0o444)
    return [(p, h) for _, p, h in staged]


# ------------------------------------------------------------------------------------------------ the per-pair job
def check_config(rc: dict[str, Any], want: dict[str, Any], what: str) -> None:
    bad = [k for k, v in want.items() if canon(rc.get(k)) != canon(v)]
    if bad:
        refuse(f"{what} was not made on this run's {bad} ({[rc.get(k) for k in bad][:3]} vs {[want[k] for k in bad][:3]})")


def bind_records(pair: str, fpr5: bool, n_draws: int, seeds: list[int], final: bool) -> dict[str, Any]:
    """The selection (healthcheck_d156.main's lines) and the records of record the anchor reads, each bound to its
    manifest and checked against this run (selection, seeds, draws, target, draw seeds, cluster manifest, registry
    run; LoRA at 5% the depth record's completeness and its self-anchor chain). Reads no cache."""
    core = HC.core_of(pair)
    lora = pair in D.LORA_DEPTH
    plan = anchor_plan(pair, fpr5)
    lora5 = "twin" in plan
    reg_stem = D.registry_stem(f"depth_{core}")
    sel_path = D.SELECTION_OF_RECORD_LORA if lora else D.SELECTION_OF_RECORD
    sel_stem = D.SELECTION_STEM.get(f"depth_{core}", reg_stem)
    sel_doc, sel_sha = read_bound(rel(sel_path))
    sel = next((m for m in sel_doc["models"] if m["pair"] == sel_stem), None)
    if sel is None or not sel["selected"]["arm"].startswith("C="):
        refuse(f"no logistic-C selection for {sel_stem}")
    C_sel, L_sel = float(sel["selected"]["arm"].split("=")[1]), int(sel["selected"]["layer"])
    cl_bytes = Path(D.CLUSTERS).read_bytes()
    cl_sha = sha256_bytes(cl_bytes)
    manifests: dict[str, str] = {}
    rec_rel = plan["twin"]["record"] if lora5 else plan["record"]
    man_rel = plan["twin"]["manifest"] if lora5 else plan["manifest"]
    man, manifests[man_rel] = manifest_map(ROOT / man_rel)
    rec_doc, rec_sha = read_bound(rec_rel, man)
    rec_npz_rel = rec_doc["draws_npz"]["file"]
    rec_z, rec_npz_sha = npz_bound(rec_npz_rel, man, rec_doc["draws_npz"]["sha256"])
    if final and rec_doc.get("registered_parameters") is not True:
        refuse(f"{rec_rel} is not a registered run")
    rec_target = plan["twin"]["target_fpr"] if lora5 else plan["target_fpr"]
    cfg = REG.run(reg_stem)
    axis = cfg["axis"]
    want_cfg = {"pair": pair, "registry_pair": reg_stem, "axis": axis, "selection_file": rel(sel_path),
                "selection_pair": sel_stem, "selection_sha256": sel_sha, "selected_C": C_sel, "selected_layer": L_sel,
                "seeds": seeds, "target_fpr": rec_target, "n_draws": n_draws, "draw_seed": P.AXIS_DRAW_SEED[axis],
                "cluster_draw_seed": P.AXIS_CLUSTER_DRAW_SEED[axis], "control": pair == CONTROL,
                **{k: cfg[k] for k in ("pre_repo", "pre_rev", "post_repo", "post_rev")}}
    check_config(rec_doc["config"], want_cfg, rec_rel)
    if rec_doc["config"]["cluster_manifest"]["sha256"] != cl_sha:
        refuse(f"{rec_rel} read another cluster manifest")
    depth_doc, depth_sha, depth_cells = None, None, {}
    if lora5:
        dman, manifests[plan["depth"]["manifest"]] = manifest_map(ROOT / plan["depth"]["manifest"])
        depth_doc, depth_sha = read_bound(plan["depth"]["record"], dman)
        dc = depth_doc["config"]
        check_config(dc, {"pair": f"depth_{core}", "registry_pair": reg_stem, "axis": axis, "selection_file": rel(sel_path),
                          "selection_sha256": sel_sha, "selection_of_pair": sel_stem, "selected_C": C_sel,
                          "selected_layer": L_sel, "seeds": seeds, "target_fpr": TARGET_5, "n_draws": n_draws,
                          "draw_seed": P.AXIS_DRAW_SEED[axis], "cluster_draw_seed": P.AXIS_CLUSTER_DRAW_SEED[axis],
                          "layers": rec_doc["config"]["layers"],
                          **{k: cfg[k] for k in ("pre_repo", "pre_rev", "post_repo", "post_rev")}}, plan["depth"]["record"])
        sa = depth_doc.get("fpr5_self_anchor") or {}
        if sa.get("n_cells_identical_at_1pct") != len(depth_doc["cells"]) or \
                (sa.get("file"), sa.get("sha256")) != (rec_doc["anchor"]["file"], rec_doc["anchor"]["sha256"]):
            refuse(f"{plan['depth']['record']} is not complete and self-anchored on the 1% depth record {rec_rel} "
                   f"anchored on (D-248)")
        if dc["cluster_manifest"]["sha256"] != cl_sha:
            refuse(f"{plan['depth']['record']} read another cluster manifest")
        depth_cells = {(c["layer"], c["probe_set"]): c for c in depth_doc["cells"]}
    return {"plan": plan, "lora5": lora5, "reg_stem": reg_stem, "sel_path": sel_path, "sel_stem": sel_stem,
            "sel_sha": sel_sha, "C_sel": C_sel, "L_sel": L_sel, "cl_bytes": cl_bytes, "cl_sha": cl_sha,
            "manifests": manifests, "cfg": cfg, "axis": axis, "rec_rel": rec_rel, "rec_doc": rec_doc, "rec_sha": rec_sha,
            "rec_npz_rel": rec_npz_rel, "rec_z": rec_z, "rec_npz_sha": rec_npz_sha, "rec_target": rec_target,
            "depth_doc": depth_doc, "depth_sha": depth_sha, "depth_cells": depth_cells}


def run_job(args: argparse.Namespace) -> int:
    t_start = time.time()
    pair = args.pair
    core = HC.core_of(pair)
    lora = pair in D.LORA_DEPTH
    target = TARGET_5 if args.fpr5 else TARGET_1
    sfx = "_fpr5" if args.fpr5 else ""
    seeds = list(args.seeds)
    registered = args.n_draws == REGISTERED["n_draws"] and seeds == REGISTERED["seeds"] and args.layers is None
    final_path = (RESULTS / f"{RUN_KIND}_{core}{sfx}.json").resolve()
    out_path = Path(args.out).resolve() if args.out else final_path
    npz_path = out_path.with_name(out_path.stem + "_draws.npz")
    final = out_path == final_path
    for p in (out_path, npz_path):
        if p.exists():
            refuse(f"{p} exists; results are never overwritten (D-152 P5)")
    if not final and out_path.is_relative_to(RESULTS.resolve()):
        refuse("a test run (--out) must not write under results/; only the unit's own final name may")
    threads = blas_threads()
    if final:
        problems = final_problems(registered, threads)
        if problems:
            refuse(f"the final name: {problems}; use --out outside results/ for a test")
    code0 = P.code_block(CODE_FILES)
    b = bind_records(pair, args.fpr5, args.n_draws, seeds, final)
    plan, lora5, reg_stem, sel_path, sel_stem, sel_sha = (b[k] for k in ("plan", "lora5", "reg_stem", "sel_path",
                                                                          "sel_stem", "sel_sha"))
    C_sel, L_sel, cl_bytes, cl_sha, manifests, cfg, axis = (b[k] for k in ("C_sel", "L_sel", "cl_bytes", "cl_sha",
                                                                           "manifests", "cfg", "axis"))
    rec_rel, rec_doc, rec_sha, rec_npz_rel, rec_z, rec_npz_sha, rec_target = (
        b[k] for k in ("rec_rel", "rec_doc", "rec_sha", "rec_npz_rel", "rec_z", "rec_npz_sha", "rec_target"))
    depth_doc, depth_sha, depth_cells = b["depth_doc"], b["depth_sha"], b["depth_cells"]

    # --- the caches through the input registry (healthcheck_d156.main's lines) ------------------------------------
    data: dict[str, dict[str, Any]] = {"pre": {}, "post": {}}
    input_sha: dict[str, str] = {}
    input_recs: dict[str, dict[str, Any]] = {}
    for side in data:
        for part in PARTS:
            z, r = REG.load(cfg[f"{side}_repo"], axis, part)
            data[side][part] = z
            input_sha[f"{side}/{part}"], input_recs[f"{side}/{part}"] = r["sha256"], r
    if final:
        outside = caches_outside_registry(input_recs)
        if outside:
            refuse(f"the final name: caches not read from the input registry's directory {REG.CACHE}: {outside[:3]}")
    reg_sources = REG.sources_sha256()
    if D.sources_now() != reg_sources:
        refuse("the extraction records or pins on disk are not the ones the registry read")
    for who, d_ in (("4c", rec_doc), ("depth", depth_doc)):
        if d_ is not None and any(d_["input_cache_sha256"].get(k) != h for k, h in input_sha.items()):
            refuse(f"the {who} record of record read other caches")
    verification = REG.verification(input_recs)
    for part in PARTS:
        a, b = data["pre"][part], data["post"][part]
        if not np.array_equal(a["item_ids"], b["item_ids"]) or not np.array_equal(a["labels"], b["labels"]):
            refuse(f"{part} items or labels differ between pre and post")
        for zz in (a, b):
            if "nonfinite" in zz.files and zz["nonfinite"].any():
                refuse(f"{part} has non-finite items")
    grids = [sorted(int(k.split("_")[1]) for k in zz.files if k.startswith("layer_")) for s in data for zz in data[s].values()]
    if any(g != grids[0] for g in grids) or grids[0] != list(rec_doc["config"]["layers"]):
        refuse(f"the caches' layer grids differ from each other or from {rec_rel}'s")
    if args.layers is not None and (not args.layers or set(args.layers) - set(grids[0])):
        refuse(f"layers {args.layers} are not a subset of the grid {grids[0]}")
    layers = [li for li in grids[0] if args.layers is None or li in args.layers]

    y_tr = data["pre"]["probe_train"]["labels"].astype(bool)
    y_cal = data["pre"]["threshold_calib"]["labels"].astype(bool)
    y_ev = data["pre"]["eval"]["labels"].astype(bool)
    ncal, neg, pos = ~y_cal, ~y_ev, y_ev
    pooled_ids = [str(i) for i in data["pre"]["threshold_calib"]["item_ids"][ncal]] + \
                 [str(i) for i in data["pre"]["eval"]["item_ids"][neg]]
    n_cal, n_pool, n_cal_pos = int(ncal.sum()), len(pooled_ids), int(y_cal.sum())
    cl = json.loads(cl_bytes.decode("utf-8"))["axes"][axis]  # listed (D-297): the cluster manifest, hashed and listed
    if cl["n_pooled_negatives"] != n_pool or cl["n_calibration"] != n_cal or \
            cl["pool_item_ids_sha256"] != sha256_bytes("\n".join(pooled_ids).encode("utf-8")):
        refuse(f"manifests/negative_clusters.json does not describe the {axis} pool")
    if {str(i) for i in data["pre"]["probe_train"]["item_ids"]} & set(pooled_ids):
        refuse("a probe_train item is in the pooled negatives (F-045)")
    if n_cal_pos == 0:
        refuse("threshold_calib holds no positives: ATC has no labelled calibration positives (D-381 (2)(b))")
    frozen_part = (np.arange(n_cal), np.arange(n_cal, n_pool))
    draws = {LEVELS[0]: P.repartition_draws(n_cal, n_pool - n_cal, args.n_draws, P.AXIS_DRAW_SEED[axis]),
             LEVELS[1]: P.cluster_repartition_draws(P.cluster_units(pooled_ids, cl["clusters"]), n_cal,
                                                    args.n_draws, P.AXIS_CLUSTER_DRAW_SEED[axis])}
    parts = {lvl: [frozen_part] + [(d[:n_cal], d[n_cal:]) for d in dr] for lvl, dr in draws.items()}
    print(f"D-381 {pair} at {target:g} ({cfg['pre_repo']} -> {cfg['post_repo']}, {axis}): layers {layers}; selected C "
          f"{C_sel:g}; n_cal {n_cal}, calibration positives {n_cal_pos}; anchor: {plan['kind']}; registered {registered}",
          flush=True)

    cells: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    confusion: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    n_anchor = {"arrays": 0, "operating_points": 0, "paired_ratio_arrays": 0, "count_arrays": 0}
    total_ties = total_undefined = 0
    for li in layers:
        k = f"layer_{li}"
        X = {s: {p: data[s][p][k] for p in PARTS} for s in data}
        for pset in PROBE_SETS:
            probes = []
            for seed in seeds:
                _, fit = P.d117_split(len(y_tr), seed)
                rows, C, order = (fit, C_sel, "d117_permutation") if pset == "selected" else (np.sort(fit), 1.0, "file_order")
                probes.append(P.fit_probe(X["pre"]["probe_train"][rows], y_tr[rows], seed, C, order))
            S = lambda x: np.column_stack([q.score(x) for q in probes])  # noqa: E731
            pre_pool = np.vstack([S(X["pre"]["threshold_calib"][ncal]), S(X["pre"]["eval"][neg])])
            post_pool = np.vstack([S(X["post"]["threshold_calib"][ncal]), S(X["post"]["eval"][neg])])
            cal_pos_pre = S(X["pre"]["threshold_calib"][y_cal])          # the labelled calibration positives
            if lora5:                                                    # the eval positives: depth's operating point
                pos_pre, pos_post = S(X["pre"]["eval"][pos]), S(X["post"]["eval"][pos])
            cblk: dict[str, Any] = {"layer": li, "probe_set": pset, "probe_C": C_sel if pset == "selected" else 1.0,
                                    "is_selected_layer": li == L_sel}
            for lvl in LEVELS:
                v, tally = partition_arrays(pre_pool, post_pool, cal_pos_pre, parts[lvl], target,
                                            TARGET_1 if lora5 else None)
                base = f"L{li}|{pset}|{lvl}"
                # ANCHOR (3), before anything is written
                if lora5:
                    n_anchor["arrays"] += anchor_arrays(v, rec_z, [(f"twin|{q}", f"{base}|{q}") for q in FOURC_STATS],
                                                        f"{rec_npz_rel} (the 1% twin, D-248)")
                    dcell = depth_cells.get((li, pset))
                    if dcell is None:
                        refuse(f"{plan['depth']['record']} has no cell L{li} {pset}")
                    rep, arr = P.repartition_estimate(pre_pool[:n_cal], pre_pool[n_cal:], post_pool[:n_cal],
                                                      post_pool[n_cal:], pos_pre, pos_post, draws[lvl], target)
                    want = dcell["operating_point"] if lvl == LEVELS[0] else dcell[LEVELS[1]]["operating_point"]
                    if json.loads(json.dumps(rep, default=str)) != want:  # not a result (D-297): 4c's comparison form
                        refuse(f"the anchor: L{li} {pset} {lvl}: the 5% operating point is not "
                               f"{plan['depth']['record']}'s; nothing written")
                    n_anchor["operating_points"] += 1
                    pr = np.concatenate([[rep["paired_ratio"]["frozen"]], arr["paired_ratio"]]).astype("<f8")
                    if not same_arrays(v[REALISED], pr):
                        refuse(f"the anchor: L{li} {pset} {lvl}: the partitions are not repartition_estimate's")
                    n_anchor["paired_ratio_arrays"] += 1
                else:
                    n_anchor["arrays"] += anchor_arrays(v, rec_z, [(q, f"{base}|{q}") for q in FOURC_STATS], rec_npz_rel)
                flags = detector_flags(v, n_cal)
                cblk[lvl] = level_block(v, flags, tally, n_cal, with_4c_stats=lora5)
                total_ties += tally["atc_fit_ties"]
                total_undefined += tally["bbse_undefined"]
                keep = [*BASE_ARRAYS, REALISED, *(HEALTH if lora5 else ())]
                arrays.update({f"{base}|{q}": v[q] for q in keep})
                add_counts(confusion.setdefault((pset, lvl), {}), v[REALISED], flags)
            cells.append(cblk)
        print(f"  layer {li:>2} done", flush=True)
    counts: dict[str, Any] = {}
    for (pset, lvl), cc in confusion.items():
        blk = counts.setdefault(pset, {}).setdefault(lvl, {})
        for key, x in cc.items():
            arrays[f"counts|{pset}|{lvl}|{key}"] = x
            blk[key] = {"d156_mean": float(x[1:].mean()), "frozen_split": int(x[0])}
    if not lora5 and args.layers is None:
        # every cell: the Gaussian and direct counts summed over cells are the 4c npz's (D-255's inputs)
        n_anchor["count_arrays"] = anchor_arrays(
            arrays, rec_z, [(f"counts|{ps}|{lvl}|{d}|{c}|{q}",) * 2 for ps in PROBE_SETS for lvl in LEVELS
                            for d in FOURC_DETECTORS for c in CLASSES for q in COUNTS], f"{rec_npz_rel} (counts)")
    n_blocks = len(layers) * len(PROBE_SETS) * len(LEVELS)
    if n_anchor["arrays"] != n_blocks * len(FOURC_STATS) or (lora5 and n_anchor["operating_points"] != n_blocks):
        refuse(f"to write: {n_anchor} anchored, {n_blocks} blocks")

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    npz_bytes = buf.getvalue()
    # --- nothing read may have changed while the run went ----------------------------------------------------------
    code1 = P.code_block(CODE_FILES)
    changed = [w for w, a, b in (("code", code1["sha256"], code0["sha256"]), ("registry sources", D.sources_now(), reg_sources),
                                 ("cluster manifest", sha256_file(D.CLUSTERS), cl_sha)) if a != b]
    changed += [r_ for r_, h in _RESULT_READS.items() if sha256_file(ROOT / r_) != h]
    changed += [rec_npz_rel] if sha256_file(ROOT / rec_npz_rel) != rec_npz_sha else []
    changed += [m for m, h in manifests.items() if sha256_file(ROOT / m) != h]
    changed += [f"cache {r['file']}" for r in input_recs.values() if sha256_file(resolve_cache(r["file"], REG.CACHE)) != r["sha256"]]
    if changed:
        refuse(f"to write: changed during the run: {changed}")
    config = {"pair": pair, "core": core, "registry_pair": reg_stem, "control": pair == CONTROL, "lora": lora,
              "axis": axis, "selection_file": rel(sel_path), "selection_pair": sel_stem, "selection_sha256": sel_sha,
              "selected_C": C_sel, "selected_layer": L_sel, "layers": layers, "seeds": seeds, "target_fpr": target,
              "n_draws": args.n_draws, "draw_seed": P.AXIS_DRAW_SEED[axis], "cluster_draw_seed": P.AXIS_CLUSTER_DRAW_SEED[axis],
              "cluster_manifest": {"file": rel(D.CLUSTERS), "sha256": cl_sha},
              "n_calibration_negatives": n_cal, "n_calibration_positives": n_cal_pos, "n_pooled_negatives": n_pool,
              "classes": {c: list(cut) for c, cut in CLASSES.items()}, "detectors": list(DETECTORS),
              "baselines": {"ks": {"statistic": "two-sample KS D (max |ECDF difference|, exact counts), cal_d pre vs "
                                                "post, per seed; score = mean over seeds", "alpha": KS_ALPHA, "c_alpha": KS_C_ALPHA,
                                   "critical_value": ks_critical(n_cal, n_cal), "n": n_cal, "m": n_cal,
                                   "flag": "mean D > critical value, one flag per draw for every class",
                                   "text": KS_TEXT},
                            "atc": {"confidence": "max(p, 1 - p), p = sigmoid(s - T_d); counted in margin space |s - T_d|",
                                    "fit_set": "cal_d negatives + threshold_calib positives, pre model",
                                    "threshold": "the (k+1)-th smallest labelled margin, k = pre errors at T_d",
                                    "estimate": "share of cal_d below t under a model, floored at 1/n_cal",
                                    "ratio": "post / pre per seed, mean over seeds", "flag": "crosses the class's cut",
                                    "roc_score": "|log ratio|"},
                            "bbse": {"text": BBSE_TEXT, "flag": None, "in_P_C5_3": False, "roc_score": "|w|",
                                     "premise": "p(x | y) unchanged: broken by an update (D-392 (1))"}},
              "pre_repo": cfg["pre_repo"], "pre_rev": cfg["pre_rev"], "post_repo": cfg["post_repo"], "post_rev": cfg["post_rev"],
              "inputs_from": "src/input_registry.py (D-165)", "input_registry_sources_sha256": reg_sources}
    records = {"healthcheck_4c": {"file": rec_rel, "sha256": rec_sha, "target_fpr": rec_target,
                                  "d170_flag": rec_doc["input_verification"]["flag"]},   # as stored there (CHOICE 10)
               "healthcheck_4c_npz": {"file": rec_npz_rel, "sha256": rec_npz_sha},
               **({"depth_fpr5": {"file": plan["depth"]["record"], "sha256": depth_sha}} if lora5 else {}),
               "manifests": manifests}
    anchor = {"kind": plan["kind"], "identical": True, "counts": n_anchor, "n_level_blocks": n_blocks,
              "rule": ("every per-draw array of healthcheck_d156's six statistics, per layer, probe set and level, equals "
                       "the 4c npz byte for byte" + ("" if lora5 else
                                                     "; with every layer, the Gaussian and direct counts summed over cells too") +
                       ("; at 1% (D-248's twin), and the 5% operating point equals the depth record's (canonical "
                        "JSON) with its per-draw paired ratios" if lora5 else "")),
              "records": records}
    payload: dict[str, Any] = {
        **provenance_block(RUN_KIND), "preregistration": PREREG, "code": code0,
        "reused_files_sha256": {f: sha256_file(CODE_ROOT / f) for f in REUSED_SHA256},
        "registered_parameters": registered, "blas_threads": threads, "elapsed_s": round(time.time() - t_start, 1),
        "input_verification": verification, "config": config, "config_hash": canonical_hash(config), "anchor": anchor,
        "population": {"group": "HEALTHCHECK_BASELINES_D381 (to be declared in src/populations.py after the pull)",
                       "role": ("negative check (GPU-change control)" if pair == CONTROL else
                                "LoRA run (D-278; all nine share one pre model)" if lora else "released pair") +
                               ("; 5% beside (D-248)" if args.fpr5 else "; 1% primary")},
        "input_files_sha256": {**dict(_RESULT_READS), rec_npz_rel: rec_npz_sha, rel(D.CLUSTERS): cl_sha, **manifests},
        "input_cache_sha256": input_sha,
        "input_cache_source": {kk: {"file": r["file"], "expected_from": r["expected_from"], "pin_only": r["pin_only"]}
                               for kk, r in input_recs.items()},
        "draws_npz": {"file": rel(npz_path), "sha256": sha256_bytes(npz_bytes), "size": len(npz_bytes), "keys": NPZ_KEYS},
        "atc_fit_ties_total": total_ties, "bbse_undefined_seeds_total": total_undefined,
        "confusion_counts_pooled_over_cells": counts, "cells": cells}
    problems = input_identity_problems(payload)
    if problems:
        refuse(f"to write (D-297 (1) results-writer rule): {problems}")
    for p, h in write_pair(out_path, npz_path, payload, npz_bytes):
        print(f"Written: {rel(p)}\nsha256:  {h}")
    print(f"anchor identical: {n_anchor}; ATC fits with a tie at t: {total_ties}; BBSE seeds undefined: {total_undefined}")
    return 0


# ------------------------------------------------------------------------------------------------ the summary
def roc_auc(z_of: dict[str, dict[str, np.ndarray]], names: list[str], ps: str, lvl: str,
            stats: dict[str, Callable[[dict[str, np.ndarray], str], np.ndarray]]) -> dict[str, Any]:
    """healthcheck_summary_d156.scoring's ROC lines (CHOICE 11): per draw, across the row's cells, the ROC AUC of each
    score for each class; D-156 mean and band."""
    keys = [(n, kk.rsplit("|", 1)[0]) for n in names for kk in z_of[n] if kk.startswith("L") and
            kk.endswith(f"|{REALISED}") and f"|{ps}|{lvl}|" in kk]
    rr = np.vstack([z_of[n][f"{base}|{REALISED}"] for n, base in keys])
    sv = {s: np.vstack([f(z_of[n], base) for n, base in keys]) for s, f in stats.items()}
    out: dict[str, Any] = {"n_cells": len(keys)}
    for c, cut in CLASSES.items():
        y = HS.crosses(rr, cut)
        for sname, v in sv.items():
            # a cell whose score is not finite in a draw (BBSE with no defined seed) is left out of that draw's AUC
            # (CHOICE 16); with every score finite this is healthcheck_summary_d156.scoring's line exactly
            a = np.full(rr.shape[1], np.nan)
            for d in range(rr.shape[1]):
                ok = np.isfinite(v[:, d])
                yd = y[ok, d]
                if 0 < yd.sum() < yd.size:
                    a[d] = M.auroc(v[ok, d][yd], v[ok, d][~yd]).value
            out[f"roc_auc|{sname}|{c}"] = P.band_summary(a[1:], a[0])
    for sname in sv:
        if sname in LEFT_OUT_REPORTED:
            out[f"cells_left_out|{sname}"] = left_out(sv[sname])
    return out


def left_out(v: np.ndarray) -> dict[str, Any]:
    """CHOICE 16: per draw, the number of the row's cells left out of the AUC (score not finite): its D-156 mean and
    maximum over the draws, the frozen split's value, and the share of draws with any cell left out."""
    n = (~np.isfinite(v)).sum(axis=0)
    return {"d156_mean": float(n[1:].mean()), "max": int(n[1:].max()), "frozen_split": int(n[0]),
            "share_of_draws_with_any": float((n[1:] > 0).mean()), "n_cells": int(v.shape[0])}


LEFT_OUT_REPORTED = ("abs_bbse_harmful_share",)


def bbse_fewer_seeds(z_of: dict[str, dict[str, np.ndarray]], names: list[str], ps: str, lvl: str) -> dict[str, Any]:
    """CHOICE 16: over the row's cells, the share of cell-draws whose BBSE mean uses fewer than all the seeds (some seed
    undefined, bbse_n_undefined_seeds > 0; a cell with every seed undefined is also left out of the AUC), the frozen
    split's count of such cells and the largest number of undefined seeds in one cell-draw."""
    keys = [(n, kk.rsplit("|", 1)[0]) for n in names for kk in z_of[n] if kk.startswith("L") and
            kk.endswith("|bbse_n_undefined_seeds") and f"|{ps}|{lvl}|" in kk]
    nu = np.vstack([z_of[n][f"{base}|bbse_n_undefined_seeds"] for n, base in keys])
    return {"share_of_cell_draws": float((nu[:, 1:] > 0).mean()), "frozen_split_cells": int((nu[:, 0] > 0).sum()),
            "max_undefined_seeds": int(nu[:, 1:].max()), "n_cells": len(keys),
            "rule": "cell-draws (frozen split excluded) whose BBSE mean over seeds uses fewer than all the seeds"}


BASELINE_SCORES: dict[str, Callable[[dict[str, np.ndarray], str], np.ndarray]] = {
    "ks_mean_D": lambda z, b: z[f"{b}|ks_mean_D"],
    "abs_log_atc_ratio": lambda z, b: np.abs(np.log(z[f"{b}|atc_ratio"])),
    "abs_bbse_harmful_share": lambda z, b: np.abs(z[f"{b}|bbse_harmful_share"])}       # beside, no flag (D-392 (1))
FOURC_SCORES: dict[str, Callable[[dict[str, np.ndarray], str], np.ndarray]] = {     # healthcheck_summary_d156's three
    "abs_log_gaussian_ratio": lambda z, b: np.abs(z[f"{b}|log_gaussian_ratio"]),
    "abs_log_direct_ratio": lambda z, b: np.abs(np.log(z[f"{b}|direct_ratio"])),
    "abs_z_mean": lambda z, b: np.abs(z[f"{b}|z_mean"])}


def scoring(z_of: dict[str, dict[str, np.ndarray]], names: list[str], ps: str, lvl: str, detectors: tuple[str, ...],
            scores: dict[str, Callable[[dict[str, np.ndarray], str], np.ndarray]]) -> dict[str, Any]:
    """One row: healthcheck_summary_d156.scoring for the given detectors and ROC scores, its estimator (pooled_counts,
    pair_cluster_index) called unchanged."""
    k = len(names)
    out: dict[str, Any] = {"pairs": names, "n_pairs": k}
    if k == 0:
        return {**out, "note": "no pair in this row"}
    idx = HS.pair_cluster_index(k)
    out["interval"] = (f"pair-cluster bootstrap B={HS.B}, seed {HS.SEED}" if idx is not None
                       else f"point only (fewer than {HS.MIN_PAIRS} pairs)")
    for c in CLASSES:
        for dname in detectors:
            out[f"{dname}|{c}"] = HS.pooled_counts({q: [z_of[n][f"counts|{ps}|{lvl}|{dname}|{c}|{q}"] for n in names]
                                                    for q in COUNTS}, idx)
    out.update(roc_auc(z_of, names, ps, lvl, scores))
    if "abs_bbse_harmful_share" in scores:
        out["bbse_mean_over_fewer_seeds"] = bbse_fewer_seeds(z_of, names, ps, lvl)
    return out


def control_row(cz: dict[str, np.ndarray], ps: str, lvl: str, detectors: tuple[str, ...]) -> dict[str, Any]:
    """healthcheck_summary_d156.main's GPU-control lines: false-alarm rate per detector and class, and the share of
    draws with a control cell in the class."""
    fa: dict[str, Any] = {}
    for c in CLASSES:
        for dname in detectors:
            fp, tn = cz[f"counts|{ps}|{lvl}|{dname}|{c}|fp"], cz[f"counts|{ps}|{lvl}|{dname}|{c}|tn"]
            far = HS.rate(fp, fp + tn)
            fa[f"{dname}|{c}"] = {"false_alarm_rate": P.band_summary(far[1:], far[0])}
        pos = cz[f"counts|{ps}|{lvl}|gaussian|{c}|tp"] + cz[f"counts|{ps}|{lvl}|gaussian|{c}|fn"]
        fa[f"share_of_draws_with_a_control_cell_in_class|{c}"] = float((pos[1:] > 0).mean())
    return fa


def score_predictions(gauss: dict[str, Any], gauss_roc: float | None, ks: dict[str, Any], atc: dict[str, Any],
                      atc_roc: float | None) -> dict[str, Any]:
    """P-C5.1-3 (D-381 (4)) on pooled point estimates ({'sensitivity', 'specificity'} as pooled_counts gives them) and
    the ROC AUC D-156 means. holds is None, with the reason, when a value is undefined (CHOICE 13)."""
    def verdict(holds: bool | None) -> str:
        return "UNDEFINED" if holds is None else "HELD" if holds else "FAILED"

    out: dict[str, Any] = {}
    a, b = ks.get("specificity"), gauss.get("specificity")
    why = None if a is not None and b is not None else "a pooled specificity is undefined (no cell outside the class)"
    h = None if why else bool(a < b)
    out["P-C5.1"] = {"statement": "KS's pooled specificity is lower than the Gaussian ratio's", "ks_specificity": a,
                     "gaussian_specificity": b, "holds": h, "verdict": verdict(h), **({"reason": why} if why else {})}
    why = None if atc_roc is not None and gauss_roc is not None else "a ROC AUC D-156 mean is undefined"
    h = None if why else bool(atc_roc < gauss_roc)
    out["P-C5.2"] = {"statement": "ATC's ROC AUC (D-156 mean) is lower than the Gaussian ratio's", "atc_roc_auc": atc_roc,
                     "gaussian_roc_auc": gauss_roc, "holds": h, "verdict": verdict(h), **({"reason": why} if why else {})}
    vals = {"gaussian": gauss, "ks": ks, "atc": atc}
    undefined = [f"{n}.{q}" for n, v in vals.items() for q in ("sensitivity", "specificity") if v.get(q) is None]
    both = {n: (None if undefined else bool(v["sensitivity"] >= gauss["sensitivity"] and v["specificity"] >= gauss["specificity"]))
            for n, v in (("ks", ks), ("atc", atc))}
    h = None if undefined else not any(both.values())
    out["P-C5.3"] = {"statement": "no baseline has both pooled sensitivity and pooled specificity at or above the "
                                  "Gaussian ratio's", "baselines": ["ks", "atc"],
                     "pooled": {n: {q: v.get(q) for q in ("sensitivity", "specificity")} for n, v in vals.items()},
                     "baseline_at_or_above_on_both": both, "holds": h, "verdict": verdict(h),
                     "bbse": "not in P-C5.3: BBSE has no flag (D-392 (1)); its ROC AUC is beside",
                     **({"reason": f"undefined: {undefined}"} if undefined else {})}
    return out


def summary_job(args: argparse.Namespace) -> int:
    t_start = time.time()
    final_path = (ROOT / SUMMARY_NAME).resolve()
    out = Path(args.out).resolve() if args.out else final_path
    final = out == final_path
    if out.exists():
        refuse(f"{out} exists; never overwritten (D-152 P5)")
    if not final and out.is_relative_to(RESULTS.resolve()):
        refuse("a test output must not be under results/")
    threads = blas_threads()
    if final:
        # the summary reads no cache: the BLAS pin and the reused files' bytes gate its final name
        problems = [p for p in final_problems(True, threads) if "GUARDRAIL_CACHE_DIRS" not in p]
        if problems:
            refuse(f"the final name: {problems}")
    code0 = P.code_block(CODE_FILES)
    man, man_sha = manifest_map(ROOT / RUN_MANIFEST)
    inputs: dict[str, str] = {RUN_MANIFEST: man_sha}
    named: set[str] = set()
    z_of: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    flag_of: dict[str, dict[str, Any]] = {}             # the records' current D-170 flags (their own input_verification)
    flag4c_of: dict[str, dict[str, Any]] = {}           # the flags their 4c records stored (CHOICE 10)
    z4c_of: dict[str, dict[str, dict[str, np.ndarray]]] = {}    # the released pairs' 4c npz (CHOICE 17)
    per_record: dict[str, dict[str, Any]] = {}
    registered_all = True
    for tkey, fpr5 in (("target_0.01", False), ("target_0.05", True)):
        z_of[tkey], flag_of[tkey], flag4c_of[tkey], z4c_of[tkey] = {}, {}, {}, {}
        man4c_rel = MANIFEST_4C["released_fpr5" if fpr5 else "released"]
        man4c, inputs[man4c_rel] = manifest_map(ROOT / man4c_rel)
        for pair in HC.PAIRS:
            core = HC.core_of(pair)
            rrel = f"results/{RUN_KIND}_{core}{'_fpr5' if fpr5 else ''}.json"
            doc, rsha = read_bound(rrel, man)
            c_ = doc.get("config") or {}
            anc = doc.get("anchor") or {}
            if doc.get("run_kind") != RUN_KIND or c_.get("pair") != pair or \
                    c_.get("target_fpr") != (TARGET_5 if fpr5 else TARGET_1) or anc.get("identical") is not True:
                refuse(f"{rrel} is not this module's anchored record of {pair} at its target")
            if c_.get("control") is not (pair == CONTROL):
                refuse(f"{rrel}: the control flag of the records is not as expected (healthcheck_summary_d156's check)")
            reg_ok = doc.get("registered_parameters") is True and c_.get("n_draws") == REGISTERED["n_draws"]
            registered_all = registered_all and reg_ok
            if final and not reg_ok:
                refuse(f"{rrel} is not a registered run")
            if canon(doc["code"]["sha256"]) != canon(code0["sha256"]):
                refuse(f"{rrel} was made by another revision of the code (its code block is not this tree's)")
            recs = anc.get("records") or {}
            rec4c = recs.get("healthcheck_4c") or {}
            plan = anchor_plan(pair, fpr5)
            if rec4c.get("file") != (plan["twin"]["record"] if "twin" in plan else plan["record"]) or "d170_flag" not in rec4c:
                refuse(f"{rrel} does not name its 4c record of record with the D-170 flag that record stored (CHOICE 10)")
            nrel = doc["draws_npz"]["file"]
            z, _ = npz_bound(nrel, man, doc["draws_npz"]["sha256"])
            inputs[nrel] = doc["draws_npz"]["sha256"]
            named.update({rrel, nrel})
            z_of[tkey][core] = z
            flag_of[tkey][core], flag4c_of[tkey][core] = doc["input_verification"]["flag"], rec4c["d170_flag"]
            per_record[rrel] = {"file": rrel, "sha256": rsha, "commit": (doc.get("code") or {}).get("commit"),
                                "config_hash": doc.get("config_hash"), "anchor_counts": anc.get("counts"),
                                "d170_flag": flag_of[tkey][core], "d170_flag_4c_record": flag4c_of[tkey][core],
                                "registered_parameters": doc.get("registered_parameters")}
            if core in RELEASED_CORES:                  # CHOICE 17: the 4c scores this record anchored on
                n4 = recs.get("healthcheck_4c_npz") or {}
                if not n4.get("file") or not n4.get("sha256"):
                    refuse(f"{rrel} does not name the 4c npz it anchored on")
                z4c_of[tkey][core], inputs[n4["file"]] = npz_bound(n4["file"], man4c, n4["sha256"])
    extra = sorted(set(man) - named)
    if extra:
        refuse(f"{RUN_MANIFEST} names files that are not the 38 records and their npz: {extra[:3]}")
    s4c = {k: read_bound(r, want=SUMMARY_4C_SHA256[r])[0] for k, r in SUMMARY_4C.items()}
    # CHOICE 17, D-392 (4): KS does not depend on T_d, so each unit's mean D arrays at 1% and 5% are the same bytes
    n_ks_identical = 0
    for core, a in z_of["target_0.01"].items():
        b = z_of["target_0.05"][core]
        ka = sorted(k for k in a if k.endswith("|ks_mean_D"))
        if not ka or ka != sorted(k for k in b if k.endswith("|ks_mean_D")) or \
                any(not same_arrays(a[k], b[k]) for k in ka):
            refuse(f"{core}: the ks_mean_D arrays at 1% and 5% are not byte-identical (D-392 (4)); nothing written")
        n_ks_identical += len(ka)
    # CHOICE 10: the unflagged row is built on the 4c summary's own pair list, refused unless well-formed and equal to
    # the released pairs whose 4c record stored no D-170 flag
    unflagged_of: dict[str, list[str]] = {}
    for tkey in ("target_0.01", "target_0.05"):
        rows4 = list(s4c[tkey]["rows"].values())
        lists = [(r.get("unflagged_pairs") or {}).get("pairs") for r in rows4]
        pl = lists[0] if lists else None
        well = (bool(lists) and all(x == pl for x in lists) and isinstance(pl, list) and all(isinstance(p, str) for p in pl)
                and len(set(pl)) == len(pl) and pl == [p for p in RELEASED_CORES if p in pl]
                and all((r.get("unflagged_pairs") or {}).get("n_pairs") == len(pl) for r in rows4))
        if not well:
            refuse(f"{SUMMARY_4C[tkey]}'s unflagged row is not one well-formed list of released pairs (CHOICE 10)")
        want_pl = [p for p in RELEASED_CORES if flag4c_of[tkey][p] is None]
        if pl != want_pl:
            refuse(f"{SUMMARY_4C[tkey]}'s unflagged row lists {pl}, not the released pairs whose 4c record stored no "
                   f"D-170 flag {want_pl} (CHOICE 10)")
        unflagged_of[tkey] = pl

    groups = {"released_pairs": RELEASED_CORES, "lora_all": LORA_CORES, **LORA_GROUPS}
    rows: dict[str, Any] = {}
    beside: dict[str, Any] = {}
    n_summary_anchor = n_roc_anchor = 0
    for tkey in ("target_0.01", "target_0.05"):
        zt = z_of[tkey]
        unflagged = unflagged_of[tkey]
        d170 = {p: {"record_now": flag_of[tkey][p], "4c_record": flag4c_of[tkey][p]} for p in RELEASED_CORES}
        rows[tkey], beside[tkey] = {}, {}
        for ps in PROBE_SETS:
            for lvl in LEVELS:
                key = f"{ps}|{lvl}"
                r_ = {g: scoring(zt, names, ps, lvl, BASELINES, BASELINE_SCORES) for g, names in groups.items()}
                r_["unflagged_pairs"] = {**scoring(zt, unflagged, ps, lvl, BASELINES, BASELINE_SCORES),
                                         "pairs_from": f"{SUMMARY_4C[tkey]}'s unflagged row (CHOICE 10)",
                                         "d170_flags": d170}
                r_["gpu_control"] = control_row(zt[CONTROL_CORE], ps, lvl, BASELINES)
                rows[tkey][key] = r_
                # the summary's anchor (CHOICE 12): this module's pooling of the records' Gaussian and direct counts
                # reproduces the 4c summaries' blocks exactly; CHOICE 17: its ROC path on the 4c scores reproduces their
                # ROC blocks, and every KS and ATC row covers the 4c row's cells
                checks: list[tuple[str, Any, Any]] = []
                mine = {g: scoring(zt, names, ps, lvl, FOURC_DETECTORS, {}) for g, names in groups.items()}
                mine["unflagged_pairs"] = scoring(zt, unflagged, ps, lvl, FOURC_DETECTORS, {})
                ctrl = control_row(zt[CONTROL_CORE], ps, lvl, FOURC_DETECTORS)
                s_rel = s4c[tkey]["rows"][key]
                for c in CLASSES:
                    for d in FOURC_DETECTORS:
                        for g in ("released_pairs", "unflagged_pairs"):
                            checks.append((f"{tkey} {key} {g} {d}|{c}", mine[g].get(f"{d}|{c}"), s_rel[g].get(f"{d}|{c}")))
                        checks.append((f"{tkey} {key} control {d}|{c}", ctrl[f"{d}|{c}"], s_rel["gpu_control"][f"{d}|{c}"]))
                        if tkey == "target_0.01":
                            for g in groups:
                                if g != "released_pairs":
                                    checks.append((f"{tkey} {key} {g} {d}|{c}", mine[g][f"{d}|{c}"],
                                                   s4c["lora_target_0.01"]["rows"][key][g][f"{d}|{c}"]))
                    checks.append((f"{tkey} {key} control share|{c}", ctrl[f"share_of_draws_with_a_control_cell_in_class|{c}"],
                                   s_rel["gpu_control"][f"share_of_draws_with_a_control_cell_in_class|{c}"]))
                for g, names in (("released_pairs", RELEASED_CORES), ("unflagged_pairs", unflagged)):
                    checks.append((f"{tkey} {key} {g} n_cells", r_[g].get("n_cells"), s_rel[g].get("n_cells")))
                    if names:
                        roc4 = roc_auc(z4c_of[tkey], names, ps, lvl, FOURC_SCORES)
                        checks += [(f"{tkey} {key} {g} 4c scores {k}", v, s_rel[g].get(k)) for k, v in roc4.items()]
                        n_roc_anchor += len(roc4)
                if tkey == "target_0.01":
                    for g in groups:
                        if g != "released_pairs":
                            checks.append((f"{tkey} {key} {g} n_cells", r_[g].get("n_cells"),
                                           s4c["lora_target_0.01"]["rows"][key][g].get("n_cells")))
                bad = [w for w, a, b in checks if canon(a) != canon(b)]
                if bad:
                    refuse(f"the summary anchor: {len(bad)} of {len(checks)} blocks differ from the 4c summaries "
                           f"(first {bad[:3]}); nothing written")
                n_summary_anchor += len(checks)
                # the 4c rows beside: read from the 4c summaries; LoRA at 5% computed here (CHOICE 9, 12)
                b_ = {"released_pairs": s_rel["released_pairs"], "unflagged_pairs": s_rel["unflagged_pairs"],
                      "gpu_control": s_rel["gpu_control"],
                      "source": SUMMARY_4C[tkey] + " (read, not recomputed; D-255)"}
                if tkey == "target_0.01":
                    s_l = s4c["lora_target_0.01"]["rows"][key]
                    b_.update({g: s_l[g] for g in groups if g != "released_pairs"})
                    b_["lora_source"] = SUMMARY_4C["lora_target_0.01"] + " (read, not recomputed)"
                else:
                    b_.update({g: {**scoring(zt, names, ps, lvl, FOURC_DETECTORS, FOURC_SCORES),
                                   "auc_rule": "not available: no record holds the AUC rule for the LoRA runs at 5%"}
                               for g, names in groups.items() if g != "released_pairs"})
                    b_["lora_source"] = ("computed here from the D-381 records' 4c statistics at 5% (healthcheck_d156's "
                                         "functions; anchored by D-248's twin and the 5% depth records): 4c has no 5% "
                                         "LoRA arm (D-278)")
                b_["bbse"] = ("BBSE's ROC AUC (|w|) is in this module's rows beside KS and ATC, a ranking line with no "
                              "flag (D-392 (1)); its identity with the direct ratio holds within a cell only")
                beside[tkey][key] = b_
    for r, h in _RESULT_READS.items():
        inputs[r] = h

    pk = f"{PRIMARY['probe_set']}|{PRIMARY['level']}"
    cls = PRIMARY["class"]
    prim = rows["target_0.01"][pk][PRIMARY["row"]]
    g4c = beside["target_0.01"][pk][PRIMARY["row"]]
    if prim.get("n_cells") != g4c.get("n_cells"):           # P-C5.2 compares ROC AUCs over the same cells
        refuse(f"the primary row covers {prim.get('n_cells')} cells, the 4c row {g4c.get('n_cells')}; nothing written")
    predictions = score_predictions(g4c[f"gaussian|{cls}"]["pooled_from_mean_counts"],
                                    g4c[f"roc_auc|abs_log_gaussian_ratio|{cls}"]["mean"],
                                    prim[f"ks|{cls}"]["pooled_from_mean_counts"], prim[f"atc|{cls}"]["pooled_from_mean_counts"],
                                    prim[f"roc_auc|abs_log_atc_ratio|{cls}"]["mean"])
    code1 = P.code_block(CODE_FILES)
    if canon(code1["sha256"]) != canon(code0["sha256"]) or any(sha256_file(ROOT / f) != h for f, h in inputs.items()):
        refuse("to write: code or an input changed during the run")
    config = {"primary": PRIMARY, "baselines": list(BASELINES), "classes": {c: list(cut) for c, cut in CLASSES.items()},
              "estimator": "healthcheck_summary_d156.pooled_counts and pair_cluster_index, unchanged (D-255, "
                           f"D-177 (b): B = {HS.B}, seed {HS.SEED}, rows of at least {HS.MIN_PAIRS} pairs)",
              "roc": ("per draw across a row's cells: mean KS D, |log ATC ratio| and |BBSE w| (beside, no flag) "
                      "(healthcheck_summary_d156's lines); a cell whose score is not finite in a draw (BBSE with no "
                      "defined seed) is left out of that draw's AUC, which is the AUC over the other cells, and the row "
                      "reports the cells left out (cells_left_out, CHOICE 16); a draw then left with no cell in a class "
                      "or none outside it has an undefined AUC, counted"),
              "unflagged_pairs": {t: {"pairs": p, "from": f"{SUMMARY_4C[t]}'s unflagged row (CHOICE 10)"}
                                  for t, p in unflagged_of.items()},
              "records": per_record}
    payload = {**provenance_block(f"{RUN_KIND}_summary"), "preregistration": PREREG, "code": code0,
               "reused_files_sha256": {f: sha256_file(CODE_ROOT / f) for f in REUSED_SHA256},
               "registered_parameters": registered_all, "blas_threads": threads,
               "elapsed_s": round(time.time() - t_start, 1),
               "config": config, "config_hash": canonical_hash(config),
               "predictions": predictions,
               "verdict_scope": ("P-C5.1-3 are scored on the released pairs, 1%, selected C, item level, the primary class, "
                                 "pooled point estimates; every other row (C=1, cluster level, 5%, LoRA, the GPU control, "
                                 "the unflagged pairs, other classes) is beside, never a verdict (D-381 (4))"),
               "band_note": "per-draw bands cover partitions only, pooled intervals cover pairs only (D-166); the nine LoRA "
                            "runs share one pre model, so a run-cluster interval covers variation between runs only (D-278)",
               "bbse": BBSE_TEXT, "ks": KS_TEXT,
               "summary_anchor": {"identical": True, "n_blocks": n_summary_anchor, "n_roc_blocks_on_4c_scores": n_roc_anchor,
                                  "n_ks_arrays_identical_across_targets": n_ks_identical,
                                  "rule": "this module's pooling of the records' Gaussian and direct counts equals the 4c "
                                          "summaries' released-pair, unflagged-pair, GPU-control and (1%) LoRA blocks "
                                          "exactly; its ROC path on the released pairs' 4c scores (their anchored 4c "
                                          "npz) equals those summaries' ROC blocks and cell counts (released and "
                                          "unflagged rows, both targets); every KS and ATC row covers the 4c row's "
                                          "cells; each unit's ks_mean_D arrays are the same bytes at 1% and 5% "
                                          "(CHOICES 10, 12, 17)"},
               "input_files_sha256": inputs, "rows": rows, "beside_4c": beside}
    digest = write_results(out, payload, exclusive=True)
    os.chmod(out, 0o444)
    print(f"Written: {rel(out)}\nsha256:  {digest}")
    for k_, v_ in predictions.items():
        print(f"{k_}: {v_['verdict']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="D-381, D-392: the C5 baselines KS and ATC, BBSE beside")
    ap.add_argument("--pair", choices=HC.PAIRS, default=None)
    ap.add_argument("--fpr5", action="store_true", help="the 5%% target (beside; D-248)")
    ap.add_argument("--summary", action="store_true", help="pool the 38 records named by " + RUN_MANIFEST)
    ap.add_argument("--n-draws", type=int, default=REGISTERED["n_draws"])
    ap.add_argument("--seeds", type=int, nargs="*", default=REGISTERED["seeds"])
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="TESTING: a subset of layers")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.summary == (args.pair is not None):
        refuse("give exactly one of --pair or --summary")
    if args.summary:
        if args.fpr5 or args.layers is not None or args.n_draws != REGISTERED["n_draws"] or args.seeds != REGISTERED["seeds"]:
            refuse("--summary takes no unit parameters (it pools both targets)")
        return summary_job(args)
    return run_job(args)


if __name__ == "__main__":
    raise SystemExit(main())
