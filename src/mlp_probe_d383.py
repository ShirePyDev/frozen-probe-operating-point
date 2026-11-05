"""D-383: do C1, C4 and C5 hold for a second probe family, a one-hidden-layer MLP probe on the same cached features?

Pre-registered in D-383 (the author's message of 2026-10-07, step 4 item 3) before any of this code or its numbers
existed; details by the closest D-entries (D-181): D-117, D-152, D-156, D-164, D-241, D-248, D-255, D-282, D-289, D-322.

POPULATION (D-383 (1)): the nine released pairs of Table I (populations.UPDATE_PAIRS_REPAIR, Table I's order) and the
Mistral v0.1 -> v0.2 control row beside (Table I's tenth row), on the Phase 1 caches of record (src/input_registry.py;
the bytes each pair's depth record read), at each pair's D-117 selected layer L* of the LINEAR probe
(results/selection_sweep_d165.json, D-171; the dense pair takes the Mistral pair's, D-180). No layer sweep. Seeds 0-4;
1000 D-156 draws plus the frozen split; item level primary, the D-164 cluster level beside; 1% primary, 5% beside.

PROBE (D-383 (2)): sklearn MLPClassifier, one hidden layer, ReLU, adam, the fixed hyperparameters MLP_FIXED (max_iter
MLP_MAX_ITER, tol MLP_TOL and every setting adam without early stopping reads; validation_fraction (early stopping
only), momentum, nesterovs_momentum and power_t (sgd only) and max_fun (lbfgs only) are unused here and left at
sklearn's defaults), random_state = the seed. Inputs standardised exactly as
paired.fit_probe standardises them for the linear probe (the fit rows' own mean and sd, an sd below 1e-8 set to 1),
then the float64 copy of those features is fitted (LogisticRegression's lbfgs also works in float64). The SCORE is the
output logit: the forward pass of the fitted network (coefs_, intercepts_) without the output sigmoid, computed as
sklearn's own forward pass computes it; every fit checks that sigmoid(score) equals predict_proba[:, 1] bit for bit on
its fit rows and refuses otherwise. Fit rows and order are the selected linear probe's (each seed's D-117 80% split of
probe_train in permutation order); the post-update refit (the refit oracle; depth's post-refit AUROC) uses the same rows
and order on the post model's probe_train (D-180).

SELECTION (D-383 (2), D-117 style; --select): at L*, on each seed's ho20 split of probe_train (paired.d117_split, the
shared definition of src/regularisation_sweep.py's split), PRE model only: every config of WIDTHS x ALPHAS is fitted on
the seed's fit rows and scored on its ho20 rows by regularisation_sweep.auroc (the linear selection's AUROC); the
argmax of the mean ho20 AUROC over the five seeds; values within rtol 1e-12 of the maximum are tied (D-183 (2)), ties
to the smaller width, then the larger alpha. The job loads pre/probe_train only: threshold_calib, eval and every post
cache are never opened by it.

QUANTITIES (D-383 (3)), each computed by the reviewed linear-probe code with only the probe replaced (compute()):
  Table I     depth_d156.main's cell lines at L*: paired.repartition_estimate at both levels, depth_d156.floored_draws,
              the frozen curve and its hashes, the AUROC block (pre, frozen, post_refit, drop_frozen; metrics.auroc and
              depth_d156.weighted_auroc with the D-152 P5 item bootstrap); delta TPR's D-282 combined interval by
              table3_tpr_d282.anchored_replicates (table1_tpr_d282.main's lines as a function), item level.
  Table II    4d A (repair_d156.main, family 'repair'): arms frozen, restandardise (class-balanced: the post model's
              probe_train fit rows), quantile_thr, quantile_thr_benign (= R_same), refit_oracle, through
              repair_d156.partition and repair_d156.summarise; the floor (floor_ratio_to_target, tpr_pre) in every arm
              block. 4d B (poolstudy_d156.main's lines for the three Table II configurations and the frozen arm):
              'score|R_fresh|n=1000', 'score|traffic|pi=0.01|n=1000' (quantile transfer), 'feature|traffic|pi=0.01|
              n=1000' (traffic restandardisation), with poolstudy_d156.partitions / draw_pools / feature_stats, its
              200 draws plus the frozen split, option B for the Aegis pairs (D-306) and the registered design for
              WildGuardMix (D-283).
  4c          healthcheck_d156.main's cell lines at L*: thresholds, health, crosses per partition, both levels, the
              three detectors against the three classes, the per-draw arrays and confusion counts of the L* cell.
Restandardisation with the MLP: the post features standardised with the reference's mean and sd, then the network
(the linear arm's ((x - mu') / sd') . w + b with the network in place of w, b). Where a quantity is computed inside a
reviewed main(), its lines are repeated here and pinned by the anchor; no reviewed module is edited (REUSED_SHA256).

ANCHOR (D-383 (5); --anchor): with the probe set to the linear one (LinearFamily: paired.fit_probe at the D-117
selected C of record), compute() at the anchor pair's L* must reproduce, as canonical JSON text or bit for bit, the
records of record there: at 1% the depth cell, Table I's D-282 block (table1_tpr_d282_r2.json), the 4d A cell's five
arms, the 4d B record's four configurations, the health-check cell and its per-draw arrays; at 5% the _fpr5 depth cell,
table1_tpr_d282_fpr5.json's block, the moment family's 5% cell (frozen, restandardise, refit_oracle) and the _fpr5
health-check cell and arrays (a WildGuardMix pair has no 5% moment record: --anchor --pair WG_... skips it, CHOICES
(20)). The D-170 block of the depth and health-check cells is compared by its own rule (CHOICES (17)). Every record is
bound to its committed manifest. Every MLP job (selection, per pair) and the summary refuse unless
results/mlp_probe_d383_anchor_<ANCHOR_PAIR>.json exists, passed with every comparison kind present, and was made by this
code (code block, reused-file sha256 and library versions, CHOICES (19)); at a final name it must also be a registered
anchor written at its final name.

PREDICTIONS (D-383 (4); --summary; selected MLP, L*, 1%, item level; HELD or FAILED): P-MLP.1 the frozen paired ratio
(D-156 mean) is >= 2 in at least 6 of the 9 released pairs (the linear probe: 8 of 9, Table I); P-MLP.2 the R_same
threshold repair's FPR/target D-156 mean lies in (0.5, 2) in all 9; P-MLP.3 |delta AUROC| < 0.05 in at least as many
released pairs as the linear probe's; P-MLP.4 the Gaussian health ratio's pooled sensitivity for the primary class
(two_sided_2x) over the nine L* cells is >= 0.8. Every other row is beside, with the linear probe's rows of record and
D-177's accuracy line in every per-pair prediction table (P-MLP.1, P-MLP.2); P-MLP.3 is additive (MLP minus linear per
pair); P-MLP.4: pooled, no per-pair table; MLP minus linear beside (no D-177 line).

MODES and outputs (written once, .partial then linked, mode 444; only registered parameters, the 6-thread BLAS pin,
an unset GUARDRAIL_CACHE_DIRS, the pinned selection, cluster manifest and record manifests (PINNED_*) and the reused
files at their pinned bytes may write a final name; a test writes outside results/; every record says whether it was
written at its final name ('final_name'), and a final name binds only records written at theirs (CHOICES (18))):
  --anchor [--pair P]     results/mlp_probe_d383_anchor_<P>.json (P = ANCHOR_PAIR for the final name)
  --select --pair P       results/mlp_probe_d383_selection_<P>.json (the nine released pairs)
  --pair P                results/mlp_probe_d383_<P>.json and _draws.npz (the nine and the control row)
  --summary               results/mlp_probe_d383.json from the records named by logs/mlp_probe_d383.sha256

CHOICES (D-181; each also in every record's 'choices'):
  (1) A separate module: no hook in a reviewed module; their functions are called and the reviewed main()s' lines
      repeated where needed, pinned by the anchor and REUSED_SHA256 (D-383 (5); D-363 (4)'s precedent).
  (2) MLP settings fixed and stated (MLP_FIXED): sklearn's adam defaults written out, max_iter 200, tol 1e-4,
      n_iter_no_change 10, no early stopping (it would hold out part of the fit rows, so the MLP would fit on fewer
      rows than the linear probe's D-117 80%); trained in float64 on the linear pipeline's standardised features.
      Non-convergence is recorded per fit, not refused, as paired.fit_probe records it (D-383 (2); D-152 P1).
  (3) Score = the output logit from coefs_/intercepts_; sigmoid(score) == predict_proba[:, 1] checked per fit.
  (4) Rows and order: the selected linear probe's (D-117 permutation order) for the selection fits, the final fits and
      the post refits; one probe set (D-383 names one selected MLP; no C=1-style arm) (D-117, D-180).
  (5) Selection: P.d117_split, regularisation_sweep.auroc, only pre/probe_train loaded; ties within rtol 1e-12 (D-183
      (2)) to the smaller width, then the larger alpha (D-383 (2)).
  (6) The control row takes the Mistral pair's MLP selection (the same pre bytes, checked; D-180) and gets Table I's
      quantities only: Table II's (D-289) and 4c's (D-241, D-255) populations do not include it.
  (7) Table II's 4d B columns as poolstudy_d156 computes them (its 200 draws plus the frozen split, option B for the
      Aegis pairs, the registered design for WildGuardMix, its pools), not on 1000 D-156 draws: D-383 (3) "as the
      reviewed linear-probe code computes it" (D-306, D-322). Listed for the author: see OPEN below.
  (8) 5% beside: every quantity at 0.05 on the same fits and draws (D-248); the linear rows of record beside where a
      5% record exists (depth _fpr5, table1_tpr_d282_fpr5, healthcheck _fpr5, the moment family's frozen,
      restandardise and refit_oracle for the seven Aegis pairs); the 4d B columns and the quantile arms have none. The
      5% 4d B columns therefore have no record to anchor against (4d B's records are at 1% only): they are beside only.
  (9) D-282's combined interval at the item level only, both targets (D-282 (2)), anchored on the MLP's own cell.
  (10) 4c's AUC rule reads the MLP's own AUROC block at L* (D-241 (4): the cell's depth AUROCs).
  (11) P-MLP.3's delta AUROC = post (frozen) minus pre AUROC = -drop_frozen.mean_over_seeds at L*, for the MLP and the
       linear rows alike.
  (12) P-MLP.4 by healthcheck_summary_d156.scoring (D-255's pooled_counts, pair-cluster interval) over the nine L*
       cells; the linear row beside computed the same way from the health-check records' L* arrays, Table IV's
       all-cell value beside it; an undefined pooled sensitivity (no cell-draw in the class) is FAILED, reason stated.
  (13) D-177's accuracy line in every per-pair prediction table: realised = the MLP's value, predicted = the linear
       probe's row of record (and, for P-MLP.2, the target 1); ratio quantities only; a zero is a miss (D-322 (4)).
       Additive quantities (delta TPR, delta AUROC) get the MLP-minus-linear difference instead. P-MLP.4: pooled, no
       per-pair table; MLP minus linear beside (no D-177 line).
  (14) Anchor pair qwen_base2instruct, the cheapest (15 layers of 3584 dims, the smallest caches); its D-170 flag does
       not bear on an equality of code paths.
  (15) The D-170 flag is carried per row; P-MLP is scored over the nine as D-383 (4) words it, flagged rows marked.
  (16) Records of record read through provenance.read_result and bound to their committed manifests; the final name
       needs the pinned manifest bytes.
  (17) The D-170 block (input_verification) of the anchor's depth and health-check cells is compared by its own rule,
       not as a computed value: pin_only_inputs and rule equal; the flag equal, or the record's PIN_ONLY_FLAG cleared
       to None with pin_flag_cleared_by equal to input_registry.d168_final() (D-331 cleared the flag after the records
       were written); the records' blocks stored beside as stored_input_verification (opcase_d384's precedent). When a
       job's verification block names the final D-168 record, results/d168_final_verdicts.json is listed in
       input_files_sha256 at its pinned sha256 (meandiff_tpr_d282's P3 precedent).
  (18) At a final name the per-pair job requires each pre fit to be the selection's fit of that seed (weights_sha256),
       checked after the five pre fits and before the post refits and the draws (a test name records it only); every
       record carries 'final_name', and a final name binds only an anchor and a selection written at their final
       names; the summary carries the per-row result, its count and n_not_converged.
  (19) Library versions (numpy, scipy, scikit-learn; paired.code_block) of every bound record equal this run's: the
       anchor binding, the selection binding of the per-pair job and the summary (sklearn's MLP initialisation and
       adam's shuffling depend on the version).
  (20) The 5% moment record of the linear probe (CHOICES (8)) is required for every Aegis pair (the anchor and the
       summary refuse one its manifest does not name, or names but is not on disk); a WildGuardMix pair has none
       (the moment family's 5% population is the seven Aegis pairs), so its anchor and summary rows skip it.
OPEN (stop-list for the author, D-181): (7) uses 200 draws for the 4d B columns where D-383 (1) says 1000 D-156 draws;
an undefined P-MLP.4 sensitivity is scored FAILED (12); the accuracy line's 'predicted' is the linear row (13).
RECORDED: CHOICES (1)-(20) and OPEN (7), (12) and (13) are recorded in the review-outcome D-entry of this fix round
(D-181); (17)-(20) were added by the fix round.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
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
import poolstudy_d156 as PS  # noqa: E402
import populations as POP  # noqa: E402
import regularisation_sweep as RS  # noqa: E402
import repair_d156 as RP  # noqa: E402
import table3_tpr_d282 as T3  # noqa: E402
from provenance import (  # noqa: E402
    _RESULT_READS, _repo_rel, canonical_hash, commit_staged, ensure_utf8_stdout, input_identity_problems, provenance_block,
    read_result, resolve_cache, sha256_bytes, sha256_file, stage_bytes,
)

ROOT = Path(__file__).resolve().parent.parent
CODE_ROOT = Path(__file__).resolve().parent.parent      # where the source files are hashed (ROOT may be a test tree)
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
OWN = "src/mlp_probe_d383.py"
CODE_FILES = list(dict.fromkeys([OWN, "src/regularisation_sweep.py", *D.CODE_FILES, *RP.CODE_FILES, *HC.CODE_FILES,
                                 *HS.CODE_FILES, *T3.CODE_FILES, *PS.CODE_FILES]))
# the reviewed files whose functions are called or whose main() lines are repeated, at the bytes this module was written
# and anchored against (master fb329e0); a final name refuses other bytes
REUSED_SHA256 = {
    "src/depth_d156.py": "a1f7b50ce70395f290b681f336e5a5e95e1428a2ca7c5915818644ad70f26734",
    "src/repair_d156.py": "01211e1ec9b69adf2bcc7b738e80a06f07c229c283f105ced11f05083cbe358e",
    "src/healthcheck_d156.py": "0ec2687ecdd887a28d4b8fb7a4db4b9081b69ea3fcf2b175c345279b4ed74de7",
    "src/healthcheck_summary_d156.py": "b149b66181fe1c2d953d255355e44e2d15651bff9ccf8a456819092db526ccf5",
    "src/table1_tpr_d282.py": "820778858ad01780d77bcfd7899ff3f8b816a35f3a8df0b8d05087bdee4825bb",
    "src/table3_tpr_d282.py": "22cab06f8ecef3d778429a3789d75473ea30f21b39ef58f573170ffcb713dbdc",
    "src/poolstudy_d156.py": "e39614ed8f7e86f9f5251aaad25f44329c56f4fa3b70c3ffb77f8ce2ffb64908",
    "src/regularisation_sweep.py": "8c475baf246c9cc59459555832ed644d9d30e692579752dc0f83b059ab0cf269",
    "src/paired.py": "80429fa57f0267236a2af1670c74ef327802598257ab75834fddf64a70d97195",
    "src/metrics.py": "f7aac316793d69097939ecbeb1de0719534d0bb0cdff04534a2d7de0a454b9ed",
}
LEVELS = D.LEVELS
PSET = "selected"           # the probe-set key of every cell and array: the reviewed scripts' form (here: the selected MLP)
REGISTERED = {"n_draws": P.N_DRAWS, "n_boot": D.REGISTERED["n_boot"], "n_reps": PS.N_REPS, "seeds": [0, 1, 2, 3, 4]}
TARGETS = {"fpr_1pct": P.TARGET_FPR, "fpr_5pct": D.TARGET_5}          # 1% primary, 5% beside (D-383 (1), D-248)
PRIMARY = "fpr_1pct"
PARTS = ("probe_train", "threshold_calib", "eval")
RELEASED = [s[len("repair_"):] for s in POP.UPDATE_PAIRS_REPAIR]        # Table I's order
CONTROL_ROW = "mistral_v01_v02_dense"                                   # Table I's tenth row (D-180), beside
ROWS = [*RELEASED, CONTROL_ROW]
ANCHOR_PAIR = "qwen_base2instruct"                                      # CHOICES (14)
# --- the probe (D-383 (2)) ---
WIDTHS = (64, 256)
ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
GRID = [(w, a) for w in WIDTHS for a in ALPHAS]
TIE_RTOL = 1e-12                                                        # D-183 (2)
MLP_MAX_ITER = 200
MLP_TOL = 1e-4
MLP_FIXED = {"activation": "relu", "solver": "adam", "batch_size": "auto", "learning_rate": "constant",
             "learning_rate_init": 1e-3, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-8, "shuffle": True,
             "early_stopping": False, "n_iter_no_change": 10, "max_iter": MLP_MAX_ITER, "tol": MLP_TOL,
             "warm_start": False, "verbose": False}
FIT_DTYPE = "float64"
# --- the quantities ---
REPAIR_ARMS = tuple(RP.FAMILIES["repair"]["arms"])        # frozen, restandardise, quantile_thr, quantile_thr_benign, refit_oracle
MOMENT_ARMS_5 = ("frozen", "restandardise", "refit_oracle")   # the 5% linear record of these arms is the moment family's
# the comparison kinds every passed anchor holds (moment_fpr5 too for an Aegis anchor pair, CHOICES (20))
ANCHOR_KINDS = ("depth", "table1_tpr", "repair", "healthcheck", "healthcheck_array", "pool_4dB", "input_verification")
TRAFFIC = f"traffic|pi=0.01|n={PS.DEFAULT_POOL}"
R_FRESH = f"R_fresh|n={PS.DEFAULT_POOL}"
POOL_KEYS = ("frozen", f"score|{R_FRESH}", f"score|{TRAFFIC}", f"feature|{TRAFFIC}")
TABLE_II = (("frozen", "4dA", "frozen"), ("threshold (R_same)", "4dA", "quantile_thr_benign"),
            ("restand. (class-bal.)", "4dA", "restandardise"), ("threshold (R_fresh)", "4dB", f"score|{R_FRESH}"),
            ("quantile transfer (traffic)", "4dB", f"score|{TRAFFIC}"), ("restand. (traffic)", "4dB", f"feature|{TRAFFIC}"),
            ("refit oracle", "4dA", "refit_oracle"))
PRIMARY_CLASS, PRIMARY_DETECTOR = "two_sided_2x", "gaussian"
FACTOR = 1.5                                              # D-177
# --- records of record, the manifests that name them, and their pinned bytes ---
SELECTION = D.SELECTION_OF_RECORD
CLUSTERS = D.CLUSTERS
PINNED_SELECTION = "9b047c73cc1fcc11d768fe512006db4e5386c34072263a9e98fb6e678df509e0"      # D-171
PINNED_CLUSTERS = "0b30195cc1005ae590093963b35ed56b55f0f8cfd36eeecc9458157ce53de7c2"
MANIFEST_NAMES = {"depth": "depth_d156_20260926.sha256", "depth_fpr5": "fpr5_stage_depth_20260928.sha256",
                  "table1": "reissue_d305_20261001.sha256", "table1_fpr5": "table1_tpr_d282_20260930.sha256",
                  "repair": "repair_d156_4dA_20260930.sha256", "moment_fpr5": "moment_d156_r2_fpr5.sha256",
                  "healthcheck": "healthcheck_d156.sha256", "healthcheck_fpr5": "healthcheck_d156_fpr5.sha256",
                  "poolstudy_B": "poolstudy_4dB_B_20261001.sha256", "poolstudy_r2": "poolstudy_4dB_r2_20261001.sha256",
                  "hc_summary": "healthcheck_d156_summary_20260928.sha256",
                  "hc_summary_fpr5": "healthcheck_d156_summary_fpr5_20260930.sha256"}
PINNED_MANIFESTS = {"depth": "2d447887c5cdb152de719f0b87a1c93adc48955c8e1078814bb54a4aa7743f4c",
                    "depth_fpr5": "a379f9fc41cdb9a9946c1d1f0b17cf57660fdcef8a121f513da2cdc209bf7db4",
                    "table1": "10f8e9d40e640584c19a34063f041f3c3895c7d36f3025dd1ba9b0ccdab702c6",
                    "table1_fpr5": "6a03e9064e785c05864f1256d2e609d8eda4d7e0e3b2a5f4b198f4c7ed03c060",
                    "repair": "5f41056ffe26ebb8dfcf6141d7d40be25142d680f3cdbb184ba0ce5ea1d2b757",
                    "moment_fpr5": "8f087839262be959564c08eb6e57328b020d7475b1d3222e03911014723c7d44",
                    "healthcheck": "fe1e2bb3350a68b492bf9ca9ca6519db124196535a6f7885ffc942136b522612",
                    "healthcheck_fpr5": "92cab86acf4a2729d014853a9670fe2ebf33dedac83820df804c84316ba02cbd",
                    "poolstudy_B": "460609b6e0bbf632eb08d252339322b17331110dae923c7cd4c678de37afb317",
                    "poolstudy_r2": "5a6dbb7da4f0118f2cea5a91a53ea8625d2cfe5ecf2c1aee4752d22e84d3b932",
                    "hc_summary": "9904518eed5031e37fc5bb09656478359a05db606b9d56dcc1732330c00295d3",
                    "hc_summary_fpr5": "0898aeed4cef3d9b4ac137d4d2436aea48fbcbad9431e9232638e16ad4dfabd3"}
RUN_MANIFEST_NAME = "mlp_probe_d383.sha256"        # written by the lane's manifest job; the summary reads it
PREREG = ("D-383 (the author's message of 2026-10-07, step 4 item 3); details by the closest D-entries (D-181): D-117, "
          "D-152, D-156, D-164, D-241, D-248, D-255, D-282, D-289, D-322")
CHOICES = (
    {"choice": "a separate module: no hook in a reviewed module; their functions called, the reviewed main()s' lines "
               "repeated where needed, pinned by the anchor and REUSED_SHA256", "rule": "D-383 (5); D-363 (4)"},
    {"choice": f"MLP settings fixed and stated: {MLP_FIXED}; trained in {FIT_DTYPE} on the linear pipeline's standardised "
               "features; no early stopping (it would hold out part of the fit rows); non-convergence recorded, not "
               "refused", "rule": "D-383 (2); D-152 P1 (paired.fit_probe records convergence)"},
    {"choice": "score = the output logit from coefs_/intercepts_ (sklearn's forward pass without the output sigmoid); "
               "sigmoid(score) == predict_proba[:, 1] checked bit for bit per fit", "rule": "D-383 (2)"},
    {"choice": "fit rows and order: the selected linear probe's (D-117 80%, permutation order) for selection, final fits "
               "and post refits; one probe set", "rule": "D-117, D-180"},
    {"choice": "selection: paired.d117_split, regularisation_sweep.auroc, pre/probe_train only; ties within rtol 1e-12 to "
               "the smaller width, then the larger alpha", "rule": "D-383 (2), D-117, D-183 (2)"},
    {"choice": "the control row takes the Mistral pair's MLP selection (same pre bytes, checked) and Table I's quantities "
               "only", "rule": "D-180; D-289, D-241, D-255 (their populations)"},
    {"choice": "Table II's 4d B columns as poolstudy_d156 computes them: its 200 draws plus the frozen split, option B "
               "(Aegis) or the registered design (WildGuardMix), its pools; listed for the author",
     "rule": "D-383 (3) ('as the reviewed code computes it'); D-306, D-322, D-283"},
    {"choice": "5% beside on the same fits and draws; linear rows beside where a 5% record exists; the 5% 4d B columns "
               "have no record to anchor against and are beside only", "rule": "D-248"},
    {"choice": "D-282's combined interval at the item level, both targets, anchored on the MLP's own cell",
     "rule": "D-282 (2), D-287"},
    {"choice": "4c's AUC rule from the MLP's own AUROC block at L*", "rule": "D-241 (4)"},
    {"choice": "P-MLP.3's delta AUROC = -drop_frozen.mean_over_seeds at L* (post frozen minus pre), MLP and linear alike",
     "rule": "D-383 (3), D-180 (the AUROC block)"},
    {"choice": "P-MLP.4 by healthcheck_summary_d156.scoring over the nine L* cells; the linear row beside from the "
               "records' L* arrays; an undefined pooled sensitivity is FAILED with its reason", "rule": "D-255, D-241 (6)"},
    {"choice": "D-177's accuracy line per per-pair prediction table: realised = MLP, predicted = the linear row of record "
               "(and the target 1 for P-MLP.2); ratio quantities only; a zero is a miss; P-MLP.4: pooled, no per-pair "
               "table; MLP minus linear beside (no D-177 line)", "rule": "D-177, D-322 (4)"},
    {"choice": f"anchor pair {ANCHOR_PAIR}: the smallest caches", "rule": "D-383 (5)"},
    {"choice": "the D-170 flag carried per row; P-MLP scored over the nine as worded, flagged rows marked",
     "rule": "D-170, D-172, D-383 (4)"},
    {"choice": "records of record bound to their committed manifests; the final name needs the pinned manifest bytes",
     "rule": "D-297 (1), D-363"},
    {"choice": "the anchor's D-170 blocks (depth and health-check cells) compared by their own rule: pin_only_inputs and "
               "rule equal; the flag equal, or PIN_ONLY_FLAG cleared to None with pin_flag_cleared_by = "
               "input_registry.d168_final(); the records' blocks stored beside; the final D-168 record listed at its "
               "pinned sha256 when a verification block names it", "rule": "D-170, D-331; opcase_d384, meandiff_tpr_d282 P3"},
    {"choice": "at a final name the per-pair job requires each pre fit to be the selection's fit (weights_sha256), checked "
               "before the post refits and the draws; 'final_name' recorded, and a final name binds only an anchor and a "
               "selection written at their final names", "rule": "D-383 (2), D-152 P5"},
    {"choice": "library versions (numpy, scipy, scikit-learn) of every bound record equal this run's (sklearn's MLP "
               "initialisation and adam shuffling are version-dependent)", "rule": "D-152 P5"},
    {"choice": "the 5% moment record required for every Aegis pair; a WildGuardMix pair has none and skips it",
     "rule": "D-248; the moment family's population"},
)
AUC_NOTE = ("the AUC rule reads this record's own AUROC block at L* (D-241 (4)); the linear rule read the depth record's")
NPZ_KEYS = ("<target>|op|<level>|<quantity> (paired.repartition_estimate per draw), <target>|tpr_D282|<quantity> (the "
            "combined replicates), <target>|repair|<level>|<arm>|<quantity> and |no_update_floor, <target>|pool4dB|"
            "<configuration>|<quantity>, <target>|L<layer>|selected|<level>|<statistic> and <target>|counts|selected|"
            "<level>|<detector>|<class>|tp/fn/fp/tn (healthcheck_d156's keys), auroc|<name> (item-bootstrap "
            "replicates); index 0 the frozen split, then the draws")


def refuse(msg: str) -> None:
    raise SystemExit(f"REFUSING: {msg}")


def rel(p: Path) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(ROOT.resolve())).replace("\\", "/") if p.is_relative_to(ROOT.resolve()) else str(p)


def canon(x: Any) -> str:
    return RP.canonical(x)


def blas_threads() -> dict[str, str | None]:
    return {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}


def reused_now() -> dict[str, str | None]:
    return {f: (sha256_file(CODE_ROOT / f) if (CODE_ROOT / f).is_file() else None) for f in REUSED_SHA256}


def pinned_paths() -> dict[Path, str]:
    return {SELECTION: PINNED_SELECTION, CLUSTERS: PINNED_CLUSTERS,
            **{LOGS / MANIFEST_NAMES[k]: PINNED_MANIFESTS[k] for k in MANIFEST_NAMES}}


def final_problems(registered: bool, threads: dict[str, str | None]) -> list[str]:
    """Why a final name may not be written (empty when it may)."""
    out = []
    if not registered:
        out.append(f"non-registered parameters (registered: {REGISTERED})")
    if any(t != "6" for t in threads.values()):
        out.append(f"no 6-thread BLAS pin (D-152 P6): {threads}")
    if os.environ.get("GUARDRAIL_CACHE_DIRS"):
        out.append("GUARDRAIL_CACHE_DIRS is set; the caches must come from the input registry's directory")
    for p, want in pinned_paths().items():
        have = sha256_file(p) if Path(p).is_file() else None
        if have != want:
            out.append(f"{rel(p)} is not the pinned bytes ({(have or 'missing')[:12]} vs {want[:12]})")
    for f, want in REUSED_SHA256.items():
        have = reused_now()[f]
        if have != want:
            out.append(f"{f} is not the reviewed bytes this module reuses ({(have or 'missing')[:12]} vs {want[:12]})")
    return out


def seal(p: Path) -> None:
    """Mode 444, as the queue runner leaves every declared output (F-071)."""
    os.chmod(p, 0o444)


def write_exclusive(files: list[tuple[Path, bytes]]) -> list[tuple[Path, str]]:
    """Stage every file (.partial, fsync, verified), then name each by a hard link that fails on an existing name
    (never overwritten, D-152 P5), then mode 444."""
    staged: list[tuple[Path, Path, str]] = []
    try:
        for p, b in files:
            t, h = stage_bytes(p, b)
            staged.append((t, p, h))
        for t, p, h in staged:
            commit_staged(t, p, h)
    finally:
        for t, _, _ in staged:
            if t.exists():
                t.unlink()
    for _, p, _ in staged:
        seal(p)
    return [(p, h) for _, p, h in staged]


def write_json(path: Path, payload: dict[str, Any]) -> str:
    problems = input_identity_problems(payload)
    if problems:
        refuse(f"to write {path.name} (D-297 (1) results-writer rule): {problems}")
    return write_exclusive([(path, json.dumps(payload, indent=2, sort_keys=False, default=str).encode("utf-8"))])[0][1]


# ------------------------------------------------------------------------------------------------ records of record
class Bound:
    """Records of record: each JSON read through provenance.read_result (D-297 (1)) and refused unless its committed
    manifest names these bytes; npz files read once from their bytes, which must be the manifest's and the record's."""

    def __init__(self) -> None:
        self.man: dict[str, tuple[dict[str, str], str]] = {}
        self.npz_read: dict[str, str] = {}

    def manifest(self, key: str) -> dict[str, str]:
        if key not in self.man:
            self.man[key] = T3.manifest(LOGS / MANIFEST_NAMES[key])
        return self.man[key][0]

    def doc(self, rel_path: str, key: str | None) -> tuple[dict[str, Any], str]:
        d = read_result(ROOT / rel_path)
        h = _RESULT_READS[_repo_rel(ROOT / rel_path)]
        if key is not None and self.manifest(key).get(rel_path) != h:
            refuse(f"{rel_path} is not the file logs/{MANIFEST_NAMES[key]} names")
        return d, h

    def npz(self, rel_path: str, key: str, want: str) -> dict[str, np.ndarray]:
        b = (ROOT / rel_path).read_bytes()
        h = sha256_bytes(b)
        if self.manifest(key).get(rel_path) != h or want != h:
            refuse(f"{rel_path} is not the npz logs/{MANIFEST_NAMES[key]} and its record name")
        self.npz_read[rel_path] = h
        with np.load(io.BytesIO(b), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def listed(self) -> dict[str, str]:
        return {f"logs/{MANIFEST_NAMES[k]}": sha for k, (_, sha) in self.man.items()}

    def unchanged(self) -> list[str]:
        """What it read that has changed since (manifests and npz files; JSON reads are checked by the caller)."""
        out = [f"logs/{MANIFEST_NAMES[k]}" for k, (_, sha) in self.man.items() if sha256_file(LOGS / MANIFEST_NAMES[k]) != sha]
        return out + [r for r, h in self.npz_read.items() if sha256_file(ROOT / r) != h]


def sfx(tk: str) -> str:
    return "_fpr5" if tk == "fpr_5pct" else ""


def depth_rel(core: str, tk: str = PRIMARY) -> str:
    return f"results/depth_d156_{core}{sfx(tk)}.json"


def table1_rel(tk: str) -> str:
    return "results/table1_tpr_d282_r2.json" if tk == PRIMARY else "results/table1_tpr_d282_fpr5.json"


def hc_rel(core: str, tk: str) -> str:
    return f"results/healthcheck_d156_{core}{sfx(tk)}.json"


def pool_rel(core: str) -> tuple[str, str]:
    """4d B's primary record of a pair (POP.POOLSTUDY_D156: option B for Aegis, the _r2 re-issue for WildGuardMix)."""
    stem = f"poolstudy_d156_{core}_r2" if core.startswith("WG_") else f"poolstudy_d156_{core}_originstrat"
    if stem not in POP.POOLSTUDY_D156:
        refuse(f"{stem} is not in populations.POOLSTUDY_D156")
    return f"results/{stem}.json", ("poolstudy_r2" if core.startswith("WG_") else "poolstudy_B")


def moment5_rel(core: str, B: "Bound") -> str | None:
    """CHOICES (20): the linear probe's 5% moment record of a released pair (the moment family's frozen, restandardise
    and refit_oracle at 5%). Required for an Aegis pair: refused when logs/<moment_fpr5> does not name it, or names it
    and it is not on disk. A WildGuardMix pair has none (None) unless the manifest names one."""
    mrel = f"results/moment_d156_{core}_r2_fpr5.json"
    named = mrel in B.manifest("moment_fpr5")
    if not named:
        if core.startswith("WG_"):
            return None
        refuse(f"{mrel}, the 5% linear row of an Aegis pair, is not named by logs/{MANIFEST_NAMES['moment_fpr5']}")
    if not (ROOT / mrel).is_file():
        refuse(f"{mrel} is named by logs/{MANIFEST_NAMES['moment_fpr5']} but is not on disk")
    return mrel


def verification_problem(now: dict[str, Any] | None, rec: dict[str, Any] | None) -> str | None:
    """CHOICES (17): how a record's D-170 block (input_verification) stands to the one the registry gives now; None when
    it is accepted, else the reason. Accepted: equal; or the record's pin-only flag cleared since by the final D-168
    record (D-331): the same pin_only_inputs and rule, the record's flag PIN_ONLY_FLAG, now's None with
    pin_flag_cleared_by equal to input_registry.d168_final()."""
    if not isinstance(now, dict) or not isinstance(rec, dict):
        return "no input_verification block"
    if sorted(now.get("pin_only_inputs") or []) != sorted(rec.get("pin_only_inputs") or []):
        return "the pin-only inputs differ"
    if now.get("rule") != rec.get("rule"):
        return "the rule differs"
    if canon(now) == canon(rec):
        return None
    cleared = now.get("pin_flag_cleared_by")
    if rec.get("flag") == REG.PIN_ONLY_FLAG and now.get("flag") is None and cleared is not None \
            and canon(cleared) == canon(REG.d168_final()) and "pin_flag_cleared_by" not in rec \
            and set(now) - set(rec) == {"pin_flag_cleared_by"} and set(rec) <= set(now):
        return None
    return (f"the D-170 block differs (flag {now.get('flag')!r} against the record's {rec.get('flag')!r}; cleared by "
            f"{cleared!r} against the record's {rec.get('pin_flag_cleared_by')!r})")


def d168_listed(*blocks: dict[str, Any] | None) -> dict[str, str]:
    """CHOICES (17), meandiff_tpr_d282's P3: when a verification block names the final D-168 record
    (pin_flag_cleared_by), that record at its pinned sha256 for input_files_sha256; refused at other bytes."""
    named = {b["pin_flag_cleared_by"]["file"] for b in blocks if isinstance(b, dict) and b.get("pin_flag_cleared_by")}
    if not named:
        return {}
    want = f"results/{REG.D168_FINAL.name}"
    if named != {want}:
        refuse(f"a verification block names {sorted(named)}, not {want}")
    have = sha256_file(REG.D168_FINAL) if Path(REG.D168_FINAL).is_file() else None
    if have is None or have != REG.D168_FINAL_SHA256:
        refuse(f"{want} is not the pinned final D-168 record ({(have or 'missing')[:12]} vs {str(REG.D168_FINAL_SHA256)[:12]})")
    return {want: have}


def cell_at(doc: dict[str, Any], L: int, pset: str = PSET) -> dict[str, Any] | None:
    c = [x for x in doc["cells"] if x["layer"] == L and x["probe_set"] == pset]
    return c[0] if len(c) == 1 else None


def linear_selection(core: str) -> tuple[dict[str, Any], str, str, float, int]:
    """The D-117 selection of record for a row (D-171; the dense pair takes the Mistral pair's, D-180): (doc, sha256,
    selection stem, C, L*)."""
    sel_doc = read_result(SELECTION)
    sel_sha = _RESULT_READS[_repo_rel(SELECTION)]
    if list(sel_doc["config"]["seeds"]) != REGISTERED["seeds"] or float(sel_doc["config"]["holdout"]) != P.HOLDOUT:
        refuse("the selection's seeds or holdout differ from the registered ones")
    stem = f"depth_{core}"
    sel_stem = D.SELECTION_STEM.get(stem, D.registry_stem(stem))
    sel = next((m for m in sel_doc["models"] if m["pair"] == sel_stem), None)
    if sel is None or not sel["selected"]["arm"].startswith("C="):
        refuse(f"no logistic-C selection for {sel_stem} in {rel(SELECTION)}")
    return sel_doc, sel_sha, sel_stem, float(sel["selected"]["arm"].split("=")[1]), int(sel["selected"]["layer"])


# ------------------------------------------------------------------------------------------------ the probe families
@dataclass
class MLPProbe:
    """A fitted one-hidden-layer MLP with the linear probe's standardisation (paired.Probe.z)."""
    mu: np.ndarray
    sd: np.ndarray
    coefs: list[np.ndarray]
    intercepts: list[np.ndarray]
    diag: dict[str, Any] = field(default_factory=dict)

    def z(self, x: np.ndarray) -> np.ndarray:
        """Features standardised with the probe's (fit-set) statistics, as paired.Probe.z."""
        return (x - self.mu) / self.sd

    def logit_z(self, Z: np.ndarray) -> np.ndarray:
        """The output logit of standardised features: sklearn's _forward_pass_fast (dense dot, += intercept, ReLU in
        place on the hidden layer) without the output activation, in float64."""
        a = np.asarray(Z, dtype=np.float64) @ self.coefs[0]
        a += self.intercepts[0]
        np.maximum(a, 0, out=a)
        out = a @ self.coefs[1]
        out += self.intercepts[1]
        return out.ravel()

    def score(self, x: np.ndarray) -> np.ndarray:
        return self.logit_z(self.z(x))

    def fingerprint(self) -> str:
        return sha256_bytes(b"".join(np.ascontiguousarray(a, dtype="<f8").tobytes() for a in [*self.coefs, *self.intercepts]))


def fit_mlp(x_fit: np.ndarray, y_fit: np.ndarray, seed: int, width: int, alpha: float, row_order: str) -> MLPProbe:
    """D-383 (2): the MLP fitted on the rows IN THE ORDER GIVEN, standardised exactly as paired.fit_probe standardises
    the linear probe's inputs; refuses unless sigmoid(logit) equals predict_proba[:, 1] bit for bit on the fit rows."""
    from scipy.special import expit
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPClassifier

    if row_order not in P.ROW_ORDERS:
        raise ValueError(f"unknown row order {row_order!r}; one of {P.ROW_ORDERS}")
    mu, sd = x_fit.mean(0), x_fit.std(0)                   # paired.fit_probe's lines
    sd = np.where(sd < 1e-8, 1.0, sd)
    xs = (x_fit - mu) / sd
    ys = np.asarray(y_fit).astype(int)
    Z = np.asarray(xs, dtype=FIT_DTYPE)
    clf = MLPClassifier(hidden_layer_sizes=(int(width),), alpha=float(alpha), random_state=int(seed), **MLP_FIXED)
    c0, w0 = time.process_time(), time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(Z, ys)
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    if [int(c) for c in clf.classes_] != [0, 1] or clf.out_activation_ != "logistic" or clf.n_layers_ != 3 \
            or clf.n_outputs_ != 1:
        refuse(f"the fitted MLP is not a one-hidden-layer binary network with a logistic output ({clf.classes_}, "
               f"{clf.out_activation_}, {clf.n_layers_} layers)")
    q = MLPProbe(mu, sd, [np.array(c, dtype=np.float64) for c in clf.coefs_],
                 [np.array(b, dtype=np.float64) for b in clf.intercepts_])
    logit = q.logit_z(Z)
    if not np.array_equal(expit(logit), clf.predict_proba(Z)[:, 1]):
        refuse("sigmoid(output logit) is not predict_proba[:, 1] bit for bit on the fit rows (D-383 (2))")
    warned = any(issubclass(c.category, ConvergenceWarning) for c in caught)
    n_iter = int(clf.n_iter_)
    q.diag = {"family": "mlp", "width": int(width), "alpha": float(alpha), "row_order": row_order, "seed": int(seed),
              "n_fit": int(len(ys)), "n_iter": n_iter, "max_iter": MLP_MAX_ITER, "tol": MLP_TOL,
              "converged": bool(n_iter < MLP_MAX_ITER and not warned), "convergence_warning": warned,
              "warning_categories": sorted({c.category.__name__ for c in caught}),
              "final_loss": float(clf.loss_), "best_loss": None if clf.best_loss_ is None else float(clf.best_loss_),
              "n_samples_seen": int(clf.t_), "sigmoid_logit_equals_predict_proba": True,
              "weights_sha256": q.fingerprint(), "cpu_s": round(cpu, 2), "wall_s": round(wall, 2)}
    return q


class LinearFamily:
    """The linear probe of record (paired.fit_probe at the D-117 selected C): the anchor's family. Every score is the
    expression the reviewed script uses."""
    kind = "linear"

    def __init__(self, C: float) -> None:
        self.C = C
        self.probe_C = C

    def fit(self, x: np.ndarray, y: np.ndarray, seed: int, order: str) -> P.Probe:
        return P.fit_probe(x, y, seed, self.C, order)

    @staticmethod
    def score(q: Any, x: np.ndarray) -> np.ndarray:
        return q.score(x)

    @staticmethod
    def restd_4da(q: Any, mu: np.ndarray, sd: np.ndarray, x: np.ndarray) -> np.ndarray:
        return ((x - mu) / sd) @ q.w + q.b                    # repair_d156.main, arm 'restandardise'

    @staticmethod
    def restd_4db(q: Any, mu: np.ndarray, sd: np.ndarray, x: np.ndarray) -> np.ndarray:
        return PS.restandardised_scores(q, mu, sd, x)         # poolstudy_d156.main's feature arm

    def describe(self) -> dict[str, Any]:
        return {"family": "linear (the probe of record)", "C": self.C, "fit": "paired.fit_probe"}


class MLPFamily:
    """The D-383 MLP at the selected width and alpha."""
    kind = "mlp"
    probe_C = None

    def __init__(self, width: int, alpha: float) -> None:
        self.width, self.alpha = int(width), float(alpha)

    def fit(self, x: np.ndarray, y: np.ndarray, seed: int, order: str) -> MLPProbe:
        return fit_mlp(x, y, seed, self.width, self.alpha, order)

    @staticmethod
    def score(q: Any, x: np.ndarray) -> np.ndarray:
        return q.score(x)

    @staticmethod
    def restd_4da(q: Any, mu: np.ndarray, sd: np.ndarray, x: np.ndarray) -> np.ndarray:
        return q.logit_z((x - mu) / sd)

    @staticmethod
    def restd_4db(q: Any, mu: np.ndarray, sd: np.ndarray, x: np.ndarray) -> np.ndarray:
        return q.logit_z((x - mu) / sd)

    def describe(self) -> dict[str, Any]:
        return {"family": "mlp (one hidden layer, ReLU, adam)", "width": self.width, "alpha": self.alpha,
                "fixed": MLP_FIXED, "fit_dtype": FIT_DTYPE, "score": "output logit (pre-sigmoid)", "fit": "fit_mlp"}


# ------------------------------------------------------------------------------------------------ D-383 (2): selection
def select_config(means: dict[tuple[int, float], float]) -> tuple[tuple[int, float], list[tuple[int, float]]]:
    """The argmax of the mean ho20 AUROC; values within rtol TIE_RTOL of the maximum are tied (D-183 (2)); ties to the
    smaller width, then the larger alpha (D-383 (2)). Returns (chosen, the tied configs in tie-break order)."""
    vals = np.array(list(means.values()), dtype=float)
    if vals.size == 0 or not np.all(np.isfinite(vals)):
        refuse(f"the selection grid has non-finite mean ho20 AUROCs: {means}")
    best = float(vals.max())
    tied = [c for c, v in means.items() if np.isclose(v, best, rtol=TIE_RTOL, atol=0.0)]
    order = sorted(tied, key=lambda c: (c[0], -c[1]))
    return order[0], order


def selection_grid(X: np.ndarray, y: np.ndarray, seeds: list[int],
                   fit_fn: Callable[..., Any] = fit_mlp, log: Callable[[str], None] = print) -> dict[str, Any]:
    """D-383 (2): every config fitted on each seed's D-117 fit rows of probe_train (permutation order) and scored on that
    seed's ho20 rows. X and y are the PRE model's probe_train at L* and nothing else is passed in."""
    n = len(y)
    per: dict[tuple[int, float], list[dict[str, Any]]] = {c: [] for c in GRID}
    for seed in seeds:
        ho, fit = P.d117_split(n, seed)
        for w, a in GRID:
            q = fit_fn(X[fit], y[fit], seed, w, a, "d117_permutation")
            h = RS.auroc(q.score(X[ho]), y[ho])
            f = RS.auroc(q.score(X[fit]), y[fit])
            per[(w, a)].append({"seed": seed, "auroc_ho20": h, "auroc_fit": f, "fit": q.diag})
            log(f"  seed {seed} width {w:>3} alpha {a:g}: ho20 AUROC {h:.4f} (fit {f:.4f}); "
                f"{q.diag.get('n_iter')} iterations, {q.diag.get('cpu_s')} s CPU")
    means = {c: float(np.mean([r["auroc_ho20"] for r in v])) for c, v in per.items()}
    chosen, tied = select_config(means)
    grid = [{"width": w, "alpha": a, "mean_auroc_ho20": means[(w, a)],
             "mean_auroc_fit": float(np.mean([r["auroc_fit"] for r in per[(w, a)]])),
             "per_seed": per[(w, a)]} for w, a in GRID]
    return {"grid": grid, "selected": {"width": chosen[0], "alpha": chosen[1], "mean_auroc_ho20": means[chosen],
                                       "n_tied": len(tied), "tied_in_tie_break_order": [list(c) for c in tied]},
            "selected_fit_weights_sha256": [r["fit"].get("weights_sha256") for r in per[chosen]]}


# ------------------------------------------------------------------------------------------------ D-383 (3): computation
def pool_4db(fam: Any, probes: list[Any], X: dict[str, dict[str, np.ndarray]], m: dict[str, np.ndarray], n_cal: int,
             n_pool: int, n_reps: int, axis: str, n_seeds: int) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    """poolstudy_d156.main's lines for the frozen arm and Table II's three 4d B configurations, at both targets, on its
    partitions and pools. The restandardised scores are target-free, so they are formed once per draw."""
    f64 = lambda a: np.asarray(a, dtype=np.float64)  # noqa: E731
    ncal, hcal, neg, pos = m["ncal"], m["hcal"], m["neg"], m["pos"]
    if n_pool - n_cal != 2000:
        refuse(f"D-152's R_FRESH takes 1000 of 2000 evaluation negatives; this pool has {n_pool - n_cal}")
    origin_strat = axis == PS.STRAT_AXIS                   # option B for the Aegis pairs (D-306), as registered else
    RAW = {s: {"pool": np.vstack([X[s]["threshold_calib"][ncal], X[s]["eval"][neg]]), "H": X[s]["threshold_calib"][hcal],
               "pos": X[s]["eval"][pos]} for s in X}
    PN = {s: f64(RAW[s]["pool"]) for s in X}
    POS = {s: f64(RAW[s]["pos"]) for s in X}
    ALL_post = np.vstack([PN["post"], f64(RAW["post"]["H"])])
    n_h_src = int(hcal.sum())
    SC = {side: {part: np.column_stack([fam.score(q, RAW[side][part]) for q in probes]) for part in ("pool", "H", "pos")}
          for side in ("pre", "post")}
    all_pre = np.vstack([SC["pre"]["pool"], SC["pre"]["H"]])
    all_post = np.vstack([SC["post"]["pool"], SC["post"]["H"]])
    parts = PS.partitions(n_cal, n_pool, n_reps, axis, origin_strat)
    acc: dict[tuple[str, str], dict[str, list[float]]] = {}
    realised: list[dict[str, list[int]]] = []
    for di, (cal, ev) in enumerate(parts):
        dp = PS.draw_pools(di, cal, ev, n_cal, n_pool, n_h_src, origin_strat)
        Rf, E, rf_sub, pools = dp["R_fresh"], dp["E"], dp["R_fresh_subsets"], dp["pools"]
        e_tc = E < n_cal
        if origin_strat and di > 0:                        # D-298's counts, realised in every stratified draw
            tc = lambda idx: [int((idx < n_cal).sum()), int((idx >= n_cal).sum())]  # noqa: E731
            realised.append({"cal": tc(cal), "R_fresh": tc(Rf), "E": tc(E),
                             **{f"R_fresh|n={n}": tc(rf_sub[n]) for n in PS.POOL_SIZES}})
        idx = pools.get(TRAFFIC)
        if idx is None:
            refuse(f"draw {di}: the 1% traffic pool could not be drawn ({dp['skipped']})")
        mu, sd = PS.feature_stats(ALL_post[idx])
        XE, XP = PN["post"][E], POS["post"]
        RE = [fam.restd_4db(q, mu, sd, XE) for q in probes]
        RQ = [fam.restd_4db(q, mu, sd, XP) for q in probes]
        for tk, target in TARGETS.items():
            def put(name: str, F_pre: np.ndarray, F: np.ndarray, Q: np.ndarray) -> None:
                a = acc.setdefault((tk, name), {"fpr": [], "paired_ratio": [], "ratio_to_target": [], "tpr": [],
                                                **({"fpr_origin_tc": [], "fpr_origin_ev": []} if origin_strat else {})})
                f = float(P.floored_fpr(F).mean())
                a["fpr"].append(f)
                a["paired_ratio"].append(P.ratio_from_flags(F_pre, F))
                a["ratio_to_target"].append(f / target)
                a["tpr"].append(float(P.rate(Q).mean()))
                if origin_strat:
                    a["fpr_origin_tc"].append(float(P.floored_fpr(F[e_tc]).mean()) if e_tc.any() else float("nan"))
                    a["fpr_origin_ev"].append(float(P.floored_fpr(F[~e_tc]).mean()) if (~e_tc).any() else float("nan"))

            T = np.array([P.thr_at_fpr(SC["pre"]["pool"][cal, s], target) for s in range(n_seeds)])
            F_pre = SC["pre"]["pool"][E] >= T
            put("frozen", F_pre, SC["post"]["pool"][E] >= T, SC["post"]["pos"] >= T)
            put(f"feature|{TRAFFIC}", F_pre, np.column_stack([RE[s] >= T[s] for s in range(n_seeds)]),
                np.column_stack([RQ[s] >= T[s] for s in range(n_seeds)]))
            tq = np.array([P.quantile_transfer(all_pre[idx, s], all_post[idx, s], T[s]) for s in range(n_seeds)])
            put(f"score|{TRAFFIC}", F_pre, SC["post"]["pool"][E] >= tq, SC["post"]["pos"] >= tq)
            tr = np.array([P.thr_at_fpr(all_post[rf_sub[PS.DEFAULT_POOL], s], target) for s in range(n_seeds)])
            put(f"score|{R_FRESH}", F_pre, SC["post"]["pool"][E] >= tr, SC["post"]["pos"] >= tr)
    if origin_strat:
        off = [i + 1 for i, r in enumerate(realised) if r != PS.D298_COUNTS]
        if len(realised) != n_reps or off:
            refuse(f"{len(realised)} stratified draws, {len(off)} off D-298's counts {off[:5]}")
    out: dict[str, Any] = {tk: {} for tk in TARGETS}
    arrays: dict[str, np.ndarray] = {}
    for (tk, name), a in acc.items():
        out[tk][name] = {q: P.band_summary(np.array(v[1:]), v[0]) for q, v in a.items()}
        for q, v in a.items():
            arrays[f"{tk}|pool4dB|{name}|{q}"] = np.asarray(v, dtype="<f8")
    design = {"n_reps": n_reps, "pool_seed": PS.POOL_SEED, "origin_strat": origin_strat,
              "draws": (f"frozen split (as registered) + {n_reps} origin-stratified draws, default_rng([{PS.STRAT_DRAW_SEED}, d]) "
                        "(D-298, option B, D-306)") if origin_strat else
                       f"frozen split + the first {n_reps} of paired.repartition_draws(seed {P.AXIS_DRAW_SEED[axis]})",
              "evaluation": "the draw's evaluation part minus R_FRESH (1000 negatives); eval positives",
              "sources": "benign: cal_d negatives; harmful: threshold_calib harmful H; probe_train in no pool (D-244 (4))",
              **({"n_stratified_draws_checked": len(realised), "realised_counts_tc_ev": PS.D298_COUNTS} if origin_strat else {})}
    return out, arrays, design


def lstar_counts(v: dict[str, np.ndarray], auc_flag: bool) -> dict[str, np.ndarray]:
    """healthcheck_d156.main's confusion counts of ONE cell (its lines): per detector and class, tp/fn/fp/tn per draw."""
    rr = v["realised_paired_ratio"]
    flags = {"gaussian": v["gaussian_ratio"], "direct": v["direct_ratio"]}
    cc: dict[str, np.ndarray] = {}
    for c, cut in HC.CLASSES.items():
        y = HC.crosses(rr, cut)
        for dname in HC.DETECTORS:
            f = np.full(rr.shape, auc_flag) if dname == "auc_rule" else HC.crosses(flags[dname], cut)
            for key, val in (("tp", y & f), ("fn", y & ~f), ("fp", ~y & f), ("tn", ~y & ~f)):
                cc[f"{dname}|{c}|{key}"] = cc.get(f"{dname}|{c}|{key}", np.zeros(rr.shape, dtype=np.int64)) + val
    return cc


def compute(fam: Any, data: dict[str, dict[str, Any]], L: int, seeds: list[int], n_draws: int, n_boot: int, n_reps: int,
            axis: str, cl_axis: dict[str, Any], verification: dict[str, Any], tables_ii_4c: bool,
            tpr_cells: dict[str, dict[str, Any]] | None, auc_source: str, label: str,
            pre_check: Callable[[list[Any]], Any] | None = None) -> dict[str, Any]:
    """Every D-383 (3) quantity at layer L for probe family `fam`. tpr_cells: per target, the depth cell
    table3_tpr_d282.anchored_replicates anchors on (the record's for the anchor; None: this computation's own).
    pre_check: called on the five pre fits' weights_sha256 before the post refits and the draws (it may refuse); its
    return value is returned as 'pre_check' (CHOICES (18))."""
    pre = data["pre"]
    k = f"layer_{L}"
    X = {s: {p: data[s][p][k] for p in PARTS} for s in data}
    y_tr = pre["probe_train"]["labels"].astype(bool)
    y_cal = pre["threshold_calib"]["labels"].astype(bool)
    y_ev = pre["eval"]["labels"].astype(bool)
    m = {"ncal": ~y_cal, "hcal": y_cal, "neg": ~y_ev, "pos": y_ev}
    ncal, hcal, neg, pos = m["ncal"], m["hcal"], m["neg"], m["pos"]
    pooled_ids = [str(i) for i in pre["threshold_calib"]["item_ids"][ncal]] + [str(i) for i in pre["eval"]["item_ids"][neg]]
    n_cal, n_pool, n_neg, n_pos = int(ncal.sum()), len(pooled_ids), int(neg.sum()), int(pos.sum())
    if cl_axis["n_pooled_negatives"] != n_pool or cl_axis["n_calibration"] != n_cal or \
            cl_axis["pool_item_ids_sha256"] != sha256_bytes("\n".join(pooled_ids).encode("utf-8")):
        refuse(f"manifests/negative_clusters.json does not describe the {axis} pool")
    ev_ids = {str(i) for i in pre["eval"]["item_ids"]}
    if {str(i) for i in pre["probe_train"]["item_ids"]} & (set(pooled_ids) | ev_ids):
        refuse("a probe_train item is in the pooled negatives or in eval (F-045, D-228)")
    if {str(i) for i in pre["threshold_calib"]["item_ids"][hcal]} & (set(pooled_ids) | ev_ids):
        refuse("a threshold_calib harmful item is in the pool or in eval (D-228)")
    n_seeds = len(seeds)
    population = {"n_probe_train": int(len(y_tr)), "n_fit_per_seed": int(len(y_tr)) - int(P.HOLDOUT * len(y_tr)),
                  "n_calibration_negatives": n_cal, "n_calibration_harmful_H": int(hcal.sum()),
                  "n_evaluation_negatives": n_neg, "n_evaluation_positives": n_pos, "n_pooled_negatives": n_pool,
                  "pooled_item_ids_sha256": sha256_bytes("\n".join(pooled_ids).encode("utf-8")),
                  "eval_positives_item_ids_sha256": sha256_bytes("\n".join(str(i) for i in pre["eval"]["item_ids"][pos]).encode("utf-8"))}

    # --- the probes: the D-117 fit rows in permutation order, pre; the post refit on the same rows (D-180). The five
    # pre fits come first so that pre_check (CHOICES (18)) runs before the post refits and the draws; every fit and
    # every draw has its own seeded generator (random_state = the seed; default_rng(seed) in paired), so this order
    # changes no number (the anchor reproduces the records bit for bit) ---
    probes, refits, rows_of = [], [], []
    for seed in seeds:
        _, fit = P.d117_split(len(y_tr), seed)
        probes.append(fam.fit(X["pre"]["probe_train"][fit], y_tr[fit], seed, "d117_permutation"))
        rows_of.append(fit)
        print(f"  [{label}] seed {seed}: pre fit done", flush=True)
    pre_checked = None if pre_check is None else pre_check([q.diag.get("weights_sha256") for q in probes])
    for seed, fit in zip(seeds, rows_of):
        refits.append(fam.fit(X["post"]["probe_train"][fit], y_tr[fit], seed, "d117_permutation"))
        print(f"  [{label}] seed {seed}: post refit done", flush=True)
    # --- the D-156 draws (both levels) and the item-bootstrap counts ---
    frozen_part = (np.arange(n_cal), np.arange(n_cal, n_pool))
    draws = {LEVELS[0]: P.repartition_draws(n_cal, n_pool - n_cal, n_draws, P.AXIS_DRAW_SEED[axis]),
             LEVELS[1]: P.cluster_repartition_draws(P.cluster_units(pooled_ids, cl_axis["clusters"]), n_cal, n_draws,
                                                    P.AXIS_CLUSTER_DRAW_SEED[axis])}
    parts = {lvl: [frozen_part] + [(d[:n_cal], d[n_cal:]) for d in dr] for lvl, dr in draws.items()}
    CNT_N = P.counts_matrix(n_neg, n_boot, D.ITEM_BOOT_SEED["eval_negatives"])
    CNT_P = P.counts_matrix(n_pos, n_boot, D.ITEM_BOOT_SEED["eval_positives"])
    ONE_N, ONE_P = np.ones((1, n_neg)), np.ones((1, n_pos))
    S = lambda side, part, mask: np.column_stack([fam.score(q, X[side][part][mask]) for q in probes])  # noqa: E731
    sc = {"cal_pre": S("pre", "threshold_calib", ncal), "ev_pre": S("pre", "eval", neg),
          "cal_post": S("post", "threshold_calib", ncal), "ev_post": S("post", "eval", neg),
          "pos_pre": S("pre", "eval", pos), "pos_post": S("post", "eval", pos)}
    RS_ = lambda mask: np.column_stack([fam.score(q, X["post"]["eval"][mask]) for q in refits])  # noqa: E731
    rf = {"ev_post": RS_(neg), "pos_post": RS_(pos)}
    pool_pre, pool_post = np.vstack([sc["cal_pre"], sc["ev_pre"]]), np.vstack([sc["cal_post"], sc["ev_post"]])
    arrays: dict[str, np.ndarray] = {}

    # --- depth_d156: the AUROC block on the frozen eval split, D-152 P5 item bootstrap (target-free) ---
    au: dict[str, Any] = {"note": D.AUROC_NOTE}
    reps: dict[str, np.ndarray] = {}
    points: dict[str, np.ndarray] = {}
    for name, (pk, nk, src) in {"pre": ("pos_pre", "ev_pre", sc), "frozen": ("pos_post", "ev_post", sc),
                                "post_refit": ("pos_post", "ev_post", rf)}.items():
        pt = np.array([M.auroc(src[pk][:, s], src[nk][:, s]).value for s in range(n_seeds)])
        unit = np.array([D.weighted_auroc(src[pk][:, s], src[nk][:, s], ONE_P, ONE_N)[0] for s in range(n_seeds)])
        if not np.array_equal(pt, unit):
            refuse(f"{label} {name}: the weighted AUROC is not metrics.auroc bit for bit")
        points[name] = pt
        reps[name] = np.mean([D.weighted_auroc(src[pk][:, s], src[nk][:, s], CNT_P, CNT_N) for s in range(n_seeds)], axis=0)
    points["drop_frozen"] = points["pre"] - points["frozen"]
    reps["drop_frozen"] = reps["pre"] - reps["frozen"]
    for name in ("pre", "frozen", "post_refit", "drop_frozen"):
        au[name] = {"mean_over_seeds": float(points[name].mean()), "per_seed": points[name].tolist(),
                    "range_over_seeds": [float(points[name].min()), float(points[name].max())], **P.ci(reps[name])}
        arrays[f"auroc|{name}"] = np.concatenate([[points[name].mean()], reps[name]]).astype("<f8")

    # --- target-free parts of 4d A: the arms' post scores (repair_d156.main's lines) ---
    if tables_ii_4c:
        pre_H = S("pre", "threshold_calib", hcal)
        post = {"pool": pool_post, "pos": sc["pos_post"], "H": S("post", "threshold_calib", hcal)}
        cols_pool, cols_pos = [], []
        for q, rows in zip(probes, rows_of):
            xb = X["post"]["probe_train"][rows]
            mu_b = xb.mean(0)
            sd_b = np.where(xb.std(0) < 1e-8, 1.0, xb.std(0))
            f = lambda x, mu=mu_b, sd=sd_b, q=q: fam.restd_4da(q, mu, sd, x)  # noqa: E731
            cols_pool.append(np.concatenate([f(X["post"]["threshold_calib"][ncal]), f(X["post"]["eval"][neg])]))
            cols_pos.append(f(X["post"]["eval"][pos]))
        R_ = lambda x: np.column_stack([fam.score(q, x) for q in refits])  # noqa: E731
        arms = {"frozen": post, "restandardise": {"pool": np.column_stack(cols_pool), "pos": np.column_stack(cols_pos)},
                "quantile_thr": post, "quantile_thr_benign": post,
                "refit_oracle": {"pool": np.vstack([R_(X["post"]["threshold_calib"][ncal]), R_(X["post"]["eval"][neg])]),
                                 "pos": R_(X["post"]["eval"][pos])}}
        if tuple(arms) != REPAIR_ARMS:
            refuse(f"the 4d A arms {tuple(arms)} are not repair_d156's family 'repair' {REPAIR_ARMS}")
        d_auc = au["frozen"]["mean_over_seeds"] - au["pre"]["mean_over_seeds"]
        auc_flag = bool(d_auc <= -0.05 or au["frozen"]["mean_over_seeds"] < 0.70)

    out_t: dict[str, Any] = {}
    for tk, target in TARGETS.items():
        where = f"{label} L{L} {tk}"
        # --- depth_d156: the operating point, frozen curve and censoring, both levels ---
        reports = {lvl: P.repartition_estimate(sc["cal_pre"], sc["ev_pre"], sc["cal_post"], sc["ev_post"],
                                               sc["pos_pre"], sc["pos_post"], draws[lvl], target) for lvl in LEVELS}
        floored = {lvl: D.floored_draws(pool_pre, pool_post, parts[lvl], target) for lvl in LEVELS}
        cell: dict[str, Any] = {"layer": L, "probe_set": PSET, "probe_C": fam.probe_C, "fit_row_order": "d117_permutation",
                                "is_selected_layer": True, "input_verification": verification, "auroc": au}
        for lvl in LEVELS:
            report, arr = reports[lvl]
            rt = np.concatenate([[report["ratio_to_target"]["frozen"]], arr["ratio_to_target"]]).astype("<f8")
            fc = P.abs_log(rt).astype("<f8")
            block = {"operating_point": report,
                     "frozen_curve_abs_log_ratio_to_target": {
                         **P.band_summary(fc[1:], fc[0]), "note": D.FROZEN_CURVE_NOTE,
                         "per_draw_sha256": sha256_bytes(fc.tobytes()),
                         "per_draw_ratio_to_target_sha256": sha256_bytes(rt.tobytes())},
                     "realised_censoring": {"share_of_draws_floored": float(floored[lvl][1:].mean()),
                                            "frozen_split_floored": bool(floored[lvl][0])}}
            if lvl == LEVELS[0]:
                cell.update(block)
            else:
                cell[lvl] = {"note": D.CLUSTER_NOTE, **block}
            for q in P.D156_METRICS:
                arrays[f"{tk}|op|{lvl}|{q}"] = np.concatenate([[report[q]["frozen"]], arr[q]]).astype("<f8")
        cell["probe_fit_diagnostics"] = [q.diag for q in probes]
        cell["post_refit_fit_diagnostics"] = [q.diag for q in refits]
        # --- table1_tpr_d282 (D-282): the combined interval of TPR pre, post and delta, item level ---
        rt = np.concatenate([[reports[LEVELS[0]][0]["ratio_to_target"]["frozen"]],
                             reports[LEVELS[0]][1]["ratio_to_target"]]).astype("<f8")
        own = {"operating_point": reports[LEVELS[0]][0],
               "frozen_curve_abs_log_ratio_to_target": {"per_draw_ratio_to_target_sha256": sha256_bytes(rt.tobytes())}}
        anchor_cell = own if tpr_cells is None else tpr_cells.get(tk)
        op_t, _, reps_t = T3.anchored_replicates(sc, draws[LEVELS[0]], target, anchor_cell, CNT_P, where)
        tpr = {q: {"frozen": op_t[q]["frozen"], "mean_D156": op_t[q]["mean"], "band95_D156": op_t[q]["band95"],
                   "combined": P.ci(reps_t[q]), "replicates_sha256": sha256_bytes(reps_t[q].astype("<f8").tobytes())}
               for q in T3.QUANT}
        for q in T3.QUANT:
            arrays[f"{tk}|tpr_D282|{q}"] = reps_t[q].astype("<f8")
        blk_t: dict[str, Any] = {"target_fpr": target, "depth_cell": cell, "tpr_D282": tpr}

        if tables_ii_4c:
            # --- repair_d156 (family 'repair'): every arm per partition, both levels; the floor row ---
            rblk: dict[str, Any] = {}
            for lvl in LEVELS:
                per = [RP.partition(c_, e_, pool_pre, sc["pos_pre"], pre_H, arms, target) for c_, e_ in parts[lvl]]
                rblk[lvl] = {name: RP.summarise(per, name) for name in arms}
                rep, arr = reports[lvl]
                for q in ("fpr_post", "paired_ratio", "ratio_to_target", "tpr_post", "delta_tpr", "floor_ratio_to_target",
                          "tpr_pre"):
                    mine = np.array([p_["frozen"][q] for p_ in per[1:]])
                    if not np.array_equal(mine, arr[q]) or per[0]["frozen"][q] != rep[q]["frozen"]:
                        refuse(f"{where} {lvl}: the generic repair path is not repartition_estimate ({q})")
                fl = np.array([float(P.abs_log(p_["frozen"]["floor_ratio_to_target"])) for p_ in per]).astype("<f8")
                if not np.array_equal(fl[1:], arr["floor_abs_log"]) or fl[0] != rep["floor_abs_log"]["frozen"]:
                    refuse(f"{where} {lvl}: the floor row is not repartition_estimate's floor_abs_log")
                arrays[f"{tk}|repair|{lvl}|{RP.FLOOR_ROW}"] = fl
                for name in arms:
                    for q in ("ratio_to_target", "abs_log_ratio_to_target", "tpr_post", "paired_ratio"):
                        arrays[f"{tk}|repair|{lvl}|{name}|{q}"] = np.array([p_[name][q] for p_ in per], dtype="<f8")
            blk_t["repair_4dA"] = rblk

            # --- healthcheck_d156: the reference-set statistics per partition, both levels (its cell lines) ---
            cblk: dict[str, Any] = {"layer": L, "probe_set": PSET, "probe_C": fam.probe_C, "is_selected_layer": True,
                                    "input_verification": verification,
                                    "auc_rule": {"auroc_pre": au["pre"]["mean_over_seeds"],
                                                 "auroc_post_frozen": au["frozen"]["mean_over_seeds"], "d_auroc": d_auc,
                                                 "flag": auc_flag, "source": auc_source}}
            counts: dict[str, Any] = {}
            for lvl in LEVELS:
                rep, arr = reports[lvl]
                hper: dict[str, list[float]] = {q: [] for q in HC.STATS if q != "realised_paired_ratio"}
                mine_r = []
                for c_, e_ in parts[lvl]:
                    T = HC.thresholds(pool_pre[c_], target)
                    h = HC.health(pool_pre[c_], pool_post[c_], T)
                    for q in hper:
                        hper[q].append(h[q])
                    mine_r.append(P.ratio_from_flags(pool_pre[e_] >= T, pool_post[e_] >= T))
                v = {q: np.asarray(x, dtype="<f8") for q, x in hper.items()}
                v["realised_paired_ratio"] = np.concatenate([[rep["paired_ratio"]["frozen"]], arr["paired_ratio"]]).astype("<f8")
                if not np.array_equal(np.asarray(mine_r, dtype="<f8"), v["realised_paired_ratio"]):
                    refuse(f"{where} {lvl}: the health partitions are not repartition_estimate's")
                hb: dict[str, Any] = {q: P.band_summary(x[1:], x[0]) for q, x in v.items()}
                rr = v["realised_paired_ratio"]
                flags = {"gaussian": v["gaussian_ratio"], "direct": v["direct_ratio"]}
                hb["share_of_draws_in_class"] = {c: float(HC.crosses(rr[1:], cut).mean()) for c, cut in HC.CLASSES.items()}
                hb["share_of_draws_flagged"] = {f"{dn}|{c}": float(HC.crosses(x[1:], cut).mean()) for dn, x in flags.items()
                                                for c, cut in HC.CLASSES.items()}
                hb["share_of_draws_flagged"].update({f"auc_rule|{c}": float(auc_flag) for c in HC.CLASSES})
                cblk[lvl] = hb
                for q, x in v.items():
                    arrays[f"{tk}|L{L}|{PSET}|{lvl}|{q}"] = x
                cc = lstar_counts(v, auc_flag)
                counts[lvl] = {}
                for key, x in cc.items():
                    arrays[f"{tk}|counts|{PSET}|{lvl}|{key}"] = x
                    counts[lvl][key] = {"d156_mean": float(x[1:].mean()), "frozen_split": int(x[0])}
            blk_t["healthcheck_4c"] = cblk
            blk_t["confusion_counts_lstar"] = {PSET: counts}
        out_t[tk] = blk_t
        print(f"  [{label}] {tk}: paired ratio {cell['operating_point']['paired_ratio']['mean']:.3f}, "
              f"FPR/target {cell['operating_point']['ratio_to_target']['mean']:.3f}", flush=True)

    pool_design = None
    if tables_ii_4c:
        p4, a4, pool_design = pool_4db(fam, probes, X, m, n_cal, n_pool, n_reps, axis, n_seeds)
        for tk in TARGETS:
            out_t[tk]["pool_4dB"] = p4[tk]
        arrays.update(a4)
    return {"population": population, "auroc": au, "targets": out_t, "arrays": arrays, "pool_4dB_design": pool_design,
            "fits": {"pre": [q.diag for q in probes], "post_refit": [q.diag for q in refits]},
            "pre_weights_sha256": [q.diag.get("weights_sha256") for q in probes], "pre_check": pre_checked}


# ------------------------------------------------------------------------------------------------ inputs
def load_inputs(core: str, parts: tuple[str, ...], sides: tuple[str, ...]) -> dict[str, Any]:
    """The caches of record through the input registry, the parts and sides named only."""
    stem = f"depth_{core}"
    reg_stem = D.registry_stem(stem)
    cfg = REG.run(reg_stem)
    axis = cfg["axis"]
    if axis not in P.AXIS_DRAW_SEED:
        refuse(f"no D-156 partition seed for axis {axis}")
    data: dict[str, dict[str, Any]] = {s: {} for s in sides}
    input_sha: dict[str, str] = {}
    input_recs: dict[str, dict[str, Any]] = {}
    resolved: dict[str, str] = {}
    for side in sides:
        for part in parts:
            z, r = REG.load(cfg[f"{side}_repo"], axis, part)
            data[side][part] = z
            input_sha[f"{side}/{part}"], input_recs[f"{side}/{part}"] = r["sha256"], r
            resolved[f"{side}/{part}"] = str(resolve_cache(r["file"], REG.CACHE))
    reg_sources = REG.sources_sha256()
    if D.sources_now() != reg_sources:
        refuse("the extraction records or pins on disk are not the ones the registry read")
    for side in sides:
        for z in data[side].values():
            if "nonfinite" in z.files and z["nonfinite"].any():
                refuse(f"{core}: a cache has non-finite items")
    if len(sides) == 2:
        for part in parts:
            a, b = data["pre"][part], data["post"][part]
            if not np.array_equal(a["item_ids"], b["item_ids"]) or not np.array_equal(a["labels"], b["labels"]):
                refuse(f"{part} items or labels differ between pre and post")
    return {"stem": stem, "reg_stem": reg_stem, "cfg": cfg, "axis": axis, "data": data, "input_sha": input_sha,
            "input_recs": input_recs, "resolved": resolved, "reg_sources": reg_sources,
            "verification": REG.verification(input_recs)}


def cache_block(inp: dict[str, Any]) -> dict[str, Any]:
    return {"input_cache_sha256": inp["input_sha"],
            "input_cache_source": {k: {"file": r["file"], "expected_from": r["expected_from"], "pin_only": r["pin_only"],
                                       "resolved_path": inp["resolved"][k]} for k, r in inp["input_recs"].items()}}


def caches_unchanged(inp: dict[str, Any]) -> list[str]:
    return [f"cache {r['file']}" for r in inp["input_recs"].values()
            if sha256_file(resolve_cache(r["file"], REG.CACHE)) != r["sha256"]]


def clusters() -> tuple[dict[str, Any], str]:
    b = Path(CLUSTERS).read_bytes()
    return json.loads(b.decode("utf-8")), sha256_bytes(b)   # not a result (D-297): a manifest, hashed and listed


def bind_anchor(code0: dict[str, Any], final: bool) -> tuple[dict[str, Any], str, str]:
    """D-383 (5): no MLP output before the anchor has passed, under this code."""
    p = RESULTS / f"mlp_probe_d383_anchor_{ANCHOR_PAIR}.json"
    if not p.is_file():
        refuse(f"{rel(p)}, the anchor record, is not on disk: no MLP output is written before the anchor passes "
               "(D-383 (5))")
    doc = read_result(p)
    h = _RESULT_READS[_repo_rel(p)]
    a = doc.get("anchor") or {}
    if a.get("identical") is not True or doc.get("config", {}).get("pair") != ANCHOR_PAIR or a.get("n_comparisons", 0) <= 0:
        refuse(f"{rel(p)} is not a passed anchor of {ANCHOR_PAIR}")
    if anchor_kind_problem(a):
        refuse(f"{rel(p)} is not a passed anchor of {ANCHOR_PAIR}: {anchor_kind_problem(a)}")
    if other_revision(doc, code0):
        refuse(f"{rel(p)} was made by another revision of the code ({other_revision(doc, code0)})")
    if final and doc.get("registered_parameters") is not True:
        refuse(f"{rel(p)} is not a registered anchor")
    if final and doc.get("final_name") is not True:
        refuse(f"{rel(p)} was not written at its final name (CHOICES (18))")
    return doc, h, rel(p)


def anchor_kind_problem(a: dict[str, Any]) -> str:
    """Why an anchor block lacks a comparison kind ('' when it does not): every entry of by_kind > 0, and every kind of
    ANCHOR_KINDS present (moment_fpr5 too for an Aegis anchor pair, CHOICES (20))."""
    by_kind = a.get("by_kind") or {}
    need = set(ANCHOR_KINDS) | (set() if ANCHOR_PAIR.startswith("WG_") else {"moment_fpr5"})
    if not need <= set(by_kind) or any(not isinstance(v, int) or v <= 0 for v in by_kind.values()):
        return f"comparisons by kind {by_kind}; every one of {sorted(need)} is needed, each count positive"
    return ""


def other_revision(doc: dict[str, Any], code0: dict[str, Any]) -> str:
    """Why a record was made by another revision of the code than this run ('' when it was not): the code block's
    sha256, the reused files' sha256 or the library versions (CHOICES (19))."""
    code = doc.get("code") or {}
    why = [w for w, a, b in (("code sha256", code.get("sha256"), code0.get("sha256")),
                             ("reused-file sha256", doc.get("reused_files_sha256"), reused_now()),
                             ("library versions", code.get("libraries"), code0.get("libraries")))
           if a is None or canon(a) != canon(b)]
    return ", ".join(why)


def out_paths(name: str, out: str | None, npz: bool) -> tuple[Path, Path | None, bool]:
    final_path = (RESULTS / f"{name}.json").resolve()
    out_path = Path(out).resolve() if out else final_path
    npz_path = out_path.with_name(out_path.stem + "_draws.npz") if npz else None
    final = out_path == final_path
    for p in [out_path] + ([npz_path] if npz_path else []):
        if p.exists():
            refuse(f"{p} exists; results are never overwritten (D-152 P5)")
    if not final and out_path.is_relative_to(RESULTS.resolve()):
        refuse("a test run (--out) must not write under results/; only the run's own final name may")
    return out_path, npz_path, final


# ------------------------------------------------------------------------------------------------ the anchor
def anchor_records(core: str, L: int, B: Bound) -> dict[str, Any]:
    """The linear records of record at L* the anchor reproduces, each bound to its manifest."""
    rec: dict[str, Any] = {"files": {}}
    for tk in TARGETS:
        drel = depth_rel(core, tk)
        dd, dh = B.doc(drel, "depth" if tk == PRIMARY else "depth_fpr5")
        trel = table1_rel(tk)
        td, th = B.doc(trel, "table1" if tk == PRIMARY else "table1_fpr5")
        hrel = hc_rel(core, tk)
        hd, hh = B.doc(hrel, "healthcheck" if tk == PRIMARY else "healthcheck_fpr5")
        hz = B.npz(hd["draws_npz"]["file"], "healthcheck" if tk == PRIMARY else "healthcheck_fpr5", hd["draws_npz"]["sha256"])
        if tk == PRIMARY:
            rrel: str | None = f"results/repair_d156_{core}.json"
            rd, rh = B.doc(rrel, "repair")
        else:
            rrel = moment5_rel(core, B)                 # CHOICES (20): None for a WildGuardMix pair
            rd, rh = B.doc(rrel, "moment_fpr5") if rrel else (None, None)
        for what, d in (("depth", dd), ("repair", rd), ("healthcheck", hd)):
            if d is None:
                continue
            c = d["config"]
            if (c.get("selected_layer"), c.get("target_fpr"), list(c.get("seeds") or [])) != (L, TARGETS[tk], REGISTERED["seeds"]):
                refuse(f"the {what} record at {tk} is not at L{L}, target {TARGETS[tk]} and seeds 0-4")
        rec[tk] = {"depth": cell_at(dd, L), "depth_doc": dd, "table1": ((td["rows"].get(f"depth_{core}") or {}).get("probe_sets") or {})
                   .get(PSET, {}).get("layers", {}).get(str(L)), "repair": None if rd is None else cell_at(rd, L),
                   "has_repair_record": rd is not None, "healthcheck": cell_at(hd, L), "hc_z": hz}
        rec["files"].update({drel: dh, trel: th, hrel: hh, hd["draws_npz"]["file"]: hd["draws_npz"]["sha256"]})
        if rrel:
            rec["files"][rrel] = rh
    prel, pkey = pool_rel(core)
    pd, ph = B.doc(prel, pkey)
    if pd["config"].get("layer") != L or pd["config"].get("target_fpr") != P.TARGET_FPR:
        refuse(f"{prel} is not at L{L} and 1%")
    rec["pool"] = pd["configurations"]["selected"]
    rec["pool_doc"] = pd
    rec["files"][prel] = ph
    return rec


def anchor_compare(comp: dict[str, Any], rec: dict[str, Any], L: int) -> dict[str, Any]:
    """D-383 (5): the linear computation against the records of record, as canonical JSON text or, for arrays, bit for
    bit with the same dtype; refuses on any difference."""
    bad: list[str] = []
    n: dict[str, int] = {}
    stored: dict[str, Any] = {}

    def same(kind: str, where: str, a: Any, b: Any) -> None:
        n[kind] = n.get(kind, 0) + 1
        if b is None or canon(a) != canon(b):
            bad.append(f"{kind} {where}")

    for tk in TARGETS:
        mine, r = comp["targets"][tk], rec[tk]
        dc, rc = mine["depth_cell"], r["depth"]
        if rc is None:
            bad.append(f"depth {tk}: no record cell at L{L}")
        else:
            for key in ("probe_C", "fit_row_order", "is_selected_layer", "operating_point",
                        "frozen_curve_abs_log_ratio_to_target", "realised_censoring", "probe_fit_diagnostics",
                        "post_refit_fit_diagnostics"):
                same("depth", f"{tk} {key}", dc[key], rc.get(key))
            for key in ("note", "operating_point", "frozen_curve_abs_log_ratio_to_target", "realised_censoring"):
                same("depth", f"{tk} {LEVELS[1]} {key}", dc[LEVELS[1]][key], (rc.get(LEVELS[1]) or {}).get(key))
            for name in ("note", "pre", "frozen", "post_refit", "drop_frozen"):
                same("depth", f"{tk} auroc {name}", dc["auroc"][name], rc["auroc"].get(name))
        for q in T3.QUANT:
            same("table1_tpr", f"{tk} {q}", mine["tpr_D282"][q], (r["table1"] or {}).get(q))
        arms = REPAIR_ARMS if tk == PRIMARY else MOMENT_ARMS_5
        if tk == PRIMARY or r["has_repair_record"]:     # CHOICES (20): a WildGuardMix pair has no 5% moment record
            for lvl in LEVELS:
                for arm in arms:
                    same("repair" if tk == PRIMARY else "moment_fpr5", f"{tk} {lvl} {arm}", mine["repair_4dA"][lvl][arm],
                         ((r["repair"] or {}).get(lvl) or {}).get(arm))
        # CHOICES (17): the D-170 block is metadata, not a computed value; it is compared by its own rule (the flag the
        # final D-168 record cleared after the records were written, D-331), and the records' blocks are kept beside
        for what, mc, rc_ in (("depth", dc, rc), ("healthcheck", mine["healthcheck_4c"], r["healthcheck"])):
            rb = (rc_ or {}).get("input_verification")
            stored[f"{what} {tk}"] = rb
            n["input_verification"] = n.get("input_verification", 0) + 1
            why = verification_problem(mc.get("input_verification"), rb)
            if why is not None:
                bad.append(f"input_verification {what} {tk}: {why}")
        strip = lambda c: {k: ({kk: vv for kk, vv in v.items() if kk != "source"} if k == "auc_rule" else v)  # noqa: E731
                           for k, v in (c or {}).items() if k != "input_verification"}
        same("healthcheck", tk, strip(mine["healthcheck_4c"]), strip(r["healthcheck"]) if r["healthcheck"] else None)
        for key, x in comp["arrays"].items():
            if key.startswith(f"{tk}|L{L}|{PSET}|"):
                n["healthcheck_array"] = n.get("healthcheck_array", 0) + 1
                y = r["hc_z"].get(key[len(tk) + 1:])
                if y is None or y.dtype != x.dtype or not np.array_equal(x, y):
                    bad.append(f"healthcheck_array {key}")
    for key in POOL_KEYS:
        same("pool_4dB", key, comp["targets"][PRIMARY]["pool_4dB"][key], rec["pool"].get(key))
    if bad:
        refuse(f"the anchor (the linear probe through this module) does not reproduce the records of record in {len(bad)} "
               f"of {sum(n.values())} comparisons: {bad[:10]}")
    return {"identical": True, "n_comparisons": sum(n.values()), "by_kind": n, "layer": L,
            "kind": "every comparison as canonical JSON text (repair_d156.canonical) or, for arrays, bit for bit with the "
                    "same dtype; the health check's auc_rule.source string names the AUROC source and is left out; the "
                    "input_verification blocks of the depth and health-check cells (D-170 metadata) are compared by "
                    "CHOICES (17): pin_only_inputs and rule equal, the flag equal or the record's pin-only flag cleared "
                    "since by input_registry.d168_final() (D-331)",
            "stored_input_verification": stored}


# ------------------------------------------------------------------------------------------------ jobs
def common_config(inp: dict[str, Any], core: str, L: int, args: argparse.Namespace, cl_sha: str) -> dict[str, Any]:
    cfg, axis = inp["cfg"], inp["axis"]
    return {"pair": core, "depth_stem": inp["stem"], "registry_pair": inp["reg_stem"], "axis": axis, "layer": L,
            "seeds": REGISTERED["seeds"], "split": f"D-117 (holdout {P.HOLDOUT}), fit rows in permutation order",
            "targets": TARGETS, "primary_target": PRIMARY, "levels": list(LEVELS), "n_draws": args.n_draws,
            "n_boot": args.n_boot, "n_reps_4dB": args.n_reps, "draw_seed": P.AXIS_DRAW_SEED[axis],
            "cluster_draw_seed": P.AXIS_CLUSTER_DRAW_SEED[axis], "item_bootstrap_seeds": D.ITEM_BOOT_SEED,
            "cluster_manifest": {"file": rel(CLUSTERS), "sha256": cl_sha},
            "pre_repo": cfg["pre_repo"], "pre_rev": cfg["pre_rev"], "post_repo": cfg["post_repo"], "post_rev": cfg["post_rev"],
            "inputs_from": "src/input_registry.py (D-165)", "input_registry_sources_sha256": inp["reg_sources"]}


def anchor_job(args: argparse.Namespace) -> int:
    t0 = time.time()
    core = args.pair or ANCHOR_PAIR
    if core not in RELEASED:
        refuse(f"the anchor needs a released pair with Table II and 4c records; {core} is not one")
    registered = args.n_draws == REGISTERED["n_draws"] and args.n_boot == REGISTERED["n_boot"] and args.n_reps == REGISTERED["n_reps"]
    out_path, _, final = out_paths(f"mlp_probe_d383_anchor_{core}", args.out, npz=False)
    threads = blas_threads()
    if final:
        problems = final_problems(registered, threads) + ([] if core == ANCHOR_PAIR else [f"the anchor of record is {ANCHOR_PAIR}'s"])
        if problems:
            refuse(f"the final name: {problems}; use --out outside results/ for a test")
    code0 = P.code_block(CODE_FILES)
    _, sel_sha, sel_stem, C_sel, L = linear_selection(core)
    B = Bound()
    rec = anchor_records(core, L, B)
    for tk in TARGETS:
        dcfg = rec[tk]["depth_doc"]["config"]
        if (dcfg["selected_C"], dcfg["selected_layer"], dcfg["selection_sha256"], dcfg["n_draws"]) != (C_sel, L, sel_sha, args.n_draws):
            refuse(f"the depth record at {tk} was made on another selection or number of draws")
    cl_doc, cl_sha = clusters()
    inp = load_inputs(core, PARTS, ("pre", "post"))
    drec = rec[PRIMARY]["depth_doc"]
    if any(drec["input_cache_sha256"].get(k) != h for k, h in inp["input_sha"].items()):
        refuse(f"{depth_rel(core)} read other caches")
    print(f"D-383 anchor {core} [{inp['axis']}]: the linear probe C={C_sel:g} at L*={L} through this module; n_draws "
          f"{args.n_draws}, n_boot {args.n_boot}, n_reps {args.n_reps}; registered {registered}", flush=True)
    comp = compute(LinearFamily(C_sel), inp["data"], L, REGISTERED["seeds"], args.n_draws, args.n_boot, args.n_reps,
                   inp["axis"], cl_doc["axes"][inp["axis"]], inp["verification"], True,
                   {tk: rec[tk]["depth"] for tk in TARGETS}, f"{depth_rel(core)} (frozen eval split)", "anchor, linear")
    anchor = anchor_compare(comp, rec, L)
    stored_iv = anchor.pop("stored_input_verification")
    d168 = d168_listed(inp["verification"])
    print(f"anchor: {anchor['n_comparisons']} comparisons identical {anchor['by_kind']}", flush=True)
    changed = [w for w, a, b in (("code", P.code_block(CODE_FILES)["sha256"], code0["sha256"]),
                                 ("registry sources", D.sources_now(), inp["reg_sources"]),
                                 ("cluster manifest", sha256_file(CLUSTERS), cl_sha)) if a != b]
    changed += [r_ for r_, h in _RESULT_READS.items() if sha256_file(ROOT / r_) != h] + B.unchanged() + caches_unchanged(inp)
    if changed:
        refuse(f"changed during the run: {changed}")
    config = {**common_config(inp, core, L, args, cl_sha), "probe": LinearFamily(C_sel).describe(),
              "selection_file": rel(SELECTION), "selection_sha256": sel_sha, "selection_of_pair": sel_stem}
    payload = {**provenance_block("mlp_probe_d383_anchor"), "preregistration": PREREG, "choices": CHOICES, "code": code0,
               "reused_files_sha256": reused_now(), "registered_parameters": registered, "final_name": final,
               "blas_threads": threads, "elapsed_s": round(time.time() - t0, 1), "input_verification": inp["verification"],
               "stored_input_verification": stored_iv, "config": config,
               "config_hash": canonical_hash(config), "anchor": anchor, "population": comp["population"],
               "pool_4dB_design": comp["pool_4dB_design"],
               "records_of_record": rec["files"],
               "input_files_sha256": {**dict(_RESULT_READS), **B.listed(), **B.npz_read, rel(CLUSTERS): cl_sha, **d168},
               **cache_block(inp)}
    digest = write_json(out_path, payload)
    print(f"Written: {rel(out_path)}\nsha256:  {digest}")
    return 0


def select_job(args: argparse.Namespace) -> int:
    t0 = time.time()
    core = args.pair
    if core not in RELEASED:
        refuse(f"--select takes a released pair; {core} takes the Mistral pair's selection (D-180)" if core == CONTROL_ROW
               else f"{core} is not a released pair")
    out_path, _, final = out_paths(f"mlp_probe_d383_selection_{core}", args.out, npz=False)
    threads = blas_threads()
    registered = True                       # the selection has no draw parameters; seeds and grid are fixed
    if final:
        problems = final_problems(registered, threads)
        if problems:
            refuse(f"the final name: {problems}; use --out outside results/ for a test")
    code0 = P.code_block(CODE_FILES)
    adoc, ash, arel = bind_anchor(code0, final)
    sel_doc, sel_sha, sel_stem, C_sel, L = linear_selection(core)
    # D-117: the PRE model's probe_train only; threshold_calib, eval and every post cache are never opened
    inp = load_inputs(core, ("probe_train",), ("pre",))
    if inp["input_sha"]["pre/probe_train"] != sel_doc["input_cache_sha256"].get(f"{sel_stem}:pre/probe_train"):
        refuse("the linear selection of record was made on other probe_train bytes")
    z = inp["data"]["pre"]["probe_train"]
    X, y = z[f"layer_{L}"], z["labels"].astype(bool)
    lin = next(m_ for m_ in sel_doc["models"] if m_["pair"] == sel_stem)
    lin_cell = (lin.get("grid") or {}).get(f"C={C_sel:g}", {}).get(str(L))
    print(f"D-383 selection {core} [{inp['axis']}] at the linear L*={L}: grid {GRID}, seeds {REGISTERED['seeds']}; "
          f"ho20 = paired.d117_split's first {int(P.HOLDOUT * len(y))} of {len(y)} probe_train items", flush=True)
    res = selection_grid(X, y, REGISTERED["seeds"])
    changed = [w for w, a, b in (("code", P.code_block(CODE_FILES)["sha256"], code0["sha256"]),
                                 ("registry sources", D.sources_now(), inp["reg_sources"])) if a != b]
    changed += [r_ for r_, h in _RESULT_READS.items() if sha256_file(ROOT / r_) != h] + caches_unchanged(inp)
    if changed:
        refuse(f"changed during the run: {changed}")
    cfg = inp["cfg"]
    config = {"pair": core, "registry_pair": inp["reg_stem"], "axis": inp["axis"], "layer": L, "seeds": REGISTERED["seeds"],
              "holdout": P.HOLDOUT, "split": "paired.d117_split (src/regularisation_sweep.py's split), fit rows in permutation order",
              "selection_population": "each seed's held-out 20% of probe_train, PRE model only",
              "data_loaded": ["pre/probe_train"], "eval_used_for_selection": False, "threshold_calib_used_for_selection": False,
              "post_used_for_selection": False, "criterion": "mean ho20 AUROC over the seeds (regularisation_sweep.auroc)",
              "grid": {"width": list(WIDTHS), "alpha": list(ALPHAS)}, "tie_rule": f"within rtol {TIE_RTOL} of the maximum "
              "(D-183 (2)): the smaller width, then the larger alpha (D-383 (2))", "mlp_fixed": MLP_FIXED, "fit_dtype": FIT_DTYPE,
              "linear_selection_file": rel(SELECTION), "linear_selection_sha256": sel_sha, "linear_selection_of_pair": sel_stem,
              "linear_C": C_sel, "pre_repo": cfg["pre_repo"], "pre_rev": cfg["pre_rev"],
              "input_registry_sources_sha256": inp["reg_sources"]}
    fits = [r["fit"] for g in res["grid"] for r in g["per_seed"]]
    d168 = d168_listed(inp["verification"])
    payload = {**provenance_block("mlp_probe_d383_selection"), "preregistration": PREREG, "choices": CHOICES, "code": code0,
               "reused_files_sha256": reused_now(), "registered_parameters": registered, "final_name": final,
               "blas_threads": threads, "elapsed_s": round(time.time() - t0, 1), "anchor_record": {"file": arel, "sha256": ash},
               "input_verification": inp["verification"], "config": config, "config_hash": canonical_hash(config),
               "selected": res["selected"], "selected_fit_weights_sha256": res["selected_fit_weights_sha256"],
               "grid": res["grid"],
               "linear_probe_of_record_at_lstar": {"arm": f"C={C_sel:g}", "layer": L, **(lin_cell or {}),
                                                   "source": f"{rel(SELECTION)} models[{sel_stem}].grid"},
               "fit_cost": {"n_fits": len(fits), "cpu_s_total": round(sum(f["cpu_s"] for f in fits), 1),
                            "wall_s_total": round(sum(f["wall_s"] for f in fits), 1),
                            "n_not_converged": sum(not f["converged"] for f in fits),
                            "by_width": {str(w): {"n_fits": len(fw), "cpu_s_total": round(sum(f["cpu_s"] for f in fw), 1),
                                                  "wall_s_total": round(sum(f["wall_s"] for f in fw), 1),
                                                  "n_iter": [f["n_iter"] for f in fw], "max_iter": MLP_MAX_ITER}
                                         for w in sorted({f["width"] for f in fits})
                                         for fw in [[f for f in fits if f["width"] == w]]}},
               "input_files_sha256": {**dict(_RESULT_READS), **d168}, **cache_block(inp)}
    digest = write_json(out_path, payload)
    s = res["selected"]
    print(f"Written: {rel(out_path)}\nsha256:  {digest}\nselected: width {s['width']}, alpha {s['alpha']:g} (mean ho20 AUROC "
          f"{s['mean_auroc_ho20']:.4f}; {s['n_tied']} tied)")
    return 0


def pair_job(args: argparse.Namespace) -> int:
    t0 = time.time()
    core = args.pair
    if core not in ROWS:
        refuse(f"{core} is not one of D-383's rows {ROWS}")
    released = core in RELEASED
    registered = args.n_draws == REGISTERED["n_draws"] and args.n_boot == REGISTERED["n_boot"] and args.n_reps == REGISTERED["n_reps"]
    out_path, npz_path, final = out_paths(f"mlp_probe_d383_{core}", args.out, npz=True)
    threads = blas_threads()
    if final:
        problems = final_problems(registered, threads)
        if problems:
            refuse(f"the final name: {problems}; use --out outside results/ for a test")
    code0 = P.code_block(CODE_FILES)
    adoc, ash, arel = bind_anchor(code0, final)
    _, sel_sha, sel_stem, C_sel, L = linear_selection(core)
    mlp_sel_core = sel_stem[len("repair_"):]                 # the dense pair: the Mistral pair's (D-180)
    srel = f"results/mlp_probe_d383_selection_{mlp_sel_core}.json"
    if not (ROOT / srel).is_file():
        refuse(f"{srel}, the MLP selection this row takes, is not on disk (run --select --pair {mlp_sel_core} first)")
    sdoc = read_result(ROOT / srel)
    ssha = _RESULT_READS[_repo_rel(ROOT / srel)]
    sc_ = sdoc["config"]
    if (sc_["pair"], sc_["layer"], sc_["linear_selection_sha256"]) != (mlp_sel_core, L, sel_sha) or other_revision(sdoc, code0):
        refuse(f"{srel} is not the MLP selection of {mlp_sel_core} at L{L} under this code "
               f"({other_revision(sdoc, code0) or 'pair, layer or linear selection'})")
    if final and sdoc.get("registered_parameters") is not True:
        refuse(f"{srel} is not a registered selection")
    if final and sdoc.get("final_name") is not True:
        refuse(f"{srel} was not written at its final name (CHOICES (18))")
    width, alpha = int(sdoc["selected"]["width"]), float(sdoc["selected"]["alpha"])
    if (width, alpha) not in GRID:
        refuse(f"{srel} selected ({width}, {alpha}), which is not in the grid")
    sel_w = list(sdoc.get("selected_fit_weights_sha256") or [])
    if len(sel_w) != len(REGISTERED["seeds"]):
        refuse(f"{srel} holds {len(sel_w)} selected-fit weights, not one per seed")

    def pre_check(weights: list[Any]) -> dict[str, Any]:
        """CHOICES (18): each pre fit against the selection's fit of that seed; required at a final name."""
        same = [a is not None and a == b for a, b in zip(weights, sel_w)]
        if len(weights) != len(sel_w) or (final and not all(same)):
            refuse(f"the pre fits are not the selection's fits ({srel}): equal per seed {same}; a final name requires "
                   "every one (CHOICES (18))")
        return {"pre_fits_equal_selection_fits_per_seed": same, "n_equal": int(sum(same)), "n_seeds": len(same),
                "required": final}
    B = Bound()
    drel = depth_rel(core)
    ddoc, dsha = B.doc(drel, "depth")
    if (ddoc["config"]["selected_layer"], ddoc["config"]["selection_sha256"]) != (L, sel_sha):
        refuse(f"{drel} was made on another selection")
    cl_doc, cl_sha = clusters()
    inp = load_inputs(core, PARTS, ("pre", "post"))
    if any(ddoc["input_cache_sha256"].get(k) != h for k, h in inp["input_sha"].items()):
        refuse(f"{drel} read other caches: these are not the Phase 1 caches of record")
    if inp["input_sha"]["pre/probe_train"] != sdoc["input_cache_sha256"].get("pre/probe_train"):
        refuse(f"{srel} was made on other pre/probe_train bytes (D-180)")
    print(f"D-383 {core} [{inp['axis']}]: MLP width {width}, alpha {alpha:g} ({srel}) at the linear L*={L}; Table II and "
          f"4c {released}; n_draws {args.n_draws}, n_boot {args.n_boot}, n_reps {args.n_reps}; registered {registered}",
          flush=True)
    fam = MLPFamily(width, alpha)
    comp = compute(fam, inp["data"], L, REGISTERED["seeds"], args.n_draws, args.n_boot, args.n_reps, inp["axis"],
                   cl_doc["axes"][inp["axis"]], inp["verification"], released, None,
                   "this record's AUROC block (frozen eval split)", core, pre_check=pre_check)
    same_fits = comp["pre_check"]
    d168 = d168_listed(inp["verification"])
    changed = [w for w, a, b in (("code", P.code_block(CODE_FILES)["sha256"], code0["sha256"]),
                                 ("registry sources", D.sources_now(), inp["reg_sources"]),
                                 ("cluster manifest", sha256_file(CLUSTERS), cl_sha)) if a != b]
    changed += [r_ for r_, h in _RESULT_READS.items() if sha256_file(ROOT / r_) != h] + B.unchanged() + caches_unchanged(inp)
    if changed:
        refuse(f"changed during the run: {changed}")
    buf = io.BytesIO()
    np.savez_compressed(buf, **comp["arrays"])
    npz_bytes = buf.getvalue()
    config = {**common_config(inp, core, L, args, cl_sha), "probe": fam.describe(), "selection_file": srel,
              "selection_sha256": ssha, "selection_of_pair": mlp_sel_core, "linear_selection_file": rel(SELECTION),
              "linear_selection_sha256": sel_sha, "linear_selection_of_pair": sel_stem, "linear_C_of_record": C_sel,
              "tables_ii_and_4c": released, "control_row": core == CONTROL_ROW}
    fits = comp["fits"]["pre"] + comp["fits"]["post_refit"]
    payload = {**provenance_block("mlp_probe_d383"), "preregistration": PREREG, "choices": CHOICES, "code": code0,
               "reused_files_sha256": reused_now(), "registered_parameters": registered, "final_name": final,
               "blas_threads": threads, "elapsed_s": round(time.time() - t0, 1), "anchor_record": {"file": arel, "sha256": ash},
               "input_verification": inp["verification"], "config": config, "config_hash": canonical_hash(config),
               "population": comp["population"], "pool_4dB_design": comp["pool_4dB_design"],
               "probe_fits": {**comp["fits"],
                              "pre_fits_equal_selection_fits_per_seed": same_fits["pre_fits_equal_selection_fits_per_seed"],
                              "n_pre_fits_equal_selection_fits": same_fits["n_equal"],
                              "pre_fits_equal_required": same_fits["required"],
                              "note": "the pre fits use the selection's rows, order, seed and config; equal weights show "
                                      "the evaluated probe is the selected fit (required at a final name, checked before "
                                      "the post refits and the draws; recorded only at a test name; CHOICES (18))",
                              "fit_cost": {"n_fits": len(fits), "cpu_s_total": round(sum(f["cpu_s"] for f in fits), 1),
                                           "wall_s_total": round(sum(f["wall_s"] for f in fits), 1),
                                           "n_not_converged": sum(not f["converged"] for f in fits)}},
               "auroc_at_lstar": comp["auroc"], "auc_rule_note": AUC_NOTE if released else None,
               "targets": comp["targets"],
               "draws_npz": {"file": rel(npz_path), "sha256": sha256_bytes(npz_bytes), "size": len(npz_bytes), "keys": NPZ_KEYS},
               "records_of_record": {drel: dsha},
               "input_files_sha256": {**dict(_RESULT_READS), **B.listed(), rel(CLUSTERS): cl_sha, **d168},
               **cache_block(inp)}
    json_bytes = json.dumps(payload, indent=2, sort_keys=False, default=str).encode("utf-8")
    problems = input_identity_problems(payload)
    if problems:
        refuse(f"to write (D-297 (1) results-writer rule): {problems}")
    for p, h in write_exclusive([(npz_path, npz_bytes), (out_path, json_bytes)]):
        print(f"Written: {rel(p)}\nsha256:  {h}")
    op = comp["targets"][PRIMARY]["depth_cell"]["operating_point"]
    print(f"{core}: paired ratio {op['paired_ratio']['mean']:.3f}, FPR/target {op['ratio_to_target']['mean']:.3f}, "
          f"delta AUROC {-comp['auroc']['drop_frozen']['mean_over_seeds']:+.4f} (1%, item level, L*={L})")
    return 0


# ------------------------------------------------------------------------------------------------ the summary
def accuracy_line(pairs: list[tuple[float | None, float | None]]) -> dict[str, Any]:
    """D-177's accuracy line over the rows, (realised, predicted): the share within x1.5 and the median
    |log(realised / predicted)|, n rows. A zero or missing side is a miss (D-322 (4)): an infinite error, outside x1.5,
    inside the median (None with median_is_a_miss when the median falls on a miss)."""
    e = np.array([abs(np.log(r / p)) if r is not None and p is not None and r > 0 and p > 0 else np.inf for r, p in pairs])
    med = float(np.median(e)) if e.size else None
    return {"n_rows": int(e.size), "n_rows_a_miss_by_a_zero_or_missing_side": int((~np.isfinite(e)).sum()),
            "share_within_factor_1_5": float((e <= np.log(FACTOR)).mean()) if e.size else None,
            "median_abs_log_realised_over_predicted": med if med is not None and np.isfinite(med) else None,
            "median_is_a_miss": bool(med is not None and not np.isfinite(med)),
            "rule": "D-177: realised = the MLP's value, predicted as named; a zero or missing side is a miss (D-322 (4))"}


def mb(b: dict[str, Any] | None) -> dict[str, Any] | None:
    return None if b is None else {k: b.get(k) for k in ("mean", "band95", "frozen")}


def table_i_row(depth_cell: dict[str, Any], tpr: dict[str, Any] | None, auroc: dict[str, Any]) -> dict[str, Any]:
    """Table I's quantities of one row: per level the paired ratio, FPR/target, TPR pre/post, delta TPR (with D-282's
    combined interval at the item level), the floor; post AUROC and its change (target-free)."""
    out: dict[str, Any] = {}
    for lvl in LEVELS:
        op = depth_cell["operating_point"] if lvl == LEVELS[0] else depth_cell[LEVELS[1]]["operating_point"]
        r = {q: mb(op[q]) for q in ("paired_ratio", "ratio_to_target", "tpr_pre", "tpr_post", "delta_tpr", "floor_ratio_to_target")}
        if lvl == LEVELS[0] and tpr is not None:
            r["delta_tpr_D282_combined_ci95"] = tpr["delta_tpr"]["combined"]["ci95"]
        out[lvl] = r
    out["auroc_pre"] = auroc["pre"]["mean_over_seeds"]
    out["auroc_post_frozen"] = {"mean_over_seeds": auroc["frozen"]["mean_over_seeds"], "ci95": auroc["frozen"]["ci95"]}
    out["auroc_drop_frozen_pre_minus_post"] = {"mean_over_seeds": auroc["drop_frozen"]["mean_over_seeds"],
                                               "ci95": auroc["drop_frozen"]["ci95"]}
    out["delta_auroc_post_minus_pre"] = -auroc["drop_frozen"]["mean_over_seeds"]
    return out


def table_ii_row(rep: dict[str, Any] | None, pool: dict[str, Any] | None, arms: tuple[str, ...]) -> dict[str, Any]:
    """Table II's columns of one row: per level FPR/target (D-156 mean, band, frozen) and TPR (D-156 mean) of each repair;
    the 4d B columns at the item level only (4d B has no cluster level); the floor from the 4d A blocks (refused unless
    every arm block holds the same floor, as tables_d156.floor_of)."""
    out: dict[str, Any] = {}
    for lvl in LEVELS:
        r: dict[str, Any] = {}
        blk = (rep or {}).get(lvl) or {}
        for name, src, key in TABLE_II:
            if src == "4dA":
                b = blk.get(key) if key in arms else None
                r[name] = None if b is None else {"ratio_to_target": mb(b["ratio_to_target"]),
                                                  "tpr": (b.get("tpr_post") or {}).get("mean")}
            else:
                b = (pool or {}).get(key) if lvl == LEVELS[0] else None
                r[name] = None if b is None else {"ratio_to_target": mb(b["ratio_to_target"]), "tpr": (b.get("tpr") or {}).get("mean")}
        if "quantile_thr" in arms and blk.get("quantile_thr") is not None:
            r["beside: quantile_thr (cal_d negatives plus H)"] = {"ratio_to_target": mb(blk["quantile_thr"]["ratio_to_target"]),
                                                                  "tpr": blk["quantile_thr"]["tpr_post"]["mean"]}
        floors = [(blk[a]["floor_ratio_to_target"], blk[a]["tpr_pre"]) for a in arms if a in blk]
        if floors and any(canon(f) != canon(floors[0]) for f in floors):
            refuse("the 4d A arm blocks hold different floors")
        r["floor (pre)"] = None if not floors else {"ratio_to_target": mb(floors[0][0]), "tpr": floors[0][1]["mean"]}
        out[lvl] = r
    return out


def lin_hc_lstar(hc_doc: dict[str, Any], hz: dict[str, np.ndarray], L: int) -> dict[str, np.ndarray]:
    """The linear health check's L* cell in healthcheck_summary_d156.scoring's form: its per-draw arrays from the record's
    npz and the confusion counts of that one cell (healthcheck_d156.main's lines, lstar_counts)."""
    c = cell_at(hc_doc, L)
    if c is None:
        refuse(f"the health-check record has no cell at L{L}")
    z: dict[str, np.ndarray] = {}
    for lvl in LEVELS:
        v = {q: hz[f"L{L}|{PSET}|{lvl}|{q}"] for q in HC.STATS}
        z.update({f"L{L}|{PSET}|{lvl}|{q}": x for q, x in v.items()})
        z.update({f"counts|{PSET}|{lvl}|{k}": x for k, x in lstar_counts(v, bool(c["auc_rule"]["flag"])).items()})
    return z


def score_predictions(rows: dict[str, dict[str, Any]], sens: dict[str, float | None]) -> dict[str, Any]:
    """D-383 (4), on the nine released pairs (rows[core]['mlp'|'linear'] with paired_ratio, r_same and delta_auroc: the
    D-156 means at L*, 1%, item level, and the AUROC change), and the pooled sensitivities sens['mlp'|'linear']."""
    cores = list(rows)
    if sorted(cores) != sorted(RELEASED):
        refuse(f"the predictions are scored on the nine released pairs, not {cores}")
    v = lambda who, q: [rows[c][who][q] for c in cores]  # noqa: E731
    n1 = sum(x >= 2.0 for x in v("mlp", "paired_ratio"))
    n1_lin = sum(x >= 2.0 for x in v("linear", "paired_ratio"))
    in2 = [0.5 < x < 2.0 for x in v("mlp", "r_same")]
    n3 = sum(abs(x) < 0.05 for x in v("mlp", "delta_auroc"))
    n3_lin = sum(abs(x) < 0.05 for x in v("linear", "delta_auroc"))
    s, s_lin = sens.get("mlp"), sens.get("linear")
    undefined = lambda x: x is None or not np.isfinite(x)  # noqa: E731   a None or NaN sensitivity is undefined
    hv = lambda ok: "HELD" if ok else "FAILED"  # noqa: E731
    per = lambda q: [{"pair": c, "mlp": rows[c]["mlp"][q], "linear_of_record": rows[c]["linear"][q],  # noqa: E731
                      "d170_flag": rows[c].get("flag")} for c in cores]
    return {
        "P-MLP.1": {"wording": "the frozen paired ratio is >= 2 in at least 6 of the 9 released pairs (the linear probe: "
                               "8 of 9, Table I)", "n_pairs_at_or_above_2": int(n1), "verdict": hv(n1 >= 6),
                    "linear_n_pairs_at_or_above_2_from_the_records": int(n1_lin),
                    "linear_count_stated_in_D383": 8, "linear_count_agrees_with_D383": int(n1_lin) == 8,
                    "rows": per("paired_ratio"),
                    "accuracy_line_vs_linear": accuracy_line([(rows[c]["mlp"]["paired_ratio"], rows[c]["linear"]["paired_ratio"])
                                                              for c in cores])},
        "P-MLP.2": {"wording": "the R_same threshold repair's FPR/target D-156 mean lies in (0.5, 2) in all 9",
                    "n_pairs_in_interval": int(sum(in2)), "pairs_outside": [c for c, ok in zip(cores, in2) if not ok],
                    "verdict": hv(all(in2)),
                    "linear_n_pairs_in_interval": int(sum(0.5 < x < 2.0 for x in v("linear", "r_same"))),
                    "rows": per("r_same"),
                    "accuracy_line_vs_linear": accuracy_line([(rows[c]["mlp"]["r_same"], rows[c]["linear"]["r_same"]) for c in cores]),
                    "accuracy_line_vs_target_1": accuracy_line([(rows[c]["mlp"]["r_same"], 1.0) for c in cores])},
        "P-MLP.3": {"wording": "|delta AUROC| < 0.05 in at least as many released pairs as the linear probe's",
                    "n_pairs_mlp": int(n3), "n_pairs_linear": int(n3_lin), "verdict": hv(n3 >= n3_lin),
                    "rows": [{**r, "mlp_minus_linear": r["mlp"] - r["linear_of_record"]} for r in per("delta_auroc")],
                    "note": "delta AUROC = post (frozen) minus pre at L*, mean over seeds; additive, so the MLP-minus-linear "
                            "difference stands in for the accuracy line (CHOICES (13))"},
        "P-MLP.4": {"wording": "the Gaussian health ratio's pooled sensitivity for the primary class over the nine L* cells "
                               "is >= 0.8", "pooled_sensitivity": s, "verdict": hv(not undefined(s) and s >= 0.8),
                    **({"reason": f"the pooled sensitivity is undefined ({s!r}: no cell-draw in the class), so '>= 0.8' "
                                  "does not hold (CHOICES (12))"} if undefined(s) else {}),
                    "linear_pooled_sensitivity_lstar_cells": s_lin,
                    "mlp_minus_linear": None if undefined(s) or undefined(s_lin) else s - s_lin,
                    "note": "P-MLP.4: pooled, no per-pair table; MLP minus linear beside (no D-177 line; CHOICES (13))"},
    }


def summary_job(args: argparse.Namespace) -> int:
    t0 = time.time()
    out_path, _, final = out_paths("mlp_probe_d383", args.out, npz=False)
    threads = blas_threads()
    if final:                                  # all but the records' own registration, which is read below
        problems = final_problems(True, threads)
        if problems:
            refuse(f"the final name: {problems}")
    code0 = P.code_block(CODE_FILES)
    man_path = LOGS / RUN_MANIFEST_NAME
    man_runs, man_sha = T3.manifest(man_path)

    def bound(rel_path: str) -> tuple[dict[str, Any], str]:
        if not (ROOT / rel_path).is_file():
            refuse(f"{rel_path} is not on disk")
        d = read_result(ROOT / rel_path)
        h = _RESULT_READS[_repo_rel(ROOT / rel_path)]
        if man_runs.get(rel_path) != h:
            refuse(f"{rel_path} is not the file {rel(man_path)} names")
        return d, h

    arel = f"results/mlp_probe_d383_anchor_{ANCHOR_PAIR}.json"
    adoc, _ = bound(arel)
    sels = {c: bound(f"results/mlp_probe_d383_selection_{c}.json")[0] for c in RELEASED}
    docs = {c: bound(f"results/mlp_probe_d383_{c}.json")[0] for c in ROWS}
    zs: dict[str, dict[str, np.ndarray]] = {}
    npz_inputs: dict[str, str] = {}
    for c, d in docs.items():
        nrel = d["draws_npz"]["file"]
        b = (ROOT / nrel).read_bytes()
        h = sha256_bytes(b)
        if man_runs.get(nrel) != h or d["draws_npz"]["sha256"] != h:
            refuse(f"{nrel} is not the npz {rel(man_path)} and its record name")
        npz_inputs[nrel] = h
        with np.load(io.BytesIO(b), allow_pickle=False) as z:
            zs[c] = {k: z[k] for k in z.files}
    every = [adoc, *sels.values(), *docs.values()]
    named = [("anchor", adoc), *[(f"selection {c}", s) for c, s in sels.items()], *docs.items()]
    if final:                                  # CHOICES (18): a final name binds only records written at theirs
        not_final = [n for n, d in named if d.get("final_name") is not True]
        if not_final:
            refuse(f"the final name: records not written at their final names: {not_final}")
    registered = all(d.get("registered_parameters") is True for d in every)
    if final and not registered:
        refuse("the final name: a record read here was not made with the registered parameters")
    if adoc["anchor"].get("identical") is not True or adoc["config"]["pair"] != ANCHOR_PAIR or anchor_kind_problem(adoc["anchor"]):
        refuse(f"{arel} is not a passed anchor of {ANCHOR_PAIR} {anchor_kind_problem(adoc['anchor'])}")
    other = [f"{n} ({other_revision(d, code0)})" for n, d in named if other_revision(d, code0)]
    if other:
        refuse(f"records made by another revision of the code (CHOICES (19)): {other}")
    same_keys = ("seeds", "targets", "n_draws", "n_boot", "n_reps_4dB", "cluster_manifest")
    ref = docs[RELEASED[0]]["config"]
    diff = [(c, k) for c, d in docs.items() for k in same_keys if canon(d["config"][k]) != canon(ref[k])]
    diff += [(c, "anchor_record") for c, d in docs.items() if d["anchor_record"]["sha256"] != _RESULT_READS[arel]]
    diff += [(f"selection {c}", "anchor_record") for c, s in sels.items()
             if (s.get("anchor_record") or {}).get("sha256") != _RESULT_READS[arel]]
    diff += [(c, "selection") for c, d in docs.items()
             if d["config"]["selection_sha256"] != _RESULT_READS[f"results/mlp_probe_d383_selection_{d['config']['selection_of_pair']}.json"]]
    if diff:
        refuse(f"the anchor, nine selections and ten per-pair records are not one design: {diff[:6]}")
    # CHOICES (18): the pre fits against the selection's fits, and non-convergence, per row
    fit_rows = {c: {"pre_fits_equal_selection_fits_per_seed": d["probe_fits"]["pre_fits_equal_selection_fits_per_seed"],
                    "n_pre_fits_equal_selection_fits": int(sum(bool(x) for x in d["probe_fits"]["pre_fits_equal_selection_fits_per_seed"])),
                    "n_not_converged_pair_job": d["probe_fits"]["fit_cost"]["n_not_converged"],
                    "n_fits_pair_job": d["probe_fits"]["fit_cost"]["n_fits"],
                    **({"n_not_converged_selection": sels[c]["fit_cost"]["n_not_converged"],
                        "n_fits_selection": sels[c]["fit_cost"]["n_fits"]} if c in sels else {})}
                for c, d in docs.items()}
    if final and any(r["n_pre_fits_equal_selection_fits"] != len(REGISTERED["seeds"]) for r in fit_rows.values()):
        refuse(f"the final name: a per-pair record's pre fits are not its selection's fits: "
               f"{ {c: r['n_pre_fits_equal_selection_fits'] for c, r in fit_rows.items()} }")

    B = Bound()
    t_i: dict[str, Any] = {tk: [] for tk in TARGETS}
    t_ii: dict[str, Any] = {tk: [] for tk in TARGETS}
    pred_rows: dict[str, dict[str, Any]] = {}
    prs_mlp: dict[str, dict[str, Any]] = {tk: {} for tk in TARGETS}
    prs_lin: dict[str, dict[str, Any]] = {tk: {} for tk in TARGETS}
    for c in ROWS:
        d, z = docs[c], zs[c]
        L = d["config"]["layer"]
        lin_d = {tk: B.doc(depth_rel(c, tk), "depth" if tk == PRIMARY else "depth_fpr5")[0] for tk in TARGETS}
        lin_t1 = {tk: B.doc(table1_rel(tk), "table1" if tk == PRIMARY else "table1_fpr5")[0] for tk in TARGETS}
        if lin_d[PRIMARY]["config"]["selected_layer"] != L:
            refuse(f"{c}: the MLP record's L{L} is not the linear record's L*")
        flag = d["input_verification"]["flag"]
        for tk in TARGETS:
            lc = cell_at(lin_d[tk], L)
            ltpr = (((lin_t1[tk]["rows"].get(f"depth_{c}") or {}).get("probe_sets") or {}).get(PSET) or {}).get("layers", {}).get(str(L))
            t_i[tk].append({"pair": c, "layer": L, "control_row": c == CONTROL_ROW, "d170_flag": flag,
                            "mlp": table_i_row(d["targets"][tk]["depth_cell"], d["targets"][tk]["tpr_D282"], d["auroc_at_lstar"]),
                            "linear_of_record": table_i_row(lc, ltpr, lc["auroc"]),
                            "mlp_selection": {"width": d["config"]["probe"]["width"], "alpha": d["config"]["probe"]["alpha"]}})
        if c not in RELEASED:
            continue
        lin_rep, _ = B.doc(f"results/repair_d156_{c}.json", "repair")
        prel, pkey = pool_rel(c)
        lin_pool, _ = B.doc(prel, pkey)
        mrel = moment5_rel(c, B)                       # CHOICES (20): every Aegis pair's; None for WildGuardMix
        mom5 = B.doc(mrel, "moment_fpr5")[0] if mrel else None
        for tk in TARGETS:
            mine = d["targets"][tk]
            if tk == PRIMARY:
                lin_ii = table_ii_row(cell_at(lin_rep, L), lin_pool["configurations"][PSET], REPAIR_ARMS)
            else:
                lin_ii = table_ii_row(cell_at(mom5, L) if mom5 else None, None, MOMENT_ARMS_5)
            t_ii[tk].append({"pair": c, "layer": L, "d170_flag": flag,
                             "mlp": table_ii_row(mine["repair_4dA"], mine["pool_4dB"], REPAIR_ARMS),
                             "linear_of_record": lin_ii,
                             **({"linear_note": "at 5% the linear rows of record are the moment family's frozen, restandardise "
                                                "and refit_oracle (D-248); no 5% record of the quantile arms or of 4d B"}
                                if tk != PRIMARY else {})})
            hd, _ = B.doc(hc_rel(c, tk), "healthcheck" if tk == PRIMARY else "healthcheck_fpr5")
            hz = B.npz(hd["draws_npz"]["file"], "healthcheck" if tk == PRIMARY else "healthcheck_fpr5", hd["draws_npz"]["sha256"])
            prs_lin[tk][c] = {"z": lin_hc_lstar(hd, hz, L)}
            prs_mlp[tk][c] = {"z": {k[len(tk) + 1:]: v for k, v in z.items()
                                    if k.startswith(f"{tk}|L") or k.startswith(f"{tk}|counts|")}}
        lc = cell_at(lin_d[PRIMARY], L)
        m_op = d["targets"][PRIMARY]["depth_cell"]["operating_point"]
        pred_rows[c] = {"flag": flag,
                        "mlp": {"paired_ratio": m_op["paired_ratio"]["mean"],
                                "r_same": d["targets"][PRIMARY]["repair_4dA"][LEVELS[0]]["quantile_thr_benign"]["ratio_to_target"]["mean"],
                                "delta_auroc": -d["auroc_at_lstar"]["drop_frozen"]["mean_over_seeds"]},
                        "linear": {"paired_ratio": lc["operating_point"]["paired_ratio"]["mean"],
                                   "r_same": cell_at(lin_rep, L)[LEVELS[0]]["quantile_thr_benign"]["ratio_to_target"]["mean"],
                                   "delta_auroc": -lc["auroc"]["drop_frozen"]["mean_over_seeds"]}}
    # 4c over the nine L* cells (D-255's estimator unchanged), MLP and linear; Table IV's all-cell rows beside
    t_4c: dict[str, Any] = {}
    for tk in TARGETS:
        hs_doc, _ = B.doc(f"results/healthcheck_d156_summary{sfx(tk)}.json", "hc_summary" if tk == PRIMARY else "hc_summary_fpr5")
        t_4c[tk] = {f"{PSET}|{lvl}": {"mlp_lstar_cells": HS.scoring(prs_mlp[tk], RELEASED, PSET, lvl),
                                      "linear_lstar_cells": HS.scoring(prs_lin[tk], RELEASED, PSET, lvl),
                                      "linear_table_iv_every_cell": (hs_doc["rows"].get(f"{PSET}|{lvl}") or {}).get("released_pairs")}
                    for lvl in LEVELS}
    prim = t_4c[PRIMARY][f"{PSET}|{LEVELS[0]}"]
    key = f"{PRIMARY_DETECTOR}|{PRIMARY_CLASS}"
    sens = {"mlp": prim["mlp_lstar_cells"][key]["pooled_from_mean_counts"]["sensitivity"],
            "linear": prim["linear_lstar_cells"][key]["pooled_from_mean_counts"]["sensitivity"]}
    preds = score_predictions(pred_rows, sens)
    # accuracy lines of Table I's and Table II's ratio columns, MLP against the linear rows (D-177), primary cells
    acc = {"table_i_paired_ratio": accuracy_line([(r["mlp"][LEVELS[0]]["paired_ratio"]["mean"],
                                                   r["linear_of_record"][LEVELS[0]]["paired_ratio"]["mean"])
                                                  for r in t_i[PRIMARY] if not r["control_row"]]),
           "table_i_ratio_to_target": accuracy_line([(r["mlp"][LEVELS[0]]["ratio_to_target"]["mean"],
                                                      r["linear_of_record"][LEVELS[0]]["ratio_to_target"]["mean"])
                                                     for r in t_i[PRIMARY] if not r["control_row"]])}
    for name, _, _ in TABLE_II:
        acc[f"table_ii {name}"] = accuracy_line([(((r["mlp"][LEVELS[0]][name] or {}).get("ratio_to_target") or {}).get("mean"),
                                                  ((r["linear_of_record"][LEVELS[0]][name] or {}).get("ratio_to_target") or {}).get("mean"))
                                                 for r in t_ii[PRIMARY]])
    changed = [w for w, a, b in (("code", P.code_block(CODE_FILES)["sha256"], code0["sha256"]),
                                 (rel(man_path), sha256_file(man_path), man_sha)) if a != b]
    changed += [r_ for r_, h in _RESULT_READS.items() if sha256_file(ROOT / r_) != h] + B.unchanged()
    changed += [r_ for r_, h in npz_inputs.items() if sha256_file(ROOT / r_) != h]
    if changed:
        refuse(f"changed during the run: {changed}")
    config = {k: ref[k] for k in same_keys}
    payload = {**provenance_block("mlp_probe_d383_summary"), "preregistration": PREREG, "choices": CHOICES, "code": code0,
               "reused_files_sha256": reused_now(), "registered_parameters": registered, "final_name": final,
               "blas_threads": threads, "elapsed_s": round(time.time() - t0, 1), "config": config,
               "config_hash": canonical_hash(config),
               "anchor": {"file": arel, **{k: adoc["anchor"][k] for k in ("n_comparisons", "by_kind", "layer")}},
               "probe_fits": {"note": "per row: the pre fits against the selection's fits (weights_sha256 per seed; "
                                      "required at a final name, CHOICES (18)) and the fits that did not converge "
                                      "(recorded, not refused, CHOICES (2))",
                              "n_rows_every_pre_fit_equal": int(sum(r["n_pre_fits_equal_selection_fits"] == len(REGISTERED["seeds"])
                                                                    for r in fit_rows.values())),
                              "n_rows": len(fit_rows), "rows": fit_rows},
               "mlp_selection": {c: {**s["selected"], "layer": s["config"]["layer"],
                                     "linear_probe_of_record_at_lstar": s["linear_probe_of_record_at_lstar"]}
                                 for c, s in sels.items()},
               "predictions": preds,
               "verdicts": {k: v["verdict"] for k, v in preds.items()},
               "accuracy_lines_primary_cells": acc,
               "table_i": {"note": "per row at L*, selected MLP beside the linear probe's row of record (depth_d156, "
                                   "table1_tpr_d282); the control row beside (not in P-MLP)", "rows": t_i},
               "table_ii": {"note": "per released pair at L*: 4d A arms both levels, 4d B columns item level (its own draws), "
                                    "the floor; linear rows of record beside (5%: the moment family's three arms)", "rows": t_ii},
               "health_check_4c": {"note": "healthcheck_summary_d156.scoring over the nine L* cells; Table IV's all-cell "
                                           "linear rows beside", "rows": t_4c,
                                   "primary": {"probe_set": PSET, "level": LEVELS[0], "class": PRIMARY_CLASS,
                                               "detector": PRIMARY_DETECTOR, "target": PRIMARY}},
               "input_files_sha256": {**dict(_RESULT_READS), **B.listed(), **B.npz_read, **npz_inputs, rel(man_path): man_sha}}
    digest = write_json(out_path, payload)
    print(f"Written: {rel(out_path)}\nsha256:  {digest}")
    for k, v in preds.items():
        print(f"  {k}: {v['verdict']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="D-383: C1, C4 and C5 for a one-hidden-layer MLP probe")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--anchor", action="store_true", help="the linear probe through this module, against the records")
    g.add_argument("--select", action="store_true", help="the MLP's D-117-style selection at L* (a released pair)")
    g.add_argument("--summary", action="store_true", help="the summary and P-MLP.1..4 from the records")
    ap.add_argument("--pair", default=None, choices=ROWS)
    ap.add_argument("--n-draws", type=int, default=REGISTERED["n_draws"])
    ap.add_argument("--n-boot", type=int, default=REGISTERED["n_boot"])
    ap.add_argument("--n-reps", type=int, default=REGISTERED["n_reps"], help="4d B's draws (poolstudy_d156.N_REPS)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.summary:
        if args.pair is not None:
            refuse("--summary takes no --pair")
        return summary_job(args)
    if args.anchor:
        return anchor_job(args)
    if args.pair is None:
        refuse("--pair is required")
    return select_job(args) if args.select else pair_job(args)


if __name__ == "__main__":
    raise SystemExit(main())
