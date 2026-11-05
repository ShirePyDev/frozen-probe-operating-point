"""D-361: the Phase 2 exit gate's mechanism test. The Phase 1 two-moment machinery (src/twomoment_cells.py and
src/twomoment_fit.py, not edited, their functions called; src/twomoment_supplement.py, not edited and not imported: its
accuracy and offset lines are repeated here and anchored on its record, CHOICES (l)) on the Phase 2 instruct updates it
was never examined on, under split v2, pre-registered in DECISIONS.md D-361 before any Phase 2 evaluation number was
read. Nothing here is fitted or tuned on Phase 2: the predictor is twomoment_cells' (PREDICTOR_ROLE, D-163
(4)), the probes are the D-117 selections of record, the thresholds are set on the pre side's calibration part.

FOUR MODES and a dry run, run as one CPU queue list (queues/laneA_20261007_p2mech.tsv); every output is written once
(exclusive):

  --anchor   (D-361 (6), first job) the Phase 2 path pointed at ONE Phase 1 pair's inputs, repair_WG_mistral_base2instruct
             (ANCHOR_STEM; why: CHOICES (p)), read through src/input_registry.py with Phase 1's partitions (its 1000 / 2000
             pooled negatives, paired.AXIS_DRAW_SEED, manifests/negative_clusters.json). Every value the Phase 2 path
             computes - each selected-C cell at every layer of the pair's grid, both levels (D-156 item, D-164 cluster),
             and every per-draw array row of those cells - must EQUAL results/twomoment_<stem>.json and its _draws.npz at
             1% (the record of record, named by results/twomoment_fit_r2.json, D-305's re-issue of twomoment_fit.json,
             bound through the pinned logs/reissue_d305_20261001.sha256) and results/twomoment_<stem>_fpr5.json and its
             npz at 5% (named by the pinned twomoment_fit_fpr5.json), as canonical JSON text and bit for bit; otherwise
             "ANCHOR FAILED" and nothing is written. Writes results/twomoment_p2_d361_anchor_<stem>.json.
  --pair K --pooling P   one Phase 2 update pair and pooling (PAIRS: the nine headline pairs and the tokenizer-change
             pair; both poolings). REFUSES unless the anchor record exists, passed, and was written by this exact code
             (every CODE_FILES sha256), libraries, BLAS threads and host class (selection_p2.HOST_KEYS); the record names it
             by sha256. Writes results/twomoment_p2_d361_<K>_<P>.json and its _draws.npz (D-163 (6)).
  --lora --pooling P   the LoRA line (D-361 (2), (4); CHOICES (v)-(x)): the 27 trained adapters of D-343 on
             Mistral-7B-Instruct-v0.2, each base -> adapter through the same steps as a pair, on the layers the storage
             decision keeps (option C: L*), the base's caches, probes and v2 matrices read once. Same anchor requirement.
             Writes results/twomoment_p2_d361_lora_<P>.json and its _draws.npz.
  --records-only (with --anchor, --pair or --lora) a dry run of the records: every check up to the first cache read (the
             anchor: up to the input registry's caches; a pair or LoRA job: steps 1-2, without the anchor record, which
             does not exist before the list runs), then "RECORDS OK" and exit 0; no cache is read, nothing is written,
             and nothing but OK or the refusal is printed.
  --summary  reads only the 20 per-pair and 2 LoRA records (and their npz) and the anchor record they name, each bound to
             its line of logs/twomoment_p2_d361.sha256 (the list's manifest job, exactly the 45 files), the Phase 1 4b
             records for its own anchor, and the D-354 summary; scores P2-M.1 to P2-M.3 and every BESIDE line of D-361
             (4). Before scoring, its scoring functions on the nine Phase 1 records must reproduce
             results/twomoment_fit_r2.json's headline (point, intervals, counts) and results/twomoment_supplement_r4.json's
             accuracy lines and per-pair offsets exactly (D-305's re-issues, bound through logs/reissue_d305_20261001
             .sha256), at both levels, or nothing is written. Writes results/twomoment_p2_d361_summary.json.

THE PER-PAIR JOB, in this order (steps 1-2 read records only, so a missing or mismatched record refuses before any
cache is read):
  1. RECORDS  the anchor record (above); results/phase2/bos_check_p2.json, both sides' models 'ok' at their revisions
     (CHOICES (y)); the pre side's selection of record by selection_tpr_v2.read_selection
     (unchanged: this model, revision, pooling, axis; its select record as it is now; this run's BLAS threads and
     libraries; selection_p2.py, regularisation_sweep.py and paired.py at the sha256 that wrote it): its arm (a logistic
     C), L*, seeds, holdout and target; the v2 record and its calibration file (selection_tpr_v2.read_v2) and evaluation
     file (float16_check_d364.read_v2_eval), at their D-362 sha256 for a final name; both models' durable rest subsets
     (selection_tpr_v2.read_subset) and their layer plans (the pre side's naming this selection record and L* in every
     'pre' entry and once for this pair; the post side's naming this pair as 'post' with the same, unless FULL_GRID);
     the post side's durable select set (CHOICES (g)); every cache keyed p2wg/<v1 split> indexing the v1 file the v2
     record names (float16_check_d364.check_v1_keyed, D-320 (4)); the cluster manifest bound through the v2 record ->
     the v1 record -> its input sha256 (CHOICES (e)).
  2. LAYERS   every layer both sides hold (D-361 (2), CHOICES (f)): the layers every cache of both sides holds; refused
     unless GRID6(NL) plus the pre side's L* (subset_p2.grid6; D-325 (2)) are among them; held layers beyond that set
     are cells too, listed in the record (config.layers_beyond_grid6_plus_pre_lstar). The v2 sizes must be D-361 (2)'s
     (5,000 / 6,000 / 4,000) for a final name.
  3. CACHES   from data/hidden only (D-307 (1)): a regular file, mode 444, in its record's ingest index
     (manifests/ingest_p2/<record stem>.sha256) with the record's sha256, hashing to it, exactly the record's layers,
     no flagged item, item for item its frozen split; the pre side's probe_train and calibration must be the caches the
     selection read; pre and post hold the same items with the same labels.
  4. PROBES   per seed paired.fit_probe on the D-117 80% split in permutation order at the selected C (twomoment_cells'
     fit, D-152 P1); at L* every seed's (mu, sd, w, b) must equal selection_tpr_v2.seed_probes' bit for bit and the ho20
     TPRs must equal the selection record's (selection_tpr_v2.anchor_diffs), else "ANCHOR FAILED".
  5. v2 GATHER per layer, v2 calibration and v2 evaluation (negatives and positives) by item_id from the cache of the
     split v1 put each item in (float16_check_d364.gather_split, D-320 (4)); the counts must be the v2 record's plan.
  6. CELLS    per layer and target (1% primary, 5% beside, D-248), the D-156 estimator as D-361 (2): v2 calibration and
     v2 evaluation negatives pooled (5,000 + 6,000), pre and post scores of the same items, 1000 re-partitions at those
     sizes (paired.repartition_draws, seed P2_DRAW_SEED) with the frozen v2 split as one more draw; the 4,000 v2
     evaluation positives fixed; five seeds; per-seed thresholds at the target on the calibration part of the PRE scores;
     FPR floored at 1/|evaluation part|; the D-164 cluster level beside (P2_CLUSTER_DRAW_SEED). Realised and predicted
     paired ratios by twomoment_cells.one_partition and paired.repartition_estimate, exactly as twomoment_cells forms
     its cells (compute_cell repeats its per-layer lines for the selected C; the anchor proves them equal).
  7. FLOAT16 CROSS-CHECK  for Mistral-7B-Instruct v0.1 -> v0.2, user-span mean, at L*: the probes' fingerprints, the six
     score matrices' sha256 and the realised D-156 reports at 1% and 5% must equal results/phase2/float16_check_d364
     .json's arm A (CHOICES (o)), else the job refuses.
  8. RECORD   cells (1%) and cells_fpr5 (5%) in twomoment_cells' cell form; per-draw arrays in its npz form (column 0
     the frozen split; keys ending '__cluster' the D-164 level; keys ending '__fpr5' the 5% twin); provenance (every
     result read with its sha256, every cache with its sha256 and source, the split files, config and config_hash).

THE SUMMARY (D-361 (3), (4)). Headline cells: the nine pairs x their layer set, user-span mean, selected C, item level,
1%. Per cell, predicted and realised paired ratios as the D-156 estimate (twomoment_fit.d156_estimates: log of the mean
over the draws where both are defined), every cell, floored draws kept. D-170: every cell's input_verification flag
must be None (twomoment_fit's rule), carried as 'D-170: 0 of n cells flagged' on the headline, each verdict and each
beside line (CHOICES (z)).
  P2-M.1 (slope)  OLS of log realised on log predicted (twomoment_fit.wols), pair-cluster bootstrap 95% interval (B =
                  5000, seed 520260925, twomoment_fit's estimator): HELD iff the interval exists and includes 1.
  P2-M.2 (accuracy, D-177)  share of cells with |log(realised/predicted)| <= log 1.5 at least 0.80, a zero or undefined
                  prediction a miss (D-322 (4)); the median |log| reported with n cells (misses as +inf).
  P2-M.3 (two moments over one)  moment_full_in_sample's share within x1.5 STRICTLY larger than translation_in_sample's
                  on the same cells.
  BESIDE (never verdicts, D-361 (4)): every other predictor's line; last-token pooling; the deployer's layer (L* cells
  only); 5%; per pair; the D-164 cluster level; the tokenizer-change line (Mistral v0.2 -> v0.3); the LoRA line (the 27
  adapters, from the --lora records); Pearson r with its pair-cluster interval in every fit line; and the D-354 (3)
  listing of the selections whose cell the cluster-grouped ho20 would change.
  Every prediction table carries the accuracy line (D-177 (3)): share within x1.5 and median |log| with n cells.

FINAL NAMES need the registered parameters (1000 draws; seeds 0-4 from the selection records; targets 1% and 5%), the
6-thread BLAS pin (selection_p2.check_threads, scripts/project_env.sh), GUARDRAIL_CACHE_DIRS unset, the pinned records
and manifests (PINNED) and the reused Phase 1 files at the bytes this module was written and tested against
(REUSED_SHA256). A test writes outside results/ (--out / --out-dir).

  python3 -u src/twomoment_p2_d361.py --anchor
  python3 -u src/twomoment_p2_d361.py --pair mistral_v01_v02 --pooling usermean
  python3 -u src/twomoment_p2_d361.py --pair mistral_v01_v02 --pooling usermean --records-only
  python3 -u src/twomoment_p2_d361.py --lora --pooling usermean
  python3 -u src/twomoment_p2_d361.py --summary

CPU only. CHOICES (D-181) below, each naming the D-entry it follows; they are written into every record.
"""

from __future__ import annotations

import argparse
import inspect
import io
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import float16_check_d364 as F16  # noqa: E402  (imported, not changed: its v2 evaluation reader, gather and checks)
import input_registry as REG  # noqa: E402
import paired as P  # noqa: E402
import populations as POP  # noqa: E402  (D305_REISSUES: the names of D-305's re-issues)
import provenance as PV  # noqa: E402
import selection_p2 as SP  # noqa: E402  (imported, not changed: its loaders and checks)
import selection_tpr_v2 as T  # noqa: E402  (imported, not changed: its selection, v2 and subset readers)
import subset_p2 as SUB  # noqa: E402  (imported, not changed: grid6)
import twomoment_cells as TC  # noqa: E402  (imported, not changed: one_partition, moments, PREDICTOR_ROLE)
import twomoment_fit as TF  # noqa: E402  (imported, not changed: wols, d156_estimates, BOOT)
from provenance import (  # noqa: E402
    _RESULT_READS,
    canonical_hash,
    commit_staged,
    ensure_utf8_stdout,
    input_identity_problems,
    provenance_block,
    read_result,
    sha256_bytes,
    sha256_file,
    stage_bytes,
    write_results,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MANIFESTS = ROOT / "manifests"
THIS_FILE = "src/twomoment_p2_d361.py"
CODE_FILES = list(dict.fromkeys([THIS_FILE, *TC.CODE_FILES, *TF.CODE_FILES, *F16.CODE_FILES, *SUB.CODE_FILES]))
# the reviewed Phase 1 and Phase 2 files whose functions this module calls, at the bytes it was written and tested
# against (master 436754f); a final name refuses other bytes, so an edit of a reused file cannot pass unnoticed
REUSED_SHA256 = {
    "src/twomoment_cells.py": "14b28c91e7f2267a752f51a925e2bb875a834a52556e21b0c620ad30c18eeafe",
    "src/twomoment_fit.py": "1a1c57668068c4e86dfb08bc61c141d9c921d92a04da3b620ec7781ae9713b26",
    "src/paired.py": "80429fa57f0267236a2af1670c74ef327802598257ab75834fddf64a70d97195",
    "src/input_registry.py": "e05d1fba11a4db8c46d0e7231fa7b804fbd8101017fda1c0889070c193684fbf",
    "src/selection_tpr_v2.py": "e928a44e969e8fc1ec87a2e9bb93f3c9ddd11f06778dd3066b1d9d0905c3746d",
    "src/selection_p2.py": "a79bb1b8a635c15190bad181e29c1ecd885ef01a71fda26e1101339f3bc59e97",
    "src/float16_check_d364.py": "6027c597844e4c684a530c15af3fd40d06bbd218275ee50ee7f00e4a2fa356b7",
    "src/subset_p2.py": "fc81a30170b0047c4d4606cd5f700001b2f7e2afb45b1e7d130f8269155576c5",
}
RUN_KIND = {"anchor": "twomoment_p2_d361_anchor", "pair": "twomoment_p2_d361", "lora": "twomoment_p2_d361_lora",
            "summary": "twomoment_p2_d361_summary"}
PREREG = ("D-361 (the pre-registration of the Phase 2 exit gate's mechanism test; details by the closest D-entries, "
          "D-181: D-152 4b, D-163, D-174, D-177, D-302, D-303, D-318 (7), D-320); D-156, D-164 (the estimator, both "
          "levels); D-117 (the selections of record); D-248 (5% beside); D-320 (4), D-362 (split v2 and its join rule); "
          "D-325 (the durable layer sets); D-307 (1) (durable caches only)")
AXIS = SP.AXIS                                          # p2wg
POOLINGS = SP.POOLINGS                                  # usermean, lastgen
PRIMARY_POOLING = "usermean"                            # D-361 (2), D-318 (7) item 2
PRIMARY_TARGET, BESIDE_TARGET = 0.01, 0.05              # D-361 (2) 1% target; D-361 (4) 5% beside (D-248)
TARGETS = (PRIMARY_TARGET, BESIDE_TARGET)
TARGET_SFX = {PRIMARY_TARGET: "", BESIDE_TARGET: "__fpr5"}
LEVELS = TC.LEVELS                                      # item_level_D156, cluster_level_D164
LEVEL_SFX = {LEVELS[0]: "", LEVELS[1]: "__cluster"}
PRED_KEYS = TC.PRED_KEYS
PREDICTOR_ROLE = TC.PREDICTOR_ROLE                      # D-361 (1): the Phase 1 predictor, unchanged (D-163 (4))
HEADLINE_PRED = "moment_full_in_sample"                 # PRIMARY
COMPARATOR_PRED = "translation_in_sample"               # P2-M.3's one-moment comparator
SEEDS = [0, 1, 2, 3, 4]
REGISTERED = {"n_draws": P.N_DRAWS, "seeds": SEEDS, "targets": list(TARGETS)}
# D-156: one partition per draw on an axis, the same in every script. paired.AXIS_DRAW_SEED holds no Phase 2 axis; the
# first Phase 2 D-156 record on p2wg (float16_check_d364) drew its item-level partitions with this seed on the same
# pooled negatives in the same order, and this module keeps it (CHOICES (c)). The D-164 cluster level takes its own
# fixed seed, distinct from the item level's, fixed here before any run.
P2_DRAW_SEED = F16.DRAW_SEED                            # 20261006
P2_CLUSTER_DRAW_SEED = 20261007
BOOT = dict(TF.BOOT)                                    # D-361 (3) P2-M.1: B = 5000, seed 520260925 (D-163)
FACTOR = 1.5                                            # D-177: within a factor of 1.5
SHARE_MIN = 0.80                                        # D-361 (3) P2-M.2
ANCHOR_STEM = "repair_WG_mistral_base2instruct"
V1_RECORD_NAME = "dataset_manifest_p2.json"            # the v1 freeze record the v2 record names by sha256
CLUSTER_MANIFEST_REL = "manifests/phase2_pool_clusters.json"
D354_SUMMARY_REL = "results/phase2/ho20cluster_summary.json"
BOS_CHECK_REL = "results/phase2/bos_check_p2.json"     # D-318 (6), D-324 (5): the Phase 2 BOS and cache-tie check
# D-305: twomoment_fit.json and twomoment_supplement_r3.json are SUPERSEDED by their re-issues (the same numbers, repo
# IDs named; populations.D305_REISSUES); the re-issues are read, bound through D-305's manifest as opcase_d384 binds them
REISSUE_MANIFEST = "logs/reissue_d305_20261001.sha256"
FIT_REL = f"results/{POP.D305_REISSUES['twomoment_fit']}.json"                        # results/twomoment_fit_r2.json
SUP_REL = f"results/{POP.D305_REISSUES['twomoment_supplement_r3']}.json"              # .../twomoment_supplement_r4.json
FIT5_REL = "results/twomoment_fit_fpr5.json"           # D-248; no re-issue (not in D-305's list)
RUN_MANIFEST = "logs/twomoment_p2_d361.sha256"         # the list's manifest job; the summary reads it (exactly 45 files)
V2_SIZES = (5000, 6000, 4000)                          # D-361 (2): v2 calibration, evaluation negatives and positives
PINNED = {
    REISSUE_MANIFEST: "10f8e9d40e640584c19a34063f041f3c3895c7d36f3025dd1ba9b0ccdab702c6",           # D-305
    FIT5_REL: "001db8b9553d0ef6f5d60f38ad7cc767f5630b3e915e90288e57d4d55980659c",                   # D-248
    "v2/calibration": "98bc80c966349bc5e03c5cdeab97d035d40447b34c5337d72e2f9062841393e5",          # D-362 (1)
    "v2/eval": "e0fcf8a30de263e502d191c7df87e11d330362602670247bc7de8620cdd17a4c",                 # D-362 (1)
    "results/phase2/float16_check_d364.json": "becb90af1bd171845d61d278e299fbc6f5c3f28a5b9de5bc658971e66f77309d",  # D-372
    D354_SUMMARY_REL: "83617cddd7369c798f32b6b293a95dd2d555879f67cc8d3710bf37783a56cfd1",           # D-354 (1)
}
# D-361 (2): the headline population, nine instruct updates (the four primary pairs and the five post-training series
# steps), and the tokenizer-change pair as its own line (D-318 (7) item 6). key: (pre, post, group, note)
HEADLINE_PAIRS = {
    "mistral_v01_v02": ("mistralai/Mistral-7B-Instruct-v0.1", "mistralai/Mistral-7B-Instruct-v0.2", "primary pair", ""),
    "llama3_to_31": ("meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "primary pair", ""),
    "qwen2_to_25": ("Qwen/Qwen2-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "primary pair", ""),
    "phi3_128k_to_35": ("microsoft/Phi-3-mini-128k-instruct", "microsoft/Phi-3.5-mini-instruct", "primary pair",
                        "the rope_scaling caveat (D-318 (7) item 7)"),
    "olmo2_base_sft": ("allenai/OLMo-2-1124-7B", "allenai/OLMo-2-1124-7B-SFT", "post-training series step", ""),
    "olmo2_sft_dpo": ("allenai/OLMo-2-1124-7B-SFT", "allenai/OLMo-2-1124-7B-DPO", "post-training series step", ""),
    "olmo2_dpo_instruct": ("allenai/OLMo-2-1124-7B-DPO", "allenai/OLMo-2-1124-7B-Instruct", "post-training series step",
                           ""),
    "tulu3_sft_dpo": ("allenai/Llama-3.1-Tulu-3-8B-SFT", "allenai/Llama-3.1-Tulu-3-8B-DPO", "post-training series step",
                      ""),
    "tulu3_dpo_final": ("allenai/Llama-3.1-Tulu-3-8B-DPO", "allenai/Llama-3.1-Tulu-3-8B", "post-training series step",
                        ""),
}
TOKENIZER_PAIRS = {
    "mistral_v02_v03": ("mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mistral-7B-Instruct-v0.3", "tokenizer change",
                        "D-318 (7) item 6: reported as its own line, outside the headline, not a no-update control"),
}
PAIRS = {**HEADLINE_PAIRS, **TOKENIZER_PAIRS}
FLOAT16_CELL = ("mistral_v01_v02", "usermean")         # D-364 (1)'s cell: the cross-check applies here only
# D-361 (2), (4): the LoRA arms (D-343: 3 corpora x 3 doses x 3 seeds = 27 adapters on Mistral-7B-Instruct-v0.2,
# extracted and cut by the storage decision, option C: L* alone, both poolings, float32, D-365, D-386) form their own
# line, outside the headline. The step-0 checkpoints are D-343 (2)'s gate against the base, not arms (CHOICES (v)).
LORA_BASE = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_CORPORA = ("alpaca", "hhharmless", "pkusafe")
LORA_STEPS = (25, 100, 400)
LORA_SEEDS = (0, 1, 2)
LORA_ARMS = [f"lora/mistral7bi-v02-{c}-r16-seed{s}-step{d}" for c in LORA_CORPORA for d in LORA_STEPS
             for s in LORA_SEEDS]
CLUSTER_NOTE_P2 = ("cluster-level re-partitions (D-164), reported beside the item level, never instead. Units: the "
                   "near-duplicate clusters of manifests/phase2_pool_clusters.json restricted to the pooled negatives, "
                   "every other negative its own unit. The 'frozen' value is the frozen v2 split; the record's "
                   "cluster_units block counts the multi-item units it splits across calibration and evaluation (a split "
                   "this distribution cannot produce when that count is not 0)")
D170_P2 = {"flag": None, "pin_only_inputs": [],
           "rule": "D-170/D-172 on Phase 2 inputs: every cache is read from data/hidden at its record's sha256, listed in "
                   "its ingest index (D-307 (1)); no input is vouched for by a pin alone, so no cell carries the flag"}

CHOICES = [
    "(a) Probe set: the selected C only (D-361 (2) names it; D-361 (4)'s beside list has no C=1 row), so no C=1 cell and "
    "no reversed-order refit is computed on Phase 2 (they are twomoment_cells' sensitivity rows, D-163 (3)).",
    "(b) Probe fit: per seed paired.fit_probe on the D-117 80% split in permutation order at the selected C, the fit "
    "twomoment_cells uses (D-152 P1); at L* it must equal the selection's own fit (selection_tpr_v2.seed_probes) bit for "
    "bit in mu, sd, w and b, and reproduce the selection record's ho20 TPRs exactly (D-117; float16_check_d364 step 4).",
    "(c) Partition seeds on p2wg: item level 20261006, the seed of the first Phase 2 D-156 record on this axis "
    "(float16_check_d364.DRAW_SEED, same pooled negatives, same order), so draw r is the same item split everywhere on "
    "the axis (D-156); cluster level 20261007, a distinct fixed per-axis seed (D-164).",
    "(d) Pool order: v2 calibration in its file's order, then the v2 evaluation negatives in theirs (D-156's 'calibration "
    "negatives then eval negatives' read through D-320 (4)'s join rule; as float16_check_d364).",
    "(e) Cluster level: units are the cluster labels of manifests/phase2_pool_clusters.json (the D-164 rule: exact "
    "Jaccard >= 0.5, connected components) restricted to the 11,000 pooled negatives, each label with two or more pooled "
    "members one unit, every other negative its own unit; the manifest is bound through the v2 record's v1 record and "
    "that record's input sha256 (D-164, D-320).",
    "(f) Layers: the cells are every layer both sides hold (D-361 (2): 'GRID6 plus the pre side's L*, D-325; every layer "
    "both sides hold', read operationally, so nothing is dropped, D-361 (5)): the layers that every cache of both sides "
    "holds, by the records. The job refuses unless GRID6(NL) (subset_p2.grid6) plus the pre side's L* for the pooling "
    "(D-325 (2)) are among them; held layers beyond that set (a subset that also carries another pair's L*) are cells "
    "too, listed per record as config.layers_beyond_grid6_plus_pre_lstar.",
    "(g) The post side's v1 calibration cache comes from its durable select set: the p2tpl select record where that is "
    "ingested (its ingest index exists), else its p2sub select record (D-325 (1), (4); D-307 (1)); exactly one must be.",
    "(h) 5% is computed in the same job on the same probes, items and draws (D-248: 'the same draws, seeds, probes and "
    "inputs as at 1%'), stored beside as cells_fpr5 and npz keys ending '__fpr5'; the anchor job proves the 5% path "
    "against the Phase 1 _fpr5 record (D-248's self-anchor purpose), so no separate 5% pipeline exists.",
    "(i) P2-M.2's accuracy line (D-177, D-322 (4)): per cell e = |log realised - log predicted| from the per-cell D-156 "
    "estimates (twomoment_fit.d156_estimates, as twomoment_supplement); a cell whose e is not finite (a zero or undefined "
    "prediction) is a miss: outside x1.5 and +inf in the median; the share is over every cell; Phase 1's convention "
    "(defined cells only) is reported beside, as D-322 (4) asks for both.",
    "(j) P2-M.1 is HELD iff the pair-cluster 95% interval exists and its ends bracket 1 (inclusive); an interval that "
    "cannot be formed is FAILED with its reason. The interval is twomoment_fit's pair bootstrap (its wols, the replicate "
    "multiplicities drawn as it draws them, paired.ci), anchored on twomoment_fit_r2.json's headline (D-305's re-issue "
    "of twomoment_fit.json; D-163, D-174).",
    "(k) P2-M.3 compares the two shares of (i) over the same headline cells; 'strictly larger', so a tie is FAILED.",
    "(l) The summary first anchors its scoring on Phase 1: on the nine Phase 1 records it must reproduce "
    "twomoment_fit_r2.json's headline (point, intervals, counts), twomoment_supplement_r4.json's all-cells accuracy "
    "lines for moment_full_in_sample and translation_in_sample, and its per-pair offset rows (mean offset, cell counts, "
    "D-156 partition band) at both levels exactly (D-361 (6); D-248's self-anchor); the re-issues of D-305 replace the "
    "superseded twomoment_fit.json and twomoment_supplement_r3.json (the same values, repo IDs named), bound through "
    "logs/reissue_d305_20261001.sha256 (pinned). twomoment_supplement is not imported: its accuracy and offset lines "
    "are repeated here (accuracy_line, within_draw_mean, offset_row) and this anchor is their check.",
    "(m) Beside lines (D-361 (4)) carry quantities only, no HELD/FAILED: the other five predictors; last-token; the "
    "deployer's layer; 5%; the D-164 cluster level; per pair (D-177 (2a): the mean offset log(realised/predicted) with its "
    "D-156 partition band, and the accuracy line; no pair-cluster interval for one pair); the tokenizer-change pair (one "
    "pair: accuracy line, mean offset with partition band, the OLS point over its cells without an interval, D-177); the "
    "LoRA line (CHOICES (v)-(x)); Pearson r with its pair-cluster interval in every fit line (D-174).",
    "(n) D-354 (3): the selections of record are used (D-361 (2)); the summary lists, beside, for every selection record "
    "the pairs use, whether D-354's cluster-grouped ho20 changes its cell, with both cells and the margins.",
    "(o) Float16 cross-check: on D-364's cell (Mistral-7B-Instruct v0.1 -> v0.2, user-span mean, L*) the probes' "
    "fingerprints, the six score matrices' sha256 and the realised 1% and 5% D-156 reports must equal "
    "float16_check_d364.json's arm A, or the job refuses: a check of the v2 join path that the Phase 1 anchor cannot reach "
    "(as twomoment_cells checks damage_d156_r2, D-173).",
    "(p) Anchor pair: repair_WG_mistral_base2instruct, the simplest Phase 1 inputs for this path: no input is pin-only "
    "(its D-170 block is unchanged since the record, so the comparison is value for value), it is on WildGuardMix, the "
    "dataset Phase 2's p2wg axis is drawn from, with 11 near-duplicate clusters so the cluster path is exercised, it has "
    "the plain 17-layer, 4096-wide grid and an ungated model (Mistral-7B v0.1 -> Instruct v0.1, whose post side is "
    "Phase 2's first pre side), and both its 1% record and its _fpr5 twin are named by the pinned fits.",
    "(q) The anchor compares every value the Phase 2 path computes (the selected-C cells at every layer, both levels, 1% "
    "and 5%, and their per-draw rows); twomoment_cells' C=1 cells and reversed-order refit are outside that path (a) and "
    "are not recomputed; the npz keys the path does not form must be exactly the reversed-order columns, NaN on every "
    "selected row.",
    "(r) The tokenizer-change pair runs both poolings, like every headline pair (D-361 (4)'s last-token line).",
    "(s) The pair-cluster bootstrap resamples the nine pairs as clusters (D-163); pairs sharing a model (OLMo-2 SFT and "
    "DPO, Tulu-3 DPO are each one pair's post side and the next pair's pre side) are not modelled as dependent, as "
    "twomoment_fit's note says of Phase 1.",
    "(t) Each Phase 2 cell's D-170 block says no input is pin-only (every cache read at its record's sha256 from its "
    "ingest index, D-307 (1)); Phase 1's registry block is kept in the anchor, where it must equal the record's.",
    "(u) Final names need GUARDRAIL_CACHE_DIRS unset (hh_sensitivity_d363's rule: the anchor's Phase 1 caches come from "
    "the input registry's directory) and the reused files at their reviewed bytes (REUSED_SHA256).",
    "(v) The LoRA line's arms are the 27 trained adapters of D-343 (alpaca, hh-harmless, pku-saferlhf x steps 25, 100, "
    "400 x seeds 0, 1, 2) on Mistral-7B-Instruct-v0.2 (the base: the pre side, its D-117 selection of record); the "
    "step-0 checkpoints are D-343 (2)'s gate against the base, not arms, and are not read. Each adapter is one update "
    "(base -> adapter) through the same path as a pair (D-361 (2): 'their own line').",
    "(w) The LoRA line's cells are the layers the storage decision keeps (D-361 (2); option C, D-365: L* alone, both "
    "poolings, float32) that the base also holds; one job per pooling loads the base's caches, fits its probes and "
    "gathers its v2 matrices once, then reads each adapter's durable p2sub caches (D-307 (1)); every adapter's caches "
    "name the base's selection record and L* in their layer plans (role 'post').",
    "(x) In the LoRA line's fit each adapter is its own cluster in the bootstrap (D-163: an update is the cluster), B = "
    "5000 with the same seed; the line is beside, never a verdict (D-361 (4)).",
    "(y) BOS check (D-361 (5), D-318 (6)): step 1 reads results/phase2/bos_check_p2.json and refuses unless both sides' "
    "models are 'ok' there: listed at the side's revision, cache_tie status 'ok', prompts_tie and plain_path ok, not in "
    "ties_failed, cache_tie_skipped or cache_tie_incomplete, and no tokenizer left unloaded; its sha256 is recorded. The "
    "LoRA line requires the base's line; the adapters are not in the scan and use the base's tokenizer (D-343).",
    "(z) Summary inputs and D-170: the summary binds the anchor record and the 22 records with their npz to their lines "
    "of logs/twomoment_p2_d361.sha256 (the list's manifest job; it must list exactly those 45 files) and records the "
    "manifest's sha256; every cell's input_verification flag must be None (twomoment_fit's rule: no cell may be "
    "vouched for by a pin alone), carried as 'D-170: 0 of n cells flagged' on the headline, each verdict and each beside "
    "line.",
]

refuse = SP.refuse


def choice(letter: str) -> str:
    """The CHOICES item labelled (letter)."""
    return next(c for c in CHOICES if c.startswith(f"({letter}) "))


# --------------------------------------------------------------------------------------------- small helpers

def canon(o: Any) -> str:
    """Canonical JSON text of `o` as written by the repo's writers (json.dumps default=str, parsed back, keys sorted):
    a record read from disk and a value computed now compare equal exactly when every value is the same."""
    return json.dumps(json.loads(json.dumps(o, default=str)), sort_keys=True)  # not a result (D-297)


def relpath(p: Path) -> str:
    """Repo-relative name for a path in the repo, else the absolute path (a test tree)."""
    p = Path(p).resolve()
    return str(p.relative_to(ROOT.resolve())).replace("\\", "/") if p.is_relative_to(ROOT.resolve()) else str(p)


def resolve(name: str) -> Path:
    """The path a record's file field names: absolute as given, else under the repo root."""
    p = Path(name)
    return p if p.is_absolute() else ROOT / p


def blas_threads() -> dict[str, str | None]:
    return {v: os.environ.get(v) for v in SP.THREAD_VARS}


def final_problems(registered: bool, pinned_keys: tuple[str, ...] = ()) -> list[str]:
    """Why a final name may not be written (empty when it may): parameters, BLAS pin, GUARDRAIL_CACHE_DIRS, the reused
    files at their reviewed bytes and the named pinned records at their bytes."""
    out = []
    if not registered:
        out.append(f"non-registered parameters (the final names need {REGISTERED})")
    th = blas_threads()
    if any(v != SP.THREADS for v in th.values()):
        out.append(f"no 6-thread BLAS pin (D-152 P6): {th}")
    if os.environ.get("GUARDRAIL_CACHE_DIRS"):
        out.append("GUARDRAIL_CACHE_DIRS is set; the caches must come from their recorded directories")
    for f, want in REUSED_SHA256.items():
        have = sha256_file(ROOT / f) if (ROOT / f).is_file() else None
        if have != want:
            out.append(f"{f} is not the reviewed bytes this module reuses ({(have or 'missing')[:12]} vs {want[:12]})")
    for k in pinned_keys:
        p = ROOT / k
        have = sha256_file(p) if p.is_file() else None
        if have != PINNED[k]:
            out.append(f"{k} is not the pinned bytes ({(have or 'missing')[:12]} vs {PINNED[k][:12]})")
    return out


def read_npz_bytes(path: Path, sha: str | None, size: int | None, owner: str) -> dict[str, np.ndarray]:
    """An npz read once as bytes, hashed and loaded from those bytes; refuses other bytes than `owner` names."""
    if not path.is_file():
        refuse(f"{path.name}, named by {owner}, is missing")
    b = path.read_bytes()
    if sha256_bytes(b) != sha or (size is not None and len(b) != size):
        refuse(f"{path.name} is not the bytes {owner} names ({sha256_bytes(b)[:16]} vs {str(sha)[:16]})")
    with np.load(io.BytesIO(b), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def read_manifest(p: Path, pinned: str | None = None) -> tuple[dict[str, str], str]:
    """({listed path: sha256} of a sha256sum manifest, the sha256 of the bytes parsed), one read; refuses a missing
    file, a malformed or repeated line, or (when `pinned` is given) other bytes."""
    if p.is_symlink() or not p.is_file():
        refuse(f"{relpath(p)} is not on disk (or not a regular file)")
    b = p.read_bytes()
    h = sha256_bytes(b)
    if pinned is not None and h != pinned:
        refuse(f"{relpath(p)} is not the pinned bytes ({h[:16]} vs {pinned[:16]})")
    out: dict[str, str] = {}
    for ln in b.decode("utf-8").splitlines():
        if not ln.strip():
            continue
        parts = ln.split()
        if len(parts) != 2 or len(parts[0]) != 64 or parts[1].lstrip("*") in out:
            refuse(f"{relpath(p)}: malformed or repeated line {ln!r}")
        out[parts[1].lstrip("*")] = parts[0]
    return out, h


def reissue_manifest() -> tuple[dict[str, str], str]:
    """D-305's manifest of its re-issues (pinned), naming twomoment_fit_r2.json and twomoment_supplement_r4.json."""
    listed, h = read_manifest(ROOT / REISSUE_MANIFEST, PINNED[REISSUE_MANIFEST])
    for r in (FIT_REL, SUP_REL):
        if r not in listed:
            refuse(f"{REISSUE_MANIFEST} does not list {r}")
    return listed, h


def read_listed(p: Path, name: str, listed: dict[str, str], man: str) -> tuple[dict[str, Any], str]:
    """A results JSON read once through read_result (D-297 (1)); refuses unless the manifest `man` lists it under
    `name` with these bytes (opcase_d384.read_bound's rule)."""
    doc = read_result(p)
    h = _RESULT_READS[SP.rel(p)]
    if listed.get(name) != h:
        refuse(f"{name} is not the file {man} names ({h[:16]} vs {str(listed.get(name))[:16]})")
    if not isinstance(doc, dict):
        refuse(f"{name} is not a JSON object")
    return doc, h


def bos_check(sides: dict[str, tuple[str, str]]) -> dict[str, Any]:
    """CHOICES (y): results/phase2/bos_check_p2.json read as an input; refuses unless every side's model (repo, rev) is
    'ok' there. Returns the block the record carries."""
    p = SP.P2 / Path(BOS_CHECK_REL).name
    if p.is_symlink() or not p.is_file():
        refuse(f"no {SP.rel(p)}: the Phase 2 BOS check (D-318 (6)) must have run (D-361 (5))")
    d = read_result(p)
    models = d.get("models") if isinstance(d.get("models"), dict) else {}
    bad = []
    if d.get("run_kind") != "phase2_bos_check" or d.get("scan_not_loaded") != []:
        bad.append("not a phase2_bos_check record with every tokenizer loaded")
    for side, (repo, rev) in sides.items():
        m = models.get(repo) if isinstance(models.get(repo), dict) else {}
        ok = (m.get("revision") == rev and (m.get("cache_tie") or {}).get("status") == "ok"
              and (m.get("cache_tie") or {}).get("ok") is True and (m.get("prompts_tie") or {}).get("ok") is True
              and (m.get("plain_path") or {}).get("ok") is True
              and not any(repo in (d.get(k) or []) for k in ("ties_failed", "cache_tie_skipped", "cache_tie_incomplete")))
        if not ok:
            bad.append(f"{side} {repo}@{str(rev)[:12]} is not 'ok'")
    if bad:
        refuse(f"{SP.rel(p)}: {bad} (D-361 (5): a pair whose caches fail a BOS check is excluded only by a D-entry)")
    return {"file": SP.rel(p), "sha256": _RESULT_READS[SP.rel(p)], "rule": choice("y"),
            "models": {side: {"repo": repo, "revision": rev, "ok": True} for side, (repo, rev) in sides.items()}}


def write_outputs(items: list[tuple[Path, bytes]]) -> None:
    """Stage every output, then name them (hard link, never overwrite, D-152 P5); a naming failure after an output was
    named leaves that output mode 444 and refuses naming it (opcase_d384.write_outputs' pattern), so a rerun's orphan
    check finds it."""
    staged: list[tuple[Path, Path, str]] = []
    committed: list[Path] = []
    try:
        for p, b in items:
            t_, h = stage_bytes(p, b)
            staged.append((t_, Path(p), h))
        for t_, p, h in staged:
            try:
                commit_staged(t_, p, h)
            except BaseException as e:
                for q in committed:
                    os.chmod(q, 0o444)
                refuse(f"naming {p} failed ({type(e).__name__}: {e}); already named, now mode 444: "
                       f"{', '.join(str(q) for q in committed) or 'none'}")
            committed.append(p)
    finally:
        for t_, _, _ in staged:
            if t_.exists():
                t_.unlink()
    for _, p, h in staged:
        print(f"Written: {relpath(p)}\nsha256:  {h}")


def end_hash_problems(files: dict[str, tuple[Path, str]]) -> list[str]:
    """The names whose file now hashes to other bytes than the run read (empty when every one is unchanged)."""
    return sorted(n for n, (p, h) in files.items() if not Path(p).is_file() or sha256_file(p) != h)


def d170_line(n: int) -> str:
    """CHOICES (z): the D-170 accounting carried on every summary line (the summary refuses a flagged cell)."""
    return f"D-170: 0 of {n} cells flagged"


# --------------------------------------------------------------------------------------------- the shared path

def fit_probes(X_tr: np.ndarray, y_tr: np.ndarray, seeds: list[int], C: float) -> list[P.Probe]:
    """Per seed, the selected-C probe exactly as twomoment_cells fits it (D-152 P1): paired.d117_split's fitting 80% in
    permutation order, paired.fit_probe."""
    out = []
    for seed in seeds:
        _, fit = P.d117_split(len(y_tr), seed)
        out.append(P.fit_probe(X_tr[fit], y_tr[fit], seed, C, "d117_permutation"))
    return out


def make_parts(draws: dict[str, np.ndarray], n_cal: int, n_pool: int) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Per level, the partitions in twomoment_cells' order: the frozen split first, then the draws."""
    frozen = (np.arange(n_cal), np.arange(n_cal, n_pool))
    return {lvl: [frozen] + [(d[:n_cal], d[n_cal:]) for d in dr] for lvl, dr in draws.items()}


def per_level(sc: dict[str, np.ndarray], report: dict[str, Any], arr: dict[str, np.ndarray], parts: list[Any],
              target: float, n_ev: int, where: str, lvl: str) -> tuple[list, dict, dict, np.ndarray, np.ndarray,
                                                                       np.ndarray]:
    """twomoment_cells' per_level, lines repeated: per-draw realised and predicted values for one level (column 0 = the
    frozen split), checked draw by draw against paired.repartition_estimate on the same partitions."""
    pool_pre = np.vstack([sc["cal_pre"], sc["ev_pre"]])
    pool_post = np.vstack([sc["cal_post"], sc["ev_post"]])
    per = [TC.one_partition(pool_pre, pool_post, c, e, target) for c, e in parts]
    real = np.array([x["ratio"] for x in per])
    fpost = np.array([x["fpr_post"] for x in per])
    if not (np.array_equal(real[1:], arr["paired_ratio"]) and real[0] == report["paired_ratio"]["frozen"]
            and np.array_equal(fpost[1:], arr["fpr_post"]) and fpost[0] == report["fpr_post"]["frozen"]):
        refuse(f"{where} {lvl}: realised values differ from paired.repartition_estimate")
    rcens = np.array([x["n_censored_seeds"] > 0 for x in per])
    cols: dict[str, np.ndarray] = {"log_real_ratio": np.log(real), "log_fpr_post": np.log(fpost),
                                   "real_censored": rcens}
    preds: dict[str, Any] = {}
    for key in PRED_KEYS:
        ok = np.array([x[key] is not None for x in per])
        ratio = np.array([x[key]["ratio"] if x[key] else np.nan for x in per])
        fpred = np.array([x[key]["fpr_pred"] if x[key] else np.nan for x in per])
        pc = np.array([bool(x[key] and x[key]["n_censored_seeds"] > 0) for x in per])
        cols[f"log_pred_ratio__{key}"] = np.log(ratio)
        cols[f"log_fpr_pred__{key}"] = np.log(fpred)
        cols[f"pred_censored__{key}"] = pc
        fz = per[0][key]
        union = (rcens | pc)[1:][ok[1:]]
        preds[key] = {"role": PREDICTOR_ROLE[key],
                      "ratio": P.band_summary(ratio[1:], ratio[0]),
                      "fpr_pred": P.band_summary(fpred[1:], fpred[0]),
                      "fraction_of_draws_floored": float(union.mean()) if union.size else None,
                      "n_draws_undefined": int((~ok[1:]).sum()),
                      "frozen_split": ({"ratio_per_seed": fz["ratio_per_seed"], "n_flagged_per_seed": fz["n_flagged"],
                                        "n_eval_negatives": n_ev, "n_censored_seeds": fz["n_censored_seeds"]}
                                       if fz else "undefined")}
    return per, cols, preds, rcens, pool_pre, pool_post


def compute_cell(sc: dict[str, np.ndarray], draws: dict[str, np.ndarray], parts: dict[str, list[Any]], target: float,
                 meta: dict[str, Any], verification: dict[str, Any], probes: list[P.Probe], n_ev: int,
                 cluster_note: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """One selected-C cell in twomoment_cells' form (its per-layer lines for the 'selected' probe set, repeated): the
    realised D-156 report at both levels (paired.repartition_estimate), every prediction per draw
    (twomoment_cells.one_partition), the moment changes with their bands, the cell dict and its per-draw columns.
    meta: pair, axis, layer, C, is_selected_layer."""
    where = f"layer {meta['layer']} selected"
    report, arr = P.repartition_estimate(sc["cal_pre"], sc["ev_pre"], sc["cal_post"], sc["ev_post"], sc["pos_pre"],
                                         sc["pos_post"], draws[LEVELS[0]], target)
    report_c, arr_c = P.repartition_estimate(sc["cal_pre"], sc["ev_pre"], sc["cal_post"], sc["ev_post"], sc["pos_pre"],
                                             sc["pos_post"], draws[LEVELS[1]], target)
    per, cols, preds, rcens, pool_pre, pool_post = per_level(sc, report, arr, parts[LEVELS[0]], target, n_ev, where,
                                                             LEVELS[0])
    per_c, cols_c, preds_c, rcens_c, _, _ = per_level(sc, report_c, arr_c, parts[LEVELS[1]], target, n_ev, where,
                                                      LEVELS[1])
    for key, v in cols_c.items():
        cols[f"{key}__cluster"] = v
    mc: dict[str, Any] = {}
    mc_c: dict[str, Any] = {}
    for s in TC.SOURCES:
        mc[s], mc_c[s] = {}, {}
        for m in ("mean_shift", "log_sd_ratio", "log_sd_tail_ratio"):
            v = np.array([x["moment_change"][s][m] for x in per])
            cols[f"moment_change__{s}__{m}"] = v
            mc[s][m] = P.band_summary(v[1:], v[0])
            vc = np.array([x["moment_change"][s][m] for x in per_c])
            cols[f"moment_change__{s}__{m}__cluster"] = vc
            mc_c[s][m] = P.band_summary(vc[1:], vc[0])
    fm = per[0]["moments"]
    mom = {"note": "D-163 (5): change_over_draws carries the D-156 band; no item bootstrap",
           "pooled_negatives_pre": {m: v.tolist() for m, v in TC.moments(pool_pre).items()},
           "pooled_negatives_post": {m: v.tolist() for m, v in TC.moments(pool_post).items()},
           "frozen_split_per_seed": {f"{part}_{side}": {m: v.tolist() for m, v in fm[s][i].items()}
                                     for s, part in (("in_sample", "eval_part"), ("out_of_sample", "calibration_part"))
                                     for i, side in enumerate(("pre", "post"))},
           "change_over_draws": mc}
    cell: dict[str, Any] = {
        "pair": meta["pair"], "axis": meta["axis"], "layer": meta["layer"], "probe_set": "selected",
        "probe_C": meta["C"], "fit_row_order": "d117_permutation", "is_selected_layer": meta["is_selected_layer"],
        "input_verification": verification, "realised": report,
        "realised_censoring": {"share_of_draws_floored": float(rcens[1:].mean()),
                               "frozen_n_censored_seeds": per[0]["n_censored_seeds"]},
        "predicted": preds,
        LEVELS[1]: {"note": cluster_note,
                    "realised": report_c,
                    "realised_censoring": {"share_of_draws_floored": float(rcens_c[1:].mean())},
                    "predicted": {k: {kk: v[kk] for kk in ("ratio", "fpr_pred", "fraction_of_draws_floored",
                                                          "n_draws_undefined")}
                                  for k, v in preds_c.items()},
                    "moments_change_over_draws": mc_c},
        "moments": mom, "probe_fit_diagnostics": [q.diag for q in probes]}
    return cell, cols


def score_matrices(probes: list[P.Probe], mats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """(n, seeds) scores of each matrix, one column per seed's probe (twomoment_cells' expression)."""
    return {k: np.column_stack([q.score(M) for q in probes]) for k, M in mats.items()}


def stack_rows(cols_by_cell: list[dict[str, np.ndarray]], sfx: str) -> dict[str, np.ndarray]:
    """Per-draw columns of the cells, one row per cell, keys suffixed by the target's suffix."""
    keys = list(cols_by_cell[0])
    if any(list(c) != keys for c in cols_by_cell):
        refuse("the cells' per-draw columns differ in keys")
    return {f"{k}{sfx}": np.vstack([c[k] for c in cols_by_cell]) for k in keys}


# --------------------------------------------------------------------------------------------- the anchor (Phase 1)

def bound_phase1(fit_rel: str, rec_rel: str, listed: dict[str, str] | None = None
                 ) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray], str]:
    """The fit of record (bound through D-305's manifest `listed` when given, else pinned), the per-pair record it names
    (by sha256) and that record's npz (by the sha256 both name)."""
    if listed is not None:
        fit, fit_sha = read_listed(ROOT / fit_rel, fit_rel, listed, REISSUE_MANIFEST)
    else:
        fit = read_result(ROOT / fit_rel)
        fit_sha = _RESULT_READS[SP.rel(ROOT / fit_rel)]
        if fit_sha != PINNED[fit_rel]:
            refuse(f"{fit_rel} is not the pinned fit of record ({fit_sha[:16]} vs {PINNED[fit_rel][:16]})")
    want = (fit.get("input_files_sha256") or {}).get(rec_rel)
    rec = read_result(ROOT / rec_rel)
    rec_sha = _RESULT_READS[SP.rel(ROOT / rec_rel)]
    if want is None or rec_sha != want:
        refuse(f"{rec_rel} is not the record {fit_rel} names")
    da = rec.get("draw_arrays") or {}
    if (fit.get("input_files_sha256") or {}).get(da.get("file")) != da.get("sha256"):
        refuse(f"{da.get('file')} is not the npz {fit_rel} names")
    z = read_npz_bytes(resolve(da["file"]), da.get("sha256"), da.get("bytes"), rec_rel)
    return fit, rec, z, rec_sha


def anchor_compare(rec: dict[str, Any], z: dict[str, np.ndarray], li: int, cell: dict[str, Any],
                   cols: dict[str, np.ndarray]) -> list[str]:
    """Every difference between the Phase 2 path's cell at layer li (selected C) and the record's: the cell as canonical
    JSON text, each per-draw column bit for bit (dtype, shape, bytes) against the npz row, and the npz keys the path
    does not form (exactly the reversed-order columns, NaN on the row)."""
    rows = [i for i, (l_, p_) in enumerate(zip(z["cell_layer"], z["cell_probe_set"]))
            if int(l_) == li and str(p_) == "selected"]
    cells = [c for c in rec["cells"] if c["layer"] == li and c["probe_set"] == "selected"]
    if len(rows) != 1 or len(cells) != 1:
        return [f"layer {li}: the record holds {len(cells)} selected cells and {len(rows)} npz rows (want 1 and 1)"]
    r, want = rows[0], cells[0]
    if rec["cells"].index(want) != r:
        return [f"layer {li}: the record's cell order and its npz rows disagree"]
    diffs = [f"layer {li}: cell key {k}" for k in sorted(set(cell) | set(want))
             if canon(cell.get(k)) != canon(want.get(k))]
    for k, v in cols.items():
        if k not in z:
            diffs.append(f"layer {li}: npz has no {k}")
            continue
        a = z[k][r]
        if a.dtype != v.dtype or a.shape != v.shape or a.tobytes() != v.tobytes():
            diffs.append(f"layer {li}: npz {k}")
    extra = set(z) - set(cols) - {"cell_layer", "cell_probe_set", "cell_is_selected_layer"}
    rev = {"log_real_ratio_reversed", "log_real_ratio_reversed__cluster"}
    if extra != rev:
        diffs.append(f"layer {li}: npz keys the Phase 2 path does not form: {sorted(extra)} (want {sorted(rev)})")
    else:
        diffs += [f"layer {li}: {k} is not NaN on a selected row" for k in sorted(rev) if not np.isnan(z[k][r]).all()]
    if bool(z["cell_is_selected_layer"][r]) != cell["is_selected_layer"]:
        diffs.append(f"layer {li}: cell_is_selected_layer")
    return diffs


ANCHOR_CONFIG_KEYS = ("pair", "axis", "pre_repo", "pre_rev", "post_repo", "post_rev", "selection_file",
                      "selection_sha256", "selected_C", "selected_layer", "layers", "seeds", "split", "target_fpr",
                      "tail_quantile", "sd_eps", "predictors", "moment_sources", "predictor_roles", "cluster_manifest",
                      "cluster_draw_seed", "n_eval_positives", "pooled_negatives", "n_draws", "draw_seed")
# not compared: input_registry_sources_sha256 (the registry's record list, which grows as extraction records are added;
# the caches' own sha256 are compared through input_cache_sha256), recorded beside in the anchor record


def run_anchor(n_draws: int = P.N_DRAWS, out: str | Path | None = None, record: str | None = None,
               record_fpr5: str | None = None, records_only: bool = False) -> Path:
    """D-361 (6): the Phase 2 path on ANCHOR_STEM's Phase 1 inputs reproduces its records at 1% and 5% exactly. A test
    passes `out` (outside results/) and the two records' repo-relative or absolute paths; a final run reads the records
    the fits of record name (twomoment_fit_r2.json through D-305's manifest at 1%, the pinned twomoment_fit_fpr5.json at
    5%). records_only: every check before the first cache read, then "RECORDS OK"; nothing is written."""
    SP.check_threads()
    code0 = P.code_block(CODE_FILES)
    final_path = (RESULTS / f"twomoment_p2_d361_anchor_{ANCHOR_STEM}.json").resolve()
    out_path = final_path if out is None else Path(out).resolve()
    final = out_path == final_path
    if out_path.exists() or out_path.is_symlink():
        refuse(f"{out_path} exists; results are never overwritten (D-152 P5)")
    if not final and out_path.is_relative_to(RESULTS.resolve()):
        refuse("a test run (--out) must not write under results/; only the final name may")
    stem = ANCHOR_STEM
    npz_read: dict[str, tuple[Path, str]] = {}
    if final:
        bad = final_problems(n_draws == P.N_DRAWS, (REISSUE_MANIFEST, FIT5_REL))
        if bad or record or record_fpr5:
            refuse(f"the final anchor name needs: {bad or 'the records of record (no record override)'}")
        listed, man_sha = reissue_manifest()
        recs = {}
        for t, fit_rel in ((PRIMARY_TARGET, FIT_REL), (BESIDE_TARGET, FIT5_REL)):
            rel = f"results/twomoment_{stem}{TARGET_SFX[t].replace('__', '_')}.json"
            fit, rec, z, sha = bound_phase1(fit_rel, rel, listed if fit_rel == FIT_REL else None)
            named_by = {"file": fit_rel, "sha256": _RESULT_READS[SP.rel(ROOT / fit_rel)]}
            if fit_rel == FIT_REL:
                named_by["bound_by"] = {"file": REISSUE_MANIFEST, "sha256": man_sha}
            recs[t] = {"rel": rel, "path": ROOT / rel, "rec": rec, "z": z, "sha256": sha, "fit": named_by}
            npz_read[rec["draw_arrays"]["file"]] = (resolve(rec["draw_arrays"]["file"]), rec["draw_arrays"]["sha256"])
    else:
        if not record or not record_fpr5:
            refuse("a test anchor names both records (1% and 5%)")
        recs = {}
        for t, name in ((PRIMARY_TARGET, record), (BESIDE_TARGET, record_fpr5)):
            p = resolve(name)
            rec = read_result(p)
            da = rec.get("draw_arrays") or {}
            z = read_npz_bytes(resolve(da.get("file", "")), da.get("sha256"), da.get("bytes"), p.name)
            recs[t] = {"rel": SP.rel(p), "path": p, "rec": rec, "z": z, "sha256": _RESULT_READS[SP.rel(p)], "fit": None}
            npz_read[str(da.get("file"))] = (resolve(da.get("file", "")), da.get("sha256"))
    r1 = recs[PRIMARY_TARGET]["rec"]
    for t, r in recs.items():
        c = r["rec"].get("config") or {}
        if r["rec"].get("run_kind") != "twomoment_cells" or c.get("pair") != stem or c.get("target_fpr") != t:
            refuse(f"{r['rel']} is not {stem}'s twomoment_cells record at {t}")
        if c.get("n_draws") != n_draws:
            refuse(f"{r['rel']} has {c.get('n_draws')} draws, this run {n_draws}")
    libs = {k: code0["libraries"].get(k) for k in SP.LIBS}
    rec_libs = {k: ((r1.get("code") or {}).get("libraries") or {}).get(k) for k in SP.LIBS}
    if libs != rec_libs:
        refuse(f"libraries {libs} are not the ones {recs[PRIMARY_TARGET]['rel']} ran with {rec_libs}")
    th = blas_threads()
    rec_th = {v: ((r1.get("env") or {}).get("blas_threads") or {}).get(v) for v in SP.THREAD_VARS}
    if th != rec_th:
        refuse(f"BLAS threads {th} are not the ones {recs[PRIMARY_TARGET]['rel']} ran with {rec_th}")
    c1 = r1["config"]
    sel_path = resolve(c1["selection_file"])
    sel_doc = read_result(sel_path)
    if _RESULT_READS[SP.rel(sel_path)] != c1["selection_sha256"]:
        refuse(f"{c1['selection_file']} is not the selection {recs[PRIMARY_TARGET]['rel']} names")
    seeds = list(sel_doc["config"]["seeds"])
    if float(sel_doc["config"]["holdout"]) != P.HOLDOUT:
        refuse("the selection's holdout is not D-117's")
    sel = next((m for m in sel_doc["models"] if m["pair"] == stem), None)
    if sel is None or not sel["selected"]["arm"].startswith("C="):
        refuse(f"no logistic-C selection for {stem} in {c1['selection_file']}")
    C_sel, L_sel = float(sel["selected"]["arm"].split("=")[1]), int(sel["selected"]["layer"])

    cfg = REG.run(stem)
    axis = cfg["axis"]
    if records_only:                                      # before the first cache read (the input registry's caches)
        print(f"RECORDS OK: anchor {stem} (records only; no cache read, nothing written)")
        return out_path
    needed = [("pre", "probe_train"), ("pre", "threshold_calib"), ("pre", "eval"), ("post", "threshold_calib"),
              ("post", "eval")]
    data: dict[str, dict[str, Any]] = {"pre": {}, "post": {}}
    input_sha: dict[str, str] = {}
    input_recs: dict[str, dict[str, Any]] = {}
    for side, part in needed:
        zz, r = REG.load(cfg[f"{side}_repo"], axis, part)
        data[side][part] = zz
        input_sha[f"{side}/{part}"] = r["sha256"]
        input_recs[f"{side}/{part}"] = r
    for t, r in recs.items():
        if r["rec"].get("input_cache_sha256") != input_sha:
            refuse(f"the caches read now are not the ones {r['rel']} read")
    verification = REG.verification(input_recs)
    for part in ("threshold_calib", "eval"):
        if not np.array_equal(data["pre"][part]["item_ids"], data["post"][part]["item_ids"]) or \
                not np.array_equal(data["pre"][part]["labels"], data["post"][part]["labels"]):
            refuse(f"{part} items or labels differ between pre and post")
    grids = [sorted(int(k.split("_")[1]) for k in zz.files if k.startswith("layer_"))
             for side in data for zz in data[side].values()]
    if any(g != grids[0] for g in grids):
        refuse(f"the input caches have different layer grids: {grids}")
    layers = grids[0]
    y_tr = data["pre"]["probe_train"]["labels"].astype(bool)
    ncal = ~data["pre"]["threshold_calib"]["labels"].astype(bool)
    y_ev = data["pre"]["eval"]["labels"].astype(bool)
    neg, pos = ~y_ev, y_ev
    pooled_ids = [str(i) for i in data["pre"]["threshold_calib"]["item_ids"][ncal]] + \
                 [str(i) for i in data["pre"]["eval"]["item_ids"][neg]]
    n_cal, n_pool = int(ncal.sum()), len(pooled_ids)
    cl_sha = sha256_file(TC.CLUSTERS)
    cl = read_result(TC.CLUSTERS)["axes"][axis]
    if cl["n_pooled_negatives"] != n_pool or cl["n_calibration"] != n_cal or \
            cl["pool_item_ids_sha256"] != sha256_bytes("\n".join(pooled_ids).encode("utf-8")):
        refuse(f"{TC.CLUSTERS.name} does not describe the {axis} pool")
    draws = {LEVELS[0]: P.repartition_draws(n_cal, n_pool - n_cal, n_draws, P.AXIS_DRAW_SEED[axis]),
             LEVELS[1]: P.cluster_repartition_draws(P.cluster_units(pooled_ids, cl["clusters"]), n_cal, n_draws,
                                                    P.AXIS_CLUSTER_DRAW_SEED[axis])}
    parts = make_parts(draws, n_cal, n_pool)
    mine_cfg = {"pair": stem, "axis": axis, "pre_repo": cfg["pre_repo"], "pre_rev": cfg["pre_rev"],
                "post_repo": cfg["post_repo"], "post_rev": cfg["post_rev"], "selection_file": c1["selection_file"],
                "selection_sha256": _RESULT_READS[SP.rel(sel_path)], "selected_C": C_sel, "selected_layer": L_sel,
                "layers": layers, "seeds": seeds, "split": f"D-117 (holdout {P.HOLDOUT})", "tail_quantile": TC.TAIL_Q,
                "sd_eps": TC.SD_EPS, "predictors": list(TC.PREDICTORS), "moment_sources": list(TC.SOURCES),
                "predictor_roles": PREDICTOR_ROLE,
                "cluster_manifest": {"file": "manifests/negative_clusters.json", "sha256": cl_sha},
                "cluster_draw_seed": P.AXIS_CLUSTER_DRAW_SEED[axis], "n_eval_positives": int(pos.sum()),
                "pooled_negatives": {"n_calibration": n_cal, "n_evaluation": n_pool - n_cal,
                                     "item_ids_sha256": sha256_bytes("\n".join(pooled_ids).encode("utf-8")),
                                     "order": "threshold_calib negatives then eval negatives, cache order"},
                "n_draws": n_draws, "draw_seed": P.AXIS_DRAW_SEED[axis]}
    diffs = []
    for t, r in recs.items():
        diffs += [f"{r['rel']}: config {k}" for k in ANCHOR_CONFIG_KEYS
                  if canon({**mine_cfg, "target_fpr": t}.get(k)) != canon(r["rec"]["config"].get(k))]
    if diffs:
        for d in diffs:
            print(f"  DIFFERENT {d}")
        raise SystemExit(f"ANCHOR FAILED: {len(diffs)} config values differ; nothing written")
    print(f"ANCHOR {stem} [{axis}]: the Phase 2 path on the Phase 1 inputs; selected C={C_sel:g} at layer {L_sel}; "
          f"{len(layers)} layers; {n_draws} draws per level; targets {list(TARGETS)}", flush=True)
    n_cells, n_cols = 0, 0
    for li in [L_sel] + [x for x in layers if x != L_sel]:
        k = f"layer_{li}"
        X = {s: {p: data[s][p][k] for p in ("threshold_calib", "eval")} for s in data}
        X_tr = data["pre"]["probe_train"][k]
        probes = fit_probes(X_tr, y_tr, seeds, C_sel)
        sc = score_matrices(probes, {"cal_pre": X["pre"]["threshold_calib"][ncal], "ev_pre": X["pre"]["eval"][neg],
                                     "cal_post": X["post"]["threshold_calib"][ncal], "ev_post": X["post"]["eval"][neg],
                                     "pos_pre": X["pre"]["eval"][pos], "pos_post": X["post"]["eval"][pos]})
        meta = {"pair": stem, "axis": axis, "layer": li, "C": C_sel, "is_selected_layer": li == L_sel}
        for t, r in recs.items():
            cell, cols = compute_cell(sc, draws, parts, t, meta, verification, probes, n_pool - n_cal, TC.CLUSTER_NOTE)
            d = anchor_compare(r["rec"], r["z"], li, cell, cols)
            if d:
                for x in d:
                    print(f"  DIFFERENT {r['rel']}: {x}")
                raise SystemExit(f"ANCHOR FAILED: {len(d)} differences at layer {li} ({t:g}) against {r['rel']}; "
                                 "nothing written")
            n_cells += 1
            n_cols += len(cols)
        print(f"  layer {li:>2}: identical at {', '.join(f'{t:g}' for t in recs)} (cell and per-draw rows, both levels)",
              flush=True)
    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"]:
        refuse("source changed during the run")
    # every result read (the fits, the records, the selection, the cluster manifest) and the records' npz, re-hashed at
    # the end (a test run checks the reads it can name: outside the repo a read is registered by its file name alone)
    known = {SP.rel(sel_path): sel_path, SP.rel(TC.CLUSTERS): Path(TC.CLUSTERS),
             **{r["rel"]: r["path"] for r in recs.values()}}
    moved = end_hash_problems({**{k: (known.get(k) or resolve(k), h) for k, h in _RESULT_READS.items()
                                  if final or k in known}, **npz_read})
    if moved:
        refuse(f"changed during the run: {moved}")
    config = {**mine_cfg, "targets": list(TARGETS), "probe_set": "selected", "anchor_stem": stem,
              "phase2_path": "fit_probes, compute_cell (twomoment_cells' lines for the selected C), make_parts, "
                             "per-draw columns; the inputs read through src/input_registry.py with Phase 1's partitions",
              "input_registry_sources_sha256": {"now": REG.sources_sha256(),
                                                "record": c1.get("input_registry_sources_sha256"),
                                                "note": "not compared: the registry's record list grows; the caches "
                                                        "read are compared by sha256 (input_cache_sha256)"},
              "blas_threads": th}
    config.pop("target_fpr", None)
    payload = {**provenance_block(RUN_KIND["anchor"]), "preregistration": PREREG, "code": code0,
               "licence": anchor_licence(code0),
               "registered_parameters": n_draws == P.N_DRAWS, "choices": CHOICES,
               "config": config, "config_hash": canonical_hash(config), "input_verification": verification,
               "inputs": dict(_RESULT_READS),
               "input_cache_sha256": input_sha,
               "input_cache_source": {key: {"file": r["file"], "expected_from": r["expected_from"],
                                            "pin_only": r["pin_only"]} for key, r in input_recs.items()},
               "anchor": {"stem": stem, "exact_equal": True, "values_differing": 0,
                          "n_cells_compared": n_cells, "n_per_draw_columns_compared": n_cols,
                          "records": {f"{t:g}": {"file": r["rel"], "sha256": r["sha256"],
                                                 "draw_arrays": {"file": r["rec"]["draw_arrays"]["file"],
                                                                 "sha256": r["rec"]["draw_arrays"]["sha256"]},
                                                 "named_by": r["fit"]} for t, r in recs.items()},
                          "compared": "every selected-C cell at every layer (canonical JSON text of the whole cell, "
                                      "both levels) and every per-draw column of those cells bit for bit; the config "
                                      f"keys {list(ANCHOR_CONFIG_KEYS)}; C=1 cells and the reversed-order refit are "
                                      "outside the Phase 2 path (CHOICES (a), (q))"}}
    digest = write_results(out_path, payload, exclusive=True)
    print(f"ANCHOR OK: {n_cells} cells and {n_cols} per-draw columns identical\nWritten: {relpath(out_path)}\n"
          f"sha256:  {digest}")
    return out_path


REGISTRY_FILE = "src/populations.py"
LICENCE_RULE = ("F-080, D-411 (the author's decision of 2026-10-08): an anchor licenses a run when it was written by the "
                "same code that computes its numbers - the sha256 of every CODE_FILES file EXCEPT the population registry "
                f"{REGISTRY_FILE} - and reads the same population groups - a hash of each registry name the CODE_FILES "
                "modules reference as POP.<name> or populations.<name>; declaring any OTHER group leaves the licence")


def _normalised(value: Any) -> Any:
    """JSON-stable form of a registry value: sets and tuples become sorted or ordered lists, so the hash does not depend
    on string-hash randomisation."""
    if isinstance(value, dict):
        return {str(k): _normalised(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((_normalised(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True, default=str))
    if isinstance(value, (list, tuple)):
        return [_normalised(v) for v in value]
    return value


SRC_ROOT = Path(__file__).resolve().parent.parent              # where the code is read from (ROOT can be a data world)


def registry_names_read(files: list[str] | None = None, root: Path | None = None) -> list[str]:
    """Every registry name a CODE_FILES module (the registry itself excepted) references as POP.<name> or
    populations.<name>, comments and strings included (over-inclusive, never under); refuses a 'from populations
    import', which this scan could not follow. The sources are read beside this module (SRC_ROOT), not under ROOT."""
    names: set[str] = set()
    for f in (files if files is not None else CODE_FILES):
        if f == REGISTRY_FILE:
            continue
        text = ((root or SRC_ROOT) / f).read_text(encoding="utf-8")
        if re.search(r"^\s*from\s+populations\s+import\b", text, flags=re.MULTILINE):
            refuse(f"{f} imports names from populations directly; the anchor licence follows POP.<name> only (F-080)")
        if f == THIS_FILE:      # the licence machinery itself (getattr over the names found); hashed in the code map
            names.update(re.findall(r"\b(?:POP|populations)\.([A-Za-z_][A-Za-z0-9_]*)\b", text))
            continue
        if re.search(r"^\s*import\s+populations\s+as\s+(?!POP\b)\w+", text, flags=re.MULTILINE):
            refuse(f"{f} imports populations under another alias; the anchor licence follows POP.<name> only (F-080)")
        bare = re.sub(r"^\s*import\s+populations\s+as\s+POP\b", "", text, flags=re.MULTILINE)
        if re.search(r"\bPOP\b(?!\.[A-Za-z_<])", bare):                   # 'POP.<name>' in prose is allowed
            refuse(f"{f} uses the registry module other than as POP.<name> (getattr, vars, an assignment); the "
                   "anchor licence could not follow it (F-080)")
        names.update(re.findall(r"\b(?:POP|populations)\.([A-Za-z_][A-Za-z0-9_]*)\b", text))
    names.discard("py")                                          # 'populations.py' in a comment names the file
    return sorted(names)


def anchor_licence(code0: dict[str, Any]) -> dict[str, Any]:
    """What an anchor licence binds (F-080, D-411): the code block's sha256 of every file but the registry, and a hash of
    each registry name the code reads (a callable's source, a value's canonical JSON; 'absent' when the registry does
    not define it)."""
    groups: dict[str, str] = {}
    for name in registry_names_read():
        if not hasattr(POP, name):
            groups[name] = "absent"
        elif callable(getattr(POP, name)):
            groups[name] = sha256_bytes(inspect.getsource(getattr(POP, name)).encode("utf-8"))
        else:
            groups[name] = canonical_hash(_normalised(getattr(POP, name)))
    return {"rule": LICENCE_RULE, "registry": REGISTRY_FILE,
            "code_sha256_without_registry": {f: h for f, h in sorted(code0["sha256"].items()) if f != REGISTRY_FILE},
            "registry_names_read": groups}


def passing_anchor(code0: dict[str, Any], records_dir: Path | None = None, final: bool = False) -> dict[str, Any]:
    """The anchor record, read as a registered input: it passed, and was written by this exact code (every CODE_FILES
    sha256), libraries, BLAS threads and host class; for a final name also with the registered parameters. Returns
    {file, sha256, ...}."""
    p = (records_dir or RESULTS) / f"twomoment_p2_d361_anchor_{ANCHOR_STEM}.json"
    if p.is_symlink() or not p.is_file():
        refuse(f"no anchor record {p.name}: run --anchor with this code first (D-361 (6)); no Phase 2 output is written "
               "before the anchor has passed")
    a = read_result(p)
    an = a.get("anchor") or {}
    host = {k: PV.env_state().get(k) for k in SP.HOST_KEYS}
    problems = []
    if a.get("run_kind") != RUN_KIND["anchor"] or an.get("exact_equal") is not True or an.get("values_differing") != 0:
        problems.append("it did not pass")
    if a.get("licence") != anchor_licence(code0):
        problems.append("it was written by other code or reads other population groups (its licence, F-080, D-411)")
    if {k: ((a.get("code") or {}).get("libraries") or {}).get(k) for k in SP.LIBS} != \
            {k: code0["libraries"].get(k) for k in SP.LIBS}:
        problems.append("other libraries")
    if (a.get("config") or {}).get("blas_threads") != blas_threads():
        problems.append("other BLAS threads")
    if {k: (a.get("env") or {}).get(k) for k in SP.HOST_KEYS} != host:
        problems.append("another host class")
    if final and a.get("registered_parameters") is not True:
        problems.append("it did not run the registered parameters (1000 draws)")
    if problems:
        refuse(f"the anchor record {p.name} cannot license this run: {problems}")
    return {"file": SP.rel(p), "sha256": _RESULT_READS[SP.rel(p)], "exact_equal": True,
            "n_cells_compared": an.get("n_cells_compared"), "records": an.get("records")}


# --------------------------------------------------------------------------------------------- Phase 2 inputs

def durable_select(repo: str) -> dict[str, Any]:
    """The post side's durable select set (CHOICES (g)): the p2tpl select record where it is ingested, else its p2sub
    select record; exactly one. Returns {kind, path, rel, rec, rev, index_path, index, source (p2sub only)}."""
    tpl = F16.read_select(repo)                                  # this repo, set select, ingest yes, documented tag
    tpl_index = SP.INGEST_INDEX / f"{tpl['path'].stem}.sha256"
    om = repo.replace("/", "__")
    subs = sorted(SP.P2.glob(f"extract__{om}__{T.SUB_TAG}@*__select.json"))
    sub_indexed = [q for q in subs if (SP.INGEST_INDEX / f"{q.stem}.sha256").is_file()]
    if tpl_index.is_file() and sub_indexed:
        refuse(f"{repo} has two durable select sets ({tpl['rel']} and {[q.name for q in sub_indexed]}); the post side's "
               "calibration cache must come from one")
    if tpl_index.is_file():
        ip, idx = SP.ingest_index(tpl["path"])
        return {"kind": "p2tpl", "path": tpl["path"], "rel": tpl["rel"], "rec": tpl["rec"], "rev": tpl["rev"],
                "index_path": ip, "index": idx}
    if len(sub_indexed) != 1 or len(subs) != 1:
        refuse(f"{repo}: no ingested p2tpl select record and {len(sub_indexed)} ingested p2sub select records of "
               f"{len(subs)} (want exactly one): ingest its durable select set first (D-325, D-307 (1))")
    p = subs[0]
    rec = read_result(p)
    want = {"run_kind": "subset_p2", "cache_tag": T.SUB_TAG, "cache_repo": f"{repo}__{T.SUB_TAG}", "repo": repo,
            "revision": tpl["rev"], "cache_revision": tpl["rev"], "set_name": "select", "ingest": "yes"}
    got = {k: rec.get(k) for k in want}
    if got != want or p.name != f"extract__{om}__{T.SUB_TAG}@{tpl['rev'][:12]}__select.json":
        refuse(f"{p.name}: {[f'{k}={got[k]!r}' for k in want if got[k] != want[k]]}, expected {want}")
    src = rec.get("source_record") or {}
    if src.get("file") != tpl["rel"] or src.get("sha256") != _RESULT_READS[tpl["rel"]]:
        refuse(f"{p.name} was not cut from {tpl['rel']} as it is now")
    ip, idx = SP.ingest_index(p)
    return {"kind": "p2sub", "path": p, "rel": SP.rel(p), "rec": rec, "rev": tpl["rev"], "index_path": ip,
            "index": idx, "source": tpl}


def check_pair_plan(sb: dict[str, Any], sr: dict[str, Any], pooling: str, pair: list[str], role: str) -> dict[str, Any]:
    """A subset record's layer plan names this pair once in `role` with this selection record, its sha256 and L* (a
    FULL_GRID plan keeps every layer and names none; recorded as such)."""
    pp = ((sb["rec"].get("layer_plan") or {}).get("per_pooling") or {}).get(pooling)
    if not isinstance(pp, dict):
        refuse(f"{sb['rel']} has no layer_plan for {pooling}")
    if pp.get("full_grid_reason"):
        return {"record": sb["rel"], "full_grid_reason": pp["full_grid_reason"], "entries": 0}
    ent = [e for e in (pp.get("l_star") or []) if isinstance(e, dict) and e.get("role") == role
           and list(e.get("pair") or []) == pair]
    want = (sr["sel_rel"], _RESULT_READS[sr["sel_rel"]], sr["layer"])
    got = [(e.get("selection_record"), e.get("selection_record_sha256"), e.get("layer")) for e in ent]
    if len(ent) != 1 or got[0] != want:
        refuse(f"{sb['rel']}'s layer plan for {pooling} does not name, once, the pair {pair[0]} -> {pair[1]} as its "
               f"{role} side with {sr['sel_rel']} ({want[1][:16]}) at layer {sr['layer']}: {got}")
    return {"record": sb["rel"], "full_grid_reason": None, "entries": 1}


def plan_grid(sb: dict[str, Any]) -> tuple[int, list[int]]:
    lp = sb["rec"].get("layer_plan") or {}
    nl = lp.get("n_model_layers")
    if not isinstance(nl, int) or isinstance(nl, bool) or nl < 1:
        refuse(f"{sb['rel']}'s layer plan names no n_model_layers")
    g6 = SUB.grid6(nl)
    if lp.get("grid6") != g6:
        refuse(f"{sb['rel']}'s layer plan names GRID6 {lp.get('grid6')}, not subset_p2.grid6({nl}) = {g6}")
    return nl, g6


def open_sub_cache(sb: dict[str, Any], s: dict[str, Any], part: str, pooling: str,
                   layers: list[int]) -> tuple[Any, list[str], list[int], dict[str, Any]]:
    """One durable subset cache (p2sub, rest or select), checked as selection_tpr_v2.load_subset_cache checks it but
    opened once for every layer: data/hidden only, a regular file, mode 444, its sha256 the record's (the entry was
    checked against the ingest index by selection_tpr_v2.subset_entry), exactly the record's layers_resolved, every
    wanted layer among them, no flagged item, n items, item for item its frozen split. Returns a lazy npz."""
    key = f"{AXIS}/{part}/{pooling}"
    name, want, lay = s["cache_file"], s["cache_sha256"], s["layers_resolved"]
    p = SP.CACHE / name
    if p.is_symlink() or not p.is_file():
        refuse(f"{name} is not a regular file in {SP.rel(SP.CACHE)}: ingest {sb['rel']} first (D-307 (1)); NVMe scratch "
               "is never read for a result")
    if (p.stat().st_mode & 0o777) != 0o444:
        refuse(f"{name} is mode {oct(p.stat().st_mode & 0o777)}, not 0o444 (an ingested cache is read-only)")
    have = sha256_file(p)
    if have != want:
        refuse(f"{name} is not the bytes {sb['rel']} names ({have[:16]} vs {want[:16]})")
    z = np.load(p, allow_pickle=False)
    in_file = sorted(int(k.split("_")[1]) for k in z.files if k.startswith("layer_"))
    if in_file != sorted(lay):
        refuse(f"{name} holds layers {in_file}, not the record's layers_resolved {lay}")
    missing = [x for x in layers if x not in lay]
    if missing:
        refuse(f"{name} lacks layers {missing} of this pair's layer set")
    if "nonfinite" not in z.files or bool(np.asarray(z["nonfinite"]).any()):
        refuse(f"{name} has no nonfinite flag, or flags a non-finite item")
    ids = [str(x) for x in z["item_ids"]]
    labels = [int(x) for x in z["labels"]]
    if len(ids) != s.get("n") or len(labels) != len(ids):
        refuse(f"{name} holds {len(ids)} items, the record says {s.get('n')}")
    rows = SP.split_rows(s, key)
    if [r["item_id"] for r in rows] != ids or [int(r["label"]) for r in rows] != labels:
        refuse(f"{name}: item ids or labels are not the frozen split's, item for item")
    return z, ids, labels, {"file": name, "sha256": have, "expected_from": sb["rel"],
                            "layers_resolved": lay, "split": {"file": s["split_file"], "sha256": s["split_sha256"]},
                            "source_cache": {"file": s.get("source_cache_file"),
                                             "sha256": s.get("source_cache_sha256")}}


def p2_cluster_units(pool_ids: list[str], n_cal: int, v2: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """CHOICES (e): the D-164 units on the pooled negatives from the Phase 2 cluster manifest, bound through the v2
    record's v1 record. Returns (unit label per pooled item, the record's cluster_units block)."""
    v1_rec = (v2["rec"].get("v1") or {})
    v1_path = SP.P2 / V1_RECORD_NAME
    if v1_rec.get("record") != SP.rel(v1_path):
        refuse(f"{v2['rel']} names the v1 record {v1_rec.get('record')!r}, not {SP.rel(v1_path)}")
    v1 = read_result(v1_path)
    if _RESULT_READS[SP.rel(v1_path)] != v1_rec.get("record_sha256"):
        refuse(f"{SP.rel(v1_path)} is not the v1 record {v2['rel']} names")
    want = (v1.get("inputs") or {}).get(CLUSTER_MANIFEST_REL)
    man_path = MANIFESTS / Path(CLUSTER_MANIFEST_REL).name
    man = read_result(man_path)
    if not isinstance(want, str) or _RESULT_READS[SP.rel(man_path)] != want:
        refuse(f"{CLUSTER_MANIFEST_REL} is not the cluster manifest {SP.rel(v1_path)} read ({str(want)[:16]})")
    pool = man.get("pool") or {}
    ids, labels = pool.get("item_id"), pool.get("cluster")
    if not isinstance(ids, list) or not isinstance(labels, list) or len(ids) != len(labels) or \
            sha256_bytes("\n".join(ids).encode("utf-8")) != pool.get("item_ids_sha256") or len(set(ids)) != len(ids):
        refuse(f"{CLUSTER_MANIFEST_REL}: the pool's item ids, labels and item_ids_sha256 do not agree")
    label = dict(zip(ids, labels))
    missing = [i for i in pool_ids if i not in label]
    if missing:
        refuse(f"{len(missing)} pooled negatives are not in {CLUSTER_MANIFEST_REL}'s pool (first {missing[0]})")
    groups: dict[str, list[str]] = {}
    for i in pool_ids:
        groups.setdefault(label[i], []).append(i)
    clusters = [g for g in groups.values() if len(g) > 1]
    unit = P.cluster_units(pool_ids, clusters)
    pos = {i: j for j, i in enumerate(pool_ids)}
    straddle = sum(1 for g in clusters if len({pos[i] < n_cal for i in g}) == 2)
    return unit, {"manifest": {"file": CLUSTER_MANIFEST_REL, "sha256": want},
                  "bound_by": {"file": SP.rel(v1_path), "sha256": _RESULT_READS[SP.rel(v1_path)]},
                  "n_units": int(len(np.unique(unit))), "n_multi_item_units": len(clusters),
                  "items_in_multi_item_units": int(sum(len(g) for g in clusters)),
                  "largest_unit": int(max((len(g) for g in clusters), default=1)),
                  "multi_item_units_split_by_the_frozen_v2_split": straddle,
                  "rule": choice("e")}


def probe_fingerprints(probes: list[P.Probe], seeds: list[int]) -> list[dict[str, Any]]:
    """float16_check_d364's fingerprint of each seed's probe (its record's probes.per_seed form)."""
    return [{"seed": s, "mu_sha256": sha256_bytes(np.ascontiguousarray(q.mu).tobytes()),
             "sd_sha256": sha256_bytes(np.ascontiguousarray(q.sd).tobytes()),
             "w_sha256": sha256_bytes(np.ascontiguousarray(q.w).tobytes()), "b": float(q.b),
             "dtypes": {"mu": str(q.mu.dtype), "sd": str(q.sd.dtype), "w": str(np.asarray(q.w).dtype)}}
            for s, q in zip(seeds, probes)]


def float16_cross_check(f16: dict[str, Any], probes: list[P.Probe], seeds: list[int], sc: dict[str, np.ndarray],
                        reports: dict[float, dict[str, Any]], mine: dict[str, Any]) -> list[str]:
    """CHOICES (o): every difference between this job's D-364 cell and float16_check_d364.json's arm A (empty when
    identical): its config (repos, pooling, layer, arm, seeds, draws, draw seed), pooled negatives and positives, each
    seed's probe fingerprint, the six score matrices' sha256 and the realised D-156 report at each target."""
    diffs = []
    c = f16.get("config") or {}
    for k in ("pre_repo", "post_repo", "pooling", "layer", "arm", "seeds", "n_draws", "draw_seed"):
        if c.get(k) != mine[k]:
            diffs.append(f"config {k}: {c.get(k)!r} vs {mine[k]!r}")
    if (f16.get("pooled_negatives") or {}).get("item_ids_sha256") != mine["pooled_item_ids_sha256"] or \
            f16.get("n_eval_positives") != mine["n_eval_positives"]:
        diffs.append("pooled negatives or positives")
    theirs = ((f16.get("probes") or {}).get("per_seed")) or []
    fp = probe_fingerprints(probes, seeds)
    if len(theirs) != len(fp) or any(canon(a) != canon(b) for a, b in zip(fp, theirs)):
        diffs.append("probe fingerprints (mu, sd, w sha256, b, dtypes)")
    want = ((f16.get("scores_sha256") or {}).get("A")) or {}
    name = {("pre", "calibration"): "cal_pre", ("pre", "eval_neg"): "ev_pre", ("pre", "eval_pos"): "pos_pre",
            ("post", "calibration"): "cal_post", ("post", "eval_neg"): "ev_post", ("post", "eval_pos"): "pos_post"}
    for (side, m), k in name.items():
        if (want.get(side) or {}).get(m) != sha256_bytes(sc[k].astype("<f8").tobytes()):
            diffs.append(f"score matrix {side}/{m}")
    ops = ((f16.get("operating_points") or {}).get("A")) or {}
    for t, rep in reports.items():
        if canon(rep) != canon(ops.get(f"target_{t:g}")):
            diffs.append(f"realised D-156 report at {t:g}")
    return diffs


# --------------------------------------------------------------------------------------------- the per-pair job

def out_paths(key: str, pooling: str, out_dir: Path) -> tuple[Path, Path]:
    j = out_dir / f"twomoment_p2_d361_{key}_{pooling}.json"
    return j, j.with_name(f"{j.stem}_draws.npz")


def run_files() -> list[str]:
    """The 45 files the list's manifest job hashes (logs/twomoment_p2_d361.sha256), repo-relative: the anchor record,
    the 20 pair records and the 2 LoRA records, each with its npz."""
    names = [f"twomoment_p2_d361_anchor_{ANCHOR_STEM}.json"]
    names += [p.name for k in PAIRS for pl in POOLINGS for p in out_paths(k, pl, Path("."))]
    names += [n for pl in POOLINGS for n in (f"twomoment_p2_d361_lora_{pl}.json", f"twomoment_p2_d361_lora_{pl}_draws.npz")]
    return [f"results/{n}" for n in names]


def run_pair(key: str, pooling: str, n_draws: int = P.N_DRAWS, out_dir: str | Path | None = None,
             records_only: bool = False) -> Path:
    """One pair and pooling (module docstring, THE PER-PAIR JOB). records_only: steps 1-2 without the anchor record
    (a dry run before the list), then "RECORDS OK"; no cache is read and nothing is written."""
    if key not in PAIRS:
        refuse(f"{key!r} is not a D-361 pair: {list(PAIRS)}")
    if pooling not in POOLINGS:
        refuse(f"pooling must be one of {POOLINGS}")
    pre_repo, post_repo, group, note = PAIRS[key]
    threads = SP.check_threads()
    code0 = P.code_block(CODE_FILES)
    final_dir = RESULTS.resolve()
    odir = final_dir if out_dir is None else Path(out_dir).resolve()
    final = odir == final_dir
    out_path, npz_path = out_paths(key, pooling, odir)
    if not final and odir.is_relative_to(RESULTS.resolve()):
        refuse("a test run (--out-dir) must not write under results/; only the final names may")
    if npz_path.exists() and not out_path.exists():
        refuse(f"{npz_path} exists without {out_path.name}: the orphan of a failed write. Move it aside, never delete "
               f"it, to {npz_path.stem}.orphan-<utc>.npz, and rerun")
    if out_path.exists() or out_path.is_symlink():
        refuse(f"{out_path} exists; results are never overwritten (D-152 P5)")

    # step 1: records only
    anchor = None if records_only else passing_anchor(code0, final=final)
    sr = T.read_selection(pre_repo, pooling, threads, code0)
    arm, L_sel, seeds, holdout = sr["arm"], sr["layer"], sr["seeds"], sr["holdout"]
    if not arm.startswith("C="):
        refuse(f"{sr['sel_rel']} selected {arm}; D-361 (2) names the selected C (a logistic arm)")
    if sr["target_fpr"] != PRIMARY_TARGET or holdout != P.HOLDOUT:
        refuse(f"{sr['sel_rel']}: target {sr['target_fpr']!r}, holdout {holdout!r} (want {PRIMARY_TARGET}, {P.HOLDOUT})")
    C_sel = float(arm.split("=")[1])
    registered = n_draws == REGISTERED["n_draws"] and list(seeds) == REGISTERED["seeds"]
    v2 = T.read_v2(code0)
    ev2 = F16.read_v2_eval(v2)
    pinned_keys: tuple[str, ...] = ()
    if (key, pooling) == FLOAT16_CELL:
        pinned_keys = ("results/phase2/float16_check_d364.json",)
    if final:
        bad = final_problems(registered, pinned_keys)
        have = {"v2/calibration": v2["sha256"], "v2/eval": ev2["sha256"]}
        bad += [f"{k} is not D-362's ({have[k][:16]})" for k in have if have[k] != PINNED[k]]
        if bad:
            refuse(f"the final names need: {bad}")
    sb = {"pre": T.read_subset(pre_repo, sr["rev"])}
    plans = {"pre_all_pre_entries": T.check_l_star(sb["pre"], sr, pooling),
             "pre_this_pair": check_pair_plan(sb["pre"], sr, pooling, [pre_repo, post_repo], "pre")}
    post_sel = durable_select(post_repo)
    bos = bos_check({"pre": (pre_repo, sr["rev"]), "post": (post_repo, post_sel["rev"])})       # CHOICES (y)
    sb["post"] = T.read_subset(post_repo, post_sel["rev"])
    plans["post_rest"] = check_pair_plan(sb["post"], sr, pooling, [pre_repo, post_repo], "post")
    if post_sel["kind"] == "p2sub":
        plans["post_select"] = check_pair_plan(post_sel, sr, pooling, [pre_repo, post_repo], "post")
    nl, g6 = plan_grid(sb["pre"])
    if plan_grid(sb["post"]) != (nl, g6):
        refuse(f"the two sides' layer plans name other layer counts or GRID6: {plan_grid(sb['post'])} vs {(nl, g6)}")
    required = sorted(set(g6) | {L_sel})                  # GRID6 plus the pre side's L* (D-325 (2))
    index_pre = SP.ingest_index(sr["rec_path"])
    # the ingest indexes this job reads, hashed when read (each must hash the same at the end)
    index_read = {SP.rel(SP.INGEST_INDEX / f"{p.stem}.sha256"): SP.INGEST_INDEX / f"{p.stem}.sha256"
                  for p in (sr["rec_path"], sb["pre"]["path"], sb["post"]["path"], post_sel["path"])}
    index_sha0 = {k: sha256_file(p) for k, p in index_read.items()}

    # the entries (records only) and the layers every cache holds
    entries: dict[str, dict[str, Any]] = {}
    s_sub: dict[str, dict[str, dict[str, Any]]] = {}
    for part in ("probe_train", "calibration"):
        e = (sr["rec"].get("splits") or {}).get(f"{AXIS}/{part}/{pooling}")
        if not isinstance(e, dict):
            refuse(f"{sr['rec_rel']} has no split {AXIS}/{part}/{pooling}")
        entries[f"pre/{AXIS}/{part}/{pooling}"] = e
    for side in ("pre", "post"):
        s_sub[side] = {part: T.subset_entry(sb[side], part, pooling, L_sel) for part in ("eval", "pool4d")}
        for part in ("eval", "pool4d"):
            entries[f"{side}/{T.SUB_TAG}/{AXIS}/{part}/{pooling}"] = s_sub[side][part]
    if post_sel["kind"] == "p2tpl":
        e = (post_sel["rec"].get("splits") or {}).get(f"{AXIS}/calibration/{pooling}")
        if not isinstance(e, dict):
            refuse(f"{post_sel['rel']} has no split {AXIS}/calibration/{pooling}")
        entries[f"post/{AXIS}/calibration/{pooling}"] = e
        post_cal_layers = list(post_sel["rec"].get("layers_resolved") or [])
    else:
        sel_sb = {**post_sel, "rest": post_sel["source"]["rec"], "rest_rel": post_sel["source"]["rel"]}
        e = T.subset_entry(sel_sb, "calibration", pooling, L_sel)
        entries[f"post/{T.SUB_TAG}/{AXIS}/calibration/{pooling}"] = e
        post_cal_layers = list(e.get("layers_resolved") or [])
    v1_keyed = F16.check_v1_keyed(entries, v2)
    held = set(sr["rec"].get("layers_resolved") or []) & set(post_cal_layers)
    for side in ("pre", "post"):
        for part in ("eval", "pool4d"):
            held &= set(s_sub[side][part].get("layers_resolved") or [])
    # step 2 (CHOICES (f)): the cells are every layer both sides hold; GRID6 + the pre side's L* must be among them
    if not set(required) <= held:
        refuse(f"the layers both sides hold {sorted(held)} do not include GRID6 + L* {required} (missing "
               f"{sorted(set(required) - held)}; D-361 (2), D-325 (2), CHOICES (f))")
    layers = sorted(held)
    beyond = sorted(held - set(required))
    neg_ids = [i for i, lab in zip(ev2["ids"], ev2["labels"]) if lab == 0]
    pool_ids = list(v2["ids"]) + neg_ids
    n_cal, n_ev, n_pos = v2["n"], ev2["n_neg"], ev2["n_pos"]
    if final and (n_cal, n_ev, n_pos) != V2_SIZES:
        refuse(f"split v2 holds {(n_cal, n_ev, n_pos)} calibration, evaluation-negative and positive items, not D-361 "
               f"(2)'s {V2_SIZES}")
    unit, units_block = p2_cluster_units(pool_ids, n_cal, v2)
    if records_only:                                      # steps 1-2 done; no cache has been read
        print(f"RECORDS OK: {key} {pooling} (records only; no cache read, nothing written)")
        return out_path
    print(f"D-361 {key} [{AXIS}, {pooling}]: {pre_repo} -> {post_repo} ({group}); {arm} at L* {L_sel} (from "
          f"{sr['sel_rel']}); layers {layers} (beyond GRID6 + L*: {beyond}); v2 pooled negatives {n_cal} + {n_ev}, "
          f"{n_pos} positives; {n_draws} draws per level; targets {list(TARGETS)}; anchor {anchor['file']}; registered "
          f"{registered}", flush=True)

    # step 3: the durable caches (D-307 (1))
    z_tr, r_tr, _ = SP.load_ingested(sr["rec"], sr["rec_rel"], index_pre[1], "probe_train", pooling)
    z_cal = {}
    r_cal = {}
    z_cal["pre"], r_cal["pre"], _ = SP.load_ingested(sr["rec"], sr["rec_rel"], index_pre[1], "calibration", pooling)
    recorded = sr["sel"].get("input_cache_sha256") if isinstance(sr["sel"].get("input_cache_sha256"), dict) else {}
    for part, r in (("probe_train", r_tr), ("calibration", r_cal["pre"])):
        k = f"pre/{AXIS}/{part}/{pooling}"
        if recorded.get(k) != r["sha256"]:
            refuse(f"{r['file']} ({r['sha256'][:16]}) is not the cache {sr['sel_rel']} read for {k}")
    if post_sel["kind"] == "p2tpl":
        z_cal["post"], r_cal["post"], _ = SP.load_ingested(post_sel["rec"], post_sel["rel"], post_sel["index"],
                                                           "calibration", pooling)
        post_cal_key = f"post/{AXIS}/calibration/{pooling}"
    else:
        z_cal["post"], ids_pc, labs_pc, r_cal["post"] = open_sub_cache(
            post_sel, entries[f"post/{T.SUB_TAG}/{AXIS}/calibration/{pooling}"], "calibration", pooling, layers)
        post_cal_key = f"post/{T.SUB_TAG}/{AXIS}/calibration/{pooling}"
    z_sub: dict[str, dict[str, Any]] = {}
    r_sub: dict[str, dict[str, dict[str, Any]]] = {}
    meta_sub: dict[str, dict[str, tuple[list[str], list[int]]]] = {}
    for side in ("pre", "post"):
        z_sub[side], r_sub[side], meta_sub[side] = {}, {}, {}
        for part in ("eval", "pool4d"):
            zz, ids, labs, r = open_sub_cache(sb[side], s_sub[side][part], part, pooling, layers)
            z_sub[side][part], r_sub[side][part], meta_sub[side][part] = zz, r, (ids, labs)
    cal_meta = {side: ([str(x) for x in z_cal[side]["item_ids"]], [int(x) for x in z_cal[side]["labels"]])
                for side in ("pre", "post")}
    if cal_meta["pre"] != cal_meta["post"] or any(meta_sub["pre"][p] != meta_sub["post"][p] for p in ("eval", "pool4d")):
        refuse("v1 calibration, eval or pool4d: the pre and post caches hold other items or labels")
    y_tr = z_tr["labels"].astype(bool)
    if sr["sel"]["model"].get("n_probe_train") != len(y_tr):
        refuse(f"probe_train holds {len(y_tr)} items, {sr['sel_rel']} says {sr['sel']['model'].get('n_probe_train')}")

    # the draws (shared by every layer, target and side)
    draws = {LEVELS[0]: P.repartition_draws(n_cal, n_ev, n_draws, P2_DRAW_SEED),
             LEVELS[1]: P.cluster_repartition_draws(unit, n_cal, n_draws, P2_CLUSTER_DRAW_SEED)}
    parts = make_parts(draws, n_cal, n_cal + n_ev)
    y_cal_v1 = np.asarray(cal_meta["pre"][1]) == 1
    if y_cal_v1.all():
        refuse("v1 calibration holds no negative")
    f16 = None
    if (key, pooling) == FLOAT16_CELL:
        p16 = SP.P2 / F16.OUT_NAME
        if p16.is_symlink() or not p16.is_file():
            refuse(f"no {SP.rel(p16)}: the cross-check of CHOICES (o) needs D-364's record")
        f16 = read_result(p16)

    cells: dict[float, dict[int, dict[str, Any]]] = {t: {} for t in TARGETS}
    cols: dict[float, dict[int, dict[str, np.ndarray]]] = {t: {} for t in TARGETS}
    gathered: dict[str, Any] = {}
    probe_anchor: dict[str, Any] = {}
    cross: dict[str, Any] = {"applies": False, "cell": list(FLOAT16_CELL)}
    for li in [L_sel] + [x for x in layers if x != L_sel]:
        k = f"layer_{li}"
        X_tr = z_tr[k]
        probes = fit_probes(X_tr, y_tr, seeds, C_sel)
        if li == L_sel:                                   # step 4: the probes are the selection's (CHOICES (b))
            sel_probes = T.seed_probes(X_tr, y_tr, arm, seeds, holdout)
            off = [q.diag.get("C") for q, s in zip(probes, sel_probes)
                   if not (q.mu.dtype == s["mu"].dtype and q.mu.tobytes() == s["mu"].tobytes()
                           and q.sd.dtype == s["sd"].dtype and q.sd.tobytes() == s["sd"].tobytes()
                           and np.asarray(q.w).dtype == np.asarray(s["w"]).dtype
                           and np.asarray(q.w).tobytes() == np.asarray(s["w"]).tobytes() and q.b == s["b"])]
            if off or len(sel_probes) != len(probes):
                raise SystemExit(f"ANCHOR FAILED: {len(off)} of {len(probes)} probes are not the selection's own fit "
                                 f"bit for bit (selection_tpr_v2.seed_probes); nothing written")
            tprs = [r["tpr"] for r in T.tprs_at(sel_probes, z_cal["pre"][k], ~y_cal_v1, sr["target_fpr"])]
            diffs, n_cmp = T.anchor_diffs(tprs, sr["selected"])
            if diffs:
                for d in diffs:
                    print(f"  DIFFERENT {d}")
                raise SystemExit(f"ANCHOR FAILED: {len(diffs)} of {n_cmp} values are not equal to {sr['sel_rel']}; "
                                 "nothing written")
            probe_anchor = {"selection_record": {"file": sr["sel_rel"], "sha256": _RESULT_READS[sr["sel_rel"]]},
                            "layer": li, "probes_equal_to_selection_fit_bit_for_bit": True,
                            "ho20_tpr_per_seed_recorded": sr["selected"]["ho20_tpr_per_seed"],
                            "ho20_tpr_per_seed_recomputed": tprs, "values_compared": n_cmp, "values_differing": 0,
                            "exact_equal": True,
                            "threshold_population": "v1 calibration's negatives (the selection's)"}
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
                refuse(f"{side} layer {li}: v2 items per v1 split (calibration {cc}, eval negatives {cn}, eval positives "
                       f"{cp}) are not the v2 record's plan")
            mats[side] = {"cal": Xc, "evn": Xn, "pos": Xp}
            gathered[side] = {"calibration": cc, "eval_negatives": cn, "eval_positives": cp}
            del sources
        sc = score_matrices(probes, {"cal_pre": mats["pre"]["cal"], "ev_pre": mats["pre"]["evn"],
                                     "cal_post": mats["post"]["cal"], "ev_post": mats["post"]["evn"],
                                     "pos_pre": mats["pre"]["pos"], "pos_post": mats["post"]["pos"]})
        del mats
        meta = {"pair": key, "axis": AXIS, "layer": li, "C": C_sel, "is_selected_layer": li == L_sel}
        for t in TARGETS:
            cells[t][li], cols[t][li] = compute_cell(sc, draws, parts, t, meta, D170_P2, probes, n_ev, CLUSTER_NOTE_P2)
        if f16 is not None and li == L_sel:               # step 7 (CHOICES (o))
            mine = {"pre_repo": pre_repo, "post_repo": post_repo, "pooling": pooling, "layer": L_sel, "arm": arm,
                    "seeds": list(seeds), "n_draws": n_draws, "draw_seed": P2_DRAW_SEED,
                    "pooled_item_ids_sha256": sha256_bytes("\n".join(pool_ids).encode("utf-8")),
                    "n_eval_positives": n_pos}
            d = float16_cross_check(f16, probes, list(seeds), sc,
                                    {t: cells[t][li]["realised"] for t in TARGETS}, mine)
            if d:
                refuse(f"the D-364 cell differs from {F16.OUT_NAME}'s arm A: {d}")
            cross = {"applies": True, "cell": list(FLOAT16_CELL), "file": SP.rel(SP.P2 / F16.OUT_NAME),
                     "sha256": _RESULT_READS[SP.rel(SP.P2 / F16.OUT_NAME)], "identical": True,
                     "compared": "config (repos, pooling, layer, arm, seeds, draws, draw seed), pooled negatives and "
                                 "positives, each seed's probe fingerprint, the six score matrices' sha256, the realised "
                                 "D-156 reports at 1% and 5% (arm A)"}
            print("  FLOAT16 CROSS-CHECK OK: probes, scores and realised reports equal D-364's arm A", flush=True)
        del sc
        print(f"  layer {li:>2} done (1% and 5%, both levels)", flush=True)
    for zz in (z_tr, *z_cal.values(), *(z for d in z_sub.values() for z in d.values())):
        zz.close()

    # step 8: the record
    arrays: dict[str, np.ndarray] = {}
    for t in TARGETS:
        arrays.update(stack_rows([cols[t][li] for li in layers], TARGET_SFX[t]))
    arrays["cell_layer"] = np.array(layers, dtype=np.int64)
    arrays["cell_probe_set"] = np.array(["selected"] * len(layers))
    arrays["cell_is_selected_layer"] = np.array([li == L_sel for li in layers])
    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"]:
        refuse("source changed during the run")
    caches = {f"pre/{AXIS}/probe_train/{pooling}": r_tr, f"pre/{AXIS}/calibration/{pooling}": r_cal["pre"],
              post_cal_key: r_cal["post"]}
    for side in ("pre", "post"):
        for part in ("eval", "pool4d"):
            caches[f"{side}/{T.SUB_TAG}/{AXIS}/{part}/{pooling}"] = r_sub[side][part]
    # the end of the run: the split files at the v2 record's bytes, each ingest index at its read-time bytes, every cache
    # at the bytes it was read with, and (final) every result read
    files_sha = {T.V2_CAL_FILE: sha256_file(SP.SPLIT_ROOT / T.V2_CAL_FILE),
                 F16.V2_EVAL_FILE: sha256_file(SP.SPLIT_ROOT / F16.V2_EVAL_FILE)}
    files_sha.update({k: sha256_file(p) for k, p in index_read.items()})
    moved = [f"{k} (the v2 record names {w[:16]})" for k, w in ((T.V2_CAL_FILE, v2["sha256"]),
                                                               (F16.V2_EVAL_FILE, ev2["sha256"])) if files_sha[k] != w]
    moved += [k for k, h in index_sha0.items() if files_sha[k] != h]
    moved += end_hash_problems({f"data/hidden/{r['file']}": (SP.CACHE / r["file"], r["sha256"]) for r in caches.values()})
    if final:
        moved += end_hash_problems({k: (resolve(k), h) for k, h in _RESULT_READS.items()})
    if moved:
        refuse(f"changed during the run: {moved}")
    files_sha.update({k: h for k, h in _RESULT_READS.items()})
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    npz_bytes = buf.getvalue()
    config = {"pair": key, "pair_group": group, "pair_note": note,
              "line": "headline (D-361 (2))" if key in HEADLINE_PAIRS else "tokenizer-change line (D-361 (2))",
              "pre_repo": pre_repo, "pre_rev": sr["rev"], "post_repo": post_repo, "post_rev": post_sel["rev"],
              "pooling": pooling, "pooling_role": "primary" if pooling == PRIMARY_POOLING else "beside (D-361 (4))",
              "axis": AXIS, "split_version": 2,
              "selection_record": {"file": sr["sel_rel"], "sha256": _RESULT_READS[sr["sel_rel"]]},
              "selected_arm": arm, "selected_C": C_sel, "selected_layer": L_sel, "probe_set": "selected",
              "n_model_layers": nl, "grid6": g6, "grid6_plus_pre_lstar": required, "layers": layers,
              "layers_beyond_grid6_plus_pre_lstar": beyond, "layer_rule": choice("f"),
              "seeds": list(seeds), "holdout": holdout, "split": f"D-117 (holdout {P.HOLDOUT})",
              "targets": {"primary": PRIMARY_TARGET, "beside": [BESIDE_TARGET]},
              "target_suffix": {f"{t:g}": s for t, s in TARGET_SFX.items()},
              "tail_quantile": TC.TAIL_Q, "sd_eps": TC.SD_EPS, "predictors": list(TC.PREDICTORS),
              "moment_sources": list(TC.SOURCES), "predictor_roles": PREDICTOR_ROLE,
              "n_draws": n_draws, "draw_seed": P2_DRAW_SEED, "cluster_draw_seed": P2_CLUSTER_DRAW_SEED,
              "levels": {LEVELS[0]: "item-level re-partitions (D-156); npz keys without '__cluster'",
                         LEVELS[1]: "cluster-level re-partitions (D-164); npz keys with '__cluster'"},
              "n_eval_positives": n_pos,
              "pooled_negatives": {"n_calibration": n_cal, "n_evaluation": n_ev,
                                   "item_ids_sha256": sha256_bytes("\n".join(pool_ids).encode("utf-8")),
                                   "order": "v2 calibration (its file's order), then v2 evaluation negatives (its "
                                            "file's order)"},
              "positives": "the v2 evaluation positives, fixed (D-156)",
              "estimator": "D-156 as D-361 (2): paired.repartition_estimate and twomoment_cells.one_partition on the "
                           "pooled v2 negatives re-partitioned at the v2 sizes, the frozen v2 split as one more draw",
              "join_rule": v2["join_rule"], "post_select_set": post_sel["kind"], "blas_threads": threads}
    payload = {**provenance_block(RUN_KIND["pair"]), "preregistration": PREREG, "code": code0,
               "registered_parameters": registered, "choices": CHOICES,
               "config": config, "config_hash": canonical_hash(config),
               "anchor": anchor, "probe_anchor": probe_anchor, "float16_cross_check": cross,
               "bos_check": bos, "input_verification": D170_P2,
               "inputs": dict(_RESULT_READS), "input_files_sha256": files_sha,
               "ingest_indexes_sha256": {"at_read": index_sha0, "unchanged_at_end": True},
               "input_cache_sha256": {k: r["sha256"] for k, r in caches.items()},
               "input_cache_source": {k: f"data/hidden/{r['file']} (expected bytes from {r['expected_from']})"
                                      for k, r in caches.items()},
               "caches_rehashed_at_end": True,
               "records": {"pre_select": {"file": sr["rec_rel"], "sha256": _RESULT_READS[sr["rec_rel"]]},
                           "post_select": {"file": post_sel["rel"], "sha256": _RESULT_READS[post_sel["rel"]],
                                           "kind": post_sel["kind"]},
                           "pre_subset": {"file": sb["pre"]["rel"], "sha256": _RESULT_READS[sb["pre"]["rel"]]},
                           "post_subset": {"file": sb["post"]["rel"], "sha256": _RESULT_READS[sb["post"]["rel"]]},
                           "v2": {"file": v2["rel"], "sha256": _RESULT_READS[v2["rel"]],
                                  "excluded_items": v2["excluded_items"]}},
               "layer_plans": plans,
               "v1_keyed_caches": {"rule": "D-320 (4): every cache read is a v1 cache keyed p2wg/<v1 split>, read only "
                                           "through the join rule", "split_files": v1_keyed},
               "split_files": {"v2/calibration": {"file": T.V2_CAL_FILE, "sha256": v2["sha256"]},
                               "v2/eval": {"file": F16.V2_EVAL_FILE, "sha256": ev2["sha256"]}},
               "v2_gather": {"per_side_at_every_layer": gathered,
                             "expected_from_the_v2_plan": {"calibration": v2["plan_counts"],
                                                           "eval_negatives": ev2["expect_neg"],
                                                           "eval_positives": ev2["expect_pos"]}},
               "cluster_units": units_block,
               "draw_arrays": {"file": relpath(npz_path), "sha256": sha256_bytes(npz_bytes), "bytes": len(npz_bytes),
                               "column_0": "the frozen v2 split; columns 1..n_draws: the re-partitions in draw order",
                               "rows": "cells in the order of 'cells' (and of 'cells_fpr5')",
                               "targets": {f"{t:g}": f"keys ending '{TARGET_SFX[t]}'" if TARGET_SFX[t] else
                                           "keys without '__fpr5' (twomoment_cells' names)" for t in TARGETS},
                               "keys": sorted(arrays)},
               "cells": [cells[PRIMARY_TARGET][li] for li in layers],
               "cells_fpr5": [cells[BESIDE_TARGET][li] for li in layers]}
    problems = input_identity_problems(payload)
    if problems:
        refuse(f"to write (D-297 (1) results-writer rule): {problems}")
    json_bytes = json.dumps(payload, indent=2, sort_keys=False, default=str).encode("utf-8")
    json.loads(json_bytes.decode("utf-8"))  # not a result (D-297): this run's own bytes, parsed before staging
    write_outputs([(npz_path, npz_bytes), (out_path, json_bytes)])
    return out_path


# --------------------------------------------------------------------------------------------- the LoRA line's job

def read_lora_sets(repo: str, base_rev: str) -> dict[str, Any]:
    """One adapter's records (no cache is read): its p2tpl select record (the adapter on the base at base_rev, its
    revision the adapter's sha256), its durable p2sub select set (ingested; cut from that record as it is now) and its
    durable rest subset (selection_tpr_v2.read_subset)."""
    om = repo.replace("/", "__")
    found = sorted(SP.P2.glob(f"extract__{om}__{SP.TAG}@*__select.json"))
    if len(found) != 1:
        refuse(f"{len(found)} select records for {repo} (want exactly one): extract the adapter first (D-343)")
    tpl_rel = SP.rel(found[0])
    tpl = read_result(found[0])
    rev = tpl.get("revision")
    want = {"repo": repo, "set_name": "select", "cache_revision": rev, "cache_repo": f"{repo}__{SP.TAG}",
            "model_repo": LORA_BASE, "model_revision": base_rev}
    got = {k: tpl.get(k) for k in want}
    if got != want or not isinstance(rev, str) or len(rev) != 64:
        refuse(f"{tpl_rel}: {[f'{k}={got[k]!r}' for k in want if got[k] != want[k]]} (an adapter on {LORA_BASE}@"
               f"{base_rev[:12]}, its revision the adapter's sha256)")
    p = SP.P2 / f"extract__{om}__{T.SUB_TAG}@{rev[:12]}__select.json"
    if p.is_symlink() or not p.is_file():
        refuse(f"no durable select set {SP.rel(p)} for {repo} (D-325, D-365)")
    rec = read_result(p)
    want_s = {"run_kind": "subset_p2", "cache_tag": T.SUB_TAG, "cache_repo": f"{repo}__{T.SUB_TAG}", "repo": repo,
              "revision": rev, "cache_revision": rev, "set_name": "select", "ingest": "yes"}
    got_s = {k: rec.get(k) for k in want_s}
    if got_s != want_s:
        refuse(f"{SP.rel(p)}: {[f'{k}={got_s[k]!r}' for k in want_s if got_s[k] != want_s[k]]}")
    src = rec.get("source_record") or {}
    if src.get("file") != tpl_rel or src.get("sha256") != _RESULT_READS[tpl_rel]:
        refuse(f"{SP.rel(p)} was not cut from {tpl_rel} as it is now")
    ip, idx = SP.ingest_index(p)
    sel = {"kind": "p2sub", "path": p, "rel": SP.rel(p), "rec": rec, "rev": rev, "index_path": ip, "index": idx,
           "source": {"path": found[0], "rel": tpl_rel, "rec": tpl, "rev": rev}}
    return {"repo": repo, "rev": rev, "sel": sel, "rest": T.read_subset(repo, rev)}


def run_lora(pooling: str, n_draws: int = P.N_DRAWS, out_dir: str | Path | None = None,
             records_only: bool = False) -> Path:
    """The LoRA line's job for one pooling (CHOICES (v)-(x)): the base's selection, caches, probes and v2 matrices once;
    then every arm through the same cells as run_pair. Writes twomoment_p2_d361_lora_<pooling>.json and its npz.
    records_only: the record checks without the anchor record, then "RECORDS OK"; no cache is read, nothing written."""
    if pooling not in POOLINGS:
        refuse(f"pooling must be one of {POOLINGS}")
    threads = SP.check_threads()
    code0 = P.code_block(CODE_FILES)
    final_dir = RESULTS.resolve()
    odir = final_dir if out_dir is None else Path(out_dir).resolve()
    final = odir == final_dir
    out_path = odir / f"twomoment_p2_d361_lora_{pooling}.json"
    npz_path = out_path.with_name(f"{out_path.stem}_draws.npz")
    if not final and odir.is_relative_to(RESULTS.resolve()):
        refuse("a test run (--out-dir) must not write under results/; only the final names may")
    if npz_path.exists() and not out_path.exists():
        refuse(f"{npz_path} exists without {out_path.name}: the orphan of a failed write. Move it aside, never delete "
               f"it, to {npz_path.stem}.orphan-<utc>.npz, and rerun")
    if out_path.exists() or out_path.is_symlink():
        refuse(f"{out_path} exists; results are never overwritten (D-152 P5)")

    # records only
    anchor = None if records_only else passing_anchor(code0, final=final)
    sr = T.read_selection(LORA_BASE, pooling, threads, code0)
    arm, L_sel, seeds, holdout = sr["arm"], sr["layer"], sr["seeds"], sr["holdout"]
    if not arm.startswith("C=") or sr["target_fpr"] != PRIMARY_TARGET or holdout != P.HOLDOUT:
        refuse(f"{sr['sel_rel']}: arm {arm}, target {sr['target_fpr']!r}, holdout {holdout!r}")
    C_sel = float(arm.split("=")[1])
    registered = n_draws == REGISTERED["n_draws"] and list(seeds) == REGISTERED["seeds"]
    v2 = T.read_v2(code0)
    ev2 = F16.read_v2_eval(v2)
    if final:
        bad = final_problems(registered)
        have = {"v2/calibration": v2["sha256"], "v2/eval": ev2["sha256"]}
        bad += [f"{k} is not D-362's ({have[k][:16]})" for k in have if have[k] != PINNED[k]]
        if bad:
            refuse(f"the final names need: {bad}")
    bos = bos_check({"base": (LORA_BASE, sr["rev"])})    # CHOICES (y): the adapters use the base's tokenizer
    index_pre = SP.ingest_index(sr["rec_path"])
    sb_pre = T.read_subset(LORA_BASE, sr["rev"])
    plans: dict[str, Any] = {"base_all_pre_entries": T.check_l_star(sb_pre, sr, pooling)}
    entries: dict[str, dict[str, Any]] = {}
    for part in ("probe_train", "calibration"):
        e = (sr["rec"].get("splits") or {}).get(f"{AXIS}/{part}/{pooling}")
        if not isinstance(e, dict):
            refuse(f"{sr['rec_rel']} has no split {AXIS}/{part}/{pooling}")
        entries[f"pre/{AXIS}/{part}/{pooling}"] = e
    s_pre = {part: T.subset_entry(sb_pre, part, pooling, L_sel) for part in ("eval", "pool4d")}
    for part in ("eval", "pool4d"):
        entries[f"pre/{T.SUB_TAG}/{AXIS}/{part}/{pooling}"] = s_pre[part]
    base_held = set(sr["rec"].get("layers_resolved") or [])
    for part in ("eval", "pool4d"):
        base_held &= set(s_pre[part].get("layers_resolved") or [])
    sets: dict[str, dict[str, Any]] = {}
    layer_sets: dict[str, list[int]] = {}
    for repo in LORA_ARMS:
        st = read_lora_sets(repo, sr["rev"])
        name = repo.split("/", 1)[1]
        plans[name] = {"rest": check_pair_plan(st["rest"], sr, pooling, [LORA_BASE, repo], "post"),
                       "select": check_pair_plan(st["sel"], sr, pooling, [LORA_BASE, repo], "post")}
        st["s_sub"] = {part: T.subset_entry(st["rest"], part, pooling, L_sel) for part in ("eval", "pool4d")}
        sel_sb = {**st["sel"], "rest": st["sel"]["source"]["rec"], "rest_rel": st["sel"]["source"]["rel"]}
        st["s_cal"] = T.subset_entry(sel_sb, "calibration", pooling, L_sel)
        for part, e in (("calibration", st["s_cal"]), ("eval", st["s_sub"]["eval"]), ("pool4d", st["s_sub"]["pool4d"])):
            entries[f"post/{name}/{T.SUB_TAG}/{AXIS}/{part}/{pooling}"] = e
        held = base_held & set(st["s_cal"].get("layers_resolved") or [])
        for part in ("eval", "pool4d"):
            held &= set(st["s_sub"][part].get("layers_resolved") or [])
        if L_sel not in held:
            refuse(f"{repo}: L* {L_sel} is not among the layers both sides hold {sorted(held)}")
        layer_sets[repo] = sorted(held)
        sets[repo] = st
    if len({tuple(v) for v in layer_sets.values()}) != 1:
        refuse(f"the arms hold different layer sets: {layer_sets}")
    layers = layer_sets[LORA_ARMS[0]]
    v1_keyed = F16.check_v1_keyed(entries, v2)
    # the ingest indexes this job reads, hashed when read (each must hash the same at the end)
    index_read = {SP.rel(SP.INGEST_INDEX / f"{p.stem}.sha256"): SP.INGEST_INDEX / f"{p.stem}.sha256"
                  for p in (sr["rec_path"], sb_pre["path"], *(sets[r]["sel"]["path"] for r in LORA_ARMS),
                            *(sets[r]["rest"]["path"] for r in LORA_ARMS))}
    index_sha0 = {k: sha256_file(p) for k, p in index_read.items()}
    neg_ids = [i for i, lab in zip(ev2["ids"], ev2["labels"]) if lab == 0]
    pool_ids = list(v2["ids"]) + neg_ids
    n_cal, n_ev, n_pos = v2["n"], ev2["n_neg"], ev2["n_pos"]
    if final and (n_cal, n_ev, n_pos) != V2_SIZES:
        refuse(f"split v2 holds {(n_cal, n_ev, n_pos)} calibration, evaluation-negative and positive items, not D-361 "
               f"(2)'s {V2_SIZES}")
    unit, units_block = p2_cluster_units(pool_ids, n_cal, v2)
    if records_only:                                      # no cache has been read
        print(f"RECORDS OK: lora {pooling} (records only; no cache read, nothing written)")
        return out_path
    print(f"D-361 LoRA line [{AXIS}, {pooling}]: {len(LORA_ARMS)} arms on {LORA_BASE}; {arm} at L* {L_sel} (from "
          f"{sr['sel_rel']}); layers {layers}; v2 pooled negatives {n_cal} + {n_ev}, {n_pos} positives; {n_draws} draws "
          f"per level; anchor {anchor['file']}; registered {registered}", flush=True)

    # the base, once
    z_tr, r_tr, _ = SP.load_ingested(sr["rec"], sr["rec_rel"], index_pre[1], "probe_train", pooling)
    z_cal_pre, r_cal_pre, _ = SP.load_ingested(sr["rec"], sr["rec_rel"], index_pre[1], "calibration", pooling)
    recorded = sr["sel"].get("input_cache_sha256") if isinstance(sr["sel"].get("input_cache_sha256"), dict) else {}
    for part, r in (("probe_train", r_tr), ("calibration", r_cal_pre)):
        if recorded.get(f"pre/{AXIS}/{part}/{pooling}") != r["sha256"]:
            refuse(f"{r['file']} ({r['sha256'][:16]}) is not the cache {sr['sel_rel']} read")
    z_pre, r_pre, meta_pre = {}, {}, {}
    for part in ("eval", "pool4d"):
        zz, ids, labs, r = open_sub_cache(sb_pre, s_pre[part], part, pooling, layers)
        z_pre[part], r_pre[part], meta_pre[part] = zz, r, (ids, labs)
    cal_meta = ([str(x) for x in z_cal_pre["item_ids"]], [int(x) for x in z_cal_pre["labels"]])
    y_tr = z_tr["labels"].astype(bool)
    if sr["sel"]["model"].get("n_probe_train") != len(y_tr):
        refuse(f"probe_train holds {len(y_tr)} items, {sr['sel_rel']} says {sr['sel']['model'].get('n_probe_train')}")
    y_cal_v1 = np.asarray(cal_meta[1]) == 1
    if y_cal_v1.all():
        refuse("v1 calibration holds no negative")
    draws = {LEVELS[0]: P.repartition_draws(n_cal, n_ev, n_draws, P2_DRAW_SEED),
             LEVELS[1]: P.cluster_repartition_draws(unit, n_cal, n_draws, P2_CLUSTER_DRAW_SEED)}
    parts = make_parts(draws, n_cal, n_cal + n_ev)
    base: dict[int, dict[str, Any]] = {}
    probe_anchor: dict[str, Any] = {}
    for li in layers:
        k = f"layer_{li}"
        X_tr = z_tr[k]
        probes = fit_probes(X_tr, y_tr, seeds, C_sel)
        if li == L_sel:
            sel_probes = T.seed_probes(X_tr, y_tr, arm, seeds, holdout)
            same = len(sel_probes) == len(probes) and all(
                q.mu.dtype == s["mu"].dtype and q.mu.tobytes() == s["mu"].tobytes() and q.sd.dtype == s["sd"].dtype
                and q.sd.tobytes() == s["sd"].tobytes() and np.asarray(q.w).dtype == np.asarray(s["w"]).dtype
                and np.asarray(q.w).tobytes() == np.asarray(s["w"]).tobytes() and q.b == s["b"]
                for q, s in zip(probes, sel_probes))
            if not same:
                raise SystemExit("ANCHOR FAILED: the probes are not the selection's own fit bit for bit; nothing written")
            tprs = [r["tpr"] for r in T.tprs_at(sel_probes, z_cal_pre[k], ~y_cal_v1, sr["target_fpr"])]
            diffs, n_cmp = T.anchor_diffs(tprs, sr["selected"])
            if diffs:
                for d in diffs:
                    print(f"  DIFFERENT {d}")
                raise SystemExit(f"ANCHOR FAILED: {len(diffs)} of {n_cmp} values are not equal to {sr['sel_rel']}; "
                                 "nothing written")
            probe_anchor = {"selection_record": {"file": sr["sel_rel"], "sha256": _RESULT_READS[sr["sel_rel"]]},
                            "layer": li, "probes_equal_to_selection_fit_bit_for_bit": True,
                            "ho20_tpr_per_seed_recorded": sr["selected"]["ho20_tpr_per_seed"],
                            "ho20_tpr_per_seed_recomputed": tprs, "values_compared": n_cmp, "values_differing": 0,
                            "exact_equal": True}
            print(f"  ANCHOR OK at L* {li}: probes bit for bit the selection's; {n_cmp} of {n_cmp} ho20 values equal",
                  flush=True)
        sources = {"calibration": (z_cal_pre[k], *cal_meta), "eval": (z_pre["eval"][k], *meta_pre["eval"]),
                   "pool4d": (z_pre["pool4d"][k], *meta_pre["pool4d"])}
        Xc, Xc_pos, cc, _ = F16.gather_split("calibration", v2["ids"], [0] * v2["n"], sources)
        Xn, Xp, cn, cp = F16.gather_split("eval", ev2["ids"], ev2["labels"], sources)
        if Xc_pos.shape[0] or cc != v2["plan_counts"] or cn != ev2["expect_neg"] or cp != ev2["expect_pos"]:
            refuse(f"base layer {li}: v2 items per v1 split are not the v2 record's plan")
        base[li] = {"probes": probes, "scores": score_matrices(probes, {"cal_pre": Xc, "ev_pre": Xn, "pos_pre": Xp})}
        del sources, Xc, Xn, Xp
    for zz in (z_tr, z_cal_pre, *z_pre.values()):
        zz.close()

    # every arm
    cells: dict[float, list[dict[str, Any]]] = {t: [] for t in TARGETS}
    cols: dict[float, list[dict[str, np.ndarray]]] = {t: [] for t in TARGETS}
    caches: dict[str, dict[str, Any]] = {f"pre/{AXIS}/probe_train/{pooling}": r_tr,
                                         f"pre/{AXIS}/calibration/{pooling}": r_cal_pre,
                                         **{f"pre/{T.SUB_TAG}/{AXIS}/{part}/{pooling}": r_pre[part]
                                            for part in ("eval", "pool4d")}}
    ident: dict[str, dict[str, str]] = {}
    rows: list[dict[str, Any]] = []
    for repo in LORA_ARMS:
        st, name = sets[repo], repo.split("/", 1)[1]
        z_post, r_post, meta_post = {}, {}, {}
        z_post["calibration"], ids_c, labs_c, r_post["calibration"] = open_sub_cache(st["sel"], st["s_cal"],
                                                                                     "calibration", pooling, layers)
        meta_post["calibration"] = (ids_c, labs_c)
        for part in ("eval", "pool4d"):
            zz, ids, labs, r = open_sub_cache(st["rest"], st["s_sub"][part], part, pooling, layers)
            z_post[part], r_post[part], meta_post[part] = zz, r, (ids, labs)
        if meta_post["calibration"] != cal_meta or any(meta_post[p] != meta_pre[p] for p in ("eval", "pool4d")):
            refuse(f"{repo}: its caches hold other items or labels than the base's")
        for part in ("calibration", "eval", "pool4d"):
            key = f"post/{name}/{T.SUB_TAG}/{AXIS}/{part}/{pooling}"
            caches[key] = r_post[part]
            ident[key] = {"repo": repo, "rev": st["rev"]}
        for li in layers:
            k = f"layer_{li}"
            sources = {p: (z_post[p][k], *meta_post[p]) for p in ("calibration", "eval", "pool4d")}
            Xc, Xc_pos, cc, _ = F16.gather_split("calibration", v2["ids"], [0] * v2["n"], sources)
            Xn, Xp, cn, cp = F16.gather_split("eval", ev2["ids"], ev2["labels"], sources)
            if Xc_pos.shape[0] or cc != v2["plan_counts"] or cn != ev2["expect_neg"] or cp != ev2["expect_pos"]:
                refuse(f"{repo} layer {li}: v2 items per v1 split are not the v2 record's plan")
            probes = base[li]["probes"]
            sc = {**base[li]["scores"], **score_matrices(probes, {"cal_post": Xc, "ev_post": Xn, "pos_post": Xp})}
            meta = {"pair": name, "axis": AXIS, "layer": li, "C": C_sel, "is_selected_layer": li == L_sel}
            for t in TARGETS:
                c, cl = compute_cell(sc, draws, parts, t, meta, D170_P2, probes, n_ev, CLUSTER_NOTE_P2)
                cells[t].append(c)
                cols[t].append(cl)
            rows.append({"arm": name, "repo": repo, "rev": st["rev"], "layer": li})
            del sources, Xc, Xn, Xp, sc
        for zz in z_post.values():
            zz.close()
        print(f"  {name} done ({len(layers)} layer(s), 1% and 5%, both levels)", flush=True)

    arrays: dict[str, np.ndarray] = {}
    for t in TARGETS:
        arrays.update(stack_rows(cols[t], TARGET_SFX[t]))
    arrays["cell_layer"] = np.array([r["layer"] for r in rows], dtype=np.int64)
    arrays["cell_arm"] = np.array([r["arm"] for r in rows])
    arrays["cell_is_selected_layer"] = np.array([r["layer"] == L_sel for r in rows])
    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"]:
        refuse("source changed during the run")
    # the end of the run (as run_pair): split files, ingest indexes, every cache, (final) every result read
    files_sha = {T.V2_CAL_FILE: sha256_file(SP.SPLIT_ROOT / T.V2_CAL_FILE),
                 F16.V2_EVAL_FILE: sha256_file(SP.SPLIT_ROOT / F16.V2_EVAL_FILE)}
    files_sha.update({k: sha256_file(p) for k, p in index_read.items()})
    moved = [f"{k} (the v2 record names {w[:16]})" for k, w in ((T.V2_CAL_FILE, v2["sha256"]),
                                                               (F16.V2_EVAL_FILE, ev2["sha256"])) if files_sha[k] != w]
    moved += [k for k, h in index_sha0.items() if files_sha[k] != h]
    moved += end_hash_problems({f"data/hidden/{r['file']}": (SP.CACHE / r["file"], r["sha256"]) for r in caches.values()})
    if final:
        moved += end_hash_problems({k: (resolve(k), h) for k, h in _RESULT_READS.items()})
    if moved:
        refuse(f"changed during the run: {moved}")
    files_sha.update(dict(_RESULT_READS))
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    npz_bytes = buf.getvalue()
    config = {"line": "the LoRA line (D-361 (2), (4)): beside, outside the headline", "pooling": pooling,
              "pooling_role": "primary" if pooling == PRIMARY_POOLING else "beside (D-361 (4))",
              "pre_repo": LORA_BASE, "pre_rev": sr["rev"], "arms": [r.split("/", 1)[1] for r in LORA_ARMS],
              "arm_revisions": {r.split("/", 1)[1]: sets[r]["rev"] for r in LORA_ARMS},
              "arm_rule": choice("v"), "layer_rule": choice("w"), "cluster_rule": choice("x"),
              "axis": AXIS, "split_version": 2,
              "selection_record": {"file": sr["sel_rel"], "sha256": _RESULT_READS[sr["sel_rel"]]},
              "selected_arm": arm, "selected_C": C_sel, "selected_layer": L_sel, "probe_set": "selected",
              "layers": layers, "seeds": list(seeds), "holdout": holdout, "split": f"D-117 (holdout {P.HOLDOUT})",
              "targets": {"primary": PRIMARY_TARGET, "beside": [BESIDE_TARGET]},
              "target_suffix": {f"{t:g}": s for t, s in TARGET_SFX.items()},
              "tail_quantile": TC.TAIL_Q, "sd_eps": TC.SD_EPS, "predictors": list(TC.PREDICTORS),
              "moment_sources": list(TC.SOURCES), "predictor_roles": PREDICTOR_ROLE,
              "n_draws": n_draws, "draw_seed": P2_DRAW_SEED, "cluster_draw_seed": P2_CLUSTER_DRAW_SEED,
              "n_eval_positives": n_pos,
              "pooled_negatives": {"n_calibration": n_cal, "n_evaluation": n_ev,
                                   "item_ids_sha256": sha256_bytes("\n".join(pool_ids).encode("utf-8")),
                                   "order": "v2 calibration (its file's order), then v2 evaluation negatives (its "
                                            "file's order)"},
              "join_rule": v2["join_rule"], "blas_threads": threads}
    payload = {**provenance_block(RUN_KIND["lora"]), "preregistration": PREREG, "code": code0,
               "registered_parameters": registered, "choices": CHOICES,
               "config": config, "config_hash": canonical_hash(config),
               "anchor": anchor, "probe_anchor": probe_anchor, "bos_check": bos, "input_verification": D170_P2,
               "inputs": dict(_RESULT_READS), "input_files_sha256": files_sha,
               "ingest_indexes_sha256": {"at_read": index_sha0, "unchanged_at_end": True},
               "input_identity": ident,
               "input_cache_sha256": {k: r["sha256"] for k, r in caches.items()},
               "input_cache_source": {k: f"data/hidden/{r['file']} (expected bytes from {r['expected_from']})"
                                      for k, r in caches.items()},
               "caches_rehashed_at_end": True,
               "layer_plans": plans,
               "v1_keyed_caches": {"rule": "D-320 (4)", "split_files": v1_keyed},
               "split_files": {"v2/calibration": {"file": T.V2_CAL_FILE, "sha256": v2["sha256"]},
                               "v2/eval": {"file": F16.V2_EVAL_FILE, "sha256": ev2["sha256"]}},
               "cluster_units": units_block,
               "draw_arrays": {"file": relpath(npz_path), "sha256": sha256_bytes(npz_bytes), "bytes": len(npz_bytes),
                               "column_0": "the frozen v2 split; columns 1..n_draws: the re-partitions in draw order",
                               "rows": "cells in the order of 'cells' (and of 'cells_fpr5'): arm by arm, layer by layer",
                               "keys": sorted(arrays)},
               "rows": rows, "cells": cells[PRIMARY_TARGET], "cells_fpr5": cells[BESIDE_TARGET]}
    problems = input_identity_problems(payload)
    if problems:
        refuse(f"to write (D-297 (1) results-writer rule): {problems}")
    json_bytes = json.dumps(payload, indent=2, sort_keys=False, default=str).encode("utf-8")
    json.loads(json_bytes.decode("utf-8"))  # not a result (D-297): this run's own bytes, parsed before staging
    write_outputs([(npz_path, npz_bytes), (out_path, json_bytes)])
    return out_path


# --------------------------------------------------------------------------------------------- the scoring

def pair_cluster_fit(x: np.ndarray, y: np.ndarray, pair_idx: np.ndarray, n_boot: int = BOOT["n_boot"],
                     seed: int = BOOT["boot_seed"]) -> dict[str, Any]:
    """twomoment_fit's headline estimator: OLS of y on x (twomoment_fit.wols) over every cell, with the pair-cluster
    bootstrap interval (B replicates of the pairs present, multiplicities drawn as twomoment_fit draws them; with every
    pair present this is its pair_bootstrap exactly). Cells with a non-finite x or y are left out by wols and counted."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pk = sorted(set(int(p) for p in pair_idx))
    local = np.array([pk.index(int(p)) for p in pair_idx], dtype=int)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pk), size=(n_boot, len(pk)))
    mult = np.stack([np.bincount(r, minlength=len(pk)) for r in idx])
    keep = np.ones(len(x), bool)
    pt = TF.wols(x[:, None], y[:, None], keep[:, None].astype(float))
    Wb = (mult[:, local] * keep[None, :]).T
    rep = TF.wols(np.repeat(x[:, None], n_boot, 1), np.repeat(y[:, None], n_boot, 1), Wb)
    return {"n_cells_used": int(pt["n"][0]), "n_cells_kept": int(keep.sum()),
            "n_cells_kept_but_undefined": int((keep & ~(np.isfinite(x) & np.isfinite(y))).sum()),
            "point": {k: (None if np.isnan(pt[k][0]) else float(pt[k][0])) for k in ("slope", "intercept", "pearson_r")},
            **{f"{k}_ci95": P.ci(rep[k])["ci95"] for k in ("slope", "intercept", "pearson_r")},
            "n_replicates_undefined": int(np.isnan(rep["slope"]).sum()),
            "n_pairs_bootstrapped": len(pk), "n_boot": n_boot, "boot_seed": seed}


def accuracy_line(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """D-177's accuracy line over the cells: e = |log realised - log predicted|; a cell whose e is not finite (a zero
    or undefined prediction) is a miss (D-322 (4)): outside x1.5 and +inf in the median; the share is over every cell.
    Beside, Phase 1's convention (twomoment_supplement: defined cells only)."""
    e = np.abs(np.asarray(y, dtype=float) - np.asarray(x, dtype=float))
    fin = np.isfinite(e)
    n = int(e.size)
    within = fin & (np.where(fin, e, np.inf) <= np.log(FACTOR))
    med = float(np.median(np.where(fin, e, np.inf))) if n else None
    ed = e[fin]
    return {"n_cells": n, "n_cells_miss_zero_or_undefined": int((~fin).sum()),
            "share_within_factor_1_5": float(within.sum() / n) if n else None,
            "median_abs_log_realised_over_predicted": med if med is not None and math.isfinite(med) else None,
            "median_is_a_miss": bool(n and med is not None and not math.isfinite(med)),
            "convention": "every cell; a zero or undefined prediction is a miss (D-322 (4))",
            "defined_cells_only_phase1_convention": {
                "n_cells": int(ed.size),
                "share_within_factor_1_5": float((ed <= np.log(FACTOR)).mean()) if ed.size else None,
                "median_abs_log_realised_over_predicted": float(np.median(ed)) if ed.size else None}}


def within_draw_mean(Xd: np.ndarray, Yd: np.ndarray) -> np.ndarray:
    """Per draw (column), the mean offset log realised - log predicted over the cells (twomoment_supplement's)."""
    E = Yd - Xd
    ok = np.isfinite(E)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok.sum(axis=0) > 0, np.where(ok, E, 0.0).sum(axis=0) / ok.sum(axis=0), np.nan)


def offset_row(x: np.ndarray, y: np.ndarray, Xd: np.ndarray, Yd: np.ndarray) -> dict[str, Any]:
    """One pair's D-177 (2a) row: mean offset over its cells, its D-156 partition band (no pair-cluster interval for
    one pair), the accuracy line and the OLS point over its cells (no interval)."""
    e = y - x
    fin = np.isfinite(e)
    v = within_draw_mean(Xd, Yd)
    pt = TF.wols(np.asarray(x, float)[:, None], np.asarray(y, float)[:, None], np.ones((len(x), 1)))
    return {"n_cells": int(fin.sum()), "n_cells_undefined": int((~fin).sum()),
            "mean_log_realised_over_predicted": float(e[fin].mean()) if fin.any() else None,
            "partition_band_D156": {**P.band_summary(v[1:], v[0]),
                                    "note": "95% band over the draws of the within-draw mean offset over the pair's "
                                            "cells, with the frozen value; a partition sensitivity only (D-166, D-177)"},
            "accuracy": accuracy_line(x, y),
            "ols_point_over_its_cells": {k: (None if np.isnan(pt[k][0]) else float(pt[k][0]))
                                         for k in ("slope", "intercept", "pearson_r")},
            "interval": "none: one pair cannot be resampled at the pair level (D-177)"}


def three_predictions(A: dict[str, np.ndarray], pair_idx: np.ndarray, sel: np.ndarray, sfx: str) -> dict[str, Any]:
    """The quantities of P2-M.1 to P2-M.3 over the cells `sel` of one line (sfx: the level and target suffix)."""
    x2, y, _ = TF.d156_estimates(A[f"log_pred_ratio__{HEADLINE_PRED}{sfx}"][sel], A[f"log_real_ratio{sfx}"][sel])
    x1, y1, _ = TF.d156_estimates(A[f"log_pred_ratio__{COMPARATOR_PRED}{sfx}"][sel], A[f"log_real_ratio{sfx}"][sel])
    fit = pair_cluster_fit(x2, y, pair_idx[sel])
    acc2, acc1 = accuracy_line(x2, y), accuracy_line(x1, y1)
    return {"n_cells": int(sel.sum()), "n_pairs": len(set(pair_idx[sel].tolist())),
            "P2-M.1_fit": fit, "P2-M.2_accuracy": acc2,
            "P2-M.3_shares": {"two_moment": acc2["share_within_factor_1_5"], "one_moment": acc1["share_within_factor_1_5"],
                              "one_moment_accuracy": acc1}}


def verdicts(q: dict[str, Any]) -> dict[str, Any]:
    """D-361 (3)'s three verdicts on the headline line's quantities (HELD or FAILED, each with its rule and reason)."""
    ci = q["P2-M.1_fit"]["slope_ci95"]
    m1 = ci is not None and all(isinstance(v, float) and math.isfinite(v) for v in ci) and ci[0] <= 1.0 <= ci[1]
    sh = q["P2-M.2_accuracy"]["share_within_factor_1_5"]
    m2 = sh is not None and sh >= SHARE_MIN
    two, one = q["P2-M.3_shares"]["two_moment"], q["P2-M.3_shares"]["one_moment"]
    m3 = two is not None and one is not None and two > one
    word = {True: "HELD", False: "FAILED"}
    return {"P2-M.1": {"verdict": word[bool(m1)], "rule": "the pair-cluster bootstrap 95% interval of the slope (B = "
                                                          "5000, seed 520260925) includes 1",
                       "slope": q["P2-M.1_fit"]["point"]["slope"], "slope_ci95": ci,
                       "reason": None if ci is not None else "no interval could be formed (FAILED, CHOICES (j))"},
            "P2-M.2": {"verdict": word[bool(m2)], "rule": f"share within x{FACTOR:g} >= {SHARE_MIN:.2f} over every "
                                                          "headline cell, a zero or undefined prediction a miss",
                       "share_within_factor_1_5": sh,
                       "median_abs_log_realised_over_predicted":
                           q["P2-M.2_accuracy"]["median_abs_log_realised_over_predicted"],
                       "n_cells": q["P2-M.2_accuracy"]["n_cells"],
                       "n_cells_miss_zero_or_undefined": q["P2-M.2_accuracy"]["n_cells_miss_zero_or_undefined"]},
            "P2-M.3": {"verdict": word[bool(m3)], "rule": f"{HEADLINE_PRED}'s share within x{FACTOR:g} strictly larger "
                                                          f"than {COMPARATOR_PRED}'s over the same cells",
                       "two_moment_share": two, "one_moment_share": one}}


BESIDE_KEYS = {"P2-M.1_fit": "fit", "P2-M.2_accuracy": "accuracy", "P2-M.3_shares": "two_vs_one_moment_shares"}


def line_block(A: dict[str, np.ndarray], pair_idx: np.ndarray, sel: np.ndarray, sfx: str, label: str,
               all_predictors: bool = False) -> dict[str, Any]:
    """A beside line: the three quantities (no verdict; the P2-M names are the headline's only, so here they are fit,
    accuracy and two_vs_one_moment_shares), its D-170 line and, optionally, every predictor's fit and accuracy line."""
    q = three_predictions(A, pair_idx, sel, sfx)
    out = {"role": "beside: not a verdict (D-361 (4))", "label": label, "d170": d170_line(int(sel.sum())),
           **{BESIDE_KEYS.get(k, k): v for k, v in q.items()}}
    if all_predictors:
        out["per_predictor"] = {}
        for p in PRED_KEYS:
            x, y, _ = TF.d156_estimates(A[f"log_pred_ratio__{p}{sfx}"][sel], A[f"log_real_ratio{sfx}"][sel])
            out["per_predictor"][p] = {"role": PREDICTOR_ROLE[p], "fit": pair_cluster_fit(x, y, pair_idx[sel]),
                                       "accuracy": accuracy_line(x, y)}
    return out


# --------------------------------------------------------------------------------------------- the summary anchor

def phase1_summary_anchor() -> dict[str, Any]:
    """CHOICES (l): the scoring functions above on the nine Phase 1 records reproduce twomoment_fit_r2.json's headline,
    twomoment_supplement_r4.json's all-cells accuracy lines (moment_full_in_sample, translation_in_sample) and its
    per-pair offset rows exactly at both levels (D-305's re-issues, bound through its pinned manifest); refuses
    otherwise."""
    fit_rel, sup_rel = FIT_REL, SUP_REL
    listed, man_sha = reissue_manifest()
    fit, fit_sha = read_listed(ROOT / fit_rel, fit_rel, listed, REISSUE_MANIFEST)
    sup, sup_sha = read_listed(ROOT / sup_rel, sup_rel, listed, REISSUE_MANIFEST)
    have = {fit_rel: fit_sha, sup_rel: sup_sha}
    names = list(fit.get("population", {}).get("pairs") or [])
    if len(names) != 9:
        refuse(f"{fit_rel} names {len(names)} pairs, not nine")
    arrays: dict[str, list[np.ndarray]] = {}
    pair_of: list[int] = []
    for i, s in enumerate(names):
        rel = f"results/twomoment_{s}.json"
        rec = read_result(ROOT / rel)
        if _RESULT_READS[SP.rel(ROOT / rel)] != fit["input_files_sha256"].get(rel):
            refuse(f"{rel} is not the record {fit_rel} read")
        da = rec["draw_arrays"]
        if fit["input_files_sha256"].get(da["file"]) != da["sha256"]:
            refuse(f"{da['file']} is not the npz {fit_rel} read")
        z = read_npz_bytes(resolve(da["file"]), da["sha256"], da["bytes"], rel)
        sel = np.array([str(p) == "selected" for p in z["cell_probe_set"]])
        if [int(v) for v in z["cell_layer"]] != [c["layer"] for c in rec["cells"]]:
            refuse(f"{da['file']} rows do not match the cells of {s}")
        for k, v in z.items():
            if not k.startswith("cell_"):
                arrays.setdefault(k, []).append(v[sel])
        pair_of += [i] * int(sel.sum())
    A = {k: np.vstack(v) for k, v in arrays.items()}
    pidx = np.array(pair_of)
    allc = np.ones(len(pidx), bool)
    n_cmp, diffs = 0, []
    fit_keys = ("n_cells_used", "n_cells_kept", "n_cells_kept_but_undefined", "point", "slope_ci95", "intercept_ci95",
                "pearson_r_ci95", "n_replicates_undefined")
    acc_keys = ("n_cells", "n_cells_undefined", "median_abs_log_realised_over_predicted", "share_within_factor_1_5")
    ref_fit = {LEVELS[0]: fit["headline"]["result"],
               LEVELS[1]: fit["headline"]["cluster_level_sensitivity_D164"]["result"]}
    for lvl, sfx in LEVEL_SFX.items():
        x, y, _ = TF.d156_estimates(A[f"log_pred_ratio__{HEADLINE_PRED}{sfx}"][allc], A[f"log_real_ratio{sfx}"][allc])
        mine = pair_cluster_fit(x, y, pidx)
        for k in fit_keys:
            n_cmp += 1
            if canon(mine[k]) != canon(ref_fit[lvl][k]):
                diffs.append(f"{lvl} fit {k}: {mine[k]!r} vs {ref_fit[lvl][k]!r}")
        for p in (HEADLINE_PRED, COMPARATOR_PRED):
            xp, yp, _ = TF.d156_estimates(A[f"log_pred_ratio__{p}{sfx}"], A[f"log_real_ratio{sfx}"])
            a = accuracy_line(xp, yp)
            got = {"n_cells": a["defined_cells_only_phase1_convention"]["n_cells"],
                   "n_cells_undefined": a["n_cells_miss_zero_or_undefined"],
                   "median_abs_log_realised_over_predicted":
                       a["defined_cells_only_phase1_convention"]["median_abs_log_realised_over_predicted"],
                   "share_within_factor_1_5": a["defined_cells_only_phase1_convention"]["share_within_factor_1_5"]}
            want = sup["results"][lvl]["prediction_accuracy"][p]["all_cells"]
            for k in acc_keys:
                n_cmp += 1
                if canon(got[k]) != canon(want.get(k)):
                    diffs.append(f"{lvl} accuracy {p} {k}: {got[k]!r} vs {want.get(k)!r}")
            if a["n_cells_miss_zero_or_undefined"] == 0:       # both conventions coincide: the miss form too
                n_cmp += 2
                if a["share_within_factor_1_5"] != got["share_within_factor_1_5"] or \
                        a["median_abs_log_realised_over_predicted"] != got["median_abs_log_realised_over_predicted"]:
                    diffs.append(f"{lvl} accuracy {p}: the miss convention differs with no miss")
        # the per-pair offset rows (D-177 (2a)) of the headline predictor: offset_row against the supplement's per_pair
        Xd, Yd = A[f"log_pred_ratio__{HEADLINE_PRED}{sfx}"], A[f"log_real_ratio{sfx}"]
        want_pp = sup["results"][lvl]["pair_offsets"]["per_pair"]
        for i, s in enumerate(names):
            m = pidx == i
            row = offset_row(x[m], y[m], Xd[m], Yd[m])
            ref = want_pp.get(s) or {}
            for k in ("n_cells", "n_cells_undefined", "mean_log_realised_over_predicted"):
                n_cmp += 1
                if canon(row[k]) != canon(ref.get(k)):
                    diffs.append(f"{lvl} offset {s} {k}: {row[k]!r} vs {ref.get(k)!r}")
            n_cmp += 1
            band = {k: v for k, v in row["partition_band_D156"].items() if k != "note"}
            if canon(band) != canon({k: v for k, v in (ref.get("partition_band_D156") or {}).items() if k != "note"}):
                diffs.append(f"{lvl} offset {s} partition_band_D156")
    if diffs:
        for d in diffs:
            print(f"  DIFFERENT {d}")
        raise SystemExit(f"ANCHOR FAILED: {len(diffs)} of {n_cmp} Phase 1 values not reproduced; nothing written")
    print(f"  SUMMARY ANCHOR OK: {n_cmp} of {n_cmp} Phase 1 values reproduced (twomoment_fit_r2 headline, "
          "twomoment_supplement_r4 accuracy lines and per-pair offsets; both levels)", flush=True)
    return {"exact_equal": True, "values_compared": n_cmp, "values_differing": 0,
            "fit": {"file": fit_rel, "sha256": have[fit_rel]},
            "supplement": {"file": sup_rel, "sha256": have[sup_rel]},
            "bound_by": {"file": REISSUE_MANIFEST, "sha256": man_sha,
                         "note": "D-305's re-issues of twomoment_fit.json and twomoment_supplement_r3.json (SUPERSEDED, "
                                 "populations.D305_REISSUES): the same values, repo IDs named"},
            "pairs": names, "n_cells": int(len(pidx)),
            "compared": f"fit headline {list(fit_keys)} and accuracy {list(acc_keys)} for {HEADLINE_PRED} and "
                        f"{COMPARATOR_PRED}; per-pair offsets of {HEADLINE_PRED} (n_cells, n_cells_undefined, "
                        "mean_log_realised_over_predicted, partition_band_D156 without its note); item and cluster "
                        "level"}


# --------------------------------------------------------------------------------------------- the summary

def pair_record_problems(key: str, pooling: str, doc: dict[str, Any], final: bool) -> list[str]:
    c = doc.get("config") or {}
    bad = []
    if doc.get("run_kind") != RUN_KIND["pair"] or c.get("pair") != key or c.get("pooling") != pooling:
        bad.append(f"not the D-361 record of {key} / {pooling}")
    if doc.get("config_hash") != canonical_hash(c):
        bad.append("its config_hash is not its config's")
    if final and doc.get("registered_parameters") is not True:
        bad.append("not a registered run")
    if c.get("predictor_roles") != PREDICTOR_ROLE:
        bad.append("other predictor roles than twomoment_cells.PREDICTOR_ROLE")
    if (c.get("pre_repo"), c.get("post_repo")) != PAIRS[key][:2]:
        bad.append("other repos than D-361's pair")
    if any((doc.get("probe_anchor") or {}).get(k) is not v for k, v in (("exact_equal", True),
                                                                       ("probes_equal_to_selection_fit_bit_for_bit", True))):
        bad.append("its probe anchor did not pass")
    fx = doc.get("float16_cross_check") or {}
    if ((key, pooling) == FLOAT16_CELL) != (fx.get("applies") is True) or (fx.get("applies") and fx.get("identical") is not True):
        bad.append("its float16 cross-check is missing or did not pass")
    if any(v != SP.THREADS for v in (c.get("blas_threads") or {}).values()) or not c.get("blas_threads"):
        bad.append("not made at the 6-thread pin")
    layers = list(c.get("layers") or [])
    req = list(c.get("grid6_plus_pre_lstar") or [])
    if not req or not set(req) <= set(layers) or sorted(set(layers) - set(req)) != c.get(
            "layers_beyond_grid6_plus_pre_lstar") or c.get("selected_layer") not in req:
        bad.append("its layers are not GRID6 + L* plus its listed layers beyond (CHOICES (f))")
    flagged = d170_flagged([*(doc.get("cells") or []), *(doc.get("cells_fpr5") or [])])
    if flagged:
        bad.append(f"D-170: {flagged}")
    return bad


def d170_flagged(cells: list[Any]) -> str | None:
    """CHOICES (z), twomoment_fit's rule: why a cell list may not enter the summary (None when every cell's
    input_verification flag is None; a cell without the key fails closed)."""
    no_key = sum(1 for c in cells if not isinstance(c, dict) or not isinstance(c.get("input_verification"), dict)
                 or "flag" not in c["input_verification"])
    n = sum(1 for c in cells if isinstance(c, dict) and isinstance(c.get("input_verification"), dict)
            and c["input_verification"].get("flag") is not None)
    if no_key or n:
        return f"{n} of {len(cells)} cells flagged, {no_key} without an input_verification flag"
    return None


def run_summary(records_dir: str | Path | None = None, out: str | Path | None = None) -> Path:
    code0 = P.code_block(CODE_FILES)
    rdir = Path(records_dir).resolve() if records_dir else RESULTS.resolve()
    final_path = (RESULTS / "twomoment_p2_d361_summary.json").resolve()
    out_path = final_path if out is None else Path(out).resolve()
    final = out_path == final_path
    if out_path.exists() or out_path.is_symlink():
        refuse(f"{out_path} exists; results are never overwritten (D-152 P5)")
    if final and rdir != RESULTS.resolve():
        refuse("the final name reads only the records in results/")
    if not final and out_path.is_relative_to(RESULTS.resolve()):
        refuse("a test output (--out) must not be under results/")
    if final:
        bad = final_problems(True, (REISSUE_MANIFEST, D354_SUMMARY_REL))
        if bad:
            refuse(f"the final name needs: {bad}")
    if rdir.name != "results":
        refuse(f"the records directory {rdir} is not a results/ directory (its parent holds {RUN_MANIFEST})")
    want = [(k, p) for k in PAIRS for p in POOLINGS]
    missing = [f"twomoment_p2_d361_{k}_{p}.json" for k, p in want if not out_paths(k, p, rdir)[0].is_file()]
    if missing:
        refuse(f"per-pair records missing in {rdir}: {missing}")
    # CHOICES (z): every record and npz is bound to its line of the list's manifest, which lists exactly the 45 files
    man_path = rdir.parent / RUN_MANIFEST
    listed, man_sha = read_manifest(man_path)
    expected = run_files()
    if sorted(listed) != sorted(expected):
        refuse(f"{RUN_MANIFEST} does not list exactly the {len(expected)} files of the list (missing "
               f"{sorted(set(expected) - set(listed))}, other {sorted(set(listed) - set(expected))})")
    anchor = passing_anchor(code0, rdir, final=final)
    an_name = f"results/twomoment_p2_d361_anchor_{ANCHOR_STEM}.json"
    if listed[an_name] != anchor["sha256"]:
        refuse(f"{an_name} is not the file {RUN_MANIFEST} names")
    s_anchor = phase1_summary_anchor()
    recs: dict[tuple[str, str], dict[str, Any]] = {}
    files: dict[str, tuple[Path, str]] = {RUN_MANIFEST: (man_path, man_sha)}
    problems: dict[str, list[str]] = {}

    def bound_npz(jp: Path, da: dict[str, Any], npz_default: Path) -> tuple[Path, dict[str, np.ndarray]]:
        """A record's npz: its bytes must be the record's and its manifest line's."""
        npz_path = resolve(da["file"]) if isinstance(da.get("file"), str) else npz_default
        if listed.get(f"results/{npz_default.name}") != da.get("sha256"):
            refuse(f"results/{npz_default.name} is not the npz {RUN_MANIFEST} and {jp.name} name")
        return npz_path, read_npz_bytes(npz_path, da.get("sha256"), da.get("bytes"), jp.name)

    for k, p in want:
        jp, npz_default = out_paths(k, p, rdir)
        doc, _ = read_listed(jp, f"results/{jp.name}", listed, RUN_MANIFEST)
        da = doc.get("draw_arrays") or {}
        npz_path, z = bound_npz(jp, da, npz_default)
        if npz_path.resolve() != npz_default.resolve():
            problems.setdefault(f"{k}/{p}", []).append(f"its npz is {da.get('file')!r}, not {npz_default.name} beside it")
        if sorted(z) != sorted(da.get("keys") or []):
            problems.setdefault(f"{k}/{p}", []).append("its npz does not hold the keys it lists")
        b = pair_record_problems(k, p, doc, final)
        an = doc.get("anchor") or {}
        if (an.get("file"), an.get("sha256")) != (anchor["file"], anchor["sha256"]):
            b.append("it names another anchor record")
        layers = list((doc.get("config") or {}).get("layers") or [])
        if [int(v) for v in z.get("cell_layer", [])] != layers or [c["layer"] for c in doc.get("cells") or []] != layers \
                or [c["layer"] for c in doc.get("cells_fpr5") or []] != layers:
            b.append("its cells, cells_fpr5 and npz rows are not its layers")
        if b:
            problems[f"{k}/{p}"] = problems.get(f"{k}/{p}", []) + b
        recs[(k, p)] = {"doc": doc, "z": z, "file": SP.rel(jp), "sha256": _RESULT_READS[SP.rel(jp)]}
        files[SP.rel(jp)] = (jp, _RESULT_READS[SP.rel(jp)])
        files[relpath(npz_path)] = (npz_path, da.get("sha256"))
    lora: dict[str, dict[str, Any]] = {}
    arm_names = [r.split("/", 1)[1] for r in LORA_ARMS]
    for p in POOLINGS:
        jp = rdir / f"twomoment_p2_d361_lora_{p}.json"
        if not jp.is_file():
            problems.setdefault(f"lora/{p}", []).append(f"{jp.name} is missing (the LoRA line's record, D-361 (2))")
            continue
        doc, _ = read_listed(jp, f"results/{jp.name}", listed, RUN_MANIFEST)
        da = doc.get("draw_arrays") or {}
        npz_default = jp.with_name(f"{jp.stem}_draws.npz")
        npz_path, z = bound_npz(jp, da, npz_default)
        c = doc.get("config") or {}
        b = []
        flagged = d170_flagged([*(doc.get("cells") or []), *(doc.get("cells_fpr5") or [])])
        if flagged:
            b.append(f"D-170: {flagged}")
        if npz_path.resolve() != npz_default.resolve() or sorted(z) != sorted(da.get("keys") or []):
            b.append("its npz is not the one beside it with the keys it lists")
        if doc.get("run_kind") != RUN_KIND["lora"] or c.get("pooling") != p or c.get("arms") != arm_names or \
                c.get("pre_repo") != LORA_BASE or doc.get("config_hash") != canonical_hash(c):
            b.append(f"not the LoRA line's record for {p} over the {len(arm_names)} arms")
        if final and doc.get("registered_parameters") is not True:
            b.append("not a registered run")
        if c.get("predictor_roles") != PREDICTOR_ROLE:
            b.append("other predictor roles than twomoment_cells.PREDICTOR_ROLE")
        if (doc.get("probe_anchor") or {}).get("exact_equal") is not True:
            b.append("its probe anchor did not pass")
        an = doc.get("anchor") or {}
        if (an.get("file"), an.get("sha256")) != (anchor["file"], anchor["sha256"]):
            b.append("it names another anchor record")
        rows = doc.get("rows") or []
        if [r.get("arm") for r in rows] != [a for a in arm_names for _ in c.get("layers") or []] or \
                [str(v) for v in z.get("cell_arm", [])] != [r.get("arm") for r in rows] or \
                len(doc.get("cells") or []) != len(rows) or len(doc.get("cells_fpr5") or []) != len(rows):
            b.append("its rows, cells and npz rows are not arm by arm over its layers")
        if b:
            problems[f"lora/{p}"] = b
        lora[p] = {"doc": doc, "z": z, "file": SP.rel(jp), "sha256": _RESULT_READS[SP.rel(jp)]}
        files[SP.rel(jp)] = (jp, _RESULT_READS[SP.rel(jp)])
        files[relpath(npz_path)] = (npz_path, da.get("sha256"))
    if problems:
        refuse(f"not every per-pair record can enter the summary: {problems}")
    ref = recs[want[0]]["doc"]
    same = ("seeds", "n_draws", "draw_seed", "cluster_draw_seed", "targets", "pooled_negatives", "n_eval_positives",
            "tail_quantile", "sd_eps", "predictors", "moment_sources")
    every = {**{f"{k}/{p}": r for (k, p), r in recs.items()}, **{f"lora/{p}": r for p, r in lora.items()}}
    off = {n: [s for s in same if canon(r["doc"]["config"].get(s)) != canon(ref["config"].get(s))]
           for n, r in every.items()}
    off = {n: v for n, v in off.items() if v}
    if (ref["config"].get("draw_seed"), ref["config"].get("cluster_draw_seed")) != (P2_DRAW_SEED, P2_CLUSTER_DRAW_SEED):
        off["seeds of the partitions"] = [ref["config"].get("draw_seed"), ref["config"].get("cluster_draw_seed")]
    codes = {n: r["doc"]["code"].get("sha256") for n, r in every.items()}
    if off or any(c != ref["code"].get("sha256") for c in codes.values()) or \
            any(r["doc"]["code"].get("libraries") != ref["code"].get("libraries") for r in every.values()):
        refuse(f"the per-pair records were not made on one partition sequence and one code block: {off}")
    # minor 6 (code review): one set of per-draw arrays in every record (the LoRA records name their rows by arm)
    key_sets = {n: sorted(r["doc"]["draw_arrays"].get("keys") or []) for n, r in every.items()}
    pair_keys = key_sets[f"{want[0][0]}/{want[0][1]}"]
    lora_keys = [k for k in pair_keys if not k.startswith("cell_")] + ["cell_arm", "cell_is_selected_layer", "cell_layer"]
    odd = sorted(n for n, ks in key_sets.items() if ks != (sorted(lora_keys) if n.startswith("lora/") else pair_keys))
    if odd:
        refuse(f"the records' draw_arrays key sets differ: {odd}")

    def stack(keys: list[str], pooling: str) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict]]:
        arrays: dict[str, list[np.ndarray]] = {}
        pidx, dep, rows = [], [], []
        for i, k in enumerate(keys):
            r = recs[(k, pooling)]
            flagged = d170_flagged([*(r["doc"].get("cells") or []), *(r["doc"].get("cells_fpr5") or [])])
            if flagged:                                   # CHOICES (z), twomoment_fit's rule
                refuse(f"{k}/{pooling}: D-170: {flagged}")
            for name, v in r["z"].items():
                if not name.startswith("cell_"):
                    arrays.setdefault(name, []).append(v)
            L = r["doc"]["config"]["selected_layer"]
            for li in r["doc"]["config"]["layers"]:
                pidx.append(i)
                dep.append(li == L)
                rows.append({"pair": k, "pooling": pooling, "layer": li, "deployer": li == L})
        return {n: np.vstack(v) for n, v in arrays.items()}, np.array(pidx), np.array(dep), rows

    head_keys = list(HEADLINE_PAIRS)
    lines: dict[str, Any] = {}
    A, pidx, dep, rows = stack(head_keys, PRIMARY_POOLING)
    allc = np.ones(len(pidx), bool)
    sfx_item_1 = LEVEL_SFX[LEVELS[0]] + TARGET_SFX[PRIMARY_TARGET]
    head = {"d170": d170_line(len(pidx)), **three_predictions(A, pidx, allc, sfx_item_1)}
    ver = verdicts(head)
    for v in ver.values():
        v["d170"] = head["d170"]
    x2, y2, nd = TF.d156_estimates(A[f"log_pred_ratio__{HEADLINE_PRED}{sfx_item_1}"],
                                   A[f"log_real_ratio{sfx_item_1}"])
    cell_rows = [{**r, "log_predicted_d156": None if not np.isfinite(a) else float(a),
                  "log_realised_d156": None if not np.isfinite(b) else float(b), "n_draws_in_estimate": int(n),
                  "abs_log_realised_over_predicted": (None if not np.isfinite(b - a) else float(abs(b - a)))}
                 for r, a, b, n in zip(rows, x2, y2, nd)]
    lines["beside_every_predictor"] = line_block(A, pidx, allc, sfx_item_1, "every predictor, the headline cells "
                                                 "(out-of-sample and tail variants, translation comparator)", True)
    lines["beside_cluster_level_D164"] = line_block(A, pidx, allc, LEVEL_SFX[LEVELS[1]] + TARGET_SFX[PRIMARY_TARGET],
                                                    "the D-164 cluster level of the headline cells")
    lines["beside_deployer_layer"] = line_block(A, pidx, dep, sfx_item_1, "the deployer's layer: L* cells only")
    lines["beside_5pct_D248"] = line_block(A, pidx, allc, LEVEL_SFX[LEVELS[0]] + TARGET_SFX[BESIDE_TARGET],
                                           "the 5% target (D-248)")
    lines["beside_per_pair"] = {"role": "beside: not a verdict (D-361 (4)); D-177 (2a)", "pairs": {}}
    for i, k in enumerate(head_keys):
        m = pidx == i
        xk, yk, _ = TF.d156_estimates(A[f"log_pred_ratio__{HEADLINE_PRED}{sfx_item_1}"][m],
                                      A[f"log_real_ratio{sfx_item_1}"][m])
        lines["beside_per_pair"]["pairs"][k] = {
            "pre": PAIRS[k][0], "post": PAIRS[k][1], "group": PAIRS[k][2], "note": PAIRS[k][3],
            "d170": d170_line(int(m.sum())),
            **offset_row(xk, yk, A[f"log_pred_ratio__{HEADLINE_PRED}{sfx_item_1}"][m], A[f"log_real_ratio{sfx_item_1}"][m])}
    A_lg, pidx_lg, dep_lg, _ = stack(head_keys, "lastgen")
    lines["beside_last_token"] = line_block(A_lg, pidx_lg, np.ones(len(pidx_lg), bool), sfx_item_1,
                                            "last-token pooling (D-318 (7) item 2: secondary), the nine pairs")
    tok = {}
    for k in TOKENIZER_PAIRS:
        tok[k] = {"pre": PAIRS[k][0], "post": PAIRS[k][1], "note": PAIRS[k][3]}
        for pooling in POOLINGS:
            At, _, _, _ = stack([k], pooling)
            xt, yt, _ = TF.d156_estimates(At[f"log_pred_ratio__{HEADLINE_PRED}{sfx_item_1}"],
                                          At[f"log_real_ratio{sfx_item_1}"])
            x1t, y1t, _ = TF.d156_estimates(At[f"log_pred_ratio__{COMPARATOR_PRED}{sfx_item_1}"],
                                            At[f"log_real_ratio{sfx_item_1}"])
            tok[k][pooling] = {"role": "beside: its own line, outside the headline (D-361 (2), (4))",
                               "d170": d170_line(len(xt)),
                               **offset_row(xt, yt, At[f"log_pred_ratio__{HEADLINE_PRED}{sfx_item_1}"],
                                            At[f"log_real_ratio{sfx_item_1}"]),
                               "one_moment_accuracy": accuracy_line(x1t, y1t)}
    lines["beside_tokenizer_change"] = tok
    lora_lines: dict[str, Any] = {"role": "beside: its own line, outside the headline (D-361 (2), (4))",
                                  "base": LORA_BASE, "arms": arm_names, "rules": [choice("v"), choice("w"), choice("x")]}
    for p, r in lora.items():
        Al = {n: v for n, v in r["z"].items() if not n.startswith("cell_")}
        arm_of = np.array([arm_names.index(str(a)) for a in r["z"]["cell_arm"]])
        blk = line_block(Al, arm_of, np.ones(len(arm_of), bool), sfx_item_1,
                         f"the LoRA line, {p}: {len(arm_names)} adapters on {LORA_BASE}, each its own cluster")
        xl, yl, nl_ = TF.d156_estimates(Al[f"log_pred_ratio__{HEADLINE_PRED}{sfx_item_1}"],
                                        Al[f"log_real_ratio{sfx_item_1}"])
        blk["cells"] = [{**row, "log_predicted_d156": None if not np.isfinite(a) else float(a),
                         "log_realised_d156": None if not np.isfinite(b) else float(b), "n_draws_in_estimate": int(n)}
                        for row, a, b, n in zip(r["doc"]["rows"], xl, yl, nl_)]
        lora_lines[p] = blk
    lines["beside_lora"] = lora_lines
    d354 = read_result(ROOT / D354_SUMMARY_REL)
    by_sel = {r.get("selection_record"): r for r in d354.get("records") or []}
    listing = []
    for (k, p), r in sorted(recs.items()):
        s = r["doc"]["config"]["selection_record"]["file"]
        e = by_sel.get(s)
        listing.append({"pair": k, "pooling": p, "selection_record": s,
                        **({kk: e.get(kk) for kk in ("selected_recorded", "selected_grouped", "selected_cell_unchanged",
                                                     "margin_under_grouping", "margin_under_item_split",
                                                     "recorded_runner_up")} if e else {"listed_in_D354": False})})
    lines["beside_d354_listing"] = {"role": "beside (D-354 (3)); the selections of record are used (D-361 (2))",
                                    "summary": {"file": D354_SUMMARY_REL,
                                                "sha256": _RESULT_READS[SP.rel(ROOT / D354_SUMMARY_REL)]},
                                    "rows": listing,
                                    "n_changed": sum(1 for x in listing if x.get("selected_cell_unchanged") is False)}
    code1 = P.code_block(CODE_FILES)
    if code1["sha256"] != code0["sha256"]:
        refuse("source changed during the run")
    changed = end_hash_problems(files)
    if changed:
        refuse(f"inputs changed during the run: {changed}")
    files_sha = {n: h for n, (_, h) in files.items()}
    files_sha.update(dict(_RESULT_READS))
    config = {"headline": {"pairs": head_keys, "pooling": PRIMARY_POOLING, "target_fpr": PRIMARY_TARGET,
                           "level": LEVELS[0], "probe_set": "selected", "predictor": HEADLINE_PRED,
                           "comparator": COMPARATOR_PRED,
                           "cells": "every layer both sides hold (GRID6 + L* at least; any layer beyond it listed per "
                                    "record, CHOICES (f))"},
              "bootstrap": BOOT, "factor": FACTOR, "share_min": SHARE_MIN, "predictor_roles": PREDICTOR_ROLE,
              "d156_cell_estimate": "twomoment_fit.d156_estimates: log of the mean over the draws where realised and "
                                    "predicted are both defined",
              "d170_rule": choice("z"),
              "tokenizer_pairs": list(TOKENIZER_PAIRS), "poolings": list(POOLINGS), "targets": list(TARGETS)}
    payload = {**provenance_block(RUN_KIND["summary"]), "preregistration": PREREG, "code": code0, "choices": CHOICES,
               "config": config, "config_hash": canonical_hash(config),
               "anchor": anchor, "summary_anchor_phase1": s_anchor,
               "run_manifest": {"file": RUN_MANIFEST, "sha256": man_sha, "n_files": len(listed),
                                "rule": "every record and npz read is the bytes of its line (CHOICES (z))"},
               "inputs": {**_RESULT_READS, RUN_MANIFEST: man_sha}, "input_files_sha256": files_sha,
               "records": {f"{k}/{p}": {"file": r["file"], "sha256": r["sha256"],
                                       "draw_arrays": {"file": r["doc"]["draw_arrays"]["file"],
                                                       "sha256": r["doc"]["draw_arrays"]["sha256"]},
                                       "commit": r["doc"]["code"].get("commit"),
                                       "layers": r["doc"]["config"]["layers"],
                                       "layers_beyond_grid6_plus_pre_lstar":
                                           r["doc"]["config"].get("layers_beyond_grid6_plus_pre_lstar"),
                                       "selected_layer": r["doc"]["config"]["selected_layer"],
                                       "selected_C": r["doc"]["config"]["selected_C"]} for (k, p), r in recs.items()},
               "lora_records": {p: {"file": r["file"], "sha256": r["sha256"],
                                    "draw_arrays": {"file": r["doc"]["draw_arrays"]["file"],
                                                    "sha256": r["doc"]["draw_arrays"]["sha256"]},
                                    "commit": r["doc"]["code"].get("commit"), "layers": r["doc"]["config"]["layers"]}
                                for p, r in lora.items()},
               "headline": {"line": "nine pairs, user-span mean, selected C, item level, 1%", **head},
               "verdicts": ver,
               "gate_note": "D-361 (5): the gate is met when D-361 was committed before any Phase 2 evaluation number was "
                            "read and the three predictions are reported against it; a FAILED prediction is a finding "
                            "reported against D-361, not a reason to change the predictor",
               "accuracy_line_D177": {"line": "the headline", "predictor": HEADLINE_PRED,
                                      **head["P2-M.2_accuracy"]},
               "beside": lines, "headline_cells": cell_rows}
    digest = write_results(out_path, payload, exclusive=True)
    for name, v in ver.items():                       # printed only once the record is written
        print(f"{name}: {v['verdict']}", flush=True)
    print(f"Written: {relpath(out_path)}\nsha256:  {digest}")
    return out_path


# --------------------------------------------------------------------------------------------- main

def main() -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="D-361: the Phase 2 mechanism test (anchor, per-pair jobs, summary)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--anchor", action="store_true", help=f"the Phase 1 anchor on {ANCHOR_STEM} (first job)")
    g.add_argument("--pair", choices=list(PAIRS), help="one D-361 pair (with --pooling)")
    g.add_argument("--lora", action="store_true", help="the LoRA line's job (with --pooling)")
    g.add_argument("--summary", action="store_true", help="score P2-M.1 to P2-M.3 from the per-pair records")
    ap.add_argument("--pooling", choices=list(POOLINGS))
    ap.add_argument("--n-draws", type=int, default=P.N_DRAWS, help="a test only; the final names need 1000")
    ap.add_argument("--out", default=None, help="a test: the anchor's or the summary's output outside results/")
    ap.add_argument("--out-dir", default=None, help="a test: the pair job's output directory outside results/")
    ap.add_argument("--records-dir", default=None, help="a test: the summary's per-pair records directory")
    ap.add_argument("--records-only", action="store_true",
                    help="with --anchor, --pair or --lora: check the records up to the first cache read, print RECORDS "
                         "OK or the refusal, read no cache, write nothing")
    args = ap.parse_args()
    if (args.pair or args.lora) and not args.pooling:
        refuse("--pair and --lora need --pooling")
    if not (args.pair or args.lora) and args.pooling:
        refuse("--pooling goes with --pair or --lora")
    if args.summary and args.records_only:
        refuse("--records-only goes with --anchor, --pair or --lora (the summary reads records only)")
    if args.anchor:
        if args.out_dir or args.records_dir:
            refuse("--anchor takes --out (a test) only")
        if args.out:
            refuse("a test anchor needs explicit records; call run_anchor() from a test")
        run_anchor(args.n_draws, None, records_only=args.records_only)
    elif args.pair:
        if args.out or args.records_dir:
            refuse("--pair takes --out-dir (a test) only")
        run_pair(args.pair, args.pooling, args.n_draws, args.out_dir, records_only=args.records_only)
    elif args.lora:
        if args.out or args.records_dir:
            refuse("--lora takes --out-dir (a test) only")
        run_lora(args.pooling, args.n_draws, args.out_dir, records_only=args.records_only)
    else:
        if args.out_dir or args.n_draws != P.N_DRAWS:
            refuse("--summary takes --out and --records-dir (a test) only")
        run_summary(args.records_dir, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
