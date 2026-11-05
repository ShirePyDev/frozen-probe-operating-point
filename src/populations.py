"""Which results files constitute which claim. Declared, never globbed (F-038).

src/synthesis.py used to aggregate every file matching results/repair_*.json. That
glob meant "the nine update pairs" when it was written; by 2026-09-19 it swept in a
GPU-change control, three safety-LoRA doses, three PKU doses and BOTH copies of a
duplicated experiment, and printed an impossible "103.2% of the gap closed".

So each population is an explicit list. Adding an experiment to results/ cannot change
a headline number unless someone edits this file, which is the point. SUPERSEDED holds
files that are real measurements but must never enter an aggregate, each with the reason.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# The primary evidence: real released update pairs, safety axis unless noted.
# 9 pairs over 6 families - the population behind "1.154 -> 0.377".
# DO NOT quote "1.2x to 68.9x" from here: D-113 retracted that range. Its 1.2x came from
# the sparse depth_mistral_v01_v02 grid that D-062 had already corrected (dense: 3.7x),
# and its 68.9x from depth_mistral_base2instruct_CHAT, an arm variant of a pair already
# counted. Corrected, over the 10 declared pairs: worst per-pair 3.7x to 59.2x, and
# 0.02x to 59.2x across all layers - two-sided, so the low side must be reported too.
UPDATE_PAIRS_REPAIR = [
    "repair_mistral_base2instruct",      # Mistral v0.1 -> Instruct v0.1
    "repair_gemma2_base2instruct",       # Gemma-2 9b -> 9b-it
    "repair_llama3_base2instruct",       # Llama-3 -> Instruct
    "repair_llama31_base2instruct",      # Llama-3.1 -> Instruct
    "repair_qwen_base2instruct",         # Qwen2.5 -> Instruct
    "repair_olmo_base2instruct",         # OLMo-2 -> Instruct
    "repair_llama3_to_31",               # REAL version increment
    "repair_WG_mistral_base2instruct",   # second dataset: WildGuardMix
    "repair_WG_llama3_base2instruct",    # second dataset: WildGuardMix
]

# Depth curves add one pair that has no repair run (Mistral v0.1 -> v0.2, dense grid).
UPDATE_PAIRS_DEPTH = [
    "depth_mistral_base2instruct", "depth_gemma2_base2instruct", "depth_llama3_base2instruct",
    "depth_llama31_base2instruct", "depth_qwen_base2instruct", "depth_olmo_base2instruct",
    "depth_llama3_to_31", "depth_WG_mistral_base2instruct", "depth_WG_llama3_base2instruct",
    "depth_mistral_v01_v02_dense",
]

# Intervention decomposition: 7 pairs, 5 matched-depth layers each.
UPDATE_PAIRS_MOMENT = [
    "moment_mistral_base2instruct_v2", "moment_gemma2_base2instruct", "moment_llama3_base2instruct",
    "moment_llama31_base2instruct", "moment_qwen_base2instruct", "moment_olmo_base2instruct",
    "moment_llama3_to_31",
]

# Alignment variants (D-093/D-094): the same 7 pairs.
UPDATE_PAIRS_ALIGNVAR = [
    "alignvar_mistral_base2instruct_v2", "alignvar_gemma2_base2instruct",
    "alignvar_llama3_base2instruct", "alignvar_llama31_base2instruct",
    "alignvar_qwen_base2instruct", "alignvar_olmo_base2instruct", "alignvar_llama3_to_31",
]

# The controlled dose series. Identical base, rank, targets, lr, dose and example count;
# only the training corpus differs. All three are hardware-matched (A100 both sides).
LORA_ARMS = {
    "general (alpaca)": ["loraA100_mistral_s25", "loraA100_mistral_s100", "loraA100_mistral_s400"],
    "safety (hh-harmless)": ["lorasafe_mistral_s25", "lorasafe_mistral_s100", "lorasafe_mistral_s400"],
    "safety (pku-saferlhf)": ["lorapku_mistral_s25", "lorapku_mistral_s100", "lorapku_mistral_s400"],
}

# The D-152 analysis queue on the selected probe. One results file per item; each names
# the population it was computed over (for 4a: LORA_ARMS above).
ANALYSIS_QUEUE = {
    "4a intent decomposition, selected probe (D-152, D-154; re-run under D-165, D-216 (6))": "intent_selected_d165",
    "nine-pair table and 4a operating points under the re-partition estimator, with the D-164 "
    "cluster-level sensitivity (D-156, D-165, D-173)": "damage_d156_r2",
    "4b two-moment prediction of paired damage, fits across all layer cells (D-152 4b, D-163, D-164, "
    "D-166, D-174)": "twomoment_fit",
    "4b supplement: the headline fit on the unflagged cells and prediction accuracy, from the 4b "
    "per-pair outputs without a re-run, and the per-pair offsets (D-175, D-176, D-177)": "twomoment_supplement_r3",
}
# 4b per-pair records (D-169 item 3): every layer cell of the nine released pairs, both levels.
TWOMOMENT_PAIRS = [f"twomoment_{s}" for s in UPDATE_PAIRS_REPAIR]
# Depth curves regenerated under D-156 (D-169 item 4, D-180): the ten pairs of UPDATE_PAIRS_DEPTH.
DEPTH_D156_PAIRS = [f"depth_d156_{s[len('depth_'):]}" for s in UPDATE_PAIRS_DEPTH]
# Read from those ten files only: the cross-pair summary (D-184, D-190) and the D-191 additions
# (layer-level correlation and frozen-vs-D-156 table; the 2x-criterion null, D-195).
DEPTH_D156_DERIVED = ["depth_d156_summary", "depth_d156_additions", "depth_d156_twox"]
# D-169 items 10 and 11, the depth parts (D-224): the nine LoRA arms' depth curves under D-156, and the
# GPU-change depth control (a no-update hardware null; never aggregated with update pairs)
DEPTH_D156_LORA = [f"depth_d156_{r}" for runs in LORA_ARMS.values() for r in runs]
DEPTH_D156_CONTROLS = ["depth_d156_hw_v100_a100"]

# Files whose OPERATING-POINT numbers (anything that depends on where the 1% threshold falls:
# realised FPR, paired ratio, ratio to target, TPR at the frozen threshold, the floor) were
# measured on the single frozen split and are superseded by the D-156 re-partition estimator.
# Kept, never deleted; quantities that do not depend on the threshold remain valid.
SUPERSEDED_BY_ESTIMATOR = {
    "intent_selected": "operating points superseded by damage_d156 (D-156); geometry (||d||, cos) "
                       "and the variance ratio remain valid",
    "intent_selected_d165": "operating points superseded by damage_d156 (D-156); geometry (||d||, cos) "
                            "and the variance ratio remain valid (D-216 (6))",
    "paired_damage": "C=1 full-probe_train fit on the frozen split; its deployer's-layer cells are "
                     "superseded by damage_d156's C=1 rows (D-156)",
}

# Controls: these must show NO effect. Never aggregated with update pairs.
CONTROLS = {
    "gpu_change": ["depth_hw_v100_a100", "repair_hw_v100_a100", "hw_v100_vs_a100_mistral"],
}

# Real measurements that must never enter an aggregate, and why.
SUPERSEDED = {
    "twomoment_p2_d361_anchor_repair_WG_mistral_base2instruct_pre_d411": (
        "the D-361 anchor the p2mech (D-399) and D-382 runs bound (sha256 in their records), renamed on 2026-10-08 so "
        "the anchor could be re-run under D-411's licence at its fixed name; the runs verify against these bytes"),
    "repair_lora_mistral_s25": "mixed-hardware run, superseded by loraA100 (D-092)",
    "repair_lora_mistral_s100": "mixed-hardware run, superseded by loraA100 (D-092)",
    "repair_lora_mistral_s400": "mixed-hardware run, superseded by loraA100 (D-092)",
    "depth_lora_mistral_s25": "mixed-hardware run, superseded by loraA100 (D-092)",
    "depth_lora_mistral_s100": "mixed-hardware run, superseded by loraA100 (D-092)",
    "depth_lora_mistral_s400": "mixed-hardware run, superseded by loraA100 (D-092)",
    "moment_lora_mistral_s25": "mixed-hardware run, superseded by loraA100 (D-092)",
    "moment_lora_mistral_s100": "mixed-hardware run, superseded by loraA100 (D-092)",
    "moment_lora_mistral_s400": "mixed-hardware run, superseded by loraA100 (D-092)",
    "overrefusal_lora_mistral_s25": "mixed-hardware run, superseded by loraA100 (D-092)",
    "overrefusal_lora_mistral_s100": "mixed-hardware run, superseded by loraA100 (D-092)",
    "overrefusal_lora_mistral_s400": "mixed-hardware run, superseded by loraA100 (D-092)",
    "preregistration_verdict_lora": "mixed-hardware scoring, superseded by loraA100 (D-092)",
    "moment_mistral_base2instruct": "earlier run missing the align_restd/align_affine arms; _v2 supersedes",
    "depth_mistral_v01_v02": "sparse layer grid understated the damage; the dense grid supersedes (D-062)",
    # D-152 P4: selected-C results fitted on the FULL probe_train are marked SUPERSEDED, not
    # superseded-by-estimator: the fit itself is wrong, so no quantity in them stands.
    "paired_damage_selected": "full-probe_train fit (D-152 P4); superseded by damage_d156 (D-156)",
    "mechanism_selected": "full-probe_train fit (D-152 P4); superseded by the 4b outputs (twomoment_*)",
    "damage_d156": "first D-156 run (superseded selection file and run records as inputs); superseded by "
                   "damage_d156_r2 (D-165 inputs, D-164 cluster level), whose item-level numbers are "
                   "bit-identical to it (D-173)",
    "twomoment_supplement": "first D-175 run, without the per-file and per-row D-170 flag (D-172); superseded "
                            "by twomoment_supplement_r2 (D-176)",
    "twomoment_supplement_r2": "D-175 blocks only; superseded by twomoment_supplement_r3, which must reproduce "
                               "them exactly and adds the per-pair offsets and the accuracy line (D-177)",
}
# D-180 (author 8): the ten depth curves are superseded by their D-156 regeneration, including the
# three script-clean curves that had input hashes but no per-file code hashes (D-169 (8)).
for _s in UPDATE_PAIRS_DEPTH:
    _why = f"superseded by depth_d156_{_s[len('depth_'):]} (D-169 item 4, D-180)"
    SUPERSEDED[_s] = f"{SUPERSEDED[_s]}; {_why}" if _s in SUPERSEDED else _why

# D-161 (author, 2026-09-25): results whose code provenance cannot be verified are superseded;
# no primary result enters the paper unless produced by committed code with recorded per-file
# hashes under D-156, and every primary population is regenerated. The 240 stems are declared
# in src/superseded_provenance.py (generated from results/stale_code_audit.json). Until the
# regenerated runs are declared here, validate() reports every affected group, as it should.
import sys as _sys  # noqa: E402

_sys.path.insert(0, str(Path(__file__).resolve().parent))
from superseded_provenance import CODE_PROVENANCE_UNVERIFIABLE  # noqa: E402

for _stem, _why in CODE_PROVENANCE_UNVERIFIABLE.items():
    SUPERSEDED[_stem] = f"{SUPERSEDED[_stem]}; {_why}" if _stem in SUPERSEDED else _why

# Non-parametric retrieval layer: a different metric, reported separately, never pooled.
NON_PARAMETRIC = ["repair_np_mistral_base2instruct", "repair_np_qwen_base2instruct"]
# Regenerated under D-156 (the author's item 8; D-228, D-230, D-232)
NP_D156 = [f"np_d156_{s[len('repair_np_'):]}_r2" for s in NON_PARAMETRIC]      # D-256 (1), D-266: the F-070 re-run
for _s in NON_PARAMETRIC:
    _old = f"np_d156_{_s[len('repair_np_'):]}"
    _why = (f"superseded by {_old}_r2 (D-256 (1), D-266): its quantile_thr and quantile_thr_benign arms used the biased "
            f"quantile transfer (F-070); every other arm is identical in the _r2 record")
    SUPERSEDED[_old] = f"{SUPERSEDED[_old]}; {_why}" if _old in SUPERSEDED else _why
# The moment family regenerated under D-156 with the floor row and per-draw arrays (the author's item 6; D-230,
# D-235 (8), D-247): the _r2 records, and the pre-registered contrasts read from them only
MOMENT_D156 = [f"moment_d156_{(s[len('moment_'):-3] if s.endswith('_v2') else s[len('moment_'):])}_r2" for s in UPDATE_PAIRS_MOMENT]
MOMENT_D156_DERIVED = ["moment_contrasts_d156_r2"]     # D-257, D-267: the paired floor test, D-117 layer primary
_why = ("superseded by moment_contrasts_d156_r2 (D-256 (2), (3), D-257, D-267): its floor rule (band overlap, D-236) "
        "and its metric as primary (the mean over layers is now the continuity row); its numbers stay as written")
SUPERSEDED["moment_contrasts_d156"] = (f"{SUPERSEDED['moment_contrasts_d156']}; {_why}"
                                       if "moment_contrasts_d156" in SUPERSEDED else _why)
# Item 12's repair part (D-238, D-254): the nine LoRA moment files under D-156 (src/repair_d156.py --family
# moment_ext), each anchored to its depth_d156 file; they supersede the old moment_<run>.json
MOMENT_D156_EXT = [f"moment_d156_{r}" for runs in LORA_ARMS.values() for r in runs]
for _r in (r for runs in LORA_ARMS.values() for r in runs):
    _why = f"superseded by moment_d156_{_r} (D-238, D-254)"
    SUPERSEDED[f"moment_{_r}"] = f"{SUPERSEDED[f'moment_{_r}']}; {_why}" if f"moment_{_r}" in SUPERSEDED else _why
# D-257 (10), D-267: the moment family's cells at the D-117 layer with per-draw arrays (src/repair_d156.py
# --d117-layer), each cell equal to its record of record's; the D-257 contrasts read them
MOMENT_D156_D117 = [f"{s.removesuffix('_r2')}_d117" for s in MOMENT_D156] + [f"{s}_d117" for s in MOMENT_D156_EXT]
# The alignment variants under D-156 with the D-228 maps (the author's item 7; D-218, D-229, D-230, D-249)
ALIGNVAR_D156 = [f"alignvar_d156_{(s[len('alignvar_'):-3] if s.endswith('_v2') else s[len('alignvar_'):])}" for s in UPDATE_PAIRS_ALIGNVAR]
for _core in ("mistral_base2instruct", "llama3_base2instruct", "gemma2_base2instruct"):
    _why = f"superseded by moment_d156_{_core}_r2 (D-235 (8)): cells and pair summary reproduced exactly, plus the floor row"
    SUPERSEDED[f"moment_d156_{_core}"] = f"{SUPERSEDED[f'moment_d156_{_core}']}; {_why}" if f"moment_d156_{_core}" in SUPERSEDED else _why
# F-045 under D-156, the pre models of the nine released repair pairs (the author's item 15; D-239, D-250); the paper
# quotes an F-045 number only from these. Cells are named as the superseded overfit_* files (overfit_d156.cells()).
OVERFIT_CELLS = ["Mistral-7B-v01__safety", "Meta-Llama-3-8B__safety", "Llama-31-8B__safety", "Qwen25-7B__safety",
                 "OLMo-2-1124-7B__safety", "gemma-2-9b__safety", "Mistral-7B-v01__safety_wildguard",
                 "Meta-Llama-3-8B__safety_wildguard"]
OVERFIT_D156 = [f"overfit_d156_{c}" for c in OVERFIT_CELLS]
# D-256 (9), D-259 (4), D-268: the holdout-FPR re-partition null for the same cells (overfit_d156.py --null); each
# record's non-null numbers equal its OVERFIT_D156 companion's
OVERFIT_D156_NULL = [f"overfit_d156_{c}_null" for c in OVERFIT_CELLS]
for _c in OVERFIT_CELLS:
    _why = f"superseded by overfit_d156_{_c} (D-239, D-250)"
    SUPERSEDED[f"overfit_{_c}"] = f"{SUPERSEDED[f'overfit_{_c}']}; {_why}" if f"overfit_{_c}" in SUPERSEDED else _why
# The mean_diff standardisation check under D-156 (the author's item 19; D-220, D-246, D-251): SECONDARY, it enters no
# primary table
MEANDIFF_D156 = [f"meandiff_d156_{s[len('repair_'):]}_r2" for s in UPDATE_PAIRS_REPAIR]
# D-297 (1), D-299: re-issued as _r2 naming every cache input's repo ID and revision; every compared key equals the first
# record (logs/meandiff_d156_r2_20261001.sha256); the first records are SUPERSEDED, never edited
for _s in MEANDIFF_D156:
    _old, _why = _s.removesuffix("_r2"), f"superseded by {_s} (D-297 (1), D-299: the same numbers, repo IDs named)"
    SUPERSEDED[_old] = f"{SUPERSEDED[_old]}; {_why}" if _old in SUPERSEDED else _why
# 4d B, the pool study (D-244). PRIMARY (D-297 (2), D-298): option B (origin-stratified draws) for the seven Aegis pairs
# and the two WildGuardMix records as registered, re-issued with repo IDs (D-299). SENSITIVITY rows beside B (D-297 (2)):
# option C as registered for the seven Aegis pairs, re-issued with repo IDs; the first records are SUPERSEDED.
POOLSTUDY_D156 = ([f"poolstudy_d156_{s[len('repair_'):]}_r2" for s in UPDATE_PAIRS_REPAIR if s.startswith("repair_WG_")]
                  + [f"poolstudy_d156_{s[len('repair_'):]}_originstrat" for s in UPDATE_PAIRS_REPAIR
                     if not s.startswith("repair_WG_")])
POOLSTUDY_D156_C = [f"poolstudy_d156_{s[len('repair_'):]}_r2" for s in UPDATE_PAIRS_REPAIR if not s.startswith("repair_WG_")]
for _s in [x for x in POOLSTUDY_D156 if x.endswith("_r2")] + POOLSTUDY_D156_C:
    _old, _why = _s.removesuffix("_r2"), f"superseded by {_s} (D-297 (1), D-299: the same numbers, repo IDs named)"
    SUPERSEDED[_old] = f"{SUPERSEDED[_old]}; {_why}" if _old in SUPERSEDED else _why

# Over-refusal on XSTest safe prompts. Quoted multipliers are MID-DEPTH layers and the
# effect reverses at the last layer on some families (D-091), so depth must be reported.
OVER_REFUSAL_PAIRS = {
    "overrefusal_mistral": ("Mistral v0.1 -> Instruct", "instruct", 16),
    "overrefusal_gemma2": ("Gemma-2 -> it", "instruct", 20),
    "overrefusal_llama3": ("Llama-3 -> Instruct", "instruct", 16),
    "overrefusal_llama31": ("Llama-3.1 -> Instruct", "instruct", 16),
    "overrefusal_qwen": ("Qwen2.5 -> Instruct (2 depths only, D-095)", "instruct", 14),
    "overrefusal_mistral_v01_v02": ("Mistral v0.1 -> v0.2 [VERSION]", "version", 16),
    "overrefusal_llama3_to_31": ("Llama-3 -> Llama-3.1 [VERSION]", "version", 16),
}

# Adversarial stratification: detection loss split by adversarial flag.
ADVERSARIAL = ["advstrat_mistral", "advstrat_llama3"]
# Regenerated under D-156 (D-169 items 8 and 9; D-186/D-188 and D-187/D-189, D-193, D-194).
# D-282 (3), D-295: the 1% records re-issued with the combined TPR interval replace the first D-156 records
ADVSTRAT_D156 = [f"advstrat_d156_{s[len('advstrat_'):]}_r2" for s in ADVERSARIAL]
for _s in ADVSTRAT_D156:
    _old, _why = _s.removesuffix("_r2"), f"superseded by {_s} (D-282: the same numbers plus the combined TPR interval, D-295)"
    SUPERSEDED[_old] = f"{SUPERSEDED[_old]}; {_why}" if _old in SUPERSEDED else _why
OVERREFUSAL_D156 = [f"overrefusal_d156_{s[len('overrefusal_'):]}" for s in OVER_REFUSAL_PAIRS]
# D-209 at 5% re-issued with D-282's combined TPR interval (D-284 (5), D-293); a sensitivity arm, not Table 1
ADVSTRAT_D156_FPR5 = [f"advstrat_d156_{s[len('advstrat_'):]}_fpr5_r2" for s in ADVERSARIAL]
for _s in ADVSTRAT_D156_FPR5:
    _old, _why = _s.removesuffix("_r2"), f"superseded by {_s} (D-282: the same numbers plus the combined TPR interval, D-293)"
    SUPERSEDED[_old] = f"{SUPERSEDED[_old]}; {_why}" if _old in SUPERSEDED else _why
# D-206 (1)-(2), D-253: the two re-runs on the tagged full-grid XSTest caches (they hold the D-117 layer the partial-grid
# files lack), and the item-9 table at the D-117 layer read from them and the five other files
OVERREFUSAL_D156_XSFULL = ["overrefusal_d156_qwen_xsfull", "overrefusal_d156_mistral_v01_v02_xsfull"]
OVERREFUSAL_D156_DERIVED = ["overrefusal_d117_table"]
# D-256 (4), D-258 (3), D-267: the XSTest hardware check (same probe on the V100 and A100 caches of one model, every
# layer both hold); a control, it enters no primary table, and D-253's v0.2 rise is quotable only through it
OVERREFUSAL_HW_CHECK = ["xstest_hw_check_d256"]
for _old, _new in [*zip(ADVERSARIAL, ADVSTRAT_D156), *zip(OVER_REFUSAL_PAIRS, OVERREFUSAL_D156)]:
    _why = f"superseded by {_new} (D-169 items 8, 9)"
    SUPERSEDED[_old] = f"{SUPERSEDED[_old]}; {_why}" if _old in SUPERSEDED else _why
# D-216 (6): the 4a geometry re-run under D-165 (inputs through the registry, the selection of record,
# the D-170 block); its numbers equal the old file's exactly
_why = ("superseded by intent_selected_d165 (D-216 (6)): its selection came from a superseded sweep and it "
        "carries no input_verification block; every number is identical in the re-run")
SUPERSEDED["intent_selected"] = f"{SUPERSEDED['intent_selected']}; {_why}" if "intent_selected" in SUPERSEDED else _why

# Re-analyses that apply the prior work's own AUC staleness rule to our cells.
# The LoRA audit covers the alpaca and HH-RLHF dose series plus the hardware control
# (7 files x 17 layers = 119 cells). The PKU-SafeRLHF series is NOT in it, so the label
# must not read "LoRA arms", which next to Table 3's three arms implies it is.
AUC_AUDITS = {"released update pairs": "auc_criterion_audit",
              "LoRA (alpaca, HH-RLHF) + GPU control": "auc_criterion_audit_lora"}
# 4c, the label-free health check under D-156 (D-152 4c, D-169 item 12, D-241, D-255): one record per released pair
# (per-draw arrays in its _draws.npz, hashed in the JSON), the GPU-change control apart, and the pooled summary, which
# replaces AUC_AUDITS (D-241 (7))
HEALTHCHECK_D156 = [f"healthcheck_d156_{s[len('repair_'):]}" for s in UPDATE_PAIRS_REPAIR]
HEALTHCHECK_D156_CONTROLS = ["healthcheck_d156_hw_v100_a100"]
HEALTHCHECK_D156_DERIVED = ["healthcheck_d156_summary"]
# 4d part A (D-289): the repair family under D-156, one record per released pair, LoRA run and the GPU control; it
# replaces the repair_* records one to one (UPDATE_PAIRS_REPAIR and the LORA groups retire into these below)
REPAIR_D156 = [f"repair_d156_{s[len('repair_'):]}" for s in UPDATE_PAIRS_REPAIR]
REPAIR_D156_LORA = {arm: [f"repair_d156_{r}" for r in runs] for arm, runs in LORA_ARMS.items()}
REPAIR_D156_CONTROLS = ["repair_d156_hw_v100_a100"]
for _old, _new in [*zip(UPDATE_PAIRS_REPAIR, REPAIR_D156),
                   *((f"repair_{r}", f"repair_d156_{r}") for runs in LORA_ARMS.values() for r in runs),
                   ("repair_hw_v100_a100", "repair_d156_hw_v100_a100")]:
    _why = f"superseded by {_new} (4d part A under D-156, D-289)"
    SUPERSEDED[_old] = f"{SUPERSEDED[_old]}; {_why}" if _old in SUPERSEDED else _why
# item 18, flag-set churn (D-219, D-269, D-291), and 4d part B's source test (D-274, D-283)
CHURN_D156 = [f"churn_d156_{s[len('repair_'):]}" for s in UPDATE_PAIRS_REPAIR]
SOURCE_TEST_4DB = ["source_test_4db"]
# 4c on the nine LoRA runs (D-278, D-295) and its summary, the D-156 replacement of auc_criterion_audit's LoRA part
HEALTHCHECK_D156_LORA = [f"healthcheck_d156_{r}" for runs in LORA_ARMS.values() for r in runs]
HEALTHCHECK_D156_LORA_DERIVED = ["healthcheck_d156_summary_lora"]
# D-363 (the author's D-357 (B)(4), option (a) of D-349; result D-380): the HH contamination sensitivity, every
# contaminated item removed from all splits: its anchor, the nine LoRA runs' records and the summary. Beside the rows
# of record, not primary: the HH rows enter the paper only as the author decides from the summary (D-363 (3)).
HH_SENSITIVITY_D363 = (["hh_sensitivity_d363_anchor_lorasafe_mistral_s400"]
                       + [f"hh_sensitivity_d363_{r}" for runs in LORA_ARMS.values() for r in runs]
                       + ["hh_sensitivity_d363"])
# D-384 (result D-394): the operational case study, two pair records (Llama-3 -> 3.1 and the largest Table I ratio,
# Mistral b->i) and the summary. Descriptive, beside the tables, not primary.
OPCASE_D384 = ["opcase_d384_llama3_to_31", "opcase_d384_mistral_base2instruct", "opcase_d384"]
# D-282 (3), D-287 (6) (result D-396): the combined TPR interval for the SECONDARY mean_diff records (MEANDIFF_D156)
# at 1%, with the logistic rows beside; SECONDARY, it enters no primary table
MEANDIFF_D156_D282 = ["meandiff_tpr_d282"]
# --- declared IN ONE BATCH at the pull of 2026-10-08 (F-080, D-411), before the single D-361 anchor re-run ---
# D-402 as amended by D-405 (result D-406, the author's D-401 (2d)): the cell-level diagnostic of Table IV's
# safety-run specificity drop, declared after its pull as D-380's group was. Its predictions record and its record;
# FLAGGED, a diagnostic reported beside the two readings of record, in no paper table, never primary.
TABLE4_SPECIFICITY_D402 = ["table4_specificity_d402_predictions", "table4_specificity_d402"]
# D-381, D-392, D-398 (result D-412): the C5 health-check baselines (KS, ATC, BBSE beside) on every 4c pair at 1% and
# 5%, and the pooled summary; P-C5.1-3 are scored on the released pairs only, every other row beside
_C5_CORES = ["mistral_base2instruct", "gemma2_base2instruct", "llama3_base2instruct", "llama31_base2instruct",
             "qwen_base2instruct", "olmo_base2instruct", "llama3_to_31", "WG_mistral_base2instruct",
             "WG_llama3_base2instruct", "hw_v100_a100"] + [f"{a}_mistral_s{s}" for a in ("loraA100", "lorasafe", "lorapku")
                                                          for s in (25, 100, 400)]
HEALTHCHECK_BASELINES_D381 = ([f"healthcheck_baselines_d381_{c}{t}" for c in _C5_CORES for t in ("", "_fpr5")]
                              + ["healthcheck_baselines_d381_summary"])
# D-383, D-397 (result D-412): the MLP probe supplement to Phase 1 (C1, C4, C5 beside the linear probe of record):
# its anchor, nine selection records, ten pair records (the nine released pairs and the v0.1 -> v0.2 dense row) and
# the summary. Post-exit supplement (D-401 (b)); it cannot reopen Phase 1.
_MLP_PAIRS = ["mistral_base2instruct", "gemma2_base2instruct", "llama3_base2instruct", "llama31_base2instruct",
              "qwen_base2instruct", "olmo_base2instruct", "llama3_to_31", "WG_mistral_base2instruct",
              "WG_llama3_base2instruct"]
MLP_PROBE_D383 = (["mlp_probe_d383_anchor_qwen_base2instruct"]
                  + [f"mlp_probe_d383_selection_{p}" for p in _MLP_PAIRS]
                  + [f"mlp_probe_d383_{p}" for p in _MLP_PAIRS + ["mistral_v01_v02_dense"]]
                  + ["mlp_probe_d383"])
# D-361, D-399 (result D-412): the Phase 2 exit gate's mechanism test; the anchor (the record re-run under D-411's
# licence; the record the runs bound is SUPERSEDED below), eleven pairs x two poolings and the summary (P2-M.1-3)
_P2_KEYS = ["mistral_v01_v02", "llama3_to_31", "qwen2_to_25", "phi3_128k_to_35", "olmo2_base_sft", "olmo2_sft_dpo",
            "olmo2_dpo_instruct", "tulu3_sft_dpo", "tulu3_dpo_final", "mistral_v02_v03", "lora"]
TWOMOMENT_P2_D361 = (["twomoment_p2_d361_anchor_repair_WG_mistral_base2instruct"]
                     + [f"twomoment_p2_d361_{k}_{p}" for k in _P2_KEYS for p in ("usermean", "lastgen")]
                     + ["twomoment_p2_d361_summary"])
# D-382 (result D-412): the reference-size trade-off: its predictions record, the nine Phase 2 headline pairs
# (primary), the nine Phase 1 released pairs (beside) and the summary (P-N.1-3)
REFSIZE_D382 = (["refsize_d382_predictions"]
                + [f"refsize_d382_p2_{k}" for k in _P2_KEYS if k not in ("mistral_v02_v03", "lora")]
                + [f"refsize_d382_p1_{p}" for p in _MLP_PAIRS]
                + ["refsize_d382_summary"])
# AUC_AUDITS retires into this position-by-position mapping (not per pair, so no core check); its stems are
# declared in their own groups
AUC_AUDITS_D156 = ["healthcheck_d156_summary", "healthcheck_d156_summary_lora"]
# Table 1's TPR columns with D-282's combined interval (D-287, D-295), supplements anchored to the depth records
DEPTH_D156_D282 = ["table1_tpr_d282"]
DEPTH_D156_D282_FPR5 = ["table1_tpr_d282_fpr5"]
# D-300 (5), D-305: the five primary records that named no repo ID or revision for their cache inputs, re-issued (every
# compared key equal to the first record; logs/reissue_d305_20261001.sha256); the first records are SUPERSEDED, never
# edited, and the groups name the re-issues
D305_REISSUES = {"depth_d156_summary": "depth_d156_summary_r2", "depth_d156_twox": "depth_d156_twox_r2",
                 "table1_tpr_d282": "table1_tpr_d282_r2", "twomoment_fit": "twomoment_fit_r2",
                 "twomoment_supplement_r3": "twomoment_supplement_r4"}
for _old, _new in D305_REISSUES.items():
    _why = f"superseded by {_new} (D-300 (5), D-305: the same numbers, repo IDs named)"
    SUPERSEDED[_old] = f"{SUPERSEDED[_old]}; {_why}" if _old in SUPERSEDED else _why
ANALYSIS_QUEUE = {k: D305_REISSUES.get(v, v) for k, v in ANALYSIS_QUEUE.items()}
DEPTH_D156_DERIVED = [D305_REISSUES.get(s, s) for s in DEPTH_D156_DERIVED]
DEPTH_D156_D282 = [D305_REISSUES.get(s, s) for s in DEPTH_D156_D282]
# the GPU-change control under D-156 (D-224, D-289) and its byte comparison re-run with a code block (D-292, D-295)
CONTROL_D156 = {"gpu_change": ["depth_d156_hw_v100_a100", "repair_d156_hw_v100_a100", "hw_v100_vs_a100_mistral_r2"]}
for _old, _new in zip(CONTROLS["gpu_change"], CONTROL_D156["gpu_change"]):
    _why = f"superseded by {_new} (D-295)"
    SUPERSEDED[_old] = f"{SUPERSEDED[_old]}; {_why}" if _old in SUPERSEDED else _why
for _s, _why in (("auc_criterion_audit", "replaced by healthcheck_d156_summary (D-241 (7), D-255)"),
                  # D-259: 4c covers the released pairs and the GPU control only; the LoRA audit has no D-156 replacement
                  ("auc_criterion_audit_lora", "its GPU-control part is covered by healthcheck_d156_summary (D-255); "
                                               "its LoRA part has no D-156 replacement yet (D-259)")):
    SUPERSEDED[_s] = f"{SUPERSEDED[_s]}; {_why}" if _s in SUPERSEDED else _why


def path(stem: str) -> Path:
    return RESULTS / f"{stem}.json"


def primary_result_stems() -> list[str]:
    """Every results stem a declared claim rests on, SUPERSEDED excluded (D-153).

    Used by verify_provenance.py to refuse a non-finite item in any cache these read.
    LoRA arms are declared by run stem; each analysis of that run that exists on disk
    (depth_/repair_/moment_/overrefusal_) counts.
    """
    stems = [*UPDATE_PAIRS_REPAIR, *UPDATE_PAIRS_DEPTH, *UPDATE_PAIRS_MOMENT,
             *UPDATE_PAIRS_ALIGNVAR, *NON_PARAMETRIC, *OVER_REFUSAL_PAIRS, *ADVERSARIAL,
             *ANALYSIS_QUEUE.values(), *TWOMOMENT_PAIRS,   # D-174: the regenerated runs are primary too
             *DEPTH_D156_PAIRS,                             # D-180
             *DEPTH_D156_DERIVED,                           # D-190, D-196
             *DEPTH_D156_LORA, *DEPTH_D156_CONTROLS,        # D-224
             *NP_D156,                                      # D-232
             *MOMENT_D156, *MOMENT_D156_DERIVED,            # D-247
             *MOMENT_D156_EXT,                              # D-254
             *MOMENT_D156_D117, *OVERREFUSAL_HW_CHECK,      # D-267
             *ALIGNVAR_D156,                                # D-249
             *OVERFIT_D156, *MEANDIFF_D156,                 # D-250, D-251 (MEANDIFF_D156 secondary)
             *OVERFIT_D156_NULL,                            # D-268
             *ADVSTRAT_D156, *OVERREFUSAL_D156,             # D-193, D-194
             *OVERREFUSAL_D156_XSFULL, *OVERREFUSAL_D156_DERIVED,  # D-253
             *HEALTHCHECK_D156, *HEALTHCHECK_D156_CONTROLS, *HEALTHCHECK_D156_DERIVED,  # D-255
             *REPAIR_D156, *[s for v in REPAIR_D156_LORA.values() for s in v], *REPAIR_D156_CONTROLS,  # D-289
             *CHURN_D156, *SOURCE_TEST_4DB,                 # D-291, D-283
             *HEALTHCHECK_D156_LORA, *HEALTHCHECK_D156_LORA_DERIVED, *DEPTH_D156_D282,  # D-295
             *[s for v in CONTROL_D156.values() for s in v],           # D-295
             *POOLSTUDY_D156, *POOLSTUDY_D156_C]                       # D-297 (2), D-299, D-306
    for runs in CONTROLS.values():
        stems += runs
    for runs in LORA_ARMS.values():
        for run in runs:
            stems += [f"{kind}_{run}" for kind in ("depth", "repair", "moment", "overrefusal")
                      if path(f"{kind}_{run}").exists()]
    return sorted({s for s in stems if s not in SUPERSEDED})


# D-256 (8), D-259: an old group is retired only when its replacement group is declared and complete (every file on
# disk, none superseded, one replacement per old stem). A retired group leaves validate()'s groups; its replacement
# is checked in its place. REPAIR, LORA and CONTROL wait for 4d; NON_PARAMETRIC for the NP _r2 re-run; AUC_AUDITS
# until its LoRA audit has a D-156 replacement.
RETIRED = {
    "UPDATE_PAIRS_DEPTH": ("DEPTH_D156_PAIRS", "D-180"),
    "UPDATE_PAIRS_MOMENT": ("MOMENT_D156", "D-247"),
    "UPDATE_PAIRS_ALIGNVAR": ("ALIGNVAR_D156", "D-249"),
    "OVER_REFUSAL": ("OVERREFUSAL_D156", "D-194"),
    "ADVERSARIAL": ("ADVSTRAT_D156", "D-193"),
    "NON_PARAMETRIC": ("NP_D156", "D-266"),
    "UPDATE_PAIRS_REPAIR": ("REPAIR_D156", "D-289"),
    **{f"LORA:{arm}": (f"REPAIR_D156_LORA:{arm}", "D-289") for arm in LORA_ARMS},
    **{f"CONTROL:{name}": (f"CONTROL_D156:{name}", "D-295") for name in CONTROL_D156},
    # not per pair: mapped by position, the core check waived, and the replacement is a mapping only (its stems are
    # declared in HEALTHCHECK_D156_DERIVED and HEALTHCHECK_D156_LORA_DERIVED)
    "AUC_AUDITS": ("AUC_AUDITS_D156", "D-295", "by_position"),
}


def retired_mapping() -> dict[str, dict[str, str]]:
    """D-259: old stem -> replacement stem for each retired group (the replacement lists are built from the old ones in
    order, so position gives the mapping; checked one to one by validate())."""
    g = {"UPDATE_PAIRS_DEPTH": UPDATE_PAIRS_DEPTH, "UPDATE_PAIRS_MOMENT": UPDATE_PAIRS_MOMENT,
         "UPDATE_PAIRS_ALIGNVAR": UPDATE_PAIRS_ALIGNVAR, "OVER_REFUSAL": list(OVER_REFUSAL_PAIRS), "ADVERSARIAL": ADVERSARIAL,
         "DEPTH_D156_PAIRS": DEPTH_D156_PAIRS, "MOMENT_D156": MOMENT_D156, "ALIGNVAR_D156": ALIGNVAR_D156,
         "OVERREFUSAL_D156": OVERREFUSAL_D156, "ADVSTRAT_D156": ADVSTRAT_D156,
         "NON_PARAMETRIC": NON_PARAMETRIC, "NP_D156": NP_D156,
         "UPDATE_PAIRS_REPAIR": UPDATE_PAIRS_REPAIR, "REPAIR_D156": REPAIR_D156,
         **{f"LORA:{arm}": [f"repair_{s}" for s in runs] for arm, runs in LORA_ARMS.items()},
         **{f"REPAIR_D156_LORA:{arm}": v for arm, v in REPAIR_D156_LORA.items()},
         **{f"CONTROL:{n}": v for n, v in CONTROLS.items()}, **{f"CONTROL_D156:{n}": v for n, v in CONTROL_D156.items()},
         "AUC_AUDITS": list(AUC_AUDITS.values()), "AUC_AUDITS_D156": AUC_AUDITS_D156}
    return {old: dict(zip(g[old], g[new])) for old, (new, *_) in RETIRED.items()}


def validate() -> list[str]:
    """Every declared file exists, nothing is declared twice, nothing primary is superseded."""
    problems: list[str] = []
    if {"auc_criterion_audit", "auc_criterion_audit_lora"} != set(AUC_AUDITS.values()):
        problems.append("AUC_AUDITS no longer names the two audits whose SUPERSEDED reasons D-259 set")
    groups = {
        "UPDATE_PAIRS_REPAIR": UPDATE_PAIRS_REPAIR, "UPDATE_PAIRS_DEPTH": UPDATE_PAIRS_DEPTH,
        "UPDATE_PAIRS_MOMENT": UPDATE_PAIRS_MOMENT, "UPDATE_PAIRS_ALIGNVAR": UPDATE_PAIRS_ALIGNVAR,
        "NON_PARAMETRIC": NON_PARAMETRIC,
    }
    for arm, stems in LORA_ARMS.items():
        groups[f"LORA:{arm}"] = [f"repair_{s}" for s in stems]
    for name, stems in CONTROLS.items():
        groups[f"CONTROL:{name}"] = stems
    groups["OVER_REFUSAL"] = list(OVER_REFUSAL_PAIRS)
    groups["ADVERSARIAL"] = ADVERSARIAL
    groups["AUC_AUDITS"] = list(AUC_AUDITS.values())
    groups["ANALYSIS_QUEUE"] = list(ANALYSIS_QUEUE.values())
    groups["TWOMOMENT_PAIRS"] = TWOMOMENT_PAIRS
    groups["DEPTH_D156_PAIRS"] = DEPTH_D156_PAIRS                    # D-180
    groups["DEPTH_D156_DERIVED"] = DEPTH_D156_DERIVED                # D-190, D-196
    groups["DEPTH_D156_LORA"] = DEPTH_D156_LORA                      # D-224
    groups["DEPTH_D156_CONTROLS"] = DEPTH_D156_CONTROLS              # D-224
    groups["NP_D156"] = NP_D156                                      # D-232
    groups["MOMENT_D156"] = MOMENT_D156                              # D-247
    groups["MOMENT_D156_DERIVED"] = MOMENT_D156_DERIVED              # D-247
    groups["MOMENT_D156_EXT"] = MOMENT_D156_EXT                      # D-254
    groups["MOMENT_D156_D117"] = MOMENT_D156_D117                    # D-267
    groups["OVERREFUSAL_HW_CHECK"] = OVERREFUSAL_HW_CHECK            # D-267
    groups["ALIGNVAR_D156"] = ALIGNVAR_D156                          # D-249
    groups["OVERFIT_D156"] = OVERFIT_D156                            # D-250
    groups["OVERFIT_D156_NULL"] = OVERFIT_D156_NULL                  # D-268
    groups["MEANDIFF_D156"] = MEANDIFF_D156                          # D-251
    groups["ADVSTRAT_D156"] = ADVSTRAT_D156                          # D-193
    groups["OVERREFUSAL_D156"] = OVERREFUSAL_D156                    # D-194
    groups["OVERREFUSAL_D156_XSFULL"] = OVERREFUSAL_D156_XSFULL      # D-253
    groups["OVERREFUSAL_D156_DERIVED"] = OVERREFUSAL_D156_DERIVED    # D-253
    groups["HEALTHCHECK_D156"] = HEALTHCHECK_D156                    # D-255
    groups["HEALTHCHECK_D156_CONTROLS"] = HEALTHCHECK_D156_CONTROLS  # D-255
    groups["HEALTHCHECK_D156_DERIVED"] = HEALTHCHECK_D156_DERIVED    # D-255
    groups["REPAIR_D156"] = REPAIR_D156                              # D-289
    for arm, stems in REPAIR_D156_LORA.items():
        groups[f"REPAIR_D156_LORA:{arm}"] = stems                    # D-289
    groups["REPAIR_D156_CONTROLS"] = REPAIR_D156_CONTROLS            # D-289
    groups["CHURN_D156"] = CHURN_D156                                # D-291
    groups["SOURCE_TEST_4DB"] = SOURCE_TEST_4DB                      # D-283
    groups["ADVSTRAT_D156_FPR5"] = ADVSTRAT_D156_FPR5                # D-293
    groups["HEALTHCHECK_D156_LORA"] = HEALTHCHECK_D156_LORA          # D-295
    groups["HEALTHCHECK_D156_LORA_DERIVED"] = HEALTHCHECK_D156_LORA_DERIVED  # D-295
    groups["DEPTH_D156_D282"] = DEPTH_D156_D282                      # D-295
    groups["DEPTH_D156_D282_FPR5"] = DEPTH_D156_D282_FPR5            # D-295
    groups["POOLSTUDY_D156"] = POOLSTUDY_D156                        # D-297 (2), D-298, D-299, D-306 (primary)
    groups["POOLSTUDY_D156_C"] = POOLSTUDY_D156_C                    # D-297 (2), D-306 (sensitivity rows)
    groups["HH_SENSITIVITY_D363"] = HH_SENSITIVITY_D363              # D-363, D-380 (beside, not primary)
    groups["OPCASE_D384"] = OPCASE_D384                              # D-384, D-394 (descriptive, beside)
    groups["MEANDIFF_D156_D282"] = MEANDIFF_D156_D282                # D-282 (3), D-396 (secondary)
    groups["TABLE4_SPECIFICITY_D402"] = TABLE4_SPECIFICITY_D402      # D-402, D-405, D-406 (diagnostic, beside)
    groups["HEALTHCHECK_BASELINES_D381"] = HEALTHCHECK_BASELINES_D381  # D-381, D-412
    groups["MLP_PROBE_D383"] = MLP_PROBE_D383                        # D-383, D-412 (post-exit supplement)
    groups["TWOMOMENT_P2_D361"] = TWOMOMENT_P2_D361                  # D-361, D-399, D-412 (Phase 2 exit gate)
    groups["REFSIZE_D382"] = REFSIZE_D382                            # D-382, D-412
    # replacement mappings last, so the double-declaration check below reads them as the CONTROL and mapping
    # groups they are (their stems are declared in their own groups too)
    for name, stems in CONTROL_D156.items():
        groups[f"CONTROL_D156:{name}"] = stems                       # D-295
    groups["AUC_AUDITS_D156"] = AUC_AUDITS_D156                      # D-295 (a mapping only)
    # D-259: a retired group leaves the check only while its replacement is declared and complete
    for old_name, (new_name, entry, *how) in RETIRED.items():
        old_stems, new_stems = groups.get(old_name), groups.get(new_name)
        if old_stems is None or new_stems is None:
            problems.append(f"RETIRED {old_name}: it or its replacement {new_name} is not a declared group")
            continue
        incomplete = [s for s in new_stems if not path(s).exists() or s in SUPERSEDED]
        mp = retired_mapping()[old_name]
        # one to one: each old stem's replacement is its own pair's D-156 stem (the name carries the pair's core)
        core = lambda s: s.split("_", 1)[1].removesuffix("_v2").removesuffix("_r2").removeprefix("d156_").removeprefix("np_")  # noqa: E731
        mism = [] if how == ["by_position"] else [(o, n) for o, n in mp.items() if core(o) != core(n)]
        if mism:
            incomplete = incomplete + [f"{o} -> {n}" for o, n in mism]
        if incomplete or len(new_stems) != len(old_stems):
            problems.append(f"RETIRED {old_name}: its replacement {new_name} ({entry}) is not complete: "
                            f"{len(new_stems)} for {len(old_stems)} stems; incomplete {incomplete}")
            continue
        del groups[old_name]
        if how == ["by_position"]:
            del groups[new_name]           # a mapping only; its stems are checked in their own groups
    for gname, stems in groups.items():
        for s in stems:
            if not path(s).exists():
                problems.append(f"{gname}: declared file missing from disk: {s}.json")
            if s in SUPERSEDED:
                problems.append(f"{gname}: includes a SUPERSEDED file: {s} ({SUPERSEDED[s]})")
    seen: dict[str, str] = {}
    for gname, stems in groups.items():
        for s in stems:
            if s in seen and not gname.startswith("CONTROL"):
                problems.append(f"{s} declared in both {seen[s]} and {gname}")
            seen[s] = gname
    return problems


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from provenance import ensure_utf8_stdout

    ensure_utf8_stdout()
    issues = validate()
    n = (len(UPDATE_PAIRS_REPAIR) + len(UPDATE_PAIRS_DEPTH) + len(UPDATE_PAIRS_MOMENT)
         + len(UPDATE_PAIRS_ALIGNVAR) + len(NON_PARAMETRIC) + len(OVER_REFUSAL_PAIRS)
         + len(ADVERSARIAL) + len(AUC_AUDITS)
         + sum(len(v) for v in LORA_ARMS.values()) + sum(len(v) for v in CONTROLS.values()))
    print(f"declared files: {n}, superseded and excluded: {len(SUPERSEDED)}")
    for _old, _mp in retired_mapping().items():            # D-256 (8): the mapping, listed
        print(f"RETIRED {_old} -> {RETIRED[_old][0]} ({RETIRED[_old][1]}): " + ", ".join(f"{o} -> {n_}" for o, n_ in _mp.items()))
    print("POPULATIONS OK" if not issues else "PROBLEMS:\n  " + "\n  ".join(issues))
    raise SystemExit(1 if issues else 0)
