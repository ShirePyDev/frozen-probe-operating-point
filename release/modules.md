# Which file does what

Generated for this release (PROVENANCE.md lists it). One row per Python file. 'Runs here' is decided by fixed rules, not by running each file: a file that names a path under `logs/` or `queues/` (neither is here, apart from one hash list) or imports a GPU-stage library (torch, transformers, peft, datasets, huggingface_hub, safetensors, tokenizers) is listed as inspection only, with the reason. 'Records' counts the result records here whose code block names the file (as the script that ran or as a module it imported); 'Read caches' counts those of them that read hidden-state caches, which are not released, so those runs cannot be repeated from this repository whatever the file's own status. 'Paper' lists the parts of the paper whose printed numbers read one of those records (release/numbers_map.tsv).

| File | Runs here? | Records | Read caches | Paper |
|---|---|---|---|---|
| `docs/paper/config_fetch_d471.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `docs/paper/config_fetch_d474.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `docs/paper/review_20261016b_calc.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `release/verify_inputs.py` | RUNS HERE: the hash check (README) | 0 | 0 | - |
| `scripts/tally_ledger_verdicts_d449.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/a1m_p3.py` | inspection only here: needs files under logs/ that are not here; job lists under queues/ that are not here; the experiment environment (peft, torch) | 3 | 2 | Supplement, V Results |
| `src/a2_p3.py` | a script with no GPU-stage import; it runs only where its input files are | 3 | 2 | Supplement, V Results |
| `src/a2_verdicts_d440.py` | a script with no GPU-stage import; it runs only where its input files are | 1 | 0 | V Results |
| `src/access_check.py` | inspection only here: needs the experiment environment (huggingface_hub) | 0 | 0 | - |
| `src/adversarial_strat.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/advstrat_d156.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/advstrat_failopen.py` | inspection only here: needs files under logs/ that are not here | 0 | 0 | - |
| `src/align_variants.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/auc_criterion_audit.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/baseline_null.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/bin_weights.py` | inspection only here: needs the experiment environment (safetensors, torch) | 5 | 2 | Supplement, V Results |
| `src/c1_recompute_d645.py` | a script with no GPU-stage import; it runs only where its input files are | 1 | 0 | V Results |
| `src/capcost_p3.py` | inspection only here: needs files under logs/ that are not here; job lists under queues/ that are not here; the experiment environment (torch, transformers) | 4 | 2 | Supplement, V Results |
| `src/cell_offset_d300.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/censoring_sensitivity.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/checkpoints.py` | inspection only here: needs the experiment environment (huggingface_hub) | 0 | 0 | - |
| `src/checks_4a.py` | a script with no GPU-stage import; it runs only where its input files are | 1 | 1 | generated table/figure file |
| `src/churn_d156.py` | inspection only here: needs files under logs/ that are not here | 0 | 0 | - |
| `src/compare_caches.py` | a script with no GPU-stage import; it runs only where its input files are | 1 | 0 | IV Method, Supplement |
| `src/compare_reruns.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/config_table_d472.py` | RUNS HERE: a table/figure writer (README, 'Reproducing the tables and figures') | 0 | 0 | - |
| `src/contamination_robustness.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/d168_rules.py` | imported by other modules | 1 | 0 | V Results |
| `src/damage_d156.py` | a script with no GPU-stage import; it runs only where its input files are | 1 | 0 | V Results |
| `src/data_access.py` | inspection only here: needs the experiment environment (huggingface_hub) | 0 | 0 | - |
| `src/defence_d1_d442.py` | inspection only here: needs files under logs/ that are not here | 2 | 2 | Supplement, V Results |
| `src/defence_d2e_d443.py` | a script with no GPU-stage import; it runs only where its input files are | 1 | 1 | V Results |
| `src/depth_curve.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/depth_d156.py` | a script with no GPU-stage import; it runs only where its input files are | 211 | 202 | Abstract, I Introduction, IV Method, Supplement, V Results, VI Discussion, VII Conclusion, generated table/figure file |
| `src/depth_d156_additions.py` | a script with no GPU-stage import; it runs only where its input files are | 1 | 0 | V Results |
| `src/depth_d156_summary.py` | a script with no GPU-stage import; it runs only where its input files are | 1 | 1 | V Results |
| `src/depth_d156_twox.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/evaluate_lora_preregistration.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/extract_hidden.py` | inspection only here: needs the experiment environment (peft, torch, transformers) | 5 | 2 | Supplement, V Results |
| `src/extract_p2.py` | inspection only here: needs the experiment environment (huggingface_hub, torch, transformers) | 5 | 2 | Supplement, V Results |
| `src/extraction_commit_audit.py` | inspection only here: needs files under logs/ that are not here | 0 | 0 | - |
| `src/fetch_dataset_snapshot.py` | a script with no GPU-stage import; it runs only where its input files are | 4 | 2 | Supplement, V Results |
| `src/fetch_weights.py` | inspection only here: needs the experiment environment (huggingface_hub) | 5 | 2 | Supplement, V Results |
| `src/figure_overrefusal_d156.py` | inspection only here: needs files under logs/ that are not here | 1 | 0 | V Results |
| `src/figures.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/float16_check_d364.py` | a script with no GPU-stage import; it runs only where its input files are | 66 | 62 | Abstract, I Introduction, Supplement, V Results, VI Discussion, VII Conclusion, generated table/figure file |
| `src/fpr5_rules_d156.py` | inspection only here: needs files under logs/ that are not here | 0 | 0 | - |
| `src/freeze_splits.py` | inspection only here: needs the experiment environment (datasets, huggingface_hub) | 7 | 2 | Supplement, V Results |
| `src/healthcheck_baselines_d381.py` | inspection only here: needs files under logs/ that are not here | 42 | 40 | I Introduction, Supplement, V Results |
| `src/healthcheck_d156.py` | a script with no GPU-stage import; it runs only where its input files are | 126 | 118 | Abstract, I Introduction, Supplement, V Results, VI Discussion, VII Conclusion, generated table/figure file |
| `src/healthcheck_roc_d262.py` | inspection only here: needs files under logs/ that are not here | 1 | 0 | Abstract, I Introduction, V Results, VI Discussion |
| `src/healthcheck_summary_d156.py` | inspection only here: needs files under logs/ that are not here | 89 | 79 | Abstract, I Introduction, Supplement, V Results, VI Discussion, VII Conclusion, generated table/figure file |
| `src/hf_env_guard.py` | a script with no GPU-stage import; it runs only where its input files are | 8 | 2 | Supplement, V Results |
| `src/hh_sensitivity_d363.py` | inspection only here: needs files under logs/ that are not here | 1 | 0 | V Results |
| `src/input_registry.py` | imported by other modules | 255 | 243 | Abstract, I Introduction, IV Method, Supplement, V Results, VI Discussion, VII Conclusion, generated table/figure file |
| `src/intent_selected.py` | a script with no GPU-stage import; it runs only where its input files are | 2 | 1 | V Results, generated table/figure file |
| `src/lora_adapter_p2.py` | imported by other modules | 4 | 2 | Supplement, V Results |
| `src/lora_contamination_p1.py` | a script with no GPU-stage import; it runs only where its input files are | 1 | 0 | V Results |
| `src/lora_jobs_p2.py` | inspection only here: needs files under logs/ that are not here; job lists under queues/ that are not here | 4 | 2 | Supplement, V Results |
| `src/lora_step0_check_p2.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/meandiff_d156.py` | inspection only here: needs files under logs/ that are not here | 0 | 0 | - |
| `src/meandiff_tpr_d282.py` | inspection only here: needs files under logs/ that are not here | 0 | 0 | - |
| `src/method_table.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/metrics.py` | imported by other modules | 215 | 202 | Abstract, I Introduction, IV Method, Supplement, V Results, VI Discussion, VII Conclusion, generated table/figure file |
| `src/mlp_probe_d383.py` | inspection only here: needs files under logs/ that are not here | 1 | 0 | V Results |
| `src/moment_contrasts_d156.py` | inspection only here: needs files under logs/ that are not here | 2 | 0 | V Results, VI Discussion |
| `src/moment_contrasts_d257.py` | inspection only here: needs files under logs/ that are not here | 2 | 0 | V Results, VI Discussion |
| `src/moment_decomposition.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/nas_restore_proof.py` | inspection only here: needs files under logs/ that are not here | 5 | 2 | Supplement, V Results |
| `src/negative_clusters.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/noise_floor.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/opcase_d384.py` | inspection only here: needs files under logs/ that are not here | 0 | 0 | - |
| `src/over_refusal_test.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/overfit_d156.py` | inspection only here: needs files under logs/ that are not here | 0 | 0 | - |
| `src/overrefusal_d117_table.py` | inspection only here: needs files under logs/ that are not here | 1 | 0 | V Results |
| `src/overrefusal_d156.py` | a script with no GPU-stage import; it runs only where its input files are | 27 | 27 | V Results, generated table/figure file |
| `src/overrefusal_frozen_rederive.py` | inspection only here: needs files under logs/ that are not here | 0 | 0 | - |
| `src/overrefusal_p2_d440.py` | inspection only here: needs job lists under queues/ that are not here | 18 | 18 | V Results, generated table/figure file |
| `src/p1mode_equivalence.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/p2auroc_d424.py` | inspection only here: needs job lists under queues/ that are not here | 38 | 38 | Abstract, I Introduction, Supplement, V Results, VI Discussion, VII Conclusion, generated table/figure file |
| `src/p2auroc_d424_grouped.py` | inspection only here: needs job lists under queues/ that are not here | 2 | 2 | Supplement |
| `src/paired.py` | imported by other modules | 273 | 246 | Abstract, I Introduction, IV Method, Supplement, V Results, VI Discussion, VII Conclusion, generated table/figure file |
| `src/pairs_table_d468.py` | RUNS HERE: a table/figure writer (README, 'Reproducing the tables and figures') | 0 | 0 | - |
| `src/paraphrase_overlap.py` | inspection only here: needs the experiment environment (datasets, huggingface_hub) | 8 | 2 | Supplement, V Results |
| `src/paraphrase_overlap_trained.py` | inspection only here: needs the experiment environment (datasets) | 0 | 0 | - |
| `src/phase2_bos_check.py` | inspection only here: needs files under logs/ that are not here | 0 | 0 | - |
| `src/phase2_config_check.py` | a script with no GPU-stage import; it runs only where its input files are | 5 | 2 | Supplement, V Results |
| `src/phase2_freeze_splits.py` | a script with no GPU-stage import; it runs only where its input files are | 3 | 0 | - |
| `src/phase2_pool_clusters.py` | a script with no GPU-stage import; it runs only where its input files are | 7 | 2 | Supplement, V Results |
| `src/phase2_resplit_v2.py` | a script with no GPU-stage import; it runs only where its input files are | 67 | 62 | Abstract, I Introduction, Supplement, V Results, VI Discussion, VII Conclusion, generated table/figure file |
| `src/phase2_tokenizer_check.py` | inspection only here: needs the experiment environment (huggingface_hub, tokenizers, transformers) | 5 | 2 | Supplement, V Results |
| `src/phase3_a1.py` | a script with no GPU-stage import; it runs only where its input files are | 15 | 11 | Abstract, I Introduction, V Results, VI Discussion, VII Conclusion |
| `src/phase3_a1_byc_d370.py` | a script with no GPU-stage import; it runs only where its input files are | 1 | 1 | V Results |
| `src/phase3_a1_pooled.py` | a script with no GPU-stage import; it runs only where its input files are | 2 | 1 | V Results |
| `src/phase3_a1_predictions.py` | a script with no GPU-stage import; it runs only where its input files are | 25 | 21 | Abstract, I Introduction, V Results, VI Discussion, VII Conclusion |
| `src/phase3_a3.py` | a script with no GPU-stage import; it runs only where its input files are | 10 | 9 | Abstract, I Introduction, V Results, VII Conclusion |
| `src/phase3_a3_pooled.py` | a script with no GPU-stage import; it runs only where its input files are | 1 | 0 | V Results |
| `src/phase3_a3_predictions.py` | a script with no GPU-stage import; it runs only where its input files are | 21 | 18 | Abstract, I Introduction, V Results, VI Discussion, VII Conclusion |
| `src/phase3_a3_sens.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/pool_propositions.py` | imported by other modules | 38 | 32 | Abstract, I Introduction, Supplement, V Results, VI Discussion, VII Conclusion |
| `src/poolstudy_d156.py` | inspection only here: needs files under logs/ that are not here | 38 | 32 | Abstract, I Introduction, Supplement, V Results, VI Discussion, VII Conclusion |
| `src/populations.py` | a script with no GPU-stage import; it runs only where its input files are | 270 | 246 | Abstract, I Introduction, IV Method, Supplement, V Results, VI Discussion, VII Conclusion, generated table/figure file |
| `src/power_analysis.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/provenance.py` | imported by other modules | 273 | 246 | Abstract, I Introduction, IV Method, Supplement, V Results, VI Discussion, VII Conclusion, generated table/figure file |
| `src/realign_d418.py` | inspection only here: needs files under logs/ that are not here; job lists under queues/ that are not here | 2 | 0 | I Introduction, V Results, VI Discussion |
| `src/realign_d423.py` | inspection only here: needs files under logs/ that are not here; job lists under queues/ that are not here | 1 | 0 | I Introduction, V Results, VI Discussion |
| `src/realign_d423_summary_d439.py` | inspection only here: needs files under logs/ that are not here | 1 | 0 | I Introduction, V Results, VI Discussion |
| `src/recalnoise_d433.py` | inspection only here: needs files under logs/ that are not here; job lists under queues/ that are not here | 0 | 0 | - |
| `src/refsize_d382.py` | inspection only here: needs files under logs/ that are not here; job lists under queues/ that are not here | 40 | 38 | Abstract, I Introduction, Supplement, V Results, VI Discussion, VII Conclusion, generated table/figure file |
| `src/regularisation_sweep.py` | a script with no GPU-stage import; it runs only where its input files are | 68 | 62 | Abstract, I Introduction, Supplement, V Results, VI Discussion, VII Conclusion, generated table/figure file |
| `src/repair_d156.py` | inspection only here: needs files under logs/ that are not here | 30 | 26 | I Introduction, Supplement, V Results, VI Discussion, generated table/figure file |
| `src/repair_nonparametric.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/repair_test.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/repdist.py` | a script with no GPU-stage import; it runs only where its input files are | 211 | 202 | Abstract, I Introduction, IV Method, Supplement, V Results, VI Discussion, VII Conclusion, generated table/figure file |
| `src/report.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/results_figures_d570.py` | RUNS HERE: a table/figure writer (README, 'Reproducing the tables and figures') | 0 | 0 | - |
| `src/results_tables_d504.py` | RUNS HERE: a table/figure writer (README, 'Reproducing the tables and figures') | 0 | 0 | - |
| `src/rotation_test.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/safe_outputs.py` | imported by other modules | 70 | 61 | Abstract, I Introduction, Supplement, V Results, VI Discussion, VII Conclusion, generated table/figure file |
| `src/score_file_export.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/selection_ho20_cluster.py` | a script with no GPU-stage import; it runs only where its input files are | 1 | 0 | Supplement |
| `src/selection_p2.py` | a script with no GPU-stage import; it runs only where its input files are | 67 | 62 | Abstract, I Introduction, Supplement, V Results, VI Discussion, VII Conclusion, generated table/figure file |
| `src/selection_p2sst.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/selection_table.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/selection_tpr_v2.py` | a script with no GPU-stage import; it runs only where its input files are | 66 | 62 | Abstract, I Introduction, Supplement, V Results, VI Discussion, VII Conclusion, generated table/figure file |
| `src/separability_test.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/shift_alignment.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/shift_alignment_summary.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/source_test_4db.py` | inspection only here: needs files under logs/ that are not here | 0 | 0 | - |
| `src/split_diagnostics.py` | a script with no GPU-stage import; it runs only where its input files are | 7 | 2 | Supplement, V Results |
| `src/sst2_containment_d365.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/stale_code_audit.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/subset_p2.py` | a script with no GPU-stage import; it runs only where its input files are | 66 | 61 | Abstract, I Introduction, Supplement, V Results, VI Discussion, VII Conclusion, generated table/figure file |
| `src/subset_p2x.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/superseded_provenance.py` | imported by other modules | 269 | 245 | Abstract, I Introduction, IV Method, Supplement, V Results, VI Discussion, VII Conclusion, generated table/figure file |
| `src/supp_tables_d585.py` | RUNS HERE: a table/figure writer (README, 'Reproducing the tables and figures') | 0 | 0 | - |
| `src/synthesis2.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/table1_tpr_d282.py` | inspection only here: needs files under logs/ that are not here | 3 | 1 | V Results, generated table/figure file |
| `src/table3_tpr_d282.py` | inspection only here: needs files under logs/ that are not here | 3 | 1 | V Results, generated table/figure file |
| `src/table4_specificity_d402.py` | inspection only here: needs files under logs/ that are not here; job lists under queues/ that are not here | 0 | 0 | - |
| `src/tables_d156.py` | inspection only here: needs files under logs/ that are not here | 0 | 0 | - |
| `src/tablev_inf_d437.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/thr_variants_check.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/train_lora.py` | inspection only here: needs the experiment environment (datasets, huggingface_hub, peft, torch, transformers) | 5 | 2 | Supplement, V Results |
| `src/train_lora_p2.py` | inspection only here: needs the experiment environment (peft, torch, transformers) | 5 | 2 | Supplement, V Results |
| `src/training_data_overlap.py` | inspection only here: needs the experiment environment (datasets, huggingface_hub) | 1 | 0 | V Results |
| `src/twomoment_cells.py` | a script with no GPU-stage import; it runs only where its input files are | 80 | 77 | Abstract, I Introduction, Supplement, V Results, VI Discussion, VII Conclusion, generated table/figure file |
| `src/twomoment_fit.py` | a script with no GPU-stage import; it runs only where its input files are | 63 | 60 | Abstract, I Introduction, Supplement, V Results, VI Discussion, VII Conclusion, generated table/figure file |
| `src/twomoment_p2_d361.py` | inspection only here: needs files under logs/ that are not here | 62 | 59 | Abstract, I Introduction, Supplement, V Results, VI Discussion, VII Conclusion, generated table/figure file |
| `src/twomoment_supplement.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/validate_git_commits.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/verify_provenance.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/write_draw_manifest.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/write_draw_manifest_d446.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/write_split_manifests.py` | a script with no GPU-stage import; it runs only where its input files are | 0 | 0 | - |
| `src/xstest_hw_check_d256.py` | inspection only here: needs files under logs/ that are not here | 1 | 0 | V Results |
