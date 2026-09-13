# Manuscript-to-artifact mapping

The supplied compiler already provides `results/compiled/19_MANUSCRIPT_TABLE_INDEX_READ_THIS_FIRST.csv`. Use that file as the authoritative mapping. In summary:

- Main Table 1 → `01_dataset_characteristics_for_table1.csv`
- Main Table 2 → `02_main_model_comparison_record_level_FOR_MAIN_TABLE.csv`
- Main Table 3 → `04_proposed_model_internal_ablation_FOR_MAIN_TABLE.csv`
- Main Table 4 / Supplement → `05_external_baseline_comparison_FOR_MAIN_TABLE.csv`
- Main Table 5 → `07_statistical_tests_FOR_SIGNIFICANCE_CLAIMS.csv`
- Calibration → `08_calibration_reliability_FOR_MAIN_OR_SUPPLEMENT.csv`
- Robustness → `09_repeated_grouped_cv_FOR_ROBUSTNESS.csv`
- Confusion matrix → `10_confusion_matrix_primary_proposed_model.csv`
- Discussion/Supplement → error analysis, subgroup, sensitivity, decision curve, runtime, deployability, and window-level results (`11`–`17`).
- Writing guardrails → `18_key_claims_and_guardrails_FOR_WRITING.csv`
