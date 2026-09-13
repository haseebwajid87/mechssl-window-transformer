# Q1 Paper Compiled Results

Created: 20260527_072954

This folder contains the clean result files to use for manuscript writing.

## Primary proposed model

MechSSL-Window Transformer = `ssl_window_transformer/window_level`

## Primary endpoint

Patient/record-level macro-F1 under strict patient-wise grouped cross-validation.

## Use these first

1. `19_MANUSCRIPT_TABLE_INDEX_READ_THIS_FIRST.csv`
2. `01_dataset_characteristics_for_table1.csv`
3. `02_main_model_comparison_record_level_FOR_MAIN_TABLE.csv`
4. `03_primary_proposed_model_summary.csv`
5. `04_proposed_model_internal_ablation_FOR_MAIN_TABLE.csv`
6. `05_external_baseline_comparison_FOR_MAIN_TABLE.csv`
7. `07_statistical_tests_FOR_SIGNIFICANCE_CLAIMS.csv`
8. `08_calibration_reliability_FOR_MAIN_OR_SUPPLEMENT.csv`
9. `09_repeated_grouped_cv_FOR_ROBUSTNESS.csv`
10. `18_key_claims_and_guardrails_FOR_WRITING.csv`

## Main interpretation rule

Use `ssl_window_transformer/window_level` as the proposed model.
Use ECG-only, IEGM-only, signal-only, no-mechanism-token, and no-handcrafted-feature rows as internal ablations.
Use classical ML, scratch transformer, MIL, CNN/ResNet/Inception/CNN-LSTM rows as comparison baselines.

## Do not overclaim

This is an internal-validation study unless external/prospective validation has been performed.
Use the statistical-test table before saying "significantly outperformed."

## Original key files

Original result files are copied into:

`original_key_files_reference/`