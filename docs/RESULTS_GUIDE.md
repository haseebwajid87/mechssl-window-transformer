# Results interpretation guide

## Primary definition

The supplied compiler defines the pre-specified primary proposed model as:

- Model: `ssl_window_transformer`
- Ablation/setting: `window_level`
- Display name: **MechSSL-Window Transformer**
- Primary endpoint: patient/record-level macro-F1 under strict patient-wise grouped cross-validation

## Primary reported performance

From `03_primary_proposed_model_summary.csv`:

- n = 28 records
- macro-F1 = 0.766667 (95% CI 0.590–0.918)
- accuracy = 0.785714
- balanced accuracy = 0.769231
- AUROC = 0.964103
- PR-AUC = 0.968378
- ECE = 0.157352
- Brier = 0.077012

The primary confusion matrix is 7/6 for true AVNRT predicted AVNRT/AVRT, and 0/15 for true AVRT predicted AVNRT/AVRT.

## Claim guardrails

Do **not** claim that the proposed model is the best observed model or that it significantly outperforms all baselines. The supplied comparison table has a higher observed macro-F1 for the signal-only internal ablation (0.891613), and the supplied claim-guardrail report classifies broad superiority support as `no_support`.

Use `07_statistical_tests_FOR_SIGNIFICANCE_CLAIMS.csv` before making any significance statement. The archive supports careful comparative wording, not blanket superiority.

## Internal validation only

No external/prospective validation is included in the supplied artifacts. Do not describe the system as clinically validated or deployment-ready.
