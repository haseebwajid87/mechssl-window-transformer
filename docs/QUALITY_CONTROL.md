# Repository quality-control report

Source files are preserved unchanged. The following automated checks are advisory:

- Repeated-CV interval column `record_macro_f1_ci_high` has 5 value(s) outside [0,1] (min=0.490166, max=1.13582).
- Repeated-CV interval column `record_balanced_accuracy_ci_high` has 5 value(s) outside [0,1] (min=0.557932, max=1.12169).
- Repeated-CV interval column `record_roc_auc_ci_high` has 8 value(s) outside [0,1] (min=0.504917, max=1.20513).
- Repeated-CV interval column `record_pr_auc_ci_high` has 10 value(s) outside [0,1] (min=0.601294, max=1.19647).
- Repeated-CV interval column `record_ece_ci_low` has 5 value(s) outside [0,1] (min=-0.0877221, max=0.361557).
- Repeated-CV interval column `record_brier_ci_low` has 4 value(s) outside [0,1] (min=-0.119339, max=0.330668).

- The original notebook contains one preserved historical error output: `KeyError('cohort_group')`, followed by an executed resume patch.
- The supplied claim guardrail reports `no_support` for broad superiority claims.
- Raw PhysioNet data, model checkpoints, cache files, and many original generated figures were not part of the supplied upload and are therefore not included.
