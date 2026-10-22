# Known issues and preserved historical behavior

1. **Historical `KeyError('cohort_group')` is intentionally preserved in the notebook output.** It occurred in the manuscript-ready runner when a subgroup column existed in both prediction and metadata frames, causing pandas merge suffixing. The next executed original cell is explicitly labeled as a resume patch and fixes the issue by dropping the duplicate subgroup column before merging and by isolating subgroup failures.

2. **Repeated-CV confidence-interval bounds are not constrained to the natural metric range.** In the supplied `09_repeated_grouped_cv_FOR_ROBUSTNESS.csv`, some reported interval endpoints extend below 0 or above 1 (for example, the proposed model macro-F1 interval has an upper endpoint >1). The original file is preserved unchanged. For publication, consider reporting a bounded bootstrap interval or explicitly describing the interval construction.

3. **The run manifest references artifacts not present in the supplied result archive.** See `MISSING_ARTIFACTS.md`.

4. **Dependency snapshot is partial.** Exact versions were recorded for Python/NumPy/pandas/PyTorch/WFDB, but not every imported library. Unrecorded versions are not fabricated in this repository.

5. **V2 archived execution error:** The matched-SSL experiment encounters a non-differentiable-loss `RuntimeError` and resumes in a later recovery cell. The notebook remains an unmodified historical record, not proof of a clean full rerun.

6. **V3 archived path errors:** Two `FileNotFoundError` outputs occur before recovery cells. Some paths are absolute local Windows paths and are not portable. The captured follow-up outputs remain available.

7. **V2/V3 complete local result folders not supplied:** The new `results/v2_reviewer/` and `results/v3_final/` files represent only visible saved notebook output, not the original full CSV/XLSX exports or checkpoints. See `V2_V3_ARCHIVE_STATUS.md`.

8. **Clinical validation limit:** There is no genuinely independent synchronized external cohort in the supplied V3 evidence, and the clinical diagnosis concordance audit is not a substitute for external validation.
