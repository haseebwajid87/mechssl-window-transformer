# Known issues and preserved historical behavior

1. **Historical `KeyError('cohort_group')` is intentionally preserved in the notebook output.** It occurred in the manuscript-ready runner when a subgroup column existed in both prediction and metadata frames, causing pandas merge suffixing. The next executed original cell is explicitly labeled as a resume patch and fixes the issue by dropping the duplicate subgroup column before merging and by isolating subgroup failures.

2. **Repeated-CV confidence-interval bounds are not constrained to the natural metric range.** In the supplied `09_repeated_grouped_cv_FOR_ROBUSTNESS.csv`, some reported interval endpoints extend below 0 or above 1 (for example, the proposed model macro-F1 interval has an upper endpoint >1). The original file is preserved unchanged. For publication, consider reporting a bounded bootstrap interval or explicitly describing the interval construction.

3. **The run manifest references artifacts not present in the supplied result archive.** See `MISSING_ARTIFACTS.md`.

4. **Dependency snapshot is partial.** Exact versions were recorded for Python/NumPy/pandas/PyTorch/WFDB, but not every imported library. Unrecorded versions are not fabricated in this repository.
