# Artifacts referenced by the run but not supplied

The supplied `notebook_run_manifest.json` records a much larger original output tree than the provided compiled archive. The repository therefore distinguishes **supplied/preserved** artifacts from **referenced-but-missing** artifacts.

Not supplied in the current upload include:

- Model checkpoints (`*.pt`) — the run manifest references 240 checkpoint files across main and repeated runs.
- Window cache (`*.npz`) — the run manifest references 13,228 cached windows.
- Many original generated PNG figures — the run manifest references hundreds of figure files.
- Numerous per-fold training-history and repeated-run CSVs beyond the curated key files copied into the compiled archive.

These files are not reconstructed or invented. The cache is reproducible from the public dataset and notebook; checkpoints would need to be supplied separately if exact weight-level reproducibility is a publication requirement. Core figures are regenerated from the supplied result tables/predictions under `figures/regenerated/`.

For large checkpoints, prefer a release/archive service (e.g. Zenodo) or Git LFS rather than committing hundreds of binary weight files directly to normal Git history.
