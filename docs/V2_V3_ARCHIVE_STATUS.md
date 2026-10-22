# V2/V3 research archive status (2026-10-23)

This repository preserves the exact user-supplied executed V2 and V3 notebooks as new independent research evidence. The notebook files were verified against their original Git object SHA-1 values. **Code and saved outputs are archived, but independent reproducibility and clinical validation are not established.**

## Version index

| Version | Notebook | Cells | Stored outputs | Notes |
| --- | --- | ---: | ---: | --- |
| V1 original | `notebooks/original/MechSSL-Window Transformer.original.ipynb` | 48 | 487 | Preserved exact original; one recorded KeyError and later resume |
| V1 canonical | `notebooks/MechSSL_Window_Transformer.ipynb` | 49 | 487 | Documentation-only lead cell |
| V2 reviewer experiments | `notebooks/MechSSL_Window_Transformer_V2_Reviewer_Experiments.ipynb` | 57 | 42 | Patient-level calibration, label-rule sensitivity, strict modality models, automatic R-peak assessment, matched SSL vs scratch |
| V3 final validation | `notebooks/MechSSL_Window_Transformer_V3_Final_Validation.ipynb` | 49 | 81 | Clinical diagnosis audit, ECG/IEGM comparisons, automatic localization, annotation-derived pacing, repeated matched SSL and random initialization |

### V2 scope and caveats
The V2 notebook has a saved `RuntimeError` in Experiment E involving a non-differentiable loss. A subsequent recovery cell resumed execution. Its displayed data and logs are historical, not proof that the notebook runs cleanly from a new environment.

### V3 scope and caveats
V3 has two historical saved `FileNotFoundError` cells followed by recovery/resume cells. V3 explicitly reports the lack of an independent synchronized external validation cohort. Its local exports were written to `C:\\Users\\hasee\\V3_AV_OUT`. These actual XLSX/CSV files were **not** among the three provided notebooks.

### Missing local exports
V2 notebook logs refer to `outputs_mechxformer_av_v2_reviewer`; V3 logs refer to `V3_AV_OUT`. Folder contents, model checkpoints, and all complete per-fold CSV exports are **not** present in this update. Notebook HTML summaries are often truncated and are NOT interchangeable with complete original result datasets. The existing V1 result archive remains unchanged.

### Reproducibility integrity
- `python scripts/validate_repository.py` checks archive structure and declared file digests, without training the model.
- `python scripts/extract_saved_results.py` can regenerate a manifest and only those table displays that can be read without truncation from the notebook outputs.
- `python scripts/regenerate_figures.py` regenerates descriptive V1 figures from existing result CSVs, not model results.
- `python scripts/extract_notebook_figures.py` reconstructs embedded original V1 notebook figure PNG files.
- Clinical and statistical claims must still be reviewed against independent validation, raw data, full exports, and manuscript methodology.

These notebooks preserve historical evidence **without silently altering scores, fixing experimental methodology, or fabricating missing raw outputs**.
