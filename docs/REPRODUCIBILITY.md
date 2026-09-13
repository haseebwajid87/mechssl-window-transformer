# Reproducibility guide

## Archived execution environment

The supplied run snapshot reports:

- Python 3.10.20 (Anaconda build)
- Windows 10
- NumPy 2.0.1
- pandas 2.3.3
- PyTorch 2.5.1
- WFDB 4.3.1
- CUDA available: yes
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- CPU count: 32
- Primary seed: 42
- Repeated-CV seeds used/planned in the notebook: 42, 2024, 3407

The snapshot did not record exact versions of SciPy, scikit-learn, matplotlib, tqdm, XGBoost, or LightGBM, so those dependencies are listed without invented pins in `requirements.txt`.

## Reproduce the workflow

1. Obtain PhysioNet dataset v1.0.0 and set `cfg.data_dir`.
2. Create the environment from `requirements.txt` or `environment.yml`.
3. Open `notebooks/MechSSL_Window_Transformer.ipynb`.
4. Review the runtime switches in the configuration cell.
5. Run cells in order. Use `SMOKE_TEST=True` for a reduced sanity run before committing compute to a full run.
6. Compare newly generated files against `results/compiled/` and `results/source_reference/` rather than overwriting the archived evidence.

## Validation design captured in the archive

The configuration records 5 outer patient-wise grouped folds and 4 inner splits, with record-level macro-F1 as the primary metric. Calibration, threshold selection, bootstrap confidence intervals, paired tests, ablations, repeated grouped CV, subgroup analyses, sensitivity analyses, and R-peak deployability checks are all represented in the supplied result artifacts.

## Provenance policy

The exact supplied notebook is preserved in `notebooks/original/`. The canonical notebook retains every original cell/output and adds one documentation-only first cell. Run `python scripts/validate_repository.py` to verify this contract.
