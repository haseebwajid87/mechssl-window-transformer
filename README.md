# MechSSL-Window Transformer

**Research companion repository for AVNRT-vs-AVRT mechanism classification from 12-lead ECG and intracardiac electrograms.**

> **Research use only.** This repository documents an internal-validation study. It is not a clinical device, is not externally/prospectively validated, and must not be used for autonomous diagnosis or treatment decisions.

## What this repository contains

This package was built from the supplied executed notebook and compiled result archive. It preserves the executed evidence while making the project easier to inspect, cite, reproduce, and upload to GitHub.

- `notebooks/MechSSL_Window_Transformer.ipynb` — GitHub-ready canonical notebook. All **48 original cells** and **487 original outputs** are retained; one documentation-only markdown cell is added at the top.
- `notebooks/original/` — exact original notebook supplied for this repository build.
- `results/compiled/` — manuscript-facing tables and the compiled workbook.
- `results/source_reference/` — all original key result/reference files from the supplied results archive, including OOF predictions, calibration, bootstrap CIs, repeated CV, statistical tests, configuration, run manifest, model card, and reporting checklist.
- `figures/regenerated/` — publication-facing figures regenerated from supplied result tables/predictions.
- `figures/notebook_embedded/` — PNG outputs embedded in the executed notebook.
- `config/` — run configuration, environment snapshot, feature dimensions, and leakage audit.
- `src/notebook_export.py` — searchable code-cell export; the notebook remains canonical.
- `scripts/` — repository integrity checks and reproducible figure generation.
- `docs/` — data access, reproducibility, result interpretation, known issues, manuscript mapping, release/DOI instructions, and a draft data-availability statement.
- `provenance/` — input hashes, preservation report, repository manifest, and checksums.

## Study snapshot

The supplied compiled results report **39 discovered records**, **28 AVNRT/AVRT records**, **1,410 binary supervised windows**, and **5 patient-wise grouped outer folds**. The pre-specified proposed model is `ssl_window_transformer/window_level` (**MechSSL-Window Transformer**) with patient/record-level macro-F1 **0.767** (95% CI **0.590–0.918**), AUROC **0.964**, PR-AUC **0.968**, ECE **0.157**, and Brier score **0.077**.

The repository deliberately preserves the study's claim guardrails: the proposed model was **not** the best observed row in the supplied comparison table (the signal-only internal ablation had macro-F1 0.892), and the statistical-test exports do not support a broad claim of superiority over all baselines. See `results/compiled/18_key_claims_and_guardrails_FOR_WRITING.csv` and `docs/RESULTS_GUIDE.md`.

## Data

The notebook uses **Leipzig Heart Center ECG-Database: Arrhythmias in Children and Patients with Congenital Heart Disease, v1.0.0** (PhysioNet DOI `10.13026/7a4j-vn37`). Raw PhysioNet files are **not redistributed** here. Download the dataset from PhysioNet and set `cfg.data_dir` in the configuration cell. See `docs/DATA.md`.

## Environment

The archived run reports Python **3.10.20**, NumPy **2.0.1**, pandas **2.3.3**, PyTorch **2.5.1**, WFDB **4.3.1**, Windows 10, and an NVIDIA GeForce RTX 4060 Laptop GPU. Additional imported packages are listed in `requirements.txt`. GPU/CPU-specific PyTorch installation may require selecting the appropriate official PyTorch build.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Open `notebooks/MechSSL_Window_Transformer.ipynb`, verify `cfg.data_dir`, and run in order. For a lightweight validation run, set `SMOKE_TEST = True` before execution.

## Repository integrity

Run the fast, data-free integrity check before publishing:

```bash
python scripts/validate_repository.py
```

This verifies the notebook preservation contract, required result files, CSV readability, file-size safety, and the intentional preservation of the historical error output.

## Citation and DOI

`CITATION.cff` is included so GitHub can expose a **Cite this repository** action. Before public release, review the author list and repository license with all coauthors/institutional requirements. After the GitHub repository is final, connect it to Zenodo and create a tagged GitHub release to obtain a persistent DOI; then update `CITATION.cff` and the paper's Data Availability statement. See `docs/RELEASE_AND_DOI.md`.

## Important completeness note

The supplied result archive contains the paper-facing results and many original key files, but the run manifest references additional generated artifacts that were **not supplied**, notably model checkpoint files (`*.pt`), the full window cache (`*.npz`), and many original generated PNG figures. These are not fabricated here. Core publication figures are regenerated from the supplied CSV/prediction files where possible. See `docs/MISSING_ARTIFACTS.md`.
