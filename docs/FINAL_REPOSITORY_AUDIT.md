# Final repository audit — 2026-10-23

## Scope and result

**Archive packaging status: complete for all three supplied notebooks and the existing committed result archive.** This means the source notebooks, saved notebook outputs, available V1 results, derived V2/V3 display snapshots, provenance materials, and checking scripts are present. It **does not** mean all experiments have been rerun or every original local export has been obtained.

### Verified supplied notebook identity

| Version | GitHub location | Cells | Saved outputs | SHA-256 of supplied notebook |
| --- | --- | ---: | ---: | --- |
| V1 original | `notebooks/original/MechSSL-Window Transformer.original.ipynb` | 48 | 487 | `aa9cc1d550f4ab8e534874714181305a62bc0fec1778d636064eb364442a441f` |
| V2 reviewer | `notebooks/MechSSL_Window_Transformer_V2_Reviewer_Experiments.ipynb` | 57 | 42 | `2dff8d5f85da5e57cb3be9f443eb22a06959a337c9cac05e7d7ec885cdf7d9de` |
| V3 final analyses | `notebooks/MechSSL_Window_Transformer_V3_Final_Validation.ipynb` | 49 | 81 | `016c1eb3ef455217411357c54e6dd01db7eb9033ec557a7027e4344bfc19b816` |

These three digests were checked against the three notebook uploads and agree with the constants in `scripts/validate_repository.py`. The canonical V1 notebook has a documentation-only lead cell and is not byte-identical to the original.

### Present

- V1/V2/V3 notebook files with saved outputs.
- `results/compiled/` and `results/source_reference/` for original study tables/predictions.
- `results/v2_reviewer/` and `results/v3_final/` for extracted, visibly saved display snapshots and historical errors.
- `scripts/validate_repository.py`, `scripts/extract_saved_results.py`, and the figure extraction/generation scripts.
- Archive-integrity GitHub Actions workflow, project instructions, environment files, and provenance.


### Checks for a user cloning the repository

```bash
python scripts/validate_repository.py
python scripts/extract_saved_results.py --check
```

For exact experiments, acquire the dataset under its source terms, adjust local data paths, supply missing dependencies/artifacts, and rerun on a documented hardware/software environment. Compare regenerated fold-level results to the archived reports rather than assuming saved notebook output reproduces itself.

**Editorial claim:** The repository is a well-documented archive of supplied code and available study evidence; it is **not** a complete independent reproduction package while the original V2/V3 exports and checkpoints remain absent.
