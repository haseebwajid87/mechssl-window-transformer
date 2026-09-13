# GitHub release and DOI checklist

1. Create a new GitHub repository (suggested name: `mechssl-window-transformer`).
2. Upload the **contents** of this repository package, preserving the directory structure.
3. Confirm `python scripts/validate_repository.py` passes locally or in GitHub Actions.
4. Review `CITATION.cff`: add every author/coauthor who should receive software credit and add ORCIDs if appropriate.
5. Review the MIT license with coauthors/institutional ownership requirements before public release.
6. Add the final GitHub repository URL to the README and citation metadata.
7. Connect the repository to Zenodo, enable the repository, then create a versioned GitHub release (for example `v1.0.0`).
8. After Zenodo creates the software record/DOI, add that DOI to `CITATION.cff`, README, and the manuscript Data Availability statement.
9. Cite the PhysioNet source dataset separately using DOI `10.13026/7a4j-vn37`.
10. Freeze the exact paper-associated GitHub release and avoid silently replacing result files after submission; use a new release for later changes.
