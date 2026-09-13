# Pre-publication repository checklist

- [ ] Confirm repository name/title matches the paper.
- [ ] Confirm all paper authors who contributed to software are represented in `CITATION.cff`.
- [ ] Confirm institutional/IP approval for the MIT code license; change it if required.
- [ ] Add final GitHub URL.
- [ ] Create Zenodo DOI from a tagged GitHub release and add it to citation metadata.
- [ ] Update the manuscript Data Availability section with the final GitHub/DOI links.
- [ ] Cite the PhysioNet dataset DOI separately.
- [ ] Run `python scripts/validate_repository.py`.
- [ ] Review `docs/KNOWN_ISSUES.md` and do not overstate comparative/clinical claims.
- [ ] Decide whether model checkpoints must be archived separately for the journal/reviewer requirements.
- [ ] Verify no private patient data, credentials, local machine paths, or secrets have been added beyond the public-source research artifacts.
