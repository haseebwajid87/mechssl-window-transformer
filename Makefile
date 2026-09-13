.PHONY: check figures notebook-figures
check:
	python scripts/validate_repository.py
figures:
	python scripts/regenerate_figures.py
notebook-figures:
	python scripts/extract_notebook_figures.py
