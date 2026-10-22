#!/usr/bin/env python3
"""Data-free structural audit of the archived MechSSL research repository.

A passing check confirms the files and saved evidence have not been changed.
It does NOT establish independent reproduction, statistical validity, or external
clinical validation. Requires only Python's standard library.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

NOTEBOOKS = {
    'V1-original': ('notebooks/original/MechSSL-Window Transformer.original.ipynb', 48, 487,
                    'aa9cc1d550f4ab8e534874714181305a62bc0fec1778d636064eb364442a441f'),
    'V1-canonical': ('notebooks/MechSSL_Window_Transformer.ipynb', 49, 487, None),
    'V2': ('notebooks/MechSSL_Window_Transformer_V2_Reviewer_Experiments.ipynb', 57, 42,
           '2dff8d5f85da5e57cb3be9f443eb22a06959a337c9cac05e7d7ec885cdf7d9de'),
    'V3': ('notebooks/MechSSL_Window_Transformer_V3_Final_Validation.ipynb', 49, 81,
           '016c1eb3ef455217411357c54e6dd01db7eb9033ec557a7027e4344bfc19b816'),
}
REQUIRED = [
    'README.md', 'CITATION.cff', 'LICENSE', 'requirements.txt',
    'results/compiled/03_primary_proposed_model_summary.csv',
    'results/compiled/18_key_claims_and_guardrails_FOR_WRITING.csv',
    'results/compiled/19_MANUSCRIPT_TABLE_INDEX_READ_THIS_FIRST.csv',
    'results/source_reference/all_fold_results.csv',
    'results/source_reference/main_record_level_results.csv',
    'docs/MISSING_ARTIFACTS.md', 'docs/RESULTS_GUIDE.md',
    'scripts/extract_saved_results.py',
]


def check(root: Path) -> tuple[list[str], list[str], dict]:
    failures, notices, data = [], [], {}
    notebooks = {}
    for path in REQUIRED:
        if not (root / path).is_file():
            failures.append(f'Missing required file: {path}')
    for tag, (relative, cells_wanted, outputs_wanted, digest) in NOTEBOOKS.items():
        path = root / relative
        if not path.is_file():
            failures.append(f'Missing notebook {tag}: {relative}')
            continue
        raw = path.read_bytes()
        if digest and hashlib.sha256(raw).hexdigest() != digest:
            failures.append(f'{tag}: SHA256 differs from supplied notebook; investigate modifications')
        try:
            nb = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            failures.append(f'{tag}: invalid notebook JSON: {exc}')
            continue
        cells = nb.get('cells', [])
        noutputs = sum(len(c.get('outputs', [])) for c in cells)
        errors = [{'cell': i, 'type': o.get('ename'), 'message': str(o.get('evalue', ''))[:160]}
                  for i, c in enumerate(cells) for o in c.get('outputs', [])
                  if o.get('output_type') == 'error']
        if nb.get('nbformat') != 4 or not isinstance(cells, list):
            failures.append(f'{tag}: unexpected notebook format')
        if len(cells) != cells_wanted or noutputs != outputs_wanted:
            failures.append(f'{tag}: cell/output count {len(cells)}/{noutputs}, expected {cells_wanted}/{outputs_wanted}')
        if errors:
            notices.append(f'{tag}: {len(errors)} historical saved execution error(s); see docs/KNOWN_ISSUES.md')
        notebooks[tag] = nb
        data[tag] = {'cells': len(cells), 'saved_outputs': noutputs,
                     'sha256': hashlib.sha256(raw).hexdigest(), 'historical_errors': errors}
    if 'V1-original' in notebooks and 'V1-canonical' in notebooks:
        if notebooks['V1-canonical']['cells'][1:] != notebooks['V1-original']['cells']:
            failures.append('Canonical V1 cells differ from the exact original beyond the documented intro cell')
    for rel in ('results/compiled', 'results/source_reference'):
        folder = root / rel
        if not folder.is_dir() or not any(folder.glob('*.csv')):
            failures.append(f'No CSV results in {rel}')
        else:
            for f in folder.glob('*.csv'):
                try:
                    with f.open(encoding='utf-8-sig', newline='') as fh:
                        header = next(csv.reader(fh), [])
                    if not header:
                        failures.append(f'CSV has no header: {f.relative_to(root)}')
                except (OSError, UnicodeError, csv.Error) as exc:
                    failures.append(f'Unreadable CSV {f.relative_to(root)}: {exc}')
    for rel in ('results/v2_reviewer/saved_display_manifest.csv',
                'results/v3_final/saved_display_manifest.csv'):
        if not (root / rel).is_file():
            failures.append(f'Missing derived display manifest: {rel}')
    notices.append('V2/V3 standalone exported workbooks, per-fold CSVs, checkpoints, and caches were not uploaded; saved notebook displays are NOT replacements.')
    notices.append('External clinical cohort validation is unavailable in the supplied evidence.')
    return failures, notices, data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--json', action='store_true', help='Print structured report')
    args = parser.parse_args()
    failures, notices, report = check(args.root.resolve())
    if args.json:
        print(json.dumps({'ok': not failures, 'failures': failures, 'notices': notices,
                          'notebooks': report}, indent=2))
    else:
        for key, stats in report.items():
            print(f"{key}: {stats['cells']} cells, {stats['saved_outputs']} saved outputs, "
                  f"{len(stats['historical_errors'])} historical errors")
        for msg in failures:
            print('FAIL:', msg, file=sys.stderr)
        for msg in notices:
            print('NOTE:', msg)
        print('PASS: archive structural checks' if not failures else 'FAIL: archive structural checks')
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
