#!/usr/bin/env python3
"""Extract ONLY complete, visible Pandas HTML tables saved in V2/V3 notebooks.

These are notebook display snapshots, which may represent a head() or other
subset and must never be confused with the standalone CSV/XLSX artifacts that
were written on the researcher's computer but were not provided here.

No dataset, GPU, third-party packages, or re-execution are required.
"""
from __future__ import annotations

import argparse
import csv
from html.parser import HTMLParser
import io
import json
from pathlib import Path

SOURCES = {
    'v2_reviewer': 'notebooks/MechSSL_Window_Transformer_V2_Reviewer_Experiments.ipynb',
    'v3_final': 'notebooks/MechSSL_Window_Transformer_V3_Final_Validation.ipynb',
}


class SimpleTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.cell_text = []
        self.row = []
        self.rows = []

    def handle_starttag(self, tag, attrs):
        if tag == 'table' and not self.in_table:
            self.in_table = True
        elif self.in_table and tag == 'tr':
            self.in_row, self.row = True, []
        elif self.in_row and tag in ('th', 'td'):
            self.in_cell, self.cell_text = True, []

    def handle_data(self, data):
        if self.in_cell:
            self.cell_text.append(data)

    def handle_endtag(self, tag):
        if self.in_cell and tag in ('th', 'td'):
            self.row.append(''.join(self.cell_text).strip())
            self.in_cell = False
        elif self.in_row and tag == 'tr':
            self.rows.append(self.row)
            self.in_row = False
        elif self.in_table and tag == 'table':
            self.in_table = False


def csv_string(rows):
    out = io.StringIO(newline='')
    writer = csv.writer(out, lineterminator='\n')
    writer.writerows(rows)
    return out.getvalue()


def generate(root: Path) -> dict[str, str]:
    files = {}
    for tag, source in SOURCES.items():
        nb = json.loads((root / source).read_text(encoding='utf8'))
        manifest = [['cell_index', 'output_index', 'filename', 'visible_rows', 'visible_columns', 'status']]
        errors = []
        for cell_idx, cell in enumerate(nb['cells']):
            for out_idx, output in enumerate(cell.get('outputs', [])):
                if output.get('output_type') == 'error':
                    errors.append({'cell_index': cell_idx, 'output_index': out_idx,
                                   'error_type': output.get('ename'),
                                   'error_message': output.get('evalue')})
                html = output.get('data', {}).get('text/html')
                if not html:
                    continue
                html = ''.join(html) if isinstance(html, list) else html
                parser = SimpleTableParser()
                parser.feed(html)
                rows = parser.rows
                if len(rows) < 2 or not rows[0]:
                    continue
                filename = f'cell_{cell_idx:03d}_output_{out_idx:03d}.csv'
                truncated = any(v.strip() in ('...', '…') for row in rows for v in row)
                aligned = all(len(row) == len(rows[0]) for row in rows)
                if not aligned or truncated:
                    manifest.append([cell_idx, out_idx, '', len(rows)-1, len(rows[0]),
                                     'skipped: display omits columns or has irregular table structure'])
                    continue
                rows[0][0] = 'notebook_row_index' if not rows[0][0] else rows[0][0]
                files[f'results/{tag}/saved_display_snapshots/{filename}'] = csv_string(rows)
                manifest.append([cell_idx, out_idx, filename, len(rows)-1, len(rows[0]),
                                 'saved display only; full source table not guaranteed'])
        files[f'results/{tag}/saved_display_manifest.csv'] = csv_string(manifest)
        files[f'results/{tag}/saved_execution_errors.json'] = json.dumps(errors, indent=2) + '\n'
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--check', action='store_true', help='Compare committed snapshots without writing')
    a = p.parse_args()
    files = generate(a.root.resolve())
    invalid = []
    for path, expected in files.items():
        dest = a.root.resolve() / path
        if a.check:
            if not dest.is_file() or dest.read_text(encoding='utf8') != expected:
                invalid.append(path)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(expected, encoding='utf8', newline='')
    print(f'{len(files)} notebook-derived text artifacts', 'verified' if a.check else 'written')
    if invalid:
        for path in invalid:
            print('MISSING OR DIFFERENT:', path)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
