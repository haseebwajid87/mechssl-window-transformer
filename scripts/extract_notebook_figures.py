#!/usr/bin/env python3
"""Extract image/png outputs actually embedded in the executed V1 notebook."""
from __future__ import annotations
import argparse
import base64
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    nb = json.loads((root / 'notebooks/original/MechSSL-Window Transformer.original.ipynb').read_text(encoding='utf8'))
    target = root / 'figures/notebook_embedded'
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    for ci, cell in enumerate(nb['cells']):
        for oi, output in enumerate(cell.get('outputs', [])):
            payload = output.get('data', {}).get('image/png')
            if not payload:
                continue
            raw = base64.b64decode(''.join(payload) if isinstance(payload, list) else payload)
            filename = f'cell_{ci}_output_{oi:03d}.png'
            (target / filename).write_bytes(raw)
            rows.append((ci, oi, filename))
    with (target / 'index.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['original_cell_index', 'output_index', 'filename'])
        writer.writerows(rows)
    print(f'Extracted {len(rows)} saved figure outputs')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
