#!/usr/bin/env python3
"""Regenerate descriptive figures from the archived V1 compiled CSV tables.

Produces separate, explicitly regenerated plots without modifying the archived
figures/regenerated/ evidence. Requires matplotlib; does not rerun any model.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path


def table(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = args.root.resolve()
    base = root / 'results/compiled'
    dest = root / 'figures/reproducible_from_csv'
    dest.mkdir(parents=True, exist_ok=True)

    for source, filename, title in [
        ('02_main_model_comparison_record_level_FOR_MAIN_TABLE.csv', 'model_comparison.png',
         'Archived V1 record-level model comparison'),
        ('04_proposed_model_internal_ablation_FOR_MAIN_TABLE.csv', 'internal_ablation.png',
         'Archived V1 model ablations'),
    ]:
        data = table(base / source)
        rows = [(r['paper_model_name'], float(r['record_macro_f1'])) for r in data
                if r.get('record_macro_f1') and r.get('paper_model_name')]
        if not rows:
            raise RuntimeError(f'No plottable values: {source}')
        rows = sorted(rows, key=lambda x: x[1])
        fig, ax = plt.subplots(figsize=(10, max(4, len(rows) * 0.45)))
        ax.barh([x[0] for x in rows], [x[1] for x in rows])
        ax.set(xlim=(0, 1), xlabel='Record-level macro-F1', title=title)
        fig.tight_layout()
        fig.savefig(dest / filename, dpi=180)
        plt.close(fig)

    cm = table(base / '10_confusion_matrix_primary_proposed_model.csv')
    import numpy as np
    matrix = np.array([[int(row['Pred_AVNRT']), int(row['Pred_AVRT'])] for row in cm])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap='Greys')
    ax.set_xticks([0, 1], ['Pred AVNRT', 'Pred AVRT'])
    ax.set_yticks([0, 1], ['True AVNRT', 'True AVRT'])
    ax.set_title('Archived V1 confusion matrix')
    for (r, c), value in np.ndenumerate(matrix):
        ax.text(c, r, str(value), ha='center', va='center')
    fig.tight_layout()
    fig.savefig(dest / 'confusion_matrix.png', dpi=180)
    plt.close(fig)
    print(f'Regenerated 3 descriptive figures to {dest.relative_to(root)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
