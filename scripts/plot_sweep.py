#!/usr/bin/env python3
"""Plot implicit sweep results as line charts."""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_rows(summary_path: str) -> list[dict]:
    rows: list[dict] = []
    with open(summary_path, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def aggregate_by_m_and_n(rows: list[dict]) -> tuple[list[int], dict[int, list[float]], dict[int, list[float]]]:
    ns = sorted({int(r['n']) for r in rows if r.get('approach') == 'implicit'})
    lift_by_m: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    acc_by_m: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get('approach') != 'implicit':
            continue
        n = int(r['n'])
        m = int(r['m'])
        acc = float(r['acc'])
        chance = 0.0 if m <= 1 else 1.0 / m
        lift = (acc - chance) / (1.0 - chance) if (1.0 - chance) > 0 else None
        acc_by_m[m][n].append(acc)
        if lift is not None:
            lift_by_m[m][n].append(lift)
    mean_acc: dict[int, list[float]] = {}
    mean_lift: dict[int, list[float]] = {}
    for m in sorted(acc_by_m.keys()):
        mean_acc[m] = []
        mean_lift[m] = []
        for n in ns:
            acc_vals = acc_by_m[m].get(n, [])
            lift_vals = lift_by_m[m].get(n, [])
            mean_acc[m].append((sum(acc_vals) / len(acc_vals)) if acc_vals else np.nan)
            mean_lift[m].append((sum(lift_vals) / len(lift_vals)) if lift_vals else np.nan)
    return ns, mean_acc, mean_lift


def plot_lines(ns: list[int], series_by_m: dict[int, list[float]], ylabel: str, title: str, out_path: Path):
    plt.figure(figsize=(10, 5))
    for m in sorted(series_by_m.keys()):
        plt.plot(ns, series_by_m[m], marker='o', label=f'm={m}')
    plt.ylim(-0.05, 1.05)
    plt.yticks([i / 10 for i in range(0, 11)])
    plt.xlabel('n (hops)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.3)
    if series_by_m:
        plt.legend(title='m', loc='best')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_implicit(rows: list[dict], outdir: Path, regime_label: str | None):
    ns, mean_acc, mean_lift = aggregate_by_m_and_n(rows)
    if not ns:
        print('No implicit rows found; nothing to plot.')
        return

    label_suffix = f" [{regime_label}]" if regime_label else ''
    plot_lines(
        ns,
        mean_acc,
        'Accuracy (EM)',
        f'Implicit: Accuracy vs n (by m){label_suffix}',
        outdir / 'lines_acc_vs_n_by_m.png',
    )
    plot_lines(
        ns,
        mean_lift,
        'Lift over chance',
        f'Implicit: Lift vs n (by m){label_suffix}',
        outdir / 'lines_lift_vs_n_by_m.png',
    )


def main():
    ap = argparse.ArgumentParser(description='Plot implicit sweep results')
    ap.add_argument('--summary', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--label', default='')
    args = ap.parse_args()

    rows = read_rows(args.summary)
    outdir = Path(args.outdir)
    ensure_dir(outdir)
    plot_implicit(rows, outdir, args.label or None)
    print(f'Plots written to: {outdir}')


if __name__ == '__main__':
    main()
