#!/usr/bin/env python3
"""
Plot sweep results as heatmaps and line plots.

Usage examples:
  python plot_sweep.py --summary runs/sweep_YYYYMMDD_HHMMSS/summary.csv --approach implicit --outdir runs/sweep_YYYYMMDD_HHMMSS/plots

Outputs:
  - For implicit: heatmap_acc_n_by_m.png, lines_acc_vs_m_by_n.png
  - For explicit: lines_acc_vs_n_by_L.png (if fields present)
"""
import argparse, csv, os
from collections import defaultdict

import matplotlib.pyplot as plt


def read_rows(summary_path: str):
    rows = []
    with open(summary_path, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def plot_implicit(rows, outdir: str):
    # Aggregate by (n,m) across seeds
    by_nm = defaultdict(list)
    for r in rows:
        if r.get('approach') != 'implicit':
            continue
        try:
            n = int(r['n']); m = int(r['m']); acc = float(r['acc'])
        except Exception:
            continue
        by_nm[(n, m)].append(acc)

    if not by_nm:
        print('No implicit rows found.')
        return

    ns = sorted({n for (n, _) in by_nm.keys()})
    ms = sorted({m for (_, m) in by_nm.keys()})

    # Heatmap matrix
    import numpy as np
    mat = np.zeros((len(ns), len(ms)))
    for i, n in enumerate(ns):
        for j, m in enumerate(ms):
            vals = by_nm.get((n, m), [])
            mat[i, j] = (sum(vals) / len(vals)) if vals else float('nan')

    # Heatmap: EM vs (n,m)
    plt.figure(figsize=(10, 5))
    im = plt.imshow(mat, aspect='auto', cmap='viridis', vmin=0.0, vmax=1.0)
    plt.colorbar(im, label='EM')
    plt.xticks(range(len(ms)), ms)
    plt.yticks(range(len(ns)), ns)
    plt.xlabel('m (chains)')
    plt.ylabel('n (hops)')
    plt.title('Implicit: EM heatmap (mean across seeds)')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'heatmap_acc_n_by_m.png'))
    plt.close()

    # Line plots: EM vs m for each n
    plt.figure(figsize=(10, 5))
    for n in ns:
        ys = []
        for m in ms:
            vals = by_nm.get((n, m), [])
            ys.append((sum(vals) / len(vals)) if vals else float('nan'))
        plt.plot(ms, ys, marker='o', label=f'n={n}')
    plt.xlabel('m (chains)')
    plt.ylabel('EM')
    plt.ylim(0, 1)
    plt.title('Implicit: EM vs m by n (mean across seeds)')
    plt.legend(ncol=min(3, len(ns)))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'lines_acc_vs_m_by_n.png'))
    plt.close()


def plot_explicit(rows, outdir: str):
    # Aggregate by (n,L) across seeds, using unified k or k0/k1/k2 if desired
    by_nL = defaultdict(list)
    for r in rows:
        if r.get('approach') == 'implicit':
            continue
        try:
            n = int(r['n']); L = int(r.get('L') or 0); acc = float(r['acc'])
        except Exception:
            continue
        by_nL[(n, L)].append(acc)

    if not by_nL:
        print('No explicit rows found or L not provided; skipping explicit plots.')
        return

    ns = sorted({n for (n, _) in by_nL.keys()})
    Ls = sorted({L for (_, L) in by_nL.keys()})

    import numpy as np
    mat = np.zeros((len(ns), len(Ls)))
    for i, n in enumerate(ns):
        for j, L in enumerate(Ls):
            vals = by_nL.get((n, L), [])
            mat[i, j] = (sum(vals) / len(vals)) if vals else float('nan')

    # Heatmap EM vs (n,L)
    plt.figure(figsize=(10, 5))
    im = plt.imshow(mat, aspect='auto', cmap='viridis', vmin=0.0, vmax=1.0)
    plt.colorbar(im, label='EM')
    plt.xticks(range(len(Ls)), Ls)
    plt.yticks(range(len(ns)), ns)
    plt.xlabel('L (decision depth)')
    plt.ylabel('n (hops)')
    plt.title('Explicit: EM heatmap (mean across seeds)')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'heatmap_explicit_acc_n_by_L.png'))
    plt.close()

    # Line plot: EM vs n for each fixed L
    by_L_then_n = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get('approach') == 'implicit':
            continue
        try:
            n = int(r['n']); L = int(r.get('L') or 0); acc = float(r['acc'])
        except Exception:
            continue
        by_L_then_n[L][n].append(acc)
    if by_L_then_n:
        plt.figure(figsize=(10, 5))
        for L in sorted(by_L_then_n.keys()):
            xs = sorted(by_L_then_n[L].keys())
            ys = [sum(by_L_then_n[L][n]) / len(by_L_then_n[L][n]) for n in xs]
            plt.plot(xs, ys, marker='o', label=f'L={L}')
        plt.xlabel('n (hops)')
        plt.ylabel('EM')
        plt.ylim(0, 1)
        plt.title('Explicit: EM vs n by L (mean across seeds)')
        plt.legend(ncol=min(3, len(by_L_then_n)))
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'lines_explicit_acc_vs_n_by_L.png'))
        plt.close()

    # (Omitted) Redundant line plot (EM vs L by n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', required=True, help='Path to sweep summary.csv')
    ap.add_argument('--approach', choices=['explicit', 'implicit', 'auto'], default='auto')
    ap.add_argument('--outdir', required=True, help='Output directory for plots')
    args = ap.parse_args()

    ensure_dir(args.outdir)
    rows = read_rows(args.summary)

    approach = args.approach
    if approach == 'auto':
        # Infer from first row
        a = next((r.get('approach') for r in rows if r.get('approach')), 'explicit')
        approach = a

    if approach == 'implicit':
        plot_implicit(rows, args.outdir)
    elif approach == 'explicit':
        plot_explicit(rows, args.outdir)
    else:
        # Try both when auto
        plot_implicit(rows, args.outdir)
        plot_explicit(rows, args.outdir)

    print(f"Plots written to: {args.outdir}")


if __name__ == '__main__':
    main()


