#!/usr/bin/env python3
"""
Plot sweep results as heatmaps and line plots with lift overlays and per-block bars.
"""
import argparse, csv, os
from collections import defaultdict

import matplotlib.pyplot as plt
from pathlib import Path


def read_rows(summary_path: str):
    rows = []
    with open(summary_path, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def plot_implicit(rows, outdir: str):
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

    import numpy as np
    mat = np.zeros((len(ns), len(ms)))
    for i, n in enumerate(ns):
        for j, m in enumerate(ms):
            vals = by_nm.get((n, m), [])
            mat[i, j] = (sum(vals) / len(vals)) if vals else float('nan')

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

    plt.figure(figsize=(10, 5))
    for n in ns:
        ys = []
        for m in ms:
            vals = by_nm.get((n, m), [])
            em = (sum(vals) / len(vals)) if vals else float('nan')
            chance = 0.0 if m <= 1 else 1.0 / m
            lift = (em - chance) / (1.0 - chance) if em == em and (1.0 - chance) > 0 else float('nan')
            ys.append(lift)
        plt.plot(ms, ys, marker='o', label=f'n={n}')
    plt.xlabel('m (chains)')
    plt.ylabel('Lift over chance')
    plt.ylim(0, 1)
    plt.title('Implicit: Lift-over-chance vs m by n')
    plt.legend(ncol=min(3, len(ns)))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'lines_lift_vs_m_by_n.png'))
    plt.close()


def plot_explicit(rows, outdir: str):
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

    by_L_then_n_lift = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get('approach') == 'implicit':
            continue
        try:
            n = int(r['n']); L = int(r.get('L') or 0); acc = float(r['acc'])
            k_last = None
            if r.get('k'):
                k_last = int(r['k'])
            else:
                ks = [int(r[x]) for x in ('k0','k1','k2') if r.get(x)]
                k_last = ks[-1] if ks else None
            if not k_last or k_last <= 1:
                continue
            chance = 1.0 / k_last
            lift = (acc - chance) / (1.0 - chance)
            by_L_then_n_lift[L][n].append(lift)
        except Exception:
            pass
    if by_L_then_n_lift:
        plt.figure(figsize=(10, 5))
        for L in sorted(by_L_then_n_lift.keys()):
            xs = sorted(by_L_then_n_lift[L].keys())
            ys = [sum(by_L_then_n_lift[L][n]) / len(by_L_then_n_lift[L][n]) for n in xs]
            plt.plot(xs, ys, marker='o', label=f'L={L}')
        plt.xlabel('n (hops)')
        plt.ylabel('Lift over chance')
        plt.ylim(0, 1)
        plt.title('Explicit: Lift-over-chance vs n by L')
        plt.legend(ncol=min(3, len(by_L_then_n_lift)))
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'lines_explicit_lift_vs_n_by_L.png'))
        plt.close()

    try:
        import importlib
        analyzer = importlib.import_module('explicit.analyze_errors')
    except Exception:
        analyzer = None

    if analyzer is not None:
        roots = []
        for r in rows:
            if r.get('approach') == 'implicit':
                continue
            d = r.get('dir')
            if d:
                p = Path(d).parent
                roots.append(str(p))
        sweep_root = None
        if roots:
            from collections import Counter
            sweep_root = Path(Counter(roots).most_common(1)[0][0])

        by_L_then_n_blocks = defaultdict(lambda: defaultdict(list))
        if sweep_root and sweep_root.exists():
            # Prefer results/ where sweep writes outputs; fall back to flat/
            results_dir = sweep_root / 'results'
            flat_dir = sweep_root / 'flat'
            for r in rows:
                if r.get('approach') == 'implicit':
                    continue
                try:
                    n = int(r['n']); L = int(r.get('L') or 0)
                    tag_dir = Path(r.get('dir',''))
                    tag = tag_dir.name if tag_dir.name else ''
                    # Try results/ first, else flat/
                    results_path = results_dir / f'results_{tag}.jsonl'
                    if not results_path.exists():
                        results_path = flat_dir / f'results_{tag}.jsonl'
                    if not results_path.exists():
                        continue
                    fn_correct = 0; fn1_chain = 0; fn2_chain = 0; total = 0
                    with results_path.open('r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            rec = __import__('json').loads(line)
                            a = analyzer.analyze_record(rec)
                            fn_correct += int(bool(a.get('pred_fn_correct')))
                            fn1_chain += int(bool(a.get('pred_fn1_from_chain')))
                            fn2_chain += int(bool(a.get('pred_fn2_from_chain')))
                            total += 1
                    if total > 0:
                        by_L_then_n_blocks[L][n].append((fn_correct/total, fn1_chain/total, fn2_chain/total))
                except Exception:
                    continue

        if by_L_then_n_blocks:
            for L in sorted(by_L_then_n_blocks.keys()):
                ns = sorted(by_L_then_n_blocks[L].keys())
                em_means = []
                fn_means = []
                fn1_means = []
                fn2_means = []
                em_by_n = defaultdict(list)
                for r in rows:
                    if r.get('approach') == 'implicit':
                        continue
                    try:
                        n = int(r['n']); LL = int(r.get('L') or 0)
                        if LL != L: continue
                        em_by_n[n].append(float(r['acc']))
                    except Exception:
                        continue
                for n in ns:
                    vals = by_L_then_n_blocks[L][n]
                    if not vals:
                        continue
                    fn_avg = sum(v[0] for v in vals)/len(vals)
                    fn1_avg = sum(v[1] for v in vals)/len(vals)
                    fn2_avg = sum(v[2] for v in vals)/len(vals)
                    em_avg = (sum(em_by_n[n])/len(em_by_n[n])) if em_by_n.get(n) else 0.0
                    em_means.append(em_avg)
                    fn_means.append(fn_avg)
                    fn1_means.append(fn1_avg)
                    fn2_means.append(fn2_avg)

                import numpy as np
                x = np.arange(len(ns))
                width = 0.2

                plt.figure(figsize=(12, 5))
                plt.bar(x - 1.5*width, em_means, width, label='EM')
                plt.bar(x - 0.5*width, fn_means, width, label='f_n correct')
                plt.bar(x + 0.5*width, fn1_means, width, label='f_{n-1} coherent')
                plt.bar(x + 1.5*width, fn2_means, width, label='f_{n-2} from-chain')
                plt.xticks(x, ns)
                plt.yticks([i/10 for i in range(0,11)])
                plt.ylim(0,1)
                plt.xlabel('n (hops)')
                plt.ylabel('Rate')
                plt.title(f'Explicit (L={L}): EM and per-block correctness by n')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(outdir, f'bars_explicit_per_block_vs_n_L{L}.png'))
                plt.close()


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
        a = next((r.get('approach') for r in rows if r.get('approach')), 'explicit')
        approach = a

    if approach == 'implicit':
        plot_implicit(rows, args.outdir)
    elif approach == 'explicit':
        plot_explicit(rows, args.outdir)
    else:
        plot_implicit(rows, args.outdir)
        plot_explicit(rows, args.outdir)

    print(f"Plots written to: {args.outdir}")


if __name__ == '__main__':
    main()



