#!/usr/bin/env python3
"""
Run diagnostics for a single parameter combination (no sweep).

Creates a run under runs/<approach>/diag_<timestamp>/ with:
  - data/
  - results/
  - plots/
  - results.csv (single-row)
  - summary.json (structured report)

Flags differ by approach:
- explicit: --n, --L, --k (or k0/k1/k2), --contexts, --path_collision_stress, --block_head_balance, --alias_heads_per_block
- implicit: --n, --m, --baseline, --ablate_inner, --ablate_hop

Common: --items, --M, --seed, --model, --temp, --max_output_tokens, --order_trials, --concurrency, --max_retries, --retry_backoff
"""
import argparse, csv, datetime as dt, json, os, subprocess, sys
from pathlib import Path


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def stream_cmd(cmd: list[str], log_path: Path | None = None, env: dict | None = None) -> int:
    print(f"[diag] RUN: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1)
    f = None
    try:
        if log_path is not None:
            ensure_dir(log_path.parent)
            f = log_path.open('w', encoding='utf-8')
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end='')
            if f is not None:
                f.write(line)
                f.flush()
        proc.wait()
        return proc.returncode or 0
    finally:
        if f is not None:
            f.close()


def compute_em(results_path: Path) -> tuple[int, int, float]:
    total = 0
    correct = 0
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
                if obj.get('em'):
                    correct += 1
            except Exception:
                pass
    acc = 0.0 if total == 0 else correct / total
    return correct, total, acc


def write_summary(summary_path: Path, row: dict):
    fields = ['approach','n','L','k0','k1','k2','k','contexts','m','seed','items','correct','acc','dir']
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(row)


def write_results_json(run_root: Path, approach: str, rows: list[dict], args):
    import json as _json
    report = {
        'approach': approach,
        'tag': run_root.name,
        'results_csv': (run_root / 'results.csv').as_posix(),
        'plots_dir': (run_root / 'plots').as_posix(),
        'params': {
            'items': args.items,
            'n': args.n,
            'M': args.M,
            'seed': args.seed,
            'model': args.model,
            'temp': args.temp,
            'max_output_tokens': args.max_output_tokens,
            'order_trials': args.order_trials,
            'concurrency': args.concurrency,
            'max_retries': args.max_retries,
            'retry_backoff': args.retry_backoff,
        },
        'variants': rows,
    }
    (run_root / 'summary.json').write_text(_json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--approach', choices=['explicit','implicit'], required=True)
    ap.add_argument('--items', type=int, default=60)
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--M', type=int, default=512)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--model', type=str, default='gpt-4.1-mini')
    ap.add_argument('--temp', type=float, default=0.0)
    ap.add_argument('--max_output_tokens', type=int, default=16)
    ap.add_argument('--order_trials', type=int, default=1)
    ap.add_argument('--concurrency', type=int, default=4)
    ap.add_argument('--max_retries', type=int, default=3)
    ap.add_argument('--retry_backoff', type=float, default=1.0)

    # explicit-specific
    ap.add_argument('--L', type=int)
    ap.add_argument('--k', type=int)
    ap.add_argument('--k0', type=int, default=6)
    ap.add_argument('--k1', type=int, default=6)
    ap.add_argument('--k2', type=int, default=6)
    ap.add_argument('--contexts', type=int, default=8)
    ap.add_argument('--path_collision_stress', action='store_true')
    ap.add_argument('--block_head_balance', action='store_true')
    ap.add_argument('--alias_heads_per_block', action='store_true')

    # implicit-specific
    ap.add_argument('--m', type=int)
    ap.add_argument('--baseline', type=str, default='', choices=['','pointer_f_n1'])
    ap.add_argument('--ablate_inner', action='store_true')
    ap.add_argument('--ablate_hop', type=int, default=2)

    # multi-variant diagnostics in one run
    ap.add_argument('--variants', type=str, default='', help='Comma-separated variant keys to run in one folder. explicit: control,path,bal,alias,all. implicit: control,ablate,baseline')

    args = ap.parse_args()

    ts = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_root = Path('runs') / args.approach / f'diag_{ts}'
    data_dir = run_root / 'data'
    results_dir = run_root / 'results'
    plots_dir = run_root / 'plots'
    ensure_dir(data_dir); ensure_dir(results_dir); ensure_dir(plots_dir)

    # Prepare variant list
    variants = [v.strip() for v in (args.variants.split(',') if args.variants else []) if v.strip()]
    if not variants:
        variants = ['control']
        if args.approach == 'explicit':
            variants += ['path','bal','alias']
        else:
            variants += ['ablate','baseline']

    rows = []
    for var in variants:
        if args.approach == 'explicit':
            assert args.L is not None, '--L is required for explicit'
            enable_path = (var == 'path')
            enable_bal = (var == 'bal')
            enable_alias = (var == 'alias')
            if var == 'all':
                enable_path = enable_bal = enable_alias = True
            tag = f"exp_n{args.n}_L{args.L}_k{args.k or 'var'}_c{args.contexts}_s{args.seed}_{var}"
            ds = data_dir / f'synth_{tag}.jsonl'
            rs = results_dir / f'results_{tag}.jsonl'
            gen_cmd = [sys.executable, 'explicit/generate.py', '--items', str(args.items), '--hops', str(args.n), '--L', str(args.L), '--context_chains', str(args.contexts), '--M', str(args.M), '--seed', str(args.seed), '--out', str(ds)]
            if args.k is not None:
                gen_cmd += ['--k', str(args.k)]
            else:
                gen_cmd += ['--k0', str(args.k0), '--k1', str(args.k1), '--k2', str(args.k2)]
            if enable_path: gen_cmd.append('--path_collision_stress')
            if enable_bal: gen_cmd.append('--block_head_balance')
            if enable_alias: gen_cmd.append('--alias_heads_per_block')
            rc = stream_cmd(gen_cmd)
            if rc != 0: sys.exit(rc)
            eval_cmd = [sys.executable, 'explicit/evaluate.py', '--in', str(ds), '--out', str(rs), '--model', args.model, '--temp', str(args.temp), '--max_output_tokens', str(args.max_output_tokens), '--n', str(args.items), '--order_trials', str(args.order_trials), '--concurrency', str(args.concurrency), '--max_retries', str(args.max_retries), '--retry_backoff', str(args.retry_backoff)]
            stream_cmd(eval_cmd)
            correct, total, acc = compute_em(rs)
            rows.append({'approach':'explicit','variant':var,'n':str(args.n),'L':str(args.L),'k0':str(args.k0),'k1':str(args.k1),'k2':str(args.k2),'k':str(args.k or ''),'contexts':str(args.contexts),'m':'','seed':str(args.seed),'items':str(total),'correct':str(correct),'acc':f"{acc:.3f}",'dir': str(run_root / tag)})
        else:
            assert args.m is not None, '--m is required for implicit'
            use_ablate = (var == 'ablate')
            use_baseline = (var == 'baseline')
            tag = f"imp_n{args.n}_m{args.m}_s{args.seed}_{var}"
            ds = data_dir / f'synth_{tag}.jsonl'
            rs = results_dir / f'results_{tag}.jsonl'
            gen_cmd = [sys.executable, 'implicit/generate.py', '--items', str(args.items), '--hops', str(args.n), '--m', str(args.m), '--M', str(args.M), '--seed', str(args.seed), '--out', str(ds)]
            if use_ablate: gen_cmd += ['--ablate_inner', '--ablate_hop', str(args.ablate_hop)]
            rc = stream_cmd(gen_cmd)
            if rc != 0: sys.exit(rc)
            eval_cmd = [sys.executable, 'implicit/evaluate.py', '--in', str(ds), '--out', str(rs), '--model', args.model, '--temp', str(args.temp), '--max_output_tokens', str(args.max_output_tokens), '--n', str(args.items), '--order_trials', str(args.order_trials), '--concurrency', str(args.concurrency), '--max_retries', str(args.max_retries), '--retry_backoff', str(args.retry_backoff)]
            if use_baseline: eval_cmd += ['--baseline', 'pointer_f_n1']
            stream_cmd(eval_cmd)
            correct, total, acc = compute_em(rs)
            rows.append({'approach':'implicit','variant':var,'n':str(args.n),'L':'','k0':'','k1':'','k2':'','k':'','contexts':'','m':str(args.m),'seed':str(args.seed),'items':str(total),'correct':str(correct),'acc':f"{acc:.3f}",'dir': str(run_root / tag)})

    # Write results with variants
    fields = ['approach','variant','n','L','k0','k1','k2','k','contexts','m','seed','items','correct','acc','dir']
    with open(run_root / 'results.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Create simple variant comparison plots
    import matplotlib.pyplot as plt
    labels = [r['variant'] for r in rows]
    ems = [float(r['acc']) for r in rows]
    plt.figure(figsize=(8,4))
    plt.bar(labels, ems)
    plt.ylim(0,1)
    plt.ylabel('EM')
    plt.title(f"{args.approach.capitalize()} diagnostics EM by variant (n={args.n}{', L='+str(args.L) if args.approach=='explicit' else ', m='+str(args.m)})")
    plt.tight_layout()
    plt.savefig(plots_dir / 'bars_em_by_variant.png')
    plt.close()

    # Lift over chance by variant
    lifts = []
    for r in rows:
        em = float(r['acc'])
        if args.approach == 'explicit':
            k_last = int(args.k or args.k2 or 6)
            chance = 1.0 / max(1, k_last)
        else:
            chance = 1.0 / max(1, int(args.m))
        lift = (em - chance) / (1.0 - chance) if (1.0 - chance) > 0 else 0.0
        lifts.append(lift)
    plt.figure(figsize=(8,4))
    plt.bar(labels, lifts)
    plt.ylim(0,1)
    plt.ylabel('Lift over chance')
    plt.title(f"{args.approach.capitalize()} diagnostics lift by variant (n={args.n}{', L='+str(args.L) if args.approach=='explicit' else ', m='+str(args.m)})")
    plt.tight_layout()
    plt.savefig(plots_dir / 'bars_lift_by_variant.png')
    plt.close()

    # Remove any stale results.md and write JSON
    md_path = run_root / 'results.md'
    try:
        if md_path.exists():
            md_path.unlink()
    except Exception:
        pass
    write_results_json(run_root, args.approach, rows, args)
    print(f"Diagnostics complete. Summary: {run_root/'results.csv'}\nPlots: {plots_dir}\nReport: {run_root/'summary.json'}")


if __name__ == '__main__':
    main()
