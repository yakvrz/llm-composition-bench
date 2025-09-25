#!/usr/bin/env python3
"""Run diagnostics for a single (n, m) configuration."""

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


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


def write_summary_csv(run_root: Path, rows: list[dict]):
    fields = ['variant','n','m','seed','items','correct','acc','dir']
    with open(run_root / 'results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_json(run_root: Path, args, rows: list[dict]):
    payload = {
        'approach': 'benchmark',
        'tag': run_root.name,
        'results_csv': (run_root / 'results.csv').as_posix(),
        'plots_dir': (run_root / 'plots').as_posix(),
        'params': {
            'items': args.items,
            'n': args.n,
            'm': args.m,
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
    (run_root / 'summary.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding='utf-8')


def plot_variant_bars(run_root: Path, rows: list[dict], m: int, n: int):
    plots_dir = run_root / 'plots'
    ensure_dir(plots_dir)
    labels = [r['variant'] for r in rows]
    ems = [float(r['acc']) for r in rows]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, ems)
    plt.ylim(0, 1)
    plt.ylabel('EM')
    plt.title(f'Benchmark diagnostics EM by variant (n={n}, m={m})')
    plt.tight_layout()
    plt.savefig(plots_dir / 'bars_em_by_variant.png')
    plt.close()

    lifts = []
    chance = 1.0 / max(1, m)
    for em in ems:
        lifts.append((em - chance) / (1.0 - chance) if (1.0 - chance) > 0 else 0.0)
    plt.figure(figsize=(8, 4))
    plt.bar(labels, lifts)
    plt.ylim(0, 1)
    plt.ylabel('Lift over chance')
    plt.title(f'Benchmark diagnostics lift by variant (n={n}, m={m})')
    plt.tight_layout()
    plt.savefig(plots_dir / 'bars_lift_by_variant.png')
    plt.close()


def main():
    ap = argparse.ArgumentParser(description='Run n-hop diagnostics')
    ap.add_argument('--items', type=int, default=60)
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--m', type=int, required=True)
    ap.add_argument('--M', type=int, default=512)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--model', type=str, default='gpt-4.1-mini')
    ap.add_argument('--temp', type=float, default=0.0)
    ap.add_argument('--max_output_tokens', type=int, default=16)
    ap.add_argument('--order_trials', type=int, default=1)
    ap.add_argument('--concurrency', type=int, default=4)
    ap.add_argument('--max_retries', type=int, default=3)
    ap.add_argument('--retry_backoff', type=float, default=1.0)
    ap.add_argument('--ablate_hop', type=int, default=2)
    ap.add_argument('--reasoning_steps', action='store_true', help='insert structured reasoning slots before final answers')
    ap.add_argument('--variants', type=str, default='control,ablate,baseline', help='Comma-separated variant keys (control, ablate, baseline)')
    args = ap.parse_args()

    ts = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_root = Path('runs') / 'benchmark' / f'diag_{ts}'
    data_dir = run_root / 'data'
    results_dir = run_root / 'results'
    ensure_dir(data_dir)
    ensure_dir(results_dir)

    variants = [v.strip() for v in args.variants.split(',') if v.strip()]
    rows = []
    for variant in variants:
        tag = f"bench_n{args.n}_m{args.m}_s{args.seed}_{variant}"
        ds_path = data_dir / f"synth_{tag}.jsonl"
        res_path = results_dir / f"results_{tag}.jsonl"
        summary_txt = results_dir / f"summary_{tag}.txt"

        gen_cmd = [
            sys.executable, 'generate.py',
            '--items', str(args.items), '--hops', str(args.n), '--m', str(args.m),
            '--M', str(args.M), '--seed', str(args.seed), '--out', str(ds_path)
        ]
        if variant == 'ablate':
            gen_cmd += ['--ablate_inner', '--ablate_hop', str(args.ablate_hop)]
        rc = stream_cmd(gen_cmd)
        if rc != 0:
            sys.exit(rc)

        eval_cmd = [
            sys.executable, 'evaluate.py', '--in', str(ds_path), '--out', str(res_path),
            '--model', args.model, '--temp', str(args.temp), '--max_output_tokens', str(args.max_output_tokens), '--n', str(args.items), '--order_trials', str(args.order_trials),
            '--concurrency', str(args.concurrency), '--max_retries', str(args.max_retries), '--retry_backoff', str(args.retry_backoff)
        ]
        if variant == 'baseline':
            eval_cmd += ['--baseline', 'pointer_f_n1']
        if args.reasoning_steps:
            eval_cmd.append('--reasoning_steps')
        rc = stream_cmd(eval_cmd, log_path=summary_txt, env=os.environ.copy())
        if rc != 0:
            sys.exit(rc)

        correct, total, acc = compute_em(res_path)
        rows.append({
            'variant': variant,
            'n': str(args.n),
            'm': str(args.m),
            'seed': str(args.seed),
            'items': str(total),
            'correct': str(correct),
            'acc': f"{acc:.3f}",
            'dir': str(run_root / tag),
        })

    write_summary_csv(run_root, rows)
    write_summary_json(run_root, args, rows)
    plot_variant_bars(run_root, rows, args.m, args.n)
    print(f"Diagnostics complete. Summary: {run_root/'results.csv'}\nPlots: {run_root/'plots'}\nReport: {run_root/'summary.json'}")


if __name__ == '__main__':
    main()
