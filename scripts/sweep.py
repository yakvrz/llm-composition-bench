#!/usr/bin/env python3
"""
Implicit-only parameter sweep runner for the symbolic n-hop benchmark.

Generates datasets and evaluates across combinations.
Writes unified run layout:
  runs/implicit/sweep_<timestamp>/
    - data/
    - results/
    - plots/  (populated by scripts/plot_sweep.py)
    - results.csv  (aggregate across combos)
"""

import argparse
import csv
import datetime as dt
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_list(arg: str) -> list[int]:
    return [int(x) for x in str(arg).split(',') if str(x).strip()]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def stream_cmd(cmd: list[str], log_path: Path | None = None, env: dict | None = None):
    print(f"[sweep] RUN: {' '.join(cmd)}", flush=True)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', type=int, default=60)
    ap.add_argument('--hops', type=str, default='4')
    ap.add_argument('--m_list', type=str, default='6', help='list of m (number of chains) values')
    ap.add_argument('--M', type=int, default=512)
    ap.add_argument('--id_width', type=int, default=4)
    ap.add_argument('--seeds', type=str, default='7')
    ap.add_argument('--model', type=str, default='gpt-4.1-mini')
    ap.add_argument('--temp', type=float, default=0.0)
    ap.add_argument('--max_output_tokens', type=int, default=16)
    ap.add_argument('--log_first', type=int, default=0)
    ap.add_argument('--save_prompt', action='store_true')
    ap.add_argument('--save_raw_output', action='store_true')
    ap.add_argument('--order_trials', type=int, default=1)
    ap.add_argument('--baseline', type=str, default='', choices=['', 'pointer_f_n1'])
    ap.add_argument('--ablate_inner', action='store_true')
    ap.add_argument('--ablate_hop', type=int, default=2)
    ap.add_argument('--concurrency', type=int, default=4)
    ap.add_argument('--max_retries', type=int, default=3)
    ap.add_argument('--retry_backoff', type=float, default=1.0)
    ap.add_argument('--reasoning_steps', action='store_true', help='insert structured reasoning slots before final answer prompts')
    args = ap.parse_args()

    hops_list = parse_list(args.hops)
    m_list = parse_list(args.m_list)
    seeds_list = parse_list(args.seeds)

    ts = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_root = Path('runs') / 'implicit' / f'sweep_{ts}'
    ensure_dir(run_root)
    data_dir = run_root / 'data'
    results_dir = run_root / 'results'
    ensure_dir(data_dir)
    ensure_dir(results_dir)

    combos = list(itertools.product(hops_list, m_list, seeds_list))
    total_combos = len(combos)
    print(f"[sweep] Starting sweep (implicit) with {total_combos} combinations...", flush=True)

    rows = []
    for idx, combo in enumerate(combos, start=1):
        n, mval, seed = combo
        tag = f"imp_n{n}_m{mval}_s{seed}"
        out_dir = run_root / tag
        ds_path = data_dir / f"synth_{tag}.jsonl"
        res_path = results_dir / f"results_{tag}.jsonl"
        sum_path = results_dir / f"summary_{tag}.txt"

        # 1) Generate
        print(f"[sweep] [{idx}/{total_combos}] GEN {tag} ...", flush=True)
        gen_cmd = [
            sys.executable, 'implicit/generate.py',
            '--items', str(args.items), '--hops', str(n), '--m', str(mval),
            '--M', str(args.M), '--id_width', str(args.id_width), '--seed', str(seed), '--out', str(ds_path),
        ]
        if args.ablate_inner:
            gen_cmd += ['--ablate_inner', '--ablate_hop', str(args.ablate_hop)]
        rc = stream_cmd(gen_cmd)
        if rc != 0:
            print(f"[sweep] [{idx}/{total_combos}] GEN FAIL {tag}", flush=True)
            continue

        # 2) Evaluate
        print(f"[sweep] [{idx}/{total_combos}] EVAL {tag} ...", flush=True)
        eval_cmd = [
            sys.executable, 'implicit/evaluate.py', '--in', str(ds_path), '--out', str(res_path),
            '--model', args.model, '--temp', str(args.temp), '--max_output_tokens', str(args.max_output_tokens), '--n', str(args.items), '--order_trials', str(args.order_trials),
            '--concurrency', str(args.concurrency), '--max_retries', str(args.max_retries), '--retry_backoff', str(args.retry_backoff)
        ]
        if args.baseline:
            eval_cmd += ['--baseline', args.baseline]
        if args.reasoning_steps:
            eval_cmd.append('--reasoning_steps')
        if args.save_prompt:
            eval_cmd.append('--save_prompt')
        if args.save_raw_output:
            eval_cmd.append('--save_raw_output')
        if args.log_first and args.log_first > 0:
            eval_cmd.extend(['--log_first', str(args.log_first)])

        rc = stream_cmd(eval_cmd, log_path=sum_path, env=os.environ.copy())
        correct, total, acc = compute_em(res_path)
        rows.append({'approach': 'implicit', 'n': n, 'm': mval, 'seed': seed, 'items': total, 'correct': correct, 'acc': f"{acc:.3f}", 'dir': str(out_dir)})
        print(f"[sweep] [{idx}/{total_combos}] DONE {tag}: {correct}/{total} = {acc:.3f}", flush=True)

    # Write CSV results
    csv_path = run_root / 'results.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['approach','n','m','seed','items','correct','acc','dir']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\n[sweep] Sweep complete. Summary: {csv_path}", flush=True)


if __name__ == '__main__':
    main()
