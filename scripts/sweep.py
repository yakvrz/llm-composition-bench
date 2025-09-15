#!/usr/bin/env python3
"""
Parameter sweep runner for the symbolic n-hop benchmark.

Generates datasets and evaluates across combinations.
Writes unified run layout:
  runs/<approach>/sweep_<timestamp>/
    - data/
    - results/
    - plots/  (populated by scripts/plot_sweep.py)
    - summary.csv  (aggregate across combos)
"""

import argparse, csv, datetime as dt, itertools, json, os, subprocess, sys
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
                if obj.get('em'): correct += 1
            except Exception:
                pass
    acc = 0.0 if total == 0 else correct / total
    return correct, total, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', type=int, default=60)
    ap.add_argument('--approach', type=str, default='explicit', choices=['explicit', 'implicit'])
    ap.add_argument('--hops', type=str, default='4')
    ap.add_argument('--k', type=int, default=None)
    ap.add_argument('--k0s', type=str, default='6')
    ap.add_argument('--k1s', type=str, default='6')
    ap.add_argument('--k2s', type=str, default='6')
    ap.add_argument('--Ls', type=str, default='3')
    ap.add_argument('--contexts', type=str, default='8')
    ap.add_argument('--m_list', type=str, default='6', help='implicit only: list of m (number of chains) values')
    ap.add_argument('--M', type=int, default=512)
    ap.add_argument('--seeds', type=str, default='7')
    ap.add_argument('--model', type=str, default='gpt-4.1-mini')
    ap.add_argument('--temp', type=float, default=0.0)
    ap.add_argument('--max_output_tokens', type=int, default=16)
    ap.add_argument('--log_first', type=int, default=0)
    ap.add_argument('--save_prompt', action='store_true')
    ap.add_argument('--save_raw_output', action='store_true')
    # Extras
    ap.add_argument('--order_trials', type=int, default=1)
    ap.add_argument('--baseline', type=str, default='', choices=['', 'pointer_f_n1'])
    ap.add_argument('--ablate_inner', action='store_true')
    ap.add_argument('--ablate_hop', type=int, default=2)
    ap.add_argument('--path_collision_stress', action='store_true')
    ap.add_argument('--block_head_balance', action='store_true')
    ap.add_argument('--alias_heads_per_block', action='store_true')
    # Evaluator concurrency controls
    ap.add_argument('--concurrency', type=int, default=4)
    ap.add_argument('--max_retries', type=int, default=3)
    ap.add_argument('--retry_backoff', type=float, default=1.0)
    args = ap.parse_args()

    hops_list = parse_list(args.hops)
    k0_list = parse_list(args.k0s)
    k1_list = parse_list(args.k1s)
    k2_list = parse_list(args.k2s)
    L_list = parse_list(args.Ls)
    contexts_list = parse_list(args.contexts)
    m_list = parse_list(args.m_list)
    seeds_list = parse_list(args.seeds)

    ts = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_root = Path('runs') / args.approach / f'sweep_{ts}'
    ensure_dir(run_root)
    data_dir = run_root / 'data'
    results_dir = run_root / 'results'
    ensure_dir(data_dir)
    ensure_dir(results_dir)

    rows = []
    if args.approach == 'explicit':
        combos = list(itertools.product(hops_list, L_list, k0_list, k1_list, k2_list, contexts_list, seeds_list))
    else:
        combos = list(itertools.product(hops_list, m_list, seeds_list))
    total_combos = len(combos)
    print(f"[sweep] Starting sweep ({args.approach}) with {total_combos} combinations...", flush=True)
    for idx, combo in enumerate(combos, start=1):
        if args.approach == 'explicit':
            n, Lval, k0, k1, k2, ctx, seed = combo
            tag = f"exp_n{n}_L{Lval}_k0{k0}_k1{k1}_k2{k2}_c{ctx}_s{seed}"
        else:
            n, mval, seed = combo
            tag = f"imp_n{n}_m{mval}_s{seed}"
        out_dir = run_root / tag
        ds_path = data_dir / f"synth_{tag}.jsonl"
        res_path = results_dir / f"results_{tag}.jsonl"
        sum_path = results_dir / f"summary_{tag}.txt"

        # 1) Generate
        print(f"[sweep] [{idx}/{total_combos}] GEN {tag} ...", flush=True)
        if args.approach == 'explicit':
            gen_cmd = [
                sys.executable, 'explicit/generate.py',
                '--items', str(args.items), '--hops', str(n), '--L', str(Lval),
            ]
            if args.k is not None:
                gen_cmd += ['--k', str(args.k)]
            else:
                gen_cmd += ['--k0', str(k0), '--k1', str(k1), '--k2', str(k2)]
            gen_cmd += [
                '--context_chains', str(ctx), '--M', str(args.M), '--seed', str(seed), '--out', str(ds_path),
            ]
            if args.path_collision_stress:
                gen_cmd += ['--path_collision_stress']
            if args.block_head_balance:
                gen_cmd += ['--block_head_balance']
            if args.alias_heads_per_block:
                gen_cmd += ['--alias_heads_per_block']
        else:
            gen_cmd = [
                sys.executable, 'implicit/generate.py',
                '--items', str(args.items), '--hops', str(n), '--m', str(mval),
                '--M', str(args.M), '--seed', str(seed), '--out', str(ds_path),
            ]
            if args.ablate_inner:
                gen_cmd += ['--ablate_inner', '--ablate_hop', str(args.ablate_hop)]
        rc = stream_cmd(gen_cmd)
        if rc != 0:
            print(f"[sweep] [{idx}/{total_combos}] GEN FAIL {tag}", flush=True)
            continue

        # 2) Evaluate
        print(f"[sweep] [{idx}/{total_combos}] EVAL {tag} ...", flush=True)
        if args.approach == 'explicit':
            eval_cmd = [
                sys.executable, 'explicit/evaluate.py', '--in', str(ds_path), '--out', str(res_path),
                '--model', args.model, '--temp', str(args.temp), '--max_output_tokens', str(args.max_output_tokens), '--n', str(args.items), '--order_trials', str(args.order_trials),
                '--concurrency', str(args.concurrency), '--max_retries', str(args.max_retries), '--retry_backoff', str(args.retry_backoff)
            ]
        else:
            eval_cmd = [
                sys.executable, 'implicit/evaluate.py', '--in', str(ds_path), '--out', str(res_path),
                '--model', args.model, '--temp', str(args.temp), '--max_output_tokens', str(args.max_output_tokens), '--n', str(args.items), '--order_trials', str(args.order_trials),
                '--concurrency', str(args.concurrency), '--max_retries', str(args.max_retries), '--retry_backoff', str(args.retry_backoff)
            ]
            if args.baseline:
                eval_cmd += ['--baseline', args.baseline]
        if args.save_prompt: eval_cmd.append('--save_prompt')
        if args.save_raw_output: eval_cmd.append('--save_raw_output')
        if args.log_first and args.log_first > 0:
            eval_cmd.extend(['--log_first', str(args.log_first)])

        rc = stream_cmd(eval_cmd, log_path=sum_path, env=os.environ.copy())
        correct, total, acc = compute_em(res_path)
        if args.approach == 'explicit':
            rows.append({'approach': 'explicit', 'n': n, 'L': Lval, 'k0': k0, 'k1': k1, 'k2': k2, 'k': (args.k if args.k is not None else ''), 'contexts': ctx, 'm': '', 'seed': seed, 'items': total, 'correct': correct, 'acc': f"{acc:.3f}", 'dir': str(out_dir)})
        else:
            rows.append({'approach': 'implicit', 'n': n, 'L': '', 'k0': '', 'k1': '', 'k2': '', 'k': '', 'contexts': '', 'm': mval, 'seed': seed, 'items': total, 'correct': correct, 'acc': f"{acc:.3f}", 'dir': str(out_dir)})
        print(f"[sweep] [{idx}/{total_combos}] DONE {tag}: {correct}/{total} = {acc:.3f}", flush=True)

    # Write CSV summary
    csv_path = run_root / 'summary.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['approach','n','L','k0','k1','k2','k','contexts','m','seed','items','correct','acc','dir'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[sweep] Sweep complete. Summary: {csv_path}", flush=True)


if __name__ == '__main__':
    main()



