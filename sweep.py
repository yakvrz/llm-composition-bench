#!/usr/bin/env python3
"""
Parameter sweep runner for the symbolic n-hop benchmark.

Generates datasets with gen_synth.py and evaluates with eval.py
across combinations of hops (n), candidate sizes (k0,k1,k2),
number of context chains, seeds, and items. Summarizes EM.

Usage examples:
  python sweep.py --items 60 --hops 4,5,6 --k0s 6 --k1s 6 --k2s 6 --contexts 8
  python sweep.py --items 60 --hops 5 --k0s 4,6,8 --k1s 4,6,8 --k2s 4,6,8 --contexts 8
"""

import argparse, csv, datetime as dt, itertools, json, os, subprocess, sys
from pathlib import Path


def parse_list(arg: str) -> list[int]:
    return [int(x) for x in str(arg).split(',') if str(x).strip()]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str], env: dict | None = None):
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    return res.returncode, res.stdout


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
    ap.add_argument('--model', type=str, default='gpt-4.1')
    ap.add_argument('--temp', type=float, default=0.0)
    ap.add_argument('--max_output_tokens', type=int, default=16)
    ap.add_argument('--log_first', type=int, default=0)
    ap.add_argument('--save_prompt', action='store_true')
    ap.add_argument('--save_raw_output', action='store_true')
    ap.add_argument('--root', type=str, default='runs')
    ap.add_argument('--flat_outputs', action='store_true', help='write outputs in a single summary dir with tagged filenames')
    # Extras
    ap.add_argument('--order_trials', type=int, default=1)
    ap.add_argument('--baseline', type=str, default='', choices=['', 'pointer_f_n1'])
    ap.add_argument('--ablate_inner', action='store_true')
    ap.add_argument('--ablate_hop', type=int, default=2)
    ap.add_argument('--path_collision_stress', action='store_true')
    ap.add_argument('--block_head_balance', action='store_true')
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
    root = Path(args.root) / f'sweep_{ts}'
    ensure_dir(root)
    if args.flat_outputs:
        flat_dir = root / 'flat'
        ensure_dir(flat_dir)
    data_dir = Path('data')
    ensure_dir(data_dir)

    rows = []
    if args.approach == 'explicit':
        combos = list(itertools.product(hops_list, L_list, k0_list, k1_list, k2_list, contexts_list, seeds_list))
    else:
        combos = list(itertools.product(hops_list, m_list, seeds_list))
    total_combos = len(combos)
    print(f"Starting sweep ({args.approach}) with {total_combos} combinations...", flush=True)
    for idx, combo in enumerate(combos, start=1):
        if args.approach == 'explicit':
            n, Lval, k0, k1, k2, ctx, seed = combo
            tag = f"exp_n{n}_L{Lval}_k0{k0}_k1{k1}_k2{k2}_c{ctx}_s{seed}"
        else:
            n, mval, seed = combo
            tag = f"imp_n{n}_m{mval}_s{seed}"
        out_dir = root / tag
        if not args.flat_outputs:
            ensure_dir(out_dir)
        ds_path = data_dir / f"synth_{tag}.jsonl"
        if args.flat_outputs:
            res_path = (flat_dir / f"results_{tag}.jsonl")
            sum_path = (flat_dir / f"summary_{tag}.txt")
        else:
            res_path = out_dir / 'results.jsonl'
            sum_path = out_dir / 'summary.txt'

        # 1) Generate
        print(f"[{idx}/{total_combos}] GEN {tag} ...", flush=True)
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
        else:
            gen_cmd = [
                sys.executable, 'implicit/generate.py',
                '--items', str(args.items), '--hops', str(n), '--m', str(mval),
                '--M', str(args.M), '--seed', str(seed), '--out', str(ds_path),
            ]
            if args.ablate_inner:
                gen_cmd += ['--ablate_inner', '--ablate_hop', str(args.ablate_hop)]
        rc, out = run_cmd(gen_cmd)
        if args.flat_outputs:
            (flat_dir / f'gen_stdout_{tag}.txt').write_text(out, encoding='utf-8')
        else:
            (out_dir / 'gen_stdout.txt').write_text(out, encoding='utf-8')
        if rc != 0:
            print(f"[{idx}/{total_combos}] GEN FAIL {tag}", flush=True)
            continue

        # 2) Evaluate
        print(f"[{idx}/{total_combos}] EVAL {tag} ...", flush=True)
        if args.approach == 'explicit':
            eval_cmd = [
                sys.executable, 'explicit/evaluate.py', '--in', str(ds_path), '--out', str(res_path),
                '--model', args.model, '--temp', str(args.temp), '--max_output_tokens', str(args.max_output_tokens), '--n', str(args.items), '--order_trials', str(args.order_trials),
            ]
        else:
            eval_cmd = [
                sys.executable, 'implicit/evaluate.py', '--in', str(ds_path), '--out', str(res_path),
                '--model', args.model, '--temp', str(args.temp), '--max_output_tokens', str(args.max_output_tokens), '--n', str(args.items), '--order_trials', str(args.order_trials),
            ]
            if args.baseline:
                eval_cmd += ['--baseline', args.baseline]
        if args.save_prompt: eval_cmd.append('--save_prompt')
        if args.save_raw_output: eval_cmd.append('--save_raw_output')
        if args.log_first and args.log_first > 0:
            eval_cmd.extend(['--log_first', str(args.log_first)])

        rc, out = run_cmd(eval_cmd, env=os.environ.copy())
        sum_path.write_text(out, encoding='utf-8')
        correct, total, acc = compute_em(res_path)
        if args.approach == 'explicit':
            rows.append({'approach': 'explicit', 'n': n, 'L': Lval, 'k0': k0, 'k1': k1, 'k2': k2, 'k': (args.k if args.k is not None else ''), 'contexts': ctx, 'm': '', 'seed': seed, 'items': total, 'correct': correct, 'acc': f"{acc:.3f}", 'dir': str(out_dir)})
        else:
            rows.append({'approach': 'implicit', 'n': n, 'L': '', 'k0': '', 'k1': '', 'k2': '', 'k': '', 'contexts': '', 'm': mval, 'seed': seed, 'items': total, 'correct': correct, 'acc': f"{acc:.3f}", 'dir': str(out_dir)})
        print(f"[{idx}/{total_combos}] DONE {tag}: {correct}/{total} = {acc:.3f}", flush=True)

    # Write CSV summary
    csv_path = root / 'summary.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['approach','n','L','k0','k1','k2','k','contexts','m','seed','items','correct','acc','dir'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nSweep complete. Summary: {csv_path}", flush=True)


if __name__ == '__main__':
    main()


