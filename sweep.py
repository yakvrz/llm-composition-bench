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
    ap.add_argument('--hops', type=str, default='4')
    ap.add_argument('--k0s', type=str, default='6')
    ap.add_argument('--k1s', type=str, default='6')
    ap.add_argument('--k2s', type=str, default='6')
    ap.add_argument('--contexts', type=str, default='8')
    ap.add_argument('--M', type=int, default=512)
    ap.add_argument('--seeds', type=str, default='7')
    ap.add_argument('--model', type=str, default='gpt-4.1')
    ap.add_argument('--temp', type=float, default=0.0)
    ap.add_argument('--max_output_tokens', type=int, default=16)
    ap.add_argument('--log_first', type=int, default=0)
    ap.add_argument('--save_prompt', action='store_true')
    ap.add_argument('--save_raw_output', action='store_true')
    ap.add_argument('--root', type=str, default='runs')
    args = ap.parse_args()

    hops_list = parse_list(args.hops)
    k0_list = parse_list(args.k0s)
    k1_list = parse_list(args.k1s)
    k2_list = parse_list(args.k2s)
    contexts_list = parse_list(args.contexts)
    seeds_list = parse_list(args.seeds)

    ts = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    root = Path(args.root) / f'sweep_{ts}'
    ensure_dir(root)
    data_dir = Path('data')
    ensure_dir(data_dir)

    rows = []
    for n, k0, k1, k2, ctx, seed in itertools.product(hops_list, k0_list, k1_list, k2_list, contexts_list, seeds_list):
        tag = f"n{n}_k0{k0}_k1{k1}_k2{k2}_c{ctx}_s{seed}"
        out_dir = root / tag
        ensure_dir(out_dir)
        ds_path = data_dir / f"synth_{tag}.jsonl"
        res_path = out_dir / 'results.jsonl'
        sum_path = out_dir / 'summary.txt'

        # 1) Generate
        gen_cmd = [
            sys.executable, 'generate_data.py',
            '--items', str(args.items),
            '--hops', str(n),
            '--k0', str(k0), '--k1', str(k1), '--k2', str(k2),
            '--context_chains', str(ctx),
            '--M', str(args.M),
            '--seed', str(seed),
            '--out', str(ds_path),
        ]
        rc, out = run_cmd(gen_cmd)
        (out_dir / 'gen_stdout.txt').write_text(out, encoding='utf-8')
        if rc != 0:
            print(f"GEN FAIL {tag}")
            continue

        # 2) Evaluate
        eval_cmd = [
            sys.executable, 'evaluate.py',
            '--in', str(ds_path),
            '--out', str(res_path),
            '--model', args.model,
            '--temp', str(args.temp),
            '--max_output_tokens', str(args.max_output_tokens),
            '--n', str(args.items),
        ]
        if args.save_prompt: eval_cmd.append('--save_prompt')
        if args.save_raw_output: eval_cmd.append('--save_raw_output')
        if args.log_first and args.log_first > 0:
            eval_cmd.extend(['--log_first', str(args.log_first)])

        rc, out = run_cmd(eval_cmd, env=os.environ.copy())
        sum_path.write_text(out, encoding='utf-8')
        correct, total, acc = compute_em(res_path)
        rows.append({
            'n': n, 'k0': k0, 'k1': k1, 'k2': k2, 'contexts': ctx, 'seed': seed,
            'items': total, 'correct': correct, 'acc': f"{acc:.3f}", 'dir': str(out_dir)
        })
        print(f"DONE {tag}: {correct}/{total} = {acc:.3f}")

    # Write CSV summary
    csv_path = root / 'summary.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['n','k0','k1','k2','contexts','seed','items','correct','acc','dir'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nSweep complete. Summary: {csv_path}")


if __name__ == '__main__':
    main()


