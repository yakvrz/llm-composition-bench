#!/usr/bin/env python3
"""
Error analysis for the n-hop composition benchmark (bag-of-facts).

Reads all results JSONL files under a sweep root's results/ directory and reports:
- Overall and per-n accuracy
- Non-EM breakdown: other-chain final tail vs other in-facts token vs OOV
- Pred token layer (L0..L_{n-1}) when derivable from facts

Usage:
  python scripts/analyze_errors.py --sweep_root runs/benchmark/sweep_YYYYMMDD_HHMMSS
  or
  python scripts/analyze_errors.py --results_csv runs/benchmark/sweep_.../results.csv
"""
import argparse, csv, json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set


def iter_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def parse_start_from_question(q: str) -> str:
    if not q:
        return ""
    s = q.strip().rstrip('?')
    parts = s.split(' of ')
    return parts[-1].strip() if parts else ""


def build_relation_maps(facts_bag: List[List[str]]) -> Dict[str, Dict[str, str]]:
    rel_maps: Dict[str, Dict[str, str]] = {}
    for h, r, t in facts_bag or []:
        rel_maps.setdefault(r, {})[h] = t
    return rel_maps


def derive_layers_from_facts(n: int, facts_bag: List[List[str]]) -> List[Set[str]]:
    """Approximate layers L0..L_{n-1} from facts by using heads/tails of f1..f_{n-1}."""
    layers: List[Set[str]] = [set() for _ in range(max(1, n))]
    if n <= 0:
        return layers
    # L0 are heads of f1
    for h, r, t in facts_bag or []:
        if r == 'f1':
            layers[0].add(h)
    # Li are tails of f_i
    for i in range(1, n):
        ri = f'f{i}'
        for h, r, t in facts_bag or []:
            if r == ri:
                layers[i].add(t)
    return layers


def analyze_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    n = int(rec.get('n') or 0)
    em = bool(rec.get('em'))
    pred = (rec.get('pred') or '').strip()
    gold_list = rec.get('gold') or []
    gold = gold_list[0] if gold_list else ''
    facts: List[List[str]] = rec.get('facts_bag') or []
    q = rec.get('question') or ''

    # Build helpers
    rel_maps = build_relation_maps(facts)
    start = parse_start_from_question(q)

    # Follow f1..f_{n-2} to get y_{n-2}
    y_nm2 = ''
    cur = start
    for hop_idx in range(1, max(1, n - 1)):
        r = f'f{hop_idx}'
        cur = rel_maps.get(r, {}).get(cur, '')
        if not cur:
            break
    y_nm2 = cur if n >= 2 else ''

    # Collect final-hop tails and all fact tokens
    final_rel = f'f{n-1}' if n >= 2 else 'f1'
    final_tails = {t for h, r, t in facts if r == final_rel}
    all_fact_tokens = {tok for h, r, t in facts for tok in (h, t)}

    # Categorize non-EM errors
    error_type = ''
    if not em:
        if pred in final_tails and pred != gold:
            error_type = 'other_chain_final_tail'
        elif pred in all_fact_tokens:
            error_type = 'in_facts_non_final_tail'
        else:
            error_type = 'oov_token'

    # Pred layer index if derivable
    layers = derive_layers_from_facts(n, facts)
    pred_layer = None
    for i, layer_set in enumerate(layers):
        if pred in layer_set:
            pred_layer = i
            break

    return {
        'n': n,
        'em': em,
        'pred': pred,
        'gold': gold,
        'error_type': error_type,
        'pred_in_final_tails': pred in final_tails,
        'pred_in_facts': pred in all_fact_tokens,
        'pred_layer': pred_layer,
    }


def load_results_paths(sweep_root: Path | None, results_csv: Path | None) -> List[Path]:
    paths: List[Path] = []
    if results_csv and results_csv.exists():
        with results_csv.open('r', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                d = r.get('dir') or ''
                if not d:
                    continue
                tag_dir = Path(d)
                res_dir = tag_dir.parent / 'results'
                # results_<tag>.jsonl
                res_path = res_dir / f"results_{tag_dir.name}.jsonl"
                if res_path.exists():
                    paths.append(res_path)
        return paths

    if sweep_root and sweep_root.exists():
        res_dir = sweep_root / 'results'
        if res_dir.exists():
            for p in sorted(res_dir.glob('results_*.jsonl')):
                paths.append(p)
    return paths


def summarize(analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    from collections import defaultdict, Counter
    total = len(analyses)
    acc = sum(1 for a in analyses if a['em'])
    by_n = defaultdict(lambda: {'n': 0, 'em': 0})
    for a in analyses:
        k = int(a['n'])
        by_n[k]['n'] += 1
        by_n[k]['em'] += int(bool(a['em']))
    err_types = Counter(a['error_type'] for a in analyses if not a['em'])
    # Pred layer distribution for non-EM
    pred_layers = Counter(a['pred_layer'] for a in analyses if not a['em'])
    return {
        'total': total,
        'acc': (0.0 if total == 0 else acc / total),
        'by_n': {k: {'n': v['n'], 'acc': (0.0 if v['n'] == 0 else v['em'] / v['n'])} for k, v in sorted(by_n.items())},
        'non_em_by_type': err_types,
        'non_em_pred_layer_counts': pred_layers,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep_root', type=str, default='')
    ap.add_argument('--results_csv', type=str, default='')
    ap.add_argument('--examples_per_type', type=int, default=0)
    args = ap.parse_args()

    sweep_root = Path(args.sweep_root) if args.sweep_root else None
    results_csv = Path(args.results_csv) if args.results_csv else None
    paths = load_results_paths(sweep_root, results_csv)
    if not paths:
        print('No results JSONL files found.')
        return

    recs: List[Dict[str, Any]] = []
    for p in paths:
        recs.extend(iter_jsonl(p))
    analyses = [analyze_record(r) for r in recs]
    summary = summarize(analyses)

    print('=== Benchmark Error Analysis ===')
    print(f"TOTAL: {summary['total']}  ACC: {summary['acc']:.3f}")
    print('By n:')
    for k, v in summary['by_n'].items():
        print(f"  n={k}: {v['n']}  ACC={v['acc']:.3f}")
    print('\nNon-EM error types:')
    for t, c in summary['non_em_by_type'].most_common():
        print(f"  {t or '—'}: {c}")
    print('\nNon-EM pred layer (approx from facts):')
    for layer, c in sorted(summary['non_em_pred_layer_counts'].items(), key=lambda x: (x[0] is None, x[0])):
        label = 'unknown' if layer is None else f'L{layer}'
        print(f"  {label}: {c}")

    if args.examples_per_type > 0:
        from collections import defaultdict
        buckets = defaultdict(list)
        for idx, a in enumerate(analyses):
            if not a['em']:
                buckets[a['error_type']].append(idx)
        for t, idxs in buckets.items():
            print(f"\n-- {t or '—'} --")
            for j in idxs[: args.examples_per_type]:
                print(json.dumps(recs[j], ensure_ascii=False))


if __name__ == '__main__':
    main()
