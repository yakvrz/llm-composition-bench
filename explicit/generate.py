#!/usr/bin/env python3
"""
Symbolic n-hop synthetic dataset generator (open-book, laddered candidates with multi-chain context).

This is the EXPLICIT (candidate-based) generator.
"""
import argparse, json, random, time
from typing import Dict, List, Tuple, Set


def layer_prefix(i: int) -> str:
    return chr(ord('A') + i)


def make_layer(prefix: str, size: int) -> List[str]:
    return [f"{prefix}_{j:04d}" for j in range(size)]


def permute_map(src: List[str], dst: List[str]) -> Dict[str, str]:
    shuffled = dst[:]
    random.shuffle(shuffled)
    return dict(zip(src, shuffled))


def build_bijections(hops: int, size_M: int, seed: int) -> Tuple[List[List[str]], List[Dict[str, str]]]:
    random.seed(seed)
    layers: List[List[str]] = [make_layer(layer_prefix(i), size_M) for i in range(hops + 1)]
    functions: List[Dict[str, str]] = []
    for i in range(1, hops + 1):
        src = layers[i - 1]
        dst = layers[i]
        functions.append(permute_map(src, dst))
    return layers, functions


def compose_chain(x: str, functions: List[Dict[str, str]]) -> Tuple[List[str], str]:
    intermediates: List[str] = []
    cur = x
    for f in functions:
        cur = f[cur]
        intermediates.append(cur)
    return intermediates[:-1], intermediates[-1]


def _question_laddered(hops: int, start_token: str) -> str:
    inner = " of ".join([f"f{i}" for i in range(hops - 1, 0, -1)])
    return f"What is f{hops} of {inner} of {start_token}?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=200, help="number of instances")
    ap.add_argument("--hops", type=int, default=4, help="number of hops (n)")
    ap.add_argument("--k", type=int, default=None, help="number of candidates per ladder block (overrides k0/k1/k2)")
    ap.add_argument("--k0", type=int, default=4, help="number of candidates for f_{n-2}")
    ap.add_argument("--k1", type=int, default=4, help="number of candidates for f_{n-1}")
    ap.add_argument("--k2", type=int, default=4, help="number of candidates for f_n")
    ap.add_argument("--L", type=int, default=3, help="decision depth: number of final functions with candidate sets (1..n)")
    ap.add_argument("--context_chains", type=int, default=6, help="number of distractor chains to mix into facts")
    ap.add_argument("--M", type=int, default=512, help="layer size (tokens per layer)")
    ap.add_argument("--out", type=str, default="data/synth.jsonl", help="output JSONL path")
    ap.add_argument("--seed", type=int, default=7, help="random seed")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to sleep between items")
    ap.add_argument("--chain_only_ordered", action="store_true", help="emit only the target chain facts in order (sanity check mode)")
    ap.add_argument("--path_collision_stress", action="store_true", help="ensure distractor candidates form coherent wrong paths across ladder blocks")
    args = ap.parse_args()

    layers, functions = build_bijections(args.hops, args.M, seed=args.seed)
    L0 = layers[0]
    L_eff = max(1, min(args.L, args.hops))

    total = min(args.items, len(L0))
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(total):
            x = L0[i]
            inters, a = compose_chain(x, functions)
            # Build chain facts for f1..f{n-2} from target and context chains
            facts_chain: List[List[str]] = []
            def chain_facts_from_start(start_x: str) -> List[List[str]]:
                ys, _ = compose_chain(start_x, functions)
                prev_local = start_x
                triples: List[List[str]] = []
                for hop_idx in range(max(0, args.hops - 2)):
                    rel = f"f{hop_idx+1}"
                    nxt_local = ys[hop_idx]
                    triples.append([prev_local, rel, nxt_local])
                    prev_local = nxt_local
                return triples

            # Target chain facts
            target_facts = chain_facts_from_start(x)
            facts_chain.extend(target_facts)

            # Multi-chain hygiene: avoid node reuse across chains within the same item (except true path)
            def tokens_in_triples(triples: List[List[str]]) -> Set[str]:
                s: Set[str] = set()
                for h_, r_, t_ in triples:
                    s.add(h_); s.add(t_)
                return s

            used_nodes: Set[str] = tokens_in_triples(target_facts)

            if not args.chain_only_ordered:
                idx_pool = list(range(len(L0)))
                random.shuffle(idx_pool)
                added = 0
                for j in idx_pool:
                    if j == i:
                        continue
                    cand_triples = chain_facts_from_start(L0[j])
                    cand_tokens = tokens_in_triples(cand_triples)
                    if used_nodes.isdisjoint(cand_tokens):
                        facts_chain.extend(cand_triples)
                        used_nodes.update(cand_tokens)
                        added += 1
                        if added >= max(0, args.context_chains):
                            break

            if not args.chain_only_ordered:
                random.shuffle(facts_chain)

            # exposure from facts: token frequency counts across the shown facts
            token_counts: Dict[str, int] = {}
            for h_, r_, t_ in facts_chain:
                token_counts[h_] = token_counts.get(h_, 0) + 1
                token_counts[t_] = token_counts.get(t_, 0) + 1

            def sample_candidates(heads: List[str], func_map: Dict[str, str], rel: str, correct_head: str, k: int) -> List[List[str]]:
                candidates: List[List[str]] = []
                used_heads: Set[str] = set()
                used_tails: Set[str] = set()
                correct_tail = func_map[correct_head]
                candidates.append([correct_head, rel, correct_tail])
                used_heads.add(correct_head)
                used_tails.add(correct_tail)
                need = max(0, k - 1)
                counts = [token_counts.get(h, 0) for h in heads]
                if counts:
                    sorted_counts = sorted(counts)
                    median = sorted_counts[len(sorted_counts)//2]
                else:
                    median = 0
                low_pool = [h for h in heads if h != correct_head and token_counts.get(h, 0) <= median]
                high_pool = [h for h in heads if h != correct_head and token_counts.get(h, 0) > median]
                random.shuffle(low_pool)
                random.shuffle(high_pool)
                target_low = need // 2
                target_high = need - target_low
                toggle = True
                while len(candidates) < k and (low_pool or high_pool):
                    choose_low = (toggle and target_low > 0 and low_pool) or not high_pool
                    choose_high = (not toggle and target_high > 0 and high_pool) or not low_pool
                    if choose_low and low_pool:
                        h = low_pool.pop()
                    elif choose_high and high_pool:
                        h = high_pool.pop()
                    else:
                        pool = low_pool if low_pool else high_pool
                        if not pool:
                            break
                        h = pool.pop()
                    t = func_map[h]
                    if h in used_heads or t in used_tails:
                        continue
                    candidates.append([h, rel, t])
                    used_heads.add(h)
                    used_tails.add(t)
                    if token_counts.get(h, 0) <= median and target_low > 0:
                        target_low -= 1
                    elif token_counts.get(h, 0) > median and target_high > 0:
                        target_high -= 1
                    toggle = not toggle
                return candidates
            
            # Determine k per level
            def k_for(i: int) -> int:
                if args.k is not None:
                    return int(args.k)
                if i == args.hops:
                    return args.k2
                if i == args.hops - 1:
                    return args.k1
                if i == args.hops - 2:
                    return args.k0
                return args.k2

            # Build dynamic candidate blocks for functions i in [n-L+1..n]
            candidates_blocks: Dict[str, List[List[str]]] = {}
            for i_func in range(args.hops - L_eff + 1, args.hops + 1):
                rel = f"f{i_func}"
                heads_layer = layers[i_func - 1]
                func_map = functions[i_func - 1]
                if i_func == 1:
                    correct_head = x
                else:
                    correct_head = inters[i_func - 2]
                block = sample_candidates(heads_layer, func_map, rel, correct_head, k_for(i_func))
                random.shuffle(block)
                candidates_blocks[f"candidates_f{i_func}"] = block

            # Path-collision stress: ensure for each non-gold head at f_{n-1}, there is a corresponding f_n triple
            if args.path_collision_stress:
                fn1_key = f"candidates_f{args.hops - 1}"
                fn_key = f"candidates_f{args.hops}"
                if fn1_key in candidates_blocks and fn_key in candidates_blocks:
                    heads_fn = {h for h, _, _ in candidates_blocks[fn_key]}
                    augmented = False
                    for h, r, t in list(candidates_blocks[fn1_key]):
                        if h not in heads_fn:
                            # add corresponding f_n mapping for this head
                            candidates_blocks[fn_key].append([h, f"f{args.hops}", functions[args.hops - 1][h]])
                            heads_fn.add(h)
                            augmented = True
                    if augmented:
                        random.shuffle(candidates_blocks[fn_key])
            q = _question_laddered(args.hops, x)
            item = {
                "id": f"SYN-{i:06d}",
                "type": "synthetic",
                "n": args.hops,
                "L": L_eff,
                "facts_chain": facts_chain,
                "question": q,
                "answer_id": a,
                "answer_aliases": [a],
            }
            for key, block in candidates_blocks.items():
                item[key] = block
            if args.hops - 2 >= 1:
                key = f"candidates_f{args.hops - 2}"
                if key in candidates_blocks:
                    item["candidates_fn_2"] = candidates_blocks[key]
            if args.hops - 1 >= 1:
                key = f"candidates_f{args.hops - 1}"
                if key in candidates_blocks:
                    item["candidates_fn_1"] = candidates_blocks[key]
            key = f"candidates_f{args.hops}"
            if key in candidates_blocks:
                item["candidates_fn"] = candidates_blocks[key]
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Wrote {total} items to {args.out}")


if __name__ == "__main__":
    main()
