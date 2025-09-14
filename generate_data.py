#!/usr/bin/env python3
"""
Symbolic n-hop synthetic dataset generator (open-book, laddered candidates with multi-chain context).

Spec:
- Layers L0..Ln (size M); relations f1..fn are bijections (random permutations)
- Emit shuffled chain facts for f1..f{n-2} from the target chain plus K distractor chains
- Provide candidate sets for f{n-2}, f{n-1} and f{n} with exposure balancing
- Answer is a = f{n}(y_{n-1})
- Output per item: id, type, n, facts_chain, candidates_fn_2, candidates_fn_1, candidates_fn,
  question, answer_id, answer_aliases

Usage:
  python gen_synth.py --items 200 --hops 4 --k0 4 --k1 4 --k2 4 --context_chains 6 --M 512 --out data/synth.jsonl
"""

import argparse, json, random, time
from typing import Dict, List, Tuple


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
    # Ask for f_n(f_{n-1}(...f1(x))) using symbolic form
    inner = " of ".join([f"f{i}" for i in range(hops - 1, 0, -1)])
    return f"What is f{hops} of {inner} of {start_token}?"


def question_text(hops: int, start_token: str) -> str:
    return f"What is f{hops} of the element reached from {start_token}?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=200, help="number of instances")
    ap.add_argument("--hops", type=int, default=4, help="number of hops (n)")
    ap.add_argument("--k0", type=int, default=4, help="number of candidates for f_{n-2}")
    ap.add_argument("--k1", type=int, default=4, help="number of candidates for f_{n-1}")
    ap.add_argument("--k2", type=int, default=4, help="number of candidates for f_n")
    ap.add_argument("--context_chains", type=int, default=6, help="number of distractor chains to mix into facts")
    ap.add_argument("--M", type=int, default=512, help="layer size (tokens per layer)")
    ap.add_argument("--out", type=str, default="data/synth.jsonl", help="output JSONL path")
    ap.add_argument("--seed", type=int, default=7, help="random seed")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to sleep between items")
    args = ap.parse_args()

    layers, functions = build_bijections(args.hops, args.M, seed=args.seed)
    L0 = layers[0]
    L_nm3 = layers[-4] if args.hops >= 3 else layers[0]
    L_nm2 = layers[-3] if args.hops >= 2 else layers[0]
    L_nm1 = layers[-2] if args.hops >= 1 else layers[0]
    f_nm2 = functions[-3] if args.hops >= 3 else functions[0]
    f_nm1 = functions[-2] if args.hops >= 2 else functions[0]
    f_n = functions[-1]

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
            facts_chain.extend(chain_facts_from_start(x))
            # Context distractor chains
            idx_pool = list(range(len(L0)))
            random.shuffle(idx_pool)
            added = 0
            for j in idx_pool:
                if j == i:
                    continue
                facts_chain.extend(chain_facts_from_start(L0[j]))
                added += 1
                if added >= max(0, args.context_chains):
                    break
            # Shuffle facts to hide order
            random.shuffle(facts_chain)

            # Laddered candidates: f_{n-2}, f_{n-1}, f_n
            y_nm2 = inters[-2] if args.hops >= 2 else prev
            y_nm1 = inters[-1]
            y_nm3 = inters[-3] if args.hops >= 3 else prev

            # exposure set from facts
            facts_tokens = set()
            for h_, r_, t_ in facts_chain:
                facts_tokens.add(h_); facts_tokens.add(t_)

            # f_{n-2} candidates (heads in L_{n-3})
            cand_fn2: List[List[str]] = []
            if args.hops >= 3:
                cand_fn2.append([y_nm3, f"f{args.hops-2}", y_nm2])
                idx_correct_nm3 = L_nm3.index(y_nm3)
                pool_nm3 = [j for j in range(len(L_nm3)) if j != idx_correct_nm3]
                random.shuffle(pool_nm3)
                # prefer heads that appear in facts
                preferred = [L_nm3[j] for j in pool_nm3 if L_nm3[j] in facts_tokens]
                others = [L_nm3[j] for j in pool_nm3 if L_nm3[j] not in facts_tokens]
                need = max(0, args.k0 - 1)
                for z in preferred[: min(len(preferred), (need + 1) // 2)]:
                    cand_fn2.append([z, f"f{args.hops-2}", f_nm2[z]])
                while len(cand_fn2) < 1 + need and others:
                    z = others.pop()
                    cand_fn2.append([z, f"f{args.hops-2}", f_nm2[z]])

            # f_{n-1} candidates (heads in L_{n-2})
            cand_fn1: List[List[str]] = []
            cand_fn1.append([y_nm2, f"f{args.hops-1}", y_nm1])
            idx_correct_nm2 = L_nm2.index(y_nm2)
            pool_nm2 = [j for j in range(len(L_nm2)) if j != idx_correct_nm2]
            random.shuffle(pool_nm2)
            # use tails from fn-2 candidates where possible
            tails_fn2 = {t for _, _, t in cand_fn2} if args.hops >= 3 else set()
            extra_heads1 = [h for h in tails_fn2 if h != y_nm2]
            random.shuffle(extra_heads1)
            used1 = {y_nm2}
            for h in extra_heads1:
                cand_fn1.append([h, f"f{args.hops-1}", f_nm1[h]])
                used1.add(h)
                if len(cand_fn1) >= args.k1:
                    break
            for j in pool_nm2:
                if len(cand_fn1) >= args.k1:
                    break
                z = L_nm2[j]
                if z in used1:
                    continue
                cand_fn1.append([z, f"f{args.hops-1}", f_nm1[z]])
                used1.add(z)

            # f_n candidates (heads in L_{n-1}); ensure overlap with tails of f_{n-1}
            cand_fn: List[List[str]] = []
            cand_fn.append([y_nm1, f"f{args.hops}", a])
            tails_fn1 = {t for _, _, t in cand_fn1}
            extra_heads = [h for h in tails_fn1 if h != y_nm1]
            random.shuffle(extra_heads)
            used_final = {y_nm1}
            for h in extra_heads:
                cand_fn.append([h, f"f{args.hops}", f_n[h]])
                used_final.add(h)
                if len(cand_fn) >= min(args.k2, max(2, args.k2)):
                    break
            pool_nm1 = list(range(len(L_nm1)))
            random.shuffle(pool_nm1)
            for j in pool_nm1:
                if len(cand_fn) >= args.k2:
                    break
                h = L_nm1[j]
                if h in used_final:
                    continue
                cand_fn.append([h, f"f{args.hops}", f_n[h]])
                used_final.add(h)

            random.shuffle(cand_fn2)
            random.shuffle(cand_fn1)
            random.shuffle(cand_fn)
            q = _question_laddered(args.hops, x)
            item = {
                "id": f"SYN-{i:06d}",
                "type": "synthetic",
                "n": args.hops,
                "facts_chain": facts_chain,
                "candidates_fn_2": cand_fn2 if args.hops >= 3 else [],
                "candidates_fn_1": cand_fn1,
                "candidates_fn": cand_fn,
                "question": q,
                "answer_id": a,
                "answer_aliases": [a],
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Wrote {total} items to {args.out}")


if __name__ == "__main__":
    main()


