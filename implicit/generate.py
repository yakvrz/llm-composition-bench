#!/usr/bin/env python3
"""
Implicit generator: bag-of-facts without explicit candidate blocks.

Spec per item:
- Build M-sized layers L0..Ln with bijective f1..fn.
- Select m >= 4 distinct chains (no node overlap between chains).
- Emit the first n-1 hops for each selected chain (balanced: exactly m lines per relation f1..f{n-1}).
- Choose one chain's start A_q as the question target.
- Shuffle all facts; output JSONL with fields: id, type, n, m, facts_bag, question, answer_id, answer_aliases.

Hygiene:
- Disjoint nodes across chains within an item (except the true target chain isn't special; all m are disjoint).
- Independent shuffles.
- Assert count of f_{n-1} lines >= 2 (i.e., m >= 2, but we require m >= 4 by default).
"""
import argparse, json, random, time
from typing import Dict, List, Tuple
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.synth import build_bijections, compose_chain


def layer_prefix(i: int) -> str:
    return chr(ord('A') + i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=200)
    ap.add_argument("--hops", type=int, default=4)
    ap.add_argument("--m", type=int, default=6, help="number of chains to include in the bag (>=4)")
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--out", type=str, default="data/implicit.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--ablate_inner", action="store_true", help="remove all edges for one inner hop f_j to create ambiguity")
    ap.add_argument("--ablate_hop", type=int, default=2, help="which inner hop j to ablate (1<j<n-1)")
    args = ap.parse_args()

    assert args.m >= 4, "m must be >= 4"

    layers, functions = build_bijections(args.hops, args.M, seed=args.seed)
    L0 = layers[0]

    total = min(args.items, len(L0))
    print(
        f"[implicit/generate] Start: items={total}, n={args.hops}, m={args.m}, M={args.M}, seed={args.seed}, "
        f"ablate_inner={args.ablate_inner}, ablate_hop={args.ablate_hop}\n→ {args.out}",
        flush=True,
    )
    progress_interval = 1 if total <= 50 else 25
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(total):
            random.seed(args.seed + i)
            # pick m distinct start indices with disjoint chains (no node reuse across chains)
            # Construct chains sequentially and ensure disjoint nodes
            selected: List[int] = []
            used_nodes = set()
            idx_pool = list(range(len(L0)))
            random.shuffle(idx_pool)

            def chain_nodes(start_idx: int) -> List[str]:
                x = L0[start_idx]
                ys, _a = compose_chain(x, functions)
                nodes = [x] + ys  # L0..Ln
                return nodes

            for j in idx_pool:
                nodes = chain_nodes(j)
                if used_nodes.isdisjoint(nodes):
                    selected.append(j)
                    used_nodes.update(nodes)
                    if len(selected) >= args.m:
                        break
            assert len(selected) >= args.m, "Could not find m disjoint chains; increase M or reduce m/items"

            # Emit first n-1 hops for each chain
            facts: List[List[str]] = []
            for j in selected:
                x = L0[j]
                prev_local = x
                for hop_idx in range(args.hops - 1):
                    rel = f"f{hop_idx+1}"
                    nxt_local = functions[hop_idx][prev_local]
                    facts.append([prev_local, rel, nxt_local])
                    prev_local = nxt_local

            # Optional ablation of one inner hop f_j
            if args.ablate_inner:
                j = int(args.ablate_hop)
                assert 1 < j < args.hops - 1, "ablate_hop must be an inner hop (2..n-2)"
                facts = [tr for tr in facts if tr[1] != f"f{j}"]

            # Balanced counts: exactly m lines per relation except ablated hop
            counts = {f"f{h+1}": 0 for h in range(args.hops - 1)}
            for _, r, _ in facts:
                counts[r] += 1
            for r, c in counts.items():
                if args.ablate_inner and r == f"f{args.ablate_hop}":
                    continue
                assert c == args.m, f"Unbalanced relation count for {r}: {c} != {args.m}"

            # Choose target chain
            q_idx = random.choice(selected)
            q_start = L0[q_idx]
            ys, a = compose_chain(q_start, functions)
            y_nm1 = ys[-1]

            # Shuffle facts bag independently
            random.shuffle(facts)

            # Build item
            q = " of ".join([f"f{i}" for i in range(args.hops, 0, -1)])
            q_text = f"What is {q} of {q_start}?".replace(f"f{args.hops} of ", "")  # ask up to f_{n-1}
            item = {
                "id": f"IMP-{i:06d}",
                "type": "implicit",
                "n": args.hops,
                "m": args.m,
                "facts_bag": facts,
                "question": q_text,
                "answer_id": y_nm1,
                "answer_aliases": [y_nm1],
            }
            # Hygiene checks
            assert sum(1 for _, r, _ in facts if r == f"f{args.hops-1}") >= 2

            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            if args.sleep > 0:
                time.sleep(args.sleep)
            if ((i + 1) % progress_interval == 0) or ((i + 1) == total):
                print(f"[implicit/generate] progress: {i+1}/{total}", flush=True)

    print(f"[implicit/generate] Done. Wrote {total} items to {args.out}", flush=True)


if __name__ == "__main__":
    main()


