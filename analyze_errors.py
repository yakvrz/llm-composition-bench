#!/usr/bin/env python3
"""
Error analysis for laddered n-hop evaluation results.

Reads a results JSONL (from eval.py) and reports:
- Overall and per-n accuracy
- Compliance: pred ∈ final candidate tails
- Structural coherence of gold/pred with the provided candidates
- Error taxonomy for non-EM cases
- Optional example dumps per error category

Usage:
  python analyze_errors.py --in runs/<ts>/results.jsonl --examples_per_type 3
"""

import argparse, json, sys, collections
from typing import Any, Dict, List, Tuple


def iter_jsonl(path: str):
    with (sys.stdin if path == "-" else open(path, "r", encoding="utf-8")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def extract_y_nm2(facts_chain: List[List[str]]) -> str:
    if not facts_chain:
        return ""
    # Last tail from the chain facts (up to f_{n-2}) is y_{n-2}
    return facts_chain[-1][2]


def tails_of(cands: List[List[str]]) -> List[str]:
    return [t for _, _, t in cands]


def heads_of(cands: List[List[str]]) -> List[str]:
    return [h for h, _, _ in cands]


def find_final_candidate_by_tail(cands: List[List[str]], tail: str) -> Tuple[str, str, str] | None:
    for h, r, t in cands:
        if t == tail:
            return h, r, t
    return None


def has_fn1_edge_from(chain_last: str, cands_fn1: List[List[str]], tail_needed: str) -> bool:
    for h, r, t in cands_fn1:
        if h == chain_last and t == tail_needed:
            return True
    return False


def analyze_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    n = int(rec.get("n", 0))
    em = bool(rec.get("em"))
    gold_list = rec.get("gold") or []
    gold = gold_list[0] if gold_list else rec.get("answer_id")
    pred = rec.get("pred", "")
    fc = rec.get("facts_chain") or []
    c1 = rec.get("candidates_fn_1") or []
    c2 = rec.get("candidates_fn") or []
    y_nm2 = extract_y_nm2(fc)

    final_tails = set(tails_of(c2))
    pred_in_final = pred in final_tails
    gold_in_final = (gold in final_tails)

    # Structural coherence for gold
    gold_coherent = False
    cand_gold = find_final_candidate_by_tail(c2, gold)
    if cand_gold is not None:
        d_gold, _, _ = cand_gold
        gold_coherent = has_fn1_edge_from(y_nm2, c1, d_gold)

    # Structural coherence for pred
    pred_coherent = False
    cand_pred = find_final_candidate_by_tail(c2, pred)
    if cand_pred is not None:
        d_pred, _, _ = cand_pred
        pred_coherent = has_fn1_edge_from(y_nm2, c1, d_pred)

    # Exposure: overlaps between chain tokens and candidate heads
    chain_tokens = set([h for h, _, _ in fc] + [t for _, _, t in fc])
    head_overlap_fn1 = len(chain_tokens.intersection(set(heads_of(c1))))
    head_overlap_fn = len(chain_tokens.intersection(set(heads_of(c2))))

    # Error categorization
    error_type = ""
    if not em:
        if not pred_in_final:
            error_type = "non_compliant_final"
        elif not pred_coherent:
            error_type = "incoherent_path"
        else:
            error_type = "coherent_but_wrong"

    return {
        "n": n,
        "em": em,
        "pred_in_final": pred_in_final,
        "gold_in_final": gold_in_final,
        "gold_coherent": gold_coherent,
        "pred_coherent": pred_coherent,
        "head_overlap_fn1": head_overlap_fn1,
        "head_overlap_fn": head_overlap_fn,
        "error_type": error_type,
    }


def summarize(analyses: List[Dict[str, Any]]):
    total = len(analyses)
    def rate(cnt):
        return 0.0 if total == 0 else cnt / total

    acc = sum(1 for a in analyses if a["em"])
    by_n = collections.defaultdict(lambda: {"n": 0, "em": 0})
    for a in analyses:
        k = a["n"]
        by_n[k]["n"] += 1
        by_n[k]["em"] += int(a["em"])                        

    err_types = collections.Counter(a["error_type"] for a in analyses if not a["em"])        
    compliant = sum(1 for a in analyses if (a["em"] or a["pred_in_final"]))
    coherent = sum(1 for a in analyses if (a["em"] or a["pred_coherent"]))

    return {
        "total": total,
        "acc": rate(acc),
        "by_n": {k: {"n": v["n"], "acc": (0.0 if v["n"] == 0 else v["em"] / v["n"]) } for k, v in sorted(by_n.items())},
        "non_em_by_type": err_types,
        "pred_in_final_rate": rate(compliant),
        "pred_coherent_rate": rate(coherent),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_path", required=True, help="results JSONL from eval.py")
    ap.add_argument("--examples_per_type", type=int, default=0, help="Show up to N examples per error type")
    args = ap.parse_args()

    recs = list(iter_jsonl(args.input_path))
    analyses = [analyze_record(r) for r in recs]
    summary = summarize(analyses)

    print("=== Error Analysis Summary ===")
    print(f"TOTAL: {summary['total']}  ACC: {summary['acc']:.3f}")
    print("By n (hops):")
    for k, v in summary["by_n"].items():
        print(f"  {k} hops: {v['n']}  ACC={v['acc']:.3f}")
    print("\nNon-EM error types:")
    for t, c in summary["non_em_by_type"].most_common():
        print(f"  {t or '—'}: {c}")
    print(f"\nPred in final candidates: {summary['pred_in_final_rate']:.3f}")
    print(f"Pred structurally coherent: {summary['pred_coherent_rate']:.3f}")

    if args.examples_per_type > 0:
        print("\n=== Examples (per error type) ===")
        # Build index from error type to indices
        buckets = collections.defaultdict(list)
        for idx, a in enumerate(analyses):
            if not a["em"]:
                buckets[a["error_type"]].append(idx)
        for t, idxs in buckets.items():
            print(f"\n-- {t or '—'} --")
            for j in idxs[: args.examples_per_type]:
                print(json.dumps(recs[j], ensure_ascii=False))


if __name__ == "__main__":
    main()


