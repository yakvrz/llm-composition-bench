#!/usr/bin/env python3
"""
Open-book evaluator for symbolic n-hop questions (laddered candidates for f_{n-2}, f_{n-1}, and f_n).

Input JSONL items per line:
{
  "id": "SYN-000001",
  "type": "synthetic",
  "n": 4,
  "facts_chain": [[L0,f1,L1], ..., [L_{n-3}, f_{n-2}, L_{n-2}]],
  "candidates_fn_2": [[L_{n-3}, f_{n-2}, L_{n-2}], ...],
  "candidates_fn_1": [[L_{n-2}, f_{n-1}, L_{n-1}], ...],
  "candidates_fn": [[L_{n-1}, f_n, L_n], ...],
  "question": "What is f_n of the element reached from <start>?",
  "answer_id": "...",
  "answer_aliases": ["..."]
}

Usage:
  pip install openai python-dotenv
  # .env must contain: OPENAI_API_KEY=sk-...
  python eval.py --in data.jsonl --out results.jsonl --temp 0.0
"""

import argparse, json, re, sys, time, unicodedata
from collections import defaultdict
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI


# --------------------------- Prompts ---------------------------

OPENBOOK_INSTRUCTIONS = (
    "You are given early chain facts (f1..f{n-2}) and candidate blocks for f{n-2}, f{n-1}, and f{n}.\n"
    "Use ONLY these lines to answer. Do not use outside knowledge.\n"
    "Answer with ONE token only (no extra words)."
)

def build_prompt(item: Dict[str, Any]) -> str:
    n = item.get("n", 0)
    chain = item.get("facts_chain") or []
    cand_fn2 = item.get("candidates_fn_2") or []
    cand_fn1 = item.get("candidates_fn_1") or []
    cand_fn = item.get("candidates_fn") or []
    fn2_block = "\n".join(
        f"{h} ->{r}-> {t}" for h, r, t in cand_fn2
    )
    chain_block = "\n".join(
        f"{h} ->{r}-> {t}" for h, r, t in chain
    )
    fn1_block = "\n".join(
        f"{h} ->{r}-> {t}" for h, r, t in cand_fn1
    )
    fn_block = "\n".join(
        f"{h} ->{r}-> {t}" for h, r, t in cand_fn
    )
    q = item["question"].strip()
    return (
        f"{OPENBOOK_INSTRUCTIONS.replace('{n}', str(n))}\n\n"
        f"Facts:\n{chain_block}\nCandidates for f{n-2}:\n{fn2_block}\nCandidates for f{n-1}:\n{fn1_block}\nCandidates for f{n}:\n{fn_block}\n\n"
        f"Question: {q}\n\n"
        "Final answer (token only):"
    )


# --------------------------- Normalization / Scoring ---------------------------

_PUNCT_RE = re.compile(r"[^\w\s\-\./]")
_WS_RE = re.compile(r"\s+")


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    s = re.sub(r"^\s*(answer|final answer|entity)\s*:\s*", "", s, flags=re.I)
    s = s.strip().strip("\"'`“”‘’")
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s)
    return s.casefold().strip()


def exact_match(pred: str, aliases: List[str]) -> bool:
    p = _normalize(pred)
    for a in aliases:
        if _normalize(a) == p:
            return True
    return False


# --------------------------- Inference ---------------------------


def query_model(client: OpenAI, prompt: str, model: str, temperature: float, max_output_tokens: int) -> tuple[str, str]:
    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    raw_text = (resp.output_text or "").strip()
    for line in raw_text.splitlines():
        line = line.strip()
        if line:
            return line, raw_text
    return raw_text, raw_text


# --------------------------- IO ---------------------------


def iter_jsonl(path: str):
    with (sys.stdin if path == "-" else open(path, "r", encoding="utf-8")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# --------------------------- Main ---------------------------


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", required=True, help="Input JSONL file(s) or '-' for stdin")
    ap.add_argument("--out", type=str, default="", help="Optional path to write per-item results JSONL")
    ap.add_argument("--model", type=str, default="gpt-4.1")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--max_output_tokens", type=int, default=16)
    ap.add_argument("--n", type=int, default=0, help="Optional cap on evaluated items (0 = all)")
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between API calls")
    ap.add_argument("--save_prompt", action="store_true", help="Include the constructed prompt in results")
    ap.add_argument("--save_raw_output", action="store_true", help="Include the raw model output in results")
    ap.add_argument("--log_first", type=int, default=0, help="If >0, only log prompt/raw for first N items")
    args = ap.parse_args()

    client = OpenAI()

    # Load data
    items: List[Dict[str, Any]] = []
    for path in args.inputs:
        items.extend(iter_jsonl(path))
    if args.n and args.n > 0:
        items = items[: args.n]

    perf = {"total": {"n": 0, "correct": 0}, "by_n": defaultdict(lambda: {"n": 0, "correct": 0})}
    outputs = []

    for idx, item in enumerate(items):
        n = int(item.get("n", 0))
        # Optional guards
        assert item.get("candidates_fn_2") is not None, "candidates_fn_2 missing"
        assert item.get("candidates_fn_1"), "candidates_fn_1 missing"
        assert item.get("candidates_fn"), "candidates_fn missing"
        aliases = item.get("answer_aliases") or [item["answer_id"]]
        prompt = build_prompt(item)

        try:
            pred, raw = query_model(
                client=client,
                prompt=prompt,
                model=args.model,
                temperature=args.temp,
                max_output_tokens=args.max_output_tokens,
            )
            err = None
        except Exception as e:
            pred = ""
            raw = ""
            err = str(e)

        is_em = exact_match(pred, aliases)
        perf["total"]["n"] += 1
        perf["total"]["correct"] += int(is_em)
        perf["by_n"][n]["n"] += 1
        perf["by_n"][n]["correct"] += int(is_em)

        rec = {
            "id": item.get("id"),
            "n": n,
            "question": item.get("question"),
            "gold": aliases,
            "pred": pred,
            "em": is_em,
            "error": err,
        }
        rec["facts_chain"] = item.get("facts_chain")
        rec["candidates_fn_2"] = item.get("candidates_fn_2") or []
        rec["candidates_fn_1"] = item.get("candidates_fn_1")
        rec["candidates_fn"] = item.get("candidates_fn")
        should_log = (args.log_first <= 0) or (idx < args.log_first)
        if args.save_prompt and should_log:
            rec["prompt"] = prompt
        if args.save_raw_output and should_log:
            rec["raw_output"] = raw
        outputs.append(rec)

        if args.sleep > 0:
            time.sleep(args.sleep)
        if (idx + 1) % 25 == 0:
            print(f"[{idx+1}/{len(items)}] last_em={is_em}", flush=True)

    # Write results if requested
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in outputs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    def acc(c, n):
        return 0.0 if n == 0 else c / n

    total_n = perf["total"]["n"]
    total_c = perf["total"]["correct"]
    print("\n=== Open-book n-hop Accuracy (Exact Match) ===")
    print(f"TOTAL: {total_c}/{total_n} = {acc(total_c, total_n):.3f}")
    print("By n (hops):")
    for n_key, v in sorted(perf["by_n"].items()):
        print(f"{n_key:>3} hops: {v['correct']}/{v['n']} = {acc(v['correct'], v['n']):.3f}")


if __name__ == "__main__":
    main()


