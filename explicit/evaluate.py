#!/usr/bin/env python3
"""
Explicit evaluator (candidate-based) – relocated from project root.
"""
import argparse, json, re, sys, time, unicodedata
from collections import defaultdict
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI


OPENBOOK_INSTRUCTIONS = (
    "You are given early chain facts (f1..f{n-2}) and candidate blocks for f{n-2}, f{n-1}, and f{n}.\n"
    "Use ONLY these lines to answer. Do not use outside knowledge.\n"
    "Answer with ONE token only (no extra words)."
)


def build_prompt(item: Dict[str, Any]) -> str:
    n = int(item.get("n", 0))
    chain = item.get("facts_chain") or []
    dynamic_blocks: list[tuple[int, list[list[str]]]] = []
    for k, v in item.items():
        if isinstance(k, str) and k.startswith("candidates_f"):
            try:
                i = int(k[len("candidates_f"):])
            except ValueError:
                continue
            if isinstance(v, list):
                dynamic_blocks.append((i, v))
    dynamic_blocks.sort(key=lambda x: x[0])

    chain_block = "\n".join(
        f"{h} ->{r}-> {t}" for h, r, t in chain
    )
    q = item["question"].strip()

    cand_sections: list[str] = []
    for i, triples in dynamic_blocks:
        triples_block = "\n".join(f"{h} ->{r}-> {t}" for h, r, t in triples)
        cand_sections.append(f"Candidates for f{i}:\n{triples_block}")

    cand_text = "\n".join(cand_sections)
    return (
        f"{OPENBOOK_INSTRUCTIONS.replace('{n}', str(n))}\n\n"
        f"Facts:\n{chain_block}\n{cand_text}\n\n"
        f"Question: {q}\n\n"
        "Final answer (token only):"
    )


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


def iter_jsonl(path: str):
    with (sys.stdin if path == "-" else open(path, "r", encoding="utf-8")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", required=True)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--model", type=str, default="gpt-4.1")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--max_output_tokens", type=int, default=16)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--save_prompt", action="store_true")
    ap.add_argument("--save_raw_output", action="store_true")
    ap.add_argument("--log_first", type=int, default=0)
    ap.add_argument("--order_trials", type=int, default=1, help="number of shuffle trials per item (facts and candidate blocks)")
    args = ap.parse_args()

    client = OpenAI()

    items: List[Dict[str, Any]] = []
    for path in args.inputs:
        items.extend(iter_jsonl(path))
    if args.n and args.n > 0:
        items = items[: args.n]

    perf = {"total": {"n": 0, "correct": 0}, "by_n": defaultdict(lambda: {"n": 0, "correct": 0})}
    outputs = []

    for idx, item in enumerate(items):
        n = int(item.get("n", 0))
        has_any_block = any(isinstance(v, list) and k.startswith("candidates_f") for k, v in item.items() if isinstance(k, str))
        assert has_any_block, "No candidate blocks present (expected keys like candidates_f{i})"
        aliases = item.get("answer_aliases") or [item["answer_id"]]
        # perform order trials by reshuffling blocks client-side
        import random as _rnd
        trial_ems = []
        for t in range(max(1, int(args.order_trials))):
            _rnd.seed(2024 + t)
            # copy item and reshuffle facts and each candidates_f* block independently
            trial_item = dict(item)
            fc = list(item.get("facts_chain") or [])
            _rnd.shuffle(fc)
            trial_item["facts_chain"] = fc
            for k, v in list(item.items()):
                if isinstance(k, str) and k.startswith("candidates_f") and isinstance(v, list):
                    vv = list(v)
                    _rnd.shuffle(vv)
                    trial_item[k] = vv
            prompt = build_prompt(trial_item)
            try:
                resp = client.responses.create(model=args.model, input=prompt, temperature=args.temp, max_output_tokens=args.max_output_tokens)
                raw_text = (resp.output_text or "").strip()
                pred = next((ln.strip() for ln in raw_text.splitlines() if ln.strip()), raw_text)
                raw = raw_text
                err = None
            except Exception as e:
                pred = ""; raw = ""; err = str(e)
            trial_ems.append(int(exact_match(pred, aliases)))
        is_em = sum(trial_ems) / len(trial_ems) >= 0.5
        perf["total"]["n"] += 1
        perf["total"]["correct"] += int(is_em)
        perf["by_n"][n]["n"] += 1
        perf["by_n"][n]["correct"] += int(is_em)
        rec = {"id": item.get("id"), "n": n, "question": item.get("question"), "gold": aliases, "pred": pred, "em": is_em, "error": err, "facts_chain": item.get("facts_chain")}
        for k, v in item.items():
            if isinstance(k, str) and k.startswith("candidates_f"):
                rec[k] = v
        rec["order_trials"] = int(args.order_trials)
        outputs.append(rec)
        if args.sleep > 0:
            time.sleep(args.sleep)
        if (idx + 1) % 25 == 0:
            print(f"[{idx+1}/{len(items)}] last_em={is_em}", flush=True)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in outputs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def acc(c, n): return 0.0 if n == 0 else c / n
    total_n = perf["total"]["n"]; total_c = perf["total"]["correct"]
    print("\n=== Open-book n-hop Accuracy (Exact Match) ===")
    print(f"TOTAL: {total_c}/{total_n} = {acc(total_c, total_n):.3f}")
    print("By n (hops):")
    for n_key, v in sorted(perf["by_n"].items()):
        print(f"{n_key:>3} hops: {v['correct']}/{v['n']} = {acc(v['correct'], v['n']):.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Wrapper for the explicit (candidate-based) evaluator.
Delegates to the repository root `evaluate.py`.
"""
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import evaluate as _eval
    _eval.main()


if __name__ == "__main__":
    main()


