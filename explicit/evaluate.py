#!/usr/bin/env python3
"""
Explicit evaluator (candidate-based) – relocated from project root.
"""
import argparse, json, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from utils.eval import exact_match
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time as _time


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
        # If alias map for this block exists, render it just above the candidates
        alias_key = f"alias_map_f{i}"
        alias_map = item.get(alias_key)
        alias_block = ""
        if isinstance(alias_map, list) and alias_map:
            alias_block = "Alias map (balanced):\n" + "\n".join(f"{a} {eq} {h}" for a, eq, h in alias_map) + "\n"
        cand_sections.append(f"{alias_block}Candidates for f{i}:\n{triples_block}")

    cand_text = "\n".join(cand_sections)
    return (
        f"{OPENBOOK_INSTRUCTIONS.replace('{n}', str(n))}\n\n"
        f"Facts:\n{chain_block}\n{cand_text}\n\n"
        f"Question: {q}\n\n"
        "Final answer (token only):"
    )


# Note: normalization moved to utils.eval


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
    ap.add_argument("--model", type=str, default="gpt-4.1-mini")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--max_output_tokens", type=int, default=16)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--save_prompt", action="store_true")
    ap.add_argument("--save_raw_output", action="store_true")
    ap.add_argument("--log_first", type=int, default=0)
    ap.add_argument("--order_trials", type=int, default=1, help="number of shuffle trials per item (facts and candidate blocks)")
    ap.add_argument("--concurrency", type=int, default=4, help="number of parallel API calls")
    ap.add_argument("--max_retries", type=int, default=3, help="max retries on API errors")
    ap.add_argument("--retry_backoff", type=float, default=1.0, help="initial backoff seconds for retries")
    args = ap.parse_args()

    client = OpenAI()

    items: List[Dict[str, Any]] = []
    for path in args.inputs:
        items.extend(iter_jsonl(path))
    if args.n and args.n > 0:
        items = items[: args.n]

    perf = {"total": {"n": 0, "correct": 0}, "by_n": defaultdict(lambda: {"n": 0, "correct": 0})}
    outputs = []
    print(f"[explicit/eval] start: model={args.model} items={len(items)} order_trials={args.order_trials}", flush=True)
    printed_api_ok = False

    lock = threading.Lock()
    fout = open(args.out, "w", encoding="utf-8") if args.out else None

    def call_model_with_retries(prompt: str):
        delay = float(args.retry_backoff)
        for attempt in range(1, int(args.max_retries) + 1):
            try:
                resp = client.responses.create(model=args.model, input=prompt, temperature=args.temp, max_output_tokens=args.max_output_tokens)
                return (resp, None)
            except Exception as e:
                if attempt >= int(args.max_retries):
                    return (None, str(e))
                _time.sleep(delay)
                delay *= 2

    def eval_one(idx: int, item: Dict[str, Any]):
        nonlocal printed_api_ok
        n_local = int(item.get("n", 0))
        has_any_block = any(isinstance(v, list) and k.startswith("candidates_f") for k, v in item.items() if isinstance(k, str))
        assert has_any_block, "No candidate blocks present (expected keys like candidates_f{i})"
        aliases_local = item.get("answer_aliases") or [item["answer_id"]]
        import random as _rnd
        trial_ems_local = []
        last_prompt_local = ""; last_raw_local = ""; pred_local = ""; err_local = None
        for t in range(max(1, int(args.order_trials))):
            _rnd.seed(2024 + t)
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
            resp, err = call_model_with_retries(prompt)
            if resp is not None:
                raw_text = (resp.output_text or "").strip()
                pred_local = next((ln.strip() for ln in raw_text.splitlines() if ln.strip()), raw_text)
                err_local = None
                last_prompt_local = prompt
                last_raw_local = raw_text
                if not printed_api_ok:
                    with lock:
                        if not printed_api_ok:
                            print(f"[explicit/eval] API OK model={args.model}", flush=True)
                            printed_api_ok = True
            else:
                pred_local = ""; last_raw_local = ""; err_local = err or ""
                with lock:
                    print(f"[explicit/eval] API ERROR item={item.get('id')} err={err_local}", flush=True)
            trial_ems_local.append(int(exact_match(pred_local, aliases_local)))
        is_em_local = sum(trial_ems_local) / len(trial_ems_local) >= 0.5
        rec_local = {"id": item.get("id"), "n": n_local, "question": item.get("question"), "gold": aliases_local, "pred": pred_local, "em": is_em_local, "error": err_local, "facts_chain": item.get("facts_chain")}
        for k, v in item.items():
            if isinstance(k, str) and k.startswith("candidates_f"):
                rec_local[k] = v
        rec_local["order_trials"] = int(args.order_trials)
        if args.save_prompt and ((args.log_first <= 0) or (idx < args.log_first)):
            rec_local["prompt"] = last_prompt_local
        if args.save_raw_output and ((args.log_first <= 0) or (idx < args.log_first)):
            rec_local["raw_output"] = last_raw_local
        with lock:
            perf["total"]["n"] += 1
            perf["total"]["correct"] += int(is_em_local)
            perf["by_n"][n_local]["n"] += 1
            perf["by_n"][n_local]["correct"] += int(is_em_local)
            if fout:
                fout.write(json.dumps(rec_local, ensure_ascii=False) + "\n")
                fout.flush()
            print(f"[explicit/eval] {idx+1}/{len(items)} em={is_em_local}", flush=True)
        return rec_local

    if int(args.concurrency) > 1:
        with ThreadPoolExecutor(max_workers=int(args.concurrency)) as ex:
            futures = [ex.submit(eval_one, idx, item) for idx, item in enumerate(items)]
            for fut in as_completed(futures):
                try:
                    outputs.append(fut.result())
                except Exception as e:
                    print(f"[explicit/eval] worker error: {e}", flush=True)
    else:
        for idx, item in enumerate(items):
            outputs.append(eval_one(idx, item))
            if args.sleep > 0:
                time.sleep(args.sleep)

    if fout:
        fout.close()

    def acc(c, n): return 0.0 if n == 0 else c / n
    total_n = perf["total"]["n"]; total_c = perf["total"]["correct"]
    print("\n=== Open-book n-hop Accuracy (Exact Match) ===")
    print(f"TOTAL: {total_c}/{total_n} = {acc(total_c, total_n):.3f}")
    print("By n (hops):")
    for n_key, v in sorted(perf["by_n"].items()):
        print(f"{n_key:>3} hops: {v['correct']}/{v['n']} = {acc(v['correct'], v['n']):.3f}")


if __name__ == "__main__":
    main()
