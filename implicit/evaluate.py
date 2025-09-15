#!/usr/bin/env python3
"""
Evaluator for the implicit approach (bag-of-facts, no candidate blocks).

Prompt:
- Show shuffled facts (m chains × first n−1 hops). Ask for y_{n-1} given the start.
Output:
- Exact Match on the returned token.
Hygiene:
- Assert there are at least two f_{n-1} lines.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typing import Any, Dict, List
from collections import defaultdict

from dotenv import load_dotenv
from openai import OpenAI
from utils.eval import exact_match
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time as _time


OPENBOOK_INSTRUCTIONS = (
    "You are given a bag of facts consisting of the first n−1 hops (f1..f{n-1}) from several disjoint chains.\n"
    "Use ONLY these lines to answer. Compose the functions from the given start token.\n"
    "Answer with ONE token only (no extra words)."
)


def build_prompt(item: Dict[str, Any]) -> str:
    n = int(item.get("n", 0))
    facts = item.get("facts_bag") or []
    facts_block = "\n".join(f"{h} ->{r}-> {t}" for h, r, t in facts)
    q = item["question"].strip()
    return (
        f"{OPENBOOK_INSTRUCTIONS.replace('{n}', str(n))}\n\n"
        f"Facts (shuffled):\n{facts_block}\n\n"
        f"Question: {q}\n\n"
        "Final answer (token only):"
    )


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
    ap.add_argument("--order_trials", type=int, default=1, help="number of shuffle trials per item with re-ordered facts")
    ap.add_argument("--baseline", type=str, default="", choices=["", "pointer_f_n1"], help="run a baseline instead of calling the model")
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
    print(f"[implicit/eval] start: model={args.model} items={len(items)} order_trials={args.order_trials} baseline={args.baseline or 'none'}", flush=True)
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
        aliases_local = item.get("answer_aliases") or [item["answer_id"]]
        # Hygiene: require >= 2 f_{n-1} lines
        assert sum(1 for _, r, _ in (item.get("facts_bag") or []) if r == f"f{n_local-1}") >= 2
        # Order trials
        trial_ems_local = []
        last_prompt_local = ""; last_raw_local = ""; pred_local = ""; err_local = None
        import random as _rnd
        for t in range(max(1, int(args.order_trials))):
            facts = list(item.get("facts_bag") or [])
            _rnd.seed(1337 + t)
            _rnd.shuffle(facts)
            trial_item = dict(item)
            trial_item["facts_bag"] = facts
            if args.baseline == "pointer_f_n1":
                # choose tail of first f_{n-1} line
                pred_local = next((tr[2] for tr in facts if tr[1] == f"f{n_local-1}"), "")
                err_local = None
                last_prompt_local = ""; last_raw_local = ""
            else:
                prompt = build_prompt(trial_item)
                resp, err = call_model_with_retries(prompt)
                if resp is not None:
                    raw_text = (resp.output_text or "").strip()
                    for line in raw_text.splitlines():
                        line = line.strip()
                        if line:
                            pred_local = line
                            break
                    err_local = None
                    last_prompt_local = prompt
                    last_raw_local = raw_text
                    if not printed_api_ok:
                        with lock:
                            if not printed_api_ok:
                                print(f"[implicit/eval] API OK model={args.model}", flush=True)
                                printed_api_ok = True
                else:
                    pred_local = ""; last_raw_local = ""; err_local = err or ""
                    with lock:
                        print(f"[implicit/eval] API ERROR item={item.get('id')} err={err_local}", flush=True)
            is_em = exact_match(pred_local, aliases_local)
            trial_ems_local.append(int(is_em))
        # aggregate over order trials (mean EM per item)
        is_em_local = sum(trial_ems_local) / len(trial_ems_local) >= 0.5  # majority as boolean em
        rec_local = {
            "id": item.get("id"),
            "n": n_local,
            "question": item.get("question"),
            "gold": aliases_local,
            "pred": pred_local,
            "em": is_em_local,
            "error": err_local,
            "facts_bag": item.get("facts_bag"),
            "order_trials": int(args.order_trials),
            "baseline": args.baseline or None,
        }
        with lock:
            perf["total"]["n"] += 1
            perf["total"]["correct"] += int(is_em_local)
            perf["by_n"][n_local]["n"] += 1
            perf["by_n"][n_local]["correct"] += int(is_em_local)
            if fout:
                fout.write(json.dumps(rec_local, ensure_ascii=False) + "\n")
                fout.flush()
            print(f"[implicit/eval] {idx+1}/{len(items)} em={is_em_local}", flush=True)
        return rec_local

    if int(args.concurrency) > 1 and not args.baseline:
        with ThreadPoolExecutor(max_workers=int(args.concurrency)) as ex:
            futures = [ex.submit(eval_one, idx, item) for idx, item in enumerate(items)]
            for fut in as_completed(futures):
                try:
                    outputs.append(fut.result())
                except Exception as e:
                    print(f"[implicit/eval] worker error: {e}", flush=True)
    else:
        for idx, item in enumerate(items):
            outputs.append(eval_one(idx, item))
            if args.sleep > 0:
                time.sleep(args.sleep)
        n = int(item.get("n", 0))
        aliases = item.get("answer_aliases") or [item["answer_id"]]
        # Hygiene: require >= 2 f_{n-1} lines
        assert sum(1 for _, r, _ in (item.get("facts_bag") or []) if r == f"f{n-1}") >= 2
        # Order trials
        trial_ems = []
        last_prompt = ""
        last_raw = ""
        for t in range(max(1, int(args.order_trials))):
            facts = list(item.get("facts_bag") or [])
            import random as _rnd
            _rnd.seed(1337 + t)
            _rnd.shuffle(facts)
            trial_item = dict(item)
            trial_item["facts_bag"] = facts
            if args.baseline == "pointer_f_n1":
                # choose tail of first f_{n-1} line
                pred = next((tr[2] for tr in facts if tr[1] == f"f{n-1}"), "")
                raw = ""
                err = None
                # keep last_prompt/last_raw as empty for baseline
            else:
                prompt = build_prompt(trial_item)
                try:
                    pred, raw = ("", "")
                    resp = client.responses.create(
                        model=args.model, input=prompt, temperature=args.temp, max_output_tokens=args.max_output_tokens
                    )
                    raw_text = (resp.output_text or "").strip()
                    for line in raw_text.splitlines():
                        line = line.strip()
                        if line:
                            pred = line
                            break
                    raw = raw_text
                    err = None
                    last_prompt = prompt
                    last_raw = raw_text
                    if not printed_api_ok:
                        print(f"[implicit/eval] API OK model={args.model}", flush=True)
                        printed_api_ok = True
                except Exception as e:
                    pred = ""; raw = ""; err = str(e)
                    print(f"[implicit/eval] API ERROR item={item.get('id')} err={err}", flush=True)
            is_em = exact_match(pred, aliases)
            trial_ems.append(int(is_em))
        # aggregate over order trials (mean EM per item)
        is_em = sum(trial_ems) / len(trial_ems) >= 0.5  # majority as boolean em
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
            "facts_bag": item.get("facts_bag"),
            "order_trials": int(args.order_trials),
            "baseline": args.baseline or None,
        }
        if args.save_prompt and ((args.log_first <= 0) or (idx < args.log_first)):
            rec["prompt"] = last_prompt
        if args.save_raw_output and ((args.log_first <= 0) or (idx < args.log_first)):
            rec["raw_output"] = last_raw
        outputs.append(rec)
        if args.sleep > 0:
            time.sleep(args.sleep)
        if (idx + 1) % 1 == 0 or (idx + 1) == len(items):
            print(f"[implicit/eval] {idx+1}/{len(items)} em={is_em}", flush=True)

    if fout:
        fout.close()

    def acc(c, n):
        return 0.0 if n == 0 else c / n
    total_n = perf["total"]["n"]
    total_c = perf["total"]["correct"]
    print("\n=== Implicit n-hop Accuracy (Exact Match) ===")
    print(f"TOTAL: {total_c}/{total_n} = {acc(total_c, total_n):.3f}")
    print("By n (hops):")
    for n_key, v in sorted(perf["by_n"].items()):
        print(f"{n_key:>3} hops: {v['correct']}/{v['n']} = {acc(v['correct'], v['n']):.3f}")


if __name__ == "__main__":
    main()


