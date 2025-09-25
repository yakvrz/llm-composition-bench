#!/usr/bin/env python3
"""
Evaluator for the n-hop composition benchmark (bag-of-facts, no candidate blocks).

Prompt:
- Show shuffled facts (m chains × first n−1 hops). Ask for y_{n-1} given the start.
Output:
- Exact Match on the returned token.
Hygiene:
- Assert there are at least two f_{n-1} lines.
"""
import argparse, json, sys, time, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typing import Any, Dict, List
from collections import defaultdict

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*_args, **_kwargs):
        return False
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


def build_prompt(item: Dict[str, Any], reasoning_steps: bool = False) -> str:
    n = int(item.get("n", 0))
    facts = item.get("facts_bag") or []
    facts_block = "\n".join(f"{h} ->{r}-> {t}" for h, r, t in facts)
    q = item["question"].strip()
    reasoning_block = ""
    if reasoning_steps and n > 1:
        step_lines = []
        for idx in range(1, n):
            step_lines.append(f"Step {idx} (apply f{idx}):")
        reasoning_block = (
            "Reasoning (fill every step before answering):\n"
            + "\n".join(step_lines)
            + "\n\n"
        )
    return (
        f"{OPENBOOK_INSTRUCTIONS.replace('{n}', str(n))}\n\n"
        f"Facts (shuffled):\n{facts_block}\n\n"
        f"Question: {q}\n\n"
        f"{reasoning_block}"
        "Final answer (token only):"
    )


TOKEN_REGEX = re.compile(r"\b[A-Z]_[0-9]{1,8}\b")


def extract_token_from_response(raw_text: str) -> str:
    lines = [ln.strip() for ln in (raw_text or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    # Prefer the Final answer line (take last occurrence)
    final_idx = -1
    for i, ln in enumerate(lines):
        if ln.lower().startswith("final answer"):
            final_idx = i
    if final_idx >= 0:
        # Try token on same line
        m = TOKEN_REGEX.search(lines[final_idx])
        if m:
            return m.group(0)
        # Else try next non-empty line
        if final_idx + 1 < len(lines):
            m2 = TOKEN_REGEX.search(lines[final_idx + 1])
            if m2:
                return m2.group(0)
        # Else try after colon as exact token
        tail = lines[final_idx].split(":")[-1].strip()
        if TOKEN_REGEX.fullmatch(tail):
            return tail
    # Fallback: last token-like anywhere in the text
    matches = list(TOKEN_REGEX.finditer(raw_text or ""))
    if matches:
        return matches[-1].group(0)
    # Last resort: first non-empty line
    return lines[0]


def analyze_reasoning(raw_text: str, n: int) -> tuple[list[str] | None, bool | None]:
    """Extract the final token from each reasoning step when structured slots are present."""
    if not raw_text or n <= 1:
        return None, None
    tokens: list[str] = []
    complete = True
    lines = [ln.strip() for ln in raw_text.splitlines()]
    for idx in range(1, n):
        step_line = next((ln for ln in lines if ln.lower().startswith(f"step {idx}")), None)
        if not step_line:
            tokens.append("")
            complete = False
            continue
        step_tokens = TOKEN_REGEX.findall(step_line)
        if step_tokens:
            tokens.append(step_tokens[-1])
        else:
            tokens.append("")
            complete = False
    return tokens, complete

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
    ap.add_argument("--log_every", type=int, default=1, help="print progress every N items (1=every item)")
    ap.add_argument("--reasoning_steps", action="store_true", help="insert structured step-by-step slots before the final answer prompt")
    args = ap.parse_args()

    import os
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    if base_url is None and os.getenv("OPENROUTER_API_KEY"):
        base_url = "https://openrouter.ai/api/v1"
    default_headers = {}
    referer = os.getenv("OPENAI_HTTP_REFERER") or os.getenv("HTTP_REFERER")
    if referer:
        default_headers["HTTP-Referer"] = referer
    title = os.getenv("OPENAI_X_TITLE") or os.getenv("X_TITLE") or os.getenv("OPENROUTER_APP_NAME")
    if title:
        default_headers["X-Title"] = title
    client = None
    if not args.baseline:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers or None,
        )

    items: List[Dict[str, Any]] = []
    for path in args.inputs:
        items.extend(iter_jsonl(path))
    if args.n and args.n > 0:
        items = items[: args.n]

    perf = {"total": {"n": 0, "correct": 0}, "by_n": defaultdict(lambda: {"n": 0, "correct": 0})}
    outputs = []
    print(f"[benchmark/eval] start: model={args.model} items={len(items)} order_trials={args.order_trials} baseline={args.baseline or 'none'}", flush=True)
    printed_api_ok = False

    lock = threading.Lock()
    fout = open(args.out, "w", encoding="utf-8") if args.out else None

    def extract_from_response(resp):
        if resp is None:
            return ""
        if isinstance(resp, str):
            lower = resp.lower()
            if "<html" in lower or "<!doctype" in lower:
                return ""
            return resp
        text = getattr(resp, "output_text", None)
        if text:
            return text
        output = getattr(resp, "output", None)
        if output:
            pieces: list[str] = []
            for block in output:
                content = getattr(block, "content", None) or block.get("content") if isinstance(block, dict) else None
                if not content:
                    continue
                for part in content:
                    if hasattr(part, "text"):
                        pieces.append(part.text)
                    elif isinstance(part, dict):
                        if part.get("type") in ("output_text", "text") and part.get("text"):
                            pieces.append(part["text"])
            if pieces:
                return "".join(pieces)
        choices = getattr(resp, "choices", None)
        if choices:
            piece = choices[0]
            message = getattr(piece, "message", None) or piece.get("message") if isinstance(piece, dict) else None
            if message:
                content = getattr(message, "content", None)
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return "".join(seg.get("text", "") if isinstance(seg, dict) else str(seg) for seg in content)
        return ""

    def extract_from_chat(chat_resp):
        if chat_resp is None:
            return ""
        choices = getattr(chat_resp, "choices", None)
        if not choices:
            return ""
        message = choices[0].message if hasattr(choices[0], "message") else choices[0].get("message")
        if message is None:
            return ""
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(seg.get("text", "") if isinstance(seg, dict) else str(seg) for seg in content)
        return ""

    def send_prompt(prompt: str) -> str:
        last_error = None
        try:
            resp = client.responses.create(
                model=args.model,
                input=prompt,
                temperature=args.temp,
                max_output_tokens=args.max_output_tokens,
            )
            text = extract_from_response(resp)
            if text:
                return text
        except Exception as exc:
            last_error = exc
        # Fallback to chat completions
        chat_resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.temp,
            max_tokens=args.max_output_tokens,
        )
        text = extract_from_chat(chat_resp)
        if text:
            return text
        if last_error:
            raise last_error
        raise RuntimeError("Empty response from model")

    def call_model_with_retries(prompt: str):
        delay = float(args.retry_backoff)
        last_err = None
        for attempt in range(1, int(args.max_retries) + 1):
            try:
                text = send_prompt(prompt)
                return text, None
            except Exception as e:
                last_err = str(e)
                if attempt >= int(args.max_retries):
                    return "", last_err
                _time.sleep(delay)
                delay *= 2
        return "", last_err

    def eval_one(idx: int, item: Dict[str, Any]):
        nonlocal printed_api_ok
        n_local = int(item.get("n", 0))
        aliases_local = item.get("answer_aliases") or [item["answer_id"]]
        # Hygiene: require >= 2 f_{n-1} lines
        assert sum(1 for _, r, _ in (item.get("facts_bag") or []) if r == f"f{n_local-1}") >= 2
        # Order trials
        trial_ems_local = []
        last_prompt_local = ""; last_raw_local = ""; pred_local = ""; err_local = None
        reasoning_tokens_local: list[str] | None = None
        reasoning_complete_local: bool | None = None
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
                prompt = build_prompt(trial_item, reasoning_steps=bool(args.reasoning_steps))
                raw_text, err = call_model_with_retries(prompt)
                if raw_text:
                    raw_text = raw_text.strip()
                    pred_local = extract_token_from_response(raw_text)
                    err_local = None
                    last_prompt_local = prompt
                    last_raw_local = raw_text
                    if args.reasoning_steps:
                        reasoning_tokens_local, reasoning_complete_local = analyze_reasoning(raw_text, n_local)
                        if reasoning_complete_local and reasoning_tokens_local:
                            final_step = reasoning_tokens_local[-1]
                            pred_local = final_step if final_step else ""
                    if not printed_api_ok:
                        with lock:
                            if not printed_api_ok:
                                print(f"[benchmark/eval] API OK model={args.model}", flush=True)
                                printed_api_ok = True
                else:
                    pred_local = ""; last_raw_local = ""; err_local = err or ""
                    with lock:
                        print(f"[benchmark/eval] API ERROR item={item.get('id')} err={err_local}", flush=True)
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
        if args.reasoning_steps:
            rec_local["reasoning_steps"] = True
            if reasoning_tokens_local is not None:
                rec_local["reasoning_tokens"] = reasoning_tokens_local
            if reasoning_complete_local is not None:
                rec_local["reasoning_complete"] = reasoning_complete_local
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
            print(f"[benchmark/eval] {idx+1}/{len(items)} em={is_em_local}", flush=True)
        return rec_local

    if int(args.concurrency) > 1 and not args.baseline:
        with ThreadPoolExecutor(max_workers=int(args.concurrency)) as ex:
            futures = [ex.submit(eval_one, idx, item) for idx, item in enumerate(items)]
            for fut in as_completed(futures):
                try:
                    outputs.append(fut.result())
                except Exception as e:
                    print(f"[benchmark/eval] worker error: {e}", flush=True)
    else:
        for idx, item in enumerate(items):
            outputs.append(eval_one(idx, item))
            if args.sleep > 0:
                time.sleep(args.sleep)

    if fout:
        fout.close()

    def acc(c, n):
        return 0.0 if n == 0 else c / n
    total_n = perf["total"]["n"]
    total_c = perf["total"]["correct"]
    print("\n=== Benchmark n-hop Accuracy (Exact Match) ===")
    print(f"TOTAL: {total_c}/{total_n} = {acc(total_c, total_n):.3f}")
    print("By n (hops):")
    for n_key, v in sorted(perf["by_n"].items()):
        print(f"{n_key:>3} hops: {v['correct']}/{v['n']} = {acc(v['correct'], v['n']):.3f}")


if __name__ == "__main__":
    main()
