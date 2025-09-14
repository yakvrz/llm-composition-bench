# n-hop Composition Benchmark (Symbolic, Open-book, Multi-chain Context)

**Goal:** measure pure compositional reasoning over n-hop chains in open-book settings. The prompt provides multi-chain context facts mixed from the target chain plus distractors, with laddered candidate sets for f_{n-2}, f_{n-1}, and f_n; the model must compose through the chain to select the correct final answer.

## What's included

- `generate_data.py` — symbolic generator for n-hop chains over typed layers (L0..Ln) with bijective maps f1..fn; emits mixed chain facts from target + K distractor chains, plus laddered candidate sets with exposure balancing.
- `evaluate.py` — open-book evaluator using the Responses API; builds multi-chain prompt with laddered candidates and reports Exact Match (EM) overall and by hop count (n).
- `analyze_errors.py` — offline error analysis (compliance, structural coherence, error taxonomy, examples).
- `sweep.py` — parameter sweep utility for n, candidate sizes, contexts, and seeds; writes CSV summaries.

## Data format (JSONL)

Each line is one item:
```json
{
  "id": "SYN-000123",
  "type": "synthetic",
  "n": 4,
  "facts_chain": [
    ["A_0047","f1","B_0132"],
    ["B_0132","f2","C_0220"],
    ["D_0555","f1","E_0234"],
    ["C_0420","f2","D_0199"],
    ["A_0234","f1","B_0789"],
    ["E_0234","f2","F_0456"]
  ],
  "candidates_fn_2": [
    ["C_0220","f3","D_0901"],  // correct f3 (target chain)
    ["C_0420","f3","D_0199"],
    ["C_1111","f3","D_0555"],
    ["C_0002","f3","D_0303"]
  ],
  "candidates_fn_1": [
    ["D_0901","f4","E_0073"],  // correct f4 (target chain)
    ["D_0420","f4","E_0199"],
    ["D_1111","f4","E_0555"],
    ["D_0002","f4","E_0303"]
  ],
  "candidates_fn": [
    ["E_0073","f5","F_0824"],  // correct f5 (target chain)
    ["E_0420","f5","F_0199"],
    ["E_1111","f5","F_0555"],
    ["E_0002","f5","F_0303"]
  ],
  "question": "What is f5 of f4 of f3 of f2 of f1 of A_0047?",
  "answer_id": "F_0824",
  "answer_aliases": ["F_0824"]
}
```

## Parameters

- `--items`: number of instances
- `--hops`: hop count n (default 4)
- `--k0`: number of candidates for f_{n-2}
- `--k1`: number of candidates for f_{n-1}
- `--k2`: number of candidates for f_n
- `--context_chains`: number of distractor chains to mix in (default 6)
- `--M`: layer size (tokens per layer; default 512)
- `--seed`: RNG seed

## Quickstart

Prereqs: Python 3.9+, `pip install openai python-dotenv`. Create a `.env` with:
```
OPENAI_API_KEY=sk-...
```

Generate and evaluate:
```bash
# 1) Generate symbolic data (multi-chain context with laddered candidates)
python generate_data.py --items 200 --hops 4 --k0 4 --k1 4 --k2 4 --context_chains 6 --M 512 --seed 123 --out data/synth.jsonl

# 2) Open-book evaluation (EM; saves per-item outputs)
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p runs/$TS
python evaluate.py --in data/synth.jsonl --out runs/$TS/results.jsonl --temp 0.0 --max_output_tokens 16 | tee runs/$TS/summary.txt

# (Optional) Error analysis
python analyze_errors.py --in runs/$TS/results.jsonl --examples_per_type 3

# (Optional) Parameter sweep
python sweep.py --items 60 --hops 4,5,6 --k0s 6 --k1s 6 --k2s 6 --contexts 8 --seeds 7,13,23
```

## Evaluation details

- **Prompt:** mixed chain facts from target + distractor chains + laddered candidate sets for f_{n-2}, f_{n-1}, and f_n; answer with the final token.
- **Metric:** Exact Match (EM) against `answer_aliases` (case/spacing-robust normalization).
- **Output:** console summary + `results.jsonl` with: `id, n, question, facts_chain, candidates_fn_2, candidates_fn_1, candidates_fn, gold, pred, em, error`.

## Design choices

- Bijective maps per hop ensure a unique answer and constant difficulty.
- Multi-chain context creates realistic distraction without changing the core reasoning task.
- Laddered candidates (f_{n-2}, f_{n-1}, f_n) with exposure balancing ensure fair evaluation.
- Controllable difficulty via `n` (chain length), `k` (candidate count), and `context_chains` (distraction level).
- Symbolic tokens avoid NL artifacts; naturalized surfaces can be layered later if needed.
