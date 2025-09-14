# n-hop Composition Benchmark (Symbolic, Open-book, laddered candidates)

**Goal:** measure pure compositional reasoning over n-hop chains in open-book settings. The prompt shows the early chain facts (f1..f{n−2}) and provides candidate sets for both f{n−1} and f{n}; the model must select the correct final token after composing.

## What’s included

- `generate_data.py` — symbolic generator for n-hop chains over typed layers (L0..Ln) with bijective maps f1..fn; emits chain facts (f1..f{n-2}) and candidate sets for f{n-1} and f{n}.
- `evaluate.py` — open-book evaluator using the Responses API; builds a laddered prompt (chain + two/three candidate blocks) and reports Exact Match (EM) overall and by hop count (n).
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
    ["B_0132","f2","C_0220"]
  ],
  "candidates_fn_1": [
    ["C_0220","f3","D_0901"],  // correct f3
    ["C_0420","f3","D_0199"],
    ["C_1111","f3","D_0555"],
    ["C_0002","f3","D_0303"]
  ],
  "candidates_fn": [
    ["D_0901","f4","E_0073"],  // correct f4
    ["D_0420","f4","E_0199"],
    ["D_1111","f4","E_0555"],
    ["D_0002","f4","E_0303"]
  ],
  "question": "What is f4 of f3 of f2 of f1 of A_0047?",
  "answer_id": "E_0073",
  "answer_aliases": ["E_0073"]
}
```

## Parameters

- `--items`: number of instances
- `--hops`: hop count n (default 4)
- `--k1`: number of candidates for f_{n-1}
- `--k2`: number of candidates for f_n
- `--M`: layer size (tokens per layer; default 512)
- `--seed`: RNG seed

## Quickstart

Prereqs: Python 3.9+, `pip install openai python-dotenv`. Create a `.env` with:
```
OPENAI_API_KEY=sk-...
```

Generate and evaluate:
```bash
# 1) Generate symbolic data (laddered)
python generate_data.py --items 200 --hops 4 --k1 4 --k2 4 --M 512 --seed 123 --out data/synth.jsonl

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

- **Prompt:** early chain facts (f1..f{n-2}) + candidate sets for f{n-1} and f{n}; answer with the final token.
- **Metric:** Exact Match (EM) against `answer_aliases` (case/spacing-robust normalization).
- **Output:** console summary + `results.jsonl` with: `id, n, question, facts_chain, candidates_fn_1, candidates_fn, gold, pred, em, error`.

## Design choices

- Bijective maps per hop ensure a unique answer and constant difficulty.
- Controllable difficulty via `n` (chain length) and `k` (candidate count).
- Symbolic tokens avoid NL artifacts; naturalized surfaces can be layered later if needed.
