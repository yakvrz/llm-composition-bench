# n-hop Composition Benchmark (Symbolic, Open-book)

**Goal:** measure pure compositional reasoning over n-hop chains in an open-book setting. The repo now focuses on the implicit regime only: a bag-of-facts prompt containing m disjoint chains whose first n−1 hops must be composed.

## What's included

- `implicit/` — bag-of-facts generator and evaluator
  - `implicit/generate.py` — emits n-hop chains with balanced relation counts across m distractors
  - `implicit/evaluate.py` — queries the composed answer token using OpenAI's Responses API
- `scripts/sweep.py` — implicit-only sweep utility (`runs/implicit/<tag>/` layout: data/, results/, plots/, results.csv)
- `scripts/run_sweep.py` — orchestration helper to launch sweeps from JSON configs, plot results, index runs/index.csv, and emit brief reports
- `scripts/plot_sweep.py` — plots accuracy and lift-over-chance vs. n (one curve per m)
- `scripts/run_diagnostics.py` — quick variant runner (control / ablate / baseline, optional reasoning) for a single (n, m)
- `tests/` — smoke tests for generation + evaluation

## Data format (JSONL)

Implicit item example:
```json
{
  "id": "IMP-000123",
  "type": "implicit",
  "n": 5,
  "m": 8,
  "facts_bag": [["A_0001","f1","B_0100"], ..., ["D_0123","f4","E_0999"]],
  "question": "What is f4 of f3 of f2 of f1 of A_0001?",
  "answer_id": "E_0420",
  "answer_aliases": ["E_0420"]
}
```

## Parameters (implicit)

- `--items`: number of instances to generate
- `--hops`: hop count n
- `--m`: number of chains mixed into the bag (balanced per relation; ≥4 recommended)
- `--M`: layer size (token pool per hop)
- `--id_width`: zero-padded width for token IDs (shorter IDs shrink prompt length)
- `--seed`: RNG seed
- `--ablate_inner --ablate_hop j`: optional ablation removing relation f_j across all chains to create ambiguity baselines
- `--reasoning_steps`: add structured step-by-step slots to the prompt so the model must spell out each hop before answering

## Quickstart

Prereqs: Python 3.9+, `pip install openai python-dotenv`. Create a `.env` with:
```
OPENAI_API_KEY=sk-...
```

Generate and evaluate a small implicit dataset:
```bash
python implicit/generate.py --items 200 --hops 5 --m 8 --M 512 --seed 123 --out data/implicit.jsonl
python implicit/evaluate.py --in data/implicit.jsonl --out runs/implicit_run.jsonl --model gpt-4.1-mini --temp 0.0 --max_output_tokens 16 | tee runs/implicit_run.log
```

Sweep over n and m for a quick baseline:
```bash
python scripts/sweep.py --items 60 --hops 4,5,6,7,8 --m_list 4,8,12 --seeds 7,13

# With structured reasoning slots
python scripts/sweep.py --items 60 --hops 4,5,6,7,8 --m_list 4 --seeds 13,27 --reasoning_steps --max_output_tokens 64
```

### Shared test set (n = 4/6/8, m = 4)

To keep evaluations comparable, the repo caches implicit datasets under `data/testsets/` (seed 13, `M=256`, `id_width=3`). Reuse them instead of regenerating each run:

```bash
python implicit/evaluate.py \
  --in data/testsets/implicit_n4_m4_seed13.jsonl \
  --in data/testsets/implicit_n6_m4_seed13.jsonl \
  --in data/testsets/implicit_n8_m4_seed13.jsonl \
  --model openai/gpt-4o --temp 0.0 --max_output_tokens 48 \
  --out runs/manual_eval/gpt-4o.jsonl
```

You can still synthesize new splits via `implicit/generate.py`, but using the shared set avoids spending tokens repeatedly and guarantees apples-to-apples comparisons across models.

### Using OpenRouter models

`implicit/evaluate.py` will automatically route through OpenRouter when the following environment variables are present:

```
export OPENROUTER_API_KEY=sk-or-...
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export OPENAI_HTTP_REFERER=https://your.site
export OPENAI_X_TITLE="lllm-comp"
```

The client first tries the Responses API and falls back to Chat Completions when needed, so both OpenAI-hosted and OpenRouter-hosted models work out of the box.

## Experiment runner

Use `scripts/run_sweep.py` to manage sweeps, plots, and indexing:
```bash
python scripts/run_sweep.py run --config configs/implicit_small.json
python scripts/run_sweep.py plot --summary runs/implicit/sweep_YYYYMMDD_HHMMSS/results.csv --outdir runs/implicit/sweep_YYYYMMDD_HHMMSS/plots --label gpt-4.1-mini
python scripts/run_sweep.py report --summary runs/implicit/sweep_YYYYMMDD_HHMMSS/results.csv
```

Runs are stored under `runs/implicit/sweep_<timestamp>/` with subfolders `data/`, `results/`, `plots/`, and an aggregate `results.csv`.

## Diagnostics and baselines

- Order invariance: the evaluator accepts `--order_trials T` to reshuffle the facts bag per item.
- Concurrency: `--concurrency`, `--max_retries`, and `--retry_backoff` control request parallelism and retry behaviour.
- Baseline: `--baseline pointer_f_n1` selects the tail of the first f_{n-1} fact (chance ≈ 1/m).
- Ablation: `--ablate_inner --ablate_hop j` in the generator removes the j-th relation, pushing accuracy towards the baseline.
- Diagnostics: `scripts/run_diagnostics.py --items 60 --n 6 --m 8 --M 256 --seed 7 --variants control,ablate,baseline --reasoning_steps` records EM/lift per variant while also logging reasoning compliance.

## Plots

`scripts/plot_sweep.py` writes two figures per sweep into `plots/`:
- `lines_acc_vs_n_by_m.png` — mean exact-match accuracy vs. n with one curve per m
- `lines_lift_vs_n_by_m.png` — lift over chance vs. n (same curves)

## Tests

Smoke tests validate end-to-end generation and evaluation:
```bash
python -m pytest -q tests/
```

Run the implicit-only smoke test directly:
```bash
python -m pytest -q tests/test_implicit.py
```

## Notes

- `scripts/sweep.py` and `scripts/run_sweep.py` now default to implicit-only behaviour.
- Scratchpad prompting has been removed; prompts now present only the shuffled facts and the final answer line.
- Shared evaluation sets live under `data/testsets/`; add new JSONL files there when you need additional seeds or hop counts.
