# n-hop Composition Benchmark (Symbolic, Open-book)

**Goal:** measure pure compositional reasoning over n-hop chains in open-book settings. Supports two approaches: explicit (candidate blocks) and implicit (bag-of-facts without candidates).

## What's included

- `explicit/` — candidate-based approach (laddered candidates with multi-chain facts)
  - `explicit/generate.py` — generator for laddered candidate blocks (supports decision depth L)
  - `explicit/evaluate.py` — evaluator for explicit prompts
  - `explicit/analyze_errors.py` — error analysis for explicit results
- `implicit/` — bag-of-facts approach (no candidates; m chains mixed)
  - `implicit/generate.py` — generator emitting m chains’ first n−1 hops (balanced per relation)
  - `implicit/evaluate.py` — evaluator asking for y_{n−1} from the bag
- `scripts/sweep.py` — sweep utility (`runs/<approach>/<tag>/` layout: data/, results/, plots/, summary.csv)

## Data formats (JSONL)

Explicit item:
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
    ["C_0220","f3","D_0901"],
    ["C_0420","f3","D_0199"],
    ["C_1111","f3","D_0555"],
    ["C_0002","f3","D_0303"]
  ],
  "candidates_fn_1": [
    ["D_0901","f4","E_0073"],
    ["D_0420","f4","E_0199"],
    ["D_1111","f4","E_0555"],
    ["D_0002","f4","E_0303"]
  ],
  "candidates_fn": [
    ["E_0073","f5","F_0824"],
    ["E_0420","f5","F_0199"],
    ["E_1111","f5","F_0555"],
    ["E_0002","f5","F_0303"]
  ],
  "question": "What is f5 of f4 of f3 of f2 of f1 of A_0047?",
  "answer_id": "F_0824",
  "answer_aliases": ["F_0824"]
}
```

Implicit item:
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

## Parameters (explicit)

- `--items`: number of instances
- `--hops`: hop count n
- `--k` (or `--k0`, `--k1`, `--k2`): candidate counts per block
- `--L`: decision depth (how many trailing functions have candidate blocks)
- `--context_chains`: number of distractor chains to mix in
- `--M`: layer size (tokens per layer)
- `--seed`: RNG seed
- `--chain_only_ordered`: emit only the true chain facts in order (sanity check; expect EM≈1). Analyses should always shuffle.

## Parameters (implicit)

- `--items`: number of instances
- `--hops`: hop count n
- `--m`: number of chains to include in the bag (balanced per relation; ≥4)
- `--M`: layer size (tokens per layer)
- `--seed`: RNG seed

## Quickstart

Prereqs: Python 3.9+, `pip install openai python-dotenv`. Create a `.env` with:
```
OPENAI_API_KEY=sk-...
```

Generate and evaluate:
```bash
# 1) Explicit: generate (laddered candidates)
python explicit/generate.py --items 200 --hops 4 --k 6 --L 3 --context_chains 6 --M 512 --seed 123 --out data/explicit.jsonl

# 2) Explicit: evaluate
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p runs/$TS
python explicit/evaluate.py --in data/explicit.jsonl --out runs/$TS/explicit.jsonl --model gpt-4.1-mini --temp 0.0 --max_output_tokens 16 | tee runs/$TS/explicit.txt

# (Optional) Explicit: error analysis
python explicit/analyze_errors.py --in runs/$TS/explicit.jsonl --examples_per_type 3

# (Optional) Explicit: parameter sweep
python sweep.py --approach explicit --items 60 --hops 4,5,6 --Ls 2,3 --k 6 --contexts 8 --seeds 7,13,23

# 3) Implicit: generate (bag-of-facts) and evaluate
python implicit/generate.py --items 200 --hops 5 --m 8 --M 512 --seed 123 --out data/implicit.jsonl
python implicit/evaluate.py --in data/implicit.jsonl --out runs/$TS/implicit.jsonl --model gpt-4.1-mini --temp 0.0 --max_output_tokens 16 | tee runs/$TS/implicit.txt

# (Optional) Implicit: sweep over n and m
python sweep.py --approach implicit --items 24 --hops 4,5,6,7,8 --m_list 4,8,12,16 --seeds 7,13
```

## Experiment runner

For convenience, use `scripts/exp.py` to orchestrate sweeps, plots, and a run index (and per-run results.md):

```bash
# Run a sweep from a JSON config, auto-plot, and index the run
python scripts/exp.py run --config configs/implicit_small.json
python scripts/exp.py run --config configs/explicit_small.json

# Generate plots for an existing summary
python scripts/exp.py plot --summary runs/implicit/sweep_YYYYMMDD_HHMMSS/summary.csv --approach implicit --outdir runs/implicit/sweep_YYYYMMDD_HHMMSS/plots

# Append a short section to REPORTS.md for a run
python scripts/exp.py report --summary runs/implicit/sweep_YYYYMMDD_HHMMSS/summary.csv --approach implicit
```

Unified run layout
- Artifacts are written under `runs/<approach>/sweep_<timestamp>/`:
  - `data/` (generated datasets)
  - `results/` (results_*.jsonl and summary_*.txt)
  - `plots/` (plots generated for the run)
  - `summary.csv` (aggregate across combos)
  - `results.md` (short per-run report)

## Diagnostics and baselines

- Order invariance: evaluators accept `--order_trials T` to reshuffle facts (and candidate blocks for explicit) T times per item and aggregate EM.
- Concurrency: evaluators support `--concurrency K` with retries (`--max_retries`, `--retry_backoff`) and progressive writes/prints.
- Pointer baseline (implicit): `--baseline pointer_f_n1` outputs the tail of the first f_{n−1} line; expected ≈1/m.
- Ablation (implicit): generator flags `--ablate_inner --ablate_hop j` remove one inner hop across all chains to induce ambiguity; EM should approach ≈1/m.
- Path-collision stress (explicit): generator flag `--path_collision_stress` ensures coherent distractor paths across ladder levels in candidate blocks.
- Head-cue mitigation (explicit): `--block_head_balance` equalizes head exposure per ladder block (against prior printed content), and `--alias_heads_per_block` prints a block-local alias map to neutralize identity cues.
- Lift-over-chance plots: plotting adds lift overlays for both approaches.

### Analyzer metrics (explicit)

Run:
```
python explicit/analyze_errors.py --in runs/<tag>/results.jsonl
```
Outputs include per-block signals:
- `f_n correct`: whether final choice matches gold
- `f_{n-1} coherent-from-chain`: chosen f_n head reachable from y_{n-2} via some f_{n-1}
- `f_{n-2} from-chain`: existence of y_{n-3} → y_{n-2} support in f_{n-2} candidates

## Current snapshot

Compact runs (gpt-4.1-mini, temp=0.0, M=256; items=24; seed=7):
- Implicit (bag-of-facts):
  - n=4: m=6 → 0.833; m=8 → 0.792
  - n=6: m=6 → 0.167; m=8 → 0.083
  - n=8: m=6 → 0.208; m=8 → 0.167
- Explicit (laddered, with `--block_head_balance --alias_heads_per_block --path_collision_stress`):
  - n=4: L=2 → 0.667; L=3 → 0.708
  - n=6: L=2 → 0.208; L=3 → 0.208
  - n=8: L=2 → 0.250; L=3 → 0.417

See plots under `plots/implicit/sweep_20250915_023900/` and `plots/explicit/sweep_20250915_024440/`.

## Evaluation details

- **Explicit Prompt:** mixed chain facts from target + distractor chains + laddered candidate sets for f_{n-2}..f_n; answer with the final token.
- **Implicit Prompt:** bag of facts (m chains × first n−1 hops); compose to find y_{n−1} and answer with that token.
- **Metric:** Exact Match (EM) against `answer_aliases` (case/spacing-robust normalization).
- **Output:** console summary + `results.jsonl`.

## Design choices

- Bijective maps per hop ensure a unique answer and constant difficulty.
- Multi-chain context creates realistic distraction without changing the core reasoning task.
- Laddered candidates (explicit) and bag-of-facts (implicit) provide complementary stress tests.
- Symbolic tokens avoid NL artifacts; naturalized surfaces can be layered later if needed.
