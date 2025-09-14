# RESULTS: Symbolic n-hop Composition (Open-book, Laddered)

This document summarizes the results and insights from the current benchmark, which measures multi-step composition over symbolic n-hop chains using an open-book prompt with laddered candidate sets.

## Setup summary
- Data: synthetic layers L0..Ln with bijective maps f1..fn (random permutations).
- Prompt (laddered):
  - Facts: shuffled chain triples for f1..f{n−2}, mixed with K distractor chains (multi-chain context).
  - Candidates: sets for f{n−2}, f{n−1}, and f_n (three-level ladder). Earlier variants used only f{n−1} and f_n (two-level) or no candidates (chain-only).
- Answer: final token a = f_n(y_{n−1}).
- Metric: Exact Match (EM) with normalization. We also track candidate compliance and structural coherence across ladder levels.

## Key results

### A. Easier settings (single-chain or two-level ladder)
- Chain-only (f1..f{n−1} shown, no candidates): EM ≈ 1.000 at n=4 (trivial; answer can be read directly).
- Two-level ladder (f{n−1}, f_n), single-chain facts: high EM even at larger n (e.g., n=5 → 0.920; n=8 → 0.950). Difficulty does not scale with n because head-equality cues suffice.

### B. Hard setting (multi-chain + shuffled facts + three-level ladder)
- Parameters (latest sweep 2025-09-14): items=24, k0=k1=k2=6, context_chains=8, M=256, seeds∈{7,13,23}, model=gpt-4.1-mini, temp=0.0.
- Per-seed EM (n=4..6):
  - n=4: 0.958, 0.958, 1.000 (mean±sd ≈ 0.972±0.020)
  - n=5: 0.917, 0.792, 0.708 (mean±sd ≈ 0.806±0.086)
  - n=6: 0.833, 0.792, 0.708 (mean±sd ≈ 0.778±0.052)
- Notes:
  - There is a clear drop from n=4 → n=5; n=5 → n=6 is relatively flat on this sample size.
  - Seed variance is noticeable at higher n (spread ≈8–21 EM points at n=5).
  - Independent shuffling, balanced head exposure, unique tails, and multi-chain hygiene prevent simple frequency/order tells.

Sweep artifacts saved under `runs/sweep_YYYYMMDD_HHMMSS/summary.csv` (e.g., `runs/sweep_20250914_201030/summary.csv`).

## Error analysis highlights (hard setting)
- Non-EM errors are largely composition mistakes across ladder levels (f_{n−2}→f_{n−1}→f_n), not candidate non-compliance.
- Chain-only ordered control remains trivial (EM≈1); analyses should always use shuffled mode.

## Takeaways
- Depth matters only when composition is required. Without shuffled multi-chain facts and a deeper ladder, models can exploit head equality and order, keeping EM high across n.
- With the harder spec (three-level ladder + multi-chain + shuffle), accuracy drops predictably with n, reflecting true multi-step composition difficulty.
- Error profile is principled: most failures are incoherent cross-level links (f_{n−1} → f_n), not candidate non-compliance or formatting issues.

## Recommended next investigations
- Parameter sensitivity:
  - Vary candidate sizes k0/k1/k2 (e.g., {4,6,8,10}) to map how EM scales with distractor density.
  - Vary number of context chains (e.g., {0,4,8,12}) to stress multi-chain disambiguation.
  - Extend depth curve to n=7 and n=8 under the hard spec.
- Prompt ablations:
  - Order invariance: confirm EM is stable under independent shuffles of facts and each candidate block.
  - Contrast sets: minimally flip one chain edge; EM should drop to ≈0 if composition is used.
- Reporting improvements:
  - Track per-item structural coherence for chosen paths across f_{n−2}→f_{n−1}→f_n.
  - Add plots (EM vs. n, EM vs. k) from `sweep.py` outputs.

## Reproduction notes
- Data generation (example):
  - `python generate_data.py --items 60 --hops 5 --k0 6 --k1 6 --k2 6 --context_chains 8 --M 512 --seed 7 --out data/synth_5_ladder3.jsonl`
- Evaluation (example):
  - `python evaluate.py --in data/synth_5_ladder3.jsonl --out runs/<ts>/results.jsonl --temp 0.0 --max_output_tokens 16`
- Error analysis (example):
  - `python analyze_errors.py --in runs/<ts>/results.jsonl --examples_per_type 3`
- Parameter sweep (example):
  - `python sweep.py --items 60 --hops 4,5,6 --k0s 6 --k1s 6 --k2s 6 --contexts 8 --seeds 7,13,23`

All artifacts are saved under `runs/<timestamp>/` with `results.jsonl` and `summary.txt`; sweeps also produce a `summary.csv` aggregating EM.
