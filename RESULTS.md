# RESULTS: Symbolic n-hop Composition (Open-book, Laddered)

This document summarizes the results and insights from the current benchmark, which measures multi-step composition over symbolic n-hop chains using an open-book prompt with laddered candidate sets.

## Setup summary
- Defaults (unless stated): model=gpt-4.1-mini, temp=0.0, M=256.
- Data: synthetic layers L0..Ln with bijective maps f1..fn (random permutations).
- Explicit prompt (laddered):
  - Facts: shuffled chain triples for f1..f{n−2}, mixed with K distractor chains (multi-chain context).
  - Candidates: sets for f{n−2}, f{n−1}, and f_n (decision depth L controls how many trailing functions are laddered).
- Implicit prompt (bag-of-facts):
  - Facts: m chains’ first n−1 hops (balanced exactly m lines per relation), shuffled.
- Answers: explicit → final token a = f_n(y_{n−1}); implicit → y_{n−1} token.
- Metric: Exact Match (EM) with normalization. Plots include lift-over-chance overlays.

Additional diagnostics supported:
- Order invariance (`--order_trials T`): reshuffle displays T times; ΔEM ≈ 0 expected.
- Pointer baseline (implicit) (`--baseline pointer_f_n1`): tail of first f_{n−1} line.
- Ablation (implicit) (`--ablate_inner --ablate_hop j`): remove one inner hop.
- Path-collision stress (explicit) (`--path_collision_stress`): coherent distractor paths.

## Key results

### A. Easier settings (single-chain or two-level ladder)
- Chain-only (f1..f{n−1} shown, no candidates): EM ≈ 1.000 at n=4 (trivial; answer can be read directly).
- Two-level ladder (f{n−1}, f_n), single-chain facts: high EM even at larger n (e.g., n=5 → 0.920; n=8 → 0.950). Difficulty does not scale with n because head-equality cues suffice.

### B. Explicit (multi-chain + shuffled facts + laddered candidates)
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

### C. Implicit (bag-of-facts; no candidates)
- Parameters (latest sweep 2025-09-14): items=24, m∈{4,8,12,16}, n∈{4..8}, M=256, seeds∈{7,13}, model=gpt-4.1-mini.
- EM by (n, m) mean±sd:
  - n=4: m=4 0.896±0.021; m=8 0.770±0.062; m=12 0.730±0.062; m=16 0.584±0.041
  - n=5: m=4 0.584±0.041; m=8 0.291±0.084; m=12 0.354±0.062; m=16 0.146±0.021
  - n=6: m=4 0.334±0.083; m=8 0.209±0.041; m=12 0.104±0.021; m=16 0.125±0.042
  - n=7: m=4 0.312±0.104; m=8 0.146±0.021; m=12 0.125±0.000; m=16 0.042±0.000
  - n=8: m=4 0.250±0.083; m=8 0.062±0.062; m=12 0.084±0.084; m=16 0.062±0.021
- Collapsed means: by n → 4:0.745, 5:0.344, 6:0.193, 7:0.156, 8:0.115; by m → 4:0.475, 8:0.296, 12:0.279, 16:0.192.
- Takeaway: accuracy drops sharply with n and with larger m, confirming the multi-chain entanglement pressure in the implicit setup.

Latest compact snapshots (model=gpt-4.1-mini, temp=0.0, M=256):
- Implicit (n∈{4,6,8}, m∈{6,8}, items=24, seed=7):
  - n=4: m=6 → 0.833; m=8 → 0.792
  - n=6: m=6 → 0.167; m=8 → 0.083
  - n=8: m=6 → 0.208; m=8 → 0.167
  Plots: `plots/implicit/sweep_20250915_023900/`
- Explicit (alias+balance+path-collision; n∈{4,6,8}, L∈{2,3}, items=24, seed=7):
  - n=4: L=2 → 0.667; L=3 → 0.708
  - n=6: L=2 → 0.208; L=3 → 0.208
  - n=8: L=2 → 0.250; L=3 → 0.417
  Plots: `plots/explicit/sweep_20250915_024440/`

Lift-over-chance (qualitative):
- Implicit lift decreases with n and m; near-chance behavior at higher n/m.
- Explicit lift drops after head-cue mitigation; L=3 no longer clearly exceeds L=2.

## Metrics and diagnostics
- Lift-over-chance:
  - Explicit: lift = (EM − 1/k_last) / (1 − 1/k_last)
  - Implicit: lift = (EM − 1/m) / (1 − 1/m)
- Analyzer (explicit):
  - f_n correct (pred==gold)
  - f_{n−1} coherent-from-chain (chosen f_n head reachable from y_{n−2} via some f_{n−1})
  - f_{n−2} from-chain (y_{n−3} → y_{n−2} candidate present)

## Historical results (pre head-cue mitigation)
These earlier runs (items=60; seeds=7/13/23) predate block head balancing and aliasing. Kept here for context.

### Easier settings (single-chain or two-level ladder)
- Chain-only (f1..f{n−1} shown, no candidates): EM ≈ 1.000 at n=4 (trivial; answer can be read directly).
- Two-level ladder (f{n−1}, f_n), single-chain facts: high EM even at larger n (e.g., n=5 → 0.920; n=8 → 0.950). Difficulty does not scale with n because head-equality cues suffice.

### Hard setting (multi-chain + shuffled facts + three-level ladder)
- Depth curve:
  - n=4: 0.933 EM (56/60)
  - n=5: 0.633 EM (38/60)
  - n=6: 0.267 EM (16/60)
- Sweep corroboration (different seed/reporting window):
  - n=4: 0.883; n=5: 0.667; n=6: 0.233

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
- Explicit (examples):
  - Generate: `python explicit/generate.py --items 60 --hops 6 --k 6 --L 3 --context_chains 8 --M 256 --seed 7 --out data/explicit.jsonl [--block_head_balance --alias_heads_per_block --path_collision_stress]`
  - Evaluate: `python explicit/evaluate.py --in data/explicit.jsonl --out runs/<ts>/explicit.jsonl --model gpt-4.1-mini --temp 0.0 --max_output_tokens 16 [--order_trials 3]`
  - Analyze: `python explicit/analyze_errors.py --in runs/<ts>/explicit.jsonl`
- Implicit (examples):
  - Generate: `python implicit/generate.py --items 60 --hops 6 --m 8 --M 256 --seed 7 --out data/implicit.jsonl [--ablate_inner --ablate_hop 3]`
  - Evaluate: `python implicit/evaluate.py --in data/implicit.jsonl --out runs/<ts>/implicit.jsonl --model gpt-4.1-mini --temp 0.0 --max_output_tokens 16 [--baseline pointer_f_n1 --order_trials 3]`
- Sweeps: `python sweep.py --approach explicit|implicit ...` (artifacts written under `runs/sweep_<ts>/flat/` as `results_<tag>.jsonl`, `summary_<tag>.txt`, `gen_stdout_<tag>.txt`) and `summary.csv` at the sweep root.
