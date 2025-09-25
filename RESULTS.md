# Composition Runs with and without Structured Reasoning

This report compares two sweeps on the n-hop benchmark (m = 4, seeds 13 & 27, 120 items per combination, gpt-4.1-mini, temp 0). *EM* denotes exact-match accuracy, and *Lift* is `(accuracy − 1/m) / (1 − 1/m)`.

1. **Baseline prompt** – no reasoning scaffold (`runs/benchmark/sweep_20250917_204638/`).
2. **Structured reasoning prompt** – added “Step _i_” slots before the answer (`runs/benchmark/sweep_20250925_171052/`).

## Aggregate Accuracy & Lift (per n, averaged over two seeds)

| n | m | Baseline EM | Reasoning EM | Δ EM | Baseline Lift | Reasoning Lift | Δ Lift |
|---|---|-------------|--------------|------|---------------|----------------|--------|
| 2 | 4 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| 3 | 4 | 0.996 | 1.000 | +0.004 | 0.995 | 1.000 | +0.005 |
| 4 | 4 | 0.859 | 0.996 | +0.138 | 0.811 | 0.995 | +0.184 |
| 5 | 4 | 0.638 | 1.000 | +0.363 | 0.517 | 1.000 | +0.483 |
| 6 | 4 | 0.413 | 0.996 | +0.584 | 0.217 | 0.995 | +0.778 |
| 7 | 4 | 0.304 | 0.000 | −0.304 | 0.073 | −0.333 | −0.406 |
| 8 | 4 | 0.242 | 0.000 | −0.242 | −0.011 | −0.333 | −0.323 |

*Lift computed relative to 1/m chance.*

## Reasoning Compliance

The structured sweep records whether all reasoning steps were populated:

| n | Reasoning complete | Notes |
|---|--------------------|-------|
| 2 | 1.3% | Single-step prompt rarely filled; answer comes from copied fact |
| 3–6 | ≥99.6% | Model walks the chain and echoes each intermediate tail |
| 7 | 0% | Final step left blank; traces stop at f₆ |
| 8 | 0% | Same failure pattern as n=7 |

## Observations

- **Major boost for n≤6:** Once the model is cued to list each hop, it reliably spells out the correct intermediate tokens. Even if the “Final answer” line remains empty, the target appears in the last step, so the evaluator extracts the right tail. Accuracy climbs from ~0.64→1.00 at n=5 and ~0.41→0.996 at n=6.
- **Collapse for deeper chains:** At n≥7 the model fills early steps but fails to complete the last slot; the final hop never surfaces, so EM drops to zero despite partial reasoning.
- **Extraction behavior:** By design the evaluator falls back to “last token-like string” in the output. With structured reasoning, that string is the final hop tail whenever the steps are complete. Tightening extraction (e.g., require a dedicated “Final answer” line) would produce stricter scores.

## Next Steps

1. Adjust the template or evaluator to enforce non-empty step slots and a separate final answer, preventing accidental credit from partial traces.
2. Probe larger m values: see whether multi-chain distractors erode the structured benefits.
3. Instrument reasoning tokens for error analysis (e.g., compare hop-level accuracy vs. final EM).
