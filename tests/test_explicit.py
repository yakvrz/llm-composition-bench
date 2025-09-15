import subprocess, sys, json, os


def test_explicit_smoke(tmp_path):
    ds = tmp_path / 'exp.jsonl'
    res = tmp_path / 'res.jsonl'
    # Generate small dataset
    cmd_gen = [sys.executable, 'explicit/generate.py', '--items', '6', '--hops', '4', '--L', '2', '--k', '4', '--context_chains', '2', '--M', '64', '--seed', '7', '--out', str(ds)]
    r = subprocess.run(cmd_gen, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert ds.exists()
    # Evaluate quickly
    cmd_eval = [sys.executable, 'explicit/evaluate.py', '--in', str(ds), '--out', str(res), '--model', 'gpt-4.1-mini', '--temp', '0.0', '--max_output_tokens', '16', '--n', '6', '--order_trials', '1', '--concurrency', '2']
    r2 = subprocess.run(cmd_eval, capture_output=True, text=True)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert res.exists()
    # Check file has 6 lines
    with open(res, 'r', encoding='utf-8') as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    assert len(lines) == 6


