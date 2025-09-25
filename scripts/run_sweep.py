#!/usr/bin/env python3
"""
Experiment helper CLI for implicit sweeps: run configs, plot results, and write per-run summaries.

Commands:
  - run   --config CONFIG.json       → runs scripts/sweep.py, plots to run root, indexes runs/index.csv
  - plot  --summary SUMMARY.csv      → generates plots into <run_root>/plots/
  - report --summary SUMMARY.csv     → writes a short markdown results.md into the run root

Config (JSON) examples: see configs/.
"""

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    print(f"[exp] RUN: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines: list[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end='')
            lines.append(line)
        proc.wait()
        return (proc.returncode or 0), ''.join(lines)
    finally:
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except Exception:
                pass


def parse_summary_path(stdout: str) -> Path | None:
    for line in stdout.strip().splitlines()[::-1]:
        if 'Summary:' in line:
            parts = line.rsplit('Summary:', 1)
            if len(parts) == 2:
                return Path(parts[1].strip())
    return None


def infer_tag_from_summary(summary: Path) -> str:
    return summary.parent.name


def write_index_row(summary: Path, config_path: Path, plots_dir: Path, extras: dict):
    index_path = Path('runs') / 'index.csv'
    ensure_dir(index_path.parent)
    header = ['ts','approach','config','sweep_root','results_csv','plots_dir','model','items','extras']
    exists = index_path.exists()
    if not exists:
        index_path.write_text(','.join(header) + '\n', encoding='utf-8')
    ts = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    sweep_root = summary.parent.as_posix()
    row = [
        ts,
        'implicit',
        config_path.as_posix(),
        sweep_root,
        summary.as_posix(),
        plots_dir.as_posix(),
        str(extras.get('model','')),
        str(extras.get('items','')),
        json.dumps(extras, ensure_ascii=False).replace(',', ';'),
    ]
    with index_path.open('a', encoding='utf-8') as f:
        f.write(','.join(row) + '\n')


def summarize_for_report(summary_path: Path) -> str:
    import csv
    import statistics
    rows = []
    with summary_path.open('r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    nm: dict[tuple[int, int], list[float]] = {}
    for r in rows:
        if r.get('approach') != 'implicit':
            continue
        n = int(r['n'])
        m = int(r['m'])
        acc = float(r['acc'])
        nm.setdefault((n, m), []).append(acc)
    lines = ['- Implicit EM (mean across seeds) by (n, m):']
    for (n, m) in sorted(nm):
        vals = nm[(n, m)]
        lines.append(f"  - n={n}, m={m}: {statistics.mean(vals):.3f}")
    return '\n'.join(lines)


def cmd_run(args):
    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    approach = cfg.get('approach', 'implicit')
    if approach != 'implicit':
        raise ValueError(f"Only implicit approach is supported (got {approach!r})")

    cmd = [sys.executable, 'scripts/sweep.py']

    def add_list(flag: str, values):
        if values is None:
            return
        if isinstance(values, list):
            cmd.extend([flag, ','.join(str(x) for x in values)])
        else:
            cmd.extend([flag, str(values)])

    if 'items' in cfg:
        cmd.extend(['--items', str(cfg['items'])])
    add_list('--hops', cfg.get('hops'))
    add_list('--m_list', cfg.get('m_list'))
    if 'M' in cfg:
        cmd.extend(['--M', str(cfg['M'])])
    if 'id_width' in cfg:
        cmd.extend(['--id_width', str(cfg['id_width'])])
    add_list('--seeds', cfg.get('seeds'))
    if 'model' in cfg:
        cmd.extend(['--model', str(cfg['model'])])
    if 'temp' in cfg:
        cmd.extend(['--temp', str(cfg['temp'])])
    if 'max_output_tokens' in cfg:
        cmd.extend(['--max_output_tokens', str(cfg['max_output_tokens'])])
    if 'order_trials' in cfg:
        cmd.extend(['--order_trials', str(cfg['order_trials'])])
    if cfg.get('baseline'):
        cmd.extend(['--baseline', str(cfg['baseline'])])
    if cfg.get('ablate_inner'):
        cmd.append('--ablate_inner')
    if cfg.get('ablate_hop') is not None:
        cmd.extend(['--ablate_hop', str(cfg['ablate_hop'])])
    if cfg.get('reasoning_steps'):
        cmd.append('--reasoning_steps')
    if cfg.get('save_prompt'):
        cmd.append('--save_prompt')
    if cfg.get('save_raw_output'):
        cmd.append('--save_raw_output')
    if cfg.get('log_first'):
        cmd.extend(['--log_first', str(cfg['log_first'])])
    if cfg.get('concurrency'):
        cmd.extend(['--concurrency', str(cfg['concurrency'])])
    if cfg.get('max_retries') is not None:
        cmd.extend(['--max_retries', str(cfg['max_retries'])])
    if cfg.get('retry_backoff') is not None:
        cmd.extend(['--retry_backoff', str(cfg['retry_backoff'])])

    rc, out = run_cmd(cmd)
    if rc != 0:
        print('Sweep failed.', file=sys.stderr)
        sys.exit(rc)
    summary = parse_summary_path(out)
    if not summary or not summary.exists():
        print('Could not locate summary.csv in output.', file=sys.stderr)
        sys.exit(2)

    plots_dir = summary.parent / 'plots'
    ensure_dir(plots_dir)
    label = cfg.get('label') or ''
    plot_cmd = [sys.executable, 'scripts/plot_sweep.py', '--summary', str(summary), '--outdir', str(plots_dir)]
    if label:
        plot_cmd += ['--label', str(label)]
    rc2, _ = run_cmd(plot_cmd)
    if rc2 != 0:
        sys.exit(rc2)

    write_index_row(summary, cfg_path, plots_dir, extras={
        'model': cfg.get('model', ''),
        'items': cfg.get('items', ''),
        'config': cfg,
    })
    print(f'Run complete. Summary: {summary}\nPlots: {plots_dir}')


def cmd_plot(args):
    outdir = Path(args.outdir)
    ensure_dir(outdir)
    plot_cmd = [sys.executable, 'scripts/plot_sweep.py', '--summary', args.summary, '--outdir', str(outdir)]
    if getattr(args, 'label', ''):
        plot_cmd += ['--label', args.label]
    rc, out = run_cmd(plot_cmd)
    print(out, end='')


def cmd_report(args):
    summary = Path(args.summary)
    tag = infer_tag_from_summary(summary)
    run_root = summary.parent
    plots_dir = run_root / 'plots'

    ensure_dir(run_root)
    report_path = run_root / 'results.md'
    body_lines = [
        f"# Sweep {tag}",
        '',
        summarize_for_report(summary),
        '',
        f"Plots: {plots_dir.as_posix()}",
    ]
    report_path.write_text('\n'.join(body_lines) + '\n', encoding='utf-8')
    print(f'Wrote {report_path}')


def main():
    ap = argparse.ArgumentParser(description='Implicit sweep helper CLI')
    sub = ap.add_subparsers(dest='cmd', required=True)

    ap_run = sub.add_parser('run', help='Run sweep, plot, and index from a JSON config')
    ap_run.add_argument('--config', required=True)
    ap_run.set_defaults(func=cmd_run)

    ap_plot = sub.add_parser('plot', help='Plot results for an existing sweep')
    ap_plot.add_argument('--summary', required=True)
    ap_plot.add_argument('--outdir', required=True)
    ap_plot.add_argument('--label', default='')
    ap_plot.set_defaults(func=cmd_plot)

    ap_report = sub.add_parser('report', help='Write a short markdown report for a sweep')
    ap_report.add_argument('--summary', required=True)
    ap_report.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
