#!/usr/bin/env python3
"""
Experiment helper CLI: run sweeps from JSON configs, plot results, and write per-run results.md.

Commands:
  - run   --config CONFIG.json              → runs scripts/sweep.py per config, plots to run root, indexes runs/index.csv
  - plot  --summary SUMMARY.csv --approach  → generates plots into <run_root>/plots/
  - report --summary SUMMARY.csv --approach → writes a short markdown results.md into the run root

Config (JSON) examples (implicit/explicit): see configs/.
"""
import argparse, json, subprocess, sys, datetime as dt
from pathlib import Path


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    print(f"[exp] RUN: {' '.join(cmd)}", flush=True)
    # Stream output line-by-line while capturing for later parsing
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
                p = parts[1].strip()
                return Path(p)
    return None


def infer_tag_from_summary(summary: Path) -> str:
    # runs/<approach>/sweep_YYYYMMDD_HHMMSS/summary.csv → sweep_YYYYMMDD_HHMMSS
    return summary.parent.name


def write_index_row(summary: Path, approach: str, config_path: Path, plots_dir: Path, extras: dict):
    index_path = Path('runs') / 'index.csv'
    ensure_dir(index_path.parent)
    header = ['ts','approach','config','sweep_root','summary_csv','plots_dir','model','items','extras']
    exists = index_path.exists()
    if not exists:
        index_path.write_text(','.join(header) + '\n', encoding='utf-8')
    ts = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    sweep_root = summary.parent.as_posix()
    row = [
        ts,
        approach,
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


def cmd_run(args):
    cfg_path = Path(args.config)
    cfg = json.loads(Path(cfg_path).read_text(encoding='utf-8'))
    approach = cfg.get('approach', 'explicit')
    # Build sweep command
    cmd = [sys.executable, 'scripts/sweep.py', '--approach', approach]
    def add_list(flag: str, values):
        if values is None: return
        if isinstance(values, list):
            if flag in ['--hops','--Ls','--contexts','--m_list','--seeds']:
                cmd.extend([flag, ','.join(str(x) for x in values)])
        else:
            cmd.extend([flag, str(values)])
    # Common
    if 'items' in cfg: cmd.extend(['--items', str(cfg['items'])])
    add_list('--hops', cfg.get('hops'))
    if 'k' in cfg: cmd.extend(['--k', str(cfg['k'])])
    add_list('--Ls', cfg.get('Ls'))
    add_list('--contexts', cfg.get('contexts'))
    if 'M' in cfg: cmd.extend(['--M', str(cfg['M'])])
    add_list('--seeds', cfg.get('seeds'))
    if 'model' in cfg: cmd.extend(['--model', str(cfg['model'])])
    if 'temp' in cfg: cmd.extend(['--temp', str(cfg['temp'])])
    if 'max_output_tokens' in cfg: cmd.extend(['--max_output_tokens', str(cfg['max_output_tokens'])])
    # Extras
    if 'order_trials' in cfg: cmd.extend(['--order_trials', str(cfg['order_trials'])])
    if approach == 'implicit':
        add_list('--m_list', cfg.get('m_list'))
        if cfg.get('baseline'): cmd.extend(['--baseline', cfg['baseline']])
        if cfg.get('ablate_inner'): cmd.append('--ablate_inner')
        if cfg.get('ablate_hop') is not None: cmd.extend(['--ablate_hop', str(cfg['ablate_hop'])])
    else:
        if cfg.get('path_collision_stress'): cmd.append('--path_collision_stress')
    rc, out = run_cmd(cmd)
    if rc != 0:
        print('Sweep failed.', file=sys.stderr)
        sys.exit(rc)
    summary = parse_summary_path(out)
    if not summary or not summary.exists():
        print('Could not locate summary.csv in output.', file=sys.stderr)
        sys.exit(2)
    # Plot into run root plots/
    plots_dir = summary.parent / 'plots'
    ensure_dir(plots_dir)
    rc2, out2 = run_cmd([sys.executable, 'scripts/plot_sweep.py', '--summary', str(summary), '--approach', approach, '--outdir', str(plots_dir)])
    # Index
    write_index_row(summary, approach, cfg_path, plots_dir, extras={
        'model': cfg.get('model',''),
        'items': cfg.get('items',''),
        'config': cfg,
    })
    print(f'Run complete. Summary: {summary}\nPlots: {plots_dir}')


def summarize_for_report(summary_path: Path, approach: str) -> str:
    import csv, statistics
    rows = []
    with summary_path.open('r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if approach == 'implicit':
        # by (n,m)
        nm = {}
        for r in rows:
            if r.get('approach')!='implicit': continue
            n=int(r['n']); m=int(r['m']); acc=float(r['acc'])
            nm.setdefault((n,m), []).append(acc)
        lines = ['- Implicit EM (mean across seeds) by (n,m):']
        for (n,m) in sorted(nm):
            vals = nm[(n,m)]
            lines.append(f"  - n={n}, m={m}: {statistics.mean(vals):.3f}")
        return '\n'.join(lines)
    else:
        # by (n,L)
        nL = {}
        for r in rows:
            if r.get('approach')=='implicit': continue
            n=int(r['n']); L=int(r.get('L') or 0); acc=float(r['acc'])
            nL.setdefault((n,L), []).append(acc)
        lines = ['- Explicit EM (mean across seeds) by (n,L):']
        for (n,L) in sorted(nL):
            vals = nL[(n,L)]
            lines.append(f"  - n={n}, L={L}: {statistics.mean(vals):.3f}")
        return '\n'.join(lines)


def cmd_plot(args):
    outdir = Path(args.outdir)
    ensure_dir(outdir)
    rc, out = run_cmd([sys.executable, 'scripts/plot_sweep.py', '--summary', args.summary, '--approach', args.approach, '--outdir', str(outdir)])
    print(out, end='')


def cmd_report(args):
    summary = Path(args.summary)
    approach = args.approach
    tag = infer_tag_from_summary(summary)
    run_root = summary.parent
    plots_dir = run_root / 'plots'
    section = [
        f"### {tag} ({approach})",
        f"- Summary: `{summary.as_posix()}`",
        f"- Plots: `{plots_dir.as_posix()}`",
        summarize_for_report(summary, approach),
        "",
    ]
    (run_root / 'results.md').open('w', encoding='utf-8').write('\n'.join(section) + '\n')
    print(f"Wrote results.md for {tag} → {run_root/'results.md'}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    ap_run = sp.add_parser('run')
    ap_run.add_argument('--config', required=True)
    ap_run.set_defaults(func=cmd_run)

    ap_plot = sp.add_parser('plot')
    ap_plot.add_argument('--summary', required=True)
    ap_plot.add_argument('--approach', default='auto', choices=['explicit','implicit','auto'])
    ap_plot.add_argument('--outdir', required=True)
    ap_plot.set_defaults(func=cmd_plot)

    ap_report = sp.add_parser('report')
    ap_report.add_argument('--summary', required=True)
    ap_report.add_argument('--approach', required=True, choices=['explicit','implicit'])
    ap_report.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()



