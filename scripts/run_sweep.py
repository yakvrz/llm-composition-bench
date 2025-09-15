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
    header = ['ts','approach','config','sweep_root','results_csv','plots_dir','model','items','extras']
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
    if 'baseline' in cfg and cfg.get('baseline'): cmd.extend(['--baseline', str(cfg['baseline'])])
    if approach == 'implicit':
        add_list('--m_list', cfg.get('m_list'))
        if cfg.get('baseline'): cmd.extend(['--baseline', cfg['baseline']])
        if cfg.get('ablate_inner'): cmd.append('--ablate_inner')
        if cfg.get('ablate_hop') is not None: cmd.extend(['--ablate_hop', str(cfg['ablate_hop'])])
        if 'id_width' in cfg: cmd.extend(['--id_width', str(cfg['id_width'])])
    else:
        if cfg.get('path_collision_stress'): cmd.append('--path_collision_stress')
        if cfg.get('block_head_balance'): cmd.append('--block_head_balance')
        if cfg.get('alias_heads_per_block'): cmd.append('--alias_heads_per_block')
        if 'id_width' in cfg: cmd.extend(['--id_width', str(cfg['id_width'])])
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
    # Include label in plots when present in config
    label = cfg.get('label') or ''
    plot_cmd = [sys.executable, 'scripts/plot_sweep.py', '--summary', str(summary), '--approach', approach, '--outdir', str(plots_dir)]
    if label:
        plot_cmd += ['--label', str(label)]
    rc2, out2 = run_cmd(plot_cmd)
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
    plot_cmd = [sys.executable, 'scripts/plot_sweep.py', '--summary', args.summary, '--approach', args.approach, '--outdir', str(outdir)]
    if getattr(args, 'label', ''):
        plot_cmd += ['--label', args.label]
    rc, out = run_cmd(plot_cmd)
    print(out, end='')


def cmd_report(args):
    summary = Path(args.summary)
    approach = args.approach
    tag = infer_tag_from_summary(summary)
    run_root = summary.parent
    plots_dir = run_root / 'plots'
    # Load extras (config/label) from runs/index.csv for this summary if available
    def load_extras_for_summary(s: Path) -> dict | None:
        index_path = Path('runs') / 'index.csv'
        if not index_path.exists():
            return None
        import csv, json as _json
        with index_path.open('r', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                match_path = r.get('results_csv') or r.get('summary_csv')
                if match_path == s.as_posix():
                    extras_str = r.get('extras') or ''
                    try:
                        # undo "," replacement for CSV safety
                        extras_json = extras_str.replace(';', ',')
                        return _json.loads(extras_json)
                    except Exception:
                        return None
        return None

    extras = load_extras_for_summary(summary) or {}
    cfg = extras.get('config') or {}

    # Label line precedence: CLI arg > config label
    label_val = (getattr(args, 'label', '') or cfg.get('label') or '').strip()
    label_line = [f"- Label: `{label_val}`"] if label_val else []

    def kv(name: str, value):
        return f"  - {name}: {value}"

    params_lines: list[str] = ["- Parameters:"]
    if cfg:
        if approach == 'explicit':
            params_lines.extend([
                kv('approach', cfg.get('approach', 'explicit')),
                kv('items', cfg.get('items')),
                kv('hops', cfg.get('hops')),
                kv('Ls', cfg.get('Ls')),
                kv('k', cfg.get('k', '')),
                kv('contexts', cfg.get('contexts')),
                kv('seeds', cfg.get('seeds')),
                kv('M', cfg.get('M')),
                kv('id_width', cfg.get('id_width', 4)),
                kv('model', cfg.get('model')),
                kv('temp', cfg.get('temp')),
                kv('max_output_tokens', cfg.get('max_output_tokens')),
                kv('order_trials', cfg.get('order_trials', 1)),
                kv('path_collision_stress', cfg.get('path_collision_stress', False)),
                kv('block_head_balance', cfg.get('block_head_balance', False)),
                kv('alias_heads_per_block', cfg.get('alias_heads_per_block', False)),
                kv('baseline', cfg.get('baseline', '')),
            ])
        else:
            params_lines.extend([
                kv('approach', cfg.get('approach', 'implicit')),
                kv('items', cfg.get('items')),
                kv('hops', cfg.get('hops')),
                kv('m_list', cfg.get('m_list')),
                kv('seeds', cfg.get('seeds')),
                kv('M', cfg.get('M')),
                kv('id_width', cfg.get('id_width', 4)),
                kv('model', cfg.get('model')),
                kv('temp', cfg.get('temp')),
                kv('max_output_tokens', cfg.get('max_output_tokens')),
                kv('order_trials', cfg.get('order_trials', 1)),
                kv('baseline', cfg.get('baseline', '')),
                kv('ablate_inner', cfg.get('ablate_inner', False)),
                kv('ablate_hop', cfg.get('ablate_hop', '')),
            ])

    # results.md deprecated; ensure any existing is removed
    md_path = run_root / 'results.md'
    try:
        if md_path.exists():
            md_path.unlink()
    except Exception:
        pass

    # Additionally write a structured JSON report including lift-over-chance aggregates
    import csv as _csv, json as _json, statistics as _stats
    rows: list[dict] = []
    with summary.open('r', encoding='utf-8') as f:
        for r in _csv.DictReader(f):
            rows.append(r)

    report: dict = {
        'approach': approach,
        'tag': tag,
        'label': label_val or None,
        'results_csv': summary.as_posix(),
        'plots_dir': plots_dir.as_posix(),
        'params': cfg or None,
        'aggregates': {},
    }

    if approach == 'implicit':
        # group by (n,m)
        from collections import defaultdict
        by_nm: dict[tuple[int,int], list[float]] = defaultdict(list)
        for r in rows:
            if r.get('approach') != 'implicit':
                continue
            try:
                n = int(r['n']); m = int(r['m']); acc = float(r['acc'])
            except Exception:
                continue
            by_nm[(n,m)].append(acc)
        out = []
        for (n,m), vals in sorted(by_nm.items()):
            if not vals:
                continue
            acc_mean = _stats.mean(vals)
            chance = (0.0 if m <= 1 else 1.0/m)
            lift = (acc_mean - chance) / (1.0 - chance) if (1.0 - chance) > 0 else None
            out.append({'n': n, 'm': m, 'acc_mean': round(acc_mean, 3), 'chance': round(chance, 3), 'lift_mean': (None if lift is None else round(lift, 3))})
        report['aggregates']['by_nm'] = out
    else:
        # group by (n,L)
        from collections import defaultdict
        by_nL: dict[tuple[int,int], list[tuple[float,int|None,list[int]]]] = defaultdict(list)
        # collect acc and k_last per row; also seeds
        for r in rows:
            if r.get('approach') == 'implicit':
                continue
            try:
                n = int(r['n']); L = int(r.get('L') or 0); acc = float(r['acc'])
            except Exception:
                continue
            k_last: int | None = None
            try:
                if r.get('k'):
                    k_last = int(r['k'])
                else:
                    ks = [int(r[x]) for x in ('k0','k1','k2') if r.get(x)]
                    k_last = ks[-1] if ks else None
            except Exception:
                k_last = None
            seed = None
            try:
                seed = int(r.get('seed') or 0)
            except Exception:
                seed = None
            by_nL[(n,L)].append((acc, k_last, [seed] if seed is not None else []))
        out = []
        for (n,L), entries in sorted(by_nL.items()):
            if not entries:
                continue
            accs = [e[0] for e in entries]
            seeds: list[int] = []
            k_vals = [e[1] for e in entries if e[1] is not None]
            for e in entries:
                seeds.extend(e[2])
            acc_mean = _stats.mean(accs)
            k_last = (k_vals[0] if k_vals else None)
            chance = (None if (k_last is None or k_last <= 1) else 1.0/ k_last)
            lift = None
            if chance is not None and (1.0 - chance) > 0:
                lift = (acc_mean - chance) / (1.0 - chance)
            out.append({'n': n, 'L': L, 'acc_mean': round(acc_mean, 3), 'k_last': k_last, 'chance': (None if chance is None else round(chance,3)), 'lift_mean': (None if lift is None else round(lift,3)), 'seeds': sorted({s for s in seeds if s is not None})})
        report['aggregates']['by_nL'] = out

    (run_root / 'summary.json').write_text(_json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding='utf-8')
    print(f"Wrote summary.json for {tag} → {run_root/'summary.json'}")


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
    ap_plot.add_argument('--label', default='')
    ap_plot.set_defaults(func=cmd_plot)

    ap_report = sp.add_parser('report')
    ap_report.add_argument('--summary', required=True)
    ap_report.add_argument('--approach', required=True, choices=['explicit','implicit'])
    ap_report.add_argument('--label', default='')
    ap_report.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()



