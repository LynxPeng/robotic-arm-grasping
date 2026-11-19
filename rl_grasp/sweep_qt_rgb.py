"""Sequential, controlled RGB-QT-Opt learning rate screen on one GPU."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys


def summary(root, rates):
    evaluations = {}
    runs = []
    for label, lr in rates:
        directory = root / f'qt-rgb-lr-{label}'
        log_path = directory / 'metrics.jsonl'
        records = [json.loads(row) for row in log_path.read_text().splitlines()] if log_path.exists() else []
        points = [row for row in records if row['event'] == 'eval']
        completed = next((row for row in records if row['event'] == 'complete'), None)
        runs.append(dict(label=label, lr=lr, completed=completed, validation=points))
        if completed:
            evaluations[label] = {row['steps']: row for row in points}
    common_steps = sorted(set.intersection(*(set(r) for r in evaluations.values())) - {0}) if evaluations else []
    comparison_steps = common_steps[-3:]
    for entry in runs:
        if comparison_steps and entry['label'] in evaluations:
            points = [evaluations[entry['label']][step] for step in comparison_steps]
            entry['late_mean'] = statistics.mean(row['success_rate'] for row in points)
            entry['late_range'] = max(row['success_rate'] for row in points) - min(row['success_rate'] for row in points)
            entry['late_by_kind'] = {kind: statistics.mean(row[f'{kind}_rate'] for row in points)
                                     for kind in ('cylinder','cube','long_box')}
    ranking = sorted((r['label'] for r in runs if 'late_mean' in r),
                     key=lambda label: next((-r['late_mean'], r['late_range']) for r in runs if r['label']==label))
    report = dict(comparison_steps=comparison_steps, ranking=ranking, runs=runs)
    temp = root / 'summary.tmp'
    temp.write_text(json.dumps(report, indent=2), encoding='utf-8')
    temp.replace(root/'summary.json')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--warm-start', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rates = [('2p5e-5', '0.000025'), ('5e-5', '0.00005'), ('1e-4', '0.0001')]
    base = [sys.executable, '-u', str(Path(__file__).with_name('train_qt_rgb.py')),
            '--warm-start', str(args.warm_start), '--workers', '16', '--batch', '80',
            '--demos', '14400', '--warmup', '0', '--steps', '80000',
            '--eval-every', '16000', '--eval-per-kind', '20', '--updates-per-tick', '4',
            '--online-replay', '14000', '--seed', '73', '--eval-seed', '300001',
            '--time-budget', '600', '--reward-log-every', '160', '--group', 'lr-screen-s73']
    failed = []
    for label, lr in rates:
        run_name = f'qt-rgb-lr-{label}'
        print(f'Starting {run_name}', flush=True)
        with (args.out/f'{run_name}.log').open('wb') as log:
            result = subprocess.run(base + ['--lr', lr, '--out', str(args.out/run_name)],
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            failed.append(run_name)
        report = summary(args.out, rates)
        print(f'Finished {run_name} exit={result.returncode} partial_order={report["ranking"]}', flush=True)
    if failed:
        raise SystemExit(f'Failed runs: {failed}')


if __name__ == '__main__':
    main()
