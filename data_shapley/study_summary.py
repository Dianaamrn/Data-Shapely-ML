"""combine the ten seeds into the summary tables.

usage: python -m data_shapley.study_summary
"""
import csv
import json
from pathlib import Path

import numpy as np

from .experiments import save_csv, save_json
from .provenance import source_sha256
from .study import ROOT, SEEDS, checksums, seed_folder

NAMES = ('clean_penguins', 'noisy_penguins', 'noisy_gaussian')
FRACTIONS = (.1, .2, .3, .4)
BOOTSTRAP_DRAWS = 5000


def paired_difference(low, random_scores, test_size):
    """low minus the random mean, from whole correct-prediction counts."""
    scores = np.r_[low, random_scores]
    counts = np.rint(scores*test_size).astype(int)
    if not np.allclose(scores*test_size, counts, rtol=0, atol=1e-10):
        raise ValueError('Accuracy must correspond to integer test counts.')
    return float((counts[0]*len(random_scores)-counts[1:].sum())/(test_size*len(random_scores)))


def bootstrap_mean_interval(values, seed_sequence, draws=BOOTSTRAP_DRAWS):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Require at least two finite per-run values.')
    rng = np.random.default_rng(seed_sequence)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(1)
    return np.quantile(means, [.025, .975], method='linear').tolist()


def load_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    destination = ROOT/'results/study'
    current = checksums()
    all_raw, all_values, run_rows, detection_rows, permutation_rows, manifests = [], [], [], [], [], {}
    for seed in SEEDS:
        folder = seed_folder(seed)
        rec = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        assert rec['status'] == 'complete', f'seed {seed} is not complete'
        assert rec['sha256'] == current, f'seed {seed} was produced by different code or data; rerun it'
        manifests[seed] = rec
        raw = load_csv(folder/'removal.csv')
        assert len(raw) == 387
        all_raw.extend(raw)
        all_values.extend(load_csv(folder/'values.csv'))
        for name in NAMES:
            condition = rec['conditions'][name]
            permutation_rows.append({'condition': name, 'run_seed': seed, 'permutations': condition['M_used'],
                                     'final_rule_value': condition['convergence_trace'][-1][1],
                                     'rule_at_minimum': condition['convergence_trace'][0][1],
                                     'converged': condition['converged']})
            n = condition['n_test']
            baseline = condition['baseline']['test_accuracy']
            group = [r for r in raw if r['condition'] == name]
            assert len(group) == 129 and sum(r['strategy'] == 'BASELINE' for r in group) == 1
            for f in FRACTIONS:
                chosen = [r for r in group if float(r['fraction_requested']) == f]
                low = next(r for r in chosen if r['strategy'] == 'LOW')
                high = next(r for r in chosen if r['strategy'] == 'HIGH')
                random = [r for r in chosen if r['strategy'] == 'RANDOM']
                assert len(random) == 30
                scores = np.array([float(r['test_accuracy']) for r in random])
                l, h = float(low['test_accuracy']), float(high['test_accuracy'])
                run_rows.append({
                    'condition': name, 'run_seed': seed, 'fraction_requested': f, 'n_removed': int(low['n_removed']),
                    'fraction_actual': float(low['fraction_actual']), 'baseline': baseline, 'low': l, 'high': h,
                    'random_mean': float(scores.mean()), 'random_within_run_sd': float(scores.std(ddof=1)),
                    'low_minus_random': paired_difference(l, scores, n),
                    'low_gain_from_baseline': float((round(l*n)-round(baseline*n))/n),
                    'low_empirical_percentile': float(np.mean((scores < l)+.5*(scores == l))),
                    'low_relative': l/baseline if baseline else None,
                    'high_relative': h/baseline if baseline else None,
                    'random_relative': float(scores.mean()/baseline) if baseline else None,
                    'low_macro_f1': float(low['macro_f1']), 'high_macro_f1': float(high['macro_f1']),
                    'random_macro_f1': float(np.mean([float(x['macro_f1']) for x in random]))})
            d = condition['detection']
            if d:
                detection_rows.append({'condition': name, 'run_seed': seed,
                                       **{k: d[k] for k in ('auroc', 'precision10', 'precision20')}})

    aggregates = []
    for ci, name in enumerate(NAMES):
        for fi, f in enumerate(FRACTIONS, 1):
            rows = [r for r in run_rows if r['condition'] == name and r['fraction_requested'] == f]
            assert [r['run_seed'] for r in rows] == list(SEEDS)
            delta = np.array([r['low_minus_random'] for r in rows])
            interval = bootstrap_mean_interval(delta, np.random.SeedSequence(100, spawn_key=(12, ci, fi)))
            row = {'condition': name, 'fraction_requested': f, 'fraction_actual': rows[0]['fraction_actual'],
                   'n_runs': len(rows), 'wins': int((delta > 0).sum()), 'ties': int((delta == 0).sum()),
                   'losses': int((delta < 0).sum()), 'bootstrap95_low': interval[0], 'bootstrap95_high': interval[1]}
            for field in ('baseline', 'low', 'high', 'random_mean', 'low_minus_random', 'low_gain_from_baseline',
                          'low_relative', 'high_relative', 'random_relative', 'low_macro_f1', 'high_macro_f1',
                          'random_macro_f1'):
                values = [r[field] for r in rows if r[field] is not None]
                row[field+'_mean'] = float(np.mean(values)) if values else None
                row[field+'_sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
                if field.endswith('_relative'):
                    row[field+'_defined_runs'] = len(values)
            aggregates.append(row)

    detection_summary = []
    for name in NAMES[1:]:
        rows = [r for r in detection_rows if r['condition'] == name]
        row = {'condition': name, 'n_runs': len(rows)}
        for metric in ('auroc', 'precision10', 'precision20'):
            values = [r[metric] for r in rows]
            row[metric+'_mean'] = float(np.mean(values))
            row[metric+'_sd'] = float(np.std(values, ddof=1))
        detection_summary.append(row)

    # value distribution per seed and condition
    value_summary = []
    for seed in SEEDS:
        for name in NAMES:
            v = np.array([float(r['shapley_value']) for r in all_values
                          if int(r['run_seed']) == seed and r['condition'] == name])
            value_summary.append({'condition': name, 'run_seed': seed, 'n': len(v), 'mean': float(v.mean()),
                                  'median': float(np.median(v)), 'min': float(v.min()), 'max': float(v.max()),
                                  'sum': float(v.sum()), 'n_positive': int((v > 1e-12).sum()),
                                  'n_negative': int((v < -1e-12).sum()), 'n_zero': int((np.abs(v) <= 1e-12).sum())})

    # training labels left per class after each removal
    class_retention = []
    for name in NAMES:
        group = [r for r in all_raw if r['condition'] == name]
        labels = [k[len('remaining_'):] for k, v in group[0].items() if k.startswith('remaining_') and v != '']
        for strategy in ('LOW', 'HIGH', 'RANDOM'):
            for f in FRACTIONS:
                rows = [r for r in group if r['strategy'] == strategy and float(r['fraction_requested']) == f]
                row = {'condition': name, 'strategy': strategy, 'fraction_requested': f, 'n_outcomes': len(rows)}
                for label in labels:
                    counts = [int(r['remaining_'+label]) for r in rows]
                    row[label+'_min'], row[label+'_max'] = min(counts), max(counts)
                row['outcomes_with_absent_class'] = sum(any(int(r['remaining_'+lb]) == 0 for lb in labels) for r in rows)
                class_retention.append(row)

    assert len(all_raw) == 3870 and len(run_rows) == 120 and len(all_values) == 5600
    for filename, rows in [('removal', all_raw), ('values', all_values), ('per_run', run_rows),
                           ('summary', aggregates), ('detection_per_run', detection_rows),
                           ('detection_summary', detection_summary), ('value_summary', value_summary),
                           ('class_retention', class_retention), ('permutations', permutation_rows)]:
        save_csv(destination/f'study_{filename}.csv', rows)
    first = manifests[SEEDS[0]]
    used = [r['permutations'] for r in permutation_rows]
    save_json(destination/'study_manifest.json', {
        'status': 'complete', 'seeds': list(SEEDS), 'M_min': first['M_min'], 'M_max': first['M_max'],
        'convergence_rule': first['convergence_rule'], 'permutations_used_range': [min(used), max(used)],
        'all_runs_converged': all(r['converged'] for r in permutation_rows), 'random_trials': 30,
        'bootstrap': {'entropy': 100, 'spawn_key': '[12, condition_index, positive_fraction_index]',
                      'draws': BOOTSTRAP_DRAWS, 'unit': 'one complete run; RANDOM averaged within run first',
                      'quantile_method': 'linear', 'interval': 'descriptive percentile 95%'},
        'source_sha256': current,
        'aggregation_source_sha256': source_sha256(Path(__file__)),
        'per_seed_files_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): source_sha256(p)
                                  for seed in SEEDS for p in sorted(seed_folder(seed).glob('*.*'))
                                  if p.suffix in ('.json', '.csv')},
        'sampling_seconds_sum': sum(c['sampling_seconds'] for r in manifests.values() for c in r['conditions'].values()),
        'run_elapsed_seconds_sum': sum(r['elapsed_seconds'] for r in manifests.values()),
        'limitations': ['Penguins split results are correlated resamples of one dataset.',
                        'Bootstrap intervals are descriptive, not population guarantees or multiplicity-adjusted tests.',
                        'Meeting the convergence rule does not make clean-Penguins ranks stable; see appendix A.1.']})
    print('Aggregated 10 seeds: 30 valuations, 3,870 test outcomes; bootstrapped whole-run differences.')


if __name__ == '__main__':
    main()
