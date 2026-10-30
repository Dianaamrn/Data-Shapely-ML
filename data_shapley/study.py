"""main study: shapley values, removal and detection for each seed.

for each seed and condition: build the data, estimate shapley values on the
validation set (at least the pilot's minimum, then until the convergence rule
of ghorbani & zou (2019) is met), remove low, high and random points, retrain
and score on the test set, and check how well low values find the flipped labels.

usage: python -m data_shapley.study [seed ...]   (default 100-109)
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from .data import (make_data_rngs, load_penguins, split_penguins, standardize_split,
                   generate_gaussian_split, corrupt_training_labels, CLASS_LABELS)
from .experiments import (FRACTIONS, removal_selections, detection, evaluate_remaining,
                          relative_accuracy, save_json, save_csv)
from .metrics import ranking_order
from .provenance import source_sha256, raw_sha256
from .shapley import SortedKNNGame

ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(100, 110))
PILOT_RECORD = ROOT / 'results/pilot/pilot_seed100.json'
CHECKSUMMED = ('data/raw/penguins.csv', 'data_shapley/data.py', 'data_shapley/knn.py',
               'data_shapley/metrics.py', 'data_shapley/shapley.py', 'data_shapley/experiments.py',
               'data_shapley/provenance.py', 'data_shapley/study.py')
RANDOM_TRIALS = 30
# stop when the average relative change over 100 permutations is below 0.05
CONVERGENCE_LAG, CONVERGENCE_THRESHOLD, MAX_PERMUTATIONS = 100, 0.05, 20000
PENGUIN_FLIPS, GAUSSIAN_FLIPS = 41, 30  # 20% of 205 and of 150


def seed_folder(seed):
    return ROOT / f'results/study/seed{seed}'


def pilot_budget():
    """the minimum number of permutations, read from the pilot."""
    record = json.loads(PILOT_RECORD.read_text(encoding='utf-8'))
    if record['status'] != 'complete':
        raise ValueError('Run the pilot (python -m data_shapley.pilot) before the study.')
    return int(record['M_selected'])


def convergence_rule(current, previous):
    """average relative change of the values over the last 100 permutations, skipping zeros."""
    current, previous = np.asarray(current, dtype=float), np.asarray(previous, dtype=float)
    keep = current != 0
    if not keep.any():
        return 0.0
    return float(np.mean(np.abs(current[keep] - previous[keep]) / np.abs(current[keep])))


def sample_until_converged(game, rng, minimum, maximum=MAX_PERMUTATIONS, label=''):
    """draw permutations in batches of 100 until the convergence rule is met.

    the rule is only checked after the minimum. it uses whole-number counts so zero values are exact.
    """
    if minimum % CONVERGENCE_LAG or maximum % CONVERGENCE_LAG or not CONVERGENCE_LAG <= minimum <= maximum:
        raise ValueError('Budgets must be multiples of 100 with minimum <= maximum.')
    q = len(game.truth)
    chunks, orders, trace = [], [], []
    total = np.zeros(game.n, dtype=np.int64)
    while True:
        batch = game.sample(CONVERGENCE_LAG, rng)
        # each order's contributions must add up to v(all) - v(empty)
        np.testing.assert_allclose(batch.marginals.sum(1), batch.prefix_utilities[:, -1] - batch.prefix_utilities[:, 0], rtol=0, atol=1e-12)
        chunks.append(batch.marginals)
        orders.append(batch.permutations)
        done = CONVERGENCE_LAG * len(chunks)
        counts = np.rint(batch.marginals * q).astype(np.int64)
        np.testing.assert_allclose(counts / q, batch.marginals, rtol=0, atol=1e-9)
        previous_total = total.copy()
        total += counts.sum(0)
        if label and done % 500 == 0:
            print(f'{label}: {done} permutations', flush=True)
        if done >= minimum:
            rule = convergence_rule(total / done, previous_total / (done - CONVERGENCE_LAG))
            trace.append([done, rule])
            if rule < CONVERGENCE_THRESHOLD or done >= maximum:
                return np.concatenate(chunks), np.concatenate(orders), trace


def checksums():
    return {p: source_sha256(ROOT/p) if p.endswith('.py') else raw_sha256(ROOT/p) for p in CHECKSUMMED}


def prepare_conditions(seed):
    """build the three conditions for one seed."""
    rngs, data_manifest = make_data_rngs(seed)
    penguins, ps = standardize_split(split_penguins(load_penguins(ROOT/'data/raw/penguins.csv'), rngs['penguins_split']))
    gaussian, gs = standardize_split(generate_gaussian_split(rngs['synthetic_train'], rngs['synthetic_validation'], rngs['synthetic_test']))
    pn = corrupt_training_labels(penguins.train, PENGUIN_FLIPS, CLASS_LABELS, rngs['penguins_noise'])
    gn = corrupt_training_labels(gaussian.train, GAUSSIAN_FLIPS, (0, 1), rngs['synthetic_noise'])
    conditions = [
        ('clean_penguins', penguins, ps, penguins.train.y, np.zeros(len(penguins.train.y), dtype=bool), CLASS_LABELS),
        ('noisy_penguins', penguins, ps, pn.training.y, pn.corruption_mask, CLASS_LABELS),
        ('noisy_gaussian', gaussian, gs, gn.training.y, gn.corruption_mask, (0, 1)),
    ]
    return conditions, data_manifest


def run_seed(seed):
    if seed not in SEEDS:
        raise ValueError('The study uses seeds 100-109.')
    destination = seed_folder(seed)
    output = destination / 'manifest.json'
    if output.exists():
        raise FileExistsError(f'{output} exists; delete results/study/seed{seed} to rerun this seed.')
    destination.mkdir(parents=True, exist_ok=True)
    M_min = pilot_budget()
    started = time.perf_counter()
    conditions, data_manifest = prepare_conditions(seed)
    manifest = {'status': 'running', 'run_seed': seed, 'M_min': M_min, 'M_max': MAX_PERMUTATIONS,
                'convergence_rule': {'lag': CONVERGENCE_LAG, 'threshold': CONVERGENCE_THRESHOLD,
                                     'statistic': 'mean over points with nonzero estimate of |phi(t) - phi(t-100)| / |phi(t)|',
                                     'source': 'Ghorbani and Zou (2019)'},
                'random_trials_per_positive_fraction': RANDOM_TRIALS,
                'data_manifest': data_manifest, 'python': platform.python_version(), 'numpy': np.__version__,
                'conditions': {},
                'checksum_policy': 'Python sources: SHA-256 after CRLF-to-LF normalization; raw data: byte-exact SHA-256',
                'sha256': checksums()}
    save_json(output, manifest)
    prepared = {}
    valuations, all_rows, summary = [], [], []

    # value every condition before any test score is computed
    for ci, (name, split, scaler, observed, mask, classes) in enumerate(conditions):
        rec = {'n_train': len(observed), 'n_validation': len(split.validation.y), 'n_test': len(split.test.y),
               'training_ids': split.train.observation_ids.tolist(),
               'validation_ids': split.validation.observation_ids.tolist(),
               'test_ids': split.test.observation_ids.tolist(), 'original_labels': split.train.y.tolist(),
               'observed_labels': observed.tolist(), 'corrupted_mask': mask.tolist(),
               'scaler_mean': scaler.mean.tolist(), 'scaler_scale': scaler.scale.tolist(),
               'valuation_stream': {'entropy': seed, 'spawn_key': [10, ci]},
               'ranking_stream': {'entropy': seed, 'spawn_key': [9, ci]}}
        # a random order for breaking exact ties
        priority = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(9, ci))).permutation(len(observed))
        rec['tie_priority_by_training_row'] = priority.tolist()
        game = SortedKNNGame(split.train.X, observed, split.validation.X, split.validation.y, split.train.y)
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(10, ci)))
        t = time.perf_counter()
        samples, orders, trace = sample_until_converged(game, rng, M_min, label=f'seed {seed} {name}')
        values = samples.mean(0)
        rec['M_used'] = len(samples)
        rec['converged'] = trace[-1][1] < CONVERGENCE_THRESHOLD
        rec['convergence_trace'] = trace
        rec['sampling_seconds'] = time.perf_counter() - t
        rec['valuation_rng_final_state'] = rng.bit_generator.state
        rec['detection'] = detection(values, mask, priority)
        rank = np.empty(len(values), dtype=int)
        rank[ranking_order(values, priority)] = np.arange(1, len(values) + 1)
        for i, value in enumerate(values):
            valuations.append({'condition': name, 'run_seed': seed, 'observation_id': int(split.train.observation_ids[i]),
                               'observed_training_label': observed[i].item(), 'original_label': split.train.y[i].item(),
                               'corrupted_flag': int(mask[i]), 'shapley_value': float(value), 'rank': int(rank[i]),
                               'tie_priority': int(priority[i])})
        np.savez_compressed(destination/f'{name}_valuation.npz', values=values, marginals=samples,
                            permutations=orders, observation_ids=split.train.observation_ids)
        manifest['conditions'][name] = rec
        prepared[name] = (values, priority)
        save_json(output, manifest)

    # remove points and score on the test set
    for ci, (name, split, scaler, observed, mask, classes) in enumerate(conditions):
        values, priority = prepared[name]
        rec = manifest['conditions'][name]
        M_used = rec['M_used']
        t = time.perf_counter()
        base = evaluate_remaining(split.train.X, observed, split.test.X, split.test.y, [], classes)
        rec['baseline'] = base
        rec['removals'] = []
        condition_rows = []

        def add(strategy, fraction, trial, removed, scores):
            row = {'condition': name, 'run_seed': seed, 'strategy': strategy, 'fraction_requested': fraction,
                   'n_removed': len(removed), 'fraction_actual': len(removed)/len(observed), 'random_trial_id': trial,
                   'monte_carlo_permutations': M_used, **scores,
                   'relative_accuracy': relative_accuracy(scores['test_accuracy'], base['test_accuracy'])}
            condition_rows.append(row)
            all_rows.append(row)

        add('BASELINE', 0., None, [], base)
        for strategy, fraction, trial, removed, key in removal_selections(values, priority, seed, ci, RANDOM_TRIALS):
            scores = evaluate_remaining(split.train.X, observed, split.test.X, split.test.y, removed, classes)
            add(strategy, fraction, trial, removed, scores)
            rec['removals'].append({'strategy': strategy, 'fraction': fraction, 'trial': trial,
                                    'removed_ids': split.train.observation_ids[removed].tolist(),
                                    'random_stream': {'entropy': seed, 'spawn_key': list(key)} if key else None})
        for fraction in FRACTIONS[1:]:
            rows = [r for r in condition_rows if r['fraction_requested'] == fraction]
            random = np.array([r['test_accuracy'] for r in rows if r['strategy'] == 'RANDOM'])
            low = next(r['test_accuracy'] for r in rows if r['strategy'] == 'LOW')
            high = next(r['test_accuracy'] for r in rows if r['strategy'] == 'HIGH')
            summary.append({'condition': name, 'run_seed': seed, 'fraction_requested': fraction,
                            'baseline': base['test_accuracy'], 'low': low, 'high': high,
                            'random_mean': float(random.mean()), 'random_sd': float(random.std(ddof=1)),
                            'random_min': float(random.min()), 'random_max': float(random.max()),
                            'low_minus_random': float(low - random.mean()),
                            'low_empirical_percentile': float(np.mean((random < low) + .5*(random == low)))})
        rec['removal_seconds'] = time.perf_counter() - t
        print(f'seed {seed} {name}: M = {M_used}, baseline {base["test_accuracy"]:.4f}, removal evaluation complete', flush=True)
        save_json(output, manifest)

    save_csv(destination/'values.csv', valuations)
    save_csv(destination/'removal.csv', all_rows)
    save_csv(destination/'summary.csv', summary)
    manifest['status'] = 'complete'
    manifest['elapsed_seconds'] = time.perf_counter() - started
    save_json(output, manifest)
    print(f'Seed {seed} complete.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('seeds', type=int, nargs='*', default=list(SEEDS))
    for s in parser.parse_args().seeds:
        run_seed(s)
