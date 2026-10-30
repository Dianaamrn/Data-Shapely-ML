"""removal and detection helpers."""
import csv
import json
import numpy as np
from .knn import predict_knn
from .metrics import accuracy, macro_f1, ranking_order

FRACTIONS = (0., .1, .2, .3, .4)


def removal_count(n, fraction):
    if n < 1 or not 0 <= fraction < 1:
        raise ValueError('Require positive n and a fraction in [0,1).')
    return int(np.floor(n*fraction+.5))


def relative_accuracy(score, baseline):
    return float(score/baseline) if baseline else None


def removal_selections(values, priority, seed, condition_index, trials=30):
    """give the rows to remove for low, high and random at each fraction."""
    n = len(values)
    low = ranking_order(values, priority)
    high = ranking_order(values, priority, descending=True)
    for fi, fraction in enumerate(FRACTIONS[1:], 1):
        count = removal_count(n, fraction)
        yield 'LOW', fraction, None, low[:count], None
        yield 'HIGH', fraction, None, high[:count], None
        for trial in range(trials):
            key = (11, condition_index, fi, trial)
            rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=key))
            yield 'RANDOM', fraction, trial, rng.choice(n, count, replace=False), key


def detection(values, mask, priority):
    """how well low values find the flipped labels."""
    values, mask = np.asarray(values), np.asarray(mask, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError('Values and corruption mask must align.')
    if not mask.any() or mask.all():
        return None
    order = ranking_order(values, priority)
    positive, negative = values[mask], values[~mask]
    auc = np.mean((positive[:,None] < negative[None,:]) + .5*(positive[:,None] == negative[None,:]))
    return {'auroc':float(auc), 'precision10':float(mask[order[:removal_count(len(mask),.1)]].mean()),
            'precision20':float(mask[order[:removal_count(len(mask),.2)]].mean()),
            'fraction_inspected':(np.arange(len(mask)+1)/len(mask)).tolist(),
            'fraction_flips_found':(np.r_[0,np.cumsum(mask[order])]/mask.sum()).tolist()}


def evaluate_remaining(X, labels, test_X, test_y, removed, classes):
    """retrain 3-nn without the removed rows and score it on the test set."""
    keep = np.ones(len(labels), dtype=bool)
    keep[np.asarray(removed, dtype=int)] = False
    prediction = predict_knn(X[keep], labels[keep], test_X)
    result = {'test_accuracy':accuracy(test_y,prediction),
              'macro_f1':macro_f1(test_y,prediction,classes)}
    for label in classes:
        result['remaining_'+str(label)] = int(np.count_nonzero(labels[keep] == label))
        truth = test_y == label
        result['recall_'+str(label)] = float(np.mean(prediction[truth] == label)) if truth.any() else None
    return result


def save_json(path, data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def save_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
