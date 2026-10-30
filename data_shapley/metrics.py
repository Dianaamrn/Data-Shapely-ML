"""accuracy, f1 and ranking helpers."""
import numpy as np


def _labels(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0 or values.dtype.kind not in "iuU":
        raise ValueError(f"{name} must be a nonempty 1D integer/string label array.")
    if values.dtype.kind == "U" and np.any(np.char.str_len(np.char.strip(values)) == 0):
        raise ValueError("Labels cannot be blank.")
    return values


def _paired(y_true, y_pred):
    truth = _labels(y_true, "y_true")
    predicted = _labels(y_pred, "y_pred")
    if truth.shape != predicted.shape or (truth.dtype.kind == "U") != (predicted.dtype.kind == "U"):
        raise ValueError("True and predicted labels must have matching shapes and label types.")
    return truth, predicted


def accuracy(y_true, y_pred):
    """share of correct predictions."""
    truth, predicted = _paired(y_true, y_pred)
    return float(np.count_nonzero(truth == predicted) / len(truth))


def macro_f1(y_true, y_pred, class_labels):
    """f1 averaged over all classes."""
    truth, predicted = _paired(y_true, y_pred)
    classes = _labels(class_labels, "class_labels")
    if len(np.unique(classes)) != len(classes):
        raise ValueError("Class vocabulary must be unique.")
    if not np.isin(truth, classes).all() or not np.isin(predicted, classes).all():
        raise ValueError("All labels must belong to the declared vocabulary.")
    scores = []
    for label in classes:
        tp = int(np.count_nonzero((truth == label) & (predicted == label)))
        fp = int(np.count_nonzero((truth != label) & (predicted == label)))
        fn = int(np.count_nonzero((truth == label) & (predicted != label)))
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return float(sum(scores) / len(scores))


def majority_class(clean_training_labels):
    """the most common label."""
    labels = _labels(clean_training_labels, "clean_training_labels")
    classes, counts = np.unique(labels, return_counts=True)
    return classes[np.argmax(counts)].item()


def average_ranks(values):
    """rank values, giving tied values their average rank."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Ranks require at least two finite values.")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def rank_agreement(left, right):
    """spearman and kendall correlation between two rankings."""
    a, b = average_ranks(left), average_ranks(right)
    if a.shape != b.shape:
        raise ValueError("Rank vectors must align.")
    a_centered, b_centered = a-a.mean(), b-b.mean()
    denominator = np.sqrt(np.sum(a_centered**2)*np.sum(b_centered**2))
    spearman = float(np.sum(a_centered*b_centered)/denominator) if denominator else None
    i, j = np.triu_indices(len(a), 1)
    da, db = np.sign(a[i]-a[j]), np.sign(b[i]-b[j])
    denominator = np.sqrt(float(np.count_nonzero(da))*float(np.count_nonzero(db)))
    kendall = float(np.sum(da*db)/denominator) if denominator else None
    return {"spearman": spearman, "kendall_tau_b": kendall}


def ranking_order(values, tie_priority, descending=False):
    """order rows by value, using a random priority for exact ties."""
    values = np.asarray(values, dtype=float)
    priority = np.asarray(tie_priority)
    average_ranks(values)
    if priority.shape != values.shape or not np.array_equal(np.sort(priority), np.arange(len(values))):
        raise ValueError("Tie priority must be a permutation of row positions.")
    return np.lexsort((priority, -values if descending else values))


def bottom_overlap(left, right, observation_ids, fraction, tie_priority=None):
    """how many of the lowest-value points two value sets share."""
    a, b, ids = np.asarray(left), np.asarray(right), np.asarray(observation_ids)
    average_ranks(a); average_ranks(b)
    if a.shape != b.shape or ids.shape != a.shape or len(np.unique(ids)) != len(ids):
        raise ValueError("Values and unique observation IDs must align.")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0,1].")
    count = int(np.floor(fraction*len(a)+0.5))
    if count < 1:
        raise ValueError("fraction selects no observations.")
    if tie_priority is None:
        tie_priority = np.random.default_rng(np.random.SeedSequence(100, spawn_key=(9, 0))).permutation(len(a))
    left_ids = ids[ranking_order(a, tie_priority)[:count]]
    right_ids = ids[ranking_order(b, tie_priority)[:count]]
    shared = len(set(left_ids.tolist()) & set(right_ids.tolist()))
    return {"count":count, "shared":shared, "overlap_fraction":shared/count,
            "jaccard":shared/(2*count-shared), "left_ids":left_ids.tolist(),
            "right_ids":right_ids.tolist()}
