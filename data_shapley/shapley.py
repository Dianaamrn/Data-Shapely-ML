"""shapley value estimators for the k-nn game."""
from dataclasses import dataclass
from itertools import combinations, permutations
from math import factorial, isfinite

import numpy as np

from .knn import predict_knn, _validate_inputs
from .metrics import accuracy, majority_class


def make_knn_utility(X_train, y_train, X_validation, y_validation, clean_training_labels):
    """build the game: a team's score is 3-nn validation accuracy."""
    X, y = np.array(X_train, copy=True), np.array(y_train, copy=True)
    queries, truth = np.array(X_validation, copy=True), np.array(y_validation, copy=True)
    clean = np.asarray(clean_training_labels)
    if clean.shape != y.shape:
        raise ValueError("Original clean labels must align with training rows.")
    baseline_class = majority_class(clean)
    # check the inputs once before building the game
    accuracy(truth, predict_knn(X, y, queries, k=3))
    baseline_predictions = np.full(len(truth), baseline_class, dtype=clean.dtype)
    baseline = accuracy(truth, baseline_predictions)
    if len(queries) != len(truth) or not np.isin(y, np.unique(clean)).all() or not np.isin(truth, np.unique(clean)).all():
        raise ValueError("Validation rows and labels must align and labels must use the clean vocabulary.")

    def utility(coalition):
        members = list(coalition)
        if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) or not 0 <= i < len(y) for i in members):
            raise ValueError("Coalition members must be valid integer row positions.")
        if len(set(members)) != len(members):
            raise ValueError("Coalitions cannot contain repeated players.")
        if not members:
            return baseline
        rows = sorted(members)
        return accuracy(truth, predict_knn(X[rows], y[rows], queries, k=3))

    return utility


def _check_size(n):
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not 1 <= n <= 8:
        raise ValueError("Exact checks require an integer player count from 1 to 8.")
    return int(n)


def _value(utility, coalition):
    value = float(utility(frozenset(coalition)))
    if not isfinite(value):
        raise ValueError("Utility must return finite scalar values.")
    return value


def exact_shapley_subsets(n, utility):
    """exact shapley values from the subset formula, for small games only."""
    n = _check_size(n)
    players = tuple(range(n))
    values = {frozenset(s): _value(utility, s)
              for size in range(n + 1) for s in combinations(players, size)}
    result = np.zeros(n)
    for i in players:
        others = tuple(j for j in players if j != i)
        for size in range(n):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for subset in combinations(others, size):
                s = frozenset(subset)
                result[i] += weight * (values[s | {i}] - values[s])
    return result


def exact_shapley_permutations(n, utility):
    """exact shapley values by averaging over every order, for small games only."""
    n = _check_size(n)
    result = np.zeros(n)
    for order in permutations(range(n)):
        prefix = []
        previous = _value(utility, prefix)
        for player in order:
            prefix.append(player)
            current = _value(utility, prefix)
            result[player] += current - previous
            previous = current
    return result / factorial(n)


@dataclass(frozen=True)
class PermutationEstimate:
    """the values plus every order and contribution used to get them."""
    values: np.ndarray
    permutations: np.ndarray
    prefix_utilities: np.ndarray
    marginals: np.ndarray


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _checked_orders(n, orders):
    n = _positive_integer(n, "n")
    orders = np.asarray(orders)
    if (orders.ndim != 2 or orders.shape[0] < 1 or orders.shape[1] != n
            or orders.dtype.kind not in "iu"):
        raise ValueError("orders must be a nonempty integer array with shape (M,n).")
    if not np.all(np.sort(orders, axis=1) == np.arange(n)):
        raise ValueError("Each order must contain every player 0..n-1 exactly once.")
    orders = orders.astype(np.intp, copy=True)
    return orders


def evaluate_permutations_naive(n, utility, orders):
    """score every prefix of the given orders, the slow reference way."""
    orders = _checked_orders(n, orders)
    count = len(orders)
    prefixes = np.empty((count, n + 1), dtype=float)
    marginals = np.empty((count, n), dtype=float)
    for draw, order in enumerate(orders):
        selected = []
        previous = _value(utility, selected)
        prefixes[draw, 0] = previous
        for position, player in enumerate(order):
            selected.append(int(player))
            current = _value(utility, selected)
            prefixes[draw, position + 1] = current
            marginals[draw, player] = current - previous
            previous = current
    if not np.isfinite(marginals).all():
        raise ValueError("Marginal differences must be finite.")
    return PermutationEstimate(marginals.mean(axis=0), orders, prefixes, marginals)


def estimate_shapley_naive(n, utility, M, rng):
    """monte carlo shapley with m random orders, the slow reference way."""
    n = _positive_integer(n, "n")
    M = _positive_integer(M, "M")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit numpy.random.Generator.")
    orders = np.array([rng.permutation(n) for _ in range(M)])
    return evaluate_permutations_naive(n, utility, orders)


class SortedKNNGame:
    """the same game, made faster by sorting distances once."""

    def __init__(self, X_train, y_train, X_validation, y_validation, clean_training_labels):
        reference = make_knn_utility(X_train, y_train, X_validation,
                                     y_validation, clean_training_labels)
        self.baseline = reference([])
        X, y, queries, _ = _validate_inputs(X_train, y_train, X_validation, 3)
        self.n = len(y)
        self.truth = np.array(y_validation, copy=True)
        self.classes, labels = np.unique(y, return_inverse=True)
        self.labels = labels.copy()
        # same distance formula as knn.py so ties come out the same

        with np.errstate(over="ignore", invalid="ignore"):
            self.distances = np.array([np.hypot.reduce(X - query, axis=1) for query in queries])
        if not np.isfinite(self.distances).all():
            raise ValueError("A Euclidean distance is not finite.")
        self.order = np.argsort(self.distances, axis=1, kind="stable")
        self.sorted_distances = np.take_along_axis(self.distances, self.order, axis=1)
        self.class_masks = [self.labels[self.order] == c for c in range(len(self.classes))]

    def evaluate(self, orders):
        """score every prefix of the given orders."""
        orders = _checked_orders(self.n, orders)
        rows = np.arange(len(self.truth))
        prefixes = np.empty((len(orders), self.n + 1))
        marginals = np.empty((len(orders), self.n))
        for draw, permutation in enumerate(orders):
            active = np.zeros(self.n, dtype=bool)
            nearest = np.full((len(rows), len(self.classes)), np.inf)
            prefixes[draw, 0] = self.baseline
            previous = self.baseline
            for position, player in enumerate(permutation):
                active[player] = True
                label = self.labels[player]
                nearest[:, label] = np.minimum(nearest[:, label], self.distances[:, player])
                included = active[self.order]
                k = min(3, position + 1)
                boundary_position = np.argmax(np.cumsum(included, axis=1) >= k, axis=1)
                boundary = self.sorted_distances[rows, boundary_position, None]
                closer = included & (self.sorted_distances < boundary)
                tied = included & (self.sorted_distances == boundary)
                b, m = closer.sum(axis=1), tied.sum(axis=1)
                # whole-number weights keep vote ties exact
                weights = closer * m[:, None] + tied * (k - b)[:, None]
                votes = np.column_stack([(weights * mask).sum(axis=1) for mask in self.class_masks])
                winners = votes == votes.max(axis=1, keepdims=True)
                winner_distances = np.where(winners, nearest, np.inf)
                # remaining ties go to the first class in sorted order
                predictions = self.classes[np.argmin(winner_distances, axis=1)]
                current = accuracy(self.truth, predictions)
                prefixes[draw, position + 1] = current
                marginals[draw, player] = current - previous
                previous = current
        return PermutationEstimate(marginals.mean(axis=0), orders, prefixes, marginals)

    def sample(self, M, rng):
        """draw m random orders and score them."""
        M = _positive_integer(M, "M")
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be an explicit numpy.random.Generator.")
        return self.evaluate(np.array([rng.permutation(self.n) for _ in range(M)]))
