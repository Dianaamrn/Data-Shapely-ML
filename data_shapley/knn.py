"""our own k-nearest-neighbour classifier."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KNNVote:
    """the details of one prediction: distances, weights and votes."""

    prediction: str | int
    distances: np.ndarray
    weights: np.ndarray
    classes: np.ndarray
    votes: np.ndarray
    vote_numerators: np.ndarray
    denominator: int
    boundary_distance: float


def _validate_inputs(X_train, y_train, X_queries, k):
    """check the inputs before predicting."""
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer.")
    if np.iscomplexobj(X_train) or np.iscomplexobj(X_queries):
        raise ValueError("Features must be real numbers.")
    try:
        train = np.asarray(X_train, dtype=np.float64)
        queries = np.asarray(X_queries, dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise ValueError("Features must be numeric arrays.") from error
    labels = np.asarray(y_train)
    if train.ndim != 2 or train.shape[1] == 0:
        raise ValueError("X_train must have shape (n_train, n_features), with features.")
    if train.shape[0] == 0:
        raise ValueError("k-NN requires at least one training observation; empty-set utility is separate.")
    if labels.ndim != 1 or labels.shape[0] != train.shape[0]:
        raise ValueError("y_train must be a 1D label array aligned with X_train.")
    if queries.ndim != 2 or queries.shape[1] != train.shape[1]:
        raise ValueError("X_queries must be 2D with the same feature count as X_train.")
    if not np.isfinite(train).all() or not np.isfinite(queries).all():
        raise ValueError("Training and query features must be finite.")
    if labels.dtype.kind not in "iuU":
        raise ValueError("Labels must be homogeneous integers or Unicode strings.")
    if labels.dtype.kind == "U" and np.any(np.char.str_len(np.char.strip(labels)) == 0):
        raise ValueError("String labels must not be empty or blank.")
    return train, labels, queries, int(k)


def _vote_for_query(train, labels, query, k):
    """predict one point, sharing the last vote between tied neighbours."""
    # hypot is a safe way to compute euclidean distance
    with np.errstate(over="ignore", invalid="ignore"):
        distances = np.hypot.reduce(train - query, axis=1)
    if not np.isfinite(distances).all():
        raise ValueError("A Euclidean distance is not finite; check feature magnitudes.")

    q = min(k, len(train))
    boundary = np.partition(distances, q - 1)[q - 1]
    closer = distances < boundary
    tied = distances == boundary
    b = int(np.count_nonzero(closer))
    m = int(np.count_nonzero(tied))
    remaining_votes = q - b

    # all weights share one denominator
    # whole numbers keep exact ties exact
    weight_numerators = np.zeros(len(train), dtype=np.int64)
    weight_numerators[closer] = m
    weight_numerators[tied] = remaining_votes
    classes, label_indices = np.unique(labels, return_inverse=True)
    vote_numerators = np.zeros(len(classes), dtype=np.int64)
    np.add.at(vote_numerators, label_indices, weight_numerators)

    winners = np.flatnonzero(vote_numerators == vote_numerators.max())
    nearest_by_class = np.full(len(classes), np.inf)
    np.minimum.at(nearest_by_class, label_indices, distances)
    closest_winner_distance = nearest_by_class[winners].min()
    winners = winners[nearest_by_class[winners] == closest_winner_distance]
    # if votes and distances tie, take the first label in sorted order

    winner = winners[0]
    return KNNVote(
        prediction=classes[winner].item(),
        distances=distances,
        weights=weight_numerators / m,
        classes=classes,
        votes=vote_numerators / m,
        vote_numerators=vote_numerators,
        denominator=m,
        boundary_distance=float(boundary),
    )


def predict_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_queries: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    """predict labels for a batch of points with k-nn."""
    train, labels, queries, k = _validate_inputs(X_train, y_train, X_queries, k)
    predictions = np.empty(len(queries), dtype=labels.dtype)
    for position, query in enumerate(queries):
        predictions[position] = _vote_for_query(train, labels, query, k).prediction
    return predictions


def explain_knn_query(
    X_train: np.ndarray,
    y_train: np.ndarray,
    query: np.ndarray,
    k: int = 3,
) -> KNNVote:
    """return the full vote for one point."""
    query = np.asarray(query)
    if query.ndim != 1:
        raise ValueError("query must be a 1D feature vector.")
    train, labels, queries, k = _validate_inputs(X_train, y_train, query[None, :], k)
    return _vote_for_query(train, labels, queries[0], k)
