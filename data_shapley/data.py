"""data loading, splitting, scaling, synthetic data and label noise."""

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FEATURE_NAMES = (
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
)
CLASS_LABELS = ("Adelie", "Chinstrap", "Gentoo")
EXPECTED_CLASS_COUNTS = {"Adelie": 151, "Chinstrap": 68, "Gentoo": 123}
MISSING_MEASUREMENTS = {"", "NA"}

# add new streams at the end only, reordering changes every run
DATA_STREAM_NAMES = (
    "penguins_split",
    "penguins_noise",
    "synthetic_train",
    "synthetic_validation",
    "synthetic_test",
    "synthetic_noise",
)
PENGUINS_SPLIT_COUNTS = {
    "Adelie": (90, 30, 31),
    "Chinstrap": (41, 14, 13),
    "Gentoo": (74, 24, 25),
}


@dataclass(frozen=True)
class PenguinData:
    """the cleaned penguins data with the original csv row ids."""

    X: np.ndarray
    y: np.ndarray
    observation_ids: np.ndarray
    dropped_observation_ids: np.ndarray
    n_raw: int


def load_penguins(csv_path: str | Path) -> PenguinData:
    """read the penguins csv and drop rows with a missing measurement."""
    features = []
    labels = []
    retained_ids = []
    dropped_ids = []
    n_raw = 0
    required = {"species", *FEATURE_NAMES}

    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        header = reader.fieldnames or []
        missing = required.difference(header)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if len(header) != len(set(header)):
            raise ValueError("Duplicate CSV column names are not allowed.")

        for observation_id, row in enumerate(reader):
            n_raw += 1
            if None in row or any(row[name] is None for name in required):
                raise ValueError(f"Malformed CSV row at observation_id={observation_id}")
            label = row["species"].strip()
            if label not in CLASS_LABELS:
                raise ValueError(
                    f"Invalid species {label!r} at observation_id={observation_id}"
                )
            values = [row[name].strip() for name in FEATURE_NAMES]
            has_missing = any(value in MISSING_MEASUREMENTS for value in values)
            numeric = []
            for name, value in zip(FEATURE_NAMES, values):
                if value in MISSING_MEASUREMENTS:
                    continue
                try:
                    number = float(value)
                except ValueError as error:
                    raise ValueError(
                        f"Invalid {name}={value!r} at observation_id={observation_id}"
                    ) from error
                if not np.isfinite(number):
                    raise ValueError(
                        f"Non-finite {name} at observation_id={observation_id}"
                    )
                numeric.append(number)
            if has_missing:
                dropped_ids.append(observation_id)
                continue
            features.append(numeric)
            labels.append(label)
            retained_ids.append(observation_id)

    if n_raw != 344:
        raise ValueError(f"Expected 344 raw rows; found {n_raw}.")
    if len(features) != 342 or len(dropped_ids) != 2:
        raise ValueError(
            f"Expected 342 retained / 2 dropped rows; found "
            f"{len(features)} retained / {len(dropped_ids)} dropped."
        )
    counts = dict(Counter(labels))
    if counts != EXPECTED_CLASS_COUNTS:
        raise ValueError(f"Unexpected cleaned class counts: {counts}")

    return PenguinData(
        X=np.asarray(features, dtype=np.float64),
        y=np.asarray(labels, dtype=str),
        observation_ids=np.asarray(retained_ids, dtype=np.int64),
        dropped_observation_ids=np.asarray(dropped_ids, dtype=np.int64),
        n_raw=n_raw,
    )


@dataclass(frozen=True)
class DataPartition:
    """features, labels and ids of one partition."""

    X: np.ndarray
    y: np.ndarray
    observation_ids: np.ndarray


@dataclass(frozen=True)
class DataSplit:
    """train, validation and test partitions."""

    train: DataPartition
    validation: DataPartition
    test: DataPartition


@dataclass(frozen=True)
class Standardizer:
    """scaling learned from the training data."""

    mean: np.ndarray
    std: np.ndarray
    scale: np.ndarray
    zero_variance_mask: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        """scale new data with the training mean and std."""
        if np.iscomplexobj(X):
            raise ValueError("Features must be real-valued.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.mean.size:
            raise ValueError("Feature dimensions do not match the fitted standardizer.")
        if not np.isfinite(X).all():
            raise ValueError("Features must be finite.")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                return (X - self.mean) / self.scale
            except FloatingPointError as exc:
                raise ValueError("Standardized features exceed floating-point range.") from exc


@dataclass(frozen=True)
class LabelNoise:
    """noisy training labels, the clean labels and which rows were flipped."""

    training: DataPartition
    original_labels: np.ndarray
    corruption_mask: np.ndarray


def make_data_rngs(run_seed: int) -> tuple[dict, dict]:
    """make the random generators for one seed's data."""
    if isinstance(run_seed, bool) or not isinstance(run_seed, (int, np.integer)) or run_seed < 0:
        raise ValueError("run_seed must be a nonnegative integer.")
    root = np.random.SeedSequence(int(run_seed), spawn_key=(0,))
    children = root.spawn(len(DATA_STREAM_NAMES))
    rngs = {}
    manifest = {"run_seed": int(run_seed), "numpy_version": np.__version__, "streams": {}}
    for name, child in zip(DATA_STREAM_NAMES, children):
        rngs[name] = np.random.Generator(np.random.PCG64(child))
        manifest["streams"][name] = {
            "entropy": child.entropy,
            "spawn_key": list(child.spawn_key),
            "pool_size": child.pool_size,
            "bit_generator": "PCG64",
        }
    return rngs, manifest


def split_penguins(data: PenguinData, rng: np.random.Generator) -> DataSplit:
    """split penguins into train, validation and test with fixed counts per species."""
    if data.X.shape != (342, 4) or data.y.shape != (342,) or data.observation_ids.shape != (342,):
        raise ValueError("Expected aligned canonical Penguins arrays.")
    if len(np.unique(data.observation_ids)) != 342 or not np.isfinite(data.X).all():
        raise ValueError("Penguins needs unique IDs and finite features.")
    if dict(Counter(data.y)) != EXPECTED_CLASS_COUNTS:
        raise ValueError("Unexpected Penguins species counts.")

    selected = [[], [], []]
    for species in CLASS_LABELS:
        positions = rng.permutation(np.flatnonzero(data.y == species))
        n_train, n_validation, _ = PENGUINS_SPLIT_COUNTS[species]
        selected[0].extend(positions[:n_train])
        selected[1].extend(positions[n_train:n_train + n_validation])
        selected[2].extend(positions[n_train + n_validation:])

    partitions = []
    for positions in selected:
        positions = rng.permutation(np.asarray(positions, dtype=np.int64))
        partitions.append(DataPartition(
            X=data.X[positions].copy(),
            y=data.y[positions].copy(),
            observation_ids=data.observation_ids[positions].copy(),
        ))
    return DataSplit(*partitions)


def fit_standardizer(X_train: np.ndarray) -> Standardizer:
    """learn the mean and std from the training data only."""
    if np.iscomplexobj(X_train):
        raise ValueError("Training features must be real-valued.")
    X = np.asarray(X_train, dtype=np.float64)
    if X.ndim != 2 or 0 in X.shape or not np.isfinite(X).all():
        raise ValueError("Training features must be a nonempty finite 2D array.")
    constant = np.all(X == X[0], axis=0)
    with np.errstate(over="raise", invalid="raise"):
        try:
            mean = X.mean(axis=0)
            std = X.std(axis=0, ddof=0)
        except FloatingPointError as exc:
            raise ValueError("Training statistics exceed floating-point range.") from exc
    # a constant column gets std 0 exactly
    mean[constant] = X[0, constant]
    std[constant] = 0.0
    zero = std == 0
    return Standardizer(mean=mean, std=std, scale=np.where(zero, 1.0, std), zero_variance_mask=zero)


def standardize_split(split: DataSplit) -> tuple[DataSplit, Standardizer]:
    """scale all three partitions with the training statistics."""
    scaler = fit_standardizer(split.train.X)
    partitions = []
    for partition in (split.train, split.validation, split.test):
        partitions.append(DataPartition(
            X=scaler.transform(partition.X),
            y=partition.y.copy(),
            observation_ids=partition.observation_ids.copy(),
        ))
    return DataSplit(*partitions), scaler


def generate_gaussian_split(
    train_rng: np.random.Generator,
    validation_rng: np.random.Generator,
    test_rng: np.random.Generator,
) -> DataSplit:
    """make the two-class gaussian data: 150 train, 90 validation, 1000 test."""
    partitions = []
    next_id = 0
    for n, rng in ((150, train_rng), (90, validation_rng), (1000, test_rng)):
        y = np.repeat(np.array([0, 1], dtype=np.int64), n // 2)
        means = np.column_stack((np.where(y == 0, -1.0, 1.0), np.zeros(n)))
        X = rng.normal(size=(n, 2)) + means
        ids = np.arange(next_id, next_id + n, dtype=np.int64)
        order = rng.permutation(n)
        partitions.append(DataPartition(X=X[order], y=y[order], observation_ids=ids[order]))
        next_id += n
    return DataSplit(*partitions)


def corrupt_training_labels(
    training: DataPartition,
    n_flips: int,
    class_labels: tuple,
    rng: np.random.Generator,
) -> LabelNoise:
    """flip n training labels to a different class."""
    n = len(training.y)
    if training.y.ndim != 1 or training.X.ndim != 2 or training.X.shape[0] != n or training.observation_ids.shape != (n,):
        raise ValueError("Training features, labels, and IDs must be aligned.")
    if len(set(class_labels)) != len(class_labels) or len(class_labels) < 2:
        raise ValueError("At least two distinct class labels are required.")
    if not np.isin(training.y, class_labels).all():
        raise ValueError("Training labels must belong to class_labels.")
    if isinstance(n_flips, bool) or not isinstance(n_flips, (int, np.integer)) or not 0 <= n_flips <= n:
        raise ValueError("n_flips must be an integer between 0 and training size.")
    # make sure the label array can hold any class name
    label_dtype = np.result_type(training.y.dtype, np.asarray(class_labels).dtype)
    labels = training.y.astype(label_dtype, copy=True)
    mask = np.zeros(n, dtype=bool)
    positions = rng.choice(n, size=n_flips, replace=False)
    for position in positions:
        alternatives = [label for label in class_labels if label != training.y[position]]
        labels[position] = rng.choice(alternatives)
    mask[positions] = True
    noisy = DataPartition(X=training.X.copy(), y=labels, observation_ids=training.observation_ids.copy())
    return LabelNoise(training=noisy, original_labels=training.y.copy(), corruption_mask=mask)
