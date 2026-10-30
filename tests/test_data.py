"""tests for data.py."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from data_shapley.data import (
    CLASS_LABELS,
    DATA_STREAM_NAMES,
    FEATURE_NAMES,
    DataPartition,
    DataSplit,
    corrupt_training_labels,
    fit_standardizer,
    generate_gaussian_split,
    load_penguins,
    make_data_rngs,
    split_penguins,
    standardize_split,
)


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "raw" / "penguins.csv"


class PenguinDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with CSV_PATH.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            cls.header = reader.fieldnames
            cls.rows = list(reader)

    def load_modified(self, rows, header=None):
        """load an edited copy of the csv."""
        with TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.csv"
            with path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=header or self.header)
                writer.writeheader()
                writer.writerows(rows)
            return load_penguins(path)

    def test_canonical_shapes_counts_and_feature_order(self):
        data = load_penguins(CSV_PATH)
        self.assertEqual(data.n_raw, 344)
        self.assertEqual(data.X.shape, (342, 4))
        self.assertEqual(data.y.shape, (342,))
        self.assertTrue(np.issubdtype(data.X.dtype, np.floating))
        self.assertTrue(np.isfinite(data.X).all())
        self.assertEqual(dict(Counter(data.y)), {"Adelie": 151, "Gentoo": 123, "Chinstrap": 68})
        self.assertEqual(FEATURE_NAMES, ("bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"))
        np.testing.assert_array_equal(data.X[0], [39.1, 18.7, 181.0, 3750.0])
        self.assertEqual(data.y[0], "Adelie")

    def test_only_rows_with_missing_selected_measurements_are_dropped(self):
        data = load_penguins(CSV_PATH)
        expected_dropped = [
            i for i, row in enumerate(self.rows)
            if any(row[name] in {"", "NA"} for name in FEATURE_NAMES)
        ]
        np.testing.assert_array_equal(data.dropped_observation_ids, expected_dropped)
        self.assertEqual(len(expected_dropped), 2)
        np.testing.assert_array_equal(
            data.observation_ids,
            [i for i in range(344) if i not in expected_dropped],
        )
        self.assertEqual(len(set(data.observation_ids)), 342)
        for position, observation_id in enumerate(data.observation_ids):
            row = self.rows[observation_id]
            np.testing.assert_array_equal(data.X[position], [float(row[name]) for name in FEATURE_NAMES])
            self.assertEqual(data.y[position], row["species"])

    def test_missing_sex_does_not_drop_complete_measurements(self):
        data = load_penguins(CSV_PATH)
        affected_ids = [
            i for i, row in enumerate(self.rows)
            if row["sex"] == "NA" and all(row[name] != "NA" for name in FEATURE_NAMES)
        ]
        self.assertTrue(affected_ids)
        self.assertTrue(set(affected_ids).issubset(data.observation_ids))

    def test_repeated_loading_is_identical_and_preserves_source_bytes(self):
        before = CSV_PATH.read_bytes()
        first, second = load_penguins(CSV_PATH), load_penguins(CSV_PATH)
        for name in ("X", "y", "observation_ids", "dropped_observation_ids"):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
        self.assertEqual(before, CSV_PATH.read_bytes())

    def test_source_checksum_matches_provenance(self):
        manifest = json.loads(CSV_PATH.with_suffix(".provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256(CSV_PATH.read_bytes()).hexdigest(), manifest["sha256"])
        self.assertEqual(CSV_PATH.stat().st_size, manifest["size_bytes"])

    def test_missing_column_is_rejected(self):
        header = [name for name in self.header if name != "bill_length_mm"]
        rows = [{name: row[name] for name in header} for row in self.rows]
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            self.load_modified(rows, header)

    def test_invalid_or_nonfinite_measurements_are_rejected(self):
        for value in ("not-a-number", "nan", "inf", "-inf"):
            with self.subTest(value=value):
                rows = [dict(row) for row in self.rows]
                rows[0]["bill_length_mm"] = value
                with self.assertRaises(ValueError):
                    self.load_modified(rows)

    def test_unknown_or_missing_species_is_rejected(self):
        for value in ("Unknown", "NA", ""):
            with self.subTest(value=value):
                rows = [dict(row) for row in self.rows]
                rows[0]["species"] = value
                with self.assertRaisesRegex(ValueError, "Invalid species"):
                    self.load_modified(rows)

    def test_blank_measurement_follows_na_policy(self):
        rows = [dict(row) for row in self.rows]
        for name in FEATURE_NAMES:
            rows[3][name] = ""
        data = self.load_modified(rows)
        self.assertIn(3, data.dropped_observation_ids)
        self.assertEqual(data.X.shape, (342, 4))

    def test_unexpected_source_or_cleaned_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Expected 344 raw rows"):
            self.load_modified(self.rows[:-1])
        rows = [dict(row) for row in self.rows]
        rows[0]["bill_length_mm"] = "NA"
        with self.assertRaisesRegex(ValueError, "Expected 342 retained"):
            self.load_modified(rows)
        rows = [dict(row) for row in self.rows]
        rows[0]["species"] = "Gentoo"
        with self.assertRaisesRegex(ValueError, "Unexpected cleaned class counts"):
            self.load_modified(rows)


class PreparationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.penguins = load_penguins(CSV_PATH)

    def test_recorded_child_streams_reconstruct_exactly(self):
        rngs, manifest = make_data_rngs(100)
        self.assertEqual(tuple(rngs), DATA_STREAM_NAMES)
        keys = []
        for name, state in manifest["streams"].items():
            child = np.random.SeedSequence(
                state["entropy"], spawn_key=tuple(state["spawn_key"]), pool_size=state["pool_size"]
            )
            replay = np.random.Generator(np.random.PCG64(child))
            np.testing.assert_array_equal(rngs[name].integers(0, 10000, 20), replay.integers(0, 10000, 20))
            keys.append(tuple(state["spawn_key"]))
        self.assertEqual(len(set(keys)), len(DATA_STREAM_NAMES))

    def test_penguins_stratification_and_source_alignment_for_all_study_seeds(self):
        expected = ({"Adelie": 90, "Chinstrap": 41, "Gentoo": 74},
                    {"Adelie": 30, "Chinstrap": 14, "Gentoo": 24},
                    {"Adelie": 31, "Chinstrap": 13, "Gentoo": 25})
        source_positions = {int(i): position for position, i in enumerate(self.penguins.observation_ids)}
        for seed in range(100, 110):
            with self.subTest(seed=seed):
                rngs, _ = make_data_rngs(seed)
                split = split_penguins(self.penguins, rngs["penguins_split"])
                all_ids = []
                for part, counts in zip((split.train, split.validation, split.test), expected):
                    self.assertEqual(dict(Counter(part.y)), counts)
                    positions = [source_positions[int(i)] for i in part.observation_ids]
                    np.testing.assert_array_equal(part.X, self.penguins.X[positions])
                    np.testing.assert_array_equal(part.y, self.penguins.y[positions])
                    all_ids.extend(part.observation_ids)
                self.assertEqual(len(all_ids), 342)
                self.assertEqual(set(all_ids), set(self.penguins.observation_ids))
                self.assertEqual(len(set(all_ids)), 342)

    def test_split_repeatability_and_noise_stream_isolation(self):
        rngs1, state1 = make_data_rngs(100)
        rngs2, state2 = make_data_rngs(100)
        rngs2["penguins_noise"].random(1000)
        first = split_penguins(self.penguins, rngs1["penguins_split"])
        second = split_penguins(self.penguins, rngs2["penguins_split"])
        self.assertEqual(state1, state2)
        for name in ("train", "validation", "test"):
            for field in ("X", "y", "observation_ids"):
                np.testing.assert_array_equal(getattr(getattr(first, name), field), getattr(getattr(second, name), field))
        other_rngs, _ = make_data_rngs(101)
        other = split_penguins(self.penguins, other_rngs["penguins_split"])
        self.assertFalse(np.array_equal(first.train.observation_ids, other.train.observation_ids))

    def test_hand_computed_standardization_and_zero_variance(self):
        X = np.array([[1.0, 10.0], [3.0, 10.0]])
        original = X.copy()
        scaler = fit_standardizer(X)
        np.testing.assert_array_equal(scaler.mean, [2.0, 10.0])
        np.testing.assert_array_equal(scaler.std, [1.0, 0.0])
        np.testing.assert_array_equal(scaler.scale, [1.0, 1.0])
        np.testing.assert_array_equal(scaler.zero_variance_mask, [False, True])
        np.testing.assert_array_equal(scaler.transform(X), [[-1.0, 0.0], [1.0, 0.0]])
        np.testing.assert_array_equal(scaler.transform([[9.0, 12.0]]), [[7.0, 2.0]])
        np.testing.assert_array_equal(X, original)

    def test_holdouts_never_influence_scaling(self):
        train = DataPartition(np.array([[1.0], [3.0]]), np.array([0, 1]), np.array([0, 1]))
        val = DataPartition(np.array([[101.0]]), np.array([0]), np.array([2]))
        test = DataPartition(np.array([[-99.0]]), np.array([1]), np.array([3]))
        scaled, scaler = standardize_split(DataSplit(train, val, test))
        np.testing.assert_array_equal(scaler.mean, [2.0])
        np.testing.assert_array_equal(scaler.std, [1.0])
        np.testing.assert_array_equal(scaled.validation.X, [[99.0]])
        np.testing.assert_array_equal(scaled.test.X, [[-101.0]])
        for before, after in zip((train, val, test), (scaled.train, scaled.validation, scaled.test)):
            np.testing.assert_array_equal(before.y, after.y)
            np.testing.assert_array_equal(before.observation_ids, after.observation_ids)
        np.testing.assert_array_equal(train.X, [[1.0], [3.0]])

    def test_real_training_scaling_and_fixed_transform(self):
        rngs, _ = make_data_rngs(100)
        split = split_penguins(self.penguins, rngs["penguins_split"])
        scaled, scaler = standardize_split(split)
        np.testing.assert_allclose(scaled.train.X.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(scaled.train.X.std(axis=0, ddof=0), 1.0, atol=1e-12)
        for source, transformed in zip((split.train, split.validation, split.test), (scaled.train, scaled.validation, scaled.test)):
            np.testing.assert_allclose(transformed.X, (source.X - split.train.X.mean(axis=0)) / split.train.X.std(axis=0))
        retained = scaled.train.X[10:]
        np.testing.assert_allclose(scaler.transform(split.train.X[10:]), retained)

    def test_gaussian_shapes_balance_and_unique_ids_across_study_seeds(self):
        for seed in range(100, 110):
            with self.subTest(seed=seed):
                rngs, _ = make_data_rngs(seed)
                split = generate_gaussian_split(rngs["synthetic_train"], rngs["synthetic_validation"], rngs["synthetic_test"])
                all_ids = []
                for part, n in zip((split.train, split.validation, split.test), (150, 90, 1000)):
                    self.assertEqual(part.X.shape, (n, 2))
                    self.assertTrue(np.isfinite(part.X).all())
                    self.assertEqual(dict(Counter(part.y)), {0: n // 2, 1: n // 2})
                    all_ids.extend(part.observation_ids)
                self.assertEqual(len(all_ids), 1240)
                self.assertEqual(set(all_ids), set(range(1240)))

    def test_synthetic_repeatability_and_partition_stream_independence(self):
        a, _ = make_data_rngs(100)
        b, _ = make_data_rngs(100)
        first = generate_gaussian_split(a["synthetic_train"], a["synthetic_validation"], a["synthetic_test"])
        same = generate_gaussian_split(b["synthetic_train"], b["synthetic_validation"], b["synthetic_test"])
        for name in ("train", "validation", "test"):
            for field in ("X", "y", "observation_ids"):
                np.testing.assert_array_equal(getattr(getattr(first, name), field), getattr(getattr(same, name), field))
        c, _ = make_data_rngs(100)
        c["synthetic_train"].random(10)
        changed = generate_gaussian_split(c["synthetic_train"], c["synthetic_validation"], c["synthetic_test"])
        self.assertFalse(np.array_equal(first.train.X, changed.train.X))
        np.testing.assert_array_equal(first.validation.X, changed.validation.X)
        np.testing.assert_array_equal(first.test.X, changed.test.X)

    def test_penguins_noise_is_exact_and_holdouts_and_inputs_stay_clean(self):
        rngs, _ = make_data_rngs(100)
        split, _ = standardize_split(split_penguins(self.penguins, rngs["penguins_split"]))
        originals = [part.y.copy() for part in (split.train, split.validation, split.test)]
        noisy = corrupt_training_labels(split.train, 41, CLASS_LABELS, rngs["penguins_noise"])
        self.assertEqual(int(noisy.corruption_mask.sum()), 41)
        np.testing.assert_array_equal(noisy.corruption_mask, noisy.training.y != split.train.y)
        np.testing.assert_array_equal(noisy.original_labels, originals[0])
        np.testing.assert_array_equal(noisy.training.X, split.train.X)
        np.testing.assert_array_equal(noisy.training.observation_ids, split.train.observation_ids)
        self.assertTrue(np.isin(noisy.training.y, CLASS_LABELS).all())
        for part, original in zip((split.train, split.validation, split.test), originals):
            np.testing.assert_array_equal(part.y, original)

    def test_binary_noise_exactly_swaps_30_labels_and_is_reproducible(self):
        a, _ = make_data_rngs(100)
        split = generate_gaussian_split(a["synthetic_train"], a["synthetic_validation"], a["synthetic_test"])
        noise1 = corrupt_training_labels(split.train, 30, (0, 1), a["synthetic_noise"])
        b, _ = make_data_rngs(100)
        noise2 = corrupt_training_labels(split.train, 30, (0, 1), b["synthetic_noise"])
        self.assertEqual(int(noise1.corruption_mask.sum()), 30)
        np.testing.assert_array_equal(noise1.training.y[noise1.corruption_mask], 1 - split.train.y[noise1.corruption_mask])
        np.testing.assert_array_equal(noise1.training.y[~noise1.corruption_mask], split.train.y[~noise1.corruption_mask])
        np.testing.assert_array_equal(noise1.training.y, noise2.training.y)
        np.testing.assert_array_equal(noise1.training.X, split.train.X)
        other = corrupt_training_labels(split.train, 30, (0, 1), np.random.default_rng(999))
        self.assertFalse(np.array_equal(noise1.training.y, other.training.y))
        np.testing.assert_array_equal(other.training.X, split.train.X)

    def test_constant_decimal_feature_scales_to_zero(self):
        X = np.full((3, 1), 0.1)
        scaler = fit_standardizer(X)
        np.testing.assert_array_equal(scaler.std, [0.0])
        np.testing.assert_array_equal(scaler.scale, [1.0])
        np.testing.assert_array_equal(scaler.transform(X), np.zeros_like(X))

    def test_noise_preserves_longer_or_wider_class_labels(self):
        for original, classes in ((np.array(["Adelie"]), ("Adelie", "Chinstrap")),
                                  (np.array([0], dtype=np.int8), (0, 1000))):
            part = DataPartition(np.ones((1, 1)), original, np.array([0]))
            noise = corrupt_training_labels(part, 1, classes, np.random.default_rng(1))
            self.assertEqual(noise.training.y[0], classes[1])
            np.testing.assert_array_equal(noise.original_labels, original)

    def test_scaling_rejects_complex_and_overflow(self):
        with self.assertRaises(ValueError):
            fit_standardizer(np.array([[1 + 2j]]))
        with self.assertRaises(ValueError):
            fit_standardizer(np.array([[-1e308], [1e308]]))
        scaler = fit_standardizer(np.array([[1.0], [2.0]]))
        with self.assertRaises(ValueError):
            scaler.transform(np.array([[1 + 2j]]))
        with self.assertRaises(ValueError):
            scaler.transform(np.array([[1e308]]))

    def test_preparation_rejects_invalid_inputs(self):
        for X in (np.array([]), np.empty((0, 2)), np.array([[np.nan]]), np.array([[np.inf]])):
            with self.subTest(X=X):
                with self.assertRaises(ValueError):
                    fit_standardizer(X)
        scaler = fit_standardizer(np.array([[1.0], [2.0]]))
        with self.assertRaises(ValueError):
            scaler.transform(np.ones((2, 2)))
        for seed in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                make_data_rngs(seed)
        part = DataPartition(np.ones((2, 1)), np.array([0, 1]), np.array([0, 1]))
        for n in (-1, 3, 1.5, True):
            with self.assertRaises(ValueError):
                corrupt_training_labels(part, n, (0, 1), np.random.default_rng(1))
        with self.assertRaises(ValueError):
            corrupt_training_labels(part, 1, (0,), np.random.default_rng(1))


if __name__ == "__main__":
    unittest.main()
