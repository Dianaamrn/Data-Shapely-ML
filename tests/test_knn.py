"""tests for knn.py, using small hand-checked cases."""

from itertools import permutations
import unittest

import numpy as np

from data_shapley.knn import explain_knn_query, predict_knn


class KNNTest(unittest.TestCase):
    def test_single_observation_predicts_its_class(self):
        actual = predict_knn([[2.0, -1.0]], np.array([1]), [[0.0, 0.0], [100.0, -100.0]])
        np.testing.assert_array_equal(actual, [1, 1])
        details = explain_knn_query([[2.0, -1.0]], np.array([1]), [0.0, 0.0])
        np.testing.assert_array_equal(details.weights, [1.0])

    def test_two_observations_use_both_and_nearest_class_wins(self):
        details = explain_knn_query([[1.0], [2.0]], np.array([1, 0]), [0.0])
        np.testing.assert_array_equal(details.weights, [1.0, 1.0])
        np.testing.assert_array_equal(details.votes, [1.0, 1.0])
        self.assertEqual(details.prediction, 1)

    def test_two_equidistant_classes_use_sorted_fallback(self):
        actual = predict_knn([[-1.0], [1.0]], np.array(["Gentoo", "Adelie"]), [[0.0]])
        np.testing.assert_array_equal(actual, ["Adelie"])

    def test_clear_three_neighbor_majority(self):
        details = explain_knn_query([[1.0], [2.0], [3.0], [10.0]], np.array([1, 0, 0, 1]), [0.0])
        self.assertEqual(details.prediction, 0)
        np.testing.assert_array_equal(details.weights, [1.0, 1.0, 1.0, 0.0])
        np.testing.assert_array_equal(details.votes, [2.0, 1.0])

    def test_euclidean_distances_match_hand_calculation(self):
        details = explain_knn_query([[3.0, 4.0], [0.0, -2.0], [0.0, 0.0], [-6.0, 8.0]], np.array([0, 1, 0, 1]), [0.0, 0.0])
        np.testing.assert_array_equal(details.distances, [5.0, 2.0, 0.0, 10.0])
        np.testing.assert_array_equal(details.weights, [1.0, 1.0, 1.0, 0.0])
        self.assertEqual(details.boundary_distance, 5.0)

    def test_fractional_boundary_weights_and_vote_totals(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [5.0, 0.0]])
        details = explain_knn_query(X, np.array([0, 1, 1, 1, 0, 0]), [0.0, 0.0])
        np.testing.assert_allclose(details.weights, [1, 1, 1 / 3, 1 / 3, 1 / 3, 0])
        np.testing.assert_array_equal(details.vote_numerators, [4, 5])
        self.assertEqual(details.denominator, 3)
        np.testing.assert_allclose(details.votes, [4 / 3, 5 / 3])
        self.assertAlmostEqual(details.weights.sum(), 3)
        self.assertEqual(details.prediction, 1)

    def test_all_equidistant_observations_share_weight(self):
        X = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        details = explain_knn_query(X, np.array([1, 0, 1, 0]), [0.0, 0.0])
        np.testing.assert_array_equal(details.weights, [0.75] * 4)
        np.testing.assert_array_equal(details.votes, [1.5, 1.5])
        self.assertEqual(details.prediction, 0)

    def test_zero_distance_duplicates_use_fractional_votes(self):
        details = explain_knn_query(np.zeros((5, 2)), np.array([1, 1, 1, 0, 0]), [0.0, 0.0])
        np.testing.assert_array_equal(details.weights, [0.6] * 5)
        np.testing.assert_allclose(details.votes, [1.2, 1.8])
        self.assertEqual(details.prediction, 1)

    def test_three_way_vote_tie_uses_nearest_member(self):
        details = explain_knn_query([[3.0], [1.0], [2.0]], np.array(["Adelie", "Gentoo", "Chinstrap"]), [0.0])
        np.testing.assert_array_equal(details.votes, [1.0, 1.0, 1.0])
        self.assertEqual(details.prediction, "Gentoo")

    def test_exact_vote_tie_is_not_broken_by_roundoff(self):
        X = np.array([[0.5], [1.0]] + [[2.0]] * 9)
        y = np.array(["Gentoo", "Adelie"] + ["Chinstrap"] * 9)
        details = explain_knn_query(X, y, [0.0])
        # nine votes of 1/9 tie with one full vote
        np.testing.assert_array_equal(details.vote_numerators, [9, 9, 9])
        np.testing.assert_array_equal(details.votes, [1.0, 1.0, 1.0])
        self.assertEqual(details.denominator, 9)
        self.assertEqual(details.prediction, "Gentoo")

    def test_distance_and_class_ties_use_fixed_species_order(self):
        details = explain_knn_query([[1.0], [-1.0], [1.0]], np.array(["Gentoo", "Chinstrap", "Adelie"]), [0.0])
        np.testing.assert_array_equal(details.classes, ["Adelie", "Chinstrap", "Gentoo"])
        self.assertEqual(details.prediction, "Adelie")
        # a class with no points cannot win
        actual = predict_knn([[1.0], [-1.0]], np.array(["Gentoo", "Chinstrap"]), [[0.0]])
        np.testing.assert_array_equal(actual, ["Chinstrap"])

    def test_near_distances_are_not_treated_as_exact_ties(self):
        X = np.array([[1.0], [1.0 + 1e-12], [1.0 + 2e-12], [1.0 + 3e-12]])
        details = explain_knn_query(X, np.array([1, 0, 0, 1]), [0.0])
        np.testing.assert_array_equal(details.weights, [1.0, 1.0, 1.0, 0.0])

    def test_reordering_rows_and_same_label_duplicates_preserves_result(self):
        X = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
        y = np.array([1, 1, 0, 0])
        reference = explain_knn_query(X, y, [0.0, 0.0])
        for order in permutations(range(4)):
            order = np.asarray(order)
            reordered = explain_knn_query(X[order], y[order], [0.0, 0.0])
            self.assertEqual(reordered.prediction, reference.prediction)
            np.testing.assert_array_equal(reordered.weights[np.argsort(order)], reference.weights)
            np.testing.assert_array_equal(reordered.vote_numerators, reference.vote_numerators)

    def test_prediction_batch_shape_dtype_and_explanation_agree(self):
        X = np.array([[0.0], [1.0], [2.0], [8.0], [9.0], [10.0]])
        y = np.array(["Adelie"] * 3 + ["Gentoo"] * 3)
        queries = np.array([[0.5], [9.5], [5.0]])
        actual = predict_knn(X, y, queries)
        self.assertEqual(actual.shape, (3,))
        self.assertEqual(actual.dtype, y.dtype)
        np.testing.assert_array_equal(actual[:2], ["Adelie", "Gentoo"])
        self.assertTrue(set(actual).issubset(set(y)))
        for row, predicted in zip(queries, actual):
            self.assertEqual(explain_knn_query(X, y, row).prediction, predicted)
        empty = predict_knn(X, y, np.empty((0, 1)))
        self.assertEqual(empty.shape, (0,))
        self.assertEqual(empty.dtype, y.dtype)

    def test_inputs_are_unchanged_and_repeated_predictions_match(self):
        X = np.array([[-2.0], [0.0], [1.0], [5.0]])
        y = np.array([0, 0, 1, 1])
        queries = np.array([[-1.0], [4.0]])
        originals = [array.copy() for array in (X, y, queries)]
        first = predict_knn(X, y, queries)
        np.testing.assert_array_equal(predict_knn(X, y, queries), first)
        for actual, original in zip((X, y, queries), originals):
            np.testing.assert_array_equal(actual, original)

    def test_uniform_scaling_and_translation_preserve_predictions(self):
        X = np.array([[-2.0, 0.0], [0.0, 1.0], [1.0, 0.0], [5.0, 1.0]])
        y = np.array([0, 0, 1, 1])
        queries = np.array([[-1.0, 0.0], [4.0, 1.0]])
        expected = predict_knn(X, y, queries)
        np.testing.assert_array_equal(predict_knn(2 * X + 10, y, 2 * queries + 10), expected)

    def test_invalid_shapes_empty_training_and_invalid_labels_are_rejected(self):
        cases = [
            (np.empty((0, 2)), np.array([], dtype=int), np.zeros((1, 2))),
            (np.ones(3), np.array([0, 1, 0]), np.ones((1, 1))),
            (np.empty((2, 0)), np.array([0, 1]), np.empty((1, 0))),
            (np.ones((2, 2)), np.array([0]), np.ones((1, 2))),
            (np.ones((2, 2)), np.array([[0], [1]]), np.ones((1, 2))),
            (np.ones((2, 2)), np.array([0, 1]), np.ones((1, 3))),
            (np.ones((2, 2)), np.array([0, 1]), np.ones(2)),
            (np.ones((2, 2)), np.array([0.0, np.nan]), np.ones((1, 2))),
            (np.ones((2, 2)), np.array(["Adelie", " "]), np.ones((1, 2))),
            (np.ones((2, 2)), np.array([0, "Adelie"], dtype=object), np.ones((1, 2))),
        ]
        for X, y, queries in cases:
            with self.subTest(X_shape=X.shape, y=y, query_shape=queries.shape):
                with self.assertRaises(ValueError):
                    predict_knn(X, y, queries)
        with self.assertRaisesRegex(ValueError, "1D feature vector"):
            explain_knn_query([[0.0]], np.array([0]), [[1.0]])

    def test_invalid_k_is_rejected(self):
        for k in (0, -1, 1.5, True, np.bool_(True), "3"):
            with self.subTest(k=k):
                with self.assertRaises(ValueError):
                    predict_knn([[0.0]], np.array([0]), [[1.0]], k=k)
        np.testing.assert_array_equal(predict_knn([[0.0]], np.array([0]), [[1.0]], k=np.int64(3)), [0])

    def test_nonfinite_complex_and_overflowing_distances_are_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    predict_knn([[value]], np.array([0]), [[0.0]])
                with self.assertRaises(ValueError):
                    predict_knn([[0.0]], np.array([0]), [[value]])
        with self.assertRaises(ValueError):
            predict_knn([[1.0 + 1.0j]], np.array([0]), [[0.0]])
        with self.assertRaisesRegex(ValueError, "distance is not finite"):
            predict_knn([[np.finfo(float).max]], np.array([0]), [[-np.finfo(float).max]])
        # large distances must not overflow
        details = explain_knn_query([[3e200, 4e200]], np.array([0]), [0.0, 0.0])
        np.testing.assert_allclose(details.distances, [5e200])


if __name__ == "__main__":
    unittest.main()
