import unittest
import numpy as np
from data_shapley.study_summary import paired_difference,bootstrap_mean_interval
from data_shapley.study import run_seed, convergence_rule, sample_until_converged
from data_shapley.shapley import SortedKNNGame

class StudyTest(unittest.TestCase):
    def test_integer_tie_not_floating_win(self):
        self.assertEqual(paired_difference(67/69,[67/69]*30,69),0.)
        self.assertEqual(paired_difference(68/69,[67/69]*30,69),1/69)
        self.assertEqual(paired_difference(67/69,[68/69]*30,69),-1/69)
        with self.assertRaises(ValueError): paired_difference(.123,[.123],69)

    def test_bootstrap_resamples_whole_runs(self):
        values=np.array([-1.,0.,1.,2.])
        expected_rng=np.random.default_rng(np.random.SeedSequence(100,spawn_key=(12,0,1)))
        expected=np.quantile([np.mean(values[expected_rng.integers(0,4,4)]) for _ in range(5000)],[.025,.975])
        got=bootstrap_mean_interval(values,np.random.SeedSequence(100,spawn_key=(12,0,1)))
        np.testing.assert_array_equal(got,expected)
        self.assertEqual(bootstrap_mean_interval([.1]*10,np.random.SeedSequence(10)),[.1,.1])

    def test_study_rejects_seeds_outside_100_to_109(self):
        for seed in [99,110]:
            with self.assertRaises(ValueError): run_seed(seed)

    def test_convergence_rule_hand_example_and_zero_estimates(self):
        # 0.1/1 and 0.5/2 averaged, the zero estimate is skipped
        self.assertAlmostEqual(convergence_rule([1., 2., 0.], [0.9, 2.5, 7.]), (0.1 + 0.25) / 2)
        self.assertEqual(convergence_rule([0., 0.], [1., 1.]), 0.)

    def test_stopping_keeps_the_same_first_permutations(self):
        X = np.array([[-2.], [-1.5], [-1.], [1.], [1.5], [2.], [0.2], [-0.2]])
        y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
        game = SortedKNNGame(X, y, np.array([[-1.8], [-0.5], [0.5], [1.8]]), np.array([0, 0, 1, 1]), y)
        marginals, orders, trace = sample_until_converged(game, np.random.default_rng(5), 200, 1000)
        reference = game.sample(len(orders), np.random.default_rng(5))
        np.testing.assert_array_equal(orders, reference.permutations)
        np.testing.assert_array_equal(marginals, reference.marginals)
        self.assertEqual([t for t, _ in trace], list(range(200, len(orders) + 1, 100)))
        self.assertTrue(all(rule >= .05 for _, rule in trace[:-1]))
        self.assertTrue(trace[-1][1] < .05 or len(orders) == 1000)
        with self.assertRaises(ValueError):
            sample_until_converged(game, np.random.default_rng(5), 150, 1000)
