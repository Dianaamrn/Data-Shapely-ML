import unittest
from itertools import permutations
import numpy as np
from data_shapley.shapley import (make_knn_utility, exact_shapley_subsets,
    exact_shapley_permutations, evaluate_permutations_naive, estimate_shapley_naive, SortedKNNGame)


def six_point_game():
    # rows 0 and 1 are identical, row 5 has a conflicting label
    X = np.array([[-2.],[-2.],[-1.],[1.],[2.],[2.]])
    y = np.array([0,0,0,1,1,0])
    return make_knn_utility(X,y,np.array([[-2.],[-1.],[1.],[2.]]),np.array([0,0,1,1]),y)


class ShapleyTest(unittest.TestCase):
    def test_linearity_for_nonadditive_games(self):
        # games with interactions, so this really tests linearity

        def first(s):
            return 0.3 + int({0, 2}.issubset(s)) - 0.7 * int({1, 3, 5}.issubset(s))

        def second(s):
            return -0.2 + len(s) ** 2 / 36 + 0.4 * int({2, 4}.issubset(s))

        for method in (exact_shapley_subsets, exact_shapley_permutations):
            expected = 2 * method(6, first) - 3 * method(6, second)
            actual = method(6, lambda s: 2 * first(s) - 3 * second(s))
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    def test_six_point_knn_exact_agreement_efficiency_symmetry(self):
        utility = six_point_game()
        subset = exact_shapley_subsets(6,utility)
        perm = exact_shapley_permutations(6,utility)
        np.testing.assert_allclose(subset,perm,rtol=0,atol=1e-12)
        self.assertAlmostEqual(sum(subset),utility(range(6))-utility([]))
        self.assertAlmostEqual(subset[0],subset[1])
        self.assertTrue(np.any(subset < 0))
        # negative contributions must still add up to the total change
        for order in permutations(range(6)):
            prefix=[]; previous=utility([]); total=0
            for i in order:
                prefix.append(i); value=utility(prefix)
                total += value-previous; previous=value
            self.assertAlmostEqual(total,utility(range(6))-utility([]))

    def test_additive_null_negative_and_nonzero_baseline(self):
        weights=np.array([2.,2.,0.,-1.,0.5,3.])
        game=lambda s: 7.0+sum(weights[i] for i in s)
        for method in (exact_shapley_subsets,exact_shapley_permutations):
            np.testing.assert_allclose(method(6,game),weights,atol=1e-12)

    def test_unanimity_interaction_symmetry_and_null(self):
        game=lambda s: 0.4+int({0,1,2}.issubset(s))
        for method in (exact_shapley_subsets,exact_shapley_permutations):
            np.testing.assert_allclose(method(6,game),[1/3,1/3,1/3,0,0,0],atol=1e-12)

    def test_independent_enumeration_counts(self):
        for method, expected in ((exact_shapley_subsets,64),(exact_shapley_permutations,720*7)):
            calls=[]
            method(6,lambda s: calls.append(s) or len(s))
            self.assertEqual(len(calls),expected)
            self.assertEqual(len(set(calls)),64)

    def test_baseline_shift_is_uniform_for_exact_values(self):
        base=lambda s: len(s)/6 if s else 0.2
        shifted=lambda s: base(s) if s else 0.5
        for method in (exact_shapley_subsets,exact_shapley_permutations):
            np.testing.assert_allclose(method(6,shifted)-method(6,base),np.full(6,-0.3/6),atol=1e-12)

    def test_baseline_uses_clean_training_not_validation_or_noisy_majority(self):
        X=np.arange(3.)[:,None]; clean=np.array([0,0,1]); noisy=np.array([1,1,0])
        val=np.array([[0.],[1.],[2.]])
        for labels in (clean,noisy):
            utility=make_knn_utility(X,labels,val,[1,1,0],clean)
            self.assertEqual(utility([]),1/3)
        tie=make_knn_utility(X[:2],[1,0],val,[1,1,0],[1,0])
        self.assertEqual(tie([]),1/3)

    def test_game_copies_inputs_and_is_coalition_order_invariant(self):
        X=np.array([[0.],[1.],[2.]]); y=np.array([0,0,1]); val=X.copy(); truth=y.copy()
        utility=make_knn_utility(X,y,val,truth,y)
        before=[utility([]),utility([0]),utility([0,1,2])]
        X[:]=999; y[:]=1; val[:]=-999; truth[:]=1
        self.assertEqual(before,[utility([]),utility([0]),utility([2,1,0])])

    def test_invalid_games_and_coalitions(self):
        for method in (exact_shapley_subsets,exact_shapley_permutations):
            for n in (0,9,True,1.5):
                with self.assertRaises(ValueError): method(n,lambda s: 0)
            with self.assertRaises(ValueError): method(2,lambda s: np.nan)
            np.testing.assert_array_equal(method(1,lambda s: 5 if s else 2),[3])
        utility=six_point_game()
        for s in ([0,0],[-1],[6],[True],[1.5]):
            with self.assertRaises(ValueError): utility(s)
        for val,truth in (([[np.nan]],[0]),([],[]),([[1.]],[0,1])):
            with self.assertRaises(ValueError): make_knn_utility([[0.],[1.]],[0,1],val,truth,[0,1])


class MonteCarloTest(unittest.TestCase):
    def test_hand_calculated_order_maps_marginals_to_players(self):
        # by hand: -1, +2, +4 starting from an empty score of 2
        def game(s):
            return 2 + 2*(0 in s) - (2 in s) + 4*({0,1}.issubset(s))
        orders = np.array([[2,0,1], [1,2,0]])
        saved = orders.copy()
        result = evaluate_permutations_naive(3, game, orders)
        np.testing.assert_array_equal(result.prefix_utilities, [[2,1,3,7],[2,2,1,7]])
        np.testing.assert_array_equal(result.marginals, [[2,4,-1],[6,0,-1]])
        np.testing.assert_array_equal(result.values, [4,2,-1])
        np.testing.assert_array_equal(orders, saved)
        orders[:] = 0
        np.testing.assert_array_equal(result.permutations, saved)

    def test_all_toy_orders_agree_with_independent_exact_formula(self):
        game = six_point_game()
        result = evaluate_permutations_naive(6, game, list(permutations(range(6))))
        np.testing.assert_allclose(result.values, exact_shapley_subsets(6, game), atol=1e-12, rtol=0)
        np.testing.assert_allclose(result.marginals.sum(1), game(range(6))-game([]), atol=1e-12)

    def test_seed_replay_and_exact_generator_consumption(self):
        game = lambda s: len(s)**2
        generator = np.random.default_rng(2606)
        expected_rng = np.random.default_rng(2606)
        expected = np.array([expected_rng.permutation(6) for _ in range(19)])
        result = estimate_shapley_naive(6, game, 19, generator)
        np.testing.assert_array_equal(result.permutations, expected)
        self.assertEqual(generator.bit_generator.state, expected_rng.bit_generator.state)
        replay = estimate_shapley_naive(6, game, 19, np.random.default_rng(2606))
        for field in ('values','permutations','prefix_utilities','marginals'):
            np.testing.assert_array_equal(getattr(result,field),getattr(replay,field))
        other = estimate_shapley_naive(6, game, 19, np.random.default_rng(2607))
        self.assertFalse(np.array_equal(other.permutations,result.permutations))

    def test_efficiency_for_any_sample_budget_and_true_null_player(self):
        weights = np.array([2.,-1.,0.,3.,0.5,1.])
        game = lambda s: 7 + sum(weights[i] for i in s) + 2*({0,3}.issubset(s))
        for count in (1,2,7,31,100):
            result = estimate_shapley_naive(6, game, count, np.random.default_rng(20))
            np.testing.assert_allclose(result.marginals.sum(1), game(range(6))-game([]),atol=1e-12)
            self.assertAlmostEqual(result.values.sum(),game(range(6))-game([]))
            np.testing.assert_array_equal(result.marginals[:,2],0)

    def test_every_prefix_is_evaluated_without_truncation(self):
        calls=[]
        result=estimate_shapley_naive(4,lambda s: calls.append(s) or len(s),3,np.random.default_rng(8))
        self.assertEqual(len(calls),15)
        for draw,order in enumerate(result.permutations):
            self.assertEqual(calls[5*draw:5*(draw+1)],[frozenset(order[:k]) for k in range(5)])

    def test_single_player_and_repeated_orders(self):
        result=estimate_shapley_naive(1,lambda s: 5 if s else 2,7,np.random.default_rng(1))
        np.testing.assert_array_equal(result.values,[3])
        result=evaluate_permutations_naive(2,lambda s: len(s),[[1,0],[1,0]])
        np.testing.assert_array_equal(result.values,[1,1])

    def test_empty_only_shift_depends_on_first_position_counts(self):
        orders=np.array([[0,1,2],[0,2,1],[1,0,2]])
        base=lambda s: len(s)/3 if s else 0.2
        changed=lambda s: base(s) if s else 0.5
        first=evaluate_permutations_naive(3,base,orders)
        second=evaluate_permutations_naive(3,changed,orders)
        np.testing.assert_allclose(second.values-first.values,-0.3*np.array([2,1,0])/3,atol=1e-12)
        centered=evaluate_permutations_naive(3,lambda s: base(s)-0.2,orders)
        np.testing.assert_allclose(centered.marginals,first.marginals,atol=1e-12)

    def test_toy_approximation_across_fixed_seeds(self):
        # store the 64 team scores to keep the test fast
        # the estimator still scores every prefix itself
        game=six_point_game()
        table={frozenset(i for i in range(6) if mask & (1<<i)):0 for mask in range(64)}
        table={s:game(s) for s in table}
        exact=exact_shapley_subsets(6,table.__getitem__)
        errors=[]
        for seed in range(10):
            result=estimate_shapley_naive(6,table.__getitem__,2048,np.random.default_rng(6000+seed))
            errors.append([np.sqrt(np.mean((result.marginals[:m].mean(0)-exact)**2)) for m in (32,2048)])
        errors=np.array(errors)
        self.assertLess(errors[:,1].mean(),errors[:,0].mean())
        self.assertLess(errors[:,1].mean(),0.01)
        # checks this seed only, not a general convergence rate

    def test_invalid_inputs_fail_before_utility_calls(self):
        def unexpected(s): raise AssertionError('Invalid input reached utility')
        for n in (0,-1,True,1.5):
            with self.assertRaises(ValueError): estimate_shapley_naive(n,unexpected,1,np.random.default_rng(1))
        for count in (0,-1,False,1.5):
            with self.assertRaises(ValueError): estimate_shapley_naive(3,unexpected,count,np.random.default_rng(1))
        for generator in (None,42,np.random.RandomState(1)):
            with self.assertRaises(ValueError): estimate_shapley_naive(3,unexpected,1,generator)
        for orders in ([],[0,1,2],[[0,1]],[[0,0,2]],[[0,1,3]],[[-1,0,1]],[[0.,1.,2.]],[[True,False,True]]):
            with self.assertRaises(ValueError): evaluate_permutations_naive(3,unexpected,orders)
        for value in (np.nan,np.inf,-np.inf):
            with self.assertRaises(ValueError): estimate_shapley_naive(3,lambda s:value,1,np.random.default_rng(1))


class OptimizedTest(unittest.TestCase):
    def compare(self, X, y, query, truth, clean=None, orders=None):
        if clean is None: clean=y
        n=len(y)
        if orders is None:
            generator=np.random.default_rng(7007)
            orders=np.array([generator.permutation(n) for _ in range(12)])
        reference=make_knn_utility(X,y,query,truth,clean)
        naive=evaluate_permutations_naive(n,reference,orders)
        fast=SortedKNNGame(X,y,query,truth,clean).evaluate(orders)
        for field in ('permutations','prefix_utilities','marginals','values'):
            np.testing.assert_array_equal(getattr(naive,field),getattr(fast,field))
        return fast

    def test_all_six_point_permutations_exactly_equal(self):
        self.compare([[-2.],[-2.],[-1.],[1.],[2.],[2.]], [0,0,0,1,1,0],
                     [[-2.],[-1.],[1.],[2.]], [0,0,1,1],orders=list(permutations(range(6))))

    def test_large_boundary_group_and_rounding_sensitive_votes(self):
        self.compare([[0.5],[1.]]+[[2.]]*9,
                     ['Gentoo','Adelie']+['Chinstrap']*9,[[0.],[2.]],['Adelie','Chinstrap'])
        self.compare([[0.5,0],[0,1],[2,0],[-2,0],[0,-2],[3,2]],
                     [0,1,1,1,0,0],[[0,0],[1,0]],[0,1])

    def test_empty_singleton_two_points_and_absent_classes(self):
        self.compare([[1.]],[7],[[0.],[1.]],[7,7])
        self.compare([[-1.],[1.]],[1,0],[[0.],[1.]],[0,1])
        result=self.compare([[0.],[1.],[2.]],[1,1,1],[[0.],[1.],[2.]],[0,1,0],clean=[0,0,1])
        np.testing.assert_array_equal(result.prefix_utilities[:,0],2/3)

    def test_integer_ties_random_geometries_and_unicode_labels(self):
        rng=np.random.default_rng(7008)
        for case in range(30):
            X=rng.integers(-2,3,size=(8,3)).astype(float)
            query=rng.integers(-2,3,size=(5,3)).astype(float)
            y=np.array(['Adelie','Gentoo','Chinstrap','Adelie','Gentoo','Chinstrap','Adelie','Gentoo'])
            truth=y[:5]
            if case%2: X+=rng.normal(scale=0.01,size=X.shape)
            self.compare(X,y,query,truth)

    def test_extreme_distances_and_nearly_equal_distances(self):
        self.compare([[1e200,0.],[0.,1e200],[-1e200,0.],[0.,0.]], [0,1,1,0],[[0.,0.]],[0])
        self.compare([[1.],[np.nextafter(1.,2.)],[-1.],[0.]], [0,1,1,0],[[0.]],[0])

    def test_same_seed_same_orders_and_all_arrays_exact(self):
        X=np.arange(8.)[:,None]; y=np.array([0,1]*4)
        game=SortedKNNGame(X,y,X,y,y)
        fast=game.sample(17,np.random.default_rng(79))
        naive=estimate_shapley_naive(8,make_knn_utility(X,y,X,y,y),17,np.random.default_rng(79))
        for field in ('permutations','prefix_utilities','marginals','values'):
            np.testing.assert_array_equal(getattr(fast,field),getattr(naive,field))

    def test_original_input_mutation_cannot_change_game(self):
        X=np.arange(4.)[:,None]; y=np.array([0,0,1,1]); query=X.copy(); truth=y.copy()
        game=SortedKNNGame(X,y,query,truth,y)
        before=game.evaluate([[0,2,1,3]])
        X[:]=100; query[:]=-100; y[:]=1; truth[:]=1
        np.testing.assert_array_equal(game.evaluate([[0,2,1,3]]).prefix_utilities,before.prefix_utilities)

    def test_invalid_inputs_and_orders(self):
        game=SortedKNNGame([[0.],[1.]],[0,1],[[0.]],[0],[0,1])
        for orders in ([],[[0,0]],[[0.,1.]],[[0,2]]):
            with self.assertRaises(ValueError): game.evaluate(orders)
        for count in (0,-1,True,1.5):
            with self.assertRaises(ValueError): game.sample(count,np.random.default_rng(1))
        with self.assertRaises(ValueError): game.sample(2,None)
        with self.assertRaises(ValueError): SortedKNNGame([[0.],[1.]],[0,1],[[np.nan]],[0],[0,1])


if __name__ == '__main__': unittest.main()
