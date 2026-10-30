import unittest
import numpy as np
from data_shapley.metrics import average_ranks, rank_agreement, bottom_overlap


class StabilityTest(unittest.TestCase):
    def test_hand_average_ranks_and_undefined_constant(self):
        np.testing.assert_array_equal(average_ranks([3,1,1,4]),[2,0.5,0.5,3])
        self.assertEqual(rank_agreement([1,1,1],[1,2,3]),{'spearman':None,'kendall_tau_b':None})

    def test_rank_correlations_hand_examples(self):
        for sign, b in [(1,[1,2,3]),(-1,[3,2,1])]:
            result=rank_agreement([1,2,3],b)
            self.assertEqual(result['spearman'],sign)
            self.assertEqual(result['kendall_tau_b'],sign)
        result=rank_agreement([1,1,2],[1,2,3])
        self.assertAlmostEqual(result['spearman'],np.sqrt(3)/2)
        self.assertAlmostEqual(result['kendall_tau_b'],2/np.sqrt(6))
        self.assertEqual(rank_agreement([1,1,2],[3,3,4]),{'spearman':1.0,'kendall_tau_b':1.0})

    def test_kendall_independent_pair_loop(self):
        a=np.array([1,1,3,2,4,2]); b=np.array([2,1,1,3,4,3])
        concordant=discordant=non_tied_a=non_tied_b=0
        for i in range(len(a)):
            for j in range(i+1,len(a)):
                da=int(a[i])-int(a[j]); db=int(b[i])-int(b[j])
                non_tied_a += da!=0; non_tied_b += db!=0
                concordant += da*db>0; discordant += da*db<0
        expected=(concordant-discordant)/np.sqrt(non_tied_a*non_tied_b)
        self.assertEqual(rank_agreement(a,b)['kendall_tau_b'],expected)

    def test_overlap_stable_ids_half_up_and_jaccard(self):
        result=bottom_overlap([0,0,2,3,4],[2,0,0,3,4],[20,10,30,40,50],0.3, [1,0,2,3,4])
        self.assertEqual(result['count'],2)
        self.assertEqual(result['left_ids'],[10,20])
        self.assertEqual(result['shared'],1)
        self.assertEqual(result['overlap_fraction'],0.5)
        self.assertEqual(result['jaccard'],1/3)
        self.assertEqual(bottom_overlap(np.arange(205),np.arange(205),np.arange(205),.1)['count'],21)

    def test_invalid_vectors_and_ids(self):
        for values in ([1],[1,np.nan],[[1,2]]):
            with self.assertRaises(ValueError): average_ranks(values)
        with self.assertRaises(ValueError): rank_agreement([1,2],[1,2,3])
        with self.assertRaises(ValueError): bottom_overlap([1,2],[1,2],[4,4],.5)
        with self.assertRaises(ValueError): bottom_overlap([1,2],[1,2],[4,5],0)


class PilotDiagnosticTest(unittest.TestCase):
    def test_constant_samples_have_undefined_correlation_and_zero_se(self):
        from data_shapley.pilot import diagnostics
        d=diagnostics(np.zeros((4,20)),np.zeros((4,20)),np.arange(20))
        self.assertIsNone(d['spearman'])
        self.assertIsNone(d['kendall_tau_b'])
        self.assertEqual(d['se90_each_stream'],[0.,0.])
        self.assertFalse(d['descriptive_benchmarks_met'])
        self.assertEqual(d['bottom']['0.2']['overlap_fraction'],1)
        self.assertIsNone(d['boundary_each_stream'][0]['0.2']['gap_over_se'])

    def test_value_difference_and_paired_boundary_se(self):
        from data_shapley.pilot import diagnostics
        a=np.arange(20.)[None,:]+np.array([0.,1.,-1.,2.])[:,None]
        b=a+0.5
        d=diagnostics(a,b,np.arange(20))
        self.assertEqual(d['mean_absolute_value_difference'],.5)
        self.assertEqual(d['boundary_each_stream'][0]['0.2']['gap'],1)
        self.assertEqual(d['boundary_each_stream'][0]['0.2']['paired_gap_se'],0)
        self.assertAlmostEqual(d['se90_each_stream'][0],np.sqrt(5/3)/2)


if __name__ == '__main__': unittest.main()

class RandomTieTest(unittest.TestCase):
    def test_low_high_and_ids(self):
        from data_shapley.metrics import ranking_order
        priority = np.array([2,0,3,1])
        np.testing.assert_array_equal(ranking_order([1,1,2,2], priority), [1,0,3,2])
        np.testing.assert_array_equal(ranking_order([1,1,2,2], priority, True), [3,2,1,0])
        a = bottom_overlap([0]*4,[0]*4,[10,20,30,40],.5,priority)
        self.assertEqual(a['left_ids'],[20,40])
        with self.assertRaises(ValueError): ranking_order([1,2],[0,0])
