import unittest
import numpy as np
from data_shapley.metrics import accuracy, macro_f1, majority_class


class MetricsTest(unittest.TestCase):
    def test_hand_accuracy_and_macro_f1(self):
        # macro-f1 worked out by hand is 59/90
        truth = ['A', 'A', 'B', 'B', 'C', 'C']
        predicted = ['A', 'A', 'A', 'B', 'B', 'C']
        self.assertAlmostEqual(accuracy(truth, predicted), 4/6)
        self.assertAlmostEqual(macro_f1(truth, predicted, ['A','B','C']), 59/90)

    def test_absent_class_and_zero_denominators(self):
        self.assertEqual(macro_f1([0,0], [0,0], [0,1,2]), 1/3)
        self.assertEqual(macro_f1([0,0], [1,1], [0,1,2]), 0)
        self.assertEqual(accuracy([0,1], [1,0]), 0)

    def test_majority_and_sorted_ties(self):
        self.assertEqual(majority_class(['Gentoo','Adelie','Gentoo']), 'Gentoo')
        self.assertEqual(majority_class(['Gentoo','Adelie']), 'Adelie')
        self.assertEqual(majority_class([1,0]), 0)

    def test_invalid_metrics(self):
        for a,b in (([],[]), ([0],[0,1]), ([[0]],[[0]]), ([float('nan')],[0]), ([0],['0'])):
            with self.assertRaises(ValueError): accuracy(a,b)
        for classes in ([0,0], [0], []):
            with self.assertRaises(ValueError): macro_f1([0,1],[0,1],classes)
        with self.assertRaises(ValueError): majority_class([])


if __name__ == '__main__': unittest.main()
