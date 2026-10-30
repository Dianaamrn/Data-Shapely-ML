import unittest
import numpy as np
from data_shapley.experiments import (removal_count, removal_selections, detection,
    relative_accuracy, evaluate_remaining)

class ExperimentTest(unittest.TestCase):
    def test_counts_and_relative_accuracy(self):
        self.assertEqual([removal_count(205,f) for f in (0,.1,.2,.3,.4)],[0,21,41,62,82])
        self.assertEqual([removal_count(150,f) for f in (0,.1,.2,.3,.4)],[0,15,30,45,60])
        self.assertEqual(relative_accuracy(.9,.75),1.2)
        self.assertIsNone(relative_accuracy(.1,0))
        with self.assertRaises(ValueError): removal_count(10,1)

    def test_selections_counts_reproducibility_and_stream_isolation(self):
        v=np.zeros(20);p=np.arange(20)[::-1]
        first=list(removal_selections(v,p,100,0))
        again=list(removal_selections(v,p,100,0))
        self.assertEqual(len(first),128)
        keys=[]
        for a,b in zip(first,again):
            strategy,f,trial,removed,key=a
            np.testing.assert_array_equal(removed,b[3])
            self.assertEqual(len(removed),removal_count(20,f))
            self.assertEqual(len(set(removed)),len(removed))
            if strategy!='RANDOM':
                np.testing.assert_array_equal(removed,np.arange(19,-1,-1)[:len(removed)])
            else:
                keys.append(key)
                expected=np.random.default_rng(np.random.SeedSequence(100,spawn_key=key)).choice(20,len(removed),replace=False)
                np.testing.assert_array_equal(removed,expected)
        self.assertEqual(len(set(keys)),120)
        other=list(removal_selections(np.arange(20),p,100,0))
        for a,b in zip(first,other):
            if a[0]=='RANDOM': np.testing.assert_array_equal(a[3],b[3])

    def test_detection_hand_computed_and_ties(self):
        values=np.arange(10.)
        mask=np.array([1,0,1,0,0,0,0,0,0,0],bool)
        result=detection(values,mask,np.arange(10))
        self.assertEqual(result['precision10'],1.)
        self.assertEqual(result['precision20'],.5)
        self.assertEqual(result['auroc'],15/16)
        self.assertEqual(result['fraction_flips_found'][:4],[0,.5,.5,1.])
        self.assertEqual(result['fraction_inspected'][-1],1.)
        tied=detection(np.zeros(10),mask,np.arange(10)[::-1])
        self.assertEqual(tied['auroc'],.5)
        self.assertEqual(tied['precision10'],0)
        self.assertIsNone(detection(values,np.zeros(10,bool),np.arange(10)))

    def test_removal_missing_class_and_unchanged_inputs(self):
        X=np.array([[0.],[1.],[10.],[11.]])
        y=np.array([0,0,1,1]);queries=np.array([[0.],[11.]])
        before=X.copy()
        result=evaluate_remaining(X,y,queries,np.array([0,1]),[2,3],(0,1))
        self.assertEqual(result['test_accuracy'],.5)
        self.assertEqual(result['remaining_1'],0)
        self.assertEqual(result['recall_1'],0)
        self.assertAlmostEqual(result['macro_f1'],1/3)
        np.testing.assert_array_equal(X,before)

if __name__=='__main__': unittest.main()
