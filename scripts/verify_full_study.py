"""rebuild every saved study result from the seeds and stop at the first mismatch.

usage: python scripts/verify_full_study.py
"""
import csv, json, sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import os
os.chdir(root)
import numpy as np
from data_shapley.study import SEEDS, checksums, seed_folder
from data_shapley.data import (make_data_rngs,load_penguins,split_penguins,standardize_split,generate_gaussian_split,CLASS_LABELS,corrupt_training_labels)
from data_shapley.knn import predict_knn
from data_shapley.shapley import make_knn_utility, evaluate_permutations_naive

def verify_seed(seed):
    base=seed_folder(seed)
    report=json.loads((base/'manifest.json').read_text());assert report['status']=='complete'
    assert report['sha256']==checksums(), f'seed {seed}: code or data differ from the recorded checksums'
    rngs,_=make_data_rngs(seed)
    p,_=standardize_split(split_penguins(load_penguins('data/raw/penguins.csv'),rngs['penguins_split']))
    g,_=standardize_split(generate_gaussian_split(rngs['synthetic_train'],rngs['synthetic_validation'],rngs['synthetic_test']))
    with (base/'removal.csv').open() as f: rows=list(csv.DictReader(f))
    assert len(rows)==387
    verification={}
    for ci,(name,rec) in enumerate(report['conditions'].items()):
     split=g if name=='noisy_gaussian' else p
     ids=np.asarray(rec['training_ids']);labels=np.asarray(rec['observed_labels']);priority=np.asarray(rec['tie_priority_by_training_row'])
     np.testing.assert_array_equal(ids,split.train.observation_ids)
     np.testing.assert_array_equal(rec['original_labels'],split.train.y)
     if name=='clean_penguins':
      np.testing.assert_array_equal(labels,split.train.y)
      assert not any(rec['corrupted_mask'])
     else:
      count,classes,key=(30,(0,1),'synthetic_noise') if name=='noisy_gaussian' else (41,CLASS_LABELS,'penguins_noise')
      noise=corrupt_training_labels(split.train,count,classes,rngs[key])
      np.testing.assert_array_equal(labels,noise.training.y)
      np.testing.assert_array_equal(rec['corrupted_mask'],noise.corruption_mask)
     assert not set(ids)&set(rec['test_ids']) and not set(rec['test_ids'])&set(rec['validation_ids'])
     np.testing.assert_array_equal(priority,np.random.default_rng(np.random.SeedSequence(seed,spawn_key=(9,ci))).permutation(len(ids)))
     with np.load(base/f'{name}_valuation.npz') as data:
      values=data['values'];marginals=data['marginals'];orders=data['permutations']
      np.testing.assert_array_equal(values,marginals.mean(0))
      rng=np.random.default_rng(np.random.SeedSequence(seed,spawn_key=(10,ci)))
      for order in orders: np.testing.assert_array_equal(order,rng.permutation(len(ids)))
      # the run must stop at the first check where the rule is below 0.05
      trace=rec['convergence_trace'];used=rec['M_used'];lag,limit=report['convergence_rule']['lag'],report['convergence_rule']['threshold']
      assert len(orders)==len(marginals)==used and [t for t,_ in trace]==list(range(report['M_min'],used+1,lag))
      counts=np.rint(marginals*len(split.validation.y)).astype(np.int64)  # whole-number counts, as in the study
      np.testing.assert_allclose(counts/len(split.validation.y),marginals,rtol=0,atol=1e-9)
      for t,rule in trace:
       cur,prev=counts[:t].sum(0)/t,counts[:t-lag].sum(0)/(t-lag);keep=counts[:t].sum(0)!=0
       np.testing.assert_allclose(rule,np.mean(np.abs(cur[keep]-prev[keep])/np.abs(cur[keep])) if keep.any() else 0.,rtol=1e-12,atol=0)
      assert all(r>=limit for _,r in trace[:-1]) and (trace[-1][1]<limit or used==report['M_max']) and rec['converged']==(trace[-1][1]<limit)
      utility=make_knn_utility(split.train.X,labels,split.validation.X,split.validation.y,split.train.y)
      check=evaluate_permutations_naive(len(ids),utility,orders[[0,-1]])
      np.testing.assert_array_equal(check.marginals,marginals[[0,-1]])
      np.testing.assert_allclose(marginals.sum(1),utility(tuple(range(len(ids))))-utility(()),rtol=0,atol=1e-12)
     low=sorted(range(len(ids)),key=lambda i:(values[i],priority[i]));high=sorted(range(len(ids)),key=lambda i:(-values[i],priority[i]))
     condition_rows=[r for r in rows if r['condition']==name]
     assert len(condition_rows)==129
     for row,selection in zip(condition_rows,[None]+rec['removals']):
      removed=[] if selection is None else selection['removed_ids']
      count=int(row['n_removed']);assert len(removed)==count==int(np.floor(float(row['fraction_requested'])*len(ids)+.5))
      assert len(set(removed))==count
      if selection:
       if selection['strategy'] in ('LOW','HIGH'):
        order=low if selection['strategy']=='LOW' else high
        assert removed==ids[order[:count]].tolist()
       else:
        stream=selection['random_stream'];rng=np.random.default_rng(np.random.SeedSequence(stream['entropy'],spawn_key=tuple(stream['spawn_key'])))
        assert removed==ids[rng.choice(len(ids),count,replace=False)].tolist()
      keep=~np.isin(ids,removed)
      pred=predict_knn(split.train.X[keep],labels[keep],split.test.X)
      acc=sum(int(a==b) for a,b in zip(pred,split.test.y))/len(pred)
      assert acc==float(row['test_accuracy'])
      assert float(row['relative_accuracy'])==acc/rec['baseline']['test_accuracy']
      classes=(0,1) if name=='noisy_gaussian' else CLASS_LABELS
      f1=[]
      for c in classes:
       tp=sum(int(a==c and b==c) for a,b in zip(pred,split.test.y));fp=sum(int(a==c and b!=c) for a,b in zip(pred,split.test.y));fn=sum(int(a!=c and b==c) for a,b in zip(pred,split.test.y))
       f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
       assert int(row['remaining_'+str(c)])==sum(labels[keep]==c)
       truth=split.test.y==c
       assert float(row['recall_'+str(c)])==float(np.mean(pred[truth]==c))
      np.testing.assert_allclose(float(row['macro_f1']),sum(f1)/len(f1),rtol=0,atol=1e-15)
     if rec['detection']:
      mask=np.asarray(rec['corrupted_mask']);positive=np.flatnonzero(mask);negative=np.flatnonzero(~mask)
      auc=sum(1 if values[i]<values[j] else .5 if values[i]==values[j] else 0 for i in positive for j in negative)/(len(positive)*len(negative))
      assert auc==rec['detection']['auroc']
      np.testing.assert_array_equal(np.r_[0,np.cumsum(mask[low])]/mask.sum(),rec['detection']['fraction_flips_found'])
     verification[name]={'test_rows_replayed':129,'full_permutation_stream_reconstructed':True,'naive_marginal_rows_checked':[0,len(orders)-1],'all_efficiency_checks_passed':True,'removal_ids_and_class_counts_verified':True,'detection_independently_verified':bool(rec['detection'])}
     print('Verified seed',seed,name,flush=True)
    return verification

if __name__ == '__main__':
    checks={str(seed):verify_seed(seed) for seed in SEEDS}
    target=root/'results/study/study_verification.json'
    target.write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('All 3,870 test outcomes reconstructed; every order, stopping point and 60 naive marginal rows checked.')
