"""recompute the summary tables another way, with exact fractions.

usage: python scripts/verify_study_summaries.py
"""
import csv,json
from pathlib import Path
from fractions import Fraction
import numpy as np
root=Path(__file__).resolve().parents[1];folder=root/'results/study'
def read(name):
 with (folder/name).open() as f:return list(csv.DictReader(f))
raw=read('study_removal.csv');summaries=read('study_summary.csv');perrun=read('study_per_run.csv')
for ci,name in enumerate(('clean_penguins','noisy_penguins','noisy_gaussian')):
 n=1000 if name=='noisy_gaussian' else 69
 for fi,f in enumerate((.1,.2,.3,.4),1):
  deltas=[];relative={k:[] for k in ('LOW','HIGH','RANDOM')}
  for seed in range(100,110):
   group=[r for r in raw if r['condition']==name and int(r['run_seed'])==seed]
   baseline=next(float(r['test_accuracy']) for r in group if r['strategy']=='BASELINE')
   rows=[r for r in group if float(r['fraction_requested'])==f]
   low=next(r for r in rows if r['strategy']=='LOW');lc=round(float(low['test_accuracy'])*n)
   random=[round(float(r['test_accuracy'])*n) for r in rows if r['strategy']=='RANDOM'];assert len(random)==30
   delta=Fraction(30*lc-sum(random),30*n);deltas.append(delta)
   saved=next(r for r in perrun if r['condition']==name and int(r['run_seed'])==seed and float(r['fraction_requested'])==f)
   np.testing.assert_allclose(float(saved['low_minus_random']),float(delta),rtol=0,atol=1e-15)
   for strategy in relative:
    relative[strategy].append(sum(float(r['test_accuracy'])/baseline for r in rows if r['strategy']==strategy)/(30 if strategy=='RANDOM' else 1))
  row=next(r for r in summaries if r['condition']==name and float(r['fraction_requested'])==f)
  assert [int(row[k]) for k in ('wins','ties','losses')]==[sum(d>0 for d in deltas),sum(d==0 for d in deltas),sum(d<0 for d in deltas)]
  np.testing.assert_allclose(float(row['low_minus_random_mean']),float(sum(deltas)/10),rtol=0,atol=1e-15)
  for strategy,field in [('LOW','low'),('HIGH','high'),('RANDOM','random')]:
   np.testing.assert_allclose(float(row[field+'_relative_mean']),sum(relative[strategy])/10,rtol=0,atol=1e-15)
  rng=np.random.default_rng(np.random.SeedSequence(100,spawn_key=(12,ci,fi)))
  estimates=[sum(float(deltas[i]) for i in rng.integers(0,10,10))/10 for _ in range(5000)]
  bounds=np.quantile(estimates,[.025,.975])
  np.testing.assert_allclose(bounds,[float(row['bootstrap95_low']),float(row['bootstrap95_high'])],rtol=0,atol=1e-15)
(folder/'study_aggregation_verification.json').write_text(json.dumps({'paired_differences':'all 120 checked using exact rational correct-count arithmetic','wins_ties_losses':'all 12 groups checked','relative_accuracy':'all 36 strategy/group means independently recomputed from per-run ratios','bootstrap':'all 12 intervals reconstructed with explicit whole-run sampling loop','random_weighting':'30 trials averaged within each run; 10 equal-weight runs'},indent=2)+'\n',encoding='utf-8',newline='\n')
print('Independent aggregate checks passed: paired differences, W/T/L, ratios and all 12 bootstrap intervals.')
