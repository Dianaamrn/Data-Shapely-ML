"""pilot: pick the minimum number of permutations using validation data only.

two independent runs per condition are compared at 50, 100, 200 and 500
permutations. if any condition does not agree well enough at 500, both runs go
on to 1000 and 2000. the largest number reached is the study's minimum.

usage: python -m data_shapley.pilot
"""
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .data import (make_data_rngs, load_penguins, split_penguins, standardize_split,
                   generate_gaussian_split, corrupt_training_labels, CLASS_LABELS)
from .shapley import SortedKNNGame
from .metrics import rank_agreement, bottom_overlap, ranking_order
from .provenance import source_sha256, raw_sha256

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def diagnostics(a, b, ids, tie_priority=None):
    if tie_priority is None:
        tie_priority = np.random.default_rng(np.random.SeedSequence(100, spawn_key=(9, 0))).permutation(len(ids))
    means = [s.mean(0) for s in (a,b)]
    result = rank_agreement(*means)
    result['mean_absolute_value_difference'] = float(np.abs(means[0]-means[1]).mean())
    result['max_absolute_value_difference'] = float(np.abs(means[0]-means[1]).max())
    result['se90_each_stream'] = [float(np.quantile(s.std(0,ddof=1)/np.sqrt(len(s)),.9)) for s in (a,b)]
    result['bottom'] = {str(f):bottom_overlap(*means,ids,f,tie_priority) for f in (.1,.2)}
    result['descriptive_benchmarks_met'] = bool(result['spearman'] is not None and
        result['spearman'] >= .9 and all(r['overlap_fraction'] >= .8 for r in result['bottom'].values()))
    # these gaps describe uncertainty, they are not a statistical test

    result['boundary_each_stream'] = []
    for samples, mean in zip((a,b),means):
        order = ranking_order(mean,tie_priority)
        boundaries = {}
        for f in (.1,.2):
            count=int(np.floor(f*len(ids)+.5))
            inside,outside=order[count-1:count+1]
            delta=samples[:,outside]-samples[:,inside]
            se=float(delta.std(ddof=1)/np.sqrt(len(delta)))
            gap=float(mean[outside]-mean[inside])
            boundaries[str(f)]={'inside_id':int(ids[inside]),'outside_id':int(ids[outside]),
                'gap':gap,'paired_gap_se':se,'gap_over_se':gap/se if se else None}
        result['boundary_each_stream'].append(boundaries)
    return result


def main():
    folder = ROOT/'results/pilot'
    output = folder/'pilot_seed100.json'
    if output.exists():
        raise FileExistsError(f'{output} exists; delete results/pilot to rerun the pilot.')
    folder.mkdir(parents=True, exist_ok=True)
    rngs, data_manifest = make_data_rngs(100)
    penguins, _ = standardize_split(split_penguins(load_penguins(ROOT/'data/raw/penguins.csv'),rngs['penguins_split']))
    synthetic, _ = standardize_split(generate_gaussian_split(rngs['synthetic_train'],rngs['synthetic_validation'],rngs['synthetic_test']))
    pn = corrupt_training_labels(penguins.train,41,CLASS_LABELS,rngs['penguins_noise'])
    sn = corrupt_training_labels(synthetic.train,30,(0,1),rngs['synthetic_noise'])
    conditions=[('clean_penguins',penguins,penguins.train.y),('noisy_penguins',penguins,pn.training.y),('noisy_gaussian',synthetic,sn.training.y)]
    report={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
        'run_seed':100,'initial_budgets_per_stream':[50,100,200,500],
        'conditional_extension_per_stream':[1000,2000],'independent_streams':2,
        'stopping_rule':'If any condition misses descriptive benchmarks at 500, extend both retained streams for every condition to 1000 then 2000; stop at 2000 regardless of outcome.',
        'benchmarks':{'spearman_min':.9,'both_overlap_fraction_min':.8,
           'interpretation':'pragmatic descriptive references, not convergence proof or statistical acceptance thresholds'},
        'budget_definition':'M is per independent stream. Do not treat pooled 2M estimates as an M-budget run.',
        'test_scored':False,'data_manifest':data_manifest,
        'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform(),
        'processor':platform.processor(),'conditions':{},
        'sha256':{p:(source_sha256(ROOT/p) if p.endswith('.py') else raw_sha256(ROOT/p)) for p in
                  ['data/raw/penguins.csv','data_shapley/data.py','data_shapley/knn.py','data_shapley/shapley.py','data_shapley/metrics.py','data_shapley/provenance.py','data_shapley/pilot.py']}}
    games={}; streams={}; generators={}
    for index,(name,split,observed) in enumerate(conditions):
        start=time.perf_counter()
        games[name]=SortedKNNGame(split.train.X,observed,split.validation.X,split.validation.y,split.train.y)
        streams[name]=[np.empty((0,len(observed))),np.empty((0,len(observed)))]
        generators[name]=[np.random.default_rng(np.random.SeedSequence(100,spawn_key=(8,index,s))) for s in range(2)]
        report['conditions'][name]={'n':len(observed),'q':len(split.validation.y),
            'training_ids':split.train.observation_ids.tolist(),'validation_ids':split.validation.observation_ids.tolist(),
            'ranking_ties':{'entropy':100,'spawn_key':[9,index], 'priority_by_training_row':np.random.default_rng(np.random.SeedSequence(100,spawn_key=(9,index))).permutation(len(observed)).tolist()},
            'seed_streams':[{'entropy':100,'spawn_key':[8,index,s]} for s in range(2)],
            'setup_seconds':time.perf_counter()-start,'sampling_seconds_each_stream':[0.,0.], 'rows':[]}
    write(output,report)  # save the settings before sampling starts
    started=time.perf_counter()
    budgets=[50,100,200,500]
    for m in budgets:
        for name,split,observed in conditions:
            rec=report['conditions'][name]
            for s in range(2):
                chunks=[streams[name][s]]
                remaining=m-len(streams[name][s])
                while remaining:
                    size=min(50,remaining)
                    start=time.perf_counter()
                    result=games[name].sample(size,generators[name][s])
                    rec['sampling_seconds_each_stream'][s]+=time.perf_counter()-start
                    np.testing.assert_allclose(result.marginals.sum(1),result.prefix_utilities[:,-1]-result.prefix_utilities[:,0],rtol=0,atol=1e-12)
                    chunks.append(result.marginals)
                    remaining-=size
                streams[name][s]=np.concatenate(chunks)
                print(name,'stream',s+1,'reached',m,flush=True)
            row=diagnostics(*streams[name],split.train.observation_ids, np.asarray(rec['ranking_ties']['priority_by_training_row']))
            row.update(M_per_stream=m,cumulative_sampling_seconds=sum(rec['sampling_seconds_each_stream']))
            rec['rows'].append(row)
            rec['rng_states']=[g.bit_generator.state for g in generators[name]]
            np.savez_compressed(folder/('pilot_'+name+'_samples.npz'),
                                stream_a=streams[name][0],stream_b=streams[name][1],observation_ids=split.train.observation_ids)
            write(output,report)
            print(name,m,'rho',round(row['spearman'],3),'overlap10',round(row['bottom']['0.1']['overlap_fraction'],3),
                  'overlap20',round(row['bottom']['0.2']['overlap_fraction'],3),flush=True)
        if m==500:
            extend=any(not r['rows'][-1]['descriptive_benchmarks_met'] for r in report['conditions'].values())
            report['extension_triggered']=extend
            if extend: budgets.extend([1000,2000])
    report['M_selected']=budgets[-1]
    report['status']='complete'
    report['completed_utc']=datetime.now(timezone.utc).isoformat()
    report['sampling_and_diagnostics_wall_seconds']=time.perf_counter()-started
    write(output,report)
    print('Pilot complete; minimum permutations per study run: M_min =',budgets[-1],flush=True)


if __name__=='__main__': main()
