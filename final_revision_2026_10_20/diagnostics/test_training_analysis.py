"""Independent toy checks for the predeclared new training analysis."""
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

# Execute the actual pure analysis function definitions while avoiding the
# unrelated torch/CUDA imports used by the standalone training runner.
path=Path(__file__).resolve().parent.parent/'training/analyze_followup.py'
tree=ast.parse(path.read_text())
names={'loss','order','keep','summary','coverage_and_strata','bootstrap','target_audit'}
functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
namespace={'np':np}
exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace)
a=SimpleNamespace(**{name:namespace[name] for name in names})


def toy():
    q=np.zeros((10,10));q[:4,0]=1;q[4:8,1]=.5;q[4:8,8]=.5;q[8:,9]=1
    p=np.tile(np.array([.6,.15,.05,.05,.05,.05,.05]),(10,1))
    p[4:8]=np.array([.1,.6,.1,.05,.05,.05,.05])
    return dict(q=q,probabilities=p,rows=np.arange(10),hashes=np.array(['a','a','b','c','d','e','f','g','h','i']))


def test_strata_and_full_coverage_accounting():
    x=toy();coverage,strata=a.coverage_and_strata(x,dict(k=7,condition='soft',state='selected',seed=17))
    assert [r['accepted_n'] for r in coverage]==[2,5,8,10]
    full=coverage[-1]
    expected=100*np.mean([0,0,0,0,.5,.5,.5,.5,1,1])
    assert full['risk_pct']==expected
    assert np.isclose(full['risk_pct'],full['unsupported_pp']+full['ambiguity_pp']+full['excess_pp'])
    assert {r['stratum']:r['stratum_n'] for r in strata}=={'full':4,'partial':4,'zero':2}
    assert [r['accepted_n'] for r in strata if r['stratum']=='zero' and r['scope']=='global80']==[0]
    assert [r['risk_pct'] for r in strata if r['stratum']=='zero' and r['scope']=='global80']==[None]


def test_paired_cluster_bootstrap_matches_explicit_small_resampling():
    x=toy();y={k:v.copy() for k,v in x.items()};y['probabilities']=np.roll(y['probabilities'],1,axis=1)
    pairs=[(x,y)]*3
    got=a.bootstrap(pairs,reps=12)
    hashes,groups=np.unique(x['hashes'],return_inverse=True)
    rng=np.random.default_rng(20261020);values=[]
    for repeat in range(12):
        selected=rng.integers(len(hashes),size=len(hashes))
        expanded=np.concatenate([np.flatnonzero(groups==group) for group in selected])
        count=int(np.ceil(.8*len(expanded)));risks=[]
        for data in (x,y):
            ranked=expanded[np.lexsort((data['rows'][expanded],-data['probabilities'][expanded].max(1)))]
            risks.append(a.loss(data)[ranked[:count]].mean())
        values.append(100*(risks[0]-risks[1]))
    lower,upper=np.quantile(values,[.025,.975])
    assert np.isclose(got['lower_pp'],lower) and np.isclose(got['upper_pp'],upper)
    assert got['pixel_clusters']==9 and got['seed_sd_pp']==0


def test_target_audit_reconstructs_from_annotation_csv_without_images():
    diagnostic=Path(__file__).resolve().parent
    spec=importlib.util.spec_from_file_location('training_only_targets',diagnostic/'new_targets.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    target_function,_=module.frozen_target_function()
    captured={}
    namespace.update(csv=csv,json=json,PAPER=diagnostic.parents[1],INPUT_HASHES={},
                     experiment=SimpleNamespace(targets=target_function),
                     sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest(),
                     write_csv=lambda name,rows:captured.update({name:rows}))
    records=a.target_audit()
    assert len(records)==7 and len(captured['target_audit.csv'])==7
    with (diagnostic/'new_target_audit.csv').open(newline='') as handle:original=list(csv.DictReader(handle))
    for row in records:
        expected=next(r for r in original if int(r['k'])==row['k'] and r['condition']==row['condition'])
        assert row['zero_supported_mass']==int(expected['zero_mass_images'])
        assert abs(row['weighted_target_entropy']-float(expected['weighted_mean_entropy_nats']))<1e-13
