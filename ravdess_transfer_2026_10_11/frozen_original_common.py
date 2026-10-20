"""Data contracts and scalar metrics for the frozen clip-transfer evaluation."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
PAPER=HERE.parent
CLASSES=('angry','disgust','fear','happy','neutral','sad','surprise')
REFERENCES=('raw9','neutral_calm_merged')
SEEDS=(17,42,89)
CONDITIONS=('soft','uniform','tie_uniform')
STATES=('selected','epoch12')
SCORES=('msp','margin','negative_entropy')
COVERAGES=(.2,.5,.8,1.)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(4*1024*1024):h.update(block)
    return h.hexdigest()


def annotations():
    path=PAPER/'recovery_2026_10_10/external_data/ravdess/video_only_speech_votes.csv'
    with path.open(newline='') as f:rows=sorted(csv.DictReader(f),key=lambda r:r['filename'])
    assert len(rows)==len({r['filename'] for r in rows})==1440
    names=np.array([r['filename'] for r in rows])
    actors=np.array([int(r['actor']) for r in rows],dtype=np.int16)
    q=np.array([[float(r['q_'+c]) for c in (*CLASSES,'calm','none')] for r in rows])
    assert np.isfinite(q).all() and (q>=0).all() and np.allclose(q.sum(1),1,atol=1e-12)
    assert all((actors==actor).sum()==60 for actor in range(1,25))
    return names,actors,q


def reference(q,name):
    if name not in REFERENCES:raise ValueError(name)
    result=np.array(q,dtype=float,copy=True)
    if name=='neutral_calm_merged':
        result[:,4]+=result[:,7];result[:,7]=0
    return result


def components(p,q,valid):
    result=np.full((len(q),4),np.nan,dtype=float)
    if valid.any():
        pp=np.asarray(p[valid],dtype=float);qq=np.asarray(q[valid],dtype=float)
        assert pp.shape[1]==7 and qq.shape[1]==9
        assert np.isfinite(pp).all() and (pp>=0).all() and np.allclose(pp.sum(1),1,atol=1e-6,rtol=0)
        chosen=qq[np.arange(len(qq)),pp.argmax(1)]
        w=qq[:,:7].sum(1);maximum=qq[:,:7].max(1)
        result[valid]=np.column_stack((1-chosen,1-w,w-maximum,maximum-chosen))
        assert np.allclose(result[valid,0],result[valid,1:].sum(1),atol=1e-12)
    return result


def scores(p,valid,name):
    if name not in SCORES:raise ValueError(name)
    out=np.full(len(p),-np.inf)
    pp=np.asarray(p[valid],dtype=float)
    if name=='msp':out[valid]=pp.max(1)
    elif name=='margin':
        order=np.sort(pp,axis=1);out[valid]=order[:,-1]-order[:,-2]
    else:
        terms=np.zeros_like(pp);nonzero=pp>0
        terms[nonzero]=pp[nonzero]*np.log(pp[nonzero])
        out[valid]=terms.sum(1)
    return out


def ranked_valid(p,valid,name):
    value=scores(p,valid,name)
    indices=np.flatnonzero(valid)
    # Arrays are in canonical filename order; positional ID therefore breaks ties.
    return indices[np.lexsort((indices,-value[indices]))]


def select(p,valid,name,coverage):
    n=math.ceil(coverage*len(valid))
    if n>int(valid.sum()):return None
    return ranked_valid(p,valid,name)[:n]


def checkpoint_path(family,condition,seed,state):
    folder='backbone_followup' if family=='resnet18' else 'evidence_revision' if condition=='tie_uniform' else 'reviewer_followup'
    return PAPER/folder/'weights'/f'{condition}_seed{seed}_{state}.pt'


def bootstrap_risk(order,loss,actors,counts,coverage=.8):
    """Re-rank a resampled actor multiset, preserving all-clip denominator."""
    weights=counts[:,actors[order]-1]
    population=counts.sum(1)*60
    target=np.ceil(coverage*population).astype(int)
    available=weights.sum(1)
    before=np.cumsum(weights,axis=1,dtype=np.int32)-weights
    accepted=np.minimum(weights,np.maximum(0,target[:,None]-before))
    risk=(accepted@loss[order])/target
    risk[available<target]=np.nan
    return risk


def write_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def require_frozen():
    frozen=json.loads((HERE/'freeze.json').read_text())
    for name,expected in frozen['source_hashes'].items():
        assert sha(PAPER/name)==expected, 'Frozen source changed: '+name
    return frozen
