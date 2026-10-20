"""Numerical tables for every declared vocabulary/permutation follow-up."""
import csv
import json
from pathlib import Path
import numpy as np
import train_followup as experiment
from contracts_followup import sha,validate_evaluation

HERE=experiment.HERE
PAPER=experiment.PAPER
DEST=HERE/'results'
INPUT_HASHES={}

def write_csv(name,rows):
    with (DEST/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def read(path,primary=True):
    INPUT_HASHES[str(path.relative_to(PAPER)).replace('\\','/')]=sha(path)
    with np.load(path,allow_pickle=False) as f:
        a={k:f[k] for k in f.files}
    if primary:
        ids=json.loads((PAPER/'final_experiment/split_manifest.json').read_text())['roles']['test']
        mask=np.isin(a['rows'],ids)
        a={k:v[mask] for k,v in a.items()}
        assert set(a['rows'])==set(ids) and len(np.unique(a['rows']))==len(ids)
    assert len(a['rows'])==3275 and np.allclose(a['q'].sum(1),1)
    p=a['probabilities'];q=a['q']
    assert p.ndim==2 and p.shape[1] in (7,8) and q.shape==(len(p),10)
    assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
    assert np.isfinite(q).all() and (q>=0).all()
    assert np.allclose(p.sum(1),1,atol=1e-6,rtol=0)
    return a

def load_new(k,condition,seed,state):
    return read(HERE/'predictions'/f'k{k}_{condition}_seed{seed}_{state}'/'test_full.npz')

def load_old(condition,seed,state):
    folder=PAPER/('evidence_revision' if condition=='tie_uniform' else 'reviewer_followup')
    return read(folder/'predictions'/f'{condition}_seed{seed}_{state}'/'test_full.npz')

def loss(a):
    p=a['probabilities'];q=a['q']
    return 1-q[np.arange(len(q)),p.argmax(1)]

def order(a):
    return np.lexsort((a['rows'],-a['probabilities'].max(1)))

def keep(a,coverage=.8):
    return order(a)[:int(np.ceil(coverage*len(a['rows'])))]

def summary(a,ids=None):
    if ids is None:ids=keep(a)
    p=a['probabilities'];q=a['q'];k=p.shape[1]
    floor=1-q[:,:k].max(1);unsupported=1-q[:,:k].sum(1)
    d=loss(a);distance=((np.pad(p.astype(float),((0,0),(0,10-k)))-q)**2).sum(1)
    assert np.allclose(d,unsupported+(floor-unsupported)+(d-floor))
    average=lambda values:float(values[ids].mean()) if len(ids) else None
    percent=lambda values:100*average(values) if len(ids) else None
    return dict(n=len(q),accepted_n=len(ids),risk_pct=percent(d),
                unsupported_pp=percent(unsupported),ambiguity_pp=percent(floor-unsupported),
                excess_pp=percent(d-floor),distance=float(distance.mean()),
                accepted_distance=average(distance))

def coverage_and_strata(a,base):
    """Full-reference metrics; stratum population and selector scope are explicit."""
    coverage_rows=[];stratum_rows=[]
    for coverage in (.2,.5,.8,1.):
        ids=keep(a,coverage)
        coverage_rows.append(dict(**base,requested_coverage=coverage,**summary(a,ids)))
    k=a['probabilities'].shape[1]
    supported=a['q'][:,:k].sum(1)
    rest=a['q'][:,k:].sum(1)
    strata={'zero':supported==0,'partial':(supported>0)&(rest>0),'full':rest==0}
    assert np.all(sum(mask.astype(int) for mask in strata.values())==1)
    global_keep=np.zeros(len(supported),bool);global_keep[keep(a,.8)]=True
    for name,mask in strata.items():
        positions=np.flatnonzero(mask)
        for scope in ('all','global80','within80'):
            if scope=='all':ids=positions
            elif scope=='global80':ids=np.flatnonzero(mask&global_keep)
            else:
                ranked=positions[np.lexsort((a['rows'][positions],-a['probabilities'][positions].max(1)))]
                ids=ranked[:int(np.ceil(.8*len(ranked)))]
            stratum_rows.append(dict(**base,stratum=name,scope=scope,stratum_n=int(mask.sum()),
                                     stratum_retention=float(len(ids)/mask.sum()) if mask.any() else None,
                                     **summary(a,ids)))
    return coverage_rows,stratum_rows

def bootstrap(pairs,reps=5000):
    reference=pairs[0][0]
    names,groups=np.unique(reference['hashes'],return_inverse=True)
    for pair in pairs:
        for a in pair:
            assert np.array_equal(reference['rows'],a['rows'])
            assert np.array_equal(reference['q'],a['q'])
            assert np.array_equal(reference['hashes'],a['hashes'])
    ranked=[(order(a),order(b)) for a,b in pairs]
    losses=[(loss(a),loss(b)) for a,b in pairs]
    rng=np.random.default_rng(20261020)
    deltas=[]
    for _ in range(reps):
        count=np.bincount(rng.integers(len(names),size=len(names)),minlength=len(names))
        multiplicity=count[groups]
        n=int(np.ceil(.8*multiplicity.sum()))
        diff=[]
        for (oa,ob),(la,lb) in zip(ranked,losses):
            risks=[]
            for o,l in ((oa,la),(ob,lb)):
                m=multiplicity[o]
                prior=np.r_[0,np.cumsum(m)[:-1]]
                accepted=np.minimum(m,np.maximum(0,n-prior))
                assert accepted.sum()==n
                risks.append(float(np.dot(accepted,l[o])/n))
            diff.append(risks[0]-risks[1])
        deltas.append(100*np.mean(diff))
    points=[100*(loss(a)[keep(a)].mean()-loss(b)[keep(b)].mean()) for a,b in pairs]
    interval=np.quantile(deltas,[.025,.975])
    return dict(mean_pp=float(np.mean(points)),seed_differences_pp=points,
                seed_sd_pp=float(np.std(points,ddof=1)),minimum_seed_pp=float(min(points)),
                maximum_seed_pp=float(max(points)),lower_pp=float(interval[0]),upper_pp=float(interval[1]),
                replicates=reps,pixel_clusters=len(names),scope='fixed fitted models/seeds/observed votes')

def target_audit():
    # Annotation-only reconstruction keeps CPU numerical replay independent of
    # source photographs, which are deliberately not redistributed.
    annotation=PAPER/'data/ferplus_annotations/fer2013new.csv'
    manifest=json.loads((PAPER/'final_experiment/split_manifest.json').read_text())
    assert sha(annotation)==manifest['source_hashes']['data/ferplus_annotations/fer2013new.csv']
    with annotation.open(newline='',encoding='utf-8-sig') as handle:source=list(csv.DictReader(handle))
    ids=np.array(manifest['roles']['train'],dtype=np.int64)
    assert all(source[int(row)]['Usage']=='Training' for row in ids)
    vocabulary=('anger','disgust','fear','happiness','neutral','sadness','surprise','contempt','unknown','NF')
    votes=np.array([[int(source[int(row)][column]) for column in vocabulary] for row in ids],dtype=float)
    a=dict(rows=ids,q=votes/votes.sum(1,keepdims=True))
    INPUT_HASHES[str(annotation.relative_to(PAPER)).replace('\\','/')]=sha(annotation)
    records=[]
    targets={}
    for k,conditions in [(7,('soft','uniform','tie_uniform','permutation')),(8,('soft','uniform','tie_uniform'))]:
        for condition in conditions:
            t,w=experiment.targets(a['q'],a['rows'],k,condition)
            entropy=-(np.where(t>0,t*np.log(np.maximum(t,np.finfo(float).tiny)),0)).sum(1)
            marginals=(w[:,None]*t).sum(0)/w.sum()
            targets[k,condition]=(t,w,marginals)
            base_t,_,base_m=targets[k,'soft']
            records.append(dict(k=k,condition=condition,training_images=len(t),
                zero_supported_mass=int((w==0).sum()),
                changed_from_soft=int(np.any(np.abs(t-base_t)>1e-14,axis=1).sum()),
                weighted_target_entropy=float(np.dot(w,entropy)/w.sum()),
                unweighted_positive_entropy=float(entropy[w>0].mean()),
                marginal_l1_from_soft=float(np.abs(marginals-base_m).sum()),
                class_marginals=json.dumps(marginals.tolist())))
    write_csv('target_audit.csv',records)
    return records

def main():
    assert (HERE/'evaluation_complete.json').exists()
    provenance=validate_evaluation(HERE,PAPER,experiment.RUNS,portable=True)
    DEST.mkdir(exist_ok=True)
    rows=[];curves=[];data={};coverage_rows=[];strata_rows=[]
    for k,condition,seed in experiment.RUNS:
        for state in ('selected','epoch12')+ (('risk_selected',) if k==8 else ()):
            a=load_new(k,condition,seed,state);data[k,condition,seed,state]=a
            base=dict(k=k,condition=condition,seed=seed,state=state)
            rows.append(dict(**base,**summary(a)))
            c,s=coverage_and_strata(a,base);coverage_rows.extend(c);strata_rows.extend(s)
            o=order(a);risk=np.cumsum(loss(a)[o])/np.arange(1,len(o)+1)
            for i,r in enumerate(risk):
                curves.append(dict(k=k,condition=condition,seed=seed,state=state,
                                   coverage=(i+1)/len(o),risk_pct=float(100*r)))
    for condition in ('soft','uniform','tie_uniform'):
        for seed in experiment.SEEDS:
            for state in ('selected','epoch12'):
                a=load_old(condition,seed,state);data[7,condition,seed,state]=a
                base=dict(k=7,condition=condition,seed=seed,state=state)
                rows.append(dict(**base,**summary(a)))
                c,s=coverage_and_strata(a,base);coverage_rows.extend(c);strata_rows.extend(s)
                o=order(a);risk=np.cumsum(loss(a)[o])/np.arange(1,len(o)+1)
                for i,r in enumerate(risk):
                    curves.append(dict(**base,coverage=(i+1)/len(o),risk_pct=float(100*r)))
    write_csv('all_states.csv',rows)
    write_csv('coverage_results.csv',coverage_rows)
    write_csv('unsupported_mass_strata.csv',strata_rows)
    write_csv('risk_coverage.csv',curves)
    contrasts=[];fixed=[]
    for k,states in [(7,('selected','epoch12')),(8,('selected','epoch12','risk_selected'))]:
        for state in states:
            for control in (('permutation',) if k==7 else ('uniform','tie_uniform')):
                pairs=[(data[k,'soft',s,state],data[k,control,s,state]) for s in experiment.SEEDS]
                record=dict(k=k,state=state,left='soft',right=control,**bootstrap(pairs))
                contrasts.append(record)
                for seed,(a,b) in zip(experiment.SEEDS,pairs):
                    for selector,x in [('soft',a),(control,b)]:
                        ids=keep(x)
                        for predictor,z in [('soft',a),(control,b)]:
                            fixed.append(dict(k=k,state=state,seed=seed,control=control,selector=selector,
                                              predictor=predictor,**summary(z,ids)))
    write_csv('contrasts.csv',[{**r,'seed_differences_pp':json.dumps(r['seed_differences_pp'])} for r in contrasts])
    write_csv('fixed_sets.csv',fixed)
    audit=target_audit()
    report=dict(created_utc=experiment.now(),training_runs=12,states_new=33,
                primary_test_images=3275,pixel_clusters=3268,
                coverage_records=len(coverage_rows),stratum_records=len(strata_rows),
                full_coverage_results=[r for r in coverage_rows if r['requested_coverage']==1.],
                contrasts=contrasts,target_audit=audit,interpretation='adaptive explanatory follow-up',
                train_freeze_sha256=experiment.base.data.sha(HERE/'freeze.json'),
                evaluation_freeze_sha256=experiment.base.data.sha(HERE/'evaluation_freeze.json'),
                evaluation_integrity=provenance,prediction_input_sha256=INPUT_HASHES)
    (DEST/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=[r'\begin{table}[!htbp]\centering\caption{New vocabulary and target controls: Soft-minus-control complete-vote disagreement at 80\% coverage. Differences and conditional intervals are percentage points.}\label{tab:newfollowup}',
           r'\small\begin{tabular}{lllr r}\toprule Outputs & State & Control & Difference & 95\% interval \\ \midrule']
    for r in contrasts:
        state={'selected':'Distance-selected','epoch12':'Epoch 12','risk_selected':'Risk-selected'}[r['state']]
        lines.append(f"{r['k']} & {state} & {r['right'].replace('_','-')} & {r['mean_pp']:+.3f} & [{r['lower_pp']:+.3f}, {r['upper_pp']:+.3f}]"+r' \\')
    lines += [r'\bottomrule\end{tabular}\par\smallskip\footnotesize Three paired seeds per condition. The shared test source was previously exposed. Intervals condition on these fits and annotations and do not establish seed-population generality.\end{table}']
    (DEST/'tables.tex').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
