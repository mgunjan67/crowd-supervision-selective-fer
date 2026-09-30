"""All predefined endpoints from the six new retained probability arrays."""
from pathlib import Path
import csv,hashlib,json,sys
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
sys.path.insert(0,str(ROOT/'code'))
from emotion_cue.metrics import classification_metrics
SEEDS=(17,42,89);CONDITIONS=('hard','soft');COVERAGES=(.2,.5,.8,1.);TARGETS=(.1,.2,.3)
CLASSES=('angry','disgust','fear','happy','neutral','sad','surprise')

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def load(condition,seed,role):
    path=OUT/'runs'/f'{condition}_seed{seed}'/f'{role}_predictions.npz'
    with np.load(path,allow_pickle=False) as f:a={k:f[k] for k in f.files}
    p=a['probabilities'].astype(float)
    assert p.shape==(len(a['rows']),7) and np.isfinite(p).all() and np.all(p>=0)
    assert np.allclose(p.sum(1),1,atol=1e-6) and np.allclose(a['q'].sum(1),1)
    a['probabilities']=p;return a

def per_image(a):
    p,q=a['probabilities'],a['q'];pred=p.argmax(1)
    risk=1-q[np.arange(len(q)),pred];floor=1-q[:,:7].max(1)
    return dict(vote_risk=risk,oracle_floor=floor,excess_risk=risk-floor,
        distribution_distance=((np.pad(p,((0,0),(0,3)))-q)**2).sum(1),
        out_of_scope_mass=q[:,7:].sum(1),eligible=q[:,:7].max(1)>.5,
        original_error=(pred!=a['labels']).astype(float),
        joint_error=((pred!=a['labels'])|(pred!=q[:,:7].argmax(1))).astype(float),
        reference_disagreement=(a['labels']!=q[:,:7].argmax(1)).astype(float))

def rank(a):
    return np.lexsort((a['rows'],-a['probabilities'].max(1)))

def metric_summary(a,keep):
    values=per_image(a);count=int(keep.sum());n=len(keep)
    report=dict(population_n=n,accepted_n=count,coverage=count/n)
    for key in ('vote_risk','oracle_floor','excess_risk','distribution_distance','out_of_scope_mass','original_error'):
        report[key]=float(values[key][keep].mean()) if count else None
    eligible=keep&values['eligible'];nn=int(eligible.sum());report['accepted_majority_n']=nn
    report['accepted_nonmajority_n']=int((keep&~values['eligible']).sum())
    for prefix,mask in (('majority',eligible),('nonmajority',keep&~values['eligible'])):
        for key in ('vote_risk','oracle_floor','excess_risk'):
            report[f'{prefix}_{key}']=float(values[key][mask].mean()) if mask.any() else None
    report['joint_error']=float(values['joint_error'][eligible].mean()) if nn else None
    report['reference_disagreement']=float(values['reference_disagreement'][eligible].mean()) if nn else None
    report['shared_reference_error_contribution']=report['joint_error']-report['reference_disagreement'] if nn else None
    return report

def threshold(score,loss,target,minimum=100):
    order=np.argsort(-score,kind='stable')
    ends=np.r_[np.flatnonzero(np.diff(score[order])!=0),len(order)-1]
    cumulative=np.cumsum(loss[order],dtype=float)
    valid=(ends+1>=minimum)&(cumulative[ends]/(ends+1)<=target+1e-12)
    return float(score[order[ends[np.flatnonzero(valid)[-1]]]]) if valid.any() else None

def common_keep(a,coverage):
    keep=np.zeros(len(a['rows']),bool)
    keep[rank(a)[:max(1,int(np.ceil(coverage*len(keep))))]]=True
    return keep

def bootstrap_primary(arrays,repetitions=5000):
    reference=arrays['hard',17]
    names,groups=np.unique(reference['hashes'],return_inverse=True)
    for a in arrays.values():
        assert np.array_equal(a['rows'],reference['rows']) and np.array_equal(a['q'],reference['q'])
    orders={k:rank(a) for k,a in arrays.items()}
    losses={k:per_image(a)['vote_risk'] for k,a in arrays.items()}
    rng=np.random.default_rng(20260921);deltas=[]
    # Cluster multiplicities are shared by all models and fixed seeds. Rankings
    # are recomputed implicitly by accumulating these multiplicities in score order.
    for _ in range(repetitions):
        cluster_counts=np.bincount(rng.integers(len(names),size=len(names)),minlength=len(names))
        multiplicity=cluster_counts[groups]
        accepted=int(np.ceil(.8*multiplicity.sum()))
        seed_differences=[]
        for seed in SEEDS:
            metrics={}
            for condition in CONDITIONS:
                key=condition,seed;order=orders[key];m=multiplicity[order]
                prior=np.r_[0,np.cumsum(m)[:-1]]
                selected=np.minimum(m,np.maximum(0,accepted-prior))
                assert selected.sum()==accepted
                metrics[condition]=float(np.dot(selected,losses[key][order])/accepted)
            seed_differences.append(metrics['soft']-metrics['hard'])
        deltas.append(np.mean(seed_differences))
    point=[]
    for seed in SEEDS:
        r={condition:metric_summary(arrays[condition,seed],common_keep(arrays[condition,seed],.8))['vote_risk'] for condition in CONDITIONS}
        point.append(r['soft']-r['hard'])
    lo,hi=np.quantile(deltas,[.025,.975])
    return dict(endpoint='soft minus hard expected vote disagreement at 80% common coverage',
        point_difference_pp=float(100*np.mean(point)),seed_differences_pp=[float(100*x) for x in point],
        lower_pp=float(100*lo),upper_pp=float(100*hi),repetitions=repetitions,pixel_clusters=len(names),
        test_n=len(reference['rows']),accepted_n=int(np.ceil(.8*len(reference['rows']))),
        bootstrap_scope='paired pixel-cluster resampling, fixed models/seeds/annotations, reranked samples')

def write_csv(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def compute():
    manifest=json.loads((OUT/'split_manifest.json').read_text());test_rows=np.array(manifest['roles']['test'])
    full=[];common=[];operating=[];curves=[];arrays={};sources={}
    for seed in SEEDS:
        for condition in CONDITIONS:
            name=f'{condition}_seed{seed}'
            a=load(condition,seed,'test_full');cal=load(condition,seed,'calibration')
            mask=np.isin(a['rows'],test_rows);primary={k:v[mask] for k,v in a.items()}
            arrays[condition,seed]=primary
            for role in ('test_full','calibration'):
                path=OUT/'runs'/name/f'{role}_predictions.npz';sources[path.relative_to(ROOT).as_posix()]=digest(path)
            for population,part in (('primary',primary),('full_official',a)):
                row=dict(condition=condition,seed=seed,population=population,**metric_summary(part,np.ones(len(part['rows']),bool)))
                p,q=part['probabilities'],part['q'];labels=p.argmax(1)
                cm=classification_metrics(part['labels'],labels,CLASSES)
                row.update(original_accuracy=cm['accuracy'],original_weighted_f1=cm['weighted_f1'],original_macro_f1=cm['macro_f1'])
                eligible=q[:,:7].max(1)>.5
                cm=classification_metrics(q[eligible,:7].argmax(1),labels[eligible],CLASSES)
                row.update(crowd_majority_n=int(eligible.sum()),crowd_weighted_f1=cm['weighted_f1'])
                order=rank(part);loss=per_image(part)['vote_risk'];riskcurve=np.cumsum(loss[order])/np.arange(1,len(order)+1)
                row['aurc_all_prefixes']=float(riskcurve.mean());full.append(row)
                if population=='primary':
                    for i,risk in enumerate(riskcurve):
                        curves.append(dict(condition=condition,seed=seed,accepted_n=i+1,coverage=(i+1)/len(order),vote_risk=float(risk)))
            for coverage in COVERAGES:
                common.append(dict(condition=condition,seed=seed,requested_coverage=coverage,
                    **metric_summary(primary,common_keep(primary,coverage))))
            cal_score=cal['probabilities'].max(1);cal_loss=per_image(cal)['vote_risk']
            for target in TARGETS:
                cutoff=threshold(cal_score,cal_loss,target)
                cal_keep=cal_score>=cutoff if cutoff is not None else np.zeros(len(cal_score),bool)
                keep=primary['probabilities'].max(1)>=cutoff if cutoff is not None else np.zeros(len(primary['rows']),bool)
                operating.append(dict(condition=condition,seed=seed,target=target,threshold=cutoff,
                    calibration_n=int(cal_keep.sum()),calibration_risk=float(cal_loss[cal_keep].mean()) if cal_keep.any() else None,
                    **metric_summary(primary,keep)))
    return full,common,operating,curves,arrays,sources

def main():
    full,common,operating,curves,arrays,sources=compute()
    for name,rows in (('full_results.csv',full),('common_coverage.csv',common),('calibrated_operating_points.csv',operating),('risk_coverage.csv',curves)):
        write_csv(name,rows)
    result=bootstrap_primary(arrays)
    # One machine-readable source for every main-manuscript result and figure.
    reported=[]
    for kind,rows in (('full',full),('common',common),('calibrated',operating),('curve',curves)):
        reported.extend(dict(record_type=kind,**row) for row in rows)
    reported.append(dict(record_type='primary',**{k:v for k,v in result.items() if not isinstance(v,list)}))
    fields=['record_type']+sorted({k for row in reported for k in row if k!='record_type'})
    with (OUT/'reported_results.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(reported)
    for name in ('PROTOCOL.md','split_manifest.json','analysis.py'):
        path=OUT/name;sources[path.relative_to(ROOT).as_posix()]=digest(path)
    (OUT/'analysis_sources.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
    (OUT/'primary_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
