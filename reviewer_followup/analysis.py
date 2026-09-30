"""Frozen follow-up endpoints; the sole numeric ledger for the revised paper."""
import csv, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
spec=importlib.util.spec_from_file_location('old_endpoint_analysis',ROOT/'final_experiment/analysis.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
SEEDS=(17,42,89);CONDITIONS=('hard','soft','uniform','tie_hard')

def load(c,s,state,role):
    with np.load(OUT/'predictions'/f'{c}_seed{s}_{state}'/f'{role}.npz',allow_pickle=False) as f:
        a={k:f[k] for k in f.files}
    a['probabilities']=a['probabilities'].astype(float)
    if role=='test_full':
        ids=json.loads((OUT/'split_manifest.json').read_text())['roles']['test']
        mask=np.isin(a['rows'],ids);a={k:v[mask] for k,v in a.items()}
    return a

def parts(a):
    p,q=a['probabilities'],a['q'];w=q[:,:7].sum(1);m=q[:,:7].max(1)
    chosen=q[np.arange(len(q)),p.argmax(1)]
    risk=1-chosen
    result={'risk':risk,'unsupported':1-w,'supported_ambiguity':w-m,'excess':m-chosen,
        'floor':1-m,'supported_mass':w,'distance':((np.pad(p,((0,0),(0,3)))-q)**2).sum(1),
        'supported_only_risk':np.divide(w-chosen,w,out=np.full_like(w,np.nan),where=w>0)}
    assert np.allclose(risk,result['unsupported']+result['supported_ambiguity']+result['excess'])
    return result

def summary(a,keep):
    result={'population_n':len(keep),'accepted_n':int(keep.sum()),'coverage':float(keep.mean()) if len(keep) else None}
    for key,v in parts(a).items():
        valid=keep&np.isfinite(v)
        result[key]=float(v[valid].mean()) if valid.any() else None
        if key=='supported_only_risk':result['positive_mass_accepted_n']=int(valid.sum())
    return result

def cut(a,mask):return {k:v[mask] for k,v in a.items()}

def conservative(score,risk,target):
    candidates=[]
    for threshold in np.linspace(0,1,101):
        keep=score>=threshold;n=int(keep.sum())
        if n<100:continue
        upper=min(1,float(risk[keep].mean())+np.sqrt(np.log(101/.05)/(2*n)))
        if upper<=target+1e-12:candidates.append((n,-float(threshold),upper))
    if not candidates:return None,None
    n,negative,upper=max(candidates)
    return -negative,upper

def bootstrap(arrays,left,right,reps=5000):
    # Reuse the old validated pixel-cluster algorithm with explicit label remapping.
    mapped={('hard',s):arrays[right,s] for s in SEEDS}
    mapped.update({('soft',s):arrays[left,s] for s in SEEDS})
    result=old.bootstrap_primary(mapped,repetitions=reps)
    result['endpoint']=f'{left} minus {right} at 80% coverage'
    return result

def calibration_resampling(a,c,s):
    rng=np.random.default_rng(20260922)
    unique=np.unique(a['hashes']);rows=[];v=parts(a)['risk'];score=a['probabilities'].max(1)
    for repeat in range(100):
        clusters=rng.permutation(unique);fit=np.isin(a['hashes'],clusters[:len(clusters)//2]);hold=~fit
        for target in (.1,.2,.3):
            threshold=old.threshold(score[fit],v[fit],target)
            fit_keep=fit&(score>=threshold) if threshold is not None else np.zeros(len(fit),bool)
            hold_keep=hold&(score>=threshold) if threshold is not None else np.zeros(len(fit),bool)
            f=float(v[fit_keep].mean()) if fit_keep.any() else None
            h=float(v[hold_keep].mean()) if hold_keep.any() else None
            rows.append(dict(record_type='calibration_resample',condition=c,seed=s,state='selected',repeat=repeat,
                target=target,threshold=threshold,fit_n=int(fit_keep.sum()),holdout_n=int(hold_keep.sum()),
                fit_risk=f,holdout_risk=h,holdout_coverage=float(hold_keep.sum()/hold.sum()),
                optimism_gap=h-f if h is not None and f is not None else None))
    return rows

def compute():
    assert (OUT/'evaluation_complete.json').exists()
    records=[];contrasts=[]
    for state in ('selected','epoch12'):
        arrays={}
        for c in CONDITIONS:
            for s in SEEDS:
                a=load(c,s,state,'test_full');cal=load(c,s,state,'calibration');arrays[c,s]=a
                base={'condition':c,'seed':s,'state':state}
                checkpoint=json.loads((OUT/'predictions'/f'{c}_seed{s}_{state}'/'evaluation.json').read_text())
                records.append(dict(record_type='checkpoint',**base,epoch=checkpoint['epoch'],
                    source_checkpoint_sha256=checkpoint['source_sha256'],released_weight_sha256=checkpoint['weight_sha256']))
                for coverage in (.2,.5,.8,1.):
                    records.append(dict(record_type='common',requested_coverage=coverage,**base,
                                        **summary(a,old.common_keep(a,coverage))))
                p,q=a['probabilities'],a['q'];pred=p.argmax(1);eligible=q[:,:7].max(1)>.5
                orig=old.classification_metrics(a['labels'],pred,old.CLASSES)
                crowd=old.classification_metrics(q[eligible,:7].argmax(1),pred[eligible],old.CLASSES)
                records.append(dict(record_type='classification',**base,population_n=len(q),majority_n=int(eligible.sum()),
                    original_weighted_f1=orig['weighted_f1'],original_macro_f1=orig['macro_f1'],
                    majority_weighted_f1=crowd['weighted_f1'],
                    reference_disagreement=float((a['labels'][eligible]!=q[eligible,:7].argmax(1)).mean())))
                if state!='selected':continue
                w=q[:,:7].sum(1);global_keep=old.common_keep(a,.8)
                # Integer zero mass and full mass use original ten-category zeros to avoid float equality drift.
                strata={'zero':w==0,'partial':(w>0)&(q[:,7:].sum(1)>0),'full':q[:,7:].sum(1)==0}
                assert sum(mask.astype(int) for mask in strata.values()).min()==1
                assert sum(mask.astype(int) for mask in strata.values()).max()==1
                for name,mask in strata.items():
                    for scope in ('global80','within80','all'):
                        subset=cut(a,mask)
                        keep=global_keep[mask] if scope=='global80' else old.common_keep(subset,.8) if scope=='within80' and mask.any() else np.ones(int(mask.sum()),bool)
                        records.append(dict(record_type='stratum',stratum=name,scope=scope,**base,**summary(subset,keep)))
                for population,aa in (('calibration',cal),('test',a)):
                    records.append(dict(record_type='population',population=population,**base,**summary(aa,np.ones(len(aa['rows']),bool))))
                for policy in ('empirical','hoeffding_grid'):
                    for target in (.1,.2,.3):
                        score=cal['probabilities'].max(1);loss=parts(cal)['risk']
                        if policy=='empirical':threshold=old.threshold(score,loss,target);upper=None
                        else:threshold,upper=conservative(score,loss,target)
                        for population,aa in (('calibration',cal),('test',a)):
                            keep=aa['probabilities'].max(1)>=threshold if threshold is not None else np.zeros(len(aa['rows']),bool)
                            records.append(dict(record_type='policy',policy=policy,target=target,threshold=threshold,
                                calibration_upper=upper,population=population,**base,**summary(aa,keep)))
                records.extend(calibration_resampling(cal,c,s))
                order=old.rank(a);v=parts(a)
                cumulative={key:np.cumsum(v[key][order])/np.arange(1,len(order)+1)
                            for key in ('risk','unsupported','supported_ambiguity','excess')}
                for i in range(len(order)):
                    records.append(dict(record_type='curve',**base,accepted_n=i+1,coverage=(i+1)/len(order),
                        **{key:float(value[i]) for key,value in cumulative.items()}))
        for left,right in (('soft','uniform'),('soft','hard'),('tie_hard','hard'),('soft','tie_hard')):
            result=bootstrap(arrays,left,right)
            contrasts.append(dict(state=state,left=left,right=right,**result))
            records.append(dict(record_type='contrast',state=state,left=left,right=right,
                **{k:v for k,v in result.items() if not isinstance(v,list)}))
            for s,delta in zip(SEEDS,result['seed_differences_pp']):
                records.append(dict(record_type='seed_contrast',state=state,left=left,right=right,seed=s,difference_pp=delta))
    return records,contrasts

def main():
    records,contrasts=compute()
    fields=['record_type']+sorted({k for row in records for k in row if k!='record_type'})
    with (OUT/'reported_results.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)
    (OUT/'contrasts.json').write_text(json.dumps(contrasts,indent=2),encoding='utf-8')
    paths=list((OUT/'predictions').glob('*/*.npz'))+list((OUT/'predictions').glob('*/*.json'))+[OUT/'analysis.py',OUT/'PROTOCOL.md',OUT/'freeze.json']
    (OUT/'analysis_sources.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():old.digest(p) for p in paths},indent=2),encoding='utf-8')
    print(json.dumps(contrasts,indent=2))

if __name__=='__main__':main()
