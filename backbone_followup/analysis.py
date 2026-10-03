"""Extend the immutable Swin ledger with labelled ResNet robustness results."""
import csv,importlib.util,json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
spec=importlib.util.spec_from_file_location('retained_backbone_analysis',ROOT/'reviewer_followup/analysis.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
SEEDS=(17,42,89);CONDITIONS=('soft','uniform','tie_uniform')
def load(c,s,state,role='test_full'):
    with np.load(OUT/'predictions'/f'{c}_seed{s}_{state}'/f'{role}.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
    a['probabilities']=a['probabilities'].astype(float)
    if role=='test_full':
        ids=json.loads((OUT/'split_manifest.json').read_text())['roles']['test'];mask=np.isin(a['rows'],ids);a={k:v[mask] for k,v in a.items()}
    return a
def main():
    assert (OUT/'evaluation_complete.json').exists()
    with (ROOT/'evidence_revision/reported_results.csv').open(newline='') as f:retained=list(csv.DictReader(f))
    for r in retained:r['backbone']='swin_t'
    records=[];contrasts=[]
    for state in ('selected','epoch12'):
        arrays={(c,s):load(c,s,state) for c in CONDITIONS for s in SEEDS}
        for (c,s),a in arrays.items():
            b={'backbone':'resnet18','condition':c,'seed':s,'state':state}
            info=json.loads((OUT/'predictions'/f'{c}_seed{s}_{state}'/'evaluation.json').read_text())
            records.append(dict(record_type='checkpoint',**b,epoch=info['epoch'],source_checkpoint_sha256=info['source_sha256'],released_weight_sha256=info['weight_sha256']))
            for coverage in (.2,.5,.8,1.):records.append(dict(record_type='common',**b,requested_coverage=coverage,**old.summary(a,old.old.common_keep(a,coverage))))
            p=a['probabilities'].argmax(1);q=a['q'];eligible=q[:,:7].max(1)>.5
            for pop,ref,y,pred in [('all','original',a['labels'],p),('majority_subset','original',a['labels'][eligible],p[eligible]),('majority_subset','crowd_majority',q[eligible,:7].argmax(1),p[eligible])]:
                metrics=old.old.classification_metrics(y,pred,old.old.CLASSES)
                records.append(dict(record_type='same_population_f1',**b,population=pop,reference=ref,n=len(y),weighted_f1=metrics['weighted_f1'],macro_f1=metrics['macro_f1']))
            if state=='selected':
                keep=old.old.common_keep(a,.8);w=q[:,:7].sum(1)
                for stratum,mask in [('full',q[:,7:].sum(1)==0),('partial',(w>0)&(q[:,7:].sum(1)>0)),('zero',w==0)]:records.append(dict(record_type='retention',**b,stratum=stratum,population_n=int(mask.sum()),accepted_n=int((mask&keep).sum())))
                for role in ('calibration','test_full'):
                    aa=load(c,s,state,role);selected=old.old.common_keep(aa,.8)
                    for scope,mask in [('all',np.ones(len(aa['rows']),bool)),('global80',selected)]:
                        for i,cat in enumerate(('contempt','unknown','not_a_face'),7):records.append(dict(record_type='unsupported_categories',**b,population='test' if role=='test_full' else 'calibration',scope=scope,category=cat,n=int(mask.sum()),mass=float(aa['q'][mask,i].mean())))
                for epoch in range(1,13):
                    log=json.loads((OUT/'runs'/f'{c}_seed{s}'/f'epoch-{epoch:02d}.json').read_text())
                    records.append(dict(record_type='learning_curve',backbone='resnet18',condition=c,seed=s,epoch=epoch,train_loss=log['train_loss'],selection_distance=log['selection_distance']))
                records.append(dict(record_type='selected_epoch',**b,epoch=info['epoch']))
        for right in ('uniform','tie_uniform'):
            result=old.bootstrap(arrays,'soft',right,5000);contrasts.append(dict(backbone='resnet18',state=state,left='soft',right=right,**result))
            records.append(dict(record_type='contrast',backbone='resnet18',state=state,left='soft',right=right,**{k:v for k,v in result.items() if not isinstance(v,list)}))
            for seed,delta in zip(SEEDS,result['seed_differences_pp']):records.append(dict(record_type='seed_contrast',backbone='resnet18',state=state,left='soft',right=right,seed=seed,difference_pp=delta))
            for seed in SEEDS:
                aa={c:arrays[c,seed] for c in ('soft',right)};keep={c:old.old.common_keep(a,.8) for c,a in aa.items()}
                for predictor in aa:
                    for selector in aa:records.append(dict(record_type='cross_evaluation',backbone='resnet18',state=state,seed=seed,comparison=right,predictor=predictor,selector=selector,**old.summary(aa[predictor],keep[selector])))
                inter=int((keep['soft']&keep[right]).sum());union=int((keep['soft']|keep[right]).sum())
                records.append(dict(record_type='accepted_overlap',backbone='resnet18',state=state,seed=seed,comparison=right,intersection_n=inter,union_n=union,jaccard=inter/union,accepted_n=int(keep['soft'].sum()),overlap_fraction=inter/keep['soft'].sum()))
                if state=='selected':
                    curves={c:np.cumsum(old.parts(a)['risk'][old.old.rank(a)])/np.arange(1,len(a['rows'])+1) for c,a in aa.items()}
                    for n,v in enumerate(100*(curves['soft']-curves[right]),1):records.append(dict(record_type='risk_difference_curve',backbone='resnet18',state=state,seed=seed,comparison=right,accepted_n=n,coverage=n/len(aa['soft']['rows']),difference_pp=float(v)))
    allrows=retained+records;fields=['record_type']+sorted({k for r in allrows for k in r if k!='record_type'})
    with (OUT/'reported_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(allrows)
    (OUT/'contrasts.json').write_text(json.dumps(contrasts,indent=2));print(json.dumps(contrasts,indent=2))
if __name__=='__main__':main()
