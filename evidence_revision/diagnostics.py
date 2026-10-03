"""Prespecified existing-prediction diagnostics; no new training results assumed."""
import csv,importlib.util,json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
spec=importlib.util.spec_from_file_location('retained_analysis',ROOT/'reviewer_followup/analysis.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
SEEDS=(17,42,89)

def records():
    result=[]
    for state in ('selected','epoch12'):
        for seed in SEEDS:
            a={c:old.load(c,seed,state,'test_full') for c in ('uniform','soft')}
            assert np.array_equal(a['uniform']['rows'],a['soft']['rows'])
            assert np.array_equal(a['uniform']['q'],a['soft']['q'])
            keep={c:old.old.common_keep(a[c],.8) for c in a}
            for predictor in a:
                for selector in a:
                    result.append(dict(record_type='cross_evaluation',state=state,seed=seed,predictor=predictor,
                        selector=selector,**old.summary(a[predictor],keep[selector])))
            intersect=int((keep['uniform']&keep['soft']).sum());union=int((keep['uniform']|keep['soft']).sum())
            result.append(dict(record_type='accepted_overlap',state=state,seed=seed,intersection_n=intersect,
                union_n=union,jaccard=intersect/union,accepted_n=int(keep['soft'].sum()),overlap_fraction=intersect/keep['soft'].sum()))
            if state=='selected':
                for n in range(1,len(a['soft']['rows'])+1):
                    risks={}
                    for c in a:
                        order=old.old.rank(a[c]);risk=old.parts(a[c])['risk']
                        risks[c]=float(risk[order[:n]].mean())
                    result.append(dict(record_type='risk_difference_curve',seed=seed,state=state,accepted_n=n,
                        coverage=n/len(a['soft']['rows']),difference_pp=100*(risks['soft']-risks['uniform'])))
        for condition in ('hard','tie_hard','uniform','soft'):
            for seed in SEEDS:
                a=old.load(condition,seed,state,'test_full');p=a['probabilities'].argmax(1);q=a['q'];eligible=q[:,:7].max(1)>.5
                for subset,ref,y,pred in [('all','original',a['labels'],p),('majority_subset','original',a['labels'][eligible],p[eligible]),
                                         ('majority_subset','crowd_majority',q[eligible,:7].argmax(1),p[eligible])]:
                    metrics=old.old.classification_metrics(y,pred,old.old.CLASSES)
                    result.append(dict(record_type='same_population_f1',condition=condition,seed=seed,state=state,
                        population=subset,reference=ref,n=len(y),weighted_f1=metrics['weighted_f1'],macro_f1=metrics['macro_f1']))
                if condition=='soft' and seed==17 and state=='selected':
                    mat=np.zeros((7,7),int);np.add.at(mat,(a['labels'][eligible],q[eligible,:7].argmax(1)),1)
                    for i in range(7):
                        for j in range(7):result.append(dict(record_type='reference_confusion',original_class=old.old.CLASSES[i],crowd_class=old.old.CLASSES[j],count=int(mat[i,j])))
                if state!='selected':continue
                accepted=old.old.common_keep(a,.8);w=q[:,:7].sum(1)
                for name,mask in [('full',q[:,7:].sum(1)==0),('partial',(w>0)&(q[:,7:].sum(1)>0)),('zero',w==0)]:
                    result.append(dict(record_type='retention',condition=condition,seed=seed,state=state,stratum=name,
                        population_n=int(mask.sum()),accepted_n=int((mask&accepted).sum())))
                for role in ('calibration','test_full'):
                    aa=old.load(condition,seed,state,role);selected=old.old.common_keep(aa,.8)
                    for scope,mask in [('all',np.ones(len(aa['rows']),bool)),('global80',selected)]:
                        for index,category in enumerate(('contempt','unknown','not_a_face'),7):
                            result.append(dict(record_type='unsupported_categories',condition=condition,seed=seed,state=state,
                                population='test' if role=='test_full' else 'calibration',scope=scope,category=category,
                                n=int(mask.sum()),mass=float(aa['q'][mask,index].mean())))
                run=(ROOT/'final_experiment' if condition in ('hard','soft') else ROOT/'reviewer_followup')/'runs'/f'{condition}_seed{seed}'
                done=json.loads((run/'training_complete.json').read_text())
                result.append(dict(record_type='selected_epoch',condition=condition,seed=seed,state=state,epoch=done['selected_epoch']))
                for epoch in range(1,13):
                    log=json.loads((run/f'epoch-{epoch:02d}.json').read_text())
                    result.append(dict(record_type='learning_curve',condition=condition,seed=seed,epoch=epoch,
                        train_loss=log['train_loss'],selection_distance=log['selection_distance']))
    return result

def main():
    rr=records();fields=['record_type']+sorted({k for r in rr for k in r if k!='record_type'})
    with (OUT/'existing_diagnostics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rr)
    summary={}
    for state in ('selected','epoch12'):
        cross=[r for r in rr if r['record_type']=='cross_evaluation' and r['state']==state]
        summary[state]={f'{predictor}_on_{selector}':float(np.mean([r['risk'] for r in cross if r['predictor']==predictor and r['selector']==selector]))*100
            for predictor in ('uniform','soft') for selector in ('uniform','soft')}
    f1=[r for r in rr if r['record_type']=='same_population_f1' and r['condition']=='soft' and r['state']=='selected']
    summary['soft_f1']={f'{pop}_{ref}':float(np.mean([r['weighted_f1'] for r in f1 if r['population']==pop and r['reference']==ref]))
        for pop,ref in [('all','original'),('majority_subset','original'),('majority_subset','crowd_majority')]}
    (OUT/'existing_diagnostics_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
