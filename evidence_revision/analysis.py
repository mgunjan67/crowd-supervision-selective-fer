"""Merge retained results and all planned revision diagnostics into one ledger."""
import csv,json
from pathlib import Path
import numpy as np
from diagnostics import old,records,SEEDS
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent

def load_new(seed,state,role='test_full'):
    with np.load(OUT/'predictions'/f'tie_uniform_seed{seed}_{state}'/f'{role}.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
    a['probabilities']=a['probabilities'].astype(float)
    if role=='test_full':
        ids=json.loads((OUT/'split_manifest.json').read_text())['roles']['test'];mask=np.isin(a['rows'],ids);a={k:v[mask] for k,v in a.items()}
    return a

def main():
    assert (OUT/'evaluation_complete.json').exists()
    with (ROOT/'reviewer_followup/reported_results.csv').open(newline='',encoding='utf-8') as f:ledger=list(csv.DictReader(f))
    ledger.extend(records());contrasts=[]
    for state in ('selected','epoch12'):
        arrays={}
        for seed in SEEDS:
            a=load_new(seed,state);arrays['tie_uniform',seed]=a;arrays['soft',seed]=old.load('soft',seed,state,'test_full')
            assert np.array_equal(a['rows'],arrays['soft',seed]['rows']) and np.array_equal(a['q'],arrays['soft',seed]['q'])
            base={'condition':'tie_uniform','seed':seed,'state':state}
            info=json.loads((OUT/'predictions'/f'tie_uniform_seed{seed}_{state}'/'evaluation.json').read_text())
            ledger.append(dict(record_type='checkpoint',**base,epoch=info['epoch'],source_checkpoint_sha256=info['source_sha256'],released_weight_sha256=info['weight_sha256']))
            for coverage in (.2,.5,.8,1.):ledger.append(dict(record_type='common',requested_coverage=coverage,**base,**old.summary(a,old.old.common_keep(a,coverage))))
            p=a['probabilities'].argmax(1);q=a['q'];eligible=q[:,:7].max(1)>.5
            for subset,ref,y,pred in [('all','original',a['labels'],p),('majority_subset','original',a['labels'][eligible],p[eligible]),('majority_subset','crowd_majority',q[eligible,:7].argmax(1),p[eligible])]:
                metrics=old.old.classification_metrics(y,pred,old.old.CLASSES)
                ledger.append(dict(record_type='same_population_f1',**base,population=subset,reference=ref,n=len(y),weighted_f1=metrics['weighted_f1'],macro_f1=metrics['macro_f1']))
            if state=='selected':
                w=a['q'][:,:7].sum(1);keep=old.old.common_keep(a,.8)
                for stratum,mask in [('full',a['q'][:,7:].sum(1)==0),('partial',(w>0)&(a['q'][:,7:].sum(1)>0)),('zero',w==0)]:
                    ledger.append(dict(record_type='retention',stratum=stratum,**base,population_n=int(mask.sum()),accepted_n=int((keep&mask).sum())))
                run=OUT/'runs'/f'tie_uniform_seed{seed}'
                done=json.loads((run/'training_complete.json').read_text())
                ledger.append(dict(record_type='selected_epoch',**base,epoch=done['selected_epoch']))
                for epoch in range(1,13):
                    log=json.loads((run/f'epoch-{epoch:02d}.json').read_text())
                    ledger.append(dict(record_type='learning_curve',condition='tie_uniform',seed=seed,epoch=epoch,train_loss=log['train_loss'],selection_distance=log['selection_distance']))
                for role in ('calibration','test_full'):
                    aa=load_new(seed,state,role);selected=old.old.common_keep(aa,.8)
                    for scope,mask in [('all',np.ones(len(aa['rows']),bool)),('global80',selected)]:
                        for index,category in enumerate(('contempt','unknown','not_a_face'),7):
                            ledger.append(dict(record_type='unsupported_categories',**base,population='test' if role=='test_full' else 'calibration',scope=scope,category=category,n=int(mask.sum()),mass=float(aa['q'][mask,index].mean())))
        contrast=old.bootstrap(arrays,'soft','tie_uniform',5000);contrasts.append(dict(state=state,**contrast))
        ledger.append(dict(record_type='contrast',left='soft',right='tie_uniform',state=state,**{k:v for k,v in contrast.items() if not isinstance(v,list)}))
        for seed,delta in zip(SEEDS,contrast['seed_differences_pp']):ledger.append(dict(record_type='seed_contrast',left='soft',right='tie_uniform',state=state,seed=seed,difference_pp=delta))
    fields=['record_type']+sorted({k for row in ledger for k in row if k!='record_type'})
    with (OUT/'reported_results.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(ledger)
    (OUT/'new_contrasts.json').write_text(json.dumps(contrasts,indent=2));print(json.dumps(contrasts,indent=2))

if __name__=='__main__':main()
