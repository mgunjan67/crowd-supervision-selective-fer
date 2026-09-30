"""Independent arithmetic audit; does not import the follow-up analysis module."""
import argparse,csv,hashlib,json
from datetime import datetime
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
SEEDS=(17,42,89);CONDITIONS=('hard','soft','uniform','tie_hard')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def same(row,key,value):
    if value is None:assert row[key]=='',(key,row[key])
    else:assert np.isclose(float(row[key]),value,rtol=1e-9,atol=1e-10),(key,row[key],value)

def values(a):
    q=a['q'];pred=a['probabilities'].argmax(1)
    unsupported=q[:,7:].sum(1);w=q[:,:7].sum(1);best=q[:,:7].max(1)
    chosen=q[np.arange(len(q)),pred]
    return {'risk':1-chosen,'unsupported':unsupported,'supported_ambiguity':w-best,'excess':best-chosen,
        'floor':1-best,'supported_mass':w,
        'distance':((np.column_stack([a['probabilities'].astype(float),np.zeros((len(q),3))])-q)**2).sum(1),
        'supported_only_risk':np.divide(w-chosen,w,out=np.full(len(q),np.nan),where=w>0)}

def check_summary(row,a,keep):
    same(row,'population_n',len(keep));same(row,'accepted_n',keep.sum())
    same(row,'coverage',keep.mean() if len(keep) else None)
    for key,v in values(a).items():
        mask=keep&np.isfinite(v)
        same(row,key,float(v[mask].mean()) if mask.any() else None)
        if key=='supported_only_risk':same(row,'positive_mass_accepted_n',mask.sum())

def threshold(a,target,policy):
    score=a['probabilities'].max(1);risk=values(a)['risk']
    grid=np.unique(score) if policy=='empirical' else np.arange(101)/100
    # First qualifying threshold in ascending order has greatest coverage.
    for t in grid:
        keep=score>=t;n=keep.sum()
        if n<100:continue
        mean=risk[keep].mean()
        upper=mean if policy=='empirical' else min(1,mean+np.sqrt(np.log(2020)/(2*n)))
        if upper<=target+1e-12:return float(t),None if policy=='empirical' else float(upper)
    return None,None

def manual_f1(y,p):
    counts=np.zeros((7,7),int);np.add.at(counts,(y,p),1)
    denom=counts.sum(0)+counts.sum(1)
    f=np.divide(2*counts.diagonal(),denom,out=np.zeros(7),where=denom>0)
    return float(f@counts.sum(1)/counts.sum()),float(f.mean())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--skip-weights',action='store_true');args=parser.parse_args()
    with (OUT/'reported_results.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    raw_source=json.loads((OUT/'raw_source_audit.json').read_text())
    assert raw_source['status']=='passed' and raw_source['rows_checked']==35887
    assert raw_source['audit_code_sha256']==sha(OUT/'verify_raw_source.py')
    assert raw_source['archive_sha256']==json.loads((ROOT/'evidence/dataset_audit.json').read_text())['archive_sha256']
    split=json.loads((OUT/'split_manifest.json').read_text())
    with (ROOT/'data/ferplus_annotations/fer2013new.csv').open(newline='',encoding='utf-8-sig') as f:annotations=list(csv.DictReader(f))
    with (ROOT/'evidence/dataset_manifest.csv').open(newline='') as f:original=list(csv.DictReader(f))
    vote_names=('anger','disgust','fear','happiness','neutral','sadness','surprise','contempt','unknown','NF')
    classes=('angry','disgust','fear','happy','neutral','sad','surprise')
    q=np.array([[int(r[k]) for k in vote_names] for r in annotations],float);q/=q.sum(1,keepdims=True)
    labels=np.array([classes.index(r['class']) for r in original]);hashes=np.array([r['pixel_sha256'] for r in original])
    roles=[set(hashes[split['roles'][role]]) for role in ('train','selection','calibration','test')]
    assert all(not a&b for i,a in enumerate(roles) for b in roles[i+1:])
    for path,digest in json.loads((OUT/'freeze.json').read_text())['sha256'].items():assert sha(ROOT/path)==digest,path
    cache={};orders={};risk_cache={}
    frozen_at=datetime.fromisoformat(json.loads((OUT/'evaluation_freeze.json').read_text())['frozen_utc'])
    configurations={};old_replays=0;new_selection_replays=0
    for c in CONDITIONS:
        for s in SEEDS:
            run_dir=(ROOT/'final_experiment' if c in ('hard','soft') else OUT)/'runs'/f'{c}_seed{s}'
            completion=json.loads((run_dir/'training_complete.json').read_text())
            assert datetime.fromisoformat(completion['finished_utc'])<=frozen_at
            configurations[c,s]=json.loads((run_dir/'configuration.json').read_text())
            logs=[json.loads((run_dir/f'epoch-{i:02d}.json').read_text()) for i in range(1,13)]
            winner=min(logs,key=lambda r:(r['selection_distance'],r['epoch']))
            assert winner['epoch']==completion['selected_epoch']
            assert winner['selection_distance']==completion['selection_value']
            for state in ('selected','epoch12'):
                directory=OUT/'predictions'/f'{c}_seed{s}_{state}'
                report=json.loads((directory/'evaluation.json').read_text())
                assert datetime.fromisoformat(report['started_utc'])>=frozen_at
                assert report['epoch']==(completion['selected_epoch'] if state=='selected' else 12)
                if state=='selected':assert report['source_sha256']==completion['checkpoint_sha256']
                if not args.skip_weights:assert sha(OUT/'weights'/f'{c}_seed{s}_{state}.pt')==report['weight_sha256']
                for role in ('selection','calibration','test_full'):
                    path=directory/f'{role}.npz';assert sha(path)==report['roles'][role]['sha256']
                    if state=='selected' and (c in ('hard','soft') or role=='selection'):
                        assert report['roles'][role]['prior_reference_bitwise_verified']
                        if c in ('hard','soft'):old_replays+=1
                        else:new_selection_replays+=1
                    with np.load(path,allow_pickle=False) as f:a={k:f[k] for k in f.files}
                    assert a['rows'].tolist()==split['roles'][role]
                    np.testing.assert_array_equal(a['q'],q[a['rows']]);np.testing.assert_array_equal(a['labels'],labels[a['rows']])
                    np.testing.assert_array_equal(a['hashes'],hashes[a['rows']])
                    assert np.isfinite(a['probabilities']).all() and np.all(a['probabilities']>=0)
                    np.testing.assert_allclose(a['probabilities'].sum(1),1,atol=1e-6)
                    if role=='selection' and state=='selected':
                        np.testing.assert_allclose(values(a)['distance'].mean(),completion['selection_value'],rtol=0,atol=1e-12)
                    if role=='test_full':
                        keep=np.isin(a['rows'],split['roles']['test']);a={k:v[keep] for k,v in a.items()}
                        role='test'
                    cache[c,s,state,role]=a
                a=cache[c,s,state,'test'];key=c,s,state
                orders[key]=np.lexsort((a['rows'],-a['probabilities'].max(1)))
                risk_cache[key]=values(a)['risk']
    matched_keys=('seed','epochs','batch_size','workers','variant','optimizer','learning_rate','weight_decay','precision',
        'scheduler','selection_metric','preprocessing','class_names','initial_state_sha256','torch','torchvision','gpu','train_n','selection_n','augmentation')
    for s in SEEDS:
        for c in CONDITIONS:
            assert all(configurations[c,s][k]==configurations['hard',s][k] for k in matched_keys),(c,s)
    assert old_replays==18 and new_selection_replays==6
    for row in rows:
        kind=row['record_type']
        if kind in ('contrast','seed_contrast'):continue
        c,s,state=row['condition'],int(row['seed']),row['state'];a=cache[c,s,state,'test']
        n=len(a['rows']);keep=np.ones(n,bool)
        if kind=='checkpoint':
            source=json.loads((OUT/'predictions'/f'{c}_seed{s}_{state}'/'evaluation.json').read_text())
            same(row,'epoch',source['epoch']);assert row['source_checkpoint_sha256']==source['source_sha256']
            assert row['released_weight_sha256']==source['weight_sha256'];continue
        elif kind=='common':
            keep[:]=False;keep[orders[c,s,state][:int(np.ceil(float(row['requested_coverage'])*n))]]=True
        elif kind=='stratum':
            w=a['q'][:,:7].sum(1);out=a['q'][:,7:].sum(1)
            masks={'zero':w==0,'partial':(w>0)&(out>0),'full':out==0};mask=masks[row['stratum']]
            if row['scope']=='global80':
                keep[:]=False;keep[orders[c,s,state][:int(np.ceil(.8*n))]]=True;keep=keep[mask]
            a={k:v[mask] for k,v in a.items()}
            if row['scope']=='all':keep=np.ones(mask.sum(),bool)
            elif row['scope']=='within80':
                keep=np.zeros(mask.sum(),bool);order=np.lexsort((a['rows'],-a['probabilities'].max(1)))
                keep[order[:int(np.ceil(.8*len(order)))]]=True
        elif kind in ('population','policy'):
            a=cache[c,s,state,row['population']];keep=np.ones(len(a['rows']),bool)
            if kind=='policy':
                cutoff,upper=threshold(cache[c,s,state,'calibration'],float(row['target']),row['policy'])
                same(row,'threshold',cutoff);same(row,'calibration_upper',upper)
                keep=a['probabilities'].max(1)>=cutoff if cutoff is not None else np.zeros(len(a['rows']),bool)
        elif kind=='classification':
            pred=a['probabilities'].argmax(1);major=a['q'][:,:7].max(1)>.5
            weighted,macro=manual_f1(a['labels'],pred)
            same(row,'original_weighted_f1',weighted);same(row,'original_macro_f1',macro)
            same(row,'majority_n',major.sum());same(row,'majority_weighted_f1',manual_f1(a['q'][major,:7].argmax(1),pred[major])[0])
            same(row,'reference_disagreement',(a['labels'][major]!=a['q'][major,:7].argmax(1)).mean())
            continue
        elif kind=='curve':
            k=int(row['accepted_n']);chosen=orders[c,s,state][:k]
            same(row,'coverage',k/n)
            for key in ('risk','unsupported','supported_ambiguity','excess'):same(row,key,values(a)[key][chosen].mean())
            continue
        elif kind=='calibration_resample':continue # Exhaustively checked separately below.
        else:raise ValueError(kind)
        check_summary(row,a,keep)
    resamples=[r for r in rows if r['record_type']=='calibration_resample']
    for c in CONDITIONS:
        for s in SEEDS:
            a=cache[c,s,'selected','calibration'];rng=np.random.default_rng(20260922);unique=np.unique(a['hashes'])
            selected=[r for r in resamples if r['condition']==c and int(r['seed'])==s]
            for repeat in range(100):
                clusters=rng.permutation(unique);fit=np.isin(a['hashes'],clusters[:len(clusters)//2]);hold=~fit
                for row in [r for r in selected if int(r['repeat'])==repeat]:
                    sub={k:v[fit] for k,v in a.items()};t,_=threshold(sub,float(row['target']),'empirical');same(row,'threshold',t)
                    fk=fit&(a['probabilities'].max(1)>=t) if t is not None else np.zeros(len(fit),bool)
                    hk=hold&(a['probabilities'].max(1)>=t) if t is not None else np.zeros(len(fit),bool)
                    loss=values(a)['risk'];fr=loss[fk].mean() if fk.any() else None;hr=loss[hk].mean() if hk.any() else None
                    for key,value in [('fit_n',fk.sum()),('holdout_n',hk.sum()),('fit_risk',fr),('holdout_risk',hr),
                        ('holdout_coverage',hk.sum()/hold.sum()),('optimism_gap',hr-fr if hr is not None and fr is not None else None)]:same(row,key,value)
    # Independent expansion-based bootstrap: one shared cluster resample for all 24 models.
    reference=cache['hard',17,'selected','test'];names,group=np.unique(reference['hashes'],return_inverse=True)
    rng=np.random.default_rng(20260921);indices=np.arange(len(group));replicates={key:[] for key in orders}
    for repeat in range(5000):
        counts=np.bincount(rng.integers(len(names),size=len(names)),minlength=len(names))
        multiplicity=counts[group];k=int(np.ceil(.8*multiplicity.sum()))
        for key,order in orders.items():
            expanded=np.repeat(order,multiplicity[order])[:k]
            replicates[key].append(risk_cache[key][expanded].mean())
    for row in [r for r in rows if r['record_type'] in ('contrast','seed_contrast')]:
        l,r,state=row['left'],row['right'],row['state'];deltas=[]
        for s in SEEDS:
            k=int(np.ceil(.8*len(reference['rows'])))
            deltas.append(100*(risk_cache[l,s,state][orders[l,s,state][:k]].mean()-risk_cache[r,s,state][orders[r,s,state][:k]].mean()))
        if row['record_type']=='seed_contrast':same(row,'difference_pp',deltas[SEEDS.index(int(row['seed']))]);continue
        sample=100*np.mean([np.array(replicates[l,s,state])-replicates[r,s,state] for s in SEEDS],axis=0)
        lo,hi=np.quantile(sample,[.025,.975]);same(row,'point_difference_pp',np.mean(deltas));same(row,'lower_pp',lo);same(row,'upper_pp',hi)
    for path,digest in json.loads((OUT/'analysis_sources.json').read_text()).items():assert sha(ROOT/path)==digest,path
    report={'status':'passed','prediction_files':72,'inference_checkpoints_verified':0 if args.skip_weights else 24,'ledger_rows':len(rows),
        'row_vote_label_hash_mapping':'independently verified','bootstrap_repetitions':5000,
        'bootstrap_method':'explicitly repeated confidence-ordered rows','resampling_rows_checked':len(resamples),
        'source_freeze_verified':True,'ledger_sha256':sha(OUT/'reported_results.csv'),'audit_sha256':sha(Path(__file__))}
    report.update(matched_configuration_verified=True,old_selected_bitwise_replays=old_replays,
        new_selected_selection_replays=new_selection_replays,all_training_complete_before_new_inference=True)
    (OUT/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
