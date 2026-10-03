"""Independent arithmetic and contract checks; not external replication."""
import csv,hashlib,json,math
from pathlib import Path
import numpy as np,torch
from model import build_model
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def keep(a,f=.8):
    order=np.lexsort((a['rows'],-a['probabilities'].max(1)));m=np.zeros(len(order),bool);m[order[:math.ceil(f*len(order))]]=True;return m
def risk(a,m):return (1-a['q'][np.arange(len(m)),a['probabilities'].argmax(1)])[m].mean()
def f1(y,p):
    cm=np.zeros((7,7),int);np.add.at(cm,(y,p),1);support=cm.sum(1);den=support+cm.sum(0);scores=np.divide(2*np.diag(cm),den,out=np.zeros(7,float),where=den>0)
    return float(np.dot(scores,support)/len(y)),float(scores.mean())
def bootstrap_explicit(arrays,state,right,reps=5000):
    # Independently expand sampled pixels instead of the retained implementation's
    # score-ordered fractional multiplicity accumulation.
    reference=arrays['soft',17,state,'test_full'];names,groups=np.unique(reference['hashes'],return_inverse=True)
    rng=np.random.default_rng(20260921);deltas=[];indices=np.arange(len(groups))
    scores={};losses={}
    for c in ('soft',right):
        for s in (17,42,89):
            a=arrays[c,s,state,'test_full'];scores[c,s]=a['probabilities'].max(1)
            losses[c,s]=1-a['q'][np.arange(len(groups)),a['probabilities'].argmax(1)]
    for _ in range(reps):
        draw=rng.integers(len(names),size=len(names));counts=np.bincount(draw,minlength=len(names))
        expanded=np.repeat(indices,counts[groups]);n=math.ceil(.8*len(expanded));pairs=[]
        for s in (17,42,89):
            risks={}
            for c in ('soft',right):
                order=np.lexsort((reference['rows'][expanded],-scores[c,s][expanded]));accepted=expanded[order[:n]]
                risks[c]=losses[c,s][accepted].mean()
            pairs.append(risks['soft']-risks[right])
        deltas.append(100*np.mean(pairs))
    return np.quantile(deltas,[.025,.975])
def main():
    assert (OUT/'evaluation_complete.json').exists()
    manifest=json.loads((OUT/'split_manifest.json').read_text());arrays={};weights=0;predictions=0;training_sources=0
    for infofile in (OUT/'predictions').glob('*/evaluation.json'):
        rec=json.loads(infofile.read_text());c,s,state=rec['condition'],rec['seed'],rec['state']
        weight=OUT/'weights'/f'{c}_seed{s}_{state}.pt';assert sha(weight)==rec['weight_sha256']
        source=ROOT/rec['source']
        if source.exists():
            assert sha(source)==rec['source_sha256'];training_sources+=1
        ck=torch.load(weight,map_location='cpu',weights_only=True);assert ck['variant']=='resnet18_gated' and ck['epoch']==rec['epoch']
        assert ck['class_names']==manifest['class_order'] and ck['seed']==s and ck['configuration']['condition']==c
        m=build_model();m.load_state_dict(ck['state_dict'],strict=True);weights+=1
        for role,record in rec['roles'].items():
            file=infofile.parent/f'{role}.npz';assert sha(file)==record['sha256']
            with np.load(file,allow_pickle=False) as z:a={k:z[k] for k in z.files}
            assert a['probabilities'].shape==(record['n'],7) and np.isfinite(a['probabilities']).all()
            assert np.allclose(a['probabilities'].sum(1),1,atol=2e-6)
            assert a['q'].shape==(record['n'],10) and np.allclose(a['q'].sum(1),1)
            assert a['rows'].tolist()==manifest['roles'][role]
            if role=='test_full':
                sel=np.isin(a['rows'],manifest['roles']['test']);a={k:v[sel] for k,v in a.items()}
            arrays[c,s,state,role]=a;predictions+=1
    assert weights==18 and predictions==54
    anchor=arrays['soft',17,'selected','test_full']
    for (c,s,state,role),a in arrays.items():
        reference=arrays['soft',17,'selected',role]
        for k in ('rows','q','hashes','labels'):assert np.array_equal(a[k],reference[k])
    rows=list(csv.DictReader((OUT/'reported_results.csv').open(newline='')));checked=0;curves={}
    retained=list(csv.DictReader((ROOT/'evidence_revision/reported_results.csv').open(newline='')))
    oldrows=[r for r in rows if r['backbone']=='swin_t']
    assert len(oldrows)==len(retained)
    for old,current in zip(retained,oldrows):
        assert all(current[k]==v for k,v in old.items()),'Retained Swin ledger changed'
    for row in rows:
        if row['backbone']!='resnet18':continue
        kind=row['record_type'];state=row['state'] or 'selected'
        if kind in ('common','same_population_f1','retention','unsupported_categories'):
            c,s=row['condition'],int(row['seed']);role='calibration' if kind=='unsupported_categories' and row['population']=='calibration' else 'test_full';a=arrays[c,s,state,role]
            if kind=='common':
                mask=keep(a,float(row['requested_coverage']));q=a['q'];p=a['probabilities'].astype(float);w=q[:,:7].sum(1);mx=q[:,:7].max(1);chosen=q[np.arange(len(q)),p.argmax(1)]
                vals={'risk':1-chosen,'unsupported':1-w,'supported_ambiguity':w-mx,'excess':mx-chosen,'floor':1-mx,'supported_mass':w,'distance':((np.pad(p,((0,0),(0,3)))-q)**2).sum(1)}
                assert int(row['accepted_n'])==mask.sum() and int(row['population_n'])==len(mask)
                for key,val in vals.items():assert np.isclose(val[mask].mean(),float(row[key]),atol=1e-12),key
            elif kind=='same_population_f1':
                mask=np.ones(len(a['rows']),bool) if row['population']=='all' else a['q'][:,:7].max(1)>.5
                y=a['labels'] if row['reference']=='original' else a['q'][:,:7].argmax(1)
                fw,fm=f1(y[mask],a['probabilities'].argmax(1)[mask]);assert np.isclose(fw,float(row['weighted_f1'])) and np.isclose(fm,float(row['macro_f1']))
            elif kind=='retention':
                w=a['q'][:,:7].sum(1);mask={'full':a['q'][:,7:].sum(1)==0,'partial':(w>0)&(a['q'][:,7:].sum(1)>0),'zero':w==0}[row['stratum']]
                assert int(row['population_n'])==mask.sum() and int(row['accepted_n'])==(mask&keep(a)).sum()
            else:
                mask=np.ones(len(a['rows']),bool) if row['scope']=='all' else keep(a);i={'contempt':7,'unknown':8,'not_a_face':9}[row['category']]
                assert np.isclose(a['q'][mask,i].mean(),float(row['mass']))
            checked+=1
        elif kind=='cross_evaluation':
            a=arrays[row['predictor'],int(row['seed']),state,'test_full'];b=arrays[row['selector'],int(row['seed']),state,'test_full'];assert np.isclose(risk(a,keep(b)),float(row['risk']));checked+=1
        elif kind=='seed_contrast':
            a=arrays['soft',int(row['seed']),state,'test_full'];b=arrays[row['right'],int(row['seed']),state,'test_full'];assert np.isclose(100*(risk(a,keep(a))-risk(b,keep(b))),float(row['difference_pp']));checked+=1
        elif kind=='risk_difference_curve':
            key=(int(row['seed']),row['comparison']);n=int(row['accepted_n'])
            if key not in curves:
                pair=[]
                for c in ('soft',key[1]):
                    a=arrays[c,key[0],'selected','test_full'];order=np.lexsort((a['rows'],-a['probabilities'].max(1)))
                    loss=1-a['q'][np.arange(len(order)),a['probabilities'].argmax(1)]
                    pair.append(np.cumsum(loss[order])/np.arange(1,len(order)+1))
                curves[key]=100*(pair[0]-pair[1])
            assert np.isclose(curves[key][n-1],float(row['difference_pp']),atol=1e-12);checked+=1
    bootstrap_checks=0
    for row in rows:
        if row['backbone']=='resnet18' and row['record_type']=='contrast':
            lo,hi=bootstrap_explicit(arrays,row['state'],row['right'])
            assert np.allclose([lo,hi],[float(row['lower_pp']),float(row['upper_pp'])],atol=1e-10)
            bootstrap_checks+=1
    assert bootstrap_checks==4
    report={'status':'passed','new_inference_checkpoints_verified':weights,'prediction_files_verified':predictions,'original_training_sources_verified_if_available':training_sources,'independent_numeric_rows_checked':checked,'independent_bootstrap_intervals_verified':bootstrap_checks,'retained_swin_rows_unchanged':len(retained),'ledger_sha256':sha(OUT/'reported_results.csv'),'internal_not_external_replication':True}
    (OUT/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
