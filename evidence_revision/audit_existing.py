"""Second implementation checks raw labels, references and cross-evaluation."""
import csv,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent

def f1(y,p):
    cm=np.zeros((7,7),int);np.add.at(cm,(y,p),1)
    den=cm.sum(0)+cm.sum(1);scores=np.divide(2*np.diag(cm),den,out=np.zeros(7),where=den>0)
    return scores@cm.sum(1)/cm.sum(),scores.mean()

def main():
    with zipfile.ZipFile(ROOT/'data/raw/fer2013-kaggle-v1.zip') as z:
        name=next(n for n in z.namelist() if n.endswith('fer2013.csv'))
        with z.open(name) as f:raw=list(csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig')))
    with (ROOT/'data/ferplus_annotations/fer2013new.csv').open(newline='',encoding='utf-8-sig') as f:votes=list(csv.DictReader(f))
    mapping=np.array([0,1,2,3,5,6,4]);vote_names=('anger','disgust','fear','happiness','neutral','sadness','surprise','contempt','unknown','NF')
    cached={}
    for state in ('selected','epoch12'):
        for c in ('hard','soft','uniform','tie_hard'):
            for seed in (17,42,89):
                with np.load(ROOT/'reviewer_followup/predictions'/f'{c}_seed{seed}_{state}'/'test_full.npz') as f:a={k:f[k] for k in f.files}
                ids=json.loads((ROOT/'reviewer_followup/split_manifest.json').read_text())['roles']['test'];a={k:v[np.isin(a['rows'],ids)] for k,v in a.items()}
                rows=a['rows'];labels=mapping[np.array([int(raw[int(i)]['emotion']) for i in rows])]
                assert np.array_equal(labels,a['labels'])
                q=np.array([[int(votes[int(i)][name]) for name in vote_names] for i in rows],float);q/=q.sum(1)[:,None]
                assert np.array_equal(q,a['q']) and all(raw[int(i)]['Usage']==votes[int(i)]['Usage']=='PrivateTest' for i in rows)
                for i,row in enumerate(rows):
                    pixels=np.fromstring(raw[int(row)]['pixels'],sep=' ',dtype=np.uint8)
                    assert hashlib.sha256(pixels.tobytes()).hexdigest()==a['hashes'][i]
                cached[c,seed,state]=a
    with (OUT/'existing_diagnostics.csv').open(newline='') as f:rr=list(csv.DictReader(f))
    checked=0
    for r in rr:
        kind=r['record_type']
        if kind=='cross_evaluation':
            a=cached[r['predictor'],int(r['seed']),r['state']];b=cached[r['selector'],int(r['seed']),r['state']]
            order=np.lexsort((b['rows'],-b['probabilities'].max(1)));keep=order[:int(np.ceil(.8*len(order)))];pred=a['probabilities'].argmax(1)
            expected=(1-a['q'][np.arange(len(pred)),pred])[keep].mean()
            assert abs(float(r['risk'])-expected)<1e-12;checked+=1
        if kind=='same_population_f1':
            a=cached[r['condition'],int(r['seed']),r['state']];mask=np.ones(len(a['rows']),bool) if r['population']=='all' else a['q'][:,:7].max(1)>.5
            y=a['labels'] if r['reference']=='original' else a['q'][:,:7].argmax(1)
            weighted,macro=f1(y[mask],a['probabilities'].argmax(1)[mask])
            assert abs(weighted-float(r['weighted_f1']))<1e-12 and abs(macro-float(r['macro_f1']))<1e-12;checked+=1
    report={'status':'passed','retained_prediction_arrays_checked':len(cached),'raw_class_mapping':[0,1,2,3,5,6,4],
        'independently_recomputed_cross_and_f1_rows':checked,'raw_pixels_labels_votes_verified':True,
        'scope':'Second internal code implementation, not independent human/coauthor replication or mirror authentication'}
    (OUT/'existing_diagnostics_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
