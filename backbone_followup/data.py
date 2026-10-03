"""Outcome-independent pixel-cluster roles and matched crowd targets."""
from pathlib import Path
import csv, hashlib, json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
CLASSES=('angry','disgust','fear','happy','neutral','sad','surprise')
VOTES=('anger','disgust','fear','happiness','neutral','sadness','surprise','contempt','unknown','NF')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def role_for_hash(pixel_hash):
    value=hashlib.sha256(('final-20260921:'+str(pixel_hash)).encode('ascii')).hexdigest()
    return 'selection' if int(value[-1],16)%2==0 else 'calibration'

def crowd_targets(q):
    weight=q[:,:7].sum(1)
    normalized=np.divide(q[:,:7],weight[:,None],out=np.zeros_like(q[:,:7]),where=weight[:,None]>0)
    hard=normalized.argmax(1)
    return normalized,weight,hard

def source_arrays(partitions=('Training','PublicTest','PrivateTest')):
    with (ROOT/'data/ferplus_annotations/fer2013new.csv').open(newline='',encoding='utf-8-sig') as f:
        annotations=list(csv.DictReader(f))
    result={}
    for partition in partitions:
        with np.load(ROOT/'data/fer2013'/f'{partition}.npz',allow_pickle=False) as f:
            item={k:f[k] for k in f.files}
        rows=item['rows']
        assert all(annotations[int(i)]['Usage']==partition for i in rows)
        votes=np.array([[int(annotations[int(i)][c]) for c in VOTES] for i in rows],dtype=np.float64)
        assert (votes.sum(1)>0).all()
        item['q']=votes/votes.sum(1,keepdims=True)
        result[partition]=item
    return result

def prepare():
    target=OUT/'split_manifest.json'
    if target.exists():
        record=json.loads(target.read_text())
        for path,digest in record['source_hashes'].items():
            assert sha(ROOT/path)==digest,path
        print(json.dumps(record['counts'],indent=2));return
    data=source_arrays();train_hashes=set(data['Training']['hashes'])
    pub=data['PublicTest'];private=data['PrivateTest']
    isolated=np.array([h not in train_hashes for h in pub['hashes']])
    selection=isolated & np.array([role_for_hash(h)=='selection' for h in pub['hashes']])
    calibration=isolated & ~selection
    excluded_hashes=train_hashes|set(pub['hashes'])
    test=np.array([h not in excluded_hashes for h in private['hashes']])
    roles={'train':data['Training']['rows'].tolist(),'selection':pub['rows'][selection].tolist(),
           'calibration':pub['rows'][calibration].tolist(),'test':private['rows'][test].tolist(),
           'test_full':private['rows'].tolist()}
    sets=[train_hashes,set(pub['hashes'][selection]),set(pub['hashes'][calibration]),set(private['hashes'][test])]
    assert all(not sets[i]&sets[j] for i in range(4) for j in range(i+1,4))
    counts={k:len(v) for k,v in roles.items()}
    counts['public_excluded_training_overlap']=int((~isolated).sum())
    counts['test_excluded_train_or_public_overlap']=int((~test).sum())
    q=data['Training']['q'];r,w,hard=crowd_targets(q)
    counts['training_zero_supported_vote_mass']=int((w==0).sum())
    counts['training_positive_mass_tied_maximum']=int(((r==r.max(1,keepdims=True)).sum(1)>1)[w>0].sum())
    sources=['final_experiment/PROTOCOL.md','final_experiment/data.py','data/ferplus_annotations/fer2013new.csv']
    sources += [f'data/fer2013/{p}.npz' for p in data]
    record={'roles':roles,'counts':counts,'class_order':CLASSES,'vote_order':VOTES,
        'split_rule':'sha256(final-20260921: + pixel_sha256), even final hex digit selection',
        'source_hashes':{p:sha(ROOT/p) for p in sources}}
    target.write_text(json.dumps(record,indent=2),encoding='utf-8')
    rows=[]
    id_role={int(i):role for role,ids in roles.items() if role!='test_full' for i in ids}
    for partition,item in data.items():
        for i,row in enumerate(item['rows']):
            rows.append({'row':int(row),'partition':partition,'role':id_role.get(int(row),'excluded_overlap'),
                'pixel_sha256':str(item['hashes'][i])})
    with (OUT/'split_manifest.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(counts,indent=2))

def load_role(role):
    manifest=json.loads((OUT/'split_manifest.json').read_text())
    partition='Training' if role=='train' else 'PublicTest' if role in ('selection','calibration') else 'PrivateTest'
    item=source_arrays((partition,))[partition]
    mask=np.isin(item['rows'],manifest['roles'][role])
    return {k:v[mask] for k,v in item.items()}

if __name__=='__main__': prepare()
