"""Independent arithmetic and provenance checks for the bounded revision."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def f1(y,p):
    mat=np.zeros((7,7),int);np.add.at(mat,(y,p),1)
    denominator=mat.sum(0)+mat.sum(1);scores=np.divide(2*np.diag(mat),denominator,out=np.zeros(7),where=denominator>0)
    return float((scores*mat.sum(1)).sum()/len(y))
def main():
    existing=json.loads((OUT/'existing_diagnostics_audit.json').read_text());assert existing['status']=='passed'
    assert json.loads((OUT/'target_audit.json').read_text())['status']=='passed'
    with (OUT/'reported_results.csv').open(newline='') as f:rr=list(csv.DictReader(f))
    roles=json.loads((OUT/'split_manifest.json').read_text())['roles'];verified=0;checked_rows=0
    frozen=json.loads((OUT/'evaluation_freeze.json').read_text())
    contracts=[];source_verified=0
    for record in frozen['checkpoints']:
        source=ROOT/record['source']
        if source.exists():assert sha(source)==record['source_sha256'];source_verified+=1
        seed,state=record['seed'],record['state'];name=f'tie_uniform_seed{seed}_{state}';p=OUT/'weights'/f'{name}.pt'
        info=json.loads((OUT/'predictions'/name/'evaluation.json').read_text());assert sha(p)==info['weight_sha256']
        checkpoint=torch.load(p,map_location='cpu',weights_only=True);assert checkpoint['configuration']['condition']=='tie_uniform' and checkpoint['epoch']==info['epoch']
        assert checkpoint['variant']=='gated' and checkpoint['dataset']=='FER2013'
        assert tuple(checkpoint['class_names'])==('angry','disgust','fear','happy','neutral','sad','surprise')
        assert checkpoint['preprocessing']=='grayscale3_resize224_bilinear_mean0.5_std0.5'
        states=checkpoint['state_dict']
        assert {k for k in states if k.startswith('channel_gate.')}=={'channel_gate.gate.0.weight','channel_gate.gate.2.weight'}
        assert tuple(states['channel_gate.gate.0.weight'].shape)==(48,768) and tuple(states['channel_gate.gate.2.weight'].shape)==(768,48)
        assert tuple(states['classifier.weight'].shape)==(7,768) and tuple(states['classifier.bias'].shape)==(7,)
        forbidden={'optimizer_state','scheduler_state','torch_rng','cuda_rng','epoch_record'};assert not forbidden.intersection(checkpoint)
        assert checkpoint['release_protocol_sha256']==sha(OUT/'PROTOCOL.md')
        for role in ('selection','calibration','test_full'):
            path=OUT/'predictions'/name/f'{role}.npz';assert sha(path)==info['roles'][role]['sha256']
            with np.load(path,allow_pickle=False) as f:a={k:f[k] for k in f.files}
            assert a['probabilities'].shape==(len(a['rows']),7) and a['q'].shape==(len(a['rows']),10)
            assert np.allclose(a['probabilities'].sum(1),1,atol=.01) and np.isfinite(a['probabilities']).all()
            assert np.array_equal(a['rows'],roles[role])
            reference=ROOT/'reviewer_followup/predictions'/f'soft_seed{seed}_{state}'/f'{role}.npz'
            with np.load(reference,allow_pickle=False) as f:
                assert np.array_equal(a['rows'],f['rows']) and np.array_equal(a['q'],f['q']) and np.array_equal(a['labels'],f['labels'])
        mask=np.isin(a['rows'],roles['test']);a={k:v[mask] for k,v in a.items()};prob=a['probabilities'].astype(float);q=a['q'];pred=prob.argmax(1)
        rank=np.lexsort((a['rows'],-prob.max(1)));risk=1-q[np.arange(len(q)),pred]
        for row in rr:
            if row['condition']!='tie_uniform' or row['seed']!=str(seed) or row['state']!=state:continue
            if row['record_type']=='common':
                n=int(np.ceil(float(row['requested_coverage'])*len(q)));assert n==int(row['accepted_n']);assert np.isclose(risk[rank[:n]].mean(),float(row['risk']),atol=1e-12);checked_rows+=1
            elif row['record_type']=='same_population_f1':
                eligible=q[:,:7].max(1)>.5;subset=np.ones(len(q),bool) if row['population']=='all' else eligible
                y=a['labels'][subset] if row['reference']=='original' else q[subset,:7].argmax(1)
                assert len(y)==int(row['n']) and np.isclose(f1(y,pred[subset]),float(row['weighted_f1']),atol=1e-12);checked_rows+=1
            elif row['record_type']=='retention':
                w=q[:,:7].sum(1);unsupported=q[:,7:].sum(1)
                mask={'full':unsupported==0,'partial':(w>0)&(unsupported>0),'zero':w==0}[row['stratum']]
                keep=np.zeros(len(q),bool);keep[rank[:int(np.ceil(.8*len(q)))]]=True
                assert int(mask.sum())==int(row['population_n']) and int((mask&keep).sum())==int(row['accepted_n']);checked_rows+=1
            elif row['record_type']=='unsupported_categories':
                role='calibration' if row['population']=='calibration' else 'test_full'
                with np.load(OUT/'predictions'/name/f'{role}.npz',allow_pickle=False) as f:aa={k:f[k] for k in f.files}
                if role=='test_full':mask=np.isin(aa['rows'],roles['test']);aa={k:v[mask] for k,v in aa.items()}
                order=np.lexsort((aa['rows'],-aa['probabilities'].max(1)))
                ids=order[:int(np.ceil(.8*len(order)))] if row['scope']=='global80' else np.arange(len(order))
                index={'contempt':7,'unknown':8,'not_a_face':9}[row['category']]
                assert len(ids)==int(row['n']) and np.isclose(aa['q'][ids,index].mean(),float(row['mass']),atol=1e-12);checked_rows+=1
        contracts.append(dict(name=name,sha256=sha(p),epoch=info['epoch']));verified+=1
    for state in ('selected','epoch12'):
        contrast=[r for r in rr if r['record_type']=='contrast' and r['left']=='soft' and r['right']=='tie_uniform' and r['state']==state][0]
        differences=[]
        for seed in (17,42,89):
            metrics={c:next(float(r['risk']) for r in rr if r['record_type']=='common' and r['condition']==c and r['seed']==str(seed) and r['state']==state and r['requested_coverage']=='0.8') for c in ('soft','tie_uniform')}
            differences.append(100*(metrics['soft']-metrics['tie_uniform']))
        assert np.isclose(np.mean(differences),float(contrast['point_difference_pp']),atol=1e-12)
        assert float(contrast['lower_pp'])<=float(contrast['upper_pp'])
    report={'status':'passed','ledger_sha256':sha(OUT/'reported_results.csv'),'new_inference_checkpoints_verified':verified,'original_training_checkpoint_files_verified':source_verified,'source_checkpoint_boundary':'Full training checkpoints can be verified on the original workspace; public inference-only archives preserve their recorded provenance hashes but do not include optimizer checkpoints','independently_recomputed_new_rows':checked_rows,'existing_diagnostics_audit':existing,'new_checkpoint_contracts':contracts,'boundary':'Second arithmetic implementation and internal checks, not independent human replication or source authenticity verification'}
    (OUT/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('existing_diagnostics_audit','new_checkpoint_contracts')},indent=2))
if __name__=='__main__':main()
