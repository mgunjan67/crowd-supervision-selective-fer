"""Evaluate/export all new weights only after three predefined runs finish."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from train_tie_uniform import base,verify_freeze
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent

def main():
    verify_freeze();assert (OUT/'training_complete.json').exists()
    records=[]
    for seed in (17,42,89):
        source=OUT/'runs'/f'tie_uniform_seed{seed}'
        done=json.loads((source/'training_complete.json').read_text())
        assert (source/'matched_design.json').exists()
        assert base.data.sha(source/'best.pt')==done['checkpoint_sha256']
        for state,name in (('selected','best.pt'),('epoch12','last.pt')):
            records.append({'condition':'tie_uniform','seed':seed,'state':state,
                'source':(source/name).relative_to(ROOT).as_posix(),'source_sha256':base.data.sha(source/name)})
    frozen=OUT/'evaluation_freeze.json'
    if frozen.exists():assert json.loads(frozen.read_text())['checkpoints']==records
    else:frozen.write_text(json.dumps({'frozen_utc':base.now(),'checkpoints':records},indent=2))
    for record in records:
        seed,state=record['seed'],record['state'];name=f'tie_uniform_seed{seed}_{state}'
        target=OUT/'predictions'/name;target.mkdir(parents=True,exist_ok=True)
        if (target/'evaluation.json').exists():continue
        checkpoint=torch.load(ROOT/record['source'],weights_only=True,map_location='cpu')
        exported={k:v for k,v in checkpoint.items() if k not in ('optimizer_state','scheduler_state','torch_rng','cuda_rng','epoch_record')}
        exported.update(release_protocol='evidence_revision/PROTOCOL.md',release_protocol_sha256=base.data.sha(OUT/'PROTOCOL.md'),release_evaluation_state=state)
        weights=OUT/'weights';weights.mkdir(exist_ok=True);weight=weights/f'{name}.pt';base.atomic_save(exported,weight)
        base.seed_all(seed);model=base.build_model('gated',7,pretrained=False).to('cuda')
        reread=torch.load(weight,weights_only=True,map_location='cpu');model.load_state_dict(reread['state_dict'])
        report={**record,'epoch':checkpoint['epoch'],'weight_sha256':base.data.sha(weight),'roles':{}}
        for role in ('selection','calibration','test_full'):
            dataset=base.Images(role,seed);loader=DataLoader(dataset,batch_size=32,shuffle=False,num_workers=0,pin_memory=True)
            p=base.predict(model,loader,torch.device('cuda'));a=dataset.arrays
            if state=='selected' and role=='selection':
                with np.load((ROOT/record['source']).parent/'best_selection_predictions.npz') as original:
                    assert np.array_equal(p,original['probabilities']) and np.array_equal(a['rows'],original['rows'])
            path=target/f'{role}.npz';np.savez_compressed(path,probabilities=p,rows=a['rows'],labels=a['labels'],q=a['q'],hashes=a['hashes'])
            report['roles'][role]={'n':len(p),'sha256':base.data.sha(path)}
        (target/'evaluation.json').write_text(json.dumps(report,indent=2))
        print('EVALUATED',name,flush=True);del model,checkpoint,exported,reread
    (OUT/'evaluation_complete.json').write_text(json.dumps({'models':6,'prediction_files':18,'finished_utc':base.now()},indent=2))

if __name__=='__main__':main()
