"""Export and evaluate selected and fixed-epoch weights after all controls finish."""
import json, shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from train_controls import base, verify_freeze
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
SEEDS=(17,42,89);CONDITIONS=('hard','soft','uniform','tie_hard')

def main():
    verify_freeze()
    for s in SEEDS:
        for c in ('uniform','tie_hard'):
            d=OUT/'runs'/f'{c}_seed{s}'
            assert (d/'training_complete.json').exists() and (d/'matched_design.json').exists()
    records=[]
    # All checkpoints are committed and hashed before any new test evaluation.
    for c in CONDITIONS:
        for s in SEEDS:
            source=(ROOT/'final_experiment' if c in ('hard','soft') else OUT)/'runs'/f'{c}_seed{s}'
            completion=json.loads((source/'training_complete.json').read_text())
            assert base.data.sha(source/'best.pt')==completion['checkpoint_sha256']
            for state,file in (('selected','best.pt'),('epoch12','last.pt')):
                records.append({'condition':c,'seed':s,'state':state,'source':(source/file).relative_to(ROOT).as_posix(),
                                'source_sha256':base.data.sha(source/file)})
    manifest=OUT/'evaluation_freeze.json'
    if manifest.exists(): assert json.loads(manifest.read_text())['checkpoints']==records
    else: manifest.write_text(json.dumps({'frozen_utc':base.now(),'checkpoints':records},indent=2),encoding='utf-8')
    for record in records:
        c,s,state=record['condition'],record['seed'],record['state']
        name=f'{c}_seed{s}_{state}';directory=OUT/'predictions'/name;directory.mkdir(parents=True,exist_ok=True)
        done=directory/'evaluation.json'
        if done.exists(): continue
        source=ROOT/record['source'];assert base.data.sha(source)==record['source_sha256']
        checkpoint=torch.load(source,weights_only=True,map_location='cpu')
        weight_path=OUT/'weights'/f'{name}.pt';weight_path.parent.mkdir(exist_ok=True)
        inference={k:v for k,v in checkpoint.items() if k not in ('optimizer_state','scheduler_state','torch_rng','cuda_rng','epoch_record')}
        inference['release_protocol']='reviewer_followup/PROTOCOL.md'
        inference['release_protocol_sha256']=base.data.sha(OUT/'PROTOCOL.md')
        inference['release_evaluation_state']=state
        if not weight_path.exists():base.atomic_save(inference,weight_path)
        if state=='epoch12': assert checkpoint['epoch']==12
        base.seed_all(s);device=torch.device('cuda')
        model=base.build_model('gated',7,pretrained=False).to(device)
        # Evaluate the exported artifact, not only the original training container.
        exported=torch.load(weight_path,weights_only=True,map_location='cpu')
        assert exported['release_protocol_sha256']==base.data.sha(OUT/'PROTOCOL.md')
        model.load_state_dict(exported['state_dict'])
        report={**record,'epoch':checkpoint['epoch'],'weight_sha256':base.data.sha(weight_path),'roles':{},'started_utc':base.now()}
        for role in ('selection','calibration','test_full'):
            dataset=base.Images(role,s)
            loader=DataLoader(dataset,batch_size=32,shuffle=False,num_workers=0,pin_memory=True)
            p=base.predict(model,loader,device);a=dataset.arrays
            reference_verified=False
            if state=='selected':
                ref=source.parent/(f'{role}_predictions.npz' if c in ('hard','soft') else 'best_selection_predictions.npz')
                if c in ('hard','soft') or role=='selection':
                    with np.load(ref,allow_pickle=False) as previous:
                        assert np.array_equal(previous['rows'],a['rows'])
                        assert np.array_equal(previous['probabilities'],p),'Exact replay failed: '+name+'/'+role
                    reference_verified=True
            path=directory/f'{role}.npz'
            np.savez_compressed(path,probabilities=p,rows=a['rows'],labels=a['labels'],q=a['q'],hashes=a['hashes'])
            report['roles'][role]={'n':len(p),'sha256':base.data.sha(path),'prior_reference_bitwise_verified':reference_verified}
        report['finished_utc']=base.now()
        done.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('EVALUATED '+name,flush=True)
        del model,checkpoint,inference,exported
    (OUT/'evaluation_complete.json').write_text(json.dumps({'models':24,'prediction_files':72,
        'old_selected_bitwise_replays':18,'new_selected_selection_replays':6,'finished_utc':base.now()},indent=2),encoding='utf-8')

if __name__=='__main__':main()
