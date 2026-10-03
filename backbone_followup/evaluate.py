"""Freeze all nine completed runs before inference; 18 weights, 54 predictions."""
import json
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from driver import base,verify_freeze,SEEDS,CONDITIONS
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
def main():
    verify_freeze();assert (OUT/'training_complete.json').exists()
    records=[]
    for seed in SEEDS:
        for c in CONDITIONS:
            directory=OUT/'runs'/f'{c}_seed{seed}'
            done=json.loads((directory/'training_complete.json').read_text())
            assert (directory/'matched_design.json').exists()
            assert base.data.sha(directory/'best.pt')==done['checkpoint_sha256']
            for state,file in (('selected','best.pt'),('epoch12','last.pt')):
                records.append({'condition':c,'seed':seed,'state':state,'source':(directory/file).relative_to(ROOT).as_posix(),'source_sha256':base.data.sha(directory/file)})
    frozen=OUT/'evaluation_freeze.json'
    if frozen.exists():assert json.loads(frozen.read_text())['checkpoints']==records
    else:frozen.write_text(json.dumps({'frozen_utc':base.now(),'checkpoints':records},indent=2))
    for rec in records:
        c,s,state=rec['condition'],rec['seed'],rec['state'];name=f'{c}_seed{s}_{state}'
        dest=OUT/'predictions'/name;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'evaluation.json').exists():continue
        ck=torch.load(ROOT/rec['source'],map_location='cpu',weights_only=True)
        weight={k:v for k,v in ck.items() if k not in ('optimizer_state','scheduler_state','torch_rng','cuda_rng','epoch_record')}
        weight.update(release_protocol='backbone_followup/PROTOCOL.md',release_protocol_sha256=base.data.sha(OUT/'PROTOCOL.md'),release_evaluation_state=state)
        folder=OUT/'weights';folder.mkdir(exist_ok=True);path=folder/f'{name}.pt';base.atomic_save(weight,path)
        base.seed_all(s);m=base.build_model(pretrained=False).to('cuda');m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict'])
        report={**rec,'epoch':ck['epoch'],'weight_sha256':base.data.sha(path),'roles':{}}
        for role in ('selection','calibration','test_full'):
            ds=base.Images(role,s);loader=DataLoader(ds,batch_size=32,shuffle=False,num_workers=0,pin_memory=True)
            p=base.predict(m,loader,torch.device('cuda'));a=ds.arrays
            if state=='selected' and role=='selection':
                with np.load((ROOT/rec['source']).parent/'best_selection_predictions.npz') as original:
                    assert np.array_equal(p,original['probabilities']) and np.array_equal(a['rows'],original['rows'])
            file=dest/f'{role}.npz';np.savez_compressed(file,probabilities=p,rows=a['rows'],labels=a['labels'],q=a['q'],hashes=a['hashes'])
            report['roles'][role]={'n':len(p),'sha256':base.data.sha(file)}
        (dest/'evaluation.json').write_text(json.dumps(report,indent=2));print('EVALUATED '+name,flush=True)
        del m,ck,weight
    (OUT/'evaluation_complete.json').write_text(json.dumps({'models':18,'prediction_files':54,'finished_utc':base.now()},indent=2))
if __name__=='__main__':main()
