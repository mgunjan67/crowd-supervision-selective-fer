"""Evaluate frozen new states only after the complete declared training family."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import train_followup as experiment
from contracts_followup import (collect_states,evaluation_sources,read_json,sha,validate_report)

HERE = experiment.HERE
PAPER = experiment.PAPER

def main():
    experiment.freeze()
    records,sources=collect_states(HERE,PAPER,experiment.RUNS)
    role_counts={role:len(sources['split']['roles'][role]) for role in ('selection','calibration','test_full')}
    source_hashes=evaluation_sources(HERE)
    freeze_path = HERE/'evaluation_freeze.json'
    if freeze_path.exists():
        old=read_json(freeze_path)
        assert old['states']==records and old['source_sha256']==source_hashes
    else:
        freeze_path.write_text(json.dumps(dict(created_utc=experiment.now(),states=records,source_sha256=source_hashes),indent=2),encoding='utf-8')
    device = torch.device('cuda')
    assert torch.cuda.is_available()
    for record in records:
        name = f'k{record["k"]}_{record["condition"]}_seed{record["seed"]}_{record["state"]}'
        target = HERE/'predictions'/name
        target.mkdir(parents=True,exist_ok=True)
        if (target/'evaluation.json').exists():
            saved=json.loads((target/'evaluation.json').read_text())
            weight=HERE/'weights'/f'{name}.pt'
            validate_report(saved,record,target,weight,role_counts)
            continue
        experiment.base.seed_all(record['seed'])
        torch.set_num_threads(1)
        path = PAPER/record['checkpoint']
        assert experiment.base.data.sha(path) == record['checkpoint_sha256']
        checkpoint = torch.load(path,map_location='cpu',weights_only=True)
        assert checkpoint['class_names']==list(experiment.CLASSES[:record['k']])
        assert checkpoint['configuration']['k']==record['k']
        assert checkpoint['configuration']['condition']==record['condition']
        assert checkpoint['configuration']['seed']==record['seed']
        assert checkpoint['epoch']==record['expected_epoch']
        model = experiment.base.build_model('gated',record['k'],pretrained=False).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        weights = HERE/'weights'/f'{name}.pt'
        weights.parent.mkdir(exist_ok=True)
        if not weights.exists():
            experiment.base.atomic_save(checkpoint,weights)
        else:
            existing=torch.load(weights,map_location='cpu',weights_only=True)
            assert existing['class_names']==checkpoint['class_names']
            assert existing['epoch']==checkpoint['epoch']
            assert all(torch.equal(value,existing['state_dict'][key])
                       for key,value in checkpoint['state_dict'].items())
            del existing
        report = dict(**record, epoch=checkpoint['epoch'], weight_sha256=experiment.base.data.sha(weights),
                      roles={}, started_utc=experiment.now())
        for role in ('selection','calibration','test_full'):
            dataset = experiment.base.Images(role,record['seed'])
            loader = DataLoader(dataset,batch_size=32,shuffle=False,num_workers=0,pin_memory=True)
            p = experiment.base.predict(model,loader,device)
            a = dataset.arrays
            assert p.shape==(role_counts[role],record['k']) and np.isfinite(p).all()
            assert (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-6,rtol=0)
            output = target/f'{role}.npz'
            np.savez_compressed(output,probabilities=p,rows=a['rows'],q=a['q'],
                                hashes=a['hashes'],labels=a['labels'])
            report['roles'][role] = dict(n=len(p),sha256=experiment.base.data.sha(output))
            if role == 'selection':
                reference=path.parent/f'epoch-{checkpoint["epoch"]:02d}-selection.npz'
                assert sha(reference)==record['selection_reference_sha256']
                with np.load(reference,allow_pickle=False) as old:
                    assert np.array_equal(old['rows'],a['rows'])
                    assert np.array_equal(old['probabilities'],p), 'Selected state did not replay exactly'
        report['finished_utc'] = experiment.now()
        (target/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        validate_report(report,record,target,weights,role_counts)
        print('EVALUATED '+name,flush=True)
        del model,checkpoint
    (HERE/'evaluation_complete.json').write_text(json.dumps(dict(
        completed_utc=experiment.now(),states=len(records),training_runs=len(experiment.RUNS),
        calibration_or_test_used_for_training=False,
        evaluation_freeze_sha256=sha(freeze_path)),indent=2),encoding='utf-8')

if __name__ == '__main__':
    main()
