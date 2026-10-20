"""CPU-only provenance and completeness contracts for the final follow-up."""
import hashlib
import json
import math
from pathlib import Path

CLASSES=('angry','disgust','fear','happy','neutral','sad','surprise','contempt')
ROLES=('selection','calibration','test_full')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_static_sources(training,paper,portable=False):
    training,paper=Path(training),Path(paper)
    original=read_json(training/'freeze.json')
    for name,expected in original['source_sha256'].items():
        assert sha(paper/name)==expected,('original training source changed',name)
    split=read_json(paper/'final_experiment/split_manifest.json')
    omitted_images=[]
    allowed_missing={f'data/fer2013/{name}.npz' for name in ('Training','PublicTest','PrivateTest')}
    for name,expected in split['source_hashes'].items():
        path=paper/name
        if path.exists():
            assert sha(path)==expected,('original data source changed',name)
        else:
            assert portable and name in allowed_missing,('required source missing',name)
            omitted_images.append(dict(path=name,retained_source_sha256=expected,
                                       verification='Source hash retained; image bytes not redistributed or verified in this replay'))
    if omitted_images:
        provenance=read_json(training/'input_data_audit.json')
        assert provenance['status']=='passed'
        data_records={row['path']:row for row in provenance['dataset_source_hashes']}
        for row in omitted_images:
            recorded=data_records[row['path']]
            assert recorded['match'] is True
            assert recorded['expected']==recorded['observed']==row['retained_source_sha256']
            row['retained_pretraining_audit_sha256']=sha(training/'input_data_audit.json')
    first=read_json(training/'amendment_001_freeze.json')
    second=read_json(training/'amendment_002_freeze.json')
    original_sha=sha(training/'freeze.json')
    assert first['original_freeze_sha256']==original_sha
    assert second['original_freeze_sha256']==original_sha
    assert second['first_amendment_freeze_sha256']==sha(training/'amendment_001_freeze.json')
    for number,record in [(1,first),(2,second)]:
        for name,expected in record['source_sha256'].items():
            assert sha(training/name)==expected,(f'amendment {number} source changed',name)
    return dict(original=original,first_amendment=first,second_amendment=second,
                original_freeze_sha256=original_sha,
                first_amendment_sha256=sha(training/'amendment_001_freeze.json'),
                second_amendment_sha256=sha(training/'amendment_002_freeze.json'),
                split=split,omitted_image_sources=omitted_images)


def selected_epochs(logs):
    assert [row['epoch'] for row in logs]==list(range(1,13))
    for row in logs:
        assert math.isfinite(row['distance']) and row['distance']>=0
        assert math.isfinite(row['risk80']) and 0<=row['risk80']<=1
    distance=min(range(len(logs)),key=lambda i:logs[i]['distance'])+1
    risk=min(range(len(logs)),key=lambda i:logs[i]['risk80'])+1
    best,best_risk=float('inf'),float('inf')
    for row in logs:
        assert row['selected']==(row['distance']<best)
        assert row['risk_selected']==(row['risk80']<best_risk)
        best=min(best,row['distance']);best_risk=min(best_risk,row['risk80'])
    return {'selected':distance,'risk_selected':risk,'epoch12':12}


def collect_states(training,paper,runs,portable=False):
    training,paper=Path(training),Path(paper)
    sources=validate_static_sources(training,paper,portable=portable)
    assert sources['original']['runs']==[list(row) for row in runs]
    assert len(runs)==12 and len(set(tuple(row) for row in runs))==12
    records=[];initializations={}
    ledger={}
    ledger_path=training/'ALL_CHECKPOINT_HASHES.json'
    if portable and ledger_path.exists():
        saved=read_json(ledger_path)
        assert saved['released_inference_states']==33
        entries=saved['checkpoints']
        assert len({r['file'] for r in entries})==len(entries),'Duplicate checkpoint ledger entries'
        ledger={r['file']:r for r in entries}
    checked={};retained_only={}
    def checkpoint_hash(path,expected=None):
        key='paper_revision/'+path.relative_to(paper).as_posix()
        entry=ledger.get(key)
        if path.exists():
            if key not in checked:
                checked[key]=sha(path) if expected is not None or entry is not None else None
            if expected is not None and checked[key] is None:checked[key]=sha(path)
            if expected is not None:assert checked[key]==expected,('checkpoint changed',key)
            if entry is not None:
                assert checked[key]==entry['sha256'] and path.stat().st_size==entry['bytes']
            return checked[key]
        assert portable and entry is not None,('missing local checkpoint or retained hash record',key)
        assert isinstance(entry['bytes'],int) and entry['bytes']>0
        assert isinstance(entry['sha256'],str) and len(entry['sha256'])==64
        if expected is not None:assert entry['sha256']==expected,('retained checkpoint hash mismatch',key)
        retained_only[key]=entry['sha256']
        return entry['sha256']
    for k,condition,seed in runs:
        folder=training/'runs'/f'k{k}_{condition}_seed{seed}'
        configuration=read_json(folder/'configuration.json')
        completion=read_json(folder/'training_complete.json')
        assert completion['calibration_or_test_used'] is False
        assert completion['configuration']==configuration
        assert configuration['k']==k and configuration['condition']==condition and configuration['seed']==seed
        assert configuration['class_names']==list(CLASSES[:k])
        assert configuration['source_sha256']==sources['original']['source_sha256']
        previous=initializations.setdefault((k,seed),configuration['initial_state_sha256'])
        assert previous==configuration['initial_state_sha256'],('unmatched within-family initialization',k,seed)
        if k==7:
            old=read_json(paper/f'final_experiment/runs/soft_seed{seed}/configuration.json')
            assert configuration['initial_state_sha256']==old['initial_state_sha256']
        execution=read_json(folder/'execution_amendment.json')
        assert execution==sources['first_amendment']
        execution2=read_json(folder/'execution_amendment_002.json')
        assert {key:execution2[key] for key in sources['second_amendment']}==sources['second_amendment']
        applied=execution2['applied_after_committed_epoch']
        assert isinstance(applied,int) and 0<=applied<=12
        logs=[read_json(folder/f'epoch-{epoch:02d}.json') for epoch in range(1,13)]
        epochs=selected_epochs(logs)
        for epoch in range(1,13):
            checkpoint_hash(folder/f'epoch-{epoch:02d}.pt')
            assert (folder/f'epoch-{epoch:02d}-selection.npz').is_file(),('missing retained selection',folder.name,epoch)
        for name,field in [('best.pt','best_sha256'),('risk_best.pt','risk_best_sha256'),('epoch-12.pt','final_sha256')]:
            checkpoint_hash(folder/name,completion[field])
        states=[('selected','best.pt'),('epoch12','epoch-12.pt')]+([('risk_selected','risk_best.pt')] if k==8 else [])
        for state,name in states:
            path=folder/name
            records.append(dict(k=k,condition=condition,seed=seed,state=state,
                                expected_epoch=epochs[state],
                                checkpoint=str(path.relative_to(paper)).replace('\\','/'),
                                checkpoint_sha256=checkpoint_hash(path,completion[{'selected':'best_sha256','risk_selected':'risk_best_sha256','epoch12':'final_sha256'}[state]]),
                                selection_reference_sha256=sha(folder/f'epoch-{epochs[state]:02d}-selection.npz'),
                                configuration_sha256=sha(folder/'configuration.json'),
                                training_complete_sha256=sha(folder/'training_complete.json'),
                                execution_amendment_002_sha256=sha(folder/'execution_amendment_002.json')))
    assert len(records)==33 and len({(r['k'],r['condition'],r['seed'],r['state']) for r in records})==33
    sources['checkpoint_verification']=dict(local_files_present=len(checked),
                                            local_checkpoint_hashes_verified=sum(v is not None for v in checked.values()),
                                            retained_only_checkpoint_records=len(retained_only),
                                            retained_only_note='Retained hashes establish provenance records; absent checkpoint bytes were not verified')
    return records,sources


def validate_report(report,record,predictions,weight,role_counts,require_weight=True):
    predictions,weight=Path(predictions),Path(weight)
    assert all(report[key]==value for key,value in record.items()),'Evaluation metadata changed'
    assert report['epoch']==record['expected_epoch']
    assert set(report['roles'])==set(ROLES),'Incomplete evaluation roles'
    if weight.exists():
        assert sha(weight)==report['weight_sha256'],'Released weights changed'
    else:
        assert not require_weight,('Released inference weights missing',str(weight))
    for role in ROLES:
        role_report=report['roles'][role]
        assert role_report['n']==role_counts[role],('Incorrect role population',role)
        assert sha(predictions/f'{role}.npz')==role_report['sha256'],('Prediction bytes changed',role)


def evaluation_sources(training):
    training=Path(training)
    return {name:sha(training/name) for name in ('evaluate_followup.py','analyze_followup.py','contracts_followup.py')}


def validate_evaluation(training,paper,runs,portable=False):
    training,paper=Path(training),Path(paper)
    records,sources=collect_states(training,paper,runs,portable=portable)
    frozen=read_json(training/'evaluation_freeze.json')
    assert frozen['states']==records
    assert frozen['source_sha256']==evaluation_sources(training)
    counts={role:len(sources['split']['roles'][role]) for role in ROLES}
    reports={};weights_verified=0;weights_not_supplied=0
    ledger_path=training/'ALL_CHECKPOINT_HASHES.json'
    ledger=({row['file']:row for row in read_json(ledger_path)['checkpoints']}
            if portable and ledger_path.exists() else {})
    for record in records:
        name=f'k{record["k"]}_{record["condition"]}_seed{record["seed"]}_{record["state"]}'
        directory=training/'predictions'/name
        report=read_json(directory/'evaluation.json')
        weight=training/'weights'/f'{name}.pt'
        weight_key='paper_revision/'+weight.relative_to(paper).as_posix()
        if weight_key in ledger:
            assert ledger[weight_key]['release_state'] is True
            assert ledger[weight_key]['sha256']==report['weight_sha256']
        if not weight.exists():
            assert portable and weight_key in ledger,('Missing released-weight hash provenance',weight_key)
        validate_report(report,record,directory,weight,counts,require_weight=not portable)
        if weight.exists():weights_verified+=1
        else:weights_not_supplied+=1
        reports[name]=sha(directory/'evaluation.json')
    completion=read_json(training/'evaluation_complete.json')
    assert completion['states']==33 and completion['training_runs']==12
    assert completion['calibration_or_test_used_for_training'] is False
    assert completion['evaluation_freeze_sha256']==sha(training/'evaluation_freeze.json')
    return dict(states=33,training_runs=12,validated_prediction_archives=99,
                evaluation_report_sha256=reports,source_sha256=frozen['source_sha256'],
                runtime_amendment_002_sha256=sources['second_amendment_sha256'],
                verification_mode='portable_analysis' if portable else 'strict_local',
                omitted_image_sources=sources['omitted_image_sources'],
                checkpoint_verification=sources['checkpoint_verification'],
                released_inference_weights_verified=weights_verified,
                released_inference_weights_not_supplied=weights_not_supplied,
                weights_note='Numerical replay verifies all prediction bytes; omitted weight bytes are not verified')
