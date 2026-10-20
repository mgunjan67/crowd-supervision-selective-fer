"""Pure CPU tests reject incomplete, mismatched or changed release evidence."""
import importlib.util
import json
from pathlib import Path
import pytest

SOURCE=Path(__file__).resolve().parent.parent/'training/contracts_followup.py'
spec=importlib.util.spec_from_file_location('audited_followup_contracts',SOURCE)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def logs():
    rows=[];best=best_risk=float('inf')
    for epoch in range(1,13):
        distance=[.9,.5,.5]+[.7]*9
        risk=[.8,.7,.3,.3]+[.4]*8
        row=dict(epoch=epoch,distance=distance[epoch-1],risk80=risk[epoch-1],
                 selected=distance[epoch-1]<best,risk_selected=risk[epoch-1]<best_risk)
        best=min(best,distance[epoch-1]);best_risk=min(best_risk,risk[epoch-1]);rows.append(row)
    return rows


def test_selection_uses_earliest_ties_and_rejects_bad_flags():
    rows=logs()
    assert c.selected_epochs(rows)==dict(selected=2,risk_selected=3,epoch12=12)
    rows[2]['selected']=True
    with pytest.raises(AssertionError):c.selected_epochs(rows)


def test_role_contract_rejects_missing_roles_and_changed_bytes(tmp_path):
    prediction=tmp_path/'pred';prediction.mkdir()
    weight=tmp_path/'weights.pt';weight.write_bytes(b'weights')
    record=dict(k=8,condition='soft',seed=17,state='selected',expected_epoch=2)
    counts=dict(selection=1667,calibration=1642,test_full=3589)
    roles={}
    for role in c.ROLES:
        path=prediction/f'{role}.npz';path.write_bytes(role.encode())
        roles[role]=dict(n=counts[role],sha256=c.sha(path))
    report=dict(**record,epoch=2,roles=roles,weight_sha256=c.sha(weight))
    c.validate_report(report,record,prediction,weight,counts)
    del report['roles']['calibration']
    with pytest.raises(AssertionError):c.validate_report(report,record,prediction,weight,counts)
    report['roles']['calibration']=dict(n=1642,sha256=c.sha(prediction/'calibration.npz'))
    (prediction/'calibration.npz').write_bytes(b'tampered')
    with pytest.raises(AssertionError):c.validate_report(report,record,prediction,weight,counts)


def make_family(root):
    paper=root/'paper';training=paper/'followup/training';training.mkdir(parents=True)
    (paper/'original.py').write_bytes(b'original frozen training code')
    (paper/'source.csv').write_bytes(b'original source counts')
    split=dict(source_hashes={'source.csv':c.sha(paper/'source.csv')},
               roles=dict(selection=list(range(1667)),calibration=list(range(1642)),test_full=list(range(3589))))
    save(paper/'final_experiment/split_manifest.json',split)
    runs=[(7,'permutation',s) for s in (17,42,89)]+[(8,name,s) for s in (17,42,89) for name in ('soft','uniform','tie_uniform')]
    freeze=dict(runs=runs,source_sha256={'original.py':c.sha(paper/'original.py')})
    save(training/'freeze.json',freeze)
    (training/'adapter1.py').write_bytes(b'adapter1');(training/'adapter2.py').write_bytes(b'adapter2')
    first=dict(source_sha256={'adapter1.py':c.sha(training/'adapter1.py')},original_freeze_sha256=c.sha(training/'freeze.json'))
    save(training/'amendment_001_freeze.json',first)
    second=dict(source_sha256={'adapter2.py':c.sha(training/'adapter2.py')},
                original_freeze_sha256=c.sha(training/'freeze.json'),
                first_amendment_freeze_sha256=c.sha(training/'amendment_001_freeze.json'))
    save(training/'amendment_002_freeze.json',second)
    for k,condition,seed in runs:
        folder=training/'runs'/f'k{k}_{condition}_seed{seed}';folder.mkdir(parents=True)
        configuration=dict(k=k,condition=condition,seed=seed,class_names=list(c.CLASSES[:k]),
                           source_sha256=freeze['source_sha256'],initial_state_sha256=f'{k}-{seed}')
        save(folder/'configuration.json',configuration)
        if k==7:save(paper/f'final_experiment/runs/soft_seed{seed}/configuration.json',dict(initial_state_sha256=f'{k}-{seed}'))
        save(folder/'execution_amendment.json',first)
        save(folder/'execution_amendment_002.json',dict(**second,applied_after_committed_epoch=0))
        for row in logs():
            epoch=row['epoch'];save(folder/f'epoch-{epoch:02d}.json',row)
            (folder/f'epoch-{epoch:02d}.pt').write_bytes(f'epoch{epoch}'.encode())
            (folder/f'epoch-{epoch:02d}-selection.npz').write_bytes(f'selection{epoch}'.encode())
        for name in ('best.pt','risk_best.pt'):(folder/name).write_bytes(name.encode())
        save(folder/'training_complete.json',dict(configuration=configuration,calibration_or_test_used=False,
             best_sha256=c.sha(folder/'best.pt'),risk_best_sha256=c.sha(folder/'risk_best.pt'),final_sha256=c.sha(folder/'epoch-12.pt')))
    return training,paper,runs


def test_complete_family_has_33_unique_states_and_guards_amendment(tmp_path):
    training,paper,runs=make_family(tmp_path)
    records,sources=c.collect_states(training,paper,runs)
    assert len(records)==33
    assert sum(r['k']==7 for r in records)==6 and sum(r['k']==8 for r in records)==27
    assert all(r['expected_epoch']==2 for r in records if r['state']=='selected')
    assert all(r['expected_epoch']==3 for r in records if r['state']=='risk_selected')
    (training/'adapter2.py').write_bytes(b'changed adapter')
    with pytest.raises(AssertionError):c.collect_states(training,paper,runs)


def test_missing_epoch_or_run_blocks_family_before_inference(tmp_path):
    training,paper,runs=make_family(tmp_path)
    missing=training/'runs/k8_tie_uniform_seed89/epoch-12-selection.npz'
    missing.unlink()
    with pytest.raises(AssertionError):c.collect_states(training,paper,runs)


def test_unmatched_eight_output_initialization_is_rejected(tmp_path):
    training,paper,runs=make_family(tmp_path)
    folder=training/'runs/k8_uniform_seed17'
    configuration=c.read_json(folder/'configuration.json');configuration['initial_state_sha256']='unmatched'
    save(folder/'configuration.json',configuration)
    completion=c.read_json(folder/'training_complete.json');completion['configuration']=configuration
    save(folder/'training_complete.json',completion)
    with pytest.raises(AssertionError):c.collect_states(training,paper,runs)


def finish_evaluation(training,paper,runs):
    for name in ('evaluate_followup.py','analyze_followup.py','contracts_followup.py'):
        (training/name).write_bytes(name.encode())
    records,sources=c.collect_states(training,paper,runs)
    frozen=dict(states=records,source_sha256=c.evaluation_sources(training))
    save(training/'evaluation_freeze.json',frozen)
    counts={role:len(sources['split']['roles'][role]) for role in c.ROLES}
    for record in records:
        name=f'k{record["k"]}_{record["condition"]}_seed{record["seed"]}_{record["state"]}'
        directory=training/'predictions'/name;directory.mkdir(parents=True)
        weights=training/'weights'/f'{name}.pt';weights.parent.mkdir(exist_ok=True);weights.write_bytes(name.encode())
        roles={}
        for role in c.ROLES:
            path=directory/f'{role}.npz';path.write_bytes((name+role).encode())
            roles[role]=dict(n=counts[role],sha256=c.sha(path))
        save(directory/'evaluation.json',dict(**record,epoch=record['expected_epoch'],
             roles=roles,weight_sha256=c.sha(weights)))
    save(training/'evaluation_complete.json',dict(states=33,training_runs=12,
         calibration_or_test_used_for_training=False,evaluation_freeze_sha256=c.sha(training/'evaluation_freeze.json')))


def test_completed_evaluation_requires_all_33_reports_and_frozen_analysis(tmp_path):
    training,paper,runs=make_family(tmp_path)
    finish_evaluation(training,paper,runs)
    result=c.validate_evaluation(training,paper,runs)
    assert result['states']==33 and result['validated_prediction_archives']==99
    (training/'analyze_followup.py').write_bytes(b'changed after inference')
    with pytest.raises(AssertionError):c.validate_evaluation(training,paper,runs)


def test_portable_replay_needs_no_images_or_unreleased_epoch_weights(tmp_path):
    training,paper,runs=make_family(tmp_path)
    split=c.read_json(paper/'final_experiment/split_manifest.json')
    image_paths=[];audit_rows=[]
    for partition in ('Training','PublicTest','PrivateTest'):
        path=paper/f'data/fer2013/{partition}.npz';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(partition.encode());key=path.relative_to(paper).as_posix()
        split['source_hashes'][key]=c.sha(path);image_paths.append(path)
        audit_rows.append(dict(path=key,expected=c.sha(path),observed=c.sha(path),match=True))
    save(paper/'final_experiment/split_manifest.json',split)
    save(training/'input_data_audit.json',dict(status='passed',dataset_source_hashes=audit_rows))
    finish_evaluation(training,paper,runs)
    hashes=[]
    for path in training.rglob('*.pt'):
        hashes.append(dict(file='paper_revision/'+path.relative_to(paper).as_posix(),
                           sha256=c.sha(path),bytes=path.stat().st_size,
                           release_state=path.parent.name=='weights'))
    save(training/'ALL_CHECKPOINT_HASHES.json',dict(released_inference_states=33,checkpoints=hashes))
    for path in image_paths:path.unlink()
    for path in (training/'runs').rglob('*.pt'):path.unlink()
    for path in (training/'weights').glob('*.pt'):path.unlink()
    with pytest.raises(AssertionError):c.validate_evaluation(training,paper,runs)
    result=c.validate_evaluation(training,paper,runs,portable=True)
    assert result['validated_prediction_archives']==99
    assert result['released_inference_weights_verified']==0
    assert result['released_inference_weights_not_supplied']==33
    assert len(result['omitted_image_sources'])==3
    assert result['checkpoint_verification']['retained_only_checkpoint_records']==168
    # Retained-only states cannot be silently changed either.
    ledger=c.read_json(training/'ALL_CHECKPOINT_HASHES.json')
    changed=next(row for row in ledger['checkpoints'] if row['file'].endswith('/best.pt'))
    changed['sha256']='0'*64;save(training/'ALL_CHECKPOINT_HASHES.json',ledger)
    with pytest.raises(AssertionError):c.validate_evaluation(training,paper,runs,portable=True)
