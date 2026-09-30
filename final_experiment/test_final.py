"""Check supervision, splitting and deterministic augmentation before training."""
import importlib.util,json
from pathlib import Path
import numpy as np
import torch

OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('bounded_final_train',OUT/'train.py')
train=importlib.util.module_from_spec(spec);spec.loader.exec_module(train)

def test_hard_soft_identical_for_onehot_supported_targets():
    torch.manual_seed(7);x=torch.randn(4,7,requires_grad=True)
    q=torch.zeros(4,10);q[range(4),[0,2,3,6]]=1
    a=train.supervised_loss(x,q,'hard');b=train.supervised_loss(x,q,'soft')
    torch.testing.assert_close(a,b)

def test_soft_loss_keeps_supported_mass_and_zero_rows_zero():
    logits=torch.zeros(2,7,requires_grad=True)
    q=torch.zeros(2,10);q[0,0]=.2;q[0,3]=.3;q[0,8]=.5;q[1,9]=1
    loss=train.supervised_loss(logits,q,'soft')
    torch.testing.assert_close(loss,torch.tensor(.25*np.log(7),dtype=torch.float32))
    loss.backward();assert torch.equal(logits.grad[1],torch.zeros(7))

def test_hard_loss_uses_same_out_of_scope_weight():
    x=torch.zeros(1,7);q=torch.zeros(1,10);q[0,1]=.2;q[0,3]=.3;q[0,9]=.5
    torch.testing.assert_close(train.supervised_loss(x,q,'hard'),torch.tensor(.5*np.log(7),dtype=torch.float32))

def test_hard_tie_rule_is_canonical_and_finite():
    q=np.array([[.5,.5,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1]],float)
    r,w,labels=train.data.crowd_targets(q)
    assert labels.tolist()==[0,0] and w.tolist()==[1,0]
    assert np.isfinite(r).all()

def test_same_flip_for_seed_epoch_row_regardless_of_order():
    rows=list(range(1000));forward={i:train.flip_for(i,17,3) for i in rows}
    reverse={i:train.flip_for(i,17,3) for i in reversed(rows)}
    assert forward==reverse
    assert 400<sum(forward.values())<600
    assert any(forward[i]!=train.flip_for(i,17,4) for i in rows)

def test_roles_are_pixel_disjoint_and_no_invented_rows():
    import csv
    with (OUT/'split_manifest.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    hashes={role:{r['pixel_sha256'] for r in rows if r['role']==role}
            for role in ('train','selection','calibration','test')}
    for i,a in enumerate(hashes):
        for b in list(hashes)[i+1:]:assert not hashes[a]&hashes[b]
    roles=json.loads((OUT/'split_manifest.json').read_text())['roles']
    assert len(roles['train'])==28709 and len(roles['test'])==3275
    assert set(roles['test'])<=set(roles['test_full'])

def test_selection_assignment_depends_only_on_pixel_hash():
    for value in ('0'*64,'a'*64,'f'*64):
        assert train.data.role_for_hash(value) in ('selection','calibration')
        assert train.data.role_for_hash(value)==train.data.role_for_hash(value)
