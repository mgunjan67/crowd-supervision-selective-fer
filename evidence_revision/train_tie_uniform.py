"""Three predefined matched runs; earlier sources untouched."""
import importlib.util,json,sys
from pathlib import Path
import torch
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
spec=importlib.util.spec_from_file_location('evidence_base_training',ROOT/'final_experiment/train.py')
base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)

def targets(q):
    supported=q[:,:7];w=supported.sum(1)
    r=supported/w.clamp_min(torch.finfo(supported.dtype).tiny)[:,None]
    m=r.max(1).values;tied=supported==supported.max(1,keepdim=True).values;k=tied.sum(1)
    residual=(1-k*m).clamp_min(0);other=residual/(7-k).clamp_min(1)
    t=torch.where(tied,m[:,None],other[:,None])
    t=torch.where((k==7)[:,None],torch.full_like(t,1/7),t)
    return t,w

def loss(logits,q,condition):
    assert condition=='tie_uniform'
    t,w=targets(q)
    return -(w*(t*logits.float().log_softmax(1)).sum(1)).mean()

def verify_freeze():
    for name,digest in json.loads((OUT/'freeze.json').read_text())['sha256'].items():
        assert base.data.sha(ROOT/name)==digest,name

def run_all():
    verify_freeze();base.OUT=OUT;base.supervised_loss=loss
    for seed in (17,42,89):
        base.run('tie_uniform',seed)
        directory=OUT/'runs'/f'tie_uniform_seed{seed}'
        current=json.loads((directory/'configuration.json').read_text())
        previous=json.loads((ROOT/'reviewer_followup/runs'/f'uniform_seed{seed}'/'configuration.json').read_text())
        keys=('seed','epochs','batch_size','workers','variant','optimizer','learning_rate','weight_decay','precision','scheduler','selection_metric','preprocessing','class_names','initial_state_sha256','torch','torchvision','gpu','train_n','selection_n','augmentation')
        assert all(current[k]==previous[k] for k in keys),'Matched design differs'
        (directory/'matched_design.json').write_text(json.dumps({'passed':True,'checked_keys':keys},indent=2))
    (OUT/'training_complete.json').write_text(json.dumps({'seeds':[17,42,89],'new_runs':3,'test_evaluated':False,'finished_utc':base.now()},indent=2))

if __name__=='__main__':run_all()
