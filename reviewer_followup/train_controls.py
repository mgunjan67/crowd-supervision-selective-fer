"""Reuse the frozen training mechanics; change only supervised targets and output root."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
spec = importlib.util.spec_from_file_location('review_frozen_training', ROOT/'final_experiment/train.py')
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

def targets(q, condition):
    supported = q[:, :7]
    w = supported.sum(1)
    r = supported / w.clamp_min(torch.finfo(supported.dtype).tiny)[:, None]
    maxima, h = r.max(1)
    if condition == 'uniform':
        t = ((1-maxima)/6)[:, None].expand(-1, 7).clone()
        t.scatter_(1, h[:, None], maxima[:, None])
    elif condition == 'tie_hard':
        tied = supported == supported.max(1, keepdim=True).values
        t = tied.to(q.dtype) / tied.sum(1, keepdim=True)
    else:
        raise ValueError(condition)
    return t, w

def loss(logits, q, condition):
    t, w = targets(q, condition)
    return -(w * (t * logits.float().log_softmax(1)).sum(1)).mean()

def verify_freeze():
    freeze = json.loads((OUT/'freeze.json').read_text())
    for path, digest in freeze['sha256'].items():
        assert base.data.sha(ROOT/path) == digest, path

def run(condition, seed):
    verify_freeze()
    base.OUT = OUT
    base.supervised_loss = loss
    base.run(condition, seed)
    current = json.loads((OUT/'runs'/f'{condition}_seed{seed}'/'configuration.json').read_text())
    previous = json.loads((ROOT/'final_experiment/runs'/f'hard_seed{seed}'/'configuration.json').read_text())
    keys = ('seed','epochs','batch_size','workers','variant','optimizer','learning_rate','weight_decay',
            'precision','scheduler','selection_metric','preprocessing','class_names','initial_state_sha256',
            'torch','torchvision','gpu','train_n','selection_n','augmentation')
    assert all(current[k] == previous[k] for k in keys), 'Matched design changed'
    (OUT/'runs'/f'{condition}_seed{seed}'/'matched_design.json').write_text(
        json.dumps({'passed': True, 'checked_keys': keys}, indent=2), encoding='utf-8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', choices=['uniform','tie_hard'], required=True)
    parser.add_argument('--seed', type=int, choices=[17,42,89], required=True)
    args = parser.parse_args()
    run(args.condition, args.seed)
