"""Frozen twelve-run vocabulary/target follow-up; never trains on test roles."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TORCH_HOME', str(PAPER / 'cache/torch'))
import numpy as np
import torch
from torch.utils.data import DataLoader

spec = importlib.util.spec_from_file_location('oct20_base_training', PAPER/'final_experiment/train.py')
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
SEEDS = (17, 42, 89)
RUNS = [(7, 'permutation', s) for s in SEEDS] + [
    (8, c, s) for s in SEEDS for c in ('soft', 'uniform', 'tie_uniform')]
CLASSES = (*base.data.CLASSES, 'contempt')

def now():
    return datetime.now(timezone.utc).isoformat()

def targets(q, rows, k, condition):
    supported = np.asarray(q, dtype=np.float64)[:, :k]
    weight = supported.sum(1)
    r = np.divide(supported, weight[:, None], out=np.zeros_like(supported), where=weight[:, None]>0)
    m = r.max(1)
    h = r.argmax(1)
    if condition == 'soft':
        t = r.copy()
    elif condition == 'uniform':
        t = np.repeat(((1-m)/(k-1))[:, None], k, axis=1)
        t[np.arange(len(t)), h] = m
    elif condition == 'tie_uniform':
        tied = r == m[:, None]
        n = tied.sum(1)
        other = np.divide(np.maximum(0, 1-n*m), k-n, out=np.zeros_like(m), where=n<k)
        t = np.where(tied, m[:, None], other[:, None])
        t[n == k] = 1/k
    elif condition == 'permutation':
        t = r.copy()
        for i, row in enumerate(rows):
            positions = np.flatnonzero(r[i] < m[i])
            if len(positions) < 2:
                continue
            digest = hashlib.sha256(f'minority-permutation-20261020:{int(row)}'.encode('ascii')).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:16], 'little'))
            positions = rng.permutation(positions)
            t[i, positions] = r[i, np.roll(positions, 1)]
    else:
        raise ValueError(condition)
    t[weight == 0] = 0
    assert np.allclose(t[weight > 0].sum(1), 1)
    if condition == 'permutation':
        assert np.allclose(np.sort(t, axis=1), np.sort(r, axis=1), atol=0, rtol=0)
        assert np.array_equal(t[r == m[:, None]], r[r == m[:, None]])
    return t, weight

def frozen_sources():
    paths = [HERE/'PROTOCOL.md', Path(__file__), HERE/'test_followup.py',
             PAPER/'final_experiment/train.py', PAPER/'final_experiment/data.py',
             PAPER/'final_experiment/split_manifest.json', PAPER/'code/emotion_cue/model.py',
             PAPER/'code/emotion_cue/data.py']
    return {str(p.relative_to(PAPER)).replace('\\', '/'): base.data.sha(p) for p in paths}

def freeze():
    record = dict(created_utc=now(), runs=RUNS, source_sha256=frozen_sources(),
                  interpretation='adaptive explanatory follow-up; previous results known')
    path = HERE/'freeze.json'
    if path.exists():
        old = json.loads(path.read_text())
        assert old['source_sha256'] == record['source_sha256'], 'Frozen training sources changed'
        assert old['runs'] == [list(x) for x in RUNS]
    else:
        path.write_text(json.dumps(record, indent=2), encoding='utf-8')

def state_digest(model):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()

def run(k, condition, seed, smoke_steps=0):
    freeze()
    directory = HERE/'runs'/f'{"smoke_" if smoke_steps else ""}k{k}_{condition}_seed{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'training_complete.json').exists():
        return
    base.seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for the declared full runs')
    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats()
    model = base.build_model('gated', k, pretrained=True).to(device)
    torch.manual_seed(1000+seed)
    model.classifier.reset_parameters()
    initial = state_digest(model)
    if k == 7:
        old = json.loads((PAPER/f'final_experiment/runs/soft_seed{seed}/configuration.json').read_text())
        assert initial == old['initial_state_sha256'], 'Seven-output initialization mismatch'
    train = base.Images('train', seed)
    selection = base.Images('selection', seed)
    t, w = targets(train.arrays['q'], train.arrays['rows'], k, condition)
    tt = torch.tensor(t, dtype=torch.float32, device=device)
    ww = torch.tensor(w, dtype=torch.float32, device=device)
    gen = torch.Generator()
    loader = DataLoader(train, batch_size=32, shuffle=True, generator=gen, num_workers=2,
                        pin_memory=True, persistent_workers=True)
    select_loader = DataLoader(selection, batch_size=32, shuffle=False, num_workers=2,
                               generator=torch.Generator().manual_seed(seed), pin_memory=True, persistent_workers=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12, eta_min=1e-6)
    config = dict(k=k, condition=condition, seed=seed, epochs=12, batch_size=32,
                  learning_rate=1e-4, weight_decay=1e-4, class_names=list(CLASSES[:k]),
                  initial_state_sha256=initial, source_sha256=frozen_sources(),
                  preprocessing='grayscale3_resize224_bilinear_mean0.5_std0.5',
                  augmentation='original stateless SplitMix64 seed/epoch/row flips',
                  selection_metric='ten_category_squared_distance', training_n=len(train),
                  selection_n=len(selection), torch=str(torch.__version__), gpu=torch.cuda.get_device_name())
    cp = directory/'configuration.json'
    if cp.exists():
        assert json.loads(cp.read_text()) == config
    else:
        cp.write_text(json.dumps(config, indent=2), encoding='utf-8')
    start = 1
    best = float('inf')
    best_risk = float('inf')
    last = directory/'restart.pt'
    if last.exists():
        state = torch.load(last, map_location='cpu', weights_only=True)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer_state'])
        scheduler.load_state_dict(state['scheduler_state'])
        start = state['epoch']+1
        best, best_risk = state['best'], state['best_risk']
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        print(f'RESUME {directory.name} epoch {start}', flush=True)
    for epoch in range(start, 13):
        train.epoch.fill_(epoch)
        gen.manual_seed(seed+epoch*100003)
        model.train()
        tick = time.perf_counter()
        total = 0.
        n = 0
        for step, (x, _, index) in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            index = index.to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x.to(device, non_blocking=True))
                loss = -(ww[index]*(tt[index]*logits.float().log_softmax(1)).sum(1)).mean()
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())*len(x)
            n += len(x)
            if smoke_steps and step+1 >= smoke_steps:
                print(json.dumps(dict(smoke=True, run=directory.name, steps=step+1,
                                      seconds=time.perf_counter()-tick, loss=total/n,
                                      peak_gpu_bytes=torch.cuda.max_memory_allocated())), flush=True)
                return
        p = base.predict(model, select_loader, device)
        q = selection.arrays['q']
        distance = float(((np.pad(p.astype(float), ((0, 0), (0, 10-k)))-q)**2).sum(1).mean())
        order = np.lexsort((selection.arrays['rows'], -p.max(1)))
        risk = float((1-q[np.arange(len(q)), p.argmax(1)])[order[:int(np.ceil(.8*len(q)))]].mean())
        is_best, is_risk = distance < best, risk < best_risk
        best, best_risk = min(best, distance), min(best_risk, risk)
        metadata = dict(state_dict=model.state_dict(), epoch=epoch, configuration=config,
                        class_names=config['class_names'], variant='gated', dataset='FER2013',
                        selection_distance=distance, selection_risk80=risk)
        base.atomic_save(metadata, directory/f'epoch-{epoch:02d}.pt')
        if is_best:
            base.atomic_save(metadata, directory/'best.pt')
        if is_risk:
            base.atomic_save(metadata, directory/'risk_best.pt')
        np.savez_compressed(directory/f'epoch-{epoch:02d}-selection.npz', probabilities=p,
                            rows=selection.arrays['rows'])
        scheduler.step()
        record = dict(run=directory.name, epoch=epoch, loss=total/n, distance=distance,
                      risk80=risk, selected=is_best, risk_selected=is_risk,
                      seconds=time.perf_counter()-tick, finished_utc=now())
        base.atomic_save(dict(metadata, optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
                             best=best, best_risk=best_risk, torch_rng=torch.get_rng_state(),
                             cuda_rng=torch.cuda.get_rng_state_all()), last)
        (directory/f'epoch-{epoch:02d}.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        (HERE/'progress.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(json.dumps(record), flush=True)
    completion = dict(configuration=config, finished_utc=now(), calibration_or_test_used=False,
                      best_sha256=base.data.sha(directory/'best.pt'),
                      risk_best_sha256=base.data.sha(directory/'risk_best.pt'),
                      final_sha256=base.data.sha(directory/'epoch-12.pt'))
    (directory/'training_complete.json').write_text(json.dumps(completion, indent=2), encoding='utf-8')
    print('TRAINING COMPLETE '+directory.name, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--k', type=int, choices=[7, 8], default=7)
    ap.add_argument('--condition', choices=['soft', 'uniform', 'tie_uniform', 'permutation'], default='permutation')
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--smoke-steps', type=int, default=0)
    args = ap.parse_args()
    for k, condition, seed in RUNS if args.all else [(args.k, args.condition, args.seed)]:
        run(k, condition, seed, args.smoke_steps)

if __name__ == '__main__':
    main()
