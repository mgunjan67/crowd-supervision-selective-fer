"""Auditable full-backbone FER2013 experiments with official CSV partitions."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from .constants import CLASS_NAMES
from .data import build_transform
from .metrics import classification_metrics
from .model import build_model


class FERPartition(Dataset):
    def __init__(self, root, partition, augment=False):
        data = np.load(Path(root) / f'{partition}.npz')
        self.images, self.labels, self.rows = data['images'], data['labels'], data['rows']
        self.hashes = data['hashes']
        self.transform = build_transform('FER2013')
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.fromarray(self.images[index]).convert('RGB')
        if self.augment and torch.rand(()) < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return self.transform(image), int(self.labels[index])


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def make_loader(dataset, batch_size, seed, workers, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed), num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0)


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x,y in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type=='cuda'):
            logits = model(x.to(device, non_blocking=True))
        probs.append(logits.float().softmax(1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probs)


def probability_metrics(labels, probabilities):
    out = classification_metrics(labels, probabilities.argmax(1), CLASS_NAMES)
    onehot = np.eye(len(CLASS_NAMES))[labels]
    out['brier_score'] = float(np.square(probabilities - onehot).sum(1).mean())
    confidence = probabilities.max(1)
    correct = probabilities.argmax(1) == labels
    bins = np.minimum((confidence * 15).astype(int), 14)
    out['ece_15_bins'] = float(sum(np.mean(bins==b) * abs(correct[bins==b].mean() - confidence[bins==b].mean()) for b in range(15) if np.any(bins==b)))
    return out


def run(args):
    if args.dataset != 'FER2013' or args.optimizer != 'adamw':
        raise ValueError('The prospective protocol supports FER2013 and AdamW only')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir/'best.pt').exists():
        raise FileExistsError('Existing run directory: choose a new output directory; no silent overwrite')
    seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args.variant, 7, pretrained=True).to(device)
    # Identical classifier initialization across variants despite extra gate draws.
    torch.manual_seed(1000 + args.seed)
    model.classifier.reset_parameters()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    datasets = {p: FERPartition(args.data_dir, p, p=='Training') for p in ['Training','PublicTest','PrivateTest']}
    loaders = {p: make_loader(d, args.batch_size, args.seed, args.workers, p=='Training') for p,d in datasets.items()}
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(learning_rate=1e-4, weight_decay=1e-4, scheduler='cosine, minimum 1e-6',
        precision='bfloat16 autocast' if device.type=='cuda' else 'float32',
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        class_names=list(CLASS_NAMES), pretrained_weights='Swin_T_Weights.IMAGENET1K_V1',
        preprocessing='grayscale3_resize224_bilinear_mean0.5_std0.5', augmentation='horizontal flip p=0.5',
        device=str(device), gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
        platform=platform.platform(), python=sys.version, torch=str(torch.__version__), torchvision=str(torchvision.__version__),
        code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    (args.output_dir/'configuration.json').write_text(json.dumps(config,indent=2))
    (args.output_dir/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    metadata = dict(dataset='FER2013',variant=args.variant,class_names=list(CLASS_NAMES),seed=args.seed,
        preprocessing=config['preprocessing'],protocol='FER2013_2026_official',configuration=config,
        split_rows={p:d.rows.tolist() for p,d in datasets.items()})
    best = -1
    for epoch in range(1,args.epochs+1):
        started = time.perf_counter(); model.train(); loss_sum=count=0
        for step,(x,y) in enumerate(loaders['Training']):
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                loss = torch.nn.functional.cross_entropy(model(x),y)
            loss.backward(); optimizer.step()
            loss_sum += float(loss.detach())*len(y); count += len(y)
            if args.smoke_steps and step+1 >= args.smoke_steps:
                if device.type=='cuda': torch.cuda.synchronize()
                print(json.dumps({'smoke_steps':step+1,'samples':count,'seconds':time.perf_counter()-started,'peak_gpu_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0}),flush=True)
                return
        labels,probs=predict(model,loaders['PublicTest'],device)
        metrics=probability_metrics(labels,probs)
        record={'epoch':epoch,'train_loss':loss_sum/count,'learning_rate':optimizer.param_groups[0]['lr'],
                'validation':metrics,'seconds':time.perf_counter()-started}
        with (args.output_dir/'epochs.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        if metrics['weighted_f1']>best:
            best=metrics['weighted_f1']
            torch.save(dict(metadata,state_dict=model.state_dict(),epoch=epoch,selection_metric='validation_weighted_f1',selection_value=best),args.output_dir/'best.pt')
        scheduler.step()
        torch.save(dict(metadata,state_dict=model.state_dict(),optimizer_state=optimizer.state_dict(),scheduler_state=scheduler.state_dict(),epoch=epoch,
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],best_validation_weighted_f1=best),args.output_dir/'last.pt')
        print(f'{args.variant} seed={args.seed} epoch={epoch}/{args.epochs} seconds={record["seconds"]:.1f} train_loss={record["train_loss"]:.4f} val_weighted_f1={metrics["weighted_f1"]:.4f}',flush=True)
    checkpoint=torch.load(args.output_dir/'best.pt',weights_only=True,map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    y,p=predict(model,loaders['PrivateTest'],device)
    np.savez_compressed(args.output_dir/'test_predictions.npz',labels=y,probabilities=p,rows=datasets['PrivateTest'].rows)
    out={'seed':args.seed,'variant':args.variant,'selected_epoch':checkpoint['epoch'],'metrics':probability_metrics(y,p)}
    train_hashes=set(datasets['Training'].hashes)
    novel=np.array([h not in train_hashes for h in datasets['PrivateTest'].hashes])
    out['test_without_exact_training_duplicates']={'n':int(novel.sum()),'metrics':probability_metrics(y[novel],p[novel])}
    (args.output_dir/'test_metrics.json').write_text(json.dumps(out,indent=2))
    print('FINISHED '+json.dumps({k:v for k,v in out.items() if k not in ['metrics','test_without_exact_training_duplicates']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',default='FER2013')
    p.add_argument('--variant',choices=['swin','gated'],required=True)
    p.add_argument('--optimizer',default='adamw')
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parents[2]/'data/fer2013')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--smoke-steps',type=int,default=0)
    run(p.parse_args())
