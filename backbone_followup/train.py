"""Nine second-backbone paired runs; calibration/test predictions are a separate stage."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, platform, random, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('TORCH_HOME',str(ROOT/'cache/torch'))
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset,DataLoader
sys.path.append(str(ROOT/'code'))
from model import build_model
from emotion_cue.data import build_transform
spec=importlib.util.spec_from_file_location('final_experiment_data',OUT/'data.py')
data=importlib.util.module_from_spec(spec);spec.loader.exec_module(data)

def now():return datetime.now(timezone.utc).isoformat()

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True);torch.set_num_threads(4)

def flip_for(row,seed,epoch):
    """Stateless SplitMix64-derived bit, independent of worker lifetime or order."""
    mask=(1<<64)-1
    z=(int(row)+int(seed)*0x9E3779B97F4A7C15+int(epoch)*0xD1B54A32D192ED03)&mask
    z=((z^(z>>30))*0xBF58476D1CE4E5B9)&mask
    z=((z^(z>>27))*0x94D049BB133111EB)&mask
    return bool((z^(z>>31))&1)

class Images(Dataset):
    def __init__(self,role,seed):
        self.arrays=data.load_role(role);self.seed=seed;self.augment=role=='train'
        self.epoch=torch.zeros((),dtype=torch.int64).share_memory_()
        self.transform=build_transform('FER2013')
    def __len__(self):return len(self.arrays['rows'])
    def __getitem__(self,index):
        image=Image.fromarray(self.arrays['images'][index]).convert('RGB')
        if self.augment and flip_for(self.arrays['rows'][index],self.seed,int(self.epoch)):
            image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return self.transform(image),self.arrays['q'][index].astype(np.float32),index

def targets(q,condition):
    supported=q[:,:7];w=supported.sum(1)
    r=supported/w.clamp_min(torch.finfo(supported.dtype).tiny)[:,None]
    m,h=r.max(1)
    if condition=='soft':t=r
    elif condition=='uniform':
        t=((1-m)/6)[:,None].expand(-1,7).clone();t.scatter_(1,h[:,None],m[:,None])
    elif condition=='tie_uniform':
        tied=supported==supported.max(1,keepdim=True).values;k=tied.sum(1)
        other=(1-k*m).clamp_min(0)/(7-k).clamp_min(1)
        t=torch.where(tied,m[:,None],other[:,None])
        t=torch.where((k==7)[:,None],torch.full_like(t,1/7),t)
    else:raise ValueError(condition)
    return t,w

def supervised_loss(logits,q,condition):
    t,w=targets(q,condition)
    return -(w*(t*logits.float().log_softmax(1)).sum(1)).mean()

@torch.inference_mode()
def predict(model,loader,device):
    model.eval();parts=[];indices=[]
    for x,_,index in loader:
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            logits=model(x.to(device,non_blocking=True))
        parts.append(logits.float().softmax(1).cpu().numpy());indices.extend(index.tolist())
    assert indices==list(range(len(loader.dataset)))
    return np.concatenate(parts)

def atomic_save(value,path):
    temporary=path.with_suffix('.partial')
    torch.save(value,temporary);os.replace(temporary,path)

def run(condition,seed,smoke_steps=0):
    directory=OUT/'runs'/f'{"smoke_" if smoke_steps else ""}{condition}_seed{seed}'
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'training_complete.json').exists():
        print('ALREADY COMPLETE '+directory.name,flush=True);return
    seed_all(seed)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; do not silently begin a multi-hour CPU run')
    device=torch.device('cuda');torch.cuda.reset_peak_memory_stats()
    model=build_model('resnet18_gated',7,pretrained=True).to(device)
    torch.manual_seed(1000+seed);model.classifier.reset_parameters()
    initialization=hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
    train=Images('train',seed);selection=Images('selection',seed)
    generator=torch.Generator()
    train_loader=DataLoader(train,batch_size=32,shuffle=True,generator=generator,num_workers=2,
                            pin_memory=True,persistent_workers=True)
    selection_loader=DataLoader(selection,batch_size=32,shuffle=False,generator=torch.Generator().manual_seed(seed),
                                num_workers=2,pin_memory=True,persistent_workers=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=12,eta_min=1e-6)
    source_files=[OUT/'PROTOCOL.md',OUT/'split_manifest.json',OUT/'data.py',Path(__file__)]
    source_files+=list((ROOT/'code/emotion_cue').glob('*.py'))+[OUT/'model.py',OUT/'driver.py']
    sources={str(p.relative_to(ROOT)).replace('\\','/'):data.sha(p) for p in source_files}
    config=dict(condition=condition,seed=seed,epochs=12,batch_size=32,workers=2,variant='resnet18_gated',
        optimizer='AdamW',learning_rate=1e-4,weight_decay=1e-4,precision='BF16 autocast',
        scheduler='cosine minimum 1e-6',selection_metric='ten_category_squared_distribution_distance',
        preprocessing='grayscale3_resize224_bilinear_mean0.5_std0.5',class_names=list(data.CLASSES),
        initial_state_sha256=initialization,source_sha256=sources,torch=str(torch.__version__),
        torchvision=str(torchvision.__version__),python=sys.version,platform=platform.platform(),
        gpu=torch.cuda.get_device_name(),train_n=len(train),selection_n=len(selection),
        augmentation='stateless SplitMix64 horizontal flip keyed by seed, epoch, original row')
    config_path=directory/'configuration.json'
    if config_path.exists():
        old=json.loads(config_path.read_text());assert old==config,'Configuration/source changed; restart not authorized'
    else:
        config_path.write_text(json.dumps(config,indent=2),encoding='utf-8')
        (directory/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    metadata=dict(dataset='FER2013',variant='resnet18_gated',class_names=list(data.CLASSES),seed=seed,
        preprocessing=config['preprocessing'],protocol='second_backbone_resnet18_robustness',configuration=config)
    best=float('inf');start=1
    last=directory/'last.pt'
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=True)
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer_state'])
        scheduler.load_state_dict(saved['scheduler_state']);best=saved['best'];start=saved['epoch']+1
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        # Recover the committed epoch log if interruption occurred between writes.
        epoch_log=directory/f'epoch-{saved["epoch"]:02d}.json'
        if not epoch_log.exists() and 'epoch_record' in saved:
            epoch_log.write_text(json.dumps(saved['epoch_record'],indent=2),encoding='utf-8')
        print(f'RESUME {directory.name} at epoch {start}',flush=True)
    for epoch in range(start,13):
        train.epoch.fill_(epoch);generator.manual_seed(seed+epoch*100003)
        model.train();started=time.perf_counter();total=0.;n=0
        for step,(x,q,_) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(x.to(device,non_blocking=True))
                loss=supervised_loss(logits,q.to(device,non_blocking=True),condition)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();optimizer.step();total+=float(loss.detach())*len(x);n+=len(x)
            if smoke_steps and step+1>=smoke_steps:
                print(json.dumps({'smoke':condition,'steps':step+1,'loss':total/n,'seconds':time.perf_counter()-started,
                    'peak_gpu_bytes':torch.cuda.max_memory_allocated()}),flush=True);return
        p=predict(model,selection_loader,device)
        q=selection.arrays['q'];distance=float(((np.pad(p.astype(float),((0,0),(0,3)))-q)**2).sum(1).mean())
        new_best=distance<best
        if new_best:
            best=distance
            atomic_save(dict(metadata,state_dict=model.state_dict(),epoch=epoch,selection_value=distance),directory/'best.pt')
            np.savez_compressed(directory/'best_selection_predictions.npz',probabilities=p,rows=selection.arrays['rows'])
        elapsed=time.perf_counter()-started
        record=dict(epoch=epoch,train_loss=total/n,selection_distance=distance,new_best=new_best,
            learning_rate=optimizer.param_groups[0]['lr'],seconds=elapsed,finished_utc=now())
        scheduler.step()
        atomic_save(dict(metadata,state_dict=model.state_dict(),optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),epoch=epoch,best=best,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),epoch_record=record),last)
        (directory/f'epoch-{epoch:02d}.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        (OUT/'progress.json').write_text(json.dumps(dict(active=directory.name,**record),indent=2),encoding='utf-8')
        print(f'{directory.name} epoch {epoch}/12: selection_distance={distance:.5f}, loss={total/n:.4f}, seconds={elapsed:.1f}',flush=True)
    checkpoint=torch.load(directory/'best.pt',weights_only=True,map_location='cpu')
    (directory/'training_complete.json').write_text(json.dumps(dict(condition=condition,seed=seed,
        selected_epoch=checkpoint['epoch'],selection_value=checkpoint['selection_value'],
        checkpoint_sha256=data.sha(directory/'best.pt'),finished_utc=now(),calibration_or_test_used=False),indent=2),encoding='utf-8')
    print('TRAINING COMPLETE '+directory.name,flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--condition',choices=['soft','uniform','tie_uniform'],required=True)
    parser.add_argument('--seed',type=int,required=True);parser.add_argument('--smoke-steps',type=int,default=0)
    args=parser.parse_args();run(args.condition,args.seed,args.smoke_steps)

if __name__=='__main__':main()
