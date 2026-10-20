"""Frozen inference only; never fits weights or selects models on RAVDESS."""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from common import HERE,PAPER,CLASSES,annotations,sha,require_frozen
sys.path.insert(0,str(PAPER/'code'))
from emotion_cue.model import build_model as build_swin

spec=importlib.util.spec_from_file_location('transfer_resnet_model',PAPER/'backbone_followup/model.py')
resnet_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(resnet_module)


def tensor_batch(images):
    # Tested against the actual shared Grayscale3/Resize224/ToTensor/Normalize.
    t=torch.from_numpy(np.array(images,copy=True)).unsqueeze(1).float().div_(255)
    return t.sub_(.5).div_(.5).repeat(1,3,1,1)


def load_cache(actors):
    image_parts=[];id_parts=[];index_parts=[];hashes={};clip_counts=np.zeros(1440,dtype=int)
    for actor in actors:
        path=HERE/f'cache/Actor_{actor:02d}.npz'
        report_path=HERE/f'cache/Actor_{actor:02d}.json'
        report=json.loads(report_path.read_text())
        assert report['freeze_sha256']==sha(HERE/'freeze.json') and sha(path)==report['cache_sha256']
        with np.load(path,allow_pickle=False) as f:
            image_parts.append(f['images']);id_parts.append(f['clip_ids']);index_parts.append(f['frame_indices'])
        hashes[str(path.relative_to(HERE))]=sha(path)
        hashes[str(report_path.relative_to(HERE))]=sha(report_path)
    images=np.concatenate(image_parts);clip_ids=np.concatenate(id_parts);frame_indices=np.concatenate(index_parts)
    for clip_id in np.unique(clip_ids):clip_counts[clip_id]=int((clip_ids==clip_id).sum())
    eligible=clip_counts>=4
    return images,clip_ids,frame_indices,eligible,clip_counts,hashes


def load_model(item):
    path=PAPER/item['path'];assert sha(path)==item['sha256']
    cp=torch.load(path,map_location='cpu',weights_only=True)
    assert tuple(cp['class_names'])==CLASSES
    assert cp['preprocessing']=='grayscale3_resize224_bilinear_mean0.5_std0.5'
    assert cp['seed']==item['seed'] and cp['configuration']['condition']==item['condition']
    assert cp['release_evaluation_state']==item['state']
    assert cp['variant']==('gated' if item['backbone']=='swin_t' else 'resnet18_gated')
    model=(build_swin('gated',7,pretrained=False) if item['backbone']=='swin_t' else
           resnet_module.build_model('resnet18_gated',7,pretrained=False))
    model.load_state_dict(cp['state_dict'],strict=True)
    model=model.cuda().eval()
    return model


@torch.inference_mode()
def predict(model,images):
    warm=torch.zeros((32,3,224,224),device='cuda')
    for _ in range(3):
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):model(warm)
    del warm
    torch.cuda.synchronize();start=time.monotonic();last=start;parts=[]
    for start_id in range(0,len(images),32):
        batch=tensor_batch(images[start_id:start_id+32]).cuda()
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=model(batch)
        probabilities=logits.float().softmax(1).cpu().numpy()
        assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1),1,atol=1e-6,rtol=0)
        parts.append(probabilities)
        if time.monotonic()-last>=20:
            print(f'INFERENCE frames {min(start_id+32,len(images))}/{len(images)}',flush=True);last=time.monotonic()
    torch.cuda.synchronize()
    return np.concatenate(parts),time.monotonic()-start


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args();frozen=require_frozen()
    if not torch.cuda.is_available():raise RuntimeError('No CUDA; do not silently start a multi-hour CPU evaluation')
    torch.set_num_threads(4);torch.manual_seed(11011);torch.cuda.manual_seed_all(11011)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    actors=[1] if args.pilot else list(range(1,25))
    images,clip_ids,frame_indices,valid,counts,cache_hashes=load_cache(actors)
    names,actor_ids,_=annotations();assert len(images)>0
    directory=HERE/('pilot' if args.pilot else 'predictions');directory.mkdir(exist_ok=True)
    items=frozen['checkpoints']
    if args.pilot:items=[r for r in items if r['condition']=='soft' and r['seed']==17 and r['state']=='selected']
    completed=[]
    for item in items:
        name=f"{item['backbone']}_{item['condition']}_seed{item['seed']}_{item['state']}"
        path=directory/(name+'.npz');report_path=directory/(name+'.json')
        if path.exists() and report_path.exists():
            report=json.loads(report_path.read_text())
            assert report['freeze_sha256']==sha(HERE/'freeze.json') and report['output_sha256']==sha(path)
            assert report['cache_hashes']==cache_hashes
            completed.append(report);print('EXISTING '+name,flush=True);continue
        model=load_model(item);torch.cuda.reset_peak_memory_stats()
        print('START '+name,flush=True)
        frames,seconds=predict(model,images)
        clip_prob=np.full((1440,7),np.nan,dtype=np.float64)
        for clip_id in np.flatnonzero(valid):clip_prob[clip_id]=frames[clip_ids==clip_id].mean(0,dtype=np.float64)
        np.savez_compressed(path,probabilities=clip_prob,valid=valid,valid_frame_count=counts,
                            filenames=names,actors=actor_ids,frame_probabilities=frames,
                            frame_clip_ids=clip_ids,frame_indices=frame_indices)
        report=dict(**item,freeze_sha256=sha(HERE/'freeze.json'),cache_hashes=cache_hashes,
                    inference_seconds=seconds,valid_frames=len(images),eligible_clips=int(valid.sum()),
                    frames_per_second=len(images)/seconds,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                    precision='CUDA BF16 autocast; float32 softmax; float64 clip mean',
                    output_sha256=sha(path),pilot_only=args.pilot,model_outcome_summaries_inspected=False)
        report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        completed.append(report)
        print(f"DONE {name}: {len(images)} frames, {seconds:.2f}s, {len(images)/seconds:.1f} frames/s",flush=True)
        del model;torch.cuda.empty_cache()
    result=dict(status='technical pilot complete' if args.pilot else '36 frozen inference states complete',
                freeze_sha256=sha(HERE/'freeze.json'),checkpoint_count=len(completed),actors=actors,
                eligible_clips=int(valid.sum()),valid_frames=len(images),new_training_runs=0,
                outputs=[dict(name=Path(r['path']).stem,backbone=r['backbone'],output_sha256=r['output_sha256'],
                              seconds=r['inference_seconds']) for r in completed])
    (HERE/('pilot_complete.json' if args.pilot else 'inference_complete.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
