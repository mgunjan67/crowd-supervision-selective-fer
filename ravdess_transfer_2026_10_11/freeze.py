"""Freeze protocol, implementation, original annotations and all 36 checkpoints."""
import json
import platform
import sys
from datetime import datetime,timezone
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import torch
import torchvision
import PIL
from common import HERE,PAPER,sha,checkpoint_path,CONDITIONS,SEEDS,STATES


def main():
    target=HERE/'freeze.json'
    if target.exists():raise RuntimeError('A freeze already exists. Never silently replace a registered local protocol.')
    assert not list((HERE/'predictions').glob('*.npz')) and not list((HERE/'pilot').glob('*.npz'))
    sources=[HERE/name for name in ('PROTOCOL.txt','acquire.py','common.py','prepare.py','infer.py','analyze.py','overlap.py','test_transfer.py','freeze.py')]
    sources += [PAPER/'code/emotion_cue'/name for name in ('runtime.py','data.py','model.py','checkpoint.py','constants.py')]
    sources += [PAPER/'backbone_followup/model.py',PAPER/'backbone_followup/split_manifest.json',
                PAPER/'recovery_2026_10_10/external_data/ravdess/video_only_speech_votes.csv',
                PAPER/'recovery_2026_10_10/external_data/ravdess/pone.0196391.s008.xlsx',HERE/'download_manifest.json']
    checkpoints=[]
    for family in ('swin_t','resnet18'):
        for condition in CONDITIONS:
            for seed in SEEDS:
                for state in STATES:
                    path=checkpoint_path(family,condition,seed,state)
                    checkpoints.append(dict(backbone=family,condition=condition,seed=seed,state=state,
                                            path=str(path.relative_to(PAPER)),sha256=sha(path)))
    detector_dir=Path(mp.__file__).resolve().parent/'modules/face_detection'
    detector_hashes={name:sha(detector_dir/name) for name in ('face_detection_full_range_cpu.binarypb','face_detection_full_range_sparse.tflite')}
    counts=np.random.default_rng(11011).multinomial(24,np.full(24,1/24),size=5000).astype(np.int16)
    np.savez_compressed(HERE/'bootstrap_actor_counts.npz',counts=counts)
    sources.append(HERE/'bootstrap_actor_counts.npz')
    report=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),client_date='2026-10-11',
                status='local prospective external-evaluation plan, not external preregistration',
                external_model_results_known=False,internal_fer_results_known=True,
                source_hashes={str(path.relative_to(PAPER)):sha(path) for path in sources},
                checkpoints=checkpoints,detector_hashes=detector_hashes,
                software=dict(python=sys.version,torch=torch.__version__,torchvision=torchvision.__version__,
                              opencv=cv2.__version__,mediapipe=mp.__version__,numpy=np.__version__,pillow=PIL.__version__,
                              platform=platform.platform(),gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None),
                resources=dict(public_download_bytes=13_357_344_723,temporary_storage_cap_bytes=30_000_000_000),
                new_training_runs=0,planned_checkpoints=36,planned_actors=24,planned_clips=1440)
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_hashes','checkpoints')},indent=2))


if __name__=='__main__':main()
