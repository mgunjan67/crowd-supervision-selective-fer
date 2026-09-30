"""Re-evaluate a retained FER2013 checkpoint without training or test selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from emotion_cue.checkpoint import validate_checkpoint_metadata
from emotion_cue.constants import CLASS_NAMES
from emotion_cue.experiment import FERPartition, make_loader, predict, probability_metrics, seed_all
from emotion_cue.model import build_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument('--dataset', default='FER2013', choices=['FER2013'])
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[1]/'data/fer2013')
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args(argv)


def evaluate(args):
    if args.output_json.exists():
        raise FileExistsError('Refusing to overwrite an existing evaluation')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    validate_checkpoint_metadata(checkpoint, expected_dataset=args.dataset)
    if checkpoint.get('protocol') != 'FER2013_2026_official':
        raise ValueError('This evaluator requires the official-partition experimental checkpoint')
    if checkpoint.get('preprocessing') != 'grayscale3_resize224_bilinear_mean0.5_std0.5':
        raise ValueError('Checkpoint preprocessing contract mismatch')
    seed_all(checkpoint['seed']); torch.set_num_threads(4)
    device = torch.device(args.device)
    model = build_model(checkpoint['variant'], 7, pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    test = FERPartition(args.data_dir, 'PrivateTest')
    if checkpoint['split_rows']['PrivateTest'] != test.rows.tolist():
        raise ValueError('PrivateTest row membership differs from the checkpoint')
    labels, probabilities = predict(model, make_loader(test,args.batch_size,checkpoint['seed'],args.workers),device)
    training_hashes = set(np.load(args.data_dir/'Training.npz')['hashes'])
    keep = np.array([h not in training_hashes for h in test.hashes])
    result = {'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'dataset':args.dataset,'variant':checkpoint['variant'],'seed':checkpoint['seed'],
        'selected_epoch':checkpoint['epoch'],'class_names':list(CLASS_NAMES),
        'device':str(device),'precision':'bfloat16 autocast' if device.type=='cuda' else 'float32',
        'metrics':probability_metrics(labels,probabilities),
        'test_without_exact_training_duplicates':{'n':int(keep.sum()),'metrics':probability_metrics(labels[keep],probabilities[keep])}}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_json.with_suffix('.predictions.npz')
    if prediction_path.exists(): raise FileExistsError(prediction_path)
    np.savez_compressed(prediction_path,labels=labels,probabilities=probabilities,rows=test.rows)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()),indent=2))
