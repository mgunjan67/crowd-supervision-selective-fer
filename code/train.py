"""Deterministic training entry point for the paper release.

Example:
    python train.py --config config.yaml --dataset FER2013 \
        --variant gated --optimizer adamw --seed 42 --output-dir output
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from emotion_cue.config import load_config
from emotion_cue.constants import MODEL_VARIANTS, OPTIMIZERS
from emotion_cue.data import build_loaders
from emotion_cue.metrics import classification_metrics
from emotion_cue.model import build_model
from emotion_cue.sam import SAM


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", choices=MODEL_VARIANTS, required=True)
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="adamw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_standard_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        count += labels.size(0)
    return total_loss / count, correct / count


def run_sam_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        first_logits = model(images)
        first_loss = criterion(first_logits, labels)
        first_loss.backward()
        optimizer.first_step(zero_grad=True)
        criterion(model(images), labels).backward()
        optimizer.second_step(zero_grad=True)
        total_loss += first_loss.item() * labels.size(0)
        correct += first_logits.argmax(1).eq(labels).sum().item()
        count += labels.size(0)
    return total_loss / count, correct / count


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = count = 0
    predictions, labels_out = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        count += labels.size(0)
        predictions.extend(logits.argmax(1).cpu().tolist())
        labels_out.extend(labels.cpu().tolist())
    return total_loss / count, labels_out, predictions


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    predictions, labels = [], []
    for images, batch_labels in loader:
        logits = model(images.to(device))
        predictions.extend(logits.argmax(1).cpu().tolist())
        labels.extend(batch_labels.tolist())
    return labels, predictions


def main():
    args = parse_args()
    seed_everything(args.seed)
    cfg = load_config(args.config)
    if args.dataset not in cfg.datasets:
        raise KeyError(f"Dataset {args.dataset!r} is not present in the config")
    paths = cfg.datasets[args.dataset]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, class_names, split_indices = build_loaders(
        paths.train,
        paths.test,
        args.dataset,
        cfg.batch_size,
        cfg.validation_fraction,
        args.seed,
        args.workers,
    )
    model = build_model(
        args.variant,
        len(class_names),
        pretrained=not args.no_pretrained,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "sam":
        optimizer = SAM(
            model.parameters(),
            torch.optim.AdamW,
            rho=cfg.sam_rho,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        train_epoch = run_sam_epoch
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        train_epoch = run_standard_epoch

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / (
        f"{args.dataset}_{args.variant}_{args.optimizer}_seed{args.seed}_best.pt"
    )
    best_val_weighted_f1 = -1.0

    for epoch in range(1, cfg.epochs + 1):
        started = time.perf_counter()
        train_loss, train_accuracy = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_labels, val_predictions = evaluate_epoch(
            model, val_loader, criterion, device
        )
        val_metrics = classification_metrics(
            val_labels, val_predictions, class_names
        )
        if val_metrics["weighted_f1"] > best_val_weighted_f1:
            best_val_weighted_f1 = val_metrics["weighted_f1"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "dataset": args.dataset,
                    "variant": args.variant,
                    "optimizer": args.optimizer,
                    "seed": args.seed,
                    "class_names": list(class_names),
                    "validation_fraction": cfg.validation_fraction,
                    "split_indices": split_indices,
                    "selection_metric": "validation_weighted_f1",
                    "selection_value": best_val_weighted_f1,
                },
                checkpoint_path,
            )
        print(
            f"epoch={epoch:03d} seconds={time.perf_counter()-started:.1f} "
            f"train_loss={train_loss:.4f} train_accuracy={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_accuracy={val_metrics['accuracy']:.4f} "
            f"val_weighted_f1={val_metrics['weighted_f1']:.4f}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    true_labels, predictions = collect_predictions(model, test_loader, device)
    metrics = classification_metrics(true_labels, predictions, class_names)
    metrics_path = checkpoint_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"best_checkpoint={checkpoint_path}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    import sys
    if '--config' in sys.argv:
        main()
    else:
        from emotion_cue.experiment import main as prospective_main
        prospective_main()
