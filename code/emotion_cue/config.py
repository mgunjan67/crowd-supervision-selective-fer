"""Configuration loading with explicit, validated keys."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetPaths:
    train: Path
    test: Path


@dataclass(frozen=True)
class TrainingConfig:
    datasets: dict[str, DatasetPaths]
    learning_rate: float = 1e-4
    batch_size: int = 64
    epochs: int = 30
    validation_fraction: float = 0.1
    weight_decay: float = 1e-4
    sam_rho: float = 0.05


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required configuration key: {key}")
    return mapping[key]


def load_config(path: str | Path) -> TrainingConfig:
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    dataset_block = _require(raw, "datasets")
    datasets: dict[str, DatasetPaths] = {}
    for name, values in dataset_block.items():
        datasets[name] = DatasetPaths(
            train=Path(_require(values, "train")),
            test=Path(_require(values, "test")),
        )

    cfg = TrainingConfig(
        datasets=datasets,
        learning_rate=float(raw.get("learning_rate", 1e-4)),
        batch_size=int(raw.get("batch_size", 64)),
        epochs=int(raw.get("epochs", 30)),
        validation_fraction=float(raw.get("validation_fraction", 0.1)),
        weight_decay=float(raw.get("weight_decay", 1e-4)),
        sam_rho=float(raw.get("sam_rho", 0.05)),
    )
    if not 0.0 < cfg.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if cfg.batch_size <= 0 or cfg.epochs <= 0 or cfg.learning_rate <= 0:
        raise ValueError('batch_size, epochs and learning_rate must be positive')
    if cfg.weight_decay < 0 or cfg.sam_rho < 0:
        raise ValueError('weight_decay and sam_rho must be non-negative')
    return cfg
