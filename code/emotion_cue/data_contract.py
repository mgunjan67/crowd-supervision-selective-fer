"""Dependency-light dataset split contract used by the training pipeline."""

from __future__ import annotations

import random


def deterministic_split_indices(
    size: int,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if size < 2:
        raise ValueError("At least two samples are required")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    validation_size = min(size - 1, max(1, round(size * validation_fraction)))
    validation = sorted(indices[:validation_size])
    training = sorted(indices[validation_size:])
    return training, validation
