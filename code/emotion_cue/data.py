"""Dataset and transform factories shared by training and evaluation."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .constants import CLASS_NAMES, IMAGE_SIZE
from .data_contract import deterministic_split_indices


def build_transform(dataset_name: str, *, runtime: bool = False):
    steps = []
    if dataset_name == "FER2013" or runtime:
        steps.append(transforms.Grayscale(num_output_channels=3))
    steps.extend(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return transforms.Compose(steps)


def build_loaders(
    train_root: Path,
    test_root: Path,
    dataset_name: str,
    batch_size: int,
    validation_fraction: float,
    seed: int,
    workers: int,
):
    train_data = datasets.ImageFolder(
        root=train_root,
        transform=build_transform(dataset_name),
    )
    test_data = datasets.ImageFolder(
        root=test_root,
        transform=build_transform(dataset_name),
    )
    if tuple(train_data.classes) != CLASS_NAMES:
        raise ValueError(
            "Training directory class order does not match the release contract: "
            f"{train_data.classes} != {CLASS_NAMES}"
        )
    if tuple(test_data.classes) != CLASS_NAMES:
        raise ValueError(
            "Test directory class order does not match the release contract: "
            f"{test_data.classes} != {CLASS_NAMES}"
        )
    train_idx, val_idx = deterministic_split_indices(
        len(train_data), validation_fraction, seed
    )
    generator = __import__("torch").Generator().manual_seed(seed)
    train_loader = DataLoader(
        Subset(train_data, train_idx),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=workers,
    )
    val_loader = DataLoader(
        Subset(train_data, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    split_indices = {"train": train_idx, "validation": val_idx}
    return (
        train_loader,
        val_loader,
        test_loader,
        tuple(train_data.classes),
        split_indices,
    )
