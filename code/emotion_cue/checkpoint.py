"""Release checkpoint metadata validation shared by evaluation and the app."""

from __future__ import annotations


REQUIRED_CHECKPOINT_KEYS = frozenset(
    {"state_dict", "dataset", "variant", "class_names"}
)


def validate_checkpoint_metadata(
    checkpoint: dict,
    *,
    expected_dataset: str | None = None,
) -> tuple[str, ...]:
    missing = sorted(REQUIRED_CHECKPOINT_KEYS.difference(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint is missing metadata: {missing}")
    if expected_dataset is not None and checkpoint["dataset"] != expected_dataset:
        raise ValueError(
            f"Checkpoint dataset {checkpoint['dataset']!r} does not match "
            f"requested dataset {expected_dataset!r}"
        )
    class_names = tuple(checkpoint["class_names"])
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("Checkpoint class_names must be non-empty and unique")
    from .constants import CLASS_NAMES
    if class_names != CLASS_NAMES:
        raise ValueError('Checkpoint class order differs from the seven-class release contract')
    if checkpoint["variant"] not in {"swin", "gated"}:
        raise ValueError(f"Unknown checkpoint variant: {checkpoint['variant']!r}")
    return class_names
