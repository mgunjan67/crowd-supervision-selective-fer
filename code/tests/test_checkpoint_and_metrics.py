import pytest

from emotion_cue.checkpoint import validate_checkpoint_metadata


def test_checkpoint_rejects_missing_metadata():
    with pytest.raises(ValueError, match="missing metadata"):
        validate_checkpoint_metadata({"state_dict": {}})


def test_checkpoint_rejects_dataset_mismatch():
    checkpoint = {
        "state_dict": {},
        "dataset": "FER2013",
        "variant": "gated",
        "class_names": ["angry", "happy"],
    }
    with pytest.raises(ValueError, match="does not match"):
        validate_checkpoint_metadata(checkpoint, expected_dataset="RAF_DB_BASIC")


def test_metric_labels_are_explicit():
    pytest.importorskip("sklearn")
    from emotion_cue.metrics import classification_metrics

    result = classification_metrics(
        [0, 0, 1, 1], [0, 1, 1, 1], ("negative", "positive")
    )
    assert set(result) == {
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "per_class",
        "confusion_matrix",
    }
