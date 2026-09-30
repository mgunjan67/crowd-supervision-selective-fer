import ast
import csv
from pathlib import Path

from emotion_cue.constants import CLASS_NAMES, PREDICTION_INTERVAL_SECONDS


ROOT = Path(__file__).parents[1]
PACKAGE_ROOT = ROOT.parent


def test_shared_class_order_is_canonical():
    assert CLASS_NAMES == (
        "angry",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise",
    )


def test_runtime_update_interval_matches_manuscript_contract():
    assert PREDICTION_INTERVAL_SECONDS == 1.0


def test_example_config_uses_correct_spelling_and_keys():
    source = (ROOT / "config.example.yaml").read_text(encoding="utf-8")
    assert "datasets:" in source
    assert "epochs:" in source
    assert "validation_fraction:" in source
    assert "dateset" not in source
    assert "num_epochs" not in source


def test_app_contains_no_unvalidated_branch():
    source = (ROOT / "app.py").read_text(encoding="utf-8").lower()
    for prohibited in ("vader", "dissonance", "retinaface", "clahe"):
        assert prohibited not in source
    assert "attach the current facial-expression cue" in source


def test_model_forward_order_is_backbone_gate_classifier():
    source = (ROOT / "emotion_cue" / "model.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    classifier = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SwinClassifier"
    )
    forward = next(
        node
        for node in classifier.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    calls = [
        node.func.attr
        for node in ast.walk(forward)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert calls.index("backbone") < calls.index("channel_gate") < calls.index(
        "classifier"
    )


def test_checkpoint_metadata_is_required_by_evaluator():
    source = (ROOT / "evaluate.py").read_text(encoding="utf-8")
    assert "validate_checkpoint_metadata" in source


def test_reported_results_are_unique_and_expected():
    rows = list(
        csv.DictReader(
            (PACKAGE_ROOT / "reported_results.csv").open(encoding="utf-8")
        )
    )
    keys = [(row["dataset"], row["variant"], row["optimizer"]) for row in rows]
    assert len(keys) == len(set(keys)) == 5
    values = {float(row["weighted_f1"]) for row in rows}
    assert values == {0.7123, 0.7677, 0.8474, 0.8971, 0.9228}
