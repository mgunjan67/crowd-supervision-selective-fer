from pathlib import Path

import pytest

pytest.importorskip("yaml")

from emotion_cue.config import load_config


def test_example_config_uses_correct_keys():
    path = Path(__file__).parents[1] / "config.example.yaml"
    config = load_config(path)
    assert "FER2013" in config.datasets
    assert "RAF_DB_BASIC" in config.datasets
    assert config.epochs == 30
    assert config.validation_fraction == pytest.approx(0.1)


def test_missing_datasets_key_is_rejected(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("epochs: 3\n", encoding="utf-8")
    with pytest.raises(KeyError, match="datasets"):
        load_config(path)
