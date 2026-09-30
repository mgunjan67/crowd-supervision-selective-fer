import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from PIL import Image

from emotion_cue.data import build_transform
from emotion_cue.model import ChannelGate, SwinClassifier


def test_channel_gate_preserves_vector_shape():
    gate = ChannelGate(32, reduction=8)
    features = torch.randn(3, 32)
    assert gate(features).shape == features.shape


def test_model_output_shape_without_downloading_weights():
    model = SwinClassifier(num_classes=7, gated=True, pretrained=False)
    model.eval()
    with torch.no_grad():
        # A small divisible image keeps the architecture-contract test fast on CPU.
        # The release transform's 224 x 224 production size is tested separately.
        output = model(torch.randn(1, 3, 32, 32))
    assert output.shape == (1, 7)


def test_runtime_transform_has_production_shape():
    image = Image.new("RGB", (19, 27), "white")
    tensor = build_transform("FER2013", runtime=True)(image)
    assert tensor.shape == (3, 224, 224)
