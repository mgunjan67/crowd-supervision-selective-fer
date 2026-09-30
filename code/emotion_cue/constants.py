"""Shared public constants used by training, evaluation, and the prototype."""

CLASS_NAMES = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)

MODEL_VARIANTS = ("swin", "gated")
OPTIMIZERS = ("adamw", "sam")
IMAGE_SIZE = 224
PREDICTION_INTERVAL_SECONDS = 1.0
