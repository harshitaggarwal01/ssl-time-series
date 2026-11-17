"""Self-supervised time-series encoders for the Stable Challenge."""

from .constants import BATCH, CHANNELS, CONTEXT, MAX_EMBED_DIM, OPSET
from .model import SeriesEncoder

__all__ = [
    "BATCH",
    "CHANNELS",
    "CONTEXT",
    "MAX_EMBED_DIM",
    "OPSET",
    "SeriesEncoder",
]
