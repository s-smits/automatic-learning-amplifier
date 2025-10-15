"""Model training and evaluation utilities."""

from .comparison import compare_anthropic, compare_initial
from .training import train_model

__all__ = ["compare_anthropic", "compare_initial", "train_model"]
