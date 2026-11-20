"""Shared typing aliases for LFKit."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ConditionalParameter",
    "Cosmology",
    "FloatArray",
    "FloatInput",
    "LuminosityFunction",
    "ParameterModel",
    "ParameterValue",
]


FloatArray = NDArray[np.float64]
FloatInput = float | Sequence[float] | FloatArray

ParameterValue = FloatInput
ParameterModel = Callable[..., ParameterValue]
ConditionalParameter = ParameterValue | ParameterModel

LuminosityFunction = Callable[..., FloatArray]

Cosmology = Any
