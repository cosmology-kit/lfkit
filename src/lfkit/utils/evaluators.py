"""Numerical parameter evaluation utilities."""

from __future__ import annotations

from typing import Literal

import numpy as np

from lfkit.utils.types import ConditionalParameter, FloatArray, FloatInput
from lfkit.utils.validators import validate_array

__all__ = [
    "evaluate_parameter",
]


def evaluate_parameter(
    parameter: ConditionalParameter,
    *coordinates: FloatInput,
    name: str = "parameter",
    domain: Literal["real", "positive", "non_negative"] = "real",
) -> FloatArray:
    """Evaluate a numerical parameter or callable on broadcastable coordinates.

    Args:
        parameter: Numerical value or callable accepting the coordinates.
        *coordinates: Independent variables passed to the callable.
        name: Parameter name used in error messages.
        domain: Allowed domain of the evaluated values.

    Returns:
        Finite float64 values broadcast to the common coordinate shape.
        If no coordinates are supplied, the evaluated shape is preserved.

    Raises:
        ValueError: If the domain is unsupported, the values are non-finite,
            the output shape is incompatible, or the domain is violated.
    """
    if domain not in ("real", "positive", "non_negative"):
        raise ValueError(f"Unsupported domain: {domain!r}.")

    arrays = tuple(
        validate_array(coordinate, name=f"coordinate {i}")
        for i, coordinate in enumerate(coordinates)
    )

    values = parameter(*arrays) if callable(parameter) else parameter
    values_arr = validate_array(values, name=name)

    if arrays:
        shape = np.broadcast_shapes(*(arr.shape for arr in arrays))

        if values_arr.shape != shape:
            try:
                values_arr = np.broadcast_to(values_arr, shape)
            except ValueError as exc:
                raise ValueError(
                    f"{name} must return values broadcastable to shape {shape}."
                ) from exc

    if domain == "positive" and np.any(values_arr <= 0.0):
        raise ValueError(f"{name} must be positive.")

    if domain == "non_negative" and np.any(values_arr < 0.0):
        raise ValueError(f"{name} must be non-negative.")

    return values_arr
