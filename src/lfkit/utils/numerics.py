"""Shared numerical utilities."""

from __future__ import annotations

import numpy as np

from lfkit.utils.types import FloatArray, FloatInput
from lfkit.utils.validators import validate_array

__all__ = [
    "safe_divide",
    "safe_power10",
]


def safe_divide(
    numerator: FloatInput,
    denominator: FloatInput,
    *,
    fill_value: float = 0.0,
) -> FloatArray:
    """Divide arrays safely using strictly positive denominators.

    Args:
        numerator: Values in the numerator.
        denominator: Values in the denominator.
        fill_value: Value returned where the denominator is non-positive or where
            the division would produce a non-finite result.

    Returns:
        Broadcasted division result with invalid entries replaced by
        ``fill_value``.
    """
    numerator_arr, denominator_arr = np.broadcast_arrays(
        np.asarray(numerator, dtype=float),
        np.asarray(denominator, dtype=float),
    )

    result = np.full(numerator_arr.shape, fill_value, dtype=float)

    valid = denominator_arr > 0.0
    np.divide(
        numerator_arr,
        denominator_arr,
        out=result,
        where=valid,
    )

    result = np.where(np.isfinite(result), result, fill_value)
    return result


def safe_power10(
    exponent: FloatInput,
    *,
    min_exponent: float = -300.0,
    max_exponent: float = 300.0,
) -> FloatArray:
    """Return clipped base-10 powers without overflow warnings."""
    exponent_arr = validate_array(exponent, name="exponent")
    clipped = np.clip(exponent_arr, min_exponent, max_exponent)
    return np.asarray(10.0**clipped, dtype=float)
