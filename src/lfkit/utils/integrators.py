"""Numerical integration utilities.

This module provides small reusable helpers for integrating tabulated values
over fixed or variable finite bounds. These helpers do not encode any
luminosity function, photometry, or cosmology assumptions.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.integrate import trapezoid

from lfkit.utils.evaluators import evaluate_parameter
from lfkit.utils.types import FloatArray, FloatInput
from lfkit.utils.validators import validate_array

__all__ = [
    "integrate_between_variable_bounds",
]


def integrate_between_variable_bounds(
    y: FloatInput,
    *,
    lower: FloatInput,
    upper: FloatInput,
    integrand_fn: Callable[[FloatArray, FloatArray], FloatArray],
    n_grid: int = 512,
    y_name: str = "y",
    lower_name: str = "lower",
    upper_name: str = "upper",
    n_grid_name: str = "n_grid",
) -> FloatArray:
    """Integrate a callable between finite bounds that may vary with ``y``.

    Args:
        y: Coordinate values at which to evaluate the integral.
        lower: Lower integration bound. May be scalar or array-like.
        upper: Upper integration bound. May be scalar or array-like.
        integrand_fn: Callable evaluated as ``integrand_fn(x_grid, y_grid)``.
        n_grid: Number of integration points.
        y_name: Name used for ``y`` in error messages.
        lower_name: Name used for ``lower`` in error messages.
        upper_name: Name used for ``upper`` in error messages.
        n_grid_name: Name used for ``n_grid`` in error messages.

    Returns:
        Integral values with the broadcast shape of ``y``, ``lower``, and
        ``upper``.
    """
    if n_grid < 2:
        raise ValueError(f"{n_grid_name} must be at least 2")

    y_arr = validate_array(y, name=y_name)

    lower_arr = validate_array(lower, name=lower_name)
    upper_arr = validate_array(upper, name=upper_name)

    y_arr, lower_arr, upper_arr = np.broadcast_arrays(y_arr, lower_arr, upper_arr)

    width = upper_arr - lower_arr
    empty = width <= 0.0

    t_grid = np.linspace(0.0, 1.0, n_grid, dtype=float)
    grid_shape = (n_grid,) + (1,) * lower_arr.ndim

    x_grid = lower_arr[None, ...] + t_grid.reshape(grid_shape) * width[None, ...]
    y_grid = np.broadcast_to(y_arr[None, ...], x_grid.shape)

    values = evaluate_parameter(
        integrand_fn,
        x_grid,
        y_grid,
        name="integrand_fn",
    )

    result = trapezoid(values, x=x_grid, axis=0)
    return np.where(empty, 0.0, result)
