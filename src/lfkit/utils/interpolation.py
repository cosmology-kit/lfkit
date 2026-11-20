"""Interpolation utilities for one-dimensional tabulated curves."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator

from lfkit.utils.types import FloatArray, FloatInput

InterpMethod = Literal["pchip", "akima", "linear"]
ExtrapMode = Literal["none", "native", "linear_tail"]
Interpolator = (
    PchipInterpolator | Akima1DInterpolator | Callable[[FloatInput], FloatArray]
)

__all__ = [
    "as_1d_finite_grid",
    "build_1d_interpolator",
    "linear_interp_extrap",
    "prep_strictly_increasing_xy",
]


def linear_interp_extrap(
    x: FloatInput,
    xp: FloatInput,
    fp: FloatInput,
) -> FloatArray:
    """Interpolate linearly with linear extrapolation.

    Args:
        x: Query points.
        xp: Strictly increasing sample locations.
        fp: Sample values.

    Returns:
        Interpolated values with the shape of the query points.
    """
    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)

    if xp.ndim != 1 or xp.shape != fp.shape or xp.size == 0:
        raise ValueError("xp and fp must be matching non-empty 1D arrays.")

    if np.any(~np.isfinite(xp)) or np.any(~np.isfinite(fp)):
        raise ValueError("xp and fp must contain only finite values.")

    if np.any(np.diff(xp) <= 0.0):
        raise ValueError("xp must be strictly increasing.")

    values = np.interp(x, xp, fp)

    if xp.size == 1:
        return np.asarray(values, dtype=float)

    return _apply_linear_tails(x, values, xp, fp)


def _apply_linear_tails(
    x: FloatArray,
    values: FloatArray,
    xp: FloatArray,
    fp: FloatArray,
) -> FloatArray:
    """Extend interpolated values using endpoint secant slopes."""
    left_slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
    right_slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])

    result = np.where(
        x < xp[0],
        fp[0] + left_slope * (x - xp[0]),
        values,
    )

    result = np.where(
        x > xp[-1],
        fp[-1] + right_slope * (x - xp[-1]),
        result,
    )

    return np.asarray(result, dtype=float)


def build_1d_interpolator(
    z: FloatInput,
    y: FloatInput,
    *,
    method: InterpMethod,
    extrapolate: bool,
    extrap_mode: ExtrapMode = "native",
) -> Interpolator:
    """Build an interpolator for one-dimensional tabulated data.

    Args:
        z: Sample locations.
        y: Sample values.
        method: Linear, PCHIP or Akima interpolation.
        extrapolate: Whether to permit extrapolation.
        extrap_mode: Native extrapolation, linear endpoint tails,
            or no extrapolation.

    Returns:
        A callable interpolator. Without extrapolation, linear
        interpolation clamps to the endpoints, while PCHIP and
        Akima return NaN outside the tabulated range.

    Raises:
        ValueError: If the method, extrapolation mode or input
            tabulation is invalid.
    """
    if method not in ("linear", "pchip", "akima"):
        raise ValueError(f"Unknown method={method!r}.")

    if extrap_mode not in ("none", "native", "linear_tail"):
        raise ValueError(f"Unknown extrap_mode={extrap_mode!r}.")

    z, y = prep_strictly_increasing_xy(z, y)

    native = extrapolate and extrap_mode == "native"
    linear_tail = extrapolate and extrap_mode == "linear_tail"

    if method == "linear":

        def interpolator(x: FloatInput) -> FloatArray:
            x_arr = np.asarray(x, dtype=float)
            values = np.interp(x_arr, z, y)

            if native or linear_tail:
                return _apply_linear_tails(x_arr, values, z, y)

            return np.asarray(values, dtype=float)

        return interpolator

    if method == "pchip":
        base = PchipInterpolator(z, y, extrapolate=native)
    else:
        base = Akima1DInterpolator(z, y, extrapolate=native)

    if not linear_tail:
        return base

    def interpolator_with_tails(x: FloatInput) -> FloatArray:
        x_arr = np.asarray(x, dtype=float)
        return _apply_linear_tails(x_arr, base(x_arr), z, y)

    return interpolator_with_tails


def prep_strictly_increasing_xy(
    z: FloatInput,
    y: FloatInput,
) -> tuple[FloatArray, FloatArray]:
    """Prepare tabulated data for one-dimensional interpolation.

    Args:
        z: Sample locations.
        y: Sample values.

    Returns:
        Sorted, finite sample pairs with strictly increasing locations.
        The first valid occurrence of each duplicate location is retained.

    Raises:
        ValueError: If the inputs are not matching one-dimensional arrays
            or fewer than two distinct finite samples remain.
    """
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)

    if z.ndim != 1 or z.shape != y.shape:
        raise ValueError("z and y must be matching 1D arrays.")

    valid = np.isfinite(z) & np.isfinite(y)
    z = z[valid]
    y = y[valid]

    order = np.argsort(z, kind="stable")
    z = z[order]
    y = y[order]

    keep = np.ones(z.size, dtype=bool)
    keep[1:] = z[1:] > z[:-1]

    z = z[keep]
    y = y[keep]

    if z.size < 2:
        raise ValueError("Need at least 2 points to build an interpolator.")

    return z, y


def as_1d_finite_grid(
    z_grid: FloatInput,
    *,
    name: str = "z_grid",
) -> FloatArray:
    """Convert input into a finite one-dimensional grid."""
    z = np.asarray(z_grid, dtype=float)

    if z.ndim != 1 or z.size < 2 or np.any(~np.isfinite(z)):
        raise ValueError(f"{name} must be a finite 1D array with >=2 points.")

    return z
