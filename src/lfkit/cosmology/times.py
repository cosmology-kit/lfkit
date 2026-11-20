"""Cosmological lookback times and redshift inversion."""

from __future__ import annotations

import numpy as np
from astropy.cosmology import FLRW

from lfkit.utils.types import Cosmology, FloatArray, FloatInput
from lfkit.utils.validators import (
    validate_array,
    validate_strictly_increasing_1d,
)

__all__ = [
    "lookback_time_gyr",
    "z_from_lookback_time",
]


def lookback_time_gyr(
    cosmo_obj: Cosmology,
    z: FloatInput,
) -> FloatArray:
    r"""Compute cosmological lookback time in gigayears.

    The lookback time is defined as

    .. math::

        t_{\mathrm{lb}}(z)
        =
        \int_0^z
        \frac{\mathrm{d}z'}{(1+z')H(z')}.

    Args:
        cosmo_obj: Astropy or PyCCL cosmology object.
        z: Non-negative redshift values.

    Returns:
        Lookback time values in Gyr with the same shape as the input.
    """
    z = validate_array(z, name="z", allow_negative=False)

    if isinstance(cosmo_obj, FLRW):
        time = cosmo_obj.lookback_time(z).to_value("Gyr")
    else:
        a = 1.0 / (1.0 + z)
        time = cosmo_obj.lookback_time(a.ravel())

    time = np.asarray(time, dtype=float).reshape(z.shape)

    # Enforce the exact present-day boundary condition.
    return np.where(z == 0.0, 0.0, time)


def z_from_lookback_time(
    cosmo_obj: Cosmology,
    t_lb_gyr: FloatInput,
    *,
    zmax: float = 20.0,
    nz: int = 4096,
) -> FloatArray:
    """Invert cosmological lookback time using linear interpolation.

    Args:
        cosmo_obj: Astropy or PyCCL cosmology object.
        t_lb_gyr: Non-negative lookback time values in Gyr.
        zmax: Maximum redshift of the inversion grid.
        nz: Number of samples used to construct the inversion grid.

    Returns:
        Redshift values with the same shape as the input lookback times.

    Raises:
        ValueError: If the grid configuration is invalid, the lookback
            time grid is not strictly increasing, or the requested
            lookback times fall outside the inversion range.
    """
    t_lb = validate_array(
        t_lb_gyr,
        name="t_lb_gyr",
        allow_negative=False,
    )

    if not np.isfinite(zmax) or zmax <= 0.0:
        raise ValueError("zmax must be finite and positive.")

    if (
        isinstance(nz, (bool, np.bool_))
        or not isinstance(nz, (int, np.integer))
        or nz < 32
    ):
        raise ValueError("nz must be an integer greater than or equal to 32.")

    z_grid = np.linspace(0.0, zmax, nz)

    t_grid = validate_strictly_increasing_1d(
        lookback_time_gyr(cosmo_obj, z_grid),
        name="lookback time grid",
    )

    if np.any(t_lb < t_grid[0]) or np.any(t_lb > t_grid[-1]):
        raise ValueError(
            "Target lookback time outside inversion range. "
            "Increase zmax or check inputs."
        )

    return np.asarray(
        np.interp(t_lb, t_grid, z_grid),
        dtype=float,
    )
