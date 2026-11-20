"""Cosmological expansion and comoving volume."""

from __future__ import annotations

import numpy as np
from astropy.cosmology import FLRW

from lfkit.cosmology.distances import transverse_comoving_distance_mpc
from lfkit.utils.types import Cosmology, FloatArray, FloatInput
from lfkit.utils.validators import validate_array

__all__ = [
    "C_KM_S",
    "differential_comoving_volume",
    "dimensionless_hubble_parameter",
    "expansion_rate",
    "hubble_parameter",
]

C_KM_S = 299792.458


def dimensionless_hubble_parameter(
    cosmo_obj: Cosmology,
) -> float:
    """Return the dimensionless Hubble parameter h.

    Args:
        cosmo_obj: Astropy or PyCCL cosmology object.

    Returns:
        Dimensionless Hubble parameter.
    """
    if isinstance(cosmo_obj, FLRW):
        h = float(cosmo_obj.h)
    else:
        h = float(cosmo_obj["h"])

    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("h must be finite and positive.")

    return h


def expansion_rate(
    cosmo_obj: Cosmology,
    z: FloatInput,
) -> FloatArray:
    r"""Compute the dimensionless expansion rate.

    .. math::

        E(z) = \frac{H(z)}{H_0}.

    Args:
        cosmo_obj: Astropy or PyCCL cosmology object.
        z: Non-negative redshift values.

    Returns:
        Dimensionless expansion rate.
    """
    z = validate_array(z, name="z", allow_negative=False)

    if isinstance(cosmo_obj, FLRW):
        ez = cosmo_obj.efunc(z)
    else:
        a = 1.0 / (1.0 + z)
        ez = cosmo_obj.h_over_h0(a.ravel())

    return np.asarray(ez, dtype=float).reshape(z.shape)


def hubble_parameter(
    cosmo_obj: Cosmology,
    z: FloatInput,
) -> FloatArray:
    """Compute the Hubble parameter in km/s/Mpc.

    Args:
        cosmo_obj: Astropy or PyCCL cosmology object.
        z: Non-negative redshift values.

    Returns:
        Hubble parameter in km/s/Mpc.
    """
    h0 = 100.0 * dimensionless_hubble_parameter(cosmo_obj)
    return np.asarray(h0 * expansion_rate(cosmo_obj, z), dtype=float)


def differential_comoving_volume(
    cosmo_obj: Cosmology,
    z: FloatInput,
    frac_sky: float = 1.0,
) -> FloatArray:
    r"""Compute differential comoving volume per unit redshift.

    .. math::

        \frac{\mathrm{d}V}{\mathrm{d}z}
        = 4\pi f_{\mathrm{sky}}
        \frac{cD_M^2(z)}{H(z)}.

    Args:
        cosmo_obj: Astropy or PyCCL cosmology object.
        z: Non-negative redshift values.
        frac_sky: Fraction of sky covered by the survey, between
            zero and one.

    Returns:
        Differential comoving volume in Mpc cubed per unit redshift.
    """
    if not np.isfinite(frac_sky) or not 0.0 <= frac_sky <= 1.0:
        raise ValueError("frac_sky must be between zero and one.")

    dm = transverse_comoving_distance_mpc(cosmo_obj, z)
    hz = hubble_parameter(cosmo_obj, z)

    return np.asarray(
        4.0 * np.pi * frac_sky * C_KM_S * dm**2 / hz,
        dtype=float,
    )
