"""Cosmological distances and distance modulus."""

from __future__ import annotations

import numpy as np
from astropy.cosmology import FLRW

from lfkit.utils.types import Cosmology, FloatArray, FloatInput
from lfkit.utils.validators import validate_array

__all__ = [
    "comoving_distance_mpc",
    "distance_modulus",
    "distance_modulus_from_luminosity_distance",
    "luminosity_distance_mpc",
    "transverse_comoving_distance_mpc",
]


def comoving_distance_mpc(
    cosmo_obj: Cosmology,
    z: FloatInput,
) -> FloatArray:
    """Compute radial comoving distance in Mpc.

    Args:
        cosmo_obj: Astropy or PyCCL cosmology object.
        z: Non-negative redshift values.

    Returns:
        Radial comoving distances in Mpc.
    """
    z = validate_array(z, name="z", allow_negative=False)

    if isinstance(cosmo_obj, FLRW):
        distance = cosmo_obj.comoving_distance(z).to_value("Mpc")
    else:
        a = 1.0 / (1.0 + z)
        distance = cosmo_obj.comoving_radial_distance(a.ravel())

    return np.asarray(distance, dtype=float).reshape(z.shape)


def transverse_comoving_distance_mpc(
    cosmo_obj: Cosmology,
    z: FloatInput,
) -> FloatArray:
    """Compute transverse comoving distance in Mpc.

    Args:
        cosmo_obj: Astropy or PyCCL cosmology object.
        z: Non-negative redshift values.

    Returns:
        Transverse comoving distances in Mpc.
    """
    z = validate_array(z, name="z", allow_negative=False)

    if isinstance(cosmo_obj, FLRW):
        distance = cosmo_obj.comoving_transverse_distance(z).to_value("Mpc")
    else:
        a = 1.0 / (1.0 + z)
        distance = cosmo_obj.comoving_angular_distance(a.ravel())

    return np.asarray(distance, dtype=float).reshape(z.shape)


def luminosity_distance_mpc(
    cosmo_obj: Cosmology,
    z: FloatInput,
) -> FloatArray:
    """Compute luminosity distance in Mpc.

    Args:
        cosmo_obj: Astropy or PyCCL cosmology object.
        z: Non-negative redshift values.

    Returns:
        Luminosity distances in Mpc.
    """
    z = validate_array(z, name="z", allow_negative=False)

    if isinstance(cosmo_obj, FLRW):
        distance = cosmo_obj.luminosity_distance(z).to_value("Mpc")
    else:
        a = 1.0 / (1.0 + z)
        distance = cosmo_obj.luminosity_distance(a.ravel())

    return np.asarray(distance, dtype=float).reshape(z.shape)


def distance_modulus_from_luminosity_distance(
    luminosity_distance: FloatInput,
    h: float | None = None,
) -> FloatArray:
    r"""Compute distance modulus from luminosity distance.

    The distance modulus is

    .. math::

        \mu = 5 \log_{10}(D_L / \mathrm{Mpc}) + 25.

    Args:
        luminosity_distance: Luminosity distance in Mpc.
        h: Optional dimensionless Hubble parameter used to rescale
            the luminosity distance.

    Returns:
        Distance modulus values.
    """
    distance = validate_array(
        luminosity_distance,
        name="luminosity_distance",
        allow_negative=False,
    )

    if h is not None:
        if not np.isfinite(h) or h <= 0.0:
            raise ValueError("h must be finite and positive.")
        distance = distance * h

    with np.errstate(divide="ignore"):
        return np.asarray(5.0 * np.log10(distance) + 25.0, dtype=float)


def distance_modulus(
    cosmo_obj: Cosmology,
    z: FloatInput,
    h: float | None = None,
) -> FloatArray:
    """Compute distance modulus for a cosmology.

    Args:
        cosmo_obj: Astropy or PyCCL cosmology object.
        z: Non-negative redshift values.
        h: Optional dimensionless Hubble parameter used to rescale
            the luminosity distance.

    Returns:
        Distance modulus values.
    """
    return distance_modulus_from_luminosity_distance(
        luminosity_distance_mpc(cosmo_obj, z),
        h=h,
    )
