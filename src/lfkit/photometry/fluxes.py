"""Photometric flux and magnitude conversion utilities."""

from __future__ import annotations

import numpy as np

from lfkit.utils.types import FloatArray, FloatInput

__all__ = [
    "mag_to_maggies",
    "magerr_to_ivar_maggies",
    "maggies_to_mag",
]


def mag_to_maggies(m_ab: FloatInput) -> FloatArray:
    """Converts AB magnitudes to maggies (linear flux units).

    In the AB system:

        m_AB = -2.5 * log10(f_nu) - 48.6

    In kcorrect-style workflows, fluxes are expressed in *maggies*:

        f_maggies = 10**(-0.4 * m_AB)

    where 1 maggie corresponds to m_AB = 0.

    Args:
        m_ab: AB magnitudes. Can be scalar-like or array-like.

    Returns:
        Fluxes in maggies with the same shape as the input.

    Notes:
        This is a purely algebraic transformation with no cosmology
        dependence. The returned values are linear flux densities in
        arbitrary units normalized such that m_AB = 0 corresponds to
        f = 1 maggie.
    """
    m = np.asarray(m_ab, dtype=float)
    return 10.0 ** (-0.4 * m)


def maggies_to_mag(maggies: FloatInput, *, floor: float = 1e-300) -> FloatArray:
    """Convert maggies to magnitudes with a safety floor.

    Args:
        maggies: Fluxes in maggies.
        floor: Minimum value to avoid log(0).

    Returns:
        Magnitudes in AB system.
    """
    m = np.asarray(maggies, dtype=float)
    m = np.maximum(m, float(floor))
    return -2.5 * np.log10(m)


def magerr_to_ivar_maggies(m_ab: FloatInput, sigma_m: FloatInput) -> FloatArray:
    """Converts magnitude uncertainties to inverse variance in maggies.

    Flux uncertainties are computed using first-order error propagation.
    Given:

        f = 10**(-0.4 * m)

    the derivative is:

        df/dm = -0.4 * ln(10) * f

    so the propagated flux uncertainty is approximated as:

        sigma_f = (0.4 * ln(10)) * f * sigma_m

    and the inverse variance is:

        ivar = 1 / sigma_f**2

    Args:
        m_ab: AB magnitudes.
        sigma_m: 1σ magnitude uncertainties corresponding to `m_ab`.

    Returns:
        Inverse variance of the fluxes in maggies (1 / sigma_f**2).
        Entries are set to zero where the propagated uncertainty is
        non-finite or non-positive.

    Notes:
        Uses linear (first-order) error propagation and assumes small
        magnitude errors. No covariance between bands is included.
        Non-finite or zero uncertainties yield zero inverse variance.
    """
    m = np.asarray(m_ab, dtype=float)
    sm = np.asarray(sigma_m, dtype=float)
    f = mag_to_maggies(m)
    sigma_f = (0.4 * np.log(10.0)) * f * sm
    ivar = np.zeros_like(sigma_f)
    ok = np.isfinite(sigma_f) & (sigma_f > 0)
    ivar[ok] = 1.0 / (sigma_f[ok] ** 2)
    return ivar
