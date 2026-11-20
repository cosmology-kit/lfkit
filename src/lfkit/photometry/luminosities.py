r"""Luminosity and magnitude conversion utilities for LFKit.

This module provides lightweight helpers for converting between
magnitude differences and luminosity ratios, as well as common
Schechter-function quantities expressed in luminosity space.

The core convention is

.. math::

    L_1 / L_2 = 10^{-0.4 (M_1 - M_2)},

where ``M1`` and ``M2`` may be absolute or apparent magnitudes,
as long as they are defined in the same photometric system.

All returned quantities are NumPy arrays of dtype float.
"""

from __future__ import annotations

import numpy as np

from lfkit.utils.numerics import safe_power10
from lfkit.utils.types import FloatArray, FloatInput
from lfkit.utils.validators import validate_array

__all__ = (
    "luminosity_from_magnitude",
    "luminosity_ratio",
    "luminosity_ratio_from_magnitudes",
    "luminosity_weight_from_magnitude",
    "magnitude_difference_from_luminosity_ratio",
    "magnitude_from_luminosity",
)

__api_aliases__ = {
    "luminosity_ratio": "ratio",
    "luminosity_ratio_from_magnitudes": "ratio_from_magnitudes",
    "magnitude_difference_from_luminosity_ratio": "magnitude_difference_from_ratio",
    "luminosity_weight_from_magnitude": "weight_from_magnitude",
    "luminosity_from_magnitude": "from_magnitude",
    "magnitude_from_luminosity": "to_magnitude",
}


def luminosity_ratio(
    absolute_mag: FloatInput,
    m_star: FloatInput,
) -> FloatArray:
    r"""Return the luminosity ratio relative to the characteristic luminosity.

    For magnitudes,

    .. math::

        \frac{L}{L_\star} = 10^{-0.4 (M - M_\star)}.

    Args:
        absolute_mag: Absolute magnitude value(s).
        m_star: Characteristic magnitude value(s).

    Returns:
        NumPy array of luminosity ratios ``L / L_star``.
    """
    return luminosity_ratio_from_magnitudes(absolute_mag, m_star)


def luminosity_ratio_from_magnitudes(
    magnitude: FloatInput,
    ref_magnitude: FloatInput,
) -> FloatArray:
    r"""Return luminosity relative to a reference magnitude.

    This uses :math:`L / L_{\mathrm{ref}} = 10^{-0.4 (m - m_{\mathrm{ref}})}`.

    Args:
        magnitude: Magnitude value(s).
        ref_magnitude: Reference magnitude value(s).

    Returns:
        NumPy array of luminosity ratios :math:`L / L_{\mathrm{ref}}`.
    """
    mag_1 = validate_array(magnitude, name="magnitude")
    mag_2 = validate_array(ref_magnitude, name="ref_magnitude")

    ratio = safe_power10(-0.4 * (mag_1 - mag_2))

    return np.asarray(ratio, dtype=float)


def magnitude_difference_from_luminosity_ratio(
    ratio: FloatInput,
) -> FloatArray:
    r"""Return the magnitude difference corresponding to a luminosity ratio.

    This uses

    .. math::

        m_1 - m_2 = -2.5 \log_{10}(L_1 / L_2).

    Args:
        ratio: Luminosity ratio value(s) ``L1 / L2``.

    Returns:
        NumPy array of magnitude differences ``m1 - m2``.

    Raises:
        ValueError: If any luminosity ratio is not strictly positive.
    """
    ratio_arr = validate_array(ratio, name="luminosity_ratio")

    if np.any(ratio_arr <= 0):
        raise ValueError("luminosity_ratio must be strictly positive.")

    return np.asarray(-2.5 * np.log10(ratio_arr), dtype=float)


def luminosity_weight_from_magnitude(
    magnitude: FloatInput,
    reference_magnitude: FloatInput = 0.0,
) -> FloatArray:
    r"""Return an unnormalized luminosity weight from a magnitude.

    This uses

    .. math::

        L \propto 10^{-0.4 M}.

    Args:
        magnitude: Magnitude value(s).
        reference_magnitude: Reference zero-point magnitude for the
            proportionality constant.

    Returns:
        NumPy array proportional to luminosity.
    """
    return luminosity_ratio_from_magnitudes(magnitude, reference_magnitude)


def luminosity_from_magnitude(
    magnitude: FloatInput,
    *,
    reference_magnitude: FloatInput = 0.0,
    reference_luminosity: float = 1.0,
) -> FloatArray:
    r"""Return luminosity corresponding to a magnitude relative to a reference.

    This uses

    .. math::

        L = L_{\mathrm{ref}} \, 10^{-0.4 (m - m_{\mathrm{ref}})}.

    Args:
        magnitude: Magnitude value(s).
        reference_magnitude: Reference magnitude ``m_ref``.
        reference_luminosity: Reference luminosity ``L_ref``.

    Returns:
        NumPy array of luminosities in the same units as
        ``reference_luminosity``.

    Raises:
        ValueError: If ``reference_luminosity`` is not strictly positive.
    """
    if not np.isfinite(reference_luminosity) or reference_luminosity <= 0:
        raise ValueError("reference_luminosity must be finite and strictly positive.")

    ratio = luminosity_ratio_from_magnitudes(
        magnitude,
        reference_magnitude,
    )
    return np.asarray(reference_luminosity * ratio, dtype=float)


def magnitude_from_luminosity(
    luminosity: FloatInput,
    *,
    reference_magnitude: FloatInput = 0.0,
    reference_luminosity: float = 1.0,
) -> FloatArray:
    r"""Return magnitude corresponding to a luminosity relative to a reference.

    This uses

    .. math::

        M = M_{\mathrm{ref}}
        - 2.5 \log_{10}
        \left(\frac{L}{L_{\mathrm{ref}}}\right).

    Args:
        luminosity: Luminosity value(s) in the same units as the reference.
        reference_magnitude: Reference magnitude value(s).
        reference_luminosity: Reference luminosity.

    Returns:
        NumPy array of magnitudes.

    Raises:
        ValueError: If luminosity or reference luminosity is non-finite
            or non-positive, or if reference magnitude is non-finite.
    """
    lum = validate_array(luminosity, name="luminosity")
    ref_mag = validate_array(reference_magnitude, name="reference_magnitude")

    if np.any(lum <= 0.0):
        raise ValueError("luminosity must be strictly positive.")

    if not np.isfinite(reference_luminosity) or reference_luminosity <= 0.0:
        raise ValueError("reference_luminosity must be finite and strictly positive.")

    return np.asarray(
        ref_mag - 2.5 * (np.log10(lum) - np.log10(reference_luminosity)),
        dtype=float,
    )
