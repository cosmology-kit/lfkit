"""Unit tests for ``lfkit.utils.units``."""

from __future__ import annotations

import numpy as np

from lfkit.utils.units import (
    h0_km_s_mpc_to_gyr_inv,
    km_per_mpc,
    sec_per_gyr,
)


def test_km_per_mpc_is_positive_and_reasonable():
    """Tests that km_per_mpc returns a positive value consistent with the IAU AU + parsec definition."""
    val = km_per_mpc()
    assert np.isfinite(val)
    assert val > 0.0

    # Use the same defining relation:
    # 1 pc = 1 AU / tan(1 arcsec) ≈ 1 AU / (1 arcsec in radians)
    # with 1 arcsec = pi / 648000 rad, so pc = AU * 648000 / pi exactly under this convention.
    au_m = 149_597_870_700.0  # IAU exact AU in meters
    arcsec_rad = np.pi / (180.0 * 3600.0)  # pi / 648000
    pc_m = au_m / arcsec_rad
    expected = (1e6 * pc_m) / 1000.0  # Mpc in km

    assert np.isclose(val, expected, rtol=0.0, atol=0.0)


def test_sec_per_gyr_is_positive_and_reasonable():
    """Tests that sec_per_gyr returns a positive value equal to 86400*365.25*1e9."""
    val = sec_per_gyr()
    assert np.isfinite(val)
    assert val > 0.0
    assert np.isclose(val, 86400.0 * 365.25 * 1e9, rtol=0.0, atol=0.0)


def test_h0_conversion_matches_manual_formula():
    """Tests that h0_km_s_mpc_to_gyr_inv matches h0/km_per_mpc*sec_per_gyr."""
    h0 = 70.0
    expected = (h0 / km_per_mpc()) * sec_per_gyr()
    got = h0_km_s_mpc_to_gyr_inv(h0)
    assert np.isfinite(got)
    assert np.isclose(got, expected, rtol=0.0, atol=0.0)


def test_h0_conversion_accepts_scalar_like_inputs() -> None:
    """Tests that H0 conversion accepts NumPy scalar inputs."""
    h0 = np.float64(100.0)
    got = h0_km_s_mpc_to_gyr_inv(h0)
    expected = (100.0 / km_per_mpc()) * sec_per_gyr()
    assert np.isclose(got, expected, rtol=0.0, atol=0.0)


def test_h0_conversion_preserves_sign() -> None:
    """Tests that H0 conversion is algebraic and preserves sign."""
    assert h0_km_s_mpc_to_gyr_inv(0.0) == 0.0
    assert h0_km_s_mpc_to_gyr_inv(-70.0) < 0.0
