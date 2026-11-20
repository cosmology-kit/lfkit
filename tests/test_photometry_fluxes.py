"""Unit tests for ``lfkit.photometry.fluxes``."""

from __future__ import annotations

import numpy as np
import pytest

from lfkit.photometry.fluxes import (
    mag_to_maggies,
    magerr_to_ivar_maggies,
    maggies_to_mag,
)


def test_mag_to_maggies_known_points_and_roundtrip():
    """Tests that mag_to_maggies maps m=0 to 1 and roundtrips with maggies_to_mag."""
    m = np.array([0.0, 2.5, 10.0])
    f = mag_to_maggies(m)
    assert np.all(np.isfinite(f))
    assert np.isclose(f[0], 1.0, rtol=0.0, atol=0.0)
    assert np.isclose(f[1], 0.1, rtol=0.0, atol=1e-15)  # 10**(-1)

    m_back = maggies_to_mag(f)
    assert np.all(np.isfinite(m_back))
    assert np.allclose(m_back, m, rtol=0.0, atol=1e-12)


def test_maggies_to_mag_floor_prevents_infs():
    """Tests that maggies_to_mag applies a floor so zero/negative fluxes yield finite magnitudes."""
    f = np.array([0.0, -1.0, 1e-320, 1.0])
    m = maggies_to_mag(f, floor=1e-300)
    assert np.all(np.isfinite(m))
    # Everything <= floor maps to the same magnitude
    assert np.isclose(m[0], m[2], rtol=0.0, atol=0.0)
    assert np.isclose(m[1], m[2], rtol=0.0, atol=0.0)


def test_magerr_to_ivar_maggies_matches_propagation_and_masks_bad():
    """Tests that magerr_to_ivar_maggies matches first-order propagation and returns 0 for non-finite/invalid errors."""
    m = np.array([20.0, 21.0, 22.0])
    sm = np.array([0.1, np.nan, 0.0])

    ivar = magerr_to_ivar_maggies(m, sm)

    f = mag_to_maggies(m)
    sigma_f = (0.4 * np.log(10.0)) * f * sm
    expected = np.zeros_like(sigma_f)
    ok = np.isfinite(sigma_f) & (sigma_f > 0)
    expected[ok] = 1.0 / (sigma_f[ok] ** 2)

    assert np.all(np.isfinite(ivar))
    assert np.allclose(ivar, expected, rtol=0.0, atol=0.0)
    assert ivar[1] == 0.0  # nan sigma_m -> 0 ivar
    assert ivar[2] == 0.0  # zero sigma_m -> 0 ivar


def test_mag_to_maggies_accepts_scalar_input() -> None:
    """Tests that scalar magnitude input returns scalar-shaped maggies."""
    got = mag_to_maggies(0.0)
    assert np.shape(got) == ()
    assert np.isclose(got, 1.0, rtol=0.0, atol=0.0)


def test_maggies_to_mag_accepts_scalar_input() -> None:
    """Tests that scalar maggie input returns scalar-shaped magnitude."""
    got = maggies_to_mag(1.0)
    assert np.shape(got) == ()
    assert np.isclose(got, 0.0, rtol=0.0, atol=0.0)


def test_mag_to_maggies_preserves_array_shape() -> None:
    """Tests that magnitude-to-maggies conversion preserves input shape."""
    m = np.array([[0.0, 2.5], [5.0, 7.5]])
    f = mag_to_maggies(m)

    assert f.shape == m.shape
    np.testing.assert_allclose(f, 10.0 ** (-0.4 * m))


def test_maggies_to_mag_preserves_array_shape() -> None:
    """Tests that maggies-to-magnitude conversion preserves input shape."""
    f = np.array([[1.0, 0.1], [0.01, 0.001]])
    m = maggies_to_mag(f)

    assert m.shape == f.shape
    np.testing.assert_allclose(m, -2.5 * np.log10(f))


def test_magnitude_difference_maps_to_flux_ratio() -> None:
    """Tests that a 2.5 mag increase lowers maggies by a factor of 10."""
    f0 = mag_to_maggies(20.0)
    f1 = mag_to_maggies(22.5)

    np.testing.assert_allclose(f1 / f0, 0.1, rtol=1e-14)


def test_maggies_to_mag_custom_floor() -> None:
    """Tests that a custom floor controls the maximum returned magnitude."""
    m = maggies_to_mag(np.array([0.0, 1e-20]), floor=1e-10)

    np.testing.assert_allclose(m, np.array([25.0, 25.0]))


def test_magerr_to_ivar_maggies_broadcasts_inputs() -> None:
    """Tests that magerr_to_ivar_maggies supports NumPy broadcasting."""
    m = np.array([20.0, 21.0, 22.0])
    sigma_m = 0.1

    ivar = magerr_to_ivar_maggies(m, sigma_m)

    f = mag_to_maggies(m)
    sigma_f = (0.4 * np.log(10.0)) * f * sigma_m
    expected = 1.0 / sigma_f**2

    assert ivar.shape == m.shape
    np.testing.assert_allclose(ivar, expected)


def test_magerr_to_ivar_maggies_negative_sigma_is_masked() -> None:
    """Tests that negative magnitude uncertainties produce zero inverse variance."""
    ivar = magerr_to_ivar_maggies(
        np.array([20.0, 21.0]),
        np.array([-0.1, 0.1]),
    )

    assert ivar[0] == 0.0
    assert ivar[1] > 0.0


def test_magerr_to_ivar_maggies_infinite_sigma_is_masked() -> None:
    """Tests that infinite propagated uncertainty produces zero inverse variance."""
    ivar = magerr_to_ivar_maggies(
        np.array([20.0, 21.0]),
        np.array([np.inf, 0.1]),
    )

    assert ivar[0] == 0.0
    assert ivar[1] > 0.0


def test_magerr_to_ivar_maggies_preserves_shape() -> None:
    """Tests that inverse-variance conversion preserves broadcasted array shape."""
    m = np.array([[20.0, 21.0], [22.0, 23.0]])
    sigma_m = np.full_like(m, 0.1)

    ivar = magerr_to_ivar_maggies(m, sigma_m)

    assert ivar.shape == m.shape
    assert np.all(ivar > 0.0)


@pytest.mark.parametrize(
    "function, args",
    [
        (mag_to_maggies, (20.0,)),
        (maggies_to_mag, (1e-8,)),
        (magerr_to_ivar_maggies, (20.0, 0.1)),
    ],
)
def test_flux_scalar_returns_float64_array(function, args) -> None:
    """Tests that scalar flux conversions return zero-dimensional float64 arrays."""
    result = function(*args)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.shape == ()


@pytest.mark.parametrize(
    "floor",
    [0.0, -1.0, np.nan, np.inf, -np.inf],
)
def test_maggies_to_mag_rejects_invalid_floor(floor) -> None:
    """Tests that flux-to-magnitude conversion rejects invalid floors."""
    with pytest.raises(ValueError, match="floor must be finite and strictly positive"):
        maggies_to_mag(1.0, floor=floor)


def test_mag_to_maggies_preserves_nonfinite_limits() -> None:
    """Tests that magnitude-to-flux conversion preserves non-finite limits."""
    result = mag_to_maggies([np.nan, np.inf, -np.inf])

    assert np.isnan(result[0])
    assert result[1] == 0.0
    assert np.isposinf(result[2])


def test_maggies_to_mag_preserves_missing_fluxes() -> None:
    """Tests that flux-to-magnitude conversion preserves missing measurements."""
    result = maggies_to_mag([np.nan, np.inf, 0.0])

    assert np.isnan(result[0])
    assert np.isneginf(result[1])
    assert np.isfinite(result[2])


def test_mag_to_maggies_extreme_magnitudes() -> None:
    """Tests that extreme magnitudes retain their floating-point limits."""
    with np.errstate(all="raise"):
        result = mag_to_maggies([-1000.0, 1000.0])

    assert np.isposinf(result[0])
    assert result[1] == 0.0


@pytest.mark.parametrize("magnitude", [np.nan, np.inf, -np.inf])
def test_inverse_variance_masks_nonfinite_magnitudes(magnitude) -> None:
    """Tests that non-finite magnitudes receive zero inverse variance."""
    with np.errstate(all="raise"):
        result = magerr_to_ivar_maggies(magnitude, 0.1)

    assert result == 0.0


@pytest.mark.parametrize("sigma_m", [0.0, -0.1, np.nan, np.inf, -np.inf])
def test_inverse_variance_masks_invalid_uncertainties(sigma_m) -> None:
    """Tests that invalid magnitude uncertainties receive zero inverse variance."""
    with np.errstate(all="raise"):
        result = magerr_to_ivar_maggies(20.0, sigma_m)

    assert result == 0.0


@pytest.mark.parametrize("magnitude", [-1000.0, -500.0, 500.0, 1000.0])
def test_inverse_variance_extreme_magnitudes(magnitude) -> None:
    """Tests that extreme magnitudes yield finite inverse variance without warnings."""
    with np.errstate(all="raise"):
        result = magerr_to_ivar_maggies(magnitude, 0.1)

    assert result == 0.0
    assert np.isfinite(result)


def test_inverse_variance_preserves_representable_subnormal_values() -> None:
    """Tests that representable inverse variances survive variance overflow."""
    result = magerr_to_ivar_maggies(0.0, 1e155)

    expected = (1.0 / (0.4 * np.log(10.0) * 1e155)) ** 2

    assert result > 0.0
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0.0)


def test_inverse_variance_masks_overflowing_weights() -> None:
    """Tests that unrepresentable inverse variances are masked."""
    with np.errstate(all="raise"):
        result = magerr_to_ivar_maggies(0.0, 1e-200)

    assert result == 0.0


def test_inverse_variance_multidimensional_broadcasting() -> None:
    """Tests that inverse variance supports multidimensional broadcasting."""
    magnitude = np.array([[20.0], [21.0]])
    uncertainty = np.array([0.1, 0.2, 0.3])

    result = magerr_to_ivar_maggies(magnitude, uncertainty)

    expected = 1.0 / (0.4 * np.log(10.0) * mag_to_maggies(magnitude) * uncertainty) ** 2

    assert result.shape == (2, 3)
    np.testing.assert_allclose(result, expected)
