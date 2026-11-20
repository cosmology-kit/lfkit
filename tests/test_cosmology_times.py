"""Unit tests for lfkit.cosmology.times."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.cosmology import FLRW, FlatLambdaCDM

from lfkit.cosmology import times
from lfkit.cosmology.times import (
    lookback_time_gyr,
    z_from_lookback_time,
)


@pytest.mark.parametrize(
    "z",
    [
        0.5,
        [0.0, 0.1, 0.5, 1.0],
        [1.0, 0.1, 0.5],
        [[0.1, 0.5], [1.0, 2.0]],
    ],
)
def test_lookback_time_shape_dtype_and_finiteness(cosmology, z):
    """Tests that lookback times preserve shape and return finite floats."""
    result = lookback_time_gyr(cosmology, z)

    assert isinstance(result, np.ndarray)
    assert result.shape == np.shape(z)
    assert result.dtype == np.float64
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)


def test_lookback_time_zero_at_origin(cosmology):
    """Tests that lookback time is exactly zero at zero redshift."""
    assert float(lookback_time_gyr(cosmology, 0.0)) == 0.0


def test_lookback_time_monotonic(cosmology):
    """Tests that lookback time increases strictly with redshift."""
    z = np.linspace(0.0, 5.0, 100)
    result = lookback_time_gyr(cosmology, z)

    assert np.all(np.diff(result) > 0.0)


def test_lookback_time_matches_backend(cosmology):
    """Tests that lookback time agrees with the native backend."""
    z = np.array([0.0, 0.1, 0.5, 1.0, 2.0])

    if isinstance(cosmology, FLRW):
        expected = cosmology.lookback_time(z).to_value("Gyr")
    else:
        expected = cosmology.lookback_time(1.0 / (1.0 + z))

    np.testing.assert_allclose(
        lookback_time_gyr(cosmology, z),
        expected,
        rtol=1e-10,
        atol=1e-10,
    )


def test_lookback_time_independent_of_input_grid(cosmology):
    """Tests that lookback time is independent of the redshift grid."""
    z = np.array([0.2, 0.5, 1.0, 2.0])

    expected = np.array([float(lookback_time_gyr(cosmology, zi)) for zi in z])

    np.testing.assert_allclose(
        lookback_time_gyr(cosmology, z),
        expected,
        rtol=1e-10,
    )


@pytest.mark.parametrize("z", [-0.1, np.nan, np.inf])
def test_invalid_redshift(z):
    """Tests that invalid redshift values are rejected."""
    cosmology = FlatLambdaCDM(H0=67.0, Om0=0.3)

    with pytest.raises(ValueError):
        lookback_time_gyr(cosmology, z)


@pytest.mark.parametrize(
    "z",
    [
        0.5,
        [0.0, 0.1, 0.5, 1.0],
        [2.0, 0.1, 0.5],
        [[0.1, 0.5], [1.0, 2.0]],
    ],
)
def test_lookback_time_redshift_roundtrip(cosmology, z):
    """Tests that lookback time inversion recovers the original redshift."""
    t_lb = lookback_time_gyr(cosmology, z)
    recovered = z_from_lookback_time(cosmology, t_lb)

    assert isinstance(recovered, np.ndarray)
    assert recovered.shape == np.shape(z)
    assert recovered.dtype == np.float64

    np.testing.assert_allclose(
        recovered,
        z,
        rtol=1e-5,
        atol=1e-5,
    )


def test_lookback_time_inversion_grid_endpoints(cosmology):
    """Tests that lookback time inversion recovers both grid endpoints."""
    z = np.array([0.0, 5.0])
    t_lb = lookback_time_gyr(cosmology, z)

    np.testing.assert_allclose(
        z_from_lookback_time(cosmology, t_lb, zmax=5.0),
        z,
        atol=1e-10,
    )


def test_lookback_time_inversion_linear_mapping(monkeypatch):
    """Tests that lookback time inversion exactly recovers a linear mapping."""
    monkeypatch.setattr(
        times,
        "lookback_time_gyr",
        lambda cosmo, z: 3.0 * np.asarray(z),
    )

    t_lb = np.array([0.0, 0.6, 3.0])

    np.testing.assert_allclose(
        z_from_lookback_time(object(), t_lb, zmax=2.0),
        t_lb / 3.0,
    )


@pytest.mark.parametrize("zmax", [-1.0, 0.0, np.nan, np.inf])
def test_invalid_inversion_zmax(zmax):
    """Tests that invalid maximum inversion redshifts are rejected."""
    with pytest.raises(ValueError):
        z_from_lookback_time(object(), 0.5, zmax=zmax)


@pytest.mark.parametrize("nz", [-1, 0, 16, 32.5, 64.0, True, np.nan, np.inf])
def test_invalid_inversion_nz(nz):
    """Tests that invalid inversion grid sizes are rejected."""
    with pytest.raises(ValueError):
        z_from_lookback_time(object(), 0.5, nz=nz)


@pytest.mark.parametrize("t_lb", [-0.1, np.nan, np.inf])
def test_invalid_lookback_time(t_lb):
    """Tests that invalid lookback time values are rejected."""
    with pytest.raises(ValueError):
        z_from_lookback_time(object(), t_lb)


def test_lookback_time_outside_inversion_range(cosmology):
    """Tests that inversion rejects lookback times outside the grid range."""
    t_lb = float(lookback_time_gyr(cosmology, 2.0))

    with pytest.raises(ValueError, match="outside inversion range"):
        z_from_lookback_time(cosmology, t_lb, zmax=1.0)


@pytest.mark.parametrize(
    "function",
    [
        lambda z: -z,
        lambda z: np.zeros_like(z),
        lambda z: np.full_like(z, np.nan),
        lambda z: np.full_like(z, np.inf),
    ],
)
def test_invalid_lookback_time_grid(monkeypatch, function):
    """Tests that nonmonotonic or nonfinite lookback time grids are rejected."""
    monkeypatch.setattr(
        times,
        "lookback_time_gyr",
        lambda cosmo, z: function(z),
    )

    with pytest.raises(ValueError):
        z_from_lookback_time(object(), 0.5)
