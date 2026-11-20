"""Unit tests for lfkit.cosmology.background."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.cosmology import FLRW, FlatLambdaCDM

from lfkit.cosmology.background import (
    C_KM_S,
    differential_comoving_volume,
    dimensionless_hubble_parameter,
    expansion_rate,
    hubble_parameter,
)


def test_speed_of_light():
    """Tests that the speed of light is expressed in km/s."""
    assert C_KM_S == pytest.approx(299792.458)


def test_dimensionless_hubble_parameter(cosmology):
    """Tests that both backends return the expected Hubble parameter h."""
    assert dimensionless_hubble_parameter(cosmology) == pytest.approx(0.67)


@pytest.mark.parametrize("h", [-0.7, 0.0, np.nan, np.inf])
def test_invalid_dimensionless_hubble_parameter(h):
    """Tests that invalid dimensionless Hubble parameters are rejected."""
    with pytest.raises(ValueError):
        dimensionless_hubble_parameter({"h": h})


@pytest.mark.parametrize(
    "function",
    [
        expansion_rate,
        hubble_parameter,
        differential_comoving_volume,
    ],
)
@pytest.mark.parametrize(
    "z",
    [
        0.5,
        [0.1, 0.5, 1.0],
        [1.0, 0.1, 0.5],
        [[0.1, 0.5], [1.0, 2.0]],
    ],
)
def test_background_shape_dtype_and_finiteness(cosmology, function, z):
    """Tests that background calculations preserve shape and return finite floats."""
    result = function(cosmology, z)

    assert isinstance(result, np.ndarray)
    assert result.shape == np.shape(z)
    assert result.dtype == np.float64
    assert np.all(np.isfinite(result))
    assert np.all(result > 0.0)


def test_expansion_rate_at_origin(cosmology):
    """Tests that the normalized expansion rate equals unity at zero redshift."""
    assert float(expansion_rate(cosmology, 0.0)) == pytest.approx(1.0, rel=1e-6)


def test_hubble_parameter_at_origin(cosmology):
    """Tests that the present-day Hubble parameter equals 100 h."""
    assert float(hubble_parameter(cosmology, 0.0)) == pytest.approx(67.0, rel=1e-6)


def test_expansion_rate_matches_backend(cosmology):
    """Tests that the expansion rate matches the native backend."""
    z = np.array([0.1, 0.5, 1.0, 2.0])

    if isinstance(cosmology, FLRW):
        expected = cosmology.efunc(z)
    else:
        expected = cosmology.h_over_h0(1.0 / (1.0 + z))

    np.testing.assert_allclose(
        expansion_rate(cosmology, z),
        expected,
        rtol=1e-10,
    )


def test_hubble_parameter_matches_backend(cosmology):
    """Tests that the physical Hubble parameter matches the native backend."""
    z = np.array([0.1, 0.5, 1.0, 2.0])

    if isinstance(cosmology, FLRW):
        expected = cosmology.H(z).to_value("km / (s Mpc)")
    else:
        expected = 100.0 * cosmology["h"] * cosmology.h_over_h0(1.0 / (1.0 + z))

    np.testing.assert_allclose(
        hubble_parameter(cosmology, z),
        expected,
        rtol=1e-10,
    )


def test_volume_zero_at_origin(cosmology):
    """Tests that differential comoving volume vanishes at zero redshift."""
    assert float(differential_comoving_volume(cosmology, 0.0)) == pytest.approx(
        0.0, abs=1e-8
    )


def test_volume_independent_of_input_grid(cosmology):
    """Tests that differential comoving volume is independent of the redshift grid."""
    z = np.array([0.2, 0.5, 1.0, 2.0])

    expected = np.array(
        [float(differential_comoving_volume(cosmology, zi)) for zi in z]
    )

    np.testing.assert_allclose(
        differential_comoving_volume(cosmology, z),
        expected,
        rtol=1e-10,
    )


@pytest.mark.parametrize("frac_sky", [0.0, 0.1, 0.5, 1.0])
def test_volume_sky_fraction(cosmology, frac_sky):
    """Tests that differential comoving volume scales linearly with sky fraction."""
    z = np.array([0.1, 0.5, 1.0])

    np.testing.assert_allclose(
        differential_comoving_volume(cosmology, z, frac_sky),
        frac_sky * differential_comoving_volume(cosmology, z),
    )


@pytest.mark.parametrize("frac_sky", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_sky_fraction(frac_sky):
    """Tests that invalid sky fractions are rejected."""
    cosmology = FlatLambdaCDM(H0=67.0, Om0=0.3)

    with pytest.raises(ValueError):
        differential_comoving_volume(cosmology, 0.5, frac_sky)


def test_volume_matches_backend(cosmology):
    """Tests that differential comoving volume agrees with the native backend."""
    z = np.array([0.1, 0.5, 1.0, 2.0])

    if isinstance(cosmology, FLRW):
        expected = (
            4.0
            * np.pi
            * cosmology.differential_comoving_volume(z).to_value("Mpc3 / sr")
        )
    else:
        a = 1.0 / (1.0 + z)
        expected = 4.0 * np.pi * a**2 * cosmology.comoving_volume_element(a)

    np.testing.assert_allclose(
        differential_comoving_volume(cosmology, z),
        expected,
        rtol=1e-6,
    )


@pytest.mark.parametrize("z", [-0.1, np.nan, np.inf])
@pytest.mark.parametrize(
    "function",
    [
        expansion_rate,
        hubble_parameter,
        differential_comoving_volume,
    ],
)
def test_invalid_redshift(function, z):
    """Tests that background calculations reject invalid redshifts."""
    cosmology = FlatLambdaCDM(H0=67.0, Om0=0.3)

    with pytest.raises(ValueError):
        function(cosmology, z)
