"""Unit tests for lfkit.cosmology.distances."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.cosmology import FLRW, FlatLambdaCDM

from lfkit.cosmology.distances import (
    comoving_distance_mpc,
    distance_modulus,
    distance_modulus_from_luminosity_distance,
    luminosity_distance_mpc,
    transverse_comoving_distance_mpc,
)


@pytest.mark.parametrize(
    "function",
    [
        comoving_distance_mpc,
        transverse_comoving_distance_mpc,
        luminosity_distance_mpc,
        distance_modulus,
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
def test_distance_shape_dtype_and_finiteness(cosmology, function, z):
    """Tests that distance calculations preserve shape and return finite floats."""
    result = function(cosmology, z)

    assert isinstance(result, np.ndarray)
    assert result.shape == np.shape(z)
    assert result.dtype == np.float64
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize(
    "function",
    [
        comoving_distance_mpc,
        transverse_comoving_distance_mpc,
        luminosity_distance_mpc,
    ],
)
def test_distance_zero_at_origin(cosmology, function):
    """Tests that cosmological distances vanish at zero redshift."""
    assert float(function(cosmology, 0.0)) == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize(
    "function",
    [
        comoving_distance_mpc,
        transverse_comoving_distance_mpc,
        luminosity_distance_mpc,
    ],
)
def test_distance_independent_of_input_grid(cosmology, function):
    """Tests that cosmological distances are independent of the redshift grid."""
    z = np.array([0.2, 0.5, 1.0, 2.0])
    expected = np.array([float(function(cosmology, zi)) for zi in z])

    np.testing.assert_allclose(
        function(cosmology, z),
        expected,
        rtol=1e-10,
    )


def test_distances_match_backend(cosmology):
    """Tests that cosmological distances agree with native backend calculations."""
    z = np.array([0.1, 0.5, 1.0, 2.0])

    if isinstance(cosmology, FLRW):
        radial = cosmology.comoving_distance(z).to_value("Mpc")
        transverse = cosmology.comoving_transverse_distance(z).to_value("Mpc")
        luminosity = cosmology.luminosity_distance(z).to_value("Mpc")
    else:
        a = 1.0 / (1.0 + z)
        radial = cosmology.comoving_radial_distance(a)
        transverse = cosmology.comoving_angular_distance(a)
        luminosity = cosmology.luminosity_distance(a)

    np.testing.assert_allclose(comoving_distance_mpc(cosmology, z), radial, rtol=1e-10)
    np.testing.assert_allclose(
        transverse_comoving_distance_mpc(cosmology, z),
        transverse,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        luminosity_distance_mpc(cosmology, z), luminosity, rtol=1e-10
    )


def test_distance_duality(cosmology):
    """Tests that luminosity and transverse distances satisfy distance duality."""
    z = np.linspace(0.1, 3.0, 20)

    np.testing.assert_allclose(
        luminosity_distance_mpc(cosmology, z),
        (1.0 + z) * transverse_comoving_distance_mpc(cosmology, z),
        rtol=1e-6,
    )


def test_curvature_changes_transverse_distance(cosmology):
    """Tests that radial and transverse distances respond to spatial curvature."""
    z = np.array([0.5, 1.0, 2.0])

    radial = comoving_distance_mpc(cosmology, z)
    transverse = transverse_comoving_distance_mpc(cosmology, z)

    omega_k = cosmology.Ok0 if isinstance(cosmology, FLRW) else cosmology["Omega_k"]

    if np.isclose(omega_k, 0.0):
        np.testing.assert_allclose(radial, transverse, rtol=1e-6)
    elif omega_k > 0.0:
        assert np.all(transverse > radial)
    else:
        assert np.all(transverse < radial)


def test_distance_modulus_from_distance():
    """Tests that distance modulus follows its analytic definition."""
    distance = np.array([1.0, 10.0, 100.0, 1000.0])

    np.testing.assert_allclose(
        distance_modulus_from_luminosity_distance(distance),
        [25.0, 30.0, 35.0, 40.0],
    )


def test_distance_modulus_h_rescaling():
    """Tests that optional h rescaling shifts the distance modulus."""
    distance = np.array([100.0, 1000.0])
    h = 0.67

    np.testing.assert_allclose(
        distance_modulus_from_luminosity_distance(distance, h=h),
        distance_modulus_from_luminosity_distance(distance) + 5.0 * np.log10(h),
    )


def test_distance_modulus_consistency(cosmology):
    """Tests that cosmology-based and distance-based modulus calculations agree."""
    z = np.array([0.1, 0.5, 1.0])

    np.testing.assert_allclose(
        distance_modulus(cosmology, z, h=0.67),
        distance_modulus_from_luminosity_distance(
            luminosity_distance_mpc(cosmology, z),
            h=0.67,
        ),
    )


def test_distance_modulus_zero_distance():
    """Tests that distance modulus returns negative infinity at zero distance."""
    assert np.isneginf(distance_modulus_from_luminosity_distance(0.0))


@pytest.mark.parametrize("distance", [-1.0, np.nan, np.inf])
def test_invalid_luminosity_distance(distance):
    """Tests that invalid luminosity distances are rejected."""
    with pytest.raises(ValueError):
        distance_modulus_from_luminosity_distance(distance)


@pytest.mark.parametrize("h", [-0.7, 0.0, np.nan, np.inf])
def test_invalid_h(h):
    """Tests that invalid Hubble rescaling parameters are rejected."""
    with pytest.raises(ValueError):
        distance_modulus_from_luminosity_distance(100.0, h=h)


@pytest.mark.parametrize("z", [-0.1, np.nan, np.inf])
@pytest.mark.parametrize(
    "function",
    [
        comoving_distance_mpc,
        transverse_comoving_distance_mpc,
        luminosity_distance_mpc,
        distance_modulus,
    ],
)
def test_invalid_redshift(function, z):
    """Tests that invalid redshifts are rejected before backend evaluation."""
    cosmology = FlatLambdaCDM(H0=67.0, Om0=0.3)

    with pytest.raises(ValueError):
        function(cosmology, z)
