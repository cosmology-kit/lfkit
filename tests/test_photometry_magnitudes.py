"""Unit tests for ``lfkit.photometry.magnitudes``."""

from __future__ import annotations

import numpy as np
import pytest

import lfkit.photometry.magnitudes as magnitudes
from lfkit.photometry.magnitudes import (
    absolute_magnitude_from_luminosity_distance,
    apparent_magnitude_from_luminosity_distance,
    total_magnitude_correction,
)


def test_total_magnitude_correction_none_inputs() -> None:
    """Tests that total_magnitude_correction returns zero when no corrections are given."""
    out = total_magnitude_correction()

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert out.shape == ()
    assert out == pytest.approx(0.0)


def test_total_magnitude_correction_k_only_scalar() -> None:
    """Tests that total_magnitude_correction returns only the k-correction."""
    out = total_magnitude_correction(k_correction=0.3)

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert out.shape == ()
    assert out == pytest.approx(0.3)


def test_total_magnitude_correction_e_only_scalar() -> None:
    """Tests that total_magnitude_correction returns minus the e-correction."""
    out = total_magnitude_correction(e_correction=0.2)

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert out.shape == ()
    assert out == pytest.approx(-0.2)


def test_total_magnitude_correction_adds_array_terms() -> None:
    """Tests that total_magnitude_correction combines K and E corrections."""
    k_corr = np.array([0.1, 0.2, 0.3])
    e_corr = np.array([0.05, 0.10, 0.15])

    out = total_magnitude_correction(
        k_correction=k_corr,
        e_correction=e_corr,
    )

    expected = np.array([0.05, 0.10, 0.15])
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, expected)


def test_total_magnitude_correction_broadcasts_scalar_and_array() -> None:
    """Tests that total_magnitude_correction broadcasts scalar and array inputs."""
    out = total_magnitude_correction(
        k_correction=0.2,
        e_correction=np.array([0.0, 0.1, 0.2]),
    )

    expected = np.array([0.2, 0.1, 0.0])
    np.testing.assert_allclose(out, expected)


def test_absolute_magnitude_without_corrections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that absolute_magnitude applies M = m - mu."""

    def mock_distance_modulus(
        cosmo_obj: object,
        z: np.ndarray,
        h: float | None = None,
    ) -> np.ndarray:
        """Return a fixed distance modulus."""
        _, _, _ = cosmo_obj, z, h
        return np.array([40.0, 41.0], dtype=float)

    def mock_luminosity_distance(cosmo_obj, z):
        """Return luminosity distances corresponding to the mocked modulus."""
        mu = mock_distance_modulus(cosmo_obj, z)
        return np.power(10.0, (mu - 25.0) / 5.0)

    monkeypatch.setattr(magnitudes, "luminosity_distance_mpc", mock_luminosity_distance)

    out = magnitudes.absolute_magnitude(
        cosmo_obj=object(),
        z=np.array([0.1, 0.2]),
        apparent_mag=np.array([20.0, 21.5]),
    )

    expected = np.array([-20.0, -19.5])
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, expected)


def test_absolute_magnitude_with_corrections(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that absolute_magnitude applies M = m - mu - K + E."""

    def mock_distance_modulus(
        cosmo_obj: object,
        z: np.ndarray,
        h: float | None = None,
    ) -> np.ndarray:
        """Return a fixed distance modulus."""
        _, _, _ = cosmo_obj, z, h
        return np.array([40.0, 41.0], dtype=float)

    def mock_luminosity_distance(cosmo_obj, z):
        """Return luminosity distances corresponding to the mocked modulus."""
        mu = mock_distance_modulus(cosmo_obj, z)
        return np.power(10.0, (mu - 25.0) / 5.0)

    monkeypatch.setattr(magnitudes, "luminosity_distance_mpc", mock_luminosity_distance)

    out = magnitudes.absolute_magnitude(
        cosmo_obj=object(),
        z=np.array([0.1, 0.2]),
        apparent_mag=np.array([20.0, 21.5]),
        k_correction=np.array([0.1, 0.2]),
        e_correction=np.array([0.3, 0.4]),
    )

    expected = (
        np.array([20.0, 21.5])
        - np.array([40.0, 41.0])
        - np.array([0.1, 0.2])
        + np.array([0.3, 0.4])
    )
    np.testing.assert_allclose(out, expected)


def test_apparent_magnitude_without_corrections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that apparent_magnitude applies m = M + mu."""

    def mock_distance_modulus(
        cosmo_obj: object,
        z: np.ndarray,
        h: float | None = None,
    ) -> np.ndarray:
        """Return a fixed distance modulus."""
        _, _, _ = cosmo_obj, z, h
        return np.array([40.0, 41.0], dtype=float)

    def mock_luminosity_distance(cosmo_obj, z):
        """Return luminosity distances corresponding to the mocked modulus."""
        mu = mock_distance_modulus(cosmo_obj, z)
        return np.power(10.0, (mu - 25.0) / 5.0)

    monkeypatch.setattr(magnitudes, "luminosity_distance_mpc", mock_luminosity_distance)

    out = magnitudes.apparent_magnitude(
        cosmo_obj=object(),
        z=np.array([0.1, 0.2]),
        absolute_mag=np.array([-20.0, -19.5]),
    )

    expected = np.array([20.0, 21.5])
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, expected)


def test_apparent_magnitude_with_corrections(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that apparent_magnitude applies m = M + mu + K - E."""

    def mock_distance_modulus(
        cosmo_obj: object,
        z: np.ndarray,
        h: float | None = None,
    ) -> np.ndarray:
        """Return a fixed distance modulus."""
        _, _, _ = cosmo_obj, z, h
        return np.array([40.0, 41.0], dtype=float)

    def mock_luminosity_distance(cosmo_obj, z):
        """Return luminosity distances corresponding to the mocked modulus."""
        mu = mock_distance_modulus(cosmo_obj, z)
        return np.power(10.0, (mu - 25.0) / 5.0)

    monkeypatch.setattr(magnitudes, "luminosity_distance_mpc", mock_luminosity_distance)

    out = magnitudes.apparent_magnitude(
        cosmo_obj=object(),
        z=np.array([0.1, 0.2]),
        absolute_mag=np.array([-20.0, -19.5]),
        k_correction=np.array([0.1, 0.2]),
        e_correction=np.array([0.3, 0.4]),
    )

    expected = (
        np.array([-20.0, -19.5])
        + np.array([40.0, 41.0])
        + np.array([0.1, 0.2])
        - np.array([0.3, 0.4])
    )
    np.testing.assert_allclose(out, expected)


def test_apparent_and_absolute_are_inverse_without_corrections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that apparent_magnitude and absolute_magnitude invert each other."""

    def mock_distance_modulus(
        cosmo_obj: object,
        z: np.ndarray,
        h: float | None = None,
    ) -> np.ndarray:
        """Return a redshift-dependent distance modulus."""
        _, _ = cosmo_obj, h
        z_arr = np.asarray(z, dtype=float)
        return 40.0 + z_arr

    def mock_luminosity_distance(cosmo_obj, z):
        """Return luminosity distances corresponding to the mocked modulus."""
        mu = mock_distance_modulus(cosmo_obj, z)
        return np.power(10.0, (mu - 25.0) / 5.0)

    monkeypatch.setattr(magnitudes, "luminosity_distance_mpc", mock_luminosity_distance)

    apparent_mag = np.array([20.0, 21.0, 22.0])
    z = np.array([0.1, 0.2, 0.3])

    absolute_mag = magnitudes.absolute_magnitude(
        cosmo_obj=object(),
        z=z,
        apparent_mag=apparent_mag,
    )
    recovered = magnitudes.apparent_magnitude(
        cosmo_obj=object(),
        z=z,
        absolute_mag=absolute_mag,
    )

    np.testing.assert_allclose(recovered, apparent_mag)


def test_apparent_and_absolute_are_inverse_with_corrections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that magnitude conversions invert each other with corrections."""

    def mock_distance_modulus(
        cosmo_obj: object,
        z: np.ndarray,
        h: float | None = None,
    ) -> np.ndarray:
        """Return a redshift-dependent distance modulus."""
        _, _ = cosmo_obj, h
        z_arr = np.asarray(z, dtype=float)
        return 39.5 + 2.0 * z_arr

    def mock_luminosity_distance(cosmo_obj, z):
        """Return luminosity distances corresponding to the mocked modulus."""
        mu = mock_distance_modulus(cosmo_obj, z)
        return np.power(10.0, (mu - 25.0) / 5.0)

    monkeypatch.setattr(magnitudes, "luminosity_distance_mpc", mock_luminosity_distance)

    apparent_mag = np.array([20.0, 21.0, 22.0])
    z = np.array([0.1, 0.2, 0.3])
    k_corr = np.array([0.05, 0.10, 0.15])
    e_corr = np.array([0.20, 0.10, 0.00])

    absolute_mag = magnitudes.absolute_magnitude(
        cosmo_obj=object(),
        z=z,
        apparent_mag=apparent_mag,
        k_correction=k_corr,
        e_correction=e_corr,
    )
    recovered = magnitudes.apparent_magnitude(
        cosmo_obj=object(),
        z=z,
        absolute_mag=absolute_mag,
        k_correction=k_corr,
        e_correction=e_corr,
    )

    np.testing.assert_allclose(recovered, apparent_mag)


def test_absolute_magnitude_broadcasts_scalar_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that absolute_magnitude broadcasts scalar corrections."""

    def mock_distance_modulus(
        cosmo_obj: object,
        z: np.ndarray,
        h: float | None = None,
    ) -> np.ndarray:
        """Return a fixed distance modulus array."""
        _, _, _ = cosmo_obj, z, h
        return np.array([40.0, 41.0, 42.0], dtype=float)

    def mock_luminosity_distance(cosmo_obj, z):
        """Return luminosity distances corresponding to the mocked modulus."""
        mu = mock_distance_modulus(cosmo_obj, z)
        return np.power(10.0, (mu - 25.0) / 5.0)

    monkeypatch.setattr(magnitudes, "luminosity_distance_mpc", mock_luminosity_distance)

    out = magnitudes.absolute_magnitude(
        cosmo_obj=object(),
        z=np.array([0.1, 0.2, 0.3]),
        apparent_mag=np.array([20.0, 21.0, 22.0]),
        k_correction=0.2,
    )

    expected = np.array([20.0, 21.0, 22.0]) - np.array([40.0, 41.0, 42.0]) - 0.2
    np.testing.assert_allclose(out, expected)


def test_apparent_magnitude_broadcasts_scalar_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that apparent_magnitude broadcasts scalar corrections."""

    def mock_distance_modulus(
        cosmo_obj: object,
        z: np.ndarray,
        h: float | None = None,
    ) -> np.ndarray:
        """Return a fixed distance modulus array."""
        _, _, _ = cosmo_obj, z, h
        return np.array([40.0, 41.0, 42.0], dtype=float)

    def mock_luminosity_distance(cosmo_obj, z):
        """Return luminosity distances corresponding to the mocked modulus."""
        mu = mock_distance_modulus(cosmo_obj, z)
        return np.power(10.0, (mu - 25.0) / 5.0)

    monkeypatch.setattr(magnitudes, "luminosity_distance_mpc", mock_luminosity_distance)

    out = magnitudes.apparent_magnitude(
        cosmo_obj=object(),
        z=np.array([0.1, 0.2, 0.3]),
        absolute_mag=np.array([-20.0, -20.0, -20.0]),
        e_correction=0.3,
    )

    expected = np.array([-20.0, -20.0, -20.0]) + np.array([40.0, 41.0, 42.0]) - 0.3
    np.testing.assert_allclose(out, expected)


def test_absolute_magnitude_from_luminosity_distance_scalar() -> None:
    """Tests apparent-to-absolute conversion from scalar luminosity distance."""
    result = absolute_magnitude_from_luminosity_distance(
        26.0,
        10.0,
    )

    assert result.shape == ()
    assert result == pytest.approx(-4.0)


def test_apparent_magnitude_from_luminosity_distance_scalar() -> None:
    """Tests absolute-to-apparent conversion from scalar luminosity distance."""
    result = apparent_magnitude_from_luminosity_distance(
        -4.0,
        10.0,
    )

    assert result.shape == ()
    assert result == pytest.approx(26.0)


def test_luminosity_distance_magnitude_conversions_are_inverse() -> None:
    """Tests that luminosity-distance magnitude conversions invert each other."""
    apparent_mag = np.array([24.0, 25.0, 26.0])
    luminosity_distance = np.array([10.0, 20.0, 30.0])

    absolute_mag = absolute_magnitude_from_luminosity_distance(
        apparent_mag,
        luminosity_distance,
    )
    recovered = apparent_magnitude_from_luminosity_distance(
        absolute_mag,
        luminosity_distance,
    )

    np.testing.assert_allclose(recovered, apparent_mag)


def test_luminosity_distance_magnitude_conversions_are_inverse_with_corrections() -> (
    None
):
    """Tests luminosity-distance conversions invert each other with corrections."""
    apparent_mag = np.array([24.0, 25.0, 26.0])
    luminosity_distance = np.array([10.0, 20.0, 30.0])
    k_corr = np.array([0.0, 0.2, 0.4])
    e_corr = np.array([0.1, 0.1, 0.2])

    absolute_mag = absolute_magnitude_from_luminosity_distance(
        apparent_mag,
        luminosity_distance,
        k_correction=k_corr,
        e_correction=e_corr,
    )
    recovered = apparent_magnitude_from_luminosity_distance(
        absolute_mag,
        luminosity_distance,
        k_correction=k_corr,
        e_correction=e_corr,
    )

    np.testing.assert_allclose(recovered, apparent_mag)


def test_absolute_magnitude_from_luminosity_distance_with_corrections() -> None:
    """Tests apparent-to-absolute conversion with K and evolution corrections."""
    result = absolute_magnitude_from_luminosity_distance(
        26.0,
        10.0,
        k_correction=0.5,
        e_correction=0.2,
    )

    assert result == pytest.approx(-4.3)


def test_apparent_magnitude_from_luminosity_distance_with_corrections() -> None:
    """Tests absolute-to-apparent conversion with K and evolution corrections."""
    result = apparent_magnitude_from_luminosity_distance(
        -4.3,
        10.0,
        k_correction=0.5,
        e_correction=0.2,
    )

    assert result == pytest.approx(26.0)


def test_absolute_magnitude_from_luminosity_distance_broadcasts_arrays() -> None:
    """Tests apparent-to-absolute conversion with broadcastable inputs."""
    result = absolute_magnitude_from_luminosity_distance(
        26.0,
        np.array([10.0, 100.0]),
    )

    np.testing.assert_allclose(result, np.array([-4.0, -9.0]))


def test_apparent_magnitude_from_luminosity_distance_broadcasts_arrays() -> None:
    """Tests absolute-to-apparent conversion with broadcastable inputs."""
    result = apparent_magnitude_from_luminosity_distance(
        np.array([-4.0, -9.0]),
        np.array([10.0, 100.0]),
    )

    np.testing.assert_allclose(result, np.array([26.0, 26.0]))


def test_absolute_magnitude_from_luminosity_distance_broadcasts_corrections() -> None:
    """Tests apparent-to-absolute conversion with array corrections."""
    result = absolute_magnitude_from_luminosity_distance(
        26.0,
        10.0,
        k_correction=np.array([0.0, 0.5]),
        e_correction=np.array([0.0, 1.0]),
    )

    np.testing.assert_allclose(result, np.array([-4.0, -3.5]))


def test_apparent_magnitude_from_luminosity_distance_broadcasts_corrections() -> None:
    """Tests absolute-to-apparent conversion with array corrections."""
    result = apparent_magnitude_from_luminosity_distance(
        -4.0,
        10.0,
        k_correction=np.array([0.0, 0.5]),
        e_correction=np.array([0.0, 1.0]),
    )

    np.testing.assert_allclose(result, np.array([26.0, 25.5]))


def test_absolute_magnitude_from_luminosity_distance_rejects_negative_distance() -> (
    None
):
    """Tests that apparent-to-absolute conversion rejects negative distance."""
    with pytest.raises(ValueError, match="negative values"):
        absolute_magnitude_from_luminosity_distance(
            26.0,
            -10.0,
        )


def test_apparent_magnitude_from_luminosity_distance_rejects_negative_distance() -> (
    None
):
    """Tests that absolute-to-apparent conversion rejects negative distance."""
    with pytest.raises(ValueError, match="negative values"):
        apparent_magnitude_from_luminosity_distance(
            -4.0,
            -10.0,
        )


def test_absolute_magnitude_from_luminosity_distance_rejects_nonfinite_distance() -> (
    None
):
    """Tests that apparent-to-absolute conversion rejects non-finite distance."""
    with pytest.raises(ValueError, match="luminosity_distance contains NaN"):
        absolute_magnitude_from_luminosity_distance(
            26.0,
            np.nan,
        )


def test_apparent_magnitude_from_luminosity_distance_rejects_nonfinite_distance() -> (
    None
):
    """Tests that absolute-to-apparent conversion rejects non-finite distance."""
    with pytest.raises(ValueError, match="luminosity_distance contains NaN"):
        apparent_magnitude_from_luminosity_distance(
            -4.0,
            np.nan,
        )


def test_total_magnitude_correction_preserves_broadcast_shape() -> None:
    """Tests that total_magnitude_correction returns the broadcasted output shape."""
    out = total_magnitude_correction(
        k_correction=np.array([[0.1], [0.2]]),
        e_correction=np.array([0.0, 0.1, 0.2]),
    )

    expected = np.array(
        [
            [0.1, 0.0, -0.1],
            [0.2, 0.1, 0.0],
        ],
    )
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, expected)


def test_absolute_magnitude_from_luminosity_distance_returns_float64_array() -> None:
    """Tests that apparent-to-absolute conversion returns a float64 array."""
    out = absolute_magnitude_from_luminosity_distance(
        np.array([26, 27]),
        np.array([10, 100]),
    )

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64


def test_apparent_magnitude_from_luminosity_distance_returns_float64_array() -> None:
    """Tests that absolute-to-apparent conversion returns a float64 array."""
    out = apparent_magnitude_from_luminosity_distance(
        np.array([-4, -8]),
        np.array([10, 100]),
    )

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64


def test_luminosity_distance_formula_matches_known_distance_modulus() -> None:
    """Tests that luminosity-distance conversions use mu = 5 log10(dL) + 25."""
    result = absolute_magnitude_from_luminosity_distance(
        apparent_mag=30.0,
        luminosity_distance_mpc=100.0,
    )

    assert result == pytest.approx(-5.0)


def test_apparent_magnitude_from_luminosity_distance_matches_known_distance_modulus() -> (
    None
):
    """Tests that absolute-to-apparent conversion uses the expected distance modulus."""
    result = apparent_magnitude_from_luminosity_distance(
        absolute_mag=0.0,
        luminosity_distance_mpc=100.0,
    )

    assert result == pytest.approx(35.0)


def test_absolute_magnitude_supports_scalar_magnitude_with_array_redshift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that absolute_magnitude broadcasts scalar magnitude over redshift."""

    def mock_distance_modulus(
        cosmo_obj: object,
        z: np.ndarray,
        h: float | None = None,
    ) -> np.ndarray:
        """Return a redshift-shaped distance modulus."""
        _, _ = cosmo_obj, h
        return np.asarray([40.0, 41.0, 42.0], dtype=float)

    def mock_luminosity_distance(cosmo_obj, z):
        """Return luminosity distances corresponding to the mocked modulus."""
        mu = mock_distance_modulus(cosmo_obj, z)
        return np.power(10.0, (mu - 25.0) / 5.0)

    monkeypatch.setattr(magnitudes, "luminosity_distance_mpc", mock_luminosity_distance)

    out = magnitudes.absolute_magnitude(
        cosmo_obj=object(),
        z=np.array([0.1, 0.2, 0.3]),
        apparent_mag=22.0,
    )

    np.testing.assert_allclose(out, np.array([-18.0, -19.0, -20.0]))


def test_apparent_magnitude_supports_scalar_magnitude_with_array_redshift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that apparent_magnitude broadcasts scalar magnitude over redshift."""

    def mock_distance_modulus(
        cosmo_obj: object,
        z: np.ndarray,
        h: float | None = None,
    ) -> np.ndarray:
        """Return a redshift-shaped distance modulus."""
        _, _ = cosmo_obj, h
        return np.asarray([40.0, 41.0, 42.0], dtype=float)

    def mock_luminosity_distance(cosmo_obj, z):
        """Return luminosity distances corresponding to the mocked modulus."""
        mu = mock_distance_modulus(cosmo_obj, z)
        return np.power(10.0, (mu - 25.0) / 5.0)

    monkeypatch.setattr(magnitudes, "luminosity_distance_mpc", mock_luminosity_distance)

    out = magnitudes.apparent_magnitude(
        cosmo_obj=object(),
        z=np.array([0.1, 0.2, 0.3]),
        absolute_mag=-20.0,
    )

    np.testing.assert_allclose(out, np.array([20.0, 21.0, 22.0]))


def test_magnitudes_exports_expected_public_names() -> None:
    """Tests that magnitudes exposes the expected public API names."""
    expected = {
        "total_magnitude_correction",
        "absolute_magnitude",
        "absolute_magnitude_from_luminosity_distance",
        "apparent_magnitude",
        "apparent_magnitude_from_luminosity_distance",
    }

    assert set(magnitudes.__all__) == expected


def test_magnitudes_api_aliases_match_public_functions() -> None:
    """Tests that magnitudes API aliases point to public functions."""
    for public_name in magnitudes.__api_aliases__:
        assert public_name in magnitudes.__all__
        assert callable(getattr(magnitudes, public_name))


def test_magnitude_roundtrip_with_cosmology(cosmology):
    """Tests that magnitude conversions are invertible across cosmological backends."""
    z = np.array([0.1, 0.5, 1.0, 2.0])
    m = np.array([19.0, 21.0, 23.0, 25.0])

    absolute = magnitudes.absolute_magnitude(
        cosmology,
        z,
        m,
        h=0.67,
        k_correction=0.3,
        e_correction=0.1,
    )

    recovered = magnitudes.apparent_magnitude(
        cosmology,
        z,
        absolute,
        h=0.67,
        k_correction=0.3,
        e_correction=0.1,
    )

    np.testing.assert_allclose(recovered, m, rtol=1e-12)


def test_magnitude_conversions_match_distance_based_conversions(cosmology):
    """Tests that cosmology-based and distance-based magnitude conversions agree."""
    from lfkit.cosmology.distances import luminosity_distance_mpc

    z = np.array([0.1, 0.5, 1.0])
    m = np.array([19.0, 21.0, 23.0])
    distance = luminosity_distance_mpc(cosmology, z)

    np.testing.assert_allclose(
        magnitudes.absolute_magnitude(cosmology, z, m),
        magnitudes.absolute_magnitude_from_luminosity_distance(m, distance),
        rtol=1e-12,
    )


@pytest.mark.parametrize("h", [None, 0.5, 0.67, 0.7, 1.0])
def test_distance_based_magnitude_roundtrip_with_h(h):
    """Tests that distance-based magnitude conversions invert each other with h."""
    distance = np.array([10.0, 100.0, 1000.0])
    apparent = np.array([20.0, 22.0, 24.0])

    absolute = absolute_magnitude_from_luminosity_distance(
        apparent,
        distance,
        h=h,
        k_correction=0.3,
        e_correction=0.1,
    )

    recovered = apparent_magnitude_from_luminosity_distance(
        absolute,
        distance,
        h=h,
        k_correction=0.3,
        e_correction=0.1,
    )

    np.testing.assert_allclose(recovered, apparent, atol=1e-12)


@pytest.mark.parametrize("h", [0.5, 0.67, 0.7, 1.0])
def test_absolute_magnitude_h_scaling(h):
    """Tests that absolute magnitudes follow the expected little-h convention."""
    physical = absolute_magnitude_from_luminosity_distance(25.0, 100.0)
    scaled = absolute_magnitude_from_luminosity_distance(25.0, 100.0, h=h)

    np.testing.assert_allclose(
        scaled,
        physical - 5.0 * np.log10(h),
        atol=1e-12,
    )


@pytest.mark.parametrize("h", [0.5, 0.67, 0.7, 1.0])
def test_apparent_magnitude_h_scaling(h):
    """Tests that apparent magnitudes follow the expected little-h convention."""
    physical = apparent_magnitude_from_luminosity_distance(-20.0, 100.0)
    scaled = apparent_magnitude_from_luminosity_distance(-20.0, 100.0, h=h)

    np.testing.assert_allclose(
        scaled,
        physical + 5.0 * np.log10(h),
        atol=1e-12,
    )


@pytest.mark.parametrize("h", [None, 0.67])
def test_absolute_magnitude_zero_distance(h):
    """Tests that zero distance produces positive infinite absolute magnitude."""
    result = absolute_magnitude_from_luminosity_distance(
        25.0,
        [0.0, 100.0],
        h=h,
    )

    assert np.isposinf(result[0])
    assert np.isfinite(result[1])


@pytest.mark.parametrize("h", [None, 0.67])
def test_apparent_magnitude_zero_distance(h):
    """Tests that zero distance produces negative infinite apparent magnitude."""
    result = apparent_magnitude_from_luminosity_distance(
        -20.0,
        [0.0, 100.0],
        h=h,
    )

    assert np.isneginf(result[0])
    assert np.isfinite(result[1])


@pytest.mark.parametrize(
    "function, magnitude",
    [
        (absolute_magnitude_from_luminosity_distance, 25.0),
        (apparent_magnitude_from_luminosity_distance, -20.0),
    ],
)
@pytest.mark.parametrize("h", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_distance_based_magnitude_rejects_invalid_h(function, magnitude, h):
    """Tests that distance-based magnitude conversions reject invalid h."""
    with pytest.raises(ValueError, match="h must be finite and positive"):
        function(magnitude, 100.0, h=h)


@pytest.mark.parametrize("h", [None, 0.67])
def test_cosmology_and_distance_conversions_with_h(cosmology, h):
    """Tests that both magnitude conversion interfaces agree across backends."""
    from lfkit.cosmology.distances import luminosity_distance_mpc

    z = np.array([0.1, 0.5, 1.0, 2.0])
    distance = luminosity_distance_mpc(cosmology, z)

    kwargs = {
        "h": h,
        "k_correction": 0.3,
        "e_correction": 0.1,
    }

    np.testing.assert_allclose(
        magnitudes.absolute_magnitude(cosmology, z, 25.0, **kwargs),
        absolute_magnitude_from_luminosity_distance(25.0, distance, **kwargs),
        rtol=1e-12,
    )

    np.testing.assert_allclose(
        magnitudes.apparent_magnitude(cosmology, z, -20.0, **kwargs),
        apparent_magnitude_from_luminosity_distance(-20.0, distance, **kwargs),
        rtol=1e-12,
    )


@pytest.mark.parametrize("h", [None, 0.67])
def test_cosmology_magnitude_conversions_at_zero_redshift(cosmology, h):
    """Tests that cosmology-based conversions preserve the zero-redshift limit."""
    absolute = magnitudes.absolute_magnitude(cosmology, 0.0, 25.0, h=h)

    apparent = magnitudes.apparent_magnitude(cosmology, 0.0, -20.0, h=h)

    assert np.isposinf(absolute)
    assert np.isneginf(apparent)
