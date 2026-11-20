"""Unit tests for ``lfkit.utils.numerics``."""

from __future__ import annotations

import numpy as np
import pytest

from lfkit.utils.numerics import safe_divide, safe_power10


def test_safe_divide_returns_ratio_for_positive_denominator() -> None:
    """Tests that safe division returns the ordinary ratio for positive denominators."""
    result = safe_divide(
        np.array([2.0, 6.0, 12.0]),
        np.array([1.0, 2.0, 3.0]),
    )

    np.testing.assert_allclose(result, np.array([2.0, 3.0, 4.0]))


def test_safe_divide_returns_zero_for_zero_denominator() -> None:
    """Tests that safe division returns zero for zero denominators."""
    result = safe_divide(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 0.0, 2.0]),
    )

    np.testing.assert_allclose(result, np.array([1.0, 0.0, 1.5]))


def test_safe_divide_returns_zero_for_negative_denominator() -> None:
    """Tests that safe division returns zero for negative denominators."""
    result = safe_divide(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, -1.0, 2.0]),
    )

    np.testing.assert_allclose(result, np.array([1.0, 0.0, 1.5]))


def test_safe_divide_supports_scalar_inputs() -> None:
    """Tests that safe division supports scalar inputs."""
    result = safe_divide(4.0, 2.0)

    assert result.shape == ()
    assert result == pytest.approx(2.0)


def test_safe_divide_returns_zero_for_scalar_zero_denominator() -> None:
    """Tests that safe division returns zero for scalar zero denominator."""
    result = safe_divide(4.0, 0.0)

    assert result.shape == ()
    assert result == pytest.approx(0.0)


def test_safe_divide_supports_broadcastable_inputs() -> None:
    """Tests that safe division supports broadcastable numerator and denominator."""
    result = safe_divide(
        np.array([[2.0], [4.0]], dtype=float),
        np.array([1.0, 2.0, 0.0], dtype=float),
    )

    expected = np.array(
        [
            [2.0, 1.0, 0.0],
            [4.0, 2.0, 0.0],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(result, expected)


def test_safe_divide_rejects_unbroadcastable_inputs() -> None:
    """Tests that safe division requires broadcastable inputs."""
    with pytest.raises(ValueError):
        safe_divide(
            np.ones((2, 3), dtype=float),
            np.ones((4,), dtype=float),
        )


def test_safe_divide_uses_custom_fill_value() -> None:
    """Tests that safe division uses the requested fill value."""
    result = safe_divide(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 0.0, -1.0]),
        fill_value=-99.0,
    )

    np.testing.assert_allclose(result, np.array([1.0, -99.0, -99.0]))


def test_safe_divide_replaces_nonfinite_results() -> None:
    """Tests that safe division replaces non-finite results with fill value."""
    result = safe_divide(
        np.array([np.inf, 2.0, np.nan]),
        np.array([1.0, 1.0, 1.0]),
        fill_value=-1.0,
    )

    np.testing.assert_allclose(result, np.array([-1.0, 2.0, -1.0]))


def test_safe_divide_returns_float64_array() -> None:
    """Tests that safe division returns a float64 array."""
    result = safe_divide(
        np.array([1, 2, 3]),
        np.array([1, 2, 3]),
    )

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


def test_safe_power10_matches_base_ten_power() -> None:
    """Tests that safe_power10 matches ordinary base-ten powers."""
    result = safe_power10(np.array([-1.0, 0.0, 2.0]))

    np.testing.assert_allclose(result, np.array([0.1, 1.0, 100.0]))


def test_safe_power10_clips_large_positive_exponents() -> None:
    """Tests that safe_power10 clips large positive exponents."""
    result = safe_power10(
        np.array([1.0, 5.0]),
        min_exponent=-2.0,
        max_exponent=2.0,
    )

    np.testing.assert_allclose(result, np.array([10.0, 100.0]))


def test_safe_power10_clips_large_negative_exponents() -> None:
    """Tests that safe_power10 clips large negative exponents."""
    result = safe_power10(
        np.array([-5.0, -1.0]),
        min_exponent=-2.0,
        max_exponent=2.0,
    )

    np.testing.assert_allclose(result, np.array([0.01, 0.1]))


def test_safe_power10_supports_scalar_input() -> None:
    """Tests that safe_power10 supports scalar input."""
    result = safe_power10(2.0)

    assert result.shape == ()
    assert result == pytest.approx(100.0)


def test_safe_power10_returns_float64_array() -> None:
    """Tests that safe_power10 returns a float64 array."""
    result = safe_power10(np.array([0, 1, 2]))

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


def test_safe_power10_rejects_nan_exponent() -> None:
    """Tests that safe_power10 rejects NaN exponents."""
    with pytest.raises(ValueError, match="exponent contains NaN"):
        safe_power10(np.nan)


def test_safe_power10_rejects_infinite_exponent() -> None:
    """Tests that safe_power10 rejects infinite exponents."""
    with pytest.raises(ValueError, match="exponent contains NaN or infinite values"):
        safe_power10(np.inf)


def test_numerics_exports_expected_public_names() -> None:
    """Tests the public numerical API."""
    from lfkit.utils import numerics

    assert numerics.__all__ == ["safe_divide", "safe_power10"]
