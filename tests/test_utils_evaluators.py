"""Unit tests for ``lfkit.utils.evaluators``."""

from __future__ import annotations

import numpy as np
import pytest

from lfkit.utils import evaluators
from lfkit.utils.evaluators import evaluate_parameter


def test_evaluators_exports_expected_public_names() -> None:
    """Tests that only the generic evaluator is exported."""
    assert evaluators.__all__ == ["evaluate_parameter"]


@pytest.mark.parametrize(
    "parameter",
    [2.0, [2.0, 2.0, 2.0], np.full(3, 2.0)],
)
def test_evaluate_parameter_constant(parameter) -> None:
    """Tests numerical parameter evaluation."""
    x = np.array([0.0, 0.5, 1.0])

    result = evaluate_parameter(parameter, x)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.shape == x.shape
    np.testing.assert_allclose(result, 2.0)


def test_evaluate_parameter_callable() -> None:
    """Tests evaluation of a coordinate-dependent callable."""
    x = np.array([0.0, 0.5, 1.0])

    result = evaluate_parameter(lambda x: 1.0 + x, x)

    np.testing.assert_allclose(result, [1.0, 1.5, 2.0])


def test_evaluate_parameter_scalar_callable_output() -> None:
    """Tests broadcasting scalar callable output."""
    x = np.arange(5.0)

    result = evaluate_parameter(lambda x: 2.0, x)

    assert result.shape == x.shape
    np.testing.assert_allclose(result, 2.0)


def test_evaluate_parameter_list_callable_output() -> None:
    """Tests conversion of callable list output."""
    x = np.arange(3.0)

    result = evaluate_parameter(lambda x: [1.0, 2.0, 3.0], x)

    assert result.dtype == np.float64
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0])


def test_evaluate_parameter_scalar_coordinate() -> None:
    """Tests scalar-coordinate evaluation."""
    result = evaluate_parameter(lambda x: 1.0 + x, 0.5)

    assert isinstance(result, np.ndarray)
    assert result.shape == ()
    assert result.dtype == np.float64
    assert result == pytest.approx(1.5)


def test_evaluate_parameter_without_coordinates() -> None:
    """Tests evaluation without independent variables."""
    result = evaluate_parameter([1.0, 2.0, 3.0])

    np.testing.assert_allclose(result, [1.0, 2.0, 3.0])


def test_evaluate_parameter_callable_without_coordinates() -> None:
    """Tests evaluation of a zero-argument callable."""
    result = evaluate_parameter(lambda: 2.0)

    assert result.shape == ()
    assert result == pytest.approx(2.0)


def test_evaluate_parameter_multidimensional_coordinates() -> None:
    """Tests preservation of multidimensional coordinate shapes."""
    x = np.arange(6.0).reshape(2, 3)

    result = evaluate_parameter(lambda x: 1.0 + x, x)

    assert result.shape == x.shape
    np.testing.assert_allclose(result, 1.0 + x)


def test_evaluate_parameter_multiple_coordinates() -> None:
    """Tests evaluation with multiple independent variables."""
    x = np.arange(3.0).reshape(3, 1)
    y = np.arange(2.0).reshape(1, 2)

    result = evaluate_parameter(lambda x, y: x + y, x, y)

    assert result.shape == (3, 2)
    np.testing.assert_allclose(result, x + y)


def test_evaluate_parameter_three_coordinates() -> None:
    """Tests evaluation with three independent variables."""
    x = np.arange(3.0).reshape(3, 1, 1)
    y = np.arange(2.0).reshape(1, 2, 1)
    z = np.arange(4.0).reshape(1, 1, 4)

    result = evaluate_parameter(lambda x, y, z: x + y + z, x, y, z)

    assert result.shape == (3, 2, 4)
    np.testing.assert_allclose(result, x + y + z)


def test_evaluate_parameter_broadcasts_partial_output() -> None:
    """Tests broadcasting output depending on only one coordinate."""
    x = np.arange(3.0).reshape(3, 1)
    y = np.arange(2.0).reshape(1, 2)

    result = evaluate_parameter(lambda x, y: x, x, y)

    assert result.shape == (3, 2)
    np.testing.assert_allclose(result, np.broadcast_to(x, (3, 2)))


def test_evaluate_parameter_rejects_incompatible_coordinates() -> None:
    """Tests rejection of incompatible coordinate shapes."""
    x = np.ones(3)
    y = np.ones(4)

    with pytest.raises(ValueError):
        evaluate_parameter(1.0, x, y)


def test_evaluate_parameter_rejects_incompatible_output() -> None:
    """Tests rejection of incompatible output shapes."""
    x = np.arange(3.0)

    with pytest.raises(ValueError, match="broadcastable"):
        evaluate_parameter(lambda x: np.ones(4), x, name="test_fn")


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_evaluate_parameter_rejects_nonfinite_output(value) -> None:
    """Tests rejection of non-finite parameter values."""
    x = np.arange(3.0)

    with pytest.raises(ValueError, match="test_fn"):
        evaluate_parameter(lambda x: value, x, name="test_fn")


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_evaluate_parameter_rejects_nonfinite_coordinates(value) -> None:
    """Tests rejection of non-finite coordinates."""
    x = np.array([0.0, value, 1.0])

    with pytest.raises(ValueError, match="coordinate"):
        evaluate_parameter(1.0, x)


@pytest.mark.parametrize(
    ("domain", "values"),
    [
        ("real", [-1.0, 0.0, 1.0]),
        ("positive", [1.0, 2.0, 3.0]),
        ("non_negative", [0.0, 1.0, 2.0]),
    ],
)
def test_evaluate_parameter_accepts_valid_domain(domain, values) -> None:
    """Tests acceptance of values within the requested domain."""
    x = np.arange(3.0)

    result = evaluate_parameter(values, x, domain=domain)

    np.testing.assert_allclose(result, values)


@pytest.mark.parametrize(
    ("domain", "values", "message"),
    [
        ("positive", [0.0, 1.0, 2.0], "must be positive"),
        ("positive", [-1.0, 1.0, 2.0], "must be positive"),
        ("non_negative", [-1.0, 0.0, 1.0], "must be non-negative"),
    ],
)
def test_evaluate_parameter_rejects_invalid_domain(
    domain,
    values,
    message,
) -> None:
    """Tests rejection of values outside the requested domain."""
    x = np.arange(3.0)

    with pytest.raises(ValueError, match=message):
        evaluate_parameter(values, x, domain=domain)


def test_evaluate_parameter_rejects_unsupported_domain() -> None:
    """Tests rejection of unsupported domain names."""
    with pytest.raises(ValueError, match="Unsupported domain"):
        evaluate_parameter(1.0, domain="invalid")
