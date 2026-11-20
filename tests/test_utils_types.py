"""Unit tests for ``lfkit.utils.types``."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, get_args, get_origin

import numpy as np
from numpy.typing import NDArray

import lfkit.utils.types as lf_types


def test_types_exports_expected_public_names() -> None:
    """Tests that the expected public typing aliases are exported."""
    expected = {
        "ConditionalParameter",
        "Cosmology",
        "FloatArray",
        "FloatInput",
        "LuminosityFunction",
        "ParameterModel",
        "ParameterValue",
    }

    assert set(lf_types.__all__) == expected


def test_float_array_alias() -> None:
    """Tests that FloatArray represents a float64 NumPy array."""
    assert lf_types.FloatArray == NDArray[np.float64]


def test_float_input_alias() -> None:
    """Tests that FloatInput supports scalar, sequence, and array inputs."""
    assert set(get_args(lf_types.FloatInput)) == {
        float,
        Sequence[float],
        lf_types.FloatArray,
    }


def test_parameter_value_alias() -> None:
    """Tests that parameter values are numerical inputs."""
    assert lf_types.ParameterValue == lf_types.FloatInput


def test_parameter_model_alias() -> None:
    """Tests that parameter models accept arbitrary arguments."""
    assert get_origin(lf_types.ParameterModel) is Callable
    assert get_args(lf_types.ParameterModel) == (
        Ellipsis,
        lf_types.ParameterValue,
    )


def test_conditional_parameter_alias() -> None:
    """Tests that conditional parameters support values and callables."""
    assert set(get_args(lf_types.ConditionalParameter)) == {
        *get_args(lf_types.ParameterValue),
        lf_types.ParameterModel,
    }


def test_luminosity_function_alias() -> None:
    """Tests that luminosity functions accept arbitrary arguments."""
    assert get_origin(lf_types.LuminosityFunction) is Callable
    assert get_args(lf_types.LuminosityFunction) == (
        Ellipsis,
        lf_types.FloatArray,
    )


def test_cosmology_alias() -> None:
    """Tests that the cosmology alias remains backend-independent."""
    assert lf_types.Cosmology is Any
