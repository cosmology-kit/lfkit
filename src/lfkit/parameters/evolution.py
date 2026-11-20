r"""Luminosity function parameter-evolution models.

This module provides helper functions for evaluating redshift-dependent
luminosity function parameters such as ``phi_star(z)``, ``M_star(z)``, and
``alpha(z)``.

These helpers are used by luminosity function models that allow parameter
evolution with redshift. They evaluate only the parameters, not the full
luminosity function.

Built-in options include constant evolution and simple linear forms commonly
used in luminosity function analyses.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from lfkit.utils.evaluators import evaluate_parameter
from lfkit.utils.types import FloatArray, FloatInput, ParameterModel, ParameterValue
from lfkit.utils.validators import validate_array

__all__ = [
    "ALPHA_MODELS",
    "M_STAR_MODELS",
    "PHI_STAR_MODELS",
    "alpha_constant",
    "alpha_linear",
    "available_lf_parameter_models",
    "evaluate_lf_parameters",
    "get_parameter_model",
    "m_star_constant",
    "m_star_linear_q",
    "phi_star_constant",
    "phi_star_linear_p",
    "register_alpha_model",
    "register_m_star_model",
    "register_phi_star_model",
]


def phi_star_constant(
    z: FloatInput,
    *,
    phi_star: ParameterValue,
) -> FloatArray:
    r"""Return a constant Schechter normalization over redshift.

    This evaluates

    .. math::

       \phi_\star(z) = \phi_\star.

    Args:
        z: Redshift value or array-like of redshift values.
        phi_star: Constant Schechter normalization.

    Returns:
        Schechter normalization evaluated at ``z`` with the same shape as ``z``.
    """
    z_arr = validate_array(z, name="z")
    return evaluate_parameter(
        phi_star,
        z_arr,
        name="phi_star",
        domain="non_negative",
    )


def phi_star_linear_p(
    z: FloatInput,
    *,
    phi_0_star: ParameterValue,
    p: ParameterValue,
) -> FloatArray:
    r"""Return a Schechter normalization with density evolution.

    This evaluates

    .. math::

       \phi_\star(z) = \phi_{0,\star} 10^{0.4 p z}.

    Args:
        z: Redshift value or array-like of redshift values.
        phi_0_star: Schechter normalization at redshift zero.
        p: Density-evolution parameter.

    Returns:
        Redshift-dependent Schechter normalization evaluated at ``z``.
    """
    z_arr = validate_array(z, name="z")
    phi_0_star = evaluate_parameter(
        phi_0_star,
        z_arr,
        name="phi_0_star",
        domain="non_negative",
    )
    p = evaluate_parameter(p, z_arr, name="p")

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        values = phi_0_star * 10.0 ** (0.4 * p * z_arr)

    return evaluate_parameter(
        values,
        z_arr,
        name="phi_star",
        domain="non_negative",
    )


def m_star_constant(
    z: FloatInput,
    *,
    m_star: ParameterValue,
) -> FloatArray:
    r"""Return a constant characteristic magnitude over redshift.

    This evaluates

    .. math::

       M_\star(z) = M_\star.

    Args:
        z: Redshift value or array-like of redshift values.
        m_star: Constant characteristic magnitude.

    Returns:
        Characteristic magnitude evaluated at ``z`` with the same shape as ``z``.
    """
    z_arr = validate_array(z, name="z")
    return evaluate_parameter(m_star, z_arr, name="m_star")


def m_star_linear_q(
    z: FloatInput,
    *,
    m_0_star: ParameterValue,
    q: ParameterValue,
    z_ref: ParameterValue = 0.1,
) -> FloatArray:
    r"""Return a characteristic magnitude with luminosity evolution.

    This evaluates

    .. math::

       M_\star(z) = M_{0,\star} - q (z - z_{\mathrm{ref}}).

    Args:
        z: Redshift value or array-like of redshift values.
        m_0_star: Characteristic magnitude at the reference redshift.
        q: Luminosity-evolution parameter.
        z_ref: Reference redshift.

    Returns:
        Redshift-dependent characteristic magnitude evaluated at ``z``.
    """
    z_arr = validate_array(z, name="z")
    m_0_star = evaluate_parameter(m_0_star, z_arr, name="m_0_star")
    q = evaluate_parameter(q, z_arr, name="q")
    z_ref = evaluate_parameter(z_ref, z_arr, name="z_ref")

    with np.errstate(over="ignore", invalid="ignore"):
        values = m_0_star - q * (z_arr - z_ref)

    return evaluate_parameter(values, z_arr, name="m_star")


def alpha_constant(
    z: FloatInput,
    *,
    alpha: ParameterValue,
) -> FloatArray:
    r"""Return a constant faint-end slope over redshift.

    This evaluates

    .. math::

       \alpha(z) = \alpha.

    Args:
        z: Redshift value or array-like of redshift values.
        alpha: Constant faint-end slope.

    Returns:
        Faint-end slope evaluated at ``z`` with the same shape as ``z``.
    """
    z_arr = validate_array(z, name="z")
    return evaluate_parameter(alpha, z_arr, name="alpha")


def alpha_linear(
    z: FloatInput,
    *,
    alpha_0: ParameterValue,
    alpha_1: ParameterValue,
    z_ref: ParameterValue = 0.1,
) -> FloatArray:
    r"""Return a faint-end slope that varies linearly with redshift.

    This evaluates

    .. math::

       \alpha(z) = \alpha_0 + \alpha_1 (z - z_{\mathrm{ref}}).

    Args:
        z: Redshift value or array-like of redshift values.
        alpha_0: Faint-end slope at the reference redshift.
        alpha_1: Linear redshift-evolution coefficient.
        z_ref: Reference redshift.

    Returns:
        Redshift-dependent faint-end slope evaluated at ``z``.
    """
    z_arr = validate_array(z, name="z")
    alpha_0 = evaluate_parameter(alpha_0, z_arr, name="alpha_0")
    alpha_1 = evaluate_parameter(alpha_1, z_arr, name="alpha_1")
    z_ref = evaluate_parameter(z_ref, z_arr, name="z_ref")

    with np.errstate(over="ignore", invalid="ignore"):
        values = alpha_0 + alpha_1 * (z_arr - z_ref)

    return evaluate_parameter(values, z_arr, name="alpha")


def get_parameter_model(
    model_name: str,
    registry: Mapping[str, ParameterModel],
    *,
    model_kind: str,
) -> ParameterModel:
    r"""Return a registered luminosity function parameter model.

    Args:
        model_name: Name of the requested parameter-evolution model.
        registry: Mapping from model names to parameter-evolution callables.
        model_kind: Human-readable parameter kind used in error messages.

    Returns:
        Registered parameter-evolution callable.

    Raises:
        ValueError: If ``model_name`` is not present in ``registry``.
    """
    try:
        return registry[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown {model_kind} '{model_name}'. Available models: {list(registry)}."
        ) from exc


def available_lf_parameter_models() -> dict[str, list[str]]:
    """Return available luminosity function parameter-evolution models.

    Returns:
        Dictionary mapping parameter names to sorted lists of registered model names.
    """
    return {
        "phi_star": sorted(PHI_STAR_MODELS),
        "m_star": sorted(M_STAR_MODELS),
        "alpha": sorted(ALPHA_MODELS),
    }


def _register_parameter_model(
    name: str,
    model: ParameterModel,
    registry: dict[str, ParameterModel],
    *,
    model_kind: str,
    overwrite: bool = False,
) -> None:
    """Register a luminosity function parameter-evolution model.

    Args:
        name: Name used to register the model.
        model: Callable parameter-evolution model.
        registry: Registry to update.
        model_kind: Human-readable parameter kind used in error messages.
        overwrite: If ``True``, replace an existing model with the same name.

    Raises:
        ValueError: If ``name`` is empty or already registered and
            ``overwrite=False``.
        TypeError: If ``model`` is not callable.
    """
    if not name:
        raise ValueError(f"{model_kind} model name cannot be empty.")

    if not callable(model):
        raise TypeError(f"{model_kind} model must be callable.")

    if name in registry and not overwrite:
        raise ValueError(
            f"{model_kind} model '{name}' is already registered. "
            "Use overwrite=True to replace it."
        )

    registry[name] = model


def register_phi_star_model(
    name: str,
    model: ParameterModel,
    *,
    overwrite: bool = False,
) -> None:
    """Register a ``phi_star`` evolution model.

    Args:
        name: Name used to register the model.
        model: Callable accepting redshift and returning ``phi_star(z)``.
        overwrite: If ``True``, replace an existing model with the same name.
    """
    _register_parameter_model(
        name,
        model,
        PHI_STAR_MODELS,
        model_kind="phi_star",
        overwrite=overwrite,
    )


def register_m_star_model(
    name: str,
    model: ParameterModel,
    *,
    overwrite: bool = False,
) -> None:
    """Register an ``M_star`` evolution model.

    Args:
        name: Name used to register the model.
        model: Callable accepting redshift and returning ``M_star(z)``.
        overwrite: If ``True``, replace an existing model with the same name.
    """
    _register_parameter_model(
        name,
        model,
        M_STAR_MODELS,
        model_kind="m_star",
        overwrite=overwrite,
    )


def register_alpha_model(
    name: str,
    model: ParameterModel,
    *,
    overwrite: bool = False,
) -> None:
    """Register an ``alpha`` evolution model.

    Args:
        name: Name used to register the model.
        model: Callable accepting redshift and returning ``alpha(z)``.
        overwrite: If ``True``, replace an existing model with the same name.
    """
    _register_parameter_model(
        name,
        model,
        ALPHA_MODELS,
        model_kind="alpha",
        overwrite=overwrite,
    )


def evaluate_lf_parameters(
    z: FloatInput,
    *,
    phi_model: str = "linear_p",
    phi_kwargs: Mapping[str, ParameterValue] | None = None,
    m_star_model: str = "linear_q",
    m_star_kwargs: Mapping[str, ParameterValue] | None = None,
    alpha_model: str = "constant",
    alpha_kwargs: Mapping[str, ParameterValue] | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    r"""Evaluate luminosity function parameters at redshift ``z``.

    Args:
        z: Redshift value or array-like of redshift values.
        phi_model: Registered evolution model used for ``phi_star``.
        phi_kwargs: Keyword arguments passed to the selected ``phi_star`` model.
        m_star_model: Registered evolution model used for ``M_star``.
        m_star_kwargs: Keyword arguments passed to the selected ``M_star`` model.
        alpha_model: Registered evolution model used for ``alpha``.
        alpha_kwargs: Keyword arguments passed to the selected ``alpha`` model.

    Returns:
        Tuple ``(phi_star, m_star, alpha)`` evaluated at ``z``.

    Raises:
        ValueError: If an unsupported parameter-evolution model is requested.
    """
    z_arr = validate_array(z, name="z")

    phi_kwargs_dict: dict[str, ParameterValue] = (
        {} if phi_kwargs is None else dict(phi_kwargs)
    )
    m_star_kwargs_dict: dict[str, ParameterValue] = (
        {} if m_star_kwargs is None else dict(m_star_kwargs)
    )
    alpha_kwargs_dict: dict[str, ParameterValue] = (
        {} if alpha_kwargs is None else dict(alpha_kwargs)
    )

    phi_fn = get_parameter_model(
        phi_model,
        PHI_STAR_MODELS,
        model_kind="phi_model",
    )
    m_star_fn = get_parameter_model(
        m_star_model,
        M_STAR_MODELS,
        model_kind="m_star_model",
    )
    alpha_fn = get_parameter_model(
        alpha_model,
        ALPHA_MODELS,
        model_kind="alpha_model",
    )

    phi_star = evaluate_parameter(
        phi_fn(z_arr, **phi_kwargs_dict),
        z_arr,
        name="phi_star",
        domain="non_negative",
    )
    m_star = evaluate_parameter(
        m_star_fn(z_arr, **m_star_kwargs_dict),
        z_arr,
        name="m_star",
    )
    alpha = evaluate_parameter(
        alpha_fn(z_arr, **alpha_kwargs_dict),
        z_arr,
        name="alpha",
    )

    return phi_star, m_star, alpha


PHI_STAR_MODELS: dict[str, ParameterModel] = {
    "constant": phi_star_constant,
    "linear_p": phi_star_linear_p,
}

M_STAR_MODELS: dict[str, ParameterModel] = {
    "constant": m_star_constant,
    "linear_q": m_star_linear_q,
}

ALPHA_MODELS: dict[str, ParameterModel] = {
    "constant": alpha_constant,
    "linear": alpha_linear,
}
