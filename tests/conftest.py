"""Shared pytest fixtures for LFKit."""

from __future__ import annotations

import pytest
from astropy.cosmology import FlatLambdaCDM, LambdaCDM, w0waCDM


@pytest.fixture(
    params=[
        "astropy_flat",
        "astropy_curved",
        "astropy_w0wa",
        "ccl_flat",
        "ccl_curved",
        "ccl_w0wa",
    ]
)
def cosmology(request):
    """Return a cosmology from either supported backend."""
    if request.param == "astropy_flat":
        return FlatLambdaCDM(H0=67.0, Om0=0.3)

    if request.param == "astropy_curved":
        return LambdaCDM(H0=67.0, Om0=0.3, Ode0=0.6)

    if request.param == "astropy_w0wa":
        return w0waCDM(
            H0=67.0,
            Om0=0.3,
            Ode0=0.6,
            w0=-0.9,
            wa=0.2,
        )

    ccl = pytest.importorskip("pyccl")

    params = {
        "Omega_c": 0.25,
        "Omega_b": 0.05,
        "h": 0.67,
        "sigma8": 0.81,
        "n_s": 0.96,
    }

    if request.param == "ccl_curved":
        params["Omega_k"] = 0.1

    if request.param == "ccl_w0wa":
        params.update(Omega_k=0.1, w0=-0.9, wa=0.2)

    return ccl.Cosmology(**params)
