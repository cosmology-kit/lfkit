"""Poggianti (1997) k- and e-correction tables and interpolators.

Loads Poggianti (1997) k- and e-correction curves from CSV tables, builds
interpolators for those curves (PCHIP, Akima, or linear), and optionally remaps
the e-correction redshift grid to a target cosmology by matching lookback time.

Generic CSV loading lives in ``lfkit.utils.io``.
Interpolation utilities live in ``lfkit.utils.interpolation``.
Unit conversions live in ``lfkit.utils.units``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

from lfkit.cosmo.cosmology import cosmo_object as build_cosmo
from lfkit.cosmology.times import z_from_lookback_time
from lfkit.utils.interpolation import (
    InterpMethod,
    Interpolator,
    build_1d_interpolator,
    prep_strictly_increasing_xy,
)
from lfkit.utils.io import load_csv, load_packaged_csv
from lfkit.utils.units import h0_km_s_mpc_to_gyr_inv

__all__ = (
    "available_from_table",
    "available_pairs",
    "describe_poggianti1997_available",
    "extract_sed_spectrum",
    "extract_series",
    "load_poggianti1997_tables",
    "make_ecorr_interpolator",
    "make_kcorr_interpolator",
    "poggianti1997_lookback_time_gyr",
    "poggianti1997_time_since_bb_gyr",
    "poggianti1997_to_accelerating_redshift",
)


POGGIANTI1997_PKG: Final[str] = "lfkit.data.poggianti1997"


def available_from_table(tab: np.ndarray) -> tuple[list[str], list[str]]:
    """Return available band labels and SED columns in a Poggianti-style table.

    Args:
        tab: Structured array loaded from a Poggianti VizieR CSV.

    Returns:
        A tuple ``(bands, seds)`` where:
        - ``bands`` are unique values from the ``Filt`` column (sorted)
        - ``seds`` are SED column names excluding metadata columns

    Raises:
        ValueError: If required columns are missing.
    """
    cols = list(tab.dtype.names or [])
    if "Filt" not in cols or "z" not in cols:
        raise ValueError(f"Unexpected Poggianti table columns: {cols}")

    filt_vals = np.asarray(tab["Filt"], dtype=str)
    bands = sorted({f.strip() for f in filt_vals if f.strip()})

    drop = {"recno", "z", "Filt"}
    seds = [c for c in cols if c not in drop]
    return bands, seds


def extract_series(
    tab: np.ndarray,
    *,
    band: str,
    sed: str,
    min_points: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a validated (z, y) correction curve for a band and SED.

    Args:
        tab: Structured array for a single correction table (k-corr or e-corr).
        band: Band label to select from the ``Filt`` column.
        sed: SED column name to extract.
        min_points: Minimum number of points required after cleaning.

    Returns:
        A tuple ``(z, y)`` where ``z`` is strictly increasing and ``y`` is the
        corresponding correction values.

    Raises:
        ValueError: If the band/SED is not present, has no finite values, or
            does not contain enough usable samples.
    """
    bands, seds = available_from_table(tab)

    if band not in bands:
        raise ValueError(f"Unknown band={band!r}. Available: {bands}")
    if sed not in seds:
        raise ValueError(f"Unknown sed={sed!r}. Available: {seds}")

    f = np.char.strip(np.asarray(tab["Filt"], dtype=str))
    mask = f == str(band)
    if not np.any(mask):
        raise ValueError(f"No rows found for band={band!r}. Available: {bands}")

    z = np.asarray(tab["z"][mask], float)
    y = np.asarray(tab[sed][mask], float)

    ok = np.isfinite(z) & np.isfinite(y)
    z = z[ok]
    y = y[ok]
    if z.size == 0:
        raise ValueError(
            f"No finite values for band={band!r}, sed={sed!r} in this table."
        )

    order = np.argsort(z, kind="stable")
    z = z[order]
    y = y[order]

    keep = np.ones_like(z, dtype=bool)
    keep[1:] = z[1:] > z[:-1]
    z = z[keep]
    y = y[keep]

    if z.size < int(min_points):
        raise ValueError(
            f"Too few points for band={band!r}, sed={sed!r}: n={z.size}. "
            f"Need at least {min_points}."
        )

    return z, y


def available_pairs(tab: np.ndarray, *, min_points: int = 5) -> dict[str, list[str]]:
    """List usable (band -> SEDs) pairs in a Poggianti-style table.

    Args:
        tab: Structured array for a single correction table (k-corr or e-corr).
        min_points: Minimum number of samples required per extracted series.

    Returns:
        Mapping from band label to a list of SED labels that have usable data.
    """
    bands, seds = available_from_table(tab)
    out: dict[str, list[str]] = {b: [] for b in bands}
    for b in bands:
        for s in seds:
            try:
                extract_series(tab, band=b, sed=s, min_points=min_points)
            except ValueError:
                continue
            out[b].append(s)
    return out


def load_poggianti1997_tables(
    *,
    band: str = "r",
    sed: str = "E",
    kcorr_path: str | Path | None = None,
    ecorr_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Poggianti (1997) k- and e-correction curves for a band and SED.

    Args:
        band: Band/filter label in the tables (e.g. ``"r"``, ``"B"``, ``"V"``).
        sed: SED column label (e.g. ``"E"``, ``"Sa"``, ``"Sc"``).
        kcorr_path: Optional path to ``kcorr.csv``. If not provided, the
            packaged file is used.
        ecorr_path: Optional path to ``ecorr.csv``. If not provided, the
            packaged file is used.

    Returns:
        Tuple ``(z_k, kcorr, z_e, ecorr)``:
        - ``z_k`` and ``kcorr`` are the k-correction curve.
        - ``z_e`` and ``ecorr`` are the e-correction curve.

    Raises:
        ValueError: If the requested (band, sed) is not available or contains
            insufficient usable samples.
    """
    ktab = (
        load_packaged_csv("kcorr.csv", pkg=POGGIANTI1997_PKG)
        if kcorr_path is None
        else load_csv(kcorr_path)
    )
    etab = (
        load_packaged_csv("ecorr.csv", pkg=POGGIANTI1997_PKG)
        if ecorr_path is None
        else load_csv(ecorr_path)
    )

    z_k, kcorr = extract_series(ktab, band=band, sed=sed)
    z_e, ecorr = extract_series(etab, band=band, sed=sed)
    return z_k, kcorr, z_e, ecorr


def describe_poggianti1997_available(
    *,
    kcorr_path: str | Path | None = None,
    ecorr_path: str | Path | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Summarize available bands and SED columns in Poggianti CSV tables.

    Args:
        kcorr_path: Optional path to ``kcorr.csv``. If not provided, the
            packaged file is used.
        ecorr_path: Optional path to ``ecorr.csv``. If not provided, the
            packaged file is used.

    Returns:
        A dictionary with keys ``"kcorr"`` and ``"ecorr"``. Each contains:
        - ``"bands"``: list of available band labels
        - ``"seds"``: list of available SED column labels
    """
    ktab = (
        load_packaged_csv("kcorr.csv", pkg=POGGIANTI1997_PKG)
        if kcorr_path is None
        else load_csv(kcorr_path)
    )
    etab = (
        load_packaged_csv("ecorr.csv", pkg=POGGIANTI1997_PKG)
        if ecorr_path is None
        else load_csv(ecorr_path)
    )

    k_bands, k_seds = available_from_table(ktab)
    e_bands, e_seds = available_from_table(etab)
    return {
        "kcorr": {"bands": k_bands, "seds": k_seds},
        "ecorr": {"bands": e_bands, "seds": e_seds},
    }


def poggianti1997_time_since_bb_gyr(z: np.ndarray | float) -> np.ndarray:
    """Return cosmic time since the Big Bang in the Poggianti (1997) cosmology.

    Args:
        z: Redshift value(s).

    Returns:
        Cosmic time since the Big Bang at each redshift, in Gyr.

    Notes:
        This uses the decelerating cosmology assumed by Poggianti (1997)
        (q0 = 0.225, H0 = 50 km/s/Mpc). It is intended for lookback-time
        matching when remapping e-about to a different cosmology.
    """
    q0 = 0.225
    h0_km_s_mpc = 50.0

    z = np.asarray(z, dtype=float)
    h0_gyr_inv = h0_km_s_mpc_to_gyr_inv(h0_km_s_mpc)

    term1 = -4.0 * q0 / (h0_gyr_inv * np.power(1.0 - 2.0 * q0, 1.5))
    root_val = np.sqrt((1.0 + 2.0 * q0 * z) / (1.0 - 2.0 * q0))
    term2 = root_val
    term3 = 2.0 * (1.0 - (1.0 + 2.0 * q0 * z) / (1.0 - 2.0 * q0))
    term4 = 0.25 * np.log(np.abs((1.0 + root_val) / (1.0 - root_val)))

    return term1 * (term2 / term3 + term4)


def poggianti1997_lookback_time_gyr(z: np.ndarray | float) -> np.ndarray:
    """Return lookback time in the Poggianti (1997) cosmology.

    Args:
        z: Redshift value(s).

    Returns:
        Lookback time to each redshift (relative to z=0), in Gyr.
    """
    z = np.asarray(z, dtype=float)
    return poggianti1997_time_since_bb_gyr(0.0) - poggianti1997_time_since_bb_gyr(z)


def poggianti1997_to_accelerating_redshift(
    z_dec: np.ndarray | float,
    *,
    cosmo_obj,
    zmax: float = 20.0,
    nz: int = 4096,
) -> np.ndarray:
    """Map Poggianti (1997) redshifts to a target cosmology via lookback time.

    Args:
        z_dec: Redshift value(s) in the Poggianti (1997) cosmology.
        cosmo_obj: Target cosmology used to compute lookback times.
        zmax: Maximum redshift used to build the inversion grid.
        nz: Number of samples used to build the inversion grid.

    Returns:
        Redshift value(s) in the target cosmology with matched lookback time.
    """
    z_dec = np.asarray(z_dec, dtype=float)
    t_lb = poggianti1997_lookback_time_gyr(z_dec)
    return z_from_lookback_time(cosmo_obj, t_lb, zmax=zmax, nz=nz)


def make_kcorr_interpolator(
    z_k: np.ndarray,
    kcorr: np.ndarray,
    *,
    method: InterpMethod = "pchip",
    extrapolate: bool = True,
    z_end: float = 20.0,
    tail: bool = True,
) -> Interpolator:
    """Create an interpolator for a Poggianti k-correction curve.

    Args:
        z_k: Redshift samples for the k-correction table.
        kcorr: K-correction values at ``z_k``.
        method: Interpolation method (``"pchip"``, ``"akima"``, or ``"linear"``).
        extrapolate: Whether to allow evaluation outside the tabulated range.
        z_end: Redshift endpoint for the optional high-z tail.
        tail: Whether to append a linear high-z tail out to ``z_end``.

    Returns:
        An interpolator callable ``K(z)``.
    """
    z = np.r_[0.0, np.asarray(z_k, float)]
    y = np.r_[0.0, np.asarray(kcorr, float)]
    z, y = prep_strictly_increasing_xy(z, y)

    if tail and z[-1] < z_end:
        i0 = max(0, z.size - 3)
        dz = z[-1] - z[i0]
        if dz <= 0:
            raise ValueError("Non-increasing z near the tail; cannot form slope.")
        slope = (y[-1] - y[i0]) / dz
        y_end = y[-1] + slope * (z_end - z[-1])
        z = np.r_[z, z_end]
        y = np.r_[y, y_end]

    return build_1d_interpolator(z, y, method=method, extrapolate=extrapolate)


def make_ecorr_interpolator(
    z_e: np.ndarray,
    ecorr: np.ndarray,
    *,
    original_z: bool,
    cosmo=None,
    zmap_zmax: float = 20.0,
    zmap_nz: int = 4096,
    method: InterpMethod = "pchip",
    extrapolate: bool = True,
) -> Interpolator:
    """Create an interpolator for a Poggianti e-correction curve.

    Args:
        z_e: Redshift samples for the e-correction table.
        ecorr: E-correction values at ``z_e``.
        original_z: If True, interpret ``z_e`` as Poggianti (1997) redshifts.
            If False, remap ``z_e`` to the target cosmology via lookback time.
        cosmo: Target cosmology used for the remapping when ``original_z=False``.
            If not provided, lfkit's default cosmology is used.
        zmap_zmax: Maximum redshift used to build the remapping inversion grid.
        zmap_nz: Number of samples used to build the remapping inversion grid.
        method: Interpolation method (``"pchip"``, ``"akima"``, or ``"linear"``).
        extrapolate: Whether to extrapolate beyond the tabulated domain.

    Returns:
        An interpolator callable ``E(z)``.
    """
    z_e = np.asarray(z_e, float)
    ecorr = np.asarray(ecorr, float)

    if not original_z:
        if cosmo is None:
            cosmo = build_cosmo()
        z_e = poggianti1997_to_accelerating_redshift(
            z_e, cosmo_obj=cosmo, zmax=zmap_zmax, nz=zmap_nz
        )

    z = np.r_[0.0, z_e]
    y = np.r_[0.0, ecorr]
    return build_1d_interpolator(z, y, method=method, extrapolate=extrapolate)


def extract_sed_spectrum(
    sed_tab: np.ndarray,
    sed_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a Poggianti sed.csv column as (wave_A, flux) in rest frame.

    Args:
        sed_tab: Table returned by lfkit.utils.io.load_csv for sed.csv.
        sed_col: Column name in sed.csv, e.g. "logF03".

    Returns:
        wave_A: Wavelength grid in Å (sorted ascending).
        flux: Linear flux values (10**logF) on the same grid.

    Raises:
        ValueError: If required columns are missing or sed_col not present.
    """
    cols = list(sed_tab.dtype.names or [])
    if "Lam" not in cols:
        raise ValueError(f"sed.csv missing Lam column. cols={cols}")
    if sed_col not in cols:
        raise ValueError(
            f"sed.csv: sed_col={sed_col!r} not present. Available: {cols[:20]} ..."
        )

    wave_nm = np.asarray(sed_tab["Lam"], float)
    logf = np.asarray(sed_tab[sed_col], float)

    ok = np.isfinite(wave_nm) & np.isfinite(logf)
    wave_A = 10.0 * wave_nm[ok]  # nm -> Å
    flux = 10.0 ** logf[ok]  # log10 -> linear

    order = np.argsort(wave_A)
    return wave_A[order], flux[order]
