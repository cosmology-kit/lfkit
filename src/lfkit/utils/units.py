"""Physical unit conversion utilities."""

from __future__ import annotations

import numpy as np

__all__ = [
    "h0_km_s_mpc_to_gyr_inv",
    "km_per_mpc",
    "sec_per_gyr",
]


def km_per_mpc() -> float:
    """Return the number of kilometers in one megaparsec.

    Returns:
        Conversion factor from megaparsecs (Mpc) to kilometers (km).
    """
    # IAU exact definition of astronomical unit (meters).
    au_m = 149_597_870_700.0

    # 1 arcsec in radians.
    arcsec_rad = np.pi / (180.0 * 3600.0)

    # Parsec in meters, then megaparsec in kilometers.
    pc_m = au_m / arcsec_rad
    mpc_km = (1e6 * pc_m) / 1000.0
    return mpc_km


def sec_per_gyr() -> float:
    """Return the number of seconds in one gigayear.

    Returns:
        Conversion factor from gigayears (Gyr) to seconds (s).
    """
    seconds_per_day = 86400.0
    days_per_year = 365.25
    years_per_gyr = 1e9
    return seconds_per_day * days_per_year * years_per_gyr


def h0_km_s_mpc_to_gyr_inv(h0_km_s_mpc: float) -> float:
    """Convert a Hubble constant from km/s/Mpc to 1/Gyr.

    Args:
        h0_km_s_mpc: Hubble constant in units of km/s/Mpc.

    Returns:
        The Hubble constant expressed in inverse gigayears (1/Gyr).
    """
    h0_s_inv = float(h0_km_s_mpc) / km_per_mpc()
    return h0_s_inv * sec_per_gyr()
