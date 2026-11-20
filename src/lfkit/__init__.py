"""Top-level package for LFKit."""

from __future__ import annotations

from lfkit.api.conditional_luminosity_function import ConditionalLuminosityFunction
from lfkit.api.corrections import Corrections
from lfkit.api.luminosity_function import LuminosityFunction

try:
    from lfkit._version import version as __version__
except ImportError:
    __version__ = "unknown"


__all__ = [
    "ConditionalLuminosityFunction",
    "Corrections",
    "LuminosityFunction",
]

__author__ = """Niko Sarcevic and collaborators."""
