"""Generic file and packaged-resource I/O utilities."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np

__all__ = [
    "load_csv",
    "load_packaged_csv",
]


def load_csv(path: str | Path) -> np.ndarray:
    """Load a headered CSV file into a NumPy structured array.

    Args:
        path: Path to the CSV file.

    Returns:
        Structured array with named columns.

    Raises:
        ValueError: If the file does not contain named columns.
    """
    arr = np.genfromtxt(
        str(path),
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    if arr.dtype.names is None:
        raise ValueError(f"Failed to read CSV with headers: {path}")

    return np.atleast_1d(arr)


def load_packaged_csv(name: str, *, pkg: str) -> np.ndarray:
    """Load a CSV resource shipped inside a Python package.

    Args:
        name: Resource filename.
        pkg: Package containing the resource.

    Returns:
        Structured array containing the CSV data.
    """
    resource = resources.files(pkg).joinpath(name)

    with resources.as_file(resource) as path:
        return load_csv(path)
