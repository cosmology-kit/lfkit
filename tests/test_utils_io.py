"""Unit tests for generic I/O utilities."""

from __future__ import annotations

import numpy as np

from lfkit.utils import io
from lfkit.utils.io import load_csv, load_packaged_csv


def test_load_csv_reads_headered_table(tmp_path):
    """Tests that load_csv reads a headered CSV into a structured array with named columns."""
    csv = tmp_path / "tab.csv"
    csv.write_text("recno,z,Filt,E,Sc\n1,0.0,b1,0.0,0.1\n2,0.1,b1,0.1,0.2\n")
    tab = load_csv(csv)
    assert tab.dtype.names is not None
    assert set(tab.dtype.names) >= {"recno", "z", "Filt", "E", "Sc"}
    assert len(tab) == 2


def test_load_csv_accepts_string_path(tmp_path) -> None:
    """Tests that load_csv accepts string paths."""
    csv = tmp_path / "tab.csv"
    csv.write_text("recno,z,Filt,E\n1,0.0,b1,0.1\n")

    tab = load_csv(str(csv))

    assert tab.dtype.names is not None
    assert set(tab.dtype.names) >= {"recno", "z", "Filt", "E"}


def test_io_exports_expected_public_names() -> None:
    """Tests the generic I/O public API."""
    assert set(io.__all__) == {"load_csv", "load_packaged_csv"}


def test_load_packaged_csv_reads_existing_resource() -> None:
    """Tests reading a CSV directly from packaged resources."""
    table = load_packaged_csv(
        "kcorr.csv",
        pkg="lfkit.data.poggianti1997",
    )

    assert table.size > 0
    assert table.dtype.names is not None
    assert "Filt" in table.dtype.names
    assert "z" in table.dtype.names


def test_load_csv_returns_one_dimensional_single_row(tmp_path) -> None:
    """Tests consistent shape for a one-row CSV file."""
    path = tmp_path / "table.csv"
    path.write_text("x,y\n1.0,2.0\n")

    table = load_csv(path)

    assert table.shape == (1,)
    np.testing.assert_allclose(table["x"], [1.0])
    np.testing.assert_allclose(table["y"], [2.0])
