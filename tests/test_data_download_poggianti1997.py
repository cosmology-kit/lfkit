"""Tests for ``lfkit.data.poggianti1997.download``."""

from __future__ import annotations

from astropy.table import Table
from astroquery.utils.commons import TableList

import lfkit.data.poggianti1997.download as dl


def test_main_creates_package_directory(tmp_path, monkeypatch):
    """Tests that main creates the package directory and init files."""
    destination = tmp_path / "poggianti1997"
    monkeypatch.setattr(dl, "PKG_DATADIR", destination)

    tables = TableList(
        [
            (
                "J/A+AS/122/399/table1",
                Table({"z": [0.0, 0.5], "correction": [0.1, 0.2]}),
            ),
        ]
    )

    monkeypatch.setattr(
        dl.Vizier,
        "get_catalogs",
        lambda catalog: tables,
    )

    dl.main()

    assert destination.is_dir()
    assert (destination / "__init__.py").exists()
    assert (tmp_path / "__init__.py").exists()


def test_main_writes_csv_files(tmp_path, monkeypatch):
    """Tests that main writes each VizieR table with its short name."""
    destination = tmp_path / "poggianti1997"
    monkeypatch.setattr(dl, "PKG_DATADIR", destination)

    tables = TableList(
        [
            (
                "J/A+AS/122/399/kcorr",
                Table({"z": [0.0, 0.5], "correction": [0.1, 0.2]}),
            ),
            (
                "J/A+AS/122/399/ecorr",
                Table({"z": [0.0, 0.5], "correction": [0.3, 0.4]}),
            ),
        ]
    )

    monkeypatch.setattr(
        dl.Vizier,
        "get_catalogs",
        lambda catalog: tables,
    )

    dl.main()

    assert {path.name for path in destination.glob("*.csv")} == {
        "kcorr.csv",
        "ecorr.csv",
    }

    kcorr = Table.read(destination / "kcorr.csv", format="csv")
    ecorr = Table.read(destination / "ecorr.csv", format="csv")

    assert list(kcorr["correction"]) == [0.1, 0.2]
    assert list(ecorr["correction"]) == [0.3, 0.4]


def test_main_uses_correct_catalog(tmp_path, monkeypatch):
    """Tests that main requests the expected VizieR catalog."""
    called = {}

    def fake_get_catalogs(catalog):
        called["catalog"] = catalog
        return TableList([])

    monkeypatch.setattr(dl.Vizier, "get_catalogs", fake_get_catalogs)
    monkeypatch.setattr(dl, "PKG_DATADIR", tmp_path / "unused")

    dl.main()

    assert called["catalog"] == dl.CATALOG
