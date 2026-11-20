"""Unit tests for K-correction grid serialization."""

from __future__ import annotations

import numpy as np

from lfkit.corrections.kcorrect_grids import (
    kcorr_interpolators,
    load_kcorr_package,
    save_kcorr_package,
)


def test_save_and_load_kcorr_package_roundtrip(tmp_path):
    """Tests that save_kcorr_package and load_kcorr_package roundtrip core fields and shapes."""
    pkg = {
        "meta": {"tag": "x"},
        "z": np.array([0.0, 0.1, 0.2]),
        "responses_in": ["bessell_V"],
        "responses_out": ["bessell_V"],
        "responses_map": ["bessell_V"],
        "anchors": ["E", "Sc"],
        "K": {
            "E": np.array([[0.0], [0.1], [0.2]]),
            "Sc": np.array([[0.0], [0.2], [0.4]]),
        },
    }

    out = tmp_path / "pkg.npz"
    save_kcorr_package(pkg, out)
    loaded = load_kcorr_package(out)

    assert np.allclose(loaded["z"], pkg["z"])
    assert loaded["anchors"] == pkg["anchors"]
    assert loaded["responses_in"] == pkg["responses_in"]
    assert loaded["responses_out"] == pkg["responses_out"]
    assert loaded["responses_map"] == pkg["responses_map"]
    for t in pkg["anchors"]:
        assert np.allclose(loaded["K"][t], pkg["K"][t])
    assert loaded["meta"]["tag"] == "x"


def test_save_kcorr_package_creates_parent_directories(tmp_path) -> None:
    """Tests that save_kcorr_package creates missing parent directories."""
    pkg = {
        "meta": {"tag": "x"},
        "z": np.array([0.0, 0.1]),
        "responses_in": ["r"],
        "responses_out": ["r"],
        "responses_map": ["r"],
        "anchors": ["E"],
        "K": {"E": np.array([[0.0], [0.1]])},
    }
    out = tmp_path / "nested" / "pkg.npz"

    save_kcorr_package(pkg, out)

    assert out.exists()


def test_save_and_load_kcorr_package_accepts_string_path(tmp_path) -> None:
    """Tests that k-correction package I/O accepts string paths."""
    pkg = {
        "meta": {"tag": "x"},
        "z": np.array([0.0, 0.1]),
        "responses_in": ["r"],
        "responses_out": ["r"],
        "responses_map": ["r"],
        "anchors": ["E"],
        "K": {"E": np.array([[0.0], [0.1]])},
    }
    out = tmp_path / "pkg.npz"

    save_kcorr_package(pkg, str(out))
    loaded = load_kcorr_package(str(out))

    np.testing.assert_allclose(loaded["z"], pkg["z"])
    np.testing.assert_allclose(loaded["K"]["E"], pkg["K"]["E"])


def test_load_kcorr_package_removes_internal_metadata_from_meta(tmp_path) -> None:
    """Tests that load_kcorr_package removes internal metadata keys from meta."""
    pkg = {
        "meta": {"tag": "x", "version": 1},
        "z": np.array([0.0, 0.1]),
        "responses_in": ["r_in"],
        "responses_out": ["r_out"],
        "responses_map": ["r_map"],
        "anchors": ["E"],
        "K": {"E": np.array([[0.0], [0.1]])},
    }
    out = tmp_path / "pkg.npz"

    save_kcorr_package(pkg, out)
    loaded = load_kcorr_package(out)

    assert loaded["meta"] == {"tag": "x", "version": 1}
    assert "responses_in" not in loaded["meta"]
    assert "responses_out" not in loaded["meta"]
    assert "responses_map" not in loaded["meta"]
    assert "anchors" not in loaded["meta"]


def test_kcorr_package_roundtrip_supports_interpolation(tmp_path) -> None:
    """Tests that serialized packages remain usable for interpolation."""
    pkg = {
        "meta": {"backend": "kcorrect"},
        "z": np.array([0.0, 0.5, 1.0]),
        "responses_in": ["r"],
        "responses_out": ["r"],
        "responses_map": ["r"],
        "anchors": ["E"],
        "K": {"E": np.array([[0.0], [1.0], [2.0]])},
    }

    path = tmp_path / "grid.npz"

    save_kcorr_package(pkg, path)
    loaded = load_kcorr_package(path)

    interpolators = kcorr_interpolators(
        loaded,
        method="linear",
    )

    np.testing.assert_allclose(
        interpolators["E"]["r"]([0.25, 0.75]),
        [0.5, 1.5],
    )


def test_kcorr_package_does_not_require_pickle(tmp_path) -> None:
    """Tests that every stored NPZ array is accessible without pickle."""
    pkg = {
        "meta": {"backend": "kcorrect"},
        "z": np.array([0.0, 1.0]),
        "responses_in": ["r"],
        "responses_out": ["r"],
        "responses_map": ["r"],
        "anchors": ["E"],
        "K": {"E": np.array([[0.0], [1.0]])},
    }

    path = tmp_path / "grid.npz"
    save_kcorr_package(pkg, path)

    with np.load(path, allow_pickle=False) as data:
        for name in data.files:
            assert not data[name].dtype.hasobject


def test_packaged_kcorr_grids_are_readable() -> None:
    """Tests that all shipped correction grids load without pickle."""
    from importlib import resources

    root = resources.files("lfkit").joinpath("data", "kcorrect", "grids")

    paths = sorted(
        (path for path in root.iterdir() if path.name.endswith(".npz")),
        key=lambda path: path.name,
    )

    assert paths

    for resource in paths:
        with resources.as_file(resource) as path:
            pkg = load_kcorr_package(path)

        assert np.all(np.isfinite(pkg["z"]))
        assert np.all(np.diff(pkg["z"]) > 0.0)

        for label in pkg["anchors"]:
            assert pkg["K"][label].shape == (
                pkg["z"].size,
                len(pkg["responses_out"]),
            )
            assert np.all(np.isfinite(pkg["K"][label]))

        interpolators = kcorr_interpolators(pkg, method="linear")

        assert set(interpolators) == set(pkg["anchors"])
