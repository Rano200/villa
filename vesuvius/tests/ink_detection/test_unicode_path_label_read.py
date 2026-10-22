from __future__ import annotations

import cv2
import numpy as np
import pytest
import zarr

from vesuvius.ink_detection.preprocessing.create_label_zarrs import (
    load_image,
    main,
)
from vesuvius.utils.io.image_io import imread_unicode

NON_ASCII_DIRS = ["Scan ü", "José", "字字"]


def _png_in(tmp_path, folder):
    """Write a small PNG label into a folder with non-ANSI characters."""
    directory = tmp_path / folder
    directory.mkdir()
    image = np.zeros((32, 48), dtype=np.uint8)
    image[8:20, 10:30] = 255
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path = directory / "20250101000000_inklabels.png"
    encoded.tofile(str(path))
    return path, image


@pytest.mark.parametrize("folder", NON_ASCII_DIRS)
def test_imread_unicode_reads_non_ascii_folder(tmp_path, folder):
    path, image = _png_in(tmp_path, folder)
    np.testing.assert_array_equal(imread_unicode(path, cv2.IMREAD_UNCHANGED), image)


def test_imread_unicode_returns_none_for_missing_or_undecodable(tmp_path):
    assert imread_unicode(tmp_path / "missing.png") is None
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not a png")
    assert imread_unicode(bad) is None
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    assert imread_unicode(empty) is None


def test_imread_unicode_honours_flags(tmp_path):
    directory = tmp_path / "José"
    directory.mkdir()
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 1] = 200
    ok, encoded = cv2.imencode(".png", rgb)
    assert ok
    path = directory / "rgb.png"
    encoded.tofile(str(path))
    assert imread_unicode(path, cv2.IMREAD_UNCHANGED).shape == (8, 8, 3)
    assert imread_unicode(path, cv2.IMREAD_GRAYSCALE).shape == (8, 8)


@pytest.mark.parametrize("folder", NON_ASCII_DIRS)
def test_create_label_zarrs_reads_png_in_non_ascii_folder(tmp_path, folder):
    path, image = _png_in(tmp_path, folder)
    np.testing.assert_array_equal(load_image(path), image)
    assert main([str(path.parent), "--levels", "2"]) == 0
    level0 = zarr.open(str(path.with_suffix(".zarr")), mode="r")["0"]
    assert (level0[:] != 0).sum() > 0
