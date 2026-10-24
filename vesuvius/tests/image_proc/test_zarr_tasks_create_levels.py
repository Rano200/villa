"""The remap and resize tasks create their output levels on every zarr the package accepts
(pyproject: zarr>=2.18.7,<4).

They called Group.create_dataset, which zarr 3 removed, so both crashed with
`AttributeError: 'Group' object has no attribute 'create_dataset'` before writing anything. They now go
through create_level_dataset like the other pyramid tasks (merge got the same change in #2244,
and open #2267 makes it for edt-dilate).
CI runs this file on zarr 2.18.7 and 3.2.1. Each task is run end to end, workers included.
"""
import json

import numpy as np
import zarr
from numcodecs import Blosc

from vesuvius.image_proc.run.zarr_tasks.tasks.remap import RemapConfig, RemapTask
from vesuvius.image_proc.run.zarr_tasks.tasks.resize import ResizeConfig, ResizeTask
from vesuvius.image_proc.run.zarr_tasks.utils import create_level_dataset, write_multiscales_metadata


def _write_pyramid(path, levels):
    """A zarr v2 OME-Zarr group with one array per level, written the same way on zarr 2 and 3."""
    for name, data in enumerate(levels):
        arr = create_level_dataset(
            str(path), str(name), data.shape, (4, 4, 4), data.dtype, Blosc(cname="zstd", clevel=1)
        )
        arr[...] = data
    write_multiscales_metadata(str(path), ["z", "y", "x"], len(levels))


def _read(path, level):
    return zarr.open(str(path / level), mode="r")[...]


def test_remap_creates_every_level_and_remaps_values(tmp_path):
    level0 = np.zeros((8, 8, 8), np.uint8)
    level0[:4] = 100
    level0[4:, :4] = 200
    level1 = level0[::2, ::2, ::2].copy()
    _write_pyramid(tmp_path / "in.zarr", [level0, level1])
    out = tmp_path / "out.zarr"
    RemapTask(
        RemapConfig(
            input_zarr=str(tmp_path / "in.zarr"),
            output_zarr=str(out),
            num_workers=1,
            remap_dict={100: 2, 200: 3},
        )
    ).run()
    table = np.arange(256, dtype=np.uint8)
    table[100], table[200] = 2, 3
    np.testing.assert_array_equal(_read(out, "0"), table[level0])
    np.testing.assert_array_equal(_read(out, "1"), table[level1])
    assert zarr.open(str(out / "1"), mode="r").dtype == np.uint8


def test_remap_copies_the_input_attributes_to_the_output(tmp_path):
    level0 = np.full((8, 8, 8), 100, np.uint8)
    _write_pyramid(tmp_path / "in.zarr", [level0, level0[::2, ::2, ::2].copy()])
    out = tmp_path / "out.zarr"
    RemapTask(
        RemapConfig(
            input_zarr=str(tmp_path / "in.zarr"),
            output_zarr=str(out),
            num_workers=1,
            remap_dict={100: 2},
        )
    ).run()
    # `".zattrs" in group` is a child lookup on zarr 3, so these used to be dropped
    expected = dict(zarr.open_group(str(tmp_path / "in.zarr"), mode="r").attrs)
    assert [d["path"] for d in expected["multiscales"][0]["datasets"]] == ["0", "1"]
    assert dict(zarr.open_group(str(out), mode="r").attrs) == expected


def test_resize_creates_every_level_at_the_reference_shape(tmp_path):
    level0 = np.arange(8 * 8 * 8, dtype=np.uint16).reshape(8, 8, 8)
    _write_pyramid(tmp_path / "in.zarr", [level0, level0[::2, ::2, ::2].copy()])
    ref0 = np.zeros((16, 16, 16), np.uint8)
    _write_pyramid(tmp_path / "ref.zarr", [ref0, ref0[::2, ::2, ::2].copy()])
    out = tmp_path / "out.zarr"
    ResizeTask(
        ResizeConfig(
            input_zarr=str(tmp_path / "in.zarr"),
            output_zarr=str(out),
            num_workers=1,
            reference_zarr=str(tmp_path / "ref.zarr"),
            interpolation="nearest",
        )
    ).run()
    for level, shape in (("0", (16, 16, 16)), ("1", (8, 8, 8))):
        arr = zarr.open(str(out / level), mode="r")
        assert arr.shape == shape
        assert arr.dtype == np.uint16
    # nearest-neighbour doubling: every output voxel is a voxel of the input, and the volume is not blank
    got = _read(out, "0")
    assert set(np.unique(got)) <= set(np.unique(level0))
    assert np.count_nonzero(got) > 0.9 * got.size
    assert got[15, 15, 15] == level0[7, 7, 7]
