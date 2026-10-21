from __future__ import annotations

import sys

import numpy as np
import pytest
import zarr

from vesuvius.models.preprocessing.patches import cli


CONFIG = """\
tr_setup:
  model_name: demo
tr_config:
  patch_size: [8, 8, 8]
  batch_size: 1
dataset_config:
  min_labeled_ratio: 0.05
  min_bbox_percent: 0.5
  targets:
    surface:
      out_channels: 2
      activation: none
"""


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["vesuvius.find_patches", *argv])
    cli.main()


def _make_dataset(root):
    for sub, name in (("images", "vol.zarr"), ("labels", "vol_surface.zarr")):
        group = zarr.open_group(str(root / sub / name), mode="w", zarr_format=2)
        arr = group.create_array("0", shape=(16, 16, 16), chunks=(8, 8, 8), dtype="uint8")
        arr[:] = np.ones((16, 16, 16), dtype="uint8")


def test_exits_nonzero_when_no_volumes_found(tmp_path, monkeypatch, capsys):
    config = tmp_path / "experiments" / "v0.yaml"
    config.parent.mkdir()
    config.write_text(CONFIG)

    with pytest.raises(SystemExit) as excinfo:
        _run(monkeypatch, "--config", str(config))

    assert excinfo.value.code == 1
    assert "no volumes found" in capsys.readouterr().err


def test_input_points_at_the_dataset(tmp_path, monkeypatch, capsys):
    config = tmp_path / "experiments" / "v0.yaml"
    config.parent.mkdir()
    config.write_text(CONFIG)
    data = tmp_path / "data"
    _make_dataset(data)

    _run(monkeypatch, "--config", str(config), "-i", str(data))

    assert "1 cache(s) written, 1 volumes scanned" in capsys.readouterr().out
    assert (data / ".patches_cache").is_dir()


def test_missing_input_directory_is_reported(tmp_path, monkeypatch, capsys):
    config = tmp_path / "v0.yaml"
    config.write_text(CONFIG)

    with pytest.raises(SystemExit):
        _run(monkeypatch, "--config", str(config), "-i", str(tmp_path / "nope"))

    assert "Input directory does not exist" in capsys.readouterr().err
