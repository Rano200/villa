"""run/ced.py calls the diffusion function, not the module of the same name.

`from vesuvius.image_proc.geometry.diffusion import coherence_enhancing_diffusion`
binds the submodule (the package __init__ is empty), so every call raised
`TypeError: 'module' object is not callable` and nothing could be processed.
"""
import sys

import numpy as np
import pytest
import tifffile

pytest.importorskip("torch")

from vesuvius.image_proc.run import ced


def test_ced_name_is_callable():
    assert callable(ced.run_coherence_diffusion)


def test_ced_main_writes_output_tiff(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    src = tmp_path / "in.tif"
    dst = tmp_path / "out.tif"
    tifffile.imwrite(src, (rng.random((48, 48)) * 255).astype(np.uint8))
    monkeypatch.setattr(sys, "argv", ["ced", str(src), str(dst), "--num-steps", "2"])
    ced.main()
    out = tifffile.imread(dst)
    assert out.shape == (48, 48)
