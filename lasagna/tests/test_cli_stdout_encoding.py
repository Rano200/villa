"""CLIs that print non-ASCII glyphs must survive a redirected stdout on Windows.

On Windows a redirected or piped stdout uses the ANSI codepage (cp1252), so
printing a glyph such as U+2588 or U+2192 raised UnicodeEncodeError and the
script exited 1 with its output lost. The tests below force cp1252 on every OS
(PYTHONIOENCODING=cp1252, PYTHONUTF8 unset), so they are meaningful on Linux
and macOS as well as Windows: on an affected tree they fail on every platform.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

LASAGNA = Path(__file__).resolve().parents[1]


def _run(script: Path, *args: str, encoding: str = "cp1252"):
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONUTF8", "PYTHONIOENCODING")}
    env["PYTHONIOENCODING"] = encoding
    return subprocess.run(
        [sys.executable, str(script), *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, cwd=str(LASAGNA), timeout=300,
    )


@pytest.mark.skipif(sys.platform != "win32",
                    reason="the fix only changes stdout on Windows, where cp1252 is the redirect default")
def test_analyze_zarr_channels_piped_cp1252(tmp_path):
    zarr = pytest.importorskip("zarr")
    path = tmp_path / "chan.zarr"
    a = zarr.open(str(path), mode="w", shape=(2, 8, 16, 16), chunks=(2, 8, 16, 16), dtype="uint8")
    a[0] = np.arange(8 * 16 * 16, dtype=np.uint32).reshape(8, 16, 16) % 256
    a[1] = 7
    r = _run(LASAGNA / "analyze_zarr_channels.py", str(path))
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-600:]
    out = r.stdout.decode("utf-8")
    assert "Total voxels per channel" in out
    assert any(ch in out for ch in "▁▂▃▄▅▆▇█")


@pytest.mark.skipif(sys.platform != "win32",
                    reason="the fix only changes stdout on Windows, where cp1252 is the redirect default")
def test_labels_to_winding_volume_piped_cp1252(tmp_path):
    pytest.importorskip("zarr")
    tifffile = pytest.importorskip("tifffile")
    vol = np.zeros((16, 32, 32), dtype=np.uint8)
    vol[:, 8:10, :] = 1
    vol[:, 20:22, :] = 1
    tif = tmp_path / "labels.tif"
    tifffile.imwrite(str(tif), vol)
    out = tmp_path / "winding.zarr"
    r = _run(LASAGNA / "labels_to_winding_volume.py", "--input", str(tif),
             "--output", str(out), "--step", "4", "--connectivity", "6")
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-600:]
    assert "border bg → ignore" in r.stdout.decode("utf-8")
    assert out.exists()


@pytest.mark.skipif(sys.platform != "win32",
                    reason="the fix only changes stdout on Windows, where cp1252 is the redirect default")
def test_estimate_stream_memory_piped_cp1252():
    r = _run(LASAGNA / "estimate_stream_memory.py")
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-600:]
    assert "Summary" in r.stdout.decode("utf-8")
