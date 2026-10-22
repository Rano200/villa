from __future__ import annotations

import gc
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import tifffile

from vesuvius.ink_detection.preprocessing import merge_predictions


@pytest.mark.parametrize("method", merge_predictions.MERGE_METHOD_CHOICES)
def test_merge_predictions_leaves_no_temp_memmaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_dir))

    preds = tmp_path / "preds"
    preds.mkdir()
    rng = np.random.default_rng(0)
    # A compressed TIFF and a PNG both go through temp memmaps.
    tifffile.imwrite(
        preds / "a_forward_x.tif",
        rng.integers(0, 256, (64, 48), dtype=np.uint8),
        compression="lzw",
    )
    Image.fromarray(rng.integers(0, 256, (64, 48), dtype=np.uint8)).save(preds / "b_forward_x.png")

    assert merge_predictions.main([str(preds), "--workers", "1", "--direction", "forward", "--method", method]) == 0

    assert (preds / f"merged_{method}_forward.tif").is_file()
    gc.collect()
    assert sorted(p.name for p in temp_dir.iterdir()) == []
