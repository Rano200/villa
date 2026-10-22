"""Small image reading helpers shared by the label readers."""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np


def imread_unicode(
    path: str | os.PathLike, flags: int = cv2.IMREAD_UNCHANGED
) -> Optional[np.ndarray]:
    """``cv2.imread`` that also opens paths with non-ANSI characters on Windows.

    ``cv2.imread`` uses the ANSI file API on Windows, so a folder whose name
    has characters outside the active code page (an accented user name, for
    example) cannot be opened. Reading the bytes with numpy and decoding them
    with ``cv2.imdecode`` gives the same array. Like ``cv2.imread``, this
    returns ``None`` when the file cannot be read or decoded.
    """
    try:
        data = np.fromfile(os.fspath(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)
