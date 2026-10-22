"""print() text in these CLIs must survive a cp1252 stdout.

On Windows, stdout redirected to a file or pipe uses the ANSI codepage
(cp1252) unless PYTHONUTF8 / PYTHONIOENCODING are set, and print() of a
character outside it raises UnicodeEncodeError. This is a static check on
the string literals inside print() calls, so it is meaningful on every OS.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "vesuvius"

MODULES = [
    "image_proc/run/ced.py",
    "models/run/blending.py",
    "structure_tensor/create_st.py",
    "structure_tensor/create_vf.py",
    "tifxyz_label_transfer/self_render_tifxyz.py",
    "utils/build_betti.py",
]


def _bad_print_literals(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad = []
    for call in ast.walk(tree):
        if not (isinstance(call, ast.Call) and getattr(call.func, "id", None) == "print"):
            continue
        for node in ast.walk(call):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                try:
                    node.value.encode("cp1252")
                except UnicodeEncodeError as exc:
                    bad.append((node.lineno, ascii(node.value[exc.start:exc.end])))
    return bad


@pytest.mark.parametrize("module", MODULES)
def test_print_literals_encode_in_cp1252(module):
    bad = _bad_print_literals(SRC / module)
    assert not bad, f"{module}: print() text not encodable in cp1252 (line, chars): {bad}"
