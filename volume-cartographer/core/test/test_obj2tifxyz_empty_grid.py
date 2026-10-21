"""CLI regression for #1320: vc_obj2tifxyz must not report success after rasterizing zero grid points.

Run with: python test_obj2tifxyz_empty_grid.py /path/to/vc_obj2tifxyz

The input is a flat two-triangle OBJ whose UVs are normalised to [0, 1], the UV convention of the published
segment OBJs. Two cases:

  default arguments   with the default stretch factor the UV grid is 2 x 2 and no grid point lands inside a
                      triangle; the run must exit non-zero and leave no x.tif / y.tif / z.tif behind, so a
                      script cannot mistake an all-sentinel tifxyz for a result
  a stretch factor    a guard: the same OBJ with a stretch factor must still convert, exit 0 and write x.tif,
                      or the case above would pass for a tool that fails on everything

Standard library only; a few seconds.
"""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

if len(sys.argv) < 2:
    raise SystemExit("usage: test_obj2tifxyz_empty_grid.py /path/to/vc_obj2tifxyz")
TOOL = str(Path(sys.argv.pop(1)).resolve())

OBJ = """\
v 100 100 50
v 300 100 50
v 300 300 50
v 100 300 50
vt 0.05 0.05
vt 0.95 0.05
vt 0.95 0.95
vt 0.05 0.95
f 1/1 2/2 3/3
f 1/1 3/3 4/4
"""


def run(tmp, *extra):
    obj = Path(tmp) / "flat.obj"
    obj.write_text(OBJ)
    out = Path(tmp) / "out"
    proc = subprocess.run([TOOL, str(obj), str(out), *extra], capture_output=True, text=True, timeout=60)
    return proc, out


class EmptyGrid(unittest.TestCase):
    def test_default_arguments_fail_without_writing_a_surface(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc, out = run(tmp)
            self.assertNotEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertNotIn("Successfully converted", proc.stdout)
            self.assertIn("no valid grid points", proc.stderr)
            for name in ("x.tif", "y.tif", "z.tif"):
                self.assertFalse((out / name).exists(), name + " was written for an empty grid")

    def test_stretch_factor_still_converts(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc, out = run(tmp, "200")
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertTrue((out / "x.tif").exists(), proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
