from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import labels_to_cc_zarr as cc

# A stand-in for ffmpeg that writes much more than any pipe buffer holds to
# stderr before it reads a single byte of stdin, like ffmpeg's banner does.
_FAKE_FFMPEG = """
import sys
sys.stderr.write("x" * 200000 + "\\nfake ffmpeg log done\\n")
sys.stderr.flush()
n = 0
while True:
    chunk = sys.stdin.buffer.read(65536)
    if not chunk:
        break
    n += len(chunk)
sys.exit(int(sys.argv[1]))
"""


class EncodeVideoStderrTest(unittest.TestCase):
	def _run(self, rc: int, timeout: float = 20.0):
		procs: list[subprocess.Popen] = []
		real_popen = subprocess.Popen

		def fake_popen(cmd, *args, **kwargs):
			assert cmd[0] == "ffmpeg"
			p = real_popen([sys.executable, "-c", _FAKE_FFMPEG, str(rc)], *args, **kwargs)
			procs.append(p)
			return p

		frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(2)]
		result: dict[str, object] = {}

		def target() -> None:
			try:
				with tempfile.TemporaryDirectory() as td:
					cc._encode_video(frames, Path(td) / "out.mp4", fps=15)
				result["ok"] = True
			except Exception as exc:  # noqa: BLE001
				result["err"] = exc

		subprocess.Popen = fake_popen  # type: ignore[assignment]
		try:
			t = threading.Thread(target=target, daemon=True)
			t.start()
			t.join(timeout)
			hung = t.is_alive()
			if hung:
				for p in procs:
					p.kill()
				t.join(5)
		finally:
			subprocess.Popen = real_popen  # type: ignore[assignment]
		return hung, result

	def test_noisy_stderr_does_not_deadlock(self) -> None:
		hung, result = self._run(rc=0)
		self.assertFalse(hung, "_encode_video deadlocked on ffmpeg's stderr")
		self.assertTrue(result.get("ok"), result)

	def test_failure_still_reports_stderr(self) -> None:
		hung, result = self._run(rc=3)
		self.assertFalse(hung, "_encode_video deadlocked on ffmpeg's stderr")
		err = result.get("err")
		self.assertIsInstance(err, RuntimeError)
		self.assertIn("rc=3", str(err))
		self.assertIn("fake ffmpeg log done", str(err))


if __name__ == "__main__":
	unittest.main()
