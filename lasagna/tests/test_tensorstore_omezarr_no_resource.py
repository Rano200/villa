import importlib
import sys
import unittest
from unittest import mock


class TensorStoreOmeZarrWithoutResourceModuleTests(unittest.TestCase):
	def test_module_imports_and_skips_rss_when_resource_is_missing(self):
		# ``resource`` does not exist on Windows. ``sys.modules[name] = None``
		# makes ``import resource`` raise ImportError, as it does there.
		sys.modules.pop("tensorstore_omezarr", None)
		try:
			with mock.patch.dict(sys.modules, {"resource": None}):
				module = importlib.import_module("tensorstore_omezarr")
				self.assertIsNone(module.resource)
				self.assertIsNone(module._rss_mib())
		finally:
			sys.modules.pop("tensorstore_omezarr", None)


if __name__ == "__main__":
	unittest.main()
