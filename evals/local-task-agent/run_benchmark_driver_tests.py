"""Run the benchmark-driver test module and fail if it contains no tests."""

import sys
import unittest


suite = unittest.defaultTestLoader.loadTestsFromName("test_benchmark_driver")
if suite.countTestCases() == 0:
    raise SystemExit("benchmark-driver test module contained zero tests")

result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
