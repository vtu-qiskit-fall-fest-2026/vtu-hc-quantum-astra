#!/usr/bin/env python3
"""
Quantum Astra — Automated Test Runner
Discovers and executes all unit test suites, verifying the 12 core requirements:
1. Valid and invalid qubit counts.
2. Target bitstring length and character validation.
3. Hadamard gate behavior on a single qubit.
4. Oracle phase inversion for the selected basis state.
5. Diffusion operator normalization.
6. Statevector normalization.
7. Probability calculations.
8. Target bitstring ordering.
9. Grover search for multiple targets and qubit counts in the supported range.
10. Probability improvement for the default four-qubit example (~96.13%).
11. Measurement output validity.
12. Classical query-count formulas.
"""

import sys
import os
import unittest
import importlib.util

# Set up paths
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

src_dir = os.path.join(base_dir, "src")
if "src" not in sys.modules:
    spec = importlib.util.spec_from_file_location("src", os.path.join(src_dir, "__init__.py"))
    src_mod = importlib.util.module_from_spec(spec)
    sys.modules["src"] = src_mod
    spec.loader.exec_module(src_mod)

from tests.test_validation import TestValidation
from tests.test_grover import TestGrover
from tests.test_probabilities import TestProbabilities


def run_all_tests():
    print("=" * 70)
    print(" QUANTUM ASTRA — AUTOMATED TEST SUITE (VTU-QFF-HC-0008)")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestGrover))
    suite.addTests(loader.loadTestsFromTestCase(TestProbabilities))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print(" TEST VERIFICATION SUMMARY")
    print("=" * 70)
    checks = [
        ("1. Valid and invalid qubit counts", True),
        ("2. Target bitstring length and character validation", True),
        ("3. Hadamard gate behavior on a single qubit", True),
        ("4. Oracle phase inversion for selected basis state", True),
        ("5. Diffusion operator normalization and properties", True),
        ("6. Statevector normalization (sum |amp|^2 = 1.0)", True),
        ("7. Probability calculations and non-negativity", True),
        ("8. Target bitstring ordering and LSB/MSB consistency", True),
        ("9. Grover search across 2, 3, 4, 5 qubits", True),
        ("10. Probability improvement for default 4-qubit example (~96.13%)", True),
        ("11. Measurement output validity and shot sampling", True),
        ("12. Classical query-count formulas and quantum comparison", True),
    ]

    all_passed = result.wasSuccessful()
    for name, status in checks:
        mark = "[PASS]" if all_passed else "[CHECK]"
        print(f" {mark} {name}")

    print("=" * 70)
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Status: {'ALL TESTS PASSED SUCCESSFULLY' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
