"""
Test Suite: Input Validation
Tests 1 & 2:
1. Valid and invalid qubit counts.
2. Target bitstring length and character validation.
"""

import unittest
import numpy as np

import tests  # Bootstraps path
from src.quantum_core import (
    validate_qubit_count,
    validate_target_bitstring,
    validate_iterations,
    validate_shots,
)


class TestValidation(unittest.TestCase):
    """Test parameter validation functions."""

    def test_valid_qubit_counts(self):
        """Test valid qubit counts within [2, 8]."""
        for n in range(2, 9):
            self.assertEqual(validate_qubit_count(n), n)
            self.assertEqual(validate_qubit_count(np.int64(n)), n)

    def test_invalid_qubit_counts(self):
        """Test invalid qubit counts (below 2, above 8, non-integers)."""
        invalid_counts = [0, 1, 9, 10, -3, 100]
        for n in invalid_counts:
            with self.assertRaises(ValueError, msg=f"Qubit count {n} should raise ValueError"):
                validate_qubit_count(n)

        type_errors = [3.5, "4", None, [4]]
        for val in type_errors:
            with self.assertRaises(TypeError, msg=f"Type {type(val)} should raise TypeError"):
                validate_qubit_count(val)

    def test_valid_target_bitstrings(self):
        """Test valid binary bitstrings matching register length."""
        test_cases = [
            ("00", 2),
            ("11", 2),
            ("101", 3),
            ("1010", 4),
            ("00000", 5),
            ("11111111", 8),
        ]
        for bstr, n in test_cases:
            self.assertEqual(validate_target_bitstring(bstr, n), bstr)

    def test_invalid_target_bitstring_length(self):
        """Test bitstring length mismatches."""
        mismatches = [
            ("10", 3),
            ("1010", 3),
            ("101", 4),
            ("", 4),
            ("101010101", 8),
        ]
        for bstr, n in mismatches:
            with self.assertRaises(ValueError, msg=f"Length mismatch for '{bstr}' with {n} qubits"):
                validate_target_bitstring(bstr, n)

    def test_invalid_target_bitstring_characters(self):
        """Test non-binary characters in bitstring."""
        invalid_strings = [
            ("102", 3),
            ("abc", 3),
            ("1 0", 3),
            ("1-0", 3),
            ("10F0", 4),
        ]
        for bstr, n in invalid_strings:
            with self.assertRaises(ValueError, msg=f"Invalid character in '{bstr}' should raise ValueError"):
                validate_target_bitstring(bstr, n)

    def test_validate_iterations(self):
        """Test iteration count bounds and type validation."""
        self.assertEqual(validate_iterations(0), 0)
        self.assertEqual(validate_iterations(3), 3)
        self.assertEqual(validate_iterations(10), 10)

        with self.assertRaises(ValueError):
            validate_iterations(-1)
        with self.assertRaises(ValueError):
            validate_iterations(105)
        with self.assertRaises(TypeError):
            validate_iterations("3")

    def test_validate_shots(self):
        """Test shot count validation."""
        self.assertEqual(validate_shots(1024), 1024)
        with self.assertRaises(ValueError):
            validate_shots(0)
        with self.assertRaises(ValueError):
            validate_shots(-100)
        with self.assertRaises(ValueError):
            validate_shots(200000)


if __name__ == "__main__":
    unittest.main()
