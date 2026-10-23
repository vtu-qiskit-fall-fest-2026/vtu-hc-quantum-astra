"""
Test Suite: Probability and Classical Comparison Metrics
Tests 6, 7, 11, 12:
6. Statevector normalization.
7. Probability calculations and non-negativity.
11. Measurement output validity and shot sampling consistency.
12. Classical query-count formulas and quantum comparison metrics.
"""

import unittest
import numpy as np

import tests  # Bootstraps path
from src.quantum_core import LocalQuantumCircuit
from src.grover import run_grover_simulation
from src.probability import (
    compute_probabilities_from_statevector,
    verify_normalization,
    build_probability_dataframe,
    export_probabilities_to_csv,
)
from src.measurements import compute_sampled_frequencies, compute_sampling_error
from src.comparisons import get_classical_metrics, get_quantum_metrics


class TestProbabilities(unittest.TestCase):
    """Test statevector normalization, probabilities, sampling, and classical formulas."""

    def test_statevector_normalization(self):
        """Test that statevector norm is 1.0 across different circuits."""
        for n in [2, 3, 4, 5]:
            qc = LocalQuantumCircuit(n)
            for i in range(n):
                qc.h(i)
            sv = qc.to_statevector()
            norm = np.linalg.norm(sv)
            self.assertAlmostEqual(norm, 1.0, places=6)

            prob_dict = compute_probabilities_from_statevector(sv, n)
            is_norm, total = verify_normalization(prob_dict)
            self.assertTrue(is_norm)
            self.assertAlmostEqual(total, 1.0, places=5)

    def test_probability_calculations_and_non_negativity(self):
        """Test probability non-negativity and correct squared magnitude computation."""
        res = run_grover_simulation(4, "1010", 3)
        prob_dict = res["ideal_probabilities"]

        self.assertEqual(len(prob_dict), 16)
        for bstr, p in prob_dict.items():
            self.assertGreaterEqual(p, 0.0, f"Probability for {bstr} must be >= 0")
            self.assertLessEqual(p, 1.0, f"Probability for {bstr} must be <= 1")

        is_norm, total = verify_normalization(prob_dict)
        self.assertTrue(is_norm)
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_dataframe_and_csv_generation(self):
        """Test DataFrame structure and CSV serialization."""
        res = run_grover_simulation(4, "1010", 3, shots=1000, seed=42)
        df = build_probability_dataframe(
            res["ideal_probabilities"],
            res["counts"],
            res["shots"],
            res["target_state"],
        )

        expected_columns = [
            "Basis State",
            "Ideal Probability (%)",
            "Ideal Probability",
            "Simulated Shots",
            "Sampled Frequency (%)",
            "Sampled Frequency",
            "Target State",
        ]
        self.assertListEqual(list(df.columns), expected_columns)
        self.assertEqual(len(df), 16)

        csv_str = export_probabilities_to_csv(df)
        self.assertIn("Basis State", csv_str)
        self.assertIn("1010", csv_str)
        self.assertIn("Yes (Target)", csv_str)

    def test_measurement_output_validity(self):
        """Test that measurement shots sum to total and sampled frequency approximates ideal."""
        shots = 5000
        res = run_grover_simulation(4, "1010", 3, shots=shots, seed=123)
        counts = res["counts"]

        # Shot count sum check
        total_shots = sum(counts.values())
        self.assertEqual(total_shots, shots)

        # Target sampled frequency should be within 3 sigma of ideal probability
        p_ideal = res["target_ideal_prob"]
        target_count = res["target_count"]
        p_sampled = target_count / shots

        sigma = compute_sampling_error(p_ideal, shots)
        error = abs(p_sampled - p_ideal)

        self.assertLess(
            error,
            3.5 * sigma,
            f"Sampled frequency ({p_sampled:.4f}) deviates from ideal ({p_ideal:.4f}) by > 3.5 sigma ({sigma:.4f})"
        )

    def test_classical_query_count_formulas(self):
        """Test classical query count formulas: best=1, avg=(N+1)/2, worst=N."""
        test_sizes = [4, 8, 16, 32, 64, 1024]
        for N in test_sizes:
            metrics = get_classical_metrics(N)
            self.assertEqual(metrics["best_case"], 1)
            self.assertEqual(metrics["worst_case"], N)
            self.assertAlmostEqual(metrics["average_case"], (N + 1) / 2.0, places=5)
            self.assertEqual(metrics["complexity"], "O(N)")

        # Test quantum metrics for 4 qubits (N=16)
        q_metrics = get_quantum_metrics(4)
        self.assertEqual(q_metrics["search_space_N"], 16)
        self.assertEqual(q_metrics["optimal_iterations"], 3)
        self.assertEqual(q_metrics["complexity"], "O(sqrt(N))")
        # Classical average = 8.5, Quantum queries = 3 -> Speedup = 8.5 / 3 = ~2.83
        self.assertAlmostEqual(q_metrics["speedup_ratio"], 2.83, places=2)


if __name__ == "__main__":
    unittest.main()
