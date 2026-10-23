"""
Test Suite: Grover Algorithm Operations
Tests 3, 4, 5, 8, 9, 10:
3. Hadamard gate behavior on a single qubit.
4. Oracle phase inversion for the selected basis state.
5. Diffusion operator normalization and properties.
8. Target bitstring ordering and LSB/MSB consistency.
9. Grover search for multiple targets and qubit counts in the supported range.
10. Probability improvement for the default four-qubit example (~96.13%).
"""

import unittest
import numpy as np

import tests  # Bootstraps path
from src.quantum_core import LocalQuantumCircuit
from src.grover import (
    build_grover_oracle,
    build_grover_diffusion,
    create_grover_circuit,
    run_grover_simulation,
    calculate_optimal_iterations,
)


class TestGrover(unittest.TestCase):
    """Test quantum gates, oracle inversion, diffusion, and Grover algorithm execution."""

    def test_hadamard_single_qubit(self):
        """Test Hadamard gate transformation on |0> and |1>, and involution H^2 = I."""
        # 1. H on |0>
        qc = LocalQuantumCircuit(2)
        qc.h(0)
        sv = qc.to_statevector()
        # Expect (|00> + |01>) / sqrt(2) since q0 is LSB
        np.testing.assert_allclose(np.abs(sv[0]) ** 2, 0.5, atol=1e-6)
        np.testing.assert_allclose(np.abs(sv[1]) ** 2, 0.5, atol=1e-6)

        # 2. H^2 = I
        qc2 = LocalQuantumCircuit(2)
        qc2.h(0)
        qc2.h(0)
        sv2 = qc2.to_statevector()
        np.testing.assert_allclose(sv2[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(sv2[1:], 0.0, atol=1e-6)

        # 3. H on |1>
        qc3 = LocalQuantumCircuit(2)
        qc3.x(0)
        qc3.h(0)
        sv3 = qc3.to_statevector()
        # (|00> - |01>) / sqrt(2)
        np.testing.assert_allclose(sv3[0], 1.0 / np.sqrt(2), atol=1e-6)
        np.testing.assert_allclose(sv3[1], -1.0 / np.sqrt(2), atol=1e-6)

    def test_oracle_phase_inversion(self):
        """Test that the oracle inverts the phase of only the marked basis state."""
        n = 3
        target = "101"
        target_idx = int(target, 2)  # 5

        # Initialize uniform superposition
        qc = LocalQuantumCircuit(n)
        for i in range(n):
            qc.h(i)
        sv_before = qc.to_statevector()

        # Apply oracle
        build_grover_oracle(qc, target)
        sv_after = qc.to_statevector()

        # Probabilities before and after oracle should be identical
        np.testing.assert_allclose(np.abs(sv_before) ** 2, np.abs(sv_after) ** 2, atol=1e-6)

        # Target state amplitude should be negated
        np.testing.assert_allclose(sv_after[target_idx], -sv_before[target_idx], atol=1e-6)

        # All other states should retain their original amplitude
        for idx in range(2 ** n):
            if idx != target_idx:
                np.testing.assert_allclose(sv_after[idx], sv_before[idx], atol=1e-6)

    def test_diffusion_operator_properties(self):
        """Test diffusion operator reflection about the mean and unitarity."""
        n = 3
        N = 2 ** n

        # Test on uniform superposition |s_0>
        qc = LocalQuantumCircuit(n)
        for i in range(n):
            qc.h(i)
        build_grover_diffusion(qc)
        sv = qc.to_statevector()

        # D |s_0> = |s_0>
        expected_amp = 1.0 / np.sqrt(N)
        for amp in sv:
            np.testing.assert_allclose(np.abs(amp), expected_amp, atol=1e-5)

        # Unitarity check: norm of statevector is preserved
        self.assertAlmostEqual(np.linalg.norm(sv), 1.0, places=6)

    def test_target_bitstring_ordering(self):
        """Test that bitstring indexing correctly marks the intended target state."""
        test_cases = [
            (2, "01"),
            (2, "10"),
            (3, "001"),
            (3, "110"),
            (4, "1010"),
            (4, "0101"),
        ]
        for n, target in test_cases:
            opt_iter = calculate_optimal_iterations(n)
            res = run_grover_simulation(n, target, opt_iter, shots=1000, seed=42)
            # The maximum probability state must be the target
            best_state = max(res["ideal_probabilities"].items(), key=lambda kv: kv[1])[0]
            self.assertEqual(
                best_state,
                target,
                f"For {n} qubits and target '{target}', expected max state '{target}', got '{best_state}'"
            )

    def test_grover_search_supported_range(self):
        """Test Grover search across 2, 3, 4, 5, 6 qubits."""
        configs = [
            (2, "11", 1, 1.0),            # N=4, 1 iteration gives 100%
            (3, "101", 2, 0.945),         # N=8, 2 iterations gives ~94.5%
            (4, "1010", 3, 0.9613),       # N=16, 3 iterations gives ~96.13%
            (5, "11001", 4, 0.99),        # N=32, 4 iterations gives ~99.9%
        ]
        for n, target, iters, min_prob in configs:
            res = run_grover_simulation(n, target, iters, shots=1000, seed=42)
            prob = res["target_ideal_prob"]
            self.assertGreaterEqual(
                prob,
                min_prob - 0.05,
                f"Qubits {n}, target {target}, iters {iters}: prob {prob} < {min_prob - 0.05}"
            )

    def test_default_four_qubit_experiment_amplification(self):
        """
        Test the default hackathon benchmark:
        4 qubits, search space 16, target '1010'.
        Verify monotonic probability improvement up to iteration 3,
        and confirm target probability after 3 iterations is ~96.13%.
        """
        target = "1010"
        n = 4

        # 0 iterations (uniform superposition)
        res0 = run_grover_simulation(n, target, 0)
        p0 = res0["target_ideal_prob"]
        self.assertAlmostEqual(p0, 1.0 / 16.0, places=5, msg="0 iters must be 6.25%")

        # 1 iteration
        res1 = run_grover_simulation(n, target, 1)
        p1 = res1["target_ideal_prob"]
        self.assertAlmostEqual(p1, 0.472656, places=4, msg="1 iter must be ~47.27%")

        # 2 iterations
        res2 = run_grover_simulation(n, target, 2)
        p2 = res2["target_ideal_prob"]
        self.assertAlmostEqual(p2, 0.908447, places=4, msg="2 iters must be ~90.84%")

        # 3 iterations (optimal)
        res3 = run_grover_simulation(n, target, 3)
        p3 = res3["target_ideal_prob"]
        self.assertAlmostEqual(p3, 0.961319, places=4, msg="3 iters must be ~96.13%")

        # Monotonic amplification verification
        self.assertLess(p0, p1)
        self.assertLess(p1, p2)
        self.assertLess(p2, p3)

        # 4 iterations (over-rotation check)
        res4 = run_grover_simulation(n, target, 4)
        p4 = res4["target_ideal_prob"]
        self.assertLess(p4, p3, msg="Iteration 4 must demonstrate over-rotation decrease")


if __name__ == "__main__":
    unittest.main()
