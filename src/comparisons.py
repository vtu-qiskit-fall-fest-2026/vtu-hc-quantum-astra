"""
Quantum Astra — Classical vs Quantum Search Comparison Module
Implements formal complexity models for Classical Linear Search vs Grover Search:
- Best-case, Average-case, and Worst-case query counts
- Theoretical query complexity (O(N) vs O(sqrt(N)))
- Speedup ratios and scaling analysis
- Clear pedagogical distinctions between query complexity, ideal math,
  sampling variance, and simulation wall-clock runtime.
"""

import math
from typing import Dict, List, Optional
import pandas as pd

try:
    from src.quantum_core import validate_qubit_count
    from src.grover import calculate_optimal_iterations
except ImportError:
    from quantum_core import validate_qubit_count
    from grover import calculate_optimal_iterations


def get_classical_metrics(search_space_size: int) -> Dict[str, float]:
    """
    Classical unstructured search query counts for N items with 1 marked item:
    - Best-case queries: 1 (marked item found on first check)
    - Average-case queries: (N + 1) / 2 (assuming uniform random distribution)
    - Worst-case queries: N (marked item is at the end or not found until last)
    - Query complexity: O(N)
    """
    N = max(1, int(search_space_size))
    return {
        "best_case": 1,
        "average_case": (N + 1) / 2.0,
        "worst_case": N,
        "complexity": "O(N)",
    }


def get_quantum_metrics(n_qubits: int, num_targets: int = 1) -> Dict[str, float]:
    """
    Grover quantum search metrics for N = 2^n with M marked items:
    - Optimal iterations: R approx floor(pi / 4 * sqrt(N / M))
    - Total oracle queries: R
    - Query complexity: O(sqrt(N / M))
    - Quadratic speedup factor: Average Classical / Quantum Queries
    """
    n = validate_qubit_count(n_qubits)
    N = 2 ** n
    M = max(1, int(num_targets))
    R = calculate_optimal_iterations(n, M)

    avg_classical = (N + 1) / (2.0 * M)
    speedup_ratio = avg_classical / float(R) if R > 0 else 1.0

    return {
        "n_qubits": n,
        "search_space_N": N,
        "num_targets": M,
        "optimal_iterations": R,
        "oracle_queries": R,
        "complexity": "O(sqrt(N))",
        "speedup_ratio": round(speedup_ratio, 2),
    }


def generate_scaling_comparison_table(max_qubits: int = 16) -> pd.DataFrame:
    """
    Generate an educational scaling table comparing Classical Search vs Grover Search
    from n=2 up to n=16 qubits.
    """
    rows = []
    for n in range(2, max_qubits + 1):
        N = 2 ** n
        classical_avg = (N + 1) / 2.0
        # Optimal Grover iterations for 1 target
        theta = math.asin(1.0 / math.sqrt(N))
        r_grover = max(1, int(round((math.pi / (4.0 * theta)) - 0.5)))
        speedup = classical_avg / float(r_grover)

        rows.append({
            "Qubits (n)": n,
            "Search Space (N = 2^n)": f"{N:,}",
            "Classical Avg Queries": f"{classical_avg:,.1f}",
            "Classical Worst Case": f"{N:,}",
            "Grover Quantum Queries": f"{r_grover:,}",
            "Speedup Factor": f"{speedup:,.1f}x",
        })

    return pd.DataFrame(rows)


def get_complexity_explanations() -> Dict[str, str]:
    """
    Pedagogical explanations clarifying crucial distinctions in quantum algorithms.
    """
    return {
        "theoretical_complexity": (
            "Grover's algorithm provides an optimal quadratic speedup in query complexity "
            "(O(sqrt(N)) oracle queries compared to classical O(N) queries). This is a proven "
            "lower bound for unstructured database search (Bennett, Bernstein, Brassard, Vazirani, 1997)."
        ),
        "ideal_probability": (
            "The exact mathematical probability of observing the marked state |s> after k iterations "
            "is given by P(s) = sin^2((2k + 1) * theta), where sin(theta) = 1/sqrt(N). Grover iterations "
            "are a unitary rotation in a 2D subspace: iterating beyond the optimal count causes over-rotation "
            "and decreases success probability."
        ),
        "finite_shot_sampling": (
            "Quantum measurements are inherently probabilistic. In physical experiments and shot-based "
            "simulations, finite sampling yields statistical variance proportional to 1/sqrt(shots). "
            "Sampled frequencies converge to ideal statevector probabilities as the shot count increases."
        ),
        "classical_simulation_caveat": (
            "Important: Running a quantum simulator on a classical CPU is NOT faster than classical search! "
            "Simulating an n-qubit quantum state classically requires storing and multiplying 2^n complex amplitudes, "
            "costing exponential O(2^n) classical memory and time. The quadratic advantage of Grover's search only "
            "materializes when executed natively on real physical quantum hardware."
        ),
    }
