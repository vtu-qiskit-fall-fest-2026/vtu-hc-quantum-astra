"""
Quantum Astra — Probability Analysis and Normalization Module
Computes basis probabilities from statevectors, verifies normalization,
constructs structured tabular data, and exports CSV reports.
"""

import io
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    from src.quantum_core import validate_qubit_count
except ImportError:
    from quantum_core import validate_qubit_count


def compute_probabilities_from_statevector(statevector: np.ndarray, n_qubits: int) -> Dict[str, float]:
    """
    Calculate computational basis state probabilities from statevector amplitudes:
        P(|x>) = |<x|psi>|^2
    Basis states are mapped to n-bit binary strings where qubit 0 is LSB.
    """
    n = validate_qubit_count(n_qubits)
    N = 2 ** n
    if len(statevector) != N:
        raise ValueError(
            f"Statevector dimension ({len(statevector)}) does not match 2^{n} = {N}"
        )

    # Squared magnitudes of amplitudes
    probs = np.abs(statevector) ** 2
    probs = np.maximum(probs, 0.0)

    prob_dict: Dict[str, float] = {}
    for idx in range(N):
        bstr = format(idx, f"0{n}b")
        prob_dict[bstr] = float(probs[idx])

    return prob_dict


def verify_normalization(probabilities: Dict[str, float], tolerance: float = 1e-4) -> Tuple[bool, float]:
    """
    Verify that all basis probabilities are non-negative and sum to 1.0 within tolerance:
        sum_x P(x) = 1.0 +/- tolerance
    """
    for bitstring, p in probabilities.items():
        if p < -1e-9:
            return False, float(p)

    total_prob = float(sum(probabilities.values()))
    is_normalized = abs(total_prob - 1.0) <= tolerance
    return is_normalized, total_prob


def build_probability_dataframe(
    probabilities: Dict[str, float],
    counts: Dict[str, int],
    shots: int,
    target_state: str,
) -> pd.DataFrame:
    """
    Construct a clean pandas DataFrame containing basis state metrics:
    - Basis State
    - Ideal Probability (%)
    - Ideal Probability (decimal)
    - Simulated Measurement Count
    - Sampled Frequency (%)
    - Is Target
    """
    rows = []
    all_states = sorted(set(probabilities.keys()).union(set(counts.keys())))

    for bstr in all_states:
        p_ideal = probabilities.get(bstr, 0.0)
        c_count = counts.get(bstr, 0)
        p_sampled = (float(c_count) / float(shots)) if shots > 0 else 0.0
        is_target = (bstr == target_state)

        rows.append({
            "Basis State": bstr,
            "Ideal Probability (%)": round(p_ideal * 100.0, 4),
            "Ideal Probability": round(p_ideal, 6),
            "Simulated Shots": c_count,
            "Sampled Frequency (%)": round(p_sampled * 100.0, 4),
            "Sampled Frequency": round(p_sampled, 6),
            "Target State": "Yes (Target)" if is_target else "No",
        })

    df = pd.DataFrame(rows)
    return df


def export_probabilities_to_csv(df: pd.DataFrame) -> str:
    """
    Export DataFrame to a clean UTF-8 CSV string for download.
    """
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()
