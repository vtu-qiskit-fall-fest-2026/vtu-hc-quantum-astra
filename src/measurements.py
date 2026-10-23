"""
Quantum Astra — Quantum Measurement and Sampling Module
Handles simulated measurement execution, shot sampling, statistical frequency
computation, and binomial error estimation.
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from src.quantum_core import validate_shots
except ImportError:
    from quantum_core import validate_shots


def compute_sampled_frequencies(counts: Dict[str, int], shots: int) -> Dict[str, float]:
    """
    Compute experimental frequencies from raw measurement counts:
        frequency = count / total_shots
    """
    shots = validate_shots(shots)
    frequencies = {}
    for bitstring, count in counts.items():
        frequencies[bitstring] = float(count) / float(shots)
    return frequencies


def compute_sampling_error(probability: float, shots: int) -> float:
    """
    Calculate the theoretical standard error (sigma) of a binomial sample:
        sigma = sqrt(p * (1 - p) / shots)
    For p = 0 or 1, sigma = 0.
    """
    shots = validate_shots(shots)
    p = max(0.0, min(1.0, float(probability)))
    return math.sqrt(p * (1.0 - p) / float(shots))


def compare_ideal_vs_sampled(
    ideal_probs: Dict[str, float],
    sampled_counts: Dict[str, int],
    shots: int,
) -> Dict[str, Dict[str, float]]:
    """
    Compare ideal quantum probabilities against finite-shot sampling outcomes.
    Calculates absolute difference and standard error bound for each basis state.
    """
    shots = validate_shots(shots)
    comparison = {}

    all_keys = set(ideal_probs.keys()).union(set(sampled_counts.keys()))
    for key in sorted(all_keys):
        p_ideal = ideal_probs.get(key, 0.0)
        c_count = sampled_counts.get(key, 0)
        p_sampled = float(c_count) / float(shots)
        diff = abs(p_sampled - p_ideal)
        sigma = compute_sampling_error(p_ideal, shots)

        comparison[key] = {
            "ideal_prob": p_ideal,
            "sampled_count": c_count,
            "sampled_freq": p_sampled,
            "absolute_error": diff,
            "standard_error": sigma,
        }

    return comparison
