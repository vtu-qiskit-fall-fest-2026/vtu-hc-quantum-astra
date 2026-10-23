"""
Quantum Astra — Grover's Algorithm Implementation
Provides mathematically rigorous construction of:
1. Initialization (uniform superposition via Hadamard gates)
2. Phase Inversion Oracle for arbitrary target bitstrings
3. Diffusion Operator (reflection about the mean amplitude)
4. Iterative amplitude amplification
5. Ideal statevector simulation and AerSimulator sampling
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from src.quantum_core import (
        LocalQuantumCircuit,
        validate_qubit_count,
        validate_target_bitstring,
        validate_iterations,
        validate_shots,
        HAS_QISKIT,
        HAS_AER,
    )
except ImportError:
    from quantum_core import (
        LocalQuantumCircuit,
        validate_qubit_count,
        validate_target_bitstring,
        validate_iterations,
        validate_shots,
        HAS_QISKIT,
        HAS_AER,
    )

if HAS_QISKIT:
    import qiskit
    from qiskit import QuantumCircuit
    if HAS_AER:
        from qiskit_aer import AerSimulator
    from qiskit.quantum_info import Statevector


def calculate_optimal_iterations(n_qubits: int, num_targets: int = 1) -> int:
    """
    Calculate the theoretically optimal number of Grover iterations.
    For N = 2^n and M marked items:
        theta = arcsin(sqrt(M / N))
        R_exact = round((pi / (4 * theta)) - 0.5)
    Standard asymptotic approximation: floor(pi / 4 * sqrt(N / M)).
    """
    n_qubits = validate_qubit_count(n_qubits)
    N = 2 ** n_qubits
    M = max(1, int(num_targets))
    if M >= N:
        return 0

    ratio = M / N
    theta = math.asin(math.sqrt(ratio))
    optimal_continuous = (math.pi / (4.0 * theta)) - 0.5
    optimal_int = max(1, int(round(optimal_continuous)))
    return optimal_int


def build_grover_oracle(qc: Union[LocalQuantumCircuit, "QuantumCircuit"], target_bitstring: str):
    """
    Construct a reversible phase oracle U_w that flips the phase of |target_bitstring>
    without changing state probabilities during the oracle operation.

    U_w |x> = -|x>  if x == target
    U_w |x> =  |x>  if x != target

    Qubit mapping:
    target[0] is MSB -> qubit (n-1)
    target[n-1] is LSB -> qubit 0
    Qubit i corresponds to character target[n - 1 - i].
    """
    n = qc.num_qubits
    target = validate_target_bitstring(target_bitstring, n)

    # 1. Flip qubits where target bit is '0'
    for i in range(n):
        if target[n - 1 - i] == "0":
            qc.x(i)

    # 2. Multi-controlled phase flip on |11...1>
    # Decomposed as H(0) -> MCX(controls=[1..n-1], target=0) -> H(0)
    controls = list(range(1, n))
    target_qubit = 0
    qc.h(target_qubit)
    if hasattr(qc, "mcx"):
        qc.mcx(controls, target_qubit)
    qc.h(target_qubit)

    # 3. Restore flipped qubits
    for i in range(n):
        if target[n - 1 - i] == "0":
            qc.x(i)


def build_grover_diffusion(qc: Union[LocalQuantumCircuit, "QuantumCircuit"]):
    """
    Construct the Grover Diffusion Operator (Inversion about the mean):
        D = 2|s_0><s_0| - I = H^{tensor n} (2|0...0><0...0| - I) H^{tensor n}

    Implemented via gate sequence:
        H^{tensor n} -> X^{tensor n} -> Multi-Controlled Z -> X^{tensor n} -> H^{tensor n}
    """
    n = qc.num_qubits

    # 1. Hadamard on all qubits
    for i in range(n):
        qc.h(i)

    # 2. Pauli-X on all qubits
    for i in range(n):
        qc.x(i)

    # 3. Multi-controlled Z via H(0) -> MCX(controls=[1..n-1], target=0) -> H(0)
    controls = list(range(1, n))
    target_qubit = 0
    qc.h(target_qubit)
    if hasattr(qc, "mcx"):
        qc.mcx(controls, target_qubit)
    qc.h(target_qubit)

    # 4. Pauli-X on all qubits
    for i in range(n):
        qc.x(i)

    # 5. Hadamard on all qubits
    for i in range(n):
        qc.h(i)


def create_grover_circuit(
    n_qubits: int,
    target_bitstring: str,
    iterations: int,
    add_measurement: bool = True,
    use_qiskit_if_available: bool = True,
) -> Union[LocalQuantumCircuit, "QuantumCircuit"]:
    """
    Assemble complete Grover search circuit:
    - Step 1: Initialize in uniform superposition H^{tensor n} |0>
    - Step 2: Apply Grover operator (Oracle + Diffusion) for requested iterations
    - Step 3: Optional measurement register
    """
    n = validate_qubit_count(n_qubits)
    target = validate_target_bitstring(target_bitstring, n)
    iterations = validate_iterations(iterations)

    if use_qiskit_if_available and HAS_QISKIT:
        qc = QuantumCircuit(n, n, name=f"Grover_{n}q_{iterations}iter")
        # Step 1: Uniform superposition
        for i in range(n):
            qc.h(i)
        # Step 2: Iterations
        for _ in range(iterations):
            build_grover_oracle(qc, target)
            build_grover_diffusion(qc)
        # Step 3: Measurement
        if add_measurement:
            qc.measure(range(n), range(n))
        return qc
    else:
        qc = LocalQuantumCircuit(n, n, name=f"Grover_{n}q_{iterations}iter")
        # Step 1: Uniform superposition
        for i in range(n):
            qc.h(i)
        # Step 2: Iterations
        for _ in range(iterations):
            build_grover_oracle(qc, target)
            build_grover_diffusion(qc)
        # Step 3: Measurement
        if add_measurement:
            qc.measure_all()
        return qc


def run_grover_simulation(
    n_qubits: int,
    target_bitstring: str,
    iterations: int,
    shots: int = 1024,
    seed: Optional[int] = 42,
) -> Dict:
    """
    Execute full simulation of Grover's search algorithm:
    1. Generates the ideal statevector and exact basis probabilities
    2. Runs simulated measurement sampling using AerSimulator (or Local Aer-Equivalent)
    3. Returns consolidated, verified simulation results
    """
    n = validate_qubit_count(n_qubits)
    target = validate_target_bitstring(target_bitstring, n)
    iterations = validate_iterations(iterations)
    shots = validate_shots(shots)
    N = 2 ** n

    # Compute ideal statevector via circuit without measurement
    # Local circuit provides exact, deterministic statevector
    circuit_ideal = create_grover_circuit(
        n_qubits=n,
        target_bitstring=target,
        iterations=iterations,
        add_measurement=False,
        use_qiskit_if_available=False,
    )
    statevector = circuit_ideal.to_statevector()
    probabilities = np.abs(statevector) ** 2
    probabilities = np.maximum(probabilities, 0.0)
    total_prob = float(np.sum(probabilities))
    if total_prob > 0:
        probabilities = probabilities / total_prob

    # Basis mapping: basis index k -> binary string format(k, f'0{n}b')
    prob_dict: Dict[str, float] = {}
    for idx in range(N):
        bstr = format(idx, f"0{n}b")
        prob_dict[bstr] = float(probabilities[idx])

    # Measurement sampling
    counts: Dict[str, int] = {}
    if HAS_QISKIT and HAS_AER:
        try:
            qiskit_circ = create_grover_circuit(
                n_qubits=n,
                target_bitstring=target,
                iterations=iterations,
                add_measurement=True,
                use_qiskit_if_available=True,
            )
            backend = AerSimulator(seed_simulator=seed)
            job = backend.run(qiskit_circ, shots=shots)
            result = job.result()
            counts = result.get_counts(qiskit_circ)
        except Exception:
            # Fall back to high-fidelity local sampler
            counts = circuit_ideal.sample_shots(shots=shots, seed=seed)
    else:
        counts = circuit_ideal.sample_shots(shots=shots, seed=seed)

    # Ensure all basis states have entries in counts
    for idx in range(N):
        bstr = format(idx, f"0{n}b")
        if bstr not in counts:
            counts[bstr] = 0

    target_ideal_prob = prob_dict[target]
    target_count = counts.get(target, 0)
    target_sampled_freq = target_count / shots

    # Text circuit representation
    circuit_text = circuit_ideal.draw(output="text")

    optimal_iter = calculate_optimal_iterations(n)

    return {
        "n_qubits": n,
        "search_space_N": N,
        "target_state": target,
        "iterations": iterations,
        "optimal_iterations": optimal_iter,
        "shots": shots,
        "statevector": statevector,
        "ideal_probabilities": prob_dict,
        "counts": counts,
        "target_ideal_prob": target_ideal_prob,
        "target_sampled_freq": target_sampled_freq,
        "target_count": target_count,
        "total_prob_check": total_prob,
        "circuit_text": circuit_text,
    }
