"""
Quantum Astra — Quantum Core Module
Provides input validation, Qiskit integration, and a local high-fidelity
quantum simulation engine for offline exploration.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# Check if native Qiskit and Qiskit Aer are installed
try:
    import qiskit
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    try:
        from qiskit_aer import AerSimulator
        HAS_AER = True
    except ImportError:
        AerSimulator = None
        HAS_AER = False
    from qiskit.quantum_info import Statevector
    HAS_QISKIT = True
except ImportError:
    qiskit = None
    HAS_QISKIT = False
    HAS_AER = False
    QuantumCircuit = None
    AerSimulator = None
    Statevector = None


def validate_qubit_count(n: int) -> int:
    """
    Validate that the qubit count is an integer between 2 and 8 inclusive.
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Qubit count must be an integer, got {type(n).__name__}")
    if n < 2 or n > 8:
        raise ValueError(f"Supported qubit count is between 2 and 8 inclusive, got {n}")
    return int(n)


def validate_target_bitstring(target: str, n_qubits: int) -> str:
    """
    Validate that the target bitstring is binary ('0' or '1')
    and matches the specified register length.
    """
    if not isinstance(target, str):
        raise TypeError(f"Target bitstring must be a string, got {type(target).__name__}")
    target = target.strip()
    if len(target) != n_qubits:
        raise ValueError(
            f"Target bitstring length ({len(target)}) does not match qubit register size ({n_qubits})"
        )
    if not all(c in ("0", "1") for c in target):
        raise ValueError(
            f"Target bitstring must contain only binary digits '0' and '1', got '{target}'"
        )
    return target


def validate_iterations(iterations: int) -> int:
    """
    Validate that Grover iteration count is a non-negative integer.
    """
    if not isinstance(iterations, (int, np.integer)):
        raise TypeError(f"Iteration count must be an integer, got {type(iterations).__name__}")
    if iterations < 0:
        raise ValueError(f"Grover iteration count cannot be negative, got {iterations}")
    if iterations > 100:
        raise ValueError(f"Grover iteration count exceeds practical educational limit (100), got {iterations}")
    return int(iterations)


def validate_shots(shots: int) -> int:
    """
    Validate that shot count is a positive integer.
    """
    if not isinstance(shots, (int, np.integer)):
        raise TypeError(f"Shot count must be an integer, got {type(shots).__name__}")
    if shots <= 0:
        raise ValueError(f"Measurement shots must be positive, got {shots}")
    if shots > 100000:
        raise ValueError(f"Measurement shots exceed recommended limit (100,000), got {shots}")
    return int(shots)


class LocalQuantumCircuit:
    """
    High-fidelity pure-NumPy quantum circuit engine.
    Ensures Quantum Astra runs completely offline with zero external network dependencies,
    matching Qiskit's exact qubit indexing (qubit 0 = LSB) and gate definitions.
    """

    def __init__(self, n_qubits: int, n_clbits: Optional[int] = None, name: str = "qc"):
        self.num_qubits = validate_qubit_count(n_qubits)
        self.num_clbits = n_qubits if n_clbits is None else int(n_clbits)
        self.name = name
        self.gates: List[Tuple[str, List[int], Dict]] = []
        self._statevector: Optional[np.ndarray] = None
        self._measured: bool = False

    def h(self, qubit: int):
        """Apply Hadamard gate to specified qubit."""
        self._check_qubit(qubit)
        self.gates.append(("h", [qubit], {}))
        self._statevector = None
        return self

    def x(self, qubit: int):
        """Apply Pauli-X gate to specified qubit."""
        self._check_qubit(qubit)
        self.gates.append(("x", [qubit], {}))
        self._statevector = None
        return self

    def z(self, qubit: int):
        """Apply Pauli-Z gate to specified qubit."""
        self._check_qubit(qubit)
        self.gates.append(("z", [qubit], {}))
        self._statevector = None
        return self

    def mcx(self, control_qubits: List[int], target_qubit: int):
        """Apply multi-controlled Pauli-X gate."""
        for q in control_qubits:
            self._check_qubit(q)
        self._check_qubit(target_qubit)
        if target_qubit in control_qubits:
            raise ValueError("Target qubit cannot be in control qubits")
        self.gates.append(("mcx", list(control_qubits) + [target_qubit], {"controls": len(control_qubits)}))
        self._statevector = None
        return self

    def mcz(self, control_qubits: List[int], target_qubit: int):
        """Apply multi-controlled Pauli-Z gate using H-MCX-H decomposition."""
        self.h(target_qubit)
        self.mcx(control_qubits, target_qubit)
        self.h(target_qubit)
        return self

    def measure(self, qubit: int, clbit: int):
        """Add measurement indicator."""
        self._check_qubit(qubit)
        self.gates.append(("measure", [qubit, clbit], {}))
        self._measured = True
        return self

    def measure_all(self):
        """Measure all qubits into matching classical bits."""
        for i in range(self.num_qubits):
            self.measure(i, i)
        return self

    def _check_qubit(self, qubit: int):
        if not (0 <= qubit < self.num_qubits):
            raise IndexError(f"Qubit index {qubit} out of range [0, {self.num_qubits - 1}]")

    def to_statevector(self) -> np.ndarray:
        """
        Compute the ideal statevector by sequential unitary gate application.
        Follows Qiskit's tensor ordering: qubit 0 is LSB, qubit (n-1) is MSB.
        Basis state index k corresponds to binary string format(k, f'0{n}b').
        """
        if self._statevector is not None:
            return self._statevector.copy()

        n = self.num_qubits
        N = 2 ** n
        # Initial state |0...0>
        psi = np.zeros(N, dtype=np.complex128)
        psi[0] = 1.0

        for gate_name, qubits, meta in self.gates:
            if gate_name == "h":
                q = qubits[0]
                psi = self._apply_1q_gate(psi, q, np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2))
            elif gate_name == "x":
                q = qubits[0]
                psi = self._apply_1q_gate(psi, q, np.array([[0, 1], [1, 0]], dtype=np.complex128))
            elif gate_name == "z":
                q = qubits[0]
                psi = self._apply_1q_gate(psi, q, np.array([[1, 0], [0, -1]], dtype=np.complex128))
            elif gate_name == "mcx":
                num_controls = meta["controls"]
                controls = qubits[:num_controls]
                target = qubits[num_controls]
                psi = self._apply_mcx(psi, controls, target)
            elif gate_name == "measure":
                # Measurement gates do not affect the ideal pre-measurement statevector
                pass

        # Verify normalization
        norm = np.linalg.norm(psi)
        if norm > 1e-12:
            psi = psi / norm

        self._statevector = psi
        return psi.copy()

    def _apply_1q_gate(self, psi: np.ndarray, target_qubit: int, matrix: np.ndarray) -> np.ndarray:
        """
        Apply a 1-qubit gate to target_qubit (where 0 is LSB, n-1 is MSB).
        """
        n = self.num_qubits
        axis = n - 1 - target_qubit
        tensor = psi.reshape([2] * n)
        tensor = np.tensordot(matrix, tensor, axes=(1, axis))
        tensor = np.moveaxis(tensor, 0, axis)
        return tensor.reshape(-1)

    def _apply_mcx(self, psi: np.ndarray, controls: List[int], target: int) -> np.ndarray:
        """
        Apply multi-controlled X gate.
        Flips target qubit when all control qubits are 1.
        """
        n = self.num_qubits
        N = 2 ** n
        new_psi = psi.copy()

        control_mask = 0
        for c in controls:
            control_mask |= (1 << c)
        target_mask = (1 << target)

        for i in range(N):
            if (i & control_mask) == control_mask:
                if (i & target_mask) == 0:
                    partner = i | target_mask
                    new_psi[i] = psi[partner]
                    new_psi[partner] = psi[i]

        return new_psi

    def sample_shots(self, shots: int = 1024, seed: Optional[int] = None) -> Dict[str, int]:
        """
        Simulate measurement counts with finite shot sampling from statevector probabilities.
        Returns a dictionary mapping bitstring 'c_{n-1}...c_0' to measurement count.
        """
        shots = validate_shots(shots)
        psi = self.to_statevector()
        probabilities = np.abs(psi) ** 2
        probabilities = np.maximum(probabilities, 0.0)
        prob_sum = np.sum(probabilities)
        if prob_sum > 0:
            probabilities = probabilities / prob_sum

        n = self.num_qubits
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(probabilities), size=shots, p=probabilities)

        counts: Dict[str, int] = {}
        for idx in indices:
            bitstring = format(int(idx), f"0{n}b")
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return dict(sorted(counts.items()))

    def draw(self, output: str = "text") -> str:
        """
        Render an ASCII circuit diagram of the gates.
        """
        n = self.num_qubits
        wire_lines = [f"q_{i}: " for i in range(n)]

        for gate_name, qubits, meta in self.gates:
            if gate_name in ("h", "x", "z"):
                q = qubits[0]
                box = f"-[ {gate_name.upper()} ]-"
                pad = "-" * len(box)
                for i in range(n):
                    wire_lines[i] += box if i == q else pad
            elif gate_name == "mcx":
                num_controls = meta.get("controls", len(qubits) - 1)
                controls = set(qubits[:num_controls])
                target = qubits[num_controls]
                for i in range(n):
                    if i in controls:
                        wire_lines[i] += "---*---"
                    elif i == target:
                        wire_lines[i] += "--(X)--"
                    else:
                        wire_lines[i] += "-------"
            elif gate_name == "measure":
                q = qubits[0]
                for i in range(n):
                    wire_lines[i] += "-[ M ]-" if i == q else "-------"

        for i in range(n):
            wire_lines[i] += "-|"

        return "\n".join(wire_lines)


def get_backend_info() -> Dict[str, Union[str, bool]]:
    """
    Return diagnostic information about the available quantum backends.
    """
    q_ver = getattr(qiskit, "__version__", "N/A") if qiskit is not None else "N/A"
    return {
        "has_qiskit": HAS_QISKIT,
        "has_aer": HAS_AER,
        "qiskit_version": q_ver,
        "default_engine": "Qiskit AerSimulator" if HAS_AER else "Quantum Astra Local Aer-Equivalent Simulator",
        "mode": "Offline Local Simulation",
        "requires_api_key": False,
    }
