# Quantum Astra: Grover Search Algorithm Specification

**Project**: Quantum Astra  
**Project ID**: VTU-QFF-HC-0008  
**Hackathon**: VTU–Qiskit Fall Fest 2026  
**Track**: Offline Quantum Computing Lab & Educational Simulator  

---

## 1. Executive Summary & Problem Formulation

In unstructured database search, given an unsorted database of $N = 2^n$ items and an oracle function $f(x)$ such that:
$$
f(x) = \begin{cases} 
1 & \text{if } x = s \text{ (marked target item)} \\ 
0 & \text{if } x \neq s 
\end{cases}
$$
A classical deterministic algorithm requires $N$ queries in the worst case, and $\frac{N+1}{2}$ queries on average. In 1996, Lov K. Grover introduced a quantum algorithm that locates the marked target item in only $\mathcal{O}(\sqrt{N})$ oracle queries, achieving a quadratic query-complexity speedup that is provably optimal for black-box search (Bennett, Bernstein, Brassard, Vazirani, 1997).

Quantum Astra implements an offline, high-fidelity simulator for Grover's search algorithm with transparent quantum circuit construction, statevector tracking, and simulated Aer measurement sampling.

---

## 2. Algorithm Workflow & Mathematical Formulation

Grover's algorithm operates in four sequential stages:

```
|0⟩^⊗n ──► [ H^⊗n ] ──► ┌──────────┐ ──► ┌──────────┐ ──► ... ──► [ Measure ] ──► Bitstring
                        │  Oracle  │     │Diffusion │        (R times)
                        │   U_w    │     │    D     │
                        └──────────┘     └──────────┘
```

### Stage A: State Initialization
All $n$ qubits start in the ground state $|0\rangle^{\otimes n}$. Applying a Walsh-Hadamard transform $H^{\otimes n}$ creates a uniform, symmetric superposition across all $N = 2^n$ computational basis states:
$$
|s_0\rangle = H^{\otimes n} |0\rangle^{\otimes n} = \frac{1}{\sqrt{N}} \sum_{x=0}^{N-1} |x\rangle
$$
At this stage, every basis state has amplitude $\frac{1}{\sqrt{N}}$ and probability $P(|x\rangle) = \frac{1}{N}$.

### Stage B: Phase Inversion Oracle ($U_w$)
The quantum oracle evaluates the predicate function $f(x)$ and inverts the phase (sign) of the marked state $|s\rangle$ without altering the probabilities during the oracle query itself:
$$
U_w |x\rangle = (-1)^{f(x)} |x\rangle = \begin{cases}
-|s\rangle & \text{if } x = s \\
+|x\rangle & \text{if } x \neq s
\end{cases}
$$
In matrix operator form:
$$
U_w = I - 2|s\rangle\langle s|
$$
**Circuit Realization**:
1. For any qubit $i$ where the target bitstring has $s[n-1-i] = 0$, apply a Pauli-$X$ gate.
2. Apply a multi-controlled Phase gate (or multi-controlled $Z$ via $H \cdot MCX \cdot H$).
3. Re-apply Pauli-$X$ gates to restore the basis.

### Stage C: Grover Diffusion Operator ($D$)
The diffusion operator performs an inversion (reflection) about the average amplitude of all states:
$$
D = 2|s_0\rangle\langle s_0| - I = H^{\otimes n} \left( 2|0\dots0\rangle\langle 0\dots0| - I \right) H^{\otimes n}
$$
For an arbitrary state $|\psi\rangle = \sum_x \alpha_x |x\rangle$ with mean amplitude $\mu = \frac{1}{N} \sum_x \alpha_x$:
$$
D |\psi\rangle = \sum_x (2\mu - \alpha_x) |x\rangle
$$
Because the target state $|s\rangle$ had its amplitude negated by the oracle ($\alpha_s < 0$), reflection about the positive mean $\mu > 0$ amplifies the target state's amplitude to $2\mu - (-|\alpha_s|) = 2\mu + |\alpha_s|$, while simultaneously suppressing the amplitudes of the unmarked states.

**Circuit Realization**:
$$
D = H^{\otimes n} \cdot X^{\otimes n} \cdot MCZ \cdot X^{\otimes n} \cdot H^{\otimes n}
$$

### Stage D: Repetition & 2D Rotation Geometry
The combined Grover iteration operator $G = D \cdot U_w$ acts as a pure rotation in the two-dimensional real subspace spanned by:
1. The target state $|s\rangle$
2. The uniform superposition of all non-target states $|s^\perp\rangle = \frac{1}{\sqrt{N-1}} \sum_{x \neq s} |x\rangle$

Defining the initial state angle:
$$
\sin\theta = \frac{1}{\sqrt{N}} \implies \theta = \arcsin\left(\frac{1}{\sqrt{N}}\right)
$$
The state after $k$ Grover iterations is exactly:
$$
|\psi_k\rangle = G^k |s_0\rangle = \cos((2k+1)\theta) |s^\perp\rangle + \sin((2k+1)\theta) |s\rangle
$$
The probability of observing the target state upon measurement is:
$$
P(s) = \sin^2((2k+1)\theta)
$$

---

## 3. Optimal Iteration Count and Over-Rotation

To maximize success probability, we require $(2k+1)\theta \approx \frac{\pi}{2}$:
$$
2k + 1 \approx \frac{\pi}{2\theta} \implies k \approx \frac{\pi}{4\theta} - \frac{1}{2}
$$
For large $N$, $\theta \approx \frac{1}{\sqrt{N}}$, yielding the well-known approximation:
$$
R_{\text{optimal}} = \left\lfloor \frac{\pi}{4} \sqrt{N} \right\rfloor
$$

### Over-Rotation Phenomenon
Because Grover's algorithm is a continuous unitary rotation in a 2D Hilbert subspace, continuing to apply iterations beyond $R_{\text{optimal}}$ does **not** increase probability; instead, the state vector rotates past $|s\rangle$ toward $-|s^\perp\rangle$, causing the success probability to oscillate periodically like a harmonic oscillator.

---

## 4. Benchmark: Default Hackathon Experiment (4 Qubits, N=16, Target '1010')

For $n=4$ qubits ($N=16$ states) and single target $s = 1010$:
- $\sin\theta = \frac{1}{\sqrt{16}} = 0.25 \implies \theta = \arcsin(0.25) \approx 0.252680 \text{ rad} \ (14.48^\circ)$
- Optimal iteration count: $R = \text{round}\left(\frac{\pi}{4 \times 0.252680} - 0.5\right) = \text{round}(2.608) = 3$ iterations.

| Iteration ($k$) | Angle $(2k+1)\theta$ | Target Amplitude | Target Probability | Status |
|:---:|:---:|:---:|:---:|:---|
| **0** (Superposition) | $1\theta = 0.2527$ rad | $0.2500$ | **$6.25\%$** | Uniform Superposition |
| **1** | $3\theta = 0.7580$ rad | $0.6875$ | **$47.27\%$** | First Amplification |
| **2** | $5\theta = 1.2634$ rad | $0.9529$ | **$90.84\%$** | Near-Optimal |
| **3** | $7\theta = 1.7688$ rad | $0.9805$ | **$96.13\%$** | **Optimal Peak** |
| **4** | $9\theta = 2.2741$ rad | $0.7629$ | **$58.20\%$** | Over-Rotation |

---

## 5. Classical vs Quantum Complexity Comparison

| Metric | Classical Linear Search | Quantum Grover Search | Advantage |
|:---|:---:|:---:|:---:|
| **Best-Case Queries** | $1$ | $1$ | Equal |
| **Average-Case Queries** | $\frac{N+1}{2} = \mathcal{O}(N)$ | $\approx \frac{\pi}{4}\sqrt{N} = \mathcal{O}(\sqrt{N})$ | **Quadratic Speedup** |
| **Worst-Case Queries** | $N = \mathcal{O}(N)$ | $\approx \frac{\pi}{4}\sqrt{N} = \mathcal{O}(\sqrt{N})$ | **Quadratic Speedup** |
| **Memory Footprint** | $\mathcal{O}(\log N)$ bits | $\mathcal{O}(\log N)$ qubits | Logarithmic Registers |
| **Optimality** | Classical lower bound $\Omega(N)$ | Quantum query bound $\Omega(\sqrt{N})$ | Provably Optimal |

### Key Distinction: Local Simulation vs Physical Quantum Execution
- **Physical Quantum Hardware**: Executes the unitary gate sequence in $\mathcal{O}(\sqrt{N})$ physical gate times.
- **Classical Software Simulator**: Simulates quantum state evolution by maintaining an explicit statevector of $2^n$ complex numbers. Each matrix-vector multiplication costs $\mathcal{O}(2^n)$ classical operations. Therefore, simulating quantum circuits classically does not accelerate classical computers; it allows engineers and students to inspect, debug, and understand the quantum dynamics offline.

---

## 6. Qiskit Qubit and Bitstring Indexing Convention

In Qiskit:
- Qubit $q_0$ is the **Least Significant Bit (LSB)**, corresponding to the rightmost character of the binary string ($2^0$).
- Qubit $q_{n-1}$ is the **Most Significant Bit (MSB)**, corresponding to the leftmost character ($2^{n-1}$).

Quantum Astra strictly respects this convention throughout circuit synthesis, statevector indexing, and measurement sampling, ensuring that basis state $|1010\rangle$ corresponds to $q_3=1, q_2=0, q_1=1, q_0=0$.
