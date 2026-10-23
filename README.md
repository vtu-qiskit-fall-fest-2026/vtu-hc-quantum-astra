# Quantum Astra — Offline Grover Search Lab

**Project Name:** Quantum Astra  
**Project ID:** `VTU-QFF-HC-0008`  
**Tagline:** Explore the Quantum World  
**Hackathon:** VTU–Qiskit Fall Fest 2026  
**Team Lead:** Sagar Zingade  
**Team Member:** Abhilash  
**GitHub Repository:** [https://github.com/vtu-qiskit-fall-fest-2026/vtu-hc-quantum-astra](https://github.com/vtu-qiskit-fall-fest-2026/vtu-hc-quantum-astra)  

---

## 1. Problem Statement
Quantum algorithms are difficult for beginners, educators, and university students to understand because their underlying mathematical state transformations, phase operations, and measurement probabilities are abstract and non-intuitive. Furthermore, many modern cloud-based quantum developer platforms require API keys (e.g., OpenAI, Gemini, IBM Quantum Experience credentials), continuous high-speed internet connections, and complex authentication flows. This presents a barrier for hands-on classroom exploration, hackathons, and offline lab environments.

---

## 2. Objectives and Novelty
**Objectives:**
- Provide a completely local, self-contained educational simulator for Grover's search algorithm ($n=2$ to $8$ qubits, $N=4$ to $256$ states).
- Visualize step-by-step amplitude amplification, demonstrating how phase inversion and inversion about the mean amplify the target state's measurement probability.
- Contrast theoretical quantum query complexity $\mathcal{O}(\sqrt{N})$ with classical linear search $\mathcal{O}(N)$.
- Distinguish between ideal mathematical statevector probabilities and finite-shot measurement statistics from Qiskit Aer simulations.

**Novelty:**
- **Zero API Key Requirement:** Absolutely no cloud API keys, LLM SDKs, or paid third-party services.
- **Dual-Engine Architecture:** Runs natively with official Qiskit and Qiskit Aer when installed, and incorporates a zero-dependency high-fidelity local simulator fallback for guaranteed offline resilience.
- **Interactive Didactic Visualizer:** Dynamic dark-themed Matplotlib charts, basis-state inspection tables, ASCII quantum circuit diagrams, and full CSV export capabilities.

---

## 3. Solution Summary
Quantum Astra delivers a demo-ready dashboard built on Streamlit, Qiskit, and Matplotlib. Students can select register sizes between 2 and 8 qubits, configure any arbitrary valid binary target bitstring (e.g., `1010`), adjust the number of Grover iterations, view the synthesized circuit, inspect the resulting probability distribution, sample measurement shots, and export structured datasets for lab reports.

---

## 4. Algorithm Methodology
Grover's algorithm provides an optimal quadratic query-complexity speedup for searching an unsorted database of $N = 2^n$ items.

1. **Initialization:**
   All qubits are initialized in $|0\rangle^{\otimes n}$. Applying a Walsh-Hadamard transform $H^{\otimes n}$ generates a uniform superposition:
   $$|s_0\rangle = \frac{1}{\sqrt{N}} \sum_{x=0}^{N-1} |x\rangle$$
   Each state initially has probability $P(x) = \frac{1}{N}$.

2. **Phase Inversion Oracle ($U_w$):**
   The oracle inverts the phase of the target marked state $|s\rangle$ without altering state probabilities:
   $$U_w |x\rangle = (-1)^{f(x)} |x\rangle = (I - 2|s\rangle\langle s|) |x\rangle$$

3. **Grover Diffusion Operator ($D$):**
   Performs reflection about the mean amplitude $\mu = \frac{1}{N}\sum_x \alpha_x$:
   $$D = 2|s_0\rangle\langle s_0| - I = H^{\otimes n} (2|0\dots0\rangle\langle 0\dots0| - I) H^{\otimes n}$$
   This amplifies the negative amplitude of the marked state to $2\mu + |\alpha_s|$ while diminishing the amplitudes of unmarked states.

4. **Optimal Iterations:**
   For a single target in $N$ states, the success probability after $k$ iterations is:
   $$P(s) = \sin^2((2k + 1)\theta), \quad \text{where } \sin\theta = \frac{1}{\sqrt{N}}$$
   The optimal iteration count is:
   $$R_{\text{optimal}} \approx \left\lfloor \frac{\pi}{4}\sqrt{N} \right\rfloor$$
   Applying additional iterations causes **over-rotation**, decreasing the target state probability.

---

## 5. System Architecture

```
quantum-astra/
├── app.py                     # Streamlit Scientific-Tech Dashboard
├── requirements.txt           # Pinned, tested Python dependencies
├── run_tests.py               # Automated test runner (18/18 tests pass)
├── README.md                  # Complete documentation and setup manual
├── .gitignore                 # Standard clean Git tracking rules
├── src/
│   ├── __init__.py            # Package initialization & path loader
│   ├── quantum_core.py        # Qiskit integration, validations & local Aer engine
│   ├── grover.py              # Phase oracle, diffusion operator & simulation runner
│   ├── measurements.py        # Finite-shot sampling & binomial error bounds
│   ├── probability.py         # Statevector probabilities & CSV export
│   └── comparisons.py         # Classical vs Quantum complexity models
├── tests/
│   ├── __init__.py            # Test suite package bootstrap
│   ├── test_validation.py     # Tests 1-2: Parameter constraints & edge cases
│   ├── test_grover.py         # Tests 3-5, 8-10: Gates, oracle, diffusion, default experiment
│   └── test_probabilities.py  # Tests 6-7, 11-12: Normalization, shots, classical formulas
├── assets/
│   └── quantum_astra.svg      # Local SVG branding emblem
└── docs/
    └── algorithm.md           # Formal mathematical and algorithmic documentation
```

---

## 6. Technology Stack
- **Language:** Python 3.11+
- **Quantum Computing Framework:** Qiskit 1.0+ and Qiskit Aer 0.14+
- **Interactive UI:** Streamlit 1.35+
- **Numerical Computation:** NumPy 1.26+, SciPy 1.12+
- **Data Handling & Tables:** Pandas 2.0+
- **Plotting & Visualization:** Matplotlib 3.8+
- **Test Automation:** Pytest 8.0+ / Python `unittest`

---

## 7. Installation & Setup Instructions

### Prerequisites
- Python 3.11 or higher installed on your local machine.
- Git (optional, for cloning).

### Step 1: Clone or Download the Project
```bash
git clone https://github.com/vtu-qiskit-fall-fest-2026/vtu-hc-quantum-astra.git
cd vtu-hc-quantum-astra
```

### Step 2: Create and Activate a Virtual Environment

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows PowerShell:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 8. Launch Instructions

To launch the interactive dashboard:

```bash
streamlit run app.py
```

The web dashboard will automatically open in your default browser at `http://localhost:8501`.

---

## 9. Automated Testing

To execute all 18 automated unit tests across the 12 evaluation criteria:

```bash
python run_tests.py
```
*(Or with pytest: `pytest tests/ -v`)*

### Test Verification Summary:
- **[PASS] 1.** Valid and invalid qubit counts ($n \in [2, 8]$)
- **[PASS] 2.** Target bitstring length and character validation
- **[PASS] 3.** Single-qubit Hadamard gate transformations ($H|0\rangle, H|1\rangle, H^2=I$)
- **[PASS] 4.** Oracle phase inversion for arbitrary target basis states
- **[PASS] 5.** Diffusion operator normalization and mean-reflection properties
- **[PASS] 6.** Statevector normalization ($\sum |\alpha_i|^2 = 1.0 \pm 10^{-6}$)
- **[PASS] 7.** Non-negative probability calculations
- **[PASS] 8.** Qiskit bitstring ordering and qubit indexing consistency (qubit 0 = LSB)
- **[PASS] 9.** Multi-target and multi-qubit Grover search across $n \in [2, 5]$
- **[PASS] 10.** Monotonic probability improvement for the default 4-qubit example ($\approx 96.13\%$)
- **[PASS] 11.** Measurement sampling output validity and standard error bounds
- **[PASS] 12.** Classical query-count formulas and quantum comparison metrics

---

## 10. Default Hackathon Experiment & Interpretation

**Default Configuration:**
- Number of qubits: `4`
- Search space: `N = 16` basis states
- Target state: `1010`
- Grover iterations: `3`
- Simulated shots: `2048`

**Results & Interpretation:**
- **Initial Superposition (0 iterations):** Each state has uniform probability $\frac{1}{16} = 6.25\%$.
- **Iteration 1:** Target state probability increases to $47.27\%$.
- **Iteration 2:** Target state probability reaches $90.84\%$.
- **Iteration 3 (Optimal Peak):** Target state probability reaches **$96.13\%$** ($96.1319\%$).
- **Iteration 4 (Over-Rotation):** Target state probability drops to $58.20\%$, illustrating the 2D geometric rotation dynamic of Grover operators.
- **Classical Comparison:** A classical linear search requires an average of $(16 + 1) / 2 = 8.5$ queries. Grover's algorithm finds the marked item with $>96\%$ confidence in just 3 quantum queries.

---

## 11. Important Scientific Disclosures & Limitations

1. **Classical Simulation Disclosure:**
   Quantum Astra is a software simulation running on classical computer hardware (CPUs/GPUs). It does **not** represent physical quantum hardware execution. Simulating an $n$-qubit quantum state classically requires tracking $2^n$ complex amplitudes in memory, which scales exponentially with $n$. Therefore, local classical simulation is practically restricted to $n \le 16$ qubits ($n \le 8$ in the interactive UI for instantaneous interactivity).
2. **Oracle Complexity vs End-to-End Speedup:**
   Grover's algorithm provides a provable quadratic speedup in terms of **oracle queries** for unstructured search. In practical applications, the computational cost of synthesizing and evaluating the quantum oracle circuit must be considered when determining end-to-end real-world performance.

---

## 12. Future Scope & Roadmap
- Implement multi-target simultaneous search with dynamic optimal angle calculation $\arcsin(\sqrt{M/N})$.
- Integrate quantum noise simulation models (depolarizing channels, bit-flip, and amplitude damping) using Qiskit Aer noise models.
- Add Quantum Counting (combining Quantum Phase Estimation with Grover iterations) to count marked items when $M$ is unknown.

---

## 13. License & Acknowledgments
Developed by **Sagar Zingade** and **Abhilash** for **VTU–Qiskit Fall Fest 2026** (Project ID: `VTU-QFF-HC-0008`).  
Built with [Qiskit](https://qiskit.org/) and [Streamlit](https://streamlit.io/).

---

## 🎥 Project Demo Video

**Project:** Quantum Astra  
**Team ID:** VTU-QFF-HC-0008

▶️ [Watch the Demo Video on YouTube](https://youtu.be/p92FF1HiGME?si=dctHeeCPM6XOI9Vm)

## 📊 Project Presentation

📥 [Download the PowerPoint Presentation](./VTU-QFF-HC-0008.pptx)
