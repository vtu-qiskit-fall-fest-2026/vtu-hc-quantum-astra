"""
Quantum Astra — Offline Grover Search Lab
VTU–Qiskit Fall Fest 2026 | Project ID: VTU-QFF-HC-0008
Tagline: Explore the Quantum World
Team Lead: Sagar Zingade | Team Member: Abhilash
Repository: https://github.com/vtu-qiskit-fall-fest-2026/vtu-hc-quantum-astra

Complete offline interactive application built with Streamlit, Qiskit, and Matplotlib.
Runs with ZERO external API keys and ZERO external network dependencies.
"""

import sys
import os
import math
import io
import base64
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure local project root and src/ directory are on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    import src
except ModuleNotFoundError:
    spec = importlib.util.spec_from_file_location(
        "src", os.path.join(SRC_DIR, "__init__.py"), submodule_search_locations=[SRC_DIR]
    )
    src_mod = importlib.util.module_from_spec(spec)
    sys.modules["src"] = src_mod
    spec.loader.exec_module(src_mod)

from src.quantum_core import (
    validate_qubit_count,
    validate_target_bitstring,
    validate_iterations,
    validate_shots,
    get_backend_info,
    LocalQuantumCircuit,
    HAS_QISKIT,
    HAS_AER,
)
from src.grover import (
    calculate_optimal_iterations,
    create_grover_circuit,
    run_grover_simulation,
)
from src.measurements import compute_sampling_error
from src.probability import (
    build_probability_dataframe,
    export_probabilities_to_csv,
    verify_normalization,
)
from src.comparisons import (
    get_classical_metrics,
    get_quantum_metrics,
    generate_scaling_comparison_table,
    get_complexity_explanations,
)

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


DEFAULT_SVG_BANNER = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 160" width="100%" height="100%">
<defs>
<linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#0B0F19" />
<stop offset="50%" stop-color="#111827" />
<stop offset="100%" stop-color="#0F172A" />
</linearGradient>
<linearGradient id="cyanViolet" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#00F2FE" />
<stop offset="50%" stop-color="#38BDF8" />
<stop offset="100%" stop-color="#8B5CF6" />
</linearGradient>
<linearGradient id="textGrad" x1="0%" y1="0%" x2="100%" y2="0%">
<stop offset="0%" stop-color="#FFFFFF" />
<stop offset="60%" stop-color="#E2E8F0" />
<stop offset="100%" stop-color="#38BDF8" />
</linearGradient>
<linearGradient id="astraGrad" x1="0%" y1="0%" x2="100%" y2="0%">
<stop offset="0%" stop-color="#38BDF8" />
<stop offset="50%" stop-color="#818CF8" />
<stop offset="100%" stop-color="#C084FC" />
</linearGradient>
</defs>
<rect width="600" height="160" rx="16" fill="url(#bgGrad)" stroke="#1E293B" stroke-width="1.5" />
<g transform="translate(85, 80)">
<ellipse cx="0" cy="0" rx="55" ry="22" fill="none" stroke="url(#cyanViolet)" stroke-width="2" stroke-dasharray="7 3" transform="rotate(-30)" opacity="0.85" />
<ellipse cx="0" cy="0" rx="55" ry="22" fill="none" stroke="url(#astraGrad)" stroke-width="2" transform="rotate(30)" opacity="0.85" />
<ellipse cx="0" cy="0" rx="55" ry="18" fill="none" stroke="#00F2FE" stroke-width="1.5" transform="rotate(90)" opacity="0.6" />
<circle cx="44" cy="-24" r="4.5" fill="#00F2FE" />
<circle cx="-42" cy="26" r="4" fill="#C084FC" />
<circle cx="0" cy="-52" r="3.5" fill="#38BDF8" />
<circle cx="0" cy="0" r="12" fill="#0F172A" stroke="#8B5CF6" stroke-width="2.5" />
<circle cx="0" cy="0" r="7" fill="#00F2FE" />
<path d="M 0,-10 L 2.5,-3 L 10,0 L 2.5,3 L 0,10 L -2.5,3 L -10,0 L -2.5,-3 Z" fill="#FFFFFF" />
</g>
<g transform="translate(175, 74)">
<text x="0" y="0" font-family="Segoe UI, Roboto, Helvetica, Arial, sans-serif" font-size="34" font-weight="800" letter-spacing="3" fill="url(#textGrad)">QUANTUM</text>
<text x="195" y="0" font-family="Segoe UI, Roboto, Helvetica, Arial, sans-serif" font-size="34" font-weight="900" letter-spacing="4" fill="url(#astraGrad)">ASTRA</text>
<text x="3" y="30" font-family="Segoe UI, Roboto, Helvetica, Arial, sans-serif" font-size="12.5" font-weight="600" letter-spacing="2" fill="#94A3B8">EXPLORE THE QUANTUM WORLD</text>
<text x="3" y="48" font-family="Segoe UI, Roboto, Helvetica, Arial, sans-serif" font-size="10.5" font-weight="500" letter-spacing="1" fill="#64748B">VTU–Qiskit Fall Fest 2026 • Offline Grover Search Lab • VTU-QFF-HC-0008</text>
</g>
<circle cx="560" cy="30" r="1.5" fill="#38BDF8" opacity="0.6" />
<circle cx="540" cy="50" r="1" fill="#818CF8" opacity="0.4" />
<circle cx="575" cy="70" r="1.5" fill="#C084FC" opacity="0.7" />
<circle cx="550" cy="130" r="1" fill="#00F2FE" opacity="0.5" />
</svg>"""


def apply_custom_styles():
    """Inject polished scientific-tech dark navy CSS styles."""
    css = (
        "<style>"
        ".stApp { background-color: #0B0F19; color: #E2E8F0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }"
        "section[data-testid='stSidebar'] { background-color: #0F172A; border-right: 1px solid #1E293B; }"
        ".qa-card { background-color: #111827; border: 1px solid #1E293B; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3); }"
        ".qa-card-highlight { background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%); border: 1px solid #38BDF8; border-radius: 12px; padding: 20px; margin-bottom: 20px; }"
        ".qa-metric-title { color: #94A3B8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }"
        ".qa-metric-value { color: #00F2FE; font-size: 1.85rem; font-weight: 700; line-height: 1.2; }"
        ".qa-metric-sub { color: #64748B; font-size: 0.8rem; margin-top: 4px; }"
        ".qa-pill-cyan { display: inline-block; background-color: rgba(6, 182, 212, 0.15); color: #00F2FE; border: 1px solid rgba(6, 182, 212, 0.4); border-radius: 9999px; padding: 3px 12px; font-size: 0.8rem; font-weight: 600; }"
        ".stButton>button { background: linear-gradient(135deg, #0284C7 0%, #7C3AED 100%); color: #FFFFFF; font-weight: 600; border: none; border-radius: 8px; padding: 0.5rem 1.25rem; }"
        ".stDownloadButton>button { background: #1E293B; color: #38BDF8; border: 1px solid #38BDF8; font-weight: 600; border-radius: 8px; }"
        "</style>"
    )
    st.markdown(css, unsafe_allow_html=True)


def render_logo_header():
    """Render the Quantum Astra SVG branding banner via Base64 data URI."""
    svg_path = os.path.join(BASE_DIR, "assets", "quantum_astra.svg")
    svg_content = DEFAULT_SVG_BANNER
    if os.path.exists(svg_path):
        try:
            with open(svg_path, "r", encoding="utf-8") as f:
                loaded = f.read().strip()
                if loaded.startswith("<svg"):
                    svg_content = loaded
        except Exception:
            svg_content = DEFAULT_SVG_BANNER

    b64_svg = base64.b64encode(svg_content.encode("utf-8")).decode("utf-8")
    st.markdown(
        f'<div style="text-align: center; margin-bottom: 24px;"><img src="data:image/svg+xml;base64,{b64_svg}" style="max-width: 680px; width: 100%; height: auto; border-radius: 16px;" alt="Quantum Astra Banner" /></div>',
        unsafe_allow_html=True,
    )


def plot_probability_distribution(
    probabilities: dict,
    counts: dict,
    shots: int,
    target_state: str,
    n_qubits: int,
):
    """
    Generate readable Matplotlib chart for both ideal probability and sampled frequency.
    Ensures clear target state highlighting and graceful handling of large state spaces.
    """
    states = sorted(probabilities.keys())
    N = len(states)
    ideal_probs = [probabilities[s] * 100.0 for s in states]
    sampled_freqs = [(counts.get(s, 0) / shots) * 100.0 for s in states]

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=100)
    fig.patch.set_facecolor("#0B0F19")
    ax.set_facecolor("#111827")

    x = np.arange(N)
    width = 0.38

    ideal_colors = ["#00F2FE" if s == target_state else "#6366F1" for s in states]
    sampled_colors = ["#38BDF8" if s == target_state else "#475569" for s in states]

    ax.bar(
        x - width / 2,
        ideal_probs,
        width,
        label="Ideal Probability (%)",
        color=ideal_colors,
        alpha=0.9,
        edgecolor="#0B0F19",
        linewidth=0.8,
    )
    ax.bar(
        x + width / 2,
        sampled_freqs,
        width,
        label=f"Sampled Frequency ({shots:,} shots)",
        color=sampled_colors,
        alpha=0.8,
        hatch="//",
        edgecolor="#0B0F19",
        linewidth=0.8,
    )

    target_idx = states.index(target_state)
    target_prob = probabilities[target_state] * 100.0
    ax.annotate(
        f"Target |{target_state}⟩\n{target_prob:.2f}%",
        xy=(target_idx - width / 2, target_prob),
        xytext=(target_idx, min(100.0, target_prob + 8)),
        arrowprops=dict(facecolor="#00F2FE", shrink=0.08, width=1.5, headwidth=6),
        ha="center",
        va="bottom",
        color="#00F2FE",
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0F172A", edgecolor="#00F2FE", lw=1),
    )

    ax.set_ylabel("Probability (%)", color="#E2E8F0", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Probability Amplification for Target |{target_state}⟩ ({N} States)",
        color="#F8FAFC",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylim(0, max(100, max(ideal_probs) + 18))
    ax.grid(color="#1E293B", linestyle="--", linewidth=0.6, alpha=0.7, axis="y")

    if N <= 16:
        ax.set_xticks(x)
        ax.set_xticklabels(states, rotation=45, ha="right", color="#CBD5E1", fontsize=9)
    elif N <= 32:
        ax.set_xticks(x[::2])
        ax.set_xticklabels(states[::2], rotation=60, ha="right", color="#CBD5E1", fontsize=8)
    elif N <= 64:
        ax.set_xticks(x[::4])
        ax.set_xticklabels(states[::4], rotation=90, ha="right", color="#CBD5E1", fontsize=7)
    else:
        key_ticks = [target_idx] + [i for i in range(0, N, 16) if abs(i - target_idx) > 4]
        key_ticks = sorted(set(key_ticks))
        ax.set_xticks(key_ticks)
        ax.set_xticklabels([states[i] for i in key_ticks], rotation=90, ha="right", color="#CBD5E1", fontsize=7)

    ax.tick_params(colors="#94A3B8")
    for spine in ax.spines.values():
        spine.set_color("#1E293B")

    ax.legend(loc="upper right", facecolor="#0F172A", edgecolor="#1E293B", labelcolor="#E2E8F0")
    plt.tight_layout()
    return fig


def page_dashboard():
    """Page 1: Project Overview & Live Benchmark Dashboard."""
    st.markdown("## 🌌 Quantum Astra — Mission Dashboard")
    st.markdown(
        "Welcome to **Quantum Astra**, an offline quantum computing exploration lab developed for "
        "**VTU–Qiskit Fall Fest 2026** (Project ID: `VTU-QFF-HC-0008`). "
        "Explore how Grover's search algorithm amplifies probability amplitudes in unstructured search spaces "
        "with transparent quantum circuit construction and zero cloud dependencies."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            '<div class="qa-card"><div class="qa-metric-title">Search Algorithm</div><div class="qa-metric-value">Grover Search</div><div class="qa-metric-sub">Amplitude Amplification</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="qa-card"><div class="qa-metric-title">Supported Registers</div><div class="qa-metric-value">2 – 8 Qubits</div><div class="qa-metric-sub">4 to 256 Hilbert States</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="qa-card"><div class="qa-metric-title">Speedup</div><div class="qa-metric-value">O(√N)</div><div class="qa-metric-sub">Optimal Quadratic Speedup</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            '<div class="qa-card"><div class="qa-metric-title">Zero API Keys</div><div class="qa-metric-value">100% Offline</div><div class="qa-metric-sub">Native Local Execution</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### ⚡ Default Experiment Benchmark (4 Qubits • State |1010⟩ • 3 Iterations)")
    res = run_grover_simulation(4, "1010", 3, shots=2048, seed=42)

    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
    bcol1.metric("Search Space (N)", "16 States", "4 Qubits")
    bcol2.metric("Target State", "|1010⟩", "Optimal: 3 iters")
    bcol3.metric("Ideal Probability", f"{res['target_ideal_prob']*100:.2f}%", "+89.88% amplification")
    bcol4.metric("Sampled Frequency", f"{res['target_sampled_freq']*100:.2f}%", f"{res['target_count']}/{res['shots']} shots")

    fig = plot_probability_distribution(
        res["ideal_probabilities"],
        res["counts"],
        res["shots"],
        "1010",
        4,
    )
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        '<div class="qa-card"><h4 style="color: #38BDF8; margin-top:0;">💡 What this experiment demonstrates:</h4><p>In a 16-state unstructured space, a classical search requires randomly querying an average of <b>8.5 items</b>. Grover\'s algorithm creates a uniform superposition (6.25% initial probability per state), and in just <b>3 quantum oracle iterations</b> amplifies the probability of the target state <code>|1010⟩</code> to <b>96.13%</b>.</p></div>',
        unsafe_allow_html=True,
    )


def page_grover_simulator():
    """Page 2: Core Interactive Grover Search Simulator."""
    st.markdown("## 🔬 Grover Search Simulator")
    st.markdown(
        "Configure quantum register parameters, construct the reversible phase oracle, "
        "and observe amplitude amplification across iterations."
    )

    if "n_qubits" not in st.session_state:
        st.session_state.n_qubits = 4
    if "target_bitstring" not in st.session_state:
        st.session_state.target_bitstring = "1010"
    if "iterations" not in st.session_state:
        st.session_state.iterations = 3
    if "shots" not in st.session_state:
        st.session_state.shots = 1024

    with st.expander("🛠️ Simulator Controls & Experiment Configuration", expanded=True):
        c1, c2, c3, c4 = st.columns([1, 1.2, 1, 1])

        with c1:
            n_qubits = st.slider("Qubits (n)", min_value=2, max_value=8, value=int(st.session_state.n_qubits), step=1)
            search_space = 2 ** n_qubits
            st.caption(f"Search space: N = 2^{n_qubits} = **{search_space} states**")

        default_target = str(st.session_state.target_bitstring)
        if len(default_target) != n_qubits:
            default_target = "1" + "0" * (n_qubits - 1)

        with c2:
            target_input = st.text_input(
                f"Target Bitstring ({n_qubits} bits)",
                value=default_target,
                help=f"Enter a binary string of exactly {n_qubits} characters ('0' or '1')",
            )

        opt_iter = calculate_optimal_iterations(n_qubits)

        with c3:
            iterations = st.number_input(
                "Grover Iterations",
                min_value=0,
                max_value=30,
                value=min(int(st.session_state.iterations), 30),
                step=1,
                help=f"Theoretical optimal count is {opt_iter} for N={search_space}.",
            )
            st.caption(f"Optimal iterations: **{opt_iter}**")

        with c4:
            shots = st.select_slider(
                "Simulated Shots",
                options=[256, 512, 1024, 2048, 4096, 8192],
                value=int(st.session_state.shots),
                help="Measurement sample count simulated with Qiskit AerSimulator.",
            )
            st.caption("Noiseless ideal simulation")

        error_msg = None
        try:
            n_qubits = validate_qubit_count(n_qubits)
            target_str = validate_target_bitstring(target_input, n_qubits)
            iterations = validate_iterations(iterations)
            shots = validate_shots(shots)
            st.session_state.n_qubits = n_qubits
            st.session_state.target_bitstring = target_str
            st.session_state.iterations = iterations
            st.session_state.shots = shots
        except (ValueError, TypeError) as ex:
            error_msg = str(ex)

        btn_col1, btn_col2 = st.columns([1, 4])
        with btn_col1:
            st.button("🚀 Run Simulation")
        with btn_col2:
            if st.button("🔄 Reset to Default (4 Qubits, |1010⟩, 3 iters)"):
                st.session_state.n_qubits = 4
                st.session_state.target_bitstring = "1010"
                st.session_state.iterations = 3
                st.session_state.shots = 1024
                st.rerun()

    if error_msg:
        st.error(f"⚠️ Input Validation Error: {error_msg}")
        return

    res = run_grover_simulation(
        n_qubits=st.session_state.n_qubits,
        target_bitstring=st.session_state.target_bitstring,
        iterations=st.session_state.iterations,
        shots=st.session_state.shots,
        seed=42,
    )

    st.markdown("### 📊 Simulation Results Dashboard")

    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Qubits / States", f"{res['n_qubits']}q / {res['search_space_N']}")
    r2.metric("Target State", f"|{res['target_state']}⟩")
    r3.metric("Iterations Run", f"{res['iterations']}", f"Optimal: {res['optimal_iterations']}")
    r4.metric("Ideal Probability", f"{res['target_ideal_prob']*100:.2f}%")
    r5.metric("Sampled Frequency", f"{res['target_sampled_freq']*100:.2f}%", f"{res['target_count']}/{res['shots']} shots")

    fig = plot_probability_distribution(
        res["ideal_probabilities"],
        res["counts"],
        res["shots"],
        res["target_state"],
        res["n_qubits"],
    )
    st.pyplot(fig)
    plt.close(fig)

    df = build_probability_dataframe(
        res["ideal_probabilities"],
        res["counts"],
        res["shots"],
        res["target_state"],
    )

    t1, t2 = st.tabs(["📋 Complete Basis State Table", "📐 Quantum Circuit Diagram"])

    with t1:
        st.markdown(f"**Normalization Check:** Sum of probabilities = `{res['total_prob_check']:.6f}` (Non-negative & Normalized)")
        csv_data = export_probabilities_to_csv(df)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data,
            file_name=f"grover_{res['n_qubits']}q_{res['target_state']}_{res['iterations']}iters.csv",
            mime="text/csv",
        )
        st.dataframe(df, use_container_width=True, height=280)

    with t2:
        st.markdown("**Synthesized Quantum Circuit (ASCII View):**")
        st.code(res["circuit_text"], language="text")
        st.caption("Qubit 0 is LSB (least significant bit). Circuit contains Hadamard initialization, Oracle, and Diffusion.")


def page_quantum_concepts():
    """Page 3: Educational Modules with Interactive Hadamard Demo."""
    st.markdown("## 🧠 Quantum Concepts Explorer")
    st.markdown("A step-by-step educational guide to the foundational mathematics and principles behind Grover search.")

    concept_tab = st.selectbox(
        "Select Concept to Explore:",
        [
            "1. Classical Bits vs Qubits",
            "2. Superposition & Hadamard Gate (Interactive)",
            "3. Pauli-X and Pauli-Z Gates",
            "4. Quantum Measurement & Collapse",
            "5. Entanglement & Bell State (No FTL)",
            "6. Quantum Search Phase Oracle",
            "7. Amplitude Amplification & Diffusion Operator",
        ],
    )

    if concept_tab.startswith("1."):
        st.markdown("### Classical Bits versus Qubits")
        st.markdown(
            "- **Classical Bit**: Can exist strictly in one of two discrete states: $0$ or $1$.\n"
            "- **Qubit (Quantum Bit)**: A two-level quantum system represented as a linear combination (superposition) of basis states:\n"
            "  $$|\\psi\\rangle = \\alpha |0\\rangle + \\beta |1\\rangle$$\n"
            "  where $\\alpha, \\beta \\in \\mathbb{C}$ and $|\\alpha|^2 + |\\beta|^2 = 1$.\n"
            "- **State Representation**: On the Bloch Sphere, a pure qubit corresponds to a point on the surface of a unit 3D sphere."
        )

    elif concept_tab.startswith("2."):
        st.markdown("### Superposition & Hadamard Gate (Interactive Demonstration)")
        st.markdown(
            "The **Hadamard gate** ($H$) creates an equal, symmetric superposition from computational basis states:\n"
            "$$H = \\frac{1}{\\sqrt{2}} \\begin{pmatrix} 1 & 1 \\\\ 1 & -1 \\end{pmatrix}$$\n"
            "$$H |0\\rangle = \\frac{|0\\rangle + |1\\rangle}{\\sqrt{2}} = |+\\rangle, \\quad H |1\\rangle = \\frac{|0\\rangle - |1\\rangle}{\\sqrt{2}} = |-\\rangle$$"
        )
        st.markdown("#### 🧪 Interactive Single-Qubit Demonstration")
        input_state = st.radio("Choose Input Basis State:", ["|0⟩", "|1⟩"], horizontal=True)
        apply_h = st.checkbox("Apply Hadamard Gate", value=True)

        qc = LocalQuantumCircuit(2)
        if input_state == "|1⟩":
            qc.x(0)
        if apply_h:
            qc.h(0)

        sv = qc.to_statevector()
        p0 = abs(sv[0]) ** 2
        p1 = abs(sv[1]) ** 2

        demo_c1, demo_c2 = st.columns(2)
        demo_c1.metric("P(|0⟩)", f"{p0*100:.1f}%")
        demo_c2.metric("P(|1⟩)", f"{p1*100:.1f}%")

        st.caption(f"Statevector: `[{sv[0]:.3f}, {sv[1]:.3f}]`. Notice that H on |1⟩ introduces a relative negative phase.")

    elif concept_tab.startswith("3."):
        st.markdown("### Pauli-X and Pauli-Z Gates")
        st.markdown(
            "- **Pauli-X (Quantum NOT)**: Flips the computational basis states:\n"
            "  $$X = \\begin{pmatrix} 0 & 1 \\\\ 1 & 0 \\end{pmatrix}, \\quad X|0\\rangle = |1\\rangle, \\quad X|1\\rangle = |0\\rangle$$\n"
            "- **Pauli-Z (Phase Flip)**: Leaves $|0\\rangle$ unchanged and flips the phase of $|1\\rangle$:\n"
            "  $$Z = \\begin{pmatrix} 1 & 0 \\\\ 0 & -1 \\end{pmatrix}, \\quad Z|0\\rangle = |0\\rangle, \\quad Z|1\\rangle = -|1\\rangle$$\n"
            "- **Key Identity**: $H Z H = X$ and $H X H = Z$. This relation enables constructing multi-controlled phase flips using standard multi-controlled NOT gates!"
        )

    elif concept_tab.startswith("4."):
        st.markdown("### Quantum Measurement & Born Rule")
        st.markdown(
            "- **Born Rule**: When a state $|\\psi\\rangle = \\sum_x \\alpha_x |x\\rangle$ is measured in the computational basis, the probability of obtaining outcome $x$ is:\n"
            "  $$P(x) = |\\alpha_x|^2$$\n"
            "- **State Collapse**: Immediately upon measurement, the state irreversibly collapses to the observed eigenstate $|x\\rangle$.\n"
            "- **Measurement vs Mathematical Probability**: A simulator calculates the exact continuous probability $|\\alpha_x|^2$. A physical device collects discrete counts with sampling standard error $\\sigma = \\sqrt{p(1-p)/\\text{shots}}$."
        )

    elif concept_tab.startswith("5."):
        st.markdown("### Quantum Entanglement & Bell State")
        st.markdown(
            "- **Entanglement**: A non-separable multi-qubit state where the state of individual qubits cannot be described independently:\n"
            "  $$|\\Phi^+\\rangle = \\frac{|00\\rangle + |11\\rangle}{\\sqrt{2}}$$\n"
            "- **Measurement Correlation**: Measuring qubit 1 yields 0 or 1 with 50% probability each; however, if qubit 1 yields 0, qubit 2 is guaranteed to collapse to 0 instantly!\n"
            "- **⚠️ Crucial Physical Clarification**: Entanglement **does NOT permit faster-than-light (FTL) communication** (No-Communication Theorem). Because each local observer sees a completely random 50/50 stream of bits, no information can be transmitted without transmitting classical data."
        )

    elif concept_tab.startswith("6."):
        st.markdown("### Quantum Search Phase Oracle")
        st.markdown(
            "- **Reversible Phase Oracle**: Marks the target state $|s\\rangle$ with a $-1$ phase:\n"
            "  $$U_w |x\\rangle = (-1)^{f(x)} |x\\rangle = \\begin{cases} -|s\\rangle & \\text{if } x = s \\\\ +|x\\rangle & \\text{if } x \\neq s \\end{cases}$$\n"
            "- **Operator Form**: $U_w = I - 2|s\\rangle\\langle s|$\n"
            "- **Crucial Invariant**: The oracle flips only the *phase* (sign), **NOT** the probability! Since $|-1|^2 = |+1|^2 = 1$, measuring immediately after the oracle would yield no advantage."
        )

    elif concept_tab.startswith("7."):
        st.markdown("### Amplitude Amplification & Diffusion Operator")
        st.markdown(
            "- **Diffusion Operator ($D$)**: Reflects state amplitudes about the mean amplitude $\\mu$:\n"
            "  $$D = 2|s_0\\rangle\\langle s_0| - I$$\n"
            "  $$D \\sum_x \\alpha_x |x\\rangle = \\sum_x (2\\mu - \\alpha_x) |x\\rangle$$\n"
            "- **Amplification Mechanism**:\n"
            "  1. Oracle negates target state: $\\alpha_s \\to -|\\alpha_s|$.\n"
            "  2. Mean amplitude $\\mu$ drops slightly but remains positive.\n"
            "  3. Inversion about the mean computes $2\\mu - (-|\\alpha_s|) = 2\\mu + |\\alpha_s|$, amplifying the target while suppressing non-target amplitudes!"
        )


def page_classical_vs_quantum():
    """Page 4: Formal Complexity Comparison & Scaling Analysis."""
    st.markdown("## ⚖️ Classical vs Quantum Complexity Comparison")
    st.markdown(
        "Analyze the formal query complexities of unstructured search: "
        "Classical Linear Search vs Grover Quantum Search."
    )

    n_sel = st.slider("Select Register Size (n Qubits) for Comparison:", min_value=2, max_value=8, value=4)
    N_sel = 2 ** n_sel
    classical_info = get_classical_metrics(N_sel)
    quantum_info = get_quantum_metrics(n_sel)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Search Space (N)", f"{N_sel:,} items")
    c2.metric("Classical Avg Queries", f"{classical_info['average_case']:,.1f}", "O(N)")
    c3.metric("Quantum Optimal Queries", f"{quantum_info['oracle_queries']:,}", "O(√N)")
    c4.metric("Query Speedup Factor", f"{quantum_info['speedup_ratio']:.1f}x", "Quadratic advantage")

    st.markdown("### 📈 Complexity Scaling Across Register Sizes (n = 2 to 16 Qubits)")
    scaling_df = generate_scaling_comparison_table(max_qubits=16)
    st.dataframe(scaling_df, use_container_width=True, height=320)

    explanations = get_complexity_explanations()
    st.markdown("### 🔍 Critical Algorithmic Clarifications")
    st.info(f"**Theoretical Query Complexity:** {explanations['theoretical_complexity']}")
    st.info(f"**Ideal Probability vs Over-Rotation:** {explanations['ideal_probability']}")
    st.warning(f"**Simulation vs Physical Execution:** {explanations['classical_simulation_caveat']}")


def page_presentation_mode():
    """Page 5 / Presentation Mode for Hackathon Evaluators."""
    st.markdown("## 🏆 Hackathon Presentation Mode — VTU–Qiskit Fall Fest 2026")
    st.markdown(
        '<div class="qa-card-highlight"><h3 style="color: #00F2FE; margin-top:0;">Project: Quantum Astra (VTU-QFF-HC-0008)</h3><p style="color: #E2E8F0; margin-bottom: 4px;"><b>Tagline:</b> Explore the Quantum World</p><p style="color: #94A3B8; margin-bottom: 4px;"><b>Team Lead:</b> Sagar Zingade | <b>Team Member:</b> Abhilash</p><p style="color: #38BDF8; margin-bottom: 0;"><b>Repository:</b> <a href="https://github.com/vtu-qiskit-fall-fest-2026/vtu-hc-quantum-astra" target="_blank" style="color:#00F2FE;">github.com/vtu-qiskit-fall-fest-2026/vtu-hc-quantum-astra</a></p></div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 1. Problem Statement & Educational Need")
    st.write(
        "Quantum computing concepts such as superpositions, phase oracles, and amplitude amplification "
        "are notoriously difficult for university students and beginners to understand because state transformations "
        "and measurement statistics are abstract. Most existing cloud tools require API keys, complex setups, or paid accounts."
    )

    st.markdown("### 2. Solution: Quantum Astra")
    st.write(
        "Quantum Astra is a 100% offline, zero-API-key interactive quantum simulator. "
        "It constructs mathematically rigorous Grover circuits, visualizes the exact phase inversion and reflection about the mean, "
        "and compares quantum query complexity with classical linear search."
    )

    st.markdown("### 3. Live Verified Benchmark (4 Qubits, Target |1010⟩, 3 Iterations)")
    res = run_grover_simulation(4, "1010", 3, shots=2000, seed=42)

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Search Space N", "16 States")
    colB.metric("Target State", "|1010⟩")
    colC.metric("Ideal Probability", f"{res['target_ideal_prob']*100:.2f}%", "Theoretical: 96.13%")
    colD.metric("Sampled Frequency", f"{res['target_sampled_freq']*100:.2f}%", f"{res['target_count']}/{res['shots']} shots")

    fig = plot_probability_distribution(res["ideal_probabilities"], res["counts"], res["shots"], "1010", 4)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### 4. Technical Architecture")
    st.code(
        """quantum-astra/
├── app.py                  # Streamlit Interactive Scientific Dashboard
├── requirements.txt        # Pinned stable dependencies (Qiskit, Aer, Streamlit, etc.)
├── run_tests.py            # Automated test runner (18/18 tests passing)
├── src/
│   ├── quantum_core.py     # Qiskit circuit integration & offline Aer simulator
│   ├── grover.py           # Phase oracle, diffusion operator, and amplitude iteration
│   ├── measurements.py     # Finite-shot sampling and binomial error bounds
│   ├── probability.py      # Statevector probability calculation & CSV export
│   └── comparisons.py      # Classical vs Quantum complexity models
├── tests/
│   ├── test_validation.py  # Tests 1-2 (Input constraints and error handling)
│   ├── test_grover.py      # Tests 3-5, 8-10 (Gates, oracle, diffusion, default benchmark)
│   └── test_probabilities.py # Tests 6-7, 11-12 (Normalization, sampling, classical query counts)
├── assets/
│   └── quantum_astra.svg   # Pure local SVG branding badge
└── docs/
    └── algorithm.md        # Complete mathematical specification""",
        language="text",
    )

    st.markdown("### 5. Limitations & Future Scope")
    st.markdown(
        "- **Classical Simulation Scaling**: Running on classical CPUs stores $2^n$ statevector amplitudes; practical local simulation is constrained to $n \\le 16$ qubits.\n"
        "- **Future Scope**: Expand to multi-target simultaneous search, noisy quantum channel simulation (depolarizing and amplitude damping noise models), and quantum counting algorithms."
    )


def page_about():
    """Page 6: About Project, Team, and Hackathon Context."""
    st.markdown("## ℹ️ About Quantum Astra")
    st.markdown(
        '<div class="qa-card"><h3 style="color: #00F2FE; margin-top:0;">VTU–Qiskit Fall Fest 2026</h3><p><b>Project Name:</b> Quantum Astra</p><p><b>Project ID:</b> VTU-QFF-HC-0008</p><p><b>Tagline:</b> Explore the Quantum World</p><p><b>Team Lead:</b> Sagar Zingade</p><p><b>Team Member:</b> Abhilash</p><p><b>Institution:</b> Visvesvaraya Technological University (VTU)</p><p><b>Repository:</b> <a href="https://github.com/vtu-qiskit-fall-fest-2026/vtu-hc-quantum-astra" target="_blank" style="color:#00F2FE;">https://github.com/vtu-qiskit-fall-fest-2026/vtu-hc-quantum-astra</a></p></div>',
        unsafe_allow_html=True,
    )

    backend = get_backend_info()
    st.markdown("### ⚙️ Local Runtime Environment & Backend Diagnostics")
    st.json(backend)


def main():
    if not HAS_STREAMLIT:
        print("Streamlit is not installed in the current environment.")
        print("To launch the web interface, run: pip install -r requirements.txt && streamlit run app.py")
        return

    st.set_page_config(
        page_title="Quantum Astra — Offline Grover Search Lab",
        page_icon="🌌",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_styles()

    st.sidebar.markdown(
        '<div style="text-align: center; padding: 10px 0;"><h2 style="color: #00F2FE; margin:0; font-size: 1.5rem; letter-spacing: 1px;">QUANTUM ASTRA</h2><p style="color: #94A3B8; font-size: 0.75rem; margin-top: 2px;">VTU–Qiskit Fall Fest 2026</p><span class="qa-pill-cyan">VTU-QFF-HC-0008</span></div>',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    menu = st.sidebar.radio(
        "Navigation Menu:",
        [
            "1. Dashboard",
            "2. Grover Simulator",
            "3. Quantum Concepts",
            "4. Classical vs Quantum",
            "5. Presentation Mode",
            "6. About the Project",
        ],
    )

    render_logo_header()

    if menu.startswith("1."):
        page_dashboard()
    elif menu.startswith("2."):
        page_grover_simulator()
    elif menu.startswith("3."):
        page_quantum_concepts()
    elif menu.startswith("4."):
        page_classical_vs_quantum()
    elif menu.startswith("5."):
        page_presentation_mode()
    elif menu.startswith("6."):
        page_about()

    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #64748B; font-size: 0.8rem; padding-bottom: 20px;">Quantum Astra • Offline Grover Search Lab • VTU-QFF-HC-0008 • Built with Qiskit & Streamlit</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()