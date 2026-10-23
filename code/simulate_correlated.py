"""
====================================================================
simulate_correlated.py

Title:
    Local vs. Correlated Unitary Scrambling: A Comparative Evaluation

Purpose:
    Extends the scrambling model beyond the single-qubit local phase
    rotation Rz(theta) used in simulate_v4.py, adding a genuinely
    two-qubit CORRELATED scrambling operation:

        U_correlated(theta) = exp(-i * theta/2 * Z (x) X)

    implemented natively in Qiskit as the RZX gate (QuantumCircuit.rzx).
    This is the same ZX-type cross-resonance interaction used to
    physically realize entangling gates on real IBM hardware, making
    it a well-motivated two-qubit coherent perturbation rather than an
    arbitrary extension -- and it directly answers this paper's own
    Limitations / Future Work statement about generalizing beyond a
    single local phase rotation.

Why this specific choice matters
---------------------------------
Local scrambling, Rz(theta) on qubit 0 only:
    - Rz(theta) is diagonal in the computational (Z) basis.
    - The depolarizing channel commutes with ANY unitary conjugation
      (it is fully isotropic).
    - Thermal relaxation is diagonal in the Z eigenbasis, so it
      commutes with any Z-axis rotation.
    - Rz(theta) acts on qubit 0 only, i.e. as Rz(theta) (x) I: local.
    Consequence (proved analytically, see simulate_v4.py discussion):
    PURITY and CONCURRENCE are exactly independent of theta, at any
    noise level gamma.

Correlated scrambling, exp(-i*theta/2*Z(x)X) on both qubits:
    - NOT diagonal in the Z(x)Z basis -> does not commute with
      thermal relaxation.
    - NOT a local operator (not of the form U0 (x) U1) -> concurrence
      is not protected by the local-unitary invariance argument.
    Consequence, in the noiseless (ideal, no noise model) limit:
        F(theta) = cos^2(theta/2)   -- IDENTICAL functional form to
                                        the local case
        C(theta) = |cos(theta)|     -- genuinely decreases with theta,
                                        down to 0 at theta = pi/2,
                                        with ZERO environmental noise

    Headline result: fidelity alone cannot distinguish local from
    correlated scrambling (both follow the same cos^2(theta/2)
    curve), but concurrence can. This is verified numerically below
    (self_test_correlated) against the closed-form predictions, then
    demonstrated across the full noise sweep.

Outputs
-------
output/
    correlated_results.csv

    concurrence_local_vs_correlated.png
    purity_local_vs_correlated.png
    fidelity_local_vs_correlated.png
    trace_distance_local_vs_correlated.png

Compatible with
---------------
qiskit >= 1.0
qiskit-aer >= 0.15

====================================================================
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter

from qiskit.quantum_info import (
    DensityMatrix,
    state_fidelity,
    purity,
    concurrence
)

from qiskit_aer import AerSimulator

from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error
)

# ==============================================================
# Logging
# ==============================================================

logging.getLogger("qiskit").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ==============================================================
# Output directory
# ==============================================================

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================
# Simulation parameters (identical to simulate_v4.py, for direct
# comparability with the existing local-scrambling results)
# ==============================================================

THETA = Parameter("θ")

THETA_VALUES = np.linspace(0, np.pi, 41)

NOISE_RATES = [0.00, 0.01, 0.03, 0.05]

T1 = 50e-6
T2 = 30e-6

GATE_TIME_1Q = 50e-9
GATE_TIME_2Q = 300e-9

# ==============================================================
# Ideal Bell state
# ==============================================================

def create_ideal_density_matrix():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return DensityMatrix.from_instruction(qc)


IDEAL_STATE = create_ideal_density_matrix()

# ==============================================================
# Trace distance (identical implementation to simulate_v4.py)
# ==============================================================

def trace_distance(rho, sigma):
    delta = rho.data - sigma.data
    singular_values = np.linalg.svd(delta, compute_uv=False)
    return 0.5 * np.sum(singular_values)

# ==============================================================
# Composite Noise Model
# (same as simulate_v4.py, extended to also cover the "rzx" gate
# under the same two-qubit composite channel as "cx")
# ==============================================================

def create_noise_model(gamma):

    dep_error_1q = depolarizing_error(gamma, 1)
    thermal_error_1q = thermal_relaxation_error(T1, T2, GATE_TIME_1Q)
    combined_error_1q = dep_error_1q.compose(thermal_error_1q)

    dep_error_2q = depolarizing_error(gamma, 2)
    thermal_error_2q = thermal_relaxation_error(T1, T2, GATE_TIME_2Q).tensor(
        thermal_relaxation_error(T1, T2, GATE_TIME_2Q)
    )
    combined_error_2q = dep_error_2q.compose(thermal_error_2q)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(combined_error_1q, ["h", "rz"])
    noise_model.add_all_qubit_quantum_error(combined_error_2q, ["cx", "rzx"])

    return noise_model

# ==============================================================
# Circuit builders
# ==============================================================

def create_local_circuit():
    """Bell state + local phase scrambling Rz(theta) on qubit 0."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(THETA, 0)
    qc.save_density_matrix()
    return qc


def create_correlated_circuit():
    """Bell state + correlated two-qubit scrambling exp(-i*theta/2*Z(x)X)."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rzx(THETA, 0, 1)
    qc.save_density_matrix()
    return qc

# ==============================================================
# Self-tests
# ==============================================================

def self_test():
    """
    Verifies, using a genuinely noiseless simulator (no noise model
    attached at all -- see the same reasoning as simulate_v4.py's
    self-test), that:

      (a) both circuits reduce to the ideal Bell state at theta=0.
      (b) across the full theta sweep, the CORRELATED circuit's
          simulated fidelity and concurrence match the closed-form
          predictions derived above:
              F(theta) = cos^2(theta/2)
              C(theta) = |cos(theta)|
          to numerical precision.

    This simultaneously validates the analytic derivation and the
    circuit implementation. Aborts the run if any check fails.
    """

    logging.info("Running self-test...\n")

    ideal_simulator = AerSimulator(method="density_matrix")
    tolerance = 1e-6

    # ---- (a) theta = 0 sanity check, both circuits ----

    for label, builder in [
        ("local", create_local_circuit),
        ("correlated", create_correlated_circuit)
    ]:

        qc = builder()
        bound = qc.assign_parameters({THETA: 0.0})
        compiled = transpile(bound, ideal_simulator, optimization_level=1)
        result = ideal_simulator.run(compiled).result()

        rho = result.data(0)["density_matrix"]
        if not isinstance(rho, DensityMatrix):
            rho = DensityMatrix(rho)

        fidelity_value = state_fidelity(IDEAL_STATE, rho)

        assert abs(fidelity_value - 1.0) < tolerance, \
            f"Self-test FAILED ({label}, theta=0): fidelity={fidelity_value}"

    logging.info("  [PASS] Both circuits reduce to the ideal Bell state at theta=0.\n")

    # ---- (b) closed-form check across full theta sweep, correlated case ----

    qc = create_correlated_circuit()
    compiled = transpile(qc, ideal_simulator, optimization_level=1)

    max_fidelity_error = 0.0
    max_concurrence_error = 0.0

    for theta in THETA_VALUES:

        bound = compiled.assign_parameters({THETA: theta})
        result = ideal_simulator.run(bound).result()

        rho = result.data(0)["density_matrix"]
        if not isinstance(rho, DensityMatrix):
            rho = DensityMatrix(rho)

        fidelity_sim = state_fidelity(IDEAL_STATE, rho)
        concurrence_sim = concurrence(rho)

        fidelity_theory = np.cos(theta / 2) ** 2
        concurrence_theory = abs(np.cos(theta))

        max_fidelity_error = max(
            max_fidelity_error,
            abs(fidelity_sim - fidelity_theory)
        )

        max_concurrence_error = max(
            max_concurrence_error,
            abs(concurrence_sim - concurrence_theory)
        )

    assert max_fidelity_error < tolerance, \
        f"Self-test FAILED: fidelity vs cos^2(theta/2) max error={max_fidelity_error}"

    assert max_concurrence_error < tolerance, \
        f"Self-test FAILED: concurrence vs |cos(theta)| max error={max_concurrence_error}"

    logging.info(
        "  [PASS] Correlated-case fidelity matches cos^2(theta/2) "
        f"(max error={max_fidelity_error:.2e})\n"
        "  [PASS] Correlated-case concurrence matches |cos(theta)| "
        f"(max error={max_concurrence_error:.2e})\n"
    )

    logging.info("Self-test PASSED.\n")

# ==============================================================
# Simulation engine (runs one case: 'local' or 'correlated')
# ==============================================================

def simulate_case(case_label, circuit_builder):

    logging.info(f"\nSimulating case: {case_label}\n")

    rows = []

    base_circuit = circuit_builder()

    for gamma in NOISE_RATES:

        logging.info(f"  Running γ = {gamma:.2f}")

        simulator = AerSimulator(
            method="density_matrix",
            noise_model=create_noise_model(gamma)
        )

        compiled = transpile(base_circuit, simulator, optimization_level=1)

        for theta in THETA_VALUES:

            bound_circuit = compiled.assign_parameters({THETA: theta})
            result = simulator.run(bound_circuit).result()

            rho = result.data(0)["density_matrix"]
            if not isinstance(rho, DensityMatrix):
                rho = DensityMatrix(rho)

            fidelity_value = state_fidelity(IDEAL_STATE, rho)
            purity_value = np.real_if_close(purity(rho)).item()
            concurrence_value = concurrence(rho)
            trace_value = trace_distance(rho, IDEAL_STATE)

            rows.append({
                "case": case_label,
                "gamma": gamma,
                "theta": theta,
                "fidelity": fidelity_value,
                "purity": purity_value,
                "concurrence": concurrence_value,
                "trace_distance": trace_value
            })

    return rows

# ==============================================================
# Comparison plotting: local vs correlated, one 2x2 grid per metric
# (one subplot per noise level)
# ==============================================================

def plot_comparison_grid(df, metric_col, ylabel, title, filename):

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), dpi=300)

    for ax, gamma in zip(axes.flat, NOISE_RATES):

        for case_label, colour in [("local", "#1f77b4"), ("correlated", "#d62728")]:

            subset = df[(df["case"] == case_label) & (df["gamma"] == gamma)]
            subset = subset.sort_values("theta")

            ax.plot(
                subset["theta"],
                subset[metric_col],
                linewidth=2.0,
                color=colour,
                label=case_label
            )

        ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        ax.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
        ax.set_xlabel(r"Scrambling angle $\theta$ (radians)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"$\\gamma={gamma:.2f}$", fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.grid(linestyle=":", alpha=0.35)
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()

# ==============================================================
# Main
# ==============================================================

def main():

    logging.info("\n=========================================")
    logging.info(" Local vs. Correlated Scrambling Comparison")
    logging.info("=========================================\n")

    self_test()

    rows = []
    rows += simulate_case("local", create_local_circuit)
    rows += simulate_case("correlated", create_correlated_circuit)

    df = pd.DataFrame(rows)

    csv_path = os.path.join(OUTPUT_DIR, "correlated_results.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"\nResults saved: {csv_path}")

    plot_comparison_grid(
        df, "fidelity", "Fidelity",
        "Fidelity: Local vs. Correlated Scrambling",
        "fidelity_local_vs_correlated.png"
    )

    plot_comparison_grid(
        df, "purity", "Purity",
        "Purity: Local vs. Correlated Scrambling",
        "purity_local_vs_correlated.png"
    )

    plot_comparison_grid(
        df, "concurrence", "Concurrence",
        "Concurrence: Local vs. Correlated Scrambling",
        "concurrence_local_vs_correlated.png"
    )

    plot_comparison_grid(
        df, "trace_distance", "Trace Distance",
        "Trace Distance: Local vs. Correlated Scrambling",
        "trace_distance_local_vs_correlated.png"
    )

    logging.info("\n==============================")
    logging.info("Summary (mean over theta, at each gamma)")
    logging.info("==============================")

    summary = df.groupby(["case", "gamma"]).mean(numeric_only=True)
    logging.info(f"\n{summary[['fidelity', 'purity', 'concurrence', 'trace_distance']]}\n")

    logging.info("Figures generated:")
    logging.info("  fidelity_local_vs_correlated.png")
    logging.info("  purity_local_vs_correlated.png")
    logging.info("  concurrence_local_vs_correlated.png")
    logging.info("  trace_distance_local_vs_correlated.png")

    logging.info("\nSimulation completed successfully.")


if __name__ == "__main__":
    main()
