"""
====================================================================
simulate_confound_control.py

Title:
    Local vs. Correlated Scrambling: Gate-Duration-Confound Control

Purpose (Reviewer Comment #8)
------------------------------
simulate_correlated.py compares:

    local circuit:       ... Rz(theta) attack gate,  duration = GATE_TIME_1Q (50 ns)
    correlated circuit:  ... RZX(theta) attack gate,  duration = GATE_TIME_2Q (300 ns)

Because the correlated attack gate is both (a) a different unitary
STRUCTURE (two-qubit ZX rotation vs. one-qubit Z rotation) and (b) a
longer-DURATION gate under the thermal-relaxation model, any observed
difference between the two cases -- e.g. in purity -- is confounded:
you cannot tell how much of it comes from the change in operator
structure vs. the change in gate duration alone.

This script adds a THIRD case that isolates the two effects:

    local               : Rz(theta),  1-qubit dep., duration = GATE_TIME_1Q
    correlated_realistic: RZX(theta), 2-qubit dep., duration = GATE_TIME_2Q  (as before)
    correlated_matched  : RZX(theta), 2-qubit dep., duration = GATE_TIME_1Q  (NEW control)

"correlated_matched" keeps the RZX gate's two-qubit depolarizing
channel exactly as before (it is still, physically, a two-qubit
operation, regardless of how long it is assumed to take), but forces
its THERMAL-RELAXATION duration down to GATE_TIME_1Q, i.e. the same
duration used for the local circuit's Rz attack gate. This isolates
"is it the correlated structure, or just the longer duration, that
drives the observed differences?" -- exactly the control the review
asked for.

Because purity/fidelity/trace-distance can all be duration-sensitive
via thermal relaxation, but concurrence's separation between local
and correlated scrambling is already proved EXACTLY in the noiseless
limit (Proposition in Section III-B of the paper, independent of any
gate-duration assumption), the concurrence result should be
essentially unaffected by this control. Purity is expected to move
noticeably, since duration directly affects thermal-relaxation
exposure. Both are checked explicitly below via a decomposition
table.

Outputs
-------
output/
    confound_control_results.csv        (raw per-theta, per-gamma, per-case rows)
    confound_decomposition.csv          (duration vs. structure decomposition table)

    fidelity_confound_control.png
    purity_confound_control.png
    concurrence_confound_control.png
    trace_distance_confound_control.png
    headline_fidelity_vs_concurrence.png   (combined "headline" figure, Reviewer #17)

Compatible with
---------------
qiskit >= 1.0
qiskit-aer >= 0.15

NOTE: This script has not been executed in the authoring environment
(no network / Qiskit access there). It closely mirrors the structure
and API calls of simulate_v4.py and simulate_correlated.py, which you
have already run successfully -- only the noise-model construction
and the case list are new. Please run self_test() (it runs
automatically at the top of main()) before trusting the results.
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
# Output directory (shared with the other scripts; filenames below
# are distinct from simulate_v4.py / simulate_correlated.py outputs)
# ==============================================================

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================
# Simulation parameters (identical to simulate_v4.py / simulate_correlated.py,
# for direct comparability)
# ==============================================================

THETA = Parameter("θ")
THETA_VALUES = np.linspace(0, np.pi, 41)
NOISE_RATES = [0.00, 0.01, 0.03, 0.05]

T1 = 50e-6
T2 = 30e-6

GATE_TIME_1Q = 50e-9    # local attack gate (Rz) duration
GATE_TIME_2Q = 300e-9   # realistic two-qubit gate duration (CX and, in the
                         # "correlated_realistic" case, RZX)

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
# Trace distance (identical implementation to the other scripts)
# ==============================================================

def trace_distance(rho, sigma):
    delta = rho.data - sigma.data
    singular_values = np.linalg.svd(delta, compute_uv=False)
    return 0.5 * np.sum(singular_values)

# ==============================================================
# Composite Noise Model, with an explicit duration control on the
# RZX ("attack") gate, independent of the CX ("state preparation")
# gate's duration.
# ==============================================================

def create_noise_model(gamma, matched_duration=False):
    """
    Parameters
    ----------
    gamma : float
        Depolarizing probability (as elsewhere in the paper).
    matched_duration : bool
        If False (default): RZX uses GATE_TIME_2Q (300 ns), the
        physically realistic assumption used everywhere else in the
        paper -- this is the "correlated_realistic" case.
        If True: RZX uses GATE_TIME_1Q (50 ns) for its
        thermal-relaxation duration only -- an artificial control
        matching the local circuit's attack-gate duration. The
        two-qubit depolarizing channel dimension (n=2) is left
        unchanged in both cases, since RZX genuinely acts on two
        qubits regardless of the assumed duration; only the
        thermal-relaxation exposure is varied. This is the
        "correlated_matched" case (Reviewer Comment #8).

    The CX gate (used only for Bell-state preparation, never as the
    attack gate) always uses GATE_TIME_2Q, in both cases -- it is not
    part of the scrambling operation under test and should not be
    altered by this control.
    """

    # ---- Single-qubit gate error (H, and Rz when used as the local attack) ----

    dep_error_1q = depolarizing_error(gamma, 1)
    thermal_error_1q = thermal_relaxation_error(T1, T2, GATE_TIME_1Q)
    combined_error_1q = dep_error_1q.compose(thermal_error_1q)

    # ---- Two-qubit depolarizing channel (shared dimension for CX and RZX) ----

    dep_error_2q = depolarizing_error(gamma, 2)

    # ---- CX (state preparation): always the realistic 300 ns duration ----

    thermal_error_2q_cx = thermal_relaxation_error(T1, T2, GATE_TIME_2Q).tensor(
        thermal_relaxation_error(T1, T2, GATE_TIME_2Q)
    )
    combined_error_cx = dep_error_2q.compose(thermal_error_2q_cx)

    # ---- RZX (correlated attack gate): duration is the controlled variable ----

    attack_gate_time = GATE_TIME_1Q if matched_duration else GATE_TIME_2Q
    thermal_error_2q_attack = thermal_relaxation_error(T1, T2, attack_gate_time).tensor(
        thermal_relaxation_error(T1, T2, attack_gate_time)
    )
    combined_error_rzx = dep_error_2q.compose(thermal_error_2q_attack)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(combined_error_1q, ["h", "rz"])
    noise_model.add_all_qubit_quantum_error(combined_error_cx, ["cx"])
    noise_model.add_all_qubit_quantum_error(combined_error_rzx, ["rzx"])

    return noise_model

# ==============================================================
# Circuit builders (identical to simulate_correlated.py)
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
# Self-test
# ==============================================================

def self_test():
    """
    Two checks, run before any real sweep:

    (a) Both circuits, with NO noise model attached at all, reduce
        to the ideal Bell state at theta=0 (same check as
        simulate_correlated.py -- verifies circuit construction).

    (b) Duration-ordering sanity check: at a fixed nonzero gamma and
        a fixed test angle, the "matched" (50 ns) RZX noise channel
        must produce a HIGHER purity than the "realistic" (300 ns)
        RZX noise channel, since less thermal-relaxation time has
        elapsed. This is a direct, cheap check that
        create_noise_model(..., matched_duration=True) is actually
        using a shorter duration for RZX and not accidentally
        falling back to GATE_TIME_2Q.

    Aborts the run if either check fails.
    """

    logging.info("Running self-test...\n")

    ideal_simulator = AerSimulator(method="density_matrix")
    tolerance = 1e-6

    # ---- (a) theta = 0 sanity check, both circuits, no noise model ----

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
            f"Self-test FAILED ({label}, theta=0, no noise): fidelity={fidelity_value}"

    logging.info("  [PASS] Both circuits reduce to the ideal Bell state at theta=0 (no noise).\n")

    # ---- (b) duration-ordering check on the noise model itself ----

    test_gamma = 0.05
    test_theta = np.pi / 2

    purities = {}
    for label, matched in [("correlated_realistic", False), ("correlated_matched", True)]:
        simulator = AerSimulator(
            method="density_matrix",
            noise_model=create_noise_model(test_gamma, matched_duration=matched)
        )
        qc = create_correlated_circuit()
        bound = qc.assign_parameters({THETA: test_theta})
        compiled = transpile(bound, simulator, optimization_level=1)
        result = simulator.run(compiled).result()

        rho = result.data(0)["density_matrix"]
        if not isinstance(rho, DensityMatrix):
            rho = DensityMatrix(rho)

        purities[label] = np.real_if_close(purity(rho)).item()

    assert purities["correlated_matched"] > purities["correlated_realistic"], (
        "Self-test FAILED (duration-ordering check): expected the duration-matched "
        f"(50 ns) RZX channel to yield HIGHER purity than the realistic (300 ns) "
        f"channel, but got matched={purities['correlated_matched']:.6f} <= "
        f"realistic={purities['correlated_realistic']:.6f}. "
        "This suggests matched_duration is not actually reducing the RZX gate's "
        "thermal-relaxation duration -- check create_noise_model()."
    )

    logging.info(
        "  [PASS] Duration-ordering check: purity(matched, 50ns)="
        f"{purities['correlated_matched']:.6f} > "
        f"purity(realistic, 300ns)={purities['correlated_realistic']:.6f} "
        f"at gamma={test_gamma}, theta=pi/2.\n"
    )

    logging.info("Self-test PASSED.\n")

# ==============================================================
# Simulation engine
# ==============================================================

def simulate_case(case_label, circuit_builder, matched_duration):

    logging.info(f"\nSimulating case: {case_label} (matched_duration={matched_duration})\n")

    rows = []
    base_circuit = circuit_builder()

    for gamma in NOISE_RATES:

        logging.info(f"  Running γ = {gamma:.2f}")

        simulator = AerSimulator(
            method="density_matrix",
            noise_model=create_noise_model(gamma, matched_duration=matched_duration)
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
# Decomposition: how much of the local-vs-correlated gap is
# attributable to gate duration vs. scrambling structure?
# ==============================================================

def build_decomposition_table(df):
    """
    For each gamma and each metric, computes:

        mean_local, mean_correlated_matched, mean_correlated_realistic
            (means taken over the full theta sweep)

        delta_structure = mean_correlated_matched - mean_local
            (effect of changing Rz -> RZX, HOLDING duration fixed at 50 ns)

        delta_duration = mean_correlated_realistic - mean_correlated_matched
            (effect of changing duration 50ns -> 300ns, HOLDING the RZX
             structure fixed)

        delta_total = mean_correlated_realistic - mean_local
            (= delta_structure + delta_duration, by construction; reported
             as a sanity check)

    Also computes the same decomposition at the single angle theta=pi/2,
    which is where the noiseless correlated-scrambling concurrence
    reaches its minimum (C_c(pi/2)=0) and is therefore the most
    diagnostic single point for the concurrence result specifically.
    """

    metrics = ["fidelity", "purity", "concurrence", "trace_distance"]
    rows = []

    def _decompose(subset, gamma, theta_label):
        means = subset.groupby("case")[metrics].mean()
        for metric in metrics:
            mean_local = means.loc["local", metric]
            mean_matched = means.loc["correlated_matched", metric]
            mean_realistic = means.loc["correlated_realistic", metric]

            rows.append({
                "gamma": gamma,
                "theta_slice": theta_label,
                "metric": metric,
                "mean_local": mean_local,
                "mean_correlated_matched": mean_matched,
                "mean_correlated_realistic": mean_realistic,
                "delta_structure": mean_matched - mean_local,
                "delta_duration": mean_realistic - mean_matched,
                "delta_total": mean_realistic - mean_local,
            })

    for gamma in NOISE_RATES:
        subset_all = df[df["gamma"] == gamma]
        _decompose(subset_all, gamma, "mean_over_theta")

        # closest sampled theta to pi/2
        nearest_theta = THETA_VALUES[np.argmin(np.abs(THETA_VALUES - np.pi / 2))]
        subset_pi2 = subset_all[np.isclose(subset_all["theta"], nearest_theta)]
        _decompose(subset_pi2, gamma, f"theta={nearest_theta:.4f} (~pi/2)")

    return pd.DataFrame(rows)

# ==============================================================
# Plotting: three-case comparison grid (one 2x2 grid per metric,
# one subplot per noise level)
# ==============================================================

CASES = [
    ("local", "#1f77b4", "-"),
    ("correlated_realistic", "#d62728", "-"),
    ("correlated_matched", "#2ca02c", "--"),
]


def plot_comparison_grid(df, metric_col, ylabel, title, filename):

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), dpi=300)

    for ax, gamma in zip(axes.flat, NOISE_RATES):

        for case_label, colour, linestyle in CASES:

            subset = df[(df["case"] == case_label) & (df["gamma"] == gamma)]
            subset = subset.sort_values("theta")

            ax.plot(
                subset["theta"],
                subset[metric_col],
                linewidth=2.0,
                linestyle=linestyle,
                color=colour,
                label=case_label
            )

        ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
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


def plot_headline_figure(df):
    """
    Combined "headline" figure (Reviewer Comment #17): fidelity on
    top, concurrence on bottom, local vs. correlated_realistic, with
    the closed-form noiseless predictions overlaid as dashed black
    reference curves. Intended as the single figure that communicates
    the paper's central result (F_l(theta) = F_c(theta) but
    C_l(theta) != C_c(theta)) at a glance.
    """

    fig, axes = plt.subplots(2, 1, figsize=(7, 8.5), dpi=300, sharex=True)

    theory_theta = np.linspace(0, np.pi, 200)
    fidelity_theory = np.cos(theory_theta / 2) ** 2
    concurrence_theory_correlated = np.abs(np.cos(theory_theta))

    colours = {0.00: "#9ecae1", 0.01: "#6baed6", 0.03: "#3182bd", 0.05: "#08519c"}
    colours_c = {0.00: "#fdae6b", 0.01: "#fd8d3c", 0.03: "#e6550d", 0.05: "#a63603"}

    ax_f, ax_c = axes

    for gamma in NOISE_RATES:
        for case_label, ax, colour_map, style in [
            ("local", ax_f, colours, "-"),
            ("correlated_realistic", ax_f, colours_c, "--"),
        ]:
            subset = df[(df["case"] == case_label) & (df["gamma"] == gamma)].sort_values("theta")
            ax.plot(subset["theta"], subset["fidelity"], style, color=colour_map[gamma],
                     linewidth=1.6,
                     label=f"{case_label}, $\\gamma={gamma:.2f}$" if gamma in (0.00, 0.05) else None)

    ax_f.plot(theory_theta, fidelity_theory, "k:", linewidth=2.0, label=r"theory: $\cos^2(\theta/2)$")
    ax_f.set_ylabel("Fidelity")
    ax_f.set_title(r"$F_l(\theta)$ vs. $F_c(\theta)$ -- nearly indistinguishable")
    ax_f.set_ylim(0, 1.02)
    ax_f.grid(linestyle=":", alpha=0.35)
    ax_f.legend(fontsize=7, ncol=2)

    for gamma in NOISE_RATES:
        for case_label, ax, colour_map, style in [
            ("local", ax_c, colours, "-"),
            ("correlated_realistic", ax_c, colours_c, "--"),
        ]:
            subset = df[(df["case"] == case_label) & (df["gamma"] == gamma)].sort_values("theta")
            ax.plot(subset["theta"], subset["concurrence"], style, color=colour_map[gamma],
                     linewidth=1.6,
                     label=f"{case_label}, $\\gamma={gamma:.2f}$" if gamma in (0.00, 0.05) else None)

    ax_c.plot(theory_theta, concurrence_theory_correlated, "k:", linewidth=2.0,
              label=r"theory: $|\cos\theta|$ (correlated)")
    ax_c.set_xlabel(r"Scrambling angle $\theta$ (radians)")
    ax_c.set_ylabel("Concurrence")
    ax_c.set_title(r"$C_l(\theta)$ vs. $C_c(\theta)$ -- sharply separated")
    ax_c.set_ylim(0, 1.02)
    ax_c.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    ax_c.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    ax_c.grid(linestyle=":", alpha=0.35)
    ax_c.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "headline_fidelity_vs_concurrence.png"), dpi=300)
    plt.close()

# ==============================================================
# Main
# ==============================================================

def main():

    logging.info("\n=========================================")
    logging.info(" Local vs. Correlated Scrambling: Gate-Duration-Confound Control")
    logging.info("=========================================\n")

    self_test()

    rows = []
    rows += simulate_case("local", create_local_circuit, matched_duration=False)
    rows += simulate_case("correlated_realistic", create_correlated_circuit, matched_duration=False)
    rows += simulate_case("correlated_matched", create_correlated_circuit, matched_duration=True)

    df = pd.DataFrame(rows)

    csv_path = os.path.join(OUTPUT_DIR, "confound_control_results.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"\nResults saved: {csv_path}")

    decomposition_df = build_decomposition_table(df)
    decomposition_csv_path = os.path.join(OUTPUT_DIR, "confound_decomposition.csv")
    decomposition_df.to_csv(decomposition_csv_path, index=False)
    logging.info(f"Decomposition table saved: {decomposition_csv_path}")

    logging.info("\n==============================")
    logging.info("Decomposition (mean over theta), gamma=0.00 and gamma=0.05:")
    logging.info("==============================")
    preview = decomposition_df[decomposition_df["theta_slice"] == "mean_over_theta"]
    logging.info(f"\n{preview[['gamma', 'metric', 'mean_local', 'mean_correlated_matched', 'mean_correlated_realistic', 'delta_structure', 'delta_duration']].to_string(index=False)}\n")

    plot_comparison_grid(df, "fidelity", "Fidelity",
                          "Fidelity: Local vs. Correlated (Realistic vs. Duration-Matched)",
                          "fidelity_confound_control.png")

    plot_comparison_grid(df, "purity", "Purity",
                          "Purity: Local vs. Correlated (Realistic vs. Duration-Matched)",
                          "purity_confound_control.png")

    plot_comparison_grid(df, "concurrence", "Concurrence",
                          "Concurrence: Local vs. Correlated (Realistic vs. Duration-Matched)",
                          "concurrence_confound_control.png")

    plot_comparison_grid(df, "trace_distance", "Trace Distance",
                          "Trace Distance: Local vs. Correlated (Realistic vs. Duration-Matched)",
                          "trace_distance_confound_control.png")

    plot_headline_figure(df)

    logging.info("Figures generated:")
    logging.info("  fidelity_confound_control.png")
    logging.info("  purity_confound_control.png")
    logging.info("  concurrence_confound_control.png")
    logging.info("  trace_distance_confound_control.png")
    logging.info("  headline_fidelity_vs_concurrence.png")

    logging.info("\nSimulation completed successfully.")


if __name__ == "__main__":
    main()
