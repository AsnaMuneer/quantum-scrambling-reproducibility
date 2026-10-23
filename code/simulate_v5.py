"""
====================================================================
simulate_v5.py
Title:
    Bell-State Robustness Evaluation under Unitary Scrambling
    and Composite NISQ Noise (Native-Gate Comparison: CX vs. CZ)

Usage
-----
    python3 simulate_v5.py cx     # entangling gate CX (matches simulate_v4.py)
    python3 simulate_v5.py cz     # entangling gate CZ (hardware-native on ibm_marrakesh)

With no argument the script runs CX. Each gate writes its own files
into output/, so the two runs can be executed in separate Terminal
windows at the same time.

Changes from simulate_v4.py
---------------------------
1. The entangling gate is selected at run time (cx or cz).
2. The Bell-state preparation is built directly with the chosen
   native gate. For CZ, CX is written as H(1) . CZ . H(1), so the
   circuit contains CZ and no CX, without any basis translation.
   This avoids Qiskit's basis-translation errors with
   save_density_matrix.
3. The two-qubit composite noise channel is attached to the chosen
   gate. The single-qubit channel is attached to h and rz in both
   runs.
4. A check confirms that the transpiled circuit contains the chosen
   two-qubit gate and no other two-qubit gate.
5. Output filenames include the gate name, so CX and CZ results
   never overwrite each other.

Caveat for the comparison
-------------------------
The CZ-based preparation contains two extra single-qubit Hadamards
on qubit 1 (from the H(1) . CZ . H(1) identity). Those gates also
receive single-qubit noise, so the CZ run differs from the CX run in
single-qubit gate count as well as in the entangling gate. State
this in the paper, or treat the CX/CZ difference as an upper bound
on the native-gate effect.

Outputs
-------
output/
    fidelity_results_<gate>.csv
    fidelity_plot_<gate>.png
    purity_plot_<gate>.png
    concurrence_plot_<gate>.png
    trace_distance_plot_<gate>.png
    summary_vs_noise_<gate>.png

where <gate> is "cx" or "cz".

Compatible with
---------------
qiskit >= 1.0
qiskit-aer >= 0.15
====================================================================
"""

# ==============================================================
# Imports
# ==============================================================

import os
import sys
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

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)

# ==============================================================
# Gate selection (command line, default "cx")
# ==============================================================

VALID_GATES = ("cx", "cz")

if len(sys.argv) > 1:
    TWO_QUBIT_GATE = sys.argv[1].lower()
else:
    TWO_QUBIT_GATE = "cx"

if TWO_QUBIT_GATE not in VALID_GATES:
    raise ValueError(
        f"Gate must be one of {VALID_GATES}, got {TWO_QUBIT_GATE!r}. "
        "Usage: python3 simulate_v5.py cx    or    python3 simulate_v5.py cz"
    )

GATE_SUFFIX = "_" + TWO_QUBIT_GATE

# ==============================================================
# Output directory
# ==============================================================

OUTPUT_DIR = "output"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ==============================================================
# Simulation parameters
# ==============================================================

THETA = Parameter("θ")

THETA_VALUES = np.linspace(
    0,
    np.pi,
    41
)

NOISE_RATES = [
    0.00,
    0.01,
    0.03,
    0.05
]

# IBM-inspired coherence values

T1 = 50e-6
T2 = 30e-6

# Single-qubit gate duration (H, Rz)
GATE_TIME_1Q = 50e-9

# Two-qubit gate duration. CX and CZ are both several times longer
# than single-qubit gates on real superconducting NISQ hardware, so
# they accumulate proportionally more thermal-relaxation error per
# operation. This value is representative, not device-specific --
# see paper Section III-B for the same caveat applied to T1/T2.
GATE_TIME_2Q = 300e-9

# ==============================================================
# Bell-state preparation (native gate selected by TWO_QUBIT_GATE)
# ==============================================================

def build_bell_prep(qc):
    """
    Appends Bell-state preparation |Phi+> on qubits 0 and 1.

    cx: H(0), CX(0,1)
    cz: H(0), H(1), CZ(0,1), H(1)

    The cz form uses the identity CX(0,1) = H(1) CZ(0,1) H(1), so the
    circuit contains CZ as its only entangling gate. Both forms
    prepare the same ideal Bell state, which the noiseless self-test
    checks.
    """
    qc.h(0)

    if TWO_QUBIT_GATE == "cz":
        qc.h(1)
        qc.cz(0, 1)
        qc.h(1)
    else:
        qc.cx(0, 1)

# ==============================================================
# Bell-state circuit
# ==============================================================

def create_parameterized_bell():

    """
    Creates

        |Phi+>  (with the selected native entangling gate)

    followed by a parameterized

        Rz(theta)

    scrambling attack on qubit 0. The attack is single-qubit and is
    identical in both runs.
    """

    qc = QuantumCircuit(2)

    build_bell_prep(qc)

    qc.rz(
        THETA,
        0
    )

    qc.save_density_matrix()

    return qc

# ==============================================================
# Ideal Bell state
# ==============================================================

def create_ideal_density_matrix():
    """
    The ideal target state is the Bell state |Phi+>. It does not
    depend on the gate choice, so it is always built with CX as the
    reference.
    """

    qc = QuantumCircuit(2)

    qc.h(0)
    qc.cx(0, 1)

    return DensityMatrix.from_instruction(
        qc
    )


IDEAL_STATE = create_ideal_density_matrix()

# ==============================================================
# Transpilation (backend default basis; no custom basis)
# ==============================================================

def transpile_for_backend(circuit, backend):
    """
    Transpiles to the backend's own default basis. Aer supports both
    CX and CZ natively, so no basis override is needed. The gate used
    is fixed by build_bell_prep(), and check_transpiled_gates()
    confirms it after transpilation.
    """
    return transpile(
        circuit,
        backend,
        optimization_level=1
    )


def check_transpiled_gates(compiled, label):
    """
    Confirms that the compiled circuit contains the selected
    two-qubit gate and no other two-qubit gate. Printed at start-up
    so the gate actually used is recorded, not assumed.
    """
    ops = compiled.count_ops()

    two_qubit_ops = {
        name: n for name, n in ops.items()
        if name in ("cx", "cz", "ecr", "rzx", "swap")
    }

    if TWO_QUBIT_GATE not in two_qubit_ops:
        raise RuntimeError(
            f"[{label}] selected gate '{TWO_QUBIT_GATE}' does not appear in the "
            f"transpiled circuit (ops: {dict(ops)})."
        )

    other = [name for name in two_qubit_ops if name != TWO_QUBIT_GATE]

    if other:
        raise RuntimeError(
            f"[{label}] unexpected two-qubit gate(s) {other} in the transpiled "
            f"circuit (ops: {dict(ops)}). The run would not use only {TWO_QUBIT_GATE}."
        )

    logging.info(f"  [{label}] transpiled gate counts: {dict(ops)}")

# ==============================================================
# Composite Noise Model
# ==============================================================

def create_noise_model(gamma):

    """
    Composite channel

        Depolarizing + Thermal relaxation

    applied to single-qubit gates (h, rz) and to the selected
    two-qubit entangling gate (cx or cz).

    Composition order:
        For both the 1-qubit and 2-qubit channels, the depolarizing
        error is composed first and the thermal-relaxation error
        second (dep_error.compose(thermal_error)). This represents
        gate-calibration imperfection occurring during the
        operation, immediately followed by decoherence accumulated
        over the gate's execution time. QuantumError.compose() is
        not generally commutative for two channels that do not
        commute, so this ordering is a stated modelling choice, not
        a physically unique one.

    Thermal relaxation population:
        thermal_relaxation_error() is called with its default
        excited_state_population=0, i.e. each qubit is assumed to
        relax fully toward the ground state (no residual thermal
        excitation). This is a modelling assumption and should be
        stated alongside T1/T2/gate-time values in the paper.
    """

    # ---- Single-qubit gate error (h, rz) ----

    dep_error_1q = depolarizing_error(
        gamma,
        1
    )

    thermal_error_1q = thermal_relaxation_error(
        T1,
        T2,
        GATE_TIME_1Q
    )

    combined_error_1q = dep_error_1q.compose(
        thermal_error_1q
    )

    # ---- Two-qubit gate error (cx or cz) ----
    # Two-qubit depolarizing channel, paired with a two-qubit
    # thermal-relaxation channel built as the tensor product of
    # independent single-qubit relaxation on each of the two
    # qubits involved in the entangling gate.

    dep_error_2q = depolarizing_error(
        gamma,
        2
    )

    thermal_error_2q = thermal_relaxation_error(
        T1,
        T2,
        GATE_TIME_2Q
    ).tensor(
        thermal_relaxation_error(
            T1,
            T2,
            GATE_TIME_2Q
        )
    )

    combined_error_2q = dep_error_2q.compose(
        thermal_error_2q
    )

    noise_model = NoiseModel()

    noise_model.add_all_qubit_quantum_error(
        combined_error_1q,
        [
            "h",
            "rz"
        ]
    )

    noise_model.add_all_qubit_quantum_error(
        combined_error_2q,
        [
            TWO_QUBIT_GATE
        ]
    )

    return noise_model

# ==============================================================
# Trace distance
# ==============================================================

def trace_distance(rho, sigma):
    """
    Computes

        0.5 ||rho - sigma||

    using the singular values of the density-matrix difference.
    """

    delta = rho.data - sigma.data

    singular_values = np.linalg.svd(
        delta,
        compute_uv=False
    )

    return 0.5 * np.sum(
        singular_values
    )

# ==============================================================
# Correctness self-test
# ==============================================================

def self_test():
    """
    Two-part correctness sanity check.

    Part A -- noiseless check, run after transpilation for the
    selected gate:
        At theta=0, the Bell-state circuit built with the selected
        gate must reproduce the ideal Bell state to numerical
        precision. A wrong decomposition (for example a CZ form
        that does not equal CX up to the H conjugation) fails here.

    Part B -- reference printout at gamma=0.00 (not asserted):
        Thermal relaxation is not tied to gamma, so gamma=0.00 still
        carries decoherence from finite gate durations. Printed for
        reference only.

    Only Part A can raise an AssertionError and abort the run.
    """

    logging.info(
        f"Running self-test (two-qubit gate = {TWO_QUBIT_GATE})...\n"
    )

    qc = QuantumCircuit(2)

    build_bell_prep(qc)
    qc.rz(0.0, 0)

    qc.save_density_matrix()

    # ---- Part A: noiseless check ----

    ideal_simulator = AerSimulator(
        method="density_matrix"
    )

    compiled = transpile_for_backend(
        qc,
        ideal_simulator
    )

    check_transpiled_gates(compiled, "self-test, noiseless")

    result = ideal_simulator.run(
        compiled
    ).result()

    rho = result.data(0)["density_matrix"]

    if not isinstance(
        rho,
        DensityMatrix
    ):
        rho = DensityMatrix(rho)

    fidelity_value = state_fidelity(
        IDEAL_STATE,
        rho
    )

    purity_value = np.real_if_close(
        purity(rho)
    ).item()

    concurrence_value = concurrence(
        rho
    )

    trace_value = trace_distance(
        rho,
        IDEAL_STATE
    )

    tolerance = 1e-6

    assert abs(fidelity_value - 1.0) < tolerance, \
        f"Self-test FAILED (no-noise, {TWO_QUBIT_GATE}): fidelity={fidelity_value}"

    assert abs(purity_value - 1.0) < tolerance, \
        f"Self-test FAILED (no-noise, {TWO_QUBIT_GATE}): purity={purity_value}"

    assert abs(concurrence_value - 1.0) < tolerance, \
        f"Self-test FAILED (no-noise, {TWO_QUBIT_GATE}): concurrence={concurrence_value}"

    assert abs(trace_value - 0.0) < tolerance, \
        f"Self-test FAILED (no-noise, {TWO_QUBIT_GATE}): trace_distance={trace_value}"

    logging.info(
        "Self-test PASSED -- no-noise check "
        f"(fidelity={fidelity_value:.6f}, "
        f"purity={purity_value:.6f}, "
        f"concurrence={concurrence_value:.6f}, "
        f"trace_distance={trace_value:.6f})\n"
    )

    # ---- Part B: reference printout at gamma=0.00 (not asserted) ----

    reference_simulator = AerSimulator(
        method="density_matrix",
        noise_model=create_noise_model(0.0)
    )

    compiled_ref = transpile_for_backend(
        qc,
        reference_simulator
    )

    result_ref = reference_simulator.run(
        compiled_ref
    ).result()

    rho_ref = result_ref.data(0)["density_matrix"]

    if not isinstance(
        rho_ref,
        DensityMatrix
    ):
        rho_ref = DensityMatrix(rho_ref)

    fidelity_ref = state_fidelity(
        IDEAL_STATE,
        rho_ref
    )

    purity_ref = np.real_if_close(
        purity(rho_ref)
    ).item()

    concurrence_ref = concurrence(
        rho_ref
    )

    trace_ref = trace_distance(
        rho_ref,
        IDEAL_STATE
    )

    logging.info(
        f"Reference -- gamma=0.00 composite model ({TWO_QUBIT_GATE}; "
        "zero depolarizing noise, thermal relaxation still present "
        "due to finite gate time -- NOT expected to equal the ideal state):\n"
        f"  fidelity={fidelity_ref:.6f}, "
        f"purity={purity_ref:.6f}, "
        f"concurrence={concurrence_ref:.6f}, "
        f"trace_distance={trace_ref:.6f}\n"
    )

# ==============================================================
# Result container
# ==============================================================

class SimulationResults:

    def __init__(self):

        self.rows = []

        self.fidelity = {}

        self.purity = {}

        self.concurrence = {}

        self.trace_distance = {}

    def add_curve(
        self,
        gamma,
        fidelity,
        purity_values,
        concurrence_values,
        trace_values
    ):

        self.fidelity[gamma] = fidelity

        self.purity[gamma] = purity_values

        self.concurrence[gamma] = concurrence_values

        self.trace_distance[gamma] = trace_values

    def add_row(
        self,
        gamma,
        theta,
        fidelity,
        purity_value,
        concurrence_value,
        trace_value
    ):

        self.rows.append(
            {
                "gamma": gamma,
                "theta": theta,
                "fidelity": fidelity,
                "purity": purity_value,
                "concurrence": concurrence_value,
                "trace_distance": trace_value
            }
        )

    def dataframe(self):

        return pd.DataFrame(
            self.rows
        )

# ==============================================================
# Simulation Engine
# ==============================================================

def simulate():

    logging.info(
        f"\nStarting Bell-state simulation (two-qubit gate = {TWO_QUBIT_GATE})...\n"
    )

    results = SimulationResults()

    # ----------------------------------------------------------
    # Create parameterized circuit once
    # ----------------------------------------------------------

    base_circuit = create_parameterized_bell()

    # ----------------------------------------------------------
    # Loop over noise strengths
    # ----------------------------------------------------------

    for gamma in NOISE_RATES:

        logging.info(
            f"Running γ = {gamma:.2f}"
        )

        simulator = AerSimulator(
            method="density_matrix",
            noise_model=create_noise_model(gamma)
        )

        # Transpile ONCE for this simulator
        compiled = transpile_for_backend(
            base_circuit,
            simulator
        )

        if gamma == NOISE_RATES[0]:
            check_transpiled_gates(compiled, "simulation")

        fidelity_curve = []
        purity_curve = []
        concurrence_curve = []
        trace_curve = []

        # ------------------------------------------------------
        # Sweep scrambling angle
        # ------------------------------------------------------

        for theta in THETA_VALUES:

            bound_circuit = compiled.assign_parameters(
                {
                    THETA: theta
                }
            )

            result = simulator.run(
                bound_circuit
            ).result()

            rho = result.data(0)["density_matrix"]

            if not isinstance(
                rho,
                DensityMatrix
            ):
                rho = DensityMatrix(rho)

            # ----------------------------------------------
            # Metrics
            # ----------------------------------------------

            fidelity_value = state_fidelity(
                IDEAL_STATE,
                rho
            )

            purity_value = np.real_if_close(
                purity(rho)
            ).item()

            concurrence_value = concurrence(
                rho
            )

            trace_value = trace_distance(
                rho,
                IDEAL_STATE
            )

            fidelity_curve.append(
                fidelity_value
            )

            purity_curve.append(
                purity_value
            )

            concurrence_curve.append(
                concurrence_value
            )

            trace_curve.append(
                trace_value
            )

            results.add_row(
                gamma,
                theta,
                fidelity_value,
                purity_value,
                concurrence_value,
                trace_value
            )

        # ------------------------------------------------------
        # Save curves
        # ------------------------------------------------------

        results.add_curve(
            gamma,
            fidelity_curve,
            purity_curve,
            concurrence_curve,
            trace_curve
        )

    logging.info(
        "\nSimulation complete.\n"
    )

    return results

# ==============================================================
# CSV Export
# ==============================================================

def save_results(results):

    dataframe = results.dataframe()

    csv_path = os.path.join(
        OUTPUT_DIR,
        f"fidelity_results{GATE_SUFFIX}.csv"
    )

    dataframe.to_csv(
        csv_path,
        index=False
    )

    logging.info(
        f"Results saved:\n{csv_path}"
    )

    return dataframe

# ==============================================================
# Summary Statistics
# ==============================================================

def print_summary(df):

    logging.info(
        "\n=============================="
    )

    logging.info(
        f"Summary Statistics (two-qubit gate = {TWO_QUBIT_GATE})"
    )

    logging.info(
        "=============================="
    )

    for gamma in NOISE_RATES:

        subset = df[
            df["gamma"] == gamma
        ]

        logging.info(
            f"\nγ = {gamma:.2f}"
        )

        logging.info(
            f"Mean Fidelity      : {subset['fidelity'].mean():.4f}"
        )

        logging.info(
            f"Mean Purity        : {subset['purity'].mean():.4f}"
        )

        logging.info(
            f"Mean Concurrence   : {subset['concurrence'].mean():.4f}"
        )

        logging.info(
            f"Mean Trace Distance: {subset['trace_distance'].mean():.4f}"
        )

# ==============================================================
# Generic Angle-Resolved Plotting Function
# ==============================================================

def plot_metric(
    curves,
    ylabel,
    title,
    filename,
    ymin=None,
    ymax=None
):
    """
    Generate a publication-quality angle-resolved plot for any metric.
    """

    plt.figure(
        figsize=(7.5, 4.8),
        dpi=300
    )

    colours = [
        "#1f77b4",
        "#2ca02c",
        "#ff7f0e",
        "#d62728"
    ]

    for colour, gamma in zip(
        colours,
        NOISE_RATES
    ):

        plt.plot(
            THETA_VALUES,
            curves[gamma],
            linewidth=2.2,
            color=colour,
            label=f"$\\gamma={gamma:.2f}$"
        )

    plt.xlabel(
        "Scrambling angle $\\theta$ (radians)",
        fontsize=11
    )

    plt.ylabel(
        ylabel,
        fontsize=11
    )

    plt.title(
        f"{title} ({TWO_QUBIT_GATE.upper()})",
        fontsize=12
    )

    plt.xticks(
        [
            0,
            np.pi/4,
            np.pi/2,
            3*np.pi/4,
            np.pi
        ],
        [
            "0",
            r"$\pi/4$",
            r"$\pi/2$",
            r"$3\pi/4$",
            r"$\pi$"
        ]
    )

    if ymin is not None or ymax is not None:

        plt.ylim(
            ymin,
            ymax
        )

    plt.grid(
        linestyle=":",
        alpha=0.35
    )

    plt.legend()

    plt.tight_layout()

    base, ext = os.path.splitext(filename)

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{base}{GATE_SUFFIX}{ext}"
        ),
        dpi=300
    )

    plt.close()

# ==============================================================
# Mean-metric-vs-noise summary plot
# ==============================================================

def plot_summary_vs_noise(df):
    """
    Plots mean metric value vs. depolarizing noise level (gamma),
    for all four metrics, each auto-scaled to its own data range.
    """

    summary = df.groupby(
        "gamma"
    ).mean(
        numeric_only=True
    ).reset_index()

    metrics = [
        ("fidelity", "Mean Fidelity"),
        ("purity", "Mean Purity"),
        ("concurrence", "Mean Concurrence"),
        ("trace_distance", "Mean Trace Distance"),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9.5, 7.5),
        dpi=300
    )

    for ax, (col, label) in zip(
        axes.flat,
        metrics
    ):

        ax.plot(
            summary["gamma"],
            summary[col],
            marker="o",
            linewidth=2.0,
            color="#1f77b4"
        )

        ax.set_xlabel(
            "Depolarizing probability $\\gamma$",
            fontsize=10
        )

        ax.set_ylabel(
            label,
            fontsize=10
        )

        ax.set_title(
            f"{label} vs. Noise Level ({TWO_QUBIT_GATE.upper()})",
            fontsize=11
        )

        ax.grid(
            linestyle=":",
            alpha=0.35
        )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"summary_vs_noise{GATE_SUFFIX}.png"
        ),
        dpi=300
    )

    plt.close()

# ==============================================================
# Generate All Angle-Resolved Figures
# ==============================================================

def generate_all_plots(results):

    plot_metric(
        results.fidelity,
        ylabel="Fidelity",
        title="Bell-State Fidelity under Unitary Scrambling",
        filename="fidelity_plot.png",
        ymin=0,
        ymax=1.02
    )

    plot_metric(
        results.purity,
        ylabel="Purity",
        title="State Purity under Unitary Scrambling",
        filename="purity_plot.png",
        ymin=0,
        ymax=1.02
    )

    plot_metric(
        results.concurrence,
        ylabel="Concurrence",
        title="Entanglement Concurrence",
        filename="concurrence_plot.png",
        ymin=0,
        ymax=1.02
    )

    plot_metric(
        results.trace_distance,
        ylabel="Trace Distance",
        title="Trace Distance from Ideal Bell State",
        filename="trace_distance_plot.png",
        ymin=0
    )

# ==============================================================
# Main
# ==============================================================

def main():

    logging.info(
        "\n========================================="
    )

    logging.info(
        f" Bell-State Simulation (v5 -- two-qubit gate = {TWO_QUBIT_GATE})"
    )

    logging.info(
        "=========================================\n"
    )

    self_test()

    results = simulate()

    dataframe = save_results(
        results
    )

    generate_all_plots(
        results
    )

    plot_summary_vs_noise(
        dataframe
    )

    print_summary(
        dataframe
    )

    logging.info(
        "\nFigures generated:"
    )

    logging.info(
        f"  fidelity_plot{GATE_SUFFIX}.png"
    )

    logging.info(
        f"  purity_plot{GATE_SUFFIX}.png"
    )

    logging.info(
        f"  concurrence_plot{GATE_SUFFIX}.png"
    )

    logging.info(
        f"  trace_distance_plot{GATE_SUFFIX}.png"
    )

    logging.info(
        f"  summary_vs_noise{GATE_SUFFIX}.png"
    )

    logging.info(
        "\nSimulation completed successfully."
    )


if __name__ == "__main__":

    main()
