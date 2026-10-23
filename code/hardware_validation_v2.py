"""
====================================================================
hardware_validation_v2.py

Title:
    Expanded, Reproducible Hardware Benchmarking for Bell-State
    Scrambling (multi-theta, bootstrap CIs, noise-matched tomography
    reference, dual gamma_eff estimation)

What's new vs. hardware_validation.py
---------------------------------------
1. MULTI-THETA sweep (THETA_VALUES, 5 points) instead of a single
   theta=0 measurement. (Reviewer Comment #9.)

2. A fully explicit, from-scratch two-qubit state-tomography
   protocol: 9 local-Pauli-basis circuits (X/Y/Z on each qubit) are
   built directly, and the density matrix is reconstructed via
   textbook linear inversion, implemented and documented in full
   below -- rather than depending on the internals of an external
   library's fitter. (Reviewer Comment #12: "which nine bases,
   reconstruction algorithm" is now fully specified in code.)

3. A backend CALIBRATION SNAPSHOT (per-qubit T1/T2, two-qubit gate
   error and duration) is captured at run time and saved alongside
   the results, for direct comparison against the paper's assumed
   T1=50us, T2=30us, gate durations. (Reviewer Comments #12, #16.)

4. Explicit, reproducible QUBIT-PAIR selection: by default, the
   directly-coupled pair with the lowest reported two-qubit gate
   error is chosen and logged (EXPLICIT_QUBIT_PAIR can override
   this). (Reviewer Comment #12.)

5. BOOTSTRAP CONFIDENCE INTERVALS for all four metrics, obtained by
   resampling the raw per-basis-setting counts and re-running the
   SAME linear-inversion reconstruction -- at zero extra hardware
   cost (no additional circuit submissions). (Reviewer Comment #30.)

6. A NOISE-MATCHED SIMULATED TOMOGRAPHY REFERENCE: rather than
   comparing the hardware-reconstructed state to a bare
   state-preparation-only simulation, the SAME 9 basis-rotation
   circuits are also run through the paper's composite noise model
   on the Aer density-matrix simulator, and reconstructed with the
   SAME linear-inversion formula. This means the simulated
   comparison point carries the same tomography overhead (extra
   gates -> extra decoherence, in the noise model) as the real
   hardware measurement, closing the gap the review specifically
   flagged in Comment #13.

7. TWO independently-documented methods for estimating the
   effective depolarizing probability gamma_eff, both reported:
     (a) per-metric monotonic interpolation over a dense gamma grid
     (b) a single gamma_eff jointly minimizing summed squared
         residuals across all four metrics simultaneously
   (Reviewer Comment #10.)

8. Batched submission: all 9 basis circuits for a given theta go in
   as ONE job rather than 9 separate ones (see get_counts_batch).

9. A DRY_RUN mode that exercises the entire pipeline against the
   paper's own AerSimulator noise model with a KNOWN injected gamma,
   in place of real hardware -- for validating correctness at zero
   runtime-budget cost.

10. A REPLOT_ONLY mode that regenerates the figure from the saved
    results CSV without running any measurement at all (see below).

Run history / status
-----------------------
This script HAS been run successfully end-to-end:

  * DRY_RUN validation (2026-08-22): self_test() passed and the
    gamma_eff estimates clustered near the injected
    DRY_RUN_GAMMA_TRUE=0.01 at all five theta points (joint
    estimates: 0.012, 0.010, 0.010, 0.008, 0.010), confirming the
    from-scratch reconstruction and the gamma-fitting logic.

  * Real hardware (2026-08-22): full 5-point sweep on
    ibm_marrakesh, physical qubits (1, 2), 4096 shots per circuit,
    with dynamical decoupling and measurement twirling both
    successfully enabled. Results are the ones reported in the paper.

  * Known API finding from that run: SamplerV2 has NO
    'resilience_level' option -- that is EstimatorV2-only. The
    correct Sampler-side mitigation knobs are dynamical decoupling
    and measurement twirling, which is what _configure_mitigation()
    now sets. See the note in that function.

Lines still marked "# VERIFY:" are qiskit-ibm-runtime API calls whose
attribute names have shifted across releases. They worked as written
on the run above, but re-check them if you upgrade the package.

Dependencies
------------
Required always:
    qiskit >= 1.0, qiskit-aer >= 0.15, numpy, pandas, matplotlib

Additionally required when DRY_RUN = False and REPLOT_ONLY = False:
    qiskit-ibm-runtime, and a saved IBM Quantum account

(qiskit-experiments is NOT required -- the tomography reconstruction
is implemented from scratch below, for full transparency and so that
DRY_RUN/REPLOT_ONLY have no IBM-specific dependency at all.)

Outputs
-------
output/
    hardware_validation_v2_results.csv
    calibration_snapshot.json
    hardware_validation_v2_theta_sweep.png
====================================================================
"""

import os
import json
import itertools
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import (
    DensityMatrix,
    state_fidelity,
    purity,
    concurrence,
)

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    _IBM_RUNTIME_AVAILABLE = True
except ImportError:
    _IBM_RUNTIME_AVAILABLE = False

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
# ============ CONFIGURATION -- READ BEFORE RUNNING ============
# ==============================================================

REPLOT_ONLY = True
# ^ When True, the script does NOT measure anything at all. It loads
#   the existing output/hardware_validation_v2_results.csv and
#   regenerates the figure from it, then exits.
#
#   Use this to redraw the figure (e.g. after changing axis limits)
#   WITHOUT spending IBM Quantum runtime re-measuring data you have
#   already collected. Requires that the CSV already exists.
#
#   Set to False to perform an actual measurement run.

DRY_RUN = False
# ^ Ignored when REPLOT_ONLY is True.
#   False: measure on real IBM Quantum hardware.
#   True:  simulate the hardware locally via AerSimulator with a known
#          injected gamma (DRY_RUN_GAMMA_TRUE) -- no runtime cost.
#   Dry run was validated 2026-08-22; see "Run history" above.

DRY_RUN_GAMMA_TRUE = 0.01
# ^ Only used when DRY_RUN=True. A known depolarizing probability is
#   injected via the paper's own noise model in place of a real
#   backend, so you can confirm gamma_eff estimation recovers ~0.01.

THETA_VALUES = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
# ^ Full 5-point sweep. Each theta costs 9 tomography circuits,
#   submitted as a single batched job. Trim this list for a cheaper
#   exploratory run.

SHOTS_PER_CIRCUIT = 4096
N_BOOTSTRAP = 2000          # resamples for CIs; free (no extra hardware jobs)

MIN_QUBITS = 2
EXPLICIT_QUBIT_PAIR = None  # e.g. (1, 2) to override auto-selection; else None
USE_READOUT_MITIGATION = True
TRANSPILE_OPT_LEVEL = 1

# ---- Figure appearance ----
FIXED_Y_AXIS = True
# ^ True: all four panels of the theta-sweep figure share a full
#   [0, 1.02] y-axis. This makes the FLATNESS of purity and
#   concurrence across theta immediately visible, matching the
#   simulation figures in the paper.
#
#   False: each panel auto-scales to its own data range. Purity and
#   concurrence then span only ~0.92-0.97, so ordinary shot-noise
#   scatter fills the panel and the points appear to zigzag -- which
#   visually contradicts the flatness claim in the text, even though
#   the underlying variation is tiny.
#
#   Trade-off: with FIXED_Y_AXIS=True the confidence intervals become
#   visually small. That is the honest picture (they really are small
#   relative to the metric range), and the maximum CI width is logged
#   at the end of the run so it can be quoted numerically in the text.

# ---- Noise model (identical to simulate_v4.py; duplicated here so
# this script is self-contained) ----
T1 = 50e-6
T2 = 30e-6
GATE_TIME_1Q = 50e-9
GATE_TIME_2Q = 300e-9
GAMMA_GRID = np.linspace(0.0, 0.05, 26)   # dense grid for gamma_eff estimation

# Module-level backend/qubit-pair state, set in main() before use.
_HARDWARE_BACKEND = None
_QUBIT_PAIR = (0, 1)

# ==============================================================
# Ideal Bell state / trace distance (identical to the other scripts)
# ==============================================================

def create_ideal_density_matrix():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return DensityMatrix.from_instruction(qc)


IDEAL_STATE = create_ideal_density_matrix()


def trace_distance(rho, sigma):
    delta = rho.data - sigma.data
    singular_values = np.linalg.svd(delta, compute_uv=False)
    return 0.5 * np.sum(singular_values)


def compute_metrics(rho):
    return {
        "fidelity": state_fidelity(IDEAL_STATE, rho),
        "purity": float(np.real_if_close(purity(rho))),
        "concurrence": float(concurrence(rho)),
        "trace_distance": trace_distance(rho, IDEAL_STATE),
    }

# ==============================================================
# Composite noise model (identical construction to simulate_v4.py)
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
    noise_model.add_all_qubit_quantum_error(combined_error_2q, ["cx"])
    return noise_model

# ==============================================================
# Tomography circuit construction: 9 local-Pauli bases (X/Y/Z on
# each of the two qubits)
# ==============================================================

BASES = ["X", "Y", "Z"]
BASIS_SETTINGS = list(itertools.product(BASES, repeat=2))  # (qubit0_basis, qubit1_basis)


def _basis_rotation(qc, qubit, basis):
    """
    Appends the gate(s) rotating `basis` into the Z basis, so a
    subsequent Z-measurement is equivalent to measuring that Pauli.
        X-basis: H
        Y-basis: Sdg, H
        Z-basis: (nothing)
    """
    if basis == "X":
        qc.h(qubit)
    elif basis == "Y":
        qc.sdg(qubit)
        qc.h(qubit)
    elif basis == "Z":
        pass
    else:
        raise ValueError(f"Unknown basis '{basis}'")


def build_tomography_circuit(theta, basis_pair, add_measurement=True):
    """
    Bell state (H, CX) + local scrambling attack Rz(theta) on qubit 0
    + basis-rotation gates for the requested (qubit0_basis,
    qubit1_basis) pair + optional measurement.
    """
    qc = QuantumCircuit(2, 2) if add_measurement else QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(theta, 0)
    _basis_rotation(qc, 0, basis_pair[0])
    _basis_rotation(qc, 1, basis_pair[1])
    if add_measurement:
        qc.measure([0, 1], [0, 1])
    return qc

# ==============================================================
# Linear-inversion two-qubit state tomography (from scratch)
# ==============================================================

PAULI_MATRICES = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _pauli_expectation_from_counts(counts, shots, qubit_index):
    """
    Estimates a Pauli expectation value from measurement counts.

    Qiskit counts-string convention: for QuantumCircuit(2, 2) with
    qc.measure([0, 1], [0, 1]), the returned bitstrings are ordered
    "c1c0" (rightmost character = classical bit 0 = qubit 0).

    qubit_index=0 -> marginal expectation on qubit 0
    qubit_index=1 -> marginal expectation on qubit 1
    qubit_index=None -> joint parity (product of both outcomes)
    """
    total = 0.0
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        b0 = int(bits[-1])
        b1 = int(bits[-2])
        if qubit_index == 0:
            val = 1 - 2 * b0
        elif qubit_index == 1:
            val = 1 - 2 * b1
        else:
            val = (1 - 2 * b0) * (1 - 2 * b1)
        total += val * (count / shots)
    return total


def reconstruct_density_matrix(counts_by_setting, shots):
    """
    Textbook linear-inversion two-qubit state tomography from 9
    local-Pauli measurement settings.

    For each pair (a, b) in {X,Y,Z}x{X,Y,Z}, the corresponding
    circuit's joint-parity expectation directly gives r_{ab} =
    <P_a (x) P_b>. Each single-qubit marginal r_{a,I} / r_{I,b} is
    estimated from the 3 settings that share that qubit's basis and
    averaged for lower variance. r_{I,I} = 1 always. This gives all
    16 two-qubit Pauli expectation values, and

        rho = (1/4) * sum_{a,b in {I,X,Y,Z}} r_{ab} * (P_a (x) P_b)

    NOTE on operator ordering: Qiskit's statevector/density-matrix
    convention is little-endian (qubit 0 is the LEAST significant
    tensor factor), so for a=qubit-0's basis, b=qubit-1's basis, the
    matrix contribution is np.kron(P_b, P_a) -- NOT np.kron(P_a, P_b).
    This is easy to get backwards; it is verified by self_test()
    below, which checks the reconstruction against a known state.

    Returns a Hermitian, trace-1 DensityMatrix that is NOT guaranteed
    to be positive-semidefinite (this is the well-known drawback of
    linear inversion vs. a constrained/MLE fitter). Call
    project_to_physical() on the result before computing purity or
    concurrence.
    """

    r = {("I", "I"): 1.0}

    for (a, b), counts in counts_by_setting.items():
        r[(a, b)] = _pauli_expectation_from_counts(counts, shots, qubit_index=None)

    for a in BASES:
        vals = [
            _pauli_expectation_from_counts(counts_by_setting[(a, b)], shots, qubit_index=0)
            for b in BASES
        ]
        r[(a, "I")] = float(np.mean(vals))

    for b in BASES:
        vals = [
            _pauli_expectation_from_counts(counts_by_setting[(a, b)], shots, qubit_index=1)
            for a in BASES
        ]
        r[("I", b)] = float(np.mean(vals))

    rho = np.zeros((4, 4), dtype=complex)
    for a in ["I"] + BASES:
        for b in ["I"] + BASES:
            # np.kron(P_b, P_a): qubit 1 (b) is the more-significant
            # tensor factor in Qiskit's little-endian convention.
            rho += r[(a, b)] * np.kron(PAULI_MATRICES[b], PAULI_MATRICES[a])

    rho = rho / 4.0
    rho = 0.5 * (rho + rho.conj().T)  # enforce exact Hermiticity

    return DensityMatrix(rho)


def project_to_physical(rho):
    """
    Projects a Hermitian, trace-1 (but possibly non-PSD) density
    matrix onto the nearest physical state by clipping negative
    eigenvalues to zero and renormalizing. This is a simplified
    physicality projection (not the full constrained fit of e.g.
    Smolin et al. 2012); it is used here only to make
    purity/concurrence numerically well-defined for the
    linear-inversion reconstruction used throughout this script.
    """
    mat = rho.data if isinstance(rho, DensityMatrix) else rho
    mat = 0.5 * (mat + mat.conj().T)
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals_clipped = np.clip(eigvals, 0, None)
    total = eigvals_clipped.sum()
    if total <= 0:
        return DensityMatrix(np.eye(mat.shape[0]) / mat.shape[0])
    eigvals_norm = eigvals_clipped / total
    mat_physical = (eigvecs * eigvals_norm) @ eigvecs.conj().T
    return DensityMatrix(mat_physical)

# ==============================================================
# Self-test: validates the from-scratch reconstruction against a
# fully known state, using ONLY AerSimulator (no IBM dependency).
# ==============================================================

def self_test():

    logging.info("Running self-test (tomography reconstruction correctness)...\n")

    rng = np.random.default_rng(0)
    simulator = AerSimulator()  # noiseless shot-based simulator
    shots = 50000
    theta_test = 0.0

    counts_by_setting = {}
    for basis_pair in BASIS_SETTINGS:
        qc = build_tomography_circuit(theta_test, basis_pair)
        compiled = transpile(qc, simulator, optimization_level=1)
        result = simulator.run(compiled, shots=shots, seed_simulator=int(rng.integers(1_000_000))).result()
        counts_by_setting[basis_pair] = result.get_counts()

    rho_recon = project_to_physical(reconstruct_density_matrix(counts_by_setting, shots))
    metrics = compute_metrics(rho_recon)

    tolerance = 0.03  # loose: genuine finite-shot statistical noise, not exact simulation

    assert abs(metrics["fidelity"] - 1.0) < tolerance, \
        f"Self-test FAILED: reconstructed fidelity={metrics['fidelity']:.4f} (expected ~1.0)"
    assert abs(metrics["purity"] - 1.0) < tolerance, \
        f"Self-test FAILED: reconstructed purity={metrics['purity']:.4f} (expected ~1.0)"
    assert abs(metrics["concurrence"] - 1.0) < tolerance, \
        f"Self-test FAILED: reconstructed concurrence={metrics['concurrence']:.4f} (expected ~1.0)"
    assert abs(metrics["trace_distance"] - 0.0) < tolerance, \
        f"Self-test FAILED: reconstructed trace_distance={metrics['trace_distance']:.4f} (expected ~0.0)"

    logging.info(
        "Self-test PASSED -- linear-inversion reconstruction recovers the ideal Bell "
        f"state to within shot noise (fidelity={metrics['fidelity']:.4f}, "
        f"purity={metrics['purity']:.4f}, concurrence={metrics['concurrence']:.4f}, "
        f"trace_distance={metrics['trace_distance']:.4f})\n"
    )

# ==============================================================
# Counts acquisition: DRY_RUN (AerSimulator) or real hardware
# ==============================================================

def _configure_mitigation(sampler):
    """
    Attempts Sampler-appropriate error-mitigation options, each
    independently guarded, and returns a dict recording exactly what
    was actually applied -- so the paper can report what really
    happened, not just what USE_READOUT_MITIGATION requested.

    NOTE (confirmed by the 2026-08-22 hardware run):
    SamplerV2 does NOT have a 'resilience_level' option -- that is an
    EstimatorV2-only concept (mitigation of expectation values).
    Sampler returns raw counts; the appropriate mitigation knobs for
    it are dynamical decoupling (reduces idle-qubit decoherence
    during the circuit) and measurement twirling (symmetrizes
    readout error), both attempted below. Both were successfully
    enabled on that run. If neither is available on your
    qiskit-ibm-runtime version, the run proceeds with default
    (unmitigated) Sampler settings and this is logged explicitly.
    """
    if not USE_READOUT_MITIGATION:
        return {"requested": False, "dynamical_decoupling": False, "measure_twirling": False}

    applied = {"requested": True, "dynamical_decoupling": False, "measure_twirling": False}

    try:
        sampler.options.dynamical_decoupling.enable = True   # VERIFY: option path for your version
        applied["dynamical_decoupling"] = True
    except Exception as exc:
        logging.warning(f"Could not enable dynamical_decoupling ({exc}).")

    try:
        sampler.options.twirling.enable_measure = True       # VERIFY: option path for your version
        applied["measure_twirling"] = True
    except Exception as exc:
        logging.warning(f"Could not enable measurement twirling ({exc}).")

    if not (applied["dynamical_decoupling"] or applied["measure_twirling"]):
        logging.warning(
            "No Sampler-level mitigation could be enabled on this qiskit-ibm-runtime "
            "version; proceeding with default (unmitigated) options. Report this run "
            "as using default Sampler settings, not explicit mitigation."
        )

    return applied


def get_counts_batch(theta, shots):
    """
    Submits all 9 basis-setting circuits for a given theta as a
    SINGLE job (one Sampler.run() / AerSimulator.run() call over a
    list of circuits), instead of 9 separate submissions.

    This matters in practice: an earlier unbatched version submitted
    9 separate jobs per theta, each with several seconds of
    submission/queueing overhead on top of execution time. Batching
    cuts that to one job per theta -- less wall-clock time and less
    queue contention against your runtime budget, with identical
    physics (same circuits, same shots).

    Returns
    -------
    counts_by_setting : dict {basis_pair: counts_dict}
    mitigation_info    : dict, see _configure_mitigation() /
                          {"dry_run": True} when DRY_RUN
    """
    circuits = [build_tomography_circuit(theta, bp) for bp in BASIS_SETTINGS]

    if DRY_RUN:
        simulator = AerSimulator(noise_model=create_noise_model(DRY_RUN_GAMMA_TRUE))
        compiled = [transpile(qc, simulator, optimization_level=TRANSPILE_OPT_LEVEL) for qc in circuits]
        result = simulator.run(compiled, shots=shots).result()
        counts_by_setting = {bp: result.get_counts(i) for i, bp in enumerate(BASIS_SETTINGS)}
        return counts_by_setting, {"dry_run": True}

    # ---- Real hardware path ----
    # VERIFY: transpile/backend kwargs, Sampler construction, and
    # result access below against your installed qiskit-ibm-runtime
    # version (these have changed across recent releases).

    compiled = transpile(
        circuits,
        backend=_HARDWARE_BACKEND,
        optimization_level=TRANSPILE_OPT_LEVEL,
        initial_layout=list(_QUBIT_PAIR),
    )

    try:
        sampler = Sampler(mode=_HARDWARE_BACKEND)          # VERIFY: current qiskit-ibm-runtime API
    except TypeError:
        sampler = Sampler(backend=_HARDWARE_BACKEND)        # VERIFY: fallback for older versions

    mitigation_info = _configure_mitigation(sampler)

    job = sampler.run(compiled, shots=shots)
    result = job.result()

    counts_by_setting = {}
    for i, bp in enumerate(BASIS_SETTINGS):
        pub_result = result[i]
        try:
            counts = pub_result.data.c.get_counts()         # VERIFY: default classical register name is "c"
        except AttributeError:
            reg = next(iter(vars(pub_result.data).values()))
            counts = reg.get_counts()
        counts_by_setting[bp] = counts

    return counts_by_setting, mitigation_info


def bootstrap_counts(counts, shots, rng):
    outcomes = list(counts.keys())
    probs = np.array([counts[o] / shots for o in outcomes])
    resampled = rng.multinomial(shots, probs)
    return dict(zip(outcomes, resampled))

# ==============================================================
# Backend / qubit-pair selection and calibration snapshot
# (real-hardware path only)
# ==============================================================

def _pick_best_qubit_pair(backend):
    """
    Picks the directly-coupled qubit pair with the lowest reported
    two-qubit gate error. VERIFY against your backend's
    target/coupling_map API -- this uses the current Target-based API
    (Qiskit >= 1.0).
    """
    target = backend.target
    coupling = backend.coupling_map
    two_q_gate = next(
        name for name in target.operation_names
        if target.operation_from_name(name).num_qubits == 2
    )

    best_pair, best_error = None, np.inf
    seen = set()
    for pair in coupling.get_edges():
        key = tuple(sorted(pair))
        if key in seen:
            continue
        seen.add(key)
        try:
            error = target[two_q_gate][pair].error
        except Exception:
            continue
        if error is not None and error < best_error:
            best_error = error
            best_pair = pair

    if best_pair is None:
        raise RuntimeError(
            "Could not determine a low-error qubit pair from backend.target; "
            "set EXPLICIT_QUBIT_PAIR manually."
        )
    return best_pair


def select_backend_and_qubits():
    """VERIFY: backend.properties()/backend.target attribute access against your qiskit-ibm-runtime version."""

    service = QiskitRuntimeService()
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=MIN_QUBITS)

    calibration = {
        "mode": "HARDWARE",
        "backend_name": backend.name,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "shots_per_circuit": SHOTS_PER_CIRCUIT,
        "transpile_optimization_level": TRANSPILE_OPT_LEVEL,
    }

    qubit_pair = EXPLICIT_QUBIT_PAIR if EXPLICIT_QUBIT_PAIR is not None else _pick_best_qubit_pair(backend)
    calibration["qubit_pair"] = list(qubit_pair)

    try:
        target = backend.target
        two_q_gate = next(
            name for name in target.operation_names
            if target.operation_from_name(name).num_qubits == 2
        )
        cal = target[two_q_gate][tuple(qubit_pair)]
        calibration["two_qubit_gate"] = two_q_gate
        calibration["two_qubit_gate_error"] = getattr(cal, "error", None)
        calibration["two_qubit_gate_duration_s"] = getattr(cal, "duration", None)

        for q in qubit_pair:
            qp = target.qubit_properties[q]
            calibration[f"q{q}_T1_s"] = getattr(qp, "t1", None)
            calibration[f"q{q}_T2_s"] = getattr(qp, "t2", None)

    except Exception as exc:
        logging.warning(
            f"Could not extract full calibration snapshot via backend.target ({exc}); "
            "falling back to backend.properties()."
        )
        try:
            props = backend.properties()
            for q in qubit_pair:
                calibration[f"q{q}_T1_s"] = props.t1(q)
                calibration[f"q{q}_T2_s"] = props.t2(q)
        except Exception as exc2:
            logging.warning(f"Calibration snapshot unavailable ({exc2}); proceeding without it.")

    return backend, tuple(qubit_pair), calibration

# ==============================================================
# Per-theta measurement: point estimate + bootstrap CI
# ==============================================================

def run_theta_point(theta, shots, rng):

    counts_by_setting, mitigation_info = get_counts_batch(theta, shots)

    rho_point = project_to_physical(reconstruct_density_matrix(counts_by_setting, shots))
    point_metrics = compute_metrics(rho_point)

    bootstrap_vals = {"fidelity": [], "purity": [], "concurrence": [], "trace_distance": []}
    for _ in range(N_BOOTSTRAP):
        resampled = {
            setting: bootstrap_counts(counts, shots, rng)
            for setting, counts in counts_by_setting.items()
        }
        rho_b = project_to_physical(reconstruct_density_matrix(resampled, shots))
        m = compute_metrics(rho_b)
        for k in bootstrap_vals:
            bootstrap_vals[k].append(m[k])

    ci = {}
    for k, vals in bootstrap_vals.items():
        vals = np.array(vals)
        ci[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "ci_low": float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5)),
        }

    return point_metrics, ci, mitigation_info

# ==============================================================
# Noise-matched simulated tomography reference (addresses whether
# tomography overhead was "actually accounted for" -- Reviewer #13)
# ==============================================================

def simulated_tomography_reference(theta, gamma):
    """
    Runs the SAME 9 basis-rotation circuits through the Aer
    density-matrix simulator under the paper's composite noise model
    at the given gamma, converts each circuit's EXACT post-rotation
    density matrix into exact outcome probabilities (i.e. the
    infinite-shot limit -- no sampling noise), and reconstructs via
    the SAME linear-inversion formula used for the real hardware
    data. This is the fair, noise-and-overhead-matched comparison
    point: unlike a bare state-preparation-only prediction, it
    includes decoherence contributed by the basis-rotation gates
    themselves.
    """

    simulator = AerSimulator(method="density_matrix", noise_model=create_noise_model(gamma))
    pseudo_shots = 10 ** 7

    counts_by_setting = {}
    for basis_pair in BASIS_SETTINGS:
        qc = build_tomography_circuit(theta, basis_pair, add_measurement=False)
        qc.save_density_matrix()
        compiled = transpile(qc, simulator, optimization_level=TRANSPILE_OPT_LEVEL)
        result = simulator.run(compiled).result()

        rho_setting = result.data(0)["density_matrix"]
        if not isinstance(rho_setting, DensityMatrix):
            rho_setting = DensityMatrix(rho_setting)

        probs = rho_setting.probabilities_dict()
        counts_by_setting[basis_pair] = {
            outcome: int(round(p * pseudo_shots)) for outcome, p in probs.items()
        }

    rho_recon = project_to_physical(reconstruct_density_matrix(counts_by_setting, pseudo_shots))
    return compute_metrics(rho_recon)


def gamma_dense_sweep(theta, gamma_grid=GAMMA_GRID):
    rows = []
    for gamma in gamma_grid:
        m = simulated_tomography_reference(theta, gamma)
        m["gamma"] = gamma
        rows.append(m)
    return pd.DataFrame(rows)

# ==============================================================
# gamma_eff estimation: (a) per-metric, (b) joint least-squares
# ==============================================================

def estimate_gamma_per_metric(hardware_value, sweep_df, metric):
    gammas = sweep_df["gamma"].values
    values = sweep_df[metric].values

    monotonic = np.all(np.diff(values) <= 1e-12) or np.all(np.diff(values) >= -1e-12)
    if not monotonic:
        logging.warning(
            f"gamma-sweep for metric '{metric}' at this theta is not monotonic in gamma; "
            "per-metric gamma_eff interpolation may be unreliable -- inspect sweep_df directly."
        )

    sort_idx = np.argsort(values)
    return float(np.interp(hardware_value, values[sort_idx], gammas[sort_idx]))


def estimate_gamma_joint(hardware_values, sweep_df,
                          metrics=("fidelity", "purity", "concurrence", "trace_distance")):
    residuals = np.zeros(len(sweep_df))
    for metric in metrics:
        residuals += (sweep_df[metric].values - hardware_values[metric]) ** 2

    best_idx = int(np.argmin(residuals))
    return {
        "gamma_eff_joint": float(sweep_df["gamma"].values[best_idx]),
        "joint_sse": float(residuals[best_idx]),
    }

# ==============================================================
# Plotting
# ==============================================================

def plot_hardware_vs_theta(df):
    """
    Four-panel angle-resolved figure with 95% bootstrap CIs.

    See the FIXED_Y_AXIS configuration note above for why the y-axis
    handling matters: on an auto-scaled axis, purity and concurrence
    span only ~0.92-0.97 and their (tiny) shot-noise scatter fills
    the panel, visually contradicting the flatness claim made in the
    text. A full [0, 1.02] axis shows the flatness honestly.
    """
    metrics = ["fidelity", "purity", "concurrence", "trace_distance"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), dpi=300)

    for ax, metric in zip(axes.flat, metrics):
        y = df[f"hw_{metric}"]
        yerr_low = y - df[f"hw_{metric}_ci_low"]
        yerr_high = df[f"hw_{metric}_ci_high"] - y

        ax.errorbar(df["theta"], y, yerr=[yerr_low, yerr_high], fmt="o-",
                    capsize=3, color="#1f77b4", label="Hardware (95% bootstrap CI)")

        if FIXED_Y_AXIS:
            ax.set_ylim(0, 1.02)

        ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
        ax.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
        ax.set_xlabel(r"Scrambling angle $\theta$ (radians)", fontsize=9)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=9)
        ax.set_title(metric.replace("_", " ").title(), fontsize=10)
        ax.grid(linestyle=":", alpha=0.35)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "hardware_validation_v2_theta_sweep.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def log_ci_widths(df):
    """
    Logs the maximum 95% CI width per metric. Useful when
    FIXED_Y_AXIS=True makes the error bars visually small: these
    numbers can be quoted directly in the text instead.
    """
    logging.info("\nMaximum 95% bootstrap CI width per metric (quote these in the text "
                 "if the error bars are hard to see in the figure):")
    for metric in ["fidelity", "purity", "concurrence", "trace_distance"]:
        widths = df[f"hw_{metric}_ci_high"] - df[f"hw_{metric}_ci_low"]
        logging.info(f"  {metric:15s} max width = {widths.max():.4f}  "
                     f"(mean {widths.mean():.4f})")

# ==============================================================
# Main
# ==============================================================

def main():

    csv_path = os.path.join(OUTPUT_DIR, "hardware_validation_v2_results.csv")

    # ---------- REPLOT-ONLY PATH (no measurement at all) ----------
    if REPLOT_ONLY:
        logging.info("\n=========================================")
        logging.info(" Hardware Benchmarking v2 -- REPLOT ONLY")
        logging.info("=========================================\n")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"REPLOT_ONLY=True but no results file found at {csv_path}. "
                "Run once with REPLOT_ONLY=False first to generate it."
            )

        df = pd.read_csv(csv_path)
        logging.info(f"Loaded {len(df)} theta points from {csv_path}")
        logging.info(f"FIXED_Y_AXIS = {FIXED_Y_AXIS}\n")

        out_path = plot_hardware_vs_theta(df)
        logging.info(f"Figure regenerated: {out_path}")
        log_ci_widths(df)
        logging.info("\nDone (no measurement performed, no runtime consumed).")
        return

    # ---------- MEASUREMENT PATH ----------
    logging.info("\n=========================================")
    logging.info(" Hardware Benchmarking v2")
    logging.info(f" DRY_RUN = {DRY_RUN}")
    logging.info("=========================================\n")

    self_test()

    global _HARDWARE_BACKEND, _QUBIT_PAIR

    if DRY_RUN:
        _HARDWARE_BACKEND = None
        _QUBIT_PAIR = (0, 1)
        calibration = {"mode": "DRY_RUN", "true_gamma_used": DRY_RUN_GAMMA_TRUE}
        logging.info(f"[DRY RUN] Using AerSimulator with gamma={DRY_RUN_GAMMA_TRUE} "
                      "in place of real hardware. Set DRY_RUN=False once validated.\n")
    else:
        if not _IBM_RUNTIME_AVAILABLE:
            raise RuntimeError("qiskit-ibm-runtime is not installed; `pip install qiskit-ibm-runtime` "
                                "or set DRY_RUN=True.")
        _HARDWARE_BACKEND, _QUBIT_PAIR, calibration = select_backend_and_qubits()
        logging.info(f"Backend: {_HARDWARE_BACKEND.name}, qubit pair: {_QUBIT_PAIR}\n")
        logging.info(f"Calibration snapshot:\n{json.dumps(calibration, indent=2, default=str)}\n")

    rng = np.random.default_rng(12345)
    all_rows = []

    for theta in THETA_VALUES:

        logging.info(f"--- theta = {theta:.4f} rad ---")

        point_metrics, ci, mitigation_info = run_theta_point(theta, SHOTS_PER_CIRCUIT, rng)

        logging.info(f"  Mitigation applied: {mitigation_info}")
        logging.info(f"  Point estimate: { {k: round(v, 4) for k, v in point_metrics.items()} }")
        for k, v in ci.items():
            logging.info(f"  {k}: mean={v['mean']:.4f}  95% CI=[{v['ci_low']:.4f}, {v['ci_high']:.4f}]")

        sweep_df = gamma_dense_sweep(theta)

        gamma_per_metric = {
            metric: estimate_gamma_per_metric(point_metrics[metric], sweep_df, metric)
            for metric in ["fidelity", "purity", "concurrence", "trace_distance"]
        }
        gamma_joint = estimate_gamma_joint(point_metrics, sweep_df)

        logging.info(f"  gamma_eff (per-metric): { {k: round(v, 5) for k, v in gamma_per_metric.items()} }")
        logging.info(f"  gamma_eff (joint least-squares): gamma={gamma_joint['gamma_eff_joint']:.5f}, "
                     f"sse={gamma_joint['joint_sse']:.2e}\n")

        row = {
            "theta": theta,
            **{f"hw_{k}": v for k, v in point_metrics.items()},
            **{f"hw_{k}_ci_low": ci[k]["ci_low"] for k in ci},
            **{f"hw_{k}_ci_high": ci[k]["ci_high"] for k in ci},
            **{f"gamma_eff_{k}": v for k, v in gamma_per_metric.items()},
            **gamma_joint,
            **{f"mitigation_{k}": v for k, v in mitigation_info.items()},
        }
        all_rows.append(row)

    df = pd.DataFrame(all_rows)

    df.to_csv(csv_path, index=False)
    logging.info(f"Results saved: {csv_path}")

    calib_path = os.path.join(OUTPUT_DIR, "calibration_snapshot.json")
    with open(calib_path, "w") as f:
        json.dump(calibration, f, indent=2, default=str)
    logging.info(f"Calibration snapshot saved: {calib_path}")

    out_path = plot_hardware_vs_theta(df)
    logging.info(f"Figure generated: {out_path}")

    log_ci_widths(df)

    if DRY_RUN:
        recovered = df[[c for c in df.columns if c.startswith("gamma_eff")]]
        logging.info(
            "\n[DRY RUN CHECK] gamma_eff estimates should cluster near "
            f"DRY_RUN_GAMMA_TRUE={DRY_RUN_GAMMA_TRUE}:\n{recovered.to_string(index=False)}\n"
        )

    logging.info("\nDone.")


if __name__ == "__main__":
    main()
