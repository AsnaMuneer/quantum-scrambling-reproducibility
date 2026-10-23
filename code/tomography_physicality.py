"""
====================================================================
tomography_physicality.py

Purpose (Reviewer Comment #19)
-------------------------------
Answers the reviewer's question about linear-inversion tomography by
producing hard evidence that the reconstructed density matrices are
physical, and by quantifying how much the choice of reconstruction
method actually matters.

For every theta, this script reports:

  1. lambda_min of the RAW linear-inversion reconstruction, BEFORE
     any physicality projection. This is the number the reviewer
     asked for. Negative values are expected occasionally at finite
     shot count -- what matters is their magnitude.

  2. lambda_min AFTER projection (>= 0 by construction, reported as
     a sanity check that the projection is actually being applied).

  3. The same statistics across all bootstrap resamples, so the
     claim covers not just the point estimate but every state used
     to build the confidence intervals.

  4. A full maximum-likelihood (MLE) reconstruction of the same
     counts, via the standard Cholesky parameterization (James,
     Kwiat, Munro & White, PRA 64, 052312 (2001)), which is
     physical by construction. Metrics from projection and from MLE
     are compared directly, so the paper can state how much the
     reconstruction choice changes the reported numbers.

  5. The raw per-basis counts, saved to JSON, so all of the above
     can be recomputed later WITHOUT re-running on hardware. (The
     original script did not persist counts, which is why this
     re-run is necessary at all.)

Two ways to run
---------------
  MODE = "hardware"  -- re-measures on the real backend. Costs one
                        batched job per theta (5 total). Produces the
                        numbers to put in the paper. Note that the
                        metrics will differ slightly from the earlier
                        run (different shots, different calibration
                        moment), so if you use this run's lambda_min
                        you should update Table II from this run too,
                        keeping everything self-consistent.

  MODE = "simulation" -- runs the identical analysis against the
                        paper's own noise model at a matched gamma.
                        Costs nothing. Useful to see the output
                        format and confirm the pipeline before
                        spending hardware time, but the resulting
                        lambda_min values are NOT the hardware's and
                        must not be reported as such.

  MODE = "from_saved" -- recomputes everything from a previously
                        saved counts JSON. No measurement at all.
                        Use this after a hardware run to redo the
                        analysis for free.

Requires: qiskit, qiskit-aer, numpy, pandas, scipy
          (+ qiskit-ibm-runtime for MODE="hardware")
====================================================================
"""

import os
import json
import itertools
import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, state_fidelity, purity, concurrence
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    _IBM_AVAILABLE = True
except ImportError:
    _IBM_AVAILABLE = False

logging.getLogger("qiskit").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(message)s")

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================
# CONFIGURATION
# ==============================================================

MODE = "hardware"          # "hardware" | "simulation" | "from_saved"
# ^ Simulation-mode dry run completed 2026-08-22 and behaved as expected
#   (lambda_min ~ -0.004 on point estimates, projection returning it to
#   zero, max |projection - MLE| = 0.008 in concurrence). Now set to
#   "hardware" for the real measurement. After that run completes, the
#   raw counts are saved to tomography_raw_counts.json, so any further
#   analysis can use MODE="from_saved" at zero hardware cost.

THETA_VALUES = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
SHOTS_PER_CIRCUIT = 4096
N_BOOTSTRAP = 500          # fewer than the main script: this is a diagnostic
RUN_MLE = True             # set False to skip the (slower) MLE comparison

SIM_GAMMA = 0.004          # only for MODE="simulation": matches the paper's
                           # inferred gamma_eff, so the simulated purity lands
                           # near the measured ~0.95 and the lambda_min
                           # statistics are representative of that regime.

EXPLICIT_BACKEND = "ibm_marrakesh"
# ^ Pin the run to a NAMED backend, so the physicality diagnostic is
#   performed on the same device as every other hardware number in the
#   paper. Set to None to fall back to service.least_busy(), but note
#   that least_busy can and does return a different device depending
#   on load at the moment you run -- a 2026-08-23 run landed on
#   ibm_fez rather than ibm_marrakesh for exactly this reason, making
#   its output unusable alongside the rest of the paper's results.

EXPLICIT_QUBIT_PAIR = (1, 2)   # match the run reported in the paper
TRANSPILE_OPT_LEVEL = 1
USE_MITIGATION = True

COUNTS_JSON = os.path.join(OUTPUT_DIR, "tomography_raw_counts.json")

# noise model parameters (identical to the paper)
T1, T2 = 50e-6, 30e-6
GATE_TIME_1Q, GATE_TIME_2Q = 50e-9, 300e-9

# ==============================================================
# Constants / helpers
# ==============================================================

I2 = np.eye(2, dtype=complex)
PAULI = {
    "I": I2,
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
BASES = ["X", "Y", "Z"]
BASIS_SETTINGS = list(itertools.product(BASES, repeat=2))

PHI_PLUS = np.zeros(4, dtype=complex)
PHI_PLUS[0] = PHI_PLUS[3] = 1 / np.sqrt(2)
IDEAL_STATE = DensityMatrix(np.outer(PHI_PLUS, PHI_PLUS.conj()))


def trace_distance(rho, sigma):
    d = (rho.data if isinstance(rho, DensityMatrix) else rho) - \
        (sigma.data if isinstance(sigma, DensityMatrix) else sigma)
    return 0.5 * np.sum(np.linalg.svd(d, compute_uv=False))


def compute_metrics(rho):
    rho_dm = rho if isinstance(rho, DensityMatrix) else DensityMatrix(rho)
    return {
        "fidelity": float(state_fidelity(IDEAL_STATE, rho_dm)),
        "purity": float(np.real_if_close(purity(rho_dm))),
        "concurrence": float(concurrence(rho_dm)),
        "trace_distance": float(trace_distance(rho_dm, IDEAL_STATE)),
    }


def lambda_min(rho):
    m = rho.data if isinstance(rho, DensityMatrix) else rho
    return float(np.min(np.real(np.linalg.eigvalsh(0.5 * (m + m.conj().T)))))

# ==============================================================
# Circuits
# ==============================================================

def _basis_rotation(qc, qubit, basis):
    if basis == "X":
        qc.h(qubit)
    elif basis == "Y":
        qc.sdg(qubit); qc.h(qubit)


def build_tomography_circuit(theta, basis_pair):
    qc = QuantumCircuit(2, 2)
    qc.h(0); qc.cx(0, 1); qc.rz(theta, 0)
    _basis_rotation(qc, 0, basis_pair[0])
    _basis_rotation(qc, 1, basis_pair[1])
    qc.measure([0, 1], [0, 1])
    return qc


def create_noise_model(gamma):
    e1 = depolarizing_error(gamma, 1).compose(
        thermal_relaxation_error(T1, T2, GATE_TIME_1Q))
    e2 = depolarizing_error(gamma, 2).compose(
        thermal_relaxation_error(T1, T2, GATE_TIME_2Q).tensor(
            thermal_relaxation_error(T1, T2, GATE_TIME_2Q)))
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(e1, ["h", "rz"])
    nm.add_all_qubit_quantum_error(e2, ["cx"])
    return nm

# ==============================================================
# Reconstruction: linear inversion, projection, MLE
# ==============================================================

def _expectation(counts, shots, qubit_index):
    tot = 0.0
    for bits, n in counts.items():
        b = bits.replace(" ", "")
        b0, b1 = int(b[-1]), int(b[-2])
        if qubit_index == 0:
            v = 1 - 2 * b0
        elif qubit_index == 1:
            v = 1 - 2 * b1
        else:
            v = (1 - 2 * b0) * (1 - 2 * b1)
        tot += v * (n / shots)
    return tot


def linear_inversion(counts_by_setting, shots):
    """Identical formula to hardware_validation_v2.py. May be non-PSD."""
    r = {("I", "I"): 1.0}
    for bp, c in counts_by_setting.items():
        r[bp] = _expectation(c, shots, None)
    for a in BASES:
        r[(a, "I")] = float(np.mean(
            [_expectation(counts_by_setting[(a, b)], shots, 0) for b in BASES]))
    for b in BASES:
        r[("I", b)] = float(np.mean(
            [_expectation(counts_by_setting[(a, b)], shots, 1) for a in BASES]))

    rho = np.zeros((4, 4), dtype=complex)
    for a in ["I"] + BASES:
        for b in ["I"] + BASES:
            rho += r[(a, b)] * np.kron(PAULI[b], PAULI[a])
    rho /= 4.0
    return 0.5 * (rho + rho.conj().T)


def project_to_physical(rho):
    """Clip negative eigenvalues to zero, renormalize trace."""
    m = 0.5 * (rho + rho.conj().T)
    ev, evec = np.linalg.eigh(m)
    ev = np.clip(ev, 0, None)
    s = ev.sum()
    if s <= 0:
        return np.eye(4) / 4
    return (evec * (ev / s)) @ evec.conj().T


def _basis_unitary(b):
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    Sdg = np.array([[1, 0], [0, -1j]], dtype=complex)
    return {"X": H, "Y": H @ Sdg, "Z": I2}[b]


def _probs_for_setting(rho, bp):
    U = np.kron(_basis_unitary(bp[1]), _basis_unitary(bp[0]))  # little-endian
    return np.clip(np.real(np.diag(U @ rho @ U.conj().T)), 1e-12, None)


def _rho_from_cholesky(t):
    T = np.zeros((4, 4), dtype=complex)
    idx = 0
    for i in range(4):
        T[i, i] = t[idx]; idx += 1
    for i in range(1, 4):
        for j in range(i):
            T[i, j] = t[idx] + 1j * t[idx + 1]; idx += 2
    M = T.conj().T @ T
    tr = np.real(np.trace(M))
    return M / tr if tr > 1e-12 else np.eye(4) / 4


def _neg_loglik(t, counts_by_setting, shots):
    rho = _rho_from_cholesky(t)
    nll = 0.0
    for bp, counts in counts_by_setting.items():
        p = _probs_for_setting(rho, bp)
        for bits, n in counts.items():
            b = bits.replace(" ", "")
            k = int(b[-2]) * 2 + int(b[-1])
            nll -= n * np.log(p[k])
    return nll


def mle_reconstruct(counts_by_setting, shots, rho_init=None):
    """
    Maximum-likelihood reconstruction via Cholesky parameterization
    (James, Kwiat, Munro & White, PRA 64, 052312 (2001)). The
    parameterization rho = T^dag T / tr(T^dag T) is positive
    semidefinite and unit trace BY CONSTRUCTION, so the result is
    always physical without any projection step.
    """
    t0 = np.zeros(16); t0[:4] = 0.5
    if rho_init is not None:
        try:
            L = np.linalg.cholesky(rho_init + 1e-9 * np.eye(4))
            idx = 0
            for i in range(4):
                t0[idx] = np.real(L[i, i]); idx += 1
            for i in range(1, 4):
                for j in range(i):
                    t0[idx] = np.real(L[i, j]); t0[idx + 1] = np.imag(L[i, j]); idx += 2
        except np.linalg.LinAlgError:
            pass
    res = minimize(_neg_loglik, t0, args=(counts_by_setting, shots),
                   method="L-BFGS-B", options={"maxiter": 2000, "ftol": 1e-12})
    return _rho_from_cholesky(res.x), res.success

# ==============================================================
# Counts acquisition
# ==============================================================

def get_counts(theta, shots):
    circuits = [build_tomography_circuit(theta, bp) for bp in BASIS_SETTINGS]

    if MODE == "simulation":
        sim = AerSimulator(noise_model=create_noise_model(SIM_GAMMA))
        compiled = [transpile(c, sim, optimization_level=TRANSPILE_OPT_LEVEL) for c in circuits]
        res = sim.run(compiled, shots=shots).result()
        return {bp: res.get_counts(i) for i, bp in enumerate(BASIS_SETTINGS)}

    # hardware
    backend = _BACKEND
    compiled = transpile(circuits, backend=backend,
                         optimization_level=TRANSPILE_OPT_LEVEL,
                         initial_layout=list(EXPLICIT_QUBIT_PAIR))
    try:
        sampler = Sampler(mode=backend)
    except TypeError:
        sampler = Sampler(backend=backend)

    if USE_MITIGATION:
        try:
            sampler.options.dynamical_decoupling.enable = True
        except Exception as e:
            logging.warning(f"  dynamical_decoupling unavailable: {e}")
        try:
            sampler.options.twirling.enable_measure = True
        except Exception as e:
            logging.warning(f"  measurement twirling unavailable: {e}")

    result = sampler.run(compiled, shots=shots).result()
    out = {}
    for i, bp in enumerate(BASIS_SETTINGS):
        pr = result[i]
        try:
            out[bp] = pr.data.c.get_counts()
        except AttributeError:
            out[bp] = next(iter(vars(pr.data).values())).get_counts()
    return out


def bootstrap(counts, shots, rng):
    keys = list(counts.keys())
    p = np.array([counts[k] / shots for k in keys], dtype=float)
    p = p / p.sum()
    return dict(zip(keys, rng.multinomial(shots, p)))

# ==============================================================
# Main
# ==============================================================

_BACKEND = None

def main():
    global _BACKEND

    logging.info("\n=========================================")
    logging.info(f" Tomography Physicality Diagnostic (MODE={MODE})")
    logging.info("=========================================\n")

    if MODE == "hardware":
        if not _IBM_AVAILABLE:
            raise RuntimeError("qiskit-ibm-runtime not installed.")
        service = QiskitRuntimeService()
        if EXPLICIT_BACKEND is not None:
            _BACKEND = service.backend(EXPLICIT_BACKEND)
            logging.info(f"Backend: {_BACKEND.name} (pinned via EXPLICIT_BACKEND), "
                         f"qubits {EXPLICIT_QUBIT_PAIR}\n")
        else:
            _BACKEND = service.least_busy(operational=True, simulator=False, min_num_qubits=2)
            logging.warning(
                f"Backend: {_BACKEND.name} (chosen by least_busy -- NOT pinned). "
                "Confirm this matches the backend used elsewhere in the paper "
                "before quoting these numbers.\n")
    elif MODE == "simulation":
        logging.info(f"Simulated at gamma={SIM_GAMMA} (NOT hardware data -- "
                     "do not report these lambda_min values as measured).\n")

    # ---- acquire or load counts ----
    if MODE == "from_saved":
        if not os.path.exists(COUNTS_JSON):
            raise FileNotFoundError(f"No saved counts at {COUNTS_JSON}.")
        with open(COUNTS_JSON) as f:
            raw = json.load(f)
        all_counts = {float(k): {tuple(bp.split("-")): c for bp, c in v.items()}
                      for k, v in raw.items()}
        logging.info(f"Loaded counts for {len(all_counts)} theta values from {COUNTS_JSON}\n")
    else:
        all_counts = {}
        for theta in THETA_VALUES:
            logging.info(f"Acquiring counts at theta = {theta:.4f} ...")
            all_counts[theta] = get_counts(theta, SHOTS_PER_CIRCUIT)
        serial = {str(t): {"-".join(bp): c for bp, c in cs.items()}
                  for t, cs in all_counts.items()}
        with open(COUNTS_JSON, "w") as f:
            json.dump(serial, f, indent=1)
        logging.info(f"\nRaw counts saved: {COUNTS_JSON}")
        logging.info("(All analysis below can now be redone with MODE='from_saved', "
                     "at zero hardware cost.)\n")

    # ---- analysis ----
    rng = np.random.default_rng(2024)
    rows = []

    for theta, counts in sorted(all_counts.items()):
        rho_raw = linear_inversion(counts, SHOTS_PER_CIRCUIT)
        lmin_raw = lambda_min(rho_raw)

        rho_proj = project_to_physical(rho_raw)
        lmin_proj = lambda_min(rho_proj)
        m_proj = compute_metrics(rho_proj)

        # bootstrap lambda_min distribution (pre-projection)
        boot_lmin = []
        for _ in range(N_BOOTSTRAP):
            rs = {bp: bootstrap(c, SHOTS_PER_CIRCUIT, rng) for bp, c in counts.items()}
            boot_lmin.append(lambda_min(linear_inversion(rs, SHOTS_PER_CIRCUIT)))
        boot_lmin = np.array(boot_lmin)

        row = {
            "theta": theta,
            "lambda_min_raw": lmin_raw,
            "lambda_min_projected": lmin_proj,
            "boot_lambda_min_worst": float(boot_lmin.min()),
            "boot_lambda_min_mean": float(boot_lmin.mean()),
            "boot_frac_negative": float((boot_lmin < 0).mean()),
            **{f"proj_{k}": v for k, v in m_proj.items()},
        }

        logging.info(f"--- theta = {theta:.4f} ---")
        logging.info(f"  lambda_min (raw linear inversion) : {lmin_raw:+.6f}")
        logging.info(f"  lambda_min (after projection)     : {lmin_proj:+.2e}")
        logging.info(f"  bootstrap lambda_min: worst={boot_lmin.min():+.6f}, "
                     f"mean={boot_lmin.mean():+.6f}, "
                     f"negative in {100*(boot_lmin<0).mean():.1f}% of resamples")

        if RUN_MLE:
            rho_mle, ok = mle_reconstruct(counts, SHOTS_PER_CIRCUIT, rho_init=rho_proj)
            m_mle = compute_metrics(rho_mle)
            row.update({f"mle_{k}": v for k, v in m_mle.items()})
            row["mle_converged"] = bool(ok)
            row["lambda_min_mle"] = lambda_min(rho_mle)
            diffs = {k: abs(m_proj[k] - m_mle[k]) for k in m_proj}
            row.update({f"absdiff_{k}": v for k, v in diffs.items()})
            logging.info(f"  MLE converged={ok}, lambda_min={lambda_min(rho_mle):+.6f}")
            logging.info("  |projection - MLE|: " +
                         ", ".join(f"{k}={v:.4f}" for k, v in diffs.items()))
        logging.info("")

        rows.append(row)

    df = pd.DataFrame(rows)
    backend_label = (_BACKEND.name if (MODE == "hardware" and _BACKEND is not None)
                     else f"SIMULATION(gamma={SIM_GAMMA})" if MODE == "simulation"
                     else "FROM_SAVED")
    df.insert(0, "backend", backend_label)
    out_csv = os.path.join(OUTPUT_DIR, "tomography_physicality.csv")
    df.to_csv(out_csv, index=False)

    # ---- headline summary for the paper ----
    logging.info("=========================================")
    logging.info(" SUMMARY -- numbers to quote in the paper")
    logging.info("=========================================")
    logging.info(f"Backend / source : {backend_label}")

    lmin_pt = df["lambda_min_raw"].min()
    lmin_boot = df["boot_lambda_min_worst"].min()
    lmin_proj = df["lambda_min_projected"].min()

    logging.info(f"Minimum lambda_min over point estimates (pre-projection) : {lmin_pt:+.6f}"
                 + ("   <-- all point estimates already physical; no clipping was needed"
                    if lmin_pt >= 0 else
                    "   <-- negative: projection was required"))
    logging.info(f"Minimum lambda_min over all bootstrap resamples          : {lmin_boot:+.6f}")
    logging.info(f"Minimum lambda_min after projection                      : {lmin_proj:+.2e}"
                 + ("   (>= 0 by construction)" if lmin_proj >= -1e-12 else
                    "   <-- UNEXPECTED: projection should never yield a negative value"))

    if RUN_MLE:
        for k in ["fidelity", "purity", "concurrence", "trace_distance"]:
            logging.info(f"max |projection - MLE|, {k:15s}: {df[f'absdiff_{k}'].max():.4f}")
    logging.info(f"\nSaved: {out_csv}")
    if MODE == "simulation":
        logging.info("\nREMINDER: MODE='simulation' -- these are NOT hardware numbers.")


if __name__ == "__main__":
    main()
