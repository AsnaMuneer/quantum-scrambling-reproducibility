"""
gamma_eff_uncertainty.py

Uncertainty and residuals for the joint gamma_eff estimator (Reviewer 1,
comment 3). Self-contained: uses only output/tomography_raw_counts.json and
AerSimulator. No hardware.
"""

import os
import json
import logging
import itertools
import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, state_fidelity, purity, concurrence
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

logging.basicConfig(level=logging.INFO, format="%(message)s")

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
COUNTS_JSON = os.path.join(OUTPUT_DIR, "tomography_raw_counts.json")

THETA_VALUES = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
GAMMA_GRID = np.linspace(0.0, 0.05, 26)
N_BOOT = 2000
SEED = 12345
METRICS = ["fidelity", "purity", "concurrence", "trace_distance"]
BASES = ["X", "Y", "Z"]
BASIS_SETTINGS = list(itertools.product(BASES, repeat=2))
T1, T2 = 50e-6, 30e-6
GATE_TIME_1Q, GATE_TIME_2Q = 50e-9, 300e-9
PSEUDO_SHOTS = 10 ** 7

PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _ideal_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


IDEAL = DensityMatrix.from_instruction(_ideal_circuit())


def load_counts(path):
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for theta_str, settings in raw.items():
        out[float(theta_str)] = {
            tuple(bp.split("-")): {bits: int(n) for bits, n in counts.items()}
            for bp, counts in settings.items()
        }
    return out


def lookup(saved, theta):
    for t, settings in saved.items():
        if np.isclose(t, theta, atol=1e-12):
            return settings
    raise KeyError(f"No saved counts for theta={theta}")


def expectation(counts, shots, qubit_index):
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


def reconstruct(counts_by_setting, shots):
    r = {("I", "I"): 1.0}
    for bp, c in counts_by_setting.items():
        r[bp] = expectation(c, shots, None)
    for a in BASES:
        r[(a, "I")] = float(np.mean([expectation(counts_by_setting[(a, b)], shots, 0) for b in BASES]))
    for b in BASES:
        r[("I", b)] = float(np.mean([expectation(counts_by_setting[(a, b)], shots, 1) for a in BASES]))
    rho = np.zeros((4, 4), dtype=complex)
    for a in ["I"] + BASES:
        for b in ["I"] + BASES:
            rho += r[(a, b)] * np.kron(PAULI[b], PAULI[a])
    rho /= 4.0
    return 0.5 * (rho + rho.conj().T)


def project(mat):
    m = 0.5 * (mat + mat.conj().T)
    ev, evec = np.linalg.eigh(m)
    ev = np.clip(ev, 0, None)
    s = ev.sum()
    if s <= 0:
        return np.eye(4) / 4
    return (evec * (ev / s)) @ evec.conj().T


def metrics_of(rho_mat):
    dm = DensityMatrix(rho_mat)
    d = rho_mat - IDEAL.data
    trd = 0.5 * np.sum(np.linalg.svd(d, compute_uv=False))
    return {
        "fidelity": float(state_fidelity(IDEAL, dm)),
        "purity": float(np.real_if_close(purity(dm))),
        "concurrence": float(concurrence(dm)),
        "trace_distance": float(trd),
    }


def bootstrap_counts(counts, shots, rng):
    keys = list(counts.keys())
    p = np.array([counts[k] / shots for k in keys], dtype=float)
    p = p / p.sum()
    return dict(zip(keys, rng.multinomial(shots, p)))


def create_noise_model(gamma):
    e1 = depolarizing_error(gamma, 1).compose(thermal_relaxation_error(T1, T2, GATE_TIME_1Q))
    e2 = depolarizing_error(gamma, 2).compose(
        thermal_relaxation_error(T1, T2, GATE_TIME_2Q).tensor(
            thermal_relaxation_error(T1, T2, GATE_TIME_2Q)))
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(e1, ["h", "rz"])
    nm.add_all_qubit_quantum_error(e2, ["cx"])
    return nm


def build_tomo(theta, basis_pair):
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(theta, 0)
    for q, b in zip([0, 1], basis_pair):
        if b == "X":
            qc.h(q)
        elif b == "Y":
            qc.sdg(q)
            qc.h(q)
    qc.save_density_matrix()
    return qc


def sweep_for_theta(theta):
    rows = []
    for g in GAMMA_GRID:
        sim = AerSimulator(method="density_matrix", noise_model=create_noise_model(g))
        counts = {}
        for bp in BASIS_SETTINGS:
            qc = build_tomo(theta, bp)
            comp = transpile(qc, sim, optimization_level=1)
            rho = sim.run(comp).result().data(0)["density_matrix"]
            probs = DensityMatrix(rho).probabilities_dict()
            counts[bp] = {o: int(round(p * PSEUDO_SHOTS)) for o, p in probs.items()}
        m = metrics_of(project(reconstruct(counts, PSEUDO_SHOTS)))
        m["gamma"] = float(g)
        rows.append(m)
    return pd.DataFrame(rows)


def joint_estimate(point, sweep_df):
    resid = np.zeros(len(sweep_df))
    for m in METRICS:
        resid += (sweep_df[m].values - point[m]) ** 2
    i = int(np.argmin(resid))
    return float(sweep_df["gamma"].values[i]), float(resid[i])


EXPECTED = {0.0: 0.004, np.pi / 4: 0.004, np.pi / 2: 0.004,
            3 * np.pi / 4: 0.002, np.pi: 0.002}


def expected_for(theta):
    for t, g in EXPECTED.items():
        if np.isclose(t, theta, atol=1e-9):
            return g
    return None


def main():
    logging.info("gamma_eff uncertainty (saved counts, no hardware)\n")
    saved = load_counts(COUNTS_JSON)
    rng = np.random.default_rng(SEED)
    summary, dist, resid_rows = [], [], []
    all_pass = True

    for theta in THETA_VALUES:
        logging.info(f"--- theta = {theta:.4f} ---")
        counts = lookup(saved, theta)
        shots = {sum(c.values()) for c in counts.values()}
        assert len(shots) == 1, f"unequal shots: {shots}"
        shots = shots.pop()

        point = metrics_of(project(reconstruct(counts, shots)))

        sweep_df = sweep_for_theta(theta)
        g_point, sse_point = joint_estimate(point, sweep_df)

        idx = int(np.argmin(np.abs(sweep_df["gamma"].values - g_point)))
        for m in METRICS:
            sim_val = float(sweep_df[m].values[idx])
            resid_rows.append({
                "theta": theta, "metric": m, "measured": point[m],
                "simulated_at_gamma_eff": sim_val,
                "residual_sim_minus_meas": sim_val - point[m],
                "gamma_eff_joint": g_point, "joint_sse": sse_point,
            })

        exp = expected_for(theta)
        if exp is not None:
            ok = np.isclose(g_point, exp, atol=1e-9)
            all_pass = all_pass and ok
            logging.info(f"  joint gamma_eff: {g_point:.3f} (Table II {exp:.3f}) -> "
                         f"{'PASS' if ok else 'MISMATCH'}")
        else:
            logging.info(f"  joint gamma_eff: {g_point:.3f}")

        boot = np.empty(N_BOOT)
        for b in range(N_BOOT):
            rs = {s: bootstrap_counts(c, shots, rng) for s, c in counts.items()}
            bp = metrics_of(project(reconstruct(rs, shots)))
            boot[b], _ = joint_estimate(bp, sweep_df)

        lo, med, hi = np.percentile(boot, [2.5, 50, 97.5])
        frac = float(np.mean(np.isclose(boot, g_point, atol=1e-9)))
        logging.info(f"  bootstrap: median={med:.3f}, 95%=[{lo:.3f}, {hi:.3f}], "
                     f"fraction at point={frac:.3f}\n")

        summary.append({"theta": theta, "gamma_eff_point": g_point,
                        "boot_ci_low": lo, "boot_median": med, "boot_ci_high": hi,
                        "fraction_at_point": frac, "n_boot": N_BOOT, "seed": SEED})
        vc = pd.Series(np.round(boot, 3)).value_counts(normalize=True).sort_index()
        for g, f in vc.items():
            dist.append({"theta": theta, "gamma_eff": g, "fraction": f})

    pd.DataFrame(summary).to_csv(os.path.join(OUTPUT_DIR, "gamma_eff_bootstrap_summary.csv"), index=False)
    pd.DataFrame(dist).to_csv(os.path.join(OUTPUT_DIR, "gamma_eff_bootstrap_distribution.csv"), index=False)
    pd.DataFrame(resid_rows).to_csv(os.path.join(OUTPUT_DIR, "gamma_eff_residuals.csv"), index=False)

    logging.info("Saved gamma_eff_bootstrap_summary.csv, gamma_eff_bootstrap_distribution.csv, "
                 "gamma_eff_residuals.csv in output/")
    logging.info("Table II consistency: " + ("PASS" if all_pass else "MISMATCH, do not use these numbers"))


if __name__ == "__main__":
    main()
