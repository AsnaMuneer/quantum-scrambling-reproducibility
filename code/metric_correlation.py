"""
metric_correlation.py

Correlation matrix among the four metrics across bootstrap resamples (Reviewer 2,
comment 10). Self-contained: uses only output/tomography_raw_counts.json and
numpy/pandas. No hardware, no simulator.

For each scrambling angle, for each of N_BOOT bootstrap resamples of the raw
counts (same resampling as the main analysis), the four metrics are recomputed.
We then report:
  - the 4x4 Pearson correlation matrix across resamples, per angle
  - the per-angle standard deviation of each metric
  - the raw resampled values (so any other statistic can be recomputed)

Outputs (in output/):
  metric_bootstrap_samples.csv        one row per (theta, resample)
  metric_correlation_matrix.csv       one row per (theta, metric pair)
"""

import os
import json
import logging
import itertools
import numpy as np
import pandas as pd

from qiskit.quantum_info import DensityMatrix, state_fidelity, purity, concurrence
from qiskit import QuantumCircuit

logging.basicConfig(level=logging.INFO, format="%(message)s")

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
COUNTS_JSON = os.path.join(OUTPUT_DIR, "tomography_raw_counts.json")

THETA_VALUES = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
N_BOOT = 2000
SEED = 12345
METRICS = ["fidelity", "purity", "concurrence", "trace_distance"]
BASES = ["X", "Y", "Z"]
BASIS_SETTINGS = list(itertools.product(BASES, repeat=2))

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


def main():
    logging.info("Metric correlation across bootstrap resamples (saved counts, no hardware)\n")
    saved = load_counts(COUNTS_JSON)
    rng = np.random.default_rng(SEED)

    sample_rows = []
    corr_rows = []

    for theta in THETA_VALUES:
        logging.info(f"--- theta = {theta:.4f} ---")
        counts = lookup(saved, theta)
        shots = {sum(c.values()) for c in counts.values()}
        assert len(shots) == 1, f"unequal shots: {shots}"
        shots = shots.pop()

        values = np.empty((N_BOOT, len(METRICS)))
        for b in range(N_BOOT):
            rs = {s: bootstrap_counts(c, shots, rng) for s, c in counts.items()}
            m = metrics_of(project(reconstruct(rs, shots)))
            values[b, :] = [m[k] for k in METRICS]
            sample_rows.append({"theta": theta, "resample": b,
                                **{k: values[b, i] for i, k in enumerate(METRICS)}})

        corr = np.corrcoef(values, rowvar=False)
        sd = values.std(axis=0, ddof=1)
        logging.info("  Pearson correlation across resamples:")
        logging.info(pd.DataFrame(corr, index=METRICS, columns=METRICS).round(3).to_string())
        logging.info("  standard deviation across resamples: "
                     + ", ".join(f"{k}={s:.5f}" for k, s in zip(METRICS, sd)))
        logging.info("")

        for i, mi in enumerate(METRICS):
            for j, mj in enumerate(METRICS):
                if j < i:
                    continue
                corr_rows.append({"theta": theta, "metric_a": mi, "metric_b": mj,
                                  "pearson_r": corr[i, j],
                                  "sd_a": sd[i], "sd_b": sd[j],
                                  "n_boot": N_BOOT, "seed": SEED})

    pd.DataFrame(sample_rows).to_csv(os.path.join(OUTPUT_DIR, "metric_bootstrap_samples.csv"), index=False)
    pd.DataFrame(corr_rows).to_csv(os.path.join(OUTPUT_DIR, "metric_correlation_matrix.csv"), index=False)
    logging.info("Saved output/metric_bootstrap_samples.csv and output/metric_correlation_matrix.csv")


if __name__ == "__main__":
    main()
