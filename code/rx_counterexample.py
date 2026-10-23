"""
rx_counterexample.py

Numerical check of the R_x(theta) sensitivity estimate in Section III-B
(Reviewer 1, comment 7). Simulates the Bell preparation followed by a local
rotation R_x(theta) on qubit 0, under the composite noise model with
depolarizing gamma and thermal relaxation, and reports purity and concurrence
across theta at two gate durations:
  - 50 ns (the paper's parameter, where the analytic estimate is ~7e-4 per gate)
  - 1 us  (a duration comparable to T2, where a visible effect is expected)

Simulation only; no hardware. Writes output/rx_counterexample.csv.
"""

import os
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import DensityMatrix, purity, concurrence
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

THETA = Parameter("theta")
THETA_VALUES = np.linspace(0, np.pi, 41)
GAMMA_VALUES = [0.00, 0.05]
T1, T2 = 50e-6, 30e-6
DURATIONS = {"50 ns": 50e-9, "1 us": 1e-6}
GATE_TIME_2Q = 300e-9


def noise_model(gamma, duration_1q):
    e1 = depolarizing_error(gamma, 1).compose(
        thermal_relaxation_error(T1, T2, duration_1q))
    e2 = depolarizing_error(gamma, 2).compose(
        thermal_relaxation_error(T1, T2, GATE_TIME_2Q).tensor(
            thermal_relaxation_error(T1, T2, GATE_TIME_2Q)))
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(e1, ["h", "rx", "rz"])
    nm.add_all_qubit_quantum_error(e2, ["cx"])
    return nm


def build_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rx(THETA, 0)
    qc.save_density_matrix()
    return qc


def main():
    base = build_circuit()
    rows = []
    for dur_label, dur in DURATIONS.items():
        for gamma in GAMMA_VALUES:
            sim = AerSimulator(method="density_matrix",
                               noise_model=noise_model(gamma, dur))
            compiled = transpile(base, sim, optimization_level=1)
            for th in THETA_VALUES:
                bound = compiled.assign_parameters({THETA: th})
                rho = sim.run(bound).result().data(0)["density_matrix"]
                dm = DensityMatrix(rho)
                rows.append({
                    "duration": dur_label,
                    "gamma": gamma,
                    "theta": th,
                    "purity": float(np.real_if_close(purity(dm))),
                    "concurrence": float(concurrence(dm)),
                })
            print(f"done: duration={dur_label}, gamma={gamma}")

    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "rx_counterexample.csv")
    df.to_csv(path, index=False)

    print("\nVariation across theta (max - min), per duration and gamma:")
    spread = (df.groupby(["duration", "gamma"])[["purity", "concurrence"]]
              .agg(lambda x: x.max() - x.min()))
    print(spread.to_string())
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
