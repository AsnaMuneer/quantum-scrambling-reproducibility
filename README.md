# quantum-scrambling-reproducibility
Reproducibility code and data for “Comparative Evaluation of Quantum-State Metrics for Local and Correlated Unitary Scrambling Under NISQ Noise” (IEEE Access, Access-2026-43146). Simulations in Qiskit Aer, saved IBM Quantum tomography counts, and scripts to regenerate every table and figure.

This repository contains the code, saved hardware data and analysis outputs behind the manuscript Comparative Evaluation of Quantum-State Metrics for Local and Correlated Unitary Scrambling Under NISQ Noise.

The paper studies how fidelity, purity, concurrence and trace distance respond to coherent unitary scrambling of a Bell state under a composite depolarizing and thermal-relaxation noise model. It proves that local single-qubit phase scrambling leaves purity and concurrence invariant, and compares this with a correlated two-qubit perturbation.

Every simulation result can be reproduced without hardware or an IBM Quantum account. The hardware results (Tables 2 and 3, Figure 7) are reproduced from the saved tomography counts of a single run on IBM’s ibm_marrakesh backend, qubits 1–2. Run order and software versions are given here.

QUANTUM-STATE METRICS UNDER UNITARY SCRAMBLING AND NISQ NOISE
Reproducibility package for the IEEE Access manuscript (Access-2026-43146)

1. ENVIRONMENT
--------------
Python:        [run: python3 --version]
qiskit:        [run: pip3 show qiskit]
qiskit-aer:    [run: pip3 show qiskit-aer]
numpy, pandas, matplotlib: [run: pip3 show numpy pandas matplotlib]

Install the packages with:
    pip3 install qiskit qiskit-aer numpy pandas matplotlib

No IBM Quantum account or hardware access is needed to reproduce any result
in the paper. The hardware data are saved as data/tomography_raw_counts.json.

2. FOLDER LAYOUT
----------------
code/      Python scripts. Run them from the folder that contains code/ and
           data/, or copy them there. Scripts read and write output/.
data/      Saved inputs: the hardware tomography counts and the calibration
           snapshot from the single ibm_marrakesh run (22 August 2026,
           qubits 1-2, 4096 shots per setting).
outputs/   The result files the paper cites. Use these to check your run.

3. HOW TO REPRODUCE
-------------------
Create an output/ folder, copy data/tomography_raw_counts.json into it, then
run the scripts in this order from the folder that contains code/:

    mkdir -p output
    cp data/tomography_raw_counts.json output/
    python3 code/simulate_v5.py cx
    python3 code/simulate_v5.py cz
    python3 code/simulate_v5_padded.py
    python3 code/simulate_correlated.py
    python3 code/simulate_confound_control.py
    python3 code/rx_counterexample.py
    python3 code/hardware_validation_v2_saved.py
    python3 code/gamma_eff_uncertainty.py
    python3 code/metric_correlation.py
    python3 code/tomography_physicality.py   (MODE = "from_saved")

4. WHICH OUTPUT SUPPORTS WHICH RESULT
-------------------------------------
Table 2 and Table 3 (hardware point estimates and bootstrap intervals):
    outputs/hardware_validation_v2_saved_results.csv
Physicality cross-check (negative eigenvalues, MLE comparison):
    outputs/tomography_physicality.csv
Table (gamma_eff uncertainty) and Table S1 (residuals):
    outputs/gamma_eff_bootstrap_summary.csv, outputs/gamma_eff_residuals.csv
Table S2 (pairwise metric correlations):
    outputs/metric_correlation_matrix.csv
Native-gate comparison (CX, CZ, padded CX):
    outputs/fidelity_results_cx.csv, _cz.csv, _cx_padded.csv
Local versus correlated scrambling:
    outputs/correlated_results.csv
Duration decomposition (Table 5):
    outputs/confound_decomposition.csv
R_x(theta) sensitivity check:
    outputs/rx_counterexample.csv
Figure 7 (hardware sweep):
    outputs/hardware_validation_v2_saved_theta_sweep.png

5. SETTINGS USED
----------------
Bootstrap:     2000 resamples, seed 12345 (all bootstrap analyses)
Shots:         4096 per tomography setting
Transpilation: optimisation level 1, Aer default basis, no custom basis
Noise:         T1 = 50 us, T2 = 30 us, gate times 50 ns (1q) and 300 ns (2q)
Gamma grid:    26 points on [0, 0.05], step 0.002
Angles:        41 points on [0, pi] for simulations; 5 angles for hardware

6. NOTES
--------
Hardware run: single backend (ibm_marrakesh), single qubit pair (1-2), single
calibration snapshot. A second backend was not run.
Simulations use the composite noise model described in the manuscript. The
gamma_eff values are phenomenological fit parameters, not device rates.
