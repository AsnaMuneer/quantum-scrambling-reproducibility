"""
hardware_validation_v2_saved.py

Saved-counts analysis: point estimates, bootstrap CIs, and gamma_eff from
output/tomography_raw_counts.json. No hardware. Imports the analysis functions
from hardware_validation_v2 so the physics is identical to the hardware path.
"""

import os
import json
import logging
import numpy as np
import pandas as pd

from hardware_validation_v2 import (
    THETA_VALUES,
    OUTPUT_DIR,
    reconstruct_density_matrix,
    project_to_physical,
    compute_metrics,
    gamma_dense_sweep,
    estimate_gamma_per_metric,
    estimate_gamma_joint,
    bootstrap_counts,
    plot_hardware_vs_theta,
    log_ci_widths,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")

COUNTS_JSON = os.path.join(OUTPUT_DIR, "tomography_raw_counts.json")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "hardware_validation_v2_saved_results.csv")
FIGURE_NAME = "hardware_validation_v2_saved_theta_sweep.png"
N_BOOTSTRAP = 2000
SEED = 12345
METRICS = ["fidelity", "purity", "concurrence", "trace_distance"]


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


def analyse(counts_by_setting, rng):
    shots_set = {sum(c.values()) for c in counts_by_setting.values()}
    if len(shots_set) != 1:
        raise ValueError(f"Unequal shots per setting: {shots_set}")
    shots = shots_set.pop()

    point = compute_metrics(project_to_physical(reconstruct_density_matrix(counts_by_setting, shots)))
    point = {k: float(point[k]) for k in METRICS}

    boot = {k: [] for k in METRICS}
    for _ in range(N_BOOTSTRAP):
        rs = {s: bootstrap_counts(c, shots, rng) for s, c in counts_by_setting.items()}
        m = compute_metrics(project_to_physical(reconstruct_density_matrix(rs, shots)))
        for k in METRICS:
            boot[k].append(float(m[k]))

    ci = {}
    for k in METRICS:
        v = np.array(boot[k])
        ci[k] = {"mean": float(v.mean()), "std": float(v.std()),
                 "ci_low": float(np.percentile(v, 2.5)),
                 "ci_high": float(np.percentile(v, 97.5))}
    return point, ci, shots


def main():
    logging.info("\n=========================================")
    logging.info(" Hardware Benchmarking v2 -- SAVED COUNTS")
    logging.info("=========================================\n")

    saved = load_counts(COUNTS_JSON)
    logging.info(f"Loaded {len(saved)} theta values from {COUNTS_JSON}\n")
    rng = np.random.default_rng(SEED)
    rows = []

    for theta in THETA_VALUES:
        logging.info(f"--- theta = {theta:.4f} rad ---")
        counts = lookup(saved, theta)
        point, ci, shots = analyse(counts, rng)
        logging.info(f"  Shots per setting: {shots}")
        logging.info(f"  Point estimate: { {k: round(v, 4) for k, v in point.items()} }")
        for k, v in ci.items():
            logging.info(f"  {k}: mean={v['mean']:.4f}  95% CI=[{v['ci_low']:.4f}, {v['ci_high']:.4f}]")

        sweep = gamma_dense_sweep(theta)
        per = {m: estimate_gamma_per_metric(point[m], sweep, m) for m in METRICS}
        joint = estimate_gamma_joint(point, sweep)
        logging.info(f"  gamma_eff (per-metric): { {k: round(v, 5) for k, v in per.items()} }")
        logging.info(f"  gamma_eff (joint): {joint['gamma_eff_joint']:.5f}\n")

        rows.append({
            "theta": theta,
            "shots_per_setting": shots,
            **{f"hw_{k}": point[k] for k in METRICS},
            **{f"hw_{k}_ci_low": ci[k]["ci_low"] for k in METRICS},
            **{f"hw_{k}_ci_high": ci[k]["ci_high"] for k in METRICS},
            **{f"gamma_eff_{k}": per[k] for k in METRICS},
            **joint,
            "counts_source": COUNTS_JSON,
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    logging.info(f"Results saved: {RESULTS_CSV}")
    out = plot_hardware_vs_theta(df)
    logging.info(f"Figure generated: {out}")
    log_ci_widths(df)


if __name__ == "__main__":
    main()
