"""Run Project 2 end to end: simulate, fuse, compare, plot, and report.

Three estimators see the same trip:

* **GPS only** — the newest GPS fix, held unchanged between fixes. During
  the outage it simply freezes at the last received position.
* **IMU only / dead reckoning** — starts from the same initial estimate
  as the EKF, then integrates the noisy IMU controls with no corrections.
* **EKF fusion** — predicts with every IMU sample and corrects with every
  GPS fix. During the outage it coasts on the IMU; when GPS returns, the
  position update pulls it back.

Outputs (default ``out/`` directory):

* ``estimates.csv`` — truth and all three estimates at 20 Hz
* ``metrics.json`` / ``metrics.txt`` — RMSE and error summaries
* ``trajectory.png`` — ground truth vs. GPS points vs. EKF vs. dead reckoning
* ``position_errors.png`` — position error over time, outage shaded
* ``state_errors.png`` — heading and speed errors over time
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ekf import ExtendedKalmanFilter, motion_step, wrap_angle  # noqa: E402
from simulate import generate_dataset, write_dataset  # noqa: E402

# EKF tuning for the simulated sensors. The process noise is per 0.05 s
# prediction step and accounts for IMU white noise plus the unmodelled
# constant IMU biases (the filter deliberately does not estimate bias
# states in this project). It was selected from a small documented sweep
# against the seeded ground truth; the GPS measurement variance stays at
# the simulator's true value (2.5 m per axis -> 6.25 m^2).
PROCESS_NOISE = np.diag([0.004, 0.004, 0.004, 0.004, 0.00004])
INITIAL_COVARIANCE = np.diag([6.25, 6.25, 0.25, 0.25, 0.02])


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def _angle_errors(estimated: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.array([wrap_angle(a - b) for a, b in zip(estimated, truth)])


def run_simulation(seed: int = 42) -> dict:
    """Run all three estimators and return data, estimates, and metrics."""
    data = generate_dataset(seed=seed)
    times: np.ndarray = data["times"]
    truth: np.ndarray = data["states"]
    imu: np.ndarray = data["imu_controls"]
    gps: dict[int, np.ndarray] = data["gps"]
    dt = float(data["dt"])

    if 0 not in gps:
        raise RuntimeError("the simulator must provide an initial GPS fix")

    initial_state = np.array(
        [gps[0][0], gps[0][1], 0.0, 0.0, 0.0], dtype=float
    )
    ekf = ExtendedKalmanFilter(
        initial_state=initial_state,
        initial_covariance=INITIAL_COVARIANCE,
        process_noise=PROCESS_NOISE,
        gps_variance=float(data["gps_sigma"]) ** 2,
    )

    n = len(times)
    ekf_states = np.zeros_like(truth)
    dr_states = np.zeros_like(truth)
    gps_hold = np.zeros((n, 2), dtype=float)
    ekf_sigma = np.zeros(n, dtype=float)

    ekf_states[0] = ekf.state
    dr_states[0] = initial_state
    gps_hold[0] = gps[0]
    ekf_sigma[0] = ekf.position_sigma
    last_fix = gps[0]

    for i in range(n - 1):
        acceleration, yaw_rate = imu[i]
        ekf.predict(float(acceleration), float(yaw_rate), dt)
        dr_states[i + 1] = motion_step(
            dr_states[i], float(acceleration), float(yaw_rate), dt
        )

        if i + 1 in gps:
            ekf.update_gps(gps[i + 1])
            last_fix = gps[i + 1]

        ekf_states[i + 1] = ekf.state
        gps_hold[i + 1] = last_fix
        ekf_sigma[i + 1] = ekf.position_sigma

    truth_xy = truth[:, :2]
    ekf_error = np.linalg.norm(ekf_states[:, :2] - truth_xy, axis=1)
    dr_error = np.linalg.norm(dr_states[:, :2] - truth_xy, axis=1)
    gps_hold_error = np.linalg.norm(gps_hold - truth_xy, axis=1)

    gps_indices = np.array(sorted(gps), dtype=int)
    gps_raw_error = np.array(
        [np.linalg.norm(gps[i] - truth_xy[i]) for i in gps_indices]
    )

    outage_start, outage_end = data["outage"]
    in_outage = (times >= outage_start) & (times < outage_end)
    after_outage = (times >= outage_end) & (times < outage_end + 5.0)

    ekf_heading_error = _angle_errors(ekf_states[:, 4], truth[:, 4])
    dr_heading_error = _angle_errors(dr_states[:, 4], truth[:, 4])
    ekf_speed_error = np.hypot(ekf_states[:, 2], ekf_states[:, 3]) - np.hypot(
        truth[:, 2], truth[:, 3]
    )
    dr_speed_error = np.hypot(dr_states[:, 2], dr_states[:, 3]) - np.hypot(
        truth[:, 2], truth[:, 3]
    )

    metrics = {
        "seed": seed,
        "duration_s": float(times[-1]),
        "samples_20hz": int(n),
        "gps_fixes": int(len(gps)),
        "gps_outage_s": [float(outage_start), float(outage_end)],
        "ekf_position_rmse_m": _rmse(ekf_error),
        "ekf_position_max_error_m": float(np.max(ekf_error)),
        "ekf_final_position_error_m": float(ekf_error[-1]),
        "ekf_outage_rmse_m": _rmse(ekf_error[in_outage]),
        "ekf_outage_max_error_m": float(np.max(ekf_error[in_outage])),
        "ekf_first_5s_after_outage_rmse_m": _rmse(ekf_error[after_outage]),
        "gps_raw_rmse_at_fixes_m": _rmse(gps_raw_error),
        "gps_hold_position_rmse_m": _rmse(gps_hold_error),
        "gps_hold_final_position_error_m": float(gps_hold_error[-1]),
        "dead_reckoning_position_rmse_m": _rmse(dr_error),
        "dead_reckoning_final_position_error_m": float(dr_error[-1]),
        "ekf_heading_rmse_deg": float(np.degrees(_rmse(ekf_heading_error))),
        "dead_reckoning_heading_rmse_deg": float(
            np.degrees(_rmse(dr_heading_error))
        ),
        "ekf_speed_rmse_mps": _rmse(ekf_speed_error),
        "dead_reckoning_speed_rmse_mps": _rmse(dr_speed_error),
    }
    metrics["ekf_rmse_improvement_vs_gps_raw_pct"] = float(
        100.0
        * (metrics["gps_raw_rmse_at_fixes_m"] - metrics["ekf_position_rmse_m"])
        / metrics["gps_raw_rmse_at_fixes_m"]
    )
    metrics["ekf_rmse_improvement_vs_gps_hold_pct"] = float(
        100.0
        * (
            metrics["gps_hold_position_rmse_m"]
            - metrics["ekf_position_rmse_m"]
        )
        / metrics["gps_hold_position_rmse_m"]
    )
    metrics["ekf_rmse_improvement_vs_dead_reckoning_pct"] = float(
        100.0
        * (
            metrics["dead_reckoning_position_rmse_m"]
            - metrics["ekf_position_rmse_m"]
        )
        / metrics["dead_reckoning_position_rmse_m"]
    )

    estimates = {
        "ekf": ekf_states,
        "dead_reckoning": dr_states,
        "gps_hold": gps_hold,
        "ekf_position_error": ekf_error,
        "dead_reckoning_position_error": dr_error,
        "gps_hold_position_error": gps_hold_error,
        "ekf_heading_error_rad": ekf_heading_error,
        "dead_reckoning_heading_error_rad": dr_heading_error,
        "ekf_speed_error": ekf_speed_error,
        "dead_reckoning_speed_error": dr_speed_error,
        "ekf_position_sigma": ekf_sigma,
    }
    return {"data": data, "estimates": estimates, "metrics": metrics}


def write_estimates(result: dict, path: Path) -> None:
    data, estimates = result["data"], result["estimates"]
    times, truth = data["times"], data["states"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time_s",
                "truth_x_m", "truth_y_m", "truth_vx_mps", "truth_vy_mps", "truth_heading_rad",
                "gps_hold_x_m", "gps_hold_y_m",
                "dr_x_m", "dr_y_m", "dr_vx_mps", "dr_vy_mps", "dr_heading_rad",
                "ekf_x_m", "ekf_y_m", "ekf_vx_mps", "ekf_vy_mps", "ekf_heading_rad",
                "ekf_position_error_m", "dr_position_error_m", "gps_hold_position_error_m",
                "ekf_position_sigma_m",
            ]
        )
        for i, t in enumerate(times):
            writer.writerow(
                [
                    f"{t:.2f}",
                    *[f"{v:.6f}" for v in truth[i]],
                    *[f"{v:.6f}" for v in estimates["gps_hold"][i]],
                    *[f"{v:.6f}" for v in estimates["dead_reckoning"][i]],
                    *[f"{v:.6f}" for v in estimates["ekf"][i]],
                    f"{estimates['ekf_position_error'][i]:.6f}",
                    f"{estimates['dead_reckoning_position_error'][i]:.6f}",
                    f"{estimates['gps_hold_position_error'][i]:.6f}",
                    f"{estimates['ekf_position_sigma'][i]:.6f}",
                ]
            )


def write_metrics(result: dict, out_dir: Path) -> None:
    metrics = result["metrics"]
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    lines = [
        "Project 2 - EKF IMU + GPS fusion results (synthetic data, seed "
        f"{metrics['seed']})",
        f"duration: {metrics['duration_s']:.0f} s at 20 Hz "
        f"({metrics['samples_20hz']} samples), {metrics['gps_fixes']} GPS fixes",
        f"GPS outage: {metrics['gps_outage_s'][0]:.0f}-{metrics['gps_outage_s'][1]:.0f} s",
        "",
        f"EKF position RMSE:              {metrics['ekf_position_rmse_m']:.2f} m",
        f"GPS raw RMSE at fix times:      {metrics['gps_raw_rmse_at_fixes_m']:.2f} m",
        f"GPS hold position RMSE:         {metrics['gps_hold_position_rmse_m']:.2f} m",
        f"Dead-reckoning position RMSE:   {metrics['dead_reckoning_position_rmse_m']:.2f} m",
        "",
        f"EKF max position error:         {metrics['ekf_position_max_error_m']:.2f} m",
        f"EKF error during GPS outage:    RMSE {metrics['ekf_outage_rmse_m']:.2f} m, "
        f"max {metrics['ekf_outage_max_error_m']:.2f} m",
        f"EKF RMSE, first 5 s after GPS returns: "
        f"{metrics['ekf_first_5s_after_outage_rmse_m']:.2f} m",
        f"EKF final position error:       {metrics['ekf_final_position_error_m']:.2f} m",
        f"Dead-reckoning final error:     {metrics['dead_reckoning_final_position_error_m']:.2f} m",
        "",
        f"EKF heading RMSE:               {metrics['ekf_heading_rmse_deg']:.2f} deg",
        f"Dead-reckoning heading RMSE:    {metrics['dead_reckoning_heading_rmse_deg']:.2f} deg",
        f"EKF speed RMSE:                 {metrics['ekf_speed_rmse_mps']:.2f} m/s",
        f"Dead-reckoning speed RMSE:      {metrics['dead_reckoning_speed_rmse_mps']:.2f} m/s",
        "",
        f"EKF RMSE improvement vs GPS raw:        "
        f"{metrics['ekf_rmse_improvement_vs_gps_raw_pct']:.1f}%",
        f"EKF RMSE improvement vs GPS hold:       "
        f"{metrics['ekf_rmse_improvement_vs_gps_hold_pct']:.1f}%",
        f"EKF RMSE improvement vs dead reckoning: "
        f"{metrics['ekf_rmse_improvement_vs_dead_reckoning_pct']:.1f}%",
    ]
    (out_dir / "metrics.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def make_plots(result: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data, estimates, metrics = result["data"], result["estimates"], result["metrics"]
    times, truth = data["times"], data["states"]
    gps = data["gps"]
    outage_start, outage_end = data["outage"]
    gps_idx = np.array(sorted(gps))
    gps_xy = np.array([gps[i] for i in gps_idx])

    # --- trajectory -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(truth[:, 0], truth[:, 1], color="#222222", lw=2, label="Ground truth")
    ax.plot(
        estimates["ekf"][:, 0], estimates["ekf"][:, 1],
        color="#0f766e", lw=1.6, label="EKF fused",
    )
    ax.plot(
        estimates["dead_reckoning"][:, 0], estimates["dead_reckoning"][:, 1],
        color="#d97706", lw=1.2, ls="--", label="IMU only (dead reckoning)",
    )
    ax.scatter(
        gps_xy[:, 0], gps_xy[:, 1], s=10, color="#dc2626", alpha=0.55,
        label="GPS fixes (1 Hz)", zorder=5,
    )
    ax.scatter([truth[0, 0]], [truth[0, 1]], s=70, marker="o",
               color="#16a34a", label="Start", zorder=6)
    ax.scatter([truth[-1, 0]], [truth[-1, 1]], s=80, marker="X",
               color="#111827", label="Finish (truth)", zorder=6)
    ax.set_xlabel("x (m, east)")
    ax.set_ylabel("y (m, north)")
    ax.set_title("Project 2 - ground truth vs. GPS, EKF, and dead reckoning")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory.png", dpi=160)
    plt.close(fig)

    # --- position errors -------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax in axes:
        ax.axvspan(outage_start, outage_end, color="#fca5a5", alpha=0.35)
    axes[0].plot(times, estimates["ekf_position_error"], color="#0f766e",
                 lw=1.4, label="EKF fused")
    axes[0].plot(times, estimates["gps_hold_position_error"], color="#dc2626",
                 lw=1.0, alpha=0.8, label="GPS only (last fix held)")
    axes[0].set_ylabel("position error (m)")
    axes[0].set_title("Position error over time (pink band = GPS outage, 72-87 s)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].plot(times, estimates["dead_reckoning_position_error"],
                 color="#d97706", lw=1.2, label="IMU only (dead reckoning)")
    axes[1].set_ylabel("position error (m)")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "position_errors.png", dpi=160)
    plt.close(fig)

    # --- heading / speed errors ------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax in axes:
        ax.axvspan(outage_start, outage_end, color="#fca5a5", alpha=0.35)
    axes[0].plot(
        times, np.degrees(np.abs(estimates["ekf_heading_error_rad"])),
        color="#0f766e", lw=1.3, label="EKF fused",
    )
    axes[0].plot(
        times, np.degrees(np.abs(estimates["dead_reckoning_heading_error_rad"])),
        color="#d97706", lw=1.1, label="IMU only",
    )
    axes[0].set_ylabel("heading error (deg)")
    axes[0].set_title("Heading and speed errors (pink band = GPS outage)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].plot(times, np.abs(estimates["ekf_speed_error"]), color="#0f766e",
                 lw=1.3, label="EKF fused")
    axes[1].plot(times, np.abs(estimates["dead_reckoning_speed_error"]),
                 color="#d97706", lw=1.1, label="IMU only")
    axes[1].set_ylabel("speed error (m/s)")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "state_errors.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    args = parser.parse_args()

    result = run_simulation(seed=args.seed)
    data_dir = args.project_dir / "data"
    out_dir = args.project_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_dataset(result["data"], data_dir)
    write_estimates(result, out_dir / "estimates.csv")
    write_metrics(result, out_dir)
    make_plots(result, out_dir)
    print(f"\noutputs in {out_dir}: estimates.csv, metrics.txt/json, "
          "trajectory.png, position_errors.png, state_errors.png")


if __name__ == "__main__":
    main()
