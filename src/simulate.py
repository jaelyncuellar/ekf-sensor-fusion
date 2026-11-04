"""Synthetic trajectory and GPS/IMU sensor simulation for Project 2.

The generator follows the Project 2 starter-sheet decision: synthetic
first, with known ground truth and no dataset download. A moving platform
follows a scripted route containing acceleration, braking, left and right
turns, and an S-turn. The IMU runs at 20 Hz with constant biases plus
white noise. GPS runs at 1 Hz with 2.5 m per-axis noise and has a scripted
15-second outage, which is where fusion matters most: dead reckoning
drifts, GPS-only output freezes, and the EKF coasts on the IMU before
re-converging when fixes return.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ekf import motion_step

DURATION_S = 120.0
DT_S = 0.05  # 20 Hz IMU / integration rate
GPS_PERIOD_S = 1.0
GPS_SIGMA_M = 2.5
GPS_OUTAGE_S = (72.0, 87.0)

ACCEL_BIAS_MPS2 = 0.06
GYRO_BIAS_RADPS = 0.004
ACCEL_NOISE_SIGMA_MPS2 = 0.15
GYRO_NOISE_SIGMA_RADPS = 0.010

# time_end, longitudinal acceleration, yaw rate, plain-language description
SEGMENTS = [
    (15.0, 8.0 / 15.0, 0.00, "accelerate 0 to 8 m/s, straight"),
    (30.0, 0.00, 0.07, "constant speed, left turn"),
    (45.0, -4.0 / 15.0, 0.00, "brake 8 to 4 m/s, straight"),
    (55.0, 0.00, 0.09, "left half of an S-turn at 4 m/s"),
    (65.0, 0.00, -0.09, "right half of an S-turn at 4 m/s"),
    (80.0, 6.0 / 15.0, 0.00, "accelerate 4 to 10 m/s, straight"),
    (95.0, 0.00, -0.06, "constant speed, right turn"),
    (110.0, -7.0 / 15.0, 0.00, "brake 10 to 3 m/s, straight"),
    (120.0, 0.00, 0.04, "gentle left turn at 3 m/s"),
]


def true_control(time_s: float) -> tuple[float, float]:
    """Return the scripted (acceleration, yaw rate) at a time."""
    for end, acceleration, yaw_rate, _description in SEGMENTS:
        if time_s < end - 1e-12:
            return acceleration, yaw_rate
    return 0.0, 0.0


def generate_dataset(
    seed: int = 42,
    duration_s: float = DURATION_S,
    dt_s: float = DT_S,
    gps_period_s: float = GPS_PERIOD_S,
    gps_sigma_m: float = GPS_SIGMA_M,
    outage_s: tuple[float, float] = GPS_OUTAGE_S,
    accel_bias_mps2: float = ACCEL_BIAS_MPS2,
    gyro_bias_radps: float = GYRO_BIAS_RADPS,
    accel_noise_sigma_mps2: float = ACCEL_NOISE_SIGMA_MPS2,
    gyro_noise_sigma_radps: float = GYRO_NOISE_SIGMA_RADPS,
) -> dict:
    """Generate ground truth, noisy IMU controls, and sparse GPS fixes."""
    rng = np.random.default_rng(seed)
    steps = int(round(duration_s / dt_s))
    gps_step_period = int(round(gps_period_s / dt_s))

    times = np.arange(steps + 1, dtype=float) * dt_s
    states = np.zeros((steps + 1, 5), dtype=float)
    true_controls = np.zeros((steps, 2), dtype=float)

    for i in range(steps):
        acceleration, yaw_rate = true_control(times[i])
        true_controls[i] = (acceleration, yaw_rate)
        states[i + 1] = motion_step(states[i], acceleration, yaw_rate, dt_s)

    imu_controls = np.column_stack(
        [
            true_controls[:, 0]
            + accel_bias_mps2
            + rng.normal(0.0, accel_noise_sigma_mps2, steps),
            true_controls[:, 1]
            + gyro_bias_radps
            + rng.normal(0.0, gyro_noise_sigma_radps, steps),
        ]
    )

    gps: dict[int, np.ndarray] = {}
    for state_index in range(0, steps + 1, gps_step_period):
        t = times[state_index]
        if outage_s[0] <= t < outage_s[1]:
            continue
        gps[state_index] = states[state_index, :2] + rng.normal(
            0.0, gps_sigma_m, 2
        )

    return {
        "times": times,
        "states": states,
        "true_controls": true_controls,
        "imu_controls": imu_controls,
        "gps": gps,
        "dt": dt_s,
        "gps_sigma": gps_sigma_m,
        "outage": outage_s,
        "seed": seed,
    }


def write_dataset(data: dict, data_dir: str | Path) -> None:
    """Write ground-truth, IMU, and GPS CSV files."""
    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    times = data["times"]
    states = data["states"]

    with open(out / "ground_truth.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["time_s", "x_m", "y_m", "vx_mps", "vy_mps", "heading_rad", "speed_mps"]
        )
        for t, state in zip(times, states):
            writer.writerow(
                [
                    f"{t:.2f}",
                    *[f"{value:.6f}" for value in state],
                    f"{np.hypot(state[2], state[3]):.6f}",
                ]
            )

    with open(out / "imu.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["time_s", "true_accel_mps2", "true_yaw_rate_radps", "measured_accel_mps2", "measured_yaw_rate_radps"]
        )
        for i, t in enumerate(times[:-1]):
            writer.writerow(
                [
                    f"{t:.2f}",
                    f"{data['true_controls'][i, 0]:.6f}",
                    f"{data['true_controls'][i, 1]:.6f}",
                    f"{data['imu_controls'][i, 0]:.6f}",
                    f"{data['imu_controls'][i, 1]:.6f}",
                ]
            )

    with open(out / "gps.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "x_m", "y_m"])
        for state_index in sorted(data["gps"]):
            z = data["gps"][state_index]
            writer.writerow(
                [f"{times[state_index]:.2f}", f"{z[0]:.6f}", f"{z[1]:.6f}"]
            )


if __name__ == "__main__":
    dataset = generate_dataset()
    write_dataset(dataset, Path(__file__).resolve().parent.parent / "data")
    print(
        f"wrote {len(dataset['times'])} truth samples, "
        f"{len(dataset['imu_controls'])} IMU samples, and "
        f"{len(dataset['gps'])} GPS fixes"
    )
