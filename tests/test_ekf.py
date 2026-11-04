"""Unit and integration tests for Project 2.

Run from the project root with the standard library only:

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ekf import (  # noqa: E402
    ExtendedKalmanFilter,
    motion_jacobian,
    motion_step,
    wrap_angle,
)
from run_fusion import PROCESS_NOISE, run_simulation  # noqa: E402
from simulate import generate_dataset  # noqa: E402


def finite_difference_jacobian(state, acceleration, yaw_rate, dt, eps=1e-6):
    numeric = np.zeros((5, 5))
    for j in range(5):
        plus = np.array(state, dtype=float)
        minus = np.array(state, dtype=float)
        plus[j] += eps
        minus[j] -= eps
        diff = motion_step(plus, acceleration, yaw_rate, dt) - motion_step(
            minus, acceleration, yaw_rate, dt
        )
        diff[4] = wrap_angle(diff[4])
        numeric[:, j] = diff / (2.0 * eps)
    return numeric


class TestMotionModel(unittest.TestCase):
    def test_wrap_angle(self):
        self.assertAlmostEqual(wrap_angle(0.25), 0.25)
        self.assertAlmostEqual(wrap_angle(np.pi * 1.5), -np.pi * 0.5)
        self.assertAlmostEqual(wrap_angle(-np.pi * 1.5), np.pi * 0.5)

    def test_jacobian_matches_finite_differences(self):
        cases = [
            (np.array([10.0, 20.0, 3.0, 1.0, 0.30]), 0.40, 0.05),
            (np.array([-5.0, 2.0, -2.0, 2.0, -0.50]), -0.10, -0.04),
            (np.array([100.0, -40.0, 0.5, -4.0, -1.20]), 0.00, 0.09),
            (np.array([0.0, 0.0, 5.0, 5.0, 0.75]), -0.30, -0.06),
        ]
        for state, acceleration, yaw_rate in cases:
            analytic = motion_jacobian(state, acceleration, yaw_rate, 0.05)
            numeric = finite_difference_jacobian(
                state, acceleration, yaw_rate, 0.05
            )
            np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-5)

    def test_rest_state_step_is_finite(self):
        stepped = motion_step(np.zeros(5), 0.5, 0.1, 0.05)
        self.assertTrue(np.all(np.isfinite(stepped)))
        self.assertGreater(stepped[2], 0.0)  # starts moving east


class TestSimulation(unittest.TestCase):
    def test_gps_outage_window_has_no_fixes(self):
        data = generate_dataset(seed=42)
        times = data["times"]
        outage_indices = [
            i for i, t in enumerate(times) if 72.0 <= t < 87.0 and i % 20 == 0
        ]
        self.assertTrue(outage_indices)
        for i in outage_indices:
            self.assertNotIn(i, data["gps"])
        # 121 one-second slots in 0..120 s, minus the 15 outage seconds.
        self.assertEqual(len(data["gps"]), 106)


class TestFusion(unittest.TestCase):
    def test_noiseless_sensors_track_truth(self):
        data = generate_dataset(
            seed=7,
            gps_sigma_m=0.0,
            accel_bias_mps2=0.0,
            gyro_bias_radps=0.0,
            accel_noise_sigma_mps2=0.0,
            gyro_noise_sigma_radps=0.0,
        )
        ekf = ExtendedKalmanFilter(
            initial_state=np.array(
                [data["gps"][0][0], data["gps"][0][1], 0.0, 0.0, 0.0]
            ),
            initial_covariance=np.diag([0.01, 0.01, 0.01, 0.01, 0.001]),
            process_noise=PROCESS_NOISE * 0.01,
            gps_variance=0.0001,
        )
        errors = []
        for i in range(len(data["times"]) - 1):
            accel, yaw = data["imu_controls"][i]
            ekf.predict(float(accel), float(yaw), float(data["dt"]))
            if i + 1 in data["gps"]:
                ekf.update_gps(data["gps"][i + 1])
            errors.append(
                np.linalg.norm(ekf.state[:2] - data["states"][i + 1, :2])
            )
        self.assertLess(float(np.sqrt(np.mean(np.array(errors) ** 2))), 0.05)

    def test_seeded_run_beats_both_baselines(self):
        metrics = run_simulation(seed=42)["metrics"]
        self.assertLess(
            metrics["ekf_position_rmse_m"], metrics["gps_raw_rmse_at_fixes_m"]
        )
        self.assertLess(metrics["ekf_position_rmse_m"], 3.0)
        self.assertLess(
            metrics["ekf_position_rmse_m"],
            0.05 * metrics["dead_reckoning_position_rmse_m"],
        )
        self.assertLess(metrics["ekf_outage_max_error_m"], 5.0)
        self.assertLess(metrics["ekf_final_position_error_m"], 2.0)


if __name__ == "__main__":
    unittest.main()
