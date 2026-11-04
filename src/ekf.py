"""Extended Kalman filter for 2D IMU + GPS sensor fusion.

State vector (the Project 2 starter-sheet contract):

    x = [x, y, vx, vy, heading]

Positions and velocities are in metres and metres/second in a local
east/north plane; heading is radians counter-clockwise from east.

The IMU supplies the control inputs for each prediction step:

* longitudinal acceleration (m/s^2), and
* yaw rate (rad/s).

The motion model is a constant-turn-rate style model: heading advances by
the measured yaw rate, speed changes by the measured longitudinal
acceleration, and the velocity vector is aligned with the new heading.
That model is nonlinear in the velocity/heading states, so prediction
uses the analytic Jacobian implemented below. GPS supplies a direct
position measurement update.
"""

from __future__ import annotations

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def motion_step(
    state: np.ndarray,
    acceleration: float,
    yaw_rate: float,
    dt: float,
) -> np.ndarray:
    """Advance the nonlinear motion model by one time step.

    The position update uses the average of the starting and ending
    velocity vectors, which behaves better than a first-order Euler step
    on the curved segments of the test trajectory.
    """
    x, y, vx, vy, heading = np.asarray(state, dtype=float)
    speed = float(np.hypot(vx, vy))
    new_heading = wrap_angle(heading + yaw_rate * dt)
    new_speed = max(0.0, speed + acceleration * dt)

    new_vx = new_speed * np.cos(new_heading)
    new_vy = new_speed * np.sin(new_heading)

    return np.array(
        [
            x + 0.5 * (vx + new_vx) * dt,
            y + 0.5 * (vy + new_vy) * dt,
            new_vx,
            new_vy,
            new_heading,
        ],
        dtype=float,
    )


def motion_jacobian(
    state: np.ndarray,
    acceleration: float,
    yaw_rate: float,
    dt: float,
) -> np.ndarray:
    """Analytic Jacobian of ``motion_step`` with respect to the state.

    Near zero speed the velocity direction is undefined; in that case the
    current heading supplies the direction used for the derivative. This
    is the standard practical treatment for a vehicle starting from rest.
    """
    _x, _y, vx, vy, heading = np.asarray(state, dtype=float)
    speed = float(np.hypot(vx, vy))
    if speed > 1e-9:
        ux, uy = vx / speed, vy / speed
    else:
        ux, uy = np.cos(heading), np.sin(heading)

    phi = heading + yaw_rate * dt
    new_speed = max(0.0, speed + acceleration * dt)
    c, s = np.cos(phi), np.sin(phi)

    # Derivatives of the new velocity components.
    dvx_dvx = c * ux
    dvx_dvy = c * uy
    dvx_dheading = -new_speed * s

    dvy_dvx = s * ux
    dvy_dvy = s * uy
    dvy_dheading = new_speed * c

    jacobian = np.eye(5)
    # x row: x + 0.5 * (vx + vx_new) * dt
    jacobian[0, 2] = 0.5 * dt * (1.0 + dvx_dvx)
    jacobian[0, 3] = 0.5 * dt * dvx_dvy
    jacobian[0, 4] = 0.5 * dt * dvx_dheading
    # y row: y + 0.5 * (vy + vy_new) * dt
    jacobian[1, 2] = 0.5 * dt * dvy_dvx
    jacobian[1, 3] = 0.5 * dt * (1.0 + dvy_dvy)
    jacobian[1, 4] = 0.5 * dt * dvy_dheading
    # Velocity rows.
    jacobian[2, 2] = dvx_dvx
    jacobian[2, 3] = dvx_dvy
    jacobian[2, 4] = dvx_dheading
    jacobian[3, 2] = dvy_dvx
    jacobian[3, 3] = dvy_dvy
    jacobian[3, 4] = dvy_dheading
    # Heading row remains [0, 0, 0, 0, 1].
    return jacobian


class ExtendedKalmanFilter:
    """EKF with IMU-driven prediction and GPS position updates."""

    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        process_noise: np.ndarray,
        gps_variance: float,
    ) -> None:
        self.state = np.asarray(initial_state, dtype=float).copy()
        self.state[4] = wrap_angle(self.state[4])
        self.covariance = np.asarray(initial_covariance, dtype=float).copy()
        self.process_noise = np.asarray(process_noise, dtype=float).copy()
        self.gps_variance = float(gps_variance)

        self._measurement_matrix = np.zeros((2, 5))
        self._measurement_matrix[0, 0] = 1.0
        self._measurement_matrix[1, 1] = 1.0

    def predict(self, acceleration: float, yaw_rate: float, dt: float) -> None:
        jacobian = motion_jacobian(self.state, acceleration, yaw_rate, dt)
        self.state = motion_step(self.state, acceleration, yaw_rate, dt)
        self.covariance = (
            jacobian @ self.covariance @ jacobian.T + self.process_noise
        )
        # Remove tiny floating-point asymmetry before the next update.
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def update_gps(self, measurement_xy: np.ndarray) -> None:
        """Fuse one GPS position fix with the Joseph covariance update."""
        z = np.asarray(measurement_xy, dtype=float)
        h = self._measurement_matrix
        r = np.eye(2) * self.gps_variance

        innovation = z - h @ self.state
        innovation_covariance = h @ self.covariance @ h.T + r
        kalman_gain = (
            self.covariance @ h.T @ np.linalg.inv(innovation_covariance)
        )

        self.state = self.state + kalman_gain @ innovation
        self.state[4] = wrap_angle(self.state[4])

        identity = np.eye(5)
        joseph = identity - kalman_gain @ h
        self.covariance = (
            joseph @ self.covariance @ joseph.T
            + kalman_gain @ r @ kalman_gain.T
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    @property
    def position_sigma(self) -> float:
        """Combined 1-sigma horizontal position uncertainty, in metres."""
        return float(np.sqrt(max(0.0, self.covariance[0, 0] + self.covariance[1, 1])))
