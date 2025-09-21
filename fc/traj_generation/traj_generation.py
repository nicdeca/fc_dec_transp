
import numpy as np
from typing import Tuple, Optional

from fc.traj_generation.bspline import BSpline


class TrajectoryGenerator:

    @classmethod
    def from_waypoints(
        cls,
        waypoints: np.ndarray,
        times: np.ndarray,
        n_der_conds: int,
        p: int,
        der_conds: Optional[np.ndarray] = None,
        t0: Optional[float] = None,
        tf: Optional[float] = None,
        **bspline_kwargs
    ) -> "TrajectoryGenerator":
        """
        Construct a TrajectoryGenerator directly from waypoints and B-spline parameters.
        Args:
            waypoints: Array of waypoints, shape (N, d)
            times: Array of parameter values for waypoints, shape (N,)
            n_der_conds: Number of endpoint derivative constraints
            p: Degree of the spline
            der_conds: Optional endpoint derivative conditions, shape (2*n_der_conds, d)
            t0: Start time (defaults to times[0])
            tf: End time (defaults to times[-1])
            bspline_kwargs: Additional keyword arguments for BSpline.from_waypoints
        Returns:
            TrajectoryGenerator instance
        """
        bspline = BSpline.from_waypoints(
            waypoints=waypoints,
            times=times,
            n_der_conds=n_der_conds,
            p=p,
            der_conds=der_conds,
            **bspline_kwargs
        )
        t0_val = float(times[0]) if t0 is None else float(t0)
        tf_val = float(times[-1]) if tf is None else float(tf)
        return cls(bspline, t0=t0_val, tf=tf_val)
    """
    Generates a time-scaled trajectory from a B-spline curve.
    Attributes:
        bspline (BSpline): The underlying B-spline curve.
        t0 (float): Start time.
        tf (float): End time.
        u_min (float): Minimum B-spline parameter.
        u_max (float): Maximum B-spline parameter.
    """
    ORDER = 3  # Derivative order for evaluate (pos, vel, acc, jerk)

    def __init__(self, bspline: "BSpline", t0: float, tf: float) -> None:
        """
        Wraps a BSpline trajectory with time-scaling.

        Args:
            bspline: The BSpline object defining the trajectory.
            t0: Start time of the trajectory.
            tf: End time of the trajectory.
        Raises:
            ValueError: If tf <= t0 or B-spline domain does not match time interval.
        """
        if tf <= t0:
            raise ValueError("End time tf must be greater than start time t0.")
        self.bspline = bspline
        self.t0 = float(t0)
        self.tf = float(tf)
        self.u_min = bspline.U[bspline.p]
        self.u_max = bspline.U[bspline.n_knot - bspline.p]
        # Optionally check that u_min/u_max match t0/tf domain (warn if not)
        if not (np.isclose(self.u_min, t0) and np.isclose(self.u_max, tf)):
            import warnings
            warnings.warn("B-spline parameter domain does not match time interval.")

    def _map_time_to_u(self, t: float) -> Tuple[float, float]:
        """
        Map real time t to the B-spline parameter u and scaling du/dt.
        Args:
            t: Time value
        Returns:
            Tuple (u, du_dt)
        """
        if t < self.t0:
            import warnings
            warnings.warn(f"t={t} is before t0={self.t0}; clamping to t0.")
            t = self.t0
        elif t > self.tf:
            import warnings
            warnings.warn(f"t={t} is after tf={self.tf}; clamping to tf.")
            t = self.tf
        tau = (t - self.t0) / (self.tf - self.t0)  # normalized time ∈ [0, 1]
        u = self.u_min + tau * (self.u_max - self.u_min)
        du_dt = (self.u_max - self.u_min) / (self.tf - self.t0)
        return u, du_dt

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate the trajectory at time t.

        Args:
            t: Time value
        Returns:
            pos: Position vector (d,)
            vel: Velocity vector (d,)
            acc: Acceleration vector (d,)
            jerk: Jerk vector (d,)
        """
        u, du_dt = self._map_time_to_u(t)
        ders = self.bspline.derivative(u, k=self.ORDER)  # up to 3rd derivative wrt u

        pos_u = ders[0, :]
        vel_u = ders[1, :]
        acc_u = ders[2, :]
        jerk_u = ders[3, :]

        # Apply chain rule for time-scaling
        vel = vel_u * du_dt
        acc = acc_u * du_dt**2
        jerk = jerk_u * du_dt**3

        return pos_u, vel, acc, jerk

    def sample(self, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample the trajectory uniformly in time (vectorized for performance).

        Args:
            num_points: Number of sample points
        Returns:
            times: Sample times
            pos, vel, acc, jerk: Arrays of shape (num_points, d)
        """
        times = np.linspace(self.t0, self.tf, num_points)
        # Vectorized mapping from t to u and du/dt
        tau = (times - self.t0) / (self.tf - self.t0)
        u_values = self.u_min + tau * (self.u_max - self.u_min)
        du_dt = (self.u_max - self.u_min) / (self.tf - self.t0)
        # Vectorized position evaluation
        pos = self.bspline._vectorized_point_batch(u_values)
        # For derivatives, still use loop (could be vectorized further)
        vel = np.empty_like(pos)
        acc = np.empty_like(pos)
        jerk = np.empty_like(pos)
        for i, u in enumerate(u_values):
            ders = self.bspline.derivative(u, k=self.ORDER)
            vel[i, :] = ders[1, :] * du_dt
            acc[i, :] = ders[2, :] * du_dt**2
            jerk[i, :] = ders[3, :] * du_dt**3
        return times, pos, vel, acc, jerk
    


from scipy.spatial.transform import Rotation as R


class OrientationTrajectoryGenerator:
    """
    Generates a smooth orientation trajectory using SLERP between quaternion waypoints.
    Attributes:
        quats (np.ndarray): Array of quaternions (N, 4) in (x, y, z, w) format.
        times (np.ndarray): Array of times for each waypoint (N,).
        t0 (float): Start time.
        tf (float): End time.
    """
    def __init__(self, quats: np.ndarray, times: np.ndarray) -> None:
        """
        Args:
            quats: Array of quaternions (N, 4) in (x, y, z, w) format.
            times: Array of times for each waypoint (N,).
        Raises:
            ValueError: If input shapes are inconsistent or times not increasing.
        """
        if quats.shape[0] != times.shape[0]:
            raise ValueError("Number of quaternions and times must match.")
        if quats.shape[1] != 4:
            raise ValueError("Quaternions must have shape (N, 4).")
        if not np.all(np.diff(times) > 0):
            raise ValueError("Times must be strictly increasing.")
        self.quats = quats.copy()
        self.times = times.copy()
        self.t0 = float(times[0])
        self.tf = float(times[-1])

    def evaluate(self, t: float, dt: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate the orientation quaternion, angular velocity, and angular acceleration at time t using SLERP.
        Args:
            t: Time value
            dt: Small time step for finite difference (default: 1e-5)
        Returns:
            quat: Quaternion (x, y, z, w) at time t
            omega: Angular velocity (3,) in body frame at time t
            alpha: Angular acceleration (3,) (zero vector for SLERP)
        """
        from scipy.spatial.transform import Slerp
        # Clamp t to valid range
        if t <= self.t0:
            quat = self.quats[0]
            omega = np.zeros(3)
            alpha = np.zeros(3)
            return quat, omega, alpha
        if t >= self.tf:
            quat = self.quats[-1]
            omega = np.zeros(3)
            alpha = np.zeros(3)
            return quat, omega, alpha
        # Find segment
        idx = np.searchsorted(self.times, t) - 1
        idx = np.clip(idx, 0, len(self.times) - 2, dtype=int)
        seg_times = self.times[idx:idx+2]
        seg_rots = R.from_quat(self.quats[idx:idx+2])
        slerp = Slerp(seg_times, seg_rots)
        quat = slerp([t]).as_quat()[0]
        # Angular velocity: finite difference in body frame
        t_prev = max(self.t0, t - dt)
        t_next = min(self.tf, t + dt)
        q_prev = slerp([t_prev]).as_quat()[0]
        q_next = slerp([t_next]).as_quat()[0]
        r_prev = R.from_quat(q_prev)
        r_next = R.from_quat(q_next)
        r_curr = R.from_quat(quat)
        # Compute relative rotation from prev to next
        r_rel = r_next * r_prev.inv()
        rotvec = r_rel.as_rotvec() / (t_next - t_prev)
        # Express in current body frame
        omega = r_curr.inv().apply(rotvec)
        # For SLERP, angular acceleration is zero (piecewise constant velocity)
        alpha = np.zeros(3)
        return quat, omega, alpha

    def sample(self, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample the orientation trajectory uniformly in time.
        Args:
            num_points: Number of sample points
        Returns:
            times: Sample times (num_points,)
            quats: Quaternions (num_points, 4)
        """
        times = np.linspace(self.t0, self.tf, num_points)
        quats = np.empty((num_points, 4))
        for i, t in enumerate(times):
            quats[i] = self.evaluate(t)
        return times, quats

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # -----------------------
    # Example waypoints + times
    # -----------------------
    waypoints = np.array([
        [0.0, 0.0, 0.0],   # start
        [1.0, 2.0, 0.5],   # mid point
        [3.0, 0.0, 1.0],   # mid point
        [4.0, 2.0, 0.0]    # end
    ])

    # times associated with waypoints (monotonic increasing)
    times = np.array([0.0, 2.0, 5.0, 8.0])

    # degree = 5, derivative conditions = 2 (pos + vel)
    p = 5
    n_der_conds = 2
    der_conds = np.zeros((2 * n_der_conds, waypoints.shape[1]))

    # Use the new integrated constructor
    traj = TrajectoryGenerator.from_waypoints(
        waypoints=waypoints,
        times=times,
        n_der_conds=n_der_conds,
        p=p,
        der_conds=der_conds
    )

    # Evaluate and sample
    pos, vel, acc, jerk = traj.evaluate(3.0)  # query at t=3s
    print("At t=3s:")
    print("pos =", pos)
    print("vel =", vel)
    print("acc =", acc)
    print("jerk =", jerk)

    # Sample entire trajectory
    t, pos, vel, acc, jerk = traj.sample(200)

    # Plot position, velocity, acceleration, jerk
    fig, axs = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
    labels = ["x", "y", "z"]
    for i in range(3):
        axs[0].plot(t, pos[:, i], label=labels[i])
        axs[1].plot(t, vel[:, i], label=labels[i])
        axs[2].plot(t, acc[:, i], label=labels[i])
        axs[3].plot(t, jerk[:, i], label=labels[i])
    axs[0].set_ylabel("Position")
    axs[1].set_ylabel("Velocity")
    axs[2].set_ylabel("Acceleration")
    axs[3].set_ylabel("Jerk")
    axs[3].set_xlabel("Time [s]")
    for ax in axs:
        ax.legend()
        ax.grid()
    plt.show()

    # -----------------------
    # Example: OrientationTrajectoryGenerator usage
    # -----------------------
    from scipy.spatial.transform import Rotation as R
    # Define quaternion waypoints (x, y, z, w)
    quat_waypoints = np.array([
        R.from_euler('z', 0, degrees=True).as_quat(),
        R.from_euler('z', 90, degrees=True).as_quat(),
        R.from_euler('z', 180, degrees=True).as_quat(),
        R.from_euler('z', 270, degrees=True).as_quat()
    ])
    quat_times = np.array([0.0, 2.0, 5.0, 8.0])
    orient_traj = OrientationTrajectoryGenerator(quat_waypoints, quat_times)

    # Evaluate at t=3s
    quat, omega, alpha = orient_traj.evaluate(3.0)
    print("\nOrientation at t=3s (quaternion):", quat)
    print("As Euler angles (deg):", R.from_quat(quat).as_euler('xyz', degrees=True))
    print("Angular velocity at t=3s:", omega)
    print("Angular acceleration at t=3s:", alpha)

    # Sample orientation trajectory (quaternion, omega, alpha)
    t_orient = np.linspace(quat_times[0], quat_times[-1], 200)
    quats = np.empty((200, 4))
    omegas = np.empty((200, 3))
    alphas = np.empty((200, 3))
    for i, t_val in enumerate(t_orient):
        quats[i], omegas[i], alphas[i] = orient_traj.evaluate(t_val)
    eulers = R.from_quat(quats).as_euler('xyz', degrees=True)

    # Plot Euler angles over time
    fig2, axs2 = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    euler_labels = ["Roll (deg)", "Pitch (deg)", "Yaw (deg)"]
    for i in range(3):
        axs2[i].plot(t_orient, eulers[:, i], label=euler_labels[i])
        axs2[i].set_ylabel(euler_labels[i])
        axs2[i].grid()
    axs2[2].set_xlabel("Time [s]")
    plt.suptitle("Orientation Trajectory (Euler angles)")
    plt.show()

    # Plot angular velocity and acceleration over time
    fig3, axs3 = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    omega_labels = ["wx", "wy", "wz"]
    for i in range(3):
        axs3[0].plot(t_orient, omegas[:, i], label=omega_labels[i])
        axs3[1].plot(t_orient, alphas[:, i], label=omega_labels[i])
    axs3[0].set_ylabel("Angular velocity [rad/s]")
    axs3[1].set_ylabel("Angular acceleration [rad/s²]")
    axs3[1].set_xlabel("Time [s]")
    axs3[0].legend()
    axs3[1].legend()
    axs3[0].grid()
    axs3[1].grid()
    plt.suptitle("Orientation Trajectory: Angular Velocity and Acceleration")
    plt.show()
