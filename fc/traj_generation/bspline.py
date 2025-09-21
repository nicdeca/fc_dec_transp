
"""
BSpline: B-spline curve generation and evaluation utilities.

Provides basis function evaluation, curve sampling, and static construction from waypoints.
"""


import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from copy import deepcopy
from numpy.linalg import solve
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3D plotting

# ---- Constants ----
_ENDPOINT_TOL = 1e-6  # Tolerance for endpoint comparison
_ENDPOINT_SHIFT = 1e-4  # Shift to avoid endpoint ambiguity


class BSpline:
    """
    B-spline curve class for evaluation, sampling, and construction from waypoints.

    Attributes:
        P (np.ndarray): Control points, shape (n_ctrl, d)
        U (np.ndarray): Knot vector
        p (int): Degree of the spline
        d (int): Dimension of the curve
        n_knot (int): Number of knots minus one
    """
    def __init__(self, control_points: np.ndarray, knot_vector: np.ndarray, degree: int) -> None:
        """
        Initialize a B-spline curve.
        Args:
            control_points: Array of control points, shape (n_ctrl, d)
            knot_vector: Knot vector, shape (n_knot,)
            degree: Degree of the spline
        Raises:
            ValueError: If input shapes are inconsistent
        """
        P = np.atleast_2d(control_points)
        if P.shape[0] < degree + 1:
            raise ValueError("Number of control points must be at least degree + 1.")
        if len(knot_vector) < 2 * degree + 2:
            raise ValueError("Knot vector too short for given degree.")
        self.P = P
        self.U = np.asarray(knot_vector)
        self.p = int(degree)
        self.d = P.shape[1]
        self.n_knot = len(self.U) - 1


    def _vectorized_point_batch(self, u_values: np.ndarray) -> np.ndarray:
        """
        Efficiently evaluate the B-spline curve at multiple parameter values (no derivatives).
        Args:
            u_values: Array of parameter values (N,)
        Returns:
            Array of curve points (N, d)
        """
        N = u_values.shape[0]
        result = np.empty((N, self.d), dtype=np.float64)
        for idx, u in enumerate(u_values):
            i = BSpline.which_span(u, self.U, self.n_knot, self.p)
            B = BSpline.basis_funs(i, u, self.p, self.U)
            result[idx, :] = np.sum([B[j] * self.P[i - self.p + j, :] for j in range(self.p + 1)], axis=0)
        return result

    # -----------------------------
    # Basis functions
    # -----------------------------
    @staticmethod
    def basis_funs(i: int, u: float, p: int, U: np.ndarray) -> np.ndarray:
        """
        Evaluate B-spline basis functions at parameter u.
        Args:
            i: Knot span index
            u: Parameter value
            p: Degree
            U: Knot vector
        Returns:
            Basis function values (np.ndarray, shape (p+1,))
        """
        if np.isclose(u, U[-1], atol=_ENDPOINT_TOL):
            u -= _ENDPOINT_SHIFT

        B = np.zeros(p + 1)
        B[0] = 1.0
        DL = np.zeros(p + 1)
        DR = np.zeros(p + 1)

        for j in range(1, p + 1):
            DL[j] = u - U[i + 1 - j]
            DR[j] = U[i + j] - u
            acc = 0.0
            for r in range(j):
                denom = DR[r + 1] + DL[j - r]
                if np.abs(denom) < 1e-12:
                    temp = 0.0
                else:
                    temp = B[r] / denom
                B[r] = acc + DR[r + 1] * temp
                acc = DL[j - r] * temp
            B[j] = acc
        return B

    @staticmethod
    def which_span(u: float, U: np.ndarray, n_knot: int, p: int) -> int:
        """
        Find the knot span index for parameter u.
        Args:
            u: Parameter value
            U: Knot vector
            n_knot: Number of knots minus one
            p: Degree
        Returns:
            Knot span index (int)
        """
        if np.isclose(u, U[-1], atol=_ENDPOINT_TOL):
            u -= _ENDPOINT_SHIFT

        high = n_knot - p
        low = p
        if u == U[high]:
            return high

        mid = (high + low) // 2
        while u < U[mid] or u >= U[mid + 1]:
            if u == U[mid + 1]:
                mid += 1
            elif u > U[mid]:
                low = mid
            else:
                high = mid
            mid = (high + low) // 2
        return mid

    @staticmethod
    def ders_basis_funs(u: float, i: int, p: int, n: int, U: np.ndarray) -> np.ndarray:
        """
        Evaluate derivatives of B-spline basis functions up to order n at u.
        Args:
            u: Parameter value
            i: Knot span index
            p: Degree
            n: Derivative order
            U: Knot vector
        Returns:
            Array of derivatives (shape (n+1, p+1))
        """
        Du = np.zeros((p + 1, p + 1))
        a = np.zeros((2, p + 1))
        Ders = np.zeros((n + 1, p + 1))
        DL = np.zeros(p + 1)
        DR = np.zeros(p + 1)

        Du[0, 0] = 1.0
        for k in range(1, p + 1):
            DL[k] = u - U[i + 1 - k]
            DR[k] = U[i + k] - u
            acc = 0.0
            for r in range(k):
                Du[k, r] = DR[r + 1] + DL[k - r]
                temp = Du[r, k - 1] / Du[k, r]
                Du[r, k] = acc + DR[r + 1] * temp
                acc = DL[k - r] * temp
            Du[k, k] = acc

        for k in range(p + 1):
            Ders[0, k] = Du[k, p]

        for r in range(p + 1):
            s1, s2 = 0, 1
            a[0, 0] = 1.0
            for k in range(1, n + 1):
                d = 0.0
                rk, pk = r - k, p - k
                if r >= k:
                    a[s2, 0] = a[s1, 0] / Du[pk + 1, rk]
                    d = a[s2, 0] * Du[rk, pk]
                j1 = 1 if rk >= -1 else -rk
                j2 = k - 1 if r - 1 <= pk else p - r
                for ii in range(j1, j2 + 1):
                    a[s2, ii] = (a[s1, ii] - a[s1, ii - 1]) / Du[pk + 1, rk + ii]
                    d += a[s2, ii] * Du[rk + ii, pk]
                if r <= pk:
                    a[s2, k] = -a[s1, k - 1] / Du[pk + 1, r]
                    d += a[s2, k] * Du[r, pk]
                Ders[k, r] = d
                s1, s2 = s2, s1

        r = p
        for k in range(1, n + 1):
            for j in range(p + 1):
                Ders[k, j] *= r
            r *= (p - k)
        return Ders

    # -----------------------------
    # Evaluation
    # -----------------------------
    def point(self, u: float) -> np.ndarray:
        """
        Evaluate the B-spline curve at parameter u.
        Args:
            u: Parameter value
        Returns:
            Point on the curve (np.ndarray, shape (d,))
        """
        i = BSpline.which_span(u, self.U, self.n_knot, self.p)
        B = BSpline.basis_funs(i, u, self.p, self.U)
        # Use np.sum for correct ndarray return type
        return np.sum([B[j] * self.P[i - self.p + j, :] for j in range(self.p + 1)], axis=0)

    def derivative(self, u: float, k: int = 1) -> np.ndarray:
        """
        Evaluate derivatives of the B-spline curve at parameter u.
        Args:
            u: Parameter value
            k: Derivative order
        Returns:
            Array of derivatives (shape (k+1, d))
        """
        i = BSpline.which_span(u, self.U, self.n_knot, self.p)
        B = BSpline.ders_basis_funs(u, i, self.p, k, self.U)
        s = np.zeros((k + 1, self.d))
        for der in range(k + 1):
            for j in range(self.p + 1):
                s[der, :] += self.P[i - self.p + j, :] * B[der, j]
        return s

    def sample_curve(self, num_points: int = 100, der: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample the B-spline curve or its derivatives at evenly spaced parameters.
        Args:
            num_points: Number of sample points
            der: Derivative order (0 for curve itself)
        Returns:
            Tuple (u_values, curve/derivative values)
        """
        u_min, u_max = self.U[self.p], self.U[self.n_knot - self.p]
        u_values = np.linspace(u_min, u_max, num_points)
        if der == 0:
            curve = self._vectorized_point_batch(u_values)
        else:
            # Derivatives are still evaluated one by one (could be vectorized further)
            curve = np.array([self.derivative(u, der)[der, :] for u in u_values])
        return u_values, curve

    # -----------------------------
    # Plotting
    # -----------------------------
    def plot2d(self, num_points: int = 100) -> None:
        """
        Plot the B-spline curve and control points in 2D.
        Args:
            num_points: Number of sample points
        Raises:
            ValueError: If curve is not 2D
        """
        if self.d != 2:
            raise ValueError("plot2d is only valid for 2D splines")
        _, curve = self.sample_curve(num_points)
        plt.plot(curve[:, 0], curve[:, 1], label="B-spline")
        plt.plot(self.P[:, 0], self.P[:, 1], "o--", label="Control points")
        plt.legend()
        plt.grid()

    def plot3d(self, waypoints: Optional[np.ndarray] = None, num_points: int = 100) -> None:
        """
        Plot the B-spline curve and control points in 3D.
        Args:
            waypoints: Optional waypoints to plot
            num_points: Number of sample points
        Raises:
            ValueError: If curve is not 3D
        """
        if self.d != 3:
            raise ValueError("plot3d is only valid for 3D splines")
        _, curve = self.sample_curve(num_points)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], label="B-spline")
        # matplotlib's scatter for 3D: x, y, z order
        # mypy and some linters may not recognize 3D scatter signature; type: ignore is safe here
        ax.scatter(self.P[:, 0], self.P[:, 1], self.P[:, 2], c="red", label="Control points")  # type: ignore
        if waypoints is not None:
            ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c="green", label="Waypoints")  # type: ignore
        ax.legend()
        plt.show()

    # -----------------------------
    # Static constructor
    # -----------------------------
    @staticmethod
    def from_waypoints(
        waypoints: np.ndarray,
        times: np.ndarray,
        n_der_conds: int,
        p: int,
        der_conds: Optional[np.ndarray] = None,
    ) -> "BSpline":
        """
        Construct a B-spline that interpolates given waypoints and (optionally) endpoint derivatives.
        Args:
            waypoints: Array of waypoints, shape (N, d)
            times: Array of parameter values for waypoints, shape (N,)
            n_der_conds: Number of endpoint derivative constraints
            p: Degree of the spline
            der_conds: Optional endpoint derivative conditions, shape (2*n_der_conds, d)
        Returns:
            BSpline instance
        Raises:
            ValueError: If input shapes or parameters are inconsistent
        """
        wp = deepcopy(waypoints)
        if wp.ndim == 1:
            wp = np.expand_dims(wp, axis=1)
            if der_conds is not None:
                der_conds = np.expand_dims(der_conds, axis=1)
        if der_conds is not None and der_conds.shape[0] != 2 * n_der_conds:
            raise ValueError("der_conds must have shape (2*n_der_conds, d)")

        num_waypoints, d = wp.shape
        if p not in (2 * n_der_conds + 1, 2 * n_der_conds):
            raise ValueError("Inconsistent degree and derivative conditions")

        if p == 2 * n_der_conds + 1:
            n_knot = num_waypoints + 2 * p
            num_control_points = num_waypoints - 1 + p
            U = np.zeros(n_knot)
            U[0:p] = times[0]
            U[p:-p] = times
            U[-p:] = times[-1]
        else:
            n_knot = num_waypoints + 2 * p + 1
            num_control_points = num_waypoints + p
            U = np.zeros(n_knot)
            U[0 : p + 1] = times[0]
            U[p + 1 : -(p + 1)] = (times[:-1] + times[1:]) / 2
            U[-(p + 1) :] = times[-1]

        interp_mat = np.zeros((num_control_points, num_control_points))
        b = np.zeros((num_control_points, d))
        b[:num_waypoints, :] = wp

        for kk, u in enumerate(times):
            i = BSpline.which_span(u, U, n_knot, p)
            if kk == 0 or kk == (times.size - 1):
                if der_conds is None:
                    raise ValueError("Endpoint derivative conditions required for endpoint interpolation.")
                ders = BSpline.ders_basis_funs(u, i, p, n_der_conds, U)
                B = ders[0, :]
                pp = 0 if kk == 0 else 1
                interp_mat[num_waypoints + pp * n_der_conds : num_waypoints + (pp + 1) * n_der_conds, i - p : i + 1] = ders[1:, :]
                b[num_waypoints + pp * n_der_conds : num_waypoints + (pp + 1) * n_der_conds, :] = der_conds[pp * n_der_conds : pp * n_der_conds + n_der_conds, :]
            else:
                B = BSpline.basis_funs(i, u, p, U)
            interp_mat[kk, i - p : i + 1] = B

        control_points = solve(interp_mat, b)
        return BSpline(control_points, U, p)
