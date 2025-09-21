
"""
movie_library.py: Utility functions for 3D plotting and animation of multi-robot systems.
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from itertools import product
from scipy.spatial.transform import Rotation as R
from typing import Any, List, Tuple, Optional

# ---- Constants ----
_DEFAULT_LOAD_COLOR = "g"
_DEFAULT_LOAD_LINEWIDTH = 2
_DEFAULT_LOAD_ALPHA = 0.8
_DEFAULT_LOAD_LINESTYLE = "solid"
_DEFAULT_CABLE_COLOR = "y"
_DEFAULT_CABLE_LINEWIDTH = 2
_DEFAULT_CAGE_COLOR = "b"
_DEFAULT_CAGE_LINEWIDTH = 2
_DEFAULT_ARM_COLOR = "k"
_DEFAULT_ARM_LINEWIDTH = 1.5
_DEFAULT_FORCE_COLOR = "red"
_DEFAULT_FORCE_LINEWIDTH = 2
_DEFAULT_FORCE_LENGTH = 0.05


def init_load(
    vertices: np.ndarray,
    ax: Any,
    color: str = _DEFAULT_LOAD_COLOR,
    linewidth: float = _DEFAULT_LOAD_LINEWIDTH,
    alpha: float = _DEFAULT_LOAD_ALPHA,
    linestyle: str = _DEFAULT_LOAD_LINESTYLE,
) -> List[Any]:
    """
    Plots a 2D or 3D load (polygon) by connecting its vertices.
    Args:
        vertices: Array of shape (N, 3) or (3, N). Each row or column is a vertex.
        ax: The matplotlib 3D axes object to plot on.
        color, linewidth, alpha, linestyle: Plotting options.
    Returns:
        List of line objects for the load sides.
    """
    vertices = np.asarray(vertices)
    if vertices.shape[0] == 3 and vertices.shape[1] != 3:
        vertices = vertices.T
    if vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    num_vertices = vertices.shape[0]
    sides = [[vertices[i], vertices[(i + 1) % num_vertices]] for i in range(num_vertices)]
    load_lines = []
    for side in sides:
        x, y, z = zip(*side)
        load_lines.append(ax.plot(x, y, z, color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle))
    return load_lines


# Update the load plot
def update_load(vertices: np.ndarray, load_lines: List[Any]) -> None:
    """
    Update the plotted load sides to new vertex positions.
    Args:
        vertices: Array of shape (N, 3) or (3, N).
        load_lines: List of line objects as returned by init_load.
    """
    vertices = np.asarray(vertices)
    if vertices.shape[0] == 3 and vertices.shape[1] != 3:
        vertices = vertices.T
    num_vertices = vertices.shape[0]
    if len(load_lines) != num_vertices:
        raise ValueError("Number of load_lines does not match number of vertices.")
    for i in range(num_vertices):
        side = [vertices[i], vertices[(i + 1) % num_vertices]]
        x, y, z = zip(*side)
        load_lines[i][0].set_data(x, y)
        load_lines[i][0].set_3d_properties(z)


def init_cable(
    rho: np.ndarray,
    pi: np.ndarray,
    ax: Any,
    linestyle: str = _DEFAULT_LOAD_LINESTYLE,
    color: str = _DEFAULT_CABLE_COLOR,
    linewidth: float = _DEFAULT_CABLE_LINEWIDTH,
) -> Any:
    """
    Plots a cable as a line connecting two points.
    Args:
        rho: Array of shape (3,) for one end of the cable.
        pi: Array of shape (3,) for the other end.
        ax: The matplotlib 3D axes object to plot on.
        linestyle, color, linewidth: Plotting options.
    Returns:
        The line object for the cable.
    """
    rho = np.asarray(rho)
    pi = np.asarray(pi)
    (cable,) = ax.plot([rho[0], pi[0]], [rho[1], pi[1]], [rho[2], pi[2]], color=color, linestyle=linestyle, linewidth=linewidth)
    return cable


def update_cable(rho: np.ndarray, pi: np.ndarray, cable: Any) -> None:
    """
    Update the plotted cable to new endpoints.
    Args:
        rho: Array of shape (3,)
        pi: Array of shape (3,)
        cable: The line object as returned by init_cable.
    """
    rho = np.asarray(rho)
    pi = np.asarray(pi)
    cable.set_data([rho[0], pi[0]], [rho[1], pi[1]])
    cable.set_3d_properties([rho[2], pi[2]])


def plot_cage(
    x0: float,
    y0: float,
    lx: float,
    ly: float,
    lz: float,
    ax: Any,
    color: str = _DEFAULT_CAGE_COLOR,
    linewidth: float = _DEFAULT_CAGE_LINEWIDTH,
) -> List[Any]:

    x = [
        x0 - lx / 2,
        x0 + lx / 2,
    ]
    y = [
        y0 - ly / 2,
        y0 + ly / 2,
    ]
    z = [0, lz]

    lines = []
    # Get all the vertices of the cube from the cartesian product
    vertices = np.array(list(product(x, y, z))).transpose()
    sides = get_cube_sides(vertices)
    # Plot the sides of the cube
    for side in sides:
        lines.append(ax.plot(*zip(*side), color=color, linewidth=linewidth))

    return lines


def get_cube_sides(vertices: np.ndarray) -> List[List[np.ndarray]]:
    # Define the 12 sides of the cube (pairs of vertex indices)
    sides = [
        [vertices[:, 0], vertices[:, 1]],
        [vertices[:, 1], vertices[:, 3]],
        [vertices[:, 2], vertices[:, 3]],
        [vertices[:, 3], vertices[:, 7]],
        [vertices[:, 0], vertices[:, 2]],
        [vertices[:, 1], vertices[:, 5]],
        [vertices[:, 2], vertices[:, 6]],
        [vertices[:, 0], vertices[:, 4]],
        [vertices[:, 4], vertices[:, 5]],
        [vertices[:, 5], vertices[:, 7]],
        [vertices[:, 6], vertices[:, 7]],
        [vertices[:, 4], vertices[:, 6]],
    ]
    return sides


def init_rob_frame(p: np.ndarray, Ri: np.ndarray, ax: Any) -> Any:
    """
    Plot a robot frame as 3D quivers (axes) at position p with orientation Ri.
    Args:
        p: Array of shape (3,)
        Ri: Rotation matrix, shape (3, 3)
        ax: The matplotlib 3D axes object
    Returns:
        The quiver object
    """
    p = np.asarray(p)
    Ri = np.asarray(Ri)
    rob_frame = ax.quiver(
        p[0] * np.ones(3),
        p[1] * np.ones(3),
        p[2] * np.ones(3),
        Ri[0, :],
        Ri[1, :],
        Ri[2, :],
        length=0.5,
        normalize=True,
        color=["r", "b", "b"],
    )
    return rob_frame


def init_drone_plot(drone_id: int, p: np.ndarray, ax: Any) -> Any:
    """
    Plot a drone as a point in 3D.
    Args:
        drone_id: Index of the drone (0 for red, others blue)
        p: Array of shape (3,)
        ax: The matplotlib 3D axes object
    Returns:
        The point object
    """
    color = "red" if drone_id == 0 else "blue"
    p = np.asarray(p)
    (drone_point,) = ax.plot(p[0], p[1], p[2], "o", color=color)
    return drone_point


def init_quad_arms(
    pi: np.ndarray,
    Ri: np.ndarray,
    arm_length: float,
    ax: Any,
    color: str = _DEFAULT_ARM_COLOR,
    linewidth: float = _DEFAULT_ARM_LINEWIDTH,
) -> List[Any]:
    """
    Plot quadrotor arms as 3 lines in 3D.
    Args:
        pi: Center position, shape (3,)
        Ri: Rotation matrix, shape (3, 3)
        arm_length: Length of each arm
        ax: The matplotlib 3D axes object
        color, linewidth: Plotting options
    Returns:
        List of line objects for the arms
    """
    pi = np.asarray(pi)
    Ri = np.asarray(Ri)
    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    xp = pi + arm_length * Ri @ e1
    (rob_arms_xp,) = ax.plot([pi[0], xp[0]], [pi[1], xp[1]], [pi[2], xp[2]], "ro-", markersize=3, linewidth=linewidth)
    xm = pi - arm_length * Ri @ e1
    (rob_arms_xm,) = ax.plot([pi[0], xm[0]], [pi[1], xm[1]], [pi[2], xm[2]], "o-", linewidth=linewidth, markersize=3, color=color)
    ym = pi - arm_length * Ri @ e2
    yp = pi + arm_length * Ri @ e2
    (rob_arms_y,) = ax.plot([ym[0], yp[0]], [ym[1], yp[1]], [ym[2], yp[2]], "o-", linewidth=linewidth, markersize=3, color=color)
    return [rob_arms_xm, rob_arms_xp, rob_arms_y]


def update_quad_arms(
    pi: np.ndarray,
    Ri: np.ndarray,
    arm_length: float,
    quad_arms: List[Any],
) -> None:
    """
    Update the plotted quadrotor arms to new position/orientation.
    Args:
        pi: Center position, shape (3,)
        Ri: Rotation matrix, shape (3, 3)
        arm_length: Length of each arm
        quad_arms: List of line objects as returned by init_quad_arms
    """
    pi = np.asarray(pi)
    Ri = np.asarray(Ri)
    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    xm = pi - arm_length * Ri @ e1
    xp = pi + arm_length * Ri @ e1
    ym = pi - arm_length * Ri @ e2
    yp = pi + arm_length * Ri @ e2
    quad_arms[0].set_data([pi[0], xm[0]], [pi[1], xm[1]])
    quad_arms[0].set_3d_properties([pi[2], xm[2]])
    quad_arms[1].set_data([pi[0], xp[0]], [pi[1], xp[1]])
    quad_arms[1].set_3d_properties([pi[2], xp[2]])
    quad_arms[2].set_data([ym[0], yp[0]], [ym[1], yp[1]])
    quad_arms[2].set_3d_properties([ym[2], yp[2]])


def init_commanded_force(
    p: np.ndarray,
    force: np.ndarray,
    ax: Any,
    linestyle: str = _DEFAULT_LOAD_LINESTYLE,
    linewidth: float = _DEFAULT_FORCE_LINEWIDTH,
    color: str = _DEFAULT_FORCE_COLOR,
) -> Any:
    """
    Plot a commanded force as a 3D arrow (quiver).
    Args:
        p: Origin, shape (3,)
        force: Force vector, shape (3,)
        ax: The matplotlib 3D axes object
        linestyle, linewidth, color: Plotting options
    Returns:
        The quiver object
    """
    p = np.asarray(p)
    force = np.asarray(force)
    force_arrow = ax.quiver(
        p[0], p[1], p[2], force[0], force[1], force[2],
        color=color, label="Commanded Force", linewidth=linewidth, length=_DEFAULT_FORCE_LENGTH, linestyle=linestyle
    )
    return force_arrow


def update_commanded_force(
    p: np.ndarray,
    force: np.ndarray,
    force_arrow: Any,
    ax: Any,
    linestyle: str = _DEFAULT_LOAD_LINESTYLE,
    color: str = _DEFAULT_FORCE_COLOR,
) -> Any:
    """
    Update the plotted commanded force arrow.
    Args:
        p: Origin, shape (3,)
        force: Force vector, shape (3,)
        force_arrow: The old quiver object
        ax: The matplotlib 3D axes object
        linestyle, color: Plotting options
    Returns:
        The new quiver object
    """
    # Remove the old arrow if it is present in ax.collections
    try:
        if force_arrow in ax.collections:
            force_arrow.remove()
    except Exception:
        pass  # If already removed or not present, ignore
    p = np.asarray(p)
    force = np.asarray(force)
    force_arrow = ax.quiver(
        p[0], p[1], p[2], force[0], force[1], force[2],
        color=color, label="Commanded Force", linewidth=_DEFAULT_FORCE_LINEWIDTH, length=_DEFAULT_FORCE_LENGTH, linestyle=linestyle
    )
    return force_arrow
    
