"""
animate_flycrane.py: Animation utility for FlyCrane simulation.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
from typing import Any, List, Dict, Tuple

from fc.plot_utils.movie_library import (
    plot_cage,
    init_load, update_load,
    init_quad_arms, update_quad_arms,
    init_cable, update_cable,
    init_commanded_force, update_commanded_force
)



# ---- Constants ----
_ANIMATION_INTERVAL_MS = 50
_ARROW_LENGTH = 0.5
_FIGSIZE = (20, 20)

def animate_flycrane(
    times: np.ndarray,
    pl: np.ndarray,
    quatl: np.ndarray,
    pld: np.ndarray,
    quatld: np.ndarray,
    alpha: List[np.ndarray],
    fdes: List[np.ndarray],
    fperp: List[np.ndarray],
    pD: List[np.ndarray],
    quatD: List[np.ndarray],
    params: Dict[str, Any],
    downsampling_factor: int = 10
) -> "FuncAnimation":
    """
    Make an animation of the Fly-Crane simulation.
    Args:
        times: (N,) time array
        pl: (N,3) payload positions
        quatl: (N,4) payload quaternions [x,y,z,w]
        pld: (N,3) desired payload positions
        quatld: (N,4) desired payload quaternions
        alpha: list of (N,) cable angles
        fdes: list of (N,3) commanded forces
        fperp: list of (N,3) perpendicular forces
        pD: list of (N,3) drone positions
        quatD: list of (N,4) drone quaternions
        params: dict with keys l_arm, rho, doffset, l, alpha_des
        downsampling_factor: frame reduction factor
    Returns:
        The FuncAnimation object
    """
    # Input validation
    times = np.asarray(times)
    pl = np.asarray(pl)
    quatl = np.asarray(quatl)
    pld = np.asarray(pld)
    quatld = np.asarray(quatld)
    N_drones = len(alpha)
    if not all(len(lst) == N_drones for lst in [fdes, fperp, pD, quatD]):
        raise ValueError("All drone lists (fdes, fperp, pD, quatD) must have length N_drones")
    if not (pl.shape[0] == times.shape[0] == quatl.shape[0] == pld.shape[0] == quatld.shape[0]):
        raise ValueError("pl, quatl, pld, quatld, and times must all have the same length")
    # Colors: use hex codes for matplotlib compatibility
    b = "#0072B3"
    r = "#D95319"
    g = "#77AC30"
    y = "#E6B800"
    colors = [b, r, g]
    drone_colors = [b, y, g]

    # Extract params
    arm_length = params["l_arm"]
    rho_ = params["rho"]
    doffset = params["doffset"]
    l = params["l"]
    alpha_des = params["alpha_des"]


    rhomid = []
    Lci = []
    beta = []
    for i in range(N_drones):
        rhomid.append((rho_[2 * i] + rho_[2 * i + 1]) / 2)
        Lci.append((rho_[2 * i] - rho_[2 * i + 1]) / np.linalg.norm(rho_[2 * i] - rho_[2 * i + 1]))
        bij = (rho_[2 * i] - rho_[2 * i + 1])
        beta.append(np.arccos(np.linalg.norm(bij) / (2 * l[i])))

    # Downsample
    times = times[::downsampling_factor]
    pl = pl[::downsampling_factor]
    quatl = quatl[::downsampling_factor]
    pld = pld[::downsampling_factor]
    quatld = quatld[::downsampling_factor]
    for i in range(N_drones):
        alpha[i] = alpha[i][::downsampling_factor]
        fdes[i] = fdes[i][::downsampling_factor]
        fperp[i] = fperp[i][::downsampling_factor]
        pD[i] = pD[i][::downsampling_factor]
        quatD[i] = quatD[i][::downsampling_factor]

    # Initial setup
    plt_ = pl[0, :]
    qLt = quatl[0, :]
    Rlt = R.from_quat([qLt[0], qLt[1], qLt[2], qLt[3]]).as_matrix()

    pldt = pld[0, :]
    qLdt = quatld[0, :]
    Rldt = R.from_quat([qLdt[0], qLdt[1], qLdt[2], qLdt[3]]).as_matrix()

    rhot = []
    rho_des = []
    for i in range(N_drones):
        rhot.append(plt_ + Rlt @ rho_[2 * i])
        rhot.append(plt_ + Rlt @ rho_[2 * i + 1])
        rho_des.append(pldt + Rldt @ rho_[2 * i])
        rho_des.append(pldt + Rldt @ rho_[2 * i + 1])


    # Plot setup
    fig = plt.figure(figsize=_FIGSIZE)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=120)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    fig_title = fig.suptitle('', fontsize=15)

    plot_cage(0, 0, 10, 10, 4, ax, color="k", linewidth=1)
    load_lines = init_load(np.array(rhot), ax, color=b, linewidth=1)
    des_load_lines = init_load(np.array(rho_des), ax, color=b, linestyle="--", linewidth=1)


    # --- Persistent desired trajectory plot ---
    ax.plot(
        pld[:, 0], pld[:, 1], pld[:, 2],
        linestyle='--', color='red', linewidth=2, label='Desired Trajectory'
    )

    # --- Trail for load position (blue) ---
    from mpl_toolkits.mplot3d.art3d import Line3D
    (load_trail_line,) = ax.plot3D([], [], [], color='blue', linewidth=2, label='Load Trail')

    # Always add legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()

    drones_arms = []
    cable_lines = []
    cable_ref_lines = []
    force_arrows = []
    fperp_arrows = []

    for i in range(N_drones):
        pit = pD[i][0, :]
        qit = quatD[i][0, :]
        Rit = R.from_quat([qit[0], qit[1], qit[2], qit[3]]).as_matrix()
        rhoi1t = rhot[2 * i]
        rhoi2t = rhot[2 * i + 1]
        alphait = alpha[i][0]
        Rcialpha = R.from_rotvec(alphait * Lci[i]).as_matrix()
        Rcialphades = R.from_rotvec(alpha_des * Lci[i]).as_matrix()
        pAit_l = plt_ + Rlt @ (
            rhomid[i] + l[i] * math.sin(beta[i]) * Rcialpha @ rhomid[i] / np.linalg.norm(rhomid[i])
        )
        pireft = plt_ + Rlt @ (
            rhomid[i] + l[i] * math.sin(beta[i]) * Rcialphades @ rhomid[i] / np.linalg.norm(rhomid[i])
        )
        fdesit = fdes[i][0, :]
        fperpit = fperp[i][0, :]
        # Use hex string for color
        drones_arms.append(init_quad_arms(pit, Rit, arm_length, ax, color=drone_colors[i], linewidth=1))
        cable_lines.append(init_cable(rhoi1t, pAit_l, ax, color=y, linewidth=1))
        cable_lines.append(init_cable(rhoi2t, pAit_l, ax, color=y, linewidth=1))
        cable_ref_lines.append(init_cable(rhoi1t, pireft, ax, linestyle="--", color=y, linewidth=1))
        cable_ref_lines.append(init_cable(rhoi2t, pireft, ax, linestyle="--", color=y, linewidth=1))
        force_arrows.append(init_commanded_force(pit, fdesit, ax, linewidth=0.5))
        fperp_arrows.append(init_commanded_force(pit, fperpit, ax, color=b, linewidth=0.5))

    # Update function
    def update(frame: int) -> List[Any]:
        fig_title.set_text(f'Time= {times[frame]:.2f}')

        plt_ = pl[frame, :]
        qLt = quatl[frame, :]
        Rlt = R.from_quat([qLt[0], qLt[1], qLt[2], qLt[3]]).as_matrix()

        pldt = pld[frame, :]
        qLdt = quatld[frame, :]
        Rldt = R.from_quat([qLdt[0], qLdt[1], qLdt[2], qLdt[3]]).as_matrix()

        rhot = []
        rho_des = []
        for i in range(N_drones):
            rhot.append(plt_ + Rlt @ rho_[2 * i])
            rhot.append(plt_ + Rlt @ rho_[2 * i + 1])
            rho_des.append(pldt + Rldt @ rho_[2 * i])
            rho_des.append(pldt + Rldt @ rho_[2 * i + 1])

            pit = pD[i][frame, :]
            qit = quatD[i][frame, :]
            Rit = R.from_quat([qit[0], qit[1], qit[2], qit[3]]).as_matrix()

            alphait = alpha[i][frame]
            Rcialpha = R.from_rotvec(alphait * Lci[i]).as_matrix()
            Rcialphades = R.from_rotvec(alpha_des * Lci[i]).as_matrix()

            pAit_l = plt_ + Rlt @ (
                rhomid[i] + l[i] * math.sin(beta[i]) * Rcialpha @ rhomid[i] / np.linalg.norm(rhomid[i])
            )
            pireft = plt_ + Rlt @ (
                rhomid[i] + l[i] * math.sin(beta[i]) * Rcialphades @ rhomid[i] / np.linalg.norm(rhomid[i])
            )

            fdesit = fdes[i][frame, :]
            fperpit = fperp[i][frame, :]

            update_quad_arms(pit, Rit, arm_length, drones_arms[i])
            update_cable(rhot[2 * i], pAit_l, cable_lines[2 * i])
            update_cable(rhot[2 * i + 1], pAit_l, cable_lines[2 * i + 1])
            update_cable(rhot[2 * i], pireft, cable_ref_lines[2 * i])
            update_cable(rhot[2 * i + 1], pireft, cable_ref_lines[2 * i + 1])
            force_arrows[i] = update_commanded_force(pit, fdesit, force_arrows[i], ax)
            fperp_arrows[i] = update_commanded_force(pit, fperpit, fperp_arrows[i], ax, color=b)

        update_load(np.array(rhot), load_lines)
        update_load(np.array(rho_des), des_load_lines)

        # Update the load trail (blue)
        x = pl[:frame+1, 0]
        y = pl[:frame+1, 1]
        z = pl[:frame+1, 2]
        load_trail_line.set_data(x, y)
        load_trail_line.set_3d_properties(z)

        return drones_arms + cable_lines + cable_ref_lines + load_lines + [load_trail_line]

    ani = FuncAnimation(fig, update, frames=len(times), interval=_ANIMATION_INTERVAL_MS)
    plt.show()
    return ani
