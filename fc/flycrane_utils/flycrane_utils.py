# flycrane_utils.py
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from fc.common_utils.math_utils import skew, normalize


# -----------------------
# Data classes (parameters / states)
# -----------------------


@dataclass
class FCParams:
    """
    Fly-Cranne parameters corresponding to a single robot
    beta: base angle (radians)
    l: cable length
    Lrho1: attachment point 1 position in load frame (3-vector)
    Lrho2: attachment point 2 position in load frame (3-vector)
    Lrhoi: midpoint attachment position in load frame (3-vector)
    Lci: base direction vector in load frame (3-vector)
    """
    beta: float = 0.0
    l: float = 0.0
    Lrho1: np.ndarray = np.zeros(3)
    Lrho2: np.ndarray = np.zeros(3)
    Lrho: np.ndarray = np.zeros(3)
    Lc: np.ndarray = np.zeros(3)


@dataclass
class FCCableState:
    """
    State of a single pair of cables
    alpha: cable angle (radians)
    omega: cable angular velocity (radians/sec)
    domega: cable angular acceleration (radians/sec^2)
    s1: cable direction vector 1 (3-vector, world frame)
    s2: cable direction vector 2 (3-vector, world frame)
    s3: cable normal vector (3-vector, world frame)
    l1: length of cable segment 1
    l2: length of cable segment 2
    """
    alpha: float = 0.0
    omega: float = 0.0
    domega: float = 0.0
    s1: np.ndarray = np.zeros(3)
    s2: np.ndarray = np.zeros(3)
    s3: np.ndarray = np.zeros(3)
    l1: float = 0.0
    l2: float = 0.0


# -----------------------
# Kinematic conversions
# -----------------------
def dronepos2attachpos(pi: np.ndarray, Ri: R, doffset: np.ndarray) -> np.ndarray:
    """
    pi: (3,) world frame
    Ri: drone orientation as a scipy Rotation
    doffset: (3,) vector in drone body frame (attach point relative to drone origin)
    returns: attach position in world frame
    """
    return pi + Ri.apply( doffset )


def dronevel2attachvel(drone_vel: np.ndarray, Ri: R, drone_body_omega: np.ndarray,
                       doffset: np.ndarray) -> np.ndarray:
    """
    drone_vel: (3,) linear velocity of drone in world frame
    drone_body_omega: (3,) angular velocity expressed in drone body frame
    Ri: rotation of drone (scipy Rotation)
    doffset: offset vector in drone body frame
    returns: attach point linear velocity in world frame
    """
    # angular velocity in world frame = R * body_omega
    omega_world = Ri.apply( drone_body_omega )
    return drone_vel + np.cross(omega_world, Ri.apply( doffset ))


def attachpos2dronepos(attach_pos: np.ndarray, Ri: R, doffset: np.ndarray) -> np.ndarray:
    return attach_pos - Ri.apply( doffset )


def attachvel2dronevel(attach_vel: np.ndarray, Ri: R, drone_body_omega: np.ndarray,
                       doffset: np.ndarray) -> np.ndarray:
    omega_world = Ri.apply( drone_body_omega )
    return attach_vel - np.cross(omega_world, Ri.apply( doffset ))


# -----------------------
# Relative velocity/position wrt load (Bi frame computations)
# -----------------------
def computeViBi_from_viL_world(viL: np.ndarray, world_omegal: np.ndarray, rhoi: np.ndarray) -> np.ndarray:
    """
    viL: robot velocity relative to load (world frame) i.e., vi - vl
    world_omegal: load angular velocity (world frame)
    rhoi: midpoint attachment point in world frame
    returns: viBi (relative velocity from Bi frame viewpoint)
    """
    return viL - skew(world_omegal) @ rhoi


def computeViBi(Rl: np.ndarray, vi: np.ndarray, vl: np.ndarray, world_omegal: np.ndarray,
                Lrhoi: np.ndarray) -> np.ndarray:
    """
    Overload: accepts load rotation Rl (3x3), vi, vl, world_omegal, and Lrhoi (in load frame)
    """
    return (vi - vl) - skew(world_omegal) @ (Rl @ Lrhoi)


def computepiBi_from_world(pi: np.ndarray, pl: np.ndarray, rhoi: np.ndarray) -> np.ndarray:
    """
    rho is midpoint attachment in world frame.
    returns vector from midpoint to robot attach point (world)
    """
    return (pi - pl) - rhoi


def computepiBi(pi: np.ndarray, pl: np.ndarray, Rl: np.ndarray, Lrhoi: np.ndarray) -> np.ndarray:
    """
    version where Lrhoi given in load frame; Rl is rotation from load to world
    """
    return (pi - pl) - Rl @ Lrhoi


# -----------------------
# Jacobians
# -----------------------
def computeJqi_from_pil(pil: np.ndarray) -> np.ndarray:
    """
    pil: relative position (pi - pl) in world frame
    returns 3x6 Jacobian (first 3 cols identity, next 3 cols = -skew(pil))
    """
    J = np.zeros((3, 6))
    J[:, 0:3] = np.eye(3)
    J[:, 3:6] = -skew(pil)
    return J


def computeJqi(pi: np.ndarray, pl: np.ndarray) -> np.ndarray:
    pil = pi - pl
    return computeJqi_from_pil(pil)


def computeJqiDerivative_from_vil(vil: np.ndarray) -> np.ndarray:
    dJ = np.zeros((3, 6))
    dJ[:, 3:6] = -skew(vil)
    return dJ


def computeJqiDerivative(vi: np.ndarray, vl: np.ndarray) -> np.ndarray:
    vil = vi - vl
    return computeJqiDerivative_from_vil(vil)


# -----------------------
# Partial Jacobian wrt cable angle alpha
# -----------------------
def computeJalphai_from_piBi_ci(piBi: np.ndarray, ci: np.ndarray) -> np.ndarray:
    """
    Returns vector (3,) equal to skew(ci) * piBi
    """
    return skew(ci) @ piBi


def computeJalphai(pi: np.ndarray, pl: np.ndarray, Rl: np.ndarray, Lci: np.ndarray,
                   Lrhoi: np.ndarray) -> np.ndarray:
    piBi = computepiBi(pi, pl, Rl, Lrhoi)
    ci = Rl @ Lci
    return computeJalphai_from_piBi_ci(piBi, ci)


def computeDJalphai_from_piBi_world(piBi: np.ndarray, world_omegal: np.ndarray,
                                    ci: np.ndarray, viBi: np.ndarray) -> np.ndarray:
    """
    Uses the algebraic identity from C++:
      (world_omegal.cross(ci)).cross(piBi) + ci.cross(viBi)
    """
    return np.cross(np.cross(world_omegal, ci), piBi) + np.cross(ci, viBi)


def computeDJalphai(pi: np.ndarray, pl: np.ndarray, Rl: np.ndarray, vi: np.ndarray,
                    vl: np.ndarray, world_omegal: np.ndarray, Lci: np.ndarray,
                    Lrhoi: np.ndarray) -> np.ndarray:
    piBi = computepiBi(pi, pl, Rl, Lrhoi)
    ci = Rl @ Lci
    viBi = computeViBi(Rl, vi, vl, world_omegal, Lrhoi)
    return computeDJalphai_from_piBi_world(piBi, world_omegal, ci, viBi)


# -----------------------
# Cable directions and normal
# -----------------------
def computeCableDirections(pi: np.ndarray, pl: np.ndarray, Rl: np.ndarray,
                           Lrhoi1: np.ndarray, Lrhoi2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rhoi1 = Rl @ Lrhoi1
    rhoi2 = Rl @ Lrhoi2
    s1 = (pl + rhoi1 - pi)
    s2 = (pl + rhoi2 - pi)
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s1, s2


def computeCableNormal(pi: np.ndarray, pl: np.ndarray, Rl: np.ndarray,
                       Lrhoi1: np.ndarray, Lrhoi2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s1, s2 = computeCableDirections(pi, pl, Rl, Lrhoi1, Lrhoi2)
    # negative of cross(s1, s2), normalized
    s3 = -(np.cross(s1, s2))
    s3 = normalize(s3)
    return s1, s2, s3


# -----------------------
# Cable angle (alpha) and derivative
# -----------------------
def computeCablesAngle(Lpil: np.ndarray, Lrhoi: np.ndarray) -> float:
    """
    Lpil: robot position in load frame (vector from load origin to robot attach)
    Lrhoi: midpoint attach in load frame
    returns alpha (signed)
    """
    h = Lpil - Lrhoi
    hproj = np.array([h[0], h[1], 0.0])
    uhproj = normalize(hproj)
    uh = normalize(h)
    # handle degenerate cases
    if np.linalg.norm(hproj) == 0 or np.linalg.norm(h) == 0:
        return 0.0
    calpha = np.copysign(np.dot(uhproj, uh), np.dot(Lrhoi, uhproj))
    cross_norm = np.linalg.norm(np.cross(uhproj, uh))
    salpha = np.copysign(cross_norm, h[2])
    return float(np.arctan2(salpha, calpha))


def computeCablesAngleDerivative(Lpil: np.ndarray, Lvil: np.ndarray,
                                 Lrhoi: np.ndarray, Lci: np.ndarray) -> float:
    """
    returns d(alpha)/dt using formula vic / norm(h)
    where ui = Lci x h normalized, vic = ui dot Lvil
    """
    h = Lpil - Lrhoi
    if np.allclose(h, 0.0):
        return 0.0
    ui = normalize(np.cross(Lci, h))
    vic = np.dot(ui, Lvil)
    return float(vic / np.linalg.norm(h))


# -----------------------
# Direct geometry (forward position of attach point)
# -----------------------
def computeDirectGeometry(pl: np.ndarray, Rl: np.ndarray, alpha_i: float,
                          Lrhoi: np.ndarray, Lci: np.ndarray, li: float,
                          beta_i: float) -> np.ndarray:
    """
    Compute attach position in world frame given load pose and cable angle.
    pl: load position world (3,)
    Rl: rotation matrix from load frame to world frame (3x3)
    alpha_i: cable rotation around Lci
    Lrhoi: midpoint attach in load frame
    Lci: base direction vector in load frame (axis of rotation)
    li: cable length
    beta_i: base constant angle of isosceles triangle
    """
    # RLci_alpha rotates around Lci by alpha_i (in load frame)
    RLci_alpha = R.from_rotvec(alpha_i * normalize(Lci))
    # vector from midpoint to robot attach point in load frame
    LpiBi = li * np.sin(beta_i) * (RLci_alpha.apply( normalize(Lrhoi) ))
    return pl + Rl @ (Lrhoi + LpiBi)


# -----------------------
# Direct kinematics (velocity / acceleration)
# -----------------------
def computeDirectKinematics(dql: np.ndarray, omega_alpha_i: float, Jqi: np.ndarray,
                            Jalphai: np.ndarray) -> np.ndarray:
    """
    dql: 6x1 twist (vL; omegaL) as array with shape (6,) or (6,1)
    Jqi: (3,6)
    Jalphai: (3,)
    """
    dql = np.asarray(dql).reshape(6,)
    return Jqi @ dql + Jalphai * omega_alpha_i


def computeSecondOrderDirectKinematics(dql: np.ndarray, omega_alpha_i: float,
                                       ddql: np.ndarray, domega_alpha_i: float,
                                       Jqi: np.ndarray, Jalphai: np.ndarray,
                                       dJqi: np.ndarray, dJalphai: np.ndarray) -> np.ndarray:
    """
    returns acceleration of attach point in world frame
    dJqi: (3,6), dJalphai: (3,)
    """
    dql = np.asarray(dql).reshape(6,)
    ddql = np.asarray(ddql).reshape(6,)
    return dJqi @ dql + Jqi @ ddql + dJalphai * omega_alpha_i + Jalphai * domega_alpha_i
