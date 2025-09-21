
import numpy as np
from dataclasses import dataclass, field
from typing import Any

from scipy.spatial.transform import Rotation as R

@dataclass
class DynamicModel:
    """
    Encapsulates the dynamic model matrices and vectors.
    Attributes:
        Ml: Mass matrix (6x6)
        Cl: Coriolis matrix (6x6)
        wgl: Gravity wrench (6,)
        Mlt: Total mass matrix (6x6)
        Clt: Total Coriolis matrix (6x6)
        wglt: Total gravity wrench (6,)
    """
    Ml: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))
    Cl: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))
    wgl: np.ndarray = field(default_factory=lambda: np.zeros(6))
    Mlt: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))
    Clt: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))
    wglt: np.ndarray = field(default_factory=lambda: np.zeros(6))

@dataclass
class BodyState:
    """
    Represents the state of the load or a drone.
    Attributes:
        p: Position (3,)
        v: Velocity (3,)
        quat: Orientation as a quaternion (4,)
        R: Rotation matrix (3x3)
        body_omega: Angular velocity in body frame (3,)
        world_omega: Angular velocity in world frame (3,)
        a: Acceleration (3,)
        world_domega: Angular acceleration in world frame (3,)
        body_domega: Angular acceleration in body frame (3,)
    """
    p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(default_factory=lambda: np.zeros(4))
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    body_omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    world_omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    a: np.ndarray = field(default_factory=lambda: np.zeros(3))
    world_domega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    body_domega: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        # Ensure unit quaternion by default
        self.quat[3] = 1.0
        self.R = np.eye(3)

    def set_orientation_from_quat(self, quat: np.ndarray) -> None:
        """
        Set the rotation matrix from a quaternion using scipy Rotation.
        """
        if quat.shape != (4,):
            raise ValueError("Quaternion must be a 4-dimensional vector.")
        self.quat = quat.copy() / np.linalg.norm(quat)  # Normalize the quaternion
        rot = R.from_quat(self.quat)
        self.R = rot.as_matrix()

    def set_orientation_from_R(self, Rmat: np.ndarray) -> None:
        """
        Set the quaternion from a rotation matrix using scipy Rotation.
        """
        if Rmat.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3.")
        self.R = Rmat.copy()
        rot = R.from_matrix(self.R)
        self.quat = rot.as_quat() / np.linalg.norm(rot.as_quat())

    def set_angular_velocity_from_body(self, body_omega: np.ndarray) -> None:
        """
        Set angular velocity in body frame and update world frame.
        """
        if body_omega.shape != (3,):
            raise ValueError("Angular velocity must be a 3-dimensional vector.")
        self.body_omega = body_omega.copy()
        self.world_omega = self.R @ self.body_omega

    def set_angular_velocity_from_world(self, world_omega: np.ndarray) -> None:
        """
        Set angular velocity in world frame and update body frame.
        """
        if world_omega.shape != (3,):
            raise ValueError("Angular velocity must be a 3-dimensional vector.")
        self.world_omega = world_omega.copy()
        self.body_omega = self.R.T @ self.world_omega

    def set_angular_acceleration_from_body(self, body_domega: np.ndarray) -> None:
        """
        Set angular acceleration in body frame and update world frame.
        """
        if body_domega.shape != (3,):
            raise ValueError("Angular acceleration must be a 3-dimensional vector.")
        self.body_domega = body_domega.copy()
        self.world_domega = self.R @ self.body_domega

    def set_angular_acceleration_from_world(self, world_domega: np.ndarray) -> None:
        """
        Set angular acceleration in world frame and update body frame.
        """
        if world_domega.shape != (3,):
            raise ValueError("Angular acceleration must be a 3-dimensional vector.")
        self.world_domega = world_domega.copy()
        self.body_domega = self.R.T @ self.world_domega