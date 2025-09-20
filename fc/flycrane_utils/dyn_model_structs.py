import numpy as np

class DynamicModel:
    """
    Encapsulates the dynamic model matrices and vectors
    """
    def __init__(self):
        self.Ml = np.zeros((6, 6))    # Mass matrix
        self.Cl = np.zeros((6, 6))    # Coriolis matrix
        self.wgl = np.zeros(6)        # Gravity wrench
        self.Mlt = np.zeros((6, 6))   # Total mass matrix
        self.Clt = np.zeros((6, 6))   # Total Coriolis matrix
        self.wglt = np.zeros(6)       # Total gravity wrench

class BodyState:
    """
    Represents the state of the load or a drone
    """
    def __init__(self):
        self.p = np.zeros(3)            # Position
        self.v = np.zeros(3)            # Velocity
        self.quat = np.zeros(4)         # Orientation as a quaternion
        self.quat[3] = 1.0              # Initialize as a unit quaternion
        self.R = np.eye(3)              # Rotation matrix
        self.body_omega = np.zeros(3)   # Angular velocity in body frame
        self.world_omega = np.zeros(3)  # Angular velocity in world frame
        self.a = np.zeros(3)            # Acceleration
        self.world_domega = np.zeros(3) # Angular acceleration in world frame
        self.body_domega = np.zeros(3)  # Angular acceleration in body frame