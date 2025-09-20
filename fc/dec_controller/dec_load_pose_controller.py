
import numpy as np
import numpy.typing as npt
import copy
from fc.common_utils.math_utils import ssat, ssat_scalar, vee
from fc.flycrane_utils.dyn_model_structs import DynamicModel
from fc.flycrane_utils.flycrane_utils import FCCableState
from fc.flycrane_utils.dyn_model_structs import BodyState

# Constants
GRAVITY: float = 9.81



class DecLoadPoseController:
    """
    Decentralized Load Pose Controller for multi-drone systems.
    Computes wrench terms for load control, both distributed and local.

    Attributes
    ----------
    mi_ : float
        Robot mass.
    gv_ : np.ndarray
        Gravity vector (3,).
    kp_p_, kd_p_, ki_p_ : float
        Position control gains.
    kp_r_, kd_r_ : np.ndarray
        Rotation control gain matrices (3x3).
    ep_max_, er_max_, ev_max_, eoxy_max_, eoz_max_, ei_max_ : float
        Error saturation limits.
    ep_, ev_, eo_, eR_ : np.ndarray
        Error vectors (3,).
    Re_ : np.ndarray
        Rotation error matrix (3x3).
    uiparpa_ : np.ndarray
        Parallel input (3,).
    dql_, ddqldes_, ei_, wprop_, wder_, y_, wld_ : np.ndarray
        Various control and error terms (6,).
    Pis3perp_ : np.ndarray
        Projection matrix (3x3).
    """

    def __init__(self) -> None:
        """
        Initialize the DecLoadPoseController with default values.
        """
        self.mi_: float = 0.0
        self.gv_: np.ndarray = np.array([0.0, 0.0, GRAVITY])

        # Controller gains
        self.kp_p_: float = 0.0
        self.kd_p_: float = 0.0
        self.ki_p_: float = 0.0
        self.kp_r_: np.ndarray = np.eye(3)
        self.kd_r_: np.ndarray = np.eye(3)

        # Error saturations
        self.ep_max_: float = 0.0
        self.er_max_: float = 0.0
        self.ev_max_: float = 0.0
        self.eoxy_max_: float = 0.0
        self.eoz_max_: float = 0.0
        self.ei_max_: float = 0.0

        # Errors
        self.ep_: np.ndarray = np.zeros(3)
        self.ev_: np.ndarray = np.zeros(3)
        self.eo_: np.ndarray = np.zeros(3)
        self.eR_: np.ndarray = np.zeros(3)
        self.Re_: np.ndarray = np.eye(3)

        # Control-related variables
        self.uiparpa_: np.ndarray = np.zeros(3)
        self.dql_: np.ndarray = np.zeros(6)
        self.ddqldes_: np.ndarray = np.zeros(6)
        self.ei_: np.ndarray = np.zeros(6)
        self.Pis3perp_: np.ndarray = np.eye(3)

        # Proportional and derivative terms
        self.wprop_: np.ndarray = np.zeros(6)
        self.wder_: np.ndarray = np.zeros(6)
        self.y_: np.ndarray = np.zeros(6)

        # Desired wrench
        self.wld_: np.ndarray = np.zeros(6)

    def setParams(
        self,
        mi: float,
        kp_p: float,
        kp_r: npt.NDArray[np.float64],
        kd_p: float,
        kd_r: npt.NDArray[np.float64],
        ki_p: float,
        dt: float,
        ep_max: float,
        er_max: float,
        ev_max: float,
        eoxy_max: float,
        eoz_max: float,
        ei_max: float
    ) -> None:
        """
        Set controller parameters and initialize state.
        """
        if mi <= 0:
            raise ValueError("mi must be positive.")
        if kp_r.shape != (3, 3) or kd_r.shape != (3, 3):
            raise ValueError("kp_r and kd_r must be 3x3 matrices.")
        if dt <= 0:
            raise ValueError("dt must be positive.")
        self.mi_ = mi
        self.kp_p_ = kp_p
        self.kp_r_ = kp_r
        self.kd_p_ = kd_p
        self.kd_r_ = kd_r
        self.ki_p_ = ki_p * dt
        self.ep_max_ = ep_max
        self.er_max_ = er_max
        self.ev_max_ = ev_max
        self.eoxy_max_ = eoxy_max
        self.eoz_max_ = eoz_max
        self.ei_max_ = ei_max
        self.ei_ = np.zeros(6)
        self.uiparpa_ = np.zeros(3)
        self.wld_ = np.zeros(6)
        self.y_ = np.zeros(6)

    def getEi(self) -> np.ndarray:
        """Return a copy of the integral error (first 3 elements)."""
        return self.ei_[:3].copy()

    def getUiparpa(self) -> np.ndarray:
        """Return a copy of the parallel input."""
        return self.uiparpa_.copy()

    def getWld(self) -> np.ndarray:
        """Return a copy of the desired wrench."""
        return self.wld_.copy()

    def getY(self) -> np.ndarray:
        """Return a copy of the feedback term y."""
        return self.y_.copy()

    def getGains(self) -> list:
        """Return controller gains and error limits as a list."""
        return [
            self.kp_p_, self.kd_p_, self.ki_p_,
            self.ep_max_, self.er_max_,
            self.ev_max_, self.eoxy_max_, self.eoz_max_, self.ei_max_
        ]

    def doControl(
        self,
        dyn_model: DynamicModel,
        cable_state: FCCableState,
        load_state: BodyState,
        des_load_state: BodyState,
        Jqi: npt.NDArray[np.float64],
        dJqi: npt.NDArray[np.float64],
        dJalphai: npt.NDArray[np.float64]
    ) -> None:
        """
        Compute the control law for the decentralized load pose controller.
        Updates internal state with computed control terms.
        """
        # Input validation
        if not (isinstance(dyn_model, DynamicModel) and isinstance(cable_state, FCCableState)
                and isinstance(load_state, BodyState) and isinstance(des_load_state, BodyState)):
            raise TypeError("Invalid input types for doControl.")
        if not (isinstance(Jqi, np.ndarray) and isinstance(dJqi, np.ndarray) and isinstance(dJalphai, np.ndarray)):
            raise TypeError("Jqi, dJqi, dJalphai must be numpy arrays.")
        if Jqi.shape[0] != 3 or dJqi.shape[0] != 3 or dJalphai.shape[0] != 3:
            raise ValueError("Jqi, dJqi, dJalphai must have shape (3, ...)")

        # Compose the velocity twist (6D vector: [linear; angular])
        self.dql_[:3] = load_state.v
        self.dql_[3:] = load_state.world_omega

        self.ddqldes_[:3] = des_load_state.a
        self.ddqldes_[3:] = des_load_state.world_domega

        # Compute position and velocity errors
        self.ep_ = load_state.p - des_load_state.p
        self.ev_ = load_state.v - des_load_state.v

        # Compute rotation error
        self.Re_ = des_load_state.R.T @ load_state.R
        self.eR_ = 0.5 * des_load_state.R @ vee(self.Re_ - self.Re_.T)
        self.eo_ = load_state.world_omega - des_load_state.world_omega

        # Proportional terms (optionally saturate errors)
        # self.ep_ = ssat(self.ep_, self.ep_max_)
        # self.eR_ = ssat(self.eR_, self.er_max_)
        self.wprop_[:3] = -self.kp_p_ * self.ep_
        self.wprop_[3:] = -self.kp_r_ @ self.eR_

        # Derivative terms (optionally saturate errors)
        # self.ev_ = ssat(self.ev_, self.ev_max_)
        # self.eo_[0] = ssat_scalar(self.eo_[0], self.eoxy_max_)
        # self.eo_[1] = ssat_scalar(self.eo_[1], self.eoxy_max_)
        # self.eo_[2] = ssat_scalar(self.eo_[2], self.eoz_max_)
        self.wder_[:3] = -self.kd_p_ * self.ev_
        self.wder_[3:] = -self.kd_r_ @ self.eo_

        # "Almost linear" feedback term
        self.y_ = self.ddqldes_ + self.wprop_ + self.wder_

        # Common wrench (model-based)
        self.wld_ = dyn_model.Ml @ self.y_ + dyn_model.Cl @ self.dql_ + dyn_model.wgl

        # Decentralized part: projection matrix for cable direction
        self.Pis3perp_ = np.eye(3) - np.outer(cable_state.s3, cable_state.s3)

        # Compute parallel input
        self.uiparpa_ = self.mi_ * self.Pis3perp_ @ (
            self.gv_ + dJqi @ self.dql_ + Jqi @ self.y_ + dJalphai * cable_state.omega
        )

        # Integral term (currently only for position error)
        self.ei_[:3] += self.ki_p_ * self.ep_[:3]
        self.ei_[:3] = ssat(self.ei_[:3], self.ei_max_)

        # No explicit return (class stores results)
