
# dec_flycrane.py
from __future__ import annotations
import numpy as np
import numpy.typing as npt
import copy
from fc.common_utils.math_utils import skew
from fc.flycrane_utils.dyn_model_structs import BodyState, DynamicModel
from fc.flycrane_utils.flycrane_utils import (
    dronepos2attachpos,
    dronevel2attachvel,
    computeCablesAngle,
    computeCablesAngleDerivative,
    computeCableNormal,
    computeJqi,
    computeJqiDerivative,
    computeJalphai,
    computeDJalphai,
    FCParams,
    FCCableState,
)

# Constants
GRAVITY: float = 9.81
TWO: float = 2.0



class DFlyCraneModel:
    """
    Decentralized FlyCrane Model for multi-drone systems.
    Handles model parameters, state updates, and Jacobian calculations.

    Attributes
    ----------
    fc_params_ : FCParams
        FlyCrane geometric parameters.
    dynamic_model_ : DynamicModel
        Dynamic model of the load.
    Jl_ : np.ndarray
        Inertia matrix of the load (3x3).
    ml_ : float
        Mass of the load.
    wgl_ : np.ndarray
        Gravity wrench (6,).
    Comega_ : np.ndarray
        Coriolis matrix (3x3).
    Jqi_, dJqi_ : np.ndarray
        Jacobians (3x6).
    Jalphai_, dJalphai_ : np.ndarray
        Jacobians (3,).
    doffset_ : np.ndarray
        Drone offset vector (3,).
    drone_attaching_state_ : BodyState
        State at drone attachment point.
    cable_state_ : FCCableState
        Cable state.
    FC_PARAMETERS_INITIALIZED_ : bool
        True if FC parameters are set.
    DYNPARAMETERS_INITIALIZED_ : bool
        True if dynamic parameters are set.
    """
    def __init__(self) -> None:
        self.fc_params_: FCParams = FCParams()
        self.dynamic_model_: DynamicModel = DynamicModel()

        self.Jl_: np.ndarray = np.zeros((3, 3))
        self.ml_: float = 0.0
        self.wgl_: np.ndarray = np.zeros(6)
        self.Comega_: np.ndarray = np.zeros((3, 3))

        self.Jqi_: np.ndarray = np.zeros((3, 6))
        self.dJqi_: np.ndarray = np.zeros((3, 6))
        self.Jalphai_: np.ndarray = np.zeros(3)
        self.dJalphai_: np.ndarray = np.zeros(3)

        self.doffset_: np.ndarray = np.zeros(3)

        self.drone_attaching_state_: BodyState = BodyState()
        self.cable_state_: FCCableState = FCCableState()

        self.FC_PARAMETERS_INITIALIZED_: bool = False
        self.DYNPARAMETERS_INITIALIZED_: bool = False

    # ======================================
    # Initialization methods

    def setFCParams(
        self,
        Lrho1: npt.NDArray[np.float64],
        Lrho2: npt.NDArray[np.float64],
        l: float,
        doffset: npt.NDArray[np.float64],
    ) -> None:
        """
        Set FlyCrane geometric parameters.
        """
        if Lrho1.shape != (3,) or Lrho2.shape != (3,):
            raise ValueError("Lrho1 and Lrho2 must be 3-element vectors.")
        if not isinstance(l, float) or l <= 0:
            raise ValueError("l must be a positive float.")
        if doffset.shape != (3,):
            raise ValueError("doffset must be a 3-element vector.")
        self.fc_params_.l = l
        self.fc_params_.Lrho1 = Lrho1.copy()
        self.fc_params_.Lrho2 = Lrho2.copy()
        self.doffset_ = doffset.copy()

        Lrho12_ = Lrho1 - Lrho2
        norm_Lrho12 = np.linalg.norm(Lrho12_)
        if norm_Lrho12 == 0:
            raise ValueError("Lrho1 and Lrho2 must not be identical.")
        self.fc_params_.Lc = Lrho12_ / norm_Lrho12
        self.fc_params_.Lrho = 0.5 * (Lrho1 + Lrho2)
        self.fc_params_.beta = np.arccos(norm_Lrho12 / (TWO * l))

        self.FC_PARAMETERS_INITIALIZED_ = True

    def setDynParams(self, ml: float, Jl: npt.NDArray[np.float64]) -> None:
        """
        Set dynamic parameters for the FlyCrane model.
        """
        if ml <= 0:
            raise ValueError("ml must be positive.")
        if Jl.shape != (3, 3):
            raise ValueError("Jl must be a 3x3 matrix.")
        self.dynamic_model_.Ml = np.zeros((6, 6))
        self.dynamic_model_.Ml[:3, :3] = np.eye(3) * ml
        self.dynamic_model_.Cl = np.zeros((6, 6))
        self.dynamic_model_.wgl = np.zeros(6)
        self.dynamic_model_.wgl[2] = ml * GRAVITY  # gravity vector

        self.Jl_ = Jl.copy()
        self.ml_ = ml

        self.DYNPARAMETERS_INITIALIZED_ = True

    def isInitialized(self) -> bool:
        """Return True if both FC and dynamic parameters are initialized."""
        return self.FC_PARAMETERS_INITIALIZED_ and self.DYNPARAMETERS_INITIALIZED_

    # ======================================
    # Model update

    def updateModel(self, drone_state: BodyState, load_state: BodyState) -> None:
        """
        Update the model with current drone and load states.
        """
        self.computeDynamicModel(load_state)
        self.computeDroneAttachingState(drone_state)
        self.computeFlyCraneModel(load_state)

    def computeDynamicModel(self, load_state: BodyState) -> None:
        """
        Compute the dynamic model (inertia, Coriolis) for the current load state.
        """
        self.dynamic_model_.Ml[3:6, 3:6] = load_state.R @ self.Jl_ @ load_state.R.T
        self.Comega_ = skew(load_state.world_omega) @ self.dynamic_model_.Ml[3:6, 3:6]
        self.dynamic_model_.Cl[3:6, 3:6] = self.Comega_

    def computeDroneAttachingState(self, drone_state: BodyState) -> None:
        """
        Compute the state at the drone's attachment point.
        """
        self.drone_attaching_state_.p = dronepos2attachpos(
            drone_state.p, drone_state.R, self.doffset_
        )
        self.drone_attaching_state_.v = dronevel2attachvel(
            drone_state.v, drone_state.R, drone_state.body_omega, self.doffset_
        )

    def computeFlyCraneModel(self, load_state: BodyState) -> None:
        """
        Compute cable geometry, Jacobians, and cable state for the current load state.
        """
        pil = self.drone_attaching_state_.p - load_state.p
        Lpil = load_state.R.T @ pil
        Lvil = load_state.R.T @ (
            self.drone_attaching_state_.v
            - load_state.v
            - skew(load_state.world_omega) @ pil
        )

        self.cable_state_.alpha = computeCablesAngle(Lpil, self.fc_params_.Lrho)
        self.cable_state_.omega = computeCablesAngleDerivative(
            Lpil, Lvil, self.fc_params_.Lrho, self.fc_params_.Lc
        )
        self.cable_state_.s1, self.cable_state_.s2, self.cable_state_.s3 = computeCableNormal(
            self.drone_attaching_state_.p,
            load_state.p,
            load_state.R,
            self.fc_params_.Lrho1,
            self.fc_params_.Lrho2,
        )

        self.Jqi_ = computeJqi(self.drone_attaching_state_.p, load_state.p)
        self.dJqi_ = computeJqiDerivative(self.drone_attaching_state_.v, load_state.v)
        self.Jalphai_ = computeJalphai(
            self.drone_attaching_state_.p,
            load_state.p,
            load_state.R,
            self.fc_params_.Lc,
            self.fc_params_.Lrho,
        )
        self.dJalphai_ = computeDJalphai(
            self.drone_attaching_state_.p,
            load_state.p,
            load_state.R,
            self.drone_attaching_state_.v,
            load_state.v,
            load_state.world_omega,
            self.fc_params_.Lc,
            self.fc_params_.Lrho,
        )

    def computeLoadAccel(self, load_state: BodyState, wl: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Compute the acceleration of the load given the current state and wrench.
        """
        dql = np.hstack([load_state.v, load_state.world_omega])
        Mlddql_ = wl - self.dynamic_model_.Cl @ dql - self.dynamic_model_.wgl

        ddql_ = np.zeros(6)
        if self.ml_ == 0:
            raise ZeroDivisionError("ml_ (mass) is zero.")
        ddql_[:3] = Mlddql_[:3] / self.ml_
        try:
            ddql_[3:] = np.linalg.solve(self.dynamic_model_.Ml[3:6, 3:6], Mlddql_[3:])
        except np.linalg.LinAlgError:
            raise ValueError("Singular inertia matrix in computeLoadAccel.")
        return ddql_

    # ======================================
    # Getters

    def getFCParams(self) -> FCParams:
        """Return a copy of the FlyCrane parameters."""
        return copy.deepcopy(self.fc_params_)

    def getDynamicModel(self) -> DynamicModel:
        """Return a copy of the dynamic model."""
        return copy.deepcopy(self.dynamic_model_)

    def getCableState(self) -> FCCableState:
        """Return a copy of the cable state."""
        return copy.deepcopy(self.cable_state_)

    def getDroneAttachingState(self) -> BodyState:
        """Return a copy of the drone attaching state."""
        return copy.deepcopy(self.drone_attaching_state_)

    def getJqi(self) -> np.ndarray:
        """Return a copy of the Jqi Jacobian."""
        return self.Jqi_.copy()

    def getdJqi(self) -> np.ndarray:
        """Return a copy of the dJqi Jacobian."""
        return self.dJqi_.copy()

    def getJalphai(self) -> np.ndarray:
        """Return a copy of the Jalphai Jacobian."""
        return self.Jalphai_.copy()

    def getdJalphai(self) -> np.ndarray:
        """Return a copy of the dJalphai Jacobian."""
        return self.dJalphai_.copy()
