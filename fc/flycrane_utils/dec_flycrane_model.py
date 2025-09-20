# dec_flycrane.py

from __future__ import annotations
import numpy as np


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


class DFlyCraneModel:
    def __init__(self) -> None:
        self.fc_params_ = FCParams()
        self.dynamic_model_ = DynamicModel()

        self.Jl_ = np.zeros((3, 3))
        self.ml_ = 0.0
        self.wgl_ = np.zeros(6)
        self.Comega_ = np.zeros((3, 3))

        self.Jqi_ = np.zeros((3, 6))
        self.dJqi_ = np.zeros((3, 6))
        self.Jalphai_ = np.zeros(3)
        self.dJalphai_ = np.zeros(3)

        self.doffset_ = np.zeros(3)

        self.drone_attaching_state_ = BodyState()
        self.cable_state_ = FCCableState()

        self.FC_PARAMETERS_INITIALIZED_ = False
        self.DYNPARAMETERS_INITIALIZED_ = False

    # ======================================
    # Initialization methods

    def setFCParams(
        self,
        Lrho1: np.ndarray,
        Lrho2: np.ndarray,
        l: float,
        doffset: np.ndarray,
    ) -> None:
        self.fc_params_.l = l
        self.fc_params_.Lrho1 = Lrho1
        self.fc_params_.Lrho2 = Lrho2
        self.doffset_ = doffset

        Lrho12_ = Lrho1 - Lrho2
        self.fc_params_.Lc = Lrho12_ / np.linalg.norm(Lrho12_)
        self.fc_params_.Lrho = 0.5 * (Lrho1 + Lrho2)
        self.fc_params_.beta = np.arccos(np.linalg.norm(Lrho12_) / (2.0 * l))

        self.FC_PARAMETERS_INITIALIZED_ = True

    def setDynParams(self, ml: float, Jl: np.ndarray) -> None:
        self.dynamic_model_.Ml = np.zeros((6, 6))
        self.dynamic_model_.Ml[:3, :3] = np.eye(3) * ml
        self.dynamic_model_.Cl = np.zeros((6, 6))
        self.dynamic_model_.wgl = np.zeros(6)
        self.dynamic_model_.wgl[2] = ml * 9.81  # gravity vector

        self.Jl_ = Jl
        self.ml_ = ml

        self.DYNPARAMETERS_INITIALIZED_ = True

    def isInitialized(self) -> bool:
        return self.FC_PARAMETERS_INITIALIZED_ and self.DYNPARAMETERS_INITIALIZED_

    # ======================================
    # Model update

    def updateModel(self, drone_state: BodyState, load_state: BodyState) -> None:
        self.computeDynamicModel(load_state)
        self.computeDroneAttachingState(drone_state)
        self.computeFlyCraneModel(load_state)

    def computeDynamicModel(self, load_state: BodyState) -> None:
        self.dynamic_model_.Ml[3:6, 3:6] = load_state.R @ self.Jl_ @ load_state.R.T
        self.Comega_ = skew(load_state.world_omega) @ self.dynamic_model_.Ml[3:6, 3:6]
        self.dynamic_model_.Cl[3:6, 3:6] = self.Comega_

    def computeDroneAttachingState(self, drone_state: BodyState) -> None:
        self.drone_attaching_state_.p = dronepos2attachpos(
            drone_state.p, drone_state.quat, self.doffset_
        )
        self.drone_attaching_state_.v = dronevel2attachvel(
            drone_state.v, drone_state.quat, drone_state.body_omega, self.doffset_
        )

    def computeFlyCraneModel(self, load_state: BodyState) -> None:
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

    def computeLoadAccel(self, load_state: BodyState, wl: np.ndarray) -> np.ndarray:
        dql = np.hstack([load_state.v, load_state.world_omega])
        Mlddql_ = wl - self.dynamic_model_.Cl @ dql - self.dynamic_model_.wgl

        ddql_ = np.zeros(6)
        ddql_[:3] = Mlddql_[:3] / self.ml_
        ddql_[3:] = np.linalg.solve(
            self.dynamic_model_.Ml[3:6, 3:6], Mlddql_[3:]
        )
        return ddql_

    # ======================================
    # Getters

    def getFCParams(self) -> FCParams:
        return self.fc_params_

    def getDynamicModel(self) -> DynamicModel:
        return self.dynamic_model_

    def getCableState(self) -> FCCableState:
        return self.cable_state_

    def getDroneAttachingState(self) -> BodyState:
        return self.drone_attaching_state_

    def getJqi(self) -> np.ndarray:
        return self.Jqi_

    def getdJqi(self) -> np.ndarray:
        return self.dJqi_

    def getJalphai(self) -> np.ndarray:
        return self.Jalphai_

    def getdJalphai(self) -> np.ndarray:
        return self.dJalphai_
