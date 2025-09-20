import numpy as np
import numpy.typing as npt

from fc.flycrane_utils.dyn_model_structs import BodyState, DynamicModel
from fc.flycrane_utils.flycrane_utils import FCCableState
from fc.flycrane_utils.dec_flycrane_model import DFlyCraneModel
from fc.dec_controller.dec_load_pose_controller import DecLoadPoseController
from fc.dec_controller.fc_cable_controller import FCCableController 


class ConfigController:
    """
    Configuration Controller Node for decentralized multi-drone systems.
    Handles initialization and parameter setting for load and cable controllers.

    Attributes
    ----------
    FCCABLE_CONTROLLER_INITIALIZED_ : bool
        True if the cable controller is initialized.
    DEC_LOAD_POSE_CONTROLLER_INITIALIZED_ : bool
        True if the load pose controller is initialized.
    kdamp_ : float
        Damping gain.
    dfc_model_ : DFlyCraneModel
        FlyCrane model instance.
    load_controller_ : DecLoadPoseController
        Load pose controller instance.
    cable_controller_ : FCCableController
        Cable controller instance.
    fdes_ : np.ndarray
        Desired force (3,).
    uipa_ : np.ndarray
        Parallel input (3,).
    """

    def __init__(self) -> None:
        """
        Initialize the ConfigController with default values and state variables.
        """
        self.FCCABLE_CONTROLLER_INITIALIZED_: bool = False
        self.DEC_LOAD_POSE_CONTROLLER_INITIALIZED_: bool = False

        self.kdamp_: float = 0.0

        self.dfc_model_: DFlyCraneModel = DFlyCraneModel()
        self.load_controller_: DecLoadPoseController = DecLoadPoseController()
        self.cable_controller_: FCCableController = FCCableController()

        self.fdes_: np.ndarray = np.zeros(3)
        self.uipa_: np.ndarray = np.zeros(3)

    def initialize(self, kdamp: float) -> None:
        """
        Initialize the controller with a damping gain and reset initialization flags.
        """
        if kdamp < 0:
            raise ValueError("kdamp must be non-negative.")
        self.kdamp_ = kdamp
        self.FCCABLE_CONTROLLER_INITIALIZED_ = False
        self.DEC_LOAD_POSE_CONTROLLER_INITIALIZED_ = False


    def setFCParams(
        self,
        Lrho1: np.ndarray,
        Lrho2: np.ndarray,
        l: float,
        doffset: np.ndarray
    ) -> None:
        """
        Set FlyCrane model parameters.
        Parameters
        ----------
        Lrho1 : np.ndarray
            First set of cable attachment points.
        Lrho2 : np.ndarray
            Second set of cable attachment points.
        l : float
            Cable length.
        doffset : np.ndarray
            Offset vector.
        """
        if Lrho1.shape != Lrho2.shape:
            raise ValueError("Lrho1 and Lrho2 must have the same shape.")
        if not isinstance(l, float) or l <= 0:
            raise ValueError("l must be a positive float.")
        self.dfc_model_.setFCParams(Lrho1, Lrho2, l, doffset)

    def setDynParams(self, ml: float, Jl: np.ndarray) -> None:
        """
        Set dynamic parameters for the FlyCrane model.
        Parameters
        ----------
        ml : float
            Mass of the load.
        Jl : np.ndarray
            Inertia matrix of the load.
        """
        if ml <= 0:
            raise ValueError("ml must be positive.")
        self.dfc_model_.setDynParams(ml, Jl)

    def setParamsDecLoadPoseController(
        self,
        mR: float,
        kp_p: float,
        kp_r: np.ndarray,
        kd_p: float,
        kd_r: np.ndarray,
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
        Set parameters for the decentralized load pose controller.
        """
        if not self.dfc_model_.isInitialized():
            raise RuntimeError("Controller not initialized. Call setFCParams() and setDynParams() first.")
        self.load_controller_.setParams(
            mR, kp_p, kp_r, kd_p, kd_r, ki_p, dt,
            ep_max, er_max, ev_max, eoxy_max, eoz_max, ei_max
        )
        self.DEC_LOAD_POSE_CONTROLLER_INITIALIZED_ = True

    def setParamsFCCableController(
        self,
        mRi: float,
        kdamp: float,
        kpalpha: float,
        kdalpha: float
    ) -> None:
        """
        Set parameters for the cable controller.
        """
        if self.dfc_model_ is None or not self.dfc_model_.isInitialized():
            raise RuntimeError("Controller not initialized. Call setFCParams() and setDynParams() first.")

        self.cable_controller_.setParams(mRi, kpalpha, kdalpha, kdamp)
        self.FCCABLE_CONTROLLER_INITIALIZED_ = True

    def doControl(
        self,
        load_state: BodyState,
        drone_state: BodyState,
        des_load_state: BodyState,
        des_cable_state: FCCableState,
        uidynapar: npt.NDArray[np.float64]
    ) -> None:
        """
        Run the full control pipeline: update model, run load pose controller, and cable controller.

        Parameters
        ----------
        load_state : BodyState
            Current state of the load.
        drone_state : BodyState
            Current state of the drone.
        des_load_state : BodyState
            Desired state of the load.
        des_cable_state : FCCableState
            Desired state of the cable.
        uidynapar : np.ndarray
            Dynamic parallel input (shape (3,)).
        """
        if not (self.DEC_LOAD_POSE_CONTROLLER_INITIALIZED_ and self.FCCABLE_CONTROLLER_INITIALIZED_):
            raise RuntimeError("Controller not initialized. Call setParamsDecLoadPoseController() and setParamsFCCableController() first.")

        # Input validation
        if not (isinstance(load_state, BodyState) and isinstance(drone_state, BodyState)
                and isinstance(des_load_state, BodyState) and isinstance(des_cable_state, FCCableState)):
            raise TypeError("load_state, drone_state, des_load_state must be BodyState, des_cable_state must be FCCableState.")
        if not (isinstance(uidynapar, np.ndarray) and uidynapar.shape == (3,)):
            raise ValueError("uidynapar must be a numpy array of shape (3,)")

        # Update model with current states
        self.dfc_model_.updateModel(drone_state, load_state)

        # Run load pose controller
        self.load_controller_.doControl(
            self.dfc_model_.getDynamicModel(),
            self.dfc_model_.getCableState(),
            load_state, des_load_state,
            self.dfc_model_.getJqi(),
            self.dfc_model_.getdJqi(),
            self.dfc_model_.getdJalphai()
        )

        uiparpa = self.load_controller_.getUiparpa()  # Parallel input from load controller
        y = self.load_controller_.getY()              # Output from load controller

        # Run cable controller
        fperp = self.cable_controller_.doControl(
            load_state, self.dfc_model_.getDroneAttachingState(),
            self.dfc_model_.getCableState(), des_cable_state,
            self.dfc_model_.getJqi(), y, self.dfc_model_.getJalphai()
        )

        # Optionally: store or return control outputs as needed
        # self.fdes_ = fperp.copy()  # Example: store the computed force

        self.uipa_ = uiparpa + fperp
        self.fdes_ = self.uipa_ + uidynapar - self.kdamp_ * drone_state.v

    # // ==========================================

    # Compute the dynamic model
    def computeDroneAttachingState(self, drone_state: BodyState) -> None:
        self.dfc_model_.computeDroneAttachingState(drone_state)
    def computeFlyCraneModel(self, load_state: BodyState) -> None:
        self.dfc_model_.computeFlyCraneModel(load_state)
    def computeDynamicModel(self, load_state: BodyState) -> None:
        self.dfc_model_.computeDynamicModel(load_state)

    def doControlLoadPoseController(self, dyn_model: DynamicModel,
                                cable_state: FCCableState,
                                load_state: BodyState,
                                des_load_state: BodyState,
                                Jqi: npt.NDArray[np.float64],
                                dJqi: npt.NDArray[np.float64],
                                dJalphai: npt.NDArray[np.float64]) -> None:
        self.load_controller_.doControl(dyn_model, cable_state, load_state,
                                        des_load_state, Jqi, dJqi, dJalphai)
    def doControlCableController(self, load_state: BodyState,
                                drone_state: BodyState,
                                cable_state: FCCableState,
                                des_cable_state: FCCableState,
                                Jqi: npt.NDArray[np.float64],
                                ddql: npt.NDArray[np.float64],
                                Jalphai: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.cable_controller_.doControl(load_state, drone_state,
                                                cable_state, des_cable_state,
                                                Jqi, ddql, Jalphai)

    # ------------------- Getters -------------------

    def getCableState(self) -> FCCableState:
        return self.dfc_model_.getCableState()

    def getFCParams(self):
        return self.dfc_model_.getFCParams()

    def getDroneAttachingState(self) -> BodyState:
        return self.dfc_model_.getDroneAttachingState()

    def getDynamicModel(self) -> DynamicModel:
        return self.dfc_model_.getDynamicModel()

    def getJqi(self) -> np.ndarray:
        return self.dfc_model_.getJqi()

    def getdJqi(self) -> np.ndarray:
        return self.dfc_model_.getdJqi()

    def getJalphai(self) -> np.ndarray:
        return self.dfc_model_.getJalphai()

    def getdJalphai(self) -> np.ndarray:
        return self.dfc_model_.getdJalphai()

    def getFdes(self) -> np.ndarray:
        return self.fdes_

    def getUiparpa(self) -> np.ndarray:
        return self.load_controller_.getUiparpa()

    def getUipa(self) -> np.ndarray:
        return self.uipa_

    def getEi(self) -> np.ndarray:
        return self.load_controller_.getEi()

    def getWld(self) -> np.ndarray:
        return self.load_controller_.getWld()

    def getY(self) -> np.ndarray:
        return self.load_controller_.getY()

    def getFperp(self) -> np.ndarray:
        return self.cable_controller_.getFperp()
