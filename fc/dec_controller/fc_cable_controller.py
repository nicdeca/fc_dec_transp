
import numpy as np
from fc.common_utils.math_utils import skew
from fc.flycrane_utils.dyn_model_structs import BodyState
from fc.flycrane_utils.flycrane_utils import FCCableState

GRAVITY: float = 9.81  # m/s^2



class FCCableController:
    """
    Cable controller for decentralized multi-drone systems.
    Computes the force in the plane orthogonal to the cable directions.

    Attributes
    ----------
    gv_ : np.ndarray
        Gravity vector (3,).
    mi_ : float
        Cable mass or inertia parameter.
    kd_ : float
        Derivative gain.
    kp_ : float
        Proportional gain.
    kdamp_ : float
        Damping gain.
    Malphai_ : float
        Cable inertia term.
    ualphai_ : float
        Cable control input.
    vil_ : np.ndarray
        Load velocity (3,).
    Pis3par_ : np.ndarray
        Projection matrix (3x3).
    uperpql_ : np.ndarray
        Perpendicular control term (3,).
    uperpalpha_ : np.ndarray
        Perpendicular control term (3,).
    fperp_ : np.ndarray
        Perpendicular force (3,).
    """

    def __init__(self) -> None:
        """
        Initialize the FCCableController with default values and state variables.
        """
        self.gv_: np.ndarray = np.array([0.0, 0.0, GRAVITY])

        # Controller parameters
        self.mi_: float = 0.0
        self.kd_: float = 0.0
        self.kp_: float = 0.0
        self.kdamp_: float = 0.0

        # Internal variables
        self.Malphai_: float = 0.0
        self.ualphai_: float = 0.0

        self.vil_: np.ndarray = np.zeros(3)
        self.Pis3par_: np.ndarray = np.eye(3)

        # Control terms
        self.uperpql_: np.ndarray = np.zeros(3)
        self.uperpalpha_: np.ndarray = np.zeros(3)
        self.fperp_: np.ndarray = np.zeros(3)

    # ================== Interface ==================

    def setParams(self, mi: float, kp: float, kd: float, kdamp: float) -> None:
        """
        Set controller parameters.

        Parameters
        ----------
        mi : float
            Cable mass or inertia parameter.
        kp : float
            Proportional gain.
        kd : float
            Derivative gain.
        kdamp : float
            Damping gain.
        """
        if mi < 0 or kp < 0 or kd < 0 or kdamp < 0:
            raise ValueError("All controller parameters must be non-negative.")
        self.mi_ = mi
        self.kp_ = kp
        self.kd_ = kd
        self.kdamp_ = kdamp

        # Reset terms
        self.uperpalpha_ = np.zeros(3)
        self.uperpql_ = np.zeros(3)
        self.fperp_ = np.zeros(3)

    def doControl(
        self,
        load_state: BodyState,
        drone_state: BodyState,
        cable_state: FCCableState,
        des_cable_state: FCCableState,
        Jqi: np.ndarray,        # shape (3, 6)
        ddql: np.ndarray,       # shape (6,)
        Jalphai: np.ndarray,    # shape (3,)
    ) -> np.ndarray:
        """
        Compute cable-plane control force.

        Parameters
        ----------
        load_state : BodyState
            State of the load.
        drone_state : BodyState
            State of the drone.
        cable_state : FCCableState
            State of the cable.
        des_cable_state : FCCableState
            Desired state of the cable.
        Jqi : np.ndarray
            Jacobian matrix (3, 6).
        ddql : np.ndarray
            Load acceleration (6,).
        Jalphai : np.ndarray
            Cable Jacobian (3,).

        Returns
        -------
        np.ndarray
            Control force in the cable plane (3,).
        """
        # Input validation
        if Jqi.shape != (3, 6):
            raise ValueError("Jqi must have shape (3, 6)")
        if ddql.shape != (6,):
            raise ValueError("ddql must have shape (6,)")
        if Jalphai.shape != (3,):
            raise ValueError("Jalphai must have shape (3,)")

        # ...existing code...

        # Projection onto cable direction
        self.Pis3par_ = np.outer(cable_state.s3, cable_state.s3)

        # ========================================================
        # Control term for cable angle dynamics
        # ========================================================
        self.Malphai_ = self.mi_ * np.dot(Jalphai, Jalphai)

        self.ualphai_ = self.Malphai_ * (
            des_cable_state.domega
            - self.kd_ * (cable_state.omega - des_cable_state.omega)
            - self.kp_ * np.sin(cable_state.alpha - des_cable_state.alpha)
        )

        if np.linalg.norm(Jalphai) > np.finfo(float).eps:
            self.uperpalpha_ = self.ualphai_ * Jalphai / (Jalphai @ Jalphai)
        else:
            self.uperpalpha_ = np.zeros(3)

        # ========================================================
        # Dynamics cancellation term
        # ========================================================
        self.vil_ = drone_state.v - load_state.v

        self.uperpql_ = (
            self.mi_
            * self.Pis3par_
            @ (Jqi @ ddql - skew(self.vil_) @ load_state.world_omega + self.gv_)
        )

        # ========================================================
        # Final control: sum of components + damping
        # ========================================================
        self.fperp_ = (
            self.uperpql_
            + self.uperpalpha_
            - self.kdamp_ * self.Pis3par_ @ drone_state.v
        )

        return self.fperp_

    # ================== Getters ==================

    def getFperp(self) -> np.ndarray:
        """Get the current perpendicular force (copy)."""
        return self.fperp_.copy()

    def getUperpAlpha(self) -> np.ndarray:
        """Get the current perpendicular alpha control term (copy)."""
        return self.uperpalpha_.copy()

    def getUperpQl(self) -> np.ndarray:
        """Get the current perpendicular ql control term (copy)."""
        return self.uperpql_.copy()
