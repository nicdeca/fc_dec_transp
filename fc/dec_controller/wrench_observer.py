
import numpy as np
from fc.flycrane_utils.dyn_model_structs import BodyState, DynamicModel
import copy

GRAVITY: float = 9.81  # m/s^2


class WrenchObserver:
    """
    Momentum-based observer for estimating the external wrench
    applied to a rigid body.

    Attributes
    ----------
    dt_ : float
        Integration time step (s).
    Kobs_ : np.ndarray
        Observer gain matrix (6x6).
    intwlhat_ : np.ndarray
        Integral of estimated wrench (6,).
    n_ : np.ndarray
        Auxiliary variable (6,).
    wlhat_ : np.ndarray
        Estimated wrench (6,).
    dn_ : np.ndarray
        Derivative of n (6,).
    dql_ : np.ndarray
        Body twist (6,).
    p_ : np.ndarray
        Momentum (6,).
    """

    def __init__(self) -> None:
        """
        Initialize the WrenchObserver with default values and state variables.
        """
        self.dt_: float = 0.0
        self.Kobs_: np.ndarray = np.zeros((6, 6))

        # Internal states
        self.intwlhat_: np.ndarray = np.zeros(6)  # integral of estimated wrench
        self.n_: np.ndarray = np.zeros(6)         # auxiliary variable
        self.wlhat_: np.ndarray = np.zeros(6)     # estimated wrench
        self.dn_: np.ndarray = np.zeros(6)        # derivative of n

        # Body twist
        self.dql_: np.ndarray = np.zeros(6)

        # Momentum
        self.p_: np.ndarray = np.zeros(6)

    # ================== Interface ==================

    def set_params(self, Kobsf: float, Kobstau: float, dt: float, ml: float) -> None:
        """
        Initialize observer gains and states.

        Parameters
        ----------
        Kobsf : float
            Observer gain for force.
        Kobstau : float
            Observer gain for torque.
        dt : float
            Integration time step (s).
        ml : float
            Mass of the load.
        """
        if Kobsf <= 0 or Kobstau <= 0 or dt <= 0 or ml <= 0:
            raise ValueError("All parameters must be positive.")

        self.Kobs_ = np.zeros((6, 6))
        self.Kobs_[0:3, 0:3] = Kobsf * np.eye(3)
        self.Kobs_[3:6, 3:6] = Kobstau * np.eye(3)

        self.dt_ = dt

        self.n_ = np.zeros(6)
        self.intwlhat_ = np.zeros(6)

        # Use named constant for gravity
        self.intwlhat_[2] = -1.0 / Kobsf * GRAVITY * ml
        self.wlhat_ = np.zeros(6)
        self.wlhat_[2] = GRAVITY * ml

    def update(self, load_state: BodyState, dynamic_model: DynamicModel) -> None:
        """
        Perform one observer update step given the load state and dynamic model.

        Parameters
        ----------
        load_state : BodyState
            The current state of the load.
        dynamic_model : DynamicModel
            The current dynamic model.
        """
        # Input validation
        if not (isinstance(load_state, BodyState) and isinstance(dynamic_model, DynamicModel)):
            raise TypeError("load_state must be BodyState and dynamic_model must be DynamicModel.")

        # Body twist: [v; omega]
        self.dql_ = np.hstack((load_state.v, load_state.world_omega))

        # Momentum
        self.p_ = dynamic_model.Ml @ self.dql_

        # Observer dynamics
        self.dn_ = dynamic_model.Cl.T @ self.dql_ - dynamic_model.wgl
        self.wlhat_ = -self.Kobs_ @ (self.intwlhat_ - (self.p_ - self.n_))

        # Integrate observer states
        self.n_ += self.dn_ * self.dt_
        self.intwlhat_ += self.wlhat_ * self.dt_

    # ================== Getters ==================

    def get_wlhat(self) -> np.ndarray:
        """Get the current estimated wrench (copy)."""
        return self.wlhat_.copy()

    def get_n(self) -> np.ndarray:
        """Get the current auxiliary variable n (copy)."""
        return self.n_.copy()

    def get_dn(self) -> np.ndarray:
        """Get the current derivative of n (copy)."""
        return self.dn_.copy()

    def get_intwlhat(self) -> np.ndarray:
        """Get the current integral of estimated wrench (copy)."""
        return self.intwlhat_.copy()
