import numpy as np
from scipy.spatial.transform import Rotation as R
import copy
from fc.flycrane_utils.flycrane_utils import (
    computeDirectGeometry,
    computeJqi,
    computeJalphai,
    computeDirectKinematics,
    computeJqiDerivative,
    computeDJalphai,
    computeCableNormal,
    FCParams,
    FCCableState,
)
from fc.flycrane_utils.dyn_model_structs import BodyState, DynamicModel
from fc.common_utils.math_utils import skew
from fc.common_utils.rot_utils import quatKDE



class FlyCrane:
    """
    FlyCrane system dynamics and simulation.

    This class models the dynamics of a load suspended by multiple cables
    attached to drones. It supports initialization of parameters, simulation
    of load and cable dynamics, and direct computation of Jacobians and forces.

    Attributes
    ----------
    dt : float
        Integration time step (s).
    USE_INPUT_FORCES : bool
        Whether to take as input robot forces or total wrench on the load.
    STATE_INITIALIZED : bool
        True if the state has been initialized.
    FCPARAMETERS_INITIALIZED : bool
        True if the FlyCrane parameters have been initialized.
    DYNPARAMETERS_INITIALIZED : bool
        True if the dynamic parameters have been initialized.
    iter_count : int
        Number of simulation iterations performed.
    fc_params : list[FCParams]
        List of cable parameters for each cable.
    load_state : BodyState
        State of the load (position, velocity, orientation, etc.).
    dynamic_model : DynamicModel
        Dynamic model matrices and vectors.
    drone_attaching_state : list[BodyState]
        State of each drone's attachment point.
    cable_state : list[FCCableState]
        State of each cable (angles, angular velocities, etc.).
    u : list[np.ndarray]
        Input forces for each cable/drone.
    wlt : np.ndarray
        Total wrench on the load (6D vector).
    dql : np.ndarray
        Load twist (6D vector: linear and angular velocity).
    ddql : np.ndarray
        Load twist derivative (6D vector: linear and angular acceleration).
    dquat : scipy.spatial.transform.Rotation
        Quaternion derivative for the load orientation.
    mR : list[float]
        Mass of each robot/drone.
    ml : float
        Mass of the load.
    Jl : np.ndarray
        Inertia matrix of the load (3x3).
    N : int
        Number of cables/drones.
    """
    def __init__(self, dt: float = 0.001, use_input_forces: bool = True):
        """
        Initialize the FlyCrane system.

        Parameters
        ----------
        dt : float, optional
            Integration time step (s), by default 0.001.
        use_input_forces : bool, optional
            Whether to take as input robot forces or total wrench on the load, by default False.
        """
        self.dt = dt
        self.USE_INPUT_FORCES = use_input_forces
        self.STATE_INITIALIZED = False
        self.FCPARAMETERS_INITIALIZED = False
        self.DYNPARAMETERS_INITIALIZED = False
        self.iter_count = 0

        # ====== Constants ======
        self.GRAVITY: float = 9.81  # m/s^2
        self.EPS_CHORD: float = 1e-12
        self.CHORD_FACTOR: float = 2.0

        # Model parameters and state
        self.fc_params = []  # List of cable parameters for each cable
        self.load_state = BodyState()  # State of the load
        self.dynamic_model = DynamicModel()  # Dynamic model matrices and vectors
        self.drone_attaching_state = []  # State of each drone's attachment point
        self.cable_state = []  # State of each cable
        self.u = []  # Input forces for each cable/drone
        self.wlt = np.zeros(6)  # Total wrench on the load
        self.dql = np.zeros(6)  # Load twist (velocity)
        self.ddql = np.zeros(6)  # Load twist derivative (acceleration)
        self.dquat = R.identity()  # Quaternion derivative for the load
        self.mR = []  # Mass of each robot/drone
        self.ml = 0.0  # Mass of the load
        self.Jl = np.zeros((3, 3))  # Inertia matrix of the load
        self.N = 0  # Number of cables/drones

        # Jacobians and intermediates
        self.Jqi = []
        self.dJqi = []
        self.Jalphai = []
        self.dJalphai = []
        self.Pis3perp = np.eye(3)
        self.JiqPispar = np.zeros((6, 3))
        self.Comega_alphai = np.zeros(6)
        self.gv = np.array([0, 0, self.GRAVITY])


    # ========================= Simulation =========================

    def simulateDynamics(self, u: list[np.ndarray], wlt: np.ndarray | None = None) -> None:
        """
        Advance system dynamics by one integration step.

        Parameters
        ----------
        u : list[np.ndarray]
            Input forces from drones, one (3,) array per cable.
        wlt : np.ndarray, optional
            External wrench applied to the load (6,). Required if USE_INPUT_FORCES is False.

        Raises
        ------
        RuntimeError
            If state or parameters are not initialized.
        ValueError
            If wlt is not provided when USE_INPUT_FORCES is False.
        """
        if not self.STATE_INITIALIZED or not self.FCPARAMETERS_INITIALIZED or not self.DYNPARAMETERS_INITIALIZED:
            raise RuntimeError("State or parameters not initialized.")

        self.u = u

        # Set wrench on the load depending on input mode
        if self.USE_INPUT_FORCES:
            self.wlt = np.zeros(6)
        else:
            if wlt is None:
                raise ValueError("wlt must be provided when USE_INPUT_FORCES is False.")
            self.wlt = wlt

        # Compute dynamics and integrate
        self.computeDynamicModel()
        self.computeDirectDynamicModelLoad()
        for i in range(self.N):
            self.computeDirectDynamicModelCable(i)
        self.integrateDynamicsLoad()
        for i in range(self.N):
            self.integrateDynamicsCable(i)


        self.iter_count += 1

    # ========================= Dynamic Model =========================

    def computeDynamicModel(self) -> None:
        """
        Compute the dynamic model of the load and cables.

        Updates the mass matrix, Coriolis matrix, gravity wrench,
        and Jacobians for each cable.
        """
        # Update rotation matrix from quaternion
        self.load_state.R = R.from_quat(self.load_state.quat).as_matrix()
        wJl = self.load_state.R @ self.Jl @ self.load_state.R.T

        # Update inertia and Coriolis matrices for the load
        self.dynamic_model.Ml[3:6, 3:6] = wJl
        self.dynamic_model.Cl[3:6, 3:6] = skew(self.load_state.world_omega) @ wJl
        self.dynamic_model.Mlt = self.dynamic_model.Ml.copy()
        self.dynamic_model.Clt = self.dynamic_model.Cl.copy()
        self.dynamic_model.wglt = self.dynamic_model.wgl.copy()

        # Update load twist (velocity and angular velocity)
        self.dql[:3] = self.load_state.v
        self.dql[3:] = self.load_state.world_omega
        self.Comega_alphai[:] = 0

        for i in range(self.N):
            # Update drone attaching state position using direct geometry
            self.drone_attaching_state[i].p = computeDirectGeometry(
                self.load_state.p, self.load_state.R,
                self.cable_state[i].alpha,
                self.fc_params[i].Lrho,
                self.fc_params[i].Lc,
                self.fc_params[i].l,
                self.fc_params[i].beta
            )

            self.Jqi[i] = computeJqi(self.drone_attaching_state[i].p, self.load_state.p)
            self.Jalphai[i] = computeJalphai(
                self.drone_attaching_state[i].p,
                self.load_state.p,
                self.load_state.R,
                self.fc_params[i].Lc,
                self.fc_params[i].Lrho
            )

            self.drone_attaching_state[i].v = computeDirectKinematics(
                self.dql,
                self.cable_state[i].omega,
                self.Jqi[i],
                self.Jalphai[i]
            )

            self.dJqi[i] = computeJqiDerivative(self.drone_attaching_state[i].v, self.load_state.v)
            self.dJalphai[i] = computeDJalphai(
                self.drone_attaching_state[i].p,
                self.load_state.p,
                self.load_state.R,
                self.drone_attaching_state[i].v,
                self.load_state.v,
                self.load_state.world_omega,
                self.fc_params[i].Lc,
                self.fc_params[i].Lrho
            )

            self.cable_state[i].s1, self.cable_state[i].s2, self.cable_state[i].s3 = computeCableNormal(
                self.drone_attaching_state[i].p,
                self.load_state.p,
                self.load_state.R,
                self.fc_params[i].Lrho1,
                self.fc_params[i].Lrho2,
            )

            # Move Pis3perp and JiqPispar to local variables
            Pis3perp = np.eye(3) - np.outer(self.cable_state[i].s3, self.cable_state[i].s3)
            JiqPispar = self.Jqi[i].T @ Pis3perp

            self.dynamic_model.Mlt += self.mR[i] * JiqPispar @ self.Jqi[i]
            self.dynamic_model.Clt += self.mR[i] * JiqPispar @ self.dJqi[i]
            self.dynamic_model.wglt += self.mR[i] * JiqPispar @ self.gv
            self.Comega_alphai += self.mR[i] * JiqPispar @ self.dJalphai[i] * self.cable_state[i].omega

            if self.USE_INPUT_FORCES:
                self.wlt += JiqPispar @ self.u[i]

    def computeDirectDynamicModelLoad(self) -> None:
        """
        Compute the direct dynamic model of the load, i.e. the relationship between
        the load's acceleration and the forces acting on it.
        Handles singular matrix exception gracefully.
        """
        try:
            self.ddql = np.linalg.solve(
                self.dynamic_model.Mlt,
                -self.dynamic_model.Clt @ self.dql - self.dynamic_model.wglt - self.Comega_alphai + self.wlt
            )
        except np.linalg.LinAlgError:
            raise RuntimeError("Dynamic model mass matrix is singular.")

    def computeDirectDynamicModelCable(self, i: int) -> None:
        """
        Compute the direct dynamic model of the cable.
        """
        Malphai = self.mR[i] * np.linalg.norm(self.Jalphai[i]) ** 2
        if Malphai < self.EPS_CHORD:
            raise RuntimeError(f"Cable {i} has near-zero effective inertia.")
        self.cable_state[i].domega = (1.0 / Malphai) * (
            self.Jalphai[i] @ (self.u[i] - self.mR[i] * (
                self.Jqi[i] @ self.ddql + self.dJqi[i] @ self.dql + self.gv
            ))
        )

    def integrateDynamicsLoad(self) -> None:
        """
        Integrate the dynamics of the load.
        Ensures correct shape and type for BodyState attributes.
        """
        self.load_state.p = (self.load_state.p + self.load_state.v * self.dt).reshape((3,))
        self.load_state.v = (self.load_state.v + self.ddql[:3] * self.dt).reshape((3,))

        dquat = quatKDE(self.load_state.quat, self.load_state.body_omega)
        quat_new = self.load_state.quat + dquat * self.dt
        quat_new = quat_new / np.linalg.norm(quat_new)
        self.load_state.quat = quat_new.reshape((4,))
        self.load_state.R = R.from_quat(self.load_state.quat).as_matrix()

        self.load_state.world_omega = (self.load_state.world_omega + self.ddql[3:] * self.dt).reshape((3,))
        self.load_state.body_omega = (self.load_state.R.T @ self.load_state.world_omega).reshape((3,))

    def integrateDynamicsCable(self, i: int) -> None:
        """
        Integrate the dynamics of a single cable.
        """
        self.cable_state[i].alpha += self.cable_state[i].omega * self.dt
        self.cable_state[i].omega += self.cable_state[i].domega * self.dt

    # ========================= Initialization =========================
    def setFCParameters(self, N: int, Lrho1: list[np.ndarray], Lrho2: list[np.ndarray], l: list[float], doffset: list[np.ndarray]|None=None) -> None:
        """
        Set the parameters for the fly crane system.

        Parameters
        ----------
        N : int
            Number of robots.
        Lrho1 : list[np.ndarray]
            First set of cable attachment points.
        Lrho2 : list[np.ndarray]
            Second set of cable attachment points.
        l : list[float]
            Lengths of the cables.
        doffset : list[np.ndarray]|None
            Offset vectors in drone body frame for each cable attachment point.
        """
        if N <= 0:
            raise ValueError("Number of cables must be positive")
        if not (len(Lrho1) == len(Lrho2) == len(l) == N):
            raise ValueError("Lrho1, Lrho2, and l must have the same size as N")

        self.N = N

        if doffset is not None:
            if len(doffset) != N:
                raise ValueError("doffset must have the same size as N")
        else:
            doffset = [np.zeros(3) for _ in range(N)]

        # Clear previous parameters to avoid accumulation
        self.fc_params.clear()
        for i in range(N):
            self.fc_params.append(FCParams())
            self.fc_params[i].Lrho1 = Lrho1[i]
            self.fc_params[i].Lrho2 = Lrho2[i]
            self.fc_params[i].l = l[i]

            Lrho12 = Lrho1[i] - Lrho2[i]
            chord_length = np.linalg.norm(Lrho12)
            if chord_length >= self.CHORD_FACTOR * l[i]:
                raise ValueError("Cable length must be greater than half the chord length")
            if chord_length < self.EPS_CHORD:
                raise ValueError("Attachment points cannot be identical")

            self.fc_params[i].beta = np.arccos(chord_length / (self.CHORD_FACTOR * l[i]))
            self.fc_params[i].Lrho = (Lrho1[i] + Lrho2[i]) / 2.0
            self.fc_params[i].Lc = Lrho12 / chord_length

            self.fc_params[i].doffset = doffset[i]

        self.initializeVariables()
        self.FCPARAMETERS_INITIALIZED = True

    def setState(self, pl: np.ndarray, quatl: np.ndarray, alpha: np.ndarray) -> None:
        """
        Set the state of the fly crane system.

        Parameters
        ----------
        pl : np.ndarray
            Position of the load (3,).
        quatl : np.ndarray
            Orientation of the load (4,).
        alpha : np.ndarray
            Cable angles (N,).
        """
        if not self.FCPARAMETERS_INITIALIZED:
            raise RuntimeError("Parameters not initialized. Call setFCParameters() first.")

        if not (len(alpha) == self.N):
            raise ValueError("Alpha must have the same size as number of cables N")
        

        self.load_state.p = pl.reshape((3,))
        self.load_state.quat = quatl.reshape((4,))
        self.load_state.R = R.from_quat(quatl).as_matrix()
        self.load_state.v = np.zeros(3, dtype=float)
        self.load_state.world_omega = np.zeros(3, dtype=float)
        self.load_state.body_omega = np.zeros(3, dtype=float)

        # Clear previous states to avoid accumulation
        self.cable_state.clear()
        self.drone_attaching_state.clear()

        for i in range(self.N):
            self.cable_state.append(FCCableState())

            self.cable_state[i].alpha = alpha[i]
            self.cable_state[i].omega = 0.0
            self.cable_state[i].domega = 0.0
            self.cable_state[i].s1 = np.zeros(3)
            self.cable_state[i].s2 = np.zeros(3)
            self.cable_state[i].s3 = np.zeros(3)

            self.drone_attaching_state.append(BodyState())

            self.drone_attaching_state[i].p = computeDirectGeometry(
                self.load_state.p,
                self.load_state.R,
                self.cable_state[i].alpha,
                self.fc_params[i].Lrho,
                self.fc_params[i].Lc,
                self.fc_params[i].l,
                self.fc_params[i].beta
            )

            self.drone_attaching_state[i].v = np.zeros(3)
            self.drone_attaching_state[i].quat = np.zeros(4)
            self.drone_attaching_state[i].quat[3] = 1.0
            self.drone_attaching_state[i].R = np.eye(3)
            self.drone_attaching_state[i].body_omega = np.zeros(3)
            self.drone_attaching_state[i].world_omega = np.zeros(3)

        self.STATE_INITIALIZED = True

    def setDynamicParameters(self, ml: float, Jl: np.ndarray, mR: list[float]) -> None:
        """
        Set the dynamic parameters for the fly crane system.

        Parameters
        ----------
        ml : float
            Mass of the load.
        Jl : np.ndarray
            Inertia matrix of the load (6, 6).
        mR : list[float]
            Masses of the robots.
        """
        self.Jl = Jl.copy()
        self.mR = copy.deepcopy(mR)
        self.ml = ml
        self.dynamic_model.Ml = np.zeros((6, 6))
        self.dynamic_model.Cl = np.zeros((6, 6))
        self.dynamic_model.wgl = np.zeros(6)
        self.dynamic_model.Mlt = np.zeros((6, 6))
        self.dynamic_model.Clt = np.zeros((6, 6))
        self.dynamic_model.wglt = np.zeros(6)
        self.dynamic_model.Ml[:3, :3] = np.eye(3) * ml
        self.dynamic_model.wgl[2] = ml * self.GRAVITY
        self.Comega_alphai = np.zeros(6)
        self.DYNPARAMETERS_INITIALIZED = True


    def initializeVariables(self) -> None:
        """
        Initialize the variables for the fly crane system.
        Refactored to avoid repeated code and ensure correct shapes.
        """
        # Initialize input forces
        self.u = [np.zeros(3) for _ in range(self.N)]

        # Initialize load twist and its derivative
        self.dql = np.zeros(6)
        self.ddql = np.zeros(6)

        # Initialize total wrench
        self.wlt = np.zeros(6)

        # Initialize Jacobians
        self.Jqi = [np.zeros((3, 6)) for _ in range(self.N)]
        self.dJqi = [np.zeros((3, 6)) for _ in range(self.N)]
        self.Jalphai = [np.zeros(3) for _ in range(self.N)]
        self.dJalphai = [np.zeros(3) for _ in range(self.N)]

        # Initialize intermediate matrices
        self.Pis3perp = np.eye(3)
        self.JiqPispar = np.zeros((6, 3))
        self.Comega_alphai = np.zeros(6)



    # Set the configuration velocity (used for testing)
    def setConfigurationVelocity(self, dql: np.ndarray, omega_alpha: np.ndarray) -> None:
        """
        Set the configuration velocity for the load and cables (used for testing).
        Ensures correct shape and type for BodyState attributes.
        """
        self.load_state.v = dql[:3].reshape((3,))
        self.load_state.world_omega = dql[3:].reshape((3,))
        self.load_state.body_omega = (self.load_state.R.T @ self.load_state.world_omega).reshape((3,))
        for i in range(self.N):
            self.cable_state[i].omega = omega_alpha[i]

    # ================== Getters ==================
    def getDynamicParameters(self) -> tuple[float, np.ndarray, list[float]]:
        """Get the current dynamic parameters: load mass, inertia, and robot masses."""
        return self.ml, self.Jl.copy(), copy.deepcopy(self.mR)

    def getFCParameters(self) -> list[FCParams]:
        """Get the current FlyCrane parameters."""
        return copy.deepcopy(self.fc_params)
    
    def getLoadState(self) -> BodyState:
        """Get the current load state."""
        return copy.deepcopy(self.load_state)

    def getCableState(self) -> list[FCCableState]:
        """Get the current cable states."""
        return copy.deepcopy(self.cable_state)

    def getDroneAttachingStates(self) -> list[BodyState]:
        """Get the current drone attaching states."""
        return copy.deepcopy(self.drone_attaching_state)

    def getDynamicModel(self) -> DynamicModel:
        """Get the current dynamic model."""
        return copy.deepcopy(self.dynamic_model)

    def getConfigurationVelocity(self) -> np.ndarray:
        """Get the current configuration velocity (6D)."""
        dql = np.zeros(6)
        dql[:3] = self.load_state.v.copy()
        dql[3:] = self.load_state.world_omega.copy()
        return dql