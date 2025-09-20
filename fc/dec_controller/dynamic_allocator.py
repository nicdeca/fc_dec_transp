
import numpy as np
import proxsuite
import copy


GRAVITY: float = 9.81  # m/s^2


class DynamicAllocator:
    """
    DynamicAllocator handles the allocation of control inputs for a multi-robot system
    with constraints and dynamic parameters. It supports parameter setting, state update,
    and QP-based allocation using ProxSuite.
    """
    def __init__(self) -> None:
        """
        Initialize the DynamicAllocator with default values and state variables.
        """
        self.g: float = GRAVITY

        # State variables
        self.iter: int = 0
        self.nvars: int = 0

        self.uipar: np.ndarray = np.zeros(3)
        self.uipar_bias: np.ndarray = np.zeros(3)
        self.duipar: np.ndarray = np.zeros(3)
        self.duid: np.ndarray = np.zeros(3)
        self.fi: np.ndarray = np.zeros(3)
        self.grad: np.ndarray = np.zeros(3)
        self.kdelta: float = 0.0

        self.ewL: np.ndarray = np.zeros(6)
        self.lambda_: np.ndarray = np.zeros(6)

        self.ht: np.ndarray = np.zeros(2)
        self.hfmax: np.ndarray = np.zeros(3)
        self.hfmin: np.ndarray = np.zeros(3)

        self.CbetaInv: np.ndarray = np.zeros((2, 2))
        self.Si: np.ndarray = np.zeros((3, 3))
        self.Pis3perp: np.ndarray = np.zeros((3, 3))
        self.ts: np.ndarray = np.zeros(3)

        self.PARAMS_SET: bool = False


    def setParams(
        self,
        beta: float,
        kdelta: float,
        kgrad: float,
        kreg: float,
        klambdap: float,
        klambdar: float,
        rhop: float,
        rhor: float,
        dumin: float,
        dumax: float,
        umin: list[float] | np.ndarray,
        umax: list[float] | np.ndarray,
        tmin: float,
        alphaf: float,
        alphafpow: float,
        alphat: float,
        alphatpow: float,
        ml: float,
        dt: float
    ) -> None:
        """
        Set the parameters for the allocator.
        Performs input validation and deep copies arrays to avoid side effects.
        """
        self.kdelta = kdelta
        self.kgrad = kgrad
        self.kreg = kreg

        self.klambda = np.zeros((6, 6))
        self.klambda[0:3, 0:3] = klambdap * np.eye(3)
        self.klambda[3:6, 3:6] = klambdar * np.eye(3)

        self.rho = np.zeros((6, 6))
        self.rho[0:3, 0:3] = rhop * np.eye(3)
        self.rho[3:6, 3:6] = rhor * np.eye(3)

        self.dumin = float(dumin) * np.ones(3)
        self.dumax = float(dumax) * np.ones(3)

        # Input validation and deep copy
        umin_arr = np.array(umin, dtype=float).copy()
        umax_arr = np.array(umax, dtype=float).copy()
        if umin_arr.shape != (3,) or umax_arr.shape != (3,):
            raise ValueError("umin and umax must be arrays of shape (3,)")
        self.umin = umin_arr
        self.umax = umax_arr

        self.tmin = tmin
        self.alphaf = alphaf
        self.alphafpow = alphafpow
        self.alphat = alphat
        self.alphatpow = alphatpow

        self.dt = dt
        self.ml = ml

        # CbetaInv matrix
        pi = np.pi
        cb = np.cos(pi - 2.0 * beta)
        self.CbetaInv = np.array([[1.0, -cb], [-cb, 1.0]])
        self.CbetaInv /= (1.0 - cb * cb)

        # Init lambda and force bias
        self.lambda_ = np.zeros(6)
        self.uipar_bias = ml * self.g / 3.0 * np.array([0,0,1])
        self.uipar = self.uipar_bias.copy()
        self.duipar = np.zeros(3)

        self.PARAMS_SET = True

    def setLambda(self, lambda_init: np.ndarray):
        if lambda_init.shape != (6,):
            raise ValueError("lambda_init must be a 6-dimensional vector.")
        self.lambda_ = lambda_init.copy()

    def initQP(self):
        neq = 0
        nineq = 2 + 2*3
        slackSize = 3
        self.nvars = 3 + slackSize
        self.iter = 0

        self.solution_qp = np.zeros(self.nvars)

        self.lb = np.zeros(nineq)
        self.ub = np.zeros(nineq)

        self.H = np.eye(self.nvars)
        for i in range(slackSize):
            self.H[3+i, 3+i] = self.kdelta

        self.c = np.zeros(self.nvars)
        self.A = np.zeros((nineq, self.nvars))

        # constraint structure
        self.A[2:5, 0:3] = np.eye(3)
        self.A[5:8, 0:3] = np.eye(3)
        self.A[5:8, 3:6] = np.eye(3)

        # bounds
        self.lb[0:2] = -np.inf
        self.ub[0:2] = 0.0

        self.lb[2:5] = -np.inf
        self.ub[2:5] = np.inf

        self.lb[5:8] = self.dumin
        self.ub[5:8] = self.dumax

        # QP solver
        self.qp_solver = proxsuite.proxqp.dense.QP(self.nvars, neq, nineq)

    def doAllocation(self, wld, wlhat, Jqi, ui_pa, cable_state):
        if not self.PARAMS_SET:
            raise ValueError("Parameters not set. Call setParams() before doAllocation().")

        # wrench error
        self.ewL = wlhat - wld

        # projector
        self.Pis3perp = np.eye(3) - np.outer(cable_state.s3, cable_state.s3)
        self.uipar = self.Pis3perp @ self.uipar

        # gradient
        self.grad = self.uipar - self.uipar_bias

        # desired derivative
        self.duid = -self.kgrad * (self.kreg * self.grad + self.Pis3perp @ Jqi @ (self.lambda_ + self.rho @ self.ewL))
        self.c[0:3] = -self.duid

        # cable tension CBF
        self.Si = np.vstack([cable_state.s1, cable_state.s2])
        At = self.CbetaInv @ self.Si
        self.ts = -At @ self.uipar
        self.ht = self.ts - self.tmin * np.ones(2)
        self.ub[0:2] = self.alphat * (self.ht ** self.alphatpow)
        self.A[0:2, 0:3] = At

        # force bounds CBF
        self.fi = self.uipar + ui_pa
        self.hfmax = self.umax - self.fi
        self.hfmin = self.fi - self.umin
        self.ub[2:5] = self.alphaf * (self.hfmax ** self.alphafpow)
        self.lb[2:5] = -self.alphaf * (self.hfmin ** self.alphafpow)

        # solve QP
        if self.iter == 0:
            self.qp_solver.init(self.H, self.c, None, None, self.A, self.lb, self.ub)
        else:
            self.qp_solver.update(self.H, self.c, None, None, self.A, self.lb, self.ub)

        self.qp_solver.solve()

        if self.qp_solver.results.info.status != proxsuite.proxqp.QPSolverOutput.PROXQP_SOLVED: 
            print("QP solver failed, using previous solution")
            return

        self.solution_qp = self.qp_solver.results.x
        self.duipar = self.solution_qp[0:3]

        # integrate
        self.uipar += self.dt * self.duipar
        self.uipar = self.Pis3perp @ self.uipar

        # dual update
        self.lambda_ += self.klambda @ (self.dt * self.ewL)

        self.iter += 1

    # ================== Getters ==================
    def getuipar(self) -> np.ndarray:
        """Get the current parallel input (copy)."""
        return self.uipar.copy()

    def getuiparbias(self) -> np.ndarray:
        """Get the current parallel input bias (copy)."""
        return self.uipar_bias.copy()

    def getht(self) -> np.ndarray:
        """Get the current CBF constraint values (copy)."""
        return self.ht.copy()

    def gethfmax(self) -> np.ndarray:
        """Get the current force upper bound CBF values (copy)."""
        return self.hfmax.copy()

    def gethfmin(self) -> np.ndarray:
        """Get the current force lower bound CBF values (copy)."""
        return self.hfmin.copy()

    def getts(self) -> np.ndarray:
        """Get the current cable tension values (copy)."""
        return self.ts.copy()

    def getduid(self) -> np.ndarray:
        """Get the current desired derivative of uipar (copy)."""
        return self.duid.copy()

    def getduipar(self) -> np.ndarray:
        """Get the current derivative of uipar (copy)."""
        return self.duipar.copy()

    def getlambda(self) -> np.ndarray:
        """Get the current lambda (copy)."""
        return self.lambda_.copy()

    def getdlambda(self) -> np.ndarray:
        """Get the current lambda update (not a copy, as it's a computed value)."""
        return self.klambda @ self.ewL