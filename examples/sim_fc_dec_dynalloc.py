
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from fc.flycrane_utils.flycrane import FlyCrane, BodyState
from fc.flycrane_utils.flycrane_utils import FCCableState
from fc.flycrane_utils.dec_flycrane_model import DFlyCraneModel
from fc.dec_controller.config_controller import ConfigController
from fc.dec_controller.dynamic_allocator import DynamicAllocator
from fc.dec_controller.wrench_observer import WrenchObserver

# =================== Constants ===================
GRAVITY = 9.81
DT_DEFAULT = 0.001
T_DEFAULT = 5.0
N_DRONES = 3
THRUST_MAX = 20.06
KDELTA = 100.0
KGRAD = 1.0
KLAMBDAP = 0.1
KLAMBDAR = 0.1
RHOP = 20.0
RHOR = 20.0
DUMAX = 2.0
KREG = 0.1
TMIN = 0.05
ALPHAF = 100.0
ALPHAFPOW = 3
ALPHAT = 5.0
ALPHATPOW = 1
KOBS_F = 20.0
KOBS_TAU = 20.0
KP_P = 2.5
KD_P = 1.0
KPA = 2.0
KDA = 1.0
KPYAW = 2.0
KDYAW = 1.0
KP_R = np.diag([KPA, KPA, KPYAW])
KD_R = np.diag([KDA, KDA, KDYAW])
KI_P = 0.0
EP_MAX = 0.3
ER_MAX = 0.3
EV_MAX = 0.3
EOXY_MAX = 0.3
EOZ_MAX = 0.3
EI_MAX = 2.5
KDAMP = 0.1
KDAMP_ALPHA = 0.1
KPALPHA = 3.0
KDALPHA = 0.02



def create_initial_conditions(dt: float = DT_DEFAULT) -> Tuple[FlyCrane, BodyState, List[FCCableState], List[DFlyCraneModel]]:
    """
    Set up initial FlyCrane conditions and return initialized objects.
    """
    Lrho1 = [np.array([0.0866, 0.45, 0.0]), np.array([-0.433, -0.15, 0.0]), np.array([0.3464, -0.3, 0.0])]
    Lrho2 = [np.array([-0.433, 0.15, 0.0]), np.array([0.0866, -0.45, 0.0]), np.array([0.3464, 0.3, 0.0])]
    l = [1.0, 1.0, 1.0]
    doffset = [np.zeros(3), np.zeros(3), np.zeros(3)]
    ml = 0.38
    Jl = np.array([[0.0154, 0.0, 0.0], [0.0, 0.0154, 0.0], [0.0, 0.0, 0.0306]])
    mR = [1.145, 1.145, 1.145]
    pl0 = np.array([0.0, 0.0, 2.0])
    quatl0 = np.array([0.0, 0.0, 0.0, 1.0])
    alpha0 = np.array([0.7854, 0.7854, 0.7854])

    des_load_state = BodyState()
    des_load_state.p = np.array([0.0, 0.0, 2.0])
    des_load_state.quat = np.array([0.0, 0.0, 0.0, 1.0])
    des_load_state.R = np.eye(3)
    des_load_state.v = np.zeros(3)
    des_load_state.world_omega = np.zeros(3)

    des_cable_state = []
    for a, o, d in zip(alpha0, [0.0]*N_DRONES, [0.0]*N_DRONES):
        s = FCCableState()
        s.alpha, s.omega, s.domega = a, o, d
        des_cable_state.append(s)

    flycrane = FlyCrane(dt=dt, use_input_forces=True)
    flycrane.setDynamicParameters(ml, Jl, mR)
    flycrane.setFCParameters(N_DRONES, Lrho1, Lrho2, l, doffset)
    flycrane.setState(pl0, quatl0, alpha0)

    dflycrane_list = [DFlyCraneModel() for _ in range(N_DRONES)]
    for dflycrane, Lrho1i, Lrho2i, li, doffseti in zip(dflycrane_list, Lrho1, Lrho2, l, doffset):
        dflycrane.setFCParams(Lrho1i, Lrho2i, li, doffseti)
        dflycrane.setDynParams(ml, Jl)

    return flycrane, des_load_state, des_cable_state, dflycrane_list

def init_config_controller(config_controller: ConfigController, flycrane: FlyCrane, droneID: int) -> None:
    """
    Initialize the ConfigController for a given drone.
    """
    dt = flycrane.dt
    fc_params = flycrane.getFCParameters()
    ml, Jl, mR_list = flycrane.getDynamicParameters()
    config_controller.initialize(KDAMP)
    config_controller.setFCParams(fc_params[droneID].Lrho1, fc_params[droneID].Lrho2, fc_params[droneID].l, fc_params[droneID].doffset)
    config_controller.setDynParams(ml, Jl)
    config_controller.setParamsDecLoadPoseController(
        mR_list[droneID],
        KP_P,
        KP_R,
        KD_P,
        KD_R,
        KI_P,
        dt,
        EP_MAX,
        ER_MAX,
        EV_MAX,
        EOXY_MAX,
        EOZ_MAX,
        EI_MAX,
    )
    config_controller.setParamsFCCableController(mR_list[droneID], KDAMP_ALPHA, KPALPHA, KDALPHA)

def init_dynamic_allocator(dynamic_allocator: DynamicAllocator, flycrane: FlyCrane, droneID: int) -> None:
    """
    Initialize the DynamicAllocator for a given drone.
    """
    dt = flycrane.dt
    fc_params = flycrane.getFCParameters()
    ml, _, _ = flycrane.getDynamicParameters()
    betai = fc_params[droneID].beta
    umin = np.array([-THRUST_MAX / 3, -THRUST_MAX / 3, 0.1 * ml * GRAVITY])
    umax = np.array([THRUST_MAX / 3, THRUST_MAX / 3, THRUST_MAX])
    dynamic_allocator.setParams(
        betai, KDELTA, KGRAD, KREG, KLAMBDAP, KLAMBDAR, RHOP, RHOR, -DUMAX,
        DUMAX, umin, umax, TMIN, ALPHAF, ALPHAFPOW, ALPHAT, ALPHATPOW, ml, dt)
    dynamic_allocator.initQP()


def init_wrench_observer(wrench_observer: WrenchObserver, flycrane: FlyCrane, droneID: int) -> None:
    """
    Initialize the WrenchObserver for a given drone.
    """
    ml, _, _ = flycrane.getDynamicParameters()
    wrench_observer.set_params(KOBS_F, KOBS_TAU, flycrane.dt, ml)


def run_simulation(T: float = T_DEFAULT, dt: float = DT_DEFAULT) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the FlyCrane decentralized dynamic allocation simulation.
    Returns:
        traj_p: np.ndarray - Actual load trajectory
        traj_pd: np.ndarray - Desired load trajectory
        time_log: np.ndarray - Time vector
    """
    flycrane, des_load_state, des_cable_state, dflycrane_list = create_initial_conditions(dt=dt)

    # Controllers
    config_controller = [ConfigController() for _ in range(flycrane.N)]
    wrench_observer = [WrenchObserver() for _ in range(flycrane.N)]
    dynamic_allocator = [DynamicAllocator() for _ in range(flycrane.N)]
    for i in range(flycrane.N):
        init_config_controller(config_controller[i], flycrane, i)
        init_dynamic_allocator(dynamic_allocator[i], flycrane, i)
        init_wrench_observer(wrench_observer[i], flycrane, i)

    # Time loop
    n_steps = int(T / dt)
    traj_p = np.zeros((n_steps, 3), dtype=np.float64)   # Load position
    traj_pd = np.zeros((n_steps, 3), dtype=np.float64)  # Desired position
    time_log = np.zeros(n_steps, dtype=np.float64)

    uidynapar = [np.zeros(3) for _ in range(flycrane.N)]
    f = [np.zeros(3) for _ in range(flycrane.N)]
    wlhat = [np.zeros(6) for _ in range(flycrane.N)]

    load_state = flycrane.getLoadState()
    # Here, it should be getDroneStates, but for the moment it is getDroneAttachingStates
    drone_state = flycrane.getDroneAttachingStates()
    drone_attaching_state = [BodyState() for _ in range(flycrane.N)]

    for kk in range(n_steps):
        t = kk * dt

        for i in range(flycrane.N):
            dflycrane_list[i].updateModel(drone_state[i], load_state)
            drone_attaching_state[i] = dflycrane_list[i].getDroneAttachingState()
            cable_state = dflycrane_list[i].getCableState()

            # Compute desired wrench from controller
            config_controller[i].doControl(
                load_state,
                drone_attaching_state[i],
                des_load_state,
                des_cable_state[i],
                uidynapar[i],
            )

            # Get the controller outputs
            wld = config_controller[i].getWld()
            ui_pa = config_controller[i].getUipa()
            f[i] = config_controller[i].getFdes()

            # Update dynamic allocator for the next iteration
            dynamic_allocator[i].doAllocation(wld, wlhat[i],
                                        dflycrane_list[i].getJqi(),
                                        ui_pa, cable_state)
            
            uidynapar[i] = dynamic_allocator[i].getuipar()
        

            # Update the wrench observer for the next iteration
            wrench_observer[i].update(load_state, dflycrane_list[i].getDynamicModel()) 

            wlhat[i] = wrench_observer[i].get_wlhat()


        # Simulate dynamics for one time step
        flycrane.simulateDynamics(f)

        load_state = flycrane.getLoadState()
        # Later flycrane should be changed to add the offset and here getDroneStates
        drone_state = flycrane.getDroneAttachingStates()

        # Store trajectory
        traj_p[kk, :] = load_state.p
        traj_pd[kk, :] = des_load_state.p
        time_log[kk] = t

    return traj_p, traj_pd, time_log


def main():
    dt = 0.001
    Tsim = 5.0
    traj_p, traj_pd, time_log = run_simulation(T=Tsim, dt=dt)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(traj_p[:, 0], traj_p[:, 1], traj_p[:, 2], label="Load trajectory")
    ax.plot(traj_pd[:, 0], traj_pd[:, 1], traj_pd[:, 2], "--", label="Desired")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("FlyCrane Load Trajectory vs Desired")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
