
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from fc.flycrane_utils.flycrane import FlyCrane, BodyState
from fc.flycrane_utils.flycrane_utils import FCCableState
from fc.flycrane_utils.dec_flycrane_model import DFlyCraneModel
from fc.dec_controller.config_controller import ConfigController
from fc.dec_controller.dynamic_allocator import DynamicAllocator
from fc.dec_controller.wrench_observer import WrenchObserver
from fc.traj_generation.traj_generation import TrajectoryGenerator, OrientationTrajectoryGenerator
from fc.plot_utils.animate_flycrane import animate_flycrane

from scipy.spatial.transform import Rotation as R

# =================== Constants ===================
# Simulation parameters
DT_DEFAULT = 0.001   # Default time step (s)
T_DEFAULT = 5.0      # Default simulation time (s)
GRAVITY = 9.81      # Gravity (m/s^2)
N_DRONES = 3         # Number of drones
USE_OFFSET = True  # Use drone offset in simulation
# Drone parameters
THRUST_MAX = 20.06   # Maximum thrust (N)
# Dynamic allocator parameters
KDELTA = 100.0      # Control gain for delta
KGRAD = 1.0         # Control gain for gradient
KLAMBDAP = 0.1      # Control gain for lambda position
KLAMBDAR = 0.1      # Control gain for lambda rotation
RHOP = 20.0         # Density of payload (kg/m^3)
RHOR = 20.0         # Density of rope (kg/m^3)
DUMAX = 2.0         # Maximum drone velocity (m/s)
KREG = 0.1         # Control gain for position regulation
TMIN = 0.05        # Minimum thrust (N)
ALPHAF = 100.0    # Control gain for alpha feedforward
ALPHAFPOW = 3     # Power for alpha feedforward
ALPHAT = 5.0      # Control gain for alpha thrust
ALPHATPOW = 1     # Power for alpha thrust
# Wrench observer parameters
KOBS_F = 20.0     # Control gain for observer force
KOBS_TAU = 20.0   # Control gain for observer torque
# Dec Load Pose Controller parameters
KP_P = 2.5        # Control gain for position
KD_P = 1.0        # Control gain for position derivative
KPA = 2.0        # Control gain for acceleration
KDA = 1.0        # Control gain for acceleration derivative
KPYAW = 2.0     # Control gain for yaw position
KDYAW = 1.0     # Control gain for yaw derivative
KP_R = np.diag([KPA, KPA, KPYAW])
KD_R = np.diag([KDA, KDA, KDYAW])
KI_P = 0.0       # Integral gain for position
EP_MAX = 0.3     # Maximum position error (m)
ER_MAX = 0.3     # Maximum rotation error (rad)
EV_MAX = 0.3     # Maximum velocity error (m/s)
EOXY_MAX = 0.3   # Maximum XY position error (m)
EOZ_MAX = 0.3    # Maximum Z position error (m)
EI_MAX = 2.5     # Maximum integral error
KDAMP = 0.1      # Damping gain
# FCCableController parameters
KDAMP_ALPHA = 0.1 # Damping gain for alpha
KPALPHA = 3.0    # Control gain for alpha position
KDALPHA = 0.02   # Control gain for alpha position derivative



def create_initial_conditions(dt: float = DT_DEFAULT) -> Tuple[FlyCrane, BodyState, List[FCCableState], List[DFlyCraneModel]]:
    """
    Set up initial FlyCrane conditions and return initialized objects.
    """
    Lrho1 = [np.array([0.0866, 0.45, 0.0]), np.array([-0.433, -0.15, 0.0]), np.array([0.3464, -0.3, 0.0])]
    Lrho2 = [np.array([-0.433, 0.15, 0.0]), np.array([0.0866, -0.45, 0.0]), np.array([0.3464, 0.3, 0.0])]
    l = [1.0, 1.0, 1.0]
    if USE_OFFSET:
        doffset = [np.array([0.0, 0.0, -0.1]), np.array([0.0, 0.0, -0.1]), np.array([0.0, 0.0, -0.1])]
    else:
        doffset = [np.zeros(3), np.zeros(3), np.zeros(3)]
    ml = 0.38
    Jl = np.array([[0.0154, 0.0, 0.0], [0.0, 0.0154, 0.0], [0.0, 0.0, 0.0306]])
    mR = [1.145, 1.145, 1.145]
    pl0 = np.array([0.0, 0.0, 2.0])
    quatl0 = np.array([0.0, 0.0, 0.0, 1.0])
    alpha0 = np.array([0.7854, 0.7854, 0.7854])

    des_load_state = BodyState()
    des_load_state.p = np.array([0.0, 0.0, 2.0], dtype=np.float64).reshape((3,))
    des_load_state.quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64).reshape((4,))
    des_load_state.R = np.eye(3, dtype=np.float64)
    des_load_state.v = np.zeros(3, dtype=np.float64)
    des_load_state.world_omega = np.zeros(3, dtype=np.float64)

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

def define_desired_trajectory(T: float, p0: np.ndarray) -> Tuple[TrajectoryGenerator, OrientationTrajectoryGenerator]:
    """
    Define the desired trajectory for the load using B-splines.
    """
    waypoints = np.array([
        [p0[0], p0[1], p0[2]],
        [p0[0] + 0.5, p0[1] + 0.5, p0[2] + 0.5],
        [p0[0] + 1.0, p0[1] + 1.0, p0[2] + 1.0]
    ])
    times = np.array([0.0, T / 2, T])
    p = 5
    n_der_conds = 2
    der_conds = np.zeros((2 * n_der_conds, waypoints.shape[1]))
    traj_gen = TrajectoryGenerator.from_waypoints(
        waypoints=waypoints,
        times=times,
        n_der_conds=n_der_conds,
        p=p,
        der_conds=der_conds
    )

    quat_waypoints = np.array([
        R.from_euler('z', 0, degrees=True).as_quat(),
        R.from_euler('z', 15, degrees=True).as_quat(),
        R.from_euler('z', 50, degrees=True).as_quat(),
    ])
    quat_times = np.array([0.0, T / 2, T])

    quat_traj_gen = OrientationTrajectoryGenerator(
        quat_waypoints,
        quat_times,
    )
    return traj_gen, quat_traj_gen

def run_simulation(T: float = T_DEFAULT, dt: float = DT_DEFAULT) -> Tuple[FlyCrane, np.ndarray, np.ndarray, np.ndarray,np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Run the FlyCrane decentralized dynamic allocation simulation.
    Returns:
        traj_p: np.ndarray - Actual load trajectory
        traj_pd: np.ndarray - Desired load trajectory
        time_log: np.ndarray - Time vector
    """
    flycrane, des_load_state, des_cable_state, dflycrane_list = create_initial_conditions(dt=dt)

    load_state = flycrane.getLoadState()
    drone_state = flycrane.getDronesStates()
    drone_attaching_state = [BodyState() for _ in range(flycrane.N)]

    traj_gen, quat_traj_gen = define_desired_trajectory(T, load_state.p)

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
    traj_quat = np.zeros((n_steps, 4), dtype=np.float64)  # Load orientation (quaternion)
    traj_quatd = np.zeros((n_steps, 4), dtype=np.float64)  # Desired orientation (quaternion)
    traj_omega = np.zeros((n_steps, 3), dtype=np.float64)  # Load angular velocity
    traj_omega_d = np.zeros((n_steps, 3), dtype=np.float64)  # Desired angular velocity
    traj_alpha = [np.zeros((n_steps), dtype=np.float64) for _ in range(flycrane.N)]  # Cable angles
    traj_alpha_d = [np.zeros((n_steps), dtype=np.float64) for _ in range(flycrane.N)]  # Desired cable angles
    traj_fdes = [np.zeros((n_steps, 3), dtype=np.float64) for _ in range(flycrane.N)]  # Commanded forces
    traj_fperp = [np.zeros((n_steps, 3), dtype=np.float64) for _ in range(flycrane.N)]  # Perpendicular forces
    traj_pD = [np.zeros((n_steps, 3), dtype=np.float64) for _ in range(flycrane.N)]  # Drone positions
    traj_quatD = [np.zeros((n_steps, 4), dtype=np.float64) for _ in range(flycrane.N)]  # Drone orientations
    time_log = np.zeros(n_steps, dtype=np.float64)

    uidynapar = [np.zeros(3) for _ in range(flycrane.N)]
    f = [np.zeros(3) for _ in range(flycrane.N)]
    wlhat = [np.zeros(6) for _ in range(flycrane.N)]


    for kk in range(n_steps):
        t = kk * dt

        # Update desired trajectory
        des_load_state.p, des_load_state.v, des_load_state.a, _ = traj_gen.evaluate(t)
        quatld, omegald_body, domegald_body = quat_traj_gen.evaluate(t) 

        des_load_state.set_orientation_from_quat(quatld)
        des_load_state.set_angular_velocity_from_body(omegald_body)
        des_load_state.set_angular_acceleration_from_body(domegald_body)
        

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

            # Store trajectory
            traj_alpha[i][kk] = cable_state.alpha
            traj_alpha_d[i][kk] = des_cable_state[i].alpha
            traj_fdes[i][kk, :] = f[i]
            traj_pD[i][kk, :] = drone_state[i].p
            traj_quatD[i][kk, :] = drone_state[i].quat
            traj_fperp[i][kk, :] = config_controller[i].getFperp()


        # Simulate dynamics for one time step
        flycrane.simulateDynamics(f)

        load_state = flycrane.getLoadState()
        # Later flycrane should be changed to add the offset and here getDroneStates
        drone_state = flycrane.getDronesStates()

        # Store trajectory
        traj_p[kk, :] = load_state.p
        traj_pd[kk, :] = des_load_state.p
        traj_quat[kk, :] = load_state.quat
        traj_quatd[kk, :] = des_load_state.quat
        traj_omega[kk, :] = load_state.world_omega
        traj_omega_d[kk, :] = des_load_state.world_omega

            
        time_log[kk] = t

    return flycrane, traj_p, traj_pd, time_log, traj_quat, traj_quatd, traj_omega, traj_omega_d, traj_alpha, traj_alpha_d, traj_fdes, traj_fperp, traj_pD, traj_quatD


def plot_results(flycrane, time_log, traj_p, traj_pd, traj_quat, traj_quatd,
                 traj_omega, traj_omega_d, traj_alpha, traj_alpha_d):

    # Convert quaternions to Euler angles (batch conversion is fast)
    eulers = R.from_quat(traj_quat).as_euler('xyz', degrees=True)
    eulers_d = R.from_quat(traj_quatd).as_euler('xyz', degrees=True)

    # --- Orientation (Euler angles) ---
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(time_log, eulers[:, 0], label=r'$\phi$')
    ax1.plot(time_log, eulers[:, 1], label=r'$\theta$')
    ax1.plot(time_log, eulers[:, 2], label=r'$\psi$')
    ax1.plot(time_log, eulers_d[:, 0], '--', label=r'$\phi_d$')
    ax1.plot(time_log, eulers_d[:, 1], '--', label=r'$\theta_d$')
    ax1.plot(time_log, eulers_d[:, 2], '--', label=r'$\psi_d$')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Euler angles [deg]')
    ax1.set_title('Load Orientation vs Desired')
    ax1.grid(True)
    ax1.legend()
    fig1.tight_layout()

    # --- Angular Velocity ---
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(time_log, traj_omega[:, 0], label=r'$\omega_x$')
    ax2.plot(time_log, traj_omega[:, 1], label=r'$\omega_y$')
    ax2.plot(time_log, traj_omega[:, 2], label=r'$\omega_z$')
    ax2.plot(time_log, traj_omega_d[:, 0], label=r'$\omega_x$')
    ax2.plot(time_log, traj_omega_d[:, 1], label=r'$\omega_y$')
    ax2.plot(time_log, traj_omega_d[:, 2], label=r'$\omega_z$')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Angular velocity [rad/s]')
    ax2.set_title('Desired Angular Velocity')
    ax2.grid(True)
    ax2.legend()
    fig2.tight_layout()

    # --- Cable Angles ---
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    for i in range(flycrane.N):
        ax3.plot(time_log, traj_alpha[i], label=f'Cable {i+1}')
        ax3.plot(time_log, traj_alpha_d[i], '--', label=f'Cable {i+1} desired')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel(r'$\alpha$ [rad]')
    ax3.set_title('Cable Angles vs Desired')
    ax3.grid(True)
    ax3.legend(ncol=2)
    fig3.tight_layout()

    # --- Load Position ---
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.plot(time_log, traj_p[:, 0], label="Load X")
    ax4.plot(time_log, traj_p[:, 1], label="Load Y")
    ax4.plot(time_log, traj_p[:, 2], label="Load Z")
    ax4.plot(time_log, traj_pd[:, 0], "--", label="Desired X")
    ax4.plot(time_log, traj_pd[:, 1], "--", label="Desired Y")
    ax4.plot(time_log, traj_pd[:, 2], "--", label="Desired Z")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Position [m]")
    ax4.set_title("Load Position vs Desired")
    ax4.grid(True)
    ax4.legend()
    fig4.tight_layout()

    # --- 3D Trajectory ---
    fig5 = plt.figure(figsize=(7, 6))
    ax5 = fig5.add_subplot(111, projection="3d")
    ax5.plot(traj_p[:, 0], traj_p[:, 1], traj_p[:, 2], label="Load")
    ax5.plot(traj_pd[:, 0], traj_pd[:, 1], traj_pd[:, 2], "--", label="Desired")
    ax5.set_xlabel("X [m]")
    ax5.set_ylabel("Y [m]")
    ax5.set_zlabel("Z [m]")
    ax5.set_title("3D Trajectory of Load")
    ax5.legend()
    fig5.tight_layout()

    plt.show()




def main():
    plt.rcParams['text.usetex'] = True

    dt = DT_DEFAULT
    Tsim = T_DEFAULT

    flycrane, traj_p, traj_pd, time_log, traj_quat, traj_quatd, traj_omega, traj_omega_d, traj_alpha, traj_alpha_d, traj_fdes, traj_fperp, traj_pD, traj_quatD = run_simulation(T=Tsim, dt=dt)

    plot_results(flycrane, time_log, traj_p, traj_pd, traj_quat, traj_quatd,
                 traj_omega, traj_omega_d, traj_alpha, traj_alpha_d)


    fc_params = flycrane.getFCParameters()
    arm_length = 0.3
    rho = []
    doffset = []
    l = []
    for i in range(flycrane.N):
        rho.append(fc_params[i].Lrho1)
        rho.append(fc_params[i].Lrho2)

        doffset.append(fc_params[i].doffset)
        l.append(fc_params[i].l)


    params = {
        "l_arm": arm_length,
        "rho": rho,  
        "doffset": doffset,  
        "l": l,  
        "alpha_des": traj_alpha_d[0][0]  
    }

    animate_flycrane(
        time_log,
        traj_p, traj_quat, traj_pd, traj_quatd,
        traj_alpha, traj_fdes, traj_fperp, traj_pD, traj_quatD,
        params
    )


if __name__ == "__main__":
    main()
