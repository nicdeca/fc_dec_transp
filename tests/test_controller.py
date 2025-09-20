import numpy as np
import pytest

from scipy.spatial.transform import Rotation as R

from fc.flycrane_utils.flycrane import (
    FlyCrane,
    FCParams,
    BodyState,
    DynamicModel,
)
from fc.flycrane_utils.flycrane_utils import (
    FCCableState,
    computeDirectGeometry,
    computeJqi,
    computeJalphai,
    computeDirectKinematics,
)
from fc.dec_controller.config_controller import ConfigController
from fc.dec_controller.dynamic_allocator import DynamicAllocator
from fc.dec_controller.wrench_observer import WrenchObserver


def test_flycrane_class_compute_from_instance():
    # ----------------------------------------------------------------------
    # Parameters setup
    # ----------------------------------------------------------------------
    N = 3
    Lrho1 = [
        np.array([0.0866, 0.45, 0.0]),
        np.array([-0.433, -0.15, 0.0]),
        np.array([0.3464, -0.3, 0.0]),
    ]
    Lrho2 = [
        np.array([-0.433, 0.15, 0.0]),
        np.array([0.0866, -0.45, 0.0]),
        np.array([0.3464, 0.3, 0.0]),
    ]
    l = [1.0, 1.0, 1.0]

    ml = 0.38
    Jl = np.array(
        [[0.0154, 0.0, 0.0], [0.0, 0.0154, 0.0], [0.0, 0.0, 0.0306]]
    )
    mR = [1.145, 1.145, 1.145]

    pl0 = np.array([0.0, 0.0, 2.0])
    quatl0 = np.array([0.0, 0.0, 0.0, 1.0])  # (x, y, z, w)
    Rl0 = np.eye(3)

    alpha0 = [0.7854, 0.7854, 0.7854]
    vl0 = np.array([1.4508, -0.1261, 1.4295])
    omegal0 = np.array([2.8181, 2.8344, 1.3430])

    dql0 = np.hstack((vl0, omegal0))
    omega_alpha0 = np.array([-2.4150, 1.4345, 3.2605])


    # ----------------------------------------------------------------------
    # Initialize the controller
    # ----------------------------------------------------------------------

    dt = 0.001

    kdamp = 0.1  # Damping coefficient
    kp_p = 2.5  # Position proportional gain
    kd_p = 1.0  # Position derivative gain
    kpa = 2.0  # Angle proportional gain
    kda = 1.0  # Angle derivative gain
    kpyaw = 2.0  # Yaw proportional gain
    kdyaw = 1.0  # Yaw derivative gain
    kp_r = np.diag([kpa, kpa, kpyaw])
    kd_r = np.diag([kda, kda, kdyaw])
    ki_p = 0.0  # Integral gain for position
    ep_max = 0.3  # Maximum position error
    er_max = 0.3  # Maximum rotation error
    ev_max = 0.3  # Maximum velocity error
    eoxy_max = 0.3  # Maximum xy error
    eoz_max = 0.3  # Maximum z error
    ei_max = 2.5  # Maximum integral error  

    kdamp_alpha = 0.1  # Damping coefficient for cable controller
    kpalpha = 3.0  # Proportional gain for cable angle
    kdalpha = 0.02  # Derivative gain for cable angle
    alpha_des = 0.785  # Desired angle for the cables

    # ============================================================
    # Wrench observer parameters
    # ============================================================
    Kobsf = 20.0
    Kobstau = 20.0

#   ============================================================
#   Dynamic allocator parameters

    thrust_max = 20.06
    kdelta = 100.0
    kgrad = 1.0
    klambdap = 0.1
    klambdar = 0.1
    rhop = 20.0
    rhor = 20.0
    dumax = 2.0
    kreg = 0.1
    umin = np.array([-thrust_max / 3, -thrust_max / 3, 1.0 / 10 * ml * 9.81])
    umax = np.array([thrust_max / 3, thrust_max / 3, thrust_max])
    tmin = 0.05
    alphaf = 100.0
    alphafpow = 3
    alphat = 5.0
    alphatpow = 1   

    lambda_alloc = [
        np.array([-0.0081, -0.0294, 0.0144, 0.0033, -0.0075, 0.0137]),
        np.array([-0.0171, -0.0010, -0.0024, 0.0032, 0.0031, -0.0086]),
        np.array([-0.0003, -0.0016, 0.0063, 0.0109, 0.0111, -0.0086]),
    ]       

    # Test values for the dynamic allocator
    wld_alloc = np.array([0.0054, 0.0183, 3.7052, 0.0086, 0.0032, -0.0131])
    wlhat_alloc = np.array([-0.0043, 0.0034, 3.7636, 0.0277, -0.0135, 0.0303])
    ui_pa_alloc = [
        np.array([-0.0121, 0.0072, 1.2589]),
        np.array([0.0049, 0.0103, 1.2499]),
        np.array([-0.0030, 0.0029, 1.2347]),
    ]

#   ============================================================
#   Create instance of des_cable_state

    des_cable_state = [
        FCCableState(alpha=0.8033, omega=-0.0087, domega=0.0346),
        FCCableState(alpha=0.8465, omega=0.0069, domega=-0.0169),
        FCCableState(alpha=0.7101, omega=0.0716, domega=0.0379),
    ]

    config_controller = [ConfigController() for _ in range(N)]
    wrench_observer = [WrenchObserver() for _ in range(N)]
    dynamic_allocator = [DynamicAllocator() for _ in range(N)]

    load_state = BodyState()
    load_state.p = pl0
    load_state.quat = quatl0
    load_state.R = Rl0
    load_state.v = vl0
    load_state.world_omega = omegal0


#   ============================================================
#   Set desired load state

    des_load_state = BodyState()
    des_load_state.p = np.array([0.2769, -0.1350, 2.3035])
    des_load_state.quat = np.array([0.0979, 0.0688, -0.0287, 0.9924])       
    des_load_state.R = R.from_quat(des_load_state.quat).as_matrix()
    des_load_state.v = np.array([1.4303, -0.1385, 1.5785])
    des_load_state.world_omega = np.array([2.8475, 2.7557, 1.4318])
    des_load_state.a = np.array([-0.1207, 0.0717, 0.1630])
    des_load_state.world_domega = np.array([-0.2944, 0.1438, 0.0325])

#   // ============================================================

    flycrane = FlyCrane(use_input_forces=True, dt=dt)
    flycrane.setDynamicParameters(ml, Jl, mR)
    flycrane.setFCParameters(N, Lrho1, Lrho2, l)
    flycrane.setState(pl0, quatl0, alpha0)

    # The drone state will be properly set in the loop below
    drone_state = BodyState()
    drone_state.p = np.zeros(3)
    drone_state.quat = R.identity().as_quat()
    drone_state.v = np.zeros(3)
    drone_state.body_omega = np.zeros(3)


#   ========================================================================
#   Expected values
#   These values are computed from matlab code that uses the same parameters

  # Expected cables normal
    expected_s3 = [
        np.array([0.3536, -0.6124, 0.7071]),
        np.array([0.3536, 0.6124, 0.7071]),
        np.array([-0.7071, 0.0, 0.7071]),
    ]

    # Expected inertia
    expected_Ml = np.diag([0.38, 0.38, 0.38, 0.0154, 0.0154, 0.0306])
    # Expected coriolis
    expected_Cl = np.zeros((6, 6))
    expected_Cl[3:6, 3:6] = np.array(
        [[0.0, -0.0207, 0.0867], [0.0207, 0.0, -0.08621], [-0.0436, 0.0434, 0.0]]
    )
    # Expected gravity vector
    expected_wgV = np.array([0.0, 0.0, 3.7278, 0.0, 0.0, 0.0])
    # Expected desired wrench acting on the load and to be allocated
    expected_wld = np.array([0.2094, -0.1057, 4.1347, 0.0598, -0.0523, 0.0002])
    expected_y = np.array([0.5510, -0.2782, 1.0708, 0.1236, 0.3382, 0.0074])
    # Expected uiparpa values
    expected_uiparpa = [
        np.array([-0.3435, 1.3213, 1.3160]),
        np.array([5.5954, 3.6659, -5.9725]),
        np.array([4.2968, -9.0853, 4.2968]),
    ]

    # Expected exp_fperp for actually used des cable state
    expected_fperp = np.array(
        [
            5.2416,
            -9.0785,
            10.4830,
            0.3857,
            0.6680,
            0.7713,
            -5.7437,
            0.0,
            5.7437,
        ]
    )
    # Expected exp_uipa
    expected_uipa = [
        np.array([4.8981, -7.7572, 11.7990]),
        np.array([5.9811, 4.3339, -5.2012]),
        np.array([-1.4469, -9.0853, 10.0405]),
    ]
    # Expected wlhat and ndot
    expected_wlhat = np.array([11.0261, -0.9584, 14.5920, 0.8680, 0.8730, 0.8219])
    # Expected ndot
    expected_ndot = np.array([0.0, 0.0, -3.7278, 0.0, 0.0, 0.0])
    # Expected duipar
    expected_dui = [
        np.array([1.1451, 0.1131, -0.3504]),
        np.array([-0.3057, 1.0318, -0.6165]),
        np.array([-0.6081, -0.3128, -0.4838]),
    ]
    # Expected dlambda
    expected_dlambda = np.array([-0.0010, -0.0015, 0.0058, 0.0019, -0.0017, 0.0043])    

    # ----------------------------------------------------------------------

    for i in range(N):
        # Initialize the controller (creates the load and cable
        config_controller[i].initialize(kdamp)
        config_controller[i].setFCParams(Lrho1[i], Lrho2[i], l[i], np.zeros(3))
        config_controller[i].setDynParams(ml, Jl)
        config_controller[i].setParamsDecLoadPoseController(
            mR[i],
            kp_p,
            kp_r,
            kd_p,
            kd_r,
            ki_p,
            dt,
            ep_max,
            er_max,
            ev_max,
            eoxy_max,
            eoz_max,
            ei_max,
        )
        config_controller[i].setParamsFCCableController(mR[i], kdamp_alpha, kpalpha, kdalpha)
        fc_params = config_controller[i].getFCParams()

        # =================================================

        alphai = flycrane.cable_state[i].alpha
        Lrhoi = flycrane.fc_params[i].Lrho
        Lci = flycrane.fc_params[i].Lc
        li = flycrane.fc_params[i].l
        betai = flycrane.fc_params[i].beta  

        # ====================================================

        pi = computeDirectGeometry(pl0, Rl0, alphai, Lrhoi, Lci, li, betai)
        Jqi = computeJqi(pi, pl0)
        Jalphai = computeJalphai(pi, pl0, Rl0, Lci, Lrhoi)
        vi = computeDirectKinematics(dql0, omega_alpha0[i], Jqi, Jalphai)

        # ====================================================
        # Set the drone state properly
        drone_state.p = pi
        drone_state.v = vi
        drone_state.quat = R.from_rotvec(np.array([0.0, 0.0, 0.0]))
        drone_state.world_omega = np.array([0.0, 0.0, 0.0])

        # ====================================================  

        uidynapar = np.zeros(3)
        config_controller[i].doControl(
            load_state,
            drone_state,
            des_load_state,
            des_cable_state[i],
            uidynapar,
        )
        uipa_fullconfig = config_controller[i].getUipa()
        # ====================================================

        # Compute the drone attaching state
        config_controller[i].computeDroneAttachingState(drone_state)    
        drone_attaching_pos = config_controller[i].getDroneAttachingState().p
        drone_attaching_vel = config_controller[i].getDroneAttachingState().v

        # Check that the drone attaching position and velocity are correct
        # (considering that there is no offset).
        assert np.allclose(drone_attaching_pos, pi, atol=1e-4), f"Drone {i} attaching position mismatch: {drone_attaching_pos} vs {pi}"
        assert np.allclose(drone_attaching_vel, vi, atol=1e-4), f"Drone {i} attaching velocity mismatch: {drone_attaching_vel} vs {vi}"     
        # ----------------------------------------------------------------------


        # Compute dynamic model
        config_controller[i].computeDynamicModel(load_state)
        dynamic_model = config_controller[i].getDynamicModel()
        assert np.allclose(dynamic_model.Ml, expected_Ml, atol=1e-4), f"Drone {i} Ml mismatch: {dynamic_model.Ml} vs {expected_Ml}"
        assert np.allclose(dynamic_model.Cl, expected_Cl, atol=1e-3), f"Drone {i} Cl mismatch: {dynamic_model.Cl} vs {expected_Cl}"
        assert np.allclose(dynamic_model.wgl, expected_wgV, atol=1e-4), f"Drone {i} WgL mismatch: {dynamic_model.wgl} vs {expected_wgV}"    
        # Compute the required Jacobians and Jacobian derivatives
        config_controller[i].computeFlyCraneModel(load_state)
        cable_state = config_controller[i].getCableState()
        assert np.isclose(cable_state.alpha, alpha0[i], atol=1e-4), f"Drone {i} cable angle mismatch: {cable_state.alpha} vs {alpha0[i]}"
        assert np.isclose(cable_state.omega, omega_alpha0[i], atol=1e-4), f"Drone {i} cable angular velocity mismatch: {cable_state.omega} vs {omega_alpha0[i]}"
        s3 = config_controller[i].getCableState().s3
        assert np.allclose(s3, expected_s3[i], atol=1e-4), f"Drone {i} cable normal mismatch: {s3} vs {expected_s3[i]}"     
        config_controller[i].doControlLoadPoseController(
            config_controller[i].getDynamicModel(),
            config_controller[i].getCableState(), load_state, des_load_state,
            config_controller[i].getJqi(), config_controller[i].getdJqi(),
            config_controller[i].getdJalphai())
        y = config_controller[i].getY()  # Used in place of the acceleration by
                                        # the cable controller
        wld = config_controller[i].getWld()  # Wrench acting on the load
        fperp = config_controller[i].doControlCableController(
            load_state, config_controller[i].getDroneAttachingState(),
            config_controller[i].getCableState(), des_cable_state[i],
            config_controller[i].getJqi(), y,
            config_controller[i].getJalphai())
        uiparpa = config_controller[i].getUiparpa()  # Pre-allocated force parallel to
                                                    # the cable plane
        # Check the results against the expected values
        assert np.allclose(wld, expected_wld, atol=1e-2), f"Drone {i} Wrench acting on the load mismatch: {wld} vs {expected_wld}"
        assert np.allclose(y, expected_y, atol=1e-2), f"Drone {i} y mismatch: {y} vs {expected_y}"
        assert np.allclose(uiparpa, expected_uiparpa[i], atol=1e-2), f"Drone {i} uiparpa mismatch: {uiparpa} vs {expected_uiparpa[i]}"
        assert np.allclose(fperp, expected_fperp[3*i:3*(i+1)], atol=1e-2), f"Drone {i} fperp mismatch: {fperp} vs {expected_fperp[3*i:3*(i+1)]}"        

        uipa = fperp + uiparpa
        assert np.allclose(uipa, expected_uipa[i], atol=1e-2), f"Drone {i} uipa mismatch: {uipa} vs {expected_uipa[i]}" 
        # ============================================================

        # Wrench observer test
        wrench_observer[i].set_params(Kobsf, Kobstau, dt, ml)
        # Update the wrench observer for the next iteration
        wrench_observer[i].update(load_state, dynamic_model)    
        # Check against the expected values from matlab
        wlhat = wrench_observer[i].get_wlhat()
        ndot = wrench_observer[i].get_dn()  
        assert np.allclose(wlhat, expected_wlhat, atol=1e-2), f"Drone {i} Wrench observer wlhat mismatch: {wlhat} vs {expected_wlhat}"
        assert np.allclose(ndot, expected_ndot, atol=1e-2), f"Drone {i} Wrench observer ndot mismatch: {ndot} vs {expected_ndot}"
        # ============================================================

        # Dynamic allocator test
        dynamic_allocator[i].setParams(
            betai, kdelta, kgrad, kreg, klambdap, klambdar, rhop, rhor, -dumax,
            dumax, umin, umax, tmin, alphaf, alphafpow, alphat, alphatpow, ml, dt)
        dynamic_allocator[i].initQP()
        dynamic_allocator[i].setLambda(lambda_alloc[i])

        # Check the uipa against the expected values
        assert np.allclose(uipa, expected_uipa[i], atol=1e-2), f"Drone {i} Dynamic allocator uipa mismatch: {uipa} vs {expected_uipa[i]}"   

        # Update the drone force allocation
        dynamic_allocator[i].doAllocation(wld_alloc, wlhat_alloc,
                                        config_controller[i].getJqi(),
                                        ui_pa_alloc[i], cable_state)
        
        # Get the derivative of the dynamically allocated force
        dui_par = dynamic_allocator[i].getduipar()

        # Get dlambda
        dlambda = dynamic_allocator[i].getdlambda()  # Change in lambda
        # Check against the expected values
        assert np.allclose(dlambda, expected_dlambda, atol=1e-2), f"Drone {i} Dynamic allocator dlambda mismatch: {dlambda} vs {expected_dlambda}"  
        assert np.allclose(dui_par, expected_dui[i], atol=1e-2), f"Drone {i} Dynamic allocator dui_par mismatch: {dui_par} vs {expected_dui[i]}"        

        uipar = dynamic_allocator[i].getuipar()
        fi = uipa + uipar


    flycrane2 = FlyCrane(use_input_forces=True, dt=dt)
    flycrane2.setDynamicParameters(ml, Jl, mR)
    flycrane2.setFCParameters(N, Lrho1, Lrho2, l)
    flycrane2.setState(pl0, quatl0, alpha0)
    flycrane2.setConfigurationVelocity(dql0, omega_alpha0)  

    # Input which has been generated randomly in matlab
    f = [np.array([0.5377, 1.8339, -2.2588]),
         np.array([0.8622, 0.3188, -1.3077]),
         np.array([-0.4336, 0.3426, 3.5784])]
    
    # Expected ddql before integration
    expected_ddql = np.array([-2.4668, -2.0516, -0.6836, -12.2478, 4.0317, 1.6989])
    # Expected dql after integration
    expected_dql = np.array([1.4483, -0.1282, 1.4288, 2.8059, 2.8384, 1.3447])
    # Expected MLt, CLt*dqL, wglt, Comega_alpha
    expected_MLt = np.array(
        [[2.9562, 0.0, -0.0, 0.0, 0.8610, 0.0],
         [0.0, 2.9563, 0.0, -0.8610, 0.0, -0.0],
         [-0.0, 0.0, 2.0975, 0.0, -0.0000, 0.0],    
         [0.0, -0.8610, 0.0, 0.8999, 0.0000, 0.000],
         [0.8610, 0.000, -0.000, 0.000, 0.8999, 0.000],
         [0.000, -0.000, 0.000, 0.000, 0.000, 3.611]]
    )
    expected_CLt = np.array(
        [[0.0, 0.0, 0.0, -0.0579, 3.0034, 2.3223],
         [0.0, 0.0, 0.0, -0.5184, 2.4129, 2.0287],
         [0.0, 0.0, 0.0, -3.1607, -2.8718, -2.3549],
         [0.0, 0.0, 0.0, -0.3389, -0.3930, 0.3945],
         [0.0, 0.0, 0.0, -1.1673, 0.9168, -0.1863],
         [0.0, 0.0, 0.0, 2.0729, -2.3728, -1.7978]]
    )
    # CLt * dqL
    expected_CLt_dqL = np.array([11.4684, 8.1026, -20.2099, -1.5392, -0.9413, -3.2984])
    # wglt
    expected_wglt = np.array([-0.0001, 0.0, 20.5765, 0.0, -0.0002, 0.0])
    # Comega_alpha
    expected_Comega_alpha = np.array([-3.6305, -11.3776, 1.1625,
                                     10.4568, -0.2123,  -3.2981])
    # Expected plt after integration
    expected_plres = np.array([0.0015, -0.0001, 2.0014])
    # Expected quatl after integration
    expected_quatres = np.array([0.0014, 0.0014, 0.0007, 1.0000])
    # Expected dqL after integration
    expected_dqLres = np.array([1.4483, -0.1282, 1.4288, 2.8059, 2.8384, 1.3447])
    # Expected dquat
    expected_dquatL = np.array([0.0, 1.4090, 1.4172, 0.6715])   
    # If using directly robots input forces
    flycrane2.simulateDynamics(f)
    # Declare load state
    load_state_res = flycrane2.getLoadState()
    plres = load_state_res.p
    quatres = load_state_res.quat
    dqlres = flycrane2.getConfigurationVelocity()
    dynamic_model_res = flycrane2.getDynamicModel()
    MLt = dynamic_model_res.Mlt
    CLt = dynamic_model_res.Clt
    wglt = dynamic_model_res.wglt

    # Check the results against the expected values
    assert np.allclose(dqlres, expected_dql, atol=1e-4), f"Load ddqlres mismatch: {dqlres} vs {expected_dql}"
    assert np.allclose(dqlres, expected_dql, atol=1e-4), f"Load dql_res mismatch: {dqlres} vs {expected_dql}"
    # Check MLt, CLt*dqL, wglt, Comega_alpha
    assert np.allclose(MLt, expected_MLt, atol=1e-4), f"MLt mismatch: {MLt} vs {expected_MLt}"
    assert np.allclose(CLt @ dql0, expected_CLt_dqL, atol=1e-3), f"CLt*dqL mismatch: {CLt @ dql0} vs {expected_CLt_dqL}"
    assert np.allclose(wglt, expected_wglt, atol=1e-4), f"wglt mismatch: {wglt} vs {expected_wglt}"
    assert np.allclose(plres, expected_plres, atol=1e-4), f"Load position mismatch: {plres} vs {expected_plres}"
    assert np.allclose(quatres, expected_quatres, atol=1e-4), f"Load quaternion mismatch: {quatres} vs {expected_quatres}"
    assert np.allclose(dqlres, expected_dqLres, atol=1e-4), f"Load dqL mismatch: {dqlres} vs {expected_dqLres}"



if __name__ == "__main__":
    test_flycrane_class_compute_from_instance()  # Run the test function directly
    print("Test completed successfully.")