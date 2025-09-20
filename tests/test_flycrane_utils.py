import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation as R

from fc.flycrane_utils.flycrane import FlyCrane
from fc.flycrane_utils.flycrane_utils import (
    computeDirectGeometry,
    computepiBi,
    computeJqi,
    computeJalphai,
    computeDirectKinematics,
    computeJqiDerivative,
    computeDJalphai,
    computeCablesAngle,
    computeCablesAngleDerivative,
)


def test_flycrane_compute_from_instance():
    # Define parameters
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

    # Dynamic parameters
    ml = 0.38
    Jl = np.array([
        [0.0154, 0.0, 0.0],
        [0.0, 0.0154, 0.0],
        [0.0, 0.0, 0.0306],
    ])
    mR = [1.145, 1.145, 1.145]

    # Initial state
    pl0 = np.array([0.0, 0.0, 2.0])
    quatl0 = np.array([0.0, 0.0, 0.0, 1.0])  # [x, y, z, w] identity quaternion
    alpha0 = np.array([0.7854, 0.7854, 0.7854])

    flycrane = FlyCrane()
    flycrane.setDynamicParameters(ml, Jl, mR)
    flycrane.setFCParameters(N, Lrho1, Lrho2, l)
    flycrane.setState(pl0, quatl0, alpha0)

    # Ground truth values
    expected_positions = [
        np.array([-0.5105, 0.8842, 2.6745]),
        np.array([-0.5105, -0.8842, 2.6745]),
        np.array([1.0209, 0.0, 2.6745]),
    ]
    expected_piBi = [
        np.array([-0.3373, 0.5842, 0.6745]),
        np.array([-0.3373, -0.5842, 0.6745]),
        np.array([0.6745, 0.0, 0.6745]),
    ]
    expected_Jqi = [
        np.array([
            [1.0, 0.0, 0.0, 0.0, 0.6745, -0.8842],
            [0.0, 1.0, 0.0, -0.6745, 0.0, -0.5105],
            [0.0, 0.0, 1.0, 0.8842, 0.5105, 0.0],
        ]),
        np.array([
            [1.0, 0.0, 0.0, 0.0, 0.6745, 0.8842],
            [0.0, 1.0, 0.0, -0.6745, 0.0, -0.5105],
            [0.0, 0.0, 1.0, -0.8842, 0.5105, 0.0],
        ]),
        np.array([
            [1.0, 0.0, 0.0, 0.0, 0.6745, 0.0],
            [0.0, 1.0, 0.0, -0.6745, 0.0, 1.0209],
            [0.0, 0.0, 1.0, 0.0, -1.0209, 0.0],
        ]),
    ]
    expected_Jalphai = [
        np.array([0.3373, -0.5842, 0.6745]),
        np.array([0.3373, 0.5842, 0.6745]),
        np.array([-0.6745, 0.0, 0.6745]),
    ]
    expected_dJqi = [
        np.array([
            [0.0, 0.0, 0.0, 0.0, 2.3095, 1.1757],
            [0.0, 0.0, 0.0, -2.3095, 0.0, -0.0901],
            [0.0, 0.0, 0.0, -1.1757, 0.0901, 0.0],
        ]),
        np.array([
            [0.0, 0.0, 0.0, 0.0, -0.0772, 1.7485],
            [0.0, 0.0, 0.0, 0.0772, 0.0, 3.5832],
            [0.0, 0.0, 0.0, -1.7485, -3.5832, 0.0],
        ]),
        np.array([
            [0.0, 0.0, 0.0, 0.0, -0.6944, 0.5298],
            [0.0, 0.0, 0.0, 0.6944, 0.0, -0.2874],
            [0.0, 0.0, 0.0, -0.5298, 0.2874, 0.0],
        ]),
    ]

    expected_dJalphai = [
        np.array([1.8819, -0.0372, -0.9732]),
        np.array([1.6112, -0.6100, -0.2774]),
        np.array([-0.2874, -2.8068, -0.2874]),
    ]

    pl = flycrane.load_state.p
    Rl = flycrane.load_state.R

    vl = np.array([1.4508, -0.1261, 1.4295])
    omegal = np.array([2.8181, 2.8344, 1.3430])
    dql = np.hstack((vl, omegal))
    omega_alpha = np.array([-2.4150, 1.4345, 3.2605])

    for i in range(N):
        alphai = flycrane.cable_state[i].alpha
        Lrhoi = flycrane.fc_params[i].Lrho
        Lci = flycrane.fc_params[i].Lc
        li = flycrane.fc_params[i].l
        betai = flycrane.fc_params[i].beta

        pi = computeDirectGeometry(pl, Rl, alphai, Lrhoi, Lci, li, betai)
        assert_allclose(pi, expected_positions[i], atol=1e-4)
        print("pi:", pi)
        print("expected:", expected_positions[i])

        piBi = computepiBi(pi, pl, Rl, Lrhoi)
        assert_allclose(piBi, expected_piBi[i], atol=1e-4)
        print("piBi:", piBi)
        print("expected:", expected_piBi[i])

        Jqi = computeJqi(pi, pl)
        assert_allclose(Jqi, expected_Jqi[i], atol=1e-4)
        print("Jqi:", Jqi)
        print("expected:", expected_Jqi[i])

        Jalphai = computeJalphai(pi, pl, Rl, Lci, Lrhoi)
        assert_allclose(Jalphai, expected_Jalphai[i], atol=1e-4)
        print("Jalphai:", Jalphai)
        print("expected:", expected_Jalphai[i])

        vi = computeDirectKinematics(dql, omega_alpha[i], Jqi, Jalphai)
        dJqi = computeJqiDerivative(vi, vl)
        assert_allclose(dJqi, expected_dJqi[i], atol=1e-4)
        print("dJqi:", dJqi)
        print("expected:", expected_dJqi[i])

        dJalphai = computeDJalphai(pi, pl, Rl, vi, vl, omegal, Lci, Lrhoi)
        print("dJalphai:", dJalphai)
        print("expected:", expected_dJalphai[i])
        assert_allclose(dJalphai, expected_dJalphai[i], atol=1e-4)

        pil = pi - pl
        Lpil = Rl.T @ pil
        alphai_inv = computeCablesAngle(Lpil, Lrhoi)
        assert np.isclose(alphai_inv, alphai, atol=1e-4)

        Lvil = Rl.T @ ((vi - vl) - np.cross(omegal, pil))
        omega_alphai_inv = computeCablesAngleDerivative(Lpil, Lvil, Lrhoi, Lci)
        assert np.isclose(omega_alphai_inv, omega_alpha[i], atol=1e-4)




if __name__ == "__main__":
    test_flycrane_compute_from_instance()  # Run the test function directly
    print("All tests passed.")