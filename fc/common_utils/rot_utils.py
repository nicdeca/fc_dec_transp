import numpy as np

from scipy.spatial.transform import Rotation as R

def ZYXeul2quat(roll, pitch, yaw):
    """
    Convert ZYX Euler angles to a quaternion.
    
    Parameters:
        roll (float): Rotation about the X-axis (in radians).
        pitch (float): Rotation about the Y-axis (in radians).
        yaw (float): Rotation about the Z-axis (in radians).
    
    Returns:
        list: Quaternion [w, x, y, z] representing the same rotation.
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    q = np.zeros(4)
    q[0] = cr * cp * cy + sr * sp * sy  # w
    q[1] = sr * cp * cy - cr * sp * sy  # x
    q[2] = cr * sp * cy + sr * cp * sy  # y
    q[3] = cr * cp * sy - sr * sp * cy  # z

    return q  


def quat_error(q_cmd, q):
  """
  Calculates the quaternion error between two quaternions.

  Args:
    q_cmd: The desired quaternion (numpy array of size 4).
    q: The current quaternion (numpy array of size 4).

  Returns:
    q_err_vec: The error quaternion in vector form (numpy array of size 3).
  """
  q_cmd_conj = np.array([q_cmd[0], -q_cmd[1], -q_cmd[2], -q_cmd[3]]) 
  q_err = quat_mul(q_cmd_conj, q)
  q_err_vec = q_err[1:]  # Extract imaginary components

  return q_err_vec


def quat_mul(q1, q2):
  """
  Multiplies two quaternions.

  Args:
    q1: The first quaternion (numpy array of size 4).
    q2: The second quaternion (numpy array of size 4).

  Returns:
    The product of the two quaternions (numpy array of size 4).
  """
  w1, x1, y1, z1 = q1
  w2, x2, y2, z2 = q2
  w = w1*w2 - x1*x2 - y1*y2 - z1*z2
  x = w1*x2 + x1*w2 + y1*z2 - z1*y2
  y = w1*y2 - x1*z2 + y1*w2 + z1*x2
  z = w1*z2 + x1*y2 - y1*x2 + z1*w2
  return np.array([w, x, y, z])




def quatKDE(q: np.ndarray, body_omega: np.ndarray) -> np.ndarray:
    """
    Compute the quaternion kinematic differential equation.

    Parameters
    ----------
    q : np.ndarray
        Current orientation as a quaternion.
    body_omega : np.ndarray
        Angular velocity vector in the body frame (3,).

    Returns
    -------
    dq : scipy.spatial.transform.Rotation
        Quaternion derivative as a Rotation object.
    """
    # Represent angular velocity as a quaternion (0, wx, wy, wz)
    omega_quat = np.concatenate(([0.0], body_omega))  # (w=0, x, y, z)

    # Multiply quaternions: dq = q * omega_quat
    # q as [x, y, z, w] for Eigen -> SciPy uses [x, y, z, w] too
    w, x, y, z = omega_quat  # w=0, x=wx, y=wy, z=wz

    # Quaternion multiplication q * omega_quat
    qx, qy, qz, qw = q  # [x, y, z, w]
    dq = np.zeros(4)
    dq[0] = qw * x + qy * z - qz * y + qx * w
    dq[1] = qw * y + qz * x - qx * z + qy * w
    dq[2] = qw * z + qx * y - qy * x + qz * w
    dq[3] = qw * w - qx * x - qy * y - qz * z

    # Multiply by 0.5 as in Eigen version
    dq *= 0.5

    return dq  # Return as a numpy array



def vrrotvec2mat(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation formula for axis-angle."""
    if np.linalg.norm(axis) < 1e-6:
        raise ValueError("Axis must not be zero")
    ax = axis / np.linalg.norm(axis)
    x, y, z = ax
    c, s, t = np.cos(angle), np.sin(angle), 1 - np.cos(angle)

    R = np.array([
        [t*x*x + c,     t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z,   t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y,   t*y*z + s*x, t*z*z + c],
    ])
    return R



def rotz(psi: float) -> np.ndarray:
    """Rotation about z axis."""
    return np.array([
        [np.cos(psi), -np.sin(psi), 0.0],
        [np.sin(psi), np.cos(psi), 0.0],
        [0.0, 0.0, 1.0],
    ])



def quat2rotm(q: np.ndarray) -> np.ndarray:
    """Quaternion to rotation matrix. q = [w, x, y, z]."""
    if abs(np.linalg.norm(q) - 1.0) > 1e-6:
        raise ValueError("Quaternion must be normalized")

    w, x, y, z = q
    R = np.array([
        [w*w + x*x - y*y - z*z, 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),         w*w - x*x + y*y - z*z, 2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),     w*w - x*x - y*y + z*z],
    ])
    return R


def yaw_from_quat(q: np.ndarray) -> float:
    """Extract yaw angle from quaternion [w, x, y, z]."""
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y*y + z*z))