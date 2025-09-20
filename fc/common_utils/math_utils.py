import numpy as np

def skew(v: np.ndarray) -> np.ndarray:
    """
    Create a skew-symmetric matrix from a 3D vector.
    
    Parameters:
        v (array-like): A 3D vector.
    
    Returns:
        np.ndarray: A skew-symmetric matrix corresponding to the vector.
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])



def normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to have unit length.
    Parameters:
        v (np.ndarray): Input vector.
    Returns:
        np.ndarray: Normalized vector.
    """
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n




# Sign function for double. Settles the sign of zero to 1.
def sgnplus(x: float) -> int:
    return 1 if x >= 0 else -1


def sat_sgn(s: float, mu: float) -> float:
    """Approximate signum with smooth transition"""
    if abs(s) > mu:
        return 1.0 if s > 0 else -1.0
    return s / mu


def sat_sgn_vec(s: np.ndarray, mu: float) -> np.ndarray:
    """Vectorized version of satSgn"""
    return np.array([sat_sgn(si, mu) for si in s])



def vee(S: np.ndarray) -> np.ndarray:
    """vee map for skew-symmetric matrix"""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def kronecker_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Kronecker product (numpy.kron wrapper)."""
    return np.kron(A, B)



def ssat(v: np.ndarray, max_norm: float) -> np.ndarray:
    """Smooth saturation of a vector."""
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-12:
        return v
    if norm_v > 1.5 * max_norm:
        return v / norm_v * max_norm
    elif norm_v > 0.5 * max_norm:
        return max_norm * np.tanh(norm_v / max_norm) * v / norm_v
    return v




def ssat_scalar(v: float, max_abs: float) -> float:
    """Smooth saturation for scalar."""
    if abs(v) > max_abs / 2.0:
        return max_abs * np.tanh(v / max_abs)
    return v


def softplus(x: float, beta: float = 1.0) -> float:
    """Softplus function with beta tuning parameter."""
    if x > 40.0:
        return x
    elif x < -20.0:
        return np.exp(x)
    else:
        return (1.0 / beta) * np.log1p(np.exp(beta * x))