import numpy as np


def RotX(angle):
    Rx = np.array([[1, 0, 0],[0, np.cos(angle), -np.sin(angle)],[0, np.sin(angle),  np.cos(angle)]])
    return Rx

def RotY(angle):
    Ry = np.array([[ np.cos(angle), 0, np.sin(angle)],[ 0,1,0],[-np.sin(angle), 0, np.cos(angle)]])
    return Ry

def RotZ(angle):
    Rz = np.array([[ np.cos(angle), -np.sin(angle), 0],[np.sin(angle), np.cos(angle),0],[ 0,0,1]])
    return Rz

# --- Quaternion helpers for rotation interpolation (slerp) ---
def rotmat_to_quat(R):
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    # Source: numerically stable conversion
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0  # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2,1] - R[1,2]) / s
        qy = (R[0,2] - R[2,0]) / s
        qz = (R[1,0] - R[0,1]) / s
    else:
        # Find major diagonal element
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            qw = (R[2,1] - R[1,2]) / s
            qx = 0.25 * s
            qy = (R[0,1] + R[1,0]) / s
            qz = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            qw = (R[0,2] - R[2,0]) / s
            qx = (R[0,1] + R[1,0]) / s
            qy = 0.25 * s
            qz = (R[1,2] + R[2,1]) / s
        else:
            s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            qw = (R[1,0] - R[0,1]) / s
            qx = (R[0,2] + R[2,0]) / s
            qy = (R[1,2] + R[2,1]) / s
            qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=float)
    return q / np.linalg.norm(q)

def quat_to_rotmat(q):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ], dtype=float)
    return R

def quat_slerp(q0, q1, t, long_path):
    """Spherical linear interpolation between quaternions q0->q1 at fraction t (0..1).
    long_path=True forces interpolation along the longer rotational arc.
    """
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    
    dot = np.dot(q0, q1)
    
    # Check if quaternions are nearly identical
    if np.abs(dot) > 1.0 - 1e-6:
        return q0  # No meaningful rotation, return start
    
    # If long_path is requested, flip to take the long route
    if long_path and dot > 0.0:
        q1 = -q1
        dot = -dot
    # Standard short path if not long_path
    # elif not long_path and dot < 0.0:
    #     q1 = -q1
    #     dot = -dot
    
    dot = np.clip(dot, -1.0, 1.0)
    DOT_THRESH = 0.9995
    if dot > DOT_THRESH:
        # Very close -> linear interpolation
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    q = s0 * q0 + s1 * q1
    return q / np.linalg.norm(q)

