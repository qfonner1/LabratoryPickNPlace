import numpy as np
import mujoco as mj

def quat_multiply(q1, q2):
    """Hamilton product of two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_conjugate(q):
    """Quaternion conjugate (inverse for unit quaternions)."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


class RobotController:
    def __init__(self, ee_site_id, model):
        self.ee_site_id = ee_site_id
        self.object_grasped_flag = False
        self.mass_blend = 0.0
        self.tau_prev = None

        # Identify geometry IDs once
        self.object_geom_id = model.geom("box_geom").id
        self.left_finger_geom_id = model.geom("panda_finger1_collision").id
        self.right_finger_geom_id = model.geom("panda_finger2_collision").id


    def controller(self, model, data, target_pos, target_rot, gripper_targets, gripper_open):
        nv = model.nv
        q = data.qpos[:nv].copy()
        qd = data.qvel[:nv].copy()

        # --- EE pose ---
        ee_pos = data.site_xpos[self.ee_site_id].copy()
        R_ee = data.site_xmat[self.ee_site_id].reshape(3, 3)

        ee_quat = np.zeros(4)
        mj.mju_mat2Quat(ee_quat, R_ee.flatten())

        target_quat = np.zeros(4)
        mj.mju_mat2Quat(target_quat, target_rot.reshape(9))

        # --- Position error ---
        pos_err = target_pos - ee_pos

        # --- Quaternion orientation error ---
        q_err = quat_multiply(target_quat, quat_conjugate(ee_quat))
        if q_err[0] < 0:
            q_err *= -1  # ensure shortest rotation
        ori_err = 3.0 * q_err[1:4]

        # --- Combined task-space error ---
        error = np.hstack((pos_err, ori_err))

        # --- Jacobian ---
        Jp = np.zeros((3, nv))
        Jr = np.zeros((3, nv))
        mj.mj_jacSite(model, data, Jp, Jr, self.ee_site_id)
        J_full = np.vstack((Jp, Jr))

        # --- Mass matrix & operational space inertia (Λ) ---
        M = np.zeros((nv, nv))
        mj.mj_fullM(model, M, data.qM)
        Minv = np.linalg.inv(M)
        Lambda = np.linalg.inv(J_full @ Minv @ J_full.T + 1e-6 * np.eye(6))

        # --- Task-space velocities ---
        xdot = J_full @ qd

        # --- Task-space PD control law (force/torque) ---
        Kp_pos, Kd_pos = 10, 5
        Kp_ori, Kd_ori = 5, 2

        Kp_task = np.diag([Kp_pos]*3 + [Kp_ori]*3)
        Kd_task = np.diag([Kd_pos]*3 + [Kd_ori]*3)

        F_task = Kp_task @ error - Kd_task @ xdot

        # --- Joint torques via OSC mapping ---
        tau_task = J_full.T @ (Lambda @ F_task)

        # --- Gravity & Coriolis compensation ---
        tau = tau_task + data.qfrc_bias

        # --- Gripper ---
        tau[7] = 200 * (gripper_targets["left"] - q[7]) - 10 * qd[7]
        tau[8] = 200 * (gripper_targets["right"] - q[8]) - 10 * qd[8]

        # --- Torque rate limiting ---
        if self.tau_prev is None:
            self.tau_prev = tau.copy()
        max_torque_change = 10.0
        delta_tau = np.clip(tau - self.tau_prev, -max_torque_change, max_torque_change)
        tau = self.tau_prev + delta_tau
        self.tau_prev = tau.copy()

        # --- Torque limits ---
        torque_limit = np.array([80, 80, 80, 80, 80, 20, 20])
        tau[:len(torque_limit)] = np.clip(tau[:len(torque_limit)], -torque_limit, torque_limit)

        # --- Apply torques ---
        data.qfrc_applied[:len(torque_limit)] = tau[:len(torque_limit)]
