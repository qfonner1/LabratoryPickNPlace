import numpy as np
import mujoco as mj
import Functions as F

class RobotController:
    def __init__(self, ee_site_id, model):
        self.ee_site_id = ee_site_id
        self.object_grasped_flag = False
        self.mass_blend = 0.0
        self.tau_prev = None
        self.int_err = np.zeros(6)

        # Identify geometry IDs once
        self.object_geom_id = model.geom("box_geom").id
        self.left_finger_geom_id = model.geom("panda_finger1_collision").id
        self.right_finger_geom_id = model.geom("panda_finger2_collision").id

    def controller(self, model, data, target_pos, target_rot, gripper_targets, gripper_open):
        nv = model.nv
        ee_pos = data.site_xpos[self.ee_site_id].copy()
        R_ee = data.site_xmat[self.ee_site_id].reshape(3, 3)

        # --- EE pose error ---
        pos_err = target_pos - ee_pos
        R_err_mat = target_rot @ R_ee.T
        R_err = 0.5 * np.array([
            R_err_mat[2, 1] - R_err_mat[1, 2],
            R_err_mat[0, 2] - R_err_mat[2, 0],
            R_err_mat[1, 0] - R_err_mat[0, 1]
        ])
        error = np.hstack((pos_err, R_err))

        # --- Jacobian ---
        Jp = np.zeros((3, nv))
        Jr = np.zeros((3, nv))
        mj.mj_jacSite(model, data, Jp, Jr, self.ee_site_id)
        J_full = np.vstack((Jp, Jr))

        # --- Adaptive damping ---
        manip = np.sqrt(np.linalg.det(J_full @ J_full.T + 1e-6 * np.eye(6)))
        lam = 1e-3 / (manip + 1e-3)
        A = J_full @ J_full.T + lam * np.eye(6)

        try:
            dq = J_full.T @ np.linalg.solve(A, error)
        except np.linalg.LinAlgError:
            dq = J_full.T @ np.linalg.pinv(A) @ error

        # Clip joint velocity
        max_dq_norm = 0.5
        dq_norm = np.linalg.norm(dq)
        if dq_norm > max_dq_norm:
            dq *= max_dq_norm / dq_norm

        # --- Joint target update ---
        q = data.qpos[:nv].copy()
        q_des = q + 0.3 * dq
        q_des[7] = gripper_targets["left"]
        q_des[8] = gripper_targets["right"]

        # --- PD torque control (decoupled gains) ---
        arm_idx = np.arange(7)
        gripper_idx = np.array([7, 8])
        Kp_arm, Kd_arm = 250, 20
        Kp_grip, Kd_grip = 100, 20

        q_err = q_des - q
        qd = data.qvel[:nv].copy()
        f = data.qfrc_bias.copy()

        tau = f.copy()
        tau[arm_idx] += Kp_arm * q_err[arm_idx] + Kd_arm * (-qd[arm_idx])
        tau[gripper_idx] += Kp_grip * q_err[gripper_idx] + Kd_grip * (-qd[gripper_idx])

        if gripper_open == 0.0:  # gripper closed -> object grasped
            if not self.object_grasped_flag:
                self.object_grasped_flag = True
            self.mass_blend = min(self.mass_blend + 0.01, 1.0)
        else:  # gripper open -> object released
            if self.object_grasped_flag:
                self.object_grasped_flag = False
            self.mass_blend = max(self.mass_blend - 0.01, 0.0)

        # Apply mass compensation only if mass_blend > 0
        if self.mass_blend > 0:
            object_mass = 0.01  # kg
            gravity = model.opt.gravity
            wrench = np.hstack((object_mass * gravity, np.zeros(3)))
            extra_tau = J_full.T @ wrench * self.mass_blend
            tau += extra_tau

        # --- Torque rate limiting ---
        if self.tau_prev is None:
            self.tau_prev = tau.copy()
        max_torque_change = 10.0  # N·m per timestep
        delta_tau = np.clip(tau - self.tau_prev, -max_torque_change, max_torque_change)
        tau = self.tau_prev + delta_tau
        self.tau_prev = tau.copy()

        # --- Torque limits ---
        torque_limit = np.array([80, 80, 80, 80, 80, 12, 12])
        tau[:len(torque_limit)] = np.clip(tau[:len(torque_limit)], -torque_limit, torque_limit)

        # --- Apply torques ---
        data.qfrc_applied[:len(torque_limit)] = tau[:len(torque_limit)]
