import numpy as np
import mujoco as mj
import Functions as F

class RobotController:
    def __init__(self, ee_site_id, model):
        self.ee_site_id = ee_site_id
        self.object_grasped_flag = False

        # Identify geometry IDs once
        self.object_geom_id = model.geom("box_geom").id
        self.left_finger_geom_id = model.geom("panda_finger1_collision").id
        self.right_finger_geom_id = model.geom("panda_finger2_collision").id

    def controller(self, model, data, target_pos, target_rot, gripper_targets, gripper_open):
        nv = model.nv
        ee_pos = data.site_xpos[self.ee_site_id].copy()
        R_ee = data.site_xmat[self.ee_site_id].reshape(3, 3)

        # --- Compute EE pose error ---
        pos_err = target_pos - ee_pos
        R_err = 0.5 * (np.cross(R_ee[:, 0], target_rot[:, 0]) +
                       np.cross(R_ee[:, 1], target_rot[:, 1]) +
                       np.cross(R_ee[:, 2], target_rot[:, 2]))
        error = np.hstack((pos_err, R_err))

        # --- Jacobian ---
        Jp = np.zeros((3, nv))
        Jr = np.zeros((3, nv))
        mj.mj_jacSite(model, data, Jp, Jr, self.ee_site_id)
        J_full = np.vstack((Jp, Jr))

        # --- Inverse Kinematics ---
        lam0 = 1e-3
        lam_scale = 1e-2
        lam = lam0 + lam_scale * np.linalg.norm(pos_err)
        A = J_full @ J_full.T + lam * np.eye(6)

        try:
            dq = J_full.T @ np.linalg.solve(A, error)
        except np.linalg.LinAlgError:
            dq = J_full.T @ np.linalg.pinv(A) @ error

        # Clip velocity
        max_dq_norm = 0.2
        dq_norm = np.linalg.norm(dq)
        if dq_norm > max_dq_norm:
            dq *= max_dq_norm / dq_norm

        alpha = 0.3
        dq_step = alpha * dq
        q = data.qpos[:nv].copy()
        q_des = q + dq_step

        # --- Gripper control (target positions already set) ---
        q_des[7] = gripper_targets["left"]
        q_des[8] = gripper_targets["right"]

        # --- PD + Feedforward Torque ---
        Kp = 250
        Kd = 20
        q_err = q_des - q
        qd = data.qvel[:nv].copy()
        f = data.qfrc_bias.copy()
        tau = (Kp * q_err) + (Kd * (-qd)) + f

        if not self.object_grasped_flag and gripper_open == 0.0:
            self.object_grasped_flag = True


        # --- Add object mass torque after grasp ---
        if self.object_grasped_flag:
            object_mass = 0.01  # kg
            gravity = model.opt.gravity
            wrench = np.hstack((object_mass * gravity, np.zeros(3)))
            extra_tau = J_full.T @ wrench
            tau += extra_tau


        # --- Apply torque ---
        torque_limit = 80.0
        tau = np.clip(tau, -torque_limit, torque_limit)
        data.qfrc_applied[:7] = tau[:7]



