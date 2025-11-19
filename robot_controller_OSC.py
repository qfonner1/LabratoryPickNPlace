import numpy as np
import mujoco as mj
import Functions as F  # External functions for quaternion math


class RobotController:
    def __init__(self, ee_site_id, model):
        self.ee_site_id = ee_site_id
        self.tau_prev = None  # For torque rate limiting

        # Identify geometry IDs once (used for potential collision or grasp logic)
        self.object_geom_id = model.geom("box_geom").id
        self.left_finger_geom_id = model.geom("panda_finger1_collision").id
        self.right_finger_geom_id = model.geom("panda_finger2_collision").id

    def controller(self, model, data, target_pos, target_rot, gripper_targets, gripper_open):
        nv = model.nv  # number of DOFs
        q = data.qpos[:nv].copy()  # current joint positions
        qd = data.qvel[:nv].copy()  # current joint velocities

        # --- End-effector (EE) pose ---
        ee_pos = data.site_xpos[self.ee_site_id].copy()  # EE position
        R_ee = data.site_xmat[self.ee_site_id].reshape(3, 3)  # EE rotation matrix

        # Convert EE rotation matrix to quaternion
        ee_quat = np.zeros(4)
        mj.mju_mat2Quat(ee_quat, R_ee.flatten())

        # Convert target rotation matrix to quaternion
        target_quat = np.zeros(4)
        mj.mju_mat2Quat(target_quat, target_rot.reshape(9))

        # --- Position error ---
        pos_err = target_pos - ee_pos

        # --- Quaternion orientation error ---
        q_err = F.quat_multiply(target_quat, F.quat_conjugate(ee_quat))  # relative rotation
        if q_err[0] < 0:
            q_err *= -1  # ensure shortest rotation
        ori_err = 3.0 * q_err[1:4]  # scale orientation error

        # --- Combined task-space error (6D) ---
        error = np.hstack((pos_err, ori_err))

        # --- Compute Jacobian at EE ---
        Jp = np.zeros((3, nv))  # linear part
        Jr = np.zeros((3, nv))  # angular part
        mj.mj_jacSite(model, data, Jp, Jr, self.ee_site_id)
        J_full = np.vstack((Jp, Jr))  # 6xnv Jacobian

        # --- Joint-space mass matrix and operational space inertia ---
        M = np.zeros((nv, nv))
        mj.mj_fullM(model, M, data.qM)  # joint-space mass matrix
        Minv = np.linalg.inv(M)
        Lambda = np.linalg.inv(J_full @ Minv @ J_full.T + 1e-6 * np.eye(6))  # task-space inertia

        # --- Task-space velocity ---
        xdot = J_full @ qd  # EE linear+angular velocity

        # --- Task-space PD gains ---
        Kp_pos, Kd_pos = 10, 5  # position gains
        Kp_ori, Kd_ori = 5, 2   # orientation gains
        Kp_task = np.diag([Kp_pos]*3 + [Kp_ori]*3)
        Kd_task = np.diag([Kd_pos]*3 + [Kd_ori]*3)

        # --- Task-space PD control law ---
        F_task = Kp_task @ error - Kd_task @ xdot  # desired EE force/torque

        # --- Map task-space forces to joint torques via Jacobian transpose ---
        tau_task = J_full.T @ (Lambda @ F_task)

        # --- Add gravity & Coriolis compensation ---
        tau = tau_task + data.qfrc_bias  # ensures dynamics are compensated

        # --- Gripper PD control ---
        tau[7] = 200 * (gripper_targets["left"] - q[7]) - 10 * qd[7]
        tau[8] = 200 * (gripper_targets["right"] - q[8]) - 10 * qd[8]

        # --- Torque rate limiting ---
        if self.tau_prev is None:
            self.tau_prev = tau.copy()
        max_torque_change = 10.0  # max change per timestep
        delta_tau = np.clip(tau - self.tau_prev, -max_torque_change, max_torque_change)
        tau = self.tau_prev + delta_tau
        self.tau_prev = tau.copy()

        # --- Torque saturation (joint limits) ---
        torque_limit = np.array([80, 80, 80, 80, 80, 20, 20])
        tau[:len(torque_limit)] = np.clip(tau[:len(torque_limit)], -torque_limit, torque_limit)

        # --- Apply torques to the robot ---
        data.qfrc_applied[:len(torque_limit)] = tau[:len(torque_limit)]
