import numpy as np
import mujoco as mj

class RobotController:
    def __init__(self, ee_site_id, model):
        # Store end-effector site ID
        self.ee_site_id = ee_site_id

        # Flags and variables for grasping and torque rate limiting
        self.object_grasped_flag = False   # True if gripper is holding object
        self.mass_blend = 0.0              # Smooth blending factor for mass compensation
        self.tau_prev = None               # Previous applied torques for rate limiting
        self.int_err = np.zeros(6)         # Placeholder for integral error (not used here)

        # Cache geometry IDs for object and fingers for contact/mass handling
        self.object_geom_id = model.geom("box_geom").id
        self.left_finger_geom_id = model.geom("panda_finger1_collision").id
        self.right_finger_geom_id = model.geom("panda_finger2_collision").id

    def controller(self, model, data, target_pos, target_rot, gripper_targets, gripper_open):
        nv = model.nv  # Number of DOFs

        # --- Current end-effector pose ---
        ee_pos = data.site_xpos[self.ee_site_id].copy()      # Current EE position
        R_ee = data.site_xmat[self.ee_site_id].reshape(3, 3) # Current EE rotation matrix

        # --- Compute pose error in operational space ---
        pos_err = target_pos - ee_pos                          # Position error
        R_err_mat = target_rot @ R_ee.T                        # Relative rotation
        # Convert rotation matrix error to rotation vector (vee operator)
        R_err = 0.5 * np.array([
            R_err_mat[2, 1] - R_err_mat[1, 2],
            R_err_mat[0, 2] - R_err_mat[2, 0],
            R_err_mat[1, 0] - R_err_mat[0, 1]
        ])
        orientation_gain = 3.0                                 # Scaling for rotation error
        error = np.hstack((pos_err, orientation_gain * R_err)) # 6D EE error [x,y,z,rx,ry,rz]

        # --- Compute end-effector Jacobian ---
        Jp = np.zeros((3, nv))  # Linear velocity part
        Jr = np.zeros((3, nv))  # Angular velocity part
        mj.mj_jacSite(model, data, Jp, Jr, self.ee_site_id)   # Compute Jacobian at EE
        J_full = np.vstack((Jp, Jr))                          # Full 6xnv Jacobian

        # --- Adaptive damping to avoid singularities ---
        manip = np.sqrt(np.linalg.det(J_full @ J_full.T + 1e-6 * np.eye(6))) # Manipulability
        lam = 1e-3 / (manip + 1e-3)                                        # Damping factor
        A = J_full @ J_full.T + lam * np.eye(6)                              # Damped matrix

        # Solve for joint velocity using operational-space PD
        try:
            dq = J_full.T @ np.linalg.solve(A, error)
        except np.linalg.LinAlgError:
            dq = J_full.T @ np.linalg.pinv(A) @ error

        # Clip joint velocity to prevent excessive motion
        max_dq_norm = 0.5
        dq_norm = np.linalg.norm(dq)
        if dq_norm > max_dq_norm:
            dq *= max_dq_norm / dq_norm

        # --- Update desired joint positions ---
        q = data.qpos[:nv].copy()      # Current joint positions
        q_des = q + 0.3 * dq           # Integrate dq to desired q
        q_des[7] = gripper_targets["left"]   # Set gripper left finger
        q_des[8] = gripper_targets["right"]  # Set gripper right finger

        # --- PD torque control ---
        arm_idx = np.arange(7)          # Indices for arm joints
        gripper_idx = np.array([7, 8])  # Indices for gripper joints
        Kp_arm, Kd_arm = 250, 20       # Arm gains
        Kp_grip, Kd_grip = 100, 20     # Gripper gains

        q_err = q_des - q              # Joint position error
        qd = data.qvel[:nv].copy()     # Joint velocities
        f = data.qfrc_bias.copy()      # Coriolis + gravity compensation

        tau = f.copy()  # Start with bias forces
        # Apply PD control to arm
        tau[arm_idx] += Kp_arm * q_err[arm_idx] + Kd_arm * (-qd[arm_idx])
        # Apply PD control to gripper
        tau[gripper_idx] += Kp_grip * q_err[gripper_idx] + Kd_grip * (-qd[gripper_idx])

        # --- Handle object grasping and mass compensation ---
        if gripper_open == 0.0:  # Gripper closed -> object grasped
            if not self.object_grasped_flag:
                self.object_grasped_flag = True
            self.mass_blend = min(self.mass_blend + 0.01, 1.0)  # Smooth blend in
        else:  # Gripper open -> object released
            if self.object_grasped_flag:
                self.object_grasped_flag = False
            self.mass_blend = max(self.mass_blend - 0.01, 0.0)  # Smooth blend out

        # Apply extra torque to compensate for object mass
        if self.mass_blend > 0:
            object_mass = 0.01  # kg
            gravity = model.opt.gravity
            wrench = np.hstack((object_mass * gravity, np.zeros(3)))  # Only force, no torque
            extra_tau = J_full.T @ wrench * self.mass_blend           # Map wrench to joint torques
            tau += extra_tau

        # --- Torque rate limiting ---
        if self.tau_prev is None:
            self.tau_prev = tau.copy()
        max_torque_change = 10.0  # N·m per timestep
        delta_tau = np.clip(tau - self.tau_prev, -max_torque_change, max_torque_change)
        tau = self.tau_prev + delta_tau
        self.tau_prev = tau.copy()

        # --- Torque limits per joint ---
        torque_limit = np.array([80, 80, 80, 80, 80, 12, 12])
        tau[:len(torque_limit)] = np.clip(tau[:len(torque_limit)], -torque_limit, torque_limit)

        # --- Apply torques to MuJoCo ---
        data.qfrc_applied[:len(torque_limit)] = tau[:len(torque_limit)]
