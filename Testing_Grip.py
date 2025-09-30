import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import sim_utils
import Functions as F

xml_path = 'franka_panda_w_objs.xml'
simend = 300

import numpy as np

class RobotController:
    def __init__(self, ee_site_id, target_body_id):
        self.ee_site_id = ee_site_id
        self.target_body_id = target_body_id

        self.current_step = 0
        self.steps = [
            self.step_move_above_block,
            self.step_open_gripper,
            self.step_wait_after_open,       # New wait step
            self.step_move_middle_block,
            self.step_move_hold_clamp,
            self.step_move_it
        ]

        self.gripper_open = False
        self.wait_duration = 5  # seconds
        self.wait_timer = 0.0
        self.waiting = False

    def step_move_above_block(self, box_pos):
        offset = np.array([0.0, 0.0, 0.10])
        return box_pos + offset
    
    def step_move_it(self, box_pos):
        offset = np.array([0.03, 0.0, 0.0])
        return box_pos + offset
    
    def step_move_up(self, box_pos):
        offset = np.array([0.03, 0.1, 0.0])
        return box_pos + offset

    def step_open_gripper(self, box_pos):
        offset = np.array([0.03, 0.0, 0.10])
        return box_pos + offset

    def step_wait_after_open(self, box_pos):
        offset = np.array([0.03, 0.0, 0.10])
        return box_pos + offset

    def step_move_middle_block(self, box_pos):
        offset = np.array([0.03, 0.0, 0.0])
        return box_pos + offset

    def step_close_gripper(self, box_pos):
        offset = np.array([0.03, 0.0, 0.0])
        return box_pos + offset

    def step_move_hold_clamp(self, box_pos):
        offset = np.array([0.0, 0.0, 0.0])
        return box_pos + offset

    def controller(self, model, data):
        nv = model.nv
        ee_pos = data.site_xpos[self.ee_site_id].copy()
        R_ee = data.site_xmat[self.ee_site_id].reshape(3, 3)
        box_pos = data.xpos[self.target_body_id].copy()

        # Get current step target
        target_pos = self.steps[self.current_step](box_pos)

        # Handle wait state
        if self.waiting:
            self.wait_timer += model.opt.timestep
            if self.wait_timer >= self.wait_duration:
                self.waiting = False
                self.wait_timer = 0.0
                self.current_step += 1
                print(f"[Wait Done] Advancing to step {self.current_step}: {self.steps[self.current_step].__name__}")
        else:
            pos_err = target_pos - ee_pos
            if np.linalg.norm(pos_err) < 0.03:
                # Trigger wait after specific steps
                if self.steps[self.current_step] in [self.step_wait_after_open, self.step_move_hold_clamp]:
                    self.waiting = True
                    self.wait_timer = 0.0
                    print(f"[Waiting] Holding for 5 seconds at step {self.current_step}: {self.steps[self.current_step].__name__}")
                else:
                    if self.current_step < len(self.steps) - 1:
                        self.current_step += 1
                        print(f"Advancing to step {self.current_step}: {self.steps[self.current_step].__name__}")

        # Set gripper state manually by step index
        if self.steps[self.current_step] in [self.step_open_gripper, self.step_wait_after_open, self.step_move_middle_block]:
            self.gripper_open = True
        else:
            self.gripper_open = False

        # Orientation tracking
        R_box = data.xmat[self.target_body_id].reshape(3, 3)
        R_box_rot = R_box @ F.RotX(np.pi/2) @ F.RotY(-np.pi/2)

        pos_err = target_pos - ee_pos
        R_err = 0.5 * (np.cross(R_ee[:, 0], R_box_rot[:, 0]) +
                       np.cross(R_ee[:, 1], R_box_rot[:, 1]) +
                       np.cross(R_ee[:, 2], R_box_rot[:, 2]))
        error = np.hstack((pos_err, R_err))

        # Jacobian and IK solution
        Jp = np.zeros((3, nv))
        Jr = np.zeros((3, nv))
        mj.mj_jacSite(model, data, Jp, Jr, self.ee_site_id)
        J_full = np.vstack((Jp, Jr))

        lam0 = 1e-3
        lam_scale = 1e-2
        lam = lam0 + lam_scale * np.linalg.norm(pos_err)
        A = J_full @ J_full.T + lam * np.eye(6)

        try:
            dq = J_full.T @ np.linalg.solve(A, error)
        except np.linalg.LinAlgError:
            dq = J_full.T @ np.linalg.pinv(A) @ error

        max_dq_norm = 0.2
        dq_norm = np.linalg.norm(dq)
        if dq_norm > max_dq_norm:
            dq *= max_dq_norm / dq_norm

        alpha = 0.25
        dq_step = alpha * dq

        q = data.qpos[:nv].copy()
        q_des = q + dq_step

        # Gripper control
        if self.gripper_open:
            q_des[7] = np.pi
            q_des[8] = -np.pi
        else:
            q_des[7] = 0.0
            q_des[8] = 0.0

        # PD + bias compensation
        Kp = 200
        Kd = 10.0
        q_err = q_des - q
        qd = data.qvel[:nv].copy()
        f = data.qfrc_bias.copy()
        tau = (Kp * q_err) + (Kd * (-qd)) + f

        torque_limit = 100.0
        tau = np.clip(tau, -torque_limit, torque_limit)
        data.qfrc_applied[:] = tau




# -------------------------
# Setup
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname, xml_path)
xml_path = abspath

model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)
cam = mj.MjvCamera()
opt = mj.MjvOption()

ee_site_name = "grip_site"
target_body_name = "obj_box_06"
ee_site_id = model.site(ee_site_name).id
target_body_id = model.body(target_body_name).id

controller_obj = RobotController(ee_site_id, target_body_id)

glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

glfw.set_key_callback(window, sim_utils.keyboard(model, data))
glfw.set_mouse_button_callback(window, sim_utils.mouse_button())
glfw.set_cursor_pos_callback(window, sim_utils.mouse_move(model, scene, cam))
glfw.set_scroll_callback(window, sim_utils.scroll(model, scene, cam))

cam.azimuth = 89.608063
cam.elevation = -11.588379
cam.distance = 5.0
cam.lookat = np.array([0.0, 0.0, 2.0])

mj.set_mjcb_control(controller_obj.controller)

while not glfw.window_should_close(window):
    simstart = data.time

    while (data.time - simstart < 1.0 / 60.0):
        mj.mj_step(model, data)

    if data.time >= simend:
        break

    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    glfw.swap_buffers(window)
    glfw.poll_events()
