import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import sim_utils
import Functions as F

xml_path = 'franka_panda_w_objs.xml'
simend = 300

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

# -------------------------
# Controller
def controller(model, data):
    """
    Stable IK-based 6-DOF controller:
    - Moves grip_site to a position 10 cm in front of obj_box_06
    - Matches gripper orientation to the box orientation
    - Uses damped least squares with adaptive damping
    - Clamps dq and uses a small alpha step toward the IK solution
    - Uses joint-PD + gravity compensation
    """
    nv = model.nv

    # --- 1) current end-effector position & orientation ---
    ee_pos = data.site_xpos[ee_site_id].copy()        # shape (3,)
    R_ee = data.site_xmat[ee_site_id].reshape(3, 3)  # 3x3 rotation

    # --- 2) target position & orientation ---
    box_pos = data.xpos[target_body_id].copy()
    target_pos = box_pos + np.array([0.0, 0.0, 0.10])  # 10 cm above box
    R_box = data.xmat[target_body_id].reshape(3, 3)    # desired rotation

    R_box_rot = R_box @ F.RotX(np.pi/2) @ F.RotY(-1*np.pi /2) 

    # --- 3) position error ---
    pos_err = target_pos - ee_pos 

    # --- 4) rotation error (3x1 vector) ---
    R_err = 0.5 * (np.cross(R_ee[:,0], R_box_rot[:,0]) +
                   np.cross(R_ee[:,1], R_box_rot[:,1]) +
                   np.cross(R_ee[:,2], R_box_rot[:,2]))

    # --- 5) combined 6D error ---
    error = np.hstack((pos_err, R_err))  # 6x1

    # --- 6) 6xnv Jacobian ---
    Jp = np.zeros((3, nv))
    Jr = np.zeros((3, nv))
    mj.mj_jacSite(model, data, Jp, Jr, ee_site_id)
    J_full = np.vstack((Jp, Jr))  # 6 x nv

    # --- 7) Damped least squares ---
    lam0 = 1e-3
    lam_scale = 1e-2
    lam = lam0 + lam_scale * np.linalg.norm(pos_err)
    A = J_full @ J_full.T + lam * np.eye(6)
    try:
        dq = J_full.T @ np.linalg.solve(A, error)
    except np.linalg.LinAlgError:
        dq = J_full.T @ np.linalg.pinv(A) @ error

    # --- 8) Safety clamp & step size ---
    max_dq_norm = 0.2
    dq_norm = np.linalg.norm(dq)
    if dq_norm > max_dq_norm:
        dq *= max_dq_norm / dq_norm
    alpha = 0.25
    dq_step = alpha * dq

    # --- 9) desired joint positions ---
    q = data.qpos[:nv].copy()
    q_des = q + dq_step

    # --- 10) PD control + gravity compensation ---
    Kp = 200
    Kd = 10.0
    q_err = q_des - q
    qd = data.qvel[:nv].copy()
    f = data.qfrc_bias.copy()
    tau = (Kp * q_err) + (Kd * (-qd)) + f

    # --- 11) Torque clamp ---
    torque_limit = 200.0
    tau = np.clip(tau, -torque_limit, torque_limit)
    data.qfrc_applied[:] = tau



# -------------------------
# get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname, xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)
cam = mj.MjvCamera()
opt = mj.MjvOption()

# Site/body IDs
ee_site_name = "grip_site"
target_body_name = "obj_box_06"
ee_site_id = model.site(ee_site_name).id
target_body_id = model.body(target_body_name).id

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, sim_utils.keyboard(model, data))
glfw.set_mouse_button_callback(window, sim_utils.mouse_button())
glfw.set_cursor_pos_callback(window, sim_utils.mouse_move(model, scene, cam))
glfw.set_scroll_callback(window, sim_utils.scroll(model, scene, cam))

# Set camera configuration
cam.azimuth = 89.608063
cam.elevation = -11.588379
cam.distance = 5.0
cam.lookat = np.array([0.0, 0.0, 2.0])

# set the controller
mj.set_mjcb_control(controller)

while not glfw.window_should_close(window):
    simstart = data.time

    while (data.time - simstart < 1.0 / 60.0):
        mj.mj_step(model, data)

    if data.time >= simend:
        break

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()
