import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import sim_utils
from robot_controller import RobotController
from task_sequence import TaskSequence

xml_path = 'franka_panda_w_objs.xml'
simend = 500

# Setup XML absolute path
dirname = os.path.dirname(__file__)
xml_path = os.path.join(dirname, xml_path)

# Load model and data
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

act1 = model.actuator("panda_gripper_finger_joint1").id
act2 = model.actuator("panda_gripper_finger_joint2").id

# fully open positions from XML ctrlrange
data.ctrl[act1] = model.actuator_ctrlrange[act1][1]  # finger1 fully open
data.ctrl[act2] = model.actuator_ctrlrange[act2][0]  # finger2 fully open (negative range)


# End-effector site
ee_site_name = "grip_site"
ee_site_id = model.site(ee_site_name).id

# Initialize controller and task sequence
controller_obj = RobotController(ee_site_id, model)
task_seq = TaskSequence(model)  # body IDs handled internally

# Initialize GLFW
if not glfw.init():
    raise RuntimeError("Failed to initialize GLFW")

window = glfw.create_window(1200, 900, "MuJoCo Demo", None, None)
if not window:
    glfw.terminate()
    raise RuntimeError("Failed to create GLFW window")

glfw.make_context_current(window)
glfw.swap_interval(1)  # enable vsync

# Camera and scene setup
cam = mj.MjvCamera()
opt = mj.MjvOption()
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# Callbacks
glfw.set_key_callback(window, sim_utils.keyboard(model, data))
glfw.set_mouse_button_callback(window, sim_utils.mouse_button())
glfw.set_cursor_pos_callback(window, sim_utils.mouse_move(model, scene, cam))
glfw.set_scroll_callback(window, sim_utils.scroll(model, scene, cam))

# Camera initial position
cam.azimuth = 89.6
cam.elevation = -11.6
cam.distance = 5.0
cam.lookat = np.array([0.0, 0.0, 2.0])

# Control callback
def control_callback(model_, data_):
    ee_pos = data_.site_xpos[ee_site_id].copy()
    ee_rot = data_.site_xmat[ee_site_id].reshape(3,3) 
    target_pos, target_rot, gripper_targets, gripper_open = task_seq.get_target(model_, data_, ee_pos, ee_rot)
    controller_obj.controller(model_, data_, target_pos, target_rot, gripper_targets, gripper_open)


    # Send finger joint targets to actuators
    data_.ctrl[act1] = gripper_targets["left"]
    data_.ctrl[act2] = gripper_targets["right"]


mj.set_mjcb_control(control_callback)

# Simulation loop
while not glfw.window_should_close(window):
    simstart = data.time
    while (data.time - simstart < 1.0 / 60.0):
        mj.mj_step(model, data)

    if data.time >= simend:
        break

    # Update and render scene
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()
