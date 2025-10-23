import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import imageio
from robot_controller import RobotController
from task_sequence import TaskSequence
import object_detection as OD

xml_path = 'franka_panda_w_objs.xml'
simend = 500  # seconds

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
task_seq = TaskSequence(model)

detected_objects = OD.object_detection("franka_panda_w_objs.xml", "overhead_cam")
task_seq.set_targets_from_vision(detected_objects)

# Initialize GLFW
if not glfw.init():
    raise RuntimeError("Failed to initialize GLFW")

window = glfw.create_window(1200, 900, "Go Get it Buddy!", None, None)
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

# Camera initial position
cam.azimuth = 90
cam.elevation = -11.6
cam.distance = 6.0
cam.lookat = np.array([1.2, 0.0, 2.0])

# --- Offscreen Renderer Setup ---
renderer = mj.Renderer(model, width=640, height=480)
frames = []  # to store video frames

# Control callback
def control_callback(model_, data_):
    ee_pos = data_.site_xpos[ee_site_id].copy()
    ee_rot = data_.site_xmat[ee_site_id].reshape(3,3)
    target_pos, target_rot, gripper_targets, gripper_open = task_seq.get_target(model_, data_, ee_pos, ee_rot)
    controller_obj.controller(model_, data_, target_pos, target_rot, gripper_targets, gripper_open)
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

    # --- Render onscreen ---
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    glfw.poll_events()

    # --- Capture offscreen frame ---
    renderer.update_scene(data, camera=cam)
    pixels = renderer.render()
    frames.append(pixels)

glfw.terminate()
renderer.close()

# --- Save recorded video ---
output_path = os.path.join(dirname, "simulation.mp4")
imageio.mimsave(output_path, frames, fps=60)
print(f"🎥 Simulation video saved to: {output_path}")
