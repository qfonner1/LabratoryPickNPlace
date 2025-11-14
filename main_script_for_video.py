import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import imageio
from robot_controller_OSC import RobotController
from task_sequence import TaskSequence
import object_detection as OD

xml_path = 'franka_panda_w_objs.xml'
simend = 500  # seconds

# --- Setup XML absolute path ---
dirname = os.path.dirname(__file__)
xml_path = os.path.join(dirname, xml_path)

# --- Load model and data ---
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# --- Set initial joint positions ---
initial_qpos = np.array([-0.25, -1.75, -1.0, -2.4, 2.4, 0.8, 1.4])
qpos = data.qpos.copy()
qpos[:7] = initial_qpos
data.qpos[:] = qpos
mj.mj_forward(model, data)

# --- Gripper setup ---
act1 = model.actuator("panda_gripper_finger_joint1").id
act2 = model.actuator("panda_gripper_finger_joint2").id
data.ctrl[act1] = model.actuator_ctrlrange[act1][1]
data.ctrl[act2] = model.actuator_ctrlrange[act2][0]

# --- Initialize controller and task sequence ---
ee_site_name = "grip_site"
ee_site_id = model.site(ee_site_name).id
controller_obj = RobotController(ee_site_id, model)
task_seq = TaskSequence(model)

detected_objects = OD.object_detection("franka_panda_w_objs.xml", "overhead_cam")
detected_targets=OD.object_detection("franka_panda_w_objs.xml", "overhead_cam2")
task_seq.set_boxes_from_vision(detected_objects)
task_seq.set_targets_from_vision(detected_targets)

ee_pos = data.site_xpos[ee_site_id].copy()  
task_seq.generate_steps(ee_pos)

# --- Initialize GLFW ---
if not glfw.init():
    raise RuntimeError("Failed to initialize GLFW")

window = glfw.create_window(1200, 900, "Go Get it Buddy!", None, None)
if not window:
    glfw.terminate()
    raise RuntimeError("Failed to create GLFW window")

glfw.make_context_current(window)
glfw.swap_interval(1) 

# --- Main viewer camera ---
cam = mj.MjvCamera()
opt = mj.MjvOption()
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# Camera initial view (for user window)
cam.azimuth = 90
cam.elevation = -12
cam.distance = 5.0
cam.lookat = np.array([0.0, 0.0, 0.0])

# --- Offscreen renderers (reuse, don't recreate) ---
width, height = 640, 480
renderer1 = mj.Renderer(model, width=width, height=height)
renderer2 = mj.Renderer(model, width=width, height=height)
renderer3 = mj.Renderer(model, width=width, height=height)

frames = []  # to store combined frames

# --- Control callback ---
def control_callback(model_, data_):
    ee_pos = data_.site_xpos[ee_site_id].copy()
    ee_rot = data_.site_xmat[ee_site_id].reshape(3, 3)
    target_pos, target_rot, gripper_targets, gripper_open = task_seq.get_target(model_, data_, ee_pos, ee_rot)
    controller_obj.controller(model_, data_, target_pos, target_rot, gripper_targets, gripper_open)
    data_.ctrl[act1] = gripper_targets["left"]
    data_.ctrl[act2] = gripper_targets["right"]

mj.set_mjcb_control(control_callback)

# --- Define two virtual cameras at opposite ends of the table ---
cam_custom1 = mj.MjvCamera()
cam_custom2 = mj.MjvCamera()
cam_custom3 = mj.MjvCamera()
mj.mjv_defaultCamera(cam_custom1)
mj.mjv_defaultCamera(cam_custom2)
mj.mjv_defaultCamera(cam_custom3)

# Camera 1: +X end of table, looking at origin
cam_custom1.lookat = np.array([0.0, 0.0, 1.0])  # look at center
cam_custom1.distance = 2.0                        # distance from lookat
cam_custom1.azimuth = 0                           # +X side
cam_custom1.elevation = -20                        # slight downward tilt

# Camera 2: -X end of table, looking at origin
cam_custom2.lookat = np.array([0.0, 0.0, 1.0])
cam_custom2.distance = 2.0
cam_custom2.azimuth = 180                          # -X side
cam_custom2.elevation = -20

# Camera 3: center
cam_custom3.lookat = np.array([0.0, 0.0, 1.0])  # look at center
cam_custom3.distance = 3.0                        # distance from lookat
cam_custom3.azimuth = 90                          # +X side
cam_custom3.elevation = -20                        # slight downward tilt

renderer1 = mj.Renderer(model, width=width, height=height)
renderer2 = mj.Renderer(model, width=width, height=height)
renderer3 = mj.Renderer(model, width=width, height=height)

def render_camera(camera_name):
    renderer_tmp = mj.Renderer(model, width=width, height=height)
    renderer_tmp.update_scene(data, camera=camera_name)
    rgb = renderer_tmp.render()
    renderer_tmp.close()
    return rgb

# --- Simulation loop ---
while not glfw.window_should_close(window):
    simstart = data.time
    while (data.time - simstart < 1.0 / 60.0):
        mj.mj_step(model, data)

    if data.time >= simend:
        break

    # Onscreen render (human viewer)
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    glfw.poll_events()

    rgb1 = render_camera(cam_custom1)
    rgb2 = render_camera(cam_custom2)
    rgb3 = render_camera(cam_custom3)

    # Combine side by side
    combined = np.concatenate([rgb1, rgb3, rgb2], axis=1)
    frames.append(combined)

# --- Cleanup ---
glfw.terminate()
renderer1.close()
renderer2.close()
renderer3.close()


# --- Save recorded video ---
output_path = os.path.join(dirname, "simulation_standingcams.mp4")
imageio.mimsave(output_path, frames, fps=60)
print(f"🎥 Three-view simulation video saved to: {output_path}")
