import sys,mujoco,time,os,json
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../MAE589Project/package/helper/')
sys.path.append('../MAE589Project/package/mujoco_usage/')
sys.path.append('../MAE589Project/package/gpt_usage/')
sys.path.append('../MAE589Project/package/detection_module/')
from mujoco_parser import *
from utility import *
from transformation import *
from gpt_helper import *
from owlv2 import *
np.set_printoptions(precision=2,suppress=True,linewidth=100)
plt.rc('xtick',labelsize=6); plt.rc('ytick',labelsize=6)

print ("Ready.")

import mujoco as mj
import imageio
from mujoco.glfw import glfw
import numpy as np
import sys
import os
from OpenGL import GL
import matplotlib.pyplot as plt  # for visualization
import sim_utils
from robot_controller import RobotController
from task_sequence import TaskSequence

# Add your sim_utils and Functions paths
sim_utils_path = sys.path.append('../MAE589Project')
if sim_utils_path not in sys.path:
    sys.path.append(sim_utils_path)
import sim_utils
import Functions as F

# XML model path
xml_path = 'franka_panda_w_objs.xml'
simend = 300

# Load model and data
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# Cameras
main_cam = mj.MjvCamera()
rgbd_cam = mj.MjvCamera()
mj.mjv_defaultCamera(main_cam)
mj.mjv_defaultCamera(rgbd_cam)

# Set main camera position
main_cam.azimuth = 90
main_cam.elevation = -15
main_cam.distance = 2.5
main_cam.lookat = np.array([0.0, 0.0, 1.0])

# GLFW setup (unchanged)
if not glfw.init():
    raise Exception("GLFW init failed")
win_width, win_height = 1280, 480  # side-by-side window
window = glfw.create_window(win_width, win_height, "Simulation + RGB-D Side-by-Side", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window creation failed")

glfw.make_context_current(window)
glfw.swap_interval(1)

context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# Options and scenes
opt = mj.MjvOption()
scene_main = mj.MjvScene(model, maxgeom=10000)
scene_rgbd = mj.MjvScene(model, maxgeom=10000)

# IDs
ee_site_id = model.site("grip_site").id
right_hand_body_id = model.body("right_hand").id
act1 = model.actuator("panda_gripper_finger_joint1").id
act2 = model.actuator("panda_gripper_finger_joint2").id

controller_obj = RobotController(ee_site_id,model)
task_seq = TaskSequence(model)  # body IDs handled internally

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

# Setup input callbacks
glfw.set_key_callback(window, sim_utils.keyboard(model, data))
glfw.set_mouse_button_callback(window, sim_utils.mouse_button())
glfw.set_cursor_pos_callback(window, sim_utils.mouse_move(model, scene_main, main_cam))
glfw.set_scroll_callback(window, sim_utils.scroll(model, scene_main, main_cam))

# Right-hand camera update function (unchanged)
def update_right_hand_camera(model, data, cam, right_hand_body_id):
    body_pos = data.xpos[right_hand_body_id].copy()
    body_rot = data.xmat[right_hand_body_id].reshape(3, 3)
    offset = body_rot @ np.array([-0.2, 0.4, 0.5])
    cam.lookat = body_pos
    cam.distance = np.linalg.norm(offset)
    dx, dy, dz = offset
    cam.azimuth = np.degrees(np.arctan2(dy, dx)) + 90
    cam.elevation = np.degrees(np.arctan2(dz, np.linalg.norm([dx, dy])))

# --- New: Function to grab egocentric RGB and depth images from a camera ---
def get_egocentric_rgbd(model, data, cam, context, width=320, height=240, fovy=45):
    # Set camera parameters (azimuth, elevation, distance, lookat) before calling this function

    # Set FOV if supported
    if hasattr(cam, 'fovy'):
        cam.fovy = fovy

    # Update scene for this camera
    opt = mj.MjvOption()
    scene = mj.MjvScene(model, maxgeom=10000)
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)

    # Create buffer for rgb and depth
    rgb_buffer = (GL.GLubyte * (width * height * 3))()
    depth_buffer = (GL.GLfloat * (width * height))()

    # Render to buffers using OpenGL Framebuffer
    mj.mjr_render(mj.MjrRect(0, 0, width, height), scene, context)

    # Read pixels from OpenGL buffers
    GL.glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, rgb_buffer)
    GL.glReadPixels(0, 0, width, height, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, depth_buffer)

    # Convert buffers to numpy arrays and flip vertically
    rgb_img = np.frombuffer(rgb_buffer, dtype=np.uint8).reshape((height, width, 3))[::-1, :, :]
    depth_img = np.frombuffer(depth_buffer, dtype=np.float32).reshape((height, width))[::-1, :]

    return rgb_img, depth_img


# --- New: Convert depth + camera intrinsics to point cloud ---
def depth_to_pointcloud(depth, fovy_deg=45):
    h, w = depth.shape
    fovy = np.deg2rad(fovy_deg)
    fx = fy = 0.5 * w / np.tan(fovy / 2)
    cx = w / 2
    cy = h / 2

    indices = np.indices((h, w), dtype=np.float32)
    x_indices = indices[1]
    y_indices = indices[0]

    z = depth
    x = (x_indices - cx) * z / fx
    y = (y_indices - cy) * z / fy
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    return points

# Flag to capture only once after 10 seconds
captured = False

while not glfw.window_should_close(window):
    simstart = data.time
    while data.time - simstart < 1.0 / 60.0:
        mj.mj_step(model, data)
    if data.time >= simend:
        break

    # Update main camera scene
    mj.mjv_updateScene(model, data, opt, None, main_cam, mj.mjtCatBit.mjCAT_ALL.value, scene_main)
    # Update right hand camera
    update_right_hand_camera(model, data, rgbd_cam, right_hand_body_id)
    mj.mjv_updateScene(model, data, opt, None, rgbd_cam, mj.mjtCatBit.mjCAT_ALL.value, scene_rgbd)

    # Clear screen
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

    # Left viewport - main camera
    GL.glViewport(0, 0, win_height, win_height)
    mj.mjr_render(mj.MjrRect(0, 0, win_width // 2, win_height), scene_main, context)

    # Right viewport - right hand camera
    GL.glViewport(win_width // 2, 0, win_width // 2, win_height)
    mj.mjr_render(mj.MjrRect(win_width // 2, 0, win_width // 2, win_height), scene_rgbd, context)

    glfw.swap_buffers(window)
    glfw.poll_events()

    # Capture and show image **once** after 10 seconds
    if not captured and data.time >= 10.0:
        rgb_img, depth_img = get_egocentric_rgbd(model, data, rgbd_cam, context, width=320, height=240, fovy=45)
        pcd = depth_to_pointcloud(depth_img, fovy_deg=45)
        h,w=depth_img.shape
        xyz_img_world=pcd.reshape(h,w,3)
        # Save RGB image as PNG
        imageio.imwrite("egocentric_rgb_10s.png", rgb_img)

        print(f"Saved egocentric RGB image at 10s as 'egocentric_rgb_10s.png'")
        captured = True  # prevent further capture

glfw.terminate()
print("Simulation finished.")
