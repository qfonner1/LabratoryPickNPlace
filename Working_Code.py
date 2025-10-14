import mujoco as mj
from mujoco.glfw import glfw
import OpenGL.GL as gl
from PIL import Image
import numpy as np
import os
import sim_utils
import Functions as F

# =========== Your XML and sim settings ===========
xml_path = 'franka_panda_w_objs.xml'
simend = 300

# For callback functions (if any)
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

# --------- Controller function (your IK controller) -----------
def controller(model, data):
    nv = model.nv
    ee_pos = data.site_xpos[ee_site_id].copy()        # shape (3,)
    R_ee = data.site_xmat[ee_site_id].reshape(3, 3)  # 3x3 rotation

    box_pos = data.xpos[target_body_id].copy()
    target_pos = box_pos + np.array([0.0, 0.0, 0.10])  # 10 cm above box
    R_box = data.xmat[target_body_id].reshape(3, 3)

    R_box_rot = R_box @ F.RotX(np.pi/2) @ F.RotY(-1*np.pi /2)

    pos_err = target_pos - ee_pos

    R_err = 0.5 * (np.cross(R_ee[:,0], R_box_rot[:,0]) +
                   np.cross(R_ee[:,1], R_box_rot[:,1]) +
                   np.cross(R_ee[:,2], R_box_rot[:,2]))

    error = np.hstack((pos_err, R_err))  # 6x1

    Jp = np.zeros((3, nv))
    Jr = np.zeros((3, nv))
    mj.mj_jacSite(model, data, Jp, Jr, ee_site_id)
    J_full = np.vstack((Jp, Jr))  # 6 x nv

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

    Kp = 200
    Kd = 10.0
    q_err = q_des - q
    qd = data.qvel[:nv].copy()
    f = data.qfrc_bias.copy()
    tau = (Kp * q_err) + (Kd * (-qd)) + f

    torque_limit = 200.0
    tau = np.clip(tau, -torque_limit, torque_limit)
    data.qfrc_applied[:] = tau

# --------- Grab RGB image from OpenGL framebuffer ----------
def grab_mujoco_rgb_image(viewport_width, viewport_height):
    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
    pixels = gl.glReadPixels(0, 0, viewport_width, viewport_height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
    image = np.frombuffer(pixels, dtype=np.uint8).reshape(viewport_height, viewport_width, 4)
    image = np.flip(image, axis=0)  # Flip vertically
    return Image.fromarray(image)

# --------- Depth to xyz point cloud in camera frame -----------
def depth_to_xyz(depth_img, cam, viewport_width, viewport_height):
    # Convert depth buffer to z-buffer range [0,1]
    depth = depth_img.astype(np.float32) / (2**24 - 1)
    # MuJoCo depth is nonlinear, convert to linear depth using near/far planes
    near = 0.01
    far = 1000.0
    z_linear = near * far / (far - (far - near) * depth)

    # Compute focal length in pixels
    fovy_rad = np.deg2rad(45)  # You can adjust if you want to use cam params differently
    fy = viewport_height / (2 * np.tan(fovy_rad / 2))
    fx = fy  # assume square pixels
    cx = viewport_width / 2
    cy = viewport_height / 2

    # Create meshgrid of pixel coordinates
    x = np.arange(viewport_width)
    y = np.arange(viewport_height)
    xv, yv = np.meshgrid(x, y)

    # Compute xyz in camera coordinates
    X = (xv - cx) * z_linear / fx
    Y = (yv - cy) * z_linear / fy
    Z = z_linear

    xyz = np.stack([X, Y, Z], axis=-1)  # shape (H,W,3)
    return xyz

# --------- Convert spherical camera params to position and rotation ----------
def cam_spherical_to_pose(cam):
    az = np.deg2rad(cam.azimuth)
    el = np.deg2rad(cam.elevation)
    dist = cam.distance
    lookat = cam.lookat

    x = dist * np.cos(el) * np.sin(az)
    y = dist * np.cos(el) * np.cos(az)
    z = dist * np.sin(el)
    cam_pos = lookat + np.array([x, y, z])

    z_axis = lookat - cam_pos
    z_axis /= np.linalg.norm(z_axis)

    up = np.array([0, 0, 1])
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)

    cam_mat = np.stack([x_axis, y_axis, z_axis], axis=1)
    return cam_pos, cam_mat

# =========== Initialization ===========
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

# Your exact camera config
cam.azimuth = 89.608063
cam.elevation = -11.588379
cam.distance = 5.0
cam.lookat = np.array([0.0, 0.0, 2.0])

mj.set_mjcb_control(controller)
captured = False

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

    if not captured:
        # Grab RGB image
        img = grab_mujoco_rgb_image(viewport_width, viewport_height)
        img.save("camera_capture.png")
        print("Saved camera_capture.png")

        # Read depth buffer
        depth_buffer = gl.glReadPixels(0, 0, viewport_width, viewport_height,
                                       gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)
        depth_img = np.frombuffer(depth_buffer, dtype=np.float32).reshape(viewport_height, viewport_width)
        depth_img = np.flip(depth_img, axis=0)  # flip vertically

        # Convert depth to xyz in camera frame
        xyz_img = depth_to_xyz(depth_img, cam, viewport_width, viewport_height)

        # Get camera pose from spherical params
        cam_pos, cam_mat = cam_spherical_to_pose(cam)

        # Transform points from camera frame to world frame
        xyz_flat = xyz_img.reshape(-1, 3).T  # shape (3, N)
        xyz_world_flat = cam_mat @ xyz_flat + cam_pos.reshape(3,1)
        xyz_img_world = xyz_world_flat.T.reshape(viewport_height, viewport_width, 3)

        # Optional: save or process xyz_img_world as needed
        print("Captured xyz point cloud in world coordinates.")

        captured = True

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()
