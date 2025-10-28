import numpy as np
import mujoco as mj
from mujoco.glfw import glfw
import OpenGL.GL as gl
from PIL import Image, ImageDraw
import cv2


def target_detection(xml_path, cam_name):
    """Detect red/green/blue targets in an overhead Mujoco scene and return world coordinates."""
    
    # --------- User config ---------
    XML_PATH = xml_path
    CAM_NAME = cam_name
    WINDOW_SIZE = (1200, 900)
    USE_CV_CONVENTION = True

    Z_TABLE_FALLBACK = 0.0
    Z_ABOVE_TABLE = 0.10
    USE_HOMOGRAPHY_CALIBRATION = True

    COLOR_CLASSES = {
        "red_target":   {"ref_rgb": (255, 0, 0),   "tol": 0},
        "green_target": {"ref_rgb": (0, 255, 0),   "tol": 0},
        "blue_target":  {"ref_rgb": (0, 0, 255),   "tol": 0},
    }

    GRID = 48
    MIN_PIXELS = 80
    # --------------------------------

    def intrinsics_from_fovy(fovy_deg, width, height):
        H, W = height, width
        fovy_rad = np.deg2rad(float(fovy_deg))
        fy = H / (2.0 * np.tan(fovy_rad / 2.0))
        fx = fy
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0
        return fx, fy, cx, cy

    def pixel_to_world_from_depth(u, v, depth_m, W, H, fovy_deg, cam_pos, cam_xmat, use_cv=True):
        fx, fy, cx, cy = intrinsics_from_fovy(fovy_deg, W, H)
        x = (u - cx) / fx
        y = (v - cy) / fy
        z = 1.0
        d_cam_cv = np.array([x, y, z], dtype=np.float32)
        d_cam_cv /= np.linalg.norm(d_cam_cv)
        if use_cv:
            d_cam_mj = np.array([d_cam_cv[0], -d_cam_cv[1], -d_cam_cv[2]], dtype=np.float32)
        else:
            d_cam_mj = np.array([x, y, -1.0], dtype=np.float32)
        R = cam_xmat.reshape(3, 3)
        p_world = cam_pos + (R.T @ d_cam_mj) * depth_m
        return p_world

    def mask_from_ref_hsv(rgb, ref_rgb, hue_tol=15, sat_min=10, val_min=10):
        hsv_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        ref_bgr = np.uint8([[ref_rgb[::-1]]])
        ref_hsv = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2HSV)[0, 0]
        h_ref, s_ref, v_ref = ref_hsv
        lower_h = (h_ref - hue_tol) % 180
        upper_h = (h_ref + hue_tol) % 180
        if lower_h <= upper_h:
            hue_mask = (hsv_img[:, :, 0] >= lower_h) & (hsv_img[:, :, 0] <= upper_h)
        else:
            hue_mask = (hsv_img[:, :, 0] >= lower_h) | (hsv_img[:, :, 0] <= upper_h)
        sat_mask = hsv_img[:, :, 1] >= sat_min
        val_mask = hsv_img[:, :, 2] >= val_min
        return hue_mask & sat_mask & val_mask

    def centroids_from_mask_grid(mask, grid=48, min_pixels=80):
        H, W = mask.shape
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return []
        gx = (xs * grid) // W
        gy = (ys * grid) // H
        gx = gx.clip(0, grid - 1)
        gy = gy.clip(0, grid - 1)
        bin_keys = (gy * grid + gx).astype(np.int32)
        order = np.argsort(bin_keys)
        bin_keys_sorted = bin_keys[order]
        unique_bins, first_idx, counts = np.unique(bin_keys_sorted, return_index=True, return_counts=True)
        keep_mask = counts >= int(min_pixels)
        if not np.any(keep_mask):
            return [(float(xs.mean()), float(ys.mean()))]
        kept_bins = unique_bins[keep_mask]
        kept_first = first_idx[keep_mask]
        kept_counts = counts[keep_mask]
        bin_to_slice = {int(b): (int(f), int(f + c)) for b, f, c in zip(kept_bins, kept_first, kept_counts)}
        cell_set = set([(int(b) // grid, int(b) % grid) for b in kept_bins])
        visited, components = set(), []
        for cell in cell_set:
            if cell in visited:
                continue
            comp, queue = [], [cell]
            visited.add(cell)
            while queue:
                cy, cx = queue.pop()
                comp.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < grid and 0 <= nx < grid and (ny, nx) in cell_set and (ny, nx) not in visited:
                        visited.add((ny, nx))
                        queue.append((ny, nx))
            components.append(comp)
        centroids = []
        for comp in components:
            comp_indices = []
            for (cy, cx) in comp:
                b = cy * grid + cx
                if b in bin_to_slice:
                    lo, hi = bin_to_slice[b]
                    comp_indices.append(order[lo:hi])
            if comp_indices:
                comp_idx = np.concatenate(comp_indices, axis=0)
                u = float(xs[comp_idx].mean())
                v = float(ys[comp_idx].mean())
                centroids.append((u, v))
        return centroids

    def detect_colors_centroids(rgb, color_classes, grid=48, min_pixels=80):
        results = {}
        for name, spec in color_classes.items():
            ref, tol = spec["ref_rgb"], int(spec.get("tol", 60))
            mask = mask_from_ref_hsv(rgb, ref, tol)
            cents = centroids_from_mask_grid(mask, grid=grid, min_pixels=min_pixels)
            results[name] = cents
        return results

    def estimate_table_z_from_known_geom(model, data):
        candidates = ["box_geom", "box_geom2", "box_geom3"]
        for gname in candidates:
            gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, gname)
            if gid >= 0:
                mj.mj_forward(model, data)
                z_center = float(data.geom_xpos[gid, 2])
                z_half = float(model.geom_size[gid, 2])
                return z_center - z_half
        return None

    def render_and_capture(model, data, cam_name, window_size):
        W, H = window_size
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(W, H, "Overhead Capture", None, None)
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        model.vis.map.znear, model.vis.map.zfar = 0.055, 5.0
        cam = mj.MjvCamera()
        opt = mj.MjvOption()
        mj.mjv_defaultCamera(cam)
        mj.mjv_defaultOption(opt)
        scene = mj.MjvScene(model, maxgeom=10000)
        context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
        cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id < 0:
            raise ValueError(f"Camera '{cam_name}' not found")
        cam.type = mj.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = cam_id
        mj.mj_forward(model, data)
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, fb_w, fb_h)
        mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        rgba_bytes = gl.glReadPixels(0, 0, fb_w, fb_h, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
        rgba = np.frombuffer(rgba_bytes, dtype=np.uint8).reshape(fb_h, fb_w, 4)
        rgb = np.flip(rgba[:, :, :3], axis=0).copy()
        glfw.destroy_window(window)
        glfw.terminate()
        return rgb

    # ---------- MAIN PIPELINE ----------
    model = mj.MjModel.from_xml_path(XML_PATH)
    data = mj.MjData(model)

    z_table = estimate_table_z_from_known_geom(model, data)
    if z_table is None:
        z_table = Z_TABLE_FALLBACK
    z_target = z_table + Z_ABOVE_TABLE

    rgb = render_and_capture(model, data, CAM_NAME, WINDOW_SIZE)
    detections = detect_colors_centroids(rgb, COLOR_CLASSES, grid=GRID, min_pixels=MIN_PIXELS)

    results_by_color = {}
    for cname, cents in detections.items():
        results_by_color[cname] = [
            np.array([u, v, z_target]) for (u, v) in cents
        ]

    # Annotate for visualization
    annotated = Image.fromarray(rgb)
    draw = ImageDraw.Draw(annotated)
    for cname, cents in detections.items():
        color = COLOR_CLASSES[cname]["ref_rgb"]
        for i, (u, v) in enumerate(cents):
            draw.ellipse((u-6, v-6, u+6, v+6), outline=color, width=3)
            draw.text((u+8, v-8), f"{cname}[{i}]", fill=color)
    annotated = np.array(annotated)

    return results_by_color, annotated
