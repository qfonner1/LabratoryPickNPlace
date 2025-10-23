import numpy as np
import mujoco as mj
from mujoco.glfw import glfw
import OpenGL.GL as gl
from PIL import Image, ImageDraw
import cv2


def object_detection(xml_path, cam_name):
    # --------- User config ---------
    XML_PATH = xml_path
    CAM_NAME = cam_name
    WINDOW_SIZE = (1200, 900)
    USE_CV_CONVENTION = True

    Z_TABLE_FALLBACK = 0.0    # fallback table height (m)
    Z_ABOVE_TABLE = 0.10        # offset above table (m)
    USE_HOMOGRAPHY_CALIBRATION = True   # <-- enable table-based mapping

    COLOR_CLASSES = {
        "red_box":   {"ref_rgb": (255, 0, 0),   "tol": 0},
        "green_box": {"ref_rgb": (0, 255, 0),   "tol": 0},
        "blue_box":  {"ref_rgb": (0, 0, 255),   "tol": 0},
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
        """
        Create mask by comparing image to a reference RGB color in HSV space,
        allowing hue tolerance and thresholds on saturation and value for darker pixels.

        Args:
            rgb: (H, W, 3) uint8 image in RGB.
            ref_rgb: (3,) reference RGB color tuple (0-255).
            hue_tol: int tolerance around hue (in degrees, 0-180 in OpenCV).
            sat_min: int minimum saturation (0-255).
            val_min: int minimum value/brightness (0-255).

        Returns:
            mask: (H, W) boolean mask where pixels match color within tolerance.
        """
        # Convert image to HSV
        hsv_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        
        # Convert reference color to HSV (single pixel)
        ref_bgr = np.uint8([[ref_rgb[::-1]]])  # RGB->BGR for cv2
        ref_hsv = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2HSV)[0,0]

        h_ref, s_ref, v_ref = ref_hsv

        # Hue lower and upper bounds, wrapping around 180 if needed
        lower_h = (h_ref - hue_tol) % 180
        upper_h = (h_ref + hue_tol) % 180

        if lower_h <= upper_h:
            hue_mask = (hsv_img[:,:,0] >= lower_h) & (hsv_img[:,:,0] <= upper_h)
        else:
            # Wrap-around case
            hue_mask = (hsv_img[:,:,0] >= lower_h) | (hsv_img[:,:,0] <= upper_h)

        sat_mask = hsv_img[:,:,1] >= sat_min
        val_mask = hsv_img[:,:,2] >= val_min

        mask = hue_mask & sat_mask & val_mask
        return mask



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
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
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

        depth_bytes = gl.glReadPixels(0, 0, fb_w, fb_h, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)
        depth = np.frombuffer(depth_bytes, dtype=np.float32).reshape(fb_h, fb_w)
        depth = np.flip(depth, axis=0)

        znear, zfar = model.vis.map.znear, model.vis.map.zfar
        linear_depth = 2.0 * znear * zfar / (zfar + znear - (2.0 * depth - 1.0) * (zfar - znear))

        fovy_deg = float(model.cam_fovy[cam_id])
        cam_pos = data.cam_xpos[cam_id].copy()
        cam_xmat = data.cam_xmat[cam_id].copy()

        glfw.destroy_window(window)
        glfw.terminate()
        return rgb, linear_depth, fovy_deg, cam_pos, cam_xmat

    def order_corners(corners):
        # corners is (4, 2)
        # Step 1: sort by y (vertical)
        sorted_by_y = corners[np.argsort(corners[:,1]), :]
        top_two = sorted_by_y[:2, :]
        bottom_two = sorted_by_y[2:, :]

        # Step 2: among top two, sort by x (horizontal)
        top_left, top_right = top_two[np.argsort(top_two[:,0]), :]
        # Step 3: among bottom two, sort by x (horizontal)
        bottom_left, bottom_right = bottom_two[np.argsort(bottom_two[:,0]), :]

        # Return in order: bottom-left, bottom-right, top-right, top-left
        return np.array([bottom_left, bottom_right, top_right, top_left], dtype=np.float32)


    def calibrate_homography_from_table(rgb, model, data):
        pos = np.array([-0.9, 0.0])
        size = np.array([0.2, 0.4])
        corners_world = np.array([
            [pos[0]-size[0], pos[1]-size[1]],
            [pos[0]+size[0], pos[1]-size[1]],
            [pos[0]+size[0], pos[1]+size[1]],
            [pos[0]-size[0], pos[1]+size[1]],
        ], dtype=np.float32)

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)  # lower threshold

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        cv2.imwrite("table_mask.png", mask)  # save to check visually

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("⚠️  Table contour not found — homography disabled.")
            return None

        largest = max(contours, key=cv2.contourArea)
        print(f"Largest contour area: {cv2.contourArea(largest)}")

        epsilon = 0.05 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)

        if len(approx) != 4:
            print(f"⚠️  Table contour not quadrilateral (found {len(approx)} corners) — trying minAreaRect fallback.")
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            approx = np.int0(box)
            if len(approx) != 4:
                print("⚠️  Fallback contour also not quadrilateral — homography disabled.")
                return None

        corners_px = np.array([p[0] if p.shape == (1,2) else p for p in approx], dtype=np.float32)

        corners_px = order_corners(corners_px)


        H, _ = cv2.findHomography(corners_px, corners_world)
        for i, pt in enumerate(corners_px):
            uv1 = np.array([pt[0], pt[1], 1.0])
            XY1 = H @ uv1
            XY1 /= XY1[2]
            print(f"Image corner {i}: pixel {pt} → world {XY1[:2]}")

        print("✅  Homography calibration successful.")
        return H



    def draw_annotations(rgb, original_detections, shifted_detections, color_classes, radius=8):
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        for cname, cents in original_detections.items():
            ref = color_classes[cname]["ref_rgb"]
            for i, (u, v) in enumerate(cents):
                # Draw original centroid in class color, bigger circle
                draw.ellipse((u-radius, v-radius, u+radius, v+radius), outline=ref, width=3)
                draw.text((u+radius+2, v-radius-2), f"{cname}[{i}]", fill=ref)

        for cname, cents in shifted_detections.items():
            ref = color_classes[cname]["ref_rgb"]
            for i, (u, v, _) in enumerate(cents):
                # Draw shifted centroid in class color, smaller circle, dashed outline (simulate)
                r = radius // 2
                # Different style: here just thinner and smaller circle in same color
                draw.ellipse((u-r, v-r, u+r, v+r), outline=ref, width=1)
                # Optionally add a marker (e.g., a dot) inside
                draw.point((u, v), fill=ref)

        return np.array(img)


    print("Loading model...")
    model = mj.MjModel.from_xml_path(XML_PATH)
    data = mj.MjData(model)

    z_table = estimate_table_z_from_known_geom(model, data)
    if z_table is None:
        z_table = Z_TABLE_FALLBACK
        print(f"Table Height Estimation Failed!")
    else:
        print(f"Estimated table height z={z_table:.3f} m")

    z_target = z_table + Z_ABOVE_TABLE
    print(f"Using centroid Z = {z_target:.3f} m")

    print("Rendering overhead image and depth map...")
    rgb, depth_map, fovy_deg, cam_pos, cam_xmat = render_and_capture(model, data, CAM_NAME, WINDOW_SIZE)
    Image.fromarray(rgb).save("overhead_rgb.png")

    H_homography = calibrate_homography_from_table(rgb, model, data) if USE_HOMOGRAPHY_CALIBRATION else None

    print("Detecting colors...")
    detections = detect_colors_centroids(rgb, COLOR_CLASSES, grid=GRID, min_pixels=MIN_PIXELS)
    results_by_color_shifted = {}
    results_by_color_original = detections
    H, W, _ = rgb.shape

    for cname, cents in detections.items():
        pts_world_shifted = []
        for (u, v) in cents:
            if H_homography is not None:
                uv1 = np.array([u, v, 1.0])
                XY1 = H_homography @ uv1
                XY1 /= XY1[2]
                X, Y = XY1[0], XY1[1]
                P = np.array([X, Y, z_target])
            else:
                u_i, v_i = int(round(u)), int(round(v))
                if 0 <= v_i < depth_map.shape[0] and 0 <= u_i < depth_map.shape[1]:
                    depth_m = float(depth_map[v_i, u_i])
                    if np.isfinite(depth_m) and depth_m > 0:
                        P = pixel_to_world_from_depth(
                            u, v, depth_m, W, H, fovy_deg,
                            cam_pos=cam_pos, cam_xmat=cam_xmat, use_cv=USE_CV_CONVENTION)
                        P[2] = z_target
                    else:
                        continue  # skip invalid depth point
                else:
                    continue  # skip out-of-bounds

            table_center = np.array([-0.9, 0.0])  # your table center (x,y)
            weight = 0.1  # tune this between 0 and 1 as you like

            P_xy = P[:2]
            shifted_xy = (1 - weight) * P_xy + weight * table_center
            P_shifted = np.array([shifted_xy[0], shifted_xy[1], P[2]])

            pts_world_shifted.append(P_shifted)

        results_by_color_shifted[cname] = pts_world_shifted

    for cname, pts in results_by_color_shifted.items():
        print(f"\n{cname}: found {len(pts)} object(s)")
        for i, P in enumerate(pts):
            print(f"  {cname}[{i}]  X={P[0]:.4f}, Y={P[1]:.4f}, Z={P[2]:.4f}")

    annotated = draw_annotations(rgb, results_by_color_original, results_by_color_shifted, COLOR_CLASSES)
    Image.fromarray(annotated).save("overhead_rgb_annotated.png")
    print("Saved: overhead_rgb_annotated.png\nDone.")

    return results_by_color_shifted

