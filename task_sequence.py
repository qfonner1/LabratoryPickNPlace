import numpy as np
import Functions as F 
from path_planner import path_planner


class TaskSequence:
    def __init__(self, model):
        self.grasped_pos = None
        self.grasped_rot = None
        self.targets = {} 
        self.current_step = 0
        self.waiting = False
        self.wait_timer = 0.0
        self.active = True  
        self.completed = False
        self.steps = []
        self.reset()
        self.substeps = []   
        self.substep_idx = 0
        self.last_target_pos = None
        self.last_target_rot = None
        self.last_gripper_open = 0.0

    # --------------------------
    # Vision updates
    # --------------------------
    def set_boxes_from_vision(self, detected_objects):
        """
        Store boxes with their color name intact, e.g. 'red_box', 'blue_box'.
        """
        for cname, points in detected_objects.items():
            if not points:
                continue
            self.targets[cname] = [p.tolist() if isinstance(p, np.ndarray) else p for p in points]

        print("[Task Sequence] Updated boxes from vision:")
        for key, pts in self.targets.items():
            for i, p in enumerate(pts):
                print(f"[Task Sequence]  {key}[{i}] -> {p}")

    def set_targets_from_vision(self, detected_targets):
        """
        Store targets with color name intact, e.g. 'red_target', 'blue_target'.
        """
        for cname, points in detected_targets.items():
            if not points:
                continue
            # Ensure target keys have 'target_' prefix
            color_name = cname.replace("_box", "")  # unify naming
            key = f"{color_name}_target"
            self.targets[key] = [p.tolist() if isinstance(p, np.ndarray) else p for p in points]

        print("[Task Sequence] Updated targets from vision:")
        for key, pts in self.targets.items():
            for i, p in enumerate(pts):
                print(f"  {key}[{i}] -> {p}")

    # --------------------------
    # Generate steps dynamically
    # --------------------------
    def generate_steps(self, ee_pos):
        boxes = [(k, np.array(p[0])) for k, p in self.targets.items() if k.endswith("_box")]
        if len(boxes) == 0:
            print("[Task Sequence] No boxes detected yet.")
            return

        # Sort boxes by distance to end-effector
        boxes.sort(key=lambda x: np.linalg.norm(x[1] - ee_pos))

        steps = []

        for box_key, box_pos in boxes:
            color = box_key.replace("_box", "")
            target_key = f"{color}_target"

            # --- Only proceed if matching target exists ---
            if target_key not in self.targets:
                print(f"[Task Sequence] ⚠️ No target for {box_key} → skipping.")
                continue

            # --- Pick up box ---
            steps += [
                {"target_id": box_key, "offset": np.array([0.05,0,0.10]), "rot": F.RotX(np.pi/2) @ F.RotY(-np.pi/2), "gripper": 1.0, "wait": 1.0, "pos_tol": "loose"},
                {"target_id": box_key, "offset": np.array([0,0,0.10]), "rot": F.RotX(np.pi/2) @ F.RotY(-np.pi/2), "gripper": 1.0, "wait": 1.0,  "pos_tol": "tight"},
                {"target_id": box_key, "offset": np.array([0,0,0.01]), "rot": F.RotX(np.pi/2) @ F.RotY(-np.pi/2), "gripper": 1.0, "wait": 1.0, "pos_tol": "tight"},
                {"target_id": box_key, "offset": np.array([0,0,0.01]), "rot": F.RotX(np.pi/2) @ F.RotY(-np.pi/2), "gripper": 0.0, "wait": 1.0},
                {"target_id": box_key, "offset": np.array([0,0,0.10]), "rot": F.RotX(np.pi/2) @ F.RotY(-np.pi/2), "gripper": 0.0, "wait": 1.0},
            ]

            # --- Place box on corresponding target (dynamic live reference) ---
            steps += [
                {"target_id": target_key, "offset": np.array([-0.05,0,0.10]), "rot": F.RotX(np.pi/2) @ F.RotY(np.pi/2), "gripper": 0.0, "wait": 1.0, "pos_tol": "loose"},
                {"target_id": target_key, "offset": np.array([0,0,0.10]), "rot": F.RotX(np.pi/2) @ F.RotY(np.pi/2), "gripper": 0.0, "wait": 1.0, "pos_tol": "tight"},
                {"target_id": target_key, "offset": np.array([0,0,0.02]), "rot": F.RotX(np.pi/2) @ F.RotY(np.pi/2), "gripper": 0.0, "wait": 1.0, "pos_tol": "tight"},
                {"target_id": target_key, "offset": np.array([0,0,0.02]), "rot": F.RotX(np.pi/2) @ F.RotY(np.pi/2), "gripper": 1.0, "wait": 1.0},
                {"target_id": target_key, "offset": np.array([-0.1,0,0.02]), "rot": F.RotX(np.pi/2) @ F.RotY(np.pi/2), "gripper": 1.0, "wait": 1.0}
            ]

        self.steps = steps
        self.current_step = 0
        self.waiting = False
        self.wait_timer = 0.0
        print(f"[Task Sequence] Steps regenerated for {len(boxes)} boxes with color matching.")

    # --------------------------
    # New: Plan and insert substeps (waypoints)
    # --------------------------
    def _get_step_world_pos_rot(self, step):
        """Return (pos, rot) in world coords for a step dict."""
        if step.get("target_id") is None:
            pos = np.array(step.get("pos", np.zeros(3)))
        else:
            target_id = step["target_id"]
            if target_id in self.targets and len(self.targets[target_id]) > 0:
                pos = np.array(self.targets[target_id][0]) + step.get("offset", np.zeros(3))
            else:
                # fallback
                pos = np.array(step.get("pos", np.zeros(3)))
        rot = step.get("rot", np.eye(3))
        return pos, rot
    

    def _create_substeps_between(self, start_pos, start_rot, end_pos, end_rot, end_gripper, end_pos_tol, max_waypoints):
        """
        Create intermediate substeps between start and end positions/rotations.
        start_rot and end_rot are 3x3 rotation matrices.
        """

        # Convert start/end positions to lists for the planner
        start_list = start_pos.tolist() if isinstance(start_pos, np.ndarray) else list(start_pos)
        end_list   = end_pos.tolist()   if isinstance(end_pos, np.ndarray)   else list(end_pos)

        # Get path positions from planner
        positions = path_planner(start=start_list, goal=end_list, max_retries=5, show_animation=False)
        positions = np.array(positions)

        if len(positions) < 2:
            print("[Task Sequence] Warning! Path has too few points; skipping interpolation.")
            positions = [start_pos, end_pos]


        # Remove last five positions
        if len(positions) > 5:
            positions = positions[:-5]  # drop the last two points

        # Subsample positions to respect max_waypoints
        if len(positions) > max_waypoints:
            indices = np.linspace(0, len(positions) - 1, max_waypoints, dtype=int)
            positions = positions[indices]

        # Convert start/end rotations to quaternions
        q_start = F.rotmat_to_quat(start_rot)
        q_end   = F.rotmat_to_quat(end_rot)

        # Compute tangent from start to a few waypoints ahead
        look_ahead = min(5, len(positions)-1)
        tangent = positions[look_ahead] - positions[0]
        tangent /= np.linalg.norm(tangent)

        # Get start end-effector x-axis
        x_axis_eef = start_rot[:, 0]  # start_rot is 3x3 rotation

        # Compute angle between tangent and x-axis
        cos_angle = np.dot(tangent, x_axis_eef)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

        # Determine long_path
        long_path = angle < (np.pi / 2)

        print(f"[Task Sequence] Initial tangent angle: {np.degrees(angle):.2f} deg, long_path: {long_path}")

        # Now use this for all substeps
        substeps = []
        for i, pos in enumerate(positions):
            t = i / (len(positions)-1)
            q_interp = F.quat_slerp(q_start, q_end, t, long_path=long_path)
            R_interp = F.quat_to_rotmat(q_interp)

            substeps.append({
                "target_id": None,
                "pos": pos,
                "rot": R_interp,
                "gripper": end_gripper,
                "wait": 1.0,
                "pos_tol": end_pos_tol
            })


        return substeps


    # --------------------------
    # Get next target for robot
    # --------------------------
    def get_target(self, model, data, ee_pos, ee_rot=None):
        if ee_rot is None:
            ee_rot = np.eye(3)

        # Early return if inactive or completed
        if not self.active:
            gripper_targets = {"left": 0.04*self.last_gripper_open, "right": -0.04*self.last_gripper_open}
            if self.last_target_pos is not None and self.last_target_rot is not None:
                return self.last_target_pos, self.last_target_rot, gripper_targets, self.last_gripper_open
            else:
                return ee_pos, ee_rot, gripper_targets, None

        # Determine current step (substep or main)
        if self.substeps and self.substep_idx < len(self.substeps):
            step = self.substeps[self.substep_idx]
            target_pos = np.array(step["pos"])
            target_rot = np.array(step.get("rot", np.eye(3)))
            
            gripper_open = step.get("gripper", 0.0)
            gripper_targets = {"left": 0.04 * gripper_open, "right": -0.04 * gripper_open}

            dist = np.linalg.norm(ee_pos - target_pos)
            tol_type = step.get("pos_tol")
            pos_tol = 0.01 if tol_type == "tight" else 0.2 if tol_type == "loose" else 0.03

            if dist < pos_tol:
                self.substep_idx += 1
                if self.substep_idx >= len(self.substeps):
                    self.advance_step()
                    self.substeps = []
                    self.substep_idx = 0

            print(f"[Task Sequence] Substep {self.substep_idx}/{len(self.substeps)} | Dist: {dist:.3f}", flush=True)
            return target_pos, target_rot, gripper_targets, gripper_open

        # Main step handling
        step = self.steps[self.current_step]
        target_id = step.get("target_id")
        offset = step.get("offset", np.zeros(3, dtype=np.float64))

        # Compute base position
        if target_id is None and "pos" in step:
            base_pos = np.array(step["pos"], dtype=np.float64)
        elif target_id in self.targets and len(self.targets[target_id]) > 0:
            base_pos = np.array(self.targets[target_id][0], dtype=np.float64)
        else:
            base_pos = ee_pos.copy()

        target_pos = base_pos + (offset if target_id is not None else 0)
        target_rot = step.get("rot", np.eye(3)).astype(np.float64)

        # distance and rotation check
        dist = np.linalg.norm(ee_pos - target_pos)
        R_diff = ee_rot.T @ target_rot
        angle_deg = np.degrees(np.arccos(np.clip((np.trace(R_diff)-1)/2, -1.0, 1.0)))

        tol_type = step.get("pos_tol", "default")
        pos_tol = 0.01 if tol_type == "tight" else 0.1 if tol_type == "loose" else 0.03

        # If within tolerance, plan substeps for next step
        if dist < pos_tol and angle_deg < 5.0:
            if self.current_step + 1 < len(self.steps):
                next_step = self.steps[self.current_step + 1]
                next_pos, next_rot = self._get_step_world_pos_rot(next_step)
                next_gripper = next_step.get("gripper", 0.0)
                end_pos_tol = next_step.get("pos_tol")
                self.substeps = self._create_substeps_between(target_pos, target_rot, next_pos, next_rot, next_gripper, end_pos_tol, max_waypoints=20)
                self.substep_idx = 0
                if not self.substeps:
                    # fallback if planning fails
                    if step.get("wait",0.0) > 0:
                        self.waiting = True
                        self.wait_timer = 0.0
                    else:
                        self.advance_step()
            else:
                if step.get("wait", 0.0) > 0:
                    self.waiting = True
                    self.wait_timer = 0.0
                    # ✅ If this is the last step, complete after waiting
                    if self.current_step == len(self.steps) - 1:
                        print("\n[Task Sequence] Tasks complete! 🎉\n")
                        self.completed = True
                        self.active = False
                else:
                    self.advance_step()

        # Gripper
        gripper_open = step.get("gripper", 0.0)
        gripper_targets = {"left": 0.04*gripper_open, "right": -0.04*gripper_open}

        # Store last pose so robot can hold it later
        self.last_target_pos = target_pos.copy()
        self.last_target_rot = target_rot.copy()
        self.last_gripper_open = gripper_open

        # Debug
        if not self.completed:
            print(f"[Task Sequence] Step {self.current_step} | EE Pos: [{', '.join(f'{x:.3f}' for x in ee_pos)}] "
                f"| Target Pos: [{', '.join(f'{x:.3f}' for x in target_pos)}] "
                f"| Dist: {dist:.3f} | Angle Diff: {angle_deg:.2f}°", flush=True)

        return target_pos, target_rot, gripper_targets, gripper_open


    # --------------------------
    # Helpers
    # --------------------------
    def advance_step(self):
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1

    def reset(self):
        self.current_step = 0
        self.waiting = False
        self.wait_timer = 0.0
        self.grasped_pos = None
        self.grasped_rot = None
        self.substeps = []
        self.substep_idx = 0
        self.completed = False
        self.active = True
        self.last_target_pos = None
        self.last_target_rot = None
        self.last_gripper_open = 0.0
