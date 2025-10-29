import numpy as np
import Functions as F 

class TaskSequence:
    def __init__(self, model):
        self.grasped_pos = None
        self.grasped_rot = None
        self.targets = {} 
        self.current_step = 0
        self.waiting = False
        self.wait_timer = 0.0
        self.active = True  # Only process steps when active
        self.completed = False
        self.steps = []
        self.reset()

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

        print("[TaskSequence] Updated boxes from vision:")
        for key, pts in self.targets.items():
            for i, p in enumerate(pts):
                print(f"  {key}[{i}] -> {p}")

        print("[TaskSequence] Updated boxes from vision:")
        for key, pts in self.targets.items():
            for i, p in enumerate(pts):
                print(f"  {key}[{i}] -> {p}")

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

        print("[TaskSequence] Updated targets from vision:")
        for key, pts in self.targets.items():
            for i, p in enumerate(pts):
                print(f"  {key}[{i}] -> {p}")

    # --------------------------
    # Generate steps dynamically
    # --------------------------
    def generate_steps(self, ee_pos):
        boxes = [(k, np.array(p[0])) for k, p in self.targets.items() if k.endswith("_box")]
        if len(boxes) == 0:
            print("[TaskSequence] No boxes detected yet.")
            return

        # Sort boxes by distance to end-effector
        boxes.sort(key=lambda x: np.linalg.norm(x[1] - ee_pos))

        steps = []

        for box_key, box_pos in boxes:
            color = box_key.replace("_box", "")
            target_key = f"{color}_target"

            # --- Only proceed if matching target exists ---
            if target_key not in self.targets:
                print(f"[TaskSequence] ⚠️ No target for {box_key} → skipping.")
                continue

            # --- Pick up box ---
            steps += [
                {"target_id": box_key, "offset": np.array([0,0,0.15]), "rot": F.RotX(np.pi/2) @ F.RotY(-np.pi/2), "gripper": 1.0, "wait": 0.5},
                {"target_id": box_key, "offset": np.array([0,0,0]), "rot": F.RotX(np.pi/2) @ F.RotY(-np.pi/2), "gripper": 1.0, "wait": 0.5, "pos_tol": "tight"},
                {"target_id": box_key, "offset": np.array([0,0,0]), "rot": F.RotX(np.pi/2) @ F.RotY(-np.pi/2), "gripper": 0.0, "wait": 0.5},
                {"target_id": box_key, "offset": np.array([0,0,0.15]), "rot": F.RotX(np.pi/2) @ F.RotY(-np.pi/2), "gripper": 0.0, "wait": 0.5},
                # --- Midpoint step (fixed) ---
                #{"target_id": None, "pos": np.array([0, -0.9, 1.25]), "rot": F.RotX(np.pi/2), "gripper": 0.0, "wait": 0.0, "pos_tol": "loose"}
            ]

            # --- Place box on corresponding target (dynamic live reference) ---
            steps += [
                {"target_id": target_key, "offset": np.array([0,0,0.15]), "rot": F.RotX(np.pi/2) @ F.RotY(np.pi/2), "gripper": 0.0, "wait": 0.5},
                {"target_id": target_key, "offset": np.array([0,0,0.01]), "rot": F.RotX(np.pi/2) @ F.RotY(np.pi/2), "gripper": 0.0, "wait": 0.5, "pos_tol": "tight"},
                {"target_id": target_key, "offset": np.array([0,0,0.01]), "rot": F.RotX(np.pi/2) @ F.RotY(np.pi/2), "gripper": 1.0, "wait": 0.5},
                {"target_id": target_key, "offset": np.array([0,0,0.15]), "rot": F.RotX(np.pi/2) @ F.RotY(np.pi/2), "gripper": 1.0, "wait": 0.5}
            ]

        # --- Final home position ---
        steps.append({"target_id": None, "pos": np.array([0, 0, 2.23]), "rot": np.eye(3), "gripper": 0.0, "wait": 0.0, "pos_tol": "loose"})

        self.steps = steps
        self.current_step = 0
        self.waiting = False
        self.wait_timer = 0.0
        print(f"[TaskSequence] Steps regenerated for {len(boxes)} boxes with color matching.")

    # --------------------------
    # Get next target for robot
    # --------------------------
    def get_target(self, model, data, ee_pos, ee_rot=None):
        if ee_rot is None:
            ee_rot = np.eye(3)

        if not self.active or self.completed:
            # Stop all motion and printing once sequence is complete
            if self.completed:
                # Suppress final debug print
                return ee_pos, np.eye(3), {"left": 0.0, "right": 0.0}, None
            gripper_targets = {"left":0.0, "right":0.0}
            return ee_pos, np.eye(3), gripper_targets, None

        step = self.steps[self.current_step]
        target_id = step.get("target_id")
        offset = step.get("offset", np.zeros(3))

        # --- Determine base position safely ---
        base_pos = ee_pos.copy()  # fallback
        if target_id is None:
            if "pos" in step and step["pos"] is not None:
                base_pos = np.array(step["pos"])
        elif target_id is not None:
            if target_id and target_id in self.targets and len(self.targets[target_id]) > 0:
                base_pos = np.array(self.targets[target_id][0])

        target_pos = base_pos + (offset if target_id is not None else 0)

        # Rotation
        target_rot = step.get("rot", np.eye(3)).copy() if step.get("rot") is not None else np.eye(3)

        # --- Wait / advance logic ---
        if self.waiting:
            self.wait_timer += model.opt.timestep
            if self.wait_timer >= step.get("wait", 0.0):
                self.waiting = False
                self.wait_timer = 0.0
                self.advance_step()
        else:
            dist = np.linalg.norm(ee_pos - target_pos)
            R_diff = ee_rot.T @ target_rot
            angle_rad = np.arccos(np.clip((np.trace(R_diff)-1)/2, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)

            tol_type = step.get("pos_tol")
            pos_tol = 0.01 if tol_type=="tight" else 0.1 if tol_type=="loose" else 0.03

            if dist < pos_tol and angle_deg < 5.0:
                if self.current_step + 1 < len(self.steps):
                    next_step = self.steps[self.current_step + 1]
                if step.get("wait",0.0) > 0:
                    self.waiting = True
                    self.wait_timer = 0.0
                else:
                    self.advance_step()

        # --- Gripper control ---
        gripper_open = step.get("gripper", 0.0)
        gripper_targets = {"left": 0.04*gripper_open, "right": -0.04*gripper_open}

        # --- Debug ---
        dist = np.linalg.norm(ee_pos - target_pos)
        R_diff = ee_rot.T @ target_rot
        angle_rad = np.arccos(np.clip((np.trace(R_diff)-1)/2, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        if not self.completed:
            print(f"Step {self.current_step} | EE Pos: [{', '.join(f'{x:.3f}' for x in ee_pos)}] "
                f"| Target Pos: [{', '.join(f'{x:.3f}' for x in target_pos)}] "
                f"| Dist: {dist:.3f} | Angle Diff: {angle_deg:.2f}°", flush=True)

        return target_pos, target_rot, gripper_targets, gripper_open

    # --------------------------
    # Helpers
    # --------------------------
    def advance_step(self):
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
        else:
            if not self.completed:
                print("\n✅ [TaskSequence] All steps completed successfully! 🎉\n")
                self.completed = True
            self.active = False


    def reset(self):
        self.current_step = 0
        self.waiting = False
        self.wait_timer = 0.0
        self.grasped_pos = None
        self.grasped_rot = None
