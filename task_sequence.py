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
        self.steps = []
        self.reset()

    # --------------------------
    # Vision updates
    # --------------------------
    def set_boxes_from_vision(self, detected_objects):
        """Assign detected objects to box0, box1, box2, ... dynamically."""
        box_index = 0
        for obj_class, points in detected_objects.items():
            for p in points:
                key = f"box{box_index}"
                self.targets[key] = [p.tolist() if isinstance(p, np.ndarray) else p]
                box_index += 1

        print("[TaskSequence] Updated boxes from vision:")
        for key, pts in self.targets.items():
            for i, p in enumerate(pts):
                print(f"  {key}[{i}] -> {p}")

    def set_targets_from_vision(self, detected_targets):
        """Assign detected objects to target0, target1, ... dynamically."""
        target_index = 0
        for obj_class, points in detected_targets.items():
            for p in points:
                key = f"target_{target_index}"
                self.targets[key] = [p.tolist() if isinstance(p, np.ndarray) else p]
                target_index += 1

        print("[TaskSequence] Updated targets from vision:")
        for key, pts in self.targets.items():
            for i, p in enumerate(pts):
                print(f"  {key}[{i}] -> {p}")

    # --------------------------
    # Generate steps dynamically
    # --------------------------
    def generate_steps(self, ee_pos):
        boxes = [(k, np.array(p[0])) for k, p in self.targets.items() if k.startswith("box")]
        if len(boxes) == 0:
            print("[TaskSequence] No boxes detected yet.")
            return

        # Sort by distance to EE
        boxes.sort(key=lambda x: np.linalg.norm(x[1] - ee_pos))

        steps = []

        for box_key, box_pos in boxes:
            color = box_key.split("_")[1] if "_" in box_key else box_key[-1]
            target_key = f"target_{color}"
            target_pos = np.array(self.targets.get(target_key, [[0,0,0]])[0])

            # --- Pick up box ---
            steps += [
                {"target_id": box_key, "target_type": "live", "offset": np.array([0,0,0.15]),
                 "rot": F.RotX(np.pi/2) @ F.RotY(-np.pi/2), "gripper": 1.0, "wait": 1.0},
                {"target_id": box_key, "target_type": "live", "offset": np.array([0,0,0]),
                 "rot": F.RotX(np.pi/2) @ F.RotY(-np.pi/2), "gripper": 1.0, "wait": 1.0, "pos_tol": "tight"},
                {"target_id": box_key, "target_type": "live", "offset": np.array([0,0,0]),
                 "rot": F.RotX(np.pi/2) @ F.RotY(-np.pi/2), "gripper": 0.0, "wait": 1.0},
                {"target_id": box_key, "target_type": "live", "offset": np.array([0,0,0.15]),
                 "rot": F.RotX(np.pi/2) @ F.RotY(-np.pi/2), "gripper": 0.0, "wait": 1.0},
                # --- Midpoint step (fixed) ---
                {"target_id": None, "target_type": "absolute", "pos": np.array([0, -0.9, 1.25]),
                 "rot": F.RotX(np.pi/2), "gripper": 0.0, "wait": 1, "pos_tol": "loose"}
            ]

            # --- Place box on corresponding target (dynamic live reference) ---
            steps += [
                {"target_id": target_key, "target_type": "live", "offset": np.array([0,0,0.15]),
                 "rot": F.RotX(np.pi/2) @ F.RotY(np.pi/2), "gripper": 0.0, "wait": 1.0},
                {"target_id": target_key, "target_type": "live", "offset": np.array([0,0,0]),
                 "rot": F.RotX(np.pi/2) @ F.RotY(np.pi/2), "gripper": 0.0, "wait": 1.0, "pos_tol": "tight"},
                {"target_id": target_key, "target_type": "live", "offset": np.array([0,0,0]),
                 "rot": F.RotX(np.pi/2) @ F.RotY(np.pi/2), "gripper": 1.0, "wait": 1.0},
                {"target_id": target_key, "target_type": "live", "offset": np.array([0,0,0.15]),
                 "rot": F.RotX(np.pi/2) @ F.RotY(np.pi/2), "gripper": 1.0, "wait": 1.0}
            ]

        # --- Final home position ---
        steps.append({"target_id": None, "target_type": "absolute",
                      "pos": np.array([0, 0, 2.23]),
                      "rot": np.eye(3), "gripper": 0.0, "wait": 0.5, "pos_tol": "loose"})

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

        if self.current_step >= len(self.steps):
            gripper_targets = {"left":0.0, "right":0.0}
            return ee_pos, np.eye(3), gripper_targets, None

        step = self.steps[self.current_step]
        target_type = step.get("target_type", "live")
        offset = step.get("offset", np.zeros(3))

        # --- Determine base position safely ---
        base_pos = ee_pos.copy()  # fallback
        if target_type in ["static", "absolute"]:
            if "pos" in step and step["pos"] is not None:
                base_pos = np.array(step["pos"])
        elif target_type == "live":
            target_id = step.get("target_id")
            if target_id and target_id in self.targets and len(self.targets[target_id]) > 0:
                base_pos = np.array(self.targets[target_id][0])

        target_pos = base_pos + (offset if target_type == "live" else 0)

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
                    if next_step.get("target_type") == "static":
                        self.grasped_pos = base_pos.copy()
                        self.grasped_rot = target_rot.copy()

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

    def reset(self):
        self.current_step = 0
        self.waiting = False
        self.wait_timer = 0.0
        self.grasped_pos = None
        self.grasped_rot = None
