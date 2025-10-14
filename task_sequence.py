import numpy as np
import Functions as F  # Ensure RotX, RotY, RotZ, axang_to_rot are defined

class TaskSequence:
    def __init__(self, model):
        self.id_map = {
            "box": model.body("obj_box_06").id,
        }

        self.grasped_pos = None
        self.grasped_rot = None

        self.steps = [
            {"target_id": "box", "target_type": "live", "offset": np.array([0.0, 0.0, 0.15]), "gripper": 1.0, "wait": 0.5},
            {"target_id": "box", "target_type": "live", "offset": np.array([0.0, 0.0, 0.0]), "gripper": 1.0, "wait": 0.5},
            {"target_id": "box", "target_type": "live", "offset": np.array([0.0, 0.0, 0.0]), "gripper": 0.0, "wait": 1.0},
            {"target_id": "box", "target_type": "static", "offset": np.array([0.0, 0.0, 0.15]), "gripper": 0.0, "wait": 1.0},
            {"target_type": "arc_to_point", "pos": np.array([0.9, 0.0, 1.7])},
        ]

        self.current_step = 0
        self.waiting = False
        self.wait_timer = 0.0

        # Arc path state
        self.path_start = None
        self.path_end = None
        self.path_alpha = 0.0
        self.arc_center = None
        self.arc_radius = None
        self.start_angle = None
        self.end_angle = None
        self.path_speed = 0.5

    def get_target(self, model, data, ee_pos, ee_rot=None):
        if ee_rot is None:
            ee_rot = np.eye(3)

        if self.current_step >= len(self.steps):
            gripper_targets = {"left": 0.0, "right": 0.0}
            return ee_pos, np.eye(3), gripper_targets, None

        step = self.steps[self.current_step]
        target_type = step.get("target_type", "live")
        offset = step.get("offset", np.zeros(3))

        # --- Determine base position ---
        if target_type == "static" and self.grasped_pos is not None:
            base_pos = self.grasped_pos.copy()
        elif target_type == "arc_to_point":
            base_pos = ee_pos.copy()  # will be overwritten
        else:
            body_id = self.id_map.get(step.get("target_id"), None)
            base_pos = data.xpos[body_id].copy() if body_id is not None else ee_pos.copy()

        target_pos = base_pos + offset

        # --- Determine target rotation (only for non-arc steps) ---
        if target_type == "static":
            target_rot = F.RotX(np.pi / 2) @ F.RotY(-np.pi / 2)
        elif target_type == "arc_to_point":
            target_rot = ee_rot.copy()  # No rotation during arc
        else:
            body_id = self.id_map.get(step.get("target_id"), None)
            if body_id is not None:
                R_base = data.xmat[body_id].reshape(3, 3)
                target_rot = R_base @ F.RotX(np.pi / 2) @ F.RotY(-np.pi / 2)
            else:
                target_rot = ee_rot.copy()

        # --- Arc logic ---
        if target_type == "arc_to_point":
            if self.path_start is None:
                self.path_start = ee_pos.copy()
                self.path_end = step["pos"].copy()

                start_xy = self.path_start[:2]
                end_xy = self.path_end[:2]
                midpoint = (start_xy + end_xy) / 2
                vec = end_xy - start_xy
                perp = np.array([-vec[1], vec[0]])
                perp /= np.linalg.norm(perp)

                self.arc_radius = np.linalg.norm(vec) / 2
                self.arc_center = midpoint + perp * self.arc_radius

                self.start_angle = np.arctan2(start_xy[1] - self.arc_center[1], start_xy[0] - self.arc_center[0])
                self.end_angle = np.arctan2(end_xy[1] - self.arc_center[1], end_xy[0] - self.arc_center[0])
                if self.end_angle < self.start_angle:
                    self.end_angle += 2 * np.pi

                self.path_alpha = 0.0


            # Advance along arc
            self.path_alpha += self.path_speed
            theta = (1 - self.path_alpha) * self.start_angle + self.path_alpha * self.end_angle
            target_pos[:2] = self.arc_center + self.arc_radius * np.array([np.cos(theta), np.sin(theta)])
            target_pos[2] = self.path_start[2]  # keep Z constant

            # No rotation update — orientation stays fixed

            # Check if arc is complete
            if np.linalg.norm(target_pos[:2] - self.path_end[:2]) < 0.03:
                self.path_start = None
                self.path_end = None
                self.arc_center = None
                self.arc_radius = None
                self.start_angle = None
                self.end_angle = None
                self.path_alpha = 0.0
                self.advance_step()

        # --- Wait or auto-advance for non-arc steps ---
        else:
            if self.waiting:
                self.wait_timer += model.opt.timestep
                if self.wait_timer >= step.get("wait", 0.0):
                    self.waiting = False
                    self.wait_timer = 0.0
                    self.advance_step()
            else:
                dist = np.linalg.norm(ee_pos - target_pos)
                R_diff = ee_rot.T @ target_rot
                angle_rad = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
                angle_deg = np.degrees(angle_rad)

                if dist < 0.03 and (target_type == "arc_to_point" or angle_deg < 5.0):
                    if self.current_step + 1 < len(self.steps):
                        next_step = self.steps[self.current_step + 1]
                        if next_step.get("target_type") == "static":
                            self.grasped_pos = base_pos.copy()
                            self.grasped_rot = target_rot.copy()

                    if step.get("wait", 0.0) > 0.0:
                        self.waiting = True
                        self.wait_timer = 0.0
                    else:
                        self.advance_step()

        # --- Gripper control ---
        gripper_open = step.get("gripper", 0.0)
        gripper_targets = {
            "left": 0.04 * gripper_open,
            "right": -0.04 * gripper_open,
        }

        # --- Debug output ---
        dist = np.linalg.norm(ee_pos - target_pos)
        R_diff = ee_rot.T @ target_rot
        angle_rad = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        print(f"Step {self.current_step} | EE Pos: [{', '.join(f'{x:.3f}' for x in ee_pos)}] "
              f"| Target Pos: [{', '.join(f'{x:.3f}' for x in target_pos)}] "
              f"| Dist: {dist:.3f} | Angle Diff: {angle_deg:.2f}°", flush=True)

        return target_pos, target_rot, gripper_targets, gripper_open

    def advance_step(self):
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
