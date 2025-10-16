import numpy as np
import Functions as F  # Ensure RotX, RotY, RotZ, axang_to_rot are defined

class TaskSequence:
    def __init__(self, model):
        self.id_map = {
            "box2": model.body("obj_box_06").id,
            "box3": model.body("obj_box_07").id,
            "box": model.body("obj_box_08").id
        }

        self.grasped_pos = None
        self.grasped_rot = None

        self.steps = [
            {"target_id": "box", "target_type": "live", "offset": np.array([0.0, 0.0, 0.15]),                                                  "gripper": 1.0, "wait": 1.0},
            {"target_id": "box", "target_type": "live", "offset": np.array([0.0, 0.0, 0.0]),                                                   "gripper": 1.0, "wait": 1.0},
            {"target_id": "box", "target_type": "live", "offset": np.array([0.0, 0.0, 0.0]),                                                   "gripper": 0.0, "wait": 1.0},
            {"target_id": "box", "target_type": "static", "offset": np.array([0.0, 0.0, 0.15]), "rot": F.RotX(np.pi / 2) @ F.RotY(-np.pi / 2), "gripper": 0.0, "wait": 1.0},
            {"target_id":  None, "target_type": "absolute", "pos": np.array([0.0, -0.9, 1.25]), "rot": F.RotX(np.pi / 2),                      "gripper": 0.0, "wait": 1.0},
            {"target_id":  None, "target_type": "absolute", "pos": np.array([0.9, 0.2, 1.25]),  "rot": F.RotX(np.pi / 2) @ F.RotY(np.pi / 2),  "gripper": 0.0, "wait": 1.0},
            {"target_id": "box", "target_type": "absolute", "pos": np.array([0.9, 0.2, 1.18]), "rot": F.RotX(np.pi / 2) @ F.RotY(np.pi / 2),   "gripper": 0.0, "wait": 1.0},
            {"target_id": "box", "target_type": "live", "offset": np.array([0.0, 0.0, 0.0]),                                                   "gripper": 1.0, "wait": 1.0},
            {"target_id":  None, "target_type": "absolute", "pos": np.array([0.9, 0.2, 1.35]), "rot": F.RotX(np.pi / 2) @ F.RotY(np.pi / 2),      "gripper": 1.0, "wait": 1.0},

            {"target_id":  None, "target_type": "absolute", "pos": np.array([0.0, -0.9, 1.25]), "rot": F.RotX(np.pi / 2),                      "gripper": 1.0, "wait": 1.0},
            {"target_id": "box2", "target_type": "live", "offset": np.array([0.0, 0.0, 0.15]),                                                  "gripper": 1.0, "wait": 1.0},
            {"target_id": "box2", "target_type": "live", "offset": np.array([0.0, 0.0, 0.0]),                                                   "gripper": 1.0, "wait": 1.0},
            {"target_id": "box2", "target_type": "live", "offset": np.array([0.0, 0.0, 0.0]),                                                   "gripper": 0.0, "wait": 1.0},
            {"target_id": "box2", "target_type": "static", "offset": np.array([0.0, 0.0, 0.15]), "rot": F.RotX(np.pi / 2) @ F.RotY(-np.pi / 2), "gripper": 0.0, "wait": 1.0},
            {"target_id":  None, "target_type": "absolute", "pos": np.array([0.0, -0.9, 1.25]), "rot": F.RotX(np.pi / 2),                      "gripper": 0.0, "wait": 1.0},
            {"target_id":  None, "target_type": "absolute", "pos": np.array([0.9, 0, 1.25]),    "rot": F.RotX(np.pi / 2) @ F.RotY(np.pi / 2),  "gripper": 0.0, "wait": 1.0},
            {"target_id": "box2", "target_type": "absolute", "pos": np.array([0.9, 0.0, 1.18]), "rot": F.RotX(np.pi / 2) @ F.RotY(np.pi / 2),   "gripper": 0.0, "wait": 1.0},
            {"target_id": "box2", "target_type": "live", "offset": np.array([0.0, 0.0, 0.0]),                                                   "gripper": 1.0, "wait": 1.0},
            {"target_id":  None, "target_type": "absolute", "pos": np.array([0.9, 0, 1.35]), "rot": F.RotX(np.pi / 2) @ F.RotY(np.pi / 2),      "gripper": 1.0, "wait": 1.0},

            {"target_id":  None, "target_type": "absolute", "pos": np.array([0.0, -0.9, 1.25]), "rot": F.RotX(np.pi / 2),                      "gripper": 1.0, "wait": 1.0},
            {"target_id": "box3", "target_type": "live", "offset": np.array([0.0, 0.0, 0.15]),                                                  "gripper": 1.0, "wait": 1.0},
            {"target_id": "box3", "target_type": "live", "offset": np.array([0.0, 0.0, 0.0]),                                                   "gripper": 1.0, "wait": 1.0},
            {"target_id": "box3", "target_type": "live", "offset": np.array([0.0, 0.0, 0.0]),                                                   "gripper": 0.0, "wait": 1.0},
            {"target_id": "box3", "target_type": "static", "offset": np.array([0.0, 0.0, 0.15]), "rot": F.RotX(np.pi / 2) @ F.RotY(-np.pi / 2), "gripper": 0.0, "wait": 1.0},
            {"target_id":  None, "target_type": "absolute", "pos": np.array([0.0, -0.9, 1.25]), "rot": F.RotX(np.pi / 2),                      "gripper": 0.0, "wait": 1.0},
            {"target_id":  None, "target_type": "absolute", "pos": np.array([0.85, -0.18, 1.25]),    "rot": F.RotX(np.pi / 2) @ F.RotY(np.pi / 2),  "gripper": 0.0, "wait": 1.0},
            {"target_id": "box3", "target_type": "absolute", "pos": np.array([0.85, -0.18, 1.18]), "rot": F.RotX(np.pi / 2) @ F.RotY(np.pi / 2),   "gripper": 0.0, "wait": 1.0},
            {"target_id": "box3", "target_type": "live", "offset": np.array([0.0, 0.0, 0.0]),                                                   "gripper": 1.0, "wait": 1.0},
            {"target_id":  None, "target_type": "absolute", "pos": np.array([0.85, -0.18, 1.35]), "rot": F.RotX(np.pi / 2) @ F.RotY(np.pi / 2),      "gripper": 1.0, "wait": 1.0},
            {"target_id":  None, "target_type": "absolute", "pos": np.array([0.0, 0.0, 2.23]), "rot": np.eye(3,3),                      "gripper": 0.0, "wait": 1.0}
        ]

        self.current_step = 0
        self.waiting = False
        self.wait_timer = 0.0


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
        elif target_type == "absolute":
            base_pos = step.get("pos").copy()
        else:
            body_id = self.id_map.get(step.get("target_id"), None)
            base_pos = data.xpos[body_id].copy() if body_id is not None else ee_pos.copy()

        target_pos = base_pos + (offset if target_type != "absolute" else 0)

        # --- Determine target rotation  ---
        if target_type in ["static", "absolute"]:
            target_rot = step.get("rot").copy()
        else:
            body_id = self.id_map.get(step.get("target_id"), None)
            if body_id is not None:
                R_base = data.xmat[body_id].reshape(3, 3)
                target_rot = R_base @ F.RotX(np.pi / 2) @ F.RotY(-np.pi / 2)
            else:
                target_rot = ee_rot.copy()


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

            if dist < 0.03 and angle_deg < 5.0:
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
