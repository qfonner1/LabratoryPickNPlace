import numpy as np
import Functions as F  # Ensure RotX and RotY are defined

class TaskSequence:
    def __init__(self, model):
        self.id_map = {
            "box": model.body("obj_box_06").id,
        }

        self.grasped_pos = None
        self.grasped_rot = None  # Optional: store rotation at grasp if needed

        self.steps = [
            {"target_id": "box", "target_type": "live", "offset": np.array([0.0, 0.0, 0.10]), "gripper": 1.0, "wait": 1.0},
            {"target_id": "box", "target_type": "live", "offset": np.array([0.0, 0.0, 0.00]), "gripper": 1.0, "wait": 1.0},
            {"target_id": "box", "target_type": "live", "offset": np.array([0.0, 0.0, 0.00]), "gripper": 0.0, "wait": 1.0},
            {"target_id": "box", "target_type": "static", "offset": np.array([0.0, 0.0, 0.10]), "gripper": 0.0, "wait": 1.0},
            {"target_type": "absolute", "position": np.array([0.0, -0.5, 1.25]), "gripper": 0.0, "wait": 1.0},
            {"target_type": "absolute2", "position": np.array([0.9, 0.0, 1.25]), "gripper": 0.0, "wait": 1.0},
            {"target_type": "absolute2", "position": np.array([0.9, 0.0, 1.15]), "gripper": 0.0, "wait": 1.0},
            {"target_type": "absolute2", "position": np.array([0.9, 0.0, 1.15]), "gripper": 1.0, "wait": 1.0},
            {"target_type": "absolute2", "position": np.array([0.9, 0.0, 1.5]), "gripper": 1.0, "wait": 1.0}

        ]

        self.current_step = 0
        self.waiting = False
        self.wait_timer = 0.0

    def get_target(self, model, data, ee_pos):
        if self.current_step >= len(self.steps):
            # Sequence complete: hold current pose
            gripper_targets = {"left": 0.0, "right": 0.0}
            return ee_pos, np.eye(3), gripper_targets, None

        step = self.steps[self.current_step]
        target_type = step.get("target_type", "live")
        offset = step.get("offset", np.zeros(3))

        # Determine base position
        if target_type in ["absolute", "absolute2"]:
            base_pos = step["position"]
        elif target_type == "static" and self.grasped_pos is not None:
            base_pos = self.grasped_pos.copy()
        else:
            body_id = self.id_map[step["target_id"]]
            base_pos = data.xpos[body_id].copy()

        # Determine rotation
        if target_type == "absolute":
            target_rot = F.RotX(np.pi/2) 

        elif target_type == "absolute2":
            target_rot = F.RotX(np.pi/2) @ F.RotY(np.pi / 2)

        elif target_type == "static":
            target_rot = F.RotX(np.pi/2) @ F.RotY(-np.pi / 2)
        else:
            body_id = self.id_map[step["target_id"]]
            R_base = data.xmat[body_id].reshape(3, 3)
            target_rot = R_base @ F.RotX(np.pi/2) @ F.RotY(-np.pi/2)

        target_pos = base_pos + offset

        # Step progression logic
        if self.waiting:
            self.wait_timer += model.opt.timestep
            if self.wait_timer >= step.get("wait", 0.0):
                self.waiting = False
                self.wait_timer = 0.0
                self.advance_step()
        else:
            dist = np.linalg.norm(ee_pos - target_pos)
            print(f"Step {self.current_step} | EE Pos: {ee_pos} | Target Pos: {target_pos} | Dist: {dist:.4f}")
            if dist < 0.01:
                # If next step is static, save current position
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

        # Gripper control
        gripper_open = step["gripper"]
        left_finger_target = 0.04 * gripper_open
        right_finger_target = -0.04 * gripper_open
        gripper_targets = {"left": left_finger_target, "right": right_finger_target}

        return target_pos, target_rot, gripper_targets, gripper_open

    def advance_step(self):
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
