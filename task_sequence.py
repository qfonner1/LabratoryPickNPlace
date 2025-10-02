import numpy as np
import Functions as F  # Ensure RotX and RotY are defined

class TaskSequence:
    def __init__(self, model):
        # Internal mapping from names to body IDs
        self.id_map = {
            "box": model.body("obj_box_06").id,
        }

        # Motion sequence with optional wait durations (seconds)
        self.steps = [
            {"target_id": "box", "offset": np.array([0.0, 0.0, 0.10]), "gripper": 1.0, "wait": 2.0},  # move above box
            {"target_id": "box", "offset": np.array([0.0, 0.0, 0.00]), "gripper": 1.0, "wait": 2.0},  # move to box center
            {"target_id": "box", "offset": np.array([0.0, 0.0, 0.00]), "gripper": 0.0, "wait": 2.0},  # grip box halfway
            {"target_id": "box", "offset": np.array([0.0, 0.0, 0.10]), "gripper": 0.0, "wait": 2.0},  # lift box
        ]

        self.current_step = 0
        self.waiting = False
        self.wait_timer = 0.0

    def get_target(self, model, data, ee_pos):
        if self.current_step >= len(self.steps):
            # Sequence finished: return current EE position, rotation, gripper as-is
            body_id = self.id_map[step["target_id"]]
            base_pos = data.xpos[body_id].copy()
            R_base = data.xmat[body_id].reshape(3, 3)
            target_rot = R_base
            gripper_targets = {"left": 0.0, "right": 0.0}  # fully closed
            return base_pos, target_rot, gripper_targets, None

        step = self.steps[self.current_step]
        body_id = self.id_map[step["target_id"]]

        base_pos = data.xpos[body_id].copy()
        R_base = data.xmat[body_id].reshape(3, 3)

        target_rot = R_base @ F.RotX(np.pi / 2) @ F.RotY(-np.pi / 2)
        target_pos = base_pos + step["offset"]

        # Step advancement / wait logic
        if self.waiting:
            self.wait_timer += model.opt.timestep
            if self.wait_timer >= step.get("wait", 0.0):
                self.waiting = False
                self.wait_timer = 0.0
                self.advance_step()
        else:
            dist = np.linalg.norm(ee_pos - target_pos)
            if dist < 0.03:  # threshold for reaching target
                if step.get("wait", 0.0) > 0.0:
                    self.waiting = True
                    self.wait_timer = 0.0
                else:
                    self.advance_step()

        # Convert gripper_open to actual finger joint positions
        gripper_open = step["gripper"]
        left_finger_target = 0.04 * gripper_open   # open=1 → 0.04, closed=0 → 0
        right_finger_target = -0.04 * gripper_open

        gripper_targets = {"left": left_finger_target, "right": right_finger_target}

        return target_pos, target_rot, gripper_targets, gripper_open

    
    def advance_step(self):
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
