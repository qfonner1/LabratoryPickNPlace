import mujoco as mj
import os
import mujoco.viewer
from robot_controller import RobotController
from task_sequence import TaskSequence


class RobotEnv:
    def __init__(self, xml_path, render_mode="human"):
        # ------------------------------
        # Model and Data Setup
        # ------------------------------
        self.xml_path = os.path.abspath(xml_path)
        self.model = mj.MjModel.from_xml_path(self.xml_path)
        self.data = mj.MjData(self.model)
        self.ee_site_id = self.model.site("grip_site").id

        # Actuator IDs for gripper
        self.act1 = self.model.actuator("panda_gripper_finger_joint1").id
        self.act2 = self.model.actuator("panda_gripper_finger_joint2").id

        # Initialize helper objects
        self.controller = RobotController(self.ee_site_id, self.model)
        self.task_seq = TaskSequence(self.model)

        # Register MuJoCo control callback
        mj.set_mjcb_control(self._control_callback)

        # Rendering
        self.render_mode = render_mode
        self.viewer = None
        if self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # Simulation timing
        self.dt = self.model.opt.timestep
        self.simend = 500  # seconds

    # ------------------------------
    # Control Callback (Core Logic)
    # ------------------------------
    def _control_callback(self, model, data):
        """Executed automatically every physics step."""
        ee_pos = data.site_xpos[self.ee_site_id].copy()
        ee_rot = data.site_xmat[self.ee_site_id].reshape(3, 3)

        # Query the task sequence for the current target
        target_pos, target_rot, gripper_targets, gripper_open = self.task_seq.get_target(
            model, data, ee_pos, ee_rot
        )

        # Compute and apply torques using your controller
        self.controller.controller(model, data, target_pos, target_rot, gripper_targets, gripper_open)

        # Apply gripper actuation commands
        data.ctrl[self.act1] = gripper_targets["left"]
        data.ctrl[self.act2] = gripper_targets["right"]

    # ------------------------------
    # Gym-style Methods
    # ------------------------------
    def reset(self):
        # Reset the MuJoCo simulation state
        mj.mj_resetData(self.model, self.data)

        # Reset TaskSequence
        self.task_seq.current_step = 0
        self.task_seq.waiting = False
        self.task_seq.wait_timer = 0.0
        self.task_seq.grasped_pos = None
        self.task_seq.grasped_rot = None

        # Reset controller
        self.controller.tau_prev = None
        self.controller.mass_blend = 0.0
        self.controller.object_grasped_flag = False

        # Optional: reset simulation time counter if you track it
        self.time_elapsed = 0.0

        # Return initial observation
        return self._get_obs()

    def step(self, action=None):
        mj.mj_step(self.model, self.data)

        # Render the current frame
        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()
        reward, done, info = self._compute_reward_done()
        return obs, reward, done, info

    # ------------------------------
    # Utility Functions
    # ------------------------------
    def _get_obs(self):
        """Return observation dictionary."""
        obs = {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ee_pos": self.data.site_xpos[self.ee_site_id].copy(),
            "ee_rot": self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy(),
            "step_idx": self.task_seq.current_step,
        }
        return obs

    def _compute_reward_done(self):
        """
        Returns done=True only when the last step has fully finished including wait.
        """
        last_step_completed = (
            self.task_seq.current_step >= len(self.task_seq.steps)
            and not self.task_seq.waiting
        )
        done = last_step_completed
        reward = 1.0 if done else 0.0
        info = {"step_idx": self.task_seq.current_step}
        return reward, done, info

    def render(self):
        if self.viewer is not None:
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None



