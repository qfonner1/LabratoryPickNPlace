import mujoco as mj
import mujoco.viewer
import os
from robot_controller import RobotController
from task_sequence import TaskSequence

class RobotEnv:
    def __init__(self, xml_path, render_mode="human"):
        self.xml_path = os.path.abspath(xml_path)
        self.model = mj.MjModel.from_xml_path(self.xml_path)
        self.data = mj.MjData(self.model)
        self.ee_site_id = self.model.site("grip_site").id

        self.act1 = self.model.actuator("panda_gripper_finger_joint1").id
        self.act2 = self.model.actuator("panda_gripper_finger_joint2").id

        self.controller = RobotController(self.ee_site_id, self.model)
        self.task_seq = TaskSequence(self.model)

        self.render_mode = render_mode
        self.viewer = None
        if self.render_mode == "human":
            # launch_passive returns a viewer object that does NOT auto-run
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.dt = self.model.opt.timestep
        self.running = True
        self.active = True  # <--- Add this

    def reset(self):
        mj.mj_resetData(self.model, self.data)   # reset simulation
        self.task_seq.reset()                     # reset sequence
        # reset controller state
        self.controller.tau_prev = None
        self.controller.mass_blend = 0.0
        self.controller.object_grasped_flag = False
        return self._get_obs()


    def step(self, action=None):
        # Get EE state
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        ee_rot = self.data.site_xmat[self.ee_site_id].reshape(3, 3)

        # Get target from task sequence
        target_pos, target_rot, gripper_targets, gripper_open = self.task_seq.get_target(
            self.model, self.data, ee_pos, ee_rot
        )

        # Apply controller
        self.controller.controller(self.model, self.data, target_pos, target_rot, gripper_targets, gripper_open)

        # Apply gripper commands
        self.data.ctrl[self.act1] = gripper_targets["left"]
        self.data.ctrl[self.act2] = gripper_targets["right"]

        # Step simulation
        mj.mj_step(self.model, self.data)

        # Render if needed
        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()
        reward, done, info = self._compute_reward_done()
        return obs, reward, done, info



    def _get_obs(self):
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ee_pos": self.data.site_xpos[self.ee_site_id].copy(),
            "ee_rot": self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy(),
            "step_idx": self.task_seq.current_step,
        }
    
    def render(self):
        """Render the current frame in human mode."""
        if self.viewer is not None:
            self.viewer.sync()


    def _compute_reward_done(self):
        done = self.task_seq.current_step >= len(self.task_seq.steps) and not self.task_seq.waiting
        reward = 1.0 if done else 0.0
        return reward, done, {"step_idx": self.task_seq.current_step}

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        # Stop sequence from running
        self.task_seq.active = False
