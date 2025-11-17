import mujoco as mj
import mujoco.viewer
import os
from robot_controller_OSC import RobotController
from task_sequence import TaskSequence
import numpy as np

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
        # Reset simulation state (positions + velocities) to initial XML
        mj.mj_resetData(self.model, self.data)
        self.task_seq.reset()

        # Reset controller state
        self.controller.tau_prev = None
        self.controller.mass_blend = 0.0
        self.controller.object_grasped_flag = False

        # ----------------------------------------------------
        # Only change the robot joints — leave object poses alone
        # ----------------------------------------------------
        # Copy current qpos so we only modify part of it
        qpos = self.data.qpos.copy()

        # Example Panda "home" configuration (7 DOF)
        qpos[:7] = np.array([-0.25, -1.75, -1.0, -2.4, 2.4, 0.8, 1.4])

        # Assign it back (this preserves object states)
        self.data.qpos[:] = qpos

        # Recompute derived quantities
        mj.mj_forward(self.model, self.data)

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
    def get_ee_position(self):
            ee_site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, "ee_site")
            return np.array(self.data.site_xpos[ee_site_id])
    
    def render(self):
        """Render the current frame in human mode."""
        if self.viewer is not None:
            self.viewer.sync()

    def _compute_reward_done(self):
        done = self.task_seq.completed or self.task_seq.current_step >= len(self.task_seq.steps) and not self.task_seq.waiting
        reward = 1.0 if done else 0.0
        return reward, done, {"step_idx": self.task_seq.current_step}

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        # Stop sequence from running
        self.task_seq.active = False

    def get_body_names(self,prefix='',excluding='world'):
        """
            Get body names with prefix
        """
        body_names = [x for x in self.body_names if x is not None and x.startswith(prefix) and excluding not in x]
        return body_names
    
    def render_egocentric_rgbd_image(self, p_cam, R_cam, width=256, height=256, fovy=45.0):

        # Ensure inputs are numpy arrays
        p_cam = np.array(p_cam)
        R_cam = np.array(R_cam)

        # Create renderer with size
        renderer = mj.Renderer(self.model, height=height, width=width)

        # Setup camera
        camera = mj.MjvCamera()
        camera.type = mj.mjtCamera.mjCAMERA_FREE

        # Assign camera position, lookat, and up vectors element-wise
        for i in range(3):
            camera.pos[i] = p_cam[i]
            camera.lookat[i] = p_cam[i] + R_cam[i, 2]  # forward = z-axis
            camera.up[i] = R_cam[i, 1]                  # up = y-axis

        # Setup rendering options
        option = mj.MjvOption()

        # Update the scene with this camera pose
        mj.mjv_updateScene(
            self.model,
            self.data,
            option,
            None,
            camera,
            mj.mjtCatBit.mjCAT_ALL,
            renderer.scene
        )

        # Render and read pixels
        renderer.render()
        rgb_img, depth_ndc = renderer.read_pixels(depth=True)

        # Convert NDC depth to meters
        extent = self.model.stat.extent
        near = self.model.vis.map.znear * extent
        far = self.model.vis.map.zfar * extent
        depth_img = near / (1.0 - depth_ndc * (1.0 - near / far))

        # Free renderer resources
        renderer.free()

        return rgb_img, depth_img


    def get_camera_pose(self, cam_body_name):
        body_id = self.model.body(cam_body_name).id
        pos = self.data.xpos[body_id].copy()
        rot = self.data.xmat[body_id].reshape(3, 3).copy()
        return pos, rot


    
