import mujoco
import mujoco.viewer
import numpy as np

def main():
    # Path to your Panda XML
    model_path = r"C:\MuJoCo\Project\franka_emika_panda\mjx_panda.xml"

    # Load the MuJoCo model
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Launch viewer
    viewer = mujoco.viewer.launch_passive(model, data)

    # Arm joint indices (assuming first 7 actuators control the arm)
    arm_ctrl_ids = list(range(7))

    # Target position (center of test tube)
    target_pos = np.array([0.6, 0, 0.89])  # adjust height to cylinder top

    # Get gripper site ID
    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")

    def get_gripper_pos():
        return data.site_xpos[gripper_id].copy()

    def move_to_target(target, steps=1500, kp=0.2, speed=0.2):
        print("Moving gripper toward the test tube...")
        for _ in range(steps):
            gripper_pos = get_gripper_pos()
            error = target - gripper_pos

            # Stop if close enough
            if np.linalg.norm(error) < 0.001:
                break

            # Compute site Jacobian
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, gripper_id)

            # Only consider first 7 joints for the arm
            jacp_arm = jacp[:, :7]

            # Compute joint velocities using pseudo-inverse
            dq = np.linalg.pinv(jacp_arm) @ (kp * error)
            dq *= speed  # slow down motion

            # Apply velocities to arm actuators
            for i, ctrl_id in enumerate(arm_ctrl_ids):
                data.ctrl[ctrl_id] += dq[i]

            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()  # renders the frame

    # Move gripper to test tube
    move_to_target(target_pos, steps=2000, kp=0.2, speed=0.1)

    # Close gripper (assuming last actuator controls gripper)
    print("Closing gripper to grab the tube...")
    data.ctrl[7] = 0.04  # fully close
    for _ in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()
    print("Gripper should now be holding the test tube!")

    # Keep simulation running
    while True:
        mujoco.mj_step(model, data)
        viewer.sync()

if __name__ == "__main__":
    main()
