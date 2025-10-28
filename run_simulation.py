import time
from robot_env import RobotEnv
import object_detection as OD

# ------------------------------
# Object Detection
# ------------------------------
print("Running object detection...")
detected_objects = OD.object_detection("franka_panda_w_objs.xml", "overhead_cam")
detected_targets=OD.object_detection("franka_panda_w_objs.xml", "overhead_cam2")
# ------------------------------
# Initialize environment
# ------------------------------
env = RobotEnv("franka_panda_w_objs.xml", render_mode="human")
env.model.opt.integrator = 1  # Runge-Kutta 4

env.task_seq.set_boxes_from_vision(detected_objects)
env.task_seq.set_targets_from_vision(detected_targets)
print("[TaskSequence] Targets updated from vision!")

obs = env.reset()

# ------------------------------
# Simulation loop
# ------------------------------
try:
    while True:
        obs, reward, done, info = env.step()

        # Stop if sequence is complete
        if done:
            print("Sequence complete!")
            break

        # Small sleep to match simulation timestep
        time.sleep(env.dt)

except KeyboardInterrupt:
    print("Simulation interrupted by user.")

finally:
    env.close()
    print("Environment closed.")
