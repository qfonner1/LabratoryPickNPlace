import time
from robot_env import RobotEnv
import object_detection as OD
import time
from saving_config import BASE_OUTPUT_DIR 
import numpy as np
from logger import Logger
import sys, os

# ------------------------------
# Output Saving
# ------------------------------
# --- Usage in your simulation ---
log_file = os.path.join(BASE_OUTPUT_DIR, "sim_output.txt")
sys.stdout = Logger(log_file)
sys.stderr = sys.stdout  # also capture errors
print(f"[Run Simulation] All outputs will be saved to: {BASE_OUTPUT_DIR}")
print("[Run Simulation] Simulation starting...")

# ------------------------------
# Object Detection
# ------------------------------
print("[Run Simulation] Running object detection...")
detected_objects = OD.object_detection("franka_panda_w_objs.xml", "overhead_cam")
detected_targets=OD.object_detection("franka_panda_w_objs.xml", "overhead_cam2")
# ------------------------------
# Initialize environment
# ------------------------------
env = RobotEnv("franka_panda_w_objs.xml", render_mode="human")
env.model.opt.integrator = 1  # Runge-Kutta 4

env.task_seq.set_boxes_from_vision(detected_objects)
env.task_seq.set_targets_from_vision(detected_targets)

# Generate dynamic steps based on current EE position
ee_pos = env.get_ee_position()  
env.task_seq.generate_steps(ee_pos)
print("[Run Simulation] Targets updated from vision!")

obs = env.reset()

# ------------------------------
# Simulation loop
# ------------------------------
try:
    while True:
        obs, reward, done, info = env.step()

        # Stop if sequence is complete
        if done:
            print("[Run Simulation] Sequence complete!")
            break

        # Small sleep to match simulation timestep
        time.sleep(env.dt)

except KeyboardInterrupt:
    print("[Run Simulation] Simulation interrupted by user.")

finally:
    env.close()
    print("[Run Simulation] Environment closed.")


# ------------------------------
# Final Stats
# ------------------------------

# Get body position by name
def get_body_position(model, data, body_name):
    body_id = model.body(body_name).id
    pos = data.xpos[body_id].copy()  # (x, y, z)
    return pos

pairs = [("box1", "target1"),
         ("box2", "target2"),
         ("box3", "target3")]

for obj_name, tgt_name in pairs:
    object_pos = get_body_position(env.model, env.data, obj_name)[:2]
    target_pos = get_body_position(env.model, env.data, tgt_name)[:2]

    error = np.linalg.norm(object_pos - target_pos)
    axis_error = object_pos - target_pos

    print(f"[Run Simulation] Pair: {obj_name} → {tgt_name} | Object position: {object_pos} | Target position: {target_pos} | Distance to target: {error:.3f} meters")

# --- After simulation ---
sys.stdout.log.close()    # close file safely
sys.stdout = sys.__stdout__  # restore original stdout
sys.stderr = sys.__stderr__  # restore stderr