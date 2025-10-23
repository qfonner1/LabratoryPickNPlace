import time
import cv2
import numpy as np
from robot_env import RobotEnv
import object_detection as OD
import mujoco as mj

# ------------------------------
# Object Detection
# ------------------------------
detected_objects = OD.object_detection("franka_panda_w_objs.xml", "overhead_cam")

# ------------------------------
# Initialize environment (offscreen for recording)
# ------------------------------
# Keep RobotEnv unchanged, just use offscreen rendering
env = RobotEnv("franka_panda_w_objs.xml", render_mode="rgb_array")  # offscreen
env.model.opt.integrator = 1
env.task_seq.set_targets_from_vision(detected_objects)
obs = env.reset()

# ------------------------------
# Video Writer setup
# ------------------------------
width, height = 640, 480
fps = int(1 / env.dt)
out = cv2.VideoWriter(
    "simulation_recording.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# ------------------------------
# Camera setup (fixed overhead)
# ------------------------------
cam_name = "cam"
cam_pos, cam_rot = env.get_camera_pose(cam_name)

# ------------------------------
# Simulation loop
# ------------------------------
try:
    while True:
        obs, reward, done, info = env.step()

        # ---- Capture frame offscreen ----
        # Use RobotEnv's render_egocentric_rgbd_image for offscreen capture
        try:
            rgb_img, _ = env.render_egocentric_rgbd_image(
                p_cam=cam_pos,
                R_cam=cam_rot,
                width=width,
                height=height
            )
            frame_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

            # Optional live preview
            cv2.imshow("Recording", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Recording interrupted by user.")
                break
        except Exception as e:
            # If render_egocentric_rgbd_image fails (e.g., MuJoCo 3.x camera issues)
            # fallback: skip frame
            print(f"Frame capture skipped: {e}")

        if done:
            print("Task sequence complete!")
            break

        time.sleep(env.dt)

except KeyboardInterrupt:
    print("Simulation interrupted by user.")

finally:
    out.release()
    cv2.destroyAllWindows()
    env.close()
    print("Video saved as simulation_recording.mp4")
