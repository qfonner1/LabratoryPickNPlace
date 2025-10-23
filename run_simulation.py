import time
from robot_env import RobotEnv

# ------------------------------
# Initialize environment
# ------------------------------
env = RobotEnv("franka_panda_w_objs.xml", render_mode="human")
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
