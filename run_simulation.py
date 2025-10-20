from robot_env import RobotEnv
import time

env = RobotEnv("franka_panda_w_objs.xml", render_mode="human")
obs = env.reset()
done = False

while not done:
    obs, reward, done, info = env.step()
    time.sleep(env.dt)

env.close()
