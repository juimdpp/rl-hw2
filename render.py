import argparse
from stable_baselines3 import PPO
import gymnasium as gym

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None, help="Path to the saved model (.zip)")
parser.add_argument("--use_3d", action="store_true", help="learning with 3d")
parser.add_argument("--motion", type=str, default="walk.bvh", help="motion name")

args = parser.parse_args()

if args.use_3d:
    from custom_humanoid3d import CustomEnvWrapper
else:
    from custom_walker2d import CustomEnvWrapper
motion_path = "/asset/motions/" + args.motion

env = CustomEnvWrapper(render_mode="human", motion_path=motion_path)
model = PPO.load(args.model, env=env) if args.model is not None else None
obs, _ = env.reset()

import numpy as np

while True:
    if model is not None:
        action, _ = model.predict(obs, deterministic=False)
    else:
        action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    print(obs[-2], np.sum(obs[3:9] - obs[12:18] > 0.34))
    if terminated or truncated:
        obs, _ = env.reset()