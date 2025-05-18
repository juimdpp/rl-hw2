import gymnasium as gym
# import d4rl
import numpy as np
from stable_baselines3 import PPO

def evaluate_policy(model, env, episodes=100):
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        returns.append(total_reward)

    mean_return = np.mean(returns)
    normalized = env.get_normalized_score(mean_return)
    print(f"Raw return: {mean_return:.2f}")
    print(f"Normalized score: {normalized:.2f}")
    return normalized



# Use the same environment DT was trained on
env = gym.make("Walker2d-medium-v0")

model = PPO.load("models/try6_walker_model_62000004.zip")

normalized_score = evaluate_policy(model, env)
print(f"Normalized score: {normalized_score:.2f}")