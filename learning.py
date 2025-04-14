from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

N_ENVS = 4

policy_kwargs = dict(
    net_arch=[dict(pi=[128, 64, 64], vf=[128, 64, 64])],
    log_std_init=-1.0 
)

checkpoint_callback = CheckpointCallback(
    save_freq=10000,  
    save_path='./checkpoints/',
    name_prefix='walker_model'
)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--use_3d", action="store_true", help="learning with 3d")
parser.add_argument("--motion", type=str, default="walk.bvh", help="motion name")
args = parser.parse_args()
if __name__ == "__main__":
    num_cpu = N_ENVS
    
    if args.use_3d:
        from custom_humanoid3d import CustomEnvWrapper
    else:
        from custom_walker2d import CustomEnvWrapper

    def make_env(motion_path):
        def _init():
            env = CustomEnvWrapper(render_mode=None, motion_path = motion_path)
            return env
        return _init
    motion_path = "/asset/motions/" + args.motion
    env = SubprocVecEnv([make_env(motion_path = motion_path) for _ in range(num_cpu)])
    env = VecMonitor(env)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/", policy_kwargs=policy_kwargs, device="cpu", learning_rate=0.0001)
    model.learn(total_timesteps=10000000000, callback=checkpoint_callback)
