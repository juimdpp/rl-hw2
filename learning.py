from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from custom_checkpoint_callback import CustomCheckpointCallback  # save the class in this file

N_ENVS = 8
freq = 500000

policy_kwargs = dict(
    net_arch=[dict(pi=[128, 64, 64], vf=[128, 64, 64])],
    log_std_init=-1.0 
)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--use_3d", action="store_true", help="learning with 3d")
parser.add_argument("--motion", type=str, default="walk.bvh", help="motion name")
parser.add_argument("--resume", action="store_true", help="resume training from latest checkpoint")
parser.add_argument("--model_name", type=str, default="./model", help="path to the model to load")

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

    import os
    import glob

    import re
    import numpy as np
    # Extract timestep from filename, e.g., "model_walker_model_50000.zip"
    def extract_step(path):
        match = re.search(r"_(\d+)\.zip$", path)
        return int(match.group(1)) if match else 0

    if args.resume:
        checkpoint_files = glob.glob(f"./checkpoints/{args.model_name}_walker_model_*.zip")
        if len(checkpoint_files) == 0:
            raise FileNotFoundError("No checkpoint files found to resume from.")
        step_list = list(map(extract_step, checkpoint_files))
        latest_idx = np.argmax(step_list)
        latest_checkpoint = checkpoint_files[latest_idx]
        last_step = step_list[latest_idx]
        print(f"Latest checkpoint found: {latest_checkpoint}")
        print(f"Extracted step: {last_step}")
        print(f"Resuming from checkpoint: {latest_checkpoint} at step {last_step}")

        model = PPO.load(latest_checkpoint, env=env, tensorboard_log=f"./logs/{args.model_name}", device="cpu")

        # Redefine checkpoint callback with offset
        checkpoint_callback = CustomCheckpointCallback(
            save_freq=freq,
            save_path='./checkpoints/',
            name_prefix=f"{args.model_name}_walker_model",
            save_start=last_step + freq,  # Start saving from next logical step
            verbose=1
        )
    else:
        last_step = 0
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"./logs/{args.model_name}",
                    policy_kwargs=policy_kwargs, device="cpu",
                    learning_rate=0.0001, batch_size=256, n_steps=4056)
        checkpoint_callback = CustomCheckpointCallback(
            save_freq=freq,
            save_path='./checkpoints/',
            name_prefix=f"{args.model_name}_walker_model",
            save_start=0,
            verbose=1
        )

    model.learn(total_timesteps=10000000000, callback=checkpoint_callback)
    # model.save(f"models/{args.model_name}")