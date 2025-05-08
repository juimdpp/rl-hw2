from motion import Motion
import numpy as np
import gymnasium as gym
import os

class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, render_mode="human", motion_path="/asset/motions/walk.bvh"):
        env = gym.make(
            "Walker2d-v5",
            xml_file=os.getcwd() + "/asset/custom_walker2d_ref.xml",
            render_mode=render_mode,
            exclude_current_positions_from_observation=False,
            max_episode_steps=500, # Only 10s
            frame_skip = 10, # 0.02 s per one step
            )
        
        self.ref_motion = Motion(os.getcwd() + motion_path, "walker2d")
        self.ref_pos = None
        self.sim_skel_dof = 9
        self.ref_skel_dof = 9
        super().__init__(env)
        obs, _ = self.reset()   
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(obs),), dtype=np.float64)
    
    def update_ref_pose(self, time, obs):
        ref_pos = self.ref_motion.get_ref_poses(time)
        self.env.unwrapped.data.qpos[-self.sim_skel_dof:] = ref_pos
        self.env.unwrapped.data.qvel[-self.sim_skel_dof:] *= 0.0
        obs[9:18] = ref_pos
        self.ref_pos = ref_pos.copy()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.update_ref_pose(self.env.unwrapped.data.time, obs)
        custom_obs = self.custom_observation(obs)
        return custom_obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action) # or self.env.step(self.custom_pd_actuator(action))
        self.update_ref_pose(self.env.unwrapped.data.time, obs)
        custom_obs = self.custom_observation(obs)
        custom_reward = self.custom_reward(obs)
        custom_terminated = self.custom_terminated(terminated)
        custom_truncated = self.custom_truncated(truncated)
        
        return custom_obs, custom_reward, custom_terminated, custom_truncated, info

    def custom_pd_actuator(self, target_offset):
        torque = np.zeros(self.sim_skel_dof - 3)

        return torque

    def custom_terminated(self, terminated):
        # TODO: Implement your own termination condition
        return terminated
    
    def custom_truncated(self, truncated):
        # TODO: Implement your own truncation condition
        return truncated

    def custom_observation(self, obs):
        # Remove reference skeleton velocities (last 9 elements) — always zero, not informative.
        obs = obs[:-self.ref_skel_dof] 
        
        # TODO : Implement your own observation
        return obs

    def custom_reward(self, obs):
        # 1. Pose imitation reward (joint angle difference)
        sim_joint_angles = obs[3:9]
        ref_joint_angles = obs[12:18]  # Only first 6 reference joints match
        pose_diff = np.square(ref_joint_angles - sim_joint_angles)
        pose_reward = np.exp(-5 * np.sum(pose_diff))

        # 2. Root height reward (z position comparison)
        sim_root_z = obs[1]
        ref_root_z = self.ref_pos[1] + 1.25  # Apply offset
        root_diff = (sim_root_z - ref_root_z) ** 2
        root_reward = np.exp(-5 * root_diff)

        # Final reward (weights from DeepMimic)
        reward = (
            0.7 * pose_reward +
            0.7 * root_reward 
            # 0.20 * vel_reward
        )
        return reward

## Test
if __name__ == "__main__":
    env = CustomEnvWrapper()
    obs, _ = env.reset() 
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(env.custom_pd_actuator(action))
        env.render()
