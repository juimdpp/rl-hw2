import numpy as np
import gymnasium as gym
import os
from motion import Motion

class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, render_mode="human", motion_path="/asset/motions/walk.bvh"):
        env = gym.make(
            "Humanoid-v5",
            xml_file = os.getcwd() + "/asset/custom_humanoid3d_lowerbody.xml",
            render_mode=render_mode,
            max_episode_steps=500, # Only 10s
            exclude_current_positions_from_observation = False,
            include_cinert_in_observation = False, 
            include_cvel_in_observation = False, 
            include_qfrc_actuator_in_observation = False, 
            include_cfrc_ext_in_observation = False,
            frame_skip = 10)
        

        self.ref_motion = Motion(os.getcwd() + motion_path, "humanoid3d_lowerbody")
        self.ref_pos = None
        self.sim_skel_dof = 18
        self.ref_skel_dof = 18        
        super().__init__(env)
        
        obs, _ = self.reset()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(obs),), dtype=np.float64)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.update_ref_pose(self.env.unwrapped.data.time, obs)
        custom_obs = self.custom_observation(obs)
        return custom_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.update_ref_pose(self.env.unwrapped.data.time, obs)
        custom_obs = self.custom_observation(obs)
        custom_reward = self.custom_reward(obs)
        custom_terminated = self.custom_terminated(terminated)
        custom_truncated = self.custom_truncated(truncated)
        return custom_obs, custom_reward, custom_terminated, custom_truncated, info

    def update_ref_pose(self, time, obs):
        ref_pos = self.ref_motion.get_ref_poses(time)
        self.env.unwrapped.data.qpos[-self.ref_skel_dof:] = ref_pos # ref_pos]
        self.env.unwrapped.data.qvel[-(self.ref_skel_dof-1):] *= 0.0
        obs[18:36] = ref_pos
        self.ref_pos = ref_pos.copy()

    def custom_pd_actuator(self, target_offset):
        torque = np.zeros(self.sim_skel_dof - 7) # minus root dof 
        # TODO : Implement your own PD actuator with reference trajectory, if you want to use
        # ex) ** Below is the conceptual logic. Be cautious with rotation operations ** 
        # - pd_target = self.ref_pos.copy()[7:] + target_offset
        # - delta_pos = pd_target - self.env.unwrapped.data.qpos[7:self.sim_skel_dof]
        # - delta_vel = -self.env.unwrapped.data.qvel[7:self.sim_skel_dof]
        # - torque = Kp * delta_pos + Kd * delta_vel
        # - return torque
        return torque

    def custom_terminated(self, terminated):
        # TODO: Implement your own termination condition
        return terminated
    
    def custom_truncated(self, truncated):
        # TODO: Implement your own truncation condition
        return truncated

    def custom_observation(self, obs):
        # Remove reference skeleton velocities (last 16 elements) — always zero, not informative.
        obs = obs[:-(self.ref_skel_dof - 1)] 
        # Step 1: Extract components
        root_pos = obs[0:3]
        # Step 2: Compute relative root position (w.r.t. reference position)
        obs[0:3] = self.ref_pos[:3] - root_pos
        
        return obs


    def custom_reward(self, obs):
        # Pose matching
        sim_joint_angles = obs[7:18]
        ref_joint_angles = obs[25:36]
        pose_diff = ref_joint_angles - sim_joint_angles
        pose_reward = np.exp(-0.1 * np.sum(np.square(pose_diff)))

        # Root position matching
        rel_root_pos = obs[-3:]
        root_pos_reward = np.exp(-0.1 * np.linalg.norm(rel_root_pos))

        # Combine multiplicatively
        reward = pose_reward * root_pos_reward 

        return reward

            

## Test Rendering
if __name__ == "__main__":
    env = CustomEnvWrapper()
    obs, _ = env.reset()
    
    for _ in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        # if terminated:
        #     obs = env.reset()
