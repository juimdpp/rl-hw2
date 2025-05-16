from motion import Motion
import numpy as np
import gymnasium as gym
import os
import math

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
        self.timestep_counter = 0
        return custom_obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action) # or self.env.step(self.custom_pd_actuator(action))
        self.update_ref_pose(self.env.unwrapped.data.time, obs)
        custom_obs = self.custom_observation(obs)
        custom_reward = self.custom_reward(obs)
        custom_terminated = self.custom_terminated(terminated)
        custom_truncated = self.custom_truncated(truncated)
        
        # Add reward components to `info` dict
        info["pose_diff"] = getattr(self, "last_pose_diff", 0)
        info["root_diff"] = getattr(self, "last_root_diff", 0)
        info["vel_diff"] = getattr(self, "last_vel_diff", 0)
        self.timestep_counter += 1

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
        obs_wo_ref_vel = obs[:-self.ref_skel_dof]

        root_x = obs_wo_ref_vel[0]
        root_z = obs_wo_ref_vel[1]

        rel_root_x = self.ref_pos[0] - root_x
        rel_root_z = (self.ref_pos[1] + 1.25) - root_z
        obs_wo_ref_vel[0] = rel_root_x
        obs_wo_ref_vel[1] = rel_root_z
        # # Joint angles (6 DoF), relative to reference
        # sim_joint_angles = obs_wo_ref_vel[3:9]
        # ref_joint_angles = obs_wo_ref_vel[12:18]
        # joint_angle_diff = ref_joint_angles - sim_joint_angles

        # # Joint velocities (6 DoF)
        # sim_joint_vels = obs[21:27]
        # time = self.env.unwrapped.data.time
        # delta_time = 0.02
        # ref_joint_angles_current = self.ref_motion.get_ref_poses(time)[3:]
        # ref_joint_angles_next = self.ref_motion.get_ref_poses(time + delta_time)[3:]
        # ref_joint_vels = (ref_joint_angles_next - ref_joint_angles_current) / delta_time
        # joint_vel_diff = ref_joint_vels - sim_joint_vels

        # Final observation: relative root x/z, root angle, joint angle diff, joint vel diff
        mod_obs = np.concatenate((
            obs_wo_ref_vel,           
            ))

        return mod_obs

    def custom_reward(self, obs):
        sim_joint_angles = obs[3:9]
        ref_joint_angles = obs[12:18]
        joint_angle_diff = ref_joint_angles - sim_joint_angles
        pose_diff = np.square(joint_angle_diff)
        pose_reward = np.exp(-0.1 * np.sum(pose_diff))

        root_z_diff = (obs[1]) ** 2
        root_x_diff = (obs[0]) ** 2
        root_z_reward = np.exp(-0.1 * np.sqrt(root_z_diff + root_x_diff))
        # root_x_reward = np.exp(-0.1 * (root_x_diff))

        # sim_joint_vels = obs[21:27]
        # vel_diff = np.square(sim_joint_vels)
        # vel_reward = np.exp(-4 * np.sum(vel_diff))

        reward = (
             (pose_reward) *
            #0.3 * (1 + root_x_reward) *
             (root_z_reward) 
            # 0.7 * vel_reward
        )

        return reward


   
## Test
if __name__ == "__main__":
    env = CustomEnvWrapper()
    obs, _ = env.reset() 
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
