# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch, torchvision
from torch import Tensor
from typing import Tuple, Dict
import random

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.terrain_parkour.terrain import Terrain_Parkour
from legged_gym.utils.terrain_parkour import get_terrain_cls
from legged_gym.utils.math import *
from legged_gym.utils.helpers import class_to_dict
from scipy.spatial.transform import Rotation as R
from .legged_robot_config import LeggedRobotCfg
from .curriculum import RewardThresholdCurriculum

from tqdm import tqdm

def euler_from_quaternion(quat_angle):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = torch.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = torch.clip(t2, -1, 1)
        pitch_y = torch.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = torch.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        self.resize_transform = torchvision.transforms.Resize((self.cfg.depth.resized[1], self.cfg.depth.resized[0]), 
                                                              interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        
        self._init_command_distribution(torch.arange(self.num_envs, device=self.device))
        
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        self.global_counter = 0
        self.total_env_steps_counter = 0

        self.glide_rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.push_rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reg_rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)


        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.post_physics_step()

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        actions.to(self.device)
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)

        self.global_counter += 1
        self.total_env_steps_counter += 1
        clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()
        delay_steps = torch.randint(0, self.cfg.control.decimation, (self.num_envs, 1), device=self.device)


        for i in range(self.cfg.control.decimation):
            delayed_actions = self.last_actions + (self.actions - self.last_actions) * (i >= delay_steps) 
            self.torques = self._compute_torques(delayed_actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self._against_rolling_friction()
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.contact_filt[:,[0,2,3]], self.current_obs_buf, self.privileged_obs_buf, self.rew_buf, self.glide_rew_buf, self.push_rew_buf, self.reg_rew_buf, self.reset_buf, self.extras

    def get_history_observations(self):
        return self.obs_history_buf

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.skate_quat = self.rigid_body_states[:,self.skateboard_deck_indices, 3:7].clone().view(self.num_envs, 4)
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt


        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        self.skate_roll, self.skate_pitch, self.skate_yaw = euler_from_quaternion(self.skate_quat)

        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.
        wheel_contact = torch.logical_or(torch.norm(self.contact_forces[:, self.wheel_link_indices], dim=-1) > 1.
                                         , torch.abs(self.rigid_body_states[:, self.wheel_link_indices, 2] - self.cfg.asset.wheel_radius) < 0.005)
        
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.wheel_contact_filt = torch.logical_or(wheel_contact, self.last_wheel_contacts)
        self.last_contacts = contact
        self.last_wheel_contacts = wheel_contact
        
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self.compute_observations()
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self._draw_height_samples()
            self._draw_marker()

    def check_termination(self):
        """ Check if environments need to be reset
        """ 
        self.reset_buf = torch.zeros((self.num_envs, ), dtype=torch.bool, device=self.device)
        roll_cutoff = torch.abs(self.roll) > 1.3
        pitch_cutoff = torch.abs(self.pitch) > 1.3
        skateboard_yaw_cutoff = torch.abs(self.yaw-self.skate_yaw) > 0.8
        skateboard_roll_cutoff = torch.abs(self.skate_roll) > 1.2
        skateboard_pitch_cutoff = torch.abs(self.skate_pitch) > 1.2

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs

        skateb_contact_num_4, ground_contact_num_4 = self._get_skateboard_contact()
        self.no_glide_time = (self.no_glide_time + self.dt) * (~skateb_contact_num_4.view(-1,1))
        self.no_push_time = (self.no_push_time + self.dt) * (~ground_contact_num_4.view(-1,1))
        no_glide_cutoff = (self.no_glide_time > 2.5).squeeze(1)
        no_push_cutoff = (self.no_push_time > 2.5).squeeze(1)
        self.no_glide_time = self.no_glide_time * ~no_glide_cutoff.unsqueeze(1)
        self.no_push_time = self.no_push_time * ~no_push_cutoff.unsqueeze(1)

        self.reset_buf |= self.time_out_buf
        self.reset_buf |= roll_cutoff
        self.reset_buf |= pitch_cutoff
        self.reset_buf |= skateboard_yaw_cutoff
        self.reset_buf |= skateboard_roll_cutoff
        self.reset_buf |= skateboard_pitch_cutoff

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self._resample_contact_phases()
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0

        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_root_vel[:] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.
        self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        self.no_glide_time [env_ids] = 0
        self.no_push_time[env_ids] = 0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.episode_length_buf[env_ids] = 0

        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        self.glide_rew_buf[:] = 0.
        self.push_rew_buf[:] = 0.
        self.reg_rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            reward_name = "_reward_" + name

            if reward_name in self.glide_reward_names:
                glide_rew = self.reward_functions[i]() * self.reward_scales[name]
                self.glide_rew_buf += glide_rew
            elif reward_name in self.push_reward_names:
                push_rew = self.reward_functions[i]() * self.reward_scales[name]
                self.push_rew_buf += push_rew
            elif reward_name in self.reg_reward_names:
                reg_reward = self.reward_functions[i]() * self.reward_scales[name]
                self.reg_rew_buf += reg_reward

            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
            self.glide_rew_buf[:] = torch.clip(self.glide_rew_buf[:], min=0.)
            self.push_rew_buf[:] = torch.clip(self.push_rew_buf[:], min=0.)
            self.reg_rew_buf[:] = torch.clip(self.reg_rew_buf[:], min=0.)

    
    def _get_info(self):
        infos = {
            "joint_pos": self.dof_pos[self.lookat_id,self.actuated_dof_indices].cpu().detach().numpy(),
            "joint_vel": self.dof_vel[self.lookat_id,self.actuated_dof_indices].cpu().detach().numpy(),
            "body_angular_vel":self.base_ang_vel[self.lookat_id, :].cpu().detach().numpy(),
            "body_linear_vel_cmd": self.commands[self.lookat_id, 0:2].cpu().detach().numpy(),
            "body_angular_vel_cmd": self.commands[self.lookat_id, 2:].cpu().detach().numpy(),
            "phase": self._get_phase().view(self.num_envs,1)[self.lookat_id,:].cpu().detach().numpy(),
            }
        return infos            
    
    def compute_observations(self):
        """ 
        Computes observations
        """
        imu_obs = torch.stack((self.roll, self.pitch,self.yaw), dim=1)
        phase = self._get_phase().view(self.num_envs,1)
        phase = (torch.norm(self.commands[:, :3],dim=1)>0.1).view(self.num_envs,1) * phase
        obs_buf =  torch.cat((      imu_obs[:,:2],
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos[:,self.actuated_dof_indices] - self.default_dof_pos[:,self.actuated_dof_indices]) * self.obs_scales.dof_pos,
                                    self.dof_vel[:,self.actuated_dof_indices] * self.obs_scales.dof_vel,
                                    self.actions,
                                    phase
                                    ),dim=-1)
        
        self.current_obs_buf =  torch.cat((
                                    imu_obs[:,:2],
                                    self.base_ang_vel * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos[:,self.actuated_dof_indices] - self.default_dof_pos[:,self.actuated_dof_indices]) * self.obs_scales.dof_pos,
                                    self.dof_vel[:,self.actuated_dof_indices] * self.obs_scales.dof_vel,
                                    phase
                                    ),dim=-1)
        
        if self.cfg.noise.add_noise:
            noised_imu_obs = imu_obs[:,:2] + torch.randn_like(imu_obs[:,:2]) * self.cfg.noise.noise_scales.imu
            noised_base_ang_vel = self.base_ang_vel.clone() + torch.randn_like(self.base_ang_vel) * self.cfg.noise.noise_scales.base_ang_vel
            noised_projected_gravity = self.projected_gravity.clone() + torch.randn_like(self.projected_gravity) * self.cfg.noise.noise_scales.gravity
            noised_dof_pos = self.dof_pos[:,self.actuated_dof_indices].clone() + torch.randn_like(self.dof_pos[:,self.actuated_dof_indices]) * self.cfg.noise.noise_scales.dof_pos
            noised_dof_vel = self.dof_vel[:,self.actuated_dof_indices].clone() + torch.randn_like(self.dof_vel[:,self.actuated_dof_indices]) * self.cfg.noise.noise_scales.dof_vel

            obs_buf_noised =  torch.cat((      noised_imu_obs[:,:2],
                                        noised_base_ang_vel  * self.obs_scales.ang_vel,
                                        noised_projected_gravity,
                                        self.commands[:, :3] * self.commands_scale,
                                        (noised_dof_pos - self.default_dof_pos[:,self.actuated_dof_indices]) * self.obs_scales.dof_pos,
                                        noised_dof_vel * self.obs_scales.dof_vel,
                                        self.actions,
                                        phase
                                        ),dim=-1)


        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, -1, 1.)

        self.contact_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([
                self.contact_buf[:, 1:],
                self.contact_filt.float().unsqueeze(1)
            ], dim=1)
        )          
        contact_buf = self.contact_buf.view(self.num_envs, -1)

        self.obs_buf = torch.cat([self.obs_history_buf.view(self.num_envs, -1), obs_buf_noised], dim=-1)

        # calculate distance
        feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
        feet_pos[:,2] = feet_pos[:,2] - 0.014
        skateb_contact_pos = self.rigid_body_states[:, self.marker_link_indices, :3]
        ground_contact_pos = self.rigid_body_states[:, self.feet_indices, :3] 
        ground_contact_pos[:, 0, -1] = 0 
        ground_contact_pos[:, 2, -1] = 0 

        dis_feet_skateb = (skateb_contact_pos - feet_pos).view(self.num_envs, -1)
        dis_feet_ground = (ground_contact_pos - feet_pos).view(self.num_envs, -1)
        dis_body_skateb = (skateb_contact_pos - self.root_states[:, None, :3]).view(self.num_envs, -1)
        dis_bofy_feet = (feet_pos - self.root_states[:, None, :3]).view(self.num_envs, -1)

        self.privileged_obs_buf = torch.cat([obs_buf,  
                                            self.base_lin_vel * self.obs_scales.lin_vel, 
                                            self.mass_params_tensor,
                                            self.friction_coeffs_tensor,
                                            self.motor_strength[0] - 1, 
                                            self.motor_strength[1] - 1,
                                            contact_buf,
                                            heights, 
                                            dis_feet_skateb * 0.1,
                                            dis_feet_ground * 0.1,
                                            dis_body_skateb * 0.1,
                                            dis_bofy_feet * 0.1,
                                            self.skate_roll.clone().view(self.num_envs, -1), 
                                            self.skate_pitch.clone().view(self.num_envs, -1), 
                                            self.skate_yaw.clone().view(self.num_envs, -1),

                                            self.obs_history_buf.view(self.num_envs, -1)], dim=-1)


        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf_noised] * (self.cfg.env.history_len-1), dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                obs_buf_noised.unsqueeze(1)
            ], dim=1)
        )

    def get_noisy_measurement(self, x, scale):
        if self.cfg.noise.add_noise:
            x = x + (2.0 * torch.rand_like(x) - 1) * scale * self.cfg.noise.noise_level
        return x

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        start = time()
        print("*"*80)
        print("Start creating ground...")
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type=='parkour':
            print("load parkour terrain")
            self._create_terrain()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        
        print("Finished creating ground. Time taken {:.2f} s".format(time() - start))
        print("*"*80)
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        props[18].rolling_friction = 0
        props[19].rolling_friction = 0
        props[21].rolling_friction = 0
        props[22].rolling_friction = 0
        # props[]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # No need to use tensors as only called upon env creation
        if self.cfg.domain_rand.randomize_base_mass:
            rng_mass = self.cfg.domain_rand.added_mass_range
            rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))
            props[0].mass += rand_mass
        else:
            rand_mass = np.zeros((1, ))
        if self.cfg.domain_rand.randomize_base_com:
            rng_com = self.cfg.domain_rand.added_com_range
            rand_com = np.random.uniform(rng_com[0], rng_com[1], size=(3, ))
            props[0].com += gymapi.Vec3(*rand_com)
        else:
            rand_com = np.zeros(3)
        mass_params = np.concatenate([rand_mass, rand_com])
        return props, mass_params
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0)
        self._resample_commands(env_ids.nonzero(as_tuple=False).flatten())
        self._resample_contact_phases()

        if self.cfg.terrain.measure_heights:
            if self.global_counter % self.cfg.depth.update_interval == 0:
                self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    
    def _resample_contact_phases(self):
        # based on phase
        self.last_contact_phase = self.contact_phase.clone()
        phase = self._get_phase()
        still = (torch.norm(self.commands[:, 0:3], dim= -1) < 0.15)
        self.contact_phase[:,0] = torch.logical_or((phase < 0.5), still) # glide
        self.contact_phase[:,1] = (phase >= 0.5) * ~still# push

        self.contact_phase = self.contact_phase * 0.35 + self.last_contact_phase * 0.75
        
    def _get_phase(self):
        phase = self.episode_length_buf * self.dt / self.cfg.rewards.cycle_time
        phase = torch.sin(2 * torch.pi * phase)
        return phase

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """

        if len(env_ids) == 0: return
        
        if self.cfg.commands.curriculum:
            timesteps = int(self.cfg.commands.resampling_time / self.dt)
            ep_len = min(self.cfg.env.max_episode_length, timesteps)
            lin_vel_rewards = self.command_sums["tracking_lin_vel"][env_ids] / ep_len
            ang_vel_rewards = self.command_sums["tracking_ang_vel"][env_ids] / ep_len
            lin_vel_threshold = self.cfg.commands.forward_curriculum_threshold * self.reward_scales["tracking_lin_vel"]
            ang_vel_threshold = self.cfg.commands.yaw_curriculum_threshold * self.reward_scales["tracking_ang_vel"]

            old_bins = self.env_command_bins[env_ids.cpu().numpy()]

            # update step just uses train env performance (for now)
            self.curriculum.update(old_bins[env_ids.cpu().numpy() < self.num_train_envs],
                                lin_vel_rewards[env_ids < self.num_train_envs].cpu().numpy(),
                                ang_vel_rewards[env_ids < self.num_train_envs].cpu().numpy(), lin_vel_threshold,
                                ang_vel_threshold, local_range=0.5, )

            new_commands, new_bin_inds = self.curriculum.sample(batch_size=len(env_ids))

            self.env_command_bins[env_ids.cpu().numpy()] = new_bin_inds
            self.commands[env_ids, :3] = torch.Tensor(new_commands).to(self.device)

            self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

            # reset command sums
            for key in self.command_sums.keys():
                self.command_sums[key][env_ids] = 0.

        else:
            self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)

            if self.cfg.commands.heading_command:
                self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            else:
                self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
                self.commands[env_ids, 2] *= torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_clip
            # set small commands to zero
            self.commands[env_ids, :2] *= torch.abs(self.commands[env_ids, 0:1]) > self.cfg.commands.lin_vel_clip

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        if control_type=="P":
            if not self.cfg.domain_rand.randomize_motor:  # TODO add strength to gain directly
                torques[:,self.actuated_dof_indices] = self.p_gains[self.actuated_dof_indices]*(actions_scaled + self.default_dof_pos_all[:,self.actuated_dof_indices] - self.dof_pos[:,self.actuated_dof_indices])- self.d_gains[self.actuated_dof_indices]*self.dof_vel[:,self.actuated_dof_indices]
                torques[:,self.underact_dof_indices] = self.p_gains[self.underact_dof_indices]*(self.default_dof_pos_all[:,self.underact_dof_indices] - self.dof_pos[:,self.underact_dof_indices]) - self.d_gains[self.underact_dof_indices]*self.dof_vel[:,self.underact_dof_indices]

            else:
                torques[:,self.actuated_dof_indices] = self.motor_strength[0] * self.p_gains[self.actuated_dof_indices]*(actions_scaled + self.default_dof_pos_all[:,self.actuated_dof_indices] - self.dof_pos[:,self.actuated_dof_indices]) - self.motor_strength[1] * self.d_gains[self.actuated_dof_indices]*self.dof_vel[:,self.actuated_dof_indices]
                torques[:,self.underact_dof_indices] = self.p_gains[self.underact_dof_indices]*(self.default_dof_pos_all[:,self.underact_dof_indices] - self.dof_pos[:,self.underact_dof_indices]) - self.d_gains[self.underact_dof_indices]*self.dof_vel[:,self.underact_dof_indices]
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_pos[env_ids[0::2]] = self.glide_default_dof_pos
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            if self.cfg.env.randomize_start_pos:
                self.root_states[env_ids, :2] += torch_rand_float(-0.3, 0.3, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            if self.cfg.env.randomize_start_yaw:
                rand_yaw = self.cfg.env.rand_yaw_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                if self.cfg.env.randomize_start_pitch:
                    rand_pitch = self.cfg.env.rand_pitch_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                else:
                    rand_pitch = torch.zeros(len(env_ids), device=self.device)
                quat = quat_from_euler_xyz(0*rand_yaw, rand_pitch, rand_yaw) 
                self.root_states[env_ids, 3:7] = quat[:, :]  
            if self.cfg.env.randomize_start_y:
                self.root_states[env_ids, 1] += self.cfg.env.rand_y_range * torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
            
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _get_skateboard_contact(self):
        feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
        feet_pos[:,2] = feet_pos[:,2] - 0.014 # z axis of the contact point of the foot

        # set contact of still and glide
        skateb_contact_pos = self.rigid_body_states[:, self.marker_link_indices, :3]

        # set contact of pushing
        ground_contact_pos = self.rigid_body_states[:, self.feet_indices, :3] 
        ground_contact_pos[:, 0, -1] = 0 
        ground_contact_pos[:, 2, -1] = 0 

        skateb_contact = (torch.norm(skateb_contact_pos - feet_pos, dim=-1) < 0.03)
        ground_contact = (torch.norm(ground_contact_pos - feet_pos, dim=-1) < 0.03)

        self.skateb_contact = skateb_contact
        self.ground_contact = ground_contact

        skateb_contact_num_4 = (torch.sum(skateb_contact, dim=-1) == 4)
        ground_contact_num_4 = (torch.sum(ground_contact, dim=-1) == 4)

        return skateb_contact_num_4, ground_contact_num_4


    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            return

        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5)
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level-1,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self.env_class[env_ids] = self.terrain_class[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        
        temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
        last_col = temp[:, -1].unsqueeze(1)
        self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.current_contact_goal = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.long, requires_grad=False) # store indices

            
        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions=torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

        str_rng = self.cfg.domain_rand.motor_strength_range
        self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(2, self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) + str_rng[0]
        self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len-1, self.cfg.env.n_proprio, device=self.device, dtype=torch.float)
        self.action_history_buf = torch.zeros(self.num_envs, self.cfg.domain_rand.action_buf_len, self.num_actions, device=self.device, dtype=torch.float)
        self.contact_buf = torch.zeros(self.num_envs, self.cfg.env.contact_buf_len, 4, device=self.device, dtype=torch.float)

        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.contact_phase = torch.zeros(self.num_envs, self.cfg.contact_phase.num_contact_phase, dtype=torch.float, device=self.device, requires_grad=False)
        self._resample_commands(torch.arange(self.num_envs, device=self.device, requires_grad=False))
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_wheel_contacts = torch.zeros(self.num_envs, self.wheel_link_indices.shape[0], dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.no_push_time = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.no_glide_time = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.glide_default_dof_pos = torch.tensor(self.cfg.init_state.glide_default_pos, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos_all = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            print(name)
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.default_dof_pos_all[:] = self.default_dof_pos[0]

        self.height_update_interval = 1
        if hasattr(self.cfg.env, "height_update_dt"):
            self.height_update_interval = int(self.cfg.env.height_update_dt / (self.cfg.sim.dt * self.cfg.control.decimation))


    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.glide_reward_functions = []
        self.push_reward_functions = []
        self.reg_reward_functions = []
        self.reward_names = []
        self.glide_reward_names = []
        self.push_reward_names = []
        self.reg_reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))
            if "glide" in name:
                self.glide_reward_names.append(name)
                self.glide_reward_functions.append(getattr(self, name))
            elif "push" in name:
                self.push_reward_names.append(name)
                self.push_reward_functions.append(getattr(self, name))
            elif "reg" in name:
                self.reg_reward_names.append(name)
                self.reg_reward_functions.append(getattr(self, name))
            name = '_reward_' + name

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.cfg.terrain.horizontal_scale
        hf_params.row_scale = self.cfg.terrain.horizontal_scale
        hf_params.vertical_scale = self.cfg.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.border
        hf_params.transform.p.y = -self.terrain.border
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples.flatten(order='C'), hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
            Very slow when horizontal_scale is small
        """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        print("Adding trimesh to simulation...")
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)  
        print("Trimesh added")
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        self.x_edge_mask = torch.tensor(self.terrain.x_edge_mask).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_terrain(self):
        terrain_cls = self.cfg.terrain_parkour.selected
        self.terrain = get_terrain_cls(terrain_cls)(self.cfg.terrain_parkour, self.num_envs)
        self.terrain.add_terrain_to_sim(self.gym, self.sim, self.device)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _against_rolling_friction(self):
        self.push_force = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.skateboard_lin_vel = quat_rotate_inverse(self.rigid_body_states[:, self.skateboard_deck_indices, 3:7].view(self.num_envs, 4), 
                                                      self.rigid_body_states[:, self.skateboard_deck_indices, 7:10].view(self.num_envs, 3))
        push_force = random.uniform(10, 25)
        self.push_force[:,self.skateboard_deck_indices, 0] = (push_force * (self.skateboard_lin_vel[:, 0]>0.3).float()).view(self.num_envs,1) - (push_force * (self.skateboard_lin_vel[:, 0]<-0.3).float()).view(self.num_envs,1)
        self.gym.apply_rigid_body_force_tensors(self.sim, forceTensor=gymtorch.unwrap_tensor(self.push_force), space=gymapi.CoordinateSpace.LOCAL_SPACE)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]

        print("\n", feet_names, "\n")


        for s in ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]:
            feet_idx = self.gym.find_asset_rigid_body_index(robot_asset, s)
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
            self.gym.create_asset_force_sensor(robot_asset, feet_idx, sensor_pose)
        
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
        hip_names = []
        for name in self.cfg.asset.hip_names:
            hip_names.extend([s for s in self.dof_names if name in s])
        thigh_names = []
        for name in self.cfg.asset.thigh_names:
            thigh_names.extend([s for s in body_names if name in s])
        calf_names = []
        for name in self.cfg.asset.calf_names:
            calf_names.extend([s for s in body_names if name in s])

        actuated_dof_names = []
        for name in self.cfg.asset.actuated_dof_names:
            actuated_dof_names.extend([s for s in self.dof_names if name in s])

        underact_dof_names = []
        for name in self.cfg.asset.underact_dof_names:
            underact_dof_names.extend([s for s in self.dof_names if name in s])

        undriven_dof_names = []
        for name in self.cfg.asset.undriven_dof_names:
            undriven_dof_names.extend([s for s in self.dof_names if name in s])

        skateboard_dof_names = []
        for name in self.cfg.asset.skateboard_dof_names:
            skateboard_dof_names.extend([s for s in self.dof_names if name in s])

        wheel_dof_names = []
        for name in self.cfg.asset.wheel_dof_names:
            wheel_dof_names.extend([s for s in self.dof_names if name in s])

        skateboard_deck_names = []
        for name in self.cfg.asset.skateboard_link_name:
            skateboard_deck_names.extend([s for s in body_names if name in s])

        wheel_link_names = []
        for name in self.cfg.asset.wheel_link_names:
            wheel_link_names.extend([s for s in body_names if name in s])

        marker_link_names = []
        for name in self.cfg.asset.marker_link_names:
            marker_link_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.cam_handles = []
        self.cam_tensors = []
        self.mass_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        
        print("Creating env...")
        for i in tqdm(range(self.num_envs)):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            if self.cfg.env.randomize_start_pos:
                pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            if self.cfg.env.randomize_start_yaw:
                rand_yaw_quat = gymapi.Quat.from_euler_zyx(0., 0., self.cfg.env.rand_yaw_range*np.random.uniform(-1, 1))
                start_pose.r = rand_yaw_quat
            start_pose.p = gymapi.Vec3(*(pos + self.base_init_state[:3]))

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            anymal_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "anymal", i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(anymal_handle)

            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device).to(torch.float)
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).to(torch.float).squeeze(-1)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])


        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_names):
            self.hip_indices[i] = self.dof_names.index(name)

        self.thigh_indices = torch.zeros(len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(thigh_names):
            self.thigh_indices[i] = self.dof_names.index(name)

        self.calf_indices = torch.zeros(len(calf_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(calf_names):
            self.calf_indices[i] = self.dof_names.index(name)

        self.actuated_dof_indices = torch.zeros(len(actuated_dof_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(actuated_dof_names):
            self.actuated_dof_indices[i] = self.dof_names.index(name)
        self.underact_dof_indices = torch.zeros(len(underact_dof_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(underact_dof_names):
            self.underact_dof_indices[i] = self.dof_names.index(name)
        self.undriven_dof_indices = torch.zeros(len(undriven_dof_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(undriven_dof_names):
            self.undriven_dof_indices[i] = self.dof_names.index(name)
        self.skateboard_dof_indices = torch.zeros(len(skateboard_dof_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(skateboard_dof_names):
            self.skateboard_dof_indices[i] = self.dof_names.index(name)

        self.skateboard_deck_indices = torch.zeros(len(skateboard_deck_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(skateboard_deck_names):
            self.skateboard_deck_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], skateboard_deck_names[i])

        self.wheel_link_indices = torch.zeros(len(wheel_link_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(wheel_link_names):
            self.wheel_link_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], wheel_link_names[i])

        self.wheel_dof_indices = torch.zeros(len(wheel_dof_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(wheel_dof_names):
            self.wheel_dof_indices[i] = self.dof_names.index(name)

        self.marker_link_indices = torch.zeros(len(marker_link_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(marker_link_names):
            self.marker_link_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], marker_link_names[i])
    
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            self.env_class = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
            
            self.terrain_class = torch.from_numpy(self.terrain.terrain_type).to(self.device).to(torch.float)
            self.env_class[:] = self.terrain_class[self.terrain_levels, self.terrain_types]

            self.terrain_goals = torch.from_numpy(self.terrain.goals).to(self.device).to(torch.float)
            self.env_goals = torch.zeros(self.num_envs, self.cfg.terrain.num_goals + self.cfg.env.num_future_goal_obs, 3, device=self.device, requires_grad=False)
            self.cur_goal_idx = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
            temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
            last_col = temp[:, -1].unsqueeze(1)
            self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]

        elif self.cfg.terrain.mesh_type in ["parkour"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            self.env_class = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain_parkour.max_init_terrain_level
            if not self.cfg.terrain_parkour.curriculum: max_init_level = self.cfg.terrain_parkour.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain_parkour.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain_parkour.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
            
            self.terrain.terrain_type = np.zeros((self.cfg.terrain_parkour.num_rows, self.cfg.terrain_parkour.num_cols))
            self.terrain_class = torch.from_numpy(self.terrain.terrain_type).to(self.device).to(torch.float)
            self.env_class[:] = self.terrain_class[self.terrain_levels, self.terrain_types]
            
            self.terrain.goals = np.zeros((self.cfg.terrain_parkour.num_rows, self.cfg.terrain_parkour.num_cols, self.cfg.terrain_parkour.num_goals, 3))
            self.terrain_goals = torch.from_numpy(self.terrain.goals).to(self.device).to(torch.float)
            self.env_goals = torch.zeros(self.num_envs, self.cfg.terrain_parkour.num_goals + self.cfg.env.num_future_goal_obs, 3, device=self.device, requires_grad=False)
            self.cur_goal_idx = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)

            temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
            last_col = temp[:, -1].unsqueeze(1)
            self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]

        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        reward_norm_factor = 1
        for rew in self.reward_scales:
            self.reward_scales[rew] = self.reward_scales[rew] / reward_norm_factor
        if self.cfg.commands.curriculum:
            self.command_ranges = class_to_dict(self.cfg.commands.curriculum_ranges)
        else:
            self.command_ranges = class_to_dict(self.cfg.commands.max_ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_height_samples(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if self.cfg.terrain.mesh_type == 'plane':
            return
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        i = self.lookat_id
        base_pos = (self.root_states[i, :3]).cpu().numpy()
        heights = self.measured_heights[i].cpu().numpy()
        height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
        for j in range(heights.shape[0]):
            x = height_points[j, 0] + base_pos[0]
            y = height_points[j, 1] + base_pos[1]
            z = heights[j]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
    
        
    def _draw_feet(self):
        if hasattr(self, 'feet_at_edge'):
            non_edge_geom = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0))
            edge_geom = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0, 0))

            feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
            for i in range(4):
                pose = gymapi.Transform(gymapi.Vec3(feet_pos[self.lookat_id, i, 0], feet_pos[self.lookat_id, i, 1], feet_pos[self.lookat_id, i, 2]), r=None)
                if self.feet_at_edge[self.lookat_id, i]:
                    gymutil.draw_lines(edge_geom, self.gym, self.viewer, self.envs[i], pose)
                else:
                    gymutil.draw_lines(non_edge_geom, self.gym, self.viewer, self.envs[i], pose)

    def _draw_marker(self):
        foot_geom = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0))
        marker_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(1, 0, 0))

        feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
        marker_pos = self.rigid_body_states[:, self.marker_link_indices, :3]
        
        for i in range(4):
            marker_pose = gymapi.Transform(gymapi.Vec3(marker_pos[self.lookat_id, i, 0], marker_pos[self.lookat_id, i, 1], marker_pos[self.lookat_id, i, 2]), r=None)
            gymutil.draw_lines(marker_geom, self.gym, self.viewer, self.envs[self.lookat_id], marker_pose)

            feet_pose = gymapi.Transform(gymapi.Vec3(feet_pos[self.lookat_id, i, 0], feet_pos[self.lookat_id, i, 1], feet_pos[self.lookat_id, i, 2]), r=None)
            gymutil.draw_lines(foot_geom, self.gym, self.viewer, self.envs[self.lookat_id], feet_pose)

    def _draw_mode(self, mode, color_list):
        num_mode = mode.shape[0]
        geom_list = []
        base_pos = self.root_states
        base_pos[:,2] = 0.4
        for i in range(num_mode):
            geom_list.append(gymutil.WireframeSphereGeometry(0.06, 48, 48, None, color=(color_list[i,0], color_list[i,1], color_list[i,2])))

        for i in range(num_mode):
            if mode[i] == 1:
                body_pos = gymapi.Transform(gymapi.Vec3(base_pos[self.lookat_id, 0], base_pos[self.lookat_id, 1], base_pos[self.lookat_id, 2]), r=None)
                gymutil.draw_lines(geom_list[i], self.gym, self.viewer, self.envs[self.lookat_id], body_pos)

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        for i in range(self.num_envs):
            offset = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points,2), device=self.device).squeeze()
            xy_noise = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points,2), device=self.device).squeeze() + offset
            points[i, :, 0] = grid_x.flatten() + xy_noise[:, 0]
            points[i, :, 1] = grid_y.flatten() + xy_noise[:, 1]
        return points

    def _init_command_distribution(self, env_ids):#todo
        self.curriculum = RewardThresholdCurriculum(seed=self.cfg.commands.curriculum_seed,
                                                    x_vel=(self.cfg.commands.limit_vel_x[0],
                                                           self.cfg.commands.limit_vel_x[1], 51),
                                                    y_vel=(self.cfg.commands.limit_vel_y[0],
                                                           self.cfg.commands.limit_vel_y[1], 2),
                                                    yaw_vel=(self.cfg.commands.limit_vel_yaw[0],
                                                             self.cfg.commands.limit_vel_yaw[1], 51))
        self.env_command_bins = np.zeros(len(env_ids), dtype=np.int)
        low = np.array(
            [self.cfg.commands.lin_vel_x[0], self.cfg.commands.lin_vel_y[0],
             self.cfg.commands.ang_vel_yaw[0]])
        high = np.array(
            [self.cfg.commands.lin_vel_x[1], self.cfg.commands.lin_vel_y[1],
             self.cfg.commands.ang_vel_yaw[1]])
        self.curriculum.set_to(low=low, high=high)


    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_heights_points(self, coords, env_ids=None):
        if env_ids:
            points = coords[env_ids]
        else:
            points = coords

        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    ##################  rewards ##################

    def _reward_tracking_goal_vel(self):
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        cur_vel = self.root_states[:, 7:9]
        rew = torch.minimum(torch.sum(target_vec_norm * cur_vel, dim=-1), self.commands[:, 0]) / (self.commands[:, 0] + 1e-5)
        return rew

    def _reward_push_tracking_lin_vel(self):
        contact_coefs = self.contact_phase.clone()
        lin_vel_error = torch.abs(self.commands[:, :2] - self.base_lin_vel[:, :2])
        lin_vel_error = torch.clip(lin_vel_error, 0.2, 1)
        lin_vel_error = torch.sum(torch.square(lin_vel_error), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma) * contact_coefs[:,1]

    def _reward_tracking_world_lin_vel(self):
        cur_vel = self.base_lin_vel[:, 0] * torch.cos(self.commands[:, 3]-self.yaw)
        lin_vel_error = torch.square(self.commands[:, 0] - cur_vel)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_yaw(self):
        rew = torch.exp(-torch.abs(self.commands[:, 3] - self.yaw))
        return rew
    
    def _reward_push_tracking_ang_vel(self):
        contact_coefs = self.contact_phase.clone()
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma_yaw) * contact_coefs[:,1]


    def _reward_reg_lin_vel_z(self):
        error = torch.clip(self.base_lin_vel[:, 2], -1.5, 1.5)
        rew = torch.square(error)
        return rew
    
    def _reward_reg_ang_vel_xy(self):
        error = torch.clip(self.base_ang_vel[:, :2], -1, 1)
        return torch.sum(torch.square(error), dim=1)
     
    def _reward_reg_orientation(self):
        rew = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return rew
    
    def _reward_push_orientation(self):
        contact_coefs = self.contact_phase.clone()
        rew = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return rew * contact_coefs[:,1]

    def _reward_reg_dof_acc(self):
        error = self.last_dof_vel[:,self.actuated_dof_indices] - self.dof_vel[:,self.actuated_dof_indices]
        error = torch.clip(error, -10,10)
        return torch.sum(torch.square(error / self.dt), dim=1)

    def _reward_reg_collision(self):
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_reg_action_rate(self):
        return torch.norm(self.last_actions - self.actions,dim=1)

    
    def _reward_smoothness(self):
        return torch.sum(torch.square(self.actions - 2*self.last_actions + self.last_last_actions), dim=1)

    def _reward_reg_delta_torques(self):
        return torch.sum(torch.square(self.torques[:,self.actuated_dof_indices] - self.last_torques[:,self.actuated_dof_indices]), dim=1)
    
    def _reward_reg_torques(self):
        return torch.sum(torch.square(self.torques[:,self.actuated_dof_indices]), dim=1)

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)
    
    def _reward_skateboard_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.skateboard_dof_indices]- self.default_dof_pos[:, self.skateboard_dof_indices]), dim=1)

    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.dof_pos[:, self.actuated_dof_indices] - 
                                           self.default_dof_pos[:, self.actuated_dof_indices]), dim=1)
        return dof_error
    
    def _reward_reg_wheel_contact_number(self):
        wheel_contact_number = torch.sum(self.wheel_contact_filt, dim=1)
        reward = wheel_contact_number == 4
        return reward
    
    def _reward_wheel_speed(self):
        wheel_speed = torch.abs(torch.sum(self.dof_vel[:,self.wheel_dof_indices], dim=1))
        return wheel_speed
    
    def _reward_feet_stumble(self):
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             4 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        return rew.float()

    def _reward_feet_edge(self):
        feet_pos_xy = ((self.rigid_body_states[:, self.feet_indices, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long() 
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0]-1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1]-1)
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
    
        self.feet_at_edge = self.contact_filt & feet_at_edge
        rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        return rew
    
    def _reward_base_height(self):
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_feet_air_time(self):
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) 
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 
        self.feet_air_time *= ~self.contact_filt
        return rew_airTime
    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
    
    def _reward_emergy_cost(self):
        emergy_cost = torch.max(torch.abs(self.torques * self.dof_vel) + 0.3*torch.square(self.torques),torch.zeros_like(self.torques))
        return torch.sum(emergy_cost,dim=1)
    
    def _reward_glide_feet_on_board(self):
        contact_coefs = self.contact_phase.clone()
        feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
        marker_pos = self.rigid_body_states[:, self.marker_link_indices, :3]
        feet_board_distance = torch.norm(marker_pos - feet_pos, dim=-1)
        reward = torch.sum(feet_board_distance < 0.05, dim=-1)
        return reward * contact_coefs[:,0]
    

    def _reward_glide_contact_num(self):
        contact_coefs = self.contact_phase.clone()

        feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
        feet_pos[:,:,2] = feet_pos[:,:,2] - 0.014

        skateb_contact_pos = self.rigid_body_states[:, self.marker_link_indices, :3]

        ground_contact_pos = self.rigid_body_states[:, self.feet_indices, :3] 
        ground_contact_pos[:, 0, -1] = 0 
        ground_contact_pos[:, 2, -1] = 0 
        ground_contact_pos[:, 3, :] = skateb_contact_pos[:, 3, :]

        skateb_feets_dis = torch.norm(skateb_contact_pos - feet_pos, dim=-1)
        ground_feets_dis = torch.norm(ground_contact_pos - feet_pos, dim=-1)

        skateb_contact = torch.logical_and((skateb_feets_dis < 0.06), torch.abs(feet_pos[:,:,2] - 0.115 + 0.014)< 0.02)
        ground_contact = ground_feets_dis < 0.06
        ground_contact[:, 3] = skateb_contact[:,3]

        skateb_contact_num_4 = (torch.sum(skateb_contact, dim=-1) == 4)
        ground_contact_num_4 = (torch.sum(ground_contact, dim=-1) == 4)

        reward = (torch.sum(skateb_contact[:,:], dim=-1)) + 4 * skateb_contact_num_4
        
        penalty = (torch.sum(~skateb_contact[:,[0,2]], dim=-1)
                +  torch.sum(ground_contact[:,[0,2]], dim=-1))
        
        reward = reward * contact_coefs[:,0]
        penalty = penalty * contact_coefs[:,0]
        
        return reward * 2 - penalty
    
    def _reward_push_feet_force(self):
        contact_coefs = self.contact_phase.clone()
        feet_z_force = self.contact_forces[:, self.feet_indices, 2]
        penalty = torch.abs(feet_z_force[:,2] - feet_z_force[:,3])
        return penalty * contact_coefs[:,1]
    
    def _reward_reg_rear_left_foot_force(self):
        feet_z_force = self.contact_forces[:, self.feet_indices, 2]
        RL_foot_force = feet_z_force[:,3]
        return RL_foot_force
    
    def _reward_reg_rele_skateboard_yaw(self):
        yaw_diff = torch.abs(self.skate_yaw - self.yaw)
        return yaw_diff

    def _reward_glide_feet_force(self):
        contact_coefs = self.contact_phase.clone()
        feet_z_force = self.contact_forces[:, self.feet_indices, 2]
        penalty = torch.max(feet_z_force[:,[0,2,3]], dim = 1)[0] - torch.min(feet_z_force[:,[0,2,3]], dim = 1)[0]
        return penalty * contact_coefs[:,0]
    
    def _reward_reg_smoothness(self):
        return torch.sum(torch.square(self.actions - 2*self.last_actions + self.last_last_actions), dim=1)


    def _reward_push_contact_num(self):
        contact_coefs = self.contact_phase.clone()

        feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
        feet_pos[:,:,2] = feet_pos[:,:,2] - 0.014

        skateb_contact_pos = self.rigid_body_states[:, self.marker_link_indices, :3]

        ground_contact_pos = self.rigid_body_states[:, self.feet_indices, :3] 
        ground_contact_pos[:, 0, -1] = 0 
        ground_contact_pos[:, 2, -1] = 0 
        ground_contact_pos[:, 3, :] = skateb_contact_pos[:, 3, :]

        skateb_feets_dis = torch.norm(skateb_contact_pos - feet_pos, dim=-1)
        ground_feets_dis = torch.norm(ground_contact_pos - feet_pos, dim=-1)

        skateb_contact = torch.logical_and((skateb_feets_dis < 0.06), torch.abs(feet_pos[:,:,2] - 0.115 + 0.014)< 0.02)
        ground_contact = ground_feets_dis < 0.06
        ground_contact[:, 3] = skateb_contact[:,3]

        skateb_contact_num_4 = (torch.sum(skateb_contact, dim=-1) == 4)
        ground_contact_num_4 = (torch.sum(ground_contact, dim=-1) == 4)

        reward = (torch.sum(ground_contact[:,:], dim=-1))
        
        penalty = (torch.sum(~ground_contact[:,[0,2]], dim=-1)
                +  torch.sum(skateb_contact[:,[0,2]], dim=-1))
        
        reward = reward * contact_coefs[:,1]
        penalty = penalty * contact_coefs[:,1]
        
        return reward
    
    def _reward_skate_pos(self):
        contact_coefs = self.contact_phase.clone().bool()
        push_dof_error = self.dof_pos[:,self.actuated_dof_indices] - self.default_dof_pos[self.actuated_dof_indices]
        push_dof_error = torch.norm(push_dof_error, dim = -1)
        glide_dof_error = self.dof_pos[:,self.actuated_dof_indices] - self.glide_default_dof_pos[:,self.actuated_dof_indices]
        glide_dof_error = torch.norm(glide_dof_error, dim = -1)

        push_pos_reward = torch.clip(contact_coefs[:,1] * push_dof_error  - contact_coefs[:,1] * glide_dof_error, 0, 30)
         
        return torch.exp(-(push_pos_reward)*4)
    
    def _reward_skate_hip(self):
        contact_coefs = self.contact_phase.clone().bool()
        glide_pos_hip_error = self.dof_pos[:, self.hip_indices] - self.glide_default_dof_pos[None, self.hip_indices]
        glide_pos_hip_error = torch.norm(glide_pos_hip_error, dim = -1)

        return glide_pos_hip_error * contact_coefs[:,1]
    
    def _reward_reg_board_body_z(self):
        skate_board_z = self.rigid_body_states[:, self.skateboard_deck_indices, 2].squeeze(1)
        body_z = self.root_states[:,2]
        return torch.exp(-torch.abs(body_z - skate_board_z -0.15)*4)

    def _reward_feet_on_board(self):
        contact_coefs = self.contact_phase.clone()
        feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
        marker_pos = self.rigid_body_states[:, self.marker_link_indices, :3]
        feet_board_distance = torch.norm(marker_pos - feet_pos, dim=-1)
        reward = torch.sum(feet_board_distance < 0.05, dim=-1)
        return reward * contact_coefs[:,0]

    def _reward_glide_feet_dis(self):
        # get contact phase
        contact_coefs = self.contact_phase.clone()

        feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
        feet_pos[:,2] = feet_pos[:,2] - 0.014

        # set contact of still and glide
        skateb_contact_pos = self.rigid_body_states[:, self.marker_link_indices, :3].clone()

        # set contact of pushing
        ground_contact_pos = self.rigid_body_states[:, self.feet_indices, :3] 
        ground_contact_pos[:, 0, -1] = 0 
        ground_contact_pos[:, 2, -1] = 0 
        ground_contact_pos[:, 3, :] = skateb_contact_pos[:, 3, :]

        skateb_contact_dis = torch.sum(torch.norm(skateb_contact_pos[:,[0,2,3],:] - feet_pos[:,[0,2,3],:], dim=-1),dim=-1)
        ground_contact_dis = torch.sum(torch.norm(ground_contact_pos[:,[0,2,3],:] - feet_pos[:,[0,2,3],:], dim=-1),dim=-1)
        
        reward = torch.exp(-skateb_contact_dis * 3) * contact_coefs[:,0]
        
        return reward
    
    def _reward_push_joint_pos(self):
        contact_coefs = self.contact_phase.clone()
        dof_error = torch.sum(torch.square(self.dof_pos[:, self.actuated_dof_indices] 
                                        - self.default_dof_pos[:, self.actuated_dof_indices]), dim=1)
        return torch.exp(-dof_error) * contact_coefs[:,1]

    def _reward_push_hip_pos(self):
        contact_coefs = self.contact_phase.clone()
        hip_error = torch.sum(torch.square(self.dof_pos[:, self.hip_indices] 
                        - self.default_dof_pos[:, self.hip_indices]), dim=1)
        return torch.exp(-hip_error)* contact_coefs[:,1]

    def _reward_glide_joint_pos(self):
        contact_coefs = self.contact_phase.clone()
        dof_error = torch.sum(torch.square(self.dof_pos[:, self.actuated_dof_indices] 
                                        - self.glide_default_dof_pos[None, self.actuated_dof_indices]), dim=1)
        return torch.exp(-dof_error) * contact_coefs[:,0]

    def _reward_glide_hip_pos(self):
        contact_coefs = self.contact_phase.clone()
        hip_error = torch.sum(torch.square(self.dof_pos[:, self.hip_indices] 
                        - self.glide_default_dof_pos[None, self.hip_indices]), dim=1)
        return torch.exp(-hip_error) * contact_coefs[:,0]