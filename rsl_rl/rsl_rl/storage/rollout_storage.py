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

import torch
import numpy as np

from rsl_rl.utils import split_and_pad_trajectories

class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.obs_future = None
            self.contact_future = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.glide_rewards = None
            self.push_rewards = None
            self.reg_rewards = None
            self.dones = None
            self.glide_values = None
            self.push_values = None
            self.reg_values = None            
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
        def clear(self):
            self.__init__()

    def __init__(self, num_envs, 
                 num_transitions_per_env, 
                 obs_shape, 
                 contact_shape,
                 obs_future_shape,
                 privileged_obs_shape, 
                 actions_shape,
                glide_advantage_w, push_advantage_w, sim2real_advantage_w, device='cpu'):

        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape
        self.glide_advantage_w = glide_advantage_w
        self.push_advantage_w = push_advantage_w
        self.sim2real_advantage_w = sim2real_advantage_w
        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.obs_future = torch.zeros(num_transitions_per_env, num_envs, *obs_future_shape, device=self.device)
        self.contact_future = torch.zeros(num_transitions_per_env, num_envs, *contact_shape, device=self.device)

        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.glide_rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.push_rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.reg_rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.glide_values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.push_values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.reg_values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.glide_rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.push_rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.reg_rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.glide_returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.push_returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.reg_returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None: self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.obs_future[self.step].copy_(transition.obs_future)
        self.contact_future[self.step].copy_(transition.contact_future)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.glide_rewards[self.step].copy_(transition.glide_rewards.view(-1, 1))
        self.push_rewards[self.step].copy_(transition.push_rewards.view(-1, 1))
        self.reg_rewards[self.step].copy_(transition.reg_rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.glide_values[self.step].copy_(transition.glide_values)
        self.push_values[self.step].copy_(transition.push_values)
        self.reg_values[self.step].copy_(transition.reg_values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)

        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states==(None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        # initialize if needed 
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))]
            self.saved_hidden_states_c = [torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])


    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, last_glide_values, last_push_values, last_reg_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_glide_values
            else:
                next_values = self.glide_values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.glide_rewards[step] + next_is_not_terminal * gamma * next_values - self.glide_values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.glide_returns[step] = advantage + self.glide_values[step]

        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_push_values
            else:
                next_values = self.push_values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.push_rewards[step] + next_is_not_terminal * gamma * next_values - self.push_values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.push_returns[step] = advantage + self.push_values[step]


        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_reg_values
            else:
                next_values = self.reg_values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.reg_rewards[step] + next_is_not_terminal * gamma * next_values - self.reg_values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.reg_returns[step] = advantage + self.reg_values[step]


        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

        self.glide_advantages = self.glide_returns - self.glide_values
        self.glide_advantages = (self.glide_advantages - self.glide_advantages.mean()) / (self.glide_advantages.std() + 1e-8)

        self.push_advantages = self.push_returns - self.push_values
        self.push_advantages = (self.push_advantages - self.push_advantages.mean()) / (self.push_advantages.std() + 1e-8)

        self.reg_advantages = self.reg_returns - self.reg_values
        self.reg_advantages = (self.reg_advantages - self.reg_advantages.mean()) / (self.reg_advantages.std() + 1e-8)

        self.advantages = self.glide_advantage_w * self.glide_advantages + self.push_advantage_w * self.push_advantages + self.sim2real_advantage_w * self.reg_advantages

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)

        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations
        contact_future = self.contact_future.flatten(0, 1)
        obs_future = self.obs_future.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        glide_values = self.glide_values.flatten(0, 1)
        push_values = self.push_values.flatten(0, 1)
        reg_values = self.reg_values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        glide_returns = self.glide_returns.flatten(0, 1)
        push_returns = self.push_returns.flatten(0, 1)
        reg_returns = self.reg_returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                obs_future_batch = obs_future[batch_idx]
                contact_future_batch = contact_future[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                glide_values_batch = glide_values[batch_idx]
                push_values_batch = push_values[batch_idx]
                reg_values_batch = reg_values[batch_idx]
                returns_batch = returns[batch_idx]
                glide_returns_batch = glide_returns[batch_idx]
                push_returns_batch = push_returns[batch_idx]
                reg_returns_batch = reg_returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                
                yield obs_batch, obs_future_batch, contact_future_batch, critic_observations_batch, actions_batch, \
                        target_values_batch, glide_values_batch, push_values_batch, reg_values_batch, advantages_batch, \
                        returns_batch, glide_returns_batch, push_returns_batch, reg_returns_batch,\
                        old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (None, None), None

    # for RNNs only
    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):

        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None: 
            padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else: 
            padded_critic_obs_trajectories = padded_obs_trajectories

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i*mini_batch_size
                stop = (i+1)*mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size
                
                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [ saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                                for saved_hidden_states in self.saved_hidden_states_a ] 
                hid_c_batch = [ saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                                for saved_hidden_states in self.saved_hidden_states_c ]
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch)==1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch)==1 else hid_a_batch

                yield obs_batch, critic_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (hid_a_batch, hid_c_batch), masks_batch
                
                first_traj = last_traj