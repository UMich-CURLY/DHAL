#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rsl_rl.storage import RolloutStorage

class PPO_HDS:

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        glide_advantage_w = 0.35,
        push_advantage_w = 0.4,
        sim2real_advantage_w = 0.25,
        device="cpu",
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # multi critic weight
        self.glide_advantage_w = glide_advantage_w
        self.push_advantage_w = push_advantage_w
        self.sim2real_advantage_w = sim2real_advantage_w

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, contact_shape, obs_future_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(
            num_envs, 
            num_transitions_per_env, 
            actor_obs_shape, 
            contact_shape,
            obs_future_shape,
            critic_obs_shape, 
            action_shape, 
            self.glide_advantage_w,
            self.push_advantage_w,
            self.sim2real_advantage_w,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values, self.transition.glide_values, self.transition.push_values, self.transition.reg_values  = self.actor_critic.evaluate(critic_obs)
        self.transition.values.detach(), self.transition.glide_values.detach(), self.transition.push_values.detach(), self.transition.reg_values.detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, obs_future, contact_future, rewards, glide_rewards, push_rewards, reg_reward, dones, infos):
        self.transition.obs_future = obs_future
        self.transition.contact_future = contact_future
        self.transition.rewards = rewards.clone()
        self.transition.glide_rewards = glide_rewards.clone()
        self.transition.push_rewards = push_rewards.clone()
        self.transition.reg_rewards = reg_reward.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
            self.transition.glide_rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
            self.transition.push_rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
            self.transition.reg_rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values, last_glide_values, last_push_values, last_reg_values = self.actor_critic.evaluate(last_critic_obs)
        last_values.detach(), last_glide_values.detach(), last_push_values.detach(), last_reg_values.detach()
        self.storage.compute_returns(last_values, last_glide_values, last_push_values, last_reg_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0  
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            obs_future_batch,
            contact_future_batch,
            critic_obs_batch,
            actions_batch,
            _,target_glide_values_batch, target_push_values_batch, target_reg_values_batch,
            advantages_batch,
            returns_batch, glide_returns_batch, push_returns_batch, reg_returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch, glide_value_batch, push_value_batch, reg_value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            mode_latent, prob = self.actor_critic.DHA(obs_batch)
            DHA_entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=-1)
            DHA_entropy = torch.mean(DHA_entropy)
            recon_loss_list = []
            recon_contact_loss_list = []
            mu_list = []
            logvar_list = []

            for i, sub_net in enumerate(self.actor_critic.TsDyn_modules):
                recon_x, recon_contact, mu, logvar = sub_net.forward(obs_batch)
                mu_list.append(mu)
                logvar_list.append(logvar)
                recon_loss = F.mse_loss(recon_x, obs_future_batch, reduction='none')
                recon_loss = recon_loss.sum(dim=1).unsqueeze(1).unsqueeze(2)
                recon_loss_list.append(recon_loss)
                recon_contact_loss = F.binary_cross_entropy(recon_contact, contact_future_batch, reduction='none')
                recon_contact_loss = recon_contact_loss.sum(dim=1).unsqueeze(1).unsqueeze(2)  
                recon_contact_loss_list.append(recon_contact_loss)

            recon_loss = torch.cat(recon_loss_list, dim=1)
            recon_loss = torch.bmm(mode_latent.unsqueeze(1), recon_loss).squeeze(1)
            recon_loss = torch.mean(recon_loss)
            recon_contact_loss = torch.cat(recon_contact_loss_list, dim=1)
            recon_contact_loss = torch.bmm(mode_latent.unsqueeze(1), recon_contact_loss).squeeze(1)
            recon_contact_loss = torch.mean(recon_contact_loss)
            mu = torch.stack(mu_list, dim=1)
            logvar = torch.stack(logvar_list, dim=1)
            kl_div = -0.5 * torch.sum(1 + logvar - (mu ** 2) - logvar.exp(), dim=-1) * mode_latent
            kl_div = torch.mean(kl_div)

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:

                # glide value loss
                glide_value_clipped = target_glide_values_batch + (glide_value_batch - target_glide_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                glide_value_losses = (glide_value_batch - glide_returns_batch).pow(2)
                glide_value_losses_clipped = (glide_value_clipped - glide_returns_batch).pow(2)
                glide_value_loss = torch.max(glide_value_losses, glide_value_losses_clipped).mean()

                # push value loss
                push_value_clipped = target_push_values_batch + (push_value_batch - target_push_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                push_value_losses = (push_value_batch - push_returns_batch).pow(2)
                push_value_losses_clipped = (push_value_clipped - push_returns_batch).pow(2)
                push_value_loss = torch.max(push_value_losses, push_value_losses_clipped).mean()

                # reg value loss
                reg_value_clipped = target_reg_values_batch + (reg_value_batch - target_reg_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                reg_value_losses = (reg_value_batch - reg_returns_batch).pow(2)
                reg_value_losses_clipped = (reg_value_clipped - reg_returns_batch).pow(2)
                reg_value_loss = torch.max(reg_value_losses, reg_value_losses_clipped).mean()

                value_loss = glide_value_loss + push_value_loss + reg_value_loss
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            vae_loss = recon_loss + recon_contact_loss * 0.8 + kl_div * 0.9
            loss = surrogate_loss + self.value_loss_coef * value_loss  - self.entropy_coef * entropy_batch.mean() + vae_loss + self.entropy_coef * DHA_entropy

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss

