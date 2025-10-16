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
import torch.nn as nn
import torch.optim as optim

from .mlp_encoder import MLP_Encoder
from .heightmap_encoder import HeightmapEncoder
from .actor_critic import ActorCritic
from .rollout_storage import RolloutStorage


class PPO:
    actor_critic: ActorCritic
    encoder: MLP_Encoder
    heightmap_encoder: HeightmapEncoder

    def __init__(
        self,
        num_group,
        encoder,
        heightmap_encoder,
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
        vae_beta=1.0,
        est_learning_rate=1.0e-3,
        ts_learning_rate=1.0e-4,
        critic_take_latent=False,
        early_stop=False,
        anneal_lr=False,
        device="cpu",
        **kwargs,
    ):
        self.device = device
        self.num_group = num_group

        self.desired_kl = desired_kl
        self.early_stop = early_stop
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.anneal_lr = anneal_lr
        self.vae_beta = vae_beta
        self.critic_take_latent = critic_take_latent
        self.critic_use_gt_heightmap = kwargs.get('critic_use_gt_heightmap', True)

        self.encoder = encoder
        self.heightmap_encoder = heightmap_encoder
        self.current_heightmaps = None  # Store current noisy heightmaps
        self.current_gt_heightmaps = None  # Store current GT heightmaps

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam([{"params": self.actor_critic.parameters()}], lr=learning_rate)

        # Setup optimizers for encoders
        encoder_params = []
        if self.encoder.num_output_dim != 0:
            encoder_params.extend(self.encoder.parameters())
        if self.heightmap_encoder.num_output_dim != 0:
            encoder_params.extend(self.heightmap_encoder.parameters())
        
        if encoder_params:
            self.extra_optimizer = optim.Adam(encoder_params, lr=est_learning_rate)
        else:
            self.extra_optimizer = None
            
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

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        obs_history_shape,
        commands_shape,
        action_shape,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            obs_history_shape,
            commands_shape,
            action_shape,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, obs_history, commands, critic_obs, heightmap=None, gt_heightmap=None):
        critic_obs = torch.cat((critic_obs, commands), dim=-1)
        # act
        encoder_out = self.encoder.encode(obs_history)
        
        # Store heightmaps for different purposes
        self.current_heightmaps = heightmap  # Noisy heightmap for actor
        self.current_gt_heightmaps = gt_heightmap  # GT heightmap for critic
        
        # Encode noisy heightmap for actor; if missing, use zeros to keep input dim consistent
        if self.heightmap_encoder.num_output_dim > 0:
            if heightmap is not None:
                heightmap_encoded = self.heightmap_encoder.encode(heightmap)
            else:
                heightmap_encoded = torch.zeros(
                    (obs.shape[0], self.heightmap_encoder.num_output_dim), device=self.device, dtype=obs.dtype
                )
            actor_input = torch.cat((encoder_out, obs, commands, heightmap_encoded), dim=-1)
        else:
            # No heightmap encoder - original behavior
            actor_input = torch.cat((encoder_out, obs, commands), dim=-1)
            
        self.transition.actions = self.actor_critic.act(actor_input).detach()

        # evaluate with GT heightmap for critic
        if self.critic_take_latent:
            if self.heightmap_encoder.num_output_dim > 0:
                if gt_heightmap is not None and getattr(self, 'critic_use_gt_heightmap', True):
                    gt_heightmap_encoded = self.heightmap_encoder.encode(gt_heightmap)
                    critic_latent = gt_heightmap_encoded
                elif heightmap is not None:
                    critic_latent = self.heightmap_encoder.encode(heightmap)
                else:
                    critic_latent = torch.zeros(
                        (critic_obs.shape[0], self.heightmap_encoder.num_output_dim), device=self.device, dtype=critic_obs.dtype
                    )
                critic_obs = torch.cat((critic_obs, encoder_out, critic_latent), dim=-1)
            else:
                # Heightmap encoder disabled - critic_obs already contains heightmap from environment
                # Just add encoder output
                critic_obs = torch.cat((critic_obs, encoder_out), dim=-1)
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()

        # storage
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_obs = critic_obs
        self.transition.observation_history = obs_history
        self.transition.commands = commands
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, next_obs=None):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # Record the transition
        self.transition.next_observations = next_obs
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        num_updates = 0
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_kl = 0
        generator = self.storage.mini_batch_generator(
            self.num_group,
            self.num_mini_batches,
            self.num_learning_epochs,
        )
        for (
            obs_batch,
            critic_obs_batch,
            obs_history_batch, _,
            group_commands_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:
            encoder_out_batch = self.encoder.encode(obs_history_batch)
            commands_batch = group_commands_batch
            
            # Add heightmap encoding for actor (use zeros since heightmap is not available in update)
            if self.heightmap_encoder.num_output_dim > 0:
                heightmap_encoded_batch = torch.zeros(
                    (obs_batch.shape[0], self.heightmap_encoder.num_output_dim), 
                    device=self.device, dtype=obs_batch.dtype
                )
                actor_input_batch = torch.cat(
                    (encoder_out_batch, obs_batch, commands_batch, heightmap_encoded_batch),
                    dim=-1,
                )
            else:
                # No heightmap encoder - original behavior
                actor_input_batch = torch.cat(
                    (encoder_out_batch, obs_batch, commands_batch),
                    dim=-1,
                )
            
            self.actor_critic.act(actor_input_batch)

            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )

            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            kl_mean = torch.tensor(0, device=self.device, requires_grad=False)
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (
                        torch.square(old_sigma_batch)
                        + torch.square(old_mu_batch - mu_batch)
                    )
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)

            # KL
            if self.desired_kl != None and self.schedule == "adaptive":
                with torch.inference_mode():
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            if self.desired_kl != None and self.early_stop:
                if kl_mean > self.desired_kl * 1.5:
                    print("early stop, num_updates =", num_updates)
                    break

            # Surrogate loss
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            # print(ratio)
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            entropy_batch_mean = entropy_batch.mean()
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch_mean
            )

            if self.anneal_lr:
                frac = 1.0 - num_updates / (
                    self.num_learning_epochs * self.num_mini_batches
                )
                self.optimizer.param_groups[0]["lr"] = frac * self.learning_rate

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            num_updates += 1
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_kl += kl_mean.item()

        num_updates_extra = 0
        mean_mlp_loss = 0
        mean_heightmap_loss = 0
        if self.extra_optimizer is not None:
            generator = self.storage.encoder_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
            for (
                next_obs_batch,
                critic_obs_batch,
                obs_history_batch,
            ) in generator:
                # MLP Encoder loss
                mlp_loss = torch.tensor(0.0, device=self.device)
                if self.encoder.is_mlp_encoder:
                    self.encoder.encode(obs_history_batch)
                    encode_batch = self.encoder.get_encoder_out()
                    mlp_loss = (
                        (encode_batch[:, 0:3] - critic_obs_batch[:, 0:3]).pow(2).mean()
                    )
                
                # MLP Encoder 학습 (단독)
                if mlp_loss.item() > 0:
                    self.extra_optimizer.zero_grad()
                    mlp_loss.backward()
                    if hasattr(self.encoder, 'parameters'):
                        nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
                    self.extra_optimizer.step()

                # Heightmap Encoder loss: noisy vs GT compressed output (단독)
                heightmap_loss = torch.tensor(0.0, device=self.device)
                if (self.heightmap_encoder.num_output_dim > 0 and 
                    hasattr(self, 'current_heightmaps') and 
                    hasattr(self, 'current_gt_heightmaps') and
                    self.current_heightmaps is not None and 
                    self.current_gt_heightmaps is not None):
                    # Sample a subset for memory efficiency
                    batch_size = min(self.current_heightmaps.shape[0], 256)
                    if self.current_heightmaps.shape[0] > batch_size:
                        indices = torch.randperm(self.current_heightmaps.shape[0])[:batch_size]
                        noisy_sample = self.current_heightmaps[indices]
                        gt_sample = self.current_gt_heightmaps[indices]
                    else:
                        noisy_sample = self.current_heightmaps
                        gt_sample = self.current_gt_heightmaps
                    
                    # Encode both noisy and GT heightmaps
                    noisy_encoded = self.heightmap_encoder.encode(noisy_sample)
                    gt_encoded = self.heightmap_encoder.encode(gt_sample)
                    
                    # Compute MSE loss between compressed representations
                    heightmap_loss = nn.functional.mse_loss(noisy_encoded, gt_encoded)
                    
                    # Heightmap Encoder 학습 (단독)
                    self.extra_optimizer.zero_grad()
                    heightmap_loss.backward()
                    if hasattr(self.heightmap_encoder, 'parameters'):
                        nn.utils.clip_grad_norm_(self.heightmap_encoder.parameters(), self.max_grad_norm)
                    self.extra_optimizer.step()

                num_updates_extra += 1
                mean_mlp_loss += mlp_loss.item()
                mean_heightmap_loss += heightmap_loss.item()

        # Prevent division by zero
        if num_updates > 0:
            mean_value_loss /= num_updates
            mean_surrogate_loss /= num_updates
            mean_kl /= num_updates
        else:
            print("Warning: No PPO updates performed due to early stopping")
            mean_value_loss = 0.0
            mean_surrogate_loss = 0.0
            mean_kl = 0.0
            
        if num_updates_extra > 0:
            mean_mlp_loss /= num_updates_extra
            mean_heightmap_loss /= num_updates_extra
        else:
            mean_mlp_loss = 0.0
            mean_heightmap_loss = 0.0
            
        self.storage.clear()

        return (mean_value_loss, mean_mlp_loss, mean_heightmap_loss, mean_surrogate_loss, mean_kl)
