import torch
import numpy as np
import os
import math

from isaacgym.torch_utils import *
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import (
    quat_apply_yaw,
    wrap_to_pi,
    torch_rand_sqrt_float,
)
from .pointfoot_flat_config import BipedCfgPF

class BipedPF(BaseTask):
    
    def __init__(
        self, cfg: BipedCfgPF, sim_params, physics_engine, sim_device, headless
    ):
        """Parses the provided config file,
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

        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.pi = torch.acos(torch.zeros(1, device=self.device)) * 2

        self.group_idx = torch.arange(0, self.cfg.env.num_envs)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        
        # Feedforward control 가중치 초기화
        self.current_feedback_gain = self.cfg.control.feedback_gain
        self.current_feedforward_gain = self.cfg.control.feedforward_gain
        # Feedforward one-shot trigger buffers (per env, per foot)
        self.ff_active = torch.zeros(
            (self.cfg.env.num_envs, len(self.feet_indices)), dtype=torch.bool, device=self.device
        )
        self.ff_phase = torch.zeros(
            (self.cfg.env.num_envs, len(self.feet_indices)), dtype=torch.float, device=self.device
        )
        self.ff_scale = torch.zeros(
            (self.cfg.env.num_envs, len(self.feet_indices)), dtype=torch.float, device=self.device
        )
        
        
        self.init_done = True

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)

        Returns:
            obs (torch.Tensor): Tensor of shape (num_envs, num_observations_per_env)
            rewards (torch.Tensor): Tensor of shape (num_envs)
            dones (torch.Tensor): Tensor of shape (num_envs)
        """
        self._action_clip(actions)
        # step physics and render each frame
        self.render()
        self.pre_physics_step()
        for _ in range(self.cfg.control.decimation):
            self.action_fifo = torch.cat(
                (self.actions.unsqueeze(1), self.action_fifo[:, :-1, :]), dim=1
            )
            self.envs_steps_buf += 1
            self.torques = self._compute_torques(
                self.action_fifo[torch.arange(self.num_envs), self.action_delay_idx, :]
            ).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            if self.cfg.domain_rand.push_robots:
                self._push_robots()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.compute_dof_vel()
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        return (
            self.obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            self.obs_history,
            self.commands[:, :3] * self.commands_scale,
            self.critic_obs_buf # make sure critic_obs update in every for loop
        )

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = (
            self.command_ranges["lin_vel_x"][env_ids, 1]
            - self.command_ranges["lin_vel_x"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "lin_vel_x"
        ][
            env_ids, 0
        ]
        self.commands[env_ids, 1] = (
            self.command_ranges["lin_vel_y"][env_ids, 1]
            - self.command_ranges["lin_vel_y"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "lin_vel_y"
        ][
            env_ids, 0
        ]
        self.commands[env_ids, 2] = (
            self.command_ranges["ang_vel_yaw"][env_ids, 1]
            - self.command_ranges["ang_vel_yaw"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "ang_vel_yaw"
        ][
            env_ids, 0
        ]
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        # set small commands to zero
        # self.commands[env_ids, :2] *= (
        #     torch.norm(self.commands[env_ids, :2], dim=1) > self.cfg.commands.min_norm
        # ).unsqueeze(1)
        zero_command_idx = (
            (
                torch_rand_float(0, 1, (len(env_ids), 1), device=self.device)
                > self.cfg.commands.zero_command_prob
            )
            .squeeze(1)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self.commands[zero_command_idx, :3] = 0
        if self.cfg.commands.heading_command:
            forward = quat_apply(
                self.base_quat[zero_command_idx], self.forward_vec[zero_command_idx]
            )
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[zero_command_idx, 3] = heading

    def _compute_torques(self, actions):
        """Compute torques from actions with Feedforward control.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # Feedforward + Policy control combination: a = k_fb * a_policy + k_ff * a_feedforward
        
        # 1. 정책 액션 (a_policy)
        a_policy = actions * self.cfg.control.action_scale
        
        # 2. Feedforward 액션 (a_feedforward) - 접촉력 기반
        #    관절 단위 추가항으로 사용 (정책은 항상 full-scale 유지)
        ff_result = self._compute_feedforward_actions()
        if isinstance(ff_result, tuple):
            a_feedforward, _ff_env_mask = ff_result
        else:
            a_feedforward = ff_result
            _ff_env_mask = None
        
        # 3. 가중치 업데이트 (학습 진행에 따라)
        self._update_control_weights()
        
        # 4. 최종 액션: 정책은 항상 full-scale, FF는 트리거된 관절에만 가산
        k_fb = self.current_feedback_gain  # 유지(보통 1.0)
        k_ff = self.current_feedforward_gain

        a_desired = a_policy.clone() * k_fb
        joint_mask = (a_feedforward.abs() > 0)
        if joint_mask.any():
            a_desired[joint_mask] = a_desired[joint_mask] + k_ff * a_feedforward[joint_mask]
        
        # 5. PD 컨트롤러로 토크 계산
        if getattr(self.cfg.control, "debug_feedforward", False):
            with torch.no_grad():
                jm = (a_feedforward.abs() > 0)
                trigger_ratio = jm.float().mean()
                ff_mag = a_feedforward.abs().mean()
                kff = self.current_feedforward_gain
                # 콘솔 디버그
                print(
                    f"FF dbg | trig={trigger_ratio.item():.4f} mag={ff_mag.item():.4f} kff={kff:.2f}"
                )
                # Wandb/Tensorboard 로깅 (가능할 때)
                if hasattr(self, "writer") and self.writer is not None:
                    step_count = int(torch.mean(self.envs_steps_buf.float()).item())
                    self.writer.add_scalar("FF/trigger_ratio", trigger_ratio.item(), step_count)
                    self.writer.add_scalar("FF/magnitude", ff_mag.item(), step_count)
                    self.writer.add_scalar("FF/kff", float(kff), step_count)
        return self._pd_controller(a_desired)

    def _compute_feedforward_actions(self):
        """발이 z축 방향으로 위로 이동 중일 때만 Feedforward 액션 계산

        Returns:
            Tuple[Tensor, Tensor]: (a_feedforward, ff_env_mask)
                - a_feedforward: (num_envs, num_actions) feedforward 액션
                - ff_env_mask: (num_envs,) 어떤 환경에서 feedforward가 활성화되었는지
        """
        a_feedforward = torch.zeros_like(self.actions)
        
        if not self.cfg.control.enable_feedforward_control:
            return a_feedforward
        # 환경 활성 마스크 (한 발이라도 조건 충족 시 True)
        ff_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 발의 z축 속도 (위로 올라가는 것이 양수)
        foot_z_velocities = self.foot_velocities[:, :, 2]  # [num_envs, num_feet]
        
        # 발이 위로 이동 중인지 확인 (z축 속도 > 임계치)
        upward_movement_threshold = self.cfg.control.upward_movement_threshold  # 설정에서 가져옴
        upward_moving = foot_z_velocities > upward_movement_threshold
        
        # 접촉력이 높은 발들 감지 (터크에 걸린 경우)
        foot_xy_forces = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=-1)
        high_xy_forces = foot_xy_forces > self.cfg.control.obstacle_force_threshold
        
        for i, foot_idx in enumerate(self.feet_indices):
            # 위로 이동 중이면서 접촉력이 높은 발에만 트리거
            trigger = (upward_moving[:, i] & high_xy_forces[:, i])
            # 새로 트리거된 경우만 활성화(지속 중 중복 트리거 방지)
            newly_triggered = trigger & ~self.ff_active[:, i]
            if newly_triggered.any():
                # 초기화
                self.ff_active[newly_triggered, i] = True
                self.ff_phase[newly_triggered, i] = 0.0
                # 접촉력에 비례한 스케일 (0~2 클램프)
                self.ff_scale[newly_triggered, i] = torch.clamp(
                    (foot_xy_forces[newly_triggered, i] / self.cfg.control.obstacle_force_threshold), 0.0, 2.0
                )

            # 활성화된 환경들에 대해 위상 증가
            if self.ff_active[:, i].any():
                dt = self.sim_params.dt
                T = self.cfg.control.feedforward_period
                # 위상 업데이트
                self.ff_phase[self.ff_active[:, i], i] += dt
                # half-cosine 프로파일: 0→T 동안 0.5*(1-cos(pi*t/T))
                phase = torch.clamp(self.ff_phase[:, i] / T, 0.0, 1.0)
                profile = 0.5 * (1 - torch.cos(torch.pi * phase))

                # 해당 발의 관절들에 Feedforward 액션 적용
                hip_joint = f"hip_{'L' if i == 0 else 'R'}_Joint"
                knee_joint = f"knee_{'L' if i == 0 else 'R'}_Joint"

                mag = profile * self.ff_scale[:, i] * self.cfg.control.feedforward_amplitude

                active_mask = self.ff_active[:, i]
                a_feedforward[active_mask, self.dof_names.index(hip_joint)] += mag[active_mask]
                a_feedforward[active_mask, self.dof_names.index(knee_joint)] += mag[active_mask] * 2.0
                ff_env_mask |= active_mask

                # 프로파일 완료 시 비활성화
                finished = phase >= 1.0
                if finished.any():
                    self.ff_active[finished, i] = False
                    self.ff_phase[finished, i] = 0.0
                    self.ff_scale[finished, i] = 0.0
        
        return a_feedforward, ff_env_mask

    def _update_control_weights(self):
        """초기 Feedforward 100% 적용, 점차 정책으로 전환. k_fb는 1.0 고정"""
        if not self.cfg.control.enable_feedforward_control:
            return
        
        avg_steps = torch.mean(self.envs_steps_buf.float())
        start_steps = self.cfg.control.feedforward_start_steps  # 0
        fade_steps = self.cfg.control.feedforward_fade_steps    # 8000
        
        if avg_steps < fade_steps:
            # 0-8000 스텝: Feedforward 점진적 감소, Feedback은 1.0 고정
            progress = avg_steps / fade_steps
            self.current_feedback_gain = 1.0
            self.current_feedforward_gain = 1.0 - progress  # 1.0 → 0.0
        else:
            # 8000 스텝 이후: 완전히 정책 기반
            self.current_feedback_gain = 1.0      # 1.0 (정책만)
            self.current_feedforward_gain = 0.0   # 0.0 (Feedforward 없음)

    def _pd_controller(self, a_desired):
        """PD 컨트롤러로 토크 계산"""
        control_type = self.cfg.control.control_type
        
        if control_type == "P":
            torques = (
                self.p_gains * (a_desired + self.default_dof_pos - self.dof_pos)
                - self.d_gains * self.dof_vel
            )
        elif control_type == "V":
            torques = (
                self.p_gains * (a_desired - self.dof_vel)
                - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
            )
        elif control_type == "T":
            torques = a_desired
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        return torch.clip(
            torques * self.torques_scale, -self.torque_limits, self.torque_limits
        )

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = (
            noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        )
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:12] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        noise_vec[12:18] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        noise_vec[18:] = 0.0  # previous actions
        return noise_vec
    
    def reset_idx(self, env_ids):
        """Reset some environments.
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
        if self.cfg.commands.curriculum:
            time_out_env_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
            self.update_command_curriculum(time_out_env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self._resample_gaits(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_base_position[env_ids] = self.base_position[env_ids]
        self.last_foot_positions[env_ids] = self.foot_positions[env_ids]
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.envs_steps_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history[env_ids] = 0
        obs_buf, _ = self.compute_group_observations()
        self.obs_history[env_ids] = obs_buf[env_ids].repeat(1, self.obs_history_length)
        self.gait_indices[env_ids] = 0
        self.fail_buf[env_ids] = 0
        self.action_fifo[env_ids] = 0
        self.dof_pos_int[env_ids] = 0
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["group_terrain_level"] = torch.mean(
                self.terrain_levels[self.group_idx].float()
            )
            self.extras["episode"]["group_terrain_level_stair_up"] = torch.mean(
                self.terrain_levels[self.stair_up_idx].float()
            )
        if self.cfg.terrain.curriculum and self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = torch.mean(
                self.command_ranges["lin_vel_x"][self.smooth_slope_idx, 1].float()
            )
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf | self.edge_reset_buf

    def compute_group_observations(self):
        # note that observation noise need to modified accordingly !!!
        obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                self.clock_inputs_sin.view(self.num_envs, 1),
                self.clock_inputs_cos.view(self.num_envs, 1),
                self.gaits,
            ),
            dim=-1,
        )
        critic_obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel, self.obs_buf), dim=-1)
        return obs_buf, critic_obs_buf
    
    # --------------------------- reward functions---------------------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        reward = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return reward

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square(self.dof_acc), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.actions - self.last_actions[:, :, 0]), dim=1)

    def _reward_action_smooth(self):
        # Penalize changes in actions
        return torch.sum(
            torch.square(
                self.actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1)

    def _reward_keep_balance(self):
        return torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.ang_tracking_sigma)

    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states

        reward = 0
        if self.reward_scales["tracking_contacts_shaped_force"] > 0:
            for i in range(len(self.feet_indices)):
                reward += (1 - desired_contact[:, i]) * torch.exp(
                    -foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma)
        else:
            for i in range(len(self.feet_indices)):
                reward += (1 - desired_contact[:, i]) * (
                    1 - torch.exp(-foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma))

        return reward / len(self.feet_indices)

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.foot_velocities, dim=-1)
        desired_contact = self.desired_contact_states
        reward = 0
        if self.reward_scales["tracking_contacts_shaped_vel"] > 0:
            for i in range(len(self.feet_indices)):
                reward += desired_contact[:, i] * torch.exp(
                    -foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma
                )
        else:
            for i in range(len(self.feet_indices)):
                reward += desired_contact[:, i] * (
                    1 - torch.exp(-foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma))
        return reward / len(self.feet_indices)

    def _reward_feet_distance(self):
        # Penalize base height away from target
        feet_distance = torch.norm(self.foot_positions[:, 0, :2] - self.foot_positions[:, 1, :2], dim=-1)
        reward = torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1)
        return reward

    def _reward_feet_regulation(self):
        feet_height = self.cfg.rewards.base_height_target * 0.001
        reward = torch.sum(
            torch.exp(-self.foot_heights / feet_height)
            * torch.square(torch.norm(self.foot_velocities[:, :, :2], dim=-1)), dim=1)
        return reward

    def _reward_collision(self):
        return torch.sum(
            torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 1.0, dim=1)

    def _reward_foot_landing_vel(self):
        z_vels = self.foot_velocities[:, :, 2]
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        about_to_land = (self.foot_heights < self.cfg.rewards.about_landing_threshold) & (~contacts) & (z_vels < 0.0)
        landing_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
        reward = torch.sum(torch.square(landing_z_vels), dim=1)
        return reward

    def _reward_foot_clearance(self):
        """
        Feet clearance = Σ_i 1_{swing,i} ⋅ 1_{h_min < h_i < h_max}
        목표값: 2.0 (양발 모두 조건 만족)
        """
        foot_heights = self.foot_heights  # [num_envs, num_feet]
        
        # 높이 범위 설정 (10-20cm)
        h_min = self.cfg.rewards.foot_clearance_min_height  # 0.10m
        h_max = self.cfg.rewards.foot_clearance_max_height  # 0.20m
        
        # 스윙 상태 확인 (접촉 안함)
        foot_contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        swing_state = ~foot_contacts  # 접촉 안함 = 스윙 상태
        
        # 높이 범위 확인
        height_in_range = (foot_heights > h_min) & (foot_heights < h_max)
        
        # 수식 구현: 1_{swing,i} ⋅ 1_{h_min < h_i < h_max}
        clearance_condition = swing_state & height_in_range
        
        # 스윙 중인 발의 개수
        swing_feet_count = torch.sum(swing_state.float(), dim=1)
        
        # 스윙 중인 발 중에서 적절한 높이에 있는 발의 개수
        clearance_feet_count = torch.sum(clearance_condition.float(), dim=1)
        
        # 스윙 중인 발이 있을 때만 보상 계산
        reward = torch.where(
            swing_feet_count > 0,
            clearance_feet_count / swing_feet_count,  # 스윙 발 중 적절한 높이 비율
            torch.zeros_like(swing_feet_count)  # 스윙 발이 없으면 보상 0
        )
        
        return reward

    def _reward_knee_ground(self):
        """Penalize knees getting too close to measured ground.
        - Use measured_heights (height map) as ground reference.
        - If any knee z is closer than knee_clearance_min to ground, penalize the deficit.
        """
        # Defensive: if rigid body names are not yet available, skip penalty this step
        if not hasattr(self, "rigid_body_names"):
            return torch.zeros(self.num_envs, device=self.device)
        # measured ground height under base footprint (same ref as base_height)
        ground_z = torch.mean(self.measured_heights, dim=1)  # [num_envs]

        # knee body indices (names may vary; match with startswith)
        def find_body(prefix):
            for idx, name in enumerate(self.rigid_body_names):
                if name.startswith(prefix):
                    return idx
            return None
        kL = find_body("knee_L")
        kR = find_body("knee_R")

        penalty = torch.zeros(self.num_envs, device=self.device)
        if kL is not None and kR is not None:
            knees_z = torch.stack([
                self.rigid_body_state[:, kL, 2],
                self.rigid_body_state[:, kR, 2],
            ], dim=1)  # [num_envs, 2]
            knee_clear_min = self.cfg.rewards.knee_clearance_min
            # clearance wrt ground
            clearance = knees_z - ground_z.unsqueeze(1)
            # deficit if below minimum clearance
            deficit = torch.clamp(knee_clear_min - clearance, min=0.0)
            # take worst knee per env
            penalty = deficit.max(dim=1).values
        return penalty

    def _post_physics_step_callback(self):
        """Post physics step callback"""
        # 부모 클래스의 callback 호출
        super()._post_physics_step_callback()