import argparse
import os
import random
import time
from collections import deque
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.distributions.normal import Normal

from exp_utils import add_common_args, setup_logging, finish_logging
from env_utils import (
    make_atari_env,
    make_minigrid_env,
    make_poc_env,
    make_classic_env,
    make_memory_gym_env,
    make_continuous_env,
)
from layers import layer_init


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)

    # SAC-specific arguments
    parser.add_argument("--hidden-dim", type=int, default=256, help="the hidden dimension of the model")
    parser.add_argument("--buffer-size", type=int, default=1_000_000, help="replay buffer size (in transitions)")
    parser.add_argument("--tau", type=float, default=0.005, help="soft update coefficient for target networks")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size for SAC updates")
    parser.add_argument("--policy-frequency", type=int, default=2, help="policy update frequency (in Q updates)")
    parser.add_argument("--target-network-frequency", type=int, default=1, help="target network update frequency (in Q updates)")
    parser.add_argument("--alpha", type=float, default=0.2, help="entropy coefficient if not autotuning")
    parser.add_argument(
        "--autotune",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="automatic entropy tuning",
    )
    parser.add_argument("--learning-starts", type=int, default=10_000, help="global step to start learning (in env transitions)")
    parser.add_argument("--warmup-steps", type=int, default=1_000, help="random action warmup steps (in env transitions)")
    parser.add_argument("--update-frequency", type=int, default=1, help="perform updates every N vector-env steps")
    parser.add_argument("--updates-per-step", type=int, default=1, help="gradient updates per update step trigger")

    # Match PPO env-arg surface where relevant
    parser.add_argument("--obs-stack", type=int, default=1, help="the number of observations to stack (for continuous/classic wrappers)")
    parser.add_argument("--masked-indices", type=str, default="", help="indices of classic-control observations to mask")
    parser.add_argument("--frame-stack", type=int, default=1, help="frame stack for image environments")

    args = parser.parse_args()
    args.masked_indices = [int(x) for x in args.masked_indices.split(",")]
    return args


def _build_encoder(obs_space: gym.Space, args) -> nn.Module:
    """
    Matches ppo.py encoder patterns:
    - MuJoCo env ids: Flatten -> Linear(?,64) Tanh -> Linear(64,hidden) Tanh
    - Else: CNN for image-like obs (3D/4D), or fallback MLP for vectors
    """
    mujoco_envs = ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"]
    if args.gym_id in mujoco_envs:
        input_dim = int(np.prod(obs_space.shape))
        return nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(input_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, args.hidden_dim)),
            nn.Tanh(),
        )

    obs_shape = obs_space.shape
    conv_input = False
    in_channels = None

    if isinstance(obs_space, gym.spaces.Box) and len(obs_shape) in [3, 4]:
        if len(obs_shape) == 3:
            # (C,H,W) or (H,W,C)
            if obs_shape[0] in [1, 3, 4]:
                in_channels = obs_shape[0]
            else:
                in_channels = obs_shape[2]
            conv_input = True
        elif len(obs_shape) == 4:
            # (frame_stack,H,W,C) for some env wrappers
            if obs_shape[-1] in [1, 3, 4]:
                in_channels = obs_shape[0] * obs_shape[-1]
                conv_input = True

    if conv_input:
        return nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, args.hidden_dim)),
            nn.ReLU(),
        )

    # Fallback vector encoder
    input_dim = int(np.prod(obs_shape))
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, args.hidden_dim),
        nn.ReLU(),
    )


def _preprocess_obs(x: torch.Tensor, gym_id: str) -> torch.Tensor:
    """
    Matches ppo.py get_states preprocessing:
    - Minigrid/Mortar: permute to CHW and scale to [0,1], supports optional frame stack
    - ALE: scale to [0,1]
    """
    gid = gym_id.lower()

    # Ensure batch dimension exists
    # (The calling code always uses batched tensors, but keep robust.)
    # We do not infer obs_space shape here, only normalize/permute for known env families.
    if "minigrid" in gid or "mortar" in gid:
        if x.ndim == 5:
            # (B,frame_stack,H,W,C) -> (B,frame_stack,C,H,W) -> (B,frame_stack*C,H,W)
            x = x.permute(0, 1, 4, 2, 3)
            b, fs, c, h, w = x.shape
            x = x.reshape(b, fs * c, h, w) / 255.0
        else:
            # (B,H,W,C) -> (B,C,H,W)
            x = x.permute(0, 3, 1, 2) / 255.0

    if "ale/" in gid:
        x = x / 255.0

    return x


class ReplayBuffer:
    """Vector-env-friendly replay buffer storing transitions on CPU (numpy), sampling to torch on device."""

    def __init__(self, obs_space: gym.Space, action_space: gym.Space, buffer_size: int, device: torch.device):
        self.buffer_size = int(buffer_size)
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs_shape = obs_space.shape
        self.action_shape = action_space.shape

        self.obs_dtype = np.uint8 if getattr(obs_space, "dtype", None) == np.uint8 else np.float32

        self.observations = np.zeros((self.buffer_size,) + self.obs_shape, dtype=self.obs_dtype)
        self.next_observations = np.zeros((self.buffer_size,) + self.obs_shape, dtype=self.obs_dtype)
        self.actions = np.zeros((self.buffer_size,) + self.action_shape, dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size,), dtype=np.float32)

    def __len__(self):
        return self.size

    def add(self, obs: np.ndarray, next_obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray):
        """
        obs, next_obs: (N, *obs_shape)
        actions: (N, *action_shape)
        rewards, dones: (N,)
        """
        n = obs.shape[0]

        # Write with wrap-aware slicing (avoid advanced indexing for speed)
        if self.ptr + n <= self.buffer_size:
            sl = slice(self.ptr, self.ptr + n)
            self.observations[sl] = obs.astype(self.obs_dtype, copy=False)
            self.next_observations[sl] = next_obs.astype(self.obs_dtype, copy=False)
            self.actions[sl] = actions.astype(np.float32, copy=False)
            self.rewards[sl] = rewards.astype(np.float32, copy=False)
            self.dones[sl] = dones.astype(np.float32, copy=False)
        else:
            first = self.buffer_size - self.ptr
            second = n - first

            self.observations[self.ptr :] = obs[:first].astype(self.obs_dtype, copy=False)
            self.next_observations[self.ptr :] = next_obs[:first].astype(self.obs_dtype, copy=False)
            self.actions[self.ptr :] = actions[:first].astype(np.float32, copy=False)
            self.rewards[self.ptr :] = rewards[:first].astype(np.float32, copy=False)
            self.dones[self.ptr :] = dones[:first].astype(np.float32, copy=False)

            self.observations[:second] = obs[first:].astype(self.obs_dtype, copy=False)
            self.next_observations[:second] = next_obs[first:].astype(self.obs_dtype, copy=False)
            self.actions[:second] = actions[first:].astype(np.float32, copy=False)
            self.rewards[:second] = rewards[first:].astype(np.float32, copy=False)
            self.dones[:second] = dones[first:].astype(np.float32, copy=False)

        self.ptr = (self.ptr + n) % self.buffer_size
        self.size = min(self.size + n, self.buffer_size)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)

        obs = torch.as_tensor(self.observations[idx], device=self.device)
        next_obs = torch.as_tensor(self.next_observations[idx], device=self.device)
        actions = torch.as_tensor(self.actions[idx], device=self.device)
        rewards = torch.as_tensor(self.rewards[idx], device=self.device)
        dones = torch.as_tensor(self.dones[idx], device=self.device)

        # Convert obs to float for networks (image obs stays 0..255 until preprocessing divides)
        obs = obs.float()
        next_obs = next_obs.float()
        actions = actions.float()
        rewards = rewards.float()
        dones = dones.float()

        return obs, actions, rewards, next_obs, dones


class QNetwork(nn.Module):
    """Single Q network (one head). Instantiate two for twin Q."""

    def __init__(self, envs: gym.vector.VectorEnv, args):
        super().__init__()
        self.args = args
        self.obs_space = envs.single_observation_space
        action_dim = int(np.prod(envs.single_action_space.shape))

        self.encoder = _build_encoder(self.obs_space, args)
        self.q = nn.Sequential(
            layer_init(nn.Linear(args.hidden_dim + action_dim, args.hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(args.hidden_dim, 1), std=1.0),
        )

    def get_states(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == len(self.obs_space.shape):
            obs = obs.unsqueeze(0)
        obs = _preprocess_obs(obs, self.args.gym_id)
        return self.encoder(obs)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.dim() == 1:
            action = action.unsqueeze(0)
        h = self.get_states(obs)
        x = torch.cat([h, action], dim=-1)
        return self.q(x)


class GaussianPolicy(nn.Module):
    """Gaussian policy with tanh squashing and action rescaling."""

    def __init__(self, envs: gym.vector.VectorEnv, args, log_std_min: float = -20, log_std_max: float = 2):
        super().__init__()
        self.args = args
        self.obs_space = envs.single_observation_space
        self.action_space = envs.single_action_space
        self.action_dim = int(np.prod(self.action_space.shape))

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.encoder = _build_encoder(self.obs_space, args)
        self.actor_body = nn.Sequential(
            layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)),
            nn.ReLU(),
        )
        self.mean_layer = layer_init(nn.Linear(args.hidden_dim, self.action_dim), std=0.01)
        self.log_std_layer = layer_init(nn.Linear(args.hidden_dim, self.action_dim), std=0.01)

        action_low = torch.as_tensor(self.action_space.low, dtype=torch.float32)
        action_high = torch.as_tensor(self.action_space.high, dtype=torch.float32)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def get_states(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == len(self.obs_space.shape):
            obs = obs.unsqueeze(0)
        obs = _preprocess_obs(obs, self.args.gym_id)
        h = self.encoder(obs)
        h = self.actor_body(h)
        return h

    def forward(self, obs: torch.Tensor):
        h = self.get_states(obs)
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs: torch.Tensor):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)

        action = y_t * self.action_scale + self.action_bias

        # log prob correction (includes action scaling Jacobian)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, log_std = self.forward(obs)
        if deterministic:
            y_t = torch.tanh(mean)
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
        return y_t * self.action_scale + self.action_bias


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target: nn.Module, source: nn.Module):
    target.load_state_dict(source.state_dict())


if __name__ == "__main__":
    args = parse_args()
    writer, run_name = setup_logging(args)

    # Seeding (match PPO structure)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cudnn.benchmark = False

    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this system.")
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    torch.set_default_device(device)

    # Environment setup (match PPO branching)
    if "ale" in args.gym_id.lower():
        envs_lst = [
            make_atari_env(args.gym_id, args.seed + i, i, args.capture_video, run_name, frame_stack=args.frame_stack)
            for i in range(args.num_envs)
        ]
    elif "minigrid" in args.gym_id.lower():
        envs_lst = [
            make_minigrid_env(
                args.gym_id,
                args.seed + i,
                i,
                args.capture_video,
                run_name,
                agent_view_size=3,
                tile_size=28,
                max_episode_steps=96,
                frame_stack=args.frame_stack,
            )
            for i in range(args.num_envs)
        ]
    elif "poc" in args.gym_id.lower():
        envs_lst = [
            make_poc_env(args.gym_id, args.seed + i, i, args.capture_video, run_name, step_size=0.02, glob=False, freeze=True, max_episode_steps=96)
            for i in range(args.num_envs)
        ]
    elif args.gym_id == "MortarMayhem-Grid-v0":
        envs_lst = [make_memory_gym_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    elif args.gym_id in ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"]:
        envs_lst = [
            make_continuous_env(args.gym_id, args.seed + i, i, args.capture_video, run_name, obs_stack=args.obs_stack)
            for i in range(args.num_envs)
        ]
    else:
        envs_lst = [
            make_classic_env(args.gym_id, args.seed + i, i, args.capture_video, run_name, masked_indices=args.masked_indices, obs_stack=args.obs_stack)
            for i in range(args.num_envs)
        ]
    envs = gym.vector.SyncVectorEnv(envs_lst)

    # SAC supports continuous actions only
    if not isinstance(envs.single_action_space, gym.spaces.Box):
        raise RuntimeError(f"SAC only supports continuous (Box) action spaces. Got: {envs.single_action_space}")

    action_dim = int(np.prod(envs.single_action_space.shape))
    action_low = envs.single_action_space.low.astype(np.float32)
    action_high = envs.single_action_space.high.astype(np.float32)

    # Networks
    actor = GaussianPolicy(envs, args).to(device)
    qf1 = QNetwork(envs, args).to(device)
    qf2 = QNetwork(envs, args).to(device)
    qf1_target = QNetwork(envs, args).to(device)
    qf2_target = QNetwork(envs, args).to(device)
    hard_update(qf1_target, qf1)
    hard_update(qf2_target, qf2)

    # Optimizers
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate, eps=1e-5)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate, eps=1e-5)

    # Entropy tuning
    if args.autotune:
        target_entropy = -float(action_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optimizer = optim.Adam([log_alpha], lr=args.learning_rate, eps=1e-5)
        alpha = log_alpha.exp().item()
    else:
        target_entropy = None
        log_alpha = None
        alpha_optimizer = None
        alpha = float(args.alpha)

    # Replay buffer
    replay_buffer = ReplayBuffer(envs.single_observation_space, envs.single_action_space, args.buffer_size, device=device)

    # Param count logging (match PPO pattern)
    total_params = sum(p.numel() for p in actor.parameters()) + sum(p.numel() for p in qf1.parameters()) + sum(p.numel() for p in qf2.parameters())
    trainable_params = (
        sum(p.numel() for p in actor.parameters() if p.requires_grad)
        + sum(p.numel() for p in qf1.parameters() if p.requires_grad)
        + sum(p.numel() for p in qf2.parameters() if p.requires_grad)
    )
    if args.track:
        wandb.config.update({"total_parameters": total_params, "trainable_parameters": trainable_params}, allow_val_change=True)
    print(f"Total parameters: {total_params / 10e6:.4f}M, trainable parameters: {trainable_params / 10e6:.4f}M")

    # Start
    global_step = 0
    env_step = 0
    update_step = 0
    start_time = time.time()
    episode_infos = deque(maxlen=100)

    next_obs_np, _ = envs.reset(seed=[args.seed + i for i in range(args.num_envs)])
    next_obs = torch.Tensor(next_obs_np).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    steps_per_update = args.num_envs * args.num_steps
    num_updates = args.total_timesteps // steps_per_update

    for update in range(1, num_updates + 1):
        update_start_time = time.time()

        # Anneal LR (match PPO style)
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / max(1, num_updates)
            lrnow = frac * args.learning_rate
            q_optimizer.param_groups[0]["lr"] = lrnow
            actor_optimizer.param_groups[0]["lr"] = lrnow
            if args.autotune:
                alpha_optimizer.param_groups[0]["lr"] = lrnow

        qf1_loss_list, qf2_loss_list, qf_loss_list = [], [], []
        actor_loss_list, alpha_loss_list = [], []
        inference_time_total = 0.0

        for step in range(args.num_steps):
            env_step += 1
            global_step += args.num_envs

            # Action logic (measure inference latency similarly to PPO)
            inf_start = time.time()
            with torch.no_grad():
                policy_action = actor.get_action(next_obs, deterministic=False)
            inference_time_total += (time.time() - inf_start)

            action = policy_action.cpu().numpy()

            # Random warmup (still computes policy forward for consistent latency logging)
            if global_step < args.warmup_steps:
                action = np.random.uniform(low=action_low, high=action_high, size=(args.num_envs, action_dim)).astype(np.float32)

            # Step env
            obs_np = next_obs_np
            next_obs_np, reward, terminated, truncated, info = envs.step(action)
            done = np.logical_or(terminated, truncated)

            # Store transition
            replay_buffer.add(
                obs=obs_np,
                next_obs=next_obs_np,
                actions=action,
                rewards=reward,
                dones=done.astype(np.float32),
            )

            # Update obs/done
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(done).to(device)

            # Episode logging (match PPO block)
            final_info = info.get("final_info")
            if final_info is not None and len(final_info) > 0:
                valid_entries = [entry for entry in final_info if entry is not None and "episode" in entry]
                if valid_entries:
                    episodic_returns = [entry["episode"]["r"] for entry in valid_entries]
                    episodic_lengths = [entry["episode"]["l"] for entry in valid_entries]
                    avg_return = float(f"{np.mean(episodic_returns):.3f}")
                    avg_length = float(f"{np.mean(episodic_lengths):.3f}")
                    episode_infos.append({"r": avg_return, "l": avg_length})
                    writer.add_scalar("charts/episode_return", avg_return, global_step)
                    writer.add_scalar("charts/episode_length", avg_length, global_step)

            # Training updates
            if (
                global_step >= args.learning_starts
                and len(replay_buffer) >= args.batch_size
                and (env_step % args.update_frequency == 0)
            ):
                for _ in range(args.updates_per_step):
                    update_step += 1

                    obs_b, act_b, rew_b, next_obs_b, done_b = replay_buffer.sample(args.batch_size)

                    # Compute alpha tensor
                    if args.autotune:
                        alpha_tensor = log_alpha.exp().detach()
                    else:
                        alpha_tensor = torch.tensor(alpha, device=device)

                    with torch.no_grad():
                        next_action_b, next_logp_b, _ = actor.sample(next_obs_b)
                        q1_next = qf1_target(next_obs_b, next_action_b)
                        q2_next = qf2_target(next_obs_b, next_action_b)
                        min_q_next = torch.min(q1_next, q2_next) - alpha_tensor * next_logp_b
                        target_q = rew_b.view(-1, 1) + (1.0 - done_b.view(-1, 1)) * args.gamma * min_q_next

                    q1 = qf1(obs_b, act_b)
                    q2 = qf2(obs_b, act_b)
                    qf1_loss = F.mse_loss(q1, target_q)
                    qf2_loss = F.mse_loss(q2, target_q)
                    qf_loss = qf1_loss + qf2_loss

                    q_optimizer.zero_grad(set_to_none=True)
                    qf_loss.backward()
                    nn.utils.clip_grad_norm_(list(qf1.parameters()) + list(qf2.parameters()), args.max_grad_norm)
                    q_optimizer.step()

                    qf1_loss_list.append(qf1_loss.item())
                    qf2_loss_list.append(qf2_loss.item())
                    qf_loss_list.append(qf_loss.item())

                    # Policy update (delayed)
                    if update_step % args.policy_frequency == 0:
                        # Freeze Q params to avoid unnecessary grads on Q during actor update
                        for p in qf1.parameters():
                            p.requires_grad = False
                        for p in qf2.parameters():
                            p.requires_grad = False

                        new_action, logp, _ = actor.sample(obs_b)
                        q1_pi = qf1(obs_b, new_action)
                        q2_pi = qf2(obs_b, new_action)
                        min_q_pi = torch.min(q1_pi, q2_pi)
                        actor_loss = (alpha_tensor * logp - min_q_pi).mean()

                        actor_optimizer.zero_grad(set_to_none=True)
                        actor_loss.backward()
                        nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                        actor_optimizer.step()

                        actor_loss_list.append(actor_loss.item())

                        # Unfreeze Q params
                        for p in qf1.parameters():
                            p.requires_grad = True
                        for p in qf2.parameters():
                            p.requires_grad = True

                        # Alpha update
                        if args.autotune:
                            alpha_loss = -(log_alpha * (logp + target_entropy).detach()).mean()
                            alpha_optimizer.zero_grad(set_to_none=True)
                            alpha_loss.backward()
                            alpha_optimizer.step()
                            alpha = log_alpha.exp().item()
                            alpha_loss_list.append(alpha_loss.item())

                    # Target updates
                    if update_step % args.target_network_frequency == 0:
                        soft_update(qf1_target, qf1, args.tau)
                        soft_update(qf2_target, qf2, args.tau)

        # End update logging
        avg_inference_latency = inference_time_total / args.num_steps
        writer.add_scalar("metrics/inference_latency", avg_inference_latency, global_step)

        avg_qf1_loss = float(np.mean(qf1_loss_list)) if qf1_loss_list else 0.0
        avg_qf2_loss = float(np.mean(qf2_loss_list)) if qf2_loss_list else 0.0
        avg_qf_loss = float(np.mean(qf_loss_list)) if qf_loss_list else 0.0
        avg_actor_loss = float(np.mean(actor_loss_list)) if actor_loss_list else 0.0
        avg_alpha_loss = float(np.mean(alpha_loss_list)) if alpha_loss_list else 0.0

        sps = int(global_step / (time.time() - start_time))
        current_return = np.mean([ep["r"] for ep in episode_infos]) if episode_infos else 0.0

        print(
            f"Update {update}: SPS={sps}, Return={current_return:.2f}, "
            f"qf_loss={avg_qf_loss:.6f}, actor_loss={avg_actor_loss:.6f}, alpha={alpha:.6f}"
        )

        writer.add_scalar("charts/learning_rate", actor_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/qf1_loss", avg_qf1_loss, global_step)
        writer.add_scalar("losses/qf2_loss", avg_qf2_loss, global_step)
        writer.add_scalar("losses/qf_loss", avg_qf_loss, global_step)
        writer.add_scalar("losses/actor_loss", avg_actor_loss, global_step)
        writer.add_scalar("losses/alpha", alpha, global_step)
        if args.autotune:
            writer.add_scalar("losses/alpha_loss", avg_alpha_loss, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)

        if episode_infos:
            avg_episode_return = float(np.mean([ep["r"] for ep in episode_infos]))
            writer.add_scalar("charts/avg_episode_return", avg_episode_return, global_step)

        update_time = time.time() - update_start_time
        writer.add_scalar("metrics/training_time_per_update", update_time, global_step)

        # GPU memory usage (match PPO metric names)
        if device.type == "cuda":
            gpu_memory_allocated = torch.cuda.memory_allocated(device)
            gpu_memory_reserved = torch.cuda.memory_reserved(device)
            total_gpu_memory = torch.cuda.get_device_properties(device).total_memory
            gpu_memory_allocated_gb = gpu_memory_allocated / (1024**3)
            gpu_memory_reserved_gb = gpu_memory_reserved / (1024**3)
            gpu_memory_allocated_percent = (gpu_memory_allocated / total_gpu_memory) * 100
            gpu_memory_reserved_percent = (gpu_memory_reserved / total_gpu_memory) * 100
        else:
            gpu_memory_allocated_gb = 0.0
            gpu_memory_reserved_gb = 0.0
            gpu_memory_allocated_percent = 0.0
            gpu_memory_reserved_percent = 0.0

        writer.add_scalar("metrics/GPU_memory_allocated_GB", gpu_memory_allocated_gb, global_step)
        writer.add_scalar("metrics/GPU_memory_reserved_GB", gpu_memory_reserved_gb, global_step)
        writer.add_scalar("metrics/GPU_memory_allocated_percent", gpu_memory_allocated_percent, global_step)
        writer.add_scalar("metrics/GPU_memory_reserved_percent", gpu_memory_reserved_percent, global_step)

        # Save model checkpoint every save_interval updates (match PPO style/path)
        if args.save_model and update % args.save_interval == 0:
            model_path = f"runs/{run_name}/{args.exp_name}_update_{update}.cleanrl_model"
            model_data = {
                "model_weights": {
                    "actor": actor.state_dict(),
                    "qf1": qf1.state_dict(),
                    "qf2": qf2.state_dict(),
                    "qf1_target": qf1_target.state_dict(),
                    "qf2_target": qf2_target.state_dict(),
                    "log_alpha": log_alpha.detach().cpu() if args.autotune else None,
                },
                "args": vars(args),
            }
            torch.save(model_data, model_path)
            print(f"Model saved to {model_path}")

    finish_logging(args, writer, run_name, envs)