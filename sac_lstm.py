import argparse
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

    # Model args (match PPO-LSTM surface)
    parser.add_argument("--hidden-dim", type=int, default=512, help="encoder hidden dim (pre-RNN)")
    parser.add_argument("--rnn-type", type=str, default="lstm", choices=["lstm", "gru"], help="recurrent cell type")
    parser.add_argument("--rnn-hidden-dim", type=int, default=512, help="RNN hidden dimension")

    # SAC args
    parser.add_argument("--buffer-size", type=int, default=1_000_000, help="replay buffer size (in transitions)")
    parser.add_argument("--tau", type=float, default=0.005, help="soft update coefficient for target networks")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size (in transitions); must be divisible by num_steps for SAC-LSTM")
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
    parser.add_argument("--learning-starts", type=int, default=10_000, help="global step to start learning (in transitions)")
    parser.add_argument("--warmup-steps", type=int, default=1_000, help="random action warmup steps (in transitions)")
    parser.add_argument("--update-frequency", type=int, default=1, help="perform updates every N vector-env steps")
    parser.add_argument("--updates-per-step", type=int, default=1, help="gradient updates per update step trigger")

    # Env args
    parser.add_argument("--obs-stack", type=int, default=1, help="obs stack for continuous/classic wrappers")
    parser.add_argument("--masked-indices", type=str, default="", help="indices of classic-control observations to mask")
    parser.add_argument("--frame-stack", type=int, default=1, help="frame stack for image environments")

    args = parser.parse_args()
    args.masked_indices = [int(x) for x in args.masked_indices.split(",")]

    # Adjust batch_size to be divisible by num_steps for proper sequence sampling
    if args.batch_size % args.num_steps != 0:
        old_batch = args.batch_size
        args.batch_size = (args.batch_size // args.num_steps) * args.num_steps
        if args.batch_size == 0:
            args.batch_size = args.num_steps
        print(f"Adjusted batch_size from {old_batch} to {args.batch_size} (must be divisible by num_steps={args.num_steps})")
    return args


def _preprocess_obs(x: torch.Tensor, gym_id: str) -> torch.Tensor:
    gid = gym_id.lower()
    if "minigrid" in gid or "mortar" in gid:
        if x.ndim == 5:
            x = x.permute(0, 1, 4, 2, 3)
            b, fs, c, h, w = x.shape
            x = x.reshape(b, fs * c, h, w) / 255.0
        else:
            x = x.permute(0, 3, 1, 2) / 255.0
    if "ale/" in gid:
        x = x / 255.0
    return x


def _build_encoder(obs_space: gym.Space, args) -> nn.Module:
    """
    Mirrors PPO-LSTM encoder patterns with added robustness for 4D stacked image obs.
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
            if obs_shape[0] in [1, 3, 4]:
                in_channels = obs_shape[0]
            else:
                in_channels = obs_shape[2]
            conv_input = True
        else:
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

    # Vector fallback encoder (matches PPO-LSTM style)
    input_dim = int(np.prod(obs_shape))
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, args.hidden_dim),
        nn.ReLU(),
        nn.Linear(args.hidden_dim, args.hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(args.hidden_dim // 2, args.hidden_dim),
        nn.ReLU(),
    )


def _init_rnn_orthogonal(rnn: nn.Module):
    for name, param in rnn.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.orthogonal_(param, 1.0)


class RecurrentGaussianPolicy(nn.Module):
    def __init__(self, envs: gym.vector.VectorEnv, args, log_std_min: float = -20, log_std_max: float = 2):
        super().__init__()
        self.args = args
        self.obs_space = envs.single_observation_space
        self.action_space = envs.single_action_space
        self.action_dim = int(np.prod(self.action_space.shape))

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.encoder = _build_encoder(self.obs_space, args)

        if args.rnn_type == "lstm":
            self.rnn = nn.LSTM(args.hidden_dim, args.rnn_hidden_dim)
        else:
            self.rnn = nn.GRU(args.hidden_dim, args.rnn_hidden_dim)
        _init_rnn_orthogonal(self.rnn)

        self.ln = nn.LayerNorm(args.rnn_hidden_dim)

        self.actor_body = nn.Sequential(
            layer_init(nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)),
            nn.ReLU(),
        )
        self.mean_layer = layer_init(nn.Linear(args.rnn_hidden_dim, self.action_dim), std=0.01)
        self.log_std_layer = layer_init(nn.Linear(args.rnn_hidden_dim, self.action_dim), std=0.01)

        action_low = torch.as_tensor(self.action_space.low, dtype=torch.float32)
        action_high = torch.as_tensor(self.action_space.high, dtype=torch.float32)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def get_states(self, x: torch.Tensor, rnn_state, done: torch.Tensor):
        # Preprocess images like PPO-LSTM/PPO
        x = _preprocess_obs(x, self.args.gym_id)
        hidden = self.encoder(x)

        if self.args.rnn_type == "lstm":
            batch_size = rnn_state[0].shape[1]
        else:
            batch_size = rnn_state.shape[1]

        hidden = hidden.reshape((-1, batch_size, self.rnn.input_size))
        done = done.reshape((-1, batch_size))

        new_hidden = []
        if self.args.rnn_type == "lstm":
            for h, d in zip(hidden, done):
                h, rnn_state = self.rnn(
                    h.unsqueeze(0),
                    (
                        (1.0 - d).view(1, -1, 1) * rnn_state[0],
                        (1.0 - d).view(1, -1, 1) * rnn_state[1],
                    ),
                )
                new_hidden.append(h)
        else:
            for h, d in zip(hidden, done):
                h, rnn_state = self.rnn(
                    h.unsqueeze(0),
                    (1.0 - d).view(1, -1, 1) * rnn_state,
                )
                new_hidden.append(h)

        new_hidden = torch.cat(new_hidden, dim=0).reshape(-1, self.args.rnn_hidden_dim)
        new_hidden = self.ln(new_hidden)
        return new_hidden, rnn_state

    def forward(self, x: torch.Tensor, rnn_state, done: torch.Tensor):
        h, rnn_state = self.get_states(x, rnn_state, done)
        h = self.actor_body(h)
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std, rnn_state

    def sample(self, x: torch.Tensor, rnn_state, done: torch.Tensor, deterministic: bool = False):
        mean, log_std, rnn_state = self.forward(x, rnn_state, done)
        std = log_std.exp()
        normal = Normal(mean, std)

        if deterministic:
            x_t = mean
        else:
            x_t = normal.rsample()

        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action, rnn_state

    def get_action(self, x: torch.Tensor, rnn_state, done: torch.Tensor, deterministic: bool = False):
        action, _, _, rnn_state = self.sample(x, rnn_state, done, deterministic=deterministic)
        return action, rnn_state


class RecurrentQNetwork(nn.Module):
    """Twin Q heads, shared encoder+RNN (reduces state bookkeeping and matches 'twin Q' requirement)."""

    def __init__(self, envs: gym.vector.VectorEnv, args):
        super().__init__()
        self.args = args
        self.obs_space = envs.single_observation_space
        action_dim = int(np.prod(envs.single_action_space.shape))

        self.encoder = _build_encoder(self.obs_space, args)

        if args.rnn_type == "lstm":
            self.rnn = nn.LSTM(args.hidden_dim, args.rnn_hidden_dim)
        else:
            self.rnn = nn.GRU(args.hidden_dim, args.rnn_hidden_dim)
        _init_rnn_orthogonal(self.rnn)

        self.ln = nn.LayerNorm(args.rnn_hidden_dim)

        self.q1 = nn.Sequential(
            layer_init(nn.Linear(args.rnn_hidden_dim + action_dim, args.rnn_hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(args.rnn_hidden_dim, 1), std=1.0),
        )
        self.q2 = nn.Sequential(
            layer_init(nn.Linear(args.rnn_hidden_dim + action_dim, args.rnn_hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(args.rnn_hidden_dim, 1), std=1.0),
        )

    def get_states(self, x: torch.Tensor, rnn_state, done: torch.Tensor):
        x = _preprocess_obs(x, self.args.gym_id)
        hidden = self.encoder(x)

        if self.args.rnn_type == "lstm":
            batch_size = rnn_state[0].shape[1]
        else:
            batch_size = rnn_state.shape[1]

        hidden = hidden.reshape((-1, batch_size, self.rnn.input_size))
        done = done.reshape((-1, batch_size))

        new_hidden = []
        if self.args.rnn_type == "lstm":
            for h, d in zip(hidden, done):
                h, rnn_state = self.rnn(
                    h.unsqueeze(0),
                    (
                        (1.0 - d).view(1, -1, 1) * rnn_state[0],
                        (1.0 - d).view(1, -1, 1) * rnn_state[1],
                    ),
                )
                new_hidden.append(h)
        else:
            for h, d in zip(hidden, done):
                h, rnn_state = self.rnn(
                    h.unsqueeze(0),
                    (1.0 - d).view(1, -1, 1) * rnn_state,
                )
                new_hidden.append(h)

        new_hidden = torch.cat(new_hidden, dim=0).reshape(-1, self.args.rnn_hidden_dim)
        new_hidden = self.ln(new_hidden)
        return new_hidden, rnn_state

    def forward(self, x: torch.Tensor, action: torch.Tensor, rnn_state, done: torch.Tensor):
        h, rnn_state = self.get_states(x, rnn_state, done)
        xcat = torch.cat([h, action], dim=-1)
        q1 = self.q1(xcat)
        q2 = self.q2(xcat)
        return q1, q2, rnn_state


class RecurrentReplayBuffer:
    """
    Stores rollout chunks of length num_steps for all envs, plus initial RNN states.
    Sampling returns full sequences (num_steps) for randomly chosen env trajectories.
    """

    def __init__(self, obs_space: gym.Space, action_space: gym.Space, args, device: torch.device, actor: RecurrentGaussianPolicy, qf: RecurrentQNetwork):
        self.device = device
        self.num_envs = args.num_envs
        self.seq_len = args.num_steps

        max_chunks = args.buffer_size // (args.num_envs * args.num_steps)
        if max_chunks < 1:
            raise ValueError("buffer-size too small for the chosen num_envs*num_steps chunking in SAC-LSTM.")
        self.max_chunks = int(max_chunks)

        self.ptr = 0
        self.size = 0

        self.obs_shape = obs_space.shape
        self.action_shape = action_space.shape
        self.obs_dtype = np.uint8 if getattr(obs_space, "dtype", None) == np.uint8 else np.float32

        # Store state sequence of length (T+1) so next-state is always available
        self.obs = np.zeros((self.max_chunks, self.seq_len + 1, self.num_envs) + self.obs_shape, dtype=self.obs_dtype)
        self.actions = np.zeros((self.max_chunks, self.seq_len, self.num_envs) + self.action_shape, dtype=np.float32)
        self.rewards = np.zeros((self.max_chunks, self.seq_len, self.num_envs), dtype=np.float32)
        self.dones = np.zeros((self.max_chunks, self.seq_len, self.num_envs), dtype=np.float32)  # done-after

        self.init_dones = np.zeros((self.max_chunks, self.num_envs), dtype=np.float32)

        # Store initial RNN states at chunk start (actor and critic)
        self.rnn_type = args.rnn_type
        self.actor_layers = actor.rnn.num_layers
        self.q_layers = qf.rnn.num_layers
        self.actor_hdim = actor.rnn.hidden_size
        self.q_hdim = qf.rnn.hidden_size

        if self.rnn_type == "lstm":
            self.actor_init_h = np.zeros((self.max_chunks, self.actor_layers, self.num_envs, self.actor_hdim), dtype=np.float32)
            self.actor_init_c = np.zeros((self.max_chunks, self.actor_layers, self.num_envs, self.actor_hdim), dtype=np.float32)
            self.q_init_h = np.zeros((self.max_chunks, self.q_layers, self.num_envs, self.q_hdim), dtype=np.float32)
            self.q_init_c = np.zeros((self.max_chunks, self.q_layers, self.num_envs, self.q_hdim), dtype=np.float32)
        else:
            self.actor_init_h = np.zeros((self.max_chunks, self.actor_layers, self.num_envs, self.actor_hdim), dtype=np.float32)
            self.q_init_h = np.zeros((self.max_chunks, self.q_layers, self.num_envs, self.q_hdim), dtype=np.float32)

    def __len__(self):
        return self.size

    def add_chunk(self, obs_seq, actions_seq, rewards_seq, dones_seq, init_dones, actor_init_state, q_init_state):
        """
        obs_seq: (T+1, N, *obs_shape)
        actions_seq: (T, N, *action_shape)
        rewards_seq: (T, N)
        dones_seq: (T, N)
        init_dones: (N,)
        actor_init_state / q_init_state: torch tensors on device (captured at chunk start)
        """
        idx = self.ptr

        self.obs[idx] = obs_seq.astype(self.obs_dtype, copy=False)
        self.actions[idx] = actions_seq.astype(np.float32, copy=False)
        self.rewards[idx] = rewards_seq.astype(np.float32, copy=False)
        self.dones[idx] = dones_seq.astype(np.float32, copy=False)
        self.init_dones[idx] = init_dones.astype(np.float32, copy=False)

        if self.rnn_type == "lstm":
            ah, ac = actor_init_state
            qh, qc = q_init_state
            self.actor_init_h[idx] = ah.detach().cpu().numpy().astype(np.float32, copy=False)
            self.actor_init_c[idx] = ac.detach().cpu().numpy().astype(np.float32, copy=False)
            self.q_init_h[idx] = qh.detach().cpu().numpy().astype(np.float32, copy=False)
            self.q_init_c[idx] = qc.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            self.actor_init_h[idx] = actor_init_state.detach().cpu().numpy().astype(np.float32, copy=False)
            self.q_init_h[idx] = q_init_state.detach().cpu().numpy().astype(np.float32, copy=False)

        self.ptr = (self.ptr + 1) % self.max_chunks
        self.size = min(self.size + 1, self.max_chunks)

    def sample_sequences(self, batch_size: int):
        """
        Returns sequences for B = batch_size//T env-trajectories:
        obs: (T+1, B, *obs_shape)
        actions: (T, B, *action_shape)
        rewards: (T, B)
        dones: (T, B)  [done-after]
        init_dones: (B,)
        actor_init_state, q_init_state in torch shapes expected by get_states
        """
        T = self.seq_len
        B = batch_size // T
        if B < 1:
            raise ValueError("batch_size too small for sequence length.")

        chunk_idx = np.random.randint(0, self.size, size=B)
        env_idx = np.random.randint(0, self.num_envs, size=B)

        obs_b = np.zeros((T + 1, B) + self.obs_shape, dtype=self.obs_dtype)
        actions_b = np.zeros((T, B) + self.action_shape, dtype=np.float32)
        rewards_b = np.zeros((T, B), dtype=np.float32)
        dones_b = np.zeros((T, B), dtype=np.float32)
        init_dones_b = np.zeros((B,), dtype=np.float32)

        if self.rnn_type == "lstm":
            actor_h = np.zeros((self.actor_layers, B, self.actor_hdim), dtype=np.float32)
            actor_c = np.zeros((self.actor_layers, B, self.actor_hdim), dtype=np.float32)
            q_h = np.zeros((self.q_layers, B, self.q_hdim), dtype=np.float32)
            q_c = np.zeros((self.q_layers, B, self.q_hdim), dtype=np.float32)
        else:
            actor_h = np.zeros((self.actor_layers, B, self.actor_hdim), dtype=np.float32)
            q_h = np.zeros((self.q_layers, B, self.q_hdim), dtype=np.float32)

        for i in range(B):
            c = chunk_idx[i]
            e = env_idx[i]
            obs_b[:, i] = self.obs[c, :, e]
            actions_b[:, i] = self.actions[c, :, e]
            rewards_b[:, i] = self.rewards[c, :, e]
            dones_b[:, i] = self.dones[c, :, e]
            init_dones_b[i] = self.init_dones[c, e]
            if self.rnn_type == "lstm":
                actor_h[:, i] = self.actor_init_h[c, :, e]
                actor_c[:, i] = self.actor_init_c[c, :, e]
                q_h[:, i] = self.q_init_h[c, :, e]
                q_c[:, i] = self.q_init_c[c, :, e]
            else:
                actor_h[:, i] = self.actor_init_h[c, :, e]
                q_h[:, i] = self.q_init_h[c, :, e]

        # Move to torch
        obs_t = torch.as_tensor(obs_b, device=self.device).float()
        actions_t = torch.as_tensor(actions_b, device=self.device).float()
        rewards_t = torch.as_tensor(rewards_b, device=self.device).float()
        dones_t = torch.as_tensor(dones_b, device=self.device).float()
        init_dones_t = torch.as_tensor(init_dones_b, device=self.device).float()

        if self.rnn_type == "lstm":
            actor_state = (
                torch.as_tensor(actor_h, device=self.device).float(),
                torch.as_tensor(actor_c, device=self.device).float(),
            )
            q_state = (
                torch.as_tensor(q_h, device=self.device).float(),
                torch.as_tensor(q_c, device=self.device).float(),
            )
        else:
            actor_state = torch.as_tensor(actor_h, device=self.device).float()
            q_state = torch.as_tensor(q_h, device=self.device).float()

        return obs_t, actions_t, rewards_t, dones_t, init_dones_t, actor_state, q_state


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target: nn.Module, source: nn.Module):
    target.load_state_dict(source.state_dict())


if __name__ == "__main__":
    args = parse_args()
    writer, run_name = setup_logging(args)

    # Seeding (match PPO-LSTM structure)
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

    if not isinstance(envs.single_action_space, gym.spaces.Box):
        raise RuntimeError(f"SAC only supports continuous (Box) action spaces. Got: {envs.single_action_space}")

    action_dim = int(np.prod(envs.single_action_space.shape))
    action_low = envs.single_action_space.low.astype(np.float32)
    action_high = envs.single_action_space.high.astype(np.float32)

    # Networks
    actor = RecurrentGaussianPolicy(envs, args).to(device)
    qf = RecurrentQNetwork(envs, args).to(device)
    qf_target = RecurrentQNetwork(envs, args).to(device)
    hard_update(qf_target, qf)

    # Optimizers
    q_optimizer = optim.Adam(qf.parameters(), lr=args.learning_rate, eps=1e-5)
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

    # Replay buffer (chunked)
    replay_buffer = RecurrentReplayBuffer(envs.single_observation_space, envs.single_action_space, args, device, actor, qf)

    # Param counts (match PPO style)
    total_params = sum(p.numel() for p in actor.parameters()) + sum(p.numel() for p in qf.parameters())
    trainable_params = sum(p.numel() for p in actor.parameters() if p.requires_grad) + sum(p.numel() for p in qf.parameters() if p.requires_grad)
    if args.track:
        wandb.config.update({"total_parameters": total_params, "trainable_parameters": trainable_params}, allow_val_change=True)
    print(f"Total parameters: {total_params / 10e6:.4f}M, trainable parameters: {trainable_params / 10e6:.4f}M")

    # Rollout state
    global_step = 0
    env_step = 0
    update_step = 0
    start_time = time.time()
    episode_infos = deque(maxlen=100)

    next_obs_np, _ = envs.reset(seed=[args.seed + i for i in range(args.num_envs)])
    next_obs = torch.Tensor(next_obs_np).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # RNN states (actor and critic)
    if args.rnn_type == "lstm":
        next_actor_rnn_state = (
            torch.zeros(actor.rnn.num_layers, args.num_envs, actor.rnn.hidden_size).to(device),
            torch.zeros(actor.rnn.num_layers, args.num_envs, actor.rnn.hidden_size).to(device),
        )
        next_q_rnn_state = (
            torch.zeros(qf.rnn.num_layers, args.num_envs, qf.rnn.hidden_size).to(device),
            torch.zeros(qf.rnn.num_layers, args.num_envs, qf.rnn.hidden_size).to(device),
        )
    else:
        next_actor_rnn_state = torch.zeros(actor.rnn.num_layers, args.num_envs, actor.rnn.hidden_size).to(device)
        next_q_rnn_state = torch.zeros(qf.rnn.num_layers, args.num_envs, qf.rnn.hidden_size).to(device)

    steps_per_update = args.num_envs * args.num_steps
    num_updates = args.total_timesteps // steps_per_update

    for update in range(1, num_updates + 1):
        update_start_time = time.time()

        # Anneal LR (match PPO)
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / max(1, num_updates)
            lrnow = frac * args.learning_rate
            q_optimizer.param_groups[0]["lr"] = lrnow
            actor_optimizer.param_groups[0]["lr"] = lrnow
            if args.autotune:
                alpha_optimizer.param_groups[0]["lr"] = lrnow

        # Capture initial RNN states for this rollout chunk
        if args.rnn_type == "lstm":
            chunk_actor_init_state = (next_actor_rnn_state[0].clone(), next_actor_rnn_state[1].clone())
            chunk_q_init_state = (next_q_rnn_state[0].clone(), next_q_rnn_state[1].clone())
        else:
            chunk_actor_init_state = next_actor_rnn_state.clone()
            chunk_q_init_state = next_q_rnn_state.clone()
        chunk_init_dones = next_done.detach().cpu().numpy().astype(np.float32, copy=False)

        # Chunk storage (CPU numpy)
        obs_dtype = np.uint8 if getattr(envs.single_observation_space, "dtype", None) == np.uint8 else np.float32
        obs_shape = envs.single_observation_space.shape
        act_shape = envs.single_action_space.shape

        chunk_obs = np.zeros((args.num_steps + 1, args.num_envs) + obs_shape, dtype=obs_dtype)
        chunk_actions = np.zeros((args.num_steps, args.num_envs) + act_shape, dtype=np.float32)
        chunk_rewards = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
        chunk_dones = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)

        chunk_obs[0] = next_obs_np.astype(obs_dtype, copy=False)

        q_loss_list, q1_loss_list, q2_loss_list = [], [], []
        actor_loss_list, alpha_loss_list = [], []
        inference_time_total = 0.0

        for t in range(args.num_steps):
            env_step += 1
            global_step += args.num_envs

            # Update critic RNN state on current obs (no grad); aligns stored initial state -> sequence processing
            with torch.no_grad():
                _, next_q_rnn_state = qf.get_states(next_obs, next_q_rnn_state, next_done)

            # Actor action (measure inference latency like PPO style)
            inf_start = time.time()
            with torch.no_grad():
                policy_action, next_actor_rnn_state = actor.get_action(next_obs, next_actor_rnn_state, next_done, deterministic=False)
            inference_time_total += (time.time() - inf_start)

            action = policy_action.cpu().numpy()

            # Random warmup (but we still ran actor forward above to keep latency measurement consistent)
            if global_step < args.warmup_steps:
                action = np.random.uniform(low=action_low, high=action_high, size=(args.num_envs, action_dim)).astype(np.float32)

            # Step env
            next_obs_np, reward, terminated, truncated, info = envs.step(action)
            done_after = np.logical_or(terminated, truncated)

            # Store transition in chunk
            chunk_actions[t] = action
            chunk_rewards[t] = reward.astype(np.float32, copy=False)
            chunk_dones[t] = done_after.astype(np.float32, copy=False)
            chunk_obs[t + 1] = next_obs_np.astype(obs_dtype, copy=False)

            # Update obs/done tensors
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(done_after).to(device)

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

            # Training updates (use previously stored chunks)
            if (
                global_step >= args.learning_starts
                and len(replay_buffer) > 0
                and (env_step % args.update_frequency == 0)
            ):
                for _ in range(args.updates_per_step):
                    update_step += 1

                    obs_seq, act_seq, rew_seq, done_seq, init_done_seq, actor_init_state, q_init_state = replay_buffer.sample_sequences(args.batch_size)
                    # Shapes:
                    # obs_seq: (T+1, B, ...)
                    # act_seq: (T, B, ...)
                    T = args.num_steps
                    B = act_seq.shape[1]

                    # done-before for state sequence: done_before[0]=init_done, done_before[1:]=done_after
                    done_before = torch.zeros((T + 1, B), device=device)
                    done_before[0] = init_done_seq
                    done_before[1:] = done_seq

                    obs_plus_flat = obs_seq.reshape((-1,) + obs_seq.shape[2:])
                    done_plus_flat = done_before.reshape(-1)

                    # Alpha tensor
                    if args.autotune:
                        alpha_tensor = log_alpha.exp().detach()
                    else:
                        alpha_tensor = torch.tensor(alpha, device=device)

                    # Critic hidden states (current)
                    hidden_q_plus, _ = qf.get_states(obs_plus_flat, q_init_state, done_plus_flat)
                    hidden_q_plus = hidden_q_plus.view(T + 1, B, -1)

                    hidden_q = hidden_q_plus[:-1].reshape(-1, args.rnn_hidden_dim)  # states 0..T-1
                    act_env = act_seq.reshape(-1, action_dim)

                    q1 = qf.q1(torch.cat([hidden_q, act_env], dim=-1))
                    q2 = qf.q2(torch.cat([hidden_q, act_env], dim=-1))

                    with torch.no_grad():
                        # Next actions and log probs from current policy at next states (1..T)
                        next_actions_plus, next_logp_plus, _, _ = actor.sample(obs_plus_flat, actor_init_state, done_plus_flat, deterministic=False)
                        next_actions_plus = next_actions_plus.view(T + 1, B, action_dim)
                        next_logp_plus = next_logp_plus.view(T + 1, B, 1)

                        next_actions = next_actions_plus[1:].reshape(-1, action_dim)
                        next_logp = next_logp_plus[1:].reshape(-1, 1)

                        # Target critic hidden (target network)
                        hidden_qt_plus, _ = qf_target.get_states(obs_plus_flat, q_init_state, done_plus_flat)
                        hidden_qt_plus = hidden_qt_plus.view(T + 1, B, -1)
                        hidden_next_t = hidden_qt_plus[1:].reshape(-1, args.rnn_hidden_dim)

                        q1_next = qf_target.q1(torch.cat([hidden_next_t, next_actions], dim=-1))
                        q2_next = qf_target.q2(torch.cat([hidden_next_t, next_actions], dim=-1))
                        min_q_next = torch.min(q1_next, q2_next) - alpha_tensor * next_logp

                        target_q = rew_seq.reshape(-1, 1) + (1.0 - done_seq.reshape(-1, 1)) * args.gamma * min_q_next

                    q1_loss = F.mse_loss(q1, target_q)
                    q2_loss = F.mse_loss(q2, target_q)
                    q_loss = q1_loss + q2_loss

                    q_optimizer.zero_grad(set_to_none=True)
                    q_loss.backward()
                    nn.utils.clip_grad_norm_(qf.parameters(), args.max_grad_norm)
                    q_optimizer.step()

                    q1_loss_list.append(q1_loss.item())
                    q2_loss_list.append(q2_loss.item())
                    q_loss_list.append(q_loss.item())

                    # Policy update (delayed)
                    if update_step % args.policy_frequency == 0:
                        # Sample actions for current states (0..T-1) with grad
                        obs_cur_flat = obs_seq[:-1].reshape((-1,) + obs_seq.shape[2:])
                        done_cur_flat = done_before[:-1].reshape(-1)

                        new_actions, logp, _, _ = actor.sample(obs_cur_flat, actor_init_state, done_cur_flat, deterministic=False)

                        # Freeze critic weights for actor update, detach critic hidden
                        for p in qf.parameters():
                            p.requires_grad = False

                        hidden_q_detached = hidden_q.detach()
                        q1_pi = qf.q1(torch.cat([hidden_q_detached, new_actions], dim=-1))
                        q2_pi = qf.q2(torch.cat([hidden_q_detached, new_actions], dim=-1))
                        min_q_pi = torch.min(q1_pi, q2_pi)

                        actor_loss = (alpha_tensor * logp - min_q_pi).mean()

                        actor_optimizer.zero_grad(set_to_none=True)
                        actor_loss.backward()
                        nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                        actor_optimizer.step()

                        actor_loss_list.append(actor_loss.item())

                        for p in qf.parameters():
                            p.requires_grad = True

                        # Alpha update
                        if args.autotune:
                            alpha_loss = -(log_alpha * (logp + target_entropy).detach()).mean()
                            alpha_optimizer.zero_grad(set_to_none=True)
                            alpha_loss.backward()
                            alpha_optimizer.step()
                            alpha = log_alpha.exp().item()
                            alpha_loss_list.append(alpha_loss.item())

                    # Target update
                    if update_step % args.target_network_frequency == 0:
                        soft_update(qf_target, qf, args.tau)

        # Add chunk to replay buffer AFTER collecting the full sequence
        replay_buffer.add_chunk(
            obs_seq=chunk_obs,
            actions_seq=chunk_actions,
            rewards_seq=chunk_rewards,
            dones_seq=chunk_dones,
            init_dones=chunk_init_dones,
            actor_init_state=chunk_actor_init_state,
            q_init_state=chunk_q_init_state,
        )

        # End update logging (match PPO metric names)
        avg_inference_latency = inference_time_total / args.num_steps
        writer.add_scalar("metrics/inference_latency", avg_inference_latency, global_step)

        avg_q1_loss = float(np.mean(q1_loss_list)) if q1_loss_list else 0.0
        avg_q2_loss = float(np.mean(q2_loss_list)) if q2_loss_list else 0.0
        avg_q_loss = float(np.mean(q_loss_list)) if q_loss_list else 0.0
        avg_actor_loss = float(np.mean(actor_loss_list)) if actor_loss_list else 0.0
        avg_alpha_loss = float(np.mean(alpha_loss_list)) if alpha_loss_list else 0.0

        sps = int(global_step / (time.time() - start_time))
        current_return = np.mean([ep["r"] for ep in episode_infos]) if episode_infos else 0.0

        print(
            f"Update {update}: SPS={sps}, Return={current_return:.2f}, "
            f"q_loss={avg_q_loss:.6f}, actor_loss={avg_actor_loss:.6f}, alpha={alpha:.6f}"
        )

        writer.add_scalar("charts/learning_rate", actor_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/qf1_loss", avg_q1_loss, global_step)
        writer.add_scalar("losses/qf2_loss", avg_q2_loss, global_step)
        writer.add_scalar("losses/qf_loss", avg_q_loss, global_step)
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

        # GPU memory usage (match PPO names)
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

        # Save model checkpoint every save_interval updates (match PPO path)
        if args.save_model and update % args.save_interval == 0:
            model_path = f"runs/{run_name}/{args.exp_name}_update_{update}.cleanrl_model"
            model_data = {
                "model_weights": {
                    "actor": actor.state_dict(),
                    "qf": qf.state_dict(),
                    "qf_target": qf_target.state_dict(),
                    "log_alpha": log_alpha.detach().cpu() if args.autotune else None,
                },
                "args": vars(args),
            }
            torch.save(model_data, model_path)
            print(f"Model saved to {model_path}")

    finish_logging(args, writer, run_name, envs)