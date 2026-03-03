import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

from exp_utils import add_common_args, setup_logging, finish_logging
from env_utils import make_classic_env, make_continuous_env
from layers import layer_init


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    
    # SAC-specific arguments
    parser.add_argument("--hidden-dim", type=int, default=256,
        help="the hidden dimension of the neural network")
    parser.add_argument("--buffer-size", type=int, default=1000000,
        help="the replay memory buffer size")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient for target network update")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size for training")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1,
        help="the frequency of updates for the target networks")
    parser.add_argument("--alpha", type=float, default=0.2,
        help="entropy regularization coefficient (fixed if not auto-tuning)")
    parser.add_argument("--autotune", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of entropy coefficient")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--warmup-steps", type=int, default=1000,
        help="random action steps for warmup")
    parser.add_argument("--update-frequency", type=int, default=1,
        help="update network every N steps")
    parser.add_argument("--updates-per-step", type=int, default=1,
        help="number of updates per step")
    
    args = parser.parse_args()
    return args


class ReplayBuffer:
    """Simple replay buffer for off-policy learning."""
    
    def __init__(self, obs_shape, action_shape, buffer_size, device):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Preallocate memory
        self.observations = torch.zeros((buffer_size,) + obs_shape, dtype=torch.float32)
        self.next_observations = torch.zeros((buffer_size,) + obs_shape, dtype=torch.float32)
        self.actions = torch.zeros((buffer_size,) + action_shape, dtype=torch.float32)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32)
        
    def add(self, obs, next_obs, action, reward, done):
        self.observations[self.ptr] = torch.from_numpy(obs).float()
        self.next_observations[self.ptr] = torch.from_numpy(next_obs).float()
        self.actions[self.ptr] = torch.from_numpy(action).float()
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
        
    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        
        obs = self.observations[indices].to(self.device)
        next_obs = self.next_observations[indices].to(self.device)
        actions = self.actions[indices].to(self.device)
        rewards = self.rewards[indices].to(self.device)
        dones = self.dones[indices].to(self.device)
        
        return obs, next_obs, actions, rewards, dones
    
    def __len__(self):
        return self.size


class QNetwork(nn.Module):
    """Twin Q-networks for SAC."""
    
    def __init__(self, env, hidden_dim=256):
        super().__init__()
        
        self.obs_space = env.single_observation_space
        obs_shape = self.obs_space.shape
        input_dim = np.prod(obs_shape)
        action_dim = np.prod(env.single_action_space.shape)
        
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(input_dim + action_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        
        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(input_dim + action_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        
    def forward(self, obs, action):
        """Return Q1 and Q2 values."""
        # Handle observation preprocessing
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
            
        x = torch.cat([obs, action], dim=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2


class GaussianPolicy(nn.Module):
    """Gaussian policy for continuous action spaces (SAC actor)."""
    
    def __init__(self, env, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        
        self.obs_space = env.single_observation_space
        obs_shape = self.obs_space.shape
        input_dim = np.prod(obs_shape)
        action_dim = np.prod(env.single_action_space.shape)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_dim = action_dim
        
        # Get action bounds for scaling
        self.action_scale = torch.tensor(
            (env.single_action_space.high - env.single_action_space.low) / 2.0,
            dtype=torch.float32
        )
        self.action_bias = torch.tensor(
            (env.single_action_space.high + env.single_action_space.low) / 2.0,
            dtype=torch.float32
        )
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )
        
        # Mean and log_std heads
        self.mean_layer = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.log_std_layer = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        
    def forward(self, obs):
        """Return mean and log_std."""
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        hidden = self.backbone(obs)
        mean = self.mean_layer(hidden)
        log_std = self.log_std_layer(hidden)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, obs):
        """Sample action using reparameterization trick."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        
        # Reparameterization trick
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterized sample
        action = torch.tanh(x_t)
        
        # Compute log probability with correction for tanh squashing
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Scale action to environment bounds
        scaled_action = action * self.action_scale.to(action.device) + self.action_bias.to(action.device)
        
        return scaled_action, log_prob, mean
    
    def get_action(self, obs, deterministic=False):
        """Get action for inference."""
        with torch.no_grad():
            mean, log_std = self.forward(obs)
            if deterministic:
                action = torch.tanh(mean)
            else:
                std = log_std.exp()
                normal = Normal(mean, std)
                x_t = normal.sample()
                action = torch.tanh(x_t)
            
            # Scale to environment bounds
            scaled_action = action * self.action_scale.to(action.device) + self.action_bias.to(action.device)
        return scaled_action.cpu().numpy()


def soft_update(target, source, tau):
    """Soft update target network parameters."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    """Hard update target network parameters."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def train_sac(args):
    """Main SAC training loop."""
    
    # Setup
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_detinistic
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # Setup logging
    writer = None
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
        ),
    )
    
    # Environment setup - SAC typically uses single env for off-policy learning
    env = make_continuous_env(args.gym_id, args.seed, capture_video=args.capture_video, run_name=run_name)
    
    assert isinstance(env.action_space, gym.spaces.Box), "SAC only supports continuous action spaces"
    
    # Initialize networks
    qf1 = QNetwork(env, args.hidden_dim).to(device)
    qf2 = QNetwork(env, args.hidden_dim).to(device)
    qf1_target = QNetwork(env, args.hidden_dim).to(device)
    qf2_target = QNetwork(env, args.hidden_dim).to(device)
    
    # Copy weights to target networks
    hard_update(qf1_target, qf1)
    hard_update(qf2_target, qf2)
    
    actor = GaussianPolicy(env, args.hidden_dim).to(device)
    
    # Optimizers
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate)
    
    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -np.prod(env.action_space.shape).astype(np.float32)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        alpha_optimizer = optim.Adam([log_alpha], lr=args.learning_rate)
    else:
        alpha = args.alpha
    
    # Replay buffer
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    replay_buffer = ReplayBuffer(obs_shape, action_shape, args.buffer_size, device)
    
    # Training loop
    obs, info = env.reset(seed=args.seed)
    episode_return = 0
    episode_length = 0
    episode_count = 0
    
    print(f"Starting SAC training on {args.gym_id}")
    print(f"Device: {device}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    start_time = time.time()
    
    for global_step in range(args.total_timesteps):
        # Select action
        if global_step < args.warmup_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action = actor.get_action(obs_tensor, deterministic=False)[0]
        
        # Environment step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_return += reward
        episode_length += 1
        
        # Store transition
        replay_buffer.add(obs, next_obs, action, reward, float(done))
        
        obs = next_obs
        
        # Episode end
        if done:
            print(f"Step {global_step} | Episode {episode_count} | Return: {episode_return:.2f} | Length: {episode_length}")
            writer.add_scalar("charts/episode_return", episode_return, global_step)
            writer.add_scalar("charts/episode_length", episode_length, global_step)
            
            obs, info = env.reset()
            episode_return = 0
            episode_length = 0
            episode_count += 1
        
        # Training
        if global_step >= args.learning_starts and len(replay_buffer) >= args.batch_size:
            for _ in range(args.updates_per_step):
                # Sample batch
                obs_batch, next_obs_batch, actions_batch, rewards_batch, dones_batch = \
                    replay_buffer.sample(args.batch_size)
                
                # Compute target Q value
                with torch.no_grad():
                    next_actions, next_log_probs, _ = actor.sample(next_obs_batch)
                    qf1_next, qf2_next = qf1_target(next_obs_batch, next_actions)
                    min_qf_next = torch.min(qf1_next, qf2_next) - alpha * next_log_probs
                    next_q_value = rewards_batch.unsqueeze(1) + (1 - dones_batch.unsqueeze(1)) * args.gamma * min_qf_next
                
                # Update Q-functions
                qf1_values, qf2_values = qf1(obs_batch, actions_batch)
                qf1_loss = F.mse_loss(qf1_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss
                
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()
                
                # Update policy (delayed)
                if global_step % args.policy_frequency == 0:
                    new_actions, log_probs, _ = actor.sample(obs_batch)
                    qf1_new, qf2_new = qf1(obs_batch, new_actions)
                    min_qf_new = torch.min(qf1_new, qf2_new)
                    
                    actor_loss = ((alpha * log_probs) - min_qf_new).mean()
                    
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    
                    # Update alpha
                    if args.autotune:
                        with torch.no_grad():
                            _, log_probs, _ = actor.sample(obs_batch)
                        alpha_loss = -(log_alpha * (log_probs + target_entropy)).mean()
                        
                        alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        alpha_optimizer.step()
                        alpha = log_alpha.exp().item()
                    
                    # Logging
                    if global_step % 1000 == 0:
                        writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                        writer.add_scalar("losses/alpha", alpha, global_step)
                        if args.autotune:
                            writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                
                # Update target networks
                if global_step % args.target_network_frequency == 0:
                    soft_update(qf1_target, qf1, args.tau)
                    soft_update(qf2_target, qf2, args.tau)
                
                # Q-function logging
                if global_step % 1000 == 0:
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                    writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
        
        # SPS logging
        if global_step % 1000 == 0 and global_step > 0:
            elapsed = time.time() - start_time
            sps = global_step / elapsed
            writer.add_scalar("charts/SPS", sps, global_step)
            print(f"Step {global_step}/{args.total_timesteps} | SPS: {sps:.2f} | Alpha: {alpha:.4f}")
        
        # Save checkpoint
        if args.save_model and global_step % (args.save_interval * 1000) == 0 and global_step > 0:
            checkpoint_path = f"checkpoints/{run_name}/sac_checkpoint_{global_step}.pt"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'actor': actor.state_dict(),
                'qf1': qf1.state_dict(),
                'qf2': qf2.state_dict(),
                'qf1_target': qf1_target.state_dict(),
                'qf2_target': qf2_target.state_dict(),
            }, checkpoint_path)
            print(f"Saved checkpoint at step {global_step}")
    
    # Save final model
    if args.save_model:
        final_path = f"checkpoints/{run_name}/sac_final.pt"
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        torch.save({
            'actor': actor.state_dict(),
            'qf1': qf1.state_dict(),
            'qf2': qf2.state_dict(),
        }, final_path)
        print(f"Saved final model to {final_path}")
    
    # Cleanup
    env.close()
    writer.close()
    
    print(f"Training completed! Total steps: {args.total_timesteps}")


if __name__ == "__main__":
    args = parse_args()
    train_sac(args)
