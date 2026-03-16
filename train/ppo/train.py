"""PPO training script for the organism evolutionary environment.

Usage:
    python train/ppo/train.py
    python train/ppo/train.py --num-envs 64 --num-agents 8 --train-time 600
"""

import argparse
import math
import time
from dataclasses import dataclass

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import organism_env

NUM_ACTIONS = 3  # (ax, ay, view_delta)
VIEW_RES = 32
TOTAL_CHANNELS = 16  # 4 object channels × 4 temporal frames
NUM_SCALAR_FEATURES = 4  # energy, vx, vy, view_size
MEMORY_DIM = 128


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig:
    # Environment
    num_envs: int = 32
    num_agents: int = 20
    env_width: float = 30.0
    env_height: float = 30.0
    food_spawn_rate: float = 100.0
    dt: float = 0.15
    energy_loss: float = 0.02
    num_obstacles: int = 3
    obstacle_radius: float = 3.0
    obstacle_weight: float = 50.0
    wall_velocity_damping: float = 0.5
    food_cap: int = 200  # 75% reduction from 800
    vision_cost: float = 0.001
    initial_view_size: float = 3.0
    min_view_size: float = 2.0
    object_radius: float = 0.3
    energy_decay_rate: float = 1.0
    memory_decay_rate: float = 0.99
    reset_interval: int = 256

    # PPO
    train_time: float = 600.0
    rollout_len: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.05
    max_grad_norm: float = 0.5
    num_epochs: int = 4
    num_minibatches: int = 4
    lr: float = 3e-4
    anneal_lr: bool = True

    # Misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    render_every: int = 100
    video_interval: float = 300.0  # save a video every N seconds
    large_run: bool = False
    resume_path: str = ""
    agent_collision: bool = False

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_agents * self.rollout_len


# ---------------------------------------------------------------------------
# Network — CNN actor-critic
# ---------------------------------------------------------------------------


class ActorCritic(nn.Module):
    """CNN actor-critic with memory reservoir.

    Input: (B, H, W, C) image + (B, 4) scalars + (B, 128) memory_in
    Output: action, log_prob, entropy, value, memory_out

    Memory is concatenated with CNN flatten output before FC layers.
    Gradient does NOT cross steps — memory_in is always detached.
    """

    def __init__(self, in_channels: int = TOTAL_CHANNELS, act_dim: int = NUM_ACTIONS,
                 scalar_dim: int = NUM_SCALAR_FEATURES, memory_dim: int = MEMORY_DIM,
                 large: bool = False):
        super().__init__()
        self.memory_dim = memory_dim

        if large:
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveMaxPool2d(2),
                nn.Flatten(),
            )
            cnn_out = 128 * 2 * 2
            fc_hidden = 576
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveMaxPool2d(2),
                nn.Flatten(),
            )
            cnn_out = 64 * 2 * 2
            fc_hidden = 256

        # CNN flatten + scalars + memory_in → FC
        self.fc = nn.Sequential(
            nn.Linear(cnn_out + scalar_dim + memory_dim, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, fc_hidden),
            nn.ReLU(),
        )

        self.actor_mean = nn.Linear(fc_hidden, act_dim)
        self.critic_head = nn.Linear(fc_hidden, 1)
        self.memory_head = nn.Linear(fc_hidden, memory_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.0))

        # Init
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)
        # Memory head starts near zero so initial memory writes are small
        nn.init.orthogonal_(self.memory_head.weight, gain=0.01)
        nn.init.zeros_(self.memory_head.bias)

    def _encode(self, obs: torch.Tensor, scalars: torch.Tensor, memory_in: torch.Tensor) -> torch.Tensor:
        x = obs.permute(0, 3, 1, 2).contiguous()
        cnn_out = self.cnn(x)
        # Memory is concatenated at the CNN flatten level (before FC)
        # NOTE: memory_in must be detached before calling this — no gradient crosses steps
        combined = torch.cat([cnn_out, scalars, memory_in], dim=-1)
        return self.fc(combined)

    def get_action_and_value(
        self, obs: torch.Tensor, scalars: torch.Tensor, memory_in: torch.Tensor,
        action: torch.Tensor | None = None
    ):
        h = self._encode(obs, scalars, memory_in)
        mean = self.actor_mean(h)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic_head(h).squeeze(-1)
        memory_out = self.memory_head(h)
        return action, log_prob, entropy, value, memory_out


# ---------------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------------


class RolloutBuffer:
    def __init__(self, cfg: PPOConfig):
        n = cfg.num_envs * cfg.num_agents
        T = cfg.rollout_len
        dev = cfg.device

        self.obs = torch.zeros(T, n, VIEW_RES, VIEW_RES, TOTAL_CHANNELS, device=dev)
        self.scalars = torch.zeros(T, n, NUM_SCALAR_FEATURES, device=dev)
        self.memory = torch.zeros(T, n, MEMORY_DIM, device=dev)
        self.actions = torch.zeros(T, n, NUM_ACTIONS, device=dev)
        self.log_probs = torch.zeros(T, n, device=dev)
        self.rewards = torch.zeros(T, n, device=dev)
        self.values = torch.zeros(T, n, device=dev)
        self.alive = torch.zeros(T, n, device=dev)
        self.cfg = cfg
        self.step = 0

    def insert(self, obs, scalars, memory, actions, log_probs, rewards, values, alive):
        self.obs[self.step] = obs
        self.scalars[self.step] = scalars
        self.memory[self.step] = memory
        self.actions[self.step] = actions
        self.log_probs[self.step] = log_probs
        self.rewards[self.step] = rewards
        self.values[self.step] = values
        self.alive[self.step] = alive
        self.step += 1

    def compute_gae(self, next_value: torch.Tensor, next_alive: torch.Tensor):
        T = self.cfg.rollout_len
        gamma, lam = self.cfg.gamma, self.cfg.gae_lambda
        advantages = torch.zeros_like(self.rewards)
        last_gae = torch.zeros_like(next_value)

        for t in reversed(range(T)):
            next_val = next_value if t == T - 1 else self.values[t + 1]
            next_mask = next_alive if t == T - 1 else self.alive[t + 1]
            mask = self.alive[t]
            delta = self.rewards[t] + gamma * next_val * next_mask - self.values[t]
            last_gae = delta + gamma * lam * next_mask * last_gae
            advantages[t] = last_gae * mask

        self.advantages = advantages
        self.returns = advantages + self.values
        self.step = 0

    def get_batches(self, num_minibatches: int):
        total = self.cfg.rollout_len * self.cfg.num_envs * self.cfg.num_agents
        obs = self.obs.reshape(total, VIEW_RES, VIEW_RES, TOTAL_CHANNELS)
        scalars = self.scalars.reshape(total, NUM_SCALAR_FEATURES)
        memory = self.memory.reshape(total, MEMORY_DIM)
        actions = self.actions.reshape(total, -1)
        log_probs = self.log_probs.reshape(total)
        advantages = self.advantages.reshape(total)
        returns = self.returns.reshape(total)
        alive = self.alive.reshape(total)

        indices = torch.randperm(total, device=self.cfg.device)
        mb_size = total // num_minibatches

        for start in range(0, total, mb_size):
            idx = indices[start : start + mb_size]
            yield (obs[idx], scalars[idx], memory[idx], actions[idx],
                   log_probs[idx], advantages[idx], returns[idx], alive[idx])


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def make_env(cfg: PPOConfig) -> organism_env.EvolutionEnv:
    return organism_env.EvolutionEnv.initialize({
        "num_organisms": cfg.num_agents,
        "height": cfg.env_height,
        "width": cfg.env_width,
        "food_spawn_rate": cfg.food_spawn_rate,
        "num_copies": cfg.num_envs,
        "dt": cfg.dt,
        "energy_loss": cfg.energy_loss,
        "wall_velocity_damping": cfg.wall_velocity_damping,
        "object_radius": cfg.object_radius,
        "num_obstacles": cfg.num_obstacles,
        "obstacle_radius": cfg.obstacle_radius,
        "obstacle_weight": cfg.obstacle_weight,
        "food_cap": cfg.food_cap,
        "vision_cost": cfg.vision_cost,
        "initial_view_size": cfg.initial_view_size,
        "min_view_size": cfg.min_view_size,
        "energy_decay_rate": cfg.energy_decay_rate,
        "rules": {"agent_collision": cfg.agent_collision},
    })


def env_observe(env, device: str):
    """Returns (obs, scalars, alive) as flat tensors on device."""
    obs_np = env.observe()       # (num_envs, max_agents, H, W, C)
    states_np = env.agent_states()  # (num_envs, max_agents, 4)
    mask_np = env.alive_mask()   # (num_envs, max_agents)
    n = obs_np.shape[0] * obs_np.shape[1]
    obs = torch.from_numpy(obs_np).reshape(n, VIEW_RES, VIEW_RES, TOTAL_CHANNELS).to(device)
    scalars = torch.from_numpy(states_np).reshape(n, NUM_SCALAR_FEATURES).to(device)
    alive = torch.from_numpy(mask_np).reshape(n).to(device)
    return obs, scalars, alive


def env_step(env, actions: torch.Tensor, num_envs: int, num_agents: int, device: str):
    """Step env, return (energy, obs, scalars, alive)."""
    act_np = actions.reshape(num_envs, num_agents, NUM_ACTIONS).cpu().numpy()
    energy_np = env.step(act_np)
    obs_np = env.observe()
    states_np = env.agent_states()
    mask_np = env.alive_mask()
    n = num_envs * num_agents
    energy = torch.from_numpy(energy_np).reshape(n).to(device)
    obs = torch.from_numpy(obs_np).reshape(n, VIEW_RES, VIEW_RES, TOTAL_CHANNELS).to(device)
    scalars = torch.from_numpy(states_np).reshape(n, NUM_SCALAR_FEATURES).to(device)
    alive = torch.from_numpy(mask_np).reshape(n).to(device)
    return energy, obs, scalars, alive


# ---------------------------------------------------------------------------
# Inference loop → video
# ---------------------------------------------------------------------------


def inference_loop(
    model: ActorCritic,
    cfg: PPOConfig,
    output_path: str = "train/ppo/final.mp4",
    max_realtime_seconds: float = 60.0,
    target_resolution: int = 800,
    seed: int = 9999,
    speed_multiplier: float = 3.0,
):
    video_dt = cfg.dt
    num_steps = int(max_realtime_seconds * speed_multiplier / video_dt)
    fps = speed_multiplier / video_dt

    # Scale pixels_per_unit so the image fits target_resolution
    max_dim = max(cfg.env_width, cfg.env_height)
    pixels_per_unit = target_resolution / max_dim
    # Round to multiple of 16 for codec compatibility
    raw_px = int(max_dim * pixels_per_unit)
    if raw_px % 16 != 0:
        pixels_per_unit = (((raw_px + 15) // 16) * 16) / max_dim

    video_env = organism_env.EvolutionEnv.initialize({
        "num_organisms": cfg.num_agents,
        "height": cfg.env_height,
        "width": cfg.env_width,
        "food_spawn_rate": cfg.food_spawn_rate,
        "num_copies": 1,
        "dt": video_dt,
        "energy_loss": cfg.energy_loss,
        "wall_velocity_damping": cfg.wall_velocity_damping,
        "object_radius": cfg.object_radius,
        "num_obstacles": cfg.num_obstacles,
        "obstacle_radius": cfg.obstacle_radius,
        "obstacle_weight": cfg.obstacle_weight,
        "food_cap": cfg.food_cap,
        "vision_cost": cfg.vision_cost,
        "initial_view_size": cfg.initial_view_size,
        "min_view_size": cfg.min_view_size,
        "energy_decay_rate": cfg.energy_decay_rate,
        "rules": {"agent_collision": cfg.agent_collision},
        "seed": seed,
    })

    model.eval()
    device = cfg.device

    frames = []
    n_video = cfg.num_agents
    obs, scalars, _ = env_observe(video_env, device)
    memory = torch.zeros(n_video, MEMORY_DIM, device=device)
    factor = cfg.memory_decay_rate ** video_dt

    with torch.no_grad():
        for step in range(num_steps):
            frame = video_env.render_array(env_index=0, pixels_per_unit=pixels_per_unit)
            frames.append(frame)

            action, _, _, _, mem_out = model.get_action_and_value(obs, scalars, memory)
            # Memory update — no gradient needed (inference)
            memory = memory * factor + (1 - factor) * mem_out

            act_np = action.reshape(1, cfg.num_agents, NUM_ACTIONS).cpu().numpy()
            video_env.step(act_np)
            obs, scalars, _ = env_observe(video_env, device)

            if (step + 1) % cfg.reset_interval == 0:
                video_env.reset()
                obs, scalars, _ = env_observe(video_env, device)
                memory = torch.zeros(n_video, MEMORY_DIM, device=device)

    frames_arr = np.stack(frames)
    iio.imwrite(output_path, frames_arr, fps=fps, codec="libx264")
    duration = num_steps * video_dt
    video_duration = len(frames) / fps
    print(f"Video saved: {output_path} ({len(frames)} frames, {duration:.1f}s sim, {video_duration:.1f}s video, {speed_multiplier:.0f}x speed)")


# ---------------------------------------------------------------------------
# Evaluation: memory vs no-memory
# ---------------------------------------------------------------------------


def evaluate_memory(model: ActorCritic, cfg: PPOConfig, num_episodes: int = 30):
    """Run episodes with and without memory to check if the agent uses it.

    Returns (mean_with, mean_without, p_value).
    Uses a two-sample t-test for statistical significance.
    """
    from scipy import stats
    device = cfg.device
    model.eval()

    eval_env = organism_env.EvolutionEnv.initialize({
        "num_organisms": cfg.num_agents,
        "height": cfg.env_height, "width": cfg.env_width,
        "food_spawn_rate": cfg.food_spawn_rate, "num_copies": 1,
        "dt": cfg.dt, "energy_loss": cfg.energy_loss,
        "wall_velocity_damping": cfg.wall_velocity_damping,
        "object_radius": cfg.object_radius,
        "num_obstacles": cfg.num_obstacles,
        "obstacle_radius": cfg.obstacle_radius,
        "obstacle_weight": cfg.obstacle_weight,
        "food_cap": cfg.food_cap, "vision_cost": cfg.vision_cost,
        "initial_view_size": cfg.initial_view_size,
        "min_view_size": cfg.min_view_size,
        "energy_decay_rate": cfg.energy_decay_rate,
        "rules": {"agent_collision": cfg.agent_collision},
    })
    factor = cfg.memory_decay_rate ** cfg.dt

    def run_one_episode(use_memory: bool) -> float:
        eval_env.reset()
        obs, scalars, alive = env_observe(eval_env, device)
        memory = torch.zeros(cfg.num_agents, MEMORY_DIM, device=device)
        total_reward = 0.0
        prev_e = alive * 10.0

        with torch.no_grad():
            for step in range(cfg.reset_interval):
                action, _, _, _, mem_out = model.get_action_and_value(obs, scalars, memory)
                if use_memory:
                    memory = memory * factor + (1 - factor) * mem_out

                act_np = action.reshape(1, cfg.num_agents, NUM_ACTIONS).cpu().numpy()
                energy_np = eval_env.step(act_np)
                energy = torch.from_numpy(energy_np).reshape(-1).to(device)
                obs, scalars, alive = env_observe(eval_env, device)

                reward = ((energy - prev_e) * alive).sum().item()
                prev_e = energy.clone()
                total_reward += reward

                if eval_env.all_dead():
                    break

        return total_reward / cfg.num_agents

    returns_with = [run_one_episode(use_memory=True) for _ in range(num_episodes)]
    returns_without = [run_one_episode(use_memory=False) for _ in range(num_episodes)]
    mean_with = np.mean(returns_with)
    mean_without = np.mean(returns_without)
    t_stat, p_value = stats.ttest_ind(returns_with, returns_without)
    return mean_with, mean_without, p_value


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(cfg: PPOConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = make_env(cfg)
    agent = ActorCritic(large=cfg.large_run).to(cfg.device)
    param_count = sum(p.numel() for p in agent.parameters())
    print(f"         model params: {param_count:,} ({'large' if cfg.large_run else 'small'})")
    if cfg.resume_path:
        agent.load_state_dict(torch.load(cfg.resume_path, map_location=cfg.device, weights_only=True))
        print(f"         resumed from {cfg.resume_path}")
    optimizer = optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)
    buf = RolloutBuffer(cfg)

    import os
    os.makedirs("train/ppo/videos", exist_ok=True)

    from collections import deque

    n = cfg.num_envs * cfg.num_agents
    obs, scalars, alive = env_observe(env, cfg.device)
    memory = torch.zeros(n, MEMORY_DIM, device=cfg.device)
    mem_factor = cfg.memory_decay_rate ** cfg.dt
    prev_energy = alive * 10.0
    ep_return_accum = torch.zeros(n, device=cfg.device)
    ep_length = 0
    recent_ep_returns = deque(maxlen=100)
    global_step = 0
    env_steps_since_reset = 0
    start_time = time.time()
    last_video_time = start_time
    video_count = 0
    update = 0

    print(f"PPO-CNN | device={cfg.device} | envs={cfg.num_envs} | agents={cfg.num_agents}")
    print(f"         obs=({VIEW_RES},{VIEW_RES},{TOTAL_CHANNELS}) | actions={NUM_ACTIONS}")
    print(f"         rollout={cfg.rollout_len} | batch={cfg.batch_size}")
    print(f"         training for {cfg.train_time:.0f}s wall-clock")
    print()

    # --- Initial video (random policy, short) ---
    print("Generating initial video (untrained policy)...")
    inference_loop(model=agent, cfg=cfg, output_path="train/ppo/videos/initial_step0.mp4", max_realtime_seconds=20.0)

    while (time.time() - start_time) < cfg.train_time:
        update += 1

        if cfg.anneal_lr:
            elapsed = time.time() - start_time
            frac = elapsed / cfg.train_time
            optimizer.param_groups[0]["lr"] = cfg.lr * 0.5 * (1.0 + math.cos(math.pi * frac))

        # --- Collect rollout ---
        rollout_reward_sum = 0.0
        rollout_reward_count = 0

        for step in range(cfg.rollout_len):
            global_step += n
            env_steps_since_reset += 1
            ep_length += 1

            with torch.no_grad():
                # NOTE: memory is detached — no gradient crosses steps
                action, log_prob, _, value, mem_out = agent.get_action_and_value(
                    obs, scalars, memory.detach()
                )
                # Memory reservoir update: exponential moving average
                # factor = decay_rate^dt, new = old * factor + (1 - factor) * output
                # Detached: gradient does not flow through memory across steps
                new_memory = memory.detach() * mem_factor + (1 - mem_factor) * mem_out.detach()

            energy, next_obs, next_scalars, next_alive = env_step(
                env, action, cfg.num_envs, cfg.num_agents, cfg.device
            )
            reward = (energy - prev_energy) * alive
            prev_energy = energy.clone()

            buf.insert(obs, scalars, memory.detach(), action, log_prob, reward, value, alive)
            memory = new_memory
            ep_return_accum += reward
            obs = next_obs
            scalars = next_scalars

            # Track mean step reward for alive agents
            alive_count_step = (alive > 0).sum().item()
            if alive_count_step > 0:
                rollout_reward_sum += reward[alive > 0].sum().item()
                rollout_reward_count += alive_count_step

            alive = next_alive

            should_reset = env_steps_since_reset >= cfg.reset_interval or env.all_dead()
            if should_reset:
                # Record per-env mean episode return
                mask = torch.from_numpy(env.alive_mask()).reshape(cfg.num_envs, cfg.num_agents).to(cfg.device)
                per_env = ep_return_accum.reshape(cfg.num_envs, cfg.num_agents)
                # Use the alive mask from BEFORE reset to know which agents were real
                initial_mask = torch.ones_like(mask)  # all agents were initialized alive
                for e in range(cfg.num_envs):
                    n_agents = min(cfg.num_agents, per_env.shape[1])
                    agent_returns = per_env[e, :n_agents]
                    if n_agents > 0:
                        recent_ep_returns.append(agent_returns.mean().item())

                env.reset()
                obs, scalars, alive = env_observe(env, cfg.device)
                memory = torch.zeros(n, MEMORY_DIM, device=cfg.device)
                prev_energy = alive * 10.0
                ep_return_accum = torch.zeros(n, device=cfg.device)
                ep_length = 0
                env_steps_since_reset = 0

        # --- Compute advantages ---
        with torch.no_grad():
            _, _, _, next_value, _ = agent.get_action_and_value(obs, scalars, memory.detach())
        buf.compute_gae(next_value, alive)

        # --- PPO update ---
        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_ent_loss = 0.0
        total_clip_frac = 0.0
        num_batches = 0

        for epoch in range(cfg.num_epochs):
            for mb_obs, mb_scalars, mb_mem, mb_act, mb_old_lp, mb_adv, mb_ret, mb_alive in buf.get_batches(
                cfg.num_minibatches
            ):
                # NOTE: mb_mem is already detached (stored detached in buffer)
                # No gradient flows through memory across steps
                _, new_lp, entropy, new_val, _ = agent.get_action_and_value(mb_obs, mb_scalars, mb_mem, mb_act)

                # Normalize advantages over alive agents
                alive_mask = mb_alive > 0
                if alive_mask.sum() > 1:
                    adv_alive = mb_adv[alive_mask]
                    mb_adv = torch.where(
                        alive_mask,
                        (mb_adv - adv_alive.mean()) / (adv_alive.std() + 1e-8),
                        torch.zeros_like(mb_adv),
                    )

                ratio = (new_lp - mb_old_lp).exp()
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps)
                pg_loss = (
                    (torch.max(pg_loss1, pg_loss2) * mb_alive).sum()
                    / mb_alive.sum().clamp(min=1)
                )

                vf_loss = (
                    ((new_val - mb_ret) ** 2 * mb_alive).sum()
                    / mb_alive.sum().clamp(min=1)
                )
                ent_loss = (entropy * mb_alive).sum() / mb_alive.sum().clamp(min=1)

                loss = pg_loss + cfg.vf_coef * vf_loss - cfg.ent_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_ent_loss += ent_loss.item()
                total_clip_frac += clip_frac
                num_batches += 1

        # --- Logging ---
        elapsed = time.time() - start_time
        sps = global_step / elapsed
        mean_ep_return = np.mean(recent_ep_returns) if recent_ep_returns else float("nan")
        mean_step_rew = rollout_reward_sum / max(rollout_reward_count, 1)

        with torch.no_grad():
            cur_obs, cur_scalars, cur_alive = env_observe(env, cfg.device)
            _, _, _, cur_values, _ = agent.get_action_and_value(cur_obs, cur_scalars, memory.detach())
            alive_mask = cur_alive > 0
            alive_count = int(alive_mask.sum().item())
            mean_value = cur_values[alive_mask].mean().item() if alive_count > 0 else 0.0
            alive_energies = prev_energy[alive_mask]
            mean_energy = alive_energies.mean().item() if alive_count > 0 else 0.0

        if update % 10 == 0 or update <= 3:
            print(
                f"update {update:4d} | "
                f"t={elapsed:5.0f}s/{cfg.train_time:.0f}s | "
                f"step {global_step:>10,} | "
                f"sps {sps:>8,.0f} | "
                f"ep_ret {mean_ep_return:+7.2f} | "
                f"r/step {mean_step_rew:+.4f} | "
                f"V(s) {mean_value:+7.2f} | "
                f"energy {mean_energy:+7.2f} | "
                f"pg {total_pg_loss / num_batches:+.4f} | "
                f"vf {total_vf_loss / num_batches:.3f} | "
                f"clip {total_clip_frac / num_batches:.3f} | "
                f"alive {alive_count}/{n} | "
                f"lr {optimizer.param_groups[0]['lr']:.1e}"
            )

        if cfg.render_every > 0 and update % cfg.render_every == 0:
            fname = f"train/ppo/render_update_{update:05d}.png"
            ppu = 800.0 / max(cfg.env_width, cfg.env_height)
            env.render(fname, env_index=0, pixels_per_unit=ppu)
            print(f"  -> saved {fname}")

        # --- Periodic video checkpoint ---
        if (time.time() - last_video_time) >= cfg.video_interval:
            video_count += 1
            vpath = f"train/ppo/videos/checkpoint_{video_count:03d}_step{global_step}.mp4"
            print(f"  Generating video checkpoint {video_count}...")
            inference_loop(model=agent, cfg=cfg, output_path=vpath)
            # Evaluate memory vs no-memory
            rw, rwo, pv = evaluate_memory(agent, cfg, num_episodes=15)
            print(f"  Memory eval: with={rw:+.2f} without={rwo:+.2f} delta={rw-rwo:+.2f} p={pv:.4f}")
            last_video_time = time.time()

    # --- Save model ---
    torch.save(agent.state_dict(), "train/ppo/model_final.pt")
    total_elapsed = time.time() - start_time
    print(f"\nTraining done: {update} updates, {global_step:,} steps in {total_elapsed:.1f}s")
    print(f"Model saved to train/ppo/model_final.pt")

    # --- Evaluate memory usage ---
    print("\nEvaluating memory reservoir usage...")
    ret_with, ret_without, p_val = evaluate_memory(agent, cfg)
    print(f"  With memory:    ep_return = {ret_with:+.2f}")
    print(f"  Without memory: ep_return = {ret_without:+.2f}")
    print(f"  Delta:          {ret_with - ret_without:+.2f} (p={p_val:.4f}, {'significant' if p_val < 0.05 else 'not significant'})")

    # --- Generate final video ---
    print("\nGenerating final video...")
    video_count += 1
    inference_loop(
        model=agent,
        cfg=cfg,
        output_path=f"train/ppo/videos/final_step{global_step}.mp4",
    )

    return agent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="PPO-CNN training for organism environment")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-agents", type=int, default=20)
    parser.add_argument("--train-time", type=float, default=600.0)
    parser.add_argument("--rollout-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-every", type=int, default=100)
    parser.add_argument("--reset-interval", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--large-run", action="store_true", help="Use ~1M param model")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--agent-collision", action="store_true", help="Enable agent-agent collision")
    args = parser.parse_args()

    cfg = PPOConfig(
        num_envs=args.num_envs,
        num_agents=args.num_agents,
        train_time=args.train_time,
        rollout_len=args.rollout_len,
        lr=args.lr,
        num_epochs=args.num_epochs,
        seed=args.seed,
        render_every=args.render_every,
        reset_interval=args.reset_interval,
        device=args.device,
        large_run=args.large_run,
        resume_path=args.resume,
        agent_collision=args.agent_collision,
    )
    train(cfg)


if __name__ == "__main__":
    main()
