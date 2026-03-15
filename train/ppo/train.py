"""PPO training script for the organism evolutionary environment.

Usage:
    python train/ppo/train.py
    python train/ppo/train.py --num-envs 64 --num-agents 8 --train-time 600
"""

import argparse
import time
from dataclasses import dataclass

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import organism_env

# Feature indices from the environment
X, Y, VX, VY, ALIVE = 0, 1, 2, 3, 4
NUM_OBS_FEATURES = 5
NUM_ACTIONS = 2  # (ax, ay)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig:
    # Environment
    num_envs: int = 32
    num_agents: int = 5
    env_width: float = 15.0
    env_height: float = 15.0
    food_spawn_rate: float = 25.0  # items per second of sim time
    dt: float = 0.2
    energy_loss: float = 0.02
    num_obstacles: int = 3
    food_cap: int = 100
    reset_interval: int = 1024

    # PPO
    train_time: float = 600.0  # seconds of wall-clock training
    rollout_len: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    num_epochs: int = 4
    num_minibatches: int = 4
    lr: float = 3e-4
    anneal_lr: bool = True

    # Misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    render_every: int = 100  # render a snapshot every N updates (0 to disable)

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_agents * self.rollout_len


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = NUM_OBS_FEATURES, act_dim: int = NUM_ACTIONS):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(128, act_dim)
        self.critic = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        for layer in [*self.backbone, self.actor_mean, self.critic]:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        h = self.backbone(obs)
        mean = self.actor_mean(h)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(h).squeeze(-1)
        return action, log_prob, entropy, value


# ---------------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------------


class RolloutBuffer:
    def __init__(self, cfg: PPOConfig):
        n = cfg.num_envs * cfg.num_agents
        T = cfg.rollout_len
        dev = cfg.device

        self.obs = torch.zeros(T, n, NUM_OBS_FEATURES, device=dev)
        self.actions = torch.zeros(T, n, NUM_ACTIONS, device=dev)
        self.log_probs = torch.zeros(T, n, device=dev)
        self.rewards = torch.zeros(T, n, device=dev)
        self.values = torch.zeros(T, n, device=dev)
        self.alive = torch.zeros(T, n, device=dev)
        self.cfg = cfg
        self.step = 0

    def insert(self, obs, actions, log_probs, rewards, values, alive):
        self.obs[self.step] = obs
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
        obs = self.obs.reshape(total, -1)
        actions = self.actions.reshape(total, -1)
        log_probs = self.log_probs.reshape(total)
        advantages = self.advantages.reshape(total)
        returns = self.returns.reshape(total)
        alive = self.alive.reshape(total)

        indices = torch.randperm(total, device=self.cfg.device)
        mb_size = total // num_minibatches

        for start in range(0, total, mb_size):
            idx = indices[start : start + mb_size]
            yield obs[idx], actions[idx], log_probs[idx], advantages[idx], returns[idx], alive[idx]


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def make_env(cfg: PPOConfig, seed: int | None = None) -> organism_env.EvolutionEnv:
    config = {
        "num_organisms": cfg.num_agents,
        "height": cfg.env_height,
        "width": cfg.env_width,
        "food_spawn_rate": cfg.food_spawn_rate,
        "num_copies": cfg.num_envs,
        "dt": cfg.dt,
        "energy_loss": cfg.energy_loss,
        "num_obstacles": cfg.num_obstacles,
        "food_cap": cfg.food_cap,
    }
    if seed is not None:
        config["seed"] = seed
    return organism_env.EvolutionEnv.initialize(config)


def env_observe(env, device: str) -> torch.Tensor:
    return torch.from_numpy(env.observe()).reshape(-1, NUM_OBS_FEATURES).to(device)


def env_step(env, actions: torch.Tensor, num_envs: int, num_agents: int, device: str):
    act_np = actions.reshape(num_envs, num_agents, NUM_ACTIONS).cpu().numpy()
    energy_np = env.step(act_np)
    obs_np = env.observe()
    return (
        torch.from_numpy(energy_np).reshape(-1).to(device),
        torch.from_numpy(obs_np).reshape(-1, NUM_OBS_FEATURES).to(device),
    )


# ---------------------------------------------------------------------------
# Inference loop → video
# ---------------------------------------------------------------------------


def inference_loop(
    model: ActorCritic,
    cfg: PPOConfig,
    output_path: str = "train/ppo/final.mp4",
    video_dt: float = 0.04,
    num_steps: int = 1024,
    pixels_per_unit: float = 40.0,
    seed: int = 9999,
):
    """Run the trained policy and capture frames into an mp4 video.

    Uses a lower dt for smooth rendering. Video frame rate is 1:1 with
    simulation time (fps = 1 / video_dt).
    """
    fps = 1.0 / video_dt

    # Use a resolution divisible by 16 (macro_block_size) to avoid ffmpeg resize warning
    raw_px = int(cfg.env_width * pixels_per_unit)
    if raw_px % 16 != 0:
        pixels_per_unit = (((raw_px + 15) // 16) * 16) / cfg.env_width

    # Create a single-copy env with the low dt for smooth video
    video_env = organism_env.EvolutionEnv.initialize({
        "num_organisms": cfg.num_agents,
        "height": cfg.env_height,
        "width": cfg.env_width,
        "food_spawn_rate": cfg.food_spawn_rate,
        "num_copies": 1,
        "dt": video_dt,
        "energy_loss": cfg.energy_loss,
        "num_obstacles": cfg.num_obstacles,
        "food_cap": cfg.food_cap,
        "seed": seed,
    })

    model.eval()
    device = cfg.device

    frames = []
    obs = torch.from_numpy(video_env.observe()).reshape(-1, NUM_OBS_FEATURES).to(device)

    with torch.no_grad():
        for step in range(num_steps):
            # Capture frame
            frame = video_env.render_array(env_index=0, pixels_per_unit=pixels_per_unit)
            frames.append(frame)

            # Get action from policy
            action, _, _, _ = model.get_action_and_value(obs)

            # Step
            act_np = action.reshape(1, cfg.num_agents, NUM_ACTIONS).cpu().numpy()
            video_env.step(act_np)
            obs = torch.from_numpy(video_env.observe()).reshape(-1, NUM_OBS_FEATURES).to(device)

            # Reset every 1024 steps
            if (step + 1) % 1024 == 0:
                video_env.reset()
                obs = torch.from_numpy(video_env.observe()).reshape(-1, NUM_OBS_FEATURES).to(device)

    # Write video
    frames_arr = np.stack(frames)
    iio.imwrite(output_path, frames_arr, fps=fps, codec="libx264")
    duration = num_steps * video_dt
    print(f"Video saved: {output_path} ({len(frames)} frames, {duration:.1f}s sim time, {fps:.0f} fps)")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(cfg: PPOConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = make_env(cfg)
    agent = ActorCritic().to(cfg.device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)
    buf = RolloutBuffer(cfg)

    n = cfg.num_envs * cfg.num_agents
    obs = env_observe(env, cfg.device)
    prev_energy = obs[:, ALIVE] * 10.0
    ep_return_accum = torch.zeros(n, device=cfg.device)
    global_step = 0
    env_steps_since_reset = 0
    start_time = time.time()
    update = 0

    print(f"PPO | device={cfg.device} | envs={cfg.num_envs} | agents={cfg.num_agents}")
    print(f"     rollout={cfg.rollout_len} | batch={cfg.batch_size}")
    print(f"     training for {cfg.train_time:.0f}s wall-clock")
    print()

    while (time.time() - start_time) < cfg.train_time:
        update += 1

        # Anneal LR based on time fraction
        if cfg.anneal_lr:
            elapsed = time.time() - start_time
            frac = 1.0 - elapsed / cfg.train_time
            optimizer.param_groups[0]["lr"] = cfg.lr * max(frac, 0.0)

        # --- Collect rollout ---
        # Episode return tracking: accumulate rewards between resets
        rollout_ep_returns = []  # completed episode returns this rollout

        for step in range(cfg.rollout_len):
            global_step += n
            env_steps_since_reset += 1

            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(obs)

            alive = obs[:, ALIVE]
            energy, next_obs = env_step(env, action, cfg.num_envs, cfg.num_agents, cfg.device)
            reward = (energy - prev_energy) * alive
            prev_energy = energy.clone()

            buf.insert(obs, action, log_prob, reward, value, alive)
            ep_return_accum += reward
            obs = next_obs

            # Reset with different seed every reset_interval steps
            if env_steps_since_reset >= cfg.reset_interval:
                # Record completed episode returns (mean per alive agent per env)
                alive_mask = ep_return_accum != 0
                if alive_mask.any():
                    # Reshape to (num_envs, num_agents) and take per-env mean
                    per_env = ep_return_accum.reshape(cfg.num_envs, cfg.num_agents)
                    for e in range(cfg.num_envs):
                        agent_returns = per_env[e][per_env[e] != 0]
                        if len(agent_returns) > 0:
                            rollout_ep_returns.append(agent_returns.mean().item())

                env.reset()
                obs = env_observe(env, cfg.device)
                prev_energy = obs[:, ALIVE] * 10.0
                ep_return_accum = torch.zeros(n, device=cfg.device)
                env_steps_since_reset = 0

        # --- Compute advantages ---
        with torch.no_grad():
            _, _, _, next_value = agent.get_action_and_value(obs)
            next_alive = obs[:, ALIVE]
        buf.compute_gae(next_value, next_alive)

        # --- PPO update ---
        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_ent_loss = 0.0
        total_clip_frac = 0.0
        num_batches = 0

        for epoch in range(cfg.num_epochs):
            for mb_obs, mb_act, mb_old_lp, mb_adv, mb_ret, mb_alive in buf.get_batches(
                cfg.num_minibatches
            ):
                _, new_lp, entropy, new_val = agent.get_action_and_value(mb_obs, mb_act)

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
                pg_loss = (torch.max(pg_loss1, pg_loss2) * mb_alive).sum() / mb_alive.sum().clamp(min=1)

                vf_loss = ((new_val - mb_ret) ** 2 * mb_alive).sum() / mb_alive.sum().clamp(min=1)
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

        # Expected episode return (mean over completed episodes this rollout)
        mean_ep_return = np.mean(rollout_ep_returns) if rollout_ep_returns else float("nan")

        # Mean critic predicted value and mean energy for alive agents
        with torch.no_grad():
            cur_obs = env_observe(env, cfg.device)
            _, _, _, cur_values = agent.get_action_and_value(cur_obs)
            alive_mask = cur_obs[:, ALIVE] > 0
            alive_count = int(alive_mask.sum().item())
            mean_value = cur_values[alive_mask].mean().item() if alive_count > 0 else 0.0
            # Get current energies from a peek step
            energy_np = np.array(env.observe()[:, :, ALIVE]).reshape(-1)  # alive flags
            # Actually read energy from rewards — just use prev_energy
            alive_energies = prev_energy[alive_mask]
            mean_energy = alive_energies.mean().item() if alive_count > 0 else 0.0

        if update % 10 == 0 or update <= 3:
            print(
                f"update {update:4d} | "
                f"t={elapsed:5.0f}s/{cfg.train_time:.0f}s | "
                f"step {global_step:>10,} | "
                f"sps {sps:>8,.0f} | "
                f"ep_return {mean_ep_return:+7.2f} | "
                f"V(s) {mean_value:+7.2f} | "
                f"energy {mean_energy:+7.2f} | "
                f"pg {total_pg_loss / num_batches:+.4f} | "
                f"vf {total_vf_loss / num_batches:.3f} | "
                f"clip {total_clip_frac / num_batches:.3f} | "
                f"alive {alive_count}/{n} | "
                f"lr {optimizer.param_groups[0]['lr']:.1e}"
            )

        # --- Render snapshot ---
        if cfg.render_every > 0 and update % cfg.render_every == 0:
            fname = f"train/ppo/render_update_{update:05d}.png"
            env.render(fname, env_index=0, pixels_per_unit=30.0)
            print(f"  -> saved {fname}")

    # --- Save model ---
    torch.save(agent.state_dict(), "train/ppo/model_final.pt")
    total_elapsed = time.time() - start_time
    print(f"\nTraining done: {update} updates, {global_step:,} steps in {total_elapsed:.1f}s")
    print(f"Model saved to train/ppo/model_final.pt")

    # --- Generate final video ---
    print("\nGenerating inference video...")
    inference_loop(
        model=agent,
        cfg=cfg,
        output_path="train/ppo/final.mp4",
        video_dt=0.04,
        num_steps=2048,
        pixels_per_unit=40.0,
    )

    return agent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="PPO training for organism environment")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--train-time", type=float, default=600.0, help="Wall-clock training seconds")
    parser.add_argument("--rollout-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-every", type=int, default=100)
    parser.add_argument("--reset-interval", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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
    )
    train(cfg)


if __name__ == "__main__":
    main()
