"""PPO training script for the organism evolutionary environment.

Usage:
    python train/ppo/train.py
    python train/ppo/train.py --num-envs 64 --num-agents 8 --total-steps 500000
"""

import argparse
import time
from dataclasses import dataclass

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
    food_spawn_rate: float = 5.0
    dt: float = 0.2
    energy_loss: float = 0.02
    num_obstacles: int = 3
    reset_interval: int = 256  # reset envs every N rollout steps to keep agents alive

    # PPO
    total_steps: int = 200_000
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
    render_every: int = 50  # render a snapshot every N updates (0 to disable)

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_agents * self.rollout_len

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_updates(self) -> int:
        return self.total_steps // (self.num_envs * self.num_agents * self.rollout_len)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class ActorCritic(nn.Module):
    """Shared-backbone actor-critic with Gaussian policy.

    Each agent is processed independently (parameter sharing).
    Input: per-agent features [x, y, vx, vy, alive] (5,)
    Actor output: mean of (ax, ay)
    Critic output: scalar state value
    """

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
        gamma = self.cfg.gamma
        lam = self.cfg.gae_lambda

        advantages = torch.zeros_like(self.rewards)
        last_gae = torch.zeros_like(next_value)

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
                next_mask = next_alive
            else:
                next_val = self.values[t + 1]
                next_mask = self.alive[t + 1]

            mask = self.alive[t]
            delta = self.rewards[t] + gamma * next_val * next_mask - self.values[t]
            last_gae = delta + gamma * lam * next_mask * last_gae
            advantages[t] = last_gae * mask

        self.advantages = advantages
        self.returns = advantages + self.values
        self.step = 0

    def get_batches(self, num_minibatches: int):
        n = self.cfg.num_envs * self.cfg.num_agents
        T = self.cfg.rollout_len
        total = T * n

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
            yield (
                obs[idx],
                actions[idx],
                log_probs[idx],
                advantages[idx],
                returns[idx],
                alive[idx],
            )


# ---------------------------------------------------------------------------
# Environment wrapper
# ---------------------------------------------------------------------------


def make_env(cfg: PPOConfig) -> organism_env.EvolutionEnv:
    config = {
        "num_organisms": cfg.num_agents,
        "height": cfg.env_height,
        "width": cfg.env_width,
        "food_spawn_rate": cfg.food_spawn_rate,
        "num_copies": cfg.num_envs,
        "dt": cfg.dt,
        "energy_loss": cfg.energy_loss,
        "num_obstacles": cfg.num_obstacles,
        "seed": cfg.seed,
    }
    return organism_env.EvolutionEnv.initialize(config)


def env_observe(env, device: str) -> torch.Tensor:
    """Get observations as a flat (num_envs * num_agents, features) tensor on device."""
    obs_np = env.observe()  # (num_envs, max_agents, 5)
    return torch.from_numpy(obs_np).reshape(-1, NUM_OBS_FEATURES).to(device)


def env_step(env, actions: torch.Tensor, cfg: PPOConfig):
    """Step the environment, return (energy, obs) as flat tensors."""
    act_np = actions.reshape(cfg.num_envs, cfg.num_agents, NUM_ACTIONS).cpu().numpy()
    energy_np = env.step(act_np)  # (num_envs, max_agents) — current energy per agent
    obs_np = env.observe()  # (num_envs, max_agents, 5)

    energy = torch.from_numpy(energy_np).reshape(-1).to(cfg.device)
    obs = torch.from_numpy(obs_np).reshape(-1, NUM_OBS_FEATURES).to(cfg.device)
    return energy, obs


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
    prev_energy = obs[:, ALIVE] * 10.0  # initial energy for alive agents
    global_step = 0
    env_steps_since_reset = 0
    start_time = time.time()

    num_updates = cfg.num_updates
    print(f"PPO | device={cfg.device} | envs={cfg.num_envs} | agents={cfg.num_agents}")
    print(f"     rollout={cfg.rollout_len} | batch={cfg.batch_size} | updates={num_updates}")
    print(f"     total_steps={cfg.total_steps:,}")
    print()

    for update in range(1, num_updates + 1):
        # Anneal LR
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            optimizer.param_groups[0]["lr"] = cfg.lr * frac

        # --- Collect rollout ---
        ep_rewards = []

        for step in range(cfg.rollout_len):
            global_step += n
            env_steps_since_reset += 1

            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(obs)

            alive = obs[:, ALIVE]

            # Step env — returns current energy
            energy, next_obs = env_step(env, action, cfg)

            # Reward = change in energy (positive = gained energy from food,
            # negative = spent energy on acceleration or lost to wall bounce)
            reward = (energy - prev_energy) * alive
            prev_energy = energy.clone()

            buf.insert(obs, action, log_prob, reward, value, alive)
            obs = next_obs

            ep_rewards.append(reward[alive > 0].mean().item() if (alive > 0).any() else 0.0)

            # Periodic reset so agents get fresh starts
            if env_steps_since_reset >= cfg.reset_interval:
                env.reset()
                obs = env_observe(env, cfg.device)
                prev_energy = obs[:, ALIVE] * 10.0
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

                # Normalize advantages (alive agents only)
                alive_mask = mb_alive > 0
                if alive_mask.sum() > 1:
                    adv_alive = mb_adv[alive_mask]
                    mb_adv = torch.where(
                        alive_mask,
                        (mb_adv - adv_alive.mean()) / (adv_alive.std() + 1e-8),
                        torch.zeros_like(mb_adv),
                    )

                # Policy loss (clipped)
                ratio = (new_lp - mb_old_lp).exp()
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps)
                pg_loss = (torch.max(pg_loss1, pg_loss2) * mb_alive).sum() / mb_alive.sum().clamp(min=1)

                # Value loss
                vf_loss = ((new_val - mb_ret) ** 2 * mb_alive).sum() / mb_alive.sum().clamp(min=1)

                # Entropy bonus
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
        avg_pg = total_pg_loss / num_batches
        avg_vf = total_vf_loss / num_batches
        avg_ent = total_ent_loss / num_batches
        avg_clip = total_clip_frac / num_batches
        mean_ep_reward = np.mean(ep_rewards)

        cur_obs = env_observe(env, cfg.device)
        alive_count = int((cur_obs[:, ALIVE] > 0).sum().item())

        print(
            f"update {update:4d}/{num_updates} | "
            f"step {global_step:>9,} | "
            f"sps {sps:>7,.0f} | "
            f"reward {mean_ep_reward:+.4f} | "
            f"pg {avg_pg:+.4f} | "
            f"vf {avg_vf:.4f} | "
            f"ent {avg_ent:.3f} | "
            f"clip {avg_clip:.2f} | "
            f"alive {alive_count}/{n} | "
            f"lr {optimizer.param_groups[0]['lr']:.1e}"
        )

        # --- Render snapshot ---
        if cfg.render_every > 0 and update % cfg.render_every == 0:
            fname = f"train/ppo/render_update_{update:05d}.png"
            env.render(fname, env_index=0, pixels_per_unit=30.0)
            print(f"  -> saved {fname}")

    # Save final model
    torch.save(agent.state_dict(), "train/ppo/model_final.pt")
    print(f"\nTraining complete. Model saved to train/ppo/model_final.pt")

    env.render("train/ppo/render_final.png", env_index=0, pixels_per_unit=30.0)
    print("Final render saved to train/ppo/render_final.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="PPO training for organism environment")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--rollout-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-every", type=int, default=50)
    parser.add_argument("--reset-interval", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = PPOConfig(
        num_envs=args.num_envs,
        num_agents=args.num_agents,
        total_steps=args.total_steps,
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
