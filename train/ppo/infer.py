"""Inference script — run trained policy and generate a video.

Usage:
    python train/ppo/infer.py
    python train/ppo/infer.py --weights train/ppo/model_final.pt --realtime-seconds 300
"""

import argparse
import math
import time

import imageio.v3 as iio
import numpy as np
import torch

import organism_env

# Must match train.py
NUM_ACTIONS = 3
VIEW_RES = 32
TOTAL_CHANNELS = 16
NUM_SCALAR_FEATURES = 4

# Import the model and config from train.py (same directory)
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from train import ActorCritic, PPOConfig


def env_observe(env, device: str):
    obs_np = env.observe()
    states_np = env.agent_states()
    mask_np = env.alive_mask()
    n = obs_np.shape[0] * obs_np.shape[1]
    obs = torch.from_numpy(obs_np).reshape(n, VIEW_RES, VIEW_RES, TOTAL_CHANNELS).to(device)
    scalars = torch.from_numpy(states_np).reshape(n, NUM_SCALAR_FEATURES).to(device)
    alive = torch.from_numpy(mask_np).reshape(n).to(device)
    return obs, scalars, alive


def main():
    parser = argparse.ArgumentParser(description="Run trained policy and generate video")
    parser.add_argument("--weights", type=str, default="train/ppo/model_final.pt")
    parser.add_argument("--output", type=str, default="train/ppo/videos/inference.mp4")
    parser.add_argument("--realtime-seconds", type=float, default=300.0,
                        help="Video duration in real-time seconds")
    parser.add_argument("--speed", type=float, default=3.0,
                        help="Playback speed multiplier (game seconds per real second)")
    parser.add_argument("--dt", type=float, default=0.04,
                        help="Simulation timestep for smooth video")
    parser.add_argument("--resolution", type=int, default=800,
                        help="Output image size (longest side in pixels)")
    parser.add_argument("--large-run", action="store_true")
    parser.add_argument("--agent-collision", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = PPOConfig()
    device = args.device

    # Compute steps needed
    sim_seconds = args.realtime_seconds * args.speed
    num_steps = int(sim_seconds / args.dt)
    fps = args.speed / args.dt

    print(f"Inference: {sim_seconds:.0f}s sim time, {args.realtime_seconds:.0f}s real time at {args.speed}x")
    print(f"           {num_steps} steps at dt={args.dt}, {fps:.0f} fps")

    # Load model
    model = ActorCritic(large=args.large_run).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded weights from {args.weights}")

    # Compute pixels_per_unit for target resolution
    max_dim = max(cfg.env_width, cfg.env_height)
    pixels_per_unit = args.resolution / max_dim
    raw_px = int(max_dim * pixels_per_unit)
    if raw_px % 16 != 0:
        pixels_per_unit = (((raw_px + 15) // 16) * 16) / max_dim

    # Create env
    env = organism_env.EvolutionEnv.initialize({
        "num_organisms": cfg.num_agents,
        "height": cfg.env_height,
        "width": cfg.env_width,
        "food_spawn_rate": cfg.food_spawn_rate,
        "num_copies": 1,
        "dt": args.dt,
        "energy_loss": cfg.energy_loss,
        "object_radius": cfg.object_radius,
        "num_obstacles": cfg.num_obstacles,
        "obstacle_radius": cfg.obstacle_radius,
        "obstacle_weight": cfg.obstacle_weight,
        "food_cap": cfg.food_cap,
        "vision_cost": cfg.vision_cost,
        "initial_view_size": cfg.initial_view_size,
        "min_view_size": cfg.min_view_size,
        "rules": {"agent_collision": args.agent_collision},
    })

    obs, scalars, alive = env_observe(env, device)
    frames = []
    start_time = time.time()

    with torch.no_grad():
        for step in range(num_steps):
            frame = env.render_array(env_index=0, pixels_per_unit=pixels_per_unit)
            frames.append(frame)

            action, _, _, _ = model.get_action_and_value(obs, scalars)
            act_np = action.reshape(1, cfg.num_agents, NUM_ACTIONS).cpu().numpy()
            env.step(act_np)
            obs, scalars, alive = env_observe(env, device)

            # Stop if all agents are dead
            if env.all_dead():
                sim_time = (step + 1) * args.dt
                print(f"All agents dead at step {step + 1} ({sim_time:.1f}s sim time)")
                break

            # Progress every 10%
            if (step + 1) % (num_steps // 10) == 0:
                pct = (step + 1) / num_steps * 100
                elapsed = time.time() - start_time
                alive_count = int((alive > 0).sum().item())
                print(f"  {pct:3.0f}% | step {step+1}/{num_steps} | alive {alive_count}/{cfg.num_agents} | {elapsed:.0f}s elapsed")

    # Write video
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    frames_arr = np.stack(frames)
    iio.imwrite(args.output, frames_arr, fps=fps, codec="libx264")
    actual_sim = len(frames) * args.dt
    actual_real = len(frames) / fps
    print(f"\nVideo saved: {args.output}")
    print(f"  {len(frames)} frames, {actual_sim:.1f}s sim, {actual_real:.1f}s video, {args.speed}x speed")


if __name__ == "__main__":
    main()
