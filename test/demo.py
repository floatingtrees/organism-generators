"""Demo: run the evolutionary environment for a few steps and render snapshots."""

import numpy as np
import organism_env

config = {
    "num_organisms": [4, 6, 3],
    "height": 15.0,
    "width": 15.0,
    "food_spawn_rate": 3.0,
    "num_copies": 3,
    "dt": 0.2,
    "energy_loss": 0.05,
    "num_obstacles": 3,
    "obstacle_weight": 3.0,
    "seed": 12345,
}

env = organism_env.EvolutionEnv.initialize(config)
print(f"Created {env.num_envs} environments, max_agents={env.max_agents}")

# Render initial state
env.render("demo_step_000.png", env_index=0, pixels_per_unit=30.0)
print("Saved demo_step_000.png")

obs = env.observe()
print(f"Initial observation shape: {obs.shape}")
print(f"Features per agent: [x, y, vx, vy, alive]")
print(f"Env 0, Agent 0: {obs[0, 0]}")

# Run simulation with random actions
num_steps = 100
for step in range(1, num_steps + 1):
    actions = np.random.randn(env.num_envs, env.max_agents, 2).astype(np.float32) * 2.0
    rewards = env.step(actions)

    if step % 25 == 0:
        obs = env.observe()
        alive_counts = obs[..., 4].sum(axis=1).astype(int)
        mean_energy = rewards[0, :alive_counts[0]].mean() if alive_counts[0] > 0 else 0

        fname = f"demo_step_{step:03d}.png"
        env.render(fname, env_index=0, pixels_per_unit=30.0)
        print(
            f"Step {step:3d} | "
            f"alive={alive_counts.tolist()} | "
            f"env0 mean_energy={mean_energy:.2f} | "
            f"saved {fname}"
        )

# Final render
env.render("demo_final.png", env_index=0, pixels_per_unit=30.0)
print(f"\nDone. Saved demo_final.png")
