# organism-generators

A high-performance evolutionary environment written in Rust with Python bindings. Designed for reinforcement learning research with batched, continuous-physics 2D worlds.

## Quick start

```bash
# Build (requires Rust toolchain + Python 3.9+)
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy torch --index-url https://download.pytorch.org/whl/cpu
cd env && maturin develop --release && cd ..

# Run tests
pip install pytest
pytest test/ -v

# Run demo
python test/demo.py
```

## API

### `EvolutionEnv.initialize(config: dict) -> EvolutionEnv`

Create a batched evolutionary environment. Config keys:

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `num_organisms` | `int \| list[int]` | yes | — | Agents per environment. Scalar applies to all; list must match `num_copies`. |
| `height` | `float` | yes | — | World height in units. |
| `width` | `float` | yes | — | World width in units. |
| `food_spawn_rate` | `float` | yes | — | Expected food items spawned per step (integer part guaranteed, fractional is probabilistic). |
| `num_copies` | `int` | yes | — | Number of parallel environments. |
| `dt` | `float` | no | `0.5` | Simulation timestep. |
| `energy_loss` | `float` | no | `0.1` | Fraction of energy lost on wall bounce (0.1 = 10%). |
| `object_radius` | `float` | no | `0.1` | Collision radius for all objects. |
| `num_obstacles` | `int` | no | `0` | Initial obstacles per environment. |
| `obstacle_weight` | `float` | no | `5.0` | Mass of obstacles (affects elastic collision response). |
| `seed` | `int` | no | `42` | Base RNG seed. Each environment gets `seed + env_index`. |
| `rules` | `dict` | no | all enabled | Interaction rule toggles (see below). |

### `env.step(actions: np.ndarray) -> np.ndarray`

Advance one simulation step.

- **Input**: `actions` — shape `(num_envs, max_agents, 2)`, float32. Each `(ax, ay)` is an acceleration vector. Padded agent slots should be zero.
- **Returns**: rewards — shape `(num_envs, max_agents)`, float32. Current energy of each agent (reward = energy).

Physics per step:
1. Apply acceleration to velocity: `vel += accel * dt`
2. Deduct energy: `energy -= |accel| * dt`
3. Update position: `pos += vel * dt`
4. Wall bounce (reflect velocity, lose `energy_loss` fraction of energy)
5. Elastic collisions with obstacles
6. Food collection (+1 energy per food within collision radius)
7. Death check (energy ≤ 0 for 10 seconds → agent dies)
8. Spawn food

### `env.observe() -> np.ndarray`

- **Returns**: shape `(num_envs, max_agents, 5)`, float32
- Features: `[x, y, vx, vy, alive]`
- Padded agent slots are all zeros (`alive=0`)

### `env.render(filepath, env_index=0, pixels_per_unit=20.0)`

Save a PNG image of a single environment to `filepath`.

- Food: small green filled circles
- Agents: colored dots with white center marker, dashed view-size circle
- Obstacles: large gray filled circles

### Module rendering (in video/PNG)

| Module | Shape | Color | Description |
|--------|-------|-------|-------------|
| Segment | Thick line/rectangle | Agent color (dimmed) | Connects modules, length 1.0, width 0.2 |
| Mouth | Large circle + white center | Green (#32FF64) | Collects food within 0.5 radius |
| OR gate | Circle + light inner | Yellow (#C8C832) | Outputs 1 if either input is 1 |
| AND gate | Square + light inner | Cyan (#32C8C8) | Outputs 1 if both inputs are 1 |
| XOR gate | Diamond + light inner | Magenta (#C832C8) | Outputs 1 if exactly one input is 1 |
| Thruster | Triangle pointing in thrust dir | Orange (#FF6432) | Applies force when receiving signal |

In the 32×32 agent observation, modules appear as dots on dedicated channels (own vs other).
- Agents: distinct hue per agent with white center marker
- Obstacles: gray circles
- Dead agents: dark red

### `env.reset()`

Reset all environments to initial state (same seed → deterministic).

### Properties

- `env.num_envs` — number of parallel environments
- `env.max_agents` — max agents across all environments (pad dimension)

## Obstacles

Obstacles are created at initialization via `num_obstacles` and `obstacle_weight` in the config. They are placed at random positions within the environment bounds. Key behaviors:

- **Elastic collisions**: When an agent collides with an obstacle, momentum is exchanged based on mass (agents have mass 1.0, obstacles use `obstacle_weight`). Heavier obstacles are harder to push.
- **Persistent**: Obstacles bounce off walls and retain velocity after being pushed. They are not consumed or destroyed.
- **Reset**: On `reset()`, obstacles return to new random positions with zero velocity.

```python
config = {
    "num_organisms": 5,
    "height": 20.0, "width": 20.0,
    "food_spawn_rate": 3.0,
    "num_copies": 4,
    "num_obstacles": 10,      # 10 obstacles per environment
    "obstacle_weight": 3.0,   # lighter = easier to push
}
```

## Interaction rules

Toggle physics interactions on/off via the `rules` dict:

```python
config = {
    ...,
    "rules": {
        "wall_bounce": True,        # agents/obstacles bounce off walls
        "food_collection": True,    # agents collect food on contact
        "obstacle_collision": True, # elastic collision with obstacles
        "agent_collision": False,   # elastic collision between agents (off by default)
    }
}
```

## Extending the library

### Adding a new feature to observations

1. **`env/src/environment.rs`** — update `get_agent_features()` to include the new value in the returned array.
2. **`env/src/batched_env.rs`** — update `NUM_FEATURES` constant.
3. **`env/src/lib.rs`** — the observe shape auto-adjusts from `NUM_FEATURES`.
4. Update Python tests.

### Adding a new interaction rule

1. **`env/src/types.rs`** — add a `bool` field to `InteractionRules` (default it in `Default` impl).
2. **`env/src/environment.rs`** — add the interaction logic in `step()`, gated by `self.config.interaction_rules.your_rule`.
3. **`env/src/lib.rs`** — extract the new rule key from the `rules` dict in `initialize()`.

Example — adding a gravity rule:

```rust
// types.rs
pub struct InteractionRules {
    pub wall_bounce: bool,
    pub food_collection: bool,
    pub obstacle_collision: bool,
    pub agent_collision: bool,
    pub gravity: bool,           // NEW
}

// environment.rs, inside step():
if self.config.interaction_rules.gravity {
    for agent in &mut self.agents {
        if !agent.alive { continue; }
        agent.vel.y += 9.81 * dt;  // downward gravity
    }
}
```

### Adding a new agent property

1. **`env/src/types.rs`** — add the field to `Agent`, initialize it in `Agent::new()`.
2. **`env/src/environment.rs`** — update `reset()` to reinitialize it, update `step()` to modify it, optionally expose via `get_agent_features()`.

### Adding a new config parameter

1. **`env/src/types.rs`** — add the field to `EnvironmentConfig`, set a default in `Default` impl.
2. **`env/src/lib.rs`** — extract it from the Python config dict in `initialize()` using `extract_or()` for optional or `extract_required()` for required parameters.

### Project structure

```
env/
  Cargo.toml          # Rust dependencies
  pyproject.toml       # maturin build config
  src/
    lib.rs             # PyO3 bindings (Python ↔ Rust bridge)
    types.rs           # Core types: Vec2, Agent, Food, Obstacle, configs
    environment.rs     # Single environment: physics, step, observe
    batched_env.rs     # Batched wrapper over multiple environments
    rendering.rs       # RGB rendering + PNG export
test/
    conftest.py        # Shared fixtures and constants
    test_initialize.py # Initialization and config tests
    test_step.py       # Physics, energy, wall bounce, death tests
    test_observe.py    # Observation shape, features, padding tests
    test_render.py     # PNG rendering tests
    test_batched.py    # Multi-environment tests
    demo.py            # Runnable demo script
```

### Build & test

```bash
# Rust unit tests (30 tests)
cd env && cargo test

# Python integration tests (53 tests)
cd .. && pytest test/ -v

# Rebuild after Rust changes
cd env && maturin develop --release
```
