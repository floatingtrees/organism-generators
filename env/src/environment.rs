/// Single environment instance with continuous 2D physics.

use crate::types::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

pub struct Environment {
    pub config: EnvironmentConfig,
    pub agents: Vec<Agent>,
    pub foods: Vec<Food>,
    pub obstacles: Vec<Obstacle>,
    pub step_count: u64,
    rng: SmallRng,
    seed: u64,
}

impl Environment {
    pub fn new(num_agents: usize, config: EnvironmentConfig, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let radius = config.object_radius;

        let agents = Self::place_agents(num_agents, &config, &mut rng);

        let mut obstacles = Vec::with_capacity(config.num_initial_obstacles);
        for _ in 0..config.num_initial_obstacles {
            let x = rng.gen_range(radius..config.width - radius);
            let y = rng.gen_range(radius..config.height - radius);
            obstacles.push(Obstacle {
                pos: Vec2::new(x, y),
                vel: Vec2::zero(),
                weight: config.obstacle_weight,
                radius,
            });
        }

        Self {
            config,
            agents,
            foods: Vec::new(),
            obstacles,
            step_count: 0,
            rng,
            seed,
        }
    }

    fn place_agents(n: usize, config: &EnvironmentConfig, rng: &mut SmallRng) -> Vec<Agent> {
        let r = config.object_radius;
        (0..n)
            .map(|id| {
                let x = rng.gen_range(r..config.width - r);
                let y = rng.gen_range(r..config.height - r);
                Agent::new(id, Vec2::new(x, y))
            })
            .collect()
    }

    // ------------------------------------------------------------------
    // Reset
    // ------------------------------------------------------------------

    pub fn reset(&mut self) {
        self.rng = SmallRng::seed_from_u64(self.seed);
        let r = self.config.object_radius;

        for agent in &mut self.agents {
            agent.pos = Vec2::new(
                self.rng.gen_range(r..self.config.width - r),
                self.rng.gen_range(r..self.config.height - r),
            );
            agent.vel = Vec2::zero();
            agent.energy = 10.0;
            agent.dead_steps = 0;
            agent.alive = true;
        }

        self.foods.clear();

        for obstacle in &mut self.obstacles {
            obstacle.pos = Vec2::new(
                self.rng.gen_range(r..self.config.width - r),
                self.rng.gen_range(r..self.config.height - r),
            );
            obstacle.vel = Vec2::zero();
        }

        self.step_count = 0;
    }

    // ------------------------------------------------------------------
    // Step
    // ------------------------------------------------------------------

    pub fn step(&mut self, actions: &[(f32, f32)]) {
        let dt = self.config.dt;
        let radius = self.config.object_radius;

        // 1. Apply accelerations to alive agents
        for (i, agent) in self.agents.iter_mut().enumerate() {
            if !agent.alive {
                continue;
            }
            let (ax, ay) = if i < actions.len() { actions[i] } else { (0.0, 0.0) };
            let accel = Vec2::new(ax, ay);

            // Velocity update
            agent.vel += accel * dt;
            // Energy cost of acceleration
            agent.energy -= accel.magnitude() * dt;
            // Position update
            agent.pos += agent.vel * dt;
        }

        // 2. Update obstacle positions
        for obstacle in &mut self.obstacles {
            obstacle.pos += obstacle.vel * dt;
        }

        // 3. Wall bouncing — agents
        if self.config.interaction_rules.wall_bounce {
            let (w, h) = (self.config.width, self.config.height);
            let loss = self.config.energy_loss_wall;
            for agent in &mut self.agents {
                if !agent.alive {
                    continue;
                }
                Self::bounce_agent_off_walls(agent, radius, w, h, loss);
            }
        }

        // 4. Wall bouncing — obstacles
        {
            let (w, h) = (self.config.width, self.config.height);
            for obstacle in &mut self.obstacles {
                Self::bounce_obstacle_off_walls(obstacle, w, h);
            }
        }

        // 5. Obstacle collisions (elastic)
        if self.config.interaction_rules.obstacle_collision {
            self.handle_obstacle_collisions(radius);
        }

        // 6. Agent-agent collisions (elastic, optional)
        if self.config.interaction_rules.agent_collision {
            self.handle_agent_collisions(radius);
        }

        // 7. Food collection
        if self.config.interaction_rules.food_collection {
            self.collect_food(radius);
        }

        // 8. Agent death check
        for agent in &mut self.agents {
            if !agent.alive {
                continue;
            }
            if agent.energy <= 0.0 {
                agent.dead_steps += 1;
                if agent.dead_steps >= self.config.dead_steps_threshold {
                    agent.alive = false;
                }
            } else {
                agent.dead_steps = 0;
            }
        }

        // 9. Spawn food
        self.spawn_food();

        self.step_count += 1;
    }

    // ------------------------------------------------------------------
    // Wall bouncing
    // ------------------------------------------------------------------

    fn bounce_agent_off_walls(
        agent: &mut Agent,
        radius: f32,
        w: f32,
        h: f32,
        energy_loss_frac: f32,
    ) {
        let mut bounced = false;

        if agent.pos.x < radius {
            agent.pos.x = radius;
            agent.vel.x = -agent.vel.x;
            bounced = true;
        } else if agent.pos.x > w - radius {
            agent.pos.x = w - radius;
            agent.vel.x = -agent.vel.x;
            bounced = true;
        }

        if agent.pos.y < radius {
            agent.pos.y = radius;
            agent.vel.y = -agent.vel.y;
            bounced = true;
        } else if agent.pos.y > h - radius {
            agent.pos.y = h - radius;
            agent.vel.y = -agent.vel.y;
            bounced = true;
        }

        if bounced {
            agent.energy -= agent.energy.abs() * energy_loss_frac;
        }
    }

    fn bounce_obstacle_off_walls(obstacle: &mut Obstacle, w: f32, h: f32) {
        let r = obstacle.radius;
        if obstacle.pos.x < r {
            obstacle.pos.x = r;
            obstacle.vel.x = -obstacle.vel.x;
        } else if obstacle.pos.x > w - r {
            obstacle.pos.x = w - r;
            obstacle.vel.x = -obstacle.vel.x;
        }
        if obstacle.pos.y < r {
            obstacle.pos.y = r;
            obstacle.vel.y = -obstacle.vel.y;
        } else if obstacle.pos.y > h - r {
            obstacle.pos.y = h - r;
            obstacle.vel.y = -obstacle.vel.y;
        }
    }

    // ------------------------------------------------------------------
    // Obstacle collisions (elastic)
    // ------------------------------------------------------------------

    fn handle_obstacle_collisions(&mut self, agent_radius: f32) {
        for agent_idx in 0..self.agents.len() {
            if !self.agents[agent_idx].alive {
                continue;
            }
            for obs_idx in 0..self.obstacles.len() {
                let collision_dist = agent_radius + self.obstacles[obs_idx].radius;
                let dist = self.agents[agent_idx].pos.distance_to(&self.obstacles[obs_idx].pos);

                if dist >= collision_dist || dist < 1e-8 {
                    continue;
                }

                let normal = Vec2::new(
                    (self.agents[agent_idx].pos.x - self.obstacles[obs_idx].pos.x) / dist,
                    (self.agents[agent_idx].pos.y - self.obstacles[obs_idx].pos.y) / dist,
                );

                let rel_vel = self.agents[agent_idx].vel - self.obstacles[obs_idx].vel;
                let vel_along_normal = rel_vel.dot(&normal);

                // Only resolve if approaching
                if vel_along_normal > 0.0 {
                    continue;
                }

                let agent_mass = 1.0f32;
                let obstacle_mass = self.obstacles[obs_idx].weight;
                let total_mass = agent_mass + obstacle_mass;

                // Elastic impulse
                let impulse = -2.0 * vel_along_normal / total_mass;

                self.agents[agent_idx].vel.x += impulse * obstacle_mass * normal.x;
                self.agents[agent_idx].vel.y += impulse * obstacle_mass * normal.y;
                self.obstacles[obs_idx].vel.x -= impulse * agent_mass * normal.x;
                self.obstacles[obs_idx].vel.y -= impulse * agent_mass * normal.y;

                // Separate overlapping bodies
                let overlap = collision_dist - dist;
                let agent_shift = overlap * obstacle_mass / total_mass;
                let obs_shift = overlap * agent_mass / total_mass;
                self.agents[agent_idx].pos.x += normal.x * agent_shift;
                self.agents[agent_idx].pos.y += normal.y * agent_shift;
                self.obstacles[obs_idx].pos.x -= normal.x * obs_shift;
                self.obstacles[obs_idx].pos.y -= normal.y * obs_shift;
            }
        }
    }

    // ------------------------------------------------------------------
    // Agent-agent collisions (elastic, optional)
    // ------------------------------------------------------------------

    fn handle_agent_collisions(&mut self, radius: f32) {
        let collision_dist = radius * 2.0;
        let n = self.agents.len();
        for i in 0..n {
            if !self.agents[i].alive {
                continue;
            }
            for j in (i + 1)..n {
                if !self.agents[j].alive {
                    continue;
                }
                let dist = self.agents[i].pos.distance_to(&self.agents[j].pos);
                if dist >= collision_dist || dist < 1e-8 {
                    continue;
                }
                let normal = Vec2::new(
                    (self.agents[i].pos.x - self.agents[j].pos.x) / dist,
                    (self.agents[i].pos.y - self.agents[j].pos.y) / dist,
                );
                let rel_vel = self.agents[i].vel - self.agents[j].vel;
                let vel_along_normal = rel_vel.dot(&normal);
                if vel_along_normal > 0.0 {
                    continue;
                }
                // Equal-mass elastic collision
                let impulse = -vel_along_normal;
                self.agents[i].vel.x += impulse * normal.x;
                self.agents[i].vel.y += impulse * normal.y;
                self.agents[j].vel.x -= impulse * normal.x;
                self.agents[j].vel.y -= impulse * normal.y;

                let overlap = collision_dist - dist;
                self.agents[i].pos.x += normal.x * overlap * 0.5;
                self.agents[i].pos.y += normal.y * overlap * 0.5;
                self.agents[j].pos.x -= normal.x * overlap * 0.5;
                self.agents[j].pos.y -= normal.y * overlap * 0.5;
            }
        }
    }

    // ------------------------------------------------------------------
    // Food collection
    // ------------------------------------------------------------------

    fn collect_food(&mut self, agent_radius: f32) {
        let food_radius = self.config.object_radius;
        let collection_dist = agent_radius + food_radius;
        let mut food_collected = vec![false; self.foods.len()];

        for agent in &mut self.agents {
            if !agent.alive {
                continue;
            }
            for (j, food) in self.foods.iter().enumerate() {
                if food_collected[j] {
                    continue;
                }
                if agent.pos.distance_to(&food.pos) < collection_dist {
                    agent.energy += 1.0;
                    food_collected[j] = true;
                }
            }
        }

        let mut idx = 0;
        self.foods.retain(|_| {
            let keep = !food_collected[idx];
            idx += 1;
            keep
        });
    }

    // ------------------------------------------------------------------
    // Food spawning
    // ------------------------------------------------------------------

    fn spawn_food(&mut self) {
        let rate = self.config.food_spawn_rate;
        let num_to_spawn = rate as usize;
        let fractional = rate - num_to_spawn as f32;

        let mut total = num_to_spawn;
        if self.rng.gen::<f32>() < fractional {
            total += 1;
        }

        let r = self.config.object_radius;
        for _ in 0..total {
            let x = self.rng.gen_range(r..self.config.width - r);
            let y = self.rng.gen_range(r..self.config.height - r);
            self.foods.push(Food {
                pos: Vec2::new(x, y),
            });
        }
    }

    // ------------------------------------------------------------------
    // Observation / reward helpers
    // ------------------------------------------------------------------

    /// Returns [x, y, vx, vy, alive] for every agent.
    pub fn get_agent_features(&self) -> Vec<[f32; 5]> {
        self.agents
            .iter()
            .map(|a| {
                [
                    a.pos.x,
                    a.pos.y,
                    a.vel.x,
                    a.vel.y,
                    if a.alive { 1.0 } else { 0.0 },
                ]
            })
            .collect()
    }

    /// Current energy of each agent (reward = energy).
    pub fn get_rewards(&self) -> Vec<f32> {
        self.agents.iter().map(|a| a.energy).collect()
    }

    pub fn num_agents(&self) -> usize {
        self.agents.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> EnvironmentConfig {
        EnvironmentConfig {
            width: 10.0,
            height: 10.0,
            dt: 0.1,
            food_spawn_rate: 0.0,
            energy_loss_wall: 0.1,
            object_radius: 0.1,
            num_initial_obstacles: 0,
            obstacle_weight: 5.0,
            dead_steps_threshold: 100,
            interaction_rules: InteractionRules::default(),
        }
    }

    #[test]
    fn create_environment() {
        let env = Environment::new(5, test_config(), 42);
        assert_eq!(env.agents.len(), 5);
        assert_eq!(env.step_count, 0);
        for agent in &env.agents {
            assert!(agent.alive);
            assert!((agent.energy - 10.0).abs() < 1e-6);
        }
    }

    #[test]
    fn zero_action_no_acceleration() {
        let mut env = Environment::new(2, test_config(), 42);
        let p0 = env.agents[0].pos;
        let p1 = env.agents[1].pos;

        // Zero acceleration, zero initial velocity → no movement
        env.step(&[(0.0, 0.0), (0.0, 0.0)]);

        assert!((env.agents[0].pos.x - p0.x).abs() < 1e-6);
        assert!((env.agents[0].pos.y - p0.y).abs() < 1e-6);
        assert!((env.agents[1].pos.x - p1.x).abs() < 1e-6);
        assert!((env.agents[1].pos.y - p1.y).abs() < 1e-6);
    }

    #[test]
    fn acceleration_changes_velocity_and_position() {
        let mut env = Environment::new(1, test_config(), 42);
        let p0 = env.agents[0].pos;
        let dt = env.config.dt;

        // Accelerate in +x
        env.step(&[(1.0, 0.0)]);

        // vel should be (1*dt, 0)
        assert!((env.agents[0].vel.x - dt).abs() < 1e-6);
        // pos should be p0.x + vel*dt = p0.x + dt*dt
        assert!((env.agents[0].pos.x - (p0.x + dt * dt)).abs() < 1e-5);
    }

    #[test]
    fn acceleration_costs_energy() {
        let mut env = Environment::new(1, test_config(), 42);
        let e0 = env.agents[0].energy;
        let dt = env.config.dt;

        env.step(&[(3.0, 4.0)]); // magnitude = 5
        let expected = e0 - 5.0 * dt;
        assert!((env.agents[0].energy - expected).abs() < 0.1);
    }

    #[test]
    fn wall_bounce_reflects_velocity() {
        let cfg = test_config();
        let mut env = Environment::new(1, cfg.clone(), 42);

        // Place agent near right wall, moving right fast
        env.agents[0].pos = Vec2::new(9.95, 5.0);
        env.agents[0].vel = Vec2::new(10.0, 0.0);
        env.agents[0].energy = 100.0;

        env.step(&[(0.0, 0.0)]);

        // Velocity x should have flipped
        assert!(env.agents[0].vel.x < 0.0);
        // Position should be clamped within bounds
        assert!(env.agents[0].pos.x <= cfg.width - cfg.object_radius + 0.001);
    }

    #[test]
    fn wall_bounce_loses_energy() {
        let cfg = test_config();
        let mut env = Environment::new(1, cfg, 42);

        env.agents[0].pos = Vec2::new(9.95, 5.0);
        env.agents[0].vel = Vec2::new(10.0, 0.0);
        env.agents[0].energy = 100.0;

        env.step(&[(0.0, 0.0)]);

        // Should have lost 10% of energy from wall bounce
        assert!(env.agents[0].energy < 100.0);
    }

    #[test]
    fn food_collection_adds_energy() {
        let cfg = test_config();
        let mut env = Environment::new(1, cfg, 42);

        env.agents[0].pos = Vec2::new(5.0, 5.0);
        env.agents[0].vel = Vec2::zero();
        let e0 = env.agents[0].energy;

        // Place food right on top of agent
        env.foods.push(Food {
            pos: Vec2::new(5.0, 5.0),
        });

        env.step(&[(0.0, 0.0)]);

        assert!(env.foods.is_empty());
        assert!((env.agents[0].energy - (e0 + 1.0)).abs() < 1e-6);
    }

    #[test]
    fn food_spawning() {
        let mut cfg = test_config();
        cfg.food_spawn_rate = 3.0;
        let mut env = Environment::new(1, cfg, 42);

        env.step(&[(0.0, 0.0)]);
        assert_eq!(env.foods.len(), 3);
    }

    #[test]
    fn agent_dies_after_threshold() {
        let mut cfg = test_config();
        cfg.dead_steps_threshold = 3;
        cfg.food_spawn_rate = 0.0;
        let mut env = Environment::new(1, cfg, 42);

        env.agents[0].energy = -1.0;
        for _ in 0..2 {
            env.step(&[(0.0, 0.0)]);
            assert!(env.agents[0].alive);
        }
        env.step(&[(0.0, 0.0)]);
        assert!(!env.agents[0].alive);
    }

    #[test]
    fn dead_agent_does_not_move() {
        let cfg = test_config();
        let mut env = Environment::new(1, cfg, 42);
        env.agents[0].alive = false;
        let pos = env.agents[0].pos;

        env.step(&[(10.0, 10.0)]);

        assert!((env.agents[0].pos.x - pos.x).abs() < 1e-6);
        assert!((env.agents[0].pos.y - pos.y).abs() < 1e-6);
    }

    #[test]
    fn reset_restores_initial_state() {
        let mut cfg = test_config();
        cfg.food_spawn_rate = 5.0;
        let mut env = Environment::new(3, cfg, 42);

        for _ in 0..10 {
            env.step(&[(1.0, 1.0), (0.0, 0.0), (-1.0, -1.0)]);
        }

        env.reset();

        assert_eq!(env.step_count, 0);
        assert!(env.foods.is_empty());
        for agent in &env.agents {
            assert!(agent.alive);
            assert!((agent.energy - 10.0).abs() < 1e-6);
            assert!((agent.vel.x).abs() < 1e-6);
        }
    }

    #[test]
    fn obstacle_elastic_collision() {
        let mut cfg = test_config();
        cfg.num_initial_obstacles = 0;
        cfg.food_spawn_rate = 0.0;
        let mut env = Environment::new(1, cfg, 42);

        env.agents[0].pos = Vec2::new(5.0, 5.0);
        env.agents[0].vel = Vec2::new(1.0, 0.0);

        env.obstacles.push(Obstacle {
            pos: Vec2::new(5.15, 5.0), // close enough to collide
            vel: Vec2::zero(),
            weight: 1.0,
            radius: 0.1,
        });

        env.step(&[(0.0, 0.0)]);

        // After elastic collision with equal mass, velocities should swap
        // Agent should have slowed down or reversed, obstacle should be moving
        assert!(env.obstacles[0].vel.x.abs() > 0.01);
    }
}
