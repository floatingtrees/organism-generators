/// Single environment instance with continuous 2D physics and egocentric views.

use crate::modules::{ModuleGraph, ModuleType, PendingBuild, ROOT_ID, point_near_segment};
use crate::spatial_hash::SpatialHash;
use crate::types::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;

/// Observation channels per frame:
/// 0: food, 1: alive_agent, 2: dead_agent, 3: obstacle,
/// 4: own_segment, 5: own_gate, 6: own_thruster, 7: own_mouth,
/// 8: other_segment, 9: other_gate, 10: other_thruster, 11: other_mouth
pub const NUM_VIEW_CHANNELS: usize = 12;
/// Scalar self-state features: energy, vx, vy, view_size, rotation.
pub const NUM_SCALAR_FEATURES: usize = 5;
/// Number of previous frames stacked with the current frame.
pub const NUM_HISTORY_FRAMES: usize = 3;
/// Total channels in an observation: 12 channels × (1 current + 3 history).
pub const TOTAL_VIEW_CHANNELS: usize = NUM_VIEW_CHANNELS * (1 + NUM_HISTORY_FRAMES);
/// Continuous action dimensions: ax, ay, view_delta, signal, destroy_x, destroy_y,
/// build_x, build_y, build_rotation, build_length.
pub const NUM_CONTINUOUS_ACTIONS: usize = 10;
/// Discrete action (build_type): 0=none, 1=segment, 2=OR, 3=AND, 4=XOR, 5=thruster, 6=mouth.
pub const NUM_BUILD_TYPES: usize = 7;
/// Total action dimensions passed from Python (continuous + discrete as one-hot or index).
pub const NUM_ACTIONS: usize = NUM_CONTINUOUS_ACTIONS + 1; // +1 for build_type index

pub struct Environment {
    pub config: EnvironmentConfig,
    pub agents: Vec<Agent>,
    pub module_graphs: Vec<ModuleGraph>,
    pub foods: Vec<Food>,
    pub obstacles: Vec<Obstacle>,
    pub step_count: u64,
    rng: SmallRng,
    seed: u64,
    spatial: SpatialHash,
    /// Current rendered views for all agents.
    current_view: Vec<f32>,
    /// Ring buffer of previous frames.
    frame_history: VecDeque<Vec<f32>>,
}

impl Environment {
    pub fn new(num_agents: usize, config: EnvironmentConfig, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut placed: Vec<(Vec2, f32)> = Vec::new();

        // Place agents (non-overlapping)
        let r = config.object_radius;
        let ivs = config.initial_view_size;
        let agents: Vec<Agent> = (0..num_agents)
            .map(|id| {
                let pos = find_non_overlapping(&placed, r, config.width, config.height, &mut rng);
                placed.push((pos, r));
                Agent::new(id, pos, ivs)
            })
            .collect();

        // Place obstacles (non-overlapping with agents and each other)
        let obs_r = config.obstacle_radius;
        let mut obstacles = Vec::with_capacity(config.num_initial_obstacles);
        for _ in 0..config.num_initial_obstacles {
            let pos = find_non_overlapping(&placed, obs_r, config.width, config.height, &mut rng);
            placed.push((pos, obs_r));
            obstacles.push(Obstacle {
                pos,
                vel: Vec2::zero(),
                weight: config.obstacle_weight,
                radius: obs_r,
            });
        }

        let cell_size = (config.object_radius * 10.0).max(0.5);
        let spatial = SpatialHash::new(config.width, config.height, cell_size);

        let frame_size = num_agents * config.view_res * config.view_res * NUM_VIEW_CHANNELS;
        let frame_history: VecDeque<Vec<f32>> = (0..NUM_HISTORY_FRAMES)
            .map(|_| vec![0.0; frame_size])
            .collect();

        let module_graphs: Vec<ModuleGraph> = (0..num_agents).map(|_| ModuleGraph::new()).collect();

        let mut env = Self {
            config,
            agents,
            module_graphs,
            foods: Vec::new(),
            obstacles,
            step_count: 0,
            rng,
            seed,
            spatial,
            current_view: vec![0.0; frame_size],
            frame_history,
        };
        env.render_views();
        env
    }

    // ------------------------------------------------------------------
    // Reset
    // ------------------------------------------------------------------

    pub fn reset(&mut self) {
        self.seed = self.rng.gen();
        self.rng = SmallRng::seed_from_u64(self.seed);
        let r = self.config.object_radius;
        let ivs = self.config.initial_view_size;
        let w = self.config.width;
        let h = self.config.height;
        let mut placed: Vec<(Vec2, f32)> = Vec::new();

        for agent in &mut self.agents {
            agent.pos = find_non_overlapping(&placed, r, w, h, &mut self.rng);
            placed.push((agent.pos, r));
            agent.vel = Vec2::zero();
            agent.energy = 10.0;
            agent.dead_steps = 0;
            agent.alive = true;
            agent.view_size = ivs;
        }

        self.foods.clear();
        for mg in &mut self.module_graphs {
            *mg = ModuleGraph::new();
        }

        for obstacle in &mut self.obstacles {
            let obs_r = obstacle.radius;
            obstacle.pos = find_non_overlapping(&placed, obs_r, w, h, &mut self.rng);
            placed.push((obstacle.pos, obs_r));
            obstacle.vel = Vec2::zero();
        }

        let frame_size =
            self.agents.len() * self.config.view_res * self.config.view_res * NUM_VIEW_CHANNELS;
        self.frame_history.clear();
        for _ in 0..NUM_HISTORY_FRAMES {
            self.frame_history.push_back(vec![0.0; frame_size]);
        }

        self.step_count = 0;
        self.render_views();
    }

    // ------------------------------------------------------------------
    // Step
    // ------------------------------------------------------------------

    /// Step the environment. `actions` is a flat slice of length num_agents * NUM_ACTIONS.
    /// Per agent: [ax, ay, view_delta, signal, destroy_x, destroy_y,
    ///             build_x, build_y, build_rotation, build_length, build_type_index]
    pub fn step(&mut self, actions: &[f32]) {
        let dt = self.config.dt;
        let radius = self.config.object_radius;
        let max_vs = self.config.width.max(self.config.height) / 2.0;
        let na = self.agents.len();
        let build_delay_steps = (1.0 / dt).ceil() as u32; // 1 sim-second

        // 1. Parse actions, apply physics, handle builds/destroys
        for i in 0..na {
            if !self.agents[i].alive {
                continue;
            }
            let base = i * NUM_ACTIONS;
            let (ax_local, ay_local) = if base + 1 < actions.len() {
                (actions[base], actions[base + 1])
            } else {
                (0.0, 0.0)
            };
            let view_delta = if base + 2 < actions.len() { actions[base + 2] } else { 0.0 };
            let signal_prob = if base + 3 < actions.len() { actions[base + 3] } else { 0.0 };
            let destroy_x = if base + 4 < actions.len() { actions[base + 4] } else { 0.0 };
            let destroy_y = if base + 5 < actions.len() { actions[base + 5] } else { 0.0 };
            let build_x = if base + 6 < actions.len() { actions[base + 6] } else { 0.0 };
            let build_y = if base + 7 < actions.len() { actions[base + 7] } else { 0.0 };
            let build_rot = if base + 8 < actions.len() { actions[base + 8] } else { 0.0 };
            let build_len = if base + 9 < actions.len() { actions[base + 9] } else { 0.0 };
            let build_type_idx = if base + 10 < actions.len() { actions[base + 10] as usize } else { 0 };

            let rot = self.agents[i].rotation;

            // Rotate acceleration from local to world frame
            let ax_world = ax_local * rot.cos() - ay_local * rot.sin();
            let ay_world = ax_local * rot.sin() + ay_local * rot.cos();
            let accel = Vec2::new(ax_world, ay_world);

            self.agents[i].vel += accel * dt;
            self.agents[i].energy -= accel.magnitude() * dt;
            let vel = self.agents[i].vel;
            self.agents[i].pos += vel * dt;

            // Rotation update from angular velocity
            let ang_vel = self.agents[i].angular_velocity;
            self.agents[i].rotation += ang_vel * dt;

            // Thruster effects (force + torque)
            let (tfx, tfy, torque) = self.module_graphs[i].compute_thruster_effects(
                self.agents[i].pos, rot, accel.magnitude().max(1.0),
            );
            self.agents[i].vel.x += tfx * dt;
            self.agents[i].vel.y += tfy * dt;
            self.agents[i].angular_velocity += torque * dt;
            // Angular velocity decay: 10% per second → factor = 0.9^dt
            self.agents[i].angular_velocity *= (0.9f32).powf(dt);

            // View size update
            let min_vs_val = radius.max(self.config.min_view_size);
            let desired_vs = self.agents[i].view_size + view_delta * dt;
            let clamped_vs = desired_vs.clamp(min_vs_val, max_vs);
            let wasted_vs = (desired_vs - clamped_vs).abs();
            self.agents[i].energy -= wasted_vs;
            self.agents[i].view_size = clamped_vs;

            // Vision cost
            self.agents[i].energy -= self.config.vision_cost
                * self.agents[i].view_size * self.agents[i].view_size * dt;

            // Energy decay
            if self.config.energy_decay_rate < 1.0 {
                self.agents[i].energy *= self.config.energy_decay_rate.powf(dt);
            }

            // Signal propagation
            let signal = if signal_prob > 0.5 { 1.0 } else { 0.0 };
            self.module_graphs[i].propagate_signal(signal);

            // Destroy action (if destroy coords are nonzero)
            let destroy_pos = Vec2::new(destroy_x, destroy_y);
            if destroy_pos.magnitude() > 0.1 {
                if let Some((mod_id, _dist)) = self.module_graphs[i].find_nearest_module(destroy_pos) {
                    let removed = self.module_graphs[i].destroy_module(mod_id);
                    // Drop food at each removed module's world position
                    for pos in removed {
                        self.foods.push(Food { pos });
                    }
                }
            }

            // Build action — instant build, auto-attach to first free slot
            if self.module_graphs[i].build_cooldown > 0 {
                self.module_graphs[i].build_cooldown -= 1;
            }
            if let Some(build_type) = ModuleType::from_index(build_type_idx) {
                if self.module_graphs[i].build_cooldown == 0 {
                    let cost = self.module_graphs[i].build_cost(build_type);

                    if self.agents[i].energy >= cost {
                        // Auto-find attachment: try root first, then existing modules
                        let attach_id = self.module_graphs[i].find_any_free_slot();
                        if let Some(attach_id) = attach_id {
                            // Position: offset from attachment point along build_rotation
                            let attach_pos = if attach_id == ROOT_ID {
                                Vec2::zero()
                            } else if let Some(m) = self.module_graphs[i].get(attach_id) {
                                if m.module_type == ModuleType::Segment {
                                    m.segment_end_local()
                                } else {
                                    m.local_pos
                                }
                            } else {
                                Vec2::zero()
                            };
                            let build_local = Vec2::new(
                                attach_pos.x + build_len.clamp(0.1, 1.0) * build_rot.cos(),
                                attach_pos.y + build_len.clamp(0.1, 1.0) * build_rot.sin(),
                            );

                            self.agents[i].energy -= cost;
                            self.module_graphs[i].add_module(
                                build_type, build_local, build_rot,
                                build_len.clamp(0.1, 1.0), attach_id,
                            );
                            self.module_graphs[i].build_cooldown = (1.0 / dt).ceil() as u32;
                        }
                        // No penalty for no free slot — just skip
                    }
                }
            }

            // Update module world positions
            self.module_graphs[i].update_world_positions(self.agents[i].pos, self.agents[i].rotation);
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
                Self::bounce_agent_off_walls(agent, radius, w, h, loss, self.config.wall_velocity_damping);
            }
        }

        // 4. Wall bouncing — obstacles
        {
            let (w, h) = (self.config.width, self.config.height);
            for obstacle in &mut self.obstacles {
                Self::bounce_obstacle_off_walls(obstacle, w, h);
            }
        }

        // 5. Obstacle-obstacle collisions
        if self.config.interaction_rules.obstacle_obstacle_collision {
            self.handle_obstacle_obstacle_collisions();
        }

        // 6. Agent-obstacle collisions (elastic)
        if self.config.interaction_rules.obstacle_collision {
            self.handle_obstacle_collisions(radius);
        }

        // 7. Agent-agent collisions (elastic, optional)
        if self.config.interaction_rules.agent_collision {
            self.handle_agent_collisions(radius);
        }

        // 8. Food collection (agent body + mouths)
        if self.config.interaction_rules.food_collection {
            // Only mouths collect food — body does NOT collect directly.
            // This makes building mouths essential for survival.
            self.collect_food_mouths();
        }

        // 9. Agent death check — drop food at module locations on death
        for i in 0..na {
            if !self.agents[i].alive {
                continue;
            }
            if self.agents[i].energy <= 0.0 {
                self.agents[i].dead_steps += 1;
                if self.agents[i].dead_steps >= self.config.dead_steps_threshold {
                    self.agents[i].alive = false;
                    // Drop food at agent body position
                    self.foods.push(Food { pos: self.agents[i].pos });
                    // Drop food at each alive module position
                    for m in &self.module_graphs[i].modules {
                        if m.alive {
                            self.foods.push(Food { pos: m.world_pos });
                        }
                    }
                    // Clear the module graph
                    self.module_graphs[i] = ModuleGraph::new();
                }
            } else {
                self.agents[i].dead_steps = 0;
            }
        }

        // 9. Spawn food
        self.spawn_food();

        // 10. Render views (pushes current to history, renders new)
        self.render_views();

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
        velocity_damping: f32,
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
            agent.vel.x *= velocity_damping;
            agent.vel.y *= velocity_damping;
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
    // Obstacle-obstacle collisions (elastic) — spatial hash
    // ------------------------------------------------------------------

    fn handle_obstacle_obstacle_collisions(&mut self) {
        let n = self.obstacles.len();
        if n < 2 {
            return;
        }
        let obs_positions: Vec<Vec2> = self.obstacles.iter().map(|o| o.pos).collect();
        let obs_radii: Vec<f32> = self.obstacles.iter().map(|o| o.radius).collect();
        self.spatial.build_with_radii(&obs_positions, &obs_radii);

        for i in 0..n {
            let nearby = self
                .spatial
                .query_nearby(self.obstacles[i].pos, self.obstacles[i].radius);
            for j in nearby {
                if j <= i {
                    continue;
                }
                let collision_dist = self.obstacles[i].radius + self.obstacles[j].radius;
                let dist = self.obstacles[i].pos.distance_to(&self.obstacles[j].pos);
                if dist >= collision_dist || dist < 1e-8 {
                    continue;
                }
                let normal = Vec2::new(
                    (self.obstacles[i].pos.x - self.obstacles[j].pos.x) / dist,
                    (self.obstacles[i].pos.y - self.obstacles[j].pos.y) / dist,
                );
                let rel_vel = self.obstacles[i].vel - self.obstacles[j].vel;
                let vel_along_normal = rel_vel.dot(&normal);
                if vel_along_normal > 0.0 {
                    continue;
                }
                let mi = self.obstacles[i].weight;
                let mj = self.obstacles[j].weight;
                let total_mass = mi + mj;
                let impulse = -2.0 * vel_along_normal / total_mass;
                self.obstacles[i].vel.x += impulse * mj * normal.x;
                self.obstacles[i].vel.y += impulse * mj * normal.y;
                self.obstacles[j].vel.x -= impulse * mi * normal.x;
                self.obstacles[j].vel.y -= impulse * mi * normal.y;

                let overlap = collision_dist - dist;
                self.obstacles[i].pos.x += normal.x * overlap * mj / total_mass;
                self.obstacles[i].pos.y += normal.y * overlap * mj / total_mass;
                self.obstacles[j].pos.x -= normal.x * overlap * mi / total_mass;
                self.obstacles[j].pos.y -= normal.y * overlap * mi / total_mass;
            }
        }
    }

    // ------------------------------------------------------------------
    // Agent-obstacle collisions (elastic) — spatial hash
    // ------------------------------------------------------------------

    fn handle_obstacle_collisions(&mut self, agent_radius: f32) {
        if self.obstacles.is_empty() {
            return;
        }
        let obs_positions: Vec<Vec2> = self.obstacles.iter().map(|o| o.pos).collect();
        let obs_radii: Vec<f32> = self.obstacles.iter().map(|o| o.radius).collect();
        self.spatial.build_with_radii(&obs_positions, &obs_radii);

        for agent_idx in 0..self.agents.len() {
            if !self.agents[agent_idx].alive {
                continue;
            }
            // Only need agent_radius for query since obstacles are already in all their cells
            let nearby = self
                .spatial
                .query_nearby(self.agents[agent_idx].pos, agent_radius);
            for obs_idx in nearby {
                let collision_dist = agent_radius + self.obstacles[obs_idx].radius;
                let dist = self.agents[agent_idx]
                    .pos
                    .distance_to(&self.obstacles[obs_idx].pos);
                if dist >= collision_dist || dist < 1e-8 {
                    continue;
                }
                let normal = Vec2::new(
                    (self.agents[agent_idx].pos.x - self.obstacles[obs_idx].pos.x) / dist,
                    (self.agents[agent_idx].pos.y - self.obstacles[obs_idx].pos.y) / dist,
                );
                let rel_vel = self.agents[agent_idx].vel - self.obstacles[obs_idx].vel;
                let vel_along_normal = rel_vel.dot(&normal);
                if vel_along_normal > 0.0 {
                    continue;
                }
                let agent_mass = 1.0f32;
                let obstacle_mass = self.obstacles[obs_idx].weight;
                let total_mass = agent_mass + obstacle_mass;
                let impulse = -2.0 * vel_along_normal / total_mass;
                self.agents[agent_idx].vel.x += impulse * obstacle_mass * normal.x;
                self.agents[agent_idx].vel.y += impulse * obstacle_mass * normal.y;
                self.obstacles[obs_idx].vel.x -= impulse * agent_mass * normal.x;
                self.obstacles[obs_idx].vel.y -= impulse * agent_mass * normal.y;
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
    // Agent-agent collisions (elastic, optional) — spatial hash
    // ------------------------------------------------------------------

    fn handle_agent_collisions(&mut self, radius: f32) {
        let collision_dist = radius * 2.0;
        let n = self.agents.len();
        if n < 2 {
            return;
        }
        let agent_positions: Vec<Vec2> = self.agents.iter().map(|a| a.pos).collect();
        self.spatial.build(&agent_positions);

        for i in 0..n {
            if !self.agents[i].alive {
                continue;
            }
            let nearby = self.spatial.query_nearby(self.agents[i].pos, collision_dist);
            for j in nearby {
                if j <= i || !self.agents[j].alive {
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
    // Food collection — spatial hash
    // ------------------------------------------------------------------

    fn collect_food(&mut self, agent_radius: f32) {
        if self.foods.is_empty() {
            return;
        }
        let food_radius = self.config.object_radius;
        let collection_dist = agent_radius + food_radius;
        let food_positions: Vec<Vec2> = self.foods.iter().map(|f| f.pos).collect();
        self.spatial.build(&food_positions);
        let mut food_collected = vec![false; self.foods.len()];

        for agent in &mut self.agents {
            if !agent.alive {
                continue;
            }
            let nearby = self.spatial.query_nearby(agent.pos, collection_dist);
            for food_idx in nearby {
                if food_collected[food_idx] {
                    continue;
                }
                if agent.pos.distance_to(&food_positions[food_idx]) < collection_dist {
                    agent.energy += 1.0;
                    food_collected[food_idx] = true;
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

    /// Food collection via mouth modules.
    fn collect_food_mouths(&mut self) {
        if self.foods.is_empty() {
            return;
        }
        let mouth_radius = 0.5; // matches visual rendering size
        let food_radius = self.config.object_radius;
        let collection_dist = mouth_radius + food_radius;

        let food_positions: Vec<Vec2> = self.foods.iter().map(|f| f.pos).collect();
        self.spatial.build(&food_positions);

        let mut food_collected = vec![false; self.foods.len()];

        // Collect all mouth positions with their agent index
        let mut all_mouths: Vec<(usize, Vec2)> = Vec::new();
        for (ai, mg) in self.module_graphs.iter().enumerate() {
            if !self.agents[ai].alive {
                continue;
            }
            for pos in mg.alive_mouths() {
                all_mouths.push((ai, pos));
            }
        }

        for (ai, mouth_pos) in all_mouths {
            let nearby = self.spatial.query_nearby(mouth_pos, collection_dist);
            for food_idx in nearby {
                if food_collected[food_idx] {
                    continue;
                }
                if mouth_pos.distance_to(&food_positions[food_idx]) < collection_dist {
                    self.agents[ai].energy += 1.0;
                    food_collected[food_idx] = true;
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
    // Food spawning — dt-independent
    // ------------------------------------------------------------------

    fn spawn_food(&mut self) {
        if let Some(cap) = self.config.food_cap {
            if self.foods.len() >= cap {
                return;
            }
        }
        let expected = self.config.food_spawn_rate * self.config.dt;
        let num_to_spawn = expected as usize;
        let fractional = expected - num_to_spawn as f32;
        let mut total = num_to_spawn;
        if self.rng.gen::<f32>() < fractional {
            total += 1;
        }
        if let Some(cap) = self.config.food_cap {
            let room = cap.saturating_sub(self.foods.len());
            total = total.min(room);
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
    // View rendering
    // ------------------------------------------------------------------

    /// Push current view to history, render fresh views for the current state.
    fn render_views(&mut self) {
        // Rotate history
        let old_view = std::mem::take(&mut self.current_view);
        if self.frame_history.len() >= NUM_HISTORY_FRAMES {
            self.frame_history.pop_back();
        }
        self.frame_history.push_front(old_view);

        // Render new current views
        self.current_view = self.render_all_agent_views();
    }

    /// Render egocentric views for all agents. Returns flat buffer of size
    /// [num_agents * view_res * view_res * NUM_VIEW_CHANNELS].
    fn render_all_agent_views(&mut self) -> Vec<f32> {
        let res = self.config.view_res;
        let nc = NUM_VIEW_CHANNELS;
        let na = self.agents.len();
        let obj_r = self.config.object_radius;
        let view_per_agent = res * res * nc;
        let mut frame = vec![0.0f32; na * view_per_agent];

        // Extract agent info to avoid borrow conflicts with self.spatial
        let agent_info: Vec<(Vec2, f32, bool, usize)> = self
            .agents
            .iter()
            .map(|a| (a.pos, a.view_size, a.alive, a.id))
            .collect();

        // --- Channel 0: Food ---
        if !self.foods.is_empty() {
            let food_positions: Vec<Vec2> = self.foods.iter().map(|f| f.pos).collect();
            self.spatial.build(&food_positions);
            for (ai, &(pos, vs, alive, _)) in agent_info.iter().enumerate() {
                if !alive || vs <= 0.0 {
                    continue;
                }
                let buf = &mut frame[ai * view_per_agent..(ai + 1) * view_per_agent];
                for fi in self.spatial.query_nearby(pos, vs + obj_r) {
                    project_to_view(food_positions[fi], obj_r, pos, vs, res, buf, nc, 0);
                }
            }
        }

        // --- Channels 1 & 2: Alive agents / Dead agents (skip self) ---
        for (ai, &(pos, vs, alive, _)) in agent_info.iter().enumerate() {
            if !alive || vs <= 0.0 {
                continue;
            }
            let buf = &mut frame[ai * view_per_agent..(ai + 1) * view_per_agent];
            for (oi, &(opos, _, oalive, _)) in agent_info.iter().enumerate() {
                if oi == ai {
                    continue;
                }
                let dx = (opos.x - pos.x).abs();
                let dy = (opos.y - pos.y).abs();
                if dx > vs + obj_r || dy > vs + obj_r {
                    continue;
                }
                let ch = if oalive { 1 } else { 2 };
                project_to_view(opos, obj_r, pos, vs, res, buf, nc, ch);
            }
        }

        // --- Channel 3: Obstacles ---
        if !self.obstacles.is_empty() {
            let obs_info: Vec<(Vec2, f32)> =
                self.obstacles.iter().map(|o| (o.pos, o.radius)).collect();
            let obs_positions: Vec<Vec2> = obs_info.iter().map(|&(p, _)| p).collect();
            let obs_radii: Vec<f32> = obs_info.iter().map(|&(_, r)| r).collect();
            self.spatial.build_with_radii(&obs_positions, &obs_radii);
            for (ai, &(pos, vs, alive, _)) in agent_info.iter().enumerate() {
                if !alive || vs <= 0.0 {
                    continue;
                }
                let buf = &mut frame[ai * view_per_agent..(ai + 1) * view_per_agent];
                for oi in self.spatial.query_nearby(pos, vs) {
                    project_to_view(obs_info[oi].0, obs_info[oi].1, pos, vs, res, buf, nc, 3);
                }
            }
        }

        // --- Channels 4-11: Modules ---
        // 4: own_segment, 5: own_gate, 6: own_thruster, 7: own_mouth
        // 8: other_segment, 9: other_gate, 10: other_thruster, 11: other_mouth
        let module_r = 0.15; // rendering radius for non-segment modules
        for (ai, &(pos, vs, alive, _)) in agent_info.iter().enumerate() {
            if !alive || vs <= 0.0 {
                continue;
            }
            let buf = &mut frame[ai * view_per_agent..(ai + 1) * view_per_agent];

            for (oi, mg) in self.module_graphs.iter().enumerate() {
                let is_own = oi == ai;
                for m in &mg.modules {
                    if !m.alive {
                        continue;
                    }
                    // Check if module is in view
                    let dx = (m.world_pos.x - pos.x).abs();
                    let dy = (m.world_pos.y - pos.y).abs();
                    if dx > vs + 1.0 || dy > vs + 1.0 {
                        continue;
                    }

                    let ch = match (is_own, m.module_type) {
                        (true, ModuleType::Segment) => 4,
                        (true, ModuleType::Or | ModuleType::And | ModuleType::Xor) => 5,
                        (true, ModuleType::Thruster) => 6,
                        (true, ModuleType::Mouth) => 7,
                        (false, ModuleType::Segment) => 8,
                        (false, ModuleType::Or | ModuleType::And | ModuleType::Xor) => 9,
                        (false, ModuleType::Thruster) => 10,
                        (false, ModuleType::Mouth) => 11,
                    };

                    if m.module_type == ModuleType::Segment {
                        // Render segment as a line with width
                        project_segment_to_view(
                            m.world_pos, m.world_end, 0.2,
                            pos, vs, res, buf, nc, ch,
                        );
                    } else {
                        project_to_view(m.world_pos, module_r, pos, vs, res, buf, nc, ch);
                    }
                }
            }
        }

        frame
    }

    // ------------------------------------------------------------------
    // Observation: current view + 3 history frames
    // ------------------------------------------------------------------

    /// Returns flat observation for all agents: [num_agents * res * res * TOTAL_VIEW_CHANNELS].
    /// Channel layout: [current_ch0..ch3, hist0_ch0..ch3, hist1_ch0..ch3, hist2_ch0..ch3].
    pub fn get_views(&self) -> Vec<f32> {
        let res = self.config.view_res;
        let na = self.agents.len();
        let nc = NUM_VIEW_CHANNELS;
        let tc = TOTAL_VIEW_CHANNELS;
        let pixels = res * res;

        let mut obs = vec![0.0f32; na * pixels * tc];

        for ai in 0..na {
            for px in 0..pixels {
                let out_base = ai * pixels * tc + px * tc;

                // Current frame
                let cur_base = ai * pixels * nc + px * nc;
                obs[out_base..out_base + nc]
                    .copy_from_slice(&self.current_view[cur_base..cur_base + nc]);

                // History frames
                for (hi, hist) in self.frame_history.iter().enumerate() {
                    let h_offset = nc * (1 + hi);
                    if cur_base + nc <= hist.len() {
                        obs[out_base + h_offset..out_base + h_offset + nc]
                            .copy_from_slice(&hist[cur_base..cur_base + nc]);
                    }
                    // else: zeros (already initialized)
                }
            }
        }

        obs
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    pub fn get_alive_mask(&self) -> Vec<f32> {
        self.agents
            .iter()
            .map(|a| if a.alive { 1.0 } else { 0.0 })
            .collect()
    }

    pub fn get_rewards(&self) -> Vec<f32> {
        self.agents.iter().map(|a| a.energy).collect()
    }

    /// Returns flat [num_agents * NUM_SCALAR_FEATURES]: energy, vx, vy, view_size, rotation.
    /// Dead/padded agents get zeros.
    pub fn get_agent_states(&self) -> Vec<f32> {
        let mut states = Vec::with_capacity(self.agents.len() * NUM_SCALAR_FEATURES);
        for a in &self.agents {
            if a.alive {
                states.extend_from_slice(&[a.energy, a.vel.x, a.vel.y, a.view_size, a.rotation]);
            } else {
                states.extend_from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0]);
            }
        }
        states
    }

    pub fn num_agents(&self) -> usize {
        self.agents.len()
    }
}

// ---------------------------------------------------------------------------
// Non-overlapping spawn helper
// ---------------------------------------------------------------------------

fn find_non_overlapping(
    placed: &[(Vec2, f32)],
    radius: f32,
    width: f32,
    height: f32,
    rng: &mut SmallRng,
) -> Vec2 {
    for _ in 0..100 {
        let x = rng.gen_range(radius..width - radius);
        let y = rng.gen_range(radius..height - radius);
        let pos = Vec2::new(x, y);
        let overlaps = placed
            .iter()
            .any(|(p, r)| pos.distance_to(p) < radius + r);
        if !overlaps {
            return pos;
        }
    }
    // Fallback after 100 attempts
    let x = rng.gen_range(radius..width - radius);
    let y = rng.gen_range(radius..height - radius);
    Vec2::new(x, y)
}

// ---------------------------------------------------------------------------
// View projection helper (free function to avoid borrow issues)
// ---------------------------------------------------------------------------

/// Project an object at `obj_pos` with `obj_radius` onto a view grid.
/// The view is centered at `agent_pos` with half-width `view_size`.
/// Sets `channel` to 1.0 for covered pixels. Zero-pads outside world.
fn project_to_view(
    obj_pos: Vec2,
    obj_radius: f32,
    agent_pos: Vec2,
    view_size: f32,
    res: usize,
    buf: &mut [f32],
    num_channels: usize,
    channel: usize,
) {
    let view_width = view_size * 2.0;
    let scale = res as f32 / view_width;

    // Object center in pixel coords
    let px = (obj_pos.x - (agent_pos.x - view_size)) * scale;
    let py = (obj_pos.y - (agent_pos.y - view_size)) * scale;
    let pr = (obj_radius * scale).max(0.5);

    let px_min = ((px - pr).floor() as i32).max(0) as usize;
    let px_max = ((px + pr).ceil() as i32).min(res as i32 - 1).max(0) as usize;
    let py_min = ((py - pr).floor() as i32).max(0) as usize;
    let py_max = ((py + pr).ceil() as i32).min(res as i32 - 1).max(0) as usize;

    for row in py_min..=py_max {
        for col in px_min..=px_max {
            let dx = col as f32 - px;
            let dy = row as f32 - py;
            if dx * dx + dy * dy <= pr * pr {
                buf[row * res * num_channels + col * num_channels + channel] = 1.0;
            }
        }
    }
}

/// Project a line segment (with width) onto the view grid.
fn project_segment_to_view(
    seg_start: Vec2,
    seg_end: Vec2,
    width: f32,
    agent_pos: Vec2,
    view_size: f32,
    res: usize,
    buf: &mut [f32],
    num_channels: usize,
    channel: usize,
) {
    let view_width = view_size * 2.0;
    let scale = res as f32 / view_width;
    let half_w = (width * scale * 0.5).max(0.5);

    // Convert segment endpoints to pixel coords
    let sx = (seg_start.x - (agent_pos.x - view_size)) * scale;
    let sy = (seg_start.y - (agent_pos.y - view_size)) * scale;
    let ex = (seg_end.x - (agent_pos.x - view_size)) * scale;
    let ey = (seg_end.y - (agent_pos.y - view_size)) * scale;

    // Bounding box of the segment in pixels
    let min_x = ((sx.min(ex) - half_w).floor() as i32).max(0) as usize;
    let max_x = ((sx.max(ex) + half_w).ceil() as i32).min(res as i32 - 1).max(0) as usize;
    let min_y = ((sy.min(ey) - half_w).floor() as i32).max(0) as usize;
    let max_y = ((sy.max(ey) + half_w).ceil() as i32).min(res as i32 - 1).max(0) as usize;

    // For each pixel in bounding box, check distance to line segment
    let seg_len_sq = (ex - sx) * (ex - sx) + (ey - sy) * (ey - sy);
    for row in min_y..=max_y {
        for col in min_x..=max_x {
            let px = col as f32;
            let py = row as f32;
            // Distance from pixel to line segment
            let dist = if seg_len_sq < 1e-6 {
                ((px - sx) * (px - sx) + (py - sy) * (py - sy)).sqrt()
            } else {
                let t = ((px - sx) * (ex - sx) + (py - sy) * (ey - sy)) / seg_len_sq;
                let t = t.clamp(0.0, 1.0);
                let cx = sx + t * (ex - sx);
                let cy = sy + t * (ey - sy);
                ((px - cx) * (px - cx) + (py - cy) * (py - cy)).sqrt()
            };
            if dist <= half_w {
                buf[row * res * num_channels + col * num_channels + channel] = 1.0;
            }
        }
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
            wall_velocity_damping: 1.0,
            object_radius: 0.1,
            num_initial_obstacles: 0,
            obstacle_weight: 5.0,
            obstacle_radius: 0.1,
            dead_steps_threshold: 100,
            food_cap: None,
            vision_cost: 0.0, // no vision cost in tests by default
            view_res: 8,      // small for fast tests
            initial_view_size: 2.0,
            min_view_size: 0.0,
            energy_decay_rate: 1.0,
            interaction_rules: InteractionRules::default(),
        }
    }

    /// Helper: create action slice for n agents. Each tuple is (ax, ay, view_delta).
    /// Other action dims (signal, destroy, build) default to 0.
    fn make_actions(agent_actions: &[(f32, f32, f32)]) -> Vec<f32> {
        let mut actions = Vec::new();
        for &(ax, ay, vd) in agent_actions {
            let mut a = vec![0.0f32; NUM_ACTIONS];
            a[0] = ax;
            a[1] = ay;
            a[2] = vd;
            actions.extend_from_slice(&a);
        }
        actions
    }

    /// Helper: zero actions for n agents.
    fn zero_actions(n: usize) -> Vec<f32> {
        vec![0.0f32; n * NUM_ACTIONS]
    }

    #[test]
    fn create_environment() {
        let env = Environment::new(5, test_config(), 42);
        assert_eq!(env.agents.len(), 5);
        assert_eq!(env.step_count, 0);
        for agent in &env.agents {
            assert!(agent.alive);
            assert!((agent.energy - 10.0).abs() < 1e-6);
            assert!((agent.view_size - 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn zero_action_no_acceleration() {
        let mut env = Environment::new(2, test_config(), 42);
        let p0 = env.agents[0].pos;
        let p1 = env.agents[1].pos;

        env.step(&make_actions(&[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]));

        assert!((env.agents[0].pos.x - p0.x).abs() < 1e-6);
        assert!((env.agents[1].pos.x - p1.x).abs() < 1e-6);
    }

    #[test]
    fn acceleration_changes_position() {
        let mut env = Environment::new(1, test_config(), 42);
        let p0 = env.agents[0].pos;
        let dt = env.config.dt;

        env.step(&make_actions(&[(1.0, 0.0, 0.0)]));

        assert!((env.agents[0].vel.x - dt).abs() < 1e-6);
        assert!((env.agents[0].pos.x - (p0.x + dt * dt)).abs() < 1e-5);
    }

    #[test]
    fn view_delta_changes_view_size() {
        let mut env = Environment::new(1, test_config(), 42);
        let dt = env.config.dt;
        let vs0 = env.agents[0].view_size;

        env.step(&make_actions(&[(0.0, 0.0, 5.0)])); // increase view size

        assert!((env.agents[0].view_size - (vs0 + 5.0 * dt)).abs() < 1e-5);
    }

    #[test]
    fn view_size_clamped_to_bounds() {
        let cfg = test_config();
        let max_vs = cfg.width.max(cfg.height) / 2.0;
        let mut env = Environment::new(1, cfg.clone(), 42);

        // Push view_size way up
        for _ in 0..1000 {
            env.step(&make_actions(&[(0.0, 0.0, 100.0)]));
        }
        assert!(env.agents[0].view_size <= max_vs + 1e-6);

        // Push view_size to minimum: max(object_radius, min_view_size)
        for _ in 0..1000 {
            env.step(&make_actions(&[(0.0, 0.0, -100.0)]));
        }
        let min_vs = cfg.object_radius.max(cfg.min_view_size);
        assert!(env.agents[0].view_size >= min_vs - 1e-6);
    }

    #[test]
    fn vision_cost_drains_energy() {
        let mut cfg = test_config();
        cfg.vision_cost = 1.0;
        cfg.food_spawn_rate = 0.0;
        let mut env = Environment::new(1, cfg, 42);

        let e0 = env.agents[0].energy;
        env.step(&make_actions(&[(0.0, 0.0, 0.0)]));
        let e1 = env.agents[0].energy;

        // vision_cost * view_size^2 * dt = 1.0 * 2.0^2 * 0.1 = 0.4
        assert!((e0 - e1 - 0.4).abs() < 1e-5);
    }

    #[test]
    fn observation_shape() {
        let cfg = test_config(); // view_res=8
        let env = Environment::new(3, cfg, 42);
        let obs = env.get_views();
        // 3 agents * 8 * 8 * 16 channels
        assert_eq!(obs.len(), 3 * 8 * 8 * TOTAL_VIEW_CHANNELS);
    }

    #[test]
    fn dead_agent_view_is_zero() {
        let cfg = test_config();
        let mut env = Environment::new(2, cfg, 42);
        env.agents[0].alive = false;
        env.render_views();

        let obs = env.get_views();
        let res = env.config.view_res;
        let tc = TOTAL_VIEW_CHANNELS;
        let agent0_view = &obs[..res * res * tc];
        assert!(agent0_view.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn food_visible_in_view() {
        let cfg = test_config();
        let mut env = Environment::new(1, cfg, 42);

        // Place food right next to agent
        let agent_pos = env.agents[0].pos;
        env.foods.push(Food {
            pos: Vec2::new(agent_pos.x + 0.5, agent_pos.y),
        });
        env.render_views();

        let obs = env.get_views();
        // Channel 0 (food) should have some nonzero pixels
        let res = env.config.view_res;
        let tc = TOTAL_VIEW_CHANNELS;
        let has_food = (0..res * res).any(|px| obs[px * tc + 0] > 0.0);
        assert!(has_food, "food should be visible in agent's view");
    }

    #[test]
    fn frame_history_stacks() {
        let mut cfg = test_config();
        cfg.food_spawn_rate = 100.0; // lots of food to see
        cfg.food_cap = Some(50);
        let mut env = Environment::new(1, cfg, 42);

        // Step a few times to build history
        for _ in 0..5 {
            env.step(&make_actions(&[(0.0, 0.0, 0.0)]));
        }

        let obs = env.get_views();
        let res = env.config.view_res;
        let tc = TOTAL_VIEW_CHANNELS;
        let nc = NUM_VIEW_CHANNELS;

        // Check that history channels (offset 4..16) have some data
        let has_history = (0..res * res).any(|px| {
            (nc..tc).any(|ch| obs[px * tc + ch] > 0.0)
        });
        assert!(has_history, "history frames should have data after several steps");
    }

    #[test]
    fn food_cap_respected() {
        let mut cfg = test_config();
        cfg.food_spawn_rate = 100.0;
        cfg.food_cap = Some(10);
        let mut env = Environment::new(0, cfg, 42);

        for _ in 0..100 {
            env.step(&zero_actions(0));
        }
        assert!(env.foods.len() <= 10);
    }

    #[test]
    fn reset_clean_state() {
        let mut cfg = test_config();
        cfg.food_spawn_rate = 50.0;
        let mut env = Environment::new(3, cfg, 42);

        for _ in 0..10 {
            env.step(&make_actions(&[(1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (-1.0, -1.0, -1.0)]));
        }

        env.reset();

        assert_eq!(env.step_count, 0);
        assert!(env.foods.is_empty());
        for agent in &env.agents {
            assert!(agent.alive);
            assert!((agent.energy - 10.0).abs() < 1e-6);
            assert!((agent.view_size - 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn reset_produces_different_positions() {
        let mut env = Environment::new(3, test_config(), 42);
        let pos_before: Vec<Vec2> = env.agents.iter().map(|a| a.pos).collect();
        env.reset();
        let pos_after: Vec<Vec2> = env.agents.iter().map(|a| a.pos).collect();
        let any_different = pos_before
            .iter()
            .zip(pos_after.iter())
            .any(|(a, b)| (a.x - b.x).abs() > 1e-6 || (a.y - b.y).abs() > 1e-6);
        assert!(any_different);
    }

    #[test]
    fn build_segment_action() {
        let mut cfg = test_config();
        cfg.food_spawn_rate = 0.0;
        let mut env = Environment::new(1, cfg, 42);

        // Build a segment: build_type=1 (segment), build_x=0.5, build_len=0.8
        let mut actions = zero_actions(1);
        actions[6] = 0.5;  // build_x
        actions[7] = 0.0;  // build_y
        actions[8] = 0.0;  // build_rotation
        actions[9] = 0.8;  // build_length
        actions[10] = 1.0; // build_type = segment

        env.step(&actions);
        // Instant build — should be materialized immediately
        assert_eq!(env.module_graphs[0].alive_count(), 1);
        // Cooldown should be active
        assert!(env.module_graphs[0].build_cooldown > 0);
    }

    #[test]
    fn destroy_drops_food() {
        let mut cfg = test_config();
        cfg.food_spawn_rate = 0.0;
        let mut env = Environment::new(1, cfg, 42);

        // Build a segment (instant)
        let mut actions = zero_actions(1);
        actions[6] = 0.5;
        actions[9] = 0.5;
        actions[10] = 1.0;
        env.step(&actions);
        assert_eq!(env.module_graphs[0].alive_count(), 1);
        let food_before = env.foods.len();

        // Destroy it
        let mut destroy_actions = zero_actions(1);
        destroy_actions[4] = 0.5; // destroy_x
        destroy_actions[5] = 0.0; // destroy_y
        env.step(&destroy_actions);

        assert_eq!(env.module_graphs[0].alive_count(), 0);
        assert!(env.foods.len() > food_before);
    }

    #[test]
    fn local_frame_acceleration() {
        let mut cfg = test_config();
        cfg.food_spawn_rate = 0.0;
        let mut env = Environment::new(1, cfg, 42);

        // Set agent rotation to 90 degrees (facing up)
        env.agents[0].rotation = std::f32::consts::FRAC_PI_2;
        let p0 = env.agents[0].pos;

        // Accelerate "forward" in local frame (ax=1, ay=0)
        // In world frame, this should move in +y direction
        env.step(&make_actions(&[(1.0, 0.0, 0.0)]));

        let p1 = env.agents[0].pos;
        let dy = p1.y - p0.y;
        let dx = p1.x - p0.x;
        // Movement should be primarily in +y
        assert!(dy.abs() > dx.abs() * 5.0, "dy={dy}, dx={dx}");
    }

    #[test]
    fn death_drops_food_at_modules() {
        let mut cfg = test_config();
        cfg.food_spawn_rate = 0.0;
        cfg.dead_steps_threshold = 1;
        let mut env = Environment::new(1, cfg, 42);

        // Build a segment (instant)
        let mut actions = zero_actions(1);
        actions[6] = 0.5;
        actions[9] = 0.5;
        actions[10] = 1.0;
        env.step(&actions);
        assert_eq!(env.module_graphs[0].alive_count(), 1);

        // Kill the agent
        env.agents[0].energy = -100.0;
        env.step(&zero_actions(1));

        assert!(!env.agents[0].alive);
        assert!(env.foods.len() >= 2); // body + module
        assert_eq!(env.module_graphs[0].alive_count(), 0);
    }

    #[test]
    fn observation_has_48_channels() {
        let cfg = test_config();
        let env = Environment::new(2, cfg, 42);
        let obs = env.get_views();
        // 2 agents * 8 * 8 * 48 channels
        assert_eq!(obs.len(), 2 * 8 * 8 * TOTAL_VIEW_CHANNELS);
    }

    #[test]
    fn scalar_states_include_rotation() {
        let cfg = test_config();
        let env = Environment::new(1, cfg, 42);
        let states = env.get_agent_states();
        assert_eq!(states.len(), NUM_SCALAR_FEATURES); // 5 features
        // rotation should be 0 initially
        assert!((states[4]).abs() < 1e-6);
    }

    #[test]
    fn module_graph_cleared_on_reset() {
        let mut cfg = test_config();
        cfg.food_spawn_rate = 0.0;
        let mut env = Environment::new(1, cfg, 42);

        let mut actions = zero_actions(1);
        actions[6] = 0.5;
        actions[9] = 0.5;
        actions[10] = 1.0;
        env.step(&actions);
        assert_eq!(env.module_graphs[0].alive_count(), 1);

        env.reset();
        assert_eq!(env.module_graphs[0].alive_count(), 0);
    }

    #[test]
    fn build_cooldown_prevents_rapid_building() {
        let mut cfg = test_config();
        cfg.food_spawn_rate = 0.0;
        let mut env = Environment::new(1, cfg, 42);

        // First build succeeds
        let mut actions = zero_actions(1);
        actions[6] = 0.5;
        actions[9] = 0.5;
        actions[10] = 1.0;
        env.step(&actions);
        assert_eq!(env.module_graphs[0].alive_count(), 1);

        // Second build on next step should fail (cooldown)
        let mut actions2 = zero_actions(1);
        actions2[6] = -0.5;
        actions2[9] = 0.5;
        actions2[10] = 1.0;
        env.step(&actions2);
        assert_eq!(env.module_graphs[0].alive_count(), 1); // still 1, second build blocked
    }
}
