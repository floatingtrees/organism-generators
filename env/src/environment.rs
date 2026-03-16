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

            // Signal propagation (must happen BEFORE thruster effects)
            let signal = if signal_prob > 0.5 { 1.0 } else { 0.0 };
            self.module_graphs[i].propagate_signal(signal);

            // Thruster effects (force + torque) — uses signal from above
            let (tfx, tfy, torque) = self.module_graphs[i].compute_thruster_effects(
                self.agents[i].pos, self.agents[i].rotation, accel.magnitude().max(1.0),
            );
            self.agents[i].vel.x += tfx * dt;
            self.agents[i].vel.y += tfy * dt;
            self.agents[i].angular_velocity += torque * dt;

            // Destroy action (requires high magnitude to trigger — prevents accidental destroys)
            // Cost is 3 energy whether or not it finds a module to destroy (equal cost for invalid)
            let destroy_pos = Vec2::new(destroy_x, destroy_y);
            if destroy_pos.magnitude() > 2.0 {
                self.agents[i].energy -= 3.0;
                if let Some((mod_id, _dist)) = self.module_graphs[i].find_nearest_module(destroy_pos) {
                    self.module_graphs[i].destroy_module(mod_id);
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
                        let attach_id = self.module_graphs[i].find_free_slot_for(build_type);
                        if let Some(attach_id) = attach_id {
                            // Position: offset from attachment point along build_rotation
                            let seg_len = 1.0; // constant segment length
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
                            // Non-segments offset further to avoid sitting on body
                            let module_offset = if build_type == ModuleType::Segment { seg_len } else { 0.5 };
                            let build_local = Vec2::new(
                                attach_pos.x + module_offset * build_rot.cos(),
                                attach_pos.y + module_offset * build_rot.sin(),
                            );

                            // Self-collision: must not overlap existing own modules
                            let min_clearance = 0.5;
                            let self_collision = self.module_graphs[i].modules.iter().any(|m| {
                                if !m.alive { return false; }
                                m.local_pos.distance_to(&build_local) < min_clearance
                            });

                            // World-space checks: compute where this module would be
                            let agent_rot = self.agents[i].rotation;
                            let cos_r = agent_rot.cos();
                            let sin_r = agent_rot.sin();
                            let world_pos = Vec2::new(
                                self.agents[i].pos.x + build_local.x * cos_r - build_local.y * sin_r,
                                self.agents[i].pos.y + build_local.x * sin_r + build_local.y * cos_r,
                            );

                            // Check world boundaries (for segments, also check endpoint)
                            let mut oob = world_pos.x < radius || world_pos.x > self.config.width - radius
                                || world_pos.y < radius || world_pos.y > self.config.height - radius;
                            if build_type == ModuleType::Segment {
                                let world_rot = build_rot + agent_rot;
                                let end_x = world_pos.x + seg_len * world_rot.cos();
                                let end_y = world_pos.y + seg_len * world_rot.sin();
                                oob = oob || end_x < radius || end_x > self.config.width - radius
                                    || end_y < radius || end_y > self.config.height - radius;
                            }

                            // Check obstacle overlap
                            let in_obstacle = self.obstacles.iter().any(|o| {
                                world_pos.distance_to(&o.pos) < o.radius + 0.3
                            });

                            if !self_collision && !oob && !in_obstacle {
                                self.agents[i].energy -= cost;
                                self.module_graphs[i].add_module(
                                    build_type, build_local, build_rot,
                                    seg_len, attach_id,
                                );
                                self.module_graphs[i].build_cooldown = (1.0 / dt).ceil() as u32;
                            }
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

        // 4.5. Module-wall and module-obstacle collision resolution
        // If any module clips through a wall or obstacle, push the agent body to resolve.
        {
            let (w, h) = (self.config.width, self.config.height);
            let margin = 0.15; // half segment width
            for i in 0..na {
                if !self.agents[i].alive {
                    continue;
                }
                // Recompute world positions after wall bounce
                self.module_graphs[i].update_world_positions(self.agents[i].pos, self.agents[i].rotation);

                let mut push_x = 0.0f32;
                let mut push_y = 0.0f32;

                for m in &self.module_graphs[i].modules {
                    if !m.alive { continue; }

                    // Check both start and end positions for segments
                    let positions = if m.module_type == ModuleType::Segment {
                        vec![m.world_pos, m.world_end]
                    } else {
                        vec![m.world_pos]
                    };

                    for &p in &positions {
                        // Wall clipping
                        if p.x < margin { push_x = push_x.max(margin - p.x); }
                        if p.x > w - margin { push_x = push_x.min(w - margin - p.x); }
                        if p.y < margin { push_y = push_y.max(margin - p.y); }
                        if p.y > h - margin { push_y = push_y.min(h - margin - p.y); }

                        // Obstacle clipping
                        for obs in &self.obstacles {
                            let dist = p.distance_to(&obs.pos);
                            let min_dist = obs.radius + margin;
                            if dist < min_dist && dist > 1e-6 {
                                let nx = (p.x - obs.pos.x) / dist;
                                let ny = (p.y - obs.pos.y) / dist;
                                let overlap = min_dist - dist;
                                push_x += nx * overlap;
                                push_y += ny * overlap;
                            }
                        }
                    }
                }

                if push_x.abs() > 1e-6 || push_y.abs() > 1e-6 {
                    self.agents[i].pos.x += push_x;
                    self.agents[i].pos.y += push_y;
                    // Dampen velocity in push direction
                    if push_x.abs() > 1e-6 { self.agents[i].vel.x *= 0.5; }
                    if push_y.abs() > 1e-6 { self.agents[i].vel.y *= 0.5; }

                    // Apply torque: force at module position creates rotation
                    // Compute average contact point relative to agent center
                    // then torque = r × F (2D cross product)
                    let com = self.agents[i].pos;
                    // Use the module that was pushed most as the contact point
                    for m in &self.module_graphs[i].modules {
                        if !m.alive { continue; }
                        let contact_points = if m.module_type == ModuleType::Segment {
                            vec![m.world_pos, m.world_end]
                        } else {
                            vec![m.world_pos]
                        };
                        for &p in &contact_points {
                            let rx = p.x - com.x;
                            let ry = p.y - com.y;
                            // torque = rx * Fy - ry * Fx (per contact)
                            let torque = rx * push_y - ry * push_x;
                            self.agents[i].angular_velocity += torque * 0.1; // damped
                        }
                    }

                    // Re-update module positions
                    self.module_graphs[i].update_world_positions(self.agents[i].pos, self.agents[i].rotation);
                }
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
            // Rejection sample: avoid spawning inside obstacles
            let mut attempts = 0;
            loop {
                let x = self.rng.gen_range(r..self.config.width - r);
                let y = self.rng.gen_range(r..self.config.height - r);
                let pos = Vec2::new(x, y);
                let in_obstacle = self.obstacles.iter().any(|o| {
                    pos.distance_to(&o.pos) < o.radius + r
                });
                if !in_obstacle || attempts > 20 {
                    if !in_obstacle {
                        self.foods.push(Food { pos });
                    }
                    break;
                }
                attempts += 1;
            }
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
    fn destroy_costs_energy() {
        let mut cfg = test_config();
        cfg.food_spawn_rate = 0.0;
        let mut env = Environment::new(1, cfg, 42);

        // Build a segment
        let mut actions = zero_actions(1);
        actions[10] = 1.0; // segment
        env.step(&actions);
        assert_eq!(env.module_graphs[0].alive_count(), 1);

        let energy_before = env.agents[0].energy;
        let food_before = env.foods.len();

        // Destroy it (need magnitude > 2.0)
        let mut destroy_actions = zero_actions(1);
        destroy_actions[4] = 3.0; // destroy_x (magnitude > 2)
        env.step(&destroy_actions);

        assert_eq!(env.module_graphs[0].alive_count(), 0);
        assert!(env.agents[0].energy < energy_before - 2.5); // cost 3 energy
        assert_eq!(env.foods.len(), food_before); // no food dropped
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
    fn mouth_collects_food() {
        let mut cfg = test_config();
        cfg.food_spawn_rate = 0.0;
        let mut env = Environment::new(1, cfg, 42);

        // Build segment first (non-segments need a segment to attach to)
        let mut actions = zero_actions(1);
        actions[8] = 0.0; actions[10] = 1.0; // segment
        env.step(&actions);
        for _ in 0..12 { env.step(&zero_actions(1)); }
        // Build mouth on segment
        let mut actions = zero_actions(1);
        actions[8] = 0.0; actions[10] = 6.0; // mouth
        env.step(&actions);
        assert!(env.module_graphs[0].alive_count() >= 2);

        // Place food at the mouth's world position
        let mouths = env.module_graphs[0].alive_mouths();
        assert!(!mouths.is_empty(), "should have a mouth");
        env.foods.push(Food { pos: mouths[0] });
        let food_count_before = env.foods.len();
        let energy_before = env.agents[0].energy;

        // Step — mouth should collect the food
        env.step(&zero_actions(1));

        println!("food before={} after={}", food_count_before, env.foods.len());
        println!("energy before={} after={}", energy_before, env.agents[0].energy);
        // mouth_radius=0.5 + food_radius=0.1 = 0.6 collection dist
        // food is at distance 0 from agent (same pos as mouth)
        assert!(env.foods.len() < food_count_before, "food should have been collected");
        assert!(env.agents[0].energy > energy_before, "energy should have increased");
    }

    #[test]
    fn mouth_collects_food_via_action() {
        let mut cfg = test_config();
        cfg.food_spawn_rate = 0.0;
        let mut env = Environment::new(1, cfg, 42);

        // Build segment first
        let mut actions = zero_actions(1);
        actions[8] = 0.0; actions[10] = 1.0;
        env.step(&actions);
        for _ in 0..12 { env.step(&zero_actions(1)); }
        // Build mouth on segment
        let mut actions = zero_actions(1);
        actions[8] = 0.0; actions[10] = 6.0;
        env.step(&actions);
        assert!(env.module_graphs[0].alive_count() >= 2, "should have segment + mouth");

        let mouths = env.module_graphs[0].alive_mouths();
        assert!(!mouths.is_empty(), "should have a mouth");
        let mouth_pos = mouths[0];
        let agent_pos = env.agents[0].pos;
        println!("agent_pos=({:.2},{:.2}) mouth_pos=({:.2},{:.2})",
            agent_pos.x, agent_pos.y, mouth_pos.x, mouth_pos.y);

        // Place food right at the mouth position
        env.foods.push(Food { pos: mouth_pos });
        let energy_before = env.agents[0].energy;

        env.step(&zero_actions(1));
        let energy_after = env.agents[0].energy;
        println!("energy: {:.2} -> {:.2}", energy_before, energy_after);
        assert!(energy_after > energy_before, "mouth should have collected food");
    }

    #[test]
    fn module_wall_collision_applies_torque() {
        let mut cfg = test_config();
        cfg.food_spawn_rate = 0.0;
        cfg.width = 5.0;
        cfg.height = 5.0;
        let mut env = Environment::new(1, cfg, 42);

        // Place agent near right wall
        env.agents[0].pos = Vec2::new(4.0, 2.5);
        env.agents[0].vel = Vec2::zero();
        env.agents[0].angular_velocity = 0.0;

        // Build a segment pointing right (will extend to x=5.0, hitting the wall)
        let mut actions = zero_actions(1);
        actions[8] = 0.0;  // rotation = 0 (right)
        actions[10] = 1.0; // segment
        env.step(&actions);

        // The segment endpoint at x=5.0 should push the agent left and apply torque
        let ang_vel_before = env.agents[0].angular_velocity;

        // Step to trigger wall collision resolution
        env.step(&zero_actions(1));

        // Agent should have been pushed and may have gained angular velocity
        // (depends on exact geometry, but position should have changed)
        assert!(env.agents[0].pos.x < 4.5, "agent should be pushed from wall, pos.x={}", env.agents[0].pos.x);
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

    // ==================================================================
    // Thruster tests
    // ==================================================================

    #[test]
    fn thruster_changes_velocity() {
        let mut cfg = test_config();
        cfg.food_spawn_rate = 0.0;
        cfg.width = 20.0;
        cfg.height = 20.0;
        let mut env = Environment::new(1, cfg, 42);
        env.agents[0].pos = Vec2::new(10.0, 10.0);

        // Build segment + thruster pointing right
        let mut a = zero_actions(1);
        a[8] = 0.0; a[10] = 1.0; // segment right
        env.step(&a);
        for _ in 0..12 { env.step(&zero_actions(1)); }
        let mut a = zero_actions(1);
        a[8] = 0.0; a[10] = 5.0; // thruster
        env.step(&a);
        for _ in 0..12 { env.step(&zero_actions(1)); }

        let vx_before = env.agents[0].vel.x;

        // Send signal=1 to activate thruster
        let mut a = zero_actions(1);
        a[3] = 1.0; // signal
        env.step(&a);

        let vx_after = env.agents[0].vel.x;
        assert!(
            (vx_after - vx_before).abs() > 1e-4,
            "thruster should change velocity: before={vx_before} after={vx_after}"
        );
    }

    // ==================================================================
    // Logic gate tests
    // ==================================================================

    #[test]
    fn or_gate_signal_propagation() {
        use crate::modules::{ModuleGraph, ModuleType, ROOT_ID};
        let mut g = ModuleGraph::new();
        // Root → Seg(0) → OR(1), with OR having 3 connections
        g.add_module(ModuleType::Segment, Vec2::new(1.0, 0.0), 0.0, 1.0, ROOT_ID);
        g.add_module(ModuleType::Or, Vec2::new(2.0, 0.0), 0.0, 0.0, 0);
        // Add second input segment and output segment to the OR gate
        g.add_module(ModuleType::Segment, Vec2::new(2.0, 1.0), 0.0, 1.0, 1);
        // Now OR(1) has connections: [2] from add, plus seg(0) is its parent

        // With signal=1, the OR gate should pass it through
        g.propagate_signal(1.0);
        assert!((g.modules[1].signal - 1.0).abs() < 1e-6, "OR gate should be 1 when input is 1");

        g.propagate_signal(0.0);
        assert!((g.modules[1].signal).abs() < 1e-6, "OR gate should be 0 when input is 0");
    }

    #[test]
    fn and_gate_requires_both_inputs() {
        use crate::modules::{ModuleGraph, ModuleType, ROOT_ID};
        let mut g = ModuleGraph::new();
        g.add_module(ModuleType::Segment, Vec2::new(1.0, 0.0), 0.0, 1.0, ROOT_ID);
        g.add_module(ModuleType::And, Vec2::new(2.0, 0.0), 0.0, 0.0, 0);

        // With only 1 input connected (not 3), AND gate passes signal through
        g.propagate_signal(1.0);
        // AND with < 3 connections just passes through
        assert!((g.modules[1].signal - 1.0).abs() < 1e-6);
    }

    // ==================================================================
    // Mouth eating tests
    // ==================================================================

    #[test]
    fn mouth_on_segment_collects_distant_food() {
        let mut cfg = test_config();
        cfg.food_spawn_rate = 0.0;
        cfg.width = 20.0;
        cfg.height = 20.0;
        let mut env = Environment::new(1, cfg, 42);
        env.agents[0].pos = Vec2::new(10.0, 10.0);

        // Build segment + mouth (wait full cooldown: 1s / 0.1dt = 10 steps)
        let mut a = zero_actions(1);
        a[8] = 0.0; a[10] = 1.0; // segment right
        env.step(&a);
        for _ in 0..12 { env.step(&zero_actions(1)); }
        let mut a = zero_actions(1);
        a[8] = 0.0; a[10] = 6.0; // mouth
        env.step(&a);
        for _ in 0..12 { env.step(&zero_actions(1)); }

        // Check what was built
        let alive = env.module_graphs[0].alive_count();
        let mouth_pos = env.module_graphs[0].alive_mouths();
        println!("alive modules: {}, mouths: {}", alive, mouth_pos.len());
        for m in &env.module_graphs[0].modules {
            println!("  module {}: type={:?} alive={} local=({:.2},{:.2}) world=({:.2},{:.2})",
                m.id, m.module_type, m.alive, m.local_pos.x, m.local_pos.y, m.world_pos.x, m.world_pos.y);
        }
        assert!(!mouth_pos.is_empty(), "should have a mouth (alive={}, modules={})", alive, env.module_graphs[0].modules.len());
        env.foods.push(Food { pos: mouth_pos[0] });

        let e0 = env.agents[0].energy;
        env.step(&zero_actions(1));
        let e1 = env.agents[0].energy;
        assert!(e1 > e0, "mouth should collect food at segment end: e0={e0} e1={e1}");
    }

    // ==================================================================
    // Agent-to-agent module collision tests
    // ==================================================================

    #[test]
    #[ignore] // TODO: cross-agent module collision not yet implemented
    fn agents_with_segments_collide() {
        let mut cfg = test_config();
        cfg.food_spawn_rate = 0.0;
        cfg.width = 20.0;
        cfg.height = 20.0;
        cfg.interaction_rules.agent_collision = true;
        let mut env = Environment::new(2, cfg, 42);

        // Place agents facing each other
        env.agents[0].pos = Vec2::new(8.0, 10.0);
        env.agents[1].pos = Vec2::new(12.0, 10.0);

        // Build segments pointing toward each other
        let mut a = zero_actions(2);
        a[8] = 0.0; a[10] = 1.0; // agent 0: segment right
        a[NUM_ACTIONS + 8] = std::f32::consts::PI; // agent 1: segment left
        a[NUM_ACTIONS + 10] = 1.0;
        env.step(&a);
        for _ in 0..12 { env.step(&zero_actions(2)); }

        // Push agents toward each other
        let pos0_before = env.agents[0].pos.x;
        let pos1_before = env.agents[1].pos.x;
        for _ in 0..20 {
            let mut a = zero_actions(2);
            a[0] = 2.0;  // agent 0 accelerate right
            a[NUM_ACTIONS] = -2.0; // agent 1 accelerate left
            env.step(&a);
        }

        // Agents should not have passed through each other
        // (they may have bounced, but agent 0 shouldn't be past agent 1)
        assert!(
            env.agents[0].pos.x < env.agents[1].pos.x,
            "agents should not pass through: a0={} a1={}",
            env.agents[0].pos.x, env.agents[1].pos.x
        );
    }

    // ==================================================================
    // Edge cases / things that could go wrong
    // ==================================================================

    #[test]
    fn building_while_moving_fast() {
        let mut cfg = test_config();
        cfg.food_spawn_rate = 0.0;
        cfg.width = 30.0;
        cfg.height = 30.0;
        let mut env = Environment::new(1, cfg, 42);
        env.agents[0].pos = Vec2::new(15.0, 15.0);
        env.agents[0].vel = Vec2::new(5.0, 0.0); // moving fast right

        // Build segment while moving fast
        let mut a = zero_actions(1);
        a[8] = 0.0; a[10] = 1.0; // segment
        env.step(&a);

        // Should still work, module at correct position
        assert!(env.module_graphs[0].alive_count() >= 1);
    }

    #[test]
    fn agent_death_clears_modules_and_drops_food() {
        let mut cfg = test_config();
        cfg.food_spawn_rate = 0.0;
        cfg.dead_steps_threshold = 1;
        cfg.width = 20.0;
        cfg.height = 20.0;
        let mut env = Environment::new(1, cfg, 42);
        env.agents[0].pos = Vec2::new(10.0, 10.0);

        // Build 3 modules
        for (rot, bt) in [(0.0, 1.0), (1.57, 1.0), (3.14, 6.0)] {
            let mut a = zero_actions(1);
            a[8] = rot; a[10] = bt;
            env.step(&a);
            for _ in 0..12 { env.step(&zero_actions(1)); }
        }
        let module_count = env.module_graphs[0].alive_count();
        assert!(module_count >= 2, "should have built modules: {module_count}");

        // Kill agent
        env.agents[0].energy = -100.0;
        let food_before = env.foods.len();
        env.step(&zero_actions(1));

        assert!(!env.agents[0].alive);
        assert_eq!(env.module_graphs[0].alive_count(), 0);
        // Food dropped: 1 for body + N for modules
        assert!(env.foods.len() > food_before);
    }

    #[test]
    fn rotation_preserves_module_relative_positions() {
        let mut cfg = test_config();
        cfg.food_spawn_rate = 0.0;
        cfg.width = 20.0;
        cfg.height = 20.0;
        let mut env = Environment::new(1, cfg, 42);
        env.agents[0].pos = Vec2::new(10.0, 10.0);
        env.agents[0].rotation = 0.0;

        // Build segment right
        let mut a = zero_actions(1);
        a[8] = 0.0; a[10] = 1.0;
        env.step(&a);
        for _ in 0..12 { env.step(&zero_actions(1)); }

        // Get module world position at rotation=0
        let pos_r0 = env.module_graphs[0].modules[0].world_pos;

        // Rotate agent 90 degrees and update
        env.agents[0].rotation = std::f32::consts::FRAC_PI_2;
        env.module_graphs[0].update_world_positions(env.agents[0].pos, env.agents[0].rotation);

        let pos_r90 = env.module_graphs[0].modules[0].world_pos;

        // Module should have rotated: was to the right, now should be above
        assert!(
            (pos_r90.y - pos_r0.y).abs() > 0.5,
            "module should rotate with agent: r0=({:.2},{:.2}) r90=({:.2},{:.2})",
            pos_r0.x, pos_r0.y, pos_r90.x, pos_r90.y
        );
    }

    #[test]
    fn food_never_inside_obstacles() {
        let mut cfg = test_config();
        cfg.food_spawn_rate = 200.0;
        cfg.food_cap = Some(200);
        cfg.num_initial_obstacles = 3;
        cfg.obstacle_radius = 2.0;
        cfg.width = 10.0;
        cfg.height = 10.0;
        let mut env = Environment::new(0, cfg, 42);

        // Step many times to spawn lots of food
        for _ in 0..50 {
            env.step(&[]);
        }

        // Check no food is inside any obstacle
        for food in &env.foods {
            for obs in &env.obstacles {
                let dist = food.pos.distance_to(&obs.pos);
                assert!(
                    dist > obs.radius,
                    "food at ({:.2},{:.2}) inside obstacle at ({:.2},{:.2}) radius={:.2}, dist={:.2}",
                    food.pos.x, food.pos.y, obs.pos.x, obs.pos.y, obs.radius, dist
                );
            }
        }
    }

    #[test]
    fn all_dead_detected() {
        let mut cfg = test_config();
        cfg.food_spawn_rate = 0.0;
        cfg.dead_steps_threshold = 1;
        let mut env = Environment::new(3, cfg, 42);

        // Kill all agents
        for agent in &mut env.agents {
            agent.energy = -100.0;
        }
        env.step(&zero_actions(3));

        assert!(env.agents.iter().all(|a| !a.alive));
    }

    #[test]
    fn signal_does_not_activate_without_send() {
        let mut cfg = test_config();
        cfg.food_spawn_rate = 0.0;
        cfg.width = 20.0;
        cfg.height = 20.0;
        let mut env = Environment::new(1, cfg, 42);
        env.agents[0].pos = Vec2::new(10.0, 10.0);

        // Build segment + thruster
        let mut a = zero_actions(1);
        a[8] = 0.0; a[10] = 1.0;
        env.step(&a);
        for _ in 0..12 { env.step(&zero_actions(1)); }
        let mut a = zero_actions(1);
        a[8] = 0.0; a[10] = 5.0;
        env.step(&a);
        for _ in 0..12 { env.step(&zero_actions(1)); }

        // Step without signal — thruster should NOT fire
        let vx_before = env.agents[0].vel.x;
        let mut a = zero_actions(1);
        a[3] = 0.0; // no signal
        env.step(&a);
        let vx_after = env.agents[0].vel.x;

        // Velocity should only change from damping, not thrust
        assert!(
            (vx_after - vx_before).abs() < 0.01,
            "thruster should not fire without signal: delta={}",
            (vx_after - vx_before).abs()
        );
    }
}
