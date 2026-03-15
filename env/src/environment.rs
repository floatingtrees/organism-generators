/// Single environment instance with continuous 2D physics and egocentric views.

use crate::spatial_hash::SpatialHash;
use crate::types::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;

/// Observation channels per frame: food, alive_agent, dead_agent, obstacle.
pub const NUM_VIEW_CHANNELS: usize = 4;
/// Scalar self-state features: energy, vx, vy, view_size.
pub const NUM_SCALAR_FEATURES: usize = 4;
/// Number of previous frames stacked with the current frame.
pub const NUM_HISTORY_FRAMES: usize = 3;
/// Total channels in an observation: 4 channels × (1 current + 3 history).
pub const TOTAL_VIEW_CHANNELS: usize = NUM_VIEW_CHANNELS * (1 + NUM_HISTORY_FRAMES);
/// Action dimensions: (ax, ay, view_delta).
pub const NUM_ACTIONS: usize = 3;

pub struct Environment {
    pub config: EnvironmentConfig,
    pub agents: Vec<Agent>,
    pub foods: Vec<Food>,
    pub obstacles: Vec<Obstacle>,
    pub step_count: u64,
    rng: SmallRng,
    seed: u64,
    spatial: SpatialHash,
    /// Current rendered views for all agents: [num_agents * res * res * NUM_VIEW_CHANNELS].
    current_view: Vec<f32>,
    /// Ring buffer of 3 previous frames (front = most recent).
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

        let mut env = Self {
            config,
            agents,
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

    pub fn step(&mut self, actions: &[(f32, f32, f32)]) {
        let dt = self.config.dt;
        let radius = self.config.object_radius;
        let max_vs = self.config.width.max(self.config.height) / 2.0;

        // 1. Apply accelerations and view updates to alive agents
        for (i, agent) in self.agents.iter_mut().enumerate() {
            if !agent.alive {
                continue;
            }
            let (ax, ay, vd) = if i < actions.len() {
                actions[i]
            } else {
                (0.0, 0.0, 0.0)
            };
            let accel = Vec2::new(ax, ay);

            agent.vel += accel * dt;
            agent.energy -= accel.magnitude() * dt;
            agent.pos += agent.vel * dt;

            // View size update
            agent.view_size = (agent.view_size + vd * dt).clamp(radius, max_vs);

            // Vision cost: vision_cost * view_size^2 per second (quadratic)
            agent.energy -= self.config.vision_cost * agent.view_size * agent.view_size * dt;
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

    /// Returns flat [num_agents * NUM_SCALAR_FEATURES]: energy, vx, vy, view_size.
    /// Dead/padded agents get zeros.
    pub fn get_agent_states(&self) -> Vec<f32> {
        let mut states = Vec::with_capacity(self.agents.len() * NUM_SCALAR_FEATURES);
        for a in &self.agents {
            if a.alive {
                states.extend_from_slice(&[a.energy, a.vel.x, a.vel.y, a.view_size]);
            } else {
                states.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
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
            obstacle_radius: 0.1,
            dead_steps_threshold: 100,
            food_cap: None,
            vision_cost: 0.0, // no vision cost in tests by default
            view_res: 8,      // small for fast tests
            initial_view_size: 2.0,
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
            assert!((agent.view_size - 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn zero_action_no_acceleration() {
        let mut env = Environment::new(2, test_config(), 42);
        let p0 = env.agents[0].pos;
        let p1 = env.agents[1].pos;

        env.step(&[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]);

        assert!((env.agents[0].pos.x - p0.x).abs() < 1e-6);
        assert!((env.agents[1].pos.x - p1.x).abs() < 1e-6);
    }

    #[test]
    fn acceleration_changes_position() {
        let mut env = Environment::new(1, test_config(), 42);
        let p0 = env.agents[0].pos;
        let dt = env.config.dt;

        env.step(&[(1.0, 0.0, 0.0)]);

        assert!((env.agents[0].vel.x - dt).abs() < 1e-6);
        assert!((env.agents[0].pos.x - (p0.x + dt * dt)).abs() < 1e-5);
    }

    #[test]
    fn view_delta_changes_view_size() {
        let mut env = Environment::new(1, test_config(), 42);
        let dt = env.config.dt;
        let vs0 = env.agents[0].view_size;

        env.step(&[(0.0, 0.0, 5.0)]); // increase view size

        assert!((env.agents[0].view_size - (vs0 + 5.0 * dt)).abs() < 1e-5);
    }

    #[test]
    fn view_size_clamped_to_bounds() {
        let cfg = test_config();
        let max_vs = cfg.width.max(cfg.height) / 2.0;
        let mut env = Environment::new(1, cfg.clone(), 42);

        // Push view_size way up
        for _ in 0..1000 {
            env.step(&[(0.0, 0.0, 100.0)]);
        }
        assert!(env.agents[0].view_size <= max_vs + 1e-6);

        // Push view_size to minimum (object_radius)
        for _ in 0..1000 {
            env.step(&[(0.0, 0.0, -100.0)]);
        }
        assert!(env.agents[0].view_size >= cfg.object_radius - 1e-6);
    }

    #[test]
    fn vision_cost_drains_energy() {
        let mut cfg = test_config();
        cfg.vision_cost = 1.0;
        cfg.food_spawn_rate = 0.0;
        let mut env = Environment::new(1, cfg, 42);

        let e0 = env.agents[0].energy;
        env.step(&[(0.0, 0.0, 0.0)]);
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
            env.step(&[(0.0, 0.0, 0.0)]);
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
            env.step(&[]);
        }
        assert!(env.foods.len() <= 10);
    }

    #[test]
    fn reset_clean_state() {
        let mut cfg = test_config();
        cfg.food_spawn_rate = 50.0;
        let mut env = Environment::new(3, cfg, 42);

        for _ in 0..10 {
            env.step(&[(1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (-1.0, -1.0, -1.0)]);
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
}
