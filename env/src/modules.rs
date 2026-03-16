/// Module graph system for agent-built structures.
///
/// Agents can build segments, logic gates (OR/AND/XOR), thrusters, and mouths.
/// Modules form a tree rooted at the agent body. Signals propagate via BFS.
/// World positions are cached and recomputed once per step from agent pose.
///
/// Efficiency:
/// - Flat Vec<Module> storage with index-based references (no heap pointers)
/// - Cached world positions recomputed once per step: O(N)
/// - Signal propagation via BFS from root: O(V+E)
/// - Destroy connectivity check via BFS from root: O(V+E)
/// - Build attachment: scan free slots on nearby modules: O(N)

use crate::types::Vec2;

// ---------------------------------------------------------------------------
// Module types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModuleType {
    Segment,
    Or,
    And,
    Xor,
    Thruster,
    Mouth,
}

impl ModuleType {
    /// Maximum number of segment connections this module type supports.
    pub fn max_connections(&self) -> usize {
        match self {
            ModuleType::Segment => 2,   // one at each end
            ModuleType::Or => 3,        // 2 input + 1 output
            ModuleType::And => 3,
            ModuleType::Xor => 3,
            ModuleType::Thruster => 1,  // leaf, attaches to one segment end
            ModuleType::Mouth => 1,     // leaf
        }
    }

    /// Whether this is a leaf module (cannot support further segments).
    pub fn is_leaf(&self) -> bool {
        matches!(self, ModuleType::Thruster | ModuleType::Mouth)
    }

    /// Whether this is a logic gate.
    pub fn is_gate(&self) -> bool {
        matches!(self, ModuleType::Or | ModuleType::And | ModuleType::Xor)
    }

    /// Index for build_type one-hot: 0=none, 1=segment, 2=OR, 3=AND, 4=XOR, 5=thruster, 6=mouth
    pub fn from_index(i: usize) -> Option<ModuleType> {
        match i {
            1 => Some(ModuleType::Segment),
            2 => Some(ModuleType::Or),
            3 => Some(ModuleType::And),
            4 => Some(ModuleType::Xor),
            5 => Some(ModuleType::Thruster),
            6 => Some(ModuleType::Mouth),
            _ => None,
        }
    }

    pub fn to_index(&self) -> usize {
        match self {
            ModuleType::Segment => 1,
            ModuleType::Or => 2,
            ModuleType::And => 3,
            ModuleType::Xor => 4,
            ModuleType::Thruster => 5,
            ModuleType::Mouth => 6,
        }
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// A single module in an agent's build tree.
#[derive(Clone, Debug)]
pub struct Module {
    pub id: usize,
    pub module_type: ModuleType,
    /// Position relative to agent center (local frame, before rotation).
    pub local_pos: Vec2,
    /// Rotation in radians (local frame).
    pub rotation: f32,
    /// Length (only meaningful for segments).
    pub length: f32,
    /// Parent module index (ROOT_ID if connected to agent body).
    pub parent: usize,
    /// Indices of child modules. For gates: first 2 are inputs, last is output.
    pub connections: Vec<usize>,
    /// Cached world position (recomputed each step).
    pub world_pos: Vec2,
    /// For segments: cached world endpoint (start = world_pos, end = world_end).
    pub world_end: Vec2,
    /// Current signal value (0.0 or 1.0).
    pub signal: f32,
    /// Whether this module is still alive (not destroyed).
    pub alive: bool,
}

impl Module {
    pub fn new(id: usize, module_type: ModuleType, local_pos: Vec2, rotation: f32, length: f32) -> Self {
        Self {
            id,
            module_type,
            local_pos,
            rotation,
            length,
            parent: ROOT_ID,
            connections: Vec::new(),
            world_pos: Vec2::zero(),
            world_end: Vec2::zero(),
            signal: 0.0,
            alive: true,
        }
    }

    pub fn has_free_slot(&self) -> bool {
        // Total occupied slots = children + 1 if has parent (except root-connected)
        let used = self.connections.len() + if self.parent != ROOT_ID { 1 } else { 0 };
        self.alive && used < self.module_type.max_connections()
    }

    /// For segments: compute the endpoint given start position and rotation.
    pub fn segment_end_local(&self) -> Vec2 {
        Vec2::new(
            self.local_pos.x + self.length * self.rotation.cos(),
            self.local_pos.y + self.length * self.rotation.sin(),
        )
    }
}

// ---------------------------------------------------------------------------
// Pending build
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct PendingBuild {
    pub module_type: ModuleType,
    pub local_pos: Vec2,
    pub rotation: f32,
    pub length: f32,
    /// Steps remaining until materialization.
    pub steps_remaining: u32,
    /// Index of the module this attaches to (validated at build time).
    pub attach_to: usize,
}

// ---------------------------------------------------------------------------
// Module graph
// ---------------------------------------------------------------------------

/// The complete module graph for one agent.
/// Root is implicit (the agent body, index = usize::MAX sentinel).
#[derive(Clone, Debug)]
pub struct ModuleGraph {
    pub modules: Vec<Module>,
    /// Indices of modules directly connected to the agent body (root).
    pub root_connections: Vec<usize>,
    /// Pending builds waiting to materialize.
    pub pending_builds: Vec<PendingBuild>,
    /// Count of each module type built (for quadratic energy cost).
    pub type_counts: [usize; 7], // indexed by ModuleType::to_index()
    /// Cooldown timer: steps remaining before next build allowed.
    pub build_cooldown: u32,
    /// Number of free modules per type (first N are free).
    pub free_modules_per_type: usize,
}

/// Sentinel value representing the agent body (root node).
pub const ROOT_ID: usize = usize::MAX;

impl ModuleGraph {
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
            root_connections: Vec::new(),
            pending_builds: Vec::new(),
            type_counts: [0; 7],
            build_cooldown: 0,
            free_modules_per_type: 2,
        }
    }

    /// Number of alive modules.
    pub fn alive_count(&self) -> usize {
        self.modules.iter().filter(|m| m.alive).count()
    }

    /// Get module by id, if alive.
    pub fn get(&self, id: usize) -> Option<&Module> {
        self.modules.get(id).filter(|m| m.alive)
    }

    pub fn get_mut(&mut self, id: usize) -> Option<&mut Module> {
        self.modules.get_mut(id).and_then(|m| if m.alive { Some(m) } else { None })
    }

    // ------------------------------------------------------------------
    // World position update (call once per step)
    // ------------------------------------------------------------------

    /// Recompute all world positions from agent position and rotation.
    pub fn update_world_positions(&mut self, agent_pos: Vec2, agent_rotation: f32) {
        let cos_r = agent_rotation.cos();
        let sin_r = agent_rotation.sin();

        for module in &mut self.modules {
            if !module.alive {
                continue;
            }
            // Rotate local position by agent rotation
            module.world_pos = Vec2::new(
                agent_pos.x + module.local_pos.x * cos_r - module.local_pos.y * sin_r,
                agent_pos.y + module.local_pos.x * sin_r + module.local_pos.y * cos_r,
            );

            // For segments, compute world endpoint
            if module.module_type == ModuleType::Segment {
                let world_rot = module.rotation + agent_rotation;
                module.world_end = Vec2::new(
                    module.world_pos.x + module.length * world_rot.cos(),
                    module.world_pos.y + module.length * world_rot.sin(),
                );
            } else {
                module.world_end = module.world_pos;
            }
        }
    }

    // ------------------------------------------------------------------
    // Build
    // ------------------------------------------------------------------

    /// Find a free slot for building a given module type.
    /// Segments attach to root. Non-segments MUST attach to a segment endpoint.
    pub fn find_free_slot_for(&self, build_type: ModuleType) -> Option<usize> {
        if build_type == ModuleType::Segment {
            // Segments attach to root
            return Some(ROOT_ID);
        }
        // Non-segments must attach to a segment endpoint with a free slot
        for m in &self.modules {
            if m.alive && m.module_type == ModuleType::Segment && m.has_free_slot() {
                return Some(m.id);
            }
        }
        // No segment with free slot — can't build non-segment module
        None
    }

    /// Find the nearest module (or root) with a free slot to the given local position.
    /// Returns (module_id_or_ROOT_ID, distance).
    pub fn find_nearest_free_slot(&self, local_pos: Vec2, agent_radius: f32) -> Option<(usize, f32)> {
        // Check root (agent body) — unlimited slots
        let root_dist = local_pos.magnitude();
        let mut best: Option<(usize, f32)> = Some((ROOT_ID, root_dist));

        // Check all alive modules with free slots
        for module in &self.modules {
            if !module.alive || !module.has_free_slot() {
                continue;
            }

            // For segments, check both endpoints
            let dist = if module.module_type == ModuleType::Segment {
                let end = module.segment_end_local();
                module.local_pos.distance_to(&local_pos).min(end.distance_to(&local_pos))
            } else {
                module.local_pos.distance_to(&local_pos)
            };

            if let Some((_, best_dist)) = best {
                if dist < best_dist {
                    best = Some((module.id, dist));
                }
            } else {
                best = Some((module.id, dist));
            }
        }

        best
    }

    /// Add a module to the graph. Returns the new module's id, or None if invalid.
    pub fn add_module(
        &mut self,
        module_type: ModuleType,
        local_pos: Vec2,
        rotation: f32,
        length: f32,
        attach_to: usize,
    ) -> Option<usize> {
        let id = self.modules.len();
        let mut module = Module::new(id, module_type, local_pos, rotation, length.min(1.0));
        module.parent = attach_to;

        // Connect to parent
        if attach_to == ROOT_ID {
            self.root_connections.push(id);
        } else {
            if let Some(parent) = self.get_mut(attach_to) {
                if !parent.has_free_slot() {
                    return None;
                }
                parent.connections.push(id);
            } else {
                return None;
            }
        }

        // For non-segment modules attaching to a segment, snap to nearest endpoint
        if module_type != ModuleType::Segment && attach_to != ROOT_ID {
            if let Some(parent) = self.get(attach_to) {
                if parent.module_type == ModuleType::Segment {
                    let end = parent.segment_end_local();
                    let dist_start = parent.local_pos.distance_to(&local_pos);
                    let dist_end = end.distance_to(&local_pos);
                    module.local_pos = if dist_start < dist_end {
                        parent.local_pos
                    } else {
                        end
                    };
                }
            }
        }

        self.type_counts[module_type.to_index()] += 1;
        self.modules.push(module);
        Some(id)
    }

    // ------------------------------------------------------------------
    // Destroy
    // ------------------------------------------------------------------

    /// Destroy a module and cascade-remove any modules that become disconnected
    /// from root. Returns positions of all removed modules (for food drops).
    ///
    /// Algorithm: BFS from root to find all reachable modules after removal.
    /// Any module not in the reachable set is cascade-destroyed. O(V+E).
    pub fn destroy_module(&mut self, module_id: usize) -> Vec<Vec2> {
        let mut removed_positions = Vec::new();

        if module_id >= self.modules.len() || !self.modules[module_id].alive {
            return removed_positions;
        }

        // Mark the target as dead
        let mtype = self.modules[module_id].module_type;
        self.modules[module_id].alive = false;
        removed_positions.push(self.modules[module_id].world_pos);
        if self.type_counts[mtype.to_index()] > 0 {
            self.type_counts[mtype.to_index()] -= 1;
        }

        // Remove from root_connections if present
        self.root_connections.retain(|&id| id != module_id);

        // Remove from parent connections
        for m in &mut self.modules {
            m.connections.retain(|&id| id != module_id);
        }

        // BFS from root to find all reachable alive modules
        let reachable = self.bfs_reachable_from_root();

        // Cascade-destroy unreachable modules
        for i in 0..self.modules.len() {
            if self.modules[i].alive && !reachable[i] {
                self.modules[i].alive = false;
                removed_positions.push(self.modules[i].world_pos);
                let mt = self.modules[i].module_type;
                if self.type_counts[mt.to_index()] > 0 {
                    self.type_counts[mt.to_index()] -= 1;
                }
            }
        }

        removed_positions
    }

    /// BFS from root, returning set of reachable alive module indices.
    fn bfs_reachable_from_root(&self) -> Vec<bool> {
        let n = self.modules.len();
        let mut visited = vec![false; n];
        let mut queue = std::collections::VecDeque::new();

        // Start from root connections
        for &id in &self.root_connections {
            if id < n && self.modules[id].alive {
                visited[id] = true;
                queue.push_back(id);
            }
        }

        while let Some(id) = queue.pop_front() {
            for &child in &self.modules[id].connections {
                if child < n && self.modules[child].alive && !visited[child] {
                    visited[child] = true;
                    queue.push_back(child);
                }
            }
        }

        visited
    }

    // ------------------------------------------------------------------
    // Signal propagation
    // ------------------------------------------------------------------

    /// Propagate signal from root through the module graph via BFS.
    /// Root signal is the agent's output signal (0.0 or 1.0).
    ///
    /// Segments pass signal through. Gates compute logic on their inputs.
    /// Thrusters and mouths receive signal but don't propagate.
    pub fn propagate_signal(&mut self, root_signal: f32) {
        // Reset all signals
        for m in &mut self.modules {
            m.signal = 0.0;
        }

        let mut visited = vec![false; self.modules.len()];
        let mut queue = std::collections::VecDeque::new();

        // Seed: root connections get the root signal
        for &id in &self.root_connections {
            if id < self.modules.len() && self.modules[id].alive {
                self.modules[id].signal = root_signal;
                visited[id] = true;
                queue.push_back(id);
            }
        }

        while let Some(id) = queue.pop_front() {
            let current_signal = self.modules[id].signal;
            let connections: Vec<usize> = self.modules[id].connections.clone();

            for &child in &connections {
                if child >= self.modules.len() || !self.modules[child].alive || visited[child] {
                    continue;
                }
                visited[child] = true;

                match self.modules[child].module_type {
                    ModuleType::Segment | ModuleType::Thruster | ModuleType::Mouth => {
                        // Passthrough: inherit parent's signal
                        self.modules[child].signal = current_signal;
                        queue.push_back(child);
                    }
                    ModuleType::Or | ModuleType::And | ModuleType::Xor => {
                        // Gate: compute output from all visited input connections
                        // Gate needs 3 connections to be active (2 in + 1 out)
                        let gate_conns: Vec<usize> = self.modules[child].connections.clone();
                        if gate_conns.len() >= 3 {
                            let in1 = if visited[gate_conns[0]] { self.modules[gate_conns[0]].signal } else { 0.0 };
                            let in2 = if visited[gate_conns[1]] { self.modules[gate_conns[1]].signal } else { 0.0 };
                            let gate_out = match self.modules[child].module_type {
                                ModuleType::Or => if in1 > 0.5 || in2 > 0.5 { 1.0 } else { 0.0 },
                                ModuleType::And => if in1 > 0.5 && in2 > 0.5 { 1.0 } else { 0.0 },
                                ModuleType::Xor => if (in1 > 0.5) ^ (in2 > 0.5) { 1.0 } else { 0.0 },
                                _ => 0.0,
                            };
                            self.modules[child].signal = gate_out;
                        } else {
                            self.modules[child].signal = current_signal;
                        }
                        queue.push_back(child);
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Thruster force/torque calculation
    // ------------------------------------------------------------------

    /// Compute total force and torque from active thrusters.
    /// Returns (force_x, force_y, torque).
    /// Torque is computed around the center of mass of non-segment modules.
    pub fn compute_thruster_effects(&self, agent_pos: Vec2, agent_rotation: f32, base_accel: f32) -> (f32, f32, f32) {
        let thrust_force = base_accel * 0.5; // half a standard acceleration

        // Compute center of mass (non-segment modules have weight 1)
        let mut com = agent_pos; // agent body has weight 1
        let mut total_weight = 1.0f32;
        for m in &self.modules {
            if !m.alive || m.module_type == ModuleType::Segment {
                continue;
            }
            com.x += m.world_pos.x;
            com.y += m.world_pos.y;
            total_weight += 1.0;
        }
        com.x /= total_weight;
        com.y /= total_weight;

        let mut fx = 0.0f32;
        let mut fy = 0.0f32;
        let mut torque = 0.0f32;

        for m in &self.modules {
            if !m.alive || m.module_type != ModuleType::Thruster || m.signal < 0.5 {
                continue;
            }

            // Thrust direction in world frame
            let world_rot = m.rotation + agent_rotation;
            let dx = thrust_force * world_rot.cos();
            let dy = thrust_force * world_rot.sin();
            fx += dx;
            fy += dy;

            // Torque = r × F (cross product in 2D = rx*Fy - ry*Fx)
            let rx = m.world_pos.x - com.x;
            let ry = m.world_pos.y - com.y;
            torque += rx * dy - ry * dx;
        }

        (fx, fy, torque)
    }

    // ------------------------------------------------------------------
    // Query helpers
    // ------------------------------------------------------------------

    /// Get all alive segment world positions as (start, end, width) for collision/rendering.
    pub fn alive_segments(&self) -> Vec<(Vec2, Vec2, f32)> {
        let segment_width = 0.2;
        self.modules
            .iter()
            .filter(|m| m.alive && m.module_type == ModuleType::Segment)
            .map(|m| (m.world_pos, m.world_end, segment_width))
            .collect()
    }

    /// Get all alive mouth world positions.
    pub fn alive_mouths(&self) -> Vec<Vec2> {
        self.modules
            .iter()
            .filter(|m| m.alive && m.module_type == ModuleType::Mouth)
            .map(|m| m.world_pos)
            .collect()
    }

    /// Get all alive module world positions with their types (for rendering/collision).
    pub fn alive_modules_with_types(&self) -> Vec<(Vec2, ModuleType, f32)> {
        self.modules
            .iter()
            .filter(|m| m.alive)
            .map(|m| (m.world_pos, m.module_type, m.rotation))
            .collect()
    }

    /// Find nearest own module to a local position. Returns (module_id, distance).
    pub fn find_nearest_module(&self, local_pos: Vec2) -> Option<(usize, f32)> {
        self.modules
            .iter()
            .filter(|m| m.alive)
            .map(|m| {
                let dist = if m.module_type == ModuleType::Segment {
                    // Check distance to segment line, not just start point
                    point_to_segment_dist(local_pos, m.local_pos, m.segment_end_local())
                } else {
                    m.local_pos.distance_to(&local_pos)
                };
                (m.id, dist)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    }

    /// Energy cost to build the next module of a given type.
    /// First `free_modules_per_type` are free. After that, cost = (count - free)^2.
    pub fn build_cost(&self, module_type: ModuleType) -> f32 {
        let count = self.type_counts[module_type.to_index()];
        if count < self.free_modules_per_type {
            0.0
        } else {
            let paid = count - self.free_modules_per_type;
            ((paid + 1) * (paid + 1)) as f32
        }
    }

    /// Tick pending builds. Returns list of builds ready to materialize.
    pub fn tick_pending(&mut self, _dt: f32) -> Vec<PendingBuild> {
        let steps_per_tick = 1; // each step decrements by 1
        let mut ready = Vec::new();
        self.pending_builds.retain_mut(|pb| {
            if pb.steps_remaining <= steps_per_tick as u32 {
                ready.push(pb.clone());
                false
            } else {
                pb.steps_remaining -= steps_per_tick as u32;
                true
            }
        });
        ready
    }
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/// Distance from point to line segment.
fn point_to_segment_dist(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let ap = p - a;
    let ab_len_sq = ab.x * ab.x + ab.y * ab.y;
    if ab_len_sq < 1e-10 {
        return p.distance_to(&a);
    }
    let t = (ap.x * ab.x + ap.y * ab.y) / ab_len_sq;
    let t = t.clamp(0.0, 1.0);
    let closest = Vec2::new(a.x + t * ab.x, a.y + t * ab.y);
    p.distance_to(&closest)
}

/// Check if a point is within distance of a line segment.
pub fn point_near_segment(p: Vec2, seg_start: Vec2, seg_end: Vec2, dist: f32) -> bool {
    point_to_segment_dist(p, seg_start, seg_end) < dist
}

/// Check if two line segments intersect or are within a minimum distance.
pub fn segments_near(a1: Vec2, a2: Vec2, b1: Vec2, b2: Vec2, min_dist: f32) -> bool {
    // Simplified: check if any endpoint of one segment is near the other
    point_near_segment(a1, b1, b2, min_dist)
        || point_near_segment(a2, b1, b2, min_dist)
        || point_near_segment(b1, a1, a2, min_dist)
        || point_near_segment(b2, a1, a2, min_dist)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_type_indices() {
        assert_eq!(ModuleType::from_index(1), Some(ModuleType::Segment));
        assert_eq!(ModuleType::from_index(6), Some(ModuleType::Mouth));
        assert_eq!(ModuleType::from_index(0), None);
        assert_eq!(ModuleType::from_index(7), None);
    }

    #[test]
    fn empty_graph() {
        let g = ModuleGraph::new();
        assert_eq!(g.alive_count(), 0);
        assert!(g.root_connections.is_empty());
    }

    #[test]
    fn add_segment_to_root() {
        let mut g = ModuleGraph::new();
        let id = g.add_module(ModuleType::Segment, Vec2::new(0.3, 0.0), 0.0, 0.8, ROOT_ID);
        assert_eq!(id, Some(0));
        assert_eq!(g.alive_count(), 1);
        assert_eq!(g.root_connections, vec![0]);
        assert_eq!(g.type_counts[1], 1); // Segment index = 1
    }

    #[test]
    fn add_mouth_to_segment_end() {
        let mut g = ModuleGraph::new();
        g.add_module(ModuleType::Segment, Vec2::new(0.3, 0.0), 0.0, 0.8, ROOT_ID);
        let mouth_id = g.add_module(ModuleType::Mouth, Vec2::new(1.1, 0.0), 0.0, 0.0, 0);
        assert_eq!(mouth_id, Some(1));
        assert_eq!(g.alive_count(), 2);
        assert_eq!(g.modules[0].connections, vec![1]);
    }

    #[test]
    fn leaf_cannot_support_segments() {
        let mut g = ModuleGraph::new();
        g.add_module(ModuleType::Segment, Vec2::new(0.3, 0.0), 0.0, 0.8, ROOT_ID);
        g.add_module(ModuleType::Mouth, Vec2::new(1.1, 0.0), 0.0, 0.0, 0);
        // Mouth (id=1) is a leaf, cannot support a segment
        let seg2 = g.add_module(ModuleType::Segment, Vec2::new(1.1, 0.0), 0.0, 0.5, 1);
        assert_eq!(seg2, None);
    }

    #[test]
    fn destroy_cascades() {
        let mut g = ModuleGraph::new();
        // Root → Segment(0) → Gate(1) → Segment(2) → Mouth(3)
        g.add_module(ModuleType::Segment, Vec2::new(0.3, 0.0), 0.0, 0.8, ROOT_ID);
        g.add_module(ModuleType::Or, Vec2::new(1.1, 0.0), 0.0, 0.0, 0);
        // Gate needs more connections to be useful, but for destroy test it's fine
        g.add_module(ModuleType::Segment, Vec2::new(1.1, 0.0), 0.5, 0.8, 1);
        g.add_module(ModuleType::Mouth, Vec2::new(1.5, 0.5), 0.0, 0.0, 2);

        // Update world positions
        g.update_world_positions(Vec2::new(5.0, 5.0), 0.0);

        // Destroy segment 0 — should cascade to gate, segment 2, mouth
        let removed = g.destroy_module(0);
        assert_eq!(removed.len(), 4); // all 4 modules removed
        assert_eq!(g.alive_count(), 0);
    }

    #[test]
    fn destroy_partial_cascade() {
        let mut g = ModuleGraph::new();
        // Root → Seg(0), Root → Seg(1) → Mouth(2)
        g.add_module(ModuleType::Segment, Vec2::new(0.3, 0.0), 0.0, 0.8, ROOT_ID);
        g.add_module(ModuleType::Segment, Vec2::new(-0.3, 0.0), std::f32::consts::PI, 0.8, ROOT_ID);
        g.add_module(ModuleType::Mouth, Vec2::new(-1.1, 0.0), 0.0, 0.0, 1);

        g.update_world_positions(Vec2::new(5.0, 5.0), 0.0);

        // Destroy seg(0) — only seg(0) removed, seg(1)+mouth(2) still connected to root
        let removed = g.destroy_module(0);
        assert_eq!(removed.len(), 1);
        assert_eq!(g.alive_count(), 2);
    }

    #[test]
    fn build_cost_with_free_modules() {
        let mut g = ModuleGraph::new();
        // First 2 are free (free_modules_per_type = 2)
        assert_eq!(g.build_cost(ModuleType::Segment), 0.0); // first free
        g.add_module(ModuleType::Segment, Vec2::new(0.3, 0.0), 0.0, 0.8, ROOT_ID);
        assert_eq!(g.build_cost(ModuleType::Segment), 0.0); // second still free
        g.add_module(ModuleType::Segment, Vec2::new(-0.3, 0.0), 0.0, 0.8, ROOT_ID);
        assert_eq!(g.build_cost(ModuleType::Segment), 1.0); // third costs 1 (1^2)
        g.add_module(ModuleType::Segment, Vec2::new(0.0, 0.3), 1.57, 0.8, ROOT_ID);
        assert_eq!(g.build_cost(ModuleType::Segment), 4.0); // fourth costs 4 (2^2)

        // Different type is independent
        assert_eq!(g.build_cost(ModuleType::Mouth), 0.0); // first mouth free
    }

    #[test]
    fn signal_propagation_simple() {
        let mut g = ModuleGraph::new();
        // Root → Segment(0) → Thruster(1)
        g.add_module(ModuleType::Segment, Vec2::new(0.3, 0.0), 0.0, 0.8, ROOT_ID);
        g.add_module(ModuleType::Thruster, Vec2::new(1.1, 0.0), 0.0, 0.0, 0);

        g.propagate_signal(1.0);
        assert!((g.modules[0].signal - 1.0).abs() < 1e-6); // segment gets signal
        assert!((g.modules[1].signal - 1.0).abs() < 1e-6); // thruster gets signal

        g.propagate_signal(0.0);
        assert!((g.modules[0].signal).abs() < 1e-6);
        assert!((g.modules[1].signal).abs() < 1e-6);
    }

    #[test]
    fn signal_propagation_or_gate() {
        let mut g = ModuleGraph::new();
        // Root → Seg(0) → OR(1), Root → Seg(2) → OR(1), OR(1) → Seg(3) → Thruster(4)
        g.add_module(ModuleType::Segment, Vec2::new(0.3, 0.0), 0.0, 0.8, ROOT_ID);  // 0
        g.add_module(ModuleType::Or, Vec2::new(1.1, 0.0), 0.0, 0.0, 0);             // 1
        g.add_module(ModuleType::Segment, Vec2::new(0.0, 0.3), 1.57, 0.8, ROOT_ID);  // 2
        // Connect seg(2) to the OR gate
        g.modules[1].connections.push(2); // manually connect for test
        g.modules[2].connections.push(1);
        // Add output segment from OR
        g.add_module(ModuleType::Segment, Vec2::new(1.5, 0.0), 0.0, 0.5, 1);        // 3
        g.add_module(ModuleType::Thruster, Vec2::new(2.0, 0.0), 0.0, 0.0, 3);       // 4

        g.propagate_signal(1.0);
        // OR gate should output 1 (at least one input is 1)
        assert!((g.modules[1].signal - 1.0).abs() < 1e-6);
    }

    #[test]
    fn world_position_update() {
        let mut g = ModuleGraph::new();
        g.add_module(ModuleType::Segment, Vec2::new(1.0, 0.0), 0.0, 1.0, ROOT_ID);

        // No rotation
        g.update_world_positions(Vec2::new(5.0, 5.0), 0.0);
        assert!((g.modules[0].world_pos.x - 6.0).abs() < 1e-5);
        assert!((g.modules[0].world_pos.y - 5.0).abs() < 1e-5);
        assert!((g.modules[0].world_end.x - 7.0).abs() < 1e-5);

        // 90 degree rotation
        let pi_half = std::f32::consts::FRAC_PI_2;
        g.update_world_positions(Vec2::new(5.0, 5.0), pi_half);
        assert!((g.modules[0].world_pos.x - 5.0).abs() < 1e-4);
        assert!((g.modules[0].world_pos.y - 6.0).abs() < 1e-4);
    }

    #[test]
    fn point_to_segment_distance() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(2.0, 0.0);
        let p = Vec2::new(1.0, 1.0);
        let dist = point_to_segment_dist(p, a, b);
        assert!((dist - 1.0).abs() < 1e-5);

        // Point beyond segment end
        let p2 = Vec2::new(3.0, 0.0);
        let dist2 = point_to_segment_dist(p2, a, b);
        assert!((dist2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn find_nearest_free_slot() {
        let mut g = ModuleGraph::new();
        g.add_module(ModuleType::Segment, Vec2::new(1.0, 0.0), 0.0, 1.0, ROOT_ID);

        // Point near segment end should find the segment
        let (id, _) = g.find_nearest_free_slot(Vec2::new(2.0, 0.0), 0.3).unwrap();
        assert_eq!(id, 0); // segment is closer than root
    }

    #[test]
    fn pending_build_ticks() {
        let mut g = ModuleGraph::new();
        g.pending_builds.push(PendingBuild {
            module_type: ModuleType::Segment,
            local_pos: Vec2::new(0.3, 0.0),
            rotation: 0.0,
            length: 0.5,
            steps_remaining: 3,
            attach_to: ROOT_ID,
        });

        let ready = g.tick_pending(0.15);
        assert!(ready.is_empty());
        assert_eq!(g.pending_builds[0].steps_remaining, 2);

        let ready = g.tick_pending(0.15);
        assert!(ready.is_empty());
        assert_eq!(g.pending_builds[0].steps_remaining, 1);

        // At steps_remaining=1, next tick should pop it
        let ready = g.tick_pending(0.15);
        assert_eq!(ready.len(), 1);
        assert!(g.pending_builds.is_empty());
    }
}
