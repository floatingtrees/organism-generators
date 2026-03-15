/// Core types for the evolutionary environment.

// ---------------------------------------------------------------------------
// Vec2 — 2D vector for positions, velocities, accelerations
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    pub fn magnitude(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn distance_to(&self, other: &Vec2) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    pub fn dot(&self, other: &Vec2) -> f32 {
        self.x * other.x + self.y * other.y
    }
}

impl std::ops::Add for Vec2 {
    type Output = Vec2;
    fn add(self, rhs: Vec2) -> Vec2 {
        Vec2::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Vec2;
    fn sub(self, rhs: Vec2) -> Vec2 {
        Vec2::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl std::ops::Mul<f32> for Vec2 {
    type Output = Vec2;
    fn mul(self, rhs: f32) -> Vec2 {
        Vec2::new(self.x * rhs, self.y * rhs)
    }
}

impl std::ops::AddAssign for Vec2 {
    fn add_assign(&mut self, rhs: Vec2) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Agent {
    pub id: usize,
    pub pos: Vec2,
    pub vel: Vec2,
    pub energy: f32,
    pub dead_steps: u32,
    pub alive: bool,
    pub view_size: f32,
}

impl Agent {
    pub fn new(id: usize, pos: Vec2, initial_view_size: f32) -> Self {
        Self {
            id,
            pos,
            vel: Vec2::zero(),
            energy: 10.0,
            dead_steps: 0,
            alive: true,
            view_size: initial_view_size,
        }
    }
}

// ---------------------------------------------------------------------------
// Food
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Food {
    pub pos: Vec2,
}

// ---------------------------------------------------------------------------
// Obstacle
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Obstacle {
    pub pos: Vec2,
    pub vel: Vec2,
    pub weight: f32,
    pub radius: f32,
}

// ---------------------------------------------------------------------------
// InteractionRules — toggleable rules for extensibility
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct InteractionRules {
    pub wall_bounce: bool,
    pub food_collection: bool,
    pub obstacle_collision: bool,
    pub agent_collision: bool,
    // Future rules can be added here without breaking existing configs.
}

impl Default for InteractionRules {
    fn default() -> Self {
        Self {
            wall_bounce: true,
            food_collection: true,
            obstacle_collision: true,
            agent_collision: false,
        }
    }
}

// ---------------------------------------------------------------------------
// EnvironmentConfig
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct EnvironmentConfig {
    pub width: f32,
    pub height: f32,
    pub dt: f32,
    pub food_spawn_rate: f32,
    pub energy_loss_wall: f32,
    pub object_radius: f32,
    pub num_initial_obstacles: usize,
    pub obstacle_weight: f32,
    pub dead_steps_threshold: u32,
    pub food_cap: Option<usize>,
    pub vision_cost: f32,
    pub view_res: usize,
    pub initial_view_size: f32,
    pub interaction_rules: InteractionRules,
}

impl EnvironmentConfig {
    /// Compute dead_steps_threshold from a "dead seconds" value and dt.
    /// e.g. 10 seconds at dt=0.5 → 20 steps.
    pub fn dead_threshold_from_seconds(seconds: f32, dt: f32) -> u32 {
        (seconds / dt).ceil() as u32
    }
}

impl Default for EnvironmentConfig {
    fn default() -> Self {
        let dt = 0.5;
        Self {
            width: 10.0,
            height: 10.0,
            dt,
            food_spawn_rate: 1.0,
            energy_loss_wall: 0.1,
            object_radius: 0.1,
            num_initial_obstacles: 0,
            obstacle_weight: 5.0,
            dead_steps_threshold: Self::dead_threshold_from_seconds(10.0, dt),
            food_cap: None,
            vision_cost: 0.1,
            view_res: 32,
            initial_view_size: 0.0,
            interaction_rules: InteractionRules::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec2_basic_ops() {
        let a = Vec2::new(3.0, 4.0);
        let b = Vec2::new(1.0, 2.0);

        let c = a + b;
        assert!((c.x - 4.0).abs() < 1e-6);
        assert!((c.y - 6.0).abs() < 1e-6);

        let d = a - b;
        assert!((d.x - 2.0).abs() < 1e-6);
        assert!((d.y - 2.0).abs() < 1e-6);

        let e = a * 2.0;
        assert!((e.x - 6.0).abs() < 1e-6);
        assert!((e.y - 8.0).abs() < 1e-6);
    }

    #[test]
    fn vec2_magnitude_and_distance() {
        let a = Vec2::new(3.0, 4.0);
        assert!((a.magnitude() - 5.0).abs() < 1e-6);

        let b = Vec2::new(0.0, 0.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn vec2_dot() {
        let a = Vec2::new(1.0, 0.0);
        let b = Vec2::new(0.0, 1.0);
        assert!((a.dot(&b)).abs() < 1e-6);

        let c = Vec2::new(2.0, 3.0);
        let d = Vec2::new(4.0, 5.0);
        assert!((c.dot(&d) - 23.0).abs() < 1e-6);
    }

    #[test]
    fn agent_defaults() {
        let a = Agent::new(0, Vec2::new(1.0, 2.0), 2.0);
        assert_eq!(a.id, 0);
        assert!((a.energy - 10.0).abs() < 1e-6);
        assert!(a.alive);
        assert_eq!(a.dead_steps, 0);
    }

    #[test]
    fn dead_threshold_calculation() {
        assert_eq!(EnvironmentConfig::dead_threshold_from_seconds(10.0, 0.5), 20);
        assert_eq!(EnvironmentConfig::dead_threshold_from_seconds(10.0, 0.1), 100);
        assert_eq!(EnvironmentConfig::dead_threshold_from_seconds(10.0, 1.0), 10);
    }

    #[test]
    fn default_config() {
        let cfg = EnvironmentConfig::default();
        assert!((cfg.dt - 0.5).abs() < 1e-6);
        assert_eq!(cfg.dead_steps_threshold, 20);
        assert!(cfg.interaction_rules.wall_bounce);
        assert!(cfg.interaction_rules.food_collection);
        assert!(!cfg.interaction_rules.agent_collision);
    }
}
