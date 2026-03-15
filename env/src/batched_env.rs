/// Batched wrapper that runs multiple environments in parallel (logically).

use crate::environment::Environment;
use crate::types::EnvironmentConfig;

pub struct BatchedEnvironment {
    pub envs: Vec<Environment>,
    pub max_agents: usize,
}

impl BatchedEnvironment {
    pub fn new(
        num_agents_per_env: Vec<usize>,
        config: EnvironmentConfig,
        base_seed: u64,
    ) -> Self {
        let max_agents = *num_agents_per_env.iter().max().unwrap_or(&0);

        let envs: Vec<Environment> = num_agents_per_env
            .iter()
            .enumerate()
            .map(|(i, &n)| Environment::new(n, config.clone(), base_seed.wrapping_add(i as u64)))
            .collect();

        Self { envs, max_agents }
    }

    pub fn num_envs(&self) -> usize {
        self.envs.len()
    }

    // ------------------------------------------------------------------
    // Reset
    // ------------------------------------------------------------------

    pub fn reset(&mut self) {
        for env in &mut self.envs {
            env.reset();
        }
        self.max_agents = self.envs.iter().map(|e| e.num_agents()).max().unwrap_or(0);
    }

    // ------------------------------------------------------------------
    // Step
    // ------------------------------------------------------------------

    /// `actions` is a flat slice of length `num_envs * max_agents * 2`.
    /// Layout: [env0_agent0_x, env0_agent0_y, env0_agent1_x, ..., env1_agent0_x, ...]
    pub fn step(&mut self, actions: &[f32]) -> Vec<f32> {
        let max_a = self.max_agents;

        for (i, env) in self.envs.iter_mut().enumerate() {
            let n = env.num_agents();
            let env_actions: Vec<(f32, f32)> = (0..n)
                .map(|j| {
                    let base = (i * max_a + j) * 2;
                    (actions[base], actions[base + 1])
                })
                .collect();
            env.step(&env_actions);
        }

        self.get_rewards()
    }

    // ------------------------------------------------------------------
    // Observe
    // ------------------------------------------------------------------

    pub const NUM_FEATURES: usize = 5; // x, y, vx, vy, alive

    /// Returns flat `[num_envs * max_agents * NUM_FEATURES]`.
    /// Features per agent: [x, y, vx, vy, alive].
    /// Padded agent slots are all zeros (alive=0).
    pub fn observe(&self) -> Vec<f32> {
        let num_envs = self.envs.len();
        let max_a = self.max_agents;
        let nf = Self::NUM_FEATURES;

        let mut features = vec![0.0f32; num_envs * max_a * nf];

        for (i, env) in self.envs.iter().enumerate() {
            let af = env.get_agent_features();
            for (j, feats) in af.iter().enumerate() {
                let base = (i * max_a + j) * nf;
                for (k, &v) in feats.iter().enumerate() {
                    features[base + k] = v;
                }
            }
        }

        features
    }

    // ------------------------------------------------------------------
    // Rewards
    // ------------------------------------------------------------------

    /// Flat `[num_envs * max_agents]` — current energy per agent, 0 for padding.
    pub fn get_rewards(&self) -> Vec<f32> {
        let num_envs = self.envs.len();
        let max_a = self.max_agents;
        let mut rewards = vec![0.0f32; num_envs * max_a];

        for (i, env) in self.envs.iter().enumerate() {
            let r = env.get_rewards();
            for (j, &val) in r.iter().enumerate() {
                rewards[i * max_a + j] = val;
            }
        }

        rewards
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::InteractionRules;

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
            food_cap: None,
            interaction_rules: InteractionRules::default(),
        }
    }

    #[test]
    fn variable_agents_per_env() {
        let be = BatchedEnvironment::new(vec![3, 5, 2], test_config(), 42);
        assert_eq!(be.num_envs(), 3);
        assert_eq!(be.max_agents, 5);
        assert_eq!(be.envs[0].num_agents(), 3);
        assert_eq!(be.envs[1].num_agents(), 5);
        assert_eq!(be.envs[2].num_agents(), 2);
    }

    #[test]
    fn observe_shapes() {
        let be = BatchedEnvironment::new(vec![3, 5, 2], test_config(), 42);
        let features = be.observe();
        // features: 3 envs * 5 max_agents * 5 features
        assert_eq!(features.len(), 3 * 5 * 5);
    }

    #[test]
    fn observe_alive_feature() {
        let be = BatchedEnvironment::new(vec![2], test_config(), 42);
        let features = be.observe();
        let nf = BatchedEnvironment::NUM_FEATURES;
        // Agent 0 alive feature (index 4)
        assert!((features[0 * nf + 4] - 1.0).abs() < 1e-6);
        // Agent 1 alive feature
        assert!((features[1 * nf + 4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn padding_for_variable_envs() {
        let be = BatchedEnvironment::new(vec![1, 3], test_config(), 42);
        let features = be.observe();
        let nf = BatchedEnvironment::NUM_FEATURES;
        let max_a = be.max_agents; // 3
        // env0 has 1 agent → slot 0 alive=1, slots 1,2 alive=0
        assert!((features[(0 * max_a + 0) * nf + 4] - 1.0).abs() < 1e-6);
        assert!((features[(0 * max_a + 1) * nf + 4] - 0.0).abs() < 1e-6);
        assert!((features[(0 * max_a + 2) * nf + 4] - 0.0).abs() < 1e-6);
        // env1 has 3 agents → all alive=1
        assert!((features[(1 * max_a + 0) * nf + 4] - 1.0).abs() < 1e-6);
        assert!((features[(1 * max_a + 1) * nf + 4] - 1.0).abs() < 1e-6);
        assert!((features[(1 * max_a + 2) * nf + 4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn step_returns_rewards() {
        let mut be = BatchedEnvironment::new(vec![2, 3], test_config(), 42);
        let actions = vec![0.0f32; be.num_envs() * be.max_agents * 2];
        let rewards = be.step(&actions);
        assert_eq!(rewards.len(), 2 * 3); // num_envs * max_agents
    }

    #[test]
    fn envs_are_independent() {
        let mut be = BatchedEnvironment::new(vec![1, 1], test_config(), 42);
        // Move env0's agent right, env1's agent left
        let mut actions = vec![0.0f32; 2 * 1 * 2];
        actions[0] = 5.0; // env0 agent0 ax
        actions[2] = -5.0; // env1 agent0 ax

        let p0 = be.envs[0].agents[0].pos;
        let p1 = be.envs[1].agents[0].pos;

        be.step(&actions);

        // env0 agent moved right
        assert!(be.envs[0].agents[0].pos.x > p0.x);
        // env1 agent moved left
        assert!(be.envs[1].agents[0].pos.x < p1.x);
    }

    #[test]
    fn reset_all() {
        let mut be = BatchedEnvironment::new(vec![2, 3], test_config(), 42);
        let actions = vec![1.0f32; be.num_envs() * be.max_agents * 2];
        for _ in 0..10 {
            be.step(&actions);
        }

        be.reset();

        for env in &be.envs {
            assert_eq!(env.step_count, 0);
            for agent in &env.agents {
                assert!(agent.alive);
                assert!((agent.energy - 10.0).abs() < 1e-6);
            }
        }
    }
}
