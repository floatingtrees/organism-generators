/// Batched wrapper that runs multiple environments in parallel (logically).

use crate::environment::{Environment, NUM_ACTIONS, TOTAL_VIEW_CHANNELS};
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

    pub fn view_res(&self) -> usize {
        self.envs.first().map_or(32, |e| e.config.view_res)
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

    /// `actions` flat slice: [num_envs * max_agents * 3] — (ax, ay, view_delta) per agent.
    pub fn step(&mut self, actions: &[f32]) -> Vec<f32> {
        let max_a = self.max_agents;

        for (i, env) in self.envs.iter_mut().enumerate() {
            let n = env.num_agents();
            let env_actions: Vec<(f32, f32, f32)> = (0..n)
                .map(|j| {
                    let base = (i * max_a + j) * NUM_ACTIONS;
                    (actions[base], actions[base + 1], actions[base + 2])
                })
                .collect();
            env.step(&env_actions);
        }

        self.get_rewards()
    }

    // ------------------------------------------------------------------
    // Observe — egocentric views
    // ------------------------------------------------------------------

    /// Returns flat observation: [num_envs * max_agents * res * res * TOTAL_VIEW_CHANNELS].
    /// Padded agent slots are all zeros.
    pub fn observe(&self) -> Vec<f32> {
        let num_envs = self.envs.len();
        let max_a = self.max_agents;
        let res = self.view_res();
        let tc = TOTAL_VIEW_CHANNELS;
        let view_per_agent = res * res * tc;

        let mut obs = vec![0.0f32; num_envs * max_a * view_per_agent];

        for (i, env) in self.envs.iter().enumerate() {
            let views = env.get_views();
            let na = env.num_agents();
            for j in 0..na {
                let src_start = j * view_per_agent;
                let dst_start = (i * max_a + j) * view_per_agent;
                obs[dst_start..dst_start + view_per_agent]
                    .copy_from_slice(&views[src_start..src_start + view_per_agent]);
            }
        }

        obs
    }

    // ------------------------------------------------------------------
    // Alive mask
    // ------------------------------------------------------------------

    /// True if every real agent in every environment is dead.
    pub fn all_dead(&self) -> bool {
        self.envs
            .iter()
            .all(|env| env.agents.iter().all(|a| !a.alive))
    }

    /// Flat [num_envs * max_agents] — 1.0 alive, 0.0 dead/padded.
    pub fn get_alive_mask(&self) -> Vec<f32> {
        let num_envs = self.envs.len();
        let max_a = self.max_agents;
        let mut mask = vec![0.0f32; num_envs * max_a];

        for (i, env) in self.envs.iter().enumerate() {
            let am = env.get_alive_mask();
            for (j, &v) in am.iter().enumerate() {
                mask[i * max_a + j] = v;
            }
        }

        mask
    }

    // ------------------------------------------------------------------
    // Rewards
    // ------------------------------------------------------------------

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
            vision_cost: 0.0,
            view_res: 8,
            initial_view_size: 2.0,
            interaction_rules: InteractionRules::default(),
        }
    }

    #[test]
    fn variable_agents_per_env() {
        let be = BatchedEnvironment::new(vec![3, 5, 2], test_config(), 42);
        assert_eq!(be.num_envs(), 3);
        assert_eq!(be.max_agents, 5);
    }

    #[test]
    fn observe_shape() {
        let be = BatchedEnvironment::new(vec![3, 5, 2], test_config(), 42);
        let obs = be.observe();
        let res = be.view_res();
        // 3 envs * 5 max_agents * 8 * 8 * 16
        assert_eq!(obs.len(), 3 * 5 * res * res * TOTAL_VIEW_CHANNELS);
    }

    #[test]
    fn alive_mask_shape() {
        let be = BatchedEnvironment::new(vec![3, 5, 2], test_config(), 42);
        let mask = be.get_alive_mask();
        assert_eq!(mask.len(), 3 * 5);
    }

    #[test]
    fn padding_mask() {
        let be = BatchedEnvironment::new(vec![1, 3], test_config(), 42);
        let mask = be.get_alive_mask();
        // env0: 1 agent alive, 2 padded
        assert!((mask[0] - 1.0).abs() < 1e-6);
        assert!((mask[1]).abs() < 1e-6);
        assert!((mask[2]).abs() < 1e-6);
        // env1: 3 agents alive
        assert!((mask[3] - 1.0).abs() < 1e-6);
        assert!((mask[4] - 1.0).abs() < 1e-6);
        assert!((mask[5] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn step_returns_rewards() {
        let mut be = BatchedEnvironment::new(vec![2, 3], test_config(), 42);
        let actions = vec![0.0f32; be.num_envs() * be.max_agents * NUM_ACTIONS];
        let rewards = be.step(&actions);
        assert_eq!(rewards.len(), 2 * 3);
    }

    #[test]
    fn envs_are_independent() {
        let mut be = BatchedEnvironment::new(vec![1, 1], test_config(), 42);
        let mut actions = vec![0.0f32; 2 * 1 * NUM_ACTIONS];
        actions[0] = 5.0; // env0 agent0 ax
        actions[3] = -5.0; // env1 agent0 ax

        let p0 = be.envs[0].agents[0].pos;
        let p1 = be.envs[1].agents[0].pos;
        be.step(&actions);

        assert!(be.envs[0].agents[0].pos.x > p0.x);
        assert!(be.envs[1].agents[0].pos.x < p1.x);
    }

    #[test]
    fn reset_all() {
        let mut be = BatchedEnvironment::new(vec![2, 3], test_config(), 42);
        let actions = vec![1.0f32; be.num_envs() * be.max_agents * NUM_ACTIONS];
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
