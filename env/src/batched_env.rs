/// Batched wrapper that runs multiple environments in parallel via rayon.

use crate::environment::{Environment, NUM_ACTIONS, NUM_SCALAR_FEATURES, TOTAL_VIEW_CHANNELS};
use crate::types::EnvironmentConfig;
use rayon::prelude::*;

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
    // Reset (parallel)
    // ------------------------------------------------------------------

    pub fn reset(&mut self) {
        self.envs.par_iter_mut().for_each(|env| env.reset());
        self.max_agents = self.envs.iter().map(|e| e.num_agents()).max().unwrap_or(0);
    }

    // ------------------------------------------------------------------
    // Step (parallel)
    // ------------------------------------------------------------------

    /// `actions` flat slice: [num_envs * max_agents * 3].
    pub fn step(&mut self, actions: &[f32]) -> Vec<f32> {
        let max_a = self.max_agents;

        // Step all environments in parallel
        self.envs
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, env)| {
                let n = env.num_agents();
                let env_actions: Vec<(f32, f32, f32)> = (0..n)
                    .map(|j| {
                        let base = (i * max_a + j) * NUM_ACTIONS;
                        (actions[base], actions[base + 1], actions[base + 2])
                    })
                    .collect();
                env.step(&env_actions);
            });

        self.get_rewards()
    }

    // ------------------------------------------------------------------
    // Observe (parallel)
    // ------------------------------------------------------------------

    pub fn observe(&self) -> Vec<f32> {
        let max_a = self.max_agents;
        let res = self.view_res();
        let tc = TOTAL_VIEW_CHANNELS;
        let view_per_agent = res * res * tc;
        let stride = max_a * view_per_agent;

        // Each env writes into its own non-overlapping slice
        let mut obs = vec![0.0f32; self.envs.len() * stride];

        // Split into per-env chunks and fill in parallel
        obs.par_chunks_mut(stride)
            .zip(self.envs.par_iter())
            .for_each(|(chunk, env)| {
                let views = env.get_views();
                let na = env.num_agents();
                for j in 0..na {
                    let src_start = j * view_per_agent;
                    let dst_start = j * view_per_agent;
                    chunk[dst_start..dst_start + view_per_agent]
                        .copy_from_slice(&views[src_start..src_start + view_per_agent]);
                }
                // Padded slots stay zero (already initialized)
            });

        obs
    }

    // ------------------------------------------------------------------
    // Alive mask
    // ------------------------------------------------------------------

    pub fn all_dead(&self) -> bool {
        self.envs
            .par_iter()
            .all(|env| env.agents.iter().all(|a| !a.alive))
    }

    pub fn get_alive_mask(&self) -> Vec<f32> {
        let max_a = self.max_agents;
        let mut mask = vec![0.0f32; self.envs.len() * max_a];

        mask.par_chunks_mut(max_a)
            .zip(self.envs.par_iter())
            .for_each(|(chunk, env)| {
                let am = env.get_alive_mask();
                for (j, &v) in am.iter().enumerate() {
                    chunk[j] = v;
                }
            });

        mask
    }

    // ------------------------------------------------------------------
    // Agent states (scalar side channel)
    // ------------------------------------------------------------------

    /// Flat [num_envs * max_agents * NUM_SCALAR_FEATURES].
    pub fn get_agent_states(&self) -> Vec<f32> {
        let max_a = self.max_agents;
        let ns = NUM_SCALAR_FEATURES;
        let stride = max_a * ns;
        let mut states = vec![0.0f32; self.envs.len() * stride];

        states
            .par_chunks_mut(stride)
            .zip(self.envs.par_iter())
            .for_each(|(chunk, env)| {
                let s = env.get_agent_states();
                let na = env.num_agents();
                for j in 0..na {
                    let src = j * ns;
                    let dst = j * ns;
                    chunk[dst..dst + ns].copy_from_slice(&s[src..src + ns]);
                }
            });

        states
    }

    // ------------------------------------------------------------------
    // Rewards
    // ------------------------------------------------------------------

    pub fn get_rewards(&self) -> Vec<f32> {
        let max_a = self.max_agents;
        let mut rewards = vec![0.0f32; self.envs.len() * max_a];

        rewards
            .par_chunks_mut(max_a)
            .zip(self.envs.par_iter())
            .for_each(|(chunk, env)| {
                let r = env.get_rewards();
                for (j, &val) in r.iter().enumerate() {
                    chunk[j] = val;
                }
            });

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
            obstacle_radius: 0.1,
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
        assert!((mask[0] - 1.0).abs() < 1e-6);
        assert!((mask[1]).abs() < 1e-6);
        assert!((mask[2]).abs() < 1e-6);
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
        actions[0] = 5.0;  // env0 ax
        actions[3] = -5.0; // env1 ax

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

    #[test]
    fn all_dead_detection() {
        let mut cfg = test_config();
        cfg.dead_steps_threshold = 1;
        let mut be = BatchedEnvironment::new(vec![1, 1], cfg, 42);
        assert!(!be.all_dead());

        // Kill all agents
        for env in &mut be.envs {
            for agent in &mut env.agents {
                agent.alive = false;
            }
        }
        assert!(be.all_dead());
    }
}
