mod batched_env;
mod environment;
mod rendering;
mod spatial_hash;
mod types;

use ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::batched_env::BatchedEnvironment;
use crate::rendering::{render_environment, save_environment_png};
use crate::types::*;

use std::path::Path;

#[pyclass]
struct EvolutionEnv {
    inner: BatchedEnvironment,
}

#[pymethods]
impl EvolutionEnv {
    /// Create a new batched evolutionary environment.
    ///
    /// Config keys:
    ///   num_organisms: int | list[int]   — agents per environment
    ///   height: float
    ///   width: float
    ///   food_spawn_rate: float           — expected food items spawned per step
    ///   num_copies: int                  — number of parallel environments
    ///   dt: float                        — simulation timestep (default 0.5)
    ///   energy_loss: float               — fraction of energy lost on wall bounce (default 0.1)
    ///   object_radius: float             — collision radius (default 0.1)
    ///   num_obstacles: int               — initial obstacles per env (default 0)
    ///   obstacle_weight: float           — mass of obstacles (default 5.0)
    ///   seed: int                        — base RNG seed (default 42)
    ///   rules: dict                      — interaction rule toggles (optional)
    #[staticmethod]
    fn initialize(config: &Bound<'_, PyDict>) -> PyResult<Self> {
        // --- required ---
        let num_copies: usize = extract_required(config, "num_copies")?;
        let height: f32 = extract_required(config, "height")?;
        let width: f32 = extract_required(config, "width")?;
        let food_spawn_rate: f32 = extract_required(config, "food_spawn_rate")?;

        // --- num_organisms: int | list[int] ---
        let num_organisms_obj = config
            .get_item("num_organisms")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("num_organisms"))?;

        let num_agents_per_env: Vec<usize> =
            if let Ok(scalar) = num_organisms_obj.extract::<usize>() {
                vec![scalar; num_copies]
            } else {
                let list: Vec<usize> = num_organisms_obj.extract()?;
                if list.len() != num_copies {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "num_organisms list length ({}) != num_copies ({})",
                        list.len(),
                        num_copies
                    )));
                }
                list
            };

        // --- optional with defaults ---
        let dt: f32 = extract_or(config, "dt", 0.5)?;
        let energy_loss: f32 = extract_or(config, "energy_loss", 0.1)?;
        let object_radius: f32 = extract_or(config, "object_radius", 0.1)?;
        let num_obstacles: usize = extract_or(config, "num_obstacles", 0)?;
        let obstacle_weight: f32 = extract_or(config, "obstacle_weight", 5.0)?;
        let seed: u64 = extract_or(config, "seed", 42)?;
        let food_cap: Option<usize> = match config.get_item("food_cap")? {
            Some(val) => Some(val.extract()?),
            None => None,
        };

        // --- interaction rules ---
        let mut rules = InteractionRules::default();
        if let Some(rules_dict) = config.get_item("rules")? {
            let rd: &Bound<'_, PyDict> = rules_dict.downcast()?;
            if let Some(v) = rd.get_item("wall_bounce")? {
                rules.wall_bounce = v.extract()?;
            }
            if let Some(v) = rd.get_item("food_collection")? {
                rules.food_collection = v.extract()?;
            }
            if let Some(v) = rd.get_item("obstacle_collision")? {
                rules.obstacle_collision = v.extract()?;
            }
            if let Some(v) = rd.get_item("agent_collision")? {
                rules.agent_collision = v.extract()?;
            }
        }

        let env_config = EnvironmentConfig {
            width,
            height,
            dt,
            food_spawn_rate,
            energy_loss_wall: energy_loss,
            object_radius,
            num_initial_obstacles: num_obstacles,
            obstacle_weight,
            dead_steps_threshold: EnvironmentConfig::dead_threshold_from_seconds(10.0, dt),
            food_cap,
            interaction_rules: rules,
        };

        let inner = BatchedEnvironment::new(num_agents_per_env, env_config, seed);

        Ok(Self { inner })
    }

    /// Reset all environments to their initial state.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Advance one simulation step.
    ///
    /// Args:
    ///     actions: numpy array of shape (num_envs, max_agents, 2) — acceleration vectors.
    ///
    /// Returns:
    ///     numpy array of shape (num_envs, max_agents) — current energy (reward) per agent.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let shape = actions.shape();
        let ne = self.inner.num_envs();
        let ma = self.inner.max_agents;

        if shape.len() != 3 || shape[0] != ne || shape[1] != ma || shape[2] != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "actions must have shape ({}, {}, 2), got {:?}",
                ne, ma, shape
            )));
        }

        let slice = actions
            .as_slice()
            .map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "actions array must be C-contiguous",
                )
            })?;

        let rewards = self.inner.step(slice);

        let arr = ArrayD::from_shape_vec(IxDyn(&[ne, ma]), rewards)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    /// Get observations for all environments.
    ///
    /// Returns:
    ///     numpy (num_envs, max_agents, 5) — features: [x, y, vx, vy, alive]
    ///     Padded agent slots are all zeros (alive=0).
    fn observe<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let ne = self.inner.num_envs();
        let ma = self.inner.max_agents;
        let nf = BatchedEnvironment::NUM_FEATURES;

        let features = self.inner.observe();

        let arr = ArrayD::from_shape_vec(IxDyn(&[ne, ma, nf]), features)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    /// Render a single environment and save as a PNG file.
    ///
    /// Args:
    ///     filepath: path to save the PNG image
    ///     env_index: which environment to render (default 0)
    ///     pixels_per_unit: rendering resolution (default 20.0)
    #[pyo3(signature = (filepath, env_index=None, pixels_per_unit=None))]
    fn render(
        &self,
        filepath: &str,
        env_index: Option<usize>,
        pixels_per_unit: Option<f32>,
    ) -> PyResult<()> {
        let idx = env_index.unwrap_or(0);
        let ppu = pixels_per_unit.unwrap_or(20.0);

        if idx >= self.inner.num_envs() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "env_index {} out of range (num_envs = {})",
                idx,
                self.inner.num_envs()
            )));
        }

        save_environment_png(&self.inner.envs[idx], ppu, Path::new(filepath))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))
    }

    /// Render a single environment and return as a numpy array.
    ///
    /// Returns:
    ///     numpy uint8 array of shape (H, W, 3)
    #[pyo3(signature = (env_index=None, pixels_per_unit=None))]
    fn render_array<'py>(
        &self,
        py: Python<'py>,
        env_index: Option<usize>,
        pixels_per_unit: Option<f32>,
    ) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
        let idx = env_index.unwrap_or(0);
        let ppu = pixels_per_unit.unwrap_or(20.0);

        if idx >= self.inner.num_envs() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "env_index {} out of range (num_envs = {})",
                idx,
                self.inner.num_envs()
            )));
        }

        let (buf, w, h) = render_environment(&self.inner.envs[idx], ppu);
        let arr = ArrayD::from_shape_vec(IxDyn(&[h, w, 3]), buf)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    /// Number of parallel environments.
    #[getter]
    fn num_envs(&self) -> usize {
        self.inner.num_envs()
    }

    /// Maximum agents across all environments (pad dimension).
    #[getter]
    fn max_agents(&self) -> usize {
        self.inner.max_agents
    }
}

// ---------------------------------------------------------------------------
// Config extraction helpers
// ---------------------------------------------------------------------------

fn extract_required<'py, T: FromPyObject<'py>>(
    dict: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<T> {
    dict.get_item(key)?
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(key.to_string()))?
        .extract()
}

fn extract_or<'py, T: FromPyObject<'py>>(
    dict: &Bound<'py, PyDict>,
    key: &str,
    default: T,
) -> PyResult<T> {
    match dict.get_item(key)? {
        Some(val) => val.extract(),
        None => Ok(default),
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn organism_env(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EvolutionEnv>()?;
    Ok(())
}
