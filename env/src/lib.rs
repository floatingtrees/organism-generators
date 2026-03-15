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
use crate::environment::{NUM_ACTIONS, NUM_SCALAR_FEATURES, TOTAL_VIEW_CHANNELS};
use crate::rendering::{render_environment, save_environment_png};
use crate::types::*;

use rand::Rng;
use std::path::Path;

#[pyclass]
struct EvolutionEnv {
    inner: BatchedEnvironment,
}

#[pymethods]
impl EvolutionEnv {
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
        let seed: u64 = match config.get_item("seed")? {
            Some(val) => val.extract()?,
            None => rand::thread_rng().gen(),
        };
        let food_cap: Option<usize> = match config.get_item("food_cap")? {
            Some(val) => Some(val.extract()?),
            None => None,
        };
        let vision_cost: f32 = extract_or(config, "vision_cost", 0.1)?;
        let view_res: usize = extract_or(config, "view_res", 32)?;
        let initial_view_size: f32 = extract_or(config, "initial_view_size", 2.0)?;
        let obstacle_radius: f32 = extract_or(config, "obstacle_radius", object_radius)?;

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
            if let Some(v) = rd.get_item("obstacle_obstacle_collision")? {
                rules.obstacle_obstacle_collision = v.extract()?;
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
            obstacle_radius,
            dead_steps_threshold: EnvironmentConfig::dead_threshold_from_seconds(10.0, dt),
            food_cap,
            vision_cost,
            view_res,
            initial_view_size,
            interaction_rules: rules,
        };

        let inner = BatchedEnvironment::new(num_agents_per_env, env_config, seed);

        Ok(Self { inner })
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Step with actions of shape (num_envs, max_agents, 3) — (ax, ay, view_delta).
    /// Returns rewards of shape (num_envs, max_agents).
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArrayDyn<'py, f32>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let shape = actions.shape();
        let ne = self.inner.num_envs();
        let ma = self.inner.max_agents;

        if shape.len() != 3 || shape[0] != ne || shape[1] != ma || shape[2] != NUM_ACTIONS {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "actions must have shape ({}, {}, {}), got {:?}",
                ne, ma, NUM_ACTIONS, shape
            )));
        }

        let slice = actions.as_slice().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("actions array must be C-contiguous")
        })?;

        let rewards = self.inner.step(slice);

        let arr = ArrayD::from_shape_vec(IxDyn(&[ne, ma]), rewards)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    /// Returns observations of shape (num_envs, max_agents, view_res, view_res, total_channels).
    /// total_channels = 16 (4 object channels × 4 frames).
    fn observe<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let ne = self.inner.num_envs();
        let ma = self.inner.max_agents;
        let res = self.inner.view_res();
        let tc = TOTAL_VIEW_CHANNELS;

        let obs = self.inner.observe();

        let arr = ArrayD::from_shape_vec(IxDyn(&[ne, ma, res, res, tc]), obs)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    /// True if all agents in all environments are dead.
    fn all_dead(&self) -> bool {
        self.inner.all_dead()
    }

    /// Returns alive mask of shape (num_envs, max_agents).
    fn alive_mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let ne = self.inner.num_envs();
        let ma = self.inner.max_agents;
        let mask = self.inner.get_alive_mask();

        let arr = ArrayD::from_shape_vec(IxDyn(&[ne, ma]), mask)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    /// Returns agent scalar states: (num_envs, max_agents, 4) — [energy, vx, vy, view_size].
    fn agent_states<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let ne = self.inner.num_envs();
        let ma = self.inner.max_agents;
        let ns = NUM_SCALAR_FEATURES;
        let data = self.inner.get_agent_states();
        let arr = ArrayD::from_shape_vec(IxDyn(&[ne, ma, ns]), data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

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

    #[getter]
    fn num_envs(&self) -> usize {
        self.inner.num_envs()
    }

    #[getter]
    fn max_agents(&self) -> usize {
        self.inner.max_agents
    }

    #[getter]
    fn view_res(&self) -> usize {
        self.inner.view_res()
    }

    #[getter]
    fn num_actions(&self) -> usize {
        NUM_ACTIONS
    }

    #[getter]
    fn total_channels(&self) -> usize {
        TOTAL_VIEW_CHANNELS
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
