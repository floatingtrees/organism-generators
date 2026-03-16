/// Rendering an environment to an RGB image buffer and saving as PNG.

use crate::environment::Environment;
use crate::modules::ModuleType;
use image::{ImageBuffer, Rgb};
use std::path::Path;

/// Render the environment to a flat RGB buffer (HWC layout).
///
/// Returns `(buffer, img_width, img_height)`.
pub fn render_environment(
    env: &Environment,
    pixels_per_unit: f32,
) -> (Vec<u8>, usize, usize) {
    let img_w = (env.config.width * pixels_per_unit).ceil() as usize;
    let img_h = (env.config.height * pixels_per_unit).ceil() as usize;
    let mut buf = vec![20u8; img_w * img_h * 3]; // Dark background

    // Background grid lines (subtle)
    for gy in 0..env.config.height as usize {
        let py = (gy as f32 * pixels_per_unit) as usize;
        if py < img_h {
            for px in 0..img_w {
                set_pixel(&mut buf, img_w, px, py, [35, 35, 35]);
            }
        }
    }
    for gx in 0..env.config.width as usize {
        let px = (gx as f32 * pixels_per_unit) as usize;
        if px < img_w {
            for py in 0..img_h {
                set_pixel(&mut buf, img_w, px, py, [35, 35, 35]);
            }
        }
    }

    // Draw food — bright green filled circles
    for food in &env.foods {
        let cx = food.pos.x * pixels_per_unit;
        let cy = food.pos.y * pixels_per_unit;
        let r = env.config.object_radius * pixels_per_unit;
        draw_circle(&mut buf, img_w, img_h, cx, cy, r.max(2.0), [50, 220, 50]);
    }

    // Draw obstacles — gray filled circles
    for obs in &env.obstacles {
        let cx = obs.pos.x * pixels_per_unit;
        let cy = obs.pos.y * pixels_per_unit;
        let r = obs.radius * pixels_per_unit;
        draw_circle(&mut buf, img_w, img_h, cx, cy, r.max(2.0), [160, 160, 160]);
    }

    // Draw agents — distinct hue per agent
    for agent in &env.agents {
        let color = if agent.alive {
            hsv_to_rgb((agent.id as f32 * 137.508) % 360.0, 0.85, 1.0)
        } else {
            [80, 30, 30] // dim red
        };
        let cx = agent.pos.x * pixels_per_unit;
        let cy = agent.pos.y * pixels_per_unit;
        let r = env.config.object_radius * pixels_per_unit;
        let circle_r = r.max(3.0);
        draw_circle(&mut buf, img_w, img_h, cx, cy, circle_r, color);

        // Draw a small marker in the centre so agents are distinguishable from food
        draw_circle(&mut buf, img_w, img_h, cx, cy, (circle_r * 0.35).max(1.0), [255, 255, 255]);

        // Draw dashed view-size circle
        if agent.alive && agent.view_size > 0.0 {
            let view_r = agent.view_size * pixels_per_unit;
            let dash_color = [color[0] / 2, color[1] / 2, color[2] / 2];
            draw_dashed_ring(&mut buf, img_w, img_h, cx, cy, view_r, dash_color, 6.0, 4.0);
        }

        // Draw modules for this agent
        let mg = &env.module_graphs[agent.id];
        let dim_color = [color[0] / 2 + 30, color[1] / 2 + 30, color[2] / 2 + 30];
        for m in &mg.modules {
            if !m.alive {
                continue;
            }
            match m.module_type {
                ModuleType::Segment => {
                    // Draw as filled rectangle along the segment line
                    let sx = m.world_pos.x * pixels_per_unit;
                    let sy = m.world_pos.y * pixels_per_unit;
                    let ex = m.world_end.x * pixels_per_unit;
                    let ey = m.world_end.y * pixels_per_unit;
                    draw_thick_line(&mut buf, img_w, img_h, sx, sy, ex, ey,
                        (0.2 * pixels_per_unit).max(2.0), dim_color);
                }
                ModuleType::Or => {
                    // OR gate: curved D-shape (simplified as circle)
                    let px = m.world_pos.x * pixels_per_unit;
                    let py = m.world_pos.y * pixels_per_unit;
                    let r = (0.15 * pixels_per_unit).max(3.0);
                    draw_circle(&mut buf, img_w, img_h, px, py, r, [200, 200, 50]);
                }
                ModuleType::And => {
                    // AND gate: D-shape (simplified as filled square)
                    let px = m.world_pos.x * pixels_per_unit;
                    let py = m.world_pos.y * pixels_per_unit;
                    let r = (0.15 * pixels_per_unit).max(3.0);
                    draw_rect(&mut buf, img_w, img_h, px, py, r, [50, 200, 200]);
                }
                ModuleType::Xor => {
                    // XOR gate: diamond
                    let px = m.world_pos.x * pixels_per_unit;
                    let py = m.world_pos.y * pixels_per_unit;
                    let r = (0.15 * pixels_per_unit).max(3.0);
                    draw_diamond(&mut buf, img_w, img_h, px, py, r, [200, 50, 200]);
                }
                ModuleType::Thruster => {
                    // Arrow pointing in thrust direction
                    let px = m.world_pos.x * pixels_per_unit;
                    let py = m.world_pos.y * pixels_per_unit;
                    let r = (0.12 * pixels_per_unit).max(3.0);
                    draw_triangle(&mut buf, img_w, img_h, px, py, r,
                        m.rotation + env.agents[agent.id].rotation, [255, 100, 50]);
                }
                ModuleType::Mouth => {
                    // Star shape
                    let px = m.world_pos.x * pixels_per_unit;
                    let py = m.world_pos.y * pixels_per_unit;
                    let r = (0.15 * pixels_per_unit).max(3.0);
                    draw_circle(&mut buf, img_w, img_h, px, py, r, [50, 255, 100]);
                    draw_circle(&mut buf, img_w, img_h, px, py, r * 0.5, [255, 255, 255]);
                }
            }
        }
    }

    // Draw border
    for px in 0..img_w {
        set_pixel(&mut buf, img_w, px, 0, [100, 100, 100]);
        set_pixel(&mut buf, img_w, px, img_h - 1, [100, 100, 100]);
    }
    for py in 0..img_h {
        set_pixel(&mut buf, img_w, 0, py, [100, 100, 100]);
        set_pixel(&mut buf, img_w, img_w - 1, py, [100, 100, 100]);
    }

    (buf, img_w, img_h)
}

/// Render the environment and save as a PNG file.
pub fn save_environment_png(
    env: &Environment,
    pixels_per_unit: f32,
    path: &Path,
) -> Result<(), String> {
    let (buf, w, h) = render_environment(env, pixels_per_unit);
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(w as u32, h as u32, buf)
            .ok_or_else(|| "failed to create image buffer".to_string())?;
    img.save(path).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn set_pixel(buf: &mut [u8], img_w: usize, x: usize, y: usize, color: [u8; 3]) {
    let idx = (y * img_w + x) * 3;
    if idx + 2 < buf.len() {
        buf[idx] = color[0];
        buf[idx + 1] = color[1];
        buf[idx + 2] = color[2];
    }
}

fn draw_circle(
    buf: &mut [u8],
    img_w: usize,
    img_h: usize,
    cx: f32,
    cy: f32,
    radius: f32,
    color: [u8; 3],
) {
    let r = radius.max(1.0);
    let x_min = ((cx - r).floor() as i32).max(0) as usize;
    let x_max = ((cx + r).ceil() as i32).min(img_w as i32 - 1).max(0) as usize;
    let y_min = ((cy - r).floor() as i32).max(0) as usize;
    let y_max = ((cy + r).ceil() as i32).min(img_h as i32 - 1).max(0) as usize;

    for y in y_min..=y_max {
        for x in x_min..=x_max {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            if dx * dx + dy * dy <= r * r {
                set_pixel(buf, img_w, x, y, color);
            }
        }
    }
}

/// Draw a dashed ring (circle outline with gaps).
/// `dash_len` and `gap_len` are in pixels along the circumference.
fn draw_dashed_ring(
    buf: &mut [u8],
    img_w: usize,
    img_h: usize,
    cx: f32,
    cy: f32,
    radius: f32,
    color: [u8; 3],
    dash_len: f32,
    gap_len: f32,
) {
    if radius < 1.0 {
        return;
    }
    let circumference = 2.0 * std::f32::consts::PI * radius;
    let total_pattern = dash_len + gap_len;
    // Walk around the circle in 1-pixel steps
    let num_steps = (circumference * 2.0).ceil() as usize;
    let thickness = 1.5f32;

    for i in 0..num_steps {
        let arc_pos = (i as f32 / num_steps as f32) * circumference;
        // Skip gap portions
        if (arc_pos % total_pattern) >= dash_len {
            continue;
        }
        let angle = (i as f32 / num_steps as f32) * 2.0 * std::f32::consts::PI;
        let px = cx + radius * angle.cos();
        let py = cy + radius * angle.sin();

        // Draw a small dot for thickness
        let x_min = ((px - thickness).floor() as i32).max(0) as usize;
        let x_max = ((px + thickness).ceil() as i32).min(img_w as i32 - 1).max(0) as usize;
        let y_min = ((py - thickness).floor() as i32).max(0) as usize;
        let y_max = ((py + thickness).ceil() as i32).min(img_h as i32 - 1).max(0) as usize;

        for y in y_min..=y_max {
            for x in x_min..=x_max {
                let dx = x as f32 - px;
                let dy = y as f32 - py;
                if dx * dx + dy * dy <= thickness * thickness {
                    set_pixel(buf, img_w, x, y, color);
                }
            }
        }
    }
}

fn draw_thick_line(
    buf: &mut [u8], img_w: usize, img_h: usize,
    x1: f32, y1: f32, x2: f32, y2: f32, thickness: f32, color: [u8; 3],
) {
    let half = thickness * 0.5;
    let min_x = ((x1.min(x2) - half).floor() as i32).max(0) as usize;
    let max_x = ((x1.max(x2) + half).ceil() as i32).min(img_w as i32 - 1).max(0) as usize;
    let min_y = ((y1.min(y2) - half).floor() as i32).max(0) as usize;
    let max_y = ((y1.max(y2) + half).ceil() as i32).min(img_h as i32 - 1).max(0) as usize;

    let dx = x2 - x1;
    let dy = y2 - y1;
    let len_sq = dx * dx + dy * dy;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let px = x as f32;
            let py = y as f32;
            let dist = if len_sq < 1e-6 {
                ((px - x1) * (px - x1) + (py - y1) * (py - y1)).sqrt()
            } else {
                let t = ((px - x1) * dx + (py - y1) * dy) / len_sq;
                let t = t.clamp(0.0, 1.0);
                let cx = x1 + t * dx;
                let cy = y1 + t * dy;
                ((px - cx) * (px - cx) + (py - cy) * (py - cy)).sqrt()
            };
            if dist <= half {
                set_pixel(buf, img_w, x, y, color);
            }
        }
    }
}

fn draw_rect(
    buf: &mut [u8], img_w: usize, img_h: usize,
    cx: f32, cy: f32, half_size: f32, color: [u8; 3],
) {
    let min_x = ((cx - half_size).floor() as i32).max(0) as usize;
    let max_x = ((cx + half_size).ceil() as i32).min(img_w as i32 - 1).max(0) as usize;
    let min_y = ((cy - half_size).floor() as i32).max(0) as usize;
    let max_y = ((cy + half_size).ceil() as i32).min(img_h as i32 - 1).max(0) as usize;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            set_pixel(buf, img_w, x, y, color);
        }
    }
}

fn draw_diamond(
    buf: &mut [u8], img_w: usize, img_h: usize,
    cx: f32, cy: f32, radius: f32, color: [u8; 3],
) {
    let r = radius.max(1.0);
    let min_x = ((cx - r).floor() as i32).max(0) as usize;
    let max_x = ((cx + r).ceil() as i32).min(img_w as i32 - 1).max(0) as usize;
    let min_y = ((cy - r).floor() as i32).max(0) as usize;
    let max_y = ((cy + r).ceil() as i32).min(img_h as i32 - 1).max(0) as usize;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let dx = (x as f32 - cx).abs();
            let dy = (y as f32 - cy).abs();
            if dx + dy <= r {
                set_pixel(buf, img_w, x, y, color);
            }
        }
    }
}

fn draw_triangle(
    buf: &mut [u8], img_w: usize, img_h: usize,
    cx: f32, cy: f32, radius: f32, rotation: f32, color: [u8; 3],
) {
    // Triangle pointing in rotation direction
    let r = radius.max(2.0);
    let tip_x = cx + r * rotation.cos();
    let tip_y = cy + r * rotation.sin();
    let back_angle = std::f32::consts::PI * 0.8;
    let l_x = cx + r * 0.6 * (rotation + back_angle).cos();
    let l_y = cy + r * 0.6 * (rotation + back_angle).sin();
    let r_x = cx + r * 0.6 * (rotation - back_angle).cos();
    let r_y = cy + r * 0.6 * (rotation - back_angle).sin();

    // Fill triangle using bounding box + barycentric test
    let min_x = (tip_x.min(l_x).min(r_x).floor() as i32).max(0) as usize;
    let max_x = (tip_x.max(l_x).max(r_x).ceil() as i32).min(img_w as i32 - 1).max(0) as usize;
    let min_y = (tip_y.min(l_y).min(r_y).floor() as i32).max(0) as usize;
    let max_y = (tip_y.max(l_y).max(r_y).ceil() as i32).min(img_h as i32 - 1).max(0) as usize;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let px = x as f32;
            let py = y as f32;
            if point_in_triangle(px, py, tip_x, tip_y, l_x, l_y, r_x, r_y) {
                set_pixel(buf, img_w, x, y, color);
            }
        }
    }
}

fn point_in_triangle(px: f32, py: f32, x1: f32, y1: f32, x2: f32, y2: f32, x3: f32, y3: f32) -> bool {
    let d1 = (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2);
    let d2 = (px - x3) * (y2 - y3) - (x2 - x3) * (py - y3);
    let d3 = (px - x1) * (y3 - y1) - (x3 - x1) * (py - y1);
    let has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
    let has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);
    !(has_neg && has_pos)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [u8; 3] {
    let c = v * s;
    let hp = h / 60.0;
    let x = c * (1.0 - (hp % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = if hp < 1.0 {
        (c, x, 0.0)
    } else if hp < 2.0 {
        (x, c, 0.0)
    } else if hp < 3.0 {
        (0.0, c, x)
    } else if hp < 4.0 {
        (0.0, x, c)
    } else if hp < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    [
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    fn test_config() -> EnvironmentConfig {
        EnvironmentConfig {
            width: 5.0,
            height: 5.0,
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
            vision_cost: 0.0,
            view_res: 8,
            initial_view_size: 2.0,
            min_view_size: 0.0,
            energy_decay_rate: 1.0,
            interaction_rules: InteractionRules::default(),
        }
    }

    #[test]
    fn render_returns_correct_dimensions() {
        let env = Environment::new(2, test_config(), 42);
        let ppu = 20.0;
        let (buf, w, h) = render_environment(&env, ppu);
        assert_eq!(w, 100); // 5 * 20
        assert_eq!(h, 100);
        assert_eq!(buf.len(), w * h * 3);
    }

    #[test]
    fn render_has_nonzero_pixels() {
        let mut env = Environment::new(1, test_config(), 42);
        env.foods.push(Food {
            pos: Vec2::new(2.5, 2.5),
        });
        let (buf, _, _) = render_environment(&env, 20.0);
        // Should have some non-dark pixels for the agent and food
        let bright_count = buf.chunks(3).filter(|px| px[0] > 40 || px[1] > 40 || px[2] > 40).count();
        assert!(bright_count > 0);
    }

    #[test]
    fn hsv_red() {
        let c = hsv_to_rgb(0.0, 1.0, 1.0);
        assert_eq!(c, [255, 0, 0]);
    }

    #[test]
    fn hsv_green() {
        let c = hsv_to_rgb(120.0, 1.0, 1.0);
        assert_eq!(c, [0, 255, 0]);
    }

    #[test]
    fn hsv_blue() {
        let c = hsv_to_rgb(240.0, 1.0, 1.0);
        assert_eq!(c, [0, 0, 255]);
    }
}
