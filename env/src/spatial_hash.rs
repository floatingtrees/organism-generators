/// Spatial hash grid for efficient proximity queries.
///
/// Divides the world into fixed-size cells and buckets objects by position.
/// Queries only check neighboring cells, giving O(1) amortized lookups
/// instead of O(N) brute force.

use crate::types::Vec2;

pub struct SpatialHash {
    cell_size: f32,
    inv_cell_size: f32,
    cols: usize,
    rows: usize,
    cells: Vec<Vec<usize>>,
}

impl SpatialHash {
    pub fn new(width: f32, height: f32, cell_size: f32) -> Self {
        let cols = (width / cell_size).ceil() as usize;
        let rows = (height / cell_size).ceil() as usize;
        let cells = vec![Vec::new(); cols * rows];
        Self {
            cell_size,
            inv_cell_size: 1.0 / cell_size,
            cols,
            rows,
            cells,
        }
    }

    pub fn clear(&mut self) {
        for cell in &mut self.cells {
            cell.clear();
        }
    }

    fn cell_coords(&self, x: f32, y: f32) -> (usize, usize) {
        let col = (x * self.inv_cell_size).floor().max(0.0) as usize;
        let row = (y * self.inv_cell_size).floor().max(0.0) as usize;
        (col.min(self.cols - 1), row.min(self.rows - 1))
    }

    /// Insert a point-sized object into its containing cell.
    pub fn insert(&mut self, idx: usize, pos: Vec2) {
        let (col, row) = self.cell_coords(pos.x, pos.y);
        self.cells[row * self.cols + col].push(idx);
    }

    /// Insert an object with radius into ALL cells it overlaps.
    pub fn insert_with_radius(&mut self, idx: usize, pos: Vec2, radius: f32) {
        let min_col = ((pos.x - radius) * self.inv_cell_size).floor().max(0.0) as usize;
        let max_col = ((pos.x + radius) * self.inv_cell_size).floor() as usize;
        let max_col = max_col.min(self.cols - 1);
        let min_row = ((pos.y - radius) * self.inv_cell_size).floor().max(0.0) as usize;
        let max_row = ((pos.y + radius) * self.inv_cell_size).floor() as usize;
        let max_row = max_row.min(self.rows - 1);

        for row in min_row..=max_row {
            let base = row * self.cols;
            for col in min_col..=max_col {
                self.cells[base + col].push(idx);
            }
        }
    }

    /// Populate the hash from a slice of positions (point-sized).
    pub fn build(&mut self, positions: &[Vec2]) {
        self.clear();
        for (i, pos) in positions.iter().enumerate() {
            self.insert(i, *pos);
        }
    }

    /// Populate the hash from positions with per-object radii.
    pub fn build_with_radii(&mut self, positions: &[Vec2], radii: &[f32]) {
        self.clear();
        for (i, (pos, &r)) in positions.iter().zip(radii.iter()).enumerate() {
            self.insert_with_radius(i, *pos, r);
        }
    }

    /// Return indices of all items in cells overlapping the query circle.
    /// Caller must do final distance check.
    pub fn query_nearby(&self, pos: Vec2, radius: f32) -> Vec<usize> {
        let mut result = Vec::new();

        let min_col = ((pos.x - radius) * self.inv_cell_size).floor().max(0.0) as usize;
        let max_col = ((pos.x + radius) * self.inv_cell_size).floor() as usize;
        let max_col = max_col.min(self.cols - 1);
        let min_row = ((pos.y - radius) * self.inv_cell_size).floor().max(0.0) as usize;
        let max_row = ((pos.y + radius) * self.inv_cell_size).floor() as usize;
        let max_row = max_row.min(self.rows - 1);

        for row in min_row..=max_row {
            let base = row * self.cols;
            for col in min_col..=max_col {
                result.extend_from_slice(&self.cells[base + col]);
            }
        }
        result
    }

    /// Return the K nearest items to `pos`, sorted by distance (ascending).
    /// `positions` must be the same slice used to build the hash.
    pub fn nearest_k(
        &self,
        pos: Vec2,
        k: usize,
        positions: &[Vec2],
    ) -> Vec<(usize, f32)> {
        if k == 0 || positions.is_empty() {
            return Vec::new();
        }

        // Expand search radius until we have enough candidates
        let mut radius = self.cell_size;
        let max_radius = self.cell_size * (self.cols.max(self.rows)) as f32;

        loop {
            let candidates = self.query_nearby(pos, radius);
            if candidates.len() >= k || radius >= max_radius {
                let mut dists: Vec<(usize, f32)> = candidates
                    .iter()
                    .map(|&i| (i, pos.distance_to(&positions[i])))
                    .collect();
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                dists.truncate(k);
                return dists;
            }
            radius *= 2.0;
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
    fn insert_and_query() {
        let mut sh = SpatialHash::new(10.0, 10.0, 2.0);
        sh.insert(0, Vec2::new(1.0, 1.0));
        sh.insert(1, Vec2::new(5.0, 5.0));
        sh.insert(2, Vec2::new(1.5, 1.5));

        let near = sh.query_nearby(Vec2::new(1.0, 1.0), 1.0);
        assert!(near.contains(&0));
        assert!(near.contains(&2));
        assert!(!near.contains(&1));
    }

    #[test]
    fn build_from_positions() {
        let positions = vec![
            Vec2::new(1.0, 1.0),
            Vec2::new(5.0, 5.0),
            Vec2::new(9.0, 9.0),
        ];
        let mut sh = SpatialHash::new(10.0, 10.0, 2.0);
        sh.build(&positions);

        let near = sh.query_nearby(Vec2::new(1.0, 1.0), 0.5);
        assert!(near.contains(&0));
        assert!(!near.contains(&1));
        assert!(!near.contains(&2));
    }

    #[test]
    fn nearest_k_basic() {
        let positions = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(5.0, 5.0),
        ];
        let mut sh = SpatialHash::new(10.0, 10.0, 2.0);
        sh.build(&positions);

        let nearest = sh.nearest_k(Vec2::new(0.5, 0.0), 2, &positions);
        assert_eq!(nearest.len(), 2);
        assert_eq!(nearest[0].0, 0); // closest
        assert_eq!(nearest[1].0, 1); // second closest
    }

    #[test]
    fn nearest_k_fewer_than_k() {
        let positions = vec![Vec2::new(1.0, 1.0)];
        let mut sh = SpatialHash::new(10.0, 10.0, 2.0);
        sh.build(&positions);

        let nearest = sh.nearest_k(Vec2::new(0.0, 0.0), 5, &positions);
        assert_eq!(nearest.len(), 1);
    }

    #[test]
    fn clear_removes_all() {
        let mut sh = SpatialHash::new(10.0, 10.0, 2.0);
        sh.insert(0, Vec2::new(1.0, 1.0));
        sh.clear();

        let near = sh.query_nearby(Vec2::new(1.0, 1.0), 5.0);
        assert!(near.is_empty());
    }

    #[test]
    fn edge_positions() {
        let mut sh = SpatialHash::new(10.0, 10.0, 2.0);
        sh.insert(0, Vec2::new(0.0, 0.0));
        sh.insert(1, Vec2::new(10.0, 10.0));
        sh.insert(2, Vec2::new(0.0, 10.0));
        sh.insert(3, Vec2::new(10.0, 0.0));

        // All should be retrievable
        let all = sh.query_nearby(Vec2::new(5.0, 5.0), 10.0);
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn query_at_boundary() {
        let mut sh = SpatialHash::new(10.0, 10.0, 2.0);
        sh.insert(0, Vec2::new(1.9, 1.0));
        sh.insert(1, Vec2::new(2.1, 1.0));

        // These are in adjacent cells. Query with radius 0.5 from cell boundary
        let near = sh.query_nearby(Vec2::new(2.0, 1.0), 0.5);
        assert!(near.contains(&0));
        assert!(near.contains(&1));
    }

    #[test]
    fn insert_with_radius_spans_cells() {
        let mut sh = SpatialHash::new(10.0, 10.0, 2.0);
        // Object at (5, 5) with radius 3 should span cells (1,1) through (3,3)
        sh.insert_with_radius(0, Vec2::new(5.0, 5.0), 3.0);

        // Should be findable from a distant point within the radius
        let near = sh.query_nearby(Vec2::new(3.0, 3.0), 0.1);
        assert!(near.contains(&0), "large object should be in cell near (3,3)");

        // Should NOT be in a cell far away
        let far = sh.query_nearby(Vec2::new(0.5, 0.5), 0.1);
        assert!(!far.contains(&0), "large object should not be in cell near (0,0)");
    }

    #[test]
    fn build_with_radii() {
        let positions = vec![Vec2::new(5.0, 5.0), Vec2::new(1.0, 1.0)];
        let radii = vec![3.0, 0.1];
        let mut sh = SpatialHash::new(10.0, 10.0, 2.0);
        sh.build_with_radii(&positions, &radii);

        // Large object should be findable from (3, 3)
        let near = sh.query_nearby(Vec2::new(3.0, 3.0), 0.1);
        assert!(near.contains(&0));
        assert!(!near.contains(&1));
    }
}
