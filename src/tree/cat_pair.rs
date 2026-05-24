//! GGFP v5.0 — JIT-CatPairSplit routing helpers.
//!
//! All walker functions (predict.rs, model/refine.rs, model/multiclass.rs)
//! call one of these helpers when `tree.is_cat_pair(node)` is true.

use super::{BinnedData, DecisionTree, MISSING_BIN};

impl DecisionTree {
    /// True iff `node` is a JIT-CatPairSplit node.
    #[inline(always)]
    pub fn is_cat_pair(&self, node: usize) -> bool {
        !self.cat_pair_feat2.is_empty()
            && node < self.cat_pair_feat2.len()
            && self.cat_pair_feat2[node] != u32::MAX
    }

    /// Route a cat-pair node using BinnedData (training-time predict).
    /// Returns Some(true)=left, Some(false)=right, None=missing-route required.
    #[inline]
    pub fn cat_pair_route_binned(
        &self,
        node: usize,
        binned: &BinnedData,
        row: usize,
    ) -> Option<bool> {
        let f1 = self.split_features[node] as usize;
        let f2 = self.cat_pair_feat2[node] as usize;
        let b1 = binned.get_bin_u16(row, f1);
        let b2 = binned.get_bin_u16(row, f2);
        if b1 == MISSING_BIN || b2 == MISSING_BIN {
            return None;
        }
        self.cat_pair_route_from_bins(node, b1 as usize, b2 as usize)
    }

    /// Route a cat-pair node using flat bin_indices layout (col*n_rows + row).
    #[inline]
    pub fn cat_pair_route_bin_indices(
        &self,
        node: usize,
        bin_indices: &[u16],
        n_rows: usize,
        row: usize,
    ) -> Option<bool> {
        let f1 = self.split_features[node] as usize;
        let f2 = self.cat_pair_feat2[node] as usize;
        let b1 = bin_indices[f1 * n_rows + row];
        let b2 = bin_indices[f2 * n_rows + row];
        if b1 == MISSING_BIN || b2 == MISSING_BIN {
            return None;
        }
        self.cat_pair_route_from_bins(node, b1 as usize, b2 as usize)
    }

    /// Route a cat-pair node from raw f64 row values (test-time predict).
    pub fn cat_pair_route_raw_row(
        &self,
        node: usize,
        binned: &BinnedData,
        raw_row: &[f64],
    ) -> Option<bool> {
        let f1 = self.split_features[node] as usize;
        let f2 = self.cat_pair_feat2[node] as usize;
        let b1 = raw_to_categorical_bin(&binned.bin_edges[f1], raw_row[f1])?;
        let b2 = raw_to_categorical_bin(&binned.bin_edges[f2], raw_row[f2])?;
        self.cat_pair_route_from_bins(node, b1, b2)
    }

    #[inline]
    fn cat_pair_route_from_bins(&self, node: usize, b1: usize, b2: usize) -> Option<bool> {
        let map_a = &self.cat_pair_bucket_map_a[node];
        let map_b = &self.cat_pair_bucket_map_b[node];
        if b1 >= map_a.len() || b2 >= map_b.len() {
            return None;
        }
        let bu1 = map_a[b1] as usize;
        let bu2 = map_b[b2] as usize;
        let k = self.cat_pair_k_buckets[node] as usize;
        if k == 0 {
            return None;
        }
        let cell = bu1 * k + bu2;
        if cell >= 64 {
            return None;
        }
        Some(((self.cat_pair_cell_mask[node] >> cell) & 1) == 1)
    }
}

#[inline]
fn raw_to_categorical_bin(edges: &[f64], val: f64) -> Option<usize> {
    if val.is_nan() {
        return None;
    }
    let mut lo = 0usize;
    let mut hi = edges.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if edges[mid] < val {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo < edges.len() && (edges[lo] - val).abs() < 0.5 {
        Some(lo)
    } else {
        None
    }
}
