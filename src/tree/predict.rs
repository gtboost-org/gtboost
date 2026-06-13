//! Inference paths on `DecisionTree`.
//!
//! All methods that route a row through the tree and produce predictions:
//!
//! - **Oblique projection** (private helpers): `oblique_proj_binned`,
//!   `oblique_proj_raw`, `oblique_proj_binned_raw`.
//! - **Routing**: `route_to_leaf`, `route_to_leaf_row`.
//! - **Single-row prediction**: `predict_binned`, `predict_binned_raw`,
//!   `predict_raw_row`, `predict_raw_row_from_node`.
//! - **Soft / pruned variants**: `predict_raw_row_soft` (smooth routing
//!   for SRP), `predict_raw_row_pruned` (PRM marginalization).
//! - **Batched accumulation**: `add_predictions_binned`,
//!   `add_predictions_loo`, `add_predictions_binned_raw`,
//!   `predict_and_add_binned` — used by the boosting loop and refinement.
//! - **Tree introspection**: `extract_split_cooccurrences`, `node_counts`.
//!
//! Private helpers (`has_cll_or_ramp`, `predict_binned_simple`, `prm_walk`,
//! `raw_cat_bin`, `child_mass_weights`, `leaf_raw_value`) live alongside
//! their callers here.

use rayon::prelude::*;

use super::*;

impl DecisionTree {
    pub fn has_self_score_splits(&self) -> bool {
        self.split_features
            .iter()
            .any(|&feat| feat == SELF_SCORE_FEATURE)
    }

    pub fn rewrite_feature_as_self_score(&mut self, feature_idx: usize, edges: &[f64]) -> bool {
        let feature = feature_idx as u32;
        let mut rewritten = false;
        for node in 0..self.split_features.len() {
            if self.split_features[node] != feature {
                continue;
            }
            if self.is_oblique_split[node] || self.is_cat_split[node] || self.is_cat_pair(node) {
                continue;
            }
            let split_bin = self.split_bins[node] as usize;
            if split_bin >= edges.len() {
                continue;
            }
            self.split_features[node] = SELF_SCORE_FEATURE;
            self.oblique_thresholds[node] = edges[split_bin] as f32;
            rewritten = true;
        }
        rewritten
    }

    #[inline]
    fn oblique_proj_binned(&self, node: usize, binned: &BinnedData, row: usize) -> Option<f64> {
        let base = node * 2;
        if base + 1 >= self.oblique_features.len() || base + 1 >= self.oblique_weights.len() {
            return None;
        }
        let f0 = self.oblique_features[base];
        if f0 == u32::MAX {
            return None;
        }
        let b0 = binned.get_bin_u16(row, f0 as usize);
        if b0 == MISSING_BIN {
            return None;
        }
        let mut proj = self.oblique_weights[base] as f64 * b0 as f64;
        let f1 = self.oblique_features[base + 1];
        if f1 != u32::MAX {
            let b1 = binned.get_bin_u16(row, f1 as usize);
            if b1 == MISSING_BIN {
                return None;
            }
            proj += self.oblique_weights[base + 1] as f64 * b1 as f64;
        }
        Some(proj)
    }

    #[inline]
    fn oblique_proj_raw(&self, node: usize, binned: &BinnedData, raw_row: &[f64]) -> Option<f64> {
        let base = node * 2;
        if base + 1 >= self.oblique_features.len() || base + 1 >= self.oblique_weights.len() {
            return None;
        }
        let f0 = self.oblique_features[base];
        if f0 == u32::MAX {
            return None;
        }
        if f0 as usize >= raw_row.len() {
            return None;
        }
        let b0 = raw_to_num_bin(&binned.bin_edges[f0 as usize], raw_row[f0 as usize])?;
        let mut proj = self.oblique_weights[base] as f64 * b0 as f64;
        let f1 = self.oblique_features[base + 1];
        if f1 != u32::MAX {
            if f1 as usize >= raw_row.len() {
                return None;
            }
            let b1 = raw_to_num_bin(&binned.bin_edges[f1 as usize], raw_row[f1 as usize])?;
            proj += self.oblique_weights[base + 1] as f64 * b1 as f64;
        }
        Some(proj)
    }

    #[inline]
    fn oblique_proj_binned_raw(
        &self,
        node: usize,
        bin_indices: &[u16],
        n_rows: usize,
        row: usize,
    ) -> Option<f64> {
        let base = node * 2;
        if base + 1 >= self.oblique_features.len() || base + 1 >= self.oblique_weights.len() {
            return None;
        }
        let f0 = self.oblique_features[base];
        if f0 == u32::MAX {
            return None;
        }
        let b0 = bin_indices[f0 as usize * n_rows + row];
        if b0 == MISSING_BIN {
            return None;
        }
        let mut proj = self.oblique_weights[base] as f64 * b0 as f64;
        let f1 = self.oblique_features[base + 1];
        if f1 != u32::MAX {
            let b1 = bin_indices[f1 as usize * n_rows + row];
            if b1 == MISSING_BIN {
                return None;
            }
            proj += self.oblique_weights[base + 1] as f64 * b1 as f64;
        }
        Some(proj)
    }

    #[inline]
    /// route_to_leaf variant that records every visited node (for tree-structured
    /// Stein: per-node aggregates along paths). Same routing logic as route_to_leaf.
    pub fn route_to_leaf_with_path(
        &self,
        binned: &BinnedData,
        row: usize,
        path: &mut Vec<u32>,
    ) -> usize {
        let mut node = 0usize;
        loop {
            path.push(node as u32);
            let feat = self.split_features[node];
            if feat == u32::MAX {
                return node;
            }
            node = self.route_step(binned, row, node);
        }
    }

    /// One routing step from an internal node (shared by path-recording walker).
    #[inline]
    fn route_step(&self, binned: &BinnedData, row: usize, node: usize) -> usize {
        let left = self.left_children[node] as usize;
        let right = self.right_children[node] as usize;
        if self.is_oblique_split[node] {
            if let Some(proj) = self.oblique_proj_binned(node, binned, row) {
                if proj <= self.oblique_thresholds[node] as f64 { left } else { right }
            } else if self.missing_goes_left[node] { left } else { right }
        } else if self.is_cat_pair(node) {
            match self.cat_pair_route_binned(node, binned, row) {
                Some(true) => left,
                Some(false) => right,
                None => if self.missing_goes_left[node] { left } else { right },
            }
        } else {
            let bin = binned.get_bin_u16(row, self.split_features[node] as usize);
            if bin == MISSING_BIN {
                if self.missing_goes_left[node] { left } else { right }
            } else if self.is_cat_split[node] {
                if bitmask_test(&self.cat_left_masks[node], bin as usize) { left } else { right }
            } else if bin <= self.split_bins[node] { left } else { right }
        }
    }

    pub fn route_to_leaf(&self, binned: &BinnedData, row: usize) -> usize {
        let mut node = 0usize;
        loop {
            let feat = self.split_features[node];
            if feat == u32::MAX {
                return node;
            }
            let left = self.left_children[node] as usize;
            let right = self.right_children[node] as usize;
            node = if self.is_oblique_split[node] {
                if let Some(proj) = self.oblique_proj_binned(node, binned, row) {
                    if proj <= self.oblique_thresholds[node] as f64 {
                        left
                    } else {
                        right
                    }
                } else if self.missing_goes_left[node] {
                    left
                } else {
                    right
                }
            } else if self.is_cat_pair(node) {
                match self.cat_pair_route_binned(node, binned, row) {
                    Some(true) => left,
                    Some(false) => right,
                    None => {
                        if self.missing_goes_left[node] {
                            left
                        } else {
                            right
                        }
                    }
                }
            } else {
                let bin = binned.get_bin_u16(row, feat as usize);
                if bin == MISSING_BIN {
                    if self.missing_goes_left[node] {
                        left
                    } else {
                        right
                    }
                } else if self.is_cat_split[node] {
                    if bitmask_test(&self.cat_left_masks[node], bin as usize) {
                        left
                    } else {
                        right
                    }
                } else if bin <= self.split_bins[node] {
                    left
                } else {
                    right
                }
            };
        }
    }

    #[inline]
    pub fn route_to_leaf_with_score(&self, binned: &BinnedData, row: usize, score: f64) -> usize {
        let mut node = 0usize;
        loop {
            let feat = self.split_features[node];
            if feat == u32::MAX {
                return node;
            }
            let left = self.left_children[node] as usize;
            let right = self.right_children[node] as usize;
            node = if feat == SELF_SCORE_FEATURE {
                if !score.is_finite() {
                    if self.missing_goes_left[node] {
                        left
                    } else {
                        right
                    }
                } else if score <= self.oblique_thresholds[node] as f64 {
                    left
                } else {
                    right
                }
            } else if self.is_oblique_split[node] {
                if let Some(proj) = self.oblique_proj_binned(node, binned, row) {
                    if proj <= self.oblique_thresholds[node] as f64 {
                        left
                    } else {
                        right
                    }
                } else if self.missing_goes_left[node] {
                    left
                } else {
                    right
                }
            } else if self.is_cat_pair(node) {
                match self.cat_pair_route_binned(node, binned, row) {
                    Some(true) => left,
                    Some(false) => right,
                    None => {
                        if self.missing_goes_left[node] {
                            left
                        } else {
                            right
                        }
                    }
                }
            } else {
                let bin = binned.get_bin_u16(row, feat as usize);
                if bin == MISSING_BIN {
                    if self.missing_goes_left[node] {
                        left
                    } else {
                        right
                    }
                } else if self.is_cat_split[node] {
                    if bitmask_test(&self.cat_left_masks[node], bin as usize) {
                        left
                    } else {
                        right
                    }
                } else if bin <= self.split_bins[node] {
                    left
                } else {
                    right
                }
            };
        }
    }

    #[inline]
    pub fn predict_binned(&self, binned: &BinnedData, row: usize) -> f64 {
        let leaf = self.route_to_leaf(binned, row);
        if let Some(ref cll) = self.cat_lookups[leaf] {
            let b = if cll.is_numeric {
                nll_bin_for_row(cll, &binned.bin_indices, binned.n_rows, row)
            } else {
                cll_bin_for_row(cll, &binned.cll_hash_bins, binned.n_rows, row)
            };
            if b < cll.bin_values.len() {
                cll.bin_values[b]
            } else {
                cll.default_value
            }
        } else {
            self.values[leaf] + self.ramp_predict(leaf, &binned.bin_indices, binned.n_rows, row)
        }
    }

    #[inline]
    pub fn predict_binned_with_score(&self, binned: &BinnedData, row: usize, score: f64) -> f64 {
        let leaf = self.route_to_leaf_with_score(binned, row, score);
        if let Some(ref cll) = self.cat_lookups[leaf] {
            let b = if cll.is_numeric {
                nll_bin_for_row(cll, &binned.bin_indices, binned.n_rows, row)
            } else {
                cll_bin_for_row(cll, &binned.cll_hash_bins, binned.n_rows, row)
            };
            if b < cll.bin_values.len() {
                cll.bin_values[b]
            } else {
                cll.default_value
            }
        } else {
            self.values[leaf] + self.ramp_predict(leaf, &binned.bin_indices, binned.n_rows, row)
        }
    }

    #[inline]
    pub fn can_predict_raw_plain_axis(&self) -> bool {
        self.oblique_features.iter().all(|&f| f == u32::MAX)
            && !self.is_oblique_split.iter().any(|&v| v)
            && !self.is_cat_split.iter().any(|&v| v)
            && !self.cat_pair_feat2.iter().any(|&f| f != u32::MAX)
            && !self.has_self_score_splits()
            && !self.has_cll_or_ramp()
            && self.leaf_pair_slopes.is_empty()
            && self.quad_slopes.is_empty()
    }

    #[inline]
    pub fn can_predict_binned_plain(&self) -> bool {
        self.oblique_features.iter().all(|&f| f == u32::MAX)
            && !self.is_oblique_split.iter().any(|&v| v)
            && !self.cat_pair_feat2.iter().any(|&f| f != u32::MAX)
            && !self.has_self_score_splits()
            && !self.has_cll_or_ramp()
            && self.leaf_pair_slopes.is_empty()
            && self.quad_slopes.is_empty()
    }

    #[inline]
    pub fn predict_raw_row_plain_axis(&self, binned: &BinnedData, raw_row: &[f64]) -> f64 {
        let mut node = 0usize;
        loop {
            let feat_u32 = self.split_features[node];
            if feat_u32 == u32::MAX {
                return self.values[node];
            }
            let feat = feat_u32 as usize;
            let left = self.left_children[node] as usize;
            let right = self.right_children[node] as usize;
            let val = binned.raw_value_for_feat(raw_row, feat);
            node = if val.is_nan() {
                if self.missing_goes_left[node] {
                    left
                } else {
                    right
                }
            } else {
                let split_bin = self.split_bins[node] as usize;
                let edges = &binned.bin_edges[feat];
                // split_bin >= edges.len() means every reachable bin is <=
                // split_bin -> LEFT, matching binned routing and the general
                // raw paths (this fast path used to route RIGHT, disagreeing
                // with every other inference entry point).
                if split_bin >= edges.len() || val <= edges[split_bin] {
                    left
                } else {
                    right
                }
            };
        }
    }

    #[inline]
    pub fn predict_binned_plain_raw(&self, bin_indices: &[u16], n_rows: usize, row: usize) -> f64 {
        let mut node = 0usize;
        loop {
            let feat = self.split_features[node];
            if feat == u32::MAX {
                return self.values[node];
            }
            let left = self.left_children[node] as usize;
            let right = self.right_children[node] as usize;
            let bin = bin_indices[feat as usize * n_rows + row];
            node = if bin == MISSING_BIN {
                if self.missing_goes_left[node] {
                    left
                } else {
                    right
                }
            } else if self.is_cat_split[node] {
                if bitmask_test(&self.cat_left_masks[node], bin as usize) {
                    left
                } else {
                    right
                }
            } else if bin <= self.split_bins[node] {
                left
            } else {
                right
            };
        }
    }

    #[inline]
    pub fn predict_binned_plain_row_major(
        &self,
        bin_indices: &[u16],
        n_features: usize,
        row: usize,
    ) -> f64 {
        let mut node = 0usize;
        let row_offset = row * n_features;
        loop {
            let feat = self.split_features[node];
            if feat == u32::MAX {
                return self.values[node];
            }
            let left = self.left_children[node] as usize;
            let right = self.right_children[node] as usize;
            let bin = bin_indices[row_offset + feat as usize];
            node = if bin == MISSING_BIN {
                if self.missing_goes_left[node] {
                    left
                } else {
                    right
                }
            } else if self.is_cat_split[node] {
                if bitmask_test(&self.cat_left_masks[node], bin as usize) {
                    left
                } else {
                    right
                }
            } else if bin <= self.split_bins[node] {
                left
            } else {
                right
            };
        }
    }

    /// Row-major counterpart to [`predict_binned_plain_row_major`] that also
    /// adds a leaf-level ramp/linear contribution. Used when the tree has
    /// leaf-linear / ramp slopes but no oblique / cat-pair / CLL extras —
    /// covers the leaf_linear configuration without the column-major slowdown
    /// of the generic `predict_binned_raw` path.
    #[inline]
    pub fn predict_binned_plain_row_major_with_ramp(
        &self,
        bin_indices: &[u16],
        n_features: usize,
        row: usize,
    ) -> f64 {
        let mut node = 0usize;
        let row_offset = row * n_features;
        loop {
            let feat = self.split_features[node];
            if feat == u32::MAX {
                return self.values[node]
                    + self.ramp_predict_row_major(node, bin_indices, n_features, row);
            }
            let left = self.left_children[node] as usize;
            let right = self.right_children[node] as usize;
            let bin = bin_indices[row_offset + feat as usize];
            node = if bin == MISSING_BIN {
                if self.missing_goes_left[node] {
                    left
                } else {
                    right
                }
            } else if self.is_cat_split[node] {
                if bitmask_test(&self.cat_left_masks[node], bin as usize) {
                    left
                } else {
                    right
                }
            } else if bin <= self.split_bins[node] {
                left
            } else {
                right
            };
        }
    }

    /// Tree is "row-major eligible with ramp": same constraints as
    /// `can_predict_binned_plain` but ramp / leaf_pair / quad slopes are OK
    /// (handled by [`predict_binned_plain_row_major_with_ramp`]).
    #[inline]
    pub fn can_predict_binned_plain_with_ramp(&self) -> bool {
        self.oblique_features.iter().all(|&f| f == u32::MAX)
            && !self.is_oblique_split.iter().any(|&v| v)
            && !self.cat_pair_feat2.iter().any(|&f| f != u32::MAX)
            && !self.has_self_score_splits()
            && self.cat_lookups.iter().all(|c| c.is_none())
    }

    /// Check if tree uses CLL or ramp (for fast-path optimization).
    #[inline]
    fn has_cll_or_ramp(&self) -> bool {
        !self.ramp_slopes.is_empty() || self.cat_lookups.iter().any(|c| c.is_some())
    }

    /// Fast predict for simple trees (no CLL, no ramp): just route + value lookup.
    #[inline]
    fn predict_binned_simple(&self, binned: &BinnedData, row: usize) -> f64 {
        self.values[self.route_to_leaf(binned, row)]
    }

    /// Batch predict: add lr * tree_prediction to predictions[i] for i in 0..n.
    /// Uses rayon for parallelism when n >= 4096.
    /// Fused margin update from builder-captured leaf assignments: per-row
    /// array lookups instead of tree traversals. Rows stamped u32::MAX (out of
    /// the build subsample) fall back to traversal; trees with per-row
    /// corrections (CLL/ramp) fall back entirely.
    pub fn add_predictions_from_leaves(
        &self,
        binned: &BinnedData,
        row_leaves: &[u32],
        predictions: &mut [f64],
        lr: f64,
    ) {
        if self.has_cll_or_ramp() || row_leaves.len() != predictions.len() {
            return self.add_predictions_binned(binned, predictions, lr);
        }
        let n = predictions.len();
        if n >= 8192 {
            let chunk_size = (n / rayon::current_num_threads()).max(2048);
            predictions
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let start = ci * chunk_size;
                    for (j, pred) in chunk.iter_mut().enumerate() {
                        let leaf = row_leaves[start + j];
                        if leaf != u32::MAX {
                            *pred += lr * self.values[leaf as usize];
                        } else {
                            *pred += lr * self.predict_binned_simple(binned, start + j);
                        }
                    }
                });
        } else {
            for (i, pred) in predictions.iter_mut().enumerate() {
                let leaf = row_leaves[i];
                if leaf != u32::MAX {
                    *pred += lr * self.values[leaf as usize];
                } else {
                    *pred += lr * self.predict_binned_simple(binned, i);
                }
            }
        }
    }

    pub fn add_predictions_binned(&self, binned: &BinnedData, predictions: &mut [f64], lr: f64) {
        let n = predictions.len();
        let simple = !self.has_cll_or_ramp();
        if n >= 4096 {
            let chunk_size = (n / rayon::current_num_threads()).max(1024);
            predictions
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let start = ci * chunk_size;
                    if simple {
                        for (j, pred) in chunk.iter_mut().enumerate() {
                            *pred += lr * self.predict_binned_simple(binned, start + j);
                        }
                    } else {
                        for (j, pred) in chunk.iter_mut().enumerate() {
                            *pred += lr * self.predict_binned(binned, start + j);
                        }
                    }
                });
        } else {
            if simple {
                for i in 0..n {
                    predictions[i] += lr * self.predict_binned_simple(binned, i);
                }
            } else {
                for i in 0..n {
                    predictions[i] += lr * self.predict_binned(binned, i);
                }
            }
        }
    }

    /// Add predictions for trees that may route on the current model score.
    pub fn add_predictions_binned_with_score(
        &self,
        binned: &BinnedData,
        predictions: &mut [f64],
        lr: f64,
    ) {
        let n = predictions.len();
        if n >= 4096 {
            let chunk_size = (n / rayon::current_num_threads()).max(1024);
            predictions
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let start = ci * chunk_size;
                    for (j, pred) in chunk.iter_mut().enumerate() {
                        let score = *pred;
                        *pred += lr * self.predict_binned_with_score(binned, start + j, score);
                    }
                });
        } else {
            for i in 0..n {
                let score = predictions[i];
                predictions[i] += lr * self.predict_binned_with_score(binned, i, score);
            }
        }
    }

    /// LOO (Leave-One-Out) prediction: for training samples, exclude each sample's own
    /// gradient/hessian from its leaf value. Prevents self-reinforcing overfitting.
    /// For samples not in build_indices, standard leaf values are used.
    pub fn add_predictions_loo(
        &self,
        binned: &BinnedData,
        predictions: &mut [f64],
        lr: f64,
        gradients: &[f64],
        hessians: &[f64],
        lambda_reg: f64,
        build_indices: &[u32],
        count_tau: f64,
    ) {
        let n = predictions.len();
        let n_nodes = self.values.len();

        let mut leaf_g = vec![0.0f64; n_nodes];
        let mut leaf_h = vec![0.0f64; n_nodes];
        let mut leaf_cnt = vec![0u32; n_nodes];
        let mut is_build = vec![false; n];

        for &idx in build_indices {
            let i = idx as usize;
            is_build[i] = true;
            let leaf = self.route_to_leaf(binned, i);
            leaf_g[leaf] += gradients[i];
            leaf_h[leaf] += hessians[i];
            leaf_cnt[leaf] += 1;
        }

        let has_extra = self.has_cll_or_ramp();
        for i in 0..n {
            let leaf = self.route_to_leaf(binned, i);
            let base = if is_build[i] && leaf_cnt[leaf] > 1 {
                let g_loo = leaf_g[leaf] - gradients[i];
                let h_loo = leaf_h[leaf] - hessians[i];
                let denom = h_loo + lambda_reg;
                if denom > 1e-10 {
                    let raw = -g_loo / denom;
                    if count_tau > 0.0 {
                        let cnt_loo = (leaf_cnt[leaf] - 1) as f64;
                        raw * (cnt_loo / (cnt_loo + count_tau))
                    } else {
                        raw
                    }
                } else {
                    self.values[leaf]
                }
            } else {
                self.values[leaf]
            };
            let extra = if has_extra {
                if let Some(ref cll) = self.cat_lookups[leaf] {
                    let b = if cll.is_numeric {
                        nll_bin_for_row(cll, &binned.bin_indices, binned.n_rows, i)
                    } else {
                        cll_bin_for_row(cll, &binned.cll_hash_bins, binned.n_rows, i)
                    };
                    let cll_val = if b < cll.bin_values.len() {
                        cll.bin_values[b]
                    } else {
                        cll.default_value
                    };
                    cll_val - self.values[leaf]
                } else {
                    self.ramp_predict(leaf, &binned.bin_indices, binned.n_rows, i)
                }
            } else {
                0.0
            };
            predictions[i] += lr * (base + extra);
        }
    }

    /// Batch predict using raw bin indices: add lr * tree_prediction to predictions[i] for i in 0..n.
    pub fn add_predictions_binned_raw(
        &self,
        bin_indices: &[u16],
        n_rows: usize,
        predictions: &mut [f64],
        lr: f64,
        cll_bins: &[u16],
    ) {
        let n = predictions.len();
        if n >= 4096 {
            let chunk_size = (n / rayon::current_num_threads()).max(1024);
            predictions
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let start = ci * chunk_size;
                    for (j, pred) in chunk.iter_mut().enumerate() {
                        *pred +=
                            lr * self.predict_binned_raw(bin_indices, n_rows, start + j, cll_bins);
                    }
                });
        } else {
            for i in 0..n {
                predictions[i] += lr * self.predict_binned_raw(bin_indices, n_rows, i, cll_bins);
            }
        }
    }

    pub fn add_predictions_binned_raw_with_score(
        &self,
        bin_indices: &[u16],
        n_rows: usize,
        predictions: &mut [f64],
        lr: f64,
        cll_bins: &[u16],
    ) {
        let n = predictions.len();
        if n >= 4096 {
            let chunk_size = (n / rayon::current_num_threads()).max(1024);
            predictions
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let start = ci * chunk_size;
                    for (j, pred) in chunk.iter_mut().enumerate() {
                        let score = *pred;
                        *pred += lr
                            * self.predict_binned_raw_with_score(
                                bin_indices,
                                n_rows,
                                start + j,
                                cll_bins,
                                score,
                            );
                    }
                });
        } else {
            for i in 0..n {
                let score = predictions[i];
                predictions[i] += lr
                    * self.predict_binned_raw_with_score(bin_indices, n_rows, i, cll_bins, score);
            }
        }
    }

    /// Batch predict: write tree predictions to output buffer and add lr * pred to predictions.
    /// Returns the output buffer for reuse.
    pub fn predict_and_add_binned(
        &self,
        binned: &BinnedData,
        predictions: &mut [f64],
        lr: f64,
        out: &mut Vec<f64>,
    ) {
        let n = predictions.len();
        out.resize(n, 0.0);
        if n >= 4096 {
            let chunk_size = (n / rayon::current_num_threads()).max(1024);
            out.par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let start = ci * chunk_size;
                    for (j, o) in chunk.iter_mut().enumerate() {
                        *o = self.predict_binned(binned, start + j);
                    }
                });
        } else {
            for i in 0..n {
                out[i] = self.predict_binned(binned, i);
            }
        }
        for i in 0..n {
            predictions[i] += lr * out[i];
        }
    }

    /// Predict using raw bin indices (column-major, like BinnedData layout).
    /// cll_bins: separate CLL hash bins for high-cardinality features (empty = use bin_indices).
    #[inline]
    pub fn predict_binned_raw(
        &self,
        bin_indices: &[u16],
        n_rows: usize,
        row: usize,
        cll_bins: &[u16],
    ) -> f64 {
        let mut node = 0usize;
        loop {
            let feat = self.split_features[node];
            if feat == u32::MAX {
                if let Some(ref cll) = self.cat_lookups[node] {
                    let b = if cll.is_numeric {
                        nll_bin_for_row(cll, bin_indices, n_rows, row)
                    } else {
                        let bins = if cll_bins.is_empty() {
                            bin_indices
                        } else {
                            cll_bins
                        };
                        cll_bin_for_row(cll, bins, n_rows, row)
                    };
                    return if b < cll.bin_values.len() {
                        cll.bin_values[b]
                    } else {
                        cll.default_value
                    };
                }
                return self.values[node] + self.ramp_predict(node, bin_indices, n_rows, row);
            }
            node = if self.is_oblique_split[node] {
                if let Some(proj) = self.oblique_proj_binned_raw(node, bin_indices, n_rows, row) {
                    if proj <= self.oblique_thresholds[node] as f64 {
                        self.left_children[node] as usize
                    } else {
                        self.right_children[node] as usize
                    }
                } else if self.missing_goes_left[node] {
                    self.left_children[node] as usize
                } else {
                    self.right_children[node] as usize
                }
            } else if self.is_cat_pair(node) {
                match self.cat_pair_route_bin_indices(node, bin_indices, n_rows, row) {
                    Some(true) => self.left_children[node] as usize,
                    Some(false) => self.right_children[node] as usize,
                    None => {
                        if self.missing_goes_left[node] {
                            self.left_children[node] as usize
                        } else {
                            self.right_children[node] as usize
                        }
                    }
                }
            } else {
                let bin = bin_indices[feat as usize * n_rows + row];
                if bin == MISSING_BIN {
                    if self.missing_goes_left[node] {
                        self.left_children[node] as usize
                    } else {
                        self.right_children[node] as usize
                    }
                } else if self.is_cat_split[node] {
                    if bitmask_test(&self.cat_left_masks[node], bin as usize) {
                        self.left_children[node] as usize
                    } else {
                        self.right_children[node] as usize
                    }
                } else if bin <= self.split_bins[node] {
                    self.left_children[node] as usize
                } else {
                    self.right_children[node] as usize
                }
            };
        }
    }

    #[inline]
    pub fn predict_binned_raw_with_score(
        &self,
        bin_indices: &[u16],
        n_rows: usize,
        row: usize,
        cll_bins: &[u16],
        score: f64,
    ) -> f64 {
        let mut node = 0usize;
        loop {
            let feat = self.split_features[node];
            if feat == u32::MAX {
                if let Some(ref cll) = self.cat_lookups[node] {
                    let b = if cll.is_numeric {
                        nll_bin_for_row(cll, bin_indices, n_rows, row)
                    } else {
                        let bins = if cll_bins.is_empty() {
                            bin_indices
                        } else {
                            cll_bins
                        };
                        cll_bin_for_row(cll, bins, n_rows, row)
                    };
                    return if b < cll.bin_values.len() {
                        cll.bin_values[b]
                    } else {
                        cll.default_value
                    };
                }
                return self.values[node] + self.ramp_predict(node, bin_indices, n_rows, row);
            }
            node = if feat == SELF_SCORE_FEATURE {
                if !score.is_finite() {
                    if self.missing_goes_left[node] {
                        self.left_children[node] as usize
                    } else {
                        self.right_children[node] as usize
                    }
                } else if score <= self.oblique_thresholds[node] as f64 {
                    self.left_children[node] as usize
                } else {
                    self.right_children[node] as usize
                }
            } else if self.is_oblique_split[node] {
                if let Some(proj) = self.oblique_proj_binned_raw(node, bin_indices, n_rows, row) {
                    if proj <= self.oblique_thresholds[node] as f64 {
                        self.left_children[node] as usize
                    } else {
                        self.right_children[node] as usize
                    }
                } else if self.missing_goes_left[node] {
                    self.left_children[node] as usize
                } else {
                    self.right_children[node] as usize
                }
            } else if self.is_cat_pair(node) {
                match self.cat_pair_route_bin_indices(node, bin_indices, n_rows, row) {
                    Some(true) => self.left_children[node] as usize,
                    Some(false) => self.right_children[node] as usize,
                    None => {
                        if self.missing_goes_left[node] {
                            self.left_children[node] as usize
                        } else {
                            self.right_children[node] as usize
                        }
                    }
                }
            } else {
                let bin = bin_indices[feat as usize * n_rows + row];
                if bin == MISSING_BIN {
                    if self.missing_goes_left[node] {
                        self.left_children[node] as usize
                    } else {
                        self.right_children[node] as usize
                    }
                } else if self.is_cat_split[node] {
                    if bitmask_test(&self.cat_left_masks[node], bin as usize) {
                        self.left_children[node] as usize
                    } else {
                        self.right_children[node] as usize
                    }
                } else if bin <= self.split_bins[node] {
                    self.left_children[node] as usize
                } else {
                    self.right_children[node] as usize
                }
            };
        }
    }

    /// Walk all root-to-leaf paths and record feature pairs at consecutive depth levels.
    /// Returns Vec of canonical (min, max) feature index pairs.
    pub fn extract_split_cooccurrences(&self, n_original_features: usize) -> Vec<(u32, u32)> {
        let mut pairs = Vec::new();
        if self.split_features.is_empty() || self.split_features[0] == u32::MAX {
            return pairs;
        }
        // DFS stack: (node_idx, parent_split_feature)
        let mut stack: Vec<(usize, Option<u32>)> = vec![(0, None)];
        let limit = n_original_features as u32;
        while let Some((node, parent_feat)) = stack.pop() {
            if node >= self.split_features.len() {
                continue;
            }
            let feat = self.split_features[node];
            if feat == u32::MAX {
                continue;
            } // leaf
              // Record co-occurrence with parent (only for original features)
            if let Some(pf) = parent_feat {
                if pf != feat && pf < limit && feat < limit {
                    let (a, b) = if pf < feat { (pf, feat) } else { (feat, pf) };
                    pairs.push((a, b));
                }
            }
            let left = self.left_children[node] as usize;
            let right = self.right_children[node] as usize;
            stack.push((left, Some(feat)));
            stack.push((right, Some(feat)));
        }
        pairs
    }

    pub fn node_counts(&self) -> (usize, usize) {
        let mut leaves = 0;
        let mut internal = 0;
        for &f in &self.split_features {
            if f == u32::MAX {
                leaves += 1;
            } else {
                internal += 1;
            }
        }
        (leaves, internal)
    }

    #[inline]
    /// Navigate tree to leaf using raw row values (for new data at predict time).
    /// Returns the leaf node index. Does NOT compute ramp/quad/cll adjustments —
    /// used for leaf-fingerprint extraction (lazy-tree KNN).
    pub fn route_to_leaf_row(&self, binned: &BinnedData, raw_row: &[f64]) -> usize {
        let mut node = 0usize;
        loop {
            if self.split_features[node] == u32::MAX {
                return node;
            }
            let feat = self.split_features[node] as usize;
            if self.is_oblique_split[node] {
                if let Some(proj) = self.oblique_proj_raw(node, binned, raw_row) {
                    node = if proj <= self.oblique_thresholds[node] as f64 {
                        self.left_children[node] as usize
                    } else {
                        self.right_children[node] as usize
                    };
                } else {
                    node = if self.missing_goes_left[node] {
                        self.left_children[node] as usize
                    } else {
                        self.right_children[node] as usize
                    };
                }
            } else if self.is_cat_pair(node) {
                match self.cat_pair_route_raw_row(node, binned, raw_row) {
                    Some(true) => node = self.left_children[node] as usize,
                    Some(false) => node = self.right_children[node] as usize,
                    None => {
                        node = if self.missing_goes_left[node] {
                            self.left_children[node] as usize
                        } else {
                            self.right_children[node] as usize
                        };
                    }
                }
            } else {
                let val = binned.raw_value_for_feat(raw_row, feat);
                if val.is_nan() {
                    node = if self.missing_goes_left[node] {
                        self.left_children[node] as usize
                    } else {
                        self.right_children[node] as usize
                    };
                } else if self.is_cat_split[node] {
                    let edges = &binned.bin_edges[feat];
                    let actual_bins = edges.len();
                    let mut lo = 0usize;
                    let mut hi = actual_bins;
                    while lo < hi {
                        let mid = lo + (hi - lo) / 2;
                        if edges[mid] < val {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    let bin = if lo < actual_bins && (edges[lo] - val).abs() < 0.5 {
                        lo
                    } else {
                        usize::MAX
                    };
                    node = if bitmask_test(&self.cat_left_masks[node], bin) {
                        self.left_children[node] as usize
                    } else {
                        self.right_children[node] as usize
                    };
                } else {
                    let split_bin = self.split_bins[node] as usize;
                    let edges = &binned.bin_edges[feat];
                    let in_left = if split_bin < edges.len() {
                        val <= edges[split_bin]
                    } else {
                        true
                    };
                    node = if in_left {
                        self.left_children[node] as usize
                    } else {
                        self.right_children[node] as usize
                    };
                }
            }
        }
    }

    /// SRP — Soft Routing Prediction. Walk the tree with per-node weights,
    /// soft-routing at numeric splits via σ((threshold - val) / (bw · feat_scale)),
    /// aggregating leaf values by accumulated path probability. Smooths the
    /// piecewise-constant step function, reducing boundary-discretization bias.
    /// Categorical splits & missing values still hard-route.
    /// feat_scales: per-feature scale (e.g., avg bin width in raw units). Used to
    /// make the sigmoid scale-invariant across features. bandwidth is dimensionless.
    pub fn predict_raw_row_soft(
        &self,
        binned: &BinnedData,
        raw_row: &[f64],
        bandwidth: f64,
        feat_scales: &[f64],
    ) -> f64 {
        // Depth-bounded stack walk. Tree depth ≤ 16 in practice; cap at 64 for safety.
        let mut stack: [(usize, f64); 128] = [(0, 0.0); 128];
        let mut sp = 0usize;
        stack[0] = (0usize, 1.0f64);
        sp = 1;
        let mut total = 0.0f64;

        while sp > 0 {
            sp -= 1;
            let (node, w) = stack[sp];
            if w < 1e-4 {
                continue;
            }
            if self.split_features[node] == u32::MAX {
                // Leaf — same logic as predict_raw_row (minus cat_lookups to keep simple)
                let mut ramp_val = 0.0f64;
                if !self.ramp_slopes.is_empty() {
                    let k = self.ramp_k;
                    let base = node * k;
                    for j in 0..k {
                        if base + j >= self.ramp_features.len() {
                            break;
                        }
                        let rf = self.ramp_features[base + j];
                        if rf == u32::MAX {
                            continue;
                        }
                        let rfu = rf as usize;
                        if rfu >= raw_row.len() {
                            continue;
                        }
                        let rv = raw_row[rfu];
                        if rv.is_nan() {
                            continue;
                        }
                        let edges = &binned.bin_edges[rfu];
                        let n_bins = edges.len();
                        if n_bins == 0 {
                            continue;
                        }
                        let mut lo = 0usize;
                        let mut hi = n_bins;
                        while lo < hi {
                            let mid = lo + (hi - lo) / 2;
                            if edges[mid] < rv {
                                lo = mid + 1;
                            } else {
                                hi = mid;
                            }
                        }
                        let bin = lo.min(n_bins.saturating_sub(1));
                        ramp_val += self.ramp_slopes[base + j] * bin as f64;
                    }
                }
                total += w * (self.values[node] + ramp_val);
                continue;
            }
            let feat = self.split_features[node] as usize;
            let left = self.left_children[node] as usize;
            let right = self.right_children[node] as usize;
            if self.is_oblique_split[node] {
                if let Some(proj) = self.oblique_proj_raw(node, binned, raw_row) {
                    let nxt = if proj <= self.oblique_thresholds[node] as f64 {
                        left
                    } else {
                        right
                    };
                    if sp < 127 {
                        stack[sp] = (nxt, w);
                        sp += 1;
                    }
                } else {
                    let nxt = if self.missing_goes_left[node] {
                        left
                    } else {
                        right
                    };
                    if sp < 127 {
                        stack[sp] = (nxt, w);
                        sp += 1;
                    }
                }
            } else {
                let val = binned.raw_value_for_feat(raw_row, feat);
                if val.is_nan() {
                    let nxt = if self.missing_goes_left[node] {
                        left
                    } else {
                        right
                    };
                    if sp < 127 {
                        stack[sp] = (nxt, w);
                        sp += 1;
                    }
                } else if self.is_cat_pair(node) {
                    // GGFP v5.0 — cat-pair hard-routes when both features known; unseen
                    // pairs fall back to child-mass split like the regular cat branch.
                    match self.cat_pair_route_raw_row(node, binned, raw_row) {
                        Some(true) => {
                            if sp < 127 {
                                stack[sp] = (left, w);
                                sp += 1;
                            }
                        }
                        Some(false) => {
                            if sp < 127 {
                                stack[sp] = (right, w);
                                sp += 1;
                            }
                        }
                        None => {
                            if let Some((wl, wr)) = self.child_mass_weights(left, right) {
                                let w_left = w * wl;
                                let w_right = w * wr;
                                if sp < 127 && w_right > 1e-4 {
                                    stack[sp] = (right, w_right);
                                    sp += 1;
                                }
                                if sp < 127 && w_left > 1e-4 {
                                    stack[sp] = (left, w_left);
                                    sp += 1;
                                }
                            } else {
                                total += w * self.values[node];
                            }
                        }
                    }
                } else if self.is_cat_split[node] {
                    // Known categories still hard-route. Unseen categories are marginalized
                    // over child training mass instead of falling through to hard-right.
                    if let Some(bin) = Self::raw_cat_bin(&binned.bin_edges[feat], val) {
                        let nxt = if bitmask_test(&self.cat_left_masks[node], bin) {
                            left
                        } else {
                            right
                        };
                        if sp < 127 {
                            stack[sp] = (nxt, w);
                            sp += 1;
                        }
                    } else if let Some((wl, wr)) = self.child_mass_weights(left, right) {
                        let w_left = w * wl;
                        let w_right = w * wr;
                        if sp < 127 && w_right > 1e-4 {
                            stack[sp] = (right, w_right);
                            sp += 1;
                        }
                        if sp < 127 && w_left > 1e-4 {
                            stack[sp] = (left, w_left);
                            sp += 1;
                        }
                    } else {
                        total += w * self.values[node];
                    }
                } else {
                    // Numeric: soft route
                    let split_bin = self.split_bins[node] as usize;
                    let edges = &binned.bin_edges[feat];
                    if split_bin >= edges.len() {
                        if sp < 127 {
                            stack[sp] = (left, w);
                            sp += 1;
                        }
                        continue;
                    }
                    let threshold = edges[split_bin];
                    let scale = if feat < feat_scales.len() && feat_scales[feat] > 0.0 {
                        feat_scales[feat]
                    } else {
                        1.0
                    };
                    let dx = (threshold - val) / (bandwidth * scale);
                    let p_left = 1.0 / (1.0 + (-dx).exp());
                    let w_left = w * p_left;
                    let w_right = w - w_left;
                    if sp < 127 && w_right > 1e-4 {
                        stack[sp] = (right, w_right);
                        sp += 1;
                    }
                    if sp < 127 && w_left > 1e-4 {
                        stack[sp] = (left, w_left);
                        sp += 1;
                    }
                }
            }
        }
        total
    }

    /// Collect soft leaf memberships as (leaf_node, path_probability) pairs using
    /// the SAME routing as `predict_raw_row_soft`: numeric splits route softly via
    /// σ((thr - x)/(bw·scale)); categorical / oblique / missing-value splits hard
    /// route (unseen categories marginalize over child training mass). Sharing one
    /// router lets soft prediction and soft leaf refit stay consistent. Constant
    /// leaves only — ramp terms are applied by the caller at predict time.
    pub(crate) fn soft_collect_leaves(
        &self,
        binned: &BinnedData,
        raw_row: &[f64],
        bandwidth: f64,
        feat_scales: &[f64],
        out: &mut Vec<(usize, f64)>,
    ) {
        let mut stack: [(usize, f64); 128] = [(0, 0.0); 128];
        stack[0] = (0usize, 1.0f64);
        let mut sp = 1usize;
        while sp > 0 {
            sp -= 1;
            let (node, w) = stack[sp];
            if w < 1e-4 {
                continue;
            }
            if self.split_features[node] == u32::MAX {
                out.push((node, w));
                continue;
            }
            let feat = self.split_features[node] as usize;
            let left = self.left_children[node] as usize;
            let right = self.right_children[node] as usize;
            if self.is_oblique_split[node] {
                let nxt = match self.oblique_proj_raw(node, binned, raw_row) {
                    Some(proj) => {
                        if proj <= self.oblique_thresholds[node] as f64 {
                            left
                        } else {
                            right
                        }
                    }
                    None => {
                        if self.missing_goes_left[node] {
                            left
                        } else {
                            right
                        }
                    }
                };
                if sp < 127 {
                    stack[sp] = (nxt, w);
                    sp += 1;
                }
                continue;
            }
            let val = binned.raw_value_for_feat(raw_row, feat);
            if val.is_nan() {
                let nxt = if self.missing_goes_left[node] {
                    left
                } else {
                    right
                };
                if sp < 127 {
                    stack[sp] = (nxt, w);
                    sp += 1;
                }
            } else if self.is_cat_pair(node) {
                match self.cat_pair_route_raw_row(node, binned, raw_row) {
                    Some(true) => {
                        if sp < 127 {
                            stack[sp] = (left, w);
                            sp += 1;
                        }
                    }
                    Some(false) => {
                        if sp < 127 {
                            stack[sp] = (right, w);
                            sp += 1;
                        }
                    }
                    None => {
                        if let Some((wl, wr)) = self.child_mass_weights(left, right) {
                            let w_left = w * wl;
                            let w_right = w - w_left;
                            if sp < 127 && w_right > 1e-4 {
                                stack[sp] = (right, w_right);
                                sp += 1;
                            }
                            if sp < 127 && w_left > 1e-4 {
                                stack[sp] = (left, w_left);
                                sp += 1;
                            }
                        } else {
                            out.push((node, w));
                        }
                    }
                }
            } else if self.is_cat_split[node] {
                if let Some(bin) = Self::raw_cat_bin(&binned.bin_edges[feat], val) {
                    let nxt = if bitmask_test(&self.cat_left_masks[node], bin) {
                        left
                    } else {
                        right
                    };
                    if sp < 127 {
                        stack[sp] = (nxt, w);
                        sp += 1;
                    }
                } else if let Some((wl, wr)) = self.child_mass_weights(left, right) {
                    let w_left = w * wl;
                    let w_right = w - w_left;
                    if sp < 127 && w_right > 1e-4 {
                        stack[sp] = (right, w_right);
                        sp += 1;
                    }
                    if sp < 127 && w_left > 1e-4 {
                        stack[sp] = (left, w_left);
                        sp += 1;
                    }
                } else {
                    out.push((node, w));
                }
            } else {
                let split_bin = self.split_bins[node] as usize;
                let edges = &binned.bin_edges[feat];
                if split_bin >= edges.len() {
                    if sp < 127 {
                        stack[sp] = (left, w);
                        sp += 1;
                    }
                    continue;
                }
                let threshold = edges[split_bin];
                let scale = if feat < feat_scales.len() && feat_scales[feat] > 0.0 {
                    feat_scales[feat]
                } else {
                    1.0
                };
                let dx = (threshold - val) / (bandwidth * scale);
                let p_left = 1.0 / (1.0 + (-dx).exp());
                let w_left = w * p_left;
                let w_right = w - w_left;
                if sp < 127 && w_right > 1e-4 {
                    stack[sp] = (right, w_right);
                    sp += 1;
                }
                if sp < 127 && w_left > 1e-4 {
                    stack[sp] = (left, w_left);
                    sp += 1;
                }
            }
        }
    }

    /// PRM — Posterior Refinement Marginalization. Walk tree with confidence-aware
    /// survival gates. At each internal node, prediction = q·(descend further) + (1-q)·node_value
    /// where q = h_node / (h_node + τ · 2^depth). High-confidence nodes (high h_sum)
    /// keep descending; low-confidence deep nodes fall back to ancestor prediction.
    /// Expected value form (deterministic, not MC) — single forward pass.
    pub fn predict_raw_row_pruned(&self, binned: &BinnedData, raw_row: &[f64], tau: f64) -> f64 {
        if self.node_h_sum.is_empty() || tau <= 0.0 {
            return self.predict_raw_row(binned, raw_row);
        }
        self.prm_walk(0, binned, raw_row, tau, 0)
    }

    fn prm_walk(
        &self,
        node: usize,
        binned: &BinnedData,
        raw_row: &[f64],
        tau: f64,
        depth: usize,
    ) -> f64 {
        if self.split_features[node] == u32::MAX {
            return self.values[node];
        }
        let h = self.node_h_sum.get(node).copied().unwrap_or(0.0);
        let depth_penalty = (1u64 << depth.min(20)) as f64;
        let denom = h + tau * depth_penalty;
        let q = if denom > 0.0 { h / denom } else { 1.0 };
        let feat = self.split_features[node] as usize;
        let next = if self.is_oblique_split[node] {
            if let Some(proj) = self.oblique_proj_raw(node, binned, raw_row) {
                if proj <= self.oblique_thresholds[node] as f64 {
                    self.left_children[node] as usize
                } else {
                    self.right_children[node] as usize
                }
            } else if self.missing_goes_left[node] {
                self.left_children[node] as usize
            } else {
                self.right_children[node] as usize
            }
        } else {
            let val = binned.raw_value_for_feat(raw_row, feat);
            if val.is_nan() {
                if self.missing_goes_left[node] {
                    self.left_children[node] as usize
                } else {
                    self.right_children[node] as usize
                }
            } else if self.is_cat_pair(node) {
                let left = self.left_children[node] as usize;
                let right = self.right_children[node] as usize;
                match self.cat_pair_route_raw_row(node, binned, raw_row) {
                    Some(true) => left,
                    Some(false) => right,
                    None => {
                        if let Some((wl, wr)) = self.child_mass_weights(left, right) {
                            let descend_val = wl
                                * self.prm_walk(left, binned, raw_row, tau, depth + 1)
                                + wr * self.prm_walk(right, binned, raw_row, tau, depth + 1);
                            return q * descend_val + (1.0 - q) * self.values[node];
                        } else {
                            return self.values[node];
                        }
                    }
                }
            } else if self.is_cat_split[node] {
                let left = self.left_children[node] as usize;
                let right = self.right_children[node] as usize;
                if let Some(bin) = Self::raw_cat_bin(&binned.bin_edges[feat], val) {
                    if bitmask_test(&self.cat_left_masks[node], bin) {
                        left
                    } else {
                        right
                    }
                } else if let Some((wl, wr)) = self.child_mass_weights(left, right) {
                    let descend_val = wl * self.prm_walk(left, binned, raw_row, tau, depth + 1)
                        + wr * self.prm_walk(right, binned, raw_row, tau, depth + 1);
                    return q * descend_val + (1.0 - q) * self.values[node];
                } else {
                    return self.values[node];
                }
            } else {
                let split_bin = self.split_bins[node] as usize;
                let edges = &binned.bin_edges[feat];
                let in_left = if split_bin < edges.len() {
                    val <= edges[split_bin]
                } else {
                    true
                };
                if in_left {
                    self.left_children[node] as usize
                } else {
                    self.right_children[node] as usize
                }
            }
        };
        let descend_val = self.prm_walk(next, binned, raw_row, tau, depth + 1);
        q * descend_val + (1.0 - q) * self.values[node]
    }

    #[inline]
    fn raw_cat_bin(edges: &[f64], val: f64) -> Option<usize> {
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

    #[inline]
    fn child_mass_weights(&self, left: usize, right: usize) -> Option<(f64, f64)> {
        let lc = self.node_count.get(left).copied().unwrap_or(0) as f64;
        let rc = self.node_count.get(right).copied().unwrap_or(0) as f64;
        let ct = lc + rc;
        if ct > 0.0 {
            return Some((lc / ct, rc / ct));
        }

        let lh = self.node_h_sum.get(left).copied().unwrap_or(0.0).max(0.0);
        let rh = self.node_h_sum.get(right).copied().unwrap_or(0.0).max(0.0);
        let ht = lh + rh;
        if ht > 0.0 {
            Some((lh / ht, rh / ht))
        } else {
            None
        }
    }

    #[inline]
    fn leaf_raw_value(&self, node: usize, binned: &BinnedData, raw_row: &[f64]) -> f64 {
        if let Some(ref cll) = self.cat_lookups[node] {
            if cll.is_numeric {
                // NLL: coarsened numeric bin lookup
                let b1 = raw_to_nll_bin(
                    cll.feature as usize,
                    raw_row[cll.feature as usize],
                    &binned.bin_edges,
                    cll.n_coarse_bins,
                );
                if b1 == usize::MAX {
                    return cll.default_value;
                }
                let b = if cll.feature2 == u32::MAX {
                    b1
                } else {
                    let f2 = cll.feature2 as usize;
                    let b2 = raw_to_nll_bin(f2, raw_row[f2], &binned.bin_edges, cll.n_coarse_bins);
                    if b2 == usize::MAX {
                        return cll.default_value;
                    }
                    if cll.pair_stride > 0 {
                        b1 * cll.pair_stride + b2
                    } else {
                        b1 * cll.n_coarse_bins + b2
                    }
                };
                return if b < cll.bin_values.len() {
                    cll.bin_values[b]
                } else {
                    cll.default_value
                };
            }
            let b1 = raw_to_cll_bin(cll.feature as usize, raw_row, binned);
            if b1 == usize::MAX {
                return cll.default_value;
            }
            let bin = if cll.feature2 == u32::MAX {
                b1
            } else {
                let b2 = raw_to_cll_bin(cll.feature2 as usize, raw_row, binned);
                if b2 == usize::MAX {
                    return cll.default_value;
                }
                if cll.pair_stride > 0 {
                    b1 * cll.pair_stride + b2
                } else {
                    let n_bins = cll.bin_values.len().max(1);
                    ((b1 as u32).wrapping_mul(257).wrapping_add(b2 as u32)) as usize % n_bins
                }
            };
            return if bin < cll.bin_values.len() {
                cll.bin_values[bin]
            } else {
                cll.default_value
            };
        }

        let mut ramp_val = 0.0f64;
        if !self.ramp_slopes.is_empty() {
            let k = self.ramp_k;
            let base = node * k;
            for j in 0..k {
                if base + j >= self.ramp_features.len() {
                    break;
                }
                let rf = self.ramp_features[base + j];
                if rf == u32::MAX {
                    continue;
                }
                let rfu = rf as usize;
                if rfu >= raw_row.len() {
                    continue;
                }
                let rv = raw_row[rfu];
                if rv.is_nan() {
                    continue;
                }
                let edges = &binned.bin_edges[rfu];
                if edges.is_empty() {
                    continue;
                }
                let mut lo = 0usize;
                let mut hi = edges.len();
                while lo < hi {
                    let mid = lo + (hi - lo) / 2;
                    if edges[mid] < rv {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                let bin = lo.min(edges.len().saturating_sub(1));
                ramp_val += self.ramp_slopes[base + j] * bin as f64;
            }
        }

        let mut pair_val = 0.0f64;
        if !self.leaf_pair_slopes.is_empty() && node < self.leaf_pair_slopes.len() {
            let base = node * 2;
            if base + 1 < self.leaf_pair_features.len() {
                let fi = self.leaf_pair_features[base];
                let fj = self.leaf_pair_features[base + 1];
                if fi != u32::MAX && fj != u32::MAX {
                    let fi = fi as usize;
                    let fj = fj as usize;
                    if fi < raw_row.len() && fj < raw_row.len() {
                        let vi = raw_row[fi];
                        let vj = raw_row[fj];
                        if !vi.is_nan() && !vj.is_nan() {
                            if let (Some(bin_i), Some(bin_j)) = (
                                raw_to_num_bin(&binned.bin_edges[fi], vi),
                                raw_to_num_bin(&binned.bin_edges[fj], vj),
                            ) {
                                pair_val +=
                                    self.leaf_pair_slopes[node] * bin_i as f64 * bin_j as f64;
                            }
                        }
                    }
                }
            }
        }

        let mut quad_val = 0.0f64;
        if !self.quad_slopes.is_empty() && self.quad_n_interactions > 0 {
            let ni = self.quad_n_interactions;
            let qbase = node * ni;
            if qbase + ni <= self.quad_slopes.len() {
                for j in 0..ni {
                    let (fi, fj) = self.quad_pairs[j];
                    if fi >= raw_row.len() || fj >= raw_row.len() {
                        continue;
                    }
                    let vi = raw_row[fi];
                    let vj = raw_row[fj];
                    if vi.is_nan() || vj.is_nan() {
                        continue;
                    }
                    let bin_i = {
                        let edges = &binned.bin_edges[fi];
                        if edges.is_empty() {
                            0usize
                        } else {
                            let mut lo = 0usize;
                            let mut hi = edges.len();
                            while lo < hi {
                                let mid = lo + (hi - lo) / 2;
                                if edges[mid] < vi {
                                    lo = mid + 1;
                                } else {
                                    hi = mid;
                                }
                            }
                            lo.min(edges.len().saturating_sub(1))
                        }
                    };
                    let bin_j = {
                        let edges = &binned.bin_edges[fj];
                        if edges.is_empty() {
                            0usize
                        } else {
                            let mut lo = 0usize;
                            let mut hi = edges.len();
                            while lo < hi {
                                let mid = lo + (hi - lo) / 2;
                                if edges[mid] < vj {
                                    lo = mid + 1;
                                } else {
                                    hi = mid;
                                }
                            }
                            lo.min(edges.len().saturating_sub(1))
                        }
                    };
                    quad_val += self.quad_slopes[qbase + j] * bin_i as f64 * bin_j as f64;
                }
            }
        }

        self.values[node] + ramp_val + pair_val + quad_val
    }

    fn predict_raw_row_from_node(
        &self,
        mut node: usize,
        binned: &BinnedData,
        raw_row: &[f64],
    ) -> f64 {
        loop {
            if self.split_features[node] == u32::MAX {
                return self.leaf_raw_value(node, binned, raw_row);
            }
            let feat = self.split_features[node] as usize;
            let left = self.left_children[node] as usize;
            let right = self.right_children[node] as usize;
            if self.is_oblique_split[node] {
                if let Some(proj) = self.oblique_proj_raw(node, binned, raw_row) {
                    node = if proj <= self.oblique_thresholds[node] as f64 {
                        left
                    } else {
                        right
                    };
                } else if self.missing_goes_left[node] {
                    node = left;
                } else {
                    node = right;
                }
            } else {
                let val = binned.raw_value_for_feat(raw_row, feat);
                if val.is_nan() {
                    node = if self.missing_goes_left[node] {
                        left
                    } else {
                        right
                    };
                } else if self.is_cat_pair(node) {
                    let left = self.left_children[node] as usize;
                    let right = self.right_children[node] as usize;
                    match self.cat_pair_route_raw_row(node, binned, raw_row) {
                        Some(true) => node = left,
                        Some(false) => node = right,
                        None => {
                            if let Some((wl, wr)) = self.child_mass_weights(left, right) {
                                return wl * self.predict_raw_row_from_node(left, binned, raw_row)
                                    + wr * self.predict_raw_row_from_node(right, binned, raw_row);
                            } else {
                                return self.values[node];
                            }
                        }
                    }
                } else if self.is_cat_split[node] {
                    let left = self.left_children[node] as usize;
                    let right = self.right_children[node] as usize;
                    if let Some(bin) = Self::raw_cat_bin(&binned.bin_edges[feat], val) {
                        node = if bitmask_test(&self.cat_left_masks[node], bin) {
                            left
                        } else {
                            right
                        };
                    } else if let Some((wl, wr)) = self.child_mass_weights(left, right) {
                        return wl * self.predict_raw_row_from_node(left, binned, raw_row)
                            + wr * self.predict_raw_row_from_node(right, binned, raw_row);
                    } else {
                        return self.values[node];
                    };
                } else {
                    let split_bin = self.split_bins[node] as usize;
                    let edges = &binned.bin_edges[feat];
                    let in_left = if split_bin < edges.len() {
                        val <= edges[split_bin]
                    } else {
                        true
                    };
                    node = if in_left { left } else { right };
                }
            }
        }
    }

    fn predict_raw_row_from_node_with_score(
        &self,
        mut node: usize,
        binned: &BinnedData,
        raw_row: &[f64],
        score: f64,
    ) -> f64 {
        loop {
            let feat_u32 = self.split_features[node];
            if feat_u32 == u32::MAX {
                return self.leaf_raw_value(node, binned, raw_row);
            }
            let left = self.left_children[node] as usize;
            let right = self.right_children[node] as usize;
            if feat_u32 == SELF_SCORE_FEATURE {
                node = if !score.is_finite() {
                    if self.missing_goes_left[node] {
                        left
                    } else {
                        right
                    }
                } else if score <= self.oblique_thresholds[node] as f64 {
                    left
                } else {
                    right
                };
                continue;
            }
            let feat = feat_u32 as usize;
            if self.is_oblique_split[node] {
                if let Some(proj) = self.oblique_proj_raw(node, binned, raw_row) {
                    node = if proj <= self.oblique_thresholds[node] as f64 {
                        left
                    } else {
                        right
                    };
                } else if self.missing_goes_left[node] {
                    node = left;
                } else {
                    node = right;
                }
            } else {
                let val = binned.raw_value_for_feat(raw_row, feat);
                if val.is_nan() {
                    node = if self.missing_goes_left[node] {
                        left
                    } else {
                        right
                    };
                } else if self.is_cat_pair(node) {
                    match self.cat_pair_route_raw_row(node, binned, raw_row) {
                        Some(true) => node = left,
                        Some(false) => node = right,
                        None => {
                            if let Some((wl, wr)) = self.child_mass_weights(left, right) {
                                return wl
                                    * self.predict_raw_row_from_node_with_score(
                                        left, binned, raw_row, score,
                                    )
                                    + wr * self.predict_raw_row_from_node_with_score(
                                        right, binned, raw_row, score,
                                    );
                            } else {
                                return self.values[node];
                            }
                        }
                    }
                } else if self.is_cat_split[node] {
                    if let Some(bin) = Self::raw_cat_bin(&binned.bin_edges[feat], val) {
                        node = if bitmask_test(&self.cat_left_masks[node], bin) {
                            left
                        } else {
                            right
                        };
                    } else if let Some((wl, wr)) = self.child_mass_weights(left, right) {
                        return wl
                            * self.predict_raw_row_from_node_with_score(
                                left, binned, raw_row, score,
                            )
                            + wr * self.predict_raw_row_from_node_with_score(
                                right, binned, raw_row, score,
                            );
                    } else {
                        return self.values[node];
                    };
                } else {
                    let split_bin = self.split_bins[node] as usize;
                    let edges = &binned.bin_edges[feat];
                    let in_left = if split_bin < edges.len() {
                        val <= edges[split_bin]
                    } else {
                        true
                    };
                    node = if in_left { left } else { right };
                }
            }
        }
    }

    pub fn predict_raw_row(&self, binned: &BinnedData, raw_row: &[f64]) -> f64 {
        self.predict_raw_row_from_node(0, binned, raw_row)
    }

    pub fn predict_raw_row_with_score(
        &self,
        binned: &BinnedData,
        raw_row: &[f64],
        score: f64,
    ) -> f64 {
        self.predict_raw_row_from_node_with_score(0, binned, raw_row, score)
    }
}
