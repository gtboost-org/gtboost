//! Post-hoc global refinement of leaf values.
//!
//! After the main boosting loop finishes, `refine_global` performs a
//! cached-routing backfitting pass that re-fits each tree's leaves while
//! holding the rest of the ensemble fixed. This is the "global leaf
//! optimization" step — it does not change tree structure, only the
//! per-leaf scalar (or ramp / leaf-linear / leaf-quadratic) values.
//!
//! Honest mode: when `tree_in_sample` masks are populated each tree's
//! leaves are re-fit on its complement (out-of-sample) rows, preserving
//! the honest property through backfitting.
//!
//! Multiclass: `refine_global_multiclass` is the K-class analogue, with
//! softmax-coupled leaf updates and parallelized class-major prediction
//! accumulation.
//!
//! Soft routing: `soft_accumulate_gradients` is the per-row sigmoid-
//! weighted leaf-routing helper used by the threshold-gradient path.

use rand::rngs::StdRng;
use rand::RngExt;
use rand::SeedableRng;
use rayon::prelude::*;

use super::GTBoostModel;
use crate::helpers::{bitvec_test, solve_spd, solve_spd_with_scratch};
use crate::tree::{bitmask_test, BinnedData, DecisionTree, MISSING_BIN};

impl GTBoostModel {
    /// Soft routing gradient accumulation: samples contribute fractionally to
    /// multiple leaves via sigmoid-weighted splits at numeric nodes.
    /// Categorical splits and missing values use hard (deterministic) routing.
    /// Optionally accumulates per-node threshold gradient terms (for split refinement).
    pub(super) fn soft_accumulate_gradients(
        tree: &DecisionTree,
        binned: &BinnedData,
        n_rows: usize,
        grad_buf: &[f64],
        hess_buf: &[f64],
        sharpness: f64,
        honest_mask: Option<&Vec<u64>>,
        refine_mask: &[bool],
        use_stochastic: bool,
        node_g: &mut [f64],
        node_h: &mut [f64],
    ) {
        let n_nodes = tree.split_features.len();
        for i in 0..n_nodes {
            node_g[i] = 0.0;
            node_h[i] = 0.0;
        }

        // Fixed-size stack for tree traversal: (node_index, weight)
        let mut stack = [(0usize, 0.0f64); 64];

        for row in 0..n_rows {
            if let Some(mask) = honest_mask {
                if bitvec_test(mask, row) {
                    continue;
                }
            }
            if use_stochastic && !refine_mask[row] {
                continue;
            }

            let g = grad_buf[row];
            let h = hess_buf[row];

            stack[0] = (0, 1.0);
            let mut sp = 1usize;

            while sp > 0 {
                sp -= 1;
                let (node, w) = stack[sp];

                let feat = tree.split_features[node];
                if feat == u32::MAX {
                    node_g[node] += w * g;
                    node_h[node] += w * h;
                    continue;
                }

                let bin = binned.bin_indices[feat as usize * n_rows + row];
                let left = tree.left_children[node] as usize;
                let right = tree.right_children[node] as usize;

                if tree.is_cat_pair(node) {
                    let child = match tree.cat_pair_route_bin_indices(
                        node,
                        &binned.bin_indices,
                        n_rows,
                        row,
                    ) {
                        Some(true) => left,
                        Some(false) => right,
                        None => {
                            if tree.missing_goes_left[node] {
                                left
                            } else {
                                right
                            }
                        }
                    };
                    if sp < 64 {
                        stack[sp] = (child, w);
                        sp += 1;
                    }
                } else if bin == MISSING_BIN {
                    let child = if tree.missing_goes_left[node] {
                        left
                    } else {
                        right
                    };
                    if sp < 64 {
                        stack[sp] = (child, w);
                        sp += 1;
                    }
                } else if tree.is_cat_split[node] {
                    let child = if bitmask_test(&tree.cat_left_masks[node], bin as usize) {
                        left
                    } else {
                        right
                    };
                    if sp < 64 {
                        stack[sp] = (child, w);
                        sp += 1;
                    }
                } else {
                    let x = (bin as f64) - (tree.split_bins[node] as f64) - 0.5;
                    let s = 1.0 / (1.0 + (-sharpness * x).exp());
                    let w_left = w * (1.0 - s);
                    let w_right = w * s;
                    if sp < 63 {
                        if w_right >= 1e-4 {
                            stack[sp] = (right, w_right);
                            sp += 1;
                        }
                        if w_left >= 1e-4 {
                            stack[sp] = (left, w_left);
                            sp += 1;
                        }
                    }
                }
            }
        }
    }

    /// Global leaf optimization: cached-routing backfitting with optional L1.
    /// When honest mode is active and tree_in_sample masks are available, each
    /// tree's leaves are optimized using only its complement data (samples NOT
    /// used to build that tree's structure), maintaining the honest property
    /// through backfitting. Predictions are still tracked for ALL samples.
    /// When ramp=true, each leaf is upgraded from constant w to w + β*x_parent,
    /// where x_parent is the parent node's split feature (1D linear regression).
    pub(super) fn refine_global(
        &mut self,
        binned: &BinnedData,
        y: &[f64],
        n_rows: usize,
        n_iters: usize,
    ) {
        let lr = self.multiclass_tree_lr();
        let lambda = self.lambda_reg;
        let n_trees = self.trees.len();
        let ramp_lambda = self.ramp_lambda;

        // Precompute leaf routing for all trees (rayon-parallel) if memory permits.
        // Vec<Vec<u16>> costs n_trees * n_rows * 2 bytes; budget 2 GB.
        // leaf_nodes/leaf_samples are built per-tree on demand (cheap).
        let la_bytes = n_trees * n_rows * 2;
        let precomputed_la: Option<Vec<Vec<u16>>> = if la_bytes <= 2_000_000_000 {
            Some(
                self.trees
                    .par_iter()
                    .map(|tree| Self::route_tree_leaves(tree, binned, n_rows))
                    .collect(),
            )
        } else {
            None
        };

        // leaf_linear: use ALL numeric features for per-leaf ridge regression
        let (do_ramp, ramp_k) = if self.leaf_linear {
            let n_numeric: usize = (0..binned.n_features)
                .filter(|&f| !binned.is_categorical.get(f).copied().unwrap_or(false))
                .count();
            if n_numeric > 0 {
                (true, n_numeric)
            } else {
                (self.ramp, self.ramp_k)
            }
        } else {
            (self.ramp, self.ramp_k)
        };

        // Initialize/reinitialize ramp data (must handle tree growth from leaf splits)
        if do_ramp {
            // Build the all-numeric-features list for leaf_linear mode
            let all_numeric_features: Vec<u32> = if self.leaf_linear {
                (0..binned.n_features)
                    .filter(|&f| !binned.is_categorical.get(f).copied().unwrap_or(false))
                    .map(|f| f as u32)
                    .collect()
            } else {
                Vec::new()
            };

            for t in 0..n_trees {
                let n_nodes = self.trees[t].split_features.len();
                let expected_len = n_nodes * ramp_k;
                if self.trees[t].ramp_slopes.len() != expected_len {
                    self.trees[t].ramp_k = ramp_k;
                    if self.leaf_linear {
                        // Every node gets the same feature list: all numeric features
                        let mut feats = Vec::with_capacity(expected_len);
                        for _ in 0..n_nodes {
                            feats.extend_from_slice(&all_numeric_features);
                        }
                        self.trees[t].ramp_features = feats;
                    } else if ramp_k == 1 {
                        self.trees[t].ramp_features = self.trees[t].compute_parent_features();
                    } else {
                        self.trees[t].ramp_features = self.trees[t].compute_path_features_k(ramp_k);
                    }
                    self.trees[t].ramp_slopes.resize(expected_len, 0.0);
                }
            }
        }

        // leaf_quadratic: build pairwise interaction pairs and resize quad_slopes
        let skip_exhaustive_quadratic =
            self.auto_interactions && !self.numeric_interaction_pairs.is_empty() && n_rows >= 5_000;
        let interaction_pairs: Vec<(usize, usize)> =
            if self.leaf_quadratic && self.leaf_linear && !skip_exhaustive_quadratic {
                let numeric_feats: Vec<usize> = (0..binned.n_features)
                    .filter(|&f| !binned.is_categorical.get(f).copied().unwrap_or(false))
                    .collect();
                let mut pairs = Vec::new();
                for i in 0..numeric_feats.len() {
                    for j in (i + 1)..numeric_feats.len() {
                        pairs.push((numeric_feats[i], numeric_feats[j]));
                    }
                }
                pairs
            } else {
                Vec::new()
            };
        let n_interactions = interaction_pairs.len();
        let interaction_norms: Vec<f64> = interaction_pairs
            .iter()
            .map(|&(fi, fj)| {
                1.0 / (binned.n_bins(fi).max(1) as f64 * binned.n_bins(fj).max(1) as f64)
            })
            .collect();
        if n_interactions > 0 {
            for t in 0..n_trees {
                let n_nodes = self.trees[t].split_features.len();
                self.trees[t].quad_pairs = interaction_pairs.clone();
                self.trees[t].quad_n_interactions = n_interactions;
                self.trees[t]
                    .quad_slopes
                    .resize(n_nodes * n_interactions, 0.0);
            }
        }

        let is_binary = self.task == "binary";
        let l1 = self.l1_reg;
        let has_honest_masks = !self.tree_in_sample.is_empty();
        let max_alpha = 2.0; // line search cap (>1 allows correcting regularization undershoot)

        // Temp buffers for per-tree line search and structure refinement
        let max_nodes = self
            .trees
            .iter()
            .map(|t| t.split_features.len())
            .max()
            .unwrap_or(0);
        let mut opt_values = vec![0.0f64; max_nodes];
        let mut opt_slopes = vec![0.0f64; max_nodes * ramp_k];
        let mut opt_quad_slopes = vec![0.0f64; max_nodes * n_interactions];
        let mut tree_preds = vec![0.0f64; n_rows];
        let mut grad_buf = vec![0.0f64; n_rows];
        let mut hess_buf = vec![0.0f64; n_rows];
        let max_expert_k = ramp_k + n_interactions;
        let mut gx_buf = vec![0.0f64; max_expert_k];
        let mut hx_buf = vec![0.0f64; max_expert_k];
        let mut hxx_buf = vec![0.0f64; max_expert_k * max_expert_k];
        let mut x_all_buf = vec![0.0f64; max_expert_k];
        let mut x_bar_buf = vec![0.0f64; max_expert_k];
        let mut gx_c_buf = vec![0.0f64; max_expert_k];
        let mut a_buf = vec![0.0f64; max_expert_k * max_expert_k];
        let mut rhs_buf = vec![0.0f64; max_expert_k];
        let mut beta_buf = vec![0.0f64; max_expert_k];
        let mut chol_buf = vec![0.0f64; max_expert_k * max_expert_k];
        let mut solve_y_buf = vec![0.0f64; max_expert_k];
        let mut ramp_feats_buf = vec![0usize; ramp_k.max(1)];
        let mut ramp_nbins_buf = vec![1.0f64; ramp_k.max(1)];

        // Fix A: Pre-transpose feature matrix for leaf_linear mode (row-major for cache locality)
        // x_row_major[i * ramp_k + j] = normalized bin value for row i, feature j
        // This converts column-major bin_indices reads into sequential row-major reads.
        let ramp_feats_global: Vec<usize> = if self.leaf_linear && ramp_k > 0 {
            (0..binned.n_features)
                .filter(|&f| !binned.is_categorical.get(f).copied().unwrap_or(false))
                .collect()
        } else {
            Vec::new()
        };
        let ramp_nbins_global: Vec<f64> = ramp_feats_global
            .iter()
            .map(|&f| binned.n_bins(f).max(1) as f64)
            .collect();
        let x_row_major: Vec<f64> = if self.leaf_linear && ramp_k > 1 {
            let p = ramp_k;
            let mut buf = vec![0.0f64; n_rows * p];
            for j in 0..p {
                let feat = ramp_feats_global[j];
                let nb = ramp_nbins_global[j];
                let col_base = feat * n_rows;
                for i in 0..n_rows {
                    let bin = binned.bin_indices[col_base + i];
                    buf[i * p + j] = if bin != MISSING_BIN {
                        bin as f64 / nb
                    } else {
                        0.0
                    };
                }
            }
            buf
        } else {
            Vec::new()
        };
        let has_x_row_major = !x_row_major.is_empty();
        let x_bin_row_major: Vec<f64> = if self.leaf_linear && ramp_k > 0 {
            let p = ramp_k;
            let mut buf = vec![0.0f64; n_rows * p];
            for j in 0..p {
                let feat = ramp_feats_global[j];
                let col_base = feat * n_rows;
                for i in 0..n_rows {
                    let bin = binned.bin_indices[col_base + i];
                    buf[i * p + j] = if bin != MISSING_BIN { bin as f64 } else { 0.0 };
                }
            }
            buf
        } else {
            Vec::new()
        };
        let has_x_bin_row_major = !x_bin_row_major.is_empty();
        let mut feature_to_ramp_pos = vec![usize::MAX; binned.n_features];
        for (pos, &feat) in ramp_feats_global.iter().enumerate() {
            if feat < feature_to_ramp_pos.len() {
                feature_to_ramp_pos[feat] = pos;
            }
        }
        let interaction_ramp_positions: Vec<(usize, usize)> = interaction_pairs
            .iter()
            .map(|&(fi, fj)| {
                (
                    feature_to_ramp_pos.get(fi).copied().unwrap_or(usize::MAX),
                    feature_to_ramp_pos.get(fj).copied().unwrap_or(usize::MAX),
                )
            })
            .collect();
        let has_interaction_row_major = !interaction_ramp_positions.is_empty()
            && interaction_ramp_positions
                .iter()
                .all(|&(pi, pj)| pi != usize::MAX && pj != usize::MAX);

        let _refine_sharpness = 0.0f64;
        let _use_soft = false;

        // Initialize predictions from current tree values (always hard routing)
        let mut predictions = vec![self.base_score; n_rows];
        for t in 0..n_trees {
            let tree_t = &self.trees[t];
            let fast_ramp = has_x_bin_row_major
                && tree_t.leaf_pair_slopes.is_empty()
                && (tree_t.quad_slopes.is_empty() || has_interaction_row_major);
            let la_tmp;
            let la: &[u16] = match &precomputed_la {
                Some(all_la) => &all_la[t],
                None => {
                    la_tmp = Self::route_tree_leaves(tree_t, binned, n_rows);
                    &la_tmp
                }
            };
            for i in 0..n_rows {
                let leaf = la[i] as usize;
                let ramp = if fast_ramp {
                    let slope_base = leaf * ramp_k;
                    let row_base = i * ramp_k;
                    let mut s = 0.0f64;
                    for j in 0..ramp_k {
                        s += tree_t.ramp_slopes[slope_base + j] * x_bin_row_major[row_base + j];
                    }
                    s
                } else {
                    tree_t.ramp_predict(leaf, &binned.bin_indices, n_rows, i)
                };
                let quad = if fast_ramp && !tree_t.quad_slopes.is_empty() {
                    let qbase = leaf * n_interactions;
                    let row_base = i * ramp_k;
                    let mut s = 0.0f64;
                    for (jj, &(pi, pj)) in interaction_ramp_positions.iter().enumerate() {
                        s += tree_t.quad_slopes[qbase + jj]
                            * x_bin_row_major[row_base + pi]
                            * x_bin_row_major[row_base + pj];
                    }
                    s
                } else if fast_ramp {
                    0.0
                } else {
                    // Covered by ramp_predict when the fast path is unavailable.
                    0.0
                };
                predictions[i] += lr * (tree_t.values[leaf] + ramp + quad);
            }
        }

        // Stochastic refinement: use 80% of rows per pass for regularization
        let refine_subsample = 0.8f64;
        let use_stochastic = n_iters > 1 && n_rows > 100;
        let mut refine_mask = vec![true; n_rows];
        let mut refine_rng = StdRng::seed_from_u64(self.seed.wrapping_add(999));

        for _pass in 0..n_iters {
            // Generate stochastic mask for this pass
            if use_stochastic {
                for i in 0..n_rows {
                    refine_mask[i] = refine_rng.random::<f64>() < refine_subsample;
                }
            }
            for t in 0..n_trees {
                let honest_mask = if has_honest_masks && t < self.tree_in_sample.len() {
                    Some(&self.tree_in_sample[t])
                } else {
                    None
                };

                let n_nodes = self.trees[t].split_features.len();
                while opt_values.len() < n_nodes {
                    opt_values.push(0.0);
                }
                while opt_slopes.len() < n_nodes * ramp_k {
                    opt_slopes.push(0.0);
                }
                while opt_quad_slopes.len() < n_nodes * n_interactions {
                    opt_quad_slopes.push(0.0);
                }

                let la_tmp;
                let leaf_assign_t: &[u16] = match &precomputed_la {
                    Some(all_la) => &all_la[t],
                    None => {
                        la_tmp = Self::route_tree_leaves(&self.trees[t], binned, n_rows);
                        &la_tmp
                    }
                };

                // Remove tree t
                let use_par = n_rows >= 4096;
                let tree_t = &self.trees[t];
                let fast_remove_ramp = has_x_bin_row_major
                    && tree_t.leaf_pair_slopes.is_empty()
                    && (tree_t.quad_slopes.is_empty() || has_interaction_row_major);
                if use_par {
                    let bin_idx = &binned.bin_indices;
                    let xbr = &x_bin_row_major;
                    let irp = &interaction_ramp_positions;
                    predictions
                        .par_chunks_mut(1024)
                        .enumerate()
                        .for_each(|(ci, chunk)| {
                            let start = ci * 1024;
                            for (j, pred) in chunk.iter_mut().enumerate() {
                                let i = start + j;
                                let leaf = leaf_assign_t[i] as usize;
                                let ramp = if fast_remove_ramp {
                                    let slope_base = leaf * ramp_k;
                                    let row_base = i * ramp_k;
                                    let mut s = 0.0f64;
                                    for jj in 0..ramp_k {
                                        s += tree_t.ramp_slopes[slope_base + jj]
                                            * xbr[row_base + jj];
                                    }
                                    s
                                } else {
                                    tree_t.ramp_predict(leaf, bin_idx, n_rows, i)
                                };
                                let quad = if fast_remove_ramp && !tree_t.quad_slopes.is_empty() {
                                    let qbase = leaf * n_interactions;
                                    let row_base = i * ramp_k;
                                    let mut s = 0.0f64;
                                    for (jj, &(pi, pj)) in irp.iter().enumerate() {
                                        s += tree_t.quad_slopes[qbase + jj]
                                            * xbr[row_base + pi]
                                            * xbr[row_base + pj];
                                    }
                                    s
                                } else {
                                    0.0
                                };
                                *pred -= lr * (tree_t.values[leaf] + ramp + quad);
                            }
                        });
                } else {
                    for i in 0..n_rows {
                        let leaf = leaf_assign_t[i] as usize;
                        let ramp = if fast_remove_ramp {
                            let slope_base = leaf * ramp_k;
                            let row_base = i * ramp_k;
                            let mut s = 0.0f64;
                            for jj in 0..ramp_k {
                                s += tree_t.ramp_slopes[slope_base + jj]
                                    * x_bin_row_major[row_base + jj];
                            }
                            s
                        } else {
                            tree_t.ramp_predict(leaf, &binned.bin_indices, n_rows, i)
                        };
                        let quad = if fast_remove_ramp && !tree_t.quad_slopes.is_empty() {
                            let qbase = leaf * n_interactions;
                            let row_base = i * ramp_k;
                            let mut s = 0.0f64;
                            for (jj, &(pi, pj)) in interaction_ramp_positions.iter().enumerate() {
                                s += tree_t.quad_slopes[qbase + jj]
                                    * x_bin_row_major[row_base + pi]
                                    * x_bin_row_major[row_base + pj];
                            }
                            s
                        } else {
                            0.0
                        };
                        predictions[i] -= lr * (tree_t.values[leaf] + ramp + quad);
                    }
                }

                // Step 0: gradients
                if use_par {
                    if is_binary {
                        grad_buf
                            .par_chunks_mut(1024)
                            .zip(hess_buf.par_chunks_mut(1024))
                            .enumerate()
                            .for_each(|(ci, (g_chunk, h_chunk))| {
                                let start = ci * 1024;
                                for (j, (g, h)) in
                                    g_chunk.iter_mut().zip(h_chunk.iter_mut()).enumerate()
                                {
                                    let i = start + j;
                                    let p = 1.0 / (1.0 + (-predictions[i]).exp());
                                    *g = p - y[i];
                                    *h = (p * (1.0 - p)).max(1e-16);
                                }
                            });
                    } else {
                        grad_buf
                            .par_chunks_mut(1024)
                            .zip(hess_buf.par_chunks_mut(1024))
                            .enumerate()
                            .for_each(|(ci, (g_chunk, h_chunk))| {
                                let start = ci * 1024;
                                for (j, (g, h)) in
                                    g_chunk.iter_mut().zip(h_chunk.iter_mut()).enumerate()
                                {
                                    let i = start + j;
                                    *g = predictions[i] - y[i];
                                    *h = 1.0;
                                }
                            });
                    }
                } else {
                    if is_binary {
                        for i in 0..n_rows {
                            let p = 1.0 / (1.0 + (-predictions[i]).exp());
                            grad_buf[i] = p - y[i];
                            hess_buf[i] = (p * (1.0 - p)).max(1e-16);
                        }
                    } else {
                        for i in 0..n_rows {
                            grad_buf[i] = predictions[i] - y[i];
                            hess_buf[i] = 1.0;
                        }
                    }
                }

                // Build leaf info (always needed for Steps 2-5). The flat
                // representation avoids one heap allocation per leaf.
                let (leaf_nodes_t, leaf_offsets_t, leaf_samples_t) =
                    Self::build_leaf_info_flat(&self.trees[t], leaf_assign_t, n_rows);

                // Step 1: leaf optimization (hard gradient accumulation)
                {
                    for (local_j, &node_idx) in leaf_nodes_t.iter().enumerate() {
                        let samples =
                            &leaf_samples_t[leaf_offsets_t[local_j]..leaf_offsets_t[local_j + 1]];
                        for j in 0..ramp_k {
                            opt_slopes[node_idx * ramp_k + j] = 0.0;
                        }
                        for j in 0..n_interactions {
                            opt_quad_slopes[node_idx * n_interactions + j] = 0.0;
                        }
                        if samples.is_empty() {
                            opt_values[node_idx] = 0.0;
                            continue;
                        }

                        let ramp_base = node_idx * ramp_k;
                        let mut n_valid_ramp = 0usize;
                        if do_ramp && self.trees[t].cat_lookups[node_idx].is_none() {
                            if self.leaf_linear && ramp_feats_global.len() == ramp_k {
                                ramp_feats_buf[..ramp_k].copy_from_slice(&ramp_feats_global);
                                ramp_nbins_buf[..ramp_k].copy_from_slice(&ramp_nbins_global);
                                n_valid_ramp = ramp_k;
                            } else {
                                for j in 0..ramp_k {
                                    if ramp_base + j >= self.trees[t].ramp_features.len() {
                                        break;
                                    }
                                    let feat = self.trees[t].ramp_features[ramp_base + j];
                                    if feat == u32::MAX {
                                        continue;
                                    }
                                    let fu = feat as usize;
                                    if binned.is_categorical.get(fu).copied().unwrap_or(false) {
                                        continue;
                                    }
                                    ramp_feats_buf[n_valid_ramp] = fu;
                                    ramp_nbins_buf[n_valid_ramp] = binned.n_bins(fu).max(1) as f64;
                                    n_valid_ramp += 1;
                                }
                            }
                        }
                        let ramp_feats = &ramp_feats_buf[..n_valid_ramp];
                        let ramp_nbins = &ramp_nbins_buf[..n_valid_ramp];

                        let mut g_sum = 0.0f64;
                        let mut h_sum = 0.0f64;

                        let k_total = n_valid_ramp + n_interactions;

                        if n_valid_ramp <= 1 && n_interactions == 0 {
                            let use_ramp = n_valid_ramp == 1;
                            let feat0 = if use_ramp { ramp_feats[0] } else { 0 };
                            let nb0 = if use_ramp { ramp_nbins[0] } else { 1.0 };
                            let mut gx = 0.0f64;
                            let mut hx = 0.0f64;
                            let mut hxx = 0.0f64;
                            for &idx in samples {
                                let i = idx as usize;
                                if let Some(mask) = honest_mask {
                                    if bitvec_test(mask, i) {
                                        continue;
                                    }
                                }
                                if use_stochastic && !refine_mask[i] {
                                    continue;
                                }
                                let g = grad_buf[i];
                                let h = hess_buf[i];
                                g_sum += g;
                                h_sum += h;
                                if use_ramp {
                                    let bin = binned.bin_indices[feat0 * n_rows + i];
                                    let x = if bin != MISSING_BIN {
                                        bin as f64 / nb0
                                    } else {
                                        0.0
                                    };
                                    gx += g * x;
                                    hx += h * x;
                                    hxx += h * x * x;
                                }
                            }
                            if h_sum <= 0.0 {
                                opt_values[node_idx] = 0.0;
                                continue;
                            }
                            let w_raw = -g_sum / (h_sum + lambda);
                            let w_opt = if l1 > 0.0 {
                                let thr = l1 / (h_sum + lambda);
                                if w_raw > thr {
                                    w_raw - thr
                                } else if w_raw < -thr {
                                    w_raw + thr
                                } else {
                                    0.0
                                }
                            } else {
                                w_raw
                            };
                            if use_ramp {
                                let x_bar = hx / h_sum;
                                let gx_c = gx - g_sum * x_bar;
                                let hxx_c = hxx - hx * x_bar;
                                let beta_c = if hxx_c + ramp_lambda > 1e-12 {
                                    -gx_c / (hxx_c + ramp_lambda)
                                } else {
                                    0.0
                                };
                                opt_slopes[ramp_base] = beta_c / nb0;
                                opt_values[node_idx] = w_opt - beta_c * x_bar;
                            } else {
                                opt_values[node_idx] = w_opt;
                            }
                        } else if samples.len() < 3 {
                            // Need at least 3 samples for a meaningful linear fit
                            for &idx in samples {
                                let i = idx as usize;
                                if let Some(mask) = honest_mask {
                                    if bitvec_test(mask, i) {
                                        continue;
                                    }
                                }
                                if use_stochastic && !refine_mask[i] {
                                    continue;
                                }
                                g_sum += grad_buf[i];
                                h_sum += hess_buf[i];
                            }
                            if h_sum <= 0.0 {
                                opt_values[node_idx] = 0.0;
                                continue;
                            }
                            let w_raw = -g_sum / (h_sum + lambda);
                            opt_values[node_idx] = if l1 > 0.0 {
                                let thr = l1 / (h_sum + lambda);
                                if w_raw > thr {
                                    w_raw - thr
                                } else if w_raw < -thr {
                                    w_raw + thr
                                } else {
                                    0.0
                                }
                            } else {
                                w_raw
                            };
                        } else {
                            let k_linear = n_valid_ramp;
                            let k = k_total;

                            gx_buf[..k].fill(0.0);
                            hx_buf[..k].fill(0.0);
                            hxx_buf[..k * k].fill(0.0);
                            for &idx in samples {
                                let i = idx as usize;
                                if let Some(mask) = honest_mask {
                                    if bitvec_test(mask, i) {
                                        continue;
                                    }
                                }
                                if use_stochastic && !refine_mask[i] {
                                    continue;
                                }
                                let g = grad_buf[i];
                                let h = hess_buf[i];
                                g_sum += g;
                                h_sum += h;
                                // Linear features (normalized) — use row-major buffer when available
                                if has_x_row_major && k_linear == ramp_k {
                                    let row_base = i * ramp_k;
                                    x_all_buf[..k_linear].copy_from_slice(
                                        &x_row_major[row_base..row_base + k_linear],
                                    );
                                } else {
                                    for j in 0..k_linear {
                                        let bin = binned.bin_indices[ramp_feats[j] * n_rows + i];
                                        x_all_buf[j] = if bin != MISSING_BIN {
                                            bin as f64 / ramp_nbins[j]
                                        } else {
                                            0.0
                                        };
                                    }
                                }
                                // Interaction features (X_i * X_j normalized)
                                if has_interaction_row_major && has_x_row_major {
                                    let row_base = i * ramp_k;
                                    for (j, &(pi, pj)) in
                                        interaction_ramp_positions.iter().enumerate()
                                    {
                                        x_all_buf[k_linear + j] =
                                            x_row_major[row_base + pi] * x_row_major[row_base + pj];
                                    }
                                } else {
                                    for j in 0..n_interactions {
                                        let (fi, fj) = interaction_pairs[j];
                                        let bi = binned.bin_indices[fi * n_rows + i];
                                        let bj = binned.bin_indices[fj * n_rows + i];
                                        x_all_buf[k_linear + j] =
                                            if bi != MISSING_BIN && bj != MISSING_BIN {
                                                (bi as f64 * bj as f64) * interaction_norms[j]
                                            } else {
                                                0.0
                                            };
                                    }
                                }
                                let gx = &mut gx_buf[..k];
                                let hx = &mut hx_buf[..k];
                                let hxx = &mut hxx_buf[..k * k];
                                let x_all = &x_all_buf[..k];
                                // Per-sample rank-1 update of the (upper-triangle) hxx
                                // matrix plus gx/hx accumulators. Splitting the inner
                                // triangle write into a contiguous slice += scalar*slice
                                // lets LLVM emit a SIMD-fma kernel; the original index
                                // form blocked auto-vectorization because the loop
                                // body still spoke in terms of `hxx[j*k+l]` arithmetic.
                                for j in 0..k {
                                    let xj = x_all[j];
                                    gx[j] += g * xj;
                                    hx[j] += h * xj;
                                    let hxj = h * xj;
                                    let row_start = j * k + j;
                                    let hxx_row = &mut hxx[row_start..row_start + (k - j)];
                                    let x_slice = &x_all[j..k];
                                    for (cell, &xl) in hxx_row.iter_mut().zip(x_slice.iter()) {
                                        *cell += hxj * xl;
                                    }
                                }
                            }
                            if h_sum <= 0.0 {
                                opt_values[node_idx] = 0.0;
                                continue;
                            }
                            let w_raw = -g_sum / (h_sum + lambda);
                            let w_opt = if l1 > 0.0 {
                                let thr = l1 / (h_sum + lambda);
                                if w_raw > thr {
                                    w_raw - thr
                                } else if w_raw < -thr {
                                    w_raw + thr
                                } else {
                                    0.0
                                }
                            } else {
                                w_raw
                            };
                            let gx = &gx_buf[..k];
                            let hx = &hx_buf[..k];
                            let hxx = &mut hxx_buf[..k * k];
                            let x_bar = &mut x_bar_buf[..k];
                            let gx_c = &mut gx_c_buf[..k];
                            for j in 0..k {
                                x_bar[j] = hx[j] / h_sum;
                                gx_c[j] = gx[j] - g_sum * x_bar[j];
                            }
                            // Adaptive regularization: λ_eff = ramp_lambda * max(1, k/n_samples)
                            let n_eff_samples = samples.len().max(1) as f64;
                            let adaptive_ramp = ramp_lambda * (k as f64 / n_eff_samples).max(1.0);
                            // Mirror upper triangle of hxx to lower triangle
                            for j in 0..k {
                                for l in (j + 1)..k {
                                    hxx[l * k + j] = hxx[j * k + l];
                                }
                            }
                            let a = &mut a_buf[..k * k];
                            // Per-row vector form: a_row = hxx_row - hx[j] * x_bar. Slice
                            // ops auto-vectorize; the original index form did not.
                            for j in 0..k {
                                let hxj = hx[j];
                                let a_row = &mut a[j * k..(j + 1) * k];
                                let hxx_row = &hxx[j * k..(j + 1) * k];
                                for ((a_cell, &hxx_cell), &xb) in
                                    a_row.iter_mut().zip(hxx_row.iter()).zip(x_bar.iter())
                                {
                                    *a_cell = hxx_cell - hxj * xb;
                                }
                                a[j * k + j] += adaptive_ramp;
                            }
                            let rhs = &mut rhs_buf[..k];
                            for j in 0..k {
                                rhs[j] = -gx_c[j];
                            }
                            if !solve_spd_with_scratch(
                                k,
                                a,
                                rhs,
                                &mut chol_buf,
                                &mut solve_y_buf,
                                &mut beta_buf,
                            ) {
                                beta_buf[..k].fill(0.0);
                            }
                            let beta_c = &beta_buf[..k];
                            let mut intercept_adj = 0.0f64;
                            // Store linear slopes
                            for j in 0..k_linear {
                                let slope_j = beta_c[j] / ramp_nbins[j];
                                if self.leaf_linear && ramp_base + j < opt_slopes.len() {
                                    opt_slopes[ramp_base + j] = slope_j;
                                } else {
                                    for jj in 0..ramp_k {
                                        if ramp_base + jj < self.trees[t].ramp_features.len() {
                                            let feat = self.trees[t].ramp_features[ramp_base + jj];
                                            if feat != u32::MAX && feat as usize == ramp_feats[j] {
                                                opt_slopes[ramp_base + jj] = slope_j;
                                                break;
                                            }
                                        }
                                    }
                                }
                                intercept_adj += beta_c[j] * x_bar[j];
                            }
                            // Store interaction slopes
                            if n_interactions > 0 {
                                let qbase = node_idx * n_interactions;
                                for j in 0..n_interactions {
                                    // Convert from normalized to bin-index scale
                                    opt_quad_slopes[qbase + j] =
                                        beta_c[k_linear + j] * interaction_norms[j];
                                    intercept_adj += beta_c[k_linear + j] * x_bar[k_linear + j];
                                }
                            }
                            opt_values[node_idx] = w_opt - intercept_adj;
                        }
                    }
                } // end Step 1 if/else

                // Step 2: hard tree predictions
                if use_par {
                    let tree_t = &self.trees[t];
                    let bin_idx = &binned.bin_indices;
                    let xbr = &x_bin_row_major;
                    let ov = &opt_values;
                    let os = &opt_slopes;
                    let oq = &opt_quad_slopes;
                    let ip = &interaction_pairs;
                    let irp = &interaction_ramp_positions;
                    let ni = n_interactions;
                    let fast_linear_pred = has_x_bin_row_major && ni == 0;
                    let fast_quad_pred = has_x_bin_row_major && ni > 0 && has_interaction_row_major;
                    tree_preds
                        .par_chunks_mut(1024)
                        .enumerate()
                        .for_each(|(ci, chunk)| {
                            let start = ci * 1024;
                            for (j, tp) in chunk.iter_mut().enumerate() {
                                let i = start + j;
                                let leaf = leaf_assign_t[i] as usize;
                                let rp = if fast_linear_pred {
                                    let base = leaf * ramp_k;
                                    let row_base = i * ramp_k;
                                    let mut s = 0.0f64;
                                    for jj in 0..ramp_k {
                                        s += os[base + jj] * xbr[row_base + jj];
                                    }
                                    s
                                } else if do_ramp && tree_t.ramp_features.len() > leaf * ramp_k {
                                    let base = leaf * ramp_k;
                                    let mut s = 0.0f64;
                                    for jj in 0..ramp_k {
                                        let feat = tree_t.ramp_features[base + jj];
                                        if feat == u32::MAX {
                                            continue;
                                        }
                                        let bin = bin_idx[feat as usize * n_rows + i];
                                        if bin != MISSING_BIN {
                                            s += os[base + jj] * bin as f64;
                                        }
                                    }
                                    s
                                } else {
                                    0.0
                                };
                                let qp = if ni > 0 {
                                    let qbase = leaf * ni;
                                    let mut s = 0.0f64;
                                    if fast_quad_pred {
                                        let row_base = i * ramp_k;
                                        for (jj, &(pi, pj)) in irp.iter().enumerate() {
                                            s += oq[qbase + jj]
                                                * xbr[row_base + pi]
                                                * xbr[row_base + pj];
                                        }
                                    } else {
                                        for jj in 0..ni {
                                            let (fi, fj) = ip[jj];
                                            let bi = bin_idx[fi * n_rows + i];
                                            let bj = bin_idx[fj * n_rows + i];
                                            if bi != MISSING_BIN && bj != MISSING_BIN {
                                                s += oq[qbase + jj] * bi as f64 * bj as f64;
                                            }
                                        }
                                    }
                                    s
                                } else {
                                    0.0
                                };
                                *tp = ov[leaf] + rp + qp;
                            }
                        });
                } else {
                    let fast_linear_pred = has_x_bin_row_major && n_interactions == 0;
                    let fast_quad_pred =
                        has_x_bin_row_major && n_interactions > 0 && has_interaction_row_major;
                    for i in 0..n_rows {
                        let leaf = leaf_assign_t[i] as usize;
                        let rp = if fast_linear_pred {
                            let base = leaf * ramp_k;
                            let row_base = i * ramp_k;
                            let mut s = 0.0f64;
                            for j in 0..ramp_k {
                                s += opt_slopes[base + j] * x_bin_row_major[row_base + j];
                            }
                            s
                        } else if do_ramp && self.trees[t].ramp_features.len() > leaf * ramp_k {
                            let base = leaf * ramp_k;
                            let mut s = 0.0f64;
                            for j in 0..ramp_k {
                                let feat = self.trees[t].ramp_features[base + j];
                                if feat == u32::MAX {
                                    continue;
                                }
                                let bin = binned.bin_indices[feat as usize * n_rows + i];
                                if bin != MISSING_BIN {
                                    s += opt_slopes[base + j] * bin as f64;
                                }
                            }
                            s
                        } else {
                            0.0
                        };
                        let qp = if n_interactions > 0 {
                            let qbase = leaf * n_interactions;
                            let mut s = 0.0f64;
                            if fast_quad_pred {
                                let row_base = i * ramp_k;
                                for (j, &(pi, pj)) in interaction_ramp_positions.iter().enumerate()
                                {
                                    s += opt_quad_slopes[qbase + j]
                                        * x_bin_row_major[row_base + pi]
                                        * x_bin_row_major[row_base + pj];
                                }
                            } else {
                                for j in 0..n_interactions {
                                    let (fi, fj) = interaction_pairs[j];
                                    let bi = binned.bin_indices[fi * n_rows + i];
                                    let bj = binned.bin_indices[fj * n_rows + i];
                                    if bi != MISSING_BIN && bj != MISSING_BIN {
                                        s += opt_quad_slopes[qbase + j] * bi as f64 * bj as f64;
                                    }
                                }
                            }
                            s
                        } else {
                            0.0
                        };
                        tree_preds[i] = opt_values[leaf] + rp + qp;
                    }
                }

                // Step 3: line search
                let alpha_opt = if is_binary {
                    let mut alpha = 1.0f64;
                    for _ in 0..5 {
                        let (d1, d2) = if use_par {
                            let preds_ref = &predictions;
                            let tp_ref = &tree_preds;
                            let rm_ref = &refine_mask;
                            (0..((n_rows + 1023) / 1024))
                                .into_par_iter()
                                .map(|ci| {
                                    let start = ci * 1024;
                                    let end = (start + 1024).min(n_rows);
                                    let mut ld1 = 0.0f64;
                                    let mut ld2 = 0.0f64;
                                    for i in start..end {
                                        if let Some(mask) = honest_mask {
                                            if bitvec_test(mask, i) {
                                                continue;
                                            }
                                        }
                                        if use_stochastic && !rm_ref[i] {
                                            continue;
                                        }
                                        let z = preds_ref[i] + lr * alpha * tp_ref[i];
                                        let p = 1.0 / (1.0 + (-z).exp());
                                        let f = lr * tp_ref[i];
                                        ld1 += f * (p - y[i]);
                                        ld2 += f * f * (p * (1.0 - p)).max(1e-16);
                                    }
                                    (ld1, ld2)
                                })
                                .reduce(|| (0.0, 0.0), |(a1, a2), (b1, b2)| (a1 + b1, a2 + b2))
                        } else {
                            let mut d1 = 0.0f64;
                            let mut d2 = 0.0f64;
                            for i in 0..n_rows {
                                if let Some(mask) = honest_mask {
                                    if bitvec_test(mask, i) {
                                        continue;
                                    }
                                }
                                if use_stochastic && !refine_mask[i] {
                                    continue;
                                }
                                let z = predictions[i] + lr * alpha * tree_preds[i];
                                let p = 1.0 / (1.0 + (-z).exp());
                                let f = lr * tree_preds[i];
                                d1 += f * (p - y[i]);
                                d2 += f * f * (p * (1.0 - p)).max(1e-16);
                            }
                            (d1, d2)
                        };
                        if d2.abs() < 1e-30 {
                            break;
                        }
                        alpha -= d1 / d2;
                        alpha = alpha.clamp(0.0, max_alpha);
                    }
                    alpha
                } else {
                    let (dot_rf, dot_ff) = if use_par {
                        let preds_ref = &predictions;
                        let tp_ref = &tree_preds;
                        let rm_ref = &refine_mask;
                        (0..((n_rows + 1023) / 1024))
                            .into_par_iter()
                            .map(|ci| {
                                let start = ci * 1024;
                                let end = (start + 1024).min(n_rows);
                                let mut lrf = 0.0f64;
                                let mut lff = 0.0f64;
                                for i in start..end {
                                    if let Some(mask) = honest_mask {
                                        if bitvec_test(mask, i) {
                                            continue;
                                        }
                                    }
                                    if use_stochastic && !rm_ref[i] {
                                        continue;
                                    }
                                    let r = preds_ref[i] - y[i];
                                    let f = tp_ref[i];
                                    lrf += r * f;
                                    lff += f * f;
                                }
                                (lrf, lff)
                            })
                            .reduce(|| (0.0, 0.0), |(a1, a2), (b1, b2)| (a1 + b1, a2 + b2))
                    } else {
                        let mut dot_rf = 0.0f64;
                        let mut dot_ff = 0.0f64;
                        for i in 0..n_rows {
                            if let Some(mask) = honest_mask {
                                if bitvec_test(mask, i) {
                                    continue;
                                }
                            }
                            if use_stochastic && !refine_mask[i] {
                                continue;
                            }
                            let r = predictions[i] - y[i];
                            let f = tree_preds[i];
                            dot_rf += r * f;
                            dot_ff += f * f;
                        }
                        (dot_rf, dot_ff)
                    };
                    if dot_ff > 1e-30 {
                        (-dot_rf / (lr * dot_ff)).clamp(0.0, max_alpha)
                    } else {
                        1.0
                    }
                };

                // Step 4: apply alpha
                for &node_idx in leaf_nodes_t.iter() {
                    self.trees[t].values[node_idx] = alpha_opt * opt_values[node_idx];
                    if do_ramp {
                        let base = node_idx * ramp_k;
                        for j in 0..ramp_k {
                            if base + j < self.trees[t].ramp_slopes.len() {
                                self.trees[t].ramp_slopes[base + j] =
                                    alpha_opt * opt_slopes[base + j];
                            }
                        }
                    }
                    if n_interactions > 0 {
                        let qbase = node_idx * n_interactions;
                        for j in 0..n_interactions {
                            if qbase + j < self.trees[t].quad_slopes.len() {
                                self.trees[t].quad_slopes[qbase + j] =
                                    alpha_opt * opt_quad_slopes[qbase + j];
                            }
                        }
                    }
                }
                // Step 5: add back
                if use_par {
                    let tp_ref = &tree_preds;
                    let lr_alpha = lr * alpha_opt;
                    predictions
                        .par_chunks_mut(1024)
                        .enumerate()
                        .for_each(|(ci, chunk)| {
                            let start = ci * 1024;
                            for (j, pred) in chunk.iter_mut().enumerate() {
                                *pred += lr_alpha * tp_ref[start + j];
                            }
                        });
                } else {
                    for i in 0..n_rows {
                        predictions[i] += lr * alpha_opt * tree_preds[i];
                    }
                }
            } // end for t
        } // end for _pass
    }

    pub(super) fn prune_similar_leaves(&mut self) {
        if self.trees.is_empty() {
            return;
        }
        let lambda = self.lambda_reg.max(1e-12);
        let z2_threshold = (0.75 + 4.0 * self.split_pessimism + self.gamma.max(0.0)).min(4.0);
        for tree in self.trees.iter_mut() {
            if !tree.split_features.is_empty() {
                Self::prune_similar_leaves_tree(tree, 0, lambda, z2_threshold);
            }
        }
    }

    fn prune_similar_leaves_tree(
        tree: &mut DecisionTree,
        node: usize,
        lambda: f64,
        z2_threshold: f64,
    ) -> bool {
        if node >= tree.split_features.len() || tree.split_features[node] == u32::MAX {
            return true;
        }
        let left = tree.left_children.get(node).copied().unwrap_or(0) as usize;
        let right = tree.right_children.get(node).copied().unwrap_or(0) as usize;
        if left >= tree.split_features.len() || right >= tree.split_features.len() {
            return false;
        }
        let left_leaf = Self::prune_similar_leaves_tree(tree, left, lambda, z2_threshold);
        let right_leaf = Self::prune_similar_leaves_tree(tree, right, lambda, z2_threshold);
        if !(left_leaf && right_leaf) {
            return false;
        }

        let lh = tree.node_h_sum.get(left).copied().unwrap_or(0.0).max(0.0);
        let rh = tree.node_h_sum.get(right).copied().unwrap_or(0.0).max(0.0);
        if lh <= 0.0 || rh <= 0.0 {
            return false;
        }
        let lv = tree.values.get(left).copied().unwrap_or(0.0);
        let rv = tree.values.get(right).copied().unwrap_or(0.0);
        if !(lv.is_finite() && rv.is_finite()) {
            return false;
        }
        let se2 = (1.0 / (lh + lambda) + 1.0 / (rh + lambda)).max(1e-12);
        let z2 = (lv - rv) * (lv - rv) / se2;
        if !(z2.is_finite() && z2 <= z2_threshold) {
            return false;
        }

        let total_h = (lh + rh).max(1e-12);
        tree.values[node] = (lv * lh + rv * rh) / total_h;
        tree.split_features[node] = u32::MAX;
        tree.left_children[node] = 0;
        tree.right_children[node] = 0;
        tree.is_cat_split[node] = false;
        tree.is_oblique_split[node] = false;
        if node < tree.cat_lookups.len() {
            tree.cat_lookups[node] = None;
        }
        if node < tree.node_h_sum.len() {
            tree.node_h_sum[node] = total_h;
        }
        if node < tree.node_count.len() {
            let lc = tree.node_count.get(left).copied().unwrap_or(0);
            let rc = tree.node_count.get(right).copied().unwrap_or(0);
            tree.node_count[node] = lc.saturating_add(rc);
        }
        if tree.ramp_k > 0 && !tree.ramp_slopes.is_empty() {
            let start = node.saturating_mul(tree.ramp_k);
            let end = (start + tree.ramp_k).min(tree.ramp_slopes.len());
            for v in tree.ramp_slopes[start..end].iter_mut() {
                *v = 0.0;
            }
        }
        true
    }

    /// Global leaf optimization for multiclass: backfitting with cached routing.
    /// Honest-aware: uses complement data when tree_in_sample masks are available.
    pub(super) fn refine_global_multiclass(
        &mut self,
        binned: &BinnedData,
        y: &[f64],
        n_rows: usize,
        n_classes: usize,
        n_iters: usize,
    ) {
        let lr = self.learning_rate;
        let lambda = self.lambda_reg;
        let ramp_lambda = self.ramp_lambda;
        let n_trees = self.trees.len();

        let la_bytes = n_trees * n_rows * 2;
        let precomputed_la: Option<Vec<Vec<u16>>> = if la_bytes <= 2_000_000_000 {
            Some(
                self.trees
                    .par_iter()
                    .map(|tree| Self::route_tree_leaves(tree, binned, n_rows))
                    .collect(),
            )
        } else {
            None
        };

        // leaf_linear: set up ramp features for all trees (same as refine_global)
        let (do_ramp, ramp_k) = if self.leaf_linear {
            let n_numeric: usize = (0..binned.n_features)
                .filter(|&f| !binned.is_categorical.get(f).copied().unwrap_or(false))
                .count();
            if n_numeric > 0 {
                (true, n_numeric)
            } else {
                (false, 0)
            }
        } else {
            (false, 0)
        };

        if do_ramp {
            let all_numeric_features: Vec<u32> = (0..binned.n_features)
                .filter(|&f| !binned.is_categorical.get(f).copied().unwrap_or(false))
                .map(|f| f as u32)
                .collect();
            for t in 0..n_trees {
                let n_nodes = self.trees[t].split_features.len();
                let expected_len = n_nodes * ramp_k;
                if self.trees[t].ramp_slopes.len() != expected_len {
                    self.trees[t].ramp_k = ramp_k;
                    let mut feats = Vec::with_capacity(expected_len);
                    for _ in 0..n_nodes {
                        feats.extend_from_slice(&all_numeric_features);
                    }
                    self.trees[t].ramp_features = feats;
                    self.trees[t].ramp_slopes.resize(expected_len, 0.0);
                }
            }
        }

        // Initialize predictions (including ramp slopes)
        let mut predictions = vec![0.0f64; n_rows * n_classes];
        if self.class_base_scores.len() == n_classes {
            for i in 0..n_rows {
                let base = i * n_classes;
                predictions[base..base + n_classes].copy_from_slice(&self.class_base_scores);
            }
        }
        for t in 0..n_trees {
            let class_k = (t / self.multiclass_trees_per_class_round()) % n_classes;
            let la_tmp;
            let la: &[u16] = match &precomputed_la {
                Some(all_la) => &all_la[t],
                None => {
                    la_tmp = Self::route_tree_leaves(&self.trees[t], binned, n_rows);
                    &la_tmp
                }
            };
            if n_rows >= 4096 {
                let tree_t = &self.trees[t];
                let bin_idx = &binned.bin_indices;
                let nc = n_classes;
                let ck = class_k;
                predictions
                    .par_chunks_mut(nc * 256)
                    .enumerate()
                    .for_each(|(ci, chunk)| {
                        let start_row = ci * 256;
                        let n_chunk_rows = chunk.len() / nc;
                        for r in 0..n_chunk_rows {
                            let i = start_row + r;
                            let leaf = la[i] as usize;
                            chunk[r * nc + ck] += lr
                                * (tree_t.values[leaf]
                                    + tree_t.ramp_predict(leaf, bin_idx, n_rows, i));
                        }
                    });
            } else {
                for i in 0..n_rows {
                    let leaf = la[i] as usize;
                    predictions[i * n_classes + class_k] += lr
                        * (self.trees[t].values[leaf]
                            + self.trees[t].ramp_predict(leaf, &binned.bin_indices, n_rows, i));
                }
            }
        }
        let mut probs = vec![0.0f64; n_rows * n_classes];
        let l1 = self.l1_reg;
        let has_honest_masks = !self.tree_in_sample.is_empty();
        let _refine_sharpness = 0.0f64;
        let _use_soft = false;
        let max_nodes = self
            .trees
            .iter()
            .map(|t| t.split_features.len())
            .max()
            .unwrap_or(0);
        let mut grad_buf = vec![0.0f64; n_rows];
        let mut hess_buf = vec![0.0f64; n_rows];
        let _refine_mask_dummy = Vec::<bool>::new();
        let mut opt_values = vec![0.0f64; max_nodes];
        let mut opt_slopes = if do_ramp {
            vec![0.0f64; max_nodes * ramp_k]
        } else {
            Vec::new()
        };

        // Fix A: Pre-transpose feature matrix for leaf_linear mode (row-major for cache locality)
        let mc_ramp_feats_global: Vec<usize> = if self.leaf_linear && ramp_k > 0 {
            (0..binned.n_features)
                .filter(|&f| !binned.is_categorical.get(f).copied().unwrap_or(false))
                .collect()
        } else {
            Vec::new()
        };
        let mc_ramp_nbins_global: Vec<f64> = mc_ramp_feats_global
            .iter()
            .map(|&f| binned.n_bins(f).max(1) as f64)
            .collect();
        let mc_x_row_major: Vec<f64> = if self.leaf_linear && ramp_k > 1 {
            let p = ramp_k;
            let mut buf = vec![0.0f64; n_rows * p];
            for j in 0..p {
                let feat = mc_ramp_feats_global[j];
                let nb = mc_ramp_nbins_global[j];
                let col_base = feat * n_rows;
                for i in 0..n_rows {
                    let bin = binned.bin_indices[col_base + i];
                    buf[i * p + j] = if bin != MISSING_BIN {
                        bin as f64 / nb
                    } else {
                        0.0
                    };
                }
            }
            buf
        } else {
            Vec::new()
        };
        let mc_has_x_row = !mc_x_row_major.is_empty();

        for _pass in 0..n_iters {
            for t in 0..n_trees {
                let class_k = (t / self.multiclass_trees_per_class_round()) % n_classes;

                let la_tmp;
                let leaf_assign_t: &[u16] = match &precomputed_la {
                    Some(all_la) => &all_la[t],
                    None => {
                        la_tmp = Self::route_tree_leaves(&self.trees[t], binned, n_rows);
                        &la_tmp
                    }
                };
                // Remove tree t (including ramp slopes)
                let use_par = n_rows >= 4096;
                if use_par {
                    let tree_t = &self.trees[t];
                    let bin_idx = &binned.bin_indices;
                    let nc = n_classes;
                    let ck = class_k;
                    predictions
                        .par_chunks_mut(nc * 256)
                        .enumerate()
                        .for_each(|(ci, chunk)| {
                            let start_row = ci * 256;
                            let n_chunk_rows = chunk.len() / nc;
                            for r in 0..n_chunk_rows {
                                let i = start_row + r;
                                let leaf = leaf_assign_t[i] as usize;
                                chunk[r * nc + ck] -= lr
                                    * (tree_t.values[leaf]
                                        + tree_t.ramp_predict(leaf, bin_idx, n_rows, i));
                            }
                        });
                } else {
                    for i in 0..n_rows {
                        let leaf = leaf_assign_t[i] as usize;
                        predictions[i * n_classes + class_k] -= lr
                            * (self.trees[t].values[leaf]
                                + self.trees[t].ramp_predict(leaf, &binned.bin_indices, n_rows, i));
                    }
                }

                Self::compute_softmax_par(&predictions, &mut probs, n_rows, n_classes);

                let honest_mask = if has_honest_masks && t < self.tree_in_sample.len() {
                    Some(&self.tree_in_sample[t])
                } else {
                    None
                };

                // Compute per-row gradients for this class
                if use_par {
                    let probs_ref = &probs;
                    let nc = n_classes;
                    let ck = class_k;
                    grad_buf
                        .par_chunks_mut(1024)
                        .zip(hess_buf.par_chunks_mut(1024))
                        .enumerate()
                        .for_each(|(ci, (g_chunk, h_chunk))| {
                            let start = ci * 1024;
                            for (j, (g, h)) in
                                g_chunk.iter_mut().zip(h_chunk.iter_mut()).enumerate()
                            {
                                let i = start + j;
                                let label = if y[i] as usize == ck { 1.0 } else { 0.0 };
                                let pk = probs_ref[i * nc + ck];
                                *g = pk - label;
                                *h = (pk * (1.0 - pk)).max(1e-16);
                            }
                        });
                } else {
                    for i in 0..n_rows {
                        let label = if y[i] as usize == class_k { 1.0 } else { 0.0 };
                        grad_buf[i] = probs[i * n_classes + class_k] - label;
                        hess_buf[i] = (probs[i * n_classes + class_k]
                            * (1.0 - probs[i * n_classes + class_k]))
                            .max(1e-16);
                    }
                }

                {
                    let (leaf_nodes_t, leaf_samples_t) =
                        Self::build_leaf_info(&self.trees[t], leaf_assign_t, n_rows);
                    // Ensure buffers are large enough for this tree's node indices
                    let n_nodes_t = self.trees[t].split_features.len();
                    while opt_values.len() < n_nodes_t {
                        opt_values.push(0.0);
                    }
                    if do_ramp {
                        while opt_slopes.len() < n_nodes_t * ramp_k {
                            opt_slopes.push(0.0);
                        }
                    }
                    for (local_j, &node_idx) in leaf_nodes_t.iter().enumerate() {
                        let samples = &leaf_samples_t[local_j];
                        if samples.is_empty() {
                            opt_values[node_idx] = 0.0;
                            continue;
                        }
                        if do_ramp {
                            for j in 0..ramp_k {
                                opt_slopes[node_idx * ramp_k + j] = 0.0;
                            }
                        }

                        let ramp_base = node_idx * ramp_k;
                        let mut n_valid_ramp = 0usize;
                        let mut ramp_feats: Vec<usize> = vec![0; ramp_k];
                        let mut ramp_nbins: Vec<f64> = vec![1.0; ramp_k];
                        if do_ramp {
                            for j in 0..ramp_k {
                                if ramp_base + j >= self.trees[t].ramp_features.len() {
                                    break;
                                }
                                let feat = self.trees[t].ramp_features[ramp_base + j];
                                if feat == u32::MAX {
                                    continue;
                                }
                                let fu = feat as usize;
                                if binned.is_categorical.get(fu).copied().unwrap_or(false) {
                                    continue;
                                }
                                ramp_feats[n_valid_ramp] = fu;
                                ramp_nbins[n_valid_ramp] = binned.n_bins(fu).max(1) as f64;
                                n_valid_ramp += 1;
                            }
                        }

                        let mut g_sum = 0.0f64;
                        let mut h_sum = 0.0f64;

                        if n_valid_ramp >= 2 && samples.len() >= 2 * n_valid_ramp {
                            // Full ridge regression (same as refine_global binary/regression path)
                            let k = n_valid_ramp;
                            let mut gx = vec![0.0f64; k];
                            let mut hx = vec![0.0f64; k];
                            let mut hxx = vec![0.0f64; k * k];
                            let mut x_buf = vec![0.0f64; k];
                            for &idx in samples {
                                let i = idx as usize;
                                if let Some(mask) = honest_mask {
                                    if bitvec_test(mask, i) {
                                        continue;
                                    }
                                }
                                let g = grad_buf[i];
                                let h = hess_buf[i];
                                g_sum += g;
                                h_sum += h;
                                // Use row-major buffer when available
                                if mc_has_x_row && k == ramp_k {
                                    let row_base = i * ramp_k;
                                    x_buf[..k]
                                        .copy_from_slice(&mc_x_row_major[row_base..row_base + k]);
                                } else {
                                    for j in 0..k {
                                        let bin = binned.bin_indices[ramp_feats[j] * n_rows + i];
                                        x_buf[j] = if bin != MISSING_BIN {
                                            bin as f64 / ramp_nbins[j]
                                        } else {
                                            0.0
                                        };
                                    }
                                }
                                for j in 0..k {
                                    gx[j] += g * x_buf[j];
                                    let hxj = h * x_buf[j];
                                    hx[j] += hxj;
                                    for l in j..k {
                                        hxx[j * k + l] += hxj * x_buf[l];
                                    }
                                }
                            }
                            if h_sum <= 0.0 {
                                opt_values[node_idx] = 0.0;
                                continue;
                            }
                            let w_raw = -g_sum / (h_sum + lambda);
                            let w_opt = if l1 > 0.0 {
                                let thr = l1 / (h_sum + lambda);
                                if w_raw > thr {
                                    w_raw - thr
                                } else if w_raw < -thr {
                                    w_raw + thr
                                } else {
                                    0.0
                                }
                            } else {
                                w_raw
                            };
                            let x_bar: Vec<f64> = (0..k).map(|j| hx[j] / h_sum).collect();
                            let gx_c: Vec<f64> = (0..k).map(|j| gx[j] - g_sum * x_bar[j]).collect();
                            // Adaptive regularization: λ_eff = ramp_lambda * max(1, k/n_samples)
                            let n_eff_samples = samples.len().max(1) as f64;
                            let adaptive_ramp = ramp_lambda * (k as f64 / n_eff_samples).max(1.0);
                            // Mirror upper triangle of hxx to lower triangle
                            for j in 0..k {
                                for l in (j + 1)..k {
                                    hxx[l * k + j] = hxx[j * k + l];
                                }
                            }
                            let mut a = vec![0.0f64; k * k];
                            for j in 0..k {
                                for l in 0..k {
                                    a[j * k + l] = hxx[j * k + l] - hx[j] * x_bar[l];
                                }
                                a[j * k + j] += adaptive_ramp;
                            }
                            let rhs: Vec<f64> = gx_c.iter().map(|v| -v).collect();
                            let beta_c = solve_spd(k, &a, &rhs);
                            let mut intercept_adj = 0.0f64;
                            for j in 0..k {
                                let slope_j = beta_c[j] / ramp_nbins[j];
                                for jj in 0..ramp_k {
                                    if ramp_base + jj < self.trees[t].ramp_features.len() {
                                        let feat = self.trees[t].ramp_features[ramp_base + jj];
                                        if feat != u32::MAX && feat as usize == ramp_feats[j] {
                                            opt_slopes[ramp_base + jj] = slope_j;
                                            break;
                                        }
                                    }
                                }
                                intercept_adj += beta_c[j] * x_bar[j];
                            }
                            opt_values[node_idx] = w_opt - intercept_adj;
                        } else {
                            // Constant leaf (not enough samples for ridge)
                            for &idx in samples {
                                let i = idx as usize;
                                if let Some(mask) = honest_mask {
                                    if bitvec_test(mask, i) {
                                        continue;
                                    }
                                }
                                g_sum += grad_buf[i];
                                h_sum += hess_buf[i];
                            }
                            if h_sum <= 0.0 {
                                opt_values[node_idx] = 0.0;
                                continue;
                            }
                            let w_raw = -g_sum / (h_sum + lambda);
                            opt_values[node_idx] = if l1 > 0.0 {
                                let thr = l1 / (h_sum + lambda);
                                if w_raw > thr {
                                    w_raw - thr
                                } else if w_raw < -thr {
                                    w_raw + thr
                                } else {
                                    0.0
                                }
                            } else {
                                w_raw
                            };
                        }
                    }

                    // Apply optimized values with blending
                    for (local_j, &node_idx) in leaf_nodes_t.iter().enumerate() {
                        let w_old = self.trees[t].values[node_idx];
                        self.trees[t].values[node_idx] =
                            w_old + self.refine_alpha * (opt_values[node_idx] - w_old);
                        if do_ramp {
                            let base = node_idx * ramp_k;
                            for j in 0..ramp_k {
                                if base + j < self.trees[t].ramp_slopes.len() {
                                    let s_old = self.trees[t].ramp_slopes[base + j];
                                    self.trees[t].ramp_slopes[base + j] =
                                        s_old + self.refine_alpha * (opt_slopes[base + j] - s_old);
                                }
                            }
                        }
                    }
                }

                // Add tree t back (including ramp slopes)
                if use_par {
                    let tree_t = &self.trees[t];
                    let bin_idx = &binned.bin_indices;
                    let nc = n_classes;
                    let ck = class_k;
                    predictions
                        .par_chunks_mut(nc * 256)
                        .enumerate()
                        .for_each(|(ci, chunk)| {
                            let start_row = ci * 256;
                            let n_chunk_rows = chunk.len() / nc;
                            for r in 0..n_chunk_rows {
                                let i = start_row + r;
                                let leaf = leaf_assign_t[i] as usize;
                                chunk[r * nc + ck] += lr
                                    * (tree_t.values[leaf]
                                        + tree_t.ramp_predict(leaf, bin_idx, n_rows, i));
                            }
                        });
                } else {
                    for i in 0..n_rows {
                        let leaf = leaf_assign_t[i] as usize;
                        predictions[i * n_classes + class_k] += lr
                            * (self.trees[t].values[leaf]
                                + self.trees[t].ramp_predict(leaf, &binned.bin_indices, n_rows, i));
                    }
                }
            }
        }
    }
}
