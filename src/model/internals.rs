//! Boosting algorithm internals.
//!
//! All non-public methods of `GTBoostModel` live here, organized by role:
//!
//! - **Setup helpers**: `extend_raw_matrix`, `posterior_leaf_tau`,
//!   `multiclass_trees_per_class_round`, `multiclass_tree_lr`,
//!   `finalize_multiclass_tree`, pool-based subsampling.
//! - **Training**: `fit_single` (binary/regression), `fit_multiclass`
//!   (K-class softmax with optional shared multi-output trees),
//!   `build_ordered_multioutput_round`.
//! - **Eval / softmax / warmup**: per-round eval loss, softmax variants
//!   (incl. parallel and temperature-scaled), warmup-tree pre-pass.
//! - **Gradients & subsampling**: `compute_gradients_hessians`, GOSS
//!   (`goss_select`, importance scoring), random-subsample utilities.
//! - **Per-tree adjustments**: HSS, EBLP, SCS, NTR, hierarchical,
//!   sibling-block correction, cyclic-pressure scoring (CIPA / SCGB),
//!   adaptive root anchor, feature-mask machinery.
//! - **Multiclass coupling**: `compute_multiclass_coupled_node_values`,
//!   guided lookup-table (CLL) choice/joint variants for shared trees.
//! - **Leaf finalization**: `leaf_split_pass*`, `route_tree_leaves`,
//!   `build_leaf_info*`, expert-leaf admission (VCEG-0).
//!
//! Cross-module callers in `model/mod.rs` (the `#[pymethods]` block) and
//! `model/refine.rs` reach these via `pub(super)` visibility.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::RngExt;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::HashMap;

use super::GTBoostModel;
use crate::helpers::{bitvec_test, solve_small_linear_system, solve_spd};
use crate::tree::{BinnedData, CatTupleConfig, DecisionTree, MISSING_BIN};

impl GTBoostModel {
    fn median_sorted(values: &mut [f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = values.len();
        if n & 1 == 1 {
            values[n / 2]
        } else {
            0.5 * (values[n / 2 - 1] + values[n / 2])
        }
    }

    /// Robust, scale-free Huber delta for regression residuals.
    ///
    /// Negative `huber_delta` enables this path.  The scale estimate is the
    /// Gaussian-consistent MAD of current residuals, and the Huber cutoff uses
    /// the classical 95%-efficient normal-theory constant.  For very large data
    /// a deterministic stride sample keeps the per-round cost bounded while
    /// preserving determinism.
    fn adaptive_huber_delta(y: &[f64], preds: &[f64]) -> f64 {
        const MAX_SCALE_SAMPLE: usize = 16_384;
        const NORMAL_MAD_SCALE: f64 = 1.4826;
        const HUBER_95_EFFICIENCY: f64 = 1.345;

        let n = y.len().min(preds.len());
        if n == 0 {
            return 0.0;
        }
        let stride = n.div_ceil(MAX_SCALE_SAMPLE).max(1);
        let mut residuals = Vec::with_capacity(n.div_ceil(stride));
        let mut i = 0usize;
        while i < n {
            let r = preds[i] - y[i];
            if r.is_finite() {
                residuals.push(r);
            }
            i += stride;
        }
        if residuals.len() < 4 {
            return 0.0;
        }

        let center = Self::median_sorted(&mut residuals);
        for r in residuals.iter_mut() {
            *r = (*r - center).abs();
        }
        let mad = Self::median_sorted(&mut residuals);
        let sigma = NORMAL_MAD_SCALE * mad;
        if sigma.is_finite() && sigma > 1e-12 {
            HUBER_95_EFFICIENCY * sigma
        } else {
            0.0
        }
    }

    pub(super) fn cat_tuple_config(&self, binned: &BinnedData) -> Option<CatTupleConfig> {
        if !self.cat_tuple_lookups || self.cat_lookup_smooth <= 0.0 || self.task == "regression" {
            return None;
        }
        let n_cat = (0..binned.n_features)
            .filter(|&c| {
                c < binned.cll_is_categorical.len()
                    && binned.cll_is_categorical[c]
                    && c < binned.cll_n_bins.len()
                    && binned.cll_n_bins[c] >= 2
            })
            .count();
        if n_cat < 2 {
            return None;
        }
        Some(CatTupleConfig {
            enabled: true,
            max_order: self.cat_tuple_max_order,
            top_features: self.cat_tuple_top_features,
            hash_bins: self.cat_tuple_hash_bins,
            min_leaf: self.cat_tuple_min_leaf,
            gain_margin: self.cat_tuple_gain_margin,
        })
    }

    pub(super) fn extend_raw_matrix(
        &self,
        x_data_raw: &[f64],
        n_rows: usize,
        n_features_raw: usize,
    ) -> (Vec<f64>, usize) {
        let mut extra_cols: Vec<Vec<f64>> = Vec::new();

        if !self.numeric_interaction_pairs.is_empty() {
            for &(fi, fj) in &self.numeric_interaction_pairs {
                let col: Vec<f64> = (0..n_rows)
                    .map(|row| {
                        let vi = x_data_raw[row * n_features_raw + fi];
                        let vj = x_data_raw[row * n_features_raw + fj];
                        if vi.is_nan() || vj.is_nan() {
                            f64::NAN
                        } else {
                            vi * vj
                        }
                    })
                    .collect();
                extra_cols.push(col);
            }
        }

        if !self.categorical_interaction_pairs.is_empty() {
            for &(fi, fj) in &self.categorical_interaction_pairs {
                let col: Vec<f64> = (0..n_rows)
                    .map(|row| {
                        let vi = x_data_raw[row * n_features_raw + fi];
                        let vj = x_data_raw[row * n_features_raw + fj];
                        if vi.is_nan() || vj.is_nan() {
                            f64::NAN
                        } else {
                            let hi = vi as i64;
                            let hj = vj as i64;
                            let h =
                                ((hi.wrapping_mul(1_000_003)) ^ (hj.wrapping_mul(1_000_033))) & 255;
                            h as f64
                        }
                    })
                    .collect();
                extra_cols.push(col);
            }
        }

        let ordered_ctr_cols = self.ordered_ctr_columns_for_raw(x_data_raw, n_rows, n_features_raw);
        if !ordered_ctr_cols.is_empty() {
            extra_cols.extend(ordered_ctr_cols);
        }

        let cfe_cols = self.cat_fold_evidence_columns_for_raw(x_data_raw, n_rows, n_features_raw);
        if !cfe_cols.is_empty() {
            extra_cols.extend(cfe_cols);
        }
        let resid_cols = self.cfe_residual_columns_for_raw(x_data_raw, n_rows, n_features_raw);
        if !resid_cols.is_empty() {
            extra_cols.extend(resid_cols);
        }

        if extra_cols.is_empty() {
            return (x_data_raw.to_vec(), n_features_raw);
        }

        let n_extra = extra_cols.len();
        let n_total = n_features_raw + n_extra;
        let mut extended = vec![0.0f64; n_rows * n_total];
        for row in 0..n_rows {
            extended[row * n_total..row * n_total + n_features_raw]
                .copy_from_slice(&x_data_raw[row * n_features_raw..(row + 1) * n_features_raw]);
            for (ci, col) in extra_cols.iter().enumerate() {
                extended[row * n_total + n_features_raw + ci] = col[row];
            }
        }
        (extended, n_total)
    }

    #[inline]
    pub(super) fn raw_matrix_extensions_active(&self) -> bool {
        !self.numeric_interaction_pairs.is_empty()
            || !self.categorical_interaction_pairs.is_empty()
            || !self.ordered_ctr_features.is_empty()
            || !self.ordered_ctr_pair_features.is_empty()
            || !self.ordered_ctr_triple_features.is_empty()
            || !self.cfe_tuples.is_empty()
            || !self.cfe_resid_tables.is_empty()
    }

    pub(super) fn apply_corrective_block_refit(
        &mut self,
        binned: &BinnedData,
        x_data_raw: &[f64],
        n_rows: usize,
        n_features_raw: usize,
        y: &[f64],
        init_score: Option<&[f64]>,
        eval_guard: Option<(&[f64], &[f64], usize)>,
    ) {
        if !self.corrective_block_refit
            || !(self.task == "regression" || self.task == "binary")
            || n_rows == 0
            || y.len() != n_rows
            || self.learning_rate.abs() <= 1e-15
            || self.trees.len() < self.corrective_min_trees
            || self.dart_rate > 0.0
        {
            return;
        }

        let n_trees = self.trees.len();
        let n_blocks = self.corrective_blocks.min(n_trees).max(1);
        if n_blocks == 0 || n_blocks > n_rows.saturating_sub(1).max(1) {
            return;
        }

        let (x_data, n_features) = self.extend_raw_matrix(x_data_raw, n_rows, n_features_raw);
        if n_features != binned.n_features || x_data.len() != n_rows * n_features {
            return;
        }

        let mut block_of_tree = vec![0usize; n_trees];
        let mut block_starts = Vec::with_capacity(n_blocks);
        let mut block_ends = Vec::with_capacity(n_blocks);
        for block in 0..n_blocks {
            let start = block * n_trees / n_blocks;
            let end = ((block + 1) * n_trees / n_blocks)
                .max(start + 1)
                .min(n_trees);
            block_starts.push(start);
            block_ends.push(end);
            for t in start..end {
                block_of_tree[t] = block;
            }
        }

        let mut phi = vec![0.0f64; n_rows * n_blocks];
        let mut target = vec![0.0f64; n_rows];
        let mut offsets = vec![0.0f64; n_rows];
        for row in 0..n_rows {
            let row_data = &x_data[row * n_features..(row + 1) * n_features];
            let offset = init_score.map(|s| s[row]).unwrap_or(self.base_score);
            offsets[row] = offset;
            target[row] = y[row] - offset;
            let mut running_score = offset;
            for (t_idx, tree) in self.trees.iter().enumerate() {
                let c = if tree.has_self_score_splits() {
                    tree.predict_raw_row_with_score(binned, row_data, running_score)
                } else {
                    tree.predict_raw_row(binned, row_data)
                };
                let contribution = self.learning_rate * c;
                let block = block_of_tree[t_idx];
                phi[row * n_blocks + block] += contribution;
                running_score += contribution;
            }
        }

        let eval_phi_offsets = eval_guard.and_then(|(eval_x_raw, eval_y, eval_n_rows)| {
            if eval_n_rows == 0
                || eval_y.len() != eval_n_rows
                || eval_x_raw.len() != eval_n_rows.saturating_mul(n_features_raw)
            {
                return None;
            }
            let (eval_x, eval_n_features) =
                self.extend_raw_matrix(eval_x_raw, eval_n_rows, n_features_raw);
            if eval_n_features != binned.n_features
                || eval_x.len() != eval_n_rows.saturating_mul(eval_n_features)
            {
                return None;
            }
            let mut eval_phi = vec![0.0f64; eval_n_rows * n_blocks];
            let mut eval_offsets = vec![0.0f64; eval_n_rows];
            for row in 0..eval_n_rows {
                let row_data = &eval_x[row * eval_n_features..(row + 1) * eval_n_features];
                let offset = self.base_score;
                eval_offsets[row] = offset;
                let mut running_score = offset;
                for (t_idx, tree) in self.trees.iter().enumerate() {
                    let c = if tree.has_self_score_splits() {
                        tree.predict_raw_row_with_score(binned, row_data, running_score)
                    } else {
                        tree.predict_raw_row(binned, row_data)
                    };
                    let contribution = self.learning_rate * c;
                    let block = block_of_tree[t_idx];
                    eval_phi[row * n_blocks + block] += contribution;
                    running_score += contribution;
                }
            }
            Some((eval_phi, eval_offsets, eval_y, eval_n_rows))
        });

        if self.task == "binary" {
            let logistic_loss = |yi: f64, margin: f64| -> Option<f64> {
                let z = margin.clamp(-50.0, 50.0);
                if yi > 0.5 {
                    Some((1.0 + (-z).exp()).ln())
                } else {
                    Some((1.0 + z.exp()).ln())
                }
            };

            let score_rows = |row_filter: &dyn Fn(usize) -> bool, weights: &[f64]| -> Option<f64> {
                let mut loss = 0.0f64;
                let mut count = 0usize;
                for row in 0..n_rows {
                    if !row_filter(row) {
                        continue;
                    }
                    let row_phi = &phi[row * n_blocks..(row + 1) * n_blocks];
                    let mut margin = offsets[row];
                    for block in 0..n_blocks {
                        margin += weights[block] * row_phi[block];
                    }
                    loss += logistic_loss(y[row], margin)?;
                    count += 1;
                }
                if count == 0 || !loss.is_finite() {
                    None
                } else {
                    Some(loss)
                }
            };

            let solve_on_rows = |row_filter: &dyn Fn(usize) -> bool| -> Option<Vec<f64>> {
                let mut theta = vec![1.0f64; n_blocks];
                let mut fit_count = 0usize;
                let mut base_energy = vec![0.0f64; n_blocks];
                for row in 0..n_rows {
                    if !row_filter(row) {
                        continue;
                    }
                    fit_count += 1;
                    let row_phi = &phi[row * n_blocks..(row + 1) * n_blocks];
                    for block in 0..n_blocks {
                        let v = row_phi[block];
                        if !v.is_finite() {
                            return None;
                        }
                        base_energy[block] += v * v;
                    }
                }
                if fit_count <= n_blocks {
                    return None;
                }
                let mean_energy =
                    base_energy.iter().sum::<f64>() / (n_blocks as f64 * fit_count as f64);
                if !mean_energy.is_finite() || mean_energy <= 1e-18 {
                    return None;
                }
                let ridge = self.corrective_lambda * mean_energy.max(1e-12);

                for _ in 0..12 {
                    let mut gram = vec![0.0f64; n_blocks * n_blocks];
                    let mut rhs = vec![0.0f64; n_blocks];
                    for row in 0..n_rows {
                        if !row_filter(row) {
                            continue;
                        }
                        let row_phi = &phi[row * n_blocks..(row + 1) * n_blocks];
                        let mut margin = offsets[row];
                        for block in 0..n_blocks {
                            margin += theta[block] * row_phi[block];
                        }
                        let p = 1.0 / (1.0 + (-margin.clamp(-50.0, 50.0)).exp());
                        let err = p - y[row];
                        let w = (p * (1.0 - p)).max(1e-8);
                        for i in 0..n_blocks {
                            let vi = row_phi[i];
                            rhs[i] -= vi * err;
                            for j in 0..=i {
                                gram[i * n_blocks + j] += w * vi * row_phi[j];
                            }
                        }
                    }
                    for i in 0..n_blocks {
                        rhs[i] -= ridge * (theta[i] - 1.0);
                        gram[i * n_blocks + i] += ridge;
                        for j in 0..i {
                            gram[j * n_blocks + i] = gram[i * n_blocks + j];
                        }
                    }
                    let step = solve_spd(n_blocks, &gram, &rhs);
                    if step.len() != n_blocks || step.iter().any(|v| !v.is_finite()) {
                        return None;
                    }
                    let mut max_abs_step = 0.0f64;
                    for block in 0..n_blocks {
                        let delta = step[block].clamp(-1.0, 1.0);
                        theta[block] = (theta[block] + delta).clamp(-5.0, 5.0);
                        max_abs_step = max_abs_step.max(delta.abs());
                    }
                    if max_abs_step < 1e-5 {
                        break;
                    }
                }
                Some(theta)
            };

            let audit_fraction = self.corrective_audit_fraction.clamp(0.0, 0.5);
            let mut fold_of_row = vec![usize::MAX; n_rows];
            let mut audit_folds = 0usize;
            let mut use_audit = false;
            if audit_fraction > 0.0 && n_rows >= 80 && n_blocks + 2 < n_rows {
                audit_folds = ((1.0 / audit_fraction).round() as usize).clamp(2, 10);
                let mut fold_counts = vec![0usize; audit_folds];
                let split_seed = self.seed ^ 0xB10C_C0A7_E57B_1A5Eu64;
                for row in 0..n_rows {
                    let mut z = (row as u64)
                        .wrapping_add(0x9E37_79B9_7F4A_7C15u64)
                        .wrapping_add(split_seed);
                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9u64);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EBu64);
                    z ^= z >> 31;
                    let fold = (z as usize) % audit_folds;
                    fold_of_row[row] = fold;
                    fold_counts[fold] += 1;
                }
                let min_selector = (n_blocks + 2).max(32);
                let min_fold = 24usize;
                let max_fold = fold_counts.iter().copied().max().unwrap_or(0);
                if fold_counts.iter().all(|&c| c >= min_fold)
                    && n_rows.saturating_sub(max_fold) >= min_selector
                {
                    use_audit = true;
                } else {
                    audit_folds = 0;
                }
            }

            let blend = self.corrective_blend;
            if use_audit {
                let ones = vec![1.0f64; n_blocks];
                let mut base_loss = 0.0f64;
                let mut refit_loss = 0.0f64;
                for fold in 0..audit_folds {
                    let coef_fold = match solve_on_rows(&|row| fold_of_row[row] != fold) {
                        Some(c) => c,
                        None => return,
                    };
                    let mut fold_weights = vec![1.0f64; n_blocks];
                    for block in 0..n_blocks {
                        let c = coef_fold[block].clamp(-5.0, 5.0);
                        fold_weights[block] = 1.0 + blend * (c - 1.0);
                    }
                    base_loss += match score_rows(&|row| fold_of_row[row] == fold, &ones) {
                        Some(v) => v,
                        None => return,
                    };
                    refit_loss += match score_rows(&|row| fold_of_row[row] == fold, &fold_weights) {
                        Some(v) => v,
                        None => return,
                    };
                }
                if !base_loss.is_finite() || !refit_loss.is_finite() || base_loss <= 1e-24 {
                    return;
                }
                let required = 1.0 - self.corrective_min_rel_improve;
                let bic_penalty = 0.5 * n_blocks as f64 * (n_rows.max(2) as f64).ln();
                if refit_loss + bic_penalty > base_loss * required {
                    return;
                }
            }

            let coef = match solve_on_rows(&|_| true) {
                Some(c) => c,
                None => return,
            };
            let mut block_weights = vec![1.0f64; n_blocks];
            for block in 0..n_blocks {
                let c = coef[block].clamp(-5.0, 5.0);
                block_weights[block] = 1.0 + blend * (c - 1.0);
            }
            if let Some((ref eval_phi, ref eval_offsets, eval_y, eval_n_rows)) = eval_phi_offsets {
                let eval_score = |weights: &[f64]| -> Option<f64> {
                    let mut loss = 0.0f64;
                    for row in 0..eval_n_rows {
                        let row_phi = &eval_phi[row * n_blocks..(row + 1) * n_blocks];
                        let mut margin = eval_offsets[row];
                        for block in 0..n_blocks {
                            margin += weights[block] * row_phi[block];
                        }
                        loss += logistic_loss(eval_y[row], margin)?;
                    }
                    if loss.is_finite() {
                        Some(loss)
                    } else {
                        None
                    }
                };
                let ones = vec![1.0f64; n_blocks];
                let base_eval = match eval_score(&ones) {
                    Some(v) => v,
                    None => return,
                };
                let refit_eval = match eval_score(&block_weights) {
                    Some(v) => v,
                    None => return,
                };
                let bic_penalty = 0.5 * n_blocks as f64 * (eval_n_rows.max(2) as f64).ln();
                if refit_eval + bic_penalty > base_eval * (1.0 - self.corrective_min_rel_improve) {
                    return;
                }
            }
            let mut weights = vec![1.0f64; n_trees];
            for block in 0..n_blocks {
                for t in block_starts[block]..block_ends[block] {
                    weights[t] = block_weights[block];
                }
            }
            self.dart_tree_weights = weights;
            return;
        }

        let solve_on_rows = |row_filter: &dyn Fn(usize) -> bool| -> Option<Vec<f64>> {
            let mut gram = vec![0.0f64; n_blocks * n_blocks];
            let mut rhs = vec![0.0f64; n_blocks];
            let mut fit_count = 0usize;
            for row in 0..n_rows {
                if !row_filter(row) {
                    continue;
                }
                let row_phi = &phi[row * n_blocks..(row + 1) * n_blocks];
                let t = target[row];
                if !t.is_finite() {
                    return None;
                }
                fit_count += 1;
                for i in 0..n_blocks {
                    let vi = row_phi[i];
                    if !vi.is_finite() {
                        return None;
                    }
                    rhs[i] += vi * t;
                    for j in 0..=i {
                        gram[i * n_blocks + j] += vi * row_phi[j];
                    }
                }
            }
            if fit_count <= n_blocks {
                return None;
            }
            for i in 0..n_blocks {
                for j in 0..i {
                    gram[j * n_blocks + i] = gram[i * n_blocks + j];
                }
            }

            let mean_energy =
                (0..n_blocks).map(|i| gram[i * n_blocks + i]).sum::<f64>() / n_blocks as f64;
            if !mean_energy.is_finite() || mean_energy <= 1e-18 {
                return None;
            }
            let ridge = self.corrective_lambda * mean_energy + 1e-12 * mean_energy;
            for i in 0..n_blocks {
                gram[i * n_blocks + i] += ridge;
            }

            let coef = solve_spd(n_blocks, &gram, &rhs);
            if coef.len() != n_blocks || coef.iter().any(|v| !v.is_finite()) {
                return None;
            }
            Some(coef)
        };

        let audit_fraction = self.corrective_audit_fraction.clamp(0.0, 0.5);
        let mut fold_of_row = vec![usize::MAX; n_rows];
        let mut audit_folds = 0usize;
        let mut use_audit = false;
        if audit_fraction > 0.0 && n_rows >= 80 && n_blocks + 2 < n_rows {
            audit_folds = ((1.0 / audit_fraction).round() as usize).clamp(2, 10);
            let mut fold_counts = vec![0usize; audit_folds];
            let split_seed = self.seed ^ 0xC0B1_0C0A_7E57_5EEDu64;
            for row in 0..n_rows {
                let mut z = (row as u64)
                    .wrapping_add(0x9E37_79B9_7F4A_7C15u64)
                    .wrapping_add(split_seed);
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9u64);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EBu64);
                z ^= z >> 31;
                let fold = (z as usize) % audit_folds;
                fold_of_row[row] = fold;
                fold_counts[fold] += 1;
            }
            let min_selector = (n_blocks + 2).max(32);
            let min_fold = 24usize;
            let max_fold = fold_counts.iter().copied().max().unwrap_or(0);
            if fold_counts.iter().all(|&c| c >= min_fold)
                && n_rows.saturating_sub(max_fold) >= min_selector
            {
                use_audit = true;
            } else {
                audit_folds = 0;
            }
        }

        let blend = self.corrective_blend;
        if use_audit {
            let mut base_sse = 0.0f64;
            let mut refit_sse = 0.0f64;
            let mut checked = 0usize;
            for fold in 0..audit_folds {
                let coef_fold = match solve_on_rows(&|row| fold_of_row[row] != fold) {
                    Some(c) => c,
                    None => return,
                };
                let mut fold_weights = vec![1.0f64; n_blocks];
                for block in 0..n_blocks {
                    let c = coef_fold[block].clamp(-5.0, 5.0);
                    fold_weights[block] = 1.0 + blend * (c - 1.0);
                }
                let mut fold_checked = 0usize;
                for row in 0..n_rows {
                    if fold_of_row[row] != fold {
                        continue;
                    }
                    let row_phi = &phi[row * n_blocks..(row + 1) * n_blocks];
                    let t = target[row];
                    if !t.is_finite() {
                        return;
                    }
                    let mut base_pred = 0.0f64;
                    let mut refit_pred = 0.0f64;
                    for block in 0..n_blocks {
                        let v = row_phi[block];
                        if !v.is_finite() {
                            return;
                        }
                        base_pred += v;
                        refit_pred += fold_weights[block] * v;
                    }
                    let rb = t - base_pred;
                    let rr = t - refit_pred;
                    base_sse += rb * rb;
                    refit_sse += rr * rr;
                    checked += 1;
                    fold_checked += 1;
                }
                if fold_checked < 24 {
                    return;
                }
            }
            if checked < 24 || !base_sse.is_finite() || !refit_sse.is_finite() || base_sse <= 1e-24
            {
                return;
            }
            let required = 1.0 - self.corrective_min_rel_improve;
            let checked_f = checked.max(2) as f64;
            let bic_delta = checked_f * (refit_sse / base_sse).max(1e-30).ln()
                + n_blocks as f64 * checked_f.ln();
            if refit_sse > base_sse * required || bic_delta > 0.0 {
                return;
            }
        }

        let coef = match solve_on_rows(&|_| true) {
            Some(c) => c,
            None => return,
        };
        let mut block_weights = vec![1.0f64; n_blocks];
        for block in 0..n_blocks {
            let c = coef[block].clamp(-5.0, 5.0);
            block_weights[block] = 1.0 + blend * (c - 1.0);
        }
        if let Some((ref eval_phi, ref eval_offsets, eval_y, eval_n_rows)) = eval_phi_offsets {
            let eval_sse = |weights: &[f64]| -> Option<f64> {
                let mut sse = 0.0f64;
                for row in 0..eval_n_rows {
                    let row_phi = &eval_phi[row * n_blocks..(row + 1) * n_blocks];
                    let mut pred = eval_offsets[row];
                    for block in 0..n_blocks {
                        pred += weights[block] * row_phi[block];
                    }
                    let r = eval_y[row] - pred;
                    sse += r * r;
                }
                if sse.is_finite() {
                    Some(sse)
                } else {
                    None
                }
            };
            let ones = vec![1.0f64; n_blocks];
            let base_eval = match eval_sse(&ones) {
                Some(v) => v,
                None => return,
            };
            let refit_eval = match eval_sse(&block_weights) {
                Some(v) => v,
                None => return,
            };
            let eval_n = eval_n_rows.max(2) as f64;
            let bic_delta = eval_n * (refit_eval / base_eval.max(1e-30)).max(1e-30).ln()
                + n_blocks as f64 * eval_n.ln();
            if refit_eval > base_eval * (1.0 - self.corrective_min_rel_improve) || bic_delta > 0.0 {
                return;
            }
        }

        let mut weights = vec![1.0f64; n_trees];
        for block in 0..n_blocks {
            for t in block_starts[block]..block_ends[block] {
                weights[t] = block_weights[block];
            }
        }
        self.dart_tree_weights = weights;
    }

    #[inline]
    fn ctr_single_key(v: f64) -> i64 {
        if v.is_nan() {
            i64::MIN
        } else {
            v as i64
        }
    }

    #[inline]
    fn ctr_pair_key(a: i64, b: i64) -> i64 {
        let mut x = (a as u64).wrapping_mul(0x9E37_79B1_85EB_CA87)
            ^ (b as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        x ^= x >> 33;
        x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        x ^= x >> 33;
        x as i64
    }

    #[inline]
    fn ctr_triple_key(a: i64, b: i64, c: i64) -> i64 {
        Self::ctr_pair_key(Self::ctr_pair_key(a, b), c)
    }

    fn ctr_stats(keys: &[i64], y: &[f64]) -> HashMap<i64, (usize, f64)> {
        let mut stats: HashMap<i64, (usize, f64)> = HashMap::new();
        for (&key, &yy) in keys.iter().zip(y.iter()) {
            let entry = stats.entry(key).or_insert((0, 0.0));
            entry.0 += 1;
            entry.1 += yy;
        }
        stats
    }

    fn ctr_score(
        stats: &HashMap<i64, (usize, f64)>,
        prior: f64,
        smooth: f64,
        min_count: usize,
    ) -> f64 {
        let mut total = 0usize;
        let mut score = 0.0;
        for &(cnt, sum) in stats.values() {
            if cnt < min_count {
                continue;
            }
            let enc = (sum + smooth * prior) / (cnt as f64 + smooth);
            score += cnt as f64 * (enc - prior) * (enc - prior);
            total += cnt;
        }
        if total >= min_count * 2 && score.is_finite() {
            score
        } else {
            0.0
        }
    }

    #[inline]
    fn ctr_clip_prob(p: f64) -> f64 {
        p.clamp(1e-6, 1.0 - 1e-6)
    }

    #[inline]
    fn ctr_binary_logloss(sum_y: f64, cnt: usize, p: f64) -> f64 {
        let pp = Self::ctr_clip_prob(p);
        let n = cnt as f64;
        -(sum_y * pp.ln() + (n - sum_y) * (1.0 - pp).ln())
    }

    fn ctr_logloss_score(
        stats: &HashMap<i64, (usize, f64)>,
        prior: f64,
        smooth: f64,
        min_count: usize,
    ) -> f64 {
        let mut total = 0usize;
        let mut score = 0.0;
        for &(cnt, sum) in stats.values() {
            if cnt < min_count {
                continue;
            }
            let enc = (sum + smooth * prior) / (cnt as f64 + smooth);
            let base_loss = Self::ctr_binary_logloss(sum, cnt, prior);
            let enc_loss = Self::ctr_binary_logloss(sum, cnt, enc);
            score += base_loss - enc_loss;
            total += cnt;
        }
        if total >= min_count * 2 && score.is_finite() {
            score
        } else {
            0.0
        }
    }

    fn ctr_multiclass_logloss_score(
        keys: &[i64],
        y: &[f64],
        n_classes: usize,
        priors: &[f64],
        smooth: f64,
        min_count: usize,
    ) -> f64 {
        if n_classes < 2 || keys.is_empty() || keys.len() != y.len() {
            return 0.0;
        }
        let mut stats: HashMap<i64, (usize, Vec<f64>)> = HashMap::new();
        for (&key, &label) in keys.iter().zip(y.iter()) {
            if !label.is_finite() || label < 0.0 {
                continue;
            }
            let cls = label.round() as usize;
            if cls >= n_classes || (label - cls as f64).abs() > 1e-6 {
                continue;
            }
            let entry = stats
                .entry(key)
                .or_insert_with(|| (0usize, vec![0.0; n_classes]));
            entry.0 += 1;
            entry.1[cls] += 1.0;
        }
        if stats.len() < 2 {
            return 0.0;
        }

        let mut total = 0usize;
        let mut score = 0.0;
        for &(cnt, ref counts) in stats.values() {
            if cnt < min_count {
                continue;
            }
            let mut base_loss = 0.0;
            let mut enc_loss = 0.0;
            for k in 0..n_classes {
                let class_count = counts[k];
                if class_count <= 0.0 {
                    continue;
                }
                let prior = Self::ctr_clip_prob(priors.get(k).copied().unwrap_or(0.0));
                let enc =
                    Self::ctr_clip_prob((class_count + smooth * prior) / (cnt as f64 + smooth));
                base_loss -= class_count * prior.ln();
                enc_loss -= class_count * enc.ln();
            }
            score += base_loss - enc_loss;
            total += cnt;
        }
        if total >= min_count * 2 && score.is_finite() {
            score
        } else {
            0.0
        }
    }

    fn ctr_oof_columns(
        keys: &[i64],
        y: &[f64],
        n_rows: usize,
        prior: f64,
        smooth: f64,
        n_perm: usize,
        rng: &mut StdRng,
    ) -> (Vec<f64>, Vec<f64>, HashMap<i64, f64>, HashMap<i64, f64>) {
        let full_stats = Self::ctr_stats(keys, y);
        let mut enc_col = vec![0.0; n_rows];
        let mut count_col = vec![0.0; n_rows];
        let n_ctr_folds = 5usize.min(n_rows).max(2);

        for _ in 0..n_perm {
            let mut order: Vec<usize> = (0..n_rows).collect();
            order.shuffle(rng);
            for fold in 0..n_ctr_folds {
                let start = fold * n_rows / n_ctr_folds;
                let end = (fold + 1) * n_rows / n_ctr_folds;
                let holdout = &order[start..end];
                let mut hold_stats: HashMap<i64, (usize, f64)> = HashMap::new();
                for &row in holdout {
                    let entry = hold_stats.entry(keys[row]).or_insert((0, 0.0));
                    entry.0 += 1;
                    entry.1 += y[row];
                }
                for &row in holdout {
                    let key = keys[row];
                    let (fc, fs) = full_stats.get(&key).copied().unwrap_or((0, 0.0));
                    let (hc, hs) = hold_stats.get(&key).copied().unwrap_or((0, 0.0));
                    let cnt = fc.saturating_sub(hc);
                    let sum = fs - hs;
                    enc_col[row] += (sum + smooth * prior) / (cnt as f64 + smooth);
                    count_col[row] += (cnt as f64).ln_1p();
                }
            }
        }

        if n_perm > 1 {
            let denom = n_perm as f64;
            for row in 0..n_rows {
                enc_col[row] /= denom;
                count_col[row] /= denom;
            }
        }

        let enc_map = full_stats
            .iter()
            .map(|(&key, &(cnt, sum))| (key, (sum + smooth * prior) / (cnt as f64 + smooth)))
            .collect();
        let count_map = full_stats
            .iter()
            .map(|(&key, &(cnt, _))| (key, (cnt as f64).ln_1p()))
            .collect();

        (enc_col, count_col, enc_map, count_map)
    }

    pub(super) fn build_ordered_ctr_features(
        &mut self,
        x_data: &[f64],
        y: &[f64],
        n_rows: usize,
        n_features: usize,
    ) -> Vec<Vec<f64>> {
        self.ordered_ctr_features.clear();
        self.ordered_ctr_priors.clear();
        self.ordered_ctr_maps.clear();
        self.ordered_ctr_count_maps.clear();
        self.ordered_ctr_pair_features.clear();
        self.ordered_ctr_pair_maps.clear();
        self.ordered_ctr_pair_count_maps.clear();
        self.ordered_ctr_triple_features.clear();
        self.ordered_ctr_triple_maps.clear();
        self.ordered_ctr_triple_count_maps.clear();
        self.ordered_ctr_prior = 0.0;

        if !self.ordered_ctr || n_rows == 0 || n_features == 0 || self.ordered_ctr_top_features == 0
        {
            return Vec::new();
        }

        let min_count = self.ordered_ctr_min_count.max(1);
        let pair_expected_min = self.ordered_ctr_min_count.max(5) as f64;
        let smooth = self.ordered_ctr_smooth.max(1e-12);
        let mut scored: Vec<(usize, f64, Vec<i64>)> = Vec::new();

        if self.task == "multiclass" {
            let n_classes = y
                .iter()
                .filter(|&&label| label.is_finite() && label >= 0.0)
                .map(|&label| label.round() as usize)
                .max()
                .unwrap_or(0)
                + 1;
            if n_classes < 2 {
                return Vec::new();
            }
            let mut class_counts = vec![0.0; n_classes];
            let mut valid_rows = 0usize;
            for &label in y {
                if !label.is_finite() || label < 0.0 {
                    continue;
                }
                let cls = label.round() as usize;
                if cls < n_classes && (label - cls as f64).abs() <= 1e-6 {
                    class_counts[cls] += 1.0;
                    valid_rows += 1;
                }
            }
            if valid_rows == 0 {
                return Vec::new();
            }
            let priors: Vec<f64> = class_counts
                .iter()
                .map(|&cnt| Self::ctr_clip_prob(cnt / valid_rows as f64))
                .collect();
            self.ordered_ctr_prior = 1.0 / n_classes as f64;

            for feat in 0..n_features {
                if feat >= self.cat_features.len() || !self.cat_features[feat] {
                    continue;
                }
                let keys: Vec<i64> = (0..n_rows)
                    .map(|row| Self::ctr_single_key(x_data[row * n_features + feat]))
                    .collect();
                let score = Self::ctr_multiclass_logloss_score(
                    &keys, y, n_classes, &priors, smooth, min_count,
                );
                if score > 0.0 {
                    scored.push((feat, score, keys));
                }
            }

            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            scored.truncate(self.ordered_ctr_top_features.min(scored.len()));
            if scored.is_empty() {
                return Vec::new();
            }

            let mut rng = StdRng::seed_from_u64(self.seed ^ 0xC7A5_CE57_u64);
            let n_perm = self.ordered_ctr_permutations.max(1);
            let mut train_cols: Vec<Vec<f64>> = Vec::new();
            for (feat, _score, keys) in &scored {
                for (class_idx, &prior) in priors.iter().enumerate() {
                    let y_binary: Vec<f64> = y
                        .iter()
                        .map(|&label| {
                            if label.is_finite()
                                && label >= 0.0
                                && (label.round() as usize) == class_idx
                            {
                                1.0
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    let (enc, count, enc_map, count_map) = Self::ctr_oof_columns(
                        keys, &y_binary, n_rows, prior, smooth, n_perm, &mut rng,
                    );
                    train_cols.push(enc);
                    train_cols.push(count);
                    self.ordered_ctr_features.push(*feat);
                    self.ordered_ctr_priors.push(prior);
                    self.ordered_ctr_maps.push(enc_map);
                    self.ordered_ctr_count_maps.push(count_map);
                }
            }

            return train_cols;
        }

        let prior = y.iter().sum::<f64>() / n_rows as f64;
        self.ordered_ctr_prior = prior;

        for feat in 0..n_features {
            if feat >= self.cat_features.len() || !self.cat_features[feat] {
                continue;
            }
            let keys: Vec<i64> = (0..n_rows)
                .map(|row| Self::ctr_single_key(x_data[row * n_features + feat]))
                .collect();
            let stats = Self::ctr_stats(&keys, y);
            if stats.len() < 2 {
                continue;
            }
            let score = if self.task == "binary" {
                Self::ctr_logloss_score(&stats, prior, smooth, min_count)
            } else {
                Self::ctr_score(&stats, prior, smooth, min_count)
            };
            if score > 0.0 {
                scored.push((feat, score, keys));
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(self.ordered_ctr_top_features.min(scored.len()));
        if scored.is_empty() {
            return Vec::new();
        }

        let selected: Vec<usize> = scored.iter().map(|(feat, _, _)| *feat).collect();
        let mut rng = StdRng::seed_from_u64(self.seed ^ 0xC7A5_CE57_u64);
        let n_perm = self.ordered_ctr_permutations.max(1);
        let mut train_cols: Vec<Vec<f64>> = Vec::new();
        let mut single_oof_cols: Vec<Vec<f64>> = Vec::new();

        for (_feat, _score, keys) in &scored {
            let (enc, count, enc_map, count_map) =
                Self::ctr_oof_columns(keys, y, n_rows, prior, smooth, n_perm, &mut rng);
            single_oof_cols.push(enc.clone());
            train_cols.push(enc);
            train_cols.push(count);
            self.ordered_ctr_priors.push(prior);
            self.ordered_ctr_maps.push(enc_map);
            self.ordered_ctr_count_maps.push(count_map);
        }
        self.ordered_ctr_features = selected.clone();

        let mut pair_scored: Vec<((usize, usize), f64, Vec<i64>)> = Vec::new();
        for i in 0..selected.len() {
            for j in (i + 1)..selected.len() {
                let fi = selected[i];
                let fj = selected[j];
                let keys: Vec<i64> = (0..n_rows)
                    .map(|row| {
                        let a = Self::ctr_single_key(x_data[row * n_features + fi]);
                        let b = Self::ctr_single_key(x_data[row * n_features + fj]);
                        Self::ctr_pair_key(a, b)
                    })
                    .collect();
                let stats = Self::ctr_stats(&keys, y);
                if stats.len() < 2 {
                    continue;
                }
                let expected = n_rows as f64 / stats.len().max(1) as f64;
                if expected < pair_expected_min {
                    continue;
                }

                let mut incremental_stats: HashMap<i64, (usize, f64, f64)> = HashMap::new();
                for row in 0..n_rows {
                    let key = keys[row];
                    let ki = Self::ctr_single_key(x_data[row * n_features + fi]);
                    let kj = Self::ctr_single_key(x_data[row * n_features + fj]);
                    let pi = self.ordered_ctr_maps[i].get(&ki).copied().unwrap_or(prior);
                    let pj = self.ordered_ctr_maps[j].get(&kj).copied().unwrap_or(prior);
                    let parent_blend = 0.5 * (pi + pj);
                    let entry = incremental_stats.entry(key).or_insert((0, 0.0, 0.0));
                    entry.0 += 1;
                    entry.1 += y[row];
                    entry.2 += parent_blend;
                }

                let mut score = 0.0;
                let mut total = 0usize;
                for &(cnt, sum_y, sum_parent) in incremental_stats.values() {
                    if cnt < min_count {
                        continue;
                    }
                    let pair_enc = (sum_y + smooth * prior) / (cnt as f64 + smooth);
                    let parent_mean = sum_parent / cnt as f64;
                    if self.task == "binary" {
                        let parent_loss = Self::ctr_binary_logloss(sum_y, cnt, parent_mean);
                        let pair_loss = Self::ctr_binary_logloss(sum_y, cnt, pair_enc);
                        score += parent_loss - pair_loss;
                    } else {
                        score += cnt as f64 * (pair_enc - parent_mean) * (pair_enc - parent_mean);
                    }
                    total += cnt;
                }
                if total < min_count * 2 || !score.is_finite() {
                    score = 0.0;
                }
                if score > 0.0 {
                    pair_scored.push(((fi, fj), score, keys));
                }
            }
        }
        pair_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        pair_scored.truncate(self.ordered_ctr_top_features.min(pair_scored.len()));

        for ((fi, fj), _score, keys) in &pair_scored {
            let (enc, count, enc_map, count_map) =
                Self::ctr_oof_columns(keys, y, n_rows, prior, smooth, n_perm, &mut rng);
            let ii = selected.iter().position(|&f| f == *fi);
            let jj = selected.iter().position(|&f| f == *fj);
            let mut residual = Vec::with_capacity(n_rows);
            if let (Some(ii), Some(jj)) = (ii, jj) {
                for row in 0..n_rows {
                    residual.push(
                        enc[row] - 0.5 * (single_oof_cols[ii][row] + single_oof_cols[jj][row]),
                    );
                }
            } else {
                residual.resize(n_rows, 0.0);
            }
            train_cols.push(enc);
            train_cols.push(count);
            train_cols.push(residual);
            self.ordered_ctr_pair_features.push((*fi, *fj));
            self.ordered_ctr_pair_maps.push(enc_map);
            self.ordered_ctr_pair_count_maps.push(count_map);
        }

        let mut triple_scored: Vec<((usize, usize, usize), f64, Vec<i64>)> = Vec::new();
        if selected.len() >= 3 && self.task == "binary" {
            for a_idx in 0..selected.len() {
                for b_idx in (a_idx + 1)..selected.len() {
                    for c_idx in (b_idx + 1)..selected.len() {
                        let fa = selected[a_idx];
                        let fb = selected[b_idx];
                        let fc = selected[c_idx];
                        let keys: Vec<i64> = (0..n_rows)
                            .map(|row| {
                                let a = Self::ctr_single_key(x_data[row * n_features + fa]);
                                let b = Self::ctr_single_key(x_data[row * n_features + fb]);
                                let c = Self::ctr_single_key(x_data[row * n_features + fc]);
                                Self::ctr_triple_key(a, b, c)
                            })
                            .collect();
                        let stats = Self::ctr_stats(&keys, y);
                        if stats.len() < 2 {
                            continue;
                        }
                        let expected = n_rows as f64 / stats.len().max(1) as f64;
                        if expected < pair_expected_min {
                            continue;
                        }

                        let mut incremental_stats: HashMap<i64, (usize, f64, f64)> = HashMap::new();
                        for row in 0..n_rows {
                            let key = keys[row];
                            let ka = Self::ctr_single_key(x_data[row * n_features + fa]);
                            let kb = Self::ctr_single_key(x_data[row * n_features + fb]);
                            let kc = Self::ctr_single_key(x_data[row * n_features + fc]);
                            let pa = self.ordered_ctr_maps[a_idx]
                                .get(&ka)
                                .copied()
                                .unwrap_or(prior);
                            let pb = self.ordered_ctr_maps[b_idx]
                                .get(&kb)
                                .copied()
                                .unwrap_or(prior);
                            let pc = self.ordered_ctr_maps[c_idx]
                                .get(&kc)
                                .copied()
                                .unwrap_or(prior);
                            let parent_blend = (pa + pb + pc) / 3.0;
                            let entry = incremental_stats.entry(key).or_insert((0, 0.0, 0.0));
                            entry.0 += 1;
                            entry.1 += y[row];
                            entry.2 += parent_blend;
                        }

                        let mut score = 0.0;
                        let mut total = 0usize;
                        for &(cnt, sum_y, sum_parent) in incremental_stats.values() {
                            if cnt < min_count {
                                continue;
                            }
                            let tuple_enc = (sum_y + smooth * prior) / (cnt as f64 + smooth);
                            let parent_mean = sum_parent / cnt as f64;
                            score += Self::ctr_binary_logloss(sum_y, cnt, parent_mean)
                                - Self::ctr_binary_logloss(sum_y, cnt, tuple_enc);
                            total += cnt;
                        }
                        if total < min_count * 2 || !score.is_finite() {
                            score = 0.0;
                        }
                        if score > 0.0 {
                            triple_scored.push(((fa, fb, fc), score, keys));
                        }
                    }
                }
            }
        }
        triple_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        triple_scored.truncate(self.ordered_ctr_top_features.min(triple_scored.len()));

        for ((fa, fb, fc), _score, keys) in &triple_scored {
            let (enc, count, enc_map, count_map) =
                Self::ctr_oof_columns(keys, y, n_rows, prior, smooth, n_perm, &mut rng);
            let ia = selected.iter().position(|&f| f == *fa);
            let ib = selected.iter().position(|&f| f == *fb);
            let ic = selected.iter().position(|&f| f == *fc);
            let mut residual = Vec::with_capacity(n_rows);
            if let (Some(ia), Some(ib), Some(ic)) = (ia, ib, ic) {
                for row in 0..n_rows {
                    residual.push(
                        enc[row]
                            - (single_oof_cols[ia][row]
                                + single_oof_cols[ib][row]
                                + single_oof_cols[ic][row])
                                / 3.0,
                    );
                }
            } else {
                residual.resize(n_rows, 0.0);
            }
            train_cols.push(enc);
            train_cols.push(count);
            train_cols.push(residual);
            self.ordered_ctr_triple_features.push((*fa, *fb, *fc));
            self.ordered_ctr_triple_maps.push(enc_map);
            self.ordered_ctr_triple_count_maps.push(count_map);
        }

        train_cols
    }

    pub(super) fn ordered_ctr_columns_for_raw(
        &self,
        x_data: &[f64],
        n_rows: usize,
        n_features: usize,
    ) -> Vec<Vec<f64>> {
        let mut cols = Vec::new();
        let prior = self.ordered_ctr_prior;

        for (mi, &feat) in self.ordered_ctr_features.iter().enumerate() {
            if feat >= n_features || mi >= self.ordered_ctr_maps.len() {
                continue;
            }
            let map = &self.ordered_ctr_maps[mi];
            let count_map = self.ordered_ctr_count_maps.get(mi);
            let col_prior = self.ordered_ctr_priors.get(mi).copied().unwrap_or(prior);
            let mut enc_col = Vec::with_capacity(n_rows);
            let mut count_col = Vec::with_capacity(n_rows);
            for row in 0..n_rows {
                let key = Self::ctr_single_key(x_data[row * n_features + feat]);
                enc_col.push(*map.get(&key).unwrap_or(&col_prior));
                count_col.push(count_map.and_then(|m| m.get(&key)).copied().unwrap_or(0.0));
            }
            cols.push(enc_col);
            cols.push(count_col);
        }

        for (mi, &(fi, fj)) in self.ordered_ctr_pair_features.iter().enumerate() {
            if fi >= n_features || fj >= n_features || mi >= self.ordered_ctr_pair_maps.len() {
                continue;
            }
            let map = &self.ordered_ctr_pair_maps[mi];
            let count_map = self.ordered_ctr_pair_count_maps.get(mi);
            let mut enc_col = Vec::with_capacity(n_rows);
            let mut count_col = Vec::with_capacity(n_rows);
            let mut resid_col = Vec::with_capacity(n_rows);
            let parent_i = self.ordered_ctr_features.iter().position(|&f| f == fi);
            let parent_j = self.ordered_ctr_features.iter().position(|&f| f == fj);
            for row in 0..n_rows {
                let a = Self::ctr_single_key(x_data[row * n_features + fi]);
                let b = Self::ctr_single_key(x_data[row * n_features + fj]);
                let key = Self::ctr_pair_key(a, b);
                let pair_enc = *map.get(&key).unwrap_or(&prior);
                let pi = parent_i
                    .and_then(|idx| self.ordered_ctr_maps.get(idx))
                    .and_then(|m| m.get(&a))
                    .copied()
                    .unwrap_or(prior);
                let pj = parent_j
                    .and_then(|idx| self.ordered_ctr_maps.get(idx))
                    .and_then(|m| m.get(&b))
                    .copied()
                    .unwrap_or(prior);
                enc_col.push(pair_enc);
                count_col.push(count_map.and_then(|m| m.get(&key)).copied().unwrap_or(0.0));
                resid_col.push(pair_enc - 0.5 * (pi + pj));
            }
            cols.push(enc_col);
            cols.push(count_col);
            cols.push(resid_col);
        }

        for (mi, &(fa, fb, fc)) in self.ordered_ctr_triple_features.iter().enumerate() {
            if fa >= n_features
                || fb >= n_features
                || fc >= n_features
                || mi >= self.ordered_ctr_triple_maps.len()
            {
                continue;
            }
            let map = &self.ordered_ctr_triple_maps[mi];
            let count_map = self.ordered_ctr_triple_count_maps.get(mi);
            let mut enc_col = Vec::with_capacity(n_rows);
            let mut count_col = Vec::with_capacity(n_rows);
            let mut resid_col = Vec::with_capacity(n_rows);
            let parent_a = self.ordered_ctr_features.iter().position(|&f| f == fa);
            let parent_b = self.ordered_ctr_features.iter().position(|&f| f == fb);
            let parent_c = self.ordered_ctr_features.iter().position(|&f| f == fc);
            for row in 0..n_rows {
                let a = Self::ctr_single_key(x_data[row * n_features + fa]);
                let b = Self::ctr_single_key(x_data[row * n_features + fb]);
                let c = Self::ctr_single_key(x_data[row * n_features + fc]);
                let key = Self::ctr_triple_key(a, b, c);
                let enc = *map.get(&key).unwrap_or(&prior);
                let pa = parent_a
                    .and_then(|idx| self.ordered_ctr_maps.get(idx))
                    .and_then(|m| m.get(&a))
                    .copied()
                    .unwrap_or(prior);
                let pb = parent_b
                    .and_then(|idx| self.ordered_ctr_maps.get(idx))
                    .and_then(|m| m.get(&b))
                    .copied()
                    .unwrap_or(prior);
                let pc = parent_c
                    .and_then(|idx| self.ordered_ctr_maps.get(idx))
                    .and_then(|m| m.get(&c))
                    .copied()
                    .unwrap_or(prior);
                enc_col.push(enc);
                count_col.push(count_map.and_then(|m| m.get(&key)).copied().unwrap_or(0.0));
                resid_col.push(enc - (pa + pb + pc) / 3.0);
            }
            cols.push(enc_col);
            cols.push(count_col);
            cols.push(resid_col);
        }

        cols
    }

    #[inline]
    pub(super) fn posterior_leaf_tau(&self) -> f64 {
        if !self.ordered_boost || self.prob_avg {
            return 0.0;
        }
        match self.task.as_str() {
            "multiclass" => 4.0,
            "regression" | "poisson" => 2.0,
            _ => 3.0,
        }
    }

    #[inline]
    pub(super) fn multiclass_trees_per_class_round(&self) -> usize {
        self.multiclass_trees_per_class.max(1)
    }

    #[inline]
    pub(super) fn multiclass_tree_lr(&self) -> f64 {
        self.learning_rate * self.multiclass_tree_lr_scale
    }

    #[inline]
    pub(super) fn finalize_multiclass_tree(
        &self,
        tree: &mut DecisionTree,
        round: usize,
        n_rounds: usize,
        n_sub: usize,
        sub_scale: f64,
        posterior_tau: f64,
    ) {
        if self.lr_decay < 1.0 && n_rounds > 1 {
            let factor = 1.0 - (1.0 - self.lr_decay) * (round as f64) / (n_rounds as f64 - 1.0);
            for v in tree.values.iter_mut() {
                *v *= factor;
            }
            tree.scale_ramp_slopes(factor);
            tree.scale_cat_lookups(factor);
        }

        if self.max_delta_step > 0.0 {
            let mds = self.max_delta_step;
            for v in tree.values.iter_mut() {
                *v = v.clamp(-mds, mds);
            }
        }

        if n_sub > 1 {
            for v in tree.values.iter_mut() {
                *v *= sub_scale;
            }
            tree.scale_ramp_slopes(sub_scale);
            tree.scale_cat_lookups(sub_scale);
        }

        if posterior_tau > 0.0 {
            tree.posterior_shrink_leaves(posterior_tau);
        }
        self.apply_hierarchical_shrinkage(tree);
    }

    pub(super) fn subsample_indices_from_pool(
        &self,
        rng: &mut StdRng,
        pool: &[u32],
        rate: f64,
    ) -> Vec<u32> {
        let n_pool = pool.len();
        if n_pool == 0 {
            return Vec::new();
        }
        let sample_size = ((rate * n_pool as f64) as usize).max(1).min(n_pool);
        if sample_size >= n_pool && !self.use_bootstrap {
            return pool.to_vec();
        }
        if self.use_bootstrap {
            (0..sample_size)
                .map(|_| pool[rng.random_range(0..n_pool)])
                .collect()
        } else {
            let mut indices = pool.to_vec();
            for i in 0..sample_size {
                let j = rng.random_range(i..n_pool);
                indices.swap(i, j);
            }
            indices.truncate(sample_size);
            indices
        }
    }

    pub(super) fn round_subsamples_from_pool(
        &self,
        rng: &mut StdRng,
        pool: &[u32],
        n_sub: usize,
    ) -> Vec<Vec<u32>> {
        if n_sub <= 1 {
            return vec![self.subsample_indices_from_pool(rng, pool, self.subsample_rate)];
        }
        if !self.antithetic_subtrees
            || self.use_bootstrap
            || self.subsample_rate <= 0.0
            || self.subsample_rate >= 1.0
        {
            return (0..n_sub)
                .map(|_| self.subsample_indices_from_pool(rng, pool, self.subsample_rate))
                .collect();
        }

        let n_pool = pool.len();
        let sample_size = ((self.subsample_rate * n_pool as f64) as usize)
            .max(1)
            .min(n_pool.saturating_sub(1).max(1));
        let mut perm = pool.to_vec();
        perm.shuffle(rng);

        let mut out = Vec::with_capacity(n_sub);
        for sub_idx in 0..n_sub {
            let start = (sub_idx * n_pool) / n_sub;
            let mut sample = Vec::with_capacity(sample_size);
            for j in 0..sample_size {
                sample.push(perm[(start + j) % n_pool]);
            }
            out.push(sample);
        }
        out
    }

    fn binary_auc_error(eval_y: &[f64], eval_preds: &[f64], n_eval: usize) -> Option<f64> {
        if n_eval == 0 || eval_y.len() < n_eval || eval_preds.len() < n_eval {
            return None;
        }
        let mut pairs: Vec<(f64, bool)> = Vec::with_capacity(n_eval);
        let mut pos = 0usize;
        for i in 0..n_eval {
            let is_pos = eval_y[i] > 0.5;
            if is_pos {
                pos += 1;
            }
            pairs.push((eval_preds[i], is_pos));
        }
        let neg = n_eval.saturating_sub(pos);
        if pos == 0 || neg == 0 {
            return None;
        }
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut rank_sum_pos = 0.0f64;
        let mut i = 0usize;
        while i < n_eval {
            let mut j = i + 1;
            while j < n_eval && (pairs[j].0 - pairs[i].0).abs() <= 1e-12 {
                j += 1;
            }
            let avg_rank = 0.5 * ((i + 1) as f64 + j as f64);
            for item in pairs.iter().take(j).skip(i) {
                if item.1 {
                    rank_sum_pos += avg_rank;
                }
            }
            i = j;
        }
        let pos_f = pos as f64;
        let neg_f = neg as f64;
        let auc = (rank_sum_pos - pos_f * (pos_f + 1.0) * 0.5) / (pos_f * neg_f);
        Some(1.0 - auc.clamp(0.0, 1.0))
    }

    pub(super) fn compute_eval_loss(
        &self,
        eval_y: &[f64],
        eval_preds: &[f64],
        n_eval: usize,
    ) -> f64 {
        match self.task.as_str() {
            "regression" => {
                let mut mse = 0.0f64;
                for i in 0..n_eval {
                    let d = eval_preds[i] - eval_y[i];
                    mse += d * d;
                }
                mse / n_eval as f64
            }
            "binary" | "rank" => {
                let metric = self.eval_metric.to_ascii_lowercase();
                if matches!(metric.as_str(), "auc" | "roc_auc" | "1-auc") {
                    if let Some(err) = Self::binary_auc_error(eval_y, eval_preds, n_eval) {
                        return err;
                    }
                }
                // For rank task: raw scores are uncalibrated, but sigmoid preserves ranking.
                // Log-loss on raw scores still decreases with better ranking → valid ES signal.
                let mut loss = 0.0f64;
                for i in 0..n_eval {
                    let p = 1.0 / (1.0 + (-eval_preds[i]).exp());
                    let p = p.clamp(1e-15, 1.0 - 1e-15);
                    loss -= eval_y[i] * p.ln() + (1.0 - eval_y[i]) * (1.0 - p).ln();
                }
                loss / n_eval as f64
            }
            "poisson" => {
                // Poisson deviance: 2 * Σ [y*ln(y/mu) - (y - mu)]
                let mut dev = 0.0f64;
                for i in 0..n_eval {
                    let mu = eval_preds[i].exp().max(1e-15);
                    let yi = eval_y[i];
                    if yi > 0.0 {
                        dev += yi * (yi / mu).ln() - (yi - mu);
                    } else {
                        dev += mu;
                    }
                }
                2.0 * dev / n_eval as f64
            }
            _ => {
                let mut mse = 0.0f64;
                for i in 0..n_eval {
                    let d = eval_preds[i] - eval_y[i];
                    mse += d * d;
                }
                mse / n_eval as f64
            }
        }
    }

    pub(super) fn compute_multiclass_eval_loss(
        &self,
        eval_y: &[f64],
        eval_preds: &[f64],
        n_eval: usize,
        n_classes: usize,
    ) -> f64 {
        let mut loss = 0.0f64;
        for i in 0..n_eval {
            let base = i * n_classes;
            let max_val = eval_preds[base..base + n_classes]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let mut sum_exp = 0.0f64;
            for k in 0..n_classes {
                sum_exp += (eval_preds[base + k] - max_val).exp();
            }
            let true_class = eval_y[i] as usize;
            let log_prob = (eval_preds[base + true_class] - max_val) - sum_exp.ln();
            loss -= log_prob;
        }
        loss / n_eval as f64
    }

    pub(super) fn compute_softmax(
        predictions: &[f64],
        probs: &mut [f64],
        n_rows: usize,
        n_classes: usize,
    ) {
        for i in 0..n_rows {
            let base = i * n_classes;
            let max_val = predictions[base..base + n_classes]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let mut sum_exp = 0.0f64;
            for k in 0..n_classes {
                probs[base + k] = (predictions[base + k] - max_val).exp();
                sum_exp += probs[base + k];
            }
            for k in 0..n_classes {
                probs[base + k] /= sum_exp;
            }
        }
    }

    /// Jensen-aware softmax: divide logits by T before normalizing. T > 1 softens
    /// output toward uniform (mimics test-time PD smoothing → aligns training
    /// gradients with the Jensen-smoothed inference path).
    pub(super) fn compute_softmax_t(
        predictions: &[f64],
        probs: &mut [f64],
        n_rows: usize,
        n_classes: usize,
        temp: f64,
    ) {
        if (temp - 1.0).abs() < 1e-9 {
            Self::compute_softmax(predictions, probs, n_rows, n_classes);
            return;
        }
        let inv_t = 1.0 / temp;
        for i in 0..n_rows {
            let base = i * n_classes;
            let max_val = predictions[base..base + n_classes]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let mut sum_exp = 0.0f64;
            for k in 0..n_classes {
                probs[base + k] = ((predictions[base + k] - max_val) * inv_t).exp();
                sum_exp += probs[base + k];
            }
            for k in 0..n_classes {
                probs[base + k] /= sum_exp;
            }
        }
    }

    pub(super) fn compute_softmax_par(
        predictions: &[f64],
        probs: &mut [f64],
        n_rows: usize,
        n_classes: usize,
    ) {
        if n_rows >= 4096 {
            let nc = n_classes;
            probs
                .par_chunks_mut(nc * 256)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let start_row = ci * 256;
                    let n_chunk_rows = chunk.len() / nc;
                    for r in 0..n_chunk_rows {
                        let i = start_row + r;
                        let p_base = i * nc;
                        let base = r * nc;
                        let max_val = predictions[p_base..p_base + nc]
                            .iter()
                            .cloned()
                            .fold(f64::NEG_INFINITY, f64::max);
                        let mut sum_exp = 0.0f64;
                        for k in 0..nc {
                            chunk[base + k] = (predictions[p_base + k] - max_val).exp();
                            sum_exp += chunk[base + k];
                        }
                        for k in 0..nc {
                            chunk[base + k] /= sum_exp;
                        }
                    }
                });
        } else {
            Self::compute_softmax(predictions, probs, n_rows, n_classes);
        }
    }

    /// Run lightweight warmup trees for interaction discovery.
    /// Returns the built trees (not stored in self). Uses depthwise builder with simple params.
    pub(super) fn run_warmup_trees(
        binned: &BinnedData,
        y: &[f64],
        n_rows: usize,
        n_warmup: usize,
        task: &str,
        huber_delta: f64,
        lambda_reg: f64,
        gamma: f64,
        max_depth: usize,
        min_child_weight: f64,
        seed: u64,
        n_features: usize,
    ) -> Vec<DecisionTree> {
        let all_features = vec![true; binned.n_features];
        let all_indices: Vec<u32> = (0..n_rows as u32).collect();
        let mono_cstr = vec![0i8; binned.n_features];

        // Compute base score
        let y_mean = y.iter().sum::<f64>() / n_rows as f64;
        let base_score = match task {
            "regression" => y_mean,
            "binary" => {
                let p = y_mean.clamp(1e-6, 1.0 - 1e-6);
                (p / (1.0 - p)).ln()
            }
            "poisson" => y_mean.max(1e-6).ln(),
            // Multiclass: use regression proxy (y = class label 0..K-1) for feature discovery
            // The warmup trees find which features separate classes, sufficient for co-occurrence scoring
            _ => y_mean,
        };

        let mut predictions = vec![base_score; n_rows];
        let mut gradients = vec![0.0f64; n_rows];
        let mut hessians = vec![0.0f64; n_rows];
        let mut trees = Vec::with_capacity(n_warmup);
        let lr = 0.3; // fixed lr for warmup — just need to discover structure

        for _round in 0..n_warmup {
            // Compute gradients
            match task {
                "regression" => {
                    if huber_delta > 0.0 {
                        for i in 0..n_rows {
                            let r = predictions[i] - y[i];
                            let abs_r = r.abs();
                            if abs_r <= huber_delta {
                                gradients[i] = r;
                                hessians[i] = 1.0;
                            } else {
                                gradients[i] = huber_delta * r.signum();
                                hessians[i] = huber_delta / abs_r;
                            }
                        }
                    } else {
                        for i in 0..n_rows {
                            gradients[i] = predictions[i] - y[i];
                            hessians[i] = 1.0;
                        }
                    }
                }
                "binary" => {
                    for i in 0..n_rows {
                        let p = 1.0 / (1.0 + (-predictions[i]).exp());
                        gradients[i] = p - y[i];
                        hessians[i] = (p * (1.0 - p)).max(1e-16);
                    }
                }
                "poisson" => {
                    for i in 0..n_rows {
                        let mu = predictions[i].exp().min(1e15);
                        gradients[i] = mu - y[i];
                        hessians[i] = mu.max(1e-16);
                    }
                }
                _ => {
                    for i in 0..n_rows {
                        gradients[i] = predictions[i] - y[i];
                        hessians[i] = 1.0;
                    }
                }
            }

            let tree = DecisionTree::build_depthwise(
                binned,
                &gradients,
                &hessians,
                &all_indices,
                lambda_reg,
                0.0,
                gamma,
                max_depth.min(6),
                min_child_weight,
                &all_features,
                1.0, // colsample_bylevel
                seed.wrapping_add(_round as u64),
                0.0, // random_strength
                0.0, // cat_smooth
                0.0, // cat_lookup_smooth
                &mono_cstr,
                0.0,   // gain_penalty
                false, // warmup always uses optimal splits
                0.0,   // lookahead_alpha (warmup is greedy)
                false, // expert_split
                false, // sparse_oblique_splits
                false, // interval_splits
                None,
                false,
                0.0, // leaf_var_shrink
                crate::tree::CatPairConfig::default(),

                                None,
                            
                None,
            
                0.0,
            
                0.5,
                1.0,
            );
            tree.add_predictions_binned(binned, &mut predictions, lr);
            trees.push(tree);
        }
        trees
    }

    pub(super) fn stable_numeric_interaction_pairs(
        &self,
        x_data: &[f64],
        y: &[f64],
        n_rows: usize,
        n_features: usize,
        numeric_indices: &[usize],
        max_pairs: usize,
    ) -> Vec<((usize, usize), f64)> {
        if n_rows < 64
            || numeric_indices.len() < 2
            || numeric_indices.len() > 128
            || x_data.len() < n_rows.saturating_mul(n_features)
        {
            return Vec::new();
        }

        #[inline]
        fn corr_from_sums(n: usize, sx: f64, sy: f64, sxx: f64, syy: f64, sxy: f64) -> f64 {
            if n < 8 {
                return 0.0;
            }
            let nf = n as f64;
            let vx = (sxx - sx * sx / nf).max(0.0);
            let vy = (syy - sy * sy / nf).max(0.0);
            if vx <= 1e-24 || vy <= 1e-24 {
                return 0.0;
            }
            (sxy - sx * sy / nf) / (vx.sqrt() * vy.sqrt())
        }

        let mut single_corr = vec![[0.0f64; 2]; n_features];
        for &feat in numeric_indices {
            for fold in 0..2usize {
                let mut n = 0usize;
                let mut sx = 0.0;
                let mut sy = 0.0;
                let mut sxx = 0.0;
                let mut syy = 0.0;
                let mut sxy = 0.0;
                for row in (fold..n_rows).step_by(2) {
                    let x = x_data[row * n_features + feat];
                    let yy = y[row];
                    if !x.is_finite() || !yy.is_finite() {
                        continue;
                    }
                    n += 1;
                    sx += x;
                    sy += yy;
                    sxx += x * x;
                    syy += yy * yy;
                    sxy += x * yy;
                }
                single_corr[feat][fold] = corr_from_sums(n, sx, sy, sxx, syy, sxy);
            }
        }

        let mut scored: Vec<((usize, usize), f64)> = Vec::new();
        for a_pos in 0..numeric_indices.len() {
            let fa = numeric_indices[a_pos];
            for &fb in numeric_indices.iter().skip(a_pos + 1) {
                let mut prod_corr = [0.0f64; 2];
                for (fold, slot) in prod_corr.iter_mut().enumerate() {
                    let mut n = 0usize;
                    let mut sx = 0.0;
                    let mut sy = 0.0;
                    let mut sxx = 0.0;
                    let mut syy = 0.0;
                    let mut sxy = 0.0;
                    for row in (fold..n_rows).step_by(2) {
                        let xa = x_data[row * n_features + fa];
                        let xb = x_data[row * n_features + fb];
                        let yy = y[row];
                        if !xa.is_finite() || !xb.is_finite() || !yy.is_finite() {
                            continue;
                        }
                        let x = xa * xb;
                        if !x.is_finite() {
                            continue;
                        }
                        n += 1;
                        sx += x;
                        sy += yy;
                        sxx += x * x;
                        syy += yy * yy;
                        sxy += x * yy;
                    }
                    *slot = corr_from_sums(n, sx, sy, sxx, syy, sxy);
                }
                if prod_corr[0] * prod_corr[1] <= 0.0 {
                    continue;
                }
                let surplus0 =
                    prod_corr[0].abs() - single_corr[fa][0].abs().max(single_corr[fb][0].abs());
                let surplus1 =
                    prod_corr[1].abs() - single_corr[fa][1].abs().max(single_corr[fb][1].abs());
                let min_surplus = surplus0.min(surplus1);
                if min_surplus <= 0.025 {
                    continue;
                }
                let mean_abs = 0.5 * (prod_corr[0].abs() + prod_corr[1].abs());
                let score = min_surplus * mean_abs;
                if score.is_finite() && score > 0.0 {
                    scored.push(((fa.min(fb), fa.max(fb)), score));
                }
            }
        }

        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(max_pairs.max(1));
        scored
    }

    pub(super) fn compute_gradients_hessians(
        &self,
        y: &[f64],
        preds: &[f64],
        gradients: &mut [f64],
        hessians: &mut [f64],
    ) {
        let n = y.len();
        let use_par = n >= 4096;
        match self.task.as_str() {
            "regression" => {
                let delta = if self.huber_delta < 0.0 {
                    Self::adaptive_huber_delta(y, preds)
                } else {
                    self.huber_delta
                };
                if delta > 0.0 {
                    if use_par {
                        gradients
                            .par_chunks_mut(1024)
                            .zip(hessians.par_chunks_mut(1024))
                            .enumerate()
                            .for_each(|(chunk_idx, (g_chunk, h_chunk))| {
                                let start = chunk_idx * 1024;
                                for (j, (g, h)) in
                                    g_chunk.iter_mut().zip(h_chunk.iter_mut()).enumerate()
                                {
                                    let i = start + j;
                                    let r = preds[i] - y[i];
                                    let abs_r = r.abs();
                                    if abs_r <= delta {
                                        *g = r;
                                        *h = 1.0;
                                    } else {
                                        *g = delta * r.signum();
                                        *h = delta / abs_r;
                                    }
                                }
                            });
                    } else {
                        for i in 0..n {
                            let r = preds[i] - y[i];
                            let abs_r = r.abs();
                            if abs_r <= delta {
                                gradients[i] = r;
                                hessians[i] = 1.0;
                            } else {
                                gradients[i] = delta * r.signum();
                                hessians[i] = delta / abs_r;
                            }
                        }
                    }
                } else {
                    if use_par {
                        gradients
                            .par_chunks_mut(1024)
                            .zip(hessians.par_chunks_mut(1024))
                            .enumerate()
                            .for_each(|(chunk_idx, (g_chunk, h_chunk))| {
                                let start = chunk_idx * 1024;
                                for (j, (g, h)) in
                                    g_chunk.iter_mut().zip(h_chunk.iter_mut()).enumerate()
                                {
                                    let i = start + j;
                                    *g = preds[i] - y[i];
                                    *h = 1.0;
                                }
                            });
                    } else {
                        for i in 0..n {
                            gradients[i] = preds[i] - y[i];
                            hessians[i] = 1.0;
                        }
                    }
                }
            }
            "binary" => {
                // MGB: when rank_mix_alpha > 0, blend log-loss with pairwise rank gradient.
                // g = (1-α)·g_logloss + α·g_rank ; same for h. Preserves calibration while
                // adding AUC-targeted ranking signal. α=0: pure log-loss (standard binary).
                let alpha = self.rank_mix_alpha;
                let use_blend = alpha > 0.0;
                let focus_gamma = self.binary_focus_gamma;
                let (neg_weight, pos_weight) = if self.class_weights.len() >= 2 {
                    let neg = if self.class_weights[0].is_finite() && self.class_weights[0] > 0.0 {
                        self.class_weights[0]
                    } else {
                        1.0
                    };
                    let pos = if self.class_weights[1].is_finite() && self.class_weights[1] > 0.0 {
                        self.class_weights[1]
                    } else {
                        1.0
                    };
                    (neg, pos)
                } else {
                    (1.0, 1.0)
                };
                if use_blend {
                    const RANK_BINS: usize = 256;
                    let pos_count = y.iter().filter(|&&v| v > 0.5).count();
                    let neg_count = n.saturating_sub(pos_count);
                    if pos_count == 0 || neg_count == 0 {
                        // Degenerate — fall back to pure log-loss
                        for i in 0..n {
                            let p = 1.0 / (1.0 + (-preds[i]).exp());
                            let w = if y[i] > 0.5 { pos_weight } else { neg_weight };
                            let focus = if focus_gamma > 0.0 {
                                let err = if y[i] > 0.5 { 1.0 - p } else { p };
                                (2.0 * err.clamp(0.0, 1.0)).powf(focus_gamma)
                            } else {
                                1.0
                            };
                            gradients[i] = w * focus * (p - y[i]);
                            hessians[i] = w * focus * (p * (1.0 - p)).max(1e-16);
                        }
                    } else {
                        let pair_count = (pos_count as u128) * (neg_count as u128);
                        if pair_count <= 2_000_000 {
                            let pair_temp = self.rank_pair_temperature.max(0.05);
                            let inv_temp = 1.0 / pair_temp;
                            let inv_temp_sq = inv_temp * inv_temp;
                            let mut rank_g = vec![0.0f64; n];
                            let mut rank_h = vec![0.0f64; n];
                            let mut pos_idx: Vec<usize> = Vec::with_capacity(pos_count);
                            let mut neg_idx: Vec<usize> = Vec::with_capacity(neg_count);
                            for i in 0..n {
                                if y[i] > 0.5 {
                                    pos_idx.push(i);
                                } else {
                                    neg_idx.push(i);
                                }
                            }
                            let pos_total = pos_count as f64;
                            let neg_total = neg_count as f64;
                            for &pi in &pos_idx {
                                let zp = preds[pi];
                                let mut gp = 0.0f64;
                                let mut hp = 0.0f64;
                                for &ni in &neg_idx {
                                    let d = ((zp - preds[ni]) * inv_temp).clamp(-50.0, 50.0);
                                    let s = 1.0 / (1.0 + (-d).exp());
                                    let h = (s * (1.0 - s)).max(1e-16);
                                    gp += s - 1.0;
                                    hp += h;
                                    rank_g[ni] += 1.0 - s;
                                    rank_h[ni] += h;
                                }
                                rank_g[pi] = inv_temp * gp / neg_total;
                                rank_h[pi] = (inv_temp_sq * hp / neg_total).max(1e-16);
                            }
                            for &ni in &neg_idx {
                                rank_g[ni] = inv_temp * rank_g[ni] / pos_total;
                                rank_h[ni] = (inv_temp_sq * rank_h[ni] / pos_total).max(1e-16);
                            }

                            let one_minus_a = 1.0 - alpha;
                            for i in 0..n {
                                let z = preds[i].clamp(-50.0, 50.0);
                                let p = 1.0 / (1.0 + (-z).exp());
                                let g_log = p - y[i];
                                let h_log = (p * (1.0 - p)).max(1e-16);
                                let w = if y[i] > 0.5 { pos_weight } else { neg_weight };
                                let focus = if focus_gamma > 0.0 {
                                    let err = if y[i] > 0.5 { 1.0 - p } else { p };
                                    (2.0 * err.clamp(0.0, 1.0)).powf(focus_gamma)
                                } else {
                                    1.0
                                };
                                gradients[i] =
                                    w * focus * (one_minus_a * g_log + alpha * rank_g[i]);
                                hessians[i] = w
                                    * focus
                                    * (one_minus_a * h_log + alpha * rank_h[i]).max(1e-16);
                            }
                        } else {
                            // Histogram pairwise AUC surrogate: approximate the all-pairs RankNet
                            // gradient against the opposite-class score distribution in O(n+B^2).
                            let mut lo = f64::INFINITY;
                            let mut hi = f64::NEG_INFINITY;
                            let mut sum = 0.0f64;
                            let mut sum_sq = 0.0f64;
                            let mut finite_n = 0usize;
                            for &z in preds {
                                if z.is_finite() {
                                    lo = lo.min(z);
                                    hi = hi.max(z);
                                    sum += z;
                                    sum_sq += z * z;
                                    finite_n += 1;
                                }
                            }
                            if finite_n == 0 {
                                lo = -1.0;
                                hi = 1.0;
                            } else {
                                let mean = sum / finite_n as f64;
                                let var = (sum_sq / finite_n as f64 - mean * mean).max(0.0);
                                let std = var.sqrt();
                                if std.is_finite() && std > 1e-12 {
                                    lo = lo.max(mean - 8.0 * std);
                                    hi = hi.min(mean + 8.0 * std);
                                }
                                if !lo.is_finite() || !hi.is_finite() || hi <= lo + 1e-12 {
                                    lo = mean - 1.0;
                                    hi = mean + 1.0;
                                }
                            }
                            let scale = (RANK_BINS as f64 - 1.0) / (hi - lo).max(1e-12);
                            let bin_of = |z: f64| -> usize {
                                if !z.is_finite() {
                                    return RANK_BINS / 2;
                                }
                                (((z.clamp(lo, hi) - lo) * scale).round() as isize)
                                    .clamp(0, RANK_BINS as isize - 1)
                                    as usize
                            };

                            let mut pos_hist = vec![0.0f64; RANK_BINS];
                            let mut neg_hist = vec![0.0f64; RANK_BINS];
                            for i in 0..n {
                                let b = bin_of(preds[i]);
                                if y[i] > 0.5 {
                                    pos_hist[b] += 1.0;
                                } else {
                                    neg_hist[b] += 1.0;
                                }
                            }

                            let pos_total = pos_count as f64;
                            let neg_total = neg_count as f64;
                            let mut rank_g_pos = vec![0.0f64; RANK_BINS];
                            let mut rank_h_pos = vec![0.0f64; RANK_BINS];
                            let mut rank_g_neg = vec![0.0f64; RANK_BINS];
                            let mut rank_h_neg = vec![0.0f64; RANK_BINS];
                            let inv_scale = 1.0 / scale;
                            let pair_temp = self.rank_pair_temperature.max(0.05);
                            let inv_temp = 1.0 / pair_temp;
                            let inv_temp_sq = inv_temp * inv_temp;

                            for b in 0..RANK_BINS {
                                let z = lo + b as f64 * inv_scale;
                                let mut s_neg = 0.0f64;
                                let mut h_neg = 0.0f64;
                                let mut s_pos = 0.0f64;
                                let mut h_pos = 0.0f64;
                                for c in 0..RANK_BINS {
                                    let zc = lo + c as f64 * inv_scale;
                                    let d = ((z - zc) * inv_temp).clamp(-50.0, 50.0);
                                    let s = 1.0 / (1.0 + (-d).exp());
                                    let h = (s * (1.0 - s)).max(1e-16);
                                    let nc = neg_hist[c];
                                    if nc > 0.0 {
                                        s_neg += nc * s;
                                        h_neg += nc * h;
                                    }
                                    let pc = pos_hist[c];
                                    if pc > 0.0 {
                                        s_pos += pc * s;
                                        h_pos += pc * h;
                                    }
                                }
                                rank_g_pos[b] = inv_temp * (s_neg / neg_total - 1.0);
                                rank_h_pos[b] = (inv_temp_sq * h_neg / neg_total).max(1e-16);
                                rank_g_neg[b] = inv_temp * s_pos / pos_total;
                                rank_h_neg[b] = (inv_temp_sq * h_pos / pos_total).max(1e-16);
                            }

                            let one_minus_a = 1.0 - alpha;
                            for i in 0..n {
                                // Log-loss component
                                let z = preds[i].clamp(-50.0, 50.0);
                                let p = 1.0 / (1.0 + (-z).exp());
                                let g_log = p - y[i];
                                let h_log = (p * (1.0 - p)).max(1e-16);
                                let b = bin_of(preds[i]);
                                let (g_rank, h_rank) = if y[i] > 0.5 {
                                    (rank_g_pos[b], rank_h_pos[b])
                                } else {
                                    (rank_g_neg[b], rank_h_neg[b])
                                };
                                // Blend
                                let w = if y[i] > 0.5 { pos_weight } else { neg_weight };
                                let focus = if focus_gamma > 0.0 {
                                    let err = if y[i] > 0.5 { 1.0 - p } else { p };
                                    (2.0 * err.clamp(0.0, 1.0)).powf(focus_gamma)
                                } else {
                                    1.0
                                };
                                gradients[i] = w * focus * (one_minus_a * g_log + alpha * g_rank);
                                hessians[i] =
                                    w * focus * (one_minus_a * h_log + alpha * h_rank).max(1e-16);
                            }
                        }
                    }
                } else if use_par {
                    gradients
                        .par_chunks_mut(1024)
                        .zip(hessians.par_chunks_mut(1024))
                        .enumerate()
                        .for_each(|(chunk_idx, (g_chunk, h_chunk))| {
                            let start = chunk_idx * 1024;
                            for (j, (g, h)) in
                                g_chunk.iter_mut().zip(h_chunk.iter_mut()).enumerate()
                            {
                                let i = start + j;
                                let p = 1.0 / (1.0 + (-preds[i]).exp());
                                let w = if y[i] > 0.5 { pos_weight } else { neg_weight };
                                let focus = if focus_gamma > 0.0 {
                                    let err = if y[i] > 0.5 { 1.0 - p } else { p };
                                    (2.0 * err.clamp(0.0, 1.0)).powf(focus_gamma)
                                } else {
                                    1.0
                                };
                                *g = w * focus * (p - y[i]);
                                *h = w * focus * (p * (1.0 - p)).max(1e-16);
                            }
                        });
                } else {
                    for i in 0..n {
                        let p = 1.0 / (1.0 + (-preds[i]).exp());
                        let w = if y[i] > 0.5 { pos_weight } else { neg_weight };
                        let focus = if focus_gamma > 0.0 {
                            let err = if y[i] > 0.5 { 1.0 - p } else { p };
                            (2.0 * err.clamp(0.0, 1.0)).powf(focus_gamma)
                        } else {
                            1.0
                        };
                        gradients[i] = w * focus * (p - y[i]);
                        hessians[i] = w * focus * (p * (1.0 - p)).max(1e-16);
                    }
                }
            }
            "poisson" => {
                for i in 0..n {
                    let mu = preds[i].exp().min(1e15);
                    gradients[i] = mu - y[i];
                    hessians[i] = mu.max(1e-16);
                }
            }
            "rank" => {
                // RankNet pairwise loss (Burges 2005). For each sample i, sample ONE
                // opposite-class opponent j. Pairwise gradient pushes f_i up / f_j down
                // proportional to rank violation sigmoid(f_i - f_j) distance from ideal.
                // Directly optimizes AUC rather than log-loss calibration.
                let mut pos_idx: Vec<usize> = Vec::new();
                let mut neg_idx: Vec<usize> = Vec::new();
                for i in 0..n {
                    if y[i] > 0.5 {
                        pos_idx.push(i);
                    } else {
                        neg_idx.push(i);
                    }
                }
                if pos_idx.is_empty() || neg_idx.is_empty() {
                    // Degenerate — fall back to log-loss
                    for i in 0..n {
                        let p = 1.0 / (1.0 + (-preds[i]).exp());
                        gradients[i] = p - y[i];
                        hessians[i] = (p * (1.0 - p)).max(1e-16);
                    }
                } else {
                    let n_pos = pos_idx.len();
                    let n_neg = neg_idx.len();
                    for i in 0..n {
                        let j = if y[i] > 0.5 {
                            neg_idx[(i.wrapping_mul(2654435761)) % n_neg]
                        } else {
                            pos_idx[(i.wrapping_mul(2654435761)) % n_pos]
                        };
                        let diff = preds[i] - preds[j];
                        let s = 1.0 / (1.0 + (-diff).exp());
                        if y[i] > 0.5 {
                            // positive: gradient σ(f_i - f_j) - 1 ∈ [-1, 0]. Leaf value > 0 pushes up.
                            gradients[i] = s - 1.0;
                        } else {
                            // negative: gradient σ(f_i - f_j) ∈ [0, 1]. Leaf value < 0 pushes down.
                            gradients[i] = s;
                        }
                        hessians[i] = (s * (1.0 - s)).max(1e-16);
                    }
                }
            }
            _ => {
                for i in 0..n {
                    gradients[i] = preds[i] - y[i];
                    hessians[i] = 1.0;
                }
            }
        }
    }

    pub(super) fn subsample_indices_rate(
        &self,
        rng: &mut StdRng,
        n_rows: usize,
        rate: f64,
    ) -> Vec<u32> {
        let sample_size = (rate * n_rows as f64) as usize;
        if sample_size >= n_rows && !self.use_bootstrap {
            return (0..n_rows as u32).collect();
        }
        if self.use_bootstrap {
            // Bootstrap: sample with replacement (RF-style bagging)
            // Each row has ~63.2% chance of appearing at least once
            let n = if sample_size >= n_rows {
                n_rows
            } else {
                sample_size
            };
            (0..n).map(|_| rng.random_range(0..n_rows) as u32).collect()
        } else {
            // Standard subsampling without replacement (GBM-style)
            let mut indices: Vec<u32> = (0..n_rows as u32).collect();
            for i in 0..sample_size {
                let j = rng.random_range(i..n_rows);
                indices.swap(i, j);
            }
            indices.truncate(sample_size);
            indices
        }
    }

    pub(super) fn subsample_indices(&self, rng: &mut StdRng, n_rows: usize) -> Vec<u32> {
        self.subsample_indices_rate(rng, n_rows, self.subsample_rate)
    }

    pub(super) fn round_subsamples(
        &self,
        rng: &mut StdRng,
        n_rows: usize,
        n_sub: usize,
    ) -> Vec<Vec<u32>> {
        if n_sub <= 1 {
            return vec![self.subsample_indices(rng, n_rows)];
        }
        if !self.antithetic_subtrees
            || self.use_bootstrap
            || self.subsample_rate <= 0.0
            || self.subsample_rate >= 1.0
        {
            return (0..n_sub)
                .map(|_| self.subsample_indices(rng, n_rows))
                .collect();
        }

        let sample_size = ((self.subsample_rate * n_rows as f64) as usize)
            .max(1)
            .min(n_rows.saturating_sub(1).max(1));
        let mut perm: Vec<u32> = (0..n_rows as u32).collect();
        perm.shuffle(rng);

        let mut out = Vec::with_capacity(n_sub);
        for sub_idx in 0..n_sub {
            let start = (sub_idx * n_rows) / n_sub;
            let mut sample = Vec::with_capacity(sample_size);
            for j in 0..sample_size {
                sample.push(perm[(start + j) % n_rows]);
            }
            out.push(sample);
        }
        out
    }

    /// NC-GOSS: Newton-Curriculum Gradient-based One-Side Sampling.
    ///
    /// Three modernizations over LightGBM's classic 2017 GOSS:
    ///   1. **Newton-gain importance** (default): rank rows by `g²/(h+λ)`
    ///      instead of `|g|`. This is the true expected single-split loss
    ///      reduction under Newton boosting. Rows with high |g| but low
    ///      hessian don't actually benefit from a split there.
    ///   2. **Curriculum annealing**: caller passes an effective `a_eff` that
    ///      decays from `a + anneal` (broad early) toward `a` (narrow late).
    ///      Same curriculum-learning logic as deep learning LR schedules.
    ///   3. **Per-class-max** for multiclass (caller's responsibility): pass
    ///      `max_k g_k²/h_k` per row rather than `sum_k`. Focuses on the most
    ///      confused class, mirroring focal-loss / hard-margin thinking.
    ///
    /// Inputs:
    ///   `importance[i]` — precomputed row importance (>= 0). Larger = harder.
    ///   `a_eff` — effective top-fraction (after annealing).
    ///   `b` — other-fraction (random-sample from bottom `1-a_eff`).
    ///
    /// Returns `(selected_indices, per_row_scales)`:
    ///   scale = 1.0 for top-set rows, `(1 - a_eff) / b` for other-set rows,
    ///   0.0 for rows not selected.
    pub(super) fn goss_select(
        &self,
        rng: &mut StdRng,
        importance: &[f64],
        a_eff: f64,
        b: f64,
    ) -> (Vec<u32>, Vec<f64>) {
        let n = importance.len();
        let a = a_eff.clamp(0.01, 0.99);
        let b = b.clamp(0.0, 0.99);
        let top_n = ((a * n as f64) as usize).max(1);
        let other_n = ((b * n as f64) as usize).max(0);

        // Sort row indices by importance descending.
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_unstable_by(|&i, &j| {
            importance[j as usize]
                .partial_cmp(&importance[i as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut scales = vec![0.0f64; n];
        let mut selected: Vec<u32> = Vec::with_capacity(top_n + other_n);

        for &i in &order[..top_n.min(n)] {
            selected.push(i);
            scales[i as usize] = 1.0;
        }

        if other_n > 0 && top_n < n {
            let remaining = &order[top_n..];
            let rem_len = remaining.len();
            let other_scale = if b > 0.0 { (1.0 - a) / b } else { 1.0 };
            let pick = other_n.min(rem_len);
            let mut pool: Vec<u32> = remaining.to_vec();
            for i in 0..pick {
                let j = rng.random_range(i..rem_len);
                pool.swap(i, j);
            }
            for &i in &pool[..pick] {
                selected.push(i);
                scales[i as usize] = other_scale;
            }
        }
        (selected, scales)
    }

    /// Compute per-row importance vector using the configured `goss_mode`.
    ///   "newton"   (default): g²/(h + λ_reg) — expected single-split gain
    ///   "classic"           : |g| — original LightGBM GOSS metric
    /// For multi-output / multiclass, caller passes (g, h) flattened with
    /// one class's values — or aggregates via `goss_importance_multi`.
    pub(super) fn goss_importance(&self, g: &[f64], h: &[f64]) -> Vec<f64> {
        let lam = self.lambda_reg.max(1e-6);
        match self.goss_mode.as_str() {
            "classic" => g.iter().map(|v| v.abs()).collect(),
            _ => g
                .iter()
                .zip(h.iter())
                .map(|(&gi, &hi)| (gi * gi) / (hi + lam))
                .collect(),
        }
    }

    /// Per-class-max importance for multiclass: importance[i] = max_k f(g[k,i], h[k,i])
    /// where f depends on goss_mode. Pass row-major grads [n_classes × n_rows].
    pub(super) fn goss_importance_multi(
        &self,
        g: &[f64],
        h: &[f64],
        n_rows: usize,
        n_classes: usize,
    ) -> Vec<f64> {
        let lam = self.lambda_reg.max(1e-6);
        let is_classic = self.goss_mode == "classic";
        let mut imp = vec![0.0f64; n_rows];
        for i in 0..n_rows {
            let mut best = 0.0f64;
            for k in 0..n_classes {
                let gi = g[k * n_rows + i];
                let hi = h[k * n_rows + i];
                let val = if is_classic {
                    gi.abs()
                } else {
                    (gi * gi) / (hi + lam)
                };
                if val > best {
                    best = val;
                }
            }
            imp[i] = best;
        }
        imp
    }

    /// EBLP: Empirical Bayes Leaf Prior.
    ///
    /// Estimate the prior variance σ²_π from the empirical distribution of
    /// leaf values across all previously-trained trees, then compute a
    /// James-Stein-style shrinkage factor to apply to the CURRENT tree's
    /// leaf values:
    ///
    ///   σ²_π ≈ Var({w : w is a leaf value in past trees})
    ///   shrinkage = σ²_π / (σ²_π + 1)     # classical James-Stein under unit noise
    ///
    /// Factor ∈ [0, 1]. When σ²_π is large (diverse leaves = real signal),
    /// factor → 1 (no shrinkage). When σ²_π is small (concentrated leaves =
    /// likely noise), factor → 0 (aggressive shrinkage toward 0).
    ///
    /// Returns 1.0 (no-op) if EBLP disabled or too few trees for a stable
    /// variance estimate.
    pub(super) fn eblp_shrinkage_factor(&self) -> f64 {
        if !self.leaf_eb || self.trees.len() < self.leaf_eb_min_trees {
            return 1.0;
        }
        let mut count = 0usize;
        let mut sum = 0.0f64;
        let mut sq = 0.0f64;
        for tree in &self.trees {
            for &v in &tree.values {
                if v.is_finite() {
                    sum += v;
                    sq += v * v;
                    count += 1;
                }
            }
        }
        if count < 4 {
            return 1.0;
        }
        let n = count as f64;
        let mean = sum / n;
        let var = (sq / n - mean * mean).max(0.0);
        let sigma2 = var * self.leaf_eb_scale;
        if sigma2 < 1e-12 {
            return 0.1; // all past leaves near zero → hard shrinkage
        }
        sigma2 / (sigma2 + 1.0)
    }

    /// HSS: Hierarchical Sibling Smoothing.
    ///
    /// For each internal node whose TWO CHILDREN are both leaves, blend the
    /// two leaves' values toward their midpoint:
    ///     μ     = (w_L + w_R) / 2
    ///     w_L' = (1-α)·w_L + α·μ
    ///     w_R' = (1-α)·w_R + α·μ
    ///
    /// Rationale: sibling leaves come from the SAME split decision. If the
    /// split is signal-driven, w_L and w_R legitimately differ. If it's
    /// noise-driven, w_L ≈ -w_R and μ ≈ 0 → smoothing pulls both to zero,
    /// neutralizing the spurious split.
    ///
    /// Contrast with `lambda_reg` (per-leaf independent shrinkage toward 0)
    /// and LightGBM's `path_smooth` (leaf toward PARENT value). HSS uses
    /// the TREE TOPOLOGY to exploit sibling correlation — nobody has shipped
    /// this in a production GBM.
    ///
    /// Ramp slopes and cat lookups are also blended pairwise so all
    /// leaf-attached learned quantities shrink consistently.
    /// SCS — Sign-Confidence Shrinkage. Per-leaf shrinkage factor = (|Σsign(g)|/n)^γ.
    /// Leaves with all-same-sign gradients (high confidence) keep their value; mixed-sign
    /// leaves (indecisive signal) shrink toward 0. Fit-time, local, per-tree.
    pub(super) fn apply_scs(
        &self,
        tree: &mut DecisionTree,
        binned: &BinnedData,
        gradients: &[f64],
        n_rows: usize,
    ) {
        if self.sign_confidence_gamma <= 0.0 {
            return;
        }
        // Binary only — regression lacks meaningful sign structure at optimum;
        // multi_output_tree multiclass gets hit by per-class shrinkage mismatch.
        if self.task != "binary" {
            return;
        }
        let gamma = self.sign_confidence_gamma;
        let n_nodes = tree.values.len();
        let mut sign_sum = vec![0i64; n_nodes];
        let mut count = vec![0i64; n_nodes];
        for i in 0..n_rows {
            let leaf = tree.route_to_leaf(binned, i);
            let g = gradients[i];
            if g > 0.0 {
                sign_sum[leaf] += 1;
            } else if g < 0.0 {
                sign_sum[leaf] -= 1;
            }
            count[leaf] += 1;
        }
        for j in 0..n_nodes {
            if tree.split_features[j] == u32::MAX && count[j] > 0 {
                let agreement = (sign_sum[j].unsigned_abs() as f64) / (count[j] as f64);
                let scale = if gamma == 1.0 {
                    agreement
                } else {
                    agreement.powf(gamma)
                };
                tree.values[j] *= scale;
                // Also shrink ramp / quad slopes to stay consistent.
                if !tree.ramp_slopes.is_empty() && tree.ramp_k > 0 {
                    let base = j * tree.ramp_k;
                    for k in 0..tree.ramp_k {
                        if base + k < tree.ramp_slopes.len() {
                            tree.ramp_slopes[base + k] *= scale;
                        }
                    }
                }
                if j < tree.leaf_pair_slopes.len() {
                    tree.leaf_pair_slopes[j] *= scale;
                }
                let qi = tree.quad_n_interactions;
                if qi > 0 && !tree.quad_slopes.is_empty() {
                    let base = j * qi;
                    for k in 0..qi {
                        if base + k < tree.quad_slopes.len() {
                            tree.quad_slopes[base + k] *= scale;
                        }
                    }
                }
            }
        }
    }

    pub(super) fn apply_hss(&self, tree: &mut DecisionTree) {
        if self.leaf_sibling_smooth <= 0.0 {
            return;
        }
        let alpha = self.leaf_sibling_smooth;
        let half_alpha = alpha * 0.5;
        let n_nodes = tree.values.len();
        for node in 0..n_nodes {
            // Only consider internal nodes (non-leaf)
            if tree.split_features[node] == u32::MAX {
                continue;
            }
            let left = tree.left_children[node] as usize;
            let right = tree.right_children[node] as usize;
            if left >= n_nodes || right >= n_nodes {
                continue;
            }
            // Require BOTH children to be leaves (we only blend leaf siblings)
            if tree.split_features[left] != u32::MAX {
                continue;
            }
            if tree.split_features[right] != u32::MAX {
                continue;
            }

            // Blend main leaf values
            let w_l = tree.values[left];
            let w_r = tree.values[right];
            tree.values[left] = w_l + half_alpha * (w_r - w_l);
            tree.values[right] = w_r + half_alpha * (w_l - w_r);

            // Blend ramp slopes (per-node slope from piecewise linear refinement)
            if !tree.ramp_slopes.is_empty() && tree.ramp_k > 0 {
                let base_l = left * tree.ramp_k;
                let base_r = right * tree.ramp_k;
                if base_l + tree.ramp_k <= tree.ramp_slopes.len()
                    && base_r + tree.ramp_k <= tree.ramp_slopes.len()
                {
                    for k in 0..tree.ramp_k {
                        let s_l = tree.ramp_slopes[base_l + k];
                        let s_r = tree.ramp_slopes[base_r + k];
                        tree.ramp_slopes[base_l + k] = s_l + half_alpha * (s_r - s_l);
                        tree.ramp_slopes[base_r + k] = s_r + half_alpha * (s_l - s_r);
                    }
                }
            }
            if left < tree.leaf_pair_slopes.len() && right < tree.leaf_pair_slopes.len() {
                let s_l = tree.leaf_pair_slopes[left];
                let s_r = tree.leaf_pair_slopes[right];
                tree.leaf_pair_slopes[left] = s_l + half_alpha * (s_r - s_l);
                tree.leaf_pair_slopes[right] = s_r + half_alpha * (s_l - s_r);
            }

            // Blend quad slopes (each child has quad_n_interactions entries)
            let qi = tree.quad_n_interactions;
            if qi > 0 && !tree.quad_slopes.is_empty() {
                let base_l = left * qi;
                let base_r = right * qi;
                if base_l + qi <= tree.quad_slopes.len() && base_r + qi <= tree.quad_slopes.len() {
                    for k in 0..qi {
                        let q_l = tree.quad_slopes[base_l + k];
                        let q_r = tree.quad_slopes[base_r + k];
                        tree.quad_slopes[base_l + k] = q_l + half_alpha * (q_r - q_l);
                        tree.quad_slopes[base_r + k] = q_r + half_alpha * (q_l - q_r);
                    }
                }
            }
        }
    }

    pub(super) fn apply_newton_trust_region(&self, tree: &mut DecisionTree) {
        let cap = self.newton_decrement_cap;
        if cap <= 0.0 || tree.values.is_empty() || tree.node_h_sum.is_empty() {
            return;
        }
        let qi = tree.quad_n_interactions;
        for node in 0..tree.values.len() {
            let h = tree.node_h_sum.get(node).copied().unwrap_or(0.0);
            let denom = h + self.lambda_reg;
            if denom <= 1e-12 {
                continue;
            }
            let max_abs = cap / denom.sqrt();
            let val = tree.values[node];
            let abs_val = val.abs();
            if abs_val <= max_abs || max_abs <= 0.0 {
                continue;
            }
            let scale = max_abs / abs_val;
            tree.values[node] *= scale;
            if let Some(ref mut lookup) = tree.cat_lookups[node] {
                lookup.default_value *= scale;
                for v in lookup.bin_values.iter_mut() {
                    *v *= scale;
                }
            }
            if !tree.ramp_slopes.is_empty() && tree.ramp_k > 0 {
                let base = node * tree.ramp_k;
                for k in 0..tree.ramp_k {
                    if base + k < tree.ramp_slopes.len() {
                        tree.ramp_slopes[base + k] *= scale;
                    }
                }
            }
            if node < tree.leaf_pair_slopes.len() {
                tree.leaf_pair_slopes[node] *= scale;
            }
            if qi > 0 && !tree.quad_slopes.is_empty() {
                let base = node * qi;
                for k in 0..qi {
                    if base + k < tree.quad_slopes.len() {
                        tree.quad_slopes[base + k] *= scale;
                    }
                }
            }
        }
    }

    /// Apply EBLP shrinkage to a newly-built tree's leaf values (in place).
    /// Also scales ramp slopes and cat lookups if present so all learned
    /// leaf-attached quantities shrink consistently.
    pub(super) fn apply_eblp(&self, tree: &mut DecisionTree) {
        if !self.leaf_eb || self.trees.len() < self.leaf_eb_min_trees {
            return;
        }
        let s = self.eblp_shrinkage_factor();
        if (s - 1.0).abs() < 1e-9 {
            return;
        }
        for v in tree.values.iter_mut() {
            *v *= s;
        }
        tree.scale_ramp_slopes(s);
        tree.scale_cat_lookups(s);
    }

    pub(super) fn cyclic_feature_pressure(
        &self,
        binned: &BinnedData,
        feature: usize,
        gradients: &[f64],
        hessians: &[f64],
        lambda: f64,
    ) -> f64 {
        if feature >= binned.n_features || binned.n_rows == 0 {
            return 0.0;
        }
        let mut sum_g: Vec<f64> = Vec::new();
        let mut sum_h: Vec<f64> = Vec::new();
        let mut fold_g: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
        let mut fold_h: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
        let mut fold_total_g = [0.0f64; 2];
        let mut fold_total_h = [0.0f64; 2];
        let mut total_g = 0.0f64;
        let mut total_h = 0.0f64;
        let offset = feature * binned.n_rows;
        for row in 0..binned.n_rows {
            let bin = binned.bin_indices[offset + row];
            if bin == MISSING_BIN {
                continue;
            }
            let b = bin as usize;
            if b >= sum_g.len() {
                if b > 4096 {
                    continue;
                }
                sum_g.resize(b + 1, 0.0);
                sum_h.resize(b + 1, 0.0);
                fold_g[0].resize(b + 1, 0.0);
                fold_g[1].resize(b + 1, 0.0);
                fold_h[0].resize(b + 1, 0.0);
                fold_h[1].resize(b + 1, 0.0);
            }
            let g = gradients[row];
            let h = hessians[row].max(1e-12);
            sum_g[b] += g;
            sum_h[b] += h;
            total_g += g;
            total_h += h;
            let mut hash = ((row as u64).wrapping_mul(0xD6E8_FD9D_50D5_1735))
                ^ ((feature as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            hash ^= hash >> 33;
            let fold = (hash as usize) & 1;
            fold_g[fold][b] += g;
            fold_h[fold][b] += h;
            fold_total_g[fold] += g;
            fold_total_h[fold] += h;
        }
        if total_h <= 1e-12 || sum_g.len() <= 1 {
            return 0.0;
        }
        let lam = lambda.max(1e-12);
        let mut score = 0.0f64;
        for b in 0..sum_g.len() {
            if sum_h[b] > 1e-12 {
                score += (sum_g[b] * sum_g[b]) / (sum_h[b] + lam);
            }
        }
        score -= (total_g * total_g) / (total_h + lam);
        if !score.is_finite() || score <= 0.0 {
            return 0.0;
        }
        let full_pressure = score;
        if self.split_pessimism <= 0.0 || binned.n_rows < 64 {
            return full_pressure;
        }

        let mut fold_pressure = [0.0f64; 2];
        for fold in 0..2 {
            if fold_total_h[fold] <= 1e-12 {
                return full_pressure;
            }
            let mut fp = 0.0f64;
            for b in 0..fold_g[fold].len() {
                let bh = fold_h[fold][b];
                if bh > 1e-12 {
                    fp += (fold_g[fold][b] * fold_g[fold][b]) / (bh + lam);
                }
            }
            fp -= (fold_total_g[fold] * fold_total_g[fold]) / (fold_total_h[fold] + lam);
            if !fp.is_finite() || fp <= 0.0 {
                fold_pressure[fold] = 0.0;
            } else {
                fold_pressure[fold] = fp;
            }
        }
        let stable_full_equiv = 2.0 * fold_pressure[0].min(fold_pressure[1]);
        if stable_full_equiv <= 0.0 {
            return full_pressure * (1.0 - self.split_pessimism.clamp(0.0, 0.5));
        }
        let stability = (stable_full_equiv / (full_pressure + 1e-12)).clamp(0.0, 1.0);
        let width = (sum_g.len().max(2) as f64).ln().max(1.0);
        let shrink = (1.0 - self.split_pessimism.clamp(0.0, 0.5) * width * (1.0 - stability))
            .clamp(0.35, 1.0);
        full_pressure.min(stable_full_equiv) * shrink
    }

    pub(super) fn cyclic_feature_order_by_pressure(
        &self,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        lambda: f64,
    ) -> Vec<usize> {
        let mut scored: Vec<(usize, f64)> = (0..binned.n_features)
            .map(|f| {
                (
                    f,
                    self.cyclic_feature_pressure(binned, f, gradients, hessians, lambda),
                )
            })
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.into_iter().map(|(f, _)| f).collect()
    }

    pub(super) fn pressure_effective_dimension(
        &self,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        lambda: f64,
    ) -> f64 {
        if binned.n_features == 0 {
            return 0.0;
        }
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for feature in 0..binned.n_features {
            let p = self
                .cyclic_feature_pressure(binned, feature, gradients, hessians, lambda)
                .max(0.0);
            if p > 0.0 && p.is_finite() {
                sum += p;
                sum_sq += p * p;
            }
        }
        if sum <= 0.0 || sum_sq <= 0.0 {
            0.0
        } else {
            (sum * sum / sum_sq).clamp(1.0, binned.n_features.max(1) as f64)
        }
    }

    pub(super) fn adaptive_subtree_count_by_pressure(
        &self,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        lambda: f64,
        requested: usize,
    ) -> usize {
        if requested <= 1 || binned.n_features <= 1 {
            return requested.max(1);
        }
        let eff_dim = self.pressure_effective_dimension(binned, gradients, hessians, lambda);
        if eff_dim <= 1.0 {
            return 1;
        }
        let supported = (eff_dim / 2.0).sqrt().floor() as usize;
        supported.clamp(1, requested)
    }

    pub(super) fn main_effect_from_scored_pressures(
        &self,
        mut scored: Vec<(usize, f64)>,
        excluded_feature: Option<usize>,
    ) -> Option<usize> {
        scored.retain(|(f, s)| Some(*f) != excluded_feature && s.is_finite() && *s > 0.0);
        if scored.is_empty() {
            return None;
        }
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        let total: f64 = scored.iter().map(|(_, s)| *s).sum();
        if !total.is_finite() || total <= 1e-12 {
            return None;
        }
        let best = scored[0].1;
        let second = scored.get(1).map(|(_, s)| *s).unwrap_or(0.0);
        let n_eff = scored.len().max(1) as f64;
        let share = best / total;
        let ratio = best / (second + 1e-12);
        let min_share = (2.0 / n_eff).clamp(0.045, 0.22);
        let absolute_floor = self.gamma.max(0.0) + 1e-12;

        if best > absolute_floor && ((share >= min_share && ratio >= 1.03) || ratio >= 1.75) {
            Some(scored[0].0)
        } else {
            None
        }
    }

    pub(super) fn main_effect_feature_by_pressure(
        &self,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        lambda: f64,
        excluded_feature: Option<usize>,
    ) -> Option<usize> {
        let scored: Vec<(usize, f64)> = (0..binned.n_features)
            .map(|f| {
                (
                    f,
                    self.cyclic_feature_pressure(binned, f, gradients, hessians, lambda),
                )
            })
            .collect();
        self.main_effect_from_scored_pressures(scored, excluded_feature)
    }

    pub(super) fn main_effect_due(&self, round: usize, n_rounds: usize) -> bool {
        if self.main_effect_interval == 0 {
            return false;
        }
        let _ = n_rounds;
        (round + 1) % self.main_effect_interval == 0
    }

    pub(super) fn cyclic_revisit_budget(&self) -> usize {
        if !self.cyclic_feature_reuse {
            return 0;
        }
        if self.cyclic_revisit_trees > 0 {
            self.cyclic_revisit_trees.min(16)
        } else {
            self.n_trees_per_round.max(1).min(16)
        }
    }

    pub(super) fn subtrees_per_boosting_round(&self, n_feat: usize) -> usize {
        if !self.cyclic_features {
            return self.n_trees_per_round.max(1);
        }
        let base = if self.cyclic_max_features_per_round > 0 {
            self.cyclic_max_features_per_round.min(n_feat).max(1)
        } else {
            n_feat.max(1)
        };
        if self.adaptive_cyclic_order && self.cyclic_feature_reuse {
            base + self.cyclic_revisit_budget()
        } else {
            base
        }
    }

    pub(super) fn max_cyclic_feature_pressure(
        &self,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        lambda: f64,
    ) -> f64 {
        let mut best = 0.0f64;
        for feature in 0..binned.n_features {
            let pressure =
                self.cyclic_feature_pressure(binned, feature, gradients, hessians, lambda);
            if pressure > best {
                best = pressure;
            }
        }
        best
    }

    pub(super) fn cyclic_pair_pressure(
        &self,
        binned: &BinnedData,
        feature_a: usize,
        feature_b: usize,
        gradients: &[f64],
        hessians: &[f64],
        lambda: f64,
    ) -> f64 {
        if feature_a >= binned.n_features
            || feature_b >= binned.n_features
            || feature_a == feature_b
            || binned.n_rows == 0
            || binned
                .is_categorical
                .get(feature_a)
                .copied()
                .unwrap_or(false)
            || binned
                .is_categorical
                .get(feature_b)
                .copied()
                .unwrap_or(false)
        {
            return 0.0;
        }
        let bins_a = binned.n_bins(feature_a).max(1);
        let bins_b = binned.n_bins(feature_b).max(1);
        if bins_a <= 1 || bins_b <= 1 {
            return 0.0;
        }
        let n_bins = self.cyclic_partner_bins.clamp(2, 32);
        let n_cells = n_bins * n_bins;
        let mut sum_g = vec![0.0f64; n_cells];
        let mut sum_h = vec![0.0f64; n_cells];
        let mut total_g = 0.0f64;
        let mut total_h = 0.0f64;
        let off_a = feature_a * binned.n_rows;
        let off_b = feature_b * binned.n_rows;
        for row in 0..binned.n_rows {
            let ba_raw = binned.bin_indices[off_a + row];
            let bb_raw = binned.bin_indices[off_b + row];
            if ba_raw == MISSING_BIN || bb_raw == MISSING_BIN {
                continue;
            }
            let ba_bin = ba_raw as usize;
            let bb_bin = bb_raw as usize;
            if ba_bin >= bins_a || bb_bin >= bins_b {
                continue;
            }
            let ba = ((ba_bin * n_bins) / bins_a).min(n_bins - 1);
            let bb = ((bb_bin * n_bins) / bins_b).min(n_bins - 1);
            let cell = ba * n_bins + bb;
            let g = gradients[row];
            let h = hessians[row].max(1e-12);
            sum_g[cell] += g;
            sum_h[cell] += h;
            total_g += g;
            total_h += h;
        }
        if total_h <= 1e-12 {
            return 0.0;
        }
        let lam = lambda.max(1e-12);
        let mut score = 0.0f64;
        let mut active = 0usize;
        for cell in 0..n_cells {
            if sum_h[cell] > 1e-12 {
                active += 1;
                score += (sum_g[cell] * sum_g[cell]) / (sum_h[cell] + lam);
            }
        }
        if active < 2 {
            return 0.0;
        }
        // FAST-style INTERACTION-PURE score: subtract both marginal (1-D) scores
        // computed on the same grid, so the pair score measures joint structure
        // BEYOND what either feature's own shape already explains. Without this
        // the partner choice is dominated by strong-main features, which the
        // cyclic mains fit anyway (verified inert on real data).
        let mut row_g = vec![0.0f64; n_bins];
        let mut row_h = vec![0.0f64; n_bins];
        let mut col_g = vec![0.0f64; n_bins];
        let mut col_h = vec![0.0f64; n_bins];
        for a in 0..n_bins {
            for b in 0..n_bins {
                let cell = a * n_bins + b;
                row_g[a] += sum_g[cell];
                row_h[a] += sum_h[cell];
                col_g[b] += sum_g[cell];
                col_h[b] += sum_h[cell];
            }
        }
        let parent = (total_g * total_g) / (total_h + lam);
        let marginal = |gs: &[f64], hs: &[f64]| -> f64 {
            let mut s = 0.0f64;
            for k in 0..n_bins {
                if hs[k] > 1e-12 {
                    s += (gs[k] * gs[k]) / (hs[k] + lam);
                }
            }
            (s - parent).max(0.0)
        };
        score -= parent;
        score -= marginal(&row_g, &row_h);
        score -= marginal(&col_g, &col_h);
        if score.is_finite() {
            score.max(0.0)
        } else {
            0.0
        }
    }

    pub(super) fn best_cyclic_partner_by_pair_pressure(
        &self,
        binned: &BinnedData,
        primary: usize,
        gradients: &[f64],
        hessians: &[f64],
        lambda: f64,
    ) -> Option<usize> {
        if primary >= binned.n_features || binned.n_features <= 1 {
            return None;
        }
        let primary_pressure =
            self.cyclic_feature_pressure(binned, primary, gradients, hessians, lambda);
        let min_score = self.cyclic_partner_min_pressure_ratio.max(0.0) * primary_pressure;
        let mut best_partner = None;
        let mut best_score = min_score;
        for partner in 0..binned.n_features {
            if partner == primary {
                continue;
            }
            let score =
                self.cyclic_pair_pressure(binned, primary, partner, gradients, hessians, lambda);
            if score > best_score {
                best_score = score;
                best_partner = Some(partner);
            }
        }
        best_partner
    }

    pub(super) fn take_best_cyclic_feature_by_pressure(
        &self,
        remaining: &mut Vec<usize>,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        lambda: f64,
    ) -> Option<usize> {
        if remaining.is_empty() {
            return None;
        }
        let mut best_pos = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for (pos, &feature) in remaining.iter().enumerate() {
            let score = self.cyclic_feature_pressure(binned, feature, gradients, hessians, lambda);
            if score > best_score {
                best_score = score;
                best_pos = pos;
            }
        }
        Some(remaining.remove(best_pos))
    }

    pub(super) fn take_cyclic_feature_by_residual_auction(
        &self,
        usage: &mut [usize],
        last_feature: Option<usize>,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        lambda: f64,
    ) -> Option<usize> {
        if usage.is_empty() {
            return None;
        }
        let mut best_feature = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for feature in 0..usage.len().min(binned.n_features) {
            let pressure =
                self.cyclic_feature_pressure(binned, feature, gradients, hessians, lambda);
            let reuse_penalty = 1.0 + 0.65 * usage[feature] as f64;
            let repeat_penalty = if Some(feature) == last_feature {
                1.35
            } else {
                1.0
            };
            let score = pressure / (reuse_penalty * repeat_penalty);
            if score > best_score {
                best_score = score;
                best_feature = feature;
            }
        }
        usage[best_feature] += 1;
        Some(best_feature)
    }

    pub(super) fn take_adaptive_root_anchor_by_pressure(
        &self,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        lambda: f64,
        usage: &mut [usize],
    ) -> Option<usize> {
        if usage.is_empty() || binned.n_features == 0 {
            return None;
        }
        let penalty = self.adaptive_root_anchor_penalty;
        let mut best_feature = None;
        let mut best_score = f64::NEG_INFINITY;
        for feature in 0..usage.len().min(binned.n_features) {
            let pressure =
                self.cyclic_feature_pressure(binned, feature, gradients, hessians, lambda);
            if pressure <= 0.0 || !pressure.is_finite() {
                continue;
            }
            let reuse = usage[feature] as f64;
            let score = pressure / (1.0 + penalty * reuse);
            if score > best_score {
                best_score = score;
                best_feature = Some(feature);
            }
        }
        if let Some(feature) = best_feature {
            usage[feature] += 1;
            Some(feature)
        } else {
            None
        }
    }

    pub(super) fn make_adaptive_feature_mask_by_pressure(
        &self,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        lambda: f64,
        usage: &mut [usize],
    ) -> Vec<bool> {
        let n_features = binned.n_features;
        let mut mask = vec![false; n_features];
        if n_features == 0 {
            return mask;
        }
        let n_select = ((self.colsample_bytree.clamp(0.0, 1.0) * n_features as f64).ceil()
            as usize)
            .max(1)
            .min(n_features);
        if n_select >= n_features && usage.is_empty() {
            mask.fill(true);
            return mask;
        }

        let penalty = self.adaptive_feature_mask_penalty;
        let mut scored: Vec<(usize, f64)> = (0..n_features)
            .map(|feature| {
                let pressure =
                    self.cyclic_feature_pressure(binned, feature, gradients, hessians, lambda);
                let reuse = usage.get(feature).copied().unwrap_or(0) as f64;
                (feature, pressure / (1.0 + penalty * reuse))
            })
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        for &(feature, _) in scored.iter().take(n_select) {
            mask[feature] = true;
            if let Some(u) = usage.get_mut(feature) {
                *u += 1;
            }
        }
        mask
    }

    pub(super) fn apply_sibling_block_correction(
        &mut self,
        binned: &BinnedData,
        y: &[f64],
        n_rows: usize,
        trees_start: usize,
        round_base: &[f64],
        eval_data: Option<&(Vec<u16>, Vec<f64>, usize, Vec<f64>, Vec<u16>)>,
        eval_base: &[f64],
        oob_masks: Option<&[Option<Vec<u64>>]>,
        predictions: &mut [f64],
        eval_preds: &mut [f64],
        oob_pred_sum: &mut [f64],
        oob_pred_sum_sq: &mut [f64],
    ) {
        let trees_end = self.trees.len();
        let m = trees_end.saturating_sub(trees_start);
        if m < 2 || round_base.len() != n_rows {
            return;
        }
        if let Some(masks) = oob_masks {
            if masks.len() != m || oob_pred_sum.len() != n_rows || oob_pred_sum_sq.len() != n_rows {
                return;
            }
        }

        let mut contribs: Vec<Vec<f64>> = Vec::with_capacity(m);
        for t_idx in trees_start..trees_end {
            let lr_weight =
                self.learning_rate * self.dart_tree_weights.get(t_idx).copied().unwrap_or(1.0);
            let tree = &self.trees[t_idx];
            let mut z = vec![0.0f64; n_rows];
            for i in 0..n_rows {
                z[i] = lr_weight * tree.predict_binned(binned, i);
            }
            contribs.push(z);
        }

        let inv_n = 1.0 / n_rows.max(1) as f64;
        let mut a = vec![vec![0.0f64; m]; m];
        let mut rhs = vec![0.0f64; m];
        if self.task == "binary" {
            let (neg_weight, pos_weight) = if self.class_weights.len() >= 2 {
                let neg = if self.class_weights[0].is_finite() && self.class_weights[0] > 0.0 {
                    self.class_weights[0]
                } else {
                    1.0
                };
                let pos = if self.class_weights[1].is_finite() && self.class_weights[1] > 0.0 {
                    self.class_weights[1]
                } else {
                    1.0
                };
                (neg, pos)
            } else {
                (1.0, 1.0)
            };
            for j in 0..m {
                for i in 0..n_rows {
                    let z = round_base[i].clamp(-50.0, 50.0);
                    let p = 1.0 / (1.0 + (-z).exp());
                    let w = if y[i] > 0.5 { pos_weight } else { neg_weight };
                    let h = w * (p * (1.0 - p)).max(1e-16);
                    let neg_g = w * (y[i] - p);
                    rhs[j] += contribs[j][i] * neg_g;
                    for k in j..m {
                        a[j][k] += h * contribs[j][i] * contribs[k][i];
                    }
                }
                rhs[j] *= inv_n;
            }
            for j in 0..m {
                for k in j..m {
                    let v = a[j][k] * inv_n;
                    a[j][k] = v;
                    a[k][j] = v;
                }
            }
        } else {
            for j in 0..m {
                for i in 0..n_rows {
                    let r = y[i] - round_base[i];
                    rhs[j] += contribs[j][i] * r;
                }
                rhs[j] *= inv_n;
                for k in j..m {
                    let mut dot = 0.0f64;
                    for i in 0..n_rows {
                        dot += contribs[j][i] * contribs[k][i];
                    }
                    let v = dot * inv_n;
                    a[j][k] = v;
                    a[k][j] = v;
                }
            }
        }

        let diag_mean = (0..m).map(|j| a[j][j]).sum::<f64>() / m as f64;
        if diag_mean <= 1e-18 || !diag_mean.is_finite() {
            return;
        }
        let ridge = 0.15 * diag_mean + 1e-12;
        for j in 0..m {
            a[j][j] += ridge;
            rhs[j] += ridge;
        }

        let Some(solution) = solve_small_linear_system(a, rhs) else {
            return;
        };
        let alpha = self.sibling_block_correction.clamp(0.0, 1.0);
        let mut coefs = vec![1.0f64; m];
        let mut changed = false;
        for j in 0..m {
            let target = solution[j].clamp(0.0, 2.5);
            let coef = 1.0 + alpha * (target - 1.0);
            if (coef - 1.0).abs() > 1e-4 {
                changed = true;
            }
            coefs[j] = coef;
        }
        if !changed {
            return;
        }

        predictions.copy_from_slice(round_base);
        for j in 0..m {
            let coef = coefs[j];
            for i in 0..n_rows {
                predictions[i] += coef * contribs[j][i];
            }
        }

        if let Some((eval_bins, _, en, _, eval_cll_bins)) = eval_data {
            let en = *en;
            if !eval_base.is_empty() && eval_preds.len() >= en && eval_base.len() >= en {
                eval_preds[..en].copy_from_slice(&eval_base[..en]);
                for (j, t_idx) in (trees_start..trees_end).enumerate() {
                    let lr_weight = self.learning_rate
                        * self.dart_tree_weights.get(t_idx).copied().unwrap_or(1.0);
                    let tree = &self.trees[t_idx];
                    let coef = coefs[j];
                    for i in 0..en {
                        eval_preds[i] += coef
                            * lr_weight
                            * tree.predict_binned_raw(eval_bins, en, i, eval_cll_bins);
                    }
                }
            }
        }

        if let Some(masks) = oob_masks {
            for j in 0..m {
                if let Some(ref mask) = masks[j] {
                    let coef = coefs[j];
                    let delta = coef - 1.0;
                    let sq_delta = coef * coef - 1.0;
                    for i in 0..n_rows {
                        if !bitvec_test(mask, i) {
                            let z = contribs[j][i];
                            oob_pred_sum[i] += delta * z;
                            oob_pred_sum_sq[i] += sq_delta * z * z;
                        }
                    }
                }
            }
        }

        for (j, t_idx) in (trees_start..trees_end).enumerate() {
            self.trees[t_idx].scale_output(coefs[j]);
        }
    }

    pub(super) fn apply_hierarchical_shrinkage(&self, tree: &mut DecisionTree) {
        if self.hierarchical_shrinkage <= 0.0 {
            return;
        }
        tree.hierarchical_shrink_experts(self.hierarchical_shrinkage);
    }

    /// Curriculum-annealed effective top_rate.
    ///   a_eff(r) = base_a + anneal * (1 - r/n_rounds)
    /// So early rounds (r ≈ 0) have wider selection (a + anneal), late rounds
    /// converge to base_a. Linear schedule.
    pub(super) fn goss_annealed_a(&self, round: usize, n_rounds: usize) -> f64 {
        let base = self.goss_top_rate;
        let ann = self.goss_anneal;
        if ann <= 0.0 || n_rounds <= 1 {
            return base;
        }
        let progress = round as f64 / (n_rounds - 1).max(1) as f64;
        (base + ann * (1.0 - progress)).clamp(0.01, 0.99)
    }

    pub(super) fn make_feature_mask(&self, rng: &mut StdRng, n_features: usize) -> Vec<bool> {
        if self.colsample_bytree >= 1.0 && self.diversity_penalty <= 0.0 {
            return vec![true; n_features];
        }
        let n_select = if self.colsample_bytree >= 1.0 {
            n_features
        } else {
            ((self.colsample_bytree * n_features as f64) as usize).max(1)
        };
        // Diversity-weighted sampling: features with high recent usage get
        // exponentially smaller probability via -penalty * ema[f].
        if self.diversity_penalty > 0.0 && self.feature_usage_ema.len() == n_features {
            let penalty = self.diversity_penalty;
            let ema_max = self
                .feature_usage_ema
                .iter()
                .cloned()
                .fold(0.0f64, f64::max)
                .max(1e-9);
            // Gumbel-style weighted top-k without replacement:
            // score = ln(-ln(U)) - weight; take smallest scores.
            // weight[f] = -penalty * ema[f]/ema_max (so high usage = smaller weight = less likely).
            let mut scored: Vec<(f64, usize)> = (0..n_features)
                .map(|f| {
                    let u: f64 = rng.random::<f64>().max(1e-15);
                    let gumbel = -(-u.ln()).ln();
                    let ema_n = self.feature_usage_ema[f] / ema_max;
                    (gumbel - penalty * ema_n, f)
                })
                .collect();
            // take top-k by largest score
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut mask = vec![false; n_features];
            for &(_, idx) in &scored[..n_select.min(n_features)] {
                mask[idx] = true;
            }
            return mask;
        }
        let mut indices: Vec<usize> = (0..n_features).collect();
        indices.shuffle(rng);
        let mut mask = vec![false; n_features];
        for &idx in &indices[..n_select.min(n_features)] {
            mask[idx] = true;
        }
        mask
    }

    /// Refresh feature-usage EMA from all trees built so far. Called once per
    /// round when diversity_penalty > 0. Recomputes EMA from scratch using
    /// the decay factor applied per-tree (newest tree has weight `1-decay`,
    /// tree k trees back has weight `(1-decay) * decay^k`).
    pub(super) fn refresh_feature_usage_ema(&mut self, n_features: usize) {
        if self.diversity_penalty <= 0.0 {
            return;
        }
        if self.feature_usage_ema.len() != n_features {
            self.feature_usage_ema = vec![0.0; n_features];
        }
        let decay = self.diversity_decay;
        let n_trees = self.trees.len();
        // Reset
        for v in self.feature_usage_ema.iter_mut() {
            *v = 0.0;
        }
        if n_trees == 0 {
            return;
        }
        // Iterate oldest→newest, applying decay each step
        let mut power = (1.0 - decay) * decay.powi((n_trees - 1) as i32);
        let mut seen = vec![false; n_features];
        for tree in self.trees.iter() {
            for v in seen.iter_mut() {
                *v = false;
            }
            for &f in tree.split_features.iter() {
                if f == u32::MAX {
                    continue;
                }
                let fi = f as usize;
                if fi < n_features && !seen[fi] {
                    seen[fi] = true;
                    self.feature_usage_ema[fi] += power;
                }
            }
            // Age forward to newer tree: divide decay out
            if decay > 1e-9 {
                power /= decay;
            }
        }
    }

    pub(super) fn make_feature_mask_for_subtree(
        &self,
        rng: &mut StdRng,
        n_features: usize,
        round: usize,
        sub_idx: usize,
    ) -> Vec<bool> {
        if self.feature_view_groups.len() != n_features {
            return self.make_feature_mask(rng, n_features);
        }
        let max_gid = match self.feature_view_groups.iter().max() {
            Some(v) => *v as usize,
            None => return self.make_feature_mask(rng, n_features),
        };
        if max_gid == 0 {
            return self.make_feature_mask(rng, n_features);
        }
        let raw_features: Vec<usize> = self
            .feature_view_groups
            .iter()
            .enumerate()
            .filter_map(|(idx, &gid)| (gid == 0).then_some(idx))
            .collect();
        let aux_group_id = 1 + (((round * self.n_trees_per_round.max(1)) + sub_idx) % max_gid);
        let aux_features: Vec<usize> = self
            .feature_view_groups
            .iter()
            .enumerate()
            .filter_map(|(idx, &gid)| (gid as usize == aux_group_id).then_some(idx))
            .collect();
        if raw_features.is_empty() || aux_features.is_empty() {
            return self.make_feature_mask(rng, n_features);
        }
        let mut candidate = raw_features.clone();
        candidate.extend(aux_features.iter().copied());
        if self.colsample_bytree >= 1.0 {
            let mut mask = vec![false; n_features];
            for &idx in &candidate {
                mask[idx] = true;
            }
            return mask;
        }
        let n_select = ((self.colsample_bytree * candidate.len() as f64) as usize)
            .max(1)
            .min(candidate.len());
        let mut shuffled = candidate;
        shuffled.shuffle(rng);
        let mut mask = vec![false; n_features];
        let mut picked_raw = false;
        let mut picked_aux = false;
        for &idx in &shuffled[..n_select] {
            mask[idx] = true;
            let gid = self.feature_view_groups[idx];
            picked_raw |= gid == 0;
            picked_aux |= gid as usize == aux_group_id;
        }
        if !picked_raw {
            let idx = raw_features[rng.random_range(0..raw_features.len())];
            mask[idx] = true;
        }
        if !picked_aux {
            let idx = aux_features[rng.random_range(0..aux_features.len())];
            mask[idx] = true;
        }
        mask
    }

    /// Leaf splitting pass: for each tree, remove its contribution, compute gradients,
    /// and try to split leaves that still have high residual variance.
    /// Uses ALL data for split decisions (not complement), because complement data is too
    /// small (~9 samples/leaf) for finding good splits. Honest estimation during tree
    /// building prevents the critical sequential leakage; post-training operations are
    /// single-pass and benefit from maximum data.
    ///

    /// Optimized paths:
    /// - MSE (no Huber): analytic leave-one-out gradients, no prediction modification needed
    /// - General (Huber/binary): cached leaf assignments for O(1) remove/add
    /// Single-pass leaf splitting with multi-level depth increase.
    /// Instead of n_leaf_splits separate passes (each routing all rows), does one pass
    /// where each tree's leaves can be split up to `max_depth_add` levels deep.
    pub(super) fn leaf_split_pass(&mut self, binned: &BinnedData, y: &[f64], n_rows: usize) {
        let lr = self.multiclass_tree_lr();
        let n_trees = self.trees.len();
        let is_mse = self.task == "regression" && self.huber_delta <= 0.0;
        let par = n_rows >= 5000; // parallelize row loops for large datasets

        // Compute initial predictions (parallel over rows for large datasets)
        let mut predictions = if par {
            let base = self.base_score;
            let trees = &self.trees;
            (0..n_rows)
                .into_par_iter()
                .map(|i| {
                    let mut sum = base;
                    for tree in trees.iter() {
                        sum += lr * tree.predict_binned(binned, i);
                    }
                    sum
                })
                .collect()
        } else {
            let mut preds = vec![self.base_score; n_rows];
            for t in 0..n_trees {
                for i in 0..n_rows {
                    preds[i] += lr * self.trees[t].predict_binned(binned, i);
                }
            }
            preds
        };

        let mut gradients = vec![0.0f64; n_rows];
        let mut hessians = vec![0.0f64; n_rows];
        let mut leaf_assigns = vec![0usize; n_rows];
        let mut tree_preds = vec![0.0f64; n_rows];

        // Pre-allocate leaf_samples buffer (reused across trees)
        let max_nodes = self
            .trees
            .iter()
            .map(|t| t.split_features.len())
            .max()
            .unwrap_or(0);
        let mut leaf_samples: Vec<Vec<u32>> = (0..max_nodes)
            .map(|_| Vec::with_capacity(n_rows / max_nodes.max(1) * 2))
            .collect();

        if is_mse {
            // ── Fast MSE path: analytic leave-one-out gradients ──
            let mut full_grads = vec![0.0f64; n_rows];
            for i in 0..n_rows {
                full_grads[i] = predictions[i] - y[i];
            }

            for t in 0..n_trees {
                // Route rows and compute per-tree predictions from leaf values
                if par {
                    leaf_assigns
                        .par_iter_mut()
                        .zip(tree_preds.par_iter_mut())
                        .enumerate()
                        .for_each(|(i, (la, tp))| {
                            *la = self.trees[t].route_to_leaf(binned, i);
                            *tp = self.trees[t].values[*la];
                        });
                } else {
                    for i in 0..n_rows {
                        leaf_assigns[i] = self.trees[t].route_to_leaf(binned, i);
                        tree_preds[i] = self.trees[t].values[leaf_assigns[i]];
                    }
                }

                // Analytic gradient without tree t
                for i in 0..n_rows {
                    gradients[i] = full_grads[i] - lr * tree_preds[i];
                    hessians[i] = 1.0;
                }

                // Build leaf samples from cached assignments (sequential — appends by leaf)
                let n_nodes = self.trees[t].split_features.len();
                while leaf_samples.len() < n_nodes {
                    leaf_samples.push(Vec::new());
                }
                for v in leaf_samples[..n_nodes].iter_mut() {
                    v.clear();
                }
                for i in 0..n_rows {
                    leaf_samples[leaf_assigns[i]].push(i as u32);
                }

                let n_splits = self.trees[t].try_split_leaves_precomputed(
                    binned,
                    &gradients,
                    &hessians,
                    &leaf_samples[..n_nodes],
                    self.lambda_reg,
                    0.0,
                    self.min_child_weight,
                    self.cat_smooth,
                );

                if n_splits > 0 {
                    // Tree changed: re-route and update full_grads incrementally
                    if par {
                        // Parallel re-route: collect (new_leaf, new_pred, old_pred) per row
                        let updates: Vec<(usize, f64)> = (0..n_rows)
                            .into_par_iter()
                            .map(|i| {
                                let new_leaf = self.trees[t].route_to_leaf(binned, i);
                                let new_pred = self.trees[t].values[new_leaf];
                                (new_leaf, new_pred)
                            })
                            .collect();
                        for (i, (_, new_pred)) in updates.iter().enumerate() {
                            full_grads[i] += lr * (new_pred - tree_preds[i]);
                        }
                    } else {
                        for i in 0..n_rows {
                            let new_leaf = self.trees[t].route_to_leaf(binned, i);
                            let new_pred = self.trees[t].values[new_leaf];
                            full_grads[i] += lr * (new_pred - tree_preds[i]);
                        }
                    }
                }
            }
        } else {
            // ── General path: cached leaf assignments for O(1) remove/add ──
            for t in 0..n_trees {
                // Route rows and cache per-tree predictions
                if par {
                    leaf_assigns
                        .par_iter_mut()
                        .zip(tree_preds.par_iter_mut())
                        .enumerate()
                        .for_each(|(i, (la, tp))| {
                            *la = self.trees[t].route_to_leaf(binned, i);
                            *tp = self.trees[t].values[*la];
                        });
                } else {
                    for i in 0..n_rows {
                        leaf_assigns[i] = self.trees[t].route_to_leaf(binned, i);
                        tree_preds[i] = self.trees[t].values[leaf_assigns[i]];
                    }
                }

                // Remove tree t using cached predictions (no tree traversal)
                for i in 0..n_rows {
                    predictions[i] -= lr * tree_preds[i];
                }

                self.compute_gradients_hessians(y, &predictions, &mut gradients, &mut hessians);

                // Build leaf samples from cached assignments (sequential — appends by leaf)
                let n_nodes = self.trees[t].split_features.len();
                while leaf_samples.len() < n_nodes {
                    leaf_samples.push(Vec::new());
                }
                for v in leaf_samples[..n_nodes].iter_mut() {
                    v.clear();
                }
                for i in 0..n_rows {
                    leaf_samples[leaf_assigns[i]].push(i as u32);
                }

                let n_splits = self.trees[t].try_split_leaves_precomputed(
                    binned,
                    &gradients,
                    &hessians,
                    &leaf_samples[..n_nodes],
                    self.lambda_reg,
                    0.0,
                    self.min_child_weight,
                    self.cat_smooth,
                );

                if n_splits > 0 {
                    // Re-route for updated predictions
                    if par {
                        leaf_assigns.par_iter_mut().enumerate().for_each(|(i, la)| {
                            *la = self.trees[t].route_to_leaf(binned, i);
                        });
                    } else {
                        for i in 0..n_rows {
                            leaf_assigns[i] = self.trees[t].route_to_leaf(binned, i);
                        }
                    }
                }
                // Add back using (updated) cached assignments (no tree traversal)
                for i in 0..n_rows {
                    predictions[i] += lr * self.trees[t].values[leaf_assigns[i]];
                }
            }
        }
    }

    /// Leaf splitting pass for multiclass.
    pub(super) fn leaf_split_pass_multiclass(
        &mut self,
        binned: &BinnedData,
        y: &[f64],
        n_rows: usize,
        n_classes: usize,
    ) {
        let lr = self.multiclass_tree_lr();
        let n_trees = self.trees.len();
        let all_indices: Vec<u32> = (0..n_rows as u32).collect();

        let mut predictions = vec![0.0f64; n_rows * n_classes];
        if self.class_base_scores.len() == n_classes {
            for i in 0..n_rows {
                let base = i * n_classes;
                predictions[base..base + n_classes].copy_from_slice(&self.class_base_scores);
            }
        }
        for t in 0..n_trees {
            let class_k = (t / self.multiclass_trees_per_class_round()) % n_classes;
            for i in 0..n_rows {
                predictions[i * n_classes + class_k] +=
                    lr * self.trees[t].predict_binned(binned, i);
            }
        }

        let mut probs = vec![0.0f64; n_rows * n_classes];
        let mut gradients = vec![0.0f64; n_rows];
        let mut hessians = vec![0.0f64; n_rows];

        for t in 0..n_trees {
            let class_k = (t / self.multiclass_trees_per_class_round()) % n_classes;

            for i in 0..n_rows {
                predictions[i * n_classes + class_k] -=
                    lr * self.trees[t].predict_binned(binned, i);
            }

            Self::compute_softmax(&predictions, &mut probs, n_rows, n_classes);
            for i in 0..n_rows {
                let label = if y[i] as usize == class_k { 1.0 } else { 0.0 };
                gradients[i] = probs[i * n_classes + class_k] - label;
                hessians[i] = (probs[i * n_classes + class_k]
                    * (1.0 - probs[i * n_classes + class_k]))
                    .max(1e-16);
            }

            self.trees[t].try_split_leaves(
                binned,
                &gradients,
                &hessians,
                &all_indices,
                self.lambda_reg,
                0.0,
                self.min_child_weight,
                self.cat_smooth,
            );

            for i in 0..n_rows {
                predictions[i * n_classes + class_k] +=
                    lr * self.trees[t].predict_binned(binned, i);
            }
        }
    }

    /// Route all rows through one tree, returning leaf node index per row as u16.
    pub(super) fn route_tree_leaves(
        tree: &DecisionTree,
        binned: &BinnedData,
        n_rows: usize,
    ) -> Vec<u16> {
        (0..n_rows)
            .map(|i| tree.route_to_leaf(binned, i) as u16)
            .collect()
    }

    /// From pre-computed leaf assignments, build (leaf_nodes, leaf_samples) for one tree.
    pub(super) fn build_leaf_info(
        tree: &DecisionTree,
        leaf_assign: &[u16],
        n_rows: usize,
    ) -> (Vec<usize>, Vec<Vec<u32>>) {
        let n_nodes = tree.split_features.len();
        let mut leaf_nodes = Vec::new();
        let mut node_to_local: Vec<usize> = vec![0; n_nodes];
        for i in 0..n_nodes {
            if tree.split_features[i] == u32::MAX {
                node_to_local[i] = leaf_nodes.len();
                leaf_nodes.push(i);
            }
        }
        let mut leaf_samples: Vec<Vec<u32>> = vec![Vec::new(); leaf_nodes.len()];
        for i in 0..n_rows {
            let node = leaf_assign[i] as usize;
            leaf_samples[node_to_local[node]].push(i as u32);
        }
        (leaf_nodes, leaf_samples)
    }

    /// Flat version of `build_leaf_info` for hot refinement loops.
    /// Returns leaf nodes, prefix offsets, and row ids in stable row order.
    pub(super) fn build_leaf_info_flat(
        tree: &DecisionTree,
        leaf_assign: &[u16],
        n_rows: usize,
    ) -> (Vec<usize>, Vec<usize>, Vec<u32>) {
        let n_nodes = tree.split_features.len();
        let mut leaf_nodes = Vec::new();
        let mut node_to_local: Vec<usize> = vec![usize::MAX; n_nodes];
        for i in 0..n_nodes {
            if tree.split_features[i] == u32::MAX {
                node_to_local[i] = leaf_nodes.len();
                leaf_nodes.push(i);
            }
        }

        let mut counts = vec![0usize; leaf_nodes.len()];
        for &node in leaf_assign.iter().take(n_rows) {
            let local = node_to_local[node as usize];
            if local != usize::MAX {
                counts[local] += 1;
            }
        }

        let mut offsets = vec![0usize; leaf_nodes.len() + 1];
        for i in 0..counts.len() {
            offsets[i + 1] = offsets[i] + counts[i];
        }

        let mut cursor = offsets[..leaf_nodes.len()].to_vec();
        let mut samples = vec![0u32; n_rows];
        for i in 0..n_rows {
            let local = node_to_local[leaf_assign[i] as usize];
            if local == usize::MAX {
                continue;
            }
            let pos = cursor[local];
            samples[pos] = i as u32;
            cursor[local] += 1;
        }

        (leaf_nodes, offsets, samples)
    }

    pub(super) fn should_use_expert_leaf_admission(&self, binned: &BinnedData) -> bool {
        if !self.expert_leaf_admission
            || !(self.task == "regression" || self.task == "binary")
            || self.prob_avg
            || self.honest
            || self.leaf_linear
            || self.leaf_quadratic
            || self.ramp
            || self.leaf_correction > 0
            || self.cat_lookup_smooth > 0.0
            || self.n_trees_per_round != 1
        {
            return false;
        }
        let n_numeric = binned
            .is_categorical
            .iter()
            .filter(|&&is_cat| !is_cat)
            .count();
        n_numeric > 0
    }

    pub(super) fn vceg_partition_indices(
        &self,
        indices: &[u32],
        round: usize,
        sub_idx: usize,
    ) -> Option<(Vec<u32>, Vec<u32>)> {
        let mut build = Vec::with_capacity(indices.len());
        let mut cal = Vec::with_capacity(indices.len() / 4 + 1);
        let salt = self
            .seed
            .wrapping_add((round as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .wrapping_add((sub_idx as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
        for &idx in indices {
            let mut x = (idx as u64).wrapping_add(salt);
            x ^= x >> 30;
            x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x ^= x >> 27;
            x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^= x >> 31;
            if x & 3 == 0 {
                cal.push(idx);
            } else {
                build.push(idx);
            }
        }
        if build.len() >= self.expert_min_leaf && cal.len() >= self.expert_min_cal {
            Some((build, cal))
        } else {
            None
        }
    }

    pub(super) fn vceg_feature_stats(
        &self,
        binned: &BinnedData,
        rows: &[u32],
        feat: usize,
    ) -> Option<(f64, f64)> {
        let offset = feat * binned.n_rows;
        let mut sum = 0.0;
        let mut cnt = 0usize;
        for &idx in rows {
            let bin = binned.bin_indices[offset + idx as usize];
            if bin != MISSING_BIN {
                sum += bin as f64;
                cnt += 1;
            }
        }
        if cnt == 0 {
            return None;
        }
        let mean = sum / cnt as f64;
        let mut ss = 0.0;
        for &idx in rows {
            let bin = binned.bin_indices[offset + idx as usize];
            if bin != MISSING_BIN {
                let d = bin as f64 - mean;
                ss += d * d;
            }
        }
        let scale = (ss / cnt as f64).sqrt();
        if scale.is_finite() && scale > 1e-9 {
            Some((mean, scale))
        } else {
            None
        }
    }

    pub(super) fn vceg_alpha(&self, g: &[f64], h: &[f64], u: &[f64]) -> f64 {
        let mut gu = 0.0;
        let mut huu = 0.0;
        for i in 0..u.len() {
            gu += g[i] * u[i];
            huu += h[i] * u[i] * u[i];
        }
        let denom = huu + 1e-6;
        if denom <= 1e-12 {
            return 0.0;
        }
        (-gu / denom).clamp(0.0, self.expert_alpha_max)
    }

    pub(super) fn vceg_loss(&self, y: f64, pred: f64) -> f64 {
        if self.task == "binary" {
            let z = pred.clamp(-50.0, 50.0);
            if y > 0.5 {
                (1.0 + (-z).exp()).ln()
            } else {
                (1.0 + z.exp()).ln()
            }
        } else {
            let r = pred - y;
            0.5 * r * r
        }
    }

    pub(super) fn vceg_shadow_shift(
        &self,
        len: usize,
        node: usize,
        trial: usize,
        salt: u64,
    ) -> usize {
        if len <= 1 {
            return 0;
        }
        let mut x = self
            .seed
            .wrapping_add(salt)
            .wrapping_add((node as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .wrapping_add((trial as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
        1 + (x as usize % (len - 1))
    }

    pub(super) fn vceg_robust_sum(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        if values.len() < 20 {
            return values.iter().sum();
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let trim = (values.len() / 10).max(1);
        let lo = trim.min(sorted.len() - 1);
        let hi = sorted.len().saturating_sub(trim).max(lo + 1);
        let kept = &sorted[lo..hi];
        let mean = kept.iter().sum::<f64>() / kept.len() as f64;
        mean * values.len() as f64
    }

    pub(super) fn apply_expert_leaf_admission(
        &self,
        tree: &mut DecisionTree,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        predictions: &[f64],
        y: &[f64],
        build_indices: &[u32],
        cal_indices: &[u32],
    ) {
        if !self.should_use_expert_leaf_admission(binned)
            || build_indices.is_empty()
            || cal_indices.is_empty()
        {
            return;
        }
        let k = self.expert_max_terms.clamp(1, 4);
        let n_nodes = tree.split_features.len();
        let numeric_features: Vec<usize> = (0..binned.n_features)
            .filter(|&f| !binned.is_categorical.get(f).copied().unwrap_or(false))
            .collect();
        if numeric_features.is_empty() {
            return;
        }

        let mut build_samples = vec![Vec::<u32>::new(); n_nodes];
        let mut cal_samples = vec![Vec::<u32>::new(); n_nodes];
        let path_features = tree.compute_path_features_k(k.max(1));
        for &idx in build_indices {
            let leaf = tree.route_to_leaf(binned, idx as usize);
            build_samples[leaf].push(idx);
        }
        for &idx in cal_indices {
            let leaf = tree.route_to_leaf(binned, idx as usize);
            cal_samples[leaf].push(idx);
        }

        let mut new_values = tree.values.clone();
        let mut ramp_features = vec![u32::MAX; n_nodes * k];
        let mut ramp_slopes = vec![0.0; n_nodes * k];
        let mut admitted = 0usize;

        for node in 0..n_nodes {
            if tree.split_features[node] != u32::MAX {
                continue;
            }
            let build = &build_samples[node];
            let cal = &cal_samples[node];
            if build.len() < self.expert_min_leaf || cal.len() < self.expert_min_cal {
                continue;
            }

            let mut path_candidate_features: Vec<usize> = Vec::new();
            let base = node * k;
            for j in 0..k {
                let feat = path_features.get(base + j).copied().unwrap_or(u32::MAX);
                if feat == u32::MAX {
                    continue;
                }
                let feat = feat as usize;
                if feat < binned.n_features
                    && !binned.is_categorical.get(feat).copied().unwrap_or(false)
                    && !path_candidate_features.contains(&feat)
                {
                    path_candidate_features.push(feat);
                }
            }

            let mut scored: Vec<(f64, usize, f64, f64)> = Vec::new();
            for &feat in &numeric_features {
                let Some((mean, scale)) = self.vceg_feature_stats(binned, build, feat) else {
                    continue;
                };
                let offset = feat * binned.n_rows;
                let mut num = 0.0;
                let mut den = 0.0;
                for &idx in build {
                    let i = idx as usize;
                    let bin = binned.bin_indices[offset + i];
                    let z = if bin == MISSING_BIN {
                        0.0
                    } else {
                        (bin as f64 - mean) / scale
                    };
                    num += z * gradients[i];
                    den += hessians[i] * z * z;
                }
                if den > 1e-12 {
                    scored.push((num.abs() / den.sqrt(), feat, mean, scale));
                }
            }
            if scored.is_empty() {
                continue;
            }
            scored.sort_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.1.cmp(&b.1))
            });
            scored.truncate(k);
            let p = scored.len();
            let dim = p + 1;
            let mut a = vec![0.0; dim * dim];
            let mut rhs = vec![0.0; dim];
            let mut x = vec![0.0; dim];
            x[0] = 1.0;
            for &idx in build {
                let i = idx as usize;
                let h = hessians[i].max(1e-6);
                let target = -gradients[i] / h;
                for (j, &(_, feat, mean, scale)) in scored.iter().enumerate() {
                    let bin = binned.bin_indices[feat * binned.n_rows + i];
                    x[j + 1] = if bin == MISSING_BIN {
                        0.0
                    } else {
                        (bin as f64 - mean) / scale
                    };
                }
                for r in 0..dim {
                    rhs[r] += h * x[r] * target;
                    for c in r..dim {
                        a[r * dim + c] += h * x[r] * x[c];
                    }
                }
            }
            for r in 0..dim {
                for c in (r + 1)..dim {
                    a[c * dim + r] = a[r * dim + c];
                }
            }
            for j in 1..dim {
                a[j * dim + j] += self.expert_ridge_lambda;
            }
            a[0] += 1e-8;
            let theta = solve_spd(dim, &a, &rhs);
            if theta.iter().any(|v| !v.is_finite()) {
                continue;
            }

            let mut intercept = theta[0];
            let mut slopes = vec![0.0; p];
            for (j, &(_, _feat, mean, scale)) in scored.iter().enumerate() {
                intercept -= theta[j + 1] * mean / scale;
                slopes[j] = theta[j + 1] / scale;
            }

            let mut g_build = Vec::with_capacity(build.len());
            let mut h_build = Vec::with_capacity(build.len());
            let mut u_scalar_build = Vec::with_capacity(build.len());
            let mut u_expert_build = Vec::with_capacity(build.len());
            for &idx in build {
                let i = idx as usize;
                let mut raw = intercept;
                for (j, &(_, feat, _mean, _scale)) in scored.iter().enumerate() {
                    let bin = binned.bin_indices[feat * binned.n_rows + i];
                    if bin != MISSING_BIN {
                        raw += slopes[j] * bin as f64;
                    }
                }
                g_build.push(gradients[i]);
                h_build.push(hessians[i]);
                u_scalar_build.push(self.learning_rate * tree.values[node]);
                u_expert_build.push(self.learning_rate * raw);
            }
            let alpha_scalar = self.vceg_alpha(&g_build, &h_build, &u_scalar_build);
            let alpha_expert = self.vceg_alpha(&g_build, &h_build, &u_expert_build);
            if alpha_expert <= 1e-9 {
                continue;
            }

            let mut diff_sum = 0.0;
            let mut diffs = Vec::with_capacity(cal.len());
            for &idx in cal {
                let i = idx as usize;
                let mut raw = intercept;
                for (j, &(_, feat, _mean, _scale)) in scored.iter().enumerate() {
                    let bin = binned.bin_indices[feat * binned.n_rows + i];
                    if bin != MISSING_BIN {
                        raw += slopes[j] * bin as f64;
                    }
                }
                let base_loss = self.vceg_loss(y[i], predictions[i]);
                let scalar_loss = self.vceg_loss(
                    y[i],
                    predictions[i] + alpha_scalar * self.learning_rate * tree.values[node],
                );
                let expert_loss = self.vceg_loss(
                    y[i],
                    predictions[i] + alpha_expert * self.learning_rate * raw,
                );
                let d = (base_loss - expert_loss) - (base_loss - scalar_loss);
                diff_sum += d;
                diffs.push(d);
            }
            let mean_diff = diff_sum / diffs.len() as f64;
            let mut var = 0.0;
            if diffs.len() > 1 {
                for &d in &diffs {
                    let d = d - mean_diff;
                    var += d * d;
                }
                var /= (diffs.len() - 1) as f64;
            }
            let sd = var.sqrt();
            let selected_are_path_local = !path_candidate_features.is_empty()
                && scored
                    .iter()
                    .take(p)
                    .all(|&(_, feat, _, _)| path_candidate_features.contains(&feat));
            let search_width = if selected_are_path_local {
                path_candidate_features.len()
            } else {
                numeric_features.len()
            };
            let p_eff = p as f64 * (1.0 + (search_width as f64 + 1.0).ln());
            let penalty = self.expert_se_multiplier * (diffs.len() as f64).sqrt() * sd
                + self.expert_param_penalty * p_eff
                + self.expert_epsilon;
            let robust_diff_sum = self.vceg_robust_sum(&diffs);

            let mut shadow_threshold = 0.0f64;
            let shadow_trials = self.expert_shadow_trials.min(8);
            if shadow_trials > 0 && build.len() > 1 && cal.len() > 1 {
                for trial in 0..shadow_trials {
                    let build_shift =
                        self.vceg_shadow_shift(build.len(), node, trial, 0xD1B5_4A32_D192_ED03);
                    let cal_shift =
                        self.vceg_shadow_shift(cal.len(), node, trial, 0xA24B_AED4_963E_E407);

                    let mut a_s = vec![0.0; dim * dim];
                    let mut rhs_s = vec![0.0; dim];
                    let mut x_s = vec![0.0; dim];
                    x_s[0] = 1.0;
                    for (pos, &idx) in build.iter().enumerate() {
                        let i = idx as usize;
                        let src_i = build[(pos + build_shift) % build.len()] as usize;
                        let h = hessians[i].max(1e-6);
                        let target = -gradients[i] / h;
                        for (j, &(_, feat, mean, scale)) in scored.iter().enumerate() {
                            let bin = binned.bin_indices[feat * binned.n_rows + src_i];
                            x_s[j + 1] = if bin == MISSING_BIN {
                                0.0
                            } else {
                                (bin as f64 - mean) / scale
                            };
                        }
                        for r in 0..dim {
                            rhs_s[r] += h * x_s[r] * target;
                            for c in r..dim {
                                a_s[r * dim + c] += h * x_s[r] * x_s[c];
                            }
                        }
                    }
                    for r in 0..dim {
                        for c in (r + 1)..dim {
                            a_s[c * dim + r] = a_s[r * dim + c];
                        }
                    }
                    for j in 1..dim {
                        a_s[j * dim + j] += self.expert_ridge_lambda;
                    }
                    a_s[0] += 1e-8;
                    let theta_s = solve_spd(dim, &a_s, &rhs_s);
                    if theta_s.iter().any(|v| !v.is_finite()) {
                        continue;
                    }

                    let mut g_shadow_build = Vec::with_capacity(build.len());
                    let mut h_shadow_build = Vec::with_capacity(build.len());
                    let mut u_shadow_build = Vec::with_capacity(build.len());
                    for (pos, &idx) in build.iter().enumerate() {
                        let i = idx as usize;
                        let src_i = build[(pos + build_shift) % build.len()] as usize;
                        let mut raw = theta_s[0];
                        for (j, &(_, feat, mean, scale)) in scored.iter().enumerate() {
                            let bin = binned.bin_indices[feat * binned.n_rows + src_i];
                            let z = if bin == MISSING_BIN {
                                0.0
                            } else {
                                (bin as f64 - mean) / scale
                            };
                            raw += theta_s[j + 1] * z;
                        }
                        g_shadow_build.push(gradients[i]);
                        h_shadow_build.push(hessians[i]);
                        u_shadow_build.push(self.learning_rate * raw);
                    }
                    let alpha_shadow =
                        self.vceg_alpha(&g_shadow_build, &h_shadow_build, &u_shadow_build);
                    if alpha_shadow <= 1e-9 {
                        continue;
                    }

                    let mut shadow_diffs = Vec::with_capacity(cal.len());
                    for (pos, &idx) in cal.iter().enumerate() {
                        let i = idx as usize;
                        let src_i = cal[(pos + cal_shift) % cal.len()] as usize;
                        let mut raw = theta_s[0];
                        for (j, &(_, feat, mean, scale)) in scored.iter().enumerate() {
                            let bin = binned.bin_indices[feat * binned.n_rows + src_i];
                            let z = if bin == MISSING_BIN {
                                0.0
                            } else {
                                (bin as f64 - mean) / scale
                            };
                            raw += theta_s[j + 1] * z;
                        }
                        let base_loss = self.vceg_loss(y[i], predictions[i]);
                        let scalar_loss = self.vceg_loss(
                            y[i],
                            predictions[i] + alpha_scalar * self.learning_rate * tree.values[node],
                        );
                        let shadow_loss = self.vceg_loss(
                            y[i],
                            predictions[i] + alpha_shadow * self.learning_rate * raw,
                        );
                        shadow_diffs.push((base_loss - shadow_loss) - (base_loss - scalar_loss));
                    }
                    shadow_threshold = shadow_threshold.max(self.vceg_robust_sum(&shadow_diffs));
                }
            }

            let win_rate = diffs.iter().filter(|&&d| d > 0.0).count() as f64 / diffs.len() as f64;
            if win_rate >= 0.55 && robust_diff_sum > shadow_threshold + penalty {
                new_values[node] = alpha_expert * intercept;
                let base = node * k;
                for (j, &(_, feat, _mean, _scale)) in scored.iter().enumerate() {
                    ramp_features[base + j] = feat as u32;
                    ramp_slopes[base + j] = alpha_expert * slopes[j];
                }
                admitted += 1;
            }
        }

        if admitted > 0 {
            tree.values = new_values;
            tree.ramp_k = k;
            tree.ramp_features = ramp_features;
            tree.ramp_slopes = ramp_slopes;
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // CFE: Categorical Fold Evidence — native fast tuple-posterior features.
    //
    // For selected categorical tuples (singles + utility-screened pairs and
    // triples), build target-statistic tables and emit per-row evidence
    // columns. Train rows get CROSS-FIT values (table minus the row's fold):
    // leak-safe like CatBoost's ordered TS but deterministic — no permutation
    // noise and no cold-start prefix rows. Eval/predict rows use the full
    // table. A PACT-style naive-Bayes aggregate compresses all tuples into a
    // handful of combined-evidence columns trees cannot reconstruct from the
    // individual lifts.
    // ════════════════════════════════════════════════════════════════════

    #[inline]
    fn cfe_mix(h: u64, v: u64) -> u64 {
        let mut z = h ^ v.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn cfe_keys(x: &[f64], n_rows: usize, n_features: usize, feats: &[usize]) -> Vec<i64> {
        (0..n_rows)
            .map(|row| {
                let mut h: u64 = 0x51_7C_C1_B7;
                for &f in feats {
                    let v = x[row * n_features + f];
                    if !v.is_finite() {
                        return i64::MIN;
                    }
                    h = Self::cfe_mix(h, (v.round() as i64) as u64);
                }
                (h >> 1) as i64
            })
            .collect()
    }

    #[inline]
    fn cfe_logit(p: f64) -> f64 {
        let p = p.clamp(1e-6, 1.0 - 1e-6);
        (p / (1.0 - p)).ln()
    }

    fn cfe_arity_clip(arity: usize) -> f64 {
        // EA-evolved schedule (cfe_equation_ea_lab, LODO-selected on kdd):
        // clip_base 3.605 * 0.7486^(arity-1).
        match arity {
            1 => 3.605,
            2 => 2.699,
            _ => 2.020,
        }
    }

    /// OOF predictive utility: build the candidate's CROSS-FIT lift column
    /// and score its honest squared correlation with the target. Memorizing
    /// junk keys self-eliminates (their out-of-fold lift is noise), so no
    /// key-count penalty is needed — this replaces the in-sample chi2 +
    /// ln(keys) band-aid and makes WIDE candidate pools safe.
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    fn cfe_tuple_score_oof(
        keys: &[i64],
        targets: &[f64],
        n_out: usize,
        prior: &[f64],
        m: f64,
        fold_of: &[u8],
        n_folds: usize,
    ) -> f64 {
        let n_rows = keys.len();
        let mut idx: HashMap<i64, usize> = HashMap::new();
        let mut cnt: Vec<f64> = Vec::new();
        let mut sums: Vec<f64> = Vec::new();
        let mut fcnt: Vec<f64> = Vec::new();
        let mut fsums: Vec<f64> = Vec::new();
        for (row, &key) in keys.iter().enumerate() {
            if key == i64::MIN {
                continue;
            }
            let ki = *idx.entry(key).or_insert_with(|| {
                cnt.push(0.0);
                sums.extend(std::iter::repeat(0.0).take(n_out));
                fcnt.extend(std::iter::repeat(0.0).take(n_folds));
                fsums.extend(std::iter::repeat(0.0).take(n_folds * n_out));
                cnt.len() - 1
            });
            let f = fold_of[row] as usize;
            cnt[ki] += 1.0;
            fcnt[ki * n_folds + f] += 1.0;
            for o in 0..n_out {
                let t = targets[row * n_out + o];
                sums[ki * n_out + o] += t;
                fsums[(ki * n_folds + f) * n_out + o] += t;
            }
        }
        if idx.len() < 2 {
            return 0.0;
        }
        // Honest squared correlation summed over outputs.
        let mut score = 0.0f64;
        for o in 0..n_out {
            let mut sxy = 0.0f64;
            let mut sxx = 0.0f64;
            let mut syy = 0.0f64;
            for (row, &key) in keys.iter().enumerate() {
                if key == i64::MIN {
                    continue;
                }
                let ki = idx[&key];
                let f = fold_of[row] as usize;
                let c = cnt[ki] - fcnt[ki * n_folds + f];
                let s_ = sums[ki * n_out + o] - fsums[(ki * n_folds + f) * n_out + o];
                let lift = (s_ + m * prior[o]) / (c + m) - prior[o];
                let yv = targets[row * n_out + o] - prior[o];
                sxy += lift * yv;
                sxx += lift * lift;
                syy += yv * yv;
            }
            if sxx > 1e-12 && syy > 1e-12 {
                score += (sxy * sxy) / (sxx * syy);
            }
        }
        score
    }

    /// Chi2-like utility of a keyed tuple against per-output targets.
    fn cfe_tuple_score(
        keys: &[i64],
        targets: &[f64],
        n_out: usize,
        prior: &[f64],
        m: f64,
    ) -> f64 {
        let mut idx: HashMap<i64, usize> = HashMap::new();
        let mut cnt: Vec<f64> = Vec::new();
        let mut sums: Vec<f64> = Vec::new();
        for (row, &key) in keys.iter().enumerate() {
            if key == i64::MIN {
                continue;
            }
            let ki = *idx.entry(key).or_insert_with(|| {
                cnt.push(0.0);
                sums.extend(std::iter::repeat(0.0).take(n_out));
                cnt.len() - 1
            });
            cnt[ki] += 1.0;
            for o in 0..n_out {
                sums[ki * n_out + o] += targets[row * n_out + o];
            }
        }
        if idx.len() < 2 {
            return 0.0;
        }
        let mut score = 0.0f64;
        for ki in 0..cnt.len() {
            let c = cnt[ki];
            for o in 0..n_out {
                let d = sums[ki * n_out + o] - c * prior[o];
                score += d * d / (c + m);
            }
        }
        // Penalize raw key count (search width / memorization pressure).
        // EA-evolved: weaker key-count penalty keeps high-card crosses (pi=0.539).
        score / (1.0 + (idx.len() as f64).ln()).powf(0.539)
    }

    /// Fit CFE tables and return TRAIN evidence columns (cross-fit values).
    pub(super) fn build_cat_fold_evidence(
        &mut self,
        x_data: &[f64],
        y: &[f64],
        n_rows: usize,
        n_features: usize,
    ) -> Vec<Vec<f64>> {
        self.cfe_tuples.clear();
        self.cfe_tables.clear();
        self.cfe_prior.clear();
        self.cfe_n_out = 0;
        if !self.cat_fold_evidence || n_rows < 40 {
            return Vec::new();
        }
        let cat_cols: Vec<usize> = (0..n_features)
            .filter(|&f| f < self.cat_features.len() && self.cat_features[f])
            .collect();
        if cat_cols.is_empty() {
            return Vec::new();
        }
        // Targets: binary/regression -> 1 output; multiclass -> one-hot K.
        let (n_out, targets): (usize, Vec<f64>) = if self.task == "multiclass" {
            let k = y
                .iter()
                .filter(|v| v.is_finite() && **v >= 0.0)
                .map(|&v| v.round() as usize)
                .max()
                .unwrap_or(0)
                + 1;
            if k < 2 {
                return Vec::new();
            }
            let mut t = vec![0.0f64; n_rows * k];
            for (row, &yv) in y.iter().enumerate().take(n_rows) {
                let c = (yv.round() as usize).min(k - 1);
                t[row * k + c] = 1.0;
            }
            (k, t)
        } else {
            (1, y[..n_rows].to_vec())
        };
        let mut prior = vec![0.0f64; n_out];
        for row in 0..n_rows {
            for o in 0..n_out {
                prior[o] += targets[row * n_out + o];
            }
        }
        for p in prior.iter_mut() {
            *p /= n_rows as f64;
        }
        let m = self.cfe_smooth.max(1e-6);
        let is_reg = self.task != "binary" && self.task != "multiclass";

        // ── Cross-fit fold assignment (shared by selection + tables) ──
        let n_folds = self.cfe_folds.clamp(2, 16).min(n_rows / 8).max(2);
        let fold_of: Vec<u8> = {
            let mut perm: Vec<usize> = (0..n_rows).collect();
            let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(0xCFE));
            perm.shuffle(&mut rng);
            let mut fo = vec![0u8; n_rows];
            for (rank, &row) in perm.iter().enumerate() {
                fo[row] = (rank % n_folds) as u8;
            }
            fo
        };

        // ── Tuple selection (EA-tuned in-sample screen; an OOF-utility screen
        // was built and A/B-tested 2026-06-10 and LOST on both kdd and Amazon —
        // the chi2 + ln(keys)^0.539 form stays) ──
        let mut singles: Vec<(f64, usize, Vec<i64>)> = cat_cols
            .par_iter()
            .map(|&f| {
                let keys = Self::cfe_keys(x_data, n_rows, n_features, &[f]);
                let s = Self::cfe_tuple_score(&keys, &targets, n_out, &prior, m);
                (s, f, keys)
            })
            .collect();
        singles.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        singles.retain(|(s, _, _)| *s > 0.0);
        // Encode ALL useful singles (CatBoost encodes every cat feature; a low
        // cap was the loss driver on wide-categorical data like kdd_internet).
        singles.truncate(64);
        if singles.is_empty() {
            return Vec::new();
        }
        let mut tuples: Vec<Vec<usize>> = singles.iter().map(|(_, f, _)| vec![*f]).collect();
        let top_feats: Vec<usize> = singles.iter().take(12).map(|(_, f, _)| *f).collect();
        if self.cfe_max_pairs > 0 && top_feats.len() >= 2 {
            let mut cands: Vec<(usize, usize)> = Vec::new();
            for i in 0..top_feats.len() {
                for j in (i + 1)..top_feats.len() {
                    cands.push((top_feats[i], top_feats[j]));
                }
            }
            let mut scored: Vec<(f64, Vec<usize>)> = cands
                .par_iter()
                .map(|&(a, b)| {
                    let keys = Self::cfe_keys(x_data, n_rows, n_features, &[a, b]);
                    (
                        Self::cfe_tuple_score(&keys, &targets, n_out, &prior, m),
                        vec![a, b],
                    )
                })
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            for (s, t) in scored.into_iter().take(self.cfe_max_pairs) {
                if s > 0.0 {
                    tuples.push(t);
                }
            }
        }
        if self.cfe_max_triples > 0 && top_feats.len() >= 3 {
            let base: Vec<usize> = top_feats.iter().take(6).copied().collect();
            let mut cands: Vec<Vec<usize>> = Vec::new();
            for i in 0..base.len() {
                for j in (i + 1)..base.len() {
                    for l in (j + 1)..base.len() {
                        cands.push(vec![base[i], base[j], base[l]]);
                    }
                }
            }
            let mut scored: Vec<(f64, Vec<usize>)> = cands
                .par_iter()
                .map(|t| {
                    let keys = Self::cfe_keys(x_data, n_rows, n_features, t);
                    (
                        Self::cfe_tuple_score(&keys, &targets, n_out, &prior, m),
                        t.clone(),
                    )
                })
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            for (s, t) in scored.into_iter().take(self.cfe_max_triples) {
                if s > 0.0 {
                    tuples.push(t);
                }
            }
        }
        if self.cfe_max_quads > 0 && top_feats.len() >= 4 {
            // Exhaustive arity-4 crosses among the strongest singles: the deep-
            // combo region CatBoost's greedy in-tree CTR growth reaches only
            // along split paths; static screening covers it completely.
            let base: Vec<usize> = top_feats.iter().take(6).copied().collect();
            let mut cands: Vec<Vec<usize>> = Vec::new();
            for i in 0..base.len() {
                for j in (i + 1)..base.len() {
                    for l in (j + 1)..base.len() {
                        for q in (l + 1)..base.len() {
                            cands.push(vec![base[i], base[j], base[l], base[q]]);
                        }
                    }
                }
            }
            let mut scored: Vec<(f64, Vec<usize>)> = cands
                .par_iter()
                .map(|t| {
                    let keys = Self::cfe_keys(x_data, n_rows, n_features, t);
                    (
                        Self::cfe_tuple_score(&keys, &targets, n_out, &prior, m),
                        t.clone(),
                    )
                })
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            for (s, t) in scored.into_iter().take(self.cfe_max_quads) {
                if s > 0.0 {
                    tuples.push(t);
                }
            }
        }


        // ── Per-tuple tables + cross-fit train lifts ──
        struct TupleOut {
            table: HashMap<i64, (f64, Vec<f64>)>,
            lifts: Vec<f64>,  // n_rows x n_out, cross-fit clipped lift (prior m)
            lifts2: Vec<f64>, // n_rows x n_out, lift at strong prior (16m)
            logcs: Vec<f64>,  // n_rows, log1p(complement key count) — Counter CTR
            arity: usize,
            n_out: usize,
        }
        impl CfeTupleView for TupleOut {
            fn lift(&self, row: usize, out: usize) -> f64 {
                self.lifts[row * self.n_out + out]
            }
            fn lift2(&self, row: usize, out: usize) -> f64 {
                self.lifts2[row * self.n_out + out]
            }
            fn logc(&self, row: usize) -> f64 {
                self.logcs[row]
            }
            fn arity(&self) -> usize {
                self.arity
            }
        }
        let outs: Vec<TupleOut> = tuples
            .par_iter()
            .map(|feats| {
                let keys = Self::cfe_keys(x_data, n_rows, n_features, feats);
                let mut idx: HashMap<i64, usize> = HashMap::new();
                let mut cnt: Vec<f64> = Vec::new();
                let mut sums: Vec<f64> = Vec::new();
                let mut fcnt: Vec<f64> = Vec::new();
                let mut fsums: Vec<f64> = Vec::new();
                for (row, &key) in keys.iter().enumerate() {
                    if key == i64::MIN {
                        continue;
                    }
                    let ki = *idx.entry(key).or_insert_with(|| {
                        cnt.push(0.0);
                        sums.extend(std::iter::repeat(0.0).take(n_out));
                        fcnt.extend(std::iter::repeat(0.0).take(n_folds));
                        fsums.extend(std::iter::repeat(0.0).take(n_folds * n_out));
                        cnt.len() - 1
                    });
                    let f = fold_of[row] as usize;
                    cnt[ki] += 1.0;
                    fcnt[ki * n_folds + f] += 1.0;
                    for o in 0..n_out {
                        let t = targets[row * n_out + o];
                        sums[ki * n_out + o] += t;
                        fsums[(ki * n_folds + f) * n_out + o] += t;
                    }
                }
                let clip = Self::cfe_arity_clip(feats.len());
                let m2 = m * 16.0;
                let mut lifts = vec![0.0f64; n_rows * n_out];
                let mut lifts2 = vec![0.0f64; n_rows * n_out];
                for (row, &key) in keys.iter().enumerate() {
                    if key == i64::MIN {
                        continue;
                    }
                    let ki = idx[&key];
                    let f = fold_of[row] as usize;
                    let c = cnt[ki] - fcnt[ki * n_folds + f];
                    // EA-evolved: reliability smoothing decoupled from posterior
                    // smoothing (m_rel = 1.32 * m).
                    let rel = c / (c + 1.32 * m);
                    let rel2 = c / (c + m2);
                    for o in 0..n_out {
                        let s = sums[ki * n_out + o] - fsums[(ki * n_folds + f) * n_out + o];
                        let lift_at = |mm: f64, relv: f64| -> f64 {
                            let p = (s + mm * prior[o]) / (c + mm);
                            let l = if is_reg {
                                p - prior[o]
                            } else {
                                (Self::cfe_logit(p) - Self::cfe_logit(prior[o]))
                                    .clamp(-clip, clip)
                            };
                            relv * l
                        };
                        lifts[row * n_out + o] = lift_at(m, rel);
                        lifts2[row * n_out + o] = lift_at(m2, rel2);
                    }
                }
                let mut logcs = vec![0.0f64; n_rows];
                for (row, &key) in keys.iter().enumerate() {
                    if key == i64::MIN {
                        continue;
                    }
                    let ki = idx[&key];
                    let f = fold_of[row] as usize;
                    logcs[row] = (cnt[ki] - fcnt[ki * n_folds + f]).max(0.0).ln_1p();
                }
                let mut table: HashMap<i64, (f64, Vec<f64>)> = HashMap::with_capacity(idx.len());
                for (key, ki) in idx.iter() {
                    table.insert(
                        *key,
                        (cnt[*ki], sums[*ki * n_out..(*ki + 1) * n_out].to_vec()),
                    );
                }
                TupleOut {
                    table,
                    lifts,
                    lifts2,
                    logcs,
                    arity: feats.len(),
                    n_out,
                }
            })
            .collect();

        let cols = Self::cfe_emit_columns(
            &outs,
            n_rows,
            n_out,
            &prior,
            is_reg,
            self.cfe_dual_prior,
            self.cfe_counter,
            self.cfe_aggmax,
        );
        self.cfe_tuples = tuples;
        self.cfe_tables = outs.into_iter().map(|o| o.table).collect();
        self.cfe_prior = prior;
        self.cfe_n_out = n_out;
        cols
    }

    /// Emit evidence columns from per-tuple lifts: one lift column per tuple
    /// (binary/regression), plus PACT-style aggregates.
    fn cfe_emit_columns(
        outs: &[impl CfeTupleView],
        n_rows: usize,
        n_out: usize,
        prior: &[f64],
        is_reg: bool,
        dual_prior: bool,
        counter_on: bool,
        aggmax_on: bool,
    ) -> Vec<Vec<f64>> {
        let mut cols: Vec<Vec<f64>> = Vec::new();
        if n_out == 1 {
            for o in outs {
                cols.push((0..n_rows).map(|r| o.lift(r, 0)).collect());
                if dual_prior {
                    cols.push((0..n_rows).map(|r| o.lift2(r, 0)).collect());
                }
                if counter_on {
                    // Counter-CTR companion: key frequency (log1p count).
                    cols.push((0..n_rows).map(|r| o.logc(r)).collect());
                }
            }
            // PACT aggregates: NB-total + per-arity NB + per-arity max.
            let prior_logit = if is_reg {
                prior[0]
            } else {
                Self::cfe_logit(prior[0])
            };
            let mut nb_total = vec![0.0f64; n_rows];
            let mut nb_arity = vec![vec![0.0f64; n_rows]; 3];
            let mut max_arity = vec![vec![f64::NEG_INFINITY; n_rows]; 3];
            let mut count_arity = [0.0f64; 3];
            for o in outs {
                let a = o.arity().min(3) - 1;
                count_arity[a] += 1.0;
                for r in 0..n_rows {
                    let v = o.lift(r, 0);
                    nb_total[r] += v;
                    nb_arity[a][r] += v;
                    if v > max_arity[a][r] {
                        max_arity[a][r] = v;
                    }
                }
            }
            let total_n: f64 = count_arity.iter().sum();
            cols.push(
                nb_total
                    .iter()
                    .map(|&s| prior_logit + s / total_n.max(1.0).powf(0.723))
                    .collect(),
            );
            for a in 0..3 {
                if count_arity[a] > 0.0 {
                    cols.push(
                        nb_arity[a]
                            .iter()
                            .map(|&s| prior_logit + s / count_arity[a].max(1.0).powf(0.723))
                            .collect(),
                    );
                    if aggmax_on {
                        cols.push(
                            max_arity[a]
                                .iter()
                                .map(|&v| if v.is_finite() { v } else { 0.0 })
                                .collect(),
                        );
                    }
                }
            }
        } else {
            // Multiclass: per-class NB aggregate columns only (K columns),
            // plus per-class max lift (K columns).
            let total_n = outs.len() as f64;
            for o_cls in 0..n_out {
                let prior_logit = Self::cfe_logit(prior[o_cls]);
                let mut nb = vec![0.0f64; n_rows];
                let mut mx = vec![f64::NEG_INFINITY; n_rows];
                for o in outs {
                    for r in 0..n_rows {
                        let v = o.lift(r, o_cls);
                        nb[r] += v;
                        if v > mx[r] {
                            mx[r] = v;
                        }
                    }
                }
                cols.push(
                    nb.iter()
                        .map(|&s| prior_logit + s / total_n.max(1.0).powf(0.723))
                        .collect(),
                );
                if aggmax_on {
                cols.push(
                    mx.iter()
                        .map(|&v| if v.is_finite() { v } else { 0.0 })
                        .collect(),
                );
                }
            }
        }
        cols
    }

    /// Evidence columns for eval/predict rows using the FULL trained tables.
    pub(super) fn cat_fold_evidence_columns_for_raw(
        &self,
        x_data: &[f64],
        n_rows: usize,
        n_features: usize,
    ) -> Vec<Vec<f64>> {
        if self.cfe_tuples.is_empty() || self.cfe_n_out == 0 || n_rows == 0 {
            return Vec::new();
        }
        let n_out = self.cfe_n_out;
        let m = self.cfe_smooth.max(1e-6);
        let is_reg = self.task != "binary" && self.task != "multiclass";
        struct FullOut {
            lifts: Vec<f64>,
            lifts2: Vec<f64>,
            logcs: Vec<f64>,
            arity: usize,
            n_out: usize,
        }
        impl CfeTupleView for FullOut {
            fn lift(&self, row: usize, out: usize) -> f64 {
                self.lifts[row * self.n_out + out]
            }
            fn lift2(&self, row: usize, out: usize) -> f64 {
                self.lifts2[row * self.n_out + out]
            }
            fn logc(&self, row: usize) -> f64 {
                self.logcs[row]
            }
            fn arity(&self) -> usize {
                self.arity
            }
        }
        let outs: Vec<FullOut> = self
            .cfe_tuples
            .par_iter()
            .zip(self.cfe_tables.par_iter())
            .map(|(feats, table)| {
                let keys = Self::cfe_keys(x_data, n_rows, n_features, feats);
                let clip = Self::cfe_arity_clip(feats.len());
                let m2 = m * 16.0;
                let mut lifts = vec![0.0f64; n_rows * n_out];
                let mut lifts2 = vec![0.0f64; n_rows * n_out];
                let mut logcs = vec![0.0f64; n_rows];
                for (row, &key) in keys.iter().enumerate() {
                    if key == i64::MIN {
                        continue;
                    }
                    if let Some((c, sums)) = table.get(&key) {
                        logcs[row] = c.max(0.0).ln_1p();
                        let rel = c / (c + 1.32 * m);
                        let rel2 = c / (c + m2);
                        for o in 0..n_out {
                            let lift_at = |mm: f64, relv: f64| -> f64 {
                                let p = (sums[o] + mm * self.cfe_prior[o]) / (c + mm);
                                let l = if is_reg {
                                    p - self.cfe_prior[o]
                                } else {
                                    (Self::cfe_logit(p) - Self::cfe_logit(self.cfe_prior[o]))
                                        .clamp(-clip, clip)
                                };
                                relv * l
                            };
                            lifts[row * n_out + o] = lift_at(m, rel);
                            lifts2[row * n_out + o] = lift_at(m2, rel2);
                        }
                    }
                }
                FullOut {
                    lifts,
                    lifts2,
                    logcs,
                    arity: feats.len(),
                    n_out,
                }
            })
            .collect();
        Self::cfe_emit_columns(
            &outs,
            n_rows,
            n_out,
            &self.cfe_prior,
            is_reg,
            self.cfe_dual_prior,
            self.cfe_counter,
            self.cfe_aggmax,
        )
    }
}

/// View trait so train-time and predict-time tuple outputs share the column
/// emission code.
impl GTBoostModel {
    /// CFE stage 2 — RESIDUAL evidence: train a small internal warmup model
    /// (high-card cats masked, mirroring demotion), then build a second
    /// evidence table set over the SAME tuples with the warmup's residual
    /// gradients as targets. Captures the categorical structure the mains
    /// cannot explain — the adaptive-encoding idea of "residual-refreshed
    /// CTRs" with purely STATIC tables (lookup-only at predict, no epochs).
    /// Returns the cross-fit train columns (one lift column per tuple + one
    /// NB aggregate).
    pub(super) fn build_cfe_residual_evidence(
        &mut self,
        binned: &BinnedData,
        x_data: &[f64],
        y: &[f64],
        n_rows: usize,
        n_features: usize,
    ) -> Vec<Vec<f64>> {
        self.cfe_resid_tables.clear();
        self.cfe_resid_prior = 0.0;
        if self.cfe_residual_rounds == 0 || self.cfe_tuples.is_empty() || n_rows < 80 {
            return Vec::new();
        }
        let is_binary = self.task == "binary";
        if !is_binary && self.task != "regression" {
            return Vec::new(); // binary + regression first
        }
        // Warmup feature mask: mirror demotion (no high-card raw cat splits,
        // so memorizable structure can't hide the residual signal).
        let mask: Vec<bool> = (0..binned.n_features)
            .map(|f| {
                if f < self.cat_features.len() && self.cat_features[f] {
                    binned.n_bins(f) <= self.cfe_demote_min_card
                } else {
                    true
                }
            })
            .collect();
        let base = if is_binary {
            let p = (y[..n_rows].iter().sum::<f64>() / n_rows as f64).clamp(1e-6, 1.0 - 1e-6);
            (p / (1.0 - p)).ln()
        } else {
            y[..n_rows].iter().sum::<f64>() / n_rows as f64
        };
        // CROSS-FIT warmup: one mini-model per fold trained on the fold's
        // COMPLEMENT; row i's residual comes from the model that never saw
        // row i's label. (An in-sample warmup leaks label noise into the
        // residual targets and was measured to HURT.)
        let n_folds = self.cfe_folds.clamp(2, 16).min(n_rows / 8).max(2);
        let fold_of: Vec<u8> = {
            let mut perm: Vec<usize> = (0..n_rows).collect();
            let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(0xCFE2));
            perm.shuffle(&mut rng);
            let mut fo = vec![0u8; n_rows];
            for (rank, &row) in perm.iter().enumerate() {
                fo[row] = (rank % n_folds) as u8;
            }
            fo
        };
        let mut honest_margins = vec![base; n_rows];
        for fold in 0..n_folds {
            let train_idx: Vec<u32> = (0..n_rows as u32)
                .filter(|&i| fold_of[i as usize] as usize != fold)
                .collect();
            let mut margins = vec![base; n_rows];
            let mut g = vec![0.0f64; n_rows];
            let mut h = vec![0.0f64; n_rows];
            let lr = 0.3f64;
            for _round in 0..self.cfe_residual_rounds {
                for i in 0..n_rows {
                    if is_binary {
                        let p = 1.0 / (1.0 + (-margins[i].clamp(-30.0, 30.0)).exp());
                        g[i] = p - y[i];
                        h[i] = (p * (1.0 - p)).max(1e-6);
                    } else {
                        g[i] = margins[i] - y[i];
                        h[i] = 1.0;
                    }
                }
                let tree = DecisionTree::build_depthwise(
                binned,
                    &g,
                    &h,
                    &train_idx,
                self.lambda_reg.max(1.0),
                0.0,
                0.0,
                3,
                1.0,
                &mask,
                1.0,
                self.seed ^ (_round as u64).wrapping_mul(0x9E37_79B9),
                0.0,
                0.0,
                0.0,
                &[],
                0.0,
                false,
                0.0,
                false,
                false,
                false,
                None,
                false,
                0.0,
                crate::tree::CatPairConfig {
                    enabled: false,
                    top_k_cat: 0,
                    k_buckets: 0,
                    min_node_rows: 0,
                    max_node_depth: 0,
                    gain_margin: 0.0,
                },

                                None,
                            
                None,
            
                0.0,
            
                0.5,
                1.0,
            );
                tree.add_predictions_binned(binned, &mut margins, lr);
            }
            for i in 0..n_rows {
                if fold_of[i] as usize == fold {
                    honest_margins[i] = margins[i];
                }
            }
        }
        // Residual targets on gradient scale from HONEST margins.
        let mut g = vec![0.0f64; n_rows];
        for i in 0..n_rows {
            if is_binary {
                let p = 1.0 / (1.0 + (-honest_margins[i].clamp(-30.0, 30.0)).exp());
                g[i] = p - y[i];
            } else {
                g[i] = honest_margins[i] - y[i];
            }
        }
        let targets: Vec<f64> = g.iter().map(|&v| -v).collect();
        let prior = targets.iter().sum::<f64>() / n_rows as f64;
        self.cfe_resid_prior = prior;
        let m = self.cfe_smooth.max(1e-6);
        let tuples = self.cfe_tuples.clone();
        let mut cols: Vec<Vec<f64>> = Vec::new();
        let mut agg = vec![0.0f64; n_rows];
        let mut tables: Vec<HashMap<i64, (f64, Vec<f64>)>> = Vec::new();
        for feats in &tuples {
            let keys = Self::cfe_keys(x_data, n_rows, n_features, feats);
            let mut idx: HashMap<i64, usize> = HashMap::new();
            let mut cnt: Vec<f64> = Vec::new();
            let mut sums: Vec<f64> = Vec::new();
            let mut fcnt: Vec<f64> = Vec::new();
            let mut fsums: Vec<f64> = Vec::new();
            for (row, &key) in keys.iter().enumerate() {
                if key == i64::MIN {
                    continue;
                }
                let ki = *idx.entry(key).or_insert_with(|| {
                    cnt.push(0.0);
                    sums.push(0.0);
                    fcnt.extend(std::iter::repeat(0.0).take(n_folds));
                    fsums.extend(std::iter::repeat(0.0).take(n_folds));
                    cnt.len() - 1
                });
                let f = fold_of[row] as usize;
                cnt[ki] += 1.0;
                sums[ki] += targets[row];
                fcnt[ki * n_folds + f] += 1.0;
                fsums[ki * n_folds + f] += targets[row];
            }
            let mut col = vec![0.0f64; n_rows];
            for (row, &key) in keys.iter().enumerate() {
                if key == i64::MIN {
                    continue;
                }
                let ki = idx[&key];
                let f = fold_of[row] as usize;
                let c = cnt[ki] - fcnt[ki * n_folds + f];
                let s_ = sums[ki] - fsums[ki * n_folds + f];
                let rel = c / (c + 1.32 * m);
                let lift = (s_ + m * prior) / (c + m) - prior;
                col[row] = rel * lift;
                agg[row] += col[row];
            }
            cols.push(col);
            let mut table: HashMap<i64, (f64, Vec<f64>)> = HashMap::with_capacity(idx.len());
            for (key, ki) in idx.iter() {
                table.insert(*key, (cnt[*ki], vec![sums[*ki]]));
            }
            tables.push(table);
        }
        let t_pow = (tuples.len().max(1) as f64).powf(0.723);
        cols.push(agg.iter().map(|&s_| prior + s_ / t_pow).collect());
        self.cfe_resid_tables = tables;
        cols
    }

    /// Residual-evidence columns for eval/predict rows from the stored tables.
    pub(super) fn cfe_residual_columns_for_raw(
        &self,
        x_data: &[f64],
        n_rows: usize,
        n_features: usize,
    ) -> Vec<Vec<f64>> {
        if self.cfe_resid_tables.is_empty() || self.cfe_tuples.is_empty() || n_rows == 0 {
            return Vec::new();
        }
        let m = self.cfe_smooth.max(1e-6);
        let prior = self.cfe_resid_prior;
        let mut cols: Vec<Vec<f64>> = Vec::new();
        let mut agg = vec![0.0f64; n_rows];
        for (feats, table) in self.cfe_tuples.iter().zip(self.cfe_resid_tables.iter()) {
            let keys = Self::cfe_keys(x_data, n_rows, n_features, feats);
            let mut col = vec![0.0f64; n_rows];
            for (row, &key) in keys.iter().enumerate() {
                if key == i64::MIN {
                    continue;
                }
                if let Some((c, sums)) = table.get(&key) {
                    let rel = c / (c + 1.32 * m);
                    let lift = (sums[0] + m * prior) / (c + m) - prior;
                    col[row] = rel * lift;
                    agg[row] += col[row];
                }
            }
            cols.push(col);
        }
        let t_pow = (self.cfe_tuples.len().max(1) as f64).powf(0.723);
        cols.push(agg.iter().map(|&s_| prior + s_ / t_pow).collect());
        cols
    }
}

pub(super) trait CfeTupleView: Sync {
    fn lift(&self, row: usize, out: usize) -> f64;
    fn lift2(&self, row: usize, out: usize) -> f64;
    fn logc(&self, row: usize) -> f64;
    fn arity(&self) -> usize;
}
