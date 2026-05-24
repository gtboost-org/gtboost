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
    }

    pub(super) fn apply_corrective_block_refit(
        &mut self,
        binned: &BinnedData,
        x_data_raw: &[f64],
        n_rows: usize,
        n_features_raw: usize,
        y: &[f64],
        init_score: Option<&[f64]>,
    ) {
        if !self.corrective_block_refit
            || self.task != "regression"
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
        for row in 0..n_rows {
            let row_data = &x_data[row * n_features..(row + 1) * n_features];
            let offset = init_score.map(|s| s[row]).unwrap_or(self.base_score);
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
            if refit_sse > base_sse * required {
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
        self.ordered_ctr_maps.clear();
        self.ordered_ctr_count_maps.clear();
        self.ordered_ctr_pair_features.clear();
        self.ordered_ctr_pair_maps.clear();
        self.ordered_ctr_pair_count_maps.clear();
        self.ordered_ctr_triple_features.clear();
        self.ordered_ctr_triple_maps.clear();
        self.ordered_ctr_triple_count_maps.clear();
        self.ordered_ctr_prior = 0.0;

        if !self.ordered_ctr
            || n_rows == 0
            || n_features == 0
            || self.task == "multiclass"
            || self.ordered_ctr_top_features == 0
        {
            return Vec::new();
        }

        let prior = y.iter().sum::<f64>() / n_rows as f64;
        self.ordered_ctr_prior = prior;
        let min_count = self.ordered_ctr_min_count.max(1);
        let pair_expected_min = self.ordered_ctr_min_count.max(5) as f64;
        let smooth = self.ordered_ctr_smooth.max(1e-12);
        let mut scored: Vec<(usize, f64, Vec<i64>)> = Vec::new();

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
            let mut enc_col = Vec::with_capacity(n_rows);
            let mut count_col = Vec::with_capacity(n_rows);
            for row in 0..n_rows {
                let key = Self::ctr_single_key(x_data[row * n_features + feat]);
                enc_col.push(*map.get(&key).unwrap_or(&prior));
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
                crate::tree::CatPairConfig::default(),
            );
            tree.add_predictions_binned(binned, &mut predictions, lr);
            trees.push(tree);
        }
        trees
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
                let delta = self.huber_delta;
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

                        for b in 0..RANK_BINS {
                            let z = lo + b as f64 * inv_scale;
                            let mut s_neg = 0.0f64;
                            let mut h_neg = 0.0f64;
                            let mut s_pos = 0.0f64;
                            let mut h_pos = 0.0f64;
                            for c in 0..RANK_BINS {
                                let zc = lo + c as f64 * inv_scale;
                                let d = (z - zc).clamp(-50.0, 50.0);
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
                            rank_g_pos[b] = s_neg / neg_total - 1.0;
                            rank_h_pos[b] = (h_neg / neg_total).max(1e-16);
                            rank_g_neg[b] = s_pos / pos_total;
                            rank_h_neg[b] = (h_pos / pos_total).max(1e-16);
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
            }
            let g = gradients[row];
            let h = hessians[row].max(1e-12);
            sum_g[b] += g;
            sum_h[b] += h;
            total_g += g;
            total_h += h;
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
        if score.is_finite() {
            score.max(0.0)
        } else {
            0.0
        }
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
        score -= (total_g * total_g) / (total_h + lam);
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
            let p_eff = p as f64 * (1.0 + (numeric_features.len() as f64 + 1.0).ln());
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
}
