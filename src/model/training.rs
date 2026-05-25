//! Single-output (binary / regression) training loop.
//!
//! `fit_single` is the main per-round driver:
//! 1. Compute gradients/hessians from current predictions and labels.
//! 2. Optionally subsample rows (subsample / bootstrap / GOSS).
//! 3. For each of `n_trees_per_round` sub-trees:
//!    - Build the tree via the chosen growth policy.
//!    - Apply per-tree adjustments (HSS / EBLP / SCS / NTR / sibling
//!      block correction / hierarchical shrinkage).
//!    - Add the tree's predictions to the running sum.
//! 4. Optionally early-stop based on validation loss.
//! 5. Optionally refine globally every `refine_every` rounds.
//!
//! The K-class equivalent (`fit_multiclass`) lives in `super::multiclass`;
//! per-tree adjustments and leaf-finalization helpers stay in
//! `super::internals` (the top-level adjustments grab-bag).

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::RngExt;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::HashMap;

use super::GTBoostModel;
use crate::helpers::{bitvec_new, bitvec_set, bitvec_test, transform_gradients_for_split};
use crate::tree::{BinnedData, CatPairConfig, DecisionTree};

impl GTBoostModel {
    #[inline]
    fn sigmoid_margin(x: f64) -> f64 {
        1.0 / (1.0 + (-x.clamp(-35.0, 35.0)).exp())
    }

    fn training_margins_from_trees(&self, binned: &BinnedData, n_rows: usize) -> Vec<f64> {
        let mut out = vec![self.base_score; n_rows];
        let has_dart_w = !self.dart_tree_weights.is_empty();
        for (t_idx, tree) in self.trees.iter().enumerate() {
            let w = if has_dart_w && t_idx < self.dart_tree_weights.len() {
                self.dart_tree_weights[t_idx]
            } else {
                1.0
            };
            let scale = self.learning_rate * w;
            for row in 0..n_rows {
                let c = if tree.has_self_score_splits() {
                    tree.predict_binned_with_score(binned, row, out[row])
                } else {
                    tree.predict_binned(binned, row)
                };
                out[row] += scale * c;
            }
        }
        out
    }

    fn fit_global_cat_offsets(
        &mut self,
        binned: &BinnedData,
        y: &[f64],
        n_rows: usize,
        x_data_raw: &[f64],
        n_features_original: usize,
        sample_weight: Option<&[f64]>,
    ) {
        self.cat_offset_maps.clear();
        if self.task != "binary"
            || self.cat_offset_smooth <= 0.0
            || self.cat_offset_passes == 0
            || n_rows == 0
            || n_features_original == 0
        {
            return;
        }
        let n_features = n_features_original.min(self.cat_features.len());
        let cat_cols: Vec<usize> = (0..n_features)
            .filter(|&j| self.cat_features.get(j).copied().unwrap_or(false))
            .collect();
        if cat_cols.is_empty() {
            return;
        }
        let mut margins = self.training_margins_from_trees(binned, n_rows);
        let smooth = self.cat_offset_smooth.max(0.0);
        let cap = 2.0f64;
        self.cat_offset_maps = vec![HashMap::new(); n_features_original];
        for _pass in 0..self.cat_offset_passes.min(4) {
            for &feat in &cat_cols {
                let mut stats: HashMap<i64, (f64, f64)> = HashMap::new();
                for row in 0..n_rows {
                    let v = x_data_raw[row * n_features_original + feat];
                    if !v.is_finite() {
                        continue;
                    }
                    let p = Self::sigmoid_margin(margins[row]);
                    let mut g = p - y[row];
                    let mut h = (p * (1.0 - p)).max(1e-12);
                    if let Some(w) = sample_weight {
                        let wi = if w[row].is_finite() {
                            w[row].max(0.0)
                        } else {
                            0.0
                        };
                        g *= wi;
                        h *= wi;
                    }
                    let entry = stats.entry(v as i64).or_insert((0.0, 0.0));
                    entry.0 += g;
                    entry.1 += h;
                }
                if stats.is_empty() {
                    continue;
                }
                let mut offsets: HashMap<i64, f64> = HashMap::with_capacity(stats.len());
                for (key, (g, h)) in stats {
                    if h > 0.0 {
                        let off = (-g / (h + smooth)).clamp(-cap, cap);
                        if off.abs() > 1e-12 {
                            offsets.insert(key, off);
                            *self.cat_offset_maps[feat].entry(key).or_insert(0.0) += off;
                        }
                    }
                }
                if offsets.is_empty() {
                    continue;
                }
                for row in 0..n_rows {
                    let v = x_data_raw[row * n_features_original + feat];
                    if v.is_finite() {
                        if let Some(off) = offsets.get(&(v as i64)) {
                            margins[row] += *off;
                        }
                    }
                }
            }
        }
    }

    #[inline]
    fn apply_sample_weights(
        sample_weight: Option<&[f64]>,
        gradients: &mut [f64],
        hessians: &mut [f64],
    ) {
        if let Some(w) = sample_weight {
            for i in 0..gradients.len() {
                let wi = if w[i].is_finite() { w[i].max(0.0) } else { 0.0 };
                gradients[i] *= wi;
                hessians[i] *= wi;
            }
        }
    }

    /// Fit for regression / binary classification (single output per sample).
    pub(super) fn fit_single(
        &mut self,
        binned: &mut BinnedData,
        y: &[f64],
        n_rows: usize,
        n_features: usize,
        n_rounds: usize,
        eval_data: &mut Option<(Vec<u16>, Vec<f64>, usize, Vec<f64>, Vec<u16>)>,
        x_data_raw: &[f64], // raw training data for progressive interaction unlocking
        n_features_original: usize, // original feature count (before interactions)
        init_score: Option<&[f64]>, // optional per-row warm-start margin
        sample_weight: Option<&[f64]>, // optional per-row gradient/hessian weights
    ) {
        let mut n_feat = n_features;
        let effective_bins = self.num_bins.min(32.max(n_rows / 4));
        // Initialize predictions: per-row init_score if provided, else the
        // global base_score. Trees fit residuals against this offset.
        let mut predictions = if let Some(s) = init_score {
            s.to_vec()
        } else {
            vec![self.base_score; n_rows]
        };
        let mut gradients = vec![0.0f64; n_rows];
        let mut hessians = vec![0.0f64; n_rows];

        self.trees.reserve(n_rounds);
        self.dart_tree_weights.clear();
        // DART prediction cache: per-tree raw predictions (without lr/weight) for fast drop/restore
        let dart_enabled = self.dart_rate > 0.0;
        let mut dart_tree_preds: Vec<Vec<f64>> = Vec::new();
        let mut rng = StdRng::seed_from_u64(self.seed);
        let grow_policy = self.grow_policy.clone();
        let grow = grow_policy.as_str();
        let max_leaves = if self.max_leaves > 0 {
            self.max_leaves
        } else {
            1 << self.max_depth
        };
        let all_indices: Vec<u32> = (0..n_rows as u32).collect();

        let self_score_active = self.self_score_splits
            && self.task != "poisson"
            && grow == "depthwise"
            && self.dart_rate <= 0.0
            && self.refine_every == 0
            && self.n_refine == 0
            && self.n_leaf_splits == 0
            && self.sign_confidence_gamma <= 0.0
            && !self.ramp
            && !self.leaf_linear
            && !self.adaptive_leaf_experts
            && self.cat_lookup_smooth <= 0.0
            && !self.sparse_oblique_splits
            && !self.interval_splits
            && !(self.ordered_boost && self.subsample_rate < 1.0);
        let self_score_feature = if self_score_active {
            let idx = binned.set_numeric_feature_from_values(None, &predictions, effective_bins);
            n_feat = binned.n_features;
            Some(idx)
        } else {
            None
        };

        // Early stopping state
        let eval_active = eval_data.is_some();
        let early_stop_active = self.early_stopping_rounds > 0;
        let mut eval_preds: Vec<f64> = if let Some((_, _, en, _, _)) = eval_data {
            vec![self.base_score; *en]
        } else {
            Vec::new()
        };
        let mut best_eval_loss = f64::MAX;
        let mut best_round = 0usize;
        let mut rounds_without_improvement = 0usize;

        // ── Phase 1: Sequential boosting ────────────────────────────────

        // Pad monotone constraints to n_feat (0 for OTS/interaction features)
        let mut mono_cstr: Vec<i8> = {
            let mut v = self.monotone_constraints.clone();
            v.resize(n_feat, 0);
            v
        };

        // Cyclic features (EBM-style): one tree per feature per round
        let n_sub = if self.cyclic_features {
            let base = n_feat.max(1);
            if self.adaptive_cyclic_order && self.cyclic_feature_reuse {
                base + self.cyclic_revisit_budget()
            } else {
                base
            }
        } else {
            self.n_trees_per_round
        };

        // Progressive interaction unlocking: disabled (initial warmup at fit() start is sufficient)
        let interaction_rescore_interval: usize = 0;
        let use_momentum = self.grad_momentum > 0.0;
        let mut prev_gradients = if use_momentum {
            vec![0.0f64; n_rows]
        } else {
            Vec::new()
        };
        let mut prev_hessians = if use_momentum {
            vec![0.0f64; n_rows]
        } else {
            Vec::new()
        };

        // Adam: per-sample 2nd moment of gradient (RMSProp-like per-sample reweighting)
        let use_adam = self.adam_beta2 > 0.0;
        let mut grad_v = if use_adam {
            vec![0.0f64; n_rows]
        } else {
            Vec::new()
        };

        // ── V-OB: Variance-Gated Ordered Boosting. Per-row reliability gate based
        // on BOTH OOB count (low count → unreliable, early rounds) and OOB variance
        // (unstable trees → unreliable). High-variance rows fall back to full prediction
        // automatically → safe on regression (fixes prior OB's variance amplification
        // catastrophe). Low-variance rows use honest OOB → classification gets leakage
        // removal. No task gating needed.
        let use_ordered = self.ordered_boost && self.subsample_rate < 1.0;
        let mut oob_pred_sum = if use_ordered {
            vec![0.0f64; n_rows]
        } else {
            Vec::new()
        };
        let mut oob_pred_sum_sq = if use_ordered {
            vec![0.0f64; n_rows]
        } else {
            Vec::new()
        };
        let mut oob_count = if use_ordered {
            vec![0u32; n_rows]
        } else {
            Vec::new()
        };
        let mut effective_predictions = if use_ordered {
            vec![self.base_score; n_rows]
        } else {
            Vec::new()
        };

        // ── ARG: Auto-Regularization via OOB signal ─────────────────────────
        // Per-round closed-loop: compute OOB loss after each round; if plateaus
        // or worsens, shrink lr for subsequent trees. On RF-favoring small-N
        // data this drives lr toward tiny values (RF-like averaging). On
        // GBDT-favoring data it stays near 1.0. Activates only with V-OB path
        // (ordered_boost + subsample<1) so no new API surface. Opt-in: require
        // subsample < 1.0 AND n_rows >= 100 (stable OOB) to avoid noise.
        let use_arg = use_ordered && n_rows >= 100;
        let mut arg_lr_mult: f64 = 1.0;
        let mut arg_best_oob_loss = f64::INFINITY;
        let mut arg_rounds_no_improve: usize = 0;
        let arg_patience: usize = 3;
        let arg_decay: f64 = 0.7;
        let arg_min_mult: f64 = 0.25;

        // Gradient orthogonalization: pointer to previous-round trees set (indices of trees built at prev round)
        let use_ortho = self.ortho_alpha > 0.0;
        let mut prev_round_trees_start: usize = 0;
        let mut prev_round_trees_end: usize = 0;

        // Residuals buffer for OTS refresh (pseudo-residuals instead of raw targets)
        let is_binary = self.task == "binary";
        // LOO leaf values: only for classification (prevents overfitting without adding bias to regression)
        let use_loo = is_binary && !self.should_use_expert_leaf_admission(binned);
        let posterior_tau = self.posterior_leaf_tau();
        let cat_tuple_cfg = self.cat_tuple_config(binned);

        // Parse phase schedule: "0.3:1,0.6:2,1.0:full" → [(0.3, 1), (0.6, 2)]
        // Rounds in each phase use the specified depth; remaining rounds use self.max_depth
        let phase_schedule: Vec<(f64, usize)> = if !self.phase_schedule.is_empty() {
            self.phase_schedule
                .split(',')
                .filter_map(|s| {
                    let parts: Vec<&str> = s.trim().split(':').collect();
                    if parts.len() == 2 {
                        let frac = parts[0].parse::<f64>().ok()?;
                        let depth = if parts[1].trim() == "full" {
                            self.max_depth
                        } else {
                            parts[1].trim().parse::<usize>().ok()?
                        };
                        Some((frac, depth))
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        let configured_rank_mix_alpha = self.rank_mix_alpha;
        let rank_mix_start_frac = self.rank_mix_start_frac;
        let configured_binary_focus_gamma = self.binary_focus_gamma;
        let binary_focus_end_frac = self.binary_focus_end_frac;
        for round in 0..n_rounds {
            let progress = round as f64 / n_rounds.max(1) as f64;
            if let Some(score_idx) = self_score_feature {
                binned.set_numeric_feature_from_values(
                    Some(score_idx),
                    &predictions,
                    effective_bins,
                );
            }
            if configured_rank_mix_alpha > 0.0 && self.task == "binary" {
                self.rank_mix_alpha = if progress + 1e-12 >= rank_mix_start_frac {
                    configured_rank_mix_alpha
                } else {
                    0.0
                };
            }
            if configured_binary_focus_gamma > 0.0 && self.task == "binary" {
                self.binary_focus_gamma =
                    if binary_focus_end_frac > 0.0 && progress + 1e-12 >= binary_focus_end_frac {
                        0.0
                    } else {
                        configured_binary_focus_gamma
                    };
            }
            let trees_before_round = self.trees.len();
            let use_sibling_block_correction = self.sibling_block_correction > 0.0
                && n_sub > 1
                && !dart_enabled
                && ((self.task == "regression" && self.huber_delta <= 0.0)
                    || self.task == "binary");
            let round_prediction_base = if use_sibling_block_correction {
                predictions.clone()
            } else {
                Vec::new()
            };
            let round_eval_base = if use_sibling_block_correction && eval_active {
                eval_preds.clone()
            } else {
                Vec::new()
            };
            let mut round_in_sample_masks: Vec<Option<Vec<u64>>> =
                if use_sibling_block_correction && use_ordered {
                    Vec::with_capacity(n_sub)
                } else {
                    Vec::new()
                };
            if self.diversity_penalty > 0.0 {
                self.refresh_feature_usage_ema(n_features);
            }
            // Staged complexity: override depth based on phase schedule
            let phase_depth = if !phase_schedule.is_empty() {
                let progress = (round + 1) as f64 / n_rounds as f64;
                let mut depth = self.max_depth;
                for &(frac, d) in &phase_schedule {
                    if progress <= frac {
                        depth = d;
                        break;
                    }
                }
                depth
            } else {
                self.max_depth
            };

            // Adaptive lambda: increases with round to regularize later trees more
            let round_lambda = if self.lambda_schedule > 0.0 && n_rounds > 1 {
                self.lambda_reg
                    * (1.0 + self.lambda_schedule * round as f64 / (n_rounds - 1).max(1) as f64)
            } else {
                self.lambda_reg
            };

            // V-OB variance-gated honesty. Reliability = sigmoid(count gate) × (1 / (1 + variance/τ)).
            // Combines cosine annealing (training-phase trust-growth) with per-row variance
            // signal (OOB-estimator epistemic uncertainty). High-variance rows fall back to
            // full pred automatically → regression-safe.
            if use_ordered && self.trees.len() >= 2 {
                let n_trees_f = self.trees.len() as f64;
                let t = round as f64;
                let total_t = n_rounds.max(1) as f64;
                let alpha_phase = 0.5 * (1.0 - (std::f64::consts::PI * t / total_t).cos());
                // Global variance scale: median of (oob_mean^2). This adapts the threshold to
                // data scale (small for binary/logits, large for regression targets).
                let mut abs_means: Vec<f64> = Vec::with_capacity(n_rows);
                for i in 0..n_rows {
                    if oob_count[i] >= 2 {
                        let mean =
                            oob_pred_sum[i] * (n_trees_f / oob_count[i] as f64) - self.base_score;
                        abs_means.push(mean.abs());
                    }
                }
                let scale_base = if !abs_means.is_empty() {
                    let mut sorted = abs_means.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    sorted[sorted.len() / 2].max(1e-6)
                } else {
                    1.0
                };
                let tau_var = (scale_base * 0.5).powi(2); // variance threshold ≈ (half the typical signal)^2
                for i in 0..n_rows {
                    if oob_count[i] >= 1 {
                        let c = oob_count[i] as f64;
                        let oob_pred = self.base_score + oob_pred_sum[i] * (n_trees_f / c);
                        // Per-tree variance of OOB contributions
                        let mean_per_tree = oob_pred_sum[i] / c;
                        let ex2 = oob_pred_sum_sq[i] / c;
                        let oob_var = (ex2 - mean_per_tree * mean_per_tree).max(0.0);
                        // Reliability: count-gate × variance-gate
                        let count_gate = 1.0 / (1.0 + (-((c - 3.0) / 2.0)).exp());
                        let var_gate = 1.0 / (1.0 + oob_var / tau_var);
                        let reliability = count_gate * var_gate;
                        let w = alpha_phase * reliability;
                        effective_predictions[i] = w * oob_pred + (1.0 - w) * predictions[i];
                    } else {
                        effective_predictions[i] = predictions[i];
                    }
                }
                self.compute_gradients_hessians(
                    y,
                    &effective_predictions,
                    &mut gradients,
                    &mut hessians,
                );
            } else {
                self.compute_gradients_hessians(y, &predictions, &mut gradients, &mut hessians);
            }
            Self::apply_sample_weights(sample_weight, &mut gradients, &mut hessians);

            // Gradient momentum: blend current gradients with previous round's
            if use_momentum && round > 0 {
                let mom = self.grad_momentum;
                let fresh = 1.0 - mom;
                for i in 0..n_rows {
                    gradients[i] = fresh * gradients[i] + mom * prev_gradients[i];
                    hessians[i] = fresh * hessians[i] + mom * prev_hessians[i];
                }
            }
            if use_momentum {
                prev_gradients.copy_from_slice(&gradients);
                prev_hessians.copy_from_slice(&hessians);
            }

            // Adam-style per-sample gradient normalization (preserves scale via mean(v))
            if use_adam {
                let b2 = self.adam_beta2;
                let one_minus_b2 = 1.0 - b2;
                let eps = self.adam_eps;
                let mut v_sum = 0.0f64;
                for i in 0..n_rows {
                    grad_v[i] = b2 * grad_v[i] + one_minus_b2 * gradients[i] * gradients[i];
                    v_sum += grad_v[i];
                }
                let v_mean = (v_sum / n_rows as f64).max(eps * eps);
                let inv_sqrt_mean = 1.0 / v_mean.sqrt();
                for i in 0..n_rows {
                    // scale factor: sqrt(v_mean) / (sqrt(v_i) + eps) — relative reweighting
                    let s = inv_sqrt_mean.recip() / (grad_v[i].sqrt() + eps);
                    gradients[i] *= s;
                    hessians[i] *= s;
                }
            }

            // Gradient orthogonalization vs previous round's trees' leaves.
            // For each prev-round tree, subtract per-leaf mean gradient. Stacks across trees.
            if use_ortho && round > 0 && prev_round_trees_end > prev_round_trees_start {
                let alpha = self.ortho_alpha;
                let n_trees = prev_round_trees_end - prev_round_trees_start;
                let alpha_per_tree = alpha / n_trees as f64;
                for t_idx in prev_round_trees_start..prev_round_trees_end {
                    let tree = &self.trees[t_idx];
                    let n_leaves = tree.values.len();
                    if n_leaves <= 1 {
                        continue;
                    }
                    let mut leaf_sum = vec![0.0f64; n_leaves];
                    let mut leaf_cnt = vec![0usize; n_leaves];
                    let mut leaf_ids = vec![0u32; n_rows];
                    for i in 0..n_rows {
                        let lid = tree.route_to_leaf(binned, i);
                        leaf_ids[i] = lid as u32;
                        leaf_sum[lid] += gradients[i];
                        leaf_cnt[lid] += 1;
                    }
                    for l in 0..n_leaves {
                        if leaf_cnt[l] > 0 {
                            leaf_sum[l] /= leaf_cnt[l] as f64;
                        }
                    }
                    for i in 0..n_rows {
                        gradients[i] -= alpha_per_tree * leaf_sum[leaf_ids[i] as usize];
                    }
                }
            }

            // Gradient clipping
            if self.grad_clip > 0.0 {
                let clip = self.grad_clip;
                for g in gradients.iter_mut() {
                    *g = g.clamp(-clip, clip);
                }
            }

            // ── DART: drop random trees using cached predictions (O(n_rows) per tree) ──
            let dart_active = dart_enabled && !self.trees.is_empty();
            let mut dart_dropped: Vec<usize> = Vec::new();
            let mut dart_dropped_wsum = 0.0f64;
            if dart_active {
                for j in 0..self.trees.len() {
                    if rng.random::<f64>() < self.dart_rate {
                        dart_dropped.push(j);
                    }
                }
                if !dart_dropped.is_empty() {
                    let lr = self.learning_rate;
                    for &j in &dart_dropped {
                        let w = if j < self.dart_tree_weights.len() {
                            self.dart_tree_weights[j]
                        } else {
                            1.0
                        };
                        dart_dropped_wsum += w;
                        let cached = &dart_tree_preds[j];
                        let lrw = lr * w;
                        if n_rows >= 4096 {
                            predictions
                                .par_chunks_mut(1024)
                                .enumerate()
                                .for_each(|(ci, chunk)| {
                                    let start = ci * 1024;
                                    for (jj, pred) in chunk.iter_mut().enumerate() {
                                        *pred -= lrw * cached[start + jj];
                                    }
                                });
                        } else {
                            for i in 0..n_rows {
                                predictions[i] -= lrw * cached[i];
                            }
                        }
                    }
                    // Recompute gradients with dropped-tree predictions
                    self.compute_gradients_hessians(y, &predictions, &mut gradients, &mut hessians);
                    Self::apply_sample_weights(sample_weight, &mut gradients, &mut hessians);
                }
            }

            // Split-criterion transform: rank / sign replace gradients for split finding only.
            // Leaves refit with original gradients below to keep leaf values in Newton scale.
            let split_mode_active = self.split_criterion != "newton";
            let split_owned = if split_mode_active {
                transform_gradients_for_split(&gradients, &self.split_criterion)
            } else {
                None
            };
            let (g_split_ref, h_split_ref): (&[f64], &[f64]) =
                if let Some((ref g, ref h)) = split_owned {
                    (g.as_slice(), h.as_slice())
                } else {
                    (&gradients, &hessians)
                };

            // Build N trees per round (all see same gradients, different subsamples)
            let use_bayesian = self.bagging_temperature > 0.0;

            let sub_scale = if n_sub > 1 { 1.0 / n_sub as f64 } else { 1.0 };

            let use_ncl = self.ncl_lambda > 0.0 && n_sub > 1;
            let cyclic_round_order = if self.cyclic_features
                && self.adaptive_cyclic_order
                && n_feat > 0
                && !use_ncl
            {
                self.cyclic_feature_order_by_pressure(binned, &gradients, &hessians, round_lambda)
            } else {
                Vec::new()
            };

            if n_sub > 1 && !use_bayesian && !use_ncl {
                // ── Parallel path: build N independent trees concurrently ──
                let planned_subsamples = self.round_subsamples(&mut rng, n_rows, n_sub);
                // Pre-generate per-subtree seeds so each thread has its own RNG
                let sub_seeds: Vec<u64> = (0..n_sub).map(|_| rng.random()).collect();

                let built_trees: Vec<(DecisionTree, Option<Vec<u64>>, Vec<u32>)> = sub_seeds
                    .into_par_iter()
                    .enumerate()
                    .map(|(sub_idx, seed)| {
                        let mut sub_rng = StdRng::seed_from_u64(seed);

                        // Hetero Architecture Cycling (HAC): cycle grow_policy + depth + lambda
                        // + extra_trees across sub-trees. Each round contains 3+ architectural
                        // perspectives on the same residuals → automatic archetype mixing without
                        // dataset-profile hints. Requires n_sub >= 3.
                        let (sub_depth, sub_lambda, sub_grow_override, sub_extra_trees) = if self
                            .hetero_trees
                            && n_sub >= 3
                        {
                            match sub_idx % 3 {
                                // Variant A: standard depthwise, moderate params
                                0 => (
                                    phase_depth,
                                    round_lambda,
                                    Some("depthwise"),
                                    self.extra_trees,
                                ),
                                // Variant B: oblivious (symmetric), shallower, more reg
                                1 => (
                                    phase_depth.saturating_sub(1).max(2),
                                    round_lambda * 2.0,
                                    Some("oblivious"),
                                    false,
                                ),
                                // Variant C: deeper + RF-style random splits (extra_trees)
                                2 => (phase_depth + 1, round_lambda * 0.5, Some("depthwise"), true),
                                _ => (phase_depth, round_lambda, None, self.extra_trees),
                            }
                        } else {
                            (phase_depth, round_lambda, None, self.extra_trees)
                        };
                        let sub_grow = sub_grow_override.unwrap_or(grow);

                        // Cyclic features: each sub-tree gets exactly one feature
                        let feature_mask = if self.cyclic_features {
                            let mut mask = vec![false; n_feat];
                            if n_feat > 0 {
                                let feat = cyclic_round_order
                                    .get(sub_idx)
                                    .copied()
                                    .unwrap_or((round + sub_idx) % n_feat);
                                // Rotate the feature order across boosting rounds. With sibling
                                // feedback enabled, this avoids always giving low-index features
                                // stale residuals and high-index features fully updated residuals.
                                mask[feat] = true;
                            }
                            mask
                        } else {
                            self.make_feature_mask_for_subtree(&mut sub_rng, n_feat, round, sub_idx)
                        };

                        let sample_indices = planned_subsamples[sub_idx].clone();

                        let need_oob_mask = self.ordered_boost && self.subsample_rate < 1.0;
                        let (structure_indices, estimation_indices, in_sample) = if self.honest {
                            if self.honest_fraction <= 0.0 && self.subsample_rate < 1.0 {
                                let mut in_s = bitvec_new(n_rows);
                                for &idx in &sample_indices {
                                    bitvec_set(&mut in_s, idx as usize);
                                }
                                let complement: Vec<u32> = (0..n_rows as u32)
                                    .filter(|&i| !bitvec_test(&in_s, i as usize))
                                    .collect();
                                (sample_indices, complement, Some(in_s))
                            } else {
                                let mut shuffled = sample_indices;
                                shuffled.shuffle(&mut sub_rng);
                                let frac = self.honest_fraction.clamp(0.1, 0.9);
                                let est_size = (shuffled.len() as f64 * frac).round() as usize;
                                let mid = shuffled.len() - est_size;
                                (shuffled[..mid].to_vec(), shuffled[mid..].to_vec(), None)
                            }
                        } else if need_oob_mask {
                            let mut in_s = bitvec_new(n_rows);
                            for &idx in &sample_indices {
                                bitvec_set(&mut in_s, idx as usize);
                            }
                            (sample_indices, Vec::new(), Some(in_s))
                        } else {
                            (sample_indices, Vec::new(), None)
                        };
                        let build_indices = &structure_indices;

                        let tree_seed: u64 = sub_rng.random();
                        let use_cdss = self.honest
                            && self.complement_debias_mode > 0
                            && sub_grow == "depthwise"
                            && !sub_extra_trees
                            && !estimation_indices.is_empty();
                        let lookup_smooth_build = if self.adaptive_leaf_experts {
                            0.0
                        } else {
                            self.cat_lookup_smooth
                        };

                        let mut tree = match sub_grow {
                            "leafwise" => DecisionTree::build_leafwise(
                                binned,
                                g_split_ref,
                                h_split_ref,
                                build_indices,
                                sub_lambda,
                                self.gamma,
                                sub_depth,
                                max_leaves,
                                self.min_child_weight,
                                &feature_mask,
                                self.colsample_bylevel,
                                tree_seed,
                                self.random_strength,
                                self.cat_smooth,
                                lookup_smooth_build,
                                &mono_cstr,
                                self.gain_penalty,
                                sub_extra_trees,
                                CatPairConfig {
                                    enabled: self.jit_catpair_enabled,
                                    top_k_cat: self.jit_catpair_top_k,
                                    k_buckets: self.jit_catpair_k_buckets,
                                    min_node_rows: self.jit_catpair_min_node_rows,
                                    max_node_depth: self.jit_catpair_max_node_depth,
                                    gain_margin: self.jit_catpair_gain_margin,
                                },
                            ),
                            "oblivious" => DecisionTree::build_oblivious(
                                binned,
                                g_split_ref,
                                h_split_ref,
                                build_indices,
                                sub_lambda,
                                self.gamma,
                                sub_depth,
                                self.min_child_weight,
                                &feature_mask,
                                self.gain_penalty,
                                sub_extra_trees,
                                tree_seed,
                            ),
                            _ if use_cdss => DecisionTree::build_depthwise_debiased(
                                binned,
                                g_split_ref,
                                h_split_ref,
                                build_indices,
                                &estimation_indices,
                                sub_lambda,
                                self.gamma,
                                sub_depth,
                                self.min_child_weight,
                                &feature_mask,
                                self.colsample_bylevel,
                                tree_seed,
                                self.random_strength,
                                self.cat_smooth,
                                lookup_smooth_build,
                                &mono_cstr,
                                self.gain_penalty,
                                sub_extra_trees,
                                self.complement_debias_mode,
                                self.lookahead_alpha,
                                self.adaptive_leaf_experts
                                    || self.leaf_linear
                                    || self.cat_lookup_smooth > 0.0,
                            ),
                            _ => DecisionTree::build_depthwise(
                                binned,
                                g_split_ref,
                                h_split_ref,
                                build_indices,
                                sub_lambda,
                                self.l1_reg,
                                self.gamma,
                                sub_depth,
                                self.min_child_weight,
                                &feature_mask,
                                self.colsample_bylevel,
                                tree_seed,
                                self.random_strength,
                                self.cat_smooth,
                                self.cat_lookup_smooth,
                                &mono_cstr,
                                self.gain_penalty,
                                sub_extra_trees,
                                self.lookahead_alpha,
                                self.adaptive_leaf_experts
                                    || self.leaf_linear
                                    || self.cat_lookup_smooth > 0.0,
                                self.sparse_oblique_splits,
                                self.interval_splits,
                                None,
                                CatPairConfig {
                                    enabled: self.jit_catpair_enabled,
                                    top_k_cat: self.jit_catpair_top_k,
                                    k_buckets: self.jit_catpair_k_buckets,
                                    min_node_rows: self.jit_catpair_min_node_rows,
                                    max_node_depth: self.jit_catpair_max_node_depth,
                                    gain_margin: self.jit_catpair_gain_margin,
                                },
                            ),
                        };

                        // Rank/sign split mode: tree was built with transformed gradients,
                        // so leaf values are in rank/sign scale. ALWAYS refit on original
                        // gradients to reset leaves to Newton scale before any honest_tau blend.
                        if split_mode_active {
                            tree.refit_leaves_l1(
                                binned,
                                &gradients,
                                &hessians,
                                build_indices,
                                sub_lambda,
                                self.l1_reg,
                            );
                        }

                        if self.honest {
                            tree.refit_leaves_robust(
                                binned,
                                &gradients,
                                &hessians,
                                &estimation_indices,
                                sub_lambda,
                                self.honest_tau,
                                self.leaf_trim_pct,
                                self.leaf_median,
                                self.leaf_median_blend,
                                self.leaf_mad_clip,
                                self.leaf_adaptive_blend_kappa,
                            );

                            if self.cat_lookup_smooth > 0.0
                                && !self.adaptive_leaf_experts
                                && sub_grow != "oblivious"
                            {
                                tree.refit_cat_lookups(
                                    binned,
                                    &gradients,
                                    &hessians,
                                    &estimation_indices,
                                    sub_lambda,
                                    self.cat_lookup_smooth,
                                    self.min_child_weight,
                                );
                            }
                        } else if self.leaf_trim_pct > 0.0
                            || self.leaf_median
                            || self.leaf_median_blend > 0.0
                            || self.leaf_mad_clip > 0.0
                            || self.leaf_adaptive_blend_kappa > 0.0
                        {
                            // Non-honest + trim: post-build robust refit on structure indices.
                            // Keeps gradient scale intact, only M-estimators leaf values.
                            tree.refit_leaves_robust(
                                binned,
                                &gradients,
                                &hessians,
                                build_indices,
                                sub_lambda,
                                0.0,
                                self.leaf_trim_pct,
                                self.leaf_median,
                                self.leaf_median_blend,
                                self.leaf_mad_clip,
                                self.leaf_adaptive_blend_kappa,
                            );
                        }

                        if self.adaptive_leaf_experts || self.cat_lookup_smooth > 0.0 {
                            if self.adaptive_leaf_experts {
                                let lookup_indices =
                                    if self.honest && !estimation_indices.is_empty() {
                                        &estimation_indices
                                    } else {
                                        build_indices
                                    };
                                tree.install_best_lookups_with_config(
                                    binned,
                                    &gradients,
                                    &hessians,
                                    lookup_indices,
                                    sub_lambda,
                                    self.gamma,
                                    self.min_child_weight,
                                    self.cat_lookup_smooth,
                                    self.adaptive_cat_lookup_smooth,
                                    cat_tuple_cfg.as_ref(),
                                );
                            } else if sub_grow == "oblivious" {
                                // Oblivious trees never install lookups during structure building.
                                tree.install_cat_lookups(
                                    binned,
                                    &gradients,
                                    &hessians,
                                    build_indices,
                                    sub_lambda,
                                    self.gamma,
                                    self.min_child_weight,
                                    self.cat_lookup_smooth,
                                );
                            }
                        }
                        // Scale for multi-tree per round
                        for v in tree.values.iter_mut() {
                            *v *= sub_scale;
                        }
                        tree.scale_ramp_slopes(sub_scale);
                        tree.scale_cat_lookups(sub_scale);

                        let leaf_indices = if self.honest && !estimation_indices.is_empty() {
                            estimation_indices
                        } else {
                            structure_indices
                        };
                        (tree, in_sample, leaf_indices)
                    })
                    .collect();

                // Apply results sequentially (predictions update + store trees)
                let lr_factor = if self.lr_decay < 1.0 && n_rounds > 1 {
                    1.0 - (1.0 - self.lr_decay) * (round as f64) / (n_rounds as f64 - 1.0)
                } else {
                    1.0
                };

                for (mut tree, in_sample, leaf_indices) in built_trees {
                    if lr_factor < 1.0 {
                        for v in tree.values.iter_mut() {
                            *v *= lr_factor;
                        }
                        tree.scale_ramp_slopes(lr_factor);
                        tree.scale_cat_lookups(lr_factor);
                    }
                    if use_arg && arg_lr_mult < 1.0 {
                        for v in tree.values.iter_mut() {
                            *v *= arg_lr_mult;
                        }
                        tree.scale_ramp_slopes(arg_lr_mult);
                        tree.scale_cat_lookups(arg_lr_mult);
                    }
                    // max_delta_step: clip leaf values
                    if self.max_delta_step > 0.0 {
                        let mds = self.max_delta_step;
                        for v in tree.values.iter_mut() {
                            *v = v.clamp(-mds, mds);
                        }
                    }

                    // DART: scale new tree (XGBoost-style: new_w = dropped_wsum / (nd + 1))
                    let dart_new_w = if !dart_dropped.is_empty() {
                        let nd = dart_dropped.len() as f64;
                        dart_dropped_wsum / (nd + 1.0)
                    } else {
                        1.0
                    };
                    let effective_lr = self.learning_rate * dart_new_w;
                    self.apply_hierarchical_shrinkage(&mut tree);
                    if use_loo && !self.honest {
                        if posterior_tau > 0.0 {
                            tree.posterior_shrink_leaves(posterior_tau);
                        }
                        tree.add_predictions_loo(
                            binned,
                            &mut predictions,
                            effective_lr,
                            &gradients,
                            &hessians,
                            round_lambda,
                            &leaf_indices,
                            posterior_tau,
                        );
                    } else {
                        if posterior_tau > 0.0 {
                            tree.posterior_shrink_leaves(posterior_tau);
                        }
                        tree.add_predictions_binned(binned, &mut predictions, effective_lr);
                    }

                    // Leaf correction: recompute gradients and refit leaf values
                    if self.leaf_correction > 0 {
                        let lr = effective_lr;
                        let mut corr_grads = vec![0.0f64; n_rows];
                        let mut corr_hess = vec![0.0f64; n_rows];
                        for _step in 0..self.leaf_correction {
                            self.compute_gradients_hessians(
                                y,
                                &predictions,
                                &mut corr_grads,
                                &mut corr_hess,
                            );
                            Self::apply_sample_weights(
                                sample_weight,
                                &mut corr_grads,
                                &mut corr_hess,
                            );
                            tree.add_predictions_binned(binned, &mut predictions, -lr);
                            tree.refit_leaves_l1(
                                binned,
                                &corr_grads,
                                &corr_hess,
                                &all_indices,
                                round_lambda,
                                self.l1_reg,
                            );
                            if self.adaptive_leaf_experts && self.cat_lookup_smooth > 0.0 {
                                tree.install_best_lookups_with_config(
                                    binned,
                                    &corr_grads,
                                    &corr_hess,
                                    &all_indices,
                                    round_lambda,
                                    self.gamma,
                                    self.min_child_weight,
                                    self.cat_lookup_smooth,
                                    self.adaptive_cat_lookup_smooth,
                                    cat_tuple_cfg.as_ref(),
                                );
                            }
                            tree.add_predictions_binned(binned, &mut predictions, lr);
                        }
                    }

                    let tree_uses_self_score = if let Some(score_idx) = self_score_feature {
                        let edges = binned.bin_edges[score_idx].clone();
                        tree.rewrite_feature_as_self_score(score_idx, &edges)
                    } else {
                        false
                    };

                    if eval_active {
                        let (eval_bins, _eval_y, en, _, eval_cll_bins) =
                            eval_data.as_ref().unwrap();
                        let en = *en;
                        if tree_uses_self_score {
                            tree.add_predictions_binned_raw_with_score(
                                eval_bins,
                                en,
                                &mut eval_preds[..en],
                                effective_lr,
                                eval_cll_bins,
                            );
                        } else {
                            tree.add_predictions_binned_raw(
                                eval_bins,
                                en,
                                &mut eval_preds[..en],
                                effective_lr,
                                eval_cll_bins,
                            );
                        }
                    }
                    if self.n_refine > 0 {
                        if let Some(ref mask) = in_sample {
                            self.tree_in_sample.push(mask.clone());
                        }
                    }
                    // Cache per-tree predictions for fast DART drop/restore
                    if dart_enabled {
                        let mut tp = vec![0.0f64; n_rows];
                        for i in 0..n_rows {
                            tp[i] = tree.predict_binned(binned, i);
                        }
                        dart_tree_preds.push(tp);
                    }
                    self.dart_tree_weights.push(dart_new_w);
                    self.apply_eblp(&mut tree);
                    self.apply_hss(&mut tree);
                    self.apply_scs(&mut tree, binned, &gradients, n_rows);
                    self.apply_newton_trust_region(&mut tree);
                    // V-OB: accumulate per-row OOB predictions AND squared predictions
                    // for variance estimation. Gate via reliability at gradient-compute time.
                    if use_ordered {
                        if let Some(ref mask) = in_sample {
                            for i in 0..n_rows {
                                if !bitvec_test(mask, i) {
                                    let tp = tree.predict_binned(binned, i);
                                    let contribution = effective_lr * tp;
                                    oob_pred_sum[i] += contribution;
                                    oob_pred_sum_sq[i] += contribution * contribution;
                                    oob_count[i] += 1;
                                }
                            }
                        }
                    }
                    if use_sibling_block_correction && use_ordered {
                        round_in_sample_masks.push(in_sample);
                    }
                    self.trees.push(tree);
                }
            } else {
                // ── Sequential path: single tree, bayesian bootstrap, or NCL ──
                let goss_on = self.goss_top_rate > 0.0
                    && self.goss_other_rate > 0.0
                    && !self.honest
                    && !use_bayesian
                    && !use_ncl;
                let planned_subsamples = if !goss_on && n_sub > 1 {
                    Some(self.round_subsamples(&mut rng, n_rows, n_sub))
                } else {
                    None
                };
                let (grad_orig, hess_orig) = if goss_on && n_sub > 1 {
                    (gradients.clone(), hessians.clone())
                } else {
                    (Vec::new(), Vec::new())
                };
                let mut cyclic_remaining: Vec<usize> =
                    if self.cyclic_features && self.adaptive_cyclic_order && n_feat > 0 {
                        (0..n_feat).collect()
                    } else {
                        Vec::new()
                    };
                let mut cyclic_usage: Vec<usize> = if self.cyclic_features
                    && self.adaptive_cyclic_order
                    && self.cyclic_feature_reuse
                    && n_feat > 0
                {
                    vec![0usize; n_feat]
                } else {
                    Vec::new()
                };
                let mut cyclic_last_feature: Option<usize> = None;
                let cyclic_round_start_pressure = if self.cyclic_features
                    && self.adaptive_cyclic_order
                    && self.cyclic_feature_reuse
                    && self.cyclic_revisit_min_pressure_ratio > 0.0
                    && n_feat > 0
                {
                    self.max_cyclic_feature_pressure(binned, &gradients, &hessians, round_lambda)
                } else {
                    0.0
                };
                let mut adaptive_mask_usage: Vec<usize> =
                    if !self.cyclic_features && self.adaptive_feature_mask && use_ncl && n_feat > 0
                    {
                        vec![0usize; n_feat]
                    } else {
                        Vec::new()
                    };
                let mut adaptive_anchor_usage: Vec<usize> = if !self.cyclic_features
                    && self.adaptive_root_anchor
                    && use_ncl
                    && n_feat > 0
                {
                    vec![0usize; n_feat]
                } else {
                    Vec::new()
                };

                for _sub_idx in 0..n_sub {
                    if let Some(score_idx) = self_score_feature {
                        binned.set_numeric_feature_from_values(
                            Some(score_idx),
                            &predictions,
                            effective_bins,
                        );
                    }
                    // Restore scaled gradients between sub-trees (GOSS modifies in-place)
                    if goss_on && n_sub > 1 && _sub_idx > 0 {
                        gradients.copy_from_slice(&grad_orig);
                        hessians.copy_from_slice(&hess_orig);
                    }

                    let indices = if goss_on {
                        let importance = self.goss_importance(&gradients, &hessians);
                        let a_eff = self.goss_annealed_a(round, n_rounds);
                        let (sel, scales) =
                            self.goss_select(&mut rng, &importance, a_eff, self.goss_other_rate);
                        // Scale gradients/hessians in-place for the "other" rows
                        for i in 0..n_rows {
                            let s = scales[i];
                            if s > 0.0 && s != 1.0 {
                                gradients[i] *= s;
                                hessians[i] *= s;
                            }
                        }
                        sel
                    } else {
                        planned_subsamples
                            .as_ref()
                            .map(|v| v[_sub_idx].clone())
                            .unwrap_or_else(|| self.subsample_indices(&mut rng, n_rows))
                    };

                    let mut root_anchor_feature: Option<usize> = None;
                    let feature_mask = if self.cyclic_features {
                        let mut mask = vec![false; n_feat];
                        if n_feat > 0 {
                            let feat = if self.adaptive_cyclic_order {
                                if self.cyclic_feature_reuse && cyclic_remaining.is_empty() {
                                    if self.cyclic_revisit_min_pressure_ratio > 0.0
                                        && cyclic_round_start_pressure > 0.0
                                    {
                                        let pressure = self.max_cyclic_feature_pressure(
                                            binned,
                                            &gradients,
                                            &hessians,
                                            round_lambda,
                                        );
                                        if pressure
                                            < self.cyclic_revisit_min_pressure_ratio
                                                * cyclic_round_start_pressure
                                        {
                                            break;
                                        }
                                    }
                                    let selected = self
                                        .take_cyclic_feature_by_residual_auction(
                                            &mut cyclic_usage,
                                            cyclic_last_feature,
                                            binned,
                                            &gradients,
                                            &hessians,
                                            round_lambda,
                                        )
                                        .unwrap_or((round + _sub_idx) % n_feat);
                                    cyclic_last_feature = Some(selected);
                                    selected
                                } else {
                                    let selected = self
                                        .take_best_cyclic_feature_by_pressure(
                                            &mut cyclic_remaining,
                                            binned,
                                            &gradients,
                                            &hessians,
                                            round_lambda,
                                        )
                                        .unwrap_or((round + _sub_idx) % n_feat);
                                    if self.cyclic_feature_reuse && selected < cyclic_usage.len() {
                                        cyclic_usage[selected] += 1;
                                        cyclic_last_feature = Some(selected);
                                    }
                                    selected
                                }
                            } else {
                                (round + _sub_idx) % n_feat
                            };
                            // Rotate the feature order across boosting rounds. With sibling
                            // feedback enabled, every feature periodically gets early, middle,
                            // and late residual views instead of a fixed construction position.
                            mask[feat] = true;
                            if self.cyclic_partner_features {
                                root_anchor_feature = Some(feat);
                                if let Some(partner) = self.best_cyclic_partner_by_pair_pressure(
                                    binned,
                                    feat,
                                    &gradients,
                                    &hessians,
                                    round_lambda,
                                ) {
                                    mask[partner] = true;
                                }
                            }
                        }
                        mask
                    } else {
                        if self.adaptive_root_anchor && use_ncl && n_feat > 0 {
                            root_anchor_feature = self.take_adaptive_root_anchor_by_pressure(
                                binned,
                                &gradients,
                                &hessians,
                                round_lambda,
                                &mut adaptive_anchor_usage,
                            );
                        }
                        let mut mask = if self.adaptive_feature_mask && use_ncl && n_feat > 0 {
                            self.make_adaptive_feature_mask_by_pressure(
                                binned,
                                &gradients,
                                &hessians,
                                round_lambda,
                                &mut adaptive_mask_usage,
                            )
                        } else {
                            self.make_feature_mask_for_subtree(&mut rng, n_feat, round, _sub_idx)
                        };
                        if let Some(anchor) = root_anchor_feature {
                            if anchor < mask.len() {
                                mask[anchor] = true;
                            }
                        }
                        mask
                    };

                    // For OB in sequential path: always build in_sample mask from subsample indices.
                    let seq_in_sample: Option<Vec<u64>> = if use_ordered {
                        let mut m = bitvec_new(n_rows);
                        for &idx in &indices {
                            bitvec_set(&mut m, idx as usize);
                        }
                        Some(m)
                    } else {
                        None
                    };
                    let mut expert_calibration_indices: Vec<u32> = Vec::new();
                    let use_expert_admission_tree =
                        self.should_use_expert_leaf_admission(binned) && !use_bayesian && !goss_on;
                    let (structure_indices, estimation_indices) = if self.honest {
                        if self.honest_fraction <= 0.0 && self.subsample_rate < 1.0 {
                            let mut in_sample = bitvec_new(n_rows);
                            for &idx in &indices {
                                bitvec_set(&mut in_sample, idx as usize);
                            }
                            if self.n_refine > 0 {
                                self.tree_in_sample.push(in_sample.clone());
                            }
                            let complement: Vec<u32> = (0..n_rows as u32)
                                .filter(|&i| !bitvec_test(&in_sample, i as usize))
                                .collect();
                            (indices.clone(), complement)
                        } else {
                            let mut shuffled = indices.clone();
                            shuffled.shuffle(&mut rng);
                            let frac = self.honest_fraction.clamp(0.1, 0.9);
                            let est_size = (shuffled.len() as f64 * frac).round() as usize;
                            let mid = shuffled.len() - est_size;
                            let si = shuffled[..mid].to_vec();
                            let ei = shuffled[mid..].to_vec();
                            (si, ei)
                        }
                    } else if use_expert_admission_tree {
                        if let Some((build, cal)) =
                            self.vceg_partition_indices(&indices, round, _sub_idx)
                        {
                            expert_calibration_indices = cal;
                            (build, Vec::new())
                        } else {
                            (indices.clone(), Vec::new())
                        }
                    } else {
                        (indices.clone(), Vec::new())
                    };
                    let build_indices = &structure_indices;

                    if use_bayesian {
                        let temp = self.bagging_temperature;
                        for &idx in build_indices {
                            let i = idx as usize;
                            let u: f64 = rng.random::<f64>().max(1e-300);
                            let w = ((-u.ln()) * temp).max(0.001);
                            gradients[i] *= w;
                            hessians[i] *= w;
                        }
                    }

                    // Sibling residual feedback: when ncl_lambda>0, this sequential
                    // path recomputes gradients after each sibling tree. Split search
                    // must therefore read the current gradients, not the stale
                    // per-round snapshot used by the parallel same-gradient path.
                    let split_local = if split_mode_active {
                        transform_gradients_for_split(&gradients, &self.split_criterion)
                    } else {
                        None
                    };
                    let (g_build_ref, h_build_ref): (&[f64], &[f64]) =
                        if let Some((ref g, ref h)) = split_local {
                            (g.as_slice(), h.as_slice())
                        } else {
                            (&gradients, &hessians)
                        };

                    let tree_seed: u64 = rng.random();
                    let use_cdss = self.honest
                        && self.complement_debias_mode > 0
                        && grow == "depthwise"
                        && !self.extra_trees
                        && !estimation_indices.is_empty();
                    let lookup_smooth_build = if self.adaptive_leaf_experts {
                        0.0
                    } else {
                        self.cat_lookup_smooth
                    };

                    let mut tree = match grow {
                        "leafwise" => DecisionTree::build_leafwise(
                            binned,
                            g_build_ref,
                            h_build_ref,
                            build_indices,
                            round_lambda,
                            self.gamma,
                            phase_depth,
                            max_leaves,
                            self.min_child_weight,
                            &feature_mask,
                            self.colsample_bylevel,
                            tree_seed,
                            self.random_strength,
                            self.cat_smooth,
                            lookup_smooth_build,
                            &mono_cstr,
                            self.gain_penalty,
                            self.extra_trees,
                            CatPairConfig {
                                enabled: self.jit_catpair_enabled,
                                top_k_cat: self.jit_catpair_top_k,
                                k_buckets: self.jit_catpair_k_buckets,
                                min_node_rows: self.jit_catpair_min_node_rows,
                                max_node_depth: self.jit_catpair_max_node_depth,
                                gain_margin: self.jit_catpair_gain_margin,
                            },
                        ),
                        "oblivious" => DecisionTree::build_oblivious(
                            binned,
                            g_build_ref,
                            h_build_ref,
                            build_indices,
                            round_lambda,
                            self.gamma,
                            phase_depth,
                            self.min_child_weight,
                            &feature_mask,
                            self.gain_penalty,
                            self.extra_trees,
                            tree_seed,
                        ),
                        _ if use_cdss => DecisionTree::build_depthwise_debiased(
                            binned,
                            g_build_ref,
                            h_build_ref,
                            build_indices,
                            &estimation_indices,
                            round_lambda,
                            self.gamma,
                            phase_depth,
                            self.min_child_weight,
                            &feature_mask,
                            self.colsample_bylevel,
                            tree_seed,
                            self.random_strength,
                            self.cat_smooth,
                            lookup_smooth_build,
                            &mono_cstr,
                            self.gain_penalty,
                            self.extra_trees,
                            self.complement_debias_mode,
                            self.lookahead_alpha,
                            self.adaptive_leaf_experts
                                || self.leaf_linear
                                || self.cat_lookup_smooth > 0.0,
                        ),
                        _ => DecisionTree::build_depthwise(
                            binned,
                            g_build_ref,
                            h_build_ref,
                            build_indices,
                            round_lambda,
                            self.l1_reg,
                            self.gamma,
                            phase_depth,
                            self.min_child_weight,
                            &feature_mask,
                            self.colsample_bylevel,
                            tree_seed,
                            self.random_strength,
                            self.cat_smooth,
                            self.cat_lookup_smooth,
                            &mono_cstr,
                            self.gain_penalty,
                            self.extra_trees,
                            self.lookahead_alpha,
                            self.adaptive_leaf_experts
                                || self.leaf_linear
                                || self.cat_lookup_smooth > 0.0,
                            self.sparse_oblique_splits,
                            self.interval_splits,
                            root_anchor_feature,
                            CatPairConfig {
                                enabled: self.jit_catpair_enabled,
                                top_k_cat: self.jit_catpair_top_k,
                                k_buckets: self.jit_catpair_k_buckets,
                                min_node_rows: self.jit_catpair_min_node_rows,
                                max_node_depth: self.jit_catpair_max_node_depth,
                                gain_margin: self.jit_catpair_gain_margin,
                            },
                        ),
                    };

                    // Rank/sign split mode: always refit on original gradients to reset
                    // leaves to Newton scale before any honest_tau blend.
                    if split_mode_active {
                        tree.refit_leaves_l1(
                            binned,
                            &gradients,
                            &hessians,
                            build_indices,
                            round_lambda,
                            self.l1_reg,
                        );
                    }

                    if use_bayesian && self.honest {
                        self.compute_gradients_hessians(
                            y,
                            &predictions,
                            &mut gradients,
                            &mut hessians,
                        );
                        Self::apply_sample_weights(sample_weight, &mut gradients, &mut hessians);
                    }

                    if self.honest {
                        tree.refit_leaves_robust(
                            binned,
                            &gradients,
                            &hessians,
                            &estimation_indices,
                            round_lambda,
                            self.honest_tau,
                            self.leaf_trim_pct,
                            self.leaf_median,
                            self.leaf_median_blend,
                            self.leaf_mad_clip,
                            self.leaf_adaptive_blend_kappa,
                        );

                        if self.cat_lookup_smooth > 0.0
                            && !self.adaptive_leaf_experts
                            && grow != "oblivious"
                        {
                            tree.refit_cat_lookups(
                                binned,
                                &gradients,
                                &hessians,
                                &estimation_indices,
                                round_lambda,
                                self.cat_lookup_smooth,
                                self.min_child_weight,
                            );
                        }
                    } else if self.leaf_trim_pct > 0.0
                        || self.leaf_median
                        || self.leaf_median_blend > 0.0
                        || self.leaf_mad_clip > 0.0
                        || self.leaf_adaptive_blend_kappa > 0.0
                    {
                        // Non-honest + trim: post-build robust refit on structure indices.
                        tree.refit_leaves_robust(
                            binned,
                            &gradients,
                            &hessians,
                            build_indices,
                            round_lambda,
                            0.0,
                            self.leaf_trim_pct,
                            self.leaf_median,
                            self.leaf_median_blend,
                            self.leaf_mad_clip,
                            self.leaf_adaptive_blend_kappa,
                        );
                    }

                    if self.adaptive_leaf_experts || self.cat_lookup_smooth > 0.0 {
                        if self.adaptive_leaf_experts {
                            let lookup_indices = if self.honest && !estimation_indices.is_empty() {
                                &estimation_indices
                            } else {
                                build_indices
                            };
                            tree.install_best_lookups_with_config(
                                binned,
                                &gradients,
                                &hessians,
                                lookup_indices,
                                round_lambda,
                                self.gamma,
                                self.min_child_weight,
                                self.cat_lookup_smooth,
                                self.adaptive_cat_lookup_smooth,
                                cat_tuple_cfg.as_ref(),
                            );
                        } else if grow == "oblivious" {
                            // Oblivious trees never install lookups during structure building.
                            tree.install_cat_lookups(
                                binned,
                                &gradients,
                                &hessians,
                                build_indices,
                                round_lambda,
                                self.gamma,
                                self.min_child_weight,
                                self.cat_lookup_smooth,
                            );
                        }
                    }
                    if use_expert_admission_tree && !expert_calibration_indices.is_empty() {
                        self.apply_expert_leaf_admission(
                            &mut tree,
                            binned,
                            &gradients,
                            &hessians,
                            &predictions,
                            y,
                            build_indices,
                            &expert_calibration_indices,
                        );
                    }
                    if n_sub > 1 {
                        for v in tree.values.iter_mut() {
                            *v *= sub_scale;
                        }
                        tree.scale_ramp_slopes(sub_scale);
                        tree.scale_cat_lookups(sub_scale);
                    }

                    if self.lr_decay < 1.0 && n_rounds > 1 {
                        let factor =
                            1.0 - (1.0 - self.lr_decay) * (round as f64) / (n_rounds as f64 - 1.0);
                        for v in tree.values.iter_mut() {
                            *v *= factor;
                        }
                        tree.scale_ramp_slopes(factor);
                        tree.scale_cat_lookups(factor);
                    }
                    if use_arg && arg_lr_mult < 1.0 {
                        for v in tree.values.iter_mut() {
                            *v *= arg_lr_mult;
                        }
                        tree.scale_ramp_slopes(arg_lr_mult);
                        tree.scale_cat_lookups(arg_lr_mult);
                    }

                    // max_delta_step: clip leaf values
                    if self.max_delta_step > 0.0 {
                        let mds = self.max_delta_step;
                        for v in tree.values.iter_mut() {
                            *v = v.clamp(-mds, mds);
                        }
                    }

                    // DART: scale new tree (XGBoost-style: new_w = dropped_wsum / (nd + 1))
                    let dart_new_w = if !dart_dropped.is_empty() {
                        let nd = dart_dropped.len() as f64;
                        dart_dropped_wsum / (nd + 1.0)
                    } else {
                        1.0
                    };
                    let effective_lr = self.learning_rate * dart_new_w;
                    self.apply_hierarchical_shrinkage(&mut tree);
                    if use_loo && !self.honest {
                        let loo_indices = &indices;
                        if posterior_tau > 0.0 {
                            tree.posterior_shrink_leaves(posterior_tau);
                        }
                        tree.add_predictions_loo(
                            binned,
                            &mut predictions,
                            effective_lr,
                            &gradients,
                            &hessians,
                            round_lambda,
                            loo_indices,
                            posterior_tau,
                        );
                    } else {
                        if posterior_tau > 0.0 {
                            tree.posterior_shrink_leaves(posterior_tau);
                        }
                        tree.add_predictions_binned(binned, &mut predictions, effective_lr);
                    }

                    // Leaf correction: recompute gradients and refit leaf values
                    if self.leaf_correction > 0 {
                        let lr = effective_lr;
                        let mut corr_grads = vec![0.0f64; n_rows];
                        let mut corr_hess = vec![0.0f64; n_rows];
                        for _step in 0..self.leaf_correction {
                            self.compute_gradients_hessians(
                                y,
                                &predictions,
                                &mut corr_grads,
                                &mut corr_hess,
                            );
                            Self::apply_sample_weights(
                                sample_weight,
                                &mut corr_grads,
                                &mut corr_hess,
                            );
                            tree.add_predictions_binned(binned, &mut predictions, -lr);
                            tree.refit_leaves_l1(
                                binned,
                                &corr_grads,
                                &corr_hess,
                                &all_indices,
                                round_lambda,
                                self.l1_reg,
                            );
                            if self.adaptive_leaf_experts && self.cat_lookup_smooth > 0.0 {
                                tree.install_best_lookups_with_config(
                                    binned,
                                    &corr_grads,
                                    &corr_hess,
                                    &all_indices,
                                    round_lambda,
                                    self.gamma,
                                    self.min_child_weight,
                                    self.cat_lookup_smooth,
                                    self.adaptive_cat_lookup_smooth,
                                    cat_tuple_cfg.as_ref(),
                                );
                            }
                            tree.add_predictions_binned(binned, &mut predictions, lr);
                        }
                    }

                    let tree_uses_self_score = if let Some(score_idx) = self_score_feature {
                        let edges = binned.bin_edges[score_idx].clone();
                        tree.rewrite_feature_as_self_score(score_idx, &edges)
                    } else {
                        false
                    };

                    if eval_active {
                        let (eval_bins, _eval_y, en, _, eval_cll_bins) =
                            eval_data.as_ref().unwrap();
                        let en = *en;
                        if tree_uses_self_score {
                            tree.add_predictions_binned_raw_with_score(
                                eval_bins,
                                en,
                                &mut eval_preds[..en],
                                effective_lr,
                                eval_cll_bins,
                            );
                        } else {
                            tree.add_predictions_binned_raw(
                                eval_bins,
                                en,
                                &mut eval_preds[..en],
                                effective_lr,
                                eval_cll_bins,
                            );
                        }
                    }

                    // Cache per-tree predictions for fast DART drop/restore
                    if dart_enabled {
                        let mut tp = vec![0.0f64; n_rows];
                        for i in 0..n_rows {
                            tp[i] = tree.predict_binned(binned, i);
                        }
                        dart_tree_preds.push(tp);
                    }
                    // Intra-round residual feedback for the next sibling tree.
                    // alpha=1.0 means exact feedback: recompute gradients at the
                    // current prediction after this sibling's contribution. Partial
                    // alpha blends old and feedback gradients for conservative use.
                    if use_ncl && _sub_idx + 1 < n_sub {
                        let alpha = self.ncl_lambda.clamp(0.0, 1.0);
                        let mut fb_grad = vec![0.0f64; n_rows];
                        let mut fb_hess = vec![0.0f64; n_rows];
                        self.compute_gradients_hessians(
                            y,
                            &predictions,
                            &mut fb_grad,
                            &mut fb_hess,
                        );
                        Self::apply_sample_weights(sample_weight, &mut fb_grad, &mut fb_hess);
                        for i in 0..n_rows {
                            gradients[i] = (1.0 - alpha) * gradients[i] + alpha * fb_grad[i];
                            hessians[i] = (1.0 - alpha) * hessians[i] + alpha * fb_hess[i];
                        }
                    }

                    self.dart_tree_weights.push(dart_new_w);
                    self.apply_eblp(&mut tree);
                    self.apply_hss(&mut tree);
                    self.apply_scs(&mut tree, binned, &gradients, n_rows);
                    self.apply_newton_trust_region(&mut tree);
                    if use_ordered {
                        if let Some(ref mask) = seq_in_sample {
                            for i in 0..n_rows {
                                if !bitvec_test(mask, i) {
                                    let tp = tree.predict_binned(binned, i);
                                    let contribution = effective_lr * tp;
                                    oob_pred_sum[i] += contribution;
                                    oob_pred_sum_sq[i] += contribution * contribution;
                                    oob_count[i] += 1;
                                }
                            }
                        }
                    }
                    if use_sibling_block_correction && use_ordered {
                        round_in_sample_masks.push(seq_in_sample.clone());
                    }
                    self.trees.push(tree);
                }
            }

            // ── DART: restore dropped trees with rescaled weights (using cache) ──
            if !dart_dropped.is_empty() {
                let nd = dart_dropped.len() as f64;
                let scale_old = nd / (nd + 1.0);
                let lr = self.learning_rate;
                for &j in &dart_dropped {
                    self.dart_tree_weights[j] *= scale_old;
                    let w = self.dart_tree_weights[j];
                    let cached = &dart_tree_preds[j];
                    let lrw = lr * w;
                    if n_rows >= 4096 {
                        predictions
                            .par_chunks_mut(1024)
                            .enumerate()
                            .for_each(|(ci, chunk)| {
                                let start = ci * 1024;
                                for (jj, pred) in chunk.iter_mut().enumerate() {
                                    *pred += lrw * cached[start + jj];
                                }
                            });
                    } else {
                        for i in 0..n_rows {
                            predictions[i] += lrw * cached[i];
                        }
                    }
                }
            }

            if use_sibling_block_correction && self.trees.len() > trees_before_round + 1 {
                self.apply_sibling_block_correction(
                    binned,
                    y,
                    n_rows,
                    trees_before_round,
                    &round_prediction_base,
                    eval_data.as_ref(),
                    &round_eval_base,
                    if use_ordered {
                        Some(round_in_sample_masks.as_slice())
                    } else {
                        None
                    },
                    &mut predictions,
                    &mut eval_preds,
                    &mut oob_pred_sum,
                    &mut oob_pred_sum_sq,
                );
            }

            // ── Progressive interaction unlocking ──────────────────────────
            if interaction_rescore_interval > 0
                && round > 0
                && round % interaction_rescore_interval == 0
            {
                let n_trees = self.trees.len();
                let start = if n_trees > interaction_rescore_interval {
                    n_trees - interaction_rescore_interval
                } else {
                    0
                };
                let recent_trees = &self.trees[start..];

                // Score co-occurring pairs from recent trees
                let mut pair_counts: std::collections::HashMap<(u32, u32), f64> =
                    std::collections::HashMap::new();
                let mut feat_counts: std::collections::HashMap<u32, usize> =
                    std::collections::HashMap::new();
                for tree in recent_trees {
                    for &f in &tree.split_features {
                        if f != u32::MAX {
                            *feat_counts.entry(f).or_insert(0) += 1;
                        }
                    }
                    for (a, b) in tree.extract_split_cooccurrences(n_features_original) {
                        *pair_counts.entry((a, b)).or_insert(0.0) += 1.0;
                    }
                }

                // Filter to numeric pairs not already unlocked
                let existing: std::collections::HashSet<(usize, usize)> =
                    self.numeric_interaction_pairs.iter().copied().collect();
                let numeric_set: std::collections::HashSet<usize> = (0..n_features_original)
                    .filter(|&i| i >= self.cat_features.len() || !self.cat_features[i])
                    .collect();
                let max_total = if self.max_interaction_features > 0 {
                    self.max_interaction_features
                } else {
                    20
                };
                let remaining_budget =
                    max_total.saturating_sub(self.numeric_interaction_pairs.len());
                if remaining_budget > 0 {
                    let mut new_pairs: Vec<((usize, usize), f64)> = pair_counts
                        .into_iter()
                        .filter(|((a, b), _)| {
                            let au = *a as usize;
                            let bu = *b as usize;
                            numeric_set.contains(&au)
                                && numeric_set.contains(&bu)
                                && !existing.contains(&(au, bu))
                        })
                        .map(|((a, b), count)| {
                            let imp_a = *feat_counts.get(&a).unwrap_or(&0) as f64;
                            let imp_b = *feat_counts.get(&b).unwrap_or(&0) as f64;
                            ((a as usize, b as usize), count * (imp_a * imp_b).sqrt())
                        })
                        .collect();
                    new_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    new_pairs.truncate(remaining_budget.min(5)); // unlock at most 5 new pairs per interval

                    if !new_pairs.is_empty() {
                        let new_selected: Vec<(usize, usize)> =
                            new_pairs.iter().map(|(p, _)| *p).collect();
                        // Compute product columns for training data
                        let product_cols: Vec<Vec<f64>> = new_selected
                            .iter()
                            .map(|&(fi, fj)| {
                                (0..n_rows)
                                    .map(|row| {
                                        let vi = x_data_raw[row * n_features_original + fi];
                                        let vj = x_data_raw[row * n_features_original + fj];
                                        if vi.is_nan() || vj.is_nan() {
                                            f64::NAN
                                        } else {
                                            vi * vj
                                        }
                                    })
                                    .collect()
                            })
                            .collect();
                        let int_start = binned.n_features;
                        binned.add_ots_features(&product_cols, effective_bins);
                        let new_edges: Vec<Vec<f64>> = binned.bin_edges[int_start..].to_vec();

                        // Expand eval data if present
                        if let Some((ref mut eval_bins, _, en, ref eval_raw, _)) = eval_data {
                            if !eval_raw.is_empty() {
                                let eval_products: Vec<Vec<f64>> = new_selected
                                    .iter()
                                    .map(|&(fi, fj)| {
                                        (0..*en)
                                            .map(|row| {
                                                let vi = eval_raw[row * n_features_original + fi];
                                                let vj = eval_raw[row * n_features_original + fj];
                                                if vi.is_nan() || vj.is_nan() {
                                                    f64::NAN
                                                } else {
                                                    vi * vj
                                                }
                                            })
                                            .collect()
                                    })
                                    .collect();
                                BinnedData::add_ots_features_with_edges(
                                    eval_bins,
                                    *en,
                                    &eval_products,
                                    &new_edges,
                                );
                            }
                        }

                        // Update state
                        self.numeric_interaction_pairs
                            .extend_from_slice(&new_selected);
                        self.numeric_interaction_edges.extend_from_slice(&new_edges);
                        n_feat = binned.n_features;
                        mono_cstr.resize(n_feat, 0);
                    }
                }
            }

            // Early stopping check (after all sub-trees in this round)
            if eval_active {
                let (_, eval_y, en, _, _) = eval_data.as_ref().unwrap();
                let eval_loss = self.compute_eval_loss(eval_y, &eval_preds, *en);
                // PASA: record val loss history for downstream plateau averaging
                self.val_losses.push(eval_loss);
                let improved = eval_loss < best_eval_loss;
                if improved {
                    best_eval_loss = eval_loss;
                    best_round = self.trees.len();
                    rounds_without_improvement = 0;
                } else {
                    rounds_without_improvement += 1;
                }

                let should_stop = early_stop_active
                    && !improved
                    && rounds_without_improvement >= self.early_stopping_rounds;
                if self.verbose > 0
                    && (round == 0 || (round + 1) % self.verbose == 0 || should_stop)
                {
                    eprintln!(
                        "[{}]\tvalid-loss={:.6}\tbest={:.6}\tbest_trees={}",
                        round + 1,
                        eval_loss,
                        best_eval_loss,
                        best_round
                    );
                }

                if should_stop {
                    if self.verbose > 0 {
                        eprintln!(
                            "early stopping at round {}, best_trees={}",
                            round + 1,
                            best_round
                        );
                    }
                    break;
                }
            }

            // ── ARG: Auto-Regularization via OOB signal ──
            // Compute OOB loss using variance-weighted OOB predictions; if it
            // plateaus for `arg_patience` rounds, shrink arg_lr_mult so later
            // trees contribute less (RF-like averaging emerges on noise-heavy
            // small-N data). No effect on GBDT-favoring datasets where OOB
            // loss keeps decreasing.
            if use_arg && self.trees.len() >= 5 {
                let n_trees_f = self.trees.len() as f64;
                let mut oob_loss_sum = 0.0f64;
                let mut oob_n = 0usize;
                match self.task.as_str() {
                    "regression" => {
                        for i in 0..n_rows {
                            if oob_count[i] >= 2 {
                                let oob_pred = self.base_score
                                    + oob_pred_sum[i] * (n_trees_f / oob_count[i] as f64);
                                let d = oob_pred - y[i];
                                oob_loss_sum += d * d;
                                oob_n += 1;
                            }
                        }
                    }
                    "binary" | "rank" => {
                        for i in 0..n_rows {
                            if oob_count[i] >= 2 {
                                let z = self.base_score
                                    + oob_pred_sum[i] * (n_trees_f / oob_count[i] as f64);
                                let p = (1.0 / (1.0 + (-z).exp())).clamp(1e-15, 1.0 - 1e-15);
                                oob_loss_sum -= y[i] * p.ln() + (1.0 - y[i]) * (1.0 - p).ln();
                                oob_n += 1;
                            }
                        }
                    }
                    _ => {
                        for i in 0..n_rows {
                            if oob_count[i] >= 2 {
                                let oob_pred = self.base_score
                                    + oob_pred_sum[i] * (n_trees_f / oob_count[i] as f64);
                                let d = oob_pred - y[i];
                                oob_loss_sum += d * d;
                                oob_n += 1;
                            }
                        }
                    }
                }
                if oob_n >= 20 {
                    let oob_loss = oob_loss_sum / oob_n as f64;
                    if oob_loss < arg_best_oob_loss * 0.999 {
                        arg_best_oob_loss = oob_loss;
                        arg_rounds_no_improve = 0;
                    } else {
                        arg_rounds_no_improve += 1;
                        if arg_rounds_no_improve >= arg_patience && arg_lr_mult > arg_min_mult {
                            arg_lr_mult = (arg_lr_mult * arg_decay).max(arg_min_mult);
                            arg_rounds_no_improve = 0;
                        }
                    }
                }
            }

            // Cyclic refinement with alpha blending
            if self.refine_every > 0 && (round + 1) % self.refine_every == 0 {
                let alpha = self.refine_alpha;
                for t_idx in 0..self.trees.len() {
                    for i in 0..n_rows {
                        predictions[i] -=
                            self.learning_rate * self.trees[t_idx].predict_binned(binned, i);
                    }
                    self.compute_gradients_hessians(y, &predictions, &mut gradients, &mut hessians);
                    Self::apply_sample_weights(sample_weight, &mut gradients, &mut hessians);
                    let old_values = self.trees[t_idx].values.clone();
                    self.trees[t_idx].refit_leaves_l1(
                        binned,
                        &gradients,
                        &hessians,
                        &all_indices,
                        self.lambda_reg,
                        self.l1_reg,
                    );
                    if posterior_tau > 0.0 {
                        self.trees[t_idx].posterior_shrink_leaves(posterior_tau);
                    }
                    if alpha < 1.0 {
                        for j in 0..self.trees[t_idx].values.len() {
                            let w_new = self.trees[t_idx].values[j];
                            self.trees[t_idx].values[j] =
                                old_values[j] + alpha * (w_new - old_values[j]);
                        }
                    }
                    for i in 0..n_rows {
                        predictions[i] +=
                            self.learning_rate * self.trees[t_idx].predict_binned(binned, i);
                    }
                }
            }

            // Track this round's trees for next round's orthogonalization
            if use_ortho {
                prev_round_trees_start = trees_before_round;
                prev_round_trees_end = self.trees.len();
            }
        }
        self.rank_mix_alpha = configured_rank_mix_alpha;
        self.binary_focus_gamma = configured_binary_focus_gamma;

        // PASA: record best_round for plateau averaging / truncated predict
        self.best_round = best_round;
        // Trim to best round if early stopping triggered (unless keep_all_trees).
        // With keep_all_trees=True we retain the plateau buffer for averaging.
        if early_stop_active && eval_active && best_round < self.trees.len() && !self.keep_all_trees
        {
            self.trees.truncate(best_round);
            if !self.tree_in_sample.is_empty() {
                self.tree_in_sample.truncate(best_round);
            }
        }

        // ── Phase 2: Interleaved leaf splitting + refinement ─────────────
        for _ in 0..self.n_leaf_splits {
            self.leaf_split_pass(binned, y, n_rows);
            self.refine_global(binned, y, n_rows, 1); // recalibrate after each split
        }

        // ── Phase 3: Final leaf optimization ─────────────────────────────
        if self.n_refine > 0 {
            self.refine_global(binned, y, n_rows, self.n_refine);
            self.prune_similar_leaves();
        }

        self.fit_global_cat_offsets(
            binned,
            y,
            n_rows,
            x_data_raw,
            n_features_original,
            sample_weight,
        );
    }
}
