//! K-class softmax training and shared-tree multi-output helpers.
//!
//! - `fit_multiclass` is the K-class analogue of `fit_single` (in
//!   `super::training`). Per round it can build either:
//!     * K independent trees (one per class), or
//!     * a single shared-structure multi-output tree fitted to all K.
//! - `fill_multiclass_grad_hess_from_probs` — softmax gradient/hessian
//!   computation per row × class.
//! - `build_ordered_multioutput_round` — CatBoost-style ordered boosting
//!   adapted to K classes (shadow buckets for unbiased gradients).
//! - `compute_multiclass_coupled_node_values` /
//!   `compute_multiclass_guided_lookup_choices` /
//!   `compute_multiclass_joint_guided_lookups` — coupled-softmax leaf
//!   updates and CLL (Category-Lookup-Leaf) installation for shared
//!   multi-output trees.
//!
//! Per-tree adjustments (HSS / EBLP / NTR / etc.) are reused from
//! `super::internals`; refinement of K-class leaves lives in
//! `super::refine`.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::RngExt;
use rand::SeedableRng;
use rayon::prelude::*;

use super::GTBoostModel;
use crate::helpers::{bitvec_new, bitvec_set, bitvec_test, solve_spd};
use crate::tree::{
    bitmask_test, BinnedData, CatLookup, DecisionTree, GuidedCatChoice, MISSING_BIN,
};

impl GTBoostModel {
    pub(super) fn fill_multiclass_grad_hess_from_probs(
        y: &[f64],
        probs: &[f64],
        n_rows: usize,
        n_classes: usize,
        label_smooth: f64,
        class_weights: &[f64],
        all_grads_flat: &mut [f64],
        all_hess_flat: &mut [f64],
    ) {
        let inv_k = 1.0 / n_classes as f64;
        for k in 0..n_classes {
            let base = k * n_rows;
            for i in 0..n_rows {
                let yi = y[i] as usize;
                let hard_label = if yi == k { 1.0 } else { 0.0 };
                let label = if label_smooth > 0.0 {
                    (1.0 - label_smooth) * hard_label + label_smooth * inv_k
                } else {
                    hard_label
                };
                let cw = if class_weights.len() == n_classes && yi < class_weights.len() {
                    class_weights[yi]
                } else {
                    1.0
                };
                let p = probs[i * n_classes + k];
                all_grads_flat[base + i] = cw * (p - label);
                all_hess_flat[base + i] = (cw * p * (1.0 - p)).max(1e-16);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_ordered_multioutput_round(
        &mut self,
        binned: &BinnedData,
        y: &[f64],
        n_rows: usize,
        n_features: usize,
        n_classes: usize,
        round: usize,
        n_rounds: usize,
        probs: &[f64],
        all_grads_flat: &mut [f64],
        all_hess_flat: &mut [f64],
        ls: f64,
        sub_scale: f64,
        posterior_tau: f64,
        predictions: &mut [f64],
        eval_data: &Option<(Vec<u16>, Vec<f64>, usize, Vec<f64>, Vec<u16>)>,
        eval_preds: &mut [f64],
        rng: &mut StdRng,
        ordered_bucket_rows: &[Vec<u32>],
        ordered_train_pools: &[Vec<u32>],
        ordered_shadow_preds: &mut [f64],
    ) {
        let n_sub = self.n_trees_per_round;
        let ordered_buckets = ordered_bucket_rows.len();
        let ordered_lr = self.learning_rate / ordered_buckets as f64;
        let shadow_lr = self.learning_rate;
        let inv_k = 1.0 / n_classes as f64;
        let cat_tuple_cfg = self.cat_tuple_config(binned);

        for k in 0..n_classes {
            let base = k * n_rows;
            for i in 0..n_rows {
                let hard_label = if y[i] as usize == k { 1.0 } else { 0.0 };
                let label = if ls > 0.0 {
                    (1.0 - ls) * hard_label + ls * inv_k
                } else {
                    hard_label
                };
                all_grads_flat[base + i] = probs[i * n_classes + k] - label;
                all_hess_flat[base + i] =
                    (probs[i * n_classes + k] * (1.0 - probs[i * n_classes + k])).max(1e-16);
            }
        }

        let total_subtrees = ordered_buckets * n_sub;
        let mut shared_trees: Vec<DecisionTree> = Vec::with_capacity(total_subtrees);
        let mut sub_indices_vec: Vec<Vec<u32>> = Vec::with_capacity(total_subtrees);
        let mut excluded_bucket_vec: Vec<usize> = Vec::with_capacity(total_subtrees);

        for excluded in 0..ordered_buckets {
            let planned_subsamples =
                self.round_subsamples_from_pool(rng, &ordered_train_pools[excluded], n_sub);
            for sub_idx in 0..n_sub {
                let indices = planned_subsamples[sub_idx].clone();
                let feature_mask =
                    self.make_feature_mask_for_subtree(rng, n_features, round, sub_idx);
                let tree_seed: u64 = rng.random();
                let mo_extra_trees = self.extra_trees && self.prob_avg;
                let shared_tree = if self.grow_policy == "oblivious" {
                    DecisionTree::build_oblivious_multi(
                        binned,
                        all_grads_flat,
                        all_hess_flat,
                        probs,
                        n_classes,
                        &indices,
                        self.lambda_reg,
                        self.gamma,
                        self.max_depth,
                        self.min_child_weight,
                        &feature_mask,
                        self.gain_penalty,
                        mo_extra_trees,
                        tree_seed,
                        self.multiclass_coupled_leaves,
                    )
                } else {
                    DecisionTree::build_depthwise_multi(
                        binned,
                        all_grads_flat,
                        all_hess_flat,
                        probs,
                        n_classes,
                        &indices,
                        self.lambda_reg,
                        self.gamma,
                        self.max_depth,
                        self.min_child_weight,
                        &feature_mask,
                        self.colsample_bylevel,
                        tree_seed,
                        self.random_strength,
                        self.cat_smooth,
                        self.gain_penalty,
                        mo_extra_trees,
                        self.multiclass_coupled_leaves,
                    )
                };
                shared_trees.push(shared_tree);
                sub_indices_vec.push(indices);
                excluded_bucket_vec.push(excluded);
            }
        }

        let use_coupled_mc_leaves = self.multiclass_coupled_leaves;
        let mut coupled_node_values: Vec<Vec<f64>> = Vec::new();
        let mut coupled_node_counts: Vec<Vec<u32>> = Vec::new();
        if use_coupled_mc_leaves {
            coupled_node_values.reserve(total_subtrees);
            coupled_node_counts.reserve(total_subtrees);
            for tree_idx in 0..total_subtrees {
                let (vals, counts) = self.compute_multiclass_coupled_node_values(
                    &shared_trees[tree_idx],
                    binned,
                    y,
                    probs,
                    &sub_indices_vec[tree_idx],
                    n_classes,
                    ls,
                );
                coupled_node_values.push(vals);
                coupled_node_counts.push(counts);
            }
        }

        let mut gradients = vec![0.0f64; n_rows];
        let mut hessians = vec![0.0f64; n_rows];

        for k in 0..n_classes {
            let class_base = k * n_rows;
            for i in 0..n_rows {
                gradients[i] = all_grads_flat[class_base + i];
                hessians[i] = all_hess_flat[class_base + i];
            }

            for tree_idx in 0..total_subtrees {
                let mut tree = shared_trees[tree_idx].clone();
                let indices = &sub_indices_vec[tree_idx];

                if self.leaf_trim_pct > 0.0
                    || self.leaf_median
                    || self.leaf_median_blend > 0.0
                    || self.leaf_mad_clip > 0.0
                    || self.leaf_adaptive_blend_kappa > 0.0
                {
                    tree.refit_leaves_robust(
                        binned,
                        &gradients,
                        &hessians,
                        indices,
                        self.lambda_reg,
                        0.0,
                        self.leaf_trim_pct,
                        self.leaf_median,
                        self.leaf_median_blend,
                        self.leaf_mad_clip,
                        self.leaf_adaptive_blend_kappa,
                    );
                } else {
                    tree.refit_leaves(binned, &gradients, &hessians, indices, self.lambda_reg);
                }

                if use_coupled_mc_leaves {
                    let vals = &coupled_node_values[tree_idx];
                    let counts = &coupled_node_counts[tree_idx];
                    for node in 0..tree.values.len() {
                        if counts[node] > 0 {
                            tree.values[node] = vals[node * n_classes + k];
                        }
                    }
                }

                if self.adaptive_leaf_experts || self.cat_lookup_smooth > 0.0 {
                    if self.adaptive_leaf_experts {
                        tree.install_best_lookups_with_config(
                            binned,
                            &gradients,
                            &hessians,
                            indices,
                            self.lambda_reg,
                            self.gamma,
                            self.min_child_weight,
                            self.cat_lookup_smooth,
                            self.adaptive_cat_lookup_smooth,
                            cat_tuple_cfg.as_ref(),
                        );
                    } else {
                        tree.install_cat_lookups(
                            binned,
                            &gradients,
                            &hessians,
                            indices,
                            self.lambda_reg,
                            self.gamma,
                            self.min_child_weight,
                            self.cat_lookup_smooth,
                        );
                    }
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
                self.apply_hierarchical_shrinkage(&mut tree);

                let excluded = excluded_bucket_vec[tree_idx];
                for i in 0..n_rows {
                    let pred = tree.predict_binned(binned, i);
                    predictions[i * n_classes + k] += ordered_lr * pred;
                    ordered_shadow_preds[(excluded * n_rows + i) * n_classes + k] +=
                        shadow_lr * pred;
                }

                if !eval_preds.is_empty() {
                    let (eval_bins, _, en, _, eval_cll_bins) = eval_data.as_ref().unwrap();
                    let en = *en;
                    for i in 0..en {
                        eval_preds[i * n_classes + k] +=
                            ordered_lr * tree.predict_binned_raw(eval_bins, en, i, eval_cll_bins);
                    }
                }

                self.dart_tree_weights.push(1.0);
                self.apply_eblp(&mut tree);
                self.apply_hss(&mut tree);
                self.apply_scs(&mut tree, binned, &gradients, n_rows);
                self.apply_newton_trust_region(&mut tree);
                self.trees.push(tree);
            }
        }
    }

    /// Fit for multiclass classification (native softmax, K trees per round).
    pub(super) fn fit_multiclass(
        &mut self,
        binned: &mut BinnedData,
        y: &[f64],
        n_rows: usize,
        n_features: usize,
        n_rounds: usize,
        eval_data: &mut Option<(Vec<u16>, Vec<f64>, usize, Vec<f64>, Vec<u16>)>,
    ) {
        let n_classes = y.iter().map(|&v| v as usize).max().unwrap_or(0) + 1;
        self.n_classes = n_classes;

        let mut predictions = vec![0.0f64; n_rows * n_classes];
        if self.class_base_scores.len() == n_classes {
            for i in 0..n_rows {
                let base = i * n_classes;
                predictions[base..base + n_classes].copy_from_slice(&self.class_base_scores);
            }
        }
        let mut probs = vec![0.0f64; n_rows * n_classes];
        let mut gradients = vec![0.0f64; n_rows];
        let mut hessians = vec![0.0f64; n_rows];
        // Multi-output: pre-allocated buffers for all K classes' gradients/hessians
        let mut all_grads_flat = if self.multi_output_tree {
            vec![0.0f64; n_classes * n_rows]
        } else {
            Vec::new()
        };
        let mut all_hess_flat = if self.multi_output_tree {
            vec![0.0f64; n_classes * n_rows]
        } else {
            Vec::new()
        };

        self.multiclass_trees_per_class = self.n_trees_per_round.max(1);
        self.multiclass_tree_lr_scale = 1.0;
        self.trees.reserve(n_rounds * n_classes);
        self.dart_tree_weights.clear();
        let mut rng = StdRng::seed_from_u64(self.seed);
        let grow_policy = self.grow_policy.clone();
        let grow = grow_policy.as_str();
        let max_leaves = if self.max_leaves > 0 {
            self.max_leaves
        } else {
            1 << self.max_depth
        };
        let all_indices: Vec<u32> = (0..n_rows as u32).collect();

        // Early stopping state
        let es_active = self.early_stopping_rounds > 0 && eval_data.is_some();
        let mut eval_preds: Vec<f64> = if let Some((_, _, en, _, _)) = eval_data {
            let mut v = vec![0.0f64; *en * n_classes];
            if self.class_base_scores.len() == n_classes {
                for i in 0..*en {
                    let base = i * n_classes;
                    v[base..base + n_classes].copy_from_slice(&self.class_base_scores);
                }
            }
            v
        } else {
            Vec::new()
        };
        let mut best_eval_loss = f64::MAX;
        let mut best_round = 0usize;
        let mut rounds_without_improvement = 0usize;

        // Pad monotone constraints to n_features (0 for OTS/interaction features)
        let mono_cstr: Vec<i8> = {
            let mut v = self.monotone_constraints.clone();
            v.resize(n_features, 0);
            v
        };

        // ── Phase 1: Sequential boosting (K × ntp trees per round) ─────────
        let n_sub = self.n_trees_per_round;
        let sub_scale = if n_sub > 1 { 1.0 / n_sub as f64 } else { 1.0 };
        let posterior_tau = self.posterior_leaf_tau();
        let cat_tuple_cfg = self.cat_tuple_config(binned);
        let use_ordered_multi = self.ordered_boost
            && self.multi_output_tree
            && !self.prob_avg
            && !self.honest
            && n_rows >= 2;
        let ordered_bucket_count = if use_ordered_multi {
            self.ordered_n_buckets.min(n_rows)
        } else {
            1
        };
        self.multiclass_trees_per_class = if self.multi_output_tree {
            n_sub * ordered_bucket_count
        } else {
            n_sub
        };
        self.multiclass_tree_lr_scale = if use_ordered_multi {
            1.0 / ordered_bucket_count as f64
        } else {
            1.0
        };
        let ordered_bucket_rows: Vec<Vec<u32>> = if use_ordered_multi {
            let mut bucket_of = vec![0usize; n_rows];
            let mut perm: Vec<usize> = (0..n_rows).collect();
            let mut bucket_rng = StdRng::seed_from_u64(self.seed.wrapping_add(9001));
            perm.shuffle(&mut bucket_rng);
            for (rank, &row) in perm.iter().enumerate() {
                bucket_of[row] = rank % ordered_bucket_count;
            }
            let mut rows = vec![Vec::new(); ordered_bucket_count];
            for (row, &bucket) in bucket_of.iter().enumerate() {
                rows[bucket].push(row as u32);
            }
            rows
        } else {
            Vec::new()
        };
        let ordered_bucket_of: Vec<usize> = if use_ordered_multi {
            let mut bucket_of = vec![0usize; n_rows];
            for (bucket, rows) in ordered_bucket_rows.iter().enumerate() {
                for &row in rows {
                    bucket_of[row as usize] = bucket;
                }
            }
            bucket_of
        } else {
            Vec::new()
        };
        let ordered_train_pools: Vec<Vec<u32>> = if use_ordered_multi {
            let mut pools = Vec::with_capacity(ordered_bucket_count);
            for excluded in 0..ordered_bucket_count {
                let mut pool = Vec::with_capacity(n_rows - ordered_bucket_rows[excluded].len());
                for bucket in 0..ordered_bucket_count {
                    if bucket != excluded {
                        pool.extend_from_slice(&ordered_bucket_rows[bucket]);
                    }
                }
                pools.push(pool);
            }
            pools
        } else {
            Vec::new()
        };
        let mut ordered_shadow_preds: Vec<f64> = if use_ordered_multi {
            let mut preds = vec![0.0f64; ordered_bucket_count * n_rows * n_classes];
            if self.class_base_scores.len() == n_classes {
                for bucket in 0..ordered_bucket_count {
                    for i in 0..n_rows {
                        let base = (bucket * n_rows + i) * n_classes;
                        preds[base..base + n_classes].copy_from_slice(&self.class_base_scores);
                    }
                }
            }
            preds
        } else {
            Vec::new()
        };
        let mut ordered_unbiased_preds: Vec<f64> = if use_ordered_multi {
            let mut preds = vec![0.0f64; n_rows * n_classes];
            if self.class_base_scores.len() == n_classes {
                for i in 0..n_rows {
                    let base = i * n_classes;
                    preds[base..base + n_classes].copy_from_slice(&self.class_base_scores);
                }
            }
            preds
        } else {
            Vec::new()
        };

        for round in 0..n_rounds {
            if self.diversity_penalty > 0.0 {
                self.refresh_feature_usage_ema(n_features);
            }
            if use_ordered_multi {
                for i in 0..n_rows {
                    let src_base = (ordered_bucket_of[i] * n_rows + i) * n_classes;
                    let dst_base = i * n_classes;
                    ordered_unbiased_preds[dst_base..dst_base + n_classes]
                        .copy_from_slice(&ordered_shadow_preds[src_base..src_base + n_classes]);
                }
                Self::compute_softmax_t(
                    &ordered_unbiased_preds,
                    &mut probs,
                    n_rows,
                    n_classes,
                    self.jensen_train_temp,
                );
            } else {
                Self::compute_softmax_t(
                    &predictions,
                    &mut probs,
                    n_rows,
                    n_classes,
                    self.jensen_train_temp,
                );
            }

            // Label smoothing: target_k = (1-ε)*one_hot_k + ε/K
            let ls = self.label_smooth;
            let inv_k = 1.0 / n_classes as f64;

            // ── Multi-output path: shared tree structure across all K classes ────
            // Splits are evaluated by summing gains across all K classes, then
            // leaf values are refitted per-class.
            if self.multi_output_tree {
                if use_ordered_multi {
                    self.build_ordered_multioutput_round(
                        binned,
                        y,
                        n_rows,
                        n_features,
                        n_classes,
                        round,
                        n_rounds,
                        &probs,
                        &mut all_grads_flat,
                        &mut all_hess_flat,
                        ls,
                        sub_scale,
                        posterior_tau,
                        &mut predictions,
                        eval_data,
                        &mut eval_preds,
                        &mut rng,
                        &ordered_bucket_rows,
                        &ordered_train_pools,
                        &mut ordered_shadow_preds,
                    );
                } else {
                    // 1. Pre-compute all K gradients/hessians.
                    Self::fill_multiclass_grad_hess_from_probs(
                        y,
                        &probs,
                        n_rows,
                        n_classes,
                        ls,
                        &self.class_weights,
                        &mut all_grads_flat,
                        &mut all_hess_flat,
                    );
                    if self.prob_avg && n_rounds == 1 {
                        for h in all_hess_flat.iter_mut() {
                            *h = 1.0;
                        }
                    }

                    // 2. Build n_sub shared trees using multi-output criterion.
                    // Probability-forest rounds build many independent subtrees;
                    // plan them deterministically, then build the tree bodies in
                    // parallel under the fit-level Rayon pool.
                    struct MultiSubPlan {
                        indices: Vec<u32>,
                        feature_mask: Vec<bool>,
                        structure_indices: Vec<u32>,
                        estimation_indices: Vec<u32>,
                        in_sample_mask: Vec<u64>,
                        tree_seed: u64,
                        sub_depth: usize,
                        sub_lambda: f64,
                        sub_oblivious: bool,
                        sub_extra_trees: bool,
                    }

                    let mut sub_plans: Vec<MultiSubPlan> = Vec::with_capacity(n_sub);
                    let planned_subsamples = self.round_subsamples(&mut rng, n_rows, n_sub);

                    for _sub_idx in 0..n_sub {
                        let indices = planned_subsamples[_sub_idx].clone();
                        let feature_mask = self
                            .make_feature_mask_for_subtree(&mut rng, n_features, round, _sub_idx);

                        let (structure_indices, estimation_indices, in_sample_mask) = if self.honest
                        {
                            if self.honest_fraction <= 0.0 && self.subsample_rate < 1.0 {
                                let mut in_sample = bitvec_new(n_rows);
                                for &idx in &indices {
                                    bitvec_set(&mut in_sample, idx as usize);
                                }
                                let complement: Vec<u32> = (0..n_rows as u32)
                                    .filter(|&i| !bitvec_test(&in_sample, i as usize))
                                    .collect();
                                let mask = in_sample;
                                (indices.clone(), complement, mask)
                            } else {
                                let mut shuffled = indices.clone();
                                shuffled.shuffle(&mut rng);
                                let frac = self.honest_fraction.clamp(0.1, 0.9);
                                let est_size = (shuffled.len() as f64 * frac).round() as usize;
                                let mid = shuffled.len() - est_size;
                                let si = shuffled[..mid].to_vec();
                                let ei = shuffled[mid..].to_vec();
                                (si, ei, Vec::new())
                            }
                        } else {
                            (indices.clone(), Vec::new(), Vec::new())
                        };
                        let tree_seed: u64 = rng.random();
                        let mo_extra_trees = self.extra_trees && self.prob_avg;
                        let (sub_depth, sub_lambda, sub_grow, sub_extra_trees) =
                            if self.hetero_trees && n_sub >= 3 {
                                let alt_grow = if grow == "oblivious" {
                                    "depthwise"
                                } else {
                                    "oblivious"
                                };
                                match _sub_idx % 3 {
                                    0 => (self.max_depth, self.lambda_reg, grow, mo_extra_trees),
                                    1 => (
                                        self.max_depth.saturating_sub(1).max(2),
                                        self.lambda_reg * 2.0,
                                        alt_grow,
                                        false,
                                    ),
                                    2 => (
                                        self.max_depth + 1,
                                        self.lambda_reg * 0.5,
                                        "depthwise",
                                        mo_extra_trees,
                                    ),
                                    _ => (self.max_depth, self.lambda_reg, grow, mo_extra_trees),
                                }
                            } else {
                                (self.max_depth, self.lambda_reg, grow, mo_extra_trees)
                            };

                        sub_plans.push(MultiSubPlan {
                            indices,
                            feature_mask,
                            structure_indices,
                            estimation_indices,
                            in_sample_mask,
                            tree_seed,
                            sub_depth,
                            sub_lambda,
                            sub_oblivious: sub_grow == "oblivious",
                            sub_extra_trees,
                        });
                    }

                    let build_shared_tree = |plan: &MultiSubPlan| {
                        let build_indices = if self.honest {
                            &plan.structure_indices
                        } else {
                            &plan.indices
                        };
                        if plan.sub_oblivious {
                            DecisionTree::build_oblivious_multi(
                                binned,
                                &all_grads_flat,
                                &all_hess_flat,
                                &probs,
                                n_classes,
                                build_indices,
                                plan.sub_lambda,
                                self.gamma,
                                plan.sub_depth,
                                self.min_child_weight,
                                &plan.feature_mask,
                                self.gain_penalty,
                                plan.sub_extra_trees,
                                plan.tree_seed,
                                self.multiclass_coupled_leaves && !self.prob_avg,
                            )
                        } else {
                            DecisionTree::build_depthwise_multi(
                                binned,
                                &all_grads_flat,
                                &all_hess_flat,
                                &probs,
                                n_classes,
                                build_indices,
                                plan.sub_lambda,
                                self.gamma,
                                plan.sub_depth,
                                self.min_child_weight,
                                &plan.feature_mask,
                                self.colsample_bylevel,
                                plan.tree_seed,
                                self.random_strength,
                                self.cat_smooth,
                                self.gain_penalty,
                                plan.sub_extra_trees,
                                self.multiclass_coupled_leaves && !self.prob_avg,
                            )
                        }
                    };

                    let parallel_sub_build =
                        self.prob_avg && n_rounds == 1 && self.ncl_lambda <= 0.0 && n_sub > 1;
                    let shared_trees: Vec<DecisionTree> = if parallel_sub_build {
                        sub_plans.par_iter().map(build_shared_tree).collect()
                    } else {
                        sub_plans.iter().map(build_shared_tree).collect()
                    };

                    let mut sub_indices_vec: Vec<Vec<u32>> = Vec::with_capacity(n_sub);
                    let mut sub_structure_vec: Vec<Vec<u32>> = Vec::with_capacity(n_sub);
                    let mut sub_estimation_vec: Vec<Vec<u32>> = Vec::with_capacity(n_sub);
                    let mut sub_in_sample_vec: Vec<Vec<u64>> = Vec::with_capacity(n_sub);
                    let mut sub_lambda_vec: Vec<f64> = Vec::with_capacity(n_sub);
                    for plan in sub_plans {
                        // NCL: modify multi-output gradients for next sibling tree.
                        if self.ncl_lambda > 0.0 && sub_indices_vec.len() + 1 < n_sub {
                            let ncl_lam = self.ncl_lambda;
                            for k in 0..n_classes {
                                let base = k * n_rows;
                                for i in 0..n_rows {
                                    let pred = shared_trees[sub_indices_vec.len()]
                                        .predict_binned(binned, i)
                                        * sub_scale;
                                    all_grads_flat[base + i] += ncl_lam * pred;
                                }
                            }
                        }

                        sub_indices_vec.push(plan.indices);
                        sub_structure_vec.push(plan.structure_indices);
                        sub_estimation_vec.push(plan.estimation_indices);
                        sub_in_sample_vec.push(plan.in_sample_mask);
                        sub_lambda_vec.push(plan.sub_lambda);
                    }

                    let use_coupled_mc_leaves = self.multiclass_coupled_leaves && !self.prob_avg;
                    let mut coupled_node_values: Vec<Vec<f64>> = Vec::new();
                    let mut coupled_node_counts: Vec<Vec<u32>> = Vec::new();
                    if use_coupled_mc_leaves {
                        coupled_node_values.reserve(n_sub);
                        coupled_node_counts.reserve(n_sub);
                        for sub_idx in 0..n_sub {
                            let fit_indices = if self.honest {
                                &sub_estimation_vec[sub_idx]
                            } else {
                                &sub_indices_vec[sub_idx]
                            };
                            let (vals, counts) = self.compute_multiclass_coupled_node_values(
                                &shared_trees[sub_idx],
                                binned,
                                y,
                                &probs,
                                fit_indices,
                                n_classes,
                                ls,
                            );
                            coupled_node_values.push(vals);
                            coupled_node_counts.push(counts);
                        }
                    }
                    let guided_lookup_choices: Vec<Vec<Option<GuidedCatChoice>>> = if self
                        .adaptive_leaf_experts
                        && self.cat_lookup_smooth > 0.0
                        && n_classes >= 3
                    {
                        let mut guided = Vec::with_capacity(n_sub);
                        for sub_idx in 0..n_sub {
                            let fit_indices = if self.honest {
                                &sub_estimation_vec[sub_idx]
                            } else {
                                &sub_indices_vec[sub_idx]
                            };
                            guided.push(self.compute_multiclass_guided_lookup_choices(
                                &shared_trees[sub_idx],
                                binned,
                                &all_grads_flat,
                                &all_hess_flat,
                                fit_indices,
                                n_classes,
                            ));
                        }
                        guided
                    } else {
                        Vec::new()
                    };
                    // §130: joint (coupled softmax) CLL. When enabled, fills the
                    // joint_lookup_tables structure that the install loop below consumes,
                    // replacing per-class diagonal CLL values with the full softmax-solve.
                    let joint_lookup_tables: Vec<Vec<Vec<Option<CatLookup>>>> = if self
                        .multiclass_joint_cll
                        && use_coupled_mc_leaves
                        && self.adaptive_leaf_experts
                        && self.cat_lookup_smooth > 0.0
                        && n_classes >= 3
                        && guided_lookup_choices.len() == n_sub
                        && coupled_node_values.len() == n_sub
                    {
                        let mut tables = Vec::with_capacity(n_sub);
                        for sub_idx in 0..n_sub {
                            let fit_idx = if self.honest {
                                &sub_estimation_vec[sub_idx]
                            } else {
                                &sub_indices_vec[sub_idx]
                            };
                            tables.push(self.compute_multiclass_joint_guided_lookups(
                                &shared_trees[sub_idx],
                                binned,
                                y,
                                &probs,
                                fit_idx,
                                n_classes,
                                ls,
                                &guided_lookup_choices[sub_idx],
                                &coupled_node_values[sub_idx],
                            ));
                        }
                        tables
                    } else {
                        Vec::new()
                    };
                    // 3. Refit and push in class-major order
                    // Experimental joint multiclass leaf-correction path disabled:
                    // proxy board regressed on 2026-04-22. Kept in code for future
                    // iteration, but not active on the default learner path.
                    if false && use_coupled_mc_leaves && self.leaf_correction > 0 {
                        let mut prepared_trees: Vec<Vec<DecisionTree>> =
                            (0..n_sub).map(|_| Vec::with_capacity(n_classes)).collect();

                        for k in 0..n_classes {
                            let class_base = k * n_rows;
                            for i in 0..n_rows {
                                gradients[i] = all_grads_flat[class_base + i];
                                hessians[i] = all_hess_flat[class_base + i];
                            }

                            for sub_idx in 0..n_sub {
                                let mut tree = shared_trees[sub_idx].clone();
                                let indices = &sub_indices_vec[sub_idx];
                                let estimation_indices = &sub_estimation_vec[sub_idx];

                                if self.honest {
                                    tree.refit_leaves_robust(
                                        binned,
                                        &gradients,
                                        &hessians,
                                        estimation_indices,
                                        self.lambda_reg,
                                        self.honest_tau,
                                        self.leaf_trim_pct,
                                        self.leaf_median,
                                        self.leaf_median_blend,
                                        self.leaf_mad_clip,
                                        self.leaf_adaptive_blend_kappa,
                                    );
                                    if self.cat_lookup_smooth > 0.0 && !self.adaptive_leaf_experts {
                                        tree.refit_cat_lookups(
                                            binned,
                                            &gradients,
                                            &hessians,
                                            estimation_indices,
                                            self.lambda_reg,
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
                                    tree.refit_leaves_robust(
                                        binned,
                                        &gradients,
                                        &hessians,
                                        indices,
                                        self.lambda_reg,
                                        0.0,
                                        self.leaf_trim_pct,
                                        self.leaf_median,
                                        self.leaf_median_blend,
                                        self.leaf_mad_clip,
                                        self.leaf_adaptive_blend_kappa,
                                    );
                                } else {
                                    tree.refit_leaves(
                                        binned,
                                        &gradients,
                                        &hessians,
                                        indices,
                                        self.lambda_reg,
                                    );
                                }

                                let vals = &coupled_node_values[sub_idx];
                                let counts = &coupled_node_counts[sub_idx];
                                for node in 0..tree.values.len() {
                                    if counts[node] > 0 {
                                        tree.values[node] = vals[node * n_classes + k];
                                    }
                                }

                                if self.adaptive_leaf_experts || self.cat_lookup_smooth > 0.0 {
                                    let lookup_idx = if self.honest {
                                        estimation_indices
                                    } else {
                                        indices
                                    };
                                    if self.adaptive_leaf_experts {
                                        if guided_lookup_choices.len() == n_sub {
                                            tree.install_best_lookups_guided(
                                                binned,
                                                &gradients,
                                                &hessians,
                                                lookup_idx,
                                                self.lambda_reg,
                                                self.gamma,
                                                self.min_child_weight,
                                                self.cat_lookup_smooth,
                                                &guided_lookup_choices[sub_idx],
                                            );
                                        } else {
                                            tree.install_best_lookups_with_config(
                                                binned,
                                                &gradients,
                                                &hessians,
                                                lookup_idx,
                                                self.lambda_reg,
                                                self.gamma,
                                                self.min_child_weight,
                                                self.cat_lookup_smooth,
                                                self.adaptive_cat_lookup_smooth,
                                                cat_tuple_cfg.as_ref(),
                                            );
                                        }
                                    } else {
                                        let build_idx = if self.honest {
                                            &sub_structure_vec[sub_idx]
                                        } else {
                                            indices
                                        };
                                        tree.install_cat_lookups(
                                            binned,
                                            &gradients,
                                            &hessians,
                                            build_idx,
                                            self.lambda_reg,
                                            self.gamma,
                                            self.min_child_weight,
                                            self.cat_lookup_smooth,
                                        );
                                    }
                                }

                                prepared_trees[sub_idx].push(tree);
                            }
                        }

                        let mut corr_all_grads_flat = vec![0.0f64; n_rows * n_classes];
                        let mut corr_all_hess_flat = vec![0.0f64; n_rows * n_classes];
                        let inv_k = 1.0 / n_classes as f64;

                        for sub_idx in 0..n_sub {
                            let fit_indices = if self.honest {
                                &sub_estimation_vec[sub_idx]
                            } else {
                                &sub_indices_vec[sub_idx]
                            };
                            let build_idx = if self.honest {
                                &sub_structure_vec[sub_idx]
                            } else {
                                &sub_indices_vec[sub_idx]
                            };

                            for _step in 0..self.leaf_correction {
                                let mut temp_trees = prepared_trees[sub_idx].clone();
                                for tree in temp_trees.iter_mut() {
                                    self.finalize_multiclass_tree(
                                        tree,
                                        round,
                                        n_rounds,
                                        n_sub,
                                        sub_scale,
                                        posterior_tau,
                                    );
                                }
                                for k in 0..n_classes {
                                    let tree = &temp_trees[k];
                                    for i in 0..n_rows {
                                        predictions[i * n_classes + k] +=
                                            self.learning_rate * tree.predict_binned(binned, i);
                                    }
                                }
                                Self::compute_softmax_par(
                                    &predictions,
                                    &mut probs,
                                    n_rows,
                                    n_classes,
                                );
                                for k in 0..n_classes {
                                    let tree = &temp_trees[k];
                                    for i in 0..n_rows {
                                        predictions[i * n_classes + k] -=
                                            self.learning_rate * tree.predict_binned(binned, i);
                                    }
                                }

                                for k in 0..n_classes {
                                    let base = k * n_rows;
                                    for i in 0..n_rows {
                                        let hard_label = if y[i] as usize == k { 1.0 } else { 0.0 };
                                        let label = if ls > 0.0 {
                                            (1.0 - ls) * hard_label + ls * inv_k
                                        } else {
                                            hard_label
                                        };
                                        let p = probs[i * n_classes + k];
                                        corr_all_grads_flat[base + i] = p - label;
                                        corr_all_hess_flat[base + i] = (p * (1.0 - p)).max(1e-16);
                                    }
                                }

                                let (vals, counts) = self.compute_multiclass_coupled_node_values(
                                    &shared_trees[sub_idx],
                                    binned,
                                    y,
                                    &probs,
                                    fit_indices,
                                    n_classes,
                                    ls,
                                );
                                let corr_guided = if self.adaptive_leaf_experts
                                    && self.cat_lookup_smooth > 0.0
                                    && n_classes >= 3
                                {
                                    self.compute_multiclass_guided_lookup_choices(
                                        &shared_trees[sub_idx],
                                        binned,
                                        &corr_all_grads_flat,
                                        &corr_all_hess_flat,
                                        fit_indices,
                                        n_classes,
                                    )
                                } else {
                                    Vec::new()
                                };

                                for k in 0..n_classes {
                                    let class_base = k * n_rows;
                                    for i in 0..n_rows {
                                        gradients[i] = corr_all_grads_flat[class_base + i];
                                        hessians[i] = corr_all_hess_flat[class_base + i];
                                    }

                                    let tree = &mut prepared_trees[sub_idx][k];
                                    for node in 0..tree.values.len() {
                                        if counts[node] > 0 {
                                            tree.values[node] = vals[node * n_classes + k];
                                        }
                                    }

                                    if self.adaptive_leaf_experts || self.cat_lookup_smooth > 0.0 {
                                        if self.adaptive_leaf_experts {
                                            if !corr_guided.is_empty() {
                                                tree.install_best_lookups_guided(
                                                    binned,
                                                    &gradients,
                                                    &hessians,
                                                    fit_indices,
                                                    self.lambda_reg,
                                                    self.gamma,
                                                    self.min_child_weight,
                                                    self.cat_lookup_smooth,
                                                    &corr_guided,
                                                );
                                            } else {
                                                tree.install_best_lookups_with_config(
                                                    binned,
                                                    &gradients,
                                                    &hessians,
                                                    fit_indices,
                                                    self.lambda_reg,
                                                    self.gamma,
                                                    self.min_child_weight,
                                                    self.cat_lookup_smooth,
                                                    self.adaptive_cat_lookup_smooth,
                                                    cat_tuple_cfg.as_ref(),
                                                );
                                            }
                                        } else {
                                            tree.install_cat_lookups(
                                                binned,
                                                &gradients,
                                                &hessians,
                                                build_idx,
                                                self.lambda_reg,
                                                self.gamma,
                                                self.min_child_weight,
                                                self.cat_lookup_smooth,
                                            );
                                        }
                                    }
                                }
                            }

                            for k in 0..n_classes {
                                let class_base = k * n_rows;
                                for i in 0..n_rows {
                                    gradients[i] = corr_all_grads_flat[class_base + i];
                                }

                                let mut tree = prepared_trees[sub_idx][k].clone();
                                self.finalize_multiclass_tree(
                                    &mut tree,
                                    round,
                                    n_rounds,
                                    n_sub,
                                    sub_scale,
                                    posterior_tau,
                                );

                                for i in 0..n_rows {
                                    predictions[i * n_classes + k] +=
                                        self.learning_rate * tree.predict_binned(binned, i);
                                }

                                if es_active {
                                    let (eval_bins, _, en, _, eval_cll_bins) =
                                        eval_data.as_ref().unwrap();
                                    let en = *en;
                                    for i in 0..en {
                                        eval_preds[i * n_classes + k] += self.learning_rate
                                            * tree.predict_binned_raw(
                                                eval_bins,
                                                en,
                                                i,
                                                eval_cll_bins,
                                            );
                                    }
                                }

                                if self.n_refine > 0 && !sub_in_sample_vec[sub_idx].is_empty() {
                                    self.tree_in_sample.push(sub_in_sample_vec[sub_idx].clone());
                                }

                                self.dart_tree_weights.push(1.0);
                                self.apply_eblp(&mut tree);
                                self.apply_hss(&mut tree);
                                self.apply_scs(&mut tree, binned, &gradients, n_rows);
                                self.apply_newton_trust_region(&mut tree);
                                self.trees.push(tree);
                            }
                        }
                    } else {
                        for k in 0..n_classes {
                            let class_base = k * n_rows;
                            for i in 0..n_rows {
                                gradients[i] = all_grads_flat[class_base + i];
                                hessians[i] = all_hess_flat[class_base + i];
                            }

                            for sub_idx in 0..n_sub {
                                let mut tree = shared_trees[sub_idx].clone();
                                let indices = &sub_indices_vec[sub_idx];
                                let estimation_indices = &sub_estimation_vec[sub_idx];
                                let sub_lambda = sub_lambda_vec[sub_idx];

                                // Refit leaf values for this class
                                if self.honest {
                                    tree.refit_leaves_robust(
                                        binned,
                                        &gradients,
                                        &hessians,
                                        estimation_indices,
                                        sub_lambda,
                                        self.honest_tau,
                                        self.leaf_trim_pct,
                                        self.leaf_median,
                                        self.leaf_median_blend,
                                        self.leaf_mad_clip,
                                        self.leaf_adaptive_blend_kappa,
                                    );
                                    if self.cat_lookup_smooth > 0.0 && !self.adaptive_leaf_experts {
                                        tree.refit_cat_lookups(
                                            binned,
                                            &gradients,
                                            &hessians,
                                            estimation_indices,
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
                                    tree.refit_leaves_robust(
                                        binned,
                                        &gradients,
                                        &hessians,
                                        indices,
                                        sub_lambda,
                                        0.0,
                                        self.leaf_trim_pct,
                                        self.leaf_median,
                                        self.leaf_median_blend,
                                        self.leaf_mad_clip,
                                        self.leaf_adaptive_blend_kappa,
                                    );
                                } else {
                                    tree.refit_leaves(
                                        binned, &gradients, &hessians, indices, sub_lambda,
                                    );
                                }

                                if use_coupled_mc_leaves {
                                    let vals = &coupled_node_values[sub_idx];
                                    let counts = &coupled_node_counts[sub_idx];
                                    for node in 0..tree.values.len() {
                                        if counts[node] > 0 {
                                            tree.values[node] = vals[node * n_classes + k];
                                        }
                                    }
                                }

                                // CLL post-hoc
                                if self.adaptive_leaf_experts || self.cat_lookup_smooth > 0.0 {
                                    let lookup_idx = if self.honest {
                                        estimation_indices
                                    } else {
                                        indices
                                    };
                                    if self.adaptive_leaf_experts {
                                        if guided_lookup_choices.len() == n_sub {
                                            tree.install_best_lookups_guided(
                                                binned,
                                                &gradients,
                                                &hessians,
                                                lookup_idx,
                                                sub_lambda,
                                                self.gamma,
                                                self.min_child_weight,
                                                self.cat_lookup_smooth,
                                                &guided_lookup_choices[sub_idx],
                                            );
                                        } else {
                                            tree.install_best_lookups_with_config(
                                                binned,
                                                &gradients,
                                                &hessians,
                                                lookup_idx,
                                                sub_lambda,
                                                self.gamma,
                                                self.min_child_weight,
                                                self.cat_lookup_smooth,
                                                self.adaptive_cat_lookup_smooth,
                                                cat_tuple_cfg.as_ref(),
                                            );
                                        }
                                    } else {
                                        let build_idx = if self.honest {
                                            &sub_structure_vec[sub_idx]
                                        } else {
                                            indices
                                        };
                                        tree.install_cat_lookups(
                                            binned,
                                            &gradients,
                                            &hessians,
                                            build_idx,
                                            sub_lambda,
                                            self.gamma,
                                            self.min_child_weight,
                                            self.cat_lookup_smooth,
                                        );
                                    }
                                }

                                if joint_lookup_tables.len() == n_sub {
                                    let node_lookups = &joint_lookup_tables[sub_idx][k];
                                    for node in 0..tree.values.len() {
                                        if let Some(lookup) =
                                            node_lookups.get(node).and_then(|v| v.as_ref())
                                        {
                                            tree.cat_lookups[node] = Some(lookup.clone());
                                            let ramp_base = node * tree.ramp_k;
                                            for j in 0..tree.ramp_k {
                                                if ramp_base + j < tree.ramp_features.len() {
                                                    tree.ramp_features[ramp_base + j] = u32::MAX;
                                                }
                                                if ramp_base + j < tree.ramp_slopes.len() {
                                                    tree.ramp_slopes[ramp_base + j] = 0.0;
                                                }
                                            }
                                            let pair_base = node * 2;
                                            if pair_base + 1 < tree.leaf_pair_features.len() {
                                                tree.leaf_pair_features[pair_base] = u32::MAX;
                                                tree.leaf_pair_features[pair_base + 1] = u32::MAX;
                                            }
                                            if node < tree.leaf_pair_slopes.len() {
                                                tree.leaf_pair_slopes[node] = 0.0;
                                            }
                                            let qi = tree.quad_n_interactions;
                                            if qi > 0 {
                                                let quad_base = node * qi;
                                                for q in 0..qi {
                                                    if quad_base + q < tree.quad_slopes.len() {
                                                        tree.quad_slopes[quad_base + q] = 0.0;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                self.finalize_multiclass_tree(
                                    &mut tree,
                                    round,
                                    n_rounds,
                                    n_sub,
                                    sub_scale,
                                    posterior_tau,
                                );

                                // LOO prediction (for non-honest)
                                if use_coupled_mc_leaves {
                                    for i in 0..n_rows {
                                        predictions[i * n_classes + k] +=
                                            self.learning_rate * tree.predict_binned(binned, i);
                                    }
                                } else if !self.honest {
                                    let n_nodes = tree.values.len();
                                    let mut leaf_g = vec![0.0f64; n_nodes];
                                    let mut leaf_h = vec![0.0f64; n_nodes];
                                    let mut leaf_cnt = vec![0u32; n_nodes];
                                    let mut is_build = vec![false; n_rows];
                                    for &idx in indices.iter() {
                                        let ii = idx as usize;
                                        is_build[ii] = true;
                                        let leaf = tree.route_to_leaf(binned, ii);
                                        leaf_g[leaf] += gradients[ii];
                                        leaf_h[leaf] += hessians[ii];
                                        leaf_cnt[leaf] += 1;
                                    }
                                    let lr = self.multiclass_tree_lr();
                                    let eff_scale = if n_sub > 1 { sub_scale } else { 1.0 };
                                    for i in 0..n_rows {
                                        let leaf = tree.route_to_leaf(binned, i);
                                        let base = if is_build[i] && leaf_cnt[leaf] > 1 {
                                            let g = leaf_g[leaf] - gradients[i];
                                            let h = leaf_h[leaf] - hessians[i];
                                            let denom = h + self.lambda_reg;
                                            if denom > 1e-10 {
                                                let raw = -g / denom * eff_scale;
                                                if posterior_tau > 0.0 {
                                                    let cnt_loo = (leaf_cnt[leaf] - 1) as f64;
                                                    raw * (cnt_loo / (cnt_loo + posterior_tau))
                                                } else {
                                                    raw
                                                }
                                            } else {
                                                tree.values[leaf]
                                            }
                                        } else {
                                            tree.values[leaf]
                                        };
                                        predictions[i * n_classes + k] += lr * base;
                                    }
                                } else {
                                    for i in 0..n_rows {
                                        predictions[i * n_classes + k] +=
                                            self.learning_rate * tree.predict_binned(binned, i);
                                    }
                                }

                                if es_active {
                                    let (eval_bins, _, en, _, eval_cll_bins) =
                                        eval_data.as_ref().unwrap();
                                    let en = *en;
                                    for i in 0..en {
                                        eval_preds[i * n_classes + k] += self.learning_rate
                                            * tree.predict_binned_raw(
                                                eval_bins,
                                                en,
                                                i,
                                                eval_cll_bins,
                                            );
                                    }
                                }

                                if self.n_refine > 0 && !sub_in_sample_vec[sub_idx].is_empty() {
                                    self.tree_in_sample.push(sub_in_sample_vec[sub_idx].clone());
                                }

                                self.dart_tree_weights.push(1.0);
                                self.apply_eblp(&mut tree);
                                self.apply_hss(&mut tree);
                                self.apply_scs(&mut tree, binned, &gradients, n_rows);
                                self.apply_newton_trust_region(&mut tree);
                                self.trees.push(tree);
                            }
                        }
                    }

                    // ── Frequency-based leaf values for RF/XT mode (prob_avg) ──────
                    // Replace Newton-step logit leaf values with class frequencies.
                    // Only for 1-round (pure RF) mode — multi-round boosting is incompatible.
                    if self.prob_avg && n_rounds == 1 {
                        let n_trees_this_round = n_classes * n_sub;
                        let tree_start = self.trees.len() - n_trees_this_round;
                        // Use label_smooth as frequency smoothing when prob_avg is active (0 = no smoothing like sklearn)
                        let smooth = self.label_smooth;

                        for sub_idx in 0..n_sub {
                            let indices = &sub_indices_vec[sub_idx];
                            let n_nodes = shared_trees[sub_idx].values.len();
                            let mut leaf_counts = vec![vec![0.0f64; n_classes]; n_nodes];
                            let mut leaf_totals = vec![0.0f64; n_nodes];

                            for &idx in indices.iter() {
                                let ii = idx as usize;
                                let leaf = shared_trees[sub_idx].route_to_leaf(binned, ii);
                                let cls = y[ii] as usize;
                                leaf_counts[leaf][cls] += 1.0;
                                leaf_totals[leaf] += 1.0;
                            }

                            // Replace leaf values with smoothed class frequencies
                            let inv_k = 1.0 / n_classes as f64;
                            for k in 0..n_classes {
                                let tree_idx = tree_start + k * n_sub + sub_idx;
                                for leaf_id in 0..n_nodes {
                                    let total = leaf_totals[leaf_id];
                                    if total > 0.0 {
                                        self.trees[tree_idx].values[leaf_id] =
                                            (leaf_counts[leaf_id][k] + smooth)
                                                / (total + n_classes as f64 * smooth);
                                    } else {
                                        self.trees[tree_idx].values[leaf_id] = inv_k;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // ── Original path: separate tree per class ──────────────────────────
            else {
                for k in 0..n_classes {
                    for i in 0..n_rows {
                        let hard_label = if y[i] as usize == k { 1.0 } else { 0.0 };
                        let label = if ls > 0.0 {
                            (1.0 - ls) * hard_label + ls * inv_k
                        } else {
                            hard_label
                        };
                        gradients[i] = probs[i * n_classes + k] - label;
                        hessians[i] = (probs[i * n_classes + k] * (1.0 - probs[i * n_classes + k]))
                            .max(1e-16);
                    }

                    // ntp: build n_sub trees per class per round (averaged)
                    let planned_subsamples = self.round_subsamples(&mut rng, n_rows, n_sub);
                    for _sub_idx in 0..n_sub {
                        let indices = planned_subsamples[_sub_idx].clone();
                        let feature_mask = self
                            .make_feature_mask_for_subtree(&mut rng, n_features, round, _sub_idx);

                        // Honest estimation for multiclass
                        let (structure_indices, estimation_indices) = if self.honest {
                            if self.honest_fraction <= 0.0 && self.subsample_rate < 1.0 {
                                let mut in_sample = bitvec_new(n_rows);
                                for &idx in &indices {
                                    bitvec_set(&mut in_sample, idx as usize);
                                }
                                // Store mask for honest refine
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
                        } else {
                            (indices.clone(), Vec::new())
                        };
                        let build_indices = if self.honest {
                            &structure_indices
                        } else {
                            &indices
                        };
                        let lookup_smooth_build = if self.adaptive_leaf_experts {
                            0.0
                        } else {
                            self.cat_lookup_smooth
                        };

                        let tree_seed: u64 = rng.random();
                        let mut tree = match grow {
                            "leafwise" => DecisionTree::build_leafwise(
                                binned,
                                &gradients,
                                &hessians,
                                build_indices,
                                self.lambda_reg,
                                self.gamma,
                                self.max_depth,
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
                                crate::tree::CatPairConfig {
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
                                &gradients,
                                &hessians,
                                build_indices,
                                self.lambda_reg,
                                self.gamma,
                                self.max_depth,
                                self.min_child_weight,
                                &feature_mask,
                                self.gain_penalty,
                                self.extra_trees,
                                tree_seed,
                            ),
                            _ => DecisionTree::build_depthwise(
                                binned,
                                &gradients,
                                &hessians,
                                build_indices,
                                self.lambda_reg,
                                0.0,
                                self.gamma,
                                self.max_depth,
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
                                self.lookahead_alpha,
                                self.adaptive_leaf_experts
                                    || self.leaf_linear
                                    || self.cat_lookup_smooth > 0.0,
                                self.sparse_oblique_splits,
                                self.interval_splits,
                                None,
                                crate::tree::CatPairConfig {
                                    enabled: self.jit_catpair_enabled,
                                    top_k_cat: self.jit_catpair_top_k,
                                    k_buckets: self.jit_catpair_k_buckets,
                                    min_node_rows: self.jit_catpair_min_node_rows,
                                    max_node_depth: self.jit_catpair_max_node_depth,
                                    gain_margin: self.jit_catpair_gain_margin,
                                },
                            ),
                        };

                        if self.honest {
                            tree.refit_leaves_robust(
                                binned,
                                &gradients,
                                &hessians,
                                &estimation_indices,
                                self.lambda_reg,
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
                                    self.lambda_reg,
                                    self.cat_lookup_smooth,
                                    self.min_child_weight,
                                );
                            }
                        }
                        // Non-honest multiclass: trim skipped (destabilizes per-class magnitudes)

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
                                    self.lambda_reg,
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
                                    self.lambda_reg,
                                    self.gamma,
                                    self.min_child_weight,
                                    self.cat_lookup_smooth,
                                );
                            }
                        }
                        // Learning rate decay for multiclass
                        if self.lr_decay < 1.0 && n_rounds > 1 {
                            let factor = 1.0
                                - (1.0 - self.lr_decay) * (round as f64) / (n_rounds as f64 - 1.0);
                            for v in tree.values.iter_mut() {
                                *v *= factor;
                            }
                            tree.scale_ramp_slopes(factor);
                            tree.scale_cat_lookups(factor);
                        }

                        // max_delta_step: clip leaf values
                        if self.max_delta_step > 0.0 {
                            let mds = self.max_delta_step;
                            for v in tree.values.iter_mut() {
                                *v = v.clamp(-mds, mds);
                            }
                        }

                        // Scale tree values by 1/ntp when ntp > 1 (averaged sub-trees)
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
                        self.apply_hierarchical_shrinkage(&mut tree);

                        // LOO prediction for multiclass: exclude each sample's own g/h from leaf value
                        if !self.honest {
                            let loo_indices = if self.honest && !estimation_indices.is_empty() {
                                &estimation_indices
                            } else {
                                &indices
                            };
                            let n_nodes = tree.values.len();
                            let mut leaf_g = vec![0.0f64; n_nodes];
                            let mut leaf_h = vec![0.0f64; n_nodes];
                            let mut leaf_cnt = vec![0u32; n_nodes];
                            let mut is_build = vec![false; n_rows];
                            for &idx in loo_indices {
                                let ii = idx as usize;
                                is_build[ii] = true;
                                let leaf = tree.route_to_leaf(binned, ii);
                                leaf_g[leaf] += gradients[ii];
                                leaf_h[leaf] += hessians[ii];
                                leaf_cnt[leaf] += 1;
                            }
                            let lr = self.multiclass_tree_lr();
                            let eff_scale = if n_sub > 1 { sub_scale } else { 1.0 };
                            for i in 0..n_rows {
                                let leaf = tree.route_to_leaf(binned, i);
                                let base = if is_build[i] && leaf_cnt[leaf] > 1 {
                                    let g = leaf_g[leaf] - gradients[i];
                                    let h = leaf_h[leaf] - hessians[i];
                                    let denom = h + self.lambda_reg;
                                    if denom > 1e-10 {
                                        let raw = -g / denom * eff_scale;
                                        if posterior_tau > 0.0 {
                                            let cnt_loo = (leaf_cnt[leaf] - 1) as f64;
                                            raw * (cnt_loo / (cnt_loo + posterior_tau))
                                        } else {
                                            raw
                                        }
                                    } else {
                                        tree.values[leaf]
                                    }
                                } else {
                                    tree.values[leaf]
                                };
                                predictions[i * n_classes + k] += lr * base;
                            }
                        } else {
                            for i in 0..n_rows {
                                predictions[i * n_classes + k] +=
                                    self.learning_rate * tree.predict_binned(binned, i);
                            }
                        }

                        if es_active {
                            let (eval_bins, _, en, _, eval_cll_bins) = eval_data.as_ref().unwrap();
                            let en = *en;
                            for i in 0..en {
                                eval_preds[i * n_classes + k] += self.learning_rate
                                    * tree.predict_binned_raw(eval_bins, en, i, eval_cll_bins);
                            }
                        }

                        // NCL: modify gradients for next sibling tree within this class
                        if self.ncl_lambda > 0.0 && _sub_idx + 1 < n_sub {
                            let ncl_lam = self.ncl_lambda;
                            for i in 0..n_rows {
                                let pred = tree.predict_binned(binned, i) * sub_scale;
                                gradients[i] += ncl_lam * pred;
                            }
                        }

                        self.dart_tree_weights.push(1.0);
                        self.apply_eblp(&mut tree);
                        self.apply_hss(&mut tree);
                        self.apply_scs(&mut tree, binned, &gradients, n_rows);
                        self.apply_newton_trust_region(&mut tree);
                        self.trees.push(tree);
                    } // end ntp sub-tree loop
                }
            } // end if/else multi_output_tree

            // Early stopping check (after full round of K trees)
            if es_active {
                let (_, eval_y, en, _, _) = eval_data.as_ref().unwrap();
                let en = *en;
                let eval_loss =
                    self.compute_multiclass_eval_loss(eval_y, &eval_preds, en, n_classes);
                // PASA: record val loss history for plateau averaging
                self.val_losses.push(eval_loss);
                if eval_loss < best_eval_loss {
                    best_eval_loss = eval_loss;
                    best_round = round + 1;
                    rounds_without_improvement = 0;
                } else {
                    rounds_without_improvement += 1;
                    if rounds_without_improvement >= self.early_stopping_rounds {
                        break;
                    }
                }
            }

            // Cyclic refinement with alpha blending
            if self.refine_every > 0 && (round + 1) % self.refine_every == 0 {
                let alpha = self.refine_alpha;
                let trees_per_class = self.multiclass_trees_per_class_round();
                let tree_lr = self.multiclass_tree_lr();
                for t_idx in 0..self.trees.len() {
                    let class_k = (t_idx / trees_per_class) % n_classes;
                    for i in 0..n_rows {
                        predictions[i * n_classes + class_k] -=
                            tree_lr * self.trees[t_idx].predict_binned(binned, i);
                    }
                    Self::compute_softmax(&predictions, &mut probs, n_rows, n_classes);
                    for i in 0..n_rows {
                        let hard_label = if y[i] as usize == class_k { 1.0 } else { 0.0 };
                        let label = if ls > 0.0 {
                            (1.0 - ls) * hard_label + ls * inv_k
                        } else {
                            hard_label
                        };
                        gradients[i] = probs[i * n_classes + class_k] - label;
                        hessians[i] = (probs[i * n_classes + class_k]
                            * (1.0 - probs[i * n_classes + class_k]))
                            .max(1e-16);
                    }
                    let old_values = self.trees[t_idx].values.clone();
                    self.trees[t_idx].refit_leaves(
                        binned,
                        &gradients,
                        &hessians,
                        &all_indices,
                        self.lambda_reg,
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
                        predictions[i * n_classes + class_k] +=
                            tree_lr * self.trees[t_idx].predict_binned(binned, i);
                    }
                }
            }
        }

        // PASA: record best_round (in trees, not rounds) for plateau averaging.
        let trees_per_round = n_classes * self.multiclass_trees_per_class_round();
        self.best_round = best_round * trees_per_round;
        // Trim to best round if early stopping triggered (unless keep_all_trees).
        if es_active && best_round * trees_per_round < self.trees.len() && !self.keep_all_trees {
            self.trees.truncate(best_round * trees_per_round);
            if !self.tree_in_sample.is_empty() {
                self.tree_in_sample.truncate(best_round * trees_per_round);
            }
        }

        // ── Phase 2: Interleaved leaf splitting + refinement ─────────────
        for _ in 0..self.n_leaf_splits {
            self.leaf_split_pass_multiclass(binned, y, n_rows, n_classes);
            self.refine_global_multiclass(binned, y, n_rows, n_classes, 1);
        }

        // ── Phase 3: Final leaf optimization (per-class coordinate descent) ──
        if self.n_refine > 0 {
            self.refine_global_multiclass(binned, y, n_rows, n_classes, self.n_refine);
            self.prune_similar_leaves();
        }
    }
    pub(super) fn compute_multiclass_coupled_node_values(
        &self,
        tree: &DecisionTree,
        binned: &BinnedData,
        y: &[f64],
        probs: &[f64],
        indices: &[u32],
        n_classes: usize,
        label_smooth: f64,
    ) -> (Vec<f64>, Vec<u32>) {
        let n_nodes = tree.values.len();
        let mut g_sums = vec![0.0f64; n_nodes * n_classes];
        let mut h_sums = vec![0.0f64; n_nodes * n_classes * n_classes];
        let mut counts = vec![0u32; n_nodes];
        let inv_k = 1.0 / n_classes as f64;

        for &idx in indices.iter() {
            let row = idx as usize;
            let row_base = row * n_classes;
            let mut node = 0usize;
            loop {
                counts[node] += 1;
                let g_base = node * n_classes;
                let h_base = node * n_classes * n_classes;
                for a in 0..n_classes {
                    let p_a = probs[row_base + a];
                    let hard_label = if y[row] as usize == a { 1.0 } else { 0.0 };
                    let label = if label_smooth > 0.0 {
                        (1.0 - label_smooth) * hard_label + label_smooth * inv_k
                    } else {
                        hard_label
                    };
                    g_sums[g_base + a] += p_a - label;
                    for b in 0..n_classes {
                        let p_b = probs[row_base + b];
                        let h = if a == b {
                            p_a * (1.0 - p_a)
                        } else {
                            -p_a * p_b
                        };
                        h_sums[h_base + a * n_classes + b] += h;
                    }
                }

                let feat = tree.split_features[node];
                if feat == u32::MAX {
                    break;
                }
                let feat = feat as usize;
                let bin = binned.get_bin_u16(row, feat);
                node = if tree.is_cat_pair(node) {
                    match tree.cat_pair_route_binned(node, binned, row) {
                        Some(true) => tree.left_children[node] as usize,
                        Some(false) => tree.right_children[node] as usize,
                        None => {
                            if tree.missing_goes_left[node] {
                                tree.left_children[node] as usize
                            } else {
                                tree.right_children[node] as usize
                            }
                        }
                    }
                } else if bin == MISSING_BIN {
                    if tree.missing_goes_left[node] {
                        tree.left_children[node] as usize
                    } else {
                        tree.right_children[node] as usize
                    }
                } else if tree.is_cat_split[node] {
                    if bitmask_test(&tree.cat_left_masks[node], bin as usize) {
                        tree.left_children[node] as usize
                    } else {
                        tree.right_children[node] as usize
                    }
                } else if bin <= tree.split_bins[node] {
                    tree.left_children[node] as usize
                } else {
                    tree.right_children[node] as usize
                };
            }
        }

        let mut values = vec![0.0f64; n_nodes * n_classes];
        for node in 0..n_nodes {
            if counts[node] == 0 {
                continue;
            }
            let g_base = node * n_classes;
            let h_base = node * n_classes * n_classes;
            let mut a = h_sums[h_base..h_base + n_classes * n_classes].to_vec();
            let lambda_eff = (self.lambda_reg
                + self.lambda_reg / (counts[node] as f64).max(1.0).sqrt())
            .max(1e-6);
            for k in 0..n_classes {
                a[k * n_classes + k] += lambda_eff;
            }
            let mut rhs = vec![0.0f64; n_classes];
            for k in 0..n_classes {
                rhs[k] = -g_sums[g_base + k];
            }
            let mut sol = solve_spd(n_classes, &a, &rhs);
            let mean = sol.iter().sum::<f64>() / n_classes as f64;
            for k in 0..n_classes {
                sol[k] -= mean;
            }
            // Trust-region in logit space: dense softmax Newton steps can overshoot
            // on tiny leaves even with ridge regularization. Cap the Newton
            // decrement so coupled updates stay in a local regime.
            let mut dec_sq = 0.0f64;
            for k in 0..n_classes {
                dec_sq += -g_sums[g_base + k] * sol[k];
            }
            let dec_cap = if self.newton_decrement_cap > 0.0 {
                self.newton_decrement_cap
            } else {
                1.0
            };
            if dec_sq.is_finite() && dec_sq > dec_cap * dec_cap {
                let scale = dec_cap / dec_sq.sqrt();
                for v in sol.iter_mut() {
                    *v *= scale;
                }
            }
            for k in 0..n_classes {
                values[g_base + k] = sol[k];
            }
        }
        (values, counts)
    }

    pub(super) fn compute_multiclass_guided_lookup_choices(
        &self,
        tree: &DecisionTree,
        binned: &BinnedData,
        all_grads_flat: &[f64],
        all_hess_flat: &[f64],
        indices: &[u32],
        n_classes: usize,
    ) -> Vec<Option<GuidedCatChoice>> {
        let n_nodes = tree.values.len();
        let mut choices = vec![None; n_nodes];
        if self.cat_lookup_smooth <= 0.0 || n_classes < 3 {
            return choices;
        }
        let cat_cols: Vec<usize> = (0..binned.n_features)
            .filter(|&col| {
                col < binned.cll_is_categorical.len()
                    && binned.cll_is_categorical[col]
                    && binned.cll_n_bins[col] >= 2
            })
            .collect();
        if cat_cols.len() < 2 {
            return choices;
        }

        let mut leaf_samples: Vec<Vec<u32>> = vec![Vec::new(); n_nodes];
        for &idx in indices {
            let leaf = tree.route_to_leaf(binned, idx as usize);
            leaf_samples[leaf].push(idx);
        }

        let n_rows = binned.n_rows;
        for node in 0..n_nodes {
            if tree.split_features[node] != u32::MAX {
                continue;
            }
            let samples = &leaf_samples[node];
            if samples.len() < (2 * n_classes).max(8) {
                continue;
            }
            let mut base_obj = 0.0f64;
            let mut g_total = vec![0.0f64; n_classes];
            let mut h_total = vec![0.0f64; n_classes];
            for &idx in samples {
                let row = idx as usize;
                for k in 0..n_classes {
                    let off = k * n_rows + row;
                    g_total[k] += all_grads_flat[off];
                    h_total[k] += all_hess_flat[off];
                }
            }
            for k in 0..n_classes {
                base_obj += g_total[k] * g_total[k] / (h_total[k] + self.lambda_reg);
            }

            let mut feat_scores: Vec<(f64, usize)> = Vec::new();
            for &col in &cat_cols {
                let n_bins = binned.cll_n_bins[col].max(1);
                let mut bin_g = vec![0.0f64; n_bins * n_classes];
                let mut bin_h = vec![0.0f64; n_bins * n_classes];
                let mut bin_h_total = vec![0.0f64; n_bins];
                let col_offset = col * n_rows;
                for &idx in samples {
                    let row = idx as usize;
                    let bin = binned.cll_hash_bins[col_offset + row];
                    if bin == MISSING_BIN {
                        continue;
                    }
                    let bu = bin as usize;
                    if bu >= n_bins {
                        continue;
                    }
                    for k in 0..n_classes {
                        let off = k * n_rows + row;
                        let base = k * n_bins + bu;
                        let g = all_grads_flat[off];
                        let h = all_hess_flat[off];
                        bin_g[base] += g;
                        bin_h[base] += h;
                        bin_h_total[bu] += h;
                    }
                }
                let mut obj = 0.0f64;
                let mut n_active = 0usize;
                for b in 0..n_bins {
                    if bin_h_total[b] < self.min_child_weight {
                        continue;
                    }
                    n_active += 1;
                    for k in 0..n_classes {
                        let base = k * n_bins + b;
                        obj += bin_g[base] * bin_g[base] / (bin_h[base] + self.lambda_reg);
                    }
                }
                if n_active < 2 {
                    continue;
                }
                let gain = 0.5 * (obj - base_obj) - self.gamma * (n_active as f64).sqrt();
                if gain.is_finite() && gain > 0.0 {
                    feat_scores.push((gain, col));
                }
            }
            if feat_scores.len() < 2 {
                continue;
            }
            feat_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let best_single_gain = feat_scores[0].0;
            let best_feat = feat_scores[0].1;
            if cat_cols.len() > 12 && feat_scores.len() > 6 {
                feat_scores.truncate(6);
            }

            let mut best_pair_gain = 0.0f64;
            let mut best_pair = (usize::MAX, usize::MAX);
            let mut best_pair_n_bins = 0usize;
            let mut best_pair_stride = 0usize;
            for i in 0..feat_scores.len() {
                let fi = feat_scores[i].1;
                let off_i = fi * n_rows;
                for j in (i + 1)..feat_scores.len() {
                    let fj = feat_scores[j].1;
                    let off_j = fj * n_rows;
                    let n_i = binned.cll_n_bins[fi].max(1);
                    let n_j = binned.cll_n_bins[fj].max(1);
                    let exact_pair_bins = n_i.checked_mul(n_j).filter(|&n| n <= 256).unwrap_or(0);
                    let (pair_n_bins, pair_stride) = if exact_pair_bins > 0 {
                        (exact_pair_bins, n_j)
                    } else {
                        ((samples.len() / 10).clamp(8, 32), 0)
                    };
                    let mut bin_g = vec![0.0f64; pair_n_bins * n_classes];
                    let mut bin_h = vec![0.0f64; pair_n_bins * n_classes];
                    let mut bin_h_total = vec![0.0f64; pair_n_bins];
                    for &idx in samples {
                        let row = idx as usize;
                        let b1 = binned.cll_hash_bins[off_i + row];
                        let b2 = binned.cll_hash_bins[off_j + row];
                        if b1 == MISSING_BIN || b2 == MISSING_BIN {
                            continue;
                        }
                        let bu = if pair_stride > 0 {
                            (b1 as usize) * pair_stride + b2 as usize
                        } else {
                            ((b1 as u32).wrapping_mul(257).wrapping_add(b2 as u32)) as usize
                                % pair_n_bins
                        };
                        for k in 0..n_classes {
                            let off = k * n_rows + row;
                            let base = k * pair_n_bins + bu;
                            let g = all_grads_flat[off];
                            let h = all_hess_flat[off];
                            bin_g[base] += g;
                            bin_h[base] += h;
                            bin_h_total[bu] += h;
                        }
                    }
                    let mut obj = 0.0f64;
                    let mut n_active = 0usize;
                    for b in 0..pair_n_bins {
                        if bin_h_total[b] < self.min_child_weight {
                            continue;
                        }
                        n_active += 1;
                        for k in 0..n_classes {
                            let base = k * pair_n_bins + b;
                            obj += bin_g[base] * bin_g[base] / (bin_h[base] + self.lambda_reg);
                        }
                    }
                    if n_active < 2 {
                        continue;
                    }
                    let gain = 0.5 * (obj - base_obj) - 1.5 * self.gamma * (n_active as f64).sqrt();
                    if gain.is_finite() && gain > best_pair_gain {
                        best_pair_gain = gain;
                        best_pair = (fi, fj);
                        best_pair_n_bins = pair_n_bins;
                        best_pair_stride = pair_stride;
                    }
                }
            }
            let mut best_triple_gain = 0.0f64;
            let mut best_triple = (usize::MAX, usize::MAX, usize::MAX);
            let mut best_triple_n_bins = 0usize;
            let mut best_triple_stride2 = 0usize;
            let mut best_triple_stride3 = 0usize;
            if feat_scores.len() >= 3 && samples.len() >= (16 * n_classes).max(48) {
                for a in 0..feat_scores.len() {
                    let f0 = feat_scores[a].1;
                    let off0 = f0 * n_rows;
                    for b in (a + 1)..feat_scores.len() {
                        let f1 = feat_scores[b].1;
                        let off1 = f1 * n_rows;
                        for c in (b + 1)..feat_scores.len() {
                            let f2 = feat_scores[c].1;
                            let off2 = f2 * n_rows;
                            let n0 = binned.cll_n_bins[f0].max(1);
                            let n1 = binned.cll_n_bins[f1].max(1);
                            let n2 = binned.cll_n_bins[f2].max(1);
                            let Some(pair_bins) = n0.checked_mul(n1) else {
                                continue;
                            };
                            let Some(triple_bins) = pair_bins.checked_mul(n2) else {
                                continue;
                            };
                            if triple_bins > 256 {
                                continue;
                            }
                            let mut bin_g = vec![0.0f64; triple_bins * n_classes];
                            let mut bin_h = vec![0.0f64; triple_bins * n_classes];
                            let mut bin_h_total = vec![0.0f64; triple_bins];
                            for &idx in samples {
                                let row = idx as usize;
                                let b0 = binned.cll_hash_bins[off0 + row];
                                let b1 = binned.cll_hash_bins[off1 + row];
                                let b2 = binned.cll_hash_bins[off2 + row];
                                if b0 == MISSING_BIN || b1 == MISSING_BIN || b2 == MISSING_BIN {
                                    continue;
                                }
                                let bu = ((b0 as usize) * n1 + b1 as usize) * n2 + b2 as usize;
                                for k in 0..n_classes {
                                    let off = k * n_rows + row;
                                    let base = k * triple_bins + bu;
                                    let g = all_grads_flat[off];
                                    let h = all_hess_flat[off];
                                    bin_g[base] += g;
                                    bin_h[base] += h;
                                    bin_h_total[bu] += h;
                                }
                            }
                            let mut obj = 0.0f64;
                            let mut n_active = 0usize;
                            for bin in 0..triple_bins {
                                if bin_h_total[bin] < self.min_child_weight {
                                    continue;
                                }
                                n_active += 1;
                                for k in 0..n_classes {
                                    let base = k * triple_bins + bin;
                                    obj +=
                                        bin_g[base] * bin_g[base] / (bin_h[base] + self.lambda_reg);
                                }
                            }
                            if n_active < 3 {
                                continue;
                            }
                            let gain = 0.5 * (obj - base_obj)
                                - 2.0 * self.gamma * (n_active as f64).sqrt();
                            if gain.is_finite() && gain > best_triple_gain {
                                best_triple_gain = gain;
                                best_triple = (f0, f1, f2);
                                best_triple_n_bins = triple_bins;
                                best_triple_stride2 = n1;
                                best_triple_stride3 = n2;
                            }
                        }
                    }
                }
            }
            if best_triple.2 != usize::MAX
                && best_triple_gain.is_finite()
                && best_triple_gain > best_pair_gain.max(best_single_gain)
            {
                choices[node] = Some(GuidedCatChoice {
                    feature: best_triple.0 as u32,
                    feature2: best_triple.1 as u32,
                    feature3: best_triple.2 as u32,
                    n_bins: best_triple_n_bins,
                    pair_stride: best_triple_stride2,
                    triple_stride: best_triple_stride3,
                });
            } else if best_pair.1 != usize::MAX
                && best_pair_gain.is_finite()
                && best_pair_gain > best_single_gain
            {
                choices[node] = Some(GuidedCatChoice {
                    feature: best_pair.0 as u32,
                    feature2: best_pair.1 as u32,
                    feature3: u32::MAX,
                    n_bins: best_pair_n_bins,
                    pair_stride: best_pair_stride,
                    triple_stride: 0,
                });
            } else {
                choices[node] = Some(GuidedCatChoice {
                    feature: best_feat as u32,
                    feature2: u32::MAX,
                    feature3: u32::MAX,
                    n_bins: binned.cll_n_bins[best_feat].max(1),
                    pair_stride: 0,
                    triple_stride: 0,
                });
            }
        }

        choices
    }

    pub(super) fn compute_multiclass_joint_guided_lookups(
        &self,
        tree: &DecisionTree,
        binned: &BinnedData,
        y: &[f64],
        probs: &[f64],
        indices: &[u32],
        n_classes: usize,
        label_smooth: f64,
        guided_choices: &[Option<GuidedCatChoice>],
        base_values: &[f64],
    ) -> Vec<Vec<Option<CatLookup>>> {
        let n_nodes = tree.values.len();
        let mut lookups = vec![vec![None; n_nodes]; n_classes];
        if self.cat_lookup_smooth <= 0.0
            || n_classes < 3
            || guided_choices.len() != n_nodes
            || base_values.len() != n_nodes * n_classes
        {
            return lookups;
        }

        let mut leaf_samples: Vec<Vec<u32>> = vec![Vec::new(); n_nodes];
        for &idx in indices {
            let leaf = tree.route_to_leaf(binned, idx as usize);
            leaf_samples[leaf].push(idx);
        }

        let inv_k = 1.0 / n_classes as f64;
        let n_rows = binned.n_rows;
        for node in 0..n_nodes {
            if tree.split_features[node] != u32::MAX {
                continue;
            }
            let Some(choice) = guided_choices[node].as_ref() else {
                continue;
            };
            let samples = &leaf_samples[node];
            if samples.len() < (2 * n_classes).max(8) {
                continue;
            }

            let feat = choice.feature as usize;
            if feat >= binned.n_features {
                continue;
            }
            let feat2 = choice.feature2 as usize;
            let use_pair = choice.feature2 != u32::MAX;
            let feat3 = choice.feature3 as usize;
            let use_triple = choice.feature3 != u32::MAX;
            if use_pair && feat2 >= binned.n_features {
                continue;
            }
            if use_triple && feat3 >= binned.n_features {
                continue;
            }

            let n_bins = choice.n_bins.max(1);
            let mut g_sums = vec![0.0f64; n_bins * n_classes];
            let mut h_sums = vec![0.0f64; n_bins * n_classes * n_classes];
            let mut diag_sums = vec![0.0f64; n_bins];
            let mut bin_counts = vec![0usize; n_bins];
            let off1 = feat * n_rows;
            let off2 = if use_pair { feat2 * n_rows } else { 0 };
            let off3 = if use_triple { feat3 * n_rows } else { 0 };

            for &idx in samples {
                let row = idx as usize;
                let b1 = binned.cll_hash_bins[off1 + row];
                if b1 == MISSING_BIN {
                    continue;
                }
                let bin = if use_pair {
                    let b2 = binned.cll_hash_bins[off2 + row];
                    if b2 == MISSING_BIN {
                        continue;
                    }
                    if use_triple {
                        let b3 = binned.cll_hash_bins[off3 + row];
                        if b3 == MISSING_BIN {
                            continue;
                        }
                        if choice.pair_stride > 0 && choice.triple_stride > 0 {
                            ((b1 as usize) * choice.pair_stride + b2 as usize)
                                * choice.triple_stride
                                + b3 as usize
                        } else {
                            ((b1 as u32)
                                .wrapping_mul(257)
                                .wrapping_add((b2 as u32).wrapping_mul(17))
                                .wrapping_add(b3 as u32)) as usize
                                % n_bins
                        }
                    } else if choice.pair_stride > 0 {
                        (b1 as usize) * choice.pair_stride + b2 as usize
                    } else {
                        ((b1 as u32).wrapping_mul(257).wrapping_add(b2 as u32)) as usize % n_bins
                    }
                } else {
                    let bu = b1 as usize;
                    if bu >= n_bins {
                        continue;
                    }
                    bu
                };

                bin_counts[bin] += 1;
                let row_base = row * n_classes;
                let g_base = bin * n_classes;
                let h_base = bin * n_classes * n_classes;
                for a in 0..n_classes {
                    let p_a = probs[row_base + a];
                    let hard_label = if y[row] as usize == a { 1.0 } else { 0.0 };
                    let label = if label_smooth > 0.0 {
                        (1.0 - label_smooth) * hard_label + label_smooth * inv_k
                    } else {
                        hard_label
                    };
                    g_sums[g_base + a] += p_a - label;
                    for b in 0..n_classes {
                        let p_b = probs[row_base + b];
                        let h = if a == b {
                            p_a * (1.0 - p_a)
                        } else {
                            -p_a * p_b
                        };
                        h_sums[h_base + a * n_classes + b] += h;
                        if a == b {
                            diag_sums[bin] += h;
                        }
                    }
                }
            }

            let active_bins = diag_sums
                .iter()
                .filter(|&&h| h >= self.min_child_weight)
                .count();
            if active_bins < 2 {
                continue;
            }

            let mut per_class_bin_values = vec![vec![0.0f64; n_bins]; n_classes];
            for k in 0..n_classes {
                let base = base_values[node * n_classes + k];
                for b in 0..n_bins {
                    per_class_bin_values[k][b] = base;
                }
            }

            for b in 0..n_bins {
                if diag_sums[b] < self.min_child_weight {
                    continue;
                }
                let h_base = b * n_classes * n_classes;
                let mut a = h_sums[h_base..h_base + n_classes * n_classes].to_vec();
                let lambda_eff = (self.lambda_reg
                    + self.lambda_reg / (bin_counts[b] as f64).max(1.0).sqrt())
                .max(1e-6);
                for k in 0..n_classes {
                    a[k * n_classes + k] += lambda_eff;
                }
                let mut rhs = vec![0.0f64; n_classes];
                let g_base = b * n_classes;
                for k in 0..n_classes {
                    rhs[k] = -g_sums[g_base + k];
                }
                let mut sol = solve_spd(n_classes, &a, &rhs);
                let mean = sol.iter().sum::<f64>() / n_classes as f64;
                for v in sol.iter_mut() {
                    *v -= mean;
                }
                let mut dec_sq = 0.0f64;
                for k in 0..n_classes {
                    dec_sq += -g_sums[g_base + k] * sol[k];
                }
                let dec_cap = if self.newton_decrement_cap > 0.0 {
                    self.newton_decrement_cap
                } else {
                    1.0
                };
                if dec_sq.is_finite() && dec_sq > dec_cap * dec_cap {
                    let scale = dec_cap / dec_sq.sqrt();
                    for v in sol.iter_mut() {
                        *v *= scale;
                    }
                }
                let smooth = self.cat_lookup_smooth.max(0.0);
                let mass = diag_sums[b];
                for k in 0..n_classes {
                    let base = base_values[node * n_classes + k];
                    per_class_bin_values[k][b] = if smooth > 0.0 {
                        (mass * sol[k] + smooth * base) / (mass + smooth)
                    } else {
                        sol[k]
                    };
                }
            }

            for k in 0..n_classes {
                lookups[k][node] = Some(CatLookup {
                    feature: choice.feature,
                    feature2: choice.feature2,
                    feature3: choice.feature3,
                    bin_values: per_class_bin_values[k].clone(),
                    default_value: base_values[node * n_classes + k],
                    is_numeric: false,
                    n_coarse_bins: 0,
                    pair_stride: choice.pair_stride,
                    triple_stride: choice.triple_stride,
                });
            }
        }

        lookups
    }
}
