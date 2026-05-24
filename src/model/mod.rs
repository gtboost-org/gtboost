use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;

use crate::tree::{BinnedData, DecisionTree};

mod internals;
mod multiclass;
mod refine;
mod training;

type CachedEvalData = Option<(Vec<u16>, Vec<f64>, usize, Vec<f64>, Vec<u16>)>;
const MODEL_FORMAT_VERSION: u32 = 1;

/// Reusable pre-binned training/eval matrix for HPO.
///
/// This caches the expensive quantile/category bin construction once, then
/// lets many GTBoostModel configs train against identical bins. It is intended
/// for the common path where PCF/interval/search features are already
/// materialized in the input matrix before Rust training starts.
#[pyclass]
pub struct GTBoostDataset {
    x_data: Vec<f64>,
    y_data: Vec<f64>,
    n_rows: usize,
    n_features_raw: usize,
    binned: BinnedData,
    eval_data: CachedEvalData,
    cat_features: Vec<bool>,
    num_bins: usize,
    max_cat_bins: usize,
}

#[pymethods]
impl GTBoostDataset {
    #[new]
    #[pyo3(signature = (x, y, cat_features = Vec::new(), num_bins = 256, max_cat_bins = 0, eval_x = None, eval_y = None, split_pessimism = 0.0))]
    pub fn new(
        x: Bound<'_, PyAny>,
        y: Bound<'_, PyAny>,
        cat_features: Vec<bool>,
        num_bins: usize,
        max_cat_bins: usize,
        eval_x: Option<Bound<'_, PyAny>>,
        eval_y: Option<Bound<'_, PyAny>>,
        split_pessimism: f64,
    ) -> PyResult<Self> {
        let x_array: Bound<'_, PyArray2<f64>> = x.extract()?;
        let y_array: Bound<'_, PyArray1<f64>> = y.extract()?;
        let n_rows = x_array.shape()[0];
        let n_features = x_array.shape()[1];
        let y_data: Vec<f64> = y_array.to_owned_array().into_raw_vec_and_offset().0;
        if y_data.len() != n_rows {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "y length ({}) must equal X rows ({})",
                y_data.len(),
                n_rows
            )));
        }

        let mut cat_mask = cat_features;
        if cat_mask.is_empty() {
            cat_mask = vec![false; n_features];
        } else if cat_mask.len() != n_features {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "cat_features length ({}) must equal X width ({})",
                cat_mask.len(),
                n_features
            )));
        }

        let x_owned = x_array.to_owned_array();
        let x_standard = if x_owned.is_standard_layout() {
            x_owned
        } else {
            x_owned.as_standard_layout().into_owned()
        };
        let x_data: Vec<f64> = x_standard.into_raw_vec_and_offset().0;
        let effective_bins = num_bins.min(32.max(n_rows / 4));
        let mut binned = BinnedData::new(
            &x_data,
            n_rows,
            n_features,
            effective_bins,
            &cat_mask,
            max_cat_bins,
        );
        binned.split_pessimism = split_pessimism.max(0.0);

        let eval_data: CachedEvalData = match (eval_x, eval_y) {
            (Some(ex), Some(ey)) => {
                let ex_array: Bound<'_, PyArray2<f64>> = ex.extract()?;
                let ey_array: Bound<'_, PyArray1<f64>> = ey.extract()?;
                let en_rows = ex_array.shape()[0];
                let en_features = ex_array.shape()[1];
                if en_features != n_features {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "eval_x width ({}) must equal X width ({})",
                        en_features, n_features
                    )));
                }
                let ey_data: Vec<f64> = ey_array.to_owned_array().into_raw_vec_and_offset().0;
                if ey_data.len() != en_rows {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "eval_y length ({}) must equal eval_x rows ({})",
                        ey_data.len(),
                        en_rows
                    )));
                }
                let ex_owned = ex_array.to_owned_array();
                let ex_standard = if ex_owned.is_standard_layout() {
                    ex_owned
                } else {
                    ex_owned.as_standard_layout().into_owned()
                };
                let ex_data: Vec<f64> = ex_standard.into_raw_vec_and_offset().0;
                let eval_bins = BinnedData::bin_with_edges(
                    &ex_data,
                    en_rows,
                    n_features,
                    &binned.bin_edges,
                    &binned.is_categorical,
                );
                let eval_cll_hash_bins = BinnedData::build_cll_hash_bins(
                    &ex_data,
                    en_rows,
                    n_features,
                    &cat_mask,
                    &binned.is_categorical,
                    &binned.bin_edges,
                );
                Some((eval_bins, ey_data, en_rows, ex_data, eval_cll_hash_bins))
            }
            (None, None) => None,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "eval_x and eval_y must be provided together",
                ));
            }
        };

        Ok(Self {
            x_data,
            y_data,
            n_rows,
            n_features_raw: n_features,
            binned,
            eval_data,
            cat_features: cat_mask,
            num_bins,
            max_cat_bins,
        })
    }

    #[getter]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    #[getter]
    pub fn n_features(&self) -> usize {
        self.binned.n_features
    }

    #[getter]
    pub fn n_features_raw(&self) -> usize {
        self.n_features_raw
    }

    #[getter]
    pub fn has_eval(&self) -> bool {
        self.eval_data.is_some()
    }
}

#[pyclass(skip_from_py_object)]
#[derive(Clone, Serialize, Deserialize)]
pub struct GTBoostModel {
    pub(super) max_depth: usize,
    pub(super) learning_rate: f64,
    pub(super) subsample_rate: f64,
    pub(super) base_score: f64,
    pub(super) class_base_scores: Vec<f64>, // multiclass prior margins (log class frequencies, centered)
    pub(super) lambda_reg: f64,
    pub(super) gamma: f64,
    pub(super) min_child_weight: f64,
    pub(super) colsample_bytree: f64,
    pub(super) task: String,
    pub(super) num_bins: usize,
    pub(super) seed: u64,
    pub(super) grow_policy: String, // "depthwise", "leafwise", or "oblivious"
    pub(super) max_leaves: usize,   // max leaves for leafwise (0 = use 2^max_depth)
    pub(super) n_refine: usize,     // refinement passes (0 = disabled)
    pub(super) n_leaf_splits: usize, // post-refinement leaf splitting passes (0 = disabled)
    pub(super) refine_every: usize, // refine all trees every N rounds during training (0 = only at end)
    pub(super) early_stopping_rounds: usize, // 0 = disabled
    pub(super) l1_reg: f64, // L1 regularization on leaf values during refinement (0 = disabled)
    pub(super) refine_alpha: f64, // refinement shrinkage: blend w_old + alpha*(w_new - w_old), 1.0 = full step
    pub(super) honest: bool, // honest estimation: build structure on half data, leaf values on other half
    pub(super) honest_fraction: f64, // fraction of subsampled data for estimation (0.0 = use complement of subsample)
    pub(super) colsample_bylevel: f64, // column subsampling ratio per tree depth level (1.0 = disabled)
    pub(super) lr_decay: f64, // learning rate decay: effective lr linearly decays to lr*lr_decay by end (1.0 = no decay)
    pub(super) n_trees_per_round: usize, // build N trees per round with averaged contributions (1 = standard)
    pub(super) cat_features: Vec<bool>,  // which features are categorical
    pub(super) grad_clip: f64,           // gradient clipping threshold (0.0 = disabled)
    pub(super) random_strength: f64, // noise added to split gains for diversity (0.0 = disabled)
    pub(super) bagging_temperature: f64, // Bayesian bootstrap temperature (0.0 = disabled, 1.0 = standard)
    pub(super) huber_delta: f64,         // Huber loss delta for regression (0.0 = use MSE)
    pub(super) cat_smooth: f64, // categorical sort smoothing (0.0 = disabled, >0 = shrink rare categories toward node mean)
    pub(super) cat_lookup_smooth: f64, // CLL: per-category leaf values (0.0 = disabled, >0 = smooth toward leaf value)
    pub(super) monotone_constraints: Vec<i8>, // per-feature monotone constraints: 0=none, 1=increasing, -1=decreasing
    pub(super) max_cat_bins: usize, // max categories for native categorical splits (0 = unlimited)
    pub(super) ramp: bool, // piecewise linear refinement: fit slope on parent's split feature per leaf
    pub(super) ramp_lambda: f64, // L2 penalty on ramp slopes (higher = more conservative)
    pub(super) ramp_k: usize, // number of ramp features per leaf (1 = parent only, 2+ = path features)
    pub(super) leaf_linear: bool, // leaf linear model: fit ridge regression on ALL numeric features per leaf
    pub(super) leaf_quadratic: bool, // add pairwise X_i*X_j interaction terms to leaf linear model
    pub(super) leaf_correction: usize, // intra-round gradient correction passes (0 = disabled)
    pub(super) grad_momentum: f64, // Gradient momentum: blend grad_t = (1-mom)*fresh + mom*prev_grad (0.0 = disabled)
    pub(super) gain_penalty: f64, // Bayesian gain penalty: penalize splits with uncertain children (0.0 = disabled)
    pub(super) split_pessimism: f64, // Evidence-corrected split gain: discounts low-support, high-search-width split wins.
    pub(super) self_score_splits: bool, // Allow trees to split on the current boosting margin.
    pub(super) hetero_trees: bool, // Heterogeneous sub-trees: cycle (depth, lambda) across sub-trees for structural diversity
    pub(super) dart_rate: f64, // DART: fraction of trees to drop per round during training (0.0 = disabled)
    pub(super) max_delta_step: f64, // Max leaf value magnitude (0.0 = unlimited, >0 clips leaf values)
    pub(super) cyclic_features: bool, // EBM-style: cycle through features, each tree uses one feature (false = disabled)
    pub(super) auto_interactions: bool, // Auto-generate pairwise numeric product features in binning (false = disabled)
    pub(super) auto_cat_interactions: bool, // Auto-generate hashed categorical pair features in binning (false = disabled)
    pub(super) max_interaction_features: usize, // Max product feature pairs to generate (0 = unlimited)
    pub(super) lambda_schedule: f64, // Adaptive lambda: effective_lambda = lambda * (1 + lambda_schedule * round/n_rounds) (0.0 = off)
    pub(super) use_bootstrap: bool, // Bootstrap sampling: sample rows with replacement (RF-style bagging)
    pub(super) extra_trees: bool, // Extra Trees: random split thresholds instead of optimal (massive variance reduction)
    pub(super) label_smooth: f64, // Label smoothing for multiclass: target = (1-ε)*one_hot + ε/K (0.0 = off)
    pub(super) multi_output_tree: bool, // Multi-output trees for multiclass: shared tree structure across all K classes
    pub(super) prob_avg: bool, // Probability averaging: softmax each sub-tree independently then average (RF-like prediction)
    pub(super) honest_tau: f64, // Bayesian leaf blending: tau > 0 blends complement estimate with structure prior
    pub(super) complement_debias_mode: u8, // CDSS: 0=off, 1=geomean(struct, honest), 2=min, 3=mean
    pub(super) phase_schedule: String, // Staged complexity: "0.3:1,0.6:2,1.0:full" (empty = off)
    pub(super) ncl_lambda: f64, // Negative correlation learning: diversify ntp sub-trees (0.0 = disabled)
    pub(super) adaptive_cyclic_order: bool, // SCGB: choose cyclic feature order from current residual pressure instead of fixed rotation.
    pub(super) cyclic_partner_features: bool, // CIPA: cyclic tree may include one residual-chosen partner feature for interactions.
    pub(super) cyclic_partner_min_pressure_ratio: f64, // Require pair pressure >= ratio * primary pressure to add partner.
    pub(super) cyclic_partner_bins: usize, // Coarse bins per feature for residual pair-pressure scoring.
    pub(super) cyclic_feature_reuse: bool, // Residual feature auction: adaptive cyclic slots may reuse high-pressure features within a round.
    pub(super) cyclic_revisit_trees: usize, // Extra residual-auction revisit trees after all features were represented once.
    pub(super) cyclic_revisit_min_pressure_ratio: f64, // Skip revisit tail when residual feature pressure collapsed vs round start.
    pub(super) adaptive_feature_mask: bool, // Pressure masks: sibling regular trees get residual-ranked feature subsets.
    pub(super) adaptive_feature_mask_penalty: f64, // Diversity penalty for reusing a feature in pressure masks within the same round.
    pub(super) adaptive_root_anchor: bool, // Pressure anchors: force depthwise root split to the current strongest residual feature.
    pub(super) adaptive_root_anchor_penalty: f64, // Diversity penalty for reusing a root anchor within the same round.
    pub(super) sparse_oblique_splits: bool, // Sparse 2-feature oblique split candidates for depthwise trees.
    pub(super) interval_splits: bool, // Bounded numeric interval split candidates: low <= x_j <= high.
    pub(super) sibling_block_correction: f64, // Joint per-round least-squares rescale for sibling trees (0.0 = disabled).
    pub(super) adam_beta2: f64, // Adam 2nd-moment decay (0.0 = disabled). Uses grad_momentum as β1.
    pub(super) adam_eps: f64,   // Adam epsilon for stability in denom
    pub(super) ortho_alpha: f64, // Gradient orthogonalization vs previous tree's leaves (0.0 = disabled)
    pub(super) split_criterion: String, // "newton" (default), "rank" (Wilcoxon-like), or "sign" (distribution-free)
    pub(super) rank_mix_alpha: f64, // MGB: for task="binary", blend α·g_rank + (1-α)·g_logloss. 0 = pure binary.
    pub(super) rank_mix_start_frac: f64, // Late-MGB: delay rank-mix until this training fraction. 0 = active from start.
    pub(super) binary_focus_gamma: f64, // Hard-row focus for binary loss: multiply g/h by (2*|y-p|)^gamma. 0 = off.
    pub(super) binary_focus_end_frac: f64, // Focus warmup: if >0, turn binary_focus_gamma off after this training fraction.
    pub(super) feature_view_groups: Vec<u32>, // Optional feature-group id per column. When set, subtrees cycle across groups instead of sampling from the full feature set.
    pub(super) leaf_trim_pct: f64, // Trimmed-mean leaf values (Huber-style robust). 0.0 = pure Newton; 0.1 = trim 10% from each tail.
    pub(super) leaf_median: bool, // §124 LAD-style: use weighted median of (-g/h) per leaf instead of Newton mean. Classical Friedman 1999 LAD-TreeBoost.
    pub(super) leaf_median_blend: f64, // §124b blend: w = (1-α)·newton + α·median; 0 = pure newton, 1 = pure LAD. Supersedes leaf_median when > 0.
    pub(super) leaf_mad_clip: f64, // Adaptive robust leaf refit: clip per-row Newton targets using leaf MAD scale. 0.0 = off, 3-5 = classical robust range.
    pub(super) leaf_adaptive_blend_kappa: f64, // Self-diagnosed robust leaf bridge: move toward median when mean and median disagree in MAD units.
    pub(super) ordered_boost: bool, // CatBoost-style ordered boosting: K shadow buckets, unbiased gradient computation
    pub(super) ordered_n_buckets: usize, // Number of shadow buckets for ordered boosting (default 4)
    pub(super) goss_top_rate: f64, // NC-GOSS: base fraction of rows to keep by importance (0.0 = disabled)
    pub(super) goss_other_rate: f64, // NC-GOSS: fraction of remaining rows to random-sample
    pub(super) goss_mode: String, // "classic" (|g|), "newton" (g²/h — default for NC-GOSS), "classmax" (max_k|g_k|)
    pub(super) goss_anneal: f64, // Round-annealed focus: top_rate linearly decays from goss_top_rate+goss_anneal to goss_top_rate (0.0 = no anneal)
    pub(super) keep_all_trees: bool, // PASA: if true, don't truncate trees post-ES (keep plateau history for averaging)
    pub(super) corrective_block_refit: bool, // CBR: post-fit ridge refit of contiguous tree-block coefficients for regression.
    pub(super) corrective_blocks: usize,     // CBR: number of tree blocks in the corrective basis.
    pub(super) corrective_lambda: f64,       // CBR: ridge strength scaled by mean block energy.
    pub(super) corrective_blend: f64, // CBR: 0=standard weights, 1=fully refit block weights.
    pub(super) corrective_min_trees: usize, // CBR: skip tiny ensembles where coefficient refit is not identifiable.
    pub(super) corrective_audit_fraction: f64, // CBR: holdout fraction for accepting/rejecting refit weights.
    pub(super) corrective_min_rel_improve: f64, // CBR: minimum audit SSE improvement required to apply weights.
    pub(super) leaf_eb: bool, // EBLP: empirical Bayes leaf prior (data-adaptive James-Stein shrinkage)
    pub(super) leaf_eb_min_trees: usize, // EBLP: min trees before activating shrinkage (need variance estimate)
    pub(super) leaf_eb_scale: f64, // EBLP: multiplier on estimated τ_π (1.0 = standard James-Stein)
    pub(super) leaf_sibling_smooth: f64, // HSS: Hierarchical Sibling Smoothing — blend each leaf with its same-parent sibling (0.0 = off)
    pub(super) hierarchical_shrinkage: f64, // HES: shrink leaf experts toward ancestor predictions using node evidence
    pub(super) multiclass_coupled_leaves: bool, // experimental: coupled softmax node fitting on shared-tree multiclass path
    pub(super) multiclass_joint_cll: bool, // §130: joint (coupled softmax) CLL install — fills joint_lookup_tables via compute_multiclass_joint_guided_lookups.
    pub(super) class_weights: Vec<f64>, // §131: per-class gradient reweighting for imbalanced multiclass (website, anneal).
    pub(super) adaptive_leaf_experts: bool, // ALE: per-leaf choose plain value vs categorical/numeric lookup post-fit
    pub(super) adaptive_cat_lookup_smooth: bool, // EB-CLL: derive lookup smoothing from leaf-local category dispersion.
    pub(super) cat_offset_smooth: f64, // CGO: post-fit global categorical Newton offsets (0.0 = disabled).
    pub(super) cat_offset_passes: usize, // CGO: coordinate-descent passes over categorical columns.
    pub(super) cat_offset_maps: Vec<HashMap<i64, f64>>, // CGO: feature -> category -> additive margin offset.
    pub(super) ordered_ctr: bool, // Ordered target statistics for categorical features.
    pub(super) ordered_ctr_top_features: usize, // Number of categorical columns to CTR-encode.
    pub(super) ordered_ctr_smooth: f64, // Smoothing strength toward the global target prior.
    pub(super) ordered_ctr_permutations: usize, // Number of ordered permutations averaged for train rows.
    pub(super) ordered_ctr_min_count: usize, // Minimum non-missing category count to consider a feature.
    pub(super) ordered_ctr_features: Vec<usize>, // Trained source feature indices.
    pub(super) ordered_ctr_prior: f64,       // Trained global target prior.
    pub(super) ordered_ctr_maps: Vec<HashMap<i64, f64>>, // Per-feature full-data category -> smoothed mean.
    pub(super) ordered_ctr_count_maps: Vec<HashMap<i64, f64>>, // Per-feature category -> log1p(full-train count).
    pub(super) ordered_ctr_pair_features: Vec<(usize, usize)>, // GCE: trained categorical pair sources.
    pub(super) ordered_ctr_pair_maps: Vec<HashMap<i64, f64>>,  // GCE: pair key -> smoothed mean.
    pub(super) ordered_ctr_pair_count_maps: Vec<HashMap<i64, f64>>, // GCE: pair key -> log1p(count).
    pub(super) ordered_ctr_triple_features: Vec<(usize, usize, usize)>, // GCE: trained categorical triple sources.
    pub(super) ordered_ctr_triple_maps: Vec<HashMap<i64, f64>>, // GCE: triple key -> smoothed mean.
    pub(super) ordered_ctr_triple_count_maps: Vec<HashMap<i64, f64>>, // GCE: triple key -> log1p(count).
    pub(super) cat_tuple_lookups: bool, // CTL: per-leaf categorical pair/triple lookup experts.
    pub(super) cat_tuple_max_order: usize, // CTL: max tuple order (2=pair, 3=triple).
    pub(super) cat_tuple_top_features: usize, // CTL: per-leaf top single cat features used for tuple candidates.
    pub(super) cat_tuple_hash_bins: usize, // CTL: hashed bins for high-card pair/triple candidates.
    pub(super) cat_tuple_min_leaf: usize, // CTL: minimum rows in a leaf before tuple candidates are evaluated.
    pub(super) cat_tuple_gain_margin: f64, // CTL: relative score margin over incumbent expert.
    pub(super) expert_leaf_admission: bool, // VCEG-0: admit tiny ridge leaf experts only when calibration gain beats scalar.
    pub(super) expert_max_terms: usize,     // VCEG-0: max numeric slope terms per admitted leaf.
    pub(super) expert_min_leaf: usize,      // VCEG-0: minimum build rows in a leaf.
    pub(super) expert_min_cal: usize,       // VCEG-0: minimum calibration rows in a leaf.
    pub(super) expert_ridge_lambda: f64,    // VCEG-0: ridge penalty for local slopes.
    pub(super) expert_alpha_max: f64,       // VCEG-0: calibration shrinkage cap.
    pub(super) expert_param_penalty: f64,   // VCEG-0: pessimism per extra local term.
    pub(super) expert_se_multiplier: f64,   // VCEG-0: pessimism by SE of per-row gain difference.
    pub(super) expert_epsilon: f64,         // VCEG-0: tiny-win guard.
    pub(super) expert_shadow_trials: usize, // NC-VCEG: node-local shadow experts as null controls.
    pub(super) antithetic_subtrees: bool, // balanced cyclic row coverage across ntp subtrees within a round
    pub(super) newton_decrement_cap: f64, // adaptive leaf trust-region cap: |w| <= cap/sqrt(h+lambda)
    pub(super) lookahead_alpha: f64, // LAS: 1-step look-ahead split selection. score = gain + α·max(child_L_gain, child_R_gain). 0 = greedy, typical 0.2-0.5.
    pub(super) sign_confidence_gamma: f64, // SCS: shrink leaf by (|Σsign(g)|/n)^γ. Mixed-sign leaves → low confidence → shrink toward 0. 0 = off, typical 0.5-2.0.
    pub(super) soft_predict_bandwidth: f64, // SRP: soft routing at predict time. σ((thresh - val) / (bw · feat_scale)). 0 = hard routing, typical 0.3-1.0.
    pub(super) jensen_train_temp: f64, // Jensen-aware training: divide predictions by T before softmax during multiclass gradient computation. T=1 = off, T>1 aligns training with test-time PD smoothing (Jensen gap).
    pub(super) diversity_penalty: f64, // §2: feature-usage EMA penalty on make_feature_mask. Features used in recent trees get lower inclusion prob. 0 = off, 0.2-0.8 typical.
    pub(super) diversity_decay: f64, // §2: EMA decay per tree for feature usage tracking. 0.9 = ~10-tree memory, 0.95 = ~20-tree memory.
    pub(super) feature_usage_ema: Vec<f64>, // Trained state: per-feature usage EMA (transient, reset at fit start).
    // GGFP v5.0 — JIT-CatPairSplit
    pub(super) jit_catpair_enabled: bool,
    pub(super) jit_catpair_top_k: usize,
    pub(super) jit_catpair_k_buckets: u8,
    pub(super) jit_catpair_min_node_rows: usize,
    pub(super) jit_catpair_max_node_depth: usize,
    pub(super) jit_catpair_gain_margin: f64,
    // GGFP v6 (LTSO) — virtual feature registry infrastructure.
    // Phase 2 test: when true, registers an Identity(0) virtual feature
    // at fit start. Validates that mid-fit BinnedData growth works.
    pub(super) jit_ltso_enabled: bool,
    // Trained state
    pub(super) val_losses: Vec<f64>, // PASA: val loss per round (populated during training when eval_data given)
    pub(super) best_round: usize, // PASA: round index of best val score (0-based, exclusive upper bound on trees)
    pub(super) trees: Vec<DecisionTree>,
    pub(super) dart_tree_weights: Vec<f64>, // DART per-tree weights (empty = all 1.0)
    pub(super) binned: Option<BinnedData>,
    pub(super) n_features: usize,
    pub(super) n_classes: usize, // 0 for regression/binary, K for multiclass
    pub(super) multiclass_trees_per_class: usize, // actual trees-per-class-per-round used by current multiclass model
    pub(super) multiclass_tree_lr_scale: f64, // effective per-tree lr multiplier for multiclass ensembles
    pub(super) tree_in_sample: Vec<Vec<u64>>, // per-tree in-sample masks for honest refine (packed bits)
    // Numeric interaction feature state
    pub(super) numeric_interaction_pairs: Vec<(usize, usize)>, // (feat_i, feat_j) pairs for product features
    pub(super) numeric_interaction_edges: Vec<Vec<f64>>,       // bin edges for product features
    pub(super) categorical_interaction_pairs: Vec<(usize, usize)>, // (feat_i, feat_j) pairs for hashed categorical combos
    pub(super) categorical_interaction_edges: Vec<Vec<f64>>, // category lists for hashed categorical combos
    // Sum/difference augmentation feature state
    pub(super) sumdiff_pairs: Vec<(usize, usize)>, // (feat_i, feat_j) pairs for sum/diff features
    pub(super) sumdiff_edges: Vec<Vec<f64>>,       // bin edges for sum/diff features
}

#[derive(Serialize, Deserialize)]
struct ModelSnapshot {
    format_version: u32,
    model: GTBoostModel,
}

// Stateless helpers (bitvec, linear-system solvers, gradient transforms)
// live in `crate::helpers`. Boosting internals (training, gradients, leaf
// finalization, per-tree adjustments) live in submodule `internals`.
// Post-hoc global refinement lives in submodule `refine`.

impl GTBoostModel {
    fn clear_trained_state_for_fit(&mut self) {
        self.trees.clear();
        self.tree_in_sample.clear();
        self.dart_tree_weights.clear();
        self.val_losses.clear();
        self.best_round = 0;
        self.n_classes = 0;
        self.multiclass_trees_per_class = 1;
        self.multiclass_tree_lr_scale = 1.0;
        self.cat_offset_maps.clear();
        self.numeric_interaction_pairs.clear();
        self.numeric_interaction_edges.clear();
        self.categorical_interaction_pairs.clear();
        self.categorical_interaction_edges.clear();
        self.sumdiff_pairs.clear();
        self.sumdiff_edges.clear();
        self.feature_usage_ema.clear();
    }

    fn compute_base_scores_for_fit(
        &mut self,
        y_data: &[f64],
        n_rows: usize,
        sample_weight_data: Option<&[f64]>,
    ) {
        let y_mean = if let Some(sw) = sample_weight_data {
            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..n_rows {
                let w = if sw[i].is_finite() {
                    sw[i].max(0.0)
                } else {
                    0.0
                };
                num += w * y_data[i];
                den += w;
            }
            if den > 0.0 {
                num / den
            } else {
                y_data.iter().sum::<f64>() / n_rows as f64
            }
        } else {
            y_data.iter().sum::<f64>() / n_rows as f64
        };
        match self.task.as_str() {
            "regression" => {
                self.base_score = y_mean;
                self.class_base_scores.clear();
            }
            "binary" => {
                let p = if self.class_weights.len() >= 2 {
                    let neg_w = if self.class_weights[0].is_finite() && self.class_weights[0] > 0.0
                    {
                        self.class_weights[0]
                    } else {
                        1.0
                    };
                    let pos_w = if self.class_weights[1].is_finite() && self.class_weights[1] > 0.0
                    {
                        self.class_weights[1]
                    } else {
                        1.0
                    };
                    let (pos, neg) = if let Some(sw) = sample_weight_data {
                        let mut pos = 0.0;
                        let mut neg = 0.0;
                        for i in 0..n_rows {
                            let w = if sw[i].is_finite() {
                                sw[i].max(0.0)
                            } else {
                                0.0
                            };
                            if y_data[i] > 0.5 {
                                pos += w;
                            } else {
                                neg += w;
                            }
                        }
                        (pos, neg)
                    } else {
                        let pos = y_data.iter().filter(|&&v| v > 0.5).count() as f64;
                        (pos, n_rows as f64 - pos)
                    };
                    let denom = pos_w * pos + neg_w * neg;
                    if denom > 0.0 {
                        (pos_w * pos / denom).clamp(1e-6, 1.0 - 1e-6)
                    } else {
                        y_mean.clamp(1e-6, 1.0 - 1e-6)
                    }
                } else {
                    y_mean.clamp(1e-6, 1.0 - 1e-6)
                };
                self.base_score = (p / (1.0 - p)).ln();
                self.class_base_scores.clear();
            }
            "poisson" => {
                self.base_score = y_mean.max(1e-6).ln();
                self.class_base_scores.clear();
            }
            "multiclass" => {
                let n_classes = y_data.iter().map(|&v| v as usize).max().unwrap_or(0) + 1;
                let alpha = 1.0;
                let denom = n_rows as f64 + alpha * n_classes as f64;
                let mut counts = vec![alpha; n_classes];
                for &yi in y_data {
                    let k = yi as usize;
                    if k < n_classes {
                        counts[k] += 1.0;
                    }
                }
                self.class_base_scores = counts.iter().map(|&c| (c / denom).ln()).collect();
                let mean_margin = self.class_base_scores.iter().sum::<f64>() / n_classes as f64;
                for v in self.class_base_scores.iter_mut() {
                    *v -= mean_margin;
                }
            }
            _ => {}
        }
    }
}

#[pymethods]
impl GTBoostModel {
    #[new]
    #[pyo3(signature = (
        learning_rate = 0.3,
        max_depth = 6,
        subsample_rate = 1.0,
        base_score = 0.0,
        lambda_reg = 1.0,
        gamma = 0.0,
        min_child_weight = 1.0,
        colsample_bytree = 1.0,
        task = "regression".to_string(),
        num_bins = 256,
        seed = None,
        grow_policy = "depthwise".to_string(),
        max_leaves = 0,
        n_refine = 0,
        n_leaf_splits = 0,
        refine_every = 0,
        early_stopping_rounds = 0,
        l1_reg = 0.0,
        refine_alpha = 1.0,
        honest = false,
        honest_fraction = 0.5,
        colsample_bylevel = 1.0,
        lr_decay = 1.0,
        n_trees_per_round = 1,
        cat_features = Vec::new(),
        grad_clip = 0.0,
        random_strength = 0.0,
        bagging_temperature = 0.0,
        huber_delta = 0.0,
        cat_smooth = 0.0,
        cat_lookup_smooth = 0.0,
        monotone_constraints = Vec::new(),
        max_cat_bins = 0,
        ramp = false,
        ramp_lambda = 10.0,
        ramp_k = 1,
        leaf_linear = false,
        leaf_quadratic = false,
        leaf_correction = 0,
        grad_momentum = 0.0,
        gain_penalty = 0.0,
        split_pessimism = 0.0,
        self_score_splits = false,
        hetero_trees = false,
        dart_rate = 0.0,
        max_delta_step = 0.0,
        cyclic_features = false,
        auto_interactions = false,
        auto_cat_interactions = false,
        max_interaction_features = 20,
        lambda_schedule = 0.0,
        use_bootstrap = false,
        extra_trees = false,
        label_smooth = 0.0,
        multi_output_tree = false,
        prob_avg = false,
        honest_tau = 0.0,
        complement_debias_mode = 0,
        phase_schedule = "".to_string(),
        ncl_lambda = 0.0,
        adaptive_cyclic_order = false,
        cyclic_partner_features = false,
        cyclic_partner_min_pressure_ratio = 0.0,
        cyclic_partner_bins = 8,
        cyclic_feature_reuse = false,
        cyclic_revisit_trees = 0,
        cyclic_revisit_min_pressure_ratio = 0.0,
        adaptive_feature_mask = false,
        adaptive_feature_mask_penalty = 0.5,
        adaptive_root_anchor = false,
        adaptive_root_anchor_penalty = 0.5,
        sparse_oblique_splits = false,
        interval_splits = false,
        sibling_block_correction = 0.0,
        adam_beta2 = 0.0,
        adam_eps = 1e-8,
        ortho_alpha = 0.0,
        split_criterion = "newton".to_string(),
        rank_mix_alpha = 0.0,
        rank_mix_start_frac = 0.0,
        binary_focus_gamma = 0.0,
        binary_focus_end_frac = 0.0,
        feature_view_groups = Vec::new(),
        leaf_trim_pct = 0.0,
        leaf_median = false,
        leaf_median_blend = 0.0,
        leaf_mad_clip = 0.0,
        leaf_adaptive_blend_kappa = 0.0,
        ordered_boost = false,
        ordered_n_buckets = 4,
        goss_top_rate = 0.0,
        goss_other_rate = 0.0,
        goss_mode = "newton".to_string(),
        goss_anneal = 0.0,
        keep_all_trees = false,
        corrective_block_refit = false,
        corrective_blocks = 16,
        corrective_lambda = 1.0,
        corrective_blend = 1.0,
        corrective_min_trees = 16,
        corrective_audit_fraction = 0.0,
        corrective_min_rel_improve = 0.0,
        leaf_eb = false,
        leaf_eb_min_trees = 10,
        leaf_eb_scale = 1.0,
        leaf_sibling_smooth = 0.0,
        hierarchical_shrinkage = 0.0,
        multiclass_coupled_leaves = false,
        multiclass_joint_cll = false,
        class_weights = Vec::<f64>::new(),
        adaptive_leaf_experts = false,
        adaptive_cat_lookup_smooth = false,
        cat_offset_smooth = 0.0,
        cat_offset_passes = 0,
        ordered_ctr = false,
        ordered_ctr_top_features = 16,
        ordered_ctr_smooth = 30.0,
        ordered_ctr_permutations = 1,
        ordered_ctr_min_count = 2,
        cat_tuple_lookups = false,
        cat_tuple_max_order = 3,
        cat_tuple_top_features = 5,
        cat_tuple_hash_bins = 128,
        cat_tuple_min_leaf = 64,
        cat_tuple_gain_margin = 0.05,
        expert_leaf_admission = false,
        expert_max_terms = 2,
        expert_min_leaf = 64,
        expert_min_cal = 12,
        expert_ridge_lambda = 25.0,
        expert_alpha_max = 1.0,
        expert_param_penalty = 1e-4,
        expert_se_multiplier = 0.5,
        expert_epsilon = 1e-5,
        expert_shadow_trials = 0,
        antithetic_subtrees = false,
        newton_decrement_cap = 0.0,
        lookahead_alpha = 0.0,
        sign_confidence_gamma = 0.0,
        soft_predict_bandwidth = 0.0,
        jensen_train_temp = 1.0,
        diversity_penalty = 0.0,
        diversity_decay = 0.9,
        jit_catpair_enabled = false,
        jit_catpair_top_k = 4,
        jit_catpair_k_buckets = 8,
        jit_catpair_min_node_rows = 512,
        jit_catpair_max_node_depth = 2,
        jit_catpair_gain_margin = 1.05,
        jit_ltso_enabled = false
    ))]
    pub fn new(
        learning_rate: f64,
        max_depth: usize,
        subsample_rate: f64,
        base_score: f64,
        lambda_reg: f64,
        gamma: f64,
        min_child_weight: f64,
        colsample_bytree: f64,
        task: String,
        num_bins: usize,
        seed: Option<u64>,
        grow_policy: String,
        max_leaves: usize,
        n_refine: usize,
        n_leaf_splits: usize,
        refine_every: usize,
        early_stopping_rounds: usize,
        l1_reg: f64,
        refine_alpha: f64,
        honest: bool,
        honest_fraction: f64,
        colsample_bylevel: f64,
        lr_decay: f64,
        n_trees_per_round: usize,
        cat_features: Vec<bool>,
        grad_clip: f64,
        random_strength: f64,
        bagging_temperature: f64,
        huber_delta: f64,
        cat_smooth: f64,
        cat_lookup_smooth: f64,
        monotone_constraints: Vec<i8>,
        max_cat_bins: usize,
        ramp: bool,
        ramp_lambda: f64,
        ramp_k: usize,
        leaf_linear: bool,
        leaf_quadratic: bool,
        leaf_correction: usize,
        grad_momentum: f64,
        gain_penalty: f64,
        split_pessimism: f64,
        self_score_splits: bool,
        hetero_trees: bool,
        dart_rate: f64,
        max_delta_step: f64,
        cyclic_features: bool,
        auto_interactions: bool,
        auto_cat_interactions: bool,
        max_interaction_features: usize,
        lambda_schedule: f64,
        use_bootstrap: bool,
        extra_trees: bool,
        label_smooth: f64,
        multi_output_tree: bool,
        prob_avg: bool,
        honest_tau: f64,
        complement_debias_mode: u8,
        phase_schedule: String,
        ncl_lambda: f64,
        adaptive_cyclic_order: bool,
        cyclic_partner_features: bool,
        cyclic_partner_min_pressure_ratio: f64,
        cyclic_partner_bins: usize,
        cyclic_feature_reuse: bool,
        cyclic_revisit_trees: usize,
        cyclic_revisit_min_pressure_ratio: f64,
        adaptive_feature_mask: bool,
        adaptive_feature_mask_penalty: f64,
        adaptive_root_anchor: bool,
        adaptive_root_anchor_penalty: f64,
        sparse_oblique_splits: bool,
        interval_splits: bool,
        sibling_block_correction: f64,
        adam_beta2: f64,
        adam_eps: f64,
        ortho_alpha: f64,
        split_criterion: String,
        rank_mix_alpha: f64,
        rank_mix_start_frac: f64,
        binary_focus_gamma: f64,
        binary_focus_end_frac: f64,
        feature_view_groups: Vec<u32>,
        leaf_trim_pct: f64,
        leaf_median: bool,
        leaf_median_blend: f64,
        leaf_mad_clip: f64,
        leaf_adaptive_blend_kappa: f64,
        ordered_boost: bool,
        ordered_n_buckets: usize,
        goss_top_rate: f64,
        goss_other_rate: f64,
        goss_mode: String,
        goss_anneal: f64,
        keep_all_trees: bool,
        corrective_block_refit: bool,
        corrective_blocks: usize,
        corrective_lambda: f64,
        corrective_blend: f64,
        corrective_min_trees: usize,
        corrective_audit_fraction: f64,
        corrective_min_rel_improve: f64,
        leaf_eb: bool,
        leaf_eb_min_trees: usize,
        leaf_eb_scale: f64,
        leaf_sibling_smooth: f64,
        hierarchical_shrinkage: f64,
        multiclass_coupled_leaves: bool,
        multiclass_joint_cll: bool,
        class_weights: Vec<f64>,
        adaptive_leaf_experts: bool,
        adaptive_cat_lookup_smooth: bool,
        cat_offset_smooth: f64,
        cat_offset_passes: usize,
        ordered_ctr: bool,
        ordered_ctr_top_features: usize,
        ordered_ctr_smooth: f64,
        ordered_ctr_permutations: usize,
        ordered_ctr_min_count: usize,
        cat_tuple_lookups: bool,
        cat_tuple_max_order: usize,
        cat_tuple_top_features: usize,
        cat_tuple_hash_bins: usize,
        cat_tuple_min_leaf: usize,
        cat_tuple_gain_margin: f64,
        expert_leaf_admission: bool,
        expert_max_terms: usize,
        expert_min_leaf: usize,
        expert_min_cal: usize,
        expert_ridge_lambda: f64,
        expert_alpha_max: f64,
        expert_param_penalty: f64,
        expert_se_multiplier: f64,
        expert_epsilon: f64,
        expert_shadow_trials: usize,
        antithetic_subtrees: bool,
        newton_decrement_cap: f64,
        lookahead_alpha: f64,
        sign_confidence_gamma: f64,
        soft_predict_bandwidth: f64,
        jensen_train_temp: f64,
        diversity_penalty: f64,
        diversity_decay: f64,
        jit_catpair_enabled: bool,
        jit_catpair_top_k: usize,
        jit_catpair_k_buckets: u8,
        jit_catpair_min_node_rows: usize,
        jit_catpair_max_node_depth: usize,
        jit_catpair_gain_margin: f64,
        jit_ltso_enabled: bool,
    ) -> Self {
        GTBoostModel {
            max_depth,
            learning_rate,
            subsample_rate,
            base_score,
            class_base_scores: Vec::new(),
            lambda_reg,
            gamma,
            min_child_weight,
            colsample_bytree,
            task,
            num_bins,
            seed: seed.unwrap_or(0),
            grow_policy,
            max_leaves,
            n_refine,
            n_leaf_splits,
            refine_every,
            early_stopping_rounds,
            l1_reg,
            refine_alpha,
            honest,
            honest_fraction,
            colsample_bylevel,
            lr_decay,
            n_trees_per_round: n_trees_per_round.max(1),
            cat_features,
            grad_clip,
            random_strength,
            bagging_temperature,
            huber_delta,
            cat_smooth,
            cat_lookup_smooth,
            monotone_constraints,
            max_cat_bins,
            ramp,
            ramp_lambda,
            ramp_k: ramp_k.max(1),
            leaf_linear,
            leaf_quadratic,
            leaf_correction,
            grad_momentum: grad_momentum.clamp(0.0, 0.99),
            gain_penalty: gain_penalty.max(0.0),
            split_pessimism: split_pessimism.max(0.0),
            self_score_splits,
            hetero_trees,
            dart_rate: dart_rate.clamp(0.0, 0.9),
            max_delta_step: max_delta_step.max(0.0),
            cyclic_features,
            auto_interactions,
            auto_cat_interactions,
            max_interaction_features,
            lambda_schedule: lambda_schedule.max(0.0),
            use_bootstrap,
            extra_trees,
            label_smooth: label_smooth.clamp(0.0, 0.5),
            multi_output_tree,
            prob_avg,
            honest_tau: honest_tau.max(0.0),
            complement_debias_mode: complement_debias_mode.min(3),
            phase_schedule,
            ncl_lambda: ncl_lambda.max(0.0),
            adaptive_cyclic_order,
            cyclic_partner_features,
            cyclic_partner_min_pressure_ratio: cyclic_partner_min_pressure_ratio.clamp(0.0, 10.0),
            cyclic_partner_bins: cyclic_partner_bins.clamp(2, 32),
            cyclic_feature_reuse,
            cyclic_revisit_trees,
            cyclic_revisit_min_pressure_ratio: cyclic_revisit_min_pressure_ratio.clamp(0.0, 1.0),
            adaptive_feature_mask,
            adaptive_feature_mask_penalty: adaptive_feature_mask_penalty.clamp(0.0, 5.0),
            adaptive_root_anchor,
            adaptive_root_anchor_penalty: adaptive_root_anchor_penalty.clamp(0.0, 5.0),
            sparse_oblique_splits,
            interval_splits,
            sibling_block_correction: sibling_block_correction.clamp(0.0, 1.0),
            adam_beta2: adam_beta2.clamp(0.0, 0.9999),
            adam_eps: adam_eps.max(1e-12),
            ortho_alpha: ortho_alpha.clamp(0.0, 2.0),
            split_criterion,
            rank_mix_alpha: rank_mix_alpha.clamp(0.0, 1.0),
            rank_mix_start_frac: rank_mix_start_frac.clamp(0.0, 1.0),
            binary_focus_gamma: binary_focus_gamma.clamp(0.0, 4.0),
            binary_focus_end_frac: binary_focus_end_frac.clamp(0.0, 1.0),
            feature_view_groups,
            leaf_trim_pct: leaf_trim_pct.clamp(0.0, 0.49),
            leaf_median,
            leaf_median_blend: leaf_median_blend.clamp(0.0, 1.0),
            leaf_mad_clip: leaf_mad_clip.max(0.0),
            leaf_adaptive_blend_kappa: leaf_adaptive_blend_kappa.max(0.0),
            ordered_boost,
            ordered_n_buckets: ordered_n_buckets.max(2).min(16),
            goss_top_rate: goss_top_rate.clamp(0.0, 0.99),
            goss_other_rate: goss_other_rate.clamp(0.0, 0.99),
            goss_mode,
            goss_anneal: goss_anneal.clamp(0.0, 0.8),
            keep_all_trees,
            corrective_block_refit,
            corrective_blocks: corrective_blocks.clamp(1, 256),
            corrective_lambda: corrective_lambda.max(0.0),
            corrective_blend: corrective_blend.clamp(0.0, 1.0),
            corrective_min_trees: corrective_min_trees.max(2),
            corrective_audit_fraction: corrective_audit_fraction.clamp(0.0, 0.5),
            corrective_min_rel_improve: corrective_min_rel_improve.clamp(0.0, 0.5),
            leaf_eb,
            leaf_eb_min_trees: leaf_eb_min_trees.max(5),
            leaf_eb_scale: leaf_eb_scale.max(0.0),
            leaf_sibling_smooth: leaf_sibling_smooth.clamp(0.0, 0.5),
            hierarchical_shrinkage: hierarchical_shrinkage.max(0.0),
            multiclass_coupled_leaves,
            multiclass_joint_cll,
            class_weights,
            adaptive_leaf_experts,
            adaptive_cat_lookup_smooth,
            cat_offset_smooth: cat_offset_smooth.max(0.0),
            cat_offset_passes: cat_offset_passes.min(4),
            cat_offset_maps: Vec::new(),
            ordered_ctr,
            ordered_ctr_top_features: ordered_ctr_top_features.min(128),
            ordered_ctr_smooth: ordered_ctr_smooth.max(0.0),
            ordered_ctr_permutations: ordered_ctr_permutations.clamp(1, 8),
            ordered_ctr_min_count: ordered_ctr_min_count.max(1),
            ordered_ctr_features: Vec::new(),
            ordered_ctr_prior: 0.0,
            ordered_ctr_maps: Vec::new(),
            ordered_ctr_count_maps: Vec::new(),
            ordered_ctr_pair_features: Vec::new(),
            ordered_ctr_pair_maps: Vec::new(),
            ordered_ctr_pair_count_maps: Vec::new(),
            ordered_ctr_triple_features: Vec::new(),
            ordered_ctr_triple_maps: Vec::new(),
            ordered_ctr_triple_count_maps: Vec::new(),
            cat_tuple_lookups,
            cat_tuple_max_order: cat_tuple_max_order.clamp(2, 3),
            cat_tuple_top_features: cat_tuple_top_features.clamp(2, 12),
            cat_tuple_hash_bins: cat_tuple_hash_bins.clamp(8, 512),
            cat_tuple_min_leaf: cat_tuple_min_leaf.max(2),
            cat_tuple_gain_margin: cat_tuple_gain_margin.max(0.0),
            expert_leaf_admission,
            expert_max_terms,
            expert_min_leaf,
            expert_min_cal,
            expert_ridge_lambda,
            expert_alpha_max,
            expert_param_penalty,
            expert_se_multiplier,
            expert_epsilon,
            expert_shadow_trials: expert_shadow_trials.min(8),
            antithetic_subtrees,
            newton_decrement_cap: newton_decrement_cap.max(0.0),
            lookahead_alpha: lookahead_alpha.clamp(0.0, 2.0),
            sign_confidence_gamma: sign_confidence_gamma.clamp(0.0, 5.0),
            soft_predict_bandwidth: soft_predict_bandwidth.clamp(0.0, 5.0),
            jensen_train_temp: jensen_train_temp.clamp(0.5, 5.0),
            diversity_penalty: diversity_penalty.clamp(0.0, 1.0),
            diversity_decay: diversity_decay.clamp(0.5, 0.999),
            feature_usage_ema: Vec::new(),
            jit_catpair_enabled,
            jit_catpair_top_k: jit_catpair_top_k.max(2),
            jit_catpair_k_buckets: jit_catpair_k_buckets.clamp(2, 8),
            jit_catpair_min_node_rows: jit_catpair_min_node_rows.max(16),
            jit_catpair_max_node_depth,
            jit_catpair_gain_margin: jit_catpair_gain_margin.max(1.0),
            jit_ltso_enabled,
            val_losses: Vec::new(),
            best_round: 0,
            trees: Vec::new(),
            dart_tree_weights: Vec::new(),
            binned: None,
            n_features: 0,
            n_classes: 0,
            multiclass_trees_per_class: 1,
            multiclass_tree_lr_scale: 1.0,
            tree_in_sample: Vec::new(),
            numeric_interaction_pairs: Vec::new(),
            numeric_interaction_edges: Vec::new(),
            categorical_interaction_pairs: Vec::new(),
            categorical_interaction_edges: Vec::new(),
            sumdiff_pairs: Vec::new(),
            sumdiff_edges: Vec::new(),
        }
    }

    #[pyo3(signature = (x, y, n_rounds, eval_x = None, eval_y = None, init_score = None, sample_weight = None))]
    pub fn fit(
        &mut self,
        py: Python<'_>,
        x: Bound<'_, PyAny>,
        y: Bound<'_, PyAny>,
        n_rounds: usize,
        eval_x: Option<Bound<'_, PyAny>>,
        eval_y: Option<Bound<'_, PyAny>>,
        init_score: Option<Bound<'_, PyAny>>,
        sample_weight: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let x_array: Bound<'_, PyArray2<f64>> = x.extract()?;
        let y_array: Bound<'_, PyArray1<f64>> = y.extract()?;

        let n_rows = x_array.shape()[0];
        let n_features = x_array.shape()[1];
        let n_features_original = n_features;
        self.n_features = n_features;

        let x_owned = x_array.to_owned_array();
        let x_standard = if x_owned.is_standard_layout() {
            x_owned
        } else {
            x_owned.as_standard_layout().into_owned()
        };
        let x_data: Vec<f64> = x_standard.into_raw_vec_and_offset().0;
        let y_data: Vec<f64> = y_array.to_owned_array().into_raw_vec_and_offset().0;

        // Optional init_score: per-row base offset added to predictions before
        // boosting begins. The trees fit residuals against this offset; predict()
        // callers must pass the matching init_score for new rows. Currently
        // wired only for binary/regression — multiclass falls through to
        // class_base_scores even when this is set.
        let init_score_data: Option<Vec<f64>> = match init_score {
            Some(s) => {
                let arr: Bound<'_, PyArray1<f64>> = s.extract().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "init_score must be a 1-D float64 numpy array",
                    )
                })?;
                let v: Vec<f64> = arr.to_owned_array().into_raw_vec_and_offset().0;
                if v.len() != n_rows {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "init_score length ({}) must equal n_rows ({})",
                        v.len(),
                        n_rows
                    )));
                }
                Some(v)
            }
            None => None,
        };

        let sample_weight_data: Option<Vec<f64>> = match sample_weight {
            Some(s) => {
                let arr: Bound<'_, PyArray1<f64>> = s.extract().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "sample_weight must be a 1-D float64 numpy array",
                    )
                })?;
                let v: Vec<f64> = arr.to_owned_array().into_raw_vec_and_offset().0;
                if v.len() != n_rows {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "sample_weight length ({}) must equal n_rows ({})",
                        v.len(),
                        n_rows
                    )));
                }
                Some(v)
            }
            None => None,
        };

        // Extract eval data here, BEFORE entering the Rayon-installed
        // closure. PyAny / Python<'_> are !Send so they can't cross into
        // the worker pool. We convert to owned Vec<f64> while we still
        // have the GIL token, then pass the raw vectors in.
        let eval_data_raw: Option<(Vec<f64>, Vec<f64>, usize)> =
            if let (Some(ex), Some(ey)) = (eval_x, eval_y) {
                let ex_array: Bound<'_, PyArray2<f64>> = ex.extract()?;
                let ey_array: Bound<'_, PyArray1<f64>> = ey.extract()?;
                let en_rows = ex_array.shape()[0];
                let ex_owned = ex_array.to_owned_array();
                let ex_standard = if ex_owned.is_standard_layout() {
                    ex_owned
                } else {
                    ex_owned.as_standard_layout().into_owned()
                };
                let ex_data: Vec<f64> = ex_standard.into_raw_vec_and_offset().0;
                let ey_data: Vec<f64> = ey_array.to_owned_array().into_raw_vec_and_offset().0;
                Some((ex_data, ey_data, en_rows))
            } else {
                None
            };

        // Adaptive thread pool: parallelism is profitable only when the work
        // per task exceeds dispatch overhead. With Rayon's ~20µs/thread
        // overhead and ~ns-scale tree ops, we want roughly one thread per
        // 50k cells (n_rows × n_features). Small data collapses to 1
        // thread (no dispatch waste); large data uses all cores.
        let max_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let tree_work_multiplier = if self.task == "multiclass" {
            let n_classes = y_data
                .iter()
                .filter(|v| v.is_finite())
                .map(|&v| v as usize)
                .max()
                .unwrap_or(0)
                + 1;
            n_rounds
                .max(1)
                .saturating_mul(self.n_trees_per_round.max(1))
                .saturating_mul(n_classes.max(1))
        } else {
            n_rounds
                .max(1)
                .saturating_mul(self.n_trees_per_round.max(1))
        };
        let work_estimate = n_rows
            .saturating_mul(n_features)
            .saturating_mul(tree_work_multiplier);
        let want_threads = (work_estimate / 400_000).clamp(1, max_cpus);
        let adaptive_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(want_threads)
            .build()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "rayon thread pool: {}",
                    e
                ))
            })?;

        // Release GIL while we run rayon work — workers can't hold it anyway.
        py.detach(|| {
            adaptive_pool.install(|| -> PyResult<()> {
                let effective_bins = self.num_bins.min(32.max(n_rows / 4));
                let mut binned = BinnedData::new(
                    &x_data,
                    n_rows,
                    n_features,
                    effective_bins,
                    &self.cat_features,
                    self.max_cat_bins,
                );
                binned.split_pessimism = self.split_pessimism;

                // (LTSO Phase 3 moved below all data augmentation — see post-eval block.)

                // Auto-generate pairwise numeric product features (X_i * X_j)
                // Tree-guided: run warmup trees, extract co-occurrence scores, select top-K pairs
                // Multiclass: warmup uses regression proxy (y = class label) to identify interacting features
                self.numeric_interaction_pairs.clear();
                self.numeric_interaction_edges.clear();
                self.categorical_interaction_pairs.clear();
                self.categorical_interaction_edges.clear();
                self.sumdiff_pairs.clear();
                self.sumdiff_edges.clear();
                self.feature_usage_ema.clear();
                let supports_interactions = true;
                if (self.auto_interactions || self.auto_cat_interactions) && supports_interactions {
                    let numeric_indices: Vec<usize> = (0..n_features)
                        .filter(|&i| i >= self.cat_features.len() || !self.cat_features[i])
                        .collect();
                    let categorical_indices: Vec<usize> = (0..n_features)
                        .filter(|&i| i < self.cat_features.len() && self.cat_features[i])
                        .collect();
                    let n_numeric = numeric_indices.len();
                    let n_categorical = categorical_indices.len();
                    if (self.auto_interactions && n_numeric >= 2)
                        || (self.auto_cat_interactions
                            && n_categorical >= 2
                            && self.task != "regression"
                            && self.task != "poisson")
                    {
                        let max_pairs = if self.max_interaction_features > 0 {
                            self.max_interaction_features
                        } else {
                            10
                        };

                        // Run warmup trees to discover interacting feature pairs. Large
                        // datasets make each warmup expensive and usually need fewer
                        // trees for stable co-occurrence counts.
                        let warmup_cap = if n_rows >= 5_000 { 6 } else { 15 };
                        let warmup_rounds = warmup_cap.min(n_rounds);
                        let warmup_trees = Self::run_warmup_trees(
                            &binned,
                            &y_data,
                            n_rows,
                            warmup_rounds,
                            &self.task,
                            self.huber_delta,
                            self.lambda_reg,
                            self.gamma,
                            self.max_depth,
                            self.min_child_weight,
                            self.seed,
                            n_features,
                        );

                        // Score pairs by co-occurrence count × sqrt(importance_i × importance_j)
                        let mut pair_counts: std::collections::HashMap<(u32, u32), f64> =
                            std::collections::HashMap::new();
                        let mut feat_counts: std::collections::HashMap<u32, usize> =
                            std::collections::HashMap::new();
                        for tree in &warmup_trees {
                            for &f in &tree.split_features {
                                if f != u32::MAX {
                                    *feat_counts.entry(f).or_insert(0) += 1;
                                }
                            }
                            for (a, b) in tree.extract_split_cooccurrences(n_features) {
                                *pair_counts.entry((a, b)).or_insert(0.0) += 1.0;
                            }
                        }

                        // Filter to numeric-only pairs and score
                        let numeric_set: std::collections::HashSet<usize> =
                            numeric_indices.iter().copied().collect();
                        let mut scored_pairs: Vec<((usize, usize), f64)> = pair_counts
                            .iter()
                            .filter(|((a, b), _)| {
                                numeric_set.contains(&(*a as usize))
                                    && numeric_set.contains(&(*b as usize))
                            })
                            .map(|((a, b), count)| {
                                let imp_a = *feat_counts.get(&a).unwrap_or(&0) as f64;
                                let imp_b = *feat_counts.get(&b).unwrap_or(&0) as f64;
                                (
                                    ((*a) as usize, (*b) as usize),
                                    *count * (imp_a * imp_b).sqrt(),
                                )
                            })
                            .collect();
                        scored_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                        scored_pairs.truncate(max_pairs);

                        if self.auto_interactions && !scored_pairs.is_empty() {
                            let selected_pairs: Vec<(usize, usize)> =
                                scored_pairs.iter().map(|(p, _)| *p).collect();
                            let product_cols: Vec<Vec<f64>> = selected_pairs
                                .iter()
                                .map(|&(fi, fj)| {
                                    (0..n_rows)
                                        .map(|row| {
                                            let vi = x_data[row * n_features + fi];
                                            let vj = x_data[row * n_features + fj];
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
                            self.numeric_interaction_pairs = selected_pairs;
                            self.numeric_interaction_edges = binned.bin_edges[int_start..].to_vec();
                        }

                        if self.auto_cat_interactions
                            && n_categorical >= 2
                            && self.task != "regression"
                            && self.task != "poisson"
                        {
                            let categorical_set: std::collections::HashSet<usize> =
                                categorical_indices.iter().copied().collect();
                            let mut scored_cat_pairs: Vec<((usize, usize), f64)> = pair_counts
                                .iter()
                                .filter(|((a, b), _)| {
                                    categorical_set.contains(&(*a as usize))
                                        && categorical_set.contains(&(*b as usize))
                                })
                                .map(|((a, b), count)| {
                                    let imp_a = *feat_counts.get(&a).unwrap_or(&0) as f64;
                                    let imp_b = *feat_counts.get(&b).unwrap_or(&0) as f64;
                                    (
                                        ((*a) as usize, (*b) as usize),
                                        *count * (imp_a * imp_b).sqrt(),
                                    )
                                })
                                .collect();
                            scored_cat_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                            scored_cat_pairs.truncate(max_pairs.min(8));

                            if !scored_cat_pairs.is_empty() {
                                let selected_pairs: Vec<(usize, usize)> =
                                    scored_cat_pairs.iter().map(|(p, _)| *p).collect();
                                let hash_cols: Vec<Vec<f64>> = selected_pairs
                                    .iter()
                                    .map(|&(fi, fj)| {
                                        (0..n_rows)
                                            .map(|row| {
                                                let vi = x_data[row * n_features + fi];
                                                let vj = x_data[row * n_features + fj];
                                                if vi.is_nan() || vj.is_nan() {
                                                    f64::NAN
                                                } else {
                                                    let hi = vi as i64;
                                                    let hj = vj as i64;
                                                    let h = ((hi.wrapping_mul(1_000_003))
                                                        ^ (hj.wrapping_mul(1_000_033)))
                                                        & 255;
                                                    h as f64
                                                }
                                            })
                                            .collect()
                                    })
                                    .collect();
                                let cat_start = binned.n_features;
                                binned.add_categorical_features(&hash_cols);
                                self.categorical_interaction_pairs = selected_pairs;
                                self.categorical_interaction_edges =
                                    binned.bin_edges[cat_start..].to_vec();
                            }
                        }
                    }
                }

                let ordered_ctr_edges: Vec<Vec<f64>> = {
                    let ctr_cols =
                        self.build_ordered_ctr_features(&x_data, &y_data, n_rows, n_features);
                    if ctr_cols.is_empty() {
                        Vec::new()
                    } else {
                        let ctr_start = binned.n_features;
                        binned.add_ots_features(&ctr_cols, effective_bins);
                        binned.bin_edges[ctr_start..].to_vec()
                    }
                };

                // Prepare eval data if provided
                let eval_data = if let Some((ex_data, ey_data, en_rows)) = eval_data_raw {
                    let mut eval_bins = BinnedData::bin_with_edges(
                        &ex_data,
                        en_rows,
                        n_features,
                        &binned.bin_edges[..n_features],
                        &binned.is_categorical[..n_features],
                    );
                    // Add numeric interaction product features to eval data
                    if !self.numeric_interaction_pairs.is_empty() {
                        let eval_products: Vec<Vec<f64>> = self
                            .numeric_interaction_pairs
                            .iter()
                            .map(|&(fi, fj)| {
                                (0..en_rows)
                                    .map(|row| {
                                        let vi = ex_data[row * n_features + fi];
                                        let vj = ex_data[row * n_features + fj];
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
                            &mut eval_bins,
                            en_rows,
                            &eval_products,
                            &self.numeric_interaction_edges,
                        );
                    }
                    if !self.categorical_interaction_pairs.is_empty() {
                        let eval_hashes: Vec<Vec<f64>> = self
                            .categorical_interaction_pairs
                            .iter()
                            .map(|&(fi, fj)| {
                                (0..en_rows)
                                    .map(|row| {
                                        let vi = ex_data[row * n_features + fi];
                                        let vj = ex_data[row * n_features + fj];
                                        if vi.is_nan() || vj.is_nan() {
                                            f64::NAN
                                        } else {
                                            let hi = vi as i64;
                                            let hj = vj as i64;
                                            let h = ((hi.wrapping_mul(1_000_003))
                                                ^ (hj.wrapping_mul(1_000_033)))
                                                & 255;
                                            h as f64
                                        }
                                    })
                                    .collect()
                            })
                            .collect();
                        BinnedData::add_categorical_features_with_edges(
                            &mut eval_bins,
                            en_rows,
                            &eval_hashes,
                            &self.categorical_interaction_edges,
                        );
                    }
                    if !ordered_ctr_edges.is_empty() {
                        let eval_ctr =
                            self.ordered_ctr_columns_for_raw(&ex_data, en_rows, n_features);
                        BinnedData::add_ots_features_with_edges(
                            &mut eval_bins,
                            en_rows,
                            &eval_ctr,
                            &ordered_ctr_edges,
                        );
                    }
                    // Build CLL hash bins for eval data (for high-cardinality categoricals)
                    let eval_cll_hash_bins = if self.cat_lookup_smooth > 0.0 {
                        let mut cll_bins = BinnedData::build_cll_hash_bins(
                            &ex_data,
                            en_rows,
                            n_features,
                            &self.cat_features,
                            &binned.is_categorical[..n_features],
                            &binned.bin_edges[..n_features],
                        );
                        if !self.categorical_interaction_pairs.is_empty() {
                            let eval_hashes: Vec<Vec<f64>> = self
                                .categorical_interaction_pairs
                                .iter()
                                .map(|&(fi, fj)| {
                                    (0..en_rows)
                                        .map(|row| {
                                            let vi = ex_data[row * n_features + fi];
                                            let vj = ex_data[row * n_features + fj];
                                            if vi.is_nan() || vj.is_nan() {
                                                f64::NAN
                                            } else {
                                                let hi = vi as i64;
                                                let hj = vj as i64;
                                                let h = ((hi.wrapping_mul(1_000_003))
                                                    ^ (hj.wrapping_mul(1_000_033)))
                                                    & 255;
                                                h as f64
                                            }
                                        })
                                        .collect()
                                })
                                .collect();
                            BinnedData::add_categorical_features_with_edges(
                                &mut cll_bins,
                                en_rows,
                                &eval_hashes,
                                &self.categorical_interaction_edges,
                            );
                        }
                        cll_bins
                    } else {
                        Vec::new()
                    };
                    // GGFP v6 (LTSO) — virtuals are registered AFTER eval_data is built
                    // (see post-eval block). We need ex_data preserved here so we can
                    // extend eval_bins with virtual columns later.
                    let eval_cll_hash_bins_mut = eval_cll_hash_bins;
                    let eval_raw = if self.auto_interactions || self.jit_ltso_enabled {
                        ex_data
                    } else {
                        Vec::new()
                    };
                    Some((
                        eval_bins,
                        ey_data,
                        en_rows,
                        eval_raw,
                        eval_cll_hash_bins_mut,
                    ))
                } else {
                    None
                };

                // GGFP v6 (LTSO Phase 3) — pre-mine N×N operators on the ORIGINAL raw
                // numerics (cols 0..n_features_raw), AFTER auto_interactions/sumdiff/
                // ordered_ctr have already extended binned. Virtuals end up as the
                // last columns, so eval data — extended in the same order — stays
                // column-aligned with the training matrix.
                let mut eval_data = eval_data;
                if self.jit_ltso_enabled && n_features >= 2 {
                    let raw_by_full_id: Vec<Vec<f64>> = (0..n_features)
                        .map(|col| {
                            (0..n_rows)
                                .map(|row| x_data[row * n_features + col])
                                .collect::<Vec<f64>>()
                        })
                        .collect();
                    let num_indices: Vec<usize> = (0..n_features)
                        .filter(|&i| i >= self.cat_features.len() || !self.cat_features[i])
                        .collect();
                    let cat_indices: Vec<usize> = (0..n_features)
                        .filter(|&i| i < self.cat_features.len() && self.cat_features[i])
                        .collect();
                    // LTSO is only a regression learner-family. It is admitted by
                    // honest residual-transfer below, so low-dimensional numeric
                    // regression can use hinge/diff/ratio operators while mixed
                    // categorical regression can still use the N|C family.
                    if self.task != "regression" || num_indices.is_empty() {
                        // Leave binned/eval data untouched.
                    } else {
                        let (residual, residual_base) = match self.task.as_str() {
                            "regression" => {
                                let mu = y_data.iter().copied().sum::<f64>() / y_data.len() as f64;
                                (y_data.iter().map(|&v| v - mu).collect::<Vec<f64>>(), mu)
                            }
                            "binary" => {
                                let p0 = y_data.iter().copied().sum::<f64>() / y_data.len() as f64;
                                (y_data.iter().map(|&v| v - p0).collect::<Vec<f64>>(), p0)
                            }
                            _ => {
                                let p0 = y_data.iter().filter(|&&v| (v - 0.0).abs() < 0.5).count()
                                    as f64
                                    / y_data.len() as f64;
                                (
                                    y_data
                                        .iter()
                                        .map(
                                            |&v| if (v - 0.0).abs() < 0.5 { 1.0 - p0 } else { -p0 },
                                        )
                                        .collect::<Vec<f64>>(),
                                    p0,
                                )
                            }
                        };
                        let numeric_only = cat_indices.is_empty();
                        let max_accept = if numeric_only {
                            std::env::var("GTBOOST_LTSO_NUMERIC_MAX_ACCEPT")
                                .ok()
                                .and_then(|v| v.parse::<usize>().ok())
                                .unwrap_or(3)
                        } else {
                            std::env::var("GTBOOST_LTSO_MIXED_MAX_ACCEPT")
                                .ok()
                                .and_then(|v| v.parse::<usize>().ok())
                                .unwrap_or(1)
                        };
                        let cfg = crate::tree::LtsoPremineConfig {
                            enabled: true,
                            max_accept,
                            ..Default::default()
                        };
                        let mut eval_raw_by_full_id: Option<Vec<Vec<f64>>> = eval_data
                            .as_ref()
                            .and_then(|(_, _, en_rows_eval, eval_raw_ref, _)| {
                                if eval_raw_ref.len() == en_rows_eval * n_features {
                                    Some(
                                        (0..n_features)
                                            .map(|col| {
                                                (0..*en_rows_eval)
                                                    .map(|row| eval_raw_ref[row * n_features + col])
                                                    .collect::<Vec<f64>>()
                                            })
                                            .collect::<Vec<Vec<f64>>>(),
                                    )
                                } else {
                                    None
                                }
                            });
                        let mut eval_residual: Option<Vec<f64>> =
                            eval_data.as_ref().map(|(_, ey_data_eval, _, _, _)| {
                                match self.task.as_str() {
                                    "regression" => {
                                        ey_data_eval.iter().map(|&v| v - residual_base).collect()
                                    }
                                    "binary" => {
                                        ey_data_eval.iter().map(|&v| v - residual_base).collect()
                                    }
                                    _ => ey_data_eval
                                        .iter()
                                        .map(|&v| {
                                            if (v - 0.0).abs() < 0.5 {
                                                1.0 - residual_base
                                            } else {
                                                -residual_base
                                            }
                                        })
                                        .collect(),
                                }
                            });
                        let mut selector_raw_by_full_id = raw_by_full_id.clone();
                        let mut selector_residual = residual.clone();
                        let mut accepted_values_are_full_rows = true;
                        if eval_raw_by_full_id.is_none() && n_rows >= 160 {
                            let mut selector_rows: Vec<usize> = Vec::with_capacity(n_rows * 4 / 5);
                            let mut honest_rows: Vec<usize> = Vec::with_capacity(n_rows / 5 + 1);
                            for row in 0..n_rows {
                                let h = ((row as u64)
                                    .wrapping_mul(6364136223846793005)
                                    .wrapping_add(self.seed)
                                    >> 32)
                                    % 5;
                                if h == 0 {
                                    honest_rows.push(row);
                                } else {
                                    selector_rows.push(row);
                                }
                            }
                            if selector_rows.len() >= 64 && honest_rows.len() >= 32 {
                                let select_cols =
                                    |cols: &[Vec<f64>], rows: &[usize]| -> Vec<Vec<f64>> {
                                        cols.iter()
                                            .map(|col| {
                                                rows.iter().map(|&r| col[r]).collect::<Vec<f64>>()
                                            })
                                            .collect::<Vec<Vec<f64>>>()
                                    };
                                selector_raw_by_full_id =
                                    select_cols(&raw_by_full_id, &selector_rows);
                                selector_residual =
                                    selector_rows.iter().map(|&r| residual[r]).collect();
                                eval_raw_by_full_id =
                                    Some(select_cols(&raw_by_full_id, &honest_rows));
                                eval_residual =
                                    Some(honest_rows.iter().map(|&r| residual[r]).collect());
                                accepted_values_are_full_rows = false;
                            }
                        }
                        let accepted = crate::tree::virtual_features::premine_candidates(
                            &selector_raw_by_full_id,
                            &num_indices,
                            &cat_indices,
                            &selector_residual,
                            &cfg,
                            eval_raw_by_full_id.as_deref(),
                            eval_residual.as_deref(),
                        );
                        let n_accepted = accepted.len();
                        for (op_def, train_vals, _score) in accepted {
                            let train_vals = if accepted_values_are_full_rows {
                                train_vals
                            } else {
                                crate::tree::virtual_features::materialize_virtual_feature(
                                    &op_def,
                                    &raw_by_full_id,
                                )
                            };
                            let (bins, edges) = crate::tree::virtual_features::quantile_bin(
                                &train_vals,
                                cfg.n_bins,
                            );
                            binned.register_virtual_feature(op_def, bins, edges);
                        }
                        if n_accepted > 0 {
                            while self.cat_features.len() < binned.n_features {
                                self.cat_features.push(false);
                            }
                            // Extend eval_data with the same virtuals (columns end up in
                            // the same order as training: [raw | derived | virtual]).
                            if let Some((
                                ref mut eval_bins_ref,
                                _,
                                en_rows_ref,
                                ref eval_raw_ref,
                                ref mut eval_cll_ref,
                            )) = eval_data
                            {
                                let en_rows_local = en_rows_ref;
                                let cll_opt = if eval_cll_ref.is_empty() {
                                    None
                                } else {
                                    Some(eval_cll_ref)
                                };
                                crate::tree::extend_eval_bins_with_virtuals(
                                    &binned,
                                    eval_bins_ref,
                                    cll_opt,
                                    eval_raw_ref,
                                    en_rows_local,
                                    n_features,
                                );
                            }
                            if std::env::var("GTBOOST_LTSO_DEBUG")
                                .map(|v| v == "1")
                                .unwrap_or(false)
                            {
                                eprintln!(
                        "[LTSO] registered {} virtual features (raw {} -> n_features {})",
                        n_accepted, n_features, binned.n_features
                    );
                            }
                        }
                    }
                }

                // Update n_features to include OTS columns (if any) — and LTSO virtuals
                let n_features = binned.n_features;
                self.n_features = n_features;

                // GIL was already released by the outer py.detach wrapping
                // the whole fit body; just continue.
                {
                    self.trees.clear();
                    self.tree_in_sample.clear();

                    // Auto-compute optimal base_score from training data.
                    let y_mean = if let Some(sw) = sample_weight_data.as_deref() {
                        let mut num = 0.0;
                        let mut den = 0.0;
                        for i in 0..n_rows {
                            let w = if sw[i].is_finite() {
                                sw[i].max(0.0)
                            } else {
                                0.0
                            };
                            num += w * y_data[i];
                            den += w;
                        }
                        if den > 0.0 {
                            num / den
                        } else {
                            y_data.iter().sum::<f64>() / n_rows as f64
                        }
                    } else {
                        y_data.iter().sum::<f64>() / n_rows as f64
                    };
                    match self.task.as_str() {
                        "regression" => {
                            self.base_score = y_mean;
                            self.class_base_scores.clear();
                        }
                        "binary" => {
                            let p = if self.class_weights.len() >= 2 {
                                let neg_w = if self.class_weights[0].is_finite()
                                    && self.class_weights[0] > 0.0
                                {
                                    self.class_weights[0]
                                } else {
                                    1.0
                                };
                                let pos_w = if self.class_weights[1].is_finite()
                                    && self.class_weights[1] > 0.0
                                {
                                    self.class_weights[1]
                                } else {
                                    1.0
                                };
                                let (pos, neg) = if let Some(sw) = sample_weight_data.as_deref() {
                                    let mut pos = 0.0;
                                    let mut neg = 0.0;
                                    for i in 0..n_rows {
                                        let w = if sw[i].is_finite() {
                                            sw[i].max(0.0)
                                        } else {
                                            0.0
                                        };
                                        if y_data[i] > 0.5 {
                                            pos += w;
                                        } else {
                                            neg += w;
                                        }
                                    }
                                    (pos, neg)
                                } else {
                                    let pos = y_data.iter().filter(|&&v| v > 0.5).count() as f64;
                                    (pos, n_rows as f64 - pos)
                                };
                                let denom = pos_w * pos + neg_w * neg;
                                if denom > 0.0 {
                                    (pos_w * pos / denom).clamp(1e-6, 1.0 - 1e-6)
                                } else {
                                    y_mean.clamp(1e-6, 1.0 - 1e-6)
                                }
                            } else {
                                y_mean.clamp(1e-6, 1.0 - 1e-6)
                            };
                            self.base_score = (p / (1.0 - p)).ln();
                            self.class_base_scores.clear();
                        }
                        "poisson" => {
                            self.base_score = y_mean.max(1e-6).ln();
                            self.class_base_scores.clear();
                        }
                        "multiclass" => {
                            let n_classes =
                                y_data.iter().map(|&v| v as usize).max().unwrap_or(0) + 1;
                            let alpha = 1.0; // Laplace prior keeps absent/rare fold classes finite.
                            let denom = n_rows as f64 + alpha * n_classes as f64;
                            let mut counts = vec![alpha; n_classes];
                            for &yi in &y_data {
                                let k = yi as usize;
                                if k < n_classes {
                                    counts[k] += 1.0;
                                }
                            }
                            self.class_base_scores =
                                counts.iter().map(|&c| (c / denom).ln()).collect();
                            let mean_margin =
                                self.class_base_scores.iter().sum::<f64>() / n_classes as f64;
                            for v in self.class_base_scores.iter_mut() {
                                *v -= mean_margin;
                            }
                        }
                        _ => {}
                    }

                    let mut eval_data = eval_data;
                    if self.task == "multiclass" {
                        self.fit_multiclass(
                            &mut binned,
                            &y_data,
                            n_rows,
                            n_features,
                            n_rounds,
                            &mut eval_data,
                        );
                    } else {
                        self.fit_single(
                            &mut binned,
                            &y_data,
                            n_rows,
                            n_features,
                            n_rounds,
                            &mut eval_data,
                            &x_data,
                            n_features_original,
                            init_score_data.as_deref(),
                            sample_weight_data.as_deref(),
                        );
                        self.apply_corrective_block_refit(
                            &binned,
                            &x_data,
                            n_rows,
                            n_features_original,
                            &y_data,
                            init_score_data.as_deref(),
                        );
                    }
                }

                self.binned = Some(binned);
                Ok(())
            }) // close adaptive_pool.install closure
        }) // close py.detach closure
    }

    #[pyo3(signature = (dataset, n_rounds, init_score = None, sample_weight = None))]
    pub fn fit_binned(
        &mut self,
        py: Python<'_>,
        dataset: PyRef<'_, GTBoostDataset>,
        n_rounds: usize,
        init_score: Option<Bound<'_, PyAny>>,
        sample_weight: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        if self.auto_interactions
            || self.auto_cat_interactions
            || self.ordered_ctr
            || self.jit_ltso_enabled
        {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "fit_binned cannot reuse cached bins when auto_interactions, \
                 auto_cat_interactions, ordered_ctr, or jit_ltso_enabled are active; \
                 use normal fit() for those configs",
            ));
        }
        if self.num_bins != dataset.num_bins {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "model num_bins ({}) must match GTBoostDataset num_bins ({})",
                self.num_bins, dataset.num_bins
            )));
        }
        if self.max_cat_bins != dataset.max_cat_bins {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "model max_cat_bins ({}) must match GTBoostDataset max_cat_bins ({})",
                self.max_cat_bins, dataset.max_cat_bins
            )));
        }
        if !self.cat_features.is_empty() && self.cat_features != dataset.cat_features {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "model cat_features must match GTBoostDataset cat_features",
            ));
        }

        let n_rows = dataset.n_rows;
        let n_features_original = dataset.n_features_raw;
        let mut binned = dataset.binned.clone();
        binned.split_pessimism = self.split_pessimism;
        let x_data = dataset.x_data.clone();
        let y_data = dataset.y_data.clone();
        let mut eval_data = dataset.eval_data.clone();
        self.cat_features = dataset.cat_features.clone();

        let init_score_data: Option<Vec<f64>> = match init_score {
            Some(s) => {
                let arr: Bound<'_, PyArray1<f64>> = s.extract().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "init_score must be a 1-D float64 numpy array",
                    )
                })?;
                let v: Vec<f64> = arr.to_owned_array().into_raw_vec_and_offset().0;
                if v.len() != n_rows {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "init_score length ({}) must equal n_rows ({})",
                        v.len(),
                        n_rows
                    )));
                }
                Some(v)
            }
            None => None,
        };

        let sample_weight_data: Option<Vec<f64>> = match sample_weight {
            Some(s) => {
                let arr: Bound<'_, PyArray1<f64>> = s.extract().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "sample_weight must be a 1-D float64 numpy array",
                    )
                })?;
                let v: Vec<f64> = arr.to_owned_array().into_raw_vec_and_offset().0;
                if v.len() != n_rows {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "sample_weight length ({}) must equal n_rows ({})",
                        v.len(),
                        n_rows
                    )));
                }
                Some(v)
            }
            None => None,
        };

        let max_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let tree_work_multiplier = if self.task == "multiclass" {
            let n_classes = y_data
                .iter()
                .filter(|v| v.is_finite())
                .map(|&v| v as usize)
                .max()
                .unwrap_or(0)
                + 1;
            n_rounds
                .max(1)
                .saturating_mul(self.n_trees_per_round.max(1))
                .saturating_mul(n_classes.max(1))
        } else {
            n_rounds
                .max(1)
                .saturating_mul(self.n_trees_per_round.max(1))
        };
        let work_estimate = n_rows
            .saturating_mul(binned.n_features)
            .saturating_mul(tree_work_multiplier);
        let want_threads = (work_estimate / 400_000).clamp(1, max_cpus);
        let adaptive_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(want_threads)
            .build()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "rayon thread pool: {}",
                    e
                ))
            })?;

        py.detach(|| {
            adaptive_pool.install(|| -> PyResult<()> {
                let n_features = binned.n_features;
                self.n_features = n_features;
                self.clear_trained_state_for_fit();
                self.compute_base_scores_for_fit(&y_data, n_rows, sample_weight_data.as_deref());

                if self.task == "multiclass" {
                    self.fit_multiclass(
                        &mut binned,
                        &y_data,
                        n_rows,
                        n_features,
                        n_rounds,
                        &mut eval_data,
                    );
                } else {
                    self.fit_single(
                        &mut binned,
                        &y_data,
                        n_rows,
                        n_features,
                        n_rounds,
                        &mut eval_data,
                        &x_data,
                        n_features_original,
                        init_score_data.as_deref(),
                        sample_weight_data.as_deref(),
                    );
                    self.apply_corrective_block_refit(
                        &binned,
                        &x_data,
                        n_rows,
                        n_features_original,
                        &y_data,
                        init_score_data.as_deref(),
                    );
                }

                self.binned = Some(binned);
                Ok(())
            })
        })
    }

    #[pyo3(signature = (x, init_score = None))]
    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        x: Bound<'py, PyAny>,
        init_score: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_array: Bound<'_, PyArray2<f64>> = x.extract()?;
        let n_rows = x_array.shape()[0];
        let n_features_raw = x_array.shape()[1];
        let x_owned = x_array.to_owned_array();
        let x_standard = if x_owned.is_standard_layout() {
            x_owned
        } else {
            x_owned.as_standard_layout().into_owned()
        };
        let x_data_raw: Vec<f64> = x_standard.into_raw_vec_and_offset().0;

        // Optional per-row init_score: must match what was passed to fit().
        // Only applied on non-multiclass paths in this revision.
        let init_score_vec: Option<Vec<f64>> = match init_score {
            Some(s) => {
                let arr: Bound<'_, PyArray1<f64>> = s.extract().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "predict: init_score must be a 1-D float64 numpy array",
                    )
                })?;
                let v: Vec<f64> = arr.to_owned_array().into_raw_vec_and_offset().0;
                if v.len() != n_rows {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "predict: init_score length ({}) must equal n_rows ({})",
                        v.len(),
                        n_rows,
                    )));
                }
                Some(v)
            }
            None => None,
        };

        let binned = match &self.binned {
            Some(b) => b,
            None => {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Model not trained yet. Call fit() first.",
                ))
            }
        };

        let (x_data_cow, n_features): (std::borrow::Cow<'_, [f64]>, usize) =
            if self.raw_matrix_extensions_active() {
                let (owned, nf) = self.extend_raw_matrix(&x_data_raw, n_rows, n_features_raw);
                (std::borrow::Cow::Owned(owned), nf)
            } else {
                (
                    std::borrow::Cow::Borrowed(x_data_raw.as_slice()),
                    n_features_raw,
                )
            };
        let x_data = x_data_cow.as_ref();

        // Release GIL for prediction computation. Use the global Rayon pool:
        // constructing a custom thread pool per predict call is expensive and
        // dominated medium-batch inference.
        let result = py.detach(|| {
            let lr = if self.task == "multiclass" {
                self.multiclass_tree_lr()
            } else {
                self.learning_rate
            };
            let trees = &self.trees;

            let dart_w = &self.dart_tree_weights;
            let has_dart_w = !dart_w.is_empty();

            let use_srp = self.soft_predict_bandwidth > 0.0;
            let plain_axis_trees: Vec<bool> = if use_srp {
                Vec::new()
            } else {
                trees
                    .iter()
                    .map(|t| t.can_predict_raw_plain_axis())
                    .collect()
            };
            let binned_plain_trees: Vec<bool> = if use_srp {
                Vec::new()
            } else {
                trees.iter().map(|t| t.can_predict_binned_plain()).collect()
            };
            let use_eval_bins = !use_srp && n_features == binned.n_features;
            let use_row_major_eval_bins =
                use_eval_bins && binned_plain_trees.iter().all(|&is_plain| is_plain);
            let eval_bins_row_major = if use_row_major_eval_bins {
                Some(BinnedData::bin_with_edges_row_major(
                    x_data,
                    n_rows,
                    n_features,
                    &binned.bin_edges,
                    &binned.is_categorical,
                ))
            } else {
                None
            };
            let eval_bins = if use_eval_bins && !use_row_major_eval_bins {
                Some(BinnedData::bin_with_edges(
                    x_data,
                    n_rows,
                    n_features,
                    &binned.bin_edges,
                    &binned.is_categorical,
                ))
            } else {
                None
            };
            let eval_cll_hash_bins = if eval_bins.is_some() && self.cat_lookup_smooth > 0.0 {
                BinnedData::build_cll_hash_bins(
                    x_data,
                    n_rows,
                    n_features,
                    &self.cat_features,
                    &binned.is_categorical,
                    &binned.bin_edges,
                )
            } else {
                Vec::new()
            };
            let srp_bw = self.soft_predict_bandwidth;
            // Per-feature raw-unit scale: avg bin width. Makes sigmoid scale-invariant.
            let feat_scales: Vec<f64> = if use_srp {
                (0..n_features)
                    .map(|f| {
                        let edges = &binned.bin_edges[f];
                        if edges.len() < 2 {
                            1.0
                        } else {
                            let range = edges[edges.len() - 1] - edges[0];
                            let nb = edges.len() as f64;
                            if range > 0.0 {
                                range / nb
                            } else {
                                1.0
                            }
                        }
                    })
                    .collect()
            } else {
                Vec::new()
            };

            if self.task == "multiclass" {
                let n_classes = self.n_classes;
                let ntp = self.multiclass_trees_per_class_round();
                let use_prob_avg = self.prob_avg && ntp > 1;
                let preds_2d: Vec<Vec<f64>> = (0..n_rows)
                    .into_par_iter()
                    .map(|row| {
                        let row_data = &x_data[row * n_features..(row + 1) * n_features];
                        if use_prob_avg {
                            // Frequency leaf mode: tree values ARE class probabilities.
                            // Return log-prob margins so Python callers can softmax exactly once.
                            let mut avg_probs = vec![0.0f64; n_classes];
                            for (t_idx, tree) in trees.iter().enumerate() {
                                let k = (t_idx / ntp) % n_classes;
                                let c = if let Some(ref row_bins) = eval_bins_row_major {
                                    tree.predict_binned_plain_row_major(row_bins, n_features, row)
                                } else if let Some(ref bins) = eval_bins {
                                    if binned_plain_trees.get(t_idx).copied().unwrap_or(false) {
                                        tree.predict_binned_plain_raw(bins, n_rows, row)
                                    } else {
                                        tree.predict_binned_raw(
                                            bins,
                                            n_rows,
                                            row,
                                            &eval_cll_hash_bins,
                                        )
                                    }
                                } else if use_srp {
                                    tree.predict_raw_row_soft(
                                        binned,
                                        row_data,
                                        srp_bw,
                                        &feat_scales,
                                    )
                                } else if plain_axis_trees.get(t_idx).copied().unwrap_or(false) {
                                    tree.predict_raw_row_plain_axis(binned, row_data)
                                } else {
                                    tree.predict_raw_row(binned, row_data)
                                };
                                avg_probs[k] += c;
                            }
                            let inv_ntp = 1.0 / ntp as f64;
                            for p in avg_probs.iter_mut() {
                                *p *= inv_ntp;
                            }
                            let norm = avg_probs.iter().sum::<f64>();
                            if norm > 0.0 && norm.is_finite() {
                                for p in avg_probs.iter_mut() {
                                    *p = (*p / norm).max(1e-15).ln();
                                }
                            } else {
                                let log_uniform = -(n_classes as f64).ln();
                                for p in avg_probs.iter_mut() {
                                    *p = log_uniform;
                                }
                            }
                            avg_probs
                        } else {
                            // Standard: return raw margins. Python/sklearn wrappers own
                            // probability conversion, avoiding double-softmax underconfidence.
                            let mut scores = if self.class_base_scores.len() == n_classes {
                                self.class_base_scores.clone()
                            } else {
                                vec![0.0f64; n_classes]
                            };
                            for (t_idx, tree) in trees.iter().enumerate() {
                                let w = if has_dart_w && t_idx < dart_w.len() {
                                    dart_w[t_idx]
                                } else {
                                    1.0
                                };
                                let c = if let Some(ref row_bins) = eval_bins_row_major {
                                    tree.predict_binned_plain_row_major(row_bins, n_features, row)
                                } else if let Some(ref bins) = eval_bins {
                                    if binned_plain_trees.get(t_idx).copied().unwrap_or(false) {
                                        tree.predict_binned_plain_raw(bins, n_rows, row)
                                    } else {
                                        tree.predict_binned_raw(
                                            bins,
                                            n_rows,
                                            row,
                                            &eval_cll_hash_bins,
                                        )
                                    }
                                } else if use_srp {
                                    tree.predict_raw_row_soft(
                                        binned,
                                        row_data,
                                        srp_bw,
                                        &feat_scales,
                                    )
                                } else {
                                    tree.predict_raw_row(binned, row_data)
                                };
                                scores[(t_idx / ntp) % n_classes] += lr * w * c;
                            }
                            scores
                        }
                    })
                    .collect();
                preds_2d.into_iter().flatten().collect()
            } else {
                let base = self.base_score;
                let init_score_ref = init_score_vec.as_deref();
                let is_poisson = self.task == "poisson";
                let cat_offset_maps = &self.cat_offset_maps;
                let has_cat_offsets = self.task == "binary" && !cat_offset_maps.is_empty();
                (0..n_rows)
                    .into_par_iter()
                    .map(|row| {
                        let row_data = &x_data[row * n_features..(row + 1) * n_features];
                        // Per-row offset: init_score if user supplied one, else
                        // the global base_score (legacy behavior).
                        let mut sum = match init_score_ref {
                            Some(s) => s[row],
                            None => base,
                        };
                        for (t_idx, tree) in trees.iter().enumerate() {
                            let w = if has_dart_w && t_idx < dart_w.len() {
                                dart_w[t_idx]
                            } else {
                                1.0
                            };
                            let c = if let Some(ref row_bins) = eval_bins_row_major {
                                tree.predict_binned_plain_row_major(row_bins, n_features, row)
                            } else if let Some(ref bins) = eval_bins {
                                if tree.has_self_score_splits() {
                                    tree.predict_binned_raw_with_score(
                                        bins,
                                        n_rows,
                                        row,
                                        &eval_cll_hash_bins,
                                        sum,
                                    )
                                } else if binned_plain_trees.get(t_idx).copied().unwrap_or(false) {
                                    tree.predict_binned_plain_raw(bins, n_rows, row)
                                } else {
                                    tree.predict_binned_raw(bins, n_rows, row, &eval_cll_hash_bins)
                                }
                            } else if tree.has_self_score_splits() {
                                tree.predict_raw_row_with_score(binned, row_data, sum)
                            } else if use_srp {
                                tree.predict_raw_row_soft(binned, row_data, srp_bw, &feat_scales)
                            } else if plain_axis_trees.get(t_idx).copied().unwrap_or(false) {
                                tree.predict_raw_row_plain_axis(binned, row_data)
                            } else {
                                tree.predict_raw_row(binned, row_data)
                            };
                            sum += lr * w * c;
                        }
                        if has_cat_offsets {
                            let raw_row =
                                &x_data_raw[row * n_features_raw..(row + 1) * n_features_raw];
                            for (feat, map) in cat_offset_maps.iter().enumerate() {
                                if map.is_empty() || feat >= n_features_raw {
                                    continue;
                                }
                                let v = raw_row[feat];
                                if v.is_finite() {
                                    if let Some(off) = map.get(&(v as i64)) {
                                        sum += *off;
                                    }
                                }
                            }
                        }
                        if is_poisson {
                            sum.exp()
                        } else {
                            sum
                        }
                    })
                    .collect()
            }
        });

        Ok(PyArray1::from_vec(py, result))
    }

    /// Post-fit setter for soft_predict_bandwidth — enables auto-bandwidth SRP:
    /// train once (fast), then sweep bandwidths on val data and pick the best.
    pub fn set_soft_predict_bandwidth(&mut self, bw: f64) {
        self.soft_predict_bandwidth = bw.clamp(0.0, 10.0);
    }

    /// PRM — Posterior Refinement Marginalization predict. Per-node confidence-aware
    /// pruning of tree refinements. τ=0 = identity (standard predict). τ>0 shrinks
    /// deep low-confidence refinements toward ancestor values. Returns flat Vec of
    /// length n_rows (regression/binary) or n_rows*n_classes (multiclass).
    pub fn predict_pruned(
        &self,
        py: Python<'_>,
        x: Bound<'_, PyAny>,
        tau: f64,
    ) -> PyResult<Vec<f64>> {
        let x_array: Bound<'_, PyArray2<f64>> = x.extract()?;
        let n_rows = x_array.shape()[0];
        let n_features_raw = x_array.shape()[1];
        let x_owned = x_array.to_owned_array();
        let x_standard = if x_owned.is_standard_layout() {
            x_owned
        } else {
            x_owned.as_standard_layout().into_owned()
        };
        let x_data_raw: Vec<f64> = x_standard.into_raw_vec_and_offset().0;
        let binned = match &self.binned {
            Some(b) => b,
            None => {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Model not trained yet. Call fit() first.",
                ))
            }
        };

        let (x_data, n_features) = self.extend_raw_matrix(&x_data_raw, n_rows, n_features_raw);

        let result = py.detach(|| {
            let lr = if self.task == "multiclass" {
                self.multiclass_tree_lr()
            } else {
                self.learning_rate
            };
            let trees = &self.trees;
            if self.task == "multiclass" {
                let n_classes = self.n_classes;
                let ntp = self.multiclass_trees_per_class_round();
                let use_prob_avg = self.prob_avg && ntp > 1;
                let preds_2d: Vec<Vec<f64>> = (0..n_rows)
                    .into_par_iter()
                    .map(|row| {
                        let row_data = &x_data[row * n_features..(row + 1) * n_features];
                        if use_prob_avg {
                            let mut avg_probs = vec![0.0f64; n_classes];
                            for (t_idx, tree) in trees.iter().enumerate() {
                                let k = (t_idx / ntp) % n_classes;
                                avg_probs[k] += tree.predict_raw_row_pruned(binned, row_data, tau);
                            }
                            let inv_ntp = 1.0 / ntp as f64;
                            for p in avg_probs.iter_mut() {
                                *p *= inv_ntp;
                            }
                            let norm = avg_probs.iter().sum::<f64>();
                            if norm > 0.0 && norm.is_finite() {
                                for p in avg_probs.iter_mut() {
                                    *p = (*p / norm).max(1e-15).ln();
                                }
                            } else {
                                let log_uniform = -(n_classes as f64).ln();
                                for p in avg_probs.iter_mut() {
                                    *p = log_uniform;
                                }
                            }
                            avg_probs
                        } else {
                            let mut scores = if self.class_base_scores.len() == n_classes {
                                self.class_base_scores.clone()
                            } else {
                                vec![0.0f64; n_classes]
                            };
                            for (t_idx, tree) in trees.iter().enumerate() {
                                scores[(t_idx / ntp) % n_classes] +=
                                    lr * tree.predict_raw_row_pruned(binned, row_data, tau);
                            }
                            scores
                        }
                    })
                    .collect();
                preds_2d.into_iter().flatten().collect()
            } else {
                let base = self.base_score;
                (0..n_rows)
                    .into_par_iter()
                    .map(|row| {
                        let row_data = &x_data[row * n_features..(row + 1) * n_features];
                        let mut sum = base;
                        for tree in trees.iter() {
                            let c = if tree.has_self_score_splits() {
                                tree.predict_raw_row_with_score(binned, row_data, sum)
                            } else {
                                tree.predict_raw_row_pruned(binned, row_data, tau)
                            };
                            sum += lr * c;
                        }
                        sum
                    })
                    .collect()
            }
        });
        Ok(result)
    }

    /// Predict with a per-tree 0/1 mask (1 = use tree, 0 = skip). Enables test-time
    /// MC-dropout ensembles: predict multiple times with different random masks,
    /// average. Variance reduction without bias for correlated tree ensembles.
    /// Returns flat Vec of length n_rows (regression/binary) or n_rows*n_classes (multiclass).
    pub fn predict_with_tree_mask(
        &self,
        py: Python<'_>,
        x: Bound<'_, PyAny>,
        mask: Vec<u8>,
    ) -> PyResult<Vec<f64>> {
        let x_array: Bound<'_, PyArray2<f64>> = x.extract()?;
        let n_rows = x_array.shape()[0];
        let n_features_raw = x_array.shape()[1];
        let x_owned = x_array.to_owned_array();
        let x_standard = if x_owned.is_standard_layout() {
            x_owned
        } else {
            x_owned.as_standard_layout().into_owned()
        };
        let x_data_raw: Vec<f64> = x_standard.into_raw_vec_and_offset().0;
        let binned = match &self.binned {
            Some(b) => b,
            None => {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Model not trained yet. Call fit() first.",
                ))
            }
        };

        let (x_data, n_features) = self.extend_raw_matrix(&x_data_raw, n_rows, n_features_raw);

        let result = py.detach(|| {
            let lr = if self.task == "multiclass" {
                self.multiclass_tree_lr()
            } else {
                self.learning_rate
            };
            let trees = &self.trees;

            if self.task == "multiclass" {
                let n_classes = self.n_classes;
                let ntp = self.multiclass_trees_per_class_round();
                let preds_2d: Vec<Vec<f64>> = (0..n_rows)
                    .into_par_iter()
                    .map(|row| {
                        let row_data = &x_data[row * n_features..(row + 1) * n_features];
                        if self.prob_avg && ntp > 1 {
                            let mut avg_probs = vec![0.0f64; n_classes];
                            let mut counts = vec![0usize; n_classes];
                            for (t_idx, tree) in trees.iter().enumerate() {
                                if t_idx < mask.len() && mask[t_idx] == 0 {
                                    continue;
                                }
                                let k = (t_idx / ntp) % n_classes;
                                avg_probs[k] += tree.predict_raw_row(binned, row_data);
                                counts[k] += 1;
                            }
                            for k in 0..n_classes {
                                if counts[k] > 0 {
                                    avg_probs[k] /= counts[k] as f64;
                                }
                            }
                            let norm = avg_probs.iter().sum::<f64>();
                            if norm > 0.0 && norm.is_finite() {
                                for p in avg_probs.iter_mut() {
                                    *p = (*p / norm).max(1e-15).ln();
                                }
                            } else {
                                let log_uniform = -(n_classes as f64).ln();
                                for p in avg_probs.iter_mut() {
                                    *p = log_uniform;
                                }
                            }
                            avg_probs
                        } else {
                            let mut scores = if self.class_base_scores.len() == n_classes {
                                self.class_base_scores.clone()
                            } else {
                                vec![0.0f64; n_classes]
                            };
                            for (t_idx, tree) in trees.iter().enumerate() {
                                if t_idx < mask.len() && mask[t_idx] == 0 {
                                    continue;
                                }
                                scores[(t_idx / ntp) % n_classes] +=
                                    lr * tree.predict_raw_row(binned, row_data);
                            }
                            scores
                        }
                    })
                    .collect();
                preds_2d.into_iter().flatten().collect()
            } else {
                let base = self.base_score;
                (0..n_rows)
                    .into_par_iter()
                    .map(|row| {
                        let row_data = &x_data[row * n_features..(row + 1) * n_features];
                        let mut sum = base;
                        for (t_idx, tree) in trees.iter().enumerate() {
                            if t_idx < mask.len() && mask[t_idx] == 0 {
                                continue;
                            }
                            let c = if tree.has_self_score_splits() {
                                tree.predict_raw_row_with_score(binned, row_data, sum)
                            } else {
                                tree.predict_raw_row(binned, row_data)
                            };
                            sum += lr * c;
                        }
                        sum
                    })
                    .collect()
            }
        });
        Ok(result)
    }

    pub fn tree_info(&self) -> Vec<(usize, usize)> {
        self.trees.iter().map(|t| t.node_counts()).collect()
    }

    pub fn task_name(&self) -> String {
        self.task.clone()
    }

    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    pub fn split_op_counts(&self) -> (usize, usize, usize, usize, usize) {
        let mut axis = 0usize;
        let mut categorical = 0usize;
        let mut interval = 0usize;
        let mut oblique = 0usize;
        let mut cat_pair = 0usize;
        let Some(binned) = self.binned.as_ref() else {
            return (axis, categorical, interval, oblique, cat_pair);
        };

        for tree in &self.trees {
            for node in 0..tree.split_features.len() {
                let feat = tree.split_features[node];
                if feat == u32::MAX {
                    continue;
                }
                if node < tree.is_oblique_split.len() && tree.is_oblique_split[node] {
                    oblique += 1;
                } else if tree.is_cat_pair(node) {
                    cat_pair += 1;
                } else if node < tree.is_cat_split.len() && tree.is_cat_split[node] {
                    let feat_idx = feat as usize;
                    if feat_idx < binned.is_categorical.len() && !binned.is_categorical[feat_idx] {
                        interval += 1;
                    } else {
                        categorical += 1;
                    }
                } else {
                    axis += 1;
                }
            }
        }
        (axis, categorical, interval, oblique, cat_pair)
    }

    pub fn save_model(&self, path: String) -> PyResult<()> {
        if self.binned.is_none() || self.trees.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Model not trained yet. Call fit() before save_model().",
            ));
        }
        let mut snapshot = self.clone();
        if let Some(binned) = snapshot.binned.as_mut() {
            binned.strip_training_storage_for_prediction();
        }
        snapshot.tree_in_sample.clear();
        let envelope = ModelSnapshot {
            format_version: MODEL_FORMAT_VERSION,
            model: snapshot,
        };
        let bytes = serde_json::to_vec(&envelope).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("serialize model: {}", e))
        })?;
        fs::write(&path, bytes).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("write model '{}': {}", path, e))
        })
    }

    #[staticmethod]
    pub fn load_model(path: String) -> PyResult<Self> {
        let bytes = fs::read(&path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("read model '{}': {}", path, e))
        })?;
        let envelope: ModelSnapshot = serde_json::from_slice(&bytes).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "deserialize model '{}': {}",
                path, e
            ))
        })?;
        if envelope.format_version != MODEL_FORMAT_VERSION {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "unsupported model format version {} (expected {})",
                envelope.format_version, MODEL_FORMAT_VERSION
            )));
        }
        Ok(envelope.model)
    }

    pub fn lookup_op_counts(&self) -> (usize, usize, usize, usize, usize) {
        let mut single_cat = 0usize;
        let mut pair_cat = 0usize;
        let mut triple_cat = 0usize;
        let mut numeric = 0usize;
        let mut bins = 0usize;
        for tree in &self.trees {
            for lookup in tree.cat_lookups.iter().flatten() {
                bins += lookup.bin_values.len();
                if lookup.is_numeric {
                    numeric += 1;
                } else if lookup.feature3 != u32::MAX {
                    triple_cat += 1;
                } else if lookup.feature2 != u32::MAX {
                    pair_cat += 1;
                } else {
                    single_cat += 1;
                }
            }
        }
        (single_cat, pair_cat, triple_cat, numeric, bins)
    }

    /// PASA: validation-loss history recorded during training (one entry per
    /// early-stopping round). Empty if no eval set was provided during fit().
    pub fn val_loss_history(&self) -> Vec<f64> {
        self.val_losses.clone()
    }

    /// PASA: index into `self.trees` corresponding to the best validation
    /// score (argmin of val_losses). Used by Python-side plateau averaging
    /// to construct the plateau range around the argmin.
    pub fn best_tree_count(&self) -> usize {
        self.best_round
    }

    /// Per-sample leaf IDs for each tree. Returns flat Vec of length n_rows * n_trees
    /// (row-major). Enables "lazy" leaf-fingerprint KNN at predict time (Friedman-Kohavi-Yun
    /// AAAI 1996 style lazy trees adapted to GBM).
    pub fn leaf_indices(&self, py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
        let x_array: Bound<'_, PyArray2<f64>> = x.extract()?;
        let n_rows = x_array.shape()[0];
        let n_features_raw = x_array.shape()[1];
        let x_owned = x_array.to_owned_array();
        let x_standard = if x_owned.is_standard_layout() {
            x_owned
        } else {
            x_owned.as_standard_layout().into_owned()
        };
        let x_data_raw: Vec<f64> = x_standard.into_raw_vec_and_offset().0;

        let binned = match &self.binned {
            Some(b) => b,
            None => {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Model not trained yet. Call fit() first.",
                ))
            }
        };

        let (x_data, n_features) = self.extend_raw_matrix(&x_data_raw, n_rows, n_features_raw);

        let n_trees = self.trees.len();
        let result = py.detach(|| {
            let trees = &self.trees;
            let out: Vec<u32> = (0..n_rows)
                .into_par_iter()
                .flat_map(|row| {
                    let row_data = &x_data[row * n_features..(row + 1) * n_features];
                    let leaf_ids: Vec<u32> = trees
                        .iter()
                        .map(|tree| tree.route_to_leaf_row(binned, row_data) as u32)
                        .collect();
                    leaf_ids
                })
                .collect();
            (out, n_trees)
        });

        Ok(result.0)
    }

    /// Predict using only the first `n_trees` trees of the ensemble.
    /// For multiclass, n_trees should be a multiple of
    /// n_classes * multiclass_trees_per_class_round().
    /// Returns raw logits for binary/multiclass, raw values for regression.
    pub fn predict_truncated(
        &self,
        py: Python<'_>,
        x: Bound<'_, PyAny>,
        n_trees: usize,
    ) -> PyResult<Vec<f64>> {
        let x_array: Bound<'_, PyArray2<f64>> = x.extract()?;
        let n_rows = x_array.shape()[0];
        let n_features_raw = x_array.shape()[1];
        let x_owned = x_array.to_owned_array();
        let x_standard = if x_owned.is_standard_layout() {
            x_owned
        } else {
            x_owned.as_standard_layout().into_owned()
        };
        let x_data_raw: Vec<f64> = x_standard.into_raw_vec_and_offset().0;

        let binned = match &self.binned {
            Some(b) => b,
            None => {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Model not trained yet. Call fit() first.",
                ))
            }
        };

        let (x_data, n_features) = self.extend_raw_matrix(&x_data_raw, n_rows, n_features_raw);

        let n_trees_total = self.trees.len();
        let n_use = n_trees.min(n_trees_total);

        let result = py.detach(|| {
            let lr = if self.task == "multiclass" {
                self.multiclass_tree_lr()
            } else {
                self.learning_rate
            };
            let trees = &self.trees[..n_use];
            let dart_w = &self.dart_tree_weights;
            let has_dart_w = !dart_w.is_empty();

            if self.task == "multiclass" {
                let n_classes = self.n_classes;
                let ntp = self.multiclass_trees_per_class_round();
                let use_prob_avg = self.prob_avg && ntp > 1;
                let preds_2d: Vec<Vec<f64>> = (0..n_rows)
                    .into_par_iter()
                    .map(|row| {
                        let row_data = &x_data[row * n_features..(row + 1) * n_features];
                        if use_prob_avg {
                            let mut avg_probs = vec![0.0f64; n_classes];
                            for (t_idx, tree) in trees.iter().enumerate() {
                                let k = (t_idx / ntp) % n_classes;
                                avg_probs[k] += tree.predict_raw_row(binned, row_data);
                            }
                            let inv_ntp = 1.0 / ntp as f64;
                            for p in avg_probs.iter_mut() {
                                *p *= inv_ntp;
                            }
                            let norm = avg_probs.iter().sum::<f64>();
                            if norm > 0.0 && norm.is_finite() {
                                for p in avg_probs.iter_mut() {
                                    *p = (*p / norm).max(1e-15).ln();
                                }
                            } else {
                                let log_uniform = -(n_classes as f64).ln();
                                for p in avg_probs.iter_mut() {
                                    *p = log_uniform;
                                }
                            }
                            avg_probs
                        } else {
                            let mut scores = if self.class_base_scores.len() == n_classes {
                                self.class_base_scores.clone()
                            } else {
                                vec![0.0f64; n_classes]
                            };
                            for (t_idx, tree) in trees.iter().enumerate() {
                                let w = if has_dart_w && t_idx < dart_w.len() {
                                    dart_w[t_idx]
                                } else {
                                    1.0
                                };
                                scores[(t_idx / ntp) % n_classes] +=
                                    lr * w * tree.predict_raw_row(binned, row_data);
                            }
                            scores
                        }
                    })
                    .collect();
                preds_2d.into_iter().flatten().collect()
            } else {
                let base = self.base_score;
                let is_poisson = self.task == "poisson";
                (0..n_rows)
                    .into_par_iter()
                    .map(|row| {
                        let row_data = &x_data[row * n_features..(row + 1) * n_features];
                        let mut sum = base;
                        for (t_idx, tree) in trees.iter().enumerate() {
                            let w = if has_dart_w && t_idx < dart_w.len() {
                                dart_w[t_idx]
                            } else {
                                1.0
                            };
                            let c = if tree.has_self_score_splits() {
                                tree.predict_raw_row_with_score(binned, row_data, sum)
                            } else {
                                tree.predict_raw_row(binned, row_data)
                            };
                            sum += lr * w * c;
                        }
                        if is_poisson {
                            sum.exp()
                        } else {
                            sum
                        }
                    })
                    .collect()
            }
        });

        Ok(result)
    }
}
