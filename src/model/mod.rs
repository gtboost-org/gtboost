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

#[derive(Clone, Serialize, Deserialize, Default)]
pub(super) struct VerticalPrior {
    mean: f64,
    edges: Vec<Vec<f64>>,
    effects: Vec<Vec<f64>>,
}

impl VerticalPrior {
    #[inline]
    fn is_active(&self) -> bool {
        !self.effects.is_empty()
    }

    #[inline]
    fn bin_index(edges: &[f64], value: f64) -> usize {
        if !value.is_finite() {
            return edges.len() + 1;
        }
        let mut lo = 0usize;
        let mut hi = edges.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if value <= edges[mid] {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo
    }

    fn predict_row(&self, row: &[f64]) -> f64 {
        let mut out = self.mean;
        for (j, (edges, effects)) in self.edges.iter().zip(self.effects.iter()).enumerate() {
            if effects.len() <= 1 || j >= row.len() {
                continue;
            }
            let bin = Self::bin_index(edges, row[j]);
            if bin < effects.len() {
                out += effects[bin];
            }
        }
        out
    }

    fn predict_matrix(&self, x_data: &[f64], n_rows: usize, n_features: usize) -> Vec<f64> {
        if !self.is_active() || x_data.len() < n_rows.saturating_mul(n_features) {
            return vec![self.mean; n_rows];
        }
        (0..n_rows)
            .map(|row| self.predict_row(&x_data[row * n_features..(row + 1) * n_features]))
            .collect()
    }
}

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
    #[serde(default)]
    pub(super) vertical_init: bool,
    #[serde(default)]
    pub(super) vertical_init_cycles: usize,
    #[serde(default)]
    pub(super) vertical_prior: VerticalPrior,
    pub(super) lambda_reg: f64,
    pub(super) gamma: f64,
    pub(super) min_child_weight: f64,
    pub(super) colsample_bytree: f64,
    pub(super) task: String,
    pub(super) num_bins: usize,
    pub(super) seed: u64,
    pub(super) grow_policy: String, // "depthwise", "leafwise", "oblivious", or "adaptive"
    pub(super) max_leaves: usize,   // max leaves for leafwise (0 = use 2^max_depth)
    pub(super) n_refine: usize,     // refinement passes (0 = disabled)
    pub(super) n_leaf_splits: usize, // post-refinement leaf splitting passes (0 = disabled)
    pub(super) refine_every: usize, // refine all trees every N rounds during training (0 = only at end)
    pub(super) early_stopping_rounds: usize, // 0 = disabled
    pub(super) eval_metric: String, // Validation metric for early stopping ("auto"/"logloss"/"auc").
    #[serde(default)]
    pub(super) verbose: usize, // progress logging interval in rounds (0 = silent)
    pub(super) l1_reg: f64, // L1 regularization on leaf values during refinement (0 = disabled)
    pub(super) refine_alpha: f64, // refinement shrinkage: blend w_old + alpha*(w_new - w_old), 1.0 = full step
    pub(super) honest: bool, // honest estimation: build structure on half data, leaf values on other half
    pub(super) honest_fraction: f64, // fraction of subsampled data for estimation (0.0 = use complement of subsample)
    pub(super) honest_arbitration: bool, // NEW 2026-06-11: splits must also show positive Newton gain on the estimation fold (kills winner's-curse splits per node)
    pub(super) feature_gain_prior: Vec<f64>, // NEW 2026-06-11: consensus-guided boosting — per-feature gain multiplier distilled from fold-model replication (empty = off)
    pub(super) thermal: f64, // NEW 2026-06-11: thermal boosting — node-adaptive split sampling at the argmax-bias scale T_node ~ sqrt(2 ln K)/sqrt(n_node); 0 = argmax
    pub(super) thermal_n_exp: f64, // thermal schedule shape: evidence exponent (T ~ 1/n^exp)
    pub(super) thermal_depth_gamma: f64, // thermal schedule shape: per-depth multiplier
    pub(super) stein_leaves: bool, // SP-GBDT — calibration-preserving positive-part Stein shrinkage of each tree's leaf vector at estimated gradient-noise scale (owner proposal). 2026-06-12 v2: leaf CONTRASTS shrink toward the global Newton step (not toward zero) — the constant score-shift direction is preserved per SP-GBDT §2.5. This repaired the old clean-data harm (friedman-clean +2.3% -> -0.4%); 10-seed profile: diabetes -1.1%, friedman-noisy -0.8%, clean neutral — never-hurts, no gate needed.
    pub(super) stein_levels: bool, // NEW 2026-06-12: tree-structured Stein — per-depth-level pooled shrinkage of path contrasts (smooth adaptive depth; pairs with deep trees)
    #[serde(default)]
    pub(super) split_audit_fraction: f64, // Audit-only split debiasing: validate split choice on this held-out fraction, then refit leaves on all rows.
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
    #[serde(default)]
    pub(super) cat_prototype_bins: usize, // Residual-prototype categorical split compression. 0 = ordinary full scan.
    #[serde(default)]
    pub(super) cat_audit_strength: f64, // Categorical-only support/search audit penalty. 0.0 = disabled.
    #[serde(default)]
    pub(super) split_contrast_penalty: f64, // Shrink weak left/right leaf contrasts after split gain scoring.
    #[serde(default)]
    pub(super) signal_gate: f64, // Free trees: admit a split only if gain > gate × within-node permutation null gain. 0.0 = off.
    #[serde(default)]
    pub(super) supervised_bins: bool, // DP bins: build a 4x fine quantile grid, DP-merge to num_bins where the initial gradient profile changes. false = plain quantile.
    #[serde(default)]
    pub(super) fold_ordered: usize, // Fold-Ordered Boosting: F honest margin tracks; each row's gradients come from a model whose leaf values exclude that row's fold (CatBoost-style ordered boosting at fold granularity). 0 = off.
    // ── CFE: Categorical Fold Evidence (native fast replacement for PCF) ──
    // Cross-fit tuple posteriors over categorical columns: leak-safe like
    // CatBoost's ordered TS but fold-deterministic (no permutation noise, no
    // cold-start rows), with utility-screened pair/triple crosses and PACT-
    // style naive-Bayes aggregate columns appended as derived features.
    #[serde(default)]
    pub(super) cat_fold_evidence: bool,
    #[serde(default)]
    pub(super) cfe_folds: usize, // cross-fit folds (default 5)
    #[serde(default)]
    pub(super) cfe_max_pairs: usize, // pair crosses kept (default 12)
    #[serde(default)]
    pub(super) cfe_max_triples: usize, // triple crosses kept (default 8)
    #[serde(default)]
    pub(super) cfe_max_quads: usize, // arity-4 crosses kept (default 6) — exhaustive static deep crosses, the region CatBoost's greedy in-tree combos miss
    #[serde(default)]
    pub(super) cfe_dual_prior: bool, // emit each tuple's lift at TWO shrinkage strengths (m, 16m) so trees pick the confidence level per region
    #[serde(default)]
    pub(super) cfe_smooth: f64, // EB prior strength m (default 20.0)
    #[serde(default = "serde_true")]
    pub(super) cfe_counter: bool, // emit log-count (Counter-CTR) companion columns
    #[serde(default = "serde_true")]
    pub(super) cfe_aggmax: bool, // emit per-arity max-evidence aggregate columns
    #[serde(default)]
    pub(super) cfe_demote_raw: bool, // flatten raw HIGH-CARD cat columns when CFE is active (their native splits are the overfit liability the evidence replaces)
    #[serde(default)]
    pub(super) cfe_demote_min_card: usize, // cardinality above which raw cats are demoted (default 64)
    #[serde(default)]
    pub(super) cfe_tuples: Vec<Vec<usize>>, // trained tuple feature sets
    #[serde(default)]
    pub(super) cfe_tables: Vec<HashMap<i64, (f64, Vec<f64>)>>, // key -> (count, per-output sums), full train
    #[serde(default)]
    pub(super) cfe_prior: Vec<f64>, // per-output global prior
    #[serde(default)]
    pub(super) cfe_residual_rounds: usize, // stage-2 residual evidence: internal warmup rounds (0 = off)
    #[serde(default)]
    pub(super) cfe_resid_tables: Vec<HashMap<i64, (f64, Vec<f64>)>>, // key -> (count, residual-gradient sums)
    #[serde(default)]
    pub(super) cfe_resid_prior: f64, // global residual prior (mean target)
    #[serde(default)]
    pub(super) cfe_n_out: usize, // 1 (binary/regression) or K (multiclass)
    pub(super) self_score_splits: bool, // Allow trees to split on the current boosting margin.
    pub(super) hetero_trees: bool, // Heterogeneous sub-trees: cycle (depth, lambda) across sub-trees for structural diversity
    pub(super) dart_rate: f64, // DART: fraction of trees to drop per round during training (0.0 = disabled)
    pub(super) max_delta_step: f64, // Max leaf value magnitude (0.0 = unlimited, >0 clips leaf values)
    #[serde(default)]
    pub(super) main_effect_interval: usize, // Insert one residual-chosen single-feature tree every N boosting rounds (0 = disabled).
    #[serde(default)]
    pub(super) main_effect_depth: usize, // Depth for inserted main-effect trees; 1 = stump, >1 = piecewise univariate.
    pub(super) cyclic_features: bool, // EBM-style: cycle through features, each tree uses one feature (false = disabled)
    pub(super) cyclic_max_features_per_round: usize, // Sparse vertical attention: max cyclic feature trees per boosting round (0 = all)
    pub(super) auto_interactions: bool, // Auto-generate pairwise numeric product features in binning (false = disabled)
    pub(super) auto_cat_interactions: bool, // Auto-generate hashed categorical pair features in binning (false = disabled)
    pub(super) max_interaction_features: usize, // Max product feature pairs to generate (0 = unlimited)
    pub(super) lambda_schedule: f64, // Adaptive lambda: effective_lambda = lambda * (1 + lambda_schedule * round/n_rounds) (0.0 = off)
    pub(super) use_bootstrap: bool, // Bootstrap sampling: sample rows with replacement (RF-style bagging)
    pub(super) extra_trees: bool, // Extra Trees: random split thresholds instead of optimal (massive variance reduction)
    pub(super) label_smooth: f64, // Label smoothing for multiclass: target = (1-ε)*one_hot + ε/K (0.0 = off)
    pub(super) multi_output_tree: bool, // Multi-output trees for multiclass: shared tree structure across all K classes
    #[serde(default)]
    pub(super) multiclass_pair_sequence: bool, // Confusion-pair boosting: one contrast tree per multiclass round.
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
    pub(super) sparse_oblique_splits: bool, // Sparse 2-feature oblique split candidates for depthwise/leafwise trees.
    pub(super) interval_splits: bool, // Bounded numeric interval split candidates: low <= x_j <= high.
    pub(super) sibling_block_correction: f64, // Joint per-round least-squares rescale for sibling trees (0.0 = disabled).
    pub(super) adam_beta2: f64, // Adam 2nd-moment decay (0.0 = disabled). Uses grad_momentum as β1.
    pub(super) adam_eps: f64,   // Adam epsilon for stability in denom
    pub(super) ortho_alpha: f64, // Gradient orthogonalization vs previous tree's leaves (0.0 = disabled)
    pub(super) split_criterion: String, // "newton" (default), "rank" (Wilcoxon-like), or "sign" (distribution-free)
    pub(super) rank_mix_alpha: f64, // MGB: for task="binary", blend α·g_rank + (1-α)·g_logloss. 0 = pure binary.
    pub(super) rank_mix_start_frac: f64, // Late-MGB: delay rank-mix until this training fraction. 0 = active from start.
    pub(super) rank_pair_temperature: f64, // Pairwise rank-loss score scale. Larger values keep AUC gradients from saturating.
    pub(super) binary_focus_gamma: f64, // Hard-row focus for binary loss: multiply g/h by (2*|y-p|)^gamma. 0 = off.
    pub(super) binary_focus_end_frac: f64, // Focus warmup: if >0, turn binary_focus_gamma off after this training fraction.
    #[serde(default)]
    pub(super) residual_focus_alpha: f64, // Generic residual-hardness focus: multiply g/h by 1+alpha*|g|/(median|g|+|g|). 0 = off.
    #[serde(default)]
    pub(super) residual_focus_max_scale: f64, // Max scale for residual focus. Default/low values are clamped to >=1.
    #[serde(default)]
    pub(super) residual_focus_mode: String, // "full" = split+leaf, "split_only" = focused structure with raw Newton leaves.
    #[serde(default)]
    pub(super) residual_focus_hessian_mode: String, // "equal" = h*=scale, "none" = raw h, "true" = detached shaped-loss curvature.
    #[serde(default)]
    pub(super) residual_focus_redescend_tau: f64, // >0 focuses moderate residuals and returns extreme residuals toward baseline.
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
    #[serde(default)]
    pub(super) ordered_ctr_priors: Vec<f64>, // Per-CTR-column fallback priors.
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
    pub(super) soft_leaf_bandwidth: f64, // SCB: train-consistent smooth boosting. After fit, refit constant leaves under soft routing at this bandwidth, then predict softly. 0 = off. Binary/regression only.
    pub(super) soft_leaf_passes: usize, // number of cyclic soft leaf-refit passes over the trees (coordinate descent on the residual). Typical 1-2.
    pub(super) leaf_var_shrink: f64, // VLS: variance-aware leaf shrinkage exponent. Scale each leaf by gradient reliability^this. 0 = off, typical 0.5-2.0.
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
        self.vertical_prior = VerticalPrior::default();
        self.cat_offset_maps.clear();
        self.numeric_interaction_pairs.clear();
        self.numeric_interaction_edges.clear();
        self.categorical_interaction_pairs.clear();
        self.categorical_interaction_edges.clear();
        self.sumdiff_pairs.clear();
        self.sumdiff_edges.clear();
        self.feature_usage_ema.clear();
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
        self.cfe_tuples.clear();
        self.cfe_tables.clear();
        self.cfe_prior.clear();
        self.cfe_n_out = 0;
        self.cfe_resid_tables.clear();
        self.cfe_resid_prior = 0.0;
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

    fn fit_vertical_prior_for_binary(
        &mut self,
        x_data: &[f64],
        y_data: &[f64],
        n_rows: usize,
        n_features: usize,
        sample_weight_data: Option<&[f64]>,
    ) -> Option<Vec<f64>> {
        if x_data.len() < n_rows.saturating_mul(n_features) || y_data.len() != n_rows {
            self.vertical_prior = VerticalPrior::default();
            return None;
        }

        let max_bins = ((n_rows as f64).sqrt().round() as usize).clamp(4, 64);
        let reg = (0.35 * (n_rows as f64).sqrt().max(n_features as f64)).max(1.0);
        let cat_features = if self.cat_features.len() == n_features {
            self.cat_features.clone()
        } else {
            vec![false; n_features]
        };

        let mut edges: Vec<Vec<f64>> = Vec::with_capacity(n_features);
        let mut effects: Vec<Vec<f64>> = Vec::with_capacity(n_features);
        let mut bins: Vec<Vec<usize>> = Vec::with_capacity(n_features);

        for feat in 0..n_features {
            if cat_features.get(feat).copied().unwrap_or(false) {
                edges.push(Vec::new());
                effects.push(vec![0.0]);
                bins.push(vec![0usize; n_rows]);
                continue;
            }
            let mut vals: Vec<f64> = (0..n_rows)
                .map(|row| x_data[row * n_features + feat])
                .filter(|v| v.is_finite())
                .collect();
            if vals.len() < 8 {
                edges.push(Vec::new());
                effects.push(vec![0.0]);
                bins.push(vec![0usize; n_rows]);
                continue;
            }
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let min_v = vals[0];
            let max_v = vals[vals.len() - 1];
            if !(max_v > min_v) {
                edges.push(Vec::new());
                effects.push(vec![0.0]);
                bins.push(vec![0usize; n_rows]);
                continue;
            }
            let n_bins = max_bins.min(((vals.len() as f64).sqrt().round() as usize).max(4));
            let mut feat_edges = Vec::with_capacity(n_bins.saturating_sub(1));
            for q in 1..n_bins {
                let pos =
                    ((q as f64) * ((vals.len() - 1) as f64) / (n_bins as f64)).round() as usize;
                let edge = vals[pos.min(vals.len() - 1)];
                if feat_edges.last().map(|last| edge > *last).unwrap_or(true) {
                    feat_edges.push(edge);
                }
            }
            let mut feat_bins = Vec::with_capacity(n_rows);
            for row in 0..n_rows {
                let v = x_data[row * n_features + feat];
                feat_bins.push(VerticalPrior::bin_index(&feat_edges, v));
            }
            let n_effects = feat_edges.len() + 2;
            edges.push(feat_edges);
            effects.push(vec![0.0; n_effects]);
            bins.push(feat_bins);
        }

        let row_weight = |row: usize, yv: f64, this: &GTBoostModel| -> f64 {
            let sw = sample_weight_data
                .and_then(|w| w.get(row).copied())
                .filter(|v| v.is_finite() && *v > 0.0)
                .unwrap_or(1.0);
            let cw = if this.class_weights.len() >= 2 {
                if yv > 0.5 {
                    this.class_weights[1].max(0.0)
                } else {
                    this.class_weights[0].max(0.0)
                }
            } else {
                1.0
            };
            sw * cw
        };

        let logistic_loss = |pred: &[f64], eff: Option<(&[usize], &[f64])>| -> f64 {
            let mut loss = 0.0f64;
            for row in 0..n_rows {
                let mut z = pred[row];
                if let Some((feat_bins, feat_effects)) = eff {
                    let b = feat_bins[row];
                    if b < feat_effects.len() {
                        z += feat_effects[b];
                    }
                }
                let y = if y_data[row] > 0.5 { 1.0 } else { 0.0 };
                let w = row_weight(row, y, self);
                let zc = z.clamp(-30.0, 30.0);
                loss += w * ((1.0 + zc.exp()).ln() - y * zc);
            }
            loss
        };

        let mut pred = vec![self.base_score; n_rows];
        let cycles = self.vertical_init_cycles.max(1).min(8);
        for _ in 0..cycles {
            for feat in 0..n_features {
                if effects[feat].len() <= 1 {
                    continue;
                }
                for row in 0..n_rows {
                    pred[row] -= effects[feat][bins[feat][row]];
                }

                let loss_before = logistic_loss(&pred, None);
                let mut g_sums = vec![0.0f64; effects[feat].len()];
                let mut h_sums = vec![0.0f64; effects[feat].len()];
                for row in 0..n_rows {
                    let z = pred[row].clamp(-30.0, 30.0);
                    let p = 1.0 / (1.0 + (-z).exp());
                    let y = if y_data[row] > 0.5 { 1.0 } else { 0.0 };
                    let w = row_weight(row, y, self);
                    let b = bins[feat][row];
                    g_sums[b] += w * (p - y);
                    h_sums[b] += w * (p * (1.0 - p)).max(1e-6);
                }

                let mut new_effects = vec![0.0f64; effects[feat].len()];
                let mut center_num = 0.0;
                let mut center_den = 0.0;
                let mut edf = 0.0;
                for b in 0..new_effects.len() {
                    let h = h_sums[b];
                    if h <= 1e-12 {
                        continue;
                    }
                    let v = (-g_sums[b] / (h + reg)).clamp(-2.0, 2.0);
                    new_effects[b] = v;
                    center_num += v * h;
                    center_den += h;
                    edf += h / (h + reg);
                }
                if center_den > 0.0 {
                    let center = center_num / center_den;
                    for v in &mut new_effects {
                        *v -= center;
                    }
                }

                let loss_after = logistic_loss(&pred, Some((&bins[feat], &new_effects)));
                let penalty = 0.02 * edf.max(1.0);
                if loss_after + penalty < loss_before {
                    effects[feat] = new_effects;
                    for row in 0..n_rows {
                        pred[row] += effects[feat][bins[feat][row]];
                    }
                } else {
                    effects[feat].fill(0.0);
                }
            }
        }

        let has_signal = effects
            .iter()
            .any(|feat_effects| feat_effects.iter().any(|v| v.abs() > 1e-12));
        if !has_signal {
            self.vertical_prior = VerticalPrior::default();
            return None;
        }

        self.vertical_prior = VerticalPrior {
            mean: self.base_score,
            edges,
            effects,
        };
        Some(pred)
    }

    fn fit_vertical_prior_for_regression(
        &mut self,
        x_data: &[f64],
        y_data: &[f64],
        n_rows: usize,
        n_features: usize,
        sample_weight_data: Option<&[f64]>,
    ) -> Option<Vec<f64>> {
        if !self.vertical_init || n_rows == 0 || n_features == 0 {
            self.vertical_prior = VerticalPrior::default();
            return None;
        }
        if self.task == "binary" {
            return self.fit_vertical_prior_for_binary(
                x_data,
                y_data,
                n_rows,
                n_features,
                sample_weight_data,
            );
        }
        if self.task != "regression" {
            self.vertical_prior = VerticalPrior::default();
            return None;
        }
        if x_data.len() < n_rows.saturating_mul(n_features) || y_data.len() != n_rows {
            self.vertical_prior = VerticalPrior::default();
            return None;
        }

        let mut weight_sum = 0.0;
        let mut weighted_y_sum = 0.0;
        for i in 0..n_rows {
            let w = sample_weight_data
                .and_then(|sw| sw.get(i).copied())
                .filter(|v| v.is_finite() && *v > 0.0)
                .unwrap_or(1.0);
            weight_sum += w;
            weighted_y_sum += w * y_data[i];
        }
        let mean = if weight_sum > 0.0 {
            weighted_y_sum / weight_sum
        } else {
            y_data.iter().sum::<f64>() / n_rows as f64
        };

        let max_bins = ((n_rows as f64).sqrt().round() as usize).clamp(4, 64);
        // Empirical-Bayes style smoothing for one-feature vertical effects.
        // The prior is intentionally conservative: it is only an initializer,
        // and noisy marginal effects are worse than no initializer.
        let reg = (n_rows as f64).sqrt().max(n_features as f64).max(1.0);
        let cat_features = if self.cat_features.len() == n_features {
            self.cat_features.clone()
        } else {
            vec![false; n_features]
        };

        let mut edges: Vec<Vec<f64>> = Vec::with_capacity(n_features);
        let mut effects: Vec<Vec<f64>> = Vec::with_capacity(n_features);
        let mut bins: Vec<Vec<usize>> = Vec::with_capacity(n_features);

        for feat in 0..n_features {
            if cat_features.get(feat).copied().unwrap_or(false) {
                edges.push(Vec::new());
                effects.push(vec![0.0]);
                bins.push(vec![0usize; n_rows]);
                continue;
            }
            let mut vals: Vec<f64> = (0..n_rows)
                .map(|row| x_data[row * n_features + feat])
                .filter(|v| v.is_finite())
                .collect();
            if vals.len() < 4 {
                edges.push(Vec::new());
                effects.push(vec![0.0]);
                bins.push(vec![0usize; n_rows]);
                continue;
            }
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let min_v = vals[0];
            let max_v = vals[vals.len() - 1];
            if !(max_v > min_v) {
                edges.push(Vec::new());
                effects.push(vec![0.0]);
                bins.push(vec![0usize; n_rows]);
                continue;
            }
            let n_bins = max_bins.min(((vals.len() as f64).sqrt().round() as usize).max(4));
            let mut feat_edges = Vec::with_capacity(n_bins.saturating_sub(1));
            for q in 1..n_bins {
                let pos =
                    ((q as f64) * ((vals.len() - 1) as f64) / (n_bins as f64)).round() as usize;
                let edge = vals[pos.min(vals.len() - 1)];
                if feat_edges.last().map(|last| edge > *last).unwrap_or(true) {
                    feat_edges.push(edge);
                }
            }
            let mut feat_bins = Vec::with_capacity(n_rows);
            for row in 0..n_rows {
                let v = x_data[row * n_features + feat];
                feat_bins.push(VerticalPrior::bin_index(&feat_edges, v));
            }
            let n_effects = feat_edges.len() + 2;
            edges.push(feat_edges);
            effects.push(vec![0.0; n_effects]);
            bins.push(feat_bins);
        }

        let mut pred = vec![mean; n_rows];
        let cycles = self.vertical_init_cycles.max(1).min(8);
        for _ in 0..cycles {
            for feat in 0..n_features {
                if effects[feat].len() <= 1 {
                    continue;
                }
                for row in 0..n_rows {
                    pred[row] -= effects[feat][bins[feat][row]];
                }
                let mut counts = vec![0.0f64; effects[feat].len()];
                let mut sums = vec![0.0f64; effects[feat].len()];
                for row in 0..n_rows {
                    let b = bins[feat][row];
                    let w = sample_weight_data
                        .and_then(|sw| sw.get(row).copied())
                        .filter(|v| v.is_finite() && *v > 0.0)
                        .unwrap_or(1.0);
                    counts[b] += w;
                    sums[b] += w * (y_data[row] - pred[row]);
                }
                let mut new_effects = vec![0.0f64; effects[feat].len()];
                let mut center_num = 0.0;
                let mut center_den = 0.0;
                for b in 0..new_effects.len() {
                    new_effects[b] = sums[b] / (counts[b] + reg);
                    center_num += new_effects[b] * counts[b];
                    center_den += counts[b];
                }
                if center_den > 0.0 {
                    let center = center_num / center_den;
                    for v in &mut new_effects {
                        *v -= center;
                    }
                }

                let mut rss_before = 0.0;
                let mut rss_after = 0.0;
                for row in 0..n_rows {
                    let w = sample_weight_data
                        .and_then(|sw| sw.get(row).copied())
                        .filter(|v| v.is_finite() && *v > 0.0)
                        .unwrap_or(1.0);
                    let r0 = y_data[row] - pred[row];
                    let r1 = r0 - new_effects[bins[feat][row]];
                    rss_before += w * r0 * r0;
                    rss_after += w * r1 * r1;
                }
                let edf: f64 = counts
                    .iter()
                    .map(|&c| if c > 0.0 { c / (c + reg) } else { 0.0 })
                    .sum();
                let n_eff = center_den.max(1.0);
                let leverage = (edf / n_eff).clamp(0.0, 0.95);
                let gcv_after = rss_after / (1.0 - leverage).powi(2);
                if gcv_after < rss_before {
                    effects[feat] = new_effects;
                    for row in 0..n_rows {
                        pred[row] += effects[feat][bins[feat][row]];
                    }
                } else {
                    effects[feat].fill(0.0);
                }
            }
        }

        let has_signal = effects
            .iter()
            .any(|feat_effects| feat_effects.iter().any(|v| v.abs() > 1e-12));
        if !has_signal {
            self.vertical_prior = VerticalPrior::default();
            return None;
        }

        self.vertical_prior = VerticalPrior {
            mean,
            edges,
            effects,
        };
        Some(pred)
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
        vertical_init = false,
        vertical_init_cycles = 2,
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
        eval_metric = "auto".to_string(),
        verbose = 0,
        l1_reg = 0.0,
        refine_alpha = 1.0,
        honest = false,
        honest_fraction = 0.5,
        honest_arbitration = false,
        feature_gain_prior = Vec::<f64>::new(),
        thermal = 0.0,
        stein_leaves = false,
        stein_levels = false,
        thermal_n_exp = 0.5,
        thermal_depth_gamma = 1.0,
        colsample_bylevel = 1.0,
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
        leaf_linear = false,
        leaf_quadratic = false,
        leaf_correction = 0,
        grad_momentum = 0.0,
        gain_penalty = 0.0,
        split_pessimism = 0.0,
        cat_audit_strength = 0.0,
        split_contrast_penalty = 0.0,
        signal_gate = 0.0,
        supervised_bins = false,
        fold_ordered = 0,
        cat_fold_evidence = false,
        cfe_max_pairs = 12,
        cfe_max_triples = 8,
        cfe_max_quads = 6,
        cfe_smooth = 20.0,
        cfe_demote_raw = true,
        cfe_demote_min_card = 64,
        self_score_splits = false,
        hetero_trees = false,
        dart_rate = 0.0,
        max_delta_step = 0.0,
        main_effect_interval = 0,
        main_effect_depth = 1,
        cyclic_features = false,
        cyclic_max_features_per_round = 0,
        auto_interactions = false,
        auto_cat_interactions = false,
        max_interaction_features = 20,
        lambda_schedule = 0.0,
        use_bootstrap = false,
        extra_trees = false,
        label_smooth = 0.0,
        multi_output_tree = false,
        multiclass_pair_sequence = false,
        complement_debias_mode = 0,
        phase_schedule = "".to_string(),
        ncl_lambda = 0.0,
        adaptive_cyclic_order = false,
        cyclic_partner_features = false,
        cyclic_partner_min_pressure_ratio = 0.0,
        cyclic_partner_bins = 8,
        cyclic_feature_reuse = false,
        cyclic_revisit_trees = 0,
        adaptive_root_anchor = false,
        sparse_oblique_splits = false,
        interval_splits = false,
        adam_beta2 = 0.0,
        adam_eps = 1e-8,
        ortho_alpha = 0.0,
        split_criterion = "newton".to_string(),
        rank_mix_alpha = 0.0,
        rank_mix_start_frac = 0.0,
        rank_pair_temperature = 1.0,
        binary_focus_gamma = 0.0,
        binary_focus_end_frac = 0.0,
        residual_focus_alpha = 0.0,
        residual_focus_max_scale = 2.0,
        residual_focus_mode = "full".to_string(),
        residual_focus_hessian_mode = "equal".to_string(),
        residual_focus_redescend_tau = 0.0,
        feature_view_groups = Vec::new(),
        leaf_trim_pct = 0.0,
        leaf_median = false,
        leaf_median_blend = 0.0,
        leaf_mad_clip = 0.0,
        leaf_adaptive_blend_kappa = 0.0,
        ordered_boost = false,
        goss_top_rate = 0.0,
        goss_other_rate = 0.0,
        keep_all_trees = false,
        corrective_block_refit = false,
        corrective_blocks = 16,
        corrective_lambda = 1.0,
        corrective_blend = 1.0,
        corrective_audit_fraction = 0.0,
        leaf_eb = false,
        leaf_eb_min_trees = 10,
        leaf_eb_scale = 1.0,
        leaf_sibling_smooth = 0.0,
        hierarchical_shrinkage = 0.0,
        multiclass_coupled_leaves = false,
        class_weights = Vec::<f64>::new(),
        adaptive_leaf_experts = false,
        adaptive_cat_lookup_smooth = false,
        cat_offset_smooth = 0.0,
        cat_offset_passes = 0,
        expert_leaf_admission = false,
        expert_min_leaf = 64,
        newton_decrement_cap = 0.0,
        lookahead_alpha = 0.0,
        sign_confidence_gamma = 0.0,
        soft_predict_bandwidth = 0.0,
        soft_leaf_bandwidth = 0.0,
        soft_leaf_passes = 1,
        leaf_var_shrink = 0.0,
        jensen_train_temp = 1.0,
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
        vertical_init: bool,
        vertical_init_cycles: usize,
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
        eval_metric: String,
        verbose: usize,
        l1_reg: f64,
        refine_alpha: f64,
        honest: bool,
        honest_fraction: f64,
        honest_arbitration: bool,
        feature_gain_prior: Vec<f64>,
        thermal: f64,
        stein_leaves: bool,
        stein_levels: bool,
        thermal_n_exp: f64,
        thermal_depth_gamma: f64,
        colsample_bylevel: f64,
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
        leaf_linear: bool,
        leaf_quadratic: bool,
        leaf_correction: usize,
        grad_momentum: f64,
        gain_penalty: f64,
        split_pessimism: f64,
        cat_audit_strength: f64,
        split_contrast_penalty: f64,
        signal_gate: f64,
        supervised_bins: bool,
        fold_ordered: usize,
        cat_fold_evidence: bool,
        cfe_max_pairs: usize,
        cfe_max_triples: usize,
        cfe_max_quads: usize,
        cfe_smooth: f64,
        cfe_demote_raw: bool,
        cfe_demote_min_card: usize,
        self_score_splits: bool,
        hetero_trees: bool,
        dart_rate: f64,
        max_delta_step: f64,
        main_effect_interval: usize,
        main_effect_depth: usize,
        cyclic_features: bool,
        cyclic_max_features_per_round: usize,
        auto_interactions: bool,
        auto_cat_interactions: bool,
        max_interaction_features: usize,
        lambda_schedule: f64,
        use_bootstrap: bool,
        extra_trees: bool,
        label_smooth: f64,
        multi_output_tree: bool,
        multiclass_pair_sequence: bool,
        complement_debias_mode: u8,
        phase_schedule: String,
        ncl_lambda: f64,
        adaptive_cyclic_order: bool,
        cyclic_partner_features: bool,
        cyclic_partner_min_pressure_ratio: f64,
        cyclic_partner_bins: usize,
        cyclic_feature_reuse: bool,
        cyclic_revisit_trees: usize,
        adaptive_root_anchor: bool,
        sparse_oblique_splits: bool,
        interval_splits: bool,
        adam_beta2: f64,
        adam_eps: f64,
        ortho_alpha: f64,
        split_criterion: String,
        rank_mix_alpha: f64,
        rank_mix_start_frac: f64,
        rank_pair_temperature: f64,
        binary_focus_gamma: f64,
        binary_focus_end_frac: f64,
        residual_focus_alpha: f64,
        residual_focus_max_scale: f64,
        residual_focus_mode: String,
        residual_focus_hessian_mode: String,
        residual_focus_redescend_tau: f64,
        feature_view_groups: Vec<u32>,
        leaf_trim_pct: f64,
        leaf_median: bool,
        leaf_median_blend: f64,
        leaf_mad_clip: f64,
        leaf_adaptive_blend_kappa: f64,
        ordered_boost: bool,
        goss_top_rate: f64,
        goss_other_rate: f64,
        keep_all_trees: bool,
        corrective_block_refit: bool,
        corrective_blocks: usize,
        corrective_lambda: f64,
        corrective_blend: f64,
        corrective_audit_fraction: f64,
        leaf_eb: bool,
        leaf_eb_min_trees: usize,
        leaf_eb_scale: f64,
        leaf_sibling_smooth: f64,
        hierarchical_shrinkage: f64,
        multiclass_coupled_leaves: bool,
        class_weights: Vec<f64>,
        adaptive_leaf_experts: bool,
        adaptive_cat_lookup_smooth: bool,
        cat_offset_smooth: f64,
        cat_offset_passes: usize,
        expert_leaf_admission: bool,
        expert_min_leaf: usize,
        newton_decrement_cap: f64,
        lookahead_alpha: f64,
        sign_confidence_gamma: f64,
        soft_predict_bandwidth: f64,
        soft_leaf_bandwidth: f64,
        soft_leaf_passes: usize,
        leaf_var_shrink: f64,
        jensen_train_temp: f64,
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
            vertical_init,
            vertical_init_cycles: vertical_init_cycles.max(1),
            vertical_prior: VerticalPrior::default(),
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
            eval_metric,
            verbose,
            l1_reg,
            refine_alpha,
            honest,
            honest_fraction,
            honest_arbitration,
            feature_gain_prior,
            thermal: thermal.max(0.0),
            stein_leaves,
            stein_levels,
            thermal_n_exp,
            thermal_depth_gamma,
            split_audit_fraction: 0.0,
            colsample_bylevel,
            lr_decay: 1.0,
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
            ramp: false,
            ramp_lambda: 10.0,
            ramp_k: 1,
            leaf_linear,
            leaf_quadratic,
            leaf_correction,
            grad_momentum: grad_momentum.clamp(0.0, 0.99),
            gain_penalty: gain_penalty.max(0.0),
            split_pessimism: split_pessimism.max(0.0),
            cat_prototype_bins: 0,
            cat_audit_strength: cat_audit_strength.max(0.0),
            split_contrast_penalty: split_contrast_penalty.max(0.0),
            signal_gate: signal_gate.max(0.0),
            supervised_bins,
            fold_ordered: if fold_ordered == 1 { 2 } else { fold_ordered.min(16) },
            cat_fold_evidence,
            cfe_folds: 5,
            cfe_max_pairs: cfe_max_pairs.min(64),
            cfe_max_triples: cfe_max_triples.min(64),
            cfe_max_quads: cfe_max_quads.min(64),
            cfe_dual_prior: false,
            cfe_counter: false,
            cfe_aggmax: true,
            cfe_smooth: cfe_smooth.max(1e-6),
            cfe_demote_raw,
            cfe_demote_min_card: cfe_demote_min_card.max(2),
            cfe_residual_rounds: 0,
            cfe_resid_tables: Vec::new(),
            cfe_resid_prior: 0.0,
            cfe_tuples: Vec::new(),
            cfe_tables: Vec::new(),
            cfe_prior: Vec::new(),
            cfe_n_out: 0,
            self_score_splits,
            hetero_trees,
            dart_rate: dart_rate.clamp(0.0, 0.9),
            max_delta_step: max_delta_step.max(0.0),
            main_effect_interval,
            main_effect_depth: main_effect_depth.clamp(1, 4),
            cyclic_features,
            cyclic_max_features_per_round,
            auto_interactions,
            auto_cat_interactions,
            max_interaction_features,
            lambda_schedule: lambda_schedule.max(0.0),
            use_bootstrap,
            extra_trees,
            label_smooth: label_smooth.clamp(0.0, 0.5),
            multi_output_tree,
            multiclass_pair_sequence,
            prob_avg: false,
            honest_tau: 0.0,
            complement_debias_mode: complement_debias_mode.min(3),
            phase_schedule,
            ncl_lambda: ncl_lambda.max(0.0),
            adaptive_cyclic_order,
            cyclic_partner_features,
            cyclic_partner_min_pressure_ratio: cyclic_partner_min_pressure_ratio.clamp(0.0, 10.0),
            cyclic_partner_bins: cyclic_partner_bins.clamp(2, 32),
            cyclic_feature_reuse,
            cyclic_revisit_trees,
            cyclic_revisit_min_pressure_ratio: 0.0,
            adaptive_feature_mask: false,
            adaptive_feature_mask_penalty: 0.5,
            adaptive_root_anchor,
            adaptive_root_anchor_penalty: 0.5,
            sparse_oblique_splits,
            interval_splits,
            sibling_block_correction: 0.0,
            adam_beta2: adam_beta2.clamp(0.0, 0.9999),
            adam_eps: adam_eps.max(1e-12),
            ortho_alpha: ortho_alpha.clamp(0.0, 2.0),
            split_criterion,
            rank_mix_alpha: rank_mix_alpha.clamp(0.0, 1.0),
            rank_mix_start_frac: rank_mix_start_frac.clamp(0.0, 1.0),
            rank_pair_temperature: rank_pair_temperature.clamp(0.05, 20.0),
            binary_focus_gamma: binary_focus_gamma.clamp(0.0, 4.0),
            binary_focus_end_frac: binary_focus_end_frac.clamp(0.0, 1.0),
            residual_focus_alpha: residual_focus_alpha.clamp(0.0, 4.0),
            residual_focus_max_scale: residual_focus_max_scale.max(1.0).min(8.0),
            residual_focus_mode,
            residual_focus_hessian_mode,
            residual_focus_redescend_tau: residual_focus_redescend_tau.max(0.0),
            feature_view_groups,
            leaf_trim_pct: leaf_trim_pct.clamp(0.0, 0.49),
            leaf_median,
            leaf_median_blend: leaf_median_blend.clamp(0.0, 1.0),
            leaf_mad_clip: leaf_mad_clip.max(0.0),
            leaf_adaptive_blend_kappa: leaf_adaptive_blend_kappa.max(0.0),
            ordered_boost,
            ordered_n_buckets: 4,
            goss_top_rate: goss_top_rate.clamp(0.0, 0.99),
            goss_other_rate: goss_other_rate.clamp(0.0, 0.99),
            goss_mode: "newton".to_string(),
            goss_anneal: 0.0,
            keep_all_trees,
            corrective_block_refit,
            corrective_blocks: corrective_blocks.clamp(1, 256),
            corrective_lambda: corrective_lambda.max(0.0),
            corrective_blend: corrective_blend.clamp(0.0, 1.0),
            corrective_min_trees: 16,
            corrective_audit_fraction: corrective_audit_fraction.clamp(0.0, 0.5),
            corrective_min_rel_improve: 0.0,
            leaf_eb,
            leaf_eb_min_trees: leaf_eb_min_trees.max(5),
            leaf_eb_scale: leaf_eb_scale.max(0.0),
            leaf_sibling_smooth: leaf_sibling_smooth.clamp(0.0, 0.5),
            hierarchical_shrinkage: hierarchical_shrinkage.max(0.0),
            multiclass_coupled_leaves,
            multiclass_joint_cll: false,
            class_weights,
            adaptive_leaf_experts,
            adaptive_cat_lookup_smooth,
            cat_offset_smooth: cat_offset_smooth.max(0.0),
            cat_offset_passes: cat_offset_passes.min(4),
            cat_offset_maps: Vec::new(),
            ordered_ctr: false,
            ordered_ctr_top_features: 16,
            ordered_ctr_smooth: 30.0,
            ordered_ctr_permutations: 1,
            ordered_ctr_min_count: 2,
            ordered_ctr_features: Vec::new(),
            ordered_ctr_prior: 0.0,
            ordered_ctr_priors: Vec::new(),
            ordered_ctr_maps: Vec::new(),
            ordered_ctr_count_maps: Vec::new(),
            ordered_ctr_pair_features: Vec::new(),
            ordered_ctr_pair_maps: Vec::new(),
            ordered_ctr_pair_count_maps: Vec::new(),
            ordered_ctr_triple_features: Vec::new(),
            ordered_ctr_triple_maps: Vec::new(),
            ordered_ctr_triple_count_maps: Vec::new(),
            cat_tuple_lookups: false,
            cat_tuple_max_order: 3,
            cat_tuple_top_features: 5,
            cat_tuple_hash_bins: 128,
            cat_tuple_min_leaf: 64,
            cat_tuple_gain_margin: 0.05,
            expert_leaf_admission,
            expert_max_terms: 2,
            expert_min_leaf,
            expert_min_cal: 12,
            expert_ridge_lambda: 25.0,
            expert_alpha_max: 1.0,
            expert_param_penalty: 1e-4,
            expert_se_multiplier: 0.5,
            expert_epsilon: 1e-5,
            expert_shadow_trials: 0,
            antithetic_subtrees: false,
            newton_decrement_cap: newton_decrement_cap.max(0.0),
            lookahead_alpha: lookahead_alpha.clamp(0.0, 2.0),
            sign_confidence_gamma: sign_confidence_gamma.clamp(0.0, 5.0),
            soft_predict_bandwidth: soft_predict_bandwidth.clamp(0.0, 5.0),
            // Negative bandwidth is the documented AUTO mode for
            // validation-selected soft-consistent leaf refit. Preserve it here;
            // the training path decides whether a soft model is accepted.
            soft_leaf_bandwidth: soft_leaf_bandwidth.clamp(-1.0, 5.0),
            soft_leaf_passes: soft_leaf_passes.min(6),
            leaf_var_shrink: leaf_var_shrink.clamp(0.0, 5.0),
            jensen_train_temp: jensen_train_temp.clamp(0.5, 5.0),
            diversity_penalty: 0.0,
            diversity_decay: 0.9,
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

    #[pyo3(signature = (x, y, n_rounds, eval_x = None, eval_y = None, init_score = None, eval_init_score = None, sample_weight = None))]
    pub fn fit(
        &mut self,
        py: Python<'_>,
        x: Bound<'_, PyAny>,
        y: Bound<'_, PyAny>,
        n_rounds: usize,
        eval_x: Option<Bound<'_, PyAny>>,
        eval_y: Option<Bound<'_, PyAny>>,
        init_score: Option<Bound<'_, PyAny>>,
        eval_init_score: Option<Bound<'_, PyAny>>,
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
        // boosting begins. Binary/regression use one margin per row; multiclass
        // uses a flat row-major matrix with n_rows * n_classes margins.
        let init_score_data: Option<Vec<f64>> = match init_score {
            Some(s) => {
                let arr: Bound<'_, PyArray1<f64>> = s.extract().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "init_score must be a 1-D float64 numpy array",
                    )
                })?;
                let v: Vec<f64> = arr.to_owned_array().into_raw_vec_and_offset().0;
                let expected_len = if self.task == "multiclass" {
                    let n_classes = y_data.iter().map(|&vv| vv as usize).max().unwrap_or(0) + 1;
                    n_rows.saturating_mul(n_classes)
                } else {
                    n_rows
                };
                if v.len() != expected_len {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "init_score length ({}) must equal expected length ({})",
                        v.len(),
                        expected_len
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

        let eval_init_score_data: Option<Vec<f64>> = match eval_init_score {
            Some(s) => {
                let Some((_, _, en_rows)) = eval_data_raw.as_ref() else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "eval_init_score requires eval_x/eval_y",
                    ));
                };
                let arr: Bound<'_, PyArray1<f64>> = s.extract().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "eval_init_score must be a 1-D float64 numpy array",
                    )
                })?;
                let v: Vec<f64> = arr.to_owned_array().into_raw_vec_and_offset().0;
                let expected_len = if self.task == "multiclass" {
                    let n_classes = y_data.iter().map(|&vv| vv as usize).max().unwrap_or(0) + 1;
                    en_rows.saturating_mul(n_classes)
                } else {
                    *en_rows
                };
                if v.len() != expected_len {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "eval_init_score length ({}) must equal expected length ({})",
                        v.len(),
                        expected_len
                    )));
                }
                Some(v)
            }
            None => None,
        };

        // Adaptive thread pool: parallelism is profitable only when the work
        // per task exceeds dispatch overhead. With Rayon's ~20µs/thread
        // overhead and ~ns-scale tree ops, we want roughly one thread per
        // 50k cells (n_rows × n_features). Small data collapses to 1
        // thread (no dispatch waste); large data uses all cores.
        let max_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let subtrees_per_round = self.subtrees_per_boosting_round(n_features);
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
                .saturating_mul(subtrees_per_round)
                .saturating_mul(n_classes.max(1))
        } else {
            n_rounds.max(1).saturating_mul(subtrees_per_round)
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
                // DP bins: bin numerics on a finer quantile grid first, then
                // DP-merge down to effective_bins where the initial gradient
                // profile changes (see BinnedData::supervised_merge).
                let build_bins = if self.supervised_bins {
                    effective_bins
                        .saturating_mul(4)
                        .min(512)
                        .min(32.max(n_rows / 2))
                        .max(effective_bins)
                } else {
                    effective_bins
                };
                let mut binned = BinnedData::new(
                    &x_data,
                    n_rows,
                    n_features,
                    build_bins,
                    &self.cat_features,
                    self.max_cat_bins,
                );
                if self.supervised_bins && build_bins > effective_bins {
                    let n_outputs = if self.task == "multiclass" {
                        (y_data
                            .iter()
                            .map(|&v| v as usize)
                            .max()
                            .unwrap_or(0)
                            + 1)
                            .max(2)
                    } else {
                        1
                    };
                    let mut scores = vec![0.0f64; n_rows * n_outputs];
                    if n_outputs == 1 {
                        let mean = y_data.iter().sum::<f64>() / n_rows.max(1) as f64;
                        for (row, &yv) in y_data.iter().enumerate().take(n_rows) {
                            scores[row] = yv - mean;
                        }
                    } else {
                        let mut prior = vec![0.0f64; n_outputs];
                        for &yv in y_data.iter().take(n_rows) {
                            let k = (yv as usize).min(n_outputs - 1);
                            prior[k] += 1.0;
                        }
                        for p in prior.iter_mut() {
                            *p /= n_rows.max(1) as f64;
                        }
                        for (row, &yv) in y_data.iter().enumerate().take(n_rows) {
                            let k = (yv as usize).min(n_outputs - 1);
                            for j in 0..n_outputs {
                                scores[row * n_outputs + j] =
                                    (if j == k { 1.0 } else { 0.0 }) - prior[j];
                            }
                        }
                    }
                    binned.supervised_merge(&scores, n_outputs, effective_bins);
                }
                binned.split_pessimism = self.split_pessimism;
                binned.cat_prototype_bins = self.cat_prototype_bins;
                binned.cat_audit_strength = self.cat_audit_strength;
                binned.split_contrast_penalty = self.split_contrast_penalty;
                binned.signal_gate = self.signal_gate;

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
                        scored_pairs.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1)
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .then_with(|| a.0.cmp(&b.0))
                        });

                        if self.auto_interactions {
                            let stable_pairs = self.stable_numeric_interaction_pairs(
                                &x_data,
                                &y_data,
                                n_rows,
                                n_features,
                                &numeric_indices,
                                max_pairs,
                            );
                            let stable_budget = (max_pairs / 2).max(1).min(max_pairs);
                            let mut selected_pairs: Vec<(usize, usize)> = Vec::new();
                            for (pair, _score) in stable_pairs.into_iter().take(stable_budget) {
                                if !selected_pairs.contains(&pair) {
                                    selected_pairs.push(pair);
                                }
                            }
                            for (pair, _score) in scored_pairs.iter() {
                                if selected_pairs.len() >= max_pairs {
                                    break;
                                }
                                if !selected_pairs.contains(pair) {
                                    selected_pairs.push(*pair);
                                }
                            }
                            if !selected_pairs.is_empty() {
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
                                self.numeric_interaction_edges =
                                    binned.bin_edges[int_start..].to_vec();
                            }
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
                            scored_cat_pairs.sort_by(|a, b| {
                                b.1.partial_cmp(&a.1)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                                    .then_with(|| a.0.cmp(&b.0))
                            });
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

                // CFE: categorical fold evidence (cross-fit tuple posteriors).
                let cfe_edges: Vec<Vec<f64>> = {
                    let cfe_cols =
                        self.build_cat_fold_evidence(&x_data, &y_data, n_rows, n_features);
                    if cfe_cols.is_empty() {
                        Vec::new()
                    } else {
                        let cfe_start = binned.n_features;
                        binned.add_ots_features(&cfe_cols, effective_bins);
                        binned.bin_edges[cfe_start..].to_vec()
                    }
                };
                // With CFE active, flatten raw HIGH-CARD categorical columns:
                // their native subset splits are exactly the memorization the
                // cross-fit evidence replaces, and removing them is where most
                // of the accuracy win comes from (verified on Amazon-access).
                if self.cfe_demote_raw && !self.cfe_tuples.is_empty() {
                    for f in 0..n_features {
                        // Demote by CARDINALITY, not bin count: >256-card cats
                        // fall back to numeric binning (n_bins <= effective_bins
                        // hides their cardinality) but are flagged via
                        // cll_is_categorical && !is_categorical.
                        let hash_fallback_cat = f < binned.cll_is_categorical.len()
                            && binned.cll_is_categorical[f]
                            && f < binned.is_categorical.len()
                            && !binned.is_categorical[f];
                        if f < self.cat_features.len()
                            && self.cat_features[f]
                            && (binned.n_bins(f) > self.cfe_demote_min_card || hash_fallback_cat)
                        {
                            for row in 0..n_rows {
                                let idx = f * binned.n_rows + row;
                                if binned.bin_indices[idx] != crate::tree::MISSING_BIN {
                                    binned.bin_indices[idx] = 0;
                                }
                            }
                            if f + 1 < binned.non_missing_offsets.len() {
                                let (lo, hi) = (
                                    binned.non_missing_offsets[f],
                                    binned.non_missing_offsets[f + 1],
                                );
                                for v in &mut binned.non_missing_bin_values[lo..hi] {
                                    if *v != crate::tree::MISSING_BIN {
                                        *v = 0;
                                    }
                                }
                            }
                            // Also disable CLL lookups on the demoted column:
                            // cll_hash_bins keep full cardinality, so leaving
                            // cll_is_categorical set would let cat_lookup_smooth
                            // / adaptive_leaf_experts memorize the column the
                            // demotion just removed (and with train/predict CLL
                            // bin spaces disagreeing after the edge flatten).
                            if f < binned.cll_is_categorical.len() {
                                binned.cll_is_categorical[f] = false;
                            }
                            binned.bin_edges[f] = vec![0.0];
                        }
                    }
                }
                // CFE stage 2: residual evidence (internal warmup -> tables
                // over the same tuples, targets = residual gradients).
                let cfe_resid_edges: Vec<Vec<f64>> = {
                    let cols = self.build_cfe_residual_evidence(
                        &binned, &x_data, &y_data, n_rows, n_features,
                    );
                    if cols.is_empty() {
                        Vec::new()
                    } else {
                        let start = binned.n_features;
                        binned.add_ots_features(&cols, effective_bins);
                        binned.bin_edges[start..].to_vec()
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
                    if !cfe_edges.is_empty() {
                        let eval_cfe =
                            self.cat_fold_evidence_columns_for_raw(&ex_data, en_rows, n_features);
                        BinnedData::add_ots_features_with_edges(
                            &mut eval_bins,
                            en_rows,
                            &eval_cfe,
                            &cfe_edges,
                        );
                    }
                    if !cfe_resid_edges.is_empty() {
                        let eval_resid =
                            self.cfe_residual_columns_for_raw(&ex_data, en_rows, n_features);
                        BinnedData::add_ots_features_with_edges(
                            &mut eval_bins,
                            en_rows,
                            &eval_resid,
                            &cfe_resid_edges,
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
                    let eval_raw = if self.auto_interactions
                        || self.jit_ltso_enabled
                        || (self.vertical_init
                            && self.task == "regression"
                            && init_score_data.is_none())
                    {
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
                    // LTSO is a regression/binary learner-family. It is admitted
                    // by honest residual-transfer below, so low-dimensional
                    // numeric tasks can use hinge/diff/ratio operators while
                    // mixed categorical tasks can still use the N|C family.
                    if !matches!(self.task.as_str(), "regression" | "binary")
                        || num_indices.is_empty()
                    {
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
                    self.vertical_prior = VerticalPrior::default();

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

                    let native_init_score = if init_score_data.is_none() {
                        self.fit_vertical_prior_for_regression(
                            &x_data,
                            &y_data,
                            n_rows,
                            n_features_original,
                            sample_weight_data.as_deref(),
                        )
                    } else {
                        self.vertical_prior = VerticalPrior::default();
                        init_score_data.clone()
                    };

                    let mut eval_data = eval_data;
                    if self.task == "multiclass" {
                        self.fit_multiclass(
                            &mut binned,
                            &y_data,
                            n_rows,
                            n_features,
                            n_rounds,
                            &mut eval_data,
                            init_score_data.as_deref(),
                            eval_init_score_data.as_deref(),
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
                            native_init_score.as_deref(),
                            eval_init_score_data.as_deref(),
                            sample_weight_data.as_deref(),
                        );
                        self.apply_corrective_block_refit(
                            &binned,
                            &x_data,
                            n_rows,
                            n_features_original,
                            &y_data,
                            native_init_score.as_deref(),
                            eval_data
                                .as_ref()
                                .map(|(_, ey, en, ex, _)| (ex.as_slice(), ey.as_slice(), *en)),
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
        binned.cat_prototype_bins = self.cat_prototype_bins;
        binned.cat_audit_strength = self.cat_audit_strength;
        binned.split_contrast_penalty = self.split_contrast_penalty;
        binned.signal_gate = self.signal_gate;
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
                let expected_len = if self.task == "multiclass" {
                    let n_classes = y_data.iter().map(|&vv| vv as usize).max().unwrap_or(0) + 1;
                    n_rows.saturating_mul(n_classes)
                } else {
                    n_rows
                };
                if v.len() != expected_len {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "init_score length ({}) must equal expected length ({})",
                        v.len(),
                        expected_len
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
        let subtrees_per_round = self.subtrees_per_boosting_round(binned.n_features);
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
                .saturating_mul(subtrees_per_round)
                .saturating_mul(n_classes.max(1))
        } else {
            n_rounds.max(1).saturating_mul(subtrees_per_round)
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
                let native_init_score = if init_score_data.is_none() {
                    self.fit_vertical_prior_for_regression(
                        &x_data,
                        &y_data,
                        n_rows,
                        n_features_original,
                        sample_weight_data.as_deref(),
                    )
                } else {
                    self.vertical_prior = VerticalPrior::default();
                    init_score_data.clone()
                };

                if self.task == "multiclass" {
                    self.fit_multiclass(
                        &mut binned,
                        &y_data,
                        n_rows,
                        n_features,
                        n_rounds,
                        &mut eval_data,
                        init_score_data.as_deref(),
                        None,
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
                        native_init_score.as_deref(),
                        None,
                        sample_weight_data.as_deref(),
                    );
                    self.apply_corrective_block_refit(
                        &binned,
                        &x_data,
                        n_rows,
                        n_features_original,
                        &y_data,
                        native_init_score.as_deref(),
                        eval_data
                            .as_ref()
                            .map(|(_, ey, en, ex, _)| (ex.as_slice(), ey.as_slice(), *en)),
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
        // Binary/regression expect one value per row; multiclass expects a flat
        // row-major n_rows * n_classes margin matrix.
        let init_score_vec: Option<Vec<f64>> = match init_score {
            Some(s) => {
                let arr: Bound<'_, PyArray1<f64>> = s.extract().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "predict: init_score must be a 1-D float64 numpy array",
                    )
                })?;
                let v: Vec<f64> = arr.to_owned_array().into_raw_vec_and_offset().0;
                let expected_len = if self.task == "multiclass" {
                    n_rows.saturating_mul(self.n_classes.max(1))
                } else {
                    n_rows
                };
                if v.len() != expected_len {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "predict: init_score length ({}) must equal expected length ({})",
                        v.len(),
                        expected_len,
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
            let binned_plain_with_ramp_trees: Vec<bool> = if use_srp {
                Vec::new()
            } else {
                trees
                    .iter()
                    .map(|t| t.can_predict_binned_plain_with_ramp())
                    .collect()
            };
            let use_eval_bins = !use_srp && n_features == binned.n_features;
            // Row-major path is OK as long as every tree is plain *or*
            // plain+ramp — the row-major-with-ramp variant handles the second
            // case using the same cache-friendly bin layout.
            let use_row_major_eval_bins =
                use_eval_bins && binned_plain_with_ramp_trees.iter().all(|&is_ok| is_ok);
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
                let init_score_ref = init_score_vec.as_deref();
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
                                    if binned_plain_trees.get(t_idx).copied().unwrap_or(false) {
                                        tree.predict_binned_plain_row_major(
                                            row_bins, n_features, row,
                                        )
                                    } else {
                                        tree.predict_binned_plain_row_major_with_ramp(
                                            row_bins, n_features, row,
                                        )
                                    }
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
                            let mut scores = if let Some(init) = init_score_ref {
                                let base = row * n_classes;
                                init[base..base + n_classes].to_vec()
                            } else if self.class_base_scores.len() == n_classes {
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
                                    if binned_plain_trees.get(t_idx).copied().unwrap_or(false) {
                                        tree.predict_binned_plain_row_major(
                                            row_bins, n_features, row,
                                        )
                                    } else {
                                        tree.predict_binned_plain_row_major_with_ramp(
                                            row_bins, n_features, row,
                                        )
                                    }
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
                let vertical_prior = &self.vertical_prior;
                let use_vertical_prior = vertical_prior.is_active() && init_score_ref.is_none();
                let is_poisson = self.task == "poisson";
                let cat_offset_maps = &self.cat_offset_maps;
                let has_cat_offsets = self.task == "binary" && !cat_offset_maps.is_empty();
                (0..n_rows)
                    .into_par_iter()
                    .map(|row| {
                        let row_data = &x_data[row * n_features..(row + 1) * n_features];
                        let raw_row = &x_data_raw[row * n_features_raw..(row + 1) * n_features_raw];
                        // Per-row offset: init_score if user supplied one, else
                        // fitted vertical prior if present, else the global base_score.
                        let mut sum = match init_score_ref {
                            Some(s) => s[row],
                            None if use_vertical_prior => vertical_prior.predict_row(raw_row),
                            None => base,
                        };
                        for (t_idx, tree) in trees.iter().enumerate() {
                            let w = if has_dart_w && t_idx < dart_w.len() {
                                dart_w[t_idx]
                            } else {
                                1.0
                            };
                            let c = if let Some(ref row_bins) = eval_bins_row_major {
                                if binned_plain_trees.get(t_idx).copied().unwrap_or(false) {
                                    tree.predict_binned_plain_row_major(row_bins, n_features, row)
                                } else {
                                    tree.predict_binned_plain_row_major_with_ramp(
                                        row_bins, n_features, row,
                                    )
                                }
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
                let vertical_prior = &self.vertical_prior;
                let use_vertical_prior = vertical_prior.is_active();
                (0..n_rows)
                    .into_par_iter()
                    .map(|row| {
                        let row_data = &x_data[row * n_features..(row + 1) * n_features];
                        let raw_row = &x_data_raw[row * n_features_raw..(row + 1) * n_features_raw];
                        let mut sum = if use_vertical_prior {
                            vertical_prior.predict_row(raw_row)
                        } else {
                            base
                        };
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
                let vertical_prior = &self.vertical_prior;
                let use_vertical_prior = vertical_prior.is_active();
                (0..n_rows)
                    .into_par_iter()
                    .map(|row| {
                        let row_data = &x_data[row * n_features..(row + 1) * n_features];
                        let raw_row = &x_data_raw[row * n_features_raw..(row + 1) * n_features_raw];
                        let mut sum = if use_vertical_prior {
                            vertical_prior.predict_row(raw_row)
                        } else {
                            base
                        };
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

    /// Return per-tree weights. Empty means every tree has weight 1.0.
    pub fn tree_weights(&self) -> Vec<f64> {
        self.dart_tree_weights.clone()
    }

    /// Set per-tree weights used by predict() and predict_truncated().
    ///
    /// This is intentionally post-fit only: Python-side validation routines can
    /// compile a guarded path ensemble back into the native single-pass model
    /// without changing the tree structure.
    pub fn set_tree_weights(&mut self, weights: Vec<f64>) -> PyResult<()> {
        if weights.is_empty() {
            self.dart_tree_weights.clear();
            return Ok(());
        }
        if weights.len() != self.trees.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "tree weight length ({}) must equal number of trees ({})",
                weights.len(),
                self.trees.len()
            )));
        }
        if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "tree weights must be finite and non-negative",
            ));
        }
        self.dart_tree_weights = weights;
        Ok(())
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
    /// Refine 3.1 deployment hook: add per-leaf corrections IN PLACE, so the
    /// refined model is a single native artifact (native predict, nothing
    /// Python-side at inference). Entries are (tree_idx, node_idx, delta)
    /// triplets; node_idx must be a leaf. Works for regression, binary, and
    /// multiclass (per-class trees are ordinary single-output trees laid out
    /// round-major, class-minor — tree t serves class t % n_classes).
    pub fn add_leaf_corrections(
        &mut self,
        tree_idx: Vec<u32>,
        node_idx: Vec<u32>,
        delta: Vec<f64>,
    ) -> PyResult<()> {
        if tree_idx.len() != node_idx.len() || node_idx.len() != delta.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "tree_idx, node_idx and delta must have equal lengths",
            ));
        }
        let n_trees = self.trees.len();
        for ((&t, &nd), &d) in tree_idx.iter().zip(node_idx.iter()).zip(delta.iter()) {
            let t = t as usize;
            let nd = nd as usize;
            let tree = self.trees.get_mut(t).ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "tree_idx {} out of range ({} trees)",
                    t, n_trees
                ))
            })?;
            if nd >= tree.values.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "node_idx {} out of range for tree {}",
                    nd, t
                )));
            }
            if tree.split_features.get(nd).copied().unwrap_or(u32::MAX) != u32::MAX {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "node {} of tree {} is not a leaf",
                    nd, t
                )));
            }
            tree.values[nd] += d;
        }
        Ok(())
    }

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
    #[pyo3(signature = (x, n_trees, init_score = None))]
    pub fn predict_truncated(
        &self,
        py: Python<'_>,
        x: Bound<'_, PyAny>,
        n_trees: usize,
        init_score: Option<Bound<'_, PyAny>>,
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

        let init_score_vec: Option<Vec<f64>> = match init_score {
            Some(s) => {
                let arr: Bound<'_, PyArray1<f64>> = s.extract().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "predict_truncated: init_score must be a 1-D float64 numpy array",
                    )
                })?;
                let v: Vec<f64> = arr.to_owned_array().into_raw_vec_and_offset().0;
                let expected_len = if self.task == "multiclass" {
                    n_rows.saturating_mul(self.n_classes.max(1))
                } else {
                    n_rows
                };
                if v.len() != expected_len {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "predict_truncated: init_score length ({}) must equal expected length ({})",
                        v.len(),
                        expected_len,
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
            let binned_plain_trees: Vec<bool> =
                trees.iter().map(|t| t.can_predict_binned_plain()).collect();
            let binned_plain_with_ramp_trees: Vec<bool> = trees
                .iter()
                .map(|t| t.can_predict_binned_plain_with_ramp())
                .collect();
            let use_eval_bins = n_features == binned.n_features;
            let use_row_major_eval_bins =
                use_eval_bins && binned_plain_with_ramp_trees.iter().all(|&is_ok| is_ok);
            let eval_bins_row_major = if use_row_major_eval_bins {
                Some(BinnedData::bin_with_edges_row_major(
                    &x_data,
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
                    &x_data,
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
                    &x_data,
                    n_rows,
                    n_features,
                    &self.cat_features,
                    &binned.is_categorical,
                    &binned.bin_edges,
                )
            } else {
                Vec::new()
            };

            if self.task == "multiclass" {
                let n_classes = self.n_classes;
                let ntp = self.multiclass_trees_per_class_round();
                let use_prob_avg = self.prob_avg && ntp > 1;
                let init_score_ref = init_score_vec.as_deref();
                let preds_2d: Vec<Vec<f64>> = (0..n_rows)
                    .into_par_iter()
                    .map(|row| {
                        let row_data = &x_data[row * n_features..(row + 1) * n_features];
                        if use_prob_avg {
                            let mut avg_probs = vec![0.0f64; n_classes];
                            for (t_idx, tree) in trees.iter().enumerate() {
                                let k = (t_idx / ntp) % n_classes;
                                let c = if let Some(ref row_bins) = eval_bins_row_major {
                                    if binned_plain_trees.get(t_idx).copied().unwrap_or(false) {
                                        tree.predict_binned_plain_row_major(
                                            row_bins, n_features, row,
                                        )
                                    } else {
                                        tree.predict_binned_plain_row_major_with_ramp(
                                            row_bins, n_features, row,
                                        )
                                    }
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
                            let mut scores = if let Some(init) = init_score_ref {
                                let base = row * n_classes;
                                init[base..base + n_classes].to_vec()
                            } else if self.class_base_scores.len() == n_classes {
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
                                    if binned_plain_trees.get(t_idx).copied().unwrap_or(false) {
                                        tree.predict_binned_plain_row_major(
                                            row_bins, n_features, row,
                                        )
                                    } else {
                                        tree.predict_binned_plain_row_major_with_ramp(
                                            row_bins, n_features, row,
                                        )
                                    }
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
                let vertical_prior = &self.vertical_prior;
                let use_vertical_prior = vertical_prior.is_active() && init_score_ref.is_none();
                let is_poisson = self.task == "poisson";
                (0..n_rows)
                    .into_par_iter()
                    .map(|row| {
                        let row_data = &x_data[row * n_features..(row + 1) * n_features];
                        let raw_row = &x_data_raw[row * n_features_raw..(row + 1) * n_features_raw];
                        let mut sum = match init_score_ref {
                            Some(s) => s[row],
                            None if use_vertical_prior => vertical_prior.predict_row(raw_row),
                            None => base,
                        };
                        for (t_idx, tree) in trees.iter().enumerate() {
                            let w = if has_dart_w && t_idx < dart_w.len() {
                                dart_w[t_idx]
                            } else {
                                1.0
                            };
                            let c = if let Some(ref row_bins) = eval_bins_row_major {
                                if binned_plain_trees.get(t_idx).copied().unwrap_or(false) {
                                    tree.predict_binned_plain_row_major(row_bins, n_features, row)
                                } else {
                                    tree.predict_binned_plain_row_major_with_ramp(
                                        row_bins, n_features, row,
                                    )
                                }
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

    /// Cheap tree count (avoids marshaling full tree_info per predict call).
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// Per-feature split usage counts across all trees (consensus-guided boosting:
    /// fold-model usage maps distill which structural choices replicate across
    /// resampled worlds; the real fit consumes them as feature_gain_prior).
    pub fn feature_split_counts(&self) -> Vec<f64> {
        let mut counts = vec![0.0f64; self.n_features.max(1)];
        for tree in &self.trees {
            for node in 0..tree.split_features.len() {
                let f = tree.split_features[node];
                if f == u32::MAX {
                    continue;
                }
                if tree.is_oblique_split.get(node).copied().unwrap_or(false) {
                    for j in 0..2 {
                        let of = tree.oblique_features.get(node * 2 + j).copied().unwrap_or(u32::MAX);
                        if of != u32::MAX && (of as usize) < counts.len() {
                            counts[of as usize] += 1.0;
                        }
                    }
                } else if (f as usize) < counts.len() {
                    counts[f as usize] += 1.0;
                }
            }
        }
        counts
    }

    /// All-checkpoints prediction in ONE pass: bins the matrix once and walks
    /// each row's trees once, snapshotting the margin at every checkpoint.
    /// Replaces N separate `predict_truncated` calls (each of which re-extends,
    /// re-bins and re-traverses) on the APX predict path. Non-multiclass only.
    /// Returns a flat vector of length n_checkpoints * n_rows, checkpoint-major.
    #[pyo3(signature = (x, checkpoints, init_score = None))]
    pub fn predict_checkpoints(
        &self,
        py: Python<'_>,
        x: Bound<'_, PyAny>,
        checkpoints: Vec<usize>,
        init_score: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Vec<f64>> {
        if self.task == "multiclass" {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "predict_checkpoints: multiclass not supported (use predict_truncated)",
            ));
        }
        if checkpoints.is_empty() {
            return Ok(Vec::new());
        }
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
        let init_score_vec: Option<Vec<f64>> = match init_score {
            Some(s) => {
                let arr: Bound<'_, PyArray1<f64>> = s.extract().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "predict_checkpoints: init_score must be a 1-D float64 numpy array",
                    )
                })?;
                Some(arr.to_owned_array().into_raw_vec_and_offset().0)
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
        let (x_data, n_features) = self.extend_raw_matrix(&x_data_raw, n_rows, n_features_raw);
        let n_trees_total = self.trees.len();
        let cps: Vec<usize> = checkpoints
            .iter()
            .map(|&c| c.min(n_trees_total))
            .collect();
        let n_use = cps.iter().copied().max().unwrap_or(0);
        let n_cps = cps.len();

        let result = py.detach(|| {
            let lr = self.learning_rate;
            let trees = &self.trees[..n_use];
            let dart_w = &self.dart_tree_weights;
            let has_dart_w = !dart_w.is_empty();
            let binned_plain_trees: Vec<bool> =
                trees.iter().map(|t| t.can_predict_binned_plain()).collect();
            let binned_plain_with_ramp_trees: Vec<bool> = trees
                .iter()
                .map(|t| t.can_predict_binned_plain_with_ramp())
                .collect();
            let use_eval_bins = n_features == binned.n_features;
            let use_row_major_eval_bins =
                use_eval_bins && binned_plain_with_ramp_trees.iter().all(|&is_ok| is_ok);
            let eval_bins_row_major = if use_row_major_eval_bins {
                Some(BinnedData::bin_with_edges_row_major(
                    &x_data,
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
                    &x_data,
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
                    &x_data,
                    n_rows,
                    n_features,
                    &self.cat_features,
                    &binned.is_categorical,
                    &binned.bin_edges,
                )
            } else {
                Vec::new()
            };

            let base = self.base_score;
            let init_score_ref = init_score_vec.as_deref();
            let vertical_prior = &self.vertical_prior;
            let use_vertical_prior = vertical_prior.is_active() && init_score_ref.is_none();
            let is_poisson = self.task == "poisson";
            let per_row: Vec<Vec<f64>> = (0..n_rows)
                .into_par_iter()
                .map(|row| {
                    let row_data = &x_data[row * n_features..(row + 1) * n_features];
                    let raw_row =
                        &x_data_raw[row * n_features_raw..(row + 1) * n_features_raw];
                    let mut sum = match init_score_ref {
                        Some(s) => s[row],
                        None if use_vertical_prior => vertical_prior.predict_row(raw_row),
                        None => base,
                    };
                    let mut snaps = vec![0.0f64; n_cps];
                    let snap = |s: f64| if is_poisson { s.exp() } else { s };
                    for (ci, &cp) in cps.iter().enumerate() {
                        if cp == 0 {
                            snaps[ci] = snap(sum);
                        }
                    }
                    for (t_idx, tree) in trees.iter().enumerate() {
                        let w = if has_dart_w && t_idx < dart_w.len() {
                            dart_w[t_idx]
                        } else {
                            1.0
                        };
                        let c = if let Some(ref row_bins) = eval_bins_row_major {
                            if binned_plain_trees.get(t_idx).copied().unwrap_or(false) {
                                tree.predict_binned_plain_row_major(row_bins, n_features, row)
                            } else {
                                tree.predict_binned_plain_row_major_with_ramp(
                                    row_bins, n_features, row,
                                )
                            }
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
                        } else {
                            tree.predict_raw_row(binned, row_data)
                        };
                        sum += lr * w * c;
                        for (ci, &cp) in cps.iter().enumerate() {
                            if cp == t_idx + 1 {
                                snaps[ci] = snap(sum);
                            }
                        }
                    }
                    snaps
                })
                .collect();
            let mut out = vec![0.0f64; n_cps * n_rows];
            for (row, snaps) in per_row.into_iter().enumerate() {
                for (ci, v) in snaps.into_iter().enumerate() {
                    out[ci * n_rows + row] = v;
                }
            }
            out
        });
        Ok(result)
    }
}


fn serde_true() -> bool {
    true
}
