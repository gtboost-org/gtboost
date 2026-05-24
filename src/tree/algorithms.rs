//! Tree-construction internals: helpers used by `impl DecisionTree::build_*`.
//!
//! Organized by role:
//!
//! - **`TreeBuilder`**: a small mutable struct that the build_* methods push
//!   nodes/leaves into; converted to a frozen `DecisionTree` via `into_tree`.
//! - **Split eval primitives**: `SplitResult`, `SplitCandidate`, leaf-expert
//!   candidate types, plus `eval_fixed_split_gain`, `combine_complement_gain`,
//!   `oblique_proj_binned_split`, `split_goes_left_binned`.
//! - **Per-node experts**: `eval_cll_for_node`, `eval_nll_for_node`,
//!   `eval_pair_cll_for_node`, `eval_best_linear_for_node`,
//!   `eval_best_bilinear_for_node`, `eval_guided_cll_for_node`,
//!   `eval_best_lookup_for_node`, plus `make_cll_lookup`/`make_nll_lookup`
//!   builders and `expert_score`/`solve_linear_3x3` numerics helpers.
//! - **Histograms**: `NodeHists`, `build_feature_hist`, `build_node_hists`,
//!   `subtract_node_hists`.
//! - **Split scanning**: `scan_feature_hist`, `eval_feature_split` (binary)
//!   and `eval_feature_split_multi` (K-class), plus `split_noise` /
//!   `normalized_bin_coord` helpers.
//! - **Multiclass categorical sorting**: `multiclass_cat_sort_direction*`,
//!   `multiclass_cat_contrast_vectors`, `sort_multiclass_cat_bins_by_contrast`,
//!   `dense_multiclass_gain`, `solve_spd_local`.
//! - **Oblique splits**: `attended_features*`, `attended_numeric_features*`,
//!   `eval_sparse_oblique_candidate*`, `find_sparse_oblique_split*`.
//! - **Extra-Trees variants**: `find_extra_trees_split*`.
//! - **Best-split orchestrators** (top-level): `find_best_split`,
//!   `find_best_split_multi`, `find_best_split_from_hists`,
//!   `find_best_split_from_hists_debiased`, `find_best_split_debiased`,
//!   `partition_indices`, `partition_indices_split`.
//!
//! All items are `pub(super)`-visible to `tree/mod.rs` (`impl DecisionTree`).

use rayon::prelude::*;
use std::cmp::Ordering;
use std::sync::OnceLock;

use super::{
    bitmask_set, bitmask_test, BinnedData, CatBitmask, CatLookup, CatTupleConfig, DecisionTree,
    GuidedCatChoice, MISSING_BIN,
};

pub(super) struct TreeBuilder {
    split_features: Vec<u32>,
    split_bins: Vec<u16>,
    values: Vec<f64>,
    left_children: Vec<u32>,
    right_children: Vec<u32>,
    missing_goes_left: Vec<bool>,
    is_oblique_split: Vec<bool>,
    is_cat_split: Vec<bool>,
    cat_left_masks: Vec<CatBitmask>,
    oblique_features: Vec<u32>,
    oblique_weights: Vec<f32>,
    oblique_thresholds: Vec<f32>,
    cat_lookups: Vec<Option<CatLookup>>,
    node_h_sum: Vec<f64>,
    node_count: Vec<u32>,
    // GGFP v5.0 — JIT-CatPairSplit per-node storage
    cat_pair_feat2: Vec<u32>,
    cat_pair_bucket_map_a: Vec<Vec<u8>>,
    cat_pair_bucket_map_b: Vec<Vec<u8>>,
    cat_pair_cell_mask: Vec<u64>,
    cat_pair_k_buckets: Vec<u8>,
}

#[inline]
fn sparse_blocks_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("GTBOOST_SPARSE_BLOCKS")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(false)
    })
}

impl TreeBuilder {
    pub(super) fn new(cap: usize) -> Self {
        TreeBuilder {
            split_features: Vec::with_capacity(cap),
            split_bins: Vec::with_capacity(cap),
            values: Vec::with_capacity(cap),
            left_children: Vec::with_capacity(cap),
            right_children: Vec::with_capacity(cap),
            missing_goes_left: Vec::with_capacity(cap),
            is_oblique_split: Vec::with_capacity(cap),
            is_cat_split: Vec::with_capacity(cap),
            cat_left_masks: Vec::with_capacity(cap),
            oblique_features: Vec::with_capacity(cap * 2),
            oblique_weights: Vec::with_capacity(cap * 2),
            oblique_thresholds: Vec::with_capacity(cap),
            cat_lookups: Vec::with_capacity(cap),
            node_h_sum: Vec::with_capacity(cap),
            node_count: Vec::with_capacity(cap),
            cat_pair_feat2: Vec::with_capacity(cap),
            cat_pair_bucket_map_a: Vec::with_capacity(cap),
            cat_pair_bucket_map_b: Vec::with_capacity(cap),
            cat_pair_cell_mask: Vec::with_capacity(cap),
            cat_pair_k_buckets: Vec::with_capacity(cap),
        }
    }

    pub(super) fn add_node(&mut self) -> usize {
        let idx = self.split_features.len();
        self.split_features.push(u32::MAX);
        self.split_bins.push(0);
        self.values.push(0.0);
        self.left_children.push(0);
        self.right_children.push(0);
        self.missing_goes_left.push(true);
        self.is_oblique_split.push(false);
        self.is_cat_split.push(false);
        self.cat_left_masks.push(Vec::new());
        self.oblique_features.push(u32::MAX);
        self.oblique_features.push(u32::MAX);
        self.oblique_weights.push(0.0);
        self.oblique_weights.push(0.0);
        self.oblique_thresholds.push(0.0);
        self.cat_lookups.push(None);
        self.node_h_sum.push(0.0);
        self.node_count.push(0);
        self.cat_pair_feat2.push(u32::MAX);
        self.cat_pair_bucket_map_a.push(Vec::new());
        self.cat_pair_bucket_map_b.push(Vec::new());
        self.cat_pair_cell_mask.push(0);
        self.cat_pair_k_buckets.push(0);
        idx
    }

    pub(super) fn set_leaf(&mut self, idx: usize, value: f64) {
        self.split_features[idx] = u32::MAX;
        self.values[idx] = value;
    }

    pub(super) fn set_node_stats(&mut self, idx: usize, h_sum: f64, count: u32) {
        self.node_h_sum[idx] = h_sum;
        self.node_count[idx] = count;
    }

    pub(super) fn set_cll(&mut self, idx: usize, lookup: CatLookup) {
        self.cat_lookups[idx] = Some(lookup);
    }

    pub(super) fn add_split(
        &mut self,
        node_idx: usize,
        feat: u32,
        bin: u16,
        value: f64,
        missing_left: bool,
        is_oblique: bool,
        oblique_feats: [u32; 2],
        oblique_weights: [f32; 2],
        oblique_threshold: f32,
        is_cat: bool,
        cat_mask: CatBitmask,
    ) -> (usize, usize) {
        let left_idx = self.add_node();
        let right_idx = self.add_node();
        self.split_features[node_idx] = feat;
        self.split_bins[node_idx] = bin;
        self.values[node_idx] = value;
        self.left_children[node_idx] = left_idx as u32;
        self.right_children[node_idx] = right_idx as u32;
        self.missing_goes_left[node_idx] = missing_left;
        self.is_oblique_split[node_idx] = is_oblique;
        self.is_cat_split[node_idx] = is_cat;
        self.cat_left_masks[node_idx] = cat_mask;
        let ob = node_idx * 2;
        self.oblique_features[ob] = oblique_feats[0];
        self.oblique_features[ob + 1] = oblique_feats[1];
        self.oblique_weights[ob] = oblique_weights[0];
        self.oblique_weights[ob + 1] = oblique_weights[1];
        self.oblique_thresholds[node_idx] = oblique_threshold;
        (left_idx, right_idx)
    }

    pub(super) fn into_tree(self) -> DecisionTree {
        DecisionTree {
            split_features: self.split_features,
            split_bins: self.split_bins,
            values: self.values,
            left_children: self.left_children,
            right_children: self.right_children,
            missing_goes_left: self.missing_goes_left,
            is_oblique_split: self.is_oblique_split,
            is_cat_split: self.is_cat_split,
            cat_left_masks: self.cat_left_masks,
            oblique_features: self.oblique_features,
            oblique_weights: self.oblique_weights,
            oblique_thresholds: self.oblique_thresholds,
            cat_lookups: self.cat_lookups,
            ramp_slopes: Vec::new(),
            ramp_features: Vec::new(),
            ramp_k: 1,
            leaf_pair_slopes: Vec::new(),
            leaf_pair_features: Vec::new(),
            quad_slopes: Vec::new(),
            quad_pairs: Vec::new(),
            quad_n_interactions: 0,
            node_h_sum: self.node_h_sum,
            node_count: self.node_count,
            cat_pair_feat2: self.cat_pair_feat2,
            cat_pair_bucket_map_a: self.cat_pair_bucket_map_a,
            cat_pair_bucket_map_b: self.cat_pair_bucket_map_b,
            cat_pair_cell_mask: self.cat_pair_cell_mask,
            cat_pair_k_buckets: self.cat_pair_k_buckets,
        }
    }

    /// Dispatch from a SplitResult to either add_split or add_cat_pair_split.
    /// Consumes the SplitResult.
    pub(super) fn add_split_from_sr(
        &mut self,
        node_idx: usize,
        sr: SplitResult,
        leaf_value: f64,
    ) -> (usize, usize) {
        if sr.is_cat_pair {
            self.add_cat_pair_split(
                node_idx,
                sr.feat as u32,
                sr.pair_feat2,
                sr.pair_map_a,
                sr.pair_map_b,
                sr.pair_cell_mask,
                sr.pair_k_buckets,
                leaf_value,
                sr.missing_left,
            )
        } else {
            self.add_split(
                node_idx,
                sr.feat as u32,
                sr.bin as u16,
                leaf_value,
                sr.missing_left,
                sr.is_oblique,
                sr.oblique_feats,
                sr.oblique_weights,
                sr.oblique_threshold,
                sr.is_cat,
                sr.cat_mask,
            )
        }
    }

    /// GGFP v5.0 — install a JIT-CatPairSplit at `node_idx`. Mutually exclusive
    /// with `add_split`. Returns (left_child_idx, right_child_idx).
    pub(super) fn add_cat_pair_split(
        &mut self,
        node_idx: usize,
        feat1: u32,
        feat2: u32,
        map_a: Vec<u8>,
        map_b: Vec<u8>,
        cell_mask: u64,
        k_buckets: u8,
        value: f64,
        missing_left: bool,
    ) -> (usize, usize) {
        let left_idx = self.add_node();
        let right_idx = self.add_node();
        self.split_features[node_idx] = feat1;
        self.split_bins[node_idx] = 0;
        self.values[node_idx] = value;
        self.left_children[node_idx] = left_idx as u32;
        self.right_children[node_idx] = right_idx as u32;
        self.missing_goes_left[node_idx] = missing_left;
        self.is_oblique_split[node_idx] = false;
        self.is_cat_split[node_idx] = false;
        self.cat_left_masks[node_idx] = Vec::new();
        self.cat_pair_feat2[node_idx] = feat2;
        self.cat_pair_bucket_map_a[node_idx] = map_a;
        self.cat_pair_bucket_map_b[node_idx] = map_b;
        self.cat_pair_cell_mask[node_idx] = cell_mask;
        self.cat_pair_k_buckets[node_idx] = k_buckets;
        (left_idx, right_idx)
    }
}

#[inline]
pub(super) fn sum_gh(gradients: &[f64], hessians: &[f64], indices: &[u32]) -> (f64, f64) {
    let mut g = 0.0f64;
    let mut h = 0.0f64;
    for &idx in indices {
        g += gradients[idx as usize];
        h += hessians[idx as usize];
    }
    (g, h)
}

#[inline]
pub(super) fn l1_shrink_gradient(g: f64, l1_reg: f64) -> f64 {
    if l1_reg <= 0.0 {
        g
    } else if g > l1_reg {
        g - l1_reg
    } else if g < -l1_reg {
        g + l1_reg
    } else {
        0.0
    }
}

#[inline]
pub(super) fn l1_gain_score(g: f64, h: f64, lambda_reg: f64, l1_reg: f64) -> f64 {
    let s = l1_shrink_gradient(g, l1_reg);
    s * s / (h + lambda_reg)
}

#[inline]
pub(super) fn l1_leaf_value(g: f64, h: f64, lambda_reg: f64, l1_reg: f64) -> f64 {
    -l1_shrink_gradient(g, l1_reg) / (h + lambda_reg)
}

/// Result from find_best_split.
#[derive(Clone)]
pub(super) struct SplitResult {
    pub(super) gain: f64,
    pub(super) feat: usize,
    pub(super) bin: usize,
    pub(super) missing_left: bool,
    pub(super) is_oblique: bool,
    pub(super) oblique_feats: [u32; 2],
    pub(super) oblique_weights: [f32; 2],
    pub(super) oblique_threshold: f32,
    pub(super) is_cat: bool,
    pub(super) cat_mask: CatBitmask,
    // GGFP v5.0 — JIT-CatPairSplit fields
    pub(super) is_cat_pair: bool,
    pub(super) pair_feat2: u32,
    pub(super) pair_map_a: Vec<u8>,
    pub(super) pair_map_b: Vec<u8>,
    pub(super) pair_cell_mask: u64,
    pub(super) pair_k_buckets: u8,
}

impl SplitResult {
    #[inline]
    pub(super) fn empty() -> Self {
        Self {
            gain: f64::NEG_INFINITY,
            feat: 0,
            bin: 0,
            missing_left: true,
            is_oblique: false,
            oblique_feats: [u32::MAX, u32::MAX],
            oblique_weights: [0.0, 0.0],
            oblique_threshold: 0.0,
            is_cat: false,
            cat_mask: Vec::new(),
            is_cat_pair: false,
            pair_feat2: u32::MAX,
            pair_map_a: Vec::new(),
            pair_map_b: Vec::new(),
            pair_cell_mask: 0,
            pair_k_buckets: 0,
        }
    }

    #[inline]
    pub(super) fn axis(
        gain: f64,
        feat: usize,
        bin: usize,
        missing_left: bool,
        is_cat: bool,
        cat_mask: CatBitmask,
    ) -> Self {
        Self {
            gain,
            feat,
            bin,
            missing_left,
            is_oblique: false,
            oblique_feats: [u32::MAX, u32::MAX],
            oblique_weights: [0.0, 0.0],
            oblique_threshold: 0.0,
            is_cat,
            cat_mask,
            is_cat_pair: false,
            pair_feat2: u32::MAX,
            pair_map_a: Vec::new(),
            pair_map_b: Vec::new(),
            pair_cell_mask: 0,
            pair_k_buckets: 0,
        }
    }

    #[inline]
    pub(super) fn oblique(
        gain: f64,
        feat: usize,
        missing_left: bool,
        oblique_feats: [u32; 2],
        oblique_weights: [f32; 2],
        oblique_threshold: f32,
    ) -> Self {
        Self {
            gain,
            feat,
            bin: 0,
            missing_left,
            is_oblique: true,
            oblique_feats,
            oblique_weights,
            oblique_threshold,
            is_cat: false,
            cat_mask: Vec::new(),
            is_cat_pair: false,
            pair_feat2: u32::MAX,
            pair_map_a: Vec::new(),
            pair_map_b: Vec::new(),
            pair_cell_mask: 0,
            pair_k_buckets: 0,
        }
    }

    #[inline]
    pub(super) fn cat_pair(
        gain: f64,
        feat1: usize,
        pair_feat2: u32,
        pair_map_a: Vec<u8>,
        pair_map_b: Vec<u8>,
        pair_cell_mask: u64,
        pair_k_buckets: u8,
        missing_left: bool,
    ) -> Self {
        Self {
            gain,
            feat: feat1,
            bin: 0,
            missing_left,
            is_oblique: false,
            oblique_feats: [u32::MAX, u32::MAX],
            oblique_weights: [0.0, 0.0],
            oblique_threshold: 0.0,
            is_cat: false,
            cat_mask: Vec::new(),
            is_cat_pair: true,
            pair_feat2,
            pair_map_a,
            pair_map_b,
            pair_cell_mask,
            pair_k_buckets,
        }
    }
}

#[inline(always)]
pub(super) fn combine_complement_gain(struct_gain: f64, comp_gain: f64, mode: u8) -> f64 {
    let gs = if struct_gain.is_finite() {
        struct_gain.max(0.0)
    } else {
        0.0
    };
    let gc = if comp_gain.is_finite() {
        comp_gain.max(0.0)
    } else {
        0.0
    };
    match mode {
        1 => (gs * gc).sqrt(),
        2 => gs.min(gc),
        3 => 0.5 * (gs + gc),
        _ => gs,
    }
}

#[inline]
pub(super) fn oblique_proj_binned_split(
    sr: &SplitResult,
    binned: &BinnedData,
    row: usize,
) -> Option<f64> {
    let f0 = sr.oblique_feats[0];
    if f0 == u32::MAX {
        return None;
    }
    let b0 = binned.get_bin_u16(row, f0 as usize);
    if b0 == MISSING_BIN {
        return None;
    }
    let mut proj = sr.oblique_weights[0] as f64 * b0 as f64;
    let f1 = sr.oblique_feats[1];
    if f1 != u32::MAX {
        let b1 = binned.get_bin_u16(row, f1 as usize);
        if b1 == MISSING_BIN {
            return None;
        }
        proj += sr.oblique_weights[1] as f64 * b1 as f64;
    }
    Some(proj)
}

#[inline]
pub(super) fn split_goes_left_binned(sr: &SplitResult, binned: &BinnedData, row: usize) -> bool {
    if sr.is_oblique {
        if let Some(proj) = oblique_proj_binned_split(sr, binned, row) {
            proj <= sr.oblique_threshold as f64
        } else {
            sr.missing_left
        }
    } else if sr.is_cat_pair {
        // GGFP v5.0 — cat-pair routing at build time
        let f1 = sr.feat;
        let f2 = sr.pair_feat2 as usize;
        let b1 = binned.get_bin_u16(row, f1);
        let b2 = binned.get_bin_u16(row, f2);
        if b1 == MISSING_BIN || b2 == MISSING_BIN {
            return sr.missing_left;
        }
        let b1u = b1 as usize;
        let b2u = b2 as usize;
        if b1u >= sr.pair_map_a.len() || b2u >= sr.pair_map_b.len() {
            return sr.missing_left;
        }
        let bu1 = sr.pair_map_a[b1u] as usize;
        let bu2 = sr.pair_map_b[b2u] as usize;
        let k = sr.pair_k_buckets as usize;
        if k == 0 {
            return sr.missing_left;
        }
        let cell = bu1 * k + bu2;
        if cell >= 64 {
            return sr.missing_left;
        }
        ((sr.pair_cell_mask >> cell) & 1) == 1
    } else {
        let bin = binned.get_bin_u16(row, sr.feat);
        if bin == MISSING_BIN {
            sr.missing_left
        } else if sr.is_cat {
            bitmask_test(&sr.cat_mask, bin as usize)
        } else {
            bin <= sr.bin as u16
        }
    }
}

#[inline]
pub(super) fn eval_fixed_split_gain(
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    g_sum: f64,
    h_sum: f64,
    feat: usize,
    split_bin: usize,
    missing_left: bool,
    is_cat: bool,
    cat_mask: &[u64],
    lambda_reg: f64,
    gamma: f64,
    min_h: f64,
    gain_penalty: f64,
) -> f64 {
    if node_indices.len() <= 1 || h_sum < min_h {
        return f64::NEG_INFINITY;
    }
    let col_bins = binned.col_bins(feat);
    let mut lg = 0.0f64;
    let mut lh = 0.0f64;
    let mut missing_h = 0.0f64;
    let split_bin_u16 = split_bin as u16;
    for &idx in node_indices {
        let bin = col_bins[idx as usize];
        let goes_left = if bin == MISSING_BIN {
            missing_h += hessians[idx as usize];
            missing_left
        } else if is_cat {
            bitmask_test(cat_mask, bin as usize)
        } else {
            bin <= split_bin_u16
        };
        if goes_left {
            lg += gradients[idx as usize];
            lh += hessians[idx as usize];
        }
    }
    let rg = g_sum - lg;
    let rh = h_sum - lh;
    if lh < min_h || rh < min_h {
        return f64::NEG_INFINITY;
    }
    let mut gain = 0.5
        * (lg * lg / (lh + lambda_reg) + rg * rg / (rh + lambda_reg)
            - g_sum * g_sum / (h_sum + lambda_reg))
        - gamma;
    if gain_penalty > 0.0 {
        gain -= gain_penalty
            * 0.5
            * (1.0 / (lh + lambda_reg) + 1.0 / (rh + lambda_reg) - 1.0 / (h_sum + lambda_reg));
    }
    gain = evidence_adjusted_gain(
        binned,
        gain,
        lh,
        rh,
        h_sum,
        lambda_reg,
        binned.n_bins(feat).saturating_sub(1),
    );
    gain -= missing_route_penalty(h_sum, missing_h);
    gain
}

#[inline(always)]
fn taylor_loss_at_weight(g: f64, h: f64, w: f64) -> f64 {
    g * w + 0.5 * h * w * w
}

#[inline]
fn fixed_split_child_sums(
    sr: &SplitResult,
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    indices: &[u32],
) -> (f64, f64) {
    let mut lg = 0.0f64;
    let mut lh = 0.0f64;
    for &idx in indices {
        let row = idx as usize;
        if split_goes_left_binned(sr, binned, row) {
            lg += gradients[row];
            lh += hessians[row];
        }
    }
    (lg, lh)
}

/// Audit a selected split by fitting fixed leaf weights on the search rows and
/// evaluating those weights on audit rows. This asks whether the selected child
/// predictions reduce held-out Taylor pseudo-loss, instead of refitting a fresh
/// gain on the audit subset.
#[inline]
pub(super) fn eval_fixed_split_pseudo_gain(
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    search_indices: &[u32],
    audit_indices: &[u32],
    search_g_sum: f64,
    search_h_sum: f64,
    audit_g_sum: f64,
    audit_h_sum: f64,
    sr: &SplitResult,
    lambda_reg: f64,
    l1_reg: f64,
    min_h: f64,
) -> f64 {
    if search_indices.len() <= 1 || audit_indices.is_empty() || search_h_sum < min_h {
        return f64::NEG_INFINITY;
    }

    let (fit_lg, fit_lh) = fixed_split_child_sums(sr, binned, gradients, hessians, search_indices);
    let fit_rg = search_g_sum - fit_lg;
    let fit_rh = search_h_sum - fit_lh;
    if fit_lh < min_h || fit_rh < min_h {
        return f64::NEG_INFINITY;
    }

    let parent_w = l1_leaf_value(search_g_sum, search_h_sum, lambda_reg, l1_reg);
    let left_w = l1_leaf_value(fit_lg, fit_lh, lambda_reg, l1_reg);
    let right_w = l1_leaf_value(fit_rg, fit_rh, lambda_reg, l1_reg);
    if !(parent_w.is_finite() && left_w.is_finite() && right_w.is_finite()) {
        return f64::NEG_INFINITY;
    }

    let (audit_lg, audit_lh) =
        fixed_split_child_sums(sr, binned, gradients, hessians, audit_indices);
    let audit_rg = audit_g_sum - audit_lg;
    let audit_rh = audit_h_sum - audit_lh;

    let parent_loss = taylor_loss_at_weight(audit_g_sum, audit_h_sum, parent_w);
    let split_loss = taylor_loss_at_weight(audit_lg, audit_lh, left_w)
        + taylor_loss_at_weight(audit_rg, audit_rh, right_w);
    parent_loss - split_loss
}

/// CLL candidate: best multi-way categorical split for a node.
/// For single features: feat2 == usize::MAX.
/// For pairs: feat and feat2 are the two features, bins are hashed pair bins.
pub(super) struct CllCandidate {
    pub(super) gain: f64,
    pub(super) feat: usize,
    pub(super) feat2: usize,
    pub(super) feat3: usize,
    pub(super) bin_g: Vec<f64>,
    pub(super) bin_h: Vec<f64>,
    pub(super) n_bins: usize,
    pub(super) n_active: usize,
    pub(super) pair_stride: usize,
    pub(super) triple_stride: usize,
}

/// NLL candidate: best coarse-binned numeric lookup for a node.
pub(super) struct NllCandidate {
    pub(super) gain: f64,
    pub(super) feat: usize,
    pub(super) bin_g: Vec<f64>,
    pub(super) bin_h: Vec<f64>,
    pub(super) n_bins: usize,
    pub(super) n_active: usize,
}

pub(super) struct LinearCandidate {
    pub(super) gain: f64,
    pub(super) feats: [usize; 2],
    pub(super) slopes: [f64; 2],
    pub(super) n_feats: usize,
    pub(super) intercept: f64,
}

#[derive(Clone)]
pub(super) enum LeafExpertKind {
    Lookup(CatLookup),
    Linear {
        feats: [usize; 2],
        slopes: [f64; 2],
        n_feats: usize,
        intercept: f64,
    },
    Bilinear {
        feats: [usize; 2],
        slopes: [f64; 2],
        n_feats: usize,
        pair_slope: f64,
        intercept: f64,
    },
}

pub(super) struct LeafExpertCandidate {
    pub(super) score: f64,
    pub(super) kind: LeafExpertKind,
}

/// Evaluate CLL (Category Lookup Leaf) for a node: find the best categorical feature
/// for multi-way splitting. Returns None if no categorical features or no gain.
pub(super) fn eval_cll_for_node(
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    g_sum: f64,
    h_sum: f64,
    lambda_reg: f64,
    gamma: f64,
    min_child_weight: f64,
) -> Option<CllCandidate> {
    let base_obj = g_sum * g_sum / (h_sum + lambda_reg);
    let mut best: Option<CllCandidate> = None;

    for col in 0..binned.n_features {
        // Use cll_is_categorical (includes >256 hashed features)
        if col >= binned.cll_is_categorical.len() || !binned.cll_is_categorical[col] {
            continue;
        }
        let n_bins = binned.cll_n_bins[col];
        if n_bins == 0 {
            continue;
        }
        let mut bin_g = vec![0.0f64; n_bins];
        let mut bin_h = vec![0.0f64; n_bins];

        // Use cll_hash_bins (same as bin_indices for native cats, hashed for >256)
        let col_offset = col * binned.n_rows;
        for &idx in node_indices {
            let row = idx as usize;
            let bin = binned.cll_hash_bins[col_offset + idx as usize];
            if bin == MISSING_BIN {
                continue;
            }
            let b = bin as usize;
            if b < n_bins {
                let g = gradients[row];
                let h = hessians[row];
                bin_g[b] += g;
                bin_h[b] += h;
            }
        }

        let mut cll_obj = 0.0f64;
        let mut n_active = 0usize;
        for b in 0..n_bins {
            if bin_h[b] >= min_child_weight {
                cll_obj += bin_g[b] * bin_g[b] / (bin_h[b] + lambda_reg);
                n_active += 1;
            }
        }
        if n_active < 2 {
            continue;
        }

        let gain = 0.5 * (cll_obj - base_obj) - gamma * (n_active as f64).sqrt();
        if gain > best.as_ref().map_or(0.0, |b| b.gain) {
            best = Some(CllCandidate {
                gain,
                feat: col,
                feat2: usize::MAX,
                feat3: usize::MAX,
                bin_g,
                bin_h,
                n_bins,
                n_active,
                pair_stride: 0,
                triple_stride: 0,
            });
        }
    }

    best
}

pub(super) fn eval_nll_for_node(
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    g_sum: f64,
    h_sum: f64,
    lambda_reg: f64,
    gamma: f64,
    min_child_weight: f64,
) -> Option<NllCandidate> {
    let n_leaf = node_indices.len();
    let nll_n_bins = (n_leaf / 8).clamp(2, 4);
    if nll_n_bins < 2 {
        return None;
    }

    let num_cols: Vec<usize> = (0..binned.n_features)
        .filter(|&c| {
            if c < binned.is_categorical.len() && binned.is_categorical[c] {
                return false;
            }
            if c < binned.cll_is_categorical.len() && binned.cll_is_categorical[c] {
                return false;
            }
            true
        })
        .collect();
    if num_cols.is_empty() {
        return None;
    }

    let base_obj = g_sum * g_sum / (h_sum + lambda_reg);
    let mut best: Option<NllCandidate> = None;

    for &col in &num_cols {
        let mut bin_g = vec![0.0f64; nll_n_bins];
        let mut bin_h = vec![0.0f64; nll_n_bins];
        let col_offset = col * binned.n_rows;

        for &idx in node_indices {
            let row = idx as usize;
            let orig_bin = binned.bin_indices[col_offset + row];
            if orig_bin == MISSING_BIN {
                continue;
            }
            let coarse = ((orig_bin as usize * nll_n_bins) >> 8).min(nll_n_bins - 1);
            let g = gradients[row];
            let h = hessians[row];
            bin_g[coarse] += g;
            bin_h[coarse] += h;
        }

        let mut nll_obj = 0.0f64;
        let mut n_active = 0usize;
        for b in 0..nll_n_bins {
            if bin_h[b] >= min_child_weight {
                nll_obj += bin_g[b] * bin_g[b] / (bin_h[b] + lambda_reg);
                n_active += 1;
            }
        }
        if n_active < 2 {
            continue;
        }

        let gain = 0.5 * (nll_obj - base_obj) - 1.25 * gamma * (n_active as f64).sqrt();
        if gain > best.as_ref().map_or(0.0, |b| b.gain) {
            best = Some(NllCandidate {
                gain,
                feat: col,
                bin_g,
                bin_h,
                n_bins: nll_n_bins,
                n_active,
            });
        }
    }

    best
}

pub(super) fn eval_pair_cll_for_node(
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    g_sum: f64,
    h_sum: f64,
    lambda_reg: f64,
    gamma: f64,
    min_child_weight: f64,
) -> Option<CllCandidate> {
    if node_indices.len() < 24 {
        return None;
    }

    let cat_cols: Vec<usize> = (0..binned.n_features)
        .filter(|&c| c < binned.cll_is_categorical.len() && binned.cll_is_categorical[c])
        .collect();
    if cat_cols.len() < 2 {
        return None;
    }

    let base_obj = g_sum * g_sum / (h_sum + lambda_reg);
    let mut feat_scores: Vec<(f64, usize)> = Vec::new();
    for &col in &cat_cols {
        let n_bins = binned.cll_n_bins[col];
        if n_bins == 0 {
            continue;
        }
        let mut bin_g = vec![0.0f64; n_bins];
        let mut bin_h = vec![0.0f64; n_bins];
        let col_offset = col * binned.n_rows;
        for &idx in node_indices {
            let bin = binned.cll_hash_bins[col_offset + idx as usize];
            if bin == MISSING_BIN {
                continue;
            }
            let b = bin as usize;
            if b < n_bins {
                bin_g[b] += gradients[idx as usize];
                bin_h[b] += hessians[idx as usize];
            }
        }
        let mut obj = 0.0f64;
        let mut n_active = 0usize;
        for b in 0..n_bins {
            if bin_h[b] >= min_child_weight {
                obj += bin_g[b] * bin_g[b] / (bin_h[b] + lambda_reg);
                n_active += 1;
            }
        }
        if n_active >= 2 {
            let gain = 0.5 * (obj - base_obj) - gamma * (n_active as f64).sqrt();
            feat_scores.push((gain, col));
        }
    }
    if feat_scores.len() < 2 {
        return None;
    }
    feat_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    if feat_scores.len() > 6 {
        feat_scores.truncate(6);
    }

    let mut best: Option<CllCandidate> = None;
    for i in 0..feat_scores.len() {
        for j in (i + 1)..feat_scores.len() {
            let fi = feat_scores[i].1;
            let fj = feat_scores[j].1;
            let pair_n_bins = 64usize;
            let pair_stride = 0usize;
            let mut bin_g = vec![0.0f64; pair_n_bins];
            let mut bin_h = vec![0.0f64; pair_n_bins];
            let off_i = fi * binned.n_rows;
            let off_j = fj * binned.n_rows;
            for &idx in node_indices {
                let row = idx as usize;
                let b1 = binned.cll_hash_bins[off_i + row];
                let b2 = binned.cll_hash_bins[off_j + row];
                if b1 == MISSING_BIN || b2 == MISSING_BIN {
                    continue;
                }
                let b =
                    ((b1 as u32).wrapping_mul(257).wrapping_add(b2 as u32)) as usize % pair_n_bins;
                let g = gradients[row];
                let h = hessians[row];
                bin_g[b] += g;
                bin_h[b] += h;
            }
            let mut pair_obj = 0.0f64;
            let mut n_active = 0usize;
            for b in 0..pair_n_bins {
                if bin_h[b] >= min_child_weight {
                    pair_obj += bin_g[b] * bin_g[b] / (bin_h[b] + lambda_reg);
                    n_active += 1;
                }
            }
            if n_active < 2 {
                continue;
            }
            let gain = 0.5 * (pair_obj - base_obj) - 1.5 * gamma * (n_active as f64).sqrt();
            if gain > best.as_ref().map_or(0.0, |b| b.gain) {
                best = Some(CllCandidate {
                    gain,
                    feat: fi,
                    feat2: fj,
                    feat3: usize::MAX,
                    bin_g,
                    bin_h,
                    n_bins: pair_n_bins,
                    n_active,
                    pair_stride,
                    triple_stride: 0,
                });
            }
        }
    }
    best
}

#[inline]
fn tuple_hash2(b1: u16, b2: u16, n_bins: usize) -> usize {
    ((b1 as u32).wrapping_mul(257).wrapping_add(b2 as u32)) as usize % n_bins.max(1)
}

#[inline]
fn tuple_hash3(b1: u16, b2: u16, b3: u16, n_bins: usize) -> usize {
    ((b1 as u32)
        .wrapping_mul(257)
        .wrapping_add((b2 as u32).wrapping_mul(17))
        .wrapping_add(b3 as u32)) as usize
        % n_bins.max(1)
}

pub(super) fn eval_cat_tuple_lookup_for_node(
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    g_sum: f64,
    h_sum: f64,
    leaf_value: f64,
    lambda_reg: f64,
    gamma: f64,
    min_child_weight: f64,
    smooth: f64,
    cfg: &CatTupleConfig,
) -> Option<LeafExpertCandidate> {
    if !cfg.enabled || cfg.max_order < 2 || node_indices.len() < cfg.min_leaf.max(2) {
        return None;
    }

    let cat_cols: Vec<usize> = (0..binned.n_features)
        .filter(|&c| {
            c < binned.cll_is_categorical.len()
                && binned.cll_is_categorical[c]
                && c < binned.cll_n_bins.len()
                && binned.cll_n_bins[c] >= 2
        })
        .collect();
    if cat_cols.len() < 2 {
        return None;
    }

    let n_leaf = node_indices.len();
    let base_obj = g_sum * g_sum / (h_sum + lambda_reg);
    let mut feat_scores: Vec<(f64, usize)> = Vec::new();

    for &col in &cat_cols {
        let n_bins = binned.cll_n_bins[col];
        if n_bins == 0 {
            continue;
        }
        let mut bin_g = vec![0.0f64; n_bins];
        let mut bin_h = vec![0.0f64; n_bins];
        let col_offset = col * binned.n_rows;
        for &idx in node_indices {
            let row = idx as usize;
            let bin = binned.cll_hash_bins[col_offset + row];
            if bin == MISSING_BIN {
                continue;
            }
            let b = bin as usize;
            if b < n_bins {
                bin_g[b] += gradients[row];
                bin_h[b] += hessians[row];
            }
        }
        let mut obj = 0.0f64;
        let mut n_active = 0usize;
        for b in 0..n_bins {
            if bin_h[b] >= min_child_weight {
                obj += bin_g[b] * bin_g[b] / (bin_h[b] + lambda_reg);
                n_active += 1;
            }
        }
        if n_active < 2 {
            continue;
        }
        let gain = 0.5 * (obj - base_obj) - gamma * (n_active as f64).sqrt();
        let df = lookup_effective_df(&bin_h, min_child_weight, smooth, lambda_reg);
        let score = expert_score_df(gain, n_leaf, df, false);
        if score.is_finite() && score > 0.0 {
            feat_scores.push((score, col));
        }
    }

    if feat_scores.len() < 2 {
        return None;
    }
    feat_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    feat_scores.truncate(cfg.top_features.clamp(2, 12).min(feat_scores.len()));

    let hash_bins = cfg.hash_bins.clamp(8, 512);
    let mut best: Option<(f64, CllCandidate)> = None;

    for i in 0..feat_scores.len() {
        let fi = feat_scores[i].1;
        let off_i = fi * binned.n_rows;
        let ni = binned.cll_n_bins[fi].max(1);
        for j in (i + 1)..feat_scores.len() {
            let fj = feat_scores[j].1;
            let off_j = fj * binned.n_rows;
            let nj = binned.cll_n_bins[fj].max(1);
            let exact_bins = ni.checked_mul(nj).filter(|&n| n <= hash_bins).unwrap_or(0);
            let (pair_n_bins, pair_stride) = if exact_bins > 0 {
                (exact_bins, nj)
            } else {
                (hash_bins, 0)
            };
            let mut bin_g = vec![0.0f64; pair_n_bins];
            let mut bin_h = vec![0.0f64; pair_n_bins];
            for &idx in node_indices {
                let row = idx as usize;
                let b1 = binned.cll_hash_bins[off_i + row];
                let b2 = binned.cll_hash_bins[off_j + row];
                if b1 == MISSING_BIN || b2 == MISSING_BIN {
                    continue;
                }
                let bu = if pair_stride > 0 {
                    let b1u = b1 as usize;
                    let b2u = b2 as usize;
                    if b1u >= ni || b2u >= nj {
                        continue;
                    }
                    b1u * pair_stride + b2u
                } else {
                    tuple_hash2(b1, b2, pair_n_bins)
                };
                bin_g[bu] += gradients[row];
                bin_h[bu] += hessians[row];
            }
            let mut obj = 0.0f64;
            let mut n_active = 0usize;
            for b in 0..pair_n_bins {
                if bin_h[b] >= min_child_weight {
                    obj += bin_g[b] * bin_g[b] / (bin_h[b] + lambda_reg);
                    n_active += 1;
                }
            }
            if n_active >= 2 {
                let gain = 0.5 * (obj - base_obj) - 1.5 * gamma * (n_active as f64).sqrt();
                let df = lookup_effective_df(&bin_h, min_child_weight, smooth, lambda_reg);
                let score = expert_score_df(gain, n_leaf, df, false);
                if score.is_finite() && score > best.as_ref().map_or(0.0, |b| b.0) {
                    best = Some((
                        score,
                        CllCandidate {
                            gain,
                            feat: fi,
                            feat2: fj,
                            feat3: usize::MAX,
                            bin_g,
                            bin_h,
                            n_bins: pair_n_bins,
                            n_active,
                            pair_stride,
                            triple_stride: 0,
                        },
                    ));
                }
            }
        }
    }

    if cfg.max_order >= 3 && feat_scores.len() >= 3 && node_indices.len() >= cfg.min_leaf.max(48) {
        for a in 0..feat_scores.len() {
            let f0 = feat_scores[a].1;
            let off0 = f0 * binned.n_rows;
            let n0 = binned.cll_n_bins[f0].max(1);
            for b in (a + 1)..feat_scores.len() {
                let f1 = feat_scores[b].1;
                let off1 = f1 * binned.n_rows;
                let n1 = binned.cll_n_bins[f1].max(1);
                for c in (b + 1)..feat_scores.len() {
                    let f2 = feat_scores[c].1;
                    let off2 = f2 * binned.n_rows;
                    let n2 = binned.cll_n_bins[f2].max(1);
                    let exact_bins = n0
                        .checked_mul(n1)
                        .and_then(|v| v.checked_mul(n2))
                        .filter(|&n| n <= hash_bins)
                        .unwrap_or(0);
                    let (triple_n_bins, pair_stride, triple_stride) = if exact_bins > 0 {
                        (exact_bins, n1, n2)
                    } else {
                        (hash_bins, 0, 0)
                    };
                    let mut bin_g = vec![0.0f64; triple_n_bins];
                    let mut bin_h = vec![0.0f64; triple_n_bins];
                    for &idx in node_indices {
                        let row = idx as usize;
                        let b0 = binned.cll_hash_bins[off0 + row];
                        let b1 = binned.cll_hash_bins[off1 + row];
                        let b2 = binned.cll_hash_bins[off2 + row];
                        if b0 == MISSING_BIN || b1 == MISSING_BIN || b2 == MISSING_BIN {
                            continue;
                        }
                        let bu = if pair_stride > 0 && triple_stride > 0 {
                            let b0u = b0 as usize;
                            let b1u = b1 as usize;
                            let b2u = b2 as usize;
                            if b0u >= n0 || b1u >= n1 || b2u >= n2 {
                                continue;
                            }
                            (b0u * pair_stride + b1u) * triple_stride + b2u
                        } else {
                            tuple_hash3(b0, b1, b2, triple_n_bins)
                        };
                        bin_g[bu] += gradients[row];
                        bin_h[bu] += hessians[row];
                    }
                    let mut obj = 0.0f64;
                    let mut n_active = 0usize;
                    for bin in 0..triple_n_bins {
                        if bin_h[bin] >= min_child_weight {
                            obj += bin_g[bin] * bin_g[bin] / (bin_h[bin] + lambda_reg);
                            n_active += 1;
                        }
                    }
                    if n_active < 3 {
                        continue;
                    }
                    let gain = 0.5 * (obj - base_obj) - 2.0 * gamma * (n_active as f64).sqrt();
                    let df = lookup_effective_df(&bin_h, min_child_weight, smooth, lambda_reg);
                    let score = expert_score_df(gain, n_leaf, df, false);
                    if score.is_finite() && score > best.as_ref().map_or(0.0, |b| b.0) {
                        best = Some((
                            score,
                            CllCandidate {
                                gain,
                                feat: f0,
                                feat2: f1,
                                feat3: f2,
                                bin_g,
                                bin_h,
                                n_bins: triple_n_bins,
                                n_active,
                                pair_stride,
                                triple_stride,
                            },
                        ));
                    }
                }
            }
        }
    }

    best.map(|(score, cll)| LeafExpertCandidate {
        score,
        kind: LeafExpertKind::Lookup(make_cll_lookup(
            &cll,
            leaf_value,
            smooth,
            lambda_reg,
            min_child_weight,
        )),
    })
}

#[inline]
pub(super) fn expert_score(raw_gain: f64, n_leaf: usize, n_active: usize, is_numeric: bool) -> f64 {
    let df = n_active.saturating_sub(1) as f64;
    expert_score_df(raw_gain, n_leaf, df, is_numeric)
}

#[inline]
fn expert_score_df(raw_gain: f64, n_leaf: usize, df: f64, is_numeric: bool) -> f64 {
    if !(raw_gain.is_finite() && raw_gain > 0.0) {
        return raw_gain;
    }
    if df <= 0.0 {
        return raw_gain;
    }
    let tau = if is_numeric { 8.0 } else { 4.0 };
    let n = n_leaf.max(1) as f64;
    raw_gain * (n / (n + tau * df))
}

#[inline]
fn lookup_effective_df(bin_h: &[f64], min_child_weight: f64, smooth: f64, lambda_reg: f64) -> f64 {
    let denom_extra = smooth.max(0.0) + lambda_reg.max(0.0);
    let mcw = min_child_weight.max(1e-12);
    let mut eff_sum = 0.0f64;
    let mut eff_h_weighted = 0.0f64;
    let mut active_h = 0.0f64;
    for &h in bin_h {
        if h > 0.0 {
            let activation = 1.0 - (-h / mcw).exp();
            let a = h / (h + denom_extra).max(1e-12);
            eff_sum += activation * a;
            eff_h_weighted += activation * h * a;
            active_h += activation * h;
        }
    }
    if active_h <= 0.0 {
        return 0.0;
    }
    (eff_sum - eff_h_weighted / active_h).max(0.0)
}

#[inline]
fn lookup_candidate_smooth(
    bin_g: &[f64],
    bin_h: &[f64],
    g_sum: f64,
    h_sum: f64,
    lambda_reg: f64,
    min_child_weight: f64,
    fallback_smooth: f64,
    adaptive: bool,
) -> f64 {
    let fallback = fallback_smooth.max(0.0);
    if !adaptive || fallback <= 0.0 || bin_g.len() != bin_h.len() {
        return fallback;
    }
    let mcw = min_child_weight.max(1e-12);
    let node_theta = -g_sum / (h_sum + lambda_reg.max(0.0)).max(1e-12);
    let mut w_sum = 0.0f64;
    let mut var_sum = 0.0f64;
    let mut noise_sum = 0.0f64;
    let mut active = 0usize;
    for (&g, &h) in bin_g.iter().zip(bin_h.iter()) {
        if h <= 0.0 {
            continue;
        }
        let activation = 1.0 - (-h / mcw).exp();
        let w = activation * h;
        if w <= 0.0 {
            continue;
        }
        let denom = (h + lambda_reg.max(0.0)).max(1e-12);
        let theta = -g / denom;
        let d = theta - node_theta;
        var_sum += w * d * d;
        noise_sum += w / denom;
        w_sum += w;
        active += 1;
    }
    if active < 2 || w_sum <= 0.0 {
        return fallback;
    }
    let tau2 = (var_sum / w_sum - noise_sum / w_sum).max(0.0);
    if tau2 <= 1e-12 || !tau2.is_finite() {
        return (fallback * 250.0).max(fallback).min(5000.0);
    }
    let smooth = 1.0 / tau2;
    let min_smooth = (fallback * 0.25).max(1.0);
    let max_smooth = (fallback * 250.0).max(min_smooth).min(5000.0);
    if max_smooth <= min_smooth {
        return min_smooth;
    }
    smooth.clamp(min_smooth, max_smooth)
}

/// Build a CatLookup from a CllCandidate with smoothing toward the leaf value.
pub(super) fn make_cll_lookup(
    cll: &CllCandidate,
    leaf_value: f64,
    smooth: f64,
    lambda_reg: f64,
    min_child_weight: f64,
) -> CatLookup {
    let mut bin_values = vec![leaf_value; cll.n_bins];
    for b in 0..cll.n_bins {
        if cll.bin_h[b] >= min_child_weight {
            let cat_value = -cll.bin_g[b] / (cll.bin_h[b] + lambda_reg);
            if smooth > 0.0 {
                bin_values[b] =
                    (cll.bin_h[b] * cat_value + smooth * leaf_value) / (cll.bin_h[b] + smooth);
            } else {
                bin_values[b] = cat_value;
            }
        }
    }
    CatLookup {
        feature: cll.feat as u32,
        feature2: cll.feat2 as u32,
        feature3: cll.feat3 as u32,
        bin_values,
        default_value: leaf_value,
        is_numeric: false,
        n_coarse_bins: 0,
        pair_stride: cll.pair_stride,
        triple_stride: cll.triple_stride,
    }
}

pub(super) fn make_nll_lookup(
    nll: &NllCandidate,
    leaf_value: f64,
    smooth: f64,
    lambda_reg: f64,
    min_child_weight: f64,
) -> CatLookup {
    let mut bin_values = vec![leaf_value; nll.n_bins];
    for b in 0..nll.n_bins {
        if nll.bin_h[b] >= min_child_weight {
            let opt_value = -nll.bin_g[b] / (nll.bin_h[b] + lambda_reg);
            if smooth > 0.0 {
                bin_values[b] =
                    (nll.bin_h[b] * opt_value + smooth * leaf_value) / (nll.bin_h[b] + smooth);
            } else {
                bin_values[b] = opt_value;
            }
        }
    }
    CatLookup {
        feature: nll.feat as u32,
        feature2: u32::MAX,
        feature3: u32::MAX,
        bin_values,
        default_value: leaf_value,
        is_numeric: true,
        n_coarse_bins: nll.n_bins,
        pair_stride: 0,
        triple_stride: 0,
    }
}

pub(super) fn eval_best_linear_for_node(
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    g_sum: f64,
    h_sum: f64,
    leaf_value: f64,
    ramp_lambda: f64,
    min_child_weight: f64,
) -> Option<LinearCandidate> {
    if node_indices.len() < 8 || h_sum <= min_child_weight.max(1e-10) {
        return None;
    }
    let lambda_eff = ramp_lambda.max(1e-12);
    let mut best: Option<LinearCandidate> = None;
    let mut top_single: Vec<(f64, usize)> = Vec::new();
    for col in 0..binned.n_features {
        if (col < binned.is_categorical.len() && binned.is_categorical[col])
            || (col < binned.cll_is_categorical.len() && binned.cll_is_categorical[col])
        {
            continue;
        }
        let n_bins = binned.n_bins(col).max(1) as f64;
        let col_offset = col * binned.n_rows;
        let mut gx = 0.0f64;
        let mut hx = 0.0f64;
        let mut hxx = 0.0f64;
        let mut n_valid = 0usize;
        let mut min_bin = usize::MAX;
        let mut max_bin = 0usize;
        for &idx in node_indices {
            let row = idx as usize;
            let bin = binned.bin_indices[col_offset + row];
            if bin == MISSING_BIN {
                continue;
            }
            let bu = bin as usize;
            let x = bin as f64 / n_bins;
            let g = gradients[row];
            let h = hessians[row];
            gx += g * x;
            hx += h * x;
            hxx += h * x * x;
            n_valid += 1;
            min_bin = min_bin.min(bu);
            max_bin = max_bin.max(bu);
        }
        if n_valid < 6 || min_bin >= max_bin {
            continue;
        }
        let x_bar = hx / h_sum.max(1e-12);
        let gx_c = gx - g_sum * x_bar;
        let hxx_c = hxx - hx * x_bar;
        if hxx_c <= 1e-12 {
            continue;
        }
        let raw_gain = 0.5 * gx_c * gx_c / (hxx_c + lambda_eff);
        let score = expert_score(raw_gain, node_indices.len(), 2, true);
        if !(score.is_finite() && score > 0.0) {
            continue;
        }
        let beta_c = -gx_c / (hxx_c + lambda_eff);
        let intercept = leaf_value - beta_c * x_bar;
        let slope = beta_c / n_bins;
        if best.as_ref().map_or(true, |cand| score > cand.gain) {
            best = Some(LinearCandidate {
                gain: score,
                feats: [col, usize::MAX],
                slopes: [slope, 0.0],
                n_feats: 1,
                intercept,
            });
        }
        top_single.push((score, col));
    }

    top_single.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    if top_single.len() > 4 {
        top_single.truncate(4);
    }

    for i in 0..top_single.len() {
        let c1 = top_single[i].1;
        for j in (i + 1)..top_single.len() {
            let c2 = top_single[j].1;
            if c1 == c2 {
                continue;
            }
            let nb1 = binned.n_bins(c1).max(1) as f64;
            let nb2 = binned.n_bins(c2).max(1) as f64;
            let off1 = c1 * binned.n_rows;
            let off2 = c2 * binned.n_rows;
            let mut gx = [0.0f64; 2];
            let mut hx = [0.0f64; 2];
            let mut hxx = [[0.0f64; 2]; 2];
            let mut valid1 = 0usize;
            let mut valid2 = 0usize;
            let mut min1 = usize::MAX;
            let mut max1 = 0usize;
            let mut min2 = usize::MAX;
            let mut max2 = 0usize;
            for &idx in node_indices {
                let row = idx as usize;
                let b1 = binned.bin_indices[off1 + row];
                let b2 = binned.bin_indices[off2 + row];
                let x1 = if b1 == MISSING_BIN {
                    0.0
                } else {
                    let bu = b1 as usize;
                    valid1 += 1;
                    min1 = min1.min(bu);
                    max1 = max1.max(bu);
                    b1 as f64 / nb1
                };
                let x2 = if b2 == MISSING_BIN {
                    0.0
                } else {
                    let bu = b2 as usize;
                    valid2 += 1;
                    min2 = min2.min(bu);
                    max2 = max2.max(bu);
                    b2 as f64 / nb2
                };
                let g = gradients[row];
                let h = hessians[row];
                gx[0] += g * x1;
                gx[1] += g * x2;
                hx[0] += h * x1;
                hx[1] += h * x2;
                hxx[0][0] += h * x1 * x1;
                hxx[0][1] += h * x1 * x2;
                hxx[1][1] += h * x2 * x2;
            }
            if valid1 < 6 || valid2 < 6 || min1 >= max1 || min2 >= max2 {
                continue;
            }
            hxx[1][0] = hxx[0][1];
            let x_bar = [hx[0] / h_sum.max(1e-12), hx[1] / h_sum.max(1e-12)];
            let gx_c = [gx[0] - g_sum * x_bar[0], gx[1] - g_sum * x_bar[1]];
            let a00 = hxx[0][0] - hx[0] * x_bar[0] + lambda_eff;
            let a01 = hxx[0][1] - hx[0] * x_bar[1];
            let a10 = hxx[1][0] - hx[1] * x_bar[0];
            let a11 = hxx[1][1] - hx[1] * x_bar[1] + lambda_eff;
            if a00 <= 1e-12 || a11 <= 1e-12 {
                continue;
            }
            let det = a00 * a11 - a01 * a10;
            if det <= 1e-12 {
                continue;
            }
            let rhs0 = -gx_c[0];
            let rhs1 = -gx_c[1];
            let beta0 = (rhs0 * a11 - a01 * rhs1) / det;
            let beta1 = (a00 * rhs1 - rhs0 * a10) / det;
            let raw_gain = 0.5 * (rhs0 * beta0 + rhs1 * beta1);
            let score = expert_score(raw_gain, node_indices.len(), 3, true);
            if !(score.is_finite() && score > 0.0) {
                continue;
            }
            let intercept = leaf_value - beta0 * x_bar[0] - beta1 * x_bar[1];
            let slopes = [beta0 / nb1, beta1 / nb2];
            if best.as_ref().map_or(true, |cand| score > cand.gain) {
                best = Some(LinearCandidate {
                    gain: score,
                    feats: [c1, c2],
                    slopes,
                    n_feats: 2,
                    intercept,
                });
            }
        }
    }
    best
}

pub(super) fn solve_linear_3x3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for i in 0..3 {
        let mut pivot = i;
        let mut pivot_abs = a[i][i].abs();
        for r in (i + 1)..3 {
            let cand = a[r][i].abs();
            if cand > pivot_abs {
                pivot = r;
                pivot_abs = cand;
            }
        }
        if pivot_abs <= 1e-12 {
            return None;
        }
        if pivot != i {
            a.swap(i, pivot);
            b.swap(i, pivot);
        }
        let diag = a[i][i];
        for c in i..3 {
            a[i][c] /= diag;
        }
        b[i] /= diag;
        for r in 0..3 {
            if r == i {
                continue;
            }
            let factor = a[r][i];
            if factor.abs() <= 1e-18 {
                continue;
            }
            for c in i..3 {
                a[r][c] -= factor * a[i][c];
            }
            b[r] -= factor * b[i];
        }
    }
    Some(b)
}

pub(super) fn eval_best_bilinear_for_node(
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    g_sum: f64,
    h_sum: f64,
    leaf_value: f64,
    ramp_lambda: f64,
    min_child_weight: f64,
) -> Option<LeafExpertCandidate> {
    if node_indices.len() < 12 || h_sum <= min_child_weight.max(1e-10) {
        return None;
    }
    let n_numeric = (0..binned.n_features)
        .filter(|&col| {
            !((col < binned.is_categorical.len() && binned.is_categorical[col])
                || (col < binned.cll_is_categorical.len() && binned.cll_is_categorical[col]))
        })
        .count();
    if !(2..=10).contains(&n_numeric) {
        return None;
    }
    let lambda_eff = ramp_lambda.max(1e-12);
    let mut top_single: Vec<(f64, usize)> = Vec::new();
    for col in 0..binned.n_features {
        if (col < binned.is_categorical.len() && binned.is_categorical[col])
            || (col < binned.cll_is_categorical.len() && binned.cll_is_categorical[col])
        {
            continue;
        }
        let n_bins = binned.n_bins(col).max(1) as f64;
        let col_offset = col * binned.n_rows;
        let mut gx = 0.0f64;
        let mut hx = 0.0f64;
        let mut hxx = 0.0f64;
        let mut n_valid = 0usize;
        let mut min_bin = usize::MAX;
        let mut max_bin = 0usize;
        for &idx in node_indices {
            let row = idx as usize;
            let bin = binned.bin_indices[col_offset + row];
            if bin == MISSING_BIN {
                continue;
            }
            let bu = bin as usize;
            let x = bin as f64 / n_bins;
            let g = gradients[row];
            let h = hessians[row];
            gx += g * x;
            hx += h * x;
            hxx += h * x * x;
            n_valid += 1;
            min_bin = min_bin.min(bu);
            max_bin = max_bin.max(bu);
        }
        if n_valid < 6 || min_bin >= max_bin {
            continue;
        }
        let x_bar = hx / h_sum.max(1e-12);
        let gx_c = gx - g_sum * x_bar;
        let hxx_c = hxx - hx * x_bar;
        if hxx_c <= 1e-12 {
            continue;
        }
        let raw_gain = 0.5 * gx_c * gx_c / (hxx_c + lambda_eff);
        let score = expert_score(raw_gain, node_indices.len(), 2, true);
        if score.is_finite() && score > 0.0 {
            top_single.push((score, col));
        }
    }
    top_single.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    if top_single.len() > 4 {
        top_single.truncate(4);
    }
    if top_single.len() < 2 {
        return None;
    }

    let mut best: Option<LeafExpertCandidate> = None;
    for i in 0..top_single.len() {
        let c1 = top_single[i].1;
        for j in (i + 1)..top_single.len() {
            let c2 = top_single[j].1;
            let nb1 = binned.n_bins(c1).max(1) as f64;
            let nb2 = binned.n_bins(c2).max(1) as f64;
            let off1 = c1 * binned.n_rows;
            let off2 = c2 * binned.n_rows;
            let mut gx = [0.0f64; 3];
            let mut hx = [0.0f64; 3];
            let mut hxx = [[0.0f64; 3]; 3];
            let mut valid_pair = 0usize;
            let mut min_prod = f64::INFINITY;
            let mut max_prod = f64::NEG_INFINITY;
            for &idx in node_indices {
                let row = idx as usize;
                let b1 = binned.bin_indices[off1 + row];
                let b2 = binned.bin_indices[off2 + row];
                let x1 = if b1 == MISSING_BIN {
                    0.0
                } else {
                    b1 as f64 / nb1
                };
                let x2 = if b2 == MISSING_BIN {
                    0.0
                } else {
                    b2 as f64 / nb2
                };
                let z = x1 * x2;
                if b1 != MISSING_BIN && b2 != MISSING_BIN {
                    valid_pair += 1;
                    min_prod = min_prod.min(z);
                    max_prod = max_prod.max(z);
                }
                let xs = [x1, x2, z];
                let g = gradients[row];
                let h = hessians[row];
                for p in 0..3 {
                    gx[p] += g * xs[p];
                    hx[p] += h * xs[p];
                    for q in p..3 {
                        hxx[p][q] += h * xs[p] * xs[q];
                    }
                }
            }
            if valid_pair < 8 || !(min_prod < max_prod) {
                continue;
            }
            for p in 0..3 {
                for q in 0..p {
                    hxx[p][q] = hxx[q][p];
                }
            }
            let mut x_bar = [0.0f64; 3];
            let mut rhs = [0.0f64; 3];
            let mut a = [[0.0f64; 3]; 3];
            let h_safe = h_sum.max(1e-12);
            for p in 0..3 {
                x_bar[p] = hx[p] / h_safe;
                rhs[p] = -(gx[p] - g_sum * x_bar[p]);
            }
            for p in 0..3 {
                for q in 0..3 {
                    a[p][q] = hxx[p][q] - hx[p] * x_bar[q];
                    if p == q {
                        a[p][q] += lambda_eff;
                    }
                }
            }
            let Some(beta) = solve_linear_3x3(a, rhs) else {
                continue;
            };
            let raw_gain = 0.5 * (rhs[0] * beta[0] + rhs[1] * beta[1] + rhs[2] * beta[2]);
            let score = expert_score(raw_gain, node_indices.len(), 4, true);
            if !(score.is_finite() && score > best.as_ref().map_or(0.0, |cand| cand.score)) {
                continue;
            }
            let intercept =
                leaf_value - beta[0] * x_bar[0] - beta[1] * x_bar[1] - beta[2] * x_bar[2];
            best = Some(LeafExpertCandidate {
                score,
                kind: LeafExpertKind::Bilinear {
                    feats: [c1, c2],
                    slopes: [beta[0] / nb1, beta[1] / nb2],
                    n_feats: 2,
                    pair_slope: beta[2] / (nb1 * nb2),
                    intercept,
                },
            });
        }
    }
    best
}

pub(super) fn eval_guided_cll_for_node(
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    g_sum: f64,
    h_sum: f64,
    lambda_reg: f64,
    gamma: f64,
    min_child_weight: f64,
    choice: &GuidedCatChoice,
) -> Option<CllCandidate> {
    let feat = choice.feature as usize;
    if feat >= binned.n_features
        || feat >= binned.cll_is_categorical.len()
        || !binned.cll_is_categorical[feat]
    {
        return None;
    }
    let feat2 = choice.feature2 as usize;
    let use_pair = choice.feature2 != u32::MAX;
    if use_pair
        && (feat2 >= binned.n_features
            || feat2 >= binned.cll_is_categorical.len()
            || !binned.cll_is_categorical[feat2])
    {
        return None;
    }
    let n_bins = choice.n_bins.max(1);
    let mut bin_g = vec![0.0f64; n_bins];
    let mut bin_h = vec![0.0f64; n_bins];
    let off1 = feat * binned.n_rows;
    let off2 = if use_pair { feat2 * binned.n_rows } else { 0 };
    for &idx in node_indices {
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
            if choice.pair_stride > 0 {
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
        let g = gradients[row];
        let h = hessians[row];
        bin_g[bin] += g;
        bin_h[bin] += h;
    }
    let base_obj = g_sum * g_sum / (h_sum + lambda_reg);
    let mut obj = 0.0f64;
    let mut n_active = 0usize;
    for b in 0..n_bins {
        if bin_h[b] >= min_child_weight {
            obj += bin_g[b] * bin_g[b] / (bin_h[b] + lambda_reg);
            n_active += 1;
        }
    }
    if n_active < 2 {
        return None;
    }
    let gain = 0.5 * (obj - base_obj)
        - if use_pair {
            1.5 * gamma * (n_active as f64).sqrt()
        } else {
            gamma * (n_active as f64).sqrt()
        };
    Some(CllCandidate {
        gain,
        feat,
        feat2: if use_pair { feat2 } else { usize::MAX },
        feat3: usize::MAX,
        bin_g,
        bin_h,
        n_bins,
        n_active,
        pair_stride: if use_pair { choice.pair_stride } else { 0 },
        triple_stride: 0,
    })
}

pub(super) fn eval_best_lookup_for_node(
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    g_sum: f64,
    h_sum: f64,
    leaf_value: f64,
    lambda_reg: f64,
    gamma: f64,
    min_child_weight: f64,
    smooth: f64,
    guided_choice: Option<&GuidedCatChoice>,
) -> Option<LeafExpertCandidate> {
    eval_best_lookup_for_node_with_config(
        binned,
        gradients,
        hessians,
        node_indices,
        g_sum,
        h_sum,
        leaf_value,
        lambda_reg,
        gamma,
        min_child_weight,
        smooth,
        false,
        guided_choice,
        None,
    )
}

pub(super) fn eval_best_lookup_for_node_with_config(
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    g_sum: f64,
    h_sum: f64,
    leaf_value: f64,
    lambda_reg: f64,
    gamma: f64,
    min_child_weight: f64,
    smooth: f64,
    adaptive_smooth: bool,
    guided_choice: Option<&GuidedCatChoice>,
    tuple_cfg: Option<&CatTupleConfig>,
) -> Option<LeafExpertCandidate> {
    let n_leaf = node_indices.len();
    let mut best: Option<LeafExpertCandidate> = None;

    if smooth > 0.0 {
        if let Some(choice) = guided_choice {
            if let Some(cll) = eval_guided_cll_for_node(
                binned,
                gradients,
                hessians,
                node_indices,
                g_sum,
                h_sum,
                lambda_reg,
                gamma,
                min_child_weight,
                choice,
            ) {
                let smooth_eff = lookup_candidate_smooth(
                    &cll.bin_g,
                    &cll.bin_h,
                    g_sum,
                    h_sum,
                    lambda_reg,
                    min_child_weight,
                    smooth,
                    adaptive_smooth,
                );
                let df = lookup_effective_df(&cll.bin_h, min_child_weight, smooth_eff, lambda_reg);
                let score = expert_score_df(cll.gain, n_leaf, df, false);
                if score > best.as_ref().map_or(0.0, |b| b.score) {
                    best = Some(LeafExpertCandidate {
                        score,
                        kind: LeafExpertKind::Lookup(make_cll_lookup(
                            &cll,
                            leaf_value,
                            smooth_eff,
                            lambda_reg,
                            min_child_weight,
                        )),
                    });
                }
            }
        } else {
            if let Some(cll) = eval_cll_for_node(
                binned,
                gradients,
                hessians,
                node_indices,
                g_sum,
                h_sum,
                lambda_reg,
                gamma,
                min_child_weight,
            ) {
                let smooth_eff = lookup_candidate_smooth(
                    &cll.bin_g,
                    &cll.bin_h,
                    g_sum,
                    h_sum,
                    lambda_reg,
                    min_child_weight,
                    smooth,
                    adaptive_smooth,
                );
                let df = lookup_effective_df(&cll.bin_h, min_child_weight, smooth_eff, lambda_reg);
                let score = expert_score_df(cll.gain, n_leaf, df, false);
                if score > best.as_ref().map_or(0.0, |b| b.score) {
                    best = Some(LeafExpertCandidate {
                        score,
                        kind: LeafExpertKind::Lookup(make_cll_lookup(
                            &cll,
                            leaf_value,
                            smooth_eff,
                            lambda_reg,
                            min_child_weight,
                        )),
                    });
                }
            }

            if let Some(cfg) = tuple_cfg {
                if let Some(pair_cll) = eval_pair_cll_for_node(
                    binned,
                    gradients,
                    hessians,
                    node_indices,
                    g_sum,
                    h_sum,
                    lambda_reg,
                    gamma,
                    min_child_weight,
                ) {
                    let smooth_eff = lookup_candidate_smooth(
                        &pair_cll.bin_g,
                        &pair_cll.bin_h,
                        g_sum,
                        h_sum,
                        lambda_reg,
                        min_child_weight,
                        smooth,
                        adaptive_smooth,
                    );
                    let df = lookup_effective_df(
                        &pair_cll.bin_h,
                        min_child_weight,
                        smooth_eff,
                        lambda_reg,
                    );
                    let score = expert_score_df(pair_cll.gain, n_leaf, df, false);
                    if score > best.as_ref().map_or(0.0, |b| b.score) {
                        best = Some(LeafExpertCandidate {
                            score,
                            kind: LeafExpertKind::Lookup(make_cll_lookup(
                                &pair_cll,
                                leaf_value,
                                smooth_eff,
                                lambda_reg,
                                min_child_weight,
                            )),
                        });
                    }
                }

                if let Some(tuple) = eval_cat_tuple_lookup_for_node(
                    binned,
                    gradients,
                    hessians,
                    node_indices,
                    g_sum,
                    h_sum,
                    leaf_value,
                    lambda_reg,
                    gamma,
                    min_child_weight,
                    smooth,
                    cfg,
                ) {
                    let margin = cfg.gain_margin.max(0.0);
                    let incumbent = best.as_ref().map_or(0.0, |b| b.score);
                    if tuple.score > incumbent * (1.0 + margin) {
                        best = Some(tuple);
                    }
                }
            }
        }

        if let Some(nll) = eval_nll_for_node(
            binned,
            gradients,
            hessians,
            node_indices,
            g_sum,
            h_sum,
            lambda_reg,
            gamma,
            min_child_weight,
        ) {
            let df = lookup_effective_df(&nll.bin_h, min_child_weight, smooth, lambda_reg);
            let score = expert_score_df(nll.gain, n_leaf, df, true);
            if score > best.as_ref().map_or(0.0, |b| b.score) {
                best = Some(LeafExpertCandidate {
                    score,
                    kind: LeafExpertKind::Lookup(make_nll_lookup(
                        &nll,
                        leaf_value,
                        smooth,
                        lambda_reg,
                        min_child_weight,
                    )),
                });
            }
        }
    }

    if let Some(linear) = eval_best_linear_for_node(
        binned,
        gradients,
        hessians,
        node_indices,
        g_sum,
        h_sum,
        leaf_value,
        lambda_reg,
        min_child_weight,
    ) {
        if linear.gain > best.as_ref().map_or(0.0, |b| b.score) {
            best = Some(LeafExpertCandidate {
                score: linear.gain,
                kind: LeafExpertKind::Linear {
                    feats: linear.feats,
                    slopes: linear.slopes,
                    n_feats: linear.n_feats,
                    intercept: linear.intercept,
                },
            });
        }
    }

    if let Some(bilinear) = eval_best_bilinear_for_node(
        binned,
        gradients,
        hessians,
        node_indices,
        g_sum,
        h_sum,
        leaf_value,
        lambda_reg,
        min_child_weight,
    ) {
        if bilinear.score > best.as_ref().map_or(0.0, |b| b.score) {
            best = Some(bilinear);
        }
    }

    best.filter(|cand| cand.score > 0.0)
}

/// Pre-built histograms for ALL tree_features of a node (used by histogram subtraction trick).
/// Layout: g[feat_idx * max_bins + bin], h[feat_idx * max_bins + bin].
pub(super) struct NodeHists {
    g: Vec<f64>,
    h: Vec<f64>,
    g_miss: Vec<f64>,
    h_miss: Vec<f64>,
    max_bins: usize,
    n_features: usize,
}

impl NodeHists {
    pub(super) fn new(n_features: usize, max_bins: usize) -> Self {
        let total = n_features * max_bins;
        Self {
            g: vec![0.0; total],
            h: vec![0.0; total],
            g_miss: vec![0.0; n_features],
            h_miss: vec![0.0; n_features],
            max_bins,
            n_features,
        }
    }

    #[inline]
    pub(super) fn resize_for(&mut self, n_features: usize, max_bins: usize) {
        let total = n_features * max_bins;
        self.g.resize(total, 0.0);
        self.h.resize(total, 0.0);
        self.g_miss.resize(n_features, 0.0);
        self.h_miss.resize(n_features, 0.0);
        self.max_bins = max_bins;
        self.n_features = n_features;
    }
}

/// Small per-tree histogram arena. It avoids repeated Vec allocation in the
/// histogram-subtraction path while keeping NodeHists ownership simple for the
/// DFS stack.
pub(super) struct HistPool {
    n_features: usize,
    max_bins: usize,
    free: Vec<NodeHists>,
}

impl HistPool {
    pub(super) fn new(n_features: usize, max_bins: usize) -> Self {
        Self {
            n_features,
            max_bins,
            free: Vec::new(),
        }
    }

    #[inline]
    pub(super) fn take(&mut self) -> NodeHists {
        match self.free.pop() {
            Some(mut h) => {
                h.resize_for(self.n_features, self.max_bins);
                h
            }
            None => NodeHists::new(self.n_features, self.max_bins),
        }
    }

    #[inline]
    pub(super) fn recycle(&mut self, h: NodeHists) {
        if h.max_bins == self.max_bins && h.n_features == self.n_features {
            self.free.push(h);
        }
    }
}

/// Build histogram for ONE feature into caller-provided buffers.
/// Returns (g_miss, h_miss) for the missing-value bin.
#[inline]
pub(super) fn build_feature_hist(
    feat: usize,
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    node_marker: Option<&[u8]>,
    node_is_full: bool,
    node_g_sum: f64,
    node_h_sum: f64,
    g_hist: &mut [f64],
    h_hist: &mut [f64],
) -> (f64, f64) {
    let feat_n_bins = binned.n_bins(feat);
    for i in 0..feat_n_bins {
        g_hist[i] = 0.0;
        h_hist[i] = 0.0;
    }

    let col_bins = binned.col_bins(feat);
    let mut g_miss = 0.0f64;
    let mut h_miss = 0.0f64;

    let globally_has_missing = binned
        .feature_has_missing
        .get(feat)
        .copied()
        .unwrap_or(true);
    if globally_has_missing {
        let global_non_missing = binned
            .feature_non_missing_count
            .get(feat)
            .copied()
            .unwrap_or(binned.n_rows as u32) as usize;
        let sparse_is_cheaper = global_non_missing.saturating_mul(4)
            <= binned.n_rows.saturating_mul(3)
            && global_non_missing < node_indices.len().saturating_mul(2);
        if sparse_is_cheaper && (node_is_full || node_marker.is_some()) {
            let (rows, bins) = binned.non_missing_block(feat);
            let mut g_present = 0.0f64;
            let mut h_present = 0.0f64;
            if node_is_full {
                for (&row, &bin) in rows.iter().zip(bins.iter()) {
                    let row = row as usize;
                    let g = gradients[row];
                    let h = hessians[row];
                    g_hist[bin as usize] += g;
                    h_hist[bin as usize] += h;
                    g_present += g;
                    h_present += h;
                }
            } else if let Some(marker) = node_marker {
                for (&row, &bin) in rows.iter().zip(bins.iter()) {
                    let row = row as usize;
                    if marker[row] != 0 {
                        let g = gradients[row];
                        let h = hessians[row];
                        g_hist[bin as usize] += g;
                        h_hist[bin as usize] += h;
                        g_present += g;
                        h_present += h;
                    }
                }
            }
            return (node_g_sum - g_present, node_h_sum - h_present);
        }
        for &idx in node_indices {
            let row = idx as usize;
            let bin = col_bins[row];
            if bin == MISSING_BIN {
                g_miss += gradients[row];
                h_miss += hessians[row];
            } else {
                g_hist[bin as usize] += gradients[row];
                h_hist[bin as usize] += hessians[row];
            }
        }
    } else {
        for &idx in node_indices {
            let bin = col_bins[idx as usize] as usize;
            g_hist[bin] += gradients[idx as usize];
            h_hist[bin] += hessians[idx as usize];
        }
    }
    (g_miss, h_miss)
}

/// Build histograms for ALL tree_features into a NodeHists.
pub(super) fn build_node_hists(
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    tree_features: &[usize],
    hists: &mut NodeHists,
) {
    let max_bins = hists.max_bins;
    let node_len = node_indices.len();
    let mut sparse_candidate_count = 0usize;
    let mut sparse_candidate_entries = 0usize;
    for &feat in tree_features {
        let has_missing = binned
            .feature_has_missing
            .get(feat)
            .copied()
            .unwrap_or(false);
        if !has_missing {
            continue;
        }
        let global_non_missing = binned
            .feature_non_missing_count
            .get(feat)
            .copied()
            .unwrap_or(binned.n_rows as u32) as usize;
        if global_non_missing.saturating_mul(4) <= binned.n_rows.saturating_mul(3)
            && global_non_missing < node_len.saturating_mul(2)
        {
            sparse_candidate_count += 1;
            sparse_candidate_entries += global_non_missing;
        }
    }
    let node_is_full = node_len == binned.n_rows
        && node_indices
            .first()
            .zip(node_indices.last())
            .map(|(&first, &last)| first == 0 && last as usize + 1 == binned.n_rows)
            .unwrap_or(false);
    let use_sparse_marker = sparse_blocks_enabled()
        && !node_is_full
        && sparse_candidate_count > 0
        && node_len.saturating_mul(sparse_candidate_count)
            > binned.n_rows.saturating_add(sparse_candidate_entries);
    let need_node_sums = sparse_blocks_enabled()
        && sparse_candidate_count > 0
        && (node_is_full || use_sparse_marker);
    let (node_marker, node_g_sum, node_h_sum) = if use_sparse_marker {
        let mut marker = vec![0u8; binned.n_rows];
        let mut g_sum = 0.0f64;
        let mut h_sum = 0.0f64;
        for &idx in node_indices {
            let row = idx as usize;
            marker[row] = 1;
            g_sum += gradients[row];
            h_sum += hessians[row];
        }
        (Some(marker), g_sum, h_sum)
    } else if need_node_sums {
        let mut g_sum = 0.0f64;
        let mut h_sum = 0.0f64;
        for &idx in node_indices {
            let row = idx as usize;
            g_sum += gradients[row];
            h_sum += hessians[row];
        }
        (None, g_sum, h_sum)
    } else {
        (None, 0.0, 0.0)
    };
    let marker_ref = node_marker.as_deref();
    let work = tree_features.len().saturating_mul(node_indices.len());
    if work >= 16_384 && tree_features.len() >= 4 {
        hists
            .g
            .par_chunks_mut(max_bins)
            .zip(hists.h.par_chunks_mut(max_bins))
            .zip(hists.g_miss.par_iter_mut())
            .zip(hists.h_miss.par_iter_mut())
            .zip(tree_features.par_iter())
            .for_each(|((((g_slice, h_slice), gm_slot), hm_slot), &feat)| {
                let (gm, hm) = build_feature_hist(
                    feat,
                    binned,
                    gradients,
                    hessians,
                    node_indices,
                    marker_ref,
                    node_is_full,
                    node_g_sum,
                    node_h_sum,
                    g_slice,
                    h_slice,
                );
                *gm_slot = gm;
                *hm_slot = hm;
            });
    } else {
        for (feat_idx, &feat) in tree_features.iter().enumerate() {
            let offset = feat_idx * max_bins;
            let g_slice = &mut hists.g[offset..offset + max_bins];
            let h_slice = &mut hists.h[offset..offset + max_bins];
            let (gm, hm) = build_feature_hist(
                feat,
                binned,
                gradients,
                hessians,
                node_indices,
                marker_ref,
                node_is_full,
                node_g_sum,
                node_h_sum,
                g_slice,
                h_slice,
            );
            hists.g_miss[feat_idx] = gm;
            hists.h_miss[feat_idx] = hm;
        }
    }
}

/// Element-wise subtraction: result = parent - child.
pub(super) fn subtract_node_hists(parent: &NodeHists, child: &NodeHists, result: &mut NodeHists) {
    for i in 0..parent.g.len() {
        result.g[i] = parent.g[i] - child.g[i];
        result.h[i] = parent.h[i] - child.h[i];
    }
    for i in 0..parent.g_miss.len() {
        result.g_miss[i] = parent.g_miss[i] - child.g_miss[i];
        result.h_miss[i] = parent.h_miss[i] - child.h_miss[i];
    }
}

/// Deterministic noise in [-1, 1] for split perturbation.
#[inline]
pub(super) fn split_noise(seed: u64, feat: usize, key: usize) -> f64 {
    let mut h = seed
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(feat as u64);
    h = h.wrapping_mul(0x517CC1B727220A95).wrapping_add(key as u64);
    h ^= h >> 32;
    h = h.wrapping_mul(0xBF58476D1CE4E5B9);
    h ^= h >> 32;
    ((h as i64) as f64) / (i64::MAX as f64)
}

#[inline]
pub(super) fn evidence_adjusted_gain(
    binned: &BinnedData,
    gain: f64,
    left_h: f64,
    right_h: f64,
    parent_h: f64,
    lambda_reg: f64,
    n_cutpoints: usize,
) -> f64 {
    let strength = binned.split_pessimism;
    if strength <= 0.0 || !gain.is_finite() || gain <= 0.0 {
        return gain;
    }

    let search_width = ((binned.n_features.max(1) * n_cutpoints.max(1)).max(2) as f64).ln();
    let evidence_h = left_h.min(right_h).max(0.0);
    let reliability = evidence_h / (evidence_h + strength * search_width).max(1e-12);

    let l = (left_h + lambda_reg).max(1e-12);
    let r = (right_h + lambda_reg).max(1e-12);
    let p = (parent_h + lambda_reg).max(1e-12);
    let curvature_risk = (0.5 * (1.0 / l + 1.0 / r - 1.0 / p)).max(0.0);

    gain * reliability - strength * search_width * curvature_risk
}

#[inline]
fn missing_route_penalty(parent_h: f64, missing_h: f64) -> f64 {
    let observed_h = (parent_h - missing_h).max(0.0);
    if missing_h <= 1e-12 || observed_h <= 1e-12 {
        return 0.0;
    }
    0.5 * (1.0 + parent_h.max(0.0)).ln()
}

#[inline]
fn eval_numeric_interval_split_from_hist(
    binned: &BinnedData,
    feat: usize,
    g_hist: &[f64],
    h_hist: &[f64],
    g_miss: f64,
    h_miss: f64,
    g_sum: f64,
    h_sum: f64,
    lambda_reg: f64,
    l1_reg: f64,
    gamma: f64,
    min_h: f64,
    random_strength: f64,
    noise_seed: u64,
    gain_penalty: f64,
) -> SplitResult {
    let feat_n_bins = binned.n_bins(feat);
    if binned.is_categorical[feat] || feat_n_bins < 4 {
        return SplitResult::empty();
    }

    let g_nm = g_sum - g_miss;
    let h_nm = h_sum - h_miss;
    let node_ratio = if h_nm > 1e-12 { g_nm / h_nm } else { 0.0 };
    let start_bin = 1usize;
    let end_bin = feat_n_bins.saturating_sub(1);
    if end_bin <= start_bin {
        return SplitResult::empty();
    }

    let mut seeds: Vec<(usize, usize)> = Vec::with_capacity(2);

    let mut cur_z = 0.0f64;
    let mut cur_lo = start_bin;
    let mut best_z = f64::NEG_INFINITY;
    let mut best_pos = (start_bin, start_bin);
    for bin in start_bin..end_bin {
        let z = g_hist[bin] - node_ratio * h_hist[bin];
        if cur_z <= 0.0 {
            cur_z = z;
            cur_lo = bin;
        } else {
            cur_z += z;
        }
        if cur_z > best_z {
            best_z = cur_z;
            best_pos = (cur_lo, bin);
        }
    }
    if best_z > 0.0 && best_z.is_finite() {
        seeds.push(best_pos);
    }

    cur_z = 0.0;
    cur_lo = start_bin;
    let mut best_neg_z = f64::INFINITY;
    let mut best_neg = (start_bin, start_bin);
    for bin in start_bin..end_bin {
        let z = g_hist[bin] - node_ratio * h_hist[bin];
        if cur_z >= 0.0 {
            cur_z = z;
            cur_lo = bin;
        } else {
            cur_z += z;
        }
        if cur_z < best_neg_z {
            best_neg_z = cur_z;
            best_neg = (cur_lo, bin);
        }
    }
    if best_neg_z < 0.0 && best_neg_z.is_finite() && !seeds.contains(&best_neg) {
        seeds.push(best_neg);
    }
    if seeds.is_empty() {
        return SplitResult::empty();
    }

    let mut candidates: Vec<(usize, usize)> = Vec::with_capacity(96);
    let refine_radius = 4usize;
    let last_interval_bin = end_bin.saturating_sub(1);
    for (seed_lo, seed_hi) in seeds {
        let lo_min = seed_lo.saturating_sub(refine_radius).max(start_bin);
        let lo_max = seed_lo.saturating_add(refine_radius).min(last_interval_bin);
        let hi_min = seed_hi.saturating_sub(refine_radius).max(start_bin);
        let hi_max = seed_hi.saturating_add(refine_radius).min(last_interval_bin);
        if lo_min > lo_max || hi_min > hi_max {
            continue;
        }
        for lo in lo_min..=lo_max {
            let min_hi = hi_min.max(lo);
            if min_hi > hi_max {
                continue;
            }
            for hi in min_hi..=hi_max {
                let cand = (lo, hi);
                if !candidates.contains(&cand) {
                    candidates.push(cand);
                }
            }
        }
    }
    if candidates.is_empty() {
        return SplitResult::empty();
    }

    // Refine the raw-contrast interval seeds by scoring nearby endpoints with
    // the real guarded split objective. This keeps search local while reducing
    // Kadane's bias toward proxy-optimal, not gain-optimal, endpoints.
    let mut prefix_g = vec![0.0f64; feat_n_bins + 1];
    let mut prefix_h = vec![0.0f64; feat_n_bins + 1];
    for bin in 0..feat_n_bins {
        prefix_g[bin + 1] = prefix_g[bin] + g_hist[bin];
        prefix_h[bin + 1] = prefix_h[bin] + h_hist[bin];
    }

    let search_width = feat_n_bins.saturating_mul(candidates.len().max(2)).max(2);
    let min_branch_h = min_h.max(0.10 * h_sum.max(0.0));
    let min_flank_h = min_h.max(0.06 * h_nm.max(0.0));
    let mut best_gain = f64::NEG_INFINITY;
    let mut best_lo = 0usize;
    let mut best_hi = 0usize;
    let mut best_missing_left = false;

    for (lo, hi) in candidates {
        let ig = prefix_g[hi + 1] - prefix_g[lo];
        let ih = prefix_h[hi + 1] - prefix_h[lo];
        if ih <= 1e-12 {
            continue;
        }
        let lg_flank = prefix_g[lo];
        let lh_flank = prefix_h[lo];
        let rg_flank = prefix_g[feat_n_bins] - prefix_g[hi + 1];
        let rh_flank = prefix_h[feat_n_bins] - prefix_h[hi + 1];
        if lh_flank < min_flank_h || rh_flank < min_flank_h {
            continue;
        }

        // A bounded interval sends both outside flanks to the same child. Test
        // that merge with the same regularized leaf objective used by splits;
        // raw gradient ratios are too fragile near small flank support.
        let lz = lg_flank - node_ratio * lh_flank;
        let rz = rg_flank - node_ratio * rh_flank;
        if lz * rz <= 0.0 {
            continue;
        }
        let og = g_nm - ig;
        let oh = h_nm - ih;
        if oh <= 1e-12 {
            continue;
        }
        let interval_w = l1_leaf_value(ig, ih, lambda_reg, l1_reg);
        let outside_w = l1_leaf_value(og, oh, lambda_reg, l1_reg);
        let left_w = l1_leaf_value(lg_flank, lh_flank, lambda_reg, l1_reg);
        let right_w = l1_leaf_value(rg_flank, rh_flank, lambda_reg, l1_reg);
        let contrast = (interval_w - outside_w).abs();
        let flank_gap = (left_w - right_w).abs();
        if !(contrast > 1e-12 && flank_gap <= 0.7 * contrast + 1e-12) {
            continue;
        }
        let interval_score =
            l1_gain_score(ig, ih, lambda_reg, l1_reg) + l1_gain_score(og, oh, lambda_reg, l1_reg);
        let three_way_score = l1_gain_score(lg_flank, lh_flank, lambda_reg, l1_reg)
            + l1_gain_score(ig, ih, lambda_reg, l1_reg)
            + l1_gain_score(rg_flank, rh_flank, lambda_reg, l1_reg);
        let tie_loss = (three_way_score - interval_score).max(0.0);
        let parent_score = l1_gain_score(g_nm, h_nm, lambda_reg, l1_reg);
        let observed_interval_gain = (interval_score - parent_score).max(0.0);
        if tie_loss > 0.2 * observed_interval_gain + 1e-12 {
            continue;
        }

        for missing_left in [false, true] {
            let (lg, lh, rg, rh) = if missing_left {
                (ig + g_miss, ih + h_miss, og, oh)
            } else {
                (ig, ih, og + g_miss, oh + h_miss)
            };
            if lh < min_branch_h || rh < min_branch_h {
                continue;
            }
            let mut gain = 0.5
                * (l1_gain_score(lg, lh, lambda_reg, l1_reg)
                    + l1_gain_score(rg, rh, lambda_reg, l1_reg)
                    - l1_gain_score(g_sum, h_sum, lambda_reg, l1_reg))
                - gamma;
            if gain_penalty > 0.0 {
                gain -= gain_penalty
                    * 0.5
                    * (1.0 / (lh + lambda_reg) + 1.0 / (rh + lambda_reg)
                        - 1.0 / (h_sum + lambda_reg));
            }
            gain = evidence_adjusted_gain(binned, gain, lh, rh, h_sum, lambda_reg, search_width);
            gain -= missing_route_penalty(h_sum, h_miss);
            if random_strength > 0.0 && gain > 0.0 {
                let key = (lo * feat_n_bins + hi) * 2 + usize::from(missing_left);
                gain *= 1.0 + random_strength * split_noise(noise_seed, feat, key);
            }
            if gain.is_finite() && gain > best_gain {
                best_gain = gain;
                best_lo = lo;
                best_hi = hi;
                best_missing_left = missing_left;
            }
        }
    }

    if !best_gain.is_finite() {
        return SplitResult::empty();
    }

    let mut mask: CatBitmask = Vec::new();
    for bin in best_lo..=best_hi {
        bitmask_set(&mut mask, bin);
    }
    SplitResult::axis(best_gain, feat, 0, best_missing_left, true, mask)
}

/// Scan a pre-built histogram for the best split point of a single feature.
/// Same categorical + numerical logic as eval_feature_split, but reads from buffers.
#[inline]
pub(super) fn scan_feature_hist(
    feat: usize,
    binned: &BinnedData,
    g_hist: &[f64],
    h_hist: &[f64],
    g_miss: f64,
    h_miss: f64,
    g_sum: f64,
    h_sum: f64,
    lambda_reg: f64,
    l1_reg: f64,
    gamma: f64,
    min_h: f64,
    random_strength: f64,
    noise_seed: u64,
    cat_smooth: f64,
    mono_cstr: i8,
    gain_penalty: f64,
    interval_splits: bool,
) -> SplitResult {
    let feat_n_bins = binned.n_bins(feat);
    if feat_n_bins <= 1 {
        return SplitResult::axis(f64::NEG_INFINITY, feat, 0, true, false, Vec::new());
    }

    let g_nm = g_sum - g_miss;
    let h_nm = h_sum - h_miss;

    let mut best_gain = f64::NEG_INFINITY;
    let mut best_bin = 0usize;
    let mut best_missing_left = true;
    let mut best_is_cat = false;
    let mut best_cat_split_idx: usize = 0;
    let missing_penalty = missing_route_penalty(h_sum, h_miss);

    if binned.is_categorical[feat] {
        let mut cat_bins: Vec<(usize, f64, f64)> = Vec::new();
        for bin in 0..feat_n_bins {
            if h_hist[bin] > 0.0 {
                cat_bins.push((bin, g_hist[bin], h_hist[bin]));
            }
        }
        if cat_bins.len() > 1 {
            let node_ratio = if h_nm > 1e-10 { g_nm / h_nm } else { 0.0 };
            cat_bins.sort_by(|a, b| {
                let ra = (a.1 + cat_smooth * node_ratio) / (a.2 + cat_smooth);
                let rb = (b.1 + cat_smooth * node_ratio) / (b.2 + cat_smooth);
                ra.partial_cmp(&rb).unwrap_or(Ordering::Equal)
            });

            let mut cum_g = 0.0f64;
            let mut cum_h = 0.0f64;
            for i in 0..cat_bins.len() - 1 {
                cum_g += cat_bins[i].1;
                cum_h += cat_bins[i].2;
                let other_g = g_nm - cum_g;
                let other_h = h_nm - cum_h;

                let lg = cum_g + g_miss;
                let lh = cum_h + h_miss;
                if lh >= min_h && other_h >= min_h {
                    let mut gain = 0.5
                        * (l1_gain_score(lg, lh, lambda_reg, l1_reg)
                            + l1_gain_score(other_g, other_h, lambda_reg, l1_reg)
                            - l1_gain_score(g_sum, h_sum, lambda_reg, l1_reg))
                        - gamma;
                    if gain_penalty > 0.0 {
                        gain -= gain_penalty
                            * 0.5
                            * (1.0 / (lh + lambda_reg) + 1.0 / (other_h + lambda_reg)
                                - 1.0 / (h_sum + lambda_reg));
                    }
                    gain = evidence_adjusted_gain(
                        binned,
                        gain,
                        lh,
                        other_h,
                        h_sum,
                        lambda_reg,
                        cat_bins.len().saturating_sub(1),
                    );
                    gain -= missing_penalty;
                    if random_strength > 0.0 && gain > 0.0 {
                        gain *= 1.0 + random_strength * split_noise(noise_seed, feat, i * 2);
                    }
                    if gain.is_finite() && gain > best_gain {
                        best_gain = gain;
                        best_bin = 0;
                        best_missing_left = true;
                        best_is_cat = true;
                        best_cat_split_idx = i;
                    }
                }

                if cum_h >= min_h && (other_h + h_miss) >= min_h {
                    let rg = other_g + g_miss;
                    let rh = other_h + h_miss;
                    let mut gain = 0.5
                        * (l1_gain_score(cum_g, cum_h, lambda_reg, l1_reg)
                            + l1_gain_score(rg, rh, lambda_reg, l1_reg)
                            - l1_gain_score(g_sum, h_sum, lambda_reg, l1_reg))
                        - gamma;
                    if gain_penalty > 0.0 {
                        gain -= gain_penalty
                            * 0.5
                            * (1.0 / (cum_h + lambda_reg) + 1.0 / (rh + lambda_reg)
                                - 1.0 / (h_sum + lambda_reg));
                    }
                    gain = evidence_adjusted_gain(
                        binned,
                        gain,
                        cum_h,
                        rh,
                        h_sum,
                        lambda_reg,
                        cat_bins.len().saturating_sub(1),
                    );
                    gain -= missing_penalty;
                    if random_strength > 0.0 && gain > 0.0 {
                        gain *= 1.0 + random_strength * split_noise(noise_seed, feat, i * 2 + 1);
                    }
                    if gain.is_finite() && gain > best_gain {
                        best_gain = gain;
                        best_bin = 0;
                        best_missing_left = false;
                        best_is_cat = true;
                        best_cat_split_idx = i;
                    }
                }
            }
        }
        if best_is_cat {
            let mut mask: CatBitmask = Vec::new();
            for j in 0..=best_cat_split_idx {
                bitmask_set(&mut mask, cat_bins[j].0);
            }
            return SplitResult::axis(best_gain, feat, best_bin, best_missing_left, true, mask);
        }
    } else {
        let mut cum_g = 0.0f64;
        let mut cum_h = 0.0f64;
        for bin in 0..feat_n_bins - 1 {
            cum_g += g_hist[bin];
            cum_h += h_hist[bin];
            let other_g = g_nm - cum_g;
            let other_h = h_nm - cum_h;

            let lg = cum_g + g_miss;
            let lh = cum_h + h_miss;
            if lh >= min_h && other_h >= min_h {
                let mono_ok = if mono_cstr == 0 {
                    true
                } else {
                    let lv = -lg / (lh + lambda_reg);
                    let rv = -other_g / (other_h + lambda_reg);
                    if mono_cstr > 0 {
                        lv <= rv
                    } else {
                        lv >= rv
                    }
                };
                if mono_ok {
                    let mut gain = 0.5
                        * (l1_gain_score(lg, lh, lambda_reg, l1_reg)
                            + l1_gain_score(other_g, other_h, lambda_reg, l1_reg)
                            - l1_gain_score(g_sum, h_sum, lambda_reg, l1_reg))
                        - gamma;
                    if gain_penalty > 0.0 {
                        gain -= gain_penalty
                            * 0.5
                            * (1.0 / (lh + lambda_reg) + 1.0 / (other_h + lambda_reg)
                                - 1.0 / (h_sum + lambda_reg));
                    }
                    gain = evidence_adjusted_gain(
                        binned,
                        gain,
                        lh,
                        other_h,
                        h_sum,
                        lambda_reg,
                        feat_n_bins.saturating_sub(1),
                    );
                    gain -= missing_penalty;
                    if random_strength > 0.0 && gain > 0.0 {
                        gain *= 1.0 + random_strength * split_noise(noise_seed, feat, bin * 2);
                    }
                    if gain.is_finite() && gain > best_gain {
                        best_gain = gain;
                        best_bin = bin;
                        best_missing_left = true;
                    }
                }
            }

            let rg = other_g + g_miss;
            let rh = other_h + h_miss;
            if cum_h >= min_h && rh >= min_h {
                let mono_ok = if mono_cstr == 0 {
                    true
                } else {
                    let lv = -cum_g / (cum_h + lambda_reg);
                    let rv = -rg / (rh + lambda_reg);
                    if mono_cstr > 0 {
                        lv <= rv
                    } else {
                        lv >= rv
                    }
                };
                if mono_ok {
                    let mut gain = 0.5
                        * (l1_gain_score(cum_g, cum_h, lambda_reg, l1_reg)
                            + l1_gain_score(rg, rh, lambda_reg, l1_reg)
                            - l1_gain_score(g_sum, h_sum, lambda_reg, l1_reg))
                        - gamma;
                    if gain_penalty > 0.0 {
                        gain -= gain_penalty
                            * 0.5
                            * (1.0 / (cum_h + lambda_reg) + 1.0 / (rh + lambda_reg)
                                - 1.0 / (h_sum + lambda_reg));
                    }
                    gain = evidence_adjusted_gain(
                        binned,
                        gain,
                        cum_h,
                        rh,
                        h_sum,
                        lambda_reg,
                        feat_n_bins.saturating_sub(1),
                    );
                    gain -= missing_penalty;
                    if random_strength > 0.0 && gain > 0.0 {
                        gain *= 1.0 + random_strength * split_noise(noise_seed, feat, bin * 2 + 1);
                    }
                    if gain.is_finite() && gain > best_gain {
                        best_gain = gain;
                        best_bin = bin;
                        best_missing_left = false;
                    }
                }
            }
        }
    }

    let axis = SplitResult::axis(
        best_gain,
        feat,
        best_bin,
        best_missing_left,
        false,
        Vec::new(),
    );
    if interval_splits && mono_cstr == 0 && !binned.is_categorical[feat] {
        let interval = eval_numeric_interval_split_from_hist(
            binned,
            feat,
            g_hist,
            h_hist,
            g_miss,
            h_miss,
            g_sum,
            h_sum,
            lambda_reg,
            l1_reg,
            gamma,
            min_h,
            random_strength,
            noise_seed,
            gain_penalty,
        );
        let required_gain = axis.gain.max(0.0) * 1.5 + 1e-12;
        if interval.gain > required_gain {
            return interval;
        }
    }
    axis
}

/// Evaluate one feature and return its best split (if any).
#[inline]
pub(super) fn eval_feature_split(
    feat: usize,
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    g_sum: f64,
    h_sum: f64,
    lambda_reg: f64,
    l1_reg: f64,
    gamma: f64,
    min_h: f64,
    random_strength: f64,
    noise_seed: u64,
    g_hist: &mut [f64],
    h_hist: &mut [f64],
    cat_smooth: f64,
    mono_cstr: i8,
    gain_penalty: f64,
    interval_splits: bool,
) -> SplitResult {
    let feat_n_bins = binned.n_bins(feat);
    if feat_n_bins <= 1 {
        return SplitResult::axis(f64::NEG_INFINITY, feat, 0, true, false, Vec::new());
    }

    let col_bins = binned.col_bins(feat);
    for i in 0..feat_n_bins {
        g_hist[i] = 0.0;
        h_hist[i] = 0.0;
    }

    let mut g_miss = 0.0f64;
    let mut h_miss = 0.0f64;

    // Fast path: skip MISSING_BIN branch when no missing values exist for this feature.
    // This eliminates a branch per sample in the histogram loop (common case).
    let has_missing = node_indices
        .iter()
        .any(|&idx| col_bins[idx as usize] == MISSING_BIN);
    if has_missing {
        for &idx in node_indices {
            let bin = col_bins[idx as usize];
            if bin == MISSING_BIN {
                g_miss += gradients[idx as usize];
                h_miss += hessians[idx as usize];
            } else {
                g_hist[bin as usize] += gradients[idx as usize];
                h_hist[bin as usize] += hessians[idx as usize];
            }
        }
    } else {
        for &idx in node_indices {
            let bin = col_bins[idx as usize] as usize;
            g_hist[bin] += gradients[idx as usize];
            h_hist[bin] += hessians[idx as usize];
        }
    }

    let g_nm = g_sum - g_miss;
    let h_nm = h_sum - h_miss;

    let mut best_gain = f64::NEG_INFINITY;
    let mut best_bin = 0usize;
    let mut best_missing_left = true;
    let mut best_is_cat = false;
    // For categorical: track the best split index into sorted cat_bins to build mask once at the end
    let mut best_cat_split_idx: usize = 0;
    let missing_penalty = missing_route_penalty(h_sum, h_miss);

    if binned.is_categorical[feat] {
        let mut cat_bins: Vec<(usize, f64, f64)> = Vec::new();
        for bin in 0..feat_n_bins {
            if h_hist[bin] > 0.0 {
                cat_bins.push((bin, g_hist[bin], h_hist[bin]));
            }
        }
        if cat_bins.len() > 1 {
            // Smooth g/h ratio toward node mean so rare categories don't dominate sort
            let node_ratio = if h_nm > 1e-10 { g_nm / h_nm } else { 0.0 };
            cat_bins.sort_by(|a, b| {
                let ra = (a.1 + cat_smooth * node_ratio) / (a.2 + cat_smooth);
                let rb = (b.1 + cat_smooth * node_ratio) / (b.2 + cat_smooth);
                ra.partial_cmp(&rb).unwrap_or(Ordering::Equal)
            });

            // Phase 1: Sort-and-scan to find best contiguous partition
            let mut cum_g = 0.0f64;
            let mut cum_h = 0.0f64;
            for i in 0..cat_bins.len() - 1 {
                cum_g += cat_bins[i].1;
                cum_h += cat_bins[i].2;
                let other_g = g_nm - cum_g;
                let other_h = h_nm - cum_h;

                let lg = cum_g + g_miss;
                let lh = cum_h + h_miss;
                if lh >= min_h && other_h >= min_h {
                    let mut gain = 0.5
                        * (l1_gain_score(lg, lh, lambda_reg, l1_reg)
                            + l1_gain_score(other_g, other_h, lambda_reg, l1_reg)
                            - l1_gain_score(g_sum, h_sum, lambda_reg, l1_reg))
                        - gamma;
                    if gain_penalty > 0.0 {
                        gain -= gain_penalty
                            * 0.5
                            * (1.0 / (lh + lambda_reg) + 1.0 / (other_h + lambda_reg)
                                - 1.0 / (h_sum + lambda_reg));
                    }
                    gain = evidence_adjusted_gain(
                        binned,
                        gain,
                        lh,
                        other_h,
                        h_sum,
                        lambda_reg,
                        cat_bins.len().saturating_sub(1),
                    );
                    gain -= missing_penalty;
                    if random_strength > 0.0 && gain > 0.0 {
                        gain *= 1.0 + random_strength * split_noise(noise_seed, feat, i * 2);
                    }
                    if gain.is_finite() && gain > best_gain {
                        best_gain = gain;
                        best_bin = 0;
                        best_missing_left = true;
                        best_is_cat = true;
                        best_cat_split_idx = i;
                    }
                }

                if cum_h >= min_h && (other_h + h_miss) >= min_h {
                    let rg = other_g + g_miss;
                    let rh = other_h + h_miss;
                    let mut gain = 0.5
                        * (l1_gain_score(cum_g, cum_h, lambda_reg, l1_reg)
                            + l1_gain_score(rg, rh, lambda_reg, l1_reg)
                            - l1_gain_score(g_sum, h_sum, lambda_reg, l1_reg))
                        - gamma;
                    if gain_penalty > 0.0 {
                        gain -= gain_penalty
                            * 0.5
                            * (1.0 / (cum_h + lambda_reg) + 1.0 / (rh + lambda_reg)
                                - 1.0 / (h_sum + lambda_reg));
                    }
                    gain = evidence_adjusted_gain(
                        binned,
                        gain,
                        cum_h,
                        rh,
                        h_sum,
                        lambda_reg,
                        cat_bins.len().saturating_sub(1),
                    );
                    gain -= missing_penalty;
                    if random_strength > 0.0 && gain > 0.0 {
                        gain *= 1.0 + random_strength * split_noise(noise_seed, feat, i * 2 + 1);
                    }
                    if gain.is_finite() && gain > best_gain {
                        best_gain = gain;
                        best_bin = 0;
                        best_missing_left = false;
                        best_is_cat = true;
                        best_cat_split_idx = i;
                    }
                }
            }
        }
        // Build the cat_mask once for the best split
        if best_is_cat {
            let mut mask: CatBitmask = Vec::new();
            for j in 0..=best_cat_split_idx {
                bitmask_set(&mut mask, cat_bins[j].0);
            }
            return SplitResult::axis(best_gain, feat, best_bin, best_missing_left, true, mask);
        }
    } else {
        let mut cum_g = 0.0f64;
        let mut cum_h = 0.0f64;
        for bin in 0..feat_n_bins - 1 {
            cum_g += g_hist[bin];
            cum_h += h_hist[bin];
            let other_g = g_nm - cum_g;
            let other_h = h_nm - cum_h;

            let lg = cum_g + g_miss;
            let lh = cum_h + h_miss;
            if lh >= min_h && other_h >= min_h {
                // Monotone constraint check: left = lower bins (miss goes left), right = higher bins
                let mono_ok = if mono_cstr == 0 {
                    true
                } else {
                    let lv = -lg / (lh + lambda_reg);
                    let rv = -other_g / (other_h + lambda_reg);
                    if mono_cstr > 0 {
                        lv <= rv
                    } else {
                        lv >= rv
                    }
                };
                if mono_ok {
                    let mut gain = 0.5
                        * (l1_gain_score(lg, lh, lambda_reg, l1_reg)
                            + l1_gain_score(other_g, other_h, lambda_reg, l1_reg)
                            - l1_gain_score(g_sum, h_sum, lambda_reg, l1_reg))
                        - gamma;
                    if gain_penalty > 0.0 {
                        gain -= gain_penalty
                            * 0.5
                            * (1.0 / (lh + lambda_reg) + 1.0 / (other_h + lambda_reg)
                                - 1.0 / (h_sum + lambda_reg));
                    }
                    gain = evidence_adjusted_gain(
                        binned,
                        gain,
                        lh,
                        other_h,
                        h_sum,
                        lambda_reg,
                        feat_n_bins.saturating_sub(1),
                    );
                    gain -= missing_penalty;
                    if random_strength > 0.0 && gain > 0.0 {
                        gain *= 1.0 + random_strength * split_noise(noise_seed, feat, bin * 2);
                    }
                    if gain.is_finite() && gain > best_gain {
                        best_gain = gain;
                        best_bin = bin;
                        best_missing_left = true;
                    }
                }
            }

            let rg = other_g + g_miss;
            let rh = other_h + h_miss;
            if cum_h >= min_h && rh >= min_h {
                // Monotone constraint check: left = lower bins (miss goes right), right = higher bins + miss
                let mono_ok = if mono_cstr == 0 {
                    true
                } else {
                    let lv = -cum_g / (cum_h + lambda_reg);
                    let rv = -rg / (rh + lambda_reg);
                    if mono_cstr > 0 {
                        lv <= rv
                    } else {
                        lv >= rv
                    }
                };
                if mono_ok {
                    let mut gain = 0.5
                        * (l1_gain_score(cum_g, cum_h, lambda_reg, l1_reg)
                            + l1_gain_score(rg, rh, lambda_reg, l1_reg)
                            - l1_gain_score(g_sum, h_sum, lambda_reg, l1_reg))
                        - gamma;
                    if gain_penalty > 0.0 {
                        gain -= gain_penalty
                            * 0.5
                            * (1.0 / (cum_h + lambda_reg) + 1.0 / (rh + lambda_reg)
                                - 1.0 / (h_sum + lambda_reg));
                    }
                    gain = evidence_adjusted_gain(
                        binned,
                        gain,
                        cum_h,
                        rh,
                        h_sum,
                        lambda_reg,
                        feat_n_bins.saturating_sub(1),
                    );
                    gain -= missing_penalty;
                    if random_strength > 0.0 && gain > 0.0 {
                        gain *= 1.0 + random_strength * split_noise(noise_seed, feat, bin * 2 + 1);
                    }
                    if gain.is_finite() && gain > best_gain {
                        best_gain = gain;
                        best_bin = bin;
                        best_missing_left = false;
                    }
                }
            }
        }
    }

    let axis = SplitResult::axis(
        best_gain,
        feat,
        best_bin,
        best_missing_left,
        false,
        Vec::new(),
    );
    if interval_splits && mono_cstr == 0 && !binned.is_categorical[feat] {
        let interval = eval_numeric_interval_split_from_hist(
            binned,
            feat,
            g_hist,
            h_hist,
            g_miss,
            h_miss,
            g_sum,
            h_sum,
            lambda_reg,
            l1_reg,
            gamma,
            min_h,
            random_strength,
            noise_seed,
            gain_penalty,
        );
        let required_gain = axis.gain.max(0.0) * 1.5 + 1e-12;
        if interval.gain > required_gain {
            return interval;
        }
    }
    axis
}

#[inline]
pub(super) fn multiclass_cat_sort_direction(
    g_sums: &[f64],
    h_sums: &[f64],
    lambda_reg: f64,
) -> Vec<f64> {
    let n_classes = g_sums.len();
    let mut dir = vec![0.0f64; n_classes];
    if n_classes == 0 {
        return dir;
    }
    if n_classes == 1 {
        dir[0] = 1.0;
        return dir;
    }

    for k in 0..n_classes {
        dir[k] = -g_sums[k] / (h_sums[k] + lambda_reg).max(1e-12);
    }
    let mean = dir.iter().sum::<f64>() / n_classes as f64;
    for v in dir.iter_mut() {
        *v -= mean;
    }

    let mut norm = dir.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm <= 1e-12 {
        let anchor = g_sums
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap_or(Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        let other = -1.0 / (n_classes as f64 - 1.0);
        for (k, v) in dir.iter_mut().enumerate() {
            *v = if k == anchor { 1.0 } else { other };
        }
        norm = dir.iter().map(|v| v * v).sum::<f64>().sqrt();
    }

    if norm > 1e-12 {
        for v in dir.iter_mut() {
            *v /= norm;
        }
    }
    dir
}

#[inline]
pub(super) fn multiclass_cat_sort_direction_dense(
    g_sums: &[f64],
    p_sums: &[f64],
    pp_sums: &[f64],
    lambda_reg: f64,
) -> Vec<f64> {
    let n_classes = g_sums.len();
    if n_classes <= 1 || p_sums.len() < n_classes || pp_sums.len() < n_classes * n_classes {
        return multiclass_cat_sort_direction(g_sums, p_sums, lambda_reg);
    }

    let mut a = vec![0.0f64; n_classes * n_classes];
    let mut rhs = vec![0.0f64; n_classes];
    for i in 0..n_classes {
        rhs[i] = -g_sums[i];
        let row_base = i * n_classes;
        for j in 0..n_classes {
            let mut v = -pp_sums[row_base + j];
            if i == j {
                v += p_sums[i] + lambda_reg;
            }
            a[row_base + j] = v;
        }
    }

    let mut dir = solve_spd_local(n_classes, &a, &rhs);
    let mean = dir.iter().sum::<f64>() / n_classes as f64;
    for v in dir.iter_mut() {
        *v -= mean;
    }
    let norm = dir.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm <= 1e-12 {
        return multiclass_cat_sort_direction(g_sums, p_sums, lambda_reg);
    }
    for v in dir.iter_mut() {
        *v /= norm;
    }
    dir
}

#[inline]
pub(super) fn push_normalized_contrast(vectors: &mut Vec<Vec<f64>>, mut v: Vec<f64>) {
    if v.is_empty() {
        return;
    }
    let n = v.len();
    let mean = v.iter().sum::<f64>() / n as f64;
    for x in v.iter_mut() {
        *x -= mean;
    }
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm <= 1e-12 || !norm.is_finite() {
        return;
    }
    for x in v.iter_mut() {
        *x /= norm;
    }
    let is_dup = vectors.iter().any(|existing| {
        existing.len() == n
            && existing
                .iter()
                .zip(v.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                < 1e-8
    });
    if !is_dup {
        vectors.push(v);
    }
}

#[inline]
pub(super) fn multiclass_cat_contrast_vectors(
    cat_sort_dir: &[f64],
    g_sums: &[f64],
) -> Vec<Vec<f64>> {
    let n_classes = cat_sort_dir.len().max(g_sums.len());
    if n_classes <= 1 {
        return Vec::new();
    }

    let mut vectors: Vec<Vec<f64>> = Vec::new();
    let mut base = vec![0.0f64; n_classes];
    for k in 0..n_classes.min(cat_sort_dir.len()) {
        base[k] = cat_sort_dir[k];
    }
    push_normalized_contrast(&mut vectors, base.clone());

    let mut residual = vec![0.0f64; n_classes];
    for k in 0..n_classes.min(g_sums.len()) {
        residual[k] = -g_sums[k];
    }
    push_normalized_contrast(&mut vectors, residual.clone());

    let mut by_residual: Vec<usize> = (0..n_classes).collect();
    by_residual.sort_unstable_by(|&a, &b| {
        g_sums[b]
            .abs()
            .partial_cmp(&g_sums[a].abs())
            .unwrap_or(Ordering::Equal)
    });

    // One-vs-rest contrasts: isolate all categories whose Newton update strongly
    // favors one class, regardless of their ordering under the parent scalar direction.
    for cls in 0..n_classes {
        let mut v = vec![-1.0 / (n_classes - 1) as f64; n_classes];
        v[cls] = 1.0;
        push_normalized_contrast(&mut vectors, v);
    }

    // Pairwise class contrasts. For small K this is still cheap and covers
    // low-frequency classes that may not dominate the parent residual but still
    // need their own categorical partition.
    let top = by_residual
        .len()
        .min(if n_classes <= 8 { n_classes } else { 5 });
    for i in 0..top {
        for j in (i + 1)..top {
            let a = by_residual[i];
            let b = by_residual[j];
            let mut v = vec![0.0f64; n_classes];
            v[a] = 1.0;
            v[b] = -1.0;
            push_normalized_contrast(&mut vectors, v);
        }
    }

    vectors
}

#[inline]
pub(super) fn sort_multiclass_cat_bins_by_contrast(
    ordered_bins: &mut [usize],
    bin_updates: &[f64],
    n_classes: usize,
    contrast: &[f64],
    scalar_scores: &[f64],
) {
    ordered_bins.sort_by(|&a, &b| {
        let a_base = a * n_classes;
        let b_base = b * n_classes;
        let mut av = 0.0f64;
        let mut bv = 0.0f64;
        for k in 0..n_classes.min(contrast.len()) {
            av += bin_updates[a_base + k] * contrast[k];
            bv += bin_updates[b_base + k] * contrast[k];
        }
        av.partial_cmp(&bv)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                scalar_scores[a]
                    .partial_cmp(&scalar_scores[b])
                    .unwrap_or(Ordering::Equal)
            })
    });
}

#[inline]
pub(super) fn solve_spd_local(n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut l = vec![0.0f64; n * n];
    for j in 0..n {
        let mut sum = 0.0f64;
        for k in 0..j {
            sum += l[j * n + k] * l[j * n + k];
        }
        let diag = a[j * n + j] - sum;
        if diag <= 1e-30 {
            return vec![0.0; n];
        }
        l[j * n + j] = diag.sqrt();
        for i in (j + 1)..n {
            let mut sum2 = 0.0f64;
            for k in 0..j {
                sum2 += l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = (a[i * n + j] - sum2) / l[j * n + j];
        }
    }

    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut sum = 0.0f64;
        for k in 0..i {
            sum += l[i * n + k] * y[k];
        }
        y[i] = (b[i] - sum) / l[i * n + i];
    }

    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = 0.0f64;
        for k in (i + 1)..n {
            sum += l[k * n + i] * x[k];
        }
        x[i] = (y[i] - sum) / l[i * n + i];
    }
    x
}

#[inline]
pub(super) fn dense_multiclass_gain(
    g: &[f64],
    p_sum: &[f64],
    pp_sum: &[f64],
    lambda_reg: f64,
    mat: &mut [f64],
    rhs: &mut [f64],
) -> f64 {
    let n_classes = g.len();
    if n_classes == 0
        || p_sum.len() < n_classes
        || pp_sum.len() < n_classes * n_classes
        || mat.len() < n_classes * n_classes
        || rhs.len() < n_classes
    {
        return 0.0;
    }

    for a in 0..n_classes {
        rhs[a] = g[a];
        let row_base = a * n_classes;
        for b in 0..n_classes {
            let mut v = -pp_sum[row_base + b];
            if a == b {
                v += p_sum[a] + lambda_reg;
            }
            mat[row_base + b] = v;
        }
    }

    let sol = solve_spd_local(n_classes, &mat[..n_classes * n_classes], &rhs[..n_classes]);
    let mut gain = 0.0f64;
    for k in 0..n_classes {
        gain += g[k] * sol[k];
    }
    if gain.is_finite() && gain > 0.0 {
        gain
    } else {
        0.0
    }
}

/// Multi-output split evaluation: sums gains across K classes for a single feature.
#[inline]
pub(super) fn eval_feature_split_multi(
    feat: usize,
    binned: &BinnedData,
    all_gradients: &[f64],
    all_hessians: &[f64],
    all_probs: &[f64],
    n_classes: usize,
    n_rows: usize,
    node_indices: &[u32],
    g_sums: &[f64],
    h_sums: &[f64],
    lambda_reg: f64,
    gamma: f64,
    min_h: f64,
    random_strength: f64,
    noise_seed: u64,
    g_hists: &mut [f64],
    h_hists: &mut [f64],
    g_miss: &mut [f64],
    h_miss: &mut [f64],
    p_hists: &mut [f64],
    pp_hists: &mut [f64],
    p_miss: &mut [f64],
    pp_miss: &mut [f64],
    cat_sort_dir: &[f64],
    cat_smooth: f64,
    gain_penalty: f64,
    coupled_split_gain: bool,
    parent_p_sums: &[f64],
    parent_pp_sums: &[f64],
    parent_dense_gain: f64,
) -> SplitResult {
    let feat_n_bins = binned.n_bins(feat);
    if feat_n_bins <= 1 {
        return SplitResult::axis(f64::NEG_INFINITY, feat, 0, true, false, Vec::new());
    }

    let use_coupled_gain = coupled_split_gain
        && n_classes >= 3
        && all_probs.len() >= n_rows * n_classes
        && parent_p_sums.len() >= n_classes
        && parent_pp_sums.len() >= n_classes * n_classes;
    let max_bins = g_hists.len() / n_classes;
    let col_bins = binned.col_bins(feat);

    for k in 0..n_classes {
        let base = k * max_bins;
        for i in 0..feat_n_bins {
            g_hists[base + i] = 0.0;
            h_hists[base + i] = 0.0;
        }
        g_miss[k] = 0.0;
        h_miss[k] = 0.0;
        if use_coupled_gain {
            p_miss[k] = 0.0;
        }
    }
    if use_coupled_gain {
        for bin in 0..feat_n_bins {
            let p_base = bin * n_classes;
            for k in 0..n_classes {
                p_hists[p_base + k] = 0.0;
            }
            let pp_base = bin * n_classes * n_classes;
            for kk in 0..(n_classes * n_classes) {
                pp_hists[pp_base + kk] = 0.0;
            }
        }
        for kk in 0..(n_classes * n_classes) {
            pp_miss[kk] = 0.0;
        }
    }

    let has_missing = node_indices
        .iter()
        .any(|&idx| col_bins[idx as usize] == MISSING_BIN);
    if has_missing {
        for &idx in node_indices {
            let bin = col_bins[idx as usize];
            let i = idx as usize;
            if bin == MISSING_BIN {
                let prob_base = i * n_classes;
                for k in 0..n_classes {
                    g_miss[k] += all_gradients[k * n_rows + i];
                    h_miss[k] += all_hessians[k * n_rows + i];
                    if use_coupled_gain {
                        p_miss[k] += all_probs[prob_base + k];
                    }
                }
                if use_coupled_gain {
                    for a in 0..n_classes {
                        let pa = all_probs[prob_base + a];
                        let row_base = a * n_classes;
                        for b in 0..n_classes {
                            pp_miss[row_base + b] += pa * all_probs[prob_base + b];
                        }
                    }
                }
            } else {
                let b = bin as usize;
                let prob_base = i * n_classes;
                for k in 0..n_classes {
                    g_hists[k * max_bins + b] += all_gradients[k * n_rows + i];
                    h_hists[k * max_bins + b] += all_hessians[k * n_rows + i];
                    if use_coupled_gain {
                        p_hists[b * n_classes + k] += all_probs[prob_base + k];
                    }
                }
                if use_coupled_gain {
                    let pp_base = b * n_classes * n_classes;
                    for a in 0..n_classes {
                        let pa = all_probs[prob_base + a];
                        let row_base = a * n_classes;
                        for c in 0..n_classes {
                            pp_hists[pp_base + row_base + c] += pa * all_probs[prob_base + c];
                        }
                    }
                }
            }
        }
    } else {
        for &idx in node_indices {
            let b = col_bins[idx as usize] as usize;
            let i = idx as usize;
            let prob_base = i * n_classes;
            for k in 0..n_classes {
                g_hists[k * max_bins + b] += all_gradients[k * n_rows + i];
                h_hists[k * max_bins + b] += all_hessians[k * n_rows + i];
                if use_coupled_gain {
                    p_hists[b * n_classes + k] += all_probs[prob_base + k];
                }
            }
            if use_coupled_gain {
                let pp_base = b * n_classes * n_classes;
                for a in 0..n_classes {
                    let pa = all_probs[prob_base + a];
                    let row_base = a * n_classes;
                    for c in 0..n_classes {
                        pp_hists[pp_base + row_base + c] += pa * all_probs[prob_base + c];
                    }
                }
            }
        }
    }

    let parent_obj = if use_coupled_gain {
        parent_dense_gain
    } else {
        let mut obj = 0.0f64;
        for k in 0..n_classes {
            obj += g_sums[k] * g_sums[k] / (h_sums[k] + lambda_reg);
        }
        obj
    };

    let mut best_gain = f64::NEG_INFINITY;
    let mut best_bin = 0usize;
    let mut best_missing_left = true;
    let mut best_is_cat = false;
    let mut best_cat_split_idx: usize = 0;
    let mut best_cat_bins: Vec<usize> = Vec::new();
    let missing_penalty = missing_route_penalty(
        h_sums.iter().copied().sum::<f64>(),
        h_miss.iter().copied().sum::<f64>(),
    );

    if binned.is_categorical[feat] {
        let mut cat_bins: Vec<usize> = Vec::new();
        for bin in 0..feat_n_bins {
            let total_h: f64 = (0..n_classes).map(|k| h_hists[k * max_bins + bin]).sum();
            if total_h > 0.0 {
                cat_bins.push(bin);
            }
        }

        if cat_bins.len() > 1 {
            let total_h_nm: f64 = (0..n_classes).map(|k| h_sums[k] - h_miss[k]).sum();
            let total_proj_nm: f64 = (0..n_classes)
                .map(|k| (g_sums[k] - g_miss[k]) * cat_sort_dir[k])
                .sum();
            let node_ratio = if total_h_nm > 1e-10 {
                total_proj_nm / total_h_nm
            } else {
                0.0
            };
            let mut scalar_scores = vec![0.0f64; feat_n_bins];
            let mut parent_updates = vec![0.0f64; n_classes];
            let mut bin_updates = vec![0.0f64; feat_n_bins * n_classes];
            for k in 0..n_classes {
                let g_nm_k = g_sums[k] - g_miss[k];
                let h_nm_k = h_sums[k] - h_miss[k];
                parent_updates[k] = -g_nm_k / (h_nm_k + lambda_reg).max(1e-12);
            }
            for &bin in &cat_bins {
                let mut proj_g = 0.0f64;
                let mut total_h = 0.0f64;
                let base = bin * n_classes;
                for k in 0..n_classes {
                    let gb = g_hists[k * max_bins + bin];
                    let hb = h_hists[k * max_bins + bin];
                    proj_g += gb * cat_sort_dir[k];
                    total_h += hb;
                    bin_updates[base + k] =
                        -(gb + cat_smooth * parent_updates[k]) / (hb + cat_smooth + 1e-12);
                }
                scalar_scores[bin] = (proj_g + cat_smooth * node_ratio) / (total_h + cat_smooth);
            }

            let mut eval_cat_order = |ordered_bins: &[usize]| {
                let mut cum_g = vec![0.0f64; n_classes];
                let mut cum_h = vec![0.0f64; n_classes];
                let mut cum_p = if use_coupled_gain {
                    vec![0.0f64; n_classes]
                } else {
                    Vec::new()
                };
                let mut cum_pp = if use_coupled_gain {
                    vec![0.0f64; n_classes * n_classes]
                } else {
                    Vec::new()
                };
                let mut left_g = if use_coupled_gain {
                    vec![0.0f64; n_classes]
                } else {
                    Vec::new()
                };
                let mut right_g = if use_coupled_gain {
                    vec![0.0f64; n_classes]
                } else {
                    Vec::new()
                };
                let mut left_p = if use_coupled_gain {
                    vec![0.0f64; n_classes]
                } else {
                    Vec::new()
                };
                let mut right_p = if use_coupled_gain {
                    vec![0.0f64; n_classes]
                } else {
                    Vec::new()
                };
                let mut left_pp = if use_coupled_gain {
                    vec![0.0f64; n_classes * n_classes]
                } else {
                    Vec::new()
                };
                let mut right_pp = if use_coupled_gain {
                    vec![0.0f64; n_classes * n_classes]
                } else {
                    Vec::new()
                };
                let mut dense_a = if use_coupled_gain {
                    vec![0.0f64; n_classes * n_classes]
                } else {
                    Vec::new()
                };
                let mut dense_rhs = if use_coupled_gain {
                    vec![0.0f64; n_classes]
                } else {
                    Vec::new()
                };

                for i in 0..ordered_bins.len() - 1 {
                    let bin = ordered_bins[i];
                    for k in 0..n_classes {
                        cum_g[k] += g_hists[k * max_bins + bin];
                        cum_h[k] += h_hists[k * max_bins + bin];
                    }
                    if use_coupled_gain {
                        let p_base = bin * n_classes;
                        let pp_base = bin * n_classes * n_classes;
                        for k in 0..n_classes {
                            cum_p[k] += p_hists[p_base + k];
                        }
                        for kk in 0..(n_classes * n_classes) {
                            cum_pp[kk] += pp_hists[pp_base + kk];
                        }
                    }
                    for miss_dir in 0..2u8 {
                        let miss_left = miss_dir == 0;
                        let mut gain = 0.0f64;
                        let mut total_lh = 0.0f64;
                        let mut total_rh = 0.0f64;
                        for k in 0..n_classes {
                            let g_nm_k = g_sums[k] - g_miss[k];
                            let h_nm_k = h_sums[k] - h_miss[k];
                            let (lg, lh, rg, rh) = if miss_left {
                                (
                                    cum_g[k] + g_miss[k],
                                    cum_h[k] + h_miss[k],
                                    g_nm_k - cum_g[k],
                                    h_nm_k - cum_h[k],
                                )
                            } else {
                                (
                                    cum_g[k],
                                    cum_h[k],
                                    g_nm_k - cum_g[k] + g_miss[k],
                                    h_nm_k - cum_h[k] + h_miss[k],
                                )
                            };
                            if use_coupled_gain {
                                left_g[k] = lg;
                                right_g[k] = rg;
                            } else {
                                gain += lg * lg / (lh + lambda_reg) + rg * rg / (rh + lambda_reg);
                            }
                            total_lh += lh;
                            total_rh += rh;
                        }
                        if use_coupled_gain {
                            for k in 0..n_classes {
                                left_p[k] = if miss_left {
                                    cum_p[k] + p_miss[k]
                                } else {
                                    cum_p[k]
                                };
                                right_p[k] = parent_p_sums[k] - left_p[k];
                            }
                            for kk in 0..(n_classes * n_classes) {
                                left_pp[kk] = if miss_left {
                                    cum_pp[kk] + pp_miss[kk]
                                } else {
                                    cum_pp[kk]
                                };
                                right_pp[kk] = parent_pp_sums[kk] - left_pp[kk];
                            }
                            let left_obj = dense_multiclass_gain(
                                &left_g,
                                &left_p,
                                &left_pp,
                                lambda_reg,
                                &mut dense_a,
                                &mut dense_rhs,
                            );
                            let right_obj = dense_multiclass_gain(
                                &right_g,
                                &right_p,
                                &right_pp,
                                lambda_reg,
                                &mut dense_a,
                                &mut dense_rhs,
                            );
                            gain = 0.5 * (left_obj + right_obj - parent_obj) - gamma;
                        } else {
                            gain = 0.5 * (gain - parent_obj) - gamma;
                        }
                        if gain_penalty > 0.0 {
                            let mut pen = 0.0;
                            for k in 0..n_classes {
                                let h_nm_k = h_sums[k] - h_miss[k];
                                let (lh, rh) = if miss_left {
                                    (cum_h[k] + h_miss[k], h_nm_k - cum_h[k])
                                } else {
                                    (cum_h[k], h_nm_k - cum_h[k] + h_miss[k])
                                };
                                pen += 1.0 / (lh + lambda_reg) + 1.0 / (rh + lambda_reg)
                                    - 1.0 / (h_sums[k] + lambda_reg);
                            }
                            gain -= gain_penalty * 0.5 * pen;
                        }
                        gain = evidence_adjusted_gain(
                            binned,
                            gain,
                            total_lh,
                            total_rh,
                            total_lh + total_rh,
                            lambda_reg,
                            ordered_bins.len().saturating_sub(1),
                        );
                        gain -= missing_penalty;
                        if random_strength > 0.0 && gain > 0.0 {
                            let noise_idx = if miss_left { i * 2 } else { i * 2 + 1 };
                            gain *=
                                1.0 + random_strength * split_noise(noise_seed, feat, noise_idx);
                        }
                        if total_lh >= min_h
                            && total_rh >= min_h
                            && gain.is_finite()
                            && gain > best_gain
                        {
                            best_gain = gain;
                            best_bin = 0;
                            best_missing_left = miss_left;
                            best_is_cat = true;
                            best_cat_split_idx = i;
                            best_cat_bins.clear();
                            best_cat_bins.extend_from_slice(ordered_bins);
                        }
                    }
                }
            };

            let mut scalar_sorted = cat_bins.clone();
            scalar_sorted.sort_by(|&a, &b| {
                scalar_scores[a]
                    .partial_cmp(&scalar_scores[b])
                    .unwrap_or(Ordering::Equal)
            });
            eval_cat_order(&scalar_sorted);

            if n_classes >= 3 {
                let contrast_vectors = multiclass_cat_contrast_vectors(cat_sort_dir, g_sums);
                for contrast in contrast_vectors {
                    let mut ordered_bins = cat_bins.clone();
                    sort_multiclass_cat_bins_by_contrast(
                        &mut ordered_bins,
                        &bin_updates,
                        n_classes,
                        &contrast,
                        &scalar_scores,
                    );
                    eval_cat_order(&ordered_bins);
                }
            }
        }
        if best_is_cat {
            let mut mask: CatBitmask = Vec::new();
            for j in 0..=best_cat_split_idx {
                bitmask_set(&mut mask, best_cat_bins[j]);
            }
            return SplitResult::axis(best_gain, feat, best_bin, best_missing_left, true, mask);
        }
    } else {
        let mut cum_g = vec![0.0f64; n_classes];
        let mut cum_h = vec![0.0f64; n_classes];
        let mut cum_p = if use_coupled_gain {
            vec![0.0f64; n_classes]
        } else {
            Vec::new()
        };
        let mut cum_pp = if use_coupled_gain {
            vec![0.0f64; n_classes * n_classes]
        } else {
            Vec::new()
        };
        let mut left_g = if use_coupled_gain {
            vec![0.0f64; n_classes]
        } else {
            Vec::new()
        };
        let mut right_g = if use_coupled_gain {
            vec![0.0f64; n_classes]
        } else {
            Vec::new()
        };
        let mut left_p = if use_coupled_gain {
            vec![0.0f64; n_classes]
        } else {
            Vec::new()
        };
        let mut right_p = if use_coupled_gain {
            vec![0.0f64; n_classes]
        } else {
            Vec::new()
        };
        let mut left_pp = if use_coupled_gain {
            vec![0.0f64; n_classes * n_classes]
        } else {
            Vec::new()
        };
        let mut right_pp = if use_coupled_gain {
            vec![0.0f64; n_classes * n_classes]
        } else {
            Vec::new()
        };
        let mut dense_a = if use_coupled_gain {
            vec![0.0f64; n_classes * n_classes]
        } else {
            Vec::new()
        };
        let mut dense_rhs = if use_coupled_gain {
            vec![0.0f64; n_classes]
        } else {
            Vec::new()
        };

        for bin in 0..feat_n_bins - 1 {
            for k in 0..n_classes {
                cum_g[k] += g_hists[k * max_bins + bin];
                cum_h[k] += h_hists[k * max_bins + bin];
            }
            if use_coupled_gain {
                let p_base = bin * n_classes;
                let pp_base = bin * n_classes * n_classes;
                for k in 0..n_classes {
                    cum_p[k] += p_hists[p_base + k];
                }
                for kk in 0..(n_classes * n_classes) {
                    cum_pp[kk] += pp_hists[pp_base + kk];
                }
            }

            let mut raw_ml = -parent_obj;
            let mut raw_mr = -parent_obj;
            let mut lh_ml = 0.0f64;
            let mut rh_ml = 0.0f64;
            let mut lh_mr = 0.0f64;
            let mut rh_mr = 0.0f64;

            for k in 0..n_classes {
                let g_nm_k = g_sums[k] - g_miss[k];
                let h_nm_k = h_sums[k] - h_miss[k];
                let other_g = g_nm_k - cum_g[k];
                let other_h = h_nm_k - cum_h[k];

                let lg_ml = cum_g[k] + g_miss[k];
                let lh_ml_k = cum_h[k] + h_miss[k];
                if use_coupled_gain {
                    left_g[k] = lg_ml;
                    right_g[k] = other_g;
                } else {
                    raw_ml += lg_ml * lg_ml / (lh_ml_k + lambda_reg)
                        + other_g * other_g / (other_h + lambda_reg);
                }
                lh_ml += lh_ml_k;
                rh_ml += other_h;

                let rg_mr = other_g + g_miss[k];
                let rh_mr_k = other_h + h_miss[k];
                if use_coupled_gain {
                    // Filled below after the loop when missing goes right.
                } else {
                    raw_mr += cum_g[k] * cum_g[k] / (cum_h[k] + lambda_reg)
                        + rg_mr * rg_mr / (rh_mr_k + lambda_reg);
                }
                lh_mr += cum_h[k];
                rh_mr += rh_mr_k;
            }

            let mut gain_ml = if use_coupled_gain {
                for k in 0..n_classes {
                    left_p[k] = cum_p[k] + p_miss[k];
                    right_p[k] = parent_p_sums[k] - left_p[k];
                }
                for kk in 0..(n_classes * n_classes) {
                    left_pp[kk] = cum_pp[kk] + pp_miss[kk];
                    right_pp[kk] = parent_pp_sums[kk] - left_pp[kk];
                }
                let left_obj = dense_multiclass_gain(
                    &left_g,
                    &left_p,
                    &left_pp,
                    lambda_reg,
                    &mut dense_a,
                    &mut dense_rhs,
                );
                let right_obj = dense_multiclass_gain(
                    &right_g,
                    &right_p,
                    &right_pp,
                    lambda_reg,
                    &mut dense_a,
                    &mut dense_rhs,
                );
                0.5 * (left_obj + right_obj - parent_obj) - gamma
            } else {
                0.5 * raw_ml - gamma
            };
            let mut gain_mr = if use_coupled_gain {
                for k in 0..n_classes {
                    left_g[k] = cum_g[k];
                    right_g[k] = (g_sums[k] - g_miss[k] - cum_g[k]) + g_miss[k];
                    left_p[k] = cum_p[k];
                    right_p[k] = parent_p_sums[k] - left_p[k];
                }
                for kk in 0..(n_classes * n_classes) {
                    left_pp[kk] = cum_pp[kk];
                    right_pp[kk] = parent_pp_sums[kk] - left_pp[kk];
                }
                let left_obj = dense_multiclass_gain(
                    &left_g,
                    &left_p,
                    &left_pp,
                    lambda_reg,
                    &mut dense_a,
                    &mut dense_rhs,
                );
                let right_obj = dense_multiclass_gain(
                    &right_g,
                    &right_p,
                    &right_pp,
                    lambda_reg,
                    &mut dense_a,
                    &mut dense_rhs,
                );
                0.5 * (left_obj + right_obj - parent_obj) - gamma
            } else {
                0.5 * raw_mr - gamma
            };

            if gain_penalty > 0.0 {
                let mut pen_ml = 0.0;
                let mut pen_mr = 0.0;
                for k in 0..n_classes {
                    let h_nm_k = h_sums[k] - h_miss[k];
                    pen_ml += 1.0 / (cum_h[k] + h_miss[k] + lambda_reg)
                        + 1.0 / (h_nm_k - cum_h[k] + lambda_reg)
                        - 1.0 / (h_sums[k] + lambda_reg);
                    pen_mr += 1.0 / (cum_h[k] + lambda_reg)
                        + 1.0 / (h_nm_k - cum_h[k] + h_miss[k] + lambda_reg)
                        - 1.0 / (h_sums[k] + lambda_reg);
                }
                gain_ml -= gain_penalty * 0.5 * pen_ml;
                gain_mr -= gain_penalty * 0.5 * pen_mr;
            }
            gain_ml = evidence_adjusted_gain(
                binned,
                gain_ml,
                lh_ml,
                rh_ml,
                lh_ml + rh_ml,
                lambda_reg,
                feat_n_bins.saturating_sub(1),
            );
            gain_mr = evidence_adjusted_gain(
                binned,
                gain_mr,
                lh_mr,
                rh_mr,
                lh_mr + rh_mr,
                lambda_reg,
                feat_n_bins.saturating_sub(1),
            );
            gain_ml -= missing_penalty;
            gain_mr -= missing_penalty;

            if random_strength > 0.0 {
                if gain_ml > 0.0 {
                    gain_ml *= 1.0 + random_strength * split_noise(noise_seed, feat, bin * 2);
                }
                if gain_mr > 0.0 {
                    gain_mr *= 1.0 + random_strength * split_noise(noise_seed, feat, bin * 2 + 1);
                }
            }

            if lh_ml >= min_h && rh_ml >= min_h && gain_ml.is_finite() && gain_ml > best_gain {
                best_gain = gain_ml;
                best_bin = bin;
                best_missing_left = true;
            }
            if lh_mr >= min_h && rh_mr >= min_h && gain_mr.is_finite() && gain_mr > best_gain {
                best_gain = gain_mr;
                best_bin = bin;
                best_missing_left = false;
            }
        }
    }

    SplitResult::axis(
        best_gain,
        feat,
        best_bin,
        best_missing_left,
        false,
        Vec::new(),
    )
}

#[inline]
pub(super) fn normalized_bin_coord(n_bins: usize, bin: u16) -> f64 {
    let denom = n_bins.saturating_sub(1).max(1) as f64;
    2.0 * (bin as f64) / denom - 1.0
}

pub(super) fn attended_numeric_features(
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    active_features: &[usize],
    g_sum: f64,
    h_sum: f64,
    lambda_reg: f64,
    monotone_constraints: &[i8],
    top_k: usize,
) -> Vec<usize> {
    let node_value = -g_sum / (h_sum + lambda_reg).max(1e-12);
    let mut scored: Vec<(f64, usize)> = Vec::new();

    for &feat in active_features {
        if binned.is_categorical[feat]
            || binned.n_bins(feat) <= 1
            || monotone_constraints.get(feat).copied().unwrap_or(0) != 0
        {
            continue;
        }
        let feat_n_bins = binned.n_bins(feat);
        let mut g_bins = vec![0.0f64; feat_n_bins];
        let mut h_bins = vec![0.0f64; feat_n_bins];
        let col_bins = binned.col_bins(feat);
        let mut cover_h = 0.0f64;
        for &idx in node_indices {
            let row = idx as usize;
            let bin = col_bins[row];
            if bin == MISSING_BIN {
                continue;
            }
            g_bins[bin as usize] += gradients[row];
            h_bins[bin as usize] += hessians[row];
            cover_h += hessians[row];
        }
        if cover_h <= 1e-12 {
            continue;
        }
        let mut score = 0.0f64;
        for bin in 0..feat_n_bins {
            let hb = h_bins[bin];
            if hb <= 1e-12 {
                continue;
            }
            let leaf_value = -g_bins[bin] / (hb + lambda_reg).max(1e-12);
            let delta = leaf_value - node_value;
            score += hb * delta * delta;
        }
        score *= cover_h / (h_sum + 1e-12);
        if score.is_finite() && score > 0.0 {
            scored.push((score, feat));
        }
    }

    scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    scored.truncate(top_k);
    scored.into_iter().map(|(_, feat)| feat).collect()
}

pub(super) fn attended_numeric_features_multi(
    binned: &BinnedData,
    all_gradients: &[f64],
    all_hessians: &[f64],
    n_classes: usize,
    n_rows: usize,
    node_indices: &[u32],
    active_features: &[usize],
    g_sums: &[f64],
    h_sums: &[f64],
    lambda_reg: f64,
    cat_sort_dir: &[f64],
    top_k: usize,
) -> Vec<usize> {
    let proj_g_sum: f64 = (0..n_classes).map(|k| cat_sort_dir[k] * g_sums[k]).sum();
    let proj_h_sum: f64 = h_sums.iter().sum();
    let node_value = -proj_g_sum / (proj_h_sum + lambda_reg).max(1e-12);
    let mut scored: Vec<(f64, usize)> = Vec::new();

    for &feat in active_features {
        if binned.is_categorical[feat] || binned.n_bins(feat) <= 1 {
            continue;
        }
        let feat_n_bins = binned.n_bins(feat);
        let mut g_bins = vec![0.0f64; feat_n_bins];
        let mut h_bins = vec![0.0f64; feat_n_bins];
        let col_bins = binned.col_bins(feat);
        let mut cover_h = 0.0f64;

        for &idx in node_indices {
            let row = idx as usize;
            let bin = col_bins[row];
            if bin == MISSING_BIN {
                continue;
            }
            let mut g_proj = 0.0f64;
            let mut h_proj = 0.0f64;
            for k in 0..n_classes {
                g_proj += cat_sort_dir[k] * all_gradients[k * n_rows + row];
                h_proj += all_hessians[k * n_rows + row];
            }
            g_bins[bin as usize] += g_proj;
            h_bins[bin as usize] += h_proj;
            cover_h += h_proj;
        }
        if cover_h <= 1e-12 {
            continue;
        }
        let mut score = 0.0f64;
        for bin in 0..feat_n_bins {
            let hb = h_bins[bin];
            if hb <= 1e-12 {
                continue;
            }
            let leaf_value = -g_bins[bin] / (hb + lambda_reg).max(1e-12);
            let delta = leaf_value - node_value;
            score += hb * delta * delta;
        }
        score *= cover_h / (proj_h_sum + 1e-12);
        if score.is_finite() && score > 0.0 {
            scored.push((score, feat));
        }
    }

    scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    scored.truncate(top_k);
    scored.into_iter().map(|(_, feat)| feat).collect()
}

pub(super) fn attended_features(
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    active_features: &[usize],
    g_sum: f64,
    h_sum: f64,
    lambda_reg: f64,
    top_k: usize,
) -> Vec<usize> {
    let node_value = -g_sum / (h_sum + lambda_reg).max(1e-12);
    let mut scored: Vec<(f64, usize)> = Vec::new();

    for &feat in active_features {
        let feat_n_bins = binned.n_bins(feat);
        if feat_n_bins <= 1 {
            continue;
        }
        let mut g_bins = vec![0.0f64; feat_n_bins];
        let mut h_bins = vec![0.0f64; feat_n_bins];
        let col_bins = binned.col_bins(feat);
        let mut cover_h = 0.0f64;
        for &idx in node_indices {
            let row = idx as usize;
            let bin = col_bins[row];
            if bin == MISSING_BIN {
                continue;
            }
            g_bins[bin as usize] += gradients[row];
            h_bins[bin as usize] += hessians[row];
            cover_h += hessians[row];
        }
        if cover_h <= 1e-12 {
            continue;
        }
        let mut score = 0.0f64;
        for bin in 0..feat_n_bins {
            let hb = h_bins[bin];
            if hb <= 1e-12 {
                continue;
            }
            let leaf_value = -g_bins[bin] / (hb + lambda_reg).max(1e-12);
            let delta = leaf_value - node_value;
            score += hb * delta * delta;
        }
        score *= cover_h / (h_sum + 1e-12);
        if score.is_finite() && score > 0.0 {
            scored.push((score, feat));
        }
    }

    scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    scored.truncate(top_k);
    scored.into_iter().map(|(_, feat)| feat).collect()
}

pub(super) fn attended_features_multi(
    binned: &BinnedData,
    all_gradients: &[f64],
    all_hessians: &[f64],
    n_classes: usize,
    n_rows: usize,
    node_indices: &[u32],
    active_features: &[usize],
    g_sums: &[f64],
    h_sums: &[f64],
    lambda_reg: f64,
    cat_sort_dir: &[f64],
    top_k: usize,
) -> Vec<usize> {
    let proj_g_sum: f64 = (0..n_classes).map(|k| cat_sort_dir[k] * g_sums[k]).sum();
    let proj_h_sum: f64 = h_sums.iter().sum();
    let node_value = -proj_g_sum / (proj_h_sum + lambda_reg).max(1e-12);
    let mut scored: Vec<(f64, usize)> = Vec::new();

    for &feat in active_features {
        let feat_n_bins = binned.n_bins(feat);
        if feat_n_bins <= 1 {
            continue;
        }
        let mut g_bins = vec![0.0f64; feat_n_bins];
        let mut h_bins = vec![0.0f64; feat_n_bins];
        let col_bins = binned.col_bins(feat);
        let mut cover_h = 0.0f64;
        for &idx in node_indices {
            let row = idx as usize;
            let bin = col_bins[row];
            if bin == MISSING_BIN {
                continue;
            }
            let mut g_proj = 0.0f64;
            let mut h_proj = 0.0f64;
            for k in 0..n_classes {
                g_proj += cat_sort_dir[k] * all_gradients[k * n_rows + row];
                h_proj += all_hessians[k * n_rows + row];
            }
            g_bins[bin as usize] += g_proj;
            h_bins[bin as usize] += h_proj;
            cover_h += h_proj;
        }
        if cover_h <= 1e-12 {
            continue;
        }
        let mut score = 0.0f64;
        for bin in 0..feat_n_bins {
            let hb = h_bins[bin];
            if hb <= 1e-12 {
                continue;
            }
            let leaf_value = -g_bins[bin] / (hb + lambda_reg).max(1e-12);
            let delta = leaf_value - node_value;
            score += hb * delta * delta;
        }
        score *= cover_h / (proj_h_sum + 1e-12);
        if score.is_finite() && score > 0.0 {
            scored.push((score, feat));
        }
    }

    scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    scored.truncate(top_k);
    scored.into_iter().map(|(_, feat)| feat).collect()
}

pub(super) fn eval_sparse_oblique_candidate(
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    g_sum: f64,
    h_sum: f64,
    lambda_reg: f64,
    gamma: f64,
    min_h: f64,
    f0: usize,
    f1: usize,
) -> SplitResult {
    let n_bins0 = binned.n_bins(f0);
    let n_bins1 = binned.n_bins(f1);
    if n_bins0 <= 1 || n_bins1 <= 1 {
        return SplitResult::empty();
    }

    let scale0 = 2.0 / n_bins0.saturating_sub(1).max(1) as f64;
    let scale1 = 2.0 / n_bins1.saturating_sub(1).max(1) as f64;
    let mut s00 = 0.0f64;
    let mut s01 = 0.0f64;
    let mut s11 = 0.0f64;
    let mut rhs0 = 0.0f64;
    let mut rhs1 = 0.0f64;
    let mut n_valid = 0usize;

    for &idx in node_indices {
        let row = idx as usize;
        let b0 = binned.get_bin_u16(row, f0);
        let b1 = binned.get_bin_u16(row, f1);
        if b0 == MISSING_BIN || b1 == MISSING_BIN {
            continue;
        }
        let x0 = normalized_bin_coord(n_bins0, b0);
        let x1 = normalized_bin_coord(n_bins1, b1);
        let h = hessians[row].max(1e-12);
        let t = -gradients[row] / (hessians[row] + lambda_reg).max(1e-12);
        s00 += h * x0 * x0;
        s01 += h * x0 * x1;
        s11 += h * x1 * x1;
        rhs0 += h * x0 * t;
        rhs1 += h * x1 * t;
        n_valid += 1;
    }
    if n_valid < 4 {
        return SplitResult::empty();
    }

    let ridge = 1e-3 * (s00 + s11).max(1e-6);
    let a00 = s00 + ridge;
    let a11 = s11 + ridge;
    let det = a00 * a11 - s01 * s01;
    if !det.is_finite() || det.abs() <= 1e-12 {
        return SplitResult::empty();
    }

    let mut w0 = (rhs0 * a11 - rhs1 * s01) / det;
    let mut w1 = (rhs1 * a00 - rhs0 * s01) / det;
    let norm = (w0 * w0 + w1 * w1).sqrt();
    if !norm.is_finite() || norm <= 1e-8 {
        return SplitResult::empty();
    }
    w0 /= norm;
    w1 /= norm;

    let stored_w0 = w0 * scale0;
    let stored_w1 = w1 * scale1;
    let stored_shift = w0 + w1;

    let mut rows: Vec<(f64, u32)> = Vec::with_capacity(n_valid);
    let mut g_miss = 0.0f64;
    let mut h_miss = 0.0f64;
    for &idx in node_indices {
        let row = idx as usize;
        let b0 = binned.get_bin_u16(row, f0);
        let b1 = binned.get_bin_u16(row, f1);
        if b0 == MISSING_BIN || b1 == MISSING_BIN {
            g_miss += gradients[row];
            h_miss += hessians[row];
            continue;
        }
        let proj = w0 * normalized_bin_coord(n_bins0, b0) + w1 * normalized_bin_coord(n_bins1, b1);
        rows.push((proj, idx));
    }
    if rows.len() < 2 {
        return SplitResult::empty();
    }
    rows.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let g_nm = g_sum - g_miss;
    let h_nm = h_sum - h_miss;
    let mut best = SplitResult::empty();
    let mut cum_g = 0.0f64;
    let mut cum_h = 0.0f64;

    for pos in 0..rows.len() - 1 {
        let row = rows[pos].1 as usize;
        cum_g += gradients[row];
        cum_h += hessians[row];
        if rows[pos].0 + 1e-12 >= rows[pos + 1].0 {
            continue;
        }
        let threshold_center = 0.5 * (rows[pos].0 + rows[pos + 1].0);
        let threshold_raw = threshold_center + stored_shift;
        let other_g = g_nm - cum_g;
        let other_h = h_nm - cum_h;

        let lg = cum_g + g_miss;
        let lh = cum_h + h_miss;
        if lh >= min_h && other_h >= min_h {
            let mut gain = 0.5
                * (lg * lg / (lh + lambda_reg) + other_g * other_g / (other_h + lambda_reg)
                    - g_sum * g_sum / (h_sum + lambda_reg))
                - gamma;
            gain = evidence_adjusted_gain(
                binned,
                gain,
                lh,
                other_h,
                h_sum,
                lambda_reg,
                rows.len().saturating_sub(1),
            );
            gain -= missing_route_penalty(h_sum, h_miss);
            if gain.is_finite() && gain > best.gain {
                best = SplitResult::oblique(
                    gain,
                    f0,
                    true,
                    [f0 as u32, f1 as u32],
                    [stored_w0 as f32, stored_w1 as f32],
                    threshold_raw as f32,
                );
            }
        }

        let rg = other_g + g_miss;
        let rh = other_h + h_miss;
        if cum_h >= min_h && rh >= min_h {
            let mut gain = 0.5
                * (cum_g * cum_g / (cum_h + lambda_reg) + rg * rg / (rh + lambda_reg)
                    - g_sum * g_sum / (h_sum + lambda_reg))
                - gamma;
            gain = evidence_adjusted_gain(
                binned,
                gain,
                cum_h,
                rh,
                h_sum,
                lambda_reg,
                rows.len().saturating_sub(1),
            );
            gain -= missing_route_penalty(h_sum, h_miss);
            if gain.is_finite() && gain > best.gain {
                best = SplitResult::oblique(
                    gain,
                    f0,
                    false,
                    [f0 as u32, f1 as u32],
                    [stored_w0 as f32, stored_w1 as f32],
                    threshold_raw as f32,
                );
            }
        }
    }

    best
}

pub(super) fn find_sparse_oblique_split(
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    active_features: &[usize],
    g_sum: f64,
    h_sum: f64,
    lambda_reg: f64,
    gamma: f64,
    min_h: f64,
    monotone_constraints: &[i8],
) -> SplitResult {
    let numeric_feats = attended_numeric_features(
        binned,
        gradients,
        hessians,
        node_indices,
        active_features,
        g_sum,
        h_sum,
        lambda_reg,
        monotone_constraints,
        4,
    );
    if numeric_feats.len() < 2 || node_indices.len() < 16 {
        return SplitResult::empty();
    }

    let mut best = SplitResult::empty();
    for i0 in 0..numeric_feats.len() {
        for i1 in i0 + 1..numeric_feats.len() {
            let cand = eval_sparse_oblique_candidate(
                binned,
                gradients,
                hessians,
                node_indices,
                g_sum,
                h_sum,
                lambda_reg,
                gamma,
                min_h,
                numeric_feats[i0],
                numeric_feats[i1],
            );
            if cand.gain > best.gain {
                best = cand;
            }
        }
    }

    best
}

pub(super) fn eval_sparse_oblique_candidate_multi(
    binned: &BinnedData,
    all_gradients: &[f64],
    all_hessians: &[f64],
    n_classes: usize,
    n_rows: usize,
    node_indices: &[u32],
    g_sums: &[f64],
    h_sums: &[f64],
    lambda_reg: f64,
    gamma: f64,
    min_h: f64,
    f0: usize,
    f1: usize,
    cat_sort_dir: &[f64],
) -> SplitResult {
    let n_bins0 = binned.n_bins(f0);
    let n_bins1 = binned.n_bins(f1);
    if n_bins0 <= 1 || n_bins1 <= 1 {
        return SplitResult::empty();
    }

    let scale0 = 2.0 / n_bins0.saturating_sub(1).max(1) as f64;
    let scale1 = 2.0 / n_bins1.saturating_sub(1).max(1) as f64;
    let mut s00 = 0.0f64;
    let mut s01 = 0.0f64;
    let mut s11 = 0.0f64;
    let mut rhs0 = 0.0f64;
    let mut rhs1 = 0.0f64;
    let mut n_valid = 0usize;

    for &idx in node_indices {
        let row = idx as usize;
        let b0 = binned.get_bin_u16(row, f0);
        let b1 = binned.get_bin_u16(row, f1);
        if b0 == MISSING_BIN || b1 == MISSING_BIN {
            continue;
        }
        let x0 = normalized_bin_coord(n_bins0, b0);
        let x1 = normalized_bin_coord(n_bins1, b1);
        let mut g_proj = 0.0f64;
        let mut h_proj = 0.0f64;
        for k in 0..n_classes {
            g_proj += cat_sort_dir[k] * all_gradients[k * n_rows + row];
            h_proj += all_hessians[k * n_rows + row];
        }
        let h = h_proj.max(1e-12);
        let t = -g_proj / (h_proj + lambda_reg).max(1e-12);
        s00 += h * x0 * x0;
        s01 += h * x0 * x1;
        s11 += h * x1 * x1;
        rhs0 += h * x0 * t;
        rhs1 += h * x1 * t;
        n_valid += 1;
    }
    if n_valid < 4 {
        return SplitResult::empty();
    }

    let ridge = 1e-3 * (s00 + s11).max(1e-6);
    let a00 = s00 + ridge;
    let a11 = s11 + ridge;
    let det = a00 * a11 - s01 * s01;
    if !det.is_finite() || det.abs() <= 1e-12 {
        return SplitResult::empty();
    }

    let mut w0 = (rhs0 * a11 - rhs1 * s01) / det;
    let mut w1 = (rhs1 * a00 - rhs0 * s01) / det;
    let norm = (w0 * w0 + w1 * w1).sqrt();
    if !norm.is_finite() || norm <= 1e-8 {
        return SplitResult::empty();
    }
    w0 /= norm;
    w1 /= norm;

    let stored_w0 = w0 * scale0;
    let stored_w1 = w1 * scale1;
    let stored_shift = w0 + w1;

    let mut rows: Vec<(f64, u32)> = Vec::with_capacity(n_valid);
    let mut g_miss = vec![0.0f64; n_classes];
    let mut h_miss = vec![0.0f64; n_classes];
    for &idx in node_indices {
        let row = idx as usize;
        let b0 = binned.get_bin_u16(row, f0);
        let b1 = binned.get_bin_u16(row, f1);
        if b0 == MISSING_BIN || b1 == MISSING_BIN {
            for k in 0..n_classes {
                g_miss[k] += all_gradients[k * n_rows + row];
                h_miss[k] += all_hessians[k * n_rows + row];
            }
            continue;
        }
        let proj = w0 * normalized_bin_coord(n_bins0, b0) + w1 * normalized_bin_coord(n_bins1, b1);
        rows.push((proj, idx));
    }
    if rows.len() < 2 {
        return SplitResult::empty();
    }
    rows.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let mut g_nm = vec![0.0f64; n_classes];
    let mut h_nm = vec![0.0f64; n_classes];
    for k in 0..n_classes {
        g_nm[k] = g_sums[k] - g_miss[k];
        h_nm[k] = h_sums[k] - h_miss[k];
    }
    let missing_penalty = missing_route_penalty(
        h_sums.iter().copied().sum::<f64>(),
        h_miss.iter().copied().sum::<f64>(),
    );

    let mut best = SplitResult::empty();
    let mut cum_g = vec![0.0f64; n_classes];
    let mut cum_h = vec![0.0f64; n_classes];

    for pos in 0..rows.len() - 1 {
        let row = rows[pos].1 as usize;
        for k in 0..n_classes {
            cum_g[k] += all_gradients[k * n_rows + row];
            cum_h[k] += all_hessians[k * n_rows + row];
        }
        if rows[pos].0 + 1e-12 >= rows[pos + 1].0 {
            continue;
        }
        let threshold_center = 0.5 * (rows[pos].0 + rows[pos + 1].0);
        let threshold_raw = threshold_center + stored_shift;

        let mut left_h_total = 0.0f64;
        let mut right_h_total = 0.0f64;
        let mut gain_left = 0.0f64;
        for k in 0..n_classes {
            let lg = cum_g[k] + g_miss[k];
            let lh = cum_h[k] + h_miss[k];
            let rg = g_nm[k] - cum_g[k];
            let rh = h_nm[k] - cum_h[k];
            left_h_total += lh;
            right_h_total += rh;
            gain_left += lg * lg / (lh + lambda_reg) + rg * rg / (rh + lambda_reg)
                - g_sums[k] * g_sums[k] / (h_sums[k] + lambda_reg);
        }
        gain_left = 0.5 * gain_left - gamma;
        gain_left = evidence_adjusted_gain(
            binned,
            gain_left,
            left_h_total,
            right_h_total,
            left_h_total + right_h_total,
            lambda_reg,
            rows.len().saturating_sub(1),
        );
        gain_left -= missing_penalty;
        if left_h_total >= min_h
            && right_h_total >= min_h
            && gain_left.is_finite()
            && gain_left > best.gain
        {
            best = SplitResult::oblique(
                gain_left,
                f0,
                true,
                [f0 as u32, f1 as u32],
                [stored_w0 as f32, stored_w1 as f32],
                threshold_raw as f32,
            );
        }

        let mut left_h_total = 0.0f64;
        let mut right_h_total = 0.0f64;
        let mut gain_right = 0.0f64;
        for k in 0..n_classes {
            let rg = (g_nm[k] - cum_g[k]) + g_miss[k];
            let rh = (h_nm[k] - cum_h[k]) + h_miss[k];
            left_h_total += cum_h[k];
            right_h_total += rh;
            gain_right += cum_g[k] * cum_g[k] / (cum_h[k] + lambda_reg)
                + rg * rg / (rh + lambda_reg)
                - g_sums[k] * g_sums[k] / (h_sums[k] + lambda_reg);
        }
        gain_right = 0.5 * gain_right - gamma;
        gain_right = evidence_adjusted_gain(
            binned,
            gain_right,
            left_h_total,
            right_h_total,
            left_h_total + right_h_total,
            lambda_reg,
            rows.len().saturating_sub(1),
        );
        gain_right -= missing_penalty;
        if left_h_total >= min_h
            && right_h_total >= min_h
            && gain_right.is_finite()
            && gain_right > best.gain
        {
            best = SplitResult::oblique(
                gain_right,
                f0,
                false,
                [f0 as u32, f1 as u32],
                [stored_w0 as f32, stored_w1 as f32],
                threshold_raw as f32,
            );
        }
    }

    best
}

pub(super) fn find_sparse_oblique_split_multi(
    binned: &BinnedData,
    all_gradients: &[f64],
    all_hessians: &[f64],
    n_classes: usize,
    n_rows: usize,
    node_indices: &[u32],
    active_features: &[usize],
    g_sums: &[f64],
    h_sums: &[f64],
    lambda_reg: f64,
    gamma: f64,
    min_h: f64,
    cat_sort_dir: &[f64],
) -> SplitResult {
    let numeric_feats = attended_numeric_features_multi(
        binned,
        all_gradients,
        all_hessians,
        n_classes,
        n_rows,
        node_indices,
        active_features,
        g_sums,
        h_sums,
        lambda_reg,
        cat_sort_dir,
        4,
    );
    if numeric_feats.len() < 2 || node_indices.len() < 16 {
        return SplitResult::empty();
    }

    let mut best = SplitResult::empty();
    for i0 in 0..numeric_feats.len() {
        for i1 in i0 + 1..numeric_feats.len() {
            let cand = eval_sparse_oblique_candidate_multi(
                binned,
                all_gradients,
                all_hessians,
                n_classes,
                n_rows,
                node_indices,
                g_sums,
                h_sums,
                lambda_reg,
                gamma,
                min_h,
                numeric_feats[i0],
                numeric_feats[i1],
                cat_sort_dir,
            );
            if cand.gain > best.gain {
                best = cand;
            }
        }
    }

    best
}

/// Extra Trees split: for each numeric feature, pick ONE random bin threshold
/// and evaluate gain only for that threshold. For categoricals, pick a random
/// partition. This maximizes tree decorrelation and reduces variance.
pub(super) fn find_extra_trees_split(
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    active_features: &[usize],
    g_sum: f64,
    h_sum: f64,
    lambda_reg: f64,
    gamma: f64,
    min_h: f64,
    g_hist: &mut [f64],
    h_hist: &mut [f64],
    node_seed: u64,
    monotone_constraints: &[i8],
) -> SplitResult {
    let mut best = SplitResult::empty();
    let n_random_cuts = 3usize;

    for (fi, &feat) in active_features.iter().enumerate() {
        let feat_n_bins = binned.n_bins(feat);
        if feat_n_bins <= 1 {
            continue;
        }

        let col_bins = binned.col_bins(feat);
        for i in 0..feat_n_bins {
            g_hist[i] = 0.0;
            h_hist[i] = 0.0;
        }

        let mut g_miss = 0.0f64;
        let mut h_miss = 0.0f64;
        for &idx in node_indices {
            let bin = col_bins[idx as usize];
            if bin == MISSING_BIN {
                g_miss += gradients[idx as usize];
                h_miss += hessians[idx as usize];
            } else {
                g_hist[bin as usize] += gradients[idx as usize];
                h_hist[bin as usize] += hessians[idx as usize];
            }
        }

        let g_nm = g_sum - g_miss;
        let h_nm = h_sum - h_miss;
        let missing_penalty = missing_route_penalty(h_sum, h_miss);

        if binned.is_categorical[feat] {
            // For categoricals: pick a random partition point in the sorted order
            let mut cat_bins: Vec<(usize, f64, f64)> = Vec::new();
            for bin in 0..feat_n_bins {
                if h_hist[bin] > 0.0 {
                    cat_bins.push((bin, g_hist[bin], h_hist[bin]));
                }
            }
            if cat_bins.len() <= 1 {
                continue;
            }
            let node_ratio = if h_nm > 1e-10 { g_nm / h_nm } else { 0.0 };
            cat_bins.sort_by(|a, b| {
                let ra = (a.1 + 10.0 * node_ratio) / (a.2 + 10.0);
                let rb = (b.1 + 10.0 * node_ratio) / (b.2 + 10.0);
                ra.partial_cmp(&rb).unwrap_or(Ordering::Equal)
            });
            let n_trials = n_random_cuts.min(cat_bins.len() - 1);
            for trial in 0..n_trials {
                let rand_idx = {
                    let h = node_seed
                        .wrapping_mul(0x9E3779B97F4A7C15)
                        .wrapping_add(fi as u64)
                        .wrapping_add((trial as u64).wrapping_mul(0xD1B54A32D192ED03));
                    let h2 = h.wrapping_mul(0x517CC1B727220A95);
                    (h2 >> 33) as usize % (cat_bins.len() - 1)
                };
                let mut cum_g = 0.0f64;
                let mut cum_h = 0.0f64;
                for bin in cat_bins.iter().take(rand_idx + 1) {
                    cum_g += bin.1;
                    cum_h += bin.2;
                }
                let lg = cum_g + g_miss;
                let lh = cum_h + h_miss;
                let other_g = g_nm - cum_g;
                let other_h = h_nm - cum_h;
                if lh >= min_h && other_h >= min_h {
                    let mut gain = 0.5
                        * (lg * lg / (lh + lambda_reg)
                            + other_g * other_g / (other_h + lambda_reg)
                            - g_sum * g_sum / (h_sum + lambda_reg))
                        - gamma;
                    gain = evidence_adjusted_gain(
                        binned, gain, lh, other_h, h_sum, lambda_reg, n_trials,
                    );
                    gain -= missing_penalty;
                    if gain.is_finite() && gain > best.gain {
                        let mut mask: CatBitmask = Vec::new();
                        for bin in cat_bins.iter().take(rand_idx + 1) {
                            bitmask_set(&mut mask, bin.0);
                        }
                        best = SplitResult::axis(gain, feat, 0, true, true, mask);
                    }
                }
            }
        } else {
            // Numeric: pick ONE random occupied bin as threshold
            let occupied: Vec<usize> = (0..feat_n_bins - 1)
                .filter(|&b| h_hist[b] > 0.0 || h_hist[b + 1] > 0.0)
                .collect();
            if occupied.is_empty() {
                continue;
            }

            let mc = if feat < monotone_constraints.len() {
                monotone_constraints[feat]
            } else {
                0
            };

            let n_trials = n_random_cuts.min(occupied.len());
            for trial in 0..n_trials {
                let rand_bin = {
                    let h = node_seed
                        .wrapping_mul(0x517CC1B727220A95)
                        .wrapping_add(fi as u64)
                        .wrapping_add((trial as u64).wrapping_mul(0x9FB21C651E98DF25));
                    let h2 = h.wrapping_mul(0x9E3779B97F4A7C15);
                    occupied[(h2 >> 33) as usize % occupied.len()]
                };

                let mut cum_g = 0.0f64;
                let mut cum_h = 0.0f64;
                for b in 0..=rand_bin {
                    cum_g += g_hist[b];
                    cum_h += h_hist[b];
                }
                let other_g = g_nm - cum_g;
                let other_h = h_nm - cum_h;

                {
                    let lg = cum_g + g_miss;
                    let lh = cum_h + h_miss;
                    if lh >= min_h && other_h >= min_h {
                        let mono_ok = if mc == 0 {
                            true
                        } else {
                            let lv = -lg / (lh + lambda_reg);
                            let rv = -other_g / (other_h + lambda_reg);
                            if mc > 0 {
                                lv <= rv
                            } else {
                                lv >= rv
                            }
                        };
                        if mono_ok {
                            let mut gain = 0.5
                                * (lg * lg / (lh + lambda_reg)
                                    + other_g * other_g / (other_h + lambda_reg)
                                    - g_sum * g_sum / (h_sum + lambda_reg))
                                - gamma;
                            gain = evidence_adjusted_gain(
                                binned, gain, lh, other_h, h_sum, lambda_reg, n_trials,
                            );
                            gain -= missing_penalty;
                            if gain.is_finite() && gain > best.gain {
                                best = SplitResult::axis(
                                    gain,
                                    feat,
                                    rand_bin,
                                    true,
                                    false,
                                    Vec::new(),
                                );
                            }
                        }
                    }
                }

                {
                    let rg = other_g + g_miss;
                    let rh = other_h + h_miss;
                    if cum_h >= min_h && rh >= min_h {
                        let mono_ok = if mc == 0 {
                            true
                        } else {
                            let lv = -cum_g / (cum_h + lambda_reg);
                            let rv = -rg / (rh + lambda_reg);
                            if mc > 0 {
                                lv <= rv
                            } else {
                                lv >= rv
                            }
                        };
                        if mono_ok {
                            let mut gain = 0.5
                                * (cum_g * cum_g / (cum_h + lambda_reg)
                                    + rg * rg / (rh + lambda_reg)
                                    - g_sum * g_sum / (h_sum + lambda_reg))
                                - gamma;
                            gain = evidence_adjusted_gain(
                                binned, gain, cum_h, rh, h_sum, lambda_reg, n_trials,
                            );
                            gain -= missing_penalty;
                            if gain.is_finite() && gain > best.gain {
                                best = SplitResult::axis(
                                    gain,
                                    feat,
                                    rand_bin,
                                    false,
                                    false,
                                    Vec::new(),
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    best
}

/// Extra-trees split for multi-output (shared tree) mode.
/// For each feature, picks ONE random threshold and computes gain summed across K classes.
pub(super) fn find_extra_trees_split_multi(
    binned: &BinnedData,
    all_gradients: &[f64],
    all_hessians: &[f64],
    n_classes: usize,
    n_rows: usize,
    node_indices: &[u32],
    active_features: &[usize],
    g_sums: &[f64], // [K]
    h_sums: &[f64], // [K]
    lambda_reg: f64,
    gamma: f64,
    min_h: f64,
    g_hists: &mut [f64], // [K * max_bins]
    h_hists: &mut [f64],
    g_miss: &mut [f64], // [K]
    h_miss: &mut [f64],
    node_seed: u64,
    cat_smooth: f64,
) -> SplitResult {
    let max_bins = g_hists.len() / n_classes;
    let mut best = SplitResult::empty();
    let n_random_cuts = 3usize;

    let cat_sort_dir = multiclass_cat_sort_direction(g_sums, h_sums, lambda_reg);

    for (fi, &feat) in active_features.iter().enumerate() {
        let feat_n_bins = binned.n_bins(feat);
        if feat_n_bins <= 1 {
            continue;
        }

        // Build multi-class histograms
        let col_bins = binned.col_bins(feat);
        for k in 0..n_classes {
            let base = k * max_bins;
            for b in 0..feat_n_bins {
                g_hists[base + b] = 0.0;
                h_hists[base + b] = 0.0;
            }
            g_miss[k] = 0.0;
            h_miss[k] = 0.0;
        }
        for &idx in node_indices {
            let bin = col_bins[idx as usize];
            for k in 0..n_classes {
                let g = all_gradients[k * n_rows + idx as usize];
                let h = all_hessians[k * n_rows + idx as usize];
                if bin == MISSING_BIN {
                    g_miss[k] += g;
                    h_miss[k] += h;
                } else {
                    g_hists[k * max_bins + bin as usize] += g;
                    h_hists[k * max_bins + bin as usize] += h;
                }
            }
        }

        // Non-missing sums per class
        let mut g_nm = vec![0.0f64; n_classes];
        let mut h_nm = vec![0.0f64; n_classes];
        for k in 0..n_classes {
            g_nm[k] = g_sums[k] - g_miss[k];
            h_nm[k] = h_sums[k] - h_miss[k];
        }
        let missing_penalty = missing_route_penalty(
            h_sums.iter().copied().sum::<f64>(),
            h_miss.iter().copied().sum::<f64>(),
        );

        if binned.is_categorical[feat] {
            // Categorical: sort by weighted ratio (averaged across classes), pick random partition
            let mut cat_bins: Vec<(usize, Vec<f64>, Vec<f64>)> = Vec::new();
            for bin in 0..feat_n_bins {
                let any_h: f64 = (0..n_classes).map(|k| h_hists[k * max_bins + bin]).sum();
                if any_h > 0.0 {
                    let gv: Vec<f64> = (0..n_classes)
                        .map(|k| g_hists[k * max_bins + bin])
                        .collect();
                    let hv: Vec<f64> = (0..n_classes)
                        .map(|k| h_hists[k * max_bins + bin])
                        .collect();
                    cat_bins.push((bin, gv, hv));
                }
            }
            if cat_bins.len() <= 1 {
                continue;
            }

            let total_h_nm: f64 = h_nm.iter().sum();
            let total_proj_nm: f64 = (0..n_classes).map(|k| g_nm[k] * cat_sort_dir[k]).sum();
            let node_ratio = if total_h_nm > 1e-10 {
                total_proj_nm / total_h_nm
            } else {
                0.0
            };
            let smooth = if cat_smooth > 0.0 { cat_smooth } else { 10.0 };
            cat_bins.sort_by(|a, b| {
                let ga: f64 = (0..n_classes).map(|k| a.1[k] * cat_sort_dir[k]).sum();
                let ha: f64 = a.2.iter().sum();
                let gb: f64 = (0..n_classes).map(|k| b.1[k] * cat_sort_dir[k]).sum();
                let hb: f64 = b.2.iter().sum();
                let ra = (ga + smooth * node_ratio) / (ha + smooth);
                let rb = (gb + smooth * node_ratio) / (hb + smooth);
                ra.partial_cmp(&rb).unwrap_or(std::cmp::Ordering::Equal)
            });

            let n_trials = n_random_cuts.min(cat_bins.len() - 1);
            for trial in 0..n_trials {
                let rand_idx = {
                    let h = node_seed
                        .wrapping_mul(0x9E3779B97F4A7C15)
                        .wrapping_add(fi as u64)
                        .wrapping_add((trial as u64).wrapping_mul(0xD1B54A32D192ED03));
                    let h2 = h.wrapping_mul(0x517CC1B727220A95);
                    (h2 >> 33) as usize % (cat_bins.len() - 1)
                };

                let mut cum_g = vec![0.0f64; n_classes];
                let mut cum_h = vec![0.0f64; n_classes];
                for bin in cat_bins.iter().take(rand_idx + 1) {
                    for k in 0..n_classes {
                        cum_g[k] += bin.1[k];
                        cum_h[k] += bin.2[k];
                    }
                }

                let mut lh_total = 0.0f64;
                let mut rh_total = 0.0f64;
                let mut gain = 0.0f64;
                for k in 0..n_classes {
                    let lg = cum_g[k] + g_miss[k];
                    let lh = cum_h[k] + h_miss[k];
                    let rg = g_nm[k] - cum_g[k];
                    let rh = h_nm[k] - cum_h[k];
                    gain += lg * lg / (lh + lambda_reg) + rg * rg / (rh + lambda_reg)
                        - g_sums[k] * g_sums[k] / (h_sums[k] + lambda_reg);
                    lh_total += lh;
                    rh_total += rh;
                }
                gain = 0.5 * gain - gamma;
                gain = evidence_adjusted_gain(
                    binned,
                    gain,
                    lh_total,
                    rh_total,
                    lh_total + rh_total,
                    lambda_reg,
                    n_trials,
                );
                gain -= missing_penalty;

                if lh_total >= min_h && rh_total >= min_h && gain.is_finite() && gain > best.gain {
                    let mut mask: CatBitmask = Vec::new();
                    for bin in cat_bins.iter().take(rand_idx + 1) {
                        bitmask_set(&mut mask, bin.0);
                    }
                    best = SplitResult::axis(gain, feat, 0, true, true, mask);
                }
            }
        } else {
            // Numeric: pick ONE random occupied bin as threshold
            let occupied: Vec<usize> = (0..feat_n_bins - 1)
                .filter(|&b| {
                    let any_h: f64 = (0..n_classes)
                        .map(|k| h_hists[k * max_bins + b] + h_hists[k * max_bins + b + 1])
                        .sum();
                    any_h > 0.0
                })
                .collect();
            if occupied.is_empty() {
                continue;
            }

            let n_trials = n_random_cuts.min(occupied.len());
            for trial in 0..n_trials {
                let rand_bin = {
                    let h = node_seed
                        .wrapping_mul(0x517CC1B727220A95)
                        .wrapping_add(fi as u64)
                        .wrapping_add((trial as u64).wrapping_mul(0x9FB21C651E98DF25));
                    let h2 = h.wrapping_mul(0x9E3779B97F4A7C15);
                    occupied[(h2 >> 33) as usize % occupied.len()]
                };

                let mut cum_g = vec![0.0f64; n_classes];
                let mut cum_h = vec![0.0f64; n_classes];
                for k in 0..n_classes {
                    let base = k * max_bins;
                    for b in 0..=rand_bin {
                        cum_g[k] += g_hists[base + b];
                        cum_h[k] += h_hists[base + b];
                    }
                }

                {
                    let mut lh_total = 0.0f64;
                    let mut rh_total = 0.0f64;
                    let mut gain = 0.0f64;
                    for k in 0..n_classes {
                        let lg = cum_g[k] + g_miss[k];
                        let lh = cum_h[k] + h_miss[k];
                        let rg = g_nm[k] - cum_g[k];
                        let rh = h_nm[k] - cum_h[k];
                        gain += lg * lg / (lh + lambda_reg) + rg * rg / (rh + lambda_reg)
                            - g_sums[k] * g_sums[k] / (h_sums[k] + lambda_reg);
                        lh_total += lh;
                        rh_total += rh;
                    }
                    gain = 0.5 * gain - gamma;
                    gain = evidence_adjusted_gain(
                        binned,
                        gain,
                        lh_total,
                        rh_total,
                        lh_total + rh_total,
                        lambda_reg,
                        n_trials,
                    );
                    gain -= missing_penalty;
                    if lh_total >= min_h
                        && rh_total >= min_h
                        && gain.is_finite()
                        && gain > best.gain
                    {
                        best = SplitResult::axis(gain, feat, rand_bin, true, false, Vec::new());
                    }
                }

                {
                    let mut lh_total = 0.0f64;
                    let mut rh_total = 0.0f64;
                    let mut gain = 0.0f64;
                    for k in 0..n_classes {
                        let rg = (g_nm[k] - cum_g[k]) + g_miss[k];
                        let rh = (h_nm[k] - cum_h[k]) + h_miss[k];
                        gain += cum_g[k] * cum_g[k] / (cum_h[k] + lambda_reg)
                            + rg * rg / (rh + lambda_reg)
                            - g_sums[k] * g_sums[k] / (h_sums[k] + lambda_reg);
                        lh_total += cum_h[k];
                        rh_total += rh;
                    }
                    gain = 0.5 * gain - gamma;
                    gain = evidence_adjusted_gain(
                        binned,
                        gain,
                        lh_total,
                        rh_total,
                        lh_total + rh_total,
                        lambda_reg,
                        n_trials,
                    );
                    gain -= missing_penalty;
                    if lh_total >= min_h
                        && rh_total >= min_h
                        && gain.is_finite()
                        && gain > best.gain
                    {
                        best = SplitResult::axis(gain, feat, rand_bin, false, false, Vec::new());
                    }
                }
            }
        }
    }

    best
}

/// Parallelism threshold: parallelize feature loop when work exceeds this.
pub(super) const PAR_SPLIT_THRESHOLD: usize = 100_000;

pub(super) fn find_best_split_from_hists_debiased(
    hists: &NodeHists,
    tree_features: &[usize],
    active_features: &[usize],
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    comp_indices: &[u32],
    g_sum: f64,
    h_sum: f64,
    comp_g_sum: f64,
    comp_h_sum: f64,
    lambda_reg: f64,
    gamma: f64,
    min_h: f64,
    random_strength: f64,
    noise_seed: u64,
    cat_smooth: f64,
    monotone_constraints: &[i8],
    gain_penalty: f64,
    complement_debias_mode: u8,
) -> SplitResult {
    let max_bins = hists.max_bins;
    let feat_to_idx =
        |feat: usize| -> Option<usize> { tree_features.iter().position(|&f| f == feat) };
    let empty_split = SplitResult::empty;

    let eval_one = |feat: usize| -> SplitResult {
        let Some(feat_idx) = feat_to_idx(feat) else {
            return empty_split();
        };
        let offset = feat_idx * max_bins;
        let g_slice = &hists.g[offset..offset + max_bins];
        let h_slice = &hists.h[offset..offset + max_bins];
        let mc = if feat < monotone_constraints.len() {
            monotone_constraints[feat]
        } else {
            0
        };
        let mut r = scan_feature_hist(
            feat,
            binned,
            g_slice,
            h_slice,
            hists.g_miss[feat_idx],
            hists.h_miss[feat_idx],
            g_sum,
            h_sum,
            lambda_reg,
            0.0,
            gamma,
            min_h,
            random_strength,
            noise_seed,
            cat_smooth,
            mc,
            gain_penalty,
            false,
        );
        if !(r.gain.is_finite() && r.gain > 0.0) {
            return empty_split();
        }
        let comp_gain = eval_fixed_split_gain(
            binned,
            gradients,
            hessians,
            comp_indices,
            comp_g_sum,
            comp_h_sum,
            feat,
            r.bin,
            r.missing_left,
            r.is_cat,
            &r.cat_mask,
            lambda_reg,
            gamma,
            min_h,
            gain_penalty,
        );
        r.gain = combine_complement_gain(r.gain, comp_gain, complement_debias_mode);
        r
    };

    if active_features.len() >= 4 {
        active_features
            .par_iter()
            .map(|&feat| eval_one(feat))
            .reduce(empty_split, |a, b| if b.gain > a.gain { b } else { a })
    } else {
        let mut best = empty_split();
        for &feat in active_features {
            let r = eval_one(feat);
            if r.gain > best.gain {
                best = r;
            }
        }
        best
    }
}

pub(super) fn find_best_split_debiased(
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    comp_indices: &[u32],
    active_features: &[usize],
    g_sum: f64,
    h_sum: f64,
    comp_g_sum: f64,
    comp_h_sum: f64,
    lambda_reg: f64,
    gamma: f64,
    min_h: f64,
    g_hist: &mut [f64],
    h_hist: &mut [f64],
    random_strength: f64,
    noise_seed: u64,
    cat_smooth: f64,
    monotone_constraints: &[i8],
    gain_penalty: f64,
    complement_debias_mode: u8,
) -> SplitResult {
    let work = active_features.len() * node_indices.len();
    let max_bins = g_hist.len();
    let empty_split = SplitResult::empty;

    if work >= PAR_SPLIT_THRESHOLD && active_features.len() >= 4 {
        active_features
            .par_iter()
            .fold(
                || {
                    (
                        vec![0.0f64; max_bins],
                        vec![0.0f64; max_bins],
                        empty_split(),
                    )
                },
                |(mut lg, mut lh, mut best), &feat| {
                    let mc = if feat < monotone_constraints.len() {
                        monotone_constraints[feat]
                    } else {
                        0
                    };
                    let mut r = eval_feature_split(
                        feat,
                        binned,
                        gradients,
                        hessians,
                        node_indices,
                        g_sum,
                        h_sum,
                        lambda_reg,
                        0.0,
                        gamma,
                        min_h,
                        random_strength,
                        noise_seed,
                        &mut lg,
                        &mut lh,
                        cat_smooth,
                        mc,
                        gain_penalty,
                        false,
                    );
                    if r.gain.is_finite() && r.gain > 0.0 {
                        let comp_gain = eval_fixed_split_gain(
                            binned,
                            gradients,
                            hessians,
                            comp_indices,
                            comp_g_sum,
                            comp_h_sum,
                            feat,
                            r.bin,
                            r.missing_left,
                            r.is_cat,
                            &r.cat_mask,
                            lambda_reg,
                            gamma,
                            min_h,
                            gain_penalty,
                        );
                        r.gain = combine_complement_gain(r.gain, comp_gain, complement_debias_mode);
                    }
                    if r.gain > best.gain {
                        best = r;
                    }
                    (lg, lh, best)
                },
            )
            .map(|(_, _, best)| best)
            .reduce(empty_split, |a, b| if b.gain > a.gain { b } else { a })
    } else {
        let mut best = empty_split();
        for &feat in active_features {
            let mc = if feat < monotone_constraints.len() {
                monotone_constraints[feat]
            } else {
                0
            };
            let mut r = eval_feature_split(
                feat,
                binned,
                gradients,
                hessians,
                node_indices,
                g_sum,
                h_sum,
                lambda_reg,
                0.0,
                gamma,
                min_h,
                random_strength,
                noise_seed,
                g_hist,
                h_hist,
                cat_smooth,
                mc,
                gain_penalty,
                false,
            );
            if r.gain.is_finite() && r.gain > 0.0 {
                let comp_gain = eval_fixed_split_gain(
                    binned,
                    gradients,
                    hessians,
                    comp_indices,
                    comp_g_sum,
                    comp_h_sum,
                    feat,
                    r.bin,
                    r.missing_left,
                    r.is_cat,
                    &r.cat_mask,
                    lambda_reg,
                    gamma,
                    min_h,
                    gain_penalty,
                );
                r.gain = combine_complement_gain(r.gain, comp_gain, complement_debias_mode);
            }
            if r.gain > best.gain {
                best = r;
            }
        }
        best
    }
}

/// Find best split from pre-built NodeHists. Parallel when >= 4 active features.
pub(super) fn find_best_split_from_hists(
    hists: &NodeHists,
    tree_features: &[usize],
    active_features: &[usize],
    binned: &BinnedData,
    g_sum: f64,
    h_sum: f64,
    lambda_reg: f64,
    l1_reg: f64,
    gamma: f64,
    min_h: f64,
    random_strength: f64,
    noise_seed: u64,
    cat_smooth: f64,
    monotone_constraints: &[i8],
    gain_penalty: f64,
    interval_splits: bool,
) -> SplitResult {
    let max_bins = hists.max_bins;

    // Build a lookup from global feature id -> index in tree_features
    // (needed to find the right histogram slice)
    let feat_to_idx =
        |feat: usize| -> Option<usize> { tree_features.iter().position(|&f| f == feat) };

    let empty_split = SplitResult::empty;

    if active_features.len() >= 4 {
        active_features
            .par_iter()
            .fold(empty_split, |mut best, &feat| {
                if let Some(feat_idx) = feat_to_idx(feat) {
                    let offset = feat_idx * max_bins;
                    let g_slice = &hists.g[offset..offset + max_bins];
                    let h_slice = &hists.h[offset..offset + max_bins];
                    let mc = if feat < monotone_constraints.len() {
                        monotone_constraints[feat]
                    } else {
                        0
                    };
                    let r = scan_feature_hist(
                        feat,
                        binned,
                        g_slice,
                        h_slice,
                        hists.g_miss[feat_idx],
                        hists.h_miss[feat_idx],
                        g_sum,
                        h_sum,
                        lambda_reg,
                        l1_reg,
                        gamma,
                        min_h,
                        random_strength,
                        noise_seed,
                        cat_smooth,
                        mc,
                        gain_penalty,
                        interval_splits,
                    );
                    if r.gain > best.gain {
                        best = r;
                    }
                }
                best
            })
            .reduce(empty_split, |a, b| if b.gain > a.gain { b } else { a })
    } else {
        let mut best = empty_split();
        for &feat in active_features {
            if let Some(feat_idx) = feat_to_idx(feat) {
                let offset = feat_idx * max_bins;
                let g_slice = &hists.g[offset..offset + max_bins];
                let h_slice = &hists.h[offset..offset + max_bins];
                let mc = if feat < monotone_constraints.len() {
                    monotone_constraints[feat]
                } else {
                    0
                };
                let r = scan_feature_hist(
                    feat,
                    binned,
                    g_slice,
                    h_slice,
                    hists.g_miss[feat_idx],
                    hists.h_miss[feat_idx],
                    g_sum,
                    h_sum,
                    lambda_reg,
                    l1_reg,
                    gamma,
                    min_h,
                    random_strength,
                    noise_seed,
                    cat_smooth,
                    mc,
                    gain_penalty,
                    interval_splits,
                );
                if r.gain > best.gain {
                    best = r;
                }
            }
        }
        best
    }
}

// ── GGFP v5.0 — JIT-CatPairSplit ────────────────────────────────────────────

/// Config for JIT-CatPairSplit search.
#[derive(Clone, Copy, Debug)]
pub struct CatPairConfig {
    pub enabled: bool,
    pub top_k_cat: usize,
    pub k_buckets: u8,
    pub min_node_rows: usize,
    pub max_node_depth: usize,
    pub gain_margin: f64,
}

impl Default for CatPairConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            top_k_cat: 4,
            k_buckets: 8,
            min_node_rows: 512,
            max_node_depth: 2,
            gain_margin: 1.05,
        }
    }
}

/// Evaluate JIT cat-pair splits at one node. Returns the best pair SplitResult
/// or `SplitResult::empty()` if none beat the gain margin.
pub(super) fn eval_cat_pair_jit_for_node(
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    active_features: &[usize],
    g_sum: f64,
    h_sum: f64,
    lambda_reg: f64,
    l1_reg: f64,
    gamma: f64,
    min_h: f64,
    cat_smooth: f64,
    depth: usize,
    raw_best_gain: f64,
    cfg: &CatPairConfig,
) -> SplitResult {
    if !cfg.enabled || depth > cfg.max_node_depth || node_indices.len() < cfg.min_node_rows {
        return SplitResult::empty();
    }

    let cat_feats: Vec<usize> = active_features
        .iter()
        .copied()
        .filter(|&f| f < binned.is_categorical.len() && binned.is_categorical[f])
        .collect();
    if cat_feats.len() < 2 {
        return SplitResult::empty();
    }

    // Step 1: score each categorical feat by its single-feature gain at this
    // node. Keep a sorted-by-gradient mapping for bucket assignment.
    let mut scored: Vec<(usize, f64, Vec<(usize, f64, f64)>)> = Vec::new();
    for &f in &cat_feats {
        let n_bins = binned.n_bins(f);
        if n_bins < 2 {
            continue;
        }
        let col = binned.col_bins(f);
        let mut g_h: Vec<(usize, f64, f64)> = (0..n_bins).map(|b| (b, 0.0, 0.0)).collect();
        let mut g_miss = 0.0f64;
        let mut h_miss = 0.0f64;
        for &idx in node_indices {
            let b = col[idx as usize];
            if b == MISSING_BIN {
                g_miss += gradients[idx as usize];
                h_miss += hessians[idx as usize];
            } else {
                let bu = b as usize;
                g_h[bu].1 += gradients[idx as usize];
                g_h[bu].2 += hessians[idx as usize];
            }
        }
        g_h.retain(|t| t.2 > 0.0);
        if g_h.len() < 2 {
            continue;
        }
        let g_nm = g_sum - g_miss;
        let h_nm = h_sum - h_miss;
        let node_ratio = if h_nm > 1e-10 { g_nm / h_nm } else { 0.0 };
        g_h.sort_by(|a, b| {
            let ra = (a.1 + cat_smooth * node_ratio) / (a.2 + cat_smooth);
            let rb = (b.1 + cat_smooth * node_ratio) / (b.2 + cat_smooth);
            ra.partial_cmp(&rb).unwrap_or(Ordering::Equal)
        });
        let mut cum_g = 0.0f64;
        let mut cum_h = 0.0f64;
        let mut best = f64::NEG_INFINITY;
        for i in 0..g_h.len() - 1 {
            cum_g += g_h[i].1;
            cum_h += g_h[i].2;
            let other_g = g_nm - cum_g;
            let other_h = h_nm - cum_h;
            if cum_h >= min_h && other_h >= min_h {
                let gain = 0.5
                    * (l1_gain_score(cum_g, cum_h, lambda_reg, l1_reg)
                        + l1_gain_score(other_g, other_h, lambda_reg, l1_reg)
                        - l1_gain_score(g_sum, h_sum, lambda_reg, l1_reg))
                    - gamma;
                let gain = evidence_adjusted_gain(
                    binned,
                    gain,
                    cum_h,
                    other_h,
                    h_sum,
                    lambda_reg,
                    g_h.len().saturating_sub(1),
                );
                if gain.is_finite() && gain > best {
                    best = gain;
                }
            }
        }
        if best.is_finite() {
            scored.push((f, best, g_h));
        }
    }
    if scored.len() < 2 {
        return SplitResult::empty();
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    let top_n = scored.len().min(cfg.top_k_cat);

    // Step 2: build bucket maps (raw_bin -> bucket 0..k-1) using equal-rank on
    // gradient-sorted categories.
    let k = cfg.k_buckets as usize;
    let cells_total = k * k;
    if cells_total > 64 {
        return SplitResult::empty(); // u64 mask only fits 64 cells
    }
    let mut bucket_maps: Vec<(usize, Vec<u8>)> = Vec::with_capacity(top_n);
    for entry in scored.iter().take(top_n) {
        let (f, _gain, ref g_h) = *entry;
        let n_bins = binned.n_bins(f);
        let mut map = vec![0u8; n_bins];
        let n_cats = g_h.len();
        for (rank, t) in g_h.iter().enumerate() {
            let bu = ((rank * k) / n_cats.max(1)).min(k - 1) as u8;
            map[t.0] = bu;
        }
        bucket_maps.push((f, map));
    }

    // Step 3: for each pair, build kxk cell histogram, sort cells by G/H,
    // prefix-scan for the best split, accept if it beats raw_best * margin.
    let mut best_pair = SplitResult::empty();
    for i in 0..bucket_maps.len() {
        for j in (i + 1)..bucket_maps.len() {
            let (f1, ref map_a) = bucket_maps[i];
            let (f2, ref map_b) = bucket_maps[j];
            let col1 = binned.col_bins(f1);
            let col2 = binned.col_bins(f2);

            let mut cell_g = [0.0f64; 64];
            let mut cell_h = [0.0f64; 64];
            let mut g_miss = 0.0f64;
            let mut h_miss = 0.0f64;
            for &idx in node_indices {
                let iu = idx as usize;
                let b1 = col1[iu];
                let b2 = col2[iu];
                if b1 == MISSING_BIN || b2 == MISSING_BIN {
                    g_miss += gradients[iu];
                    h_miss += hessians[iu];
                    continue;
                }
                let bu1 = map_a[b1 as usize] as usize;
                let bu2 = map_b[b2 as usize] as usize;
                let cell = bu1 * k + bu2;
                cell_g[cell] += gradients[iu];
                cell_h[cell] += hessians[iu];
            }
            let g_nm = g_sum - g_miss;
            let h_nm = h_sum - h_miss;
            let node_ratio = if h_nm > 1e-10 { g_nm / h_nm } else { 0.0 };

            let mut order: Vec<usize> = (0..cells_total).filter(|&c| cell_h[c] > 0.0).collect();
            if order.len() < 2 {
                continue;
            }
            order.sort_by(|&a, &b| {
                let ra = (cell_g[a] + cat_smooth * node_ratio) / (cell_h[a] + cat_smooth);
                let rb = (cell_g[b] + cat_smooth * node_ratio) / (cell_h[b] + cat_smooth);
                ra.partial_cmp(&rb).unwrap_or(Ordering::Equal)
            });

            let mut cum_g = 0.0f64;
            let mut cum_h = 0.0f64;
            let mut best_gain = f64::NEG_INFINITY;
            let mut best_split_idx = 0usize;
            let mut best_missing_left = true;
            for idx in 0..order.len() - 1 {
                let c = order[idx];
                cum_g += cell_g[c];
                cum_h += cell_h[c];
                let other_g = g_nm - cum_g;
                let other_h = h_nm - cum_h;

                let lg = cum_g + g_miss;
                let lh = cum_h + h_miss;
                if lh >= min_h && other_h >= min_h {
                    let gain = 0.5
                        * (l1_gain_score(lg, lh, lambda_reg, l1_reg)
                            + l1_gain_score(other_g, other_h, lambda_reg, l1_reg)
                            - l1_gain_score(g_sum, h_sum, lambda_reg, l1_reg))
                        - gamma;
                    let gain = evidence_adjusted_gain(
                        binned,
                        gain,
                        lh,
                        other_h,
                        h_sum,
                        lambda_reg,
                        order.len().saturating_sub(1),
                    );
                    if gain.is_finite() && gain > best_gain {
                        best_gain = gain;
                        best_split_idx = idx;
                        best_missing_left = true;
                    }
                }
                let rg = other_g + g_miss;
                let rh = other_h + h_miss;
                if cum_h >= min_h && rh >= min_h {
                    let gain = 0.5
                        * (l1_gain_score(cum_g, cum_h, lambda_reg, l1_reg)
                            + l1_gain_score(rg, rh, lambda_reg, l1_reg)
                            - l1_gain_score(g_sum, h_sum, lambda_reg, l1_reg))
                        - gamma;
                    let gain = evidence_adjusted_gain(
                        binned,
                        gain,
                        cum_h,
                        rh,
                        h_sum,
                        lambda_reg,
                        order.len().saturating_sub(1),
                    );
                    if gain.is_finite() && gain > best_gain {
                        best_gain = gain;
                        best_split_idx = idx;
                        best_missing_left = false;
                    }
                }
            }
            if !best_gain.is_finite() {
                continue;
            }
            // Strict margin vs raw best; also must beat any prior pair best.
            let raw_threshold = if raw_best_gain.is_finite() && raw_best_gain > 0.0 {
                raw_best_gain * cfg.gain_margin
            } else {
                0.0
            };
            if best_gain <= raw_threshold {
                continue;
            }
            if best_gain <= best_pair.gain {
                continue;
            }

            let mut mask: u64 = 0;
            for idx2 in 0..=best_split_idx {
                let c = order[idx2];
                mask |= 1u64 << c;
            }
            best_pair = SplitResult::cat_pair(
                best_gain,
                f1,
                f2 as u32,
                map_a.clone(),
                map_b.clone(),
                mask,
                cfg.k_buckets,
                best_missing_left,
            );
        }
    }
    best_pair
}

/// GGFP v5.0 wrapper: runs `find_best_split` then optionally augments with a
/// JIT-CatPairSplit candidate. If cat-pair beats raw_best * gain_margin, it wins.
pub(super) fn find_best_split_v5(
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    active_features: &[usize],
    g_sum: f64,
    h_sum: f64,
    lambda_reg: f64,
    l1_reg: f64,
    gamma: f64,
    min_h: f64,
    g_hist: &mut [f64],
    h_hist: &mut [f64],
    random_strength: f64,
    noise_seed: u64,
    cat_smooth: f64,
    monotone_constraints: &[i8],
    gain_penalty: f64,
    interval_splits: bool,
    cat_pair_cfg: &CatPairConfig,
    depth: usize,
) -> SplitResult {
    let raw_best = find_best_split(
        binned,
        gradients,
        hessians,
        node_indices,
        active_features,
        g_sum,
        h_sum,
        lambda_reg,
        l1_reg,
        gamma,
        min_h,
        g_hist,
        h_hist,
        random_strength,
        noise_seed,
        cat_smooth,
        monotone_constraints,
        gain_penalty,
        interval_splits,
    );
    if !cat_pair_cfg.enabled || raw_best.is_oblique {
        return raw_best;
    }
    let pair = eval_cat_pair_jit_for_node(
        binned,
        gradients,
        hessians,
        node_indices,
        active_features,
        g_sum,
        h_sum,
        lambda_reg,
        l1_reg,
        gamma,
        min_h,
        cat_smooth,
        depth,
        raw_best.gain,
        cat_pair_cfg,
    );
    if pair.gain > raw_best.gain {
        pair
    } else {
        raw_best
    }
}

pub(super) fn find_best_split(
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    active_features: &[usize],
    g_sum: f64,
    h_sum: f64,
    lambda_reg: f64,
    l1_reg: f64,
    gamma: f64,
    min_h: f64,
    g_hist: &mut [f64],
    h_hist: &mut [f64],
    random_strength: f64,
    noise_seed: u64,
    cat_smooth: f64,
    monotone_constraints: &[i8],
    gain_penalty: f64,
    interval_splits: bool,
) -> SplitResult {
    let work = active_features.len() * node_indices.len();
    let max_bins = g_hist.len();

    if work >= PAR_SPLIT_THRESHOLD && active_features.len() >= 4 {
        // ── Parallel path: fold reuses histogram buffers per-thread ──
        let empty_split = SplitResult::empty;
        active_features
            .par_iter()
            .fold(
                || {
                    (
                        vec![0.0f64; max_bins],
                        vec![0.0f64; max_bins],
                        empty_split(),
                    )
                },
                |(mut lg, mut lh, mut best), &feat| {
                    let mc = if feat < monotone_constraints.len() {
                        monotone_constraints[feat]
                    } else {
                        0
                    };
                    let r = eval_feature_split(
                        feat,
                        binned,
                        gradients,
                        hessians,
                        node_indices,
                        g_sum,
                        h_sum,
                        lambda_reg,
                        l1_reg,
                        gamma,
                        min_h,
                        random_strength,
                        noise_seed,
                        &mut lg,
                        &mut lh,
                        cat_smooth,
                        mc,
                        gain_penalty,
                        interval_splits,
                    );
                    if r.gain > best.gain {
                        best = r;
                    }
                    (lg, lh, best)
                },
            )
            .map(|(_, _, best)| best)
            .reduce(empty_split, |a, b| if b.gain > a.gain { b } else { a })
    } else {
        // ── Sequential path: reuse caller's histogram buffers ──
        let mut best = SplitResult::empty();
        for &feat in active_features {
            let mc = if feat < monotone_constraints.len() {
                monotone_constraints[feat]
            } else {
                0
            };
            let r = eval_feature_split(
                feat,
                binned,
                gradients,
                hessians,
                node_indices,
                g_sum,
                h_sum,
                lambda_reg,
                l1_reg,
                gamma,
                min_h,
                random_strength,
                noise_seed,
                g_hist,
                h_hist,
                cat_smooth,
                mc,
                gain_penalty,
                interval_splits,
            );
            if r.gain > best.gain {
                best = r;
            }
        }
        best
    }
}

/// Multi-output find_best_split: evaluates all features using summed gains across K classes.
pub(super) fn find_best_split_multi(
    binned: &BinnedData,
    all_gradients: &[f64],
    all_hessians: &[f64],
    all_probs: &[f64],
    n_classes: usize,
    n_rows: usize,
    node_indices: &[u32],
    active_features: &[usize],
    g_sums: &[f64],
    h_sums: &[f64],
    lambda_reg: f64,
    gamma: f64,
    min_h: f64,
    g_hists: &mut [f64],
    h_hists: &mut [f64],
    g_miss: &mut [f64],
    h_miss: &mut [f64],
    p_hists: &mut [f64],
    pp_hists: &mut [f64],
    p_miss: &mut [f64],
    pp_miss: &mut [f64],
    random_strength: f64,
    noise_seed: u64,
    cat_smooth: f64,
    gain_penalty: f64,
    coupled_split_gain: bool,
) -> SplitResult {
    let work = active_features.len() * node_indices.len();
    let max_bins = g_hists.len() / n_classes;
    let use_coupled_gain =
        coupled_split_gain && n_classes >= 3 && all_probs.len() >= n_rows * n_classes;
    let mut parent_p_sums = vec![0.0f64; n_classes];
    let mut parent_pp_sums = vec![0.0f64; n_classes * n_classes];
    let mut parent_dense_gain = 0.0f64;
    if use_coupled_gain {
        for &idx in node_indices {
            let row = idx as usize;
            let prob_base = row * n_classes;
            for a in 0..n_classes {
                let pa = all_probs[prob_base + a];
                parent_p_sums[a] += pa;
                let row_base = a * n_classes;
                for b in 0..n_classes {
                    parent_pp_sums[row_base + b] += pa * all_probs[prob_base + b];
                }
            }
        }
        let mut dense_a = vec![0.0f64; n_classes * n_classes];
        let mut dense_rhs = vec![0.0f64; n_classes];
        parent_dense_gain = dense_multiclass_gain(
            g_sums,
            &parent_p_sums,
            &parent_pp_sums,
            lambda_reg,
            &mut dense_a,
            &mut dense_rhs,
        );
    }
    let cat_sort_dir = if use_coupled_gain {
        multiclass_cat_sort_direction_dense(g_sums, &parent_p_sums, &parent_pp_sums, lambda_reg)
    } else {
        multiclass_cat_sort_direction(g_sums, h_sums, lambda_reg)
    };

    if !use_coupled_gain && work >= PAR_SPLIT_THRESHOLD && active_features.len() >= 4 {
        let empty_split = SplitResult::empty;
        active_features
            .par_iter()
            .fold(
                || {
                    (
                        vec![0.0f64; n_classes * max_bins],
                        vec![0.0f64; n_classes * max_bins],
                        vec![0.0f64; n_classes],
                        vec![0.0f64; n_classes],
                        empty_split(),
                    )
                },
                |(mut lg, mut lh, mut gm, mut hm, mut best), &feat| {
                    let r = eval_feature_split_multi(
                        feat,
                        binned,
                        all_gradients,
                        all_hessians,
                        all_probs,
                        n_classes,
                        n_rows,
                        node_indices,
                        g_sums,
                        h_sums,
                        lambda_reg,
                        gamma,
                        min_h,
                        random_strength,
                        noise_seed,
                        &mut lg,
                        &mut lh,
                        &mut gm,
                        &mut hm,
                        &mut [],
                        &mut [],
                        &mut [],
                        &mut [],
                        &cat_sort_dir,
                        cat_smooth,
                        gain_penalty,
                        false,
                        &[],
                        &[],
                        0.0,
                    );
                    if r.gain > best.gain {
                        best = r;
                    }
                    (lg, lh, gm, hm, best)
                },
            )
            .map(|(_, _, _, _, best)| best)
            .reduce(empty_split, |a, b| if b.gain > a.gain { b } else { a })
    } else {
        let mut best = SplitResult::empty();
        for &feat in active_features {
            let r = eval_feature_split_multi(
                feat,
                binned,
                all_gradients,
                all_hessians,
                all_probs,
                n_classes,
                n_rows,
                node_indices,
                g_sums,
                h_sums,
                lambda_reg,
                gamma,
                min_h,
                random_strength,
                noise_seed,
                g_hists,
                h_hists,
                g_miss,
                h_miss,
                p_hists,
                pp_hists,
                p_miss,
                pp_miss,
                &cat_sort_dir,
                cat_smooth,
                gain_penalty,
                use_coupled_gain,
                &parent_p_sums,
                &parent_pp_sums,
                parent_dense_gain,
            );
            if r.gain > best.gain {
                best = r;
            }
        }
        best
    }
}

pub(super) fn partition_indices(
    row_buf: &mut [u32],
    start: usize,
    end: usize,
    binned: &BinnedData,
    feat: usize,
    split_bin: u16,
    missing_goes_left: bool,
    is_cat: bool,
    cat_mask: &[u64],
) -> usize {
    let col_bins = binned.col_bins(feat);
    let mut left_end = start;
    let mut i = start;
    while i < end {
        let bin = col_bins[row_buf[i] as usize];
        let goes_left = if bin == MISSING_BIN {
            missing_goes_left
        } else if is_cat {
            bitmask_test(cat_mask, bin as usize)
        } else {
            bin <= split_bin
        };
        if goes_left {
            row_buf.swap(i, left_end);
            left_end += 1;
        }
        i += 1;
    }
    left_end
}

pub(super) fn partition_indices_split(
    row_buf: &mut [u32],
    start: usize,
    end: usize,
    binned: &BinnedData,
    sr: &SplitResult,
) -> usize {
    let mut left_end = start;
    let mut i = start;
    while i < end {
        if split_goes_left_binned(sr, binned, row_buf[i] as usize) {
            row_buf.swap(i, left_end);
            left_end += 1;
        }
        i += 1;
    }
    left_end
}
