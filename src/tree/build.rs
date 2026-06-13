//! Tree-construction entry points: the six `DecisionTree::build_*` methods.
//!
//! Each builder picks a different growth policy / objective:
//!
//! - `build_depthwise` — classic GBDT depth-by-depth growth (binary/regression).
//! - `build_depthwise_debiased` — depthwise + complement-debiased gain
//!   (CDSS variants for honest mode).
//! - `build_depthwise_multi` — depthwise multi-output (shared structure
//!   across K classes).
//! - `build_oblivious` — symmetric splits at each depth (CatBoost-style).
//! - `build_oblivious_multi` — oblivious K-class shared structure.
//! - `build_leafwise` — best-first growth with priority queue (LightGBM-style).
//!
//! The actual split-finding, histogram building, and per-node expert
//! evaluation live in `super::algorithms`. These methods are the
//! orchestration: they manage indices, recursion, and `TreeBuilder`
//! population, then convert to a frozen `DecisionTree`.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::algorithms::*;
use super::*;

#[inline]
fn adaptive_structural_min_gain(
    depth: usize,
    max_depth: usize,
    node_rows: usize,
    h_sum: f64,
    root_h: f64,
    active_features: usize,
    split: &SplitResult,
) -> f64 {
    if max_depth == 0 || node_rows <= 2 || !h_sum.is_finite() || h_sum <= 0.0 {
        return 0.0;
    }

    let root_h = root_h.max(h_sum).max(1e-12);
    let support = (h_sum / root_h).clamp(1e-9, 1.0);
    let n = (node_rows as f64).max(2.0);
    let depth_frac = ((depth + 1) as f64 / max_depth.max(1) as f64).clamp(0.0, 1.0);
    let search_width = (active_features.max(2) as f64).ln();

    // More flexible split families have a larger winner's-curse surface.  The
    // policy stays generic: it prices structural degrees of freedom, not a
    // dataset or metric.
    let split_df = if split.is_cat_pair {
        2.2
    } else if split.is_oblique {
        1.8
    } else if split.is_cat {
        1.35
    } else {
        1.0
    };

    // Mild MDL-style gate.  It grows with search width and depth, and grows
    // quickly when a node carries little Hessian support.  The small constant
    // keeps this as a structure selector rather than a second loss function.
    let mdl = 0.020 * split_df * (0.5 * n.ln() + 0.35 * search_width);
    mdl * (1.0 + 1.5 * depth_frac * depth_frac) / support.sqrt()
}

#[inline]
fn adaptive_child_h_sums(
    binned: &BinnedData,
    hessians: &[f64],
    node_indices: &[u32],
    h_sum: f64,
    split: &SplitResult,
) -> Option<(f64, f64)> {
    if split.child_h_left.is_finite() {
        let lh = split.child_h_left;
        let rh = h_sum - lh;
        return (lh.is_finite() && rh.is_finite()).then_some((lh, rh));
    }

    let mut lh = 0.0f64;
    for &idx in node_indices {
        let row = idx as usize;
        if split_goes_left_binned(split, binned, row) {
            lh += hessians[row];
        }
    }
    let rh = h_sum - lh;
    (lh.is_finite() && rh.is_finite()).then_some((lh, rh))
}

#[inline]
fn leafwise_heap_priority(
    split_utility: f64,
    trunk1_balanced: bool,
    binned: &BinnedData,
    hessians: &[f64],
    node_indices: &[u32],
    h_sum: f64,
    split: &SplitResult,
) -> f64 {
    if !trunk1_balanced || split_utility <= 0.0 || !split_utility.is_finite() {
        return split_utility;
    }
    let Some((lh, rh)) = adaptive_child_h_sums(binned, hessians, node_indices, h_sum, split) else {
        return split_utility;
    };
    if h_sum <= 0.0 || lh <= 0.0 || rh <= 0.0 {
        return split_utility;
    }
    let lp = (lh / h_sum).clamp(1e-9, 1.0);
    let rp = (rh / h_sum).clamp(1e-9, 1.0);
    let balance = (4.0 * lp * rp).clamp(1e-4, 1.0);

    // Semi-oblivious trunk policy found by the architecture lab: after the
    // unavoidable root split, spend leaves on high-gain splits whose children
    // both retain support. This is a fixed growth law, not a dataset-specific
    // controller.
    split_utility * (0.25 + balance)
}

#[inline]
fn gradient_reliability(sum_g: f64, sum_g2: f64, n: f64, strength: f64) -> f64 {
    if strength <= 0.0 || n < 2.0 {
        return 1.0;
    }
    let mean = sum_g / n;
    let var = (sum_g2 / n - mean * mean).max(0.0);
    let denom = mean * mean + var / n;
    if denom <= 1e-30 || !denom.is_finite() {
        return 1.0;
    }
    (mean * mean / denom).clamp(0.0, 1.0).powf(strength)
}

#[inline]
fn shrink_gain_factor(shrink: f64) -> f64 {
    let a = shrink.clamp(0.0, 1.0);
    (2.0 * a - a * a).clamp(0.0, 1.0)
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn post_shrink_gain_ratio_from_stats(
    pg: f64,
    ph: f64,
    pg2: f64,
    pn: f64,
    lg: f64,
    lh: f64,
    lg2: f64,
    ln: f64,
    rg: f64,
    rh: f64,
    rg2: f64,
    rn: f64,
    lambda_reg: f64,
    l1_reg: f64,
    min_h: f64,
    leaf_var_shrink: f64,
) -> Option<f64> {
    if leaf_var_shrink <= 0.0 || pn < 8.0 {
        return None;
    }
    if lh < min_h || rh < min_h || ln < 1.0 || rn < 1.0 {
        return None;
    }

    let left_score = l1_gain_score(lg, lh, lambda_reg, l1_reg);
    let right_score = l1_gain_score(rg, rh, lambda_reg, l1_reg);
    let parent_score = l1_gain_score(pg, ph, lambda_reg, l1_reg);
    let raw = 0.5 * (left_score + right_score - parent_score);
    if raw <= 1e-12 || !raw.is_finite() {
        return None;
    }

    let lf = shrink_gain_factor(gradient_reliability(lg, lg2, ln, leaf_var_shrink));
    let rf = shrink_gain_factor(gradient_reliability(rg, rg2, rn, leaf_var_shrink));
    let pf = shrink_gain_factor(gradient_reliability(pg, pg2, pn, leaf_var_shrink));
    let post = 0.5 * (lf * left_score + rf * right_score - pf * parent_score);
    if !post.is_finite() {
        return None;
    }
    Some((post / raw).clamp(0.0, 1.0))
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn post_shrink_split_gain_ratio(
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    g_sum: f64,
    h_sum: f64,
    lambda_reg: f64,
    l1_reg: f64,
    min_h: f64,
    split: &SplitResult,
    leaf_var_shrink: f64,
) -> Option<f64> {
    if leaf_var_shrink <= 0.0 || split.gain <= 0.0 || node_indices.len() < 8 {
        return None;
    }

    let mut lg = 0.0f64;
    let mut lh = 0.0f64;
    let mut lg2 = 0.0f64;
    let mut ln = 0.0f64;
    let mut pg2 = 0.0f64;
    let mut pn = 0.0f64;
    for &idx in node_indices {
        let row = idx as usize;
        let g = gradients[row];
        pg2 += g * g;
        pn += 1.0;
        if split_goes_left_binned(split, binned, row) {
            lg += g;
            lh += hessians[row];
            lg2 += g * g;
            ln += 1.0;
        }
    }
    let rg = g_sum - lg;
    let rh = h_sum - lh;
    let rn = pn - ln;
    let rg2 = pg2 - lg2;
    post_shrink_gain_ratio_from_stats(
        g_sum,
        h_sum,
        pg2,
        pn,
        lg,
        lh,
        lg2,
        ln,
        rg,
        rh,
        rg2,
        rn,
        lambda_reg,
        l1_reg,
        min_h,
        leaf_var_shrink,
    )
}

#[allow(clippy::too_many_arguments)]
fn score_oblivious_multi_split(
    binned: &BinnedData,
    all_gradients: &[f64],
    all_hessians: &[f64],
    all_probs: &[f64],
    n_classes: usize,
    n_rows: usize,
    row_buf: &[u32],
    node_ranges: &[(usize, usize)],
    node_g: &[f64],
    node_h: &[f64],
    node_p: &[f64],
    node_pp: &[f64],
    node_parent_obj: &[f64],
    lambda_reg: f64,
    gamma: f64,
    min_h: f64,
    gain_penalty: f64,
    use_coupled_gain: bool,
    split: &SplitResult,
) -> f64 {
    let mut total_gain = 0.0f64;
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

    for (ni, &(start, end)) in node_ranges.iter().enumerate() {
        if start == end {
            continue;
        }
        let g_base = ni * n_classes;
        let pp_base = ni * n_classes * n_classes;
        let mut left_g = vec![0.0f64; n_classes];
        let mut left_h = vec![0.0f64; n_classes];
        let mut left_p = if use_coupled_gain {
            vec![0.0f64; n_classes]
        } else {
            Vec::new()
        };
        let mut left_pp = if use_coupled_gain {
            vec![0.0f64; n_classes * n_classes]
        } else {
            Vec::new()
        };

        for &idx in &row_buf[start..end] {
            let row = idx as usize;
            if !split_goes_left_binned(split, binned, row) {
                continue;
            }
            let prob_base = row * n_classes;
            for k in 0..n_classes {
                let off = k * n_rows + row;
                left_g[k] += all_gradients[off];
                left_h[k] += all_hessians[off];
                if use_coupled_gain {
                    left_p[k] += all_probs[prob_base + k];
                }
            }
            if use_coupled_gain {
                for a in 0..n_classes {
                    let pa = all_probs[prob_base + a];
                    let row_base = a * n_classes;
                    for b in 0..n_classes {
                        left_pp[row_base + b] += pa * all_probs[prob_base + b];
                    }
                }
            }
        }

        let mut right_g = vec![0.0f64; n_classes];
        let mut total_lh = 0.0f64;
        let mut total_rh = 0.0f64;
        let mut gain = if use_coupled_gain {
            let mut right_p = vec![0.0f64; n_classes];
            let mut right_pp = vec![0.0f64; n_classes * n_classes];
            for k in 0..n_classes {
                right_g[k] = node_g[g_base + k] - left_g[k];
                right_p[k] = node_p[g_base + k] - left_p[k];
                total_lh += left_h[k];
                total_rh += node_h[g_base + k] - left_h[k];
            }
            for kk in 0..(n_classes * n_classes) {
                right_pp[kk] = node_pp[pp_base + kk] - left_pp[kk];
            }
            if total_lh < min_h || total_rh < min_h {
                continue;
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
            0.5 * (left_obj + right_obj - node_parent_obj[ni]) - gamma
        } else {
            let mut raw = -node_parent_obj[ni];
            for k in 0..n_classes {
                let lh = left_h[k];
                let rh = node_h[g_base + k] - lh;
                right_g[k] = node_g[g_base + k] - left_g[k];
                total_lh += lh;
                total_rh += rh;
                raw += left_g[k] * left_g[k] / (lh + lambda_reg)
                    + right_g[k] * right_g[k] / (rh + lambda_reg);
            }
            if total_lh < min_h || total_rh < min_h {
                continue;
            }
            0.5 * raw - gamma
        };

        if gain_penalty > 0.0 {
            let mut pen = 0.0f64;
            for k in 0..n_classes {
                let lh = left_h[k];
                let rh = node_h[g_base + k] - lh;
                pen += 1.0 / (lh + lambda_reg) + 1.0 / (rh + lambda_reg)
                    - 1.0 / (node_h[g_base + k] + lambda_reg);
            }
            gain -= gain_penalty * 0.5 * pen;
        }
        total_gain += gain;
    }

    total_gain
}

#[inline]
fn adaptive_balance_multiplier(lh: f64, rh: f64, h_sum: f64) -> f64 {
    if h_sum <= 0.0 || lh <= 0.0 || rh <= 0.0 {
        return f64::INFINITY;
    }
    let lp = (lh / h_sum).clamp(1e-9, 1.0);
    let rp = (rh / h_sum).clamp(1e-9, 1.0);
    let balance = (4.0 * lp * rp).clamp(1e-4, 1.0);

    // A balanced split pays the base structural cost.  A skinny split may still
    // grow, but only when the gain is strong enough to justify a fragile branch.
    1.0 + 0.25 * (balance.sqrt().recip() - 1.0)
}

#[inline]
fn adaptive_risk_weight(
    depth: usize,
    max_depth: usize,
    h_sum: f64,
    root_h: f64,
    lh: f64,
    rh: f64,
    split: &SplitResult,
) -> f64 {
    let depth_frac = ((depth + 1) as f64 / max_depth.max(1) as f64).clamp(0.0, 1.0);
    let support = (h_sum / root_h.max(h_sum).max(1e-12)).clamp(1e-9, 1.0);
    let balance = if h_sum > 0.0 {
        let lp = (lh / h_sum).clamp(1e-9, 1.0);
        let rp = (rh / h_sum).clamp(1e-9, 1.0);
        (4.0 * lp * rp).clamp(1e-4, 1.0)
    } else {
        1e-4
    };
    let family_risk = if split.is_cat_pair {
        0.75
    } else if split.is_oblique {
        0.55
    } else if split.is_cat {
        0.35
    } else {
        0.15
    };

    let support_risk = 1.0 - support.sqrt();
    let balance_risk = 1.0 - balance.sqrt();
    (0.25 * depth_frac + 0.25 * support_risk + 0.20 * balance_risk + 0.15 * family_risk)
        .clamp(0.0, 0.60)
}

#[inline]
fn path_feature_budget(path_budget: &[(usize, f64)], feat: usize) -> f64 {
    path_budget
        .iter()
        .find_map(|&(f, v)| (f == feat).then_some(v))
        .unwrap_or(0.0)
}

#[inline]
fn path_region_budget(path_budget: &[(usize, f64)]) -> f64 {
    path_feature_budget(path_budget, usize::MAX)
}

#[inline]
fn adaptive_apply_feature_budget(structural_cost: f64, feature_budget: f64, depth: usize) -> f64 {
    if structural_cost <= 0.0 || !structural_cost.is_finite() {
        return structural_cost;
    }

    // Feature-budgeted adaptive growth: after a path repeatedly proves the
    // same feature useful, make further cuts on that feature cheaper. New
    // features in deep nodes do not get that discount, so the tree can model
    // sharp univariate boundaries without freely opening high-order
    // interactions on low-support regions.
    let earned = feature_budget.clamp(0.0, 4.0);
    let repeat_discount = 1.0 / (1.0 + 0.22 * earned);
    let novelty_penalty = if depth >= 2 && earned < 0.15 {
        1.0 + 0.08 * ((depth - 1) as f64).powi(2)
    } else {
        1.0
    };
    structural_cost * repeat_discount * novelty_penalty
}

#[inline]
fn crossfit_split_pseudo_gain(
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    g_sum: f64,
    h_sum: f64,
    lambda_reg: f64,
    l1_reg: f64,
    min_h: f64,
    split: &SplitResult,
    tree_seed: u64,
    node_idx: usize,
    depth: usize,
) -> Option<f64> {
    if node_indices.len() < 48 {
        return None;
    }
    let audit_seed = tree_seed
        ^ ((depth as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        ^ ((node_idx as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    let mut fit_indices: Vec<u32> = Vec::with_capacity(node_indices.len() / 2 + 1);
    let mut audit_indices: Vec<u32> = Vec::with_capacity(node_indices.len() / 2 + 1);
    for &idx in node_indices {
        let h = (idx as u64)
            .wrapping_mul(0xD6E8_FD9D_50D5_1735)
            .wrapping_add(audit_seed);
        if (h >> 63) == 0 {
            audit_indices.push(idx);
        } else {
            fit_indices.push(idx);
        }
    }
    if fit_indices.len() < 16 || audit_indices.len() < 16 {
        return None;
    }

    let (audit_g, audit_h) = sum_gh(gradients, hessians, &audit_indices);
    let fit_g = g_sum - audit_g;
    let fit_h = h_sum - audit_h;
    if fit_h < min_h || audit_h < min_h {
        return None;
    }

    let audit_ab = eval_fixed_split_pseudo_gain(
        binned,
        gradients,
        hessians,
        &fit_indices,
        &audit_indices,
        fit_g,
        fit_h,
        audit_g,
        audit_h,
        split,
        lambda_reg,
        l1_reg,
        min_h,
    );
    let audit_ba = eval_fixed_split_pseudo_gain(
        binned,
        gradients,
        hessians,
        &audit_indices,
        &fit_indices,
        audit_g,
        audit_h,
        fit_g,
        fit_h,
        split,
        lambda_reg,
        l1_reg,
        min_h,
    );
    // Each direction scores about half of the node.  Add the two held-out
    // improvements so the audit lives on the same scale as full-node gain.
    (audit_ab.is_finite() && audit_ba.is_finite()).then_some(audit_ab + audit_ba)
}

#[inline]
fn split_contrast_stability(
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    lambda_reg: f64,
    l1_reg: f64,
    min_h: f64,
    split: &SplitResult,
    tree_seed: u64,
    node_idx: usize,
    depth: usize,
) -> Option<f64> {
    if node_indices.len() < 64 {
        return None;
    }

    let audit_seed = tree_seed
        ^ ((depth as u64).wrapping_mul(0x94D0_49BB_1331_11EB))
        ^ ((node_idx as u64).wrapping_mul(0xD2B7_4407_B1CE_6E93));
    let mut lg = [0.0f64; 2];
    let mut lh = [0.0f64; 2];
    let mut tg = [0.0f64; 2];
    let mut th = [0.0f64; 2];
    let mut cnt = [0usize; 2];

    for &idx in node_indices {
        let row = idx as usize;
        let mut h = (idx as u64)
            .wrapping_mul(0xD6E8_FD9D_50D5_1735)
            .wrapping_add(audit_seed);
        h ^= h >> 32;
        let fold = (h as usize) & 1;
        let g = gradients[row];
        let hv = hessians[row];
        tg[fold] += g;
        th[fold] += hv;
        cnt[fold] += 1;
        if split_goes_left_binned(split, binned, row) {
            lg[fold] += g;
            lh[fold] += hv;
        }
    }

    let mut contrast = [0.0f64; 2];
    for fold in 0..2 {
        let rh = th[fold] - lh[fold];
        if cnt[fold] < 16 || lh[fold] < min_h || rh < min_h {
            return None;
        }
        let rg = tg[fold] - lg[fold];
        let wl = l1_leaf_value(lg[fold], lh[fold], lambda_reg, l1_reg);
        let wr = l1_leaf_value(rg, rh, lambda_reg, l1_reg);
        if !(wl.is_finite() && wr.is_finite()) {
            return None;
        }
        contrast[fold] = wl - wr;
    }

    let a = contrast[0];
    let b = contrast[1];
    if !a.is_finite() || !b.is_finite() {
        return None;
    }
    let aa = a.abs();
    let bb = b.abs();
    if aa <= 1e-15 || bb <= 1e-15 {
        return Some(0.0);
    }
    if a.signum() != b.signum() {
        return Some(0.0);
    }
    Some((aa.min(bb) / aa.max(bb)).clamp(0.0, 1.0))
}

#[inline]
fn apply_contrast_stability_shrink(
    utility: f64,
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    lambda_reg: f64,
    l1_reg: f64,
    min_h: f64,
    split: &SplitResult,
    depth: usize,
    tree_seed: u64,
    node_idx: usize,
    risky: bool,
) -> f64 {
    if !risky || utility <= 0.0 || binned.split_pessimism <= 0.0 || node_indices.len() < 64 {
        return utility;
    }
    let Some(stability) = split_contrast_stability(
        binned,
        gradients,
        hessians,
        node_indices,
        lambda_reg,
        l1_reg,
        min_h,
        split,
        tree_seed,
        node_idx,
        depth,
    ) else {
        return utility;
    };

    // This is a stability shrinker, not a second gain formula. If the two
    // deterministic half-samples agree on the split contrast, utility is nearly
    // unchanged; if the contrast flips or is dominated by one half, the selected
    // split pays an empirical winner's-curse tax.
    let strength = (0.20 + 2.0 * binned.split_pessimism).clamp(0.0, 0.45);
    utility * (1.0 - strength * (1.0 - stability.clamp(0.0, 1.0)))
}

#[inline]
fn shadow_null_split_gain(
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    g_sum: f64,
    h_sum: f64,
    lambda_reg: f64,
    l1_reg: f64,
    gamma: f64,
    min_h: f64,
    seed: u64,
) -> f64 {
    if node_indices.len() < 64 || h_sum < 2.0 * min_h {
        return 0.0;
    }
    let n_bins = 32usize;
    let mut best = 0.0f64;
    for s in 0..2u64 {
        let mut g_hist = [0.0f64; 32];
        let mut h_hist = [0.0f64; 32];
        let salt = seed
            ^ (s.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            ^ ((node_indices.len() as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
        for &idx in node_indices {
            let mut h = (idx as u64).wrapping_mul(0xD6E8_FD9D_50D5_1735) ^ salt;
            h ^= h >> 33;
            h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
            h ^= h >> 33;
            let bin = (h as usize) & (n_bins - 1);
            let row = idx as usize;
            g_hist[bin] += gradients[row];
            h_hist[bin] += hessians[row];
        }
        let mut lg = 0.0f64;
        let mut lh = 0.0f64;
        for bin in 0..n_bins - 1 {
            lg += g_hist[bin];
            lh += h_hist[bin];
            let rh = h_sum - lh;
            if lh < min_h || rh < min_h {
                continue;
            }
            let rg = g_sum - lg;
            let gain = 0.5
                * (l1_gain_score(lg, lh, lambda_reg, l1_reg)
                    + l1_gain_score(rg, rh, lambda_reg, l1_reg)
                    - l1_gain_score(g_sum, h_sum, lambda_reg, l1_reg))
                - gamma;
            if gain.is_finite() && gain > best {
                best = gain;
            }
        }
    }
    best
}

#[inline]
fn adaptive_update_path_feature_budget(
    path_budget: &[(usize, f64)],
    feat: usize,
    split_utility: f64,
    structural_cost: f64,
) -> Vec<(usize, f64)> {
    let mut next = path_budget.to_vec();
    let signal = if split_utility > 0.0 && structural_cost > 0.0 {
        (split_utility / (structural_cost + 1e-12))
            .max(0.0)
            .ln_1p()
            .clamp(0.0, 4.0)
    } else {
        0.0
    };
    if let Some((_, v)) = next.iter_mut().find(|(f, _)| *f == feat) {
        *v = (0.65 * *v + 0.35 * signal).clamp(0.0, 4.0);
    } else if signal > 0.0 {
        next.push((feat, signal));
    }
    if let Some((_, v)) = next.iter_mut().find(|(f, _)| *f == usize::MAX) {
        *v = (0.75 * *v + 0.25 * signal).clamp(0.0, 4.0);
    } else if signal > 0.0 {
        next.push((usize::MAX, signal));
    }
    next
}

#[inline]
fn adaptive_candidate_structural_cost(
    depth: usize,
    max_depth: usize,
    node_rows: usize,
    h_sum: f64,
    root_h: f64,
    active_features: usize,
    split: &SplitResult,
    lh: f64,
    rh: f64,
    feature_budget: f64,
) -> f64 {
    let base_cost = adaptive_structural_min_gain(
        depth,
        max_depth,
        node_rows,
        h_sum,
        root_h,
        active_features,
        split,
    );
    let balanced_cost = base_cost * adaptive_balance_multiplier(lh, rh, h_sum);
    adaptive_apply_feature_budget(balanced_cost, feature_budget, depth)
}

/// Free-tree permutation noise floor: the best split gain attainable on a
/// within-node PERMUTATION of the gradients, scanned over the same features and
/// thresholds as the real search. This is an empirical estimate of the
/// multiple-testing "winner's gain" a pure-noise gradient would produce here;
/// a real split must beat `signal_gate` times this floor to be admitted.
/// Hessians stay row-aligned so the feasible split set (min_child_weight) is
/// identical to the real scan — only the gradient signal is destroyed.
fn permutation_null_best_gain(
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    split_features: &[usize],
    g_sum: f64,
    h_sum: f64,
    lambda_reg: f64,
    l1_reg: f64,
    min_h: f64,
    seed: u64,
) -> f64 {
    if node_indices.is_empty() || split_features.is_empty() {
        return 0.0;
    }
    let mut vals: Vec<f64> = node_indices
        .iter()
        .map(|&i| gradients[i as usize])
        .collect();
    let mut rng = StdRng::seed_from_u64(seed);
    vals.shuffle(&mut rng);
    let mut g_perm = gradients.to_vec();
    for (slot, &i) in node_indices.iter().enumerate() {
        g_perm[i as usize] = vals[slot];
    }
    let max_bins = split_features
        .iter()
        .map(|&f| binned.n_bins(f))
        .max()
        .unwrap_or(1)
        .max(1);
    let mut g_hist = vec![0.0f64; max_bins];
    let mut h_hist = vec![0.0f64; max_bins];
    let r = find_best_split(
        binned,
        &g_perm,
        hessians,
        node_indices,
        split_features,
        g_sum,
        h_sum,
        lambda_reg,
        l1_reg,
        0.0,
        min_h,
        &mut g_hist,
        &mut h_hist,
        0.0,
        seed,
        0.0,
        &[],
        0.0,
        false,
    );
    if r.gain.is_finite() {
        r.gain.max(0.0)
    } else {
        0.0
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn adaptive_candidate_utility(
    adaptive_growth: bool,
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    g_sum: f64,
    h_sum: f64,
    lambda_reg: f64,
    l1_reg: f64,
    min_h: f64,
    split: &SplitResult,
    depth: usize,
    max_depth: usize,
    root_h: f64,
    split_features: &[usize],
    tree_seed: u64,
    node_idx: usize,
    feature_budget: f64,
    leaf_var_shrink: f64,
) -> f64 {
    let utility = adaptive_candidate_utility_ungated(
        adaptive_growth,
        binned,
        gradients,
        hessians,
        node_indices,
        g_sum,
        h_sum,
        lambda_reg,
        l1_reg,
        min_h,
        split,
        depth,
        max_depth,
        root_h,
        split_features.len(),
        tree_seed,
        node_idx,
        feature_budget,
        leaf_var_shrink,
    );
    // Free-tree signal gate: only consulted for splits that would otherwise be
    // admitted. The gate compares the REAL best gain against the permutation
    // noise floor over the same search space; failing it turns the node into a
    // leaf (growth stops where signal is exhausted, independent of max_depth).
    if utility > 0.0
        && binned.signal_gate > 0.0
        && node_indices.len() >= 16
        && split.gain.is_finite()
        && split.gain > 0.0
    {
        let seed = tree_seed
            ^ (node_idx as u64).wrapping_mul(0x517C_C1B7_2722_0A95)
            ^ (depth as u64).wrapping_mul(0x2545_F491_4F6C_DD1D);
        let null = permutation_null_best_gain(
            binned,
            gradients,
            hessians,
            node_indices,
            split_features,
            g_sum,
            h_sum,
            lambda_reg,
            l1_reg,
            min_h,
            seed,
        );
        if split.gain < binned.signal_gate * null {
            return f64::NEG_INFINITY;
        }
    }
    utility
}

#[inline]
fn adaptive_candidate_utility_ungated(
    adaptive_growth: bool,
    binned: &BinnedData,
    gradients: &[f64],
    hessians: &[f64],
    node_indices: &[u32],
    g_sum: f64,
    h_sum: f64,
    lambda_reg: f64,
    l1_reg: f64,
    min_h: f64,
    split: &SplitResult,
    depth: usize,
    max_depth: usize,
    root_h: f64,
    active_features: usize,
    tree_seed: u64,
    node_idx: usize,
    feature_budget: f64,
    leaf_var_shrink: f64,
) -> f64 {
    if !split.gain.is_finite() {
        return f64::NEG_INFINITY;
    }
    if !adaptive_growth {
        let mut utility = split.gain;
        if utility > 0.0 && leaf_var_shrink > 0.0 {
            if let Some(ratio) = post_shrink_split_gain_ratio(
                binned,
                gradients,
                hessians,
                node_indices,
                g_sum,
                h_sum,
                lambda_reg,
                l1_reg,
                min_h,
                split,
                leaf_var_shrink,
            ) {
                utility *= ratio;
            }
        }
        let child_h = adaptive_child_h_sums(binned, hessians, node_indices, h_sum, split);
        let mut shadow_gain = 0.0;
        if utility > 0.0 && binned.split_pessimism > 0.0 {
            shadow_gain = shadow_null_split_gain(
                gradients,
                hessians,
                node_indices,
                g_sum,
                h_sum,
                lambda_reg,
                l1_reg,
                0.0,
                min_h,
                tree_seed
                    ^ ((node_idx as u64).wrapping_mul(0xA24B_AED4_963E_E407))
                    ^ ((depth as u64).wrapping_mul(0x9FB2_1C65_1E98_DF25)),
            );
            if shadow_gain.is_finite() && shadow_gain > 0.0 {
                let null_strength = (5.0 * binned.split_pessimism).clamp(0.0, 0.60);
                let null_tax = (null_strength * shadow_gain).min(0.80 * utility);
                utility -= null_tax;
            }
        }
        if utility > 0.0 && binned.split_pessimism > 0.0 {
            if let Some((lh, rh)) = child_h.filter(|(lh, rh)| *lh >= min_h && *rh >= min_h) {
                let balance = if h_sum > 0.0 {
                    (4.0 * (lh / h_sum).clamp(1e-9, 1.0) * (rh / h_sum).clamp(1e-9, 1.0))
                        .clamp(1e-4, 1.0)
                } else {
                    1e-4
                };
                let null_ratio = shadow_gain / split.gain.max(1e-12);
                let risky = depth >= 2
                    || split.is_cat
                    || split.is_cat_pair
                    || split.is_oblique
                    || balance < 0.30
                    || null_ratio > 0.25;
                utility = apply_contrast_stability_shrink(
                    utility,
                    binned,
                    gradients,
                    hessians,
                    node_indices,
                    lambda_reg,
                    l1_reg,
                    min_h,
                    split,
                    depth,
                    tree_seed,
                    node_idx,
                    risky,
                );
                if risky && node_indices.len() >= 64 {
                    if let Some(honest_gain) = crossfit_split_pseudo_gain(
                        binned,
                        gradients,
                        hessians,
                        node_indices,
                        g_sum,
                        h_sum,
                        lambda_reg,
                        l1_reg,
                        min_h,
                        split,
                        tree_seed,
                        node_idx,
                        depth,
                    ) {
                        let audit_weight = (0.45
                            * adaptive_risk_weight(depth, max_depth, h_sum, root_h, lh, rh, split)
                            + 1.5 * binned.split_pessimism)
                            .clamp(0.0, 0.35);
                        let audited = (1.0 - audit_weight) * utility + audit_weight * honest_gain;
                        utility = utility.min(audited);
                    }
                }
            }
        }
        return utility;
    }

    let (lh, rh) = match adaptive_child_h_sums(binned, hessians, node_indices, h_sum, split) {
        Some((lh, rh)) if lh >= min_h && rh >= min_h => (lh, rh),
        _ => return f64::NEG_INFINITY,
    };

    let structural_cost = adaptive_candidate_structural_cost(
        depth,
        max_depth,
        node_indices.len(),
        h_sum,
        root_h,
        active_features,
        split,
        lh,
        rh,
        feature_budget,
    );
    let mut raw_utility = split.gain - structural_cost;
    if raw_utility > 0.0 && leaf_var_shrink > 0.0 {
        if let Some(ratio) = post_shrink_split_gain_ratio(
            binned,
            gradients,
            hessians,
            node_indices,
            g_sum,
            h_sum,
            lambda_reg,
            l1_reg,
            min_h,
            split,
            leaf_var_shrink,
        ) {
            raw_utility *= ratio;
        }
    }
    if raw_utility <= 0.0 {
        return raw_utility;
    }

    let balance = if h_sum > 0.0 {
        (4.0 * (lh / h_sum).clamp(1e-9, 1.0) * (rh / h_sum).clamp(1e-9, 1.0)).clamp(1e-4, 1.0)
    } else {
        1e-4
    };
    let risky =
        depth >= 2 || split.is_cat || split.is_cat_pair || split.is_oblique || balance < 0.25;
    let raw_utility = apply_contrast_stability_shrink(
        raw_utility,
        binned,
        gradients,
        hessians,
        node_indices,
        lambda_reg,
        l1_reg,
        min_h,
        split,
        depth,
        tree_seed,
        node_idx,
        risky,
    );
    if raw_utility <= 0.0 {
        return raw_utility;
    }
    if !risky || node_indices.len() < 48 {
        return raw_utility;
    }

    let Some(honest_gain) = crossfit_split_pseudo_gain(
        binned,
        gradients,
        hessians,
        node_indices,
        g_sum,
        h_sum,
        lambda_reg,
        l1_reg,
        min_h,
        split,
        tree_seed,
        node_idx,
        depth,
    ) else {
        return raw_utility;
    };

    let honest_utility = honest_gain - structural_cost;
    let audit_weight = adaptive_risk_weight(depth, max_depth, h_sum, root_h, lh, rh, split);
    let audited_utility = (1.0 - audit_weight) * raw_utility + audit_weight * honest_utility;

    // The audit is a shrinkage estimator for selection risk, not an alternate
    // way to create gain.  It can reduce a suspicious split but never promote it.
    raw_utility.min(audited_utility)
}

impl DecisionTree {
    pub fn build_depthwise(
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        indices: &[u32],
        lambda_reg: f64,
        l1_reg: f64,
        gamma: f64,
        max_depth: usize,
        min_child_weight: f64,
        feature_mask: &[bool],
        colsample_bylevel: f64,
        tree_seed: u64,
        random_strength: f64,
        cat_smooth: f64,
        cat_lookup_smooth: f64,
        monotone_constraints: &[i8],
        gain_penalty: f64,
        extra_trees: bool,
        lookahead_alpha: f64,
        expert_split: bool,
        sparse_oblique_splits: bool,
        interval_splits: bool,
        root_anchor_feature: Option<usize>,
        adaptive_growth: bool,
        leaf_var_shrink: f64,
        cat_pair_cfg: CatPairConfig,
        est_arb: Option<&[u32]>,
        feature_prior: Option<&[f64]>,
        thermal: f64,
        thermal_n_exp: f64,
        thermal_depth_gamma: f64,
    ) -> Self {
        let max_nodes = (1usize << (max_depth + 1)).min(65536);
        let mut tree = TreeBuilder::new(max_nodes);
        let mut row_buf: Vec<u32> = indices.to_vec();
        // Honest split arbitration: estimation rows are partitioned alongside the
        // build rows; a split is committed only if it also has positive Newton gain
        // on this independent fold (kills winner's-curse splits at every node).
        let mut est_buf: Vec<u32> = est_arb.map(|e| e.to_vec()).unwrap_or_default();
        let mut est_seg: std::collections::HashMap<usize, (usize, usize)> =
            std::collections::HashMap::new();
        if !est_buf.is_empty() {
            est_seg.insert(0, (0, est_buf.len()));
        }

        tree.add_node();
        let (root_g, root_h) = sum_gh(gradients, hessians, &row_buf);

        let min_h = min_child_weight.max(1e-10);
        let tree_features: Vec<usize> = (0..binned.n_features)
            .filter(|&f| feature_mask[f])
            .collect();
        let root_anchor_features: Vec<usize> = root_anchor_feature
            .filter(|&f| f < binned.n_features && feature_mask[f])
            .map(|f| vec![f])
            .unwrap_or_default();
        let cbl_n_select = if colsample_bylevel < 1.0 {
            ((colsample_bylevel * tree_features.len() as f64) as usize).max(1)
        } else {
            0
        };
        let mut node_features: Vec<usize> = Vec::with_capacity(tree_features.len());
        let max_bins = (0..binned.n_features)
            .map(|c| binned.n_bins(c))
            .max()
            .unwrap_or(1);

        let use_hist_sub = !extra_trees;

        if use_hist_sub {
            // ── Histogram subtraction path ──
            // Stack entries: (start, end, depth, node_idx, g_sum, h_sum,
            // cached_hists, path_feature_budget)
            let mut stack: Vec<(
                usize,
                usize,
                usize,
                usize,
                f64,
                f64,
                Option<NodeHists>,
                Vec<(usize, f64)>,
            )> = Vec::with_capacity(max_nodes);
            let mut hist_pool = HistPool::new(tree_features.len(), max_bins);
            let mut g_hist = vec![0.0f64; max_bins];
            let mut h_hist = vec![0.0f64; max_bins];

            // Build root histograms
            let mut root_hists = hist_pool.take();
            build_node_hists(
                binned,
                gradients,
                hessians,
                &row_buf,
                &tree_features,
                &mut root_hists,
            );
            stack.push((
                0,
                row_buf.len(),
                0,
                0,
                root_g,
                root_h,
                Some(root_hists),
                Vec::new(),
            ));

            while let Some((start, end, depth, node_idx, g_sum, h_sum, cached_hists, path_budget)) =
                stack.pop()
            {
                let node_indices = &row_buf[start..end];
                let n_leaf = (end - start) as f64;
                let leaf_value = l1_leaf_value(
                    g_sum,
                    h_sum,
                    lambda_reg + lambda_reg / n_leaf.max(1.0).sqrt(),
                    l1_reg,
                );
                // PRM: record per-node training stats for refinement-dropout at predict time.
                tree.set_node_stats(node_idx, h_sum, (end - start) as u32);

                if depth >= max_depth || (end - start) <= 1 || h_sum < min_h {
                    tree.set_leaf(node_idx, leaf_value);
                    if cat_lookup_smooth > 0.0 {
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
                            tree.set_cll(
                                node_idx,
                                make_cll_lookup(
                                    &cll,
                                    leaf_value,
                                    cat_lookup_smooth,
                                    lambda_reg,
                                    min_child_weight,
                                ),
                            );
                        }
                    }
                    continue;
                }

                // Per-node feature subsampling
                let active_features: &[usize] = if depth == 0 && !root_anchor_features.is_empty() {
                    &root_anchor_features
                } else if cbl_n_select > 0 {
                    let node_seed = tree_seed.wrapping_mul(2654435761).wrapping_add(
                        (depth as u64)
                            .wrapping_mul(1000003)
                            .wrapping_add(node_idx as u64),
                    );
                    let mut rng = StdRng::seed_from_u64(node_seed);
                    node_features.clear();
                    node_features.extend_from_slice(&tree_features);
                    node_features.shuffle(&mut rng);
                    node_features.truncate(cbl_n_select);
                    &node_features
                } else {
                    &tree_features
                };
                let split_features: &[usize] = active_features;
                // Thermal boosting: sample splits at the node's own argmax-bias scale.
                // E[max of K noisy gains] - max true gain ~ sigma*sqrt(2 ln K); relative
                // gain noise ~ 1/sqrt(n_node). Hot where evidence is thin, frozen at the
                // data-rich root, no tuning per node.
                let rs_node = if thermal > 0.0 {
                    let k_cand = ((split_features.len().max(2) * 64) as f64).ln();
                    (thermal * (2.0 * k_cand).sqrt() * thermal_depth_gamma.powi(depth as i32)
                        / ((end - start) as f64).powf(thermal_n_exp))
                    .min(3.0)
                } else {
                    random_strength
                };
                let use_interval_splits = interval_splits && depth == 0;

                // If we have cached histograms, scan them; otherwise fall back to find_best_split
                let (mut split_result, mut node_hists) = if let Some(nh) = cached_hists {
                    let sr = find_best_split_from_hists(
                        &nh,
                        &tree_features,
                        split_features,
                        binned,
                        Some(gradients),
                        Some(hessians),
                        Some(node_indices),
                        g_sum,
                        h_sum,
                        lambda_reg,
                        l1_reg,
                        gamma,
                        min_h,
                        rs_node,
                        tree_seed.wrapping_add(depth as u64),
                        cat_smooth,
                        monotone_constraints,
                        gain_penalty,
                        use_interval_splits,
                    
                    feature_prior,
                );
                    (sr, Some(nh))
                } else {
                    let sr = find_best_split_v5(
                        binned,
                        gradients,
                        hessians,
                        node_indices,
                        split_features,
                        g_sum,
                        h_sum,
                        lambda_reg,
                        l1_reg,
                        gamma,
                        min_h,
                        &mut g_hist,
                        &mut h_hist,
                        rs_node,
                        tree_seed.wrapping_add(depth as u64),
                        cat_smooth,
                        monotone_constraints,
                        gain_penalty,
                        use_interval_splits,
                        &cat_pair_cfg,
                        depth,
                    );
                    (sr, None)
                };

                // GGFP v5.0 — augment cached-hists path with cat-pair too
                if cat_pair_cfg.enabled && !split_result.is_oblique {
                    let pair = eval_cat_pair_jit_for_node(
                        binned,
                        gradients,
                        hessians,
                        node_indices,
                        split_features,
                        g_sum,
                        h_sum,
                        lambda_reg,
                        l1_reg,
                        gamma,
                        min_h,
                        cat_smooth,
                        depth,
                        split_result.gain,
                        &cat_pair_cfg,
                    );
                    if pair.gain > split_result.gain {
                        split_result = pair;
                    }
                }

                if sparse_oblique_splits
                    && !extra_trees
                    && depth < max_depth
                    && node_indices.len() >= 16
                    && split_features.len() >= 2
                {
                    let oblique = find_sparse_oblique_split(
                        binned,
                        gradients,
                        hessians,
                        node_indices,
                        split_features,
                        g_sum,
                        h_sum,
                        lambda_reg,
                        gamma,
                        min_h,
                        monotone_constraints,
                        node_hists.as_ref().map(|nh| (nh, tree_features.as_slice())),
                    );
                    if oblique.gain.is_finite() && oblique.gain > split_result.gain {
                        split_result = oblique;
                    }
                }

                // ── LAS: 1-step look-ahead split selection (all-features variant) ──
                // For EACH active feature's best split, provisionally partition and compute
                // max child-split gain. Score = own_gain + α · max(left_child_gain, right_child_gain).
                // Picks the split that enables the best follow-up. Cost: O(F) extra
                // find_best_split calls per node (parent + 2 child per feature).
                if lookahead_alpha > 0.0
                    && depth + 1 < max_depth
                    && split_result.gain > 0.0
                    && (end - start) >= 16
                    && split_features.len() >= 2
                {
                    let noise_seed = tree_seed
                        .wrapping_add((depth as u64).wrapping_mul(31))
                        .wrapping_add(node_idx as u64);

                    // Helper: for a candidate split, provisionally partition a temp copy
                    // of node_indices and return max child best-split gain.
                    let eval_future = |cand: &SplitResult| -> f64 {
                        if cand.gain <= 0.0 || !cand.gain.is_finite() {
                            return 0.0;
                        }
                        let mut tmp: Vec<u32> = node_indices.to_vec();
                        let tmp_len = tmp.len();
                        let tmp_left_end =
                            partition_indices_split(&mut tmp, 0, tmp_len, binned, cand);
                        if tmp_left_end == 0 || tmp_left_end == tmp_len {
                            return 0.0;
                        }
                        let left_slice = &tmp[..tmp_left_end];
                        let right_slice = &tmp[tmp_left_end..];
                        let (lg, lh) = child_left_sums(cand, gradients, hessians, left_slice);
                        let rg = g_sum - lg;
                        let rh = h_sum - lh;
                        if lh < min_h || rh < min_h {
                            return 0.0;
                        }
                        if expert_split {
                            let left_leaf = -lg
                                / (lh
                                    + lambda_reg
                                    + lambda_reg / (left_slice.len() as f64).max(1.0).sqrt());
                            let right_leaf = -rg
                                / (rh
                                    + lambda_reg
                                    + lambda_reg / (right_slice.len() as f64).max(1.0).sqrt());
                            let mut future = 0.0f64;
                            if let Some(best) = eval_best_lookup_for_node(
                                binned,
                                gradients,
                                hessians,
                                left_slice,
                                lg,
                                lh,
                                left_leaf,
                                lambda_reg,
                                gamma,
                                min_h,
                                cat_lookup_smooth,
                                None,
                            ) {
                                future += best.score.max(0.0);
                            }
                            if let Some(best) = eval_best_lookup_for_node(
                                binned,
                                gradients,
                                hessians,
                                right_slice,
                                rg,
                                rh,
                                right_leaf,
                                lambda_reg,
                                gamma,
                                min_h,
                                cat_lookup_smooth,
                                None,
                            ) {
                                future += best.score.max(0.0);
                            }
                            return future;
                        }
                        let mut gh1 = vec![0.0f64; max_bins];
                        let mut hh1 = vec![0.0f64; max_bins];
                        let mut left_best = 0.0f64;
                        if left_slice.len() > 1 {
                            let r = find_best_split(
                                binned,
                                gradients,
                                hessians,
                                left_slice,
                                split_features,
                                lg,
                                lh,
                                lambda_reg,
                                l1_reg,
                                gamma,
                                min_h,
                                &mut gh1,
                                &mut hh1,
                                0.0,
                                noise_seed.wrapping_add(101),
                                cat_smooth,
                                monotone_constraints,
                                gain_penalty,
                                false,
                            );
                            if r.gain.is_finite() {
                                left_best = r.gain.max(0.0);
                            }
                        }
                        let mut right_best = 0.0f64;
                        if right_slice.len() > 1 {
                            let r = find_best_split(
                                binned,
                                gradients,
                                hessians,
                                right_slice,
                                split_features,
                                rg,
                                rh,
                                lambda_reg,
                                l1_reg,
                                gamma,
                                min_h,
                                &mut gh1,
                                &mut hh1,
                                0.0,
                                noise_seed.wrapping_add(202),
                                cat_smooth,
                                monotone_constraints,
                                gain_penalty,
                                false,
                            );
                            if r.gain.is_finite() {
                                right_best = r.gain.max(0.0);
                            }
                        }
                        left_best.max(right_best)
                    };

                    // EA/SR-gated lookahead score.
                    //
                    // The old LAS score was `gain + alpha * future_gain`, which
                    // over-promotes wide/noisy search spaces.  The architecture
                    // lab found a more stable rule: keep raw gain as the base,
                    // and add child-lookahead only after it clears structural
                    // risk plus feature-search pressure.  The score is used
                    // only to choose among candidate splits; stored leaf values
                    // and split gain remain the ordinary Newton quantities.
                    let lookahead_score = |cand: &SplitResult, future_gain: f64| -> f64 {
                        if !(cand.gain.is_finite() && cand.gain > 0.0) {
                            return f64::NEG_INFINITY;
                        }
                        let support = (h_sum / root_h.max(1e-12)).clamp(1e-9, 1.0);
                        let (lh, rh) = adaptive_child_h_sums(
                            binned,
                            hessians,
                            node_indices,
                            h_sum,
                            cand,
                        )
                        .unwrap_or((0.0, 0.0));
                        let balance = if h_sum > 0.0 {
                            (lh.min(rh) / h_sum).clamp(0.0, 0.5)
                        } else {
                            0.0
                        };
                        let depth_frac =
                            ((depth + 1) as f64 / max_depth.max(1) as f64).clamp(0.0, 1.0);
                        let risk = depth_frac
                            * (1.0 - 2.0 * balance).max(0.0)
                            / support.sqrt();
                        let feature_pressure =
                            (split_features.len().max(2) as f64).ln() / 65.0_f64.ln();
                        let signal =
                            ((1.0 + future_gain.max(0.0)).ln() - 0.53 * risk - 0.23 * feature_pressure)
                                .max(0.0);
                        (1.0 + cand.gain).ln() + lookahead_alpha * signal
                    };

                    // Start with the incumbent best split's score.
                    let mut best_score =
                        lookahead_score(&split_result, eval_future(&split_result));

                    // Evaluate every active feature's best split, keep the highest LAS score.
                    for &feat in split_features {
                        if feat == split_result.feat {
                            continue;
                        }
                        let mut gh_local = vec![0.0f64; max_bins];
                        let mut hh_local = vec![0.0f64; max_bins];
                        let cand = find_best_split(
                            binned,
                            gradients,
                            hessians,
                            node_indices,
                            &[feat],
                            g_sum,
                            h_sum,
                            lambda_reg,
                            l1_reg,
                            gamma,
                            min_h,
                            &mut gh_local,
                            &mut hh_local,
                            random_strength,
                            noise_seed.wrapping_add(feat as u64 * 17 + 7),
                            cat_smooth,
                            monotone_constraints,
                            gain_penalty,
                            use_interval_splits,
                        );
                        if !(cand.gain.is_finite() && cand.gain > 0.0) {
                            continue;
                        }
                        let cand_score = lookahead_score(&cand, eval_future(&cand));
                        if cand_score > best_score {
                            best_score = cand_score;
                            split_result = cand;
                        }
                    }
                }

                if cat_lookup_smooth > 0.0 {
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
                        if cll.gain > split_result.gain.max(0.0) {
                            tree.set_leaf(node_idx, leaf_value);
                            tree.set_cll(
                                node_idx,
                                make_cll_lookup(
                                    &cll,
                                    leaf_value,
                                    cat_lookup_smooth,
                                    lambda_reg,
                                    min_child_weight,
                                ),
                            );
                            continue;
                        }
                    }
                }

                let is_numeric_interval = split_result.is_cat
                    && !split_result.is_cat_pair
                    && split_result.feat < binned.is_categorical.len()
                    && !binned.is_categorical[split_result.feat];
                if use_interval_splits
                    && is_numeric_interval
                    && node_indices.len() >= 96
                    && split_result.gain.is_finite()
                    && split_result.gain > 0.0
                {
                    let axis_cf = if let Some(nh) = node_hists.as_ref() {
                        find_best_split_from_hists(
                            nh,
                            &tree_features,
                            split_features,
                            binned,
                            Some(gradients),
                            Some(hessians),
                            Some(node_indices),
                            g_sum,
                            h_sum,
                            lambda_reg,
                            l1_reg,
                            gamma,
                            min_h,
                            random_strength,
                            tree_seed.wrapping_add(depth as u64),
                            cat_smooth,
                            monotone_constraints,
                            gain_penalty,
                            false,
                        
                    feature_prior,
                )
                    } else {
                        find_best_split_v5(
                            binned,
                            gradients,
                            hessians,
                            node_indices,
                            split_features,
                            g_sum,
                            h_sum,
                            lambda_reg,
                            l1_reg,
                            gamma,
                            min_h,
                            &mut g_hist,
                            &mut h_hist,
                            random_strength,
                            tree_seed.wrapping_add(depth as u64),
                            cat_smooth,
                            monotone_constraints,
                            gain_penalty,
                            false,
                            &cat_pair_cfg,
                            depth,
                        )
                    };
                    if axis_cf.gain.is_finite()
                        && axis_cf.gain > 0.0
                        && !axis_cf.is_oblique
                        && !axis_cf.is_cat_pair
                    {
                        let audit_seed = tree_seed
                            ^ ((depth as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                            ^ ((node_idx as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
                        let mut audit_indices: Vec<u32> =
                            Vec::with_capacity(node_indices.len() / 2 + 1);
                        for &idx in node_indices {
                            let h = (idx as u64)
                                .wrapping_mul(0xD6E8_FD9D_50D5_1735)
                                .wrapping_add(audit_seed);
                            if (h >> 63) == 0 {
                                audit_indices.push(idx);
                            }
                        }
                        if audit_indices.len() >= 48 && audit_indices.len() < node_indices.len() {
                            let (ag, ah) = sum_gh(gradients, hessians, &audit_indices);
                            let interval_audit = eval_fixed_split_pseudo_gain(
                                binned,
                                gradients,
                                hessians,
                                node_indices,
                                &audit_indices,
                                g_sum,
                                h_sum,
                                ag,
                                ah,
                                &split_result,
                                lambda_reg,
                                l1_reg,
                                min_h,
                            );
                            let axis_audit = eval_fixed_split_pseudo_gain(
                                binned,
                                gradients,
                                hessians,
                                node_indices,
                                &audit_indices,
                                g_sum,
                                h_sum,
                                ag,
                                ah,
                                &axis_cf,
                                lambda_reg,
                                l1_reg,
                                min_h,
                            );
                            if !(interval_audit.is_finite()
                                && interval_audit > axis_audit.max(0.0) * 1.15 + 1e-12)
                            {
                                split_result = axis_cf;
                            }
                        }
                    }
                }

                let feature_budget = path_feature_budget(&path_budget, split_result.feat);
                let split_utility = adaptive_candidate_utility(
                    adaptive_growth,
                    binned,
                    gradients,
                    hessians,
                    node_indices,
                    g_sum,
                    h_sum,
                    lambda_reg,
                    l1_reg,
                    min_h,
                    &split_result,
                    depth,
                    max_depth,
                    root_h,
                    split_features,
                    tree_seed,
                    node_idx,
                    feature_budget,
                    leaf_var_shrink,
                );
                if split_utility <= 0.0 {
                    tree.set_leaf(node_idx, leaf_value);
                    continue;
                }

                // Honest arbitration: the chosen split must also show positive gain
                // on the estimation fold at this node, else don't commit.
                let mut est_children: Option<(usize, usize, usize)> = None;
                if !est_buf.is_empty() {
                    if let Some(&(es, ee)) = est_seg.get(&node_idx) {
                        if ee - es >= 8 {
                            let emid =
                                partition_indices_split(&mut est_buf, es, ee, binned, &split_result);
                            let (elg, elh) =
                                child_left_sums(&split_result, gradients, hessians, &est_buf[es..emid]);
                            let (etg, eth) = sum_gh(gradients, hessians, &est_buf[es..ee]);
                            let (erg, erh) = (etg - elg, eth - elh);
                            let est_gain = elg * elg / (elh + lambda_reg)
                                + erg * erg / (erh + lambda_reg)
                                - etg * etg / (eth + lambda_reg);
                            if !est_gain.is_finite() || est_gain <= 0.0 {
                                tree.set_leaf(node_idx, leaf_value);
                                continue;
                            }
                            est_children = Some((es, emid, ee));
                        }
                    }
                }

                let node_len = node_indices.len();
                let left_end =
                    partition_indices_split(&mut row_buf, start, end, binned, &split_result);
                if left_end == start || left_end == end {
                    tree.set_leaf(node_idx, leaf_value);
                    continue;
                }

                let left_indices = &row_buf[start..left_end];
                let (lg, lh) = child_left_sums(&split_result, gradients, hessians, left_indices);
                let rg = g_sum - lg;
                let rh = h_sum - lh;
                let split_structural_cost = adaptive_candidate_structural_cost(
                    depth,
                    max_depth,
                    node_len,
                    h_sum,
                    root_h,
                    split_features.len(),
                    &split_result,
                    lh,
                    rh,
                    feature_budget,
                );
                let child_path_budget = if adaptive_growth {
                    adaptive_update_path_feature_budget(
                        &path_budget,
                        split_result.feat,
                        split_utility,
                        split_structural_cost,
                    )
                } else {
                    path_budget
                };

                let (left_idx, right_idx) =
                    tree.add_split_from_sr(node_idx, split_result, leaf_value);
                if let Some((es, emid, ee)) = est_children {
                    est_seg.insert(left_idx, (es, emid));
                    est_seg.insert(right_idx, (emid, ee));
                }

                // Histogram subtraction trick
                let n_left = left_end - start;
                let n_right = end - left_end;
                let child_depth = depth + 1;
                let left_needs_hists = child_depth < max_depth && n_left > 1;
                let right_needs_hists = child_depth < max_depth && n_right > 1;

                if let Some(ref parent_hists) = node_hists {
                    if left_needs_hists && right_needs_hists {
                        // Both need hists: build smaller, subtract for larger
                        let mut smaller_hists = hist_pool.take();
                        let mut larger_hists = hist_pool.take();
                        if n_left <= n_right {
                            build_node_hists(
                                binned,
                                gradients,
                                hessians,
                                &row_buf[start..left_end],
                                &tree_features,
                                &mut smaller_hists,
                            );
                            subtract_node_hists(parent_hists, &smaller_hists, &mut larger_hists);
                            stack.push((
                                left_end,
                                end,
                                child_depth,
                                right_idx,
                                rg,
                                rh,
                                Some(larger_hists),
                                child_path_budget.clone(),
                            ));
                            stack.push((
                                start,
                                left_end,
                                child_depth,
                                left_idx,
                                lg,
                                lh,
                                Some(smaller_hists),
                                child_path_budget,
                            ));
                        } else {
                            build_node_hists(
                                binned,
                                gradients,
                                hessians,
                                &row_buf[left_end..end],
                                &tree_features,
                                &mut smaller_hists,
                            );
                            subtract_node_hists(parent_hists, &smaller_hists, &mut larger_hists);
                            stack.push((
                                left_end,
                                end,
                                child_depth,
                                right_idx,
                                rg,
                                rh,
                                Some(smaller_hists),
                                child_path_budget.clone(),
                            ));
                            stack.push((
                                start,
                                left_end,
                                child_depth,
                                left_idx,
                                lg,
                                lh,
                                Some(larger_hists),
                                child_path_budget,
                            ));
                        }
                    } else if left_needs_hists {
                        let mut left_hists = hist_pool.take();
                        let mut right_tmp = hist_pool.take();
                        build_node_hists(
                            binned,
                            gradients,
                            hessians,
                            &row_buf[left_end..end],
                            &tree_features,
                            &mut right_tmp,
                        );
                        subtract_node_hists(parent_hists, &right_tmp, &mut left_hists);
                        hist_pool.recycle(right_tmp);
                        stack.push((
                            left_end,
                            end,
                            child_depth,
                            right_idx,
                            rg,
                            rh,
                            None,
                            child_path_budget.clone(),
                        ));
                        stack.push((
                            start,
                            left_end,
                            child_depth,
                            left_idx,
                            lg,
                            lh,
                            Some(left_hists),
                            child_path_budget,
                        ));
                    } else if right_needs_hists {
                        let mut right_hists = hist_pool.take();
                        let mut left_tmp = hist_pool.take();
                        build_node_hists(
                            binned,
                            gradients,
                            hessians,
                            &row_buf[start..left_end],
                            &tree_features,
                            &mut left_tmp,
                        );
                        subtract_node_hists(parent_hists, &left_tmp, &mut right_hists);
                        hist_pool.recycle(left_tmp);
                        stack.push((
                            left_end,
                            end,
                            child_depth,
                            right_idx,
                            rg,
                            rh,
                            Some(right_hists),
                            child_path_budget.clone(),
                        ));
                        stack.push((
                            start,
                            left_end,
                            child_depth,
                            left_idx,
                            lg,
                            lh,
                            None,
                            child_path_budget,
                        ));
                    } else {
                        stack.push((
                            left_end,
                            end,
                            child_depth,
                            right_idx,
                            rg,
                            rh,
                            None,
                            child_path_budget.clone(),
                        ));
                        stack.push((
                            start,
                            left_end,
                            child_depth,
                            left_idx,
                            lg,
                            lh,
                            None,
                            child_path_budget,
                        ));
                    }
                } else {
                    // No parent hists (fallback) — push None, children will use find_best_split
                    stack.push((
                        left_end,
                        end,
                        child_depth,
                        right_idx,
                        rg,
                        rh,
                        None,
                        child_path_budget.clone(),
                    ));
                    stack.push((
                        start,
                        left_end,
                        child_depth,
                        left_idx,
                        lg,
                        lh,
                        None,
                        child_path_budget,
                    ));
                }
                if let Some(h) = node_hists.take() {
                    hist_pool.recycle(h);
                }
            }
        } else {
            // ── Original path for extra_trees ──
            let mut stack: Vec<(usize, usize, usize, usize, f64, f64, Vec<(usize, f64)>)> =
                Vec::with_capacity(max_nodes);
            let mut g_hist = vec![0.0f64; max_bins];
            let mut h_hist = vec![0.0f64; max_bins];
            stack.push((0, row_buf.len(), 0, 0, root_g, root_h, Vec::new()));

            while let Some((start, end, depth, node_idx, g_sum, h_sum, path_budget)) = stack.pop() {
                let node_indices = &row_buf[start..end];
                let n_leaf = (end - start) as f64;
                let leaf_value = l1_leaf_value(
                    g_sum,
                    h_sum,
                    lambda_reg + lambda_reg / n_leaf.max(1.0).sqrt(),
                    l1_reg,
                );
                tree.set_node_stats(node_idx, h_sum, (end - start) as u32);

                if depth >= max_depth || (end - start) <= 1 || h_sum < min_h {
                    tree.set_leaf(node_idx, leaf_value);
                    if cat_lookup_smooth > 0.0 {
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
                            tree.set_cll(
                                node_idx,
                                make_cll_lookup(
                                    &cll,
                                    leaf_value,
                                    cat_lookup_smooth,
                                    lambda_reg,
                                    min_child_weight,
                                ),
                            );
                        }
                    }
                    continue;
                }

                let active_features: &[usize] = if depth == 0 && !root_anchor_features.is_empty() {
                    &root_anchor_features
                } else if cbl_n_select > 0 {
                    let node_seed = tree_seed.wrapping_mul(2654435761).wrapping_add(
                        (depth as u64)
                            .wrapping_mul(1000003)
                            .wrapping_add(node_idx as u64),
                    );
                    let mut rng = StdRng::seed_from_u64(node_seed);
                    node_features.clear();
                    node_features.extend_from_slice(&tree_features);
                    node_features.shuffle(&mut rng);
                    node_features.truncate(cbl_n_select);
                    &node_features
                } else {
                    &tree_features
                };

                let split_result = find_extra_trees_split(
                    binned,
                    gradients,
                    hessians,
                    node_indices,
                    active_features,
                    g_sum,
                    h_sum,
                    lambda_reg,
                    gamma,
                    min_h,
                    &mut g_hist,
                    &mut h_hist,
                    tree_seed
                        .wrapping_add(depth as u64)
                        .wrapping_add(node_idx as u64),
                    monotone_constraints,
                );

                if cat_lookup_smooth > 0.0 {
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
                        if cll.gain > split_result.gain.max(0.0) {
                            tree.set_leaf(node_idx, leaf_value);
                            tree.set_cll(
                                node_idx,
                                make_cll_lookup(
                                    &cll,
                                    leaf_value,
                                    cat_lookup_smooth,
                                    lambda_reg,
                                    min_child_weight,
                                ),
                            );
                            continue;
                        }
                    }
                }

                let feature_budget = path_feature_budget(&path_budget, split_result.feat);
                let split_utility = adaptive_candidate_utility(
                    adaptive_growth,
                    binned,
                    gradients,
                    hessians,
                    node_indices,
                    g_sum,
                    h_sum,
                    lambda_reg,
                    l1_reg,
                    min_h,
                    &split_result,
                    depth,
                    max_depth,
                    root_h,
                    active_features,
                    tree_seed,
                    node_idx,
                    feature_budget,
                    leaf_var_shrink,
                );
                if split_utility <= 0.0 {
                    tree.set_leaf(node_idx, leaf_value);
                    continue;
                }

                // Honest arbitration: the chosen split must also show positive gain
                // on the estimation fold at this node, else don't commit.
                let mut est_children: Option<(usize, usize, usize)> = None;
                if !est_buf.is_empty() {
                    if let Some(&(es, ee)) = est_seg.get(&node_idx) {
                        if ee - es >= 8 {
                            let emid =
                                partition_indices_split(&mut est_buf, es, ee, binned, &split_result);
                            let (elg, elh) =
                                child_left_sums(&split_result, gradients, hessians, &est_buf[es..emid]);
                            let (etg, eth) = sum_gh(gradients, hessians, &est_buf[es..ee]);
                            let (erg, erh) = (etg - elg, eth - elh);
                            let est_gain = elg * elg / (elh + lambda_reg)
                                + erg * erg / (erh + lambda_reg)
                                - etg * etg / (eth + lambda_reg);
                            if !est_gain.is_finite() || est_gain <= 0.0 {
                                tree.set_leaf(node_idx, leaf_value);
                                continue;
                            }
                            est_children = Some((es, emid, ee));
                        }
                    }
                }

                let node_len = node_indices.len();
                let left_end =
                    partition_indices_split(&mut row_buf, start, end, binned, &split_result);
                if left_end == start || left_end == end {
                    tree.set_leaf(node_idx, leaf_value);
                    continue;
                }

                let left_indices = &row_buf[start..left_end];
                let (lg, lh) = child_left_sums(&split_result, gradients, hessians, left_indices);
                let rg = g_sum - lg;
                let rh = h_sum - lh;
                let split_structural_cost = adaptive_candidate_structural_cost(
                    depth,
                    max_depth,
                    node_len,
                    h_sum,
                    root_h,
                    active_features.len(),
                    &split_result,
                    lh,
                    rh,
                    feature_budget,
                );
                let child_path_budget = if adaptive_growth {
                    adaptive_update_path_feature_budget(
                        &path_budget,
                        split_result.feat,
                        split_utility,
                        split_structural_cost,
                    )
                } else {
                    path_budget
                };

                let (left_idx, right_idx) =
                    tree.add_split_from_sr(node_idx, split_result, leaf_value);
                if let Some((es, emid, ee)) = est_children {
                    est_seg.insert(left_idx, (es, emid));
                    est_seg.insert(right_idx, (emid, ee));
                }
                stack.push((
                    left_end,
                    end,
                    depth + 1,
                    right_idx,
                    rg,
                    rh,
                    child_path_budget.clone(),
                ));
                stack.push((
                    start,
                    left_end,
                    depth + 1,
                    left_idx,
                    lg,
                    lh,
                    child_path_budget,
                ));
            }
        }

        tree.into_tree()
    }

    /// Honest depthwise builder with complement-debiased split selection (CDSS).
    /// Split score must survive both the structure rows and the honest estimation rows.
    /// This reduces winner's curse at split selection without increasing tree count.
    pub fn build_depthwise_debiased(
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        indices: &[u32],
        complement_indices: &[u32],
        lambda_reg: f64,
        gamma: f64,
        max_depth: usize,
        min_child_weight: f64,
        feature_mask: &[bool],
        colsample_bylevel: f64,
        tree_seed: u64,
        random_strength: f64,
        cat_smooth: f64,
        cat_lookup_smooth: f64,
        monotone_constraints: &[i8],
        gain_penalty: f64,
        extra_trees: bool,
        complement_debias_mode: u8,
        _lookahead_alpha: f64,
        expert_split: bool,
    ) -> Self {
        if complement_debias_mode == 0 || complement_indices.is_empty() || extra_trees {
            return Self::build_depthwise(
                binned,
                gradients,
                hessians,
                indices,
                lambda_reg,
                0.0,
                gamma,
                max_depth,
                min_child_weight,
                feature_mask,
                colsample_bylevel,
                tree_seed,
                random_strength,
                cat_smooth,
                cat_lookup_smooth,
                monotone_constraints,
                gain_penalty,
                extra_trees,
                0.0,
                expert_split,
                false,
                false,
                None,
                false,
                0.0,
                CatPairConfig::default(),

                                None,

                None,

                0.0,

                0.5,
                1.0,
            );
        }

        let max_nodes = (1usize << (max_depth + 1)).min(65536);
        let mut tree = TreeBuilder::new(max_nodes);
        let mut row_buf: Vec<u32> = indices.to_vec();
        let mut comp_row_buf: Vec<u32> = complement_indices.to_vec();

        tree.add_node();
        let (root_g, root_h) = sum_gh(gradients, hessians, &row_buf);
        let (root_comp_g, root_comp_h) = sum_gh(gradients, hessians, &comp_row_buf);

        let min_h = min_child_weight.max(1e-10);
        let tree_features: Vec<usize> = (0..binned.n_features)
            .filter(|&f| feature_mask[f])
            .collect();
        let cbl_n_select = if colsample_bylevel < 1.0 {
            ((colsample_bylevel * tree_features.len() as f64) as usize).max(1)
        } else {
            0
        };
        let mut node_features: Vec<usize> = Vec::with_capacity(tree_features.len());
        let max_bins = (0..binned.n_features)
            .map(|c| binned.n_bins(c))
            .max()
            .unwrap_or(1);

        let mut hist_pool = HistPool::new(tree_features.len(), max_bins);
        let mut g_hist = vec![0.0f64; max_bins];
        let mut h_hist = vec![0.0f64; max_bins];

        let mut root_hists = hist_pool.take();
        build_node_hists(
            binned,
            gradients,
            hessians,
            &row_buf,
            &tree_features,
            &mut root_hists,
        );

        let mut stack: Vec<(
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            f64,
            f64,
            f64,
            f64,
            Option<NodeHists>,
        )> = Vec::with_capacity(max_nodes);
        stack.push((
            0,
            row_buf.len(),
            0,
            comp_row_buf.len(),
            0,
            0,
            root_g,
            root_h,
            root_comp_g,
            root_comp_h,
            Some(root_hists),
        ));

        while let Some((
            start,
            end,
            comp_start,
            comp_end,
            depth,
            node_idx,
            g_sum,
            h_sum,
            comp_g_sum,
            comp_h_sum,
            mut cached_hists,
        )) = stack.pop()
        {
            let node_indices = &row_buf[start..end];
            let node_comp_indices = &comp_row_buf[comp_start..comp_end];
            let n_leaf = (end - start) as f64;
            let leaf_value = -g_sum / (h_sum + lambda_reg + lambda_reg / n_leaf.max(1.0).sqrt());
            tree.set_node_stats(node_idx, h_sum, (end - start) as u32);

            if depth >= max_depth || (end - start) <= 1 || h_sum < min_h {
                tree.set_leaf(node_idx, leaf_value);
                if cat_lookup_smooth > 0.0 {
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
                        tree.set_cll(
                            node_idx,
                            make_cll_lookup(
                                &cll,
                                leaf_value,
                                cat_lookup_smooth,
                                lambda_reg,
                                min_child_weight,
                            ),
                        );
                    }
                }
                continue;
            }

            let active_features: &[usize] = if cbl_n_select > 0 {
                let node_seed = tree_seed.wrapping_mul(2654435761).wrapping_add(
                    (depth as u64)
                        .wrapping_mul(1000003)
                        .wrapping_add(node_idx as u64),
                );
                let mut rng = StdRng::seed_from_u64(node_seed);
                node_features.clear();
                node_features.extend_from_slice(&tree_features);
                node_features.shuffle(&mut rng);
                node_features.truncate(cbl_n_select);
                &node_features
            } else {
                &tree_features
            };
            let split_features: &[usize] = active_features;

            let mut split_result = if let Some(nh) = cached_hists.as_ref() {
                find_best_split_from_hists_debiased(
                    nh,
                    &tree_features,
                    split_features,
                    binned,
                    gradients,
                    hessians,
                    node_comp_indices,
                    g_sum,
                    h_sum,
                    comp_g_sum,
                    comp_h_sum,
                    lambda_reg,
                    gamma,
                    min_h,
                    random_strength,
                    tree_seed.wrapping_add(depth as u64),
                    cat_smooth,
                    monotone_constraints,
                    gain_penalty,
                    complement_debias_mode,
                )
            } else {
                find_best_split_debiased(
                    binned,
                    gradients,
                    hessians,
                    node_indices,
                    node_comp_indices,
                    split_features,
                    g_sum,
                    h_sum,
                    comp_g_sum,
                    comp_h_sum,
                    lambda_reg,
                    gamma,
                    min_h,
                    &mut g_hist,
                    &mut h_hist,
                    random_strength,
                    tree_seed.wrapping_add(depth as u64),
                    cat_smooth,
                    monotone_constraints,
                    gain_penalty,
                    complement_debias_mode,
                )
            };

            if cat_lookup_smooth > 0.0 {
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
                    if cll.gain > split_result.gain.max(0.0) {
                        tree.set_leaf(node_idx, leaf_value);
                        tree.set_cll(
                            node_idx,
                            make_cll_lookup(
                                &cll,
                                leaf_value,
                                cat_lookup_smooth,
                                lambda_reg,
                                min_child_weight,
                            ),
                        );
                        continue;
                    }
                }
            }

            if split_result.gain <= 0.0 || !split_result.gain.is_finite() {
                tree.set_leaf(node_idx, leaf_value);
                continue;
            }

            let left_end = partition_indices_split(&mut row_buf, start, end, binned, &split_result);
            if left_end == start || left_end == end {
                tree.set_leaf(node_idx, leaf_value);
                continue;
            }
            let comp_left_end = partition_indices_split(
                &mut comp_row_buf,
                comp_start,
                comp_end,
                binned,
                &split_result,
            );

            let left_indices = &row_buf[start..left_end];
            let (lg, lh) = child_left_sums(&split_result, gradients, hessians, left_indices);
            let rg = g_sum - lg;
            let rh = h_sum - lh;
            let left_comp_indices = &comp_row_buf[comp_start..comp_left_end];
            let (comp_lg, comp_lh) = sum_gh(gradients, hessians, left_comp_indices);
            let comp_rg = comp_g_sum - comp_lg;
            let comp_rh = comp_h_sum - comp_lh;

            let (left_idx, right_idx) = tree.add_split(
                node_idx,
                split_result.feat as u32,
                split_result.bin as u16,
                leaf_value,
                split_result.missing_left,
                split_result.is_oblique,
                split_result.oblique_feats,
                split_result.oblique_weights,
                split_result.oblique_threshold,
                split_result.is_cat,
                split_result.cat_mask,
            );

            let n_left = left_end - start;
            let n_right = end - left_end;
            let child_depth = depth + 1;
            let left_needs_hists = child_depth < max_depth && n_left > 1;
            let right_needs_hists = child_depth < max_depth && n_right > 1;

            if let Some(ref parent_hists) = cached_hists {
                if left_needs_hists && right_needs_hists {
                    let mut smaller_hists = hist_pool.take();
                    let mut larger_hists = hist_pool.take();
                    if n_left <= n_right {
                        build_node_hists(
                            binned,
                            gradients,
                            hessians,
                            &row_buf[start..left_end],
                            &tree_features,
                            &mut smaller_hists,
                        );
                        subtract_node_hists(parent_hists, &smaller_hists, &mut larger_hists);
                        stack.push((
                            left_end,
                            end,
                            comp_left_end,
                            comp_end,
                            child_depth,
                            right_idx,
                            rg,
                            rh,
                            comp_rg,
                            comp_rh,
                            Some(larger_hists),
                        ));
                        stack.push((
                            start,
                            left_end,
                            comp_start,
                            comp_left_end,
                            child_depth,
                            left_idx,
                            lg,
                            lh,
                            comp_lg,
                            comp_lh,
                            Some(smaller_hists),
                        ));
                    } else {
                        build_node_hists(
                            binned,
                            gradients,
                            hessians,
                            &row_buf[left_end..end],
                            &tree_features,
                            &mut smaller_hists,
                        );
                        subtract_node_hists(parent_hists, &smaller_hists, &mut larger_hists);
                        stack.push((
                            left_end,
                            end,
                            comp_left_end,
                            comp_end,
                            child_depth,
                            right_idx,
                            rg,
                            rh,
                            comp_rg,
                            comp_rh,
                            Some(smaller_hists),
                        ));
                        stack.push((
                            start,
                            left_end,
                            comp_start,
                            comp_left_end,
                            child_depth,
                            left_idx,
                            lg,
                            lh,
                            comp_lg,
                            comp_lh,
                            Some(larger_hists),
                        ));
                    }
                } else if left_needs_hists {
                    let mut left_hists = hist_pool.take();
                    let mut right_tmp = hist_pool.take();
                    build_node_hists(
                        binned,
                        gradients,
                        hessians,
                        &row_buf[left_end..end],
                        &tree_features,
                        &mut right_tmp,
                    );
                    subtract_node_hists(parent_hists, &right_tmp, &mut left_hists);
                    hist_pool.recycle(right_tmp);
                    stack.push((
                        left_end,
                        end,
                        comp_left_end,
                        comp_end,
                        child_depth,
                        right_idx,
                        rg,
                        rh,
                        comp_rg,
                        comp_rh,
                        None,
                    ));
                    stack.push((
                        start,
                        left_end,
                        comp_start,
                        comp_left_end,
                        child_depth,
                        left_idx,
                        lg,
                        lh,
                        comp_lg,
                        comp_lh,
                        Some(left_hists),
                    ));
                } else if right_needs_hists {
                    let mut right_hists = hist_pool.take();
                    let mut left_tmp = hist_pool.take();
                    build_node_hists(
                        binned,
                        gradients,
                        hessians,
                        &row_buf[start..left_end],
                        &tree_features,
                        &mut left_tmp,
                    );
                    subtract_node_hists(parent_hists, &left_tmp, &mut right_hists);
                    hist_pool.recycle(left_tmp);
                    stack.push((
                        left_end,
                        end,
                        comp_left_end,
                        comp_end,
                        child_depth,
                        right_idx,
                        rg,
                        rh,
                        comp_rg,
                        comp_rh,
                        Some(right_hists),
                    ));
                    stack.push((
                        start,
                        left_end,
                        comp_start,
                        comp_left_end,
                        child_depth,
                        left_idx,
                        lg,
                        lh,
                        comp_lg,
                        comp_lh,
                        None,
                    ));
                } else {
                    stack.push((
                        left_end,
                        end,
                        comp_left_end,
                        comp_end,
                        child_depth,
                        right_idx,
                        rg,
                        rh,
                        comp_rg,
                        comp_rh,
                        None,
                    ));
                    stack.push((
                        start,
                        left_end,
                        comp_start,
                        comp_left_end,
                        child_depth,
                        left_idx,
                        lg,
                        lh,
                        comp_lg,
                        comp_lh,
                        None,
                    ));
                }
            } else {
                stack.push((
                    left_end,
                    end,
                    comp_left_end,
                    comp_end,
                    child_depth,
                    right_idx,
                    rg,
                    rh,
                    comp_rg,
                    comp_rh,
                    None,
                ));
                stack.push((
                    start,
                    left_end,
                    comp_start,
                    comp_left_end,
                    child_depth,
                    left_idx,
                    lg,
                    lh,
                    comp_lg,
                    comp_lh,
                    None,
                ));
            }
            if let Some(h) = cached_hists.take() {
                hist_pool.recycle(h);
            }
        }

        tree.into_tree()
    }

    /// Multi-output depthwise tree builder. Evaluates splits by summing gains
    /// across all K classes. Returns a single tree with class-0 leaf values;
    /// caller should refit_leaves for each class.
    pub fn build_depthwise_multi(
        binned: &BinnedData,
        all_gradients: &[f64], // K * n_rows flat: all_gradients[k * n_rows + i]
        all_hessians: &[f64],  // K * n_rows flat
        all_probs: &[f64],     // n_rows * K flat: probs[i * K + k]
        n_classes: usize,
        indices: &[u32],
        lambda_reg: f64,
        gamma: f64,
        max_depth: usize,
        min_child_weight: f64,
        feature_mask: &[bool],
        colsample_bylevel: f64,
        tree_seed: u64,
        random_strength: f64,
        cat_smooth: f64,
        gain_penalty: f64,
        extra_trees: bool,
        coupled_split_gain: bool,
        cat_pair_cfg: CatPairConfig,
        adaptive_growth: bool,
    ) -> Self {
        let n_rows = binned.n_rows;
        let max_nodes = (1usize << (max_depth + 1)).min(65536);
        let mut tree = TreeBuilder::new(max_nodes);
        let mut row_buf: Vec<u32> = indices.to_vec();
        // Stack entries: (start, end, depth, node_idx)
        let mut stack: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(max_nodes);

        // Per-node K g/h sums stored flat: node_g[node * n_classes + k]
        let mut node_g = vec![0.0f64; n_classes * max_nodes];
        let mut node_h = vec![0.0f64; n_classes * max_nodes];

        tree.add_node();

        // Root g/h sums for all classes
        for k in 0..n_classes {
            let base = k * n_rows;
            let mut gk = 0.0f64;
            let mut hk = 0.0f64;
            for &idx in row_buf.iter() {
                gk += all_gradients[base + idx as usize];
                hk += all_hessians[base + idx as usize];
            }
            node_g[k] = gk;
            node_h[k] = hk;
        }
        let root_total_h: f64 = (0..n_classes).map(|k| node_h[k]).sum();
        stack.push((0, row_buf.len(), 0, 0));

        let min_h = min_child_weight.max(1e-10);
        let tree_features: Vec<usize> = (0..binned.n_features)
            .filter(|&f| feature_mask[f])
            .collect();
        let cbl_n_select = if colsample_bylevel < 1.0 {
            ((colsample_bylevel * tree_features.len() as f64) as usize).max(1)
        } else {
            0
        };
        let mut node_features: Vec<usize> = Vec::with_capacity(tree_features.len());

        let max_bins = (0..binned.n_features)
            .map(|c| binned.n_bins(c))
            .max()
            .unwrap_or(1);
        let mut g_hists = vec![0.0f64; n_classes * max_bins];
        let mut h_hists = vec![0.0f64; n_classes * max_bins];
        let mut g_miss = vec![0.0f64; n_classes];
        let mut h_miss = vec![0.0f64; n_classes];
        let mut p_hists = vec![0.0f64; n_classes * max_bins];
        let mut pp_hists = vec![0.0f64; n_classes * n_classes * max_bins];
        let mut p_miss = vec![0.0f64; n_classes];
        let mut pp_miss = vec![0.0f64; n_classes * n_classes];

        while let Some((start, end, depth, node_idx)) = stack.pop() {
            let node_indices = &row_buf[start..end];
            let g_base = node_idx * n_classes;
            let n_leaf = (end - start) as f64;

            // Use class 0 for leaf value (will be refitted for all classes)
            let g0 = node_g[g_base];
            let h0 = node_h[g_base];
            let leaf_value = -g0 / (h0 + lambda_reg + lambda_reg / n_leaf.max(1.0).sqrt());

            // Total hessian across classes for stopping criterion
            let total_h: f64 = (0..n_classes).map(|k| node_h[g_base + k]).sum();
            tree.set_node_stats(node_idx, total_h, (end - start) as u32);

            if depth >= max_depth || (end - start) <= 1 || total_h < min_h {
                tree.set_leaf(node_idx, leaf_value);
                continue;
            }

            // Per-node feature subsampling (more diverse than per-level, RF-like)
            let active_features: &[usize] = if cbl_n_select > 0 {
                let node_seed = tree_seed.wrapping_mul(2654435761).wrapping_add(
                    (depth as u64)
                        .wrapping_mul(1000003)
                        .wrapping_add(node_idx as u64),
                );
                let mut rng = StdRng::seed_from_u64(node_seed);
                node_features.clear();
                node_features.extend_from_slice(&tree_features);
                node_features.shuffle(&mut rng);
                node_features.truncate(cbl_n_select);
                &node_features
            } else {
                &tree_features
            };
            let cat_sort_dir = multiclass_cat_sort_direction(
                &node_g[g_base..g_base + n_classes],
                &node_h[g_base..g_base + n_classes],
                lambda_reg,
            );
            let split_features: &[usize] = active_features;

            let mut split_result = if extra_trees && !coupled_split_gain {
                find_extra_trees_split_multi(
                    binned,
                    all_gradients,
                    all_hessians,
                    n_classes,
                    n_rows,
                    node_indices,
                    split_features,
                    &node_g[g_base..g_base + n_classes],
                    &node_h[g_base..g_base + n_classes],
                    lambda_reg,
                    gamma,
                    min_h,
                    &mut g_hists,
                    &mut h_hists,
                    &mut g_miss,
                    &mut h_miss,
                    tree_seed
                        .wrapping_add(depth as u64)
                        .wrapping_add(node_idx as u64),
                    cat_smooth,
                )
            } else {
                find_best_split_multi(
                    binned,
                    all_gradients,
                    all_hessians,
                    all_probs,
                    n_classes,
                    n_rows,
                    node_indices,
                    split_features,
                    &node_g[g_base..g_base + n_classes],
                    &node_h[g_base..g_base + n_classes],
                    lambda_reg,
                    gamma,
                    min_h,
                    &mut g_hists,
                    &mut h_hists,
                    &mut g_miss,
                    &mut h_miss,
                    &mut p_hists,
                    &mut pp_hists,
                    &mut p_miss,
                    &mut pp_miss,
                    random_strength,
                    tree_seed.wrapping_add(depth as u64),
                    cat_smooth,
                    gain_penalty,
                    coupled_split_gain,
                    &cat_pair_cfg,
                    depth,
                )
            };

            if split_result.gain <= 0.0 || !split_result.gain.is_finite() {
                tree.set_leaf(node_idx, leaf_value);
                continue;
            }

            // Free-tree signal gate, multi-output form: jointly permute the
            // per-row gradient vectors (same row permutation for every class,
            // probs permuted alongside) so cross-class structure survives but
            // the feature-to-gradient link is destroyed; the real split must
            // beat `signal_gate` times the best permuted-scan gain.
            if binned.signal_gate > 0.0
                && node_indices.len() >= 16
                && !(extra_trees && !coupled_split_gain)
            {
                let seed = tree_seed
                    ^ (node_idx as u64).wrapping_mul(0x517C_C1B7_2722_0A95)
                    ^ (depth as u64).wrapping_mul(0x2545_F491_4F6C_DD1D);
                let mut rng = StdRng::seed_from_u64(seed);
                let mut perm: Vec<u32> = node_indices.to_vec();
                perm.shuffle(&mut rng);
                let mut g_perm = all_gradients.to_vec();
                let mut p_perm = all_probs.to_vec();
                for (slot, &dst) in node_indices.iter().enumerate() {
                    let src = perm[slot] as usize;
                    let dst = dst as usize;
                    for k in 0..n_classes {
                        g_perm[k * n_rows + dst] = all_gradients[k * n_rows + src];
                    }
                    if !all_probs.is_empty() {
                        for k in 0..n_classes {
                            p_perm[dst * n_classes + k] = all_probs[src * n_classes + k];
                        }
                    }
                }
                let null = find_best_split_multi(
                    binned,
                    &g_perm,
                    all_hessians,
                    &p_perm,
                    n_classes,
                    n_rows,
                    node_indices,
                    split_features,
                    &node_g[g_base..g_base + n_classes],
                    &node_h[g_base..g_base + n_classes],
                    lambda_reg,
                    0.0,
                    min_h,
                    &mut g_hists,
                    &mut h_hists,
                    &mut g_miss,
                    &mut h_miss,
                    &mut p_hists,
                    &mut pp_hists,
                    &mut p_miss,
                    &mut pp_miss,
                    0.0,
                    seed,
                    cat_smooth,
                    gain_penalty,
                    coupled_split_gain,
                    &cat_pair_cfg,
                    depth,
                );
                let null_gain = if null.gain.is_finite() {
                    null.gain.max(0.0)
                } else {
                    0.0
                };
                if split_result.gain < binned.signal_gate * null_gain {
                    tree.set_leaf(node_idx, leaf_value);
                    continue;
                }
            }

            let left_end = partition_indices_split(&mut row_buf, start, end, binned, &split_result);
            if left_end == start || left_end == end {
                tree.set_leaf(node_idx, leaf_value);
                continue;
            }

            let left_indices = &row_buf[start..left_end];
            let mut child_left_g = vec![0.0f64; n_classes];
            let mut child_left_h = vec![0.0f64; n_classes];
            let mut child_right_g = vec![0.0f64; n_classes];
            let mut child_right_h = vec![0.0f64; n_classes];
            let mut left_total_h = 0.0f64;
            let mut right_total_h = 0.0f64;
            for k in 0..n_classes {
                let kb = k * n_rows;
                let mut lg = 0.0f64;
                let mut lh = 0.0f64;
                for &idx in left_indices {
                    lg += all_gradients[kb + idx as usize];
                    lh += all_hessians[kb + idx as usize];
                }
                child_left_g[k] = lg;
                child_left_h[k] = lh;
                child_right_g[k] = node_g[g_base + k] - lg;
                child_right_h[k] = node_h[g_base + k] - lh;
                left_total_h += child_left_h[k];
                right_total_h += child_right_h[k];
            }
            if adaptive_growth {
                let structural_cost =
                    adaptive_structural_min_gain(
                        depth,
                        max_depth,
                        (end - start) as usize,
                        total_h,
                        root_total_h,
                        active_features.len(),
                        &split_result,
                    ) * adaptive_balance_multiplier(left_total_h, right_total_h, total_h);
                if split_result.gain <= structural_cost {
                    tree.set_leaf(node_idx, leaf_value);
                    continue;
                }
            }

            let (left_idx, right_idx) = tree.add_split_from_sr(node_idx, split_result, leaf_value);

            // Ensure node_g/node_h buffers are large enough for new children
            let needed = (right_idx + 1) * n_classes;
            if needed > node_g.len() {
                node_g.resize(needed, 0.0);
                node_h.resize(needed, 0.0);
            }

            // Store child K sums computed during adaptive-growth auditing.
            let l_base = left_idx * n_classes;
            let r_base = right_idx * n_classes;
            for k in 0..n_classes {
                node_g[l_base + k] = child_left_g[k];
                node_h[l_base + k] = child_left_h[k];
                node_g[r_base + k] = child_right_g[k];
                node_h[r_base + k] = child_right_h[k];
            }

            stack.push((left_end, end, depth + 1, right_idx));
            stack.push((start, left_end, depth + 1, left_idx));
        }

        tree.into_tree()
    }

    /// Shared-structure multiclass oblivious tree: all nodes at a given depth
    /// use the same split, while leaf values are still refit per class later.
    pub fn build_oblivious_multi(
        binned: &BinnedData,
        all_gradients: &[f64],
        all_hessians: &[f64],
        all_probs: &[f64],
        n_classes: usize,
        indices: &[u32],
        lambda_reg: f64,
        gamma: f64,
        max_depth: usize,
        min_child_weight: f64,
        feature_mask: &[bool],
        gain_penalty: f64,
        extra_trees: bool,
        tree_seed: u64,
        coupled_split_gain: bool,
        cat_pair_cfg: CatPairConfig,
    ) -> Self {
        let n_leaves_max = 1usize << max_depth;
        let max_nodes = 2 * n_leaves_max;
        let mut tree = TreeBuilder::new(max_nodes);
        let mut row_buf: Vec<u32> = indices.to_vec();
        let mut node_ranges: Vec<(usize, usize)> = vec![(0, row_buf.len())];
        let mut node_ids: Vec<usize> = vec![tree.add_node()];

        let n_rows = binned.n_rows;
        let min_h = min_child_weight.max(1e-10);
        let active_features: Vec<usize> = (0..binned.n_features)
            .filter(|&f| feature_mask[f])
            .collect();
        let use_coupled_gain =
            coupled_split_gain && n_classes >= 3 && all_probs.len() >= n_rows * n_classes;

        for depth in 0..max_depth {
            let n_nodes = node_ranges.len();
            let mut node_g = vec![0.0f64; n_nodes * n_classes];
            let mut node_h = vec![0.0f64; n_nodes * n_classes];
            let mut node_p = if use_coupled_gain {
                vec![0.0f64; n_nodes * n_classes]
            } else {
                Vec::new()
            };
            let mut node_pp = if use_coupled_gain {
                vec![0.0f64; n_nodes * n_classes * n_classes]
            } else {
                Vec::new()
            };

            for (ni, &(start, end)) in node_ranges.iter().enumerate() {
                let g_base = ni * n_classes;
                let pp_base = ni * n_classes * n_classes;
                for &idx in &row_buf[start..end] {
                    let row = idx as usize;
                    let prob_base = row * n_classes;
                    for k in 0..n_classes {
                        let off = k * n_rows + row;
                        node_g[g_base + k] += all_gradients[off];
                        node_h[g_base + k] += all_hessians[off];
                        if use_coupled_gain {
                            node_p[g_base + k] += all_probs[prob_base + k];
                        }
                    }
                    if use_coupled_gain {
                        for a in 0..n_classes {
                            let pa = all_probs[prob_base + a];
                            let row_base = a * n_classes;
                            for b in 0..n_classes {
                                node_pp[pp_base + row_base + b] += pa * all_probs[prob_base + b];
                            }
                        }
                    }
                }
            }

            let mut node_parent_obj = vec![0.0f64; n_nodes];
            let mut global_g = vec![0.0f64; n_classes];
            let mut global_h = vec![0.0f64; n_classes];
            let mut global_p = if use_coupled_gain {
                vec![0.0f64; n_classes]
            } else {
                Vec::new()
            };
            let mut global_pp = if use_coupled_gain {
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

            for ni in 0..n_nodes {
                let g_base = ni * n_classes;
                let count = (node_ranges[ni].1 - node_ranges[ni].0) as f64;
                let g0 = node_g[g_base];
                let h0 = node_h[g_base];
                let leaf_value = -g0 / (h0 + lambda_reg + lambda_reg / count.max(1.0).sqrt());
                let total_h: f64 = (0..n_classes).map(|k| node_h[g_base + k]).sum();
                tree.set_node_stats(node_ids[ni], total_h, count as u32);
                tree.set_leaf(node_ids[ni], leaf_value);

                for k in 0..n_classes {
                    global_g[k] += node_g[g_base + k];
                    global_h[k] += node_h[g_base + k];
                }
                if use_coupled_gain {
                    for k in 0..n_classes {
                        global_p[k] += node_p[g_base + k];
                    }
                    let pp_base = ni * n_classes * n_classes;
                    for kk in 0..(n_classes * n_classes) {
                        global_pp[kk] += node_pp[pp_base + kk];
                    }
                    node_parent_obj[ni] = dense_multiclass_gain(
                        &node_g[g_base..g_base + n_classes],
                        &node_p[g_base..g_base + n_classes],
                        &node_pp[pp_base..pp_base + n_classes * n_classes],
                        lambda_reg,
                        &mut dense_a,
                        &mut dense_rhs,
                    );
                } else {
                    let mut obj = 0.0f64;
                    for k in 0..n_classes {
                        obj += node_g[g_base + k] * node_g[g_base + k]
                            / (node_h[g_base + k] + lambda_reg);
                    }
                    node_parent_obj[ni] = obj;
                }
            }

            let cat_sort_dir = if use_coupled_gain {
                multiclass_cat_sort_direction_dense(&global_g, &global_p, &global_pp, lambda_reg)
            } else {
                multiclass_cat_sort_direction(&global_g, &global_h, lambda_reg)
            };
            let global_parent_dense_gain = if use_coupled_gain {
                let mut dense_a = vec![0.0f64; n_classes * n_classes];
                let mut dense_rhs = vec![0.0f64; n_classes];
                dense_multiclass_gain(
                    &global_g,
                    &global_p,
                    &global_pp,
                    lambda_reg,
                    &mut dense_a,
                    &mut dense_rhs,
                )
            } else {
                0.0
            };

            type OblivMultiResult = (f64, usize, usize, bool, bool, CatBitmask);
            // Level axis-scan parameterized over the gradient/prob buffers so
            // the free-tree signal gate can re-run it on permuted gradients.
            // Parameter names intentionally shadow the outer buffers.
            let mut scan_axis = |all_gradients: &[f64],
                                 all_probs: &[f64]|
             -> OblivMultiResult {
            let mut best: OblivMultiResult =
                (f64::NEG_INFINITY, 0usize, 0usize, true, false, Vec::new());

            for &feat in &active_features {
                let feat_n_bins = binned.n_bins(feat);
                if feat_n_bins <= 1 {
                    continue;
                }

                let mut flat_g = vec![0.0f64; n_nodes * n_classes * feat_n_bins];
                let mut flat_h = vec![0.0f64; n_nodes * n_classes * feat_n_bins];
                let mut g_miss = vec![0.0f64; n_nodes * n_classes];
                let mut h_miss = vec![0.0f64; n_nodes * n_classes];
                let mut flat_p = if use_coupled_gain {
                    vec![0.0f64; n_nodes * feat_n_bins * n_classes]
                } else {
                    Vec::new()
                };
                let mut flat_pp = if use_coupled_gain {
                    vec![0.0f64; n_nodes * feat_n_bins * n_classes * n_classes]
                } else {
                    Vec::new()
                };
                let mut p_miss = if use_coupled_gain {
                    vec![0.0f64; n_nodes * n_classes]
                } else {
                    Vec::new()
                };
                let mut pp_miss = if use_coupled_gain {
                    vec![0.0f64; n_nodes * n_classes * n_classes]
                } else {
                    Vec::new()
                };

                let col_bins = binned.col_bins(feat);
                for (ni, &(start, end)) in node_ranges.iter().enumerate() {
                    for &idx in &row_buf[start..end] {
                        let row = idx as usize;
                        let bin = col_bins[row];
                        let prob_base = row * n_classes;
                        if bin == MISSING_BIN {
                            let miss_base = ni * n_classes;
                            for k in 0..n_classes {
                                let off = k * n_rows + row;
                                g_miss[miss_base + k] += all_gradients[off];
                                h_miss[miss_base + k] += all_hessians[off];
                                if use_coupled_gain {
                                    p_miss[miss_base + k] += all_probs[prob_base + k];
                                }
                            }
                            if use_coupled_gain {
                                let miss_pp_base = ni * n_classes * n_classes;
                                for a in 0..n_classes {
                                    let pa = all_probs[prob_base + a];
                                    let row_base = a * n_classes;
                                    for b in 0..n_classes {
                                        pp_miss[miss_pp_base + row_base + b] +=
                                            pa * all_probs[prob_base + b];
                                    }
                                }
                            }
                            continue;
                        }

                        let bu = bin as usize;
                        let gh_base = ni * n_classes * feat_n_bins;
                        for k in 0..n_classes {
                            let off = k * n_rows + row;
                            flat_g[gh_base + k * feat_n_bins + bu] += all_gradients[off];
                            flat_h[gh_base + k * feat_n_bins + bu] += all_hessians[off];
                        }
                        if use_coupled_gain {
                            let p_base = ni * feat_n_bins * n_classes + bu * n_classes;
                            for k in 0..n_classes {
                                flat_p[p_base + k] += all_probs[prob_base + k];
                            }
                            let pp_base = ni * feat_n_bins * n_classes * n_classes
                                + bu * n_classes * n_classes;
                            for a in 0..n_classes {
                                let pa = all_probs[prob_base + a];
                                let row_base = a * n_classes;
                                for b in 0..n_classes {
                                    flat_pp[pp_base + row_base + b] +=
                                        pa * all_probs[prob_base + b];
                                }
                            }
                        }
                    }
                }

                let mut feat_best: OblivMultiResult =
                    (f64::NEG_INFINITY, feat, 0, true, false, Vec::new());

                if binned.is_categorical[feat] {
                    let mut cat_bins: Vec<usize> = Vec::new();
                    for bin in 0..feat_n_bins {
                        let mut total_h = 0.0f64;
                        for ni in 0..n_nodes {
                            let gh_base = ni * n_classes * feat_n_bins;
                            for k in 0..n_classes {
                                total_h += flat_h[gh_base + k * feat_n_bins + bin];
                            }
                        }
                        if total_h > 0.0 {
                            cat_bins.push(bin);
                        }
                    }

                    if cat_bins.len() > 1 {
                        let total_h_nm: f64 = (0..n_classes)
                            .map(|k| {
                                global_h[k]
                                    - (0..n_nodes)
                                        .map(|ni| h_miss[ni * n_classes + k])
                                        .sum::<f64>()
                            })
                            .sum();
                        let total_proj_nm: f64 = (0..n_classes)
                            .map(|k| {
                                let miss: f64 =
                                    (0..n_nodes).map(|ni| g_miss[ni * n_classes + k]).sum();
                                (global_g[k] - miss) * cat_sort_dir[k]
                            })
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
                            let miss_h: f64 =
                                (0..n_nodes).map(|ni| h_miss[ni * n_classes + k]).sum();
                            let miss_g: f64 =
                                (0..n_nodes).map(|ni| g_miss[ni * n_classes + k]).sum();
                            parent_updates[k] = -(global_g[k] - miss_g)
                                / (global_h[k] - miss_h + lambda_reg).max(1e-12);
                        }
                        for &bin in &cat_bins {
                            let mut proj_g = 0.0f64;
                            let mut total_h = 0.0f64;
                            let base = bin * n_classes;
                            for k in 0..n_classes {
                                let mut gb = 0.0f64;
                                let mut hb = 0.0f64;
                                for ni in 0..n_nodes {
                                    let gh_base = ni * n_classes * feat_n_bins;
                                    gb += flat_g[gh_base + k * feat_n_bins + bin];
                                    hb += flat_h[gh_base + k * feat_n_bins + bin];
                                }
                                proj_g += gb * cat_sort_dir[k];
                                total_h += hb;
                                bin_updates[base + k] = -(gb + lambda_reg * parent_updates[k])
                                    / (hb + lambda_reg + 1e-12);
                            }
                            scalar_scores[bin] =
                                (proj_g + lambda_reg * node_ratio) / (total_h + lambda_reg);
                        }

                        const EXACT_MULTI_CAT_SUBSET_MAX_BINS: usize = 0;
                        if cat_bins.len() <= EXACT_MULTI_CAT_SUBSET_MAX_BINS {
                            let n_active = cat_bins.len();
                            let exact_search_width = (1usize << n_active.saturating_sub(1))
                                .saturating_sub(1)
                                .max(1);
                            let exact_subset_search_penalty = 0.25
                                * (exact_search_width as f64).ln_1p()
                                * ((n_classes as f64) - 1.0).max(1.0).sqrt();
                            for subset in 1usize..(1usize << n_active.saturating_sub(1)) {
                                let mut subset_bins: Vec<usize> = Vec::new();
                                let mut subset_g = vec![0.0f64; n_nodes * n_classes];
                                let mut subset_h = vec![0.0f64; n_nodes * n_classes];
                                let mut subset_p = if use_coupled_gain {
                                    vec![0.0f64; n_nodes * n_classes]
                                } else {
                                    Vec::new()
                                };
                                let mut subset_pp = if use_coupled_gain {
                                    vec![0.0f64; n_nodes * n_classes * n_classes]
                                } else {
                                    Vec::new()
                                };
                                for (pos, &bin) in cat_bins.iter().enumerate() {
                                    if ((subset >> pos) & 1) == 0 {
                                        continue;
                                    }
                                    subset_bins.push(bin);
                                    for ni in 0..n_nodes {
                                        let gh_base = ni * n_classes * feat_n_bins;
                                        let gc_base = ni * n_classes;
                                        for k in 0..n_classes {
                                            subset_g[gc_base + k] +=
                                                flat_g[gh_base + k * feat_n_bins + bin];
                                            subset_h[gc_base + k] +=
                                                flat_h[gh_base + k * feat_n_bins + bin];
                                        }
                                        if use_coupled_gain {
                                            let p_base =
                                                ni * feat_n_bins * n_classes + bin * n_classes;
                                            for k in 0..n_classes {
                                                subset_p[gc_base + k] += flat_p[p_base + k];
                                            }
                                            let pp_src = ni * feat_n_bins * n_classes * n_classes
                                                + bin * n_classes * n_classes;
                                            let pp_dst = ni * n_classes * n_classes;
                                            for kk in 0..(n_classes * n_classes) {
                                                subset_pp[pp_dst + kk] += flat_pp[pp_src + kk];
                                            }
                                        }
                                    }
                                }
                                if subset_bins.is_empty() {
                                    continue;
                                }
                                for miss_left in [true, false] {
                                    let mut total_gain = 0.0f64;
                                    for ni in 0..n_nodes {
                                        let gc_base = ni * n_classes;
                                        let pp_base = ni * n_classes * n_classes;
                                        let mut left_g = vec![0.0f64; n_classes];
                                        let mut right_g = vec![0.0f64; n_classes];
                                        let mut total_lh = 0.0f64;
                                        let mut total_rh = 0.0f64;
                                        for k in 0..n_classes {
                                            let g_nm = node_g[gc_base + k] - g_miss[gc_base + k];
                                            let h_nm = node_h[gc_base + k] - h_miss[gc_base + k];
                                            let (lg, lh, rg, rh) = if miss_left {
                                                (
                                                    subset_g[gc_base + k] + g_miss[gc_base + k],
                                                    subset_h[gc_base + k] + h_miss[gc_base + k],
                                                    g_nm - subset_g[gc_base + k],
                                                    h_nm - subset_h[gc_base + k],
                                                )
                                            } else {
                                                (
                                                    subset_g[gc_base + k],
                                                    subset_h[gc_base + k],
                                                    g_nm - subset_g[gc_base + k]
                                                        + g_miss[gc_base + k],
                                                    h_nm - subset_h[gc_base + k]
                                                        + h_miss[gc_base + k],
                                                )
                                            };
                                            left_g[k] = lg;
                                            right_g[k] = rg;
                                            total_lh += lh;
                                            total_rh += rh;
                                        }
                                        if total_lh < min_h || total_rh < min_h {
                                            continue;
                                        }

                                        let mut gain = if use_coupled_gain {
                                            let mut left_p = vec![0.0f64; n_classes];
                                            let mut right_p = vec![0.0f64; n_classes];
                                            let mut left_pp = vec![0.0f64; n_classes * n_classes];
                                            let mut right_pp = vec![0.0f64; n_classes * n_classes];
                                            for k in 0..n_classes {
                                                left_p[k] = if miss_left {
                                                    subset_p[gc_base + k] + p_miss[gc_base + k]
                                                } else {
                                                    subset_p[gc_base + k]
                                                };
                                                right_p[k] = node_p[gc_base + k] - left_p[k];
                                            }
                                            for kk in 0..(n_classes * n_classes) {
                                                left_pp[kk] = if miss_left {
                                                    subset_pp[pp_base + kk] + pp_miss[pp_base + kk]
                                                } else {
                                                    subset_pp[pp_base + kk]
                                                };
                                                right_pp[kk] = node_pp[pp_base + kk] - left_pp[kk];
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
                                            0.5 * (left_obj + right_obj - node_parent_obj[ni])
                                                - gamma
                                        } else {
                                            let mut raw = -node_parent_obj[ni];
                                            for k in 0..n_classes {
                                                let lh = if miss_left {
                                                    subset_h[gc_base + k] + h_miss[gc_base + k]
                                                } else {
                                                    subset_h[gc_base + k]
                                                };
                                                let rh = node_h[gc_base + k] - lh;
                                                raw += left_g[k] * left_g[k] / (lh + lambda_reg)
                                                    + right_g[k] * right_g[k] / (rh + lambda_reg);
                                            }
                                            0.5 * raw - gamma
                                        };

                                        if gain_penalty > 0.0 {
                                            let mut pen = 0.0f64;
                                            for k in 0..n_classes {
                                                let lh = if miss_left {
                                                    subset_h[gc_base + k] + h_miss[gc_base + k]
                                                } else {
                                                    subset_h[gc_base + k]
                                                };
                                                let rh = node_h[gc_base + k] - lh;
                                                pen += 1.0 / (lh + lambda_reg)
                                                    + 1.0 / (rh + lambda_reg)
                                                    - 1.0 / (node_h[gc_base + k] + lambda_reg);
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
                                            exact_search_width,
                                        );
                                        gain = categorical_audit_adjusted_gain(
                                            binned,
                                            gain,
                                            total_lh,
                                            total_rh,
                                            total_lh + total_rh,
                                            lambda_reg,
                                            exact_search_width,
                                        );
                                        gain -= exact_subset_search_penalty;
                                        total_gain += gain;
                                    }
                                    if total_gain > feat_best.0 {
                                        let mut mask: CatBitmask = Vec::new();
                                        for &cat_bin in &subset_bins {
                                            bitmask_set(&mut mask, cat_bin);
                                        }
                                        feat_best = (total_gain, feat, 0, miss_left, true, mask);
                                    }
                                }
                            }
                        }

                        let mut eval_cat_order = |ordered_bins: &[usize]| {
                            let mut cum_g = vec![0.0f64; n_nodes * n_classes];
                            let mut cum_h = vec![0.0f64; n_nodes * n_classes];
                            let mut cum_p = if use_coupled_gain {
                                vec![0.0f64; n_nodes * n_classes]
                            } else {
                                Vec::new()
                            };
                            let mut cum_pp = if use_coupled_gain {
                                vec![0.0f64; n_nodes * n_classes * n_classes]
                            } else {
                                Vec::new()
                            };

                            for ci in 0..ordered_bins.len() - 1 {
                                let bin = ordered_bins[ci];
                                for ni in 0..n_nodes {
                                    let gh_base = ni * n_classes * feat_n_bins;
                                    let gc_base = ni * n_classes;
                                    for k in 0..n_classes {
                                        cum_g[gc_base + k] +=
                                            flat_g[gh_base + k * feat_n_bins + bin];
                                        cum_h[gc_base + k] +=
                                            flat_h[gh_base + k * feat_n_bins + bin];
                                    }
                                    if use_coupled_gain {
                                        let p_base = ni * feat_n_bins * n_classes + bin * n_classes;
                                        for k in 0..n_classes {
                                            cum_p[gc_base + k] += flat_p[p_base + k];
                                        }
                                        let pp_base = ni * feat_n_bins * n_classes * n_classes
                                            + bin * n_classes * n_classes;
                                        let cbase = ni * n_classes * n_classes;
                                        for kk in 0..(n_classes * n_classes) {
                                            cum_pp[cbase + kk] += flat_pp[pp_base + kk];
                                        }
                                    }
                                }

                                for miss_left in [true, false] {
                                    let mut total_gain = 0.0f64;
                                    for ni in 0..n_nodes {
                                        let gc_base = ni * n_classes;
                                        let pp_base = ni * n_classes * n_classes;
                                        let mut left_g = vec![0.0f64; n_classes];
                                        let mut right_g = vec![0.0f64; n_classes];
                                        let mut total_lh = 0.0f64;
                                        let mut total_rh = 0.0f64;
                                        for k in 0..n_classes {
                                            let g_nm = node_g[gc_base + k] - g_miss[gc_base + k];
                                            let h_nm = node_h[gc_base + k] - h_miss[gc_base + k];
                                            let (lg, lh, rg, rh) = if miss_left {
                                                (
                                                    cum_g[gc_base + k] + g_miss[gc_base + k],
                                                    cum_h[gc_base + k] + h_miss[gc_base + k],
                                                    g_nm - cum_g[gc_base + k],
                                                    h_nm - cum_h[gc_base + k],
                                                )
                                            } else {
                                                (
                                                    cum_g[gc_base + k],
                                                    cum_h[gc_base + k],
                                                    g_nm - cum_g[gc_base + k] + g_miss[gc_base + k],
                                                    h_nm - cum_h[gc_base + k] + h_miss[gc_base + k],
                                                )
                                            };
                                            left_g[k] = lg;
                                            right_g[k] = rg;
                                            total_lh += lh;
                                            total_rh += rh;
                                        }
                                        if total_lh < min_h || total_rh < min_h {
                                            continue;
                                        }

                                        let mut gain = if use_coupled_gain {
                                            let mut left_p = vec![0.0f64; n_classes];
                                            let mut right_p = vec![0.0f64; n_classes];
                                            let mut left_pp = vec![0.0f64; n_classes * n_classes];
                                            let mut right_pp = vec![0.0f64; n_classes * n_classes];
                                            for k in 0..n_classes {
                                                left_p[k] = if miss_left {
                                                    cum_p[gc_base + k] + p_miss[gc_base + k]
                                                } else {
                                                    cum_p[gc_base + k]
                                                };
                                                right_p[k] = node_p[gc_base + k] - left_p[k];
                                            }
                                            for kk in 0..(n_classes * n_classes) {
                                                left_pp[kk] = if miss_left {
                                                    cum_pp[pp_base + kk] + pp_miss[pp_base + kk]
                                                } else {
                                                    cum_pp[pp_base + kk]
                                                };
                                                right_pp[kk] = node_pp[pp_base + kk] - left_pp[kk];
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
                                            0.5 * (left_obj + right_obj - node_parent_obj[ni])
                                                - gamma
                                        } else {
                                            let mut raw = -node_parent_obj[ni];
                                            for k in 0..n_classes {
                                                let lh = if miss_left {
                                                    cum_h[gc_base + k] + h_miss[gc_base + k]
                                                } else {
                                                    cum_h[gc_base + k]
                                                };
                                                let rh = if miss_left {
                                                    (node_h[gc_base + k] - h_miss[gc_base + k])
                                                        - cum_h[gc_base + k]
                                                } else {
                                                    (node_h[gc_base + k] - h_miss[gc_base + k])
                                                        - cum_h[gc_base + k]
                                                        + h_miss[gc_base + k]
                                                };
                                                raw += left_g[k] * left_g[k] / (lh + lambda_reg)
                                                    + right_g[k] * right_g[k] / (rh + lambda_reg);
                                            }
                                            0.5 * raw - gamma
                                        };

                                        if gain_penalty > 0.0 {
                                            let mut pen = 0.0f64;
                                            for k in 0..n_classes {
                                                let lh = if miss_left {
                                                    cum_h[gc_base + k] + h_miss[gc_base + k]
                                                } else {
                                                    cum_h[gc_base + k]
                                                };
                                                let rh = node_h[gc_base + k] - lh;
                                                pen += 1.0 / (lh + lambda_reg)
                                                    + 1.0 / (rh + lambda_reg)
                                                    - 1.0 / (node_h[gc_base + k] + lambda_reg);
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
                                            ordered_bins.len().saturating_sub(1).max(1),
                                        );
                                        gain = categorical_audit_adjusted_gain(
                                            binned,
                                            gain,
                                            total_lh,
                                            total_rh,
                                            total_lh + total_rh,
                                            lambda_reg,
                                            ordered_bins.len().saturating_sub(1).max(1),
                                        );
                                        total_gain += gain;
                                    }
                                    if total_gain > feat_best.0 {
                                        let mut mask: CatBitmask = Vec::new();
                                        for &cat_bin in &ordered_bins[..=ci] {
                                            bitmask_set(&mut mask, cat_bin);
                                        }
                                        feat_best = (total_gain, feat, 0, miss_left, true, mask);
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
                        if cat_bins.len() <= 32 {
                            eval_cat_order(&cat_bins);
                        }
                        eval_cat_order(&scalar_sorted);

                        if n_classes >= 3 {
                            let contrast_vectors =
                                multiclass_cat_contrast_vectors(&cat_sort_dir, &global_g);
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
                } else {
                    let mut cum_g = vec![0.0f64; n_nodes * n_classes];
                    let mut cum_h = vec![0.0f64; n_nodes * n_classes];
                    let mut cum_p = if use_coupled_gain {
                        vec![0.0f64; n_nodes * n_classes]
                    } else {
                        Vec::new()
                    };
                    let mut cum_pp = if use_coupled_gain {
                        vec![0.0f64; n_nodes * n_classes * n_classes]
                    } else {
                        Vec::new()
                    };

                    let bins_to_try: Vec<usize> = if extra_trees && feat_n_bins > 1 {
                        let h = tree_seed
                            .wrapping_mul(0x517CC1B727220A95)
                            .wrapping_add(feat as u64)
                            .wrapping_add(depth as u64);
                        let h2 = h.wrapping_mul(0x9E3779B97F4A7C15);
                        vec![(h2 >> 33) as usize % (feat_n_bins - 1)]
                    } else {
                        (0..feat_n_bins - 1).collect()
                    };

                    let mut scan_bin = 0usize;
                    for bin in 0..feat_n_bins - 1 {
                        for ni in 0..n_nodes {
                            let gh_base = ni * n_classes * feat_n_bins;
                            let gc_base = ni * n_classes;
                            for k in 0..n_classes {
                                cum_g[gc_base + k] += flat_g[gh_base + k * feat_n_bins + bin];
                                cum_h[gc_base + k] += flat_h[gh_base + k * feat_n_bins + bin];
                            }
                            if use_coupled_gain {
                                let p_base = ni * feat_n_bins * n_classes + bin * n_classes;
                                for k in 0..n_classes {
                                    cum_p[gc_base + k] += flat_p[p_base + k];
                                }
                                let pp_base = ni * feat_n_bins * n_classes * n_classes
                                    + bin * n_classes * n_classes;
                                let cbase = ni * n_classes * n_classes;
                                for kk in 0..(n_classes * n_classes) {
                                    cum_pp[cbase + kk] += flat_pp[pp_base + kk];
                                }
                            }
                        }

                        if scan_bin >= bins_to_try.len() || bin != bins_to_try[scan_bin] {
                            continue;
                        }
                        scan_bin += 1;

                        for miss_left in [true, false] {
                            let mut total_gain = 0.0f64;
                            for ni in 0..n_nodes {
                                let gc_base = ni * n_classes;
                                let pp_base = ni * n_classes * n_classes;
                                let mut left_g = vec![0.0f64; n_classes];
                                let mut right_g = vec![0.0f64; n_classes];
                                let mut total_lh = 0.0f64;
                                let mut total_rh = 0.0f64;
                                for k in 0..n_classes {
                                    let g_nm = node_g[gc_base + k] - g_miss[gc_base + k];
                                    let h_nm = node_h[gc_base + k] - h_miss[gc_base + k];
                                    let (lg, lh, rg, rh) = if miss_left {
                                        (
                                            cum_g[gc_base + k] + g_miss[gc_base + k],
                                            cum_h[gc_base + k] + h_miss[gc_base + k],
                                            g_nm - cum_g[gc_base + k],
                                            h_nm - cum_h[gc_base + k],
                                        )
                                    } else {
                                        (
                                            cum_g[gc_base + k],
                                            cum_h[gc_base + k],
                                            g_nm - cum_g[gc_base + k] + g_miss[gc_base + k],
                                            h_nm - cum_h[gc_base + k] + h_miss[gc_base + k],
                                        )
                                    };
                                    left_g[k] = lg;
                                    right_g[k] = rg;
                                    total_lh += lh;
                                    total_rh += rh;
                                }
                                if total_lh < min_h || total_rh < min_h {
                                    continue;
                                }

                                let mut gain = if use_coupled_gain {
                                    let mut left_p = vec![0.0f64; n_classes];
                                    let mut right_p = vec![0.0f64; n_classes];
                                    let mut left_pp = vec![0.0f64; n_classes * n_classes];
                                    let mut right_pp = vec![0.0f64; n_classes * n_classes];
                                    for k in 0..n_classes {
                                        left_p[k] = if miss_left {
                                            cum_p[gc_base + k] + p_miss[gc_base + k]
                                        } else {
                                            cum_p[gc_base + k]
                                        };
                                        right_p[k] = node_p[gc_base + k] - left_p[k];
                                    }
                                    for kk in 0..(n_classes * n_classes) {
                                        left_pp[kk] = if miss_left {
                                            cum_pp[pp_base + kk] + pp_miss[pp_base + kk]
                                        } else {
                                            cum_pp[pp_base + kk]
                                        };
                                        right_pp[kk] = node_pp[pp_base + kk] - left_pp[kk];
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
                                    0.5 * (left_obj + right_obj - node_parent_obj[ni]) - gamma
                                } else {
                                    let mut raw = -node_parent_obj[ni];
                                    for k in 0..n_classes {
                                        let lh = if miss_left {
                                            cum_h[gc_base + k] + h_miss[gc_base + k]
                                        } else {
                                            cum_h[gc_base + k]
                                        };
                                        let rh = node_h[gc_base + k] - lh;
                                        raw += left_g[k] * left_g[k] / (lh + lambda_reg)
                                            + right_g[k] * right_g[k] / (rh + lambda_reg);
                                    }
                                    0.5 * raw - gamma
                                };

                                if gain_penalty > 0.0 {
                                    let mut pen = 0.0f64;
                                    for k in 0..n_classes {
                                        let lh = if miss_left {
                                            cum_h[gc_base + k] + h_miss[gc_base + k]
                                        } else {
                                            cum_h[gc_base + k]
                                        };
                                        let rh = node_h[gc_base + k] - lh;
                                        pen += 1.0 / (lh + lambda_reg) + 1.0 / (rh + lambda_reg)
                                            - 1.0 / (node_h[gc_base + k] + lambda_reg);
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
                                    feat_n_bins.saturating_sub(1).max(1),
                                );
                                total_gain += gain;
                            }

                            if total_gain > feat_best.0 {
                                feat_best = (total_gain, feat, bin, miss_left, false, Vec::new());
                            }
                        }
                    }
                }

                if feat_best.0 > best.0 {
                    best = feat_best;
                }
            }
            best
            };
            let best = scan_axis(all_gradients, all_probs);

            let mut best_split = SplitResult::axis(best.0, best.1, best.2, best.3, best.4, best.5);
            if cat_pair_cfg.enabled {
                let mut pair = eval_cat_pair_jit_multi_for_node(
                    binned,
                    all_gradients,
                    all_hessians,
                    all_probs,
                    n_classes,
                    n_rows,
                    &row_buf,
                    &active_features,
                    &global_g,
                    &global_h,
                    lambda_reg,
                    gamma,
                    min_h,
                    1.0,
                    gain_penalty,
                    &cat_sort_dir,
                    use_coupled_gain,
                    &global_p,
                    &global_pp,
                    global_parent_dense_gain,
                    depth,
                    0.0,
                    &cat_pair_cfg,
                );
                if pair.gain.is_finite() {
                    let exact_pair_gain = score_oblivious_multi_split(
                        binned,
                        all_gradients,
                        all_hessians,
                        all_probs,
                        n_classes,
                        n_rows,
                        &row_buf,
                        &node_ranges,
                        &node_g,
                        &node_h,
                        &node_p,
                        &node_pp,
                        &node_parent_obj,
                        lambda_reg,
                        gamma,
                        min_h,
                        gain_penalty,
                        use_coupled_gain,
                        &pair,
                    );
                    if exact_pair_gain.is_finite()
                        && exact_pair_gain > best_split.gain.max(0.0) * cat_pair_cfg.gain_margin
                    {
                        pair.gain = exact_pair_gain;
                        best_split = pair;
                    }
                }
            }

            let best_total_gain = best_split.gain;

            if best_total_gain <= 0.0 || !best_total_gain.is_finite() {
                break;
            }

            // Free-tree signal gate, oblivious multi form: jointly permute the
            // per-row gradient vectors within each node at this level (same row
            // permutation across classes, probs alongside) and re-run the axis
            // scan; the level is admitted only if the real total gain beats
            // `signal_gate` times this permutation noise floor.
            if binned.signal_gate > 0.0 && row_buf.len() >= 16 {
                let seed = tree_seed
                    ^ (depth as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ 0x0B11_710B_5EED_2026;
                let mut rng = StdRng::seed_from_u64(seed);
                let mut g_perm = all_gradients.to_vec();
                let mut p_perm = all_probs.to_vec();
                for &(start, end) in &node_ranges {
                    if end - start < 2 {
                        continue;
                    }
                    let mut perm: Vec<u32> = row_buf[start..end].to_vec();
                    perm.shuffle(&mut rng);
                    for (slot, &dst) in row_buf[start..end].iter().enumerate() {
                        let src = perm[slot] as usize;
                        let dst = dst as usize;
                        for k in 0..n_classes {
                            g_perm[k * n_rows + dst] = all_gradients[k * n_rows + src];
                        }
                        if use_coupled_gain {
                            for k in 0..n_classes {
                                p_perm[dst * n_classes + k] = all_probs[src * n_classes + k];
                            }
                        }
                    }
                }
                let null = scan_axis(&g_perm, &p_perm);
                if best_total_gain < binned.signal_gate * null.0.max(0.0) {
                    break;
                }
            }

            let mut new_ranges = Vec::with_capacity(node_ranges.len() * 2);
            let mut new_ids = Vec::with_capacity(node_ids.len() * 2);
            for (ni, &nid) in node_ids.iter().enumerate() {
                let (start, end) = node_ranges[ni];
                if start == end {
                    let (lid, rid) = tree.add_split_from_sr(nid, best_split.clone(), 0.0);
                    tree.set_node_stats(lid, 0.0, 0);
                    tree.set_node_stats(rid, 0.0, 0);
                    tree.set_leaf(lid, 0.0);
                    tree.set_leaf(rid, 0.0);
                    new_ranges.push((start, start));
                    new_ranges.push((start, start));
                    new_ids.push(lid);
                    new_ids.push(rid);
                    continue;
                }

                let left_end =
                    partition_indices_split(&mut row_buf, start, end, binned, &best_split);
                let g_base = ni * n_classes;
                let g0 = node_g[g_base];
                let h0 = node_h[g_base];
                let count = (end - start) as f64;
                let leaf_value = -g0 / (h0 + lambda_reg + lambda_reg / count.max(1.0).sqrt());
                let (lid, rid) = tree.add_split_from_sr(nid, best_split.clone(), leaf_value);

                let left_indices = &row_buf[start..left_end];
                let mut lg = vec![0.0f64; n_classes];
                let mut lh = vec![0.0f64; n_classes];
                for &idx in left_indices {
                    let row = idx as usize;
                    for k in 0..n_classes {
                        let off = k * n_rows + row;
                        lg[k] += all_gradients[off];
                        lh[k] += all_hessians[off];
                    }
                }
                let mut left_total_h = 0.0f64;
                let mut right_total_h = 0.0f64;
                for k in 0..n_classes {
                    left_total_h += lh[k];
                    right_total_h += node_h[g_base + k] - lh[k];
                }
                let n_left = left_indices.len() as f64;
                let n_right = (end - left_end) as f64;
                tree.set_node_stats(lid, left_total_h, left_indices.len() as u32);
                tree.set_node_stats(rid, right_total_h, (end - left_end) as u32);
                tree.set_leaf(
                    lid,
                    -lg[0] / (lh[0] + lambda_reg + lambda_reg / n_left.max(1.0).sqrt()),
                );
                tree.set_leaf(
                    rid,
                    -(node_g[g_base] - lg[0])
                        / ((node_h[g_base] - lh[0])
                            + lambda_reg
                            + lambda_reg / n_right.max(1.0).sqrt()),
                );

                new_ranges.push((start, left_end));
                new_ranges.push((left_end, end));
                new_ids.push(lid);
                new_ids.push(rid);
            }

            node_ranges = new_ranges;
            node_ids = new_ids;
        }

        tree.into_tree()
    }

    pub fn build_leafwise(
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        indices: &[u32],
        lambda_reg: f64,
        l1_reg: f64,
        gamma: f64,
        max_depth: usize,
        max_leaves: usize,
        min_child_weight: f64,
        feature_mask: &[bool],
        colsample_bylevel: f64,
        tree_seed: u64,
        random_strength: f64,
        cat_smooth: f64,
        cat_lookup_smooth: f64,
        monotone_constraints: &[i8],
        gain_penalty: f64,
        extra_trees: bool,
        sparse_oblique_splits: bool,
        interval_splits: bool,
        adaptive_growth: bool,
        trunk1_balanced: bool,
        leaf_var_shrink: f64,
        cat_pair_cfg: CatPairConfig,
        mut row_leaf_out: Option<&mut Vec<u32>>,
    ) -> Self {
        let max_nodes = max_leaves * 2 + 2;
        let mut tree = TreeBuilder::new(max_nodes);
        let mut row_buf: Vec<u32> = indices.to_vec();
        let mut heap: BinaryHeap<SplitCandidate> = BinaryHeap::with_capacity(max_leaves);

        // Fused-margin capture: stamp each row's CURRENT leaf at every child
        // creation; later splits re-stamp, so the final stamp is the row's true
        // leaf. Rows outside `indices` (subsampling) stay u32::MAX and fall
        // back to traversal in the consumer.
        if let Some(out) = row_leaf_out.as_deref_mut() {
            out.clear();
            out.resize(binned.n_rows, u32::MAX);
        }

        tree.add_node();

        let min_h = min_child_weight.max(1e-10);
        let tree_features: Vec<usize> = (0..binned.n_features)
            .filter(|&f| feature_mask[f])
            .collect();
        // Pre-generate per-level feature subsets for colsample_bylevel
        let level_features: Vec<Vec<usize>> = if colsample_bylevel < 1.0 {
            let mut level_rng = StdRng::seed_from_u64(tree_seed.wrapping_mul(2654435761));
            (0..max_depth + 1)
                .map(|_| {
                    let n_select =
                        ((colsample_bylevel * tree_features.len() as f64) as usize).max(1);
                    let mut shuffled = tree_features.clone();
                    shuffled.shuffle(&mut level_rng);
                    shuffled.truncate(n_select);
                    shuffled
                })
                .collect()
        } else {
            Vec::new()
        };
        let max_bins = (0..binned.n_features)
            .map(|c| binned.n_bins(c))
            .max()
            .unwrap_or(1);
        let mut g_hist = vec![0.0f64; max_bins];
        let mut h_hist = vec![0.0f64; max_bins];
        let mut hist_pool = HistPool::new(tree_features.len(), max_bins);
        let hist_cache_min_rows = max_bins.saturating_mul(64).max(8192);
        let use_hist_cache =
            !extra_trees && indices.len() >= hist_cache_min_rows && tree_features.len() >= 4;

        let get_features = |depth: usize| -> &Vec<usize> {
            if colsample_bylevel < 1.0 && depth < level_features.len() {
                &level_features[depth]
            } else {
                &tree_features
            }
        };

        let root_indices = &row_buf[0..indices.len()];
        let (g_sum, h_sum) = sum_gh(gradients, hessians, root_indices);
        let root_h_total = h_sum;
        let n_root = indices.len() as f64;
        let root_leaf_val = l1_leaf_value(
            g_sum,
            h_sum,
            lambda_reg + lambda_reg / n_root.max(1.0).sqrt(),
            l1_reg,
        );
        tree.set_node_stats(0, h_sum, indices.len() as u32);
        tree.set_leaf(0, root_leaf_val);
        if let Some(out) = row_leaf_out.as_deref_mut() {
            for &r in indices {
                out[r as usize] = 0;
            }
        }

        if indices.len() > 1 && h_sum >= min_h && max_depth > 0 {
            let mut root_hists_for_push = None;
            let mut sr = if extra_trees {
                find_extra_trees_split(
                    binned,
                    gradients,
                    hessians,
                    root_indices,
                    get_features(0),
                    g_sum,
                    h_sum,
                    lambda_reg,
                    gamma,
                    min_h,
                    &mut g_hist,
                    &mut h_hist,
                    tree_seed,
                    monotone_constraints,
                )
            } else if use_hist_cache {
                let mut root_hists = hist_pool.take();
                build_node_hists(
                    binned,
                    gradients,
                    hessians,
                    root_indices,
                    &tree_features,
                    &mut root_hists,
                );
                let sr = find_best_split_from_hists(
                    &root_hists,
                    &tree_features,
                    get_features(0),
                    binned,
                    Some(gradients),
                    Some(hessians),
                    Some(root_indices),
                    g_sum,
                    h_sum,
                    lambda_reg,
                    l1_reg,
                    gamma,
                    min_h,
                    random_strength,
                    tree_seed,
                    cat_smooth,
                    monotone_constraints,
                    gain_penalty,
                    interval_splits,
                
                    None,
                );
                root_hists_for_push = Some(root_hists);
                sr
            } else {
                find_best_split(
                    binned,
                    gradients,
                    hessians,
                    root_indices,
                    get_features(0),
                    g_sum,
                    h_sum,
                    lambda_reg,
                    l1_reg,
                    gamma,
                    min_h,
                    &mut g_hist,
                    &mut h_hist,
                    random_strength,
                    tree_seed,
                    cat_smooth,
                    monotone_constraints,
                    gain_penalty,
                    interval_splits,
                )
            };
            if cat_pair_cfg.enabled && !sr.is_oblique {
                let pair = eval_cat_pair_jit_for_node(
                    binned,
                    gradients,
                    hessians,
                    root_indices,
                    get_features(0),
                    g_sum,
                    h_sum,
                    lambda_reg,
                    l1_reg,
                    gamma,
                    min_h,
                    cat_smooth,
                    0,
                    sr.gain,
                    &cat_pair_cfg,
                );
                if pair.gain > sr.gain {
                    sr = pair;
                }
            }
            if sparse_oblique_splits
                && !extra_trees
                && root_indices.len() >= 16
                && get_features(0).len() >= 2
            {
                let oblique = find_sparse_oblique_split(
                    binned,
                    gradients,
                    hessians,
                    root_indices,
                    get_features(0),
                    g_sum,
                    h_sum,
                    lambda_reg,
                    gamma,
                    min_h,
                    monotone_constraints,
                    None,
                );
                if oblique.gain.is_finite() && oblique.gain > sr.gain {
                    sr = oblique;
                }
            }
            let split_utility = adaptive_candidate_utility(
                adaptive_growth,
                binned,
                gradients,
                hessians,
                root_indices,
                g_sum,
                h_sum,
                lambda_reg,
                l1_reg,
                min_h,
                &sr,
                0,
                max_depth,
                root_h_total,
                get_features(0),
                tree_seed,
                0,
                0.0,
                leaf_var_shrink,
            );
            let mut push_split = split_utility > 0.0;
            if cat_lookup_smooth > 0.0 {
                if let Some(cll) = eval_cll_for_node(
                    binned,
                    gradients,
                    hessians,
                    root_indices,
                    g_sum,
                    h_sum,
                    lambda_reg,
                    gamma,
                    min_child_weight,
                ) {
                    if cll.gain > sr.gain.max(0.0) {
                        tree.set_cll(
                            0,
                            make_cll_lookup(
                                &cll,
                                root_leaf_val,
                                cat_lookup_smooth,
                                lambda_reg,
                                min_child_weight,
                            ),
                        );
                        push_split = false;
                    }
                }
            }
            if push_split {
                let priority = leafwise_heap_priority(
                    split_utility,
                    trunk1_balanced,
                    binned,
                    hessians,
                    root_indices,
                    h_sum,
                    &sr,
                );
                heap.push(SplitCandidate {
                    gain: split_utility,
                    priority,
                    node_idx: 0,
                    start: 0,
                    end: indices.len(),
                    depth: 0,
                    split: sr,
                    g_sum,
                    h_sum,
                    hists: root_hists_for_push,
                });
            } else if let Some(h) = root_hists_for_push {
                hist_pool.recycle(h);
            }
        } else if cat_lookup_smooth > 0.0 && indices.len() > 1 {
            if let Some(cll) = eval_cll_for_node(
                binned,
                gradients,
                hessians,
                root_indices,
                g_sum,
                h_sum,
                lambda_reg,
                gamma,
                min_child_weight,
            ) {
                tree.set_cll(
                    0,
                    make_cll_lookup(
                        &cll,
                        root_leaf_val,
                        cat_lookup_smooth,
                        lambda_reg,
                        min_child_weight,
                    ),
                );
            }
        }

        let mut n_leaves = 1usize;

        while let Some(cand) = heap.pop() {
            if n_leaves >= max_leaves {
                break;
            }

            let left_end =
                partition_indices_split(&mut row_buf, cand.start, cand.end, binned, &cand.split);
            if left_end == cand.start || left_end == cand.end {
                continue;
            }

            let n_cand = (cand.end - cand.start) as f64;
            let leaf_value = l1_leaf_value(
                cand.g_sum,
                cand.h_sum,
                lambda_reg + lambda_reg / n_cand.max(1.0).sqrt(),
                l1_reg,
            );
            let (left_idx, right_idx) =
                tree.add_split_from_sr(cand.node_idx, cand.split.clone(), leaf_value);
            n_leaves += 1;

            let child_depth = cand.depth + 1;
            let child_feats = get_features(child_depth);

            // Left child
            let left_indices = &row_buf[cand.start..left_end];
            let (lg, lh) =
                if cand.split.child_g_left.is_finite() && cand.split.child_h_left.is_finite() {
                    (cand.split.child_g_left, cand.split.child_h_left)
                } else {
                    sum_gh(gradients, hessians, left_indices)
                };
            let n_left = left_indices.len() as f64;
            let left_leaf_val = l1_leaf_value(
                lg,
                lh,
                lambda_reg + lambda_reg / n_left.max(1.0).sqrt(),
                l1_reg,
            );
            tree.set_node_stats(left_idx, lh, left_indices.len() as u32);
            tree.set_leaf(left_idx, left_leaf_val);

            // Right child: derive sums from parent - left (avoids scanning right indices)
            let right_indices = &row_buf[left_end..cand.end];
            let rg = cand.g_sum - lg;
            let rh = cand.h_sum - lh;
            let n_right = right_indices.len() as f64;
            let right_leaf_val = l1_leaf_value(
                rg,
                rh,
                lambda_reg + lambda_reg / n_right.max(1.0).sqrt(),
                l1_reg,
            );
            tree.set_node_stats(right_idx, rh, right_indices.len() as u32);
            tree.set_leaf(right_idx, right_leaf_val);
            if let Some(out) = row_leaf_out.as_deref_mut() {
                for &r in left_indices {
                    out[r as usize] = left_idx as u32;
                }
                for &r in right_indices {
                    out[r as usize] = right_idx as u32;
                }
            }

            let left_can_split = left_indices.len() > 1
                && lh >= min_h
                && child_depth < max_depth
                && n_leaves < max_leaves;
            let right_can_split = right_indices.len() > 1
                && rh >= min_h
                && child_depth < max_depth
                && n_leaves < max_leaves;

            let mut left_hists: Option<NodeHists> = None;
            let mut right_hists: Option<NodeHists> = None;
            let parent_hists = cand.hists;
            if use_hist_cache {
                if let Some(ref parent_hists_ref) = parent_hists {
                    match (left_can_split, right_can_split) {
                        (true, true) => {
                            let mut smaller_hists = hist_pool.take();
                            let mut larger_hists = hist_pool.take();
                            if left_indices.len() <= right_indices.len() {
                                build_node_hists(
                                    binned,
                                    gradients,
                                    hessians,
                                    left_indices,
                                    &tree_features,
                                    &mut smaller_hists,
                                );
                                subtract_node_hists(
                                    parent_hists_ref,
                                    &smaller_hists,
                                    &mut larger_hists,
                                );
                                left_hists = Some(smaller_hists);
                                right_hists = Some(larger_hists);
                            } else {
                                build_node_hists(
                                    binned,
                                    gradients,
                                    hessians,
                                    right_indices,
                                    &tree_features,
                                    &mut smaller_hists,
                                );
                                subtract_node_hists(
                                    parent_hists_ref,
                                    &smaller_hists,
                                    &mut larger_hists,
                                );
                                right_hists = Some(smaller_hists);
                                left_hists = Some(larger_hists);
                            }
                        }
                        (true, false) => {
                            if left_indices.len() <= right_indices.len() {
                                let mut h = hist_pool.take();
                                build_node_hists(
                                    binned,
                                    gradients,
                                    hessians,
                                    left_indices,
                                    &tree_features,
                                    &mut h,
                                );
                                left_hists = Some(h);
                            } else {
                                let mut right_tmp = hist_pool.take();
                                let mut left_derived = hist_pool.take();
                                build_node_hists(
                                    binned,
                                    gradients,
                                    hessians,
                                    right_indices,
                                    &tree_features,
                                    &mut right_tmp,
                                );
                                subtract_node_hists(
                                    parent_hists_ref,
                                    &right_tmp,
                                    &mut left_derived,
                                );
                                hist_pool.recycle(right_tmp);
                                left_hists = Some(left_derived);
                            }
                        }
                        (false, true) => {
                            if right_indices.len() <= left_indices.len() {
                                let mut h = hist_pool.take();
                                build_node_hists(
                                    binned,
                                    gradients,
                                    hessians,
                                    right_indices,
                                    &tree_features,
                                    &mut h,
                                );
                                right_hists = Some(h);
                            } else {
                                let mut left_tmp = hist_pool.take();
                                let mut right_derived = hist_pool.take();
                                build_node_hists(
                                    binned,
                                    gradients,
                                    hessians,
                                    left_indices,
                                    &tree_features,
                                    &mut left_tmp,
                                );
                                subtract_node_hists(
                                    parent_hists_ref,
                                    &left_tmp,
                                    &mut right_derived,
                                );
                                hist_pool.recycle(left_tmp);
                                right_hists = Some(right_derived);
                            }
                        }
                        (false, false) => {}
                    }
                } else {
                    if left_can_split {
                        let mut h = hist_pool.take();
                        build_node_hists(
                            binned,
                            gradients,
                            hessians,
                            left_indices,
                            &tree_features,
                            &mut h,
                        );
                        left_hists = Some(h);
                    }
                    if right_can_split {
                        let mut h = hist_pool.take();
                        build_node_hists(
                            binned,
                            gradients,
                            hessians,
                            right_indices,
                            &tree_features,
                            &mut h,
                        );
                        right_hists = Some(h);
                    }
                }
            }
            if let Some(h) = parent_hists {
                hist_pool.recycle(h);
            }

            if left_can_split {
                let mut sr = if extra_trees {
                    find_extra_trees_split(
                        binned,
                        gradients,
                        hessians,
                        left_indices,
                        child_feats,
                        lg,
                        lh,
                        lambda_reg,
                        gamma,
                        min_h,
                        &mut g_hist,
                        &mut h_hist,
                        tree_seed
                            .wrapping_add(child_depth as u64)
                            .wrapping_add(left_idx as u64),
                        monotone_constraints,
                    )
                } else if let Some(ref hists) = left_hists {
                    find_best_split_from_hists(
                        hists,
                        &tree_features,
                        child_feats,
                        binned,
                        Some(gradients),
                        Some(hessians),
                        Some(left_indices),
                        lg,
                        lh,
                        lambda_reg,
                        l1_reg,
                        gamma,
                        min_h,
                        random_strength,
                        tree_seed.wrapping_add(child_depth as u64),
                        cat_smooth,
                        monotone_constraints,
                        gain_penalty,
                        interval_splits,
                    
                    None,
                )
                } else {
                    find_best_split(
                        binned,
                        gradients,
                        hessians,
                        left_indices,
                        child_feats,
                        lg,
                        lh,
                        lambda_reg,
                        l1_reg,
                        gamma,
                        min_h,
                        &mut g_hist,
                        &mut h_hist,
                        random_strength,
                        tree_seed.wrapping_add(child_depth as u64),
                        cat_smooth,
                        monotone_constraints,
                        gain_penalty,
                        interval_splits,
                    )
                };
                if cat_pair_cfg.enabled && !sr.is_oblique {
                    let pair = eval_cat_pair_jit_for_node(
                        binned,
                        gradients,
                        hessians,
                        left_indices,
                        child_feats,
                        lg,
                        lh,
                        lambda_reg,
                        l1_reg,
                        gamma,
                        min_h,
                        cat_smooth,
                        child_depth,
                        sr.gain,
                        &cat_pair_cfg,
                    );
                    if pair.gain > sr.gain {
                        sr = pair;
                    }
                }
                if sparse_oblique_splits
                    && !extra_trees
                    && left_indices.len() >= 16
                    && child_feats.len() >= 2
                {
                    let oblique = find_sparse_oblique_split(
                        binned,
                        gradients,
                        hessians,
                        left_indices,
                        child_feats,
                        lg,
                        lh,
                        lambda_reg,
                        gamma,
                        min_h,
                        monotone_constraints,
                        None,
                    );
                    if oblique.gain.is_finite() && oblique.gain > sr.gain {
                        sr = oblique;
                    }
                }
                let split_utility = adaptive_candidate_utility(
                    adaptive_growth,
                    binned,
                    gradients,
                    hessians,
                    left_indices,
                    lg,
                    lh,
                    lambda_reg,
                    l1_reg,
                    min_h,
                    &sr,
                    child_depth,
                    max_depth,
                    root_h_total,
                    child_feats,
                    tree_seed,
                    left_idx,
                    0.0,
                    leaf_var_shrink,
                );
                let mut push_split = split_utility > 0.0;
                if cat_lookup_smooth > 0.0 {
                    if let Some(cll) = eval_cll_for_node(
                        binned,
                        gradients,
                        hessians,
                        left_indices,
                        lg,
                        lh,
                        lambda_reg,
                        gamma,
                        min_child_weight,
                    ) {
                        if cll.gain > sr.gain.max(0.0) {
                            tree.set_cll(
                                left_idx,
                                make_cll_lookup(
                                    &cll,
                                    left_leaf_val,
                                    cat_lookup_smooth,
                                    lambda_reg,
                                    min_child_weight,
                                ),
                            );
                            push_split = false;
                        }
                    }
                }
                if push_split {
                    let priority = leafwise_heap_priority(
                        split_utility,
                        trunk1_balanced,
                        binned,
                        hessians,
                        left_indices,
                        lh,
                        &sr,
                    );
                    heap.push(SplitCandidate {
                        gain: split_utility,
                        priority,
                        node_idx: left_idx,
                        start: cand.start,
                        end: left_end,
                        depth: child_depth,
                        split: sr,
                        g_sum: lg,
                        h_sum: lh,
                        hists: left_hists,
                    });
                } else if let Some(h) = left_hists {
                    hist_pool.recycle(h);
                }
            } else if cat_lookup_smooth > 0.0 && left_indices.len() > 1 {
                if let Some(h) = left_hists {
                    hist_pool.recycle(h);
                }
                if let Some(cll) = eval_cll_for_node(
                    binned,
                    gradients,
                    hessians,
                    left_indices,
                    lg,
                    lh,
                    lambda_reg,
                    gamma,
                    min_child_weight,
                ) {
                    tree.set_cll(
                        left_idx,
                        make_cll_lookup(
                            &cll,
                            left_leaf_val,
                            cat_lookup_smooth,
                            lambda_reg,
                            min_child_weight,
                        ),
                    );
                }
            } else if let Some(h) = left_hists {
                hist_pool.recycle(h);
            }

            if right_can_split {
                let mut sr = if extra_trees {
                    find_extra_trees_split(
                        binned,
                        gradients,
                        hessians,
                        right_indices,
                        child_feats,
                        rg,
                        rh,
                        lambda_reg,
                        gamma,
                        min_h,
                        &mut g_hist,
                        &mut h_hist,
                        tree_seed
                            .wrapping_add(child_depth as u64)
                            .wrapping_add(right_idx as u64),
                        monotone_constraints,
                    )
                } else if let Some(ref hists) = right_hists {
                    find_best_split_from_hists(
                        hists,
                        &tree_features,
                        child_feats,
                        binned,
                        Some(gradients),
                        Some(hessians),
                        Some(right_indices),
                        rg,
                        rh,
                        lambda_reg,
                        l1_reg,
                        gamma,
                        min_h,
                        random_strength,
                        tree_seed.wrapping_add(child_depth as u64),
                        cat_smooth,
                        monotone_constraints,
                        gain_penalty,
                        interval_splits,
                    
                    None,
                )
                } else {
                    find_best_split(
                        binned,
                        gradients,
                        hessians,
                        right_indices,
                        child_feats,
                        rg,
                        rh,
                        lambda_reg,
                        l1_reg,
                        gamma,
                        min_h,
                        &mut g_hist,
                        &mut h_hist,
                        random_strength,
                        tree_seed.wrapping_add(child_depth as u64),
                        cat_smooth,
                        monotone_constraints,
                        gain_penalty,
                        interval_splits,
                    )
                };
                if cat_pair_cfg.enabled && !sr.is_oblique {
                    let pair = eval_cat_pair_jit_for_node(
                        binned,
                        gradients,
                        hessians,
                        right_indices,
                        child_feats,
                        rg,
                        rh,
                        lambda_reg,
                        l1_reg,
                        gamma,
                        min_h,
                        cat_smooth,
                        child_depth,
                        sr.gain,
                        &cat_pair_cfg,
                    );
                    if pair.gain > sr.gain {
                        sr = pair;
                    }
                }
                if sparse_oblique_splits
                    && !extra_trees
                    && right_indices.len() >= 16
                    && child_feats.len() >= 2
                {
                    let oblique = find_sparse_oblique_split(
                        binned,
                        gradients,
                        hessians,
                        right_indices,
                        child_feats,
                        rg,
                        rh,
                        lambda_reg,
                        gamma,
                        min_h,
                        monotone_constraints,
                        None,
                    );
                    if oblique.gain.is_finite() && oblique.gain > sr.gain {
                        sr = oblique;
                    }
                }
                let split_utility = adaptive_candidate_utility(
                    adaptive_growth,
                    binned,
                    gradients,
                    hessians,
                    right_indices,
                    rg,
                    rh,
                    lambda_reg,
                    l1_reg,
                    min_h,
                    &sr,
                    child_depth,
                    max_depth,
                    root_h_total,
                    child_feats,
                    tree_seed,
                    right_idx,
                    0.0,
                    leaf_var_shrink,
                );
                let mut push_split = split_utility > 0.0;
                if cat_lookup_smooth > 0.0 {
                    if let Some(cll) = eval_cll_for_node(
                        binned,
                        gradients,
                        hessians,
                        right_indices,
                        rg,
                        rh,
                        lambda_reg,
                        gamma,
                        min_child_weight,
                    ) {
                        if cll.gain > sr.gain.max(0.0) {
                            tree.set_cll(
                                right_idx,
                                make_cll_lookup(
                                    &cll,
                                    right_leaf_val,
                                    cat_lookup_smooth,
                                    lambda_reg,
                                    min_child_weight,
                                ),
                            );
                            push_split = false;
                        }
                    }
                }
                if push_split {
                    let priority = leafwise_heap_priority(
                        split_utility,
                        trunk1_balanced,
                        binned,
                        hessians,
                        right_indices,
                        rh,
                        &sr,
                    );
                    heap.push(SplitCandidate {
                        gain: split_utility,
                        priority,
                        node_idx: right_idx,
                        start: left_end,
                        end: cand.end,
                        depth: child_depth,
                        split: sr,
                        g_sum: rg,
                        h_sum: rh,
                        hists: right_hists,
                    });
                } else if let Some(h) = right_hists {
                    hist_pool.recycle(h);
                }
            } else if cat_lookup_smooth > 0.0 && right_indices.len() > 1 {
                if let Some(h) = right_hists {
                    hist_pool.recycle(h);
                }
                if let Some(cll) = eval_cll_for_node(
                    binned,
                    gradients,
                    hessians,
                    right_indices,
                    rg,
                    rh,
                    lambda_reg,
                    gamma,
                    min_child_weight,
                ) {
                    tree.set_cll(
                        right_idx,
                        make_cll_lookup(
                            &cll,
                            right_leaf_val,
                            cat_lookup_smooth,
                            lambda_reg,
                            min_child_weight,
                        ),
                    );
                }
            } else if let Some(h) = right_hists {
                hist_pool.recycle(h);
            }
        }

        tree.into_tree()
    }

    /// Build an oblivious (symmetric) tree: all nodes at the same depth share the same split.
    /// This is CatBoost's approach — strong regularization, 2^depth leaves.
    pub fn build_oblivious(
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        indices: &[u32],
        lambda_reg: f64,
        l1_reg: f64,
        gamma: f64,
        max_depth: usize,
        min_child_weight: f64,
        feature_mask: &[bool],
        gain_penalty: f64,
        extra_trees: bool,
        tree_seed: u64,
        leaf_var_shrink: f64,
    ) -> Self {
        let n_leaves_max = 1usize << max_depth;
        let max_nodes = 2 * n_leaves_max;
        let mut tree = TreeBuilder::new(max_nodes);

        // row_buf partitioned into groups for each node at current level
        let mut row_buf: Vec<u32> = indices.to_vec();
        // (start, end) ranges in row_buf for each node at current depth
        let mut node_ranges: Vec<(usize, usize)> = vec![(0, row_buf.len())];
        // node indices in the tree builder
        let mut node_ids: Vec<usize> = vec![tree.add_node()];

        let min_h = min_child_weight.max(1e-10);
        let active_features: Vec<usize> = (0..binned.n_features)
            .filter(|&f| feature_mask[f])
            .collect();
        let max_bins = (0..binned.n_features)
            .map(|c| binned.n_bins(c))
            .max()
            .unwrap_or(1);

        for _depth in 0..max_depth {
            // Set leaf values for all nodes at this level
            let mut node_gh: Vec<(f64, f64)> = Vec::with_capacity(node_ranges.len());
            let mut node_g2n: Vec<(f64, f64)> = Vec::with_capacity(node_ranges.len());
            for &(start, end) in &node_ranges {
                let mut g = 0.0f64;
                let mut h = 0.0f64;
                let mut g2 = 0.0f64;
                let mut n = 0.0f64;
                for &idx in &row_buf[start..end] {
                    let row = idx as usize;
                    let gi = gradients[row];
                    g += gi;
                    h += hessians[row];
                    g2 += gi * gi;
                    n += 1.0;
                }
                node_gh.push((g, h));
                node_g2n.push((g2, n));
            }
            for (i, &nid) in node_ids.iter().enumerate() {
                let (g, h) = node_gh[i];
                let (ns, ne) = node_ranges[i];
                let nc = (ne - ns) as f64;
                tree.set_node_stats(nid, h, (ne - ns) as u32);
                tree.set_leaf(
                    nid,
                    l1_leaf_value(g, h, lambda_reg + lambda_reg / nc.max(1.0).sqrt(), l1_reg),
                );
            }

            // Find the BEST SINGLE SPLIT across ALL nodes at this level
            let mut best_total_gain = 0.0f64;
            let mut best_feat = 0usize;
            let mut best_bin = 0usize;
            let mut best_missing_left = true;
            let mut best_is_cat = false;
            let mut best_cat_mask: CatBitmask = Vec::new();

            // Parallel feature evaluation for oblivious splits
            let n_nodes = node_ranges.len();
            let total_rows: usize = node_ranges.iter().map(|&(s, e)| e - s).sum();
            let use_par = active_features.len() >= 4
                && total_rows * active_features.len() >= PAR_SPLIT_THRESHOLD;

            // Result type: (total_gain, feat, bin, missing_left, is_cat, cat_mask)
            type OblivResult = (f64, usize, usize, bool, bool, CatBitmask);
            let empty_result: OblivResult = (0.0, 0, 0, true, false, Vec::new());

            let eval_feat_obliv = |feat: usize, grads: &[f64]| -> OblivResult {
                let feat_n_bins = binned.n_bins(feat);
                if feat_n_bins <= 1 {
                    return (f64::NEG_INFINITY, feat, 0, true, false, Vec::new());
                }

                // Per-thread histogram buffers
                let mut flat_g = vec![0.0f64; n_nodes * feat_n_bins];
                let mut flat_h = vec![0.0f64; n_nodes * feat_n_bins];
                let mut flat_g2 = vec![0.0f64; n_nodes * feat_n_bins];
                let mut flat_n = vec![0.0f64; n_nodes * feat_n_bins];
                let mut g_miss = vec![0.0f64; n_nodes];
                let mut h_miss = vec![0.0f64; n_nodes];
                let mut g2_miss = vec![0.0f64; n_nodes];
                let mut n_miss = vec![0.0f64; n_nodes];

                let col_offset = feat * binned.n_rows;
                for ni in 0..n_nodes {
                    let (start, end) = node_ranges[ni];
                    let base = ni * feat_n_bins;
                    for &idx in &row_buf[start..end] {
                        let bin = binned.bin_indices[col_offset + idx as usize] as usize;
                        let g = grads[idx as usize];
                        let h = hessians[idx as usize];
                        if bin == MISSING_BIN as usize {
                            g_miss[ni] += g;
                            h_miss[ni] += h;
                            g2_miss[ni] += g * g;
                            n_miss[ni] += 1.0;
                        } else if bin < feat_n_bins {
                            flat_g[base + bin] += g;
                            flat_h[base + bin] += h;
                            flat_g2[base + bin] += g * g;
                            flat_n[base + bin] += 1.0;
                        }
                    }
                }

                let mut best_gain = 0.0f64;
                let mut best_bin_val = 0usize;
                let mut best_ml = true;
                let mut best_cat = false;
                let mut best_mask: CatBitmask = Vec::new();

                if binned.is_categorical[feat] {
                    let mut global_g = vec![0.0f64; feat_n_bins];
                    let mut global_h = vec![0.0f64; feat_n_bins];
                    for ni in 0..n_nodes {
                        let base = ni * feat_n_bins;
                        for b in 0..feat_n_bins {
                            global_g[b] += flat_g[base + b];
                            global_h[b] += flat_h[base + b];
                        }
                    }
                    let mut cat_bins_local: Vec<(usize, f64, f64)> = Vec::new();
                    for b in 0..feat_n_bins {
                        if global_h[b] > 0.0 {
                            cat_bins_local.push((b, global_g[b], global_h[b]));
                        }
                    }
                    if cat_bins_local.len() > 1 {
                        let total_g: f64 = cat_bins_local.iter().map(|c| c.1).sum();
                        let total_h: f64 = cat_bins_local.iter().map(|c| c.2).sum();
                        let global_ratio = if total_h > 1e-10 {
                            total_g / total_h
                        } else {
                            0.0
                        };
                        let mut best_ci = 0usize;
                        let mut best_order: Vec<(usize, f64, f64)> = Vec::new();
                        let cat_orders = single_output_cat_orders(
                            &cat_bins_local,
                            global_ratio,
                            lambda_reg,
                            binned.cat_prototype_bins,
                        );
                        for ordered_bins in cat_orders {
                            let n_cats = ordered_bins.len();
                            let mut cum_g = vec![0.0f64; n_nodes * n_cats];
                            let mut cum_h = vec![0.0f64; n_nodes * n_cats];
                            let mut cum_g2 = vec![0.0f64; n_nodes * n_cats];
                            let mut cum_n = vec![0.0f64; n_nodes * n_cats];
                            for ni in 0..n_nodes {
                                let base = ni * feat_n_bins;
                                let cbase = ni * n_cats;
                                let mut sg = 0.0f64;
                                let mut sh = 0.0f64;
                                let mut sg2 = 0.0f64;
                                let mut sn = 0.0f64;
                                for (ci, &(bin, _, _)) in ordered_bins.iter().enumerate() {
                                    sg += flat_g[base + bin];
                                    sh += flat_h[base + bin];
                                    sg2 += flat_g2[base + bin];
                                    sn += flat_n[base + bin];
                                    cum_g[cbase + ci] = sg;
                                    cum_h[cbase + ci] = sh;
                                    cum_g2[cbase + ci] = sg2;
                                    cum_n[cbase + ci] = sn;
                                }
                            }

                            let cat_cutpoints =
                                cat_scan_cutpoints(&ordered_bins, binned.cat_prototype_bins);
                            let cat_search_width =
                                cat_cutpoints.iter().filter(|&&v| v).count().max(1);
                            for ci in 0..n_cats - 1 {
                                if !cat_cutpoints.get(ci).copied().unwrap_or(true) {
                                    continue;
                                }
                                for miss_left in [true, false] {
                                    let mut total_gain = 0.0f64;
                                    for (ni, &(g_total, h_total)) in node_gh.iter().enumerate() {
                                        let (start, end) = node_ranges[ni];
                                        if end - start <= 1 {
                                            continue;
                                        }
                                        let g_nm = g_total - g_miss[ni];
                                        let h_nm = h_total - h_miss[ni];
                                        let cbase = ni * n_cats;
                                        let cg = cum_g[cbase + ci];
                                        let ch = cum_h[cbase + ci];
                                        let cg2 = cum_g2[cbase + ci];
                                        let cn = cum_n[cbase + ci];
                                        let (parent_g2, parent_n) = node_g2n[ni];
                                        let (lg, lh, lg2, ln, rg, rh, rg2, rn) = if miss_left {
                                            (
                                                cg + g_miss[ni],
                                                ch + h_miss[ni],
                                                cg2 + g2_miss[ni],
                                                cn + n_miss[ni],
                                                g_nm - cg,
                                                h_nm - ch,
                                                parent_g2 - cg2 - g2_miss[ni],
                                                parent_n - cn - n_miss[ni],
                                            )
                                        } else {
                                            (
                                                cg,
                                                ch,
                                                cg2,
                                                cn,
                                                g_nm - cg + g_miss[ni],
                                                h_nm - ch + h_miss[ni],
                                                parent_g2 - cg2,
                                                parent_n - cn,
                                            )
                                        };
                                        if lh < min_h || rh < min_h {
                                            continue;
                                        }
                                        let mut gain = 0.5
                                            * (l1_gain_score(lg, lh, lambda_reg, l1_reg)
                                                + l1_gain_score(rg, rh, lambda_reg, l1_reg)
                                                - l1_gain_score(
                                                    g_total, h_total, lambda_reg, l1_reg,
                                                ))
                                            - gamma;
                                        if gain_penalty > 0.0 {
                                            gain -= gain_penalty
                                                * 0.5
                                                * (1.0 / (lh + lambda_reg)
                                                    + 1.0 / (rh + lambda_reg)
                                                    - 1.0 / (h_total + lambda_reg));
                                        }
                                        gain = evidence_adjusted_gain(
                                            binned,
                                            gain,
                                            lh,
                                            rh,
                                            h_total,
                                            lambda_reg,
                                            cat_search_width,
                                        );
                                        gain = categorical_audit_adjusted_gain(
                                            binned,
                                            gain,
                                            lh,
                                            rh,
                                            h_total,
                                            lambda_reg,
                                            cat_search_width,
                                        );
                                        gain = contrast_adjusted_gain(
                                            binned,
                                            feat,
                                            gain,
                                            lg,
                                            lh,
                                            rg,
                                            rh,
                                            lambda_reg,
                                            l1_reg,
                                            cat_search_width,
                                        );
                                        if gain > 0.0 && leaf_var_shrink > 0.0 {
                                            if let Some(ratio) = post_shrink_gain_ratio_from_stats(
                                                g_total,
                                                h_total,
                                                parent_g2,
                                                parent_n,
                                                lg,
                                                lh,
                                                lg2,
                                                ln,
                                                rg,
                                                rh,
                                                rg2,
                                                rn,
                                                lambda_reg,
                                                l1_reg,
                                                min_h,
                                                leaf_var_shrink,
                                            ) {
                                                gain *= ratio;
                                            }
                                        }
                                        total_gain += gain;
                                    }
                                    if total_gain > best_gain {
                                        best_gain = total_gain;
                                        best_ml = miss_left;
                                        best_cat = true;
                                        best_ci = ci;
                                        best_order.clear();
                                        best_order.extend_from_slice(&ordered_bins);
                                    }
                                }
                            }
                        }
                        if best_cat {
                            best_mask = Vec::new();
                            for j in 0..=best_ci {
                                bitmask_set(&mut best_mask, best_order[j].0);
                            }
                        }
                    }
                } else {
                    // Numeric: prefix sum and scan
                    let mut cum_g = vec![0.0f64; n_nodes * feat_n_bins];
                    let mut cum_h = vec![0.0f64; n_nodes * feat_n_bins];
                    let mut cum_g2 = vec![0.0f64; n_nodes * feat_n_bins];
                    let mut cum_n = vec![0.0f64; n_nodes * feat_n_bins];
                    for ni in 0..n_nodes {
                        let base = ni * feat_n_bins;
                        cum_g[base] = flat_g[base];
                        cum_h[base] = flat_h[base];
                        cum_g2[base] = flat_g2[base];
                        cum_n[base] = flat_n[base];
                        for b in 1..feat_n_bins {
                            cum_g[base + b] = cum_g[base + b - 1] + flat_g[base + b];
                            cum_h[base + b] = cum_h[base + b - 1] + flat_h[base + b];
                            cum_g2[base + b] = cum_g2[base + b - 1] + flat_g2[base + b];
                            cum_n[base + b] = cum_n[base + b - 1] + flat_n[base + b];
                        }
                    }

                    // Extra Trees: pick ONE random bin; Standard: scan all bins
                    let bins_to_try: Vec<usize> = if extra_trees {
                        let global_h: f64 = (0..feat_n_bins)
                            .map(|b| {
                                (0..n_nodes)
                                    .map(|ni| flat_h[ni * feat_n_bins + b])
                                    .sum::<f64>()
                            })
                            .sum();
                        if global_h <= 0.0 {
                            Vec::new()
                        } else {
                            let h = tree_seed
                                .wrapping_mul(0x517CC1B727220A95)
                                .wrapping_add(feat as u64)
                                .wrapping_add(_depth as u64);
                            let h2 = h.wrapping_mul(0x9E3779B97F4A7C15);
                            vec![(h2 >> 33) as usize % (feat_n_bins - 1)]
                        }
                    } else {
                        (0..feat_n_bins - 1).collect()
                    };

                    for bin in bins_to_try {
                        for miss_left in [true, false] {
                            let mut total_gain = 0.0f64;
                            for (ni, &(g_total, h_total)) in node_gh.iter().enumerate() {
                                let (start, end) = node_ranges[ni];
                                if end - start <= 1 {
                                    continue;
                                }
                                let g_nm = g_total - g_miss[ni];
                                let h_nm = h_total - h_miss[ni];
                                let base = ni * feat_n_bins;
                                let cg = cum_g[base + bin];
                                let ch = cum_h[base + bin];
                                let cg2 = cum_g2[base + bin];
                                let cn = cum_n[base + bin];
                                let (parent_g2, parent_n) = node_g2n[ni];
                                let (lg, lh, lg2, ln, rg, rh, rg2, rn) = if miss_left {
                                    (
                                        cg + g_miss[ni],
                                        ch + h_miss[ni],
                                        cg2 + g2_miss[ni],
                                        cn + n_miss[ni],
                                        g_nm - cg,
                                        h_nm - ch,
                                        parent_g2 - cg2 - g2_miss[ni],
                                        parent_n - cn - n_miss[ni],
                                    )
                                } else {
                                    (
                                        cg,
                                        ch,
                                        cg2,
                                        cn,
                                        g_nm - cg + g_miss[ni],
                                        h_nm - ch + h_miss[ni],
                                        parent_g2 - cg2,
                                        parent_n - cn,
                                    )
                                };
                                if lh < min_h || rh < min_h {
                                    continue;
                                }
                                let mut gain = 0.5
                                    * (l1_gain_score(lg, lh, lambda_reg, l1_reg)
                                        + l1_gain_score(rg, rh, lambda_reg, l1_reg)
                                        - l1_gain_score(g_total, h_total, lambda_reg, l1_reg))
                                    - gamma;
                                if gain_penalty > 0.0 {
                                    gain -= gain_penalty
                                        * 0.5
                                        * (1.0 / (lh + lambda_reg) + 1.0 / (rh + lambda_reg)
                                            - 1.0 / (h_total + lambda_reg));
                                }
                                gain = evidence_adjusted_gain(
                                    binned,
                                    gain,
                                    lh,
                                    rh,
                                    h_total,
                                    lambda_reg,
                                    feat_n_bins.saturating_sub(1),
                                );
                                gain = contrast_adjusted_gain(
                                    binned,
                                    feat,
                                    gain,
                                    lg,
                                    lh,
                                    rg,
                                    rh,
                                    lambda_reg,
                                    l1_reg,
                                    feat_n_bins.saturating_sub(1),
                                );
                                if gain > 0.0 && leaf_var_shrink > 0.0 {
                                    if let Some(ratio) = post_shrink_gain_ratio_from_stats(
                                        g_total,
                                        h_total,
                                        parent_g2,
                                        parent_n,
                                        lg,
                                        lh,
                                        lg2,
                                        ln,
                                        rg,
                                        rh,
                                        rg2,
                                        rn,
                                        lambda_reg,
                                        l1_reg,
                                        min_h,
                                        leaf_var_shrink,
                                    ) {
                                        gain *= ratio;
                                    }
                                }
                                total_gain += gain;
                            }
                            if total_gain > best_gain {
                                best_gain = total_gain;
                                best_bin_val = bin;
                                best_ml = miss_left;
                                best_cat = false;
                            }
                        }
                    }
                }

                (best_gain, feat, best_bin_val, best_ml, best_cat, best_mask)
            };

            let run_level_scan = |grads: &[f64]| -> OblivResult {
                if use_par {
                    active_features
                        .par_iter()
                        .map(|&f| eval_feat_obliv(f, grads))
                        .reduce(
                            || empty_result.clone(),
                            |a, b| if b.0 > a.0 { b } else { a },
                        )
                } else {
                    let mut best = empty_result.clone();
                    for &f in &active_features {
                        let r = eval_feat_obliv(f, grads);
                        if r.0 > best.0 {
                            best = r;
                        }
                    }
                    best
                }
            };
            let winner: OblivResult = run_level_scan(gradients);

            let (
                best_total_gain,
                best_feat,
                best_bin,
                best_missing_left,
                best_is_cat,
                best_cat_mask,
            ) = winner;

            if best_total_gain <= 0.0 {
                break; // No good split found, stop growing
            }

            // Free-tree signal gate, oblivious form: the level's best total gain
            // must beat `signal_gate` times the best total gain found on a
            // within-node permutation of the gradients (node sums are invariant,
            // so leaf values and the feasible split set are unchanged — only
            // the within-node ordering signal is destroyed). Depth stops by
            // itself when the level signal is indistinguishable from noise.
            if binned.signal_gate > 0.0 && total_rows >= 16 {
                let mut g_perm = gradients.to_vec();
                let seed = tree_seed
                    ^ (_depth as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ 0xF1EA_5EED_0BAD_C0DE;
                let mut rng = StdRng::seed_from_u64(seed);
                for &(start, end) in &node_ranges {
                    if end - start >= 2 {
                        let mut vals: Vec<f64> = row_buf[start..end]
                            .iter()
                            .map(|&i| gradients[i as usize])
                            .collect();
                        vals.shuffle(&mut rng);
                        for (slot, &i) in row_buf[start..end].iter().enumerate() {
                            g_perm[i as usize] = vals[slot];
                        }
                    }
                }
                let null = run_level_scan(&g_perm);
                if best_total_gain < binned.signal_gate * null.0.max(0.0) {
                    break;
                }
            }

            // Apply the same split to ALL nodes at this level
            let mut new_ranges = Vec::with_capacity(node_ranges.len() * 2);
            let mut new_ids = Vec::with_capacity(node_ids.len() * 2);

            for (i, &nid) in node_ids.iter().enumerate() {
                let (start, end) = node_ranges[i];
                if start == end {
                    // Empty node — create empty children
                    let (lid, rid) = tree.add_split(
                        nid,
                        best_feat as u32,
                        best_bin as u16,
                        0.0,
                        best_missing_left,
                        false,
                        [u32::MAX, u32::MAX],
                        [0.0, 0.0],
                        0.0,
                        best_is_cat,
                        best_cat_mask.clone(),
                    );
                    tree.set_node_stats(lid, 0.0, 0);
                    tree.set_node_stats(rid, 0.0, 0);
                    tree.set_leaf(lid, 0.0);
                    tree.set_leaf(rid, 0.0);
                    new_ranges.push((start, start));
                    new_ranges.push((start, start));
                    new_ids.push(lid);
                    new_ids.push(rid);
                    continue;
                }

                let left_end = partition_indices(
                    &mut row_buf,
                    start,
                    end,
                    binned,
                    best_feat,
                    best_bin as u16,
                    best_missing_left,
                    best_is_cat,
                    &best_cat_mask,
                );

                let (g_node, h_node) = node_gh[i];
                let nc_node = (end - start) as f64;
                let (lid, rid) = tree.add_split(
                    nid,
                    best_feat as u32,
                    best_bin as u16,
                    l1_leaf_value(
                        g_node,
                        h_node,
                        lambda_reg + lambda_reg / nc_node.max(1.0).sqrt(),
                        l1_reg,
                    ),
                    best_missing_left,
                    false,
                    [u32::MAX, u32::MAX],
                    [0.0, 0.0],
                    0.0,
                    best_is_cat,
                    best_cat_mask.clone(),
                );

                let (lg, lh) = sum_gh(gradients, hessians, &row_buf[start..left_end]);
                let (rg, rh) = sum_gh(gradients, hessians, &row_buf[left_end..end]);
                let nc_l = (left_end - start) as f64;
                let nc_r = (end - left_end) as f64;
                tree.set_node_stats(lid, lh, (left_end - start) as u32);
                tree.set_node_stats(rid, rh, (end - left_end) as u32);
                tree.set_leaf(
                    lid,
                    l1_leaf_value(
                        lg,
                        lh,
                        lambda_reg + lambda_reg / nc_l.max(1.0).sqrt(),
                        l1_reg,
                    ),
                );
                tree.set_leaf(
                    rid,
                    l1_leaf_value(
                        rg,
                        rh,
                        lambda_reg + lambda_reg / nc_r.max(1.0).sqrt(),
                        l1_reg,
                    ),
                );

                new_ranges.push((start, left_end));
                new_ranges.push((left_end, end));
                new_ids.push(lid);
                new_ids.push(rid);
            }

            node_ranges = new_ranges;
            node_ids = new_ids;
        }

        tree.into_tree()
    }
}
