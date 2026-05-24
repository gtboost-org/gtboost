//! GGFP v6 (LTSO Phases 1-3) — VirtualFeatureRegistry + pre-mine operators.
//!
//! Virtual features are engineered columns that GGFP-v6 (LTSO) emits during
//! fit and registers into `BinnedData` as if they were raw features. Once
//! registered, the rest of the booster (tree builder, predict path, refit,
//! multiclass) treats them as ordinary numeric features — no walker changes.
//!
//! The single tricky path is `predict_raw_row*`: it indexes `raw_row[feat]`,
//! but raw_row only has the original columns. For virtual features, the
//! stored `VirtualFeatureDef` knows how to recompute the value from raw_row.
//!
//! Phase 1 = infrastructure only. Phase 2 (operator catalog) ships next.

use super::BinnedData;
use serde::{Deserialize, Serialize};

/// A virtual feature's operator definition. Stored on `BinnedData.virtual_defs`,
/// keyed by feature_id = n_raw_features + virtual_idx.
///
/// Each variant carries the raw feature ids it depends on so `eval_raw_row` can
/// recompute the value at test time from a raw input row.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum VirtualFeatureDef {
    /// Copy of a raw feature. Used for testing the registry round-trip.
    Identity(u32),
    /// Safe ratio: `raw_row[i] / (|raw_row[j]| + eps_floor)`.
    /// eps_floor = max(1e-6, 1e-3 * stored_scale).
    Ratio { i: u32, j: u32, eps_floor: f64 },
    /// Difference: `raw_row[i] - raw_row[j]`.
    Diff { i: u32, j: u32 },
    /// Product: `raw_row[i] * raw_row[j]`.
    Product { i: u32, j: u32 },
    /// Positive hinge: `max(0, raw_row[i] - threshold)`.
    HingePos { i: u32, threshold: f64 },
    /// Negative hinge: `max(0, threshold - raw_row[i])`.
    HingeNeg { i: u32, threshold: f64 },
    /// Gated numeric value: `raw_row[value_j]` if `raw_row[gate_i] > threshold`, else 0.
    GatedAbove {
        gate_i: u32,
        value_j: u32,
        threshold: f64,
    },
    /// Gated numeric value: `raw_row[value_j]` if `raw_row[gate_i] <= threshold`, else 0.
    GatedBelow {
        gate_i: u32,
        value_j: u32,
        threshold: f64,
    },
    /// Empirical-Bayes group mean of numeric feature `num_i` conditioned on
    /// categorical feature `cat_j`.
    CatMeanBy {
        num_i: u32,
        cat_j: u32,
        levels: Vec<f64>,
        values: Vec<f64>,
        default: f64,
    },
    /// Deviation from the EB group mean: `raw_row[num_i] - mean(num_i | cat_j)`.
    CatDevBy {
        num_i: u32,
        cat_j: u32,
        levels: Vec<f64>,
        means: Vec<f64>,
        default_mean: f64,
    },
    /// EB group standard deviation of numeric feature `num_i` conditioned on
    /// categorical feature `cat_j`.
    CatStdBy {
        num_i: u32,
        cat_j: u32,
        levels: Vec<f64>,
        values: Vec<f64>,
        default: f64,
    },
    /// Shifted log: `log(raw_row[i] - shift + 1)` where shift = min - 1 at fit time.
    Log1p { i: u32, shift: f64 },
    /// `sqrt(|raw_row[i]|)`.
    SqrtAbs { i: u32 },
    /// `raw_row[i] * raw_row[i]`.
    Square { i: u32 },
}

impl VirtualFeatureDef {
    /// Compute the virtual feature value from a raw input row.
    /// Returns NaN if any input is missing — caller's missing-route logic
    /// handles it identically to a missing raw feature.
    #[inline]
    pub fn eval_raw_row(&self, raw_row: &[f64]) -> f64 {
        match *self {
            VirtualFeatureDef::Identity(i) => raw_row[i as usize],
            VirtualFeatureDef::Ratio { i, j, eps_floor } => {
                let a = raw_row[i as usize];
                let b = raw_row[j as usize];
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a / (b.abs() + eps_floor)
                }
            }
            VirtualFeatureDef::Diff { i, j } => {
                let a = raw_row[i as usize];
                let b = raw_row[j as usize];
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a - b
                }
            }
            VirtualFeatureDef::Product { i, j } => {
                let a = raw_row[i as usize];
                let b = raw_row[j as usize];
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a * b
                }
            }
            VirtualFeatureDef::HingePos { i, threshold } => {
                let v = raw_row[i as usize];
                if v.is_nan() {
                    f64::NAN
                } else {
                    (v - threshold).max(0.0)
                }
            }
            VirtualFeatureDef::HingeNeg { i, threshold } => {
                let v = raw_row[i as usize];
                if v.is_nan() {
                    f64::NAN
                } else {
                    (threshold - v).max(0.0)
                }
            }
            VirtualFeatureDef::GatedAbove {
                gate_i,
                value_j,
                threshold,
            } => {
                let gate = raw_row[gate_i as usize];
                let value = raw_row[value_j as usize];
                if gate.is_nan() || value.is_nan() {
                    f64::NAN
                } else if gate > threshold {
                    value
                } else {
                    0.0
                }
            }
            VirtualFeatureDef::GatedBelow {
                gate_i,
                value_j,
                threshold,
            } => {
                let gate = raw_row[gate_i as usize];
                let value = raw_row[value_j as usize];
                if gate.is_nan() || value.is_nan() {
                    f64::NAN
                } else if gate <= threshold {
                    value
                } else {
                    0.0
                }
            }
            VirtualFeatureDef::CatMeanBy {
                cat_j,
                ref levels,
                ref values,
                default,
                ..
            } => {
                let cat = raw_row[cat_j as usize];
                if cat.is_nan() {
                    f64::NAN
                } else {
                    lookup_level(levels, values, cat, default)
                }
            }
            VirtualFeatureDef::CatDevBy {
                num_i,
                cat_j,
                ref levels,
                ref means,
                default_mean,
            } => {
                let num = raw_row[num_i as usize];
                let cat = raw_row[cat_j as usize];
                if num.is_nan() || cat.is_nan() {
                    f64::NAN
                } else {
                    num - lookup_level(levels, means, cat, default_mean)
                }
            }
            VirtualFeatureDef::CatStdBy {
                cat_j,
                ref levels,
                ref values,
                default,
                ..
            } => {
                let cat = raw_row[cat_j as usize];
                if cat.is_nan() {
                    f64::NAN
                } else {
                    lookup_level(levels, values, cat, default)
                }
            }
            VirtualFeatureDef::Log1p { i, shift } => {
                let v = raw_row[i as usize];
                if v.is_nan() {
                    f64::NAN
                } else {
                    (v - shift + 1.0).max(1e-9).ln()
                }
            }
            VirtualFeatureDef::SqrtAbs { i } => {
                let v = raw_row[i as usize];
                if v.is_nan() {
                    f64::NAN
                } else {
                    v.abs().sqrt()
                }
            }
            VirtualFeatureDef::Square { i } => {
                let v = raw_row[i as usize];
                if v.is_nan() {
                    f64::NAN
                } else {
                    v * v
                }
            }
        }
    }
}

// ── Phase 3 — pre-mine operator catalog at fit start ───────────────────────

/// Config for Phase 3 pre-mining.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct LtsoPremineConfig {
    pub enabled: bool,
    /// Top-K numeric features by variance to consider for pair operators.
    pub top_var_k: usize,
    /// Max accepted operators per fit.
    pub max_accept: usize,
    /// Perm-Bonferroni alpha (false-positive rate).
    pub alpha: f64,
    /// Quantile bin count for the registered virtual.
    pub n_bins: usize,
    /// Eps floor for ratio's divisor.
    pub ratio_eps: f64,
    /// Empirical-Bayes shrinkage strength for N|C tools.
    pub eb_tau: f64,
    /// Minimum validation/train correlation retention when eval data exists.
    pub min_eval_fraction: f64,
    /// Minimum eval residual-SSE reduction for honest stump admission.
    pub min_eval_gain_fraction: f64,
}

impl Default for LtsoPremineConfig {
    fn default() -> Self {
        // v6.1: keep alpha=0.05 (extreme α=0.01 starved the gate); modest
        // max_accept=5 reduction from v1's 8. Product is still dropped
        // (duplicates auto_interactions). Unary monotone transforms are also
        // dropped after v6.0-v2.1 showed they crowd out useful pair operators.
        Self {
            enabled: false,
            top_var_k: 8,
            max_accept: 1,
            alpha: 0.05,
            n_bins: 32,
            ratio_eps: 1e-3,
            eb_tau: 10.0,
            min_eval_fraction: 0.15,
            min_eval_gain_fraction: 0.001,
        }
    }
}

/// Quantile-bin a numeric column. Returns (bins, edges).
/// `n_bins` cuts → `n_bins+1` regions, but we cap bin index at `n_bins`.
pub fn quantile_bin(values: &[f64], n_bins: usize) -> (Vec<u16>, Vec<f64>) {
    let n = values.len();
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = sorted.len();
    if m < 2 {
        return (vec![0u16; n], vec![0.0]);
    }
    let mut edges: Vec<f64> = Vec::with_capacity(n_bins);
    for i in 1..n_bins {
        let frac = i as f64 / n_bins as f64;
        let idx = ((frac * (m - 1) as f64) as usize).min(m - 1);
        edges.push(sorted[idx]);
    }
    // Dedup successive equal edges
    edges.dedup_by(|a, b| *a == *b);
    if edges.is_empty() {
        edges.push(sorted[m / 2]);
    }
    let mut bins = vec![0u16; n];
    let max_bin = edges.len().saturating_sub(1);
    for (i, &v) in values.iter().enumerate() {
        if !v.is_finite() {
            bins[i] = super::MISSING_BIN;
        } else {
            // binary search for edge >= v
            let mut lo = 0usize;
            let mut hi = edges.len();
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if edges[mid] < v {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            // Cap at edges.len() - 1 to match histogram allocation [0, n_bins).
            bins[i] = lo.min(max_bin) as u16;
        }
    }
    (bins, edges)
}

/// Score correlation between feature values and residual.
#[inline]
fn corr_score(values: &[f64], residual: &[f64]) -> f64 {
    let n = values.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    let mut count = 0.0;
    for (v, r) in values.iter().zip(residual.iter()) {
        if v.is_finite() && r.is_finite() {
            sx += v;
            sy += r;
            sxx += v * v;
            syy += r * r;
            sxy += v * r;
            count += 1.0;
        }
    }
    if count < 2.0 {
        return 0.0;
    }
    let mx = sx / count;
    let my = sy / count;
    let var_x = (sxx / count - mx * mx).max(1e-12);
    let var_y = (syy / count - my * my).max(1e-12);
    let cov = sxy / count - mx * my;
    let denom = (var_x * var_y).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        (cov / denom).abs()
    }
}

#[inline]
fn signed_corr_score(values: &[f64], residual: &[f64]) -> f64 {
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    let mut count = 0.0;
    for (v, r) in values.iter().zip(residual.iter()) {
        if v.is_finite() && r.is_finite() {
            sx += v;
            sy += r;
            sxx += v * v;
            syy += r * r;
            sxy += v * r;
            count += 1.0;
        }
    }
    if count < 2.0 {
        return 0.0;
    }
    let mx = sx / count;
    let my = sy / count;
    let var_x = (sxx / count - mx * mx).max(1e-12);
    let var_y = (syy / count - my * my).max(1e-12);
    let cov = sxy / count - mx * my;
    let denom = (var_x * var_y).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        cov / denom
    }
}

/// Pre-mine candidate operators against round-0 residual.
/// Returns list of (op_def, train_values, score) sorted by score desc.
///
/// `raw_values_by_col[f]` must be the original (unbinned) f64 values for
/// feature `f` on training rows. `residual` is per-row gradient-aligned target.
pub fn premine_candidates(
    raw_values_by_col: &[Vec<f64>],
    num_indices: &[usize],
    cat_indices: &[usize],
    residual: &[f64],
    cfg: &LtsoPremineConfig,
    eval_values_by_col: Option<&[Vec<f64>]>,
    eval_residual: Option<&[f64]>,
) -> Vec<(VirtualFeatureDef, Vec<f64>, f64)> {
    if !cfg.enabled || (num_indices.is_empty() && cat_indices.is_empty()) {
        return Vec::new();
    }
    let has_honest_eval = eval_values_by_col.is_some() && eval_residual.is_some();
    let allow_no_eval = std::env::var("GTBOOST_LTSO_ALLOW_NO_EVAL")
        .map(|v| v == "1")
        .unwrap_or(false);
    if !has_honest_eval && !allow_no_eval {
        return Vec::new();
    }
    let debug = std::env::var("GTBOOST_LTSO_DEBUG")
        .map(|v| v == "1")
        .unwrap_or(false);

    // Pick top-K numerics by variance.
    let mut var_of: Vec<(usize, f64)> = num_indices
        .iter()
        .map(|&i| {
            let v = &raw_values_by_col[i];
            let n = v.iter().filter(|x| x.is_finite()).count() as f64;
            if n < 2.0 {
                (i, 0.0)
            } else {
                let mu: f64 = v.iter().filter(|x| x.is_finite()).copied().sum::<f64>() / n;
                let var: f64 = v
                    .iter()
                    .filter(|x| x.is_finite())
                    .map(|x| (x - mu).powi(2))
                    .sum::<f64>()
                    / n;
                (i, var)
            }
        })
        .collect();
    var_of.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top: Vec<usize> = var_of.iter().take(cfg.top_var_k).map(|&(i, _)| i).collect();

    // Build candidate pool:
    // - ratio + diff per pair (no Product: duplicates auto_interactions)
    // - gradient-stump hinges on each top numeric
    // - gated numeric values from stump thresholds, e.g. x_j * 1[x_i > t]
    //
    // Explicitly do NOT add Log/Sqrt/Square unary monotone transforms. Trees
    // are invariant to monotone transforms, and v6.0-v2.1 showed they crowd out
    // genuinely useful pair operators on diabetes.
    let mut candidates: Vec<(VirtualFeatureDef, Vec<f64>, Option<Vec<f64>>)> = Vec::new();
    let mut stump_thresholds: Vec<(usize, f64)> = Vec::new();
    for (idx_i, &i) in top.iter().enumerate() {
        let xi = &raw_values_by_col[i];
        if let Some(threshold) = best_gradient_stump_threshold(xi, residual, 16) {
            stump_thresholds.push((i, threshold));
            let hinge_pos: Vec<f64> = xi
                .iter()
                .map(|&a| {
                    if !a.is_finite() {
                        f64::NAN
                    } else {
                        (a - threshold).max(0.0)
                    }
                })
                .collect();
            let op = VirtualFeatureDef::HingePos {
                i: i as u32,
                threshold,
            };
            let train_vals = clip_quantile(&hinge_pos, 0.001);
            let eval_vals = eval_values_by_col
                .map(|cols| clip_quantile(&materialize_virtual_feature(&op, cols), 0.001));
            candidates.push((op, train_vals, eval_vals));
            let hinge_neg: Vec<f64> = xi
                .iter()
                .map(|&a| {
                    if !a.is_finite() {
                        f64::NAN
                    } else {
                        (threshold - a).max(0.0)
                    }
                })
                .collect();
            let op = VirtualFeatureDef::HingeNeg {
                i: i as u32,
                threshold,
            };
            let train_vals = clip_quantile(&hinge_neg, 0.001);
            let eval_vals = eval_values_by_col
                .map(|cols| clip_quantile(&materialize_virtual_feature(&op, cols), 0.001));
            candidates.push((op, train_vals, eval_vals));
        }
        // ── Pair ops on (i, j) — ratio + diff (no product) ──────────────
        for &j in top.iter().skip(idx_i + 1) {
            let xj = &raw_values_by_col[j];
            let std_j: f64 = {
                let n = xj.iter().filter(|x| x.is_finite()).count() as f64;
                if n < 2.0 {
                    1.0
                } else {
                    let mu = xj.iter().filter(|x| x.is_finite()).copied().sum::<f64>() / n;
                    (xj.iter()
                        .filter(|x| x.is_finite())
                        .map(|x| (x - mu).powi(2))
                        .sum::<f64>()
                        / n)
                        .sqrt()
                }
            };
            let eps_floor_ij = (cfg.ratio_eps * std_j).max(1e-6);
            let ratio_vals: Vec<f64> = xi
                .iter()
                .zip(xj.iter())
                .map(|(&a, &b)| {
                    if !a.is_finite() || !b.is_finite() {
                        f64::NAN
                    } else {
                        a / (b.abs() + eps_floor_ij)
                    }
                })
                .collect();
            let ratio_vals = clip_quantile(&ratio_vals, 0.001);
            let op = VirtualFeatureDef::Ratio {
                i: i as u32,
                j: j as u32,
                eps_floor: eps_floor_ij,
            };
            let eval_vals = eval_values_by_col
                .map(|cols| clip_quantile(&materialize_virtual_feature(&op, cols), 0.001));
            candidates.push((op, ratio_vals, eval_vals));
            let diff_vals: Vec<f64> = xi
                .iter()
                .zip(xj.iter())
                .map(|(&a, &b)| {
                    if !a.is_finite() || !b.is_finite() {
                        f64::NAN
                    } else {
                        a - b
                    }
                })
                .collect();
            let op = VirtualFeatureDef::Diff {
                i: i as u32,
                j: j as u32,
            };
            let eval_vals = eval_values_by_col.map(|cols| materialize_virtual_feature(&op, cols));
            candidates.push((op, diff_vals, eval_vals));
        }
    }
    for &(gate_i, threshold) in &stump_thresholds {
        let gate = &raw_values_by_col[gate_i];
        for &value_j in &top {
            if value_j == gate_i {
                continue;
            }
            let xj = &raw_values_by_col[value_j];
            let above_vals: Vec<f64> = gate
                .iter()
                .zip(xj.iter())
                .map(|(&g, &x)| {
                    if !g.is_finite() || !x.is_finite() {
                        f64::NAN
                    } else if g > threshold {
                        x
                    } else {
                        0.0
                    }
                })
                .collect();
            let op = VirtualFeatureDef::GatedAbove {
                gate_i: gate_i as u32,
                value_j: value_j as u32,
                threshold,
            };
            let train_vals = clip_quantile(&above_vals, 0.001);
            let eval_vals = eval_values_by_col
                .map(|cols| clip_quantile(&materialize_virtual_feature(&op, cols), 0.001));
            candidates.push((op, train_vals, eval_vals));
            let below_vals: Vec<f64> = gate
                .iter()
                .zip(xj.iter())
                .map(|(&g, &x)| {
                    if !g.is_finite() || !x.is_finite() {
                        f64::NAN
                    } else if g <= threshold {
                        x
                    } else {
                        0.0
                    }
                })
                .collect();
            let op = VirtualFeatureDef::GatedBelow {
                gate_i: gate_i as u32,
                value_j: value_j as u32,
                threshold,
            };
            let train_vals = clip_quantile(&below_vals, 0.001);
            let eval_vals = eval_values_by_col
                .map(|cols| clip_quantile(&materialize_virtual_feature(&op, cols), 0.001));
            candidates.push((op, train_vals, eval_vals));
        }
    }
    // N|C tools. These are the Rust version of the GGFP-v3 feature family
    // that actually transferred on mixed categorical/numeric tables: EB group
    // mean, deviation, and std. The op stores train-only maps, then eval/test
    // rows are mapped through the same maps with a global default for unseen
    // categories.
    if !top.is_empty() && !cat_indices.is_empty() {
        for &cat_j in cat_indices {
            if distinct_count(&raw_values_by_col[cat_j], 128) > 128 {
                continue;
            }
            for &num_i in &top {
                if let Some((levels, means, stds, default_mean, default_std)) = eb_group_stats(
                    &raw_values_by_col[cat_j],
                    &raw_values_by_col[num_i],
                    cfg.eb_tau,
                ) {
                    let mean_op = VirtualFeatureDef::CatMeanBy {
                        num_i: num_i as u32,
                        cat_j: cat_j as u32,
                        levels: levels.clone(),
                        values: means.clone(),
                        default: default_mean,
                    };
                    let mean_vals = materialize_virtual_feature(&mean_op, raw_values_by_col);
                    let mean_eval =
                        eval_values_by_col.map(|cols| materialize_virtual_feature(&mean_op, cols));
                    candidates.push((mean_op, mean_vals, mean_eval));

                    let dev_op = VirtualFeatureDef::CatDevBy {
                        num_i: num_i as u32,
                        cat_j: cat_j as u32,
                        levels: levels.clone(),
                        means: means.clone(),
                        default_mean,
                    };
                    let dev_vals = clip_quantile(
                        &materialize_virtual_feature(&dev_op, raw_values_by_col),
                        0.001,
                    );
                    let dev_eval = eval_values_by_col.map(|cols| {
                        clip_quantile(&materialize_virtual_feature(&dev_op, cols), 0.001)
                    });
                    candidates.push((dev_op, dev_vals, dev_eval));

                    let std_op = VirtualFeatureDef::CatStdBy {
                        num_i: num_i as u32,
                        cat_j: cat_j as u32,
                        levels,
                        values: stds,
                        default: default_std,
                    };
                    let std_vals = materialize_virtual_feature(&std_op, raw_values_by_col);
                    let std_eval =
                        eval_values_by_col.map(|cols| materialize_virtual_feature(&std_op, cols));
                    candidates.push((std_op, std_vals, std_eval));
                }
            }
        }
    }
    if candidates.is_empty() {
        return Vec::new();
    }
    let total_candidates = candidates.len();

    // Perm-Bonferroni threshold: one permutation, take (1 - alpha/|pool|) quantile.
    let mut perm = residual.to_vec();
    let mut rng_state: u64 = 0x9E3779B97F4A7C15;
    fisher_yates(&mut perm, &mut rng_state);
    let mut null_scores: Vec<f64> = candidates
        .iter()
        .map(|(_, v, _)| corr_score(v, &perm))
        .collect();
    null_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = (1.0 - cfg.alpha / null_scores.len().max(1) as f64).clamp(0.0, 1.0);
    let q_idx = ((null_scores.len() - 1) as f64 * q) as usize;
    let threshold = null_scores[q_idx];

    // OMP-style selection: after accepting a feature, project it out of the
    // residual before scoring the next one. This is the missing guard that v6
    // lacked; it prevents redundant transforms of the same direction from
    // filling all accept slots.
    let mut residual_work = residual.to_vec();
    center_in_place(&mut residual_work);
    let mut accepted: Vec<(VirtualFeatureDef, Vec<f64>, f64)> = Vec::new();
    let mut active = vec![true; candidates.len()];
    let mut honest_rejected = 0usize;
    while accepted.len() < cfg.max_accept {
        let mut best_idx = usize::MAX;
        let mut best_score = if has_honest_eval {
            cfg.min_eval_gain_fraction
        } else {
            threshold
        };
        let mut best_train_score = 0.0;
        let mut best_eval_score = 0.0;
        for (idx, (_, values, eval_values)) in candidates.iter().enumerate() {
            if !active[idx] {
                continue;
            }
            let score = if let (Some(ev), Some(er)) = (eval_values.as_ref(), eval_residual) {
                let train_signed = signed_corr_score(values, residual);
                let eval_signed = signed_corr_score(ev, er);
                let train_abs = train_signed.abs();
                let eval_abs = eval_signed.abs();
                let min_eval_corr = (train_abs * cfg.min_eval_fraction).max(0.01);
                let honest = honest_stump_transfer(values, residual, ev, er);
                if train_signed * eval_signed <= 0.0
                    || eval_abs < min_eval_corr
                    || honest
                        .map(|(_, eg)| eg < cfg.min_eval_gain_fraction)
                        .unwrap_or(true)
                {
                    active[idx] = false;
                    honest_rejected += 1;
                    continue;
                }
                let (train_gain, eval_gain) = honest.unwrap();
                // Eval gain is the admission score. Train gain is retained for
                // diagnostics so we can see train/eval drift.
                if eval_gain > best_score {
                    best_train_score = train_gain;
                    best_eval_score = eval_gain;
                }
                eval_gain
            } else {
                corr_score(values, &residual_work)
            };
            if accepted
                .iter()
                .any(|(_, picked_values, _)| corr_signed(values, picked_values).abs() > 0.985)
            {
                active[idx] = false;
                continue;
            }
            if score > best_score {
                best_score = score;
                best_idx = idx;
                if !has_honest_eval {
                    best_train_score = score;
                    best_eval_score = 0.0;
                }
            }
        }
        if best_idx == usize::MAX {
            break;
        }
        active[best_idx] = false;
        let (op, values, _) = candidates[best_idx].clone();
        if debug {
            eprintln!(
                "[LTSO:accept] op={} score={:.6} train_gain={:.6} eval_gain={:.6}",
                op_label(&op),
                best_score,
                best_train_score,
                best_eval_score,
            );
        }
        project_out(&mut residual_work, &values);
        accepted.push((op, values, best_score));
    }
    if debug {
        eprintln!(
            "[LTSO:summary] candidates={} accepted={} honest_eval={} honest_rejected={} threshold={:.6}",
            total_candidates,
            accepted.len(),
            has_honest_eval,
            honest_rejected,
            threshold,
        );
    }
    accepted
}

fn op_label(op: &VirtualFeatureDef) -> &'static str {
    match op {
        VirtualFeatureDef::Identity(_) => "identity",
        VirtualFeatureDef::Ratio { .. } => "ratio",
        VirtualFeatureDef::Diff { .. } => "diff",
        VirtualFeatureDef::Product { .. } => "product",
        VirtualFeatureDef::HingePos { .. } => "hinge_pos",
        VirtualFeatureDef::HingeNeg { .. } => "hinge_neg",
        VirtualFeatureDef::GatedAbove { .. } => "gated_above",
        VirtualFeatureDef::GatedBelow { .. } => "gated_below",
        VirtualFeatureDef::CatMeanBy { .. } => "cat_mean_by",
        VirtualFeatureDef::CatDevBy { .. } => "cat_dev_by",
        VirtualFeatureDef::CatStdBy { .. } => "cat_std_by",
        VirtualFeatureDef::Log1p { .. } => "log1p",
        VirtualFeatureDef::SqrtAbs { .. } => "sqrt_abs",
        VirtualFeatureDef::Square { .. } => "square",
    }
}

fn honest_stump_transfer(
    train_values: &[f64],
    train_residual: &[f64],
    eval_values: &[f64],
    eval_residual: &[f64],
) -> Option<(f64, f64)> {
    let finite_train = train_values
        .iter()
        .zip(train_residual.iter())
        .filter(|(x, r)| x.is_finite() && r.is_finite())
        .count();
    let min_leaf = (finite_train / 20).clamp(8, 64);
    let mut rows: Vec<(f64, f64)> = train_values
        .iter()
        .zip(train_residual.iter())
        .filter_map(|(&x, &r)| {
            if x.is_finite() && r.is_finite() {
                Some((x, r))
            } else {
                None
            }
        })
        .collect();
    if rows.len() < min_leaf.saturating_mul(2).max(2) {
        return None;
    }
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let n = rows.len();
    let total_r: f64 = rows.iter().map(|(_, r)| *r).sum();
    let total_r2: f64 = rows.iter().map(|(_, r)| r * r).sum();
    let base_train_sse = total_r2 - total_r * total_r / n as f64;
    if base_train_sse <= 1e-12 {
        return None;
    }
    let mut left_sum = 0.0;
    let mut left_sse_sum = 0.0;
    let mut best_gain = 0.0;
    let mut best_t = 0.0;
    let mut best_left_mu = 0.0;
    let mut best_right_mu = 0.0;
    for pos in 0..n - 1 {
        let r = rows[pos].1;
        left_sum += r;
        left_sse_sum += r * r;
        let n_l = pos + 1;
        let n_r = n - n_l;
        if n_l < min_leaf || n_r < min_leaf || rows[pos].0 == rows[pos + 1].0 {
            continue;
        }
        let right_sum = total_r - left_sum;
        let right_sse_sum = total_r2 - left_sse_sum;
        let left_sse = left_sse_sum - left_sum * left_sum / n_l as f64;
        let right_sse = right_sse_sum - right_sum * right_sum / n_r as f64;
        let gain = base_train_sse - left_sse - right_sse;
        if gain > best_gain {
            best_gain = gain;
            best_t = 0.5 * (rows[pos].0 + rows[pos + 1].0);
            best_left_mu = left_sum / n_l as f64;
            best_right_mu = right_sum / n_r as f64;
        }
    }
    if best_gain <= 0.0 {
        return None;
    }
    let train_gain_frac = best_gain / base_train_sse.max(1e-12);

    let mut eval_base = 0.0;
    let mut eval_after = 0.0;
    let mut eval_n = 0usize;
    for (&x, &r) in eval_values.iter().zip(eval_residual.iter()) {
        if !r.is_finite() {
            continue;
        }
        eval_base += r * r;
        let pred = if x.is_finite() {
            if x <= best_t {
                best_left_mu
            } else {
                best_right_mu
            }
        } else {
            0.0
        };
        let e = r - pred;
        eval_after += e * e;
        eval_n += 1;
    }
    if eval_n < 2 || eval_base <= 1e-12 {
        return None;
    }
    let eval_gain_frac = (eval_base - eval_after) / eval_base;
    Some((train_gain_frac, eval_gain_frac))
}

fn lookup_level(levels: &[f64], values: &[f64], key: f64, default: f64) -> f64 {
    match levels
        .binary_search_by(|probe| probe.partial_cmp(&key).unwrap_or(std::cmp::Ordering::Equal))
    {
        Ok(idx) => values.get(idx).copied().unwrap_or(default),
        Err(_) => default,
    }
}

pub fn materialize_virtual_feature(
    op: &VirtualFeatureDef,
    raw_values_by_col: &[Vec<f64>],
) -> Vec<f64> {
    if raw_values_by_col.is_empty() {
        return Vec::new();
    }
    let n = raw_values_by_col[0].len();
    (0..n)
        .map(|row| match op {
            VirtualFeatureDef::Identity(i) => raw_values_by_col[*i as usize][row],
            VirtualFeatureDef::Ratio { i, j, eps_floor } => {
                let a = raw_values_by_col[*i as usize][row];
                let b = raw_values_by_col[*j as usize][row];
                if !a.is_finite() || !b.is_finite() {
                    f64::NAN
                } else {
                    a / (b.abs() + eps_floor)
                }
            }
            VirtualFeatureDef::Diff { i, j } => {
                let a = raw_values_by_col[*i as usize][row];
                let b = raw_values_by_col[*j as usize][row];
                if !a.is_finite() || !b.is_finite() {
                    f64::NAN
                } else {
                    a - b
                }
            }
            VirtualFeatureDef::Product { i, j } => {
                let a = raw_values_by_col[*i as usize][row];
                let b = raw_values_by_col[*j as usize][row];
                if !a.is_finite() || !b.is_finite() {
                    f64::NAN
                } else {
                    a * b
                }
            }
            VirtualFeatureDef::HingePos { i, threshold } => {
                let v = raw_values_by_col[*i as usize][row];
                if !v.is_finite() {
                    f64::NAN
                } else {
                    (v - threshold).max(0.0)
                }
            }
            VirtualFeatureDef::HingeNeg { i, threshold } => {
                let v = raw_values_by_col[*i as usize][row];
                if !v.is_finite() {
                    f64::NAN
                } else {
                    (threshold - v).max(0.0)
                }
            }
            VirtualFeatureDef::GatedAbove {
                gate_i,
                value_j,
                threshold,
            } => {
                let gate = raw_values_by_col[*gate_i as usize][row];
                let value = raw_values_by_col[*value_j as usize][row];
                if !gate.is_finite() || !value.is_finite() {
                    f64::NAN
                } else if gate > *threshold {
                    value
                } else {
                    0.0
                }
            }
            VirtualFeatureDef::GatedBelow {
                gate_i,
                value_j,
                threshold,
            } => {
                let gate = raw_values_by_col[*gate_i as usize][row];
                let value = raw_values_by_col[*value_j as usize][row];
                if !gate.is_finite() || !value.is_finite() {
                    f64::NAN
                } else if gate <= *threshold {
                    value
                } else {
                    0.0
                }
            }
            VirtualFeatureDef::CatMeanBy {
                cat_j,
                levels,
                values,
                default,
                ..
            } => {
                let cat = raw_values_by_col[*cat_j as usize][row];
                if !cat.is_finite() {
                    f64::NAN
                } else {
                    lookup_level(levels, values, cat, *default)
                }
            }
            VirtualFeatureDef::CatDevBy {
                num_i,
                cat_j,
                levels,
                means,
                default_mean,
            } => {
                let num = raw_values_by_col[*num_i as usize][row];
                let cat = raw_values_by_col[*cat_j as usize][row];
                if !num.is_finite() || !cat.is_finite() {
                    f64::NAN
                } else {
                    num - lookup_level(levels, means, cat, *default_mean)
                }
            }
            VirtualFeatureDef::CatStdBy {
                cat_j,
                levels,
                values,
                default,
                ..
            } => {
                let cat = raw_values_by_col[*cat_j as usize][row];
                if !cat.is_finite() {
                    f64::NAN
                } else {
                    lookup_level(levels, values, cat, *default)
                }
            }
            VirtualFeatureDef::Log1p { i, shift } => {
                let v = raw_values_by_col[*i as usize][row];
                if !v.is_finite() {
                    f64::NAN
                } else {
                    (v - shift + 1.0).max(1e-9).ln()
                }
            }
            VirtualFeatureDef::SqrtAbs { i } => {
                let v = raw_values_by_col[*i as usize][row];
                if !v.is_finite() {
                    f64::NAN
                } else {
                    v.abs().sqrt()
                }
            }
            VirtualFeatureDef::Square { i } => {
                let v = raw_values_by_col[*i as usize][row];
                if !v.is_finite() {
                    f64::NAN
                } else {
                    v * v
                }
            }
        })
        .collect()
}

fn distinct_count(values: &[f64], cap: usize) -> usize {
    let mut uniq: Vec<f64> = Vec::new();
    'outer: for &v in values {
        if !v.is_finite() {
            continue;
        }
        for &u in &uniq {
            if u == v {
                continue 'outer;
            }
        }
        uniq.push(v);
        if uniq.len() > cap {
            break;
        }
    }
    uniq.len()
}

fn eb_group_stats(
    cat_values: &[f64],
    num_values: &[f64],
    tau: f64,
) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>, f64, f64)> {
    let mut rows: Vec<(f64, f64)> = cat_values
        .iter()
        .zip(num_values.iter())
        .filter_map(|(&c, &x)| {
            if c.is_finite() && x.is_finite() {
                Some((c, x))
            } else {
                None
            }
        })
        .collect();
    if rows.len() < 4 {
        return None;
    }
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let n = rows.len() as f64;
    let global_mean = rows.iter().map(|(_, x)| *x).sum::<f64>() / n;
    let global_var = rows
        .iter()
        .map(|(_, x)| {
            let d = *x - global_mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let global_std = global_var.max(0.0).sqrt().max(1e-6);

    let mut levels = Vec::new();
    let mut means = Vec::new();
    let mut stds = Vec::new();
    let mut start = 0usize;
    while start < rows.len() {
        let level = rows[start].0;
        let mut end = start + 1;
        while end < rows.len() && rows[end].0 == level {
            end += 1;
        }
        let cnt = (end - start) as f64;
        let sum = rows[start..end].iter().map(|(_, x)| *x).sum::<f64>();
        let mean = sum / cnt;
        let var = rows[start..end]
            .iter()
            .map(|(_, x)| {
                let d = *x - mean;
                d * d
            })
            .sum::<f64>()
            / cnt;
        let std = var.max(0.0).sqrt();
        levels.push(level);
        means.push((cnt * mean + tau * global_mean) / (cnt + tau));
        stds.push((cnt * std + tau * global_std) / (cnt + tau));
        start = end;
    }
    if levels.len() < 2 {
        return None;
    }
    Some((levels, means, stds, global_mean, global_std))
}

fn best_gradient_stump_threshold(values: &[f64], residual: &[f64], min_leaf: usize) -> Option<f64> {
    let mut rows: Vec<(f64, f64)> = values
        .iter()
        .zip(residual.iter())
        .filter_map(|(&x, &r)| {
            if x.is_finite() && r.is_finite() {
                Some((x, r))
            } else {
                None
            }
        })
        .collect();
    if rows.len() < min_leaf.saturating_mul(2).max(2) {
        return None;
    }
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let n = rows.len();
    let total_r: f64 = rows.iter().map(|(_, r)| *r).sum();
    let mut left_r = 0.0;
    let mut best_gain = 0.0;
    let mut best_t = None;
    for pos in 0..n - 1 {
        left_r += rows[pos].1;
        let n_l = pos + 1;
        let n_r = n - n_l;
        if n_l < min_leaf || n_r < min_leaf || rows[pos].0 == rows[pos + 1].0 {
            continue;
        }
        let right_r = total_r - left_r;
        let gain = left_r * left_r / n_l as f64 + right_r * right_r / n_r as f64;
        if gain > best_gain {
            best_gain = gain;
            best_t = Some(0.5 * (rows[pos].0 + rows[pos + 1].0));
        }
    }
    best_t
}

fn center_in_place(values: &mut [f64]) {
    let mut sum = 0.0;
    let mut n = 0.0;
    for &v in values.iter() {
        if v.is_finite() {
            sum += v;
            n += 1.0;
        }
    }
    if n == 0.0 {
        return;
    }
    let mean = sum / n;
    for v in values.iter_mut() {
        if v.is_finite() {
            *v -= mean;
        }
    }
}

fn corr_signed(values_a: &[f64], values_b: &[f64]) -> f64 {
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    let mut count = 0.0;
    for (&x, &y) in values_a.iter().zip(values_b.iter()) {
        if x.is_finite() && y.is_finite() {
            sx += x;
            sy += y;
            sxx += x * x;
            syy += y * y;
            sxy += x * y;
            count += 1.0;
        }
    }
    if count < 2.0 {
        return 0.0;
    }
    let mx = sx / count;
    let my = sy / count;
    let var_x = (sxx / count - mx * mx).max(1e-12);
    let var_y = (syy / count - my * my).max(1e-12);
    let cov = sxy / count - mx * my;
    cov / (var_x * var_y).sqrt()
}

fn project_out(residual: &mut [f64], values: &[f64]) {
    let mut sx = 0.0;
    let mut count = 0.0;
    for &x in values {
        if x.is_finite() {
            sx += x;
            count += 1.0;
        }
    }
    if count < 2.0 {
        return;
    }
    let mx = sx / count;
    let mut denom = 0.0;
    let mut numer = 0.0;
    for (&x, &r) in values.iter().zip(residual.iter()) {
        if x.is_finite() && r.is_finite() {
            let xc = x - mx;
            denom += xc * xc;
            numer += xc * r;
        }
    }
    if denom <= 1e-12 {
        return;
    }
    let beta = numer / denom;
    for (r, &x) in residual.iter_mut().zip(values.iter()) {
        if x.is_finite() && r.is_finite() {
            *r -= beta * (x - mx);
        }
    }
    center_in_place(residual);
}

/// Clip values to robust quantiles, replace nan/inf with 0.
fn clip_quantile(values: &[f64], q: f64) -> Vec<f64> {
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if sorted.len() < 2 {
        return values
            .iter()
            .map(|v| if v.is_finite() { *v } else { 0.0 })
            .collect();
    }
    let lo_idx = ((sorted.len() - 1) as f64 * q) as usize;
    let hi_idx = ((sorted.len() - 1) as f64 * (1.0 - q)) as usize;
    let lo = sorted[lo_idx];
    let hi = sorted[hi_idx];
    values
        .iter()
        .map(|&v| {
            if !v.is_finite() {
                0.0
            } else if v < lo {
                lo
            } else if v > hi {
                hi
            } else {
                v
            }
        })
        .collect()
}

/// Simple Fisher-Yates with xorshift RNG (no external dep).
fn fisher_yates(arr: &mut [f64], state: &mut u64) {
    let n = arr.len();
    for i in (1..n).rev() {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        let j = (*state as usize) % (i + 1);
        arr.swap(i, j);
    }
}

/// Append virtual-feature bin columns to a flat `eval_bin_indices` Vec<u16>
/// using the op defs + edges stored on `train_binned`. Mirrors
/// `BinnedData::bin_with_edges` semantics — caller-side eval pipeline keeps
/// its existing tuple-of-vectors layout.
///
/// Also optionally extends `eval_cll_hash_bins` with zeros to keep it
/// parallel (caller passes `None` if it doesn't track CLL hash bins).
pub fn extend_eval_bins_with_virtuals(
    train_binned: &BinnedData,
    eval_bin_indices: &mut Vec<u16>,
    eval_cll_hash_bins: Option<&mut Vec<u16>>,
    eval_raw_data: &[f64],
    en_rows: usize,
    n_raw_features_eval: usize,
) {
    if train_binned.virtual_defs.is_empty() {
        return;
    }
    let mut cll = eval_cll_hash_bins;
    for (v_idx, op_def) in train_binned.virtual_defs.iter().enumerate() {
        // virtual_first_id is the feature id of virtual_defs[0]; subsequent
        // virtuals are contiguous. Edges are stored at bin_edges[feat_id].
        let edges_idx = train_binned.virtual_first_id + v_idx;
        let edges = &train_binned.bin_edges[edges_idx];
        let n_bins = edges.len();
        let mut bins = vec![0u16; en_rows];
        let max_bin = n_bins.saturating_sub(1);
        for row in 0..en_rows {
            let row_data =
                &eval_raw_data[row * n_raw_features_eval..(row + 1) * n_raw_features_eval];
            let v = op_def.eval_raw_row(row_data);
            if !v.is_finite() {
                bins[row] = super::MISSING_BIN;
            } else {
                let mut lo = 0usize;
                let mut hi = n_bins;
                while lo < hi {
                    let mid = lo + (hi - lo) / 2;
                    if edges[mid] < v {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                // Cap to [0, n_bins-1] to match histogram allocation.
                bins[row] = lo.min(max_bin) as u16;
            }
        }
        eval_bin_indices.extend_from_slice(&bins);
        if let Some(ref mut c) = cll {
            c.extend(std::iter::repeat(0u16).take(en_rows));
        }
    }
}

/// Convenience wrapper: pre-mine + bin + register all in one call.
/// Returns the number of accepted/registered virtual features.
pub fn premine_and_register(
    binned: &mut BinnedData,
    raw_values_by_col: &[Vec<f64>],
    residual: &[f64],
    cfg: &LtsoPremineConfig,
) -> usize {
    if !cfg.enabled {
        return 0;
    }
    let num_indices: Vec<usize> = (0..binned.n_raw_features)
        .filter(|&i| i < binned.is_categorical.len() && !binned.is_categorical[i])
        .collect();
    let cat_indices: Vec<usize> = (0..binned.n_raw_features)
        .filter(|&i| i < binned.is_categorical.len() && binned.is_categorical[i])
        .collect();
    let accepted = premine_candidates(
        raw_values_by_col,
        &num_indices,
        &cat_indices,
        residual,
        cfg,
        None,
        None,
    );
    let n_accept = accepted.len();
    for (op_def, train_vals, _score) in accepted {
        let (bins, edges) = quantile_bin(&train_vals, cfg.n_bins);
        binned.register_virtual_feature(op_def, bins, edges);
    }
    n_accept
}

impl BinnedData {
    /// Return raw value for `feat` from a test-time raw row.
    /// For raw / derived features (id < virtual_first_id), returns raw_row[feat].
    /// For virtual features, computes from the stored VirtualFeatureDef.
    #[inline]
    pub fn raw_value_for_feat(&self, raw_row: &[f64], feat: usize) -> f64 {
        if self.virtual_first_id == usize::MAX || feat < self.virtual_first_id {
            // Raw or non-virtual derived feature — look up in raw_row.
            if feat < raw_row.len() {
                raw_row[feat]
            } else {
                // Out of raw_row range; happens when caller passed an
                // unextended row for a feat id that requires extension.
                f64::NAN
            }
        } else {
            let v_idx = feat - self.virtual_first_id;
            if v_idx < self.virtual_defs.len() {
                self.virtual_defs[v_idx].eval_raw_row(raw_row)
            } else {
                f64::NAN
            }
        }
    }

    /// Register a new virtual feature into the design matrix.
    /// `train_bins` must have length `self.n_rows`. `edges` are the quantile
    /// cut points used to bin the virtual at training time; the same edges
    /// are applied at predict via binary search.
    ///
    /// Returns the new feature id (== old `self.n_features`).
    pub fn register_virtual_feature(
        &mut self,
        op: VirtualFeatureDef,
        train_bins: Vec<u16>,
        edges: Vec<f64>,
    ) -> usize {
        assert_eq!(
            train_bins.len(),
            self.n_rows,
            "virtual feature train_bins must have length n_rows"
        );
        let new_feature_id = self.n_features;
        if self.virtual_first_id == usize::MAX {
            self.virtual_first_id = new_feature_id;
        }

        // 1. Append column to flat bin_indices.
        self.bin_indices.extend_from_slice(&train_bins);

        // 2. Append edges.
        self.bin_edges.push(edges);

        // 3. Virtuals are numeric, never categorical.
        self.is_categorical.push(false);
        let has_missing = train_bins.iter().any(|&b| b == super::MISSING_BIN);
        self.feature_has_missing.push(has_missing);
        self.feature_non_missing_count.push(
            (self.n_rows
                - train_bins
                    .iter()
                    .filter(|&&b| b == super::MISSING_BIN)
                    .count()) as u32,
        );
        if has_missing {
            for (row, &bin) in train_bins.iter().enumerate() {
                if bin != super::MISSING_BIN {
                    self.non_missing_row_indices.push(row as u32);
                    self.non_missing_bin_values.push(bin);
                }
            }
        }
        self.non_missing_offsets
            .push(self.non_missing_row_indices.len());
        self.cll_is_categorical.push(false);
        self.cll_n_bins
            .push(self.bin_edges[new_feature_id].len().max(1));
        self.cll_hash_bins
            .extend(std::iter::repeat(0u16).take(self.n_rows));
        // 4. Register the op definition (used at predict_raw_row time).
        self.virtual_defs.push(op);

        // 5. Bump n_features. This is what makes the new column visible to
        //    every downstream consumer (find_best_split, histograms, etc.).
        self.n_features += 1;
        new_feature_id
    }
}
