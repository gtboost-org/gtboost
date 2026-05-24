//! Post-build leaf refinement on `DecisionTree`.
//!
//! Once the tree structure is fixed, leaf values can be re-fit on different
//! data (e.g., honest complement rows) or with different objectives (median,
//! trimmed mean, MAD-clipped, robust adaptive blends). This module also
//! holds:
//!
//! - **Leaf refit variants**: `refit_leaves`, `refit_leaves_tau`,
//!   `refit_leaves_trimmed`, `refit_leaves_madclip`, `refit_leaves_median`,
//!   `refit_leaves_adaptive_blend`, `refit_leaves_robust`,
//!   `halley_adjust_leaves`.
//! - **Path features & ramp**: `compute_parent_features`,
//!   `compute_path_features_k`, `ramp_predict`.
//! - **Lookup install / refit**: `install_cat_lookups`,
//!   `install_best_lookups`, `install_best_lookups_guided`,
//!   `install_numeric_lookups`, plus `refit_*_lookups` companions.
//! - **Scaling & shrinkage**: `scale_ramp_slopes`, `scale_cat_lookups`,
//!   `scale_output`, `posterior_shrink_leaves`,
//!   `hierarchical_shrink_experts`.
//! - **Post-build leaf splits**: `try_split_leaves`,
//!   `try_split_leaves_precomputed`, `try_split_leaves_multi`.

use std::cmp::Ordering;

use super::algorithms::{
    eval_best_lookup_for_node_with_config, find_best_split, l1_leaf_value, split_goes_left_binned,
    sum_gh, LeafExpertKind,
};
use super::*;

impl DecisionTree {
    /// Re-optimize leaf values only (structure unchanged). O(n_rows) per tree.
    /// When used for honest estimation (complement data), leaves with few samples
    /// get stronger regularization to prevent noisy estimates from dominating.
    pub fn refit_leaves(
        &mut self,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        row_indices: &[u32],
        lambda_reg: f64,
    ) {
        self.refit_leaves_l1(binned, gradients, hessians, row_indices, lambda_reg, 0.0);
    }

    pub fn refit_leaves_l1(
        &mut self,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        row_indices: &[u32],
        lambda_reg: f64,
        l1_reg: f64,
    ) {
        self.refit_leaves_tau_l1(
            binned,
            gradients,
            hessians,
            row_indices,
            lambda_reg,
            0.0,
            l1_reg,
        );
    }

    /// Refit leaf values from complement data, with optional Bayesian blending.
    /// tau > 0: blend complement estimate with structure-set prior (current values[i]).
    /// w_blend = (n_comp / (n_comp + tau)) * w_complement + (tau / (n_comp + tau)) * w_structure
    pub fn refit_leaves_tau(
        &mut self,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        row_indices: &[u32],
        lambda_reg: f64,
        tau: f64,
    ) {
        self.refit_leaves_tau_l1(
            binned,
            gradients,
            hessians,
            row_indices,
            lambda_reg,
            tau,
            0.0,
        );
    }

    fn refit_leaves_tau_l1(
        &mut self,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        row_indices: &[u32],
        lambda_reg: f64,
        tau: f64,
        l1_reg: f64,
    ) {
        let n_nodes = self.split_features.len();
        let mut leaf_g = vec![0.0f64; n_nodes];
        let mut leaf_h = vec![0.0f64; n_nodes];
        let mut leaf_count = vec![0u32; n_nodes];

        for &idx in row_indices {
            let leaf = self.route_to_leaf(binned, idx as usize);
            leaf_g[leaf] += gradients[idx as usize];
            leaf_h[leaf] += hessians[idx as usize];
            leaf_count[leaf] += 1;
        }

        // Adaptive lambda: leaves with fewer samples get more regularization.
        const MIN_SAMPLES: f64 = 10.0;

        for i in 0..n_nodes {
            if self.split_features[i] == u32::MAX && leaf_h[i] > 0.0 {
                let count = leaf_count[i] as f64;
                let lambda_eff = lambda_reg * (MIN_SAMPLES / count).max(1.0);
                let w_complement = l1_leaf_value(leaf_g[i], leaf_h[i], lambda_eff, l1_reg);
                self.node_h_sum[i] = leaf_h[i];
                self.node_count[i] = leaf_count[i];
                if tau > 0.0 {
                    let w_structure = self.values[i]; // prior from structure set
                    let blend = count / (count + tau);
                    self.values[i] = blend * w_complement + (1.0 - blend) * w_structure;
                } else {
                    self.values[i] = w_complement;
                }
            }
        }
    }

    /// Refit leaves with Huber-style gradient trimming for robust leaf values.
    /// Trims top and bottom trim_pct fraction of gradients per leaf before Newton step.
    /// Keeps gradient SCALE unchanged at training time (so ES behavior is preserved) —
    /// only affects leaf-value computation. Classical robust M-estimator (Huber 1964).
    /// When trim_pct = 0.0, behaves identically to refit_leaves_tau.
    pub fn refit_leaves_trimmed(
        &mut self,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        row_indices: &[u32],
        lambda_reg: f64,
        tau: f64,
        trim_pct: f64,
    ) {
        if trim_pct <= 0.0 {
            self.refit_leaves_tau(binned, gradients, hessians, row_indices, lambda_reg, tau);
            return;
        }

        let n_nodes = self.split_features.len();
        // Collect per-leaf (g, h) tuples for trimming
        let mut leaf_samples: Vec<Vec<(f64, f64)>> = (0..n_nodes).map(|_| Vec::new()).collect();

        for &idx in row_indices {
            let leaf = self.route_to_leaf(binned, idx as usize);
            leaf_samples[leaf].push((gradients[idx as usize], hessians[idx as usize]));
        }

        const MIN_SAMPLES: f64 = 10.0;
        const MIN_LEAF_FOR_TRIM: usize = 10; // don't trim tiny leaves

        for i in 0..n_nodes {
            if self.split_features[i] != u32::MAX {
                continue;
            }
            let samples = &mut leaf_samples[i];
            let n = samples.len();
            if n == 0 {
                continue;
            }

            // Compute trimmed sums
            let (leaf_g, leaf_h, count) = if n >= MIN_LEAF_FOR_TRIM && trim_pct > 0.0 {
                // Sort by |gradient|, drop top trim_pct fraction
                samples.sort_by(|a, b| {
                    a.0.abs()
                        .partial_cmp(&b.0.abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let keep = ((n as f64) * (1.0 - trim_pct)).round().max(1.0) as usize;
                let keep = keep.min(n);
                let mut g_sum = 0.0f64;
                let mut h_sum = 0.0f64;
                for &(g, h) in &samples[..keep] {
                    g_sum += g;
                    h_sum += h;
                }
                (g_sum, h_sum, keep as f64)
            } else {
                let mut g_sum = 0.0f64;
                let mut h_sum = 0.0f64;
                for &(g, h) in samples.iter() {
                    g_sum += g;
                    h_sum += h;
                }
                (g_sum, h_sum, n as f64)
            };

            if leaf_h > 0.0 {
                self.node_h_sum[i] = leaf_h;
                self.node_count[i] = n as u32;
                let lambda_eff = lambda_reg * (MIN_SAMPLES / count).max(1.0);
                let w_complement = -leaf_g / (leaf_h + lambda_eff);
                if tau > 0.0 {
                    let w_structure = self.values[i];
                    let blend = count / (count + tau);
                    self.values[i] = blend * w_complement + (1.0 - blend) * w_structure;
                } else {
                    self.values[i] = w_complement;
                }
            }
        }
    }

    /// Adaptive robust leaf refit using a per-leaf MAD scale estimate.
    /// Steps:
    /// 1. Convert each row to its per-row Newton target r_i = -g_i / h_i
    /// 2. Estimate a robust leaf center via weighted median(r_i; h_i)
    /// 3. Estimate leaf noise scale via weighted median(|r_i - center|; h_i)
    /// 4. Winsorize r_i to [center - c*scale, center + c*scale] and take the
    ///    h-weighted mean
    ///
    /// This is a one-step Hampel/Huber-style M-estimator in leaf space. Unlike
    /// fixed trim_pct, the clipping radius adapts to each leaf's own noise.
    /// `mad_clip` is the cutoff multiplier (classical Tukey/Huber values are
    /// around 3-5). 0.0 disables this path.
    pub fn refit_leaves_madclip(
        &mut self,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        row_indices: &[u32],
        lambda_reg: f64,
        tau: f64,
        mad_clip: f64,
    ) {
        if mad_clip <= 0.0 {
            self.refit_leaves_tau(binned, gradients, hessians, row_indices, lambda_reg, tau);
            return;
        }

        let n_nodes = self.split_features.len();
        let mut leaf_samples: Vec<Vec<(f64, f64)>> = (0..n_nodes).map(|_| Vec::new()).collect();

        for &idx in row_indices {
            let leaf = self.route_to_leaf(binned, idx as usize);
            leaf_samples[leaf].push((gradients[idx as usize], hessians[idx as usize]));
        }

        const MIN_SAMPLES: f64 = 10.0;
        const MIN_LEAF_FOR_MAD: usize = 8;
        const MAD_EPS: f64 = 1e-6;

        for i in 0..n_nodes {
            if self.split_features[i] != u32::MAX {
                continue;
            }
            let samples = &leaf_samples[i];
            let n = samples.len();
            if n == 0 {
                continue;
            }

            let mut responses: Vec<(f64, f64)> = Vec::with_capacity(n);
            let mut h_sum = 0.0f64;
            let mut g_sum = 0.0f64;
            for &(g, h) in samples.iter() {
                if h <= 0.0 {
                    continue;
                }
                responses.push((-g / h.max(1e-12), h));
                h_sum += h;
                g_sum += g;
            }
            if responses.is_empty() || h_sum <= 0.0 {
                continue;
            }

            let count = responses.len() as f64;
            let lambda_eff = lambda_reg * (MIN_SAMPLES / count).max(1.0);

            // Robust center via weighted median of r_i = -g_i / h_i.
            responses.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let half_h = h_sum / 2.0;
            let mut cum_h = 0.0f64;
            let mut center = responses[0].0;
            for &(r, h) in responses.iter() {
                cum_h += h;
                center = r;
                if cum_h >= half_h {
                    break;
                }
            }

            let w_complement = if responses.len() < MIN_LEAF_FOR_MAD {
                // Tiny leaves: MAD is unstable; fall back to the standard Newton step.
                -g_sum / (h_sum + lambda_eff)
            } else {
                let mut devs: Vec<(f64, f64)> = responses
                    .iter()
                    .map(|&(r, h)| ((r - center).abs(), h))
                    .collect();
                devs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                let mut cum_dev_h = 0.0f64;
                let mut mad = devs[0].0;
                for &(d, h) in devs.iter() {
                    cum_dev_h += h;
                    mad = d;
                    if cum_dev_h >= half_h {
                        break;
                    }
                }
                let scale = (1.4826 * mad).max(MAD_EPS);
                let cutoff = mad_clip * scale;
                let lo = center - cutoff;
                let hi = center + cutoff;
                let mut clipped_sum = 0.0f64;
                for &(r, h) in responses.iter() {
                    clipped_sum += h * r.clamp(lo, hi);
                }
                clipped_sum / (h_sum + lambda_eff)
            };

            self.node_h_sum[i] = h_sum;
            self.node_count[i] = n as u32;
            if tau > 0.0 {
                let w_structure = self.values[i];
                let blend = count / (count + tau);
                self.values[i] = blend * w_complement + (1.0 - blend) * w_structure;
            } else {
                self.values[i] = w_complement;
            }
        }
    }

    /// §124 LAD-TreeBoost (Friedman 1999): weighted median leaf values for robust regression.
    /// Solves min_w Σ h_i |g_i + h_i w| (L1 leaf loss) instead of Σ 0.5 h_i (g_i + h_i w)^2 (Newton).
    /// For MSE (h=1): equivalent to median(-g_i) = median residual. Classical LAD boosting.
    /// For binary/multiclass: weighted median of -g_i/h_i with weights h_i.
    /// Preserves gradient scale (§112 invariant) — only changes leaf-value summary.
    /// `blend` ∈ [0, 1]: 1.0 = pure median (default when called from refit_leaves_robust with
    /// use_median=true), intermediate values blend `(1-blend)·newton + blend·median` — §124b.
    pub fn refit_leaves_median(
        &mut self,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        row_indices: &[u32],
        lambda_reg: f64,
        tau: f64,
        blend: f64,
    ) {
        let n_nodes = self.split_features.len();
        let mut leaf_samples: Vec<Vec<(f64, f64)>> = (0..n_nodes).map(|_| Vec::new()).collect();

        for &idx in row_indices {
            let leaf = self.route_to_leaf(binned, idx as usize);
            leaf_samples[leaf].push((gradients[idx as usize], hessians[idx as usize]));
        }

        const MIN_SAMPLES: f64 = 10.0;

        for i in 0..n_nodes {
            if self.split_features[i] != u32::MAX {
                continue;
            }
            let samples = &mut leaf_samples[i];
            let n = samples.len();
            if n == 0 {
                continue;
            }

            // Compute H_sum for regularization scaling
            let mut h_sum = 0.0f64;
            for &(_, h) in samples.iter() {
                h_sum += h;
            }
            if h_sum <= 0.0 {
                continue;
            }

            // Sort by per-sample optimal direction r_i = -g_i / max(h_i, eps)
            samples.sort_by(|a, b| {
                let ra = -a.0 / a.1.max(1e-12);
                let rb = -b.0 / b.1.max(1e-12);
                ra.partial_cmp(&rb).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Weighted median by hessian weights. L1-optimal leaf direction.
            let half = h_sum / 2.0;
            let mut cum_h = 0.0f64;
            let mut median = -samples[0].0 / samples[0].1.max(1e-12);
            let mut g_sum = 0.0f64;
            for &(g, h) in samples.iter() {
                g_sum += g;
                cum_h += h;
                if cum_h < half {
                    median = -g / h.max(1e-12);
                } else if (cum_h - h) < half {
                    // cross point
                    median = -g / h.max(1e-12);
                }
            }

            let count = n as f64;
            let lambda_eff = lambda_reg * (MIN_SAMPLES / count).max(1.0);
            // L1 ridge-shrunk leaf.
            let w_median = median * h_sum / (h_sum + lambda_eff);
            // Newton (L2) step on same (g, h) pool for blend.
            let w_newton = -g_sum / (h_sum + lambda_eff);
            // §124b: blend 0 → pure newton (backwards compat with median-off); 1 → pure LAD.
            let blend_c = blend.clamp(0.0, 1.0);
            let w_complement = blend_c * w_median + (1.0 - blend_c) * w_newton;

            self.node_h_sum[i] = h_sum;
            self.node_count[i] = n as u32;
            if tau > 0.0 {
                let w_structure = self.values[i];
                let blend = count / (count + tau);
                self.values[i] = blend * w_complement + (1.0 - blend) * w_structure;
            } else {
                self.values[i] = w_complement;
            }
        }
    }

    /// Adaptive Newton↔median bridge based on each leaf's own response shape.
    /// The leaf computes:
    /// - Newton center: h-weighted mean of r_i = -g_i / h_i
    /// - Robust center: h-weighted median of r_i
    /// - Robust scale: weighted MAD around the median
    ///
    /// Then it measures how far mean and median disagree in MAD units. Clean,
    /// symmetric leaves stay close to Newton; skewed / outlier-contaminated
    /// leaves shift smoothly toward the median.
    pub fn refit_leaves_adaptive_blend(
        &mut self,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        row_indices: &[u32],
        lambda_reg: f64,
        tau: f64,
        blend_kappa: f64,
    ) {
        if blend_kappa <= 0.0 {
            self.refit_leaves_tau(binned, gradients, hessians, row_indices, lambda_reg, tau);
            return;
        }

        let n_nodes = self.split_features.len();
        let mut leaf_samples: Vec<Vec<(f64, f64)>> = (0..n_nodes).map(|_| Vec::new()).collect();

        for &idx in row_indices {
            let leaf = self.route_to_leaf(binned, idx as usize);
            leaf_samples[leaf].push((gradients[idx as usize], hessians[idx as usize]));
        }

        const MIN_SAMPLES: f64 = 10.0;
        const MIN_LEAF_FOR_ADAPTIVE: usize = 8;
        const MAD_EPS: f64 = 1e-6;

        for i in 0..n_nodes {
            if self.split_features[i] != u32::MAX {
                continue;
            }
            let samples = &mut leaf_samples[i];
            let n = samples.len();
            if n == 0 {
                continue;
            }

            let mut responses: Vec<(f64, f64)> = Vec::with_capacity(n);
            let mut h_sum = 0.0f64;
            let mut g_sum = 0.0f64;
            for &(g, h) in samples.iter() {
                if h <= 0.0 {
                    continue;
                }
                responses.push((-g / h.max(1e-12), h));
                h_sum += h;
                g_sum += g;
            }
            if responses.is_empty() || h_sum <= 0.0 {
                continue;
            }

            let count = responses.len() as f64;
            let lambda_eff = lambda_reg * (MIN_SAMPLES / count).max(1.0);
            let mean = -g_sum / h_sum.max(1e-12);

            responses.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            let half_h = h_sum / 2.0;
            let mut cum_h = 0.0f64;
            let mut median = responses[0].0;
            for &(r, h) in responses.iter() {
                cum_h += h;
                median = r;
                if cum_h >= half_h {
                    break;
                }
            }

            let center = if responses.len() < MIN_LEAF_FOR_ADAPTIVE {
                mean
            } else {
                let mut devs: Vec<(f64, f64)> = responses
                    .iter()
                    .map(|&(r, h)| ((r - median).abs(), h))
                    .collect();
                devs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
                let mut cum_dev_h = 0.0f64;
                let mut mad = devs[0].0;
                for &(d, h) in devs.iter() {
                    cum_dev_h += h;
                    mad = d;
                    if cum_dev_h >= half_h {
                        break;
                    }
                }
                let scale = (1.4826 * mad).max(MAD_EPS);
                let z = (mean - median).abs() / scale;
                let z2 = z * z;
                let k2 = blend_kappa * blend_kappa;
                let blend = z2 / (z2 + k2);
                blend * median + (1.0 - blend) * mean
            };

            let w_complement = center * h_sum / (h_sum + lambda_eff);

            self.node_h_sum[i] = h_sum;
            self.node_count[i] = n as u32;
            if tau > 0.0 {
                let w_structure = self.values[i];
                let blend = count / (count + tau);
                self.values[i] = blend * w_complement + (1.0 - blend) * w_structure;
            } else {
                self.values[i] = w_complement;
            }
        }
    }

    /// §124 dispatch: unified robust-leaf entry.
    /// - use_median_blend > 0 → median with blend factor (1 = pure LAD, 0 = pure Newton).
    /// - use_median=true overrides blend to 1 (pure median) for backwards compat.
    /// - both off → trimmed-mean.
    #[inline]
    pub fn refit_leaves_robust(
        &mut self,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        row_indices: &[u32],
        lambda_reg: f64,
        tau: f64,
        trim_pct: f64,
        use_median: bool,
        median_blend: f64,
        mad_clip: f64,
        adaptive_blend_kappa: f64,
    ) {
        if mad_clip > 0.0 {
            self.refit_leaves_madclip(
                binned,
                gradients,
                hessians,
                row_indices,
                lambda_reg,
                tau,
                mad_clip,
            );
            return;
        }
        let blend = if use_median { 1.0 } else { median_blend };
        if blend > 0.0 {
            self.refit_leaves_median(
                binned,
                gradients,
                hessians,
                row_indices,
                lambda_reg,
                tau,
                blend,
            );
        } else if adaptive_blend_kappa > 0.0 {
            self.refit_leaves_adaptive_blend(
                binned,
                gradients,
                hessians,
                row_indices,
                lambda_reg,
                tau,
                adaptive_blend_kappa,
            );
        } else {
            self.refit_leaves_trimmed(
                binned,
                gradients,
                hessians,
                row_indices,
                lambda_reg,
                tau,
                trim_pct,
            );
        }
    }

    /// Halley's method: apply 3rd-order correction to leaf values for binary classification.
    /// w_halley = -G / (H + λ - G*T / (2*(H+λ)))  where T = Σ p(1-p)(1-2p) per leaf.
    pub fn halley_adjust_leaves(
        &mut self,
        binned: &BinnedData,
        thirds: &[f64],
        gradients: &[f64],
        hessians: &[f64],
        row_indices: &[u32],
        lambda_reg: f64,
    ) {
        let n_nodes = self.split_features.len();
        let mut leaf_g = vec![0.0f64; n_nodes];
        let mut leaf_h = vec![0.0f64; n_nodes];
        let mut leaf_t = vec![0.0f64; n_nodes];

        for &idx in row_indices {
            let leaf = self.route_to_leaf(binned, idx as usize);
            leaf_g[leaf] += gradients[idx as usize];
            leaf_h[leaf] += hessians[idx as usize];
            leaf_t[leaf] += thirds[idx as usize];
        }

        for i in 0..n_nodes {
            if self.split_features[i] == u32::MAX && leaf_h[i] > 0.0 {
                let g = leaf_g[i];
                let h = leaf_h[i] + lambda_reg;
                let t = leaf_t[i];
                let denom = h - g * t / (2.0 * h);
                if denom.abs() > 1e-10 {
                    self.values[i] = -g / denom;
                }
                // else keep Newton value (already set)
            }
        }
    }

    /// Compute parent split feature for each node (K=1 case).
    pub fn compute_parent_features(&self) -> Vec<u32> {
        let n = self.split_features.len();
        let mut parent_feat = vec![u32::MAX; n];
        for i in 0..n {
            if self.split_features[i] != u32::MAX {
                let left = self.left_children[i] as usize;
                let right = self.right_children[i] as usize;
                if left < n {
                    parent_feat[left] = self.split_features[i];
                }
                if right < n {
                    parent_feat[right] = self.split_features[i];
                }
            }
        }
        parent_feat
    }

    /// Compute K path features per node (recent ancestors' split features).
    /// Returns flat vec: result[node*k + j] = j-th ancestor's split feature (0=parent, 1=grandparent, ...).
    pub fn compute_path_features_k(&self, k: usize) -> Vec<u32> {
        let n = self.split_features.len();
        let mut result = vec![u32::MAX; n * k];
        // Process nodes in order (BFS order: parent always before children)
        for i in 0..n {
            if self.split_features[i] != u32::MAX {
                let left = self.left_children[i] as usize;
                let right = self.right_children[i] as usize;
                for child in [left, right] {
                    if child < n {
                        // child's feature[0] = this node's split feature (parent)
                        result[child * k] = self.split_features[i];
                        // child's feature[j+1] = this node's feature[j] (shift ancestors down)
                        for j in 0..k - 1 {
                            result[child * k + j + 1] = result[i * k + j];
                        }
                    }
                }
            }
        }
        result
    }

    /// Ramp prediction: returns the slope contribution for a leaf.
    #[inline]
    pub fn ramp_predict(&self, node: usize, bin_indices: &[u16], n_rows: usize, row: usize) -> f64 {
        if self.ramp_slopes.is_empty()
            && self.leaf_pair_slopes.is_empty()
            && self.quad_slopes.is_empty()
        {
            return 0.0;
        }
        let mut sum = 0.0f64;
        if !self.ramp_slopes.is_empty() {
            let k = self.ramp_k;
            let base = node * k;
            if base + k <= self.ramp_features.len() {
                for j in 0..k {
                    let feat = self.ramp_features[base + j];
                    if feat == u32::MAX {
                        continue;
                    }
                    let bin = bin_indices[feat as usize * n_rows + row];
                    if bin == MISSING_BIN {
                        continue;
                    }
                    sum += self.ramp_slopes[base + j] * bin as f64;
                }
            }
        }
        if !self.leaf_pair_slopes.is_empty() {
            let base = node * 2;
            if base + 1 < self.leaf_pair_features.len() && node < self.leaf_pair_slopes.len() {
                let f0 = self.leaf_pair_features[base];
                let f1 = self.leaf_pair_features[base + 1];
                if f0 != u32::MAX && f1 != u32::MAX {
                    let b0 = bin_indices[f0 as usize * n_rows + row];
                    let b1 = bin_indices[f1 as usize * n_rows + row];
                    if b0 != MISSING_BIN && b1 != MISSING_BIN {
                        sum += self.leaf_pair_slopes[node] * b0 as f64 * b1 as f64;
                    }
                }
            }
        }
        // Quadratic interaction contributions
        if !self.quad_slopes.is_empty() && self.quad_n_interactions > 0 {
            let ni = self.quad_n_interactions;
            let qbase = node * ni;
            if qbase + ni <= self.quad_slopes.len() {
                for j in 0..ni {
                    let (fi, fj) = self.quad_pairs[j];
                    let bi = bin_indices[fi * n_rows + row];
                    let bj = bin_indices[fj * n_rows + row];
                    if bi == MISSING_BIN || bj == MISSING_BIN {
                        continue;
                    }
                    sum += self.quad_slopes[qbase + j] * bi as f64 * bj as f64;
                }
            }
        }
        sum
    }

    /// Refit CLL bin values using new data (e.g., complement set for honest estimation).
    /// Keeps the CLL selection (which feature, which node) but recomputes values.
    pub fn refit_cat_lookups(
        &mut self,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        row_indices: &[u32],
        lambda_reg: f64,
        smooth: f64,
        min_child_weight: f64,
    ) {
        let n_nodes = self.split_features.len();

        // Route samples to leaves
        let mut leaf_samples: Vec<Vec<u32>> = vec![Vec::new(); n_nodes];
        for &idx in row_indices {
            let leaf = self.route_to_leaf(binned, idx as usize);
            leaf_samples[leaf].push(idx);
        }

        for node in 0..n_nodes {
            if self.cat_lookups[node].is_none() {
                continue;
            }
            let samples = &leaf_samples[node];
            let cll = self.cat_lookups[node].as_ref().unwrap();
            if cll.is_numeric {
                continue;
            }
            let feat = cll.feature as usize;
            let feat2 = cll.feature2;
            let n_bins = cll.bin_values.len();

            let mut g_total = 0.0f64;
            let mut h_total = 0.0f64;
            for &idx in samples {
                g_total += gradients[idx as usize];
                h_total += hessians[idx as usize];
            }
            let nc_leaf = samples.len() as f64;
            let leaf_value = if h_total > 0.0 {
                -g_total / (h_total + lambda_reg + lambda_reg / nc_leaf.max(1.0).sqrt())
            } else {
                self.values[node]
            };
            self.values[node] = leaf_value;

            // Recompute per-bin values using cll_bin_for_row (handles pairs)
            let mut bin_g = vec![0.0f64; n_bins];
            let mut bin_h = vec![0.0f64; n_bins];
            for &idx in samples {
                let b = cll_bin_for_row(cll, &binned.cll_hash_bins, binned.n_rows, idx as usize);
                if b == usize::MAX || b >= n_bins {
                    continue;
                }
                bin_g[b] += gradients[idx as usize];
                bin_h[b] += hessians[idx as usize];
            }

            let mut bin_values = vec![leaf_value; n_bins];
            for b in 0..n_bins {
                if bin_h[b] >= min_child_weight {
                    let cat_value = -bin_g[b] / (bin_h[b] + lambda_reg);
                    if smooth > 0.0 {
                        bin_values[b] =
                            (bin_h[b] * cat_value + smooth * leaf_value) / (bin_h[b] + smooth);
                    } else {
                        bin_values[b] = cat_value;
                    }
                }
            }

            self.cat_lookups[node] = Some(CatLookup {
                feature: feat as u32,
                feature2: feat2,
                feature3: cll.feature3,
                bin_values,
                default_value: leaf_value,
                is_numeric: false,
                n_coarse_bins: 0,
                pair_stride: cll.pair_stride,
                triple_stride: cll.triple_stride,
            });
        }
    }

    /// Refit whichever lookup type each leaf currently owns.
    pub fn refit_best_lookups(
        &mut self,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        row_indices: &[u32],
        lambda_reg: f64,
        smooth: f64,
        min_child_weight: f64,
    ) {
        self.refit_cat_lookups(
            binned,
            gradients,
            hessians,
            row_indices,
            lambda_reg,
            smooth,
            min_child_weight,
        );
        self.refit_numeric_lookups(
            binned,
            gradients,
            hessians,
            row_indices,
            lambda_reg,
            smooth,
            min_child_weight,
        );
    }

    /// Install Category Lookup Leaves: for each leaf, find the best categorical feature
    /// and replace the single leaf value with per-category values if the gain is positive.
    /// `smooth` controls regularization (higher = more shrinkage toward leaf value).
    pub fn install_cat_lookups(
        &mut self,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        row_indices: &[u32],
        lambda_reg: f64,
        gamma: f64,
        min_child_weight: f64,
        smooth: f64,
    ) {
        let n_nodes = self.split_features.len();

        // Find categorical features
        let cat_cols: Vec<usize> = (0..binned.n_features)
            .filter(|&c| c < binned.is_categorical.len() && binned.is_categorical[c])
            .collect();
        if cat_cols.is_empty() {
            return;
        }

        // Route all samples to leaves
        let mut leaf_samples: Vec<Vec<u32>> = vec![Vec::new(); n_nodes];
        for &idx in row_indices {
            let leaf = self.route_to_leaf(binned, idx as usize);
            leaf_samples[leaf].push(idx);
        }

        // For each leaf, evaluate all categorical features
        for node in 0..n_nodes {
            if self.split_features[node] != u32::MAX {
                continue;
            } // skip internal nodes
            let samples = &leaf_samples[node];
            if samples.len() < 2 {
                continue;
            } // need at least 2 samples

            // Compute total g/h for this leaf
            let mut g_total = 0.0f64;
            let mut h_total = 0.0f64;
            for &idx in samples {
                g_total += gradients[idx as usize];
                h_total += hessians[idx as usize];
            }
            let base_obj = g_total * g_total / (h_total + lambda_reg);

            let mut best_gain = 0.0f64;
            let mut best_feature = 0usize;
            let mut best_bin_g: Vec<f64> = Vec::new();
            let mut best_bin_h: Vec<f64> = Vec::new();
            let mut best_n_bins = 0usize;

            for &col in &cat_cols {
                let n_bins = binned.n_bins(col);
                let mut bin_g = vec![0.0f64; n_bins];
                let mut bin_h = vec![0.0f64; n_bins];

                for &idx in samples {
                    let bin = binned.get_bin_u16(idx as usize, col);
                    if bin == MISSING_BIN {
                        continue;
                    }
                    let b = bin as usize;
                    if b < n_bins {
                        bin_g[b] += gradients[idx as usize];
                        bin_h[b] += hessians[idx as usize];
                    }
                }

                // CLL gain: sum(g_cat^2 / (h_cat + lambda)) - g_total^2 / (h_total + lambda)
                let mut cll_obj = 0.0f64;
                let mut n_active = 0usize;
                for b in 0..n_bins {
                    if bin_h[b] >= min_child_weight {
                        cll_obj += bin_g[b] * bin_g[b] / (bin_h[b] + lambda_reg);
                        n_active += 1;
                    }
                    // Categories below min_child_weight contribute nothing (use leaf default)
                }
                if n_active < 2 {
                    continue;
                } // need at least 2 active categories

                let gain = 0.5 * (cll_obj - base_obj) - gamma * (n_active as f64).sqrt();
                if gain > best_gain {
                    best_gain = gain;
                    best_feature = col;
                    best_bin_g = bin_g;
                    best_bin_h = bin_h;
                    best_n_bins = n_bins;
                }
            }

            if best_gain > 0.0 {
                // Install CLL for this leaf
                let leaf_value = self.values[node];
                let mut bin_values = vec![leaf_value; best_n_bins];
                for b in 0..best_n_bins {
                    if best_bin_h[b] >= min_child_weight {
                        let cat_value = -best_bin_g[b] / (best_bin_h[b] + lambda_reg);
                        // Smooth toward leaf value
                        if smooth > 0.0 {
                            bin_values[b] = (best_bin_h[b] * cat_value + smooth * leaf_value)
                                / (best_bin_h[b] + smooth);
                        } else {
                            bin_values[b] = cat_value;
                        }
                    }
                    // else: keep leaf_value as default for rare categories
                }
                self.cat_lookups[node] = Some(CatLookup {
                    feature: best_feature as u32,
                    feature2: u32::MAX,
                    feature3: u32::MAX,
                    bin_values,
                    default_value: leaf_value,
                    is_numeric: false,
                    n_coarse_bins: 0,
                    pair_stride: 0,
                    triple_stride: 0,
                });
            }
        }
    }

    /// Adaptive Leaf Experts: for each leaf, choose the best post-fit local expert.
    /// Candidates: categorical lookup or coarse-binned numeric lookup.
    pub fn install_best_lookups(
        &mut self,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        row_indices: &[u32],
        lambda_reg: f64,
        gamma: f64,
        min_child_weight: f64,
        smooth: f64,
    ) {
        self.install_best_lookups_inner(
            binned,
            gradients,
            hessians,
            row_indices,
            lambda_reg,
            gamma,
            min_child_weight,
            smooth,
            false,
            None,
            None,
        );
    }

    pub fn install_best_lookups_with_config(
        &mut self,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        row_indices: &[u32],
        lambda_reg: f64,
        gamma: f64,
        min_child_weight: f64,
        smooth: f64,
        adaptive_smooth: bool,
        tuple_cfg: Option<&CatTupleConfig>,
    ) {
        self.install_best_lookups_inner(
            binned,
            gradients,
            hessians,
            row_indices,
            lambda_reg,
            gamma,
            min_child_weight,
            smooth,
            adaptive_smooth,
            None,
            tuple_cfg,
        );
    }

    pub fn install_best_lookups_guided(
        &mut self,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        row_indices: &[u32],
        lambda_reg: f64,
        gamma: f64,
        min_child_weight: f64,
        smooth: f64,
        guided_choices: &[Option<GuidedCatChoice>],
    ) {
        self.install_best_lookups_inner(
            binned,
            gradients,
            hessians,
            row_indices,
            lambda_reg,
            gamma,
            min_child_weight,
            smooth,
            false,
            Some(guided_choices),
            None,
        );
    }

    fn install_best_lookups_inner(
        &mut self,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        row_indices: &[u32],
        lambda_reg: f64,
        gamma: f64,
        min_child_weight: f64,
        smooth: f64,
        adaptive_smooth: bool,
        guided_choices: Option<&[Option<GuidedCatChoice>]>,
        tuple_cfg: Option<&CatTupleConfig>,
    ) {
        let n_nodes = self.split_features.len();
        if self.ramp_k != 2 {
            self.ramp_k = 2;
        }
        let ramp_len = n_nodes * self.ramp_k;
        if self.ramp_features.len() != ramp_len {
            self.ramp_features.resize(ramp_len, u32::MAX);
        }
        if self.ramp_slopes.len() != ramp_len {
            self.ramp_slopes.resize(ramp_len, 0.0);
        }
        let pair_feat_len = n_nodes * 2;
        if self.leaf_pair_features.len() != pair_feat_len {
            self.leaf_pair_features.resize(pair_feat_len, u32::MAX);
        }
        if self.leaf_pair_slopes.len() != n_nodes {
            self.leaf_pair_slopes.resize(n_nodes, 0.0);
        }
        let mut leaf_samples: Vec<Vec<u32>> = vec![Vec::new(); n_nodes];
        for &idx in row_indices {
            let leaf = self.route_to_leaf(binned, idx as usize);
            leaf_samples[leaf].push(idx);
        }

        for node in 0..n_nodes {
            if self.split_features[node] != u32::MAX {
                continue;
            }
            let samples = &leaf_samples[node];
            if samples.len() < 2 {
                continue;
            }
            let ramp_base = node * self.ramp_k;
            self.cat_lookups[node] = None;
            for j in 0..self.ramp_k {
                self.ramp_features[ramp_base + j] = u32::MAX;
                self.ramp_slopes[ramp_base + j] = 0.0;
            }
            let pair_base = node * 2;
            self.leaf_pair_features[pair_base] = u32::MAX;
            self.leaf_pair_features[pair_base + 1] = u32::MAX;
            self.leaf_pair_slopes[node] = 0.0;

            let mut g_total = 0.0f64;
            let mut h_total = 0.0f64;
            for &idx in samples {
                g_total += gradients[idx as usize];
                h_total += hessians[idx as usize];
            }

            if let Some(best) = eval_best_lookup_for_node_with_config(
                binned,
                gradients,
                hessians,
                samples,
                g_total,
                h_total,
                self.values[node],
                lambda_reg,
                gamma,
                min_child_weight,
                smooth,
                adaptive_smooth,
                guided_choices
                    .and_then(|choices| choices.get(node))
                    .and_then(|c| c.as_ref()),
                tuple_cfg,
            ) {
                match best.kind {
                    LeafExpertKind::Lookup(lookup) => {
                        self.cat_lookups[node] = Some(lookup);
                    }
                    LeafExpertKind::Linear {
                        feats,
                        slopes,
                        n_feats,
                        intercept,
                    } => {
                        self.values[node] = intercept;
                        for j in 0..n_feats.min(self.ramp_k) {
                            self.ramp_features[ramp_base + j] = feats[j] as u32;
                            self.ramp_slopes[ramp_base + j] = slopes[j];
                        }
                    }
                    LeafExpertKind::Bilinear {
                        feats,
                        slopes,
                        n_feats,
                        pair_slope,
                        intercept,
                    } => {
                        self.values[node] = intercept;
                        for j in 0..n_feats.min(self.ramp_k) {
                            self.ramp_features[ramp_base + j] = feats[j] as u32;
                            self.ramp_slopes[ramp_base + j] = slopes[j];
                        }
                        if n_feats >= 2 {
                            self.leaf_pair_features[pair_base] = feats[0] as u32;
                            self.leaf_pair_features[pair_base + 1] = feats[1] as u32;
                            self.leaf_pair_slopes[node] = pair_slope;
                        }
                    }
                }
            }
        }
    }

    /// Install Numeric Leaf Lookups (NLL): for each leaf without an existing CLL,
    /// evaluate all numeric features and install a coarsened-bin lookup where gain > 0.
    /// This is a post-training refinement step.
    pub fn install_numeric_lookups(
        &mut self,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        row_indices: &[u32],
        lambda_reg: f64,
        gamma: f64,
        min_child_weight: f64,
        smooth: f64,
        nll_n_bins: usize,
    ) {
        if nll_n_bins < 2 {
            return;
        }
        let n_nodes = self.split_features.len();

        // Identify numeric features (not categorical)
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
            return;
        }

        // Route all samples to leaves
        let mut leaf_samples: Vec<Vec<u32>> = vec![Vec::new(); n_nodes];
        for &idx in row_indices {
            let leaf = self.route_to_leaf(binned, idx as usize);
            leaf_samples[leaf].push(idx);
        }

        for node in 0..n_nodes {
            // Only leaf nodes without existing CLL
            if self.split_features[node] != u32::MAX {
                continue;
            }
            if self.cat_lookups[node].is_some() {
                continue;
            }
            let samples = &leaf_samples[node];
            if samples.len() < 2 * nll_n_bins {
                continue;
            } // need enough samples

            let mut g_total = 0.0f64;
            let mut h_total = 0.0f64;
            for &idx in samples {
                g_total += gradients[idx as usize];
                h_total += hessians[idx as usize];
            }
            let base_obj = g_total * g_total / (h_total + lambda_reg);

            let mut best_gain = 0.0f64;
            let mut best_feature = 0usize;
            let mut best_bin_g: Vec<f64> = Vec::new();
            let mut best_bin_h: Vec<f64> = Vec::new();

            for &col in &num_cols {
                let mut bin_g = vec![0.0f64; nll_n_bins];
                let mut bin_h = vec![0.0f64; nll_n_bins];
                let col_offset = col * binned.n_rows;

                for &idx in samples {
                    let orig_bin = binned.bin_indices[col_offset + idx as usize];
                    if orig_bin == MISSING_BIN {
                        continue;
                    }
                    let coarse = ((orig_bin as usize * nll_n_bins) >> 8).min(nll_n_bins - 1);
                    bin_g[coarse] += gradients[idx as usize];
                    bin_h[coarse] += hessians[idx as usize];
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

                let gain = 0.5 * (nll_obj - base_obj) - gamma * (n_active as f64).sqrt();
                if gain > best_gain {
                    best_gain = gain;
                    best_feature = col;
                    best_bin_g = bin_g;
                    best_bin_h = bin_h;
                }
            }

            if best_gain > 0.0 {
                let leaf_value = self.values[node];
                let mut bin_values = vec![leaf_value; nll_n_bins];
                for b in 0..nll_n_bins {
                    if best_bin_h[b] >= min_child_weight {
                        let opt_value = -best_bin_g[b] / (best_bin_h[b] + lambda_reg);
                        if smooth > 0.0 {
                            bin_values[b] = (best_bin_h[b] * opt_value + smooth * leaf_value)
                                / (best_bin_h[b] + smooth);
                        } else {
                            bin_values[b] = opt_value;
                        }
                    }
                }
                self.cat_lookups[node] = Some(CatLookup {
                    feature: best_feature as u32,
                    feature2: u32::MAX,
                    feature3: u32::MAX,
                    bin_values,
                    default_value: leaf_value,
                    is_numeric: true,
                    n_coarse_bins: nll_n_bins,
                    pair_stride: 0,
                    triple_stride: 0,
                });
            }
        }
    }

    /// Refit NLL bin values using new data (for honest estimation).
    /// Keeps the feature selection but recomputes values.
    pub fn refit_numeric_lookups(
        &mut self,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        row_indices: &[u32],
        lambda_reg: f64,
        smooth: f64,
        min_child_weight: f64,
    ) {
        let n_nodes = self.split_features.len();

        let mut leaf_samples: Vec<Vec<u32>> = vec![Vec::new(); n_nodes];
        for &idx in row_indices {
            let leaf = self.route_to_leaf(binned, idx as usize);
            leaf_samples[leaf].push(idx);
        }

        for node in 0..n_nodes {
            let is_nll = self.cat_lookups[node]
                .as_ref()
                .map_or(false, |c| c.is_numeric);
            if !is_nll {
                continue;
            }

            let samples = &leaf_samples[node];
            let cll = self.cat_lookups[node].as_ref().unwrap();
            let feat = cll.feature as usize;
            let n_bins = cll.n_coarse_bins;

            let mut g_total = 0.0f64;
            let mut h_total = 0.0f64;
            for &idx in samples {
                g_total += gradients[idx as usize];
                h_total += hessians[idx as usize];
            }
            let nc_leaf = samples.len() as f64;
            let leaf_value = if h_total > 0.0 {
                -g_total / (h_total + lambda_reg + lambda_reg / nc_leaf.max(1.0).sqrt())
            } else {
                self.values[node]
            };
            self.values[node] = leaf_value;

            let col_offset = feat * binned.n_rows;
            let mut bin_g = vec![0.0f64; n_bins];
            let mut bin_h = vec![0.0f64; n_bins];
            for &idx in samples {
                let orig_bin = binned.bin_indices[col_offset + idx as usize];
                if orig_bin == MISSING_BIN {
                    continue;
                }
                let coarse = ((orig_bin as usize * n_bins) >> 8).min(n_bins - 1);
                bin_g[coarse] += gradients[idx as usize];
                bin_h[coarse] += hessians[idx as usize];
            }

            let mut bin_values = vec![leaf_value; n_bins];
            for b in 0..n_bins {
                if bin_h[b] >= min_child_weight {
                    let opt_value = -bin_g[b] / (bin_h[b] + lambda_reg);
                    if smooth > 0.0 {
                        bin_values[b] =
                            (bin_h[b] * opt_value + smooth * leaf_value) / (bin_h[b] + smooth);
                    } else {
                        bin_values[b] = opt_value;
                    }
                }
            }

            self.cat_lookups[node] = Some(CatLookup {
                feature: feat as u32,
                feature2: u32::MAX,
                feature3: u32::MAX,
                bin_values,
                default_value: leaf_value,
                is_numeric: true,
                n_coarse_bins: n_bins,
                pair_stride: 0,
                triple_stride: 0,
            });
        }
    }

    /// Scale all CLL values by a factor (used for sub_scale, lr_factor, lr_decay).
    pub fn scale_ramp_slopes(&mut self, factor: f64) {
        for v in self.ramp_slopes.iter_mut() {
            *v *= factor;
        }
        for v in self.leaf_pair_slopes.iter_mut() {
            *v *= factor;
        }
        for v in self.quad_slopes.iter_mut() {
            *v *= factor;
        }
    }

    pub fn scale_cat_lookups(&mut self, factor: f64) {
        for cll in self.cat_lookups.iter_mut() {
            if let Some(ref mut lookup) = cll {
                for v in lookup.bin_values.iter_mut() {
                    *v *= factor;
                }
                lookup.default_value *= factor;
            }
        }
    }

    pub fn scale_output(&mut self, factor: f64) {
        for v in self.values.iter_mut() {
            *v *= factor;
        }
        self.scale_ramp_slopes(factor);
        self.scale_cat_lookups(factor);
    }

    /// Confidence-aware posterior shrinkage toward zero. Small leaves should not
    /// emit full-strength residual updates; large leaves keep most of their value.
    /// Scales leaf value, CLL bins, and per-leaf ramp/quadratic refinements.
    pub fn posterior_shrink_leaves(&mut self, count_tau: f64) {
        if count_tau <= 0.0 {
            return;
        }
        let k = self.ramp_k;
        let ni = self.quad_n_interactions;
        for node in 0..self.split_features.len() {
            if self.split_features[node] != u32::MAX {
                continue;
            }
            let count = self.node_count.get(node).copied().unwrap_or(0) as f64;
            if count <= 0.0 {
                continue;
            }
            let shrink = count / (count + count_tau);
            if shrink >= 0.999_999 {
                continue;
            }
            self.values[node] *= shrink;
            if let Some(ref mut lookup) = self.cat_lookups[node] {
                for v in lookup.bin_values.iter_mut() {
                    *v *= shrink;
                }
                lookup.default_value *= shrink;
            }
            if !self.ramp_slopes.is_empty() {
                let base = node * k;
                let end = (base + k).min(self.ramp_slopes.len());
                for v in self.ramp_slopes[base..end].iter_mut() {
                    *v *= shrink;
                }
            }
            if node < self.leaf_pair_slopes.len() {
                self.leaf_pair_slopes[node] *= shrink;
            }
            if !self.quad_slopes.is_empty() && ni > 0 {
                let qbase = node * ni;
                let qend = (qbase + ni).min(self.quad_slopes.len());
                for v in self.quad_slopes[qbase..qend].iter_mut() {
                    *v *= shrink;
                }
            }
        }
    }

    /// Hierarchical Expert Shrinkage (HES): shrink each node prediction toward its
    /// ancestor prediction using node evidence, and shrink leaf-attached experts
    /// toward the node's new base prediction by the same factor.
    ///
    /// This extends hierarchical shrinkage from constant leaves to lookup / local
    /// expert leaves. It is cheap, post-hoc, and data-type agnostic.
    pub fn hierarchical_shrink_experts(&mut self, count_tau: f64) {
        if count_tau <= 0.0 || self.values.is_empty() {
            return;
        }
        let root_value = self.values[0];
        self.hierarchical_shrink_node(0, root_value, count_tau);
    }

    fn hierarchical_shrink_node(&mut self, node: usize, parent_value: f64, count_tau: f64) {
        if node >= self.values.len() {
            return;
        }
        let count = self.node_count.get(node).copied().unwrap_or(0) as f64;
        let local_scale = if node == 0 {
            1.0
        } else if count > 0.0 {
            count / (count + count_tau)
        } else {
            0.0
        };
        let old_value = self.values[node];
        let new_value = if node == 0 {
            old_value
        } else {
            parent_value + local_scale * (old_value - parent_value)
        };
        self.values[node] = new_value;

        if self.split_features[node] == u32::MAX {
            if let Some(ref mut lookup) = self.cat_lookups[node] {
                let old_default = lookup.default_value;
                for v in lookup.bin_values.iter_mut() {
                    *v = new_value + local_scale * (*v - old_value);
                }
                lookup.default_value = new_value + local_scale * (old_default - old_value);
            }
            if !self.ramp_slopes.is_empty() && self.ramp_k > 0 {
                let base = node * self.ramp_k;
                let end = (base + self.ramp_k).min(self.ramp_slopes.len());
                for v in self.ramp_slopes[base..end].iter_mut() {
                    *v *= local_scale;
                }
            }
            if node < self.leaf_pair_slopes.len() {
                self.leaf_pair_slopes[node] *= local_scale;
            }
            if !self.quad_slopes.is_empty() && self.quad_n_interactions > 0 {
                let base = node * self.quad_n_interactions;
                let end = (base + self.quad_n_interactions).min(self.quad_slopes.len());
                for v in self.quad_slopes[base..end].iter_mut() {
                    *v *= local_scale;
                }
            }
            return;
        }

        let left = self.left_children[node] as usize;
        let right = self.right_children[node] as usize;
        self.hierarchical_shrink_node(left, new_value, count_tau);
        self.hierarchical_shrink_node(right, new_value, count_tau);
    }

    /// Try to split existing leaves using current gradients. Returns number of splits made.
    /// Called post-refinement to surgically add depth where residual variance is high.
    pub fn try_split_leaves(
        &mut self,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        row_indices: &[u32],
        lambda_reg: f64,
        gamma: f64,
        min_child_weight: f64,
        cat_smooth: f64,
    ) -> usize {
        let n_nodes = self.split_features.len();

        // Route all samples to their current leaves
        let leaf_assignments: Vec<usize> = row_indices
            .iter()
            .map(|&i| self.route_to_leaf(binned, i as usize))
            .collect();

        // Group samples by leaf
        let mut leaf_samples: Vec<Vec<u32>> = vec![Vec::new(); n_nodes];
        for (pos, &idx) in row_indices.iter().enumerate() {
            leaf_samples[leaf_assignments[pos]].push(idx);
        }

        self.try_split_leaves_inner(
            binned,
            gradients,
            hessians,
            &leaf_samples,
            lambda_reg,
            gamma,
            min_child_weight,
            cat_smooth,
        )
    }

    /// Try to split leaves using pre-computed leaf sample grouping (avoids routing step).
    pub fn try_split_leaves_precomputed(
        &mut self,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        leaf_samples: &[Vec<u32>],
        lambda_reg: f64,
        gamma: f64,
        min_child_weight: f64,
        cat_smooth: f64,
    ) -> usize {
        self.try_split_leaves_inner(
            binned,
            gradients,
            hessians,
            leaf_samples,
            lambda_reg,
            gamma,
            min_child_weight,
            cat_smooth,
        )
    }

    fn try_split_leaves_inner(
        &mut self,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        leaf_samples: &[Vec<u32>],
        lambda_reg: f64,
        gamma: f64,
        min_child_weight: f64,
        cat_smooth: f64,
    ) -> usize {
        let n_nodes = self.split_features.len();

        // Collect leaf nodes with enough samples (direct iteration, no cloning)
        let leaf_nodes: Vec<usize> = (0..n_nodes)
            .filter(|&i| {
                self.split_features[i] == u32::MAX
                    && i < leaf_samples.len()
                    && leaf_samples[i].len() >= 2
            })
            .collect();

        let active_features: Vec<usize> = (0..binned.n_features).collect();
        let max_bins = (0..binned.n_features)
            .map(|c| binned.n_bins(c))
            .max()
            .unwrap_or(1);
        let mut g_hist = vec![0.0f64; max_bins];
        let mut h_hist = vec![0.0f64; max_bins];
        let min_h = min_child_weight.max(1e-10);
        let mut n_splits = 0usize;

        for &leaf_idx in &leaf_nodes {
            let samples = &leaf_samples[leaf_idx];
            let (g_sum, h_sum) = sum_gh(gradients, hessians, samples);
            if h_sum < min_h {
                continue;
            }

            let sr = find_best_split(
                binned,
                gradients,
                hessians,
                samples,
                &active_features,
                g_sum,
                h_sum,
                lambda_reg,
                0.0,
                gamma,
                min_h,
                &mut g_hist,
                &mut h_hist,
                0.0,
                0,
                cat_smooth,
                &[], // no monotone constraints for leaf splits
                0.0, // no gain penalty for post-hoc leaf splits
                false,
            );
            if sr.gain <= 0.0 || !sr.gain.is_finite() {
                continue;
            }

            let left_new = self.split_features.len();
            let right_new = left_new + 1;
            self.split_features.push(u32::MAX);
            self.split_features.push(u32::MAX);
            self.split_bins.push(0);
            self.split_bins.push(0);
            self.values.push(0.0);
            self.values.push(0.0);
            self.left_children.push(0);
            self.left_children.push(0);
            self.right_children.push(0);
            self.right_children.push(0);
            self.missing_goes_left.push(true);
            self.missing_goes_left.push(true);
            self.is_oblique_split.push(false);
            self.is_oblique_split.push(false);
            self.is_cat_split.push(false);
            self.is_cat_split.push(false);
            self.cat_left_masks.push(Vec::new());
            self.cat_left_masks.push(Vec::new());
            self.oblique_features.push(u32::MAX);
            self.oblique_features.push(u32::MAX);
            self.oblique_features.push(u32::MAX);
            self.oblique_features.push(u32::MAX);
            self.oblique_weights.push(0.0);
            self.oblique_weights.push(0.0);
            self.oblique_weights.push(0.0);
            self.oblique_weights.push(0.0);
            self.oblique_thresholds.push(0.0);
            self.oblique_thresholds.push(0.0);
            self.cat_lookups.push(None);
            self.cat_lookups.push(None);
            self.node_h_sum.push(0.0);
            self.node_h_sum.push(0.0);
            self.node_count.push(0);
            self.node_count.push(0);
            // GGFP v5.0 cat-pair fields (refit creates raw splits only)
            self.cat_pair_feat2.push(u32::MAX);
            self.cat_pair_feat2.push(u32::MAX);
            self.cat_pair_bucket_map_a.push(Vec::new());
            self.cat_pair_bucket_map_a.push(Vec::new());
            self.cat_pair_bucket_map_b.push(Vec::new());
            self.cat_pair_bucket_map_b.push(Vec::new());
            self.cat_pair_cell_mask.push(0);
            self.cat_pair_cell_mask.push(0);
            self.cat_pair_k_buckets.push(0);
            self.cat_pair_k_buckets.push(0);

            // Clear CLL from the old leaf (it's becoming an internal node)
            self.cat_lookups[leaf_idx] = None;

            self.split_features[leaf_idx] = sr.feat as u32;
            self.split_bins[leaf_idx] = sr.bin as u16;
            self.left_children[leaf_idx] = left_new as u32;
            self.right_children[leaf_idx] = right_new as u32;
            self.missing_goes_left[leaf_idx] = sr.missing_left;
            self.is_oblique_split[leaf_idx] = sr.is_oblique;
            self.is_cat_split[leaf_idx] = sr.is_cat;
            // Refit installs only raw splits — clear any pre-existing cat-pair state at this node
            if leaf_idx < self.cat_pair_feat2.len() {
                self.cat_pair_feat2[leaf_idx] = u32::MAX;
                self.cat_pair_bucket_map_a[leaf_idx] = Vec::new();
                self.cat_pair_bucket_map_b[leaf_idx] = Vec::new();
                self.cat_pair_cell_mask[leaf_idx] = 0;
                self.cat_pair_k_buckets[leaf_idx] = 0;
            }
            let ob = leaf_idx * 2;
            self.oblique_features[ob] = sr.oblique_feats[0];
            self.oblique_features[ob + 1] = sr.oblique_feats[1];
            self.oblique_weights[ob] = sr.oblique_weights[0];
            self.oblique_weights[ob + 1] = sr.oblique_weights[1];
            self.oblique_thresholds[leaf_idx] = sr.oblique_threshold;

            let col_bins = binned.col_bins(sr.feat);
            let mut g_left = 0.0f64;
            let mut h_left = 0.0f64;
            let mut g_right = 0.0f64;
            let mut h_right = 0.0f64;
            let mut cnt_left = 0u32;
            let mut cnt_right = 0u32;

            for &idx in samples {
                let bin = col_bins[idx as usize];
                let goes_left = if sr.is_oblique {
                    split_goes_left_binned(&sr, binned, idx as usize)
                } else if bin == MISSING_BIN {
                    sr.missing_left
                } else if sr.is_cat {
                    bitmask_test(&sr.cat_mask, bin as usize)
                } else {
                    bin <= sr.bin as u16
                };
                if goes_left {
                    g_left += gradients[idx as usize];
                    h_left += hessians[idx as usize];
                    cnt_left += 1;
                } else {
                    g_right += gradients[idx as usize];
                    h_right += hessians[idx as usize];
                    cnt_right += 1;
                }
            }

            self.cat_left_masks[leaf_idx] = sr.cat_mask;
            self.node_h_sum[leaf_idx] = h_sum;
            self.node_count[leaf_idx] = samples.len() as u32;
            self.node_h_sum[left_new] = h_left;
            self.node_h_sum[right_new] = h_right;
            self.node_count[left_new] = cnt_left;
            self.node_count[right_new] = cnt_right;

            let nl = cnt_left as f64;
            let nr = cnt_right as f64;
            self.values[left_new] =
                -g_left / (h_left + lambda_reg + lambda_reg / nl.max(1.0).sqrt());
            self.values[right_new] =
                -g_right / (h_right + lambda_reg + lambda_reg / nr.max(1.0).sqrt());
            n_splits += 1;
        }

        n_splits
    }

    /// Multi-level leaf splitting: try to split leaves up to `max_depth_add` times recursively.
    /// With max_depth_add=1, equivalent to single-pass splitting.
    /// With max_depth_add=2, after splitting a leaf, also tries to split the new children.
    pub fn try_split_leaves_multi(
        &mut self,
        binned: &BinnedData,
        gradients: &[f64],
        hessians: &[f64],
        leaf_samples: &[Vec<u32>],
        lambda_reg: f64,
        gamma: f64,
        min_child_weight: f64,
        max_depth_add: usize,
        cat_smooth: f64,
    ) -> usize {
        let n_nodes = self.split_features.len();

        let active_features: Vec<usize> = (0..binned.n_features).collect();
        let max_bins = (0..binned.n_features)
            .map(|c| binned.n_bins(c))
            .max()
            .unwrap_or(1);
        let mut g_hist = vec![0.0f64; max_bins];
        let mut h_hist = vec![0.0f64; max_bins];
        let min_h = min_child_weight.max(1e-10);
        let mut n_splits = 0usize;

        // Work queue: (node_idx, samples, depth_remaining)
        // Start with all existing leaves that have enough samples
        let mut queue: Vec<(usize, Vec<u32>, usize)> = Vec::new();
        for i in 0..n_nodes {
            if self.split_features[i] == u32::MAX
                && i < leaf_samples.len()
                && leaf_samples[i].len() >= 2
            {
                queue.push((i, leaf_samples[i].clone(), max_depth_add));
            }
        }

        while let Some((leaf_idx, samples, depth_remaining)) = queue.pop() {
            if depth_remaining == 0 || samples.len() < 2 {
                continue;
            }

            let (g_sum, h_sum) = sum_gh(gradients, hessians, &samples);
            if h_sum < min_h {
                continue;
            }

            let sr = find_best_split(
                binned,
                gradients,
                hessians,
                &samples,
                &active_features,
                g_sum,
                h_sum,
                lambda_reg,
                0.0,
                gamma,
                min_h,
                &mut g_hist,
                &mut h_hist,
                0.0,
                0,
                cat_smooth,
                &[], // no monotone constraints for leaf splits
                0.0, // no gain penalty for post-hoc leaf splits
                false,
            );

            if sr.gain <= 0.0 || !sr.gain.is_finite() {
                continue;
            }

            // Create two new leaf nodes
            let left_new = self.split_features.len();
            let right_new = left_new + 1;

            self.split_features.push(u32::MAX);
            self.split_features.push(u32::MAX);
            self.split_bins.push(0);
            self.split_bins.push(0);
            self.values.push(0.0);
            self.values.push(0.0);
            self.left_children.push(0);
            self.left_children.push(0);
            self.right_children.push(0);
            self.right_children.push(0);
            self.missing_goes_left.push(true);
            self.missing_goes_left.push(true);
            self.is_oblique_split.push(false);
            self.is_oblique_split.push(false);
            self.is_cat_split.push(false);
            self.is_cat_split.push(false);
            self.cat_left_masks.push(Vec::new());
            self.cat_left_masks.push(Vec::new());
            self.oblique_features.push(u32::MAX);
            self.oblique_features.push(u32::MAX);
            self.oblique_features.push(u32::MAX);
            self.oblique_features.push(u32::MAX);
            self.oblique_weights.push(0.0);
            self.oblique_weights.push(0.0);
            self.oblique_weights.push(0.0);
            self.oblique_weights.push(0.0);
            self.oblique_thresholds.push(0.0);
            self.oblique_thresholds.push(0.0);
            self.cat_lookups.push(None);
            self.cat_lookups.push(None);
            self.node_h_sum.push(0.0);
            self.node_h_sum.push(0.0);
            self.node_count.push(0);
            self.node_count.push(0);
            // GGFP v5.0 cat-pair fields (refit creates raw splits only)
            self.cat_pair_feat2.push(u32::MAX);
            self.cat_pair_feat2.push(u32::MAX);
            self.cat_pair_bucket_map_a.push(Vec::new());
            self.cat_pair_bucket_map_a.push(Vec::new());
            self.cat_pair_bucket_map_b.push(Vec::new());
            self.cat_pair_bucket_map_b.push(Vec::new());
            self.cat_pair_cell_mask.push(0);
            self.cat_pair_cell_mask.push(0);
            self.cat_pair_k_buckets.push(0);
            self.cat_pair_k_buckets.push(0);
            // Clear CLL from the old leaf (it's becoming an internal node)
            self.cat_lookups[leaf_idx] = None;

            // Convert leaf to split node
            self.split_features[leaf_idx] = sr.feat as u32;
            self.split_bins[leaf_idx] = sr.bin as u16;
            self.left_children[leaf_idx] = left_new as u32;
            self.right_children[leaf_idx] = right_new as u32;
            self.missing_goes_left[leaf_idx] = sr.missing_left;
            self.is_oblique_split[leaf_idx] = sr.is_oblique;
            self.is_cat_split[leaf_idx] = sr.is_cat;
            // Refit installs only raw splits — clear any pre-existing cat-pair state at this node
            if leaf_idx < self.cat_pair_feat2.len() {
                self.cat_pair_feat2[leaf_idx] = u32::MAX;
                self.cat_pair_bucket_map_a[leaf_idx] = Vec::new();
                self.cat_pair_bucket_map_b[leaf_idx] = Vec::new();
                self.cat_pair_cell_mask[leaf_idx] = 0;
                self.cat_pair_k_buckets[leaf_idx] = 0;
            }
            let ob = leaf_idx * 2;
            self.oblique_features[ob] = sr.oblique_feats[0];
            self.oblique_features[ob + 1] = sr.oblique_feats[1];
            self.oblique_weights[ob] = sr.oblique_weights[0];
            self.oblique_weights[ob + 1] = sr.oblique_weights[1];
            self.oblique_thresholds[leaf_idx] = sr.oblique_threshold;

            // Route samples to children and compute leaf values
            let col_bins = binned.col_bins(sr.feat);
            let mut g_left = 0.0f64;
            let mut h_left = 0.0f64;
            let mut g_right = 0.0f64;
            let mut h_right = 0.0f64;
            let mut left_samples = Vec::new();
            let mut right_samples = Vec::new();

            for &idx in &samples {
                let bin = col_bins[idx as usize];
                let goes_left = if sr.is_oblique {
                    split_goes_left_binned(&sr, binned, idx as usize)
                } else if bin == MISSING_BIN {
                    sr.missing_left
                } else if sr.is_cat {
                    bitmask_test(&sr.cat_mask, bin as usize)
                } else {
                    bin <= sr.bin as u16
                };
                if goes_left {
                    g_left += gradients[idx as usize];
                    h_left += hessians[idx as usize];
                    left_samples.push(idx);
                } else {
                    g_right += gradients[idx as usize];
                    h_right += hessians[idx as usize];
                    right_samples.push(idx);
                }
            }

            // Move cat_mask after use in the loop above
            self.cat_left_masks[leaf_idx] = sr.cat_mask;
            self.node_h_sum[leaf_idx] = h_sum;
            self.node_count[leaf_idx] = samples.len() as u32;
            self.node_h_sum[left_new] = h_left;
            self.node_h_sum[right_new] = h_right;
            self.node_count[left_new] = left_samples.len() as u32;
            self.node_count[right_new] = right_samples.len() as u32;

            let nl = left_samples.len() as f64;
            let nr = right_samples.len() as f64;
            self.values[left_new] =
                -g_left / (h_left + lambda_reg + lambda_reg / nl.max(1.0).sqrt());
            self.values[right_new] =
                -g_right / (h_right + lambda_reg + lambda_reg / nr.max(1.0).sqrt());
            n_splits += 1;

            if depth_remaining > 1 {
                if left_samples.len() >= 2 {
                    queue.push((left_new, left_samples, depth_remaining - 1));
                }
                if right_samples.len() >= 2 {
                    queue.push((right_new, right_samples, depth_remaining - 1));
                }
            }
        }

        n_splits
    }
}
