use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;

mod algorithms;
mod build;
mod cat_pair;
mod predict;
mod refit;
pub mod virtual_features;

pub use algorithms::CatPairConfig;
pub use virtual_features::{
    extend_eval_bins_with_virtuals, premine_and_register, LtsoPremineConfig, VirtualFeatureDef,
};

/// Sentinel bin value for missing (NaN) data.
pub const MISSING_BIN: u16 = u16::MAX;
/// Sentinel feature id for dynamic self-score splits. These nodes route on the
/// model margin immediately before the current tree is applied.
pub const SELF_SCORE_FEATURE: u32 = u32::MAX - 1;

/// Bitmask for categorical splits: which bins go left. Dynamically sized.
pub type CatBitmask = Vec<u64>;

#[inline(always)]
pub(super) fn bitmask_set(mask: &mut CatBitmask, bin: usize) {
    let word = bin / 64;
    if word >= mask.len() {
        mask.resize(word + 1, 0);
    }
    mask[word] |= 1u64 << (bin % 64);
}

#[inline(always)]
pub fn bitmask_test(mask: &[u64], bin: usize) -> bool {
    let word = bin / 64;
    word < mask.len() && (mask[word] >> (bin % 64)) & 1 != 0
}

/// Pre-binned feature data shared across all trees and nodes (zero-copy).
/// Stored COLUMN-MAJOR: bin_indices[col * n_rows + row] for cache-friendly histogram building.
#[derive(Clone, Serialize, Deserialize)]
pub struct BinnedData {
    pub bin_indices: Vec<u16>,
    pub bin_edges: Vec<Vec<f64>>,
    pub n_features: usize,
    pub n_rows: usize,
    pub is_categorical: Vec<bool>,
    /// True when a feature contains at least one missing value in the training
    /// matrix. Split search uses this to skip per-node missing scans on dense
    /// features, matching XGBoost's sparsity-aware fast path.
    pub feature_has_missing: Vec<bool>,
    /// Number of non-missing rows per feature in the training matrix. This is
    /// cheap metadata for sparse-aware gates and diagnostics.
    pub feature_non_missing_count: Vec<u32>,
    /// XGBoost-style sparse sidecar blocks. For feature `f`, non-missing rows
    /// live in `non_missing_row_indices[offsets[f]..offsets[f+1]]` and their
    /// corresponding bins live in `non_missing_bin_values` at the same range.
    /// Dense column-major bins remain the canonical representation for
    /// prediction and dense features; histograms can choose either path.
    pub non_missing_offsets: Vec<usize>,
    pub non_missing_row_indices: Vec<u32>,
    pub non_missing_bin_values: Vec<u16>,
    /// Hash-based CLL bins for high-cardinality categoricals (>256 unique).
    /// Layout: cll_hash_bins[col * n_rows + row]. Only populated for cols where
    /// cll_is_categorical[col] && !is_categorical[col].
    pub cll_hash_bins: Vec<u16>,
    /// True for any feature originally categorical (used for CLL evaluation).
    /// Superset of is_categorical — includes >256 features that fell back to numeric splits.
    pub cll_is_categorical: Vec<bool>,
    /// Number of CLL bins per feature (256 for hashed high-cardinality, same as n_bins for native).
    pub cll_n_bins: Vec<usize>,
    /// GGFP v6 (LTSO Phase 1) — count of original raw features at construction.
    /// Features with id >= n_raw_features are derived (auto_interactions,
    /// sumdiff, ordered_ctr, etc.). For legacy / unfit models, equals n_features.
    pub n_raw_features: usize,
    /// GGFP v6 — first feature id that points into `virtual_defs[0]`.
    /// Set once when register_virtual_feature is first called. Decoupled
    /// from n_raw_features so later add_* calls don't shift the mapping.
    /// usize::MAX = no virtuals registered.
    pub virtual_first_id: usize,
    /// GGFP v6 — operator definitions for virtual features. Indexed by
    /// (feature_id - virtual_first_id).
    pub virtual_defs: Vec<VirtualFeatureDef>,
    /// Evidence-corrected split gain. Positive values turn optimistic in-node
    /// gain into a lower-confidence score that accounts for child support and
    /// split search width.
    pub split_pessimism: f64,
}

impl BinnedData {
    pub fn new(
        data: &[f64],
        n_rows: usize,
        n_features: usize,
        n_bins: usize,
        cat_features: &[bool],
        max_cat_bins: usize,
    ) -> Self {
        let per_feature: Vec<(Vec<f64>, Vec<u16>, bool, bool, u32)> = (0..n_features)
            .into_par_iter()
            .map(|col| {
                let mut col_vals: Vec<f64> = Vec::with_capacity(n_rows);
                for row in 0..n_rows {
                    col_vals.push(data[row * n_features + col]);
                }

                if cat_features.len() > col && cat_features[col] {
                    // Categorical: one bin per unique value (label-encoded integers)
                    let mut sorted_vals: Vec<f64> =
                        col_vals.iter().copied().filter(|v| !v.is_nan()).collect();
                    sorted_vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                    sorted_vals.dedup();

                    let cat_limit = if max_cat_bins == 0 {
                        MISSING_BIN as usize
                    } else {
                        max_cat_bins
                    };
                    if sorted_vals.len() <= cat_limit && (sorted_vals.len() as u16) < MISSING_BIN {
                        // One bin per unique value (native categorical)
                        let edges = sorted_vals.clone();
                        let actual_bins = edges.len().max(1);
                        let mut col_bins = vec![0u16; n_rows];
                        for row in 0..n_rows {
                            let v = col_vals[row];
                            if v.is_nan() {
                                col_bins[row] = MISSING_BIN;
                            } else {
                                let mut lo = 0usize;
                                let mut hi = actual_bins;
                                while lo < hi {
                                    let mid = lo + (hi - lo) / 2;
                                    if edges[mid] < v {
                                        lo = mid + 1;
                                    } else {
                                        hi = mid;
                                    }
                                }
                                col_bins[row] = lo.min(actual_bins - 1) as u16;
                            }
                        }
                        let has_missing = col_bins.iter().any(|&b| b == MISSING_BIN);
                        let non_missing = (n_rows
                            - col_bins.iter().filter(|&&b| b == MISSING_BIN).count())
                            as u32;
                        (edges, col_bins, true, has_missing, non_missing)
                    } else {
                        // High cardinality: frequency-based binning
                        // Top K values (by frequency) get individual bins, rest → "other" bin
                        let k = cat_limit; // number of individual bins
                                           // Count frequency of each unique value
                        let mut freq: HashMap<i64, usize> = HashMap::new();
                        for &v in &col_vals {
                            if !v.is_nan() {
                                *freq.entry(v as i64).or_insert(0) += 1;
                            }
                        }
                        let mut freq_vec: Vec<(i64, usize)> = freq.into_iter().collect();
                        freq_vec.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                        // Top-K values sorted by value (for binary search)
                        let mut top_vals: Vec<f64> =
                            freq_vec.iter().take(k).map(|&(v, _)| v as f64).collect();
                        top_vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                        let edges = top_vals;
                        let actual_bins = edges.len();
                        // "other" bin = actual_bins (one past the last individual bin)
                        let other_bin = actual_bins as u16;
                        let mut col_bins = vec![0u16; n_rows];
                        for row in 0..n_rows {
                            let v = col_vals[row];
                            if v.is_nan() {
                                col_bins[row] = MISSING_BIN;
                            } else {
                                // Binary search for exact match
                                let mut lo = 0usize;
                                let mut hi = actual_bins;
                                while lo < hi {
                                    let mid = lo + (hi - lo) / 2;
                                    if edges[mid] < v {
                                        lo = mid + 1;
                                    } else {
                                        hi = mid;
                                    }
                                }
                                col_bins[row] = if lo < actual_bins && edges[lo] == v {
                                    lo as u16
                                } else {
                                    other_bin
                                };
                            }
                        }
                        let has_missing = col_bins.iter().any(|&b| b == MISSING_BIN);
                        let non_missing = (n_rows
                            - col_bins.iter().filter(|&&b| b == MISSING_BIN).count())
                            as u32;
                        (edges, col_bins, true, has_missing, non_missing)
                    }
                } else {
                    // Numeric: quantile binning (original logic)
                    let mut sorted_vals: Vec<f64> =
                        col_vals.iter().copied().filter(|v| !v.is_nan()).collect();
                    sorted_vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

                    let n_valid = sorted_vals.len();
                    let mut edges: Vec<f64> = Vec::with_capacity(n_bins);

                    if n_valid == 0 {
                        edges.push(0.0);
                    } else if sorted_vals.first() == sorted_vals.last() {
                        edges.push(sorted_vals[0]);
                    } else {
                        for b in 1..n_bins {
                            let quantile_idx = (b as f64 / n_bins as f64 * n_valid as f64) as usize;
                            let quantile_idx = quantile_idx.min(n_valid - 1);
                            let edge = sorted_vals[quantile_idx];
                            if edges.is_empty() || edge > *edges.last().unwrap() {
                                edges.push(edge);
                            }
                        }
                        let last = *sorted_vals.last().unwrap();
                        if edges.is_empty() || last > *edges.last().unwrap() {
                            edges.push(last);
                        }
                    }

                    let actual_bins = edges.len();
                    let mut col_bins = vec![0u16; n_rows];
                    for row in 0..n_rows {
                        let v = col_vals[row];
                        if v.is_nan() {
                            col_bins[row] = MISSING_BIN;
                        } else {
                            let mut lo = 0usize;
                            let mut hi = actual_bins;
                            while lo < hi {
                                let mid = lo + (hi - lo) / 2;
                                if edges[mid] < v {
                                    lo = mid + 1;
                                } else {
                                    hi = mid;
                                }
                            }
                            col_bins[row] = lo.min(actual_bins - 1) as u16;
                        }
                    }
                    let has_missing = col_bins.iter().any(|&b| b == MISSING_BIN);
                    let non_missing =
                        (n_rows - col_bins.iter().filter(|&&b| b == MISSING_BIN).count()) as u32;
                    (edges, col_bins, false, has_missing, non_missing)
                }
            })
            .collect();

        let mut bin_indices = vec![0u16; n_rows * n_features];
        let mut bin_edges = Vec::with_capacity(n_features);
        let mut is_categorical = Vec::with_capacity(n_features);
        let mut feature_has_missing = Vec::with_capacity(n_features);
        let mut feature_non_missing_count = Vec::with_capacity(n_features);
        let mut non_missing_offsets = Vec::with_capacity(n_features + 1);
        let mut non_missing_row_indices = Vec::new();
        let mut non_missing_bin_values = Vec::new();
        non_missing_offsets.push(0);
        for (col, (edges, col_bins, is_cat, has_missing, non_missing)) in
            per_feature.into_iter().enumerate()
        {
            let offset = col * n_rows;
            bin_indices[offset..offset + n_rows].copy_from_slice(&col_bins);
            bin_edges.push(edges);
            is_categorical.push(is_cat);
            feature_has_missing.push(has_missing);
            feature_non_missing_count.push(non_missing);
            if has_missing {
                for (row, &bin) in col_bins.iter().enumerate() {
                    if bin != MISSING_BIN {
                        non_missing_row_indices.push(row as u32);
                        non_missing_bin_values.push(bin);
                    }
                }
            }
            non_missing_offsets.push(non_missing_row_indices.len());
        }

        // Build CLL hash bins for high-cardinality categoricals (>256 unique → is_categorical=false)
        let mut cll_is_categorical = is_categorical.clone();
        let mut cll_n_bins = vec![0usize; n_features];
        let mut cll_hash_bins = vec![0u16; n_rows * n_features];
        for col in 0..n_features {
            if is_categorical[col] {
                // Native categorical: CLL uses same bins as tree splits
                cll_n_bins[col] = bin_edges[col].len() + 1;
                let offset = col * n_rows;
                cll_hash_bins[offset..offset + n_rows]
                    .copy_from_slice(&bin_indices[offset..offset + n_rows]);
            } else if col < cat_features.len() && cat_features[col] {
                // High-cardinality categorical that fell back to numeric: hash to 256 bins for CLL
                cll_is_categorical[col] = true;
                cll_n_bins[col] = 256;
                let offset = col * n_rows;
                for row in 0..n_rows {
                    let v = data[row * n_features + col];
                    if v.is_nan() {
                        cll_hash_bins[offset + row] = MISSING_BIN;
                    } else {
                        // Hash the integer category value to 0..255
                        let iv = v as i64;
                        let h = ((iv.wrapping_mul(0x9E3779B97F4A7C15_u64 as i64)) as u64) >> 56;
                        cll_hash_bins[offset + row] = h as u16;
                    }
                }
            }
        }

        BinnedData {
            bin_indices,
            bin_edges,
            n_features,
            n_rows,
            is_categorical,
            feature_has_missing,
            feature_non_missing_count,
            non_missing_offsets,
            non_missing_row_indices,
            non_missing_bin_values,
            cll_hash_bins,
            cll_is_categorical,
            cll_n_bins,
            n_raw_features: n_features,
            virtual_first_id: usize::MAX,
            virtual_defs: Vec::new(),
            split_pessimism: 0.0,
        }
    }

    /// Bin new data using pre-computed edges (for eval sets).
    pub fn bin_with_edges(
        data: &[f64],
        n_rows: usize,
        n_features: usize,
        bin_edges: &[Vec<f64>],
        is_categorical: &[bool],
    ) -> Vec<u16> {
        let per_feature: Vec<Vec<u16>> = (0..n_features)
            .into_par_iter()
            .map(|col| {
                let edges = &bin_edges[col];
                let actual_bins = edges.len();
                let is_cat = col < is_categorical.len() && is_categorical[col];
                let mut col_bins = vec![0u16; n_rows];
                for row in 0..n_rows {
                    let v = data[row * n_features + col];
                    if v.is_nan() || actual_bins == 0 {
                        col_bins[row] = MISSING_BIN;
                    } else {
                        let mut lo = 0usize;
                        let mut hi = actual_bins;
                        while lo < hi {
                            let mid = lo + (hi - lo) / 2;
                            if edges[mid] < v {
                                lo = mid + 1;
                            } else {
                                hi = mid;
                            }
                        }
                        col_bins[row] = if is_cat {
                            if lo < actual_bins && edges[lo] == v {
                                lo as u16
                            } else {
                                MISSING_BIN
                            }
                        } else {
                            lo.min(actual_bins - 1) as u16
                        };
                    }
                }
                col_bins
            })
            .collect();

        let mut bin_indices = vec![0u16; n_rows * n_features];
        for (col, col_bins) in per_feature.into_iter().enumerate() {
            let offset = col * n_rows;
            bin_indices[offset..offset + n_rows].copy_from_slice(&col_bins);
        }
        bin_indices
    }

    /// Bin new data using pre-computed edges into row-major layout.
    ///
    /// Training keeps bins column-major for histogram construction. Prediction
    /// routes rows through trees, so row-major bins avoid strided loads in the
    /// common plain-tree inference path.
    pub fn bin_with_edges_row_major(
        data: &[f64],
        n_rows: usize,
        n_features: usize,
        bin_edges: &[Vec<f64>],
        is_categorical: &[bool],
    ) -> Vec<u16> {
        let mut bin_indices = vec![0u16; n_rows * n_features];
        bin_indices
            .par_chunks_mut(n_features)
            .enumerate()
            .for_each(|(row, row_bins)| {
                let row_offset = row * n_features;
                for col in 0..n_features {
                    let edges = &bin_edges[col];
                    let actual_bins = edges.len();
                    let is_cat = col < is_categorical.len() && is_categorical[col];
                    let v = data[row_offset + col];
                    if v.is_nan() || actual_bins == 0 {
                        row_bins[col] = MISSING_BIN;
                    } else {
                        let mut lo = 0usize;
                        let mut hi = actual_bins;
                        while lo < hi {
                            let mid = lo + (hi - lo) / 2;
                            if edges[mid] < v {
                                lo = mid + 1;
                            } else {
                                hi = mid;
                            }
                        }
                        row_bins[col] = if is_cat {
                            if lo < actual_bins && edges[lo] == v {
                                lo as u16
                            } else {
                                MISSING_BIN
                            }
                        } else {
                            lo.min(actual_bins - 1) as u16
                        };
                    }
                }
            });
        bin_indices
    }

    pub fn strip_training_storage_for_prediction(&mut self) {
        self.n_rows = 0;
        self.bin_indices.clear();
        self.feature_has_missing.clear();
        self.feature_non_missing_count.clear();
        self.non_missing_offsets.clear();
        self.non_missing_row_indices.clear();
        self.non_missing_bin_values.clear();
        self.cll_hash_bins.clear();
    }

    #[inline(always)]
    pub fn get_bin(&self, row: usize, col: usize) -> usize {
        self.bin_indices[col * self.n_rows + row] as usize
    }

    #[inline(always)]
    pub fn get_bin_u16(&self, row: usize, col: usize) -> u16 {
        self.bin_indices[col * self.n_rows + row]
    }

    #[inline(always)]
    pub fn col_bins(&self, col: usize) -> &[u16] {
        let offset = col * self.n_rows;
        &self.bin_indices[offset..offset + self.n_rows]
    }

    #[inline(always)]
    pub fn non_missing_block(&self, col: usize) -> (&[u32], &[u16]) {
        if col + 1 >= self.non_missing_offsets.len() {
            return (&[], &[]);
        }
        let start = self.non_missing_offsets[col];
        let end = self.non_missing_offsets[col + 1];
        (
            &self.non_missing_row_indices[start..end],
            &self.non_missing_bin_values[start..end],
        )
    }

    fn append_non_missing_block(&mut self, col_bins: &[u16], has_missing: bool) {
        if self.non_missing_offsets.is_empty() {
            self.rebuild_non_missing_blocks();
            return;
        }
        debug_assert_eq!(self.non_missing_offsets.len(), self.n_features + 1);
        if has_missing {
            for (row, &bin) in col_bins.iter().enumerate() {
                if bin != MISSING_BIN {
                    self.non_missing_row_indices.push(row as u32);
                    self.non_missing_bin_values.push(bin);
                }
            }
        }
        self.non_missing_offsets
            .push(self.non_missing_row_indices.len());
    }

    fn rebuild_non_missing_blocks(&mut self) {
        self.non_missing_offsets.clear();
        self.non_missing_row_indices.clear();
        self.non_missing_bin_values.clear();
        self.non_missing_offsets.reserve(self.n_features + 1);
        self.non_missing_offsets.push(0);
        for col in 0..self.n_features {
            let offset = col * self.n_rows;
            let col_bins = &self.bin_indices[offset..offset + self.n_rows];
            if self.feature_has_missing.get(col).copied().unwrap_or(false) {
                for (row, &bin) in col_bins.iter().enumerate() {
                    if bin != MISSING_BIN {
                        self.non_missing_row_indices.push(row as u32);
                        self.non_missing_bin_values.push(bin);
                    }
                }
            }
            self.non_missing_offsets
                .push(self.non_missing_row_indices.len());
        }
    }

    /// Number of bins for a feature, including the "other" bin for categoricals.
    #[inline(always)]
    pub fn n_bins(&self, col: usize) -> usize {
        if col < self.is_categorical.len() && self.is_categorical[col] {
            self.bin_edges[col].len() + 1
        } else {
            self.bin_edges[col].len()
        }
    }

    /// Append OTS-encoded numeric features to the binned data.
    /// `ots_values` is n_ots_cols x n_rows. Each column is quantile-binned and appended.
    pub fn add_ots_features(&mut self, ots_values: &[Vec<f64>], n_bins: usize) {
        for col_vals in ots_values {
            assert_eq!(col_vals.len(), self.n_rows);
            let mut sorted: Vec<f64> = col_vals.iter().copied().filter(|v| !v.is_nan()).collect();
            sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

            let n_valid = sorted.len();
            let mut edges: Vec<f64> = Vec::with_capacity(n_bins);
            if n_valid == 0 {
                edges.push(0.0);
            } else if sorted.first() == sorted.last() {
                edges.push(sorted[0]);
            } else {
                for b in 1..n_bins {
                    let qi = (b as f64 / n_bins as f64 * n_valid as f64) as usize;
                    let qi = qi.min(n_valid - 1);
                    let edge = sorted[qi];
                    if edges.is_empty() || edge > *edges.last().unwrap() {
                        edges.push(edge);
                    }
                }
                let last = *sorted.last().unwrap();
                if edges.is_empty() || last > *edges.last().unwrap() {
                    edges.push(last);
                }
            }

            let actual_bins = edges.len();
            let mut col_bins = vec![0u16; self.n_rows];
            for row in 0..self.n_rows {
                let v = col_vals[row];
                if v.is_nan() {
                    col_bins[row] = MISSING_BIN;
                } else {
                    let mut lo = 0usize;
                    let mut hi = actual_bins;
                    while lo < hi {
                        let mid = lo + (hi - lo) / 2;
                        if edges[mid] < v {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    col_bins[row] = lo.min(actual_bins - 1) as u16;
                }
            }

            self.bin_indices.extend_from_slice(&col_bins);
            self.bin_edges.push(edges);
            self.is_categorical.push(false);
            let has_missing = col_bins.iter().any(|&b| b == MISSING_BIN);
            self.feature_has_missing.push(has_missing);
            self.feature_non_missing_count.push(
                (self.n_rows - col_bins.iter().filter(|&&b| b == MISSING_BIN).count()) as u32,
            );
            self.append_non_missing_block(&col_bins, has_missing);
            self.n_features += 1;
            self.n_raw_features += 1;
        }
    }

    /// Append or replace a numeric feature with quantile bins from per-row values.
    /// Used for dynamic learner-state features during training.
    pub fn set_numeric_feature_from_values(
        &mut self,
        col_idx: Option<usize>,
        values: &[f64],
        n_bins: usize,
    ) -> usize {
        assert_eq!(values.len(), self.n_rows);
        let mut sorted: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let n_valid = sorted.len();
        let mut edges: Vec<f64> = Vec::with_capacity(n_bins);
        if n_valid == 0 {
            edges.push(0.0);
        } else if sorted.first() == sorted.last() {
            edges.push(sorted[0]);
        } else {
            for b in 1..n_bins {
                let qi = (b as f64 / n_bins as f64 * n_valid as f64) as usize;
                let qi = qi.min(n_valid - 1);
                let edge = sorted[qi];
                if edges.is_empty() || edge > *edges.last().unwrap() {
                    edges.push(edge);
                }
            }
            let last = *sorted.last().unwrap();
            if edges.is_empty() || last > *edges.last().unwrap() {
                edges.push(last);
            }
        }

        let actual_bins = edges.len();
        let mut col_bins = vec![0u16; self.n_rows];
        for row in 0..self.n_rows {
            let v = values[row];
            if v.is_nan() {
                col_bins[row] = MISSING_BIN;
            } else {
                let mut lo = 0usize;
                let mut hi = actual_bins;
                while lo < hi {
                    let mid = lo + (hi - lo) / 2;
                    if edges[mid] < v {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                col_bins[row] = lo.min(actual_bins.saturating_sub(1)) as u16;
            }
        }

        if let Some(col) = col_idx {
            assert!(col < self.n_features);
            let offset = col * self.n_rows;
            let old_had_missing = self.feature_has_missing.get(col).copied().unwrap_or(false);
            let new_has_missing = col_bins.iter().any(|&b| b == MISSING_BIN);
            self.bin_indices[offset..offset + self.n_rows].copy_from_slice(&col_bins);
            self.bin_edges[col] = edges;
            self.is_categorical[col] = false;
            if col < self.feature_has_missing.len() {
                self.feature_has_missing[col] = new_has_missing;
            }
            if col < self.feature_non_missing_count.len() {
                self.feature_non_missing_count[col] =
                    (self.n_rows - col_bins.iter().filter(|&&b| b == MISSING_BIN).count()) as u32;
            }
            if old_had_missing || new_has_missing {
                self.rebuild_non_missing_blocks();
            }
            col
        } else {
            let col = self.n_features;
            self.bin_indices.extend_from_slice(&col_bins);
            self.bin_edges.push(edges);
            self.is_categorical.push(false);
            let has_missing = col_bins.iter().any(|&b| b == MISSING_BIN);
            self.feature_has_missing.push(has_missing);
            self.feature_non_missing_count.push(
                (self.n_rows - col_bins.iter().filter(|&&b| b == MISSING_BIN).count()) as u32,
            );
            self.append_non_missing_block(&col_bins, has_missing);
            self.n_features += 1;
            self.n_raw_features += 1;
            col
        }
    }

    /// Append discrete categorical features to the binned data.
    /// Each column is treated like a native categorical with exact-match bins plus
    /// an "other" bucket for unseen values.
    pub fn add_categorical_features(&mut self, cat_values: &[Vec<f64>]) {
        for col_vals in cat_values {
            assert_eq!(col_vals.len(), self.n_rows);
            let mut edges: Vec<f64> = col_vals.iter().copied().filter(|v| !v.is_nan()).collect();
            edges.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            edges.dedup_by(|a, b| *a == *b);
            if edges.is_empty() {
                edges.push(0.0);
            }

            let actual_bins = edges.len();
            let other_bin = actual_bins as u16;
            let mut col_bins = vec![0u16; self.n_rows];
            for row in 0..self.n_rows {
                let v = col_vals[row];
                if v.is_nan() {
                    col_bins[row] = MISSING_BIN;
                } else {
                    let mut lo = 0usize;
                    let mut hi = actual_bins;
                    while lo < hi {
                        let mid = lo + (hi - lo) / 2;
                        if edges[mid] < v {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    col_bins[row] = if lo < actual_bins && edges[lo] == v {
                        lo as u16
                    } else {
                        other_bin
                    };
                }
            }

            self.bin_indices.extend_from_slice(&col_bins);
            self.bin_edges.push(edges);
            self.is_categorical.push(true);
            let has_missing = col_bins.iter().any(|&b| b == MISSING_BIN);
            self.feature_has_missing.push(has_missing);
            self.feature_non_missing_count.push(
                (self.n_rows - col_bins.iter().filter(|&&b| b == MISSING_BIN).count()) as u32,
            );
            self.append_non_missing_block(&col_bins, has_missing);
            self.n_features += 1;
            self.n_raw_features += 1;
        }
    }

    /// Append OTS features to eval/test binned data using pre-computed edges from training.
    /// `ots_values` is n_ots_cols x n_rows, `ots_edges` are the bin edges from training.
    pub fn add_ots_features_with_edges(
        bin_indices: &mut Vec<u16>,
        n_rows: usize,
        ots_values: &[Vec<f64>],
        ots_edges: &[Vec<f64>],
    ) {
        for (col_vals, edges) in ots_values.iter().zip(ots_edges.iter()) {
            assert_eq!(col_vals.len(), n_rows);
            let actual_bins = edges.len();
            let mut col_bins = vec![0u16; n_rows];
            for row in 0..n_rows {
                let v = col_vals[row];
                if v.is_nan() {
                    col_bins[row] = MISSING_BIN;
                } else {
                    let mut lo = 0usize;
                    let mut hi = actual_bins;
                    while lo < hi {
                        let mid = lo + (hi - lo) / 2;
                        if edges[mid] < v {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    col_bins[row] = lo.min(actual_bins - 1) as u16;
                }
            }
            bin_indices.extend_from_slice(&col_bins);
        }
    }

    /// Append categorical features to eval/test binned data using pre-computed
    /// training categories. Unknown categories route to the reserved "other" bin.
    pub fn add_categorical_features_with_edges(
        bin_indices: &mut Vec<u16>,
        n_rows: usize,
        cat_values: &[Vec<f64>],
        cat_edges: &[Vec<f64>],
    ) {
        for (col_vals, edges) in cat_values.iter().zip(cat_edges.iter()) {
            assert_eq!(col_vals.len(), n_rows);
            let actual_bins = edges.len();
            let other_bin = actual_bins as u16;
            let mut col_bins = vec![0u16; n_rows];
            for row in 0..n_rows {
                let v = col_vals[row];
                if v.is_nan() {
                    col_bins[row] = MISSING_BIN;
                } else {
                    let mut lo = 0usize;
                    let mut hi = actual_bins;
                    while lo < hi {
                        let mid = lo + (hi - lo) / 2;
                        if edges[mid] < v {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    col_bins[row] = if lo < actual_bins && edges[lo] == v {
                        lo as u16
                    } else {
                        other_bin
                    };
                }
            }
            bin_indices.extend_from_slice(&col_bins);
        }
    }

    /// Build CLL hash bins for external data (eval/test) given training metadata.
    /// Returns column-major Vec<u16> with CLL bins for all features.
    /// - Native categoricals (is_categorical[col]=true): same bins as bin_edges lookup
    /// - High-cardinality categoricals (cat_features[col] && !is_categorical[col]): hash to 256 bins
    /// - Numeric: 0 (unused for CLL)
    pub fn build_cll_hash_bins(
        raw_data: &[f64],
        n_rows: usize,
        n_features: usize,
        cat_features: &[bool],
        is_categorical: &[bool],
        bin_edges: &[Vec<f64>],
    ) -> Vec<u16> {
        let mut cll_bins = vec![0u16; n_rows * n_features];
        for col in 0..n_features {
            let offset = col * n_rows;
            if col < is_categorical.len() && is_categorical[col] {
                // Native categorical: bin via edge lookup (same as tree splits)
                let edges = &bin_edges[col];
                let actual_bins = edges.len();
                for row in 0..n_rows {
                    let v = raw_data[row * n_features + col];
                    if v.is_nan() {
                        cll_bins[offset + row] = MISSING_BIN;
                    } else {
                        let mut lo = 0usize;
                        let mut hi = actual_bins;
                        while lo < hi {
                            let mid = lo + (hi - lo) / 2;
                            if edges[mid] < v {
                                lo = mid + 1;
                            } else {
                                hi = mid;
                            }
                        }
                        cll_bins[offset + row] = if lo < actual_bins && edges[lo] == v {
                            lo as u16
                        } else {
                            MISSING_BIN
                        };
                    }
                }
            } else if col < cat_features.len() && cat_features[col] {
                // High-cardinality categorical: hash to 0..255
                for row in 0..n_rows {
                    let v = raw_data[row * n_features + col];
                    if v.is_nan() {
                        cll_bins[offset + row] = MISSING_BIN;
                    } else {
                        let iv = v as i64;
                        let h = ((iv.wrapping_mul(0x9E3779B97F4A7C15_u64 as i64)) as u64) >> 56;
                        cll_bins[offset + row] = h as u16;
                    }
                }
            }
        }
        cll_bins
    }

    /// Get the CLL bin for a given row and column (uses cll_hash_bins).
    #[inline(always)]
    pub fn get_cll_bin(&self, row: usize, col: usize) -> u16 {
        self.cll_hash_bins[col * self.n_rows + row]
    }
}

// ── Leaf-wise priority queue ────────────────────────────────────────────────

struct SplitCandidate {
    gain: f64,
    node_idx: usize,
    start: usize,
    end: usize,
    depth: usize,
    best_feat: usize,
    best_bin: usize,
    best_missing_left: bool,
    best_cat_mask: CatBitmask,
    best_is_cat: bool,
    g_sum: f64,
    h_sum: f64,
}

impl PartialEq for SplitCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.gain == other.gain
    }
}
impl Eq for SplitCandidate {}
impl PartialOrd for SplitCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for SplitCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.gain
            .partial_cmp(&other.gain)
            .unwrap_or(Ordering::Equal)
    }
}

// ── Decision Tree ───────────────────────────────────────────────────────────

/// Category Lookup Leaf: per-category prediction values for a leaf node.
/// Instead of a single leaf value, the leaf uses a categorical feature's bin
/// to look up a category-specific value, with regularization toward the overall leaf value.
/// Supports both single-feature and pairwise-feature lookups (feature2 != u32::MAX).
#[derive(Clone, Serialize, Deserialize)]
pub struct CatLookup {
    pub feature: u32,         // which categorical feature (primary)
    pub feature2: u32,        // secondary feature for pairs (u32::MAX = single feature)
    pub feature3: u32,        // third feature for triples (u32::MAX = unused)
    pub bin_values: Vec<f64>, // prediction value per bin (indexed by bin id)
    pub default_value: f64,   // fallback for missing/unseen bins
    pub is_numeric: bool, // NLL: true if this lookup is on a numeric feature (uses bin_indices, not cll_hash_bins)
    pub n_coarse_bins: usize, // NLL: number of coarsened bins (0 = categorical, uses bin_values.len())
    pub pair_stride: usize, // categorical pairs: >0 means exact table with bin = b1 * pair_stride + b2
    pub triple_stride: usize, // categorical triples: >0 means bin = (b1 * pair_stride + b2) * triple_stride + b3
}

#[derive(Clone, Serialize, Deserialize)]
pub struct GuidedCatChoice {
    pub feature: u32,
    pub feature2: u32,
    pub feature3: u32,
    pub n_bins: usize,
    pub pair_stride: usize,
    pub triple_stride: usize,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct CatTupleConfig {
    pub enabled: bool,
    pub max_order: usize,
    pub top_features: usize,
    pub hash_bins: usize,
    pub min_leaf: usize,
    pub gain_margin: f64,
}

/// Convert a raw feature value to its CLL bin index. Returns usize::MAX for missing/unseen.
#[inline]
pub(super) fn raw_to_cll_bin(feat: usize, raw_row: &[f64], binned: &BinnedData) -> usize {
    let val = raw_row[feat];
    if val.is_nan() {
        return usize::MAX;
    }
    if feat < binned.cll_is_categorical.len()
        && binned.cll_is_categorical[feat]
        && (feat >= binned.is_categorical.len() || !binned.is_categorical[feat])
    {
        // High-cardinality hashed: hash to 0..255
        let iv = val as i64;
        ((iv.wrapping_mul(0x9E3779B97F4A7C15_u64 as i64)) as u64 >> 56) as usize
    } else {
        // Native categorical: edge-based bin lookup
        let edges = &binned.bin_edges[feat];
        let actual_bins = edges.len();
        let mut lo = 0usize;
        let mut hi = actual_bins;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if edges[mid] < val {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo < actual_bins && (edges[lo] - val).abs() < 0.5 {
            lo
        } else {
            usize::MAX
        }
    }
}

/// Compute the CLL bin for a row, handling both single-feature and pairwise CLL.
/// Returns usize::MAX for missing values.
#[inline(always)]
pub(super) fn cll_bin_for_row(
    cll: &CatLookup,
    cll_bins: &[u16],
    n_rows: usize,
    row: usize,
) -> usize {
    let b1 = cll_bins[cll.feature as usize * n_rows + row];
    if b1 == MISSING_BIN {
        return usize::MAX;
    }
    if cll.feature2 == u32::MAX {
        // Single feature
        b1 as usize
    } else {
        let b2 = cll_bins[cll.feature2 as usize * n_rows + row];
        if b2 == MISSING_BIN {
            return usize::MAX;
        }
        if cll.feature3 != u32::MAX {
            let b3 = cll_bins[cll.feature3 as usize * n_rows + row];
            if b3 == MISSING_BIN {
                return usize::MAX;
            }
            if cll.pair_stride > 0 && cll.triple_stride > 0 {
                ((b1 as usize) * cll.pair_stride + b2 as usize) * cll.triple_stride + b3 as usize
            } else {
                let n_bins = cll.bin_values.len().max(1);
                ((b1 as u32)
                    .wrapping_mul(257)
                    .wrapping_add((b2 as u32).wrapping_mul(17))
                    .wrapping_add(b3 as u32)) as usize
                    % n_bins
            }
        } else if cll.pair_stride > 0 {
            (b1 as usize) * cll.pair_stride + b2 as usize
        } else {
            let n_bins = cll.bin_values.len().max(1);
            ((b1 as u32).wrapping_mul(257).wrapping_add(b2 as u32)) as usize % n_bins
        }
    }
}

/// Same as cll_bin_for_row but for raw bin_indices / cll_bins arrays from eval data.
#[inline(always)]
pub(super) fn cll_bin_for_row_raw(
    cll: &CatLookup,
    cll_bins: &[u16],
    n_rows: usize,
    row: usize,
) -> usize {
    cll_bin_for_row(cll, cll_bins, n_rows, row)
}

/// NLL: compute coarsened numeric bin from bin_indices for a row.
/// Maps original 0..n_original_bins to 0..n_coarse_bins.
#[inline(always)]
pub(super) fn nll_bin_for_row(
    cll: &CatLookup,
    bin_indices: &[u16],
    n_rows: usize,
    row: usize,
) -> usize {
    let bin = bin_indices[cll.feature as usize * n_rows + row];
    if bin == MISSING_BIN {
        return usize::MAX;
    }
    let b1 = ((bin as usize * cll.n_coarse_bins) >> 8).min(cll.n_coarse_bins - 1);
    if cll.feature2 == u32::MAX {
        return b1;
    }
    let bin2 = bin_indices[cll.feature2 as usize * n_rows + row];
    if bin2 == MISSING_BIN {
        return usize::MAX;
    }
    let b2 = ((bin2 as usize * cll.n_coarse_bins) >> 8).min(cll.n_coarse_bins - 1);
    if cll.pair_stride > 0 {
        b1 * cll.pair_stride + b2
    } else {
        b1 * cll.n_coarse_bins + b2
    }
}

/// NLL: compute coarsened numeric bin from raw feature value at inference time.
#[inline]
pub(super) fn raw_to_nll_bin(
    feat: usize,
    val: f64,
    bin_edges: &[Vec<f64>],
    n_coarse_bins: usize,
) -> usize {
    if val.is_nan() {
        return usize::MAX;
    }
    let edges = &bin_edges[feat];
    let n_original = edges.len(); // number of edges = number of bins for numeric
    if n_original == 0 {
        return 0;
    }
    // Binary search: find bin index in 0..n_original
    let mut lo = 0usize;
    let mut hi = n_original;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if edges[mid] <= val {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    // lo is now the original bin (0..n_original)
    let coarse = (lo * n_coarse_bins) / (n_original + 1);
    coarse.min(n_coarse_bins - 1)
}

#[inline]
pub(super) fn raw_to_num_bin(edges: &[f64], val: f64) -> Option<usize> {
    if val.is_nan() || edges.is_empty() {
        return None;
    }
    let mut lo = 0usize;
    let mut hi = edges.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if edges[mid] <= val {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    Some(lo.min(edges.len().saturating_sub(1)))
}

#[derive(Clone, Serialize, Deserialize)]
pub struct DecisionTree {
    pub split_features: Vec<u32>,
    pub split_bins: Vec<u16>,
    pub values: Vec<f64>,
    pub left_children: Vec<u32>,
    pub right_children: Vec<u32>,
    pub missing_goes_left: Vec<bool>,
    pub is_oblique_split: Vec<bool>,
    pub is_cat_split: Vec<bool>,
    pub cat_left_masks: Vec<CatBitmask>,
    pub oblique_features: Vec<u32>, // flattened [node*2 + j], u32::MAX = unused
    pub oblique_weights: Vec<f32>,  // flattened [node*2 + j], scaled for bin-index space
    pub oblique_thresholds: Vec<f32>,
    pub cat_lookups: Vec<Option<CatLookup>>, // per-node, Some only for CLL leaves
    pub ramp_slopes: Vec<f64>, // per-node slope for piecewise linear refinement (empty = disabled)
    pub ramp_features: Vec<u32>, // per-node parent split feature index (u32::MAX = no ramp)
    pub ramp_k: usize, // number of ramp features per node (1 = single parent, 2+ = path features)
    pub leaf_pair_slopes: Vec<f64>, // per-node bilinear slope for local arithmetic experts
    pub leaf_pair_features: Vec<u32>, // flattened [node*2 + j], u32::MAX = unused
    pub quad_slopes: Vec<f64>, // per-node quadratic interaction slopes (empty = disabled)
    pub quad_pairs: Vec<(usize, usize)>, // (feat_i, feat_j) pairs for quadratic interactions
    pub quad_n_interactions: usize, // number of interaction pairs per node
    // PRM (Posterior Refinement Marginalization): per-node training stats for
    // confidence-aware test-time pruning. Empty Vec = disabled (old trees).
    pub node_h_sum: Vec<f64>, // sum of hessians at this node during training
    pub node_count: Vec<u32>, // number of training rows at this node during training
    // GGFP v5.0 — JIT-CatPairSplit. All vectors co-indexed with split_features.
    // cat_pair_feat2[n] == u32::MAX means node n is not a cat-pair split.
    // All five may be Vec::new() entirely (legacy trees / feature disabled).
    pub cat_pair_feat2: Vec<u32>,
    pub cat_pair_bucket_map_a: Vec<Vec<u8>>,
    pub cat_pair_bucket_map_b: Vec<Vec<u8>>,
    pub cat_pair_cell_mask: Vec<u64>,
    pub cat_pair_k_buckets: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_major_eval_bins_match_column_major_bins() {
        let n_rows = 6usize;
        let n_features = 3usize;
        let data = vec![
            -1.5,
            10.0,
            f64::NAN, //
            -1.0,
            15.0,
            1.0, //
            -0.2,
            20.0,
            2.2, //
            0.0,
            30.0,
            3.0, //
            0.7,
            f64::NAN,
            4.5, //
            2.0,
            10.0,
            -5.0, //
        ];
        let bin_edges = vec![
            vec![-1.0, 0.0, 1.0, 2.0],
            vec![10.0, 20.0, 30.0],
            vec![1.0, 2.0, 3.0, 4.0],
        ];
        let is_categorical = vec![false, true, false];

        let column_major =
            BinnedData::bin_with_edges(&data, n_rows, n_features, &bin_edges, &is_categorical);
        let row_major = BinnedData::bin_with_edges_row_major(
            &data,
            n_rows,
            n_features,
            &bin_edges,
            &is_categorical,
        );

        for row in 0..n_rows {
            for col in 0..n_features {
                assert_eq!(
                    column_major[col * n_rows + row],
                    row_major[row * n_features + col],
                    "row={row} col={col}"
                );
            }
        }
    }

    #[test]
    fn eval_binning_handles_empty_edges_as_missing() {
        let data = vec![1.0, f64::NAN, 3.0];
        let bin_edges = vec![Vec::new()];
        let is_categorical = vec![false];

        let column_major = BinnedData::bin_with_edges(&data, 3, 1, &bin_edges, &is_categorical);
        let row_major =
            BinnedData::bin_with_edges_row_major(&data, 3, 1, &bin_edges, &is_categorical);

        assert_eq!(column_major, vec![MISSING_BIN; 3]);
        assert_eq!(row_major, vec![MISSING_BIN; 3]);
    }
}
