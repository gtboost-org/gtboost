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
        cat_pair_cfg: CatPairConfig,
    ) -> Self {
        let max_nodes = (1usize << (max_depth + 1)).min(65536);
        let mut tree = TreeBuilder::new(max_nodes);
        let mut row_buf: Vec<u32> = indices.to_vec();

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
            // Stack entries: (start, end, depth, node_idx, g_sum, h_sum, cached_hists)
            let mut stack: Vec<(usize, usize, usize, usize, f64, f64, Option<NodeHists>)> =
                Vec::with_capacity(max_nodes);
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
            stack.push((0, row_buf.len(), 0, 0, root_g, root_h, Some(root_hists)));

            while let Some((start, end, depth, node_idx, g_sum, h_sum, cached_hists)) = stack.pop()
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
                let use_interval_splits = interval_splits && depth == 0;

                // If we have cached histograms, scan them; otherwise fall back to find_best_split
                let (mut split_result, mut node_hists) = if let Some(nh) = cached_hists {
                    let sr = find_best_split_from_hists(
                        &nh,
                        &tree_features,
                        split_features,
                        binned,
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
                        use_interval_splits,
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
                        random_strength,
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
                        let (lg, lh) = sum_gh(gradients, hessians, left_slice);
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

                    // Start with the incumbent best split's score.
                    let mut best_score =
                        split_result.gain + lookahead_alpha * eval_future(&split_result);

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
                        let cand_score = cand.gain + lookahead_alpha * eval_future(&cand);
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

                if split_result.gain <= 0.0 || !split_result.gain.is_finite() {
                    tree.set_leaf(node_idx, leaf_value);
                    continue;
                }

                let left_end =
                    partition_indices_split(&mut row_buf, start, end, binned, &split_result);
                if left_end == start || left_end == end {
                    tree.set_leaf(node_idx, leaf_value);
                    continue;
                }

                let left_indices = &row_buf[start..left_end];
                let (lg, lh) = sum_gh(gradients, hessians, left_indices);
                let rg = g_sum - lg;
                let rh = h_sum - lh;

                let (left_idx, right_idx) =
                    tree.add_split_from_sr(node_idx, split_result, leaf_value);

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
                            ));
                            stack.push((
                                start,
                                left_end,
                                child_depth,
                                left_idx,
                                lg,
                                lh,
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
                                child_depth,
                                right_idx,
                                rg,
                                rh,
                                Some(smaller_hists),
                            ));
                            stack.push((
                                start,
                                left_end,
                                child_depth,
                                left_idx,
                                lg,
                                lh,
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
                        stack.push((left_end, end, child_depth, right_idx, rg, rh, None));
                        stack.push((
                            start,
                            left_end,
                            child_depth,
                            left_idx,
                            lg,
                            lh,
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
                            child_depth,
                            right_idx,
                            rg,
                            rh,
                            Some(right_hists),
                        ));
                        stack.push((start, left_end, child_depth, left_idx, lg, lh, None));
                    } else {
                        stack.push((left_end, end, child_depth, right_idx, rg, rh, None));
                        stack.push((start, left_end, child_depth, left_idx, lg, lh, None));
                    }
                } else {
                    // No parent hists (fallback) — push None, children will use find_best_split
                    stack.push((left_end, end, child_depth, right_idx, rg, rh, None));
                    stack.push((start, left_end, child_depth, left_idx, lg, lh, None));
                }
                if let Some(h) = node_hists.take() {
                    hist_pool.recycle(h);
                }
            }
        } else {
            // ── Original path for extra_trees ──
            let mut stack: Vec<(usize, usize, usize, usize, f64, f64)> =
                Vec::with_capacity(max_nodes);
            let mut g_hist = vec![0.0f64; max_bins];
            let mut h_hist = vec![0.0f64; max_bins];
            stack.push((0, row_buf.len(), 0, 0, root_g, root_h));

            while let Some((start, end, depth, node_idx, g_sum, h_sum)) = stack.pop() {
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

                if split_result.gain <= 0.0 || !split_result.gain.is_finite() {
                    tree.set_leaf(node_idx, leaf_value);
                    continue;
                }

                let left_end =
                    partition_indices_split(&mut row_buf, start, end, binned, &split_result);
                if left_end == start || left_end == end {
                    tree.set_leaf(node_idx, leaf_value);
                    continue;
                }

                let left_indices = &row_buf[start..left_end];
                let (lg, lh) = sum_gh(gradients, hessians, left_indices);
                let rg = g_sum - lg;
                let rh = h_sum - lh;

                let (left_idx, right_idx) =
                    tree.add_split_from_sr(node_idx, split_result, leaf_value);
                stack.push((left_end, end, depth + 1, right_idx, rg, rh));
                stack.push((start, left_end, depth + 1, left_idx, lg, lh));
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
                CatPairConfig::default(),
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
            let (lg, lh) = sum_gh(gradients, hessians, left_indices);
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
                )
            };

            if split_result.gain <= 0.0 || !split_result.gain.is_finite() {
                tree.set_leaf(node_idx, leaf_value);
                continue;
            }

            let left_end = partition_indices_split(&mut row_buf, start, end, binned, &split_result);
            if left_end == start || left_end == end {
                tree.set_leaf(node_idx, leaf_value);
                continue;
            }

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

            // Ensure node_g/node_h buffers are large enough for new children
            let needed = (right_idx + 1) * n_classes;
            if needed > node_g.len() {
                node_g.resize(needed, 0.0);
                node_h.resize(needed, 0.0);
            }

            // Compute left child K sums, derive right by subtraction
            let left_indices = &row_buf[start..left_end];
            let l_base = left_idx * n_classes;
            let r_base = right_idx * n_classes;
            for k in 0..n_classes {
                let kb = k * n_rows;
                let mut lg = 0.0f64;
                let mut lh = 0.0f64;
                for &idx in left_indices {
                    lg += all_gradients[kb + idx as usize];
                    lh += all_hessians[kb + idx as usize];
                }
                node_g[l_base + k] = lg;
                node_h[l_base + k] = lh;
                node_g[r_base + k] = node_g[g_base + k] - lg;
                node_h[r_base + k] = node_h[g_base + k] - lh;
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

            type OblivMultiResult = (f64, usize, usize, bool, bool, CatBitmask);
            let mut best = (f64::NEG_INFINITY, 0usize, 0usize, true, false, Vec::new());

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

            let (
                best_total_gain,
                best_feat,
                best_bin,
                best_missing_left,
                best_is_cat,
                best_cat_mask,
            ) = best;

            if best_total_gain <= 0.0 || !best_total_gain.is_finite() {
                break;
            }

            let mut new_ranges = Vec::with_capacity(node_ranges.len() * 2);
            let mut new_ids = Vec::with_capacity(node_ids.len() * 2);
            for (ni, &nid) in node_ids.iter().enumerate() {
                let (start, end) = node_ranges[ni];
                if start == end {
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
                let g_base = ni * n_classes;
                let g0 = node_g[g_base];
                let h0 = node_h[g_base];
                let count = (end - start) as f64;
                let leaf_value = -g0 / (h0 + lambda_reg + lambda_reg / count.max(1.0).sqrt());
                let (lid, rid) = tree.add_split(
                    nid,
                    best_feat as u32,
                    best_bin as u16,
                    leaf_value,
                    best_missing_left,
                    false,
                    [u32::MAX, u32::MAX],
                    [0.0, 0.0],
                    0.0,
                    best_is_cat,
                    best_cat_mask.clone(),
                );

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
        cat_pair_cfg: CatPairConfig,
    ) -> Self {
        let max_nodes = max_leaves * 2 + 2;
        let mut tree = TreeBuilder::new(max_nodes);
        let mut row_buf: Vec<u32> = indices.to_vec();
        let mut heap: BinaryHeap<SplitCandidate> = BinaryHeap::with_capacity(max_leaves);

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

        let get_features = |depth: usize| -> &Vec<usize> {
            if colsample_bylevel < 1.0 && depth < level_features.len() {
                &level_features[depth]
            } else {
                &tree_features
            }
        };

        let root_indices = &row_buf[0..indices.len()];
        let (g_sum, h_sum) = sum_gh(gradients, hessians, root_indices);
        let n_root = indices.len() as f64;
        let root_leaf_val = -g_sum / (h_sum + lambda_reg + lambda_reg / n_root.max(1.0).sqrt());
        tree.set_node_stats(0, h_sum, indices.len() as u32);
        tree.set_leaf(0, root_leaf_val);

        if indices.len() > 1 && h_sum >= min_h && max_depth > 0 {
            let sr = if extra_trees {
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
                    0.0,
                    gamma,
                    min_h,
                    &mut g_hist,
                    &mut h_hist,
                    random_strength,
                    tree_seed,
                    cat_smooth,
                    monotone_constraints,
                    gain_penalty,
                    false,
                )
            };
            let mut push_split = sr.gain > 0.0 && sr.gain.is_finite();
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
                heap.push(SplitCandidate {
                    gain: sr.gain,
                    node_idx: 0,
                    start: 0,
                    end: indices.len(),
                    depth: 0,
                    best_feat: sr.feat,
                    best_bin: sr.bin,
                    best_missing_left: sr.missing_left,
                    best_cat_mask: sr.cat_mask,
                    best_is_cat: sr.is_cat,
                    g_sum,
                    h_sum,
                });
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

            let left_end = partition_indices(
                &mut row_buf,
                cand.start,
                cand.end,
                binned,
                cand.best_feat,
                cand.best_bin as u16,
                cand.best_missing_left,
                cand.best_is_cat,
                &cand.best_cat_mask,
            );
            if left_end == cand.start || left_end == cand.end {
                continue;
            }

            let n_cand = (cand.end - cand.start) as f64;
            let leaf_value =
                -cand.g_sum / (cand.h_sum + lambda_reg + lambda_reg / n_cand.max(1.0).sqrt());
            let (left_idx, right_idx) = tree.add_split(
                cand.node_idx,
                cand.best_feat as u32,
                cand.best_bin as u16,
                leaf_value,
                cand.best_missing_left,
                false,
                [u32::MAX, u32::MAX],
                [0.0, 0.0],
                0.0,
                cand.best_is_cat,
                cand.best_cat_mask,
            );
            n_leaves += 1;

            let child_depth = cand.depth + 1;
            let child_feats = get_features(child_depth);

            // Left child
            let left_indices = &row_buf[cand.start..left_end];
            let (lg, lh) = sum_gh(gradients, hessians, left_indices);
            let n_left = left_indices.len() as f64;
            let left_leaf_val = -lg / (lh + lambda_reg + lambda_reg / n_left.max(1.0).sqrt());
            tree.set_node_stats(left_idx, lh, left_indices.len() as u32);
            tree.set_leaf(left_idx, left_leaf_val);

            if left_indices.len() > 1
                && lh >= min_h
                && child_depth < max_depth
                && n_leaves < max_leaves
            {
                let sr = if extra_trees {
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
                        0.0,
                        gamma,
                        min_h,
                        &mut g_hist,
                        &mut h_hist,
                        random_strength,
                        tree_seed.wrapping_add(child_depth as u64),
                        cat_smooth,
                        monotone_constraints,
                        gain_penalty,
                        false,
                    )
                };
                let mut push_split = sr.gain > 0.0 && sr.gain.is_finite();
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
                    heap.push(SplitCandidate {
                        gain: sr.gain,
                        node_idx: left_idx,
                        start: cand.start,
                        end: left_end,
                        depth: child_depth,
                        best_feat: sr.feat,
                        best_bin: sr.bin,
                        best_missing_left: sr.missing_left,
                        best_cat_mask: sr.cat_mask,
                        best_is_cat: sr.is_cat,
                        g_sum: lg,
                        h_sum: lh,
                    });
                }
            } else if cat_lookup_smooth > 0.0 && left_indices.len() > 1 {
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
            }

            // Right child: derive sums from parent - left (avoids scanning right indices)
            let right_indices = &row_buf[left_end..cand.end];
            let rg = cand.g_sum - lg;
            let rh = cand.h_sum - lh;
            let n_right = right_indices.len() as f64;
            let right_leaf_val = -rg / (rh + lambda_reg + lambda_reg / n_right.max(1.0).sqrt());
            tree.set_node_stats(right_idx, rh, right_indices.len() as u32);
            tree.set_leaf(right_idx, right_leaf_val);

            if right_indices.len() > 1
                && rh >= min_h
                && child_depth < max_depth
                && n_leaves < max_leaves
            {
                let sr = if extra_trees {
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
                        0.0,
                        gamma,
                        min_h,
                        &mut g_hist,
                        &mut h_hist,
                        random_strength,
                        tree_seed.wrapping_add(child_depth as u64),
                        cat_smooth,
                        monotone_constraints,
                        gain_penalty,
                        false,
                    )
                };
                let mut push_split = sr.gain > 0.0 && sr.gain.is_finite();
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
                    heap.push(SplitCandidate {
                        gain: sr.gain,
                        node_idx: right_idx,
                        start: left_end,
                        end: cand.end,
                        depth: child_depth,
                        best_feat: sr.feat,
                        best_bin: sr.bin,
                        best_missing_left: sr.missing_left,
                        best_cat_mask: sr.cat_mask,
                        best_is_cat: sr.is_cat,
                        g_sum: rg,
                        h_sum: rh,
                    });
                }
            } else if cat_lookup_smooth > 0.0 && right_indices.len() > 1 {
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
        gamma: f64,
        max_depth: usize,
        min_child_weight: f64,
        feature_mask: &[bool],
        gain_penalty: f64,
        extra_trees: bool,
        tree_seed: u64,
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
            for &(start, end) in &node_ranges {
                let (g, h) = sum_gh(gradients, hessians, &row_buf[start..end]);
                node_gh.push((g, h));
            }
            for (i, &nid) in node_ids.iter().enumerate() {
                let (g, h) = node_gh[i];
                let (ns, ne) = node_ranges[i];
                let nc = (ne - ns) as f64;
                tree.set_node_stats(nid, h, (ne - ns) as u32);
                tree.set_leaf(nid, -g / (h + lambda_reg + lambda_reg / nc.max(1.0).sqrt()));
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

            let eval_feat_obliv = |feat: usize| -> OblivResult {
                let feat_n_bins = binned.n_bins(feat);
                if feat_n_bins <= 1 {
                    return (f64::NEG_INFINITY, feat, 0, true, false, Vec::new());
                }

                // Per-thread histogram buffers
                let mut flat_g = vec![0.0f64; n_nodes * feat_n_bins];
                let mut flat_h = vec![0.0f64; n_nodes * feat_n_bins];
                let mut g_miss = vec![0.0f64; n_nodes];
                let mut h_miss = vec![0.0f64; n_nodes];

                let col_offset = feat * binned.n_rows;
                for ni in 0..n_nodes {
                    let (start, end) = node_ranges[ni];
                    let base = ni * feat_n_bins;
                    for &idx in &row_buf[start..end] {
                        let bin = binned.bin_indices[col_offset + idx as usize] as usize;
                        let g = gradients[idx as usize];
                        let h = hessians[idx as usize];
                        if bin == MISSING_BIN as usize {
                            g_miss[ni] += g;
                            h_miss[ni] += h;
                        } else if bin < feat_n_bins {
                            flat_g[base + bin] += g;
                            flat_h[base + bin] += h;
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
                        let smooth = lambda_reg;
                        cat_bins_local.sort_by(|a, b| {
                            let ra = (a.1 + smooth * global_ratio) / (a.2 + smooth);
                            let rb = (b.1 + smooth * global_ratio) / (b.2 + smooth);
                            ra.partial_cmp(&rb).unwrap_or(Ordering::Equal)
                        });

                        // Cumulative sums per node in sorted order
                        let n_cats = cat_bins_local.len();
                        let mut cum_g = vec![0.0f64; n_nodes * n_cats];
                        let mut cum_h = vec![0.0f64; n_nodes * n_cats];
                        for ni in 0..n_nodes {
                            let base = ni * feat_n_bins;
                            let cbase = ni * n_cats;
                            let mut sg = 0.0f64;
                            let mut sh = 0.0f64;
                            for (ci, &(bin, _, _)) in cat_bins_local.iter().enumerate() {
                                sg += flat_g[base + bin];
                                sh += flat_h[base + bin];
                                cum_g[cbase + ci] = sg;
                                cum_h[cbase + ci] = sh;
                            }
                        }

                        let mut best_ci = 0usize;
                        for ci in 0..n_cats - 1 {
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
                                    let (lg, lh, rg, rh) = if miss_left {
                                        (cg + g_miss[ni], ch + h_miss[ni], g_nm - cg, h_nm - ch)
                                    } else {
                                        (cg, ch, g_nm - cg + g_miss[ni], h_nm - ch + h_miss[ni])
                                    };
                                    if lh < min_h || rh < min_h {
                                        continue;
                                    }
                                    let mut gain = 0.5
                                        * (lg * lg / (lh + lambda_reg)
                                            + rg * rg / (rh + lambda_reg)
                                            - g_total * g_total / (h_total + lambda_reg))
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
                                        n_cats.saturating_sub(1),
                                    );
                                    total_gain += gain;
                                }
                                if total_gain > best_gain {
                                    best_gain = total_gain;
                                    best_ml = miss_left;
                                    best_cat = true;
                                    best_ci = ci;
                                }
                            }
                        }
                        if best_cat {
                            best_mask = Vec::new();
                            for j in 0..=best_ci {
                                bitmask_set(&mut best_mask, cat_bins_local[j].0);
                            }
                        }
                    }
                } else {
                    // Numeric: prefix sum and scan
                    let mut cum_g = vec![0.0f64; n_nodes * feat_n_bins];
                    let mut cum_h = vec![0.0f64; n_nodes * feat_n_bins];
                    for ni in 0..n_nodes {
                        let base = ni * feat_n_bins;
                        cum_g[base] = flat_g[base];
                        cum_h[base] = flat_h[base];
                        for b in 1..feat_n_bins {
                            cum_g[base + b] = cum_g[base + b - 1] + flat_g[base + b];
                            cum_h[base + b] = cum_h[base + b - 1] + flat_h[base + b];
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
                                let (lg, lh, rg, rh) = if miss_left {
                                    (cg + g_miss[ni], ch + h_miss[ni], g_nm - cg, h_nm - ch)
                                } else {
                                    (cg, ch, g_nm - cg + g_miss[ni], h_nm - ch + h_miss[ni])
                                };
                                if lh < min_h || rh < min_h {
                                    continue;
                                }
                                let mut gain = 0.5
                                    * (lg * lg / (lh + lambda_reg) + rg * rg / (rh + lambda_reg)
                                        - g_total * g_total / (h_total + lambda_reg))
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

            let winner: OblivResult = if use_par {
                active_features
                    .par_iter()
                    .map(|&f| eval_feat_obliv(f))
                    .reduce(
                        || empty_result.clone(),
                        |a, b| if b.0 > a.0 { b } else { a },
                    )
            } else {
                let mut best = empty_result.clone();
                for &f in &active_features {
                    let r = eval_feat_obliv(f);
                    if r.0 > best.0 {
                        best = r;
                    }
                }
                best
            };

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
                    -g_node / (h_node + lambda_reg + lambda_reg / nc_node.max(1.0).sqrt()),
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
                    -lg / (lh + lambda_reg + lambda_reg / nc_l.max(1.0).sqrt()),
                );
                tree.set_leaf(
                    rid,
                    -rg / (rh + lambda_reg + lambda_reg / nc_r.max(1.0).sqrt()),
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
