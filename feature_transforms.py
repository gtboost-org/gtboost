"""Optional feature-transform hooks used by legacy experiments.

`build_ordered_posterior_views` remains a no-op: the ordered-posterior
transform was rejected in the experiment ledger.

Two production entry points:
  apply_tle  - Tree-Leaf Embedding. One small extractor GBT learns joint
               cat+num conjunctions; leaf indices and OOF E[r|leaf] become
               features for the main booster. Auto-gated with a high-cardinality
               override.
  apply_tpl  - Tuple Posterior Layer (legacy). Hash-based posterior coordinates
               over singles/pairs/triples with multi-stage residual refresh.
  apply_hat_oof_tools - HAT v0 global OOF categorical/cohort tool compiler with
               paired-delta + shadow critic logs.
  build_pcf_geometry_views - PCF posterior-coordinate feature views. This is the
               current CatBoost-counterfactual microscope: replace raw
               categorical ids by leakage-safe posterior geometry, with
               fold-local maps for fair HPO.
All hooks return the original matrices unchanged when disabled or rejected.
"""
import itertools
import os
from typing import Optional, Sequence

import numpy as np


def build_ordered_posterior_views(*_args, **_kwargs):
    return None


# ── PCF: posterior-coordinate feature views ──────────────────────────────────

def _pcf_entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    p = counts / total
    return float(-(p * np.log(np.clip(p, 1e-15, 1.0))).sum())


def _pcf_entropy(col: np.ndarray) -> float:
    vals = np.asarray(col)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    _u, counts = np.unique(vals.astype(np.int64, copy=False), return_counts=True)
    return _pcf_entropy_from_counts(counts)


def _pcf_effective_support(X: np.ndarray, cat_mask: Sequence[bool]) -> dict:
    card = []
    repeated = []
    entropy_eff = []
    for j, is_cat in enumerate(cat_mask):
        if not is_cat:
            continue
        col = np.asarray(X[:, j])
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        _u, counts = np.unique(col.astype(np.int64, copy=False), return_counts=True)
        card.append(int(len(counts)))
        repeated.append(int(np.sum(counts >= 5)))
        denom = float(np.sum(counts.astype(np.float64) ** 2))
        entropy_eff.append(float(counts.sum() ** 2 / denom) if denom > 0.0 else 0.0)
    return {
        "n_cat": int(sum(bool(c) for c in cat_mask)),
        "max_cardinality": int(max(card) if card else 0),
        "sum_repeated_levels": int(sum(repeated)),
        "max_entropy_eff": float(max(entropy_eff) if entropy_eff else 0.0),
    }


def _pcf_eligible(X: np.ndarray, cat_mask: Sequence[bool], cfg: dict) -> tuple[bool, dict]:
    stats = _pcf_effective_support(X, cat_mask)
    ok = (
        stats["n_cat"] >= int(cfg.get("min_cat_features", 2))
        and stats["max_cardinality"] >= int(cfg.get("min_max_cardinality", 32))
        and stats["sum_repeated_levels"] >= int(cfg.get("min_repeated_levels", 50))
        and stats["max_entropy_eff"] >= float(cfg.get("min_entropy_eff", 16.0))
    )
    stats["eligible"] = bool(ok)
    if not ok:
        stats["reason"] = (
            "PCF effective-support rejected "
            f"n_cat={stats['n_cat']} max_card={stats['max_cardinality']} "
            f"repeated={stats['sum_repeated_levels']} "
            f"entropy_eff={stats['max_entropy_eff']:.2f}"
        )
    return bool(ok), stats


def _pcf_selected_cat_columns(X: np.ndarray, cat_mask: Sequence[bool], max_cat: int) -> list[int]:
    cat_idx = [i for i, c in enumerate(cat_mask) if c]
    ranked = sorted(((_pcf_entropy(X[:, j]), j) for j in cat_idx), reverse=True)
    return [j for _, j in ranked[:max(0, int(max_cat))]]


def _pcf_splitmix64_scalar(z: int) -> int:
    mask = (1 << 64) - 1
    z = (int(z) + 0x9E3779B97F4A7C15) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return (z ^ (z >> 31)) & mask


def _pcf_full_tuple_keys_py(X: np.ndarray, cols: Sequence[int]) -> np.ndarray:
    out = np.empty(int(X.shape[0]), dtype=np.int64)
    mask = (1 << 64) - 1
    positive = (1 << 63) - 1
    for i in range(int(X.shape[0])):
        h = 0xCBF29CE484222325
        for k, col in enumerate(cols):
            val = int(X[i, col]) & mask
            pos = ((k + 1) * 0x9E3779B97F4A7C15) & mask
            mixed = _pcf_splitmix64_scalar(val ^ pos)
            h ^= mixed
            h = (h * 0x00000100000001B3) & mask
            h = _pcf_splitmix64_scalar(h)
        out[i] = int(h & positive)
    return out


def _pcf_key_matrix(X: np.ndarray, cols: Sequence[int], hash_bins: int) -> np.ndarray:
    hash_bins = int(hash_bins)
    if hash_bins < 0:
        raise ValueError("hash_bins must be non-negative; use 0 for full tuple keys")
    try:
        import gtboost

        rust_hash = getattr(gtboost, "pcf_hash_keys", None)
    except Exception:
        rust_hash = None
    if rust_hash is not None:
        return np.asarray(
            rust_hash(
                np.asarray(X, dtype=np.float64),
                [int(c) for c in cols],
                int(hash_bins),
            ),
            dtype=np.int64,
        )
    if len(cols) == 1:
        return np.asarray(X[:, cols[0]], dtype=np.int64)
    if hash_bins == 0:
        return _pcf_full_tuple_keys_py(X, cols)
    h = np.zeros(X.shape[0], dtype=np.int64)
    for k, col in enumerate(cols):
        vals = np.asarray(X[:, col], dtype=np.int64)
        h ^= vals * (1_000_003 + 2_654_435_761 * (k + 1))
    return np.mod(h, hash_bins).astype(np.int64)


def _pcf_fit_table(
    keys: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    alpha: float,
    global_prior: np.ndarray,
) -> dict[int, tuple[np.ndarray, float]]:
    uniq, inv = np.unique(keys.astype(np.int64, copy=False), return_inverse=True)
    counts = np.zeros((len(uniq), n_classes), dtype=np.float64)
    np.add.at(counts, (inv, y.astype(np.int64)), 1.0)
    totals = counts.sum(axis=1)
    post = (counts + float(alpha) * global_prior[None, :]) / (
        totals[:, None] + float(alpha)
    )
    return {int(k): (post[i], float(totals[i])) for i, k in enumerate(uniq)}


def _pcf_apply_table(
    keys: np.ndarray,
    table: dict[int, tuple[np.ndarray, float]],
    n_classes: int,
    alpha: float,
    global_prior: np.ndarray,
) -> np.ndarray:
    out = np.empty((len(keys), n_classes + 3), dtype=np.float64)
    global_entropy = -float(
        np.sum(global_prior * np.log(np.clip(global_prior, 1e-15, 1.0)))
    )
    for i, key in enumerate(keys.astype(np.int64, copy=False)):
        row = table.get(int(key))
        if row is None:
            p = global_prior
            n = 0.0
        else:
            p, n = row
        entropy_i = -float(np.sum(p * np.log(np.clip(p, 1e-15, 1.0))))
        out[i, :n_classes] = p
        out[i, n_classes] = np.log1p(n)
        out[i, n_classes + 1] = n / (n + float(alpha))
        out[i, n_classes + 2] = global_entropy - entropy_i
    return out


def _pcf_apply_from_fit(
    keys_fit: np.ndarray,
    y_fit: np.ndarray,
    keys_apply: np.ndarray,
    n_classes: int,
    alpha: float,
    global_prior: np.ndarray,
) -> np.ndarray:
    try:
        import gtboost  # local Rust extension, optional during development

        rust_apply = getattr(gtboost, "pcf_posterior_apply", None)
    except Exception:
        rust_apply = None
    if rust_apply is not None:
        return np.asarray(
            rust_apply(
                np.asarray(keys_fit, dtype=np.int64),
                np.asarray(y_fit, dtype=np.int64),
                np.asarray(keys_apply, dtype=np.int64),
                int(n_classes),
                float(alpha),
                np.asarray(global_prior, dtype=np.float64).tolist(),
            ),
            dtype=np.float64,
        )
    table = _pcf_fit_table(keys_fit, y_fit, n_classes, alpha, global_prior)
    return _pcf_apply_table(keys_apply, table, n_classes, alpha, global_prior)


def _pcf_fit_table_regression(
    keys: np.ndarray,
    y: np.ndarray,
    alpha: float,
    global_mean: float,
) -> dict[int, tuple[float, float]]:
    uniq, inv = np.unique(keys.astype(np.int64, copy=False), return_inverse=True)
    counts = np.zeros(len(uniq), dtype=np.float64)
    sums = np.zeros(len(uniq), dtype=np.float64)
    np.add.at(counts, inv, 1.0)
    np.add.at(sums, inv, np.asarray(y, dtype=np.float64))
    means = (sums + float(alpha) * float(global_mean)) / (counts + float(alpha))
    return {int(k): (float(means[i]), float(counts[i])) for i, k in enumerate(uniq)}


def _pcf_apply_table_regression(
    keys: np.ndarray,
    table: dict[int, tuple[float, float]],
    alpha: float,
    global_mean: float,
) -> np.ndarray:
    out = np.empty((len(keys), 4), dtype=np.float64)
    for i, key in enumerate(keys.astype(np.int64, copy=False)):
        row = table.get(int(key))
        if row is None:
            mean = float(global_mean)
            n = 0.0
        else:
            mean, n = row
        out[i, 0] = mean
        out[i, 1] = mean - float(global_mean)
        out[i, 2] = np.log1p(n)
        out[i, 3] = n / (n + float(alpha))
    return out


def _pcf_apply_from_fit_regression(
    keys_fit: np.ndarray,
    y_fit: np.ndarray,
    keys_apply: np.ndarray,
    alpha: float,
    global_mean: float,
) -> np.ndarray:
    table = _pcf_fit_table_regression(keys_fit, y_fit, alpha, global_mean)
    return _pcf_apply_table_regression(keys_apply, table, alpha, global_mean)


def _pcf_oof_block(
    keys_fit: np.ndarray,
    y_fit: np.ndarray,
    apply_keys: list[np.ndarray],
    n_classes: int,
    alpha: float,
    n_folds: int,
    seed: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    from sklearn.model_selection import StratifiedKFold

    y_int = y_fit.astype(np.int64)
    class_counts = np.bincount(y_int, minlength=n_classes).astype(np.float64)
    global_prior = (class_counts + 1.0) / (class_counts.sum() + n_classes)
    min_class = int(np.min(class_counts[class_counts > 0])) if np.any(class_counts > 0) else 0
    if min_class < 2:
        oof = _pcf_apply_from_fit(
            keys_fit, y_int, keys_fit, n_classes, alpha, global_prior
        )
    else:
        folds = min(max(2, int(n_folds)), min_class)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))
        fold_ids = np.full(len(y_fit), -1, dtype=np.int64)
        for fold_id, (_tr, va) in enumerate(skf.split(keys_fit.reshape(-1, 1), y_int)):
            fold_ids[va] = int(fold_id)
        try:
            import gtboost

            rust_oof = getattr(gtboost, "pcf_posterior_oof_apply", None)
        except Exception:
            rust_oof = None
        if rust_oof is not None:
            first_keys = apply_keys[0] if apply_keys else keys_fit[:0]
            oof, first_apply = rust_oof(
                np.asarray(keys_fit, dtype=np.int64),
                y_int,
                fold_ids,
                np.asarray(first_keys, dtype=np.int64),
                int(n_classes),
                float(alpha),
                np.asarray(global_prior, dtype=np.float64).tolist(),
            )
            applied = []
            if apply_keys:
                applied.append(np.asarray(first_apply, dtype=np.float64))
                for keys in apply_keys[1:]:
                    applied.append(
                        _pcf_apply_from_fit(keys_fit, y_int, keys, n_classes, alpha, global_prior)
                    )
            return np.asarray(oof, dtype=np.float64), applied
        oof = np.zeros((len(y_fit), n_classes + 3), dtype=np.float64)
        for fold_id in range(folds):
            tr = fold_ids != fold_id
            va = fold_ids == fold_id
            oof[va] = _pcf_apply_from_fit(
                keys_fit[tr], y_int[tr], keys_fit[va], n_classes, alpha, global_prior
            )
    applied = [
        _pcf_apply_from_fit(keys_fit, y_int, keys, n_classes, alpha, global_prior)
        for keys in apply_keys
    ]
    return oof, applied


def _pcf_oof_block_regression(
    keys_fit: np.ndarray,
    y_fit: np.ndarray,
    apply_keys: list[np.ndarray],
    alpha: float,
    n_folds: int,
    seed: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    from sklearn.model_selection import KFold

    y_float = np.asarray(y_fit, dtype=np.float64)
    global_mean = float(np.mean(y_float)) if y_float.size else 0.0
    if len(y_float) < 4:
        oof = _pcf_apply_from_fit_regression(
            keys_fit, y_float, keys_fit, alpha, global_mean
        )
    else:
        folds = min(max(2, int(n_folds)), len(y_float))
        kf = KFold(n_splits=folds, shuffle=True, random_state=int(seed))
        oof = np.zeros((len(y_float), 4), dtype=np.float64)
        for tr, va in kf.split(keys_fit.reshape(-1, 1)):
            oof[va] = _pcf_apply_from_fit_regression(
                keys_fit[tr], y_float[tr], keys_fit[va], alpha, global_mean
            )
    applied = [
        _pcf_apply_from_fit_regression(keys_fit, y_float, keys, alpha, global_mean)
        for keys in apply_keys
    ]
    return oof, applied


def _pcf_project_block(
    block: np.ndarray,
    *,
    n_classes: int,
    global_prior: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Project raw PCF posterior block into a tree-friendly coordinate basis."""
    mode = str(mode or "current").lower()
    if mode in {"current", "raw", "legacy"}:
        return np.asarray(block, dtype=np.float64)
    if int(n_classes) != 2:
        return np.asarray(block, dtype=np.float64)

    b = np.asarray(block, dtype=np.float64)
    p1 = np.clip(b[:, 1], 1e-6, 1.0 - 1e-6)
    log_count = b[:, 2]
    reliability = b[:, 3]
    unseen = (log_count <= 0.0).astype(np.float64)
    if mode == "prob3":
        return np.column_stack([p1, log_count, reliability]).astype(np.float64)
    if mode == "prob4":
        return np.column_stack([p1, log_count, reliability, unseen]).astype(np.float64)
    if mode in {"drop_p0", "prob_entropy4"}:
        return np.column_stack([p1, log_count, reliability, b[:, 4]]).astype(np.float64)
    prior = float(np.clip(global_prior[1], 1e-6, 1.0 - 1e-6))
    logit_delta = np.log(p1 / (1.0 - p1)) - np.log(prior / (1.0 - prior))
    if mode == "logit2":
        return np.column_stack([logit_delta, log_count]).astype(np.float64)
    if mode == "logit3":
        return np.column_stack([logit_delta, log_count, reliability]).astype(np.float64)
    if mode == "logit4":
        return np.column_stack([logit_delta, log_count, reliability, unseen]).astype(np.float64)
    if mode == "logit_entropy4":
        return np.column_stack([logit_delta, log_count, reliability, b[:, 4]]).astype(np.float64)
    raise ValueError(f"unknown PCF coordinate_mode: {mode}")


def _pcf_binary_logit(p: np.ndarray | float) -> np.ndarray:
    p_arr = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p_arr / (1.0 - p_arr))


def _pcf_pact_nb_aggregate(
    blocks: Sequence[np.ndarray],
    arities: Sequence[int],
    *,
    global_prior: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """Compress PCF tuple blocks into parameter-free PACT/NB summaries.

    This is intentionally not a learned-teacher replacement. It uses the same
    leak-safe PCF posterior blocks, clips their logit lift against the global
    prior, weights by PCF reliability, and emits a small row-level summary by
    tuple arity. Rare tuples therefore contribute little without requiring a
    separate teacher or admission gate.
    """
    if not blocks:
        return np.zeros((0, 0), dtype=np.float64)
    n_rows = int(blocks[0].shape[0])
    prior_arr = np.asarray(global_prior, dtype=np.float64)
    if prior_arr.size > 2:
        return _pcf_pact_multiclass_aggregate(
            blocks,
            arities,
            global_prior=prior_arr,
            cfg=cfg,
        )
    prior = float(np.clip(prior_arr[1], 1e-6, 1.0 - 1e-6))
    prior_logit = float(_pcf_binary_logit(prior))
    clip_by_arity = {
        1: float(cfg.get("pact_clip_single", 3.0)),
        2: float(cfg.get("pact_clip_pair", 2.5)),
        3: float(cfg.get("pact_clip_triple", 2.0)),
    }

    states: dict[int, dict[str, np.ndarray | float]] = {}
    for arity in (1, 2, 3):
        states[arity] = {
            "sum": np.zeros(n_rows, dtype=np.float64),
            "rel_sum": np.zeros(n_rows, dtype=np.float64),
            "max": np.full(n_rows, -np.inf, dtype=np.float64),
            "min": np.full(n_rows, np.inf, dtype=np.float64),
            "max_rel": np.zeros(n_rows, dtype=np.float64),
            "unseen": np.zeros(n_rows, dtype=np.float64),
            "count": 0.0,
        }

    for block, arity_raw in zip(blocks, arities):
        arity = min(3, max(1, int(arity_raw)))
        b = np.asarray(block, dtype=np.float64)
        if b.ndim != 2 or b.shape[1] < 4:
            continue
        p1 = np.clip(b[:, 1], 1e-6, 1.0 - 1e-6)
        log_count = b[:, 2]
        rel = np.clip(b[:, 3], 0.0, 1.0)
        lift = _pcf_binary_logit(p1) - prior_logit
        clip = max(float(clip_by_arity.get(arity, 2.0)), 1e-6)
        weighted = rel * np.clip(lift, -clip, clip)
        state = states[arity]
        state["sum"] = state["sum"] + weighted
        state["rel_sum"] = state["rel_sum"] + rel
        state["max"] = np.maximum(state["max"], weighted)
        state["min"] = np.minimum(state["min"], weighted)
        state["max_rel"] = np.maximum(state["max_rel"], rel)
        state["unseen"] = state["unseen"] + (log_count <= 0.0).astype(np.float64)
        state["count"] = float(state["count"]) + 1.0

    total_sum = np.zeros(n_rows, dtype=np.float64)
    total_count = 0.0
    out_parts: list[np.ndarray] = []
    arity_parts: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for arity in (1, 2, 3):
        state = states[arity]
        count = float(state["count"])
        sum_v = np.asarray(state["sum"], dtype=np.float64)
        rel_sum = np.asarray(state["rel_sum"], dtype=np.float64)
        total_sum += sum_v
        total_count += count
        mean_v = np.divide(sum_v, np.maximum(rel_sum, 1e-12))
        max_v = np.asarray(state["max"], dtype=np.float64)
        min_v = np.asarray(state["min"], dtype=np.float64)
        max_v = np.where(np.isfinite(max_v), max_v, 0.0)
        min_v = np.where(np.isfinite(min_v), min_v, 0.0)
        max_rel = np.asarray(state["max_rel"], dtype=np.float64)
        unseen_rate = np.asarray(state["unseen"], dtype=np.float64) / max(count, 1.0)
        arity_parts[arity] = (mean_v, max_v, min_v, max_rel, unseen_rate)

    def nb_logit(sum_v: np.ndarray, count: float) -> np.ndarray:
        return np.clip(prior_logit + sum_v / np.sqrt(max(float(count), 1.0)), -10.0, 10.0)

    out_parts.append(nb_logit(total_sum, total_count))
    for arity in (1, 2, 3):
        out_parts.append(nb_logit(np.asarray(states[arity]["sum"], dtype=np.float64), float(states[arity]["count"])))
    for arity in (1, 2, 3):
        out_parts.extend(arity_parts[arity])
    return np.column_stack(out_parts).astype(np.float64)


def _pcf_pact_multiclass_aggregate(
    blocks: Sequence[np.ndarray],
    arities: Sequence[int],
    *,
    global_prior: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """Fixed-width multiclass posterior evidence.

    Each raw PCF block is a smoothed posterior for one categorical tuple. This
    compressor converts tuple posteriors into centered log-prior lifts and
    aggregates them by arity, so the downstream tree sees a small set of dense
    evidence coordinates instead of dozens of K+3 posterior blocks.
    """
    if not blocks:
        return np.zeros((0, 0), dtype=np.float64)
    n_rows = int(blocks[0].shape[0])
    prior = np.clip(np.asarray(global_prior, dtype=np.float64), 1e-6, 1.0)
    prior = prior / np.sum(prior)
    n_classes = int(prior.shape[0])
    clip = float(cfg.get("pact_clip_multiclass", 2.5))
    clip = max(clip, 1e-6)

    total = np.zeros((n_rows, n_classes), dtype=np.float64)
    rel_total = np.zeros(n_rows, dtype=np.float64)
    entropy_total = np.zeros(n_rows, dtype=np.float64)
    by_arity: dict[int, dict[str, np.ndarray | float]] = {}
    for arity in (1, 2, 3):
        by_arity[arity] = {
            "sum": np.zeros((n_rows, n_classes), dtype=np.float64),
            "rel": np.zeros(n_rows, dtype=np.float64),
            "entropy": np.zeros(n_rows, dtype=np.float64),
            "count": 0.0,
        }

    for block, arity_raw in zip(blocks, arities):
        b = np.asarray(block, dtype=np.float64)
        if b.ndim != 2 or b.shape[1] < n_classes + 2:
            continue
        arity = min(3, max(1, int(arity_raw)))
        p = np.clip(b[:, :n_classes], 1e-6, 1.0)
        p = p / np.maximum(np.sum(p, axis=1, keepdims=True), 1e-12)
        rel = np.clip(b[:, n_classes + 1], 0.0, 1.0)
        if b.shape[1] > n_classes + 2:
            entropy_gain = np.asarray(b[:, n_classes + 2], dtype=np.float64)
        else:
            entropy_gain = np.zeros(n_rows, dtype=np.float64)
        lift = np.log(p) - np.log(prior.reshape(1, -1))
        lift -= np.mean(lift, axis=1, keepdims=True)
        weighted = rel[:, None] * np.clip(lift, -clip, clip)

        total += weighted
        rel_total += rel
        entropy_total += entropy_gain
        state = by_arity[arity]
        state["sum"] = np.asarray(state["sum"], dtype=np.float64) + weighted
        state["rel"] = np.asarray(state["rel"], dtype=np.float64) + rel
        state["entropy"] = np.asarray(state["entropy"], dtype=np.float64) + entropy_gain
        state["count"] = float(state["count"]) + 1.0

    def centered_logits(x: np.ndarray, rel: np.ndarray) -> np.ndarray:
        z = x / np.sqrt(np.maximum(rel, 1.0))[:, None]
        z -= np.mean(z, axis=1, keepdims=True)
        return np.clip(z, -8.0, 8.0)

    parts: list[np.ndarray] = [centered_logits(total, rel_total)]
    for arity in (1, 2, 3):
        state = by_arity[arity]
        parts.append(centered_logits(np.asarray(state["sum"], dtype=np.float64), np.asarray(state["rel"], dtype=np.float64)))
    parts.append(np.log1p(rel_total).reshape(-1, 1))
    parts.append((entropy_total / np.maximum(rel_total, 1.0)).reshape(-1, 1))
    for arity in (1, 2, 3):
        state = by_arity[arity]
        rel = np.asarray(state["rel"], dtype=np.float64)
        count = max(float(state["count"]), 1.0)
        parts.append((rel / count).reshape(-1, 1))
    return np.column_stack(parts).astype(np.float64)


def _binary_logloss_from_prob(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def _multiclass_logloss_from_prob(y: np.ndarray, p: np.ndarray) -> float:
    y_int = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] != y_int.shape[0] or p.shape[1] < 2:
        return float("inf")
    rows = np.arange(y_int.shape[0])
    q = np.clip(p[rows, y_int], 1e-15, 1.0)
    return float(-np.mean(np.log(q)))


def _pcf_oof_logloss_utility(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    cols: Sequence[int],
    n_classes: int,
    alpha: float,
    folds: int,
    seed: int,
    hash_bins: int,
    global_prior: np.ndarray,
    keys_fit: Optional[np.ndarray] = None,
) -> float:
    n_classes = int(n_classes)
    if n_classes < 2:
        return 0.0
    if keys_fit is None:
        keys_fit = _pcf_key_matrix(X_fit, cols, hash_bins)
    fit_block, _applied = _pcf_oof_block(
        keys_fit,
        y_fit,
        [],
        n_classes,
        float(alpha),
        int(folds),
        int(seed),
    )
    if n_classes == 2:
        global_loss = _binary_logloss_from_prob(y_fit, np.full(len(y_fit), global_prior[1]))
        block_loss = _binary_logloss_from_prob(y_fit, fit_block[:, 1])
    else:
        global_prob = np.repeat(
            np.asarray(global_prior, dtype=np.float64).reshape(1, -1),
            len(y_fit),
            axis=0,
        )
        global_loss = _multiclass_logloss_from_prob(y_fit, global_prob)
        block_loss = _multiclass_logloss_from_prob(y_fit, fit_block[:, :n_classes])
    return float(global_loss - block_loss)


def _pcf_oof_regression_utility(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    cols: Sequence[int],
    alpha: float,
    folds: int,
    seed: int,
    hash_bins: int,
    keys_fit: Optional[np.ndarray] = None,
) -> float:
    y_float = np.asarray(y_fit, dtype=np.float64)
    if y_float.size == 0:
        return 0.0
    if keys_fit is None:
        keys_fit = _pcf_key_matrix(X_fit, cols, hash_bins)
    fit_block, _applied = _pcf_oof_block_regression(
        keys_fit,
        y_float,
        [],
        float(alpha),
        int(folds),
        int(seed),
    )
    global_mean = float(np.mean(y_float))
    base_mse = float(np.mean((y_float - global_mean) ** 2))
    block_mse = float(np.mean((y_float - fit_block[:, 0]) ** 2))
    return float(base_mse - block_mse)


def _pcf_select_tuple_candidates(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    candidates: list[tuple[int, ...]],
    max_keep: int,
    *,
    n_classes: int,
    alpha: float,
    folds: int,
    seed: int,
    hash_bins: int,
    global_prior: np.ndarray,
    selection: str,
    task_type: str = "classification",
    key_cache: Optional[dict[tuple[int, ...], np.ndarray]] = None,
) -> tuple[list[tuple[int, ...]], list[float]]:
    max_keep = max(0, int(max_keep))
    if max_keep == 0 or not candidates:
        return [], []
    selection = str(selection).lower()
    if selection not in {"oof_logloss", "utility"}:
        kept = candidates[:max_keep]
        return kept, [float("nan")] * len(kept)

    scored: list[tuple[float, tuple[int, ...]]] = []
    for i, cols in enumerate(candidates):
        key = tuple(int(c) for c in cols)
        if key_cache is not None and key in key_cache:
            keys_fit = key_cache[key]
        else:
            keys_fit = _pcf_key_matrix(X_fit, key, hash_bins)
            if key_cache is not None:
                key_cache[key] = keys_fit
        if str(task_type).lower() == "regression" or int(n_classes) <= 0:
            score = _pcf_oof_regression_utility(
                X_fit,
                y_fit,
                key,
                alpha,
                folds,
                seed + 1009 * (i + 1),
                hash_bins,
                keys_fit=keys_fit,
            )
        else:
            score = _pcf_oof_logloss_utility(
                X_fit,
                y_fit,
                key,
                n_classes,
                alpha,
                folds,
                seed + 1009 * (i + 1),
                hash_bins,
                global_prior,
                keys_fit=keys_fit,
            )
        scored.append((score, key))
    scored.sort(key=lambda item: item[0], reverse=True)
    kept_scored = scored[:max_keep]
    return [cols for _score, cols in kept_scored], [float(score) for score, _cols in kept_scored]


def _pcf_build_blocks(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    apply_mats: list[np.ndarray],
    cat_mask: Sequence[bool],
    n_classes: int,
    cfg: dict,
    seed: int,
) -> tuple[list[np.ndarray], list[list[np.ndarray]], dict]:
    max_cat = int(cfg.get("max_cat", 9))
    max_pairs = int(cfg.get("max_pairs", 36))
    max_triples = int(cfg.get("max_triples", 84))
    hash_bins = int(cfg.get("hash_bins", 1_048_576))
    folds = int(cfg.get("folds", 5))
    alpha = float(cfg.get("alpha", 80.0))
    pair_alpha = float(cfg.get("pair_alpha", 160.0))
    triple_alpha = float(cfg.get("triple_alpha", 300.0))
    tuple_selection = str(cfg.get("tuple_selection", "prefix"))
    coordinate_mode = str(cfg.get("coordinate_mode", "current"))
    aggregate_mode = str(cfg.get("aggregate_mode", "none")).lower()

    cat_cols = _pcf_selected_cat_columns(X_fit, cat_mask, max_cat)
    if int(n_classes) <= 0:
        if aggregate_mode not in {"none", "", "off", "false", "0"}:
            raise ValueError("regression PCF supports aggregate_mode='none'")
        y_float = np.asarray(y_fit, dtype=np.float64)
        global_mean = float(np.mean(y_float)) if y_float.size else 0.0
        global_std = float(np.std(y_float)) if y_float.size else 0.0
        global_prior = np.array([global_mean, global_std], dtype=np.float64)
        fit_blocks: list[np.ndarray] = []
        apply_blocks: list[list[np.ndarray]] = [[] for _ in apply_mats]
        block_arities: list[int] = []
        output_block_widths: list[int] = []
        output_block_arities: list[int] = []
        key_cache: dict[tuple[int, ...], np.ndarray] = {}

        def fit_keys_for(cols: Sequence[int]) -> np.ndarray:
            key = tuple(int(c) for c in cols)
            cached = key_cache.get(key)
            if cached is not None:
                return cached
            keys = _pcf_key_matrix(X_fit, key, hash_bins)
            key_cache[key] = keys
            return keys

        def add_block(cols: Sequence[int], block_alpha: float, block_seed: int) -> None:
            keys_fit = fit_keys_for(cols)
            keys_apply = [_pcf_key_matrix(Xa, cols, hash_bins) for Xa in apply_mats]
            fit_block, applied = _pcf_oof_block_regression(
                keys_fit, y_float, keys_apply, block_alpha, folds, block_seed
            )
            fit_blocks.append(fit_block)
            output_block_widths.append(int(fit_block.shape[1]))
            arity = len(tuple(cols))
            block_arities.append(arity)
            output_block_arities.append(arity)
            for out, block in zip(apply_blocks, applied):
                out.append(block)

        for col in cat_cols:
            add_block([col], alpha, seed)

        pair_candidates_all = [
            tuple(int(c) for c in cols) for cols in itertools.combinations(cat_cols, 2)
        ]
        pair_candidates, pair_scores = _pcf_select_tuple_candidates(
            X_fit,
            y_float,
            pair_candidates_all,
            max_pairs,
            n_classes=0,
            alpha=pair_alpha,
            folds=folds,
            seed=seed + 17,
            hash_bins=hash_bins,
            global_prior=global_prior,
            selection=tuple_selection,
            task_type="regression",
            key_cache=key_cache,
        )
        pair_keep = [
            (cols, score)
            for cols, score in zip(pair_candidates, pair_scores)
            if float(score) > 0.0
        ]
        pair_candidates = [cols for cols, _score in pair_keep]
        pair_scores = [float(score) for _cols, score in pair_keep]
        for cols in pair_candidates:
            add_block(cols, pair_alpha, seed + 17)

        triple_candidates_all = [
            tuple(int(c) for c in cols) for cols in itertools.combinations(cat_cols, 3)
        ]
        triple_candidates, triple_scores = _pcf_select_tuple_candidates(
            X_fit,
            y_float,
            triple_candidates_all,
            max_triples,
            n_classes=0,
            alpha=triple_alpha,
            folds=folds,
            seed=seed + 29,
            hash_bins=hash_bins,
            global_prior=global_prior,
            selection=tuple_selection,
            task_type="regression",
            key_cache=key_cache,
        )
        triple_keep = [
            (cols, score)
            for cols, score in zip(triple_candidates, triple_scores)
            if float(score) > 0.0
        ]
        triple_candidates = [cols for cols, _score in triple_keep]
        triple_scores = [float(score) for _cols, score in triple_keep]
        for cols in triple_candidates:
            add_block(cols, triple_alpha, seed + 29)

        meta = {
            "selected_cat_idx": [int(c) for c in cat_cols],
            "n_single_blocks": int(len(cat_cols)),
            "n_pair_blocks": int(len(pair_candidates)),
            "n_triple_blocks": int(len(triple_candidates)),
            "n_blocks": int(len(fit_blocks)),
            "n_pcf_features": int(sum(block.shape[1] for block in fit_blocks)),
            "pcf_block_widths": [int(w) for w in output_block_widths],
            "pcf_block_arities": [int(a) for a in output_block_arities],
            "coordinate_mode": "regression_mean4",
            "aggregate_mode": aggregate_mode,
            "tuple_selection": tuple_selection,
            "pair_scores_top": [float(x) for x in pair_scores[:5]],
            "triple_scores_top": [float(x) for x in triple_scores[:5]],
            "key_cache_blocks": int(len(key_cache)),
            "global_mean": global_mean,
            "global_std": global_std,
        }
        return fit_blocks, apply_blocks, meta

    class_counts = np.bincount(y_fit.astype(np.int64), minlength=n_classes).astype(np.float64)
    global_prior = (class_counts + 1.0) / (class_counts.sum() + n_classes)
    fit_blocks: list[np.ndarray] = []
    apply_blocks: list[list[np.ndarray]] = [[] for _ in apply_mats]
    block_arities: list[int] = []
    output_block_widths: list[int] = []
    output_block_arities: list[int] = []
    key_cache: dict[tuple[int, ...], np.ndarray] = {}

    def fit_keys_for(cols: Sequence[int]) -> np.ndarray:
        key = tuple(int(c) for c in cols)
        cached = key_cache.get(key)
        if cached is not None:
            return cached
        keys = _pcf_key_matrix(X_fit, key, hash_bins)
        key_cache[key] = keys
        return keys

    def add_block(cols: Sequence[int], block_alpha: float, block_seed: int) -> None:
        keys_fit = fit_keys_for(cols)
        keys_apply = [_pcf_key_matrix(Xa, cols, hash_bins) for Xa in apply_mats]
        fit_block, applied = _pcf_oof_block(
            keys_fit, y_fit, keys_apply, n_classes, block_alpha, folds, block_seed
        )
        if aggregate_mode != "pact_nb":
            fit_block = _pcf_project_block(
                fit_block,
                n_classes=n_classes,
                global_prior=global_prior,
                mode=coordinate_mode,
            )
            applied = [
                _pcf_project_block(
                    block,
                    n_classes=n_classes,
                    global_prior=global_prior,
                    mode=coordinate_mode,
                )
                for block in applied
            ]
        fit_blocks.append(fit_block)
        output_block_widths.append(int(fit_block.shape[1]))
        arity = len(tuple(cols))
        block_arities.append(arity)
        output_block_arities.append(arity)
        for out, block in zip(apply_blocks, applied):
            out.append(block)

    for col in cat_cols:
        add_block([col], alpha, seed)

    pair_candidates_all = [tuple(int(c) for c in cols) for cols in itertools.combinations(cat_cols, 2)]
    pair_candidates, pair_scores = _pcf_select_tuple_candidates(
        X_fit,
        y_fit,
        pair_candidates_all,
        max_pairs,
        n_classes=n_classes,
        alpha=pair_alpha,
        folds=folds,
        seed=seed + 17,
        hash_bins=hash_bins,
        global_prior=global_prior,
        selection=tuple_selection,
        key_cache=key_cache,
    )
    for cols in pair_candidates:
        add_block(cols, pair_alpha, seed + 17)

    triple_candidates_all = [tuple(int(c) for c in cols) for cols in itertools.combinations(cat_cols, 3)]
    triple_candidates, triple_scores = _pcf_select_tuple_candidates(
        X_fit,
        y_fit,
        triple_candidates_all,
        max_triples,
        n_classes=n_classes,
        alpha=triple_alpha,
        folds=folds,
        seed=seed + 29,
        hash_bins=hash_bins,
        global_prior=global_prior,
        selection=tuple_selection,
        key_cache=key_cache,
    )
    for cols in triple_candidates:
        add_block(cols, triple_alpha, seed + 29)

    if aggregate_mode == "pact_nb":
        fit_blocks = [
            _pcf_pact_nb_aggregate(
                fit_blocks,
                block_arities,
                global_prior=global_prior,
                cfg=cfg,
            )
        ]
        output_block_widths = [int(fit_blocks[0].shape[1])]
        output_block_arities = [0]
        apply_blocks = [
            [
                _pcf_pact_nb_aggregate(
                    blocks,
                    block_arities,
                    global_prior=global_prior,
                    cfg=cfg,
                )
            ]
            for blocks in apply_blocks
        ]
    elif aggregate_mode not in {"none", "", "off", "false", "0"}:
        raise ValueError(f"unknown PCF aggregate_mode: {aggregate_mode}")

    meta = {
        "selected_cat_idx": [int(c) for c in cat_cols],
        "n_single_blocks": int(len(cat_cols)),
        "n_pair_blocks": int(len(pair_candidates)),
        "n_triple_blocks": int(len(triple_candidates)),
        "n_blocks": int(len(fit_blocks)),
        "n_pcf_features": int(sum(block.shape[1] for block in fit_blocks)),
        "pcf_block_widths": [int(w) for w in output_block_widths],
        "pcf_block_arities": [int(a) for a in output_block_arities],
        "coordinate_mode": coordinate_mode,
        "aggregate_mode": aggregate_mode,
        "tuple_selection": tuple_selection,
        "pair_scores_top": [float(x) for x in pair_scores[:5]],
        "triple_scores_top": [float(x) for x in triple_scores[:5]],
        "key_cache_blocks": int(len(key_cache)),
    }
    return fit_blocks, apply_blocks, meta


def _pcf_feature_view_groups(
    *,
    n_original_features: int,
    numeric_cols: Sequence[int],
    pcf_block_widths: Sequence[int],
    pcf_block_arities: Sequence[int] | None = None,
    view: str,
    mode: str,
) -> list[int]:
    """Return optional group ids for PCF block-wise feature sampling.

    Group 0 is the raw/numeric base view expected by the Rust feature-view
    sampler. Each PCF tuple block gets one auxiliary group id. This prevents a
    multi-coordinate tuple block from getting several independent chances under
    colsample and lets the booster cycle tuple views across trees.
    """
    mode_norm = str(mode or "off").lower()
    if mode_norm in {"0", "false", "off", "none", ""}:
        if view == "pcf_only":
            return [0] * int(sum(pcf_block_widths))
        if view == "pcf_append":
            return [0] * int(n_original_features + sum(pcf_block_widths))
        return [0] * int(len(numeric_cols) + sum(pcf_block_widths))

    groups: list[int] = []
    if view == "pcf_append":
        groups.extend([0] * int(n_original_features))
    elif view == "pcf_replace_cats":
        groups.extend([0] * int(len(numeric_cols)))
    elif view == "pcf_only":
        # The Rust group sampler intentionally falls back unless group 0 is
        # present. Keep PCF-only ungrouped; there is no raw anchor view.
        return [0] * int(sum(pcf_block_widths))
    else:
        return []

    next_group = 1
    tuple_to_group: dict[int, int] = {}
    arities = list(pcf_block_arities or [])
    for block_idx, width in enumerate(pcf_block_widths):
        w = int(width)
        if w <= 0:
            continue
        if mode_norm == "arity":
            arity = int(arities[block_idx]) if block_idx < len(arities) else 1
            group_id = max(1, min(3, arity))
        else:
            group_id = tuple_to_group.setdefault(block_idx, next_group)
            if group_id == next_group:
                next_group += 1
        groups.extend([group_id] * w)
    return groups


def _pcf_assemble_view(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    apply_mats: list[np.ndarray],
    cat_mask: Sequence[bool],
    task_type: str,
    n_classes: int,
    cfg: dict,
    seed: int,
) -> tuple[np.ndarray, list[np.ndarray], list[bool], dict]:
    if task_type not in {"binary", "multiclass", "regression"}:
        raise ValueError("PCF geometry requires a supervised task")
    if task_type in {"binary", "multiclass"} and int(n_classes) < 2:
        raise ValueError("classification PCF geometry requires at least two classes")
    build_n_classes = 0 if task_type == "regression" else int(n_classes)
    fit_blocks, apply_blocks, meta = _pcf_build_blocks(
        X_fit, y_fit, apply_mats, cat_mask, build_n_classes, cfg, seed
    )
    if not fit_blocks:
        raise ValueError("PCF geometry produced no posterior blocks")
    pcf_fit = np.hstack(fit_blocks).astype(np.float64)
    pcf_apply = [np.hstack(blocks).astype(np.float64) for blocks in apply_blocks]
    view = str(cfg.get("view", "pcf_replace_cats"))
    numeric_cols = [i for i, is_cat in enumerate(cat_mask) if not is_cat]
    if view == "pcf_only":
        out_fit = pcf_fit
        out_apply = pcf_apply
    elif view == "pcf_append":
        out_fit = np.hstack([X_fit, pcf_fit]).astype(np.float64)
        out_apply = [
            np.hstack([Xa, pa]).astype(np.float64)
            for Xa, pa in zip(apply_mats, pcf_apply)
        ]
    elif view == "pcf_replace_cats":
        if numeric_cols:
            out_fit = np.hstack([X_fit[:, numeric_cols], pcf_fit]).astype(np.float64)
            out_apply = [
                np.hstack([Xa[:, numeric_cols], pa]).astype(np.float64)
                for Xa, pa in zip(apply_mats, pcf_apply)
            ]
        else:
            out_fit = pcf_fit
            out_apply = pcf_apply
    else:
        raise ValueError(f"unknown PCF view: {view}")
    meta["view"] = view
    meta["n_numeric_kept"] = int(len(numeric_cols) if view == "pcf_replace_cats" else 0)
    meta["n_output_features"] = int(out_fit.shape[1])
    group_cfg = cfg.get("pcf_group_colsample", False)
    if isinstance(group_cfg, str):
        group_mode = group_cfg
    elif bool(group_cfg):
        group_mode = str(cfg.get("pcf_group_colsample_mode", "tuple"))
    else:
        group_mode = "off"
    meta["feature_view_groups"] = _pcf_feature_view_groups(
        n_original_features=int(X_fit.shape[1]),
        numeric_cols=numeric_cols,
        pcf_block_widths=meta.get("pcf_block_widths", []),
        pcf_block_arities=meta.get("pcf_block_arities", []),
        view=view,
        mode=group_mode,
    )
    if view == "pcf_append":
        out_cat_mask = list(cat_mask) + [False] * int(pcf_fit.shape[1])
    else:
        out_cat_mask = [False] * int(out_fit.shape[1])
    return out_fit, out_apply, out_cat_mask, meta


class PCFGeometryRuntime:
    """Stateful PCF transformer for the public GTBoostModel API.

    This is the production-shaped wrapper around the Rust PCF kernels. It
    creates private cross-fit folds inside ``fit`` so train rows never see their
    own target-derived posterior, then stores full-train posterior tables for
    future ``predict`` calls.
    """

    def __init__(
        self,
        *,
        task_type: str = "binary",
        n_classes: int = 2,
        config: Optional[dict] = None,
        seed: int = 42,
        fallback_raw: bool = True,
    ):
        self.task_type = task_type
        self.n_classes = int(n_classes)
        self.config = dict(config or {})
        self.seed = int(seed)
        self.fallback_raw = bool(fallback_raw)
        self.enabled = False
        self.meta_: dict = {}
        self.cat_mask_: list[bool] | None = None
        self.out_cat_mask_: list[bool] | None = None
        self.numeric_cols_: list[int] = []
        self.view_: str = "pcf_replace_cats"
        self.coordinate_mode_: str = "current"
        self.aggregate_mode_: str = "none"
        self.hash_bins_: int = 1_048_576
        self.blocks_: list[dict] = []
        self.y_fit_: np.ndarray | None = None
        self.global_prior_: np.ndarray | None = None

    def _cfg(self) -> dict:
        cfg = dict(self.config)
        profile = str(cfg.get("profile", "") or "").lower()
        if profile in {"mc_fast", "fast_mc", "multiclass_fast", "lite"} and self.task_type == "multiclass":
            cfg.setdefault("view", "pcf_append")
            cfg.setdefault("max_cat", 9)
            cfg.setdefault("max_pairs", 6)
            cfg.setdefault("max_triples", 0)
            cfg.setdefault("hash_bins", 8192)
            cfg.setdefault("alpha", 24.0)
            cfg.setdefault("pair_alpha", 64.0)
            cfg.setdefault("triple_alpha", 128.0)
            cfg.setdefault("folds", int(cfg.get("internal_folds", 3)))
            cfg.setdefault("coordinate_mode", "current")
            cfg.setdefault("aggregate_mode", "pact_nb")
            cfg.setdefault("tuple_selection", "oof_logloss")
            cfg.setdefault("eligibility_gate", True)
            cfg.setdefault("min_cat_features", 2)
            cfg.setdefault("min_max_cardinality", 2)
            cfg.setdefault("min_repeated_levels", 4)
            cfg.setdefault("min_entropy_eff", 1.5)
        elif profile in {"mc_blocks", "multiclass_blocks", "fast_mc_blocks"} and self.task_type == "multiclass":
            cfg.setdefault("view", "pcf_append")
            cfg.setdefault("max_cat", 9)
            cfg.setdefault("max_pairs", 6)
            cfg.setdefault("max_triples", 0)
            cfg.setdefault("hash_bins", 8192)
            cfg.setdefault("alpha", 24.0)
            cfg.setdefault("pair_alpha", 64.0)
            cfg.setdefault("triple_alpha", 128.0)
            cfg.setdefault("folds", int(cfg.get("internal_folds", 3)))
            cfg.setdefault("coordinate_mode", "current")
            cfg.setdefault("aggregate_mode", "none")
            cfg.setdefault("tuple_selection", "oof_logloss")
            cfg.setdefault("eligibility_gate", True)
            cfg.setdefault("min_cat_features", 2)
            cfg.setdefault("min_max_cardinality", 2)
            cfg.setdefault("min_repeated_levels", 4)
            cfg.setdefault("min_entropy_eff", 1.5)
        cfg.setdefault(
            "view",
            "pcf_append" if self.task_type == "regression" else "pcf_replace_cats",
        )
        cfg.setdefault("max_cat", 9)
        cfg.setdefault("max_pairs", 36)
        cfg.setdefault("max_triples", 84)
        cfg.setdefault("hash_bins", 1_048_576)
        cfg.setdefault("alpha", 80.0)
        cfg.setdefault("pair_alpha", 160.0)
        cfg.setdefault("triple_alpha", 300.0)
        # Public API users should not have to think about folds. Keep the old
        # ``folds`` key as a compatibility alias for experiment configs.
        cfg.setdefault("folds", int(cfg.get("internal_folds", 5)))
        cfg.setdefault("coordinate_mode", "current")
        cfg.setdefault("aggregate_mode", "none")
        cfg.setdefault("eligibility_gate", True)
        cfg.setdefault(
            "pcf_group_colsample",
            int(os.environ.get("GTBOOST_PCF_GROUP_COLSAMPLE", "0")) == 1,
        )
        cfg.setdefault(
            "pcf_group_colsample_mode",
            os.environ.get("GTBOOST_PCF_GROUP_COLSAMPLE_MODE", "tuple"),
        )
        return cfg

    def _raw_outputs(self, X: np.ndarray, cat_mask: Sequence[bool], reason: str):
        self.enabled = False
        self.cat_mask_ = list(cat_mask)
        self.out_cat_mask_ = list(cat_mask)
        self.meta_ = {"enabled": False, "reason": reason}
        return X, [], list(cat_mask), self.meta_

    def _assemble(self, X: np.ndarray, pcf: np.ndarray) -> np.ndarray:
        if self.view_ == "pcf_only":
            return pcf.astype(np.float64, copy=False)
        if self.view_ == "pcf_append":
            return np.hstack([X, pcf]).astype(np.float64)
        if self.view_ == "pcf_replace_cats":
            if self.numeric_cols_:
                return np.hstack([X[:, self.numeric_cols_], pcf]).astype(np.float64)
            return pcf.astype(np.float64, copy=False)
        raise ValueError(f"unknown PCF view: {self.view_}")

    def fit_transform(
        self,
        X_fit: np.ndarray,
        y_fit: np.ndarray,
        cat_mask: Sequence[bool],
        *,
        apply_mats: Optional[Sequence[np.ndarray]] = None,
    ) -> tuple[np.ndarray, list[np.ndarray], list[bool], dict]:
        X_fit = np.asarray(X_fit, dtype=np.float64)
        is_regression = str(self.task_type).lower() == "regression"
        y_arr = np.asarray(y_fit, dtype=np.float64)
        y_int = None if is_regression else np.asarray(y_fit, dtype=np.int64)
        apply_list = [np.asarray(x, dtype=np.float64) for x in (apply_mats or [])]
        cat_mask = list(cat_mask)

        if self.task_type not in {"binary", "multiclass", "regression"} or (
            not is_regression and self.n_classes < 2
        ):
            if self.fallback_raw:
                return self._raw_outputs(
                    X_fit,
                    cat_mask,
                    "PCF runtime requires a supervised classification/regression task",
                )
            raise ValueError("PCF runtime requires a supervised classification/regression task")
        if len(cat_mask) != X_fit.shape[1]:
            raise ValueError(
                f"cat_features length {len(cat_mask)} does not match X width {X_fit.shape[1]}"
            )

        cfg = self._cfg()
        ok, support = _pcf_eligible(X_fit, cat_mask, cfg)
        if bool(cfg.get("eligibility_gate", True)) and not ok:
            if self.fallback_raw:
                return self._raw_outputs(X_fit, cat_mask, support.get("reason", "PCF ineligible"))
            raise ValueError(support.get("reason", "PCF ineligible"))

        max_cat = int(cfg.get("max_cat", 9))
        max_pairs = int(cfg.get("max_pairs", 36))
        max_triples = int(cfg.get("max_triples", 84))
        self.hash_bins_ = int(cfg.get("hash_bins", 1_048_576))
        self.view_ = str(cfg.get("view", "pcf_replace_cats"))
        self.coordinate_mode_ = str(cfg.get("coordinate_mode", "current"))
        self.aggregate_mode_ = str(cfg.get("aggregate_mode", "none")).lower()
        folds = int(cfg.get("folds", 5))
        alpha = float(cfg.get("alpha", 80.0))
        pair_alpha = float(cfg.get("pair_alpha", 160.0))
        triple_alpha = float(cfg.get("triple_alpha", 300.0))
        tuple_selection = str(cfg.get("tuple_selection", "prefix"))

        cat_cols = _pcf_selected_cat_columns(X_fit, cat_mask, max_cat)
        tuple_defs: list[tuple[tuple[int, ...], float, int]] = []
        tuple_defs.extend(((int(c),), alpha, self.seed) for c in cat_cols)
        if is_regression:
            global_for_selection = np.array(
                [
                    float(np.mean(y_arr)) if y_arr.size else 0.0,
                    float(np.std(y_arr)) if y_arr.size else 0.0,
                ],
                dtype=np.float64,
            )
            select_n_classes = 0
            select_y = y_arr
            select_task = "regression"
        else:
            class_counts_for_selection = np.bincount(
                y_int, minlength=self.n_classes
            ).astype(np.float64)
            global_for_selection = (class_counts_for_selection + 1.0) / (
                len(y_int) + self.n_classes
            )
            select_n_classes = self.n_classes
            select_y = y_int
            select_task = "classification"
        pair_candidates_all = [
            tuple(int(c) for c in cols) for cols in itertools.combinations(cat_cols, 2)
        ]
        pair_candidates, pair_scores = _pcf_select_tuple_candidates(
            X_fit,
            select_y,
            pair_candidates_all,
            max_pairs,
            n_classes=select_n_classes,
            alpha=pair_alpha,
            folds=folds,
            seed=self.seed + 17,
            hash_bins=self.hash_bins_,
            global_prior=global_for_selection,
            selection=tuple_selection,
            task_type=select_task,
        )
        if is_regression:
            pair_keep = [
                (cols, score)
                for cols, score in zip(pair_candidates, pair_scores)
                if float(score) > 0.0
            ]
            pair_candidates = [cols for cols, _score in pair_keep]
            pair_scores = [float(score) for _cols, score in pair_keep]
        tuple_defs.extend(
            (tuple(int(c) for c in cols), pair_alpha, self.seed + 17)
            for cols in pair_candidates
        )
        triple_candidates_all = [
            tuple(int(c) for c in cols) for cols in itertools.combinations(cat_cols, 3)
        ]
        triple_candidates, triple_scores = _pcf_select_tuple_candidates(
            X_fit,
            select_y,
            triple_candidates_all,
            max_triples,
            n_classes=select_n_classes,
            alpha=triple_alpha,
            folds=folds,
            seed=self.seed + 29,
            hash_bins=self.hash_bins_,
            global_prior=global_for_selection,
            selection=tuple_selection,
            task_type=select_task,
        )
        if is_regression:
            triple_keep = [
                (cols, score)
                for cols, score in zip(triple_candidates, triple_scores)
                if float(score) > 0.0
            ]
            triple_candidates = [cols for cols, _score in triple_keep]
            triple_scores = [float(score) for _cols, score in triple_keep]
        tuple_defs.extend(
            (tuple(int(c) for c in cols), triple_alpha, self.seed + 29)
            for cols in triple_candidates
        )
        if not tuple_defs:
            if self.fallback_raw:
                return self._raw_outputs(X_fit, cat_mask, "PCF selected no categorical tuples")
            raise ValueError("PCF selected no categorical tuples")

        if is_regression:
            self.global_prior_ = global_for_selection
            self.y_fit_ = y_arr.copy()
        else:
            class_counts = np.bincount(y_int, minlength=self.n_classes).astype(np.float64)
            self.global_prior_ = (class_counts + 1.0) / (class_counts.sum() + self.n_classes)
            self.y_fit_ = y_int.copy()
        self.cat_mask_ = list(cat_mask)
        self.numeric_cols_ = [i for i, is_cat in enumerate(cat_mask) if not is_cat]
        self.blocks_ = []
        fit_blocks: list[np.ndarray] = []
        apply_blocks: list[list[np.ndarray]] = [[] for _ in apply_list]
        block_arities: list[int] = []
        output_block_widths: list[int] = []
        output_block_arities: list[int] = []

        for cols, block_alpha, block_seed in tuple_defs:
            keys_fit = _pcf_key_matrix(X_fit, cols, self.hash_bins_)
            keys_apply = [_pcf_key_matrix(Xa, cols, self.hash_bins_) for Xa in apply_list]
            if is_regression:
                fit_block, applied = _pcf_oof_block_regression(
                    keys_fit,
                    y_arr,
                    keys_apply,
                    block_alpha,
                    folds,
                    int(block_seed),
                )
            else:
                fit_block, applied = _pcf_oof_block(
                    keys_fit,
                    y_int,
                    keys_apply,
                    self.n_classes,
                    block_alpha,
                    folds,
                    int(block_seed),
                )
            if self.aggregate_mode_ != "pact_nb" and not is_regression:
                fit_block = _pcf_project_block(
                    fit_block,
                    n_classes=self.n_classes,
                    global_prior=self.global_prior_,
                    mode=self.coordinate_mode_,
                )
                applied = [
                    _pcf_project_block(
                        block,
                        n_classes=self.n_classes,
                        global_prior=self.global_prior_,
                        mode=self.coordinate_mode_,
                    )
                    for block in applied
                ]
            elif self.aggregate_mode_ == "pact_nb" and is_regression:
                raise ValueError("regression PCF supports aggregate_mode='none'")
            fit_blocks.append(fit_block)
            output_block_widths.append(int(fit_block.shape[1]))
            arity = len(tuple(cols))
            block_arities.append(arity)
            output_block_arities.append(arity)
            for out, block in zip(apply_blocks, applied):
                out.append(block)
            self.blocks_.append({
                "cols": tuple(int(c) for c in cols),
                "alpha": float(block_alpha),
                "keys_fit": np.asarray(keys_fit, dtype=np.int64),
            })

        if self.aggregate_mode_ == "pact_nb":
            pcf_fit = _pcf_pact_nb_aggregate(
                fit_blocks,
                block_arities,
                global_prior=self.global_prior_,
                cfg=cfg,
            )
            output_block_widths = [int(pcf_fit.shape[1])]
            output_block_arities = [0]
            pcf_apply = [
                _pcf_pact_nb_aggregate(
                    blocks,
                    block_arities,
                    global_prior=self.global_prior_,
                    cfg=cfg,
                )
                for blocks in apply_blocks
            ]
        elif self.aggregate_mode_ in {"none", "", "off", "false", "0"}:
            pcf_fit = np.hstack(fit_blocks).astype(np.float64)
            pcf_apply = [np.hstack(blocks).astype(np.float64) for blocks in apply_blocks]
        else:
            raise ValueError(f"unknown PCF aggregate_mode: {self.aggregate_mode_}")
        X_out = self._assemble(X_fit, pcf_fit)
        apply_out = [self._assemble(Xa, pa) for Xa, pa in zip(apply_list, pcf_apply)]
        if self.view_ == "pcf_append":
            self.out_cat_mask_ = list(cat_mask) + [False] * int(pcf_fit.shape[1])
        else:
            self.out_cat_mask_ = [False] * int(X_out.shape[1])
        self.enabled = True
        self.meta_ = {
            "enabled": True,
            "reason": "PCF geometry enabled",
            "support": support,
            "selected_cat_idx": [int(c) for c in cat_cols],
            "n_single_blocks": int(len(cat_cols)),
            "n_pair_blocks": int(len(pair_candidates)),
            "n_triple_blocks": int(len(triple_candidates)),
            "n_blocks": int(len(self.blocks_)),
            "n_pcf_features": int(pcf_fit.shape[1]),
            "pcf_block_widths": [int(w) for w in output_block_widths],
            "pcf_block_arities": [int(a) for a in output_block_arities],
            "n_numeric_kept": int(len(self.numeric_cols_) if self.view_ == "pcf_replace_cats" else 0),
            "n_output_features": int(X_out.shape[1]),
            "feature_view_groups": _pcf_feature_view_groups(
                n_original_features=int(X_fit.shape[1]),
                numeric_cols=self.numeric_cols_,
                pcf_block_widths=output_block_widths,
                pcf_block_arities=output_block_arities,
                view=self.view_,
                mode=(
                    str(cfg.get("pcf_group_colsample", "off"))
                    if isinstance(cfg.get("pcf_group_colsample", False), str)
                    else str(cfg.get("pcf_group_colsample_mode", "tuple"))
                    if bool(cfg.get("pcf_group_colsample", False))
                    else "off"
                ),
            ),
            "view": self.view_,
            "coordinate_mode": self.coordinate_mode_,
            "aggregate_mode": self.aggregate_mode_,
            "internal_folds": int(folds),
            "tuple_selection": tuple_selection,
            "pair_scores_top": [float(x) for x in pair_scores[:5]],
            "triple_scores_top": [float(x) for x in triple_scores[:5]],
        }
        return X_out, apply_out, list(self.out_cat_mask_), dict(self.meta_)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if not self.enabled:
            return X
        if self.y_fit_ is None or self.global_prior_ is None:
            raise RuntimeError("PCFGeometryRuntime is not fitted")
        blocks = []
        arities = []
        for block in self.blocks_:
            keys_apply = _pcf_key_matrix(X, block["cols"], self.hash_bins_)
            if self.task_type == "regression":
                raw_block = _pcf_apply_from_fit_regression(
                    block["keys_fit"],
                    self.y_fit_,
                    keys_apply,
                    float(block["alpha"]),
                    float(self.global_prior_[0]),
                )
            else:
                raw_block = _pcf_apply_from_fit(
                    block["keys_fit"],
                    self.y_fit_,
                    keys_apply,
                    self.n_classes,
                    float(block["alpha"]),
                    self.global_prior_,
                )
            arities.append(len(tuple(block["cols"])))
            if self.aggregate_mode_ == "pact_nb":
                blocks.append(raw_block)
            elif self.task_type == "regression":
                blocks.append(raw_block)
            else:
                blocks.append(
                    _pcf_project_block(
                        raw_block,
                        n_classes=self.n_classes,
                        global_prior=self.global_prior_,
                        mode=self.coordinate_mode_,
                    )
                )
        if self.aggregate_mode_ == "pact_nb":
            pcf = _pcf_pact_nb_aggregate(
                blocks,
                arities,
                global_prior=self.global_prior_,
                cfg=self._cfg(),
            )
        else:
            pcf = np.hstack(blocks).astype(np.float64)
        return self._assemble(X, pcf)


def _pcf_hgbt_gate(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    cat_mask: Sequence[bool],
    task_type: str,
    n_classes: int,
    cfg: dict,
    seed: int,
) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedShuffleSplit

    if task_type != "binary" or int(n_classes) != 2:
        return {"accepted": False, "reason": "PCF gate is binary-only"}

    repeats = max(1, int(cfg.get("outer_gate_repeats", 3)))
    holdout_size = float(cfg.get("outer_gate_size", 0.25))
    rounds = max(20, int(cfg.get("outer_gate_rounds", 500)))
    min_rel = float(cfg.get("outer_gate_min_rel", 0.002))
    win_frac = float(cfg.get("outer_gate_win_frac", 0.67))
    max_bad_rel = float(cfg.get("outer_gate_max_bad_rel", 0.01))
    splitter = StratifiedShuffleSplit(
        n_splits=repeats,
        test_size=holdout_size,
        random_state=int(seed) + 991,
    )
    raw_scores = []
    pcf_scores = []
    for rep, (idx_build, idx_hold) in enumerate(splitter.split(X_tr, y_tr)):
        X_b, y_b = X_tr[idx_build], y_tr[idx_build]
        X_h, y_h = X_tr[idx_hold], y_tr[idx_hold]
        raw_model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=float(cfg.get("gate_lr", 0.02)),
            max_iter=rounds,
            max_leaf_nodes=int(cfg.get("gate_max_leaf_nodes", 31)),
            l2_regularization=float(cfg.get("gate_l2", 0.1)),
            min_samples_leaf=int(cfg.get("gate_min_leaf", 20)),
            random_state=int(seed) + rep,
        )
        raw_model.fit(X_b, y_b)
        raw_pred = raw_model.predict_proba(X_h)[:, 1]
        raw_err = 1.0 - roc_auc_score(y_h, raw_pred)

        try:
            Xp_b, (Xp_h,), _mask, _meta = _pcf_assemble_view(
                X_b, y_b, [X_h], cat_mask, task_type, n_classes,
                cfg, int(seed) + rep * 101,
            )
        except Exception as exc:
            return {"accepted": False, "reason": f"PCF gate build failed: {exc}"}
        pcf_model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=float(cfg.get("gate_lr", 0.02)),
            max_iter=rounds,
            max_leaf_nodes=int(cfg.get("gate_max_leaf_nodes", 31)),
            l2_regularization=float(cfg.get("gate_l2", 0.1)),
            min_samples_leaf=int(cfg.get("gate_min_leaf", 20)),
            random_state=int(seed) + rep,
        )
        pcf_model.fit(Xp_b, y_b)
        pcf_pred = pcf_model.predict_proba(Xp_h)[:, 1]
        pcf_err = 1.0 - roc_auc_score(y_h, pcf_pred)
        raw_scores.append(float(raw_err))
        pcf_scores.append(float(pcf_err))

    raw_arr = np.asarray(raw_scores, dtype=np.float64)
    pcf_arr = np.asarray(pcf_scores, dtype=np.float64)
    delta = raw_arr - pcf_arr
    wins = int(np.sum(pcf_arr <= raw_arr * (1.0 - min_rel)))
    max_bad = float(np.max((pcf_arr - raw_arr) / np.maximum(raw_arr, 1e-12)))
    accepted = (
        float(np.mean(pcf_arr)) <= float(np.mean(raw_arr)) * (1.0 - min_rel)
        and wins >= max(1, int(np.ceil(repeats * win_frac)))
        and max_bad <= max_bad_rel
    )
    return {
        "accepted": bool(accepted),
        "reason": "accepted" if accepted else "outer HGBT critic rejected",
        "raw_mean": float(np.mean(raw_arr)),
        "pcf_mean": float(np.mean(pcf_arr)),
        "mean_delta": float(np.mean(delta)),
        "deltas": [float(x) for x in delta.tolist()],
        "wins": wins,
        "repeats": repeats,
        "max_bad_rel": max_bad,
    }


def build_pcf_geometry_views(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    cat_mask: Sequence[bool],
    splits,
    task_type: str,
    n_classes: int,
    *,
    config: Optional[dict] = None,
):
    """Build fold-local PCF views for fair HPO.

    Train rows are OOF-encoded inside every training fold. Validation/test rows
    use only the corresponding fold's training rows. If the optional outer gate
    rejects, the returned dict has ``enabled=False`` and should be ignored.
    """
    cfg = dict(config or {})
    cfg.setdefault(
        "pcf_group_colsample",
        int(os.environ.get("GTBOOST_PCF_GROUP_COLSAMPLE", "0")) == 1,
    )
    cfg.setdefault(
        "pcf_group_colsample_mode",
        os.environ.get("GTBOOST_PCF_GROUP_COLSAMPLE_MODE", "tuple"),
    )
    seed = int(cfg.get("seed", 42))
    if task_type not in {"binary", "multiclass", "regression"} or (
        task_type in {"binary", "multiclass"} and int(n_classes) < 2
    ):
        return {
            "enabled": False,
            "gate_info": {
                "accepted": False,
                "reason": "PCF requires a supervised classification/regression task",
            },
        }
    ok, support = _pcf_eligible(X_tr, cat_mask, cfg)
    if bool(cfg.get("eligibility_gate", True)) and not ok:
        return {"enabled": False, "gate_info": support}
    if int(cfg.get("outer_gate", 1)) == 1 and task_type == "binary" and int(n_classes) == 2:
        if int(cfg.get("outer_gate_screen", 0)) == 1:
            screen_cfg = dict(cfg)
            screen_cfg["max_pairs"] = int(cfg.get("outer_gate_screen_pairs", 12))
            screen_cfg["max_triples"] = int(cfg.get("outer_gate_screen_triples", 0))
            screen_cfg["outer_gate_rounds"] = int(
                cfg.get("outer_gate_screen_rounds", min(int(cfg.get("outer_gate_rounds", 500)), 160))
            )
            screen_cfg["outer_gate_repeats"] = int(
                cfg.get("outer_gate_screen_repeats", min(int(cfg.get("outer_gate_repeats", 3)), 2))
            )
            gate_info = _pcf_hgbt_gate(
                X_tr, y_tr, cat_mask, task_type, n_classes, screen_cfg, seed
            )
            gate_info["screen_gate"] = True
            if not gate_info.get("accepted", False):
                gate_info["support"] = support
                return {"enabled": False, "gate_info": gate_info}
            if int(cfg.get("outer_gate_full_after_screen", 0)) == 1:
                full_info = _pcf_hgbt_gate(
                    X_tr, y_tr, cat_mask, task_type, n_classes, cfg, seed + 17
                )
                full_info["screen_gate"] = gate_info
                gate_info = full_info
            else:
                gate_info["reason"] = "accepted by cheap PCF screen"
        else:
            gate_info = _pcf_hgbt_gate(X_tr, y_tr, cat_mask, task_type, n_classes, cfg, seed)
        if not gate_info.get("accepted", False):
            gate_info["support"] = support
            return {"enabled": False, "gate_info": gate_info}
    elif int(cfg.get("outer_gate", 1)) == 1:
        gate_info = {
            "accepted": True,
            "reason": "outer_gate skipped for non-binary PCF",
            "support": support,
        }
    else:
        gate_info = {"accepted": True, "reason": "outer_gate disabled", "support": support}

    full_train, (full_test,), mask, meta = _pcf_assemble_view(
        X_tr, y_tr, [X_te], cat_mask, task_type, n_classes, cfg, seed
    )
    fold_views = []
    for fold_idx, (tr_idx, va_idx) in enumerate(splits):
        X_fit, y_fit = X_tr[tr_idx], y_tr[tr_idx]
        X_val = X_tr[va_idx]
        X_fit_view, (X_val_view, X_test_view), fold_mask, _fold_meta = _pcf_assemble_view(
            X_fit,
            y_fit,
            [X_val, X_te],
            cat_mask,
            task_type,
            n_classes,
            cfg,
            seed + 1009 * (fold_idx + 1),
        )
        if len(fold_mask) != len(mask):
            raise ValueError("PCF fold view shape drifted across folds")
        fold_views.append({
            "train": X_fit_view,
            "valid": X_val_view,
            "test": X_test_view,
        })
    meta["support"] = support
    meta["gate_info"] = gate_info
    return {
        "enabled": True,
        "full_train": full_train,
        "full_test": full_test,
        "folds": fold_views,
        "cat_features": mask,
        "selected_cat_idx": meta["selected_cat_idx"],
        "n_extra_features": int(full_train.shape[1] - X_tr.shape[1]),
        "n_pcf_features": int(meta["n_pcf_features"]),
        "feature_view_groups": list(
            meta.get("feature_view_groups", [0] * int(full_train.shape[1]))
        ),
        "meta": meta,
        "gate_info": gate_info,
    }

def apply_tle(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    cat_mask: Sequence[bool],
    task_type: str,
    n_classes: int,
    *,
    enabled: bool = True,
    seed: int = 0,
    probe_params_override: Optional[dict] = None,
    cfg_override: Optional[object] = None,
):
    """Build TLE features. Returns (X_tr_aug, X_te_aug, cat_mask_aug, gate_info)."""
    if not enabled:
        return X_tr, X_te, list(cat_mask), {"used_tle": False, "reason": "disabled"}
    from tle import TLEConfig, auto_gate_tle

    cfg = cfg_override or TLEConfig(seed=seed)
    Xa_tr, Xa_te, mask, _info, gate_info = auto_gate_tle(
        X_tr, y_tr, X_te, cat_mask, task_type, n_classes,
        cfg=cfg, probe_seed=seed,
        probe_params_override=probe_params_override,
    )
    return Xa_tr, Xa_te, mask, gate_info


def apply_tpl(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    cat_mask: Sequence[bool],
    task_type: str,
    n_classes: int,
    *,
    enabled: bool = True,
    seed: int = 0,
    probe_params_override: Optional[dict] = None,
    cfg_override: Optional[object] = None,
):
    """Legacy TPL hook. See apply_tle for the current default."""
    if not enabled:
        return X_tr, X_te, list(cat_mask), {"used_tpl": False, "reason": "disabled"}
    from tpl import TPLConfig, auto_gate_tpl

    cfg = cfg_override or TPLConfig(seed=seed)
    Xa_tr, Xa_te, mask, _info, gate_info = auto_gate_tpl(
        X_tr, y_tr, X_te, cat_mask, task_type, n_classes,
        cfg=cfg, probe_seed=seed,
        probe_params_override=probe_params_override,
    )
    return Xa_tr, Xa_te, mask, gate_info


def apply_hat_oof_tools(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    cat_mask: Sequence[bool],
    task_type: str,
    n_classes: int,
    *,
    enabled: bool = True,
    seed: int = 0,
    config: Optional[dict] = None,
):
    """HAT v0 global OOF tool compiler.

    Returns (X_tr_aug, X_te_aug, cat_mask_aug, gate_info). Target-dependent train
    features are strictly OOF; test features use the full-train recipe.
    """
    if not enabled:
        return X_tr, X_te, list(cat_mask), {"used_hat": False, "reason": "disabled"}
    from hat_tools import apply_hat_oof_tools as _apply

    cfg = dict(config or {})
    allowed_tasks = cfg.get("allowed_tasks")
    if allowed_tasks is not None:
        if isinstance(allowed_tasks, str):
            allowed = {x.strip() for x in allowed_tasks.split(",") if x.strip()}
        else:
            allowed = {str(x) for x in allowed_tasks}
        if task_type not in allowed:
            return X_tr, X_te, list(cat_mask), {
                "used_hat": False,
                "reason": f"task_type {task_type} blocked by HAT eligibility",
                "n_features_added": 0,
                "accepted": [],
                "rejected": [],
            }
    n_cat = int(sum(bool(c) for c in cat_mask))
    min_rows = int(cfg.get("min_rows", 0))
    min_cat_features = int(cfg.get("min_cat_features", 1))
    if X_tr.shape[0] < min_rows or n_cat < min_cat_features:
        return X_tr, X_te, list(cat_mask), {
            "used_hat": False,
            "reason": (
                f"HAT eligibility rejected rows={X_tr.shape[0]} "
                f"n_cat={n_cat} min_rows={min_rows} min_cat={min_cat_features}"
            ),
            "n_features_added": 0,
            "accepted": [],
            "rejected": [],
        }
    Xa_tr, Xa_te, mask, info = _apply(
        X_tr, y_tr, X_te, cat_mask, task_type, n_classes,
        seed=int(cfg.get("seed", seed)),
        n_folds=int(cfg.get("n_folds", 5)),
        prior=cfg.get("prior", 30.0),
        max_cat_features=int(cfg.get("max_cat_features", 10)),
        max_pairs=int(cfg.get("max_pairs", 16)),
        se_mult=float(cfg.get("se_mult", 1.0)),
        shadow_margin=float(cfg.get("shadow_margin", 0.0)),
        max_features=int(cfg.get("max_features", 16)),
        expected_cell_count_min=float(cfg.get("expected_cell_count_min", 0.0)),
        emit_count_feature=bool(cfg.get("emit_count_feature", True)),
        emit_residual_posterior_for_pairs=bool(
            cfg.get("emit_residual_posterior_for_pairs", True)
        ),
        reject_if_auc_degrades=bool(cfg.get("reject_if_auc_degrades", True)),
        shadow_k_factor=float(cfg.get("shadow_k_factor", 1.0)),
        strict_inner_folds=bool(cfg.get("strict_inner_folds", True)),
        hcf_complexity_gamma=float(cfg.get("hcf_complexity_gamma", 0.0)),
        hcf_bound_mult=float(cfg.get("hcf_bound_mult", 0.0)),
        hcf_delta=float(cfg.get("hcf_delta", 0.05)),
        match_oof_test_distribution=bool(cfg.get("match_oof_test_distribution", False)),
        emit_posterior_variance=bool(cfg.get("emit_posterior_variance", False)),
        emit_parent_credibility_for_pairs=bool(
            cfg.get("emit_parent_credibility_for_pairs", False)
        ),
        emit_fallback_level=bool(cfg.get("emit_fallback_level", False)),
        emit_pair_raw_posterior=bool(cfg.get("emit_pair_raw_posterior", True)),
    )
    gate_info = {
        "used_hat": bool(info.used_hat),
        "reason": info.reason,
        "n_features_added": int(info.n_features_added),
        "accepted": info.accepted,
        "rejected": info.rejected,
    }
    return Xa_tr, Xa_te, mask, gate_info
