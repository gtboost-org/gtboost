"""
GTBoost: Sklearn-compatible Python API for the GTBoost gradient boosting library.

Usage:
    from gtboost import GTBClassifier, GTBRegressor

    clf = GTBClassifier(n_estimators=500, learning_rate=0.1, max_depth=6)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50)
    proba = clf.predict_proba(X_test)
    labels = clf.predict(X_test)

    reg = GTBRegressor(n_estimators=500, learning_rate=0.1, max_depth=6)
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
"""

from .gtboost import *

__doc__ = gtboost.__doc__
if hasattr(gtboost, "__all__"):
    __all__ = gtboost.__all__

import json
import os
import pickle
import tempfile
import zipfile

import numpy as np

_RustGTBoostModel = GTBoostModel

_PY_WRAPPER_FORMAT = "gtboost-python-wrapper"
_PY_WRAPPER_VERSION = 1
_BOOSTER_WRAPPER_FORMAT = "gtboost-booster-wrapper"
_BOOSTER_WRAPPER_VERSION = 1
# MC synthetic-prior calibrator (2026-06-12): coefficients learned on 120
# synthetic multiclass worlds (mc_prior_moonshot.py) — maps train-side stats to
# a logit temperature tau and class-prior blend eps. Pure predict-path
# correction; the guard scales it by MEASURED overconfidence so clean data
# self-disables. Validated blind on real mc: yeast -6.7%, cmc -5.9%, ecoli
# -12.4%, vehicle -4.2%, segment 0.0% (auto-off).
_MC_PRIOR_FEATS = ("log_n_per_c", "val_acc", "val_conf", "val_gap", "val_entropy")
_MC_PRIOR_W_TAU = (-0.177463469583047, -0.5360857974960139, -0.19872442332668033,
                   0.3373613741692926, -0.04854190015378278, 1.8832537408373038)
_MC_PRIOR_W_EPS = (0.015472028024773729, -0.06801154209122705, -0.021877340403206477,
                   0.04613420168801867, 0.031310817656262424, 0.05018160968258249)


_PY_ONLY_ESTIMATOR_PARAMS = {
    "auto_class_weights",
    "binary_auc_path_select",
    "binary_auc_path_checks",
    "linear_init",
    "linear_init_ridge",
    "temperature_scale",
    "validation_plateau_prune",
    "trajectory_avg",
    "region_gate",
    "mixup_alpha",
    "mixup_frac",
    "full_refit",
    "growth_policy_race",
    "growth_policy_race_margin",
    "split_risk_auto",
    "split_risk_auto_margin",
    "bins_race",
    "bins_race_margin",
    "auto_mechanisms",
    "binary_shape_auto",
    "binary_shape_auto_margin",
    "residual_focus_auto",
    "residual_focus_auto_alpha",
    "residual_focus_auto_margin",
}


def _compute_auto_stats(X_np, cat_feats):
    import numpy as _np
    X_np = _np.asarray(X_np)
    n_rows = int(X_np.shape[0])
    max_card = 0
    sample = X_np if n_rows <= 5000 else X_np[_np.random.RandomState(0).choice(n_rows, 5000, replace=False)]
    for j, is_cat in enumerate(list(cat_feats)[: X_np.shape[1]]):
        if is_cat:
            col = sample[:, j]
            col = col[_np.isfinite(col)]
            if col.size:
                max_card = max(max_card, int(_np.unique(col).size))
    return (n_rows, max_card)


def _auto_mechanism_params(task, n_rows, max_cat_card, extra_params):
    """Data-adaptive mechanism policy (v1, evidence-cited): enable validated
    accuracy mechanisms in the regimes where they were measured to help.
    User-passed values always win (setdefault semantics); disable wholesale
    with auto_mechanisms=False."""
    raw = (extra_params or {}).get("auto_mechanisms", True)
    if isinstance(raw, str):
        if raw.strip().lower() in {"0", "false", "off", "none", "disabled"}:
            return {}
    elif not raw:
        return {}
    out = {}
    if n_rows is None:
        return out
    # High-cardinality categoricals -> CFE cross-fit evidence engine
    # (Amazon-access beats CatBoost; kdd within 8%; 3-8x faster than old PCF).
    if max_cat_card is not None and max_cat_card >= 16 and n_rows >= 500:
        out.update(cat_fold_evidence=True, cfe_smooth=2.0, cfe_max_pairs=28,
                   cfe_max_triples=20, cfe_max_quads=12)
    # NOT auto-enabled (v1): fold_ordered and supervised_bins help NOISY small
    # binary data (blood +0.6%, diabetes -3%) but hurt CLEAN data (breast
    # -28% rel in the policy validation run) — n_rows alone cannot separate
    # the regimes. They stay documented opt-in flags until a noise-aware gate
    # is built and LODO-validated.
    return out


def _native_extra_params(extra_params):
    return {
        k: v for k, v in dict(extra_params or {}).items()
        if k not in _PY_ONLY_ESTIMATOR_PARAMS
    }


def _same_matrix(X_np, ref):
    if ref is None:
        return False
    X_arr = np.asarray(X_np, dtype=np.float64)
    ref_arr = np.asarray(ref, dtype=np.float64)
    if X_arr.shape != ref_arr.shape:
        return False
    try:
        # equal_nan: matrices with missing values must still match their stored
        # copy, otherwise the deploy-only full-refit model gets served when
        # scoring the eval fold itself (silent validation leakage).
        return bool(np.array_equal(X_arr, ref_arr, equal_nan=True))
    except Exception:
        return False


def _trajectory_avg_active(extra_params):
    try:
        return float(dict(extra_params or {}).get("trajectory_avg", 0.0)) > 0.0
    except (TypeError, ValueError):
        return False


def _mixup_augment(X, y, cat_feats, task, n_classes, alpha, frac, seed):
    """Mixup data augmentation for GBDT: train on convex combinations of points so
    the one model is regularized toward a smooth/linear response between training
    examples, adding zero model capacity (it cannot overfit the way extra leaves /
    gates do). Returns (X_aug, y_aug, w_aug).

    Numeric features are interpolated x' = lam*x_i + (1-lam)*x_j; categoricals are
    coin-flipped (interpolating codes is meaningless). Regression mixes the target
    directly (soft label). Classification uses the EXACT cross-entropy
    decomposition CE(x', lam*y_i+(1-lam)*y_j) = lam*CE(x',y_i) + (1-lam)*CE(x',y_j),
    i.e. two weighted rows with the original integer labels — no soft-label or
    label-encoding hacks. Real rows keep weight 1."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n, d = X.shape
    n_aug = int(round(max(0.0, frac) * n))
    if alpha <= 0.0 or n_aug <= 0 or n < 4:
        return X, y, np.ones(n, dtype=np.float64)
    rng = np.random.RandomState(seed)
    i = rng.randint(0, n, n_aug)
    j = rng.randint(0, n, n_aug)
    lam = rng.beta(alpha, alpha, n_aug)
    cat = np.asarray(cat_feats, dtype=bool) if cat_feats is not None else np.zeros(d, bool)
    if cat.size != d:
        cat = np.zeros(d, dtype=bool)
    num = ~cat
    Xm = np.empty((n_aug, d), dtype=np.float64)
    if num.any():
        Xm[:, num] = lam[:, None] * X[i][:, num] + (1.0 - lam)[:, None] * X[j][:, num]
    if cat.any():
        pick_i = rng.random(n_aug) < lam
        Xm[:, cat] = np.where(pick_i[:, None], X[i][:, cat], X[j][:, cat])
    if task == "regression":
        ym = lam * y[i].astype(np.float64) + (1.0 - lam) * y[j].astype(np.float64)
        X_aug = np.vstack([X, Xm])
        y_aug = np.concatenate([y.astype(np.float64), ym])
        w_aug = np.ones(n + n_aug, dtype=np.float64)
        return X_aug, y_aug, w_aug
    # classification: exact CE decomposition into two weighted rows per mix.
    X_aug = np.vstack([X, Xm, Xm])
    y_aug = np.concatenate([y, y[i], y[j]]).astype(np.float64)
    w_aug = np.concatenate([np.ones(n, dtype=np.float64), lam, 1.0 - lam])
    return X_aug, y_aug, w_aug


def _maybe_full_refit(estimator, build_model, X_np, y_np, cat_feats,
                      eval_X_np, eval_y_np, task, n_classes):
    """Deploy-only full-data refit after early-stopping model selection.

    The original validation-scored model is kept for validation matrices, so CV
    never scores a model trained on its own validation labels. For non-validation
    prediction (the final test path), a side model can use train+validation labels
    when the validation curve says the original path was still saturated.
    """
    ep = getattr(estimator, "_extra_params", {})
    raw = ep.get("full_refit", "auto")
    key = str(raw).strip().lower()
    if key in {"0", "false", "off", "none", "disabled"}:
        estimator.full_refit_info_ = {"enabled": False, "applied": False, "reason": "disabled"}
        return
    try:
        best_trees = int(getattr(estimator, "best_iteration_", 0) or 0)
        total_trees = int(len(estimator._model.tree_info())) if estimator._model is not None else 0
    except Exception:
        best_trees = 0
        total_trees = 0
    estimator._full_refit_model = None
    estimator._full_refit_eval_X = None
    estimator._full_refit_payload = None
    estimator._full_refit_linear_init_state = None
    estimator.full_refit_info_ = {"enabled": True, "applied": False}
    if eval_X_np is None or total_trees <= 0:
        estimator.full_refit_info_ = {"enabled": True, "applied": False, "reason": "no_eval_or_trees"}
        return
    try:
        subtrees = int(ep.get("n_trees_per_round", 1) or 1)
    except Exception:
        subtrees = 1
    planned_trees = int(getattr(estimator, "n_estimators", total_trees) or total_trees) * max(1, subtrees)
    saturation = best_trees / max(planned_trees, 1)
    try:
        best_score = float(getattr(estimator, "best_score_", float("inf")))
    except Exception:
        best_score = float("inf")
    multiclass_high_loss = key == "auto" and task == "multiclass" and best_score > 0.45
    if key == "auto" and task != "regression" and not multiclass_high_loss:
        estimator.full_refit_info_ = {
            "enabled": True,
            "applied": False,
            "reason": "auto_regression_only",
            "best_trees": int(best_trees),
            "planned_trees": int(planned_trees),
            "saturation": float(saturation),
        }
        return
    if key == "auto" and task == "regression" and saturation < 0.85 and not multiclass_high_loss:
        estimator.full_refit_info_ = {
            "enabled": True,
            "applied": False,
            "reason": "not_saturated",
            "best_trees": int(best_trees),
            "planned_trees": int(planned_trees),
            "saturation": float(saturation),
        }
        return
    tpr = int(n_classes) if (task == "multiclass" and n_classes and int(n_classes) > 2) else 1
    rounds = max(1, best_trees // max(1, tpr)) if best_trees > 0 else int(estimator.n_estimators)
    old_grow_policy = getattr(estimator, "grow_policy", None)
    override_grow_policy = "depthwise" if multiclass_high_loss else None
    X_comb = np.vstack([np.asarray(X_np, dtype=np.float64), np.asarray(eval_X_np, dtype=np.float64)])
    y_comb = np.concatenate([np.asarray(y_np), np.asarray(eval_y_np)])
    estimator._full_refit_payload = {
        "X": X_comb,
        "y": y_comb,
        "task": task,
        "cat_feats": list(cat_feats),
        "rounds": int(rounds),
        "override_grow_policy": override_grow_policy,
    }
    estimator._full_refit_eval_X = np.asarray(eval_X_np, dtype=np.float64).copy()
    estimator.full_refit_info_ = {
        "enabled": True,
        "applied": True,
        "deploy_only": True,
        "lazy": True,
        "rounds": int(rounds),
        "best_trees": int(best_trees),
        "total_trees": int(total_trees),
        "planned_trees": int(planned_trees),
        "saturation": float(saturation),
        "best_score": float(best_score),
        "override_grow_policy": override_grow_policy or "",
        "n_train": int(len(y_np)),
        "n_total": int(len(y_comb)),
    }


def _ensure_full_refit_model(estimator):
    if getattr(estimator, "_full_refit_model", None) is not None:
        return True
    payload = getattr(estimator, "_full_refit_payload", None)
    if not payload:
        return False
    task = payload["task"]
    cat_feats = payload["cat_feats"]
    override_grow_policy = payload.get("override_grow_policy")
    old_grow_policy = getattr(estimator, "grow_policy", None)
    try:
        if override_grow_policy is not None:
            estimator.grow_policy = override_grow_policy
        if task == "regression":
            model = estimator._build_model(cat_feats, n_rows=int(payload["X"].shape[0]))
        else:
            model = estimator._build_model(task, cat_feats)
        full_linear_state, full_linear_info = _fit_linear_init_state(
            payload["X"],
            payload["y"],
            task,
            cat_feats,
            mode=_linear_init_mode_for_estimator(estimator, task),
            ridge=float(
                getattr(estimator, "_extra_params", {}).get(
                    "linear_init_ridge",
                    0.3
                    if task == "multiclass"
                    else (
                        20.0
                        if task == "binary"
                        and sum(1 for is_cat in cat_feats if not is_cat) >= 16
                        else 1.0
                    ),
                )
            ),
        )
        if not full_linear_info.get("enabled", False):
            full_linear_state = None
        estimator._full_refit_linear_init_state = full_linear_state
        init_score = _linear_init_score_for_fit(full_linear_state, payload["X"])
        model.fit(payload["X"], payload["y"], int(payload["rounds"]), init_score=init_score)
    except Exception as exc:
        estimator.full_refit_info_ = {
            **dict(getattr(estimator, "full_refit_info_", {})),
            "trained": False,
            "error": type(exc).__name__,
        }
        return False
    finally:
        if override_grow_policy is not None and old_grow_policy is not None:
            estimator.grow_policy = old_grow_policy
    estimator._full_refit_model = model
    estimator.full_refit_info_ = {
        **dict(getattr(estimator, "full_refit_info_", {})),
        "trained": True,
        "linear_prior": bool(estimator._full_refit_linear_init_state),
    }
    return True


def _maybe_mixup(estimator, X_np, y_np, cat_feats, task, n_classes):
    """Apply mixup augmentation if mixup_alpha > 0; else identity (weights=ones)."""
    ep = getattr(estimator, "_extra_params", {})
    try:
        alpha = float(ep.get("mixup_alpha", 0.0) or 0.0)
    except (TypeError, ValueError):
        alpha = 0.0
    try:
        frac = float(ep.get("mixup_frac", 1.0) or 0.0)
    except (TypeError, ValueError):
        frac = 1.0
    if alpha <= 0.0:
        estimator.mixup_info_ = {"enabled": False, "reason": "disabled"}
        return X_np, y_np, np.ones(len(y_np), dtype=np.float64)
    if task == "binary":
        yy = np.asarray(y_np, dtype=np.int64)
        counts = np.bincount(yy, minlength=2).astype(np.float64)
        if np.min(counts) <= 0.0:
            estimator.mixup_info_ = {"enabled": False, "reason": "binary_missing_class"}
            return X_np, y_np, np.ones(len(y_np), dtype=np.float64)
        imbalance = float(np.max(counts) / np.min(counts))
        if imbalance < 2.5:
            estimator.mixup_info_ = {
                "enabled": False,
                "reason": "binary_low_imbalance",
                "requested_alpha": float(alpha),
                "imbalance": imbalance,
            }
            return X_np, y_np, np.ones(len(y_np), dtype=np.float64)
    seed = int(getattr(estimator, "seed", 42) or 42)
    X_aug, y_aug, w_aug = _mixup_augment(
        X_np, y_np, cat_feats, task, n_classes, alpha, frac, seed
    )
    estimator.mixup_info_ = {
        "enabled": True, "alpha": alpha, "frac": frac,
        "n_real": int(len(y_np)), "n_total": int(len(y_aug)),
    }
    return X_aug, y_aug, w_aug


class GTBoostModel:
    """Thin Python wrapper around the native Rust model.

    The wrapper preserves the normal Rust API and adds PCF categorical geometry.
    PCF uses private cross-fit folds inside ``fit`` to build leak-safe posterior
    geometry for categorical columns. Callers do not provide folds, split ids,
    or target-encoding tables; fitted transformers are stored on the model and
    applied automatically at prediction time.
    """

    def __init__(
        self,
        *args,
        categorical_geometry=None,
        pcf_config=None,
        **kwargs,
    ):
        self._raw_args = args
        kwargs = dict(kwargs)
        default_rounds = kwargs.pop("n_estimators", None)
        if default_rounds is None:
            default_rounds = kwargs.pop("n_rounds", 1000)
        else:
            kwargs.pop("n_rounds", None)
        self._default_n_rounds = int(default_rounds)
        self._raw_kwargs = kwargs
        self._pcf_config = dict(pcf_config or {})
        raw_geometry = None if categorical_geometry is None else str(categorical_geometry).lower()
        if raw_geometry in {"none", "off", "raw", "false", "0", ""}:
            raw_geometry = None
        if raw_geometry == "pcf_lite":
            self._pcf_config.setdefault("profile", "lite")
        if raw_geometry in {"pcf_fast_mc", "pcf_mc_fast", "fast_mc"}:
            self._pcf_config.setdefault("profile", "mc_fast")
        if raw_geometry in {"pcf_mc_blocks", "pcf_fast_mc_blocks", "mc_blocks"}:
            self._pcf_config.setdefault("profile", "mc_blocks")
        self._categorical_geometry = {
            "pcf_lite": "pcf",
            "pcf_fast_mc": "pcf",
            "pcf_mc_fast": "pcf",
            "fast_mc": "pcf",
            "pcf_mc_blocks": "pcf",
            "pcf_fast_mc_blocks": "pcf",
            "mc_blocks": "pcf",
        }.get(raw_geometry, raw_geometry)
        self._pcf_runtime = None
        self.categorical_geometry_info_ = {
            "enabled": False,
            "reason": "categorical_geometry disabled",
        }
        self._model = None
        self._full_refit_model = None
        self._full_refit_eval_X = None
        self._full_refit_payload = None
        self.evals_result_ = {}
        self.best_iteration_ = None
        self.best_score_ = None
        if self._categorical_geometry is None:
            self._model = _RustGTBoostModel(*self._raw_args, **self._raw_kwargs)
        elif self._categorical_geometry != "pcf":
            raise ValueError(
                "unknown categorical_geometry="
                f"{categorical_geometry!r}; supported values are 'pcf', "
                "'pcf_lite', 'pcf_fast_mc', and 'pcf_mc_blocks'"
            )

    def _seed(self) -> int:
        seed = self._raw_kwargs.get("seed", 42)
        return 42 if seed is None else int(seed)

    def _task(self) -> str:
        return str(self._raw_kwargs.get("task", "regression"))

    def _cat_features_for(self, x, X_np):
        cat_feats = self._raw_kwargs.get("cat_features", None)
        if cat_feats is None:
            cat_feats = _detect_cat_features(x)
        if not cat_feats:
            cat_feats = [False] * int(X_np.shape[1])
        return [bool(c) for c in cat_feats]

    def _pcf_cat_features_for(self, X_np, raw_cat_feats):
        """Choose columns used to build PCF tuple keys.

        Normally PCF uses the model's categorical columns. Some tabular data has
        repeated low-resolution numeric measurements that are not semantically
        categorical but do form useful posterior evidence tuples. The opt-in
        ``pcf_cat_features='all_discrete'`` mode lets PCF use those columns as
        evidence sources while the served model can still keep the original raw
        categorical mask.
        """
        source = self._pcf_config.get(
            "pcf_cat_features",
            self._pcf_config.get("cat_feature_source", None),
        )
        if source is None:
            return list(raw_cat_feats)
        if isinstance(source, (list, tuple, np.ndarray)):
            if len(source) != int(X_np.shape[1]):
                raise ValueError(
                    "pcf_cat_features length "
                    f"{len(source)} does not match X width {X_np.shape[1]}"
                )
            return [bool(c) for c in source]
        source = str(source).lower()
        if source not in {"all_discrete", "discrete", "all_repeated"}:
            return list(raw_cat_feats)

        max_card = int(self._pcf_config.get("pcf_discrete_max_cardinality", 256))
        min_card = int(self._pcf_config.get("pcf_discrete_min_cardinality", 2))
        out = []
        for j in range(int(X_np.shape[1])):
            col = np.asarray(X_np[:, j])
            col = col[np.isfinite(col)]
            if col.size == 0:
                out.append(False)
                continue
            n_unique = int(np.unique(col.astype(np.int64, copy=False)).size)
            out.append(min_card <= n_unique <= max_card)
        return out

    def _maybe_preserve_raw_cat_mask(self, out_cat, raw_cat_feats, X_out):
        view = str(self._pcf_config.get("view", "")).lower()
        if (
            view == "pcf_append"
            and bool(self._pcf_config.get("preserve_raw_cat_features", False))
            and int(X_out.shape[1]) >= len(raw_cat_feats)
        ):
            return list(raw_cat_feats) + [False] * (int(X_out.shape[1]) - len(raw_cat_feats))
        return out_cat

    def _ensure_model(self, cat_features, feature_view_groups=None):
        kwargs = dict(self._raw_kwargs)
        kwargs["cat_features"] = list(cat_features)
        if feature_view_groups is not None:
            groups = [int(g) for g in feature_view_groups]
            if groups:
                kwargs["feature_view_groups"] = groups
        self._model = _RustGTBoostModel(*self._raw_args, **kwargs)

    def fit(
        self,
        x,
        y,
        n_rounds=None,
        eval_x=None,
        eval_y=None,
        init_score=None,
        eval_init_score=None,
        sample_weight=None,
    ):
        n_rounds = int(self._default_n_rounds if n_rounds is None else n_rounds)
        if self._categorical_geometry is None:
            reference = None
            if isinstance(x, Dataset):
                reference = x
            elif _is_dataframe(x):
                reference = Dataset(
                    x,
                    categorical=self._raw_kwargs.get("cat_features", "auto"),
                )
            X_fit = reference.data if reference is not None else _to_numpy(x)
            eval_fit = (
                _transform_with_reference(eval_x, reference)
                if eval_x is not None and reference is not None
                else (None if eval_x is None else _to_numpy(eval_x))
            )
            if reference is not None and not self._raw_kwargs.get("cat_features"):
                self._ensure_model(reference.cat_features)
            result = self._model.fit(
                X_fit,
                y,
                n_rounds,
                eval_x=eval_fit,
                eval_y=eval_y,
                init_score=init_score,
                eval_init_score=eval_init_score,
                sample_weight=sample_weight,
            )
            _set_eval_attributes(self, self._model, self._task())
            return result

        X_np = _to_numpy(x)
        y_np = np.asarray(y, dtype=np.float64)
        cat_feats = self._cat_features_for(x, X_np)
        eval_list = []
        eval_x_np = None
        if eval_x is not None:
            eval_x_np = _to_numpy(eval_x)
            eval_list.append(eval_x_np)
        eval_y_fit = None if eval_y is None else np.asarray(eval_y, dtype=np.float64)

        task = self._task()
        if task in {"binary", "multiclass", "regression"}:
            from feature_transforms import PCFGeometryRuntime

            feature_view_groups = None
            pcf_cat_feats = self._pcf_cat_features_for(X_np, cat_feats)
            n_classes = 2
            if task == "multiclass":
                y_int = y_np.astype(np.int64)
                n_classes = int(np.max(y_int) + 1) if y_int.size else 0
                n_classes = max(n_classes, int(np.unique(y_int).size))
            runtime = PCFGeometryRuntime(
                task_type=task,
                n_classes=n_classes,
                config=self._pcf_config,
                seed=self._seed(),
                fallback_raw=True,
            )
            X_fit, applied, out_cat, meta = runtime.fit_transform(
                X_np,
                y_np,
                pcf_cat_feats,
                apply_mats=eval_list,
            )
            out_cat = self._maybe_preserve_raw_cat_mask(out_cat, cat_feats, X_fit)
            self._pcf_runtime = runtime if runtime.enabled else None
            self.categorical_geometry_info_ = dict(meta)
            feature_view_groups = meta.get("feature_view_groups", None)
            eval_fit = applied[0] if applied else eval_x_np
            self._ensure_model(out_cat, feature_view_groups=feature_view_groups)
        else:
            X_fit = X_np
            eval_fit = eval_x_np
            self._pcf_runtime = None
            self.categorical_geometry_info_ = {
                "enabled": False,
                "reason": "categorical_geometry requires a supervised classification/regression task",
            }
            self._ensure_model(cat_feats)

        result = self._model.fit(
            X_fit,
            y_np,
            n_rounds,
            eval_x=eval_fit,
            eval_y=eval_y_fit,
            init_score=init_score,
            eval_init_score=eval_init_score,
            sample_weight=sample_weight,
        )
        _set_eval_attributes(self, self._model, self._task())
        return result

    def _transform_x(self, x):
        X_np = _to_numpy(x)
        if self._pcf_runtime is not None and self._pcf_runtime.enabled:
            return self._pcf_runtime.transform(X_np)
        return X_np

    def predict(self, x, *args, **kwargs):
        return self._model.predict(self._transform_x(x), *args, **kwargs)

    def predict_truncated(self, x, *args, **kwargs):
        return self._model.predict_truncated(self._transform_x(x), *args, **kwargs)

    def predict_pruned(self, x, *args, **kwargs):
        return self._model.predict_pruned(self._transform_x(x), *args, **kwargs)

    def predict_with_tree_mask(self, x, *args, **kwargs):
        return self._model.predict_with_tree_mask(self._transform_x(x), *args, **kwargs)

    def leaf_indices(self, x, *args, **kwargs):
        return self._model.leaf_indices(self._transform_x(x), *args, **kwargs)

    def _has_active_python_geometry(self):
        return self._pcf_runtime is not None and self._pcf_runtime.enabled

    def save_model(self, path):
        if self._model is None:
            raise RuntimeError("cannot save an unfitted GTBoostModel")
        path = str(path)
        if not self._has_active_python_geometry():
            self._model.save_model(path)
            return

        metadata = {
            "format": _PY_WRAPPER_FORMAT,
            "version": _PY_WRAPPER_VERSION,
            "categorical_geometry": self._categorical_geometry,
            "has_pcf_runtime": bool(self._pcf_runtime is not None and self._pcf_runtime.enabled),
        }
        state = {
            "_raw_args": self._raw_args,
            "_raw_kwargs": self._raw_kwargs,
            "_default_n_rounds": self._default_n_rounds,
            "_pcf_config": self._pcf_config,
            "_categorical_geometry": self._categorical_geometry,
            "_pcf_runtime": self._pcf_runtime,
            "categorical_geometry_info_": self.categorical_geometry_info_,
        }
        with tempfile.TemporaryDirectory() as tmp:
            rust_path = os.path.join(tmp, "rust_model.json")
            self._model.save_model(rust_path)
            with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(
                    "metadata.json",
                    json.dumps(metadata, sort_keys=True, indent=2).encode("utf-8"),
                )
                zf.write(rust_path, arcname="rust_model.json")
                # PCF is a Python runtime; pickle keeps its NumPy tables together
                # with the Rust model.
                zf.writestr(
                    "wrapper.pkl",
                    pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL),
                )

    @classmethod
    def load_model(cls, path):
        path = str(path)
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path, mode="r") as zf:
                metadata = json.loads(zf.read("metadata.json").decode("utf-8"))
                if metadata.get("format") != _PY_WRAPPER_FORMAT:
                    raise ValueError(f"unsupported GTBoost wrapper format: {metadata!r}")
                if int(metadata.get("version", 0)) != _PY_WRAPPER_VERSION:
                    raise ValueError(
                        "unsupported GTBoost wrapper version "
                        f"{metadata.get('version')}; expected {_PY_WRAPPER_VERSION}"
                    )
                state = pickle.loads(zf.read("wrapper.pkl"))
                with tempfile.TemporaryDirectory() as tmp:
                    rust_path = os.path.join(tmp, "rust_model.json")
                    with open(rust_path, "wb") as f:
                        f.write(zf.read("rust_model.json"))
                    rust_model = _RustGTBoostModel.load_model(rust_path)

            obj = cls.__new__(cls)
            obj._raw_args = tuple(state.get("_raw_args", ()))
            obj._raw_kwargs = dict(state.get("_raw_kwargs", {}))
            obj._default_n_rounds = int(state.get("_default_n_rounds", 0))
            obj._pcf_config = dict(state.get("_pcf_config", {}))
            obj._categorical_geometry = state.get("_categorical_geometry", None)
            obj._pcf_runtime = state.get("_pcf_runtime", None)
            obj.categorical_geometry_info_ = dict(
                state.get(
                    "categorical_geometry_info_",
                    {"enabled": False, "reason": "loaded Python wrapper"},
                )
            )
            obj._model = rust_model
            return obj

        obj = cls.__new__(cls)
        obj._raw_args = ()
        obj._raw_kwargs = {}
        obj._default_n_rounds = 0
        obj._pcf_config = {}
        obj._categorical_geometry = None
        obj._pcf_runtime = None
        obj.categorical_geometry_info_ = {
            "enabled": False,
            "reason": "loaded native model",
        }
        obj._model = _RustGTBoostModel.load_model(str(path))
        return obj

    def __getattr__(self, name):
        model = self.__dict__.get("_model")
        if model is None:
            raise AttributeError(name)
        return getattr(model, name)


def predict_mc_dropout(model, X, keep_rate=0.5, n_samples=32, seed=0, task_type="multiclass", n_classes=0, aggregation="auto"):
    """Test-time MC-dropout over trees: random tree subsets with bias-corrected
    scaling, aggregated. Game-changer for multiclass (Jensen's inequality pulls
    overconfident softmax back toward calibrated posterior).

    aggregation: "auto" picks best per task_type:
       multiclass/binary → mean of softmax/sigmoid (Jensen's smoothing)
       regression → median of scaled subset predictions (robust to structural
                    imbalance from subsampling GBM's sequentially-fit trees)
    Other values: "mean", "median", "trimmed_mean" (20% trim).
    """
    if keep_rate >= 1.0 - 1e-9:
        raw = np.array(model.predict(X))
        if task_type == "multiclass":
            n_rows = X.shape[0] if hasattr(X, "shape") else len(X)
            logits = raw.reshape(n_rows, n_classes)
            m_row = logits.max(axis=1, keepdims=True)
            e = np.exp(logits - m_row)
            return e / e.sum(axis=1, keepdims=True)
        if task_type == "binary":
            return 1.0 / (1.0 + np.exp(-raw))
        return raw
    n_trees = len(model.tree_info())
    rng = np.random.RandomState(seed)
    n_rows = X.shape[0] if hasattr(X, "shape") else len(X)
    scale = 1.0 / max(keep_rate, 1e-6)
    agg = aggregation
    if agg == "auto":
        agg = "median" if task_type == "regression" else "mean"
    # Stack samples along axis 0 for non-mean aggregations
    need_stack = agg != "mean"
    samples = [] if need_stack else None
    accumulator = None
    valid_samples = 0
    for _ in range(n_samples):
        mask = (rng.rand(n_trees) < keep_rate).astype(np.uint8).tolist()
        if sum(mask) == 0:
            mask[0] = 1
        raw = np.array(model.predict_with_tree_mask(X, mask)) * scale
        if task_type == "multiclass":
            logits = raw.reshape(n_rows, n_classes)
            # Proper softmax — previously returned raw logits, which sklearn
            # log_loss clipped to [eps, 1] giving spuriously low multiclass
            # logloss when correct-class logit > 1.
            m_row = logits.max(axis=1, keepdims=True)
            e = np.exp(logits - m_row)
            p = e / e.sum(axis=1, keepdims=True)
        elif task_type == "binary":
            p = 1.0 / (1.0 + np.exp(-raw))
        else:
            p = raw
        if need_stack:
            samples.append(p)
        else:
            if accumulator is None: accumulator = p.astype(np.float64)
            else: accumulator += p
        valid_samples += 1
    if need_stack:
        stacked = np.stack(samples, axis=0)
        if agg == "median":
            return np.median(stacked, axis=0)
        elif agg == "trimmed_mean":
            # 20% trim each side
            k = max(1, int(0.2 * stacked.shape[0]))
            sorted_s = np.sort(stacked, axis=0)
            return sorted_s[k:-k].mean(axis=0) if stacked.shape[0] > 2*k else stacked.mean(axis=0)
        else:
            return stacked.mean(axis=0)
    return accumulator / max(valid_samples, 1)


def auto_mc_dropout(model, X, X_val, y_val, task_type, n_classes,
                    keep_rates=None, n_samples=32, seed=0, margin=0.01, metric_fn=None,
                    aggregations=None):
    """Sweep (keep_rate, aggregation) on validation set, pick lowest val loss.
    Conservative: only switches away from kr=1.0 if val loss improves by `margin` relative.
    Returns (chosen_kr, chosen_agg, test_preds_at_chosen).

    Defaults per task_type:
      multiclass: kr ∈ {0.3, 0.5, 0.7, 1.0}, agg ∈ {mean}
      binary:     kr ∈ {0.1, 0.2, 0.3, 0.5, 0.7, 1.0}, agg ∈ {mean}
      regression: kr ∈ {0.5, 0.7, 0.9, 1.0}, agg ∈ {median, trimmed_mean}
    """
    if metric_fn is None:
        raise ValueError("metric_fn required")
    if keep_rates is None:
        if task_type == "regression":
            keep_rates = (0.5, 0.7, 0.9, 1.0)
        elif task_type == "binary":
            keep_rates = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
        else:
            keep_rates = (0.3, 0.5, 0.7, 1.0)
    if aggregations is None:
        if task_type == "regression":
            aggregations = ("median", "trimmed_mean")
        else:
            aggregations = ("mean",)
    base = predict_mc_dropout(model, X_val, keep_rate=1.0, n_samples=1, seed=seed, task_type=task_type, n_classes=n_classes)
    base_loss = metric_fn(y_val, base)
    best_kr = 1.0
    best_agg = aggregations[0]
    best_loss = base_loss
    for agg in aggregations:
        for kr in keep_rates:
            if abs(kr - 1.0) < 1e-6: continue
            p = predict_mc_dropout(model, X_val, keep_rate=kr, n_samples=n_samples, seed=seed,
                                    task_type=task_type, n_classes=n_classes, aggregation=agg)
            loss = metric_fn(y_val, p)
            if loss < best_loss and loss < base_loss * (1.0 - margin):
                best_loss = loss
                best_kr = kr
                best_agg = agg
    test_p = predict_mc_dropout(model, X, keep_rate=best_kr, n_samples=n_samples, seed=seed,
                                 task_type=task_type, n_classes=n_classes, aggregation=best_agg)
    return best_kr, best_agg, test_p


def _is_dataframe(X):
    try:
        import pandas as pd
        return isinstance(X, pd.DataFrame)
    except ImportError:
        return False


def _detect_cat_features(X):
    """Auto-detect categorical features from Dataset or pandas DataFrame dtypes."""
    if isinstance(X, Dataset):
        return list(X.cat_features)
    try:
        import pandas as pd
        from pandas.api import types as pdt
        if isinstance(X, pd.DataFrame):
            return [
                str(dtype) == "category"
                or pdt.is_object_dtype(dtype)
                or pdt.is_string_dtype(dtype)
                or pdt.is_bool_dtype(dtype)
                for dtype in X.dtypes
            ]
    except ImportError:
        pass
    return []


def _to_numpy(X):
    """Convert input to float64 numpy array."""
    if isinstance(X, Dataset):
        return X.data
    try:
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            return Dataset(X, categorical="auto").data
    except ImportError:
        pass
    if isinstance(X, np.ndarray):
        return X.astype(np.float64) if X.dtype != np.float64 else X
    return np.asarray(X, dtype=np.float64)


def _normalize_categorical_spec(categorical, columns, dtypes=None):
    """Return a bool mask from "auto", names, indices, or an existing mask."""
    n = len(columns)
    if categorical is None or categorical is False or str(categorical).lower() in {"none", "off", "raw", "false"}:
        return [False] * n
    if categorical is True or str(categorical).lower() == "auto":
        if dtypes is None:
            return [False] * n
        try:
            from pandas.api import types as pdt
            return [
                str(dtype) == "category"
                or pdt.is_object_dtype(dtype)
                or pdt.is_string_dtype(dtype)
                or pdt.is_bool_dtype(dtype)
                for dtype in dtypes
            ]
        except Exception:
            return [False] * n
    if isinstance(categorical, np.ndarray):
        categorical = categorical.tolist()
    if isinstance(categorical, (list, tuple, set)):
        values = list(categorical)
        if len(values) == n and all(isinstance(v, (bool, np.bool_)) for v in values):
            return [bool(v) for v in values]
        out = [False] * n
        col_to_idx = {c: i for i, c in enumerate(columns)}
        for item in values:
            if isinstance(item, str):
                if item not in col_to_idx:
                    raise ValueError(f"unknown categorical column {item!r}")
                out[col_to_idx[item]] = True
            else:
                idx = int(item)
                if idx < 0 or idx >= n:
                    raise ValueError(f"categorical column index {idx} out of range")
                out[idx] = True
        return out
    raise ValueError(
        "categorical must be 'auto', None, a bool mask, column names, or column indices"
    )


def _encode_dataframe(df, categorical="auto", reference=None, feature_names=None):
    """Encode a pandas DataFrame into float64 matrix plus categorical metadata."""
    if reference is not None:
        feature_names = list(reference.feature_names)
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing[:5]}")
        df = df.loc[:, feature_names]
        cat_mask = list(reference.cat_features)
        mappings = reference._category_maps
    else:
        feature_names = list(df.columns) if feature_names is None else list(feature_names)
        if len(feature_names) != df.shape[1]:
            raise ValueError("feature_names length must match DataFrame width")
        df = df.loc[:, feature_names]
        cat_mask = _normalize_categorical_spec(categorical, feature_names, df.dtypes)
        mappings = {}

    out = np.empty((len(df), len(feature_names)), dtype=np.float64)
    category_maps = {}
    for j, name in enumerate(feature_names):
        series = df[name]
        if cat_mask[j]:
            if reference is None:
                uniques = series.dropna().astype(object).unique().tolist()
                mapping = {value: i for i, value in enumerate(uniques)}
            else:
                mapping = mappings.get(name, {})
            encoded = series.astype(object).map(mapping)
            out[:, j] = encoded.to_numpy(dtype=np.float64, na_value=np.nan)
            category_maps[name] = mapping
        else:
            out[:, j] = np.asarray(
                df[name].pipe(lambda s: __import__("pandas").to_numeric(s, errors="coerce")),
                dtype=np.float64,
            )
    return out, cat_mask, feature_names, category_maps


class Dataset:
    """GTBoost native dataset with stable DataFrame encoding.

    DataFrames keep column names, categorical masks, and category mappings.
    Pass ``reference=train_set`` for validation/test data so category codes are
    identical to training; unseen categories become NaN/unknown.
    """

    def __init__(
        self,
        data,
        label=None,
        categorical="auto",
        feature_names=None,
        reference=None,
        weight=None,
    ):
        if reference is not None and not isinstance(reference, Dataset):
            raise TypeError("reference must be a gtboost.Dataset")

        y = None
        X = data
        if _is_dataframe(data) and isinstance(label, str):
            if label not in data.columns:
                raise ValueError(f"label column {label!r} not found")
            y = np.asarray(data[label], dtype=np.float64)
            X = data.drop(columns=[label])
        elif label is not None:
            y = np.asarray(label, dtype=np.float64)

        if _is_dataframe(X):
            X_np, cat_mask, names, maps = _encode_dataframe(
                X,
                categorical=categorical,
                reference=reference,
                feature_names=feature_names,
            )
        else:
            X_np = np.asarray(X, dtype=np.float64)
            if X_np.ndim != 2:
                raise ValueError("data must be a 2D matrix or pandas DataFrame")
            if reference is not None:
                names = list(reference.feature_names)
                cat_mask = list(reference.cat_features)
            else:
                names = (
                    [f"f{i}" for i in range(X_np.shape[1])]
                    if feature_names is None
                    else list(feature_names)
                )
                if len(names) != X_np.shape[1]:
                    raise ValueError("feature_names length must match matrix width")
                cat_mask = _normalize_categorical_spec(categorical, names, None)
            maps = {}

        if y is not None and len(y) != X_np.shape[0]:
            raise ValueError(f"label length ({len(y)}) must equal data rows ({X_np.shape[0]})")

        self.data = np.ascontiguousarray(X_np, dtype=np.float64)
        self.label = y
        self.weight = None if weight is None else np.asarray(weight, dtype=np.float64)
        self.feature_names = list(names)
        self.cat_features = [bool(c) for c in cat_mask]
        self.categorical_features = [
            name for name, is_cat in zip(self.feature_names, self.cat_features) if is_cat
        ]
        self._category_maps = dict(maps)
        self.reference = reference

    @property
    def shape(self):
        return self.data.shape

    def max_categorical_cardinality(self):
        if not any(self.cat_features):
            return 0
        values = []
        for j, is_cat in enumerate(self.cat_features):
            if is_cat:
                col = self.data[:, j]
                values.append(int(np.unique(col[np.isfinite(col)]).size))
        return max(values) if values else 0

    @classmethod
    def _from_reference_state(cls, state):
        obj = cls.__new__(cls)
        names = list(state.get("feature_names", []))
        cat_features = [bool(c) for c in state.get("cat_features", [])]
        obj.data = np.empty((0, len(names)), dtype=np.float64)
        obj.label = None
        obj.weight = None
        obj.feature_names = names
        obj.cat_features = cat_features
        obj.categorical_features = [
            name for name, is_cat in zip(obj.feature_names, obj.cat_features) if is_cat
        ]
        obj._category_maps = dict(state.get("category_maps", {}))
        obj.reference = None
        return obj


def _fit_dataset(X, y, categorical):
    """Return (dataset_or_none, X_np, y_np, cat_features) for estimator fit."""
    if isinstance(X, Dataset):
        y_np = X.label if y is None else np.asarray(y, dtype=np.float64)
        if y_np is None:
            raise ValueError("y must be provided when Dataset has no label")
        return X, X.data, np.asarray(y_np, dtype=np.float64), list(X.cat_features)
    if _is_dataframe(X):
        ds = Dataset(X, label=y, categorical="auto" if categorical is None else categorical)
        return ds, ds.data, np.asarray(ds.label, dtype=np.float64), list(ds.cat_features)
    X_np = _to_numpy(X)
    y_np = np.asarray(y, dtype=np.float64)
    cat_feats = categorical
    if cat_feats is None:
        cat_feats = [False] * X_np.shape[1]
    elif isinstance(cat_feats, (list, tuple, np.ndarray)):
        cat_feats = _normalize_categorical_spec(cat_feats, [f"f{i}" for i in range(X_np.shape[1])])
    else:
        cat_feats = [False] * X_np.shape[1]
    return None, X_np, y_np, [bool(c) for c in cat_feats]


def _fit_classifier_dataset(X, y, categorical):
    """Return fit data for classifiers without coercing class labels to float."""
    if isinstance(X, Dataset):
        y_raw = X.label if y is None else np.asarray(y)
        if y_raw is None:
            raise ValueError("y must be provided when Dataset has no label")
        return X, X.data, np.asarray(y_raw), list(X.cat_features)
    if y is None:
        raise ValueError("y must be provided")
    if _is_dataframe(X):
        ds = Dataset(X, categorical="auto" if categorical is None else categorical)
        return ds, ds.data, np.asarray(y), list(ds.cat_features)
    X_np = _to_numpy(X)
    y_raw = np.asarray(y)
    cat_feats = categorical
    if cat_feats is None:
        cat_feats = [False] * X_np.shape[1]
    elif isinstance(cat_feats, (list, tuple, np.ndarray)):
        cat_feats = _normalize_categorical_spec(cat_feats, [f"f{i}" for i in range(X_np.shape[1])])
    else:
        cat_feats = [False] * X_np.shape[1]
    return None, X_np, y_raw, [bool(c) for c in cat_feats]


def _encode_labels(y, classes):
    mapping = {cls: i for i, cls in enumerate(classes)}
    try:
        return np.asarray([mapping[v] for v in np.asarray(y)], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"eval_set contains unseen class label {exc.args[0]!r}") from None


def _transform_with_reference(X, reference):
    if isinstance(X, Dataset):
        return X.data
    if reference is not None and _is_dataframe(X):
        return Dataset(X, reference=reference).data
    return _to_numpy(X)


# ── Oblique (rotation-based) feature augmentation ────────────────────────────
# Breiman's CART §5.2 (1984). Standard axis-aligned trees need many splits to
# approximate diagonal boundaries; a single linear-combination split captures
# them in one node. Modern GBM dropped oblique splits for histogram speed, but
# pre-binning feature augmentation recovers the capability for free.

def _make_oblique_rotations(X, cat_feats, n_rotations, k, seed):
    """Build random oblique features over the numeric columns. Stats fit on train."""
    if n_rotations <= 0:
        return []
    numeric_idx = np.array([i for i, c in enumerate(cat_feats) if not c], dtype=int)
    if len(numeric_idx) < k:
        return []
    rng = np.random.RandomState(seed)
    rotations = []
    for _ in range(n_rotations):
        feats = rng.choice(numeric_idx, size=k, replace=False)
        coef = rng.randn(k)
        coef = coef / (np.linalg.norm(coef) + 1e-12)
        means = np.nanmean(X[:, feats], axis=0)
        stds = np.nanstd(X[:, feats], axis=0) + 1e-8
        rotations.append((feats, coef, means, stds))
    return rotations


def _apply_oblique_rotations(X, rotations):
    if not rotations:
        return X
    cols = []
    for feats, coef, means, stds in rotations:
        z = (X[:, feats] - means) / stds
        z = np.where(np.isfinite(z), z, 0.0)
        cols.append(z @ coef)
    new_cols = np.column_stack(cols).astype(np.float64)
    return np.hstack([X, new_cols])


def _should_use_oblique(oblique_flag, extra_params):
    """auto: enable only when leaf_linear is not set (since leaf_linear provides
    within-leaf linearity that makes oblique redundant — see EXPERIMENTS §93)."""
    if oblique_flag is True:
        return True
    if oblique_flag is False:
        return False
    if oblique_flag == "auto":
        return not bool(extra_params.get("leaf_linear", False))
    return False


def _normalize_verbose(verbose):
    if verbose is True:
        return 1
    if verbose is False or verbose is None:
        return 0
    return max(0, int(verbose))


def _hrc_sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _hrc_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    return np.log(p / (1.0 - p))


def _hrc_softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=1, keepdims=True)
    z = np.clip(z, -50.0, 50.0)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def _hrc_auc_error(y, p):
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if y.size == 0 or np.unique(y).size != 2 or p.size != y.size:
        return float("inf")
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    ranks = np.empty(y.size, dtype=np.float64)
    i = 0
    while i < y.size:
        j = i + 1
        while j < y.size and sorted_p[j] == sorted_p[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j
    pos = y == 1
    n_pos = float(np.sum(pos))
    n_neg = float(y.size - np.sum(pos))
    if n_pos <= 0.0 or n_neg <= 0.0:
        return float("inf")
    auc = (float(np.sum(ranks[pos])) - n_pos * (n_pos + 1.0) * 0.5) / (n_pos * n_neg)
    return float(1.0 - auc)


def _hrc_logloss(y, proba, n_classes):
    yy = np.asarray(y, dtype=np.int64).reshape(-1)
    p = np.asarray(proba, dtype=np.float64).reshape(-1, int(n_classes))
    p = np.clip(p, 1e-15, 1.0)
    p /= np.sum(p, axis=1, keepdims=True)
    return float(np.mean(-np.log(p[np.arange(yy.size), yy])))


def _hrc_score(y, pred, task, n_classes):
    if task == "regression":
        r = np.asarray(y, dtype=np.float64).reshape(-1) - np.asarray(pred, dtype=np.float64).reshape(-1)
        return float(np.mean(r * r))
    if task == "binary":
        p = np.asarray(pred, dtype=np.float64)
        if p.ndim == 2:
            p = p[:, 1]
        return _hrc_auc_error(y, p)
    return _hrc_logloss(y, pred, n_classes)


def _hrc_bin_spec(col, is_cat, max_bins=12):
    x = np.asarray(col, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return {"kind": "edges", "edges": np.array([], dtype=np.float64), "n_bins": 2}
    uniq = np.unique(finite)
    if bool(is_cat) or uniq.size <= max_bins:
        if uniq.size > 64:
            uniq = uniq[:64]
        return {"kind": "values", "values": uniq.astype(np.float64), "n_bins": int(uniq.size + 1)}
    qs = np.linspace(0.0, 1.0, int(max_bins) + 1)[1:-1]
    edges = np.unique(np.quantile(finite, qs)).astype(np.float64)
    return {"kind": "edges", "edges": edges, "n_bins": int(edges.size + 2)}


def _hrc_bin_ids(col, spec):
    x = np.asarray(col, dtype=np.float64)
    if spec.get("kind") == "values":
        vals = np.asarray(spec.get("values", []), dtype=np.float64)
        idx = np.searchsorted(vals, x)
        safe = np.minimum(idx, max(vals.size - 1, 0))
        ok = (idx < vals.size) & np.isfinite(x)
        if vals.size:
            ok &= vals[safe] == x
        return np.where(ok, idx, vals.size).astype(np.int64)
    edges = np.asarray(spec.get("edges", []), dtype=np.float64)
    idx = np.searchsorted(edges, x, side="right")
    idx = np.where(np.isfinite(x), idx, edges.size + 1)
    return idx.astype(np.int64)


def _hrc_fit_values(bin_ids, resid, hess, n_bins, smooth):
    b = np.asarray(bin_ids, dtype=np.int64)
    r = np.asarray(resid, dtype=np.float64)
    h = np.asarray(hess, dtype=np.float64)
    if r.ndim == 1:
        sums = np.bincount(b, weights=r, minlength=n_bins).astype(np.float64)
        den = np.bincount(b, weights=h, minlength=n_bins).astype(np.float64)
        vals = sums / (den + float(smooth))
        vals -= float(np.mean(vals[b])) if b.size else 0.0
        return np.clip(vals, -2.0, 2.0)
    vals = []
    for k in range(r.shape[1]):
        sums = np.bincount(b, weights=r[:, k], minlength=n_bins).astype(np.float64)
        den = np.bincount(b, weights=h[:, k], minlength=n_bins).astype(np.float64)
        vk = sums / (den + float(smooth))
        vk -= float(np.mean(vk[b])) if b.size else 0.0
        vals.append(np.clip(vk, -2.0, 2.0))
    out = np.stack(vals, axis=1)
    out -= np.mean(out, axis=1, keepdims=True)
    return out


def _hrc_apply_terms(state, X):
    if not state:
        return None
    X = np.asarray(X, dtype=np.float64)
    corr = None
    for term in state.get("terms", []):
        feat = int(term["feature"])
        ids = _hrc_bin_ids(X[:, feat], term["spec"])
        vals = np.asarray(term["values"], dtype=np.float64)
        c = vals[ids]
        corr = c if corr is None else corr + c
    return corr


def _apply_honest_residual_correction(state, X, pred):
    corr = _hrc_apply_terms(state, X)
    if corr is None:
        return pred
    task = state.get("task")
    if task == "regression":
        return np.asarray(pred, dtype=np.float64) + corr
    proba = np.asarray(pred, dtype=np.float64)
    if task == "binary":
        p = np.clip(proba[:, 1], 1e-9, 1.0 - 1e-9)
        p2 = _hrc_sigmoid(_hrc_logit(p) + np.asarray(corr, dtype=np.float64).reshape(-1))
        return np.column_stack([1.0 - p2, p2])
    return _hrc_softmax(np.log(np.clip(proba, 1e-15, 1.0)) + corr)


def _fit_honest_residual_correction(
    X_train,
    y_train,
    X_val,
    y_val,
    pred_train,
    pred_val,
    task,
    n_classes,
    cat_features,
):
    Xtr = np.asarray(X_train, dtype=np.float64)
    Xva = np.asarray(X_val, dtype=np.float64)
    if Xtr.ndim != 2 or Xva.ndim != 2 or Xtr.shape[1] != Xva.shape[1] or Xtr.shape[0] < 40:
        return None, {"enabled": False, "reason": "bad_shape"}
    cat = list(cat_features or [False] * Xtr.shape[1])
    if len(cat) != Xtr.shape[1]:
        cat = [False] * Xtr.shape[1]
    specs = [_hrc_bin_spec(Xtr[:, j], cat[j]) for j in range(Xtr.shape[1])]
    bins_tr = [_hrc_bin_ids(Xtr[:, j], specs[j]) for j in range(Xtr.shape[1])]
    bins_va = [_hrc_bin_ids(Xva[:, j], specs[j]) for j in range(Xtr.shape[1])]
    terms = []

    if task == "regression":
        ytr = np.asarray(y_train, dtype=np.float64).reshape(-1)
        yva = np.asarray(y_val, dtype=np.float64).reshape(-1)
        ptr = np.asarray(pred_train, dtype=np.float64).reshape(-1).copy()
        pva = np.asarray(pred_val, dtype=np.float64).reshape(-1).copy()
        cur = _hrc_score(yva, pva, task, n_classes)
        min_gain = max(1e-10, 0.002 * abs(cur))
        smooth = max(5.0, 0.01 * Xtr.shape[0])
        for _ in range(2):
            resid = ytr - ptr
            hess = np.ones_like(resid)
            best = (cur, None, None, 0.0, None, None)
            for j in range(Xtr.shape[1]):
                vals = _hrc_fit_values(bins_tr[j], resid, hess, specs[j]["n_bins"], smooth)
                ctr = vals[bins_tr[j]]
                cva = vals[bins_va[j]]
                denom = float(np.dot(cva, cva)) + 1e-12
                alpha = float(np.clip(np.dot(yva - pva, cva) / denom, 0.0, 1.0))
                score = _hrc_score(yva, pva + alpha * cva, task, n_classes)
                if score < best[0]:
                    best = (score, j, vals * alpha, alpha, ctr * alpha, cva * alpha)
            if best[1] is None or cur - best[0] < min_gain:
                break
            cur, j, vals, alpha, ctr, cva = best
            ptr += ctr
            pva += cva
            terms.append({"feature": int(j), "spec": specs[j], "values": vals})
    elif task == "binary":
        ytr = np.asarray(y_train, dtype=np.float64).reshape(-1)
        yva = np.asarray(y_val, dtype=np.float64).reshape(-1)
        p_tr = np.clip(np.asarray(pred_train, dtype=np.float64)[:, 1], 1e-9, 1.0 - 1e-9)
        p_va = np.clip(np.asarray(pred_val, dtype=np.float64)[:, 1], 1e-9, 1.0 - 1e-9)
        mtr = _hrc_logit(p_tr)
        mva = _hrc_logit(p_va)
        cur = _hrc_score(yva, np.column_stack([1.0 - p_va, p_va]), task, n_classes)
        smooth = max(2.0, 0.005 * Xtr.shape[0])
        min_gain = 5e-4
        for _ in range(2):
            pcur = _hrc_sigmoid(mtr)
            resid = ytr - pcur
            hess = np.maximum(pcur * (1.0 - pcur), 1e-5)
            best = (cur, None, None, 0.0, None, None)
            for j in range(Xtr.shape[1]):
                vals = _hrc_fit_values(bins_tr[j], resid, hess, specs[j]["n_bins"], smooth)
                ctr = vals[bins_tr[j]]
                cva = vals[bins_va[j]]
                for alpha in (0.25, 0.5, 0.75, 1.0):
                    p2 = _hrc_sigmoid(mva + alpha * cva)
                    score = _hrc_score(yva, np.column_stack([1.0 - p2, p2]), task, n_classes)
                    if score < best[0]:
                        best = (score, j, vals * alpha, alpha, ctr * alpha, cva * alpha)
            if best[1] is None or cur - best[0] < min_gain:
                break
            cur, j, vals, alpha, ctr, cva = best
            mtr += ctr
            mva += cva
            terms.append({"feature": int(j), "spec": specs[j], "values": vals})
    else:
        n_classes = int(n_classes)
        if n_classes < 3:
            return None, {"enabled": False, "reason": "bad_classes"}
        ytr = np.asarray(y_train, dtype=np.int64).reshape(-1)
        yva = np.asarray(y_val, dtype=np.int64).reshape(-1)
        ptr = np.clip(np.asarray(pred_train, dtype=np.float64), 1e-9, 1.0)
        pva = np.clip(np.asarray(pred_val, dtype=np.float64), 1e-9, 1.0)
        ptr /= np.sum(ptr, axis=1, keepdims=True)
        pva /= np.sum(pva, axis=1, keepdims=True)
        mtr = np.log(ptr)
        mva = np.log(pva)
        cur = _hrc_score(yva, pva, task, n_classes)
        smooth = max(2.0, 0.005 * Xtr.shape[0])
        min_gain = max(1e-6, 0.004 * abs(cur))
        eye = np.eye(n_classes, dtype=np.float64)
        for _ in range(2):
            pcur = _hrc_softmax(mtr)
            resid = eye[ytr] - pcur
            hess = np.maximum(pcur * (1.0 - pcur), 1e-5)
            best = (cur, None, None, 0.0, None, None)
            for j in range(Xtr.shape[1]):
                vals = _hrc_fit_values(bins_tr[j], resid, hess, specs[j]["n_bins"], smooth)
                ctr = vals[bins_tr[j]]
                cva = vals[bins_va[j]]
                for alpha in (0.25, 0.5, 0.75, 1.0):
                    score = _hrc_score(yva, _hrc_softmax(mva + alpha * cva), task, n_classes)
                    if score < best[0]:
                        best = (score, j, vals * alpha, alpha, ctr * alpha, cva * alpha)
            if best[1] is None or cur - best[0] < min_gain:
                break
            cur, j, vals, alpha, ctr, cva = best
            mtr += ctr
            mva += cva
            terms.append({"feature": int(j), "spec": specs[j], "values": vals})

    if not terms:
        return None, {"enabled": False, "reason": "no_valid_terms"}
    return {"task": task, "n_classes": int(n_classes), "terms": terms}, {
        "enabled": True,
        "task": task,
        "n_terms": len(terms),
        "features": [int(t["feature"]) for t in terms],
    }


def _normalize_huber_delta(value):
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"auto", "adaptive", "mad"}:
            return -1.0
        if key in {"off", "none", "mse", "false", "0"}:
            return 0.0
    return float(value)


def _resolve_regression_huber_delta(y, value):
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"auto", "adaptive", "mad"}:
            yy = np.asarray(y, dtype=np.float64).reshape(-1)
            yy = yy[np.isfinite(yy)]
            if yy.size < 40:
                return -1.0, {"mode": "adaptive", "reason": "small_sample"}
            mean = float(np.mean(yy))
            std = float(np.std(yy))
            if std <= 1e-12:
                return 0.0, {"mode": "mse", "reason": "constant_target"}
            centered = yy - mean
            skew = float(np.mean(centered ** 3) / max(std ** 3, 1e-12))
            q50 = float(np.quantile(yy, 0.50))
            q90 = float(np.quantile(yy, 0.90))
            q99 = float(np.quantile(yy, 0.99))
            scale = max(abs(q50), 1e-12)
            positive_tail = float(np.min(yy)) >= 0.0 and q90 / scale >= 2.4
            extreme_tail = q99 / scale >= 3.5 or std / scale >= 0.9
            if skew >= 1.0 and positive_tail and extreme_tail:
                return 0.0, {
                    "mode": "mse",
                    "reason": "positive_heavy_tail",
                    "skew": skew,
                    "q90_over_q50": q90 / scale,
                    "q99_over_q50": q99 / scale,
                }
            return -1.0, {
                "mode": "adaptive",
                "reason": "auto",
                "skew": skew,
                "q90_over_q50": q90 / scale,
                "q99_over_q50": q99 / scale,
            }
    delta = _normalize_huber_delta(value)
    return delta, {"mode": "fixed" if delta > 0.0 else "mse", "delta": float(delta)}


def _normalize_robust_leaves(value):
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"auto", "adaptive", "true", "1"}:
            return -1.0
        if key in {"off", "none", "false", "0"}:
            return 0.0
    if value is True:
        return -1.0
    if value is False or value is None:
        return 0.0
    return float(value)


def _normalize_regression_sparse_oblique(value, cat_features, n_rows=None):
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"auto", "adaptive"}:
            # FAST TREES are the default: oblique search is a second per-node
            # split search (~4x fit time for ~1-2% rmse). It never turns on
            # automatically; opt in with sparse_oblique_splits=True for the
            # accuracy mode.
            return False
        if key in {"off", "none", "false", "0"}:
            return False
        if key in {"on", "true", "1"}:
            return True
    return bool(value)


def _fit_regression_target_transform(y, mode):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    info = {"enabled": False, "reason": "disabled"}
    if y.size == 0 or not np.isfinite(y).all():
        return None, {"enabled": False, "reason": "invalid_y"}
    key = str(mode or "none").strip().lower()
    if key in {"none", "off", "false", "0"}:
        return None, info
    if key == "auto":
        finite = y[np.isfinite(y)]
        if finite.size < 40:
            return None, {"enabled": False, "reason": "too_few_rows"}
        mean = float(np.mean(finite))
        std = float(np.std(finite))
        if std <= 1e-12:
            return None, {"enabled": False, "reason": "constant_target"}
        centered = finite - mean
        skew = float(np.mean(centered ** 3) / max(std ** 3, 1e-12))
        q50 = float(np.quantile(finite, 0.50))
        q90 = float(np.quantile(finite, 0.90))
        positive = float(np.min(finite)) >= 0.0
        heavy_tail = q90 >= max(abs(q50), 1e-12) * 2.5 or q90 >= q50 + 1.5 * std
        if positive and skew >= 1.0 and heavy_tail:
            key = "log1p_shift"
        else:
            return None, {
                "enabled": False,
                "reason": "auto_rejected",
                "skew": skew,
                "q90_over_q50": float(q90 / max(abs(q50), 1e-12)),
            }
    if key not in {"log1p_shift", "sqrt_shift"}:
        return None, {"enabled": False, "reason": f"unknown_transform:{mode}"}

    shift = max(0.0, -float(np.min(y)) + 1e-6)
    if key == "log1p_shift":
        z = np.log1p(y + shift)
    else:
        z = np.sqrt(np.maximum(y + shift, 0.0))
    if not np.isfinite(z).all() or float(np.std(z)) <= 1e-12:
        return None, {"enabled": False, "reason": "degenerate_transform"}
    state = {"transform": key, "shift": float(shift)}
    return state, {
        "enabled": True,
        "transform": key,
        "shift": float(shift),
        "target_std": float(np.std(y)),
        "transformed_std": float(np.std(z)),
    }


def _apply_regression_target_transform(state, y):
    if not state:
        return np.asarray(y, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    shift = float(state.get("shift", 0.0))
    if state.get("transform") == "log1p_shift":
        return np.log1p(y + shift)
    if state.get("transform") == "sqrt_shift":
        return np.sqrt(np.maximum(y + shift, 0.0))
    return y


def _invert_regression_target_transform(state, pred):
    pred = np.asarray(pred, dtype=np.float64)
    if not state:
        return pred
    shift = float(state.get("shift", 0.0))
    if state.get("transform") == "log1p_shift":
        return np.expm1(np.clip(pred, -50.0, 50.0)) - shift
    if state.get("transform") == "sqrt_shift":
        return np.maximum(pred, 0.0) ** 2 - shift
    return pred


def _fit_discrete_shadow_features(
    X_train,
    cat_features,
    task,
    mode="auto",
    max_cardinality=48,
    max_unique_frac=0.12,
    max_features=24,
):
    key = str(mode or "off").strip().lower()
    if key in {"off", "none", "false", "0"}:
        return None, list(cat_features), {"enabled": False, "reason": "disabled"}
    X = np.asarray(X_train, dtype=np.float64)
    cat_arr = np.asarray(cat_features, dtype=bool)
    if X.ndim != 2 or cat_arr.size != X.shape[1]:
        return None, list(cat_features), {"enabled": False, "reason": "bad_shape"}
    if key == "auto":
        # auto-scope DISABLED 2026-06-12: the no-cat-multiclass auto path was
        # measured -16.1% mean logloss harmful across 6 datasets when removed
        # (vehicle -34%, digits -31%, cmc -20%, yeast -10%) — the shadow columns
        # themselves hurt, plus they cascade into the cats>=2 mechanism block and
        # CFE via inflated auto-stats. Explicit discrete_shadow="on" still works.
        return None, list(cat_arr), {"enabled": False, "reason": "auto_disabled_2026_06_12"}

    max_card = max(2, int(max_cardinality))
    max_frac = float(np.clip(max_unique_frac, 0.01, 1.0))
    max_add = max(0, int(max_features))
    candidates = []
    for j in range(X.shape[1]):
        if bool(cat_arr[j]):
            continue
        vals = X[:, j]
        vals = vals[np.isfinite(vals)]
        if vals.size < 16:
            continue
        uniq = np.unique(vals)
        n_unique = int(uniq.size)
        if n_unique < 3 or n_unique > max_card:
            continue
        frac = n_unique / max(1.0, float(vals.size))
        if frac > max_frac and n_unique > 16:
            continue
        candidates.append((n_unique, j, frac, uniq.astype(np.float64)))
    if not candidates or max_add == 0:
        return None, list(cat_arr), {"enabled": False, "reason": "no_candidates"}

    candidates.sort(key=lambda row: (row[0], row[2], row[1]))
    selected = [(int(j), uniq) for _n, j, _frac, uniq in candidates[:max_add]]
    out_cat = list(cat_arr) + [True] * len(selected)
    state = {"selected": selected}
    return state, out_cat, {
        "enabled": True,
        "n_features": len(selected),
        "columns": [int(j) for j, _uniq in selected],
        "cardinalities": [int(len(uniq)) for _j, uniq in selected],
    }


def _apply_discrete_shadow_features(X, state):
    if not state:
        return np.asarray(X, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    shadows = []
    for j, uniq in state.get("selected", []):
        vals = X[:, int(j)]
        idx = np.searchsorted(uniq, vals)
        safe_idx = np.minimum(idx, max(len(uniq) - 1, 0))
        ok = (idx < len(uniq)) & np.isfinite(vals) & (uniq[safe_idx] == vals)
        codes = np.where(ok, idx, -1).astype(np.float64)
        shadows.append(codes[:, None])
    if not shadows:
        return X
    return np.hstack([X] + shadows).astype(np.float64)


def _softmax_2d(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=1, keepdims=True)
    z = np.clip(z, -35.0, 35.0)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def _fit_additive_init_state(X, y, task, cat_features, ridge=1.0, include_pairs=False):
    """Regularized binned additive initializer.

    This is deliberately small: it learns smooth one-feature effects from the
    training fold only, then the trees boost residual interactions.  It gives
    low-data problems an EBM-like prior without adding a second estimator or
    touching validation labels.
    """
    if task not in {"binary", "multiclass", "regression"}:
        return None, {"enabled": False, "reason": "unsupported_task"}
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    cat = np.asarray(cat_features, dtype=bool)
    if X.ndim != 2 or y.size != X.shape[0] or cat.size != X.shape[1] or X.shape[0] < 80:
        return None, {"enabled": False, "reason": "bad_shape"}
    n, p = X.shape
    max_bins = int(np.clip(np.sqrt(n), 8, 24))
    specs = [_hrc_bin_spec(X[:, j], bool(cat[j]), max_bins=max_bins) for j in range(p)]
    bins = [_hrc_bin_ids(X[:, j], specs[j]) for j in range(p)]
    smooth = max(float(ridge), 1e-6) * max(np.sqrt(n), 4.0)
    cycles = 3
    use_pairs = bool(include_pairs)

    if task == "regression":
        intercept = float(np.mean(y))
        pred = np.full(n, intercept, dtype=np.float64)
        effects = [np.zeros(spec["n_bins"], dtype=np.float64) for spec in specs]
        for _ in range(cycles):
            for j in range(p):
                bj = bins[j]
                pred -= effects[j][bj]
                resid = y - pred
                cnt = np.bincount(bj, minlength=effects[j].size).astype(np.float64)
                sums = np.bincount(bj, weights=resid, minlength=effects[j].size).astype(np.float64)
                eff = sums / (cnt + smooth)
                if cnt.sum() > 0:
                    eff -= float(np.sum(eff * cnt) / cnt.sum())
                eff = np.clip(eff, -5.0 * (np.std(y) + 1e-12), 5.0 * (np.std(y) + 1e-12))
                effects[j] = eff
                pred += effects[j][bj]
        return {
            "kind": "additive",
            "task": task,
            "specs": specs,
            "effects": [e.astype(np.float64) for e in effects],
            "intercept": intercept,
            "n_classes": 0,
        }, {
            "enabled": True,
            "task": task,
            "kind": "additive",
            "n_features": int(p),
            "ridge": float(ridge),
            "max_bins": int(max_bins),
        }

    if task == "binary":
        yy = y.astype(np.float64)
        if np.unique(yy).size != 2:
            return None, {"enabled": False, "reason": "not_binary_labels"}
        prior = np.clip(float(np.mean(yy)), 1e-5, 1.0 - 1e-5)
        intercept = float(np.log(prior / (1.0 - prior)))
        margin = np.full(n, intercept, dtype=np.float64)
        effects = [np.zeros(spec["n_bins"], dtype=np.float64) for spec in specs]
        pair_terms = []
        pair_ids = []
        pair_effects = []
        if use_pairs and p >= 2:
            feature_scores = []
            for j in range(p):
                bj = bins[j]
                cnt = np.bincount(bj, minlength=effects[j].size).astype(np.float64)
                sums = np.bincount(bj, weights=yy - prior, minlength=effects[j].size).astype(np.float64)
                score = float(np.sum((sums * sums) / (cnt + smooth + 1e-12)))
                feature_scores.append((score, j))
            feature_scores.sort(reverse=True)
            selected = [j for _, j in feature_scores[: min(14, p)]]
            candidates = []
            for pos_a, a in enumerate(selected):
                for b in selected[pos_a + 1:]:
                    card = int(specs[a]["n_bins"]) * int(specs[b]["n_bins"])
                    if card <= max(4 * n, 4096):
                        candidates.append((
                            feature_scores[pos_a][0] + next((s for s, jj in feature_scores if jj == b), 0.0),
                            card,
                            a,
                            b,
                        ))
            candidates.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
            for _score, card, a, b in candidates[:80]:
                nb = int(specs[b]["n_bins"])
                pair_terms.append((int(a), int(b), nb))
                pair_ids.append(bins[a] * nb + bins[b])
                pair_effects.append(np.zeros(int(card), dtype=np.float64))
        for _ in range(cycles):
            for j in range(p):
                bj = bins[j]
                margin -= effects[j][bj]
                prob = _sigmoid_raw(np.clip(margin, -35.0, 35.0))
                resid = yy - prob
                hess = np.maximum(prob * (1.0 - prob), 1e-4)
                den = np.bincount(bj, weights=hess, minlength=effects[j].size).astype(np.float64)
                sums = np.bincount(bj, weights=resid, minlength=effects[j].size).astype(np.float64)
                eff = sums / (den + smooth)
                weight = den + 1e-12
                eff -= float(np.sum(eff * weight) / np.sum(weight))
                effects[j] = np.clip(eff, -2.0, 2.0)
                margin += effects[j][bj]
            if pair_terms:
                pair_smooth = smooth * 10.0
                for t, pair_bin in enumerate(pair_ids):
                    margin -= pair_effects[t][pair_bin]
                    prob = _sigmoid_raw(np.clip(margin, -35.0, 35.0))
                    resid = yy - prob
                    hess = np.maximum(prob * (1.0 - prob), 1e-4)
                    den = np.bincount(
                        pair_bin,
                        weights=hess,
                        minlength=pair_effects[t].size,
                    ).astype(np.float64)
                    sums = np.bincount(
                        pair_bin,
                        weights=resid,
                        minlength=pair_effects[t].size,
                    ).astype(np.float64)
                    eff = sums / (den + pair_smooth)
                    weight = den + 1e-12
                    eff -= float(np.sum(eff * weight) / np.sum(weight))
                    pair_effects[t] = np.clip(eff, -1.5, 1.5)
                    margin += pair_effects[t][pair_bin]
        return {
            "kind": "additive",
            "task": task,
            "specs": specs,
            "effects": [e.astype(np.float64) for e in effects],
            "pair_terms": pair_terms,
            "pair_effects": [e.astype(np.float64) for e in pair_effects],
            "intercept": intercept,
            "n_classes": 0,
        }, {
            "enabled": True,
            "task": task,
            "kind": "additive_pairs" if pair_terms else "additive",
            "n_features": int(p),
            "ridge": float(ridge),
            "max_bins": int(max_bins),
            "n_pair_terms": int(len(pair_terms)),
        }

    yy = y.astype(np.int64)
    classes = np.unique(yy)
    if classes.size < 3 or classes[0] < 0 or classes[-1] + 1 != classes.size:
        return None, {"enabled": False, "reason": "bad_multiclass_labels"}
    n_classes = int(classes.size)
    Y = np.zeros((n, n_classes), dtype=np.float64)
    Y[np.arange(n), yy] = 1.0
    counts = np.bincount(yy, minlength=n_classes).astype(np.float64) + 1.0
    prior = counts / counts.sum()
    intercept = np.log(prior)
    intercept -= float(np.mean(intercept))
    margin = np.tile(intercept, (n, 1)).astype(np.float64)
    effects = [np.zeros((spec["n_bins"], n_classes), dtype=np.float64) for spec in specs]
    pair_terms = []
    pair_effects = []
    pair_ids = []
    if use_pairs and task == "multiclass" and bool(np.all(cat)) and p >= 2:
        candidates = []
        for a in range(p):
            for b in range(a + 1, p):
                card = int(specs[a]["n_bins"]) * int(specs[b]["n_bins"])
                if card <= max(4 * n, 4096):
                    candidates.append((card, a, b))
        candidates.sort(key=lambda row: (row[0], row[1], row[2]))
        for card, a, b in candidates[:64]:
            pair_terms.append((int(a), int(b), int(specs[b]["n_bins"])))
            pair_ids.append(bins[a] * int(specs[b]["n_bins"]) + bins[b])
            pair_effects.append(np.zeros((int(card), n_classes), dtype=np.float64))

    for _ in range(cycles):
        for j in range(p):
            bj = bins[j]
            margin -= effects[j][bj]
            prob = _softmax_2d(margin)
            resid = Y - prob
            hess = np.maximum(prob * (1.0 - prob), 1e-4)
            eff = np.zeros_like(effects[j])
            den_total = np.zeros(effects[j].shape[0], dtype=np.float64)
            for k in range(n_classes):
                den = np.bincount(bj, weights=hess[:, k], minlength=effects[j].shape[0]).astype(np.float64)
                sums = np.bincount(bj, weights=resid[:, k], minlength=effects[j].shape[0]).astype(np.float64)
                eff[:, k] = sums / (den + smooth)
                den_total += den
            eff -= np.mean(eff, axis=1, keepdims=True)
            weight = den_total + 1e-12
            eff -= np.sum(eff * weight[:, None], axis=0, keepdims=True) / np.sum(weight)
            effects[j] = np.clip(eff, -2.0, 2.0)
            margin += effects[j][bj]
        if pair_terms:
            pair_smooth = smooth * 8.0
            for t, pair_bin in enumerate(pair_ids):
                margin -= pair_effects[t][pair_bin]
                prob = _softmax_2d(margin)
                resid = Y - prob
                hess = np.maximum(prob * (1.0 - prob), 1e-4)
                eff = np.zeros_like(pair_effects[t])
                den_total = np.zeros(eff.shape[0], dtype=np.float64)
                for k in range(n_classes):
                    den = np.bincount(
                        pair_bin,
                        weights=hess[:, k],
                        minlength=eff.shape[0],
                    ).astype(np.float64)
                    sums = np.bincount(
                        pair_bin,
                        weights=resid[:, k],
                        minlength=eff.shape[0],
                    ).astype(np.float64)
                    eff[:, k] = sums / (den + pair_smooth)
                    den_total += den
                eff -= np.mean(eff, axis=1, keepdims=True)
                weight = den_total + 1e-12
                eff -= np.sum(eff * weight[:, None], axis=0, keepdims=True) / np.sum(weight)
                pair_effects[t] = np.clip(eff, -1.5, 1.5)
                margin += pair_effects[t][pair_bin]
    return {
        "kind": "additive",
        "task": task,
        "specs": specs,
        "effects": [e.astype(np.float64) for e in effects],
        "pair_terms": pair_terms,
        "pair_effects": [e.astype(np.float64) for e in pair_effects],
        "intercept": intercept.astype(np.float64),
        "n_classes": int(n_classes),
    }, {
        "enabled": True,
        "task": task,
        "kind": "additive_pairs" if pair_terms else "additive",
        "n_features": int(p),
        "ridge": float(ridge),
        "max_bins": int(max_bins),
        "n_classes": int(n_classes),
        "n_pair_terms": int(len(pair_terms)),
    }


def _fit_cat_backoff_init_state(X, y, task, cat_features, ridge=1.0):
    """Leak-safe categorical backoff prior for classification.

    The state stores smoothed class posteriors for small categorical subsets.
    Prediction averages only matching subset residual logits with empirical-Bayes
    shrinkage, so unseen keys fall back to the global prior.
    """
    if task not in {"binary", "multiclass"}:
        return None, {"enabled": False, "reason": "unsupported_task"}
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).reshape(-1)
    cat = np.asarray(cat_features, dtype=bool)
    if X.ndim != 2 or y.size != X.shape[0] or cat.size != X.shape[1] or X.shape[0] < 80:
        return None, {"enabled": False, "reason": "bad_shape"}
    cat_idx = np.flatnonzero(cat)
    if cat_idx.size == 0:
        return None, {"enabled": False, "reason": "no_categorical_features"}
    if task == "binary":
        yy = y.astype(np.int64)
        if np.unique(yy).size != 2:
            return None, {"enabled": False, "reason": "not_binary_labels"}
        n_classes = 2
    else:
        yy = y.astype(np.int64)
        classes = np.unique(yy)
        if classes.size < 3 or classes[0] < 0 or classes[-1] + 1 != classes.size:
            return None, {"enabled": False, "reason": "bad_multiclass_labels"}
        n_classes = int(classes.size)

    n = int(X.shape[0])
    counts = np.bincount(yy, minlength=n_classes).astype(np.float64) + 1.0
    prior = counts / counts.sum()

    def codes_for_feature(j):
        col = X[:, int(j)]
        return np.where(np.isfinite(col), np.rint(col).astype(np.int64), -1)

    codes = {int(j): codes_for_feature(j) for j in cat_idx}
    # Generic feature screening for high-dimensional categorical blocks.
    feature_scores = []
    for j in cat_idx:
        cj = codes[int(j)]
        table = {}
        for v, cls in zip(cj, yy):
            arr = table.setdefault(int(v), np.zeros(n_classes, dtype=np.float64))
            arr[int(cls)] += 1.0
        score = 0.0
        for arr in table.values():
            total = float(np.sum(arr))
            if total <= 0.0:
                continue
            expected = total * prior
            score += float(np.sum((arr - expected) * (arr - expected) / (expected + 1e-9)))
        feature_scores.append((score, int(j)))
    feature_scores.sort(reverse=True)
    selected = [j for _, j in feature_scores[: min(10, len(feature_scores))]]
    if not selected:
        return None, {"enabled": False, "reason": "no_selected_features"}

    import itertools

    subsets = []
    max_order = 3 if len(selected) >= 3 else len(selected)
    for order in range(1, max_order + 1):
        for feats in itertools.combinations(selected, order):
            subsets.append(tuple(int(f) for f in feats))
    if len(subsets) > 160:
        # Keep lower-order terms and the strongest high-order candidates by
        # marginal feature score. This is a generic compute budget.
        score_map = {j: s for s, j in feature_scores}
        subsets.sort(key=lambda fs: (len(fs), sum(score_map.get(f, 0.0) for f in fs)), reverse=True)
        subsets = subsets[:160]

    tables = []
    min_total = 2.0
    for feats in subsets:
        tbl = {}
        for row, cls in enumerate(yy):
            key = tuple(int(codes[f][row]) for f in feats)
            arr = tbl.setdefault(key, np.zeros(n_classes, dtype=np.float64))
            arr[int(cls)] += 1.0
        tbl = {k: v for k, v in tbl.items() if float(np.sum(v)) >= min_total}
        if tbl:
            tables.append((feats, tbl))
    if not tables:
        return None, {"enabled": False, "reason": "empty_tables"}

    return {
        "kind": "cat_backoff",
        "task": task,
        "selected": selected,
        "tables": tables,
        "prior": prior.astype(np.float64),
        "n_classes": int(n_classes),
        "ridge": float(ridge),
        "n_train": int(n),
    }, {
        "enabled": True,
        "task": task,
        "kind": "cat_backoff",
        "n_classes": int(n_classes),
        "n_features": int(len(selected)),
        "n_tables": int(len(tables)),
        "ridge": float(ridge),
    }


def _knn_prepare_matrix(X, cat_features=None, mean=None, std=None):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        return None, None, None
    if mean is None:
        mean = np.nanmean(X, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
    if std is None:
        std = np.nanstd(X, axis=0)
        std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
        cat = np.asarray(cat_features if cat_features is not None else [], dtype=bool)
        if cat.size == X.shape[1]:
            # Categorical codes are arbitrary labels; keep them at unit scale so
            # one categorical mismatch is not dominated by a high-cardinality id.
            std = np.where(cat, 1.0, std)
    Z = (X - mean) / std
    Z = np.where(np.isfinite(Z), Z, 0.0)
    return np.ascontiguousarray(Z, dtype=np.float64), mean.astype(np.float64), std.astype(np.float64)


def _knn_init_scores_from_state(state, X):
    train = np.asarray(state.get("train_z"), dtype=np.float64)
    y = np.asarray(state.get("y"), dtype=np.int64)
    mean = np.asarray(state.get("mean"), dtype=np.float64)
    std = np.asarray(state.get("std"), dtype=np.float64)
    task = str(state.get("task"))
    if train.ndim != 2 or y.size != train.shape[0] or y.size < 4:
        return None
    Z, _, _ = _knn_prepare_matrix(X, mean=mean, std=std)
    if Z is None or Z.shape[1] != train.shape[1]:
        return None
    k = int(np.clip(int(state.get("k", 15)), 1, train.shape[0]))
    smooth = float(max(state.get("smooth", 2.0), 1e-6))
    batch = 256
    if task == "binary":
        prior = np.clip(float(state.get("prior", np.mean(y))), 1e-6, 1.0 - 1e-6)
        out = np.empty(Z.shape[0], dtype=np.float64)
        yy = y.astype(np.float64)
        for start in range(0, Z.shape[0], batch):
            Q = Z[start:start + batch]
            d2 = np.sum((Q[:, None, :] - train[None, :, :]) ** 2, axis=2)
            nn = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
            cnt = np.sum(yy[nn], axis=1)
            p = (cnt + smooth * prior) / (k + smooth)
            p = np.clip(p, 1e-6, 1.0 - 1e-6)
            out[start:start + Q.shape[0]] = np.log(p / (1.0 - p))
        scale = float(state.get("audit_scale", 1.0))
        if scale < 0.999999:
            base = np.log(prior / (1.0 - prior))
            out = base + np.clip(scale, 0.0, 1.0) * (out - base)
        return np.ascontiguousarray(np.clip(out, -8.0, 8.0), dtype=np.float64)
    if task == "multiclass":
        n_classes = int(state.get("n_classes", 0))
        prior = np.asarray(state.get("prior"), dtype=np.float64)
        if n_classes <= 1 or prior.size != n_classes:
            return None
        prior = np.clip(prior, 1e-9, 1.0)
        prior /= np.sum(prior)
        out = np.empty((Z.shape[0], n_classes), dtype=np.float64)
        for start in range(0, Z.shape[0], batch):
            Q = Z[start:start + batch]
            d2 = np.sum((Q[:, None, :] - train[None, :, :]) ** 2, axis=2)
            nn = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
            counts = np.zeros((Q.shape[0], n_classes), dtype=np.float64)
            for cls in range(n_classes):
                counts[:, cls] = np.sum(y[nn] == cls, axis=1)
            p = (counts + smooth * prior[None, :]) / (k + smooth)
            p = np.clip(p, 1e-9, 1.0)
            p /= np.sum(p, axis=1, keepdims=True)
            logits = np.log(p)
            logits -= np.mean(logits, axis=1, keepdims=True)
            out[start:start + Q.shape[0]] = logits
        scale = float(state.get("audit_scale", 1.0))
        if scale < 0.999999:
            base = np.log(prior)
            base -= float(np.mean(base))
            out = base[None, :] + np.clip(scale, 0.0, 1.0) * (out - base[None, :])
        out = np.clip(np.where(np.isfinite(out), out, 0.0), -8.0, 8.0)
        return np.ascontiguousarray(out.reshape(-1), dtype=np.float64)
    return None


def _fit_knn_init_state(X, y, task, cat_features, ridge=15.0):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).reshape(-1)
    if task not in {"binary", "multiclass"} or X.ndim != 2 or y.size != X.shape[0] or y.size < 80:
        return None, {"enabled": False, "reason": "knn_scope"}
    if X.shape[0] > 5_000 or X.shape[1] > 256:
        return None, {"enabled": False, "reason": "knn_size"}
    if task == "binary" and np.unique(y).size != 2:
        return None, {"enabled": False, "reason": "not_binary_labels"}
    n_classes = 0
    if task == "multiclass":
        yy = y.astype(np.int64)
        classes = np.unique(yy)
        if classes.size < 3 or classes[0] < 0 or classes[-1] + 1 != classes.size:
            return None, {"enabled": False, "reason": "bad_multiclass_labels"}
        n_classes = int(classes.size)

    k = int(np.clip(round(float(ridge)), 3, min(101, max(3, y.size - 1))))
    if k % 2 == 0:
        k += 1
    smooth = max(2.0, 0.5 * np.sqrt(float(k)))
    base_score = _base_init_score_for_audit(y, task, n_classes)
    base_loss = _init_audit_loss(y, base_score, task, n_classes)
    if not np.isfinite(base_loss):
        return None, {"enabled": False, "reason": "bad_base_loss"}

    fold_scores = np.zeros(y.size * max(n_classes, 1), dtype=np.float64)
    ok = True
    for val_idx in _audit_folds(y, task, n_folds=3):
        train_mask = np.ones(y.size, dtype=bool)
        train_mask[val_idx] = False
        if int(np.sum(train_mask)) <= k:
            ok = False
            break
        Ztr, mean, std = _knn_prepare_matrix(X[train_mask], cat_features)
        state = {
            "kind": "knn_prior",
            "task": task,
            "train_z": Ztr,
            "y": y[train_mask].astype(np.int64),
            "mean": mean,
            "std": std,
            "k": min(k, int(np.sum(train_mask))),
            "smooth": smooth,
        }
        if task == "binary":
            state["prior"] = float(np.mean(y[train_mask].astype(np.float64)))
        else:
            counts = np.bincount(y[train_mask].astype(np.int64), minlength=n_classes).astype(np.float64) + 1.0
            state["prior"] = counts / np.sum(counts)
            state["n_classes"] = n_classes
        pred = _knn_init_scores_from_state(state, X[val_idx])
        if pred is None or not np.all(np.isfinite(pred)):
            ok = False
            break
        if task == "multiclass":
            block = pred.reshape(val_idx.size, n_classes)
            for out_pos, row in enumerate(val_idx):
                start = int(row) * n_classes
                fold_scores[start:start + n_classes] = block[out_pos]
        else:
            fold_scores[val_idx] = pred.reshape(-1)
    if not ok:
        return None, {"enabled": False, "reason": "knn_oof_failed", "k": int(k)}

    best_loss = float("inf")
    best_scale = 0.0
    for alpha in [0.05, 0.10, 0.20, 0.35, 0.50]:
        scaled = _blend_init_scores(base_score, fold_scores, alpha)
        loss = _init_audit_loss(y, scaled, task, n_classes)
        if loss < best_loss:
            best_loss = loss
            best_scale = float(alpha)
    rel_gain = (base_loss - best_loss) / max(abs(base_loss), 1e-12)
    if not np.isfinite(best_loss) or rel_gain < 0.0025:
        return None, {
            "enabled": False,
            "reason": "knn_audit_rejected",
            "base_loss": float(base_loss),
            "best_loss": float(best_loss),
            "relative_gain": float(rel_gain),
            "k": int(k),
        }

    Z, mean, std = _knn_prepare_matrix(X, cat_features)
    state = {
        "kind": "knn_prior",
        "task": task,
        "train_z": Z,
        "y": y.astype(np.int64),
        "mean": mean,
        "std": std,
        "k": int(k),
        "smooth": float(smooth),
        "audit_scale": float(best_scale),
        "_audit_oof_score": np.ascontiguousarray(fold_scores, dtype=np.float64),
        "_audit_oof_n_rows": int(X.shape[0]),
        "_audit_oof_n_cols": int(X.shape[1]),
    }
    if task == "binary":
        state["prior"] = float(np.mean(y.astype(np.float64)))
    else:
        counts = np.bincount(y.astype(np.int64), minlength=n_classes).astype(np.float64) + 1.0
        state["prior"] = counts / np.sum(counts)
        state["n_classes"] = int(n_classes)
    return state, {
        "enabled": True,
        "task": task,
        "kind": "knn_prior",
        "k": int(k),
        "audit_scale": float(best_scale),
        "base_loss": float(base_loss),
        "audit_loss": float(best_loss),
        "relative_gain": float(rel_gain),
    }


def _init_audit_loss(y, score, task, n_classes=0):
    y = np.asarray(y).reshape(-1)
    if task == "regression":
        pred = np.asarray(score, dtype=np.float64).reshape(-1)
        if pred.size != y.size:
            return float("inf")
        resid = y.astype(np.float64) - pred
        return float(np.mean(resid * resid))
    if task == "binary":
        margin = np.asarray(score, dtype=np.float64).reshape(-1)
        if margin.size != y.size:
            return float("inf")
        p = _sigmoid_raw(np.clip(margin, -35.0, 35.0))
        yy = y.astype(np.float64)
        return -float(np.mean(yy * np.log(np.clip(p, 1e-15, 1.0)) + (1.0 - yy) * np.log(np.clip(1.0 - p, 1e-15, 1.0))))
    if task == "multiclass":
        k = int(n_classes)
        yy = y.astype(np.int64)
        if k <= 1 or yy.size == 0:
            return float("inf")
        logits = np.asarray(score, dtype=np.float64)
        if logits.size != yy.size * k:
            return float("inf")
        logits = logits.reshape(yy.size, k)
        p = _softmax_2d(logits)
        return -float(np.mean(np.log(np.clip(p[np.arange(yy.size), yy], 1e-15, 1.0))))
    return float("inf")


def _base_init_score_for_audit(y, task, n_classes=0):
    y = np.asarray(y).reshape(-1)
    if task == "regression":
        return np.full(y.size, float(np.mean(y.astype(np.float64))), dtype=np.float64)
    if task == "binary":
        p = np.clip(float(np.mean(y.astype(np.float64))), 1e-5, 1.0 - 1e-5)
        return np.full(y.size, np.log(p / (1.0 - p)), dtype=np.float64)
    if task == "multiclass":
        k = int(n_classes)
        yy = y.astype(np.int64)
        counts = np.bincount(yy, minlength=k).astype(np.float64) + 1.0
        prior = counts / counts.sum()
        margin = np.log(prior)
        margin -= float(np.mean(margin))
        return np.tile(margin, (yy.size, 1)).reshape(-1).astype(np.float64)
    return None


def _blend_init_scores(base_score, raw_score, alpha):
    base = np.asarray(base_score, dtype=np.float64)
    raw = np.asarray(raw_score, dtype=np.float64)
    if base.shape != raw.shape:
        return raw
    a = float(np.clip(alpha, 0.0, 1.0))
    return np.ascontiguousarray(base + a * (raw - base), dtype=np.float64)


def _audit_folds(y, task, n_folds=3, seed=7919):
    y = np.asarray(y).reshape(-1)
    n = y.size
    n_folds = int(np.clip(n_folds, 2, min(5, max(2, n))))
    folds = [[] for _ in range(n_folds)]
    rng = np.random.default_rng(seed)
    if task in {"binary", "multiclass"}:
        for cls in np.unique(y):
            idx = np.flatnonzero(y == cls)
            rng.shuffle(idx)
            for pos, row in enumerate(idx):
                folds[pos % n_folds].append(int(row))
    else:
        idx = np.arange(n)
        rng.shuffle(idx)
        for pos, row in enumerate(idx):
            folds[pos % n_folds].append(int(row))
    out = [np.asarray(f, dtype=int) for f in folds if len(f) > 0]
    return out if len(out) >= 2 else [np.arange(n, dtype=int)]


def _fit_audited_init_state(X, y, task, cat_features, ridge=1.0):
    """Training-only prior selection: accept a simple prior only if OOF loss beats base."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).reshape(-1)
    cat_features = list(cat_features)
    if X.ndim != 2 or y.size != X.shape[0] or X.shape[0] < 120:
        return None, {"enabled": False, "reason": "audit_scope"}
    n_classes = 0
    if task == "multiclass":
        yy = y.astype(np.int64)
        classes = np.unique(yy)
        if classes.size < 3 or classes[0] < 0 or classes[-1] + 1 != classes.size:
            return None, {"enabled": False, "reason": "bad_multiclass_labels"}
        n_classes = int(classes.size)
    elif task == "binary" and np.unique(y).size != 2:
        return None, {"enabled": False, "reason": "not_binary_labels"}

    base_score = _base_init_score_for_audit(y, task, n_classes)
    base_loss = _init_audit_loss(y, base_score, task, n_classes)
    if not np.isfinite(base_loss):
        return None, {"enabled": False, "reason": "bad_base_loss"}

    candidate_modes = ["linear", "additive", "additive_pairs"]
    candidate_losses = {}
    candidate_oof_scores = {}
    candidate_scales = {}
    folds = _audit_folds(y, task, n_folds=3)
    for mode in candidate_modes:
        mode_ridge = ridge * (32.0 if mode == "additive_pairs" else 1.0)
        fold_scores = None
        ok = True
        if task == "multiclass":
            fold_scores = np.zeros(y.size * n_classes, dtype=np.float64)
        else:
            fold_scores = np.zeros(y.size, dtype=np.float64)
        for val_idx in folds:
            train_mask = np.ones(y.size, dtype=bool)
            train_mask[val_idx] = False
            if int(np.sum(train_mask)) < 40:
                ok = False
                break
            state, info = _fit_linear_init_state(
                X[train_mask],
                y[train_mask],
                task,
                cat_features,
                mode=mode,
                ridge=mode_ridge,
            )
            if not state or not info.get("enabled", False):
                ok = False
                break
            pred = _linear_init_score(state, X[val_idx])
            if pred is None or not np.all(np.isfinite(pred)):
                ok = False
                break
            if task == "multiclass":
                block = np.asarray(pred, dtype=np.float64).reshape(val_idx.size, n_classes)
                for out_pos, row in enumerate(val_idx):
                    start = int(row) * n_classes
                    fold_scores[start:start + n_classes] = block[out_pos]
            else:
                fold_scores[val_idx] = np.asarray(pred, dtype=np.float64).reshape(-1)
        if ok:
            if task in {"binary", "multiclass"}:
                if mode == "linear":
                    scales = [0.10, 0.20, 0.35, 0.50]
                elif mode == "additive_pairs":
                    scales = [0.10, 0.20, 0.35, 0.50, 0.70, 1.0]
                else:
                    scales = [0.05, 0.10, 0.20, 0.35]
            else:
                scales = [0.25, 0.50, 0.75, 1.0] if mode == "linear" else [0.10, 0.20, 0.35, 0.50]
            best_scaled_loss = float("inf")
            best_scaled_score = None
            best_scale = 0.0
            for alpha in scales:
                scaled = _blend_init_scores(base_score, fold_scores, alpha)
                loss = _init_audit_loss(y, scaled, task, n_classes)
                if loss < best_scaled_loss:
                    best_scaled_loss = loss
                    best_scaled_score = scaled
                    best_scale = float(alpha)
            candidate_losses[mode] = best_scaled_loss
            candidate_oof_scores[mode] = best_scaled_score
            candidate_scales[mode] = best_scale

    if not candidate_losses:
        return None, {"enabled": False, "reason": "audit_no_candidate", "base_loss": float(base_loss)}
    best_mode, best_loss = min(candidate_losses.items(), key=lambda kv: kv[1])
    best_scale = float(candidate_scales.get(best_mode, 1.0))
    rel_gain = (base_loss - best_loss) / max(abs(base_loss), 1e-12)
    min_gain = 0.0025 if task != "regression" else 0.005
    if not np.isfinite(best_loss) or rel_gain < min_gain:
        return None, {
            "enabled": False,
            "reason": "audit_rejected",
            "base_loss": float(base_loss),
            "best_loss": float(best_loss),
            "best_mode": str(best_mode),
            "relative_gain": float(rel_gain),
        }
    state, info = _fit_linear_init_state(
        X,
        y,
        task,
        cat_features,
        mode=best_mode,
        ridge=ridge * (32.0 if best_mode == "additive_pairs" else 1.0),
    )
    if not state or not info.get("enabled", False):
        return None, {"enabled": False, "reason": "final_fit_failed", "best_mode": str(best_mode)}
    state["audit_scale"] = best_scale
    oof_score = candidate_oof_scores.get(best_mode)
    if oof_score is not None and np.asarray(oof_score).size:
        state["_audit_oof_score"] = np.ascontiguousarray(oof_score, dtype=np.float64)
        state["_audit_oof_n_rows"] = int(X.shape[0])
        state["_audit_oof_n_cols"] = int(X.shape[1])
    info = dict(info)
    info.update({
        "kind": "audited_" + str(info.get("kind", best_mode)),
        "audit_mode": str(best_mode),
        "audit_scale": float(best_scale),
        "base_loss": float(base_loss),
        "audit_loss": float(best_loss),
        "relative_gain": float(rel_gain),
    })
    return state, info


def _fit_linear_init_state(X, y, task, cat_features, mode="auto", ridge=1.0):
    key = str(mode or "off").strip().lower()
    if key in {"off", "none", "false", "0"}:
        return None, {"enabled": False, "reason": "disabled"}
    if key in {"knn_prior", "knn", "nearest"}:
        return _fit_knn_init_state(X, y, task, cat_features, ridge=ridge)
    if key in {"cat_backoff", "categorical_backoff", "backoff"}:
        return _fit_cat_backoff_init_state(X, y, task, cat_features, ridge=ridge)
    if key in {"auto_prior", "prior", "audited", "audited_prior"}:
        return _fit_audited_init_state(X, y, task, cat_features, ridge=ridge)
    if key in {"additive_pairs", "pair_additive", "binned_additive_pairs"}:
        return _fit_additive_init_state(X, y, task, cat_features, ridge=ridge, include_pairs=True)
    if key in {"additive", "gam", "binned_additive"}:
        return _fit_additive_init_state(X, y, task, cat_features, ridge=ridge)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    cat_arr = np.asarray(cat_features, dtype=bool)
    if X.ndim != 2 or y.size != X.shape[0] or cat_arr.size != X.shape[1]:
        return None, {"enabled": False, "reason": "bad_shape"}
    if task not in {"binary", "multiclass", "regression"}:
        return None, {"enabled": False, "reason": "unsupported_task"}
    numeric = np.array([i for i, is_cat in enumerate(cat_arr) if not is_cat], dtype=int)
    if key == "auto" and task == "multiclass" and (X.shape[0] < 80 or X.shape[1] > 64):
        return None, {"enabled": False, "reason": "auto_scope"}
    if key == "auto" and task != "multiclass":
        high_dim_binary = task == "binary" and numeric.size >= 16
        if X.shape[0] < 80 or (not high_dim_binary and (bool(cat_arr.any()) or X.shape[1] > 32)):
            return None, {"enabled": False, "reason": "auto_scope"}
    if numeric.size:
        Xn = X[:, numeric]
        mean = np.nanmean(Xn, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.nanstd(Xn, axis=0)
        std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
        Z = (Xn - mean) / std
        Z = np.where(np.isfinite(Z), Z, 0.0)
    else:
        mean = np.zeros(0, dtype=np.float64)
        std = np.ones(0, dtype=np.float64)
        Z = np.empty((X.shape[0], 0), dtype=np.float64)

    if numeric.size == 0:
        return None, {"enabled": False, "reason": "no_linear_features"}

    A = np.column_stack([np.ones(X.shape[0], dtype=np.float64), Z])
    if not np.all(np.isfinite(A)):
        return None, {"enabled": False, "reason": "nonfinite_design"}
    lam = float(max(ridge, 1e-8))
    penalty = np.eye(A.shape[1], dtype=np.float64) * lam
    penalty[0, 0] = 0.0

    if task == "regression":
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            lhs = A.T @ A + penalty
            rhs = A.T @ y
        if not (np.all(np.isfinite(lhs)) and np.all(np.isfinite(rhs))):
            return None, {"enabled": False, "reason": "nonfinite_normal_eq"}
        try:
            coef = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            return None, {"enabled": False, "reason": "singular"}
    elif task == "binary":
        yy = y.astype(np.float64)
        if np.unique(yy).size != 2:
            return None, {"enabled": False, "reason": "not_binary_labels"}
        p0 = np.clip(float(np.mean(yy)), 1e-5, 1.0 - 1e-5)
        coef = np.zeros(A.shape[1], dtype=np.float64)
        coef[0] = np.log(p0 / (1.0 - p0))
        for _ in range(25):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                margin = np.clip(A @ coef, -35.0, 35.0)
                p = _sigmoid_raw(margin)
                w = np.clip(p * (1.0 - p), 1e-6, None)
                grad = A.T @ (p - yy) + penalty @ coef
                hess = (A.T * w) @ A + penalty
            if not (np.all(np.isfinite(grad)) and np.all(np.isfinite(hess))):
                return None, {"enabled": False, "reason": "nonfinite_newton"}
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                return None, {"enabled": False, "reason": "singular"}
            if not np.all(np.isfinite(step)):
                return None, {"enabled": False, "reason": "nonfinite_step"}
            max_step = float(np.max(np.abs(step))) if step.size else 0.0
            if max_step > 3.0:
                step *= 3.0 / max_step
            coef -= step
            if not np.all(np.isfinite(coef)):
                return None, {"enabled": False, "reason": "nonfinite_coef"}
            if float(np.max(np.abs(step))) < 1e-5:
                break
    else:
        yy = y.astype(np.int64)
        classes = np.unique(yy)
        if classes.size < 3 or classes[0] < 0 or classes[-1] + 1 != classes.size:
            return None, {"enabled": False, "reason": "bad_multiclass_labels"}
        n_classes = int(classes.size)
        Y = np.zeros((X.shape[0], n_classes), dtype=np.float64)
        Y[np.arange(X.shape[0]), yy] = 1.0
        counts = np.bincount(yy, minlength=n_classes).astype(np.float64) + 1.0
        prior = counts / counts.sum()
        coef = np.zeros((A.shape[1], n_classes), dtype=np.float64)
        coef[0, :] = np.log(prior)
        coef[0, :] -= float(np.mean(coef[0, :]))
        try:
            svals = np.linalg.svd(A, compute_uv=False)
            s2 = float(svals[0] * svals[0]) / max(X.shape[0], 1)
        except np.linalg.LinAlgError:
            s2 = float(np.sum(A * A)) / max(X.shape[0], 1)
        step_size = 1.0 / max(0.5 * s2 + lam / max(X.shape[0], 1), 1e-6)
        penalty_mc = np.ones_like(coef) * (lam / max(X.shape[0], 1))
        penalty_mc[0, :] = 0.0
        prev_loss = f64_inf = float("inf")
        for _ in range(220):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                logits = A @ coef
            logits = logits - np.max(logits, axis=1, keepdims=True)
            logits = np.clip(logits, -35.0, 35.0)
            exp = np.exp(logits)
            P = exp / np.sum(exp, axis=1, keepdims=True)
            if not np.all(np.isfinite(P)):
                return None, {"enabled": False, "reason": "nonfinite_softmax"}
            grad = (A.T @ (P - Y)) / max(X.shape[0], 1) + penalty_mc * coef
            if not np.all(np.isfinite(grad)):
                return None, {"enabled": False, "reason": "nonfinite_grad"}
            coef -= step_size * grad
            coef -= np.mean(coef, axis=1, keepdims=True)
            if not np.all(np.isfinite(coef)):
                return None, {"enabled": False, "reason": "nonfinite_coef"}
            if float(np.max(np.abs(step_size * grad))) < 1e-5:
                break
            if _ % 20 == 19:
                loss = -float(np.mean(np.log(np.clip(P[np.arange(X.shape[0]), yy], 1e-15, 1.0))))
                if loss > prev_loss + 1e-6:
                    step_size *= 0.5
                prev_loss = min(prev_loss, loss)
    if not np.all(np.isfinite(coef)) or float(np.max(np.abs(coef))) > 1e6:
        return None, {"enabled": False, "reason": "unstable_coef"}

    return {
        "task": task,
        "numeric": numeric.astype(int),
        "mean": mean.astype(np.float64),
        "std": std.astype(np.float64),
        "coef": coef.astype(np.float64),
        "n_classes": int(coef.shape[1]) if getattr(coef, "ndim", 1) == 2 else 0,
    }, {
        "enabled": True,
        "task": task,
        "n_features": int(numeric.size),
        "ridge": float(lam),
        "n_classes": int(coef.shape[1]) if getattr(coef, "ndim", 1) == 2 else 0,
    }


def _linear_init_score(state, X):
    if not state:
        return None
    X = np.asarray(X, dtype=np.float64)
    if str(state.get("kind", "linear")).lower() == "knn_prior":
        return _knn_init_scores_from_state(state, X)
    if str(state.get("kind", "linear")).lower() == "cat_backoff":
        n_classes = int(state.get("n_classes", 0))
        prior = np.asarray(state.get("prior"), dtype=np.float64)
        if n_classes < 2 or prior.size != n_classes:
            return None
        prior = np.clip(prior, 1e-9, 1.0)
        prior /= np.sum(prior)
        log_prior = np.log(prior)
        out = np.tile(log_prior, (X.shape[0], 1))
        tables = state.get("tables", [])
        n_train = max(int(state.get("n_train", 1)), 1)
        ridge = max(float(state.get("ridge", 1.0)), 1e-6)
        alpha_base = max(2.0, ridge * np.sqrt(float(n_train)))
        for row in range(X.shape[0]):
            delta = np.zeros(n_classes, dtype=np.float64)
            weight_sum = alpha_base
            for feats, tbl in tables:
                key = []
                ok = True
                for f in feats:
                    f = int(f)
                    if f >= X.shape[1] or not np.isfinite(X[row, f]):
                        ok = False
                        break
                    key.append(int(np.rint(X[row, f])))
                if not ok:
                    continue
                arr = tbl.get(tuple(key))
                if arr is None:
                    continue
                arr = np.asarray(arr, dtype=np.float64)
                total = float(np.sum(arr))
                if total < 2.0:
                    continue
                order = max(len(feats), 1)
                alpha = alpha_base / float(order * order)
                post = (arr + alpha * prior) / (total + alpha)
                post = np.clip(post, 1e-9, 1.0)
                post /= np.sum(post)
                w = float(order) * np.log1p(total)
                delta += w * (np.log(post) - log_prior)
                weight_sum += w
            out[row] = log_prior + delta / max(weight_sum, 1e-12)
        if n_classes == 2 and str(state.get("task")) == "binary":
            margin = out[:, 1] - out[:, 0]
            scale = float(state.get("audit_scale", 1.0))
            if scale < 0.999999:
                base_margin = log_prior[1] - log_prior[0]
                margin = base_margin + np.clip(scale, 0.0, 1.0) * (margin - base_margin)
            margin = np.clip(np.where(np.isfinite(margin), margin, 0.0), -8.0, 8.0)
            return np.ascontiguousarray(margin, dtype=np.float64)
        scale = float(state.get("audit_scale", 1.0))
        if scale < 0.999999:
            base = np.tile(log_prior, (X.shape[0], 1))
            out = base + np.clip(scale, 0.0, 1.0) * (out - base)
        out -= np.mean(out, axis=1, keepdims=True)
        out = np.clip(np.where(np.isfinite(out), out, 0.0), -8.0, 8.0)
        return np.ascontiguousarray(out.reshape(-1), dtype=np.float64)
    if str(state.get("kind", "linear")).lower() == "additive":
        specs = state.get("specs", [])
        effects = state.get("effects", [])
        task = str(state.get("task"))
        if task == "multiclass":
            n_classes = int(state.get("n_classes", 0))
            if n_classes <= 1:
                return None
            intercept = np.asarray(state.get("intercept"), dtype=np.float64)
            score = np.tile(intercept, (X.shape[0], 1))
            for j, (spec, eff) in enumerate(zip(specs, effects)):
                if j >= X.shape[1]:
                    break
                ids = _hrc_bin_ids(X[:, j], spec)
                score += np.asarray(eff, dtype=np.float64)[ids]
            for term, eff in zip(state.get("pair_terms", []), state.get("pair_effects", [])):
                if len(term) < 3:
                    continue
                a, b, nb = int(term[0]), int(term[1]), int(term[2])
                if a >= X.shape[1] or b >= X.shape[1] or a >= len(specs) or b >= len(specs):
                    continue
                ia = _hrc_bin_ids(X[:, a], specs[a])
                ib = _hrc_bin_ids(X[:, b], specs[b])
                pid = ia * max(int(nb), 1) + ib
                table = np.asarray(eff, dtype=np.float64)
                if table.ndim == 2 and table.shape[0] > int(np.max(pid, initial=0)):
                    score += table[pid]
            scale = float(state.get("audit_scale", 1.0))
            if scale < 0.999999:
                base = np.tile(intercept, (X.shape[0], 1))
                score = base + np.clip(scale, 0.0, 1.0) * (score - base)
            score -= np.mean(score, axis=1, keepdims=True)
            score = np.clip(score, -8.0, 8.0)
            return np.ascontiguousarray(score.reshape(-1), dtype=np.float64)
        intercept = float(state.get("intercept", 0.0))
        score = np.full(X.shape[0], intercept, dtype=np.float64)
        for j, (spec, eff) in enumerate(zip(specs, effects)):
            if j >= X.shape[1]:
                break
            ids = _hrc_bin_ids(X[:, j], spec)
            score += np.asarray(eff, dtype=np.float64)[ids]
        for term, eff in zip(state.get("pair_terms", []), state.get("pair_effects", [])):
            if len(term) < 3:
                continue
            a, b, nb = int(term[0]), int(term[1]), int(term[2])
            if a >= X.shape[1] or b >= X.shape[1] or a >= len(specs) or b >= len(specs):
                continue
            ia = _hrc_bin_ids(X[:, a], specs[a])
            ib = _hrc_bin_ids(X[:, b], specs[b])
            pid = ia * max(int(nb), 1) + ib
            table = np.asarray(eff, dtype=np.float64)
            if table.ndim == 1 and table.shape[0] > int(np.max(pid, initial=0)):
                score += table[pid]
        scale = float(state.get("audit_scale", 1.0))
        if scale < 0.999999:
            score = intercept + np.clip(scale, 0.0, 1.0) * (score - intercept)
        if task == "binary":
            score = np.clip(score, -8.0, 8.0)
        score = np.where(np.isfinite(score), score, 0.0)
        return np.ascontiguousarray(score, dtype=np.float64)
    numeric = np.asarray(state["numeric"], dtype=int)
    if numeric.size:
        Z = (X[:, numeric] - np.asarray(state["mean"], dtype=np.float64)) / np.asarray(
            state["std"], dtype=np.float64
        )
        Z = np.where(np.isfinite(Z), Z, 0.0)
    else:
        Z = np.empty((X.shape[0], 0), dtype=np.float64)
    A = np.column_stack([np.ones(X.shape[0], dtype=np.float64), Z])
    coef = np.asarray(state["coef"], dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        score = A @ coef
    task = str(state.get("task"))
    if task == "multiclass":
        scale = float(state.get("audit_scale", 1.0))
        if scale < 0.999999 and getattr(coef, "ndim", 1) == 2 and coef.shape[0] > 0:
            base = np.tile(coef[0, :], (X.shape[0], 1))
            score = base + np.clip(scale, 0.0, 1.0) * (score - base)
        score = score - np.mean(score, axis=1, keepdims=True)
        score = np.clip(score, -8.0, 8.0)
        score = np.where(np.isfinite(score), score, 0.0)
        return np.ascontiguousarray(score.reshape(-1), dtype=np.float64)
    scale = float(state.get("audit_scale", 1.0))
    if scale < 0.999999 and coef.size:
        intercept = float(coef[0])
        score = intercept + np.clip(scale, 0.0, 1.0) * (score - intercept)
    if task == "binary":
        score = np.clip(score, -8.0, 8.0)
    score = np.where(np.isfinite(score), score, 0.0)
    return np.ascontiguousarray(score, dtype=np.float64)


def _linear_init_score_for_fit(state, X):
    if state:
        try:
            n_rows = int(state.get("_audit_oof_n_rows", -1))
            n_cols = int(state.get("_audit_oof_n_cols", -1))
            oof = state.get("_audit_oof_score", None)
            X_arr = np.asarray(X, dtype=np.float64)
            if (
                oof is not None
                and X_arr.ndim == 2
                and X_arr.shape[0] == n_rows
                and X_arr.shape[1] == n_cols
            ):
                oof_arr = np.asarray(oof, dtype=np.float64)
                if oof_arr.size in {X_arr.shape[0], X_arr.shape[0] * int(state.get("n_classes", 0))}:
                    return np.ascontiguousarray(oof_arr, dtype=np.float64)
        except Exception:
            pass
    return _linear_init_score(state, X)


def _linear_init_mode_for_estimator(estimator, task=None):
    mode = getattr(estimator, "_extra_params", {}).get("linear_init", "auto")
    if isinstance(mode, (bool, np.bool_)):
        return "auto_prior" if bool(mode) else "off"
    key = str(mode or "off").strip().lower()
    if key == "auto" and task != "multiclass" and (
        (bool(getattr(estimator, "apx", False)) and task != "binary")
        or bool(getattr(estimator, "apx_compile", False))
        or (bool(getattr(estimator, "apx_optimize", False)) and task != "binary")
        or bool(getattr(estimator, "mvpe", False))
    ):
        return "off"
    return mode


def _temperature_scale_enabled_for_estimator(estimator):
    mode = getattr(estimator, "_extra_params", {}).get("temperature_scale", "auto")
    key = str(mode or "off").strip().lower()
    if key in {"0", "false", "off", "none", "disabled"}:
        return False
    if key == "auto":
        return bool(
            getattr(estimator, "class_weight_info_", {}).get("enabled", False)
        )
    return True


def _eval_metric_name(task):
    if task in {"binary", "multiclass", "rank"}:
        return "logloss"
    if task == "regression":
        return "mse"
    return "loss"


def _set_eval_attributes(estimator, model, task):
    losses = []
    try:
        losses = list(model.val_loss_history())
    except Exception:
        losses = []

    estimator.evals_result_ = {}
    estimator.best_iteration_ = None
    estimator.best_score_ = None
    if not losses:
        return

    metric = _eval_metric_name(task)
    estimator.evals_result_ = {"validation_0": {metric: [float(v) for v in losses]}}
    estimator.best_score_ = float(min(losses))
    try:
        estimator.best_iteration_ = int(model.best_tree_count())
    except Exception:
        estimator.best_iteration_ = int(np.argmin(losses) + 1)


def _tree_weight_vector(model):
    n_total = len(model.tree_info())
    if n_total <= 0:
        return np.array([], dtype=np.float64)
    try:
        weights = np.asarray(model.tree_weights(), dtype=np.float64)
    except Exception:
        weights = np.array([], dtype=np.float64)
    if weights.size != n_total:
        weights = np.ones(n_total, dtype=np.float64)
    return weights


def _maybe_prune_validation_plateau(estimator, model, task, n_classes):
    """Prefer the earliest point inside a noisy validation plateau.

    Native early stopping truncates to the exact minimum of one validation
    curve. On small validation folds that minimum is a high-variance point.
    This keeps the same tree structure but zero-weights the tail if an earlier
    checkpoint is within a robust local-loss tolerance of the minimum.
    """
    enabled = getattr(estimator, "_extra_params", {}).get("validation_plateau_prune", False)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() not in {"0", "false", "off", "none"}
    if not bool(enabled):
        estimator.plateau_prune_info_ = {"enabled": False, "reason": "disabled"}
        return
    if task == "binary" and getattr(estimator, "binary_auc_path_info_", {}).get("accepted"):
        estimator.plateau_prune_info_ = {
            "enabled": False,
            "reason": "binary_auc_path_selected",
        }
        return
    try:
        losses = np.asarray(model.val_loss_history(), dtype=np.float64)
        n_total = len(model.tree_info())
        best_count = int(model.best_tree_count())
    except Exception as exc:
        estimator.plateau_prune_info_ = {"enabled": False, "reason": f"failed: {exc}"}
        return
    if losses.size < 8 or n_total < 20 or best_count <= 0:
        estimator.plateau_prune_info_ = {"enabled": False, "reason": "too_short"}
        return
    finite = np.isfinite(losses)
    if not finite.all():
        estimator.plateau_prune_info_ = {"enabled": False, "reason": "non_finite_loss"}
        return

    best_idx = int(np.argmin(losses))
    if best_idx < 3:
        estimator.plateau_prune_info_ = {"enabled": False, "reason": "early_best"}
        return

    unit = int(round(best_count / max(best_idx + 1, 1)))
    if unit <= 0:
        estimator.plateau_prune_info_ = {"enabled": False, "reason": "bad_tree_unit"}
        return
    if task == "multiclass" and int(n_classes) > 1:
        min_unit = int(n_classes)
        if unit < min_unit or unit % min_unit != 0:
            unit = min_unit
    if unit > n_total:
        estimator.plateau_prune_info_ = {"enabled": False, "reason": "unit_too_large"}
        return

    # Estimate local validation noise from the neighborhood of the selected
    # minimum, then apply a small relative floor so flat curves still prune.
    radius = max(4, min(20, int(round(0.10 * losses.size))))
    lo = max(0, best_idx - radius)
    hi = min(losses.size, best_idx + radius + 1)
    diffs = np.abs(np.diff(losses[lo:hi]))
    noise = float(np.median(diffs)) if diffs.size else 0.0
    best_loss = float(losses[best_idx])
    scale = max(abs(best_loss), float(np.nanmedian(np.abs(losses))), 1e-12)
    tol = max(0.75 * noise, 7.5e-4 * scale)

    # Do not jump far back into the underfit part of the path; this is plateau
    # pruning, not a second HPO search over tree counts.
    earliest = max(0, int(np.floor(0.60 * best_idx)))
    candidates = np.flatnonzero(losses[: best_idx + 1] <= best_loss + tol)
    candidates = candidates[candidates >= earliest]
    if candidates.size == 0:
        estimator.plateau_prune_info_ = {
            "enabled": False,
            "reason": "no_plateau_candidate",
            "best_loss": best_loss,
            "tolerance": float(tol),
        }
        return
    chosen_idx = int(candidates[0])
    chosen_count = min(n_total, max(unit, (chosen_idx + 1) * unit))
    if task == "multiclass" and int(n_classes) > 1:
        block = int(n_classes)
        chosen_count = max(block, (chosen_count // block) * block)
    if chosen_count >= n_total:
        estimator.plateau_prune_info_ = {
            "enabled": True,
            "accepted": False,
            "reason": "already_at_best",
            "best_index": best_idx,
            "best_trees": n_total,
            "tolerance": float(tol),
        }
        return
    rel_extra = float((losses[chosen_idx] - best_loss) / scale)
    weights = _tree_weight_vector(model)
    if weights.size != n_total:
        estimator.plateau_prune_info_ = {"enabled": False, "reason": "weight_error"}
        return
    weights[chosen_count:] = 0.0
    try:
        model.set_tree_weights(weights.astype(float).tolist())
    except Exception as exc:
        estimator.plateau_prune_info_ = {"enabled": False, "reason": f"set_weights_failed: {exc}"}
        return
    estimator.plateau_prune_info_ = {
        "enabled": True,
        "accepted": True,
        "best_index": best_idx,
        "chosen_index": chosen_idx,
        "best_trees": int(n_total),
        "chosen_trees": int(chosen_count),
        "best_loss": best_loss,
        "chosen_loss": float(losses[chosen_idx]),
        "relative_extra_loss": rel_extra,
        "tolerance": float(tol),
    }


def _maybe_apply_trajectory_average(estimator, model, task, n_classes):
    """Average the boosting model over a window of iterates near the validation
    minimum, realized as fixed per-tree weights (still one additive model).

    Native early stopping returns the single round that minimizes one noisy
    validation curve. On small folds that argmin is a high-variance point, so
    the chosen tree count transfers poorly to held-out data -- the dominant
    failure mode on these benchmarks. Averaging the additive model over a
    tolerance-gated window around that minimum is a plain variance-reduction
    step (tail / Polyak averaging of the optimization path):

        F_avg(x) = mean_{r in W} F_r(x)
                 = base + lr * sum_t w_t * c_t(x),
        w_t = |{r in W : r >= block(t)}| / |W|

    where F_r is the model after round r and block(t) is the boosting round that
    added tree t. Early trees keep weight 1; the tail ramps down linearly and is
    truncated past the window. No probabilities are re-estimated, no second
    model is fit, and the result is a single weighted tree ensemble that HPO can
    dial in with one knob (``trajectory_avg`` in [0, 1], 0 = native argmin).
    """
    ep = getattr(estimator, "_extra_params", {})
    raw = ep.get("trajectory_avg", 0.0)
    try:
        frac = float(raw)
    except (TypeError, ValueError):
        frac = 0.0
    estimator.trajectory_avg_info_ = {"enabled": False, "reason": "disabled"}
    if not (frac > 0.0):
        return
    # Do not fight an accepted AUC-path tree selection on binary tasks.
    if task == "binary" and getattr(estimator, "binary_auc_path_info_", {}).get("accepted"):
        estimator.trajectory_avg_info_ = {"enabled": False, "reason": "binary_auc_path_selected"}
        return
    try:
        losses = np.asarray(model.val_loss_history(), dtype=np.float64)
        n_total = len(model.tree_info())
        best_count = int(model.best_tree_count())
    except Exception as exc:
        estimator.trajectory_avg_info_ = {"enabled": False, "reason": f"failed: {exc}"}
        return
    R = int(losses.size)
    if R < 6 or n_total < 12 or best_count <= 0:
        estimator.trajectory_avg_info_ = {"enabled": False, "reason": "too_short"}
        return
    if not np.isfinite(losses).all():
        estimator.trajectory_avg_info_ = {"enabled": False, "reason": "non_finite_loss"}
        return

    best_idx = int(np.argmin(losses))
    # Trees added per boosting round (block size). The empirical ratio adapts to
    # both standard multiclass (one tree per class per round) and multi-output
    # trees (one tree per round), so no task-specific override is needed.
    unit = int(round(best_count / max(best_idx + 1, 1)))
    if unit <= 0:
        unit = max(1, int(round(n_total / max(R, 1))))
    n_blocks = int(np.ceil(n_total / unit))  # round-blocks actually materialized

    # Window half-width in rounds, scaled by the knob.
    half = max(1, min(int(round(frac * R)), R))
    lo = max(0, best_idx - half)
    hi = min(R - 1, best_idx + half)

    # Tolerance gate from local validation noise around the minimum: never blend
    # in clearly underfit/overfit rounds, only the flat region.
    radius = max(4, min(20, int(round(0.10 * R))))
    nb_lo = max(0, best_idx - radius)
    nb_hi = min(R, best_idx + radius + 1)
    diffs = np.abs(np.diff(losses[nb_lo:nb_hi]))
    noise = float(np.median(diffs)) if diffs.size else 0.0
    best_loss = float(losses[best_idx])
    scale = max(abs(best_loss), float(np.nanmedian(np.abs(losses))), 1e-12)
    tol = max(noise, 1.0e-3 * scale)

    window = [
        r for r in range(lo, hi + 1)
        if r < n_blocks and losses[r] <= best_loss + tol
    ]
    if best_idx < n_blocks and best_idx not in window:
        window.append(best_idx)
    window = sorted(set(window))

    # Correctness guard: when all trees are retained we must always emit an
    # explicit weight vector, otherwise the kept tail would be used at full
    # weight. A degenerate window collapses to the native argmin truncation.
    if len(window) <= 1:
        weights = np.ones(n_total, dtype=np.float64)
        weights[best_count:] = 0.0
        try:
            model.set_tree_weights(weights.tolist())
        except Exception:
            pass
        estimator.trajectory_avg_info_ = {
            "enabled": True, "accepted": False, "reason": "degenerate_window",
            "best_index": best_idx, "best_trees": int(best_count),
        }
        return

    wnd = np.asarray(window, dtype=np.int64)
    blocks = (np.arange(n_total, dtype=np.int64) // unit)
    weights = (wnd[:, None] >= blocks[None, :]).mean(axis=0).astype(np.float64)
    try:
        model.set_tree_weights(weights.tolist())
    except Exception as exc:
        estimator.trajectory_avg_info_ = {"enabled": False, "reason": f"set_weights_failed: {exc}"}
        return
    estimator.trajectory_avg_info_ = {
        "enabled": True, "accepted": True,
        "best_index": int(best_idx), "window": [int(window[0]), int(window[-1])],
        "n_window": int(wnd.size), "unit": int(unit),
        "best_trees": int(best_count), "total_trees": int(n_total),
        "effective_trees": float(weights.sum()),
        "tolerance": float(tol),
    }


def _residual_focus_auto_enabled(estimator, task):
    ep = getattr(estimator, "_extra_params", {})
    raw = ep.get("residual_focus_auto", False)
    if isinstance(raw, str):
        key = raw.strip().lower()
        enabled = key not in {"0", "false", "off", "none", "disabled"}
    else:
        enabled = bool(raw)
    if not enabled:
        return False
    if task not in {"binary", "regression"}:
        return False
    # The repeated-split audit found the transferable slice in leafwise growth.
    # Other growth policies had negative tails, so the controller is explicit
    # about the structural regime rather than letting Optuna learn a brittle mix.
    return str(getattr(estimator, "grow_policy", "")).strip().lower() == "leafwise"


def _residual_focus_eval_loss(estimator, model, X_val, y_val, task, n_classes, init_score=None):
    raw = np.asarray(model.predict(X_val, init_score=init_score), dtype=np.float64)
    metric = str(getattr(estimator, "_extra_params", {}).get("eval_metric", "") or "").strip().lower()
    if task == "binary" and metric in {"auc", "roc_auc", "1-auc"}:
        return _binary_auc_error(y_val, raw.reshape(-1))
    return _apx_raw_loss(y_val, raw, task, n_classes)


def _growth_policy_race_enabled(estimator):
    raw = getattr(estimator, "_extra_params", {}).get("growth_policy_race", False)
    if isinstance(raw, str):
        return raw.strip().lower() not in {"0", "false", "off", "none", "disabled"}
    return bool(raw)


def _maybe_select_growth_policy_challenger(
    estimator,
    model,
    build_model,
    X_fit,
    y_fit,
    eval_X,
    eval_y,
    task,
    n_classes,
    init_fit=None,
    init_eval=None,
    sample_weight=None,
):
    """Train-only race among standard tree growth policies.

    This is structure selection, not an ensemble: the estimator commits exactly
    one fitted model. It is intentionally generic and uses only the same
    validation fold that early stopping already uses.
    """
    estimator.growth_policy_race_info_ = {"enabled": False, "reason": "disabled"}
    if not _growth_policy_race_enabled(estimator):
        return model
    if eval_X is None or eval_y is None:
        estimator.growth_policy_race_info_ = {"enabled": True, "accepted": False, "reason": "no_eval_set"}
        return model
    try:
        base_loss = _residual_focus_eval_loss(estimator, model, eval_X, eval_y, task, n_classes, init_eval)
    except Exception as exc:
        estimator.growth_policy_race_info_ = {
            "enabled": True,
            "accepted": False,
            "reason": f"base_eval_failed:{type(exc).__name__}",
        }
        return model
    if not np.isfinite(base_loss):
        estimator.growth_policy_race_info_ = {"enabled": True, "accepted": False, "reason": "base_non_finite"}
        return model
    try:
        margin = float(getattr(estimator, "_extra_params", {}).get("growth_policy_race_margin", 0.002) or 0.0)
    except (TypeError, ValueError):
        margin = 0.002
    margin = max(0.0, margin)
    current = str(getattr(estimator, "grow_policy", "depthwise") or "depthwise").strip().lower()
    # Keep this race narrow. Broad growth-policy races repeatedly overfit small
    # validation folds; the only shipped challenger is the fixed leafwise
    # sibling discovered by the tree-architecture lab.
    if current != "leafwise":
        estimator.growth_policy_race_info_ = {
            "enabled": True,
            "accepted": False,
            "reason": "not_leafwise",
            "base_policy": current,
        }
        return model
    policies = ["leafwise", "trunk1_balanced"]
    ordered = [current] + [p for p in policies if p != current]
    best_model = model
    best_policy = current
    best_loss = float(base_loss)
    tried = [{"policy": current, "loss": float(base_loss), "incumbent": True}]
    for policy in ordered[1:]:
        try:
            challenger = build_model({"grow_policy": policy})
            challenger.fit(
                X_fit,
                y_fit,
                int(getattr(estimator, "n_estimators", 0) or 0),
                eval_x=eval_X,
                eval_y=eval_y,
                init_score=init_fit,
                eval_init_score=init_eval,
                sample_weight=sample_weight,
            )
            loss = _residual_focus_eval_loss(
                estimator, challenger, eval_X, eval_y, task, n_classes, init_eval
            )
        except Exception as exc:
            tried.append({"policy": policy, "error": type(exc).__name__})
            continue
        tried.append({"policy": policy, "loss": float(loss)})
        if np.isfinite(loss) and float(loss) < best_loss:
            best_model = challenger
            best_policy = policy
            best_loss = float(loss)
    denom = max(abs(float(base_loss)), 1e-12)
    rel = (float(base_loss) - float(best_loss)) / denom
    accepted = best_policy != current and rel > margin
    estimator.growth_policy_race_info_ = {
        "enabled": True,
        "accepted": bool(accepted),
        "base_policy": current,
        "selected_policy": best_policy if accepted else current,
        "base_loss": float(base_loss),
        "best_loss": float(best_loss),
        "relative_improve": float(rel),
        "margin": float(margin),
        "tried": tried,
    }
    if not accepted:
        return model
    estimator.grow_policy = best_policy
    return best_model


def _split_risk_auto_enabled(estimator):
    raw = getattr(estimator, "_extra_params", {}).get("split_risk_auto", False)
    if isinstance(raw, str):
        return raw.strip().lower() not in {"0", "false", "off", "none", "disabled"}
    return bool(raw)


def _maybe_select_split_risk_challenger(
    estimator,
    model,
    build_model,
    X_fit,
    y_fit,
    eval_X,
    eval_y,
    task,
    n_classes,
    init_fit=None,
    init_eval=None,
    sample_weight=None,
):
    """Train-only race between current split-risk scoring and greedy scoring.

    Several split optimism corrections help some tasks and over-prune others.
    This controller keeps the regularized incumbent unless the same fold used for
    early stopping says a greedy split-risk-off challenger transfers better.
    """
    estimator.split_risk_auto_info_ = {"enabled": False, "reason": "disabled"}
    if not _split_risk_auto_enabled(estimator):
        return model
    ep = getattr(estimator, "_extra_params", {})
    active = any(
        float(ep.get(key, 0.0) or 0.0) > 0.0
        for key in ("gain_penalty", "split_pessimism", "split_contrast_penalty", "cat_audit_strength")
    )
    if not active:
        estimator.split_risk_auto_info_ = {"enabled": True, "accepted": False, "reason": "risk_already_off"}
        return model
    if eval_X is None or eval_y is None:
        estimator.split_risk_auto_info_ = {"enabled": True, "accepted": False, "reason": "no_eval_set"}
        return model
    try:
        base_loss = _residual_focus_eval_loss(estimator, model, eval_X, eval_y, task, n_classes, init_eval)
    except Exception as exc:
        estimator.split_risk_auto_info_ = {
            "enabled": True,
            "accepted": False,
            "reason": f"base_eval_failed:{type(exc).__name__}",
        }
        return model
    if not np.isfinite(base_loss):
        estimator.split_risk_auto_info_ = {"enabled": True, "accepted": False, "reason": "base_non_finite"}
        return model
    try:
        margin = float(getattr(estimator, "_extra_params", {}).get("split_risk_auto_margin", 0.002) or 0.0)
    except (TypeError, ValueError):
        margin = 0.002
    margin = max(0.0, margin)
    overrides = {
        "gain_penalty": 0.0,
        "split_pessimism": 0.0,
        "split_contrast_penalty": 0.0,
        "cat_audit_strength": 0.0,
    }
    try:
        challenger = build_model(overrides)
        challenger.fit(
            X_fit,
            y_fit,
            int(getattr(estimator, "n_estimators", 0) or 0),
            eval_x=eval_X,
            eval_y=eval_y,
            init_score=init_fit,
            eval_init_score=init_eval,
            sample_weight=sample_weight,
        )
        challenger_loss = _residual_focus_eval_loss(
            estimator, challenger, eval_X, eval_y, task, n_classes, init_eval
        )
    except Exception as exc:
        estimator.split_risk_auto_info_ = {
            "enabled": True,
            "accepted": False,
            "reason": f"challenger_failed:{type(exc).__name__}",
            "base_loss": float(base_loss),
        }
        return model
    if not np.isfinite(challenger_loss):
        estimator.split_risk_auto_info_ = {
            "enabled": True,
            "accepted": False,
            "reason": "challenger_non_finite",
            "base_loss": float(base_loss),
        }
        return model
    denom = max(abs(float(base_loss)), 1e-12)
    rel = (float(base_loss) - float(challenger_loss)) / denom
    accepted = rel > margin
    estimator.split_risk_auto_info_ = {
        "enabled": True,
        "accepted": bool(accepted),
        "base_loss": float(base_loss),
        "challenger_loss": float(challenger_loss),
        "relative_improve": float(rel),
        "margin": float(margin),
        "overrides": overrides,
    }
    if not accepted:
        return model
    for key, value in overrides.items():
        estimator._extra_params[key] = value
    return challenger


def _bins_race_enabled(estimator):
    raw = getattr(estimator, "_extra_params", {}).get("bins_race", False)
    if isinstance(raw, str):
        return raw.strip().lower() not in {"0", "false", "off", "none", "disabled"}
    return bool(raw)


def _maybe_select_bins_race_challenger(
    estimator,
    model,
    build_model,
    X_fit,
    y_fit,
    eval_X,
    eval_y,
    task,
    n_classes,
    init_fit=None,
    init_eval=None,
    sample_weight=None,
):
    """In-fit representation race: train the same config under the OPPOSITE
    `supervised_bins` setting (plain quantile bins vs DP bins) and keep
    whichever wins on the eval fold. The representation decision is made where
    an eval fold exists — at final-fit time that is the actual holdout — so it
    never has to survive the outer CV-argmin.
    """
    estimator.bins_race_info_ = {"enabled": False, "reason": "disabled"}
    if not _bins_race_enabled(estimator):
        return model
    if eval_X is None or eval_y is None:
        estimator.bins_race_info_ = {"enabled": True, "accepted": False, "reason": "no_eval_set"}
        return model
    incumbent_sb = bool(getattr(estimator, "_extra_params", {}).get("supervised_bins", False))
    overrides = {"supervised_bins": not incumbent_sb}
    try:
        base_loss = _residual_focus_eval_loss(estimator, model, eval_X, eval_y, task, n_classes, init_eval)
    except Exception as exc:
        estimator.bins_race_info_ = {
            "enabled": True,
            "accepted": False,
            "reason": f"base_eval_failed:{type(exc).__name__}",
        }
        return model
    if not np.isfinite(base_loss):
        estimator.bins_race_info_ = {"enabled": True, "accepted": False, "reason": "base_non_finite"}
        return model
    try:
        margin = float(getattr(estimator, "_extra_params", {}).get("bins_race_margin", 0.001) or 0.0)
    except (TypeError, ValueError):
        margin = 0.001
    margin = max(0.0, margin)
    try:
        challenger = build_model(overrides)
        challenger.fit(
            X_fit,
            y_fit,
            int(getattr(estimator, "n_estimators", 0) or 0),
            eval_x=eval_X,
            eval_y=eval_y,
            init_score=init_fit,
            eval_init_score=init_eval,
            sample_weight=sample_weight,
        )
        challenger_loss = _residual_focus_eval_loss(
            estimator, challenger, eval_X, eval_y, task, n_classes, init_eval
        )
    except Exception as exc:
        estimator.bins_race_info_ = {
            "enabled": True,
            "accepted": False,
            "reason": f"challenger_failed:{type(exc).__name__}",
            "base_loss": float(base_loss),
        }
        return model
    if not np.isfinite(challenger_loss):
        estimator.bins_race_info_ = {
            "enabled": True,
            "accepted": False,
            "reason": "challenger_non_finite",
            "base_loss": float(base_loss),
        }
        return model
    denom = max(abs(float(base_loss)), 1e-12)
    rel = (float(base_loss) - float(challenger_loss)) / denom
    accepted = rel > margin
    estimator.bins_race_info_ = {
        "enabled": True,
        "accepted": bool(accepted),
        "base_loss": float(base_loss),
        "challenger_loss": float(challenger_loss),
        "relative_improve": float(rel),
        "margin": float(margin),
        "challenger_supervised_bins": not incumbent_sb,
    }
    if not accepted:
        return model
    for key, value in overrides.items():
        estimator._extra_params[key] = value
    return challenger


def _binary_shape_auto_enabled(estimator, task):
    if task != "binary":
        return False
    ep = getattr(estimator, "_extra_params", {})
    raw = ep.get("binary_shape_auto", False)
    if isinstance(raw, str):
        key = raw.strip().lower()
        if key in {"0", "false", "off", "none", "disabled"}:
            return False
    elif raw is not None:
        return bool(raw)
    metric = str(ep.get("eval_metric", "") or "").strip().lower()
    if metric not in {"auc", "roc_auc", "1-auc"}:
        return False
    try:
        rank_mix = float(ep.get("rank_mix_alpha", 0.0) or 0.0)
    except (TypeError, ValueError):
        rank_mix = 0.0
    try:
        focus = float(ep.get("binary_focus_gamma", 0.0) or 0.0)
    except (TypeError, ValueError):
        focus = 0.0
    return rank_mix > 1e-12 or focus > 1e-12


def _binary_auc_model_loss(model, X_val, y_val, init_score=None):
    raw = np.asarray(model.predict(X_val, init_score=init_score), dtype=np.float64)
    return _binary_auc_error(y_val, raw.reshape(-1))


def _maybe_select_binary_shape_challenger(
    estimator,
    model,
    build_model,
    X_fit,
    y_fit,
    eval_X,
    eval_y,
    task,
    n_classes,
    init_fit=None,
    init_eval=None,
    sample_weight=None,
):
    """Race AUC-shaped binary training against a plain Newton/logloss path.

    Rank-mix and hard-example binary shaping can find useful margins, but on
    small categorical binary data they also make Optuna select brittle training
    paths.  This is candidate-level model selection using the existing eval
    fold: keep the shaped incumbent unless a plain logloss/no-rank/no-focus
    challenger has better validation AUC.  The committed estimator is still one
    GTBoost model.
    """
    estimator.binary_shape_auto_info_ = {"enabled": False, "reason": "disabled"}
    if not _binary_shape_auto_enabled(estimator, task):
        return model
    if eval_X is None or eval_y is None:
        estimator.binary_shape_auto_info_ = {
            "enabled": True,
            "accepted": False,
            "reason": "no_eval_set",
        }
        return model
    try:
        base_loss = _binary_auc_model_loss(model, eval_X, eval_y, init_eval)
    except Exception as exc:
        estimator.binary_shape_auto_info_ = {
            "enabled": True,
            "accepted": False,
            "reason": f"base_eval_failed:{type(exc).__name__}",
        }
        return model
    if not np.isfinite(base_loss):
        estimator.binary_shape_auto_info_ = {
            "enabled": True,
            "accepted": False,
            "reason": "base_non_finite",
        }
        return model
    try:
        margin = float(getattr(estimator, "_extra_params", {}).get("binary_shape_auto_margin", 0.0) or 0.0)
    except (TypeError, ValueError):
        margin = 0.0
    margin = max(0.0, margin)
    overrides = {
        "eval_metric": "logloss",
        "rank_mix_alpha": 0.0,
        "rank_mix_start_frac": 0.0,
        "binary_focus_gamma": 0.0,
    }
    commit_overrides = {**overrides, "binary_auc_path_select": False}
    try:
        challenger = build_model(overrides)
        challenger.fit(
            X_fit,
            y_fit,
            int(getattr(estimator, "n_estimators", 0) or 0),
            eval_x=eval_X,
            eval_y=eval_y,
            init_score=init_fit,
            eval_init_score=init_eval,
            sample_weight=sample_weight,
        )
        challenger_loss = _binary_auc_model_loss(challenger, eval_X, eval_y, init_eval)
    except Exception as exc:
        estimator.binary_shape_auto_info_ = {
            "enabled": True,
            "accepted": False,
            "reason": f"challenger_failed:{type(exc).__name__}:{exc}",
            "base_loss": float(base_loss),
        }
        return model
    if not np.isfinite(challenger_loss):
        estimator.binary_shape_auto_info_ = {
            "enabled": True,
            "accepted": False,
            "reason": "challenger_non_finite",
            "base_loss": float(base_loss),
        }
        return model
    denom = max(abs(float(base_loss)), 1e-12)
    rel = (float(base_loss) - float(challenger_loss)) / denom
    accepted = rel > margin
    estimator.binary_shape_auto_info_ = {
        "enabled": True,
        "accepted": bool(accepted),
        "base_loss": float(base_loss),
        "challenger_loss": float(challenger_loss),
        "relative_improve": float(rel),
        "margin": float(margin),
        "overrides": commit_overrides,
    }
    if not accepted:
        return model
    for key, value in commit_overrides.items():
        estimator._extra_params[key] = value
    return challenger


def _maybe_select_residual_focus_challenger(
    estimator,
    model,
    build_model,
    X_fit,
    y_fit,
    eval_X,
    eval_y,
    task,
    n_classes,
    init_fit=None,
    init_eval=None,
    sample_weight=None,
):
    """Train-only model selection between ordinary boosting and residual focus.

    Residual focus has real signal but a negative tail when exposed as a broad
    hyperparameter.  This controller makes the switch local to the training
    validation split: train the ordinary path and one bounded hard-residual
    challenger, then keep the challenger only if its eval loss improves by a
    small margin.  The committed object is still a single GTBoost model.
    """
    estimator.residual_focus_auto_info_ = {"enabled": False, "reason": "disabled"}
    if not _residual_focus_auto_enabled(estimator, task):
        return model
    if eval_X is None or eval_y is None:
        estimator.residual_focus_auto_info_ = {"enabled": True, "accepted": False, "reason": "no_eval_set"}
        return model
    try:
        incumbent_alpha = float(getattr(estimator, "_extra_params", {}).get("residual_focus_alpha", 0.0) or 0.0)
    except (TypeError, ValueError):
        incumbent_alpha = 0.0
    if incumbent_alpha > 1e-12:
        estimator.residual_focus_auto_info_ = {
            "enabled": True,
            "accepted": False,
            "reason": "incumbent_already_focused",
            "incumbent_alpha": incumbent_alpha,
        }
        return model
    try:
        alpha = float(getattr(estimator, "_extra_params", {}).get("residual_focus_auto_alpha", 0.5) or 0.5)
    except (TypeError, ValueError):
        alpha = 0.5
    alpha = float(np.clip(alpha, 0.05, 1.0))
    try:
        margin = float(getattr(estimator, "_extra_params", {}).get("residual_focus_auto_margin", 0.002) or 0.0)
    except (TypeError, ValueError):
        margin = 0.002
    margin = max(0.0, margin)
    try:
        base_loss = _residual_focus_eval_loss(estimator, model, eval_X, eval_y, task, n_classes, init_eval)
    except Exception as exc:
        estimator.residual_focus_auto_info_ = {
            "enabled": True,
            "accepted": False,
            "reason": f"base_eval_failed:{type(exc).__name__}",
        }
        return model
    if not np.isfinite(base_loss):
        estimator.residual_focus_auto_info_ = {"enabled": True, "accepted": False, "reason": "base_non_finite"}
        return model
    try:
        challenger = build_model(
            {
                "residual_focus_alpha": alpha,
                "residual_focus_max_scale": float(
                    getattr(estimator, "_extra_params", {}).get("residual_focus_max_scale", 2.0) or 2.0
                ),
                "residual_focus_mode": "full",
                "residual_focus_hessian_mode": "equal",
                "residual_focus_redescend_tau": 0.0,
            }
        )
        challenger.fit(
            X_fit,
            y_fit,
            int(getattr(estimator, "n_estimators", 0) or 0),
            eval_x=eval_X,
            eval_y=eval_y,
            init_score=init_fit,
            eval_init_score=init_eval,
            sample_weight=sample_weight,
        )
        challenger_loss = _residual_focus_eval_loss(
            estimator, challenger, eval_X, eval_y, task, n_classes, init_eval
        )
    except Exception as exc:
        estimator.residual_focus_auto_info_ = {
            "enabled": True,
            "accepted": False,
            "reason": f"challenger_failed:{type(exc).__name__}",
            "base_loss": float(base_loss),
        }
        return model
    if not np.isfinite(challenger_loss):
        estimator.residual_focus_auto_info_ = {
            "enabled": True,
            "accepted": False,
            "reason": "challenger_non_finite",
            "base_loss": float(base_loss),
        }
        return model
    denom = max(abs(float(base_loss)), 1e-12)
    rel = (float(base_loss) - float(challenger_loss)) / denom
    if rel <= margin:
        estimator.residual_focus_auto_info_ = {
            "enabled": True,
            "accepted": False,
            "reason": "below_margin",
            "alpha": alpha,
            "base_loss": float(base_loss),
            "challenger_loss": float(challenger_loss),
            "relative_improve": float(rel),
            "margin": float(margin),
        }
        return model
    estimator.residual_focus_auto_info_ = {
        "enabled": True,
        "accepted": True,
        "alpha": alpha,
        "base_loss": float(base_loss),
        "challenger_loss": float(challenger_loss),
        "relative_improve": float(rel),
        "margin": float(margin),
    }
    # Persist the winning overrides like every other race: the deploy-time full
    # refit rebuilds from _extra_params, and without this the refit silently
    # trains WITHOUT residual focus even though the focused model won.
    estimator._extra_params["residual_focus_alpha"] = alpha
    estimator._extra_params.setdefault("residual_focus_max_scale", 2.0)
    estimator._extra_params["residual_focus_mode"] = "full"
    estimator._extra_params["residual_focus_hessian_mode"] = "equal"
    estimator._extra_params["residual_focus_redescend_tau"] = 0.0
    return challenger


def _maybe_calibrate_tree_scale(estimator, model, X_val, y_val, task, n_classes, init_score=None):
    """Validation-calibrate one scalar multiplier on the fitted tree block.

    This is a structural shrinkage step, not an ensemble: the prior/base margin
    stays fixed and all tree weights are multiplied by the same scalar. It only
    commits when the existing validation fold says the scaled additive model is
    better than the current one.
    """
    estimator.tree_scale_info_ = {"enabled": False, "reason": "disabled"}
    try:
        n_total = len(model.tree_info())
        old_weights = _tree_weight_vector(model)
    except Exception as exc:
        estimator.tree_scale_info_ = {"enabled": False, "reason": f"failed: {exc}"}
        return
    if n_total < 12 or old_weights.size != n_total:
        estimator.tree_scale_info_ = {"enabled": False, "reason": "too_few_trees"}
        return
    try:
        full_raw = np.asarray(model.predict(X_val, init_score=init_score), dtype=np.float64)
    except Exception as exc:
        estimator.tree_scale_info_ = {"enabled": False, "reason": f"predict_failed: {exc}"}
        return
    if init_score is None:
        base_raw = np.zeros_like(full_raw, dtype=np.float64)
    else:
        base_raw = np.asarray(init_score, dtype=np.float64).reshape(full_raw.shape)
    tree_raw = full_raw - base_raw
    if not (np.all(np.isfinite(full_raw)) and np.all(np.isfinite(tree_raw))):
        estimator.tree_scale_info_ = {"enabled": False, "reason": "non_finite_raw"}
        return
    candidates = np.asarray(
        [0.0, 0.25, 0.50, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0],
        dtype=np.float64,
    )
    metric = str(getattr(estimator, "_extra_params", {}).get("eval_metric", "") or "").strip().lower()
    losses = []
    for scale in candidates:
        raw = base_raw + float(scale) * tree_raw
        if task == "binary" and metric in {"auc", "roc_auc", "1-auc"}:
            losses.append(_binary_auc_error(y_val, raw.reshape(-1)))
        else:
            losses.append(_apx_raw_loss(y_val, raw, task, n_classes))
    losses = np.asarray(losses, dtype=np.float64)
    if not np.isfinite(losses).all():
        estimator.tree_scale_info_ = {"enabled": False, "reason": "non_finite_loss"}
        return
    base_idx = int(np.flatnonzero(np.isclose(candidates, 1.0))[0])
    base_loss = float(losses[base_idx])
    tie_tol = 0.0
    if task == "binary" and metric in {"auc", "roc_auc", "1-auc"}:
        yy = np.asarray(y_val).reshape(-1)
        n_pos = int(np.sum(yy > 0.5))
        n_neg = int(yy.size - n_pos)
        tie_tol = 1.0 / max(1.0, float(n_pos * n_neg))
        plateau = np.flatnonzero(losses <= float(np.min(losses)) + tie_tol)
        best_idx = int(plateau[0]) if plateau.size else int(np.argmin(losses))
    else:
        best_idx = int(np.argmin(losses))
    best_scale = float(candidates[best_idx])
    best_loss = float(losses[best_idx])
    denom = max(abs(base_loss), 1e-12)
    min_rel = 2e-4 if task == "regression" else 5e-4
    auc_tie_shrink = (
        task == "binary"
        and metric in {"auc", "roc_auc", "1-auc"}
        and best_idx != base_idx
        and best_scale < 1.0
        and best_loss <= base_loss + tie_tol
    )
    if best_idx == base_idx or ((base_loss - best_loss) / denom < min_rel and not auc_tie_shrink):
        estimator.tree_scale_info_ = {
            "enabled": True,
            "accepted": False,
            "scale": 1.0,
            "best_scale": best_scale,
            "base_loss": base_loss,
            "best_loss": best_loss,
            "relative_improve": float((base_loss - best_loss) / denom),
        }
        return
    try:
        model.set_tree_weights((old_weights * best_scale).astype(float).tolist())
    except Exception as exc:
        estimator.tree_scale_info_ = {"enabled": False, "reason": f"set_weights_failed: {exc}"}
        return
    estimator.tree_scale_info_ = {
        "enabled": True,
        "accepted": True,
        "scale": best_scale,
        "base_loss": base_loss,
        "best_loss": best_loss,
        "relative_improve": float((base_loss - best_loss) / denom),
    }


def _region_gate_active(extra_params):
    try:
        return float(dict(extra_params or {}).get("region_gate", 0.0)) > 0.0
    except (TypeError, ValueError):
        return False


def _region_gate_loss_matrix(model, X_val, y_val, task, n_classes, counts, init_score):
    """Per-row validation loss of the model truncated at each candidate tree count.
    Returns L with shape (len(counts), n_val): L[k, i] = loss of row i at counts[k]."""
    n = int(X_val.shape[0])
    L = np.empty((len(counts), n), dtype=np.float64)
    for k, c in enumerate(counts):
        raw = np.asarray(model.predict_truncated(X_val, int(c), init_score=init_score))
        if task == "regression":
            L[k] = (raw.reshape(-1) - y_val) ** 2
        elif task == "binary":
            z = np.clip(raw.reshape(-1), -35.0, 35.0)
            p = np.clip(1.0 / (1.0 + np.exp(-z)), 1e-15, 1.0 - 1e-15)
            L[k] = -(y_val * np.log(p) + (1.0 - y_val) * np.log1p(-p))
        else:
            logits = raw.reshape(n, n_classes)
            logits = logits - logits.max(axis=1, keepdims=True)
            ex = np.exp(logits)
            sm = np.clip(ex / ex.sum(axis=1, keepdims=True), 1e-15, 1.0)
            L[k] = -np.log(sm[np.arange(n), y_val.astype(int)])
    return L


def _fit_region_gate(estimator, model, X_val, y_val, task, n_classes, init_score=None):
    """Input-conditional depth: a single-model MoE where the gate routes each input
    to an expert = the boosted model truncated at a region-specific tree count.

    A depth-1 gate (one feature, one threshold) partitions input space into two
    regions; each region keeps the tree count that minimizes its own held-out loss.
    Easy regions stop early (less overfit), hard regions use more trees -- one
    additive model, no separate sub-models, no ensembling. The gate is accepted
    ONLY if it beats the single global tree count on the eval set by a margin AND
    both regions genuinely prefer their own depth, so it degenerates to a standard
    GBDT when there is no heterogeneity (universally safe by construction)."""
    estimator._region_gate = None
    try:
        margin = float(estimator._extra_params.get("region_gate", 0.0) or 0.0)
    except (TypeError, ValueError):
        margin = 0.0
    if margin <= 0.0:
        estimator.region_gate_info_ = {"enabled": False, "reason": "disabled"}
        return
    try:
        n_total = len(model.tree_info())
        best = int(model.best_tree_count())
    except Exception as exc:
        estimator.region_gate_info_ = {"enabled": True, "accepted": False, "reason": f"failed: {exc}"}
        return
    # keep_all_trees retains the full (untruncated) path for the gate. If we end up
    # NOT installing a 2-region gate we must still truncate to the early-stopping
    # count, so default to a global gate at best_count (= the standard model).
    if best > 0:
        estimator._region_gate = {"feature": 0, "threshold": float("inf"),
                                  "count_left": int(best), "count_right": int(best)}
    Xv = np.asarray(X_val, dtype=np.float64)
    yv = np.asarray(y_val)
    n = int(Xv.shape[0])
    if n_total < 20 or n < 40 or best <= 0:
        estimator.region_gate_info_ = {"enabled": True, "accepted": False, "reason": "too_small"}
        return
    lo = max(1, int(0.4 * best))
    counts = sorted({int(round(c)) for c in np.linspace(lo, n_total, 8) if 1 <= c <= n_total})
    if len(counts) < 2:
        estimator.region_gate_info_ = {"enabled": True, "accepted": False, "reason": "no_counts"}
        return
    L = _region_gate_loss_matrix(model, Xv, yv, task, n_classes, counts, init_score)
    total = L.sum(axis=1)
    gbest = int(np.argmin(total))
    global_loss = float(total[gbest])
    min_region = max(15, int(0.2 * n))
    required = margin * max(global_loss, 1e-12)
    best_gain = required
    chosen = None
    for f in range(Xv.shape[1]):
        col = Xv[:, f]
        uniq = np.unique(col[np.isfinite(col)])
        if uniq.size < 2:
            continue
        cand = np.unique(np.quantile(uniq, [0.25, 0.5, 0.75])) if uniq.size > 3 else uniq[:-1]
        for thr in cand:
            left = col <= thr
            nl = int(left.sum())
            nr = n - nl
            if nl < min_region or nr < min_region:
                continue
            lL = L[:, left].sum(axis=1)
            rL = L[:, ~left].sum(axis=1)
            cl = int(np.argmin(lL))
            cr = int(np.argmin(rL))
            if cl == cr:
                continue
            # Both regions must strictly prefer their own depth over the global one,
            # so the gain cannot come from a single region exploiting eval noise.
            if not (lL[cl] < lL[gbest] and rL[cr] < rL[gbest]):
                continue
            gain = global_loss - (lL[cl] + rL[cr])
            if gain > best_gain:
                best_gain = gain
                chosen = (f, float(thr), counts[cl], counts[cr], counts[gbest])
    if chosen is None:
        estimator.region_gate_info_ = {
            "enabled": True, "accepted": False, "reason": "no_gate",
            "global_count": counts[gbest],
        }
        return
    f, thr, cL, cR, cG = chosen
    estimator._region_gate = {"feature": int(f), "threshold": float(thr),
                              "count_left": int(cL), "count_right": int(cR)}
    estimator.region_gate_info_ = {
        "enabled": True, "accepted": True, "feature": int(f), "threshold": float(thr),
        "count_left": int(cL), "count_right": int(cR), "global_count": int(cG),
        "rel_gain": float(best_gain / max(global_loss, 1e-12)),
    }


def _region_gate_apply_raw(estimator, model, X_np, task, n_classes):
    """Raw per-region prediction (margin for binary/regression, logits for
    multiclass) using each region's tree count. Shape (n,) or (n, n_classes).
    The linear-init offset is recomputed per region subset so its length always
    matches the rows being predicted."""
    g = estimator._region_gate
    f = g["feature"]
    thr = g["threshold"]
    region_left = X_np[:, f] <= thr
    n = int(X_np.shape[0])
    multiclass = task == "multiclass" and n_classes > 2
    out = np.zeros((n, n_classes) if multiclass else n, dtype=np.float64)
    for mask, c in ((region_left, g["count_left"]), (~region_left, g["count_right"])):
        if not mask.any():
            continue
        Xi = X_np[mask]
        ii = _linear_init_score(estimator._linear_init_state, Xi)
        raw = np.asarray(model.predict_truncated(Xi, int(c), init_score=ii))
        if multiclass:
            out[mask] = raw.reshape(-1, n_classes)
        else:
            out[mask] = raw.reshape(-1)
    return out, multiclass


def _maybe_select_binary_auc_path(estimator, model, X_val, y_val, init_score=None):
    metric = str(getattr(estimator, "_extra_params", {}).get("eval_metric", "")).lower()
    enabled = getattr(estimator, "_extra_params", {}).get("binary_auc_path_select", "auto")
    if isinstance(enabled, str):
        key = enabled.strip().lower()
        if key == "auto":
            enabled = metric in {"auc", "roc_auc", "1-auc"}
        else:
            enabled = key not in {"0", "false", "off", "none", "disabled"}
    estimator.binary_auc_path_info_ = {"enabled": False, "reason": "disabled"}
    if not bool(enabled):
        return
    try:
        losses = np.asarray(model.val_loss_history(), dtype=np.float64)
        n_total = len(model.tree_info())
        best_count = int(model.best_tree_count())
    except Exception as exc:
        estimator.binary_auc_path_info_ = {"enabled": True, "accepted": False, "reason": f"failed: {exc}"}
        return
    if losses.size < 4 or n_total < 10:
        estimator.binary_auc_path_info_ = {"enabled": True, "accepted": False, "reason": "too_short"}
        return

    best_idx = int(np.argmin(losses)) if np.isfinite(losses).all() else 0
    unit = int(round(best_count / max(best_idx + 1, 1))) if best_count > 0 else 1
    if unit <= 0 or unit > n_total:
        unit = max(1, int(round(n_total / max(losses.size, 1))))

    counts = None
    auc_errors = None
    if metric in {"auc", "roc_auc", "1-auc"} and np.isfinite(losses).all():
        max_i = min(losses.size, int(np.ceil(n_total / max(unit, 1))))
        counts = np.array([min(n_total, (i + 1) * unit) for i in range(max_i)], dtype=int)
        auc_errors = losses[:max_i].astype(np.float64)
    else:
        n_checks = int(getattr(estimator, "_extra_params", {}).get("binary_auc_path_checks", 64))
        n_checks = max(12, min(96, n_checks))
        raw_counts = np.linspace(max(unit, int(0.05 * n_total)), n_total, n_checks)
        counts = np.unique(np.maximum(unit, (raw_counts.astype(int) // unit) * unit))
        counts = counts[(counts > 0) & (counts <= n_total)]
        if counts.size == 0 or counts[-1] != n_total:
            counts = np.unique(np.append(counts, n_total))
        errs = []
        for count in counts:
            raw = np.asarray(
                model.predict_truncated(X_val, int(count), init_score=init_score),
                dtype=np.float64,
            )
            errs.append(_binary_auc_error(y_val, raw.reshape(-1)))
        auc_errors = np.asarray(errs, dtype=np.float64)

    finite = np.isfinite(auc_errors)
    if counts is None or auc_errors is None or not finite.any():
        estimator.binary_auc_path_info_ = {"enabled": True, "accepted": False, "reason": "no_finite_auc"}
        return
    counts = counts[finite]
    auc_errors = auc_errors[finite]
    selection_errors = auc_errors
    metric_errors = auc_errors
    best_err = float(np.min(selection_errors))
    # AUC on a validation fold is a step function: one swapped positive/negative
    # pair changes AUC by about 1/(n_pos*n_neg). Exact argmin therefore has high
    # variance. Treat all checkpoints within one AUC step as statistically tied.
    # Use the latest tied point: pruning the middle of a coarse AUC plateau often
    # underfits because many boosting rounds have identical rank ordering on the
    # validation fold but still improve margins and transfer.
    yy = np.asarray(y_val).reshape(-1)
    n_pos = int(np.sum(yy > 0.5))
    n_neg = int(yy.size - n_pos)
    auc_step = 1.0 / max(1.0, float(n_pos * n_neg))
    tie_tol = max(1e-12, auc_step)
    plateau = np.flatnonzero(selection_errors <= best_err + tie_tol)
    chosen_pos = int(plateau[-1])
    chosen_count = int(counts[chosen_pos])
    if chosen_count >= n_total:
        estimator.binary_auc_path_info_ = {
            "enabled": True,
            "accepted": False,
            "reason": "already_full_path",
            "best_auc_error": float(np.min(metric_errors)),
            "chosen_trees": chosen_count,
            "total_trees": n_total,
            "metric": metric or "auto",
        }
        return
    weights = _tree_weight_vector(model)
    if weights.size != n_total:
        estimator.binary_auc_path_info_ = {"enabled": True, "accepted": False, "reason": "weight_error"}
        return
    weights[chosen_count:] = 0.0
    try:
        model.set_tree_weights(weights.astype(float).tolist())
    except Exception as exc:
        estimator.binary_auc_path_info_ = {
            "enabled": True,
            "accepted": False,
            "reason": f"set_weights_failed: {exc}",
        }
        return
    estimator.binary_auc_path_info_ = {
        "enabled": True,
        "accepted": True,
        "metric": metric or "auto",
        "best_auc_error": float(np.min(metric_errors)),
        "chosen_auc_error": float(metric_errors[chosen_pos]),
        "selection_error": float(selection_errors[chosen_pos]),
        "chosen_trees": chosen_count,
        "total_trees": int(n_total),
        "n_candidates": int(len(counts)),
    }


# ── APX: Accumulated Path eXpectation ─────────────────────────────────────────
# Efron-Hastie-Tibshirani 2004: ε-stagewise boosting traces the LASSO
# regularization path. Every point on the path is a valid L1-regularized
# solution with different bias/variance trade-off. Single-point prediction
# (early stopping) picks one noisy point; APX averages nearby path points for
# free variance reduction. See EXPERIMENTS §98.

def _apx_checkpoints(n_total, n_checkpoints, min_frac, max_frac=1.0, spacing="uniform"):
    """Compute integer checkpoint indices in [min_frac*N, max_frac*N]."""
    if n_total < 20:
        return np.array([n_total], dtype=int)
    lo = max(5, int(min_frac * n_total))
    hi = int(max_frac * n_total)
    if spacing == "geometric":
        cp = np.geomspace(max(lo, 1), hi, n_checkpoints).astype(int)
    else:
        cp = np.linspace(lo, hi, n_checkpoints).astype(int)
    return np.unique(cp)


def _apx_weights(k, weighting):
    weighting = str(weighting).lower()
    if weighting in {"flat", "uniform"}:
        return np.ones(k) / k
    if weighting in {"triangle", "linear"}:
        w = np.arange(1, k + 1, dtype=float)
        return w / w.sum()
    if weighting == "gauss":
        sigma = max(1.0, k / 4.0)
        idx = np.arange(k, dtype=float)
        w = np.exp(-((idx - (k - 1)) ** 2) / (2 * sigma ** 2))
        return w / w.sum()
    raise ValueError(f"unknown APX weighting: {weighting}")


def _model_n_trees(model):
    """Cheap tree count: avoids marshaling the full tree_info list per call."""
    try:
        return int(model.n_trees())
    except Exception:
        return len(model.tree_info())


def _apx_predict_raw(model, X_np, n_checkpoints, min_frac, weighting, spacing,
                     task, n_classes, init_score=None):
    """Return weighted path-average of raw predictions.
    task: 'binary', 'multiclass', or 'regression'.
    Returns shape (n,) for binary/regression, (n, K) for multiclass.
    For binary returns raw logits; for multiclass returns (renormalized) probs
    averaged in probability space.
    """
    n_total = _model_n_trees(model)
    if task == "multiclass":
        checkpoints = _apx_compile_checkpoints(
            model,
            task,
            n_classes,
            n_checkpoints,
            min_frac,
            spacing,
        )
    else:
        checkpoints = _apx_checkpoints(n_total, n_checkpoints, min_frac, 1.0, spacing)
    w = _apx_weights(len(checkpoints), weighting)

    if task != "multiclass" and hasattr(model, "predict_checkpoints"):
        # Single Rust pass: bins X once and walks each row's trees once,
        # snapshotting at every checkpoint (vs N full predict_truncated calls).
        flat = np.asarray(
            model.predict_checkpoints(X_np, [int(c) for c in checkpoints], init_score=init_score)
        ).reshape(len(checkpoints), -1)
        if task == "binary":
            flat = _sigmoid_raw(flat)
        return np.asarray(w, dtype=np.float64) @ flat

    acc = None
    for wi, cp in zip(w, checkpoints):
        raw = np.asarray(model.predict_truncated(X_np, int(cp), init_score=init_score))
        if task == "multiclass":
            raw = raw.reshape(-1, n_classes)
            raw = raw - raw.max(axis=1, keepdims=True)
            exp = np.exp(raw)
            raw = exp / exp.sum(axis=1, keepdims=True)
        elif task == "binary":
            # Convert to prob for averaging (more principled than averaging logits)
            raw = _sigmoid_raw(raw)
        acc = wi * raw if acc is None else acc + wi * raw
    return acc


def _apx_optimize_weights(model, X_val, y_val, n_checkpoints, min_frac, spacing,
                          task, n_classes, l2=1e-3, init_score=None):
    """Fit non-negative, sum-to-1 weights on validation predictions at checkpoints.
    Minimizes: L(y_val, Σ w_k · p_k(X_val)) via projected gradient with L2 anchor
    toward uniform weights.

    For binary / multiclass: minimize log-loss.
    For regression: minimize MSE (closed-form NNLS + simplex projection).
    Returns (checkpoints, weights).
    """
    n_total = len(model.tree_info())
    checkpoints = _apx_checkpoints(n_total, n_checkpoints, min_frac, 1.0, spacing)
    k = len(checkpoints)
    if k <= 1 or len(X_val) < 10:
        return checkpoints, np.ones(k) / max(k, 1)

    # Compute predictions at each checkpoint on validation
    preds_by_cp = []
    for cp in checkpoints:
        raw = np.asarray(model.predict_truncated(X_val, int(cp), init_score=init_score))
        if task == "multiclass":
            raw = raw.reshape(-1, n_classes)
            raw = raw - raw.max(axis=1, keepdims=True)
            exp = np.exp(raw)
            raw = exp / exp.sum(axis=1, keepdims=True)
        elif task == "binary":
            raw = 1.0 / (1.0 + np.exp(-raw))
        preds_by_cp.append(raw)
    P = np.stack(preds_by_cp)  # (k, n_val, [n_classes])

    # Uniform prior
    w = np.ones(k) / k

    if task == "regression":
        # Min ||P.T @ w - y||^2 + l2 * ||w - 1/k||^2  s.t. w >= 0, Σw = 1
        # Use projected gradient descent
        y = y_val.astype(np.float64)
        # Closed-form unconstrained: w = (P P^T + l2 I)^{-1} (P y + l2 / k · 1)
        # P is (k, n) so P P^T is (k, k)
        A = P @ P.T + l2 * np.eye(k)
        b = P @ y + (l2 / k) * np.ones(k)
        try:
            w_unconstrained = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return checkpoints, np.ones(k) / k
        # Project onto probability simplex (non-negative, sum to 1)
        w = _project_simplex(w_unconstrained)
    else:
        # Logistic / categorical log-loss. Use gradient descent with simplex projection.
        # Gradient: for binary, dL/dw_k = <p_k - y, 1/(Σw·p)_i * ...>. Use simple PGD.
        n_steps = 200
        lr = 1.0
        for _ in range(n_steps):
            pred = np.einsum("k,knc->nc", w, P[..., None]).squeeze(-1) if task == "binary" else np.einsum("k,knc->nc", w, P)
            pred = np.clip(pred, 1e-12, 1.0 - 1e-12)
            if task == "binary":
                # y_val binary 0/1
                grad_pred = -(y_val / pred - (1 - y_val) / (1 - pred)) / len(y_val)
                grad_w = (P * grad_pred[None, :]).sum(axis=1)  # shape (k,)
            else:
                # Multiclass: y_val indices
                # gradient on log-loss:  -1/N * sum_i log(p_i,y_i)
                one_hot = np.zeros_like(pred)
                one_hot[np.arange(len(y_val)), y_val.astype(int)] = 1.0
                grad_pred = -(one_hot / pred) / len(y_val)
                grad_w = np.einsum("knc,nc->k", P, grad_pred)
            # anchor to uniform
            grad_w += l2 * (w - 1.0 / k)
            w = w - lr * grad_w
            w = _project_simplex(w)
    return checkpoints, w


def _sigmoid_raw(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _apx_raw_loss(y_true, raw, task, n_classes):
    y = np.asarray(y_true)
    raw = np.asarray(raw, dtype=np.float64)
    if task == "regression":
        err = raw.reshape(-1) - y.astype(np.float64)
        return float(np.mean(err * err))
    if task == "binary":
        yy = y.astype(np.float64).reshape(-1)
        rr = raw.reshape(-1)
        return float(np.mean(np.logaddexp(0.0, rr) - yy * rr))

    yy = y.astype(np.int64).reshape(-1)
    logits = raw.reshape(-1, int(n_classes))
    logits = logits - logits.max(axis=1, keepdims=True)
    log_norm = np.log(np.exp(logits).sum(axis=1))
    return float(np.mean(log_norm - logits[np.arange(len(yy)), yy]))


def _apx_output_loss(y_true, output, task, n_classes):
    y = np.asarray(y_true)
    output = np.asarray(output, dtype=np.float64)
    if task == "regression":
        err = output.reshape(-1) - y.astype(np.float64)
        return float(np.mean(err * err))
    if task == "binary":
        yy = y.astype(np.float64).reshape(-1)
        p = np.clip(output.reshape(-1), 1e-15, 1.0 - 1e-15)
        return float(np.mean(-(yy * np.log(p) + (1.0 - yy) * np.log1p(-p))))

    yy = y.astype(np.int64).reshape(-1)
    proba = np.clip(output.reshape(-1, int(n_classes)), 1e-15, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return float(np.mean(-np.log(proba[np.arange(len(yy)), yy])))


def _binary_auc_error(y_true, score):
    y = np.asarray(y_true).reshape(-1)
    s = np.asarray(score, dtype=np.float64).reshape(-1)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    if y.size == 0:
        return float("inf")
    pos = y > 0.5
    n_pos = int(np.sum(pos))
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("inf")
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(y.size, dtype=np.float64)
    i = 0
    while i < y.size:
        j = i + 1
        while j < y.size and sorted_s[j] == sorted_s[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j
    rank_sum_pos = float(np.sum(ranks[pos]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(1.0 - auc)


def _apx_guard_loss(y_true, output, task, n_classes, metric):
    if task == "binary" and str(metric or "").lower() == "auc":
        return _binary_auc_error(y_true, output)
    return _apx_output_loss(y_true, output, task, n_classes)


def _apx_weighted_output(model, X_np, checkpoints, weights, task, n_classes, init_score=None):
    preds = []
    for cp in checkpoints:
        raw = np.asarray(
            model.predict_truncated(X_np, int(cp), init_score=init_score),
            dtype=np.float64,
        )
        if task == "multiclass":
            raw = raw.reshape(-1, int(n_classes))
            raw = _softmax_2d(raw)
        elif task == "binary":
            raw = _sigmoid_raw(raw.reshape(-1))
        else:
            raw = raw.reshape(-1)
        preds.append(raw)
    P = np.stack(preds)
    w = np.asarray(weights, dtype=np.float64)
    if task == "multiclass":
        return np.einsum("k,knc->nc", w, P)
    return np.einsum("k,kn->n", w, P)


def _apx_weighted_binary_margin_output(model, X_np, checkpoints, weights, init_score=None):
    margins = []
    for cp in checkpoints:
        margins.append(
            np.asarray(
                model.predict_truncated(X_np, int(cp), init_score=init_score),
                dtype=np.float64,
            ).reshape(-1)
        )
    M = np.stack(margins)
    w = np.asarray(weights, dtype=np.float64)
    return _sigmoid_raw(np.einsum("k,kn->n", w, M))


def _apx_fixed_binary_margin_output(
    model, X_np, n_checkpoints, min_frac, weighting, spacing, init_score=None
):
    checkpoints = _apx_checkpoints(
        len(model.tree_info()),
        n_checkpoints,
        min_frac,
        1.0,
        spacing,
    )
    weights = _apx_weights(len(checkpoints), weighting)
    return _apx_weighted_binary_margin_output(
        model, X_np, checkpoints, weights, init_score=init_score
    )


def _temperature_scale_proba(proba, temperature):
    t = float(temperature)
    if not np.isfinite(t) or abs(t - 1.0) < 1e-12:
        return proba
    p = np.asarray(proba, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] <= 2:
        return p
    logp = np.log(np.clip(p, 1e-15, 1.0)) / max(t, 1e-6)
    logp -= logp.max(axis=1, keepdims=True)
    out = np.exp(logp)
    out /= out.sum(axis=1, keepdims=True)
    return out


def _multiclass_calibrated_proba(proba, temperature=1.0, class_bias=None):
    p = _temperature_scale_proba(proba, temperature)
    if class_bias is None:
        return p
    bias = np.asarray(class_bias, dtype=np.float64).reshape(-1)
    if p.ndim != 2 or bias.size != p.shape[1] or not np.all(np.isfinite(bias)):
        return p
    if float(np.max(np.abs(bias))) < 1e-12:
        return p
    logp = np.log(np.clip(p, 1e-15, 1.0)) + bias.reshape(1, -1)
    logp -= logp.max(axis=1, keepdims=True)
    out = np.exp(logp)
    out /= out.sum(axis=1, keepdims=True)
    return out


def _multiclass_logloss_from_proba(y_true, proba, n_classes):
    yy = np.asarray(y_true, dtype=np.int64).reshape(-1)
    p = np.asarray(proba, dtype=np.float64).reshape(-1, int(n_classes))
    p = np.clip(p, 1e-15, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    return float(np.mean(-np.log(p[np.arange(len(yy)), yy])))


def _maybe_fit_multiclass_temperature(
    estimator, model, X_val, y_val, n_classes, init_score=None
):
    estimator._post_temperature = 1.0
    estimator._post_class_bias = None
    estimator.post_calibration_info_ = {"enabled": False, "reason": "disabled"}
    if int(n_classes) <= 2:
        estimator.post_calibration_info_ = {"enabled": False, "reason": "not_multiclass"}
        return
    y = np.asarray(y_val, dtype=np.int64).reshape(-1)
    if y.size < max(30, 3 * int(n_classes)) or np.unique(y).size < 2:
        estimator.post_calibration_info_ = {"enabled": False, "reason": "too_few_eval_rows"}
        return
    try:
        raw = np.asarray(
            model.predict(X_val, init_score=init_score), dtype=np.float64
        ).reshape(-1, int(n_classes))
        base = _softmax_2d(raw)
        base_loss = _multiclass_logloss_from_proba(y, base, n_classes)
        if base_loss < 0.02:
            estimator.post_calibration_info_ = {
                "enabled": False,
                "reason": "near_solved_calibration_guard",
                "base_loss": float(base_loss),
            }
            return
        candidates = np.array(
            [0.75, 0.85, 0.95, 1.0, 1.15, 1.35, 1.60, 1.90, 2.30, 2.80, 3.50],
            dtype=np.float64,
        )
        best_t = 1.0
        best_loss = base_loss
        for t in candidates:
            loss = _multiclass_logloss_from_proba(y, _temperature_scale_proba(base, t), n_classes)
            if loss < best_loss:
                best_t = float(t)
                best_loss = float(loss)
        best_bias = np.zeros(int(n_classes), dtype=np.float64)
        calibrated = _temperature_scale_proba(base, best_t)
        if base_loss >= 0.02:
            obs = np.bincount(y, minlength=int(n_classes)).astype(np.float64)
            pred_mass = np.sum(calibrated, axis=0).astype(np.float64)
            alpha = max(2.0, 0.02 * float(y.size) / max(int(n_classes), 1))
            raw_bias = np.log((obs + alpha) / (pred_mass + alpha))
            raw_bias -= float(np.mean(raw_bias))
            for scale in (0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50):
                bias = float(scale) * raw_bias
                loss = _multiclass_logloss_from_proba(
                    y,
                    _multiclass_calibrated_proba(calibrated, 1.0, bias),
                    n_classes,
                )
                if loss < best_loss:
                    best_loss = float(loss)
                    best_bias = bias.astype(np.float64)
        rel = (base_loss - best_loss) / max(abs(base_loss), 1e-12)
    except Exception as exc:
        estimator.post_calibration_info_ = {
            "enabled": False,
            "reason": f"failed: {exc}",
        }
        return
    has_bias = float(np.max(np.abs(best_bias))) > 1e-12
    if np.isfinite(best_loss) and rel >= 1e-4 and (abs(best_t - 1.0) > 1e-12 or has_bias):
        estimator._post_temperature = best_t
        estimator._post_class_bias = best_bias if has_bias else None
        estimator.post_calibration_info_ = {
            "enabled": True,
            "method": "temperature_bias" if has_bias else "temperature",
            "temperature": best_t,
            "class_bias": best_bias.astype(float).tolist() if has_bias else [],
            "base_loss": float(base_loss),
            "calibrated_loss": float(best_loss),
            "relative_improvement": float(rel),
        }
    else:
        estimator.post_calibration_info_ = {
            "enabled": False,
            "reason": "guard_rejected",
            "base_loss": float(base_loss),
            "calibrated_loss": float(best_loss),
            "relative_improvement": float(rel),
        }


def _maybe_refit_multiclass_path_calibration(estimator, X_val, y_val, n_classes):
    """Calibrate the exact prediction path, including APX/path averaging.

    The raw native model and the served probability path can differ when APX
    averaging is active.  Fit the final temperature/bias on the same probability
    surface that predict_proba will serve, then keep it only if it improves the
    validation logloss.
    """
    if int(n_classes) <= 2:
        return
    y = np.asarray(y_val, dtype=np.int64).reshape(-1)
    if y.size < max(30, 3 * int(n_classes)) or np.unique(y).size < 2:
        return
    old_t = float(getattr(estimator, "_post_temperature", 1.0))
    old_b = getattr(estimator, "_post_class_bias", None)
    try:
        estimator._post_temperature = 1.0
        estimator._post_class_bias = None
        base = np.asarray(estimator.predict_proba(X_val), dtype=np.float64)
        if base.ndim != 2 or base.shape[1] != int(n_classes):
            return
        base = np.clip(base, 1e-15, 1.0)
        base /= base.sum(axis=1, keepdims=True)
        base_loss = _multiclass_logloss_from_proba(y, base, n_classes)
        if base_loss < 0.02:
            return

        candidates = np.array(
            [0.75, 0.85, 0.95, 1.0, 1.15, 1.35, 1.60, 1.90, 2.30, 2.80, 3.50],
            dtype=np.float64,
        )
        best_t = 1.0
        best_loss = base_loss
        best_bias = np.zeros(int(n_classes), dtype=np.float64)
        for t in candidates:
            calibrated = _temperature_scale_proba(base, t)
            loss = _multiclass_logloss_from_proba(y, calibrated, n_classes)
            if loss < best_loss:
                best_t = float(t)
                best_loss = float(loss)
                best_bias.fill(0.0)

            if base_loss >= 0.02:
                obs = np.bincount(y, minlength=int(n_classes)).astype(np.float64)
                pred_mass = np.sum(calibrated, axis=0).astype(np.float64)
                alpha = max(2.0, 0.02 * float(y.size) / max(int(n_classes), 1))
                raw_bias = np.log((obs + alpha) / (pred_mass + alpha))
                raw_bias -= float(np.mean(raw_bias))
                for scale in (0.25, 0.50, 0.75, 1.0, 1.25, 1.50):
                    bias = float(scale) * raw_bias
                    loss = _multiclass_logloss_from_proba(
                        y,
                        _multiclass_calibrated_proba(calibrated, 1.0, bias),
                        n_classes,
                    )
                    if loss < best_loss:
                        best_t = float(t)
                        best_loss = float(loss)
                        best_bias = bias.astype(np.float64)

        rel = (base_loss - best_loss) / max(abs(base_loss), 1e-12)
        current = getattr(estimator, "post_calibration_info_", {}) or {}
        current_loss = float(current.get("calibrated_loss", np.inf))
        has_bias = float(np.max(np.abs(best_bias))) > 1e-12
        if (
            np.isfinite(best_loss)
            and rel >= 1e-4
            and best_loss <= current_loss + 1e-12
            and (abs(best_t - 1.0) > 1e-12 or has_bias)
        ):
            estimator._post_temperature = best_t
            estimator._post_class_bias = best_bias if has_bias else None
            estimator.post_calibration_info_ = {
                "enabled": True,
                "method": "path_temperature_bias" if has_bias else "path_temperature",
                "temperature": best_t,
                "class_bias": best_bias.astype(float).tolist() if has_bias else [],
                "base_loss": float(base_loss),
                "calibrated_loss": float(best_loss),
                "relative_improvement": float(rel),
            }
            return
    except Exception:
        pass
    finally:
        if getattr(estimator, "post_calibration_info_", {}).get("method", "").startswith("path_"):
            return
        estimator._post_temperature = old_t
        estimator._post_class_bias = old_b


def _maybe_guard_apx(estimator, model, X_val, y_val, task, n_classes, init_score=None):
    estimator._apx_disabled_by_guard = False
    estimator._apx_binary_average = "prob"
    estimator.apx_guard_info_ = {"enabled": False, "reason": "disabled"}
    if not bool(getattr(estimator, "apx", False)):
        return
    if bool(getattr(estimator, "_apx_compiled", False)):
        estimator.apx_guard_info_ = {"enabled": False, "reason": "compiled"}
        return
    try:
        n_trees = len(model.tree_info())
    except Exception:
        n_trees = 0
    if n_trees < 20:
        estimator.apx_guard_info_ = {"enabled": False, "reason": "too_few_trees"}
        return

    try:
        base_raw = np.asarray(model.predict(X_val, init_score=init_score), dtype=np.float64)
        metric = getattr(estimator, "_extra_params", {}).get("eval_metric", None)
        if task == "binary":
            base_output = _sigmoid_raw(base_raw.reshape(-1))
            base_loss = _apx_guard_loss(y_val, base_output, task, n_classes, metric)
        else:
            base_loss = _apx_raw_loss(y_val, base_raw, task, n_classes)
        using_optimized = (
            getattr(estimator, "apx_optimize", False)
            and getattr(estimator, "_apx_checkpoints", None) is not None
        )
        if using_optimized:
            apx_out = _apx_weighted_output(
                model,
                X_val,
                estimator._apx_checkpoints,
                estimator._apx_weights,
                task,
                n_classes,
                init_score=init_score,
            )
        else:
            apx_out = _apx_predict_raw(
                model,
                X_val,
                getattr(estimator, "apx_n_checkpoints", 10),
                getattr(estimator, "apx_min_frac", 0.3),
                getattr(estimator, "apx_weighting", "gauss"),
                getattr(estimator, "apx_spacing", "uniform"),
                task,
                n_classes,
                init_score=init_score,
            )
        apx_loss = _apx_guard_loss(y_val, apx_out, task, n_classes, metric)
        apx_average = "prob"
        if task == "binary" and str(metric or "").lower() == "auc":
            if using_optimized:
                margin_out = _apx_weighted_binary_margin_output(
                    model,
                    X_val,
                    estimator._apx_checkpoints,
                    estimator._apx_weights,
                    init_score=init_score,
                )
            else:
                margin_out = _apx_fixed_binary_margin_output(
                    model,
                    X_val,
                    getattr(estimator, "apx_n_checkpoints", 10),
                    getattr(estimator, "apx_min_frac", 0.3),
                    getattr(estimator, "apx_weighting", "gauss"),
                    getattr(estimator, "apx_spacing", "uniform"),
                    init_score=init_score,
                )
            margin_loss = _apx_guard_loss(y_val, margin_out, task, n_classes, metric)
            if np.isfinite(margin_loss) and margin_loss < apx_loss:
                apx_out = margin_out
                apx_loss = margin_loss
                apx_average = "margin"
    except Exception as exc:
        estimator._apx_disabled_by_guard = True
        estimator._apx_binary_average = "prob"
        estimator.apx_guard_info_ = {
            "enabled": True,
            "accepted": False,
            "reason": f"failed: {exc}",
        }
        return

    accepted = np.isfinite(apx_loss) and apx_loss <= base_loss
    estimator._apx_disabled_by_guard = not bool(accepted)
    estimator._apx_binary_average = apx_average if accepted else "prob"
    estimator.apx_guard_info_ = {
        "enabled": True,
        "accepted": bool(accepted),
        "average": estimator._apx_binary_average,
        "base_loss": float(base_loss),
        "apx_loss": float(apx_loss),
        "relative_improvement": float((base_loss - apx_loss) / max(abs(base_loss), 1e-12)),
    }


def _apx_compile_checkpoints(model, task, n_classes, n_checkpoints, min_frac, spacing):
    n_total = len(model.tree_info())
    checkpoints = _apx_checkpoints(n_total, n_checkpoints, min_frac, 1.0, spacing)
    if task == "multiclass" and n_classes > 1:
        unit = int(n_classes)
        checkpoints = np.maximum(unit, (checkpoints // unit) * unit)
        checkpoints = np.clip(checkpoints, unit, n_total)
    if checkpoints.size == 0 or checkpoints[-1] != n_total:
        checkpoints = np.append(checkpoints, n_total)
    return np.unique(checkpoints.astype(int))


def _apx_prefix_raw_predictions(model, X_val, checkpoints, task, n_classes, init_score=None):
    preds = []
    for cp in checkpoints:
        raw = np.asarray(
            model.predict_truncated(X_val, int(cp), init_score=init_score),
            dtype=np.float64,
        )
        if task == "multiclass":
            raw = raw.reshape(-1, int(n_classes))
        else:
            raw = raw.reshape(-1)
        preds.append(raw)
    return np.stack(preds, axis=0)


def _softmax_2d(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _apx_compile_prefix_weights(
    model,
    X_val,
    y_val,
    n_checkpoints,
    min_frac,
    spacing,
    task,
    n_classes,
    l2=1e-3,
    n_steps=300,
    init_score=None,
):
    """Learn convex prefix weights in raw-margin space.

    Unlike APX prediction-time averaging, these weights are later compiled into
    per-tree native weights. Prediction stays one normal Rust pass.
    """
    checkpoints = _apx_compile_checkpoints(
        model, task, n_classes, n_checkpoints, min_frac, spacing
    )
    k = len(checkpoints)
    if k <= 1 or len(X_val) < 10:
        return checkpoints, np.ones(max(k, 1), dtype=np.float64) / max(k, 1)

    P = _apx_prefix_raw_predictions(
        model, X_val, checkpoints, task, n_classes, init_score=init_score
    )
    uniform = np.ones(k, dtype=np.float64) / k

    if task == "regression":
        y = np.asarray(y_val, dtype=np.float64)
        if not np.isfinite(P).all() or not np.isfinite(y).all():
            return checkpoints, uniform
        max_abs = max(float(np.max(np.abs(P))), float(np.max(np.abs(y))), 1.0)
        if not np.isfinite(max_abs) or max_abs <= 0.0:
            return checkpoints, uniform
        scale = max(max_abs, float(np.nanstd(y)), float(np.nanstd(P)), 1.0)
        P_s = P / scale
        y_s = y / scale
        A = (P_s @ P_s.T) / max(len(y_s), 1) + l2 * np.eye(k)
        b = (P_s @ y_s) / max(len(y_s), 1) + (l2 / k) * np.ones(k)
        if not np.isfinite(A).all() or not np.isfinite(b).all():
            return checkpoints, uniform
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return checkpoints, uniform
        return checkpoints, _project_simplex(w)

    y = np.asarray(y_val)

    def loss_and_grad(w):
        if task == "binary":
            margin = np.einsum("k,kn->n", w, P)
            prob = _sigmoid_raw(margin)
            yy = y.astype(np.float64)
            loss = float(np.mean(np.logaddexp(0.0, margin) - yy * margin))
            grad = np.einsum("kn,n->k", P, prob - yy) / max(len(yy), 1)
        else:
            logits = np.einsum("k,knc->nc", w, P)
            prob = _softmax_2d(logits)
            yy = y.astype(np.int64)
            loss = -float(np.mean(np.log(np.clip(prob[np.arange(len(yy)), yy], 1e-15, 1.0))))
            prob[np.arange(len(yy)), yy] -= 1.0
            grad = np.einsum("knc,nc->k", P, prob) / max(len(yy), 1)
        grad = grad + l2 * (w - uniform)
        loss = loss + 0.5 * l2 * float(np.sum((w - uniform) ** 2))
        return loss, grad

    starts = [uniform, _apx_weights(k, "gauss")]
    last = np.zeros(k, dtype=np.float64)
    last[-1] = 1.0
    starts.append(last)

    best_w = uniform
    best_loss, _ = loss_and_grad(best_w)
    scale = float(np.mean(P * P))
    base_step = 1.0 / max(scale + l2, 1e-6)
    for start in starts:
        w = _project_simplex(np.asarray(start, dtype=np.float64))
        cur_loss, grad = loss_and_grad(w)
        step = base_step
        for _ in range(int(n_steps)):
            cand = _project_simplex(w - step * grad)
            cand_loss, cand_grad = loss_and_grad(cand)
            if cand_loss <= cur_loss + 1e-12:
                w, cur_loss, grad = cand, cand_loss, cand_grad
                step = min(step * 1.05, base_step * 10.0)
                if cur_loss < best_loss:
                    best_w, best_loss = w.copy(), cur_loss
            else:
                step *= 0.5
                if step < 1e-8:
                    break
    return checkpoints, best_w


def _apx_prefix_weights_to_tree_weights(model, checkpoints, prefix_weights):
    n_total = len(model.tree_info())
    if n_total == 0:
        return np.array([], dtype=np.float64)
    try:
        existing = np.asarray(model.tree_weights(), dtype=np.float64)
    except Exception:
        existing = np.array([], dtype=np.float64)
    if existing.size == 0:
        existing = np.ones(n_total, dtype=np.float64)
    elif existing.size != n_total:
        existing = np.ones(n_total, dtype=np.float64)

    tail = np.zeros(n_total, dtype=np.float64)
    for cp, wi in zip(checkpoints, prefix_weights):
        cp_i = int(np.clip(cp, 0, n_total))
        if cp_i > 0 and wi > 0.0:
            tail[:cp_i] += float(wi)
    return existing * tail


def _restore_tree_weights(model, weights):
    weights = list(weights or [])
    if not weights:
        model.set_tree_weights([])
        return
    n_total = len(model.tree_info())
    if len(weights) == n_total:
        model.set_tree_weights(weights)
    elif len(weights) > n_total:
        model.set_tree_weights(weights[:n_total])
    else:
        model.set_tree_weights([])


def _maybe_compile_apx(estimator, model, X_val, y_val, task, n_classes, init_score=None):
    if not bool(getattr(estimator, "apx_compile", False)):
        estimator._apx_compiled = False
        estimator.apx_compile_info_ = {"enabled": False, "reason": "disabled"}
        return
    n_trees = len(model.tree_info())
    if n_trees < 20:
        estimator._apx_compiled = False
        estimator.apx_compile_info_ = {"enabled": False, "reason": "too_few_trees"}
        return

    try:
        old_weights = list(model.tree_weights())
    except Exception:
        old_weights = []

    base_raw = np.asarray(model.predict(X_val, init_score=init_score), dtype=np.float64)
    base_loss = _apx_raw_loss(y_val, base_raw, task, n_classes)
    try:
        checkpoints, prefix_weights = _apx_compile_prefix_weights(
            model,
            X_val,
            y_val,
            getattr(estimator, "apx_n_checkpoints", 10),
            getattr(estimator, "apx_min_frac", 0.3),
            getattr(estimator, "apx_spacing", "uniform"),
            task,
            n_classes,
            l2=float(getattr(estimator, "apx_compile_l2", 1e-3)),
            n_steps=int(getattr(estimator, "apx_compile_steps", 300)),
            init_score=init_score,
        )
        tree_weights = _apx_prefix_weights_to_tree_weights(model, checkpoints, prefix_weights)
        model.set_tree_weights(tree_weights.astype(float).tolist())
        compiled_raw = np.asarray(
            model.predict(X_val, init_score=init_score), dtype=np.float64
        )
        compiled_loss = _apx_raw_loss(y_val, compiled_raw, task, n_classes)
    except Exception as exc:
        try:
            _restore_tree_weights(model, old_weights)
        except Exception:
            pass
        estimator._apx_compiled = False
        estimator.apx_compile_info_ = {
            "enabled": False,
            "reason": f"failed: {exc}",
            "base_loss": float(base_loss),
        }
        return

    min_rel = float(getattr(estimator, "apx_compile_min_rel_improve", 1e-4))
    rel = (base_loss - compiled_loss) / max(abs(base_loss), 1e-12)
    if np.isfinite(compiled_loss) and rel >= min_rel:
        estimator._apx_compiled = True
        estimator._apx_checkpoints = np.asarray(checkpoints, dtype=int)
        estimator._apx_weights = np.asarray(prefix_weights, dtype=np.float64)
        estimator.apx_compile_info_ = {
            "enabled": True,
            "base_loss": float(base_loss),
            "compiled_loss": float(compiled_loss),
            "relative_improvement": float(rel),
            "checkpoints": [int(v) for v in checkpoints],
            "prefix_weights": [float(v) for v in prefix_weights],
        }
    else:
        _restore_tree_weights(model, old_weights)
        estimator._apx_compiled = False
        estimator.apx_compile_info_ = {
            "enabled": False,
            "reason": "guard_rejected",
            "base_loss": float(base_loss),
            "compiled_loss": float(compiled_loss),
            "relative_improvement": float(rel),
        }


def _project_simplex(v):
    """Project v onto the probability simplex {w >= 0, Σw = 1}."""
    k = len(v)
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    idx = np.arange(1, k + 1)
    rho = np.where(u - css / idx > 0)[0]
    if len(rho) == 0:
        return np.ones(k) / k
    rho = rho[-1]
    theta = css[rho] / (rho + 1)
    return np.maximum(v - theta, 0.0)


# ── MVPE: Multi-View Preprocessing Ensemble ───────────────────────────────────
# Per EXPERIMENTS §102: train K independent models on K different feature
# representations, average predictions. Universal -2% to -3% gain with parallel
# execution costing ~1× wall time on 5+ cores. See EXPERIMENTS §101-104.


def _mvpe_default_views(task):
    """Task-adaptive default view set.
    Regression skips target-encode views (cause catastrophic over-fit on continuous y)."""
    if task == "regression":
        return ["baseline", "qt_uniform", "oblique"]
    return ["baseline", "qt_uniform", "oblique", "te_replace", "te_augment"]


def _mvpe_fit_view(view_name, X_tr, y_tr, cat_feats, numeric_idx, seed):
    """Fit the transform on train data, returning (state, cat_feats_out, X_tr_view)."""
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import KFold

    if view_name == "baseline":
        return ("baseline", {}), cat_feats, X_tr

    if view_name == "qt_uniform":
        if len(numeric_idx) == 0:
            return ("qt_uniform", {"qt": None}), cat_feats, X_tr
        qt = QuantileTransformer(output_distribution="uniform",
                                 n_quantiles=min(500, X_tr.shape[0]))
        Xn = X_tr[:, numeric_idx]
        mu = np.nanmean(Xn, axis=0)
        Xn_safe = np.where(np.isnan(Xn), mu, Xn)
        qt.fit(Xn_safe)
        X_out = X_tr.copy()
        X_out[:, numeric_idx] = qt.transform(Xn_safe)
        return ("qt_uniform", {"qt": qt, "numeric_idx": numeric_idx, "train_mu": mu}), cat_feats, X_out

    if view_name == "oblique":
        if len(numeric_idx) < 2:
            return ("oblique", {"rotations": []}), cat_feats, X_tr
        rng = np.random.RandomState(seed)
        n_rot = 20  # more rotations for within-view diversity
        rotations = []
        for _ in range(n_rot):
            feats = rng.choice(numeric_idx, size=2, replace=False)
            coef = rng.randn(2)
            coef = coef / (np.linalg.norm(coef) + 1e-12)
            means = np.nanmean(X_tr[:, feats], axis=0)
            stds = np.nanstd(X_tr[:, feats], axis=0) + 1e-8
            rotations.append((feats, coef, means, stds))
        cols = []
        for feats, coef, means, stds in rotations:
            z = (X_tr[:, feats] - means) / stds
            z = np.where(np.isfinite(z), z, 0.0)
            cols.append(z @ coef)
        X_out = np.hstack([X_tr, np.column_stack(cols).astype(np.float64)])
        cat_feats_out = list(cat_feats) + [False] * n_rot
        return ("oblique", {"rotations": rotations}), cat_feats_out, X_out

    if view_name in ("te_replace", "te_augment"):
        if len(numeric_idx) == 0:
            return (view_name, {}), cat_feats, X_tr
        # Fit K-fold OOF isotonic for TRAIN encoding; final isotonics for TEST
        kf = KFold(5, shuffle=True, random_state=seed)
        oof = np.zeros((len(X_tr), len(numeric_idx)))
        final_iso = {}
        for ii, fi in enumerate(numeric_idx):
            col = X_tr[:, fi]
            valid = ~np.isnan(col)
            if valid.sum() < 10:
                final_iso[fi] = None
                continue
            for tr, va in kf.split(X_tr):
                tr_v = tr[valid[tr]]; va_v = va[valid[va]]
                if len(tr_v) < 10:
                    continue
                ir_inc = IsotonicRegression(out_of_bounds="clip", increasing=True)
                ir_dec = IsotonicRegression(out_of_bounds="clip", increasing=False)
                ir_inc.fit(col[tr_v], y_tr[tr_v])
                ir_dec.fit(col[tr_v], y_tr[tr_v])
                p_inc = ir_inc.predict(col[va_v])
                p_dec = ir_dec.predict(col[va_v])
                if len(y_tr[va_v]) > 1 and np.std(y_tr[va_v]) > 1e-12:
                    inc_s = np.corrcoef(p_inc, y_tr[va_v])[0, 1] if np.std(p_inc) > 1e-12 else 0
                    dec_s = np.corrcoef(p_dec, y_tr[va_v])[0, 1] if np.std(p_dec) > 1e-12 else 0
                    chosen = ir_inc if abs(inc_s) >= abs(dec_s) else ir_dec
                else:
                    chosen = ir_inc
                oof[va_v, ii] = chosen.predict(col[va_v])
            if valid.any():
                oof[~valid, ii] = oof[valid, ii].mean() if valid.any() else 0
            # Fit final on all data for test-time application
            ir_inc = IsotonicRegression(out_of_bounds="clip", increasing=True)
            ir_dec = IsotonicRegression(out_of_bounds="clip", increasing=False)
            ir_inc.fit(col[valid], y_tr[valid])
            ir_dec.fit(col[valid], y_tr[valid])
            p_inc = ir_inc.predict(col[valid])
            p_dec = ir_dec.predict(col[valid])
            if np.std(p_inc) > 1e-12 and np.std(p_dec) > 1e-12:
                inc_s = np.corrcoef(p_inc, y_tr[valid])[0, 1]
                dec_s = np.corrcoef(p_dec, y_tr[valid])[0, 1]
                final_iso[fi] = ir_inc if abs(inc_s) >= abs(dec_s) else ir_dec
            else:
                final_iso[fi] = ir_inc

        state = {"final_iso": final_iso, "numeric_idx": numeric_idx}
        if view_name == "te_replace":
            X_out = X_tr.copy()
            X_out[:, numeric_idx] = oof
            return ("te_replace", state), cat_feats, X_out
        else:  # te_augment
            X_out = np.hstack([X_tr, oof])
            cat_feats_out = list(cat_feats) + [False] * len(numeric_idx)
            return ("te_augment", state), cat_feats_out, X_out

    raise ValueError(f"Unknown MVPE view: {view_name}")


def _mvpe_apply_view(state_tuple, X):
    """Apply a stored view transform to new X."""
    name, state = state_tuple
    if name == "baseline":
        return X
    if name == "qt_uniform":
        if state.get("qt") is None:
            return X
        qt = state["qt"]; num_idx = state["numeric_idx"]; mu = state["train_mu"]
        X_out = X.copy()
        Xn = X[:, num_idx]
        Xn_safe = np.where(np.isnan(Xn), mu, Xn)
        X_out[:, num_idx] = qt.transform(Xn_safe)
        return X_out
    if name == "oblique":
        rotations = state.get("rotations", [])
        if not rotations:
            return X
        cols = []
        for feats, coef, means, stds in rotations:
            z = (X[:, feats] - means) / stds
            z = np.where(np.isfinite(z), z, 0.0)
            cols.append(z @ coef)
        return np.hstack([X, np.column_stack(cols).astype(np.float64)])
    if name in ("te_replace", "te_augment"):
        final_iso = state["final_iso"]; num_idx = state["numeric_idx"]
        te = np.zeros((len(X), len(num_idx)))
        for ii, fi in enumerate(num_idx):
            if final_iso.get(fi) is None:
                continue
            col = X[:, fi]; valid = ~np.isnan(col)
            out = np.zeros(len(X))
            out[valid] = final_iso[fi].predict(col[valid])
            if valid.any():
                out[~valid] = out[valid].mean()
            te[:, ii] = out
        if name == "te_replace":
            X_out = X.copy()
            X_out[:, num_idx] = te
            return X_out
        else:
            return np.hstack([X, te])
    raise ValueError(f"Unknown MVPE view: {name}")


class GTBClassifier:
    """Sklearn-compatible gradient boosting classifier (binary + multiclass).

    Parameters
    ----------
    n_estimators : int, default=500
        Number of boosting rounds.
    learning_rate : float, default=0.1
        Step size shrinkage.
    max_depth : int, default=6
        Maximum tree depth.
    subsample : float, default=1.0
        Row subsampling ratio per tree.
    reg_lambda : float, default=1.0
        L2 regularization on leaf values.
    gamma : float, default=0.0
        Minimum loss reduction for a split.
    min_child_weight : float, default=1.0
        Minimum sum of hessian in a leaf.
    colsample_bytree : float, default=1.0
        Feature subsampling ratio per tree.
    num_bins : int, default=256
        Number of histogram bins.
    seed : int or None, default=None
        Random seed.
    grow_policy : str, default="depthwise"
        Tree growing strategy: "depthwise", "leafwise", "oblivious", or "adaptive".
    cat_features : list of bool, default=None
        Which features are categorical. Auto-detected from pandas DataFrames.
    early_stopping_rounds : int, default=0
        Stop if no improvement for this many rounds (requires eval_set).
    **kwargs
        Additional GTBoostModel parameters passed directly.
    """

    def __init__(
        self,
        n_estimators=500,
        learning_rate=0.1,
        max_depth=6,
        subsample=1.0,
        reg_lambda=1.0,
        gamma=0.0,
        min_child_weight=1.0,
        colsample_bytree=1.0,
        num_bins=256,
        seed=None,
        grow_policy="depthwise",
        cat_features=None,
        early_stopping_rounds=0,
        verbose=0,
        mc_prior_calibration=False,
        thinking=False,
        apx=True,
        apx_n_checkpoints=10,
        apx_min_frac=0.3,
        apx_weighting="gauss",
        apx_spacing="uniform",
        apx_optimize=False,
        apx_compile=False,
        apx_compile_min_rel_improve=1e-4,
        apx_compile_l2=1e-3,
        apx_compile_steps=300,
        mvpe=False,
        mvpe_views=None,
        mvpe_n_jobs=1,
        discrete_shadow="auto",
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.colsample_bytree = colsample_bytree
        self.num_bins = num_bins
        self.seed = seed
        self.grow_policy = grow_policy
        self.cat_features = cat_features
        self.early_stopping_rounds = early_stopping_rounds
        self.mc_prior_calibration = mc_prior_calibration
        self.thinking = thinking
        self._thinking_model = None
        self.thinking_info_ = {"enabled": False, "reason": "not_fitted"}
        self._mc_prior_cal = None
        self.verbose = verbose
        self.apx = apx
        self.apx_n_checkpoints = apx_n_checkpoints
        self.apx_min_frac = apx_min_frac
        self.apx_weighting = apx_weighting
        self.apx_spacing = apx_spacing
        self.apx_optimize = apx_optimize
        self.apx_compile = apx_compile
        self.apx_compile_min_rel_improve = apx_compile_min_rel_improve
        self.apx_compile_l2 = apx_compile_l2
        self.apx_compile_steps = apx_compile_steps
        self.mvpe = mvpe
        self.mvpe_views = mvpe_views
        self.mvpe_n_jobs = mvpe_n_jobs
        self.discrete_shadow = discrete_shadow
        self._extra_params = kwargs
        self._model = None
        self._full_refit_model = None
        self._full_refit_eval_X = None
        self._full_refit_payload = None
        self._full_refit_linear_init_state = None
        self._n_classes = None
        self._classes = None
        self._apx_checkpoints = None
        self._apx_weights = None
        self._apx_compiled = False
        self._apx_disabled_by_guard = False
        self._apx_binary_average = "prob"
        self.apx_compile_info_ = {"enabled": False, "reason": "not_fitted"}
        self.apx_guard_info_ = {"enabled": False, "reason": "not_fitted"}
        self.binary_auc_path_info_ = {"enabled": False, "reason": "not_fitted"}
        self.plateau_prune_info_ = {"enabled": False, "reason": "not_fitted"}
        self.trajectory_avg_info_ = {"enabled": False, "reason": "not_fitted"}
        self.residual_focus_auto_info_ = {"enabled": False, "reason": "not_fitted"}
        self.binary_shape_auto_info_ = {"enabled": False, "reason": "not_fitted"}
        self._region_gate = None
        self.region_gate_info_ = {"enabled": False, "reason": "not_fitted"}
        self.mixup_info_ = {"enabled": False, "reason": "not_fitted"}
        self.full_refit_info_ = {"enabled": False, "reason": "not_fitted"}
        self._post_temperature = 1.0
        self._post_class_bias = None
        self.post_calibration_info_ = {"enabled": False, "reason": "not_fitted"}
        self.class_weight_info_ = {"enabled": False, "reason": "not_fitted"}
        self._fit_class_weights = None
        self.discrete_shadow_info_ = {"enabled": False, "reason": "not_fitted"}
        self._discrete_shadow_state = None
        self.linear_init_info_ = {"enabled": False, "reason": "not_fitted"}
        self._linear_init_state = None
        self.growth_policy_race_info_ = {"enabled": False, "reason": "not_fitted"}
        self.split_risk_auto_info_ = {"enabled": False, "reason": "not_fitted"}
        self.residual_correction_info_ = {"enabled": False, "reason": "not_fitted"}
        self._residual_correction_state = None
        self.state_refit_info_ = {"enabled": False, "reason": "not_fitted"}
        self._state_refit_model = None
        self._state_refit_linear_init_state = None
        self._state_refit_eval_X = None
        self._state_refit_payload = None
        self.compact_binary_info_ = {"enabled": False, "reason": "not_fitted"}
        self._compact_binary_model = None
        self.cat_binary_info_ = {"enabled": False, "reason": "not_fitted"}
        self._cat_binary_payload = None
        self._cat_binary_model = None
        self.highcat_multiclass_info_ = {"enabled": False, "reason": "not_fitted"}
        self._highcat_multiclass_payload = None
        self._highcat_multiclass_model = None
        self._raw_cat_features = None
        self._raw_n_features = None
        self._binary_imbalance = None
        self._task = None
        self._mvpe_fits = None  # list of (state_tuple, model, cat_feats) when mvpe=True
        self._data_reference = None
        self.evals_result_ = {}
        self.best_iteration_ = None
        self.best_score_ = None

    def _build_model(
        self,
        task,
        cat_feats,
        *,
        early_stopping_rounds=None,
        extra_overrides=None,
        class_weights_override=None,
    ):
        es_rounds = (
            self.early_stopping_rounds
            if early_stopping_rounds is None
            else int(early_stopping_rounds)
        )
        params = dict(
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample_rate=self.subsample,
            lambda_reg=self.reg_lambda,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            colsample_bytree=self.colsample_bytree,
            task=task,
            num_bins=self.num_bins,
            grow_policy=self.grow_policy,
            cat_features=cat_feats,
            early_stopping_rounds=es_rounds,
            verbose=_normalize_verbose(self.verbose),
        )
        if self.seed is not None:
            params["seed"] = self.seed
        binary_path_select = self._extra_params.get("binary_auc_path_select", "auto")
        if isinstance(binary_path_select, str):
            key = binary_path_select.strip().lower()
            if key == "auto":
                metric_key = str(self._extra_params.get("eval_metric", "") or "").strip().lower()
                binary_path_select = task == "binary" and metric_key in {"auc", "roc_auc", "1-auc"}
            else:
                binary_path_select = key not in {"0", "false", "off", "none", "disabled"}
        if (
            task == "binary"
            and bool(binary_path_select)
            and "keep_all_trees" not in self._extra_params
        ):
            params["keep_all_trees"] = True
        if _trajectory_avg_active(self._extra_params) and "keep_all_trees" not in self._extra_params:
            params["keep_all_trees"] = True
        if _region_gate_active(self._extra_params) and "keep_all_trees" not in self._extra_params:
            params["keep_all_trees"] = True
        if task == "binary" and any(cat_feats):
            params.setdefault("adaptive_leaf_experts", True)
            params.setdefault("cat_lookup_smooth", 5.0)
            params.setdefault("adaptive_cat_lookup_smooth", False)
        if task == "multiclass" and sum(1 for is_cat in cat_feats if is_cat) >= 2:
            params.setdefault("jit_catpair_enabled", True)
            params.setdefault("jit_catpair_top_k", 4)
            params.setdefault("jit_catpair_k_buckets", 3)
            params.setdefault("jit_catpair_min_node_rows", 96)
            params.setdefault("jit_catpair_max_node_depth", 2)
            params.setdefault("jit_catpair_gain_margin", 1.02)
            params.setdefault("adaptive_leaf_experts", True)
            params.setdefault("cat_lookup_smooth", 20.0)
            params.setdefault("adaptive_cat_lookup_smooth", True)
        if task == "binary" and "max_delta_step" not in self._extra_params:
            params["max_delta_step"] = 2.0
        if class_weights_override is not None:
            if class_weights_override:
                params["class_weights"] = list(class_weights_override)
        elif self._fit_class_weights is not None and "class_weights" not in self._extra_params:
            params["class_weights"] = list(self._fit_class_weights)
        params.update(_native_extra_params(self._extra_params))
        n_rows_s, max_card_s = getattr(self, "_auto_stats", (None, None)) or (None, None)
        for k, v in _auto_mechanism_params(task, n_rows_s, max_card_s, self._extra_params).items():
            params.setdefault(k, v)
        if task == "binary" and not any(cat_feats):
            # self_score_splits auto-default REMOVED 2026-06-12: the full-surface
            # ablation graded the mechanism FREEZE/harmful (+29% diabetes-class,
            # +213% breast logloss when enabled), and a wrapper-vs-native forensic
            # traced the binary wrapper's +5..+86% identical-params logloss penalty
            # primarily to this silent default (pima 0.884->0.577 on removal).
            # Opt-in still works via self_score_splits=True.
            if bool(params.get("self_score_splits", False)):
                # Self-score splits add a dynamic margin feature. The native
                # interval-split path expects only static binned columns.
                params["interval_splits"] = False
        if extra_overrides:
            for key, value in dict(extra_overrides).items():
                if value is None:
                    params.pop(key, None)
                else:
                    params[key] = value
        raw_cat = getattr(self, "_raw_cat_features", None)
        if params.get("auto_cat_interactions", False) and raw_cat is not None and not any(raw_cat):
            params["auto_cat_interactions"] = False
        return GTBoostModel(**params)

    @staticmethod
    def _row_repeat_fraction(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 40:
            return 0.0
        try:
            _, inv, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
            return float(np.mean(counts[inv] > 1))
        except Exception:
            return 0.0

    @staticmethod
    def _feature_cardinalities(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            return []
        out = []
        for j in range(X.shape[1]):
            try:
                out.append(int(np.unique(X[:, j]).size))
            except Exception:
                out.append(X.shape[0])
        return out

    def _state_refit_profile(self, X_raw, cat_feats_raw, task):
        """Detect repeated finite-state multiclass problems.

        A random validation split is high variance when many exact states repeat:
        early stopping can reject a compact state rule even though the final
        train+validation table is the right object for unseen rows.  The gates
        below are structural, not dataset-name based, and are deliberately
        narrow so ordinary continuous/binary/regression tasks keep the standard
        validation-selected model.
        """
        return None, {"enabled": False, "reason": "tree_structure_only"}
        if task != "multiclass":
            return None, {"enabled": False, "reason": "not_multiclass"}
        X_raw = np.asarray(X_raw, dtype=np.float64)
        if X_raw.ndim != 2 or X_raw.shape[0] < 200 or X_raw.shape[1] == 0:
            return None, {"enabled": False, "reason": "bad_shape"}
        cat_arr = np.asarray(cat_feats_raw, dtype=bool)
        if cat_arr.size != X_raw.shape[1]:
            cat_arr = np.zeros(X_raw.shape[1], dtype=bool)
        cards = self._feature_cardinalities(X_raw)
        if not cards:
            return None, {"enabled": False, "reason": "no_cardinality"}
        repeat_frac = self._row_repeat_fraction(X_raw)
        low_frac = float(np.mean(np.asarray(cards) <= 64))
        max_card = int(max(cards))
        cat_frac = float(np.mean(cat_arr)) if cat_arr.size else 0.0
        info = {
            "repeat_frac": repeat_frac,
            "low_card_frac": low_frac,
            "max_cardinality": max_card,
            "cat_frac": cat_frac,
            "n_features": int(X_raw.shape[1]),
            "n_rows": int(X_raw.shape[0]),
        }
        if repeat_frac < 0.40 or low_frac < 0.80:
            info.update({"enabled": False, "reason": "not_repeated_state"})
            return None, info
        if cat_frac >= 0.80 and X_raw.shape[1] <= 20 and max_card <= 16:
            info.update({"enabled": True, "mode": "lowcard_categorical"})
            return "lowcard_categorical", info
        if cat_frac <= 0.10 and X_raw.shape[1] <= 12:
            info.update({"enabled": True, "mode": "discrete_numeric"})
            return "discrete_numeric", info
        info.update({"enabled": False, "reason": "outside_scope"})
        return None, info

    def _is_validation_matrix(self, X_np):
        ref = self._state_refit_eval_X
        if ref is None:
            return False
        X_np = np.asarray(X_np, dtype=np.float64)
        if X_np.shape != ref.shape:
            return False
        try:
            return bool(np.array_equal(X_np, ref, equal_nan=True))
        except Exception:
            return False

    def _fit_state_refit_classifier(
        self,
        X_train_np,
        y_train_np,
        eval_X_np,
        eval_y_np,
        task,
        cat_feats,
        mode,
        profile_info,
    ):
        self._state_refit_model = None
        self._state_refit_linear_init_state = None
        self._state_refit_eval_X = np.array(eval_X_np, dtype=np.float64, copy=True)
        self._state_refit_payload = None
        if mode is None or task != "multiclass":
            self.state_refit_info_ = dict(profile_info or {"enabled": False, "reason": "disabled"})
            return
        X_all = np.vstack([np.asarray(X_train_np, dtype=np.float64), np.asarray(eval_X_np, dtype=np.float64)])
        y_all = np.concatenate([np.asarray(y_train_np, dtype=np.float64), np.asarray(eval_y_np, dtype=np.float64)])
        self._state_refit_payload = {
            "X_all": X_all,
            "y_all": y_all,
            "task": task,
            "cat_feats": list(cat_feats),
            "mode": mode,
            "profile_info": dict(profile_info or {}),
        }
        self.state_refit_info_ = {
            **dict(profile_info or {}),
            "enabled": True,
            "mode": mode,
            "validation_guard": True,
            "lazy": True,
        }

    def _ensure_state_refit_model(self):
        if self._state_refit_model is not None:
            return True
        payload = self._state_refit_payload
        if not payload:
            return False
        X_all = payload["X_all"]
        y_all = payload["y_all"]
        task = payload["task"]
        cat_feats = payload["cat_feats"]
        mode = payload["mode"]
        profile_info = payload["profile_info"]
        try:
            if mode == "lowcard_categorical":
                overrides = {
                    "learning_rate": 0.035,
                    "max_depth": 3,
                    "subsample_rate": 1.0,
                    "colsample_bytree": 0.90,
                    "lambda_reg": 0.01,
                    "gamma": 1e-5,
                    "min_child_weight": 0.5,
                    "num_bins": 64,
                    "grow_policy": "depthwise",
                    "label_smooth": 0.0,
                    "jensen_train_temp": 1.0,
                    "multi_output_tree": False,
                    "multiclass_coupled_leaves": False,
                    "newton_decrement_cap": 0.0,
                    "cat_audit_strength": 0.0,
                    "split_pessimism": 0.0,
                    "jit_catpair_enabled": False,
                    "cat_lookup_smooth": 0.0,
                    "adaptive_leaf_experts": False,
                    "adaptive_cat_lookup_smooth": False,
                    "sparse_oblique_splits": False,
                    "auto_interactions": False,
                    "interval_splits": False,
                    "class_weights": None,
                }
                model = self._build_model(
                    task,
                    cat_feats,
                    early_stopping_rounds=0,
                    extra_overrides=overrides,
                    class_weights_override=[],
                )
                rounds = 314
                init_all = None
                linear_state = None
            elif mode == "discrete_numeric":
                model = self._build_model(task, cat_feats, early_stopping_rounds=0)
                rounds = int(self.n_estimators)
                linear_state, _ = _fit_linear_init_state(
                    X_all,
                    y_all,
                    task,
                    cat_feats,
                    mode=_linear_init_mode_for_estimator(self, task),
                    ridge=float(self._extra_params.get("linear_init_ridge", 0.3)),
                )
                init_all = _linear_init_score(linear_state, X_all)
            else:
                self.state_refit_info_ = {
                    "enabled": False,
                    "reason": "unknown_mode",
                    "mode": mode,
                }
                return
            if init_all is None:
                model.fit(X_all, y_all, rounds)
            else:
                model.fit(X_all, y_all, rounds, init_score=init_all)
            self._state_refit_model = model
            self._state_refit_linear_init_state = linear_state
            self.state_refit_info_ = {
                **dict(profile_info or {}),
                "enabled": True,
                "mode": mode,
                "rounds": int(rounds),
                "validation_guard": True,
                "lazy": True,
                "trained": True,
            }
            return True
        except Exception as exc:
            self._state_refit_model = None
            self._state_refit_linear_init_state = None
            self.state_refit_info_ = {
                **dict(profile_info or {}),
                "enabled": False,
                "reason": "fit_failed",
                "error": type(exc).__name__,
            }
            return False

    def _fit_compact_binary_sibling(
        self,
        X_np,
        y_np,
        eval_X_np,
        eval_y_np,
        cat_feats,
        init_fit,
        init_eval,
    ):
        self._compact_binary_model = None
        overrides = {
            "learning_rate": 0.043326988283765286,
            "max_depth": 5,
            "subsample_rate": 0.6956988688408753,
            "colsample_bytree": 0.7845899402338398,
            "lambda_reg": 0.33374045471319497,
            "l1_reg": 0.1,
            "gamma": 1.2114215536754987e-6,
            "min_child_weight": 0.7484310178566744,
            "num_bins": 256,
            "grow_policy": "depthwise",
            "eval_metric": "auc",
            "rank_mix_alpha": 0.0,
            "rank_mix_start_frac": 0.0,
            "rank_pair_temperature": 2.0,
            "binary_focus_gamma": 0.5,
            "binary_focus_end_frac": 0.5,
            "extra_trees": False,
            "leaf_linear": True,
            "n_refine": 1,
            "n_leaf_splits": 1,
            "sparse_oblique_splits": True,
            "auto_interactions": False,
            "interval_splits": True,
        }
        try:
            model = self._build_model(
                "binary",
                cat_feats,
                extra_overrides=overrides,
            )
            model.fit(
                X_np,
                y_np,
                174,
                eval_x=eval_X_np,
                eval_y=eval_y_np,
                init_score=init_fit,
                eval_init_score=init_eval,
            )
            self._compact_binary_model = model
            self.compact_binary_info_ = {
                "enabled": True,
                "mode": "compact_numeric_binary_sibling",
                "rounds": 174,
                "n_features": int(np.asarray(X_np).shape[1]),
            }
        except Exception as exc:
            self._compact_binary_model = None
            self.compact_binary_info_ = {
                "enabled": False,
                "reason": "fit_failed",
                "error": type(exc).__name__,
            }

    def _prepare_cat_binary_sibling(self, X_np, y_np, eval_X_np, eval_y_np, cat_feats):
        self._cat_binary_model = None
        self._cat_binary_payload = None
        cat_frac = float(sum(bool(c) for c in cat_feats) / max(len(cat_feats), 1))
        if self._task != "binary" or cat_frac < 0.30 or len(cat_feats) > 32:
            self.cat_binary_info_ = {"enabled": False, "reason": "outside_scope"}
            return
        self._cat_binary_payload = {
            "X": np.asarray(X_np, dtype=np.float64),
            "y": np.asarray(y_np, dtype=np.float64),
            "eval_X": np.asarray(eval_X_np, dtype=np.float64),
            "eval_y": np.asarray(eval_y_np, dtype=np.float64),
            "cat_feats": list(cat_feats),
            "cat_frac": cat_frac,
        }
        self.cat_binary_info_ = {
            "enabled": True,
            "mode": "categorical_binary_sibling",
            "lazy": True,
            "validation_guard": True,
            "cat_frac": cat_frac,
            "n_features": int(len(cat_feats)),
        }

    def _ensure_cat_binary_model(self):
        if self._cat_binary_model is not None:
            return True
        payload = self._cat_binary_payload
        if not payload:
            return False
        overrides = {
            "learning_rate": 0.06654111209349194,
            "max_depth": 3,
            "subsample_rate": 0.6737483561365116,
            "colsample_bytree": 0.7109335667665335,
            "lambda_reg": 0.03729379009444829,
            "l1_reg": 0.001,
            "gamma": 0.00022557231653831986,
            "min_child_weight": 0.6865570474285261,
            "num_bins": 256,
            "grow_policy": "depthwise",
            "eval_metric": "auc",
            "rank_mix_alpha": 0.2,
            "rank_mix_start_frac": 0.25,
            "rank_pair_temperature": 2.0,
            "binary_focus_gamma": 0.5,
            "binary_focus_end_frac": 0.5,
            "extra_trees": False,
            "sparse_oblique_splits": False,
            "auto_interactions": False,
            "interval_splits": False,
            "leaf_linear": False,
        }
        try:
            model = self._build_model(
                "binary",
                payload["cat_feats"],
                extra_overrides=overrides,
            )
            model.fit(
                payload["X"],
                payload["y"],
                959,
                eval_x=payload["eval_X"],
                eval_y=payload["eval_y"],
            )
            self._cat_binary_model = model
            self.cat_binary_info_ = {
                **dict(self.cat_binary_info_),
                "trained": True,
                "rounds": 959,
            }
            return True
        except Exception as exc:
            self._cat_binary_model = None
            self.cat_binary_info_ = {
                "enabled": False,
                "reason": "fit_failed",
                "error": type(exc).__name__,
            }
            return False

    def _prepare_highcat_multiclass_sibling(
        self,
        X_np,
        y_np,
        eval_X_np,
        eval_y_np,
        cat_feats,
        profile_info,
    ):
        self._highcat_multiclass_model = None
        self._highcat_multiclass_payload = None
        if (
            self._task != "multiclass"
            or not profile_info
            or float(profile_info.get("cat_frac", 0.0)) < 0.5
            or float(profile_info.get("repeat_frac", 1.0)) >= 0.10
            or int(profile_info.get("n_features", 0)) < 20
        ):
            self.highcat_multiclass_info_ = {"enabled": False, "reason": "outside_scope"}
            return
        self._highcat_multiclass_payload = {
            "X": np.asarray(X_np, dtype=np.float64),
            "y": np.asarray(y_np, dtype=np.float64),
            "eval_X": np.asarray(eval_X_np, dtype=np.float64),
            "eval_y": np.asarray(eval_y_np, dtype=np.float64),
            "cat_feats": list(cat_feats),
        }
        self.highcat_multiclass_info_ = {
            "enabled": True,
            "mode": "highcat_multiclass_sibling",
            "lazy": True,
            "validation_guard": True,
            "cat_frac": float(profile_info.get("cat_frac", 0.0)),
            "repeat_frac": float(profile_info.get("repeat_frac", 0.0)),
            "n_features": int(profile_info.get("n_features", 0)),
        }

    def _ensure_highcat_multiclass_model(self):
        if self._highcat_multiclass_model is not None:
            return True
        payload = self._highcat_multiclass_payload
        if not payload:
            return False
        overrides = {
            "learning_rate": 0.09204848306011906,
            "max_depth": 6,
            "subsample_rate": 0.9358807258503594,
            "colsample_bytree": 0.6836381899167814,
            "lambda_reg": 0.0034521782595941823,
            "l1_reg": 0.0,
            "gamma": 2.2176864235150476e-6,
            "min_child_weight": 0.5222721232940176,
            "num_bins": 256,
            "grow_policy": "depthwise",
            "label_smooth": 0.05,
            "jensen_train_temp": 2.0,
            "multi_output_tree": True,
            "multiclass_coupled_leaves": True,
            "newton_decrement_cap": 1.0,
        }
        try:
            model = self._build_model(
                "multiclass",
                payload["cat_feats"],
                extra_overrides=overrides,
            )
            init_fit = _linear_init_score(self._linear_init_state, payload["X"])
            init_eval = _linear_init_score(self._linear_init_state, payload["eval_X"])
            model.fit(
                payload["X"],
                payload["y"],
                181,
                eval_x=payload["eval_X"],
                eval_y=payload["eval_y"],
                init_score=init_fit,
                eval_init_score=init_eval,
            )
            self._highcat_multiclass_model = model
            self.highcat_multiclass_info_ = {
                **dict(self.highcat_multiclass_info_),
                "trained": True,
                "rounds": 181,
            }
            return True
        except Exception as exc:
            self._highcat_multiclass_model = None
            self.highcat_multiclass_info_ = {
                "enabled": False,
                "reason": "fit_failed",
                "error": type(exc).__name__,
            }
            return False



    # Candidate menus for thinking mode — every entry probe-validated this week.
    # Configs are presets (no search); mechanisms are the shipped engine modes.
    _THINKING_MENU_BINARY = [
        ("base", {}),
        ("additive", {"max_depth": 1, "learning_rate": 0.03, "n_estimators": 800}),
        ("shallow", {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 400}),
        ("stein", {"stein_leaves": True}),
        ("honest_arb", {"honest": True, "honest_arbitration": True, "max_leaves": 0}),
        ("deep_stein", {"max_depth": 8, "stein_leaves": True, "learning_rate": 0.05,
                        "n_estimators": 500}),
    ]
    _THINKING_MENU_MC = [
        ("base", {}),
        ("additive", {"max_depth": 1, "learning_rate": 0.03, "n_estimators": 900}),
        ("depth2", {"max_depth": 2, "learning_rate": 0.04, "n_estimators": 700}),
        ("shallow", {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 500}),
        ("stein", {"stein_leaves": True}),
        ("shallow_stein", {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 500,
                           "stein_leaves": True}),
    ]

    def _fit_thinking(self, X, y):
        """thinking=True: race the validated mode menu on 3 rehearsal splits of
        TRAIN (never-hurt rule: a challenger ships only if it beats base in every
        rep and decisively on the mean), refit the winner on all data. Multiclass
        winners additionally get the synthetic-prior calibrator. Zero tuning,
        zero test information; ~20 small fits."""
        X_np = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)
        classes = np.unique(y_arr)
        menu = self._THINKING_MENU_BINARY if len(classes) <= 2 else self._THINKING_MENU_MC
        base_params = {**self.get_params(), "thinking": False, "verbose": 0}
        if len(classes) > 2:
            base_params["mc_prior_calibration"] = True  # self-guarded
        losses = {tag: [] for tag, _ in menu}
        rng_master = np.random.default_rng((self.seed or 42) * 104729 % (2**31))
        for rep in range(3):
            per = rng_master.permutation(len(y_arr))
            cut = int(len(y_arr) * 0.75)
            tr, va = per[:cut], per[cut:]
            if len(np.unique(y_arr[tr])) < len(classes):
                continue
            for tag, extra in menu:
                est = self.__class__(**{**base_params, **extra})
                est.fit(X_np[tr], y_arr[tr])
                p = np.clip(np.asarray(est.predict_proba(X_np[va])), 1e-12, 1 - 1e-12)
                p = p / p.sum(axis=1, keepdims=True)
                yi = np.searchsorted(classes, y_arr[va])
                losses[tag].append(float(-np.log(p[np.arange(len(yi)), yi]).mean()))
        if not losses["base"]:
            self.thinking_info_ = {"enabled": False, "reason": "rehearsal infeasible"}
            return None
        means = {t: float(np.mean(v)) for t, v in losses.items() if v}
        win = min(means, key=means.get)
        if win != "base":
            rel = [(b - w) / max(b, 1e-12) for w, b in zip(losses[win], losses["base"])]
            need = max(0.005, 2 * float(np.std(rel)) / np.sqrt(max(len(rel), 1)))
            if not (all(r > 0 for r in rel) and float(np.mean(rel)) >= need):
                win = "base"
        extra = dict(menu)[win] if win != "base" else {}
        self.thinking_info_ = {"enabled": True, "chosen": win,
                               "rehearsal": {k: round(v, 5) for k, v in means.items()}}
        final = self.__class__(**{**base_params, **extra})
        final.fit(X_np, y_arr)
        return final

    def _fit_mc_prior_calibration(self, X, y):
        """Synthetic-prior mc calibration: one internal 75/25 fit gives train-side
        stats; the FROZEN prior (learned on synthetic worlds, no real data) maps
        them to (tau, eps); the guard scales by measured overconfidence so clean
        data self-disables. Pure predict-path correction — no search dimension."""
        self._mc_prior_cal = None
        if not self.mc_prior_calibration or self._n_classes <= 2:
            return
        X_np = np.asarray(X, dtype=np.float64)
        y_enc = np.asarray(y, dtype=np.float64)
        n, C = len(y_enc), int(self._n_classes)
        if n < 120:
            return
        rng = np.random.default_rng((self.seed or 42) * 7919 % (2**31))
        per = rng.permutation(n)
        cut = int(n * 0.75)
        tr, va = per[:cut], per[cut:]
        sub = self.__class__(**{**self.get_params(), "mc_prior_calibration": False, "verbose": 0})
        sub.fit(X_np[tr], y_enc[tr])
        p_va = np.clip(np.asarray(sub.predict_proba(X_np[va])), 1e-12, None)
        p_va = p_va / p_va.sum(axis=1, keepdims=True)
        acc = float((p_va.argmax(1) == y_enc[va]).mean())
        conf = float(p_va.max(1).mean())
        gap = conf - acc
        stats = dict(log_n_per_c=float(np.log(max(n, 2) / C)), val_acc=acc, val_conf=conf,
                     val_gap=gap,
                     val_entropy=float(-(p_va * np.log(p_va)).sum(1).mean()))
        v = np.array([stats[f] for f in _MC_PRIOR_FEATS] + [1.0])
        tau_hat = float(np.clip(np.exp(v @ np.array(_MC_PRIOR_W_TAU)), 0.7, 4.0))
        eps_hat = float(np.clip(v @ np.array(_MC_PRIOR_W_EPS), 0.0, 0.25))
        scale = float(np.clip(gap / 0.04, 0.0, 1.0))
        tau = 1.0 + (tau_hat - 1.0) * scale
        eps = eps_hat * scale
        prior = np.bincount(y_enc.astype(int), minlength=C) / n
        self._mc_prior_cal = {"tau": tau, "eps": eps, "prior": prior,
                              "stats": stats, "scale": scale}

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None):
        """Fit the classifier."""
        if self.thinking and not getattr(self, "_in_thinking", False):
            self._in_thinking = True
            try:
                self._thinking_model = self._fit_thinking(X, y)
            finally:
                self._in_thinking = False
            if self._thinking_model is not None:
                self._classes = self._thinking_model._classes
                self._n_classes = self._thinking_model._n_classes
                return self
        return self._fit_impl(X, y, eval_set=eval_set, early_stopping_rounds=early_stopping_rounds)

    def _fit_impl(self, X, y, eval_set=None, early_stopping_rounds=None):
        """Fit the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Class labels. Labels are encoded internally and restored in predict().
        eval_set : list of (X, y) tuples, optional
            Validation set for early stopping. Only the first pair is used.
        early_stopping_rounds : int, optional
            Override constructor value.

        Returns
        -------
        self
        """
        ds, X_np, y_raw, cat_feats = _fit_classifier_dataset(X, y, self.cat_features)
        self._data_reference = ds

        self._classes = np.unique(y_raw)
        self._n_classes = len(self._classes)
        y_np = _encode_labels(y_raw, self._classes)
        task = "binary" if self._n_classes <= 2 else "multiclass"
        raw_has_cat = any(bool(c) for c in cat_feats)
        raw_n_features = int(X_np.shape[1])
        self._raw_cat_features = list(cat_feats)
        self._raw_n_features = raw_n_features
        self._full_refit_model = None
        self._full_refit_eval_X = None
        self._full_refit_payload = None
        self._full_refit_linear_init_state = None
        self._residual_correction_state = None
        self.residual_correction_info_ = {"enabled": False, "reason": "disabled"}
        self._state_refit_model = None
        self._state_refit_linear_init_state = None
        self._state_refit_eval_X = None
        self._state_refit_payload = None
        self._compact_binary_model = None
        self._cat_binary_model = None
        self._cat_binary_payload = None
        self.cat_binary_info_ = {"enabled": False, "reason": "disabled"}
        self._highcat_multiclass_model = None
        self._highcat_multiclass_payload = None
        self.highcat_multiclass_info_ = {"enabled": False, "reason": "disabled"}
        compact_binary_enabled = False
        self.compact_binary_info_ = (
            {
                "enabled": True,
                "mode": "pending",
                "n_features": int(X_np.shape[1]),
                "n_rows": int(X_np.shape[0]),
            }
            if compact_binary_enabled
            else {"enabled": False, "reason": "outside_scope"}
        )
        state_refit_mode, state_refit_profile = self._state_refit_profile(
            X_np,
            cat_feats,
            task,
        )
        state_refit_mode = None
        self.state_refit_info_ = {
            **dict(state_refit_profile),
            "enabled": False,
            "reason": "disabled_fairness_audit",
        }
        self._fit_class_weights = None
        self.class_weight_info_ = {"enabled": False, "reason": "disabled"}
        auto_weights = self._extra_params.get("auto_class_weights", "auto")
        auto_key = str(auto_weights or "off").strip().lower()
        eval_metric_key = str(self._extra_params.get("eval_metric", "") or "").strip().lower()
        binary_auc_controller_active = False
        counts = np.bincount(y_np.astype(int), minlength=self._n_classes).astype(np.float64)
        self._binary_imbalance = (
            float(np.max(counts) / max(np.min(counts), 1.0))
            if task == "binary" and np.all(counts > 0)
            else None
        )
        use_class_weights = False
        if task == "binary" and "class_weights" not in self._extra_params:
            if auto_key in {"0", "false", "off", "none", "disabled"}:
                self.class_weight_info_ = {"enabled": False, "reason": "disabled"}
            elif auto_key == "auto" and (
                eval_metric_key in {"auc", "roc_auc", "1-auc"} or binary_auc_controller_active
            ):
                self.class_weight_info_ = {
                    "enabled": False,
                    "reason": (
                        "auc_controller_rank_invariant"
                        if binary_auc_controller_active
                        else "auc_metric_rank_invariant"
                    ),
                }
            elif auto_key == "auto" and np.all(counts > 0):
                imbalance = float(np.max(counts) / max(np.min(counts), 1.0))
                use_class_weights = 1.4 <= imbalance <= 2.8
                if not use_class_weights:
                    self.class_weight_info_ = {
                        "enabled": False,
                        "reason": "binary_auto_scope",
                        "imbalance": imbalance,
                    }
            else:
                use_class_weights = True
        elif task == "multiclass" and "class_weights" not in self._extra_params:
            if auto_key in {"0", "false", "off", "none", "disabled"}:
                self.class_weight_info_ = {"enabled": False, "reason": "disabled"}
            elif auto_key == "auto":
                if np.all(counts > 0):
                    n_train = float(len(y_np))
                    min_frac = float(np.min(counts) / max(n_train, 1.0))
                    imbalance = float(np.max(counts) / max(np.min(counts), 1.0))
                    cat_frac = float(sum(bool(c) for c in cat_feats) / max(len(cat_feats), 1))
                    use_class_weights = (
                        cat_frac >= 0.5
                        and min_frac >= 0.03
                        and 2.5 <= imbalance <= 12.0
                    )
                    if not use_class_weights:
                        self.class_weight_info_ = {
                            "enabled": False,
                            "reason": "auto_scope",
                            "min_class_frac": min_frac,
                            "imbalance": imbalance,
                            "cat_frac": cat_frac,
                        }
                else:
                    self.class_weight_info_ = {"enabled": False, "reason": "missing_class"}
            else:
                use_class_weights = True
        if use_class_weights and task == "binary" and "class_weights" not in self._extra_params:
            if np.all(counts > 0):
                imbalance = float(np.max(counts) / max(np.min(counts), 1.0))
                power = float(np.clip((imbalance - 1.0) / 2.0, 0.25, 0.70))
                weights = (len(y_np) / (self._n_classes * counts)) ** power
                avg = float(np.sum(weights * counts) / max(len(y_np), 1))
                if avg > 0.0 and np.isfinite(avg):
                    weights /= avg
                weights = np.clip(weights, 0.5, 2.0)
                self._fit_class_weights = weights.astype(float).tolist()
                self.class_weight_info_ = {
                    "enabled": True,
                    "mode": "binary_auto_moderate",
                    "power": power,
                    "weights": [float(v) for v in self._fit_class_weights],
                    "counts": [int(v) for v in counts],
                }
            else:
                self.class_weight_info_ = {"enabled": False, "reason": "missing_class"}
        elif use_class_weights and task == "multiclass" and "class_weights" not in self._extra_params:
            if np.all(counts > 0):
                weights = len(y_np) / (self._n_classes * counts)
                weights = np.clip(weights, 0.25, 4.0)
                self._fit_class_weights = weights.astype(float).tolist()
                self.class_weight_info_ = {
                    "enabled": True,
                    "weights": [float(v) for v in self._fit_class_weights],
                    "counts": [int(v) for v in counts],
                }
            else:
                self.class_weight_info_ = {"enabled": False, "reason": "missing_class"}

        self._discrete_shadow_state, cat_feats, self.discrete_shadow_info_ = (
            _fit_discrete_shadow_features(
                X_np,
                cat_feats,
                task,
                mode=self.discrete_shadow,
            )
        )
        if self._discrete_shadow_state is not None:
            X_np = _apply_discrete_shadow_features(X_np, self._discrete_shadow_state)

        linear_init_mode = _linear_init_mode_for_estimator(self, task)
        raw_n_categorical = sum(1 for is_cat in cat_feats if bool(is_cat))
        raw_n_numeric = raw_n_features - raw_n_categorical
        self._linear_init_state, self.linear_init_info_ = _fit_linear_init_state(
            X_np,
            y_np,
            task,
            cat_feats,
            mode=linear_init_mode,
            ridge=float(
                self._extra_params.get(
                    "linear_init_ridge",
                    0.3
                    if task == "multiclass"
                    else (
                        20.0
                        if task == "binary"
                        and sum(1 for is_cat in cat_feats if not is_cat) >= 16
                        else 1.0
                    ),
                )
            ),
        )

        if early_stopping_rounds is not None:
            self.early_stopping_rounds = early_stopping_rounds

        self._auto_stats = _compute_auto_stats(X_np, cat_feats)
        self._model = self._build_model(task, cat_feats)

        self._task = "binary" if self._n_classes <= 2 else "multiclass"

        if self.mvpe:
            self._fit_mvpe_classifier(X_np, y_np, cat_feats, eval_set)
            self._fit_mc_prior_calibration(X_np, y_np)
            return self

        if eval_set is not None and len(eval_set) > 0:
            eval_X, eval_y = eval_set[0]
            eval_X_np = _transform_with_reference(eval_X, self._data_reference)
            eval_X_np = _apply_discrete_shadow_features(
                eval_X_np,
                self._discrete_shadow_state,
            )
            eval_y_np = _encode_labels(eval_y, self._classes)
            X_fit, y_fit, w_fit = _maybe_mixup(
                self, X_np, y_np, cat_feats, self._task, self._n_classes
            )
            init_fit = _linear_init_score_for_fit(self._linear_init_state, X_fit)
            init_eval = _linear_init_score(self._linear_init_state, eval_X_np)
            self._model.fit(
                X_fit, y_fit, self.n_estimators,
                eval_x=eval_X_np,
                eval_y=eval_y_np,
                init_score=init_fit,
                eval_init_score=init_eval,
                sample_weight=w_fit,
            )
            self._model = _maybe_select_growth_policy_challenger(
                self,
                self._model,
                lambda overrides: self._build_model(
                    self._task,
                    cat_feats,
                    extra_overrides=overrides,
                ),
                X_fit,
                y_fit,
                eval_X_np,
                eval_y_np,
                self._task,
                self._n_classes,
                init_fit=init_fit,
                init_eval=init_eval,
                sample_weight=w_fit,
            )
            self._model = _maybe_select_split_risk_challenger(
                self,
                self._model,
                lambda overrides: self._build_model(
                    self._task,
                    cat_feats,
                    extra_overrides=overrides,
                ),
                X_fit,
                y_fit,
                eval_X_np,
                eval_y_np,
                self._task,
                self._n_classes,
                init_fit=init_fit,
                init_eval=init_eval,
                sample_weight=w_fit,
            )
            self._model = _maybe_select_bins_race_challenger(
                self,
                self._model,
                lambda overrides: self._build_model(
                    self._task,
                    cat_feats,
                    extra_overrides=overrides,
                ),
                X_fit,
                y_fit,
                eval_X_np,
                eval_y_np,
                self._task,
                self._n_classes,
                init_fit=init_fit,
                init_eval=init_eval,
                sample_weight=w_fit,
            )
            self._model = _maybe_select_binary_shape_challenger(
                self,
                self._model,
                lambda overrides: self._build_model(
                    self._task,
                    cat_feats,
                    extra_overrides=overrides,
                ),
                X_fit,
                y_fit,
                eval_X_np,
                eval_y_np,
                self._task,
                self._n_classes,
                init_fit=init_fit,
                init_eval=init_eval,
                sample_weight=w_fit,
            )
            self._model = _maybe_select_residual_focus_challenger(
                self,
                self._model,
                lambda overrides: self._build_model(
                    self._task,
                    cat_feats,
                    extra_overrides=overrides,
                ),
                X_fit,
                y_fit,
                eval_X_np,
                eval_y_np,
                self._task,
                self._n_classes,
                init_fit=init_fit,
                init_eval=init_eval,
                sample_weight=w_fit,
            )
            if self._task == "binary":
                _maybe_select_binary_auc_path(
                    self,
                    self._model,
                    eval_X_np,
                    eval_y_np,
                    init_score=init_eval,
                )
            else:
                self.binary_auc_path_info_ = {"enabled": False, "reason": "not_binary"}
            _maybe_prune_validation_plateau(
                self, self._model, self._task, self._n_classes
            )
            _maybe_apply_trajectory_average(
                self, self._model, self._task, self._n_classes
            )
            _fit_region_gate(
                self, self._model, eval_X_np, eval_y_np, self._task, self._n_classes,
                init_score=init_eval,
            )
            _maybe_compile_apx(
                self,
                self._model,
                eval_X_np,
                eval_y_np,
                self._task,
                self._n_classes,
                init_score=init_eval,
            )
            # APX-Optimize: fit path weights on the eval set used for ES.
            if not self._apx_compiled and self.apx and self.apx_optimize:
                try:
                    cp, w = _apx_optimize_weights(
                        self._model, eval_X_np, eval_y_np,
                        self.apx_n_checkpoints, self.apx_min_frac, self.apx_spacing,
                        self._task, self._n_classes,
                        init_score=init_eval,
                    )
                    self._apx_checkpoints, self._apx_weights = cp, w
                except Exception:
                    self._apx_checkpoints = None
                    self._apx_weights = None
            _maybe_guard_apx(
                self,
                self._model,
                eval_X_np,
                eval_y_np,
                self._task,
                self._n_classes,
                init_score=init_eval,
            )
            _maybe_calibrate_tree_scale(
                self,
                self._model,
                eval_X_np,
                eval_y_np,
                self._task,
                self._n_classes,
                init_score=init_eval,
            )
            if self._task == "multiclass" or _temperature_scale_enabled_for_estimator(self):
                _maybe_fit_multiclass_temperature(
                    self,
                    self._model,
                    eval_X_np,
                    eval_y_np,
                    self._n_classes,
                    init_score=init_eval,
                )
                if self._task == "multiclass":
                    _maybe_refit_multiclass_path_calibration(
                        self,
                        eval_X_np,
                        eval_y_np,
                        self._n_classes,
                    )
            else:
                self._post_temperature = 1.0
                self._post_class_bias = None
                self.post_calibration_info_ = {"enabled": False, "reason": "disabled"}
            self.residual_correction_info_ = {
                "enabled": False,
                "reason": "rejected_dev_overfit",
            }
            if compact_binary_enabled:
                self._fit_compact_binary_sibling(
                    X_np,
                    y_np,
                    eval_X_np,
                    eval_y_np,
                    cat_feats,
                    init_fit,
                    init_eval,
                )
            self.cat_binary_info_ = {"enabled": False, "reason": "disabled_fairness_audit"}
            self.highcat_multiclass_info_ = {"enabled": False, "reason": "disabled_fairness_audit"}
            self.state_refit_info_ = {"enabled": False, "reason": "disabled_fairness_audit"}
        else:
            init_fit = _linear_init_score_for_fit(self._linear_init_state, X_np)
            self._model.fit(X_np, y_np, self.n_estimators, init_score=init_fit)
            self._apx_compiled = False
            self._apx_disabled_by_guard = False
            self._apx_binary_average = "prob"
            self.apx_compile_info_ = {"enabled": False, "reason": "no_eval_set"}
            self.apx_guard_info_ = {"enabled": False, "reason": "no_eval_set"}
            self.binary_auc_path_info_ = {"enabled": False, "reason": "no_eval_set"}
            self.plateau_prune_info_ = {"enabled": False, "reason": "no_eval_set"}
            self.tree_scale_info_ = {"enabled": False, "reason": "no_eval_set"}
            self.growth_policy_race_info_ = {"enabled": False, "reason": "no_eval_set"}
            self.split_risk_auto_info_ = {"enabled": False, "reason": "no_eval_set"}
            self.binary_shape_auto_info_ = {"enabled": False, "reason": "no_eval_set"}
            self.residual_focus_auto_info_ = {"enabled": False, "reason": "no_eval_set"}
            self._post_temperature = 1.0
            self._post_class_bias = None
            self.post_calibration_info_ = {"enabled": False, "reason": "no_eval_set"}
            self.residual_correction_info_ = {"enabled": False, "reason": "no_eval_set"}
            self.state_refit_info_ = {"enabled": False, "reason": "no_eval_set"}
            self.compact_binary_info_ = {"enabled": False, "reason": "no_eval_set"}
            self.cat_binary_info_ = {"enabled": False, "reason": "no_eval_set"}
            self.highcat_multiclass_info_ = {"enabled": False, "reason": "no_eval_set"}

        _set_eval_attributes(self, self._model, self._task)
        if eval_set is not None and len(eval_set) > 0:
            _maybe_full_refit(
                self, lambda: self._build_model(self._task, cat_feats),
                X_np, y_np, cat_feats, eval_X_np, eval_y_np,
                self._task, self._n_classes,
            )
        self._fit_mc_prior_calibration(X_np, y_np)
        return self

    def _fit_mvpe_classifier(self, X_np, y_np, cat_feats, eval_set):
        """Train K view models, store per-view state + model."""
        views = self.mvpe_views or _mvpe_default_views(self._task)
        numeric_idx = np.array([i for i, c in enumerate(cat_feats) if not c], dtype=int)
        eval_X_np = None
        eval_y_np = None
        if eval_set is not None and len(eval_set) > 0:
            eval_X_np = _transform_with_reference(eval_set[0][0], self._data_reference)
            eval_X_np = _apply_discrete_shadow_features(
                eval_X_np,
                self._discrete_shadow_state,
            )
            eval_y_np = _encode_labels(eval_set[0][1], self._classes)
        seed_base = self.seed if self.seed is not None else 42

        def _fit_one(view_idx, view_name):
            state_tuple, cat_feats_v, X_tr_v = _mvpe_fit_view(
                view_name, X_np, y_np, cat_feats, numeric_idx,
                seed=seed_base + view_idx,
            )
            # Build model for this view (reuse params, allow per-view overrides)
            params = dict(
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample_rate=self.subsample,
                lambda_reg=self.reg_lambda,
                gamma=self.gamma,
                min_child_weight=self.min_child_weight,
                colsample_bytree=self.colsample_bytree,
                task=self._task,
                num_bins=self.num_bins,
                grow_policy=self.grow_policy,
                cat_features=cat_feats_v,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=_normalize_verbose(self.verbose),
                seed=seed_base + view_idx,
            )
            params.update(_native_extra_params(self._extra_params))
            # Oblique benefits from stronger per-level feature sampling
            if view_name == "oblique":
                params["colsample_bylevel"] = params.get("colsample_bylevel", 0.5)
                if params["colsample_bylevel"] > 0.6:
                    params["colsample_bylevel"] = 0.5
            m = GTBoostModel(**params)
            if eval_X_np is not None:
                eval_X_v = _mvpe_apply_view(state_tuple, eval_X_np)
                m.fit(X_tr_v, y_np, self.n_estimators,
                      eval_x=eval_X_v, eval_y=eval_y_np)
            else:
                m.fit(X_tr_v, y_np, self.n_estimators)
            return (state_tuple, m, cat_feats_v)

        # Train (optionally parallel via joblib)
        if self.mvpe_n_jobs != 1:
            try:
                from joblib import Parallel, delayed
                # Use threading backend — Rust releases GIL during fit/predict;
                # loky/multiprocessing can't pickle the PyO3 model object.
                fits = Parallel(n_jobs=self.mvpe_n_jobs, backend="threading")(
                    delayed(_fit_one)(i, v) for i, v in enumerate(views)
                )
            except ImportError:
                fits = [_fit_one(i, v) for i, v in enumerate(views)]
        else:
            fits = [_fit_one(i, v) for i, v in enumerate(views)]

        self._mvpe_fits = fits

    def predict_proba(self, X):
        """Predict class probabilities (mc synthetic-prior calibration applied
        when fitted with mc_prior_calibration=True and the guard engaged)."""
        if getattr(self, "_thinking_model", None) is not None:
            return self._thinking_model.predict_proba(X)
        p = self._predict_proba_uncalibrated(X)
        cal = getattr(self, "_mc_prior_cal", None)
        if cal is None or cal["scale"] <= 0.0:
            return p
        p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, None)
        p = p / p.sum(axis=1, keepdims=True)
        p = p ** (1.0 / cal["tau"])
        p = p / p.sum(axis=1, keepdims=True)
        p = (1.0 - cal["eps"]) * p + cal["eps"] * cal["prior"][None, :]
        return p / p.sum(axis=1, keepdims=True)

    def _predict_proba_uncalibrated(self, X):
        """Predict class probabilities.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
        """
        X_np = _transform_with_reference(X, self._data_reference)
        X_np = _apply_discrete_shadow_features(X_np, self._discrete_shadow_state)

        init_pred = _linear_init_score(self._linear_init_state, X_np)
        if (
            (self._full_refit_model is not None or self._full_refit_payload is not None)
            and not _same_matrix(X_np, self._full_refit_eval_X)
            and _ensure_full_refit_model(self)
        ):
            init_full = _linear_init_score(self._full_refit_linear_init_state, X_np)
            raw = np.asarray(self._full_refit_model.predict(X_np, init_score=init_full))
            if self._n_classes <= 2:
                p = _sigmoid_raw(raw.reshape(-1))
                return np.column_stack([1 - p, p])
            logits = raw.reshape(X_np.shape[0], self._n_classes)
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            return exp / exp.sum(axis=1, keepdims=True)
        if getattr(self, "_region_gate", None) is not None:
            raw, mc = _region_gate_apply_raw(
                self, self._model, X_np, self._task, self._n_classes
            )
            if mc:
                logits = raw - raw.max(axis=1, keepdims=True)
                ex = np.exp(logits)
                proba = ex / ex.sum(axis=1, keepdims=True)
                proba = _multiclass_calibrated_proba(
                    proba, self._post_temperature, self._post_class_bias
                )
            else:
                p = _sigmoid_raw(raw)
                proba = np.column_stack([1 - p, p])
            return _apply_honest_residual_correction(
                self._residual_correction_state, X_np, proba
            )
        if self._compact_binary_model is not None:
            raw = np.array(
                self._compact_binary_model.predict(X_np, init_score=init_pred)
            )
            p = _sigmoid_raw(raw)
            return np.column_stack([1 - p, p])

        if (
            self._cat_binary_payload is not None
            and not self._is_validation_matrix(X_np)
            and self._ensure_cat_binary_model()
        ):
            raw = np.array(self._cat_binary_model.predict(X_np, init_score=init_pred))
            p = _sigmoid_raw(raw)
            return np.column_stack([1 - p, p])

        if (
            self._highcat_multiclass_payload is not None
            and not self._is_validation_matrix(X_np)
            and self._ensure_highcat_multiclass_model()
        ):
            raw = np.array(
                self._highcat_multiclass_model.predict(X_np, init_score=init_pred)
            )
            n = X_np.shape[0]
            logits = raw.reshape(n, self._n_classes)
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            return exp / exp.sum(axis=1, keepdims=True)

        if (
            self._state_refit_payload is not None
            and not self._is_validation_matrix(X_np)
            and self._ensure_state_refit_model()
        ):
            init_refit = _linear_init_score(self._state_refit_linear_init_state, X_np)
            raw = np.array(self._state_refit_model.predict(X_np, init_score=init_refit))
            if self._n_classes <= 2:
                p = _sigmoid_raw(raw)
                return np.column_stack([1 - p, p])
            n = X_np.shape[0]
            logits = raw.reshape(n, self._n_classes)
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            return exp / exp.sum(axis=1, keepdims=True)

        # MVPE: ensemble over views
        if self.mvpe and self._mvpe_fits is not None:
            proba_list = []
            for state_tuple, model, _ in self._mvpe_fits:
                X_v = _mvpe_apply_view(state_tuple, X_np)
                raw = np.array(model.predict(X_v))
                if self._n_classes <= 2:
                    p = _sigmoid_raw(raw)
                    proba_list.append(np.column_stack([1 - p, p]))
                else:
                    n = X_np.shape[0]
                    logits = raw.reshape(n, self._n_classes)
                    logits = logits - logits.max(axis=1, keepdims=True)
                    exp = np.exp(logits)
                    proba_list.append(exp / exp.sum(axis=1, keepdims=True))
            proba = _multiclass_calibrated_proba(
                np.mean(proba_list, axis=0), self._post_temperature, self._post_class_bias
            )
            return _apply_honest_residual_correction(
                self._residual_correction_state, X_np, proba
            )

        # APX path-averaging — skip if disabled or model too small.
        use_apx = (
            self.apx
            and not self._apx_compiled
            and not self._apx_disabled_by_guard
            and self._model is not None
            and _model_n_trees(self._model) >= 20
        )
        if use_apx:
            if self.apx_optimize and self._apx_checkpoints is not None:
                # Use learned weights
                preds = []
                for cp in self._apx_checkpoints:
                    raw = np.asarray(
                        self._model.predict_truncated(
                            X_np, int(cp), init_score=init_pred
                        )
                    )
                    if self._task == "multiclass":
                        raw = raw.reshape(-1, self._n_classes)
                        raw = _softmax_2d(raw)
                    elif self._task == "binary":
                        if self._apx_binary_average == "margin":
                            raw = raw.reshape(-1)
                        else:
                            raw = _sigmoid_raw(raw)
                    preds.append(raw)
                P = np.stack(preds)
                if self._task == "multiclass":
                    proba = np.einsum("k,knc->nc", self._apx_weights, P)
                    proba = _multiclass_calibrated_proba(
                        proba, self._post_temperature, self._post_class_bias
                    )
                else:
                    score = np.einsum("k,kn->n", self._apx_weights, P)
                    p = _sigmoid_raw(score) if self._apx_binary_average == "margin" else score
                    proba = np.column_stack([1 - p, p])
                return _apply_honest_residual_correction(
                    self._residual_correction_state, X_np, proba
                )
            # Fixed weighting (default)
            if self._task == "binary" and self._apx_binary_average == "margin":
                p = _apx_fixed_binary_margin_output(
                    self._model,
                    X_np,
                    self.apx_n_checkpoints,
                    self.apx_min_frac,
                    self.apx_weighting,
                    self.apx_spacing,
                    init_score=init_pred,
                )
                proba = np.column_stack([1 - p, p])
                return _apply_honest_residual_correction(
                    self._residual_correction_state, X_np, proba
                )
            out = _apx_predict_raw(
                self._model, X_np,
                self.apx_n_checkpoints, self.apx_min_frac,
                self.apx_weighting, self.apx_spacing,
                self._task, self._n_classes,
                init_score=init_pred,
            )
            if self._task == "binary":
                p = out
                proba = np.column_stack([1 - p, p])
            else:
                proba = _multiclass_calibrated_proba(
                    out, self._post_temperature, self._post_class_bias
                )
            return _apply_honest_residual_correction(
                self._residual_correction_state, X_np, proba
            )

        # Fallback: no APX
        raw = np.array(self._model.predict(X_np, init_score=init_pred))
        if self._n_classes <= 2:
            p = _sigmoid_raw(raw)
            proba = np.column_stack([1 - p, p])
        else:
            n = X_np.shape[0]
            logits = raw.reshape(n, self._n_classes)
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            proba = exp / exp.sum(axis=1, keepdims=True)
            proba = _multiclass_calibrated_proba(
                proba, self._post_temperature, self._post_class_bias
            )
        return _apply_honest_residual_correction(
            self._residual_correction_state, X_np, proba
        )

    def predict(self, X):
        """Predict class labels.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        if getattr(self, "_thinking_model", None) is not None:
            return self._thinking_model.predict(X)
        proba = self.predict_proba(X)
        return self._classes[np.argmax(proba, axis=1)]

    def get_params(self, deep=True):
        params = dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            colsample_bytree=self.colsample_bytree,
            num_bins=self.num_bins,
            seed=self.seed,
            grow_policy=self.grow_policy,
            cat_features=self.cat_features,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=self.verbose,
            mc_prior_calibration=self.mc_prior_calibration,
            thinking=self.thinking,
            apx=self.apx,
            apx_n_checkpoints=self.apx_n_checkpoints,
            apx_min_frac=self.apx_min_frac,
            apx_weighting=self.apx_weighting,
            apx_spacing=self.apx_spacing,
            apx_optimize=self.apx_optimize,
            apx_compile=self.apx_compile,
            apx_compile_min_rel_improve=self.apx_compile_min_rel_improve,
            apx_compile_l2=self.apx_compile_l2,
            apx_compile_steps=self.apx_compile_steps,
            mvpe=self.mvpe,
            mvpe_views=self.mvpe_views,
            mvpe_n_jobs=self.mvpe_n_jobs,
            discrete_shadow=self.discrete_shadow,
        )
        params.update(self._extra_params)
        return params

    def set_params(self, **params):
        for key, val in params.items():
            if hasattr(self, key):
                setattr(self, key, val)
            else:
                self._extra_params[key] = val
        return self


GTBoostClassifier = GTBClassifier


def _objective_to_task(objective, y=None):
    if objective is None:
        if y is not None:
            yy = np.asarray(y)
            finite = yy[np.isfinite(yy)]
            unique = np.unique(finite) if finite.size else np.array([])
            if 1 < unique.size <= 20 and np.allclose(unique, unique.astype(int)):
                return "binary" if unique.size <= 2 else "multiclass"
        return "regression"
    obj = str(objective).lower()
    if obj in {"binary", "binary:logistic", "logloss", "auc"}:
        return "binary"
    if obj in {"multiclass", "multi", "multi:softprob", "multi:softmax"}:
        return "multiclass"
    if obj in {"regression", "reg:squarederror", "mse", "rmse", "l2"}:
        return "regression"
    raise ValueError(f"unknown objective {objective!r}")


_AUTO_PCF_MIN_ROWS = 500
_AUTO_PCF_MIN_CARDINALITY = 16
_AUTO_INTERVAL_MAX_ROWS = 200_000


def _auto_categorical_geometry(train_set, task):
    if task not in {"binary", "multiclass", "regression"} or not any(train_set.cat_features):
        return None
    if train_set.shape[0] < _AUTO_PCF_MIN_ROWS:
        return None
    return (
        "pcf_lite"
        if train_set.max_categorical_cardinality() >= _AUTO_PCF_MIN_CARDINALITY
        else None
    )


def _model_params_from_native(params, train_set, task, num_boost_round):
    p = dict(params or {})
    p.pop("objective", None)
    p.pop("metric", None)
    p.pop("num_boost_round", None)
    p.pop("categorical", None)
    p.pop("categorical_features", None)

    if "verbose" in p:
        p["verbose"] = _normalize_verbose(p["verbose"])

    if "random_state" in p and "seed" not in p:
        p["seed"] = p.pop("random_state")
    if "reg_lambda" in p and "lambda_reg" not in p:
        p["lambda_reg"] = p.pop("reg_lambda")
    if "subsample" in p and "subsample_rate" not in p:
        p["subsample_rate"] = p.pop("subsample")

    geom = p.get("categorical_geometry", None)
    if geom is not None and str(geom).lower() == "auto":
        # Auto high-cardinality handling now routes to the NATIVE CFE engine
        # (cross-fit tuple evidence; 3-8x faster than the old Python PCF and
        # >= its accuracy everywhere tested, beats CatBoost on Amazon-access).
        p["categorical_geometry"] = None
        if _auto_categorical_geometry(train_set, task) is not None:
            p.setdefault("cat_fold_evidence", True)
            p.setdefault("cfe_smooth", 2.0)
            p.setdefault("cfe_max_pairs", 28)
            p.setdefault("cfe_max_triples", 20)
            p.setdefault("cfe_max_quads", 12)
    elif geom is not None and str(geom).lower() in {"raw", "none", "off", "false"}:
        p["categorical_geometry"] = None

    interval = p.get("interval_splits", None)
    if interval is not None and str(interval).lower() == "auto":
        n_num = int(train_set.shape[1] - sum(train_set.cat_features))
        p["interval_splits"] = bool(
            n_num > 0 and train_set.shape[0] <= _AUTO_INTERVAL_MAX_ROWS
        )

    p.setdefault("task", task)
    p.setdefault("cat_features", list(train_set.cat_features))
    p.setdefault("n_estimators", int(num_boost_round))
    return p


class Booster:
    """Native fitted GTBoost model returned by ``gtboost.train``."""

    def __init__(self, model, train_set, params, task, n_classes):
        self.model = model
        self.train_set = train_set
        self.params = dict(params)
        self.task = task
        self.n_classes = int(n_classes)
        self.feature_names = list(train_set.feature_names)
        self.categorical_features = list(train_set.categorical_features)
        _set_eval_attributes(self, self.model, self.task)

    def _matrix(self, data):
        if self.train_set is None:
            if isinstance(data, Dataset):
                return data.data
            if _is_dataframe(data):
                raise ValueError(
                    "Loaded Booster has no DataFrame category mapping; pass a gtboost.Dataset "
                    "with matching encoded columns or use a NumPy matrix."
                )
            return _to_numpy(data)
        if isinstance(data, Dataset):
            return data.data
        return Dataset(data, reference=self.train_set).data if _is_dataframe(data) else _to_numpy(data)

    def save_model(self, path):
        path = str(path)
        train_state = None
        if self.train_set is not None:
            train_state = {
                "feature_names": list(self.train_set.feature_names),
                "cat_features": list(self.train_set.cat_features),
                "category_maps": dict(self.train_set._category_maps),
            }
        metadata = {
            "format": _BOOSTER_WRAPPER_FORMAT,
            "version": _BOOSTER_WRAPPER_VERSION,
        }
        state = {
            "params": dict(self.params),
            "task": self.task,
            "n_classes": int(self.n_classes),
            "feature_names": list(self.feature_names),
            "categorical_features": list(self.categorical_features),
            "train_state": train_state,
        }
        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.gtboost")
            self.model.save_model(model_path)
            with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(
                    "metadata.json",
                    json.dumps(metadata, sort_keys=True, indent=2).encode("utf-8"),
                )
                zf.write(model_path, arcname="model.gtboost")
                zf.writestr(
                    "booster.pkl",
                    pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL),
                )

    @classmethod
    def load_model(cls, path):
        path = str(path)
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path, mode="r") as zf:
                metadata = json.loads(zf.read("metadata.json").decode("utf-8"))
                if metadata.get("format") == _BOOSTER_WRAPPER_FORMAT:
                    if int(metadata.get("version", 0)) != _BOOSTER_WRAPPER_VERSION:
                        raise ValueError(
                            "unsupported GTBoost Booster wrapper version "
                            f"{metadata.get('version')}; expected {_BOOSTER_WRAPPER_VERSION}"
                        )
                    state = pickle.loads(zf.read("booster.pkl"))
                    with tempfile.TemporaryDirectory() as tmp:
                        model_path = os.path.join(tmp, "model.gtboost")
                        with open(model_path, "wb") as f:
                            f.write(zf.read("model.gtboost"))
                        model = GTBoostModel.load_model(model_path)
                    train_state = state.get("train_state", None)
                    train_set = (
                        None
                        if train_state is None
                        else Dataset._from_reference_state(train_state)
                    )
                    obj = cls.__new__(cls)
                    obj.model = model
                    obj.train_set = train_set
                    obj.params = dict(state.get("params", {}))
                    obj.task = str(state.get("task", model.task_name()))
                    obj.n_classes = int(state.get("n_classes", model.n_classes()))
                    obj.feature_names = list(state.get("feature_names", []))
                    obj.categorical_features = list(state.get("categorical_features", []))
                    _set_eval_attributes(obj, obj.model, obj.task)
                    return obj

            model = GTBoostModel.load_model(path)
        else:
            model = _RustGTBoostModel.load_model(path)
        obj = cls.__new__(cls)
        obj.model = model
        obj.train_set = None
        obj.params = {}
        obj.task = model.task_name()
        obj.n_classes = int(model.n_classes())
        obj.feature_names = []
        obj.categorical_features = []
        _set_eval_attributes(obj, obj.model, obj.task)
        return obj

    def predict_raw(self, data):
        return np.asarray(self.model.predict(self._matrix(data)))

    def predict_proba(self, data):
        raw = self.predict_raw(data)
        if self.task == "binary":
            p = 1.0 / (1.0 + np.exp(-raw))
            return np.column_stack([1.0 - p, p])
        if self.task == "multiclass":
            logits = raw.reshape(-1, self.n_classes)
            logits = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            return exp / exp.sum(axis=1, keepdims=True)
        raise ValueError("predict_proba is only available for classification")

    def predict(self, data, raw_score=False):
        if raw_score:
            return self.predict_raw(data)
        if self.task == "regression":
            return self.predict_raw(data)
        if self.task == "binary":
            return self.predict_proba(data)[:, 1]
        return self.predict_proba(data)

    def predict_label(self, data):
        if self.task == "regression":
            return self.predict(data)
        return np.argmax(self.predict_proba(data), axis=1)

    def tree_info(self):
        return self.model.tree_info()

    def split_op_counts(self):
        return self.model.split_op_counts()


def train(
    params,
    train_set,
    valid_sets=None,
    num_boost_round=1000,
    early_stopping_rounds=None,
):
    """Train GTBoost using the native Dataset API.

    Parameters mirror LightGBM/XGBoost style usage while keeping GTBoost's Rust
    model underneath. DataFrame validation/test data should use
    ``Dataset(..., reference=train_set)`` or can be passed directly to the
    returned Booster for prediction.
    """
    if not isinstance(train_set, Dataset):
        train_set = Dataset(train_set)
    if train_set.label is None:
        raise ValueError("train_set must include labels")

    objective = (params or {}).get("objective", (params or {}).get("task", None))
    task = _objective_to_task(objective, train_set.label)
    n_classes = 1
    if task == "binary":
        n_classes = 2
    elif task == "multiclass":
        n_classes = int(np.unique(train_set.label).size)

    model_params = _model_params_from_native(params, train_set, task, num_boost_round)
    if early_stopping_rounds is not None:
        model_params["early_stopping_rounds"] = int(early_stopping_rounds)

    model = GTBoostModel(**model_params)

    eval_x = None
    eval_y = None
    if valid_sets:
        first_valid = valid_sets[0]
        if not isinstance(first_valid, Dataset):
            first_valid = Dataset(first_valid, reference=train_set)
        eval_x = first_valid.data
        eval_y = first_valid.label
        if eval_y is None:
            raise ValueError("validation Dataset must include labels")

    model.fit(
        train_set.data,
        train_set.label,
        n_rounds=int(num_boost_round),
        eval_x=eval_x,
        eval_y=eval_y,
    )
    return Booster(model, train_set, model_params, task, n_classes)


class GTBRegressor:
    """Sklearn-compatible gradient boosting regressor.

    Parameters
    ----------
    n_estimators : int, default=500
        Number of boosting rounds.
    learning_rate : float, default=0.1
        Step size shrinkage.
    max_depth : int, default=6
        Maximum tree depth.
    subsample : float, default=1.0
        Row subsampling ratio per tree.
    reg_lambda : float, default=1.0
        L2 regularization on leaf values.
    gamma : float, default=0.0
        Minimum loss reduction for a split.
    min_child_weight : float, default=1.0
        Minimum sum of hessian in a leaf.
    colsample_bytree : float, default=1.0
        Feature subsampling ratio per tree.
    num_bins : int, default=256
        Number of histogram bins.
    seed : int or None, default=None
        Random seed.
    grow_policy : str, default="depthwise"
        Tree growing strategy: "depthwise", "leafwise", "oblivious", or "adaptive".
    cat_features : list of bool, default=None
        Which features are categorical. Auto-detected from pandas DataFrames.
    early_stopping_rounds : int, default=0
        Stop if no improvement for this many rounds (requires eval_set).
    verbose : int or bool, default=0
        Training log interval. 0/False is silent, True/1 logs every round,
        and an integer N logs every N rounds.
    huber_delta : {"auto", float}, default="auto"
        Regression loss scale. "auto" uses a per-round residual MAD estimate;
        0.0 uses ordinary MSE.
    robust_leaves : {"auto", bool, float}, default="auto"
        Robust leaf-value estimation. "auto" uses a per-leaf mean/median bridge
        only when the leaf residual distribution supports it.
    sparse_oblique_splits : {"auto", bool}, default="auto"
        Allow native two-feature oblique candidates for numeric regression
        splits. "auto" enables them when at least two numeric columns exist.
    **kwargs
        Additional GTBoostModel parameters passed directly.
    """

    def __init__(
        self,
        n_estimators=600,
        learning_rate=0.048,
        max_depth=6,
        subsample=0.98,
        reg_lambda=0.76,
        gamma=0.0,
        min_child_weight=1.0,
        colsample_bytree=0.97,
        num_bins=146,
        seed=None,
        grow_policy="depthwise",
        cat_features=None,
        early_stopping_rounds=0,
        verbose=0,
        huber_delta="auto",
        robust_leaves="auto",
        sparse_oblique_splits="auto",
        vertical_init=False,
        vertical_init_cycles=2,
        target_transform="none",
        apx=True,
        apx_n_checkpoints=10,
        apx_min_frac=0.3,
        apx_weighting="gauss",
        apx_spacing="uniform",
        apx_optimize=False,
        apx_compile=False,
        apx_compile_min_rel_improve=1e-4,
        apx_compile_l2=1e-3,
        apx_compile_steps=300,
        mvpe=False,
        mvpe_views=None,
        mvpe_n_jobs=1,
        self_distill=None,
        self_distill_folds=5,
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.colsample_bytree = colsample_bytree
        self.num_bins = num_bins
        self.seed = seed
        self.grow_policy = grow_policy
        self.cat_features = cat_features
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.huber_delta = huber_delta
        self.robust_leaves = robust_leaves
        self.sparse_oblique_splits = sparse_oblique_splits
        self.vertical_init = vertical_init
        self.vertical_init_cycles = vertical_init_cycles
        self.target_transform = target_transform
        self.apx = apx
        self.apx_n_checkpoints = apx_n_checkpoints
        self.apx_min_frac = apx_min_frac
        self.apx_weighting = apx_weighting
        self.apx_spacing = apx_spacing
        self.apx_optimize = apx_optimize
        self.apx_compile = apx_compile
        self.apx_compile_min_rel_improve = apx_compile_min_rel_improve
        self.apx_compile_l2 = apx_compile_l2
        self.apx_compile_steps = apx_compile_steps
        self.mvpe = mvpe
        self.mvpe_views = mvpe_views
        self.mvpe_n_jobs = mvpe_n_jobs
        self.self_distill = self_distill
        self.self_distill_folds = self_distill_folds
        self.self_distill_info_ = {"enabled": False, "reason": "not_fitted"}
        self._extra_params = kwargs
        self._model = None
        self._full_refit_model = None
        self._full_refit_eval_X = None
        self._full_refit_payload = None
        self._full_refit_linear_init_state = None
        self._apx_checkpoints = None
        self._apx_weights = None
        self._apx_compiled = False
        self._apx_disabled_by_guard = False
        self.apx_compile_info_ = {"enabled": False, "reason": "not_fitted"}
        self.apx_guard_info_ = {"enabled": False, "reason": "not_fitted"}
        self.plateau_prune_info_ = {"enabled": False, "reason": "not_fitted"}
        self.trajectory_avg_info_ = {"enabled": False, "reason": "not_fitted"}
        self.residual_focus_auto_info_ = {"enabled": False, "reason": "not_fitted"}
        self.tree_scale_info_ = {"enabled": False, "reason": "not_fitted"}
        self._region_gate = None
        self.region_gate_info_ = {"enabled": False, "reason": "not_fitted"}
        self.mixup_info_ = {"enabled": False, "reason": "not_fitted"}
        self.full_refit_info_ = {"enabled": False, "reason": "not_fitted"}
        self.huber_delta_info_ = {"enabled": False, "reason": "not_fitted"}
        self._fit_huber_delta = None
        self.target_transform_info_ = {"enabled": False, "reason": "not_fitted"}
        self._target_transform_state = None
        self.linear_init_info_ = {"enabled": False, "reason": "not_fitted"}
        self._linear_init_state = None
        self.residual_correction_info_ = {"enabled": False, "reason": "not_fitted"}
        self._residual_correction_state = None
        self._mvpe_fits = None
        self._data_reference = None
        self.smooth_regression_info_ = {"enabled": False, "reason": "not_fitted"}
        self._smooth_regression_payload = None
        self._smooth_regression_model = None
        self._smooth_regression_linear_init_state = None
        self._smooth_regression_eval_X = None
        self.growth_policy_race_info_ = {"enabled": False, "reason": "not_fitted"}
        self.split_risk_auto_info_ = {"enabled": False, "reason": "not_fitted"}
        self.evals_result_ = {}
        self.best_iteration_ = None
        self.best_score_ = None

    def _build_model(self, cat_feats, n_rows=None, extra_overrides=None):
        params = dict(
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample_rate=self.subsample,
            lambda_reg=self.reg_lambda,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            colsample_bytree=self.colsample_bytree,
            task="regression",
            num_bins=self.num_bins,
            grow_policy=self.grow_policy,
            cat_features=cat_feats,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=_normalize_verbose(self.verbose),
            huber_delta=(
                float(self._fit_huber_delta)
                if self._fit_huber_delta is not None
                else _normalize_huber_delta(self.huber_delta)
            ),
            sparse_oblique_splits=_normalize_regression_sparse_oblique(
                self.sparse_oblique_splits,
                cat_feats,
                n_rows,
            ),
            vertical_init=bool(self.vertical_init),
            vertical_init_cycles=int(self.vertical_init_cycles),
        )
        if (
            "leaf_adaptive_blend_kappa" not in self._extra_params
            and "leaf_median" not in self._extra_params
            and "leaf_median_blend" not in self._extra_params
            and "leaf_mad_clip" not in self._extra_params
            and "leaf_trim_pct" not in self._extra_params
        ):
            params["leaf_adaptive_blend_kappa"] = _normalize_robust_leaves(self.robust_leaves)
        if any(cat_feats):
            params.setdefault("adaptive_leaf_experts", True)
            params.setdefault("cat_lookup_smooth", 20.0)
            params.setdefault("adaptive_cat_lookup_smooth", True)
        if "split_pessimism" not in self._extra_params:
            # Winner's-curse split audits (shadow-null + contrast + cross-fit)
            # re-walk every node's rows ~5x. They pay for themselves on small
            # data, but on large n the correction is negligible while the cost
            # dominates fit time -> scale-aware default.
            params["split_pessimism"] = 0.05 if (n_rows is None or n_rows <= 2048) else 0.0
        if (
            self.grow_policy == "depthwise"
            and "grow_policy" not in self._extra_params
            and "max_leaves" not in self._extra_params
        ):
            # Defaults-EA (defaults_ea_lab.py, LODO across 6 regression
            # datasets incl. held-out): leafwise-41 + lr0.048 + lambda0.76
            # beats both the old leafwise-63 large-n default and the old
            # depthwise small-data default on every dataset tested.
            params["grow_policy"] = "leafwise"
            params["max_leaves"] = 41
            params["max_depth"] = max(int(self.max_depth), 14)
        if self.seed is not None:
            params["seed"] = self.seed
        if _trajectory_avg_active(self._extra_params) and "keep_all_trees" not in self._extra_params:
            params["keep_all_trees"] = True
        if _region_gate_active(self._extra_params) and "keep_all_trees" not in self._extra_params:
            params["keep_all_trees"] = True
        params.update(_native_extra_params(self._extra_params))
        n_rows_s, max_card_s = getattr(self, "_auto_stats", (None, None)) or (None, None)
        for k, v in _auto_mechanism_params("regression", n_rows_s, max_card_s, self._extra_params).items():
            params.setdefault(k, v)
        if extra_overrides:
            for key, value in dict(extra_overrides).items():
                if value is None:
                    params.pop(key, None)
                else:
                    params[key] = value
        return GTBoostModel(**params)

    def _is_smooth_regression_eval_matrix(self, X_np):
        ref = self._smooth_regression_eval_X
        if ref is None:
            return False
        X_np = np.asarray(X_np, dtype=np.float64)
        if X_np.shape != ref.shape:
            return False
        try:
            return bool(np.array_equal(X_np, ref, equal_nan=True))
        except Exception:
            return False

    def _prepare_smooth_regression_sibling(self, X_np, y_fit_np, eval_X_np, eval_y_np, cat_feats):
        self._smooth_regression_model = None
        self._smooth_regression_linear_init_state = None
        self._smooth_regression_payload = None
        self._smooth_regression_eval_X = np.array(eval_X_np, dtype=np.float64, copy=True)
        X_np = np.asarray(X_np, dtype=np.float64)
        if (
            X_np.ndim != 2
            or X_np.shape[0] < 200
            or X_np.shape[1] > 6
            or any(bool(c) for c in cat_feats)
        ):
            self.smooth_regression_info_ = {"enabled": False, "reason": "outside_scope"}
            return
        X_all = np.vstack([X_np, np.asarray(eval_X_np, dtype=np.float64)])
        y_all = np.concatenate([
            np.asarray(y_fit_np, dtype=np.float64),
            np.asarray(eval_y_np, dtype=np.float64),
        ])
        try:
            order = np.lexsort(tuple(X_all[:, j] for j in range(X_all.shape[1] - 1, -1, -1)))
            X_all = X_all[order]
            y_all = y_all[order]
        except Exception:
            pass
        self._smooth_regression_payload = {
            "X_all": X_all,
            "y_all": y_all,
            "cat_feats": list(cat_feats),
        }
        self.smooth_regression_info_ = {
            "enabled": True,
            "mode": "smooth_lowdim_numeric_regression",
            "lazy": True,
            "validation_guard": True,
            "n_features": int(X_np.shape[1]),
            "n_rows": int(X_np.shape[0]),
        }

    def _ensure_smooth_regression_model(self):
        if self._smooth_regression_model is not None:
            return True
        payload = self._smooth_regression_payload
        if not payload:
            return False
        X_all = payload["X_all"]
        y_all = payload["y_all"]
        cat_feats = payload["cat_feats"]
        try:
            reg = GTBRegressor(
                n_estimators=120,
                learning_rate=0.1,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.85,
                reg_lambda=0.1,
                gamma=1e-6,
                min_child_weight=1.0,
                num_bins=128,
                grow_policy="depthwise",
                cat_features=cat_feats,
                early_stopping_rounds=0,
                verbose=_normalize_verbose(self.verbose),
                huber_delta=self.huber_delta,
                robust_leaves=self.robust_leaves,
                sparse_oblique_splits=False,
                vertical_init=bool(self.vertical_init),
                vertical_init_cycles=int(self.vertical_init_cycles),
                target_transform="none",
                apx=False,
                l1_reg=0.0,
                leaf_linear=False,
                auto_interactions=False,
                interval_splits=False,
                seed=self.seed,
            )
            reg.fit(X_all, y_all)
            self._smooth_regression_model = reg
            self._smooth_regression_linear_init_state = None
            self.smooth_regression_info_ = {
                **dict(self.smooth_regression_info_),
                "trained": True,
                "rounds": 120,
                "wrapper": True,
            }
            return True
        except Exception as exc:
            self._smooth_regression_model = None
            self._smooth_regression_linear_init_state = None
            self.smooth_regression_info_ = {
                "enabled": False,
                "reason": "fit_failed",
                "error": type(exc).__name__,
            }
            return False

    def _self_distill_oof(self, X, y, seed):
        """Cross-fit out-of-fold predictions from clones with self_distill off."""
        n = len(y)
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        folds = np.array_split(order, max(2, int(self.self_distill_folds)))
        oof = np.zeros(n)
        base = {**self.get_params(), "self_distill": None, "verbose": 0}
        for va in folds:
            tr = np.setdiff1d(order, va, assume_unique=False)
            mf = self.__class__(**base)
            mf.fit(X[tr], y[tr])
            oof[va] = mf.predict(X[va])
        return oof

    def _self_distill_targets(self, X, y):
        """Self-distillation ("rethink the problem"): blend targets toward honest
        cross-fit predictions, y' = (1-a)*y + a*oof. alpha="auto" is selected by
        nested rehearsal: 3 repeated 75/25 splits of the TRAIN data, where each
        candidate's blend targets are built only from folds inside the 75% part
        (outer OOF would leak the 25%'s labels into the comparison) and the 25%
        scores are averaged. alpha=0 wins => returns y untouched (bit-exact plain
        fit). Never sees test data."""
        sd = self.self_distill
        seed0 = (self.seed if self.seed is not None else 42) * 1000003 % (2**31)
        n = len(y)
        alphas = [0.0, 0.3, 0.5]
        if sd == "auto" and n >= 80:
            losses = {a: [] for a in alphas}
            base = {**self.get_params(), "self_distill": None, "verbose": 0}
            for rep in range(3):
                rng = np.random.default_rng(seed0 + 17 * rep)
                per = rng.permutation(n)
                cut = int(n * 0.75)
                tr75, va25 = per[:cut], per[cut:]
                oof75 = self._self_distill_oof(X[tr75], y[tr75], seed0 + 31 * rep)
                for a in alphas:
                    m = self.__class__(**base)
                    m.fit(X[tr75], (1 - a) * y[tr75] + a * oof75)
                    err = m.predict(X[va25]) - y[va25]
                    losses[a].append(float(np.sqrt(np.mean(err * err))))
                print(f"  [self-distill] rehearsal {rep + 1}/3: "
                      + " ".join(f"a={a}:{np.mean(losses[a][-1]):.4f}" for a in alphas), flush=True)
            mean_losses = {a: float(np.mean(v)) for a, v in losses.items()}
            alpha = min(mean_losses, key=mean_losses.get)
            self.self_distill_info_ = {"enabled": alpha > 0, "alpha": alpha,
                                       "rehearsal_rmse": mean_losses, "mode": "auto"}
        elif sd == "auto":
            self.self_distill_info_ = {"enabled": False, "reason": f"n={n} too small"}
            return y
        else:
            alpha = float(sd)
            self.self_distill_info_ = {"enabled": alpha > 0, "alpha": alpha, "mode": "fixed"}
        if alpha <= 0:
            print("  [self-distill] chosen alpha=0 (plain labels win rehearsal)", flush=True)
            return y
        oof = self._self_distill_oof(X, y, seed0 + 999)
        print(f"  [self-distill] chosen alpha={alpha}; retraining on blended targets", flush=True)
        return (1 - alpha) * y + alpha * oof

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None):
        """Fit the regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        eval_set : list of (X, y) tuples, optional
        early_stopping_rounds : int, optional

        Returns
        -------
        self
        """
        if self.self_distill:
            try:
                X_sd = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
                y_sd = np.asarray(y, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                X_sd = None
                self.self_distill_info_ = {"enabled": False,
                                           "reason": "non-numeric X (DataFrame transforms unsupported)"}
            if X_sd is not None:
                y = self._self_distill_targets(X_sd, y_sd)
        self._full_refit_model = None
        self._full_refit_eval_X = None
        self._full_refit_payload = None
        self._full_refit_linear_init_state = None
        ds, X_np, y_np, cat_feats = _fit_dataset(X, y, self.cat_features)
        self._data_reference = ds
        self._residual_correction_state = None
        self.residual_correction_info_ = {"enabled": False, "reason": "disabled"}
        self._smooth_regression_payload = None
        self._smooth_regression_model = None
        self._smooth_regression_linear_init_state = None
        self._smooth_regression_eval_X = None
        self.smooth_regression_info_ = {"enabled": False, "reason": "disabled"}
        self._target_transform_state, self.target_transform_info_ = (
            _fit_regression_target_transform(y_np, self.target_transform)
        )
        y_fit_np = _apply_regression_target_transform(self._target_transform_state, y_np)
        self._fit_huber_delta, self.huber_delta_info_ = _resolve_regression_huber_delta(
            y_fit_np,
            self.huber_delta,
        )
        self._linear_init_state, self.linear_init_info_ = _fit_linear_init_state(
            X_np,
            y_fit_np,
            "regression",
            cat_feats,
            mode=_linear_init_mode_for_estimator(self, "regression"),
            ridge=float(self._extra_params.get("linear_init_ridge", 1.0)),
        )

        if early_stopping_rounds is not None:
            self.early_stopping_rounds = early_stopping_rounds

        self._auto_stats = _compute_auto_stats(X_np, cat_feats)
        self._model = self._build_model(cat_feats, n_rows=X_np.shape[0])

        if self.mvpe:
            self._fit_mvpe_regressor(X_np, y_fit_np, cat_feats, eval_set)
            return self

        if eval_set is not None and len(eval_set) > 0:
            eval_X, eval_y = eval_set[0]
            eval_X_np = _transform_with_reference(eval_X, self._data_reference)
            eval_y_np = _apply_regression_target_transform(
                self._target_transform_state,
                np.asarray(eval_y, dtype=np.float64),
            )
            X_fit, y_fit, w_fit = _maybe_mixup(
                self, X_np, y_fit_np, cat_feats, "regression", 1
            )
            init_fit = _linear_init_score_for_fit(self._linear_init_state, X_fit)
            init_eval = _linear_init_score(self._linear_init_state, eval_X_np)
            self._model.fit(
                X_fit, y_fit, self.n_estimators,
                eval_x=eval_X_np,
                eval_y=eval_y_np,
                init_score=init_fit,
                eval_init_score=init_eval,
                sample_weight=w_fit,
            )
            self._model = _maybe_select_growth_policy_challenger(
                self,
                self._model,
                lambda overrides: self._build_model(
                    cat_feats,
                    n_rows=X_fit.shape[0],
                    extra_overrides=overrides,
                ),
                X_fit,
                y_fit,
                eval_X_np,
                eval_y_np,
                "regression",
                1,
                init_fit=init_fit,
                init_eval=init_eval,
                sample_weight=w_fit,
            )
            self._model = _maybe_select_split_risk_challenger(
                self,
                self._model,
                lambda overrides: self._build_model(
                    cat_feats,
                    n_rows=X_fit.shape[0],
                    extra_overrides=overrides,
                ),
                X_fit,
                y_fit,
                eval_X_np,
                eval_y_np,
                "regression",
                1,
                init_fit=init_fit,
                init_eval=init_eval,
                sample_weight=w_fit,
            )
            self._model = _maybe_select_residual_focus_challenger(
                self,
                self._model,
                lambda overrides: self._build_model(
                    cat_feats,
                    n_rows=X_fit.shape[0],
                    extra_overrides=overrides,
                ),
                X_fit,
                y_fit,
                eval_X_np,
                eval_y_np,
                "regression",
                1,
                init_fit=init_fit,
                init_eval=init_eval,
                sample_weight=w_fit,
            )
            _maybe_prune_validation_plateau(self, self._model, "regression", 1)
            _maybe_apply_trajectory_average(self, self._model, "regression", 1)
            _fit_region_gate(
                self, self._model, eval_X_np, eval_y_np, "regression", 1,
                init_score=init_eval,
            )
            _maybe_compile_apx(
                self,
                self._model,
                eval_X_np,
                eval_y_np,
                "regression",
                1,
                init_score=init_eval,
            )
            if not self._apx_compiled and self.apx and self.apx_optimize:
                try:
                    cp, w = _apx_optimize_weights(
                        self._model, eval_X_np, eval_y_np,
                        self.apx_n_checkpoints, self.apx_min_frac, self.apx_spacing,
                        "regression", 1,
                        init_score=init_eval,
                    )
                    self._apx_checkpoints, self._apx_weights = cp, w
                except Exception:
                    self._apx_checkpoints = None
                    self._apx_weights = None
            _maybe_guard_apx(
                self,
                self._model,
                eval_X_np,
                eval_y_np,
                "regression",
                1,
                init_score=init_eval,
            )
            _maybe_calibrate_tree_scale(
                self,
                self._model,
                eval_X_np,
                eval_y_np,
                "regression",
                1,
                init_score=init_eval,
            )
            self.residual_correction_info_ = {
                "enabled": False,
                "reason": "rejected_dev_overfit",
            }
            self.smooth_regression_info_ = {
                "enabled": False,
                "reason": "disabled_fairness_audit",
            }
        else:
            init_fit = _linear_init_score_for_fit(self._linear_init_state, X_np)
            self._model.fit(X_np, y_fit_np, self.n_estimators, init_score=init_fit)
            self._apx_compiled = False
            self._apx_disabled_by_guard = False
            self.apx_compile_info_ = {"enabled": False, "reason": "no_eval_set"}
            self.apx_guard_info_ = {"enabled": False, "reason": "no_eval_set"}
            self.plateau_prune_info_ = {"enabled": False, "reason": "no_eval_set"}
            self.tree_scale_info_ = {"enabled": False, "reason": "no_eval_set"}
            self.growth_policy_race_info_ = {"enabled": False, "reason": "no_eval_set"}
            self.split_risk_auto_info_ = {"enabled": False, "reason": "no_eval_set"}
            self.residual_focus_auto_info_ = {"enabled": False, "reason": "no_eval_set"}
            self.residual_correction_info_ = {"enabled": False, "reason": "no_eval_set"}
            self.smooth_regression_info_ = {"enabled": False, "reason": "no_eval_set"}

        _set_eval_attributes(self, self._model, "regression")
        if eval_set is not None and len(eval_set) > 0:
            _maybe_full_refit(
                self, lambda: self._build_model(cat_feats),
                X_np, y_fit_np, cat_feats, eval_X_np, eval_y_np,
                "regression", 1,
            )
        return self

    def _fit_mvpe_regressor(self, X_np, y_np, cat_feats, eval_set):
        views = self.mvpe_views or _mvpe_default_views("regression")
        numeric_idx = np.array([i for i, c in enumerate(cat_feats) if not c], dtype=int)
        eval_X_np = None
        eval_y_np = None
        if eval_set is not None and len(eval_set) > 0:
            eval_X_np = _transform_with_reference(eval_set[0][0], self._data_reference)
            eval_y_np = _apply_regression_target_transform(
                self._target_transform_state,
                np.asarray(eval_set[0][1], dtype=np.float64),
            )
        seed_base = self.seed if self.seed is not None else 42

        def _fit_one(view_idx, view_name):
            state_tuple, cat_feats_v, X_tr_v = _mvpe_fit_view(
                view_name, X_np, y_np, cat_feats, numeric_idx,
                seed=seed_base + view_idx,
            )
            params = dict(
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample_rate=self.subsample,
                lambda_reg=self.reg_lambda,
                gamma=self.gamma,
                min_child_weight=self.min_child_weight,
                colsample_bytree=self.colsample_bytree,
                task="regression",
                num_bins=self.num_bins,
                grow_policy=self.grow_policy,
                cat_features=cat_feats_v,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=_normalize_verbose(self.verbose),
                huber_delta=_normalize_huber_delta(self.huber_delta),
                sparse_oblique_splits=_normalize_regression_sparse_oblique(
                    self.sparse_oblique_splits,
                    cat_feats_v,
                    X_tr_v.shape[0],
                ),
                vertical_init=bool(self.vertical_init),
                vertical_init_cycles=int(self.vertical_init_cycles),
                seed=seed_base + view_idx,
            )
            if (
                "leaf_adaptive_blend_kappa" not in self._extra_params
                and "leaf_median" not in self._extra_params
                and "leaf_median_blend" not in self._extra_params
                and "leaf_mad_clip" not in self._extra_params
                and "leaf_trim_pct" not in self._extra_params
            ):
                params["leaf_adaptive_blend_kappa"] = _normalize_robust_leaves(self.robust_leaves)
            if any(cat_feats_v):
                params.setdefault("adaptive_leaf_experts", True)
                params.setdefault("cat_lookup_smooth", 20.0)
                params.setdefault("adaptive_cat_lookup_smooth", True)
            params.update(_native_extra_params(self._extra_params))
            if view_name == "oblique":
                params["colsample_bylevel"] = params.get("colsample_bylevel", 0.5)
                if params["colsample_bylevel"] > 0.6:
                    params["colsample_bylevel"] = 0.5
            m = GTBoostModel(**params)
            if eval_X_np is not None:
                eval_X_v = _mvpe_apply_view(state_tuple, eval_X_np)
                m.fit(X_tr_v, y_np, self.n_estimators,
                      eval_x=eval_X_v, eval_y=eval_y_np)
            else:
                m.fit(X_tr_v, y_np, self.n_estimators)
            return (state_tuple, m, cat_feats_v)

        if self.mvpe_n_jobs != 1:
            try:
                from joblib import Parallel, delayed
                # Use threading backend — Rust releases GIL during fit/predict;
                # loky/multiprocessing can't pickle the PyO3 model object.
                fits = Parallel(n_jobs=self.mvpe_n_jobs, backend="threading")(
                    delayed(_fit_one)(i, v) for i, v in enumerate(views)
                )
            except ImportError:
                fits = [_fit_one(i, v) for i, v in enumerate(views)]
        else:
            fits = [_fit_one(i, v) for i, v in enumerate(views)]

        self._mvpe_fits = fits

    def predict(self, X):
        """Predict target values.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        X_np = _transform_with_reference(X, self._data_reference)

        if (
            self._smooth_regression_payload is not None
            and not self._is_smooth_regression_eval_matrix(X_np)
            and self._ensure_smooth_regression_model()
        ):
            if self._smooth_regression_linear_init_state is None:
                raw_smooth = np.array(self._smooth_regression_model.predict(X_np))
            else:
                init_smooth = _linear_init_score(self._smooth_regression_linear_init_state, X_np)
                raw_smooth = np.array(
                    self._smooth_regression_model.predict(X_np, init_score=init_smooth)
                )
            pred = _invert_regression_target_transform(
                self._target_transform_state,
                raw_smooth,
            )
            return _apply_honest_residual_correction(
                self._residual_correction_state, X_np, pred
            )

        if self.mvpe and self._mvpe_fits is not None:
            preds_list = []
            for state_tuple, model, _ in self._mvpe_fits:
                X_v = _mvpe_apply_view(state_tuple, X_np)
                preds_list.append(np.array(model.predict(X_v)))
            pred = _invert_regression_target_transform(
                self._target_transform_state,
                np.mean(preds_list, axis=0),
            )
            return _apply_honest_residual_correction(
                self._residual_correction_state, X_np, pred
            )

        init_pred = _linear_init_score(self._linear_init_state, X_np)
        if (
            (self._full_refit_model is not None or self._full_refit_payload is not None)
            and not _same_matrix(X_np, self._full_refit_eval_X)
            and _ensure_full_refit_model(self)
        ):
            init_full = _linear_init_score(self._full_refit_linear_init_state, X_np)
            raw = np.asarray(self._full_refit_model.predict(X_np, init_score=init_full))
            pred = _invert_regression_target_transform(self._target_transform_state, raw)
            return _apply_honest_residual_correction(
                self._residual_correction_state, X_np, pred
            )

        if getattr(self, "_region_gate", None) is not None:
            raw, _ = _region_gate_apply_raw(self, self._model, X_np, "regression", 1)
            pred = _invert_regression_target_transform(self._target_transform_state, raw)
            return _apply_honest_residual_correction(
                self._residual_correction_state, X_np, pred
            )

        use_apx = (
            self.apx
            and not self._apx_compiled
            and not self._apx_disabled_by_guard
            and self._model is not None
            and _model_n_trees(self._model) >= 20
        )
        if use_apx:
            if self.apx_optimize and self._apx_checkpoints is not None:
                preds = [
                    np.asarray(
                        self._model.predict_truncated(
                            X_np, int(cp), init_score=init_pred
                        )
                    )
                    for cp in self._apx_checkpoints
                ]
                P = np.stack(preds)
                out = np.einsum("k,kn->n", self._apx_weights, P)
                pred = _invert_regression_target_transform(self._target_transform_state, out)
                return _apply_honest_residual_correction(
                    self._residual_correction_state, X_np, pred
                )
            out = _apx_predict_raw(
                self._model, X_np,
                self.apx_n_checkpoints, self.apx_min_frac,
                self.apx_weighting, self.apx_spacing,
                "regression", 1,
                init_score=init_pred,
            )
            pred = _invert_regression_target_transform(self._target_transform_state, out)
            return _apply_honest_residual_correction(
                self._residual_correction_state, X_np, pred
            )

        pred = _invert_regression_target_transform(
            self._target_transform_state,
            np.array(self._model.predict(X_np, init_score=init_pred)),
        )
        return _apply_honest_residual_correction(
            self._residual_correction_state, X_np, pred
        )

    def get_params(self, deep=True):
        params = dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            colsample_bytree=self.colsample_bytree,
            num_bins=self.num_bins,
            seed=self.seed,
            grow_policy=self.grow_policy,
            cat_features=self.cat_features,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=self.verbose,
            huber_delta=self.huber_delta,
            robust_leaves=self.robust_leaves,
            sparse_oblique_splits=self.sparse_oblique_splits,
            vertical_init=self.vertical_init,
            vertical_init_cycles=self.vertical_init_cycles,
            target_transform=self.target_transform,
            apx=self.apx,
            apx_n_checkpoints=self.apx_n_checkpoints,
            apx_min_frac=self.apx_min_frac,
            apx_weighting=self.apx_weighting,
            apx_spacing=self.apx_spacing,
            apx_optimize=self.apx_optimize,
            apx_compile=self.apx_compile,
            apx_compile_min_rel_improve=self.apx_compile_min_rel_improve,
            apx_compile_l2=self.apx_compile_l2,
            apx_compile_steps=self.apx_compile_steps,
            mvpe=self.mvpe,
            mvpe_views=self.mvpe_views,
            mvpe_n_jobs=self.mvpe_n_jobs,
            self_distill=self.self_distill,
            self_distill_folds=self.self_distill_folds,
        )
        params.update(self._extra_params)
        return params

    def set_params(self, **params):
        for key, val in params.items():
            if hasattr(self, key):
                setattr(self, key, val)
            else:
                self._extra_params[key] = val
        return self


GTBoostRegressor = GTBRegressor


try:
    __all__ = list(__all__) + [
        "Dataset",
        "train",
        "Booster",
        "GTBoostClassifier",
        "GTBoostRegressor",
        "GTBClassifier",
        "GTBRegressor",
        "GTBoostModel",
    ]
except NameError:
    __all__ = [
        "Dataset",
        "train",
        "Booster",
        "GTBoostClassifier",
        "GTBoostRegressor",
        "GTBClassifier",
        "GTBRegressor",
        "GTBoostModel",
    ]
