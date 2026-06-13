"""Small public Optuna tuner for GTBoost.

This module is intentionally conservative.  It tunes ordinary model
hyperparameters plus optional GTBoost feature families requested by the caller.
It does not contain dataset-specific anchors or hidden environment-variable
gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from . import GTBClassifier, GTBRegressor


Metric = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class TuneResult:
    """Result returned by :func:`tune_gtboost`."""

    best_params: dict[str, Any]
    best_score: float
    best_rounds: int
    study: Any
    task: str
    best_score_std: float = 0.0
    selection_score: float = 0.0
    complexity_key: tuple[int, ...] = ()


def _infer_task(y: np.ndarray, task: str) -> str:
    if task != "auto":
        if task not in {"binary", "multiclass", "regression"}:
            raise ValueError("task must be 'auto', 'binary', 'multiclass', or 'regression'")
        return task
    y_arr = np.asarray(y)
    if np.issubdtype(y_arr.dtype, np.floating):
        unique = np.unique(y_arr[~np.isnan(y_arr)])
        if unique.size > max(20, int(0.05 * y_arr.size)):
            return "regression"
    return "binary" if np.unique(y_arr).size <= 2 else "multiclass"


def _default_metric(task: str) -> Metric:
    if task == "regression":
        def rmse(y_true: np.ndarray, pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            pred = np.asarray(pred, dtype=np.float64)
            return float(np.sqrt(np.mean((y_true - pred) ** 2)))

        return rmse

    def logloss(y_true: np.ndarray, pred: np.ndarray) -> float:
        y_true = np.asarray(y_true)
        pred = np.asarray(pred, dtype=np.float64)
        eps = 1e-15
        if pred.ndim == 1:
            p = np.clip(pred, eps, 1.0 - eps)
            y_bin = y_true.astype(np.int64)
            return float(-np.mean(y_bin * np.log(p) + (1 - y_bin) * np.log1p(-p)))
        p = np.clip(pred, eps, 1.0)
        p = p / p.sum(axis=1, keepdims=True)
        y_idx = y_true.astype(np.int64)
        return float(-np.mean(np.log(p[np.arange(y_idx.size), y_idx])))

    return logloss


def _make_splits(X: Any, y: np.ndarray, task: str, n_folds: int, seed: int):
    from sklearn.model_selection import KFold, StratifiedKFold

    n_folds = max(2, int(n_folds))
    if task == "regression":
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return list(splitter.split(X))
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(splitter.split(X, y))


def _slice_rows(X: Any, rows: np.ndarray):
    if hasattr(X, "iloc"):
        return X.iloc[rows]
    return np.asarray(X)[rows]


def _normalize_choices(values: Iterable[str] | None, allowed: set[str]) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for value in values:
        item = str(value).strip().lower()
        if item in {"none", "off"}:
            item = "raw"
        if item not in allowed:
            raise ValueError(f"unsupported choice {value!r}; allowed={sorted(allowed)}")
        if item not in out:
            out.append(item)
    return out


def _suggest_params(
    trial: Any,
    *,
    task: str,
    n_features: int,
    has_categoricals: bool,
    max_rounds: int,
    categorical_geometry_choices: list[str],
    interval_choices: list[bool],
    operator_mode: str,
) -> dict[str, Any]:
    max_rounds = max(1, int(max_rounds))
    min_rounds = min(100, max(1, max_rounds))
    extended_ops = operator_mode == "extended"
    grow_choices = ["depthwise", "leafwise"]
    if extended_ops:
        grow_choices.append("oblivious")
    params: dict[str, Any] = {
        "n_estimators": trial.suggest_int("n_estimators", min_rounds, max_rounds, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 0.1, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 50.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "num_bins": trial.suggest_categorical("num_bins", [64, 128, 256]),
        "grow_policy": trial.suggest_categorical("grow_policy", grow_choices),
    }

    if extended_ops and task in {"binary", "regression"} and n_features >= 2:
        params["sparse_oblique_splits"] = trial.suggest_categorical(
            "sparse_oblique_splits", [False, True]
        )
        params["auto_interactions"] = trial.suggest_categorical(
            "auto_interactions", [False, True]
        )
    leaf_linear = False
    if n_features <= 16:
        params["leaf_linear"] = trial.suggest_categorical("leaf_linear", [False, True])
        leaf_linear = bool(params["leaf_linear"])
        if leaf_linear:
            params["n_refine"] = trial.suggest_int("n_refine", 0, 2)
            params["n_leaf_splits"] = trial.suggest_int("n_leaf_splits", 0, 1)

    if extended_ops and task in {"binary", "regression"} and not leaf_linear:
        params["expert_leaf_admission"] = trial.suggest_categorical(
            "expert_leaf_admission", [False, True]
        )
        if params["expert_leaf_admission"]:
            params["expert_min_leaf"] = 24

    if extended_ops and task in {"binary", "regression"}:
        params["corrective_block_refit"] = trial.suggest_categorical(
            "corrective_block_refit", [False, True]
        )
        if params["corrective_block_refit"]:
            params["corrective_blocks"] = trial.suggest_categorical(
                "corrective_blocks", [4, 8, 16, 32]
            )
            params["corrective_lambda"] = trial.suggest_float(
                "corrective_lambda", 1e-3, 3.0, log=True
            )
            params["corrective_blend"] = trial.suggest_float(
                "corrective_blend", 0.25, 1.0
            )
            params["corrective_audit_fraction"] = 0.2

    if interval_choices:
        params["interval_splits"] = trial.suggest_categorical(
            "interval_splits", interval_choices
        )

    if (
        has_categoricals
        and task in {"binary", "multiclass", "regression"}
        and len(categorical_geometry_choices) >= 2
    ):
        geom = trial.suggest_categorical(
            "categorical_geometry", categorical_geometry_choices
        )
        if geom != "raw":
            params["categorical_geometry"] = geom

    return params


def _complexity_key(params: Mapping[str, Any]) -> tuple[int, ...]:
    """Lexicographic model complexity exposed for tuner diagnostics.

    The first element counts optional high-capacity operator families.  Later
    elements describe leaf-linear usage, depth, bin resolution, and rounds.
    """

    categorical_geometry = str(params.get("categorical_geometry", "raw")).lower()
    optional_ops = (
        bool(params.get("sparse_oblique_splits", False)),
        bool(params.get("auto_interactions", False)),
        bool(params.get("interval_splits", False)),
        bool(params.get("leaf_linear", False)),
        bool(params.get("expert_leaf_admission", False)),
        bool(params.get("corrective_block_refit", False)),
        categorical_geometry not in {"", "raw", "none", "off"},
    )
    bin_rank = {64: 0, 128: 1, 256: 2}.get(int(params.get("num_bins", 256)), 3)
    return (
        int(sum(optional_ops)),
        int(bool(params.get("leaf_linear", False))),
        int(params.get("max_depth", 0)),
        int(bin_rank),
        int(params.get("n_estimators", 0)),
    )


def _fit_and_score(
    *,
    task: str,
    X_train: Any,
    y_train: np.ndarray,
    X_valid: Any,
    y_valid: np.ndarray,
    params: Mapping[str, Any],
    cat_features: Any,
    early_stopping_rounds: int,
    metric: Metric,
    seed: int,
) -> tuple[float, int]:
    model_params = dict(params)
    rounds = int(model_params.get("n_estimators", 500))
    model_params["seed"] = seed
    model_params["cat_features"] = cat_features
    model_params["early_stopping_rounds"] = int(early_stopping_rounds)

    if task == "regression":
        model = GTBRegressor(**model_params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
        pred = model.predict(X_valid)
    else:
        model = GTBClassifier(**model_params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
        proba = model.predict_proba(X_valid)
        pred = proba[:, 1] if task == "binary" else proba
    n_trees = len(model._model.tree_info()) if getattr(model, "_model", None) is not None else rounds
    return float(metric(y_valid, pred)), int(max(1, n_trees))


def tune_gtboost(
    X: Any,
    y: Any,
    *,
    task: str = "auto",
    cat_features: Any = None,
    n_trials: int = 50,
    n_folds: int = 3,
    max_rounds: int = 1000,
    early_stopping_rounds: int = 50,
    metric: Metric | None = None,
    seed: int = 42,
    timeout: float | None = None,
    param_overrides: Mapping[str, Any] | None = None,
    categorical_geometry_choices: Iterable[str] | None = None,
    interval_splits: str | bool = "auto",
    operator_mode: str = "core",
    robust_selection: bool = False,
    robust_z: float = 1.0,
    verbose: bool = False,
) -> TuneResult:
    """Tune GTBoost with a compact, general Optuna search.

    Parameters are intentionally algorithmic.  The default search space is
    conservative: standard GTBoost trees only, with experimental operator
    families enabled only when ``operator_mode="extended"``.  Set
    ``robust_selection=True`` to optimize mean CV loss plus a one-standard-error
    uncertainty charge.
    """

    try:
        import optuna
    except ImportError as exc:
        raise ImportError("tune_gtboost requires the 'tuning' extra: pip install gtboost[tuning]") from exc

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    y_arr = np.asarray(y)
    task = _infer_task(y_arr, task)
    operator_mode = str(operator_mode).strip().lower()
    if operator_mode not in {"core", "extended"}:
        raise ValueError("operator_mode must be 'core' or 'extended'")
    metric_fn = metric or _default_metric(task)
    n_features = int(X.shape[1]) if hasattr(X, "shape") else int(np.asarray(X).shape[1])

    if isinstance(cat_features, str) and cat_features == "auto" and hasattr(X, "dtypes"):
        has_categoricals = any(str(dtype) in {"category", "object", "string"} for dtype in X.dtypes)
    elif isinstance(cat_features, (list, tuple, np.ndarray)):
        has_categoricals = any(bool(v) for v in cat_features)
    else:
        has_categoricals = bool(cat_features) if cat_features is not None else False

    geom_choices = _normalize_choices(
        categorical_geometry_choices,
        {"raw", "pcf", "pcf_lite"},
    )
    if geom_choices and "raw" not in geom_choices:
        geom_choices.insert(0, "raw")

    if interval_splits == "auto":
        interval_choices = [False, True] if task in {"binary", "regression"} else []
    elif isinstance(interval_splits, bool):
        interval_choices = [bool(interval_splits)]
    else:
        raise ValueError("interval_splits must be 'auto', True, or False")

    splits = _make_splits(X, y_arr, task, n_folds, seed)
    overrides = dict(param_overrides or {})

    def objective(trial: Any) -> float:
        params = _suggest_params(
            trial,
            task=task,
            n_features=n_features,
            has_categoricals=has_categoricals,
            max_rounds=max_rounds,
            categorical_geometry_choices=geom_choices,
            interval_choices=interval_choices,
            operator_mode=operator_mode,
        )
        params.update(overrides)
        scores: list[float] = []
        rounds: list[int] = []
        for fold_idx, (tr, va) in enumerate(splits):
            score, n_trees = _fit_and_score(
                task=task,
                X_train=_slice_rows(X, tr),
                y_train=y_arr[tr],
                X_valid=_slice_rows(X, va),
                y_valid=y_arr[va],
                params=params,
                cat_features=cat_features,
                early_stopping_rounds=early_stopping_rounds,
                metric=metric_fn,
                seed=seed + fold_idx,
            )
            scores.append(score)
            rounds.append(n_trees)
        mean_score = float(np.mean(scores))
        score_std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
        score_se = score_std / max(len(scores), 1) ** 0.5
        selection_score = mean_score + max(0.0, float(robust_z)) * score_se
        if not robust_selection:
            selection_score = mean_score
        trial.set_user_attr("params", dict(params))
        trial.set_user_attr("rounds", int(round(np.mean(rounds))))
        trial.set_user_attr("mean_score", mean_score)
        trial.set_user_attr("score_std", score_std)
        trial.set_user_attr("selection_score", float(selection_score))
        trial.set_user_attr("complexity_key", list(_complexity_key(params)))
        return float(selection_score)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=int(n_trials), timeout=timeout, show_progress_bar=False)

    best_trial = study.best_trial
    best_params = dict(best_trial.user_attrs.get("params", best_trial.params))
    best_rounds = int(best_trial.user_attrs.get("rounds", best_params.get("n_estimators", max_rounds)))
    best_params["n_estimators"] = best_rounds
    best_mean_score = float(best_trial.user_attrs.get("mean_score", best_trial.value))
    best_score_std = float(best_trial.user_attrs.get("score_std", 0.0))
    selection_score = float(best_trial.user_attrs.get("selection_score", best_trial.value))
    complexity_key = tuple(best_trial.user_attrs.get("complexity_key", _complexity_key(best_params)))
    return TuneResult(
        best_params=best_params,
        best_score=best_mean_score,
        best_rounds=best_rounds,
        study=study,
        task=task,
        best_score_std=best_score_std,
        selection_score=selection_score,
        complexity_key=complexity_key,
    )


tune_optuna = tune_gtboost

__all__ = ["TuneResult", "tune_gtboost", "tune_optuna"]
