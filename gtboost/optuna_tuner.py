"""Backward-compatible import path for the public GTBoost tuner.

The public package exposes a compact, schema-driven tuner in ``gtboost.tuner``.
This module keeps old imports working.
"""

from __future__ import annotations

from typing import Any

from .tuner import TuneResult, tune_gtboost


def tune_optuna(
    X_tr: Any,
    y_tr: Any,
    task_type: str = "auto",
    n_classes: int | None = None,
    cat_features: Any = None,
    n_trials: int = 50,
    seed: int = 42,
    verbose: bool = True,
    metric_fn: Any = None,
    n_folds: int = 3,
    top_k: int = 1,
    param_overrides: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[dict[str, Any], int, float, list[dict[str, Any]]]:
    """Compatibility wrapper returning the legacy tuple shape.

    New code should call :func:`gtboost.tuner.tune_gtboost`, which returns a
    :class:`gtboost.tuner.TuneResult`.
    """

    _ = n_classes, top_k
    allowed_kwargs = {
        "timeout",
        "max_rounds",
        "early_stopping_rounds",
        "categorical_geometry_choices",
        "interval_splits",
    }
    tuner_kwargs = {k: v for k, v in kwargs.items() if k in allowed_kwargs}
    result = tune_gtboost(
        X_tr,
        y_tr,
        task=task_type,
        cat_features=cat_features,
        n_trials=n_trials,
        n_folds=n_folds,
        metric=metric_fn,
        seed=seed,
        param_overrides=param_overrides,
        verbose=verbose,
        **tuner_kwargs,
    )
    return result.best_params, result.best_rounds, result.best_score, [result.best_params]


__all__ = ["TuneResult", "tune_gtboost", "tune_optuna"]
