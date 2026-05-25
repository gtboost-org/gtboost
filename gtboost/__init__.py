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
        self._categorical_geometry = {
            "pcf_lite": "pcf",
        }.get(raw_geometry, raw_geometry)
        self._pcf_runtime = None
        self.categorical_geometry_info_ = {
            "enabled": False,
            "reason": "categorical_geometry disabled",
        }
        self._model = None
        self.evals_result_ = {}
        self.best_iteration_ = None
        self.best_score_ = None
        if self._categorical_geometry is None:
            self._model = _RustGTBoostModel(*self._raw_args, **self._raw_kwargs)
        elif self._categorical_geometry != "pcf":
            raise ValueError(
                "unknown categorical_geometry="
                f"{categorical_geometry!r}; supported values are 'pcf' and alias 'pcf_lite'"
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

        if self._task() == "binary":
            from feature_transforms import PCFGeometryRuntime

            feature_view_groups = None
            pcf_cat_feats = self._pcf_cat_features_for(X_np, cat_feats)
            runtime = PCFGeometryRuntime(
                task_type="binary",
                n_classes=2,
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
                "reason": "categorical_geometry is currently binary-only",
            }
            self._ensure_model(cat_feats)

        result = self._model.fit(
            X_fit,
            y_np,
            n_rounds,
            eval_x=eval_fit,
            eval_y=eval_y_fit,
            init_score=init_score,
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
    if weighting == "flat":
        return np.ones(k) / k
    if weighting == "triangle":
        w = np.arange(1, k + 1, dtype=float)
        return w / w.sum()
    if weighting == "gauss":
        sigma = max(1.0, k / 4.0)
        idx = np.arange(k, dtype=float)
        w = np.exp(-((idx - (k - 1)) ** 2) / (2 * sigma ** 2))
        return w / w.sum()
    raise ValueError(f"unknown APX weighting: {weighting}")


def _apx_predict_raw(model, X_np, n_checkpoints, min_frac, weighting, spacing,
                     task, n_classes):
    """Return weighted path-average of raw predictions.
    task: 'binary', 'multiclass', or 'regression'.
    Returns shape (n,) for binary/regression, (n, K) for multiclass.
    For binary returns raw logits; for multiclass returns (renormalized) probs
    averaged in probability space.
    """
    n_total = len(model.tree_info())
    checkpoints = _apx_checkpoints(n_total, n_checkpoints, min_frac, 1.0, spacing)
    w = _apx_weights(len(checkpoints), weighting)

    acc = None
    for wi, cp in zip(w, checkpoints):
        raw = np.asarray(model.predict_truncated(X_np, int(cp)))
        if task == "multiclass":
            raw = raw.reshape(-1, n_classes)
            raw = raw - raw.max(axis=1, keepdims=True)
            exp = np.exp(raw)
            raw = exp / exp.sum(axis=1, keepdims=True)
        elif task == "binary":
            # Convert to prob for averaging (more principled than averaging logits)
            raw = 1.0 / (1.0 + np.exp(-raw))
        acc = wi * raw if acc is None else acc + wi * raw
    return acc


def _apx_optimize_weights(model, X_val, y_val, n_checkpoints, min_frac, spacing,
                          task, n_classes, l2=1e-3):
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
        raw = np.asarray(model.predict_truncated(X_val, int(cp)))
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
        Tree growing strategy: "depthwise", "leafwise", or "oblivious".
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
        apx=True,
        apx_n_checkpoints=10,
        apx_min_frac=0.3,
        apx_weighting="gauss",
        apx_spacing="uniform",
        apx_optimize=False,
        mvpe=False,
        mvpe_views=None,
        mvpe_n_jobs=1,
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
        self.apx = apx
        self.apx_n_checkpoints = apx_n_checkpoints
        self.apx_min_frac = apx_min_frac
        self.apx_weighting = apx_weighting
        self.apx_spacing = apx_spacing
        self.apx_optimize = apx_optimize
        self.mvpe = mvpe
        self.mvpe_views = mvpe_views
        self.mvpe_n_jobs = mvpe_n_jobs
        self._extra_params = kwargs
        self._model = None
        self._n_classes = None
        self._classes = None
        self._apx_checkpoints = None
        self._apx_weights = None
        self._task = None
        self._mvpe_fits = None  # list of (state_tuple, model, cat_feats) when mvpe=True
        self._data_reference = None
        self.evals_result_ = {}
        self.best_iteration_ = None
        self.best_score_ = None

    def _build_model(self, task, cat_feats):
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
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=_normalize_verbose(self.verbose),
        )
        if self.seed is not None:
            params["seed"] = self.seed
        params.update(self._extra_params)
        return GTBoostModel(**params)

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None):
        """Fit the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Class labels (integers starting from 0).
        eval_set : list of (X, y) tuples, optional
            Validation set for early stopping. Only the first pair is used.
        early_stopping_rounds : int, optional
            Override constructor value.

        Returns
        -------
        self
        """
        ds, X_np, y_np, cat_feats = _fit_dataset(X, y, self.cat_features)
        self._data_reference = ds

        self._classes = np.unique(y_np).astype(int)
        self._n_classes = len(self._classes)
        task = "binary" if self._n_classes <= 2 else "multiclass"

        if early_stopping_rounds is not None:
            self.early_stopping_rounds = early_stopping_rounds

        self._model = self._build_model(task, cat_feats)

        self._task = "binary" if self._n_classes <= 2 else "multiclass"

        if self.mvpe:
            self._fit_mvpe_classifier(X_np, y_np, cat_feats, eval_set)
            return self

        if eval_set is not None and len(eval_set) > 0:
            eval_X, eval_y = eval_set[0]
            eval_X_np = _transform_with_reference(eval_X, self._data_reference)
            eval_y_np = np.asarray(eval_y, dtype=np.float64)
            self._model.fit(
                X_np, y_np, self.n_estimators,
                eval_x=eval_X_np,
                eval_y=eval_y_np,
            )
            # APX-Optimize: fit path weights on the eval set used for ES.
            if self.apx and self.apx_optimize:
                try:
                    cp, w = _apx_optimize_weights(
                        self._model, eval_X_np, eval_y_np,
                        self.apx_n_checkpoints, self.apx_min_frac, self.apx_spacing,
                        self._task, self._n_classes,
                    )
                    self._apx_checkpoints, self._apx_weights = cp, w
                except Exception:
                    self._apx_checkpoints = None
                    self._apx_weights = None
        else:
            self._model.fit(X_np, y_np, self.n_estimators)

        _set_eval_attributes(self, self._model, self._task)
        return self

    def _fit_mvpe_classifier(self, X_np, y_np, cat_feats, eval_set):
        """Train K view models, store per-view state + model."""
        views = self.mvpe_views or _mvpe_default_views(self._task)
        numeric_idx = np.array([i for i, c in enumerate(cat_feats) if not c], dtype=int)
        eval_X_np = None
        eval_y_np = None
        if eval_set is not None and len(eval_set) > 0:
            eval_X_np = _transform_with_reference(eval_set[0][0], self._data_reference)
            eval_y_np = np.asarray(eval_set[0][1], dtype=np.float64)
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
            params.update(self._extra_params)
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
        """Predict class probabilities.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
        """
        X_np = _transform_with_reference(X, self._data_reference)

        # MVPE: ensemble over views
        if self.mvpe and self._mvpe_fits is not None:
            proba_list = []
            for state_tuple, model, _ in self._mvpe_fits:
                X_v = _mvpe_apply_view(state_tuple, X_np)
                raw = np.array(model.predict(X_v))
                if self._n_classes <= 2:
                    p = 1.0 / (1.0 + np.exp(-raw))
                    proba_list.append(np.column_stack([1 - p, p]))
                else:
                    n = X_np.shape[0]
                    logits = raw.reshape(n, self._n_classes)
                    logits = logits - logits.max(axis=1, keepdims=True)
                    exp = np.exp(logits)
                    proba_list.append(exp / exp.sum(axis=1, keepdims=True))
            return np.mean(proba_list, axis=0)

        # APX path-averaging — skip if disabled or model too small.
        use_apx = self.apx and self._model is not None and len(self._model.tree_info()) >= 20
        if use_apx:
            if self.apx_optimize and self._apx_checkpoints is not None:
                # Use learned weights
                preds = []
                for cp in self._apx_checkpoints:
                    raw = np.asarray(self._model.predict_truncated(X_np, int(cp)))
                    if self._task == "multiclass":
                        raw = raw.reshape(-1, self._n_classes)
                        raw = raw - raw.max(axis=1, keepdims=True)
                        exp = np.exp(raw)
                        raw = exp / exp.sum(axis=1, keepdims=True)
                    elif self._task == "binary":
                        raw = 1.0 / (1.0 + np.exp(-raw))
                    preds.append(raw)
                P = np.stack(preds)
                if self._task == "multiclass":
                    proba = np.einsum("k,knc->nc", self._apx_weights, P)
                else:
                    p = np.einsum("k,kn->n", self._apx_weights, P)
                    proba = np.column_stack([1 - p, p])
                return proba
            # Fixed weighting (default)
            out = _apx_predict_raw(
                self._model, X_np,
                self.apx_n_checkpoints, self.apx_min_frac,
                self.apx_weighting, self.apx_spacing,
                self._task, self._n_classes,
            )
            if self._task == "binary":
                p = out
                return np.column_stack([1 - p, p])
            return out

        # Fallback: no APX
        raw = np.array(self._model.predict(X_np))
        if self._n_classes <= 2:
            p = 1.0 / (1.0 + np.exp(-raw))
            return np.column_stack([1 - p, p])
        else:
            n = X_np.shape[0]
            logits = raw.reshape(n, self._n_classes)
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X):
        """Predict class labels.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
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
    if task != "binary" or not any(train_set.cat_features):
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
        p["categorical_geometry"] = _auto_categorical_geometry(train_set, task)
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
        Tree growing strategy: "depthwise", "leafwise", or "oblivious".
    cat_features : list of bool, default=None
        Which features are categorical. Auto-detected from pandas DataFrames.
    early_stopping_rounds : int, default=0
        Stop if no improvement for this many rounds (requires eval_set).
    verbose : int or bool, default=0
        Training log interval. 0/False is silent, True/1 logs every round,
        and an integer N logs every N rounds.
    huber_delta : float, default=0.0
        Huber loss delta (0.0 = MSE).
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
        huber_delta=0.0,
        apx=True,
        apx_n_checkpoints=10,
        apx_min_frac=0.3,
        apx_weighting="gauss",
        apx_spacing="uniform",
        apx_optimize=False,
        mvpe=False,
        mvpe_views=None,
        mvpe_n_jobs=1,
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
        self.apx = apx
        self.apx_n_checkpoints = apx_n_checkpoints
        self.apx_min_frac = apx_min_frac
        self.apx_weighting = apx_weighting
        self.apx_spacing = apx_spacing
        self.apx_optimize = apx_optimize
        self.mvpe = mvpe
        self.mvpe_views = mvpe_views
        self.mvpe_n_jobs = mvpe_n_jobs
        self._extra_params = kwargs
        self._model = None
        self._apx_checkpoints = None
        self._apx_weights = None
        self._mvpe_fits = None
        self._data_reference = None
        self.evals_result_ = {}
        self.best_iteration_ = None
        self.best_score_ = None

    def _build_model(self, cat_feats):
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
            huber_delta=self.huber_delta,
        )
        if self.seed is not None:
            params["seed"] = self.seed
        params.update(self._extra_params)
        return GTBoostModel(**params)

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
        ds, X_np, y_np, cat_feats = _fit_dataset(X, y, self.cat_features)
        self._data_reference = ds

        if early_stopping_rounds is not None:
            self.early_stopping_rounds = early_stopping_rounds

        self._model = self._build_model(cat_feats)

        if self.mvpe:
            self._fit_mvpe_regressor(X_np, y_np, cat_feats, eval_set)
            return self

        if eval_set is not None and len(eval_set) > 0:
            eval_X, eval_y = eval_set[0]
            eval_X_np = _transform_with_reference(eval_X, self._data_reference)
            eval_y_np = np.asarray(eval_y, dtype=np.float64)
            self._model.fit(
                X_np, y_np, self.n_estimators,
                eval_x=eval_X_np,
                eval_y=eval_y_np,
            )
            if self.apx and self.apx_optimize:
                try:
                    cp, w = _apx_optimize_weights(
                        self._model, eval_X_np, eval_y_np,
                        self.apx_n_checkpoints, self.apx_min_frac, self.apx_spacing,
                        "regression", 1,
                    )
                    self._apx_checkpoints, self._apx_weights = cp, w
                except Exception:
                    self._apx_checkpoints = None
                    self._apx_weights = None
        else:
            self._model.fit(X_np, y_np, self.n_estimators)

        _set_eval_attributes(self, self._model, "regression")
        return self

    def _fit_mvpe_regressor(self, X_np, y_np, cat_feats, eval_set):
        views = self.mvpe_views or _mvpe_default_views("regression")
        numeric_idx = np.array([i for i, c in enumerate(cat_feats) if not c], dtype=int)
        eval_X_np = None
        eval_y_np = None
        if eval_set is not None and len(eval_set) > 0:
            eval_X_np = _transform_with_reference(eval_set[0][0], self._data_reference)
            eval_y_np = np.asarray(eval_set[0][1], dtype=np.float64)
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
                huber_delta=self.huber_delta,
                seed=seed_base + view_idx,
            )
            params.update(self._extra_params)
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

        if self.mvpe and self._mvpe_fits is not None:
            preds_list = []
            for state_tuple, model, _ in self._mvpe_fits:
                X_v = _mvpe_apply_view(state_tuple, X_np)
                preds_list.append(np.array(model.predict(X_v)))
            return np.mean(preds_list, axis=0)

        use_apx = self.apx and self._model is not None and len(self._model.tree_info()) >= 20
        if use_apx:
            if self.apx_optimize and self._apx_checkpoints is not None:
                preds = [np.asarray(self._model.predict_truncated(X_np, int(cp)))
                         for cp in self._apx_checkpoints]
                P = np.stack(preds)
                return np.einsum("k,kn->n", self._apx_weights, P)
            return _apx_predict_raw(
                self._model, X_np,
                self.apx_n_checkpoints, self.apx_min_frac,
                self.apx_weighting, self.apx_spacing,
                "regression", 1,
            )

        return np.array(self._model.predict(X_np))

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
