import numpy as np
import pytest

import gtboost as gtb
from gtboost import GTBoostClassifier, GTBoostRegressor


pd = pytest.importorskip("pandas")


def _binary_frame(n=80):
    rng = np.random.default_rng(7)
    city = np.where(rng.random(n) > 0.5, "north", "south")
    x = rng.normal(size=n)
    y = ((x + (city == "north") * 0.8 + rng.normal(scale=0.3, size=n)) > 0.4).astype(float)
    return pd.DataFrame({"x": x, "city": city, "target": y})


def test_native_dataset_train_predict_dataframe():
    df = _binary_frame()
    train = gtb.Dataset(df.iloc[:60], label="target", categorical="auto")
    valid = gtb.Dataset(df.iloc[60:], label="target", reference=train)

    model = gtb.train(
        {
            "objective": "binary",
            "learning_rate": 0.2,
            "max_depth": 2,
            "categorical_geometry": "raw",
            "interval_splits": False,
            "random_state": 3,
        },
        train,
        valid_sets=[valid],
        num_boost_round=8,
        early_stopping_rounds=0,
    )

    pred = model.predict(df.drop(columns=["target"]).iloc[60:])
    proba = model.predict_proba(df.drop(columns=["target"]).iloc[60:])
    assert pred.shape == (20,)
    assert proba.shape == (20, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_booster_save_load_preserves_dataframe_categories(tmp_path):
    df = _binary_frame()
    train = gtb.Dataset(df.iloc[:60], label="target", categorical="auto")
    valid = gtb.Dataset(df.iloc[60:], label="target", reference=train)

    model = gtb.train(
        {
            "objective": "binary",
            "learning_rate": 0.2,
            "max_depth": 2,
            "categorical_geometry": "raw",
            "interval_splits": False,
            "random_state": 33,
        },
        train,
        valid_sets=[valid],
        num_boost_round=8,
        early_stopping_rounds=0,
    )

    X_eval = df.drop(columns=["target"]).iloc[60:]
    before = model.predict(X_eval)
    path = tmp_path / "booster.gtboost"
    model.save_model(path)
    loaded = gtb.Booster.load_model(path)
    after = loaded.predict(X_eval)

    assert np.allclose(before, after, atol=1e-12, rtol=1e-12)


def test_sklearn_classifier_dataframe_smoke():
    df = _binary_frame()
    X = df.drop(columns=["target"])
    y = df["target"].to_numpy()
    clf = GTBoostClassifier(
        n_estimators=8,
        learning_rate=0.2,
        max_depth=2,
        apx=False,
        interval_splits=False,
        categorical_geometry="raw",
        seed=4,
    )
    clf.fit(X.iloc[:60], y[:60], eval_set=[(X.iloc[60:], y[60:])])
    proba = clf.predict_proba(X.iloc[60:])
    assert proba.shape == (20, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_sklearn_regressor_numpy_smoke():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(70, 3))
    X[::7, 1] = np.nan
    y = X[:, 0] * 0.8 + np.nan_to_num(X[:, 1]) * -0.2 + rng.normal(scale=0.1, size=70)
    reg = GTBoostRegressor(
        n_estimators=8,
        learning_rate=0.2,
        max_depth=2,
        apx=False,
        interval_splits=False,
        seed=5,
    )
    reg.fit(X[:50], y[:50], eval_set=[(X[50:], y[50:])])
    pred = reg.predict(X[50:])
    assert pred.shape == (20,)
    assert np.isfinite(pred).all()


def test_public_tuner_smoke():
    pytest.importorskip("optuna")
    from gtboost.tuner import tune_gtboost

    df = _binary_frame(72)
    X = df.drop(columns=["target"])
    y = df["target"].to_numpy()
    result = tune_gtboost(
        X,
        y,
        task="binary",
        cat_features="auto",
        n_trials=2,
        n_folds=2,
        max_rounds=12,
        early_stopping_rounds=0,
        categorical_geometry_choices=["raw"],
        interval_splits=False,
        seed=17,
    )

    assert result.task == "binary"
    assert np.isfinite(result.best_score)
    assert result.best_rounds >= 1
    assert result.best_params["n_estimators"] == result.best_rounds


def test_interval_prediction_is_reentrant_and_single_row_safe():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(96, 4))
    X[::9, 2] = np.nan
    y = 0.7 * X[:, 0] - 0.4 * np.nan_to_num(X[:, 2]) + rng.normal(scale=0.05, size=96)

    train = gtb.Dataset(X[:72], label=y[:72])
    model = gtb.train(
        {
            "objective": "regression",
            "learning_rate": 0.2,
            "max_depth": 3,
            "num_bins": 32,
            "interval_splits": True,
            "categorical_geometry": "raw",
            "random_state": 11,
        },
        train,
        num_boost_round=12,
    )

    batch = X[72:84]
    pred_a = model.predict(batch)
    pred_b = model.predict(batch.copy())
    single = model.predict(batch[:1])
    assert pred_a.shape == (12,)
    assert single.shape == (1,)
    assert np.allclose(pred_a, pred_b)
    assert np.allclose(single[0], pred_a[0])


def test_predict_handles_non_contiguous_numpy_input():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(90, 5))
    y = X[:, 0] * 0.5 + X[:, 3] * -0.3 + rng.normal(scale=0.05, size=90)
    reg = GTBoostRegressor(
        n_estimators=10,
        learning_rate=0.2,
        max_depth=2,
        apx=False,
        interval_splits=True,
        seed=13,
    )
    reg.fit(X[:60], y[:60])

    contiguous = X[60:80].copy()
    non_contiguous = np.asfortranarray(contiguous)
    assert not non_contiguous.flags.c_contiguous
    assert np.allclose(reg.predict(contiguous), reg.predict(non_contiguous))


def test_model_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(17)
    X = rng.normal(size=(120, 4))
    X[::8, 1] = np.nan
    y = 0.8 * X[:, 0] + 0.3 * np.nan_to_num(X[:, 1]) + rng.normal(scale=0.05, size=120)

    model = gtb.GTBoostModel(
        task="regression",
        n_estimators=20,
        learning_rate=0.15,
        max_depth=3,
        num_bins=32,
        interval_splits=True,
        categorical_geometry="raw",
        seed=17,
    )
    model.fit(X[:90], y[:90])
    before = model.predict(X[90:])

    path = tmp_path / "model.gtboost.json"
    model.save_model(path)
    loaded = gtb.GTBoostModel.load_model(path)
    after = loaded.predict(X[90:])

    assert np.allclose(before, after, atol=1e-12, rtol=1e-12)


def test_pcf_geometry_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(18)
    n = 180
    X = np.column_stack(
        [
            np.arange(n) % 12,
            np.arange(n) % 9,
            rng.normal(size=n),
        ]
    ).astype(float)
    y = (((X[:, 0] == 3) & (X[:, 1] == 4)) | (X[:, 2] > 0.8)).astype(float)

    model = gtb.GTBoostModel(
        task="binary",
        n_estimators=14,
        learning_rate=0.2,
        max_depth=2,
        cat_features=[True, True, False],
        categorical_geometry="pcf_lite",
        pcf_config={
            "eligibility_gate": False,
            "max_cat": 2,
            "max_pairs": 1,
            "max_triples": 0,
            "folds": 3,
            "coordinate_mode": "prob3",
        },
        seed=18,
    )
    model.fit(X[:140], y[:140])
    assert model._pcf_runtime is not None and model._pcf_runtime.enabled
    before = model.predict(X[140:])

    path = tmp_path / "pcf_model.gtboost"
    model.save_model(path)
    loaded = gtb.GTBoostModel.load_model(path)
    after = loaded.predict(X[140:])

    assert loaded._pcf_runtime is not None and loaded._pcf_runtime.enabled
    assert np.allclose(before, after, atol=1e-12, rtol=1e-12)


def test_unknown_categorical_geometry_is_rejected():
    with pytest.raises(ValueError, match="categorical_geometry"):
        gtb.GTBoostModel(categorical_geometry="unsupported_geometry")


def test_training_is_deterministic_for_same_seed():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(140, 5))
    y = (X[:, 0] - 0.5 * X[:, 2] + rng.normal(scale=0.1, size=140) > 0.0).astype(float)

    params = dict(
        n_estimators=18,
        learning_rate=0.15,
        max_depth=3,
        apx=False,
        interval_splits=True,
        categorical_geometry="raw",
        seed=19,
        subsample=0.85,
        colsample_bytree=0.8,
    )
    a = GTBoostClassifier(**params).fit(X[:100], y[:100])
    b = GTBoostClassifier(**params).fit(X[:100], y[:100])

    assert np.allclose(a.predict_proba(X[100:]), b.predict_proba(X[100:]), atol=0.0, rtol=0.0)


def test_advanced_prediction_paths_smoke():
    rng = np.random.default_rng(23)

    # PCF path: Python-side categorical geometry should remain reentrant.
    df = pd.DataFrame(
        {
            "city": np.where(np.arange(160) % 3 == 0, "a", np.where(np.arange(160) % 3 == 1, "b", "c")),
            "seg": np.where(np.arange(160) % 5 < 2, "x", "y"),
            "x": rng.normal(size=160),
        }
    )
    y_bin = ((df["city"].to_numpy() == "a").astype(float) + df["x"].to_numpy() > 0.2).astype(float)
    pcf = GTBoostClassifier(
        n_estimators=12,
        learning_rate=0.2,
        max_depth=2,
        categorical_geometry="pcf_lite",
        apx=False,
        seed=23,
    ).fit(df.iloc[:120], y_bin[:120])
    p1 = pcf.predict_proba(df.iloc[120:])
    p2 = pcf.predict_proba(df.iloc[120:].copy())
    assert p1.shape == (40, 2)
    assert np.allclose(p1, p2)

    # JIT categorical-pair path.
    X_pair = np.column_stack(
        [
            (np.arange(180) % 7).astype(float),
            (np.arange(180) % 5).astype(float),
            rng.normal(size=180),
        ]
    )
    y_pair = (((X_pair[:, 0] == 2) & (X_pair[:, 1] == 3)) | (X_pair[:, 2] > 1.0)).astype(float)
    pair = GTBoostClassifier(
        n_estimators=16,
        learning_rate=0.2,
        max_depth=3,
        cat_features=[True, True, False],
        categorical_geometry="raw",
        jit_catpair_enabled=True,
        jit_catpair_min_node_rows=16,
        apx=False,
        seed=24,
    ).fit(X_pair[:140], y_pair[:140])
    assert np.allclose(pair.predict_proba(X_pair[140:]), pair.predict_proba(X_pair[140:].copy()))

    # Multiclass path.
    X_mc = rng.normal(size=(150, 4))
    y_mc = np.argmax(
        np.column_stack([X_mc[:, 0], -X_mc[:, 1] + 0.2, X_mc[:, 2] - X_mc[:, 3]]),
        axis=1,
    ).astype(float)
    mc = GTBoostClassifier(
        n_estimators=12,
        learning_rate=0.15,
        max_depth=2,
        categorical_geometry="raw",
        apx=False,
        seed=25,
    ).fit(X_mc[:110], y_mc[:110])
    proba = mc.predict_proba(X_mc[110:])
    assert proba.shape == (40, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)

    # Self-score and ramp fallback path.
    X_reg = rng.normal(size=(150, 4))
    y_reg = X_reg[:, 0] * 0.6 + X_reg[:, 1] * X_reg[:, 2] * 0.1 + rng.normal(scale=0.05, size=150)
    reg = GTBoostRegressor(
        n_estimators=14,
        learning_rate=0.15,
        max_depth=3,
        categorical_geometry="raw",
        interval_splits=True,
        self_score_splits=True,
        ramp=True,
        apx=False,
        seed=26,
    ).fit(X_reg[:110], y_reg[:110])
    assert np.allclose(reg.predict(X_reg[110:]), reg.predict(X_reg[110:].copy()))
