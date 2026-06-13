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


def test_multiclass_pcf_dataframe_smoke():
    rng = np.random.default_rng(21)
    n = 150
    group = np.array(["a", "b", "c", "d", "e"])[rng.integers(0, 5, size=n)]
    x = rng.normal(size=n)
    y = ((group == "b").astype(int) + 2 * (group == "d").astype(int) + (x > 0.8).astype(int)) % 3
    df = pd.DataFrame({"x": x, "group": group})

    clf = GTBoostClassifier(
        n_estimators=16,
        learning_rate=0.2,
        max_depth=2,
        apx=False,
        interval_splits=False,
        categorical_geometry="pcf_lite",
        pcf_config={
            "eligibility_gate": False,
            "max_cat": 1,
            "max_pairs": 0,
            "max_triples": 0,
            "folds": 3,
            "coordinate_mode": "raw",
        },
        seed=21,
    )
    clf.fit(df.iloc[:110], y[:110], eval_set=[(df.iloc[110:], y[110:])])
    proba = clf.predict_proba(df.iloc[110:])

    assert proba.shape == (40, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert clf._model.categorical_geometry_info_["enabled"] is True


def test_regression_pcf_dataframe_smoke():
    rng = np.random.default_rng(22)
    n = 150
    city = np.array(["north", "south", "east", "west", "central"])[rng.integers(0, 5, size=n)]
    x = rng.normal(size=n)
    y = 2.0 * x + np.where(city == "north", 1.5, 0.0) + np.where(city == "west", -0.8, 0.0)
    y = y + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({"x": x, "city": city})

    reg = GTBoostRegressor(
        n_estimators=16,
        learning_rate=0.2,
        max_depth=2,
        apx=False,
        interval_splits=False,
        categorical_geometry="pcf_lite",
        pcf_config={
            "eligibility_gate": False,
            "max_cat": 1,
            "max_pairs": 0,
            "max_triples": 0,
            "folds": 3,
        },
        seed=22,
    )
    reg.fit(df.iloc[:110], y[:110], eval_set=[(df.iloc[110:], y[110:])])
    pred = reg.predict(df.iloc[110:])

    assert pred.shape == (40,)
    assert np.all(np.isfinite(pred))
    assert reg._model.categorical_geometry_info_["enabled"] is True


def test_verbose_and_eval_attributes_for_sklearn_and_native(capfd):
    df = _binary_frame(90)
    X = df.drop(columns=["target"])
    y = df["target"].to_numpy()

    clf = GTBoostClassifier(
        n_estimators=10,
        learning_rate=0.2,
        max_depth=2,
        apx=False,
        interval_splits=False,
        categorical_geometry="raw",
        early_stopping_rounds=4,
        verbose=3,
        seed=44,
    )
    clf.fit(X.iloc[:64], y[:64], eval_set=[(X.iloc[64:], y[64:])])
    captured = capfd.readouterr()
    assert "valid-loss" in captured.err
    assert clf.best_iteration_ is not None
    assert clf.best_score_ is not None
    assert "validation_0" in clf.evals_result_
    assert len(clf.evals_result_["validation_0"]["logloss"]) >= 1

    train = gtb.Dataset(df.iloc[:64], label="target", categorical="auto")
    valid = gtb.Dataset(df.iloc[64:], label="target", reference=train)
    booster = gtb.train(
        {
            "objective": "binary",
            "learning_rate": 0.2,
            "max_depth": 2,
            "categorical_geometry": "raw",
            "interval_splits": False,
            "verbose": False,
            "random_state": 45,
        },
        train,
        valid_sets=[valid],
        num_boost_round=10,
        early_stopping_rounds=4,
    )
    captured = capfd.readouterr()
    assert captured.err == ""
    assert booster.best_iteration_ is not None
    assert booster.best_score_ is not None
    assert len(booster.evals_result_["validation_0"]["logloss"]) >= 1


def test_binary_corrective_block_refit_smoke():
    df = _binary_frame(140)
    X = df.drop(columns=["target"])
    y = df["target"].to_numpy()

    clf = GTBoostClassifier(
        n_estimators=24,
        learning_rate=0.12,
        max_depth=2,
        apx=False,
        interval_splits=False,
        categorical_geometry="raw",
        corrective_block_refit=True,
        corrective_blocks=4,
        corrective_lambda=0.1,
        corrective_blend=0.5,
        corrective_audit_fraction=0.2,
        seed=23,
    )
    clf.fit(X.iloc[:100], y[:100], eval_set=[(X.iloc[100:], y[100:])])
    proba = clf.predict_proba(X.iloc[100:])

    assert proba.shape == (40, 2)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_compiled_apx_tree_weights_smoke():
    rng = np.random.default_rng(24)
    X = rng.normal(size=(180, 5))
    y = 0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.2 * X[:, 2] + rng.normal(scale=0.25, size=180)

    reg = GTBoostRegressor(
        n_estimators=36,
        learning_rate=0.12,
        max_depth=2,
        apx=False,
        apx_compile=True,
        apx_compile_min_rel_improve=-1.0,
        apx_compile_steps=20,
        interval_splits=False,
        seed=24,
    )
    reg.fit(X[:130], y[:130], eval_set=[(X[130:], y[130:])])
    pred = reg.predict(X[130:])
    weights = reg._model.tree_weights()

    assert pred.shape == (50,)
    assert np.isfinite(pred).all()
    assert reg.apx_compile_info_["enabled"] is True
    assert len(weights) == len(reg._model.tree_info())


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


def test_leafwise_sparse_oblique_path_is_active():
    rng = np.random.default_rng(44)
    X = rng.normal(size=(300, 4))
    y = X[:, 0] + X[:, 1] + rng.normal(scale=0.1, size=300)

    reg = GTBoostRegressor(
        n_estimators=20,
        learning_rate=0.2,
        max_depth=3,
        grow_policy="leafwise",
        sparse_oblique_splits=True,
        interval_splits=False,
        apx=False,
        seed=44,
    )
    reg.fit(X[:220], y[:220])
    pred = reg.predict(X[220:])
    _, _, _, oblique, _ = reg._model.split_op_counts()

    assert oblique > 0
    assert pred.shape == (80,)
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
    assert np.isfinite(result.selection_score)
    assert result.selection_score >= result.best_score
    assert isinstance(result.complexity_key, tuple)
    assert result.best_rounds >= 1
    assert result.best_params["n_estimators"] == result.best_rounds


def test_classifier_encodes_nonzero_labels():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(90, 4))
    y = np.where(X[:, 0] + 0.3 * X[:, 1] > 0.0, 2, 1)

    clf = GTBoostClassifier(
        n_estimators=18,
        learning_rate=0.2,
        max_depth=2,
        apx=False,
        seed=19,
    )
    clf.fit(X[:70], y[:70], eval_set=[(X[70:], y[70:])])
    pred = clf.predict(X[70:])
    proba = clf.predict_proba(X[70:])

    assert set(clf._classes.tolist()) == {1, 2}
    assert set(np.unique(pred)).issubset({1, 2})
    assert proba.shape == (20, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_classifier_encodes_nonzero_multiclass_labels():
    rng = np.random.default_rng(20)
    X = rng.normal(size=(120, 5))
    cls = np.argmax(
        np.column_stack([X[:, 0], X[:, 1] - 0.2 * X[:, 2], -X[:, 0] - X[:, 1]]),
        axis=1,
    )
    y = np.asarray([10, 20, 30], dtype=int)[cls]

    clf = GTBoostClassifier(
        n_estimators=16,
        learning_rate=0.2,
        max_depth=2,
        apx=False,
        seed=20,
    )
    clf.fit(X[:90], y[:90], eval_set=[(X[90:], y[90:])])
    pred = clf.predict(X[90:])
    proba = clf.predict_proba(X[90:])

    assert set(clf._classes.tolist()) == {10, 20, 30}
    assert set(np.unique(pred)).issubset({10, 20, 30})
    assert proba.shape == (30, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


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


def test_vertical_init_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(31)
    X = rng.normal(size=(140, 5))
    y = np.sin(X[:, 0]) + 0.4 * X[:, 2] + rng.normal(scale=0.06, size=140)

    model = gtb.GTBoostModel(
        task="regression",
        n_estimators=18,
        learning_rate=0.12,
        max_depth=3,
        num_bins=32,
        interval_splits=False,
        vertical_init=True,
        vertical_init_cycles=2,
        seed=31,
    )
    model.fit(X[:100], y[:100])
    before = model.predict(X[100:])

    path = tmp_path / "vertical_model.gtboost"
    model.save_model(path)
    loaded = gtb.GTBoostModel.load_model(path)
    after = loaded.predict(X[100:])

    assert before.shape == (40,)
    assert np.isfinite(before).all()
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


def test_auto_interactions_are_deterministic_for_same_seed():
    rng = np.random.default_rng(29)
    X = rng.normal(size=(180, 8))
    y = (
        0.7 * X[:, 0]
        - 0.4 * X[:, 1]
        + 0.5 * X[:, 2] * X[:, 3]
        + rng.normal(scale=0.1, size=180)
    )

    params = dict(
        n_estimators=28,
        learning_rate=0.12,
        max_depth=3,
        subsample=0.85,
        colsample_bytree=0.75,
        auto_interactions=True,
        max_interaction_features=6,
        interval_splits=False,
        apx=False,
        seed=29,
    )
    a = GTBoostRegressor(**params).fit(X[:130], y[:130], eval_set=[(X[130:155], y[130:155])])
    b = GTBoostRegressor(**params).fit(X[:130], y[:130], eval_set=[(X[130:155], y[130:155])])

    assert np.allclose(a.predict(X[155:]), b.predict(X[155:]), atol=0.0, rtol=0.0)


def test_apx_weighting_aliases_predict():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(150, 5))
    y = (X[:, 0] + 0.4 * X[:, 1] + rng.normal(scale=0.2, size=150) > 0.1).astype(int)

    for weighting in ["uniform", "linear", "flat", "triangle", "gauss"]:
        clf = GTBoostClassifier(
            n_estimators=28,
            learning_rate=0.12,
            max_depth=2,
            apx=True,
            apx_weighting=weighting,
            apx_n_checkpoints=5,
            apx_min_frac=0.35,
            early_stopping_rounds=0,
            seed=21,
        ).fit(X[:110], y[:110])
        proba = clf.predict_proba(X[110:])
        assert proba.shape == (40, 2)
        assert np.all(np.isfinite(proba))
        assert np.allclose(proba.sum(axis=1), 1.0)


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

    # Self-score fallback path. (ramp param removed 2026-06-10 — inert in ablation.)
    X_reg = rng.normal(size=(150, 4))
    y_reg = X_reg[:, 0] * 0.6 + X_reg[:, 1] * X_reg[:, 2] * 0.1 + rng.normal(scale=0.05, size=150)
    reg = GTBoostRegressor(
        n_estimators=14,
        learning_rate=0.15,
        max_depth=3,
        categorical_geometry="raw",
        interval_splits=True,
        self_score_splits=True,
        apx=False,
        seed=26,
    ).fit(X_reg[:110], y_reg[:110])
    assert np.allclose(reg.predict(X_reg[110:]), reg.predict(X_reg[110:].copy()))
