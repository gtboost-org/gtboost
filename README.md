# GTBoost

Rust/Python gradient boosting for tabular data.

GTBoost combines histogram tree boosting with native NaN handling, optional interval split candidates, and optional PCF-lite categorical geometry. The project is alpha software: APIs may change, but the public surface is intentionally small.

## Install

```bash
pip install maturin
maturin develop --release
```

## Native API

```python
import gtboost as gtb

train = gtb.Dataset(train_df, label="target", categorical="auto")
valid = gtb.Dataset(valid_df, label="target", reference=train)

model = gtb.train(
    {
        "objective": "binary",
        "learning_rate": 0.05,
        "max_depth": 6,
        "categorical_geometry": "auto",  # raw or pcf_lite when useful
        "interval_splits": "auto",
        "random_state": 42,
    },
    train,
    valid_sets=[valid],
    num_boost_round=1000,
    early_stopping_rounds=100,
)

p = model.predict(test_df)          # probabilities for binary tasks
labels = model.predict_label(test_df)
```

## Sklearn API

```python
from gtboost import GTBoostClassifier, GTBoostRegressor

clf = GTBoostClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    cat_features=None,  # pandas category/object/string columns auto-detected
    categorical_geometry="pcf_lite",
    interval_splits=True,
    seed=42,
)

clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=100)
proba = clf.predict_proba(X_test)
```

## DataFrames

`gtboost.Dataset` preserves column names and encodes pandas categorical columns consistently. Use `reference=train` for validation/test data so unseen categories map to the same unknown/NaN path.

```python
dtrain = gtb.Dataset(df_train, label="target", categorical=["city", "segment"])
dtest = gtb.Dataset(df_test, reference=dtrain)
```

## Tuning

```python
from gtboost.tuner import tune_gtboost

result = tune_gtboost(
    X_train,
    y_train,
    task="binary",
    cat_features="auto",
    categorical_geometry_choices=["raw", "pcf_lite"],
    interval_splits="auto",
    n_trials=30,
    n_folds=3,
)

model = GTBoostClassifier(**result.best_params).fit(X_train, y_train)
```

## Alpha Scope

Stable-facing:

- `Dataset`, `train`, `Booster`
- `GTBoostClassifier`, `GTBoostRegressor`
- native NaN handling
- interval splits
- `categorical_geometry="raw"` and `"pcf_lite"`

Experimental:

- CLT categorical teacher paths
- broad research knobs in `GTBoostModel`
