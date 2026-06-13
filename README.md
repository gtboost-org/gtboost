# GTBoost

Gradient boosting for tabular data. Rust core, Python/sklearn API. Native NaN and categorical handling.

Alpha software — APIs may change.

## Install

```bash
pip install gtboost
```

From source:

```bash
git clone https://github.com/gtboost-org/gtboost.git
cd gtboost
pip install maturin
maturin develop --release
```

## Regression — California housing

```python
from gtboost import GTBoostRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, y = fetch_california_housing(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

model = GTBoostRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, seed=0)
model.fit(Xtr, ytr, eval_set=[(Xte, yte)], early_stopping_rounds=50)

print(mean_squared_error(yte, model.predict(Xte)) ** 0.5)
```

## Classification — diabetes (binarized)

```python
import numpy as np
from gtboost import GTBoostClassifier
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
y = (y > np.median(y)).astype(int)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

clf = GTBoostClassifier(n_estimators=500, learning_rate=0.05, max_depth=3, seed=0)
clf.fit(Xtr, ytr, eval_set=[(Xte, yte)], early_stopping_rounds=50)

proba = clf.predict_proba(Xte)[:, 1]
```

## Main features

All are constructor kwargs (defaults shown). Descriptions are mechanical, not benchmark claims.

| kwarg | default | what it does |
|---|---|---|
| `honest` | `False` | fit split structure and leaf values on disjoint halves of the data |
| `stein_leaves` | `False` | positive-part shrinkage of each tree's leaf vector toward the global step |
| `vertical_init` | `False` | fit per-feature linear effects before the trees |
| `monotone_constraints` | `[]` | per-feature monotonicity: `1` up, `-1` down, `0` none |
| `interval_splits` | `False` | bounded split candidates `a <= x <= b` |
| `sparse_oblique_splits` | `False` | 2-feature diagonal split candidates |
| `grow_policy` | `"depthwise"` | `depthwise` / `leafwise` / `oblivious` / `adaptive` |
| `huber_delta` | `0.0` | Huber loss for regression (`0.0` = squared error) |
| `n_refine` | `0` | post-fit leaf-value refit passes |
| `apx` | `True` | average predictions across boosting-trajectory checkpoints |
| `cat_features` | `None` | mark categorical columns; pandas category/object/string auto-detected |

Noisy, small data (diabetes) — structure-honest fit with leaf shrinkage and a linear prior:

```python
reg = GTBoostRegressor(honest=True, stein_leaves=True, vertical_init=True, seed=0)
reg.fit(Xtr, ytr)
```

Monotone constraint (California, target non-decreasing in feature 0, `MedInc`):

```python
reg = GTBoostRegressor(monotone_constraints=[1, 0, 0, 0, 0, 0, 0, 0], seed=0)
reg.fit(Xtr, ytr)
```

## Categorical features

Pandas `category`/`object`/`string` columns are detected automatically. Pass `cat_features` to override. Fit and predict on DataFrames with the same columns so categories map consistently.

```python
clf = GTBoostClassifier(cat_features=["city", "segment"], seed=0)
clf.fit(df_train, y_train)
proba = clf.predict_proba(df_test)
```

## Tuning

```python
from gtboost.tuner import tune_gtboost

result = tune_gtboost(Xtr, ytr, task="regression", n_trials=30, n_folds=3)
model = GTBoostRegressor(**result.best_params).fit(Xtr, ytr)
```

## Refine

Post-hoc joint leaf re-solve (regression, binary, multiclass): re-fits all leaf
values against the final residual, shrunk toward the greedy values, behind an
evidence gate. See `REFINE_2.md`.

## Scope

Stable: `GTBoostRegressor`, `GTBoostClassifier`, `Dataset`, native NaN and
categorical handling, interval splits. A native `gtboost.train` / `Dataset` API
also exists.

Experimental: research knobs on `GTBoostModel` — see `GTBOOST_FEATURES.md`.
