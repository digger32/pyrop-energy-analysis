"""18-model rolling-origin benchmark with optional GPU acceleration.

The benchmark covers three families of regressors:

- Baselines: Persistence-at-lag-24, Seasonal-Naive-at-lag-168.
- Linear (6): LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, Huber.
- Trees / kernels / NN (10): KNN, DecisionTree, RandomForest, ExtraTrees,
  AdaBoost, GradientBoosting, XGBoost, LightGBM, CatBoost, SVR (RBF), MLP.

Validation: TimeSeriesSplit with 5 folds (rolling-origin). Metrics: MAE,
RMSE, MAPE, sMAPE, R^2, adjusted R^2. The leaderboard is sorted by mean
RMSE across folds.

GPU acceleration is enabled when CUDA is detected: XGBoost is configured
with `device='cuda'` + `tree_method='hist'`, CatBoost with `task_type='GPU'`.
LightGBM stays on CPU because the default Colab build is not compiled with
GPU support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import time, subprocess

import numpy as np
import pandas as pd

from sklearn.linear_model    import (LinearRegression, Ridge, Lasso,
                                     ElasticNet, BayesianRidge, HuberRegressor)
from sklearn.ensemble        import (RandomForestRegressor, ExtraTreesRegressor,
                                     AdaBoostRegressor, GradientBoostingRegressor)
from sklearn.tree            import DecisionTreeRegressor
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.svm             import SVR
from sklearn.neural_network  import MLPRegressor
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics         import (mean_absolute_error, mean_squared_error,
                                     mean_absolute_percentage_error, r2_score)

try:
    from xgboost  import XGBRegressor;          HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMRegressor;         HAS_LGB = True
except ImportError:
    HAS_LGB = False
try:
    from catboost import CatBoostRegressor;     HAS_CAT = True
except ImportError:
    HAS_CAT = False


SEED = 42


def detect_gpu() -> Optional[str]:
    """Return the GPU device name if CUDA is available, else None."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return None


@dataclass
class Bundle:
    """A model bundle: name, estimator, scaling flag, family tag."""
    name: str
    estimator: Any
    needs_scaling: bool = True
    family: str = "ml"


class _LagBaseline:
    """Trivial baseline that predicts a fixed lag column."""
    def __init__(self, lag_col: str): self.lag_col = lag_col
    def fit(self, X, y): return self
    def predict(self, X): return X[self.lag_col].values


def make_models(lag24: str = "P_lag_24",
                lag168: str = "P_lag_168",
                use_gpu: Optional[bool] = None) -> list[Bundle]:
    """Return the full list of 18 model bundles configured for the benchmark."""
    if use_gpu is None:
        use_gpu = detect_gpu() is not None

    M: list[Bundle] = []
    if lag24:
        M.append(Bundle("Persistence_lag24", _LagBaseline(lag24), False, "baseline"))
    if lag168:
        M.append(Bundle("SeasonalNaive_lag168", _LagBaseline(lag168), False, "baseline"))

    M += [
        Bundle("LinearReg",     LinearRegression(),                                              family="linear"),
        Bundle("Ridge",         Ridge(alpha=1.0, random_state=SEED),                             family="linear"),
        Bundle("Lasso",         Lasso(alpha=0.001, max_iter=10000, random_state=SEED),           family="linear"),
        Bundle("ElasticNet",    ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000, random_state=SEED), family="linear"),
        Bundle("BayesianRidge", BayesianRidge(),                                                 family="linear"),
        Bundle("Huber",         HuberRegressor(max_iter=500),                                    family="linear"),
        Bundle("KNN",           KNeighborsRegressor(n_neighbors=10),                             family="ml"),
        Bundle("DecisionTree",  DecisionTreeRegressor(max_depth=10, random_state=SEED), False,   "ml"),
        Bundle("RandomForest",  RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED), False, "ml"),
        Bundle("ExtraTrees",    ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=SEED),  False, "ml"),
        Bundle("AdaBoost",      AdaBoostRegressor(n_estimators=200, random_state=SEED), False,   "ml"),
        Bundle("GradBoosting",  GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                    learning_rate=0.05, random_state=SEED), False,               "ml"),
        Bundle("SVR_RBF",       SVR(kernel="rbf", C=10.0, epsilon=0.05),                         family="ml"),
        Bundle("MLP",           MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500,
                                    early_stopping=True, random_state=SEED),                    family="ml"),
    ]

    if HAS_XGB:
        kw = dict(n_estimators=600, max_depth=6, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8,
                  random_state=SEED, verbosity=0, tree_method="hist")
        if use_gpu: kw["device"] = "cuda"
        else:        kw["n_jobs"] = -1
        M.append(Bundle("XGBoost", XGBRegressor(**kw), False, "ml"))

    if HAS_LGB:
        M.append(Bundle("LightGBM",
                        LGBMRegressor(n_estimators=600, num_leaves=63, learning_rate=0.05,
                                      subsample=0.8, colsample_bytree=0.8,
                                      random_state=SEED, n_jobs=-1, verbose=-1),
                        False, "ml"))

    if HAS_CAT:
        kw = dict(iterations=600, depth=6, learning_rate=0.05,
                  random_seed=SEED, verbose=False)
        if use_gpu:
            kw["task_type"] = "GPU"; kw["devices"] = "0"
        M.append(Bundle("CatBoost", CatBoostRegressor(**kw), False, "ml"))

    return M


def _smape(y_true, y_pred):
    den = (np.abs(y_true) + np.abs(y_pred)) / 2
    m = den != 0
    return float(np.mean(np.abs(y_true[m] - y_pred[m]) / den[m]) * 100)


def _adj_r2(r2: float, n: int, p: int) -> float:
    return 1 - (1 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else float("nan")


def _metrics(y_true, y_pred, n_features: int) -> dict:
    return {
        "MAE":       float(mean_absolute_error(y_true, y_pred)),
        "RMSE":      float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_pct":  float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "sMAPE_pct": _smape(y_true, y_pred),
        "R2":        float(r2_score(y_true, y_pred)),
        "R2_adj":    _adj_r2(r2_score(y_true, y_pred), len(y_true), n_features),
    }


def cv_evaluate(X: pd.DataFrame, y: pd.Series, bundle: Bundle, n_splits: int = 5):
    """Five-fold rolling-origin cross-validation for one model bundle."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds, fits, preds = [], [], []
    for fold, (tr, te) in enumerate(tscv.split(X)):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        try:
            est = (Pipeline([("sc", StandardScaler()), ("m", bundle.estimator)])
                   if bundle.needs_scaling else bundle.estimator)
            t0 = time.time(); est.fit(Xtr, ytr); fits.append(time.time() - t0)
            t1 = time.time(); pred = est.predict(Xte); preds.append(time.time() - t1)
            mt = _metrics(yte.values, np.asarray(pred), Xtr.shape[1])
            mt["fold"] = fold
            folds.append(mt)
        except Exception as exc:
            print(f"   [{bundle.name} fold {fold}] {exc}")
    if not folds:
        return None
    fdf = pd.DataFrame(folds)
    return {
        "name": bundle.name, "family": bundle.family,
        "MAE_mean":  fdf["MAE"].mean(),  "MAE_std":  fdf["MAE"].std(),
        "RMSE_mean": fdf["RMSE"].mean(), "RMSE_std": fdf["RMSE"].std(),
        "MAPE_mean": fdf["MAPE_pct"].mean(),
        "sMAPE_mean":fdf["sMAPE_pct"].mean(),
        "R2_mean":   fdf["R2"].mean(),    "R2_std":   fdf["R2"].std(),
        "R2_adj_mean": fdf["R2_adj"].mean(),
        "fit_time_mean":  float(np.mean(fits)),
        "pred_time_mean": float(np.mean(preds)),
        "fold_table": fdf,
    }


def run_benchmark(X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """Run all 18 models on (X, y) and return (leaderboard, full_results, bundles)."""
    bundles = make_models()
    rows, full = [], {}
    for b in bundles:
        print(f"  -> {b.name}")
        r = cv_evaluate(X, y, b, n_splits=n_splits)
        if r:
            full[b.name] = r
            rows.append({k: v for k, v in r.items() if k != "fold_table"})
    leaderboard = pd.DataFrame(rows).sort_values("RMSE_mean").reset_index(drop=True)
    return leaderboard, full, {b.name: b for b in bundles}
