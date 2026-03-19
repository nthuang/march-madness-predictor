import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, brier_score_loss
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

#load data
df = pd.read_csv("data/M_2025_train_aug.csv")

ID_COLS = ["Season","Team1","Team2","w","margin"]
TARGET = "margin"

X_all = df.drop(columns=ID_COLS)
y_margin = df["margin"].astype(int).values
y_win = df["w"].astype(int).values
seasons = df["Season"].values

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "seed": 42,
    "tree_method": "hist",
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "max_depth": 2,
    "min_child_weight": 2,
    "gamma": 0.25,
    "lambda": 2.0,
    "alpha": 0.25,
    "eta": 0.01,
}

#store season based cv fold results
results = []
oof_margin_hat = np.zeros(len(df), dtype=float)

#leave one season out cv
for s in sorted(df["Season"].unique()):
    tr_idx = seasons != s
    va_idx = seasons == s

    dtrain = xgb.DMatrix(X_all.loc[tr_idx], label=y_margin[tr_idx], missing=np.nan)
    dval   = xgb.DMatrix(X_all.loc[va_idx], label=y_margin[va_idx], missing=np.nan)

    bst = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=1500,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=100,
        verbose_eval=False
    )

    best_iter = bst.best_iteration
    mhat = bst.predict(dval, iteration_range=(0, best_iter + 1))
    oof_margin_hat[va_idx] = mhat

    mse = mean_squared_error(y_margin[va_idx], mhat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_margin[va_idx], mhat)
    results.append({"Season": int(s), "best_iter": int(best_iter), "rmse": rmse, "mae": mae})

#results
res = pd.DataFrame(results).sort_values("Season")
print(res.to_string(index=False))
print("\nCV mean RMSE:", res["rmse"].mean(), "| std:", res["rmse"].std())
print("CV mean MAE: ", res["mae"].mean(),  "| std:", res["mae"].std())

