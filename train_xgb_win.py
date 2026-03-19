import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, brier_score_loss
import matplotlib.pyplot as plt

#load data
df = pd.read_csv("data/M_2025_train_aug.csv")

ID_COLS = ["Season","Team1","Team2","w","margin"]
TARGET = "w"

X_all = df.drop(columns=ID_COLS)
y_win = df["w"].astype(int).values
seasons = df["Season"].values

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
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

results = []
#leave one season out cv
for s in sorted(df["Season"].unique()):
    tr_idx = seasons != s
    va_idx = seasons == s

    dtrain = xgb.DMatrix(X_all.loc[tr_idx], label=y_win[tr_idx], missing=np.nan)
    dval   = xgb.DMatrix(X_all.loc[va_idx], label=y_win[va_idx], missing=np.nan)

    bst = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=1500,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=100,
        verbose_eval=False
    )

    best_iter = int(bst.best_iteration) 
    yhat_val = bst.predict(dval, iteration_range=(0, best_iter + 1))

    brier = brier_score_loss(y_win[va_idx], yhat_val)
    ll = log_loss(y_win[va_idx], yhat_val)

    results.append({
        "Season": int(s),
        "best_iter": best_iter,
        "Brier Score": float(brier),
        "Log Loss": float(ll)
    })

#results
results_df = pd.DataFrame(results).sort_values("Season")
print(results_df.to_string(index=False))
print("\nCV mean LogLoss:", results_df["Log Loss"].mean(skipna=True))
print("CV mean Brier:  ", results_df["Brier Score"].mean(skipna=True))
