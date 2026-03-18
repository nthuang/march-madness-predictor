import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, brier_score_loss
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

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

results = []
oof_margin_hat = np.zeros(len(df), dtype=float)

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


res = pd.DataFrame(results).sort_values("Season")
print(res.to_string(index=False))
print("\nCV mean RMSE:", res["rmse"].mean(), "| std:", res["rmse"].std())
print("CV mean MAE: ", res["mae"].mean(),  "| std:", res["mae"].std())

plt.figure()
plt.plot(res["Season"], res["rmse"], marker="o")
plt.title("Margin model RMSE by season (LOSO)")
plt.xlabel("Season")
plt.ylabel("RMSE (points)")
plt.xticks(res["Season"], rotation=45)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(res["Season"], res["mae"], marker="o")
plt.title("Margin model MAE by season (LOSO)")
plt.xlabel("Season")
plt.ylabel("MAE (points)")
plt.xticks(res["Season"], rotation=45)
plt.tight_layout()
plt.show()

y_true = y_margin.astype(float)
y_pred = oof_margin_hat.astype(float)

rmse_oof = np.sqrt(np.mean((y_true - y_pred) ** 2))
mae_oof  = np.mean(np.abs(y_true - y_pred))

plt.figure(figsize=(7, 6))
plt.scatter(y_pred, y_true, s=14, alpha=0.25)

# y = x reference line
lo = min(y_pred.min(), y_true.min())
hi = max(y_pred.max(), y_true.max())
plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)

plt.xlabel("Predicted margin")
plt.ylabel("Actual margin")
plt.title(f"Predicted vs Actual Margin\nRMSE={rmse_oof:.2f}, MAE={mae_oof:.2f}")
plt.tight_layout()
plt.show()

p_oof_logistic = np.zeros(len(df))
p_oof_spline = np.zeros(len(df))

for s in sorted(df["Season"].unique()):
    tr = seasons != s
    va = seasons == s

    cal_fold = LogisticRegression(max_iter=1000)
    cal_fold.fit(oof_margin_hat[tr].reshape(-1,1), y_win[tr])
    p_oof_logistic[va] = cal_fold.predict_proba(oof_margin_hat[va].reshape(-1,1))[:,1]

    x = oof_margin_hat[tr]
    y = y_win[tr].astype(float)
    order = np.argsort(x)

    spline = UnivariateSpline(x[order], y[order], k=5) 

    p = spline(oof_margin_hat[va])
    p_oof_spline[va] = np.clip(p, 0.0, 1.0)

print("\n=== Cross-fit metrics ===")
print("Logistic Brier:", brier_score_loss(y_win, p_oof_logistic))
print("Logistic LogLoss:", log_loss(y_win, np.clip(p_oof_logistic, 1e-15, 1-1e-15)))

print("LogLoss:", log_loss(y_win, np.clip(p_oof_spline, 1e-15, 1-1e-15)))
print("Brier:", brier_score_loss(y_win, p_oof_spline))

cal_all = LogisticRegression(max_iter=1000)
cal_all.fit(oof_margin_hat.reshape(-1, 1), y_win)

x_all = oof_margin_hat
order = np.argsort(x_all)
spline_all = UnivariateSpline(x_all[order], y_win.astype(float)[order], k=5)


print("\nFinal logistic params (fit on all data):")
print("a (slope):", float(cal_all.coef_[0][0]))
print("b (intercept):", float(cal_all.intercept_[0]))


m_grid = np.linspace(oof_margin_hat.min(), oof_margin_hat.max(), 300)

p_grid_log = cal_all.predict_proba(m_grid.reshape(-1, 1))[:, 1]
p_grid_spl = spline_all(m_grid)
p_grid_spl = np.clip(p_grid_spl, 0.0, 1.0)

plt.figure()
plt.scatter(oof_margin_hat, p_oof_spline, alpha=0.2, s=10, label="predited probs")
plt.plot(m_grid, p_grid_log, label="Logistic calibration")
plt.plot(m_grid, p_grid_spl, label="Spline calibration ")
plt.axvline(0, linestyle="--", linewidth=1)
plt.title("Win margin → Win probability")
plt.xlabel("Predicted win margin")
plt.ylabel("Win probability (Team1 wins)")
plt.legend()
plt.tight_layout()
plt.show()

p = p_oof_spline  

tmp = pd.DataFrame({"m": oof_margin_hat, "p": p, "y": y_win})
tmp["bin"] = pd.cut(tmp["m"], bins=25)

xdf = tmp.groupby("bin").agg(
    m_mid=("m", "mean"),
    emp_win_rate=("y", "mean"),
    avg_pred_prob=("p", "mean"),
    n=("y", "size")
).dropna()

plt.figure()
plt.plot(xdf["m_mid"], xdf["emp_win_rate"], label="Empirical win rate")
plt.plot(xdf["m_mid"], xdf["avg_pred_prob"], label="Avg predicted prob")
plt.axvline(0, linestyle="--", linewidth=1)
plt.xlabel("Predicted margin (binned)")
plt.ylabel("Win probability")
plt.title("Calibration check: empirical vs predicted")
plt.legend()
plt.tight_layout()
plt.show()
