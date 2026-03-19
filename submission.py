import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

ID_COLS = ["Season", "Team1", "Team2", "w", "margin"]

#best paramters from hyper paramater tuning
XGB_PARAMS = {
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

#XGBoost win probability model
def fit_winprob_model(train_aug_path, xgb_params):
    df = pd.read_csv(train_aug_path)

    X_all = df.drop(columns=ID_COLS, errors="ignore")
    y_win = df["w"].astype(int).to_numpy()
    seasons = df["Season"].to_numpy()

    best_iters = []
    oof_p = np.zeros(len(df), dtype=float)

    for s in sorted(df["Season"].unique()):
        tr_idx = seasons != s
        va_idx = seasons == s

        dtrain = xgb.DMatrix(X_all.loc[tr_idx], label=y_win[tr_idx], missing=np.nan)
        dval = xgb.DMatrix(X_all.loc[va_idx], label=y_win[va_idx], missing=np.nan)

        bst = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=1500,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=100,
            verbose_eval=False
        )

        best_iter = int(bst.best_iteration)
        best_iters.append(best_iter)

        p_hat = bst.predict(dval, iteration_range=(0, best_iter + 1))
        oof_p[va_idx] = np.clip(p_hat, 0.0, 1.0)

    oof_p_clipped = np.clip(oof_p, 1e-15, 1 - 1e-15)
    brier = brier_score_loss(y_win, oof_p_clipped)
    ll = log_loss(y_win, oof_p_clipped, labels=[0, 1])
    oof_pred = (oof_p >= 0.5).astype(int)      
    acc = accuracy_score(y_win, oof_pred)

    final_rounds = int(np.mean(best_iters)) + 1
    final_rounds = max(50, min(final_rounds, 5000))

    dtrain_full = xgb.DMatrix(X_all, label=y_win, missing=np.nan)
    final_bst = xgb.train(
        params=xgb_params,
        dtrain=dtrain_full,
        num_boost_round=final_rounds,
        evals=[(dtrain_full, "train")],
        verbose_eval=False,
    )

    print(f"\n[{train_aug_path}] Brier Score = {brier:.6f}")
    print(f"[{train_aug_path}] LogLoss = {ll:.6f}")
    print(f"[{train_aug_path}] Accuracy = {acc:.6f}")

    return final_bst, list(X_all.columns), final_rounds

#generate win probabilities for every possible matchup
def predict_from_matchup_features(matchup_path, model, feature_cols, num_rounds):
    df = pd.read_csv(matchup_path)

    X = df.drop(columns=["ID"], errors="ignore")

    X = X.reindex(columns=feature_cols)

    dmat = xgb.DMatrix(X, missing=np.nan)

    p = model.predict(dmat, iteration_range=(0, num_rounds))
    p = np.clip(p, 0.0, 1.0)

    return pd.DataFrame({"ID": df["ID"].to_numpy(), "Pred": p})



m_model, m_cols, m_rounds = fit_winprob_model("data/M_2026_train_aug.csv", XGB_PARAMS)
w_model, w_cols, w_rounds = fit_winprob_model("data/W_2026_train_aug.csv", XGB_PARAMS)

out_m = predict_from_matchup_features("data/M_2026_matchups_features.csv", m_model, m_cols, m_rounds)
out_w = predict_from_matchup_features("data/W_2026_matchups_features.csv", w_model, w_cols, w_rounds)

submission = pd.concat([out_m, out_w], ignore_index=True).sort_values("ID")
submission.to_csv("data/2026_submission.csv", index=False)

print("\nWrote 2026_submission.csv rows:", len(submission))

M_teams = pd.read_csv("mmlm2026/MTeams.csv")
W_teams = pd.read_csv("mmlm2026/WTeams.csv")
m_id_to_name = dict(zip(M_teams["TeamID"], M_teams["TeamName"]))
w_id_to_name = dict(zip(W_teams["TeamID"], W_teams["TeamName"]))

def parse_id(id_str):
    s, t1, t2 = id_str.split("_")
    return int(s), int(t1), int(t2)

#used to create a separate file repalces team id with team names
def add_names(pred_df, id_to_name, league):
    out = pred_df.copy()

    parsed = out["ID"].apply(parse_id)
    out["Season"] = parsed.apply(lambda x: x[0])
    out["TeamA_ID"] = parsed.apply(lambda x: x[1])
    out["TeamB_ID"] = parsed.apply(lambda x: x[2])

    out["TeamA"] = out["TeamA_ID"].map(id_to_name)
    out["TeamB"] = out["TeamB_ID"].map(id_to_name)

    out.insert(0, "League", league)
    return out[["League", "Season", "TeamA", "TeamB", "Pred", "ID"]]

readable_m = add_names(out_m, m_id_to_name, "M")
readable_w = add_names(out_w, w_id_to_name, "W")

readable = pd.concat([readable_m, readable_w], ignore_index=True)
readable.to_csv("data/2026_submission_with_names.csv", index=False)

print("Wrote data/2026_submission_with_names.csv rows:", len(readable))
