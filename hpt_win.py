import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV


train_df = pd.read_csv("data/M_2026_train_aug.csv")

train_years = list(range(2003, 2025))   
train_split = train_df[train_df["Season"].isin(train_years)].reset_index(drop=True)
hold_split = train_df[train_df["Season"] == 2025].reset_index(drop=True)

id_cols = ["Season","Team1","Team2","w","margin"]

X_train = train_split.drop(columns=id_cols)
y_train = train_split["w"].astype(int).values
groups = train_split["Season"].values

X_hold = hold_split.drop(columns=id_cols)
y_hold = hold_split["w"].astype(int).values

logo = LeaveOneGroupOut()

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=1,
)

param_distributions = {
    "n_estimators": [600, 1000, 1500, 2500, 3500],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
    "min_child_weight": [1, 2, 5, 10, 15, 20],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "gamma": [0.0, 0.25, 0.5, 1.0, 2.0],
    "reg_lambda": [0.0, 0.5, 1.0, 2.0, 5.0],
    "reg_alpha": [0.0, 0.1, 0.25, 0.5, 1.0],
}

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=50,
    scoring="neg_brier_score",
    cv=logo,
    verbose=2,
    n_jobs=-1,
    random_state=42,
    refit=True,
)

search.fit(X_train, y_train, groups=groups)

print("\nBest CV Brier:", -search.best_score_)
print("Best params:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")

best_model = search.best_estimator_
p_hold = best_model.predict_proba(X_hold)[:, 1] 

print("\n2025 holdout Brier Score:", brier_score_loss(y_hold, p_hold))
print("2025 holdout Log Loss:", log_loss(y_hold, p_hold))