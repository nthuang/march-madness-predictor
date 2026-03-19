import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score, log_loss, classification_report,brier_score_loss
from scipy.stats import randint, uniform

# Load data
df = pd.read_csv('W_2026_train_aug.csv')
print("Dividing data...")

feature_cols = [col for col in df.columns if col.startswith("Diff_") or col.startswith("T2_") or col.startswith("T1_")]
print(feature_cols)
train = df[df['Season'] <= 2022]
test = df[df['Season'] >= 2023]

x_train = train[feature_cols].reset_index(drop=True)
x_test = test[feature_cols].reset_index(drop=True)
y_train = train['w'].reset_index(drop=True)
y_test = test['w'].reset_index(drop=True)
seasons_train = train['Season'].reset_index(drop=True)


# Each fold holds out one entire season; trains on all prior seasons (walk-forward)
def season_walk_forward_splits(seasons):
    unique_seasons = sorted(seasons.unique())
    for val_season in unique_seasons[1:]:  # skip the first season
        train_idx = np.where(seasons < val_season)[0]
        val_idx   = np.where(seasons == val_season)[0]
        yield train_idx, val_idx

cv_splits = list(season_walk_forward_splits(seasons_train))
print(f"Number of CV folds (seasons): {len(cv_splits)}")

# parameter search matrix
param_dist = {
    'n_estimators':      randint(200, 1200),
    'max_depth':         [4, 6, 8, 10, 12, None],
    'min_samples_leaf':  randint(2, 25),
    'min_samples_split': randint(2, 20),
    'max_features':      ['sqrt', 'log2', 0.3, 0.4, 0.5],
    'class_weight':      [None, 'balanced'],
}

base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=100,
    scoring={
        'log_loss': 'neg_log_loss',
        'brier':    'neg_brier_score',
        'accuracy': 'accuracy'
    },
    refit='log_loss',        
    cv=cv_splits,
    verbose=2,
    random_state=42,
    n_jobs=-1,
)
print("hyperparameter search")
search.fit(x_train, y_train)

# --- Average CV Scores Across All Folds (best model only) ---
best_idx = search.best_index_
results_df = pd.DataFrame(search.cv_results_)

for metric, label, negate in [
    ('log_loss', 'Log Loss',    True),
    ('brier',    'Brier Score', True),
    ('accuracy', 'Accuracy',    False),
]:
    fold_cols = [col for col in results_df.columns if col.startswith('split') and col.endswith(f'test_{metric}')]
    fold_scores = results_df.loc[best_idx, fold_cols].values.astype(float)
    if negate:
        fold_scores = -fold_scores

    print(f"\n--- {label} Across Folds (Best Model) ---")
    for i, score in enumerate(fold_scores):
        print(f"  Fold {i+1}: {score:.4f}")
    print(f"  Mean: {fold_scores.mean():.4f}")
    print(f"  Std:  {fold_scores.std():.4f}")


# --- Results ---
print(f"\nBest Log Loss (CV): {-search.best_score_:.4f}")
print(f"Best Params: {search.best_params_}")

# --- Evaluate best model on held-out test set ---
best_model = search.best_estimator_
probs = best_model.predict_proba(x_test)[:, 1]
preds = best_model.predict(x_test)

print(f"\n--- Test Set Performance ---")
print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
print(f"Log Loss:  {log_loss(y_test, probs):.4f}")
print(f"Brier Score:{brier_score_loss(y_test, probs):.4f}")

print(classification_report(y_test, preds))

# --- Save best model ---
joblib.dump(best_model, 'tuned_womens_model.pkl')
print("Saved tuned model to tuned_mens_model.pkl")

