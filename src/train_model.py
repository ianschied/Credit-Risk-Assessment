"""
Phase 2: LightGBM (improved) model with Optuna hyperparameter tuning.

"""

from pathlib import Path

import joblib
import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score

from src.data_prep import TARGET
from src.train_baseline import evaluate, load_split

MODELS_DIR = Path("models")

# Paper's reported LightGBM AUC, for reference in the printed comparison.
# See README for why this is a directional benchmark, not a target to match.
PAPER_LIGHTGBM_AUC = 0.96

N_TRIALS = 30  # keep modest -- this doesn't need to be exhaustive


def make_objective(X_train, y_train, X_val, y_val, scale_pos_weight):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "scale_pos_weight": scale_pos_weight,
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "n_estimators": 500,
        }
        model = lgb.LGBMClassifier(**params, random_state=42)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, proba)

    return objective


def main():
    train = load_split("train")
    val = load_split("val")

    feature_cols = [c for c in train.columns if c != TARGET]
    X_train, y_train = train[feature_cols], train[TARGET]
    X_val, y_val = val[feature_cols], val[TARGET]

    # LightGBM's native imbalance handling, computed from the training
    # split -- equivalent in spirit to the class_weight='balanced' used
    # for the LR baseline, but expressed the way LightGBM expects it.
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos
    print(f"scale_pos_weight = {scale_pos_weight:.2f} (class imbalance ratio)")

    print(f"\nRunning Optuna search ({N_TRIALS} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        make_objective(X_train, y_train, X_val, y_val, scale_pos_weight),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print(f"\nBest val AUC during search: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Refit final model with best params (fixed n_estimators + early
    # stopping already accounted for during the search; refit plainly
    # here for a clean saved artifact)
    best_params = {
        **study.best_params,
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "scale_pos_weight": scale_pos_weight,
        "n_estimators": 500,
    }
    final_model = lgb.LGBMClassifier(**best_params, random_state=42)
    final_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    # Reuse the SAME evaluate() function as the baseline, so the
    # comparison in the README table is apples-to-apples.
    # evaluate() expects a scaler; LightGBM doesn't need feature scaling,
    # so pass a pass-through "scaler".
    class NoOpScaler:
        def transform(self, X):
            return X

    train_auc, _ = evaluate(final_model, NoOpScaler(), X_train, y_train, "Train")
    val_auc, _ = evaluate(final_model, NoOpScaler(), X_val, y_val, "Validation")

    print(f"\nPaper's reported LightGBM AUC (proprietary data, reference only): {PAPER_LIGHTGBM_AUC}")
    print(f"This model's validation AUC: {val_auc:.4f}")
    print("\nCompare this against your baseline's validation AUC from")
    print("`python -m src.train_baseline` to get the LR -> LightGBM gap")
    print("for your README results table.")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(final_model, MODELS_DIR / "lightgbm.joblib")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.joblib")
    print(f"\nSaved model to {MODELS_DIR}/lightgbm.joblib")


if __name__ == "__main__":
    main()
