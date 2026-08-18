"""
Baseline model: Logistic Regression, matching the comparison point used
in the reference paper.

"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src.data_prep import TARGET, PROCESSED_DIR

MODELS_DIR = Path("models")

# Paper's reported LR AUC, for reference in the printed comparison.
# See README for why this is a directional benchmark, not a target to match.
PAPER_LR_AUC = 0.82


def load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.data_prep` first."
        )
    return pd.read_csv(path)


def evaluate(model, scaler, X, y, label: str):
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y, proba)
    pr_auc = average_precision_score(y, proba)

    print(f"\n--- {label} ---")
    print(f"ROC AUC: {auc:.4f}")
    print(f"PR AUC:  {pr_auc:.4f}")
    print("\nConfusion matrix (threshold=0.5):")
    print(confusion_matrix(y, preds))
    print("\nClassification report:")
    print(classification_report(y, preds, digits=3))

    return auc, pr_auc


def main():
    train = load_split("train")
    val = load_split("val")

    feature_cols = [c for c in train.columns if c != TARGET]
    X_train, y_train = train[feature_cols], train[TARGET]
    X_val, y_val = val[feature_cols], val[TARGET]

    # Logistic regression needs scaled features; class_weight='balanced'
    # addresses the class imbalance instead of naive resampling.
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    )
    model.fit(X_train_scaled, y_train)

    train_auc, _ = evaluate(model, scaler, X_train, y_train, "Train")
    val_auc, _ = evaluate(model, scaler, X_val, y_val, "Validation")

    print(f"\nPaper's reported LR AUC (proprietary data, reference only): {PAPER_LR_AUC}")
    print(f"This baseline's validation AUC: {val_auc:.4f}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "baseline_lr.joblib")
    joblib.dump(scaler, MODELS_DIR / "baseline_scaler.joblib")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.joblib")
    print(f"\nSaved model + scaler to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
