"""
Phase 3: SHAP explainability for the LightGBM model.

Run as a script to generate the global summary plot + a sanity-check
local explanation:
    python -m src.explain

Provides explain_prediction(), which src/api.py calls at request time.

"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.train_baseline import load_split

MODELS_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")


def load_model_and_features():
    model = joblib.load(MODELS_DIR / "lightgbm.joblib")
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.joblib")
    return model, feature_cols


def compute_shap_values(model, X: pd.DataFrame):
    """TreeSHAP -- same method used in the reference paper, and the
    right choice here since it's exact and polynomial-time for tree
    ensembles (unlike KernelSHAP, which would be far too slow for a
    served API)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    return explainer, shap_values


def explain_prediction(model, explainer, feature_cols: list[str], applicant: dict, top_n: int = 5) -> dict:
    """Given a single applicant's features, return a risk score plus
    the top N SHAP contributors. This is what src/api.py calls per
    request -- keep it fast (single-row TreeSHAP call, not a full
    dataset pass).
    """
    X = pd.DataFrame([applicant])[feature_cols]
    risk_score = float(model.predict_proba(X)[0, 1])

    shap_values = explainer(X)
    contributions = list(zip(feature_cols, shap_values.values[0]))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    top_reasons = [
        {
            "feature": name,
            "value": applicant[name],
            "impact": round(float(impact), 4),
            "direction": "increases risk" if impact > 0 else "decreases risk",
        }
        for name, impact in contributions[:top_n]
    ]

    return {"risk_score": round(risk_score, 4), "top_reasons": top_reasons}


def main():
    model, feature_cols = load_model_and_features()
    test = load_split("test")
    X_test = test[feature_cols]

    print("Computing SHAP values on test set...")
    explainer, shap_values = compute_shap_values(model, X_test)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Global summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150)
    plt.close()
    print(f"Saved global SHAP summary plot to {FIGURES_DIR}/shap_summary.png")

    # Local explanation sanity check -- pick the highest-risk test
    # applicant and confirm the explanation is directionally sensible
    proba = model.predict_proba(X_test)[:, 1]
    highest_risk_idx = proba.argmax()
    applicant = X_test.iloc[highest_risk_idx].to_dict()

    result = explain_prediction(model, explainer, feature_cols, applicant)
    print(f"\nSanity check -- highest-risk applicant in test set:")
    print(f"Risk score: {result['risk_score']}")
    print("Top reasons:")
    for reason in result["top_reasons"]:
        print(f"  {reason['feature']} = {reason['value']} -> {reason['direction']} "
              f"(impact {reason['impact']:+.4f})")
    print("\nConfirm this makes intuitive sense (e.g. high utilization or")
    print("multiple late payments should show up as risk-increasing)")
    print("before relying on this for the API or README.")


if __name__ == "__main__":
    main()
