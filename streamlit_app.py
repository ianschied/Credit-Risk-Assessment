"""
Phase 5: Streamlit demo.

Run locally:
    streamlit run streamlit_app.py

This loads the model directly (not via the API) to keep the demo
self-contained and deployable on its own (e.g. Streamlit Community
Cloud) without also needing the FastAPI service running. If you'd
rather demo the full client/server split, point this at the API
instead -- see the commented-out `call_api()` alternative below.

NOTE: written and syntax-checked but NOT execution-tested in the
sandbox this project was scaffolded in (streamlit/shap aren't
installed there, no network to pip install). Run locally after
`pip install -r requirements.txt`.
"""

from pathlib import Path

import joblib
import pandas as pd
import shap
import streamlit as st

from src.explain import explain_prediction

MODELS_DIR = Path("models")

PRESETS = {
    "Low risk": {
        "RevolvingUtilizationOfUnsecuredLines": 0.1,
        "age": 52,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.2,
        "MonthlyIncome": 7000,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 2,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1,
    },
    "Medium risk": {
        "RevolvingUtilizationOfUnsecuredLines": 0.5,
        "age": 35,
        "NumberOfTime30-59DaysPastDueNotWorse": 1,
        "DebtRatio": 0.6,
        "MonthlyIncome": 3500,
        "NumberOfOpenCreditLinesAndLoans": 5,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2,
    },
    "High risk": {
        "RevolvingUtilizationOfUnsecuredLines": 0.95,
        "age": 28,
        "NumberOfTime30-59DaysPastDueNotWorse": 3,
        "DebtRatio": 1.7,
        "MonthlyIncome": 1800,
        "NumberOfOpenCreditLinesAndLoans": 3,
        "NumberOfTimes90DaysLate": 2,
        "NumberRealEstateLoansOrLines": 0,
        "NumberOfTime60-89DaysPastDueNotWorse": 1,
        "NumberOfDependents": 3,
    },
}


@st.cache_resource
def load_model():
    model = joblib.load(MODELS_DIR / "lightgbm.joblib")
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.joblib")
    explainer = shap.TreeExplainer(model)
    return model, explainer, feature_cols


def main():
    st.set_page_config(page_title="Explainable Credit Risk", page_icon="\U0001F4B3")
    st.title("Explainable Credit Risk")
    st.caption(
        "Reproduces the methodology of de Lange et al. (2022), "
        "*Explainable AI for Credit Assessment in Banks*, on the public "
        "Give Me Some Credit dataset. LightGBM risk score + SHAP-based "
        "explanation of the top drivers."
    )

    model, explainer, feature_cols = load_model()

    st.subheader("Applicant")

    preset_name = st.radio("Quick-load an example", list(PRESETS.keys()), horizontal=True)
    preset = PRESETS[preset_name]

    col1, col2 = st.columns(2)
    with col1:
        revolving_util = st.slider(
            "Revolving utilization (balance / credit limit)",
            0.0, 2.0, float(preset["RevolvingUtilizationOfUnsecuredLines"]), 0.01,
        )
        age = st.number_input("Age", 18, 100, int(preset["age"]))
        late_30_59 = st.number_input(
            "Times 30-59 days late", 0, 20, int(preset["NumberOfTime30-59DaysPastDueNotWorse"])
        )
        debt_ratio = st.slider("Debt ratio", 0.0, 3.0, float(preset["DebtRatio"]), 0.01)
        monthly_income = st.number_input(
            "Monthly income ($)", 0, 100_000, int(preset["MonthlyIncome"])
        )
    with col2:
        open_credit_lines = st.number_input(
            "Open credit lines/loans", 0, 50, int(preset["NumberOfOpenCreditLinesAndLoans"])
        )
        late_90 = st.number_input(
            "Times 90+ days late", 0, 20, int(preset["NumberOfTimes90DaysLate"])
        )
        real_estate_loans = st.number_input(
            "Real estate loans/lines", 0, 20, int(preset["NumberRealEstateLoansOrLines"])
        )
        late_60_89 = st.number_input(
            "Times 60-89 days late", 0, 20, int(preset["NumberOfTime60-89DaysPastDueNotWorse"])
        )
        dependents = st.number_input("Dependents", 0, 15, int(preset["NumberOfDependents"]))

    applicant = {
        "RevolvingUtilizationOfUnsecuredLines": revolving_util,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": monthly_income,
        "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
        "NumberOfTimes90DaysLate": late_90,
        "NumberRealEstateLoansOrLines": real_estate_loans,
        "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
        "NumberOfDependents": dependents,
        "MonthlyIncome_missing": 0,
    }

    if st.button("Assess risk", type="primary"):
        result = explain_prediction(model, explainer, feature_cols, applicant)

        st.subheader("Result")
        score_pct = result["risk_score"] * 100
        st.metric("Predicted default risk", f"{score_pct:.1f}%")

        st.subheader("Top factors")
        reasons_df = pd.DataFrame(result["top_reasons"])
        st.bar_chart(reasons_df.set_index("feature")["impact"])
        st.dataframe(reasons_df, hide_index=True, use_container_width=True)

    st.divider()
    st.caption(
        "Educational project, not a real lending decision tool. "
        "See the paper: https://doi.org/10.3390/jrfm15120556"
    )


# --- Alternative: call the FastAPI service instead of loading the model
# directly, if you're demoing the deployed client/server split:
#
# import requests
# API_URL = "http://localhost:8000"
#
# def call_api(applicant: dict) -> dict:
#     response = requests.post(f"{API_URL}/predict", json=applicant, timeout=10)
#     response.raise_for_status()
#     return response.json()


if __name__ == "__main__":
    main()
