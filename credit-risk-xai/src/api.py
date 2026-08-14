"""
Phase 4: FastAPI serving layer.

Run locally:
    uvicorn src.api:app --reload

Then:
    curl -X POST http://localhost:8000/predict \\
      -H "Content-Type: application/json" \\
      -d '{"RevolvingUtilizationOfUnsecuredLines": 0.3, "age": 45,
           "NumberOfTime30-59DaysPastDueNotWorse": 0, "DebtRatio": 0.4,
           "MonthlyIncome": 5000, "NumberOfOpenCreditLinesAndLoans": 6,
           "NumberOfTimes90DaysLate": 0, "NumberRealEstateLoansOrLines": 1,
           "NumberOfTime60-89DaysPastDueNotWorse": 0, "NumberOfDependents": 1}'

"""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.explain import explain_prediction

MODELS_DIR = Path("models")

# Populated at startup (see lifespan below) so the model/explainer are
# loaded once per process, not re-loaded from disk on every request.
state: dict = {}


class ApplicantFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    revolving_utilization: float = Field(
        alias="RevolvingUtilizationOfUnsecuredLines",
        ge=0,
        description="Total balance on credit cards/lines divided by credit limits",
    )
    age: int = Field(ge=18, le=110)
    late_30_59: int = Field(alias="NumberOfTime30-59DaysPastDueNotWorse", ge=0)
    debt_ratio: float = Field(alias="DebtRatio", ge=0)
    monthly_income: float = Field(alias="MonthlyIncome", ge=0)
    open_credit_lines: int = Field(alias="NumberOfOpenCreditLinesAndLoans", ge=0)
    late_90: int = Field(alias="NumberOfTimes90DaysLate", ge=0)
    real_estate_loans: int = Field(alias="NumberRealEstateLoansOrLines", ge=0)
    late_60_89: int = Field(alias="NumberOfTime60-89DaysPastDueNotWorse", ge=0)
    dependents: int = Field(alias="NumberOfDependents", ge=0)


class Reason(BaseModel):
    feature: str
    value: float
    impact: float
    direction: str


class PredictionResponse(BaseModel):
    risk_score: float
    risk_band: str
    top_reasons: list[Reason]


def risk_band(score: float) -> str:
    if score < 0.40:
        return "low"
    if score < 0.70:
        return "medium"
    return "high"

@asynccontextmanager
async def lifespan(app: FastAPI):
    model = joblib.load(MODELS_DIR / "lightgbm.joblib")
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.joblib")
    explainer = shap.TreeExplainer(model)

    state["model"] = model
    state["feature_cols"] = feature_cols
    state["explainer"] = explainer
    yield
    state.clear()


app = FastAPI(title="Explainable Credit Risk API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict", response_model=PredictionResponse)
def predict(applicant: ApplicantFeatures):
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    applicant_dict = applicant.model_dump(by_alias=True)
    # MonthlyIncome_missing is an engineered feature added during training
    # (src/data_prep.py) -- API applicants always supply income directly,
    # so this is always 0 at inference time.
    applicant_dict["MonthlyIncome_missing"] = 0

    result = explain_prediction(
        state["model"], state["explainer"], state["feature_cols"], applicant_dict
    )
    result["risk_band"] = risk_band(result["risk_score"])
    return result
