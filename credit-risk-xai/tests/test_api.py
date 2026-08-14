"""
Tests for the FastAPI service.

"""

import pytest
from fastapi.testclient import TestClient

from src.api import app

VALID_APPLICANT = {
    "RevolvingUtilizationOfUnsecuredLines": 0.3,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.4,
    "MonthlyIncome": 5000,
    "NumberOfOpenCreditLinesAndLoans": 6,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 1,
}


@pytest.fixture
def client():
    with TestClient(app) as c:  # triggers lifespan startup/shutdown
        yield c


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_predict_valid_applicant_returns_200(client):
    response = client.post("/predict", json=VALID_APPLICANT)
    assert response.status_code == 200


def test_predict_response_shape(client):
    response = client.post("/predict", json=VALID_APPLICANT)
    body = response.json()
    assert "risk_score" in body
    assert "risk_band" in body
    assert "top_reasons" in body
    assert body["risk_band"] in {"low", "medium", "high"}


def test_predict_risk_score_in_valid_range(client):
    response = client.post("/predict", json=VALID_APPLICANT)
    score = response.json()["risk_score"]
    assert 0.0 <= score <= 1.0


def test_predict_top_reasons_not_empty(client):
    response = client.post("/predict", json=VALID_APPLICANT)
    reasons = response.json()["top_reasons"]
    assert len(reasons) > 0
    assert "feature" in reasons[0]
    assert "direction" in reasons[0]


def test_predict_missing_field_returns_422(client):
    incomplete = VALID_APPLICANT.copy()
    del incomplete["age"]
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_negative_value_returns_422(client):
    invalid = VALID_APPLICANT.copy()
    invalid["DebtRatio"] = -1
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_predict_underage_applicant_returns_422(client):
    invalid = VALID_APPLICANT.copy()
    invalid["age"] = 10
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_predict_high_risk_applicant_scores_higher(client):
    """Sanity check that the model responds sensibly to obviously
    higher-risk input -- not a strict correctness proof, but catches
    grossly broken wiring (e.g. features passed in the wrong order).
    """
    high_risk = VALID_APPLICANT.copy()
    high_risk.update(
        {
            "RevolvingUtilizationOfUnsecuredLines": 0.95,
            "NumberOfTimes90DaysLate": 4,
            "DebtRatio": 1.8,
        }
    )
    low_score = client.post("/predict", json=VALID_APPLICANT).json()["risk_score"]
    high_score = client.post("/predict", json=high_risk).json()["risk_score"]
    assert high_score > low_score
