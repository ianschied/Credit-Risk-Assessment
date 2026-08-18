# Explainable Credit Risk Model

An explainable credit-default prediction project that reproduces the methodology of an academic paper on a public dataset, then serves it as an API with SHAP-based explanations and a demo UI.

## Paper

> de Lange, P. E., Melsom, B., Vennerød, C. B., & Westgaard, S. (2022). **Explainable AI for Credit Assessment in Banks.** *Journal of Risk and Financial Management, 15*(12), 556. https://doi.org/10.3390/jrfm15120556

The paper combined a **LightGBM** model with **SHAP** explanations to predict consumer loan default at a Norwegian bank, benchmarked against the bank's existing **Logistic Regression** model. LightGBM outperformed the LR baseline with an ROC AUC of 0.96 vs. 0.82 (a 17% improvement), and SHAP was used to explain both global feature importance and individual predictions.

**Note:** the original paper's dataset is proprietary bank data and is not publicly available. This project applies the same methodology (LR baseline → LightGBM → SHAP explainability) to the public **[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)** dataset instead. The target variable definitions line up closely: the paper defines default as 90+ days overdue on payment; Give Me Some Credit's target (`SeriousDlqin2yrs`) is defined as experiencing 90+ days past due delinquency within 2 years.

Results here are **not expected to match the paper's numbers exactly** — this is a methodology reproduction on different data, not an identical replication. The comparison worth reporting is the *relative* improvement of LightGBM over the LR baseline on this dataset, and how it stacks up against the *relative* improvement reported in the paper.

## Architecture

```
Kaggle CSV
    |
    v
data_prep.py  ->  train/val/test splits (data/processed/)
    |
    +--> train_baseline.py  ->  models/baseline_lr.joblib
    |
    +--> train_model.py     ->  models/lightgbm.joblib
                                     |
                                     v
                               explain.py (SHAP)
                                     |
                     +---------------+---------------+
                     v                                v
               api.py (FastAPI)               streamlit_app.py
               POST /predict                   interactive demo
               GET  /health                    (loads model directly)
```

## Project structure

```
credit-risk-xai/
|-- data/                  # raw + processed data (gitignored, not committed)
|-- notebooks/
|   `-- 01_eda.py          # EDA (cell-marked, run in Jupyter or as a script)
|-- scripts/
|   `-- generate_synthetic_data.py  # placeholder data for testing before Kaggle download
|-- src/
|   |-- data_prep.py       # loading, cleaning, splitting
|   |-- train_baseline.py  # logistic regression baseline
|   |-- train_model.py     # LightGBM + Optuna tuning
|   |-- explain.py         # SHAP explainability
|   `-- api.py             # FastAPI serving layer
|-- streamlit_app.py       # interactive demo UI
|-- tests/
|   |-- test_data_prep.py
|   `-- test_api.py
|-- reports/figures/       # SHAP plots land here (gitignored)
|-- models/                # saved model artifacts (gitignored)
|-- Dockerfile
|-- render.yaml            # Render deploy config
|-- Procfile                # generic deploy target (Railway etc.)
|-- Makefile
|-- requirements.txt
`-- README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Getting the data

1. Download `cs-training.csv` from [Kaggle: Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/data) (requires a free Kaggle account)
2. Place it at `data/raw/cs-training.csv`

**Don't have the Kaggle file yet?** `make synthetic-data` (or `python -m scripts.generate_synthetic_data`) writes a placeholder dataset with the same schema and similar-shaped signal, so you can run the whole pipeline immediately. It is **not real data** — swap in the real CSV before reporting any results.

## Running the pipeline

```bash
make data       # clean + split
make baseline   # logistic regression baseline
make model      # LightGBM + Optuna tuning
make explain    # SHAP summary plot + sanity check
make api        # FastAPI on localhost:8000
make demo       # Streamlit UI on localhost:8501
make test       # pytest
```

(Or run the underlying `python -m src.____` / `uvicorn` / `streamlit run` commands directly — see the Makefile.)

## Results

*(This project was scaffolded against a synthetic placeholder dataset — see note above. Numbers below are from that placeholder run and exist only to confirm the pipeline works end to end. Replace with your real Kaggle-data results before this goes on your resume.)*

| Model | Val AUC (synthetic placeholder data) |
|---|---|
| Logistic Regression (baseline) | 0.733 |
| LightGBM (this project, real data) | *fill in after running `make model` on the real dataset* |
| LightGBM (paper, reference only, proprietary data) | 0.96 |
| Logistic Regression (paper, reference only, proprietary data) | 0.82 |

A quick equivalent-logic check using sklearn's `HistGradientBoostingClassifier` (no tuning) on the synthetic data reached **val AUC 0.80** vs. the LR baseline's 0.73 — confirming the pipeline's imbalance handling and data flow produce a real, meaningful gap in the expected direction before you've even installed LightGBM.

## API

```
GET  /health           -> {"status": "ok", "model_loaded": true}
POST /predict
  body: applicant features (RevolvingUtilizationOfUnsecuredLines, age,
        NumberOfTime30-59DaysPastDueNotWorse, DebtRatio, MonthlyIncome,
        NumberOfOpenCreditLinesAndLoans, NumberOfTimes90DaysLate,
        NumberRealEstateLoansOrLines, NumberOfTime60-89DaysPastDueNotWorse,
        NumberOfDependents)
  returns: {"risk_score": 0.23, "risk_band": "medium", "top_reasons": [...]}
```

## Deployment

- **API:** `render.yaml` is set up for [Render](https://render.com)'s free tier — connect the repo and it deploys automatically. `Procfile` works for Railway/Heroku-style platforms too.
- **Demo:** [Streamlit Community Cloud](https://streamlit.io/cloud) — point it at `streamlit_app.py`, free tier.
- **Docker:** `docker build -t credit-risk-api .` / `docker run -p 8000:8000 credit-risk-api` (trains locally first — see note in `Dockerfile`).

## What was verified vs. not (read this before trusting the results)

This project was built in an environment without internet access, so parts of it were written and syntax-checked but **not execution-tested**:

**Actually run and verified:**
- `data_prep.py` — cleaning logic tested against both a synthetic pytest fixture and a full synthetic dataset run
- `train_baseline.py` — trained and evaluated on synthetic data (val AUC 0.733)
- The LightGBM/imbalance-handling *approach* used in `train_model.py` — validated via an equivalent-logic smoke test with sklearn's `HistGradientBoostingClassifier` (val AUC 0.80, meaningful gap over baseline)

**Written and syntax-checked (`python -m py_compile` passes) but not run**, because `lightgbm`, `shap`, `fastapi`, `streamlit`, `optuna`, and `pytest` weren't installable in that sandbox (no network access):
- `train_model.py`, `explain.py`, `api.py`, `streamlit_app.py`, `tests/test_api.py`
- `Dockerfile` (Docker itself wasn't available to test the build)

**Your first move after cloning this:** `pip install -r requirements.txt`, then `make data && make baseline && make model && make explain && make test` in order, checking output at each step. If something breaks, it's most likely in the untested files above — they're the highest-value places to look first.

## Limitations

*(Write this section last — it's what shows you understand the method, not just implemented it.)*

- Give Me Some Credit has far fewer, less granular features than the paper's proprietary daily-transaction data (no balance-volatility or account-behavior features), so a smaller LR to LightGBM gap than the paper's 17% would be expected even with a bug-free pipeline
- Class imbalance is handled via `class_weight='balanced'` (LR) and `scale_pos_weight` (LightGBM) rather than resampling (SMOTE, etc.) — worth discussing the tradeoff if asked
- SHAP explanations are computed per-request in the API; for higher request volume you'd want to cache or batch this
- *(add your own findings here once you've run it on real data)*
