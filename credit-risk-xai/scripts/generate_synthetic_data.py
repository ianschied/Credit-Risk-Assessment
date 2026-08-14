"""
Generates a synthetic, GMSC-shaped dataset for exercising the pipeline
locally without the real Kaggle download.

Run:
    python -m scripts.generate_synthetic_data
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT_PATH = Path("data/raw/cs-training.csv")


def generate(n: int = 20000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    revolving_util = np.clip(rng.gamma(shape=1.2, scale=0.3, size=n), 0, 3)
    age = np.clip(rng.normal(45, 14, size=n).round(), 21, 90)
    late_30_59 = rng.poisson(0.3, size=n)
    debt_ratio = np.clip(rng.gamma(shape=1.0, scale=0.4, size=n), 0, 4)
    monthly_income = np.clip(rng.lognormal(mean=8.4, sigma=0.6, size=n), 500, 50000)
    open_credit_lines = rng.poisson(7, size=n)
    late_90 = rng.poisson(0.08, size=n)
    real_estate_loans = rng.poisson(1, size=n)
    late_60_89 = rng.poisson(0.08, size=n)
    dependents = rng.poisson(0.8, size=n)

    # Default probability driven by a mix of linear effects plus genuinely
    # nonlinear structure a plain logistic regression can't represent
    # without manual feature engineering:
    #   - a U-shaped age effect (both very young and elderly customers
    #     are riskier than middle-aged ones -- non-monotonic)
    #   - an AND-type threshold interaction (high utilization AND high
    #     debt ratio together is much riskier than either alone)
    # This mirrors why boosted trees have real headroom over LR on the
    # actual dataset, rather than just adding unlearnable noise.
    age_ushape = ((age - 45) / 20) ** 2  # low near 45, high at the extremes
    high_risk_combo = ((revolving_util > 0.7) & (debt_ratio > 1.2)).astype(float)

    logit = (
        -4.6
        + 1.4 * revolving_util
        + 0.35 * late_30_59
        + 0.5 * debt_ratio
        + 1.0 * late_90
        + 0.6 * late_60_89
        + 0.9 * age_ushape
        + 2.5 * high_risk_combo
        - 0.00004 * monthly_income
        + rng.normal(0, 0.35, size=n)  # modest noise
    )
    prob_default = 1 / (1 + np.exp(-logit))
    target = rng.binomial(1, prob_default)

    df = pd.DataFrame(
        {
            "SeriousDlqin2yrs": target,
            "RevolvingUtilizationOfUnsecuredLines": revolving_util,
            "age": age.astype(int),
            "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
            "DebtRatio": debt_ratio,
            "MonthlyIncome": monthly_income,
            "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
            "NumberOfTimes90DaysLate": late_90,
            "NumberRealEstateLoansOrLines": real_estate_loans,
            "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
            "NumberOfDependents": dependents,
        }
    )

    # Inject the same known real-world data quality issues so data_prep's
    # cleaning logic has something realistic to work on:
    # ~20% missing income, ~3% missing dependents, ~0.3% age errors,
    # ~0.3% sentinel late-payment values (96/98)
    income_missing = rng.random(n) < 0.20
    df.loc[income_missing, "MonthlyIncome"] = np.nan

    dep_missing = rng.random(n) < 0.03
    df.loc[dep_missing, "NumberOfDependents"] = np.nan

    age_error = rng.random(n) < 0.003
    df.loc[age_error, "age"] = 0

    sentinel_rows = rng.random(n) < 0.003
    df.loc[sentinel_rows, "NumberOfTime30-59DaysPastDueNotWorse"] = 98
    df.loc[sentinel_rows, "NumberOfTimes90DaysLate"] = 98
    df.loc[sentinel_rows, "NumberOfTime60-89DaysPastDueNotWorse"] = 98

    return df


def main():
    df = generate()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} synthetic rows to {OUT_PATH}")
    print(f"Synthetic default rate: {df['SeriousDlqin2yrs'].mean():.2%}")
    print("\nReminder: this is placeholder data for pipeline testing only.")
    print("Replace with the real Kaggle file before reporting results.")


if __name__ == "__main__":
    main()
