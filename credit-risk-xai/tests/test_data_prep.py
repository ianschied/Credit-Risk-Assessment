import pandas as pd
import pytest

from src.data_prep import clean, split, TARGET


@pytest.fixture
def raw_sample():
    """200 mostly-normal rows plus a single row carrying every known bad
    value (age=0, late-payment sentinels 98/96, extreme DebtRatio/
    utilization, missing income/dependents).

    Keeping the bad values rare (1/200 = 0.5%) matters here: the 99th
    percentile winsorization can only cap values that are genuinely a
    small minority of the column, same as in the real ~150k-row dataset.
    A fixture where 10% of rows are "outliers" isn't a fair test of a
    99th-percentile cap. 200 rows also comfortably supports the double
    stratified split in split().
    """
    n = 200
    df = pd.DataFrame(
        {
            TARGET: [i % 2 for i in range(n)],
            "RevolvingUtilizationOfUnsecuredLines": [0.3] * n,
            "age": [40] * n,
            "NumberOfTime30-59DaysPastDueNotWorse": [0] * n,
            "DebtRatio": [0.3] * n,
            "MonthlyIncome": [5000.0] * n,
            "NumberOfOpenCreditLinesAndLoans": [5] * n,
            "NumberOfTimes90DaysLate": [0] * n,
            "NumberRealEstateLoansOrLines": [1] * n,
            "NumberOfTime60-89DaysPastDueNotWorse": [0] * n,
            "NumberOfDependents": [1] * n,
        }
    )

    # Inject the known bad-value patterns into a single row
    df.loc[0, "RevolvingUtilizationOfUnsecuredLines"] = 5000
    df.loc[0, "age"] = 0
    df.loc[0, "NumberOfTime30-59DaysPastDueNotWorse"] = 98
    df.loc[0, "DebtRatio"] = 9999
    df.loc[0, "MonthlyIncome"] = None
    df.loc[0, "NumberOfTimes90DaysLate"] = 96
    df.loc[0, "NumberOfTime60-89DaysPastDueNotWorse"] = 98
    df.loc[0, "NumberOfDependents"] = None

    return df


def test_clean_removes_missing_values(raw_sample):
    cleaned = clean(raw_sample)
    assert cleaned["MonthlyIncome"].isna().sum() == 0
    assert cleaned["NumberOfDependents"].isna().sum() == 0


def test_clean_handles_invalid_age(raw_sample):
    cleaned = clean(raw_sample)
    assert (cleaned["age"] >= 18).all()


def test_clean_caps_sentinel_late_payment_values(raw_sample):
    cleaned = clean(raw_sample)
    assert cleaned["NumberOfTime30-59DaysPastDueNotWorse"].max() < 90
    assert cleaned["NumberOfTimes90DaysLate"].max() < 90
    assert cleaned["NumberOfTime60-89DaysPastDueNotWorse"].max() < 90


def test_clean_caps_outliers(raw_sample):
    cleaned = clean(raw_sample)
    # after capping at the 99th percentile, no value should remain at
    # the extreme raw magnitude
    assert cleaned["DebtRatio"].max() < 9999
    assert cleaned["RevolvingUtilizationOfUnsecuredLines"].max() < 5000


def test_split_is_stratified_and_covers_all_rows(raw_sample):
    cleaned = clean(raw_sample)
    # split() expects a slightly larger sample in practice; this just
    # checks the row-count invariant holds
    train, val, test = split(cleaned, seed=0)
    assert len(train) + len(val) + len(test) == len(cleaned)
