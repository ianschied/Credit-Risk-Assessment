"""
Data loading, cleaning, and splitting for the Give Me Some Credit dataset.

Writes cleaned, split CSVs to data/processed/{train,val,test}.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw/cs-training.csv")
PROCESSED_DIR = Path("data/processed")

TARGET = "SeriousDlqin2yrs"

# Known data quality issues in this dataset (documented on the Kaggle
# competition forums) that are worth handling deliberately rather than
# silently -- this is a good thing to call out in your README/notes.
#
# - MonthlyIncome: ~20% missing
# - NumberOfDependents: small % missing
# - age: contains at least one row with age == 0 (data error)
# - NumberOfTime30-59DaysPastDueNotWorse / 60-89 / 90DaysLate: contain
#   sentinel values of 96 and 98, which are known data errors, not real
#   counts of 96/98 late payments
# - RevolvingUtilizationOfUnsecoredLines and DebtRatio: contain extreme
#   outliers (values in the thousands where the ratio should be ~0-2)


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Download cs-training.csv from "
            "https://www.kaggle.com/c/GiveMeSomeCredit/data and place it there."
        )
    df = pd.read_csv(path)
    # Kaggle's export includes an unnamed index column
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=unnamed_cols)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Missing value handling ---
    # MonthlyIncome: impute with median, keep a missingness flag since
    # "income not reported" may itself carry signal
    df["MonthlyIncome_missing"] = df["MonthlyIncome"].isna().astype(int)
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())

    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(
        df["NumberOfDependents"].median()
    )

    # --- Known sentinel/error values ---
    # 96 and 98 in the "times late" columns are data errors, not real counts.
    # Cap them at the 99th percentile of the remaining valid values instead
    # of dropping rows, to avoid losing the (rare) actual high-delinquency cases.
    late_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]
    for col in late_cols:
        valid_mask = df[col] < 90
        cap = df.loc[valid_mask, col].quantile(0.99)
        df[col] = np.where(df[col] >= 90, cap, df[col])

    # age == 0 is not a valid age; replace with median.
    # Cast to float first -- pandas raises on assigning a float median
    # into a strict int64 column.
    df["age"] = df["age"].astype(float)
    df.loc[df["age"] < 18, "age"] = df["age"].median()

    # --- Outlier capping (winsorize at 99th percentile) ---
    for col in ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio"]:
        cap = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=cap)

    return df


def split(df: pd.DataFrame, seed: int = 42):
    """Stratified 70/15/15 train/val/test split (class imbalance -> stratify)."""
    train, temp = train_test_split(
        df, test_size=0.3, stratify=df[TARGET], random_state=seed
    )
    val, test = train_test_split(
        temp, test_size=0.5, stratify=temp[TARGET], random_state=seed
    )
    return train, val, test


def main():
    df = load_raw()
    print(f"Loaded {len(df):,} rows")
    print(f"Default rate: {df[TARGET].mean():.2%}")

    df_clean = clean(df)
    train, val, test = split(df_clean)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print(f"Train: {len(train):,} rows ({train[TARGET].mean():.2%} default)")
    print(f"Val:   {len(val):,} rows ({val[TARGET].mean():.2%} default)")
    print(f"Test:  {len(test):,} rows ({test[TARGET].mean():.2%} default)")
    print(f"Saved to {PROCESSED_DIR}/")


if __name__ == "__main__":
    main()
