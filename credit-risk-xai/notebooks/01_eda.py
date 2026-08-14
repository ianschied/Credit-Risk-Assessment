"""
Phase 1 EDA starter.

"""

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_prep import load_raw, TARGET

df = load_raw()
df.head()

# %% Class imbalance -- the central challenge called out in the paper
print(df[TARGET].value_counts(normalize=True))
sns.countplot(x=TARGET, data=df)
plt.title("Class balance: SeriousDlqin2yrs")
plt.show()

# %% Missing values
print(df.isna().mean().sort_values(ascending=False))

# %% Distributions of key features -- look for the known outliers
# (RevolvingUtilizationOfUnsecuredLines and DebtRatio should show extreme
# right-skew before cleaning; age should show a min of 0 before cleaning)
num_cols = df.select_dtypes("number").columns.drop(TARGET)
df[num_cols].describe().T

# %%
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for ax, col in zip(axes.flat, num_cols):
    df[col].hist(ax=ax, bins=50)
    ax.set_title(col, fontsize=9)
plt.tight_layout()
plt.show()

# %% Correlation with target -- gives you a preview of what LR/LightGBM
# will likely find important (compare this later to your SHAP output
# in Phase 3 -- differences are worth discussing in your README)
corr_with_target = df[num_cols].corrwith(df[TARGET]).sort_values(key=abs, ascending=False)
print(corr_with_target)

# %% Correlation heatmap between features (paper does the same thing --
# Figure 1 in the paper)
plt.figure(figsize=(10, 8))
sns.heatmap(df[num_cols].corr(), cmap="coolwarm", center=0, annot=False)
plt.title("Feature correlation heatmap")
plt.show()
