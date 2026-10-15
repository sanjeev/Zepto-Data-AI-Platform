"""Shared cleaning rules for the Titanic analytics pipeline.

Both notebooks import from here so that 02_modeling continues from exactly the
same cleaning decisions made (and justified) in 01_eda.

Threshold rule (measured on the raw 891-row dataset):
  * embarked, embark_town  0.22% missing  (< 5%)   -> drop those 2 rows
  * age                    19.87% missing (5-30%)  -> impute
  * deck                   77.22% missing (> 30%)  -> too high to impute reliably;
                                                      encode "Unknown" as its own category
"""

from __future__ import annotations

import pandas as pd

DROP_ROWS_BELOW = 5.0    # % missing
IMPUTE_UP_TO = 30.0      # % missing


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Missing count / percentage for every column that has any, plus the rule it triggers."""
    pct = df.isna().mean() * 100
    rep = pd.DataFrame({"n_missing": df.isna().sum(), "pct_missing": pct.round(2)})
    rep = rep[rep.n_missing > 0].sort_values("pct_missing")
    rep["rule"] = pd.cut(
        rep.pct_missing,
        bins=[0, DROP_ROWS_BELOW, IMPUTE_UP_TO, 100],
        labels=["< 5%: drop rows", "5-30%: impute", "> 30%: drop column or 'missing' category"],
        include_lowest=True,
    )
    return rep


def structural_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning steps that learn nothing from the data (safe before a train/test split).

    * drop rows missing embarked / embark_town (< 5% rule)
    * deck -> 'Unknown' category (> 30% rule)
    """
    out = df.dropna(subset=["embarked", "embark_town"]).copy()
    out["deck"] = out["deck"].astype("object").fillna("Unknown")
    return out.reset_index(drop=True)


def impute_age(df: pd.DataFrame) -> pd.DataFrame:
    """EDA-stage age imputation (5-30% rule): median age within each sex x pclass group.

    Group medians respect the fact that 1st-class passengers were markedly older
    than 3rd-class ones, which a single global median would blur. The modeling
    notebook does NOT use this (it would leak test rows into the medians); there
    a SimpleImputer is fit on the training split only.
    """
    out = df.copy()
    out["age"] = out["age"].fillna(out.groupby(["sex", "pclass"])["age"].transform("median"))
    return out


def full_clean(df: pd.DataFrame) -> pd.DataFrame:
    return impute_age(structural_clean(df))
