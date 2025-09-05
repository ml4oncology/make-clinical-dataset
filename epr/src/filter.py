"""
Module to filter features and samples
"""

import logging
from collections.abc import Sequence
from typing import Optional
from warnings import simplefilter

import numpy as np
import pandas as pd
from make_clinical_dataset.epr.util import get_excluded_numbers
from make_clinical_dataset.shared.constants import EPR_DRUG_COLS
from ml_common.summary import get_nmissing

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)


###############################################################################
# General Filters
###############################################################################
def drop_highly_missing_features(
    df: pd.DataFrame, missing_thresh: float, keep_cols: Optional[list[str]] = None
) -> pd.DataFrame:
    """Drop features with high level of missingness

    Args:
        keep_cols: list of feature names to keep regardless of high missingness
    """
    nmissing = get_nmissing(df)
    mask = nmissing["Missing (%)"] > missing_thresh
    exclude_cols = nmissing.index[mask].drop(keep_cols, errors="ignore").tolist()
    msg = f"Dropping the following {len(exclude_cols)} features for missingness over {missing_thresh}%: {exclude_cols}"
    logger.info(msg)
    return df.drop(columns=exclude_cols)


def drop_samples_outside_study_date(
    df: pd.DataFrame, start_date: str = "2014-01-01", end_date: str = "2019-12-31"
) -> pd.DataFrame:
    mask = df["assessment_date"].between(start_date, end_date)
    get_excluded_numbers(df, mask, context=f" before {start_date} and after {end_date}")
    df = df[mask]
    return df
    
    
def drop_samples_with_no_targets(
    df: pd.DataFrame, targ_cols: Sequence[str], missing_val=None
) -> pd.DataFrame:
    if missing_val is None:
        mask = df[targ_cols].isnull()
    else:
        mask = df[targ_cols] == missing_val
    mask = ~mask.all(axis=1)
    get_excluded_numbers(df, mask, context=" with no targets")
    df = df[mask]
    return df


def keep_only_one_per_week(df: pd.DataFrame, date_col: str = "assessment_date") -> list[int]:
    """Keep only the first visit of a given week
    Drop all other sessions
    """
    if df.index.nunique() != len(df):
        raise ValueError("Make sure indices are unique")

    keep_idxs = []
    for mrn, group in df.groupby("mrn"):
        previous_date = pd.Timestamp.min
        for i, visit_date in group[date_col].items():
            if visit_date >= previous_date + pd.Timedelta(days=7):
                keep_idxs.append(i)
                previous_date = visit_date
    get_excluded_numbers(
        df, mask=df.index.isin(keep_idxs), context=" not first of a given week"
    )
    df = df.loc[keep_idxs]
    return df


###############################################################################
# Specialized Filters
###############################################################################
def drop_unused_drug_features(df: pd.DataFrame) -> pd.DataFrame:
    # use 0 as a placeholder for nans and inf
    df[EPR_DRUG_COLS] = df[EPR_DRUG_COLS].fillna(0).replace(np.inf, 0)

    # remove drugs given less than 10 times
    mask = (df[EPR_DRUG_COLS] != 0).sum() < 10
    exclude = mask.index[mask].tolist()
    logger.info(
        f"Removing the following features for drugs given less than 10 times: {exclude}"
    )
    df = df.drop(columns=exclude)

    return df
