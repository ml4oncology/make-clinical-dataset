"""
Module to extract symptom deterioration labels
"""

from typing import Optional

from functools import partial

import pandas as pd

from ml_common.anchor import measurement_stat_extractor
from ml_common.constants import SYMP_COLS
from ml_common.util import split_and_parallelize

###############################################################################
# Events
###############################################################################
def get_event_labels(df: pd.DataFrame, event: pd.DataFrame):
    df = df.sort_values(by='assessment_date')
    event = event.sort_values(by='event_date')

    # merge the closest event row after the assessment date, matched on patients
    df = pd.merge_asof(
        df, event, left_on='assessment_date', right_on='event_date', by='mrn', direction='forward', 
        allow_exact_matches=False
    )

    return df.sort_values(by=['mrn', 'assessment_date'])


###############################################################################
# Symptoms
###############################################################################
def convert_to_binary_symptom_labels(
    df: pd.DataFrame, scoring_map: Optional[dict[str, int]] = None
) -> pd.DataFrame:
    """Convert label to 1 (positive), 0 (negative), or -1 (missing/exclude)

    Label is positive if symptom deteriorates (score increases) by X points
    """
    if scoring_map is None:
        scoring_map = {col: 3 for col in SYMP_COLS}

    for symp, pt in scoring_map.items():
        discrete_targ_col = f"target_{symp}_{pt}pt_change"
        change = df[f"target_{symp}"] - df[symp]
        missing_mask = change.isnull()
        df[discrete_targ_col] = (change >= pt).astype(int)
        df.loc[missing_mask, discrete_targ_col] = -1

        # If baseline score is alrady high, we exclude them
        df.loc[df[symp] > 10 - pt, discrete_targ_col] = -1
    return df


def get_symptom_labels(
    main_df: pd.DataFrame, 
    symp_df: pd.DataFrame, 
    lookahead_window: int = 30
) -> pd.DataFrame:
    """Extract labels for symptom deterioration within the next X days after visit date

    Args:
        symp_df: The processed symptom data from https://github.com/ml4oncology/make-clinical-dataset
        lookahead_window: The lookahead window in terms of days after visit date in which labels can be extracted
    """
    mask = main_df['mrn'].isin(symp_df['mrn'])
    worker = partial(
        measurement_stat_extractor, main_date_col='assessment_date', meas_date_col='survey_date', 
        time_window=(1, lookahead_window), stats=['max']
    )
    result = split_and_parallelize((main_df[mask], symp_df), worker)
    result = pd.DataFrame(result).set_index('index')
    result.columns = 'target_' + result.columns.str.replace('_MAX', '')
    main_df = main_df.join(result)
    return main_df
