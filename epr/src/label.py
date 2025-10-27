"""
Module to extract labels
"""
import pandas as pd
from make_clinical_dataset.epr.combine import (
    combine_meas_to_main_data,
    merge_closest_measurements,
)
from make_clinical_dataset.shared.constants import (
    CTCAE_CONSTANTS,
    MAP_CTCAE_LAB,
    SYMP_COLS,
)


###############################################################################
# Death
###############################################################################
def get_death_labels(df: pd.DataFrame, lookahead_window: int | list[int] = 30) -> pd.DataFrame:
    if isinstance(lookahead_window, int):
        lookahead_window = [lookahead_window]

    ghost_mask = df['last_seen_date'] > df['date_of_death']

    for days in lookahead_window:
        max_date = df['assessment_date'] + pd.Timedelta(days=days)
        censored = df['date_of_death'].isna() & (df['last_seen_date'] < max_date)
        label = df['date_of_death'] < max_date
        df[f'target_death_in_{days}d'] = label.astype(int)

        # exclude censored patients and patients seen after death if labeled positive
        df.loc[censored | (ghost_mask & (label == 1)), f'target_death_in_{days}d'] = -1

    return df


###############################################################################
# Emergency Department Visits
###############################################################################
def get_ED_labels(
    df: pd.DataFrame, 
    event: pd.DataFrame, 
    lookahead_window: int | list[int] = 30
) -> pd.DataFrame:
    if isinstance(lookahead_window, int):
        lookahead_window = [lookahead_window]

    event['ED_date'] = event['event_date']
    df = merge_closest_measurements(
        df, event, main_date_col='assessment_date', meas_date_col='event_date', 
        direction='forward', time_window=(0, max(lookahead_window))
    )
    df = df.rename(columns={'ED_date': 'target_ED_date'})

    for days in lookahead_window:
        max_date = df['assessment_date'] + pd.Timedelta(days=days)
        censored = df['target_ED_date'].isna() & (df['last_seen_date'] < max_date)
        immediate = df['target_ED_date'] < (df['assessment_date'] + pd.Timedelta(days=1))
        df[f'target_ED_{days}d'] = (df['target_ED_date'] < max_date).astype(int)
        # exclude censored patients or immediate visits
        df.loc[censored | immediate, f'target_ED_{days}d'] = -1

    return df.sort_values(by=['mrn', 'assessment_date'])


###############################################################################
# Symptom Deterioration
###############################################################################
def get_symptom_labels(
    main_df: pd.DataFrame, 
    symp_df: pd.DataFrame, 
    lookahead_window: int = 30,
    scoring_map: dict[str, int] | None = None
) -> pd.DataFrame:
    """Extract labels for symptom deterioration

    Label is positive if symptom deteriorates (score increases) by X points within the lookahead window

    Args:
        symp_df: The processed symptom data from https://github.com/ml4oncology/make-clinical-dataset
        lookahead_window: The lookahead window in terms of days after visit date in which labels can be extracted
    """
    if scoring_map is None:
        scoring_map = {col: 3 for col in SYMP_COLS}

    # get the maximum symptom score within lookahead window
    df = combine_meas_to_main_data(
        main_df, symp_df, 'assessment_date', 'survey_date', time_window=(1, lookahead_window), stats=['max'], 
    )
    df.columns = [f'target_{col.lower()}' if "_MAX" in col else col for col in df.columns]

    # compute binary labels: 1 (positive), 0 (negative), or -1 (missing/exclude)
    for symp, pt in scoring_map.items():
        discrete_targ_col = f"target_{symp}_{pt}pt_change"
        change = df[f"target_{symp}_max"] - df[symp]
        missing_mask = change.isnull()
        df[discrete_targ_col] = (change >= pt).astype(int)
        df.loc[missing_mask, discrete_targ_col] = -1

        # If baseline score is alrady high, we exclude them
        df.loc[df[symp] > 10 - pt, discrete_targ_col] = -1

    return df


###############################################################################
# Abnormal Lab Findings / CTCAE (Common Terminology Criteria for Adverse Events)
###############################################################################
def get_CTCAE_labels(
    df_main: pd.DataFrame,
    df_lab: pd.DataFrame,
    lookahead_window: int = 30
) -> pd.DataFrame:
    """Compute lookahead lab values and apply the threshold functions to generate CTCAE grade labels
    """
    # Get the minimum / maximum lab test values within lookahead window, prior to next treatment session
    df = combine_meas_to_main_data(
        main=df_main, meas=df_lab, main_date_col='assessment_date', meas_date_col='obs_date', 
        time_window=(1, lookahead_window), stat_func=_CTCAE_stat_func,
    )
    df.columns = [f'target_{col.lower()}' if "_MIN" in col or "_MAX" in col else col for col in df.columns]

    # Apply threshold functions for each grade
    for ctcae, constants in CTCAE_CONSTANTS.items():
        lab_col = MAP_CTCAE_LAB[ctcae]
        for grade in [2, 3]:
            target_col = f'target_{ctcae}_grade{grade}plus'

            if ctcae in ['hemoglobin', 'neutrophil', 'platelet']:
                lab_lookahead_val = df[f'target_{lab_col}_min']
                threshold = constants[f'grade{grade}plus']
                df[target_col] = (lab_lookahead_val < threshold).astype(int)
                df.loc[lab_lookahead_val.isnull(), target_col] = -1
            else:
                lab_lookahead_val = df[f'target_{lab_col}_max']
                if ctcae == 'AKI':
                    lab_base_val = df[lab_col].clip(upper=constants['ULN']).fillna(constants['ULN'])
                else:
                    lab_base_val = df[lab_col].clip(lower=constants['ULN']).fillna(constants['ULN'])
                threshold = constants[f'grade{grade}plus'] * lab_base_val
                df[target_col] = (lab_lookahead_val > threshold).astype(int)
                df.loc[lab_lookahead_val.isnull(), target_col] = -1

    return df


def _CTCAE_stat_func(df):
    data = {}
    for col in ['hemoglobin', 'platelet', 'neutrophil']:
        if df[col].isnull().all():
            continue
        idx = df[col].idxmin()
        data[f'{col}_MIN'] = df.loc[idx, col]
        data[f'{col}_MIN_date'] = df.loc[idx, 'obs_date']

    for col in ['creatinine', 'alanine_aminotransferase', 'aspartate_aminotransferase', 'total_bilirubin']:
        if df[col].isnull().all():
            continue
        idx = df[col].idxmax()
        data[f'{col}_MAX'] = df.loc[idx, col]
        data[f'{col}_MAX_date'] = df.loc[idx, 'obs_date']

    return data