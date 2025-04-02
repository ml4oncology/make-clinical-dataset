"""
Module to combine features
"""
from functools import partial

import pandas as pd
from ml_common.anchor import combine_meas_to_main_data
from ml_common.util import get_excluded_numbers

from .feat_eng import (
    get_days_since_last_event,
    get_perc_ideal_dose_given,
    get_visit_month_feature,
    get_years_diff,
)
from .preprocess.opis import clean_drug_name


def combine_demographic_to_main_data(
    main: pd.DataFrame, 
    demographic: pd.DataFrame, 
    main_date_col: str,
) -> pd.DataFrame:
    """
    Args:
        main_date_col: The column name of the main asessment date
    """
    df = pd.merge(main, demographic, on='mrn', how='left')

    # exclude patients with missing birth date
    mask = df['date_of_birth'].notnull()
    get_excluded_numbers(df, mask, context=' with missing birth date')
    df = df[mask]

    # exclude patients under 18 years of age
    df['age'] = get_years_diff(df, main_date_col, 'date_of_birth')
    mask = df['age'] >= 18
    get_excluded_numbers(df, mask, context=' under 18 years of age')
    df = df[mask]

    # convert each cancer site / morphology datetime columns into ternary indicator variables
    # 0 - diagnosis date did not occur before treatment date
    # 1 - prior diagnosis but not the most recent diagnosis before treatment date
    # 2 - most recent diagnosis prior to treatment date
    cols = df.columns
    for category in ['cancer_site', 'morphology']:
        date_cols = cols[cols.str.contains(category)]
        diag_dates = df[date_cols]
        prior_to_trt = diag_dates.lt(df['treatment_date'], axis=0)
        most_recent_date = diag_dates[prior_to_trt].max(axis=1)
        df[date_cols] = prior_to_trt.astype(int)
        for col, dates in diag_dates.items():
            df.loc[dates == most_recent_date, col] = 2

    return df.copy()


def combine_treatment_to_main_data(
    main: pd.DataFrame, 
    treatment: pd.DataFrame, 
    main_date_col: str, 
    time_window: tuple[int, int] = (-28,0),
    parallelize: bool = True,
) -> pd.DataFrame:
    """Combine treatment information to the main data
    For drug dosage features, add up the treatment drug dosages of the past x days for each drug
    For other treatment features, forward fill the features available in the past x days 

    Args:
        main_date_col: The column name of the main asessment date
    """
    cols = treatment.columns
    drug_cols = cols[cols.str.startswith('drug_')].tolist()
    treatment_drugs = treatment[drug_cols + ['mrn', 'treatment_date']] # treatment drug dosage features
    treatment_feats = treatment.drop(columns=drug_cols) # other treatment features
    df = combine_meas_to_main_data(
        main, treatment_feats, main_date_col, 'treatment_date', stats=['last'], time_window=time_window, 
        parallelize=parallelize, include_meas_date=True
    )
    df = combine_meas_to_main_data(
        df, treatment_drugs, main_date_col, 'treatment_date', stats=['sum'], time_window=time_window, 
        parallelize=parallelize, include_meas_date=False
    )
    df.columns = df.columns.str.replace('_LAST', '').str.replace('_SUM', '')
    return df


def combine_perc_dose_to_main_data(main: pd.DataFrame, included_drugs: pd.DataFrame) -> pd.DataFrame:
    """Combine percentage of ideal dose given to main data 
    And remove the raw dosages features (regimen dose and given dose)

    NOTE: The given dose is already set ~2 days in advance prior to treatment date (i.e. no data leakage)
    """
    # create drug to dose formula map
    included_drugs['name'] = [clean_drug_name(name)[0] for name in included_drugs['name']]
    included_drugs = included_drugs.drop_duplicates()
    assert not any(included_drugs['name'].duplicated())
    drug_to_dose_formula_map = dict(included_drugs[['name', 'recommended_dose_formula']].to_numpy())

    # combine the percentage of ideal dose given features
    given_dose_over_ideal_dose = get_perc_ideal_dose_given(main, drug_to_dose_formula_map)
    df = main.join(given_dose_over_ideal_dose)

    # remove the raw dosage features
    cols = df.columns
    df = df.drop(columns=cols[cols.str.startswith('drug_')])

    return df


def combine_event_to_main_data(
    main: pd.DataFrame, 
    event: pd.DataFrame, 
    main_date_col: str, 
    event_date_col: str, 
    event_name: str,
    lookback_window: int = 5,
    parallelize: bool = True,
) -> pd.DataFrame:
    """Combine features extracted from event data (emergency department visits, hospitalization, etc) to the main 
    dataset

    Args:
        main_date_col: The column name of the main visit date
        event_date_col: The column name of the event date
        lookback_window: The lookback window in terms of number of years from treatment date to extract event features
    """
    # Compute features
    stat_func = partial(_event_stat_func, event_date_col=event_date_col)
    df = combine_meas_to_main_data(
        main=main, meas=event, main_date_col=main_date_col, meas_date_col=event_date_col, parallelize=parallelize,
        time_window=(-lookback_window * 365, 0), stat_func=stat_func,
    )

    # 1. number of days since closest event prior to main visit date
    df[f'days_since_prev_{event_name}'] = (df[main_date_col] - df.pop('prev_event_date')).dt.days

    # 2. number of events within the lookback window 
    df[f'num_prior_{event_name}s_within_{lookback_window}_years'] = df.pop('num_prior_events')

    return df


def _event_stat_func(df, event_date_col):
    return {
        'num_prior_events': len(df),
        'prev_event_date': df[event_date_col].iloc[-1]
    }


def add_engineered_features(df, date_col: str = 'treatment_date') -> pd.DataFrame:
    df = get_visit_month_feature(df, col=date_col)
    df['days_since_starting_treatment'] = (df[date_col] - df['first_treatment_date']).dt.days
    get_days_since_last_treatment = partial(
        get_days_since_last_event, main_date_col=date_col, event_date_col='treatment_date'
    )
    df['days_since_last_treatment'] = df.groupby('mrn', group_keys=False).apply(get_days_since_last_treatment)
    return df