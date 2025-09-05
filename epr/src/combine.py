"""
Module to anchor/combine features or targets to the main dates
"""
from functools import partial
from typing import Callable

import pandas as pd
from make_clinical_dataset.epr.engineer import (
    get_days_since_last_event,
    get_perc_ideal_dose_given,
    get_visit_month_feature,
    get_years_diff,
)
from make_clinical_dataset.epr.preprocess.opis import clean_drug_name
from make_clinical_dataset.epr.util import get_excluded_numbers
from ml_common.util import split_and_parallelize


###############################################################################
# General Combiners
###############################################################################
def combine_meas_to_main_data(
    main: pd.DataFrame, 
    meas: pd.DataFrame, 
    main_date_col: str, 
    meas_date_col: str, 
    parallelize: bool = True,
    **kwargs
) -> pd.DataFrame:
    """Combine separate measurement data to the main dataset

    Both main and meas should have mrn and date columns
    """
    mask = main['mrn'].isin(meas['mrn'])
    worker = partial(measurement_stat_extractor, main_date_col=main_date_col, meas_date_col=meas_date_col, **kwargs)
    result = split_and_parallelize((main[mask], meas), worker) if parallelize else worker((main[mask], meas))
    if not result:
        return main
    result = pd.DataFrame(result).set_index('index')
    df = main.join(result)
    return df


def measurement_stat_extractor(
    partition, 
    main_date_col: str,
    meas_date_col: str,
    stats: list[str] | None = None, 
    stat_func: Callable | None = None,
    time_window: tuple[int, int] = (-5,0),
    include_meas_date: bool = False,
) -> list:
    """Extract either the first, last, sum, max, min, mean, or count of measurements (lab tests, symptom scores, etc) 
    taken within the time window (centered on each main date)

    Args:
        main_date_col: The column name of the main visit date
        meas_date_col: The column name of the measurement date
        time_window: The start and end of the window in terms of number of days after(+)/before(-) the main visit dates
        stat: What aggregate functions to use for the measurements taken within the time window. 
            Options are first, last, sum, max, min, avg, or count
        include_meas_date: If True, stores the date of first / last measurement
    """
    if stats is None:
        stats = ['max']
    main_df, meas_df = partition
    lower_limit, upper_limit = time_window

    results = []
    meas_groups = meas_df.groupby('mrn')
    for mrn, main_group in main_df.groupby('mrn'):
        meas_group = meas_groups.get_group(mrn)

        for main_idx, date in main_group[main_date_col].items():
            earliest_date = date + pd.Timedelta(days=lower_limit)
            latest_date = date + pd.Timedelta(days=upper_limit)

            mask = meas_group[meas_date_col].between(earliest_date, latest_date)
            if not mask.any(): 
                continue

            # if user provided their own stat function
            if stat_func is not None:
                data = stat_func(meas_group[mask])
            else:
                meas = meas_group[mask].drop(columns=['mrn'])
                data = _measurement_stat_extractor(meas, meas_date_col, stats, include_meas_date)
    
            data['index'] = main_idx
            results.append(data)
    
    return results


def _measurement_stat_extractor(
    meas: pd.DataFrame, 
    meas_date_col: str,
    stats: list[str] | None = None, 
    include_meas_date: bool = True
) -> dict:
    meas_dates = meas.pop(meas_date_col)
    data = {}
    if 'first' in stats:
        result = meas.iloc[0]
        result.index += '_FIRST'
        data.update(result.to_dict())
        if include_meas_date: 
            # TODO: support extraction of first measurement dates for each measurement column
            data[f'{meas_date_col}_FIRST'] = meas_dates.iloc[0]
    if 'last' in stats:
        result = meas.ffill().iloc[-1]
        result.index += '_LAST'
        data.update(result.to_dict())
        if include_meas_date: 
            # TODO: support extraction of last measurement dates for each measurement column
            data[f'{meas_date_col}_LAST'] = meas_dates.iloc[-1]
    if 'sum' in stats:
        result = meas.sum()
        result.index += '_SUM'
        data.update(result.to_dict())
    if 'max' in stats:
        idxs = meas.loc[:, ~meas.isnull().all()].idxmax()
        for col, idx in idxs.items():
            data[f'{col}_MAX'] = meas.loc[idx, col]
            if include_meas_date: 
                data[f'{col}_MAX_date'] = meas_dates[idx]
    if 'min' in stats:
        idxs = meas.loc[:, ~meas.isnull().all()].idxmin()
        for col, idx in idxs.items():
            data[f'{col}_MIN'] = meas.loc[idx, col]
            if include_meas_date:
                data[f'{col}_MIN_date'] = meas_dates[idx]
    if 'avg' in stats:
        result = meas.mean()
        result.index += '_AVG'
        data.update(result.to_dict())
    if 'count' in stats:
        result = meas.count()
        result.index += '_COUNT'
        data.update(result.to_dict())
    return data


# An alternate approach to retrieving first or last measurement within a time window
# More performant than measurement_stat_extractor
def merge_closest_measurements(
    main: pd.DataFrame, 
    meas: pd.DataFrame, 
    main_date_col: str,
    meas_date_col: str, 
    direction: str = 'backward',
    time_window: tuple[int, int] = (-5,0),
    merge_individually: bool = True,
    include_meas_date: bool = False
) -> pd.DataFrame:
    """Extract the closest measurements (lab tests, symptom scores, etc) prior to / after the main date 
    within a lookback / lookahead window and combine them to the main dataset

    Both main and meas should have mrn and date columns
    
    Args:
        main_date_col: The column name of the main visit date
        meas_date_col: The column name of the measurement date
        time_window: The start and end of the window in terms of number of days after(+)/before(-) the main visit dates
        direction: specifies whether to merge measurements before or after the main date. Either 'backward' or 'forward'
        merge_individually: If True, merges each measurement column separately
        include_meas_date: If True, include the date of the closest measurement that was merged
    """
    lower_limit, upper_limit = time_window
    if direction == 'backward':
        main['main_date'] = main[main_date_col] + pd.Timedelta(days=upper_limit)
    elif direction == 'forward':
        main['main_date'] = main[main_date_col] + pd.Timedelta(days=lower_limit)

    # pd.merge_asof uses binary search, requires input to be sorted by the time
    main = main.sort_values(by='main_date')
    meas = meas.sort_values(by=meas_date_col)
    meas[meas_date_col] = meas[meas_date_col].astype(main['main_date'].dtype) # ensure date types match

    merge_kwargs = dict(
        left_on='main_date', right_on=meas_date_col, by='mrn', direction=direction, allow_exact_matches=True,
        tolerance=pd.Timedelta(days=upper_limit - lower_limit)
    )

    if merge_individually:
        # merge each measurement column individually
        for col in meas.columns:
            if col in ['mrn', meas_date_col]: continue

            data_to_merge = meas.loc[meas[col].notnull(), ['mrn', meas_date_col, col]]

            # merges the closest row to main date while matching on mrn
            main = pd.merge_asof(main, data_to_merge, **merge_kwargs)
            
            if include_meas_date:
                # rename the date column
                main = main.rename(columns={meas_date_col: f'{col}_{meas_date_col}'})
            else:
                del main[meas_date_col]

    else:
        main = pd.merge_asof(main, meas, **merge_kwargs)

    del main['main_date']
    return main


###############################################################################
# Specific Combiners
###############################################################################
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
    # 0 - diagnosis date did not occur before assessment date
    # 1 - prior diagnosis but not the most recent diagnosis before assessment date
    # 2 - most recent diagnosis prior to assessment date
    cols = df.columns
    for category in ['cancer_site', 'morphology']:
        date_cols = cols[cols.str.contains(category)]
        diag_dates = df[date_cols]
        prior_to_assessment = diag_dates.lt(df[main_date_col], axis=0)
        most_recent_date = diag_dates[prior_to_assessment].max(axis=1)
        df[date_cols] = prior_to_assessment.astype(int)
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
    if df['mrn'].nunique() == 1:
        df['days_since_last_treatment'] = get_days_since_last_treatment(df)
    else:
        df['days_since_last_treatment'] = df.groupby('mrn', group_keys=False).apply(get_days_since_last_treatment)
    return df