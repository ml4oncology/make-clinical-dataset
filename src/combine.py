"""
Module to combine features
"""
from functools import partial

from tqdm import tqdm
import pandas as pd

from ml_common.anchor import combine_feat_to_main_data
from ml_common.util import get_excluded_numbers, split_and_parallelize

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
    main_date_col: str
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

    # convert each cancer site / morphology datetime columns into binary indicator variables based on whether diagnosis
    # date occured before treatment date
    cols = df.columns
    cols = cols[cols.str.contains('cancer_site|morphology')]
    # TODO: find out why df[cols] = df[cols] < df[visit_date_col] is throwing errors
    for col in cols: df[col] = df[col] < df[main_date_col]

    return df


def combine_treatment_to_main_data(
    main: pd.DataFrame, 
    treatment: pd.DataFrame, 
    main_date_col: str, 
    **kwargs
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
    treatment_feats['trt_date'] = treatment_feats['treatment_date'] # include treatment date as a feature
    df = combine_feat_to_main_data(main, treatment_feats, main_date_col, 'treatment_date', **kwargs)
    df = combine_feat_to_main_data(df, treatment_drugs, main_date_col, 'treatment_date', keep='sum', **kwargs)
    df = df.rename(columns={'trt_date': 'treatment_date'})
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
    **kwargs
) -> pd.DataFrame:
    """Combine features extracted from event data (emergency department visits, hospitalization, etc) to the main 
    dataset

    Args:
        main_date_col: The column name of the main visit date
        event_date_col: The column name of the event date
        lookback_window: The lookback window in terms of number of years from treatment date to extract event features
    """
    mask = main['mrn'].isin(event['mrn'])
    if parallelize:
        worker = partial(
            event_feature_extractor, 
            main_date_col=main_date_col, 
            event_date_col=event_date_col, 
            lookback_window=lookback_window, 
            **kwargs
        )
        result = split_and_parallelize((main[mask], event), worker)
    else:
        result = event_feature_extractor((main[mask], event), main_date_col, event_date_col, lookback_window, **kwargs)
    cols = ['index', f'num_prior_{event_name}s_within_{lookback_window}_years', f'days_since_prev_{event_name}']
    result = pd.DataFrame(result, columns=cols).set_index('index')
    df = main.join(result)
    return df


def event_feature_extractor(
    partition, 
    main_date_col: str,
    event_date_col: str,
    lookback_window: int = 5,
) -> list:
    """Extract features from the event data, namely
    1. Number of days since the most recent event
    2. Number of prior events in the past X years

    Args:
        main_date_col: The column name of the main visit date
        event_date_col: The column name of the event date
        lookback_window: The lookback window in terms of number of years from treatment date to extract event features
    """
    main_df, event_df = partition
    result = []
    for mrn, main_group in tqdm(main_df.groupby('mrn')):
        event_group = event_df.query('mrn == @mrn')
        event_dates = event_group[event_date_col]
        
        for idx, date in main_group[main_date_col].items():
            # get feature
            # 1. number of days since closest event prior to main visit date
            # 2. number of events within the lookback window 
            earliest_date = date - pd.Timedelta(days=lookback_window*365)
            mask = event_dates.between(earliest_date, date, inclusive='left')
            if mask.any():
                N_prior_events = mask.sum()
                # assert(sum(adm_dates == adm_dates[mask].max()) == 1)
                N_days = (date - event_dates[mask].iloc[-1]).days
                result.append([idx, N_prior_events, N_days])
    return result


def add_engineered_features(df, date_col: str = 'treatment_date') -> pd.DataFrame:
    df = get_visit_month_feature(df, col=date_col)
    df['days_since_starting_treatment'] = (df[date_col] - df['first_treatment_date']).dt.days
    get_days_since_last_treatment = partial(
        get_days_since_last_event, main_date_col=date_col, event_date_col='treatment_date'
    )
    df['days_since_last_treatment'] = df.groupby('mrn', group_keys=False).apply(get_days_since_last_treatment)
    return df