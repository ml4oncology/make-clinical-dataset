"""
Module to combine features
"""
from functools import partial
from typing import Tuple

from tqdm import tqdm
import pandas as pd

from .feat_eng import ( 
    get_days_since_last_event, 
    get_line_of_therapy, 
    get_perc_ideal_dose_given,
    get_visit_month_feature,
    get_years_diff, 
)
from .preprocess.opis import clean_drug_name
from .util import get_num_removed_patients, split_and_parallelize

def combine_demographic_to_main_data(main: pd.DataFrame, demographic: pd.DataFrame, main_date_col: str):
    """
    Args:
        main_date_col: The column name of the main asessment date
    """
    df = pd.merge(main, demographic, on='mrn', how='left')

    # exclude patients with missing birth date
    mask = df['date_of_birth'].notnull()
    get_num_removed_patients(df, mask, context='with missing birth date')
    df = df[mask]

    # exclude patients under 18 years of age
    df['age'] = get_years_diff(df, main_date_col, 'date_of_birth')
    mask = df['age'] >= 18
    get_num_removed_patients(df, mask, context='under 18 years of age')
    df = df[mask]

    # convert each cancer site / morphology datetime columns into binary indicator variables based on whether diagnosis
    # date occured before treatment date
    cols = df.columns
    cols = cols[cols.str.contains('cancer_site|morphology')]
    # TODO: find out why df[cols] = df[cols] < df[visit_date_col] is throwing errors
    for col in cols: df[col] = df[col] < df[main_date_col]

    return df

def combine_treatment_to_main_data(main: pd.DataFrame, treatment: pd.DataFrame, main_date_col: str, **kwargs):
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

def combine_perc_dose_to_main_data(main: pd.DataFrame, included_drugs: pd.DataFrame):
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

def combine_feat_to_main_data(
    main: pd.DataFrame, 
    feat: pd.DataFrame, 
    main_date_col: str, 
    feat_date_col: str, 
    **kwargs
):
    """Combine feature(s) to the main dataset

    Both main and feat should have mrn and date columns
    """
    mask = main['mrn'].isin(feat['mrn'])
    worker = partial(extractor, main_date_col=main_date_col, feat_date_col=feat_date_col, **kwargs)
    result = split_and_parallelize((main[mask], feat), worker)
    cols = ['index'] + feat.columns.drop(['mrn', feat_date_col]).tolist()
    result = pd.DataFrame(result, columns=cols).set_index('index')
    df = main.join(result)
    return df

def extractor(
    partition, 
    main_date_col: str,
    feat_date_col: str,
    keep: str = 'last', 
    time_window: Tuple[int, int] = (-5,0)
):
    """Extract either the sum, first, or last forward filled feature measurements (lab tests, symptom scores, etc) 
    taken within the time window (centered on each main visit date)

    Args:
        main_date_col: The column name of the main visit date
        feat_date_col: The column name of the feature measurement date
        time_window: The start and end of the window in terms of number of days after(+)/before(-) each visit date
        keep: Which measurements taken within the time window to keep, either `sum`, `first`, `last`
    """
    if keep not in ['first', 'last', 'sum']:
        raise ValueError('keep must be either first, last, or sum')
    
    main_df, feat_df = partition
    lower_limit, upper_limit = time_window
    keep_cols = feat_df.columns.drop(['mrn', feat_date_col])

    results = []
    for mrn, main_group in tqdm(main_df.groupby('mrn')):
        feat_group = feat_df.query('mrn == @mrn')

        for idx, date in main_group[main_date_col].items():
            earliest_date = date + pd.Timedelta(days=lower_limit)
            latest_date = date + pd.Timedelta(days=upper_limit)

            mask = feat_group[feat_date_col].between(earliest_date, latest_date)
            if not mask.any(): 
                continue

            feats = feat_group.loc[mask, keep_cols]
            if keep == 'sum':
                result = feats.sum()
            elif keep == 'first':
                result = feats.iloc[0]
            elif keep == 'last':
                result = feats.ffill().iloc[-1]

            results.append([idx]+result.tolist())
    
    return results

def add_engineered_features(df, date_col: str = 'treatment_date'):
    df = get_visit_month_feature(df, col=date_col)
    df['line_of_therapy'] = df.groupby('mrn', group_keys=False).apply(get_line_of_therapy)
    df['days_since_starting_treatment'] = (df[date_col] - df['first_treatment_date']).dt.days
    get_days_since_last_treatment = partial(
        get_days_since_last_event, main_date_col=date_col, event_date_col='treatment_date'
    )
    df['days_since_last_treatment'] = df.groupby('mrn', group_keys=False).apply(get_days_since_last_treatment)
    return df