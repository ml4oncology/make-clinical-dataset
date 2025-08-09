"""
Module to preprocess ESAS (Edmonton Symptom Assessment Score) and ECOG (Eastern Cooperative Oncology Group) Performance Status data

NOTE: The completion rate of these surveys dropped from ~70% to ~30% during COVID and has never rebounded. 
We might have to rethink on the relevance of these as features, if we want to use them for predicting future outcomes. 
"""
import logging

import pandas as pd
from ml_common.util import get_excluded_numbers

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s %(levelname)s:%(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)

ESAS_MAP = {
    'Anxiety': 'anxiety',
    'Appetite': 'lack_of_appetite', 
    'Depression': 'depression',
    'Drowsiness': 'drowsiness',
    'ECOG (Patient reported)': 'ecog',
    'Feeling of Well-being': 'well_being',
    'Lack of Appetite': 'lack_of_appetite',
    'Nausea': 'nausea',
    'Pain': 'pain', 
    'Shortness of breath': 'shortness_of_breath',
    'Tiredness': 'tiredness',
    'Wellbeing': 'well_being',
}


def get_symp_data(
    mrn_map: dict[str, str], 
    data_dir: str | None = None,
    verbose: bool = False
) -> pd.DataFrame:
    """Load, clean, filter, process symptom observation data."""
    if data_dir is None:
        data_dir = './data/raw/ESAS'
    symp = pd.read_parquet(data_dir)
    symp = clean_symp_data(symp, mrn_map=mrn_map)
    symp = filter_symp_data(symp, verbose=verbose)
    symp = process_symp_data(symp)
    return symp


def clean_symp_data(df: pd.DataFrame, mrn_map: dict[str, int]) -> pd.DataFrame:
    """Clean and rename column names and entries. 
    
    Merge same columns together.
    """
    # merge the string entries and numerical entries together into one column (no overlap)
    # assert (df['obs_val_str'].isna() == df['obs_val_num'].notna()).all()
    df['obs_val'] = df.pop('obs_val_str').fillna(df.pop('obs_val_num'))
    
    # map the patient ID to mrns
    df['mrn'] = df.pop('patient').map(mrn_map)
    
    # rename the columns
    df = df.rename(columns={'occurrence_datetime_from_order': 'obs_datetime'})
    
    # rename the observations
    df['obs_name'] = df['obs_name'].map(ESAS_MAP)
    
    return df


def filter_symp_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Filter out observations based on various conditions (missingness, duplicates, etc)."""
    # exclude rows without a date
    mask = df['obs_datetime'].notna()
    if verbose:
        get_excluded_numbers(df, mask=mask, context=" without a date")
    df = df[mask].copy()
    
    # exclude rows whose date is prior to 2005
    mask = df['obs_datetime'] >= '2005-01-01'
    if verbose:
        get_excluded_numbers(df, mask=mask, context=" that 'occured' before 2005")
    df = df[mask].copy()
    
    # exclude observations that were not included in the mapping
    mask = df['obs_name'].notna()
    if verbose:
        context = " whose observations were not included in the mapping"
        get_excluded_numbers(df, mask=mask, context=context)
    df = df[mask].copy()
    # observation value should all be numbers now, convert to int
    df['obs_val'] = df['obs_val'].astype(int)
    
    # keep only useful columns
    df = df[['mrn', 'obs_datetime', 'obs_name', 'obs_val']].copy()
    
    # exclude duplicates
    mask = ~df.duplicated()
    if verbose:
        get_excluded_numbers(df, mask=mask, context=" who are duplicates")
    df = df[mask].copy()
    
    return df


def process_symp_data(df):
    """Sort and pivot the observation data."""
    df['obs_date'] = pd.to_datetime(df['obs_datetime'].dt.date)

    df = df.sort_values(by=['mrn', 'obs_date', 'obs_name'])
    
    # make each observation name into a new column, handle conflicting data by taking the max
    df = df.pivot_table(index=['mrn', 'obs_date'], columns='obs_name', values='obs_val', aggfunc='max')
    df = df.reset_index()
    df.columns.name = None
    
    return df
