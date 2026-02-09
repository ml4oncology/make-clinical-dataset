"""
Module to preprocess ESAS (Edmonton Symptom Assessment Score) and ECOG (Eastern Cooperative Oncology Group) Performance Status data

NOTE: The completion rate of these surveys dropped from ~70% to ~30% during COVID and has never rebounded. 
We might have to rethink on the relevance of these as features, if we want to use them for predicting future outcomes. 
"""
import pandas as pd
from make_clinical_dataset.epr.util import get_excluded_numbers
from make_clinical_dataset.shared.constants import ESAS_MAP, SYMP_COLS
from ml_common.util import load_table


###############################################################################
# EPIC
###############################################################################
def get_epic_symp_data(filepath: str) -> pd.DataFrame:
    """Load, clean, filter, process EPIC symptom survey data."""
    df = load_table(filepath)

    # rename the columns
    df = df.rename(columns={
        "RESEARCH_ID": "mrn",
        "SURVEY_DATE": "obs_date",
        "ESAS_PAIN": "pain",
        "ESAS_TIREDNESS": "tiredness",
        "ESAS_NAUSEA": "nausea",
        "ESAS_DEPRESSION": "depression",
        "ESAS_ANXIETY": "anxiety",
        "ESAS_DROWSINESS": "drowsiness",
        "ESAS_APPETITE": "lack_of_appetite",
        "ESAS_WELL_BEING": "well_being",
        "ESAS_SHORTNESS_OF_BREATH": "shortness_of_breath",
        # "ESAS_CONSTIPATION": "constipation",
        # "ESAS_DIARRHEA": "diarrhea", 
        # "ESAS_SLEEP": "sleep",
        "PATIENT_ECOG": "ecog",
    })

    # clean ecog entries
    # NOTE: ecog = 5 means death so make sure there are no label leakage.
    df['ecog'] = df['ecog'].astype(str).replace('Not Applicable', None)
    # some entries have the following format: score-description. Remove the descriptions
    df['ecog'] = df['ecog'].str.split('-').str[0]

    # fix dtypes
    df[SYMP_COLS] = df[SYMP_COLS].astype(float)
    df['obs_date'] = pd.to_datetime(df['obs_date']).dt.normalize()

    # keep only useful columns
    df = df[['mrn', 'obs_date'] + SYMP_COLS].copy()

    # handle conflicting data by taking the max
    df = df.groupby(['mrn', 'obs_date']).agg('max').reset_index()

    # sort by patient and date
    df = df.sort_values(by=['mrn', 'obs_date'])

    return df


###############################################################################
# Pre-EPIC
###############################################################################
def get_symp_data(
    id_to_mrn: dict[str, str], 
    data_dir: str | None = None,
    verbose: bool = False
) -> pd.DataFrame:
    """Load, clean, filter, process Pre-EPIC symptom observation data."""
    if data_dir is None:
        data_dir = './data/raw/ESAS'
    symp = pd.read_parquet(data_dir)
    symp = clean_symp_data(symp, id_to_mrn=id_to_mrn)
    symp = filter_symp_data(symp, verbose=verbose)
    symp = process_symp_data(symp)
    return symp


def clean_symp_data(df: pd.DataFrame, id_to_mrn: dict[str, int]) -> pd.DataFrame:
    """Clean and rename column names and entries. 
    
    Merge same columns together.
    """
    # merge the string entries and numerical entries together into one column (no overlap)
    # assert (df['obs_val_str'].isna() == df['obs_val_num'].notna()).all()
    df['obs_val'] = df.pop('obs_val_str').fillna(df.pop('obs_val_num'))
    
    # map the patient ID to mrns
    df['mrn'] = df.pop('patient').map(id_to_mrn)

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
