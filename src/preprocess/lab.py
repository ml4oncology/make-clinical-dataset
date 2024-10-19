"""
Module to preprocess laboratory test data, which includes hematology and biochemistry data
"""
from typing import Optional

import pandas as pd

from .. import ROOT_DIR
from ..constants import OBS_MAP

def get_lab_data(mrn_map: dict[str, int], data_dir: Optional[str] = None):
    if data_dir is None:
        data_dir = f'{ROOT_DIR}/data/raw'

    hema = pd.read_parquet(f'{data_dir}/hematology.parquet.gzip')
    hema = filter_lab_data(hema, obs_name_map=OBS_MAP['Hematology'])

    biochem = pd.read_parquet(f'{data_dir}/biochemistry.parquet.gzip')
    biochem = filter_lab_data(biochem, obs_name_map=OBS_MAP['Biochemistry'])

    lab = pd.concat([hema, biochem])
    lab = process_lab_data(lab)
    lab['mrn'] = lab.pop('patientid').map(mrn_map) # map mrn to patientid
    return lab

def process_lab_data(df):
    df['obs_datetime'] = pd.to_datetime(df['obs_datetime'], utc=True)
    df['obs_date'] = pd.to_datetime(df['obs_datetime'].dt.date)
    df = df.sort_values(by='obs_datetime')

    # save the units for each observation name
    unit_map = dict(df[['obs_name', 'obs_unit']].value_counts().index.tolist())
    # TODO: save unit map in feature store for later use
    print(unit_map)

    # take the most recent value if multiple lab tests taken in the same day
    # NOTE: dataframe already sorted by obs_datetime
    df = df.groupby(['patientid', 'obs_date', 'obs_name']).agg({'obs_value': 'last'}).reset_index()

    # make each observation name into a new column
    df = df.pivot(index=['patientid', 'obs_date'], columns='obs_name', values='obs_value').reset_index()

    df.columns.name = None
    return df

def filter_lab_data(df, obs_name_map: Optional[dict] = None):
    df = clean_lab_data(df)
    
    # exclude rows where observation value is missing
    df = df[df['obs_value'].notnull()]

    if obs_name_map is not None:
        df['obs_name'] = df['obs_name'].map(obs_name_map)
        # exclude observations not in the name map
        df = df[df['obs_name'].notnull()]

    df = filter_units(df)
    df = df.drop_duplicates(subset=['patientid', 'obs_value', 'obs_name', 'obs_unit', 'obs_datetime'])
    return df

def filter_units(df):
    # clean the units
    df['obs_unit'] = df['obs_unit'].replace({'bil/L': 'x10e9/L', 'fl': 'fL'})

    # some observations have measurements in different units (e.g. neutrophil observations contain measurements in 
    # x10e9/L (the majority) and % (the minority))
    # only keep one measurement unit for simplicity
    exclude_unit_map = {
        'creatinine': ['mmol/d', 'mmol/CP'],
        'eosinophil': ['%'], 
        'lymphocyte': ['%'], 
        'monocyte': ['%'], 
        'neutrophil': ['%'],
        'red_blood_cell': ['x10e6/L'],
        'white_blood_cell': ['x10e6/L'],
    }
    mask = False
    for obs_name, exclude_units in exclude_unit_map.items():
        mask |= (df['obs_name'] == obs_name) & df['obs_unit'].isin(exclude_units)
    df = df[~mask]

    return df

def clean_lab_data(df):
    # clean column names
    col_map = {
        # assign obs_ prefix to ensure no conflict with preexisting columns
        'component-code-coding-0-display': 'obs_display',
        'component-code-text': 'obs_text', 
        'component-valueQuantity-unit': 'obs_unit',
        'component-valueQuantity-value': 'obs_value',
        'effectiveDateTime': 'effective_datetime',
        'lastUpdated': 'updated_datetime',
    }
    df = df.rename(columns=col_map)

    # the observation name is captured in two different columns, combine them together
    df['obs_name'] = df['obs_display'].fillna(df['obs_text'])
    # there are no cases where both display and text are filled
    assert not any(df['obs_display'].notnull() & df['obs_text'].notnull())

    # the datetime is captured in two different columns, combine them together
    df['obs_datetime'] = df['effective_datetime'].fillna(df['updated_datetime'])
    # effective datetime is always earlier (as in more accurate) than last updated datetime
    mask = df['effective_datetime'].notnull()
    assert all(df.loc[mask, 'effective_datetime'] < df.loc[mask, 'updated_datetime'])
    
    return df
