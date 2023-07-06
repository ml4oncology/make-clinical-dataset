"""
Module to preprocess the cancer registry (cancer patient demographic data)
"""
from typing import Optional
import pandas as pd

from .. import ROOT_DIR, logger
from ..constants import cancer_code_map
from ..util import get_num_removed_patients

def get_demographic_data(data_dir: Optional[str] = None, external_data: Optional[pd.DataFrame] = None):
    if data_dir is None:
        data_dir = f'{ROOT_DIR}/data/raw'

    df = pd.read_parquet(f'{data_dir}/cancer_registry.parquet.gzip')
    df = filter_demographic_data(df)
    df = process_demographic_data(df)
    if external_data is not None:
        df = add_external_demographic_data(df, external_data)
        
    return df

def process_demographic_data(df):
    # order by diagnosis date
    df = df.sort_values(by='diagnosis_date')

    # make each cancer site and morphology into a new column with diagnosis date as entry
    cancer_site = df.pivot(columns='primary_site', values='diagnosis_date').loc[df.index]
    morphology = df.pivot(columns='morphology', values='diagnosis_date').loc[df.index]
    cancer_site.columns = 'cancer_site_' + cancer_site.columns
    morphology.columns = 'morphology_' + morphology.columns
    cancer = pd.concat([cancer_site, morphology], axis=1)
    df = df.join(cancer)

    # combine patients with mutliple diagnoses into one row
    df = (
        df
        .groupby(['mrn'])
        .agg({
            # handle conflicting data by taking the most recent entries
            'date_of_birth': 'last',
            'female': 'last',
            # if two diagnoses dates for same cancer site/morphology (e.g. first diagnoses in 2005, cancer returns in 
            # 2013) take the first date (e.g. 2005)
            **{col: 'min' for col in cancer.columns}
        })
    )
    df = df.reset_index()

    return df
    
def filter_demographic_data(df):
    # clean column names
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace('start', 'start_date')
    df = df.rename(columns= {'medical_record_number': 'mrn'})

    # filter out patients without medical record numbers
    mask = df['mrn'].notnull()
    get_num_removed_patients(df, mask, context='with no MRN')
    df = df[mask]

    # clean data types
    df['mrn'] = df['mrn'].astype(int)
    df['morphology'] = df['morphology'].astype(int).astype(str)

    # sanity check - ensure vital status and death date matches and makes sense
    mask = df['vital_status'].map({'Dead': False, 'Alive': True}) == df['date_of_death'].isnull()
    assert mask.all()

    # filter out patients whose sex is not Male/Female
    mask = df['sex'].isin(['Male', 'Female'])
    get_num_removed_patients(df, mask, context='whose sex is other than Male/Female')
    df = df[mask].copy()
    df['female'] = df.pop('sex') == 'Female'

    # clean cancer site and morphology feature
    for col in ['primary_site', 'morphology']: 
        # only keep first three characters - the rest are for specifics
        # e.g. C50 Breast: C501 Central portion, C504 Upper-outer quadrant, etc
        df[col] = df[col].str[:3]
        # map code to english
        # df[col] = df[col].map(cancer_code_map)

    return df

def add_external_demographic_data(df, external_df):
    """Combine external demographic data (from DART) to cancer registry"""
    assert external_df['mrn'].nunique() == len(external_df) # ensure no conflicts in the external demographic data
    mask = ~external_df['mrn'].isin(df['mrn']) # get all patients that does not exist in cancer registry
    df = pd.concat([df, external_df[mask]])
    
    msg = f'Number of patients in cancer registry = {len(df)}. Adding an additional {sum(mask)} patients from DART.'
    logger.info(msg)

    return df
