"""
Module to preprocess RECIST (Response evaluation criteria in solid tumors) data from COMPASS trial
"""

from typing import Optional

import pandas as pd
from make_clinical_dataset.shared.constants import RECIST_RANKING


def get_recist_data(data_dir: Optional[str] = None) -> pd.DataFrame:
    """Loads and clean the df (Response evaluation criteria in solid tumors) data taken from 
    patients in COMPASS trial
    """
    if data_dir is None:
        data_dir = "./data/external"
    df = pd.read_csv(f'{data_dir}/RECIST_data.csv')
    compass_mrns = pd.read_csv(f'{data_dir}/COMPASS_ID_to_MRN_map.csv')
    df = filter_and_process_recist_data(compass_mrns, df)
    return df


def filter_and_process_recist_data(compass_mrns, df: pd.DataFrame) -> pd.DataFrame:
    # clean the columns
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df['date'] = pd.to_datetime(df['date'], format='%Y %b %d', utc=True)
    df['overall_response'] = df['overall_response'].replace({'Non-CR/Non-PD': 'PR'})
    
    # remove rows without a date or overall response 
    df = df[df['date'].notnull() & df['overall_response'].notnull()].copy()
    
    # make COMPASS ID consistent (i.e. COMPXXXX -> COMPASS-XXXX or COMP-XXXX -> COMPASS-XXXX)
    mask = df['subject'].str.startswith('COMPASS-')
    df.loc[~mask, 'subject'] = df.loc[~mask, 'subject'].str.replace('COMP-', 'COMPASS-')
    mask = df['subject'].str.startswith('COMPASS-')
    df.loc[~mask, 'subject'] = df.loc[~mask, 'subject'].str.replace('COMP', 'COMPASS-')
    
    # map the mrn to the compass ID
    compass_mrns.rename(columns={'MRN': 'mrn', 'Registration/ Randomization No.': 'subject'}, inplace=True)
    compass_mrns = compass_mrns.set_index('subject')['mrn']
    df['mrn'] = df['subject'].map(compass_mrns)

    # remove rows without a mapped mrn
    df = df[df['mrn'].notnull()]

    # add response ranking
    df['overall_response_rank'] = df['overall_response'].map(RECIST_RANKING)

    # sort by patient and date
    df = df.sort_values(by=['mrn', 'date'])
    
    return df