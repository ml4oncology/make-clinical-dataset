"""
Module to preprocess emergency department visit data
"""
from typing import Optional

import pandas as pd

from .. import ROOT_DIR, logger

def get_emergency_department_data(data_dir: Optional[str] = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = f'{ROOT_DIR}/data/raw'

    df = pd.read_parquet(f'{data_dir}/ED.parquet.gzip')
    df = filter_emergency_department_data(df)
    return df

def filter_emergency_department_data(ED: pd.DataFrame) -> pd.DataFrame:
    # clean column names
    ED.columns = ED.columns.str.lower()
    ED = ED.rename(columns={'medical_record_number': 'mrn', 'admission_date_time': 'adm_date'})
    # clean data types
    ED['adm_date'] = pd.to_datetime(ED['adm_date'], format="%d%b%Y:%H:%M:%S")

    # remove duplicate visits
    mask = ED[['mrn', 'adm_date']].duplicated(keep='last')
    logger.info(f'Removing {sum(mask)} duplicate emergency department visits') 
    ED = ED[~mask]
    return ED