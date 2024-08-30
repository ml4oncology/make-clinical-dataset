"""
Module to preprocess EPR radiology report data
"""
from typing import Optional

import pandas as pd

from ml_common.text import remove_duplicate_lines
from .. import ROOT_DIR

def get_radiology_data(mrn_map: dict[str, int], data_dir: Optional[str] = None, **kwargs) -> pd.DataFrame:
    if data_dir is None:
        data_dir = f"{ROOT_DIR}/data/raw"

    df = pd.read_parquet(f"{data_dir}/radiology.parquet.gzip")
    df['mrn'] = df['patientid'].map(mrn_map) # map the mrn to the patient id

    df = filter_radiology_data(df, **kwargs)
    df = process_radiology_data(df)
    return df


def process_radiology_data(df: pd.DataFrame) -> pd.DataFrame:
    df['processed_text'] = remove_duplicate_lines(df['raw_text'])

    # remove useless info for faster inference
    end_text = (
        "FOR PHYSICIAN USE ONLY - Please note that this report was generated using voice recognition software. "
        "The Joint Department of Medical Imaging (JDMI) supports a culture of continuous improvement. "
        "If you require further clarification or feel there has been an error in this report "
        "please contact the JDMI Call Centre from 8am - 8pm Monday to Friday, "
        "8am - 4pm Saturday to Sunday (excludes statutory holidays) at 416-946-2809."
    )
    # print(df['processed_text'].str.endswith(end_text).value_counts())
    df['processed_text'] = df['processed_text'].str.replace(end_text, '')

    # extract the date written in the report
    assert all(df['raw_text'].str.startswith('REPORT (FINAL '))
    start_idx = len('REPORT (FINAL ')
    end_idx = start_idx + len('YYYY/MM/DD')
    df['date_in_report'] = pd.to_datetime(df['raw_text'].str[start_idx:end_idx])

    # sort by patient and date written in the report
    df = df.sort_values(by=['mrn', 'date_in_report'])
    return df


def filter_radiology_data(df: pd.DataFrame) -> pd.DataFrame:
    # cleaning up columns
    df = df.rename(columns={
        'component-valueString': 'raw_text', 
        'effectiveDateTime': 'effective_datetime', 
        'lastUpdated': 'updated_datetime'
    })
    updated_datetime = pd.to_datetime(df['updated_datetime'], utc=True, format='ISO8601')
    effective_datettime = pd.to_datetime(df['effective_datetime'], utc=True)
    df['datetime'] = effective_datettime.fillna(updated_datetime)
    df['datetime'] = df['datetime'].dt.tz_localize(None)
    df['date'] = pd.to_datetime(df['datetime'].dt.date, utc=True)

    # only keep rows that contains report text
    mask = df['raw_text'].fillna('').str.startswith('REPORT')
    df = df[mask]

    # remove rows without a mapped mrn
    df = df[df['mrn'].notnull()]

    df = df[['mrn', 'datetime', 'date', 'proc_name', 'raw_text']]
    return df