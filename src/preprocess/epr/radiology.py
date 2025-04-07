"""
Module to preprocess EPR radiology report data
"""
from typing import Optional

import pandas as pd

from ml_common.text import remove_duplicate_lines

def get_radiology_data(
    mrn_map: dict[str, int], 
    data_dir: Optional[str] = None, 
    drop_duplicates: bool = True
) -> pd.DataFrame:
    if data_dir is None:
        data_dir = "./data/raw"

    df = pd.read_parquet(f"{data_dir}/radiology.parquet.gzip")
    df['mrn'] = df['patientid'].map(mrn_map) # map the mrn to the patient id

    df = filter_radiology_data(df)
    df = process_radiology_data(df)

    if drop_duplicates:
        # Drop duplicate reports
        # some duplicate reports have different proc names, merge proc names if reports are the same
        proc_names = df.groupby('raw_text')['proc_name'].apply(lambda names: ', '.join(set(names)))
        df['proc_name'] = df['raw_text'].map(proc_names)
        # sort data by patient and final date
        df = df.sort_values(by=['mrn', 'date'])
        df = df.drop_duplicates(subset=['raw_text'], keep='first')

    return df


def process_radiology_data(df: pd.DataFrame) -> pd.DataFrame:
    df['processed_text'] = remove_duplicate_lines(df['raw_text'])

    # Remove useless info for faster inference
    end_text = (
        "FOR PHYSICIAN USE ONLY - Please note that this report was generated using voice recognition software. "
        "The Joint Department of Medical Imaging (JDMI) supports a culture of continuous improvement. "
        "If you require further clarification or feel there has been an error in this report "
        "please contact the JDMI Call Centre from 8am - 8pm Monday to Friday, "
        "8am - 4pm Saturday to Sunday (excludes statutory holidays) at 416-946-2809."
    )
    # print(df['processed_text'].str.endswith(end_text).value_counts())
    df['processed_text'] = df['processed_text'].str.replace(end_text, '')

    # ensure all reports start with "REPORT (FINAL YYYY/MM/DD)"
    assert all(df['raw_text'].str.startswith('REPORT (FINAL '))
    # ensure there is only one line with "REPORT (FINAL YYYY/MM/DD)" in each report
    assert all(df['raw_text'].str.count('REPORT \(FINAL') == 1)

    # Extract the date written in the report
    start_idx = len('REPORT (FINAL ')
    end_idx = start_idx + len('YYYY/MM/DD')
    df['initial_report_date'] = pd.to_datetime(df['raw_text'].str[start_idx:end_idx])
    # the report may include addendums that were added at later date
    # to prevent data leakage, extract the final date in which an addendum was added
    res = {}
    for i, text in df['raw_text'].items():
        start_idx = text.rfind('ADDENDUM (FINAL ') # finds the rightmost index (last addendum)
        if start_idx == -1: continue
        start_idx += len('ADDENDUM (FINAL ')
        end_idx = start_idx + len('YYYY/MM/DD')
        res[i] = pd.to_datetime(text[start_idx:end_idx])
    df['last_addendum_date'] = pd.Series(res)

    # Set up the final date
    df['date'] = df['initial_report_date']
    mask = df['initial_report_date'] < df['last_addendum_date']
    df.loc[mask, 'date'] = df['last_addendum_date']

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
    df['epr_datetime'] = effective_datettime.fillna(updated_datetime)
    df['epr_datetime'] = df['epr_datetime'].dt.tz_localize(None)

    # only keep rows that contains report text
    mask = df['raw_text'].fillna('').str.startswith('REPORT')
    df = df[mask]

    # remove rows without a mapped mrn
    df = df[df['mrn'].notnull()]

    df = df[['mrn', 'epr_datetime', 'proc_name', 'raw_text']]
    return df