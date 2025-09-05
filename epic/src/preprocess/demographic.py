"""
Module to preprocess demographic and cancer diagnosis data
"""
import pandas as pd
from make_clinical_dataset.shared.constants import INFO_DIR


def get_demographic_data() -> pd.DataFrame:
    """Load, clean, filter, process demographic data."""
    df = pd.read_csv(f'{INFO_DIR}/cancer_diag.csv')
    pc = pd.read_csv(f'{INFO_DIR}/postal_codes.csv')

    df = clean_demographic_data(df)
    df = filter_demographic_data(df)

    df = pd.merge(df, pc, on='mrn', how='left')
    return df


def clean_demographic_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and rename column names and entries."""
    # rename the columns
    df.columns = df.columns.str.lower()
    df = df.rename(columns={
        'type': 'cancer_type', 
        'medical_record_number': 'mrn', 
        'date_of_birth': 'birth_date'
    })

    # ensure correct data type
    df['mrn'] = df['mrn'].astype(int)
    for col in ['birth_date', 'diagnosis_date']: 
        df[col] = pd.to_datetime(df[col])

    return df


def filter_demographic_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['morphology_desc']).drop_duplicates()
    return df