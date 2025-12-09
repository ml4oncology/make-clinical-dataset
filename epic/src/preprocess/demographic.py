"""
Module to preprocess demographic and cancer diagnosis data
"""
import pandas as pd
from make_clinical_dataset.shared.constants import INFO_DIR


###############################################################################
# Demographic
###############################################################################
def get_demographic_data(
    filepath: str,
    id_to_mrn: dict[str, str],
) -> pd.DataFrame:
    """Load, clean, filter, process demographic data."""
    df = pd.read_parquet(filepath)

    # rename the columns
    df = df.rename(columns={
        "PATIENT_RESEARCH_ID": "patient",
        "gender": "sex",
        "deceasedDateTime": "death_date",
    })
    df.columns = df.columns.str.lower()

    # fix dtypes
    df['death_date'] = pd.to_datetime(df['death_date'], format='ISO8601', utc=True).dt.tz_convert(None)

    # map the patient ID to mrns
    df['mrn'] = df.pop('patient').map(id_to_mrn)

    # fix language and and religion entries
    for col in ['preferred_language', 'religion']:
        df[col] = df[col].str.strip().str.lower()

    # remove race and ethnicity
    assert df['race'].isna().all()
    assert df['ethnicity'].isna().all()
    df = df.drop(columns=["race", "ethnicity"])
    
    return df


###############################################################################
# Diagnosis
###############################################################################
def get_diagnosis_data() -> pd.DataFrame:
    """Load, clean, filter, process diagnosis data."""
    df = pd.read_csv(f'{INFO_DIR}/cancer_diag.csv')
    df = clean_diagnosis_data(df)
    df = process_diagnosis_data(df)
    return df


def clean_diagnosis_data(df: pd.DataFrame) -> pd.DataFrame:
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


def process_diagnosis_data(df: pd.DataFrame) -> pd.DataFrame:
    df['cancer_desc'] = df["primary_site_desc"] + '\n' + df["morphology_desc"]
    return df