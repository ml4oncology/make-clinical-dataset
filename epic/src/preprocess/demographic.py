"""
Module to preprocess demographic and cancer diagnosis data
"""
import pandas as pd
from make_clinical_dataset.shared.constants import INFO_DIR
from ml_common.constants import CANCER_CODE_MAP


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

    # fix language and and religion entries
    for col in ['preferred_language', 'religion']:
        df[col] = df[col].str.strip().str.lower()

    # remove race and ethnicity
    assert df['race'].isna().all()
    assert df['ethnicity'].isna().all()
    df = df.drop(columns=["race", "ethnicity"])

    # map the patient ID to mrns
    df['mrn'] = df.pop('patient').map(id_to_mrn)
    
    return df


###############################################################################
# Diagnosis
###############################################################################
def get_diagnosis_data() -> pd.DataFrame:
    """Load, clean, filter, process diagnosis data."""
    df = pd.read_csv(f'{INFO_DIR}/cancer_diag.csv')
    
    # get the site mapping
    site_map = pd.read_csv(f'{INFO_DIR}/site_names_normalized_v1.csv')
    site_to_code = site_map.set_index('PRIMARY_SITE_DESC')['cancer_code_ICD10'].to_dict()

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

    # exclude rows without diagnosis date
    df = df[df['diagnosis_date'].notna()]

    # map the raw primary site description to ICD-10 code
    df['primary_site_code'] = df['primary_site_desc'].map(site_to_code)

    # combine the raw primary site and raw morphology description into a single column
    df['cancer_desc'] = df["primary_site_desc"] + '\n' + df["morphology_desc"]

    # map code back to a normalized primary site description
    df['primary_site_desc'] = df['primary_site_code'].map(CANCER_CODE_MAP)

    # sort the data
    df = df.sort_values(by=['mrn', 'diagnosis_date'])

    return df