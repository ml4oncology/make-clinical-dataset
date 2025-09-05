"""
Module to preprocess the cancer registry (cancer patient demographic data)
and EPR death dates
"""
import pandas as pd
from make_clinical_dataset.epr.util import get_excluded_numbers
from ml_common.constants import CANCER_CODE_MAP


def get_demographic_data(data_dir: str | None = None):
    if data_dir is None:
        data_dir = './data/raw'

    df = pd.read_parquet(f'{data_dir}/cancer_registry.parquet.gzip')
    df = filter_demographic_data(df)
    df = process_demographic_data(df)

    ddates = pd.read_parquet(f'{data_dir}/death_dates.parquet.gzip')
    df = update_death_dates(df, ddates)
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
            'date_of_death': 'last',
            'last_contact_date': 'last',
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
    get_excluded_numbers(df, mask, context=' with no MRN')
    df = df[mask]

    # clean data types
    df['mrn'] = df['mrn'].astype(int)
    df['morphology'] = df['morphology'].astype(int).astype(str)

    # sanity check - ensure vital status and death date matches and makes sense
    mask = df['vital_status'].map({'Dead': False, 'Alive': True}) == df['date_of_death'].isnull()
    assert mask.all()

    # filter out patients whose sex is not Male/Female
    mask = df['sex'].isin(['Male', 'Female'])
    get_excluded_numbers(df, mask, context=' in which sex is other than Male/Female')
    df = df[mask].copy()
    df['female'] = df.pop('sex') == 'Female'

    # clean cancer site and morphology feature
    for col in ['primary_site', 'morphology']: 
        # only keep first three characters - the rest are for specifics
        # e.g. C50 Breast: C501 Central portion, C504 Upper-outer quadrant, etc
        df[col] = df[col].str[:3]
        # map code to english
        # df[col] = df[col].map(CANCER_CODE_MAP)

    return df


def update_death_dates(df: pd.DataFrame, ddates: pd.DataFrame) -> pd.DataFrame:
    ddates.rename(columns={'MEDICAL_RECORD_NUMBER': 'mrn', 'DATE_OF_DEATH': 'date_of_death'}, inplace=True)
    ddates['date_of_death'] = pd.to_datetime(ddates['date_of_death'], format='%d%b%Y:%H:%M:%S')
    df['date_of_death2'] = df['mrn'].map(dict(ddates.to_numpy()))

    # remove errornous entry
    mask = df['date_of_death2'] == df['date_of_birth']
    df.loc[mask, 'date_of_death2'] = pd.NaT

    # take the earlier death date
    df['date_of_death'] = df[['date_of_death', 'date_of_death2']].min(axis=1)
    
    del df['date_of_death2']
    return df
