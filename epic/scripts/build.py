"""Script to turn raw data into features / targets for modelling"""

import pandas as pd
import polars as pl
from make_clinical_dataset.epic.preprocess.acu import get_acu_data, get_triage_data
from make_clinical_dataset.epic.preprocess.demographic import (
    get_demographic_data,
    get_diagnosis_data,
)
from make_clinical_dataset.epic.preprocess.esas import get_epic_symp_data, get_symp_data
from make_clinical_dataset.epic.preprocess.lab import get_lab_data
from make_clinical_dataset.epic.preprocess.radiology import get_radiology_data
from make_clinical_dataset.epic.preprocess.treatment import (
    get_chemo_data,
    get_radiation_data,
)
from make_clinical_dataset.epic.util import load_lab_map
from make_clinical_dataset.shared.constants import INFO_DIR, ROOT_DIR

# Paths and Configurations
DATE = '2025-01-08' # Please ask Wayne Uy about the merged_processed_cleaned_clinical_notes dataset
ACU_PATH = f'{ROOT_DIR}/data/processed/clinical_notes/data_pull_{DATE}/merged_processed_cleaned_clinical_notes.parquet.gzip'

DATE = '2025-03-29'
ER_TRIAGE_DIR = f'{ROOT_DIR}/data/processed/ED/ED_{DATE}'
LAB_DIR = f'{ROOT_DIR}/data/processed/lab/lab_{DATE}'
PRE_EPIC_ESAS_DIR = f'{ROOT_DIR}/data/processed/ESAS/ESAS_{DATE}'
RAD_DIR = f'{ROOT_DIR}/data/processed/radiology/radiology_{DATE}'
OUTPUT_DIR = f'{ROOT_DIR}/data/final/data_{DATE}/interim'

DATE = '2025-07-02'
PRE_EPIC_CHEMO_PATH = f'{ROOT_DIR}/data/processed/treatment/chemo_{DATE}.parquet'
RT_PATH = f'{ROOT_DIR}/data/processed/treatment/radiation_{DATE}.parquet'

DATE = '2025-10-08'
DEMOG_PATH = f'{ROOT_DIR}/data/processed/cancer_registry/demographic_{DATE}.parquet'

DATE = '2025-11-03'
EPIC_CHEMO_PATH = f'{ROOT_DIR}/data/processed/treatment/chemo_{DATE}.parquet'
EPIC_ESAS_PATH = f'{ROOT_DIR}/data/processed/ESAS/ESAS_{DATE}.parquet'
EPIC_ED_ADMIT_PATH = f'{ROOT_DIR}/data/processed/ED/ED_{DATE}.parquet' 


def build_chemo_and_radiation_treatments(id_to_mrn: dict[str, int], drug_map: pl.DataFrame):
    chemo = get_chemo_data(PRE_EPIC_CHEMO_PATH, id_to_mrn, drug_map)
    # chemo = pl.concat([chemo_pre_epic, chemo_epic], how="diagonal")
    chemo.write_parquet(f'{OUTPUT_DIR}/chemo.parquet')
    rad = get_radiation_data(RT_PATH, id_to_mrn)
    rad.to_parquet(f'{OUTPUT_DIR}/radiation.parquet', compression='zstd', index=False)


def build_laboratory_tests(id_to_mrn: dict[str, int], lab_map: dict[str, str]):
    lab = get_lab_data(id_to_mrn, lab_map, data_dir=LAB_DIR)
    lab.write_parquet(f'{OUTPUT_DIR}/lab.parquet')


def build_symptoms(id_to_mrn: dict[str, int]):
    pre_epic_symp = get_symp_data(id_to_mrn, data_dir=PRE_EPIC_ESAS_DIR)
    epic_symp = get_epic_symp_data(EPIC_ESAS_PATH)
    symp = pd.concat([pre_epic_symp, epic_symp]).sort_values(by=['mrn', 'obs_date'])
    assert not symp.duplicated(subset=['mrn', 'obs_date']).any()
    symp.to_parquet(f'{OUTPUT_DIR}/symptom.parquet', compression='zstd', index=False)


def build_radiology_reports(id_to_mrn: dict[str, int]):
    reports = get_radiology_data(id_to_mrn, data_dir=RAD_DIR)
    reports.write_parquet(f'{OUTPUT_DIR}/reports.parquet')
    

def build_acute_care_use():
    acu = get_acu_data(ACU_PATH)
    acu.sink_parquet(f'{OUTPUT_DIR}/acute_care_use.parquet')

    triage = get_triage_data(ER_TRIAGE_DIR)
    triage.write_parquet(f'{OUTPUT_DIR}/triage.parquet')


def build_demographic(id_to_mrn: dict[str, int]):
    diag = get_diagnosis_data()
    demog = get_demographic_data(DEMOG_PATH, id_to_mrn)
    demog = pd.merge(diag, demog, how='left', on='mrn')
    demog.to_parquet(f'{OUTPUT_DIR}/demographic.parquet', compression='zstd', index=False)


def build_last_seen_dates():
    # TODO: use polars
    # TODO: use all the raw data? or request from EPIC database the LAST_CONTACT_DATE
    last_seen = pd.DataFrame()
    dataset_map = {
        'lab': 'obs_date', 
        'symptom': 'obs_date', 
        'chemo': 'treatment_date', 
        'radiation': 'treatment_start_date',
        'demographic': 'death_date',
    }
    for dataset, date_col in dataset_map.items():
        df = pd.read_parquet(f'{OUTPUT_DIR}/{dataset}.parquet')
        last_seen_in_database = df.groupby('mrn')[date_col].max().rename(f'{dataset}_last_seen_date')
        last_seen = pd.concat([last_seen, last_seen_in_database], axis=1)
    last_seen['last_seen_date'] = last_seen.max(axis=1)
    last_seen = last_seen.reset_index(names='mrn')
    last_seen.to_parquet(f'{OUTPUT_DIR}/last_seen_dates.parquet', compression='zstd', index=False)
    

def main():
    # get the lab name mapping
    lab_map = load_lab_map(data_dir=INFO_DIR)
    
    # get the mrn mapping
    mrn_map = pd.read_csv(f'{INFO_DIR}/mrn_map.csv')
    id_to_mrn = mrn_map.set_index('PATIENT_RESEARCH_ID')['MRN'].to_dict()

    # get the drug mapping
    drug_map = pd.read_excel(f'{INFO_DIR}/drug_names_normalized_reviewed.xlsx')
    columns = {'type': 'drug_type', 'dose': 'drug_dose', 'unit': 'drug_unit', 'orig_text': 'orig_drug_name'}
    drug_map = drug_map.rename(columns=columns).drop(columns=['failed_output'])
    drug_map = pl.from_pandas(drug_map)

    build_chemo_and_radiation_treatments(id_to_mrn, drug_map)
    build_laboratory_tests(id_to_mrn, lab_map)
    build_symptoms(id_to_mrn)
    build_radiology_reports(id_to_mrn)
    build_acute_care_use()
    build_demographic(id_to_mrn)
    build_last_seen_dates()

    
if __name__ == '__main__':
    main()