"""Script to turn raw data into features / targets for modelling"""

import pandas as pd
import polars as pl
from make_clinical_dataset.epic.preprocess.acu import get_acu_data
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
LAB_DIR = f'{ROOT_DIR}/data/processed/lab/lab_{DATE}'
PRE_EPIC_ESAS_DIR = f'{ROOT_DIR}/data/processed/ESAS/ESAS_{DATE}'
RAD_DIR = f'{ROOT_DIR}/data/processed/radiology/radiology_{DATE}'
OUTPUT_DIR = f'{ROOT_DIR}/data/final/data_{DATE}/interim'

DATE = '2025-07-02'
CHEMO_PATH = f'{ROOT_DIR}/data/processed/treatment/chemo_{DATE}.parquet'
RT_PATH = f'{ROOT_DIR}/data/processed/treatment/radiation_{DATE}.parquet'

DATE = '2025-10-08'
EPIC_ESAS_PATH = f'{ROOT_DIR}/data/processed/ESAS/ESAS_{DATE}.parquet'


def build_chemo_and_radiation_treatments(id_to_mrn: dict[str, int], drug_map: pl.LazyFrame):
    chemo = get_chemo_data(CHEMO_PATH, id_to_mrn, drug_map)
    chemo.sink_parquet(f'{OUTPUT_DIR}/chemo.parquet')
    # chemo.to_parquet(f'{OUTPUT_DIR}/chemo.parquet')
    rad = get_radiation_data(RT_PATH, id_to_mrn)
    rad.to_parquet(f'{OUTPUT_DIR}/radiation.parquet', compression='zstd', index=False)


def build_laboratory_tests(id_to_mrn: dict[str, int], lab_map: dict[str, str]):
    lab = get_lab_data(id_to_mrn, lab_map, data_dir=LAB_DIR)
    lab.write_parquet(f'{OUTPUT_DIR}/lab.parquet')


def build_symptoms(id_to_mrn: dict[str, int]):
    pre_epic_symp = get_symp_data(id_to_mrn, data_dir=PRE_EPIC_ESAS_DIR)
    epic_symp = get_epic_symp_data(EPIC_ESAS_PATH, id_to_mrn)
    symp = pd.concat([pre_epic_symp, epic_symp])
    assert not symp.duplicated(subset=['mrn', 'obs_date']).any()
    symp.to_parquet(f'{OUTPUT_DIR}/symptom.parquet', compression='zstd', index=False)


def build_radiology_reports(id_to_mrn: dict[str, int]):
    reports = get_radiology_data(id_to_mrn, data_dir=RAD_DIR)
    reports.write_parquet(f'{OUTPUT_DIR}/reports.parquet')
    

def build_acute_care_use():
    acu = get_acu_data(ACU_PATH)
    acu.sink_parquet(f'{OUTPUT_DIR}/acute_care_use.parquet')
    

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
    drug_map = pl.from_pandas(drug_map).lazy()

    build_chemo_and_radiation_treatments(id_to_mrn, drug_map)
    build_laboratory_tests(id_to_mrn, lab_map)
    build_symptoms(id_to_mrn)
    build_radiology_reports(id_to_mrn)
    build_acute_care_use()

    
if __name__ == '__main__':
    main()