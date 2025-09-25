"""
Script to turn raw data into features / targets for modelling
"""
import argparse
import os

import pandas as pd
from make_clinical_dataset.epr.preprocess.cancer_registry import get_demographic_data
from make_clinical_dataset.epr.preprocess.clinic import get_clinical_notes_data
from make_clinical_dataset.epr.preprocess.dart import get_symptoms_data
from make_clinical_dataset.epr.preprocess.emergency import get_emergency_room_data
from make_clinical_dataset.epr.preprocess.lab import get_lab_data
from make_clinical_dataset.epr.preprocess.opis import get_treatment_data
from make_clinical_dataset.epr.preprocess.radiology import get_radiology_data
from make_clinical_dataset.epr.preprocess.recist import get_recist_data
from make_clinical_dataset.epr.util import load_included_drugs, load_included_regimens


def get_last_seen_dates(data_dir: str):
    last_seen = pd.DataFrame()
    dataset_map = {
        'lab': 'obs_date', 
        'symptom': 'survey_date', 
        'treatment': 'treatment_date', 
        'demographic': 'last_contact_date'
    }
    for dataset, date_col in dataset_map.items():
        df = pd.read_parquet(f'{data_dir}/{dataset}.parquet')
        last_seen_in_database = df.groupby('mrn')[date_col].max().rename(f'{dataset}_last_seen_date')
        last_seen = pd.concat([last_seen, last_seen_in_database], axis=1)
    last_seen['last_seen_date'] = last_seen.max(axis=1)
    return last_seen


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data_dir = args.data_dir
    if not os.path.exists(f'{data_dir}/interim'): os.makedirs(f'{data_dir}/interim')

    included_drugs = load_included_drugs(data_dir=f'{data_dir}/external')
    included_regimens = load_included_regimens(data_dir=f'{data_dir}/external')
    mrn_map = pd.read_csv(f'{data_dir}/external/MRN_map.csv')
    mrn_map = mrn_map.set_index('RESEARCH_ID')['PATIENT_MRN'].to_dict()

    # symptoms
    dart = get_symptoms_data(data_dir=f'{data_dir}/raw')
    dart.to_parquet(f'{data_dir}/interim/symptom.parquet', compression='zstd', index=False)

    # demographics
    canc_reg = get_demographic_data(data_dir=f'{data_dir}/raw')
    canc_reg.to_parquet(f'{data_dir}/interim/demographic.parquet', compression='zstd', index=False)

    # treatment
    opis = get_treatment_data(included_drugs, included_regimens, data_dir=f'{data_dir}/raw')
    opis.to_parquet(f'{data_dir}/interim/treatment.parquet', compression='zstd', index=False)

    # laboratory tests
    lab = get_lab_data(mrn_map, data_dir=f'{data_dir}/raw')
    lab.to_parquet(f'{data_dir}/interim/lab.parquet', compression='zstd', index=False)

    # emergency room visits
    er_visit = get_emergency_room_data(data_dir=f'{data_dir}/raw')
    er_visit.to_parquet(f'{data_dir}/interim/emergency_room_visit.parquet', compression='zstd', index=False)

    # radiology reports
    reports = get_radiology_data(mrn_map, data_dir=f'{data_dir}/raw')
    reports.to_parquet(f'{data_dir}/interim/reports.parquet', compression='zstd', index=False)

    # clinical notes
    clinical_notes = get_clinical_notes_data(data_dir=f'{data_dir}/raw')
    clinical_notes.to_parquet(f'{data_dir}/data/interim/clinical_notes.parquet', compression='zstd', index=False)

    # tumor response - COMPASS trial
    recist = get_recist_data(data_dir=f'{data_dir}/external')
    recist.to_parquet(f'{data_dir}/interim/recist.parquet', compression='zstd', index=False)

    # last seen date in each dataset
    last_seen = get_last_seen_dates(data_dir=f'{data_dir}/interim')
    last_seen.to_parquet(f'{data_dir}/interim/last_seen_dates.parquet', compression='zstd')

    
if __name__ == '__main__':
    main()