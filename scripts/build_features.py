"""
Script to turn raw data into features for modelling
"""
import argparse
import os

import pandas as pd

from make_clinical_dataset.preprocess.cancer_registry import get_demographic_data
from make_clinical_dataset.preprocess.dart import get_symptoms_data
from make_clinical_dataset.preprocess.emergency import get_emergency_room_data
from make_clinical_dataset.preprocess.lab import get_lab_data
from make_clinical_dataset.preprocess.opis import get_treatment_data
from make_clinical_dataset.preprocess.radiology import get_radiology_data
from make_clinical_dataset.util import load_included_drugs, load_included_regimens

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
    dart.to_parquet(f'{data_dir}/interim/symptom.parquet.gzip', compression='gzip', index=False)

    # demographics
    canc_reg = get_demographic_data(data_dir=f'{data_dir}/raw')
    canc_reg.to_parquet(f'{data_dir}/interim/demographic.parquet.gzip', compression='gzip', index=False)

    # treatment
    opis = get_treatment_data(included_drugs, included_regimens, data_dir=f'{data_dir}/raw')
    opis.to_parquet(f'{data_dir}/interim/treatment.parquet.gzip', compression='gzip', index=False)

    # laboratory tests
    lab = get_lab_data(mrn_map, data_dir=f'{data_dir}/raw')
    lab.to_parquet(f'{data_dir}/interim/lab.parquet.gzip', compression='gzip', index=False)

    # emergency room visits
    er_visit = get_emergency_room_data(data_dir=f'{data_dir}/raw')
    er_visit.to_parquet(f'{data_dir}/interim/emergency_room_visit.parquet.gzip', compression='gzip', index=False)

    # radiology reports
    reports = get_radiology_data(mrn_map, data_dir=f'{data_dir}/raw')
    reports.to_parquet(f'{data_dir}/interim/reports.parquet.gzip', compression='gzip', index=False)
    
if __name__ == '__main__':
    main()