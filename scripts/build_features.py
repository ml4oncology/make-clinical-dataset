"""
Script to turn raw data into features for modelling
"""
from pathlib import Path
import argparse
import os
import sys
ROOT_DIR = Path(__file__).parent.parent.as_posix()
sys.path.append(ROOT_DIR)

import pandas as pd

from src.preprocess.cancer_registry import get_demographic_data
from src.preprocess.dart import get_symptoms_data
from src.preprocess.lab import get_lab_data
from src.preprocess.opis import get_treatment_data
from src.util import load_included_drugs, load_included_regimens

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=f'{ROOT_DIR}/data')
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
    dart, dart_demog = get_symptoms_data(data_dir=f'{data_dir}/raw')
    dart.to_parquet(f'{data_dir}/interim/symptom.parquet.gzip', compression='gzip', index=False)

    # demographics
    canc_reg = get_demographic_data(data_dir=f'{data_dir}/raw', external_data=dart_demog)
    canc_reg.to_parquet(f'{data_dir}/interim/demographic.parquet.gzip', compression='gzip', index=False)

    # treatment
    opis = get_treatment_data(included_drugs, included_regimens, data_dir=f'{data_dir}/raw')
    opis.to_parquet(f'{data_dir}/interim/treatment.parquet.gzip', compression='gzip', index=False)

    # laboratory tests
    lab = get_lab_data(mrn_map, data_dir=f'{data_dir}/raw')
    lab.to_parquet(f'{data_dir}/interim/lab.parquet.gzip', compression='gzip', index=False)
    
if __name__ == '__main__':
    main()