"""
Script to turn raw data into features / targets for modelling
"""
import argparse
import os

import pandas as pd

from make_clinical_dataset.preprocess.epic.lab import get_lab_data
from make_clinical_dataset.preprocess.epic.esas import get_symp_data
from make_clinical_dataset.util import load_lab_map

# Paths and Configurations
DATE = '2025-03-29'
ROOT_DIR = '/cluster/projects/gliugroup/2BLAST'
INFO_DIR = f'{ROOT_DIR}/data/info'
LAB_DIR = f'{ROOT_DIR}/data/processed/lab/lab_{DATE}'
ESAS_DIR = f'{ROOT_DIR}/data/processed/ESAS/ESAS_{DATE}'
OUTPUT_DIR = f'{ROOT_DIR}/data/final/data_{DATE}/interim'

def main():
    # get the lab name mapping
    lab_map = load_lab_map(data_dir=INFO_DIR)
    
    # get the mrn mapping
    mrn_map = pd.read_csv(f'{INFO_DIR}/mrn_map.csv')
    mrn_map = mrn_map.set_index('PATIENT_RESEARCH_ID')['MRN'].to_dict()

    # laboratory tests
    lab = get_lab_data(mrn_map, lab_map, data_dir=LAB_DIR)
    lab.write_parquet(f'{OUTPUT_DIR}/lab.parquet') #TODO: change unify.py to load parquet file, remove gzip compression, convert epr data to ztsd parquet

    # symptoms
    symp = get_symp_data(mrn_map, data_dir=ESAS_DIR)
    symp.to_parquet(f'{output_dir}/symptom.parquet', compression='zstd', index=False)

    
if __name__ == '__main__':
    main()