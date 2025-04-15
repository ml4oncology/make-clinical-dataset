"""
Script to turn raw data into features / targets for modelling
"""
import argparse
import os

import pandas as pd

from make_clinical_dataset.preprocess.epic.lab import get_lab_data

# Paths and Configurations
DATE = '2025-03-29'
ROOT_DIR = '/cluster/projects/gliugroup/2BLAST'
INFO_DIR = f'{ROOT_DIR}/data/info'
LAB_DIR = f'{ROOT_DIR}/data/processed/lab/lab_{DATE}'
OUTPUT_DIR = f'{ROOT_DIR}/data/final/data_{DATE}/interim'

def main():
    # get the lab name mapping
    lab_map = load_lab_map(data_dir=INFO_DIR)
    
    # get the mrn mapping
    mrn_map = pd.read_csv(f'{INFO_DIR}/mrn_map.csv')
    mrn_map = mrn_map.set_index('PATIENT_RESEARCH_ID')['MRN'].to_dict()

    # laboratory tests
    lab = get_lab_data(mrn_map, lab_map, data_dir=LAB_DIR)
    lab.write_parquet(f'{OUTPUT_DIR}/lab.parquet')

    
if __name__ == '__main__':
    main()