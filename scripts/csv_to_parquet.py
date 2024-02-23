"""
Script to convert the large csv files to parquet files
Move them from 2BLAST folder in the H4H cluster to the data/raw directory

Comparison of csv file format vs parquet file format:
patient_radiology_records_merged: 1.7GB -> 129MB
patient_biochemistry_merged: 4.9GB -> 229MB
patient_hematology_merged: 2.6GB -> 83MB
"""
from pathlib import Path
import glob
import os

import pandas as pd

ROOT_DIR = Path(__file__).parent.parent.as_posix()

def main():
    data_root_dir = '/cluster/projects/gliugroup'
    merged_dir = f'{data_root_dir}/2BLAST/Merged and Cleaned Datasets'
    raw_dir = f'{data_root_dir}/2BLAST/Raw Data'
    biochem_dir = f'{merged_dir}/merged_data/biochemistry'
    hemo_dir = f'{merged_dir}/merged_data/hematology'
    dart_dir = f'{raw_dir}/DART/From CDI'
    opis_dir = f'{raw_dir}/OPIS'
    other_epr_dir = f'{raw_dir}/Other EPR Data'
    canc_reg_dir = f'{raw_dir}/Cancer Registry Data/Regsitry Pulls February 24 2022/'
    biochem_file = f'{biochem_dir}/patient_biochemistry_merged.csv'
    hema_file = f'{hemo_dir}/patient_hematology_merged.csv'
    dart_file = f'{dart_dir}/dart-data-final.csv'
    opis_file = f'{opis_dir}/OPIS_data_29Nov2022.xlsx'
    rad_file = f'{merged_dir}/patient_radiology_records_merged.csv'
    emergency_department_file = f'{other_epr_dir}/ED/Request_1481_ED_Visits_deathdate.csv'
    emergency_room_file = f'{other_epr_dir}/ED/Request_1481_ERTriageAssmt.csv'
    death_file = f'{other_epr_dir}/Death Data/EPR_death_pts.csv'

    save_dir = f'{ROOT_DIR}/data/raw/'
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    mapping = {rad_file: 'radiology', biochem_file: 'biochemistry', hema_file: 'hematology'}
    for filepath, filename in mapping.items():
        df = pd.read_csv(f'{filepath}')
        df = df.drop(columns=['Unnamed: 0'], errors='ignore')
        if 'proc_code' in df.columns: df['proc_code'] = df['proc_code'].astype(str)
        df.to_parquet(f'{save_dir}/{filename}.parquet.gzip', compression='gzip', index=False)

    df = pd.read_excel(opis_file)
    df.to_parquet(f'{save_dir}/opis.parquet.gzip', compression='gzip', index=False)

    df = pd.read_csv(dart_file)
    df = df.drop(columns=['Unnamed: 22'])
    df.to_parquet(f'{save_dir}/dart.parquet.gzip', compression='gzip', index=False)

    df = pd.concat([pd.read_excel(filepath) for filepath in glob.glob(f'{canc_reg_dir}/*')])
    df['BRM_START'] = pd.to_datetime(df['BRM_START'], errors='coerce')
    df['INSURANCE_NUMBER'] = df['INSURANCE_NUMBER'].astype(str)
    df.to_parquet(f'{save_dir}/cancer_registry.parquet.gzip', compression='gzip', index=False)

    df = pd.read_csv(emergency_department_file)
    df.to_parquet(f'{save_dir}/ED.parquet.gzip', compression='gzip', index=False)
    df = pd.read_csv(emergency_room_file)
    df.to_parquet(f'{save_dir}/ER.parquet.gzip', compression='gzip', index=False)

    df = pd.read_csv(death_file)
    df.to_parquet(f'{save_dir}/death_dates.parquet.gzip', compression='gzip', index=False)
    
if __name__ == '__main__':
    main()