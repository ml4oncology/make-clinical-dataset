import os
import subprocess

import pandas as pd
import pytest

def test_combine_features_mock(tmp_path):
    """NOTE: tmp_path is an inbuilt Pytest fixture and pathlib.Path` object"""
    os.makedirs(f'{tmp_path}/interim')
    os.makedirs(f'{tmp_path}/external')
    mrn = "TEST"
    lab = pd.DataFrame([[mrn, pd.Timestamp('2020-01-01'), 1, 2, 3]], columns=['mrn', 'obs_date', 'feat1', 'feat2', 'feat3'])
    lab.to_parquet(f'{tmp_path}/interim/lab.parquet.gzip', compression='gzip', index=False)
    
    cols = ['mrn', 'treatment_date', 'first_treatment_date', 'drug_d_regimen_dose', 'drug_d_given_dose', 'weight', 'intent']
    trt = pd.DataFrame([[mrn, pd.Timestamp('2020-01-04'), pd.Timestamp('2019-01-01'), 1, 2, 3, 'PALLIATIVE']], columns=cols)
    trt.to_parquet(f'{tmp_path}/interim/treatment.parquet.gzip', compression='gzip', index=False)
    
    dmg = pd.DataFrame([[mrn, pd.Timestamp('2000-01-01'), 4, 5, 6]], columns=['mrn', 'date_of_birth', 'feat4', 'feat5', 'feat6'])
    dmg.to_parquet(f'{tmp_path}/interim/demographic.parquet.gzip', compression='gzip', index=False)
    
    sym = pd.DataFrame([[mrn, pd.Timestamp('2020-01-01'), 7, 8, 9]], columns=['mrn', 'survey_date', 'feat7', 'feat8', 'feat9'])
    sym.to_parquet(f'{tmp_path}/interim/symptom.parquet.gzip', compression='gzip', index=False)
    
    erv = pd.DataFrame([[mrn, pd.Timestamp('2020-01-01'), 10, 11, 12]], columns=['mrn', 'event_date', 'feat10', 'feat11', 'feat12'])
    erv.to_parquet(f'{tmp_path}/interim/emergency_room_visit.parquet.gzip', compression='gzip', index=False)
    
    drug_list = pd.DataFrame([['d', 'regimen_dose * weight', 1, 'INCLUDE']], columns=['name', 'recommended_dose_formula', 'counts', 'category'])
    drug_list.to_csv(f'{tmp_path}/external/opis_drug_list.csv', index=False)
    
    cmd = ['python', 'scripts/combine_features.py', '--data-dir', tmp_path, '--output-dir', tmp_path]
    result = subprocess.run(cmd, capture_output=True)
    assert result.returncode == 0

@pytest.mark.xfail
def test_combine_features(tmp_path):
    """NOTE: tmp_path is an inbuilt Pytest fixture and pathlib.Path` object"""
    cmd = ['python', 'scripts/combine_features.py', '--output-dir', tmp_path]
    result = subprocess.run(cmd, capture_output=True)
    assert result.returncode == 0