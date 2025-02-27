import os
import subprocess

import pandas as pd
import pytest

from make_clinical_dataset.constants import MAP_CTCAE_LAB
from ml_common.constants import SYMP_COLS

def test_combine_features_mock(tmp_path):
    """NOTE: tmp_path is an inbuilt Pytest fixture and pathlib.Path` object"""
    os.makedirs(f'{tmp_path}/interim')
    os.makedirs(f'{tmp_path}/external')
    mrn = "TEST"
    feats = list(MAP_CTCAE_LAB.values())
    lab = pd.DataFrame([
        [mrn, pd.Timestamp('2020-01-01')] + [i for i, _ in enumerate(feats)],
        [mrn, pd.Timestamp('2020-01-15')] + [i * 3 for i, _ in enumerate(feats)]
    ], columns=['mrn', 'obs_date'] + feats)
    lab.to_parquet(f'{tmp_path}/interim/lab.parquet.gzip', compression='gzip', index=False)
    
    cols = ['mrn', 'treatment_date', 'first_treatment_date', 'drug_d_regimen_dose', 'drug_d_given_dose', 'weight', 'intent']
    trt = pd.DataFrame([[mrn, pd.Timestamp('2020-01-04'), pd.Timestamp('2019-01-01'), 1, 2, 3, 'PALLIATIVE']], columns=cols)
    trt.to_parquet(f'{tmp_path}/interim/treatment.parquet.gzip', compression='gzip', index=False)
    
    dmg = pd.DataFrame([[mrn, pd.Timestamp('2000-01-01'), pd.Timestamp('2020-01-01'), 4, 5, 6]], columns=['mrn', 'date_of_birth', 'date_of_death', 'feat4', 'feat5', 'feat6'])
    dmg.to_parquet(f'{tmp_path}/interim/demographic.parquet.gzip', compression='gzip', index=False)
    
    sym = pd.DataFrame([
        [mrn, pd.Timestamp('2020-01-01')] + [i for i, _ in enumerate(SYMP_COLS)],
        [mrn, pd.Timestamp('2020-01-15')] + [i + 3 for i, _ in enumerate(SYMP_COLS)]
    ], columns=['mrn', 'survey_date'] + SYMP_COLS)
    sym.to_parquet(f'{tmp_path}/interim/symptom.parquet.gzip', compression='gzip', index=False)
    
    erv = pd.DataFrame([[mrn, pd.Timestamp('2020-01-01'), 10, 11, 12]], columns=['mrn', 'event_date', 'feat10', 'feat11', 'feat12'])
    erv.to_parquet(f'{tmp_path}/interim/emergency_room_visit.parquet.gzip', compression='gzip', index=False)
    
    drug_list = pd.DataFrame([['d', 'regimen_dose * weight', 1, 'INCLUDE']], columns=['name', 'recommended_dose_formula', 'counts', 'category'])
    drug_list.to_csv(f'{tmp_path}/external/opis_drug_list.csv', index=False)

    last_seen = pd.DataFrame([pd.Timestamp('2021-01-01')], index=[mrn], columns=['last_seen_date'])
    last_seen.to_parquet(f'{tmp_path}/interim/last_seen_dates.parquet.gzip', compression='gzip')
    
    cmd = ['python', 'scripts/unify.py', '--data-dir', tmp_path, '--output-dir', tmp_path]
    result = subprocess.run(cmd, capture_output=True)
    assert result.returncode == 0

@pytest.mark.xfail
def test_combine_features(tmp_path):
    """NOTE: tmp_path is an inbuilt Pytest fixture and pathlib.Path` object"""
    cmd = ['python', 'scripts/unify.py', '--output-dir', tmp_path]
    result = subprocess.run(cmd, capture_output=True)
    assert result.returncode == 0