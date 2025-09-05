import pandas as pd
import pytest
from make_clinical_dataset.shared.constants import CTCAE_CONSTANTS, MAP_CTCAE_LAB, SYMP_COLS
from make_clinical_dataset.shared.constants import (
    CTCAE_CONSTANTS,
    MAP_CTCAE_LAB,
    SYMP_COLS,
)

@pytest.fixture
def df():
    return pd.read_parquet('./data/processed/treatment_centered_dataset.parquet.gzip')

@pytest.fixture
def sym():
    return pd.read_parquet('./data/interim/symptom.parquet.gzip')

@pytest.fixture
def lab():
    return pd.read_parquet('./data/interim/lab.parquet.gzip')

@pytest.mark.xfail
def test_ed_visit_label(df):
    tmp = df[df['target_ED_30d']]
    assert all((tmp['target_ED_date'] - tmp['assessment_date']).dt.days < 30)
    tmp = df[~df['target_ED_30d']]
    assert all((tmp['target_ED_date'].fillna(pd.Timestamp.max) - tmp['assessment_date']).dt.days >= 30)

@pytest.mark.xfail
@pytest.mark.parametrize("col", SYMP_COLS)
def test_symptom_deterioration_label(df, sym, col):
    # make sure score change is indeed greater than or equal to 3
    tmp = df[df[f'target_{col}_3pt_change'] == 1]
    assert all((tmp[f'target_{col}_max'] - tmp[col]) >= 3)
    
    # make sure the extracted lookahead values in the final dataset aligns with the values in the symptom dataset
    sampled_data = tmp.sample(min(100, len(tmp)), random_state=42)[['mrn', 'assessment_date', f'target_{col}_max']].to_numpy()
    for mrn, assessment_date, lookahead_val in sampled_data:
        begin, end = assessment_date + pd.Timedelta(days=1), assessment_date + pd.Timedelta(days=30)
        assert sym.query('mrn == @mrn & @begin <= survey_date <= @end')[col].max() == lookahead_val

@pytest.mark.xfail
@pytest.mark.parametrize("targ", ['hemoglobin', 'neutrophil', 'platelet'])
def test_ctcae_lab_values(df, lab, targ):
    col = MAP_CTCAE_LAB[targ]
    
    # make sure target follows the CTCAE definitions
    tmp = df[df[f'target_{targ}_grade2plus'] == 1]
    assert all(tmp[f'target_{col}_min'] < CTCAE_CONSTANTS[targ]['grade2plus'])
    
    tmp = df[df[f'target_{targ}_grade3plus'] == 1]
    assert all(tmp[f'target_{col}_min'] < CTCAE_CONSTANTS[targ]['grade3plus'])
    
    # make sure the extracted lookahead values in the final dataset aligns with the values in the symptom dataset
    sampled_data = tmp.sample(min(100, len(tmp)), random_state=42)[['mrn', 'assessment_date', f'target_{col}_min']].to_numpy()
    for mrn, assessment_date, lookahead_val in sampled_data:
        begin, end = assessment_date + pd.Timedelta(days=1), assessment_date + pd.Timedelta(days=30)
        assert lab.query('mrn == @mrn & @begin <= obs_date <= @end')[col].min() == lookahead_val

@pytest.mark.xfail
@pytest.mark.parametrize("targ", ['bilirubin', 'AKI', 'AST', 'ALT'])
def test_ctcae_lab_uln(df, lab, targ):
    col = MAP_CTCAE_LAB[targ]
    ULN = CTCAE_CONSTANTS[targ]['ULN']
    clip_kwargs = dict(upper=ULN) if targ == 'AKI' else dict(lower=ULN)
    
    # make sure target follows the CTCAE definitions
    tmp = df[df[f'target_{targ}_grade2plus'] == 1]
    base = tmp[col].fillna(ULN).clip(**clip_kwargs)
    assert all(tmp[f'target_{col}_max'] > CTCAE_CONSTANTS[targ]['grade2plus'] * base)
    
    tmp = df[df[f'target_{targ}_grade3plus'] == 1]
    base = tmp[col].fillna(ULN).clip(**clip_kwargs)
    assert all(tmp[f'target_{col}_max'] > CTCAE_CONSTANTS[targ]['grade3plus'] * base)
    
    # make sure the extracted lookahead values in the final dataset aligns with the values in the symptom dataset
    sampled_data = tmp.sample(min(100, len(tmp)), random_state=42)[['mrn', 'assessment_date', f'target_{col}_max']].to_numpy()
    for mrn, assessment_date, lookahead_val in sampled_data:
        begin, end = assessment_date + pd.Timedelta(days=1), assessment_date + pd.Timedelta(days=30)
        assert lab.query('mrn == @mrn & @begin <= obs_date <= @end')[col].max() == lookahead_val
