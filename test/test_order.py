import pandas as pd
import pytest
from tqdm import tqdm

@pytest.mark.xfail
class TestOrdering():
    def test_ER(self):
        df = pd.read_parquet('./data/interim/emergency_room_visit.parquet.gzip')
        date_is_ordered(df, date_col='event_date', patient_col='mrn')

    def test_lab(self):
        df = pd.read_parquet('./data/interim/lab.parquet.gzip')
        date_is_ordered(df, date_col='obs_date', patient_col='mrn')

    def test_symptom(self):
        df = pd.read_parquet('./data/interim/symptom.parquet.gzip')
        date_is_ordered(df, date_col='survey_date', patient_col='mrn')

    def test_treatment(self):
        df = pd.read_parquet('./data/interim/treatment.parquet.gzip')
        date_is_ordered(df, date_col='treatment_date', patient_col='mrn')

def date_is_ordered(df: pd.DataFrame, date_col: str, patient_col: str) -> None:
    """Ensure that for each patient, the date is in ascending order"""
    for mrn, group in tqdm(df.groupby(patient_col)):
        assert all(group[date_col].to_numpy() == group[date_col].sort_values().to_numpy())