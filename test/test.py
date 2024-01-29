import unittest

import pandas as pd

from .util import date_is_ordered

class Tester(unittest.TestCase):
    def test_ER(self):
        df = pd.read_parquet('./data/interim/emergency_room_visit.parquet.gzip')
        date_is_ordered(df, date_col='event_date', patient_col='mrn')

    def test_ED(self):
        df = pd.read_parquet('./data/interim/emergency_department_visit.parquet.gzip')
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

if __name__ == '__main__':
    """
    > python -m unittest test/test.py
    Reference: docs.python.org/3/library/unittest.html
    """
    unittest.main()