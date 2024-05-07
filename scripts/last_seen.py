"""
Script to gather patient's last seen date in each database
"""
from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent.as_posix()

import pandas as pd

def main():
    last_seen = pd.DataFrame()
    database_map = {'lab': 'obs_date', 'symptom': 'survey_date', 'treatment': 'treatment_date', 'demographic': 'last_contact_date'}
    for database, date_col in database_map.items():
        df = pd.read_parquet(f'{ROOT_DIR}/data/interim/{database}.parquet.gzip')
        last_seen_in_database = df.groupby('mrn')[date_col].max().rename(f'{database}_last_seen_date')
        last_seen = pd.concat([last_seen, last_seen_in_database], axis=1)
    last_seen['last_seen_date'] = last_seen.max(axis=1)
    last_seen.to_parquet(f'{ROOT_DIR}/data/processed/last_seen_dates.parquet.gzip', compression='gzip')
    
if __name__ == '__main__':
    main()