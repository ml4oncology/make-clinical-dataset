"""Process the chemo and radiation therapy datasets"""
from glob import glob

import pandas as pd
from make_clinical_dataset.constants import ROOT_DIR, TRT_INTENT
from make_clinical_dataset.model import (
    CHEMO_EPIC_COL_MAP,
    CHEMO_PRE_EPIC_COL_MAP,
    RAD_COL_MAP,
)

# Paths and Configurations
DATE = '2025-07-02'
DATA_DIR = f'{ROOT_DIR}/data/raw/data_pull_{DATE}'
OUTPUT_DIR = f'{ROOT_DIR}/data/processed/treatment'


# Read all CSV files in the folder
def read_csvs(folder: str) -> pd.DataFrame:
    paths = sorted(glob(f'{DATA_DIR}/{folder}/*.csv'))
    return pd.concat([pd.read_csv(path, encoding='cp1252') for path in paths], ignore_index=True)
ct_pre_epic = read_csvs('chemo_pre_epic_csv')
ct_epic = read_csvs('chemo_epic_csv')
rad = read_csvs('radiation_therapy_csv')


# Process the Pre-EPIC chemotherapy
# rename the columns
chemo_pre_epic = ct_pre_epic.rename(columns=CHEMO_PRE_EPIC_COL_MAP)
# clean intent feature
chemo_pre_epic['intent'] = chemo_pre_epic['intent'].map(TRT_INTENT)
# cature the department as a separate column
chemo_pre_epic['department'] = chemo_pre_epic['regimen'].str.split('-').str[0]
# add data source
chemo_pre_epic['data_source'] = 'Pre-EPIC'


# Process the EPIC chemotherapy
# rename the columns
chemo_epic = ct_epic.rename(columns=CHEMO_EPIC_COL_MAP)
# clean intent feature
chemo_epic['intent'] = chemo_epic['intent'].str.lower()
# cature the department as a separate column
chemo_epic['department'] = chemo_epic['regimen'].str.split(' ').str[0]
# filter treatment dates past DATE (are they scheduled treatments?)
chemo_epic = chemo_epic[chemo_epic['treatment_date'] <= DATE]
# add data source
chemo_epic['data_source'] = 'EPIC'


# Combine the chemotherapies (Pre-EPIC and EPIC)
chemo = pd.concat([chemo_pre_epic, chemo_epic], ignore_index=True)


# Process the combined chemotherapy
# convert to datetime
chemo['treatment_date'] = pd.to_datetime(chemo['treatment_date'])
chemo['first_treatment_date'] = pd.to_datetime(chemo['first_treatment_date'])
# replace departments that did not exist pre-epic as None
mask = chemo['department'].isin(chemo_pre_epic['department'])
chemo.loc[~mask, 'department'] = None


# Process the radiation therapy
rad = rad.rename(columns=RAD_COL_MAP)
rad['treatment_start_date'] = pd.to_datetime(rad['treatment_start_date'])
rad['treatment_end_date'] = pd.to_datetime(rad['treatment_end_date'])


# Write chemotherapy dataset
chemo.to_parquet(f'{OUTPUT_DIR}/chemo_{DATE}.parquet', compression='zstd', index=False)


# Write radiation therapy dataset
rad.to_parquet(f'{OUTPUT_DIR}/radiation_{DATE}.parquet', compression='zstd', index=False)