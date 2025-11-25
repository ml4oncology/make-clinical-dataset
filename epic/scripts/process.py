"""Process 
1. the pre-epic and epic chemo and radiation therapy datasets
2. the epic symptom dataset
3. the epic emergency department admission dataset
4. the demographic dataset
"""
import glob
import json

import polars as pl
from make_clinical_dataset.shared.constants import ROOT_DIR
from make_clinical_dataset.shared.model import (
    CHEMO_EPIC_COL_MAP,
    CHEMO_PRE_EPIC_COL_MAP,
    RAD_COL_MAP,
)

OUTPUT_DIR = f'{ROOT_DIR}/data/processed'

###############################################################################
# 2025-07-02
###############################################################################
# Paths and Configurations
DATE = '2025-07-02'
RAD_DIR = f"{ROOT_DIR}/data/raw/data_pull_{DATE}/radiation_therapy_csv"
PRE_EPIC_CHEMO_DIR = f'{ROOT_DIR}/data/raw/data_pull_{DATE}/chemo_pre_epic_csv'

# Process radiation data
rad = pl.scan_csv(f'{RAD_DIR}/*.csv', encoding='utf8-lossy')
rad = (
    rad
    .rename(RAD_COL_MAP)
    .with_columns([
        pl.col('treatment_start_date').str.to_datetime(),
        pl.col('treatment_end_date').str.to_datetime(),
    ])
)
rad.sink_parquet(f'{OUTPUT_DIR}/treatment/radiation_{DATE}.parquet')
del rad

# Process the Pre-EPIC chemotherapy
chemo = pl.read_csv(f'{PRE_EPIC_CHEMO_DIR}/*.csv', encoding='utf8-lossy')
chemo = (
    chemo
    .rename(CHEMO_PRE_EPIC_COL_MAP)
    .with_columns([
        # capture the department as a separate column
        pl.col('regimen').str.split('-').list.get(0).alias('department'),
        # add data source
        pl.lit('Pre-EPIC').alias('data_source'),
        # fix dtypes
        pl.col('treatment_date').str.to_datetime(),
        pl.col('first_treatment_date').str.to_datetime(),
    ])
)
# separate the supportive vs chemo care
mask = pl.col('treatment_category').is_in(['Chemo'])
supportive, chemo = chemo.filter(~mask), chemo.filter(mask)
chemo.write_parquet(f'{OUTPUT_DIR}/treatment/chemo_{DATE}.parquet')
supportive.write_parquet(f'{OUTPUT_DIR}/treatment/supportive_{DATE}.parquet')
del chemo, supportive

###############################################################################
# 2025-10-08
###############################################################################
# Paths and Configurations
DATE = '2025-10-08'
DEMOG_DIR = f"{ROOT_DIR}/data/raw/data_pull_{DATE}/demog"

# Process demographic data
# convert multiple jsons into parquet
demog = []
for filepath in glob.glob(f'{DEMOG_DIR}/*.json'):
    with open(filepath, 'r') as file:
        data = json.load(file) # format: [{idx: {col: val}}]
    assert(len(data) == 1)
    demog += list(data[0].values())
demog = pl.DataFrame(demog)
demog.write_parquet(f'{OUTPUT_DIR}/cancer_registry/demographic_{DATE}.parquet')
del demog

###############################################################################
# 2025-11-03
###############################################################################
# Paths and Configurations
DATE = '2025-11-03'
EPIC_CHEMO_DIR = f"{ROOT_DIR}/data/raw/data_pull_{DATE}/chemo_EPIC"
EPIC_ED_DIR = f"{ROOT_DIR}/data/raw/data_pull_{DATE}/ED_EPIC"
EPIC_ESAS_DIR = f"{ROOT_DIR}/data/raw/data_pull_{DATE}/ESAS_EPIC"

# Process EPIC emergency department admissions data
# convert multiple csvs into parquet
ed = pl.read_csv(f'{EPIC_ED_DIR}/*.csv')
ed.write_parquet(f'{OUTPUT_DIR}/ED/ED_{DATE}.parquet')
del ed

# Process EPIC ESAS data
# convert multiple csvs into parquet
esas = pl.read_csv(f'{EPIC_ESAS_DIR}/*.csv')
esas.write_parquet(f'{OUTPUT_DIR}/ESAS/ESAS_{DATE}.parquet')
del esas

# Process the EPIC chemotherapy
chemo = pl.read_csv(f'{EPIC_CHEMO_DIR}/*.csv')
chemo = (
    chemo
    .rename(CHEMO_EPIC_COL_MAP)
    .select(list(CHEMO_EPIC_COL_MAP.values()))
    .with_columns([
        # capture the department as a separate column
        pl.col('regimen').str.split(' ').list.get(0).alias('department'),
        # add data source
        pl.lit('EPIC').alias('data_source'),
        # fix dtypes
        # pl.col('treatment_date').alias('str_treatment_date'), # for debugging
        pl.col('treatment_date').str.to_datetime("%b %d %Y  %I:%M%p"),
        pl.col('scheduled_treatment_date').str.to_datetime(),
        pl.col('first_treatment_date').str.to_datetime(),
    ])
)
chemo.write_parquet(f'{OUTPUT_DIR}/treatment/chemo_{DATE}.parquet')
del chemo