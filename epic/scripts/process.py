"""Process 
1. the pre-epic and epic chemo and radiation therapy datasets
2. the epic symptom dataset
3. the demographic dataset
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

# Process the Pre-EPIC chemotherapy (combine with EPIC chemo later)
chemo_pre_epic = pl.scan_csv(f'{PRE_EPIC_CHEMO_DIR}/*.csv', encoding='utf8-lossy')
chemo_pre_epic = (
    chemo_pre_epic
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


###############################################################################
# 2025-10-08
###############################################################################
# Paths and Configurations
DATE = '2025-10-08'
EPIC_CHEMO_DIR = f"{ROOT_DIR}/data/raw/data_pull_{DATE}/chemo"
EPIC_ESAS_DIR = f"{ROOT_DIR}/data/raw/data_pull_{DATE}/ESAS"
DEMOG_DIR = f"{ROOT_DIR}/data/raw/data_pull_{DATE}/demog"

# Process EPIC ESAS data
# convert multiple csvs into parquet
esas = pl.read_csv(f'{EPIC_ESAS_DIR}/*.csv')
esas.write_parquet(f'{OUTPUT_DIR}/ESAS/ESAS_{DATE}.parquet')
del esas

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

# Process the EPIC chemotherapy
chemo_epic = pl.scan_csv(f'{EPIC_CHEMO_DIR}/*.csv')
num_cols = [
    'weight', 'height', 'body_surface_area', 'cycle_number', 
    'given_dose', 'diluent_volume', 'percentage_of_ideal_dose'
]
date_cols = ['treatment_date', 'scheduled_treatment_date', 'first_scheduled_treatment_date']
chemo_epic = (
    chemo_epic
    .rename(CHEMO_EPIC_COL_MAP)
    .select(list(CHEMO_EPIC_COL_MAP.values()))
    .with_columns([
        # capture the department as a separate column
        pl.col('regimen').str.split(' ').list.get(0).alias('department'),
        # add data source
        pl.lit('EPIC').alias('data_source'),
        # fix dtypes
        *[pl.col(col).cast(pl.Float64) for col in num_cols],
        *[pl.col(col).str.to_datetime() for col in date_cols]
    ])
)

# Combine the chemotherapies (Pre-EPIC and EPIC)
chemo = pl.concat([chemo_pre_epic, chemo_epic], how="diagonal")

# Process the combined chemotherapy
# replace departments that did not exist pre-epic as None
pre_epic_deps = chemo_pre_epic.select(
    pl.col("department").unique()
).collect().to_numpy().flatten()
chemo = chemo.with_columns(
    pl.when(pl.col("department").is_in(pre_epic_deps))
    .then(pl.col("department"))
    .otherwise(None)
)

# Separate the supportive vs chemo care
mask = pl.col('treatment_category').is_in(['Chemotherapy', 'Chemo', 'Take Home Cancer Drugs'])
supportive = chemo.filter(~mask)
chemo = chemo.filter(mask)

# Write chemotherapy dataset
chemo.collect().write_parquet(f'{OUTPUT_DIR}/treatment/chemo_{DATE}.parquet')
supportive.collect().write_parquet(f'{OUTPUT_DIR}/treatment/supportive_{DATE}.parquet')