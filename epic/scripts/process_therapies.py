"""Process the chemo and radiation therapy datasets"""
import polars as pl
from make_clinical_dataset.shared.constants import ROOT_DIR
from make_clinical_dataset.epic.model import (
    CHEMO_EPIC_COL_MAP,
    CHEMO_PRE_EPIC_COL_MAP,
    RAD_COL_MAP,
)

# Paths and Configurations
DATE = '2025-07-02'
DATA_DIR = f'{ROOT_DIR}/data/raw/data_pull_{DATE}'
OUTPUT_DIR = f'{ROOT_DIR}/data/processed/treatment'


# Read all CSV files in the folder
def read_csvs(folder):
    # utf8-lossy can cause data loss, but mostly with non utf8 characters, so not losing out on much
    # 2025-07-02 - the only character we lost is a non-utf8 em dash (– was replaced with �)
    return pl.scan_csv(f'{DATA_DIR}/{folder}/*.csv', encoding='utf8-lossy').lazy()
chemo_pre_epic = read_csvs('chemo_pre_epic_csv')
chemo_epic = read_csvs('chemo_epic_csv')
rad = read_csvs('radiation_therapy_csv')


# Process the Pre-EPIC chemotherapy
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
# separate the pre/post chemo care (all are Antiemetic (AE) protocols)
mask = pl.col('treatment_type').is_in(['Pre', 'Post'])
pre_post_chemo = chemo_pre_epic.filter(mask)
chemo_pre_epic = chemo_pre_epic.filter(~mask).drop('treatment_type')


# Process the EPIC chemotherapy
chemo_epic = (
    chemo_epic
    .rename(CHEMO_EPIC_COL_MAP)
    .with_columns([
        # capture the department as a separate column
        pl.col('regimen').str.split(' ').list.get(0).alias('department'),
        # add data source
        pl.lit('EPIC').alias('data_source'),
        # merge drug names together
        pl.concat_str(
            [pl.col("drug_name"), pl.col("drug_name_ext")], separator=" - ", ignore_nulls=True
        ).alias("drug_name_ext"),
        # fix dtypes
        pl.col("height").cast(pl.Float64),
        pl.col("cycle_number").cast(pl.Float64),
        pl.col('treatment_date').str.to_datetime(),
        pl.col('first_treatment_date').str.to_datetime(),
    ])
    .drop(['generic_name_strength']) # generic_name_strength is the same as given_dose
)
# filter treatment dates past DATE (are they scheduled treatments?)
chemo_epic = chemo_epic.filter(pl.col('treatment_date') <= pl.lit(DATE).cast(pl.Date))


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

# Process the radiation therapy
rad = (
    rad
    .rename(RAD_COL_MAP)
    .with_columns([
        pl.col('treatment_start_date').str.to_datetime(),
        pl.col('treatment_end_date').str.to_datetime(),
    ])
)

# Write chemotherapy dataset
# TODO: why does sink_parquet hangs? seems its due to collecting beforehand...
chemo.collect().write_parquet(f'{OUTPUT_DIR}/chemo_{DATE}.parquet')
pre_post_chemo.collect().write_parquet(f'{OUTPUT_DIR}/pre_post_chemo_{DATE}.parquet')

# Write radiation therapy dataset
rad.collect().write_parquet(f'{OUTPUT_DIR}/radiation_{DATE}.parquet')