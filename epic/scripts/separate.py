"""Separate the Observation data into ED, ESAS, radiology, and lab datasets"""
import os

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from tqdm import tqdm

from make_clinical_dataset.shared.constants import INFO_DIR, ROOT_DIR

# Configurations
today = '2025-03-29'
date = '2025-01-08'
data_dir = f'{ROOT_DIR}/data/raw/data_pull_{date}/observation_parquet'

# Initialize a Spark session
# Recommended to request the following resources on SLURM:
# srun -p himem -c 8 --mem 64GB -t 0-08:00:00 --pty bash
spark = SparkSession.builder.appName("EPIC-data").config("spark.driver.memory", "50G").getOrCreate()

# Read all Parquet files in the folder
# make sure its all parquet.gzip files in this folder (no one created some weird files in there)
assert all([fname.endswith('parquet.gzip') for fname in os.listdir(data_dir)])
df = spark.read.parquet(data_dir)

# Clean up Columns
# replace dots with underscores in column names (pyspark dont like dots)
# alternatively, you can wrap the column name with "`col.name`", but I don't like dots
rename_expr = [f'`{col}` as {col.replace(".", "_")}' for col in df.columns]
df = df.selectExpr(*rename_expr)

# rename the column names to a shortened form
col_map = {
    "PATIENT_RESEARCH_ID": 'patient',
    "Observations_ProcCode": 'proc_code',
    "Observations_ProcName": 'proc_name',
    'Observations_Observation__id': 'id',
    'Observations_Observation_status': 'obs_status',
    "Observations_Observation_component_valueString": 'obs_val_str',
    'Observations_Observation_component_valueQuantity_value': 'obs_val_num',
    'Observations_Observation_component_valueQuantity_unit': 'obs_unit',
    'Observations_Observation_component_code_coding_0_display': 'obs_display',
    'Observations_Observation_component_code_text': 'obs_text',

    'Observations_Observation_effectiveDateTime': 'effective_datetime',
    'Observations_OccurrenceDateTimeFromOrder': 'occurrence_datetime_from_order',

    # either duplicate or unnecessary info
    # 'Observations_Observation_attr_lastModified_0_at': 'last_updated_datetime',
    # 'Observations_Observation_attr_procCode': 'another_proc_code',
    # 'Observations_Observation_attr_procName': 'another_proc_name'
}
rename_expr = [f"{old} as {new}" for old, new in col_map.items()]
df = df.select(list(col_map)).selectExpr(*rename_expr)

# Merge columns
# merge obs_display and obs_text - fill missing values in obs_display with values from obs_text
# assert df.filter(df['obs_display'].isNotNull() & df['obs_text'].isNotNull()).count() == 0
df = df.withColumn("obs_name", F.coalesce(F.col("obs_display"), F.col("obs_text")))
df = df.drop("obs_display", "obs_text")

# Clean up text
# remove leading and trailing whitespaces
df = df.withColumn("proc_name", F.trim('proc_name'))
df = df.withColumn("obs_name", F.trim('obs_name'))
df = df.withColumn("obs_val_str", F.trim('obs_val_str'))

# convert 'nan' or 'None' to NULL
df = df.na.replace(["nan", "None"], None, subset=["obs_val_num"])
# convert '.' to NULL
df = df.na.replace(["."], None, subset=["obs_val_str"])

# Clean up dtype
# cast to double
# prev_null_count = df.filter(df['obs_val_num'].isNull()).count()
df = df.withColumn("obs_val_num", F.col("obs_val_num").cast("double"))
# assert df.filter(df['obs_val_num'].isNull()).count() == prev_null_count, "Casting to double caused some rows to be NULL. Please double check those rows"

# cast to timestamp
def to_timestamp(df: DataFrame, col: str):
    return df.withColumn(col, F.to_timestamp(F.col(col), "yyyy-MM-dd'T'HH:mm:ssXXX"))
df = to_timestamp(df, "occurrence_datetime_from_order")
df = to_timestamp(df, "effective_datetime")

# Separate the data
proc_codes = pd.read_csv(f'{INFO_DIR}/proc_codes.csv')
proc_names = pd.read_csv(f'{INFO_DIR}/proc_names.csv')
for category in tqdm(['lab', 'HW', 'ED', 'ESAS', 'radiology']):
    # partition the data with respect to the corresponding category
    codes = proc_codes.query('category == @category')['value'].tolist()
    names = proc_names.query('category == @category')['value'].tolist()
    mask = df['proc_name'].isin(names) | df['proc_code'].isin(codes)
    if category == 'lab': 
        mask |= df['proc_code'].startswith('LAB')
    partition = df.filter(mask)

    if category == 'lab': 
        # remove the following unimportant info - contains 45M rows of Dept #CODE DATE, Spec #CODE DATE, Req #CODE occasional comments
        mask = ~partition['obs_name'].isin(['Tech Comment', 'Technologist Comment', 'Collection Info'])
        partition = partition.filter(mask)

    # remove rows with missing observations
    partition = partition.filter(partition['obs_val_num'].isNotNull() | partition['obs_val_str'].isNotNull())
    
    # print(f'{category} dataset size: Num Records: {partition.count()}. Num Patients: {partition.select("patient").distinct().count()}')

    # write to disk
    output_path = f'{ROOT_DIR}/data/processed/{category}/{category}_{today}'
    partition = partition.coalesce(20) # reduce the number of spark partitions to 20
    # partition.write.option("parquet.block.size", 256 * 1024 * 1024).parquet(output_path)
    partition.write.mode("overwrite").parquet(output_path)

# Stop the Spark session
spark.stop()