"""
Module to preprocess radiology report data, which includes CT Scans, X-rays, Ultrasounds, etc
"""
from typing import Optional

from datetime import datetime
import pandas as pd
import polars as pl

from make_clinical_dataset import logger
from make_clinical_dataset.constants import OBS_MAP

def get_radiology_data(
    mrn_map: dict[str, str], 
    data_dir: str | None = None,
    verbose: bool = False
) -> pl.DataFrame:
    """Load, clean, filter, process radiology observation data."""
    if data_dir is None:
        data_dir = './data/raw/radiology'
    # NOTE: df["COL"] only works in eager mode, use pl.col("COL") in lazy mode
    rad = pl.read_parquet(f'{data_dir}/*.parquet').lazy()
    rad = clean_radiology_data(rad, mrn_map=mrn_map)
    rad = filter_radiology_data(rad, verbose=verbose)
    rad = process_radiology_data(rad)
    return rad.collect()


def clean_radiology_data(df: pl.LazyFrame, mrn_map: dict[str, int]) -> pl.LazyFrame:
    """Clean and rename column names and entries. 
    
    Merge same columns together.
    """
    # the datetime is captured in two different columns, combine them together
    main_date_col, secondary_date_col = "effective_datetime", "occurrence_datetime_from_order"
    df = df.with_columns([
        pl.coalesce([pl.col(main_date_col), pl.col(secondary_date_col)]).alias("epr_datetime")
    ])
    # df = df.with_columns(pl.col("epr_datetime").dt.date().alias("epr_date"))

    # map the patient ID to mrns
    df = df.with_columns(
        pl.col("patient").replace(mrn_map)
    ).rename({'patient': 'mrn'})
    
    return df


def filter_radiology_data(df: pl.LazyFrame, verbose: bool = False) -> pl.LazyFrame:
    """Filter out reports based on various conditions"""

    # TODO: keep metadata rows (i.e. obs_name = Date Dictated, Verified By, Read By, etc)
    # only keep rows that contains report text
    df = df.filter(
        pl.col('obs_val_str').str.starts_with('\nREPORT')
    )
    
    # keep only useful columns
    df = df.select('mrn', 'epr_datetime', 'proc_name', 'obs_val_str') # 'obs_name'
    
    return df


END_TEXT = (
    "FOR PHYSICIAN USE ONLY - Please note that this report was generated using voice recognition software. "
    "The Joint Department of Medical Imaging (JDMI) supports a culture of continuous improvement. "
    "If you require further clarification or feel there has been an error in this report "
    "please contact the JDMI Call Centre from 8am - 8pm Monday to Friday, "
    "8am - 4pm Saturday to Sunday (excludes statutory holidays) at 416-946-2809.\n"
)


def process_radiology_data(df: pl.LazyFrame):
    """Process the reports."""
    # Remove useless info for faster inference
    df = df.with_columns(
        pl.col("obs_val_str").str.replace(END_TEXT, "").alias("processed_text")
    )
    # Replace the following for convenience
    df = df.with_columns(
        pl.col("processed_text")
        .str.replace("\nREPORT  (VERIFIED ", "\nREPORT (FINAL ", literal=True)
        .str.replace("\nREPORT  (TRANSCRIBED ", "\nREPORT (FINAL ", literal=True)
    )

    # ensure all reports start with "REPORT (FINAL YYYY/MM/DD)"
    mask = pl.col("processed_text").str.starts_with("\nREPORT (FINAL ") 
    assert df.select(mask).collect().to_series().all()
    # ensure there is only one line with "REPORT (FINAL YYYY/MM/DD)" in each report
    mask = df.select(pl.col('processed_text').str.count_matches('\nREPORT \(FINAL')).collect() == 1
    assert df.select(mask).collect().to_series().all()

    # Extract the date written in the report
    start_len, date_len = len("\nREPORT (FINAL "), len("YYYY/MM/DD")
    df = df.with_columns(
        pl.col("processed_text")
        .str.slice(start_len, date_len)
        .str.strptime(pl.Date, format="%Y/%m/%d", strict=False)
        .alias("initial_report_date")
    )
    # the report may include addendums that were added at later date
    # to prevent data leakage, extract the final date in which an addendum was added
    def extract_last_addendum_date(text: str) -> datetime | None:
        start_idx = text.rfind("ADDENDUM (FINAL ")
        if start_idx == -1:
            return None
        start_idx += len("ADDENDUM (FINAL ")
        end_idx = start_idx + date_len
        date_str = text[start_idx:end_idx]
        return datetime.strptime(date_str, "%Y/%m/%d").date()
    df = df.with_columns(
        pl.col("processed_text")
        .map_elements(extract_last_addendum_date, return_dtype=pl.Date)
        .alias("last_addendum_date")
    )

    # Set up the final date
    df = df.with_columns(
        pl.when(pl.col("initial_report_date") < pl.col("last_addendum_date"))
        .then(pl.col("last_addendum_date"))
        .otherwise(pl.col("initial_report_date"))
        .alias("date")
    )

    # Drop duplicate reports for same patient
    # some duplicate reports have different proc names, merge them
    proc_names = df.group_by("mrn", "processed_text").agg(
        pl.col("proc_name").unique().sort().map_elements(', '.join, return_dtype=pl.String)
    )
    df = (
        df
        .drop('proc_name')
        .join(proc_names, on=["mrn", "processed_text"], how="left")
        .sort(["mrn", "date"])
        .unique(subset=["mrn", "processed_text"], keep="first")
    )
    
    return df
