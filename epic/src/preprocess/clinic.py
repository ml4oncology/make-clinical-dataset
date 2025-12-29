"""
Module to preprocess clinic visit data
"""
import hashlib

import polars as pl


def get_clinic_data(filepath: str) -> pl.LazyFrame:
    """Load, clean, filter, process clinic visit data."""
    df = pl.scan_parquet(filepath)

    # rename the columns
    df = df.rename({
        "Observations.ProcName": "proc_name", 
        "clinical_notes": "note", 
        "EPIC_FLAG": "epic_flag", 
        "processed_date": "clinic_date",
        "processed_physician_name": "physician_name",
    })

    # ensure correct data type
    df = df.with_columns(pl.col('clinic_date').cast(pl.Datetime))

    # select relevant columns
    df = df.select("mrn", "proc_name", "clinic_date", "note", "physician_name", "epic_flag")

    # only keep relevant clinic notes
    df = df.filter(pl.col('proc_name').is_in([
        # Pre-EPIC
        "Clinic Note",
        "Clinic Note (Non-dictated)", 
        "Consultation Note",
        "Letter",
        
        # EPIC
        "PROGRESS" 
    ]))

    # if multiple notes on the same day, keep the longest one
    # as it has the best chance of having all relevant information
    # TODO: explore other strategies, like concatenation or merging+deduplication
    df = (
        df
        .with_columns(pl.col('note').str.len_chars().alias('note_length'))
        .sort('note_length', descending=True)
        .unique(subset=['mrn', 'clinic_date'])
    )

    # create unique id for each note based on the content of the note
    df = df.with_columns(
        pl.col("note")
        .map_elements(_hash_note, return_dtype=pl.String)
        .alias('note_id')
    )

    df = df.sort('mrn', 'clinic_date')
    return df


def _hash_note(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()