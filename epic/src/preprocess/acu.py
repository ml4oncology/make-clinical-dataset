"""
Module to preprocess emergency department visits and hospitalizations, aka acute care use (ACU)

NOTE: We currently use discharge summaries (from clinical notes data) to indicate acute care use, 
which may include palliative care and surgical appointments as well

NOTE: Why are we not using ED data from separate.py? Because it is mostly triage assessment data,
and we do not know if patients were admitted after triage assessment.
"""
import pandas as pd
import polars as pl
from ml_common.util import load_table


###############################################################################
# Admission Dates
###############################################################################
def get_epic_admission_dates(filepath: str) -> pl.DataFrame:
    """Load, clean, filter, process EPIC ED admission dates."""
    df = load_table(filepath)

    # rename the columns
    df = df.rename(columns={'PATIENT_ID': 'mrn', 'EMERGENCY_ADMISSION_DATE': 'admission_date'})

    # fix dtypes
    df["admission_date"] = pd.to_datetime(df["admission_date"])

    # keep only useful columns
    df = df[['mrn', 'admission_date']]

    # remove rows with missing dates
    df = df[df["admission_date"].notna()]

    # remove duplicates (multiple entries for the same admission)
    # TODO: handle patients with multiple admissions on the same day (occurs rarely)
    df = df.drop_duplicates()

    # sort by patient and date
    df = df.sort_values(by=['mrn', 'admission_date'])

    return df


def get_admission_dates(discharge: pl.DataFrame, epic_admission_dates: pd.DataFrame) -> pl.DataFrame:
    """Combine ED admission dates from both sources (EPIC ED admission dates and from discharge summaries).
    
    Args:
        discharge: The output of get_discharge_data().
        epic_admission_dates: The output of get_epic_admission_dates().
    """
    discharge_summary_admission_dates = (
        discharge
        .select('mrn', 'admission_date')
        .filter(pl.col('admission_date').is_not_null())
        .with_columns(pl.lit("Discharge Summary").alias('data_source'))
    )
    epic_admission_dates = (
        pl.from_pandas(epic_admission_dates)
        .with_columns([
            pl.col('admission_date').cast(pl.Date),
            pl.lit("EPIC ED Admission Dates").alias('data_source'),
        ])
    )
    admission_dates = (
        pl.concat([epic_admission_dates, discharge_summary_admission_dates])
        .group_by(["mrn", "admission_date"])
        .agg(pl.col("data_source").unique().sort())
        .sort('mrn', 'admission_date')
    )
    return admission_dates


###############################################################################
# Discharge Summary
###############################################################################
def get_discharge_data(filepath: str) -> pl.DataFrame | pl.LazyFrame:
    """Load, clean, filter, process discharge data."""
    # Please ask Wayne Uy about the merged_processed_cleaned_clinical_notes dataset
    df = pl.read_parquet(filepath)
    # df = pl.scan_parquet(filepath)
    df = clean_discharge_data(df)
    df = filter_discharge_data(df)
    df = process_discharge_data(df)
    return df


def clean_discharge_data(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    df = df.rename({"Observations.ProcName": "proc_name"})
    return df


def filter_discharge_data(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    procs = ["Unscheduled Discharge Summary", "ED Prov Note", "Disch Summ", "Discharge Summary"]
    df = df.filter(pl.col('proc_name').is_in(procs))
    df = df.select(
        "mrn", "clinical_notes", "processed_physician_name",
        "processed_date", "epr_date" # only use these dates as references 
    )
    return df


def process_discharge_data(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    df = extract_admission_and_discharge_dates(df)
    df = df.sort('mrn', 'admission_date')
    return df


def extract_admission_and_discharge_dates(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """Extract the admission and discharge dates from the discharge summary notes
    """
    # extract "date-like" token after admission/discharge
    date_patterns = [
        r"[0-9]{1,2}(?:st|nd|rd|th)?[-/\s,]*[A-Za-z]{3,9}[-/\s,]*[0-9]{2,4}", # e.g. 13-Jan-2016 or 13th Jan, 2016
        r"[A-Za-z]{3,9}[-/\s]*?[0-9]{1,2}(?:st|nd|rd|th)?[-/\s,]*?[0-9]{4}", # e.g. January 13th, 2016
        r"[0-9]{1,4}[-/][0-9]{1,2}[-/][0-9]{2,4}" # e.g 2016-01-13 or 2016/01/13
    ]
    date_pattern = f"({'|'.join(date_patterns)})"
    # ( ... ) - capture group
    # (?: ... ) - non-capture group
    # (?i) - case insensitive
    # [:\s\n]* - zero or more colon or white spaces or new lines
    # [\s\S]*? - zero or more white space and non-white space
    admit_pattern = rf"(?i)(?:date of[\s\S]* admission|date of visit|admission date|admit date|admitted[\s\S]*? on)[:\s\n]*{date_pattern}"
    discharge_pattern = rf"(?i)(?:date of[\s\S]* discharge|discharge date|discharged[\s\S]*? on)[:\s\n]*{date_pattern}"
    df = df.with_columns(
        pl.col("clinical_notes").str.extract(admit_pattern, 1).alias("admission_date_raw"),
        pl.col("clinical_notes").str.extract(discharge_pattern, 1).alias("discharge_date_raw"),
    )

    # parse the dates
    formats = [
        "%d%b%y",        # 15Jul22
        "%d%b%Y",        # 15Jul2022
        "%d/%m/%y",      # 15/07/22
        "%d/%m/%Y",      # 15/07/2022
        "%Y-%m-%d",      # 2022-07-15
        "%d-%m-%Y",      # 15-07-2022
        "%d-%b-%Y",      # 15-Jul-2022
        "%d-%B-%Y",      # 15-July-2022
        "%d-%b %Y",      # 15-Jul 2022
        "%d%B-%Y",       # 15July-2022
        "%B-%d-%Y",      # July-15-2022
        "%B %d-%Y",      # July 15-2022
        "%B-%d %Y",      # July-15 2022
        "%d/%b/%Y",      # 15/Jul/2022
        "%d/%B/%Y",      # 15/July/2022
        "%d%b/%Y",       # 15Jul/2022
        "%d %b/%Y",      # 15 Jul/2022
        "%d %B/%Y",      # 15 July/2022
        "%m/%d/%Y",      # 07/15/2022
        "%Y/%m/%d",      # 2022/07/15
        "%B/%d/%Y",      # July/15/2022
        "%b %d/%Y",      # Jul 15/2022
        "%B %d/%Y",      # July 15/2022
        "%B %d, %Y",     # July 15, 2022
        "%B %d , %Y",    # July 15 , 2022
        "%d %B, %Y",     # 15 July, 2022
        "%B %d %Y",      # July 15 2022
        "%d %B %Y",      # 15 July 2022
    ]
    def parse_date_expr(colname):
        # edge case with september
        col = pl.col(colname).str.replace("September", "Sep").str.replace("Sept", "Sep")
        # edge case with the suffixes
        col = col.str.replace_all(r"(\d+)(st|nd|rd|th)", r"$1")
        # try parsing with all formats, then coalesce into one Date column
        parsed = [col.str.strptime(pl.Date, fmt, strict=False) for fmt in formats]
        return pl.coalesce(parsed)
    
    df = df.with_columns([
        parse_date_expr("admission_date_raw").alias("admission_date"),
        parse_date_expr("discharge_date_raw").alias("discharge_date"),
    ])

    return df

###############################################################################
# Triage Assessment
###############################################################################
def get_triage_data(data_dir: str) -> pl.DataFrame:
    """Load, clean, filter, process triage assessment data."""
    df = pl.scan_parquet(f'{data_dir}/*.parquet')
    df = df.unique()
    obs_val = pl.col("obs_val_num").cast(str) + pl.lit(" ") + pl.col("obs_unit")
    df = df.with_columns(
        pl.when(pl.col("obs_val_str").is_not_null()).then(pl.col("obs_val_str")).otherwise(obs_val).alias("obs_val"),
        pl.coalesce([pl.col("effective_datetime"), pl.col("occurrence_datetime_from_order")]).alias("datetime"),
    ).drop(['obs_val_str', 'obs_val_num', 'obs_unit', 'effective_datetime', 'occurrence_datetime_from_order'])
    df = df.collect()
    df = df.pivot(on='obs_name', values='obs_val', aggregate_function=pl.element().str.join("/n/n"))
    df = df.sort('patient', 'datetime')
    return df