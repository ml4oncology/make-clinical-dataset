"""
Module to preprocess emergency department visits and hospitalizations, aka acute care use (ACU).

We use 
1. (Pre-EPIC) ER triage assessment, (EPIC) ED provider notes, and (EPIC) ED arrival dates 
    to indicate emergency department visits
2. (Pre-EPIC + EPIC) Discharge summaries to indicate hospitalizations. 

NOTE: discharge summaries and ED provider notes come from the same clinical notes dataset
TODO: separate planned (e.g. elective surgeries) and unplanned hospitalizations based on discharge summary.
"""
import pandas as pd
import polars as pl
from ml_common.util import load_table


###############################################################################
# Acute Care Use
###############################################################################
def get_acute_care_use(
    triage: pl.DataFrame,
    discharge: pl.DataFrame, 
    provider_notes: pl.DataFrame,
    epic_arrival_dates: pd.DataFrame
) -> pl.DataFrame:
    """Combine acute care use from multiple sources.

    1. EPIC ED Arrival Dates
    2. ED Provider Notes
    3. Discharge Summaries
    4. ER Triage Assessments
    """
    discharge = (
        discharge
        .select('mrn', 'hosp_admission_date', 'hosp_discharge_date', 'clinical_notes', 'length_of_stay')
        .rename({'clinical_notes': 'note'})
        .with_columns(pl.lit("Discharge Summary").alias('data_source'))
    )
    triage = (
        triage
        .select('mrn', 'ED_arrival_date', 'CTAS Score', 'History/Assessment')
        .rename({'CTAS Score': 'CTAS_score', 'History/Assessment': 'note'})
        .with_columns(pl.col('ED_arrival_date').cast(pl.Date))
        .group_by(["mrn", "ED_arrival_date"])
        .agg([
            # if patients visited multiple times in a single day, 
            # take the first CTAS score but concatenate the assessment note
            pl.col('CTAS_score').first(),
            pl.col('note').str.join('\n-----------\n')
        ])
        .with_columns(pl.lit("ER Triage Assessment").alias('data_source'))
    )
    provider_notes = (
        provider_notes
        .select('mrn', 'ED_arrival_date', 'clinical_notes')
        .rename({'clinical_notes': 'note'})
        .with_columns(pl.lit("ED Provider Notes").alias('data_source'))
    )
    epic_arrival_dates = (
        pl.from_pandas(epic_arrival_dates)
        .with_columns([
            pl.col('ED_arrival_date').cast(pl.Date),
            pl.lit("EPIC ED Arrival Dates").alias('data_source')
        ])
    )
    emerg = (
        pl.concat([epic_arrival_dates, provider_notes, triage], how='diagonal')
        .group_by(["mrn", "ED_arrival_date"])
        .agg([
            pl.col("data_source").str.join(", "), 
            pl.all().exclude("data_source").drop_nulls().first()
        ])
        .sort('mrn', 'ED_arrival_date')
    )
    acute_care_use = pl.concat([emerg, discharge], how='diagonal')
    return acute_care_use


###############################################################################
# ED Arrival Date
###############################################################################
def get_epic_arrival_dates(filepath: str) -> pl.DataFrame:
    """Load, clean, filter, process EPIC ED arrival dates."""
    df = load_table(filepath)

    # rename the columns
    df = df.rename(columns={'PATIENT_ID': 'mrn', 'EMERGENCY_ADMISSION_DATE': 'ED_arrival_date'})

    # fix dtypes
    df["ED_arrival_date"] = pd.to_datetime(df["ED_arrival_date"])

    # keep only useful columns
    df = df[['mrn', 'ED_arrival_date']]

    # remove rows with missing dates
    df = df[df["ED_arrival_date"].notna()]

    # remove duplicates (multiple entries for the same arrival)
    # TODO: handle patients with multiple arrivals on the same day (occurs rarely)
    df = df.drop_duplicates()

    # sort by patient and date
    df = df.sort_values(by=['mrn', 'ED_arrival_date'])

    return df


###############################################################################
# Discharge Summary + ED Provider Note
###############################################################################
def get_acu_notes_data(filepath: str) -> dict[str, pl.DataFrame]:
    """Load, clean, filter, process discharge summaries and ED provider notes data.
    """
    # Please ask Wayne Uy about the merged_processed_cleaned_clinical_notes dataset
    df = pl.scan_parquet(filepath)
    df = clean_acu_notes_data(df)
    
    procs = [
        "Unscheduled Discharge Summary", # Pre-EPIC
        "Discharge Summary", # Pre-EPIC
        "Disch Summ", # EPIC
    ]
    discharge = filter_acu_notes_data(df, procs=procs)
    discharge = process_discharge_data(discharge)

    procs = [
        "ED Prov Note" # EPIC
    ]
    provider_notes = filter_acu_notes_data(df, procs=procs)
    provider_notes = process_provider_notes_data(provider_notes)

    return {
        "discharge": discharge.collect(), 
        "provider_notes": provider_notes.collect()
    }


def clean_acu_notes_data(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    df = df.rename({"Observations.ProcName": "proc_name"})
    return df


def filter_acu_notes_data(df: pl.DataFrame | pl.LazyFrame, procs: list[str],) -> pl.DataFrame | pl.LazyFrame:
    df = df.filter(pl.col('proc_name').is_in(procs))
    df = df.select(
        "mrn", "proc_name", "clinical_notes", "processed_physician_name",
        "processed_date", "epr_date" # only use these dates as references 
    )
    return df


def process_discharge_data(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    df = extract_admission_and_discharge_dates(df)
    df = df.with_columns(
        # fill missing discharge dates with the processed date
        pl.coalesce([
            pl.col("hosp_discharge_date"), 
            pl.col('processed_date').cast(pl.Date)
        ]).alias("hosp_discharge_date"),
        # get length of stay
        (pl.col('hosp_discharge_date') - pl.col('hosp_admission_date')).dt.total_days().alias('length_of_stay')
    )
    df = df.sort('mrn', 'hosp_admission_date')
    return df


def process_provider_notes_data(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    # extract ED visit date
    df = df.with_columns(
        pl.coalesce([
            pl.col("clinical_notes")
            .str.extract(r"Date of Visit:\s*(\d{2}/\d{2}/\d{4})", 1)
            .str.strptime(pl.Date, format="%d/%m/%Y"),
            pl.col('processed_date').cast(pl.Date)
        ]).alias("ED_arrival_date")
    )
    df = df.sort('mrn', 'ED_arrival_date')
    return df


def extract_admission_and_discharge_dates(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """Extract the hospital admission and discharge dates
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
    admit_pattern = rf"(?i)(?:date of[\s\S]* admission|date of visit|admission date|admit date|admitted since|admitted[\s\S]*? on)[:\s\n]*{date_pattern}"
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
        parse_date_expr("admission_date_raw").alias("hosp_admission_date"),
        parse_date_expr("discharge_date_raw").alias("hosp_discharge_date"),
    ])

    return df


###############################################################################
# ER Triage Assessment
###############################################################################
def get_triage_data(id_to_mrn: dict[str, int], data_dir: str) -> pl.DataFrame:
    """Load, clean, filter, process ER triage assessment data."""
    df = pl.scan_parquet(f'{data_dir}/*.parquet')

    # map the patient ID to mrns
    df = df.with_columns(
        pl.col("patient").replace_strict(id_to_mrn).alias("mrn")
    ).drop('patient')

    df = filter_triage_data(df)
    df = process_triage_data(df)
    return df


def filter_triage_data(df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
    procs = ["ER Triage Assessment"]
    df = df.filter(pl.col('proc_name').is_in(procs))
    df = df.filter(pl.col('obs_status').is_null() | (pl.col('obs_status') != 'unknown'))
    df = df.unique()
    return df


def process_triage_data(df: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    # combine the obervation values (numeric + unit, or use string value if available)
    # combine the two datetime columns
    obs_val = pl.col("obs_val_num").cast(str) + pl.lit(" ") + pl.col("obs_unit")
    drop_cols = [
        'obs_val_str', 'obs_val_num', 'obs_unit', 
        'effective_datetime', 'occurrence_datetime_from_order'
    ]
    df = df.with_columns(
        pl.when(pl.col("obs_val_str").is_not_null())
          .then(pl.col("obs_val_str"))
          .otherwise(obs_val)
          .alias("obs_val"),
        pl.coalesce([
            pl.col("effective_datetime"), 
            pl.col("occurrence_datetime_from_order")
        ]).alias("ED_arrival_date"),
    ).drop(drop_cols)

    # transform each obs_name into its own column
    if isinstance(df, pl.LazyFrame):
        df = df.collect() # need to collect before pivoting
    df = df.pivot(
        on='obs_name', 
        values='obs_val', 
        aggregate_function=pl.element().str.join("\n\n")
    )

    # convert CTAS description into a numeric score
    df = convert_CTAS_score(df)

    # drop columns that are >99% missing or have only one unique value
    drop_cols = [
        col for col in df.columns 
        if (df[col].is_null().mean() > 0.99) or 
           (df[col].drop_nulls().n_unique() == 1)
    ]
    df = df.drop(drop_cols)

    # if multiple entries for same assessment, take the "final" entry
    df = df.sort(
        pl.when(pl.col('obs_status') == 'final').then(0).otherwise(1)
    ).group_by('mrn', 'ED_arrival_date').first()

    # reorder the columns
    cols = [
        'mrn', 'ED_arrival_date', 'CEDIS Complaint', 'CTAS Score', 
        'Chief Complaint', 'History/Assessment', 'Smoker?'
    ]
    df = df.select(cols + [pl.exclude(cols)])

    # sort by patient and date
    df = df.sort('mrn', 'ED_arrival_date')

    return df


def convert_CTAS_score(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """Convert the CTAS (Canadian Triage and Acuity Scale) score from descriptive to quantitative.

    CTAS score should be between 1 - 5
    1: severely ill, requires resuscitation 
    2: requires emergent care and rapid medical intervention 
    3: requires urgent care 
    4: requires less-urgent care 
    5: requires non-urgent care
    """
    score_map = {
        'resuscitation': 1, 
        'emergent': 2, 
        'urgent': 3, 
        'less urgent': 4, 
        'non urgent': 5
    }
    df = df.with_columns(pl.col('CTAS Score').replace_strict(score_map))
    return df