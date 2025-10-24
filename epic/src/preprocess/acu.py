"""
Module to preprocess emergency department visits and hospitalizations (aka ACU - acute care use)

NOTE: we currently use discharge summaries to indicate acute care use, 
which may include palliative care and surgical appointments as well
"""
import polars as pl


def get_acu_data(filepath: str) -> pl.DataFrame | pl.LazyFrame:
    """Load, clean, filter, process acute care use data."""
    # Please ask Wayne Uy about the merged_processed_cleaned_clinical_notes dataset
    df = pl.read_parquet(filepath)
    # df = pl.scan_parquet(filepath)
    df = clean_acu_data(df)
    df = filter_acu_data(df)
    df = process_acu_data(df)
    return df


def clean_acu_data(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    df = df.rename({"Observations.ProcName": "proc_name"})
    return df


def filter_acu_data(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    procs = ["Unscheduled Discharge Summary", "ED Prov Note", "Disch Summ", "Discharge Summary"]
    df = df.filter(pl.col('proc_name').is_in(procs))
    df = df.select(
        "mrn", "clinical_notes", "processed_physician_name",
        "processed_date", "epr_date" # only use these dates as references 
    )
    return df


def process_acu_data(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
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
