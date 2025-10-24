"""
Module to preprocess laboratory test data, which includes hematology and biochemistry data
"""
import polars as pl
from make_clinical_dataset.epic.util import get_excluded_numbers


def get_lab_data(
    id_to_mrn: dict[str, str], 
    lab_map: dict[str, str], 
    data_dir: str | None = None,
    verbose: bool = False
) -> pl.DataFrame:
    """Load, clean, filter, process lab observation data."""
    if data_dir is None:
        data_dir = './data/raw/lab'
    # NOTE: df["COL"] only works in eager mode, use pl.col("COL") in lazy mode
    lab = pl.scan_parquet(f'{data_dir}/*.parquet')
    lab = clean_lab_data(lab, id_to_mrn=id_to_mrn, lab_map=lab_map)
    lab = filter_lab_data(lab, verbose=verbose)
    lab = process_lab_data(lab)
    return lab


def clean_lab_data(df: pl.LazyFrame, id_to_mrn: dict[str, int], lab_map: dict[str, str]) -> pl.LazyFrame:
    """Clean and rename column names and entries. 
    
    Merge same columns together.
    """
    # the datetime is captured in two different columns, combine them together
    main_date_col, secondary_date_col = "effective_datetime", "occurrence_datetime_from_order"
    df = df.with_columns([
        pl.coalesce([pl.col(main_date_col), pl.col(secondary_date_col)]).alias("obs_datetime")
    ])

    # map the patient ID to mrns
    df = df.with_columns(
        pl.col("patient").replace(id_to_mrn).cast(pl.Int64)
    ).rename({'patient': 'mrn'})

    # rename the observations
    df = df.with_columns(
        pl.col("obs_name").alias("orig_obs_name"),
        pl.col("obs_name").replace(lab_map)
    )
    
    # rename the units
    df = df.with_columns(
        pl.col("obs_unit").replace({"bil/L": "x10e9/L", "fl": "fL"})
    )
    
    return df


def filter_lab_data(df: pl.LazyFrame, verbose: bool = False) -> pl.LazyFrame:
    """Filter out observations based on various conditions (missingness, differing units, etc)."""
    # exclude rows without a date (only makes up 0.005% of the data)
    if verbose:
        get_excluded_numbers(df, mask=pl.col("obs_datetime").is_null(), context=" without a date")
    df = df.filter(pl.col('obs_datetime').is_not_null())
    
    # exclude observations that were not included in the mapping
    if verbose:
        context = " whose observations were not included in the mapping"
        get_excluded_numbers(df, mask=pl.col("obs_name") == "nan", context=context)
    df = df.filter(pl.col("obs_name") != "nan")
    
    # exclude rows where numerical observation values are missing 
    # (they will have string entries, but only makes up <1% of the data)
    if verbose:
        get_excluded_numbers(df, mask=pl.col("obs_val_num").is_null(), context=" without numerical observation values")
    df = df.filter(pl.col("obs_val_num").is_not_null())

    # exclude rows with different unit measurements
    # TODO: basophil is really funky, investigate further
    df = filter_units(df, verbose=verbose)
    
    # keep only useful columns
    df = df.select('mrn', 'obs_val_num', 'obs_name', 'obs_unit', 'obs_datetime')
    
    return df


def filter_units(df: pl.LazyFrame, verbose: bool = False) -> pl.LazyFrame:
    """Filter out observations with inconsistent units.

    Some observations have measurements in different units.
    (e.g. neutrophil observations contain measurements in x10e9/L (the majority) and % (the minority))
    Only keep one measurement unit for simplicity.
    """
    exclude_unit_map = {
        'creatinine': ['mmol/d', 'mmol/CP', 'mmol/cp'],
        'eosinophil': ['%'], 
        'lymphocyte': ['%'], 
        'monocyte': ['%'], 
        'neutrophil': ['%'],
        'red_blood_cell': ['x10e6/L'],
        'white_blood_cell': ['x10e6/L'],
    }
    mask = pl.lit(False)
    for obs_name, exclude_units in exclude_unit_map.items():
        condition = (pl.col("obs_name") == obs_name) & pl.col("obs_unit").is_in(exclude_units)
        mask |= condition
        
    if verbose:
        get_excluded_numbers(df, mask=mask, context=" with inconsistent units")
        
    df = df.filter(~mask)
    
    return df


def process_lab_data(df: pl.LazyFrame) -> pl.DataFrame:
    """Sort and pivot the observation data."""
    df = df.with_columns(
        pl.col("obs_datetime").dt.truncate('1d').alias("obs_date")
    )

    # save the units for each observation name
    unit_map = dict(
        df.group_by("obs_name", "obs_unit").len().sort("len")
        .select("obs_name", "obs_unit").collect().to_numpy()
    )
    # TODO: save unit map in feature store for later use
    print(unit_map)
    
    # take the most recent value if multiple lab tests taken in the same day
    df = df.sort("obs_datetime")
    # NOTE: group_by's maintain_order=True is not efficient, better to sort it again right after
    df = df.group_by("mrn", "obs_date", "obs_name").agg(pl.col("obs_val_num").last())
    df = df.sort("mrn", "obs_date")

    # make each observation name into a new column
    # (only works in eager mode...need to collect beforehand)
    df = df.collect().pivot(
        on="obs_name",
        index=["mrn", "obs_date"],
        values="obs_val_num",
    )
    
    return df
