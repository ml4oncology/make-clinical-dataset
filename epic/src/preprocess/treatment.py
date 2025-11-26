import pandas as pd
import polars as pl
from make_clinical_dataset.epic.util import get_excluded_numbers, load_table
from make_clinical_dataset.shared import logger
from make_clinical_dataset.shared.constants import TRT_INTENT


###############################################################################
# Radiation therapy
###############################################################################
def get_radiation_data(
    filepath: str,
    id_to_mrn: dict[str, int],
) -> pd.DataFrame:
    """Load, clean, filter, process radiation therapy data."""
    df = pd.read_parquet(filepath)

    # map the patient ID to mrns
    df['mrn'] = df.pop('patient_id').map(id_to_mrn)

    # clean up intent
    df['intent'] = df['intent'].str.lower()

    # reorder the columns
    cols = [
        'mrn', 'treatment_start_date', 'treatment_end_date', 'intent', 
        'given_dose', 'fractions_given', 'dose_prescribed', 'fractions_prescribed', 
        'diagnosis_icd_code', 'diagnosis_desc', 'diagnosis_category', 
        'morphology', 'site_treated', 'technique', 
    ]
    df = df[cols]

    return df


###############################################################################
# EPIC Chemotherapy
###############################################################################
def get_epic_chemo_data(
    filepath: str,
    drug_map: pl.DataFrame, 
    verbose: bool = False
) -> pl.DataFrame:
    """Load, clean, filter, process EPIC chemotherapy data.
    
    NOTE: There's too many differences between retrospective code (in ml4oncology/make-clinical-dataset) 
    and deployment code (in ml4oncology/model-deployer) requirements, best to keep them separate for now.
    """
    df = load_table(filepath)

    # map the normalized drug names
    df = df.join(drug_map, on="drug_name", how="left")

    df = clean_epic_chemo_data(df)
    df = filter_epic_chemo_data(df, verbose=verbose)
    df = process_epic_chemo_data(df, verbose=verbose)
    return df


def clean_epic_chemo_data(df: pl.DataFrame) -> pl.DataFrame:
    body_meas_cols = ["height", "weight", "body_surface_area"]

    route_map = {
        "intravenous": "IV",
        "subcutaneous": "SC",
        "oral": "PO",
        "intraperitoneal": "IP",
        "intramuscular": "IM",
        "intravenous push": "IVP",
        "intralesional": "IL",
        "sublingual": "SL",
        "intratumoral": "IT", # Intrathecal?
    }

    # clean up features
    df = df.with_columns([
        # abbreviate the administration routes
        pl.col('route').replace(route_map),
        # clean up intent of treatment
        pl.col('intent').str.to_lowercase(),
        # take the avg patient body measurements for each date
        *[pl.col(col).mean().over(['mrn', 'treatment_date']) for col in body_meas_cols]
    ])

    # forward fill patient body measurements
    df = df.sort(by=['mrn', 'treatment_date']) # need to sort first
    df = df.with_columns([pl.col(col).forward_fill().over("mrn") for col in body_meas_cols])

    return df


def filter_epic_chemo_data(df: pl.DataFrame, verbose: bool = False) -> pl.DataFrame:
    # drop rows with incomplete treatment status
    mask = pl.col('day_status') == "Completed"
    if verbose:
        get_excluded_numbers(df, mask=~mask, context=" with incomplete treatment status")
    df = df.filter(mask)

    # drop rows without treatment date
    mask = pl.col('treatment_date').is_not_null()
    if verbose:
        get_excluded_numbers(df, mask=~mask, context=" without a treatment date")
    df = df.filter(mask)

    # drop rows without mapped drug name
    mask = pl.col('drug_name_normalized').is_not_null()
    if verbose:
        get_excluded_numbers(df, mask=~mask, context=" without a mapped drug name")
    df = df.filter(mask)

    # drop rows with given dosage values of 0 (i.e. 0 mg, 0 mL, 0 mEq, etc) or null
    mask = pl.col('given_dose').str.starts_with('0 ') | pl.col('given_dose').is_null()
    if verbose:
        get_excluded_numbers(df, mask=mask, context=" with 0 or null given dosage values")
    # even though many rows are discarded, the number of unique (mrn, treatment_date) is not significantly affected
    # i.e these are duplicate / erroneous / unnecessary entries
    before = df.unique(subset=['mrn', 'treatment_date']).height
    after = df.filter(~mask).unique(subset=['mrn', 'treatment_date']).height
    assert (before - after) / before < 0.01
    df = df.filter(~mask)

    return df


def process_epic_chemo_data(df: pl.DataFrame, verbose: bool = False) -> pl.DataFrame:
    # process dosage data - split into value and unit
    for col in ["given_dose", "dose_ordered", "regimen_dose"]:
        df = df.with_columns(
            pl.col(col).str.split(" ").alias("split")
        ).with_columns([
            pl.col("split").list.get(0).cast(pl.Float64).alias(col),
            pl.col("split").list.get(1).alias(f"{col}_unit"),
        ]).drop("split")

    # reorder select columns
    cols = [
        'mrn', 'treatment_date', 'first_treatment_date', 'cycle_number', 
        'drug_name_normalized', 'given_dose', 'given_dose_unit', 
        'body_surface_area', 'height', 'weight', 
        'intent', 'regimen', 'department',
        'drug_name', 'drug_type', 'drug_dose', 'drug_unit',
        'dose_ordered', 'dose_ordered_unit', 'regimen_dose', 'regimen_dose_unit', 
        'route', 'data_source'
        # 'scheduled_treatment_date', 'day_status', 'cycle_status', 'mar_action', 'discontinue_reason', 'cancel_day_reason'
    ]
    df = df.select(cols)

    # process full and partial duplicates
    df = process_duplicates(df, verbose=verbose)

    return df

###############################################################################
# Pre-EPIC Chemotherapy
###############################################################################
def get_chemo_data(
    filepath: str,
    id_to_mrn: dict[str, int], 
    drug_map: pl.DataFrame, 
    verbose: bool = False
) -> pl.DataFrame:
    """Load, clean, filter, process chemotherapy data."""
    df = pl.read_parquet(filepath)

    # map the patient ID to mrns
    df = df.with_columns(
        pl.col("patient_id").replace_strict(id_to_mrn).alias("mrn")
    ).drop('patient_id')

    # map the normalized drug names
    df = df.join(drug_map, on="drug_name", how="left")

    df = clean_chemo_data(df)
    df = filter_chemo_data(df, verbose=verbose)
    df = process_chemo_data(df, verbose=verbose)
    return df


def clean_chemo_data(df: pl.DataFrame) -> pl.DataFrame:
    body_meas_cols = ["height", "weight", "body_surface_area"]

    # clean up features
    df = df.with_columns([
        # clean up the Cancer Care Ontario regimen entries
        pl.col('cco_regimen').str.slice(1, None),
        # clean up intent of treatment
        pl.col('intent').replace(TRT_INTENT).str.to_lowercase(),
        # take the avg patient body measurements for each date
        *[pl.col(col).mean().over(['mrn', 'treatment_date']) for col in body_meas_cols],
        # fix dtypes
        pl.col('cycle_number').cast(pl.Int64),
    ])

    # forward fill patient body measurements
    df = df.sort(by=['mrn', 'treatment_date']) # need to sort first
    df = df.with_columns([pl.col(col).forward_fill().over("mrn") for col in body_meas_cols])

    return df


def filter_chemo_data(df: pl.DataFrame, verbose: bool = False) -> pl.DataFrame:
    # drop rows without treatment date
    mask = pl.col('treatment_date').is_not_null()
    if verbose:
        get_excluded_numbers(df, mask=~mask, context=" without a treatment date")
    df = df.filter(mask)

    # drop rows without mapped drug name
    mask = pl.col('drug_name_normalized').is_not_null()
    if verbose:
        get_excluded_numbers(df, mask=~mask, context=" without a mapped drug name")
    df = df.filter(mask)

    return df


def process_chemo_data(df: pl.DataFrame, verbose: bool = False) -> pl.DataFrame:
    # process given dosage information
    df = process_given_dosage(df, verbose=verbose)

    # fill missing route entries
    route_map = {
        "IV": " IV ",
        "PO": "CAPSULE|TABLET",
        "SC": "SUBCUTANEOUS",
        "IM": "INTRAMUSCULAR",
        "IP": "INTRAPERITONEAL",
    }
    route_is_missing = pl.col("route").is_null()
    for route, pattern in route_map.items():
        contain_pattern = pl.col("drug_name").str.contains(pattern)
        df = df.with_columns(
            pl.when(route_is_missing & contain_pattern)
            .then(pl.lit(route))
            .otherwise(pl.col("route"))
            .alias("route")
        )

    # reorder select columns
    cols = [
        'mrn', 'treatment_date', 'first_treatment_date', 'cycle_number', 
        'drug_name_normalized', 'given_dose', 'given_dose_unit', 
        'body_surface_area', 'height', 'weight', 
        'intent', 'cco_regimen', 'regimen', 'department',
        'drug_name', 'drug_type', 'drug_dose', 'drug_unit',
        'dose_ordered', 'route', 'data_source'
        # 'drug_id', 'fdb_drug_code', 'treatment_category'
    ]
    df = df.select(cols)

    # process full and partial duplicates
    df = process_duplicates(df, verbose=verbose)

    return df


def process_given_dosage(df: pl.DataFrame, verbose: bool = False) -> pl.DataFrame:
    """Process the given dosage information, split them into value and unit"""
    # Clean the dosing feature
    custom_dose_map = {
        "125 mg (1)- 80 mg (2)": "125 mg",
        "4 gram/5 gram": "4.5 gram"
    }
    for old, new in custom_dose_map.items():
        df = df.with_columns(
            pl.when(pl.col("given_dose") == old)
              .then(pl.lit(new))
              .otherwise(pl.col("given_dose"))
        )

    df = df.with_columns(
        pl.col("given_dose")
        # Remove surrounding parentheses
        .str.strip_chars("()")
        # Remove commas in numbers (e.g., 1,000 → 1000)
        .str.replace_all(r',000', '000', literal=False)
        # Insert space before % ONLY IF not already spaced (e.g., "0.9%" → "0.9 %")
        .str.replace_all(r'(\d)%', r'$1 %', literal=False)
        # Remove space in denominator (e.g., "300 mcg/0.5 mL" → "300 mcg/0.5mL")
        .str.replace_all(r'(/[\d.]+) (\w+)', r'$1$2', literal=False)
    )
    
    # Discard rows where given_dose does not match the following pattern. Only makes up ~0.35% of the dataset.
    # (i.e. first dilution, placebo, dosing for two drugs at once, etc)
    pattern1 = r'^\d+(?:\.\d+)?\s[\w/.\-%²μ]+$' # regex pattern for "<num> <unit>", where unit can be mg, mg/mL, etc
    pattern2 = r'nan\s[\w/.\-%²μ]+$' # regex pattern for "nan <unit>", where unit can be mg, mg/mL, etc
    df = df.filter(
        pl.col("given_dose").str.contains(pattern1, literal=False) |
        pl.col("given_dose").str.contains(pattern2, literal=False) |
        pl.col("given_dose").is_null() # keep rows with missing dosages for now
    )
    
    # Split into value and unit
    df = df.with_columns(
        pl.col("given_dose").str.split(" ").alias("split")
    ).with_columns([
        pl.col("split").list.get(0).alias("given_dose"),
        pl.col("split").list.get(1).alias("given_dose_unit"),
    ]).drop("split")
    
    # Fill "nan" values with dose_ordered
    df = df.with_columns(
        pl.when(pl.col("given_dose") == "nan")
          .then(pl.col("dose_ordered"))
          .otherwise(pl.col("given_dose"))
          .alias("given_dose")
    )

    # Convert to float
    df = df.with_columns(pl.col("given_dose").cast(pl.Float64))

    # Drop rows with value of 0 (i.e. 0 mg, 0 mL, 0 mEq, etc)
    mask = pl.col('given_dose') == 0
    if verbose:
        get_excluded_numbers(df, mask=mask, context=" with given dosage value of 0")
    df = df.filter(~mask)

    return df


###############################################################################
# Helpers
###############################################################################
def process_duplicates(df: pl.DataFrame, verbose: bool = False) -> pl.DataFrame:
    """Merge partial duplicate rows and aggregate their non-duplicate info"""
    if verbose:
        prev_size = df.shape[0]
        col_names = df.columns
    else:
        col_names = df.collect_schema().names()

    # remove any full duplicates
    # NOTE: maintain_order=True is not efficient, better to sort again at the end
    df = df.unique()

    # collapse rows where everything matches except given_dose and dose_ordered
    # sum up the dosages
    cols = [col for col in col_names if col not in ['given_dose', 'dose_ordered']]
    assert df["given_dose"].is_not_null().all()
    df = df.group_by(cols).agg([
        pl.col("given_dose").sum(),
        pl.col("dose_ordered").sum()
    ]).select(col_names)
    if verbose:
        count = prev_size - df.shape[0]
        logger.info('Merged partial duplicates with different dosage fields for '
                    f'{count} ({(count)/(prev_size)*100:0.3f}%) rows')

    # sort the data
    df = df.sort(by=['mrn', 'treatment_date'])

    return df


# NOTE: not used anymore, can be deprecated
def collapse_antiemetic_regimens(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse various AE regimens into a single "AE" regimen
    e.g. AE-APREP80;DEX2BID+OLA => AE
    
    AE regimens are antiemetic protocols that are paired with chemo regimens to prevent chemo-induced nausea/vomiting.
    All drugs under AE are supportive.

    Common abbreviations, courtesy of ChatGPT
    | Abbrev          | Drug name        | Drug class                              |
    | --------------- | ---------------- | --------------------------------------- |
    | D / DEX         | Dexamethasone    | Corticosteroid                          |
    | APREP           | Aprepitant       | NK1 receptor antagonist                 |
    | OLA             | Olanzapine       | Atypical antipsychotic (antiemetic use) |
    | PRO             | Prochlorperazine | Dopamine antagonist                     |
    | G / GRA         | Granisetron      | 5-HT3 receptor antagonist               |
    | PALO            | Palonosetron     | 5-HT3 receptor antagonist               |
    | METO            | Metoclopramide   | Dopamine antagonist                     |
    | OND             | Ondansetron      | 5-HT3 receptor antagonist               |
    """
    mask = df["department"] == "AE"
    df.loc[mask, "regimen"] = "AE"
    return df