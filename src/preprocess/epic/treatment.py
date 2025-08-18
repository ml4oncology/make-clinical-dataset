import pandas as pd
import polars as pl
from make_clinical_dataset.constants import TRT_INTENT
from make_clinical_dataset.util import get_excluded_numbers


def clean_treatment_data(
    df: pl.LazyFrame, 
    id_to_mrn: dict[str, int], 
    drug_map: pl.LazyFrame
) -> pl.LazyFrame:
    #TODO: move mapping out of clean
    # map the patient ID to mrns
    df = df.with_columns(
        pl.col("patient_id").replace_strict(id_to_mrn).alias("mrn")
    ).drop('patient_id')
    #TODO: move mapping out of clean
    # map the normalized drug names
    df = df.rename({"drug_name": "orig_drug_name"})
    df = df.join(drug_map, on="orig_drug_name", how="left")

    # clean up features
    df = df.with_columns([
        # clean up the Cancer Care Ontario regimen entries
        pl.col('cco_regimen').str.slice(1, None),
        # clean up intent of treatment
        pl.col('intent').replace(TRT_INTENT).str.to_lowercase()
    ])
    return df


def filter_treatment_data(df: pl.LazyFrame, verbose: bool = False) -> pl.LazyFrame:
    # drop duplicates
    df = df.unique()

    # drop rows without mapped drug name
    mask = pl.col('drug_name').is_not_null()
    if verbose:
        get_excluded_numbers(df, mask=~mask, context=" without a mapped drug name")
    df = df.filter(mask)

    # drop purely supportive regimens 
    drop_regimens = ["CARBO DESENSITIZATION"]
    mask = pl.col('regimen').is_in(drop_regimens)
    if verbose:
        get_excluded_numbers(df, mask=mask, context=" of purely supportive regimens")
    df = df.filter(~mask)

    # drop rows without body_surface_area
    mask = pl.col('body_surface_area').is_not_null()
    if verbose:
        get_excluded_numbers(df, mask=~mask, context=" with missing body_surface_area")
    df = df.filter(mask)

    return df


def process_treatment_data(df: pl.LazyFrame) -> pl.LazyFrame:
    # process given dosage information
    df = process_given_dosage(df)

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
        contain_pattern = pl.col("orig_drug_name").str.contains(pattern)
        df = df.with_columns(
            pl.when(route_is_missing & contain_pattern)
            .then(pl.lit(route))
            .otherwise(pl.col("route"))
            .alias("route")
        )

    # reorder select columns
    cols = [
        'mrn', 'treatment_date',
        'drug_name', 'orig_drug_name', 'drug_type', 'drug_dose', 'drug_unit',
        'given_dose', 'given_dose_unit', 'dose_ordered', 'route',
        'drug_id', 'fdb_drug_code', 'uhn_drug_code',
        'cco_regimen', 'regimen', 'department', 'intent',
        'body_surface_area', 'height', 'weight',
        'first_treatment_date', 'cycle_number', 'data_source', 
    ]
    df = df.select(cols)

    # sort the data
    df = df.sort(by=['mrn', 'treatment_date', 'drug_name'])

    return df


###############################################################################
# Helpers
###############################################################################
def process_given_dosage(df: pl.LazyFrame) -> pl.LazyFrame:
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
              .alias("given_dose")
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
    df = df.with_columns(
        pl.col("given_dose").cast(pl.Float64)
    )
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