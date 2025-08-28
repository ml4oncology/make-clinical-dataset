"""
Module to preprocess clinical notes and clinic visits data
"""
from typing import Optional

import pandas as pd


###############################################################################
# Clinical Notes
###############################################################################
def get_clinical_notes_data(
    data_dir: Optional[str] = None, 
    drop_duplicates: bool = True
) -> pd.DataFrame:
    if data_dir is None:
        data_dir = "./data/raw"

    df = pd.read_parquet(f'{data_dir}/all_notes.parquet.gzip')
    df = filter_notes_data(df)
    return df


def filter_notes_data(df: pd.DataFrame) -> pd.DataFrame:
    # cleaning up columns
    df = df.rename(
        columns={
            "Observations.ProcName": "proc_name",
            # date of visit
            "processed_date": "clinic_date",
            # date the note was uploaded/last updated to EPR
            "EPRDate": "last_updated_date",
            "MRN": "mrn",
        }
    )
    df["clinic_date"] = df["clinic_date"].dt.tz_localize(None)
    df["last_updated_date"] = df["last_updated_date"].dt.tz_localize(None)

    # only include relevant clinical notes
    proc_names = ["Clinic Note", "Clinic Note (Non-dictated)", "Letter", "History & Physical Note", "Consultation Note"]
    df = df[df["proc_name"].isin(proc_names)]

    # removing erroneous entries
    mask = df["clinic_date"] <= df["clinic_date"].quantile(0.0001)
    print(f'Removing {sum(mask)} visits that "occured before" {df[mask].clinic_date.max()}')
    df = df[~mask]
    df = df.sort_values(by=["mrn", "clinic_date"])
    return df


###############################################################################
# Clinic Visits
###############################################################################
def get_clinic_visits_during_treatment(clinic: pd.DataFrame, treatment: pd.DataFrame) -> pd.DataFrame:
    # combine clinic and treatment
    clinic = clinic[["mrn", "clinic_date", "last_updated_date"]]
    cols = ["treatment_date", "regimen", "line_of_therapy", "intent", "cycle_number", "height", "weight", "body_surface_area"]
    treatment = treatment[["mrn"] + cols]
    df = pd.merge(clinic, treatment, on="mrn", how="inner")
    df = df.rename(columns={col: f"next_{col}" for col in cols})

    # filter out clinic visits where the next treatment session does not occur within 5 days
    mask = df["next_treatment_date"].between(
        df["clinic_date"], df["clinic_date"] + pd.Timedelta(days=5)
    )
    df = df[mask]

    # filter out clinic visits where notes were uploaded/updated after the next treatment session
    mask = df["last_updated_date"] < df["next_treatment_date"]
    df = df[mask]

    # remove duplicates from the merging
    df = df.sort_values(by=["mrn", "next_treatment_date"])
    df = df.drop_duplicates(subset=["mrn", "clinic_date"], keep="first")

    return df


def backfill_treatment_info(df: pd.DataFrame):
    """Backfill missing treatment information

    Patient visit flow
        -> first clinic visit (book treatment plan)
        -> second clinic visit (check up)
        -> start treatment
    For the clinic visits right before starting a new treatment, we are missing treatment information
    But the treatment is pre-booked at that point
    So we can pull treatment information back

    NOTE: this procedure occurs after anchoring treatment information to clinic visits
    NOTE: it's not data leakage to backfill height, weight, and body surface area because patients will not suddenly
    grow / shrink within 5 days
    """
    # backfill treatment information if no prior treatments within 5 days
    no_trts_prior = df["treatment_date"].isnull()
    for col in ["regimen", "line_of_therapy", "intent", "cycle_number", "height", "weight", "body_surface_area"]:
        df.loc[no_trts_prior, col] = df.pop(f"next_{col}")
    df.loc[no_trts_prior, "days_since_starting_treatment"] = 0
    return df