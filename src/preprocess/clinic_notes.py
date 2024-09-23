"""
Module to preprocess clinical notes data
"""
from typing import Optional

import pandas as pd

from .. import ROOT_DIR

def get_clinic_notes_data(
    data_dir: Optional[str] = None, 
    drop_duplicates: bool = True
) -> pd.DataFrame:
    if data_dir is None:
        data_dir = f"{ROOT_DIR}/data/raw"

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
            # date the note was uploaded to EPR (meaning the note could have been revised)
            "EPRDate": "epr_date",
            "MRN": "mrn",
        }
    )
    df["clinic_date"] = df["clinic_date"].dt.tz_localize(None)
    df["epr_date"] = df["epr_date"].dt.tz_localize(None)

    # only include relevant clinical df
    proc_names = ["Clinic Note", "Clinic Note (Non-dictated)", "Letter", "History & Physical Note", "Consultation Note"]
    df = df[df["proc_name"].isin(proc_names)]

    # removing erroneous entries
    mask = df["clinic_date"] <= df["clinic_date"].quantile(0.0001)
    print(f'Removing {sum(mask)} visits that "occured before" {df[mask].clinic_date.max()}')
    df = df[~mask]
    df = df.sort_values(by=["mrn", "clinic_date"])
    return df