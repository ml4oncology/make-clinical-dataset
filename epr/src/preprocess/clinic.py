"""
Module to preprocess clinical notes and clinic visits data
"""
import pandas as pd


###############################################################################
# Clinical Notes
###############################################################################
def get_clinical_notes_data(
    data_dir: str | None = None, 
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
def get_clinic_visits_during_treatment(clinic: pd.DataFrame, treatment: pd.DataFrame, lookahead: int) -> pd.DataFrame:
    # keep only rows at the first treatment_date for each (mrn, regimen, cycle_number) --
    # a regimen/cycle can span multiple treatment dates, and multiple rows can share the same
    # (mrn, regimen, cycle_number, treatment_date) if drug doses are split across rows, so we
    # keep all tied rows here and let the later drug_sums groupby collapse them correctly
    cycle_start_date = treatment.groupby(['mrn', 'regimen', 'cycle_number'])['treatment_date'].transform('min')
    treatment = treatment[treatment['treatment_date'] == cycle_start_date]

    # combine clinic and treatment
    clinic = clinic[["mrn", "clinic_date", "last_updated_date"]]
    cols = ["treatment_date", "regimen", "line_of_therapy", "intent", "cycle_number", "height", "weight", "body_surface_area"]
    drug_cols = treatment.columns[treatment.columns.str.startswith('drug_')].tolist()
    treatment = treatment[["mrn", "first_treatment_date"] + cols + drug_cols]
    df = pd.merge(clinic, treatment, on="mrn", how="inner")
    df = df.rename(columns={col: f"next_{col}" for col in cols})

    # filter out clinic visits where the next treatment session does not occur within 5 days
    mask = df["next_treatment_date"].between(
        df["clinic_date"], df["clinic_date"] + pd.Timedelta(days=lookahead)
    )
    df = df[mask]

    # filter out clinic visits where notes were uploaded/updated after the next treatment session
    mask = df["last_updated_date"] < df["next_treatment_date"]
    df = df[mask]

    # sum drug doses within the same upcoming session (mrn + next_treatment_date), so doses
    # from a *different* upcoming session for the same clinic visit don't get blended in
    # NOTE: it seems that this is not necessary since for a given mrn, clinic date, and next treatment date, there
    # is only 1 row
    drug_sums = df.groupby(["mrn", "clinic_date", "next_treatment_date"])[drug_cols].sum()

    # remove duplicates from the merging
    # For every treatment date, we only want to keep most recent clinic date
    df = df.sort_values(by=["mrn", "clinic_date", "next_treatment_date"])
    df = df.drop_duplicates(subset=["mrn", "next_treatment_date"], keep="last")
    # there are some edge cases a clinic date is mapped to multiple treatment dates
    # because there might be a change in regimen
    # in live deployment, we won't know this a priori so we keep the first clinic date
    df = df.drop_duplicates(subset=["mrn", "clinic_date"], keep="first")
    df = df.drop(columns=drug_cols)

    # attach the summed drug doses
    df = df.join(drug_sums, on=["mrn", "clinic_date", "next_treatment_date"])

    # rename back to unprefixed names to stay compatible with downstream code;
    # next_treatment_date is the one exception that keeps its prefix
    rename_cols = [c for c in cols if c != "treatment_date"]
    df = df.rename(columns={f"next_{c}": c for c in rename_cols})
    df['treatment_date'] = df['next_treatment_date']

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


def fill_body_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing height, weight, and body_surface_area values per patient.

    height: forward/backward filled per mrn (assumed roughly stable over time)
    weight, body_surface_area: filled using the nearest available value (same mrn) within
        30 days of next_treatment_date -- looking backward first, then forward for anything
        still missing
    """
    # NOTE: not much of an issue now but may want to think about using formula
    # for body_surface_area instead of backfilling
    df = df.sort_values(['mrn', 'next_treatment_date'])

    # height: simple ffill then bfill per patient
    df['height'] = df.groupby('mrn')['height'].transform(lambda s: s.ffill().bfill())

    for col in ['weight', 'body_surface_area']:
        lookup = (
            df.loc[df[col].notnull(), ['mrn', 'next_treatment_date', col]]
            .sort_values('next_treatment_date')
        )
        target = (
            df[['mrn', 'next_treatment_date']]
            .sort_values('next_treatment_date')
            .reset_index()
            .rename(columns={'index': '_orig_idx'})
        )

        back = pd.merge_asof(
            target, lookup, on='next_treatment_date', by='mrn',
            direction='backward', tolerance=pd.Timedelta(days=30)
        )
        fwd = pd.merge_asof(
            target, lookup, on='next_treatment_date', by='mrn',
            direction='forward', tolerance=pd.Timedelta(days=30)
        )

        filled = back[col].fillna(fwd[col])
        filled.index = target['_orig_idx'].values  # restore original row alignment

        df[col] = df[col].fillna(filled)

    return df