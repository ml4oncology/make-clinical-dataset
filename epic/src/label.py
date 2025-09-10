"""
Module to extract labels
"""
import polars as pl
from make_clinical_dataset.epic.combine import merge_closest_measurements
from make_clinical_dataset.shared.constants import (
    CTCAE_CONSTANTS,
    MAP_CTCAE_LAB,
    SYMP_COLS,
)


###############################################################################
# Acute Care Use
###############################################################################
def get_acu_labels(
    main: pl.DataFrame | pl.LazyFrame, 
    acu: pl.DataFrame | pl.LazyFrame, 
    main_date_col: str, 
    lookahead_window: int | list[int] = 30
) -> pl.DataFrame | pl.LazyFrame:
    if isinstance(lookahead_window, int):
        lookahead_window = [lookahead_window]

    # Further preprocess the acu data
    acu = acu.group_by('mrn', 'admission_date').agg(pl.col('clinical_notes').str.join("\n\n"))
    acu = acu.rename({'admission_date': 'target_ED_date', "clinical_notes": "target_ED_note"})

    main = merge_closest_measurements(
        main, acu, main_date_col=main_date_col, meas_date_col="target_ED_date", 
        merge_individually=False, direction='forward', time_window=(0, max(lookahead_window))
    )

    for days in lookahead_window:
        mask = pl.col('target_ED_date').fill_null(strategy="max") < pl.col(main_date_col) + pl.duration(days=days)
        main = main.with_columns(mask.alias(f'target_ED_{days}d'))

    return main


###############################################################################
# Abnormal Lab Findings / CTCAE (Common Terminology Criteria for Adverse Events)
###############################################################################
def get_CTCAE_labels(
    main: pl.DataFrame | pl.LazyFrame, 
    lab: pl.DataFrame | pl.LazyFrame, 
    main_date_col: str, 
    lookahead_window: int = 30 # days
) -> pl.DataFrame | pl.LazyFrame:
    """Compute lookahead lab values and apply the threshold functions to generate CTCAE grade labels
    """
    # Extract the CTCAE targets
    # main = main.lazy()
    # lab = lab.lazy()
    ctcae_targs = (
        main
        .join(lab, on="mrn", how="left", suffix="_target") # WARNING: beware of exploding joins, lazy evaluation is imperative
        .filter(
            (pl.col("obs_date") > pl.col(main_date_col)) &
            (pl.col("obs_date") <= (pl.col(main_date_col) + pl.duration(days=lookahead_window)))
        )
        .group_by(["mrn", main_date_col])
        .agg(
            pl.col('hemoglobin_target').min().alias('target_hemoglobin_min'),
            pl.col('platelet_target').min().alias('target_platelet_min'),
            pl.col('neutrophil_target').min().alias('target_neutrophil_min'),
            pl.col('creatinine_target').max().alias('target_creatinine_max'),
            pl.col('alanine_aminotransferase_target').max().alias('target_alanine_aminotransferase_max'),
            pl.col('aspartate_aminotransferase_target').max().alias('target_aspartate_aminotransferase_max'),
            pl.col('total_bilirubin_target').max().alias('target_total_bilirubin_max'),
        )
    )

    # Merge the CTCAE targets to main
    main = main.join(ctcae_targs, on=["mrn", main_date_col], how="left")

    # Apply threshold functions for each grade
    exps = []
    for ctcae, constants in CTCAE_CONSTANTS.items():
        lab_col = MAP_CTCAE_LAB[ctcae]
        for grade in [2, 3]:
            target_col = f'target_{ctcae}_grade{grade}plus'
            threshold = constants[f'grade{grade}plus']

            if ctcae in ['hemoglobin', 'neutrophil', 'platelet']:
                lab_lookahead_col = f'target_{lab_col}_min'
                exp = (
                    pl.when(pl.col(lab_lookahead_col).is_null())
                    .then(-1)
                    .when(pl.col(lab_lookahead_col) < threshold)
                    .then(1)
                    .otherwise(0)
                    .alias(target_col)
                )
            else:
                lab_lookahead_col = f'target_{lab_col}_max'
                key = 'upper_bound' if ctcae == 'AKI' else 'lower_bound'
                lab_base_val = pl.col(lab_col).clip(**{key: constants['ULN']}).fill_null(constants['ULN'])
                exp = (
                    pl.when(pl.col(lab_lookahead_col).is_null())
                    .then(-1)
                    .when(pl.col(lab_lookahead_col) > threshold * lab_base_val)
                    .then(1)
                    .otherwise(0)
                    .alias(target_col)
                )
            exps.append(exp)
    main = main.with_columns(exps)

    return main


###############################################################################
# Symptom Deterioration
###############################################################################
def get_symptom_labels(
    main: pl.DataFrame | pl.LazyFrame, 
    symp: pl.DataFrame | pl.LazyFrame, 
    main_date_col: str, 
    lookahead_window: int = 30, # days
    scoring_map: dict[str, int] | None = None
) -> pl.DataFrame | pl.LazyFrame:
    """Extract labels for symptom deterioration

    Label is positive if symptom deteriorates (score increases) by X points within the lookahead window
    """
    if scoring_map is None:
        scoring_map = {col: 3 for col in SYMP_COLS}
        del scoring_map['ecog']

    # Extract the symptom deterioration targets
    symp_targs = (
        main
        .join(symp, on="mrn", how="left", suffix="_target")
        .filter(
            (pl.col("obs_date") > pl.col(main_date_col)) &
            (pl.col("obs_date") <= (pl.col(main_date_col) + pl.duration(days=lookahead_window)))
        )
        .group_by(["mrn", main_date_col])
        .agg([pl.col(f'{key}_target').max().alias(f'target_{key}_max') for key in scoring_map])
    )
    
    # Merge the symptom deterioration targets to main
    main = main.join(symp_targs, on=["mrn", main_date_col], how="left")

    # Compute binary labels: 1 (positive), 0 (negative), or -1 (missing/exclude)
    exps = []
    for symp, pt in scoring_map.items():
        targ_col = f"target_{symp}_{pt}pt_change"
        change = pl.col(f"target_{symp}_max") - pl.col(symp)
        exp = (
            pl.when(change.is_null())
            .then(-1)
            # If baseline score is alrady high, exclude
            .when(pl.col(symp) > 10 - pt)
            .then(-1)
            .when(change >= pt)
            .then(1)
            .otherwise(0)
            .alias(targ_col)
        )
        exps.append(exp)
    main = main.with_columns(exps)

    return main