"""
Module to anchor/combine features or targets to the main dates
"""
import datetime

import polars as pl


###############################################################################
# General Combiners
###############################################################################
def merge_closest_measurements(
    main: pl.DataFrame | pl.LazyFrame, 
    meas: pl.DataFrame | pl.LazyFrame, 
    main_date_col: str,
    meas_date_col: str, 
    direction: str = 'backward',
    time_window: tuple[int, int] = (-5,0),
    merge_individually: bool = True,
    include_meas_date: bool = False
) -> pl.DataFrame | pl.LazyFrame:
    """Extract the closest measurements (lab tests, symptom scores, etc) prior to / after the main date 
    within a lookback / lookahead window and combine them to the main dataset

    Both main and meas should have mrn and date columns
    
    Args:
        main_date_col: The column name of the main visit date
        meas_date_col: The column name of the measurement date
        time_window: The start and end of the window in terms of number of days after(+)/before(-) the main visit dates
        direction: specifies whether to merge measurements before or after the main date. Either 'backward' or 'forward'
        merge_individually: If True, merges each measurement column separately
        include_meas_date: If True, include the date of the closest measurement that was merged
    """
    lower_limit, upper_limit = time_window
    if direction == 'backward':
        main_date = pl.col(main_date_col) + pl.duration(days=upper_limit)
    elif direction == 'forward':
        main_date = pl.col(main_date_col) + pl.duration(days=lower_limit)
    main = main.with_columns(main_date.alias("main_date"))

    # ensure date types match
    meas = meas.with_columns(pl.col(meas_date_col).cast(main.schema["main_date"]))

    merge_kwargs = dict(
        left_on='main_date', right_on=meas_date_col, by='mrn', strategy=direction,
        tolerance=datetime.timedelta(days=upper_limit - lower_limit), check_sortedness=False
    )
    
    if merge_individually:
        # merge each measurement column individually
        for col in meas.columns:
            if col in ["mrn", meas_date_col]: continue

            data_to_merge = meas.filter(pl.col(col).is_not_null()).select(["mrn", meas_date_col, col])

            # merges the closest row to main date while matching on mrn
            main = main.join_asof(data_to_merge, **merge_kwargs)

            if include_meas_date:
                main = main.rename({meas_date_col: f"{col}_{meas_date_col}"})
            else:
                main = main.drop(meas_date_col)

    else:
        main = main.join_asof(meas, **merge_kwargs)

    main = main.drop("main_date")
    return main


###############################################################################
# Specific Combiners
###############################################################################
def combine_chemo_to_main_data(
    main: pl.DataFrame | pl.LazyFrame, 
    chemo: pl.DataFrame | pl.LazyFrame, 
    main_date_col: str, 
    time_window: tuple[int, int] = (-28,0),
) -> pl.DataFrame | pl.LazyFrame:
    """Combine chemo treatment information to the main data"""
    # Further preprocess the chemo
    chemo = chemo.filter(pl.col('drug_type') == "direct")
    chemo = chemo.select(
        'mrn', 'treatment_date', 'drug_name', 'first_treatment_date', 'intent', 
        'cycle_number', 'body_surface_area', 'height', 'weight'
    )
    # one-hot encode drugs - TODO: retrive the dosages instead of binary 0/1 for each drug
    drugs = chemo['drug_name'].unique()
    chemo = chemo.with_columns(pl.col('drug_name').alias('drug')) # keep the orignal string column
    chemo = chemo.to_dummies(columns='drug')
    # merge same-day rows
    chemo = chemo.group_by('mrn', 'treatment_date').agg(
        pl.col('body_surface_area').mean(),
        pl.col('height').mean(),
        pl.col('weight').mean(),
        # if two treatments (the old regimen and new regimen) overlap on same day, use data associated with the most recent regimen 
        # NOTE: examples found thru df.group_by('mrn', 'treatment_date').agg(pl.col('first_treatment_date').n_unique() > 1)
        pl.col('cycle_number').last(),
        pl.col('first_treatment_date').last(),
        pl.col('intent').last(),
        # combine dosages together
        *(pl.col(f'drug_{col}').max() for col in drugs),
        # concat the drugs together
        pl.col("drug_name").str.join("\n")
    )
    # NOTE: group_by's maintain_order=True is not efficient, better to sort it again right after
    chemo = chemo.sort('mrn', 'treatment_date')
    # add line of treatment
    new_regimen = pl.col('first_treatment_date') != pl.col('first_treatment_date').shift()
    palliative_intent = pl.col('intent') == 'palliative'
    chemo = chemo.with_columns((new_regimen & palliative_intent).cum_sum().alias('line_of_treatment'))

    # Merge them together
    main = merge_closest_measurements(
        main, chemo, main_date_col=main_date_col, meas_date_col="treatment_date", merge_individually=False,
        time_window=time_window
    )

    # Create days since starting treatment column
    days_since_start = (pl.col(main_date_col) - pl.col('first_treatment_date')).dt.total_days()
    main = main.with_columns(days_since_start.alias('days_since_starting_treatment'))

    # Create days since last treatment column
    mask = main.select((pl.col(main_date_col) == pl.col('treatment_date')).alias('is_same'))
    same_date = mask['is_same'].all() if isinstance(main, pl.DataFrame) else mask.collect()['is_same'].all()
    if same_date: 
        days_since_last = pl.col("treatment_date").diff().dt.total_days()
        main = main.with_columns(days_since_last.over("mrn").alias('days_since_last_treatment'))
    else:
        days_since_last = (pl.col(main_date_col) - pl.col('treatment_date')).dt.total_days()
        main = main.with_columns(days_since_last.alias('days_since_last_treatment'))

    return main


def combine_radiation_to_main_data(
    main: pl.DataFrame | pl.LazyFrame, 
    rad: pl.DataFrame | pl.LazyFrame, 
    main_date_col: str, 
    time_window: tuple[int, int] = (-28,0),
) -> pl.DataFrame | pl.LazyFrame:
    rad = (
        rad
        .select('mrn', 'treatment_start_date', 'dose_given')
        .rename({'treatment_start_date': 'radiation_date', 'dose_given': 'radiation_dose_given'})
    )
    main = merge_closest_measurements(
        main, rad, main_date_col=main_date_col, meas_date_col="radiation_date", merge_individually=False, 
        time_window=time_window
    )
    return main


def combine_demographic_to_main_data(
    main: pl.DataFrame | pl.LazyFrame, 
    demog: pl.DataFrame | pl.LazyFrame, 
    main_date_col: str, 
) -> pl.DataFrame | pl.LazyFrame:
    main = merge_closest_measurements(
        main, demog, main_date_col=main_date_col, meas_date_col="diagnosis_date", 
        merge_individually=False, time_window=[-1e8, 0]
    )

    # exclude patients with missing birth date
    main = main.filter(pl.col("birth_date").is_not_null())

    # create age column
    age = (pl.col("assessment_date") - pl.col("birth_date")).dt.total_days() / 365.25
    main = main.with_columns(age.alias('age'))

    # exclude patients under 18 years of age
    main = main.filter(pl.col('age') >= 18)

    return main


def combine_event_to_main_data(
    main: pl.DataFrame | pl.LazyFrame, 
    event: pl.DataFrame | pl.LazyFrame, 
    main_date_col: str, 
    event_name: str,
    lookback_window: int = 5, # years
) -> pl.DataFrame | pl.LazyFrame:
    # Additional features to be added
    # 1. date of most recent event prior to main date
    prev_date_col = f'prev_{event_name}_date'
    # 2. days since most recent event
    days_since_col = f'days_since_prev_{event_name}'
    # 3. number of events within the lookback window 
    num_events_col = f'num_prior_{event_name}s_within_{lookback_window}_years'
    # TODO: add length of stay

    # Extract the event features
    # main = main.lazy()
    # event = event.lazy()
    event_feats = (
        main
        .join(event, on="mrn", how="left") # WARNING: beware of exploding joins, use lazy evaluation when necessary
        .filter(
            (pl.col("admission_date") < pl.col(main_date_col)) &
            (pl.col("admission_date") >= (pl.col(main_date_col) - pl.duration(days=365 * lookback_window)))
        )
        .group_by(["mrn", main_date_col])
        .agg(
            pl.len().alias(num_events_col),
            pl.col("admission_date").max().alias(prev_date_col)
        )
        .with_columns(
            (pl.col(main_date_col) - pl.col(prev_date_col)).dt.total_days().alias(days_since_col)
        )
    )

    # Merge the event features to main
    main = (   
        main
        .join(event_feats, on=["mrn", main_date_col], how="left")
        .with_columns(
            pl.col(num_events_col).fill_null(0),
            pl.col(days_since_col).fill_null(365 * lookback_window)
        )
    )

    return main