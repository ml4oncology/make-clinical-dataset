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