"""
Module to extract labels
"""
import polars as pl
from make_clinical_dataset.epic.combine import merge_closest_measurements


###############################################################################
# Acute Care Use
###############################################################################
def get_acu_labels(
    main: pl.DataFrame | pl.LazyFrame, 
    acu: pl.DataFrame | pl.LazyFrame, 
    lookahead_window: int | list[int] = 30
) -> pl.DataFrame | pl.LazyFrame:
    if isinstance(lookahead_window, int):
        lookahead_window = [lookahead_window]

    # Further preprocess the acu data
    acu = acu.group_by('mrn', 'admission_date').agg(pl.col('clinical_notes').str.join("\n\n"))
    acu = acu.rename({'admission_date': 'target_ED_date', "clinical_notes": "target_ED_note"})

    main = merge_closest_measurements(
        main, acu, main_date_col="assessment_date", meas_date_col="target_ED_date", 
        merge_individually=False, direction='forward', time_window=(0, max(lookahead_window))
    )

    for days in lookahead_window:
        mask = pl.col('target_ED_date').fill_null(strategy="max") < pl.col('assessment_date') + pl.duration(days=days)
        main = main.with_columns(mask.alias(f'target_ED_{days}d'))

    return main