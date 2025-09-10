"""
Script to combine all the data into one unified dataset
"""
import argparse
import os

import polars as pl
import yaml
from make_clinical_dataset.epic.combine import (
    combine_chemo_to_main_data,
    combine_demographic_to_main_data,
    combine_event_to_main_data,
    combine_radiation_to_main_data,
    merge_closest_measurements,
)
from make_clinical_dataset.epic.label import (
    get_acu_labels,
    get_CTCAE_labels,
    get_symptom_labels,
)
from make_clinical_dataset.epic.preprocess.demographic import get_demographic_data
from make_clinical_dataset.shared.constants import DEFAULT_CONFIG_PATH, ROOT_DIR

DATE = '2025-03-29'
DATA_DIR = f"{ROOT_DIR}/data/final/data_{DATE}"

def parse_args():
    parser = argparse.ArgumentParser()
    msg = """
    Specifies the events/dates features will be aligned on. Available options are:
    - "treatment-dates": Aligns features based on treatment dates.
    - A filepath to a parquet or csv with a datetime column: Aligns features based on the datetime values
    """
    parser.add_argument('--align-on', type=str, default='treatment-dates', help=msg)
    parser.add_argument('--date-column', type=str, default='treatment_date', help='Name of the datetime column in the main data')
    parser.add_argument('--output-file-prefix', type=str, default='treatment_centered', help='Name of the output file prefix')
    parser.add_argument('--output-dir', type=str, default=f"{DATA_DIR}/processed/")
    parser.add_argument('--data-dir', type=str, default=f"{DATA_DIR}/interim/")
    parser.add_argument('--config-path', type=str, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    align_on = args.align_on
    main_date_col = args.date_column
    output_file_prefix = args.output_file_prefix
    output_dir = args.output_dir
    data_dir = args.data_dir
    config_path = args.config_path

    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    chemo = pl.read_parquet(f'{data_dir}/chemo.parquet')
    rad = pl.read_parquet(f'{DATA_DIR}/interim/radiation.parquet')
    lab = pl.read_parquet(f'{data_dir}/lab.parquet')
    lab = lab.with_columns(pl.col('mrn').cast(pl.Int64))
    sym = pl.read_parquet(f'{data_dir}/symptom.parquet')
    acu = pl.read_parquet(f'{data_dir}/acute_care_use.parquet')
    demog = pl.from_pandas(get_demographic_data())
    with open(config_path) as file:
        cfg = yaml.safe_load(file)

    if align_on == 'treatment-dates':
        main_date_col = "assessment_date"
        main = (
            chemo
            .filter(pl.col('drug_type') == "direct")
            .select('mrn', 'treatment_date').unique()
            .rename({'treatment_date': main_date_col})
            .sort('mrn', main_date_col)
        )
    elif align_on.endswith('.parquet.gzip') or align_on.endswith('.parquet'):
        main = pl.read_parquet(align_on)
    elif align_on.endswith('.csv'):
        main = pl.read_csv(align_on, try_parse_dates=True)
    else:
        raise ValueError(f'Sorry, aligning features on {align_on} is not supported yet')
    
    # Extract features
    main = combine_demographic_to_main_data(main, demog, main_date_col)
    main = combine_chemo_to_main_data(main, chemo, main_date_col, time_window=(-28,0))
    main = combine_radiation_to_main_data(main, rad, main_date_col, time_window=(-28,0))
    main = merge_closest_measurements(main, lab, main_date_col, "obs_date", include_meas_date=True, time_window=(-5,0))
    main = merge_closest_measurements(main, sym, main_date_col, "obs_date", include_meas_date=True, time_window=(-30,0))
    main = combine_event_to_main_data(main, acu, main_date_col, "ED_visit", lookback_window=5)

    # Extract targets
    main = get_acu_labels(main, acu, lookahead_window=[30, 60, 90])
    main = get_CTCAE_labels(main.lazy(), lab.lazy(), main_date_col, lookahead_window=30).collect()
    main = get_symptom_labels(main, sym, main_date_col)
    
    date_cols = ['mrn'] + [col for col in main.columns if col.endswith('date')]
    str_cols = ['cancer_type', 'primary_site_desc', 'intent', 'drug_name', 'postal_code']
    feat_cols = ['mrn', main_date_col] + str_cols + [col for col in main.columns if col not in date_cols+str_cols]
    main_dates = main.select(date_cols)
    main_dates.write_parquet(f'{output_dir}/{output_file_prefix}_dates.parquet')
    main_data = main.select(feat_cols)
    main_data.write_parquet(f'{output_dir}/{output_file_prefix}_data.parquet')
    
if __name__ == '__main__':
    main()