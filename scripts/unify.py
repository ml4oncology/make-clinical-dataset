"""
Script to combine all the data into one unified dataset
"""
from itertools import product
import argparse
import os

import pandas as pd
import yaml

from make_clinical_dataset.combine import (
    add_engineered_features,
    combine_demographic_to_main_data, 
    combine_event_to_main_data,
    combine_perc_dose_to_main_data,
    combine_treatment_to_main_data
)
from make_clinical_dataset.label import get_CTCAE_labels, get_death_labels, get_ED_labels, get_symptom_labels
from make_clinical_dataset.util import load_included_drugs

from ml_common.anchor import merge_closest_measurements

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--align-on', 
        type=str, 
        default='treatment-dates',
        help=('Specifies the events/dates features will be aligned on. Available options are:\n'
              '- "treatment-dates": Aligns features based on treatment dates.\n'
              '- "weekly-mondays": Aligns features based on the dates of every Monday.\n'
              '- A filepath to a parquet object or CSV table with a datetime column: Aligns features based on the '
              'datetime values')
    )
    parser.add_argument(
        '--date-column', 
        type=str, 
        default='treatment_date', 
        help='Name of the datetime column in the main data'
    )
    parser.add_argument(
        '--output-filename', 
        type=str, 
        default='treatment_centered_clinical_dataset', 
        help='Name of the output file, do not include file extension'
    )
    parser.add_argument('--output-dir', type=str, default='./data/processed')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--config-path', type=str, default='./config.yaml')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    align_on = args.align_on
    main_date_col = args.date_column
    output_filename = args.output_filename
    output_dir = args.output_dir
    data_dir = args.data_dir
    config_path = args.config_path

    if align_on != "treatment-dates":
        bad_cols = ["survey_date", "obs_date", "event_date", "treatment_date"]
        if main_date_col in bad_cols:
            raise ValueError(
                f"If not --align-on treatment-dates, the --date-column can not be any of {bad_cols}"
            )

    if not os.path.exists(output_dir): os.makedirs(output_dir)
    lab = pd.read_parquet(f'{data_dir}/interim/lab.parquet')
    trt = pd.read_parquet(f'{data_dir}/interim/treatment.parquet')
    dmg = pd.read_parquet(f'{data_dir}/interim/demographic.parquet')
    sym = pd.read_parquet(f'{data_dir}/interim/symptom.parquet')
    erv = pd.read_parquet(f'{data_dir}/interim/emergency_room_visit.parquet')
    last_seen = pd.read_parquet(f'{data_dir}/interim/last_seen_dates.parquet')
    included_drugs = load_included_drugs(data_dir=f'{data_dir}/external')
    with open(config_path) as file:
        cfg = yaml.safe_load(file)

    if align_on == 'treatment-dates':
        df = trt

    elif align_on == 'weekly-mondays':
        mrns = dmg['mrn'].unique()
        dates = pd.date_range(start='2018-01-01', end='2018-12-31', freq='W-MON')
        df = pd.DataFrame(product(mrns, dates), columns=['mrn', main_date_col])

    elif align_on.endswith('.parquet.gzip') or align_on.endswith('.parquet'):
        df = pd.read_parquet(align_on)

    elif align_on.endswith('.csv'):
        df = pd.read_csv(align_on, parse_dates=[main_date_col])

    else:
        raise ValueError(f'Sorry, aligning features on {align_on} is not supported yet')
    
    # Extract features
    df['last_seen_date'] = df['mrn'].map(last_seen['last_seen_date'])
    df['assessment_date'] = df[main_date_col]
    if align_on != 'treatment-dates':
        df = combine_treatment_to_main_data(df, trt, main_date_col, cfg['trt_lookback_window'])
    df = combine_demographic_to_main_data(df, dmg, main_date_col)
    df = merge_closest_measurements(df, sym, 'assessment_date', 'survey_date', time_window=cfg['symp_lookback_window'])
    df = merge_closest_measurements(df, lab, 'assessment_date', 'obs_date', time_window=cfg['lab_lookback_window'])
    df = combine_event_to_main_data(df, erv, 'assessment_date', 'event_date', event_name='ED_visit', lookback_window=cfg['ed_visit_lookback_window'])
    df = combine_perc_dose_to_main_data(df, included_drugs)
    df = add_engineered_features(df, 'assessment_date')

    # Extract targets
    df = get_death_labels(df, lookahead_window=[30, 365])
    df = get_ED_labels(df, erv[['mrn', 'event_date']].copy(), lookahead_window=30)
    df = get_symptom_labels(df, sym, lookahead_window=30)
    df = get_CTCAE_labels(df, lab, lookahead_window=30)
    
    df.to_parquet(f'{output_dir}/{output_filename}.parquet', compression='zstd', index=False)
    
if __name__ == '__main__':
    """
    Example of running the script:

    > python scripts/unify.py --align-on weekly-mondays --date-column assessment_date --output-filename \
        monday_centered_clinical_dataset
    """
    main()