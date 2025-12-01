import pandas as pd
import polars as pl
from make_clinical_dataset.shared import logger
from make_clinical_dataset.shared.constants import OBS_MAP


###############################################################################
# I/O
###############################################################################
def load_lab_map(data_dir: str | None = None) -> dict[str, str]:
    """Get the lab name mappings. Due to EPR->EPIC migration, we have two mappings."""
    if data_dir is None:
        data_dir = './data/external'
        
    lab_name = pd.read_csv(f'{data_dir}/lab_names.csv')
    new_map = dict(lab_name[['obs_name', 'final']].astype(str).to_numpy())
    old_map = {**OBS_MAP['Hematology'], **OBS_MAP['Biochemistry']}
    lab_map = {k: old_map.get(v, v) for k, v in new_map.items()}
    return lab_map


def load_table(data_path: str, mode: str = 'eager') -> pl.DataFrame | pl.LazyFrame:
    # TODO: move to ml-common?
    if data_path.endswith(('.parquet', '.parquet.gzip')):
        df = pl.read_parquet(data_path) if mode == 'eager' else pl.scan_parquet(data_path)
    if data_path.endswith('.csv'):
        df = pl.read_csv(data_path) if mode == 'eager' else pl.scan_csv(data_path)
    if data_path.endswith('.xlsx'):
        df = pl.read_excel(data_path)
    return df


###############################################################################
# Helpers
###############################################################################
def get_excluded_numbers(df: pl.LazyFrame | pl.DataFrame, mask: pl.Expr, context: str = ".") -> None:
    """Report the number of rows that were excluded"""
    if isinstance(df, pl.DataFrame):
        mean, count = df.select(mask.mean()).item(), df.select(mask.sum()).item()   
    else:
        mean, count = df.select(mask.mean()).collect().item(), df.select(mask.sum()).collect().item()
    logger.info(f'Removing {count} ({mean*100:0.3f}%) rows{context}')