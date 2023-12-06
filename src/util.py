from typing import Optional
import itertools
import multiprocessing as mp

import numpy as np
import pandas as pd

from . import ROOT_DIR, logger

###############################################################################
# I/O
###############################################################################
def load_included_drugs(data_dir: Optional[str] = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = f'{ROOT_DIR}/data/external'
        
    df = pd.read_csv(f'{data_dir}/opis_drug_list.csv')
    col_map = {'Drug_name': 'name', 'chemo': 'category', 'Recommended_dose_multiplier': 'recommended_dose_formula'}
    df = df.rename(columns=col_map)
    df = df.drop(columns=['counts'])
    df = df.query('category == "INCLUDE"')
    return df

def load_included_regimens(data_dir: Optional[str] = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = f'{ROOT_DIR}/data/external'
        
    df = pd.read_csv(f'{data_dir}/opis_regimen_list.csv')
    df.columns = df.columns.str.lower()
    return df

###############################################################################
# Multiprocessing
###############################################################################
def parallelize(generator, worker, processes: int = 4):
    pool = mp.Pool(processes=processes)
    result = pool.map(worker, generator)
    pool.close()
    pool.join() # wait for all threads
    result = list(itertools.chain(*result))
    return result

def split_and_parallelize(data, worker, split_by_mrns: bool = True, processes: int = 4):
    """Split up the data and parallelize processing of data
    
    Args:
        data: Supports a sequence, pd.DataFrame, or tuple of pd.DataFrames 
            sharing the same patient ids
        split_by_mrns: If True, split up the data by patient ids
    """
    generator = []
    if split_by_mrns:
        mrns = data[0]['mrn'] if isinstance(data, tuple) else data['mrn']
        mrn_groupings = np.array_split(mrns.unique(), processes)
        if isinstance(data, tuple):
            for mrn_grouping in mrn_groupings:
                items = tuple(df[df['mrn'].isin(mrn_grouping)] for df in data)
                generator.append(items)
        else:
            for mrn_grouping in mrn_groupings:
                item = data[mrns.isin(mrn_grouping)]
                generator.append(item)
    else:
        # splits df into x number of partitions, where x is number of processes
        generator = np.array_split(data, processes)
    return parallelize(generator, worker, processes=processes)

###############################################################################
# Data Descriptions
###############################################################################
def get_nunique_categories(df: pd.DataFrame) -> pd.DataFrame:
    catcols = df.dtypes[df.dtypes == object].index.tolist()
    return pd.DataFrame(
        df[catcols].nunique(), columns=['Number of Unique Categories']
    ).T

def get_excluded_numbers(df, mask: pd.Series, context: str = '.') -> None:
    """Report the number of patients and sessions that were excluded"""
    N_sessions = sum(~mask)
    N_patients = len(set(df['mrn']) - set(df.loc[mask, 'mrn']))
    logger.info(f'Removing {N_patients} patients and {N_sessions} sessions{context}')