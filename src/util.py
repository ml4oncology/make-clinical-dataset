from typing import Optional

import pandas as pd
from make_clinical_dataset.constants import OBS_MAP


###############################################################################
# I/O
###############################################################################
def load_included_drugs(data_dir: Optional[str] = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = './data/external'
        
    df = pd.read_csv(f'{data_dir}/opis_drug_list.csv')
    col_map = {'Drug_name': 'name', 'chemo': 'category', 'Recommended_dose_multiplier': 'recommended_dose_formula'}
    df = df.rename(columns=col_map)
    df = df.drop(columns=['counts'])
    df = df.query('category == "INCLUDE"')
    return df


def load_included_regimens(data_dir: Optional[str] = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = './data/external'
        
    df = pd.read_csv(f'{data_dir}/opis_regimen_list.csv')
    df.columns = df.columns.str.lower()
    return df


def load_lab_map(data_dir: str | None = None) -> dict[str, str]:
    """Get the lab name mappings. Due to EPR->EPIC migration, we have two mappings."""
    if data_dir is None:
        data_dir = './data/external'
        
    lab_name = pd.read_csv(f'{data_dir}/lab_names.csv')
    new_map = dict(lab_name[['obs_name', 'final']].astype(str).to_numpy())
    old_map = {**OBS_MAP['Hematology'], **OBS_MAP['Biochemistry']}
    lab_map = {k: old_map.get(v, v) for k, v in new_map.items()}
    return lab_map