from typing import Optional

import pandas as pd
from make_clinical_dataset.shared.constants import OBS_MAP


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