import difflib
from typing import Optional

import pandas as pd
from make_clinical_dataset.shared import logger


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


###############################################################################
# Helpers
###############################################################################
def get_excluded_numbers(df, mask: pd.Series, context: str = ".") -> None:
    """Report the number of patients and sessions that were excluded"""
    N_sessions = sum(~mask)
    N_patients = len(set(df["mrn"]) - set(df.loc[mask, "mrn"]))
    logger.info(f"Removing {N_patients} patients and {N_sessions} sessions{context}")


###############################################################################
# Text
###############################################################################
def combine_text(
    df: pd.DataFrame, 
    date_col: str = 'date', 
    text_col: str = 'text',
    add_prev_text: bool = False,
    no_same_lines: bool = True,
) -> pd.DataFrame:
    """Combine notes/reports that occured on the same day

    Also supports combining all previous text with the current text

    Args:
        add_prev_text: If True, combines all past text to the current text
        no_same_lines: If True, removes duplicate lines
    """
    if add_prev_text:
        raise NotImplementedError('Combining all past text not supported yet')

    # combine same day texts
    def combine(data, fill_value=''): 
        return '\n'.join(dict.fromkeys(data.fillna(fill_value)))
    string_cols = df.columns[df.dtypes == 'object']
    other_cols = df.columns[df.dtypes != 'object'].drop(['mrn', date_col])
    df = df.groupby(['mrn', date_col]).agg({
        **{col: combine for col in string_cols},
        **{col: 'max' for col in other_cols}
    }).reset_index()

    if no_same_lines:
        # remove duplicate lines (some may be same text with addendums)
        df[text_col] = remove_duplicate_lines(df[text_col])

    return df


def remove_duplicate_lines(text: pd.Series) -> pd.Series:
    lines = text.str.split('\n')
    return lines.apply(lambda text: "\n".join(dict.fromkeys(text)))


def get_text_size(text: pd.Series):
    return pd.concat([
        text.str.len().describe(),
        text.str.split().str.len().describe(),
    ], keys=['Length of text', 'Number of words'], axis=1)


def get_text_diff(text1: str, text2: str) -> str:
    diff = difflib.unified_diff(
        text1.splitlines(), text2.splitlines(),
        fromfile='text1', tofile='text2', lineterm=''
    )
    return '\n'.join(diff)