"""
Module for feature engineering
"""
import numpy as np
import pandas as pd
from make_clinical_dataset.shared import logger
from make_clinical_dataset.shared.constants import (
    LAB_CHANGE_COLS,
    LAB_COLS,
    SYMP_CHANGE_COLS,
    SYMP_COLS,
)


###############################################################################
# General
###############################################################################
def get_change_since_prev_session(df: pd.DataFrame) -> pd.DataFrame:
    """Get change in measurements since previous session"""
    cols = LAB_COLS + SYMP_COLS
    change_cols = LAB_CHANGE_COLS + SYMP_CHANGE_COLS

    # use only the columns that exist in the data
    mask = [col in df.columns for col in cols]
    cols, change_cols = np.array(cols)[mask], np.array(change_cols)[mask]

    result = df.groupby('mrn')[cols].diff() # assumes dataframe is sorted by visit date already
    result.columns = change_cols
    df = pd.concat([df, result], axis=1)
    return df


def get_missingness_features(
    df: pd.DataFrame, exclude_keyword: str = "target"
) -> pd.DataFrame:
    cols_with_nan = df.columns[df.isnull().any()]
    cols_with_nan = cols_with_nan[~cols_with_nan.str.contains(exclude_keyword)]
    df[cols_with_nan + "_is_missing"] = df[cols_with_nan].isnull()
    return df


def collapse_rare_categories(
    df: pd.DataFrame, catcols: list[str], N: int = 6
) -> pd.DataFrame:
    """Collapse rare categories to 'other'

    Args:
        N: the minimum number of patients a category must be assigned to
    """
    for feature in catcols:
        other_mask = False
        drop_cols = []
        for col in df.columns[df.columns.str.startswith(feature)]:
            mask = df[col]
            if df.loc[mask, "mrn"].nunique() < N:
                drop_cols.append(col)
                other_mask |= mask
        
        if not drop_cols:
            logger.info(f'No rare categories for {feature} was found')
            continue

        df = df.drop(columns=drop_cols)
        df[f"{feature}_other"] = other_mask
        msg = f"Reassigning the following {len(drop_cols)} indicators with less than {N} patients as other: {drop_cols}"
        logger.info(msg)

    return df


###############################################################################
# Time
###############################################################################
def get_visit_month_feature(df, col: str = 'treatment_date'):
    # convert to cyclical features
    month = df[col].dt.month - 1
    df['visit_month_sin'] = np.sin(2*np.pi*month/12)
    df['visit_month_cos'] = np.cos(2*np.pi*month/12)
    return df

def get_days_since_last_event(df, main_date_col: str = 'treatment_date', event_date_col: str = 'treatment_date'):
    if (df[main_date_col] == df[event_date_col]).all():
        return (df[main_date_col] - df[event_date_col].shift()).dt.days
    else:
        return (df[main_date_col] - df[event_date_col]).dt.days

def get_years_diff(df, col1: str, col2: str):
    return df[col1].dt.year - df[col2].dt.year


###############################################################################
# Treatment
###############################################################################
def get_line_of_therapy(df):
    # identify line of therapy (the nth different palliative intent treatment taken)
    # NOTE: all other intent treatment are given line of therapy of 0. Usually (not always but oh well) once the first
    # palliative treatment appears, the rest of the treatments remain palliative
    new_regimen = (df['first_treatment_date'] != df['first_treatment_date'].shift())
    palliative_intent = df['intent'] == 'PALLIATIVE'
    return (new_regimen & palliative_intent).cumsum()


###############################################################################
# Drug dosages
###############################################################################
def get_perc_ideal_dose_given(df, drug_to_dose_formula_map: dict[str, str]):
    """Convert given dose as a percentage of ideal (recommended) dose

    df must have weight, body surface area, age, female, and creatinine columns along with the dosage columns
    """
    result = {}
    for drug, dose_formula in drug_to_dose_formula_map.items():
        dose_col = f'drug_{drug}_given_dose'
        if dose_col not in df.columns: continue
        ideal_dose = get_ideal_dose(df, drug, dose_formula)
        perc_ideal_dose_given = df[dose_col] / ideal_dose # NOTE: 0/0 = np.nan, x/0 = np.inf
        perc_ideal_dose_given = perc_ideal_dose_given
        result[drug] = perc_ideal_dose_given
    result = pd.DataFrame(result)
    result.columns = '%_ideal_dose_given_' + result.columns
    return result

def get_ideal_dose(df, drug: str, dose_formula: str):
    col = f'drug_{drug}_regimen_dose'
    carboplatin_dose_formula = ('min(regimen_dose * 150, regimen_dose * (((140-age[yrs]) * weight [kg] * 1.23 * '
                                '(0.85 if female) / creatinine [umol/L]) + 25))')
    if dose_formula == 'regimen_dose': 
        return df[col]
    
    elif dose_formula == 'regimen_dose * bsa': 
        return df[col] * df['body_surface_area']
    
    elif dose_formula == 'regimen_dose * weight': 
        return df[col] * df['weight']
    
    elif dose_formula == carboplatin_dose_formula:
        return pd.concat([df[col] * 150, df[col] * (get_creatinine_clearance(df) + 25)], axis=1).min(axis=1)
    
    else:
        raise ValueError(f'Ideal dose formula {dose_formula} not supported')


###############################################################################
# Special Formulas
###############################################################################
def get_creatinine_clearance(df):
    return (140 - df['age']) * df['weight'] * 1.23 * df['female'].map({True: 0.85, False: 1}) / df['creatinine']