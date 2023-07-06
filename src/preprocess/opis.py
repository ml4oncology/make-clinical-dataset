"""
Module to preprocess OPIS (systemic therapy treatment data)
"""
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import re

from .. import ROOT_DIR
from ..util import get_num_removed_patients

# regex expressions
any_digit = '\d'
any_one_or_more_digit = '\d+'
any_char = '[a-zA-Z]'
any_one_or_more_char = '[a-zA-Z]+'
any_alphanumeric = '[a-zA-Z0-9]'
space_or_dash_or_plus = '[ \-+]'
optional = lambda char: f'{char}?'
either = lambda char1, char2: f'[{char1}|{char2}]'
not_match = lambda exp: f'(?!{exp})' # negative lookahead

def get_treatment_data(included_drugs: pd.DataFrame, data_dir: Optional[str] = None):
    if data_dir is None:
        data_dir = f'{ROOT_DIR}/data/raw'

    df = pd.read_parquet(f'{data_dir}/opis.parquet.gzip')
    df = filter_treatment_data(df, included_drugs)
    df = process_treatment_data(df)
    return df
    
def process_treatment_data(df):
    # order by date and regimen
    df = df.sort_values(by=['treatment_date', 'regimen'])

    # make each drug into two new columns (drug_given_dose, drug_regimen_dose), used to compute recommended ideal
    # dose and percentage of recommended ideal dose that was given
    given_dose = df.pivot(columns='drug_name', values='given_dose').loc[df.index]
    regimen_dose = df.pivot(columns='drug_name', values='regimen_dose').loc[df.index]
    given_dose.columns = 'drug_' + given_dose.columns + '_given_dose'
    regimen_dose.columns = 'drug_' + regimen_dose.columns + '_regimen_dose'
    dosage = pd.concat([given_dose, regimen_dose], axis=1).fillna(0)
    df = df.join(dosage)

    # merge rows with same treatment days
    format_regimens = lambda regs: '+'.join(sorted(set([item for reg in regs for item in reg.split('+')])))
    df = (
        df
        .groupby(['mrn', 'treatment_date'])
        .agg({
            # handle conflicting data by 
            # 1. join them togehter
            'regimen': format_regimens,
            # 2. take the mean 
            'height': 'mean',
            'weight': 'mean',
            'body_surface_area': 'mean',
            # 3. output True if any are True
            'with_radiation_therapy': 'any',
            
            # if two treatments (the old regimen and new regimen) overlap on same day, use data associated with the 
            # most recent regimen 
            # NOTE: examples found thru df.groupby(['mrn', 'treatment_date'])['first_treatment_date'].nunique() > 1
            'cycle_number': 'min',
            'first_treatment_date': 'max',
            
            'intent': 'first', # TODO: come up with robust way to handle conflicting intents
            'department': 'first', # TODO: come up with robust way to handle conflicting departments

            # sum the dosages together
            **{col: 'sum' for col in dosage.columns}
        })
    )
    df = df.reset_index()

    # forward fill height and weight
    for col in ['height', 'weight']: df[col] = df.groupby('mrn')[col].ffill()

    return df

def filter_treatment_data(df, included_drugs: pd.DataFrame):
    # clean column names
    df.columns = df.columns.str.lower()
    col_map = {
        'hosp_chart': 'mrn', 
        'trt_date': 'treatment_date', 
        'first_trt_date': 'first_treatment_date',
        'dose_ord': 'dose_ordered',
        'dose_given': 'given_dose'
    }
    df = df.rename(columns=col_map)
    
    # clean intent feature
    df['intent'] = df['intent'].replace('U', np.nan)
    
    df = filter_regimens(df)
    df = filter_drugs(df, included_drugs)

    df = process_regimens(df)
    df = process_drugs(df)

    # remove one-off duplicate rows (all values are same except for one, most likely due to human error)
    for col in ['first_treatment_date', 'cycle_number']: 
        cols = df.columns.drop(col)
        df = df.drop_duplicates(subset=cols, keep='first')
    
    return df

def filter_regimens(df):
    # filter out rows with missing regimen info
    mask = df['regimen'].notnull()
    get_num_removed_patients(df, mask, context='with missing regimen info')
    df = df[mask].copy()

    # separate department into a new column 
    df[['department', 'regimen']] = df['regimen'].str.split('-', n=1, expand=True)

    # only keep clinical trials (CT) and aerodigestive (gastrointestinal (GI), head & neck (HN), lung (LU)) regimens
    mask = df['department'].isin(['CT', 'GI', 'HN', 'LU'])
    df = df[mask]
    
    # group all clinical trials into TRIAL regimen
    mask = df['department'] == 'CT'
    df.loc[mask, 'regimen'] = 'TRIALS'
    return df

def process_regimens(df):
    df['original_regimen_entry'] = df['regimen'].copy()
    
    # separate modification into a new column
    pattern = f'{space_or_dash_or_plus}MOD'
    mask = df['regimen'].str.contains(pattern)
    df['modified_treatment'] = mask
    df['regimen'] = df['regimen'].str.replace(pattern, '', regex=True)
    
    # separate dose type (maintenance dose vs loading dose) into a new column
    df['dose_type'] = np.nan
    df['regimen'] = df['regimen'].str.replace('MAINT', 'MAIN')
    for pattern, dose_type in {'MAIN': 'maintenance', 'LOAD': 'loading'}.items():
        mask = df['regimen'].str.contains(pattern)
        df.loc[mask, 'dose_type'] = dose_type
        df['regimen'] = df['regimen'].str.replace(pattern, '')

    # separate radiation therapy into a new column
    df['with_radiation_therapy'] = np.nan
    for pattern, with_rt in {'NO RT': False, f'{space_or_dash_or_plus}RT': True}.items():
        mask = df['regimen'].str.contains(pattern)
        df['regimen'] = df['regimen'].str.replace(pattern, '', regex=True)
        df.loc[mask, 'with_radiation_therapy'] = with_rt

    # separate COMPASS trial into a new column
    pattern = 'COMPASS|COMP|COM'
    df['COMPASS_trial'] = df['regimen'].str.contains(pattern)
    df['regimen'] = df['regimen'].str.replace(pattern, '', regex=True)
    
    # clean regimen feature
    # TODO: debug why the below line doesn't work
    # df[['regimen', 'curated_regimen_notes']] = df['regimen'].apply(clean_regimen, result_type='expand')
    regimens_map, notes_map = {}, {}
    for regimen in df['regimen'].unique():
        cleaned_regimen, note = clean_regimen(regimen)
        regimens_map[regimen] = cleaned_regimen
        notes_map[regimen] = note
    df['curated_regimen_notes'] = df['regimen'].map(notes_map)
    df['regimen'] = df['regimen'].map(regimens_map)

    return df

def filter_drugs(df, included_drugs: pd.DataFrame):
    # filter out rows with trial, supportive, or non-aerodigestive drug entries
    mask = df['drug_name'].isin(included_drugs['name'])
    get_num_removed_patients(df, mask, context='who received only trial, supportive, and/or non-aerodigestive drugs')
    df = df[mask]

    # filter out rows where no dosage is given
    # e.g. patients get vital sign examination but don't receive treatment
    mask = df['given_dose'] > 0
    get_num_removed_patients(df, mask, context='where dosage is not provided')
    df = df[mask]
    return df

def process_drugs(df):
    df['original_drug_entry'] = df['drug_name'].copy()
    
    # separate receival of placebo into a new column
    mask = df['drug_name'].str.contains('/PLACEBO')
    df['with_placebo'] = mask
    df['drug_name'] = df['drug_name'].str.replace('/PLACEBO', '')
    
    # clean drug feature
    drug_map, notes_map = {}, {}
    for drug in df['drug_name'].unique():
        cleaned_drug, note = clean_drug_name(drug)
        drug_map[drug] = cleaned_drug
        notes_map[drug] = note
    df['curated_drug_notes'] = df['drug_name'].map(notes_map)
    df['drug_name'] = df['drug_name'].map(drug_map)
    return df

###############################################################################
# Cleaners
###############################################################################
def clean_regimen(regimen: str) -> Tuple[str, str]:
    note = ''
    
    # make entries consistent (same abbreviations)
    replace_map = {
        'TRAS': ['TRAST'],
        'BEVA': ['BEVACIZUMAB'],
        'RAMU': ['RAMUCIRUMAB', 'RAMUC'],
        'PEMB': ['PEMBROLIZUMAB', 'PEMBRO'],
        'PNTM': ['PANITUMUMAB'],
        'NIVL': ['NIVOLUMAB', 'NIVO'],
        'DURVA': ['DURVALUMAB'],
        'CETU': ['CETUXIMAB', 'CETUX'],
        'PEME': ['PEMETREXED'],
        'RALT': ['RALTITREXED', 'RALTI'],
        'IRIN': ['IRINO'],
        'CRBP': ['CARBO', 'CARB'],
        'CISP': ['CISPLATIN', 'CISPLAT'],
        'DOCE': ['DOCETAXEL'],
        'PACL': ['PACLITAXEL', 'PACLITAX', 'PACLI'],
        'NPAC': ['ABRAXANE', 'ABRAX'],
        'FU': ['5FU'],
        'APR': ['APREPITANT', 'APREP'],
        'W': ['WEEKLY', 'WEEKS', 'WEEK', 'WKLY', 'WK']
    }
    for new_substr, old_substrs in replace_map.items():
        for old_substr in old_substrs:
            regimen = regimen.replace(old_substr, new_substr)
            
    # remove excess regimen information from the regimen entries
    note = ''
    substrs = [
        'IND-NPC', # TODO: Ask what does it stand for?
        'BS', # TODO: Ask what does it stand for?
        'CCO', # Cancer Care Ontario
        'SAP', # Special Access Program
        'ADJ', # Adjuvant therapy
        'CIV', # Continuous intravenous infusion
        'FIXED',
        'MVASI', # Biosimilar version of Bevacizumab
        'NSCLC', # Non-small cell lung cancer,
        'ELDERLY',
        'BILIARY',
        'PANCREAS',
        'GASTRIC',
        'ESOPHAGEAL',
        'ANAL',
        'THYMOMA'
    ]
    for substr in substrs:
        if substr in regimen:
            regimen = regimen.replace(substr, '')
            note += f'{substr}; '
            
    patterns = [
        f'Q{any_digit}W', # e.g. Q2W
        f'WX{any_digit}', # e.g. WX2
        f'{any_digit}-W', # e.g. 2-W
        f'{any_digit}X/W', # e.g. 2X/W
        f'X{any_digit}{any_one_or_more_char}', # e.g. X6MON
        f'{either("D", "C")}{any_digit},{any_one_or_more_digit},{any_one_or_more_digit}', # e.g. D1,8,15
        f'D{any_digit},{any_one_or_more_digit}', # e.g. D1,15
        f'D{any_digit}-{any_digit}', # e.g. D1-4
        f'{either("D", "C")}{any_digit}', # e.g. C1
        f'CYC {any_digit},{any_digit}', # e.g. CYC 1,2
        f'{any_digit} DAY{optional("S")}', # e.g. 3 DAYS
        f'{any_one_or_more_digit}MG/{any_char}{any_alphanumeric}' # e.g. 20MG/M2
    ]
    for pattern in patterns:
        substrs = re.findall(pattern, regimen)
        if len(substrs) > 0:
            assert len(substrs) == 1
            substr = substrs[0]
            regimen = regimen.replace(substr, '')
            note += f'{substr}; '
    
    if 'W' in regimen:
        regimen = regimen.replace('W', '')
        note += f'Weekly; '
    
    # clean up punctuation marks
    for chars in ['()', '/', ';', ' ']:
        regimen = regimen.replace(chars, '') # remove empty brackets, slashes, semicolons, white spaces
    regimen = regimen.rstrip('-(+') # remove trailing dash, open bracket, plus sign
    
    # elongate some of the shortened abbreviations to make entries consistent
    pattern_map = {
        'CISP': f'CIS{not_match("P")}',
        'PEME': f'PEM{not_match("E|B")}',
    }
    for replacement, pattern in pattern_map.items():
        regimen = re.sub(pattern, replacement, regimen)
    
    return regimen, note

def clean_drug_name(drug: str) -> Tuple[str, str]:
    note = ''
    
    # remove excess drug information from the drug entries
    note = ''
    substrs = [
        '- PAID',
        'SAP',
        'SPECIAL ACCESS',
        'STUDY',
        'TRIAL',
        'COMPASSIONATE',
        'SUPPLY',
        'SUPPL',
        'SUP',
        'MVASI', # Biosimilar version of bevacizumab
        'AVASTIN', # Brand name for bevacizumab
        'OGIVRI', # Brand name of biosimilar version of trastuzumab
        'HERCEPTIN', # Brand name of trastuzumab
        'ABRAXANE', # Brand name of paclitaxel
        'ONIVYDE', # Brand name of irinotecan liposome injection
        'HCL', # hydrochloride
        'DISODIUM',
        'TARTRATE',
    ]
    for substr in substrs:
        if substr in drug:
            drug = drug.replace(substr, '')
            note += f'{substr}; '
            
    drug = drug.replace('PACLITAXEL', 'PACL')
            
    # clean up punctuation marks
    drug = drug.replace('()', '') # remove empty brackets
    drug = drug.strip() 
    
    return drug, note
