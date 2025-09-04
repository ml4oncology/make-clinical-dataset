"""
Module to preprocess emergency department visit (old pull) / emergency room data (new pull)
"""
import pandas as pd
from make_clinical_dataset.epr.util import get_excluded_numbers
from make_clinical_dataset.shared import logger
from ml_common.util import split_and_parallelize


###############################################################################
# ER (Emergency Room - NEW PULL)
###############################################################################
def get_emergency_room_data(data_dir: str | None = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = './data/raw'

    df = pd.read_parquet(f'{data_dir}/ER.parquet.gzip')
    df = filter_emergency_room_data(df)
    df = process_emergency_room_data(df)
    return df


def filter_emergency_room_data(ER: pd.DataFrame) -> pd.DataFrame:
    ER = clean_emergency_data(ER, date_col='event_date_time')
    ER = ER.drop_duplicates()

    # confirming the following rows are unnecessary and can be removed
    assert all(ER.loc[ER['result_name'].isnull(), 'result_value'] == 'Triage Assessment')
    admin_notes = [
        '*** NOTE: Patients presenting with rash or fever:Chickenpox, shingles or measles require immeditate airbourne precautions Meningitis requires immediate droplet precautions ***',
        'GEM Assessment',
        '*** NOTE: 1. Patients presenting with full body rash with or without fever (ie. chicken pox, shingles or measles) require immediate airborne precautions.  2. Suspected meningitis requires immediate droplet precautions. ***',
        '*** NOTE: Suspected meningitis requires immediate droplet precautions. ***',
        'Note:  POCT Glucose Reference Range 3.8 to 7 mmol/L'
    ]
    assert all(ER.loc[ER['result_name'] == "---------------", 'result_value'].isin(admin_notes))
    assert all(ER.loc[ER['result_name'] == "--------", 'result_value'].isin(admin_notes))
        
    # each visit has its assessment report separated into single rows (one row per sentence / assessment item)
    # remove rows that does not provide useful information or contains unnecessary admin notes
    ER['result_name'] = ER['result_name'].replace({"---------------": None, "--------": None})
    ER = ER[ER['result_name'].notnull()]

    return ER


def process_emergency_room_data(ER: pd.DataFrame) -> pd.DataFrame:
    # process the ER data - combine conflicting info and collapse the assessment report into one row, 
    # where each assessment item is its own column
    result = split_and_parallelize(ER, emergency_room_worker)
    ER = pd.DataFrame({visit_number: info for visit_number, info in result}).T

    # organize the new columns
    def clean_col(col):
        col = col.replace('/', ' and ').replace('(', '').replace(')', '').replace('?', '')
        return '_'.join([word.lower() if not word.isupper() else word for word in col.split(' ')])
    ER.columns = [clean_col(col) for col in ER.columns]
    main_cols = ['mrn', 'event_date', 'CTAS_score', 'CEDIS_complaint', 'chief_complaint']
    ER = ER[main_cols + ER.columns.drop(main_cols).tolist()]

    # clean up the CTAS score
    ER['CTAS_score'] = clean_CTAS_score(ER['CTAS_score'])

    # order by patient and date
    ER = ER.sort_values(by=['mrn', 'event_date'])

    # remove partially duplicate entries
    # i.e. ER visits occuring within 30 minutes, 80% of the entries duplicate
    ER = remove_partially_duplicate_entries(ER)

    return ER


def emergency_room_worker(partition: pd.DataFrame) -> list:
    """Worker to process the ER data"""
    result = []
    for (mrn, event_date), group in partition.groupby(['mrn', 'event_date']):
        assert group['visit_number'].nunique() == 1
        visit_number = group['visit_number'].iloc[0]

        # collapse conflicting information into one row
        group = group.groupby('result_name').agg({'result_value': collapse})
        # convert to a dict
        info = group['result_value'].to_dict()
        # insert rest of info
        info['mrn'], info['event_date'] = mrn, event_date
        
        result.append((visit_number, info))

    return result


def clean_CTAS_score(score: pd.Series) -> pd.Series:
    """Clean up the CTAS (Canadian Triage and Acuity Scale) score

    CTAS score should be between 1 - 5
    CTAS I: severely ill, requires resuscitation 
    CTAS II: requires emergent care and rapid medical intervention 
    CTAS III: requires urgent care 
    CTAS IV: requires less-urgent care 
    CTAS V: requires non-urgent care
    """
    score = score.astype(float)
    # Not sure why it's between 6 - 10. Quality check reveals severity is also in decreasing order
    assert all(score.between(6, 10) | score.isnull())
    score -= 5 # adjust the score to be between 1 - 5
    return score


def remove_partially_duplicate_entries(ER: pd.DataFrame) -> pd.DataFrame:
    exclude = []
    for mrn, group in ER.groupby('mrn'):
        # if event date occurs within 30 minutes of each other, it's most likely duplicate entries
        # even though they are given different visit numbers
        time_since_next_visit = group['event_date'].shift(-1) - group['event_date']
        mask = time_since_next_visit < pd.Timedelta(seconds=60*30)
        if mask.any(): 
            # ensure the rows are really duplicates (80% of the entries are the same)
            # NOTE: able to assess multiple duplicate rows
            tmp1 = group[mask].fillna('NULL').reset_index(drop=True)
            tmp2 = group[mask.shift(1).fillna(False)].fillna('NULL').reset_index(drop=True)
            assert all((tmp1 == tmp2).mean(axis=1) > 0.80)

            # we take the most recent entry, excluding the previous entries
            exclude += group.index[mask].tolist()

    mask = ~ER.index.isin(exclude)
    get_excluded_numbers(ER, mask, context=' which are duplicate entries')
    ER = ER[mask]
    return ER

###############################################################################
# Helpers
###############################################################################
def clean_emergency_data(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    # clean column names
    df.columns = df.columns.str.lower()
    df = df.rename(columns={'medical_record_number': 'mrn', date_col: 'event_date'})
    # clean data types
    df['event_date'] = pd.to_datetime(df['event_date'], format="%d%b%Y:%H:%M:%S")
    return df


def collapse(g: pd.Series) -> pd.Series:
    """Collapse conflicting information into one row"""
    combined_info = ' && '.join(sorted(set(g.astype(str))))
    return pd.Series(combined_info, index=g.index[:1])


###############################################################################
# ED (Emergency Department - OLD PULL)
###############################################################################
def get_emergency_department_data(data_dir: str | None = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = './data/raw'
    df = pd.read_parquet(f'{data_dir}/ED.parquet.gzip')
    df = process_emergency_department_data(df)
    return df


def process_emergency_department_data(ED: pd.DataFrame) -> pd.DataFrame:
    ED = clean_emergency_data(ED, date_col='admission_date_time')

    # sort by date
    ED = ED.sort_values(by=['mrn', 'event_date'])

    # remove duplicate visits
    mask = ED[['mrn', 'event_date']].duplicated(keep='last')
    logger.info(f'Removing {sum(mask)} duplicate emergency department visits') 
    ED = ED[~mask]
    return ED