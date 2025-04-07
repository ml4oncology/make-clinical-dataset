from glob import glob
import pandas as pd
from pathlib import Path

root_dir = '/cluster/projects/gliugroup/2BLAST'

# Centralize the curated procedure names and procedure codes
mapping = []
for file in glob(f'{root_dir}/data/info/procs_2022-06-01/*.csv'):
    category = Path(file).stem.split('_', 2)[-1]
    df = pd.read_csv(file, encoding='latin')
    df['Category'] = category
    mapping.append(df[['Code', 'Description', 'Category']])
mapping = pd.concat(mapping)
mapping['Code'] = mapping['Code'].astype(str)
mapping['Description'] = mapping['Description'].apply(lambda txt: txt.strip().replace('\xa0', ' ').lower())
mapping = mapping.groupby('Code').agg({
    'Description': lambda regs: ' && '.join(sorted(set(regs))),
    'Category': lambda regs: ' && '.join(sorted(set(regs))),
})
mapping.to_csv(f'{root_dir}/data/info/code_descs.csv')

proc_names, proc_codes = [], []
for file in glob(f'{root_dir}/data/info/procs_2025-03-01/*.csv'):
    category, proc_type = Path(file).stem.split('_', 1)
    df = pd.read_csv(file, header=None).drop_duplicates()
    df['value'] = df.pop(0).astype(str)
    
    if category in ['Biochemistry', 'Hematology']:
        df['category'] = 'lab'
        df['sub-category'] = category
    else:
        df['category'] = category
        
    if proc_type == 'proc_codes': 
        proc_codes.append(df)
    elif proc_type == 'proc_names': 
        proc_names.append(df)
    else:
        raise ValueError
proc_names, proc_codes = pd.concat(proc_names), pd.concat(proc_codes)
proc_codes['description'] = proc_codes['value'].map(mapping['Description'])
assert not proc_codes['value'].duplicated().any()
assert not proc_names['value'].duplicated().any()
proc_names.to_csv(f'{root_dir}/data/info/proc_names.csv', index=False)
proc_codes.to_csv(f'{root_dir}/data/info/proc_codes.csv', index=False)