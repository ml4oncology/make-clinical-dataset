import pandas as pd

###############################################################################
# Cleaners
###############################################################################
def clean_dosage(df: pd.DataFrame) -> pd.DataFrame:
    # Clean the dosing feature
    df['given_dose'] = df['given_dose'].apply(_clean_dosage)
    
    # Discard rows where given_dose does not match the following pattern. Only makes up ~0.26% of the dataset.
    # (i.e. first dilution, placebo, dosing for two drugs at once, etc)
    pattern1 = r'^\d+(?:\.\d+)?\s[\w/.\-%²μ]+$' # regex pattern for "<num> <unit>", where unit can be mg, mg/mL, etc
    pattern2 = r'nan\s[\w/.\-%²μ]+$' # regex pattern for "nan <unit>", where unit can be mg, mg/mL, etc
    mask = df['given_dose'].str.match(pattern1, na=True) | df['given_dose'].str.match(pattern2, na=True)
    # print(df.loc[~mask, 'given_dose'].unique())
    df = df[mask].copy()
    
    # Separate into its value and unit component
    df[['given_dose', 'given_dose_unit']] = df['given_dose'].str.split(' ', expand=True)
    
    # Fill nan with the ordered dose
    mask = df['given_dose'] == 'nan'
    df.loc[mask, 'given_dose'] = df.loc[mask, 'dose_ordered']
    
    # Convert to float
    df['given_dose'] = df['given_dose'].astype(float)

    return df


def _clean_dosage(text: str) -> str:
    if pd.isna(text): 
        return None

    # Special case
    if text == "125 mg (1)- 80 mg (2)":
        return "125 mg"

    # Remove surrounding parentheses
    if text.startswith('(') and text.endswith(')'):
        text = text[1:-1]

    # Remove commas in numbers (e.g., 1,000 → 1000)
    text = text.replace(',000', '000')

    # Insert space before % ONLY IF not already spaced (e.g., "0.9%" → "0.9 %")
    if '%' in text and ' ' not in text:
        text = text.replace('%', ' %')

    # Remove space in denominator (e.g., "300 mcg/0.5 mL" → "300 mcg/0.5mL")
    if '/' in text and text[-3] == ' ':
        text = text[:-3] + text[-2:]
        
    return text