import polars as pl


###############################################################################
# Checker
###############################################################################
def check_data_loss(df):
    """Check what characters were lost when encoding was set to utf8-lossy when reading files with polars.
    """
    # Check if any column contains the replacement character
    def contains_replacement_char(s: pl.Series) -> bool:
        # Returns True if any value contains '�'
        return s.str.contains("�").any()
    
    # Apply to all string columns
    string_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.Utf8]
    found_issues = {col: contains_replacement_char(df[col]) for col in string_cols}
    
    print("Columns with potential data loss (replacement chars):", 
          {k: v for k, v in found_issues.items() if v})