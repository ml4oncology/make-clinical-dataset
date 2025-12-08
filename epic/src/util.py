import matplotlib.ticker as ticker
import pandas as pd
import polars as pl
import seaborn as sns
from make_clinical_dataset.shared import logger
from make_clinical_dataset.shared.constants import OBS_MAP


###############################################################################
# I/O
###############################################################################
def load_lab_map(data_dir: str | None = None) -> dict[str, str]:
    """Get the lab name mappings. Due to EPR->EPIC migration, we have two mappings."""
    if data_dir is None:
        data_dir = './data/external'
        
    lab_name = pd.read_csv(f'{data_dir}/lab_names.csv')
    new_map = dict(lab_name[['obs_name', 'final']].astype(str).to_numpy())
    old_map = {**OBS_MAP['Hematology'], **OBS_MAP['Biochemistry']}
    lab_map = {k: old_map.get(v, v) for k, v in new_map.items()}
    return lab_map


def load_table(data_path: str, mode: str = 'eager') -> pl.DataFrame | pl.LazyFrame:
    # TODO: move to ml-common?
    if data_path.endswith(('.parquet', '.parquet.gzip')):
        df = pl.read_parquet(data_path) if mode == 'eager' else pl.scan_parquet(data_path)
    if data_path.endswith('.csv'):
        df = pl.read_csv(data_path) if mode == 'eager' else pl.scan_csv(data_path)
    if data_path.endswith('.xlsx'):
        df = pl.read_excel(data_path)
    return df


###############################################################################
# EDA / Visualizations
###############################################################################
def plot_count_over_time(
    counts: pd.DataFrame, 
    catcol: str,
    x: str = "year", 
    y: str = "counts",
    **kwargs
) -> sns.FacetGrid:
    """Plot counts over time for each category in a categorical column.
    
    Args:
        counts: DataFrame with counts over time.
        catcol: The categorical column to create separate plots for.
        x: The x-axis column (default: "year").
        y: The y-axis column (default: "counts").
        **kwargs: Additional keyword arguments for sns.relplot().
    """
    defaults = {
        'col_wrap': 4,
        'kind': 'line',
        'markers': True,
        'markersize': 5,
        'marker': 'o',
        'facet_kws': {'sharex': False, 'sharey': False}
    }
    plot_kwargs = {**defaults, **kwargs} # user kwargs override defaults
    g = sns.relplot(data=counts, x=x, y=y, col=catcol, **plot_kwargs)
    for ax in g.axes.flat:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    return g


###############################################################################
# Helpers
###############################################################################
def get_excluded_numbers(df: pl.LazyFrame | pl.DataFrame, mask: pl.Expr, context: str = ".") -> None:
    """Report the number of rows that were excluded"""
    if isinstance(df, pl.DataFrame):
        mean, count = df.select(mask.mean()).item(), df.select(mask.sum()).item()   
    else:
        mean, count = df.select(mask.mean()).collect().item(), df.select(mask.sum()).collect().item()
    logger.info(f'Removing {count} ({mean*100:0.3f}%) rows{context}')