import logging
from collections.abc import Sequence

import pandas as pd
from make_clinical_dataset.shared.constants import (
    EPR_DRUG_COLS,
    LAB_CHANGE_COLS,
    LAB_COLS,
    SYMP_CHANGE_COLS,
    SYMP_COLS,
)
from ml_common.engineer import collapse_rare_categories
from ml_common.util import get_excluded_numbers
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


###############################################################################
# Default columns for data transformations
###############################################################################
# imputation
DEFAULT_IMPUTE_COLS = {
    "mean": LAB_COLS + LAB_CHANGE_COLS,
    "most_frequent": SYMP_COLS + SYMP_CHANGE_COLS,
    "median": [],
}
# one-hot encoding
DEFAULT_ENCODE_COLS = ["regimen", "intent"]
# outlier clipping
DEFAULT_CLIP_COLS = (
    ["body_surface_area", "height", "weight"] + LAB_COLS + LAB_CHANGE_COLS
)
# normalization
DEFAULT_NORM_COLS = (
    [
        "age",
        "body_surface_area",
        "cycle_number",
        "days_since_last_treatment",
        "days_since_prev_ED_visit",
        "days_since_starting_treatment",
        "height",
        "line_of_therapy",
        "num_prior_ED_visits_within_5_years",
        "visit_month_sin",
        "visit_month_cos",
        "weight",
    ]
    + EPR_DRUG_COLS + LAB_COLS + LAB_CHANGE_COLS + SYMP_COLS + SYMP_CHANGE_COLS
)


###############################################################################
# Splitting
###############################################################################
class Splitter:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def split_data(
        self,
        df: pd.DataFrame,
        split_date: str,
        **kwargs
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create the training, validation, and testing set"""
        # split data temporally based on patients first visit date
        train_data, test_data = self.temporal_split(df, split_date=split_date, **kwargs)

        def disp(x):
            return f"NSessions={len(x)}. NPatients={x.mrn.nunique()}. Contains all patients whose first visit was "

        logger.info(f"Development Cohort: {disp(train_data)} on or before {split_date}")
        logger.info(f"Test Cohort: {disp(test_data)} after {split_date}")

        # create validation set from train data (80-20 split)
        train_data, valid_data = self.random_split(
            train_data, test_size=0.2, random_state=self.random_state
        )

        return train_data, valid_data, test_data

    def temporal_split(
        self,
        df: pd.DataFrame,
        split_date: str,
        visit_col: str = "treatment_date",
        exclude_after_split: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the data temporally based on patient's first visit date"""
        first_date = df.groupby("mrn")[visit_col].min()
        first_date = df["mrn"].map(first_date)
        mask = first_date <= split_date
        dev_cohort, test_cohort = df[mask].copy(), df[~mask].copy()

        if exclude_after_split:
            # remove visits in the dev_cohort that occured after split_date
            mask = dev_cohort[visit_col] <= split_date
            get_excluded_numbers(
                dev_cohort,
                mask,
                f" that occured after {split_date} in the development cohort",
            )
            dev_cohort = dev_cohort[mask]

        return dev_cohort, test_cohort

    def random_split(
        self, df: pd.DataFrame, test_size: float, random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the data randomly based on patient id"""
        gss = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_idxs, test_idxs = next(gss.split(df, groups=df["mrn"]))
        return df.iloc[train_idxs].copy(), df.iloc[test_idxs].copy()


###############################################################################
# Imputation
###############################################################################
class Imputer:
    """Impute missing data by mean, mode, or median"""

    def __init__(self, impute_cols: dict | None = None):
        self.impute_cols = DEFAULT_IMPUTE_COLS if impute_cols is None else impute_cols
        self.imputer = {"mean": None, "most_frequent": None, "median": None}
        # ensure the provided impute_cols has matching keys
        assert all([key in self.impute_cols for key in self.imputer])

    def impute(self, data: pd.DataFrame) -> pd.DataFrame:
        # loop through the mean, mode, and median imputer
        for strategy, imputer in self.imputer.items():
            # use only the columns that exist in the data
            cols = [col for col in self.impute_cols[strategy] if col in data.columns]
            
            if not cols:
                continue

            if imputer is None:
                # create the imputer and impute the data
                imputer = SimpleImputer(strategy=strategy)
                data[cols] = imputer.fit_transform(data[cols])
                self.imputer[strategy] = imputer  # save the imputer
            else:
                # use existing imputer to impute the data
                data[cols] = imputer.transform(data[cols])
        return data


def fill_missing_data_heuristically(
    df: pd.DataFrame, 
    zero_fills: Sequence[str] | None = None, 
    max_fills: Sequence[str] | None = None,
    min_fills: Sequence[str] | None = None,
    custom_fills: dict | None = None,
) -> pd.DataFrame:
    """
    Args:
        zero_fills: column names whose missing values will be filled with zeros
        max_fills: column names whose missing values will be filled with the max value in the column
        min_fills: column names whose missing values will be filled with the min value in the column
        custom_fills: a dictionary of column names and their corresponding fill values
    """
    if zero_fills is None: 
        zero_fills = ["num_prior_ED_visits_within_5_years", "days_since_starting_treatment"]
    if max_fills is None:
        max_fills = ["days_since_last_treatment", "days_since_prev_ED_visit"]
    if min_fills is None:
        min_fills = []
    if custom_fills is None:
        custom_fills = {}

    fill_vals = {
        **{col: 0 for col in zero_fills},
        **{col: df[col].max() for col in max_fills},
        **{col: df[col].min() for col in min_fills},
        **custom_fills,
    }
    df = df.fillna(fill_vals)
    return df

###############################################################################
# One-Hot Encoding
###############################################################################
class OneHotEncoder:
    """One-hot encode (OHE) categorical data.

    Create separate indicator columns for each unique category and assign binary values of 1 or 0 to indicate the
    category's presence.
    """

    def __init__(self, encode_cols: Sequence | None = None):
        self.encode_cols = DEFAULT_ENCODE_COLS if encode_cols is None else encode_cols
        self.final_columns = None  # the final feature names after OHE

    def encode(
        self, data: pd.DataFrame, collapse: bool = True, verbose: bool = True
    ) -> pd.DataFrame:
        """
        Args:
            collapse: If True, rare entries are collapsed into the 'other' category
        """
        # one-hot encode categorical columns
        # use only the columns that exist in the data
        cols = [col for col in self.encode_cols if col in data.columns]
        data = pd.get_dummies(data, columns=cols)

        if self.final_columns is None:
            if collapse:
                data = collapse_rare_categories(data, catcols=cols)
            self.final_columns = data.columns
            return data

        # reassign any indicator columns that did not exist in final columns as other
        for feature in cols:
            indicator_cols = data.columns[data.columns.str.startswith(feature)]
            extra_cols = indicator_cols.difference(self.final_columns)
            if extra_cols.empty:
                continue

            if verbose:
                count = data[extra_cols].sum()
                msg = (
                    f"Reassigning the following {feature} indicator columns "
                    f"that did not exist in train set as other:\n{count}"
                )
                logger.info(msg)

            other_col = f"{feature}_other"
            if other_col not in data:
                data[other_col] = 0
            data[other_col] |= data[extra_cols].any(axis=1).astype(int)
            data = data.drop(columns=extra_cols)

        # fill in any missing columns
        missing_cols = self.final_columns.difference(data.columns)
        # use concat instead of data[missing_cols] = 0 to prevent perf warning
        data = pd.concat(
            [data, pd.DataFrame(0, index=data.index, columns=missing_cols)], axis=1
        )

        return data


###############################################################################
# Transformation
###############################################################################
class PrepData:
    """Prepare the data for model training"""

    def __init__(
        self,
        clip_cols: Sequence | None = None,
        norm_cols: Sequence | None = None,
        encode_cols: Sequence | None = None,
        impute_cols: dict | None = None,
    ):
        self.imp = Imputer(impute_cols=impute_cols)
        self.ohe = OneHotEncoder(encode_cols=encode_cols)
        self.scaler = None  # normalizer
        self.clip_thresh = None  # outlier clippers
        self.norm_cols = DEFAULT_NORM_COLS if norm_cols is None else norm_cols
        self.clip_cols = DEFAULT_CLIP_COLS if clip_cols is None else clip_cols

    def transform_data(
        self,
        data,
        one_hot_encode: bool = True,
        clip: bool = True,
        impute: bool = True,
        normalize: bool = True,
        ohe_kwargs: dict | None = None,
        data_name: str | None = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Transform (one-hot encode, clip, impute, normalize) the data.

        Args:
            ohe_kwargs (dict): a mapping of keyword arguments fed into OneHotEncoder.encode

        IMPORTANT: always make sure train data is done first before valid or test data
        """
        if ohe_kwargs is None:
            ohe_kwargs = {}
        if data_name is None:
            data_name = "the"

        if one_hot_encode:
            # One-hot encode categorical data
            if verbose:
                logger.info(f"One-hot encoding {data_name} data")
            data = self.ohe.encode(data, **ohe_kwargs)

        if clip:
            # Clip the outliers based on the train data quantiles
            data = self.clip_outliers(data)

        if impute:
            # Impute missing data based on the train data mode/median/mean
            data = self.imp.impute(data)

        if normalize:
            # Scale the data based on the train data distribution
            data = self.normalize_data(data)

        return data

    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # use only the columns that exist in the data
        norm_cols = [col for col in self.norm_cols if col in data.columns]

        if self.scaler is None:
            self.scaler = StandardScaler()
            data[norm_cols] = self.scaler.fit_transform(data[norm_cols])
        else:
            data[norm_cols] = self.scaler.transform(data[norm_cols])
        return data

    def clip_outliers(
        self,
        data: pd.DataFrame,
        lower_percentile: float = 0.001,
        upper_percentile: float = 0.999,
    ) -> pd.DataFrame:
        """Clip the upper and lower percentiles for the columns indicated below"""
        # use only the columns that exist in the data
        cols = [col for col in self.clip_cols if col in data.columns]

        if self.clip_thresh is None:
            percentiles = [lower_percentile, upper_percentile]
            self.clip_thresh = data[cols].quantile(percentiles)

        data[cols] = data[cols].clip(
            lower=self.clip_thresh.loc[lower_percentile],
            upper=self.clip_thresh.loc[upper_percentile],
            axis=1,
        )
        return data
