import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    StandardScaler,
    normalize,
)

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT.parent))


class DataFrameTypeConverter(BaseEstimator, TransformerMixin):
    """Transformer to convert data types of specified columns in a pandas DataFrame."""

    def __init__(self, conversion_dict: dict[str, Any]) -> None:
        """Initialize the DataFrameTypeConverter.

        Parameters
        ----------
        conversion_dict : dict
            Dictionary specifying column names and their target data types.

        """
        self.conversion_dict = conversion_dict

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DataFrameTypeConverter":
        """Fit method — does nothing as no fitting is required.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.
        y : pd.Series, optional
            Target variable. Ignored.

        Returns
        -------
        self : DataFrameTypeConverter
            Returns self.

        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply data type conversion based on the provided dictionary.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with converted column data types.

        """
        X = X.copy()
        for column, new_type in self.conversion_dict.items():
            X[column] = X[column].astype(new_type)
        return X


# dropping columns
class DropColumns(BaseEstimator, TransformerMixin):
    """Transformer to drop specified columns from a pandas DataFrame."""

    def __init__(self, variables_to_drop: list | str | None = None) -> None:
        """Initialize the DropColumns transformer.

        Parameters
        ----------
        variables_to_drop : list or str, optional
            Column name(s) to drop. Default is None.

        """
        self.variables_to_drop = variables_to_drop

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DropColumns":
        """Fit method — does nothing since no fitting is required.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.
        y : pd.Series, optional
            Target column. Ignored.

        Returns
        -------
        self : DropColumns
            Returns self.

        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop the specified columns from the DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with specified columns dropped.

        """
        X = X.copy()
        return X.drop(columns=self.variables_to_drop, errors="ignore")


# droping duplicates rows
class DropDuplicatesTransformer(BaseEstimator, TransformerMixin):
    """Transformer that removes duplicate rows from a DataFrame."""

    def __init__(self) -> None:
        """Initialize the transformer. No parameters required."""
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DropDuplicatesTransformer":
        """Fit method — does nothing as dropping duplicates doesn't require fitting.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y : pd.Series, optional
            Target variable (ignored).

        Returns
        -------
        self : DropDuplicatesTransformer
            Returns self.

        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows from the DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame with duplicates removed.

        """
        X = X.copy()
        X.drop_duplicates(inplace=True)
        return X


# imputing mode for categorical features
class ModeImputer(BaseEstimator, TransformerMixin):
    """Imputes missing categorical values using the mode of each specified column."""

    def __init__(self, variables: list[str] | None = None) -> None:
        """Initialize the imputer.

        Parameters
        ----------
        variables : list of str, optional
            List of column names to apply mode imputation.

        """
        self.variables = variables
        self.mode_dict: dict[str, str] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ModeImputer":
        """Compute the mode for each specified column.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.
        y : pd.Series, optional
            Target variable (ignored).

        Returns
        -------
        self : ModeImputer
            Fitted instance.

        """
        for col in self.variables:
            self.mode_dict[col] = X[col].mode()[0]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with the computed mode values.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame with missing values.

        Returns
        -------
        pd.DataFrame
            DataFrame with missing values filled using mode.

        """
        X = X.copy()
        for col in self.variables:
            X[col] = X[col].fillna(self.mode_dict[col])
        return X


# imputing median for numerical features
class MedianImputer(BaseEstimator, TransformerMixin):
    """Imputes missing numerical values using the median of each specified column."""

    def __init__(self, variables: list[str] | None = None) -> None:
        """Initialize MedianImputer.

        Parameters
        ----------
        variables : list of str, optional
            List of column names to impute. If None, nothing is imputed.

        """
        self.variables = variables
        self.median_dict: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "MedianImputer":
        """Compute the median for each specified column.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame.
        y : pd.Series, optional
            Target variable (ignored).

        Returns
        -------
        self : MedianImputer
            Fitted instance.

        """
        for col in self.variables:
            self.median_dict[col] = X[col].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with the computed medians.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame with missing values.

        Returns
        -------
        pd.DataFrame
            DataFrame with imputed values.

        """
        X = X.copy()
        for col in self.variables:
            X[col] = X[col].fillna(self.median_dict[col])
        return X


# class MedianImputer:
#     def __init__(self, variables=None):
#         self.variables = variables
#         self.median_dict = {}

#     def fit(self, X, y=None):
#         for col in self.variables:
#             # Calculate and store the median for each variable
#             self.median_dict[col] = X[col].median()
#         return self

#     def transform(self, X):
#         X = X.copy()
#         for col in self.variables:
#             # Fill missing values with the stored median for each variable
#             X[col].fillna(self.median_dict[col], inplace=True)
#         return X


# imputing mean for numerical features
class MeanImputer(BaseEstimator, TransformerMixin):
    """Imputes missing numerical values using the mean of each column."""

    def __init__(self, variables: list[str] | None = None) -> None:
        """Initialize MeanImputer.

        Parameters
        ----------
        variables : list of str, optional
            List of column names to impute. If None, nothing is imputed.

        """
        self.variables = variables
        self.mean_dict: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "MeanImputer":
        """Compute the mean for each specified column.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame.
        y : pd.Series, optional
            Target variable (ignored).

        Returns
        -------
        self : MeanImputer
            Fitted instance.

        """
        for col in self.variables:
            self.mean_dict[col] = X[col].mean()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with the computed means.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame with missing values.

        Returns
        -------
        pd.DataFrame
            DataFrame with imputed values.

        """
        X = X.copy()
        for col in self.variables:
            X[col] = X[col].fillna(self.mean_dict[col])
        return X


# Winsorizing all the numerical features
class Winsorizer(BaseEstimator, TransformerMixin):
    """Applies winsorization to specified numerical features to limit the influence of outliers."""

    def __init__(self, numerical_features: list[str], limits: list[float] | None = None) -> None:
        """Initialize Winsorizer.

        Parameters
        ----------
        numerical_features : list of str
            List of numerical columns to winsorize.
        limits : list of float, optional
            Lower and upper percentile limits for winsorization.
            Defaults to [0.025, 0.025].

        """
        self.numerical_features = numerical_features
        self.limits = limits if limits is not None else [0.025, 0.025]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "Winsorizer":
        """No fitting needed for winsorization.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        y : pd.Series, optional
            Target variable (ignored).

        Returns
        -------
        self : Winsorizer
            This instance.

        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply winsorization to the specified features.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with new winsorized features.

        """
        X = X.copy()
        for feature in self.numerical_features:
            X[f"{feature}_winsor"] = winsorize(X[feature], limits=self.limits)
        return X


class FeatureCreator(BaseEstimator, TransformerMixin):
    """Transformer to create new engineered features from existing columns."""

    def __init__(self) -> None:
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureCreator":
        """Fit method does nothing as this is a stateless transformer."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create new features based on combinations of existing columns."""
        X = X.copy()

        X["age_flight_distance_feat"] = X["age"] * X["flight_distance"]
        X["wifi_flight_distance_feat"] = X["inflight_wifi_service"] * X["flight_distance"]
        X["online_boarding_booking_feat"] = X["online_boarding"] * X["ease_of_online_booking"]

        X["inflight_service_feat"] = (
            X["inflight_wifi_service"] + X["food_and_drink"] + X["inflight_entertainment"] + X["inflight_service"]
        ) / 4

        X["preflight_experience_feat"] = (
            X["ease_of_online_booking"]
            + X["checkin_service"]
            + X["gate_location"]
            + X["departure_arrival_time_convenient"]
        ) / 4

        X["total_delay_feat"] = X["departure_delay_in_minutes"] + X["arrival_delay_in_minutes"]
        X["significant_delay_feat"] = np.where(
            (X["departure_delay_in_minutes"] > 15) | (X["arrival_delay_in_minutes"] > 15), 1, 0
        )
        X["short_flight_feat"] = np.where(X["flight_distance"] < 300, 1, 0)
        X["service_difference_feat"] = X["preflight_experience_feat"] - X["inflight_service_feat"]
        X["departure_delay_ratio_feat"] = X["departure_delay_in_minutes"] / (1 + X["total_delay_feat"])
        X["on_time_arrival_feat"] = np.where((X["arrival_delay_in_minutes"] == 0), 1, 0)

        X["satisfied_boarding_process_feat"] = np.where(
            (X["ease_of_online_booking"] > 4) & (X["online_boarding"] > 4), 1, 0
        )
        X["comfortable_flight_feat"] = np.where((X["seat_comfort"] > 4) & (X["leg_room_service"] > 4), 1, 0)
        X["long_delay_feat"] = np.where(
            (X["departure_delay_in_minutes"] > 45) | (X["arrival_delay_in_minutes"] > 45), 1, 0
        )
        X["total_service_time_feat"] = (
            X["gate_location"] + X["checkin_service"] + X["departure_arrival_time_convenient"]
        )
        X["long_haul_feat"] = np.where(X["flight_distance"] > 2000, 1, 0)

        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fit and transform in one step by creating new features."""
        return self.fit(X, y).transform(X)


# Considering the numerical features of the huge dataset
class NumericalFeatureSelector(BaseEstimator, TransformerMixin):
    """Transformer to select numerical and categorical columns, optionally including datetime columns.

    Removes specified columns from the numerical set and retains categorical columns for downstream use.
    """

    def __init__(self, remove_columns: list[str], include_datetime: bool = False) -> None:
        """Initialize the selector.

        Parameters
        ----------
        remove_columns : list of str
            Columns to exclude from the numerical feature list.
        include_datetime : bool, default=False
            Whether to include datetime columns as numerical features.

        """
        self.remove_columns = remove_columns
        self.include_datetime = include_datetime

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "NumericalFeatureSelector":
        """Identify and store columns for selection.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : pd.Series, optional
            Target variable (not used).

        Returns
        -------
        self : NumericalFeatureSelector
            Fitted instance.

        """
        self.categorical_columns_ = X.select_dtypes(include=["category", "object", "bool"]).columns.tolist()

        self.numerical_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()

        # Remove explicitly specified columns
        self.numerical_columns_ = [col for col in self.numerical_columns_ if col not in self.remove_columns]

        if self.include_datetime:
            datetime_cols = X.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
            self.numerical_columns_ += datetime_cols

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select and return the appropriate columns.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with selected columns.

        """
        return X[self.categorical_columns_ + self.numerical_columns_]


# Considering the numerical features of the huge dataset
class ChosenFeatures(BaseEstimator, TransformerMixin):
    """Select a subset of specified columns from the input DataFrame."""

    def __init__(self, columns: list[str]) -> None:
        """Initialize with the list of columns to retain.

        Parameters
        ----------
        columns : list of str
            List of column names to select.

        """
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ChosenFeatures":
        """No fitting necessary; return self."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with only the selected columns."""
        X = X.copy()
        return X[self.columns]


# Log transformation for numerical features
class LogTransforms(BaseEstimator, TransformerMixin):
    """Apply log transformation to specified or all numerical columns."""

    def __init__(self, variables: list[str] | None = None) -> None:
        """Initialize the LogTransforms instance.

        Parameters
        ----------
        variables : list of str, optional
            List of columns to apply log transform. If None, all numeric columns are used.

        """
        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "LogTransforms":
        """Identify numerical columns if none provided.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.
        y : pd.Series, optional
            Target column (ignored).

        Returns
        -------
        self : LogTransforms
            Fitted instance.

        """
        if self.variables is None:
            self.variables = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply log(1 + abs(x)) transformation to selected columns.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with log-transformed values.

        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        X = X.copy()
        for col in self.variables:
            if col in X.columns:
                X[col] = np.log(np.abs(X[col]) + 1)
            else:
                raise ValueError(f"The column '{col}' is not in the DataFrame")
        return X


# Scaling the data
class DataScaler(BaseEstimator, TransformerMixin):
    """Scales numerical features using a combination of StandardScaler and MinMaxScaler."""

    def __init__(self) -> None:
        """Initialize the scaler objects."""
        self.std_scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        self.numerical_columns: pd.Index | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DataScaler":
        """Fit scalers on the numerical columns of the DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to fit the scalers on.
        y : pd.Series, optional
            Target variable (ignored).

        Returns
        -------
        self : DataScaler
            Fitted instance.

        """
        self.numerical_columns = X.select_dtypes(include=["float64", "int64"]).columns
        if self.numerical_columns.size > 0:
            self.std_scaler.fit(X[self.numerical_columns])
            X_transformed = self.std_scaler.transform(X[self.numerical_columns])
            self.min_max_scaler.fit(X_transformed)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply both StandardScaler and MinMaxScaler to the numerical columns.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing the data to transform.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with scaled numerical columns.

        """
        X = X.copy()
        if self.numerical_columns is not None and self.numerical_columns.size > 0:
            scaled_standard = self.std_scaler.transform(X[self.numerical_columns])
            scaled_min_max = self.min_max_scaler.transform(scaled_standard)
            X[self.numerical_columns] = scaled_min_max
        return X


# finding the correlations
class CorrelationMatrixProcessor(BaseEstimator, TransformerMixin):
    """Transformer that removes highly correlated features based on a threshold."""

    def __init__(self, numerical_features: list[str] | None = None, threshold: float = 0.7) -> None:
        """Initialize the processor.

        Parameters
        ----------
        numerical_features : list of str, optional
            List of numerical features to evaluate. If None, all numerical columns will be used.
        threshold : float
            Correlation threshold above which features are dropped.

        """
        self.numerical_features = numerical_features
        self.threshold = threshold
        self.features_to_drop_: list[str] = []
        self.corr_matrix: pd.DataFrame = pd.DataFrame()

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CorrelationMatrixProcessor":
        """Fit the processor by computing the correlation matrix and identifying redundant features.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y : pd.Series, optional
            Ignored.

        Returns
        -------
        self : CorrelationMatrixProcessor
            Fitted instance.

        """
        if self.numerical_features is None:
            self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

        self.corr_matrix = X[self.numerical_features].corr().abs()
        upper = self.corr_matrix.where(np.triu(np.ones(self.corr_matrix.shape), k=1).astype(bool))
        self.features_to_drop_ = [col for col in upper.columns if any(upper[col] > self.threshold)]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove correlated features from the input DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        pd.DataFrame
            DataFrame with highly correlated features dropped.

        """
        return X.drop(columns=self.features_to_drop_, errors="ignore")

    def get_sorted_correlations(self) -> pd.DataFrame:
        """Return a sorted DataFrame of feature correlations.

        Returns
        -------
        pd.DataFrame
            Sorted pairwise correlation values.

        """
        s = self.corr_matrix.unstack()
        so = s.sort_values(kind="quicksort", ascending=False)
        return pd.DataFrame(so, columns=["Pearson Correlation"])


# low variance filtering
class FeatureVariance(BaseEstimator, TransformerMixin):
    """Removes features with variance below a given threshold."""

    def __init__(self, numerical_features: list[str] | None = None, threshold: float = 0.001) -> None:
        """Initialize the transformer.

        Parameters
        ----------
        numerical_features : list of str, optional
            Columns to evaluate for variance.
        threshold : float
            Minimum variance required to retain a feature.

        """
        self.numerical_features = numerical_features
        self.threshold = threshold
        self.variance_: pd.Series
        self.categorical_columns_: list[str]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureVariance":
        """Compute variance of features.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        y : pd.Series, optional
            Ignored.

        Returns
        -------
        self : FeatureVariance
            Fitted instance.

        """
        if self.numerical_features is None:
            self.numerical_features = X.select_dtypes(include=[float, int]).columns.tolist()

        normalized_data = normalize(X[self.numerical_features])
        data_scaled = pd.DataFrame(normalized_data, columns=self.numerical_features)
        self.variance_ = data_scaled.var()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Filter out low-variance features.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        pd.DataFrame
            DataFrame with high-variance numerical and all categorical columns.

        """
        significant_vars = [var for var in self.variance_.index if self.variance_[var] >= self.threshold]
        self.categorical_columns_ = X.select_dtypes(include=["category", "object", "bool"]).columns.tolist()
        return X[self.categorical_columns_ + significant_vars]

    def get_variance_dataframe(self) -> pd.DataFrame:
        """Get a DataFrame with variance values per column.

        Returns
        -------
        pd.DataFrame
            Variance statistics per column.

        """
        variance_df = pd.DataFrame(self.variance_, columns=["Variance"])
        variance_df["Column"] = variance_df.index
        variance_df.reset_index(drop=True, inplace=True)
        return variance_df


# transforming the categorical features to numericalimport pandas as pd
class LabelEncoderProcessor(BaseEstimator, TransformerMixin):
    """Apply Label Encoding to specified categorical columns."""

    def __init__(self, columns: list[str] | None = None) -> None:
        """Initialize the LabelEncoderProcessor.

        Parameters
        ----------
        columns : list of str, optional
            Categorical columns to encode. If None, encodes all object/category columns.

        """
        self.columns = columns
        self.encoders: dict[str, LabelEncoder] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "LabelEncoderProcessor":
        """Fit label encoders to the specified columns.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.
        y : pd.Series, optional
            Ignored.

        Returns
        -------
        LabelEncoderProcessor
            Fitted transformer instance.

        """
        if self.columns is None:
            self.columns = X.select_dtypes(include=["object", "category"]).columns.tolist()

        for column in self.columns:
            if column in X.columns:
                encoder = LabelEncoder()
                self.encoders[column] = encoder.fit(X[column])
            else:
                raise ValueError(f"Column '{column}' not found in DataFrame")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform specified columns using fitted LabelEncoders.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with label-encoded columns.

        """
        X = X.copy()
        for column in self.columns:
            if column in X.columns:
                encoder = self.encoders[column]
                X[column] = encoder.transform(X[column])
            else:
                raise ValueError(f"Column '{column}' not found in DataFrame")
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fit and transform in a single step.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.
        y : pd.Series, optional
            Ignored.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with label-encoded columns.

        """
        return self.fit(X, y).transform(X)


# one hot encoding creation features
class OneHotEncoderProcessor(BaseEstimator, TransformerMixin):
    """Applies one-hot encoding to specified columns."""

    def __init__(self, columns: list[str] | None = None, prefix: str | None = None) -> None:
        """Initialize the processor.

        Parameters
        ----------
        columns : list of str, optional
            Columns to one-hot encode.
        prefix : str, optional
            Prefix for new encoded columns.

        """
        self.columns = columns
        self.prefix = prefix

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "OneHotEncoderProcessor":
        """Fit step (no fitting needed for one-hot encoding).

        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        y : pd.Series, optional
            Ignored.

        Returns
        -------
        self : OneHotEncoderProcessor
            Fitted instance.

        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply one-hot encoding to the input DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        pd.DataFrame
            One-hot encoded DataFrame.

        """
        X = X.copy()
        X_encoded = pd.get_dummies(X, columns=self.columns, prefix=self.prefix)
        dummy_columns = [col for col in X_encoded.columns if self.prefix in col]
        X_encoded[dummy_columns] = X_encoded[dummy_columns].astype(int)
        return X_encoded

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fit and transform the input data.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        y : pd.Series, optional
            Ignored.

        Returns
        -------
        pd.DataFrame
            Transformed data.

        """
        return self.fit(X, y).transform(X)
