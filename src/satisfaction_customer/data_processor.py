import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from satisfaction_customer.config import ProjectConfig
from satisfaction_customer.processing.data_handling import save_pipeline

RANDOM_SEED = 20230916


class DataProcessor:
    """A class to manage training, prediction, and integration of an sklearn pipeline with Databricks.

    Attributes
    ----------
    df : pd.DataFrame
        The input DataFrame containing features and target.
    pipeline : sklearn.pipeline.Pipeline
        The sklearn pipeline including preprocessing and estimator.
    spark : SparkSession
        Spark session to interface with Databricks.
    config : ProjectConfig
        Loaded configuration object containing feature lists and catalog details.

    """

    def __init__(
        self,
        df: pd.DataFrame,
        pipeline: Pipeline,
        spark: SparkSession,
        config: ProjectConfig,
    ) -> None:
        """Initialize the DataProcessor."""
        self.df = df
        self.pipeline = pipeline
        self.spark = spark
        self.config = config

    def transform(self) -> pd.DataFrame:
        """Apply the pipeline's fit_transform to the internal DataFrame."""
        df_transformed = self.pipeline.fit_transform(self.df)
        return df_transformed

    def split_data(
        self, test_size: float = 0.2, random_state: int = RANDOM_SEED
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and testing sets.

        Parameters
        ----------
        X : pd.DataFrame
            Features.
        test_size : float
            Proportion of the dataset to include in the test split.
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            X_train, X_test

        """
        df_transformed = self.pipeline.fit_transform(self.df)
        train_set, test_set = train_test_split(
            df_transformed, test_size=test_size, random_state=random_state
        )
        return train_set, test_set

    def _sanitize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sanitize column names for Delta Lake compatibility."""
        df.columns = [
            col.strip().replace(" ", "_").replace("(", "").replace(")", "").replace("\n", "")
            for col in df.columns
        ]
        return df

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        """Save the train and test sets into Databricks tables."""

        # 🧼 Clean column names
        train_set = self._sanitize_column_names(train_set)
        test_set = self._sanitize_column_names(test_set)

        # Convert to Spark DataFrames
        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )
        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        # Catalog/table path
        catalog = self.config.catalog_name
        schema = self.config.schema_name
        full_train_table = f"{catalog}.{schema}.train_set"
        full_test_table = f"{catalog}.{schema}.test_set"

        print(f"Saving training data to: {full_train_table}")
        print(f"Saving test data to: {full_test_table}")

        # Ensure context
        self.spark.sql(f"USE CATALOG {catalog}")
        self.spark.sql(f"USE SCHEMA {schema}")

        # Save
        train_set_with_timestamp.write.mode("overwrite").format("delta").saveAsTable(
            full_train_table
        )
        test_set_with_timestamp.write.mode("overwrite").format("delta").saveAsTable(
            full_test_table
        )

    def enable_change_data_feed(self) -> None:
        """Enable Change Data Feed for train and test set tables.

        This method alters the tables to enable Change Data Feed functionality.
        """
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

    def train(self) -> None:
        """Fit the sklearn pipeline on the current DataFrame and persist it using joblib."""
        X, y = separate_data(self.df)
        self.pipeline.fit(X, y)
        save_pipeline(self.pipeline)

    def predict(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions using the trained pipeline.

        Parameters
        ----------
        new_data : pd.DataFrame
            New input data to predict on.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with an additional 'prediction' column.

        """
        X, _ = separate_data(new_data)
        preds = self.pipeline.predict(X)
        df_out = new_data.copy()
        df_out["prediction"] = preds
        return df_out
