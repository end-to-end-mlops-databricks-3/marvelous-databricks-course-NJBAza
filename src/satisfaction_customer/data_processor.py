import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.pipeline import Pipeline

from satisfaction_customer.config import ProjectConfig
from satisfaction_customer.processing.data_handling import save_pipeline, separate_data, split_data


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

    def __init__(self, df: pd.DataFrame, pipeline: Pipeline, spark: SparkSession, config: ProjectConfig) -> None:
        """Initialize the DataProcessor."""
        self.df = df
        self.pipeline = pipeline
        self.spark = spark
        self.config = config

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

    def split_data(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split the internal DataFrame into training and testing sets.

        Parameters
        ----------
        test_size : float
            Proportion of the dataset to include in the test split.
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        tuple : X_train, X_test, y_train, y_test

        """
        X, y = separate_data(self.df)
        return split_data(X, y, test_size=test_size, random_state=random_state)

    def save_to_catalog(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        table_prefix: str = "",
    ) -> None:
        """Save training and test datasets to Databricks Unity Catalog tables.

        Parameters
        ----------
        X_train, y_train, X_test, y_test : pd.DataFrame
            Train/test split data to persist.
        table_prefix : str
            Optional prefix for table names.

        """
        df_train = pd.concat([X_train, y_train], axis=1)
        df_test = pd.concat([X_test, y_test], axis=1)

        train_spark = self.spark.createDataFrame(df_train).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )
        test_spark = self.spark.createDataFrame(df_test).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_spark.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.{table_prefix}train_set"
        )
        test_spark.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.{table_prefix}test_set"
        )

    def enable_change_data_feed(self, table_prefix: str = "") -> None:
        """Enable Delta Change Data Feed for the saved Unity Catalog tables.

        Parameters
        ----------
        table_prefix : str
            Prefix used when saving train/test tables.

        """
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.{table_prefix}train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
        )
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.{table_prefix}test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
        )
