"""Basic model implementation.

infer_signature (from mlflow.models) → Captures input-output schema for model tracking.

num_features → List of numerical feature names.
cat_features → List of categorical feature names.
target → The column to predict.
parameters → Hyperparameters for Logistic Regression.
catalog_name, schema_name → Database schema names for Databricks tables.
"""

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from loguru import logger
from mlflow import MlflowClient
from mlflow.data.dataset_source import DatasetSource
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from satisfaction_customer.config import ProjectConfig, Tags

RANDOM_SEED = 20230916


class BasicModel:
    """A basic model class for house price prediction using LightGBM.

    This class handles data loading, feature preparation, model training, and MLflow logging.
    """

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the model with project configuration.

        :param config: Project configuration object
        :param tags: Tags object
        :param spark: SparkSession object
        """
        self.config = config
        self.spark = spark

        # Extract settings from the config
        self.features = self.config.features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name_basic
        self.model_name = (
            f"{self.catalog_name}.{self.schema_name}.satisfaction_customer_model_basic"
        )
        self.tags = tags.dict()

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        Splits data into features (X_train, X_test) and target (y_train, y_test).
        """
        logger.info("Loading data from Databricks tables...")
        self.train_set_spark = self.spark.table(
            f"{self.catalog_name}.{self.schema_name}.train_set"
        )
        self.train_set = self.train_set_spark.toPandas()
        self.test_set = self.spark.table(
            f"{self.catalog_name}.{self.schema_name}.test_set"
        ).toPandas()
        self.data_version = "0"  # describe history -> retrieve

        self.X_train = self.train_set.drop(self.config.target, axis=1).select_dtypes(
            exclude=["datetime64"]
        )
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set.drop(self.config.target, axis=1).select_dtypes(
            exclude=["datetime64"]
        )
        self.y_test = self.test_set[self.target]
        logger.info("Data successfully loaded.")

    def prepare_features(self) -> None:
        """Encode categorical features and define a preprocessing pipeline.

        Creates a ColumnTransformer for one-hot encoding categorical features while passing through numerical
        features. Constructs a pipeline combining preprocessing and Logistic regression model.
        """
        logger.info("Defining preprocessing pipeline...")

        self.model = LogisticRegression(**self.config.parameters, random_state=RANDOM_SEED)

        logger.info("Preprocessing pipeline defined.")

    def train(self) -> None:
        """Train the model."""
        logger.info("Starting training...")
        self.model.fit(self.X_train, self.y_train)

    def log_model(self) -> None:
        """Log the logistic regression model using MLflow."""
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(tags={k: str(v) for k, v in self.tags.items()}) as run:
            self.run_id = run.info.run_id

            y_pred = self.model.predict(self.X_test)

            # Convert string labels to binary (1 for "satisfied", 0 otherwise)
            y_test_bin = [1 if y == "satisfied" else 0 for y in self.y_test]
            y_pred_bin = [1 if y == "satisfied" else 0 for y in y_pred]

            # Classification metrics
            accuracy = accuracy_score(y_test_bin, y_pred_bin)
            precision = precision_score(y_test_bin, y_pred_bin)
            recall = recall_score(y_test_bin, y_pred_bin)
            f1 = f1_score(y_test_bin, y_pred_bin)

            logger.info(f"Accuracy: {accuracy}")
            logger.info(f"Precision: {precision}")
            logger.info(f"Recall: {recall}")
            logger.info(f"F1 Score: {f1}")

            # Log parameters and metrics
            mlflow.log_param("model_type", "Logistic Regression with preprocessing")
            mlflow.log_params(self.parameters)
            mlflow.log_metrics(
                {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                }
            )

            # Log model
            signature = infer_signature(model_input=self.X_train, model_output=y_pred)
            dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                version=self.data_version,
            )
            mlflow.log_input(dataset, context="training")
            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="logregression-model",
                signature=signature,
            )

    def register_model(self) -> None:
        """Register model in Unity Catalog."""
        logger.info("Registering the model in UC...")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/logregression-model",
            name=self.model_name,
            tags=self.tags,
        )
        logger.info(f"Model registered as version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=self.model_name,
            alias="latest-model",
            version=latest_version,
        )

    def retrieve_current_run_dataset(self) -> DatasetSource:
        """Retrieve MLflow run dataset.

        :return: Loaded dataset source
        """
        run = mlflow.get_run(self.run_id)
        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)
        logger.info("Dataset source loaded.")
        return dataset_source.load()

    def retrieve_current_run_metadata(self) -> tuple[dict, dict]:
        """Retrieve MLflow run metadata.

        :return: Tuple containing metrics and parameters dictionaries
        """
        run = mlflow.get_run(self.run_id)
        metrics = run.data.to_dictionary()["metrics"]
        params = run.data.to_dictionary()["params"]
        logger.info("Dataset metadata loaded.")
        return metrics, params

    def load_latest_model_and_predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """Load the latest model from MLflow (alias=latest-model) and make predictions.

        Alias latest is not allowed -> we use latest-model instead as an alternative.

        :param input_data: Pandas DataFrame containing input features for prediction.
        :return: Pandas DataFrame with predictions.
        """
        logger.info("Loading model from MLflow alias 'production'...")

        model_uri = f"models:/{self.model_name}@latest-model"
        model = mlflow.sklearn.load_model(model_uri)

        logger.info("Model successfully loaded.")

        # Make predictions
        predictions = model.predict(input_data)

        # Return predictions as a DataFrame
        return predictions
