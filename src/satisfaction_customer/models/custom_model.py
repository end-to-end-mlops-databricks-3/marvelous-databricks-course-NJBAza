"""Custom model implementation.

infer_signature (from mlflow.models) → Captures input-output schema for model tracking.

num_features → List of numerical feature names.
cat_features → List of categorical feature names.
target → The column to predict.
parameters → Hyperparameters for LightGBM.
catalog_name, schema_name → Database schema names for Databricks tables.
"""

from typing import Literal

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from loguru import logger
from mlflow import MlflowClient
from mlflow.data.dataset_source import DatasetSource
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from satisfaction_customer.config import ProjectConfig, Tags
from satisfaction_customer.utils import adjust_predictions

RANDOM_SEED = 20230916


class SatisfactionCustomerModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper class for machine learning models to be used with MLflow.

    This class wraps a machine learning model for predicting satisfaction customer.
    """

    def __init__(self, model: object) -> None:
        """Initialize the SatisfactionCustomerModelWrapper.

        :param model: The underlying machine learning model.
        """
        self.model = model

    def predict(
        self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame | np.ndarray
    ) -> dict[str, float]:
        """Make predictions using the wrapped model.

        :param context: The MLflow context (unused in this implementation).
        :param model_input: Input data for making predictions.
        :return: A dictionary containing the adjusted prediction.
        """
        logger.info(f"model_input:{model_input}")
        predictions = self.model.predict(model_input)
        logger.info(f"predictions: {predictions}")
        # looks like {"Prediction": 10000.0}
        adjusted_predictions = adjust_predictions(predictions)
        logger.info(f"adjusted_predictions: {adjusted_predictions}")
        return adjusted_predictions


class CustomModel:
    """Custom model class for house price prediction.

    This class encapsulates the entire workflow of loading data, preparing features,
    training the model, and making predictions.
    """

    def __init__(
        self, config: ProjectConfig, tags: Tags, spark: SparkSession, code_paths: list[str]
    ) -> None:
        """Initialize the CustomModel.

        :param config: Configuration object containing model settings.
        :param tags: Tags for MLflow logging.
        :param spark: SparkSession object.
        :param code_paths: List of paths to additional code dependencies.
        """
        self.config = config
        self.spark = spark

        # Extract settings from the config
        self.features = self.config.features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name_custom
        self.tags = tags.dict()
        self.code_paths = code_paths

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        This method loads data from Databricks tables and splits it into features and target variables.
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
        """Prepare features for model training.

        This method sets up a preprocessing pipeline including one-hot encoding for categorical
        features and LightGBM regression model.
        """
        self.model = LogisticRegression(**self.config.parameters, random_state=RANDOM_SEED)

        logger.info("Preprocessing pipeline defined.")

    def train(self) -> None:
        """Train the model using the prepared pipeline."""
        logger.info("Starting training...")
        self.model.fit(self.X_train, self.y_train)

    def log_model(
        self, dataset_type: Literal["PandasDataset", "SparkDataset"] = "SparkDataset"
    ) -> None:
        """Log the trained model and its metrics to MLflow.

        This method evaluates the model, logs parameters and metrics, and saves the model in MLflow.
        """
        mlflow.set_experiment(self.experiment_name)
        additional_pip_deps = ["pyspark==3.5.0"]
        for package in self.code_paths:
            whl_name = package.split("/")[-1]
            additional_pip_deps.append(f"./code/{whl_name}")

        print("Tags being used:", self.tags)
        
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

            # Log the model
            signature = infer_signature(model_input=self.X_train, model_output=y_pred)

            if dataset_type == "PandasDataset":
                dataset = mlflow.data.from_pandas(
                    self.train_set,
                    name="train_set",
                )
            elif dataset_type == "SparkDataset":
                dataset = mlflow.data.from_spark(
                    self.train_set_spark,
                    table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                    version=self.data_version,
                )
            else:
                raise ValueError("Unsupported dataset type.")

            mlflow.log_input(dataset, context="training")

            conda_env = _mlflow_conda_env(additional_pip_deps=additional_pip_deps)

            mlflow.pyfunc.log_model(
                python_model=SatisfactionCustomerModelWrapper(self.model),
                artifact_path="pyfunc-satisfaction-customer-model",
                code_paths=self.code_paths,
                conda_env=conda_env,
                signature=signature,
                input_example=self.X_train.iloc[0:1],
            )

    def register_model(self) -> None:
        """Register the trained model in MLflow Model Registry.

        This method registers the model and sets an alias for the latest version.
        """
        logger.info("Registering the model in UC...")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/pyfunc-satisfaction-customer-model",
            name=f"{self.catalog_name}.{self.schema_name}.satisfaction_customer_model_custom",
            tags=self.tags,
        )
        logger.info(f"Model registered as version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.satisfaction_customer_model_custom",
            alias="latest-model",
            version=latest_version,
        )

    def retrieve_current_run_dataset(self) -> DatasetSource:
        """Retrieve the dataset used in the current MLflow run.

        :return: The loaded dataset source.
        """
        run = mlflow.get_run(self.run_id)
        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)
        return dataset_source.load()
        logger.info("Dataset source loaded.")

    def retrieve_current_run_metadata(self) -> tuple[dict, dict]:
        """Retrieve metadata from the current MLflow run.

        :return: A tuple containing metrics and parameters of the current run.
        """
        run = mlflow.get_run(self.run_id)
        metrics = run.data.to_dictionary()["metrics"]
        params = run.data.to_dictionary()["params"]
        return metrics, params
        logger.info("Dataset metadata loaded.")

    def load_latest_model_and_predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """Load the latest model (alias=latest-model) from MLflow and make predictions.

        Alias latest is not allowed -> we use latest-model instead as an alternative.

        :param input_data: Input data for prediction.
        :return: Predictions.

        Note:
        This also works
        model.unwrap_python_model().predict(None, input_data)
        check out this article:
        https://medium.com/towards-data-science/algorithm-agnostic-model-building-with-mlflow-b106a5a29535

        """
        logger.info("Loading model from MLflow alias 'production'...")

        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.satisfaction_customer_model_custom@latest-model"
        model = mlflow.pyfunc.load_model(model_uri)

        logger.info("Model successfully loaded.")

        # Make predictions: None is context
        predictions = model.predict(input_data)

        # Return predictions as a DataFrame
        return predictions
