"""FeatureLookUp model implementation."""

from datetime import datetime

import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from pyspark.sql.functions import when, col
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from satisfaction_customer.config import ProjectConfig, Tags
from satisfaction_customer.pipeline.pipeline import preprocess_pipeline, pretrain_pipeline

RANDOM_SEED = 20230916

class FeatureLookUpModel:
    """A class to manage FeatureLookupModel."""

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the model with project configuration."""
        self.config = config
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient()

        # Extract settings from the config
        self.features = self.config.features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name

        # Define table names and function name
        self.feature_table_name = f"{self.catalog_name}.{self.schema_name}.satisfaction_features"
        self.function_name = f"{self.catalog_name}.{self.schema_name}.calculate_delay_rate"

        # MLflow configuration
        self.experiment_name = self.config.experiment_name_fe
        self.tags = tags.dict()

    def create_feature_table(self) -> None:
        """Create or update the satisfaction_features table and populate it.

        This table stores features related to satisfactions.
        """
        self.spark.sql(
            f"""
        CREATE OR REPLACE TABLE {self.feature_table_name}
        (id STRING NOT NULL, flight_distance INT, inflight_wifi_service INT, online_boarding INT);
        """
        )
        self.spark.sql(
            f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT satisfaction_pk PRIMARY KEY(id);"
        )
        self.spark.sql(
            f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT id, flight_distance, inflight_wifi_service, online_boarding FROM {self.catalog_name}.{self.schema_name}.train_set"
        )
        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT id, flight_distance, inflight_wifi_service, online_boarding FROM {self.catalog_name}.{self.schema_name}.test_set"
        )
        logger.info("Feature table created and populated.")

    def define_feature_function(self) -> None:
        """Define a function to calculate the satisfaction's age.

        This function subtracts the year built from the current year.
        """
        self.spark.sql(
            f"""
            CREATE OR REPLACE FUNCTION {self.function_name}(customer_type STRING, type_of_travel STRING)
            RETURNS DOUBLE
            LANGUAGE PYTHON
            AS $$
            def is_loyal_and_business_travel_demo(customer_type, type_of_travel):
                if customer_type is None or type_of_travel is None:
                    return 0.0
                if customer_type.strip().lower() == "loyal customer" and type_of_travel.strip().lower() == "business travel":
                    return 1.0
                return 0.0
            return is_loyal_and_business_travel_demo(customer_type, type_of_travel)
            $$
        """
        )
        logger.info("Feature function defined.")

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        Drops specified columns and casts 'YearBuilt' to integer type.
        """
        self.train_set = self.spark.table(
            f"{self.catalog_name}.{self.schema_name}.train_set"
        ).drop("flight_distance", "inflight_wifi_service", "online_boarding")
        self.test_set = self.spark.table(
            f"{self.catalog_name}.{self.schema_name}.test_set"
        ).toPandas()

        self.train_set = self.train_set.withColumn(
            "customer_type", col("customer_type").cast("string")
        )
        self.train_set = self.train_set.withColumn(
            "type_of_travel", col("type_of_travel").cast("string")
        )

        self.train_set = self.train_set.withColumn("id", self.train_set["id"].cast("string"))

        logger.info("Data successfully loaded.")

    def feature_engineering(self) -> None:
        """Perform feature engineering by linking data with feature tables.

        Creates a training set using FeatureLookup and FeatureFunction.
        """
        self.training_set = self.fe.create_training_set(
            df=self.train_set.drop("flight_distance", "inflight_wifi_service", "online_boarding"),
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_table_name,
                    feature_names=["flight_distance", "inflight_wifi_service", "online_boarding"],
                    lookup_key="id",
                ),
                FeatureFunction(
                    udf_name=self.function_name,
                    output_name="is_loyal_biz",
                    input_bindings={
                        "customer_type": "customer_type",
                        "type_of_travel": "type_of_travel",
                    },
                ),
            ],
            exclude_columns=["update_timestamp_utc"],
        )

        self.training_df = self.training_set.load_df().toPandas()
        self.test_set["is_loyal_biz"] = (
            ((self.test_set["customer_type"] == "Loyal Customer") & 
            (self.test_set["type_of_travel"] == "Business travel"))
            .astype(int)
        )


        self.X_train = self.training_df[self.features + ["is_loyal_biz"]]
        print(self.X_train.columns)
        self.y_train = self.training_df[self.target]
        self.X_test = self.test_set[self.features + ["is_loyal_biz"]]
        print(self.X_test.columns)
        self.y_test = self.test_set[self.target]

        logger.info("Feature engineering completed.")

    def train(self) -> None:
        """Train the model and log results to MLflow.

        Uses a pipeline with preprocessing and LightGBM regressor.
        """
        logger.info("Starting training...")

        full_preprocessing = Pipeline(steps=preprocess_pipeline.steps + pretrain_pipeline.steps)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", full_preprocessing),
                (
                    "classifier",
                    LogisticRegression(**self.config.parameters, random_state=RANDOM_SEED),
                ),
            ]
        )

        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(tags={k: str(v) for k, v in self.tags.items()}) as run:
            self.run_id = run.info.run_id
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)

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

            signature = infer_signature(self.X_train, y_pred)

            self.fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path="logregression-pipeline-model-fe",
                training_set=self.training_set,
                signature=signature,
            )

    def register_model(self) -> str:
        """Register the trained model to MLflow registry.

        Registers the model and sets alias to 'latest-model'.
        """
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/logregression-pipeline-model-fe",
            name=f"{self.catalog_name}.{self.schema_name}.satisfaction_customer_model_fe",
            tags=self.tags,
        )

        # Fetch the latest version dynamically
        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.satisfaction_customer_model_fe",
            alias="latest-model",
            version=latest_version,
        )

        return latest_version

    def load_latest_model_and_predict(self, X: DataFrame) -> DataFrame:
        """Load the trained model from MLflow using Feature Engineering Client and make predictions.

        Loads the model with the alias 'latest-model' and scores the batch.
        :param X: DataFrame containing the input features.
        :return: DataFrame containing the predictions.
        """
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.satisfaction_customer_model_fe@latest-model"

        predictions = self.fe.score_batch(model_uri=model_uri, df=X)
        return predictions