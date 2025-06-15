# Databricks notebook source
import os
import mlflow
from satisfaction_customer import __version__ as satisfaction_customer_v
from satisfaction_customer.config import ProjectConfig, Tags
from satisfaction_customer.models.custom_model import CustomModel
from pyspark.sql import SparkSession

# COMMAND ----------

# Default profile:

profile = os.environ["PROFILE"]
mlflow.set_tracking_uri(f"databricks://{profile}")
mlflow.set_registry_uri(f"databricks-uc://{profile}")

config = ProjectConfig.from_yaml(config_path="../project_config.yml")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "week2"})

# COMMAND ----------

# Initialize model with the config path
custom_model = CustomModel(
    config=config,
    tags=tags,
    spark=spark,
    code_paths=[f"../dist/satisfaction_customer-{satisfaction_customer_v}-py3-none-any.whl"],
)

# COMMAND ----------

custom_model.load_data()
custom_model.prepare_features()

# COMMAND ----------

# Train + log the model (runs everything including MLflow logging)
custom_model.train()

# COMMAND ----------

custom_model.log_model()

# COMMAND ----------

run_id = mlflow.search_runs(experiment_names=["/Shared/satisfaction-customer-custom"]).run_id[0]

model = mlflow.pyfunc.load_model(f"runs:/{run_id}/pyfunc-satisfaction-customer-model")

# COMMAND ----------

# Retrieve dataset for the current run
custom_model.retrieve_current_run_dataset()

# COMMAND ----------

# Retrieve metadata for the current run
custom_model.retrieve_current_run_metadata()

# COMMAND ----------

# Register model
custom_model.register_model()

# COMMAND ----------

# Predict on the test set

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

X_test = test_set.drop(config.target).toPandas()

predictions_df = custom_model.load_latest_model_and_predict(X_test)
