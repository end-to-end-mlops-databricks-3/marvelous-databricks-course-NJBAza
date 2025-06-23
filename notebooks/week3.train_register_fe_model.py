# Databricks notebook source
# MAGIC %md
# MAGIC install satisfaction_customer-1.0.1-py3-none-any.whl

# COMMAND ----------

import sys
print(sys.version)

# COMMAND ----------

# install dependencies
%pip install -e ..
token = dbutils.secrets.get("my-secrets", "GIT_TOKEN")
url = f"git+https://oauth:{token}@github.com/end-to-end-mlops-databricks-3/marvelous@0.1.0"
%pip install $url
#restart python
%restart_python

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# system path update, must be after %restart_python
# caution! This is not a great approach
from pathlib import Path
import sys
PACKAGE_ROOT = Path.cwd().parent
sys.path.append(str(PACKAGE_ROOT / "src"))

# COMMAND ----------

# Configure tracking uri
import mlflow
from loguru import logger
from pyspark.sql import SparkSession

from satisfaction_customer.config import ProjectConfig, Tags
from satisfaction_customer.models.feature_lookup_model import FeatureLookUpModel

# Configure tracking uri
# mlflow.set_tracking_uri("databricks")
# mlflow.set_registry_uri("databricks-uc")

spark = SparkSession.builder.getOrCreate()
tags_dict = {"git_sha": "abcd12345", "branch": "week2"}
tags = Tags(**tags_dict)

config = ProjectConfig.from_yaml(config_path="../project_config.yml")


# COMMAND ----------

# Initialize model
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# COMMAND ----------

# Create feature table
fe_model.create_feature_table()

# COMMAND ----------

# Define is_loyal_and_business_travel feature function
fe_model.define_feature_function()

# COMMAND ----------

# Load data
fe_model.load_data()

# COMMAND ----------

# Perform feature engineering
fe_model.feature_engineering()

# COMMAND ----------

# Train the model
fe_model.train()  

# COMMAND ----------

# Train the model
fe_model.register_model()

# COMMAND ----------

# Lets run prediction on the last production model
# Load test set from Delta table
spark = SparkSession.builder.getOrCreate()

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

# Drop feature lookup columns and target
X_test = test_set.drop("flight_distance", "inflight_wifi_service", "online_boarding", config.target)


# COMMAND ----------

display(X_test)

# COMMAND ----------

fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# Make predictions
predictions = fe_model.load_latest_model_and_predict(X_test)

# Display predictions
logger.info(predictions)

# COMMAND ----------

predictions.head(5)

# COMMAND ----------


