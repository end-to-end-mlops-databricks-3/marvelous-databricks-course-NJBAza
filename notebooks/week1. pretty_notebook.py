# Databricks notebook source
# % pip install -e ..
# %restart_python

# from pathlib import Path
# import sys
# sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------

import sys
print(sys.version)

# COMMAND ----------

# MAGIC %pip list

# COMMAND ----------

# MAGIC %pip install -e ..

# COMMAND ----------

import os

token = dbutils.secrets.get("my-secrets", "GIT_TOKEN")
url = f"git+https://{token}@github.com/end-to-end-mlops-databricks-3/marvelous@0.1.0"
%pip install $url

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from pathlib import Path
import sys
PACKAGE_ROOT = Path.cwd().parent
sys.path.append(str(PACKAGE_ROOT / "src"))
print("Current working directory:", Path.cwd())
print("Parent directory:", Path.cwd().parent)
print("Target src path added:", Path.cwd().parent / 'src')

print(PACKAGE_ROOT)
DATAPATH = PACKAGE_ROOT/"data"
print(DATAPATH)

# COMMAND ----------

import pandas as pd
import yaml
from satisfaction_customer.config import ProjectConfig
from satisfaction_customer.data_processor import DataProcessor
from satisfaction_customer.pipeline.pipeline import preprocess_pipeline
from loguru import logger
from marvelous.logging import setup_logging
from marvelous.timer import Timer
from pyspark.sql import SparkSession

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

setup_logging(log_file="logs/marvelous-1.log")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

# Load the satisfaction customer dataset
spark = SparkSession.builder.getOrCreate()

filepath = DATAPATH / "data.csv"

# Load the data
df = pd.read_csv(filepath)

# COMMAND ----------

df.head()

# COMMAND ----------

df.shape

# COMMAND ----------

# Creating the data processor object
with Timer() as preprocess_timer:
    # Initialize DataProcessor
    data_processor = DataProcessor(df, preprocess_pipeline, spark, config)

logger.info(f"Data preprocessing: {data_processor}")
print(f"The original data {data_processor}")

# COMMAND ----------

# Split the data
X_train, X_test = data_processor.split_data()
logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

# COMMAND ----------

X_train

# COMMAND ----------

# Show available catalogs
spark.sql("SHOW CATALOGS").show()

# Use your target catalog and schema
spark.sql("USE CATALOG mlops_dev")
spark.sql("SHOW SCHEMAS IN mlops_dev").show()

spark.sql("USE SCHEMA njavierb")

spark.sql("SELECT current_catalog(), current_schema()").show()

spark.sql("DROP TABLE IF EXISTS mlops_dev.njavierb.train_set")
spark.sql("DROP TABLE IF EXISTS mlops_dev.njavierb.test_set")

# COMMAND ----------

# Show available catalogs
spark.sql("SHOW CATALOGS").show()

# Use your target catalog and schema
spark.sql("USE CATALOG mlops_dev")
spark.sql("SHOW SCHEMAS IN mlops_dev").show()

spark.sql("USE SCHEMA njavierb")

spark.sql("SELECT current_catalog(), current_schema()").show()

spark.sql("DROP TABLE IF EXISTS mlops_dev.njavierb.train_set")
spark.sql("DROP TABLE IF EXISTS mlops_dev.njavierb.test_set")

# COMMAND ----------

spark.catalog.tableExists("mlops_dev.njavierb.train_set")
spark.sql("SHOW TABLES IN mlops_dev.njavierb").show()

# COMMAND ----------

# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)

# COMMAND ----------

# Enable change data feed (only once!)
logger.info("Enable change data feed")
data_processor.enable_change_data_feed()
