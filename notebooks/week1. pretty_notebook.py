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
sys.path.append(str(Path.cwd().parent / 'src'))

print("Current working directory:", Path.cwd())
print("Parent directory:", Path.cwd().parent)
print("Target src path added:", Path.cwd().parent / 'src')

# COMMAND ----------

import pandas as pd
import yaml
from satisfaction_customer.config import ProjectConfig
from satisfaction_customer.data_processor import DataProcessor
from loguru import logger
from marvelous.logging import setup_logging
from marvelous.timer import Timer
from pyspark.sql import SparkSession

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

setup_logging(log_file="logs/marvelous-1.log")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

# Load the house prices dataset
spark = SparkSession.builder.getOrCreate()

filepath = "../data/data.csv"

# Load the data
df = pd.read_csv(filepath)


# COMMAND ----------

# Load the house prices dataset
with Timer() as preprocess_timer:
    # Initialize DataProcessor
    data_processor = DataProcessor(df, config, spark)

    # Preprocess the data
    data_processor.preprocess()

logger.info(f"Data preprocessing: {preprocess_timer}")

# COMMAND ----------

# Split the data
X_train, X_test = data_processor.split_data()
logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

# COMMAND ----------

# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)

# Enable change data feed (only once!)
logger.info("Enable change data feed")
data_processor.enable_change_data_feed()
