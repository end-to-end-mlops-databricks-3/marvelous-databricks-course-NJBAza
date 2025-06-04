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
from satisfaction_customer.pipeline.pipeline import transform_pipeline
from satisfaction_customer.processing import preprocessing as pp
from satisfaction_customer.settings import settings
from sklearn.pipeline import Pipeline
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

pipeline_features = Pipeline(
    [
        (
            "DataFrameTypeConverter",
            pp.DataFrameTypeConverter(conversion_dict=settings.CONVERSION_DICT),
        ),
        (
            "DropColumns",
            pp.DropColumns(variables_to_drop=settings.VARIABLES_TO_DROP),
        ),
        (
            "DropDuplicatesTransformer",
            pp.DropDuplicatesTransformer(),
        ),
        (
            "ModeImputer",
            pp.ModeImputer(variables=settings.CATEGORICAL_FEATURES),
        ),
        (
            "MedianImputer",
            pp.MedianImputer(variables=settings.NUMERICAL_FEATURES0),
        ),
        (
            "FeatureCreator",
            pp.FeatureCreator(),
        ),
        (
            "Winsorizer",
            pp.Winsorizer(numerical_features=settings.NUMERICAL_FEATURES2, limits=[0.025, 0.025]),
        ),
        (
            "Chosen",
            pp.ChosenFeatures(
                columns=settings.NUMERICAL_FEATURES_3 + settings.CATEGORICAL_FEATURES + [settings.TARGET]
            ),
        ),
        (
            "LogTransforms",
            pp.LogTransforms(),
        ),
        (
            "DataScaler",
            pp.DataScaler(),
        ),
        (
            "CorrelationMatrixProcessor",
            pp.CorrelationMatrixProcessor(threshold=0.8),
        ),
        (
            "FeatureVariance",
            pp.FeatureVariance(threshold=0.001),
        ),
        (
            "OneHotEncoderProcessor",
            pp.OneHotEncoderProcessor(columns=["Gender"], prefix="Gender"),
        ),
        (
            "OneHotEncoderProcessor1",
            pp.OneHotEncoderProcessor(columns=["Customer Type"], prefix="Customer Type"),
        ),
        (
            "OneHotEncoderProcessor2",
            pp.OneHotEncoderProcessor(columns=["Type of Travel"], prefix="Type of Travel"),
        ),
        (
            "OneHotEncoderProcessor3",
            pp.OneHotEncoderProcessor(columns=settings.FEATURES_ONE_HOT, prefix="Class"),
        ),
        (
            "DataFrameTypeConverter2",
            pp.DataFrameTypeConverter(conversion_dict=settings.CONVERSION_DICT2),
        ),
        (
            "DropColumns2",
            pp.DropColumns(variables_to_drop=settings.TO_DROP),
        ),
    ]
)

# COMMAND ----------

transformed_df = transform_pipeline.fit_transform(df)
transformed_df.head()

# COMMAND ----------

# Creating the data processor object
with Timer() as preprocess_timer:
    # Initialize DataProcessor
    data_processor = DataProcessor(df, transform_pipeline, config, spark)

logger.info(f"Data preprocessing: {data_processor}")
print(f"The original data {data_processor}")

# COMMAND ----------

data_processor.transform()

# COMMAND ----------

# Split the data
X_train, X_test = data_processor.split_data()
logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

# COMMAND ----------

X_train

# COMMAND ----------

# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)

# Enable change data feed (only once!)
logger.info("Enable change data feed")
data_processor.enable_change_data_feed()
