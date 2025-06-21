# Databricks notebook source
# install dependencies
%pip install -e ..
token = dbutils.secrets.get("my-secrets", "GIT_TOKEN")
url = f"git+https://oauth:{token}@github.com/end-to-end-mlops-databricks-3/marvelous@0.1.0"
%pip install $url
#restart python
%restart_python

# COMMAND ----------

# MAGIC %pip list

# COMMAND ----------

# system path update, must be after %restart_python
# caution! This is not a great approach
from pathlib import Path
import sys
PACKAGE_ROOT = Path.cwd().parent
sys.path.append(str(PACKAGE_ROOT / "src"))

# COMMAND ----------

# A better approach (this file must be present in a notebook folder, achieved via synchronization)
#%pip install satisfaction_customer-1.0.1-py3-none-any.whl

# COMMAND ----------

from pyspark.sql import SparkSession
import mlflow

from satisfaction_customer.config import ProjectConfig
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from mlflow.models import infer_signature
from marvelous.common import is_databricks
from dotenv import load_dotenv
import os
from mlflow import MlflowClient
import pandas as pd
from satisfaction_customer import __version__
from satisfaction_customer.pipeline.pipeline import (preprocess_pipeline, pretrain_pipeline)
from satisfaction_customer.processing import preprocessing as pp
from satisfaction_customer.settings import settings
from mlflow.utils.environment import _mlflow_conda_env
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from pyspark.errors import AnalysisException
from pyspark.sql.functions import col
import numpy as np
from datetime import datetime
import boto3


# COMMAND ----------

if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")


config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
fe = feature_engineering.FeatureEngineeringClient()

train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set")
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set")

# COMMAND ----------

# create feature table with information about houses

feature_table_name = f"{config.catalog_name}.{config.schema_name}.satisfaction_customer_demo"
lookup_features = ["flight_distance", "arrival_delay_in_minutes", "type_of_travel", "online_boarding"]


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Option 1: feature engineering client
# MAGIC feature_table = fe.create_table(
# MAGIC    name=feature_table_name,
# MAGIC    primary_keys=["id"],
# MAGIC    df=train_set[["id"]+lookup_features],
# MAGIC    description="Satisfaction features table",
# MAGIC )
# MAGIC
# MAGIC spark.sql(f"ALTER TABLE {feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
# MAGIC
# MAGIC fe.write_table(
# MAGIC    name=feature_table_name,
# MAGIC    df=test_set[["id"]+lookup_features],
# MAGIC    mode="merge",
# MAGIC )
# MAGIC
# MAGIC ***I DO NOT USE THIS OPTION BECAUSE IT IS PROBLEMATIC FOR CHANGES OF DATA. IT IS BETTER TO DELETE PREVIOUS DATA AS IN THE SQL CASE***

# COMMAND ----------

# create feature table with information about customers
# Option 2: SQL

spark.sql(f"""
          CREATE OR REPLACE TABLE {feature_table_name}
          (id STRING NOT NULL, flight_distance INT, arrival_delay_in_minutes INT, type_of_travel STRING, online_boarding INT);
          """)
# primary key on Databricks is not enforced!
try:
    spark.sql(f"ALTER TABLE {feature_table_name} ADD CONSTRAINT satisfaction_pk_demo PRIMARY KEY(id);")
except AnalysisException:
    pass
spark.sql(f"ALTER TABLE {feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
spark.sql(f"""
          INSERT INTO {feature_table_name}
          SELECT id, flight_distance, arrival_delay_in_minutes, type_of_travel, online_boarding
          FROM {config.catalog_name}.{config.schema_name}.train_set
          """)
spark.sql(f"""
          INSERT INTO {feature_table_name}
          SELECT id, flight_distance, arrival_delay_in_minutes, type_of_travel, online_boarding
          FROM {config.catalog_name}.{config.schema_name}.test_set
          """)

# COMMAND ----------

# create feature function
# docs: https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-syntax-ddl-create-sql-function

# problems with feature functions:
# functions are not versioned 
# functions may behave differently depending on the runtime (and version of packages and python)
# there is no way to enforce python version & package versions for the function 
# this is only supported from runtime 17
# advised to use only for simple calculations

function_name = f"{config.catalog_name}.{config.schema_name}.calculate_delay_ratio_demo"

# COMMAND ----------

spark.sql(f"""
        CREATE OR REPLACE FUNCTION mlops_dev.njavierb.calculate_delay_ratio_demo(departure_delay DOUBLE, arrival_delay DOUBLE)
        RETURNS DOUBLE
        LANGUAGE PYTHON
        AS $$
        def calculate_delay_ratio_demo(departure_delay, arrival_delay):
            if arrival_delay is None:
                arrival_delay = 0.0
            if departure_delay is None:
                departure_delay = 0.0
            return float(departure_delay) / (float(arrival_delay) + 1.0)
        return calculate_delay_ratio_demo(departure_delay, arrival_delay)
        $$
""")

# COMMAND ----------

# MAGIC %md
# MAGIC # it is possible to define simple functions in sql only without python
# MAGIC # Option 2
# MAGIC spark.sql(f"""
# MAGIC         CREATE OR REPLACE FUNCTION {function_name}_sql(
# MAGIC         departure_delay_in_minutes BIGINT,
# MAGIC         arrival_delay_in_minutes BIGINT
# MAGIC         )
# MAGIC         RETURNS DOUBLE
# MAGIC         RETURN ROUND(
# MAGIC         departure_delay_in_minutes / (arrival_delay_in_minutes + 1), 2
# MAGIC         )
# MAGIC """)

# COMMAND ----------

spark.sql("USE CATALOG mlops_dev")
spark.sql("USE SCHEMA njavierb")

# COMMAND ----------

spark.sql("SHOW USER FUNCTIONS").show(truncate=False)

# COMMAND ----------

# execute function
spark.sql("SELECT mlops_dev.njavierb.calculate_delay_ratio_demo(10, 5) AS delay_ratio").show()

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType

train_set = (
    train_set
    .withColumn("departure_delay_in_minutes", col("departure_delay_in_minutes").cast(DoubleType()))
    .withColumn("arrival_delay_in_minutes", col("arrival_delay_in_minutes").cast(DoubleType()))
)

train_set.select("departure_delay_in_minutes", "arrival_delay_in_minutes").printSchema()


# COMMAND ----------

training_set = fe.create_training_set(
    df=train_set.drop("flight_distance", "type_of_travel", "online_boarding"),
    label=config.target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["flight_distance", "type_of_travel", "online_boarding"],  # removed delay columns
            lookup_key="id",
        ),
        FeatureFunction(
            udf_name="mlops_dev.njavierb.calculate_delay_ratio_demo",
            output_name="delay_ratio",
            input_bindings={
                "departure_delay": "departure_delay_in_minutes",
                "arrival_delay": "arrival_delay_in_minutes",
            },
        ),
    ],
    exclude_columns=["update_timestamp_utc"],
)


# COMMAND ----------

training_df = training_set.load_df().toPandas()
display(training_df)

# COMMAND ----------

# Train & register a model
training_df = training_set.load_df().toPandas()
X_train = training_df[config.features + ["delay_ratio"]]
y_train = training_df[config.target]

# COMMAND ----------

config.features + ["delay_ratio"]

# COMMAND ----------

pretrain_pipeline_test = Pipeline(
    [
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
                columns=settings.NUMERICAL_FEATURES_3
                + settings.CATEGORICAL_FEATURES
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
            pp.OneHotEncoderProcessor(columns=["gender"], prefix="gender"),
        ),
        (
            "OneHotEncoderProcessor1",
            pp.OneHotEncoderProcessor(columns=["customer_type"], prefix="customer_type"),
        ),
        (
            "OneHotEncoderProcessor2",
            pp.OneHotEncoderProcessor(columns=["type_of_travel"], prefix="type_of_travel"),
        ),
        (
            "OneHotEncoderProcessor3",
            pp.OneHotEncoderProcessor(columns=settings.FEATURES_ONE_HOT, prefix="class"),
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

full_preprocessing = Pipeline(steps=preprocess_pipeline.steps + 
                                    pretrain_pipeline.steps)

# COMMAND ----------

import os
import pickle

DATAPATH = PACKAGE_ROOT/"data"
RANDOM_SEED = 20230916

DATAPATH


# COMMAND ----------

df_transformed = full_preprocessing.fit_transform(X_train)

# COMMAND ----------

df_transformed.head()

# COMMAND ----------

pipeline = Pipeline(
    steps=[
        ("preprocessor", full_preprocessing),
        ("classifier", LogisticRegression(**config.parameters, random_state=RANDOM_SEED)),
    ]
)

pipeline.fit(X_train, y_train)

# COMMAND ----------

mlflow.set_experiment("/Shared/demo-model-fe")
with mlflow.start_run(run_name="demo-run-model-fe",
                      tags={"git_sha": "1234567890abcd",
                            "branch": "week2"},
                            description="demo run for FE model logging") as run:
    # Log parameters and metrics
    run_id = run.info.run_id
    mlflow.log_param("model_type", "Logistic with preprocessing")
    mlflow.log_params(config.parameters)

    # Log the model
    signature = infer_signature(model_input=X_train, model_output=pipeline.predict(X_train))
    fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path="logistic-pipeline-model-fe",
                training_set=training_set,
                signature=signature,
            )
    

# COMMAND ----------

model_name = f"{config.catalog_name}.{config.schema_name}.model_fe_demo"
model_version = mlflow.register_model(
    model_uri=f'runs:/{run_id}/logistic-pipeline-model-fe',
    name=model_name,
    tags={"git_sha": "1234567890abcd"})

# COMMAND ----------

# make predictions
features = [f for f in ["Id"] + config.num_features + config.cat_features if f not in lookup_features]
predictions = fe.score_batch(
    model_uri=f"models:/{model_name}/{model_version.version}",
    df=test_set[features]
)

# COMMAND ----------

predictions.select("prediction").show(5)

# COMMAND ----------

from pyspark.sql.functions import col

features = [f for f in ["Id"] + config.num_features + config.cat_features if f not in lookup_features]
test_set_with_new_id = test_set.select(*features).withColumn(
    "Id",
    (col("Id").cast("long") + 1000000).cast("string")
)

predictions = fe.score_batch(
    model_uri=f"models:/{model_name}/{model_version.version}",
    df=test_set_with_new_id 
)

# COMMAND ----------

# make predictions for a non-existing entry -> error!
predictions.select("prediction").show(5)

# COMMAND ----------

overallqual_function = f"{config.catalog_name}.{config.schema_name}.replace_overallqual_missing"
spark.sql(f"""
        CREATE OR REPLACE FUNCTION {overallqual_function}(OverallQual INT)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        if OverallQual is None:
            return 5
        else:
            return OverallQual
        $$
        """)

grlivarea_function = f"{config.catalog_name}.{config.schema_name}.replace_grlivarea_missing"
spark.sql(f"""
        CREATE OR REPLACE FUNCTION {grlivarea_function}(GrLivArea INT)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        if GrLivArea is None:
            return 1000
        else:
            return GrLivArea
        $$
        """)

garagecars_function = f"{config.catalog_name}.{config.schema_name}.replace_garagecars_missing"
spark.sql(f"""
        CREATE OR REPLACE FUNCTION {garagecars_function}(GarageCars INT)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        if GarageCars is None:
            return 2
        else:
            return GarageCars
        $$
        """)

# COMMAND ----------

# what if we want to replace with a default value if entry is not found
# what if we want to look up value in another table? the logics get complex
# problems that arize: functions/ lookups always get executed (if statememt is not possible)
# it can get slow...

# step 1: create 3 feature functions

# step 2: redefine create training set

# try again

# create a training set
training_set = fe.create_training_set(
    df=train_set.drop("OverallQual", "GrLivArea", "GarageCars"),
    label=config.target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["OverallQual", "GrLivArea", "GarageCars"],
            lookup_key="Id",
            rename_outputs={"OverallQual": "lookup_OverallQual",
                            "GrLivArea": "lookup_GrLivArea",
                            "GarageCars": "lookup_GarageCars"}
                ),
        FeatureFunction(
            udf_name=overallqual_function,
            output_name="OverallQual",
            input_bindings={"OverallQual": "lookup_OverallQual"},
            ),
        FeatureFunction(
            udf_name=grlivarea_function,
            output_name="GrLivArea",
            input_bindings={"GrLivArea": "lookup_GrLivArea"},
        ),
        FeatureFunction(
            udf_name=garagecars_function,
            output_name="GarageCars",
            input_bindings={"GarageCars": "lookup_GarageCars"},
        ),
        FeatureFunction(
            udf_name=function_name,
            output_name="house_age",
            input_bindings={"year_built": "YearBuilt"},
            ),
    ],
    exclude_columns=["update_timestamp_utc"],
    )

# COMMAND ----------

# Train & register a model
training_df = training_set.load_df().toPandas()
X_train = training_df[config.num_features + config.cat_features + ["house_age"]]
y_train = training_df[config.target]

#pipeline
pipeline = Pipeline(
        steps=[("preprocessor", ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"),
                           config.cat_features)],
            remainder="passthrough")
            ),
               ("regressor", LGBMRegressor(**config.parameters))]
        )

pipeline.fit(X_train, y_train)

# COMMAND ----------

mlflow.set_experiment("/Shared/demo-model-fe")
with mlflow.start_run(run_name="demo-run-model-fe",
                      tags={"git_sha": "1234567890abcd",
                            "branch": "week2"},
                            description="demo run for FE model logging") as run:
    # Log parameters and metrics
    run_id = run.info.run_id
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(config.parameters)

    # Log the model
    signature = infer_signature(model_input=X_train, model_output=pipeline.predict(X_train))
    fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path="lightgbm-pipeline-model-fe",
                training_set=training_set,
                signature=signature,
            )
model_name = f"{config.catalog_name}.{config.schema_name}.model_fe_demo"
model_version = mlflow.register_model(
    model_uri=f'runs:/{run_id}/lightgbm-pipeline-model-fe',
    name=model_name,
    tags={"git_sha": "1234567890abcd"})

# COMMAND ----------

from pyspark.sql.functions import col

features = [f for f in ["Id"] + config.num_features + config.cat_features if f not in lookup_features]
test_set_with_new_id = test_set.select(*features).withColumn(
    "Id",
    (col("Id").cast("long") + 1000000).cast("string")
)

predictions = fe.score_batch(
    model_uri=f"models:/{model_name}/{model_version.version}",
    df=test_set_with_new_id 
)

# COMMAND ----------

# make predictions for a non-existing entry -> no error!
predictions.select("prediction").show(5)

# COMMAND ----------

import boto3

region_name = "eu-west-1"
aws_access_key_id = os.environ["aws_access_key_id"]
aws_secret_access_key = os.environ["aws_secret_access_key"]

client = boto3.client(
    'dynamodb',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

# COMMAND ----------

response = client.create_table(
    TableName='HouseFeatures',
    KeySchema=[
        {
            'AttributeName': 'Id',
            'KeyType': 'HASH'  # Partition key
        }
    ],
    AttributeDefinitions=[
        {
            'AttributeName': 'Id',
            'AttributeType': 'S'  # String
        }
    ],
    ProvisionedThroughput={
        'ReadCapacityUnits': 5,
        'WriteCapacityUnits': 5
    }
)

print("Table creation initiated:", response['TableDescription']['TableName'])

# COMMAND ----------

client.put_item(
    TableName='HouseFeatures',
    Item={
        'Id': {'S': 'house_001'},
        'OverallQual': {'N': '8'},
        'GrLivArea': {'N': '2450'},
        'GarageCars': {'N': '2'}
    }
)

# COMMAND ----------

response = client.get_item(
    TableName='HouseFeatures',
    Key={
        'Id': {'S': 'house_001'}
    }
)

# Extract the item from the response
item = response.get('Item')
print(item)

# COMMAND ----------



# COMMAND ----------

from itertools import islice

rows = spark.table(feature_table_name).toPandas().to_dict(orient="records")

def to_dynamodb_item(row):
    return {
        'PutRequest': {
            'Item': {
                'Id': {'S': str(row['Id'])},
                'OverallQual': {'N': str(row['OverallQual'])},
                'GrLivArea': {'N': str(row['GrLivArea'])},
                'GarageCars': {'N': str(row['GarageCars'])}
            }
        }
    }

items = [to_dynamodb_item(row) for row in rows]

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

for batch in chunks(items, 25):
    response = client.batch_write_item(
        RequestItems={
            'HouseFeatures': batch
        }
    )
    # Handle any unprocessed items if needed
    unprocessed = response.get('UnprocessedItems', {})
    if unprocessed:
        print("Warning: Some items were not processed. Retry logic needed.")

# COMMAND ----------

# We ran into more limitations when we tried complex data types as output of a feature function
# and then tried to use it for serving
# al alternatve solution: using an external database (we use DynamoDB here)

# create a DynamoDB table
# insert records into dynamo DB & read from dynamoDB

# create a pyfunc model

# COMMAND ----------


class HousePriceModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper class for machine learning models to be used with MLflow.

    This class wraps a machine learning model for predicting house prices.
    """

    def __init__(self, model: object) -> None:
        """Initialize the HousePriceModelWrapper.

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
        client = boto3.client('dynamodb',
                                   aws_access_key_id=os.environ["aws_access_key_id"],
                                   aws_secret_access_key=os.environ["aws_secret_access_key"],
                                   region_name=os.environ["region_name"])
        
        parsed = []
        for lookup_id in model_input["Id"]:
            raw_item = client.get_item(
                TableName='HouseFeatures',
                Key={'Id': {'S': lookup_id}})["Item"]     
            parsed_dict = {key: int(value['N']) if 'N' in value else value['S']
                      for key, value in raw_item.items()}
            parsed.append(parsed_dict)
        lookup_df=pd.DataFrame(parsed)
        merged_df = model_input.merge(lookup_df, on="Id", how="left").drop("Id", axis=1)
        
        merged_df["GarageCars"] = merged_df["GarageCars"].fillna(2)
        merged_df["GrLivArea"] = merged_df["GrLivArea"].fillna(1000)
        merged_df["OverallQual"] = merged_df["OverallQual"].fillna(5)
        merged_df["house_age"] = datetime.now().year - merged_df["YearBuilt"]
        predictions = self.model.predict(merged_df)

        return [int(x) for x in predictions]

# COMMAND ----------

custom_model = HousePriceModelWrapper(pipeline)

# COMMAND ----------

features = [f for f in ["Id"] + config.num_features + config.cat_features if f not in lookup_features]
data = test_set.select(*features).toPandas()
data

# COMMAND ----------

custom_model.predict(context=None, model_input=data)

# COMMAND ----------

#log model
mlflow.set_experiment("/Shared/demo-model-fe-pyfunc")
with mlflow.start_run(run_name="demo-run-model-fe-pyfunc",
                      tags={"git_sha": "1234567890abcd",
                            "branch": "week2"},
                            description="demo run for FE model logging") as run:
    # Log parameters and metrics
    run_id = run.info.run_id
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(config.parameters)

    # Log the model
    signature = infer_signature(model_input=data, model_output=custom_model.predict(context=None, model_input=data))
    mlflow.pyfunc.log_model(
                python_model=custom_model,
                artifact_path="lightgbm-pipeline-model-fe",
                signature=signature,
            )
    

# COMMAND ----------

# predict
mlflow.models.predict(f"runs:/{run_id}/lightgbm-pipeline-model-fe", data[0:1])
