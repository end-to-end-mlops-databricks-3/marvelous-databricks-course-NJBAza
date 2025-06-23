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
lookup_features = ["flight_distance", "inflight_wifi_service", "online_boarding"]


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
          (id STRING NOT NULL, flight_distance INT, inflight_wifi_service INT, online_boarding INT);
          """)
# primary key on Databricks is not enforced!
try:
    spark.sql(f"ALTER TABLE {feature_table_name} ADD CONSTRAINT satisfaction_pk_demo PRIMARY KEY(id);")
except AnalysisException:
    pass
spark.sql(f"ALTER TABLE {feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
spark.sql(f"""
          INSERT INTO {feature_table_name}
          SELECT id, flight_distance, inflight_wifi_service, online_boarding
          FROM {config.catalog_name}.{config.schema_name}.train_set
          """)
spark.sql(f"""
          INSERT INTO {feature_table_name}
          SELECT id, flight_distance, inflight_wifi_service, online_boarding
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

function_name = f"{config.catalog_name}.{config.schema_name}.is_loyal_and_business_travel_demo"

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE FUNCTION mlops_dev.njavierb.is_loyal_and_business_travel_demo(customer_type STRING, type_of_travel STRING)
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

spark.sql("""
    SELECT mlops_dev.njavierb.is_loyal_and_business_travel_demo('Loyal Customer', 'Business travel') AS is_loyal_biz
""").show()


# COMMAND ----------

training_set = fe.create_training_set(
    df=train_set.drop("flight_distance", "inflight_wifi_service", "online_boarding"),
    label=config.target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["flight_distance", "inflight_wifi_service", "online_boarding"], 
            lookup_key="id",
        ),
        FeatureFunction(
            udf_name=function_name,
            output_name="is_loyal_biz",
            input_bindings={
                "customer_type": "customer_type",
                "type_of_travel": "type_of_travel",
            },
        ),
    ],
    exclude_columns=["update_timestamp_utc"],
)

# COMMAND ----------

# Train & register a model
training_df = training_set.load_df().toPandas()
X_train = training_df[config.features + ["is_loyal_biz"]]
y_train = training_df[config.target]

# COMMAND ----------

X_train.head()

# COMMAND ----------

full_preprocessing = Pipeline(steps=preprocess_pipeline.steps + 
                                    pretrain_pipeline.steps)

# COMMAND ----------

df_transformed = full_preprocessing.fit_transform(X_train)

# COMMAND ----------

df_transformed.columns

# COMMAND ----------

RANDOM_SEED = 20230916

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

lookup_features

# COMMAND ----------

features = [f for f in config.features if f not in lookup_features]
# Step 1: Convert Spark DataFrame to Pandas
test_set_pdf = test_set.select(*config.features).toPandas()

X_array = full_preprocessing.transform(test_set_pdf)
X_sdf = spark.createDataFrame(X_array)
X_sdf

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.functions import lower, regexp_replace

features = [f for f in config.features if f not in lookup_features]

columns_to_fix = [
    "flight_distance",
    "inflight_wifi_service",
    "online_boarding"
]

for col_name in columns_to_fix:
    test_set = test_set.withColumn(col_name, col(col_name).cast("int"))

for c in ["gender", "customer_type", "type_of_travel", "class"]:
    test_set = test_set.withColumn(c, regexp_replace(lower(col(c)), " ", "_"))

predictions = fe.score_batch(
    model_uri=f"models:/{model_name}/{model_version.version}",
    df=test_set[features]
)

# COMMAND ----------

predictions

# COMMAND ----------

predictions.select("prediction").show(5)

# COMMAND ----------

from pyspark.sql.functions import col

features = [f for f in config.features if f not in lookup_features]
test_set_with_new_id = test_set.select(*features).withColumn(
    "id",
    (col("id").cast("long") + 1000000).cast("string")
)

predictions = fe.score_batch(
    model_uri=f"models:/{model_name}/{model_version.version}",
    df=test_set_with_new_id 
)

# COMMAND ----------

# make predictions for a non-existing entry -> error!
predictions.select("prediction").show(5)

# COMMAND ----------

flightdistance_function = f"{config.catalog_name}.{config.schema_name}.replace_flightdistance_missing"
spark.sql(f"""
        CREATE OR REPLACE FUNCTION {flightdistance_function}(flight_distance INT)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        if flight_distance is None:
            return 100
        else:
            return flight_distance
        $$
        """)

inflightwifiservice_function = f"{config.catalog_name}.{config.schema_name}.replace_inflightwifiservice_missing"
spark.sql(f"""
        CREATE OR REPLACE FUNCTION {inflightwifiservice_function}(inflight_wifi_service INT)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        if inflight_wifi_service is None:
            return 3
        else:
            return inflight_wifi_service
        $$
        """)

onlineboarding_function = f"{config.catalog_name}.{config.schema_name}.replace_onlineboarding_missing"
spark.sql(f"""
        CREATE OR REPLACE FUNCTION {onlineboarding_function}(online_boarding INT)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        if online_boarding is None:
            return 3
        else:
            return online_boarding
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
    df=train_set.drop("flight_distance", "inflight_wifi_service", "online_boarding"),
    label=config.target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["flight_distance", "inflight_wifi_service", "online_boarding"],
            lookup_key="id",
            rename_outputs={"flight_distance": "lookup_flightdistance",
                            "inflight_wifi_service": "lookup_inflightwifiservice",
                            "online_boarding": "lookup_onlineboarding"}
                ),
        FeatureFunction(
            udf_name=flightdistance_function,
            output_name="flight_distance",
            input_bindings={"flight_distance": "lookup_flightdistance"},
            ),
        FeatureFunction(
            udf_name=inflightwifiservice_function,
            output_name="inflight_wifi_service",
            input_bindings={"inflight_wifi_service": "lookup_inflightwifiservice"},
        ),
        FeatureFunction(
            udf_name=onlineboarding_function,
            output_name="online_boarding",
            input_bindings={"online_boarding": "lookup_onlineboarding"},
        ),
        FeatureFunction(
            udf_name=function_name,
            output_name="is_loyal_biz",
            input_bindings={
                "customer_type": "customer_type",
                "type_of_travel": "type_of_travel",
            },
        ),
    ],
    exclude_columns=["update_timestamp_utc"],
    )

# COMMAND ----------

# Train & register a model
training_df = training_set.load_df().toPandas()
X_train = training_df[config.features + ["is_loyal_biz"]]
y_train = training_df[config.target]

#pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", full_preprocessing),
        ("classifier", LogisticRegression(**config.parameters, random_state=RANDOM_SEED)),
    ]
)

pipeline.fit(X_train, y_train)

# COMMAND ----------

X_train

# COMMAND ----------

mlflow.set_experiment("/Shared/demo-model-fe")
with mlflow.start_run(run_name="demo-run-model-fe",
                      tags={"git_sha": "1234567890abcd",
                            "branch": "week3"},
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
model_name = f"{config.catalog_name}.{config.schema_name}.model_fe_demo"
model_version = mlflow.register_model(
    model_uri=f'runs:/{run_id}/logistic-pipeline-model-fe',
    name=model_name,
    tags={"git_sha": "1234567890abcd"})

# COMMAND ----------

from pyspark.sql.functions import col

features = [f for f in config.features if f not in lookup_features]
test_set_with_new_id = test_set.select(*features).withColumn(
    "id",
    (col("id").cast("long") + 1000000).cast("string")
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
    TableName='SatisfactionCustomer',
    KeySchema=[
        {
            'AttributeName': 'id',
            'KeyType': 'HASH'  # Partition key
        }
    ],
    AttributeDefinitions=[
        {
            'AttributeName': 'id',
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
    TableName='SatisfactionCustomer',
    Item={
        'id': {'S': 'satisfaction_001'},
        'flight_distance': {'N': '1000'},
        "inflight_wifi_service": {'N': '5'},
        "online_boarding": {'N': '4'}
    }
)

# COMMAND ----------

response = client.get_item(
    TableName='SatisfactionCustomer',
    Key={
        'id': {'S': 'satisfaction_001'}
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
                'id': {'S': str(row['id'])},
                'flight_distance': {'N': str(row['flight_distance'])},
                'inflight_wifi_service': {'N': str(row['inflight_wifi_service'])},
                'online_boarding': {'N': str(row['online_boarding'])}
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
            'SatisfactionCustomer': batch
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
