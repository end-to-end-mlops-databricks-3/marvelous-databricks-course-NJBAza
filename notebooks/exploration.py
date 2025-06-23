# Databricks notebook source
import os
import pickle
import sys
import warnings

from pathlib import Path

import numpy as np
import pandas as pd

from pathlib import Path

from scipy.stats.mstats import winsorize
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
)
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

# COMMAND ----------

PACKAGE_ROOT = Path(os.getcwd()).parent
sys.path.append(str(PACKAGE_ROOT / "src"))
PACKAGE_ROOT

# COMMAND ----------

DATAPATH = os.path.join(PACKAGE_ROOT, "data")
print(DATAPATH)

# COMMAND ----------

df = pd.read_csv(os.path.join(DATAPATH, "data.csv"))

# COMMAND ----------

royal = df.copy()
royal.sample(3)

# COMMAND ----------

royal.dtypes

# COMMAND ----------

ORIGINAL_FEATURES = list(royal.columns)

with open(os.path.join(DATAPATH, "ORIGINAL_FEATURES"), "wb") as fp0:
    pickle.dump(ORIGINAL_FEATURES, fp0)

# COMMAND ----------

ORIGINAL_FEATURES

# COMMAND ----------

royal.isnull().sum().sum()

# COMMAND ----------

royal.shape

# COMMAND ----------

royal["id"] = royal["id"].astype(str)
royal["gender"] = royal["gender"].astype(str)
royal["customer_type"] = royal["customer_type"].astype(str)
royal["type_of_travel"] = royal["type_of_travel"].astype(str)
royal["class"] = royal["class"].astype(str)

# COMMAND ----------

from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)


class DataFrameTypeConverter(BaseEstimator, TransformerMixin):
    def __init__(self, conversion_dict):
        self.conversion_dict = conversion_dict

    def fit(self, X, y=None):
        return self  # Nothing to fit, so just return self

    def transform(self, X):
        X = X.copy()
        for column, new_type in self.conversion_dict.items():
            X[column] = X[column].astype(new_type)
        return X

# COMMAND ----------

CONVERSION_DICT = {
    "id": str,
    "gender": str,
    "customer_type": str,
    "type_of_travel": str,
    "class": str,
}

TO_CONVERT = list(CONVERSION_DICT)

# COMMAND ----------

with open(os.path.join(DATAPATH, "CONVERSION_DICT"), "wb") as fp1:
    pickle.dump(CONVERSION_DICT, fp1)

with open(os.path.join(DATAPATH, "TO_CONVERT"), "wb") as fp2:
    pickle.dump(TO_CONVERT, fp2)

# COMMAND ----------

CONVERSION_DICT, TO_CONVERT

# COMMAND ----------

royal.shape

# COMMAND ----------

royal["id"].nunique()

# COMMAND ----------

VARIABLES_TO_DROP = ["id"]

with open(os.path.join(DATAPATH, "VARIABLES_TO_DROP"), "wb") as fp3:
    pickle.dump(VARIABLES_TO_DROP, fp3)

# COMMAND ----------

# we are able to remove the id feature

royal.drop(VARIABLES_TO_DROP, axis=1, inplace=True)

# COMMAND ----------

royal[royal.duplicated()]

royal.drop_duplicates(inplace=True)

royal.shape

# COMMAND ----------

## Preprocessing Data

### Categorical features

CATEGORICAL_FEATURES = list(royal.select_dtypes(include=["category", "object", "bool"]).columns)

# COMMAND ----------

royal.dtypes

# COMMAND ----------

CATEGORICAL_FEATURES

# COMMAND ----------

royal[CATEGORICAL_FEATURES].isnull().sum().sum()

# COMMAND ----------

# We don't have missing values associated to the categorical features. However, in a general framework we can impute most frequent value for the missing values associated to the categorical features.

class ModeImputer:
    def __init__(self, variables=None):
        self.variables = variables
        self.mode_dict = {}

    def fit(self, df):
        for col in self.variables:
            self.mode_dict[col] = df[col].mode()[0]
        return self

    def transform(self, df):
        df = df.copy()
        for col in self.variables:
            df[col].fillna(self.mode_dict[col], inplace=True)
        return df

# COMMAND ----------

royal[CATEGORICAL_FEATURES] = (
    ModeImputer(variables=CATEGORICAL_FEATURES)
    .fit(royal[CATEGORICAL_FEATURES])
    .transform(royal[CATEGORICAL_FEATURES])
)

# COMMAND ----------

royal[CATEGORICAL_FEATURES].dtypes

# COMMAND ----------

royal[CATEGORICAL_FEATURES].nunique().reset_index().sort_values(by=0, ascending=False)

# COMMAND ----------

for element in CATEGORICAL_FEATURES:
    print(royal[element].value_counts())

# COMMAND ----------

# And we note that all the values have a well represented quantity. Thus, it is not necessary to consider frecuency encoding techniques.

royal[CATEGORICAL_FEATURES].describe().T

# COMMAND ----------

CATEGORICAL_FEATURES.remove("satisfaction_v2")

# COMMAND ----------

with open(os.path.join(DATAPATH, "CATEGORICAL_FEATURES"), "wb") as fp4:
    pickle.dump(CATEGORICAL_FEATURES, fp4)

# COMMAND ----------

NUMERICAL_FEATURES0 = list(royal.select_dtypes(include=["int64", "float64", "Int64"]).columns)

# COMMAND ----------

NUMERICAL_FEATURES0

# COMMAND ----------

royal[NUMERICAL_FEATURES0].describe().T

# COMMAND ----------

royal[NUMERICAL_FEATURES0].isnull().sum()

# COMMAND ----------

class MedianImputer:
    def __init__(self, variables=None):
        self.variables = variables
        self.median_dict = {}

    def fit(self, df):
        for col in self.variables:
            # Calculate and store the median for each variable
            self.median_dict[col] = df[col].median()
        return self

    def transform(self, df):
        df = df.copy()
        for col in self.variables:
            # Fill missing values with the stored median for each variable
            df[col].fillna(self.median_dict[col], inplace=True)
        return df


# COMMAND ----------

royal[NUMERICAL_FEATURES0] = (
    MedianImputer(variables=NUMERICAL_FEATURES0)
    .fit(royal[NUMERICAL_FEATURES0])
    .transform(royal[NUMERICAL_FEATURES0])
)

# COMMAND ----------

with open(os.path.join(DATAPATH, "NUMERICAL_FEATURES0"), "wb") as fp14:
    pickle.dump(NUMERICAL_FEATURES0, fp14)

# COMMAND ----------

royal.isnull().sum().sum()

# COMMAND ----------

NUMERICAL_FEATURES = list(royal.select_dtypes(include=["int64", "float64", "Int64"]).columns)

# COMMAND ----------

with open(os.path.join(DATAPATH, "NUMERICAL_FEATURES"), "wb") as fp5:
    pickle.dump(NUMERICAL_FEATURES, fp5)

# COMMAND ----------

NUMERICAL_FEATURES0, NUMERICAL_FEATURES

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature tools - Creation of new features
# MAGIC
# MAGIC ## Interaction Features:
# MAGIC
# MAGIC *Age and Flight Distance Interaction:* Older passengers might have a different satisfaction level on longer flights.
# MAGIC
# MAGIC Age * Flight Distance
# MAGIC
# MAGIC *Wifi service and Flight Distance:* Longer flights might increase the importance of in-flight wifi service.
# MAGIC
# MAGIC Inflight wifi service * Flight Distance
# MAGIC
# MAGIC *Online Boarding and Ease of Online Booking:* Passengers who find online booking easy may also appreciate online boarding.
# MAGIC
# MAGIC Online boarding * Ease of Online booking

# COMMAND ----------

royal["age_flight_distance_feat"] = royal["age"] * royal["flight_distance"]
royal["wifi_flight_distance_feat"] = royal["inflight_wifi_service"] * royal["flight_distance"]
royal["online_boarding_booking_feat"] = royal["online_boarding"] * royal["ease_of_online_booking"]

# COMMAND ----------

royal["inflight_service_feat"] = (
    royal["inflight_wifi_service"]
    + royal["food_and_drink"]
    + royal["inflight_entertainment"]
    + royal["inflight_service"]
) / 4
royal["preflight_experience_feat"] = (
    royal["ease_of_online_booking"]
    + royal["checkin_service"]
    + royal["gate_location"]
    + royal["departure_arrival_time_convenient"]
) / 4
royal["total_delay_feat"] = royal["departure_delay_in_minutes"] + royal["arrival_delay_in_minutes"]


# COMMAND ----------

binary_features = ["departure_delay_in_minutes", "arrival_delay_in_minutes", "flight_distance"]

# COMMAND ----------

royal[binary_features].describe().T

# COMMAND ----------

royal[binary_features].quantile(0.75), royal[binary_features].quantile(0.15)

# COMMAND ----------

royal["significant_delay_feat"] = np.where(
    (royal["departure_delay_in_minutes"] > 15) | (royal["arrival_delay_in_minutes"] > 15), 1, 0
)
royal["short_flight_feat"] = np.where((royal["flight_distance"] < 300), 1, 0)

# COMMAND ----------

royal["service_difference_feat"] = (
    royal["preflight_experience_feat"] - royal["inflight_service_feat"]
)

# COMMAND ----------

royal["departure_delay_ratio_feat"] = royal["departure_delay_in_minutes"] / (
    1 + royal["total_delay_feat"]
)  # adding 1 in the denominator in order to avoid divergences
royal["on_time_arrival_feat"] = np.where((royal["arrival_delay_in_minutes"] == 0), 1, 0)

# COMMAND ----------

royal["satisfied_boarding_process_feat"] = np.where(
    (royal["ease_of_online_booking"] > 4 & (royal["online_boarding"] > 4)), 1, 0
)
royal["comfortable_flight_feat"] = np.where(
    ((royal["seat_comfort"] > 4) & (royal["leg_room_service"] > 4)), 1, 0
)

# COMMAND ----------

royal["satisfied_boarding_process_feat"].value_counts()

# COMMAND ----------

royal["comfortable_flight_feat"].value_counts()

# COMMAND ----------

royal[["departure_delay_in_minutes", "arrival_delay_in_minutes"]].quantile(0.9)

# COMMAND ----------

royal["long_delay_feat"] = np.where(
    (royal["departure_delay_in_minutes"] > 45) | (royal["arrival_delay_in_minutes"] > 45), 1, 0
)
royal["total_service_time_feat"] = (
    royal["gate_location"] + royal["checkin_service"] + royal["departure_arrival_time_convenient"]
)

# COMMAND ----------

royal["flight_distance"].describe()

# COMMAND ----------

royal["long_haul_feat"] = np.where(royal["flight_distance"] > 2000, 1, 0)

# COMMAND ----------

NUMERICAL_FEATURES_CREATED = [feature for feature in list(royal.columns) if "_feat" in feature]

# COMMAND ----------

NUMERICAL_FEATURES_CREATED

# COMMAND ----------

with open(os.path.join(DATAPATH, "NUMERICAL_FEATURES_CREATED"), "wb") as fp7:
    pickle.dump(NUMERICAL_FEATURES, fp7)

# COMMAND ----------

# MAGIC %md
# MAGIC # Outliers treatment

# COMMAND ----------

from scipy.stats.mstats import winsorize

# COMMAND ----------

NUMERICAL_FEATURES2 = list(royal.select_dtypes(include=["int64", "float64", "Int64"]).columns)

# COMMAND ----------

NUMERICAL_FEATURES2

# COMMAND ----------

for feature in NUMERICAL_FEATURES2:
    royal[feature + "_winsor"] = winsorize(royal[feature], limits=[0.025, 0.025])

# COMMAND ----------

with open(os.path.join(DATAPATH, "NUMERICAL_FEATURES2"), "wb") as fp15:
    pickle.dump(NUMERICAL_FEATURES2, fp15)

# COMMAND ----------

royal.columns

# COMMAND ----------

NUMERICAL_FEATURES_WINSOR = [feature + "_winsor" for feature in NUMERICAL_FEATURES2]

with open(os.path.join(DATAPATH, "NUMERICAL_FEATURES_WINSOR"), "wb") as fp6:
    pickle.dump(NUMERICAL_FEATURES_WINSOR, fp6)

# COMMAND ----------

royal.shape

# COMMAND ----------

NUMERICAL_FEATURES_WINSOR

# COMMAND ----------

print(
    f"The number of features that will be useful for the model building is: {len(NUMERICAL_FEATURES_WINSOR)}"
)

# COMMAND ----------

NUMERICAL_FEATURES_3 = list(
    royal[NUMERICAL_FEATURES_WINSOR].select_dtypes(include=["int64", "float64", "Int64"]).columns
)


# COMMAND ----------

NUMERICAL_FEATURES_3

# COMMAND ----------

with open(os.path.join(DATAPATH, "NUMERICAL_FEATURES_3"), "wb") as fp8:
    pickle.dump(NUMERICAL_FEATURES_3, fp8)

# COMMAND ----------

royal[NUMERICAL_FEATURES_3].sample()

# COMMAND ----------

royal[NUMERICAL_FEATURES_3].isnull().sum().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC # Filtering - Elimination Methods
# MAGIC
# MAGIC ## Log Transformation:
# MAGIC
# MAGIC We use the `log` transformation to reduce the skewness of the data. It help make distributions more symmetric. It compresses the range of the data by reducing the impact of outliers and brings the data closer to a normal distribution.

# COMMAND ----------

royal_key = royal[CATEGORICAL_FEATURES + NUMERICAL_FEATURES_3]

# COMMAND ----------

royal_log = np.log1p(abs(royal_key[NUMERICAL_FEATURES_3]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Standard Scaler:
# MAGIC
# MAGIC We use `StandardScaler` to transform the data to have a mean of 0 and a standard deviation of 1. This is particularly useful for algorithms that assume or perform better when features are centered around zero. It standardizes the range of continuous features, which can improve the performance of many algorithms.

# COMMAND ----------

std_royal = StandardScaler()
royal_scaled = std_royal.fit_transform(royal_log)
royal_scaled = pd.DataFrame(royal_scaled, columns=NUMERICAL_FEATURES_3)

# COMMAND ----------

# MAGIC %md
# MAGIC # Min-Max Scaler:
# MAGIC
# MAGIC `Min-Max Scaler` scales each feature to $0$ and $1$. Thus, it rescales the data to fit within a specific range, making it easier to compare different features on the same scale.

# COMMAND ----------

min_max_royal = MinMaxScaler()
scaled_min_max_royal = min_max_royal.fit_transform(royal_scaled)
scaled_min_max_royal = pd.DataFrame(scaled_min_max_royal, columns=NUMERICAL_FEATURES_3)

# COMMAND ----------

# creating the correlation matrix
corr_matrix = scaled_min_max_royal.corr().abs()

# COMMAND ----------

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))

# COMMAND ----------

s = corr_matrix.unstack()
so = s.sort_values(kind="quicksort", ascending=False)
so = pd.DataFrame(so, columns=["Pearson Correlation"])

# COMMAND ----------

# fining index of variables with correlation greater than the threshold
TO_DROP = [column for column in upper.columns if any(upper[column] > 0.8)]

# COMMAND ----------

NUMERICAL_CONSIDER = [feature for feature in NUMERICAL_FEATURES_3 if feature not in TO_DROP]
NUMERICAL_CONSIDER

# COMMAND ----------

with open(os.path.join(DATAPATH, "NUMERICAL_CONSIDER"), "wb") as fp9:
    pickle.dump(NUMERICAL_CONSIDER, fp9)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Low variance filter

# COMMAND ----------

from sklearn.preprocessing import normalize

normalize = normalize(scaled_min_max_royal[NUMERICAL_CONSIDER])
data_scaled = pd.DataFrame(normalize)

variance = data_scaled.var()
columns = scaled_min_max_royal[NUMERICAL_CONSIDER].columns

# COMMAND ----------

variance_df = pd.DataFrame(variance, columns=["Variance"])
variance_df["Column"] = columns
variance_df = variance_df.reset_index(drop=True)

# COMMAND ----------

variance_df.shape

# COMMAND ----------

variable = []

for i in range(0, len(variance)):
    if variance[i] >= 0.001:
        variable.append(columns[i])

# COMMAND ----------

variable

# COMMAND ----------

# MAGIC %md
# MAGIC ## Categorical Tranformations

# COMMAND ----------

royal[CATEGORICAL_FEATURES + NUMERICAL_CONSIDER].shape

# COMMAND ----------

royal[CATEGORICAL_FEATURES].nunique()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Categorical to numerical features:

# COMMAND ----------

royal['gender'].value_counts()

# COMMAND ----------

FEATURES_ENCODE = ["gender", "customer_type", "type_of_travel"]

#for column in FEATURES_ENCODE:
#    le = LabelEncoder()
#    royal[column] = le.fit_transform(royal[column])

# COMMAND ----------

royal[CATEGORICAL_FEATURES + NUMERICAL_CONSIDER].sample()

# COMMAND ----------

with open(os.path.join(DATAPATH, "FEATURES_ENCODE"), "wb") as fp10:
    pickle.dump(NUMERICAL_CONSIDER, fp10)

# COMMAND ----------

royal_final = pd.get_dummies(
    royal[CATEGORICAL_FEATURES + NUMERICAL_CONSIDER + ["satisfaction_v2"]],
    columns=["gender", "customer_type", "type_of_travel", "class"], drop_first=False, dtype=int,
)

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessor_cat = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         ["gender", "customer_type", "type_of_travel", "class"])
    ],
    remainder="passthrough"
)

royal_cat = preprocessor_cat.fit_transform(royal[["gender", "customer_type", "type_of_travel", "class"]])
pd.DataFrame(royal_cat)

# COMMAND ----------

royal_final

# COMMAND ----------

# Normalize dummy column names after get_dummies
royal_final.columns = [
    col.lower().replace(" ", "_") for col in royal_final.columns
]

# Extract normalized dummy column names
dummy_columns = [
    col for col in royal_final.columns
    if "gender" in col or "customer_type" in col or "type_of_travel" in col or "class" in col
]


# COMMAND ----------

dummy_columns

# COMMAND ----------

royal_final[dummy_columns] = royal_final[dummy_columns].astype(int)

# COMMAND ----------

royal_final.sample(3)

# COMMAND ----------

FEATURES_ONE_HOT = ["class"]

# COMMAND ----------

with open(os.path.join(DATAPATH, "FEATURES_ONE_HOT"), "wb") as fp11:
    pickle.dump(FEATURES_ONE_HOT, fp11)

# COMMAND ----------

royal_final.sample()

# COMMAND ----------

final_columns = list(royal_final.columns)

# COMMAND ----------

TO_CONVERT2 = [element for element in final_columns if element != "satisfaction_v2"]

# COMMAND ----------

TO_CONVERT2

# COMMAND ----------

CONVERSION_DICT2 = {}

for element in TO_CONVERT2:
    if element not in CONVERSION_DICT2:
        CONVERSION_DICT2[element] = float

# COMMAND ----------

CONVERSION_DICT2

# COMMAND ----------

with open(os.path.join(DATAPATH, "CONVERSION_DICT2"), "wb") as fp12:
    pickle.dump(CONVERSION_DICT2, fp12)

with open(os.path.join(DATAPATH, "TO_CONVERT2"), "wb") as fp13:
    pickle.dump(TO_CONVERT2, fp13)

# COMMAND ----------

TO_DROP = [
    "gender_female",
    "customer_type_disloyal_customer",
    "type_of_travel_personal_travel",
    "class_business",
]

# COMMAND ----------

with open(os.path.join(DATAPATH, "TO_DROP"), "wb") as fp16:
    pickle.dump(TO_DROP, fp16)

# COMMAND ----------

# MAGIC %md
# MAGIC # TRAINING

# COMMAND ----------

from sklearn.pipeline import Pipeline

# COMMAND ----------

import satisfaction_customer.processing.preprocessing as pp
from satisfaction_customer.settings import settings

# COMMAND ----------

pipeline = Pipeline(
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
                columns=settings.NUMERICAL_FEATURES_3 + settings.CATEGORICAL_FEATURES
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

settings.CONVERSION_DICT2

# COMMAND ----------

settings.TO_DROP

# COMMAND ----------

df_preprocess = df.copy()

# COMMAND ----------

df_preprocess.sample(3)

# COMMAND ----------

X_preprocess = df_preprocess.drop('satisfaction_v2', axis=1)
y_preprocess = df_preprocess['satisfaction_v2']

# COMMAND ----------

df_transformed = pipeline.fit_transform(X_preprocess)

pipeline.fit(X_preprocess)

# Get final features from the last step (before estimator)
X_final = pipeline.transform(X_preprocess)

if isinstance(X_final, pd.DataFrame):
    print(X_final.columns.tolist())


# COMMAND ----------

 df_transformed

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from satisfaction_customer.config import ProjectConfig

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
RANDOM_SEED = 20230916


model = Pipeline(
        steps=[
                ("preprocessor", pipeline),
                (
                    "classifier",
                    LogisticRegression(**config.parameters, random_state=RANDOM_SEED),
                ),
            ]
        )

# COMMAND ----------

model.fit(X_preprocess, y_preprocess)

# COMMAND ----------

print(model.named_steps["pipeline"].get_feature_names_out())

# COMMAND ----------


