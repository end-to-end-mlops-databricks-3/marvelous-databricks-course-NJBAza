#!/usr/bin/env python
# coding: utf-8
# COMMAND ----------

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


# In[4]:


PACKAGE_ROOT = Path(os.getcwd()).parent
sys.path.append(str(PACKAGE_ROOT.parent))


# In[5]:


DATAPATH = os.path.join(PACKAGE_ROOT, "data")

# COMMAND ----------
print(DATAPATH)

# In[6]:


df = pd.read_csv(os.path.join(DATAPATH, "data.csv"))


# In[7]:


#!pip freeze > requirements.txt


# In[9]:


royal = df.copy()


# ---
#
# <center><h1>💻💻 Data Preparation 💻 💻</h1></center>
#
# ---

# In[10]:


royal.sample(3)


# In[11]:


royal.dtypes


# In[12]:


ORIGINAL_FEATURES = list(royal.columns)

with open(os.path.join(DATAPATH, "ORIGINAL_FEATURES"), "wb") as fp0:
    pickle.dump(ORIGINAL_FEATURES, fp0)


# In[13]:


royal.isnull().sum().sum()


# In[14]:


royal.shape


# In[15]:


# considering the data types

royal["gender"] = royal["gender"].astype(str)
royal["customer_type"] = royal["customer_type"].astype(str)
royal["type_of_travel"] = royal["type_of_travel"].astype(str)
royal["class"] = royal["class"].astype(str)


# Alternatively we can create a **DataFrameTypeConverter** of different features

# In[16]:


# datatype converter
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


# In[17]:


CONVERSION_DICT = {"gender": str, "customer_type": str, "type_of_travel": str, "class": str}

TO_CONVERT = list(CONVERSION_DICT)


# In[18]:


with open(os.path.join(DATAPATH, "CONVERSION_DICT"), "wb") as fp1:
    pickle.dump(CONVERSION_DICT, fp1)

with open(os.path.join(DATAPATH, "TO_CONVERT"), "wb") as fp2:
    pickle.dump(TO_CONVERT, fp2)


# In[17]:


royal.shape


# In[18]:


royal["id"].nunique()


# In[19]:


VARIABLES_TO_DROP = ["id"]

with open(os.path.join(DATAPATH, "VARIABLES_TO_DROP"), "wb") as fp3:
    pickle.dump(VARIABLES_TO_DROP, fp3)


# In[20]:


# we are able to remove the id feature

royal.drop(VARIABLES_TO_DROP, axis=1, inplace=True)


# In[21]:


royal[royal.duplicated()]


# In[22]:


royal.drop_duplicates(inplace=True)


# In[23]:


royal.shape


# ## Preprocessing Data

# ### Categorical features

# In[24]:


CATEGORICAL_FEATURES = list(royal.select_dtypes(include=["category", "object", "bool"]).columns)


# In[25]:


royal.dtypes


# In[26]:


CATEGORICAL_FEATURES


# In[27]:


royal[CATEGORICAL_FEATURES].isnull().sum().sum()


# We don't have missing values associated to the categorical features. However, in a general framework we can impute most frequent value for the missing values associated to the categorical features.

# In[28]:


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


# In[29]:


royal[CATEGORICAL_FEATURES] = (
    ModeImputer(variables=CATEGORICAL_FEATURES)
    .fit(royal[CATEGORICAL_FEATURES])
    .transform(royal[CATEGORICAL_FEATURES])
)


# In[30]:


royal[CATEGORICAL_FEATURES].dtypes


# In[31]:


royal[CATEGORICAL_FEATURES].nunique().reset_index().sort_values(by=0, ascending=False)


# In[32]:


for element in CATEGORICAL_FEATURES:
    print(royal[element].value_counts())


# And we note that all the values have a well represented quantity. Thus, it is not necessary to consider frecuency encoding techniques.

# In[33]:


royal[CATEGORICAL_FEATURES].describe().T


# In[34]:


# Custom function for easy visualisation of Categorical Variables


def UVA_category(data, var_group):
    """
    Univariate_Analysis_categorical
    takes a group of variables (category) and plot/print all the value_counts and barplot.
    """
    # setting figure_size
    size = len(var_group)
    plt.figure(figsize=(7 * size, 5), dpi=100)

    # for every variable
    for j, i in enumerate(var_group):
        norm_count = data[i].value_counts(normalize=True) * 100
        n_uni = data[i].nunique()

        # Plotting the variable with every information
        plt.subplot(1, size, j + 1)
        sns.barplot(data, x=norm_count, y=norm_count.index, order=norm_count.index)
        plt.xlabel("fraction/percent", fontsize=20)
        plt.ylabel("{}".format(i), fontsize=20)
        plt.title("n_uniques = {} \n value counts \n {};".format(n_uni, norm_count))


# In[35]:


UVA_category(royal, CATEGORICAL_FEATURES)


# In[36]:


CATEGORICAL_FEATURES.remove("satisfaction_v2")


# In[37]:


with open(os.path.join(DATAPATH, "CATEGORICAL_FEATURES"), "wb") as fp4:
    pickle.dump(CATEGORICAL_FEATURES, fp4)


# ## Univariate Analysis: Numerical Variables

# In[38]:


# Listing the Numerical and Categorical datatypes

NUMERICAL_FEATURES0 = list(royal.select_dtypes(include=["int64", "float64", "Int64"]).columns)


# In[39]:


NUMERICAL_FEATURES0


# In[40]:


# custom function for easy and efficient analysis of numerical univariate


def UVA_numeric(data, var_group):
    """
    Univariate_Analysis_numeric
    takes a group of variables (INTEGER and FLOAT) and plot/print all the descriptives and properties along with KDE.

    Runs a loop: calculate all the descriptives of i(th) variable and plot/print it
    """

    size = len(var_group)
    plt.figure(figsize=(7 * size, 3), dpi=100)

    # looping for each variable
    for j, i in enumerate(var_group):

        # calculating descriptives of variable
        mini = data[i].min()
        maxi = data[i].max()
        ran = data[i].max() - data[i].min()
        mean = data[i].mean()
        median = data[i].median()
        st_dev = data[i].std()
        skew = data[i].skew()
        kurt = data[i].kurtosis()

        # calculating points of standard deviation
        points = mean - st_dev, mean + st_dev

        # Plotting the variable with every information
        plt.subplot(1, size, j + 1)
        sns.kdeplot(x=data[i], shade=True)
        sns.lineplot(x=points, y=[0, 0], color="black", label="std_dev")
        sns.scatterplot(x=[mini, maxi], y=[0, 0], color="orange", label="min/max")
        sns.scatterplot(x=[mean], y=[0], color="red", label="mean")
        sns.scatterplot(x=[median], y=[0], color="blue", label="median")
        plt.xlabel("{}".format(i), fontsize=20)
        plt.ylabel("density")
        plt.title(
            "within 1 std_dev = {}; kurtosis = {};\nskew = {}; range = {}\nmean = {}; median = {}".format(
                (round(points[0], 2), round(points[1], 2)),
                round(kurt, 2),
                round(skew, 2),
                (round(mini, 2), round(maxi, 2), round(ran, 2)),
                round(mean, 2),
                round(median, 2),
            )
        )


# In[41]:


UVA_numeric(royal, NUMERICAL_FEATURES0)


# In[42]:


royal[NUMERICAL_FEATURES0].describe().T


# In[43]:


royal[NUMERICAL_FEATURES0].isnull().sum()


# We observe that we are able to use median imputers in order to deal, in a general sense, with the behaviour of the underlying outlier values.

# In[44]:


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


# In[45]:


royal[NUMERICAL_FEATURES0] = (
    MedianImputer(variables=NUMERICAL_FEATURES0)
    .fit(royal[NUMERICAL_FEATURES0])
    .transform(royal[NUMERICAL_FEATURES0])
)


# In[46]:


with open(os.path.join(DATAPATH, "NUMERICAL_FEATURES0"), "wb") as fp14:
    pickle.dump(NUMERICAL_FEATURES0, fp14)


# we verify that everithing works

# In[47]:


royal.isnull().sum()


# ### Univariate Analysis: Outliers

# In[48]:


# custom function for easy outlier analysis


def UVA_outlier(data, var_group, include_outlier=True):
    """
    Univariate_Analysis_outlier:
    takes a group of variables (INTEGER and FLOAT) and plot/print boplot and descriptives\n
    Runs a loop: calculate all the descriptives of i(th) variable and plot/print it \n\n

    data : dataframe from which to plot from\n
    var_group : {list} type Group of Continuous variables\n
    include_outlier : {bool} whether to include outliers or not, default = True\n
    """

    size = len(var_group)
    plt.figure(figsize=(7 * size, 4), dpi=100)

    # looping for each variable
    for j, i in enumerate(var_group):

        # calculating descriptives of variable
        quant25 = data[i].quantile(0.25)
        quant75 = data[i].quantile(0.75)
        IQR = quant75 - quant25
        med = data[i].median()
        whis_low = med - (1.5 * IQR)
        whis_high = med + (1.5 * IQR)

        # Calculating Number of Outliers
        outlier_high = len(data[i][data[i] > whis_high])
        outlier_low = len(data[i][data[i] < whis_low])

        if include_outlier == True:
            print(include_outlier)
            # Plotting the variable with every information
            plt.subplot(1, size, j + 1)
            sns.boxplot(data[i], orient="v")
            plt.ylabel("{}".format(i))
            plt.title(
                "With Outliers\nIQR = {}; Median = {} \n 2nd,3rd  quartile = {};\n Outlier (low/high) = {} \n".format(
                    round(IQR, 2),
                    round(med, 2),
                    (round(quant25, 2), round(quant75, 2)),
                    (outlier_low, outlier_high),
                )
            )

        else:
            # replacing outliers with max/min whisker
            data2 = data[var_group][:]
            data2[i][data2[i] > whis_high] = whis_high + 1
            data2[i][data2[i] < whis_low] = whis_low - 1

            # plotting without outliers
            plt.subplot(1, size, j + 1)
            sns.boxplot(data2[i], orient="v")
            plt.ylabel("{}".format(i))
            plt.title(
                "Without Outliers\nIQR = {}; Median = {} \n 2nd,3rd  quartile = {};\n Outlier (low/high) = {} \n".format(
                    round(IQR, 2),
                    round(med, 2),
                    (round(quant25, 2), round(quant75, 2)),
                    (outlier_low, outlier_high),
                )
            )


# In[49]:


UVA_outlier(
    royal,
    NUMERICAL_FEATURES0,
)


# In[50]:


UVA_outlier(royal, NUMERICAL_FEATURES0, False)


# In[51]:


# Listing the Numerical and Categorical datatypes
NUMERICAL_FEATURES = list(royal.select_dtypes(include=["int64", "float64", "Int64"]).columns)


# In[52]:


with open(os.path.join(DATAPATH, "NUMERICAL_FEATURES"), "wb") as fp5:
    pickle.dump(NUMERICAL_FEATURES, fp5)


# # Feature tools - Creation of new features

# ## Interaction Features:
#
#    *Age and Flight Distance Interaction:* Older passengers might have a different satisfaction level on longer flights.
#
#    Age * Flight Distance
#
#    *Wifi service and Flight Distance:* Longer flights might increase the importance of in-flight wifi service.
#
#    Inflight wifi service * Flight Distance
#
#    *Online Boarding and Ease of Online Booking:* Passengers who find online booking easy may also appreciate online boarding.
#
#    Online boarding * Ease of Online booking

# In[53]:


royal["age_flight_distance_feat"] = royal["age"] * royal["flight_distance"]
royal["wifi_flight_distance_feat"] = royal["inflight_wifi_service"] * royal["flight_distance"]
royal["online_boarding_booking_feat"] = royal["online_boarding"] * royal["ease_of_online_booking"]


# ## Aggregated Ratings
# *Average In-Flight Service Rating:* Combine all ratings related to in-flight service into one overall rating.
#
# Inflight Service Score = (Inflight wifi service + Food and drink + Inflight entertainment + Inflight service) / 4
#
# *Average Pre-Flight Experience:* Combine ratings related to pre-flight experiences.
#
# Pre-Flight Experience Score = (Ease of Online booking + Checkin service + Gate location + Departure/Arrival time convenient) / 4
#
# *Total Delay Time:* Combine departure and arrival delays.
#
# Total Delay Time = Departure Delay in Minutes + Arrival Delay in Minutes

# In[54]:


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


# ## Binary Features

# In[55]:


binary_features = ["departure_delay_in_minutes", "arrival_delay_in_minutes", "flight_distance"]


# In[56]:


royal[binary_features].describe().T


# In[57]:


royal[binary_features].quantile(0.75), royal[binary_features].quantile(0.15)


# *Significant Delay:* Create a binary feature to flag significant delays (e.g., over 30 minutes).
#
# Significant Delay = (Departure Delay in Minutes > 30) or (Arrival Delay in Minutes > 30)
#
# *Short Flight:* Create a binary feature indicating if the flight distance is short (e.g., less than 300 km).
#
# Short Flight = (Flight Distance < 300)

# In[58]:


royal["significant_delay_feat"] = np.where(
    (royal["departure_delay_in_minutes"] > 15) | (royal["arrival_delay_in_minutes"] > 15), 1, 0
)
royal["short_flight_feat"] = np.where((royal["flight_distance"] < 300), 1, 0)


# ## Service Differences
# *Pre vs In-flight Service Rating Difference:* Difference between pre-flight service ratings and in-flight service ratings, indicating a mismatch in customer experience.
#
# Service Difference = Pre-Flight Experience Score - Inflight Service Score

# In[59]:


royal["service_difference_feat"] = (
    royal["preflight_experience_feat"] - royal["inflight_service_feat"]
)


# ## Delay Features
# *Departure Delay Ratio:* Ratio of departure delay to total delay.
#
# Departure Delay Ratio = Departure Delay in Minutes / Total Delay Time
#
# *On-time Arrival:* A binary feature that flags if the flight arrived on time.
#
# On-time Arrival = (Arrival Delay in Minutes == 0)

# In[60]:


royal["departure_delay_ratio_feat"] = royal["departure_delay_in_minutes"] / (
    1 + royal["total_delay_feat"]
)  # adding 1 in the denominator in order to avoid divergences
royal["on_time_arrival_feat"] = np.where((royal["arrival_delay_in_minutes"] == 0), 1, 0)


# ## **Satisfaction-Related Features**
#
# *Satisfied with Boarding Process:* Binary feature indicating if passengers rated both ease of online booking and online boarding highly.
#
# Satisfied with Boarding Process = (Ease of Online booking > 4) and (Online boarding > 4)
#
# *Comfortable Flight:* Create a binary feature that indicates if seat comfort and legroom service were rated highly.
#
# Comfortable Flight = (Seat comfort > 4) and (Leg room service > 4)

# In[61]:


royal["satisfied_boarding_process_feat"] = np.where(
    (royal["ease_of_online_booking"] > 4 & (royal["online_boarding"] > 4)), 1, 0
)
royal["comfortable_flight_feat"] = np.where(
    ((royal["seat_comfort"] > 4) & (royal["leg_room_service"] > 4)), 1, 0
)


# In[62]:


royal["satisfied_boarding_process_feat"].value_counts()


# In[63]:


royal["comfortable_flight_feat"].value_counts()


# ## Time-Related Features

# In[64]:


royal[["departure_delay_in_minutes", "arrival_delay_in_minutes"]].quantile(0.9)


# *Long Delay:* Binary feature indicating if either the departure or arrival delay exceeds the threshold of 45 minutes.
#
# Long Delay = (Departure Delay in Minutes > 45) or (Arrival Delay in Minutes > 45)
#
# *Total Service Time:* Sum of the gate location (representing time spent walking), check-in service, and other time-related features.
#
# Total Service Time = Gate location + Checkin service + Departure/Arrival time convenient

# In[65]:


royal["long_delay_feat"] = np.where(
    (royal["departure_delay_in_minutes"] > 45) | (royal["arrival_delay_in_minutes"] > 45), 1, 0
)
royal["total_service_time_feat"] = (
    royal["gate_location"] + royal["checkin_service"] + royal["departure_arrival_time_convenient"]
)


# ## **Flight Type**

# In[66]:


royal["flight_distance"].describe()


# *Long-Haul vs Short-Haul Flight:* Categorize flights as long-haul or short-haul based on distance.
#
# Long-Haul = if Flight Distance > 2000

# In[67]:


royal["long_haul_feat"] = np.where(royal["flight_distance"] > 2000, 1, 0)


# In[68]:


NUMERICAL_FEATURES_CREATED = [feature for feature in list(royal.columns) if "_feat" in feature]


# In[69]:


NUMERICAL_FEATURES_CREATED


# In[70]:


UVA_numeric(royal, NUMERICAL_FEATURES_CREATED)


# In[71]:


with open(os.path.join(DATAPATH, "NUMERICAL_FEATURES_CREATED"), "wb") as fp7:
    pickle.dump(NUMERICAL_FEATURES, fp7)


# # Outliers treatment

# In[72]:


from scipy.stats.mstats import winsorize


# In[73]:


NUMERICAL_FEATURES2 = list(royal.select_dtypes(include=["int64", "float64", "Int64"]).columns)


# In[74]:


NUMERICAL_FEATURES2


# In[75]:


for feature in NUMERICAL_FEATURES2:
    royal[feature + "_winsor"] = winsorize(royal[feature], limits=[0.025, 0.025])


# In[76]:


with open(os.path.join(DATAPATH, "NUMERICAL_FEATURES2"), "wb") as fp15:
    pickle.dump(NUMERICAL_FEATURES2, fp15)


# In[77]:


royal.describe().T


# In[78]:


NUMERICAL_FEATURES_WINSOR = [feature + "_winsor" for feature in NUMERICAL_FEATURES2]

with open(os.path.join(DATAPATH, "NUMERICAL_FEATURES_WINSOR"), "wb") as fp6:
    pickle.dump(NUMERICAL_FEATURES_WINSOR, fp6)


# In[79]:


UVA_numeric(royal, NUMERICAL_FEATURES)


# In[80]:


UVA_numeric(royal, NUMERICAL_FEATURES_WINSOR)


# In[81]:


royal.shape


# We then consider the following dataset:

# In[82]:


NUMERICAL_FEATURES_WINSOR


# In[83]:


print(
    f"The number of features that will be useful for the model building is: {len(NUMERICAL_FEATURES_WINSOR)}"
)


# In[84]:


NUMERICAL_FEATURES_3 = list(
    royal[NUMERICAL_FEATURES_WINSOR].select_dtypes(include=["int64", "float64", "Int64"]).columns
)


# In[85]:


with open(os.path.join(DATAPATH, "NUMERICAL_FEATURES_3"), "wb") as fp8:
    pickle.dump(NUMERICAL_FEATURES_3, fp8)


# In[86]:


royal[NUMERICAL_FEATURES_3].sample()


# In[87]:


royal[NUMERICAL_FEATURES_3].isnull().sum()


# # Filtering - Elimination Methods

# ## Log Transformation:
#
# We use the `log` transformation to reduce the skewness of the data. It help make distributions more symmetric. It compresses the range of the data by reducing the impact of outliers and brings the data closer to a normal distribution.

# In[88]:


royal_key = royal[CATEGORICAL_FEATURES + NUMERICAL_FEATURES_3]


# In[89]:


royal_key.shape


# In[90]:


royal_log = np.log1p(abs(royal_key[NUMERICAL_FEATURES_3]))


# ## Standard Scaler:
#
# We use `StandardScaler` to transform the data to have a mean of 0 and a standard deviation of 1. This is particularly useful for algorithms that assume or perform better when features are centered around zero. It standardizes the range of continuous features, which can improve the performance of many algorithms.

# In[91]:


std_royal = StandardScaler()
royal_scaled = std_royal.fit_transform(royal_log)
royal_scaled = pd.DataFrame(royal_scaled, columns=NUMERICAL_FEATURES_3)


# In[92]:


royal_scaled.describe().T


# In[93]:


UVA_numeric(royal_scaled, NUMERICAL_FEATURES_3)


# ## Min-Max Scaler:
#
# `Min-Max Scaler` scales each feature to $0$ and $1$. Thus, it rescales the data to fit within a specific range, making it easier to compare different features on the same scale.

# In[94]:


min_max_royal = MinMaxScaler()
scaled_min_max_royal = min_max_royal.fit_transform(royal_scaled)
scaled_min_max_royal = pd.DataFrame(scaled_min_max_royal, columns=NUMERICAL_FEATURES_3)


# In[95]:


scaled_min_max_royal.describe().T


# In[96]:


UVA_numeric(scaled_min_max_royal, NUMERICAL_FEATURES_3)


# ### High correlation filtering

# In[97]:


# creating the correlation matrix
corr_matrix = scaled_min_max_royal.corr().abs()


# In[98]:


# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))


# In[99]:


s = corr_matrix.unstack()
so = s.sort_values(kind="quicksort", ascending=False)
so = pd.DataFrame(so, columns=["Pearson Correlation"])


# In[100]:


so[so["Pearson Correlation"] < 1].head()


# For higher correlation filter conditions we have that the following features will not appear in the sequel:

# In[101]:


# fining index of variables with correlation greater than the threshold
TO_DROP = [column for column in upper.columns if any(upper[column] > 0.8)]


# In[102]:


NUMERICAL_CONSIDER = [feature for feature in NUMERICAL_FEATURES_3 if feature not in TO_DROP]


# In[103]:


NUMERICAL_CONSIDER


# In[104]:


with open(os.path.join(DATAPATH, "NUMERICAL_CONSIDER"), "wb") as fp9:
    pickle.dump(NUMERICAL_CONSIDER, fp9)


# ### Low variance filter

# In[105]:


from sklearn.preprocessing import normalize


# In[106]:


normalize = normalize(scaled_min_max_royal[NUMERICAL_CONSIDER])


# In[107]:


data_scaled = pd.DataFrame(normalize)


# In[108]:


# storing the variance and name of variables
variance = data_scaled.var()
columns = scaled_min_max_royal[NUMERICAL_CONSIDER].columns


# In[109]:


# Creating a DataFrame from the variances
variance_df = pd.DataFrame(variance, columns=["Variance"])
variance_df["Column"] = columns
variance_df = variance_df.reset_index(drop=True)


# In[110]:


variance_df.shape


# In[111]:


variance_df.sort_values(by="Variance", ascending=False).head()


# In[112]:


variance_df.describe()


# In[113]:


# saving the names of variables having variance more than a threshold value
variable = []

for i in range(0, len(variance)):
    if variance[i] >= 0.001:
        variable.append(columns[i])


# In[114]:


len(variable)


# In[115]:


variable


# We observe that the numerical features to consider coincides with our variable feature list.

# ## Categorical Tranformations

# In[1]:


royal[CATEGORICAL_FEATURES + NUMERICAL_CONSIDER].shape


# In[117]:


royal[CATEGORICAL_FEATURES].nunique()


# #### Categorical to numerical features:

# In[118]:


FEATURES_ENCODE = ["gender", "customer_type", "type_of_travel"]

for column in FEATURES_ENCODE:
    le = LabelEncoder()
    royal[column] = le.fit_transform(royal[column])
# In[119]:


royal[CATEGORICAL_FEATURES + NUMERICAL_CONSIDER].sample()


# In[120]:


with open(os.path.join(DATAPATH, "FEATURES_ENCODE"), "wb") as fp10:
    pickle.dump(NUMERICAL_CONSIDER, fp10)


# In[121]:


royal_final = pd.get_dummies(
    royal[CATEGORICAL_FEATURES + NUMERICAL_CONSIDER + ["satisfaction_v2"]],
    columns=["gender", "customer_type", "type_of_travel", "class"],
)


# In[132]:


dummy_columns = [
    col
    for col in royal_final.columns
    if "gender" in col or "customer" in col or "type" in col or "class" in col
]

# Apply .astype(int) only to the dummy columns
royal_final[dummy_columns] = royal_final[dummy_columns].astype(int)


# In[135]:


royal_final.sample(3)


# In[136]:


royal_final["satisfaction_v2"].value_counts()


# In[137]:


FEATURES_ONE_HOT = ["class"]


# In[138]:


with open(os.path.join(DATAPATH, "FEATURES_ONE_HOT"), "wb") as fp11:
    pickle.dump(FEATURES_ONE_HOT, fp11)


# In[139]:


royal_final.dtypes


# In order to train in next steps we now transform all the format of the different numeric columns no float:

# In[140]:


final_columns = list(royal_final.columns)


# In[141]:


TO_CONVERT2 = [element for element in final_columns if element != "satisfaction_v2"]

CONVERSION_DICT2 = {}

for element in TO_CONVERT2:
    if element not in CONVERSION_DICT2:
        CONVERSION_DICT2[element] = float


# In[144]:


CONVERSION_DICT2


# In[145]:


with open(os.path.join(DATAPATH, "CONVERSION_DICT2"), "wb") as fp12:
    pickle.dump(CONVERSION_DICT2, fp12)

with open(os.path.join(DATAPATH, "TO_CONVERT2"), "wb") as fp13:
    pickle.dump(TO_CONVERT2, fp13)


# In[146]:


royal_final[:1]


# In[134]:


royal_final.shape


# In[147]:


TO_DROP = [
    "gender_female",
    "customer_type_disloyal_customer",
    "type_of_travel_personal_travel",
    "class_business",
]

with open(os.path.join(DATAPATH, "TO_DROP"), "wb") as fp16:
    pickle.dump(TO_DROP, fp16)


# In[ ]:
