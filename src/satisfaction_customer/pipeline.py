import os
import sys
from pathlib import Path

import processing.preprocessing as pp
from config import config
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Assuming 'prediction_model' is in the parent directory
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT.parent))


RANDOM_SEED = 20230916

classification_pipeline = Pipeline(
    [
        (
            "DataFrameTypeConverter",
            pp.DataFrameTypeConverter(conversion_dict=config.CONVERSION_DICT),
        ),
        (
            "DropColumns",
            pp.DropColumns(variables_to_drop=config.VARIABLES_TO_DROP),
        ),
        (
            "ModeImputer",
            pp.ModeImputer(variables=config.CATEGORICAL_FEATURES),
        ),
        (
            "MedianImputer",
            pp.MedianImputer(variables=config.NUMERICAL_FEATURES0),
        ),
        (
            "FeatureCreator",
            pp.FeatureCreator(),
        ),
        (
            "Winsorizer",
            pp.Winsorizer(numerical_features=config.NUMERICAL_FEATURES2, limits=[0.025, 0.025]),
        ),
        (
            "Chosen",
            pp.ChosenFeatures(columns=config.NUMERICAL_FEATURES_3 + config.CATEGORICAL_FEATURES),
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
            pp.OneHotEncoderProcessor(columns=config.FEATURES_ONE_HOT, prefix="Class"),
        ),
        (
            "DataFrameTypeConverter2",
            pp.DataFrameTypeConverter(conversion_dict=config.CONVERSION_DICT2),
        ),
        (
            "DropColumns2",
            pp.DropColumns(variables_to_drop=config.TO_DROP),
        ),
        (
            "Logistic",
            LogisticRegression(
                solver="liblinear",
                class_weight="balanced",
                penalty="l1",
                max_iter=200,
                C=0.01,
                random_state=RANDOM_SEED,
            ),
        ),
    ]
)
