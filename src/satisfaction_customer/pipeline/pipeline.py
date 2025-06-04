from sklearn.pipeline import Pipeline

# Assuming 'prediction_model' is in the parent directory
from satisfaction_customer.processing import preprocessing as pp
from satisfaction_customer.settings import settings

RANDOM_SEED = 20230916

transform_pipeline = Pipeline(
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