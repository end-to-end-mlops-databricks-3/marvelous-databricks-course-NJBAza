import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from satisfaction_customer.settings import settings

RANDOM_SEED = 20230916


def separate_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split input DataFrame into features (X) and target (y).

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset containing features and target.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Features and target variable as separate objects.

    """
    X = data.drop(settings.TARGET, axis=1)
    y = data[settings.TARGET]
    return X, y


def load_dataset(file_name: str) -> pd.DataFrame:
    """Load dataset from CSV file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    """
    filepath = os.path.join(settings.DATAPATH, file_name)
    _data = pd.read_csv(filepath)
    return _data


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into training and testing sets.

    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Target variable.
    test_size : float
        Proportion of the dataset to include in the test split.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test

    """
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED)


def save_pipeline(pipeline_to_save: Pipeline) -> None:
    """Serialize and save the trained pipeline to disk.

    Parameters
    ----------
    pipeline_to_save : Pipeline
        Trained sklearn pipeline object to save.

    """
    save_path = os.path.join(settings.SAVE_MODEL_PATH, settings.MODEL_NAME)
    joblib.dump(pipeline_to_save, save_path)
    print(f"Model has been saved under the name {settings.MODEL_NAME}")


def load_pipeline(pipeline_to_load: str) -> Pipeline:
    """Load a trained pipeline from disk.

    Parameters
    ----------
    pipeline_to_load : str
        Name of the model file to load (ignored, kept for API compatibility).

    Returns
    -------
    Pipeline
        Loaded sklearn pipeline.

    """
    save_path = os.path.join(settings.SAVE_MODEL_PATH, settings.MODEL_NAME)
    model_loaded = joblib.load(save_path)
    print("Model has been loaded")
    return model_loaded
