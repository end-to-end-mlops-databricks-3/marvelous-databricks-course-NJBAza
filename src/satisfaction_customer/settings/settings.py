import os
import pickle
from pathlib import Path

# PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
# sys.path.append(str(PACKAGE_ROOT.parent))

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DATAPATH = PACKAGE_ROOT / "data"

FILE_NAME = "satisfaction.csv"
VALIDATION_FILE = "test_data.csv"
TEST_FILE = "satisfaction_prod.csv"
OUTPUT = "satisfaction_prod_predictions.csv"

MODEL_NAME = "classification.pkl"
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT, "trained_models")

TARGET = "satisfaction_v2"
MAP = {"neutral or dissatisfied": 1, "satisfied": 0}

with open(os.path.join(DATAPATH, "CONVERSION_DICT"), "rb") as fp:
    CONVERSION_DICT = pickle.load(fp)

with open(os.path.join(DATAPATH, "TO_CONVERT"), "rb") as fp:
    TO_CONVERT = pickle.load(fp)

with open(os.path.join(DATAPATH, "VARIABLES_TO_DROP"), "rb") as fp:
    VARIABLES_TO_DROP = pickle.load(fp)

with open(os.path.join(DATAPATH, "CATEGORICAL_FEATURES"), "rb") as fp:
    CATEGORICAL_FEATURES = pickle.load(fp)

with open(os.path.join(DATAPATH, "TO_DROP"), "rb") as fp:
    TO_DROP = pickle.load(fp)

with open(os.path.join(DATAPATH, "FEATURES_ENCODE"), "rb") as fp:
    FEATURES_ENCODE = pickle.load(fp)

with open(os.path.join(DATAPATH, "FEATURES_ONE_HOT"), "rb") as fp:
    FEATURES_ONE_HOT = pickle.load(fp)

with open(os.path.join(DATAPATH, "NUMERICAL_FEATURES"), "rb") as fp:
    NUMERICAL_FEATURES = pickle.load(fp)

with open(os.path.join(DATAPATH, "NUMERICAL_FEATURES0"), "rb") as fp:
    NUMERICAL_FEATURES0 = pickle.load(fp)

with open(os.path.join(DATAPATH, "NUMERICAL_FEATURES2"), "rb") as fp:
    NUMERICAL_FEATURES2 = pickle.load(fp)

with open(os.path.join(DATAPATH, "NUMERICAL_CONSIDER"), "rb") as fp:
    NUMERICAL_FEATURES = pickle.load(fp)

with open(os.path.join(DATAPATH, "NUMERICAL_FEATURES_WINSOR"), "rb") as fp:
    NUMERICAL_FEATURES_WINSOR = pickle.load(fp)

with open(os.path.join(DATAPATH, "NUMERICAL_FEATURES_3"), "rb") as fp:
    NUMERICAL_FEATURES_3 = pickle.load(fp)

with open(os.path.join(DATAPATH, "ORIGINAL_FEATURES"), "rb") as fp:
    ORIGINAL_FEATURES = pickle.load(fp)

with open(os.path.join(DATAPATH, "NUMERICAL_FEATURES_CREATED"), "rb") as fp:
    NUMERICAL_FEATURES_CREATED = pickle.load(fp)

with open(os.path.join(DATAPATH, "CONVERSION_DICT2"), "rb") as fp:
    CONVERSION_DICT2 = pickle.load(fp)

with open(os.path.join(DATAPATH, "TO_CONVERT2"), "rb") as fp:
    TO_CONVERT2 = pickle.load(fp)
