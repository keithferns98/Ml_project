import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.logger import logging
from src.exception import CustomException
import warnings
import os

warnings.filterwarnings("ignore")


@dataclass
class DataTransformerConfig:
    preprocessor_obj_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation(DataTransformerConfig):
    def __init__(self):
        super().__init__()
        self.train_data = pd.read_csv(os.path.join("artifacts", "train.csv"))
        self.test_data = pd.read_csv(os.path.join("artifacts", "test.csv"))

    def get_data_transformer_object(self):
        """Data Transformation"""
        try:
            num_features = self.train_data.select_dtypes(exclude="objects").columns
            cat_features = self.test_data.select_dtypes(include="object").columns

            num_pipeline = Pipeline([()])
        except:
            pass
