import os
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.logger import logging
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    train_csv=os.path.join('artifacts','train.csv')
    test_csv=os.path.join('artifacts','test.csv')
class DataTransformation(DataTransformationConfig):
    def __init__(self):
        super().__init__()
        
    def get_data_transformer_object(self):
        try:
            train_df=pd.read_csv(self.train_csv)
            numerical_cols=list(train_df.select_dtypes(include='int').columns[1:])
            categorical_cols=list(train_df.select_dtypes(include='object').columns[1:])
            num_pipeline=Pipeline(steps=[('imputer',SimpleImputer(strategy="median")),
                                         ('scaler',StandardScaler())])

            cat_pipeline=Pipeline(steps=[('imputer',SimpleImputer(strategy="most_frequent")),
                                         ('one_hot_encoder',OneHotEncoder()),
                                         ('scaler',StandardScaler(with_mean=False))])
            logging.info(f'Numerical columns standard scaling completed:{numerical_cols}')
            logging.info(f'Categorical columns encoding completed:{categorical_cols}')

            merge_preprocess=ColumnTransformer([("num_pipeline",num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)])
            return merge_preprocess
        except Exception as error:
            raise CustomException(error,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info('Read train and test data')
            logging.info('Obtaining preprocessing object')
            preprocessing_obj=self.get_data_transformer_object()
            numeric_columns=list(train_df.select_dtypes(include='int').columns)
            target_column_name=numeric_columns[0]
            numerical_cols=numeric_columns[1:]
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info('Applying preprocessing object on training dataframe and testing datafreame')

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            logging.info('Saved preprocessing object.')

            save_object(file_path=self.preprocess_obj_file_path,
                     obj=preprocessing_obj)
            return (train_arr,test_arr,self.preprocess_obj_file_path)
        except Exception as error:
            raise CustomException(error,sys)
if __name__=="__main__":
    c=DataTransformation()
    c.get_data_transformer_object()