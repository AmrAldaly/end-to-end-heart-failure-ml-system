import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.base import TransformerMixin ,BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler,StandardScaler
from scipy.stats.mstats import winsorize

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, limits=[0, 0.02]):
        self.limits = limits

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            X[col] = winsorize(X[col], limits=self.limits)
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data trnasformation
        
        '''
        try:
            robust_cols= [
                "creatinine_phosphokinase",
                "serum_creatinine",
                "platelets",
            ]
            standard_cols=[
                "age",
                "ejection_fraction",
                "serum_sodium",
                "time",
            ]

            standard_pipeline= Pipeline(
                steps=[
                ("standard",StandardScaler())
                ]
            )

            robust_pipeline=Pipeline(

                steps=[
                ("winsorize", Winsorizer(limits=[0, 0.02])),
                ("robust",RobustScaler())
                ]

            )

            logging.info(f"Robust columns: {robust_cols}")
            logging.info(f"Standard columns: {standard_cols}")

            preprocessor=ColumnTransformer(

                [
                ("standard_pipeline",standard_pipeline,standard_cols),
                ("robust_pipeline",robust_pipeline,robust_cols)
                ],
                remainder="passthrough"
            )
            

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="DEATH_EVENT"

            input_feature_train_df=train_df.drop(columns=[target_column_name])
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name])
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info(
                f"Preprocessor columns order: {preprocessing_obj.get_feature_names_out()}"
            )

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)