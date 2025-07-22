import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.Logger import logging
import traceback
import os 

from src.Utils import save_object, load_object

@dataclass
class DataTransformationConfig:
    preprosessor_obj_file = os.path.join('artifacts', "Preprosessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation     
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[('imputer', SimpleImputer(strategy="median")),
                       ('Standard_scaling', StandardScaler())
                       ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
                    ('standardscaling', StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")


            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ['writing_score','reading_score']
            
            input_feature_train_df=train_data.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_data[target_column_name]    

            input_feature_test_df=test_data.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_data[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )


            input_feature_train_array = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_array, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_array, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")
            file_path =  self.data_transformation_config.preprosessor_obj_file 

            save_object(filepath = file_path,
                        obj=preprocessing_obj
                        )
            return(
                train_arr, test_arr, file_path
            )
        except Exception as e:
            print("FULL TRACEBACK:\n", traceback.format_exc())  
            raise CustomException(e, sys)

