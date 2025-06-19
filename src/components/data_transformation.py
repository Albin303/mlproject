import sys
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utilis import save_object 


class DataTransformationConfig:
    preprocessor_obj_file=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        logging.info("Obtaining data transformer object") # Added a log for clarity
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = ["gender", "race_ethnicity", "arental_level_of_education", "lunch", "test_preparation_course"]

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])
            logging.info("Data transformer object obtained successfully.") # Added a log for clarity
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        logging.info("Entered the Data Transformation method/component") # Added a log for clarity
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            # Standardize column names to snake_case
            train_df.columns = train_df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("/", "_")
            test_df.columns = test_df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("/", "_")

            logging.info("Read train and test data for transformation.") # Added a log for clarity

            preprocessor_obj = self.get_data_transformer_object()
            target_column_name = "math_score"

            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            logging.info("Splitting features and target columns.") # Added a log for clarity

            input_feature_train_array = preprocessor_obj.fit_transform(input_features_train_df)
            input_feature_test_array = preprocessor_obj.transform(input_features_test_df)
            logging.info("Applying preprocessor object on training and testing data.") # Added a log for clarity


            train_arr = np.c_[input_feature_train_array, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_array, np.array(target_feature_test_df)]
            logging.info("Combined processed features with target column.") # Added a log for clarity


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file,
                obj=preprocessor_obj
            )
            logging.info(f"Preprocessor object saved to: {self.data_transformation_config.preprocessor_obj_file}") # Added a log for clarity

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file
            )

        except Exception as e:
            raise CustomException(e, sys)