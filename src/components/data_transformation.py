import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            # Define numerical and categorical columns for loan dataset
            numerical_columns = [
                'no_of_dependents', 
                'income_annum', 
                'loan_amount', 
                'loan_term', 
                'cibil_score',
                'residential_assets_value', 
                'commercial_assets_value', 
                'luxury_assets_value', 
                'bank_asset_value',
                'total_assets'  # Include the created feature
            ]
            
            categorical_columns = [
                'education', 
                'self_employed', 
                'credit_category'  # Include the created feature
            ]
            
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),  # Use median for numerical features
                    ('scaler', StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),  # Use most_frequent for categorical
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),  # Handle unknown categories
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f'Numerical columns: {numerical_columns}')
            logging.info(f'Categorical columns: {categorical_columns}')
            
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)   

    def initiate_data_transformation(self, train_path, test_path):      
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            logging.info(f"Train data shape: {train_df.shape}")
            logging.info(f"Test data shape: {test_df.shape}")

            # Ensure data consistency - apply the same cleaning as in ingestion
            for df in [train_df, test_df]:
                # Strip spaces and standardize column names (if needed)
                df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
                
                # Standardize categorical values
                if 'loan_status' in df.columns:
                    df['loan_status'] = df['loan_status'].str.strip().str.title()
                if 'self_employed' in df.columns:
                    df['self_employed'] = df['self_employed'].str.strip().str.title()
                if 'education' in df.columns:
                    df['education'] = df['education'].str.strip().str.title()
                if 'credit_category' in df.columns:
                    df['credit_category'] = df['credit_category'].astype(str).str.strip().str.title()

            logging.info("Data cleaning applied to train and test sets")

            # Log value counts for verification
            logging.info(f"Loan Status value counts in train: {dict(train_df['loan_status'].value_counts())}")
            logging.info(f"Self Employed value counts in train: {dict(train_df['self_employed'].value_counts())}")
            logging.info(f"Education value counts in train: {dict(train_df['education'].value_counts())}")
            logging.info(f"Credit Category value counts in train: {dict(train_df['credit_category'].value_counts())}")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()
            
            # Target column is 'loan_status' for classification
            target_column_name = 'loan_status'
            
            # Define which columns to drop (excluding the target and ID columns)
            columns_to_drop = ['loan_id']  # Add any other columns you want to exclude
            
            # Prepare input features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name] + columns_to_drop, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name] + columns_to_drop, axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")
            
            # Apply preprocessing
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Convert target to numerical (if not already done)
            target_feature_train_df = target_feature_train_df.map({'Approved': 1, 'Rejected': 0})
            target_feature_test_df = target_feature_test_df.map({'Approved': 1, 'Rejected': 0})

            # Combine features and target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")
            
            # Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info(f"Final train array shape: {train_arr.shape}")
            logging.info(f"Final test array shape: {test_arr.shape}")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.error("Error in data transformation")
            raise CustomException(e, sys)