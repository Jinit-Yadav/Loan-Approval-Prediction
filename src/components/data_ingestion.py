import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig
from src.utils import save_object
from src.logger import logging

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def clean_and_preprocess_data(self, df):
        """Clean and preprocess the loan dataset"""
        logging.info("Starting data cleaning and preprocessing")
        
        try:
            # Strip spaces and standardize column names
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
            logging.info("Standardized column names")

            # Remove leading/trailing spaces and standardize capitalization
            df['loan_status'] = df['loan_status'].str.strip().str.title()       # 'Approved' / 'Rejected'
            df['self_employed'] = df['self_employed'].str.strip().str.title()   # 'Yes' / 'No'
            df['education'] = df['education'].str.strip().str.title()           # 'Graduate' / 'Not Graduate'
            logging.info("Standardized categorical values")

            # Check and log value counts
            logging.info(f"Loan Status value counts: {dict(df['loan_status'].value_counts())}")
            logging.info(f"Self Employed value counts: {dict(df['self_employed'].value_counts())}")
            logging.info(f"Education value counts: {dict(df['education'].value_counts())}")

            # Create total assets and credit category features
            df['total_assets'] = df['residential_assets_value'] + df['commercial_assets_value'] + df['luxury_assets_value'] + df['bank_asset_value']
            
            # Create credit category with proper bin handling
            bins = [0, 500, 600, 700, 850]
            labels = ['Poor', 'Fair', 'Good', 'Excellent']
            df['credit_category'] = pd.cut(df['cibil_score'], bins=bins, labels=labels, include_lowest=True)
            
            logging.info("Created new features: total_assets and credit_category")
            logging.info(f"Credit category value counts: {dict(df['credit_category'].value_counts())}")

            return df

        except Exception as e:
            logging.error("Error during data cleaning and preprocessing")
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the dataset
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')

            # Clean and preprocess the data
            df = self.clean_and_preprocess_data(df)
            logging.info('Data cleaning and preprocessing completed')

            # Create artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved")

            # Train test split
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")
            logging.info(f"Train data shape: {train_set.shape}")
            logging.info(f"Test data shape: {test_set.shape}")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error("Error in data ingestion")
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_array, test_array, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_array, test_array))