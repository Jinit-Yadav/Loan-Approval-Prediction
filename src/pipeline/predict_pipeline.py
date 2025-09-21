import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            # Apply preprocessing
            data_scaled = preprocessor.transform(features)
            
            # Make prediction
            preds = model.predict(data_scaled)
            
            # Convert numerical prediction back to categorical label
            preds_labels = ['Approved' if pred == 1 else 'Rejected' for pred in preds]
            
            return preds_labels
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 no_of_dependents: int,
                 education: str,
                 self_employed: str,
                 income_annum: float,
                 loan_amount: float,
                 loan_term: int,
                 cibil_score: int,
                 residential_assets_value: float,
                 commercial_assets_value: float,
                 luxury_assets_value: float,
                 bank_asset_value: float):
        
        self.no_of_dependents = no_of_dependents
        self.education = education
        self.self_employed = self_employed
        self.income_annum = income_annum
        self.loan_amount = loan_amount
        self.loan_term = loan_term
        self.cibil_score = cibil_score
        self.residential_assets_value = residential_assets_value
        self.commercial_assets_value = commercial_assets_value
        self.luxury_assets_value = luxury_assets_value
        self.bank_asset_value = bank_asset_value

    def clean_input_data(self):
        """Clean and standardize input data similar to training data"""
        # Standardize categorical values
        education = self.education.strip().title()
        self_employed = self.self_employed.strip().title()
        
        # Calculate total assets (same as in training)
        total_assets = (self.residential_assets_value + 
                       self.commercial_assets_value + 
                       self.luxury_assets_value + 
                       self.bank_asset_value)
        
        # Calculate credit category (same as in training)
        if self.cibil_score <= 500:
            credit_category = 'Poor'
        elif self.cibil_score <= 600:
            credit_category = 'Fair'
        elif self.cibil_score <= 700:
            credit_category = 'Good'
        else:
            credit_category = 'Excellent'
        
        return {
            'no_of_dependents': self.no_of_dependents,
            'education': education,
            'self_employed': self_employed,
            'income_annum': self.income_annum,
            'loan_amount': self.loan_amount,
            'loan_term': self.loan_term,
            'cibil_score': self.cibil_score,
            'residential_assets_value': self.residential_assets_value,
            'commercial_assets_value': self.commercial_assets_value,
            'luxury_assets_value': self.luxury_assets_value,
            'bank_asset_value': self.bank_asset_value,
            'total_assets': total_assets,
            'credit_category': credit_category
        }

    def get_data_as_data_frame(self):
        try:
            # Clean and process the input data
            cleaned_data = self.clean_input_data()
            
            custom_data_input_dict = {
                'no_of_dependents': [cleaned_data['no_of_dependents']],
                'education': [cleaned_data['education']],
                'self_employed': [cleaned_data['self_employed']],
                'income_annum': [cleaned_data['income_annum']],
                'loan_amount': [cleaned_data['loan_amount']],
                'loan_term': [cleaned_data['loan_term']],
                'cibil_score': [cleaned_data['cibil_score']],
                'residential_assets_value': [cleaned_data['residential_assets_value']],
                'commercial_assets_value': [cleaned_data['commercial_assets_value']],
                'luxury_assets_value': [cleaned_data['luxury_assets_value']],
                'bank_asset_value': [cleaned_data['bank_asset_value']],
                'total_assets': [cleaned_data['total_assets']],
                'credit_category': [cleaned_data['credit_category']]
            }
            
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)