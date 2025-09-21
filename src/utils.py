import os
import sys 
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        model_report = {}

        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            param = params[model_name]
            
            logging.info(f"Training {model_name} with hyperparameter tuning...")
            
            # Perform grid search for hyperparameter tuning
            gs = GridSearchCV(model, param, cv=3, scoring='accuracy', n_jobs=-1)
            gs.fit(X_train, y_train)

            # Set the best parameters and train the model
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Make predictions
            y_test_pred = model.predict(X_test)

            # Calculate accuracy score (primary metric for model selection)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            # Store additional metrics for reference
            test_precision = precision_score(y_test, y_test_pred)
            test_recall = recall_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred)

            model_report[model_name] = test_accuracy

            logging.info(f"{model_name} - Best params: {gs.best_params_}")
            logging.info(f"{model_name} - Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

        return model_report

    except Exception as e:
        logging.error("Error in model evaluation")
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)