import os
import sys
from dataclasses import dataclass  

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]

            # Classification models
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "XGB Classifier": XGBClassifier(),
                "CatBoost Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "SVC": SVC()
            }

            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05],
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                },
                "Logistic Regression": {
                    'C': [0.1, 1, 10, 100],
                    'solver': ['liblinear', 'lbfgs']
                },
                "K-Neighbors Classifier": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                },
                "XGB Classifier": {
                    'learning_rate': [0.1, 0.01, 0.05],
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                },
                "CatBoost Classifier": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [50, 100, 200],
                },
                "AdaBoost Classifier": {
                    'learning_rate': [0.1, 0.01, 0.5],
                    'n_estimators': [50, 100, 200],
                },
                "SVC": {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
            }

            model_report: dict = evaluate_model(X_train=X_train, y_train=y_train, 
                                              X_test=X_test, y_test=y_test, 
                                              models=models, params=params)

            # To get the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))

            # To get the name of the best model
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            # For classification, we typically want higher accuracy (e.g., > 0.7-0.8)
            if best_model_score < 0.7:
                raise CustomException("No sufficiently accurate model found")
                
            logging.info(f"Best model found: {best_model_name} with accuracy score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Make predictions and calculate additional metrics
            predicted = best_model.predict(X_test)
            
            accuracy = accuracy_score(y_test, predicted)
            precision = precision_score(y_test, predicted)
            recall = recall_score(y_test, predicted)
            f1 = f1_score(y_test, predicted)

            logging.info(f"Final model performance:")
            logging.info(f"Accuracy: {accuracy:.4f}")
            logging.info(f"Precision: {precision:.4f}")
            logging.info(f"Recall: {recall:.4f}")
            logging.info(f"F1 Score: {f1:.4f}")

            return {
                'model_name': best_model_name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }

        except Exception as e:
            logging.error("Error in model training")
            raise CustomException(e, sys)