import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, Y_train, x_test, y_test, models):
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(X_train, Y_train)  # Use parameters as defined
            Y_train_pred = model.predict(X_train)
            Y_test_pred = model.predict(x_test)
            train_model_score = r2_score(Y_train, Y_train_pred)
            test_model_score = r2_score(y_test, Y_test_pred)
            report[model_name] = test_model_score  # Correctly add to dict
        return report  # Return after all models are evaluated
    except Exception as e:
        raise CustomException(e, sys)
