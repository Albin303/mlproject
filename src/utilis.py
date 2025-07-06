# some general code that we will use across the entire project(like reading the dataset)
import os,sys

import dill
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params, cv=3, n_jobs=3, verbose=1, refit=False):
    try:
        report = {}
        fitted_models = {}

        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = params.get(model_name, {})

            gs = GridSearchCV(model, para, cv=cv, n_jobs=n_jobs, verbose=verbose, refit=refit)
            gs.fit(X_train, y_train)

            # Set best parameters
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Save report and model
            report[model_name] = test_model_score
            fitted_models[model_name] = model

        return report, fitted_models  # ✅ Now returning both

    except Exception as e:
        raise CustomException(e, sys)

    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)