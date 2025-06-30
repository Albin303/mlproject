import os
import sys
import dill
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, x_test, y_test, models, param):
    try:
        report = {}
        fitted_models={}
        for model_name, model in models.items():
            para = param[model_name]
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_
            y_test_pred = best_model.predict(x_test)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score
            fitted_models[model_name]=best_model
        return report,fitted_models
    except Exception as e:
        raise CustomException(e, sys)
