import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
from src.utilis import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def intiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("splitting training data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "LinearRegression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False),
                "AdaBoostRegressor": AdaBoostRegressor()
            }

            # HYPERPARAMETER TUNING
            params = {
                "LinearRegression": {},
                "Lasso": {},
                "Ridge": {},
                "KNeighborsRegressor": {"n_neighbors": [5, 7, 9, 11]},
                "RandomForestRegressor": {"n_estimators": [8, 16, 32, 64, 128, 256]},
                "DecisionTreeRegressor": {"criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"]},
                "GradientBoostingRegressor": {
                    "learning_rate": [.1, .01, .05, .001],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "XGBRegressor": {
                    "learning_rate": [.1, .01, .05, .001],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "CatBoostRegressor": {
                    "depth": [6, 8, 10],
                    "iterations": [30, 50, 100]
                },
                "AdaBoostRegressor": {
                    "learning_rate": [.1, .01, 0.5, .001],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                }
            }

            model_report ,fitted_models = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                x_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = fitted_models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best Model found")

            logging.info("Best Found model on both training and testing dataset")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
