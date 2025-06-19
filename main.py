import os
import sys

# Ensure the project root is in the Python path
# This allows absolute imports like 'from src.components.data_ingestion import DataIngestion'
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Insert at the beginning to prioritize

# Now, import your custom modules and components
try:
    from src.logger import logging
    from src.exception import CustomException
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation
    # from src.components.model_trainer import ModelTrainer # Uncomment when you create this

except ImportError as e:
    print(f"Error importing a module: {e}. Please ensure your project structure is correct and __init__.py files are in place.")
    print("Current sys.path:", sys.path)
    sys.exit(1)


if __name__ == "__main__":
    logging.info("ML Project pipeline started from main.py") # Logs to the file created by logger.py

    try:
        # --- Data Ingestion Stage ---
        logging.info("Starting Data Ingestion stage...")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data Ingestion completed.")
        logging.info(f"Train data saved to: {train_data_path}")
        logging.info(f"Test data saved to: {test_data_path}")

        # --- Data Transformation Stage ---
        logging.info("Starting Data Transformation stage...")
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_obj_file_path = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        logging.info("Data Transformation completed.")
        logging.info(f"Preprocessor object saved to: {preprocessor_obj_file_path}")

        # --- Model Training Stage (To be implemented) ---
        # logging.info("Starting Model Training stage...")
        # model_trainer = ModelTrainer()
        # best_model_path = model_trainer.initiate_model_trainer(train_arr, test_arr, preprocessor_obj_file_path)
        # logging.info(f"Model Training completed. Best model saved to: {best_model_path}")

        logging.info("ML Project pipeline finished successfully!")

    except CustomException as e:
        logging.error(f"A custom error occurred during the pipeline execution: {e}")
        sys.exit(1)

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        # You can re-raise as CustomException for consistent error handling
        raise CustomException(e, sys)