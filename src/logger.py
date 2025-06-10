import logging
import os
from datetime import datetime

# Generate a log file name with timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y+%H,%M,%S')}.log"

# Create 'logs' folder in the current working directory
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

# Create full path for the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure logging
logging.basicConfig(
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),  # Log to file
        logging.StreamHandler()              # Log to console
    ]
)

