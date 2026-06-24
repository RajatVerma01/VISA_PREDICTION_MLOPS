import logging
import os
from from_root import from_root
from datetime import datetime

#
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"


log_dir_path = os.path.join(from_root(), 'logs')


os.makedirs(log_dir_path, exist_ok=True)


logs_path = os.path.join(log_dir_path, LOG_FILE)


logging.basicConfig(
    filename=logs_path,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG
)