# logger_config.py
from logging import Logger


import logging
import sys


def setup_logger(name: str = "root", level: int = logging.INFO) -> Logger:
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding handlers multiple times
    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)

        # Optional: add file handler for persistent logging
        file_handler = logging.FileHandler("application.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
