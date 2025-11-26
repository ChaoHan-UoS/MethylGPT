import logging
import sys
from pathlib import Path


formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")


def setup_logger(name, log_file, level=logging.INFO):
    """
    Create a named logger and attach a file handler to write logs to `log_file`.
    """
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False

    return logger


def add_console_handler(logger):
    """
    Add a console (stdout) handler to the given logger to also output logs to the console.
    """
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

