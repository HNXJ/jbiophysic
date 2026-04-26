# src/jbiophysic/common/utils/logging.py
import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Axis 0: Centralized logger for the jbiophysic package.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '[%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
