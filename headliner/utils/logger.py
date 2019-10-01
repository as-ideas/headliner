import sys
import logging


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # stream handler ensures that logging events are passed to stdout
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter('%(message)s')
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

    return logger
