import logging
import os
import sys


def get_logger(log_directory, clear_prev_log):
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)

    log_path = os.path.join(log_directory, 'my_log_info.log')
    if os.path.exists(log_path) and clear_prev_log:
        os.remove(log_path)

    fh = logging.FileHandler(log_path)
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s',
                                  datefmt='%a, %d %b %Y %H:%M:%S')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger
