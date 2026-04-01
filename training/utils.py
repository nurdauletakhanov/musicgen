import logging
import datetime

def print_logger(save_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    logger.propagate = False

    name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    handler = logging.FileHandler("{}/{}.log".format(save_path, name))
    console = logging.StreamHandler()

    logger.addHandler(handler)
    logger.addHandler(console)

    logger.info("Start logging...")
    return logger