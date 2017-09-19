#coding=utf-8
import os
import logging
from logging.handlers import TimedRotatingFileHandler
log_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../log/')

logger = logging.getLogger()
log_path = os.path.join(log_dir, 'service_log.txt')
hdlr = TimedRotatingFileHandler(log_path, when='H', interval=1, backupCount=40)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

def write_info(msg):
    logger.info(msg)

def write_error(msg):
    logger.error(msg)
