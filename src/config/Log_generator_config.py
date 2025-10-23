from multiprocessing import Pool
import numpy as np
import os
import SimpleITK as sitk
import argparse
import tqdm
import logging


class LogGenerator:
    """
    Generates a logger for processing log files.
    """

    def __init__(self,
                 log_file_name: str,  # log file name
                 logger_topic: str = "RPTK",  # repeated log topic
                 log_format: str = '%(asctime)s %(name)s - %(levelname)-8s: %(message)s'  # log format
                 ):

        self.log_file_name = log_file_name
        self.logger_topic = logger_topic
        self.log_format = log_format

    def generate_log(self):

        ### Config Logger ###
        # create file handler which logs info messages
        logging.basicConfig(filename=self.log_file_name,
                           filemode='w',
                           level=logging.INFO,
                           format=self.log_format,
                           datefmt='%m/%d/%Y %I:%M:%S %p'
                           )
        # create logger
        logger = logging.getLogger(self.logger_topic)
        print("Logging for RPTK:", self.log_file_name)
        formatter = logging.Formatter(self.log_format)
        handler = logging.FileHandler(self.log_file_name)
        handler.setFormatter(formatter)

        logger.addHandler(handler)

        return logger

