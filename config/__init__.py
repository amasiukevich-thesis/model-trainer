import yaml
from datetime import datetime, timedelta
import os

CONFIG_PATH = os.environ.get("CONFIG_PATH")

def parse_yaml():

    yaml_dict = {}
    with open(CONFIG_PATH, "r") as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print("ERROR")
            print(exc)

    return yaml_dict


yaml_dict = parse_yaml()

MODEL_CONFIG = yaml_dict.get('MODEL_CONFIG')
DATA_MODULE_PARAMS = yaml_dict.get('DATA_MODULE_PARAMS')
TRAINER_PARAMS = yaml_dict.get('TRAINER_PARAMS')

MODEL_CLASS_NAME = yaml_dict.get('MODEL_CLASS_NAME')
TRAIN_LOGS_DIR = yaml_dict.get('TRAIN_LOGS_DIR')
MODEL_SAVE_DIR = yaml_dict.get('MODEL_SAVE_DIR')

TRAIN_FILE_PATH = yaml_dict.get('TRAIN_FILE_PATH')
VAL_FILE_PATH = yaml_dict.get('VAL_FILE_PATH')


print("Hey there")

