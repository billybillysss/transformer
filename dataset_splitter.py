import random
from utils import *
from config import *
import os
import json


dataset_meta = {"train": 0.7, "validation": 0.2, "test": 0.1}
raw_data_file = os.path.join(RAW_DATA_PATH, RAW_DATA_FILE)


def split_data(meta, data):
    n_line = len(data)
    n_train = int(n_line * meta["train"])
    n_validation = int(n_line * meta["validation"])

    random.shuffle(data)
    train_data = data[:n_train]
    validate_data = data[n_train : n_validation + n_train]
    test_data = data[n_validation + n_train :]
    return {"train": train_data, "validation": validate_data, "test": test_data}


def split_sentence(line):
    pair = line.split("\t")
    return [sentence.strip() for sentence in pair]


raw_data = file_reader(raw_data_file, line_mode=True)
raw_data = [line for line in raw_data if line.strip() != ""]

dataset = split_data(dataset_meta, raw_data)


for type, data in dataset.items():
    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH)
    file_name = f"{type}.json"
    target_path = os.path.join(RAW_DATA_PATH, file_name)
    data = [split_sentence(line) for line in data]
    json_data = json.dumps(data, ensure_ascii=False)
    file_writer(target_path, content=json_data, line_mode=True)
    print("completed")
