import os
import torch

TRAIN_DATASET_PATH = "./data/raw/train.json"
VALIDATION_DATASET_PATH = "./data/raw/validation.json"
TEST_DATASET_PATH = "./data/raw/test.json"

RAW_DATA_PATH = "./data/raw"
RAW_SRC_FILE = "src.txt"
RAW_TAR_FILE = "tar.txt"
# TRAIN_DATASET_PATH = "./data/sample_raw/train.json"
# VALIDATION_DATASET_PATH = "./data/sample_raw/validation.json"
# TEST_DATASET_PATH = "./data/sample_raw/test.json"

# RAW_DATA_PATH = "./data/sample_raw"
RAW_DATA_FILE = "dataset.tsv"


TOKENIZER_DIR = "./data/tokenizer"
TOKENIZER_SRC_MODEL = 'src-sp'
TOKENIZER_TAR_MODEL = 'tar-sp'
ENCODING="bpe"
TOKENIZER_VOCAB_SIZE = 5000

MODEL_DIR = "./data/checkpoints"
MODEL_NAME = "/best_model_news_test.pth"

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


N_HEAD = 8
N_LAYER = 6
DROPOUT = 0.1
BATCH_SIZE = 100

D_MODEL = 512
LR = 1e-5
INPUT_MAX_LENGTH = 5000
OUTPUT_MAX_LENGTH = 100
EPOCH = 400
D_FF = 2048
SMOOTHING = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

