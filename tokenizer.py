import sentencepiece as spm
from config import *
import os


def get_tokenizer(kind):
    if kind == "src":
        model_name = TOKENIZER_SRC_MODEL + ".model"
    elif kind == "tar":
        model_name = TOKENIZER_TAR_MODEL + ".model"
    else:
        raise KeyError("Proivde 'src' or 'tar'")

    model_path = os.path.join(TOKENIZER_DIR, model_name)
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(model_path)
    return tokenizer
