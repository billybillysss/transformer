import sentencepiece as spm
from config import *
import os


# train tokenizer
src_path = os.path.join(RAW_DATA_PATH, RAW_SRC_FILE)
tar_path = os.path.join(RAW_DATA_PATH, RAW_TAR_FILE)


paths = []


spm.SentencePieceTrainer.train(
    input=src_path,
    model_prefix=TOKENIZER_SRC_MODEL,
    model_type=ENCODING,
    vocab_size=TOKENIZER_VOCAB_SIZE,
    pad_id=PAD_ID,
    bos_id=BOS_ID,
    eos_id=EOS_ID,
    unk_id=UNK_ID,
)
spm.SentencePieceTrainer.train(
    input=tar_path,
    model_prefix=TOKENIZER_TAR_MODEL,
    model_type=ENCODING,
    vocab_size=TOKENIZER_VOCAB_SIZE,
    pad_id=PAD_ID,
    bos_id=BOS_ID,
    eos_id=EOS_ID,
    unk_id=UNK_ID,
)
