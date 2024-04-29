from config import *
from utils import *
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
import json
import torch
from typing import List, Tuple
from tokenizer import get_tokenizer
import math


class Dataset(data.Dataset):

    def __init__(self, path, n_partition, idx) -> None:
        super().__init__()
        _data = json.loads(file_reader(path))
        self.dataset = self._partition(_data, n_partition, idx)
        self.src_tokenizer = get_tokenizer("src")
        self.tar_tokenizer = get_tokenizer("tar")

    def _partition(self, data, n_partition, idx):

        n_data = len(data)
        size = math.ceil(n_data / n_partition)
        print(size)
        start = idx * size
        end = (idx + 1) * size if idx < n_partition else n_data
        return data[start:end]

    def collate_fn(self, batch: List[Tuple[List[int], List[int], str]]) -> Tuple[
        torch.LongTensor,
        torch.BoolTensor,
        torch.LongTensor,
        torch.BoolTensor,
        torch.LongTensor,
        List[str],
    ]:
        """
        Collate function for batching.

        Args:
            batch: List[Tuple[List[int], List[int], str]]

        Returns:
            Tuple[
                torch.LongTensor, torch.BoolTensor, torch.LongTensor, torch.BoolTensor, torch.LongTensor, List[str]
            ]
        """
        batch_src, batch_tar, tar_text = zip(*batch)

        src_x = pad_sequence(
            special_char_wrapper(batch_src),
            True,
            PAD_ID,
        ).to(DEVICE)
        src_mask = get_padding_mask(src_x, PAD_ID).to(DEVICE)
        tar_f = pad_sequence(
            special_char_wrapper(batch_tar),
            True,
            PAD_ID,
        ).to(DEVICE)

        tar_x = tar_f[:, :-1]

        tar_mask = get_padding_mask(tar_x, PAD_ID).to(DEVICE)
        tar_subsquent_mask = get_subsequent_mask(tar_x.size(1)).to(DEVICE)
        tar_mask = tar_mask | tar_subsquent_mask
        tar_y = tar_f[:, 1:]
        return src_x, src_mask, tar_x, tar_mask, tar_y, tar_text

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        src_text, tar_text = self.dataset[index]
        src_text = src_text.lower()
        source = self.src_tokenizer.encode(src_text)
        target = self.tar_tokenizer.encode(tar_text)
        return source, target, tar_text


def load_dataset(n_paritiion, idx):
    return {
        "train": Dataset(TRAIN_DATASET_PATH, n_paritiion, idx),
        "validation": Dataset(VALIDATION_DATASET_PATH, n_paritiion, idx),
        "test": Dataset(TEST_DATASET_PATH, n_paritiion, idx),
    }
