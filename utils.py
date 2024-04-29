import re
from config import *
from tokenizer import get_tokenizer
import sacrebleu
import os
import glob
import torch

# import torch
# from snownlp import SnowNLP


def file_reader(path, line_mode=False):
    with open(path, "r", encoding="utf-8") as f:
        if line_mode:
            content = f.readlines()
        else:
            content = f.read()
    return content


def file_writer(path, content, mode="w", line_mode=False):
    with open(path, mode, encoding="utf-8") as f:
        if line_mode:
            f.writelines(content)

        else:
            f.write(content)


def get_padding_mask(x, padding_idx):
    return (x == padding_idx).unsqueeze(1)


def get_subsequent_mask(size):
    return (1 - torch.tril(torch.ones(1, size, size))).bool()


def bleu_score(hyp, refs):
    bleu = sacrebleu.corpus_bleu(hyp, refs, tokenize="zh")
    return round(bleu.score, 3)


def batch_greedy_decode(model, src_x, src_mask, max_length=50):
    tar_tokenizer = get_tokenizer("tar")

    memory = model.encoder(src_x, src_mask)

    # initiate that all sentence start with BOS_ID
    predict_x = torch.tensor([[BOS_ID]] * src_x.size(0)).to(DEVICE)
    for _ in range(max_length):
        predict_mask = get_padding_mask(predict_x, PAD_ID)
        output = model.decoder(predict_x, predict_mask, memory, src_mask)
        output = model.generator(output[:, -1, :])
        predict = torch.argmax(output, dim=-1, keepdim=True)
        predict_x = torch.concat([predict_x, predict], dim=-1)

        # Stop generating if all the sentence includes EOS_ID
        if torch.all(predict_x == EOS_ID).item():
            break

    batch_predict_text = []
    for predict in predict_x:
        predict_id = []

        for id in predict:
            if id == BOS_ID:
                continue
            elif id == EOS_ID:
                break
            predict_id.append(id.item())
        # print(predict_id)
        batch_predict_text.append(tar_tokenizer.decode(predict_id))

    return batch_predict_text


def print_memory():
    # get number of GPU
    num_gpus = torch.cuda.device_count()
    print(f"num of GPU: {num_gpus}")
    # check the usage of the GPUs
    for i in range(num_gpus):
        gpu = torch.cuda.get_device_name(i)
        utilization = round(torch.cuda.max_memory_allocated(i) / 1024**3, 2)
        print(f"GPU {i}: {gpu}, Memory Utilization: {utilization} GB")


def lr_lambda_fn(step, warmup):
    lr = step / warmup * 10 if step <= warmup else warmup / step * 10
    return max(lr, 0.1)


def evaluate(loader, model, max_length=50):
    tar_sentence = []
    predict_sentence = []
    for src_x, src_mask, tar_x, tar_mask, tar_y, tar_text in loader:

        batch_prob_text = batch_greedy_decode(model, src_x, src_mask, max_length)
        tar_sentence += tar_text
        predict_sentence += batch_prob_text

    print(tar_sentence)
    print(predict_sentence)
    return bleu_score(predict_sentence, [tar_sentence])


def latest_checkpoint_path():
    path = "lightning_logs"
    folders = os.listdir(path)
    if len(folders) == 0:
        return None
    folders.sort(reverse=True)
    for folder in folders:
        latest_folder = folders[0]
        ckpt_list = glob.glob(f"./{path}/{folder}/checkpoints/*ckpt")
        if len(ckpt_list) > 0:
            return ckpt_list[-1]
    return None

def special_char_wrapper(batch_src):
    return [torch.LongTensor([BOS_ID] + src + [EOS_ID]) for src in batch_src]