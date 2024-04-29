from torch import nn
import math
import torch
from typing import Optional, Tuple
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch
from torch.optim.lr_scheduler import LambdaLR
from utils import *
from config import *
import pytorch_lightning as pl
from data_loader import *


# Function
def attension(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute 'Attention' module from 'Attention is All You Need' paper

    :param query: (batch_size, query_length, d_model)
    :param key: (batch_size, key_length, d_model)
    :param value: (batch_size, value_length, d_model)
    :param mask: (batch_size, query_length, key_length)
    :param dropout: nn.Module
    :return: output, attention weights of each head
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


# class


class Embeddings(nn.Module):
    """Embedding module with strongly typed methods and docstrings.

    Arguments:
        vocab_size: The number of unique tokens in the input (int)
        d_model: The number of expected features in the input (int)

    Returns:
        x: The embedded input as a tensor of shape (batch_size, seq_len, d_model)
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the embedded input as a tensor of shape (batch_size, seq_len, d_model).

        Arguments:
            x: The input sequence (torch.tensor) of shape (batch_size, seq_len)

        Returns:
            x: The embedded input of shape (batch_size, seq_len, d_model)
        """
        # N~(0,1/d_model) the bigger the d_model is, the smaller the variance is. Multiplying
        return self.emb(x) * math.sqrt(self.d_model)


class PositionEncoding(nn.Module):
    """
    PositionEncoding module from 'Attention is All You Need' paper

    Arguments:
        d_model: The number of expected features in the input (int)
        max_length: Maximum length of input sequence (int)
        dropout: The dropout value (float)

    Returns:
        x: Final output of the PositionEncoding (torch.tensor)
    """

    def __init__(
        self, d_model: int, max_length: int = 5000, dropout: float = 0.1
    ) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even number")

        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1).to(dtype=torch.float)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -math.log(10000)
        )
        pe[:, ::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add one dimension for the number of batch
        pe = pe.unsqueeze(0)
        # Register pe as parameter without calculating the gradient
        # why not use self.pe = pe? we do not want to calculated pe on the fly and use the calculated result. And make sure pe will be moved to the 'device' if implement 'to_device()'

        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PositionEncoding

        Arguments:
            x: The input sequence (torch.tensor)

        Returns:
            x: Final output of the PositionEncoding (torch.tensor)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module from 'Attention is All You Need' paper

    :param int d_model: dimension of model
    :param int n_head: number of attention heads
    :param float dropout: dropout rate
    """

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(
                f"d_model({d_model}) should be divided by n_head({n_head}) evenly(no remainder)"
            )
        self.d_k = d_model // n_head  # the number of key dimension for each head
        self.n_head = n_head
        self.dropout = nn.Dropout(dropout)

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.linear = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Multi-Head Attention

        :param torch.Tensor query: (batch_size, query_length, d_model)
        :param torch.Tensor key: (batch_size, key_length, d_model)
        :param torch.Tensor value: (batch_size, value_length, d_model)
        :param torch.Tensor mask: (batch_size, query_length, key_length)
        :return: output, attention weights of each head
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        residual = query
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)
        query = (
            self.w_q(query).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        )
        key = self.w_k(key).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        value = (
            self.w_v(value).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        )
        feature, p_attn = attension(query, key, value, mask, self.dropout)
        feature = feature.transpose(1, 2).reshape(
            batch_size, -1, self.n_head * self.d_k
        )

        # Normalization and residual connection handling
        return self.norm(self.linear(feature) + residual)


class FeedForwardNeuralNetwork(nn.Module):
    """
    Feed Forward Neural Network

    Arguments:
        d_model: The number of expected features in the input (int)
        d_ff: The hidden dimension of the feed forward network model (int)
        dropout: The dropout value (float)

    Returns:
        torch.Tensor: The output tensor of shape (N, d_model)

    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the feed forward neural network.

        Arguments:
            x: The input tensor of shape (N, d_model)

        Returns:
            torch.Tensor: The output tensor of shape (N, d_model)

        """
        residual = x
        x = self.relu(self.w1(x))
        x = self.dropout(self.w2(x))
        return self.norm(x + residual)


class EncoderLayer(nn.Module):
    """
    Encoder Layer module

    Arguments:
        d_model: The number of expected features in the input (int)
        n_head: The number of heads in the multi-head attention models (int)
        d_ff: The hidden dimension of the feed forward network model (int)
        dropout: The dropout value (float)

    Returns:
        torch.tensor: The output of the encoder layer
    """

    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.multi_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ffn = FeedForwardNeuralNetwork(d_model, d_ff, dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # multi-head attension and feed forward neural network
        feature = self.multi_attn(x, x, x, mask)
        return self.ffn(feature)


class Encoder(nn.Module):
    """
    Encoder module

    Arguments:
        vocab_size: The size of the vocabulary (int)
        d_model: The number of expected features in the input (int)
        n_head: The number of heads in the multi-head attention models (int)
        d_ff: The dimension of the feed forward network model (int)
        max_length: Maximum length of input sequence (int)
        n_layer: Number of layers in the Encoder (int)
        dropout: The dropout value (float)

    Returns:
        x: Final output of the Encoder (torch.tensor)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_head: int,
        d_ff: int,
        max_length: int = 5000,
        n_layer: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.emb = Embeddings(vocab_size, d_model)
        self.pe = PositionEncoding(
            d_model=d_model, max_length=max_length, dropout=dropout
        )
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_head, d_ff) for _ in range(n_layer)]
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the Encoder

        Arguments:
            x: The input sequence (torch.tensor)
            mask: Mask for input sequence (torch.tensor)

        Returns:
            x: Final output of the Encoder (torch.tensor)
        """
        x = self.emb(x)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecoderLayer(nn.Module):
    """
    Multi-head attention layer with feed forward neural network

    Arguments:
        d_model: The number of expected features in the input (required).
        n_head: The number of heads in the multi-head attention models (required).
        d_ff: The dimension of the feed forward network model (required).
        dropout: The dropout value (optional). Default=0.1.
    """

    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.multi_self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.multi_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ffn = FeedForwardNeuralNetwork(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        # multi-head attention and feed forward neural network
        feature = self.multi_self_attn(x, x, x, mask)
        feature = self.multi_attn(x, memory, memory, src_mask)
        return self.ffn(feature)


class Decoder(nn.Module):
    """
    Decoder module

    Arguments:
        vocab_size: The size of the input vocabulary (required).
        d_model: The number of expected features in the input (required).
        n_head: The number of heads in the multi-head attention models (required).
        d_ff: The dimension of the feed forward network model (required).
        max_length: The maximum length of the input sequence (optional). Default=5000.
        n_layer: The number of decoder layers to include in the model (optional). Default=6.
        dropout: The dropout value (optional). Default=0.1.

    Returns:
        x: The output of the decoder as a tensor of shape (batch_size, max_length, d_model)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_head: int,
        d_ff: int,
        max_length: int = 5000,
        n_layer: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # self.sub_mask = get_subsequent_mask(vocab_size)
        self.emb = Embeddings(vocab_size, d_model)
        self.pe = PositionEncoding(d_model, max_length, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_head, d_ff) for _ in range(n_layer)]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the decoder

        Arguments:
            x: The input tensor of shape (batch_size, max_length)
            mask: The padding mask of shape (batch_size, max_length)
            memory: The output of the encoder of shape (batch_size, max_length, d_model)
            src_mask: The padding mask for the encoder output of shape (batch_size, max_length)

        Returns:
            x: The output of the decoder as a tensor of shape (batch_size, max_length, d_model)
        """

        x = self.emb(x)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask, memory, src_mask)
        return x


class Generator(nn.Module):
    """
    Generates a probability distribution over the target vocabulary given the
    decoder's final state.
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        Initializes the generator.

        Arguments:
            d_model: The number of expected features in the decoder's final state.
            vocab_size: The size of the target vocabulary.
        """
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates a probability distribution over the target vocabulary given the
        input tensor.

        Arguments:
            x: The input tensor of shape (batch_size, d_model)

        Returns:
            A tensor of shape (batch_size, vocab_size) representing a probability
            distribution over the target vocabulary.
        """
        return torch.softmax(self.linear(x), dim=-1)
        # return self.linear(x)


class Transformer(pl.LightningModule):
    def __init__(
        self,
        src_vocab_size,
        tar_vocab_size,
        d_model,
        n_head,
        d_ff,
        max_length=5000,
        n_layer=6,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = Encoder(
            src_vocab_size, d_model, n_head, d_ff, max_length, n_layer, dropout
        )
        self.decoder = Decoder(
            tar_vocab_size, d_model, n_head, d_ff, max_length, n_layer, dropout
        )
        self.generator = Generator(d_model, tar_vocab_size)
        self.loss_fn = CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=SMOOTHING)
        self.save_hyperparameters()

    def forward(self, src_x, src_mask, tar_x, tar_mask):
        memory = self.encoder(src_x, src_mask)
        output = self.decoder(tar_x, tar_mask, memory, src_mask)
        return self.generator(output)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=BATCH_SIZE
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=BATCH_SIZE
        )
        return loss

    def _step(self, batch, batch_idx):
        src_x, src_mask, tar_x, tar_mask, tar_y, tar_text = batch
        output = self.forward(src_x, src_mask, tar_x, tar_mask)
        loss = self.loss_fn(output.reshape(-1, output.shape[-1]), tar_y.reshape(-1))
        return loss

    def on_train_epoch_start(self):
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        print(f"Current Learning Rate: {cur_lr}")

    def on_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get("train_loss")
        val_loss = self.trainer.callback_metrics.get("val_loss")
        # val_bleu = self.trainer.callback_metrics.get("val_bleu")
        print(
            f"Epoch {self.current_epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, BLEU: {val_bleu}"
        )

    # def prepare_data(self):
    #     dataset_dict = load_dataset(100,0)
    #     self.train_ds = dataset_dict["train"]
    #     self.val_dataset = dataset_dict["validation"]

    # def train_dataloader(self):
    #     self.train_dataset = data.DataLoader(
    #         self.train_ds, BATCH_SIZE, shuffle=True, collate_fn=self.train_ds.collate_fn
    #     )
    #     return self.train_dataset

    # def val_dataloader(self):
    #     self.val_dataset = data.DataLoader(
    #         self.val_dataset,
    #         BATCH_SIZE,
    #         shuffle=False,
    #         collate_fn=self.val_dataset.collate_fn,
    #         num_workers=15,
    #         persistent_workers=True,
    #     )
    #     return self.val_dataset

    # def on_validation_epoch_end(self):
    #     cur_bleu = evaluate(self.val_dataset, self, OUTPUT_MAX_LENGTH)
    #     self.log("val_bleu", cur_bleu, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=LR)
        lr_scheduler = LambdaLR(
            optimizer, lr_lambda=lambda step: lr_lambda_fn(step, EPOCH / 4)
        )
        return ([optimizer], [lr_scheduler])


def init_model(
    src_vocab_size: int,
    tar_vocab_size: int,
    d_model: int,
    n_head: int,
    d_ff: int,
    max_length: int = 5000,
    n_layer: int = 6,
    dropout: float = 0.1,
) -> Transformer:
    """
    Init a Transformer model with Xavier uniform initialization.

    Arguments:
        src_vocab_size: The number of unique source tokens.
        tar_vocab_size: The number of unique target tokens.
        d_model: The number of expected features in the input (int)
        n_head: The number of attention heads.
        d_ff: The dimension of the feed forward network.
        max_length: The maximum length of input sequence.
        n_layer: The number of transformer layers.
        dropout: The dropout rate.

    Returns:
        A Transformer model.
    """
    model = Transformer(
        src_vocab_size,
        tar_vocab_size,
        d_model,
        n_head,
        d_ff,
        max_length,
        n_layer,
        dropout,
    )
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
    return model
