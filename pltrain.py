from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from model import *
from config import *
from utils import *
from data_loader import *
from tokenizer import get_tokenizer

checkpoint_callback = ModelCheckpoint(
    mode="max",
    save_top_k=1,
    every_n_epochs=5,
    filename="model-{epoch:03d}",
    verbose=True,
)

if __name__ == "__main__":
    SRC_VOCAB_SIZE = get_tokenizer("src").vocab_size()
    TAR_VOCAB_SIZE = get_tokenizer("tar").vocab_size()

    n_partition = 25

    for idx in range(1,n_partition):

        dataset_dict = load_dataset(n_partition, idx)
        train_loader = data.DataLoader(
            dataset_dict["train"],
            BATCH_SIZE,
            shuffle=True,
            collate_fn=dataset_dict["train"].collate_fn,
            num_workers=19,
            persistent_workers=True,
        )
        val_loader = data.DataLoader(
            dataset_dict["validation"],
            BATCH_SIZE,
            shuffle=False,
            collate_fn=dataset_dict["validation"].collate_fn,
            num_workers=19,
            persistent_workers=True,
        )

        ckpt_path = latest_checkpoint_path()
        if ckpt_path is not None:
            model = Transformer.load_from_checkpoint(ckpt_path)
            print(f"Load from checkpoint: {ckpt_path}")

        else:
            model = init_model(
                SRC_VOCAB_SIZE,
                TAR_VOCAB_SIZE,
                D_MODEL,
                N_HEAD,
                D_FF,
                OUTPUT_MAX_LENGTH,
                N_LAYER,
                DROPOUT,
            )
            print(f"Model initiated")
        torch.set_float32_matmul_precision("high")

        trainer = pl.Trainer(
            max_epochs=EPOCH, callbacks=[checkpoint_callback], enable_progress_bar=True
        )

        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
