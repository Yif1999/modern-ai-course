from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from prepare_chinese_dataset import prepare_dataset


CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = CURRENT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_TOKENS_PATH = PROCESSED_DIR / "train_tokens.npy"
VAL_TOKENS_PATH = PROCESSED_DIR / "val_tokens.npy"


class ChineseTextDataset:
    def __init__(self, block_size: int = 32, batch_size: int = 32, auto_prepare: bool = True):
        self.block_size = block_size
        self.batch_size = batch_size

        if auto_prepare and (not TRAIN_TOKENS_PATH.exists() or not VAL_TOKENS_PATH.exists()):
            prepare_dataset()

        self.train_tokens = np.load(TRAIN_TOKENS_PATH).astype(np.int32)
        self.val_tokens = np.load(VAL_TOKENS_PATH).astype(np.int32)

    def get_batch(self, split: str = "train"):
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        source = self.train_tokens if split == "train" else self.val_tokens
        if len(source) <= self.block_size + 1:
            raise ValueError(
                f"{split} split is too short for block_size={self.block_size}; "
                "add more text or lower block_size"
            )

        starts = np.random.randint(
            0,
            len(source) - self.block_size - 1,
            size=(self.batch_size,),
        )
        x = np.stack([source[i : i + self.block_size] for i in starts]).astype(np.int32)
        y = np.stack([source[i + 1 : i + self.block_size + 1] for i in starts]).astype(np.int32)
        return mx.array(x), mx.array(y)


def main() -> None:
    dataset = ChineseTextDataset(block_size=32, batch_size=4)
    x, y = dataset.get_batch("train")
    mx.eval(x, y)
    print("=== ChineseTextDataset Batch Demo ===")
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("x first row:", x[0].tolist())
    print("y first row:", y[0].tolist())
    print("y is x shifted by one token for next-token prediction.")


if __name__ == "__main__":
    main()
