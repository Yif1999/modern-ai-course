from __future__ import annotations

import numpy as np
import mlx.core as mx

from config import TrainConfig
from prepare_dataset import prepare_dataset


class TinyTextDataset:
    def __init__(self, config: TrainConfig, auto_prepare: bool = True):
        self.config = config
        if auto_prepare and not config.processed_path.exists():
            prepare_dataset(config)

        data = np.load(config.processed_path)
        self.train_ids = data["train_ids"].astype(np.int32)
        self.val_ids = data["val_ids"].astype(np.int32)

    def get_batch(self, split: str):
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        source = self.train_ids if split == "train" else self.val_ids
        if len(source) <= self.config.block_size + 1:
            raise ValueError(f"{split} data is too short for block_size={self.config.block_size}")

        starts = np.random.randint(
            0,
            len(source) - self.config.block_size - 1,
            size=(self.config.batch_size,),
        )
        x = np.stack(
            [source[i : i + self.config.block_size] for i in starts],
        ).astype(np.int32)
        y = np.stack(
            [source[i + 1 : i + self.config.block_size + 1] for i in starts],
        ).astype(np.int32)
        return mx.array(x), mx.array(y)
