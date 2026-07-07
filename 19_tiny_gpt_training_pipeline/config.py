from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
SAMPLES_DIR = OUTPUT_DIR / "samples"


@dataclass
class TrainConfig:
    seed: int = 42
    raw_file_name: str = "tiny_text.txt"
    processed_file_name: str = "tiny_text_processed.npz"
    vocab_file_name: str = "vocab.json"
    train_split: float = 0.9

    block_size: int = 32
    batch_size: int = 32
    n_embd: int = 64
    num_heads: int = 4
    num_layers: int = 2
    learning_rate: float = 3e-3

    max_iters: int = 1200
    eval_interval: int = 100
    eval_iters: int = 10
    sample_interval: int = 300
    checkpoint_interval: int = 300

    sample_prompt: str = "hello "
    sample_tokens: int = 220
    sample_temperature: float = 0.8
    sample_top_k: int | None = 8

    @property
    def raw_text_path(self) -> Path:
        return RAW_DATA_DIR / self.raw_file_name

    @property
    def processed_path(self) -> Path:
        return PROCESSED_DATA_DIR / self.processed_file_name

    @property
    def vocab_path(self) -> Path:
        return PROCESSED_DATA_DIR / self.vocab_file_name

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload.update(
            {
                "project_dir": str(PROJECT_DIR),
                "data_dir": str(DATA_DIR),
                "raw_data_dir": str(RAW_DATA_DIR),
                "processed_data_dir": str(PROCESSED_DATA_DIR),
                "output_dir": str(OUTPUT_DIR),
                "checkpoint_dir": str(CHECKPOINT_DIR),
                "samples_dir": str(SAMPLES_DIR),
                "raw_text_path": str(self.raw_text_path),
                "processed_path": str(self.processed_path),
                "vocab_path": str(self.vocab_path),
            }
        )
        return payload


def ensure_project_dirs() -> None:
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        OUTPUT_DIR,
        CHECKPOINT_DIR,
        SAMPLES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
