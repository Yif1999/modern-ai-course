from __future__ import annotations

import shutil
import unicodedata
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, trainers


CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = CURRENT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = CURRENT_DIR / "outputs"
TOKENIZER_DIR = OUTPUT_DIR / "tokenizers"

RAW_TEXT_PATH = RAW_DIR / "input_zh.txt"
TOKENIZER_PATH = TOKENIZER_DIR / "chinese_bpe_tokenizer.json"
PREVIOUS_RAW_TEXT_PATH = (
    CURRENT_DIR.parent
    / "20_chinese_open_text_pretraining_dataset"
    / "data"
    / "raw"
    / "input_zh.txt"
)
PREVIOUS_TOKENIZER_PATH = (
    CURRENT_DIR.parent
    / "21_chinese_bpe_tokenizer_intro"
    / "outputs"
    / "tokenizers"
    / "chinese_bpe_vocab512.json"
)

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
DEFAULT_VOCAB_SIZE = 512


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u3000", " ")
    return text


def build_default_raw_text() -> str:
    lines = [
        "人工智能正在改变我们的学习方式。",
        "大语言模型可以根据上下文预测下一个词。",
        "今天我们用 MLX 在 MacBook Pro 上训练一个中文 Tiny GPT。",
        "中文、English、数字123和标点符号！都需要 tokenizer 处理。",
        "BPE tokenizer 会把常见片段合并成更少的 token。",
        "Tiny GPT 读取的是 BPE token ids，而不是原始字符串。",
        "训练时模型在每个位置预测下一个 BPE token。",
        "生成时模型先输出 token id，再通过 tokenizer decode 成中文文本。",
    ]
    return "\n".join(lines) + "\n"


def ensure_raw_text() -> str:
    ensure_dirs()
    if RAW_TEXT_PATH.exists():
        return normalize_text(RAW_TEXT_PATH.read_text(encoding="utf-8"))
    if PREVIOUS_RAW_TEXT_PATH.exists():
        shutil.copyfile(PREVIOUS_RAW_TEXT_PATH, RAW_TEXT_PATH)
        return normalize_text(RAW_TEXT_PATH.read_text(encoding="utf-8"))
    RAW_TEXT_PATH.write_text(build_default_raw_text(), encoding="utf-8")
    return normalize_text(RAW_TEXT_PATH.read_text(encoding="utf-8"))


class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self.tokenizer.token_to_id("<pad>")

    @property
    def unk_id(self) -> int:
        return self.tokenizer.token_to_id("<unk>")

    @property
    def bos_id(self) -> int:
        return self.tokenizer.token_to_id("<bos>")

    @property
    def eos_id(self) -> int:
        return self.tokenizer.token_to_id("<eos>")

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.tokenizer.encode(normalize_text(text)).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def encode_tokens(self, text: str) -> list[str]:
        return self.tokenizer.encode(normalize_text(text)).tokens

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode([int(i) for i in ids], skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise KeyError(f"token not found: {token}")
        return int(token_id)

    def save(self, path: Path = TOKENIZER_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))


def train_bpe_tokenizer(corpus_lines: list[str], vocab_size: int = DEFAULT_VOCAB_SIZE) -> BPETokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.decoder = decoders.Fuse()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )
    tokenizer.train_from_iterator(corpus_lines, trainer=trainer)
    wrapper = BPETokenizer(tokenizer)
    wrapper.save(TOKENIZER_PATH)
    return wrapper


def load_or_train_tokenizer(corpus_lines: list[str] | None = None) -> tuple[BPETokenizer, str]:
    ensure_dirs()

    if TOKENIZER_PATH.exists():
        tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        return BPETokenizer(tokenizer), "loaded_from_current_course"

    if PREVIOUS_TOKENIZER_PATH.exists():
        shutil.copyfile(PREVIOUS_TOKENIZER_PATH, TOKENIZER_PATH)
        tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        return BPETokenizer(tokenizer), "copied_from_lesson_21"

    if corpus_lines is None:
        raw_text = ensure_raw_text()
        corpus_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    tokenizer = train_bpe_tokenizer(corpus_lines)
    return tokenizer, "trained_in_lesson_22"
