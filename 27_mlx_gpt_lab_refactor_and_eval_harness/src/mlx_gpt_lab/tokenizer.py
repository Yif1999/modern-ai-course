from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


class TokenizerLike(Protocol):
    vocab_size: int

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str: ...

    def save(self, path: Path) -> None: ...


class CharTokenizer:
    def __init__(self, stoi: dict[str, int], itos: dict[int, str]):
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(stoi)
        self.unk_id = stoi.get("<unk>", 1)

    @classmethod
    def train(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        tokens = SPECIAL_TOKENS + [ch for ch in chars if ch not in SPECIAL_TOKENS]
        stoi = {token: i for i, token in enumerate(tokens)}
        itos = {i: token for token, i in stoi.items()}
        return cls(stoi, itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        pieces = []
        for idx in ids:
            token = self.itos.get(int(idx), "<unk>")
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            pieces.append(token)
        return "".join(pieces)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tokenizer_type": "char",
            "special_tokens": SPECIAL_TOKENS,
            "stoi": self.stoi,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        stoi = {str(k): int(v) for k, v in payload["stoi"].items()}
        itos = {i: token for token, i in stoi.items()}
        return cls(stoi, itos)


class BPETokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode([int(i) for i in ids], skip_special_tokens=skip_special_tokens)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - 只在缺少依赖时触发
            raise RuntimeError("需要安装 tokenizers 才能加载 BPE tokenizer") from exc
        return cls(Tokenizer.from_file(str(path)))


def train_bpe_tokenizer(corpus_path: Path, vocab_size: int, output_path: Path) -> BPETokenizer:
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except ImportError as exc:  # pragma: no cover - 只在缺少依赖时触发
        raise RuntimeError("需要安装 tokenizers 才能训练 BPE tokenizer") from exc

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train([str(corpus_path)], trainer=trainer)
    wrapped = BPETokenizer(tokenizer)
    wrapped.save(output_path)
    return wrapped


def load_tokenizer(path: Path, tokenizer_type: str) -> TokenizerLike:
    if tokenizer_type == "char":
        return CharTokenizer.load(path)
    if tokenizer_type == "bpe":
        return BPETokenizer.load(path)
    raise ValueError(f"未知 tokenizer_type: {tokenizer_type}")
