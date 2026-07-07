from __future__ import annotations

import json
from pathlib import Path


class CharacterTokenizer:
    def __init__(self, chars: list[str]):
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        return cls(sorted(list(set(text))))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        missing = sorted(set(text) - set(self.stoi))
        if missing:
            raise ValueError(f"text contains tokens outside vocab: {missing}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path: Path) -> None:
        payload = {
            "type": "character",
            "chars": self.chars,
            "vocab_size": self.vocab_size,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharacterTokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload["chars"])
