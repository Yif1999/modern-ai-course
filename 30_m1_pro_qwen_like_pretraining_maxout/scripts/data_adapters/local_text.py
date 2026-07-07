from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .base import AdaptedDocument, AdapterConfig, limit_documents


class LocalTextAdapter:
    def __init__(self, config: AdapterConfig):
        if not config.path:
            raise ValueError("LocalTextAdapter requires config.path")
        self.config = config
        self.path = Path(config.path)

    def iter_documents(self) -> Iterable[AdaptedDocument]:
        text = self.path.read_text(encoding="utf-8", errors="replace")
        separator = str(self.config.options.get("separator", "\n\n<|doc_sep|>\n\n"))
        pieces = [piece.strip() for piece in text.split(separator) if piece.strip()]

        def generate():
            for i, piece in enumerate(pieces):
                yield AdaptedDocument(
                    text=piece,
                    source_name=self.config.source_name,
                    source_type=self.config.source_type,
                    source_group=self.config.source_group,
                    source_id=str(i),
                    metadata={"adapter": "local_text"},
                )

        return limit_documents(generate(), max_docs=self.config.max_docs, max_chars=self.config.max_chars)
