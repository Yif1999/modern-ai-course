from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .base import AdaptedDocument, AdapterConfig, limit_documents
from .formatters import format_conversations, format_dialogue_turns, format_instruction, format_qa


class LocalJsonlAdapter:
    def __init__(self, config: AdapterConfig):
        if not config.path:
            raise ValueError("LocalJsonlAdapter requires config.path")
        self.config = config
        self.path = Path(config.path)

    def iter_rows(self) -> Iterable[dict[str, Any]]:
        with self.path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def row_to_text(self, row: dict[str, Any]) -> str:
        fields = self.config.field_map
        if "text" in fields:
            return str(row.get(fields["text"], "")).strip()

        if "question" in fields and "answer" in fields:
            context = str(row.get(fields["context"], "")) if fields.get("context") else None
            return format_qa(
                str(row.get(fields["question"], "")),
                str(row.get(fields["answer"], "")),
                context=context,
            )

        if "instruction" in fields and "response" in fields:
            return format_instruction(
                str(row.get(fields["instruction"], "")),
                str(row.get(fields["response"], "")),
                str(row.get(fields["input"], "")) if fields.get("input") else None,
            )

        if "conversations" in fields:
            return format_conversations(row.get(fields["conversations"]))

        if "dialogue" in fields:
            value = row.get(fields["dialogue"])
            if isinstance(value, list):
                return format_dialogue_turns([str(item) for item in value])
            return str(value or "").strip()

        for candidate in ["text", "content", "document", "answer", "response"]:
            value = row.get(candidate)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def iter_documents(self) -> Iterable[AdaptedDocument]:
        def generate():
            for i, row in enumerate(self.iter_rows()):
                text = self.row_to_text(row)
                if not text:
                    continue
                metadata = {"adapter": "local_jsonl"}
                for key in ["clean_text_hash", "clean_version", "source_cache_path"]:
                    if row.get(key):
                        metadata[key] = row[key]
                yield AdaptedDocument(
                    text=text,
                    source_name=self.config.source_name,
                    source_type=self.config.source_type,
                    source_group=self.config.source_group,
                    source_id=str(row.get("id", i)),
                    metadata=metadata,
                )

        return limit_documents(generate(), max_docs=self.config.max_docs, max_chars=self.config.max_chars)
