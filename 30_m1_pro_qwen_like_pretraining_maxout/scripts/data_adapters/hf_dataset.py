from __future__ import annotations

from typing import Any, Iterable

from .base import AdaptedDocument, AdapterConfig, limit_documents
from .formatters import format_conversations, format_instruction, format_qa


class HFDatasetAdapter:
    def __init__(self, config: AdapterConfig):
        if not config.dataset_name:
            raise ValueError("HFDatasetAdapter requires config.dataset_name")
        self.config = config

    def iter_rows(self) -> Iterable[dict[str, Any]]:
        from datasets import load_dataset

        if self.config.options.get("data_files"):
            dataset = load_dataset(
                self.config.dataset_name or "json",
                data_files=self.config.options["data_files"],
                split=self.config.split,
                streaming=self.config.streaming,
                trust_remote_code=bool(self.config.options.get("trust_remote_code", False)),
            )
        else:
            dataset = load_dataset(
                self.config.dataset_name,
                split=self.config.split,
                streaming=self.config.streaming,
                trust_remote_code=bool(self.config.options.get("trust_remote_code", False)),
            )
        shuffle_buffer = int(self.config.options.get("shuffle_buffer", 0) or 0)
        if self.config.streaming and shuffle_buffer > 0 and hasattr(dataset, "shuffle"):
            seed = int(self.config.options.get("seed", 2060))
            dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
        skip_rows = int(self.config.options.get("skip_rows", 0) or 0)
        if skip_rows > 0:
            if hasattr(dataset, "skip"):
                dataset = dataset.skip(skip_rows)
            else:
                dataset = self._manual_skip(dataset, skip_rows)
        for row in dataset:
            if isinstance(row, dict):
                yield row

    @staticmethod
    def _manual_skip(dataset, skip_rows: int):
        def generate():
            for i, row in enumerate(dataset):
                if i < skip_rows:
                    continue
                yield row

        return generate()

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

        for candidate in ["text", "content", "document", "answer", "response", "completion"]:
            value = row.get(candidate)
            if isinstance(value, str) and value.strip():
                return value.strip()
        title = row.get("title")
        content = row.get("content")
        if isinstance(title, str) and isinstance(content, str):
            return f"标题：{title.strip()}\n正文：{content.strip()}".strip()
        return ""

    def iter_documents(self) -> Iterable[AdaptedDocument]:
        def generate():
            for i, row in enumerate(self.iter_rows()):
                text = self.row_to_text(row)
                if not text:
                    continue
                yield AdaptedDocument(
                    text=text,
                    source_name=self.config.source_name,
                    source_type=self.config.source_type,
                    source_group=self.config.source_group,
                    source_id=str(row.get("id", i)),
                    metadata={"adapter": "hf_dataset", "dataset_name": self.config.dataset_name},
                )

        return limit_documents(generate(), max_docs=self.config.max_docs, max_chars=self.config.max_chars)
