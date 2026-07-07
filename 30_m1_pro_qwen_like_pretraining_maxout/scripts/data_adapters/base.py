from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Protocol


@dataclass
class AdaptedDocument:
    text: str
    source_name: str
    source_type: str
    source_group: str = ""
    source_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["char_count"] = len(self.text)
        return payload


@dataclass
class AdapterConfig:
    source_name: str
    source_type: str
    source_group: str = ""
    path: str | None = None
    dataset_name: str | None = None
    split: str = "train"
    max_docs: int | None = None
    max_chars: int | None = None
    enabled: bool = True
    streaming: bool = True
    field_map: dict[str, str] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)


class BaseAdapter(Protocol):
    config: AdapterConfig

    def iter_documents(self) -> Iterable[AdaptedDocument]:
        ...


def limit_documents(
    docs: Iterable[AdaptedDocument],
    *,
    max_docs: int | None = None,
    max_chars: int | None = None,
) -> Iterable[AdaptedDocument]:
    seen_docs = 0
    seen_chars = 0
    for doc in docs:
        if max_docs is not None and seen_docs >= max_docs:
            break
        if max_chars is not None and seen_chars >= max_chars:
            break
        seen_docs += 1
        seen_chars += len(doc.text)
        yield doc


def join_nonempty(parts: list[str], sep: str = "\n") -> str:
    return sep.join(part.strip() for part in parts if isinstance(part, str) and part.strip()).strip()
