from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal


@dataclass(frozen=True)
class MMARSample:
    id: str
    audio_path: str
    question: str
    choices: tuple[str, ...]
    answer: str

    modality: str | None = None
    category: str | None = None
    sub_category: str | None = None
    language: str | None = None
    source: str | None = None
    url: str | None = None
    timestamp: str | None = None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "MMARSample":
        # The dataset uses both "sub-category" and possibly other spellings.
        sub_category = d.get("sub-category")
        return MMARSample(
            id=str(d["id"]),
            audio_path=str(d.get("audio_path", "")),
            question=str(d["question"]),
            choices=tuple(str(x) for x in d.get("choices", [])),
            answer=str(d["answer"]),
            modality=d.get("modality"),
            category=d.get("category"),
            sub_category=sub_category,
            language=d.get("language"),
            source=d.get("source"),
            url=d.get("url"),
            timestamp=d.get("timestamp"),
        )


def load_mmar_meta(path: str | Path) -> list[MMARSample]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"Expected top-level list, got {type(data)}")
    return [MMARSample.from_dict(item) for item in data]


def iter_mmar_meta(path: str | Path) -> Iterable[MMARSample]:
    # Simple wrapper to allow future streaming.
    yield from load_mmar_meta(path)
