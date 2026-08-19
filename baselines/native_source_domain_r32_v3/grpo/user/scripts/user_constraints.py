"""Unified constraint diagnostics shared by Action and Chain routes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ConstraintViolation:
    kind: str
    message: str
    char_start: int
    char_end: int
    route: str
    severity: str = "hard"
    token_start: int | None = None
    token_end: int | None = None
    token_ids: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def attribute(self, mapper) -> "ConstraintViolation":
        span = mapper.map(self.char_start, self.char_end)
        self.token_start = span.start
        self.token_end = span.end
        self.token_ids = list(span.token_ids)
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def violation(kind: str, message: str, span: tuple[int, int], route: str, **metadata: Any) -> ConstraintViolation:
    return ConstraintViolation(kind, message, span[0], span[1], route, metadata=metadata)
