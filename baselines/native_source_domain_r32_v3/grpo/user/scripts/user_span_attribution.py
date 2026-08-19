"""Character-to-token span attribution using the frozen parent tokenizer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenSpan:
    start: int
    end: int
    token_ids: tuple[int, ...]


class TokenSpanMapper:
    def __init__(self, tokenizer, text: str):
        self.text = text
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        self.input_ids = list(encoded["input_ids"])
        self.offsets = [tuple(item) for item in encoded["offset_mapping"]]

    def map(self, char_start: int, char_end: int) -> TokenSpan:
        indices = [
            index
            for index, (start, end) in enumerate(self.offsets)
            if end > char_start and start < char_end and end > start
        ]
        if not indices:
            boundary = next((i for i, (start, _) in enumerate(self.offsets) if start >= char_start), len(self.input_ids))
            return TokenSpan(boundary, boundary, ())
        start, end = indices[0], indices[-1] + 1
        return TokenSpan(start, end, tuple(self.input_ids[start:end]))


def sid_component_token_spans(tokenizer, sid: str) -> dict[str, TokenSpan]:
    mapper = TokenSpanMapper(tokenizer, sid)
    markers = {
        "domain": (0, sid.index("><s_a_") + 1),
        "a": (sid.index("<s_a_"), sid.index("><s_b_") + 1),
        "b": (sid.index("<s_b_"), sid.index("><s_c_") + 1),
        "c": (sid.index("<s_c_"), len(sid)),
    }
    return {name: mapper.map(*span) for name, span in markers.items()}
