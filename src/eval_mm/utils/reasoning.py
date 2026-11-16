"""Separate a model's reasoning from its final answer."""

from __future__ import annotations

import re

_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)


def split_reasoning(text: str | None) -> tuple[str, str | None]:
    """Split *text* into ``(answer, reasoning)``.

    Handles the three shapes that reasoning models emit when the server does
    not separate reasoning itself:

    * ``<think>…</think>answer`` — the block is the reasoning.
    * ``…</think>answer`` — the chat template already opened the block, so
      only the closing tag appears; everything before it is reasoning.
    * ``<think>…`` with no closing tag — generation stopped while reasoning,
      so there is no answer.

    ``reasoning`` is ``None`` when *text* contains no reasoning.
    """
    if not text:
        return "", None

    blocks = _THINK_BLOCK.findall(text)
    rest = _THINK_BLOCK.sub("", text)
    reasoning_parts = [b.strip() for b in blocks]

    close = None
    for close in _THINK_CLOSE.finditer(rest):
        pass
    if close is not None:
        reasoning_parts.append(rest[: close.start()].strip())
        rest = rest[close.end() :]

    opened = _THINK_OPEN.search(rest)
    if opened is not None:
        reasoning_parts.append(rest[opened.end() :].strip())
        rest = rest[: opened.start()]

    reasoning = "\n\n".join(p for p in reasoning_parts if p)
    if not blocks and close is None and opened is None:
        return text.strip(), None
    return rest.strip(), reasoning
