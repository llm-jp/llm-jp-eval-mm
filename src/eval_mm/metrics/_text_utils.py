"""Shared text normalization helpers for scorers.

Many recent VLMs emit a reasoning/chain-of-thought prefix in a
``<think>...</think>`` block before the final answer. Leaving this
prefix in place distorts both strict scorers (the prefix causes the
format check to fail — e.g. ``<think>...B</think>\\n\\nB`` no longer
matches the expected ``B``) and permissive scorers (substring-match
spuriously hits candidate answers mentioned during reasoning).

:func:`strip_reasoning` normalizes away the reasoning portion. It is
a no-op for models that don't emit ``<think>`` tags, so existing
scores for those models are unaffected.
"""

from __future__ import annotations

import re

# Match a full <think>...</think> block (greedy, single or multi-line).
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Stray opening tag without close — take everything after.
_THINK_OPEN = re.compile(r"^.*?<think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove ``<think>…</think>`` reasoning blocks from *text*.

    * Complete blocks are dropped in-place.
    * A ``<think>`` with no matching close is treated as "everything up
      to end of string is reasoning"; in that case we return the empty
      string (the model never produced a final answer).
    * The surrounding answer text is preserved and ``strip()``'d.
    """
    if not text:
        return text
    cleaned = _THINK_BLOCK.sub("", text)
    if "<think>" in cleaned.lower():
        # Opening tag but no close — treat as no final answer.
        cleaned = _THINK_OPEN.sub("", cleaned, count=1)
        if "<think>" in cleaned.lower():
            cleaned = ""
    return cleaned.strip()
