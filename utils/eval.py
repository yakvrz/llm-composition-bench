#!/usr/bin/env python3
"""
Shared evaluation helpers: normalization and exact match.
"""
import re
import unicodedata
from typing import List

_PUNCT_RE = re.compile(r"[^\w\s\-\./]")
_WS_RE = re.compile(r"\s+")


def normalize_answer(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    s = re.sub(r"^\s*(answer|final answer|entity)\s*:\s*", "", s, flags=re.I)
    s = s.strip().strip("\"'`“”‘’")
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s)
    return s.casefold().strip()


def exact_match(pred: str, aliases: List[str]) -> bool:
    p = normalize_answer(pred)
    for a in aliases:
        if normalize_answer(a) == p:
            return True
    return False


