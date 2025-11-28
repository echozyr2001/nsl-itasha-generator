from __future__ import annotations

from typing import Callable, List, Tuple

KEYWORD_CHECKS: List[Tuple[str, Callable[[str], bool]]] = [
    ("mask", lambda p: "mask overlay" in p.lower() or "do not overlay the mask" in p.lower()),
    ("divider", lambda p: "front/back split" in p.lower() or "front/back divider" in p.lower()),
    ("screen", lambda p: "screen" in p.lower() and ("avoid" in p.lower() or "background" in p.lower())),
    ("reuse", lambda p: "reuse" in p.lower() or "do not invent" in p.lower()),
]


def score_prompt(prompt: str) -> float:
    total = 0
    for name, check in KEYWORD_CHECKS:
        total += 1 if check(prompt) else 0
    return total / len(KEYWORD_CHECKS)
