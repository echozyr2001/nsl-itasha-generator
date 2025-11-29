from __future__ import annotations

from typing import Callable, List, Tuple

KEYWORD_CHECKS: List[Tuple[str, Callable[[str], bool], float]] = [
    # (name, check_function, weight)
    ("mask_instruction", lambda p: "mask" in p.lower() and ("overlay" in p.lower() or "do not" in p.lower()), 1.0),
    ("divider_alignment", lambda p: ("divider" in p.lower() or "front/back" in p.lower()) and ("%" in p or "coordinate" in p.lower()), 1.0),
    ("screen_avoidance", lambda p: "screen" in p.lower() and ("avoid" in p.lower() or "background" in p.lower() or "grey" in p.lower()), 1.0),
    ("reference_reuse", lambda p: ("reference" in p.lower() and "exact" in p.lower()) or "do not invent" in p.lower(), 1.0),
    ("layout_slots", lambda p: "layout" in p.lower() and ("slot" in p.lower() or "position" in p.lower()), 0.8),
    ("multi_image", lambda p: "multiple" in p.lower() or ("image" in p.lower() and "source" in p.lower()), 0.8),
    ("coordinates", lambda p: "x=" in p.lower() or "y=" in p.lower() or "coordinate" in p.lower(), 0.7),
    ("no_hardware", lambda p: ("hardware" in p.lower() and "do not" in p.lower()) or "2d" in p.lower() or "flat" in p.lower(), 0.7),
    ("aspect_ratio", lambda p: "1:1" in p or "aspect" in p.lower(), 0.5),
    ("seamless", lambda p: "seamless" in p.lower() or "continuous" in p.lower() or "unified" in p.lower(), 0.6),
]


def score_prompt(prompt: str) -> float:
    """
    Score a prompt based on presence of key quality indicators.
    Returns a score between 0.0 and 1.0.
    """
    if not prompt or len(prompt.strip()) < 100:
        return 0.0  # Too short to be a valid prompt
    
    total_weight = sum(weight for _, _, weight in KEYWORD_CHECKS)
    weighted_score = 0.0
    
    for name, check, weight in KEYWORD_CHECKS:
        if check(prompt):
            weighted_score += weight
    
    # Normalize by total weight and add a base score for length/completeness
    base_score = min(0.3, len(prompt) / 5000.0)  # Up to 0.3 for length
    keyword_score = weighted_score / total_weight if total_weight > 0 else 0.0
    keyword_score = keyword_score * 0.7  # Scale keyword score
    
    final_score = min(1.0, base_score + keyword_score)
    return final_score
