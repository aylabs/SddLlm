from typing import Literal
import logging

Category = Literal["violence", "sexual", "hate", "self-harm", "illegal"]

BANNED_KEYWORDS = {
    "violence": ["kill", "attack", "murder", "assault"],
    "sexual": ["explicit", "porn", "nsfw"],
    "hate": ["racist", "hate", "bigot"],
    "self-harm": ["suicide", "self harm", "cut myself"],
    "illegal": ["how to steal", "fake id", "hack into"],
}

# Rationale mapping for auditable outcomes
CATEGORY_RATIONALE = {
    "violence": "Content promotes or describes violent acts",
    "sexual": "Content contains explicit sexual material",
    "hate": "Content promotes hatred or discrimination",
    "self-harm": "Content promotes self-harm or suicide",
    "illegal": "Content describes illegal activities",
}


def check_safety(text: str) -> tuple[str, Category | None]:
    """Check prompt safety against minimal policy categories."""
    lowered = text.lower()
    for cat, kws in BANNED_KEYWORDS.items():
        if any(k in lowered for k in kws):
            logging.info(f"Safety refuse: {cat} - {CATEGORY_RATIONALE[cat]}")
            return "refuse", cat  # type: ignore
    return "allow", None


def get_safety_rationale(category: Category | None) -> str:
    """Get human-readable rationale for safety decision."""
    if category is None:
        return "Prompt passed safety checks"
    return CATEGORY_RATIONALE.get(category, "Unknown category")
