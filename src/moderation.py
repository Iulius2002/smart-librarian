# src/moderation.py
import re

# listă minimală; extinde după nevoie
BANNED = [
    r"\bidiot\b",
    r"\bprost\b",
    r"\bfraier\b",
    r"\b(?:fut|pula|muie)\b",
]

def contains_profanity(text: str) -> bool:
    low = text.lower()
    return any(re.search(pat, low, flags=re.IGNORECASE) for pat in BANNED)
