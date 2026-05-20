"""
agent/emergency.py
Deterministic emergency detection for safety-critical complaints.

These checks run before the LLM tool loop so known emergency phrases are
fast-tracked without depending on probabilistic model behavior.
"""

import re


EMERGENCY_PATTERNS = [
    r"\bgas\s*(leak|smell|odor)\b",
    r"\b(smell|smelling)\s*(of\s+)?gas\b",
    r"\bflood(ing|ed)?\b",
    r"\bfire\b",
    r"\bsmoke\b(?!.*detector)",  # smoke but not "smoke detector" alone
    r"\bburning\s*smell\b",
    r"\bsmell\w*\s+(\w+\s+)*burning\b",  # "smell something burning"
    r"\belectrical\s*fire\b",
    r"\bcarbon\s*monoxide\b",
    r"\bstructural\s*(collapse|damage|failure)\b",
    r"\bceiling\s*(cav|collaps|fall)\w*\b",
    r"\bcollaps(ing|ed|e)\b",  # standalone "collapsing"
]


def detect_emergency(complaint: str) -> bool:
    """
    Return True when the complaint contains known emergency language.

    This is intentionally conservative: matching these patterns causes a
    critical urgency override and human review.
    """
    complaint_lower = complaint.lower()
    return any(re.search(pattern, complaint_lower) for pattern in EMERGENCY_PATTERNS)


def get_emergency_category(complaint: str) -> str:
    """Determine the most likely category from emergency keywords."""
    complaint_lower = complaint.lower()
    if any(kw in complaint_lower for kw in ["gas leak", "gas smell", "smell gas", "smelling gas"]):
        return "appliance"
    if any(kw in complaint_lower for kw in ["flood", "flooding", "flooded"]):
        return "plumbing"
    if any(kw in complaint_lower for kw in ["fire", "burning smell", "burning", "electrical fire", "smoke"]):
        return "electrical"
    if "carbon monoxide" in complaint_lower:
        return "electrical"
    if any(kw in complaint_lower for kw in ["collapse", "collapsing", "structural"]):
        return "general_maintenance"
    return "general_maintenance"
