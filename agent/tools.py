"""
agent/tools.py
All 5 agent tools as standalone, independently testable Python functions.

Tool 1: retrieve_similar_tickets  → RAG search over ChromaDB
Tool 2: classify_request          → Claude classifies urgency + category
Tool 3: get_vendor                → Looks up vendors.json by category + urgency
Tool 4: draft_response            → Claude writes empathetic resident message
Tool 5: log_ticket                → Writes full ticket to SQLite
"""

import json
import os

import anthropic
from dotenv import load_dotenv

from agent.prompts import CLASSIFY_PROMPT, DRAFT_RESPONSE_PROMPT
from agent.tool_definitions import CATEGORIES, URGENCY_LEVELS
from rag.retriever import search_similar_tickets
from db_manager.sqlite_manager import generate_ticket_id, insert_ticket

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDORS_PATH = os.path.join(PROJECT_ROOT, "data", "vendors.json")
MODEL = "claude-sonnet-4-20250514"
RECURRING_ISSUE_SIMILARITY_THRESHOLD = 0.75
RECURRING_ISSUE_MIN_MATCHES = 2

_client = None


def _get_anthropic_client() -> anthropic.Anthropic:
    """Lazy-load the Anthropic client."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def _load_vendors() -> list[dict]:
    """Load vendor database from JSON file."""
    with open(VENDORS_PATH, "r") as f:
        data = json.load(f)
    return data["vendors"]


def _format_similar_tickets_context(tickets: list[dict]) -> str:
    """Format similar tickets into a readable context string for prompts."""
    if not tickets:
        return "No similar past tickets found."

    lines = []
    for i, t in enumerate(tickets, 1):
        lines.append(
            f"{i}. [{t['ticket_id']}] (similarity: {t['similarity_score']:.2f})\n"
            f"   Unit: {t.get('unit', 'N/A')}\n"
            f"   Complaint: {t['complaint']}\n"
            f"   Category: {t['category']} | Urgency: {t['urgency']}\n"
            f"   Resolution: {t['resolution']}"
        )
    return "\n\n".join(lines)


def _append_reason(existing_reason: str, new_reason: str) -> str:
    """Append an escalation reason without losing the model's original reason."""
    if not existing_reason:
        return new_reason
    if new_reason in existing_reason:
        return existing_reason
    return f"{existing_reason}; {new_reason}"


def _get_recurring_issue_matches(
    similar_tickets: list[dict],
    unit: str,
    category: str,
) -> list[dict]:
    """
    Return similar historical tickets for the same unit and category.

    This keeps the recurrence rule targeted: an AC issue across many apartments
    should not automatically raise one resident's ticket, but repeated AC issues
    in the same unit can indicate an unresolved root cause.
    """
    if not unit or not category:
        return []

    return [
        ticket
        for ticket in similar_tickets
        if ticket.get("unit") == unit
        and ticket.get("category") == category
        and ticket.get("similarity_score", 0) >= RECURRING_ISSUE_SIMILARITY_THRESHOLD
    ]


def _apply_recurring_issue_escalation(
    result: dict,
    similar_tickets: list[dict],
    unit: str,
) -> dict:
    """
    Upgrade low/medium requests to high when the same issue repeats in the same unit.

    The LLM still classifies the initial category/urgency, but this deterministic
    guardrail catches repeated unresolved patterns from historical tickets.
    """
    if result.get("urgency") not in {"low", "medium"}:
        return result

    matches = _get_recurring_issue_matches(
        similar_tickets=similar_tickets,
        unit=unit,
        category=result.get("category", ""),
    )

    if len(matches) < RECURRING_ISSUE_MIN_MATCHES:
        return result

    matched_ids = ", ".join(ticket["ticket_id"] for ticket in matches)
    reason = (
        f"Recurring issue pattern detected for unit {unit} from similar "
        f"historical tickets ({matched_ids}); upgraded urgency to high."
    )

    result["urgency"] = "high"
    result["requires_human_review"] = True
    result["escalation_reason"] = _append_reason(
        result.get("escalation_reason", ""),
        reason,
    )
    result["reasoning"] = _append_reason(result.get("reasoning", ""), reason)

    return result


# =========================================================================
# Tool 1: retrieve_similar_tickets
# =========================================================================
def retrieve_similar_tickets_tool(complaint: str, top_k: int = 5) -> dict:
    """
    RAG search over ChromaDB to find similar past maintenance tickets.

    Args:
        complaint: The resident's complaint text.
        top_k: Number of similar tickets to return.

    Returns:
        dict with:
          - similar_tickets: list of matching ticket dicts
          - count: number of results found
    """
    results = search_similar_tickets(query=complaint, top_k=top_k)

    return {
        "similar_tickets": results,
        "count": len(results),
    }


# =========================================================================
# Tool 2: classify_request
# =========================================================================
def classify_request_tool(
    complaint: str,
    unit: str,
    resident_name: str,
    similar_tickets: list[dict],
) -> dict:
    """
    Use Claude to classify a maintenance complaint into category + urgency,
    with confidence scoring and human escalation logic.

    Args:
        complaint: The resident's complaint text.
        unit: The apartment/unit number.
        resident_name: Name of the resident.
        similar_tickets: Previously retrieved similar tickets for context.

    Returns:
        dict with:
          - category: one of (plumbing, hvac, electrical, pest_control, appliance, general_maintenance)
          - urgency: one of (critical, high, medium, low)
          - confidence_score: float 0.0-1.0
          - requires_human_review: bool
          - escalation_reason: str (reason for escalation, if any)
          - reasoning: brief explanation of the classification
    """
    context = _format_similar_tickets_context(similar_tickets)

    prompt = CLASSIFY_PROMPT.format(
        unit=unit,
        resident_name=resident_name,
        complaint=complaint,
        similar_tickets_context=context,
    )

    client = _get_anthropic_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        temperature=0.0,  # deterministic classification
        messages=[{"role": "user", "content": prompt}],
    )

    # Parse the JSON response
    response_text = response.content[0].text.strip()

    # Handle potential markdown code fences in response
    if response_text.startswith("```"):
        response_text = response_text.split("\n", 1)[1]
        response_text = response_text.rsplit("```", 1)[0].strip()

    result = json.loads(response_text)

    # Validate the values
    if result.get("category") not in CATEGORIES:
        result["category"] = "general_maintenance"
    if result.get("urgency") not in URGENCY_LEVELS:
        result["urgency"] = "medium"

    # Validate and clamp confidence score
    confidence = result.get("confidence_score", 0.5)
    if not isinstance(confidence, (int, float)):
        confidence = 0.5
    result["confidence_score"] = max(0.0, min(1.0, float(confidence)))

    # Enforce auto-escalation if confidence is below threshold
    if result["confidence_score"] < 0.75:
        result["requires_human_review"] = True
        if not result.get("escalation_reason"):
            result["escalation_reason"] = f"Low confidence score ({result['confidence_score']:.2f})"

    # Ensure boolean type
    result["requires_human_review"] = bool(result.get("requires_human_review", False))
    result["escalation_reason"] = result.get("escalation_reason", "")

    result = _apply_recurring_issue_escalation(result, similar_tickets, unit)

    return result


# =========================================================================
# Tool 3: get_vendor
# =========================================================================
def get_vendor_tool(category: str, urgency: str) -> dict:
    """
    Look up the appropriate vendor from vendors.json based on category and urgency.

    Args:
        category: The classified category (e.g., "plumbing").
        urgency: The classified urgency level (e.g., "high").

    Returns:
        dict with:
          - vendor_name: name of the vendor
          - vendor_phone: contact phone number
          - vendor_email: contact email
          - sla_hours: SLA deadline in hours for this urgency level
          - available_24_7: whether vendor is available 24/7
          - rating: vendor rating
          - found: whether a matching vendor was found
    """
    vendors = _load_vendors()

    # Find vendors matching the category
    matching = [v for v in vendors if v["category"] == category]

    if not matching:
        # Fallback to general maintenance
        matching = [v for v in vendors if v["category"] == "general_maintenance"]

    if not matching:
        return {
            "vendor_name": "Property Management Office",
            "vendor_phone": "555-0000",
            "vendor_email": "management@property.com",
            "sla_hours": 24,
            "available_24_7": False,
            "rating": None,
            "found": False,
        }

    # If multiple vendors match, pick the best rated one
    # For critical/high urgency, prefer 24/7 vendors
    if urgency in ("critical", "high"):
        available_vendors = [v for v in matching if v.get("available_24_7", False)]
        if available_vendors:
            matching = available_vendors

    vendor = max(matching, key=lambda v: v.get("rating", 0))

    sla_hours = vendor.get("sla_hours", {}).get(urgency, 24)

    return {
        "vendor_name": vendor["name"],
        "vendor_phone": vendor["phone"],
        "vendor_email": vendor["email"],
        "sla_hours": sla_hours,
        "available_24_7": vendor.get("available_24_7", False),
        "rating": vendor.get("rating"),
        "found": True,
    }


# =========================================================================
# Tool 4: draft_response
# =========================================================================
def draft_response_tool(
    complaint: str,
    unit: str,
    resident_name: str,
    category: str,
    urgency: str,
    vendor_name: str,
    vendor_phone: str,
    sla_hours: int,
    similar_tickets: list[dict],
    requires_human_review: bool = False,
    escalation_reason: str = "",
) -> dict:
    """
    Use Claude to draft an empathetic response message to the resident.

    Args:
        complaint: The resident's complaint text.
        unit: The apartment/unit number.
        resident_name: Name of the resident.
        category: Classified category.
        urgency: Classified urgency.
        vendor_name: Assigned vendor name.
        vendor_phone: Vendor phone number.
        sla_hours: SLA deadline in hours.
        similar_tickets: Similar past tickets for resolution context.
        requires_human_review: Whether escalated for human review.
        escalation_reason: Reason for escalation.

    Returns:
        dict with:
          - message: the drafted response text
    """
    context = _format_similar_tickets_context(similar_tickets)

    prompt = DRAFT_RESPONSE_PROMPT.format(
        resident_name=resident_name,
        unit=unit,
        complaint=complaint,
        category=category,
        urgency=urgency,
        vendor_name=vendor_name,
        vendor_phone=vendor_phone,
        sla_hours=sla_hours,
        requires_human_review="Yes" if requires_human_review else "No",
        escalation_reason=escalation_reason or "N/A",
        similar_tickets_context=context,
    )

    client = _get_anthropic_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        temperature=0.7,  # slightly creative for empathetic writing
        messages=[{"role": "user", "content": prompt}],
    )

    message = response.content[0].text.strip()

    return {"message": message}


# =========================================================================
# Tool 5: log_ticket
# =========================================================================
def log_ticket_tool(
    unit: str,
    resident_name: str,
    complaint: str,
    category: str,
    urgency: str,
    vendor_name: str,
    vendor_phone: str,
    vendor_email: str,
    sla_hours: int,
    similar_ticket_ids: list[str],
    resident_message: str,
    confidence_score: float = 0.0,
    requires_human_review: bool = False,
    escalation_reason: str = "",
) -> dict:
    """
    Log the complete triage result to the SQLite database.

    Args:
        unit: The apartment/unit number.
        resident_name: Name of the resident.
        complaint: The resident's complaint text.
        category: Classified category.
        urgency: Classified urgency.
        vendor_name: Assigned vendor name.
        vendor_phone: Vendor phone number.
        vendor_email: Vendor email.
        sla_hours: SLA deadline in hours.
        similar_ticket_ids: List of similar ticket IDs from RAG.
        resident_message: The drafted empathetic response.
        confidence_score: AI confidence in classification (0.0-1.0).
        requires_human_review: Whether escalated for human review.
        escalation_reason: Reason for escalation.

    Returns:
        dict with:
          - ticket_id: the generated ticket ID
          - status: "open"
          - logged: True if successful
    """
    ticket_id = generate_ticket_id()

    ticket = insert_ticket(
        ticket_id=ticket_id,
        unit=unit,
        resident_name=resident_name,
        complaint=complaint,
        category=category,
        urgency=urgency,
        vendor_name=vendor_name,
        vendor_phone=vendor_phone,
        vendor_email=vendor_email,
        sla_hours=sla_hours,
        similar_tickets=similar_ticket_ids,
        resident_message=resident_message,
        confidence_score=confidence_score,
        requires_human_review=requires_human_review,
        escalation_reason=escalation_reason,
        status="open",
    )

    return {
        "ticket_id": ticket["ticket_id"],
        "status": ticket["status"],
        "created_at": ticket["created_at"],
        "logged": True,
    }
