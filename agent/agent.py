"""
agent/agent.py
Core Claude agent with tool-calling loop.

This is the brain of the system. It:
  1. Receives a raw maintenance complaint
  2. Checks for emergency keywords (gas, flood, fire) → fast-track bypass
  3. Sends the complaint to Claude with all 5 tools defined
  4. Handles the tool-calling loop: parse tool calls → execute → return results
  5. Falls back to human review if classification fails
  6. Returns the complete triage result

The agent uses Claude's native tool_use capability — Claude decides which
tool to call at each step, and we execute them in a loop until Claude
produces a final text response.
"""

import json
import re
import logging
from typing import Optional

import anthropic
from dotenv import load_dotenv

from agent.prompts import SYSTEM_PROMPT
from agent.tools import (
    retrieve_similar_tickets_tool,
    classify_request_tool,
    get_vendor_tool,
    draft_response_tool,
    log_ticket_tool,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("maintenance_agent")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL = "claude-sonnet-4-20250514"
MAX_TOOL_ROUNDS = 10  # safety limit to prevent infinite loops

# Emergency keywords that trigger immediate critical override
EMERGENCY_PATTERNS = [
    r"\bgas\s*(leak|smell|odor)\b",
    r"\b(smell|smelling)\s*(of\s+)?gas\b",
    r"\bflood(ing|ed)?\b",
    r"\bfire\b",
    r"\bsmoke\b(?!.*detector)",  # smoke but not "smoke detector" alone
    r"\bburning\s*smell\b",
    r"\belectrical\s*fire\b",
    r"\bcarbon\s*monoxide\b",
    r"\bstructural\s*(collapse|damage|failure)\b",
    r"\bceiling\s*(cav|collaps|fall)\w*\b",
]

# ---------------------------------------------------------------------------
# Tool definitions for Claude's tool_use API
# ---------------------------------------------------------------------------
TOOL_DEFINITIONS = [
    {
        "name": "retrieve_similar_tickets",
        "description": (
            "Search the historical knowledge base (ChromaDB) to find past maintenance "
            "tickets similar to the current complaint. Returns the top 3 most similar "
            "tickets with their category, urgency, resolution, and similarity score. "
            "Always call this FIRST to get context before classifying."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "complaint": {
                    "type": "string",
                    "description": "The resident's raw complaint text.",
                },
            },
            "required": ["complaint"],
        },
    },
    {
        "name": "classify_request",
        "description": (
            "Classify the maintenance complaint into a category and urgency level. "
            "Uses the complaint text and similar past tickets to make an informed "
            "classification. Call this AFTER retrieve_similar_tickets."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "complaint": {
                    "type": "string",
                    "description": "The resident's raw complaint text.",
                },
                "unit": {
                    "type": "string",
                    "description": "The apartment/unit number.",
                },
                "resident_name": {
                    "type": "string",
                    "description": "Name of the resident.",
                },
                "similar_tickets": {
                    "type": "array",
                    "description": "List of similar past tickets from retrieve_similar_tickets.",
                    "items": {"type": "object"},
                },
            },
            "required": ["complaint", "unit", "resident_name", "similar_tickets"],
        },
    },
    {
        "name": "get_vendor",
        "description": (
            "Look up the best vendor for the job based on the classified category "
            "and urgency level. Returns vendor name, contact info, and SLA hours. "
            "Call this AFTER classify_request."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "The classified category (e.g., 'plumbing', 'hvac').",
                    "enum": [
                        "plumbing", "hvac", "electrical",
                        "pest_control", "appliance", "general_maintenance",
                    ],
                },
                "urgency": {
                    "type": "string",
                    "description": "The classified urgency level.",
                    "enum": ["critical", "high", "medium", "low"],
                },
            },
            "required": ["category", "urgency"],
        },
    },
    {
        "name": "draft_response",
        "description": (
            "Draft an empathetic response message to send to the resident. "
            "Includes acknowledgment, vendor info, and expected timeline. "
            "Call this AFTER get_vendor."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "complaint": {
                    "type": "string",
                    "description": "The resident's raw complaint text.",
                },
                "unit": {
                    "type": "string",
                    "description": "The apartment/unit number.",
                },
                "resident_name": {
                    "type": "string",
                    "description": "Name of the resident.",
                },
                "category": {
                    "type": "string",
                    "description": "The classified category.",
                },
                "urgency": {
                    "type": "string",
                    "description": "The classified urgency level.",
                },
                "vendor_name": {
                    "type": "string",
                    "description": "Name of the assigned vendor.",
                },
                "vendor_phone": {
                    "type": "string",
                    "description": "Vendor's phone number.",
                },
                "sla_hours": {
                    "type": "integer",
                    "description": "SLA deadline in hours.",
                },
                "similar_tickets": {
                    "type": "array",
                    "description": "Similar past tickets for resolution context.",
                    "items": {"type": "object"},
                },
            },
            "required": [
                "complaint", "unit", "resident_name", "category",
                "urgency", "vendor_name", "vendor_phone", "sla_hours",
                "similar_tickets",
            ],
        },
    },
    {
        "name": "log_ticket",
        "description": (
            "Log the complete triage result to the SQLite database. "
            "This is the FINAL step — call this after all other tools. "
            "Records the ticket for future reference and knowledge base."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "unit": {
                    "type": "string",
                    "description": "The apartment/unit number.",
                },
                "resident_name": {
                    "type": "string",
                    "description": "Name of the resident.",
                },
                "complaint": {
                    "type": "string",
                    "description": "The resident's raw complaint text.",
                },
                "category": {
                    "type": "string",
                    "description": "The classified category.",
                },
                "urgency": {
                    "type": "string",
                    "description": "The classified urgency level.",
                },
                "vendor_name": {
                    "type": "string",
                    "description": "Name of the assigned vendor.",
                },
                "vendor_phone": {
                    "type": "string",
                    "description": "Vendor's phone number.",
                },
                "vendor_email": {
                    "type": "string",
                    "description": "Vendor's email address.",
                },
                "sla_hours": {
                    "type": "integer",
                    "description": "SLA deadline in hours.",
                },
                "similar_ticket_ids": {
                    "type": "array",
                    "description": "List of similar ticket IDs from RAG retrieval.",
                    "items": {"type": "string"},
                },
                "resident_message": {
                    "type": "string",
                    "description": "The drafted empathetic response message.",
                },
            },
            "required": [
                "unit", "resident_name", "complaint", "category", "urgency",
                "vendor_name", "vendor_phone", "vendor_email", "sla_hours",
                "similar_ticket_ids", "resident_message",
            ],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatcher — maps tool names to actual functions
# ---------------------------------------------------------------------------
def _execute_tool(tool_name: str, tool_input: dict) -> str:
    """
    Execute a tool by name and return the JSON result string.

    This is the dispatcher that connects Claude's tool calls to our
    actual Python tool functions.
    """
    logger.info(f"Executing tool: {tool_name}")
    logger.debug(f"  Input: {json.dumps(tool_input, indent=2)[:500]}")

    try:
        if tool_name == "retrieve_similar_tickets":
            result = retrieve_similar_tickets_tool(
                complaint=tool_input["complaint"],
            )

        elif tool_name == "classify_request":
            result = classify_request_tool(
                complaint=tool_input["complaint"],
                unit=tool_input["unit"],
                resident_name=tool_input["resident_name"],
                similar_tickets=tool_input.get("similar_tickets", []),
            )

        elif tool_name == "get_vendor":
            result = get_vendor_tool(
                category=tool_input["category"],
                urgency=tool_input["urgency"],
            )

        elif tool_name == "draft_response":
            result = draft_response_tool(
                complaint=tool_input["complaint"],
                unit=tool_input["unit"],
                resident_name=tool_input["resident_name"],
                category=tool_input["category"],
                urgency=tool_input["urgency"],
                vendor_name=tool_input["vendor_name"],
                vendor_phone=tool_input["vendor_phone"],
                sla_hours=tool_input["sla_hours"],
                similar_tickets=tool_input.get("similar_tickets", []),
            )

        elif tool_name == "log_ticket":
            result = log_ticket_tool(
                unit=tool_input["unit"],
                resident_name=tool_input["resident_name"],
                complaint=tool_input["complaint"],
                category=tool_input["category"],
                urgency=tool_input["urgency"],
                vendor_name=tool_input["vendor_name"],
                vendor_phone=tool_input["vendor_phone"],
                vendor_email=tool_input["vendor_email"],
                sla_hours=tool_input["sla_hours"],
                similar_ticket_ids=tool_input.get("similar_ticket_ids", []),
                resident_message=tool_input["resident_message"],
            )

        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        logger.info(f"  Tool {tool_name} completed successfully")
        return json.dumps(result)

    except Exception as e:
        logger.error(f"  Tool {tool_name} failed: {e}")
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Emergency keyword detection
# ---------------------------------------------------------------------------
def _detect_emergency(complaint: str) -> bool:
    """
    Check if the complaint contains emergency keywords that require
    immediate critical-level response, bypassing the normal agent flow.
    """
    complaint_lower = complaint.lower()
    for pattern in EMERGENCY_PATTERNS:
        if re.search(pattern, complaint_lower):
            return True
    return False


def _get_emergency_category(complaint: str) -> str:
    """Determine category from emergency keywords."""
    complaint_lower = complaint.lower()
    if any(kw in complaint_lower for kw in ["gas leak", "gas smell", "smell gas", "smelling gas"]):
        return "appliance"
    if any(kw in complaint_lower for kw in ["flood", "flooding", "flooded"]):
        return "plumbing"
    if any(kw in complaint_lower for kw in ["fire", "burning smell", "electrical fire", "smoke"]):
        return "electrical"
    if "carbon monoxide" in complaint_lower:
        return "electrical"
    if any(kw in complaint_lower for kw in ["collapse", "structural"]):
        return "general_maintenance"
    return "general_maintenance"


def _handle_emergency(complaint: str, unit: str, resident_name: str) -> dict:
    """
    Fast-track emergency complaints, bypassing the full agent loop.
    Still uses tools but calls them directly in a deterministic order.
    """
    logger.warning(f"🚨 EMERGENCY DETECTED for unit {unit}!")

    # Step 1: Quick RAG retrieval
    rag_result = retrieve_similar_tickets_tool(complaint)
    similar_tickets = rag_result["similar_tickets"]
    similar_ids = [t["ticket_id"] for t in similar_tickets]

    # Step 2: Force critical urgency, detect category from keywords
    category = _get_emergency_category(complaint)
    urgency = "critical"
    reasoning = "EMERGENCY OVERRIDE: complaint contains safety-critical keywords"

    # Step 3: Get vendor (prioritize 24/7 for emergencies)
    vendor = get_vendor_tool(category, urgency)

    # Step 4: Draft response with emergency tone
    draft = draft_response_tool(
        complaint=complaint,
        unit=unit,
        resident_name=resident_name,
        category=category,
        urgency=urgency,
        vendor_name=vendor["vendor_name"],
        vendor_phone=vendor["vendor_phone"],
        sla_hours=vendor["sla_hours"],
        similar_tickets=similar_tickets,
    )

    # Step 5: Log the ticket
    log = log_ticket_tool(
        unit=unit,
        resident_name=resident_name,
        complaint=complaint,
        category=category,
        urgency=urgency,
        vendor_name=vendor["vendor_name"],
        vendor_phone=vendor["vendor_phone"],
        vendor_email=vendor["vendor_email"],
        sla_hours=vendor["sla_hours"],
        similar_ticket_ids=similar_ids,
        resident_message=draft["message"],
    )

    return {
        "ticket_id": log["ticket_id"],
        "unit": unit,
        "resident_name": resident_name,
        "complaint": complaint,
        "category": category,
        "urgency": urgency,
        "reasoning": reasoning,
        "vendor": {
            "name": vendor["vendor_name"],
            "phone": vendor["vendor_phone"],
            "email": vendor["vendor_email"],
            "sla_hours": vendor["sla_hours"],
        },
        "similar_tickets": similar_ids,
        "resident_message": draft["message"],
        "emergency_override": True,
        "status": "open",
        "needs_human_review": False,
    }


# ---------------------------------------------------------------------------
# Human review fallback
# ---------------------------------------------------------------------------
def _handle_fallback(
    complaint: str, unit: str, resident_name: str, error_msg: str
) -> dict:
    """
    Fallback when the agent fails — flag the ticket for human review.
    Still logs the ticket with what we know so nothing is lost.
    """
    logger.error(f"⚠️ Agent failed, flagging for human review: {error_msg}")

    # Try to at least get RAG results
    try:
        rag_result = retrieve_similar_tickets_tool(complaint)
        similar_ids = [t["ticket_id"] for t in rag_result["similar_tickets"]]
    except Exception:
        similar_ids = []

    # Log with unknown classification — human will review
    try:
        log = log_ticket_tool(
            unit=unit,
            resident_name=resident_name,
            complaint=complaint,
            category="general_maintenance",
            urgency="medium",
            vendor_name="PENDING HUMAN REVIEW",
            vendor_phone="N/A",
            vendor_email="N/A",
            sla_hours=24,
            similar_ticket_ids=similar_ids,
            resident_message=(
                f"Dear {resident_name.split()[0]},\n\n"
                "Thank you for reaching out. We've received your maintenance request "
                "and our team is currently reviewing it. A member of our property "
                "management team will contact you within 24 hours with an update.\n\n"
                "If this is an emergency, please call our emergency line immediately.\n\n"
                "Best regards,\nProperty Management Team"
            ),
        )
        ticket_id = log["ticket_id"]
    except Exception:
        ticket_id = "FAILED_TO_LOG"

    return {
        "ticket_id": ticket_id,
        "unit": unit,
        "resident_name": resident_name,
        "complaint": complaint,
        "category": "general_maintenance",
        "urgency": "medium",
        "reasoning": f"FALLBACK: {error_msg}",
        "vendor": {
            "name": "PENDING HUMAN REVIEW",
            "phone": "N/A",
            "email": "N/A",
            "sla_hours": 24,
        },
        "similar_tickets": similar_ids,
        "resident_message": (
            "We've received your request and our team is reviewing it. "
            "You'll hear back within 24 hours."
        ),
        "emergency_override": False,
        "status": "open",
        "needs_human_review": True,
        "error": error_msg,
    }


# ---------------------------------------------------------------------------
# Main agent function — the tool-calling loop
# ---------------------------------------------------------------------------
def process_complaint(
    complaint: str,
    unit: str,
    resident_name: str,
) -> dict:
    """
    Process a resident maintenance complaint end-to-end.

    This is the main entry point for the agent. It:
      1. Checks for emergency keywords → fast-track if found
      2. Sends complaint to Claude with 5 tool definitions
      3. Runs the tool-calling loop until Claude produces a final response
      4. Extracts the triage result from accumulated tool outputs
      5. Falls back to human review if anything fails

    Args:
        complaint: The resident's raw complaint text.
        unit: The apartment/unit number.
        resident_name: The resident's full name.

    Returns:
        dict with complete triage result:
          - ticket_id, category, urgency, vendor info, resident_message, etc.
    """
    logger.info(f"Processing complaint from {resident_name} in unit {unit}")
    logger.info(f"  Complaint: {complaint[:100]}...")

    # --- Emergency keyword override ---
    if _detect_emergency(complaint):
        return _handle_emergency(complaint, unit, resident_name)

    # --- Normal agent flow via Claude tool calling ---
    try:
        client = anthropic.Anthropic()

        # Build the initial user message
        user_message = (
            f"Process this maintenance complaint:\n\n"
            f"Resident Name: {resident_name}\n"
            f"Unit: {unit}\n"
            f"Complaint: {complaint}\n\n"
            f"Follow the standard triage process: retrieve similar tickets → "
            f"classify → get vendor → draft response → log ticket.\n"
            f"Process all 5 steps."
        )

        messages = [{"role": "user", "content": user_message}]

        # Track accumulated results from each tool call
        tool_results = {}

        # --- Tool-calling loop ---
        for round_num in range(MAX_TOOL_ROUNDS):
            logger.info(f"  Agent round {round_num + 1}")

            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )

            logger.info(f"  Stop reason: {response.stop_reason}")

            # If Claude is done (no more tool calls), break
            if response.stop_reason == "end_turn":
                logger.info("  Agent completed — final response received")
                break

            # Process tool calls in this response
            if response.stop_reason == "tool_use":
                # Add assistant's response (with tool_use blocks) to messages
                messages.append({"role": "assistant", "content": response.content})

                # Execute each tool call and collect results
                tool_result_contents = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input
                        tool_id = block.id

                        # Execute the tool
                        result_str = _execute_tool(tool_name, tool_input)
                        result_data = json.loads(result_str)

                        # Store results for later extraction
                        tool_results[tool_name] = result_data

                        # Add tool result to message for Claude
                        tool_result_contents.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": result_str,
                        })

                # Send all tool results back to Claude
                messages.append({"role": "user", "content": tool_result_contents})

            else:
                # Unexpected stop reason
                logger.warning(f"  Unexpected stop reason: {response.stop_reason}")
                break

        # --- Extract the final triage result ---
        return _extract_result(
            tool_results=tool_results,
            complaint=complaint,
            unit=unit,
            resident_name=resident_name,
        )

    except Exception as e:
        logger.error(f"Agent error: {e}")
        return _handle_fallback(complaint, unit, resident_name, str(e))


def _extract_result(
    tool_results: dict,
    complaint: str,
    unit: str,
    resident_name: str,
) -> dict:
    """
    Extract the final structured result from accumulated tool outputs.

    If any critical tool didn't run, flags for human review.
    """
    # Check if all essential tools ran
    essential_tools = {"retrieve_similar_tickets", "classify_request", "get_vendor", "draft_response", "log_ticket"}
    missing_tools = essential_tools - set(tool_results.keys())

    if missing_tools:
        logger.warning(f"  Missing tools: {missing_tools}")
        # If classification or logging failed, fallback
        if "classify_request" in missing_tools or "log_ticket" in missing_tools:
            return _handle_fallback(
                complaint, unit, resident_name,
                f"Agent did not complete all steps. Missing: {missing_tools}",
            )

    # Build the result from tool outputs
    classification = tool_results.get("classify_request", {})
    vendor = tool_results.get("get_vendor", {})
    draft = tool_results.get("draft_response", {})
    log = tool_results.get("log_ticket", {})
    rag = tool_results.get("retrieve_similar_tickets", {})

    similar_ids = [t["ticket_id"] for t in rag.get("similar_tickets", [])]

    return {
        "ticket_id": log.get("ticket_id", "UNKNOWN"),
        "unit": unit,
        "resident_name": resident_name,
        "complaint": complaint,
        "category": classification.get("category", "general_maintenance"),
        "urgency": classification.get("urgency", "medium"),
        "reasoning": classification.get("reasoning", ""),
        "vendor": {
            "name": vendor.get("vendor_name", "UNKNOWN"),
            "phone": vendor.get("vendor_phone", "N/A"),
            "email": vendor.get("vendor_email", "N/A"),
            "sla_hours": vendor.get("sla_hours", 24),
        },
        "similar_tickets": similar_ids,
        "resident_message": draft.get("message", ""),
        "emergency_override": False,
        "status": log.get("status", "open"),
        "needs_human_review": False,
    }
