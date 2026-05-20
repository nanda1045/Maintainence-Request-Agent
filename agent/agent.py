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
import logging

import anthropic
from dotenv import load_dotenv

from agent.emergency import detect_emergency, get_emergency_category
from agent.prompts import SYSTEM_PROMPT
from agent.tool_definitions import TOOL_DEFINITIONS
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
                requires_human_review=tool_input.get("requires_human_review", False),
                escalation_reason=tool_input.get("escalation_reason", ""),
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
                confidence_score=tool_input.get("confidence_score", 0.0),
                requires_human_review=tool_input.get("requires_human_review", False),
                escalation_reason=tool_input.get("escalation_reason", ""),
            )

        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        logger.info(f"  Tool {tool_name} completed successfully")
        return json.dumps(result)

    except Exception as e:
        logger.error(f"  Tool {tool_name} failed: {e}")
        return json.dumps({"error": str(e)})


def _add_classification_context(tool_name: str, tool_input: dict, tool_results: dict) -> dict:
    """
    Carry escalation metadata from classification into later tools.

    Claude usually passes these fields itself, but the orchestrator enforces this
    propagation so drafting and logging cannot accidentally drop human-review data.
    """
    classification = tool_results.get("classify_request", {})
    if not classification:
        return tool_input

    if tool_name in {"draft_response", "log_ticket"}:
        tool_input.setdefault(
            "requires_human_review",
            classification.get("requires_human_review", False),
        )
        tool_input.setdefault(
            "escalation_reason",
            classification.get("escalation_reason", ""),
        )

    if tool_name == "log_ticket":
        tool_input.setdefault(
            "confidence_score",
            classification.get("confidence_score", 0.0),
        )

    return tool_input


# Backwards-compatible names used by eval/run_eval.py.
_detect_emergency = detect_emergency
_get_emergency_category = get_emergency_category


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
    requires_human_review = True
    confidence_score = 1.0
    escalation_reason = "Safety risk detected; emergency maintenance request requires property manager review."

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
        requires_human_review=requires_human_review,
        escalation_reason=escalation_reason,
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
        confidence_score=confidence_score,
        requires_human_review=requires_human_review,
        escalation_reason=escalation_reason,
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
        "confidence_score": confidence_score,
        "needs_human_review": requires_human_review,
        "escalation_reason": escalation_reason,
        "recommended_action": "Contact emergency services if there is immediate danger, dispatch the emergency vendor, and have a property manager personally review the ticket.",
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
            confidence_score=0.0,
            requires_human_review=True,
            escalation_reason=f"Agent fallback: {error_msg}",
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
        "confidence_score": 0.0,
        "needs_human_review": True,
        "escalation_reason": f"Agent fallback: {error_msg}",
        "recommended_action": "Route this ticket to a property manager for manual triage.",
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
                        tool_input = _add_classification_context(
                            tool_name=tool_name,
                            tool_input=tool_input,
                            tool_results=tool_results,
                        )

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
    confidence_score = _clamp_confidence(classification.get("confidence_score", 0.0))
    urgency = classification.get("urgency", "medium")
    needs_human_review = bool(classification.get("requires_human_review", False))

    if confidence_score < 0.75:
        needs_human_review = True
    if urgency == "critical":
        needs_human_review = True

    escalation_reason = classification.get("escalation_reason", "")
    if needs_human_review and not escalation_reason:
        if urgency == "critical":
            escalation_reason = "Safety risk detected; critical request requires property manager review."
        elif confidence_score < 0.75:
            escalation_reason = f"Low confidence score ({confidence_score:.2f})"
        else:
            escalation_reason = "Request meets human review criteria."

    return {
        "ticket_id": log.get("ticket_id", "UNKNOWN"),
        "unit": unit,
        "resident_name": resident_name,
        "complaint": complaint,
        "category": classification.get("category", "general_maintenance"),
        "urgency": urgency,
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
        "confidence_score": confidence_score,
        "needs_human_review": needs_human_review,
        "escalation_reason": escalation_reason,
        "recommended_action": _build_recommended_action(
            needs_human_review=needs_human_review,
            urgency=urgency,
            vendor_name=vendor.get("vendor_name", "UNKNOWN"),
            sla_hours=vendor.get("sla_hours", 24),
        ),
    }


def _clamp_confidence(value: object) -> float:
    """Return a confidence score in the inclusive 0.0-1.0 range."""
    if not isinstance(value, (int, float)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _build_recommended_action(
    needs_human_review: bool,
    urgency: str,
    vendor_name: str,
    sla_hours: int,
) -> str:
    """Summarize the operational next step for staff/API consumers."""
    if urgency == "critical":
        return (
            "Treat as an emergency: advise the resident to contact emergency services "
            "if there is immediate danger, dispatch the assigned vendor immediately, "
            "and have a property manager review the ticket."
        )
    if needs_human_review:
        return (
            "Dispatch the assigned vendor if appropriate and route the ticket to a "
            "property manager for review before closing automation."
        )
    return f"Dispatch {vendor_name} within the {sla_hours}-hour SLA and continue automated handling."
