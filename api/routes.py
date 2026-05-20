"""
api/routes.py
API endpoints for the Maintenance Triage Agent.

Endpoints:
  POST /ticket              — Submit a complaint, run the full agent, return triage result
  GET  /tickets             — List all logged tickets (filterable by urgency, category, status)
  GET  /tickets/{id}        — Get a single ticket by ID
  PATCH /tickets/{id}/resolution — Update ticket resolution lifecycle fields
  GET  /tickets/stats       — Get aggregate ticket statistics
  GET  /responses           — List all drafted responses (filterable by urgency)
  GET  /responses/{id}      — Get the response for a specific ticket
  GET  /health              — Health check
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.models import (
    TicketRequest,
    TicketResolutionUpdate,
    TicketResponse,
    TicketListItem,
    TicketListResponse,
    TicketStatsResponse,
    HealthResponse,
    VendorInfo,
    ResponseItem,
    ResponseListResponse,
)
from agent.agent import process_complaint
from db_manager.sqlite_manager import (
    get_all_tickets,
    get_ticket_by_id,
    get_ticket_stats,
    update_ticket_resolution,
)

logger = logging.getLogger("maintenance_agent")

router = APIRouter()


def _ticket_response_from_agent_result(result: dict) -> TicketResponse:
    """Map the agent's internal result dictionary to the public API schema."""
    return TicketResponse(
        ticket_id=result["ticket_id"],
        unit=result["unit"],
        resident_name=result["resident_name"],
        complaint=result["complaint"],
        category=result["category"],
        urgency=result["urgency"],
        confidence_score=result.get("confidence_score", 0.0),
        reasoning=result.get("reasoning", ""),
        vendor=VendorInfo(
            name=result["vendor"]["name"],
            phone=result["vendor"]["phone"],
            email=result["vendor"]["email"],
            sla_hours=result["vendor"]["sla_hours"],
        ),
        similar_tickets=result.get("similar_tickets", []),
        resident_message=result.get("resident_message", ""),
        emergency_override=result.get("emergency_override", False),
        status=result.get("status", "open"),
        resolution_notes=result.get("resolution_notes", ""),
        resolved_at=result.get("resolved_at"),
        closed_at=result.get("closed_at"),
        resident_confirmed=result.get("resident_confirmed", False),
        ready_for_rag_ingestion=result.get("ready_for_rag_ingestion", False),
        needs_human_review=result.get("needs_human_review", False),
        escalation_reason=result.get("escalation_reason", ""),
        recommended_action=result.get("recommended_action", ""),
    )


def _ticket_list_item_from_row(ticket: dict) -> TicketListItem:
    """Map a stored ticket row to the list-view API schema."""
    return TicketListItem(
        ticket_id=ticket["ticket_id"],
        unit=ticket["unit"],
        resident_name=ticket["resident_name"],
        complaint=ticket["complaint"],
        category=ticket["category"],
        urgency=ticket["urgency"],
        confidence_score=ticket.get("confidence_score", 0.0),
        needs_human_review=ticket.get("requires_human_review", False),
        escalation_reason=ticket.get("escalation_reason", ""),
        vendor_name=ticket.get("vendor_name"),
        sla_hours=ticket.get("sla_hours"),
        status=ticket["status"],
        resolution_notes=ticket.get("resolution_notes", ""),
        resolved_at=ticket.get("resolved_at"),
        closed_at=ticket.get("closed_at"),
        resident_confirmed=ticket.get("resident_confirmed", False),
        ready_for_rag_ingestion=ticket.get("ready_for_rag_ingestion", False),
        created_at=ticket["created_at"],
        updated_at=ticket["updated_at"],
    )


def _response_item_from_row(ticket: dict) -> ResponseItem:
    """Map a stored ticket row to the drafted-response API schema."""
    return ResponseItem(
        ticket_id=ticket["ticket_id"],
        unit=ticket["unit"],
        resident_name=ticket["resident_name"],
        complaint=ticket["complaint"],
        category=ticket["category"],
        urgency=ticket["urgency"],
        confidence_score=ticket.get("confidence_score", 0.0),
        needs_human_review=ticket.get("requires_human_review", False),
        escalation_reason=ticket.get("escalation_reason", ""),
        resident_message=ticket["resident_message"],
        vendor_name=ticket.get("vendor_name"),
        sla_hours=ticket.get("sla_hours"),
        status=ticket["status"],
        created_at=ticket["created_at"],
    )


# ---------------------------------------------------------------------------
# POST /ticket — Submit a new maintenance complaint
# ---------------------------------------------------------------------------
@router.post(
    "/ticket",
    response_model=TicketResponse,
    summary="Submit a maintenance complaint",
    description=(
        "Accepts a resident's maintenance complaint and runs the full AI triage pipeline: "
        "RAG retrieval → classification → vendor assignment → response drafting → logging."
    ),
    tags=["Tickets"],
)
async def create_ticket(request: TicketRequest) -> TicketResponse:
    """
    Process a new maintenance complaint through the full agent pipeline.

    The agent will:
    1. Search for similar past tickets (RAG)
    2. Classify urgency and category
    3. Assign the right vendor
    4. Draft an empathetic response
    5. Log everything to the database
    """
    logger.info(f"API: New ticket from {request.resident_name} in unit {request.unit}")

    # Run the agent
    result = process_complaint(
        complaint=request.complaint,
        unit=request.unit,
        resident_name=request.resident_name,
    )

    return _ticket_response_from_agent_result(result)


# ---------------------------------------------------------------------------
# GET /tickets — List all tickets (with optional filters)
# ---------------------------------------------------------------------------
@router.get(
    "/tickets",
    response_model=TicketListResponse,
    summary="List all tickets",
    description="Retrieve all logged tickets. Optionally filter by urgency, category, or status.",
    tags=["Tickets"],
)
async def list_tickets(
    urgency: Optional[str] = Query(
        None,
        description="Filter by urgency level (critical, high, medium, low).",
        enum=["critical", "high", "medium", "low"],
    ),
    category: Optional[str] = Query(
        None,
        description="Filter by category.",
        enum=[
            "plumbing", "hvac", "electrical",
            "pest_control", "appliance", "general_maintenance",
        ],
    ),
    status: Optional[str] = Query(
        None,
        description="Filter by ticket status.",
        enum=["open", "assigned", "vendor_accepted", "in_progress", "resolved", "closed"],
    ),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of tickets to return."),
) -> TicketListResponse:
    """Retrieve tickets with optional filters."""

    tickets = get_all_tickets(
        status=status,
        category=category,
        urgency=urgency,
        limit=limit,
    )

    ticket_items = [_ticket_list_item_from_row(t) for t in tickets]

    # Build filters applied dict
    filters = {}
    if urgency:
        filters["urgency"] = urgency
    if category:
        filters["category"] = category
    if status:
        filters["status"] = status

    return TicketListResponse(
        tickets=ticket_items,
        total=len(ticket_items),
        filters_applied=filters,
    )


# ---------------------------------------------------------------------------
# GET /tickets/stats — Aggregate statistics
# ---------------------------------------------------------------------------
@router.get(
    "/tickets/stats",
    response_model=TicketStatsResponse,
    summary="Get ticket statistics",
    description="Returns aggregate counts grouped by status, urgency, and category.",
    tags=["Tickets"],
)
async def ticket_stats() -> TicketStatsResponse:
    """Get aggregate ticket statistics."""
    stats = get_ticket_stats()
    return TicketStatsResponse(**stats)


# ---------------------------------------------------------------------------
# GET /tickets/{ticket_id} — Get a single ticket
# ---------------------------------------------------------------------------
@router.get(
    "/tickets/{ticket_id}",
    summary="Get a single ticket",
    description="Retrieve full details for a specific ticket by its ID.",
    tags=["Tickets"],
)
async def get_ticket(ticket_id: str):
    """Retrieve a single ticket by ID."""
    ticket = get_ticket_by_id(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found.")
    return ticket


# ---------------------------------------------------------------------------
# PATCH /tickets/{ticket_id}/resolution — Update resolution lifecycle
# ---------------------------------------------------------------------------
@router.patch(
    "/tickets/{ticket_id}/resolution",
    summary="Update ticket resolution lifecycle",
    description=(
        "Update ticket status, resolution notes, resident confirmation, and whether "
        "the ticket is ready to be ingested into the RAG history."
    ),
    tags=["Tickets"],
)
async def update_resolution(ticket_id: str, request: TicketResolutionUpdate):
    """Update resolution lifecycle fields for a ticket."""
    try:
        ticket = update_ticket_resolution(
            ticket_id=ticket_id,
            status=request.status.value if request.status else None,
            resolution_notes=request.resolution_notes,
            resident_confirmed=request.resident_confirmed,
            ready_for_rag_ingestion=request.ready_for_rag_ingestion,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found.")

    return ticket


# ---------------------------------------------------------------------------
# GET /responses — List all drafted responses
# ---------------------------------------------------------------------------
@router.get(
    "/responses",
    response_model=ResponseListResponse,
    summary="List all responses",
    description="Retrieve all drafted responses sent to residents. Optionally filter by urgency.",
    tags=["Responses"],
)
async def list_responses(
    urgency: Optional[str] = Query(
        None,
        description="Filter by urgency level.",
        enum=["critical", "high", "medium", "low"],
    ),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of responses to return."),
) -> ResponseListResponse:
    """Retrieve all drafted responses."""
    tickets = get_all_tickets(urgency=urgency, limit=limit)

    responses = [
        _response_item_from_row(t)
        for t in tickets
        if t.get("resident_message")  # only include tickets that have a drafted response
    ]

    return ResponseListResponse(responses=responses, total=len(responses))


# ---------------------------------------------------------------------------
# GET /responses/{ticket_id} — Get response for a specific ticket
# ---------------------------------------------------------------------------
@router.get(
    "/responses/{ticket_id}",
    response_model=ResponseItem,
    summary="Get response for a ticket",
    description="Retrieve the drafted response for a specific ticket by its ID.",
    tags=["Responses"],
)
async def get_response(ticket_id: str) -> ResponseItem:
    """Retrieve the drafted response for a specific ticket."""
    ticket = get_ticket_by_id(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found.")

    if not ticket.get("resident_message"):
        raise HTTPException(status_code=404, detail=f"No response found for ticket {ticket_id}.")

    return _response_item_from_row(ticket)


# ---------------------------------------------------------------------------
# GET /health — Health check
# ---------------------------------------------------------------------------
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns service health status.",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Service health check."""
    return HealthResponse()
