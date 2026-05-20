"""
api/models.py
Pydantic request/response models for the Maintenance Triage API.

Provides strict validation for incoming complaints and structured
output for triage results.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class UrgencyLevel(str, Enum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"


class Category(str, Enum):
    plumbing = "plumbing"
    hvac = "hvac"
    electrical = "electrical"
    pest_control = "pest_control"
    appliance = "appliance"
    general_maintenance = "general_maintenance"


class TicketStatus(str, Enum):
    open = "open"
    assigned = "assigned"
    vendor_accepted = "vendor_accepted"
    in_progress = "in_progress"
    resolved = "resolved"
    closed = "closed"


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class TicketRequest(BaseModel):
    """Incoming maintenance complaint from a resident."""

    resident_name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Full name of the resident filing the complaint.",
        examples=["Maria Garcia"],
    )
    unit: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Apartment/unit number.",
        examples=["9B"],
    )
    complaint: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Full description of the maintenance issue.",
        examples=["The dishwasher is leaking water all over the kitchen floor every time I run it."],
    )


class TicketResolutionUpdate(BaseModel):
    """Manager/vendor update for the ticket resolution lifecycle."""

    status: Optional[TicketStatus] = Field(
        None,
        description="New lifecycle status for the ticket.",
    )
    resolution_notes: Optional[str] = Field(
        None,
        min_length=1,
        max_length=2000,
        description="How the issue was resolved or the latest work notes.",
    )
    resident_confirmed: Optional[bool] = Field(
        None,
        description="Whether the resident confirmed the issue is resolved.",
    )
    ready_for_rag_ingestion: Optional[bool] = Field(
        None,
        description="Whether this closed/resolved ticket is eligible for future RAG ingestion.",
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
class VendorInfo(BaseModel):
    """Assigned vendor details."""

    name: str = Field(..., description="Vendor company name.")
    phone: str = Field(..., description="Vendor contact phone number.")
    email: str = Field(..., description="Vendor contact email.")
    sla_hours: int = Field(..., description="Expected response time in hours.")


class TicketResponse(BaseModel):
    """Full triage result returned after processing a complaint."""

    ticket_id: str = Field(..., description="Unique ticket identifier (e.g., TK-20250514-0001).")
    unit: str = Field(..., description="Apartment/unit number.")
    resident_name: str = Field(..., description="Name of the resident.")
    complaint: str = Field(..., description="Original complaint text.")
    category: str = Field(..., description="Classified maintenance category.")
    urgency: str = Field(..., description="Classified urgency level.")
    confidence_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="AI confidence in the classification.",
    )
    reasoning: str = Field("", description="Explanation for the classification.")
    vendor: VendorInfo = Field(..., description="Assigned vendor details.")
    similar_tickets: list[str] = Field(
        default_factory=list,
        description="IDs of similar past tickets from RAG retrieval.",
    )
    resident_message: str = Field(..., description="Empathetic response drafted for the resident.")
    emergency_override: bool = Field(
        False,
        description="Whether emergency keyword bypass was triggered.",
    )
    status: str = Field("open", description="Current ticket status.")
    resolution_notes: str = Field("", description="How the issue was resolved, once available.")
    resolved_at: Optional[str] = Field(None, description="When the issue was marked resolved.")
    closed_at: Optional[str] = Field(None, description="When the ticket was closed.")
    resident_confirmed: bool = Field(False, description="Whether the resident confirmed resolution.")
    ready_for_rag_ingestion: bool = Field(
        False,
        description="Whether this ticket can be embedded into the historical RAG store.",
    )
    needs_human_review: bool = Field(
        False,
        description="Whether the ticket was flagged for manual review.",
    )
    escalation_reason: str = Field(
        "",
        description="Reason the ticket requires human review, if applicable.",
    )
    recommended_action: str = Field(
        "",
        description="Recommended next operational action.",
    )


class TicketListItem(BaseModel):
    """Summarized ticket for list views."""

    ticket_id: str
    unit: str
    resident_name: str
    complaint: str
    category: str
    urgency: str
    confidence_score: float = 0.0
    needs_human_review: bool = False
    escalation_reason: str = ""
    vendor_name: Optional[str] = None
    sla_hours: Optional[int] = None
    status: str
    resolution_notes: str = ""
    resolved_at: Optional[str] = None
    closed_at: Optional[str] = None
    resident_confirmed: bool = False
    ready_for_rag_ingestion: bool = False
    created_at: str
    updated_at: str


class TicketListResponse(BaseModel):
    """Response for the GET /tickets endpoint."""

    tickets: list[TicketListItem]
    total: int
    filters_applied: dict = Field(
        default_factory=dict,
        description="Filters that were applied to the query.",
    )


class TicketStatsResponse(BaseModel):
    """Aggregate statistics across all tickets."""

    total: int
    by_status: dict[str, int] = Field(default_factory=dict)
    by_urgency: dict[str, int] = Field(default_factory=dict)
    by_category: dict[str, int] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    service: str = "maintenance-triage-agent"
    version: str = "1.0.0"


class ResponseItem(BaseModel):
    """A drafted response associated with a ticket."""

    ticket_id: str = Field(..., description="The ticket this response belongs to.")
    unit: str = Field(..., description="Apartment/unit number.")
    resident_name: str = Field(..., description="Resident's name.")
    complaint: str = Field(..., description="Original complaint text.")
    category: str = Field(..., description="Classified category.")
    urgency: str = Field(..., description="Classified urgency level.")
    confidence_score: float = Field(0.0, description="AI confidence in the classification.")
    needs_human_review: bool = Field(False, description="Whether the ticket was escalated.")
    escalation_reason: str = Field("", description="Reason for escalation, if applicable.")
    resident_message: str = Field(..., description="The empathetic response drafted for the resident.")
    vendor_name: Optional[str] = Field(None, description="Assigned vendor.")
    sla_hours: Optional[int] = Field(None, description="SLA deadline in hours.")
    status: str = Field("open", description="Current ticket status.")
    created_at: str = Field(..., description="When the ticket was created.")


class ResponseListResponse(BaseModel):
    """Response for the GET /responses endpoint."""

    responses: list[ResponseItem]
    total: int
