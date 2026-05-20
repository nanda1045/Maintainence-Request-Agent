"""
agent/tool_definitions.py
Claude tool schemas exposed to the maintenance triage orchestrator.

The actual Python implementations live in agent/tools.py. Keeping schemas in
one file makes the orchestrator easier to scan during code walkthroughs.
"""


CATEGORIES = [
    "plumbing",
    "hvac",
    "electrical",
    "pest_control",
    "appliance",
    "general_maintenance",
]

URGENCY_LEVELS = ["critical", "high", "medium", "low"]

TOOL_DEFINITIONS = [
    {
        "name": "retrieve_similar_tickets",
        "description": (
            "Search the historical knowledge base (ChromaDB) to find past maintenance "
            "tickets similar to the current complaint. Returns the top 5 most similar "
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
            "Classify the maintenance complaint into a category, urgency level, "
            "confidence score, and whether human review is needed. "
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
                    "enum": CATEGORIES,
                },
                "urgency": {
                    "type": "string",
                    "description": "The classified urgency level.",
                    "enum": URGENCY_LEVELS,
                },
            },
            "required": ["category", "urgency"],
        },
    },
    {
        "name": "draft_response",
        "description": (
            "Draft an empathetic response message to send to the resident. "
            "Includes acknowledgment, vendor info, expected timeline, and "
            "mentions human review if escalated. Call this AFTER get_vendor."
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
                "requires_human_review": {
                    "type": "boolean",
                    "description": "Whether the ticket has been escalated for human review.",
                },
                "escalation_reason": {
                    "type": "string",
                    "description": "Reason for escalation, if applicable.",
                },
            },
            "required": [
                "complaint",
                "unit",
                "resident_name",
                "category",
                "urgency",
                "vendor_name",
                "vendor_phone",
                "sla_hours",
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
                "confidence_score": {
                    "type": "number",
                    "description": "AI confidence in classification (0.0-1.0).",
                },
                "requires_human_review": {
                    "type": "boolean",
                    "description": "Whether the ticket was escalated for human review.",
                },
                "escalation_reason": {
                    "type": "string",
                    "description": "Reason for escalation, if applicable.",
                },
            },
            "required": [
                "unit",
                "resident_name",
                "complaint",
                "category",
                "urgency",
                "vendor_name",
                "vendor_phone",
                "vendor_email",
                "sla_hours",
                "similar_ticket_ids",
                "resident_message",
            ],
        },
    },
]
