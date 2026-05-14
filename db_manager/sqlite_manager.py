"""
db_manager/sqlite_manager.py
SQLite read/write operations for maintenance ticket logging.

Schema:
  - tickets: stores every processed maintenance ticket with full triage results
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional


# Default database path — db/ directory at project root
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db")
DB_PATH = os.path.join(DB_DIR, "tickets.db")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Get a SQLite connection with row factory enabled."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """
    Initialize the database and create the tickets table if it doesn't exist.

    Columns:
      - id                : auto-increment primary key
      - ticket_id         : unique string identifier (e.g., "TK-20250513-0001")
      - unit              : apartment/unit number
      - resident_name     : name of the resident who filed the complaint
      - complaint         : full text of the resident's complaint
      - category          : classified category (plumbing, hvac, electrical,
                            pest_control, appliance, general_maintenance)
      - urgency           : classified urgency level (critical, high, medium, low)
      - vendor_name       : assigned vendor name
      - vendor_phone      : assigned vendor phone
      - vendor_email      : assigned vendor email
      - sla_hours         : SLA deadline in hours
      - similar_tickets   : JSON array of similar ticket IDs from RAG retrieval
      - resident_message  : the empathetic response drafted for the resident
      - confidence_score  : AI confidence in classification (0.0 - 1.0)
      - requires_human_review : whether the ticket was escalated for human review
      - escalation_reason : reason for escalation (if applicable)
      - status            : ticket status (open, in_progress, resolved, closed)
      - created_at        : timestamp when ticket was created
      - updated_at        : timestamp when ticket was last updated
    """
    conn = get_connection(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id       TEXT UNIQUE NOT NULL,
                unit            TEXT NOT NULL,
                resident_name   TEXT NOT NULL,
                complaint       TEXT NOT NULL,
                category        TEXT NOT NULL,
                urgency         TEXT NOT NULL,
                vendor_name     TEXT,
                vendor_phone    TEXT,
                vendor_email    TEXT,
                sla_hours       INTEGER,
                similar_tickets TEXT DEFAULT '[]',
                resident_message TEXT,
                confidence_score REAL DEFAULT 0.0,
                requires_human_review INTEGER DEFAULT 0,
                escalation_reason TEXT DEFAULT '',
                status          TEXT DEFAULT 'open',
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL
            )
        """)

        _ensure_column(conn, "tickets", "confidence_score", "REAL DEFAULT 0.0")
        _ensure_column(conn, "tickets", "requires_human_review", "INTEGER DEFAULT 0")
        _ensure_column(conn, "tickets", "escalation_reason", "TEXT DEFAULT ''")

        # Index on category + urgency for quick lookups
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tickets_category_urgency
            ON tickets (category, urgency)
        """)

        # Index on status for filtering active tickets
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tickets_status
            ON tickets (status)
        """)

        conn.commit()
    finally:
        conn.close()


def _ensure_column(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_definition: str,
) -> None:
    """Add a column to an existing SQLite table if it is missing."""
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    columns = {row["name"] for row in cursor.fetchall()}
    if column_name not in columns:
        conn.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
        )


def generate_ticket_id() -> str:
    """Generate a unique ticket ID in format TK-YYYYMMDD-XXXX."""
    date_str = datetime.now().strftime("%Y%m%d")
    conn = get_connection()
    try:
        cursor = conn.execute(
            "SELECT COUNT(*) FROM tickets WHERE ticket_id LIKE ?",
            (f"TK-{date_str}-%",)
        )
        count = cursor.fetchone()[0]
        return f"TK-{date_str}-{count + 1:04d}"
    finally:
        conn.close()


def insert_ticket(
    ticket_id: str,
    unit: str,
    resident_name: str,
    complaint: str,
    category: str,
    urgency: str,
    vendor_name: Optional[str] = None,
    vendor_phone: Optional[str] = None,
    vendor_email: Optional[str] = None,
    sla_hours: Optional[int] = None,
    similar_tickets: Optional[list] = None,
    resident_message: Optional[str] = None,
    confidence_score: float = 0.0,
    requires_human_review: bool = False,
    escalation_reason: str = "",
    status: str = "open",
    db_path: str = DB_PATH,
) -> dict:
    """
    Insert a new ticket into the database.

    Returns the full ticket record as a dictionary.
    """
    now = datetime.now().isoformat()
    similar_json = json.dumps(similar_tickets or [])

    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO tickets (
                ticket_id, unit, resident_name, complaint, category, urgency,
                vendor_name, vendor_phone, vendor_email, sla_hours,
                similar_tickets, resident_message,
                confidence_score, requires_human_review, escalation_reason,
                status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticket_id, unit, resident_name, complaint, category, urgency,
                vendor_name, vendor_phone, vendor_email, sla_hours,
                similar_json, resident_message,
                confidence_score, 1 if requires_human_review else 0, escalation_reason,
                status, now, now,
            ),
        )
        conn.commit()

        return get_ticket_by_id(ticket_id, db_path)
    finally:
        conn.close()


def get_ticket_by_id(ticket_id: str, db_path: str = DB_PATH) -> Optional[dict]:
    """Retrieve a single ticket by its ticket_id."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return _row_to_dict(row)
    finally:
        conn.close()


def get_all_tickets(
    status: Optional[str] = None,
    category: Optional[str] = None,
    urgency: Optional[str] = None,
    limit: int = 50,
    db_path: str = DB_PATH,
) -> list[dict]:
    """
    Retrieve tickets with optional filters.

    Args:
        status: Filter by status (open, in_progress, resolved, closed)
        category: Filter by category
        urgency: Filter by urgency level
        limit: Maximum number of tickets to return
    """
    conn = get_connection(db_path)
    try:
        query = "SELECT * FROM tickets WHERE 1=1"
        params: list = []

        if status:
            query += " AND status = ?"
            params.append(status)
        if category:
            query += " AND category = ?"
            params.append(category)
        if urgency:
            query += " AND urgency = ?"
            params.append(urgency)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(query, params)
        return [_row_to_dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def update_ticket_status(
    ticket_id: str, status: str, db_path: str = DB_PATH
) -> Optional[dict]:
    """Update the status of a ticket."""
    now = datetime.now().isoformat()
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE tickets SET status = ?, updated_at = ? WHERE ticket_id = ?",
            (status, now, ticket_id),
        )
        conn.commit()
        return get_ticket_by_id(ticket_id, db_path)
    finally:
        conn.close()


def get_ticket_stats(db_path: str = DB_PATH) -> dict:
    """Get summary statistics of all tickets."""
    conn = get_connection(db_path)
    try:
        stats = {}

        # Total tickets
        cursor = conn.execute("SELECT COUNT(*) FROM tickets")
        stats["total"] = cursor.fetchone()[0]

        # By status
        cursor = conn.execute(
            "SELECT status, COUNT(*) as count FROM tickets GROUP BY status"
        )
        stats["by_status"] = {row["status"]: row["count"] for row in cursor.fetchall()}

        # By urgency
        cursor = conn.execute(
            "SELECT urgency, COUNT(*) as count FROM tickets GROUP BY urgency"
        )
        stats["by_urgency"] = {row["urgency"]: row["count"] for row in cursor.fetchall()}

        # By category
        cursor = conn.execute(
            "SELECT category, COUNT(*) as count FROM tickets GROUP BY category"
        )
        stats["by_category"] = {row["category"]: row["count"] for row in cursor.fetchall()}

        return stats
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a sqlite3.Row to a plain dictionary, parsing JSON fields."""
    d = dict(row)
    # Parse the similar_tickets JSON string back to a list
    if "similar_tickets" in d and isinstance(d["similar_tickets"], str):
        try:
            d["similar_tickets"] = json.loads(d["similar_tickets"])
        except json.JSONDecodeError:
            d["similar_tickets"] = []
    # Convert requires_human_review from int to bool
    if "requires_human_review" in d:
        d["requires_human_review"] = bool(d["requires_human_review"])
    return d


# Auto-initialize the database when this module is imported
init_db()
