"""
api/main.py
FastAPI application entry point for the Maintenance Triage Agent.

Run with:
    uvicorn api.main:app --reload

API docs available at:
    http://localhost:8000/docs     (Swagger UI)
    http://localhost:8000/redoc    (ReDoc)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from db_manager.sqlite_manager import init_db

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("maintenance_agent")


# ---------------------------------------------------------------------------
# Lifespan — runs on startup/shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, clean up on shutdown."""
    logger.info("🚀 Starting Maintenance Triage Agent API...")
    init_db()
    logger.info("✅ Database initialized")
    logger.info("📍 API docs at: http://localhost:8000/docs")
    yield
    logger.info("👋 Shutting down Maintenance Triage Agent API")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Maintenance Triage Agent API",
    description=(
        "An agentic AI system that receives resident maintenance complaints, "
        "retrieves similar past tickets from a knowledge base, classifies urgency, "
        "routes to the right vendor, drafts an empathetic response, and logs everything."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — allow all origins for local development / demos
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Include routes
# ---------------------------------------------------------------------------
app.include_router(router)
