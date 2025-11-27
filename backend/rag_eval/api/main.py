"""FastAPI application entrypoint"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rag_eval.api.routes import query, metrics, meta_eval
from rag_eval.core.config import Config
from rag_eval.core.logging import setup_logging

# Setup logging
logger = setup_logging()

# Load configuration
config = Config.from_env()

# Create FastAPI app
app = FastAPI(
    title="RAG Evaluation Platform API",
    description="API for RAG pipeline, evaluation, and meta-evaluation",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(query.router, prefix="/api", tags=["query"])
app.include_router(metrics.router, prefix="/api", tags=["metrics"])
app.include_router(meta_eval.router, prefix="/api", tags=["meta_eval"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "RAG Evaluation Platform API", "version": "0.1.0"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

