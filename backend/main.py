"""FastAPI Main Application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os

from core.config import settings
from api import (
    data_router,
    analysis_router,
    chat_router,
    viz_router,
    session_router,
    code_router,
    export_router
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    print(f"Starting {settings.PROJECT_NAME}")
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.FIGURES_DIR, exist_ok=True)
    yield
    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="""
# Medical Statistical Analysis Platform

A comprehensive platform for automated statistical analysis of medical data
with LLM-powered recommendations and interpretations.

## Features

- **Data Upload**: Support for CSV and Excel files
- **Automatic Profiling**: Intelligent data type detection and quality assessment
- **Statistical Analysis**: 74+ validated statistical methods
- **Code Generation**: Reproducible Python and R code for all analyses
- **Export**: Word, PDF, LaTeX, HTML, and Markdown reports
- **LLM Integration**: Research question analysis and result interpretation
- **Visualization**: Publication-ready figures
- **Conversational Interface**: Natural language interaction

## Workflow

1. Upload your data file
2. Ask your research question
3. Review recommended analyses
4. Execute analyses with real-time progress
5. Get interpreted results and visualizations
6. Export reproducible code and publication-ready reports
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for figures
if os.path.exists(settings.FIGURES_DIR):
    app.mount("/figures", StaticFiles(directory=settings.FIGURES_DIR), name="figures")

# Include routers
app.include_router(session_router, prefix=settings.API_V1_PREFIX)
app.include_router(data_router, prefix=settings.API_V1_PREFIX)
app.include_router(analysis_router, prefix=settings.API_V1_PREFIX)
app.include_router(chat_router, prefix=settings.API_V1_PREFIX)
app.include_router(viz_router, prefix=settings.API_V1_PREFIX)
app.include_router(code_router, prefix=settings.API_V1_PREFIX)
app.include_router(export_router, prefix=settings.API_V1_PREFIX)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.PROJECT_NAME,
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
