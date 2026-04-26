"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime


# ==================== ENUMS ====================

class AnalysisCategory(str, Enum):
    DESCRIPTIVE = "descriptive"
    COMPARATIVE = "comparative"
    CORRELATION = "correlation"
    REGRESSION = "regression"
    SURVIVAL = "survival"
    DIAGNOSTIC = "diagnostic"
    AGREEMENT = "agreement"


class VariableType(str, Enum):
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    ORDINAL = "ordinal"
    DATETIME = "datetime"
    TEXT = "text"


# ==================== DATA MODELS ====================

class VariableInfo(BaseModel):
    """Information about a single variable."""
    name: str
    dtype: Optional[str] = None
    statistical_type: Optional[str] = None
    n_total: Optional[int] = None
    n_missing: Optional[int] = 0
    missing_pct: Optional[float] = 0.0
    n_unique: Optional[int] = None
    summary_stats: Optional[Dict[str, Any]] = None


class DataProfile(BaseModel):
    """Complete data profile."""
    session_id: str
    filename: str
    n_rows: int
    n_columns: int
    n_continuous: int
    n_categorical: int
    n_binary: int
    variables: List[Dict[str, Any]] = []
    potential_outcomes: List[str] = []
    potential_groups: List[str] = []
    warnings: List[str] = []
    uploaded_at: datetime = Field(default_factory=datetime.now)


class DataUploadResponse(BaseModel):
    """Response after data upload."""
    success: bool
    session_id: str
    message: str
    profile: Optional[DataProfile] = None
    preview: Optional[List[Dict[str, Any]]] = None


# ==================== ANALYSIS MODELS ====================

class AnalysisRequest(BaseModel):
    """Request for statistical analysis."""
    session_id: str
    analysis_type: str
    parameters: Dict[str, Any]


class AnalysisResult(BaseModel):
    """Result of a statistical analysis."""
    analysis_id: str
    analysis_type: str
    test_name: str
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    effect_size_name: Optional[str] = None
    confidence_interval: Optional[List[float]] = None
    conclusion: Optional[str] = None
    details: Dict[str, Any] = {}
    interpretation: Optional[str] = None
    figures: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)


class AnalysisResponse(BaseModel):
    """Response containing analysis results."""
    success: bool
    message: str
    results: Optional[AnalysisResult] = None
    errors: List[str] = []


# ==================== LLM/CHAT MODELS ====================

class ChatMessage(BaseModel):
    """Single chat message."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatRequest(BaseModel):
    """Request for chat interaction."""
    session_id: str
    message: str
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Response from chat."""
    message: str
    analysis_suggestions: Optional[List[Dict[str, Any]]] = None
    action_required: Optional[str] = None


class AnalysisPlanItem(BaseModel):
    """Single item in analysis plan."""
    test_name: str
    category: AnalysisCategory
    priority: int
    rationale: str
    variables: Dict[str, str]
    assumptions: List[str]


class AnalysisPlan(BaseModel):
    """Complete analysis plan from LLM."""
    research_question: str
    research_type: str
    primary_analyses: List[AnalysisPlanItem]
    secondary_analyses: List[AnalysisPlanItem]
    assumption_checks: List[str]
    visualizations: List[str]
    notes: List[str]


class ResearchQuestionRequest(BaseModel):
    """Request to analyze research question."""
    session_id: str
    question: str


class ResearchQuestionResponse(BaseModel):
    """Response with analysis plan."""
    success: bool
    plan: Optional[AnalysisPlan] = None
    message: str


# ==================== VISUALIZATION MODELS ====================

class VisualizationRequest(BaseModel):
    """Request for visualization."""
    session_id: str
    plot_type: str
    variables: Dict[str, str]
    options: Optional[Dict[str, Any]] = None


class VisualizationResponse(BaseModel):
    """Response with visualization."""
    success: bool
    figure_url: Optional[str] = None
    figure_base64: Optional[str] = None
    message: str


# ==================== REPORT MODELS ====================

class ReportRequest(BaseModel):
    """Request for report generation."""
    session_id: str
    include_analyses: List[str]
    format: str = "pdf"  # pdf, docx, html


class ReportResponse(BaseModel):
    """Response with report."""
    success: bool
    report_url: Optional[str] = None
    message: str


# ==================== SESSION MODELS ====================

class SessionInfo(BaseModel):
    """Information about analysis session."""
    session_id: str
    created_at: datetime
    data_loaded: bool
    filename: Optional[str] = None
    n_analyses_performed: int = 0
    chat_history: List[ChatMessage] = []


class SessionListResponse(BaseModel):
    """List of sessions."""
    sessions: List[SessionInfo]
