"""LLM orchestration modules."""

# Use lazy imports to avoid circular import issues
__all__ = [
    "LLMOrchestrator",
    "PromptTemplates",
    "ComprehensiveStatisticalPlanner",
    "PlanExecutor",
    "ReportGenerator"
]

def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == "LLMOrchestrator":
        from .orchestrator import LLMOrchestrator
        return LLMOrchestrator
    elif name == "PromptTemplates":
        from .prompts import PromptTemplates
        return PromptTemplates
    elif name == "ComprehensiveStatisticalPlanner":
        from .comprehensive_planner import ComprehensiveStatisticalPlanner
        return ComprehensiveStatisticalPlanner
    elif name == "PlanExecutor":
        from .plan_executor import PlanExecutor
        return PlanExecutor
    elif name == "ReportGenerator":
        from .report_generator import ReportGenerator
        return ReportGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
