"""Analysis Pipeline Package.

Provides automated analysis capabilities:
- automated_analysis: One-click full analysis pipeline
- analysis_templates: Pre-built study templates
- natural_language: Natural language query interface
- data_quality: Data quality dashboard
"""

from .automated_analysis import AutomatedAnalysisPipeline, StudyDesign, VariableRole
from .analysis_templates import AnalysisTemplates
from .natural_language import NaturalLanguageAnalysis
from .data_quality import DataQualityDashboard

__all__ = [
    "AutomatedAnalysisPipeline",
    "StudyDesign",
    "VariableRole",
    "AnalysisTemplates",
    "NaturalLanguageAnalysis",
    "DataQualityDashboard",
]
