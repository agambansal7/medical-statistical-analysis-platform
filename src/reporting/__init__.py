"""Reporting Package.

Provides report generation capabilities:
- report_generator: Complete Methods + Results sections
- journal_templates: NEJM, JAMA, Lancet formats
- export: Word/LaTeX/PDF export
- checklists: STROBE/CONSORT compliance
"""

from .report_generator import ReportGenerator, ReportSection
from .journal_templates import JournalFormatter
from .export import ReportExporter
from .checklists import ComplianceChecker

__all__ = [
    "ReportGenerator",
    "ReportSection",
    "JournalFormatter",
    "ReportExporter",
    "ComplianceChecker",
]
