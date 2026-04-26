"""Reporting Checklists Module.

Provides compliance checking for reporting guidelines:
- STROBE (observational studies)
- CONSORT (RCTs)
- PRISMA (systematic reviews)
- STARD (diagnostic studies)
- TRIPOD (prediction models)
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ChecklistItem:
    """Single checklist item."""

    def __init__(
        self,
        number: str,
        section: str,
        item: str,
        description: str,
        required: bool = True
    ):
        self.number = number
        self.section = section
        self.item = item
        self.description = description
        self.required = required
        self.completed = False
        self.location = ""  # Page/paragraph location
        self.notes = ""


class ChecklistStatus(Enum):
    """Checklist completion status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class ChecklistResult:
    """Result of checklist assessment."""
    guideline: str
    total_items: int
    completed_items: int
    required_items: int
    required_completed: int
    completion_percentage: float
    status: ChecklistStatus
    items: List[ChecklistItem]
    missing_required: List[str]
    recommendations: List[str]


class ComplianceChecker:
    """Check compliance with reporting guidelines."""

    def __init__(self):
        self.checklists = {
            'strobe': self._build_strobe_checklist(),
            'consort': self._build_consort_checklist(),
            'prisma': self._build_prisma_checklist(),
            'stard': self._build_stard_checklist()
        }

    def get_checklist(self, guideline: str) -> List[ChecklistItem]:
        """Get checklist for a specific guideline.

        Args:
            guideline: Guideline name (strobe, consort, prisma, stard)

        Returns:
            List of ChecklistItem objects
        """
        return self.checklists.get(guideline.lower(), [])

    def assess_compliance(
        self,
        guideline: str,
        report_text: str,
        completed_items: Optional[List[str]] = None
    ) -> ChecklistResult:
        """Assess compliance with reporting guideline.

        Args:
            guideline: Guideline name
            report_text: Full text of the report
            completed_items: List of completed item numbers

        Returns:
            ChecklistResult with assessment
        """
        checklist = self.get_checklist(guideline)

        if not checklist:
            raise ValueError(f"Unknown guideline: {guideline}")

        completed_items = completed_items or []

        # Mark completed items
        for item in checklist:
            item.completed = item.number in completed_items

            # Auto-detect some items from text
            if not item.completed:
                item.completed = self._auto_detect_item(item, report_text)

        # Calculate statistics
        total = len(checklist)
        completed = sum(1 for item in checklist if item.completed)
        required = sum(1 for item in checklist if item.required)
        required_completed = sum(
            1 for item in checklist
            if item.required and item.completed
        )

        completion_pct = (completed / total * 100) if total > 0 else 0

        # Determine status
        if completed == total:
            status = ChecklistStatus.COMPLETE
        elif completed > 0:
            status = ChecklistStatus.IN_PROGRESS
        else:
            status = ChecklistStatus.NOT_STARTED

        # Find missing required items
        missing_required = [
            f"{item.number}: {item.item}"
            for item in checklist
            if item.required and not item.completed
        ]

        # Generate recommendations
        recommendations = self._generate_recommendations(
            checklist, missing_required, guideline
        )

        return ChecklistResult(
            guideline=guideline.upper(),
            total_items=total,
            completed_items=completed,
            required_items=required,
            required_completed=required_completed,
            completion_percentage=completion_pct,
            status=status,
            items=checklist,
            missing_required=missing_required,
            recommendations=recommendations
        )

    def generate_checklist_document(
        self,
        guideline: str,
        result: ChecklistResult
    ) -> str:
        """Generate formatted checklist document.

        Args:
            guideline: Guideline name
            result: ChecklistResult from assessment

        Returns:
            Formatted checklist as string
        """
        lines = [
            f"# {guideline.upper()} Checklist",
            f"Completion: {result.completion_percentage:.1f}% ({result.completed_items}/{result.total_items})",
            f"Required items: {result.required_completed}/{result.required_items}",
            "",
            "## Items",
            ""
        ]

        current_section = ""
        for item in result.items:
            if item.section != current_section:
                current_section = item.section
                lines.append(f"### {current_section}")
                lines.append("")

            status = "✓" if item.completed else "☐"
            required = "*" if item.required else ""
            lines.append(f"- [{status}] **{item.number}{required}** {item.item}")
            if item.location:
                lines.append(f"  - Location: {item.location}")
            lines.append("")

        if result.missing_required:
            lines.extend([
                "## Missing Required Items",
                ""
            ])
            for missing in result.missing_required:
                lines.append(f"- {missing}")

        if result.recommendations:
            lines.extend([
                "",
                "## Recommendations",
                ""
            ])
            for rec in result.recommendations:
                lines.append(f"- {rec}")

        return '\n'.join(lines)

    def _auto_detect_item(self, item: ChecklistItem, text: str) -> bool:
        """Auto-detect if item is present in text."""
        text_lower = text.lower()

        # Keywords for common items
        detection_keywords = {
            'title': ['title', 'study design'],
            'abstract': ['abstract', 'background', 'methods', 'results', 'conclusions'],
            'objectives': ['objective', 'aim', 'purpose', 'hypothesis'],
            'setting': ['setting', 'institution', 'hospital', 'clinic'],
            'participants': ['participants', 'patients', 'subjects', 'inclusion criteria'],
            'variables': ['variables', 'outcome', 'exposure'],
            'statistical': ['statistical analysis', 'statistics', 'p-value'],
            'results': ['results', 'findings'],
            'discussion': ['discussion', 'interpretation'],
            'limitations': ['limitations', 'weaknesses'],
            'funding': ['funding', 'grant', 'supported by']
        }

        item_key = item.item.lower().split()[0]
        keywords = detection_keywords.get(item_key, [item_key])

        return any(kw in text_lower for kw in keywords)

    def _generate_recommendations(
        self,
        checklist: List[ChecklistItem],
        missing_required: List[str],
        guideline: str
    ) -> List[str]:
        """Generate recommendations for improving compliance."""
        recommendations = []

        if missing_required:
            recommendations.append(
                f"Address {len(missing_required)} missing required items for full {guideline.upper()} compliance"
            )

        # Specific recommendations based on guideline
        if guideline == 'strobe':
            if any('confounding' in m.lower() for m in missing_required):
                recommendations.append(
                    "Describe how confounding was addressed (adjustment, matching, etc.)"
                )
            if any('bias' in m.lower() for m in missing_required):
                recommendations.append(
                    "Discuss potential sources of bias and their direction"
                )

        elif guideline == 'consort':
            if any('randomization' in m.lower() for m in missing_required):
                recommendations.append(
                    "Fully describe randomization procedure including sequence generation and concealment"
                )
            if any('blinding' in m.lower() for m in missing_required):
                recommendations.append(
                    "Specify who was blinded (participants, care providers, outcome assessors)"
                )

        elif guideline == 'prisma':
            if any('search' in m.lower() for m in missing_required):
                recommendations.append(
                    "Provide full search strategy for at least one database"
                )
            if any('risk of bias' in m.lower() for m in missing_required):
                recommendations.append(
                    "Use a validated tool (e.g., RoB 2, ROBINS-I) for risk of bias assessment"
                )

        return recommendations

    def _build_strobe_checklist(self) -> List[ChecklistItem]:
        """Build STROBE checklist for observational studies."""
        items = [
            # Title and abstract
            ChecklistItem("1a", "Title and abstract", "Title",
                         "Indicate the study's design with a commonly used term in the title or abstract"),
            ChecklistItem("1b", "Title and abstract", "Abstract",
                         "Provide an informative and balanced summary of what was done and found"),

            # Introduction
            ChecklistItem("2", "Introduction", "Background/rationale",
                         "Explain the scientific background and rationale for the investigation"),
            ChecklistItem("3", "Introduction", "Objectives",
                         "State specific objectives, including any prespecified hypotheses"),

            # Methods
            ChecklistItem("4", "Methods", "Study design",
                         "Present key elements of study design early in the paper"),
            ChecklistItem("5", "Methods", "Setting",
                         "Describe the setting, locations, and relevant dates"),
            ChecklistItem("6", "Methods", "Participants",
                         "Give eligibility criteria, sources, methods of selection and follow-up"),
            ChecklistItem("7", "Methods", "Variables",
                         "Clearly define all outcomes, exposures, predictors, confounders"),
            ChecklistItem("8", "Methods", "Data sources",
                         "Describe sources of data and details of methods of assessment"),
            ChecklistItem("9", "Methods", "Bias",
                         "Describe any efforts to address potential sources of bias"),
            ChecklistItem("10", "Methods", "Study size",
                         "Explain how the study size was arrived at"),
            ChecklistItem("11", "Methods", "Quantitative variables",
                         "Explain how quantitative variables were handled in the analyses"),
            ChecklistItem("12", "Methods", "Statistical methods",
                         "Describe all statistical methods, including those for confounding"),

            # Results
            ChecklistItem("13", "Results", "Participants",
                         "Report numbers of individuals at each stage of study"),
            ChecklistItem("14", "Results", "Descriptive data",
                         "Give characteristics of participants and information on exposures"),
            ChecklistItem("15", "Results", "Outcome data",
                         "Report numbers of outcome events or summary measures"),
            ChecklistItem("16", "Results", "Main results",
                         "Give unadjusted and adjusted estimates with confidence intervals"),
            ChecklistItem("17", "Results", "Other analyses",
                         "Report other analyses done—e.g., sensitivity analyses"),

            # Discussion
            ChecklistItem("18", "Discussion", "Key results",
                         "Summarise key results with reference to study objectives"),
            ChecklistItem("19", "Discussion", "Limitations",
                         "Discuss limitations and potential bias"),
            ChecklistItem("20", "Discussion", "Interpretation",
                         "Give a cautious overall interpretation of results"),
            ChecklistItem("21", "Discussion", "Generalisability",
                         "Discuss the generalisability of the study results"),

            # Other
            ChecklistItem("22", "Other", "Funding",
                         "Give the source of funding and role of funders")
        ]

        return items

    def _build_consort_checklist(self) -> List[ChecklistItem]:
        """Build CONSORT 2010 checklist for RCTs."""
        items = [
            # Title and abstract
            ChecklistItem("1a", "Title and abstract", "Title",
                         "Identification as a randomised trial in the title"),
            ChecklistItem("1b", "Title and abstract", "Abstract",
                         "Structured summary of trial design, methods, results, conclusions"),

            # Introduction
            ChecklistItem("2a", "Introduction", "Background",
                         "Scientific background and explanation of rationale"),
            ChecklistItem("2b", "Introduction", "Objectives",
                         "Specific objectives or hypotheses"),

            # Methods
            ChecklistItem("3a", "Methods", "Trial design",
                         "Description of trial design including allocation ratio"),
            ChecklistItem("4a", "Methods", "Participants",
                         "Eligibility criteria for participants"),
            ChecklistItem("4b", "Methods", "Settings",
                         "Settings and locations where data were collected"),
            ChecklistItem("5", "Methods", "Interventions",
                         "Interventions for each group with sufficient details"),
            ChecklistItem("6a", "Methods", "Outcomes",
                         "Completely defined primary and secondary outcomes"),
            ChecklistItem("7a", "Methods", "Sample size",
                         "How sample size was determined"),
            ChecklistItem("8a", "Methods", "Randomisation sequence",
                         "Method used to generate random allocation sequence"),
            ChecklistItem("8b", "Methods", "Randomisation type",
                         "Type of randomisation and any restriction"),
            ChecklistItem("9", "Methods", "Allocation concealment",
                         "Mechanism used to implement allocation concealment"),
            ChecklistItem("10", "Methods", "Implementation",
                         "Who generated sequence, enrolled, and assigned participants"),
            ChecklistItem("11a", "Methods", "Blinding",
                         "Who was blinded after assignment and how"),
            ChecklistItem("12a", "Methods", "Statistical methods",
                         "Statistical methods for comparing groups"),

            # Results
            ChecklistItem("13a", "Results", "Participant flow",
                         "Numbers randomised, allocated, followed up, analysed"),
            ChecklistItem("14a", "Results", "Recruitment",
                         "Dates of recruitment and follow-up"),
            ChecklistItem("15", "Results", "Baseline data",
                         "Baseline demographic and clinical characteristics"),
            ChecklistItem("16", "Results", "Numbers analysed",
                         "Number of participants in each analysis and if ITT"),
            ChecklistItem("17a", "Results", "Outcomes and estimation",
                         "Results for each primary and secondary outcome"),
            ChecklistItem("18", "Results", "Ancillary analyses",
                         "Results of any other analyses including subgroup"),
            ChecklistItem("19", "Results", "Harms",
                         "All important harms or unintended effects"),

            # Discussion
            ChecklistItem("20", "Discussion", "Limitations",
                         "Trial limitations addressing sources of potential bias"),
            ChecklistItem("21", "Discussion", "Generalisability",
                         "Generalisability of the trial findings"),
            ChecklistItem("22", "Discussion", "Interpretation",
                         "Interpretation consistent with results and balancing benefits/harms"),

            # Other
            ChecklistItem("23", "Other", "Registration",
                         "Registration number and name of trial registry"),
            ChecklistItem("24", "Other", "Protocol",
                         "Where full trial protocol can be accessed"),
            ChecklistItem("25", "Other", "Funding",
                         "Sources of funding and other support")
        ]

        return items

    def _build_prisma_checklist(self) -> List[ChecklistItem]:
        """Build PRISMA 2020 checklist for systematic reviews."""
        items = [
            # Title
            ChecklistItem("1", "Title", "Title",
                         "Identify the report as a systematic review"),

            # Abstract
            ChecklistItem("2", "Abstract", "Abstract",
                         "Structured summary following PRISMA abstract checklist"),

            # Introduction
            ChecklistItem("3", "Introduction", "Rationale",
                         "Rationale for the review in context of existing knowledge"),
            ChecklistItem("4", "Introduction", "Objectives",
                         "Explicit statement of the questions being addressed"),

            # Methods
            ChecklistItem("5", "Methods", "Eligibility criteria",
                         "Specify inclusion and exclusion criteria"),
            ChecklistItem("6", "Methods", "Information sources",
                         "Specify databases, registers, other sources searched"),
            ChecklistItem("7", "Methods", "Search strategy",
                         "Present full search strategies for all databases"),
            ChecklistItem("8", "Methods", "Selection process",
                         "Specify methods for study selection"),
            ChecklistItem("9", "Methods", "Data collection",
                         "Specify methods for data extraction"),
            ChecklistItem("10", "Methods", "Data items",
                         "List and define all outcome variables"),
            ChecklistItem("11", "Methods", "Study risk of bias",
                         "Specify methods for risk of bias assessment"),
            ChecklistItem("12", "Methods", "Effect measures",
                         "Specify effect measures used"),
            ChecklistItem("13", "Methods", "Synthesis methods",
                         "Describe methods for synthesis of results"),

            # Results
            ChecklistItem("16", "Results", "Study selection",
                         "Numbers of studies screened, assessed, included"),
            ChecklistItem("17", "Results", "Study characteristics",
                         "Characteristics and citations of included studies"),
            ChecklistItem("18", "Results", "Risk of bias",
                         "Present risk of bias assessment results"),
            ChecklistItem("20", "Results", "Results of syntheses",
                         "Results of all syntheses with CIs and measures of heterogeneity"),

            # Discussion
            ChecklistItem("23", "Discussion", "Discussion",
                         "General interpretation of results in context of other evidence"),
            ChecklistItem("24", "Discussion", "Limitations",
                         "Limitations of evidence and review process"),
            ChecklistItem("25", "Discussion", "Conclusions",
                         "General interpretation and implications"),

            # Other
            ChecklistItem("26", "Other", "Registration",
                         "Registration information for the review"),
            ChecklistItem("27", "Other", "Funding",
                         "Sources of support for the review")
        ]

        return items

    def _build_stard_checklist(self) -> List[ChecklistItem]:
        """Build STARD 2015 checklist for diagnostic accuracy studies."""
        items = [
            ChecklistItem("1", "Title/Abstract", "Title",
                         "Identify as study of diagnostic accuracy using sensitivity/specificity"),
            ChecklistItem("2", "Title/Abstract", "Abstract",
                         "Structured summary of study design, methods, results, conclusions"),

            ChecklistItem("3", "Introduction", "Background",
                         "Scientific and clinical background including intended use"),
            ChecklistItem("4", "Introduction", "Objectives",
                         "Study objectives and hypotheses"),

            ChecklistItem("5", "Methods", "Study design",
                         "Prospective or retrospective data collection"),
            ChecklistItem("6", "Methods", "Participants",
                         "Eligibility criteria"),
            ChecklistItem("7", "Methods", "Participants sampling",
                         "On what basis potentially eligible were identified"),
            ChecklistItem("8", "Methods", "Participant recruitment",
                         "Where and when potentially eligible were identified"),
            ChecklistItem("9", "Methods", "Sample size",
                         "How sample size was determined"),
            ChecklistItem("10a", "Methods", "Index test",
                         "Index test, how and when performed"),
            ChecklistItem("10b", "Methods", "Reference standard",
                         "Reference standard, how and when performed"),
            ChecklistItem("11", "Methods", "Rationale for threshold",
                         "Rationale for choosing test positivity cutoffs"),
            ChecklistItem("12a", "Methods", "Blinding",
                         "Whether assessors were blinded to other test results"),
            ChecklistItem("13a", "Methods", "Statistical methods",
                         "Methods for diagnostic accuracy measures"),

            ChecklistItem("19", "Results", "Flow of participants",
                         "Flow of participants through the study"),
            ChecklistItem("21a", "Results", "Estimates",
                         "Estimates of diagnostic accuracy and their precision"),
            ChecklistItem("22", "Results", "Adverse events",
                         "Any adverse events from performing tests"),

            ChecklistItem("23", "Discussion", "Limitations",
                         "Study limitations including sources of bias"),
            ChecklistItem("24", "Discussion", "Implications",
                         "Implications for practice including intended use"),

            ChecklistItem("27", "Other", "Funding",
                         "Sources of funding")
        ]

        return items
