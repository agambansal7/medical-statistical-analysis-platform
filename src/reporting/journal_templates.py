"""Journal-specific formatting templates.

Supports formatting for major medical journals:
- NEJM (New England Journal of Medicine)
- JAMA (Journal of the American Medical Association)
- Lancet
- BMJ (British Medical Journal)
- Annals of Internal Medicine
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from .report_generator import FullReport, ReportSection


@dataclass
class JournalStyle:
    """Journal-specific style settings."""
    name: str
    abbreviation: str

    # Table formatting
    table_style: str  # "three_line", "full_grid", "minimal"
    p_value_threshold: float
    effect_decimals: int
    percentage_decimals: int

    # Text formatting
    abstract_word_limit: int
    methods_first: bool
    use_subheadings: bool

    # Statistical reporting
    report_ci: bool
    report_effect_sizes: bool
    use_apa_style: bool

    # References
    reference_style: str  # "vancouver", "apa", "numbered"
    max_references: int


class JournalFormatter:
    """Format reports according to journal guidelines."""

    JOURNAL_STYLES = {
        'nejm': JournalStyle(
            name="New England Journal of Medicine",
            abbreviation="NEJM",
            table_style="three_line",
            p_value_threshold=0.05,
            effect_decimals=2,
            percentage_decimals=1,
            abstract_word_limit=250,
            methods_first=False,
            use_subheadings=True,
            report_ci=True,
            report_effect_sizes=True,
            use_apa_style=False,
            reference_style="vancouver",
            max_references=50
        ),
        'jama': JournalStyle(
            name="Journal of the American Medical Association",
            abbreviation="JAMA",
            table_style="three_line",
            p_value_threshold=0.05,
            effect_decimals=2,
            percentage_decimals=1,
            abstract_word_limit=350,
            methods_first=False,
            use_subheadings=True,
            report_ci=True,
            report_effect_sizes=True,
            use_apa_style=False,
            reference_style="vancouver",
            max_references=50
        ),
        'lancet': JournalStyle(
            name="The Lancet",
            abbreviation="Lancet",
            table_style="three_line",
            p_value_threshold=0.05,
            effect_decimals=2,
            percentage_decimals=1,
            abstract_word_limit=300,
            methods_first=True,
            use_subheadings=True,
            report_ci=True,
            report_effect_sizes=True,
            use_apa_style=False,
            reference_style="vancouver",
            max_references=30
        ),
        'bmj': JournalStyle(
            name="British Medical Journal",
            abbreviation="BMJ",
            table_style="three_line",
            p_value_threshold=0.05,
            effect_decimals=2,
            percentage_decimals=1,
            abstract_word_limit=300,
            methods_first=False,
            use_subheadings=True,
            report_ci=True,
            report_effect_sizes=True,
            use_apa_style=False,
            reference_style="vancouver",
            max_references=50
        ),
        'annals': JournalStyle(
            name="Annals of Internal Medicine",
            abbreviation="Ann Intern Med",
            table_style="three_line",
            p_value_threshold=0.05,
            effect_decimals=2,
            percentage_decimals=1,
            abstract_word_limit=350,
            methods_first=False,
            use_subheadings=True,
            report_ci=True,
            report_effect_sizes=True,
            use_apa_style=False,
            reference_style="vancouver",
            max_references=40
        )
    }

    def __init__(self, journal: str = 'nejm'):
        """Initialize formatter for specific journal.

        Args:
            journal: Journal key (nejm, jama, lancet, bmj, annals)
        """
        self.journal = journal.lower()
        self.style = self.JOURNAL_STYLES.get(
            self.journal,
            self.JOURNAL_STYLES['nejm']
        )

    def format_report(self, report: FullReport) -> Dict[str, Any]:
        """Format full report according to journal style.

        Args:
            report: Full report to format

        Returns:
            Dict with formatted sections
        """
        formatted = {
            'journal': self.style.name,
            'title': report.title,
            'sections': {},
            'tables': [],
            'figures': [],
            'references': []
        }

        # Format sections
        for section_name, section in report.sections.items():
            formatted['sections'][section_name] = self._format_section(section)

        # Format tables
        for table in report.tables:
            formatted['tables'].append(self._format_table(table))

        # Format references
        if report.references:
            formatted['references'] = self._format_references(
                report.references[:self.style.max_references]
            )

        return formatted

    def format_abstract(
        self,
        background: str,
        methods: str,
        results: str,
        conclusions: str
    ) -> str:
        """Format structured abstract.

        Args:
            background: Background section
            methods: Methods section
            results: Results section
            conclusions: Conclusions section

        Returns:
            Formatted abstract string
        """
        sections = [
            ("Background", background),
            ("Methods", methods),
            ("Results", results),
            ("Conclusions", conclusions)
        ]

        formatted_parts = []
        for header, content in sections:
            if self.style.use_subheadings:
                formatted_parts.append(f"**{header}:** {content}")
            else:
                formatted_parts.append(content)

        abstract = " ".join(formatted_parts)

        # Check word limit
        word_count = len(abstract.split())
        if word_count > self.style.abstract_word_limit:
            print(f"Warning: Abstract exceeds {self.style.name} word limit "
                  f"({word_count} > {self.style.abstract_word_limit})")

        return abstract

    def format_p_value(self, p: float) -> str:
        """Format p-value according to journal style."""
        threshold = self.style.p_value_threshold

        if p < 0.001:
            return "P<.001" if self.journal in ['nejm', 'jama'] else "p < 0.001"
        elif p < threshold:
            if self.journal in ['nejm', 'jama']:
                return f"P={p:.3f}".replace("0.", ".")
            return f"p = {p:.3f}"
        else:
            if self.journal in ['nejm', 'jama']:
                return f"P={p:.2f}".replace("0.", ".")
            return f"p = {p:.2f}"

    def format_ci(
        self,
        lower: float,
        upper: float,
        level: int = 95
    ) -> str:
        """Format confidence interval according to journal style."""
        decimals = self.style.effect_decimals

        if self.journal in ['nejm', 'jama']:
            return f"{level}% CI, {lower:.{decimals}f} to {upper:.{decimals}f}"
        elif self.journal == 'lancet':
            return f"{level}% CI {lower:.{decimals}f}-{upper:.{decimals}f}"
        else:
            return f"{level}% CI [{lower:.{decimals}f}, {upper:.{decimals}f}]"

    def format_effect(
        self,
        effect: float,
        effect_type: str,
        ci_lower: Optional[float] = None,
        ci_upper: Optional[float] = None,
        p_value: Optional[float] = None
    ) -> str:
        """Format effect size with CI and p-value."""
        decimals = self.style.effect_decimals
        parts = []

        # Effect type abbreviation
        type_abbrev = {
            'odds_ratio': 'OR',
            'hazard_ratio': 'HR',
            'risk_ratio': 'RR',
            'mean_difference': 'MD',
            'standardized_mean_difference': 'SMD'
        }.get(effect_type, effect_type)

        parts.append(f"{type_abbrev}, {effect:.{decimals}f}")

        # CI
        if self.style.report_ci and ci_lower is not None and ci_upper is not None:
            parts.append(self.format_ci(ci_lower, ci_upper))

        # P-value
        if p_value is not None:
            parts.append(self.format_p_value(p_value))

        if self.journal in ['nejm', 'jama']:
            return "; ".join(parts)
        return ", ".join(parts)

    def _format_section(self, section: ReportSection) -> Dict[str, Any]:
        """Format a section according to journal style."""
        formatted = {
            'title': section.title,
            'content': section.content,
            'level': section.level
        }

        if section.subsections:
            formatted['subsections'] = [
                self._format_section(sub) for sub in section.subsections
            ]

        return formatted

    def _format_table(self, table: Dict[str, Any]) -> Dict[str, Any]:
        """Format table according to journal style."""
        formatted = {
            'id': table.get('id', ''),
            'title': table.get('title', ''),
            'style': self.style.table_style,
            'data': table.get('data', []),
            'columns': table.get('columns', []),
            'footnotes': table.get('footnotes', [])
        }

        # Add journal-specific formatting
        if self.style.table_style == 'three_line':
            formatted['border_top'] = True
            formatted['border_bottom'] = True
            formatted['border_header'] = True
            formatted['column_lines'] = False

        return formatted

    def _format_references(self, references: List[str]) -> List[str]:
        """Format references according to journal style."""
        # This would implement journal-specific reference formatting
        # For now, return as-is with numbered format
        return [f"{i+1}. {ref}" for i, ref in enumerate(references)]

    def get_guidelines(self) -> Dict[str, Any]:
        """Get formatting guidelines for the journal."""
        return {
            'journal': self.style.name,
            'abstract_word_limit': self.style.abstract_word_limit,
            'max_references': self.style.max_references,
            'table_style': self.style.table_style,
            'reference_style': self.style.reference_style,
            'tips': [
                f"Abstract should not exceed {self.style.abstract_word_limit} words",
                f"Maximum {self.style.max_references} references",
                f"Use {self.style.reference_style} reference style",
                "Report 95% confidence intervals for all effect estimates" if self.style.report_ci else "",
                "Include effect sizes" if self.style.report_effect_sizes else ""
            ]
        }
