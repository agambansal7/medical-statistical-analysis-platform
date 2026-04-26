"""Report Generator Module.

Generates publication-ready manuscript sections:
- Methods section with statistical analysis details
- Results section with formatted statistics
- Tables with proper formatting
- Figure captions
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ReportSection:
    """A section of the report."""
    title: str
    content: str
    level: int = 1  # Heading level
    subsections: List['ReportSection'] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    figures: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FullReport:
    """Complete generated report."""
    title: str
    sections: Dict[str, ReportSection]
    tables: List[Dict[str, Any]]
    figures: List[Dict[str, Any]]
    references: List[str]
    generated_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReportGenerator:
    """Generate publication-ready reports."""

    def __init__(
        self,
        style: str = "apa",
        journal: Optional[str] = None,
        language: str = "en"
    ):
        """Initialize report generator.

        Args:
            style: Citation style (apa, vancouver)
            journal: Target journal for formatting
            language: Output language
        """
        self.style = style
        self.journal = journal
        self.language = language

    def generate_methods_section(
        self,
        study_design: str,
        sample_size: int,
        primary_analysis: Dict[str, Any],
        secondary_analyses: List[Dict[str, Any]],
        confounders: List[str],
        alpha: float = 0.05,
        software: str = "Python Statistical Analysis Platform"
    ) -> ReportSection:
        """Generate statistical methods section.

        Args:
            study_design: Type of study design
            sample_size: Total sample size
            primary_analysis: Primary analysis details
            secondary_analyses: List of secondary analyses
            confounders: Variables adjusted for
            alpha: Significance level
            software: Statistical software used

        Returns:
            ReportSection with methods content
        """
        paragraphs = []

        # Sample and design
        paragraphs.append(
            f"This {study_design.lower()} included {sample_size:,} participants. "
        )

        # Primary analysis
        primary_text = self._format_analysis_method(primary_analysis)
        paragraphs.append(primary_text)

        # Confounders
        if confounders:
            confounder_text = self._format_confounder_list(confounders)
            paragraphs.append(
                f"All multivariable models were adjusted for {confounder_text}."
            )

        # Secondary analyses
        if secondary_analyses:
            secondary_text = self._format_secondary_analyses(secondary_analyses)
            paragraphs.append(secondary_text)

        # Statistical significance
        paragraphs.append(
            f"Two-sided p-values < {alpha} were considered statistically significant. "
        )

        # Software
        paragraphs.append(
            f"All analyses were performed using {software}."
        )

        content = " ".join(paragraphs)

        return ReportSection(
            title="Statistical Analysis",
            content=content,
            level=2
        )

    def generate_results_section(
        self,
        baseline_table: Optional[Dict[str, Any]] = None,
        primary_results: Optional[Dict[str, Any]] = None,
        secondary_results: Optional[List[Dict[str, Any]]] = None,
        figures: Optional[List[Dict[str, Any]]] = None
    ) -> ReportSection:
        """Generate results section.

        Args:
            baseline_table: Baseline characteristics table
            primary_results: Primary analysis results
            secondary_results: Secondary analysis results
            figures: Figure references

        Returns:
            ReportSection with results content
        """
        subsections = []
        tables = []
        figs = []

        # Baseline characteristics
        if baseline_table:
            baseline_section = self._generate_baseline_text(baseline_table)
            subsections.append(baseline_section)
            tables.append({
                'id': 'table1',
                'title': 'Baseline Characteristics',
                'data': baseline_table
            })

        # Primary results
        if primary_results:
            primary_section = self._generate_primary_results(primary_results)
            subsections.append(primary_section)

        # Secondary results
        if secondary_results:
            for i, result in enumerate(secondary_results):
                secondary_section = self._generate_secondary_result(result, i + 1)
                subsections.append(secondary_section)

        # Compile main content
        content = "\n\n".join(s.content for s in subsections)

        return ReportSection(
            title="Results",
            content=content,
            level=2,
            subsections=subsections,
            tables=tables,
            figures=figs
        )

    def format_statistic(
        self,
        value: float,
        stat_type: str,
        ci: Optional[Tuple[float, float]] = None,
        p_value: Optional[float] = None
    ) -> str:
        """Format a statistic for publication.

        Args:
            value: The statistic value
            stat_type: Type of statistic (OR, HR, RR, mean_diff, etc.)
            ci: Confidence interval
            p_value: P-value

        Returns:
            Formatted string
        """
        # Format the main value
        if stat_type in ['OR', 'HR', 'RR']:
            value_str = f"{stat_type} {value:.2f}"
        elif stat_type == 'mean_diff':
            value_str = f"mean difference {value:.2f}"
        elif stat_type == 'correlation':
            value_str = f"r = {value:.2f}"
        else:
            value_str = f"{value:.2f}"

        # Add CI
        if ci:
            ci_str = f"95% CI [{ci[0]:.2f}, {ci[1]:.2f}]"
            value_str = f"{value_str} ({ci_str})"

        # Add p-value
        if p_value is not None:
            if p_value < 0.001:
                p_str = "p < 0.001"
            else:
                p_str = f"p = {p_value:.3f}"
            value_str = f"{value_str}, {p_str}"

        return value_str

    def format_table(
        self,
        data: pd.DataFrame,
        title: str,
        footnotes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Format a table for publication.

        Args:
            data: Table data
            title: Table title
            footnotes: Table footnotes

        Returns:
            Formatted table dictionary
        """
        return {
            'title': title,
            'data': data.to_dict('records'),
            'columns': list(data.columns),
            'footnotes': footnotes or [],
            'format': self.style
        }

    def generate_full_report(
        self,
        title: str,
        methods: ReportSection,
        results: ReportSection,
        additional_sections: Optional[List[ReportSection]] = None,
        references: Optional[List[str]] = None
    ) -> FullReport:
        """Generate complete report.

        Args:
            title: Report title
            methods: Methods section
            results: Results section
            additional_sections: Additional sections
            references: Reference list

        Returns:
            Complete FullReport
        """
        sections = {
            'methods': methods,
            'results': results
        }

        if additional_sections:
            for i, section in enumerate(additional_sections):
                sections[f'additional_{i}'] = section

        # Collect all tables and figures
        all_tables = methods.tables + results.tables
        all_figures = methods.figures + results.figures

        return FullReport(
            title=title,
            sections=sections,
            tables=all_tables,
            figures=all_figures,
            references=references or [],
            generated_at=datetime.now().isoformat()
        )

    def _format_analysis_method(self, analysis: Dict[str, Any]) -> str:
        """Format analysis method description."""
        method = analysis.get('method', '')
        outcome_type = analysis.get('outcome_type', '')

        descriptions = {
            'logistic_regression': (
                "Multivariable logistic regression was used to estimate adjusted "
                "odds ratios (aOR) with 95% confidence intervals (CI)."
            ),
            'linear_regression': (
                "Multivariable linear regression was used to estimate adjusted "
                "mean differences with 95% confidence intervals."
            ),
            'cox_regression': (
                "Cox proportional hazards regression was used to estimate adjusted "
                "hazard ratios (aHR) with 95% confidence intervals. "
                "The proportional hazards assumption was assessed using Schoenfeld residuals."
            ),
            'kaplan_meier': (
                "Survival curves were estimated using the Kaplan-Meier method "
                "and compared using the log-rank test."
            ),
            'ttest': (
                "Continuous outcomes were compared between groups using "
                "independent samples t-tests."
            ),
            'chi_square': (
                "Categorical variables were compared using chi-square tests "
                "or Fisher's exact test when expected cell counts were less than 5."
            ),
            'mann_whitney': (
                "Non-normally distributed continuous variables were compared "
                "using Mann-Whitney U tests."
            )
        }

        return descriptions.get(method, f"Analysis was performed using {method}.")

    def _format_confounder_list(self, confounders: List[str]) -> str:
        """Format list of confounders for text."""
        if len(confounders) == 1:
            return confounders[0]
        elif len(confounders) == 2:
            return f"{confounders[0]} and {confounders[1]}"
        else:
            return ", ".join(confounders[:-1]) + f", and {confounders[-1]}"

    def _format_secondary_analyses(self, analyses: List[Dict[str, Any]]) -> str:
        """Format secondary analyses description."""
        descriptions = []

        for analysis in analyses:
            method = analysis.get('method', '')
            if method == 'subgroup':
                descriptions.append(
                    f"Subgroup analyses were performed by {analysis.get('subgroup_var', 'key variables')}"
                )
            elif method == 'sensitivity':
                descriptions.append(
                    "Sensitivity analyses were conducted to assess robustness of findings"
                )

        if descriptions:
            return "Secondary analyses included: " + "; ".join(descriptions) + "."
        return ""

    def _generate_baseline_text(self, table: Dict[str, Any]) -> ReportSection:
        """Generate text describing baseline characteristics."""
        content = (
            "Baseline characteristics of the study population are presented in Table 1. "
        )

        # Add summary statistics if available
        if 'summary' in table:
            summary = table['summary']
            if 'n_per_group' in summary:
                groups = summary['n_per_group']
                group_text = ", ".join(
                    f"{name} (n={n})" for name, n in groups.items()
                )
                content += f"The analysis included {group_text}. "

        return ReportSection(
            title="Baseline Characteristics",
            content=content,
            level=3
        )

    def _generate_primary_results(self, results: Dict[str, Any]) -> ReportSection:
        """Generate primary results text."""
        content_parts = []

        # Extract key results
        if 'coefficients' in results:
            coeffs = results['coefficients']
            for var, stats in coeffs.items():
                if var.lower() not in ['intercept', 'const']:
                    # Format based on analysis type
                    if 'odds_ratio' in stats:
                        stat_str = self.format_statistic(
                            stats['odds_ratio'],
                            'OR',
                            ci=(stats.get('or_ci_lower'), stats.get('or_ci_upper')),
                            p_value=stats.get('p_value')
                        )
                        content_parts.append(
                            f"{var} was associated with the outcome ({stat_str})"
                        )
                    elif 'hazard_ratio' in stats:
                        stat_str = self.format_statistic(
                            stats['hazard_ratio'],
                            'HR',
                            ci=(stats.get('hr_ci_lower'), stats.get('hr_ci_upper')),
                            p_value=stats.get('p_value')
                        )
                        content_parts.append(
                            f"{var} was associated with the outcome ({stat_str})"
                        )
                    elif 'coefficient' in stats:
                        stat_str = self.format_statistic(
                            stats['coefficient'],
                            'mean_diff',
                            ci=(stats.get('ci_lower'), stats.get('ci_upper')),
                            p_value=stats.get('p_value')
                        )
                        content_parts.append(
                            f"{var}: {stat_str}"
                        )

        content = ". ".join(content_parts) + "." if content_parts else ""

        return ReportSection(
            title="Primary Analysis",
            content=content,
            level=3
        )

    def _generate_secondary_result(
        self,
        result: Dict[str, Any],
        index: int
    ) -> ReportSection:
        """Generate secondary result text."""
        result_type = result.get('type', f'Secondary Analysis {index}')
        content = f"Results of {result_type}: "

        if 'summary' in result:
            content += result['summary']

        return ReportSection(
            title=f"Secondary Analysis {index}",
            content=content,
            level=3
        )


class StatisticalFormatter:
    """Format statistics according to publication standards."""

    @staticmethod
    def format_p_value(p: float) -> str:
        """Format p-value for publication."""
        if p < 0.001:
            return "p < 0.001"
        elif p < 0.01:
            return f"p = {p:.3f}"
        else:
            return f"p = {p:.2f}"

    @staticmethod
    def format_ci(
        lower: float,
        upper: float,
        level: int = 95,
        decimals: int = 2
    ) -> str:
        """Format confidence interval."""
        return f"{level}% CI [{lower:.{decimals}f}, {upper:.{decimals}f}]"

    @staticmethod
    def format_mean_sd(mean: float, sd: float, decimals: int = 1) -> str:
        """Format mean ± SD."""
        return f"{mean:.{decimals}f} ± {sd:.{decimals}f}"

    @staticmethod
    def format_median_iqr(
        median: float,
        q1: float,
        q3: float,
        decimals: int = 1
    ) -> str:
        """Format median (IQR)."""
        return f"{median:.{decimals}f} ({q1:.{decimals}f}-{q3:.{decimals}f})"

    @staticmethod
    def format_n_percent(n: int, total: int, decimals: int = 1) -> str:
        """Format n (%)."""
        pct = n / total * 100 if total > 0 else 0
        return f"{n} ({pct:.{decimals}f}%)"

    @staticmethod
    def format_effect_size(
        effect: float,
        effect_type: str,
        ci_lower: Optional[float] = None,
        ci_upper: Optional[float] = None,
        p_value: Optional[float] = None
    ) -> str:
        """Format effect size with CI and p-value."""
        parts = [f"{effect_type} {effect:.2f}"]

        if ci_lower is not None and ci_upper is not None:
            parts.append(f"(95% CI {ci_lower:.2f}-{ci_upper:.2f})")

        if p_value is not None:
            parts.append(StatisticalFormatter.format_p_value(p_value))

        return ", ".join(parts)
