"""
Statistical Report Generator
============================

Generates comprehensive, publication-ready reports from execution results.
Supports multiple output formats:
- Markdown (for display)
- HTML (for web)
- Word document (for publication)
- LaTeX (for journals)
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import anthropic

from .comprehensive_planner import ComprehensiveStatisticalPlan
from .plan_executor import ExecutionReport, AnalysisResult


@dataclass
class GeneratedReport:
    """Container for generated report."""
    title: str
    generated_at: str
    format: str
    content: str
    sections: Dict[str, str]
    figures: List[Dict[str, str]]  # List of {name, base64_image}
    tables: List[Dict[str, Any]]


class ReportGenerator:
    """
    Generates comprehensive statistical reports.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.client = None
        self.model = "claude-sonnet-4-20250514"
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
        self.use_llm = self.client is not None

    def generate_report(
        self,
        plan: ComprehensiveStatisticalPlan,
        execution_report: ExecutionReport,
        output_format: str = 'markdown',
        include_methods: bool = True,
        include_raw_results: bool = False
    ) -> GeneratedReport:
        """
        Generate a comprehensive statistical report.

        Parameters:
        -----------
        plan : ComprehensiveStatisticalPlan
            The analysis plan
        execution_report : ExecutionReport
            Results from plan execution
        output_format : str
            'markdown', 'html', 'latex', or 'word'
        include_methods : bool
            Whether to include detailed methods section
        include_raw_results : bool
            Whether to include raw result data

        Returns:
        --------
        GeneratedReport
        """
        # Generate each section
        sections = {}

        # Title and Abstract
        sections['title'] = self._generate_title(plan)
        sections['abstract'] = self._generate_abstract(plan, execution_report)

        # Methods
        if include_methods:
            sections['methods'] = self._generate_methods_section(plan)

        # Results
        sections['results'] = self._generate_results_section(plan, execution_report)

        # Discussion
        sections['discussion'] = self._generate_discussion(plan, execution_report)

        # Tables
        tables = self._extract_tables(execution_report)

        # Figures
        figures = self._extract_figures(execution_report)

        # Compile full report
        if output_format == 'markdown':
            content = self._compile_markdown(sections, tables, figures)
        elif output_format == 'html':
            content = self._compile_html(sections, tables, figures)
        elif output_format == 'latex':
            content = self._compile_latex(sections, tables, figures)
        else:
            content = self._compile_markdown(sections, tables, figures)

        return GeneratedReport(
            title=sections['title'],
            generated_at=datetime.now().isoformat(),
            format=output_format,
            content=content,
            sections=sections,
            figures=figures,
            tables=tables
        )

    def _generate_title(self, plan: ComprehensiveStatisticalPlan) -> str:
        """Generate appropriate title."""
        if not self.use_llm:
            # Fallback title without LLM
            return f"Statistical Analysis: {plan.research_type.title()} Study (n={plan.sample_size})"

        # Use LLM to generate a concise title
        prompt = f"""Generate a concise, professional title for a statistical analysis report.

Research Question: {plan.research_question}
Research Type: {plan.research_type}
Study Design: {plan.study_design}
Sample Size: {plan.sample_size}

Respond with ONLY the title, nothing else."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text.strip()

    def _generate_abstract(
        self,
        plan: ComprehensiveStatisticalPlan,
        execution_report: ExecutionReport
    ) -> str:
        """Generate structured abstract."""
        # Collect key results
        key_results = []
        for result in execution_report.analysis_results:
            if result.success and result.step_id.startswith('primary'):
                p_val = result.results.get('p_value')
                if p_val is not None:
                    sig = "significant" if p_val < plan.significance_level else "not significant"
                    key_results.append(f"{result.step_name}: p={p_val:.4f} ({sig})")

        if not self.use_llm:
            # Fallback abstract without LLM
            abstract_parts = [
                "## Abstract\n",
                f"**Background:** {plan.research_question}\n",
                f"**Methods:** This {plan.study_design} study included {plan.sample_size} participants. ",
                f"Statistical analyses included {', '.join([a.name for a in plan.primary_analyses[:3]])}.\n",
                f"**Results:** {chr(10).join(key_results) if key_results else 'See detailed results section.'}\n",
                "**Conclusion:** Further details are provided in the full report."
            ]
            return ''.join(abstract_parts)

        prompt = f"""Generate a structured abstract (Background, Methods, Results, Conclusion) for this statistical analysis.

RESEARCH QUESTION:
{plan.research_question}

STUDY DESIGN: {plan.study_design}
SAMPLE SIZE: {plan.sample_size}

KEY RESULTS:
{chr(10).join(key_results) if key_results else 'See detailed results section'}

PRIMARY ANALYSES PERFORMED:
{chr(10).join([a.name for a in plan.primary_analyses])}

Write a professional, concise abstract (250-300 words) with clear sections."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def _generate_methods_section(self, plan: ComprehensiveStatisticalPlan) -> str:
        """Generate detailed methods section."""
        methods_parts = []

        methods_parts.append("## Statistical Methods\n")

        # Study Design
        methods_parts.append(f"### Study Design\n")
        methods_parts.append(f"This {plan.study_design} study included {plan.sample_size} participants.\n")

        # Variables
        methods_parts.append(f"### Variables\n")
        methods_parts.append(f"**Primary Outcome(s):** {', '.join(plan.outcome_variables)}\n")
        methods_parts.append(f"**Primary Exposure(s):** {', '.join(plan.exposure_variables)}\n")
        if plan.covariates:
            methods_parts.append(f"**Covariates:** {', '.join(plan.covariates)}\n")

        # Descriptive Statistics
        if plan.descriptive_analyses:
            methods_parts.append("### Descriptive Statistics\n")
            methods_parts.append("Continuous variables were summarized as mean ± standard deviation or median (interquartile range) as appropriate. ")
            methods_parts.append("Categorical variables were reported as frequencies and percentages.\n")

        # Primary Analyses
        if plan.primary_analyses:
            methods_parts.append("### Primary Analyses\n")
            for analysis in plan.primary_analyses:
                methods_parts.append(f"**{analysis.name}:** {analysis.method} was used to {analysis.description.lower()}. ")
                if analysis.assumptions:
                    methods_parts.append(f"Assumptions of {', '.join(analysis.assumptions)} were assessed. ")
                if analysis.fallback_method:
                    methods_parts.append(f"If assumptions were violated, {analysis.fallback_method} was used as an alternative. ")
                methods_parts.append("\n")

        # Secondary Analyses
        if plan.secondary_analyses:
            methods_parts.append("### Secondary Analyses\n")
            for analysis in plan.secondary_analyses:
                methods_parts.append(f"- {analysis.name}: {analysis.method}\n")

        # Sensitivity Analyses
        if plan.sensitivity_analyses:
            methods_parts.append("### Sensitivity Analyses\n")
            for analysis in plan.sensitivity_analyses:
                methods_parts.append(f"- {analysis.name}: {analysis.description}\n")

        # Subgroup Analyses
        if plan.subgroup_analyses:
            methods_parts.append("### Subgroup Analyses\n")
            methods_parts.append("Pre-specified subgroup analyses were performed to assess treatment effect heterogeneity. ")
            methods_parts.append("Interaction tests were conducted to evaluate statistical significance of subgroup differences.\n")

        # Statistical Considerations
        methods_parts.append("### Statistical Considerations\n")
        methods_parts.append(f"**Missing Data:** {plan.missing_data_strategy}\n")
        methods_parts.append(f"**Multiple Testing:** {plan.multiple_testing_correction}\n")
        methods_parts.append(f"**Significance Level:** A two-sided p-value < {plan.significance_level} was considered statistically significant.\n")

        methods_parts.append("\nAll analyses were performed using the Medical Statistical Analysis Platform (Python-based).\n")

        return "".join(methods_parts)

    def _generate_results_section(
        self,
        plan: ComprehensiveStatisticalPlan,
        execution_report: ExecutionReport
    ) -> str:
        """Generate results section with LLM interpretation."""
        results_parts = []
        results_parts.append("## Results\n")

        # Group results by category
        descriptive_results = []
        primary_results = []
        secondary_results = []
        sensitivity_results = []
        subgroup_results = []

        for result in execution_report.analysis_results:
            if not result.success:
                continue

            if 'desc' in result.step_id:
                descriptive_results.append(result)
            elif 'primary' in result.step_id:
                primary_results.append(result)
            elif 'secondary' in result.step_id:
                secondary_results.append(result)
            elif 'sensitivity' in result.step_id:
                sensitivity_results.append(result)
            elif 'subgroup' in result.step_id:
                subgroup_results.append(result)

        # Descriptive Results
        if descriptive_results:
            results_parts.append("### Baseline Characteristics\n")
            for result in descriptive_results:
                if 'table' in result.results:
                    results_parts.append("See Table 1 for baseline characteristics.\n")
                else:
                    results_parts.append(f"{result.interpretation}\n")

        # Primary Results
        if primary_results:
            results_parts.append("### Primary Outcomes\n")

            for result in primary_results:
                results_parts.append(f"#### {result.step_name}\n")

                # Report key statistics
                r = result.results
                stats_line = []

                if 'mean_difference' in r:
                    stats_line.append(f"mean difference = {r['mean_difference']:.3f}")
                if 'odds_ratio' in r:
                    stats_line.append(f"OR = {r['odds_ratio']:.3f}")
                if 'hazard_ratio' in r:
                    stats_line.append(f"HR = {r['hazard_ratio']:.3f}")
                if 'ci_lower' in r and 'ci_upper' in r:
                    stats_line.append(f"95% CI [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]")
                if 'p_value' in r:
                    p = r['p_value']
                    if p < 0.001:
                        stats_line.append("p < 0.001")
                    else:
                        stats_line.append(f"p = {p:.4f}")

                if stats_line:
                    results_parts.append(f"**Results:** {', '.join(stats_line)}\n")

                # Add interpretation
                if result.interpretation:
                    results_parts.append(f"\n{result.interpretation}\n")

                # Note if fallback was used
                if result.used_fallback:
                    results_parts.append(f"\n*Note: Due to assumption violations, {result.method_used} was used instead of the planned method.*\n")

                results_parts.append("\n")

        # Secondary Results
        if secondary_results:
            results_parts.append("### Secondary Outcomes\n")
            for result in secondary_results:
                p_val = result.results.get('p_value')
                if p_val is not None:
                    sig = "significant" if p_val < plan.significance_level else "not significant"
                    results_parts.append(f"- **{result.step_name}:** p = {p_val:.4f} ({sig})\n")

        # Sensitivity Results
        if sensitivity_results:
            results_parts.append("### Sensitivity Analyses\n")
            results_parts.append("Sensitivity analyses were performed to assess the robustness of the primary findings.\n")
            for result in sensitivity_results:
                results_parts.append(f"- {result.step_name}: {result.interpretation}\n")

        # Subgroup Results
        if subgroup_results:
            results_parts.append("### Subgroup Analyses\n")
            for result in subgroup_results:
                if 'interaction_p' in result.results:
                    for var, p in result.results['interaction_p'].items():
                        sig = "significant" if p < plan.significance_level else "not significant"
                        results_parts.append(f"- Interaction with {var}: p = {p:.4f} ({sig})\n")

        return "".join(results_parts)

    def _generate_discussion(
        self,
        plan: ComprehensiveStatisticalPlan,
        execution_report: ExecutionReport
    ) -> str:
        """Generate discussion section using LLM."""
        # Collect key findings
        key_findings = []
        for result in execution_report.analysis_results:
            if result.success and 'primary' in result.step_id:
                p_val = result.results.get('p_value')
                key_findings.append({
                    'name': result.step_name,
                    'p_value': p_val,
                    'interpretation': result.interpretation
                })

        if not self.use_llm:
            # Fallback discussion without LLM
            discussion = "## Discussion\n\n"
            discussion += "This analysis examined the research question using multiple statistical approaches. "
            if key_findings:
                discussion += "Key findings from the primary analyses are summarized below.\n\n"
                for finding in key_findings:
                    p = finding.get('p_value')
                    p_str = f"p = {p:.4f}" if p is not None else "p-value not available"
                    discussion += f"- **{finding['name']}**: {p_str}\n"
            if plan.limitations:
                discussion += "\n\n### Limitations\n"
                for lim in plan.limitations:
                    discussion += f"- {lim}\n"
            return discussion

        prompt = f"""Write a brief discussion section for a statistical analysis report.

RESEARCH QUESTION:
{plan.research_question}

KEY FINDINGS:
{json.dumps(key_findings, indent=2)}

LIMITATIONS:
{chr(10).join(['- ' + lim for lim in plan.limitations]) if plan.limitations else 'None specified'}

Write a professional discussion (2-3 paragraphs) that:
1. Summarizes the main findings
2. Interprets the results in context
3. Acknowledges limitations
4. Suggests implications or future directions

Do NOT include section headers - just the prose."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )

        discussion = "## Discussion\n\n"
        discussion += response.content[0].text

        if plan.limitations:
            discussion += "\n\n### Limitations\n"
            for lim in plan.limitations:
                discussion += f"- {lim}\n"

        return discussion

    def _extract_tables(self, execution_report: ExecutionReport) -> List[Dict[str, Any]]:
        """Extract tables from results."""
        tables = []

        for result in execution_report.analysis_results:
            if not result.success:
                continue

            if 'table' in result.results:
                tables.append({
                    'name': result.step_name,
                    'data': result.results['table']
                })

        return tables

    def _extract_figures(self, execution_report: ExecutionReport) -> List[Dict[str, str]]:
        """Extract figures from visualization results."""
        figures = []

        for viz in execution_report.visualization_results:
            if viz.success and viz.image_base64:
                figures.append({
                    'name': viz.name,
                    'image_base64': viz.image_base64,
                    'format': viz.format
                })

        return figures

    def _compile_markdown(
        self,
        sections: Dict[str, str],
        tables: List[Dict],
        figures: List[Dict]
    ) -> str:
        """Compile full markdown report."""
        md = []

        # Title
        md.append(f"# {sections['title']}")
        md.append("")
        md.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
        md.append("")

        # Abstract
        if 'abstract' in sections:
            md.append("## Abstract")
            md.append("")
            md.append(sections['abstract'])
            md.append("")

        # Methods
        if 'methods' in sections:
            md.append(sections['methods'])
            md.append("")

        # Results
        if 'results' in sections:
            md.append(sections['results'])
            md.append("")

        # Tables
        if tables:
            md.append("## Tables")
            md.append("")
            for i, table in enumerate(tables, 1):
                md.append(f"### Table {i}: {table['name']}")
                md.append("")
                # Convert table data to markdown
                if isinstance(table['data'], list) and table['data']:
                    headers = list(table['data'][0].keys())
                    md.append("| " + " | ".join(headers) + " |")
                    md.append("| " + " | ".join(["---"] * len(headers)) + " |")
                    for row in table['data']:
                        md.append("| " + " | ".join(str(row.get(h, '')) for h in headers) + " |")
                md.append("")

        # Figures
        if figures:
            md.append("## Figures")
            md.append("")
            for i, fig in enumerate(figures, 1):
                md.append(f"### Figure {i}: {fig['name']}")
                md.append("")
                md.append(f"![{fig['name']}](data:image/{fig['format']};base64,{fig['image_base64']})")
                md.append("")

        # Discussion
        if 'discussion' in sections:
            md.append(sections['discussion'])
            md.append("")

        return "\n".join(md)

    def _compile_html(
        self,
        sections: Dict[str, str],
        tables: List[Dict],
        figures: List[Dict]
    ) -> str:
        """Compile HTML report."""
        # Convert markdown to HTML (simplified)
        md_content = self._compile_markdown(sections, tables, figures)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{sections['title']}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        img {{ max-width: 100%; height: auto; }}
        .abstract {{ background-color: #ecf0f1; padding: 20px; border-radius: 5px; }}
        .methods {{ background-color: #fafafa; padding: 15px; border-left: 4px solid #3498db; }}
    </style>
</head>
<body>
    <h1>{sections['title']}</h1>
    <p><em>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</em></p>
"""

        if 'abstract' in sections:
            html += f"""
    <div class="abstract">
        <h2>Abstract</h2>
        <p>{sections['abstract'].replace(chr(10), '</p><p>')}</p>
    </div>
"""

        if 'methods' in sections:
            # Simple markdown to HTML conversion
            methods_html = sections['methods']
            methods_html = methods_html.replace('## ', '<h2>').replace('\n### ', '</h2>\n<h3>')
            methods_html = methods_html.replace('\n\n', '</p>\n<p>')
            html += f"""
    <div class="methods">
        {methods_html}
    </div>
"""

        if 'results' in sections:
            results_html = sections['results']
            results_html = results_html.replace('## ', '<h2>').replace('### ', '<h3>')
            results_html = results_html.replace('\n\n', '</p>\n<p>')
            html += f"""
    <div class="results">
        {results_html}
    </div>
"""

        # Tables
        if tables:
            html += "<h2>Tables</h2>\n"
            for i, table in enumerate(tables, 1):
                html += f"<h3>Table {i}: {table['name']}</h3>\n"
                if isinstance(table['data'], list) and table['data']:
                    html += "<table>\n<thead><tr>"
                    headers = list(table['data'][0].keys())
                    for h in headers:
                        html += f"<th>{h}</th>"
                    html += "</tr></thead>\n<tbody>\n"
                    for row in table['data']:
                        html += "<tr>"
                        for h in headers:
                            html += f"<td>{row.get(h, '')}</td>"
                        html += "</tr>\n"
                    html += "</tbody></table>\n"

        # Figures
        if figures:
            html += "<h2>Figures</h2>\n"
            for i, fig in enumerate(figures, 1):
                html += f"<h3>Figure {i}: {fig['name']}</h3>\n"
                html += f"<img src='data:image/{fig['format']};base64,{fig['image_base64']}' alt='{fig['name']}'>\n"

        if 'discussion' in sections:
            discussion_html = sections['discussion'].replace('## ', '<h2>').replace('\n\n', '</p>\n<p>')
            html += f"""
    <div class="discussion">
        {discussion_html}
    </div>
"""

        html += """
</body>
</html>
"""
        return html

    def _compile_latex(
        self,
        sections: Dict[str, str],
        tables: List[Dict],
        figures: List[Dict]
    ) -> str:
        """Compile LaTeX report."""
        latex = r"""\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage[margin=1in]{geometry}

\title{""" + sections['title'] + r"""}
\author{Medical Statistical Analysis Platform}
\date{""" + datetime.now().strftime('%B %d, %Y') + r"""}

\begin{document}
\maketitle

"""

        if 'abstract' in sections:
            latex += r"\begin{abstract}" + "\n"
            latex += sections['abstract'] + "\n"
            latex += r"\end{abstract}" + "\n\n"

        if 'methods' in sections:
            # Convert markdown to latex
            methods = sections['methods']
            methods = methods.replace('## ', r'\section{').replace('\n### ', '}\n\\subsection{')
            methods = methods.replace('**', r'\textbf{').replace('**', '}')
            latex += methods + "}\n\n"

        if 'results' in sections:
            results = sections['results']
            results = results.replace('## ', r'\section{').replace('### ', r'\subsection{')
            results = results.replace('#### ', r'\subsubsection{')
            latex += results + "\n\n"

        if 'discussion' in sections:
            discussion = sections['discussion']
            discussion = discussion.replace('## ', r'\section{').replace('### ', r'\subsection{')
            latex += discussion + "\n\n"

        latex += r"\end{document}"

        return latex

    def save_report(
        self,
        report: GeneratedReport,
        output_path: str,
        include_figures_separately: bool = False
    ) -> List[str]:
        """
        Save report to file(s).

        Returns list of saved file paths.
        """
        saved_files = []
        output_path = Path(output_path)

        # Save main report
        if report.format == 'markdown':
            main_file = output_path.with_suffix('.md')
        elif report.format == 'html':
            main_file = output_path.with_suffix('.html')
        elif report.format == 'latex':
            main_file = output_path.with_suffix('.tex')
        else:
            main_file = output_path.with_suffix('.txt')

        main_file.write_text(report.content)
        saved_files.append(str(main_file))

        # Save figures separately if requested
        if include_figures_separately and report.figures:
            fig_dir = output_path.parent / f"{output_path.stem}_figures"
            fig_dir.mkdir(exist_ok=True)

            import base64
            for i, fig in enumerate(report.figures, 1):
                fig_file = fig_dir / f"figure_{i}_{fig['name'].replace(' ', '_')}.{fig['format']}"
                img_data = base64.b64decode(fig['image_base64'])
                fig_file.write_bytes(img_data)
                saved_files.append(str(fig_file))

        return saved_files
