"""Notebook Generator for Jupyter and R Markdown."""

from typing import Dict, Any, Optional
import json
from datetime import datetime


class NotebookGenerator:
    """Generate Jupyter notebooks and R Markdown documents."""

    def to_jupyter(self, code_obj, title: str = "", methodology: str = "") -> str:
        """Convert Python code to Jupyter notebook."""
        cells = []

        # Title cell
        cells.append(self._markdown_cell(f"# {title or code_obj.analysis_type.replace('_', ' ').title()}"))

        # Metadata cell
        meta_text = f"""## Analysis Information
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Analysis Type**: {code_obj.analysis_type}
- **Random Seed**: {code_obj.seed}
- **Code Hash**: {code_obj.code_hash}"""
        cells.append(self._markdown_cell(meta_text))

        # Methodology
        if methodology:
            cells.append(self._markdown_cell(f"## Methodology\n{methodology}"))

        # Required packages
        pkg_text = "## Required Packages\n```\n" + "\n".join(code_obj.packages_required) + "\n```"
        cells.append(self._markdown_cell(pkg_text))

        # Parse and add code sections
        code_sections = self._parse_code_sections(code_obj.code)
        for section_title, section_code in code_sections:
            if section_title:
                cells.append(self._markdown_cell(f"### {section_title}"))
            cells.append(self._code_cell(section_code))

        notebook = {
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {"name": "python", "version": "3.9.0"}
            },
            "nbformat": 4,
            "nbformat_minor": 5,
            "cells": cells
        }
        return json.dumps(notebook, indent=2)

    def to_rmarkdown(self, code_obj, title: str = "", methodology: str = "") -> str:
        """Convert R code to R Markdown document."""
        lines = []

        # YAML header
        lines.append("---")
        lines.append(f'title: "{title or code_obj.analysis_type.replace("_", " ").title()}"')
        lines.append(f'date: "{datetime.now().strftime("%Y-%m-%d")}"')
        lines.append("output:")
        lines.append("  html_document:")
        lines.append("    toc: true")
        lines.append("    toc_float: true")
        lines.append("    code_folding: show")
        lines.append("---")
        lines.append("")

        # Setup chunk
        lines.append("```{r setup, include=FALSE}")
        lines.append("knitr::opts_chunk$set(echo = TRUE, warning = FALSE, message = FALSE)")
        lines.append("```")
        lines.append("")

        # Analysis info
        lines.append("## Analysis Information")
        lines.append(f"- **Analysis Type**: {code_obj.analysis_type}")
        lines.append(f"- **Random Seed**: {code_obj.seed}")
        lines.append(f"- **Code Hash**: {code_obj.code_hash}")
        lines.append("")

        # Methodology
        if methodology:
            lines.append("## Methodology")
            lines.append(methodology)
            lines.append("")

        # Required packages
        lines.append("## Required Packages")
        lines.append("```{r packages}")
        for pkg in code_obj.packages_required:
            lines.append(f'library({pkg})')
        lines.append("```")
        lines.append("")

        # Code sections
        code_sections = self._parse_code_sections(code_obj.code)
        chunk_num = 1
        for section_title, section_code in code_sections:
            if section_title:
                lines.append(f"## {section_title}")
            lines.append(f"```{{r chunk{chunk_num}}}")
            lines.append(section_code.strip())
            lines.append("```")
            lines.append("")
            chunk_num += 1

        return "\n".join(lines)

    def _markdown_cell(self, content: str) -> Dict:
        return {"cell_type": "markdown", "metadata": {}, "source": [content]}

    def _code_cell(self, code: str) -> Dict:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [code]
        }

    def _parse_code_sections(self, code: str) -> list:
        """Parse code into sections based on comment headers."""
        sections = []
        current_section = ""
        current_code = []

        for line in code.split('\n'):
            if line.startswith('# ===') or line.startswith('# ---'):
                if current_code:
                    sections.append((current_section, '\n'.join(current_code)))
                    current_code = []
                current_section = ""
            elif line.startswith('# ') and line.endswith(' #'):
                # Section header
                current_section = line.strip('# ').strip()
            elif '===' in line:
                continue
            else:
                current_code.append(line)

        if current_code:
            sections.append((current_section, '\n'.join(current_code)))

        return [(s, c) for s, c in sections if c.strip()]
