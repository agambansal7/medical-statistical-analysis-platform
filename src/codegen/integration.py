"""Integration Module for Code Generation.

Provides easy integration with the analysis pipeline to automatically
generate reproducible code after each analysis.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from .code_generator import CodeGenerator, AnalysisCodeBundle, CodeFormat


class CodeGenerationIntegrator:
    """Integrates code generation with analysis results."""

    def __init__(
        self,
        output_dir: str = "./generated_code",
        seed: int = 42,
        languages: List[str] = ['python', 'r'],
        generate_notebooks: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.languages = languages
        self.generate_notebooks = generate_notebooks
        self.generator = CodeGenerator(default_seed=seed)

    def generate_from_analysis(
        self,
        analysis_type: str,
        parameters: Dict[str, Any],
        results: Dict[str, Any],
        data_info: Optional[Dict[str, Any]] = None,
        save: bool = True
    ) -> AnalysisCodeBundle:
        """Generate code from completed analysis.

        Args:
            analysis_type: Type of analysis performed
            parameters: Parameters used in analysis
            results: Analysis results
            data_info: Information about the data
            save: Whether to save files to disk

        Returns:
            AnalysisCodeBundle with generated code
        """
        formats = [CodeFormat.PYTHON_SCRIPT, CodeFormat.R_SCRIPT]
        if self.generate_notebooks:
            formats.extend([CodeFormat.JUPYTER_NOTEBOOK, CodeFormat.R_MARKDOWN])

        bundle = self.generator.generate(
            analysis_type=analysis_type,
            parameters=parameters,
            data_info=data_info,
            languages=self.languages,
            formats=formats,
            seed=self.seed
        )

        if save:
            self._save_bundle(bundle, analysis_type)

        return bundle

    def _save_bundle(self, bundle: AnalysisCodeBundle, analysis_type: str):
        """Save all generated code files."""
        base_name = analysis_type.replace(' ', '_').lower()
        analysis_dir = self.output_dir / base_name
        analysis_dir.mkdir(exist_ok=True)

        # Python script
        if bundle.python_code:
            py_path = analysis_dir / f"{base_name}.py"
            bundle.python_code.save(str(py_path))

        # R script
        if bundle.r_code:
            r_path = analysis_dir / f"{base_name}.R"
            bundle.r_code.save(str(r_path))

        # Jupyter notebook
        if bundle.jupyter_notebook:
            nb_path = analysis_dir / f"{base_name}.ipynb"
            with open(nb_path, 'w') as f:
                f.write(bundle.jupyter_notebook)

        # R Markdown
        if bundle.r_markdown:
            rmd_path = analysis_dir / f"{base_name}.Rmd"
            with open(rmd_path, 'w') as f:
                f.write(bundle.r_markdown)

        # README
        readme = self._generate_readme(bundle)
        readme_path = analysis_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme)

    def _generate_readme(self, bundle: AnalysisCodeBundle) -> str:
        """Generate README for the code bundle."""
        lines = [
            f"# {bundle.analysis_name}",
            "",
            "## Description",
            bundle.methodology_notes or "Statistical analysis code.",
            "",
            "## Files",
        ]

        if bundle.python_code:
            lines.append(f"- `{bundle.analysis_type}.py` - Python script")
        if bundle.r_code:
            lines.append(f"- `{bundle.analysis_type}.R` - R script")
        if bundle.jupyter_notebook:
            lines.append(f"- `{bundle.analysis_type}.ipynb` - Jupyter notebook")
        if bundle.r_markdown:
            lines.append(f"- `{bundle.analysis_type}.Rmd` - R Markdown document")

        lines.extend([
            "",
            "## Requirements",
            "",
            "### Python",
            "```bash",
        ])

        if bundle.python_code:
            lines.append(f"pip install {' '.join(bundle.python_code.packages_required)}")
        lines.append("```")

        lines.extend([
            "",
            "### R",
            "```r",
        ])
        if bundle.r_code:
            for pkg in bundle.r_code.packages_required:
                lines.append(f'install.packages("{pkg}")')
        lines.append("```")

        lines.extend([
            "",
            "## Reproducibility",
            f"- Random seed: {bundle.python_code.seed if bundle.python_code else bundle.r_code.seed}",
            f"- Code hash (Python): {bundle.python_code.code_hash if bundle.python_code else 'N/A'}",
            f"- Code hash (R): {bundle.r_code.code_hash if bundle.r_code else 'N/A'}",
        ])

        return "\n".join(lines)


def attach_code_to_results(
    results: Dict[str, Any],
    analysis_type: str,
    parameters: Dict[str, Any],
    data_info: Optional[Dict[str, Any]] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """Attach generated code to analysis results.

    This function adds reproducible code to existing analysis results,
    making it easy to integrate with the current analysis pipeline.

    Args:
        results: Original analysis results
        analysis_type: Type of analysis
        parameters: Analysis parameters
        data_info: Data information
        seed: Random seed

    Returns:
        Results dict with 'reproducible_code' key added
    """
    generator = CodeGenerator(default_seed=seed)
    bundle = generator.generate(
        analysis_type=analysis_type,
        parameters=parameters,
        data_info=data_info,
        languages=['python', 'r'],
        seed=seed
    )

    results['reproducible_code'] = {
        'python': bundle.python_code.code if bundle.python_code else None,
        'r': bundle.r_code.code if bundle.r_code else None,
        'python_packages': bundle.python_code.packages_required if bundle.python_code else [],
        'r_packages': bundle.r_code.packages_required if bundle.r_code else [],
        'seed': seed,
        'methodology': bundle.methodology_notes
    }

    return results
