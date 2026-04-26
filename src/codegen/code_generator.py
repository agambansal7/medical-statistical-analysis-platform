"""Main Code Generator Engine.

Orchestrates code generation for statistical analyses in Python and R.
Provides a unified interface for generating reproducible analysis code.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib
import json


class CodeFormat(Enum):
    """Output format for generated code."""
    PYTHON_SCRIPT = "python_script"
    R_SCRIPT = "r_script"
    JUPYTER_NOTEBOOK = "jupyter_notebook"
    R_MARKDOWN = "r_markdown"
    QUARTO = "quarto"


@dataclass
class GeneratedCode:
    """Container for generated code."""
    language: str  # 'python' or 'r'
    code: str
    format: CodeFormat
    analysis_type: str
    parameters: Dict[str, Any]
    packages_required: List[str]
    seed: int
    generated_at: str
    code_hash: str
    documentation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'language': self.language,
            'code': self.code,
            'format': self.format.value,
            'analysis_type': self.analysis_type,
            'parameters': self.parameters,
            'packages_required': self.packages_required,
            'seed': self.seed,
            'generated_at': self.generated_at,
            'code_hash': self.code_hash,
            'documentation': self.documentation
        }

    def save(self, filepath: str) -> None:
        """Save code to file."""
        with open(filepath, 'w') as f:
            f.write(self.code)


@dataclass
class AnalysisCodeBundle:
    """Bundle containing both Python and R code for an analysis."""
    analysis_type: str
    analysis_name: str
    python_code: Optional[GeneratedCode] = None
    r_code: Optional[GeneratedCode] = None
    jupyter_notebook: Optional[str] = None
    r_markdown: Optional[str] = None
    data_description: str = ""
    methodology_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'analysis_type': self.analysis_type,
            'analysis_name': self.analysis_name,
            'python_code': self.python_code.to_dict() if self.python_code else None,
            'r_code': self.r_code.to_dict() if self.r_code else None,
            'jupyter_notebook': self.jupyter_notebook,
            'r_markdown': self.r_markdown,
            'data_description': self.data_description,
            'methodology_notes': self.methodology_notes
        }


class CodeGenerator:
    """Main code generator orchestrator.

    Generates reproducible statistical analysis code in Python and R.
    Supports multiple output formats including notebooks and markdown.
    """

    # Mapping of analysis types to their categories
    ANALYSIS_CATEGORIES = {
        # Basic Tests
        'ttest': 'basic',
        'ttest_independent': 'basic',
        'ttest_paired': 'basic',
        'ttest_one_sample': 'basic',
        'anova': 'basic',
        'anova_oneway': 'basic',
        'anova_twoway': 'basic',
        'anova_repeated': 'basic',
        'chi_square': 'basic',
        'fisher_exact': 'basic',
        'mcnemar': 'basic',
        'mann_whitney': 'basic',
        'wilcoxon': 'basic',
        'kruskal_wallis': 'basic',
        'friedman': 'basic',

        # Correlation
        'correlation': 'correlation',
        'pearson': 'correlation',
        'spearman': 'correlation',
        'kendall': 'correlation',
        'partial_correlation': 'correlation',

        # Regression
        'linear_regression': 'regression',
        'multiple_regression': 'regression',
        'logistic_regression': 'regression',
        'ordinal_regression': 'regression',
        'multinomial_regression': 'regression',
        'poisson_regression': 'regression',
        'negative_binomial': 'regression',

        # Survival Analysis
        'kaplan_meier': 'survival',
        'cox_regression': 'survival',
        'log_rank': 'survival',
        'competing_risks': 'survival',
        'parametric_survival': 'survival',

        # Longitudinal
        'mixed_model': 'longitudinal',
        'linear_mixed_model': 'longitudinal',
        'generalized_mixed_model': 'longitudinal',
        'gee': 'longitudinal',
        'growth_curve': 'longitudinal',

        # Causal Inference
        'instrumental_variables': 'causal',
        'iv_2sls': 'causal',
        'regression_discontinuity': 'causal',
        'rdd': 'causal',
        'difference_in_differences': 'causal',
        'did': 'causal',
        'propensity_score': 'causal',

        # Machine Learning
        'lasso': 'ml',
        'ridge': 'ml',
        'elastic_net': 'ml',
        'random_forest': 'ml',
        'gradient_boosting': 'ml',
        'xgboost': 'ml',

        # Bayesian
        'bayesian_ttest': 'bayesian',
        'bayesian_regression': 'bayesian',
        'bayesian_correlation': 'bayesian',
        'bayesian_proportion': 'bayesian',

        # Mediation
        'mediation': 'mediation',
        'baron_kenny': 'mediation',
        'bootstrap_mediation': 'mediation',
        'causal_mediation': 'mediation',

        # Sensitivity Analysis
        'e_value': 'sensitivity',
        'tipping_point': 'sensitivity',
        'influence_diagnostics': 'sensitivity',
        'specification_curve': 'sensitivity',

        # Power Analysis
        'power_ttest': 'power',
        'power_anova': 'power',
        'power_chi_square': 'power',
        'power_regression': 'power',
        'power_survival': 'power',

        # Diagnostic
        'roc_analysis': 'diagnostic',
        'sensitivity_specificity': 'diagnostic',
        'diagnostic_accuracy': 'diagnostic',

        # Meta-analysis
        'meta_analysis': 'meta',
        'random_effects_meta': 'meta',
        'fixed_effects_meta': 'meta',
        'network_meta_analysis': 'meta',
    }

    def __init__(
        self,
        default_seed: int = 42,
        include_documentation: bool = True,
        include_installation: bool = True
    ):
        """Initialize the code generator.

        Args:
            default_seed: Default random seed for reproducibility
            include_documentation: Include explanatory comments
            include_installation: Include package installation code
        """
        self.default_seed = default_seed
        self.include_documentation = include_documentation
        self.include_installation = include_installation

        # Import specific generators lazily
        self._python_generator = None
        self._r_generator = None
        self._notebook_generator = None

    @property
    def python_generator(self):
        """Lazy load Python generator."""
        if self._python_generator is None:
            from .python_generator import PythonCodeGenerator
            self._python_generator = PythonCodeGenerator(
                seed=self.default_seed,
                include_docs=self.include_documentation,
                include_install=self.include_installation
            )
        return self._python_generator

    @property
    def r_generator(self):
        """Lazy load R generator."""
        if self._r_generator is None:
            from .r_generator import RCodeGenerator
            self._r_generator = RCodeGenerator(
                seed=self.default_seed,
                include_docs=self.include_documentation,
                include_install=self.include_installation
            )
        return self._r_generator

    @property
    def notebook_generator(self):
        """Lazy load notebook generator."""
        if self._notebook_generator is None:
            from .notebook_generator import NotebookGenerator
            self._notebook_generator = NotebookGenerator()
        return self._notebook_generator

    def generate(
        self,
        analysis_type: str,
        parameters: Dict[str, Any],
        data_info: Optional[Dict[str, Any]] = None,
        languages: List[str] = ['python', 'r'],
        formats: List[CodeFormat] = None,
        seed: Optional[int] = None
    ) -> AnalysisCodeBundle:
        """Generate code for a statistical analysis.

        Args:
            analysis_type: Type of analysis (e.g., 'ttest', 'logistic_regression')
            parameters: Analysis parameters
            data_info: Information about the data (columns, types, etc.)
            languages: Languages to generate ('python', 'r', or both)
            formats: Output formats (defaults to scripts)
            seed: Random seed (uses default if not specified)

        Returns:
            AnalysisCodeBundle with generated code
        """
        if formats is None:
            formats = [CodeFormat.PYTHON_SCRIPT, CodeFormat.R_SCRIPT]

        seed = seed or self.default_seed

        bundle = AnalysisCodeBundle(
            analysis_type=analysis_type,
            analysis_name=self._get_analysis_name(analysis_type),
            data_description=self._generate_data_description(data_info),
            methodology_notes=self._get_methodology_notes(analysis_type)
        )

        # Generate Python code
        if 'python' in languages:
            bundle.python_code = self.python_generator.generate(
                analysis_type=analysis_type,
                parameters=parameters,
                data_info=data_info,
                seed=seed
            )

            # Generate Jupyter notebook if requested
            if CodeFormat.JUPYTER_NOTEBOOK in formats:
                bundle.jupyter_notebook = self.notebook_generator.to_jupyter(
                    bundle.python_code,
                    title=bundle.analysis_name,
                    methodology=bundle.methodology_notes
                )

        # Generate R code
        if 'r' in languages:
            bundle.r_code = self.r_generator.generate(
                analysis_type=analysis_type,
                parameters=parameters,
                data_info=data_info,
                seed=seed
            )

            # Generate R Markdown if requested
            if CodeFormat.R_MARKDOWN in formats:
                bundle.r_markdown = self.notebook_generator.to_rmarkdown(
                    bundle.r_code,
                    title=bundle.analysis_name,
                    methodology=bundle.methodology_notes
                )

        return bundle

    def generate_from_results(
        self,
        analysis_results: Dict[str, Any],
        data_info: Optional[Dict[str, Any]] = None,
        languages: List[str] = ['python', 'r']
    ) -> AnalysisCodeBundle:
        """Generate code from analysis results.

        This method extracts parameters from completed analysis results
        and generates reproducible code.

        Args:
            analysis_results: Results from a completed analysis
            data_info: Information about the data
            languages: Languages to generate

        Returns:
            AnalysisCodeBundle with generated code
        """
        # Extract analysis type and parameters from results
        analysis_type = analysis_results.get('analysis_type', 'unknown')
        parameters = analysis_results.get('parameters', {})

        # If results contain detailed info, extract it
        if 'method' in analysis_results:
            analysis_type = analysis_results['method']

        return self.generate(
            analysis_type=analysis_type,
            parameters=parameters,
            data_info=data_info,
            languages=languages
        )

    def get_supported_analyses(self) -> Dict[str, List[str]]:
        """Get all supported analysis types by category.

        Returns:
            Dict mapping categories to analysis types
        """
        categories = {}
        for analysis, category in self.ANALYSIS_CATEGORIES.items():
            if category not in categories:
                categories[category] = []
            categories[category].append(analysis)
        return categories

    def _get_analysis_name(self, analysis_type: str) -> str:
        """Get human-readable name for analysis type."""
        names = {
            'ttest': 'Independent Samples t-Test',
            'ttest_independent': 'Independent Samples t-Test',
            'ttest_paired': 'Paired Samples t-Test',
            'ttest_one_sample': 'One-Sample t-Test',
            'anova': 'One-Way ANOVA',
            'anova_oneway': 'One-Way ANOVA',
            'anova_twoway': 'Two-Way ANOVA',
            'anova_repeated': 'Repeated Measures ANOVA',
            'chi_square': 'Chi-Square Test',
            'fisher_exact': "Fisher's Exact Test",
            'mcnemar': "McNemar's Test",
            'mann_whitney': 'Mann-Whitney U Test',
            'wilcoxon': 'Wilcoxon Signed-Rank Test',
            'kruskal_wallis': 'Kruskal-Wallis H Test',
            'friedman': 'Friedman Test',
            'correlation': 'Correlation Analysis',
            'pearson': 'Pearson Correlation',
            'spearman': 'Spearman Correlation',
            'kendall': 'Kendall Tau Correlation',
            'partial_correlation': 'Partial Correlation',
            'linear_regression': 'Linear Regression',
            'multiple_regression': 'Multiple Linear Regression',
            'logistic_regression': 'Logistic Regression',
            'ordinal_regression': 'Ordinal Regression',
            'multinomial_regression': 'Multinomial Logistic Regression',
            'poisson_regression': 'Poisson Regression',
            'negative_binomial': 'Negative Binomial Regression',
            'kaplan_meier': 'Kaplan-Meier Survival Analysis',
            'cox_regression': 'Cox Proportional Hazards Regression',
            'log_rank': 'Log-Rank Test',
            'competing_risks': 'Competing Risks Analysis',
            'mixed_model': 'Linear Mixed Effects Model',
            'linear_mixed_model': 'Linear Mixed Effects Model',
            'generalized_mixed_model': 'Generalized Linear Mixed Model',
            'gee': 'Generalized Estimating Equations',
            'growth_curve': 'Growth Curve Model',
            'instrumental_variables': 'Instrumental Variables (2SLS)',
            'iv_2sls': 'Two-Stage Least Squares',
            'regression_discontinuity': 'Regression Discontinuity Design',
            'rdd': 'Regression Discontinuity Design',
            'difference_in_differences': 'Difference-in-Differences',
            'did': 'Difference-in-Differences',
            'propensity_score': 'Propensity Score Analysis',
            'lasso': 'LASSO Regression',
            'ridge': 'Ridge Regression',
            'elastic_net': 'Elastic Net Regression',
            'random_forest': 'Random Forest',
            'gradient_boosting': 'Gradient Boosting',
            'xgboost': 'XGBoost',
            'bayesian_ttest': 'Bayesian t-Test',
            'bayesian_regression': 'Bayesian Regression',
            'bayesian_correlation': 'Bayesian Correlation',
            'mediation': 'Mediation Analysis',
            'baron_kenny': 'Baron-Kenny Mediation',
            'bootstrap_mediation': 'Bootstrap Mediation Analysis',
            'causal_mediation': 'Causal Mediation Analysis',
            'e_value': 'E-Value Sensitivity Analysis',
            'tipping_point': 'Tipping Point Analysis',
            'influence_diagnostics': 'Influence Diagnostics',
            'specification_curve': 'Specification Curve Analysis',
            'power_ttest': 't-Test Power Analysis',
            'power_anova': 'ANOVA Power Analysis',
            'roc_analysis': 'ROC Curve Analysis',
            'meta_analysis': 'Meta-Analysis',
        }
        return names.get(analysis_type, analysis_type.replace('_', ' ').title())

    def _generate_data_description(self, data_info: Optional[Dict[str, Any]]) -> str:
        """Generate description of the data structure."""
        if not data_info:
            return "Data structure not specified."

        desc_parts = []

        if 'n_rows' in data_info:
            desc_parts.append(f"Sample size: {data_info['n_rows']} observations")

        if 'columns' in data_info:
            desc_parts.append(f"Variables: {', '.join(data_info['columns'])}")

        if 'outcome' in data_info:
            desc_parts.append(f"Outcome variable: {data_info['outcome']}")

        if 'predictors' in data_info:
            desc_parts.append(f"Predictor variables: {', '.join(data_info['predictors'])}")

        return "\n".join(desc_parts)

    def _get_methodology_notes(self, analysis_type: str) -> str:
        """Get methodology notes for analysis type."""
        notes = {
            'ttest': """
The independent samples t-test compares means between two groups.
Assumptions: normal distribution within groups, homogeneity of variance (assessed by Levene's test).
If variances are unequal, Welch's t-test is recommended.
""",
            'anova': """
One-way ANOVA tests for differences in means across three or more groups.
Assumptions: normality, homogeneity of variance, independence.
Post-hoc tests (e.g., Tukey HSD) identify which groups differ.
""",
            'chi_square': """
Chi-square test examines the association between two categorical variables.
Expected cell counts should be >= 5; use Fisher's exact test for small samples.
""",
            'logistic_regression': """
Logistic regression models binary outcomes as a function of predictors.
Coefficients are log-odds; exponentiate for odds ratios.
Assess model fit with Hosmer-Lemeshow test and discrimination with ROC-AUC.
""",
            'cox_regression': """
Cox proportional hazards regression models time-to-event outcomes.
The proportional hazards assumption should be verified using Schoenfeld residuals.
Hazard ratios represent relative risk of the event.
""",
            'mixed_model': """
Linear mixed models account for both fixed and random effects.
Useful for repeated measures, clustered data, or longitudinal designs.
Random effects capture correlation within clusters/subjects.
""",
            'gee': """
Generalized Estimating Equations provide population-averaged estimates.
Working correlation structures: independent, exchangeable, AR(1), unstructured.
Robust standard errors account for within-cluster correlation.
""",
            'instrumental_variables': """
Instrumental variables address endogeneity through two-stage estimation.
Instruments must be: (1) relevant (correlated with endogenous variable),
(2) valid (uncorrelated with error term).
Test instrument strength with first-stage F-statistic (>10).
""",
            'lasso': """
LASSO (L1 regularization) performs variable selection and regularization.
Use cross-validation to select optimal lambda.
Produces sparse models by shrinking some coefficients to zero.
""",
            'bayesian_regression': """
Bayesian regression provides posterior distributions for parameters.
Specify prior distributions reflecting existing knowledge.
Credible intervals represent probability of parameter values.
""",
            'mediation': """
Mediation analysis examines indirect effects through intervening variables.
Total effect = Direct effect + Indirect effect.
Bootstrap confidence intervals are preferred for inference.
""",
            'e_value': """
E-values quantify unmeasured confounding needed to explain away an effect.
Higher E-values indicate more robust findings.
Useful for sensitivity analysis in observational studies.
""",
        }

        category = self.ANALYSIS_CATEGORIES.get(analysis_type, 'basic')
        if analysis_type in notes:
            return notes[analysis_type].strip()

        # Return generic note based on category
        category_notes = {
            'basic': "This is a standard hypothesis test. Check assumptions before interpretation.",
            'regression': "Regression models estimate relationships between variables. Check model diagnostics.",
            'survival': "Survival analysis models time-to-event data. Verify proportional hazards assumption for Cox models.",
            'longitudinal': "Longitudinal methods account for within-subject correlation over time.",
            'causal': "Causal inference methods aim to estimate treatment effects under specific assumptions.",
            'ml': "Machine learning methods prioritize prediction; use cross-validation for model selection.",
            'bayesian': "Bayesian methods require prior specification and yield posterior distributions.",
            'sensitivity': "Sensitivity analyses assess robustness to violations of assumptions.",
        }
        return category_notes.get(category, "Statistical analysis methodology.")


def compute_code_hash(code: str) -> str:
    """Compute hash of generated code for versioning."""
    return hashlib.sha256(code.encode()).hexdigest()[:12]
