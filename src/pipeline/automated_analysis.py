"""Automated Analysis Pipeline.

One-click full analysis:
- Auto-detect study design
- Smart variable role detection
- Assumption-driven test selection
- Complete analysis workflow
- Automatic report generation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import warnings
from datetime import datetime


class StudyDesign(Enum):
    """Types of study designs."""
    RCT = "Randomized Controlled Trial"
    COHORT = "Cohort Study"
    CASE_CONTROL = "Case-Control Study"
    CROSS_SECTIONAL = "Cross-Sectional Study"
    BEFORE_AFTER = "Before-After Study"
    SURVIVAL = "Time-to-Event Analysis"
    DIAGNOSTIC = "Diagnostic Accuracy Study"
    META_ANALYSIS = "Meta-Analysis"
    UNKNOWN = "Unknown"


class VariableRole(Enum):
    """Roles that variables can play."""
    OUTCOME = "outcome"
    EXPOSURE = "exposure"
    TIME = "time"
    EVENT = "event"
    CONFOUNDER = "confounder"
    EFFECT_MODIFIER = "effect_modifier"
    ID = "identifier"
    UNKNOWN = "unknown"


@dataclass
class VariableInfo:
    """Information about a variable."""
    name: str
    dtype: str
    role: VariableRole
    variable_type: str  # continuous, categorical, binary, ordinal
    n_missing: int
    n_unique: int
    distribution: Optional[str] = None
    suggested_transformations: List[str] = field(default_factory=list)


@dataclass
class AnalysisPlan:
    """Complete analysis plan."""
    study_design: StudyDesign
    variables: Dict[str, VariableInfo]
    primary_analysis: Dict[str, Any]
    secondary_analyses: List[Dict[str, Any]]
    assumption_checks: List[str]
    sensitivity_analyses: List[str]
    visualizations: List[str]
    warnings: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Result of automated analysis."""
    plan: AnalysisPlan
    results: Dict[str, Any]
    figures: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    report_sections: Dict[str, str]
    audit_log: List[Dict[str, Any]]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class AutomatedAnalysisPipeline:
    """Automated analysis pipeline."""

    # Pattern matching for variable roles
    ROLE_PATTERNS = {
        VariableRole.OUTCOME: [
            'outcome', 'result', 'death', 'mortality', 'event', 'endpoint',
            'primary', 'secondary', 'response', 'success', 'failure'
        ],
        VariableRole.EXPOSURE: [
            'treatment', 'group', 'arm', 'intervention', 'drug', 'therapy',
            'exposed', 'exposure', 'case', 'control'
        ],
        VariableRole.TIME: [
            'time', 'duration', 'follow', 'days', 'months', 'years', 'date',
            'survival', 'fu_time', 'os_time', 'pfs_time', 'tte'
        ],
        VariableRole.EVENT: [
            'event', 'status', 'censor', 'death', 'died', 'mortality'
        ],
        VariableRole.ID: [
            'id', 'patient_id', 'subject_id', 'record_id', 'study_id', 'mrn'
        ],
        VariableRole.CONFOUNDER: [
            'age', 'sex', 'gender', 'bmi', 'race', 'ethnicity', 'smoking',
            'diabetes', 'hypertension', 'ckd', 'chf', 'cad', 'copd',
            'education', 'income', 'insurance'
        ]
    }

    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level
        self.audit_log = []

    def run_full_analysis(
        self,
        data: pd.DataFrame,
        outcome: Optional[str] = None,
        exposure: Optional[str] = None,
        research_question: Optional[str] = None
    ) -> AnalysisResult:
        """Run complete automated analysis.

        Args:
            data: Input DataFrame
            outcome: Outcome variable (auto-detected if None)
            exposure: Exposure/treatment variable (auto-detected if None)
            research_question: Optional research question for guidance

        Returns:
            AnalysisResult with complete analysis
        """
        self._log("Starting automated analysis pipeline")

        # Step 1: Profile data and detect variable roles
        variables = self._profile_variables(data)
        self._log(f"Profiled {len(variables)} variables")

        # Auto-detect outcome and exposure if not provided
        if outcome is None:
            outcome = self._auto_detect_outcome(variables)
            self._log(f"Auto-detected outcome: {outcome}")

        if exposure is None:
            exposure = self._auto_detect_exposure(variables, outcome)
            self._log(f"Auto-detected exposure: {exposure}")

        # Update roles based on user input
        if outcome and outcome in variables:
            variables[outcome].role = VariableRole.OUTCOME
        if exposure and exposure in variables:
            variables[exposure].role = VariableRole.EXPOSURE

        # Step 2: Detect study design
        study_design = self._detect_study_design(data, variables, outcome, exposure)
        self._log(f"Detected study design: {study_design.value}")

        # Step 3: Create analysis plan
        plan = self._create_analysis_plan(
            data, variables, study_design, outcome, exposure, research_question
        )
        self._log(f"Created analysis plan with {len(plan.secondary_analyses)} secondary analyses")

        # Step 4: Run analyses
        results = self._execute_analyses(data, plan, outcome, exposure)
        self._log("Completed statistical analyses")

        # Step 5: Generate visualizations
        figures = self._generate_figures(data, plan, results, outcome, exposure)
        self._log(f"Generated {len(figures)} figures")

        # Step 6: Generate tables
        tables = self._generate_tables(data, plan, results)
        self._log(f"Generated {len(tables)} tables")

        # Step 7: Generate report sections
        report_sections = self._generate_report(plan, results, figures, tables)
        self._log("Generated report sections")

        return AnalysisResult(
            plan=plan,
            results=results,
            figures=figures,
            tables=tables,
            report_sections=report_sections,
            audit_log=self.audit_log.copy()
        )

    def _profile_variables(self, data: pd.DataFrame) -> Dict[str, VariableInfo]:
        """Profile all variables in the dataset."""
        variables = {}

        for col in data.columns:
            dtype = str(data[col].dtype)
            n_missing = data[col].isna().sum()
            n_unique = data[col].nunique()

            # Determine variable type
            if data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                if n_unique <= 2:
                    var_type = 'binary'
                elif n_unique <= 10:
                    var_type = 'ordinal'
                else:
                    var_type = 'continuous'
            else:
                if n_unique <= 2:
                    var_type = 'binary'
                elif n_unique <= 20:
                    var_type = 'categorical'
                else:
                    var_type = 'text'

            # Check distribution for continuous
            distribution = None
            transformations = []
            if var_type == 'continuous' and n_unique > 10:
                clean_data = data[col].dropna()
                if len(clean_data) > 8:
                    from scipy import stats
                    _, p_normal = stats.shapiro(clean_data.sample(min(len(clean_data), 5000)))
                    if p_normal < 0.05:
                        distribution = 'non-normal'
                        # Check skewness
                        skew = clean_data.skew()
                        if skew > 1:
                            transformations.append('log')
                        elif skew < -1:
                            transformations.append('reflect_log')
                    else:
                        distribution = 'normal'

            # Detect role
            role = self._detect_variable_role(col)

            variables[col] = VariableInfo(
                name=col,
                dtype=dtype,
                role=role,
                variable_type=var_type,
                n_missing=n_missing,
                n_unique=n_unique,
                distribution=distribution,
                suggested_transformations=transformations
            )

        return variables

    def _detect_variable_role(self, col_name: str) -> VariableRole:
        """Detect variable role from column name."""
        col_lower = col_name.lower().replace('_', '').replace('-', '')

        for role, patterns in self.ROLE_PATTERNS.items():
            for pattern in patterns:
                pattern_clean = pattern.replace('_', '')
                if pattern_clean in col_lower:
                    return role

        return VariableRole.UNKNOWN

    def _auto_detect_outcome(self, variables: Dict[str, VariableInfo]) -> Optional[str]:
        """Auto-detect the outcome variable."""
        # First look for explicitly marked outcomes
        for name, var in variables.items():
            if var.role == VariableRole.OUTCOME:
                return name

        # Look for common outcome patterns
        outcome_priority = ['death', 'mortality', 'outcome', 'event', 'endpoint', 'response']
        for pattern in outcome_priority:
            for name, var in variables.items():
                if pattern in name.lower() and var.variable_type in ['binary', 'continuous']:
                    return name

        return None

    def _auto_detect_exposure(
        self,
        variables: Dict[str, VariableInfo],
        outcome: Optional[str]
    ) -> Optional[str]:
        """Auto-detect the exposure/treatment variable."""
        for name, var in variables.items():
            if var.role == VariableRole.EXPOSURE and name != outcome:
                return name

        # Look for common exposure patterns
        exposure_priority = ['treatment', 'group', 'arm', 'intervention', 'drug']
        for pattern in exposure_priority:
            for name, var in variables.items():
                if pattern in name.lower() and name != outcome:
                    if var.variable_type in ['binary', 'categorical']:
                        return name

        return None

    def _detect_study_design(
        self,
        data: pd.DataFrame,
        variables: Dict[str, VariableInfo],
        outcome: Optional[str],
        exposure: Optional[str]
    ) -> StudyDesign:
        """Detect study design from data structure."""
        # Check for survival data
        has_time = any(v.role == VariableRole.TIME for v in variables.values())
        has_event = any(v.role == VariableRole.EVENT for v in variables.values())

        if has_time and has_event:
            return StudyDesign.SURVIVAL

        # Check for RCT-like structure
        if exposure:
            exp_var = variables.get(exposure)
            if exp_var and exp_var.variable_type in ['binary', 'categorical']:
                # Look for balanced groups (suggestive of RCT)
                groups = data[exposure].value_counts()
                if len(groups) == 2:
                    ratio = groups.max() / groups.min()
                    if ratio < 1.5:
                        return StudyDesign.RCT

        # Check for case-control
        if outcome:
            out_var = variables.get(outcome)
            if out_var and out_var.variable_type == 'binary':
                # Check if cases/controls are balanced
                outcome_dist = data[outcome].value_counts()
                if len(outcome_dist) == 2:
                    ratio = outcome_dist.max() / outcome_dist.min()
                    if ratio < 2:
                        return StudyDesign.CASE_CONTROL

        # Check for diagnostic study
        diag_patterns = ['sensitivity', 'specificity', 'test', 'diagnostic', 'reference']
        if any(any(p in c.lower() for p in diag_patterns) for c in data.columns):
            return StudyDesign.DIAGNOSTIC

        # Default to cohort or cross-sectional
        return StudyDesign.COHORT

    def _create_analysis_plan(
        self,
        data: pd.DataFrame,
        variables: Dict[str, VariableInfo],
        study_design: StudyDesign,
        outcome: Optional[str],
        exposure: Optional[str],
        research_question: Optional[str]
    ) -> AnalysisPlan:
        """Create comprehensive analysis plan."""
        warnings = []

        # Identify confounders
        confounders = [
            name for name, var in variables.items()
            if var.role == VariableRole.CONFOUNDER
            and name != outcome and name != exposure
        ]

        # Primary analysis based on study design and variable types
        primary_analysis = self._select_primary_analysis(
            data, variables, study_design, outcome, exposure, confounders
        )

        # Secondary analyses
        secondary_analyses = self._select_secondary_analyses(
            data, variables, study_design, outcome, exposure, confounders
        )

        # Required assumption checks
        assumption_checks = self._select_assumption_checks(
            variables, outcome, exposure, primary_analysis
        )

        # Sensitivity analyses
        sensitivity_analyses = self._select_sensitivity_analyses(
            study_design, primary_analysis
        )

        # Visualizations
        visualizations = self._select_visualizations(
            variables, study_design, outcome, exposure
        )

        # Add warnings
        if not outcome:
            warnings.append("No outcome variable detected - please specify")
        if not exposure:
            warnings.append("No exposure/treatment variable detected - please specify")
        if len(confounders) == 0:
            warnings.append("No confounders detected - consider specifying covariates")

        return AnalysisPlan(
            study_design=study_design,
            variables=variables,
            primary_analysis=primary_analysis,
            secondary_analyses=secondary_analyses,
            assumption_checks=assumption_checks,
            sensitivity_analyses=sensitivity_analyses,
            visualizations=visualizations,
            warnings=warnings
        )

    def _select_primary_analysis(
        self,
        data: pd.DataFrame,
        variables: Dict[str, VariableInfo],
        study_design: StudyDesign,
        outcome: Optional[str],
        exposure: Optional[str],
        confounders: List[str]
    ) -> Dict[str, Any]:
        """Select primary analysis based on study design and data."""
        if not outcome or not exposure:
            return {'type': 'descriptive', 'method': 'summary_statistics'}

        outcome_var = variables.get(outcome)
        exposure_var = variables.get(exposure)

        if study_design == StudyDesign.SURVIVAL:
            return {
                'type': 'survival',
                'method': 'cox_regression',
                'outcome': outcome,
                'exposure': exposure,
                'confounders': confounders,
                'secondary_method': 'kaplan_meier'
            }

        if outcome_var.variable_type == 'binary':
            return {
                'type': 'binary_outcome',
                'method': 'logistic_regression',
                'outcome': outcome,
                'exposure': exposure,
                'confounders': confounders,
                'unadjusted_method': 'chi_square'
            }

        if outcome_var.variable_type == 'continuous':
            if outcome_var.distribution == 'non-normal':
                return {
                    'type': 'continuous_outcome',
                    'method': 'linear_regression',
                    'outcome': outcome,
                    'exposure': exposure,
                    'confounders': confounders,
                    'unadjusted_method': 'mann_whitney' if exposure_var.variable_type == 'binary' else 'kruskal_wallis'
                }
            else:
                return {
                    'type': 'continuous_outcome',
                    'method': 'linear_regression',
                    'outcome': outcome,
                    'exposure': exposure,
                    'confounders': confounders,
                    'unadjusted_method': 'ttest' if exposure_var.variable_type == 'binary' else 'anova'
                }

        return {'type': 'descriptive', 'method': 'summary_statistics'}

    def _select_secondary_analyses(
        self,
        data: pd.DataFrame,
        variables: Dict[str, VariableInfo],
        study_design: StudyDesign,
        outcome: Optional[str],
        exposure: Optional[str],
        confounders: List[str]
    ) -> List[Dict[str, Any]]:
        """Select secondary analyses."""
        analyses = []

        # Descriptive statistics always
        analyses.append({
            'type': 'descriptive',
            'method': 'table1',
            'group_by': exposure
        })

        # Subgroup analyses for key confounders
        for conf in confounders[:3]:  # Top 3 confounders
            analyses.append({
                'type': 'subgroup',
                'method': 'stratified_analysis',
                'subgroup_variable': conf
            })

        # Missing data analysis if applicable
        missing_vars = [
            name for name, var in variables.items()
            if var.n_missing > 0 and var.n_missing < len(data) * 0.5
        ]
        if missing_vars:
            analyses.append({
                'type': 'missing_data',
                'method': 'multiple_imputation',
                'variables': missing_vars
            })

        return analyses

    def _select_assumption_checks(
        self,
        variables: Dict[str, VariableInfo],
        outcome: Optional[str],
        exposure: Optional[str],
        primary_analysis: Dict[str, Any]
    ) -> List[str]:
        """Select assumption checks based on primary analysis."""
        checks = []

        if primary_analysis.get('method') == 'linear_regression':
            checks.extend([
                'normality_residuals',
                'homoscedasticity',
                'linearity',
                'multicollinearity'
            ])
        elif primary_analysis.get('method') == 'logistic_regression':
            checks.extend([
                'linearity_logit',
                'multicollinearity',
                'influential_observations'
            ])
        elif primary_analysis.get('method') == 'cox_regression':
            checks.extend([
                'proportional_hazards',
                'influential_observations'
            ])

        # Add normality check for continuous outcome
        if outcome and outcome in variables:
            if variables[outcome].variable_type == 'continuous':
                checks.append('normality_outcome')

        return checks

    def _select_sensitivity_analyses(
        self,
        study_design: StudyDesign,
        primary_analysis: Dict[str, Any]
    ) -> List[str]:
        """Select sensitivity analyses."""
        analyses = []

        # E-value for observational studies
        if study_design in [StudyDesign.COHORT, StudyDesign.CASE_CONTROL]:
            analyses.append('e_value')

        # Complete case vs imputed
        if 'missing_data' in str(primary_analysis):
            analyses.append('complete_case_comparison')

        # Alternative model specifications
        analyses.append('alternative_confounders')
        analyses.append('influential_observation_removal')

        return analyses

    def _select_visualizations(
        self,
        variables: Dict[str, VariableInfo],
        study_design: StudyDesign,
        outcome: Optional[str],
        exposure: Optional[str]
    ) -> List[str]:
        """Select visualizations to generate."""
        viz = ['distribution_plots', 'missing_data_pattern']

        if study_design == StudyDesign.SURVIVAL:
            viz.extend(['kaplan_meier_curve', 'forest_plot'])
        elif outcome and variables.get(outcome, {}).variable_type == 'binary':
            viz.extend(['odds_ratio_forest_plot', 'roc_curve'])
        elif outcome and variables.get(outcome, {}).variable_type == 'continuous':
            viz.extend(['regression_diagnostics', 'effect_plot'])

        if exposure:
            viz.append('comparison_boxplot')

        return viz

    def _execute_analyses(
        self,
        data: pd.DataFrame,
        plan: AnalysisPlan,
        outcome: Optional[str],
        exposure: Optional[str]
    ) -> Dict[str, Any]:
        """Execute all planned analyses."""
        results = {
            'primary': None,
            'secondary': [],
            'assumptions': {},
            'sensitivity': {}
        }

        # Import analysis modules
        import sys
        from pathlib import Path
        stats_path = Path(__file__).parent.parent / "stats"
        sys.path.insert(0, str(stats_path))

        try:
            from stats.descriptive import DescriptiveStats
            from stats.comparative import ComparativeTests
            from stats.regression import RegressionAnalysis
            from stats.assumptions import AssumptionChecker

            descriptive = DescriptiveStats()
            comparative = ComparativeTests()
            regression = RegressionAnalysis()
            assumptions = AssumptionChecker()

            # Primary analysis
            primary = plan.primary_analysis
            if primary.get('method') == 'logistic_regression' and outcome and exposure:
                confounders = primary.get('confounders', [])
                predictors = [exposure] + confounders
                result = regression.logistic_regression(data, outcome, predictors)
                results['primary'] = result.to_dict()

            elif primary.get('method') == 'linear_regression' and outcome and exposure:
                confounders = primary.get('confounders', [])
                predictors = [exposure] + confounders
                result = regression.linear_regression(data, outcome, predictors)
                results['primary'] = result.to_dict()

            elif primary.get('method') == 'ttest' and outcome and exposure:
                groups = data[exposure].unique()
                if len(groups) == 2:
                    group1 = data[data[exposure] == groups[0]][outcome]
                    group2 = data[data[exposure] == groups[1]][outcome]
                    result = comparative.independent_ttest(group1, group2)
                    results['primary'] = result.to_dict()

            # Assumption checks
            for check in plan.assumption_checks:
                if check == 'normality_outcome' and outcome:
                    result = assumptions.test_normality(data[outcome].dropna())
                    results['assumptions'][check] = result.to_dict()
                elif check == 'multicollinearity' and outcome and exposure:
                    confounders = plan.primary_analysis.get('confounders', [])
                    if confounders:
                        result = assumptions.test_multicollinearity(
                            data, [exposure] + confounders
                        )
                        results['assumptions'][check] = result.to_dict()

            # Secondary analyses - Table 1
            for secondary in plan.secondary_analyses:
                if secondary.get('method') == 'table1' and exposure:
                    continuous_vars = [
                        name for name, var in plan.variables.items()
                        if var.variable_type == 'continuous' and name != exposure
                    ][:10]
                    categorical_vars = [
                        name for name, var in plan.variables.items()
                        if var.variable_type in ['binary', 'categorical'] and name != exposure
                    ][:10]

                    if continuous_vars or categorical_vars:
                        result = descriptive.table1(
                            data, exposure,
                            continuous_vars=continuous_vars,
                            categorical_vars=categorical_vars
                        )
                        results['secondary'].append({
                            'type': 'table1',
                            'result': result.to_dict()
                        })

        except Exception as e:
            results['error'] = str(e)

        return results

    def _generate_figures(
        self,
        data: pd.DataFrame,
        plan: AnalysisPlan,
        results: Dict[str, Any],
        outcome: Optional[str],
        exposure: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Generate visualizations."""
        figures = []

        # This would integrate with visualization modules
        for viz_type in plan.visualizations:
            figures.append({
                'type': viz_type,
                'status': 'planned',
                'description': f'{viz_type} visualization'
            })

        return figures

    def _generate_tables(
        self,
        data: pd.DataFrame,
        plan: AnalysisPlan,
        results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate summary tables."""
        tables = []

        # Table 1 from secondary analyses
        for secondary in results.get('secondary', []):
            if secondary.get('type') == 'table1':
                tables.append({
                    'type': 'table1',
                    'title': 'Baseline Characteristics',
                    'data': secondary.get('result', {})
                })

        return tables

    def _generate_report(
        self,
        plan: AnalysisPlan,
        results: Dict[str, Any],
        figures: List[Dict[str, Any]],
        tables: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Generate report sections."""
        sections = {}

        # Methods section
        sections['methods'] = self._generate_methods_section(plan)

        # Results section
        sections['results'] = self._generate_results_section(plan, results)

        return sections

    def _generate_methods_section(self, plan: AnalysisPlan) -> str:
        """Generate statistical methods section."""
        lines = [
            "## Statistical Methods",
            "",
            f"**Study Design:** {plan.study_design.value}",
            ""
        ]

        # Primary analysis
        primary = plan.primary_analysis
        if primary.get('method') == 'logistic_regression':
            lines.append("**Primary Analysis:** Multivariable logistic regression was used to estimate adjusted odds ratios (aOR) with 95% confidence intervals.")
            confounders = primary.get('confounders', [])
            if confounders:
                lines.append(f"Models were adjusted for: {', '.join(confounders)}.")
        elif primary.get('method') == 'linear_regression':
            lines.append("**Primary Analysis:** Multivariable linear regression was used to estimate adjusted differences with 95% confidence intervals.")
        elif primary.get('method') == 'cox_regression':
            lines.append("**Primary Analysis:** Cox proportional hazards regression was used to estimate adjusted hazard ratios (aHR) with 95% confidence intervals.")

        lines.append("")
        lines.append(f"Statistical significance was set at α = {self.alpha}.")
        lines.append("All analyses were performed using an automated statistical analysis platform.")

        return '\n'.join(lines)

    def _generate_results_section(
        self,
        plan: AnalysisPlan,
        results: Dict[str, Any]
    ) -> str:
        """Generate results section."""
        lines = ["## Results", ""]

        # Primary results
        primary = results.get('primary')
        if primary:
            if plan.primary_analysis.get('method') == 'logistic_regression':
                coefficients = primary.get('coefficients', {})
                exposure = plan.primary_analysis.get('exposure')
                if exposure and exposure in coefficients:
                    coef = coefficients[exposure]
                    or_val = coef.get('odds_ratio', np.exp(coef.get('coefficient', 0)))
                    ci_lower = coef.get('or_ci_lower', np.exp(coef.get('ci_lower', 0)))
                    ci_upper = coef.get('or_ci_upper', np.exp(coef.get('ci_upper', 0)))
                    p_val = coef.get('p_value', 1)

                    lines.append(f"In the adjusted analysis, {exposure} was associated with {'increased' if or_val > 1 else 'decreased'} odds of the outcome (aOR: {or_val:.2f}, 95% CI: {ci_lower:.2f}-{ci_upper:.2f}, p{'<0.001' if p_val < 0.001 else f'={p_val:.3f}'}).")

        return '\n'.join(lines)

    def _log(self, message: str):
        """Add to audit log."""
        self.audit_log.append({
            'timestamp': datetime.now().isoformat(),
            'message': message
        })
