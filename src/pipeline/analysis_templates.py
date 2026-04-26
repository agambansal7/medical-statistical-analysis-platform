"""Pre-built analysis templates for common study types.

Provides standardized workflows for:
- RCT Analysis (ITT, per-protocol, subgroups)
- Case-Control Study
- Cohort Study with Survival
- Diagnostic Accuracy Study
- Systematic Review / Meta-Analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings


@dataclass
class TemplateStep:
    """A single step in an analysis template."""
    name: str
    description: str
    method: str
    parameters: Dict[str, Any]
    required: bool = True
    depends_on: List[str] = field(default_factory=list)
    educational_note: Optional[str] = None


@dataclass
class AnalysisTemplate:
    """Complete analysis template."""
    name: str
    description: str
    study_type: str
    steps: List[TemplateStep]
    required_variables: Dict[str, str]  # name: description
    optional_variables: Dict[str, str]
    reporting_guidelines: str
    references: List[str] = field(default_factory=list)


class AnalysisTemplates:
    """Pre-built analysis templates for common study designs."""

    def __init__(self):
        self.templates = self._build_templates()

    def get_template(self, template_name: str) -> Optional[AnalysisTemplate]:
        """Get a specific template by name."""
        return self.templates.get(template_name)

    def list_templates(self) -> List[str]:
        """List available templates."""
        return list(self.templates.keys())

    def execute_template(
        self,
        template_name: str,
        data: pd.DataFrame,
        variable_mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """Execute a template with provided data and variable mapping.

        Args:
            template_name: Name of template to execute
            data: DataFrame with study data
            variable_mapping: Dict mapping template variables to actual column names

        Returns:
            Dict with results from each step
        """
        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Unknown template: {template_name}")

        # Validate required variables
        missing_vars = []
        for var in template.required_variables:
            if var not in variable_mapping:
                missing_vars.append(var)

        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")

        # Execute steps
        results = {'template': template_name, 'steps': {}}
        completed_steps = set()

        for step in template.steps:
            # Check dependencies
            deps_met = all(dep in completed_steps for dep in step.depends_on)
            if not deps_met:
                results['steps'][step.name] = {'status': 'skipped', 'reason': 'dependencies not met'}
                continue

            try:
                step_result = self._execute_step(step, data, variable_mapping, results)
                results['steps'][step.name] = {
                    'status': 'completed',
                    'result': step_result
                }
                completed_steps.add(step.name)
            except Exception as e:
                results['steps'][step.name] = {
                    'status': 'error',
                    'error': str(e)
                }
                if step.required:
                    break

        return results

    def _execute_step(
        self,
        step: TemplateStep,
        data: pd.DataFrame,
        variable_mapping: Dict[str, str],
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single template step."""
        # Map parameters
        params = {}
        for key, value in step.parameters.items():
            if isinstance(value, str) and value.startswith('$'):
                # This is a variable reference
                var_name = value[1:]  # Remove $
                if var_name in variable_mapping:
                    params[key] = variable_mapping[var_name]
                else:
                    params[key] = value
            else:
                params[key] = value

        # Import and execute method
        method_parts = step.method.split('.')
        module_name = '.'.join(method_parts[:-1])
        func_name = method_parts[-1]

        # This would dynamically import and call the appropriate method
        # For now, return a placeholder
        return {
            'method': step.method,
            'parameters': params,
            'status': 'executed'
        }

    def _build_templates(self) -> Dict[str, AnalysisTemplate]:
        """Build all available templates."""
        templates = {}

        # RCT Template
        templates['rct'] = self._build_rct_template()
        templates['case_control'] = self._build_case_control_template()
        templates['cohort_survival'] = self._build_cohort_survival_template()
        templates['diagnostic_accuracy'] = self._build_diagnostic_accuracy_template()
        templates['meta_analysis'] = self._build_meta_analysis_template()

        return templates

    def _build_rct_template(self) -> AnalysisTemplate:
        """Build RCT analysis template."""
        steps = [
            TemplateStep(
                name="baseline_characteristics",
                description="Generate Table 1 comparing baseline characteristics between treatment arms",
                method="stats.descriptive.DescriptiveStats.table1",
                parameters={
                    'group_var': '$treatment_arm',
                    'continuous_vars': '$continuous_vars',
                    'categorical_vars': '$categorical_vars'
                },
                educational_note="Table 1 displays baseline characteristics to assess randomization balance. Standardized Mean Differences (SMD) > 0.1 may indicate imbalance."
            ),
            TemplateStep(
                name="itt_analysis",
                description="Intention-to-treat analysis: analyze all randomized patients as assigned",
                method="stats.regression.RegressionAnalysis.logistic_regression",
                parameters={
                    'outcome': '$primary_outcome',
                    'predictors': ['$treatment_arm'],
                    'covariates': '$adjustment_variables'
                },
                depends_on=["baseline_characteristics"],
                educational_note="ITT analysis preserves randomization and avoids selection bias. All randomized patients are analyzed according to their assigned treatment regardless of compliance."
            ),
            TemplateStep(
                name="per_protocol_analysis",
                description="Per-protocol analysis: analyze only patients who completed treatment as assigned",
                method="stats.regression.RegressionAnalysis.logistic_regression",
                parameters={
                    'outcome': '$primary_outcome',
                    'predictors': ['$treatment_arm'],
                    'covariates': '$adjustment_variables',
                    'filter': '$per_protocol_flag'
                },
                depends_on=["itt_analysis"],
                required=False,
                educational_note="Per-protocol analysis may overestimate treatment effect but shows biological efficacy. Compare with ITT to assess robustness."
            ),
            TemplateStep(
                name="subgroup_analyses",
                description="Pre-specified subgroup analyses with interaction testing",
                method="stats.subgroup_analysis.SubgroupAnalyzer.analyze_subgroups",
                parameters={
                    'outcome': '$primary_outcome',
                    'treatment': '$treatment_arm',
                    'subgroup_vars': '$subgroup_variables'
                },
                depends_on=["itt_analysis"],
                educational_note="Subgroup analyses test for effect modification. Look for statistically significant interactions, not just within-subgroup p-values."
            ),
            TemplateStep(
                name="safety_analysis",
                description="Analyze adverse events and safety outcomes",
                method="stats.comparative.ComparativeTests.chi_square_test",
                parameters={
                    'data_col1': '$adverse_events',
                    'data_col2': '$treatment_arm'
                },
                depends_on=["baseline_characteristics"],
                educational_note="Safety analysis should include both serious adverse events and all adverse events. Consider number needed to harm (NNH)."
            ),
            TemplateStep(
                name="secondary_outcomes",
                description="Analyze pre-specified secondary outcomes",
                method="stats.regression.RegressionAnalysis.multiple_outcomes",
                parameters={
                    'outcomes': '$secondary_outcomes',
                    'predictors': ['$treatment_arm'],
                    'covariates': '$adjustment_variables'
                },
                depends_on=["itt_analysis"],
                educational_note="Secondary outcomes are hypothesis-generating. Consider multiple testing adjustments."
            ),
            TemplateStep(
                name="nnt_calculation",
                description="Calculate Number Needed to Treat",
                method="stats.effect_sizes.EffectSizeCalculator.nnt",
                parameters={
                    'treatment_risk': '$treatment_event_rate',
                    'control_risk': '$control_event_rate'
                },
                depends_on=["itt_analysis"],
                educational_note="NNT indicates how many patients need to be treated to prevent one additional event. Lower NNT = more effective treatment."
            )
        ]

        return AnalysisTemplate(
            name="rct",
            description="Comprehensive Randomized Controlled Trial Analysis",
            study_type="RCT",
            steps=steps,
            required_variables={
                'treatment_arm': "Binary treatment assignment (0=control, 1=treatment)",
                'primary_outcome': "Primary efficacy outcome variable",
            },
            optional_variables={
                'continuous_vars': "List of continuous baseline variables",
                'categorical_vars': "List of categorical baseline variables",
                'adjustment_variables': "Pre-specified adjustment covariates",
                'per_protocol_flag': "Binary indicator for per-protocol population",
                'subgroup_variables': "Pre-specified subgroup variables",
                'adverse_events': "Adverse event indicator",
                'secondary_outcomes': "List of secondary outcome variables"
            },
            reporting_guidelines="CONSORT 2010",
            references=[
                "Schulz KF et al. CONSORT 2010 Statement. BMJ 2010;340:c332",
                "Moher D et al. CONSORT 2010 Explanation and Elaboration. BMJ 2010;340:c869"
            ]
        )

    def _build_case_control_template(self) -> AnalysisTemplate:
        """Build case-control study template."""
        steps = [
            TemplateStep(
                name="case_control_characteristics",
                description="Compare characteristics between cases and controls",
                method="stats.descriptive.DescriptiveStats.table1",
                parameters={
                    'group_var': '$case_status',
                    'continuous_vars': '$continuous_vars',
                    'categorical_vars': '$categorical_vars'
                },
                educational_note="Comparing case-control characteristics helps identify potential confounders."
            ),
            TemplateStep(
                name="univariate_analysis",
                description="Univariate analysis of exposure-outcome association",
                method="stats.comparative.ComparativeTests.chi_square_test",
                parameters={
                    'exposure': '$exposure',
                    'outcome': '$case_status'
                },
                educational_note="Univariate OR provides unadjusted estimate. May be confounded."
            ),
            TemplateStep(
                name="multivariable_analysis",
                description="Multivariable logistic regression adjusting for confounders",
                method="stats.regression.RegressionAnalysis.logistic_regression",
                parameters={
                    'outcome': '$case_status',
                    'predictors': ['$exposure'],
                    'covariates': '$confounders'
                },
                depends_on=["univariate_analysis"],
                educational_note="Adjusted OR accounts for confounding. Compare with crude OR to assess confounding."
            ),
            TemplateStep(
                name="matched_analysis",
                description="Conditional logistic regression for matched pairs",
                method="stats.regression.RegressionAnalysis.conditional_logistic",
                parameters={
                    'outcome': '$case_status',
                    'exposure': '$exposure',
                    'strata': '$match_id'
                },
                depends_on=["univariate_analysis"],
                required=False,
                educational_note="Use conditional logistic regression for matched case-control studies to preserve matching."
            ),
            TemplateStep(
                name="sensitivity_unmeasured_confounding",
                description="E-value for sensitivity to unmeasured confounding",
                method="stats.sensitivity_analysis.SensitivityAnalysis.e_value",
                parameters={
                    'estimate': '$adjusted_or',
                    'ci_lower': '$or_ci_lower',
                    'estimate_type': 'OR'
                },
                depends_on=["multivariable_analysis"],
                educational_note="E-value quantifies how strong unmeasured confounding would need to be to explain away the observed association."
            )
        ]

        return AnalysisTemplate(
            name="case_control",
            description="Case-Control Study Analysis",
            study_type="Case-Control",
            steps=steps,
            required_variables={
                'case_status': "Binary case/control indicator (1=case, 0=control)",
                'exposure': "Primary exposure of interest"
            },
            optional_variables={
                'continuous_vars': "Continuous variables for comparison",
                'categorical_vars': "Categorical variables for comparison",
                'confounders': "Potential confounding variables",
                'match_id': "Matching strata identifier for matched studies"
            },
            reporting_guidelines="STROBE",
            references=[
                "von Elm E et al. STROBE Statement. Ann Intern Med 2007;147:573-577"
            ]
        )

    def _build_cohort_survival_template(self) -> AnalysisTemplate:
        """Build cohort study with survival analysis template."""
        steps = [
            TemplateStep(
                name="baseline_characteristics",
                description="Compare baseline characteristics by exposure status",
                method="stats.descriptive.DescriptiveStats.table1",
                parameters={
                    'group_var': '$exposure',
                    'continuous_vars': '$continuous_vars',
                    'categorical_vars': '$categorical_vars'
                },
                educational_note="Baseline table identifies potential confounders needing adjustment."
            ),
            TemplateStep(
                name="follow_up_summary",
                description="Summarize follow-up time and event rates",
                method="stats.survival.SurvivalAnalysis.follow_up_summary",
                parameters={
                    'time': '$time_to_event',
                    'event': '$event_indicator',
                    'group': '$exposure'
                },
                educational_note="Report median follow-up, person-years, and incidence rates by group."
            ),
            TemplateStep(
                name="kaplan_meier",
                description="Kaplan-Meier survival curves with log-rank test",
                method="stats.survival.SurvivalAnalysis.kaplan_meier",
                parameters={
                    'time': '$time_to_event',
                    'event': '$event_indicator',
                    'group': '$exposure'
                },
                depends_on=["follow_up_summary"],
                educational_note="KM curves visualize survival over time. Log-rank test compares overall survival between groups."
            ),
            TemplateStep(
                name="cox_unadjusted",
                description="Unadjusted Cox proportional hazards model",
                method="stats.survival.SurvivalAnalysis.cox_regression",
                parameters={
                    'time': '$time_to_event',
                    'event': '$event_indicator',
                    'predictors': ['$exposure']
                },
                depends_on=["kaplan_meier"],
                educational_note="Crude HR may be confounded. Compare with adjusted HR."
            ),
            TemplateStep(
                name="cox_adjusted",
                description="Multivariable Cox model adjusting for confounders",
                method="stats.survival.SurvivalAnalysis.cox_regression",
                parameters={
                    'time': '$time_to_event',
                    'event': '$event_indicator',
                    'predictors': ['$exposure'],
                    'covariates': '$confounders'
                },
                depends_on=["cox_unadjusted"],
                educational_note="Adjusted HR accounts for confounding. Primary analysis for causal inference."
            ),
            TemplateStep(
                name="proportional_hazards_test",
                description="Test proportional hazards assumption",
                method="stats.survival.SurvivalAnalysis.test_proportional_hazards",
                parameters={
                    'time': '$time_to_event',
                    'event': '$event_indicator',
                    'predictors': ['$exposure'] + ['$confounders']
                },
                depends_on=["cox_adjusted"],
                educational_note="Schoenfeld residuals test PH assumption. If violated, consider time-varying effects or RMST."
            ),
            TemplateStep(
                name="rmst_analysis",
                description="Restricted Mean Survival Time analysis",
                method="stats.advanced_survival.AdvancedSurvivalAnalysis.rmst_comparison",
                parameters={
                    'time': '$time_to_event',
                    'event': '$event_indicator',
                    'group': '$exposure',
                    'tau': '$restriction_time'
                },
                depends_on=["cox_adjusted"],
                required=False,
                educational_note="RMST difference provides absolute effect measure and doesn't require PH assumption."
            ),
            TemplateStep(
                name="competing_risks",
                description="Competing risks analysis if applicable",
                method="stats.advanced_survival.AdvancedSurvivalAnalysis.competing_risks",
                parameters={
                    'time': '$time_to_event',
                    'event': '$event_indicator',
                    'competing_event': '$competing_event',
                    'group': '$exposure'
                },
                depends_on=["cox_adjusted"],
                required=False,
                educational_note="Competing risks account for events that preclude the outcome of interest (e.g., death before recurrence)."
            ),
            TemplateStep(
                name="propensity_score",
                description="Propensity score analysis for confounding control",
                method="stats.propensity_score.PropensityScoreAnalysis.full_analysis",
                parameters={
                    'treatment': '$exposure',
                    'outcome_time': '$time_to_event',
                    'outcome_event': '$event_indicator',
                    'covariates': '$confounders'
                },
                depends_on=["baseline_characteristics"],
                required=False,
                educational_note="PS methods provide additional confounding control. Compare results with traditional adjustment."
            )
        ]

        return AnalysisTemplate(
            name="cohort_survival",
            description="Cohort Study with Survival/Time-to-Event Analysis",
            study_type="Cohort",
            steps=steps,
            required_variables={
                'exposure': "Primary exposure of interest",
                'time_to_event': "Follow-up time variable",
                'event_indicator': "Event indicator (1=event, 0=censored)"
            },
            optional_variables={
                'continuous_vars': "Continuous baseline variables",
                'categorical_vars': "Categorical baseline variables",
                'confounders': "Potential confounding variables",
                'restriction_time': "RMST restriction time (tau)",
                'competing_event': "Competing event indicator"
            },
            reporting_guidelines="STROBE",
            references=[
                "von Elm E et al. STROBE Statement. Ann Intern Med 2007;147:573-577",
                "Royston P, Parmar MK. RMST for RCTs. BMC Med Res Methodol 2013;13:152"
            ]
        )

    def _build_diagnostic_accuracy_template(self) -> AnalysisTemplate:
        """Build diagnostic accuracy study template."""
        steps = [
            TemplateStep(
                name="study_flow",
                description="Document patient flow and exclusions",
                method="stats.descriptive.DescriptiveStats.flow_diagram",
                parameters={
                    'eligible': '$eligible_patients',
                    'enrolled': '$enrolled_patients',
                    'analyzed': '$analyzed_patients'
                },
                educational_note="STARD flow diagram shows patient selection and exclusions."
            ),
            TemplateStep(
                name="population_characteristics",
                description="Describe study population characteristics",
                method="stats.descriptive.DescriptiveStats.summary_statistics",
                parameters={
                    'variables': '$demographic_vars'
                },
                educational_note="Population characteristics help assess generalizability."
            ),
            TemplateStep(
                name="sensitivity_specificity",
                description="Calculate sensitivity, specificity, PPV, NPV",
                method="stats.diagnostic.DiagnosticTests.confusion_matrix_metrics",
                parameters={
                    'test_result': '$index_test',
                    'reference': '$reference_standard'
                },
                educational_note="Basic diagnostic accuracy metrics. Report with 95% CIs."
            ),
            TemplateStep(
                name="roc_analysis",
                description="ROC curve analysis with AUC",
                method="stats.diagnostic.DiagnosticTests.roc_analysis",
                parameters={
                    'predictions': '$test_probability',
                    'reference': '$reference_standard'
                },
                depends_on=["sensitivity_specificity"],
                educational_note="ROC curve shows trade-off between sensitivity and specificity. AUC summarizes overall discrimination."
            ),
            TemplateStep(
                name="optimal_threshold",
                description="Determine optimal diagnostic threshold",
                method="stats.diagnostic.DiagnosticTests.optimal_cutoff",
                parameters={
                    'predictions': '$test_probability',
                    'reference': '$reference_standard',
                    'method': 'youden'
                },
                depends_on=["roc_analysis"],
                educational_note="Youden index maximizes (sensitivity + specificity - 1). Consider clinical context for threshold selection."
            ),
            TemplateStep(
                name="likelihood_ratios",
                description="Calculate positive and negative likelihood ratios",
                method="stats.diagnostic.DiagnosticTests.likelihood_ratios",
                parameters={
                    'test_result': '$index_test',
                    'reference': '$reference_standard'
                },
                depends_on=["sensitivity_specificity"],
                educational_note="LR+ > 10 and LR- < 0.1 indicate useful test. LRs allow Bayesian updating of disease probability."
            ),
            TemplateStep(
                name="calibration",
                description="Assess calibration if continuous predictions",
                method="stats.diagnostic.DiagnosticTests.calibration_plot",
                parameters={
                    'predictions': '$test_probability',
                    'outcomes': '$reference_standard'
                },
                depends_on=["roc_analysis"],
                required=False,
                educational_note="Calibration assesses whether predicted probabilities match observed frequencies."
            )
        ]

        return AnalysisTemplate(
            name="diagnostic_accuracy",
            description="Diagnostic Accuracy Study Analysis",
            study_type="Diagnostic",
            steps=steps,
            required_variables={
                'index_test': "Result of diagnostic test being evaluated",
                'reference_standard': "Reference standard (gold standard) result"
            },
            optional_variables={
                'test_probability': "Continuous test probability/score",
                'demographic_vars': "Demographic variables",
                'eligible_patients': "Number of eligible patients",
                'enrolled_patients': "Number of enrolled patients",
                'analyzed_patients': "Number of patients in analysis"
            },
            reporting_guidelines="STARD 2015",
            references=[
                "Cohen JF et al. STARD 2015 guidelines. BMJ 2016;351:h5527"
            ]
        )

    def _build_meta_analysis_template(self) -> AnalysisTemplate:
        """Build systematic review / meta-analysis template."""
        steps = [
            TemplateStep(
                name="study_characteristics",
                description="Summarize characteristics of included studies",
                method="stats.descriptive.DescriptiveStats.study_table",
                parameters={
                    'study_id': '$study_id',
                    'characteristics': '$study_characteristics'
                },
                educational_note="Characteristics table describes included studies (design, population, intervention, outcomes)."
            ),
            TemplateStep(
                name="risk_of_bias",
                description="Assess and display risk of bias",
                method="stats.meta_analysis.MetaAnalysis.risk_of_bias_summary",
                parameters={
                    'study_id': '$study_id',
                    'domains': '$rob_domains',
                    'ratings': '$rob_ratings'
                },
                educational_note="Use appropriate tool (RoB 2 for RCTs, ROBINS-I for observational). Display summary and individual study assessments."
            ),
            TemplateStep(
                name="forest_plot_primary",
                description="Forest plot for primary outcome",
                method="stats.meta_analysis.MetaAnalysis.forest_plot",
                parameters={
                    'study_id': '$study_id',
                    'effect': '$effect_estimate',
                    'se': '$standard_error',
                    'model': 'random'
                },
                educational_note="Forest plot shows individual study effects and pooled estimate. Random effects accounts for between-study heterogeneity."
            ),
            TemplateStep(
                name="heterogeneity",
                description="Assess statistical heterogeneity",
                method="stats.meta_analysis.MetaAnalysis.heterogeneity_stats",
                parameters={
                    'study_id': '$study_id',
                    'effect': '$effect_estimate',
                    'se': '$standard_error'
                },
                depends_on=["forest_plot_primary"],
                educational_note="I² < 25% low, 25-75% moderate, >75% high heterogeneity. Consider sources if substantial."
            ),
            TemplateStep(
                name="subgroup_analysis",
                description="Pre-specified subgroup analyses",
                method="stats.meta_analysis.MetaAnalysis.subgroup_meta_analysis",
                parameters={
                    'study_id': '$study_id',
                    'effect': '$effect_estimate',
                    'se': '$standard_error',
                    'subgroup': '$subgroup_variable'
                },
                depends_on=["forest_plot_primary"],
                required=False,
                educational_note="Subgroup analyses explore heterogeneity. Test for interaction between subgroups."
            ),
            TemplateStep(
                name="publication_bias",
                description="Assess publication bias",
                method="stats.meta_analysis.MetaAnalysis.publication_bias",
                parameters={
                    'effect': '$effect_estimate',
                    'se': '$standard_error'
                },
                depends_on=["forest_plot_primary"],
                educational_note="Funnel plot asymmetry may indicate publication bias. Use Egger's test for formal assessment."
            ),
            TemplateStep(
                name="sensitivity_analysis",
                description="Sensitivity analyses (leave-one-out, high RoB exclusion)",
                method="stats.meta_analysis.MetaAnalysis.sensitivity_analysis",
                parameters={
                    'study_id': '$study_id',
                    'effect': '$effect_estimate',
                    'se': '$standard_error',
                    'rob': '$rob_overall'
                },
                depends_on=["forest_plot_primary"],
                educational_note="Sensitivity analyses assess robustness. Results should be consistent after excluding influential studies."
            ),
            TemplateStep(
                name="grade_assessment",
                description="GRADE certainty of evidence assessment",
                method="stats.meta_analysis.MetaAnalysis.grade_summary",
                parameters={
                    'domains': ['risk_of_bias', 'inconsistency', 'indirectness', 'imprecision', 'publication_bias'],
                    'ratings': '$grade_ratings'
                },
                depends_on=["forest_plot_primary", "heterogeneity", "publication_bias"],
                required=False,
                educational_note="GRADE assesses certainty of evidence (high, moderate, low, very low) considering 5 domains."
            )
        ]

        return AnalysisTemplate(
            name="meta_analysis",
            description="Systematic Review and Meta-Analysis",
            study_type="Meta-Analysis",
            steps=steps,
            required_variables={
                'study_id': "Unique study identifier",
                'effect_estimate': "Effect estimate (log OR, log RR, MD, etc.)",
                'standard_error': "Standard error of effect estimate"
            },
            optional_variables={
                'study_characteristics': "Study characteristics variables",
                'rob_domains': "Risk of bias domain names",
                'rob_ratings': "Risk of bias domain ratings",
                'rob_overall': "Overall risk of bias rating",
                'subgroup_variable': "Variable for subgroup analysis",
                'grade_ratings': "GRADE domain ratings"
            },
            reporting_guidelines="PRISMA 2020",
            references=[
                "Page MJ et al. PRISMA 2020. BMJ 2021;372:n71",
                "Higgins JPT et al. Cochrane Handbook for Systematic Reviews"
            ]
        )
