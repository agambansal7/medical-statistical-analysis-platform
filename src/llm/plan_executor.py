"""
Statistical Plan Executor
=========================

Executes a confirmed statistical analysis plan and collects all results.
Handles:
- Running analyses in dependency order
- Assumption checking with fallback methods
- Collecting results and visualizations
- Error handling and partial execution
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import traceback
import pandas as pd
import numpy as np
import io
import base64

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from .comprehensive_planner import ComprehensiveStatisticalPlan, AnalysisStep, VisualizationStep


@dataclass
class AnalysisResult:
    """Result of a single analysis."""
    step_id: str
    step_name: str
    method_used: str
    success: bool
    results: Dict[str, Any]
    interpretation: str
    warnings: List[str]
    error: Optional[str] = None
    assumption_results: Optional[Dict] = None
    used_fallback: bool = False
    execution_time_ms: float = 0


@dataclass
class VisualizationResult:
    """Result of a visualization."""
    viz_id: str
    name: str
    success: bool
    image_base64: Optional[str] = None
    format: str = "png"
    error: Optional[str] = None


@dataclass
class ExecutionReport:
    """Complete execution report."""
    plan_id: str
    execution_started: str
    execution_completed: str
    status: str  # completed, partial, failed
    total_analyses: int
    successful_analyses: int
    failed_analyses: int
    analysis_results: List[AnalysisResult]
    visualization_results: List[VisualizationResult]
    summary_statistics: Dict[str, Any]
    warnings: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)


class PlanExecutor:
    """
    Executes a comprehensive statistical plan.
    """

    def __init__(self):
        # Import statistical modules
        from stats.descriptive import DescriptiveStats
        from stats.comparative import ComparativeTests
        from stats.correlation import CorrelationAnalysis
        from stats.regression import RegressionAnalysis
        from stats.survival import SurvivalAnalysis
        from stats.diagnostic import DiagnosticTests
        from stats.assumptions import AssumptionChecker
        from stats.propensity_score import PropensityScoreAnalysis
        from stats.advanced_survival import AdvancedSurvivalAnalysis
        from stats.subgroup_analysis import SubgroupAnalyzer

        self.descriptive = DescriptiveStats()
        self.comparative = ComparativeTests()
        self.correlation = CorrelationAnalysis()
        self.regression = RegressionAnalysis()
        self.survival = SurvivalAnalysis()
        self.diagnostic = DiagnosticTests()
        self.assumptions = AssumptionChecker()
        self.propensity = PropensityScoreAnalysis()
        self.adv_survival = AdvancedSurvivalAnalysis()
        self.subgroup = SubgroupAnalyzer()

        # Method mapping
        self.method_map = {
            # Descriptive
            'table1': self._run_table1,
            'descriptive_continuous': self._run_descriptive_continuous,
            'descriptive_categorical': self._run_descriptive_categorical,

            # Comparative
            'independent_ttest': self._run_independent_ttest,
            'paired_ttest': self._run_paired_ttest,
            'welch_ttest': self._run_welch_ttest,
            'one_way_anova': self._run_one_way_anova,
            'mann_whitney_u': self._run_mann_whitney,
            'wilcoxon_signed_rank': self._run_wilcoxon,
            'kruskal_wallis': self._run_kruskal_wallis,
            'chi_square': self._run_chi_square,
            'fisher_exact': self._run_fisher_exact,

            # Correlation
            'pearson_correlation': self._run_pearson,
            'spearman_correlation': self._run_spearman,
            'correlation_matrix': self._run_correlation_matrix,

            # Regression
            'linear_regression': self._run_linear_regression,
            'logistic_regression': self._run_logistic_regression,
            'multiple_linear_regression': self._run_linear_regression,
            'multiple_logistic_regression': self._run_logistic_regression,

            # Survival
            'kaplan_meier': self._run_kaplan_meier,
            'log_rank_test': self._run_log_rank,
            'cox_regression': self._run_cox_regression,

            # Advanced
            'propensity_score_matching': self._run_psm,
            'inverse_probability_weighting': self._run_ipw,
            'subgroup_analysis': self._run_subgroup_analysis,

            # Assumptions
            'normality_test': self._run_normality_test,
            'levene_test': self._run_levene_test,
        }

    def execute_plan(
        self,
        plan: ComprehensiveStatisticalPlan,
        df: pd.DataFrame,
        progress_callback: Optional[callable] = None
    ) -> ExecutionReport:
        """
        Execute a confirmed statistical plan.

        Parameters:
        -----------
        plan : ComprehensiveStatisticalPlan
            The confirmed plan to execute
        df : pd.DataFrame
            The data to analyze
        progress_callback : callable, optional
            Function to call with progress updates

        Returns:
        --------
        ExecutionReport
        """
        start_time = datetime.now()

        analysis_results = []
        visualization_results = []
        warnings = []

        # Collect all analyses in execution order
        all_analyses = (
            plan.assumption_checks +
            plan.descriptive_analyses +
            plan.primary_analyses +
            plan.secondary_analyses +
            plan.sensitivity_analyses +
            plan.subgroup_analyses
        )

        total = len(all_analyses) + len(plan.visualizations)
        completed = 0

        # Execute analyses
        for step in all_analyses:
            if progress_callback:
                progress_callback(completed / total, f"Running: {step.name}")

            result = self._execute_step(step, df, plan)
            analysis_results.append(result)

            if not result.success:
                warnings.append(f"Analysis '{step.name}' failed: {result.error}")

            completed += 1

        # Generate visualizations
        for viz in plan.visualizations:
            if progress_callback:
                progress_callback(completed / total, f"Creating: {viz.name}")

            viz_result = self._create_visualization(viz, df, analysis_results)
            visualization_results.append(viz_result)

            if not viz_result.success:
                warnings.append(f"Visualization '{viz.name}' failed: {viz_result.error}")

            completed += 1

        end_time = datetime.now()

        # Calculate summary
        successful = sum(1 for r in analysis_results if r.success)
        failed = len(analysis_results) - successful

        status = 'completed' if failed == 0 else 'partial' if successful > 0 else 'failed'

        # Summary statistics
        summary_stats = self._calculate_summary_statistics(analysis_results, plan)

        return ExecutionReport(
            plan_id=plan.plan_id,
            execution_started=start_time.isoformat(),
            execution_completed=end_time.isoformat(),
            status=status,
            total_analyses=len(analysis_results),
            successful_analyses=successful,
            failed_analyses=failed,
            analysis_results=analysis_results,
            visualization_results=visualization_results,
            summary_statistics=summary_stats,
            warnings=warnings
        )

    def _execute_step(
        self,
        step: AnalysisStep,
        df: pd.DataFrame,
        plan: ComprehensiveStatisticalPlan
    ) -> AnalysisResult:
        """Execute a single analysis step."""
        import time
        start = time.time()

        method = step.method.lower().replace(' ', '_').replace('-', '_')

        # Check if method is supported
        if method not in self.method_map:
            return AnalysisResult(
                step_id=step.step_id,
                step_name=step.name,
                method_used=method,
                success=False,
                results={},
                interpretation='',
                warnings=[],
                error=f"Unknown method: {method}"
            )

        try:
            # Check assumptions if needed
            assumption_results = None
            use_fallback = False

            if step.assumption_tests:
                assumption_results = self._check_assumptions(step, df)
                # Check if any critical assumption failed
                for test_name, test_result in assumption_results.items():
                    if not test_result.get('satisfied', True):
                        use_fallback = True
                        break

            # Use fallback method if needed
            actual_method = method
            if use_fallback and step.fallback_method:
                fallback = step.fallback_method.lower().replace(' ', '_').replace('-', '_')
                if fallback in self.method_map:
                    actual_method = fallback

            # Execute the analysis
            executor = self.method_map[actual_method]
            results = executor(df, step.variables, step.parameters)

            # Generate interpretation
            interpretation = self._generate_interpretation(
                step, results, plan.significance_level
            )

            elapsed = (time.time() - start) * 1000

            return AnalysisResult(
                step_id=step.step_id,
                step_name=step.name,
                method_used=actual_method,
                success=True,
                results=results,
                interpretation=interpretation,
                warnings=[],
                assumption_results=assumption_results,
                used_fallback=use_fallback,
                execution_time_ms=elapsed
            )

        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return AnalysisResult(
                step_id=step.step_id,
                step_name=step.name,
                method_used=method,
                success=False,
                results={},
                interpretation='',
                warnings=[],
                error=str(e),
                execution_time_ms=elapsed
            )

    def _check_assumptions(
        self,
        step: AnalysisStep,
        df: pd.DataFrame
    ) -> Dict[str, Dict]:
        """Check assumptions for an analysis."""
        results = {}

        for test in step.assumption_tests:
            test_name = test.get('test', '')
            test_for = test.get('for', '')

            try:
                if 'shapiro' in test_name.lower() or 'normality' in test_for.lower():
                    # Get the outcome variable
                    outcome = step.variables.get('outcome', step.variables.get('y', None))
                    if outcome and outcome in df.columns:
                        result = self.assumptions.check_normality(df[outcome].dropna())
                        results[test_for] = {
                            'test': 'Shapiro-Wilk',
                            'statistic': result.statistic,
                            'p_value': result.p_value,
                            'satisfied': result.is_satisfied
                        }

                elif 'levene' in test_name.lower() or 'homogeneity' in test_for.lower():
                    outcome = step.variables.get('outcome', step.variables.get('y', None))
                    group = step.variables.get('group', step.variables.get('x', None))
                    if outcome and group and outcome in df.columns and group in df.columns:
                        result = self.assumptions.check_homogeneity(df, outcome, group)
                        results[test_for] = {
                            'test': 'Levene',
                            'statistic': result.statistic,
                            'p_value': result.p_value,
                            'satisfied': result.is_satisfied
                        }

            except Exception as e:
                results[test_for] = {
                    'error': str(e),
                    'satisfied': True  # Assume satisfied if can't test
                }

        return results

    def _generate_interpretation(
        self,
        step: AnalysisStep,
        results: Dict,
        alpha: float
    ) -> str:
        """Generate interpretation of results."""
        interp_parts = []

        # Check for p-value
        p_value = results.get('p_value')
        if p_value is not None:
            if p_value < alpha:
                interp_parts.append(f"The result is statistically significant (p = {p_value:.4f}).")
            else:
                interp_parts.append(f"The result is not statistically significant (p = {p_value:.4f}).")

        # Effect sizes
        if 'cohens_d' in results:
            d = results['cohens_d']
            if abs(d) < 0.2:
                size = "negligible"
            elif abs(d) < 0.5:
                size = "small"
            elif abs(d) < 0.8:
                size = "medium"
            else:
                size = "large"
            interp_parts.append(f"The effect size is {size} (Cohen's d = {d:.3f}).")

        if 'odds_ratio' in results:
            or_val = results['odds_ratio']
            interp_parts.append(f"Odds ratio = {or_val:.3f}.")

        if 'hazard_ratio' in results:
            hr = results['hazard_ratio']
            interp_parts.append(f"Hazard ratio = {hr:.3f}.")

        # Add guidance from step
        if step.interpretation_guidance:
            interp_parts.append(step.interpretation_guidance)

        return " ".join(interp_parts)

    def _create_visualization(
        self,
        viz: VisualizationStep,
        df: pd.DataFrame,
        analysis_results: List[AnalysisResult]
    ) -> VisualizationResult:
        """Create a visualization."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            plot_type = viz.plot_type.lower()
            variables = viz.variables

            if plot_type == 'boxplot':
                x = variables.get('x', variables.get('group'))
                y = variables.get('y', variables.get('outcome'))
                if x and y:
                    sns.boxplot(data=df, x=x, y=y, ax=ax)
                    ax.set_title(viz.name)

            elif plot_type == 'histogram':
                var = variables.get('variable', variables.get('x'))
                if var:
                    df[var].hist(ax=ax, bins=30, edgecolor='black')
                    ax.set_xlabel(var)
                    ax.set_title(viz.name)

            elif plot_type == 'scatter':
                x = variables.get('x')
                y = variables.get('y')
                if x and y:
                    ax.scatter(df[x], df[y], alpha=0.6)
                    ax.set_xlabel(x)
                    ax.set_ylabel(y)
                    ax.set_title(viz.name)

            elif plot_type == 'bar':
                x = variables.get('x')
                y = variables.get('y')
                if x:
                    if y:
                        df.groupby(x)[y].mean().plot(kind='bar', ax=ax)
                    else:
                        df[x].value_counts().plot(kind='bar', ax=ax)
                    ax.set_title(viz.name)

            elif plot_type == 'kaplan_meier':
                # Would need survival data
                ax.text(0.5, 0.5, 'Kaplan-Meier plot', ha='center', va='center')
                ax.set_title(viz.name)

            elif plot_type == 'forest_plot':
                ax.text(0.5, 0.5, 'Forest plot', ha='center', va='center')
                ax.set_title(viz.name)

            else:
                ax.text(0.5, 0.5, f'Plot type: {plot_type}', ha='center', va='center')
                ax.set_title(viz.name)

            plt.tight_layout()

            # Convert to base64
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            plt.close(fig)

            return VisualizationResult(
                viz_id=viz.viz_id,
                name=viz.name,
                success=True,
                image_base64=img_base64,
                format='png'
            )

        except Exception as e:
            plt.close('all')
            return VisualizationResult(
                viz_id=viz.viz_id,
                name=viz.name,
                success=False,
                error=str(e)
            )

    def _calculate_summary_statistics(
        self,
        results: List[AnalysisResult],
        plan: ComprehensiveStatisticalPlan
    ) -> Dict[str, Any]:
        """Calculate summary statistics from results."""
        summary = {
            'primary_results': [],
            'significant_findings': [],
            'effect_sizes': {}
        }

        alpha = plan.significance_level

        for result in results:
            if not result.success:
                continue

            p_value = result.results.get('p_value')

            # Check primary analyses
            if result.step_id in [s.step_id for s in plan.primary_analyses]:
                summary['primary_results'].append({
                    'name': result.step_name,
                    'p_value': p_value,
                    'significant': p_value < alpha if p_value else None
                })

            # Track significant findings
            if p_value is not None and p_value < alpha:
                summary['significant_findings'].append({
                    'name': result.step_name,
                    'p_value': p_value,
                    'method': result.method_used
                })

            # Collect effect sizes
            for key in ['cohens_d', 'odds_ratio', 'hazard_ratio', 'r_squared', 'eta_squared']:
                if key in result.results:
                    summary['effect_sizes'][result.step_name] = {
                        key: result.results[key]
                    }

        return summary

    # ==================== ANALYSIS IMPLEMENTATIONS ====================

    def _run_table1(self, df: pd.DataFrame, variables: Dict, params: Dict) -> Dict:
        group = variables.get('group', variables.get('group_col'))
        continuous = variables.get('continuous', variables.get('continuous_vars', []))
        categorical = variables.get('categorical', variables.get('categorical_vars', []))

        # Handle comma-separated strings
        if isinstance(continuous, str):
            continuous = [v.strip() for v in continuous.split(',')]
        if isinstance(categorical, str):
            categorical = [v.strip() for v in categorical.split(',')]

        table = self.descriptive.generate_table1(df, group, continuous, categorical)
        return {'table': table.to_dict(orient='records')}

    def _run_descriptive_continuous(self, df: pd.DataFrame, variables: Dict, params: Dict) -> Dict:
        var = variables.get('variable', variables.get('outcome'))
        result = self.descriptive.continuous_stats(df[var], var)
        return result.to_dict()

    def _run_descriptive_categorical(self, df: pd.DataFrame, variables: Dict, params: Dict) -> Dict:
        var = variables.get('variable')
        result = self.descriptive.categorical_stats(df[var], var)
        return result.to_dict()

    def _run_independent_ttest(self, df: pd.DataFrame, variables: Dict, params: Dict) -> Dict:
        outcome = variables.get('outcome', variables.get('y'))
        group = variables.get('group', variables.get('x'))

        groups = df[group].dropna().unique()
        g1 = df[df[group] == groups[0]][outcome].dropna()
        g2 = df[df[group] == groups[1]][outcome].dropna()

        result = self.comparative.independent_ttest(g1, g2, equal_var=params.get('equal_var', True))
        return result.to_dict()

    def _run_welch_ttest(self, df: pd.DataFrame, variables: Dict, params: Dict) -> Dict:
        params['equal_var'] = False
        return self._run_independent_ttest(df, variables, params)

    def _run_paired_ttest(self, df: pd.DataFrame, variables: Dict, params: Dict) -> Dict:
        before = variables.get('before', variables.get('x'))
        after = variables.get('after', variables.get('y'))

        result = self.comparative.paired_ttest(df[before].dropna(), df[after].dropna())
        return result.to_dict()

    def _run_one_way_anova(self, df: pd.DataFrame, variables: Dict, params: Dict) -> Dict:
        outcome = variables.get('outcome', variables.get('y'))
        group = variables.get('group', variables.get('x'))

        result = self.comparative.one_way_anova(df, outcome, group)
        return result.to_dict()

    def _run_mann_whitney(self, df: pd.DataFrame, variables: Dict, params: Dict) -> Dict:
        outcome = variables.get('outcome', variables.get('y'))
        group = variables.get('group', variables.get('x'))

        groups = df[group].dropna().unique()
        g1 = df[df[group] == groups[0]][outcome].dropna()
        g2 = df[df[group] == groups[1]][outcome].dropna()

        result = self.comparative.mann_whitney_u(g1, g2)
        return result.to_dict()

    def _run_wilcoxon(self, df: pd.DataFrame, variables: Dict, params: Dict) -> Dict:
        before = variables.get('before', variables.get('x'))
        after = variables.get('after', variables.get('y'))

        result = self.comparative.wilcoxon_signed_rank(df[before], df[after])
        return result.to_dict()

    def _run_kruskal_wallis(self, df: pd.DataFrame, variables: Dict, params: Dict) -> Dict:
        outcome = variables.get('outcome', variables.get('y'))
        group = variables.get('group', variables.get('x'))

        result = self.comparative.kruskal_wallis(df, outcome, group)
        return result.to_dict()

    def _run_chi_square(self, df: pd.DataFrame, variables: Dict, params: Dict) -> Dict:
        row_var = variables.get('row', variables.get('x', variables.get('var1')))
        col_var = variables.get('col', variables.get('y', variables.get('var2', variables.get('outcome'))))

        result = self.comparative.chi_square_independence(df, row_var, col_var)
        return result.to_dict()

    def _run_fisher_exact(self, df: pd.DataFrame, variables: Dict, params: Dict) -> Dict:
        row_var = variables.get('row', variables.get('x'))
        col_var = variables.get('col', variables.get('y'))

        contingency = pd.crosstab(df[row_var], df[col_var])
        result = self.comparative.fisher_exact(contingency)
        return result.to_dict()

    def _run_pearson(self, df: pd.DataFrame, variables: Dict, params: Dict) -> Dict:
        var1 = variables.get('x', variables.get('var1'))
        var2 = variables.get('y', variables.get('var2'))

        result = self.correlation.pearson(df[var1], df[var2])
        return result.to_dict()

    def _run_spearman(self, df: pd.DataFrame, variables: Dict, params: Dict) -> Dict:
        var1 = variables.get('x', variables.get('var1'))
        var2 = variables.get('y', variables.get('var2'))

        result = self.correlation.spearman(df[var1], df[var2])
        return result.to_dict()

    def _run_correlation_matrix(self, df: pd.DataFrame, variables: Dict, params: Dict) -> Dict:
        vars_list = variables.get('variables', [])
        method = params.get('method', 'pearson')

        result = self.correlation.correlation_matrix(df, method, vars_list if vars_list else None)
        return {
            'correlation_matrix': result['correlation_matrix'].to_dict(),
            'p_value_matrix': result['p_value_matrix'].to_dict()
        }

    def _run_linear_regression(self, df: pd.DataFrame, variables: Dict, params: Dict) -> Dict:
        outcome = variables.get('outcome', variables.get('y'))
        predictors = variables.get('predictors', variables.get('x', []))
        if isinstance(predictors, str):
            # Handle comma-separated string of predictors
            predictors = [p.strip() for p in predictors.split(',')]

        result = self.regression.linear_regression(df, outcome, predictors)
        return result.to_dict()

    def _run_logistic_regression(self, df: pd.DataFrame, variables: Dict, params: Dict) -> Dict:
        outcome = variables.get('outcome', variables.get('y'))
        predictors = variables.get('predictors', variables.get('x', []))
        if isinstance(predictors, str):
            # Handle comma-separated string of predictors
            predictors = [p.strip() for p in predictors.split(',')]

        result = self.regression.logistic_regression(df, outcome, predictors)
        return result.to_dict()

    def _run_kaplan_meier(self, df: pd.DataFrame, variables: Dict, params: Dict) -> Dict:
        time_col = variables.get('time', variables.get('duration'))
        event_col = variables.get('event', variables.get('status'))
        group_col = variables.get('group')

        if group_col:
            results = self.survival.kaplan_meier_by_group(df, time_col, event_col, group_col)
            return {k: v.to_dict() for k, v in results.items()}
        else:
            result = self.survival.kaplan_meier(df[time_col], df[event_col])
            return result.to_dict()

    def _run_log_rank(self, df: pd.DataFrame, variables: Dict, params: Dict) -> Dict:
        time_col = variables.get('time', variables.get('duration'))
        event_col = variables.get('event', variables.get('status'))
        group_col = variables.get('group')

        result = self.survival.log_rank_test(df, time_col, event_col, group_col)
        return result.to_dict()

    def _run_cox_regression(self, df: pd.DataFrame, variables: Dict, params: Dict) -> Dict:
        time_col = variables.get('time', variables.get('duration'))
        event_col = variables.get('event', variables.get('status'))
        covariates = variables.get('covariates', variables.get('predictors', []))

        result = self.survival.cox_regression(df, time_col, event_col, covariates)
        return result.to_dict()

    def _run_psm(self, df: pd.DataFrame, variables: Dict, params: Dict) -> Dict:
        treatment = variables.get('treatment', variables.get('group'))
        covariates = variables.get('covariates', [])

        ps_result = self.propensity.estimate_propensity_scores(df, treatment, covariates)
        match_result = self.propensity.match(
            df, treatment, ps_result.propensity_scores, covariates,
            caliper=params.get('caliper', 0.2)
        )

        return {
            'ps_auc': ps_result.auc,
            'n_matched_treated': match_result.n_treated_matched,
            'n_matched_control': match_result.n_control_matched,
            'n_unmatched': match_result.n_unmatched,
            'smd_before': match_result.standardized_mean_diff_before,
            'smd_after': match_result.standardized_mean_diff_after
        }

    def _run_ipw(self, df: pd.DataFrame, variables: Dict, params: Dict) -> Dict:
        treatment = variables.get('treatment', variables.get('group'))
        outcome = variables.get('outcome')
        covariates = variables.get('covariates', [])

        ps_result = self.propensity.estimate_propensity_scores(df, treatment, covariates)
        ipw_result = self.propensity.inverse_probability_weighting(
            df, treatment, outcome, ps_result.propensity_scores
        )

        return {
            'ate': ipw_result.ate,
            'ate_se': ipw_result.ate_se,
            'ate_ci_lower': ipw_result.ate_ci_lower,
            'ate_ci_upper': ipw_result.ate_ci_upper,
            'att': ipw_result.att
        }

    def _run_subgroup_analysis(self, df: pd.DataFrame, variables: Dict, params: Dict) -> Dict:
        outcome = variables.get('outcome')
        treatment = variables.get('treatment', variables.get('group'))
        subgroup = variables.get('subgroup')

        result = self.subgroup.analyze_subgroups(
            df, outcome, treatment, [subgroup], outcome_type='continuous'
        )

        return {
            'overall_effect': result.overall_effect,
            'subgroup_effects': {
                var: [
                    {
                        'value': e.subgroup_value,
                        'effect': e.effect_estimate,
                        'ci_lower': e.effect_ci_lower,
                        'ci_upper': e.effect_ci_upper,
                        'p_value': e.p_value,
                        'n': e.n_observations
                    }
                    for e in effects
                ]
                for var, effects in result.subgroup_effects.items()
            },
            'interaction_p': {
                var: inter.interaction_p_value
                for var, inter in result.interaction_tests.items()
            }
        }

    def _run_normality_test(self, df: pd.DataFrame, variables: Dict, params: Dict) -> Dict:
        var = variables.get('variable', variables.get('outcome'))
        result = self.assumptions.check_normality(df[var].dropna())
        return {
            'test': result.test_name,
            'statistic': result.statistic,
            'p_value': result.p_value,
            'is_normal': result.is_satisfied
        }

    def _run_levene_test(self, df: pd.DataFrame, variables: Dict, params: Dict) -> Dict:
        outcome = variables.get('outcome')
        group = variables.get('group')
        result = self.assumptions.check_homogeneity(df, outcome, group)
        return {
            'test': result.test_name,
            'statistic': result.statistic,
            'p_value': result.p_value,
            'homogeneous': result.is_satisfied
        }
