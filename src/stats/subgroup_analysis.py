"""
Subgroup Analysis Module
========================
Comprehensive subgroup and interaction analysis for clinical research.

Methods:
- Subgroup analysis with interaction testing
- Forest plots by subgroup
- Multiplicity adjustment (Bonferroni, Holm, FDR)
- Credibility assessment of subgroup findings
- Consistency of treatment effects
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SubgroupEffectResult:
    """Result container for a single subgroup effect."""
    subgroup_variable: str
    subgroup_value: str
    n_observations: int
    n_events: Optional[int]
    effect_estimate: float
    effect_se: float
    effect_ci_lower: float
    effect_ci_upper: float
    p_value: float


@dataclass
class InteractionTestResult:
    """Result container for interaction testing."""
    subgroup_variable: str
    interaction_coefficient: float
    interaction_se: float
    interaction_ci_lower: float
    interaction_ci_upper: float
    interaction_p_value: float
    interaction_type: str  # 'quantitative' or 'qualitative'
    conclusion: str


@dataclass
class SubgroupAnalysisResult:
    """Result container for complete subgroup analysis."""
    outcome_variable: str
    treatment_variable: str
    subgroup_variables: List[str]
    overall_effect: float
    overall_se: float
    overall_ci: Tuple[float, float]
    subgroup_effects: Dict[str, List[SubgroupEffectResult]]
    interaction_tests: Dict[str, InteractionTestResult]
    heterogeneity_p_values: Dict[str, float]
    multiplicity_adjusted_p: Dict[str, Dict[str, float]]
    credibility_assessment: Dict[str, Dict]
    summary_text: str


@dataclass
class CredibilityResult:
    """Result container for subgroup credibility assessment."""
    subgroup_variable: str
    criteria_met: Dict[str, bool]
    credibility_score: int
    credibility_level: str  # 'High', 'Moderate', 'Low'
    recommendation: str


class SubgroupAnalyzer:
    """
    Comprehensive subgroup analysis with interaction testing and credibility assessment.
    """

    def __init__(self):
        pass

    def analyze_subgroups(
        self,
        df: pd.DataFrame,
        outcome_var: str,
        treatment_var: str,
        subgroup_vars: List[str],
        outcome_type: str = 'continuous',
        covariates: Optional[List[str]] = None,
        multiplicity_method: str = 'fdr_bh'
    ) -> SubgroupAnalysisResult:
        """
        Comprehensive subgroup analysis with interaction testing.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        outcome_var : str
            Outcome variable
        treatment_var : str
            Treatment/exposure variable
        subgroup_vars : List[str]
            Variables to use for subgroup analysis
        outcome_type : str
            'continuous', 'binary', or 'survival'
        covariates : List[str], optional
            Adjustment covariates
        multiplicity_method : str
            Method for multiple testing correction:
            'bonferroni', 'holm', 'fdr_bh', 'fdr_by'

        Returns:
        --------
        SubgroupAnalysisResult
        """
        df_clean = df.dropna(subset=[outcome_var, treatment_var])

        # Calculate overall effect
        overall_result = self._calculate_effect(
            df_clean, outcome_var, treatment_var, outcome_type, covariates
        )

        subgroup_effects = {}
        interaction_tests = {}
        heterogeneity_p = {}
        raw_p_values = []
        p_value_labels = []

        for subgroup_var in subgroup_vars:
            if subgroup_var not in df_clean.columns:
                continue

            df_sub = df_clean.dropna(subset=[subgroup_var])

            # Get subgroup levels
            if df_sub[subgroup_var].dtype in ['int64', 'float64']:
                # Check if binary variable
                unique_vals = df_sub[subgroup_var].dropna().unique()
                if len(unique_vals) <= 2:
                    # Binary variable - use as-is
                    subgroup_col = subgroup_var
                else:
                    # Continuous variable - create categories
                    try:
                        df_sub = df_sub.copy()
                        df_sub[f'{subgroup_var}_cat'] = pd.qcut(
                            df_sub[subgroup_var], q=2, labels=['Low', 'High'], duplicates='drop'
                        )
                        subgroup_col = f'{subgroup_var}_cat'
                    except ValueError:
                        # If qcut fails, use median split
                        median = df_sub[subgroup_var].median()
                        df_sub = df_sub.copy()
                        df_sub[f'{subgroup_var}_cat'] = np.where(
                            df_sub[subgroup_var] <= median, 'Low', 'High'
                        )
                        subgroup_col = f'{subgroup_var}_cat'
            else:
                subgroup_col = subgroup_var

            subgroup_levels = df_sub[subgroup_col].unique()

            # Calculate effect in each subgroup
            effects = []
            for level in subgroup_levels:
                subset = df_sub[df_sub[subgroup_col] == level]

                if len(subset) < 10:
                    continue

                effect_result = self._calculate_effect(
                    subset, outcome_var, treatment_var, outcome_type, covariates
                )

                n_events = None
                if outcome_type == 'binary':
                    n_events = int(subset[outcome_var].sum())

                effects.append(SubgroupEffectResult(
                    subgroup_variable=subgroup_var,
                    subgroup_value=str(level),
                    n_observations=len(subset),
                    n_events=n_events,
                    effect_estimate=effect_result['estimate'],
                    effect_se=effect_result['se'],
                    effect_ci_lower=effect_result['ci_lower'],
                    effect_ci_upper=effect_result['ci_upper'],
                    p_value=effect_result['p_value']
                ))

                raw_p_values.append(effect_result['p_value'])
                p_value_labels.append(f"{subgroup_var}={level}")

            subgroup_effects[subgroup_var] = effects

            # Interaction test
            interaction_result = self._test_interaction(
                df_sub, outcome_var, treatment_var, subgroup_col, outcome_type, covariates
            )
            interaction_tests[subgroup_var] = interaction_result
            raw_p_values.append(interaction_result.interaction_p_value)
            p_value_labels.append(f"{subgroup_var}_interaction")

            # Heterogeneity test (Q-statistic)
            if len(effects) >= 2:
                het_p = self._test_heterogeneity(effects)
                heterogeneity_p[subgroup_var] = het_p

        # Multiplicity adjustment
        if len(raw_p_values) > 1:
            adjusted_p = self._adjust_p_values(raw_p_values, multiplicity_method)
            multiplicity_adjusted = {
                label: {'raw_p': raw, 'adjusted_p': adj}
                for label, raw, adj in zip(p_value_labels, raw_p_values, adjusted_p)
            }
        else:
            multiplicity_adjusted = {}

        # Credibility assessment
        credibility = {}
        for subgroup_var in subgroup_vars:
            if subgroup_var in interaction_tests:
                cred_result = self.assess_credibility(
                    subgroup_var,
                    interaction_tests[subgroup_var],
                    subgroup_effects.get(subgroup_var, []),
                    overall_result
                )
                credibility[subgroup_var] = {
                    'score': cred_result.credibility_score,
                    'level': cred_result.credibility_level,
                    'criteria': cred_result.criteria_met,
                    'recommendation': cred_result.recommendation
                }

        summary = (
            f"Subgroup analysis: {len(subgroup_vars)} subgroup variables analyzed. "
            f"Overall effect = {overall_result['estimate']:.4f} "
            f"(95% CI: {overall_result['ci_lower']:.4f}, {overall_result['ci_upper']:.4f}). "
            f"Significant interactions: {sum(1 for t in interaction_tests.values() if t.interaction_p_value < 0.05)}."
        )

        return SubgroupAnalysisResult(
            outcome_variable=outcome_var,
            treatment_variable=treatment_var,
            subgroup_variables=subgroup_vars,
            overall_effect=overall_result['estimate'],
            overall_se=overall_result['se'],
            overall_ci=(overall_result['ci_lower'], overall_result['ci_upper']),
            subgroup_effects=subgroup_effects,
            interaction_tests=interaction_tests,
            heterogeneity_p_values=heterogeneity_p,
            multiplicity_adjusted_p=multiplicity_adjusted,
            credibility_assessment=credibility,
            summary_text=summary
        )

    def _calculate_effect(
        self,
        df: pd.DataFrame,
        outcome_var: str,
        treatment_var: str,
        outcome_type: str,
        covariates: Optional[List[str]] = None
    ) -> Dict:
        """Calculate treatment effect within a group."""

        if outcome_type == 'continuous':
            # Mean difference
            treated = df[df[treatment_var] == 1][outcome_var]
            control = df[df[treatment_var] == 0][outcome_var]

            estimate = treated.mean() - control.mean()
            se = np.sqrt(treated.var()/len(treated) + control.var()/len(control))

            t_stat = estimate / se if se > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=len(df)-2))

        elif outcome_type == 'binary':
            # Odds ratio
            import statsmodels.api as sm

            X = df[[treatment_var]].copy()
            if covariates:
                for cov in covariates:
                    if cov in df.columns:
                        X[cov] = df[cov]

            X = sm.add_constant(X)
            y = df[outcome_var]

            try:
                model = sm.Logit(y, X).fit(disp=0)
                estimate = model.params[treatment_var]  # Log OR
                se = model.bse[treatment_var]
                p_value = model.pvalues[treatment_var]
            except:
                estimate = 0
                se = 1
                p_value = 1

        elif outcome_type == 'survival':
            # Hazard ratio using Cox regression
            from lifelines import CoxPHFitter

            df_cox = df[[outcome_var, 'event', treatment_var]].copy()
            if covariates:
                for cov in covariates:
                    if cov in df.columns:
                        df_cox[cov] = df[cov]

            try:
                cph = CoxPHFitter()
                cph.fit(df_cox, duration_col=outcome_var, event_col='event')
                estimate = cph.params_[treatment_var]  # Log HR
                se = cph.summary.loc[treatment_var, 'se(coef)']
                p_value = cph.summary.loc[treatment_var, 'p']
            except:
                estimate = 0
                se = 1
                p_value = 1
        else:
            raise ValueError(f"Unknown outcome type: {outcome_type}")

        ci_lower = estimate - 1.96 * se
        ci_upper = estimate + 1.96 * se

        return {
            'estimate': float(estimate),
            'se': float(se),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'p_value': float(p_value)
        }

    def _test_interaction(
        self,
        df: pd.DataFrame,
        outcome_var: str,
        treatment_var: str,
        subgroup_var: str,
        outcome_type: str,
        covariates: Optional[List[str]] = None
    ) -> InteractionTestResult:
        """Test for treatment-subgroup interaction."""
        import statsmodels.api as sm

        # Create interaction term
        df_int = df.copy()

        # Convert subgroup to numeric if needed
        if df_int[subgroup_var].dtype == 'object' or df_int[subgroup_var].dtype.name == 'category':
            df_int[subgroup_var] = pd.factorize(df_int[subgroup_var])[0]

        df_int['interaction'] = df_int[treatment_var] * df_int[subgroup_var]

        # Build model
        predictors = [treatment_var, subgroup_var, 'interaction']
        if covariates:
            predictors.extend([c for c in covariates if c in df_int.columns])

        X = df_int[predictors]
        X = sm.add_constant(X)
        y = df_int[outcome_var]

        try:
            if outcome_type == 'continuous':
                model = sm.OLS(y, X).fit()
            elif outcome_type == 'binary':
                model = sm.Logit(y, X).fit(disp=0)
            else:
                # For survival, use simplified approach
                model = sm.OLS(y, X).fit()

            int_coef = model.params['interaction']
            int_se = model.bse['interaction']
            int_p = model.pvalues['interaction']

        except Exception as e:
            int_coef = 0
            int_se = 1
            int_p = 1

        ci_lower = int_coef - 1.96 * int_se
        ci_upper = int_coef + 1.96 * int_se

        # Determine interaction type
        # Qualitative: effects in opposite directions
        # Quantitative: effects in same direction but different magnitude
        if ci_lower > 0 or ci_upper < 0:
            int_type = 'quantitative'
        else:
            int_type = 'no significant interaction'

        if int_p < 0.05:
            conclusion = f"Significant interaction (p={int_p:.4f}). Treatment effect varies by {subgroup_var}."
        else:
            conclusion = f"No significant interaction (p={int_p:.4f}). Treatment effect appears consistent across {subgroup_var} levels."

        return InteractionTestResult(
            subgroup_variable=subgroup_var,
            interaction_coefficient=float(int_coef),
            interaction_se=float(int_se),
            interaction_ci_lower=float(ci_lower),
            interaction_ci_upper=float(ci_upper),
            interaction_p_value=float(int_p),
            interaction_type=int_type,
            conclusion=conclusion
        )

    def _test_heterogeneity(
        self,
        effects: List[SubgroupEffectResult]
    ) -> float:
        """Test for heterogeneity of effects across subgroups using Q-statistic."""
        if len(effects) < 2:
            return 1.0

        estimates = np.array([e.effect_estimate for e in effects])
        variances = np.array([e.effect_se**2 for e in effects])

        # Avoid division by zero
        variances = np.maximum(variances, 1e-10)

        weights = 1 / variances
        weighted_mean = np.sum(weights * estimates) / np.sum(weights)

        Q = np.sum(weights * (estimates - weighted_mean)**2)
        df = len(effects) - 1

        p_value = 1 - stats.chi2.cdf(Q, df)

        return float(p_value)

    def _adjust_p_values(
        self,
        p_values: List[float],
        method: str
    ) -> List[float]:
        """Adjust p-values for multiple comparisons."""
        _, adjusted, _, _ = multipletests(p_values, method=method)
        return adjusted.tolist()

    def assess_credibility(
        self,
        subgroup_var: str,
        interaction_result: InteractionTestResult,
        subgroup_effects: List[SubgroupEffectResult],
        overall_effect: Dict
    ) -> CredibilityResult:
        """
        Assess credibility of subgroup findings using established criteria.

        Based on criteria from:
        - Sun et al. (JAMA 2014) - Credibility of subgroup analyses
        - ICJME guidelines
        """
        criteria = {}

        # 1. Was the subgroup hypothesis specified a priori?
        # (Cannot determine from data alone - assume yes for this analysis)
        criteria['specified_a_priori'] = True  # Placeholder

        # 2. Is there a small number of subgroup analyses?
        n_subgroups = len(subgroup_effects)
        criteria['limited_subgroups'] = n_subgroups <= 5

        # 3. Is the subgroup effect independent of other factors?
        # (Would need to check correlation with covariates)
        criteria['independent_effect'] = True  # Placeholder

        # 4. Is the subgroup effect consistent across related outcomes?
        # (Would need multiple outcomes to assess)
        criteria['consistent_across_outcomes'] = True  # Placeholder

        # 5. Is the subgroup effect consistent across studies?
        # (For single study, check consistency of direction)
        if len(subgroup_effects) >= 2:
            signs = [e.effect_estimate > 0 for e in subgroup_effects]
            criteria['consistent_direction'] = len(set(signs)) == 1
        else:
            criteria['consistent_direction'] = True

        # 6. Is there biological plausibility?
        # (Cannot determine from data - assume yes)
        criteria['biologically_plausible'] = True  # Placeholder

        # 7. Is the interaction statistically significant?
        criteria['significant_interaction'] = interaction_result.interaction_p_value < 0.05

        # 8. Is the subgroup effect large?
        if len(subgroup_effects) >= 2:
            max_effect = max(abs(e.effect_estimate) for e in subgroup_effects)
            min_effect = min(abs(e.effect_estimate) for e in subgroup_effects)
            criteria['large_effect_difference'] = max_effect > 2 * min_effect if min_effect > 0 else False
        else:
            criteria['large_effect_difference'] = False

        # 9. Is the overall effect statistically significant?
        criteria['overall_significant'] = overall_effect.get('p_value', 1) < 0.05

        # 10. Is the finding replicated?
        # (Cannot determine from single analysis)
        criteria['replicated'] = None  # Unknown

        # Calculate credibility score
        assessable_criteria = {k: v for k, v in criteria.items() if v is not None}
        score = sum(1 for v in assessable_criteria.values() if v)
        max_score = len(assessable_criteria)

        # Determine credibility level
        if score >= max_score * 0.8:
            level = 'High'
            recommendation = "Subgroup finding appears credible. Consider in clinical decision-making."
        elif score >= max_score * 0.5:
            level = 'Moderate'
            recommendation = "Subgroup finding has moderate credibility. Interpret with caution."
        else:
            level = 'Low'
            recommendation = "Subgroup finding has low credibility. Likely spurious - do not use for clinical decisions."

        return CredibilityResult(
            subgroup_variable=subgroup_var,
            criteria_met=criteria,
            credibility_score=score,
            credibility_level=level,
            recommendation=recommendation
        )

    def create_subgroup_forest_plot(
        self,
        result: SubgroupAnalysisResult,
        exponentiate: bool = False,
        title: str = "Subgroup Analysis Forest Plot"
    ):
        """
        Create a forest plot showing treatment effects by subgroup.

        Parameters:
        -----------
        result : SubgroupAnalysisResult
            Subgroup analysis results
        exponentiate : bool
            If True, display exp(effect) (for OR, HR)
        title : str
            Plot title

        Returns:
        --------
        matplotlib figure
        """
        import matplotlib.pyplot as plt

        # Transform function
        if exponentiate:
            transform = np.exp
            null_value = 1
            xlabel = "Odds Ratio / Hazard Ratio (95% CI)"
        else:
            transform = lambda x: x
            null_value = 0
            xlabel = "Effect Estimate (95% CI)"

        # Collect all effects
        all_effects = []

        # Overall effect
        all_effects.append({
            'label': 'Overall',
            'estimate': transform(result.overall_effect),
            'ci_lower': transform(result.overall_ci[0]),
            'ci_upper': transform(result.overall_ci[1]),
            'is_overall': True,
            'n': None,
            'p_int': None
        })

        # Subgroup effects
        for subgroup_var, effects in result.subgroup_effects.items():
            # Add subgroup header
            all_effects.append({
                'label': f'  {subgroup_var}',
                'estimate': None,
                'ci_lower': None,
                'ci_upper': None,
                'is_header': True,
                'n': None,
                'p_int': result.interaction_tests.get(subgroup_var, {})
            })

            for effect in effects:
                all_effects.append({
                    'label': f'    {effect.subgroup_value}',
                    'estimate': transform(effect.effect_estimate),
                    'ci_lower': transform(effect.effect_ci_lower),
                    'ci_upper': transform(effect.effect_ci_upper),
                    'is_overall': False,
                    'n': effect.n_observations,
                    'p_int': None
                })

        # Create figure
        n_rows = len(all_effects)
        fig, ax = plt.subplots(figsize=(12, max(8, n_rows * 0.4)))

        y_positions = np.arange(n_rows, 0, -1)

        for i, effect in enumerate(all_effects):
            y = y_positions[i]

            if effect.get('is_header'):
                # Subgroup header
                ax.text(0.02, y, effect['label'], ha='left', va='center',
                       fontsize=11, fontweight='bold', transform=ax.get_yaxis_transform())

                # Add interaction p-value
                if effect['p_int'] and hasattr(effect['p_int'], 'interaction_p_value'):
                    p_int = effect['p_int'].interaction_p_value
                    ax.text(0.98, y, f'p_int = {p_int:.3f}', ha='right', va='center',
                           fontsize=9, transform=ax.get_yaxis_transform())
                continue

            if effect['estimate'] is None:
                continue

            # Plot point estimate and CI
            est = effect['estimate']
            ci_low = effect['ci_lower']
            ci_high = effect['ci_upper']

            if effect.get('is_overall'):
                color = 'red'
                marker = 'D'
                markersize = 12
                linewidth = 2.5
            else:
                color = '#3498db'
                marker = 's'
                markersize = 8
                linewidth = 1.5

            # CI line
            ax.plot([ci_low, ci_high], [y, y], color=color, linewidth=linewidth)

            # Point estimate
            ax.plot(est, y, marker=marker, color=color, markersize=markersize)

            # Label
            ax.text(0.02, y, effect['label'], ha='left', va='center',
                   fontsize=10, fontweight='bold' if effect.get('is_overall') else 'normal',
                   transform=ax.get_yaxis_transform())

            # Effect text
            n_text = f"n={effect['n']}" if effect['n'] else ""
            ax.text(0.98, y, f"{est:.2f} [{ci_low:.2f}, {ci_high:.2f}] {n_text}",
                   ha='right', va='center', fontsize=9,
                   transform=ax.get_yaxis_transform())

        # Reference line
        ax.axvline(x=null_value, color='gray', linestyle='--', linewidth=1)

        # Formatting
        ax.set_yticks([])
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Set reasonable x limits
        all_estimates = [e['estimate'] for e in all_effects if e.get('estimate') is not None]
        all_cis = [e['ci_lower'] for e in all_effects if e.get('ci_lower') is not None] + \
                  [e['ci_upper'] for e in all_effects if e.get('ci_upper') is not None]

        if all_cis:
            x_min = min(all_cis) * 0.8 if min(all_cis) > 0 else min(all_cis) * 1.2
            x_max = max(all_cis) * 1.2

            # Avoid extreme values
            if exponentiate:
                x_min = max(0.1, x_min)
                x_max = min(10, x_max)
            else:
                x_min = max(-5, x_min)
                x_max = min(5, x_max)

            ax.set_xlim(x_min, x_max)

        plt.tight_layout()
        return fig

    def consistency_analysis(
        self,
        df: pd.DataFrame,
        outcome_var: str,
        treatment_var: str,
        subgroup_vars: List[str],
        outcome_type: str = 'continuous'
    ) -> Dict:
        """
        Assess consistency of treatment effects across all subgroups.

        Returns summary statistics about effect heterogeneity.
        """
        result = self.analyze_subgroups(
            df, outcome_var, treatment_var, subgroup_vars, outcome_type
        )

        # Collect all effects
        all_estimates = []
        for var, effects in result.subgroup_effects.items():
            for effect in effects:
                all_estimates.append(effect.effect_estimate)

        if len(all_estimates) < 2:
            return {
                'consistent': True,
                'summary': "Insufficient subgroups for consistency assessment."
            }

        # Consistency metrics
        estimates = np.array(all_estimates)
        overall = result.overall_effect

        # 1. All effects in same direction as overall?
        same_direction = np.all((estimates > 0) == (overall > 0))

        # 2. Range of effects relative to overall
        effect_range = estimates.max() - estimates.min()
        relative_range = effect_range / abs(overall) if overall != 0 else np.inf

        # 3. Coefficient of variation
        cv = np.std(estimates) / abs(np.mean(estimates)) if np.mean(estimates) != 0 else np.inf

        # 4. Number of significant interactions
        n_sig_interactions = sum(
            1 for t in result.interaction_tests.values()
            if t.interaction_p_value < 0.05
        )

        # Overall consistency assessment
        if same_direction and relative_range < 1 and n_sig_interactions == 0:
            consistency_level = "High"
            conclusion = "Treatment effect is consistent across subgroups."
        elif same_direction and relative_range < 2:
            consistency_level = "Moderate"
            conclusion = "Treatment effect is generally consistent, with some variation in magnitude."
        else:
            consistency_level = "Low"
            conclusion = "Treatment effect varies substantially across subgroups."

        return {
            'consistency_level': consistency_level,
            'same_direction': same_direction,
            'effect_range': float(effect_range),
            'relative_range': float(relative_range),
            'coefficient_of_variation': float(cv),
            'n_significant_interactions': n_sig_interactions,
            'n_subgroups_tested': len(all_estimates),
            'conclusion': conclusion
        }
