"""Causal inference methods module.

Provides methods for causal effect estimation:
- Instrumental Variables (IV) / Two-Stage Least Squares (2SLS)
- Regression Discontinuity Design (RDD)
- Difference-in-Differences (DiD)
- Synthetic Control Method
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
import warnings


@dataclass
class IVResult:
    """Result of instrumental variables analysis."""
    model_type: str = "Instrumental Variables (2SLS)"
    n_observations: int = 0
    n_instruments: int = 0

    # First stage results
    first_stage_f_statistic: Optional[float] = None
    first_stage_r_squared: Optional[float] = None
    weak_instrument_warning: bool = False

    # Second stage (causal effect) results
    treatment_effect: Optional[float] = None
    std_error: Optional[float] = None
    t_statistic: Optional[float] = None
    p_value: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None

    # All coefficients
    coefficients: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Diagnostics
    sargan_statistic: Optional[float] = None  # Over-identification test
    sargan_pvalue: Optional[float] = None
    durbin_wu_hausman: Optional[float] = None  # Endogeneity test
    dwh_pvalue: Optional[float] = None

    # Interpretation
    summary_text: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    educational_notes: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class RDDResult:
    """Result of regression discontinuity design analysis."""
    model_type: str = "Regression Discontinuity Design"
    design_type: str = "sharp"  # or "fuzzy"
    n_observations: int = 0
    n_treated: int = 0
    n_control: int = 0
    cutoff: float = 0.0
    bandwidth: Optional[float] = None

    # Treatment effect at cutoff
    treatment_effect: Optional[float] = None
    std_error: Optional[float] = None
    t_statistic: Optional[float] = None
    p_value: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None

    # Robustness
    effect_half_bandwidth: Optional[float] = None
    effect_double_bandwidth: Optional[float] = None
    mccrary_test: Optional[float] = None  # Manipulation test

    # Interpretation
    summary_text: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    educational_notes: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class DiDResult:
    """Result of difference-in-differences analysis."""
    model_type: str = "Difference-in-Differences"
    n_observations: int = 0
    n_treated: int = 0
    n_control: int = 0
    n_periods: int = 0

    # DiD estimate
    did_estimate: Optional[float] = None
    std_error: Optional[float] = None
    t_statistic: Optional[float] = None
    p_value: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None

    # Group means
    treated_before: Optional[float] = None
    treated_after: Optional[float] = None
    control_before: Optional[float] = None
    control_after: Optional[float] = None

    # Parallel trends test
    parallel_trends_pvalue: Optional[float] = None
    parallel_trends_satisfied: Optional[bool] = None

    # All coefficients
    coefficients: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Interpretation
    summary_text: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    educational_notes: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class CausalInference:
    """Causal inference methods."""

    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level
        self.confidence_level = 1 - significance_level

    def instrumental_variables(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        instruments: List[str],
        covariates: Optional[List[str]] = None
    ) -> IVResult:
        """Two-Stage Least Squares (2SLS) instrumental variables estimation.

        Args:
            data: DataFrame with the data
            outcome: Dependent variable name
            treatment: Endogenous treatment variable
            instruments: List of instrumental variable names
            covariates: List of exogenous control variables

        Returns:
            IVResult object
        """
        covariates = covariates or []

        # Prepare data
        all_vars = [outcome, treatment] + instruments + covariates
        clean_data = data[all_vars].dropna()

        n_obs = len(clean_data)
        n_inst = len(instruments)

        if n_obs < len(instruments) + len(covariates) + 5:
            return self._insufficient_iv_data()

        try:
            # Prepare matrices
            y = clean_data[outcome].values
            # Endogenous regressor (treatment) plus covariates
            X = clean_data[[treatment] + covariates].values
            X = sm.add_constant(X)

            # Instruments plus covariates
            Z = clean_data[instruments + covariates].values
            Z = sm.add_constant(Z)

            # === First Stage ===
            # Regress treatment on instruments
            first_stage = sm.OLS(clean_data[treatment], Z).fit()
            first_stage_f = first_stage.fvalue
            first_stage_r2 = first_stage.rsquared

            # Check for weak instruments (F < 10 rule of thumb)
            weak_instruments = first_stage_f < 10

            # === Second Stage (2SLS) ===
            # Use statsmodels IV2SLS
            model = IV2SLS(y, X, Z)
            result = model.fit()

            # Extract treatment effect
            treatment_idx = 1  # After constant
            treatment_effect = float(result.params[treatment_idx])
            std_error = float(result.bse[treatment_idx])
            t_stat = treatment_effect / std_error if std_error > 0 else 0
            p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), n_obs - X.shape[1])))
            ci_lower = treatment_effect - 1.96 * std_error
            ci_upper = treatment_effect + 1.96 * std_error

            # Build coefficients dict
            coeffs = {}
            var_names = ['const', treatment] + covariates
            for i, var in enumerate(var_names):
                coeffs[var] = {
                    'coefficient': float(result.params[i]),
                    'std_error': float(result.bse[i]),
                    'significant': abs(result.params[i] / result.bse[i]) > 1.96
                }

            # Sargan over-identification test (if instruments > endogenous vars)
            sargan_stat = None
            sargan_p = None
            if n_inst > 1:
                # Calculate Sargan statistic
                residuals = y - X @ result.params
                sargan_stat = float(n_obs * sm.OLS(residuals, Z).fit().rsquared)
                sargan_df = n_inst - 1  # Degrees of freedom
                sargan_p = float(1 - stats.chi2.cdf(sargan_stat, sargan_df))

            # Durbin-Wu-Hausman endogeneity test
            ols_result = sm.OLS(y, X).fit()
            dwh_stat = float(((result.params[treatment_idx] - ols_result.params[treatment_idx]) ** 2) /
                            (result.bse[treatment_idx] ** 2 - ols_result.bse[treatment_idx] ** 2))
            dwh_p = float(1 - stats.chi2.cdf(abs(dwh_stat), 1)) if dwh_stat > 0 else None

            # Generate summary
            summary = self._generate_iv_summary(
                treatment_effect, p_value, treatment,
                first_stage_f, weak_instruments
            )

            # Educational notes
            edu_notes = {
                'what_is_iv': 'Instrumental Variables (IV) addresses endogeneity bias when treatment is correlated with unobserved confounders.',
                'instrument_requirements': 'A valid instrument must: (1) Be correlated with treatment (relevance), (2) Not directly affect outcome (exclusion), (3) Not share confounders with outcome (exogeneity).',
                'first_stage_f': f'First-stage F={first_stage_f:.1f}. {"Weak instruments warning (F<10)" if weak_instruments else "Instruments appear strong"}',
                'interpreting_effect': 'The IV estimate is the Local Average Treatment Effect (LATE) - the effect for compliers who change treatment due to the instrument.',
                'sargan_test': f'{"Over-identification test suggests instruments may be invalid (p=" + str(round(sargan_p, 3)) + ")" if sargan_p and sargan_p < 0.05 else "Over-identification test passed" if sargan_p else "Not applicable (just-identified)"}',
                'when_to_use': 'Use IV when randomization is not possible and treatment is endogenous (e.g., confounded by unobservables).'
            }

            warnings_list = []
            if weak_instruments:
                warnings_list.append("Weak instruments detected (F < 10). Estimates may be biased.")
            if sargan_p and sargan_p < 0.05:
                warnings_list.append("Sargan test suggests instruments may be invalid.")

            return IVResult(
                n_observations=n_obs,
                n_instruments=n_inst,
                first_stage_f_statistic=float(first_stage_f),
                first_stage_r_squared=float(first_stage_r2),
                weak_instrument_warning=weak_instruments,
                treatment_effect=treatment_effect,
                std_error=std_error,
                t_statistic=t_stat,
                p_value=p_value,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                coefficients=coeffs,
                sargan_statistic=sargan_stat,
                sargan_pvalue=sargan_p,
                durbin_wu_hausman=dwh_stat if dwh_stat > 0 else None,
                dwh_pvalue=dwh_p,
                summary_text=summary,
                warnings=warnings_list,
                educational_notes=edu_notes
            )

        except Exception as e:
            result = self._insufficient_iv_data()
            result.warnings = [str(e)]
            return result

    def regression_discontinuity(
        self,
        data: pd.DataFrame,
        outcome: str,
        running_variable: str,
        cutoff: float,
        treatment: Optional[str] = None,
        bandwidth: Optional[float] = None,
        polynomial_order: int = 1,
        covariates: Optional[List[str]] = None
    ) -> RDDResult:
        """Regression Discontinuity Design analysis.

        Args:
            data: DataFrame with the data
            outcome: Dependent variable name
            running_variable: Variable determining treatment assignment
            cutoff: Cutoff value for treatment assignment
            treatment: Treatment variable (for fuzzy RDD). If None, sharp RDD assumed.
            bandwidth: Bandwidth around cutoff. If None, uses IK optimal bandwidth.
            polynomial_order: Order of polynomial for local regression
            covariates: Additional covariates

        Returns:
            RDDResult object
        """
        covariates = covariates or []

        # Determine design type
        design_type = "fuzzy" if treatment else "sharp"

        # Prepare data
        all_vars = [outcome, running_variable] + covariates
        if treatment:
            all_vars.append(treatment)
        clean_data = data[all_vars].dropna().copy()

        # Calculate optimal bandwidth if not provided (IK method approximation)
        if bandwidth is None:
            bandwidth = self._ik_bandwidth(clean_data, outcome, running_variable, cutoff)

        # Subset to bandwidth
        rdd_data = clean_data[
            (clean_data[running_variable] >= cutoff - bandwidth) &
            (clean_data[running_variable] <= cutoff + bandwidth)
        ].copy()

        n_obs = len(rdd_data)
        if n_obs < 20:
            return self._insufficient_rdd_data()

        # Create treatment indicator for sharp RDD
        if design_type == "sharp":
            rdd_data['_treatment'] = (rdd_data[running_variable] >= cutoff).astype(int)
            treatment_var = '_treatment'
        else:
            treatment_var = treatment

        n_treated = (rdd_data[treatment_var] == 1).sum()
        n_control = (rdd_data[treatment_var] == 0).sum()

        # Center running variable at cutoff
        rdd_data['_centered'] = rdd_data[running_variable] - cutoff

        try:
            # Build regression formula
            # Include polynomial terms and interaction with treatment
            terms = ['_treatment']
            for p in range(1, polynomial_order + 1):
                rdd_data[f'_centered_{p}'] = rdd_data['_centered'] ** p
                terms.append(f'_centered_{p}')
                terms.append(f'_treatment:_centered_{p}')

            terms.extend(covariates)

            # Fit model
            formula = f"{outcome} ~ {' + '.join(terms)}"
            model = sm.OLS.from_formula(formula, data=rdd_data).fit()

            # Treatment effect is coefficient on treatment
            treatment_effect = float(model.params['_treatment'])
            std_error = float(model.bse['_treatment'])
            t_stat = float(model.tvalues['_treatment'])
            p_value = float(model.pvalues['_treatment'])
            ci_lower = float(model.conf_int().loc['_treatment', 0])
            ci_upper = float(model.conf_int().loc['_treatment', 1])

            # Robustness: effects at half and double bandwidth
            effect_half = self._rdd_effect_at_bandwidth(
                clean_data, outcome, running_variable, cutoff,
                bandwidth / 2, polynomial_order, treatment_var if treatment else None
            )
            effect_double = self._rdd_effect_at_bandwidth(
                clean_data, outcome, running_variable, cutoff,
                bandwidth * 2, polynomial_order, treatment_var if treatment else None
            )

            # McCrary manipulation test (density test)
            mccrary = self._mccrary_test(clean_data, running_variable, cutoff)

            # Generate summary
            summary = self._generate_rdd_summary(
                treatment_effect, p_value, cutoff, bandwidth, design_type, mccrary
            )

            # Educational notes
            edu_notes = {
                'what_is_rdd': 'RDD exploits a known cutoff rule to estimate causal effects by comparing units just above and below the threshold.',
                'sharp_vs_fuzzy': f'This is a {"sharp" if design_type == "sharp" else "fuzzy"} RDD. {"Treatment is deterministically assigned at cutoff." if design_type == "sharp" else "Treatment probability changes at cutoff but is not deterministic."}',
                'bandwidth_choice': f'Using bandwidth of {bandwidth:.2f}. Narrower bandwidths reduce bias but increase variance.',
                'key_assumption': 'Key assumption: Units cannot precisely manipulate their running variable value around the cutoff.',
                'mccrary_test': f'{"McCrary test suggests possible manipulation (p<0.05). Validity threatened." if mccrary and mccrary < 0.05 else "No evidence of manipulation at cutoff."}',
                'when_to_use': 'Use RDD when treatment is assigned based on a cutoff rule (e.g., test scores, age thresholds, income limits).'
            }

            warnings_list = []
            if mccrary and mccrary < 0.05:
                warnings_list.append("McCrary test suggests possible manipulation at cutoff.")
            if n_treated < 10 or n_control < 10:
                warnings_list.append("Small sample size near cutoff may reduce precision.")

            return RDDResult(
                design_type=design_type,
                n_observations=n_obs,
                n_treated=n_treated,
                n_control=n_control,
                cutoff=cutoff,
                bandwidth=bandwidth,
                treatment_effect=treatment_effect,
                std_error=std_error,
                t_statistic=t_stat,
                p_value=p_value,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                effect_half_bandwidth=effect_half,
                effect_double_bandwidth=effect_double,
                mccrary_test=mccrary,
                summary_text=summary,
                warnings=warnings_list,
                educational_notes=edu_notes
            )

        except Exception as e:
            result = self._insufficient_rdd_data()
            result.warnings = [str(e)]
            return result

    def difference_in_differences(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment_group: str,
        post_period: str,
        covariates: Optional[List[str]] = None,
        cluster_var: Optional[str] = None
    ) -> DiDResult:
        """Difference-in-Differences analysis.

        Args:
            data: DataFrame with the data (panel or repeated cross-section)
            outcome: Dependent variable name
            treatment_group: Binary indicator for treatment group (1=treated, 0=control)
            post_period: Binary indicator for post-treatment period (1=post, 0=pre)
            covariates: Additional control variables
            cluster_var: Variable to cluster standard errors on

        Returns:
            DiDResult object
        """
        covariates = covariates or []

        # Prepare data
        all_vars = [outcome, treatment_group, post_period] + covariates
        if cluster_var:
            all_vars.append(cluster_var)
        clean_data = data[all_vars].dropna().copy()

        n_obs = len(clean_data)
        n_treated = (clean_data[treatment_group] == 1).sum()
        n_control = (clean_data[treatment_group] == 0).sum()

        if n_obs < 20:
            return self._insufficient_did_data()

        # Calculate group means
        treated_before = clean_data[(clean_data[treatment_group] == 1) & (clean_data[post_period] == 0)][outcome].mean()
        treated_after = clean_data[(clean_data[treatment_group] == 1) & (clean_data[post_period] == 1)][outcome].mean()
        control_before = clean_data[(clean_data[treatment_group] == 0) & (clean_data[post_period] == 0)][outcome].mean()
        control_after = clean_data[(clean_data[treatment_group] == 0) & (clean_data[post_period] == 1)][outcome].mean()

        # Simple DiD estimate
        did_simple = (treated_after - treated_before) - (control_after - control_before)

        try:
            # Create interaction term
            clean_data['_did_interaction'] = clean_data[treatment_group] * clean_data[post_period]

            # Build regression formula
            terms = [treatment_group, post_period, '_did_interaction'] + covariates
            formula = f"{outcome} ~ {' + '.join(terms)}"

            # Fit model
            model = sm.OLS.from_formula(formula, data=clean_data).fit()

            # Use clustered standard errors if specified
            if cluster_var:
                model = model.get_robustcov_results(
                    cov_type='cluster',
                    groups=clean_data[cluster_var]
                )

            # DiD estimate is the interaction coefficient
            did_estimate = float(model.params['_did_interaction'])
            std_error = float(model.bse['_did_interaction'])
            t_stat = float(model.tvalues['_did_interaction'])
            p_value = float(model.pvalues['_did_interaction'])
            ci_lower = float(model.conf_int().loc['_did_interaction', 0])
            ci_upper = float(model.conf_int().loc['_did_interaction', 1])

            # Extract all coefficients
            coeffs = {}
            for var in model.params.index:
                coeffs[var] = {
                    'coefficient': float(model.params[var]),
                    'std_error': float(model.bse[var]),
                    'p_value': float(model.pvalues[var]),
                    'significant': model.pvalues[var] < self.alpha
                }

            # Parallel trends test (if we have multiple pre-periods)
            # For simple 2-period DiD, we can't test this from data
            parallel_pvalue = None
            parallel_satisfied = None

            # Generate summary
            summary = self._generate_did_summary(
                did_estimate, p_value, treatment_group, treated_before, treated_after,
                control_before, control_after
            )

            # Educational notes
            edu_notes = {
                'what_is_did': 'DiD compares changes over time between treatment and control groups, removing time-invariant confounders.',
                'did_formula': f'DiD = (Treated_After - Treated_Before) - (Control_After - Control_Before) = ({treated_after:.2f} - {treated_before:.2f}) - ({control_after:.2f} - {control_before:.2f}) = {did_simple:.3f}',
                'key_assumption': 'Key assumption: Parallel trends - without treatment, treated and control groups would have followed parallel paths.',
                'interpreting_effect': f'The treatment {"increased" if did_estimate > 0 else "decreased"} the outcome by {abs(did_estimate):.3f} units relative to what would have happened without treatment.',
                'clustering': f'{"Standard errors clustered on " + cluster_var if cluster_var else "Consider clustering SEs if observations are correlated within groups."}',
                'when_to_use': 'Use DiD for policy evaluations, natural experiments, or when treatment timing varies across groups.'
            }

            warnings_list = []
            if n_treated < 20 or n_control < 20:
                warnings_list.append("Small sample sizes may reduce precision.")

            return DiDResult(
                n_observations=n_obs,
                n_treated=n_treated,
                n_control=n_control,
                n_periods=2,  # Standard DiD
                did_estimate=did_estimate,
                std_error=std_error,
                t_statistic=t_stat,
                p_value=p_value,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                treated_before=float(treated_before),
                treated_after=float(treated_after),
                control_before=float(control_before),
                control_after=float(control_after),
                parallel_trends_pvalue=parallel_pvalue,
                parallel_trends_satisfied=parallel_satisfied,
                coefficients=coeffs,
                summary_text=summary,
                warnings=warnings_list,
                educational_notes=edu_notes
            )

        except Exception as e:
            result = self._insufficient_did_data()
            result.warnings = [str(e)]
            return result

    def _ik_bandwidth(
        self,
        data: pd.DataFrame,
        outcome: str,
        running_var: str,
        cutoff: float
    ) -> float:
        """Calculate Imbens-Kalyanaraman optimal bandwidth (simplified)."""
        # Simplified IK bandwidth calculation
        n = len(data)
        sigma = data[outcome].std()
        x_range = data[running_var].max() - data[running_var].min()

        # Approximate optimal bandwidth
        h_opt = 1.06 * sigma * (n ** (-1/5)) * (x_range / 4)
        return max(h_opt, x_range * 0.1)  # At least 10% of range

    def _rdd_effect_at_bandwidth(
        self,
        data: pd.DataFrame,
        outcome: str,
        running_var: str,
        cutoff: float,
        bandwidth: float,
        polynomial_order: int,
        treatment_var: Optional[str] = None
    ) -> Optional[float]:
        """Calculate RDD treatment effect at specific bandwidth."""
        try:
            subset = data[
                (data[running_var] >= cutoff - bandwidth) &
                (data[running_var] <= cutoff + bandwidth)
            ].copy()

            if len(subset) < 10:
                return None

            if treatment_var is None:
                subset['_treatment'] = (subset[running_var] >= cutoff).astype(int)
                treatment_var = '_treatment'

            subset['_centered'] = subset[running_var] - cutoff

            formula = f"{outcome} ~ {treatment_var} + _centered"
            model = sm.OLS.from_formula(formula, data=subset).fit()

            return float(model.params[treatment_var])
        except:
            return None

    def _mccrary_test(
        self,
        data: pd.DataFrame,
        running_var: str,
        cutoff: float
    ) -> Optional[float]:
        """Simplified McCrary manipulation test."""
        try:
            # Count observations in bins around cutoff
            bandwidth = (data[running_var].max() - data[running_var].min()) / 20
            below = ((data[running_var] >= cutoff - bandwidth) &
                    (data[running_var] < cutoff)).sum()
            above = ((data[running_var] >= cutoff) &
                    (data[running_var] < cutoff + bandwidth)).sum()

            # Simple binomial test for discontinuity in density
            if below + above > 0:
                result = stats.binom_test(below, below + above, 0.5)
                return float(result)
            return None
        except:
            return None

    def _generate_iv_summary(
        self,
        effect: float,
        p_value: float,
        treatment: str,
        first_stage_f: float,
        weak_instruments: bool
    ) -> str:
        """Generate IV analysis summary."""
        sig_text = "statistically significant" if p_value < self.alpha else "not statistically significant"
        direction = "positive" if effect > 0 else "negative"

        lines = [
            f"Instrumental Variables (2SLS) Analysis:",
            f"- Treatment effect of {treatment}: {effect:.4f} ({direction}, {sig_text}, p={p_value:.4f})",
            f"- First-stage F-statistic: {first_stage_f:.2f} {'(WEAK INSTRUMENTS)' if weak_instruments else '(adequate)'}",
        ]

        if weak_instruments:
            lines.append("- WARNING: Consider finding stronger instruments or using weak-instrument robust methods.")

        return '\n'.join(lines)

    def _generate_rdd_summary(
        self,
        effect: float,
        p_value: float,
        cutoff: float,
        bandwidth: float,
        design_type: str,
        mccrary: Optional[float]
    ) -> str:
        """Generate RDD analysis summary."""
        sig_text = "statistically significant" if p_value < self.alpha else "not statistically significant"

        lines = [
            f"Regression Discontinuity Design ({design_type.capitalize()}):",
            f"- Treatment effect at cutoff ({cutoff}): {effect:.4f} ({sig_text}, p={p_value:.4f})",
            f"- Bandwidth: {bandwidth:.3f}",
        ]

        if mccrary:
            if mccrary < 0.05:
                lines.append(f"- WARNING: McCrary test (p={mccrary:.3f}) suggests possible manipulation.")
            else:
                lines.append(f"- McCrary test (p={mccrary:.3f}): No evidence of manipulation.")

        return '\n'.join(lines)

    def _generate_did_summary(
        self,
        did: float,
        p_value: float,
        treatment_group: str,
        t_before: float,
        t_after: float,
        c_before: float,
        c_after: float
    ) -> str:
        """Generate DiD analysis summary."""
        sig_text = "statistically significant" if p_value < self.alpha else "not statistically significant"
        direction = "increased" if did > 0 else "decreased"

        lines = [
            f"Difference-in-Differences Analysis:",
            f"- DiD estimate: {did:.4f} ({sig_text}, p={p_value:.4f})",
            f"- Treatment {direction} outcome by {abs(did):.4f} units",
            f"",
            f"Group means:",
            f"- Treated before: {t_before:.3f}, after: {t_after:.3f} (change: {t_after - t_before:+.3f})",
            f"- Control before: {c_before:.3f}, after: {c_after:.3f} (change: {c_after - c_before:+.3f})",
        ]

        return '\n'.join(lines)

    def _insufficient_iv_data(self) -> IVResult:
        """Return IV result indicating insufficient data."""
        return IVResult(
            n_observations=0,
            n_instruments=0,
            warnings=["Insufficient data for IV analysis"]
        )

    def _insufficient_rdd_data(self) -> RDDResult:
        """Return RDD result indicating insufficient data."""
        return RDDResult(
            n_observations=0,
            warnings=["Insufficient data for RDD analysis"]
        )

    def _insufficient_did_data(self) -> DiDResult:
        """Return DiD result indicating insufficient data."""
        return DiDResult(
            n_observations=0,
            warnings=["Insufficient data for DiD analysis"]
        )
