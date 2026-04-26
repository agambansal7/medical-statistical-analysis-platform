"""Sensitivity analysis module.

Provides methods for assessing robustness of findings:
- E-values for unmeasured confounding
- Tipping point analysis for missing data
- Influence diagnostics (leave-one-out)
- Specification curve analysis
- Multiple testing corrections
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
import warnings


@dataclass
class EValueResult:
    """Result of E-value calculation."""
    estimate_type: str  # "RR", "OR", "HR"
    point_estimate: float
    confidence_limit: float  # Usually lower CI for protective, upper for harmful

    # E-values
    e_value_point: float
    e_value_ci: float

    # Interpretation
    interpretation: str
    summary_text: Optional[str] = None
    educational_notes: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class TippingPointResult:
    """Result of tipping point analysis."""
    original_estimate: float
    original_pvalue: float
    original_significant: bool

    # Tipping point
    tipping_point_value: Optional[float] = None
    tipping_point_percent: Optional[float] = None

    # Scenarios tested
    scenarios: List[Dict[str, Any]] = field(default_factory=list)

    # Robustness
    robustness_assessment: str = ""
    summary_text: Optional[str] = None
    educational_notes: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class InfluenceResult:
    """Result of influence diagnostics."""
    n_observations: int

    # Influential observations
    influential_indices: List[int] = field(default_factory=list)
    influential_values: Dict[int, Dict[str, float]] = field(default_factory=dict)

    # Cook's distance
    cooks_d_threshold: float = 0.0
    high_cooks_d: List[int] = field(default_factory=list)

    # DFBETAS
    high_dfbetas: Dict[str, List[int]] = field(default_factory=dict)

    # Leave-one-out results
    loo_estimates: List[float] = field(default_factory=list)
    loo_std: float = 0.0
    loo_range: Tuple[float, float] = (0.0, 0.0)

    # Stability
    estimate_stable: bool = True
    sign_changes: int = 0
    significance_changes: int = 0

    summary_text: Optional[str] = None
    educational_notes: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['loo_range'] = list(self.loo_range)
        return {k: v for k, v in result.items() if v is not None}


@dataclass
class SpecificationCurveResult:
    """Result of specification curve analysis."""
    n_specifications: int
    median_estimate: float
    estimate_range: Tuple[float, float]

    # Proportion significant
    prop_significant: float
    prop_positive: float

    # All specifications
    specifications: List[Dict[str, Any]] = field(default_factory=list)

    # Robustness
    robust: bool = True
    robustness_score: float = 0.0

    summary_text: Optional[str] = None
    educational_notes: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['estimate_range'] = list(self.estimate_range)
        return {k: v for k, v in result.items() if v is not None}


class SensitivityAnalysis:
    """Sensitivity analysis methods."""

    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level

    def e_value(
        self,
        estimate: float,
        ci_lower: Optional[float] = None,
        ci_upper: Optional[float] = None,
        estimate_type: str = "RR",
        rare_outcome: bool = False
    ) -> EValueResult:
        """Calculate E-value for unmeasured confounding.

        The E-value is the minimum strength of association that an unmeasured
        confounder would need to have with both treatment and outcome to fully
        explain away the observed association.

        Args:
            estimate: Point estimate (RR, OR, or HR)
            ci_lower: Lower confidence limit
            ci_upper: Upper confidence limit
            estimate_type: "RR" (risk ratio), "OR" (odds ratio), "HR" (hazard ratio)
            rare_outcome: If True and estimate_type is OR, approximates RR

        Returns:
            EValueResult object
        """
        # Convert OR to RR approximation if needed
        if estimate_type == "OR" and not rare_outcome:
            # Square root transformation for common outcomes
            estimate = self._or_to_rr(estimate)
            if ci_lower:
                ci_lower = self._or_to_rr(ci_lower)
            if ci_upper:
                ci_upper = self._or_to_rr(ci_upper)
            estimate_type = "RR (converted from OR)"

        # Ensure estimate is >= 1 (for protective effects, take reciprocal)
        protective = estimate < 1
        if protective:
            estimate = 1 / estimate
            if ci_lower and ci_upper:
                ci_lower, ci_upper = 1 / ci_upper, 1 / ci_lower

        # Calculate E-value for point estimate
        e_value_point = self._calculate_e_value(estimate)

        # Calculate E-value for confidence interval
        if protective:
            # For protective effect, use upper CI (closest to null)
            ci_limit = ci_upper if ci_upper else estimate
        else:
            # For harmful effect, use lower CI (closest to null)
            ci_limit = ci_lower if ci_lower else estimate

        if ci_limit and ci_limit > 1:
            e_value_ci = self._calculate_e_value(ci_limit)
        else:
            e_value_ci = 1.0  # CI includes null

        # Generate interpretation
        interpretation = self._interpret_e_value(e_value_point, e_value_ci, protective)

        summary = f"""E-Value Analysis:
- Point estimate: {estimate:.2f} ({"protective" if protective else "harmful"})
- E-value for point estimate: {e_value_point:.2f}
- E-value for confidence limit: {e_value_ci:.2f}

{interpretation}"""

        edu_notes = {
            'what_is_e_value': 'The E-value quantifies the minimum strength of unmeasured confounding needed to explain away an observed association.',
            'interpretation': f'E-value of {e_value_point:.2f} means an unmeasured confounder would need RR ≥ {e_value_point:.2f} with both exposure and outcome to reduce the observed effect to null.',
            'ci_e_value': f'E-value for CI ({e_value_ci:.2f}) indicates robustness of statistical significance to unmeasured confounding.',
            'benchmarks': 'Compare to known strong confounders: smoking-lung cancer RR ≈ 10, obesity-diabetes RR ≈ 3-5.',
            'limitations': 'E-values assume a single unmeasured confounder. Multiple weaker confounders could also explain the association.',
            'when_to_use': 'Use E-values in observational studies to assess how robust findings are to potential unmeasured confounding.'
        }

        return EValueResult(
            estimate_type=estimate_type,
            point_estimate=1/estimate if protective else estimate,
            confidence_limit=ci_limit if ci_limit else estimate,
            e_value_point=e_value_point,
            e_value_ci=e_value_ci,
            interpretation=interpretation,
            summary_text=summary,
            educational_notes=edu_notes
        )

    def tipping_point_analysis(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        covariates: Optional[List[str]] = None,
        missing_var: Optional[str] = None,
        imputed_values: Optional[List[float]] = None
    ) -> TippingPointResult:
        """Tipping point analysis for missing data sensitivity.

        Tests how conclusions change under different assumptions about missing data.

        Args:
            data: DataFrame with the data
            outcome: Outcome variable
            treatment: Treatment variable
            covariates: Control variables
            missing_var: Variable with missing data to analyze
            imputed_values: List of values to impute for sensitivity analysis

        Returns:
            TippingPointResult object
        """
        covariates = covariates or []

        # Determine which variable has missing data
        if missing_var is None:
            missing_var = outcome  # Default to outcome

        # Original complete case analysis
        all_vars = [outcome, treatment] + covariates
        complete_data = data[all_vars].dropna()

        try:
            formula = f"{outcome} ~ {treatment}"
            if covariates:
                formula += " + " + " + ".join(covariates)

            model = sm.OLS.from_formula(formula, data=complete_data).fit()
            original_estimate = float(model.params[treatment])
            original_se = float(model.bse[treatment])
            original_p = float(model.pvalues[treatment])
            original_sig = original_p < self.alpha

            # Find missing data
            missing_mask = data[missing_var].isna()
            n_missing = missing_mask.sum()

            if n_missing == 0:
                return TippingPointResult(
                    original_estimate=original_estimate,
                    original_pvalue=original_p,
                    original_significant=original_sig,
                    robustness_assessment="No missing data in specified variable",
                    summary_text="No missing data to analyze for tipping point."
                )

            # Generate imputation scenarios if not provided
            if imputed_values is None:
                observed_mean = data[missing_var].mean()
                observed_std = data[missing_var].std()
                imputed_values = [
                    observed_mean - 2 * observed_std,
                    observed_mean - observed_std,
                    observed_mean,
                    observed_mean + observed_std,
                    observed_mean + 2 * observed_std
                ]

            # Test each scenario
            scenarios = []
            tipping_point = None
            tipping_percent = None

            for imp_val in imputed_values:
                test_data = data.copy()
                test_data.loc[missing_mask, missing_var] = imp_val

                test_clean = test_data[all_vars].dropna()
                test_model = sm.OLS.from_formula(formula, data=test_clean).fit()

                test_estimate = float(test_model.params[treatment])
                test_p = float(test_model.pvalues[treatment])
                test_sig = test_p < self.alpha

                # Check for sign change or significance change
                sign_change = (test_estimate * original_estimate) < 0
                sig_change = test_sig != original_sig

                scenarios.append({
                    'imputed_value': float(imp_val),
                    'estimate': test_estimate,
                    'pvalue': test_p,
                    'significant': test_sig,
                    'sign_change': sign_change,
                    'significance_change': sig_change
                })

                # Find tipping point (where significance changes)
                if sig_change and tipping_point is None:
                    tipping_point = imp_val
                    # Express as percentage deviation from observed mean
                    if observed_std > 0:
                        tipping_percent = (imp_val - observed_mean) / observed_std

            # Assess robustness
            n_sig_changes = sum(s['significance_change'] for s in scenarios)
            n_sign_changes = sum(s['sign_change'] for s in scenarios)

            if n_sig_changes == 0 and n_sign_changes == 0:
                robustness = "Robust: Conclusions stable across all imputation scenarios"
            elif n_sign_changes > 0:
                robustness = f"Not robust: Sign changes in {n_sign_changes}/{len(scenarios)} scenarios"
            else:
                robustness = f"Moderately robust: Significance changes in {n_sig_changes}/{len(scenarios)} scenarios"

            summary = f"""Tipping Point Analysis:
- Original estimate: {original_estimate:.4f} (p={original_p:.4f}, {'significant' if original_sig else 'not significant'})
- Missing observations: {n_missing} ({n_missing/len(data)*100:.1f}%)
- Scenarios tested: {len(scenarios)}
- Tipping point: {f'{tipping_point:.2f} ({tipping_percent:+.1f} SD from mean)' if tipping_point else 'Not found'}

{robustness}"""

            edu_notes = {
                'what_is_tipping_point': 'Tipping point analysis tests how extreme missing data would need to be to change conclusions.',
                'interpretation': f'{"A tipping point was found - conclusions depend on missing data assumptions." if tipping_point else "No tipping point found - conclusions are robust to missing data."}',
                'missing_at_random': 'This analysis is especially important when MNAR (Missing Not At Random) is plausible.',
                'clinical_plausibility': 'Consider whether tipping point values are clinically plausible.',
                'when_to_use': 'Use when missing data is substantial and you need to assess how sensitive conclusions are to missing data assumptions.'
            }

            return TippingPointResult(
                original_estimate=original_estimate,
                original_pvalue=original_p,
                original_significant=original_sig,
                tipping_point_value=tipping_point,
                tipping_point_percent=tipping_percent,
                scenarios=scenarios,
                robustness_assessment=robustness,
                summary_text=summary,
                educational_notes=edu_notes
            )

        except Exception as e:
            return TippingPointResult(
                original_estimate=0,
                original_pvalue=1,
                original_significant=False,
                robustness_assessment=f"Error: {str(e)}"
            )

    def influence_diagnostics(
        self,
        data: pd.DataFrame,
        outcome: str,
        predictors: List[str],
        cooks_d_threshold: Optional[float] = None
    ) -> InfluenceResult:
        """Leave-one-out influence diagnostics.

        Identifies influential observations that unduly affect results.

        Args:
            data: DataFrame with the data
            outcome: Dependent variable
            predictors: Predictor variables
            cooks_d_threshold: Threshold for influential Cook's D (default: 4/n)

        Returns:
            InfluenceResult object
        """
        all_vars = [outcome] + predictors
        clean_data = data[all_vars].dropna().reset_index(drop=True)
        n = len(clean_data)

        if n < 10:
            return InfluenceResult(n_observations=0, summary_text="Insufficient data")

        if cooks_d_threshold is None:
            cooks_d_threshold = 4 / n

        try:
            # Fit full model
            formula = f"{outcome} ~ {' + '.join(predictors)}"
            full_model = sm.OLS.from_formula(formula, data=clean_data).fit()

            # Get influence measures
            influence = full_model.get_influence()
            cooks_d = influence.cooks_distance[0]
            dfbetas = influence.dfbetas

            # Identify high Cook's D
            high_cooks = list(np.where(cooks_d > cooks_d_threshold)[0])

            # Identify high DFBETAS (threshold: 2/sqrt(n))
            dfbetas_threshold = 2 / np.sqrt(n)
            high_dfbetas = {}
            for i, var in enumerate(full_model.params.index):
                high_idx = list(np.where(np.abs(dfbetas[:, i]) > dfbetas_threshold)[0])
                if high_idx:
                    high_dfbetas[var] = high_idx

            # Leave-one-out analysis
            loo_estimates = []
            original_estimate = float(full_model.params[predictors[0]])  # First predictor
            original_sign = np.sign(original_estimate)
            original_sig = full_model.pvalues[predictors[0]] < self.alpha

            sign_changes = 0
            sig_changes = 0

            for i in range(n):
                loo_data = clean_data.drop(i)
                loo_model = sm.OLS.from_formula(formula, data=loo_data).fit()
                loo_est = float(loo_model.params[predictors[0]])
                loo_estimates.append(loo_est)

                if np.sign(loo_est) != original_sign:
                    sign_changes += 1
                if (loo_model.pvalues[predictors[0]] < self.alpha) != original_sig:
                    sig_changes += 1

            loo_estimates = np.array(loo_estimates)
            loo_std = float(np.std(loo_estimates))
            loo_range = (float(np.min(loo_estimates)), float(np.max(loo_estimates)))

            # Compile influential observations info
            influential_indices = list(set(high_cooks))
            influential_values = {}
            for idx in influential_indices:
                influential_values[idx] = {
                    'cooks_d': float(cooks_d[idx]),
                    'loo_estimate': float(loo_estimates[idx]),
                    'loo_change': float(loo_estimates[idx] - original_estimate)
                }

            # Stability assessment
            estimate_stable = sign_changes == 0 and sig_changes < n * 0.05

            summary = f"""Influence Diagnostics:
- Observations: {n}
- High Cook's D (>{cooks_d_threshold:.4f}): {len(high_cooks)} observations
- Leave-one-out estimate range: [{loo_range[0]:.4f}, {loo_range[1]:.4f}]
- Sign changes: {sign_changes}, Significance changes: {sig_changes}
- Stability: {'Stable' if estimate_stable else 'Potentially unstable'}

Most influential observations: {influential_indices[:5]}"""

            edu_notes = {
                'cooks_d': f"Cook's D measures each observation's influence on all fitted values. Threshold: {cooks_d_threshold:.4f}",
                'dfbetas': 'DFBETAS measure influence on individual coefficients.',
                'loo': 'Leave-one-out analysis shows how removing each observation affects estimates.',
                'interpretation': f'{"Results appear stable to individual observations." if estimate_stable else "Results may be sensitive to specific observations."}',
                'action': 'Investigate influential observations for data errors or legitimate extreme cases.',
                'when_to_use': 'Use to identify observations that unduly influence results and assess robustness.'
            }

            return InfluenceResult(
                n_observations=n,
                influential_indices=influential_indices,
                influential_values=influential_values,
                cooks_d_threshold=cooks_d_threshold,
                high_cooks_d=high_cooks,
                high_dfbetas=high_dfbetas,
                loo_estimates=list(loo_estimates),
                loo_std=loo_std,
                loo_range=loo_range,
                estimate_stable=estimate_stable,
                sign_changes=sign_changes,
                significance_changes=sig_changes,
                summary_text=summary,
                educational_notes=edu_notes
            )

        except Exception as e:
            return InfluenceResult(
                n_observations=n,
                summary_text=f"Error: {str(e)}"
            )

    def specification_curve(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        required_covariates: Optional[List[str]] = None,
        optional_covariates: Optional[List[str]] = None,
        transformations: Optional[Dict[str, List[str]]] = None
    ) -> SpecificationCurveResult:
        """Specification curve analysis.

        Tests robustness across many reasonable model specifications.

        Args:
            data: DataFrame with the data
            outcome: Outcome variable
            treatment: Treatment variable
            required_covariates: Covariates always included
            optional_covariates: Covariates to try including/excluding
            transformations: Dict of variable transformations to try

        Returns:
            SpecificationCurveResult object
        """
        required_covariates = required_covariates or []
        optional_covariates = optional_covariates or []
        transformations = transformations or {}

        # Generate all covariate combinations
        from itertools import combinations

        covariate_sets = [[]]
        for r in range(1, len(optional_covariates) + 1):
            for combo in combinations(optional_covariates, r):
                covariate_sets.append(list(combo))

        specifications = []

        for cov_set in covariate_sets:
            covariates = required_covariates + list(cov_set)

            try:
                all_vars = [outcome, treatment] + covariates
                clean_data = data[all_vars].dropna()

                if len(clean_data) < 20:
                    continue

                formula = f"{outcome} ~ {treatment}"
                if covariates:
                    formula += " + " + " + ".join(covariates)

                model = sm.OLS.from_formula(formula, data=clean_data).fit()

                estimate = float(model.params[treatment])
                se = float(model.bse[treatment])
                pvalue = float(model.pvalues[treatment])
                ci_lower = float(model.conf_int().loc[treatment, 0])
                ci_upper = float(model.conf_int().loc[treatment, 1])

                specifications.append({
                    'covariates': covariates,
                    'n_covariates': len(covariates),
                    'n_observations': len(clean_data),
                    'estimate': estimate,
                    'se': se,
                    'pvalue': pvalue,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'significant': pvalue < self.alpha,
                    'positive': estimate > 0
                })

            except:
                continue

        if not specifications:
            return SpecificationCurveResult(
                n_specifications=0,
                median_estimate=0,
                estimate_range=(0, 0),
                prop_significant=0,
                prop_positive=0,
                robust=False,
                robustness_score=0,
                summary_text="Could not fit any specifications"
            )

        # Summarize results
        estimates = [s['estimate'] for s in specifications]
        median_est = float(np.median(estimates))
        est_range = (float(np.min(estimates)), float(np.max(estimates)))

        prop_sig = np.mean([s['significant'] for s in specifications])
        prop_pos = np.mean([s['positive'] for s in specifications])

        # Robustness score: proportion with same sign and significance
        modal_sign = np.sign(median_est)
        modal_sig = prop_sig >= 0.5

        same_sign = np.mean([np.sign(s['estimate']) == modal_sign for s in specifications])
        same_sig = np.mean([s['significant'] == modal_sig for s in specifications])
        robustness_score = (same_sign + same_sig) / 2

        robust = robustness_score >= 0.8 and same_sign >= 0.9

        # Sort specifications by estimate for visualization
        specifications = sorted(specifications, key=lambda x: x['estimate'])

        summary = f"""Specification Curve Analysis:
- Specifications tested: {len(specifications)}
- Median estimate: {median_est:.4f}
- Estimate range: [{est_range[0]:.4f}, {est_range[1]:.4f}]
- Proportion significant: {prop_sig:.1%}
- Proportion positive: {prop_pos:.1%}
- Robustness score: {robustness_score:.2f}

Conclusion: {'Results are robust across specifications' if robust else 'Results are sensitive to specification choices'}"""

        edu_notes = {
            'what_is_spec_curve': 'Specification curve shows how results vary across all reasonable model specifications.',
            'interpretation': f'{prop_sig*100:.0f}% of specifications are statistically significant.',
            'robustness': f'{"Results are robust - consistent across specifications." if robust else "Results vary - conclusions depend on modeling choices."}',
            'visualization': 'Plot estimates sorted by magnitude with confidence intervals to visualize the specification curve.',
            'when_to_use': 'Use to demonstrate robustness or identify specification-dependent findings.'
        }

        return SpecificationCurveResult(
            n_specifications=len(specifications),
            median_estimate=median_est,
            estimate_range=est_range,
            prop_significant=prop_sig,
            prop_positive=prop_pos,
            specifications=specifications,
            robust=robust,
            robustness_score=robustness_score,
            summary_text=summary,
            educational_notes=edu_notes
        )

    def _calculate_e_value(self, rr: float) -> float:
        """Calculate E-value from risk ratio."""
        if rr <= 1:
            return 1.0
        return float(rr + np.sqrt(rr * (rr - 1)))

    def _or_to_rr(self, or_val: float, baseline_risk: float = 0.1) -> float:
        """Convert OR to RR approximation."""
        # Zhang & Yu formula
        if or_val == 1:
            return 1.0
        rr = or_val / ((1 - baseline_risk) + baseline_risk * or_val)
        return rr

    def _interpret_e_value(self, e_point: float, e_ci: float, protective: bool) -> str:
        """Generate E-value interpretation."""
        effect = "protective" if protective else "harmful"

        if e_ci < 1.5:
            strength = "very weak"
        elif e_ci < 2.0:
            strength = "weak"
        elif e_ci < 3.0:
            strength = "moderate"
        elif e_ci < 5.0:
            strength = "strong"
        else:
            strength = "very strong"

        lines = [
            f"For the {effect} effect to be explained away by an unmeasured confounder:",
            f"- To nullify the point estimate: confounder needs RR ≥ {e_point:.2f} with both exposure and outcome",
            f"- To nullify the confidence interval: confounder needs RR ≥ {e_ci:.2f}",
            f"",
            f"Robustness: {strength.upper()} - E-value of {e_ci:.2f} suggests {'' if e_ci >= 2 else 'limited '}robustness to unmeasured confounding"
        ]

        return '\n'.join(lines)
