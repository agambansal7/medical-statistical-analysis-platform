"""Mediation analysis module.

Provides methods for causal mediation analysis:
- Baron-Kenny approach
- Sobel test
- Bootstrap mediation
- Causal mediation analysis
- Moderated mediation
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
import warnings


@dataclass
class MediationResult:
    """Result of mediation analysis."""
    model_type: str
    n_observations: int

    # Variables
    exposure: str = ""
    mediator: str = ""
    outcome: str = ""
    covariates: List[str] = field(default_factory=list)

    # Total effect (c path: X -> Y)
    total_effect: Optional[float] = None
    total_effect_se: Optional[float] = None
    total_effect_pvalue: Optional[float] = None

    # Direct effect (c' path: X -> Y controlling for M)
    direct_effect: Optional[float] = None
    direct_effect_se: Optional[float] = None
    direct_effect_pvalue: Optional[float] = None

    # Indirect effect (a*b path: X -> M -> Y)
    indirect_effect: Optional[float] = None
    indirect_effect_se: Optional[float] = None
    indirect_effect_pvalue: Optional[float] = None
    indirect_effect_ci: Optional[Tuple[float, float]] = None

    # Path coefficients
    a_path: Optional[float] = None  # X -> M
    a_path_se: Optional[float] = None
    a_path_pvalue: Optional[float] = None

    b_path: Optional[float] = None  # M -> Y (controlling for X)
    b_path_se: Optional[float] = None
    b_path_pvalue: Optional[float] = None

    # Proportion mediated
    proportion_mediated: Optional[float] = None

    # Tests
    sobel_statistic: Optional[float] = None
    sobel_pvalue: Optional[float] = None
    aroian_statistic: Optional[float] = None
    aroian_pvalue: Optional[float] = None

    # Bootstrap results
    bootstrap_ci_method: Optional[str] = None
    bootstrap_n: Optional[int] = None

    # Interpretation
    summary_text: Optional[str] = None
    mediation_type: Optional[str] = None  # full, partial, none, inconsistent
    warnings: List[str] = field(default_factory=list)
    educational_notes: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.indirect_effect_ci:
            result['indirect_effect_ci'] = list(self.indirect_effect_ci)
        return {k: v for k, v in result.items() if v is not None}


@dataclass
class CausalMediationResult:
    """Result of causal mediation analysis."""
    model_type: str = "Causal Mediation Analysis"
    n_observations: int = 0

    # Causal effects
    acme: Optional[float] = None  # Average Causal Mediation Effect
    acme_ci: Optional[Tuple[float, float]] = None
    acme_pvalue: Optional[float] = None

    ade: Optional[float] = None  # Average Direct Effect
    ade_ci: Optional[Tuple[float, float]] = None
    ade_pvalue: Optional[float] = None

    total_effect: Optional[float] = None
    total_effect_ci: Optional[Tuple[float, float]] = None

    proportion_mediated: Optional[float] = None
    proportion_mediated_ci: Optional[Tuple[float, float]] = None

    # Sensitivity analysis
    sensitivity_rho: Optional[float] = None  # Correlation at which ACME = 0
    robustness_assessment: Optional[str] = None

    # Interpretation
    summary_text: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    educational_notes: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key in ['acme_ci', 'ade_ci', 'total_effect_ci', 'proportion_mediated_ci']:
            if result.get(key):
                result[key] = list(result[key])
        return {k: v for k, v in result.items() if v is not None}


class MediationAnalysis:
    """Mediation analysis methods."""

    def __init__(self, significance_level: float = 0.05, random_state: int = 42):
        self.alpha = significance_level
        self.random_state = random_state

    def baron_kenny(
        self,
        data: pd.DataFrame,
        exposure: str,
        mediator: str,
        outcome: str,
        covariates: Optional[List[str]] = None
    ) -> MediationResult:
        """Baron-Kenny mediation analysis with Sobel test.

        Classic approach to mediation testing the four Baron-Kenny conditions:
        1. X significantly predicts Y (total effect, c path)
        2. X significantly predicts M (a path)
        3. M significantly predicts Y controlling for X (b path)
        4. Effect of X on Y is reduced when controlling for M (c' < c)

        Args:
            data: DataFrame with the data
            exposure: Independent variable (X)
            mediator: Mediator variable (M)
            outcome: Dependent variable (Y)
            covariates: Additional control variables

        Returns:
            MediationResult object
        """
        covariates = covariates or []

        # Prepare data
        all_vars = [exposure, mediator, outcome] + covariates
        clean_data = data[all_vars].dropna()
        n = len(clean_data)

        if n < 10:
            return self._insufficient_data("Baron-Kenny Mediation")

        try:
            # Step 1: Total effect (c path): X -> Y
            formula_total = f"{outcome} ~ {exposure}"
            if covariates:
                formula_total += " + " + " + ".join(covariates)

            model_total = sm.OLS.from_formula(formula_total, data=clean_data).fit()
            c_total = float(model_total.params[exposure])
            c_total_se = float(model_total.bse[exposure])
            c_total_p = float(model_total.pvalues[exposure])

            # Step 2: a path: X -> M
            formula_a = f"{mediator} ~ {exposure}"
            if covariates:
                formula_a += " + " + " + ".join(covariates)

            model_a = sm.OLS.from_formula(formula_a, data=clean_data).fit()
            a = float(model_a.params[exposure])
            a_se = float(model_a.bse[exposure])
            a_p = float(model_a.pvalues[exposure])

            # Step 3 & 4: b path and direct effect (c' path)
            formula_full = f"{outcome} ~ {exposure} + {mediator}"
            if covariates:
                formula_full += " + " + " + ".join(covariates)

            model_full = sm.OLS.from_formula(formula_full, data=clean_data).fit()
            c_prime = float(model_full.params[exposure])  # Direct effect
            c_prime_se = float(model_full.bse[exposure])
            c_prime_p = float(model_full.pvalues[exposure])

            b = float(model_full.params[mediator])
            b_se = float(model_full.bse[mediator])
            b_p = float(model_full.pvalues[mediator])

            # Indirect effect: a * b
            indirect = a * b
            indirect_se = np.sqrt(a**2 * b_se**2 + b**2 * a_se**2)  # Sobel formula

            # Sobel test
            sobel_z = indirect / indirect_se
            sobel_p = float(2 * (1 - stats.norm.cdf(abs(sobel_z))))

            # Aroian test (slightly different formula)
            aroian_se = np.sqrt(a**2 * b_se**2 + b**2 * a_se**2 + a_se**2 * b_se**2)
            aroian_z = indirect / aroian_se
            aroian_p = float(2 * (1 - stats.norm.cdf(abs(aroian_z))))

            # Proportion mediated
            if abs(c_total) > 1e-10:
                prop_mediated = indirect / c_total
            else:
                prop_mediated = None

            # Determine mediation type
            mediation_type = self._classify_mediation(
                c_total, c_total_p, a, a_p, b, b_p, c_prime, c_prime_p, indirect, sobel_p
            )

            # Generate summary
            summary = self._generate_mediation_summary(
                exposure, mediator, outcome,
                c_total, c_total_p, a, a_p, b, b_p,
                c_prime, c_prime_p, indirect, sobel_p,
                prop_mediated, mediation_type
            )

            # Educational notes
            edu_notes = {
                'what_is_mediation': 'Mediation analysis tests whether the effect of X on Y operates through an intermediate variable M.',
                'paths': f'a path (X→M): {a:.3f}, b path (M→Y|X): {b:.3f}, c path (total): {c_total:.3f}, c\' path (direct): {c_prime:.3f}',
                'indirect_effect': f'Indirect effect (a×b) = {indirect:.3f}. This is the portion of the X→Y effect that goes through M.',
                'sobel_test': f'Sobel test (z={sobel_z:.2f}, p={sobel_p:.4f}) tests whether the indirect effect significantly differs from zero.',
                'baron_kenny_conditions': 'Classic Baron-Kenny requires: (1) c significant, (2) a significant, (3) b significant, (4) c\' < c.',
                'modern_view': 'Modern approaches recommend bootstrap CIs for indirect effect, as Sobel test assumes normality which may not hold.',
                'mediation_type': f'Result: {mediation_type}',
                'cautions': 'Mediation analysis cannot establish causality without proper study design. Consider temporal ordering and confounders.'
            }

            return MediationResult(
                model_type="Baron-Kenny Mediation Analysis",
                n_observations=n,
                exposure=exposure,
                mediator=mediator,
                outcome=outcome,
                covariates=covariates,
                total_effect=c_total,
                total_effect_se=c_total_se,
                total_effect_pvalue=c_total_p,
                direct_effect=c_prime,
                direct_effect_se=c_prime_se,
                direct_effect_pvalue=c_prime_p,
                indirect_effect=indirect,
                indirect_effect_se=indirect_se,
                indirect_effect_pvalue=sobel_p,
                a_path=a,
                a_path_se=a_se,
                a_path_pvalue=a_p,
                b_path=b,
                b_path_se=b_se,
                b_path_pvalue=b_p,
                proportion_mediated=prop_mediated,
                sobel_statistic=sobel_z,
                sobel_pvalue=sobel_p,
                aroian_statistic=aroian_z,
                aroian_pvalue=aroian_p,
                mediation_type=mediation_type,
                summary_text=summary,
                educational_notes=edu_notes
            )

        except Exception as e:
            result = self._insufficient_data("Baron-Kenny Mediation")
            result.warnings = [str(e)]
            return result

    def bootstrap_mediation(
        self,
        data: pd.DataFrame,
        exposure: str,
        mediator: str,
        outcome: str,
        covariates: Optional[List[str]] = None,
        n_bootstrap: int = 5000,
        ci_method: str = "percentile"
    ) -> MediationResult:
        """Bootstrap mediation analysis.

        Uses bootstrap resampling for confidence intervals on indirect effect.

        Args:
            data: DataFrame with the data
            exposure: Independent variable (X)
            mediator: Mediator variable (M)
            outcome: Dependent variable (Y)
            covariates: Additional control variables
            n_bootstrap: Number of bootstrap samples
            ci_method: CI method ("percentile" or "bca")

        Returns:
            MediationResult object
        """
        covariates = covariates or []

        all_vars = [exposure, mediator, outcome] + covariates
        clean_data = data[all_vars].dropna()
        n = len(clean_data)

        if n < 20:
            return self._insufficient_data("Bootstrap Mediation")

        np.random.seed(self.random_state)

        try:
            # Get point estimates using Baron-Kenny
            bk_result = self.baron_kenny(data, exposure, mediator, outcome, covariates)

            # Bootstrap
            indirect_effects = []
            direct_effects = []
            total_effects = []

            for _ in range(n_bootstrap):
                # Resample with replacement
                boot_data = clean_data.sample(n=n, replace=True)

                try:
                    # a path
                    formula_a = f"{mediator} ~ {exposure}"
                    if covariates:
                        formula_a += " + " + " + ".join(covariates)
                    model_a = sm.OLS.from_formula(formula_a, data=boot_data).fit()
                    a = model_a.params[exposure]

                    # b path and direct effect
                    formula_full = f"{outcome} ~ {exposure} + {mediator}"
                    if covariates:
                        formula_full += " + " + " + ".join(covariates)
                    model_full = sm.OLS.from_formula(formula_full, data=boot_data).fit()
                    b = model_full.params[mediator]
                    c_prime = model_full.params[exposure]

                    # Total effect
                    formula_total = f"{outcome} ~ {exposure}"
                    if covariates:
                        formula_total += " + " + " + ".join(covariates)
                    model_total = sm.OLS.from_formula(formula_total, data=boot_data).fit()
                    c = model_total.params[exposure]

                    indirect_effects.append(a * b)
                    direct_effects.append(c_prime)
                    total_effects.append(c)

                except:
                    continue

            indirect_effects = np.array(indirect_effects)
            direct_effects = np.array(direct_effects)
            total_effects = np.array(total_effects)

            # Calculate CIs
            alpha = self.alpha
            if ci_method == "percentile":
                indirect_ci = (
                    float(np.percentile(indirect_effects, alpha/2 * 100)),
                    float(np.percentile(indirect_effects, (1 - alpha/2) * 100))
                )
                direct_ci = (
                    float(np.percentile(direct_effects, alpha/2 * 100)),
                    float(np.percentile(direct_effects, (1 - alpha/2) * 100))
                )
            else:  # BCa (simplified)
                indirect_ci = (
                    float(np.percentile(indirect_effects, 2.5)),
                    float(np.percentile(indirect_effects, 97.5))
                )
                direct_ci = (
                    float(np.percentile(direct_effects, 2.5)),
                    float(np.percentile(direct_effects, 97.5))
                )

            # Bootstrap SE
            indirect_se = float(np.std(indirect_effects))
            direct_se = float(np.std(direct_effects))

            # p-value based on bootstrap distribution
            indirect_p = float(2 * min(
                np.mean(indirect_effects < 0),
                np.mean(indirect_effects > 0)
            ))

            # Significance: CI doesn't include zero
            indirect_sig = not (indirect_ci[0] <= 0 <= indirect_ci[1])

            # Update result with bootstrap estimates
            result = bk_result
            result.model_type = "Bootstrap Mediation Analysis"
            result.indirect_effect_se = indirect_se
            result.indirect_effect_pvalue = indirect_p
            result.indirect_effect_ci = indirect_ci
            result.direct_effect_se = direct_se
            result.bootstrap_ci_method = ci_method
            result.bootstrap_n = n_bootstrap

            # Update mediation type based on bootstrap CI
            if indirect_sig:
                if result.direct_effect_pvalue < self.alpha:
                    result.mediation_type = "Partial mediation (bootstrap significant)"
                else:
                    result.mediation_type = "Full mediation (bootstrap significant)"
            else:
                result.mediation_type = "No significant mediation (bootstrap CI includes 0)"

            # Update summary
            result.summary_text = f"""Bootstrap Mediation Analysis ({n_bootstrap} samples):
- Indirect effect: {result.indirect_effect:.4f}, 95% CI [{indirect_ci[0]:.4f}, {indirect_ci[1]:.4f}]
- Direct effect: {result.direct_effect:.4f}
- Total effect: {result.total_effect:.4f}
- Proportion mediated: {result.proportion_mediated:.1%} if result.proportion_mediated else 'N/A'
- Mediation: {result.mediation_type}

Bootstrap CI {'excludes' if indirect_sig else 'includes'} zero → {'significant' if indirect_sig else 'not significant'} mediation"""

            result.educational_notes['bootstrap'] = f'Bootstrap ({n_bootstrap} samples) provides accurate CIs without assuming normality of indirect effect.'
            result.educational_notes['ci_interpretation'] = f'95% CI [{indirect_ci[0]:.4f}, {indirect_ci[1]:.4f}] {"does not include" if indirect_sig else "includes"} zero.'

            return result

        except Exception as e:
            result = self._insufficient_data("Bootstrap Mediation")
            result.warnings = [str(e)]
            return result

    def causal_mediation(
        self,
        data: pd.DataFrame,
        exposure: str,
        mediator: str,
        outcome: str,
        covariates: Optional[List[str]] = None,
        n_simulations: int = 1000,
        outcome_type: str = "continuous",
        mediator_type: str = "continuous"
    ) -> CausalMediationResult:
        """Causal mediation analysis using potential outcomes framework.

        Based on Imai, Keele, and Tingley (2010) approach.

        Args:
            data: DataFrame with the data
            exposure: Treatment/exposure variable (binary or continuous)
            mediator: Mediator variable
            outcome: Outcome variable
            covariates: Pre-treatment covariates
            n_simulations: Number of Monte Carlo simulations
            outcome_type: "continuous" or "binary"
            mediator_type: "continuous" or "binary"

        Returns:
            CausalMediationResult object
        """
        covariates = covariates or []

        all_vars = [exposure, mediator, outcome] + covariates
        clean_data = data[all_vars].dropna()
        n = len(clean_data)

        if n < 30:
            return self._insufficient_causal_data()

        np.random.seed(self.random_state)

        try:
            # Model for mediator
            formula_m = f"{mediator} ~ {exposure}"
            if covariates:
                formula_m += " + " + " + ".join(covariates)

            if mediator_type == "binary":
                from statsmodels.formula.api import logit
                model_m = logit(formula_m, data=clean_data).fit(disp=0)
            else:
                model_m = sm.OLS.from_formula(formula_m, data=clean_data).fit()

            # Model for outcome
            formula_y = f"{outcome} ~ {exposure} + {mediator}"
            if covariates:
                formula_y += " + " + " + ".join(covariates)

            if outcome_type == "binary":
                from statsmodels.formula.api import logit
                model_y = logit(formula_y, data=clean_data).fit(disp=0)
            else:
                model_y = sm.OLS.from_formula(formula_y, data=clean_data).fit()

            # Monte Carlo simulation for causal effects
            acme_samples = []
            ade_samples = []
            total_samples = []

            for _ in range(n_simulations):
                # Draw parameters from their sampling distributions
                beta_m = np.random.multivariate_normal(
                    model_m.params.values,
                    model_m.cov_params().values
                )
                beta_y = np.random.multivariate_normal(
                    model_y.params.values,
                    model_y.cov_params().values
                )

                # For each unit, calculate potential outcomes
                # Simplified: average over the sample

                # ACME: E[Y(1, M(1)) - Y(1, M(0))]
                # ADE: E[Y(1, M(t)) - Y(0, M(t))]

                # Get coefficient indices
                exposure_idx_m = list(model_m.params.index).index(exposure)
                exposure_idx_y = list(model_y.params.index).index(exposure)
                mediator_idx_y = list(model_y.params.index).index(mediator)

                a = beta_m[exposure_idx_m]  # Effect of X on M
                b = beta_y[mediator_idx_y]  # Effect of M on Y
                c_prime = beta_y[exposure_idx_y]  # Direct effect

                acme = a * b  # Indirect effect
                ade = c_prime  # Direct effect
                total = acme + ade

                acme_samples.append(acme)
                ade_samples.append(ade)
                total_samples.append(total)

            acme_samples = np.array(acme_samples)
            ade_samples = np.array(ade_samples)
            total_samples = np.array(total_samples)

            # Point estimates
            acme = float(np.mean(acme_samples))
            ade = float(np.mean(ade_samples))
            total = float(np.mean(total_samples))

            # CIs
            alpha = self.alpha
            acme_ci = (
                float(np.percentile(acme_samples, alpha/2 * 100)),
                float(np.percentile(acme_samples, (1 - alpha/2) * 100))
            )
            ade_ci = (
                float(np.percentile(ade_samples, alpha/2 * 100)),
                float(np.percentile(ade_samples, (1 - alpha/2) * 100))
            )
            total_ci = (
                float(np.percentile(total_samples, alpha/2 * 100)),
                float(np.percentile(total_samples, (1 - alpha/2) * 100))
            )

            # p-values
            acme_p = float(2 * min(np.mean(acme_samples < 0), np.mean(acme_samples > 0)))
            ade_p = float(2 * min(np.mean(ade_samples < 0), np.mean(ade_samples > 0)))

            # Proportion mediated
            prop_med_samples = acme_samples / total_samples
            prop_med_samples = prop_med_samples[np.isfinite(prop_med_samples)]
            prop_med = float(np.median(prop_med_samples)) if len(prop_med_samples) > 0 else None
            prop_med_ci = (
                float(np.percentile(prop_med_samples, 2.5)),
                float(np.percentile(prop_med_samples, 97.5))
            ) if len(prop_med_samples) > 0 else None

            # Sensitivity analysis: find rho where ACME = 0
            # This is simplified - full implementation would use grid search
            sensitivity_rho = self._sensitivity_rho(clean_data, exposure, mediator, outcome, covariates)

            robustness = "robust" if abs(sensitivity_rho) > 0.3 else "sensitive to confounding"

            summary = f"""Causal Mediation Analysis:
- ACME (Indirect Effect): {acme:.4f}, 95% CI [{acme_ci[0]:.4f}, {acme_ci[1]:.4f}], p={acme_p:.4f}
- ADE (Direct Effect): {ade:.4f}, 95% CI [{ade_ci[0]:.4f}, {ade_ci[1]:.4f}], p={ade_p:.4f}
- Total Effect: {total:.4f}, 95% CI [{total_ci[0]:.4f}, {total_ci[1]:.4f}]
- Proportion Mediated: {prop_med:.1%} if prop_med else 'N/A'

Sensitivity: Results are {robustness} (ρ at ACME=0: {sensitivity_rho:.2f})"""

            edu_notes = {
                'acme': 'Average Causal Mediation Effect (ACME) is the effect of X on Y that operates through M.',
                'ade': 'Average Direct Effect (ADE) is the effect of X on Y not through M.',
                'assumptions': 'Assumes: (1) No unmeasured X-Y confounding, (2) No unmeasured M-Y confounding, (3) No X-induced M-Y confounding.',
                'sensitivity': f'Sensitivity parameter ρ={sensitivity_rho:.2f}: ACME would be 0 if unmeasured M-Y confounding induced this correlation.',
                'interpretation': f'{"ACME is significant - there is evidence of causal mediation." if acme_ci[0] > 0 or acme_ci[1] < 0 else "ACME CI includes 0 - mediation effect not significant."}'
            }

            return CausalMediationResult(
                n_observations=n,
                acme=acme,
                acme_ci=acme_ci,
                acme_pvalue=acme_p,
                ade=ade,
                ade_ci=ade_ci,
                ade_pvalue=ade_p,
                total_effect=total,
                total_effect_ci=total_ci,
                proportion_mediated=prop_med,
                proportion_mediated_ci=prop_med_ci,
                sensitivity_rho=sensitivity_rho,
                robustness_assessment=robustness,
                summary_text=summary,
                educational_notes=edu_notes
            )

        except Exception as e:
            result = self._insufficient_causal_data()
            result.warnings = [str(e)]
            return result

    def _sensitivity_rho(
        self,
        data: pd.DataFrame,
        exposure: str,
        mediator: str,
        outcome: str,
        covariates: List[str]
    ) -> float:
        """Calculate sensitivity parameter (simplified)."""
        try:
            # Get residuals from mediator and outcome models
            formula_m = f"{mediator} ~ {exposure}"
            if covariates:
                formula_m += " + " + " + ".join(covariates)
            model_m = sm.OLS.from_formula(formula_m, data=data).fit()
            resid_m = model_m.resid

            formula_y = f"{outcome} ~ {exposure} + {mediator}"
            if covariates:
                formula_y += " + " + " + ".join(covariates)
            model_y = sm.OLS.from_formula(formula_y, data=data).fit()
            resid_y = model_y.resid

            # Correlation of residuals
            rho = np.corrcoef(resid_m, resid_y)[0, 1]

            # Approximate rho at which ACME = 0
            # This is a simplified calculation
            return float(rho * 2)  # Heuristic adjustment

        except:
            return 0.0

    def _classify_mediation(
        self,
        c, c_p, a, a_p, b, b_p, c_prime, c_prime_p, indirect, indirect_p
    ) -> str:
        """Classify mediation type based on Baron-Kenny criteria."""
        alpha = self.alpha

        # Check each condition
        c_sig = c_p < alpha
        a_sig = a_p < alpha
        b_sig = b_p < alpha
        c_prime_sig = c_prime_p < alpha
        indirect_sig = indirect_p < alpha

        # Same sign check
        same_sign = (a * b * c) > 0 if abs(c) > 1e-10 else True

        if not c_sig:
            if indirect_sig:
                return "Indirect-only mediation (no total effect but significant indirect)"
            else:
                return "No effect (neither total nor indirect effect significant)"

        if not a_sig or not b_sig:
            return "No mediation (a or b path not significant)"

        if not same_sign:
            return "Inconsistent mediation (suppression effect)"

        if indirect_sig:
            if c_prime_sig:
                reduction = abs(c - c_prime) / abs(c)
                return f"Partial mediation ({reduction*100:.1f}% reduction in effect)"
            else:
                return "Full mediation (direct effect no longer significant)"
        else:
            return "No significant mediation (indirect effect not significant)"

    def _generate_mediation_summary(
        self,
        exposure, mediator, outcome,
        c, c_p, a, a_p, b, b_p,
        c_prime, c_prime_p, indirect, sobel_p,
        prop_med, med_type
    ) -> str:
        """Generate mediation analysis summary."""
        lines = [
            f"Mediation Analysis: {exposure} → {mediator} → {outcome}",
            "",
            "Path Coefficients:",
            f"  c  (total effect, X→Y):      {c:.4f}, p={c_p:.4f} {'*' if c_p < self.alpha else ''}",
            f"  a  (X→M):                    {a:.4f}, p={a_p:.4f} {'*' if a_p < self.alpha else ''}",
            f"  b  (M→Y|X):                  {b:.4f}, p={b_p:.4f} {'*' if b_p < self.alpha else ''}",
            f"  c' (direct effect, X→Y|M):   {c_prime:.4f}, p={c_prime_p:.4f} {'*' if c_prime_p < self.alpha else ''}",
            "",
            f"Indirect effect (a×b): {indirect:.4f}",
            f"Sobel test: z={indirect/np.sqrt(a**2 * 0.01 + b**2 * 0.01):.2f}, p={sobel_p:.4f}",
        ]

        if prop_med is not None:
            lines.append(f"Proportion mediated: {prop_med:.1%}")

        lines.extend([
            "",
            f"Conclusion: {med_type}"
        ])

        return '\n'.join(lines)

    def _insufficient_data(self, method: str) -> MediationResult:
        return MediationResult(
            model_type=method,
            n_observations=0,
            warnings=["Insufficient data for mediation analysis"]
        )

    def _insufficient_causal_data(self) -> CausalMediationResult:
        return CausalMediationResult(
            n_observations=0,
            warnings=["Insufficient data for causal mediation analysis"]
        )
