"""Enhanced power and sample size calculator.

Provides comprehensive power analysis for:
- Standard tests (t-test, ANOVA, chi-square)
- Regression models
- Survival analysis
- Complex designs (cluster RCT, non-inferiority, adaptive)
- Simulation-based power
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import brentq
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict, field
import warnings


@dataclass
class PowerResult:
    """Result of power analysis."""
    analysis_type: str
    test_type: str

    # Input parameters
    effect_size: Optional[float] = None
    sample_size: Optional[int] = None
    alpha: float = 0.05
    power: Optional[float] = None

    # Calculated value
    calculated_value: str = ""  # "power" or "sample_size"
    result: float = 0.0

    # Additional info
    n_per_group: Optional[int] = None
    total_n: Optional[int] = None
    allocation_ratio: float = 1.0

    # Design parameters
    design_parameters: Dict[str, Any] = field(default_factory=dict)

    # Interpretation
    summary_text: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)
    educational_notes: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class SimulationPowerResult:
    """Result of simulation-based power analysis."""
    n_simulations: int
    simulated_power: float
    power_ci: Tuple[float, float]

    # Simulation details
    rejection_rate: float
    mean_estimate: float
    std_estimate: float

    # Design
    design_description: str = ""

    summary_text: Optional[str] = None
    educational_notes: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['power_ci'] = list(self.power_ci)
        return {k: v for k, v in result.items() if v is not None}


class PowerCalculator:
    """Comprehensive power and sample size calculator."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    # =========================================================================
    # Standard Tests
    # =========================================================================

    def ttest_power(
        self,
        effect_size: Optional[float] = None,
        n: Optional[int] = None,
        alpha: float = 0.05,
        power: Optional[float] = None,
        alternative: str = "two-sided",
        ratio: float = 1.0
    ) -> PowerResult:
        """Power analysis for independent samples t-test.

        Provide any three of: effect_size (Cohen's d), n, alpha, power.
        The fourth will be calculated.

        Args:
            effect_size: Cohen's d
            n: Sample size per group
            alpha: Significance level
            power: Desired power
            alternative: "two-sided", "greater", "less"
            ratio: Allocation ratio (n2/n1)

        Returns:
            PowerResult object
        """
        # Determine what to calculate
        none_count = sum(x is None for x in [effect_size, n, power])
        if none_count != 1:
            raise ValueError("Provide exactly 3 of: effect_size, n, power")

        # Sided adjustment
        if alternative == "two-sided":
            alpha_adj = alpha / 2
        else:
            alpha_adj = alpha

        if n is None:
            # Calculate sample size
            n = self._ttest_n(effect_size, alpha_adj, power, ratio, alternative)
            calculated = "sample_size"
            result_value = n
        elif power is None:
            # Calculate power
            power = self._ttest_power(effect_size, n, alpha_adj, ratio, alternative)
            calculated = "power"
            result_value = power
        else:
            # Calculate effect size (for sensitivity analysis)
            effect_size = self._ttest_effect(n, alpha_adj, power, ratio, alternative)
            calculated = "effect_size"
            result_value = effect_size

        n1 = n
        n2 = int(n * ratio)
        total_n = n1 + n2

        summary = f"""Power Analysis: Independent Samples t-test
- Effect size (Cohen's d): {effect_size:.3f}
- Sample size per group: n1={n1}, n2={n2} (total={total_n})
- Significance level: {alpha}
- Power: {power:.3f}
- Alternative: {alternative}

{'Sample size calculated' if calculated == 'sample_size' else 'Power calculated' if calculated == 'power' else 'Effect size calculated'}: {result_value:.3f}"""

        recommendations = self._power_recommendations(power, effect_size, "ttest")

        edu_notes = {
            'cohens_d': 'Cohen\'s d: 0.2=small, 0.5=medium, 0.8=large effect',
            'power_meaning': f'Power of {power:.1%} means {power*100:.0f}% chance of detecting a true effect of d={effect_size}',
            'interpretation': f'Need n={n1} per group to detect d={effect_size} with {power*100:.0f}% power at α={alpha}'
        }

        return PowerResult(
            analysis_type="Power Analysis",
            test_type="Independent Samples t-test",
            effect_size=effect_size,
            sample_size=n,
            alpha=alpha,
            power=power,
            calculated_value=calculated,
            result=result_value,
            n_per_group=n,
            total_n=total_n,
            allocation_ratio=ratio,
            design_parameters={'alternative': alternative},
            summary_text=summary,
            recommendations=recommendations,
            educational_notes=edu_notes
        )

    def anova_power(
        self,
        effect_size: Optional[float] = None,
        n_per_group: Optional[int] = None,
        k_groups: int = 3,
        alpha: float = 0.05,
        power: Optional[float] = None
    ) -> PowerResult:
        """Power analysis for one-way ANOVA.

        Args:
            effect_size: Cohen's f (f = sqrt(eta_squared / (1 - eta_squared)))
            n_per_group: Sample size per group
            k_groups: Number of groups
            alpha: Significance level
            power: Desired power

        Returns:
            PowerResult object
        """
        none_count = sum(x is None for x in [effect_size, n_per_group, power])
        if none_count != 1:
            raise ValueError("Provide exactly 3 of: effect_size, n_per_group, power")

        df_between = k_groups - 1

        if n_per_group is None:
            n_per_group = self._anova_n(effect_size, k_groups, alpha, power)
            calculated = "sample_size"
            result_value = n_per_group
        elif power is None:
            total_n = n_per_group * k_groups
            df_within = total_n - k_groups
            ncp = effect_size ** 2 * total_n  # Non-centrality parameter

            # F critical value
            f_crit = stats.f.ppf(1 - alpha, df_between, df_within)

            # Power = P(F > f_crit | H1)
            power = 1 - stats.ncf.cdf(f_crit, df_between, df_within, ncp)
            calculated = "power"
            result_value = power
        else:
            # Calculate minimum detectable effect
            effect_size = self._anova_effect(n_per_group, k_groups, alpha, power)
            calculated = "effect_size"
            result_value = effect_size

        total_n = n_per_group * k_groups

        summary = f"""Power Analysis: One-way ANOVA
- Effect size (Cohen's f): {effect_size:.3f}
- Groups: {k_groups}
- Sample size per group: {n_per_group} (total={total_n})
- Significance level: {alpha}
- Power: {power:.3f}"""

        edu_notes = {
            'cohens_f': 'Cohen\'s f: 0.1=small, 0.25=medium, 0.4=large',
            'eta_squared': f'Equivalent η² ≈ {effect_size**2 / (1 + effect_size**2):.3f}',
            'interpretation': f'Need n={n_per_group} per group ({k_groups} groups) to detect f={effect_size}'
        }

        return PowerResult(
            analysis_type="Power Analysis",
            test_type="One-way ANOVA",
            effect_size=effect_size,
            sample_size=n_per_group,
            alpha=alpha,
            power=power,
            calculated_value=calculated,
            result=result_value,
            n_per_group=n_per_group,
            total_n=total_n,
            design_parameters={'k_groups': k_groups},
            summary_text=summary,
            educational_notes=edu_notes
        )

    def chi_square_power(
        self,
        effect_size: Optional[float] = None,
        n: Optional[int] = None,
        df: int = 1,
        alpha: float = 0.05,
        power: Optional[float] = None
    ) -> PowerResult:
        """Power analysis for chi-square test.

        Args:
            effect_size: Cohen's w
            n: Total sample size
            df: Degrees of freedom
            alpha: Significance level
            power: Desired power

        Returns:
            PowerResult object
        """
        none_count = sum(x is None for x in [effect_size, n, power])
        if none_count != 1:
            raise ValueError("Provide exactly 3 of: effect_size, n, power")

        if n is None:
            n = self._chi2_n(effect_size, df, alpha, power)
            calculated = "sample_size"
            result_value = n
        elif power is None:
            ncp = n * effect_size ** 2
            chi2_crit = stats.chi2.ppf(1 - alpha, df)
            power = 1 - stats.ncx2.cdf(chi2_crit, df, ncp)
            calculated = "power"
            result_value = power
        else:
            effect_size = self._chi2_effect(n, df, alpha, power)
            calculated = "effect_size"
            result_value = effect_size

        summary = f"""Power Analysis: Chi-square Test
- Effect size (Cohen's w): {effect_size:.3f}
- Total sample size: {n}
- Degrees of freedom: {df}
- Significance level: {alpha}
- Power: {power:.3f}"""

        edu_notes = {
            'cohens_w': 'Cohen\'s w: 0.1=small, 0.3=medium, 0.5=large',
            'for_2x2': 'For 2x2 table: w ≈ |OR-1|/(OR+1) for balanced design'
        }

        return PowerResult(
            analysis_type="Power Analysis",
            test_type="Chi-square Test",
            effect_size=effect_size,
            sample_size=n,
            alpha=alpha,
            power=power,
            calculated_value=calculated,
            result=result_value,
            total_n=n,
            design_parameters={'df': df},
            summary_text=summary,
            educational_notes=edu_notes
        )

    # =========================================================================
    # Regression
    # =========================================================================

    def regression_power(
        self,
        f2: Optional[float] = None,
        n: Optional[int] = None,
        n_predictors: int = 1,
        alpha: float = 0.05,
        power: Optional[float] = None
    ) -> PowerResult:
        """Power analysis for multiple regression.

        Args:
            f2: Cohen's f² effect size
            n: Total sample size
            n_predictors: Number of predictors
            alpha: Significance level
            power: Desired power

        Returns:
            PowerResult object
        """
        none_count = sum(x is None for x in [f2, n, power])
        if none_count != 1:
            raise ValueError("Provide exactly 3 of: f2, n, power")

        df1 = n_predictors
        df2_func = lambda n: n - n_predictors - 1

        if n is None:
            n = self._regression_n(f2, n_predictors, alpha, power)
            calculated = "sample_size"
            result_value = n
        elif power is None:
            df2 = df2_func(n)
            ncp = f2 * n
            f_crit = stats.f.ppf(1 - alpha, df1, df2)
            power = 1 - stats.ncf.cdf(f_crit, df1, df2, ncp)
            calculated = "power"
            result_value = power
        else:
            f2 = self._regression_effect(n, n_predictors, alpha, power)
            calculated = "effect_size"
            result_value = f2

        # Convert f2 to R2
        r2_equivalent = f2 / (1 + f2)

        summary = f"""Power Analysis: Multiple Regression
- Effect size (Cohen's f²): {f2:.3f}
- Equivalent R²: {r2_equivalent:.3f}
- Number of predictors: {n_predictors}
- Total sample size: {n}
- Significance level: {alpha}
- Power: {power:.3f}"""

        edu_notes = {
            'f2': 'Cohen\'s f²: 0.02=small, 0.15=medium, 0.35=large',
            'rule_of_thumb': f'Minimum n = 50 + 8*k = {50 + 8*n_predictors} for medium effect'
        }

        return PowerResult(
            analysis_type="Power Analysis",
            test_type="Multiple Regression",
            effect_size=f2,
            sample_size=n,
            alpha=alpha,
            power=power,
            calculated_value=calculated,
            result=result_value,
            total_n=n,
            design_parameters={'n_predictors': n_predictors, 'r2_equivalent': r2_equivalent},
            summary_text=summary,
            educational_notes=edu_notes
        )

    # =========================================================================
    # Survival Analysis
    # =========================================================================

    def survival_power(
        self,
        hazard_ratio: Optional[float] = None,
        n_events: Optional[int] = None,
        alpha: float = 0.05,
        power: Optional[float] = None,
        allocation_ratio: float = 1.0,
        two_sided: bool = True
    ) -> PowerResult:
        """Power analysis for Cox regression / log-rank test.

        Uses Schoenfeld formula for events needed.

        Args:
            hazard_ratio: Expected hazard ratio
            n_events: Number of events
            alpha: Significance level
            power: Desired power
            allocation_ratio: Ratio of group sizes (n2/n1)
            two_sided: Two-sided test

        Returns:
            PowerResult object
        """
        none_count = sum(x is None for x in [hazard_ratio, n_events, power])
        if none_count != 1:
            raise ValueError("Provide exactly 3 of: hazard_ratio, n_events, power")

        if two_sided:
            z_alpha = stats.norm.ppf(1 - alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)

        # Proportion in each arm
        p1 = 1 / (1 + allocation_ratio)
        p2 = allocation_ratio / (1 + allocation_ratio)

        if n_events is None:
            # Schoenfeld formula
            z_beta = stats.norm.ppf(power)
            log_hr = np.log(hazard_ratio)
            n_events = int(np.ceil((z_alpha + z_beta) ** 2 / (log_hr ** 2 * p1 * p2)))
            calculated = "sample_size"
            result_value = n_events
        elif power is None:
            log_hr = np.log(hazard_ratio)
            se = 1 / np.sqrt(n_events * p1 * p2)
            z = abs(log_hr) / se
            power = stats.norm.cdf(z - z_alpha) + stats.norm.cdf(-z - z_alpha)
            calculated = "power"
            result_value = power
        else:
            # Calculate detectable HR
            z_beta = stats.norm.ppf(power)
            log_hr = (z_alpha + z_beta) / np.sqrt(n_events * p1 * p2)
            hazard_ratio = np.exp(abs(log_hr))
            calculated = "effect_size"
            result_value = hazard_ratio

        summary = f"""Power Analysis: Survival Analysis (Log-rank / Cox)
- Hazard ratio: {hazard_ratio:.2f}
- Events needed: {n_events}
- Allocation ratio: {allocation_ratio}
- Significance level: {alpha}
- Power: {power:.3f}
- {'Two' if two_sided else 'One'}-sided test"""

        edu_notes = {
            'schoenfeld': 'Using Schoenfeld formula for sample size calculation',
            'events_vs_subjects': 'Events are more important than subjects for power. Account for expected event rate.',
            'example': f'If 50% event rate expected, need ~{int(n_events / 0.5)} subjects total'
        }

        return PowerResult(
            analysis_type="Power Analysis",
            test_type="Survival Analysis (Log-rank/Cox)",
            effect_size=hazard_ratio,
            sample_size=n_events,
            alpha=alpha,
            power=power,
            calculated_value=calculated,
            result=result_value,
            allocation_ratio=allocation_ratio,
            design_parameters={'two_sided': two_sided},
            summary_text=summary,
            educational_notes=edu_notes
        )

    # =========================================================================
    # Complex Designs
    # =========================================================================

    def cluster_rct_power(
        self,
        effect_size: Optional[float] = None,
        n_clusters: Optional[int] = None,
        cluster_size: int = 20,
        icc: float = 0.05,
        alpha: float = 0.05,
        power: Optional[float] = None
    ) -> PowerResult:
        """Power analysis for cluster randomized controlled trial.

        Args:
            effect_size: Cohen's d
            n_clusters: Number of clusters per arm
            cluster_size: Average subjects per cluster
            icc: Intraclass correlation coefficient
            alpha: Significance level
            power: Desired power

        Returns:
            PowerResult object
        """
        # Design effect
        deff = 1 + (cluster_size - 1) * icc

        # Effective sample size per arm
        if n_clusters is not None:
            n_eff = n_clusters * cluster_size / deff
        else:
            n_eff = None

        # Use t-test power with effective n
        if n_clusters is None:
            # First calculate individual-level n
            ttest_result = self.ttest_power(
                effect_size=effect_size,
                n=None,
                alpha=alpha,
                power=power
            )
            n_individual = ttest_result.n_per_group

            # Inflate for clustering
            n_clusters = int(np.ceil(n_individual * deff / cluster_size))
            calculated = "sample_size"
            result_value = n_clusters
        elif power is None:
            n_eff_per_arm = n_clusters * cluster_size / deff
            ttest_result = self.ttest_power(
                effect_size=effect_size,
                n=int(n_eff_per_arm),
                alpha=alpha,
                power=None
            )
            power = ttest_result.power
            calculated = "power"
            result_value = power
        else:
            # Calculate detectable effect
            n_eff_per_arm = n_clusters * cluster_size / deff
            ttest_result = self.ttest_power(
                effect_size=None,
                n=int(n_eff_per_arm),
                alpha=alpha,
                power=power
            )
            effect_size = ttest_result.effect_size
            calculated = "effect_size"
            result_value = effect_size

        total_clusters = n_clusters * 2
        total_subjects = total_clusters * cluster_size

        summary = f"""Power Analysis: Cluster RCT
- Effect size (Cohen's d): {effect_size:.3f}
- Clusters per arm: {n_clusters} (total={total_clusters})
- Subjects per cluster: {cluster_size}
- ICC: {icc}
- Design effect: {deff:.2f}
- Total subjects: {total_subjects}
- Significance level: {alpha}
- Power: {power:.3f}"""

        edu_notes = {
            'design_effect': f'Design effect of {deff:.2f} inflates required sample by this factor',
            'icc_impact': f'Higher ICC requires more clusters. With ICC={icc}, each cluster contributes {cluster_size/deff:.1f} effective subjects',
            'clusters_vs_size': 'Generally better to have more clusters than larger clusters'
        }

        return PowerResult(
            analysis_type="Power Analysis",
            test_type="Cluster RCT",
            effect_size=effect_size,
            sample_size=n_clusters,
            alpha=alpha,
            power=power,
            calculated_value=calculated,
            result=result_value,
            n_per_group=n_clusters,
            total_n=total_subjects,
            design_parameters={
                'cluster_size': cluster_size,
                'icc': icc,
                'design_effect': deff,
                'total_clusters': total_clusters
            },
            summary_text=summary,
            educational_notes=edu_notes
        )

    def non_inferiority_power(
        self,
        expected_difference: float = 0.0,
        margin: float = 0.1,
        sigma: float = 1.0,
        n: Optional[int] = None,
        alpha: float = 0.025,
        power: Optional[float] = None,
        allocation_ratio: float = 1.0
    ) -> PowerResult:
        """Power analysis for non-inferiority trial.

        Args:
            expected_difference: Expected true difference (new - reference)
            margin: Non-inferiority margin (positive value)
            sigma: Standard deviation
            n: Sample size per arm
            alpha: One-sided significance level
            power: Desired power
            allocation_ratio: n_reference / n_new

        Returns:
            PowerResult object
        """
        none_count = sum(x is None for x in [n, power])
        if none_count != 1:
            raise ValueError("Provide exactly one of: n, power")

        z_alpha = stats.norm.ppf(1 - alpha)

        # Standard error multiplier
        se_mult = np.sqrt(1 + 1 / allocation_ratio)

        if n is None:
            z_beta = stats.norm.ppf(power)
            # Sample size formula for non-inferiority
            n = int(np.ceil(
                2 * (sigma ** 2) * (z_alpha + z_beta) ** 2 /
                (margin - expected_difference) ** 2
            ))
            calculated = "sample_size"
            result_value = n
        else:
            se = sigma * se_mult / np.sqrt(n)
            z = (margin - expected_difference) / se
            power = stats.norm.cdf(z - z_alpha)
            calculated = "power"
            result_value = power

        n_new = n
        n_ref = int(n * allocation_ratio)
        total_n = n_new + n_ref

        summary = f"""Power Analysis: Non-Inferiority Trial
- Non-inferiority margin: {margin}
- Expected difference: {expected_difference}
- Standard deviation: {sigma}
- Sample size per arm: n_new={n_new}, n_ref={n_ref}
- One-sided alpha: {alpha}
- Power: {power:.3f}

Note: Non-inferiority declared if lower CI bound > -{margin}"""

        edu_notes = {
            'margin_choice': 'Margin should be based on clinically acceptable difference',
            'one_sided': 'Non-inferiority uses one-sided alpha (typically 0.025)',
            'interpretation': f'Trial powered to declare non-inferiority if new treatment is within {margin} of reference'
        }

        return PowerResult(
            analysis_type="Power Analysis",
            test_type="Non-Inferiority Trial",
            effect_size=margin,
            sample_size=n,
            alpha=alpha,
            power=power,
            calculated_value=calculated,
            result=result_value,
            n_per_group=n,
            total_n=total_n,
            allocation_ratio=allocation_ratio,
            design_parameters={
                'margin': margin,
                'expected_difference': expected_difference,
                'sigma': sigma
            },
            summary_text=summary,
            educational_notes=edu_notes
        )

    # =========================================================================
    # Simulation-Based Power
    # =========================================================================

    def simulation_power(
        self,
        data_generator: Callable,
        analysis_function: Callable,
        n_simulations: int = 1000,
        alpha: float = 0.05,
        seed: Optional[int] = None
    ) -> SimulationPowerResult:
        """Simulation-based power analysis for complex scenarios.

        Args:
            data_generator: Function that generates one dataset
            analysis_function: Function that returns p-value from data
            n_simulations: Number of simulations
            alpha: Significance level
            seed: Random seed

        Returns:
            SimulationPowerResult object
        """
        if seed is None:
            seed = self.random_state
        np.random.seed(seed)

        pvalues = []
        estimates = []

        for _ in range(n_simulations):
            try:
                data = data_generator()
                result = analysis_function(data)

                if isinstance(result, dict):
                    pvalues.append(result.get('pvalue', 1.0))
                    estimates.append(result.get('estimate', 0.0))
                elif isinstance(result, (float, int)):
                    pvalues.append(float(result))
                else:
                    pvalues.append(float(result))

            except:
                pvalues.append(1.0)  # Conservative: treat errors as non-significant

        pvalues = np.array(pvalues)
        estimates = np.array(estimates) if estimates else None

        # Calculate power
        rejections = pvalues < alpha
        power = np.mean(rejections)

        # 95% CI for power (binomial)
        se_power = np.sqrt(power * (1 - power) / n_simulations)
        power_ci = (
            max(0, power - 1.96 * se_power),
            min(1, power + 1.96 * se_power)
        )

        summary = f"""Simulation-Based Power Analysis:
- Simulations: {n_simulations}
- Simulated power: {power:.3f} (95% CI: [{power_ci[0]:.3f}, {power_ci[1]:.3f}])
- Rejection rate at α={alpha}: {power*100:.1f}%"""

        if estimates is not None:
            mean_est = np.mean(estimates)
            std_est = np.std(estimates)
            summary += f"\n- Mean estimate: {mean_est:.4f} (SD: {std_est:.4f})"
        else:
            mean_est = 0
            std_est = 0

        edu_notes = {
            'simulation_advantage': 'Simulation can handle complex designs, dependencies, and non-standard analyses',
            'n_simulations': f'{n_simulations} simulations gives SE of power ≈ {se_power:.3f}',
            'interpretation': f'Based on simulations, expect to reject H0 {power*100:.0f}% of the time'
        }

        return SimulationPowerResult(
            n_simulations=n_simulations,
            simulated_power=power,
            power_ci=power_ci,
            rejection_rate=power,
            mean_estimate=mean_est,
            std_estimate=std_est,
            summary_text=summary,
            educational_notes=edu_notes
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _ttest_power(self, d, n, alpha, ratio, alternative):
        """Calculate t-test power."""
        df = n + int(n * ratio) - 2
        se = np.sqrt(1/n + 1/(n * ratio))
        ncp = d / se

        t_crit = stats.t.ppf(1 - alpha, df)
        if alternative == "two-sided":
            power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
        elif alternative == "greater":
            power = 1 - stats.nct.cdf(t_crit, df, ncp)
        else:
            power = stats.nct.cdf(-t_crit, df, ncp)

        return float(power)

    def _ttest_n(self, d, alpha, power, ratio, alternative):
        """Calculate t-test sample size."""
        def power_diff(n):
            return self._ttest_power(d, int(n), alpha, ratio, alternative) - power

        try:
            n = int(np.ceil(brentq(power_diff, 2, 10000)))
        except:
            n = 100  # Default
        return n

    def _ttest_effect(self, n, alpha, power, ratio, alternative):
        """Calculate detectable effect size."""
        def power_diff(d):
            return self._ttest_power(d, n, alpha, ratio, alternative) - power

        try:
            d = brentq(power_diff, 0.01, 5)
        except:
            d = 0.5
        return float(d)

    def _anova_n(self, f, k, alpha, power):
        """Calculate ANOVA sample size per group."""
        def power_diff(n):
            n = int(n)
            total_n = n * k
            df_between = k - 1
            df_within = total_n - k
            ncp = f ** 2 * total_n
            f_crit = stats.f.ppf(1 - alpha, df_between, df_within)
            return 1 - stats.ncf.cdf(f_crit, df_between, df_within, ncp) - power

        try:
            n = int(np.ceil(brentq(power_diff, 3, 1000)))
        except:
            n = 30
        return n

    def _anova_effect(self, n, k, alpha, power):
        """Calculate detectable effect size for ANOVA."""
        def power_diff(f):
            total_n = n * k
            df_between = k - 1
            df_within = total_n - k
            ncp = f ** 2 * total_n
            f_crit = stats.f.ppf(1 - alpha, df_between, df_within)
            return 1 - stats.ncf.cdf(f_crit, df_between, df_within, ncp) - power

        try:
            f = brentq(power_diff, 0.01, 2)
        except:
            f = 0.25
        return float(f)

    def _chi2_n(self, w, df, alpha, power):
        """Calculate chi-square sample size."""
        def power_diff(n):
            n = int(n)
            ncp = n * w ** 2
            chi2_crit = stats.chi2.ppf(1 - alpha, df)
            return 1 - stats.ncx2.cdf(chi2_crit, df, ncp) - power

        try:
            n = int(np.ceil(brentq(power_diff, 10, 10000)))
        except:
            n = 100
        return n

    def _chi2_effect(self, n, df, alpha, power):
        """Calculate detectable effect size for chi-square."""
        def power_diff(w):
            ncp = n * w ** 2
            chi2_crit = stats.chi2.ppf(1 - alpha, df)
            return 1 - stats.ncx2.cdf(chi2_crit, df, ncp) - power

        try:
            w = brentq(power_diff, 0.01, 2)
        except:
            w = 0.3
        return float(w)

    def _regression_n(self, f2, k, alpha, power):
        """Calculate regression sample size."""
        def power_diff(n):
            n = int(n)
            df1 = k
            df2 = n - k - 1
            if df2 < 1:
                return -1
            ncp = f2 * n
            f_crit = stats.f.ppf(1 - alpha, df1, df2)
            return 1 - stats.ncf.cdf(f_crit, df1, df2, ncp) - power

        try:
            n = int(np.ceil(brentq(power_diff, k + 5, 5000)))
        except:
            n = 50 + 8 * k
        return n

    def _regression_effect(self, n, k, alpha, power):
        """Calculate detectable effect size for regression."""
        def power_diff(f2):
            df1 = k
            df2 = n - k - 1
            ncp = f2 * n
            f_crit = stats.f.ppf(1 - alpha, df1, df2)
            return 1 - stats.ncf.cdf(f_crit, df1, df2, ncp) - power

        try:
            f2 = brentq(power_diff, 0.001, 1)
        except:
            f2 = 0.15
        return float(f2)

    def _power_recommendations(self, power, effect_size, test_type):
        """Generate power recommendations."""
        recommendations = []

        if power < 0.5:
            recommendations.append("WARNING: Power < 50% - study severely underpowered")
        elif power < 0.8:
            recommendations.append("Consider increasing sample size to achieve 80% power")

        if effect_size is not None:
            if test_type == "ttest":
                if effect_size < 0.2:
                    recommendations.append("Small effect size - requires large sample")
                elif effect_size > 0.8:
                    recommendations.append("Large effect assumed - verify this is realistic")

        return recommendations
