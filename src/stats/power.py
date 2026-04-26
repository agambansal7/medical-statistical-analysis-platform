"""Power analysis and sample size calculation module."""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class PowerResult:
    """Result of power analysis."""
    analysis_type: str
    calculated_value: float
    calculated_parameter: str  # 'power', 'sample_size', or 'effect_size'
    alpha: float
    power: Optional[float] = None
    effect_size: Optional[float] = None
    sample_size: Optional[int] = None
    details: Dict[str, Any] = None
    interpretation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class PowerAnalysis:
    """Power analysis and sample size calculations."""

    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level

    # ==================== T-TESTS ====================

    def ttest_ind_power(self, effect_size: float = None,
                        n: int = None,
                        power: float = None,
                        ratio: float = 1.0,
                        alternative: str = 'two-sided') -> PowerResult:
        """Power analysis for independent samples t-test.

        Provide two of three parameters (effect_size, n, power) to calculate the third.

        Args:
            effect_size: Cohen's d
            n: Sample size per group
            power: Desired power
            ratio: Ratio of n2/n1
            alternative: 'two-sided' or 'one-sided'

        Returns:
            PowerResult object
        """
        try:
            from statsmodels.stats.power import TTestIndPower
            analysis = TTestIndPower()

            if alternative == 'two-sided':
                alt = 'two-sided'
            else:
                alt = 'larger'

            if effect_size is None:
                # Calculate required effect size
                result = analysis.solve_power(
                    effect_size=None, nobs1=n, alpha=self.alpha,
                    power=power, ratio=ratio, alternative=alt
                )
                return PowerResult(
                    analysis_type="Independent t-test power",
                    calculated_value=float(result),
                    calculated_parameter="effect_size",
                    alpha=self.alpha,
                    power=power,
                    effect_size=float(result),
                    sample_size=n,
                    interpretation=f"Minimum detectable effect size: d = {result:.3f}"
                )

            elif n is None:
                # Calculate required sample size
                result = analysis.solve_power(
                    effect_size=effect_size, nobs1=None, alpha=self.alpha,
                    power=power, ratio=ratio, alternative=alt
                )
                n_per_group = int(np.ceil(result))
                n_total = int(np.ceil(n_per_group * (1 + ratio)))

                return PowerResult(
                    analysis_type="Independent t-test sample size",
                    calculated_value=n_per_group,
                    calculated_parameter="sample_size",
                    alpha=self.alpha,
                    power=power,
                    effect_size=effect_size,
                    sample_size=n_per_group,
                    details={'n_per_group': n_per_group, 'n_total': n_total, 'ratio': ratio},
                    interpretation=f"Required: {n_per_group} per group ({n_total} total)"
                )

            else:
                # Calculate achieved power
                result = analysis.solve_power(
                    effect_size=effect_size, nobs1=n, alpha=self.alpha,
                    power=None, ratio=ratio, alternative=alt
                )
                return PowerResult(
                    analysis_type="Independent t-test power",
                    calculated_value=float(result),
                    calculated_parameter="power",
                    alpha=self.alpha,
                    power=float(result),
                    effect_size=effect_size,
                    sample_size=n,
                    interpretation=f"Achieved power: {result*100:.1f}%"
                )

        except Exception as e:
            return PowerResult(
                analysis_type="Independent t-test power",
                calculated_value=np.nan,
                calculated_parameter="error",
                alpha=self.alpha,
                interpretation=str(e)
            )

    def ttest_paired_power(self, effect_size: float = None,
                          n: int = None,
                          power: float = None,
                          alternative: str = 'two-sided') -> PowerResult:
        """Power analysis for paired samples t-test.

        Args:
            effect_size: Cohen's d for paired samples
            n: Number of pairs
            power: Desired power
            alternative: 'two-sided' or 'one-sided'

        Returns:
            PowerResult object
        """
        try:
            from statsmodels.stats.power import TTestPower
            analysis = TTestPower()

            if alternative == 'two-sided':
                alt = 'two-sided'
            else:
                alt = 'larger'

            if effect_size is None:
                result = analysis.solve_power(
                    effect_size=None, nobs=n, alpha=self.alpha,
                    power=power, alternative=alt
                )
                return PowerResult(
                    analysis_type="Paired t-test power",
                    calculated_value=float(result),
                    calculated_parameter="effect_size",
                    alpha=self.alpha,
                    power=power,
                    effect_size=float(result),
                    sample_size=n
                )

            elif n is None:
                result = analysis.solve_power(
                    effect_size=effect_size, nobs=None, alpha=self.alpha,
                    power=power, alternative=alt
                )
                n_pairs = int(np.ceil(result))
                return PowerResult(
                    analysis_type="Paired t-test sample size",
                    calculated_value=n_pairs,
                    calculated_parameter="sample_size",
                    alpha=self.alpha,
                    power=power,
                    effect_size=effect_size,
                    sample_size=n_pairs,
                    interpretation=f"Required: {n_pairs} pairs"
                )

            else:
                result = analysis.solve_power(
                    effect_size=effect_size, nobs=n, alpha=self.alpha,
                    power=None, alternative=alt
                )
                return PowerResult(
                    analysis_type="Paired t-test power",
                    calculated_value=float(result),
                    calculated_parameter="power",
                    alpha=self.alpha,
                    power=float(result),
                    effect_size=effect_size,
                    sample_size=n
                )

        except Exception as e:
            return PowerResult(
                analysis_type="Paired t-test power",
                calculated_value=np.nan,
                calculated_parameter="error",
                alpha=self.alpha,
                interpretation=str(e)
            )

    # ==================== ANOVA ====================

    def anova_power(self, effect_size: float = None,
                   n_groups: int = 3,
                   n_per_group: int = None,
                   power: float = None) -> PowerResult:
        """Power analysis for one-way ANOVA.

        Args:
            effect_size: Cohen's f
            n_groups: Number of groups
            n_per_group: Sample size per group
            power: Desired power

        Returns:
            PowerResult object
        """
        try:
            from statsmodels.stats.power import FTestAnovaPower
            analysis = FTestAnovaPower()

            if effect_size is None:
                result = analysis.solve_power(
                    effect_size=None, nobs=n_per_group, alpha=self.alpha,
                    power=power, k_groups=n_groups
                )
                return PowerResult(
                    analysis_type="ANOVA power",
                    calculated_value=float(result),
                    calculated_parameter="effect_size",
                    alpha=self.alpha,
                    power=power,
                    effect_size=float(result),
                    sample_size=n_per_group,
                    details={'n_groups': n_groups}
                )

            elif n_per_group is None:
                result = analysis.solve_power(
                    effect_size=effect_size, nobs=None, alpha=self.alpha,
                    power=power, k_groups=n_groups
                )
                n = int(np.ceil(result))
                return PowerResult(
                    analysis_type="ANOVA sample size",
                    calculated_value=n,
                    calculated_parameter="sample_size",
                    alpha=self.alpha,
                    power=power,
                    effect_size=effect_size,
                    sample_size=n,
                    details={'n_groups': n_groups, 'n_total': n * n_groups},
                    interpretation=f"Required: {n} per group ({n * n_groups} total)"
                )

            else:
                result = analysis.solve_power(
                    effect_size=effect_size, nobs=n_per_group, alpha=self.alpha,
                    power=None, k_groups=n_groups
                )
                return PowerResult(
                    analysis_type="ANOVA power",
                    calculated_value=float(result),
                    calculated_parameter="power",
                    alpha=self.alpha,
                    power=float(result),
                    effect_size=effect_size,
                    sample_size=n_per_group,
                    details={'n_groups': n_groups}
                )

        except Exception as e:
            return PowerResult(
                analysis_type="ANOVA power",
                calculated_value=np.nan,
                calculated_parameter="error",
                alpha=self.alpha,
                interpretation=str(e)
            )

    # ==================== CORRELATION ====================

    def correlation_power(self, r: float = None,
                         n: int = None,
                         power: float = None,
                         alternative: str = 'two-sided') -> PowerResult:
        """Power analysis for correlation.

        Args:
            r: Expected correlation coefficient
            n: Sample size
            power: Desired power
            alternative: 'two-sided' or 'one-sided'

        Returns:
            PowerResult object
        """
        try:
            from statsmodels.stats.power import NormalIndPower

            # Convert r to z (Fisher transformation)
            if r is not None:
                effect_size = np.arctanh(r)
            else:
                effect_size = None

            analysis = NormalIndPower()

            if r is None:
                z = analysis.solve_power(
                    effect_size=None, nobs=n, alpha=self.alpha,
                    power=power, alternative='two-sided' if alternative == 'two-sided' else 'larger'
                )
                r_result = np.tanh(z)
                return PowerResult(
                    analysis_type="Correlation power",
                    calculated_value=float(r_result),
                    calculated_parameter="effect_size",
                    alpha=self.alpha,
                    power=power,
                    effect_size=float(r_result),
                    sample_size=n,
                    interpretation=f"Minimum detectable r: {r_result:.3f}"
                )

            elif n is None:
                result = analysis.solve_power(
                    effect_size=effect_size, nobs=None, alpha=self.alpha,
                    power=power, alternative='two-sided' if alternative == 'two-sided' else 'larger'
                )
                n_required = int(np.ceil(result)) + 3  # Add 3 for Fisher transformation
                return PowerResult(
                    analysis_type="Correlation sample size",
                    calculated_value=n_required,
                    calculated_parameter="sample_size",
                    alpha=self.alpha,
                    power=power,
                    effect_size=r,
                    sample_size=n_required,
                    interpretation=f"Required sample size: {n_required}"
                )

            else:
                result = analysis.solve_power(
                    effect_size=effect_size, nobs=n-3, alpha=self.alpha,
                    power=None, alternative='two-sided' if alternative == 'two-sided' else 'larger'
                )
                return PowerResult(
                    analysis_type="Correlation power",
                    calculated_value=float(result),
                    calculated_parameter="power",
                    alpha=self.alpha,
                    power=float(result),
                    effect_size=r,
                    sample_size=n,
                    interpretation=f"Achieved power: {result*100:.1f}%"
                )

        except Exception as e:
            return PowerResult(
                analysis_type="Correlation power",
                calculated_value=np.nan,
                calculated_parameter="error",
                alpha=self.alpha,
                interpretation=str(e)
            )

    # ==================== PROPORTIONS ====================

    def proportions_power(self, p1: float, p2: float,
                         n: int = None,
                         power: float = None,
                         ratio: float = 1.0) -> PowerResult:
        """Power analysis for comparing two proportions.

        Args:
            p1: Proportion in group 1
            p2: Proportion in group 2
            n: Sample size per group
            power: Desired power
            ratio: Ratio of n2/n1

        Returns:
            PowerResult object
        """
        try:
            from statsmodels.stats.power import NormalIndPower

            # Effect size (Cohen's h)
            h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

            analysis = NormalIndPower()

            if n is None:
                result = analysis.solve_power(
                    effect_size=abs(h), nobs=None, alpha=self.alpha,
                    power=power, ratio=ratio, alternative='two-sided'
                )
                n_per_group = int(np.ceil(result))
                n_total = int(np.ceil(n_per_group * (1 + ratio)))

                return PowerResult(
                    analysis_type="Proportions sample size",
                    calculated_value=n_per_group,
                    calculated_parameter="sample_size",
                    alpha=self.alpha,
                    power=power,
                    effect_size=abs(h),
                    sample_size=n_per_group,
                    details={
                        'p1': p1, 'p2': p2, 'cohens_h': float(h),
                        'n_per_group': n_per_group, 'n_total': n_total
                    },
                    interpretation=f"Required: {n_per_group} per group ({n_total} total)"
                )

            else:
                result = analysis.solve_power(
                    effect_size=abs(h), nobs=n, alpha=self.alpha,
                    power=None, ratio=ratio, alternative='two-sided'
                )
                return PowerResult(
                    analysis_type="Proportions power",
                    calculated_value=float(result),
                    calculated_parameter="power",
                    alpha=self.alpha,
                    power=float(result),
                    effect_size=abs(h),
                    sample_size=n,
                    details={'p1': p1, 'p2': p2, 'cohens_h': float(h)},
                    interpretation=f"Achieved power: {result*100:.1f}%"
                )

        except Exception as e:
            return PowerResult(
                analysis_type="Proportions power",
                calculated_value=np.nan,
                calculated_parameter="error",
                alpha=self.alpha,
                interpretation=str(e)
            )

    # ==================== CHI-SQUARE ====================

    def chi_square_power(self, effect_size: float = None,
                        n: int = None,
                        power: float = None,
                        df: int = 1) -> PowerResult:
        """Power analysis for chi-square test.

        Args:
            effect_size: Cohen's w
            n: Total sample size
            power: Desired power
            df: Degrees of freedom

        Returns:
            PowerResult object
        """
        try:
            from statsmodels.stats.power import GofChisquarePower
            analysis = GofChisquarePower()

            if effect_size is None:
                result = analysis.solve_power(
                    effect_size=None, nobs=n, alpha=self.alpha,
                    power=power, n_bins=df+1
                )
                return PowerResult(
                    analysis_type="Chi-square power",
                    calculated_value=float(result),
                    calculated_parameter="effect_size",
                    alpha=self.alpha,
                    power=power,
                    effect_size=float(result),
                    sample_size=n,
                    details={'df': df}
                )

            elif n is None:
                result = analysis.solve_power(
                    effect_size=effect_size, nobs=None, alpha=self.alpha,
                    power=power, n_bins=df+1
                )
                return PowerResult(
                    analysis_type="Chi-square sample size",
                    calculated_value=int(np.ceil(result)),
                    calculated_parameter="sample_size",
                    alpha=self.alpha,
                    power=power,
                    effect_size=effect_size,
                    sample_size=int(np.ceil(result)),
                    details={'df': df},
                    interpretation=f"Required total sample size: {int(np.ceil(result))}"
                )

            else:
                result = analysis.solve_power(
                    effect_size=effect_size, nobs=n, alpha=self.alpha,
                    power=None, n_bins=df+1
                )
                return PowerResult(
                    analysis_type="Chi-square power",
                    calculated_value=float(result),
                    calculated_parameter="power",
                    alpha=self.alpha,
                    power=float(result),
                    effect_size=effect_size,
                    sample_size=n,
                    details={'df': df}
                )

        except Exception as e:
            return PowerResult(
                analysis_type="Chi-square power",
                calculated_value=np.nan,
                calculated_parameter="error",
                alpha=self.alpha,
                interpretation=str(e)
            )

    # ==================== SURVIVAL ANALYSIS ====================

    def survival_logrank_power(self, hazard_ratio: float,
                               n_events: int = None,
                               power: float = None,
                               p1: float = 0.5) -> PowerResult:
        """Power analysis for log-rank test (survival analysis).

        Args:
            hazard_ratio: Expected hazard ratio
            n_events: Number of events required
            power: Desired power
            p1: Proportion in group 1 (default 0.5 = equal groups)

        Returns:
            PowerResult object
        """
        # Schoenfeld formula
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)

        if n_events is None and power is not None:
            z_beta = stats.norm.ppf(power)
            log_hr = np.log(hazard_ratio)
            d = 4 * (z_alpha + z_beta)**2 / (log_hr**2)
            n_events = int(np.ceil(d))

            return PowerResult(
                analysis_type="Log-rank test sample size",
                calculated_value=n_events,
                calculated_parameter="sample_size",
                alpha=self.alpha,
                power=power,
                effect_size=hazard_ratio,
                sample_size=n_events,
                details={'hazard_ratio': hazard_ratio},
                interpretation=f"Required events: {n_events}"
            )

        elif power is None and n_events is not None:
            log_hr = np.log(hazard_ratio)
            z_beta = (np.sqrt(n_events) * abs(log_hr) / 2) - z_alpha
            achieved_power = stats.norm.cdf(z_beta)

            return PowerResult(
                analysis_type="Log-rank test power",
                calculated_value=float(achieved_power),
                calculated_parameter="power",
                alpha=self.alpha,
                power=float(achieved_power),
                effect_size=hazard_ratio,
                sample_size=n_events,
                details={'hazard_ratio': hazard_ratio}
            )

        return PowerResult(
            analysis_type="Log-rank test power",
            calculated_value=np.nan,
            calculated_parameter="error",
            alpha=self.alpha,
            interpretation="Provide either n_events or power"
        )

    # ==================== REGRESSION ====================

    def regression_power(self, f2: float = None,
                        n: int = None,
                        n_predictors: int = 1,
                        power: float = None) -> PowerResult:
        """Power analysis for multiple regression.

        Args:
            f2: Cohen's f² (effect size)
            n: Total sample size
            n_predictors: Number of predictors
            power: Desired power

        Returns:
            PowerResult object
        """
        try:
            from statsmodels.stats.power import FTestPower
            analysis = FTestPower()

            # df_num = n_predictors, df_denom = n - n_predictors - 1
            if f2 is None:
                result = analysis.solve_power(
                    effect_size=None, df_num=n_predictors,
                    df_denom=n-n_predictors-1, alpha=self.alpha, power=power
                )
                return PowerResult(
                    analysis_type="Regression power",
                    calculated_value=float(result),
                    calculated_parameter="effect_size",
                    alpha=self.alpha,
                    power=power,
                    effect_size=float(result),
                    sample_size=n,
                    details={'n_predictors': n_predictors}
                )

            elif n is None:
                # Iterative solution
                for n_try in range(n_predictors + 10, 10000):
                    try:
                        pwr = analysis.solve_power(
                            effect_size=np.sqrt(f2), df_num=n_predictors,
                            df_denom=n_try-n_predictors-1, alpha=self.alpha, power=None
                        )
                        if pwr >= power:
                            return PowerResult(
                                analysis_type="Regression sample size",
                                calculated_value=n_try,
                                calculated_parameter="sample_size",
                                alpha=self.alpha,
                                power=power,
                                effect_size=f2,
                                sample_size=n_try,
                                details={'n_predictors': n_predictors},
                                interpretation=f"Required sample size: {n_try}"
                            )
                    except:
                        continue

                return PowerResult(
                    analysis_type="Regression sample size",
                    calculated_value=np.nan,
                    calculated_parameter="error",
                    alpha=self.alpha,
                    interpretation="Could not find required sample size"
                )

            else:
                result = analysis.solve_power(
                    effect_size=np.sqrt(f2), df_num=n_predictors,
                    df_denom=n-n_predictors-1, alpha=self.alpha, power=None
                )
                return PowerResult(
                    analysis_type="Regression power",
                    calculated_value=float(result),
                    calculated_parameter="power",
                    alpha=self.alpha,
                    power=float(result),
                    effect_size=f2,
                    sample_size=n,
                    details={'n_predictors': n_predictors}
                )

        except Exception as e:
            return PowerResult(
                analysis_type="Regression power",
                calculated_value=np.nan,
                calculated_parameter="error",
                alpha=self.alpha,
                interpretation=str(e)
            )
