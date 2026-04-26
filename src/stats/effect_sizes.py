"""Effect size calculation module."""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class EffectSizeResult:
    """Result of effect size calculation."""
    name: str
    value: float
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    interpretation: Optional[str] = None
    variance: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class EffectSizeCalculator:
    """Calculate various effect sizes."""

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level

    # ==================== STANDARDIZED MEAN DIFFERENCES ====================

    def cohens_d(self, group1: pd.Series, group2: pd.Series,
                 pooled: bool = True) -> EffectSizeResult:
        """Cohen's d for two independent groups.

        Args:
            group1: First group data
            group2: Second group data
            pooled: Use pooled standard deviation

        Returns:
            EffectSizeResult object
        """
        g1 = group1.dropna()
        g2 = group2.dropna()
        n1, n2 = len(g1), len(g2)

        if n1 < 2 or n2 < 2:
            return EffectSizeResult(name="Cohen's d", value=np.nan,
                                   interpretation="Insufficient data")

        mean_diff = g1.mean() - g2.mean()

        if pooled:
            # Pooled standard deviation
            pooled_var = ((n1 - 1) * g1.var() + (n2 - 1) * g2.var()) / (n1 + n2 - 2)
            sd = np.sqrt(pooled_var)
        else:
            # Simple average of SDs
            sd = np.sqrt((g1.var() + g2.var()) / 2)

        d = mean_diff / sd if sd > 0 else np.nan

        # Variance and CI
        variance = (n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2))
        se = np.sqrt(variance)
        z = stats.norm.ppf((1 + self.confidence_level) / 2)
        ci_lower = d - z * se
        ci_upper = d + z * se

        return EffectSizeResult(
            name="Cohen's d",
            value=float(d),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            interpretation=self._interpret_d(d),
            variance=float(variance)
        )

    def hedges_g(self, group1: pd.Series, group2: pd.Series) -> EffectSizeResult:
        """Hedges' g (bias-corrected Cohen's d).

        Args:
            group1: First group data
            group2: Second group data

        Returns:
            EffectSizeResult object
        """
        d_result = self.cohens_d(group1, group2)

        if np.isnan(d_result.value):
            return EffectSizeResult(name="Hedges' g", value=np.nan,
                                   interpretation="Insufficient data")

        n1 = len(group1.dropna())
        n2 = len(group2.dropna())
        df = n1 + n2 - 2

        # Correction factor
        j = 1 - (3 / (4 * df - 1))
        g = d_result.value * j

        # Adjusted variance
        variance = j**2 * d_result.variance

        se = np.sqrt(variance)
        z = stats.norm.ppf((1 + self.confidence_level) / 2)
        ci_lower = g - z * se
        ci_upper = g + z * se

        return EffectSizeResult(
            name="Hedges' g",
            value=float(g),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            interpretation=self._interpret_d(g),
            variance=float(variance)
        )

    def glass_delta(self, treatment: pd.Series, control: pd.Series) -> EffectSizeResult:
        """Glass's Δ (uses control group SD).

        Args:
            treatment: Treatment group data
            control: Control group data

        Returns:
            EffectSizeResult object
        """
        treat = treatment.dropna()
        ctrl = control.dropna()

        if len(treat) < 2 or len(ctrl) < 2:
            return EffectSizeResult(name="Glass's Δ", value=np.nan,
                                   interpretation="Insufficient data")

        mean_diff = treat.mean() - ctrl.mean()
        sd_control = ctrl.std()

        delta = mean_diff / sd_control if sd_control > 0 else np.nan

        return EffectSizeResult(
            name="Glass's Δ",
            value=float(delta),
            interpretation=self._interpret_d(delta)
        )

    def cohens_d_paired(self, before: pd.Series, after: pd.Series) -> EffectSizeResult:
        """Cohen's d for paired samples.

        Args:
            before: Pre-treatment measurements
            after: Post-treatment measurements

        Returns:
            EffectSizeResult object
        """
        mask = before.notna() & after.notna()
        b = before[mask]
        a = after[mask]
        n = len(b)

        if n < 2:
            return EffectSizeResult(name="Cohen's d (paired)", value=np.nan,
                                   interpretation="Insufficient data")

        diff = a - b
        d = diff.mean() / diff.std()

        # CI
        variance = (1/n) + (d**2 / (2*n))
        se = np.sqrt(variance)
        z = stats.norm.ppf((1 + self.confidence_level) / 2)

        return EffectSizeResult(
            name="Cohen's d (paired)",
            value=float(d),
            ci_lower=float(d - z * se),
            ci_upper=float(d + z * se),
            interpretation=self._interpret_d(d),
            variance=float(variance)
        )

    # ==================== ANOVA EFFECT SIZES ====================

    def eta_squared(self, ss_between: float, ss_total: float) -> EffectSizeResult:
        """Eta-squared for ANOVA.

        Args:
            ss_between: Between-groups sum of squares
            ss_total: Total sum of squares

        Returns:
            EffectSizeResult object
        """
        eta_sq = ss_between / ss_total if ss_total > 0 else np.nan

        return EffectSizeResult(
            name="Eta-squared (η²)",
            value=float(eta_sq) if not np.isnan(eta_sq) else np.nan,
            interpretation=self._interpret_eta_squared(eta_sq)
        )

    def partial_eta_squared(self, ss_effect: float,
                           ss_error: float) -> EffectSizeResult:
        """Partial eta-squared.

        Args:
            ss_effect: Effect sum of squares
            ss_error: Error sum of squares

        Returns:
            EffectSizeResult object
        """
        partial_eta = ss_effect / (ss_effect + ss_error) if (ss_effect + ss_error) > 0 else np.nan

        return EffectSizeResult(
            name="Partial eta-squared (ηp²)",
            value=float(partial_eta) if not np.isnan(partial_eta) else np.nan,
            interpretation=self._interpret_eta_squared(partial_eta)
        )

    def omega_squared(self, ss_between: float, ss_total: float,
                      ms_error: float, k: int) -> EffectSizeResult:
        """Omega-squared (less biased than eta-squared).

        Args:
            ss_between: Between-groups sum of squares
            ss_total: Total sum of squares
            ms_error: Mean square error
            k: Number of groups

        Returns:
            EffectSizeResult object
        """
        omega_sq = (ss_between - (k - 1) * ms_error) / (ss_total + ms_error)
        omega_sq = max(0, omega_sq)  # Can't be negative

        return EffectSizeResult(
            name="Omega-squared (ω²)",
            value=float(omega_sq),
            interpretation=self._interpret_eta_squared(omega_sq)
        )

    def cohens_f(self, eta_squared: float) -> EffectSizeResult:
        """Cohen's f from eta-squared.

        Args:
            eta_squared: Eta-squared value

        Returns:
            EffectSizeResult object
        """
        if eta_squared >= 1 or eta_squared < 0:
            return EffectSizeResult(name="Cohen's f", value=np.nan)

        f = np.sqrt(eta_squared / (1 - eta_squared))

        if f < 0.1:
            interp = "negligible"
        elif f < 0.25:
            interp = "small"
        elif f < 0.4:
            interp = "medium"
        else:
            interp = "large"

        return EffectSizeResult(
            name="Cohen's f",
            value=float(f),
            interpretation=interp
        )

    # ==================== CORRELATION EFFECT SIZES ====================

    def r_to_d(self, r: float) -> EffectSizeResult:
        """Convert correlation to Cohen's d.

        Args:
            r: Correlation coefficient

        Returns:
            EffectSizeResult object
        """
        if abs(r) >= 1:
            return EffectSizeResult(name="d (from r)", value=np.nan)

        d = (2 * r) / np.sqrt(1 - r**2)

        return EffectSizeResult(
            name="Cohen's d (from r)",
            value=float(d),
            interpretation=self._interpret_d(d)
        )

    def d_to_r(self, d: float, n1: int = None, n2: int = None) -> EffectSizeResult:
        """Convert Cohen's d to correlation.

        Args:
            d: Cohen's d
            n1: Sample size group 1 (optional, for correction)
            n2: Sample size group 2 (optional, for correction)

        Returns:
            EffectSizeResult object
        """
        if n1 and n2:
            a = (n1 + n2)**2 / (n1 * n2)
        else:
            a = 4

        r = d / np.sqrt(d**2 + a)

        return EffectSizeResult(
            name="r (from d)",
            value=float(r),
            interpretation=self._interpret_r(r)
        )

    # ==================== CATEGORICAL EFFECT SIZES ====================

    def odds_ratio(self, a: int, b: int, c: int, d: int) -> EffectSizeResult:
        """Calculate odds ratio from 2x2 table.

        Args:
            a: Exposed cases
            b: Exposed non-cases
            c: Unexposed cases
            d: Unexposed non-cases

        Returns:
            EffectSizeResult object
        """
        if b * c == 0:
            return EffectSizeResult(name="Odds Ratio", value=np.nan)

        OR = (a * d) / (b * c)

        # Log odds CI
        se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d) if min(a, b, c, d) > 0 else np.nan
        z = stats.norm.ppf((1 + self.confidence_level) / 2)

        if not np.isnan(se_log_or):
            log_or = np.log(OR)
            ci_lower = np.exp(log_or - z * se_log_or)
            ci_upper = np.exp(log_or + z * se_log_or)
        else:
            ci_lower = ci_upper = np.nan

        return EffectSizeResult(
            name="Odds Ratio",
            value=float(OR),
            ci_lower=float(ci_lower) if not np.isnan(ci_lower) else None,
            ci_upper=float(ci_upper) if not np.isnan(ci_upper) else None,
            interpretation=self._interpret_or(OR)
        )

    def risk_ratio(self, a: int, b: int, c: int, d: int) -> EffectSizeResult:
        """Calculate risk ratio (relative risk) from 2x2 table.

        Args:
            a: Exposed cases
            b: Exposed non-cases
            c: Unexposed cases
            d: Unexposed non-cases

        Returns:
            EffectSizeResult object
        """
        risk_exposed = a / (a + b) if (a + b) > 0 else 0
        risk_unexposed = c / (c + d) if (c + d) > 0 else 0

        if risk_unexposed == 0:
            return EffectSizeResult(name="Risk Ratio", value=np.nan)

        RR = risk_exposed / risk_unexposed

        # Log RR CI
        se_log_rr = np.sqrt((b / (a * (a + b))) + (d / (c * (c + d)))) if min(a, c) > 0 else np.nan
        z = stats.norm.ppf((1 + self.confidence_level) / 2)

        if not np.isnan(se_log_rr):
            log_rr = np.log(RR)
            ci_lower = np.exp(log_rr - z * se_log_rr)
            ci_upper = np.exp(log_rr + z * se_log_rr)
        else:
            ci_lower = ci_upper = np.nan

        return EffectSizeResult(
            name="Risk Ratio (Relative Risk)",
            value=float(RR),
            ci_lower=float(ci_lower) if not np.isnan(ci_lower) else None,
            ci_upper=float(ci_upper) if not np.isnan(ci_upper) else None
        )

    def risk_difference(self, a: int, b: int, c: int, d: int) -> EffectSizeResult:
        """Calculate risk difference (absolute risk reduction).

        Args:
            a: Exposed cases
            b: Exposed non-cases
            c: Unexposed cases
            d: Unexposed non-cases

        Returns:
            EffectSizeResult object
        """
        risk_exposed = a / (a + b) if (a + b) > 0 else 0
        risk_unexposed = c / (c + d) if (c + d) > 0 else 0

        RD = risk_exposed - risk_unexposed

        # CI
        se = np.sqrt((risk_exposed * (1 - risk_exposed) / (a + b)) +
                     (risk_unexposed * (1 - risk_unexposed) / (c + d)))
        z = stats.norm.ppf((1 + self.confidence_level) / 2)

        return EffectSizeResult(
            name="Risk Difference",
            value=float(RD),
            ci_lower=float(RD - z * se),
            ci_upper=float(RD + z * se)
        )

    def number_needed_to_treat(self, a: int, b: int, c: int, d: int) -> EffectSizeResult:
        """Calculate number needed to treat (NNT).

        Args:
            a: Exposed cases
            b: Exposed non-cases
            c: Unexposed cases
            d: Unexposed non-cases

        Returns:
            EffectSizeResult object
        """
        rd_result = self.risk_difference(a, b, c, d)

        if rd_result.value == 0:
            return EffectSizeResult(name="NNT", value=np.inf)

        nnt = 1 / abs(rd_result.value)

        # NNT is for benefit if RD is negative (treatment reduces risk)
        name = "NNT (Benefit)" if rd_result.value < 0 else "NNT (Harm)"

        return EffectSizeResult(
            name=name,
            value=float(nnt),
            ci_lower=float(1 / abs(rd_result.ci_upper)) if rd_result.ci_upper else None,
            ci_upper=float(1 / abs(rd_result.ci_lower)) if rd_result.ci_lower else None
        )

    def cramers_v(self, contingency_table: pd.DataFrame) -> EffectSizeResult:
        """Calculate Cramér's V from contingency table.

        Args:
            contingency_table: Contingency table

        Returns:
            EffectSizeResult object
        """
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1

        v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else np.nan

        return EffectSizeResult(
            name="Cramér's V",
            value=float(v) if not np.isnan(v) else np.nan,
            interpretation=self._interpret_cramers_v(v, min_dim)
        )

    # ==================== INTERPRETATION METHODS ====================

    def _interpret_d(self, d: float) -> str:
        """Interpret Cohen's d."""
        if np.isnan(d):
            return "N/A"
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    def _interpret_r(self, r: float) -> str:
        """Interpret correlation."""
        if np.isnan(r):
            return "N/A"
        r = abs(r)
        if r < 0.1:
            return "negligible"
        elif r < 0.3:
            return "small"
        elif r < 0.5:
            return "medium"
        else:
            return "large"

    def _interpret_eta_squared(self, eta_sq: float) -> str:
        """Interpret eta-squared."""
        if np.isnan(eta_sq):
            return "N/A"
        if eta_sq < 0.01:
            return "negligible"
        elif eta_sq < 0.06:
            return "small"
        elif eta_sq < 0.14:
            return "medium"
        else:
            return "large"

    def _interpret_or(self, or_value: float) -> str:
        """Interpret odds ratio."""
        if np.isnan(or_value):
            return "N/A"
        if 0.9 <= or_value <= 1.1:
            return "negligible effect"
        elif 0.67 <= or_value <= 1.5:
            return "small effect"
        elif 0.33 <= or_value <= 3.0:
            return "medium effect"
        else:
            return "large effect"

    def _interpret_cramers_v(self, v: float, df: int) -> str:
        """Interpret Cramér's V."""
        if np.isnan(v):
            return "N/A"
        # Thresholds depend on degrees of freedom
        if df == 1:
            if v < 0.1:
                return "negligible"
            elif v < 0.3:
                return "small"
            elif v < 0.5:
                return "medium"
            else:
                return "large"
        else:
            if v < 0.07:
                return "negligible"
            elif v < 0.21:
                return "small"
            elif v < 0.35:
                return "medium"
            else:
                return "large"
