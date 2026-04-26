"""Comparative statistical tests module."""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field


@dataclass
class TestResult:
    """Generic test result container."""
    test_name: str
    statistic: float
    p_value: float
    df: Optional[float] = None
    effect_size: Optional[float] = None
    effect_size_name: Optional[str] = None
    effect_size_interpretation: Optional[str] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    conclusion: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


class ComparativeTests:
    """Statistical tests for comparing groups."""

    def __init__(self, significance_level: float = 0.05,
                 confidence_level: float = 0.95):
        self.alpha = significance_level
        self.confidence_level = confidence_level

    # ==================== T-TESTS ====================

    def one_sample_ttest(self, data: pd.Series,
                         population_mean: float) -> TestResult:
        """One-sample t-test.

        Args:
            data: Sample data
            population_mean: Hypothesized population mean

        Returns:
            TestResult object
        """
        clean_data = data.dropna()
        n = len(clean_data)

        if n < 2:
            return self._insufficient_data_result("One-sample t-test")

        t_stat, p_value = stats.ttest_1samp(clean_data, population_mean)

        # Effect size (Cohen's d)
        sample_mean = clean_data.mean()
        sample_std = clean_data.std()
        cohens_d = (sample_mean - population_mean) / sample_std

        # Confidence interval for the mean
        sem = sample_std / np.sqrt(n)
        t_crit = stats.t.ppf((1 + self.confidence_level) / 2, n - 1)
        ci_lower = sample_mean - t_crit * sem
        ci_upper = sample_mean + t_crit * sem

        effect_interp = self._interpret_cohens_d(cohens_d)

        if p_value < self.alpha:
            conclusion = (f"The sample mean ({sample_mean:.3f}) is significantly different "
                         f"from the population mean ({population_mean}), "
                         f"t({n-1}) = {t_stat:.3f}, p = {self._format_p(p_value)}.")
        else:
            conclusion = (f"No significant difference between sample mean ({sample_mean:.3f}) "
                         f"and population mean ({population_mean}), "
                         f"t({n-1}) = {t_stat:.3f}, p = {self._format_p(p_value)}.")

        return TestResult(
            test_name="One-sample t-test",
            statistic=float(t_stat),
            p_value=float(p_value),
            df=float(n - 1),
            effect_size=float(cohens_d),
            effect_size_name="Cohen's d",
            effect_size_interpretation=effect_interp,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            conclusion=conclusion,
            details={
                'sample_mean': float(sample_mean),
                'sample_std': float(sample_std),
                'n': n,
                'population_mean': population_mean
            }
        )

    def independent_ttest(self, group1: pd.Series, group2: pd.Series,
                          equal_var: bool = True) -> TestResult:
        """Independent samples t-test.

        Args:
            group1: First group data
            group2: Second group data
            equal_var: Assume equal variances (if False, uses Welch's t-test)

        Returns:
            TestResult object
        """
        g1 = group1.dropna()
        g2 = group2.dropna()
        n1, n2 = len(g1), len(g2)

        if n1 < 2 or n2 < 2:
            return self._insufficient_data_result("Independent samples t-test")

        t_stat, p_value = stats.ttest_ind(g1, g2, equal_var=equal_var)

        # Calculate degrees of freedom
        if equal_var:
            df = n1 + n2 - 2
        else:
            # Welch-Satterthwaite approximation
            v1, v2 = g1.var(), g2.var()
            df = ((v1/n1 + v2/n2)**2 /
                  ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1)))

        # Effect size (Cohen's d)
        mean_diff = g1.mean() - g2.mean()
        if equal_var:
            # Pooled standard deviation
            pooled_std = np.sqrt(((n1-1)*g1.var() + (n2-1)*g2.var()) / (n1+n2-2))
        else:
            pooled_std = np.sqrt((g1.var() + g2.var()) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else np.nan

        # Confidence interval for mean difference
        se_diff = np.sqrt(g1.var()/n1 + g2.var()/n2)
        t_crit = stats.t.ppf((1 + self.confidence_level) / 2, df)
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff

        effect_interp = self._interpret_cohens_d(cohens_d)
        test_name = "Independent samples t-test" if equal_var else "Welch's t-test"

        if p_value < self.alpha:
            conclusion = (f"Significant difference between groups, "
                         f"t({df:.1f}) = {t_stat:.3f}, p = {self._format_p(p_value)}, "
                         f"d = {cohens_d:.3f} ({effect_interp} effect).")
        else:
            conclusion = (f"No significant difference between groups, "
                         f"t({df:.1f}) = {t_stat:.3f}, p = {self._format_p(p_value)}.")

        return TestResult(
            test_name=test_name,
            statistic=float(t_stat),
            p_value=float(p_value),
            df=float(df),
            effect_size=float(cohens_d) if not np.isnan(cohens_d) else None,
            effect_size_name="Cohen's d",
            effect_size_interpretation=effect_interp,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            conclusion=conclusion,
            details={
                'group1_mean': float(g1.mean()),
                'group1_std': float(g1.std()),
                'group1_n': n1,
                'group2_mean': float(g2.mean()),
                'group2_std': float(g2.std()),
                'group2_n': n2,
                'mean_difference': float(mean_diff)
            }
        )

    def paired_ttest(self, before: pd.Series, after: pd.Series) -> TestResult:
        """Paired samples t-test.

        Args:
            before: Pre-treatment/baseline measurements
            after: Post-treatment measurements

        Returns:
            TestResult object
        """
        # Align data (remove pairs with missing values)
        mask = before.notna() & after.notna()
        b = before[mask]
        a = after[mask]
        n = len(b)

        if n < 2:
            return self._insufficient_data_result("Paired samples t-test")

        t_stat, p_value = stats.ttest_rel(b, a)

        # Effect size (Cohen's d for paired samples)
        diff = a - b
        mean_diff = diff.mean()
        std_diff = diff.std()
        cohens_d = mean_diff / std_diff if std_diff > 0 else np.nan

        # Confidence interval
        se_diff = std_diff / np.sqrt(n)
        t_crit = stats.t.ppf((1 + self.confidence_level) / 2, n - 1)
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff

        effect_interp = self._interpret_cohens_d(cohens_d)

        if p_value < self.alpha:
            conclusion = (f"Significant change from before to after, "
                         f"t({n-1}) = {t_stat:.3f}, p = {self._format_p(p_value)}, "
                         f"d = {cohens_d:.3f} ({effect_interp} effect).")
        else:
            conclusion = (f"No significant change, "
                         f"t({n-1}) = {t_stat:.3f}, p = {self._format_p(p_value)}.")

        return TestResult(
            test_name="Paired samples t-test",
            statistic=float(t_stat),
            p_value=float(p_value),
            df=float(n - 1),
            effect_size=float(cohens_d) if not np.isnan(cohens_d) else None,
            effect_size_name="Cohen's d",
            effect_size_interpretation=effect_interp,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            conclusion=conclusion,
            details={
                'before_mean': float(b.mean()),
                'after_mean': float(a.mean()),
                'mean_difference': float(mean_diff),
                'n_pairs': n
            }
        )

    # ==================== ANOVA ====================

    def one_way_anova(self, data: pd.DataFrame,
                      value_col: str,
                      group_col: str) -> TestResult:
        """One-way ANOVA.

        Args:
            data: DataFrame with the data
            value_col: Column name for the dependent variable
            group_col: Column name for the grouping variable

        Returns:
            TestResult object
        """
        groups = [group[value_col].dropna().values
                  for name, group in data.groupby(group_col)]

        if len(groups) < 2 or any(len(g) < 2 for g in groups):
            return self._insufficient_data_result("One-way ANOVA")

        f_stat, p_value = stats.f_oneway(*groups)

        # Effect size (eta-squared)
        all_data = np.concatenate(groups)
        grand_mean = all_data.mean()

        ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
        ss_total = sum((all_data - grand_mean)**2)
        eta_squared = ss_between / ss_total if ss_total > 0 else np.nan

        # Degrees of freedom
        k = len(groups)
        n_total = sum(len(g) for g in groups)
        df_between = k - 1
        df_within = n_total - k

        effect_interp = self._interpret_eta_squared(eta_squared)

        if p_value < self.alpha:
            conclusion = (f"Significant difference among groups, "
                         f"F({df_between}, {df_within}) = {f_stat:.3f}, "
                         f"p = {self._format_p(p_value)}, η² = {eta_squared:.3f} ({effect_interp}).")
        else:
            conclusion = (f"No significant difference among groups, "
                         f"F({df_between}, {df_within}) = {f_stat:.3f}, "
                         f"p = {self._format_p(p_value)}.")

        return TestResult(
            test_name="One-way ANOVA",
            statistic=float(f_stat),
            p_value=float(p_value),
            df=float(df_between),
            effect_size=float(eta_squared) if not np.isnan(eta_squared) else None,
            effect_size_name="Eta-squared (η²)",
            effect_size_interpretation=effect_interp,
            conclusion=conclusion,
            details={
                'n_groups': k,
                'df_between': df_between,
                'df_within': df_within,
                'ss_between': float(ss_between),
                'ss_total': float(ss_total),
                'group_means': {str(name): float(group[value_col].mean())
                               for name, group in data.groupby(group_col)}
            }
        )

    def welch_anova(self, data: pd.DataFrame,
                    value_col: str,
                    group_col: str) -> TestResult:
        """Welch's ANOVA (does not assume equal variances).

        Args:
            data: DataFrame with the data
            value_col: Column name for the dependent variable
            group_col: Column name for the grouping variable

        Returns:
            TestResult object
        """
        try:
            import pingouin as pg
            result = pg.welch_anova(data=data, dv=value_col, between=group_col)

            f_stat = result['F'].values[0]
            p_value = result['p-unc'].values[0]
            df1 = result['ddof1'].values[0]
            df2 = result['ddof2'].values[0]
            eta_sq = result['np2'].values[0]

            effect_interp = self._interpret_eta_squared(eta_sq)

            if p_value < self.alpha:
                conclusion = (f"Significant difference among groups (Welch's ANOVA), "
                             f"F({df1:.1f}, {df2:.1f}) = {f_stat:.3f}, "
                             f"p = {self._format_p(p_value)}.")
            else:
                conclusion = (f"No significant difference among groups, "
                             f"F({df1:.1f}, {df2:.1f}) = {f_stat:.3f}, "
                             f"p = {self._format_p(p_value)}.")

            return TestResult(
                test_name="Welch's ANOVA",
                statistic=float(f_stat),
                p_value=float(p_value),
                df=float(df1),
                effect_size=float(eta_sq),
                effect_size_name="Partial eta-squared (ηp²)",
                effect_size_interpretation=effect_interp,
                conclusion=conclusion,
                details={'df1': float(df1), 'df2': float(df2)}
            )

        except ImportError:
            # Fallback to scipy's Alexander-Govern test (similar purpose)
            return self._insufficient_data_result("Welch's ANOVA (pingouin required)")

    # ==================== NON-PARAMETRIC TESTS ====================

    def mann_whitney_u(self, group1: pd.Series, group2: pd.Series) -> TestResult:
        """Mann-Whitney U test (Wilcoxon rank-sum test).

        Args:
            group1: First group data
            group2: Second group data

        Returns:
            TestResult object
        """
        g1 = group1.dropna()
        g2 = group2.dropna()
        n1, n2 = len(g1), len(g2)

        if n1 < 1 or n2 < 1:
            return self._insufficient_data_result("Mann-Whitney U test")

        u_stat, p_value = stats.mannwhitneyu(g1, g2, alternative='two-sided')

        # Effect size (rank-biserial correlation)
        n_total = n1 * n2
        r = 1 - (2 * u_stat) / n_total

        effect_interp = self._interpret_r(r)

        if p_value < self.alpha:
            conclusion = (f"Significant difference between groups, "
                         f"U = {u_stat:.1f}, p = {self._format_p(p_value)}, "
                         f"r = {r:.3f} ({effect_interp} effect).")
        else:
            conclusion = (f"No significant difference between groups, "
                         f"U = {u_stat:.1f}, p = {self._format_p(p_value)}.")

        return TestResult(
            test_name="Mann-Whitney U test",
            statistic=float(u_stat),
            p_value=float(p_value),
            effect_size=float(r),
            effect_size_name="Rank-biserial correlation",
            effect_size_interpretation=effect_interp,
            conclusion=conclusion,
            details={
                'group1_median': float(g1.median()),
                'group2_median': float(g2.median()),
                'n1': n1,
                'n2': n2
            }
        )

    def wilcoxon_signed_rank(self, before: pd.Series, after: pd.Series) -> TestResult:
        """Wilcoxon signed-rank test (paired non-parametric).

        Args:
            before: Pre-treatment measurements
            after: Post-treatment measurements

        Returns:
            TestResult object
        """
        mask = before.notna() & after.notna()
        b = before[mask]
        a = after[mask]
        diff = a - b
        n = len(diff)

        # Remove zero differences
        nonzero_diff = diff[diff != 0]

        if len(nonzero_diff) < 1:
            return self._insufficient_data_result("Wilcoxon signed-rank test")

        w_stat, p_value = stats.wilcoxon(nonzero_diff)

        # Effect size (matched-pairs rank biserial correlation)
        n_nonzero = len(nonzero_diff)
        r = 1 - (2 * w_stat) / (n_nonzero * (n_nonzero + 1) / 2)

        effect_interp = self._interpret_r(r)

        if p_value < self.alpha:
            conclusion = (f"Significant change from before to after, "
                         f"W = {w_stat:.1f}, p = {self._format_p(p_value)}, "
                         f"r = {r:.3f} ({effect_interp} effect).")
        else:
            conclusion = (f"No significant change, "
                         f"W = {w_stat:.1f}, p = {self._format_p(p_value)}.")

        return TestResult(
            test_name="Wilcoxon signed-rank test",
            statistic=float(w_stat),
            p_value=float(p_value),
            effect_size=float(r),
            effect_size_name="Matched-pairs rank biserial r",
            effect_size_interpretation=effect_interp,
            conclusion=conclusion,
            details={
                'before_median': float(b.median()),
                'after_median': float(a.median()),
                'median_difference': float(diff.median()),
                'n_pairs': n,
                'n_nonzero_pairs': len(nonzero_diff)
            }
        )

    def kruskal_wallis(self, data: pd.DataFrame,
                       value_col: str,
                       group_col: str) -> TestResult:
        """Kruskal-Wallis H test (non-parametric alternative to one-way ANOVA).

        Args:
            data: DataFrame with the data
            value_col: Column name for the dependent variable
            group_col: Column name for the grouping variable

        Returns:
            TestResult object
        """
        groups = [group[value_col].dropna().values
                  for name, group in data.groupby(group_col)]

        if len(groups) < 2 or any(len(g) < 1 for g in groups):
            return self._insufficient_data_result("Kruskal-Wallis test")

        h_stat, p_value = stats.kruskal(*groups)

        # Effect size (eta-squared based on H)
        n_total = sum(len(g) for g in groups)
        k = len(groups)
        eta_h = (h_stat - k + 1) / (n_total - k)

        effect_interp = self._interpret_eta_squared(eta_h)

        if p_value < self.alpha:
            conclusion = (f"Significant difference among groups, "
                         f"H({k-1}) = {h_stat:.3f}, p = {self._format_p(p_value)}.")
        else:
            conclusion = (f"No significant difference among groups, "
                         f"H({k-1}) = {h_stat:.3f}, p = {self._format_p(p_value)}.")

        return TestResult(
            test_name="Kruskal-Wallis H test",
            statistic=float(h_stat),
            p_value=float(p_value),
            df=float(k - 1),
            effect_size=float(eta_h) if eta_h > 0 else None,
            effect_size_name="Eta-squared (H-based)",
            effect_size_interpretation=effect_interp,
            conclusion=conclusion,
            details={
                'n_groups': k,
                'group_medians': {str(name): float(group[value_col].median())
                                  for name, group in data.groupby(group_col)}
            }
        )

    def friedman_test(self, data: pd.DataFrame,
                      subject_col: str,
                      condition_col: str,
                      value_col: str) -> TestResult:
        """Friedman test (non-parametric repeated measures).

        Args:
            data: Long-format DataFrame
            subject_col: Subject identifier column
            condition_col: Condition/time column
            value_col: Dependent variable column

        Returns:
            TestResult object
        """
        # Pivot to wide format
        try:
            wide = data.pivot(index=subject_col, columns=condition_col, values=value_col)
            wide = wide.dropna()

            if wide.shape[0] < 2 or wide.shape[1] < 2:
                return self._insufficient_data_result("Friedman test")

            groups = [wide[col].values for col in wide.columns]
            chi_stat, p_value = stats.friedmanchisquare(*groups)

            # Effect size (Kendall's W)
            n = wide.shape[0]  # subjects
            k = wide.shape[1]  # conditions
            w = chi_stat / (n * (k - 1))

            if p_value < self.alpha:
                conclusion = (f"Significant difference across conditions, "
                             f"χ²({k-1}) = {chi_stat:.3f}, p = {self._format_p(p_value)}, "
                             f"W = {w:.3f}.")
            else:
                conclusion = (f"No significant difference across conditions, "
                             f"χ²({k-1}) = {chi_stat:.3f}, p = {self._format_p(p_value)}.")

            return TestResult(
                test_name="Friedman test",
                statistic=float(chi_stat),
                p_value=float(p_value),
                df=float(k - 1),
                effect_size=float(w),
                effect_size_name="Kendall's W",
                conclusion=conclusion,
                details={
                    'n_subjects': n,
                    'n_conditions': k
                }
            )

        except Exception as e:
            return self._insufficient_data_result(f"Friedman test: {str(e)}")

    # ==================== CHI-SQUARE TESTS ====================

    def chi_square_independence(self, data: pd.DataFrame,
                                row_var: str,
                                col_var: str) -> TestResult:
        """Chi-square test of independence.

        Args:
            data: DataFrame with the data
            row_var: Row variable name
            col_var: Column variable name

        Returns:
            TestResult object
        """
        contingency = pd.crosstab(data[row_var], data[col_var])

        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        # Effect size (Cramér's V)
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else np.nan

        effect_interp = self._interpret_cramers_v(cramers_v, min_dim)

        # Check if any expected counts < 5
        low_expected = (expected < 5).sum()

        if p_value < self.alpha:
            conclusion = (f"Significant association between {row_var} and {col_var}, "
                         f"χ²({dof}) = {chi2:.3f}, p = {self._format_p(p_value)}, "
                         f"V = {cramers_v:.3f} ({effect_interp}).")
        else:
            conclusion = (f"No significant association between {row_var} and {col_var}, "
                         f"χ²({dof}) = {chi2:.3f}, p = {self._format_p(p_value)}.")

        return TestResult(
            test_name="Chi-square test of independence",
            statistic=float(chi2),
            p_value=float(p_value),
            df=float(dof),
            effect_size=float(cramers_v) if not np.isnan(cramers_v) else None,
            effect_size_name="Cramér's V",
            effect_size_interpretation=effect_interp,
            conclusion=conclusion,
            details={
                'contingency_table': contingency.to_dict(),
                'expected_counts': expected.tolist(),
                'cells_with_expected_lt_5': int(low_expected),
                'n_total': int(n)
            }
        )

    def fisher_exact(self, contingency: Union[pd.DataFrame, np.ndarray]) -> TestResult:
        """Fisher's exact test (for 2x2 tables).

        Args:
            contingency: 2x2 contingency table

        Returns:
            TestResult object
        """
        if isinstance(contingency, pd.DataFrame):
            table = contingency.values
        else:
            table = contingency

        if table.shape != (2, 2):
            return self._insufficient_data_result("Fisher's exact test (requires 2x2 table)")

        odds_ratio, p_value = stats.fisher_exact(table)

        if p_value < self.alpha:
            conclusion = (f"Significant association, "
                         f"OR = {odds_ratio:.3f}, p = {self._format_p(p_value)}.")
        else:
            conclusion = (f"No significant association, "
                         f"OR = {odds_ratio:.3f}, p = {self._format_p(p_value)}.")

        return TestResult(
            test_name="Fisher's exact test",
            statistic=float(odds_ratio),
            p_value=float(p_value),
            effect_size=float(odds_ratio),
            effect_size_name="Odds Ratio",
            conclusion=conclusion
        )

    def mcnemar_test(self, before: pd.Series, after: pd.Series) -> TestResult:
        """McNemar's test for paired categorical data.

        Args:
            before: Before/control condition (binary)
            after: After/treatment condition (binary)

        Returns:
            TestResult object
        """
        # Create contingency table
        mask = before.notna() & after.notna()
        b = before[mask].astype(int)
        a = after[mask].astype(int)

        # Count discordant pairs
        b_yes_a_no = ((b == 1) & (a == 0)).sum()  # Changed from positive to negative
        b_no_a_yes = ((b == 0) & (a == 1)).sum()  # Changed from negative to positive

        n_discordant = b_yes_a_no + b_no_a_yes

        if n_discordant < 1:
            return self._insufficient_data_result("McNemar's test (no discordant pairs)")

        # McNemar's chi-square (with continuity correction)
        chi2 = (abs(b_yes_a_no - b_no_a_yes) - 1)**2 / n_discordant
        p_value = 1 - stats.chi2.cdf(chi2, 1)

        if p_value < self.alpha:
            conclusion = (f"Significant change in proportions, "
                         f"χ² = {chi2:.3f}, p = {self._format_p(p_value)}.")
        else:
            conclusion = (f"No significant change in proportions, "
                         f"χ² = {chi2:.3f}, p = {self._format_p(p_value)}.")

        return TestResult(
            test_name="McNemar's test",
            statistic=float(chi2),
            p_value=float(p_value),
            df=1.0,
            conclusion=conclusion,
            details={
                'b_yes_a_no': int(b_yes_a_no),
                'b_no_a_yes': int(b_no_a_yes),
                'n_discordant': int(n_discordant)
            }
        )

    # ==================== POST-HOC TESTS ====================

    def tukey_hsd(self, data: pd.DataFrame,
                  value_col: str,
                  group_col: str) -> Dict[str, TestResult]:
        """Tukey's HSD post-hoc test.

        Args:
            data: DataFrame with the data
            value_col: Column name for the dependent variable
            group_col: Column name for the grouping variable

        Returns:
            Dictionary of pairwise TestResults
        """
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd

            clean_data = data[[value_col, group_col]].dropna()
            result = pairwise_tukeyhsd(clean_data[value_col], clean_data[group_col])

            pairwise_results = {}
            for i in range(len(result.summary().data) - 1):
                row = result.summary().data[i + 1]
                g1, g2 = row[0], row[1]
                mean_diff = float(row[2])
                p_adj = float(row[3])
                ci_low = float(row[4])
                ci_high = float(row[5])
                reject = row[6]

                pair_name = f"{g1} vs {g2}"
                pairwise_results[pair_name] = TestResult(
                    test_name="Tukey HSD",
                    statistic=mean_diff,
                    p_value=p_adj,
                    ci_lower=ci_low,
                    ci_upper=ci_high,
                    conclusion=f"{'Significant' if reject else 'Not significant'} difference",
                    details={'group1': str(g1), 'group2': str(g2)}
                )

            return pairwise_results

        except ImportError:
            return {'error': self._insufficient_data_result("Tukey HSD (statsmodels required)")}

    def dunn_test(self, data: pd.DataFrame,
                  value_col: str,
                  group_col: str,
                  p_adjust: str = 'bonferroni') -> Dict[str, TestResult]:
        """Dunn's test (post-hoc for Kruskal-Wallis).

        Args:
            data: DataFrame with the data
            value_col: Column name for the dependent variable
            group_col: Column name for the grouping variable
            p_adjust: P-value adjustment method

        Returns:
            Dictionary of pairwise TestResults
        """
        try:
            import scikit_posthocs as sp

            clean_data = data[[value_col, group_col]].dropna()
            result = sp.posthoc_dunn(clean_data, val_col=value_col,
                                     group_col=group_col, p_adjust=p_adjust)

            pairwise_results = {}
            groups = result.index.tolist()

            for i, g1 in enumerate(groups):
                for g2 in groups[i+1:]:
                    p_val = result.loc[g1, g2]
                    pair_name = f"{g1} vs {g2}"
                    pairwise_results[pair_name] = TestResult(
                        test_name=f"Dunn's test ({p_adjust})",
                        statistic=np.nan,
                        p_value=float(p_val),
                        conclusion=f"{'Significant' if p_val < self.alpha else 'Not significant'}",
                        details={'group1': str(g1), 'group2': str(g2)}
                    )

            return pairwise_results

        except ImportError:
            return {'error': self._insufficient_data_result("Dunn's test (scikit-posthocs required)")}

    # ==================== HELPER METHODS ====================

    def _insufficient_data_result(self, test_name: str) -> TestResult:
        """Create result for insufficient data."""
        return TestResult(
            test_name=test_name,
            statistic=np.nan,
            p_value=np.nan,
            conclusion="Insufficient data to perform test"
        )

    def _format_p(self, p: float) -> str:
        """Format p-value for display."""
        if p < 0.001:
            return "< 0.001"
        return f"{p:.3f}"

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
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

    def _interpret_eta_squared(self, eta_sq: float) -> str:
        """Interpret eta-squared effect size."""
        if np.isnan(eta_sq) or eta_sq < 0:
            return "N/A"
        if eta_sq < 0.01:
            return "negligible"
        elif eta_sq < 0.06:
            return "small"
        elif eta_sq < 0.14:
            return "medium"
        else:
            return "large"

    def _interpret_r(self, r: float) -> str:
        """Interpret correlation effect size."""
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

    def _interpret_cramers_v(self, v: float, min_dim: int) -> str:
        """Interpret Cramér's V effect size."""
        if np.isnan(v):
            return "N/A"
        # Thresholds depend on degrees of freedom
        if min_dim == 1:
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
