"""Agreement and reliability analysis module."""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class AgreementResult:
    """Result of agreement analysis."""
    analysis_type: str
    n_observations: int
    statistic: float
    statistic_name: str
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    p_value: Optional[float] = None
    interpretation: Optional[str] = None
    details: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class AgreementAnalysis:
    """Agreement and reliability analysis methods."""

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level

    def cohens_kappa(self, rater1: pd.Series, rater2: pd.Series,
                     weighted: bool = False) -> AgreementResult:
        """Cohen's Kappa for inter-rater agreement.

        Args:
            rater1: Ratings from first rater
            rater2: Ratings from second rater
            weighted: Use weighted kappa for ordinal data

        Returns:
            AgreementResult object
        """
        mask = rater1.notna() & rater2.notna()
        r1 = rater1[mask].values
        r2 = rater2[mask].values
        n = len(r1)

        if n < 2:
            return self._insufficient_data("Cohen's Kappa")

        # Get unique categories
        categories = np.unique(np.concatenate([r1, r2]))
        n_cat = len(categories)

        # Create confusion matrix
        cat_to_idx = {cat: i for i, cat in enumerate(categories)}
        conf_matrix = np.zeros((n_cat, n_cat))

        for i in range(n):
            conf_matrix[cat_to_idx[r1[i]], cat_to_idx[r2[i]]] += 1

        # Observed agreement
        p_o = np.diag(conf_matrix).sum() / n

        # Expected agreement
        row_marginals = conf_matrix.sum(axis=1)
        col_marginals = conf_matrix.sum(axis=0)
        p_e = (row_marginals @ col_marginals) / (n ** 2)

        if weighted and n_cat > 2:
            # Weighted kappa (linear weights)
            weights = np.abs(np.subtract.outer(np.arange(n_cat), np.arange(n_cat)))
            weights = 1 - weights / weights.max()

            p_o = np.sum(weights * conf_matrix) / n
            p_e = np.sum(weights * np.outer(row_marginals, col_marginals)) / (n ** 2)

        # Kappa
        kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 0

        # Standard error and CI (Fleiss's formula)
        se = np.sqrt((p_o * (1 - p_o)) / (n * (1 - p_e)**2))
        z = stats.norm.ppf((1 + self.confidence_level) / 2)
        ci_lower = kappa - z * se
        ci_upper = kappa + z * se

        # Interpretation
        interpretation = self._interpret_kappa(kappa)

        return AgreementResult(
            analysis_type="Weighted Cohen's Kappa" if weighted else "Cohen's Kappa",
            n_observations=n,
            statistic=float(kappa),
            statistic_name="κ",
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            interpretation=interpretation,
            details={
                'observed_agreement': float(p_o),
                'expected_agreement': float(p_e),
                'categories': categories.tolist()
            }
        )

    def fleiss_kappa(self, ratings: pd.DataFrame) -> AgreementResult:
        """Fleiss' Kappa for multiple raters.

        Args:
            ratings: DataFrame where rows are subjects and columns are raters

        Returns:
            AgreementResult object
        """
        # Convert to numpy and handle missing
        data = ratings.dropna()
        n_subjects = len(data)
        n_raters = len(data.columns)

        if n_subjects < 2 or n_raters < 2:
            return self._insufficient_data("Fleiss' Kappa")

        # Get unique categories
        categories = np.unique(data.values.flatten())
        n_categories = len(categories)

        # Count ratings per category per subject
        cat_counts = np.zeros((n_subjects, n_categories))
        for i, cat in enumerate(categories):
            cat_counts[:, i] = (data == cat).sum(axis=1)

        # Proportion of pairs in agreement
        p_j = cat_counts.sum(axis=0) / (n_subjects * n_raters)
        P_i = (cat_counts ** 2).sum(axis=1) - n_raters
        P_i = P_i / (n_raters * (n_raters - 1))

        P_bar = P_i.mean()
        P_e = (p_j ** 2).sum()

        kappa = (P_bar - P_e) / (1 - P_e) if P_e < 1 else 0

        interpretation = self._interpret_kappa(kappa)

        return AgreementResult(
            analysis_type="Fleiss' Kappa",
            n_observations=n_subjects,
            statistic=float(kappa),
            statistic_name="κ",
            interpretation=interpretation,
            details={
                'n_raters': n_raters,
                'n_categories': n_categories
            }
        )

    def intraclass_correlation(self, data: pd.DataFrame,
                               model: str = '2',
                               type_: str = 'k',
                               unit: str = 'single') -> AgreementResult:
        """Intraclass Correlation Coefficient (ICC).

        Args:
            data: DataFrame where rows are subjects and columns are raters/measurements
            model: '1' (one-way), '2' (two-way random), '3' (two-way mixed)
            type_: 'k' (average of k raters), '1' (single rater)
            unit: 'single' or 'average'

        Returns:
            AgreementResult object
        """
        try:
            import pingouin as pg

            # Reshape to long format
            data_clean = data.dropna()
            n_subjects = len(data_clean)
            n_raters = len(data_clean.columns)

            if n_subjects < 2 or n_raters < 2:
                return self._insufficient_data("ICC")

            # Create long format
            long_data = data_clean.reset_index().melt(
                id_vars='index',
                var_name='rater',
                value_name='rating'
            )
            long_data.columns = ['subject', 'rater', 'rating']

            # Calculate ICC
            icc_result = pg.intraclass_corr(
                data=long_data,
                targets='subject',
                raters='rater',
                ratings='rating'
            )

            # Select appropriate ICC type
            icc_type = f"ICC{model},{type_}"
            row = icc_result[icc_result['Type'].str.contains(f'{model}')]

            if unit == 'single':
                row = row[row['Type'].str.contains('1')]
            else:
                row = row[row['Type'].str.contains('k')]

            if len(row) == 0:
                row = icc_result.iloc[0]
            else:
                row = row.iloc[0]

            icc = row['ICC']
            ci_lower = row['CI95%'][0]
            ci_upper = row['CI95%'][1]
            f_val = row['F']
            p_val = row['pval']

            interpretation = self._interpret_icc(icc)

            return AgreementResult(
                analysis_type=f"ICC ({row['Type']})",
                n_observations=n_subjects,
                statistic=float(icc),
                statistic_name="ICC",
                ci_lower=float(ci_lower),
                ci_upper=float(ci_upper),
                p_value=float(p_val),
                interpretation=interpretation,
                details={
                    'n_raters': n_raters,
                    'F_statistic': float(f_val)
                }
            )

        except ImportError:
            return self._insufficient_data("ICC (pingouin required)")
        except Exception as e:
            result = self._insufficient_data("ICC")
            result.interpretation = str(e)
            return result

    def bland_altman(self, method1: pd.Series, method2: pd.Series,
                     log_transform: bool = False) -> Dict[str, Any]:
        """Bland-Altman analysis for method comparison.

        Args:
            method1: Measurements from first method
            method2: Measurements from second method
            log_transform: Apply log transformation (for ratio limits)

        Returns:
            Dictionary with Bland-Altman results
        """
        mask = method1.notna() & method2.notna()
        m1 = method1[mask].values
        m2 = method2[mask].values
        n = len(m1)

        if n < 3:
            return {'error': 'Insufficient data'}

        if log_transform:
            m1 = np.log(m1)
            m2 = np.log(m2)

        # Calculate differences and means
        diff = m1 - m2
        mean = (m1 + m2) / 2

        # Bias and limits of agreement
        bias = diff.mean()
        sd = diff.std()
        loa_lower = bias - 1.96 * sd
        loa_upper = bias + 1.96 * sd

        # Confidence intervals
        se_bias = sd / np.sqrt(n)
        se_loa = np.sqrt(3 * sd**2 / n)
        t_crit = stats.t.ppf((1 + self.confidence_level) / 2, n - 1)

        bias_ci = (bias - t_crit * se_bias, bias + t_crit * se_bias)
        loa_lower_ci = (loa_lower - t_crit * se_loa, loa_lower + t_crit * se_loa)
        loa_upper_ci = (loa_upper - t_crit * se_loa, loa_upper + t_crit * se_loa)

        # Test for proportional bias (correlation between diff and mean)
        r, p_value = stats.pearsonr(mean, diff)

        interpretation = []
        if abs(bias) < 0.1 * np.abs(mean).mean():
            interpretation.append("No systematic bias detected.")
        else:
            interpretation.append(f"Systematic bias of {bias:.3f} detected.")

        if p_value < 0.05:
            interpretation.append("Proportional bias present (methods differ depending on magnitude).")

        return {
            'n': n,
            'bias': float(bias),
            'bias_ci': bias_ci,
            'sd_of_differences': float(sd),
            'loa_lower': float(loa_lower),
            'loa_upper': float(loa_upper),
            'loa_lower_ci': loa_lower_ci,
            'loa_upper_ci': loa_upper_ci,
            'proportional_bias_r': float(r),
            'proportional_bias_p': float(p_value),
            'means': mean.tolist(),
            'differences': diff.tolist(),
            'interpretation': " ".join(interpretation)
        }

    def cronbachs_alpha(self, data: pd.DataFrame) -> AgreementResult:
        """Cronbach's Alpha for internal consistency.

        Args:
            data: DataFrame where columns are items/questions

        Returns:
            AgreementResult object
        """
        data_clean = data.dropna()
        n_items = len(data_clean.columns)
        n_subjects = len(data_clean)

        if n_items < 2 or n_subjects < 3:
            return self._insufficient_data("Cronbach's Alpha")

        # Variance of each item
        item_vars = data_clean.var()

        # Variance of total scores
        total_var = data_clean.sum(axis=1).var()

        # Cronbach's alpha
        alpha = (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)

        # Item-total correlations
        total_scores = data_clean.sum(axis=1)
        item_total_corr = {}
        for col in data_clean.columns:
            rest_total = total_scores - data_clean[col]
            r, _ = stats.pearsonr(data_clean[col], rest_total)
            item_total_corr[col] = float(r)

        # Alpha if item deleted
        alpha_if_deleted = {}
        for col in data_clean.columns:
            subset = data_clean.drop(columns=[col])
            if len(subset.columns) > 1:
                item_v = subset.var().sum()
                total_v = subset.sum(axis=1).var()
                k = len(subset.columns)
                alpha_deleted = (k / (k - 1)) * (1 - item_v / total_v)
                alpha_if_deleted[col] = float(alpha_deleted)

        interpretation = self._interpret_alpha(alpha)

        return AgreementResult(
            analysis_type="Cronbach's Alpha",
            n_observations=n_subjects,
            statistic=float(alpha),
            statistic_name="α",
            interpretation=interpretation,
            details={
                'n_items': n_items,
                'item_total_correlations': item_total_corr,
                'alpha_if_item_deleted': alpha_if_deleted
            }
        )

    def concordance_correlation(self, x: pd.Series, y: pd.Series) -> AgreementResult:
        """Lin's Concordance Correlation Coefficient.

        Args:
            x: First set of measurements
            y: Second set of measurements

        Returns:
            AgreementResult object
        """
        mask = x.notna() & y.notna()
        x_clean = x[mask].values
        y_clean = y[mask].values
        n = len(x_clean)

        if n < 3:
            return self._insufficient_data("Concordance Correlation")

        # Calculate CCC
        mean_x = x_clean.mean()
        mean_y = y_clean.mean()
        var_x = x_clean.var()
        var_y = y_clean.var()
        cov_xy = np.cov(x_clean, y_clean)[0, 1]

        ccc = (2 * cov_xy) / (var_x + var_y + (mean_x - mean_y)**2)

        # Standard error and CI (Fisher transformation)
        z = np.arctanh(ccc)
        se = np.sqrt(1 / (n - 3))
        z_crit = stats.norm.ppf((1 + self.confidence_level) / 2)
        ci_lower = np.tanh(z - z_crit * se)
        ci_upper = np.tanh(z + z_crit * se)

        interpretation = self._interpret_ccc(ccc)

        return AgreementResult(
            analysis_type="Lin's Concordance Correlation",
            n_observations=n,
            statistic=float(ccc),
            statistic_name="ρc",
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            interpretation=interpretation
        )

    def _interpret_kappa(self, kappa: float) -> str:
        """Interpret kappa value."""
        if kappa < 0:
            return "Less than chance agreement"
        elif kappa < 0.20:
            return "Slight agreement"
        elif kappa < 0.40:
            return "Fair agreement"
        elif kappa < 0.60:
            return "Moderate agreement"
        elif kappa < 0.80:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"

    def _interpret_icc(self, icc: float) -> str:
        """Interpret ICC value."""
        if icc < 0.5:
            return "Poor reliability"
        elif icc < 0.75:
            return "Moderate reliability"
        elif icc < 0.9:
            return "Good reliability"
        else:
            return "Excellent reliability"

    def _interpret_alpha(self, alpha: float) -> str:
        """Interpret Cronbach's alpha."""
        if alpha < 0.5:
            return "Unacceptable internal consistency"
        elif alpha < 0.6:
            return "Poor internal consistency"
        elif alpha < 0.7:
            return "Questionable internal consistency"
        elif alpha < 0.8:
            return "Acceptable internal consistency"
        elif alpha < 0.9:
            return "Good internal consistency"
        else:
            return "Excellent internal consistency"

    def _interpret_ccc(self, ccc: float) -> str:
        """Interpret concordance correlation."""
        if ccc < 0.9:
            return "Poor agreement"
        elif ccc < 0.95:
            return "Moderate agreement"
        elif ccc < 0.99:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"

    def _insufficient_data(self, analysis_type: str) -> AgreementResult:
        """Create result for insufficient data."""
        return AgreementResult(
            analysis_type=analysis_type,
            n_observations=0,
            statistic=np.nan,
            statistic_name="N/A",
            interpretation="Insufficient data"
        )
