"""Correlation analysis module."""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field


@dataclass
class CorrelationResult:
    """Result of correlation analysis."""
    test_name: str
    variable1: str
    variable2: str
    correlation: float
    p_value: float
    n: int
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    interpretation: Optional[str] = None
    conclusion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class CorrelationAnalysis:
    """Correlation analysis methods."""

    def __init__(self, significance_level: float = 0.05,
                 confidence_level: float = 0.95):
        self.alpha = significance_level
        self.confidence_level = confidence_level

    def pearson(self, x: pd.Series, y: pd.Series) -> CorrelationResult:
        """Pearson correlation coefficient.

        Args:
            x: First variable
            y: Second variable

        Returns:
            CorrelationResult object
        """
        mask = x.notna() & y.notna()
        x_clean = x[mask]
        y_clean = y[mask]
        n = len(x_clean)

        if n < 3:
            return self._insufficient_data("Pearson correlation", x.name, y.name)

        r, p_value = stats.pearsonr(x_clean, y_clean)

        # Fisher's z transformation for confidence interval
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        z_crit = stats.norm.ppf((1 + self.confidence_level) / 2)
        z_lower = z - z_crit * se
        z_upper = z + z_crit * se
        ci_lower = np.tanh(z_lower)
        ci_upper = np.tanh(z_upper)

        interpretation = self._interpret_correlation(r)

        if p_value < self.alpha:
            conclusion = (f"Significant {interpretation} {'positive' if r > 0 else 'negative'} "
                         f"correlation, r({n-2}) = {r:.3f}, p = {self._format_p(p_value)}, "
                         f"95% CI [{ci_lower:.3f}, {ci_upper:.3f}].")
        else:
            conclusion = (f"No significant correlation, "
                         f"r({n-2}) = {r:.3f}, p = {self._format_p(p_value)}.")

        return CorrelationResult(
            test_name="Pearson correlation",
            variable1=str(x.name) or "X",
            variable2=str(y.name) or "Y",
            correlation=float(r),
            p_value=float(p_value),
            n=n,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            interpretation=interpretation,
            conclusion=conclusion
        )

    def spearman(self, x: pd.Series, y: pd.Series) -> CorrelationResult:
        """Spearman rank correlation coefficient.

        Args:
            x: First variable
            y: Second variable

        Returns:
            CorrelationResult object
        """
        mask = x.notna() & y.notna()
        x_clean = x[mask]
        y_clean = y[mask]
        n = len(x_clean)

        if n < 3:
            return self._insufficient_data("Spearman correlation", x.name, y.name)

        rho, p_value = stats.spearmanr(x_clean, y_clean)

        interpretation = self._interpret_correlation(rho)

        if p_value < self.alpha:
            conclusion = (f"Significant {interpretation} {'positive' if rho > 0 else 'negative'} "
                         f"monotonic relationship, ρ = {rho:.3f}, p = {self._format_p(p_value)}.")
        else:
            conclusion = (f"No significant monotonic relationship, "
                         f"ρ = {rho:.3f}, p = {self._format_p(p_value)}.")

        return CorrelationResult(
            test_name="Spearman correlation",
            variable1=str(x.name) or "X",
            variable2=str(y.name) or "Y",
            correlation=float(rho),
            p_value=float(p_value),
            n=n,
            interpretation=interpretation,
            conclusion=conclusion
        )

    def kendall(self, x: pd.Series, y: pd.Series) -> CorrelationResult:
        """Kendall's tau correlation coefficient.

        Args:
            x: First variable
            y: Second variable

        Returns:
            CorrelationResult object
        """
        mask = x.notna() & y.notna()
        x_clean = x[mask]
        y_clean = y[mask]
        n = len(x_clean)

        if n < 3:
            return self._insufficient_data("Kendall's tau", x.name, y.name)

        tau, p_value = stats.kendalltau(x_clean, y_clean)

        interpretation = self._interpret_correlation(tau)

        if p_value < self.alpha:
            conclusion = (f"Significant {interpretation} {'positive' if tau > 0 else 'negative'} "
                         f"association, τ = {tau:.3f}, p = {self._format_p(p_value)}.")
        else:
            conclusion = (f"No significant association, "
                         f"τ = {tau:.3f}, p = {self._format_p(p_value)}.")

        return CorrelationResult(
            test_name="Kendall's tau",
            variable1=str(x.name) or "X",
            variable2=str(y.name) or "Y",
            correlation=float(tau),
            p_value=float(p_value),
            n=n,
            interpretation=interpretation,
            conclusion=conclusion
        )

    def point_biserial(self, continuous: pd.Series,
                       binary: pd.Series) -> CorrelationResult:
        """Point-biserial correlation (continuous vs binary).

        Args:
            continuous: Continuous variable
            binary: Binary variable (0/1)

        Returns:
            CorrelationResult object
        """
        mask = continuous.notna() & binary.notna()
        cont = continuous[mask]
        bin_var = binary[mask]
        n = len(cont)

        if n < 3:
            return self._insufficient_data("Point-biserial correlation",
                                          continuous.name, binary.name)

        # Ensure binary is 0/1
        unique_vals = bin_var.unique()
        if len(unique_vals) != 2:
            return self._insufficient_data("Point-biserial (requires exactly 2 groups)",
                                          continuous.name, binary.name)

        r_pb, p_value = stats.pointbiserialr(bin_var, cont)

        interpretation = self._interpret_correlation(r_pb)

        if p_value < self.alpha:
            conclusion = (f"Significant {interpretation} correlation between "
                         f"continuous and binary variable, r_pb = {r_pb:.3f}, "
                         f"p = {self._format_p(p_value)}.")
        else:
            conclusion = (f"No significant correlation, "
                         f"r_pb = {r_pb:.3f}, p = {self._format_p(p_value)}.")

        return CorrelationResult(
            test_name="Point-biserial correlation",
            variable1=str(continuous.name) or "Continuous",
            variable2=str(binary.name) or "Binary",
            correlation=float(r_pb),
            p_value=float(p_value),
            n=n,
            interpretation=interpretation,
            conclusion=conclusion
        )

    def partial_correlation(self, x: pd.Series, y: pd.Series,
                           covariates: pd.DataFrame) -> CorrelationResult:
        """Partial correlation controlling for covariates.

        Args:
            x: First variable
            y: Second variable
            covariates: DataFrame of covariates to control for

        Returns:
            CorrelationResult object
        """
        try:
            import pingouin as pg

            # Combine into single DataFrame
            df = pd.DataFrame({'x': x, 'y': y})
            for col in covariates.columns:
                df[col] = covariates[col]

            df = df.dropna()
            n = len(df)

            if n < 3 + len(covariates.columns):
                return self._insufficient_data("Partial correlation", x.name, y.name)

            result = pg.partial_corr(data=df, x='x', y='y',
                                    covar=covariates.columns.tolist())

            r = result['r'].values[0]
            p_value = result['p-val'].values[0]
            ci = result['CI95%'].values[0]

            interpretation = self._interpret_correlation(r)

            if p_value < self.alpha:
                conclusion = (f"Significant partial correlation controlling for "
                             f"{', '.join(covariates.columns)}, "
                             f"r = {r:.3f}, p = {self._format_p(p_value)}.")
            else:
                conclusion = (f"No significant partial correlation, "
                             f"r = {r:.3f}, p = {self._format_p(p_value)}.")

            return CorrelationResult(
                test_name="Partial correlation",
                variable1=str(x.name) or "X",
                variable2=str(y.name) or "Y",
                correlation=float(r),
                p_value=float(p_value),
                n=n,
                ci_lower=float(ci[0]) if ci else None,
                ci_upper=float(ci[1]) if ci else None,
                interpretation=interpretation,
                conclusion=conclusion
            )

        except ImportError:
            return self._insufficient_data("Partial correlation (pingouin required)",
                                          x.name, y.name)

    def correlation_matrix(self, data: pd.DataFrame,
                          method: str = 'pearson',
                          variables: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculate correlation matrix.

        Args:
            data: DataFrame with the data
            method: 'pearson', 'spearman', or 'kendall'
            variables: List of variables (uses all numeric if None)

        Returns:
            Dictionary with correlation matrix, p-values, and n
        """
        if variables:
            df = data[variables].copy()
        else:
            df = data.select_dtypes(include=[np.number]).copy()

        n_vars = len(df.columns)
        var_names = df.columns.tolist()

        # Initialize matrices
        corr_matrix = np.zeros((n_vars, n_vars))
        p_matrix = np.zeros((n_vars, n_vars))
        n_matrix = np.zeros((n_vars, n_vars))

        # Calculate pairwise correlations
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    p_matrix[i, j] = 0.0
                    n_matrix[i, j] = df[var_names[i]].notna().sum()
                else:
                    mask = df[var_names[i]].notna() & df[var_names[j]].notna()
                    x = df.loc[mask, var_names[i]]
                    y = df.loc[mask, var_names[j]]

                    if len(x) >= 3:
                        if method == 'pearson':
                            r, p = stats.pearsonr(x, y)
                        elif method == 'spearman':
                            r, p = stats.spearmanr(x, y)
                        else:  # kendall
                            r, p = stats.kendalltau(x, y)

                        corr_matrix[i, j] = r
                        p_matrix[i, j] = p
                        n_matrix[i, j] = len(x)
                    else:
                        corr_matrix[i, j] = np.nan
                        p_matrix[i, j] = np.nan
                        n_matrix[i, j] = len(x)

        return {
            'method': method,
            'variables': var_names,
            'correlation_matrix': pd.DataFrame(corr_matrix,
                                               index=var_names,
                                               columns=var_names),
            'p_value_matrix': pd.DataFrame(p_matrix,
                                           index=var_names,
                                           columns=var_names),
            'n_matrix': pd.DataFrame(n_matrix.astype(int),
                                     index=var_names,
                                     columns=var_names),
            'significant_pairs': self._find_significant_pairs(
                corr_matrix, p_matrix, var_names
            )
        }

    def _find_significant_pairs(self, corr_matrix: np.ndarray,
                                p_matrix: np.ndarray,
                                var_names: List[str]) -> List[Dict[str, Any]]:
        """Find significantly correlated pairs."""
        significant = []
        n = len(var_names)

        for i in range(n):
            for j in range(i + 1, n):
                if p_matrix[i, j] < self.alpha and not np.isnan(corr_matrix[i, j]):
                    significant.append({
                        'var1': var_names[i],
                        'var2': var_names[j],
                        'r': round(corr_matrix[i, j], 3),
                        'p': round(p_matrix[i, j], 4),
                        'interpretation': self._interpret_correlation(corr_matrix[i, j])
                    })

        # Sort by absolute correlation
        significant.sort(key=lambda x: abs(x['r']), reverse=True)
        return significant

    def compare_correlations(self, r1: float, n1: int,
                            r2: float, n2: int) -> Dict[str, Any]:
        """Compare two independent correlations (Fisher's z test).

        Args:
            r1: First correlation coefficient
            n1: Sample size for first correlation
            r2: Second correlation coefficient
            n2: Sample size for second correlation

        Returns:
            Dictionary with test results
        """
        # Fisher z transformation
        z1 = np.arctanh(r1)
        z2 = np.arctanh(r2)

        # Standard error of difference
        se = np.sqrt(1/(n1-3) + 1/(n2-3))

        # Z-score
        z_score = (z1 - z2) / se

        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        if p_value < self.alpha:
            conclusion = (f"The correlations are significantly different, "
                         f"z = {z_score:.3f}, p = {self._format_p(p_value)}.")
        else:
            conclusion = (f"No significant difference between correlations, "
                         f"z = {z_score:.3f}, p = {self._format_p(p_value)}.")

        return {
            'test_name': "Fisher's z-test for comparing correlations",
            'r1': r1,
            'r2': r2,
            'z_score': float(z_score),
            'p_value': float(p_value),
            'is_significant': p_value < self.alpha,
            'conclusion': conclusion
        }

    def _insufficient_data(self, test_name: str,
                          var1: Any, var2: Any) -> CorrelationResult:
        """Create result for insufficient data."""
        return CorrelationResult(
            test_name=test_name,
            variable1=str(var1) or "X",
            variable2=str(var2) or "Y",
            correlation=np.nan,
            p_value=np.nan,
            n=0,
            conclusion="Insufficient data to calculate correlation"
        )

    def _format_p(self, p: float) -> str:
        """Format p-value for display."""
        if p < 0.001:
            return "< 0.001"
        return f"{p:.3f}"

    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation strength."""
        if np.isnan(r):
            return "N/A"
        r = abs(r)
        if r < 0.1:
            return "negligible"
        elif r < 0.3:
            return "weak"
        elif r < 0.5:
            return "moderate"
        elif r < 0.7:
            return "strong"
        else:
            return "very strong"
