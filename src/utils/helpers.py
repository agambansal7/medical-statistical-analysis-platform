"""Helper functions for formatting and utilities."""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from tabulate import tabulate


def format_pvalue(p: float, threshold: float = 0.001) -> str:
    """Format p-value for publication.

    Args:
        p: P-value to format
        threshold: Threshold below which to show as < threshold

    Returns:
        Formatted p-value string
    """
    if p is None or np.isnan(p):
        return "N/A"
    elif p < threshold:
        return f"< {threshold}"
    elif p < 0.01:
        return f"{p:.3f}"
    elif p < 0.05:
        return f"{p:.3f}"
    else:
        return f"{p:.2f}"


def format_ci(lower: float, upper: float, decimals: int = 2) -> str:
    """Format confidence interval.

    Args:
        lower: Lower bound
        upper: Upper bound
        decimals: Number of decimal places

    Returns:
        Formatted CI string
    """
    return f"[{lower:.{decimals}f}, {upper:.{decimals}f}]"


def format_mean_sd(mean: float, sd: float, decimals: int = 2) -> str:
    """Format mean ± SD."""
    return f"{mean:.{decimals}f} ± {sd:.{decimals}f}"


def format_median_iqr(median: float, q1: float, q3: float, decimals: int = 2) -> str:
    """Format median [IQR]."""
    return f"{median:.{decimals}f} [{q1:.{decimals}f}, {q3:.{decimals}f}]"


def format_percentage(count: int, total: int, decimals: int = 1) -> str:
    """Format count and percentage."""
    pct = (count / total) * 100 if total > 0 else 0
    return f"{count} ({pct:.{decimals}f}%)"


def create_table(data: List[Dict[str, Any]],
                 headers: Optional[List[str]] = None,
                 format: str = "grid") -> str:
    """Create formatted table from data.

    Args:
        data: List of dictionaries with data
        headers: Optional custom headers
        format: Table format (grid, pipe, html, latex)

    Returns:
        Formatted table string
    """
    if not data:
        return "No data available"

    df = pd.DataFrame(data)
    if headers:
        df.columns = headers

    return tabulate(df, headers='keys', tablefmt=format, showindex=False)


def interpret_effect_size(effect_size: float,
                          effect_type: str = "cohen_d") -> str:
    """Interpret effect size magnitude.

    Args:
        effect_size: The effect size value
        effect_type: Type of effect size

    Returns:
        Interpretation string
    """
    abs_effect = abs(effect_size)

    if effect_type in ["cohen_d", "hedges_g"]:
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"

    elif effect_type in ["eta_squared", "partial_eta_squared"]:
        if abs_effect < 0.01:
            return "negligible"
        elif abs_effect < 0.06:
            return "small"
        elif abs_effect < 0.14:
            return "medium"
        else:
            return "large"

    elif effect_type == "r":
        if abs_effect < 0.1:
            return "negligible"
        elif abs_effect < 0.3:
            return "small"
        elif abs_effect < 0.5:
            return "medium"
        else:
            return "large"

    elif effect_type == "odds_ratio":
        if 0.9 <= abs_effect <= 1.1:
            return "negligible"
        elif 0.67 <= abs_effect <= 1.5:
            return "small"
        elif 0.33 <= abs_effect <= 3.0:
            return "medium"
        else:
            return "large"

    return "unknown"


def get_significance_stars(p_value: float) -> str:
    """Get significance stars for p-value.

    Args:
        p_value: The p-value

    Returns:
        Significance stars string
    """
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"


def safe_divide(numerator: float, denominator: float,
                default: float = 0.0) -> float:
    """Safely divide two numbers."""
    if denominator == 0:
        return default
    return numerator / denominator


def detect_variable_type(series: pd.Series) -> str:
    """Detect the statistical type of a variable.

    Args:
        series: Pandas series to analyze

    Returns:
        Variable type: 'continuous', 'categorical', 'binary', 'ordinal', 'datetime'
    """
    # Remove missing values for analysis
    clean_series = series.dropna()

    if len(clean_series) == 0:
        return "unknown"

    # Check for datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # Check for numeric
    if pd.api.types.is_numeric_dtype(series):
        n_unique = clean_series.nunique()

        # Binary
        if n_unique == 2:
            return "binary"

        # Likely categorical (few unique values)
        if n_unique <= 10 and n_unique / len(clean_series) < 0.05:
            return "categorical"

        # Continuous
        return "continuous"

    # String/object type
    n_unique = clean_series.nunique()

    if n_unique == 2:
        return "binary"
    elif n_unique <= 20:
        return "categorical"
    else:
        return "text"
