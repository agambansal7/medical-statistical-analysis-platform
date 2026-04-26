"""Descriptive statistics module."""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict


@dataclass
class ContinuousStats:
    """Descriptive statistics for continuous variables."""
    variable: str
    n: int
    n_missing: int
    mean: float
    std: float
    sem: float
    median: float
    min: float
    max: float
    range: float
    q1: float
    q3: float
    iqr: float
    skewness: float
    kurtosis: float
    ci_lower: float
    ci_upper: float
    cv: float  # Coefficient of variation

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def formatted_summary(self, use_mean: bool = True) -> str:
        """Return formatted summary string."""
        if use_mean:
            return f"{self.mean:.2f} ± {self.std:.2f}"
        else:
            return f"{self.median:.2f} [{self.q1:.2f}, {self.q3:.2f}]"


@dataclass
class CategoricalStats:
    """Descriptive statistics for categorical variables."""
    variable: str
    n: int
    n_missing: int
    n_categories: int
    mode: str
    mode_count: int
    mode_percentage: float
    categories: Dict[str, Dict[str, Any]]  # {category: {count, percentage}}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DescriptiveStats:
    """Compute descriptive statistics."""

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level

    def continuous_stats(self, data: pd.Series,
                        variable_name: Optional[str] = None) -> ContinuousStats:
        """Calculate descriptive statistics for continuous variable.

        Args:
            data: Numeric data series
            variable_name: Optional name for the variable

        Returns:
            ContinuousStats object
        """
        name = variable_name or str(data.name) or "Variable"
        clean_data = data.dropna()

        n = len(clean_data)
        n_missing = len(data) - n

        if n == 0:
            return ContinuousStats(
                variable=name, n=0, n_missing=n_missing,
                mean=np.nan, std=np.nan, sem=np.nan, median=np.nan,
                min=np.nan, max=np.nan, range=np.nan,
                q1=np.nan, q3=np.nan, iqr=np.nan,
                skewness=np.nan, kurtosis=np.nan,
                ci_lower=np.nan, ci_upper=np.nan, cv=np.nan
            )

        mean = float(clean_data.mean())
        std = float(clean_data.std())
        sem = std / np.sqrt(n)

        # Confidence interval
        t_crit = stats.t.ppf((1 + self.confidence_level) / 2, n - 1)
        ci_margin = t_crit * sem
        ci_lower = mean - ci_margin
        ci_upper = mean + ci_margin

        # Coefficient of variation
        cv = (std / mean * 100) if mean != 0 else np.nan

        return ContinuousStats(
            variable=name,
            n=n,
            n_missing=n_missing,
            mean=mean,
            std=std,
            sem=float(sem),
            median=float(clean_data.median()),
            min=float(clean_data.min()),
            max=float(clean_data.max()),
            range=float(clean_data.max() - clean_data.min()),
            q1=float(clean_data.quantile(0.25)),
            q3=float(clean_data.quantile(0.75)),
            iqr=float(clean_data.quantile(0.75) - clean_data.quantile(0.25)),
            skewness=float(clean_data.skew()),
            kurtosis=float(clean_data.kurtosis()),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            cv=float(cv) if not np.isnan(cv) else np.nan
        )

    def categorical_stats(self, data: pd.Series,
                         variable_name: Optional[str] = None) -> CategoricalStats:
        """Calculate descriptive statistics for categorical variable.

        Args:
            data: Categorical data series
            variable_name: Optional name for the variable

        Returns:
            CategoricalStats object
        """
        name = variable_name or str(data.name) or "Variable"
        clean_data = data.dropna()

        n = len(clean_data)
        n_missing = len(data) - n

        if n == 0:
            return CategoricalStats(
                variable=name, n=0, n_missing=n_missing,
                n_categories=0, mode="N/A", mode_count=0,
                mode_percentage=0, categories={}
            )

        value_counts = clean_data.value_counts()
        categories = {}
        for cat, count in value_counts.items():
            categories[str(cat)] = {
                'count': int(count),
                'percentage': round(count / n * 100, 2)
            }

        mode = str(value_counts.index[0])
        mode_count = int(value_counts.iloc[0])
        mode_percentage = round(mode_count / n * 100, 2)

        return CategoricalStats(
            variable=name,
            n=n,
            n_missing=n_missing,
            n_categories=len(value_counts),
            mode=mode,
            mode_count=mode_count,
            mode_percentage=mode_percentage,
            categories=categories
        )

    def grouped_continuous_stats(self, data: pd.DataFrame,
                                 value_col: str,
                                 group_col: str) -> Dict[str, ContinuousStats]:
        """Calculate continuous stats by group.

        Args:
            data: DataFrame containing the data
            value_col: Column name for the continuous variable
            group_col: Column name for the grouping variable

        Returns:
            Dictionary of ContinuousStats by group
        """
        results = {}
        for group_name, group_data in data.groupby(group_col):
            results[str(group_name)] = self.continuous_stats(
                group_data[value_col],
                f"{value_col} ({group_name})"
            )
        return results

    def generate_table1(self, data: pd.DataFrame,
                       group_col: Optional[str] = None,
                       continuous_vars: Optional[List[str]] = None,
                       categorical_vars: Optional[List[str]] = None,
                       use_median: Optional[List[str]] = None) -> pd.DataFrame:
        """Generate Table 1 (baseline characteristics table).

        Args:
            data: DataFrame with the data
            group_col: Optional grouping column
            continuous_vars: List of continuous variable names
            categorical_vars: List of categorical variable names
            use_median: List of variables to report as median [IQR]

        Returns:
            DataFrame with Table 1 format
        """
        use_median = use_median or []
        rows = []

        if group_col:
            groups = data[group_col].dropna().unique()
        else:
            groups = ['All']

        # Header row
        header = ['Variable']
        for g in groups:
            header.append(str(g))
        if group_col and len(groups) > 1:
            header.append('p-value')

        # Total N row
        n_row = {'Variable': 'N'}
        for g in groups:
            if group_col:
                n = len(data[data[group_col] == g])
            else:
                n = len(data)
            n_row[str(g)] = str(n)
        rows.append(n_row)

        # Continuous variables
        if continuous_vars:
            for var in continuous_vars:
                var_row = {'Variable': var}

                for g in groups:
                    if group_col:
                        subset = data[data[group_col] == g][var]
                    else:
                        subset = data[var]

                    stats_result = self.continuous_stats(subset)

                    if var in use_median:
                        var_row[str(g)] = stats_result.formatted_summary(use_mean=False)
                    else:
                        var_row[str(g)] = stats_result.formatted_summary(use_mean=True)

                # Add p-value if comparing groups
                if group_col and len(groups) > 1:
                    group_data = [data[data[group_col] == g][var].dropna()
                                 for g in groups]
                    if len(groups) == 2:
                        _, p = stats.ttest_ind(group_data[0], group_data[1])
                    else:
                        _, p = stats.f_oneway(*group_data)
                    var_row['p-value'] = self._format_pvalue(p)

                rows.append(var_row)

        # Categorical variables
        if categorical_vars:
            for var in categorical_vars:
                categories = data[var].dropna().unique()

                # Variable header row
                var_row = {'Variable': f"{var}, n (%)"}
                for g in groups:
                    var_row[str(g)] = ''
                rows.append(var_row)

                # Category rows
                for cat in categories:
                    cat_row = {'Variable': f"  {cat}"}

                    for g in groups:
                        if group_col:
                            subset = data[data[group_col] == g][var]
                        else:
                            subset = data[var]

                        count = (subset == cat).sum()
                        total = subset.notna().sum()
                        pct = count / total * 100 if total > 0 else 0
                        cat_row[str(g)] = f"{count} ({pct:.1f}%)"

                    rows.append(cat_row)

                # Add p-value for categorical variable
                if group_col and len(groups) > 1:
                    contingency = pd.crosstab(data[group_col], data[var])
                    _, p, _, _ = stats.chi2_contingency(contingency)
                    # Add p-value to variable header row
                    rows[-len(categories)]['p-value'] = self._format_pvalue(p)
                    for i in range(len(categories) - 1):
                        rows[-len(categories) + i + 1]['p-value'] = ''

        return pd.DataFrame(rows)

    def generate_enhanced_table1(self, data: pd.DataFrame,
                                  group_col: str,
                                  continuous_vars: Optional[List[str]] = None,
                                  categorical_vars: Optional[List[str]] = None,
                                  use_median: Optional[List[str]] = None,
                                  include_smd: bool = True,
                                  include_missing: bool = True,
                                  auto_detect: bool = True) -> Dict[str, Any]:
        """Generate enhanced Table 1 with SMD and better formatting.

        Args:
            data: DataFrame with the data
            group_col: Grouping column (required)
            continuous_vars: List of continuous variable names
            categorical_vars: List of categorical variable names
            use_median: List of variables to report as median [IQR]
            include_smd: Whether to include Standardized Mean Difference
            include_missing: Whether to include missing data row for each variable
            auto_detect: Automatically detect variable types if not specified

        Returns:
            Dictionary with table data, HTML, and metadata
        """
        use_median = use_median or []

        # Auto-detect variable types if not specified
        if auto_detect and not continuous_vars and not categorical_vars:
            continuous_vars = []
            categorical_vars = []
            for col in data.columns:
                if col == group_col:
                    continue
                if data[col].dtype in ['float64', 'int64'] and data[col].nunique() > 10:
                    continuous_vars.append(col)
                elif data[col].dtype in ['object', 'category', 'bool'] or data[col].nunique() <= 10:
                    categorical_vars.append(col)

        continuous_vars = continuous_vars or []
        categorical_vars = categorical_vars or []

        groups = sorted(data[group_col].dropna().unique())
        rows = []

        # Total N row
        n_row = {
            'Variable': 'N',
            'Overall': str(len(data)),
        }
        for g in groups:
            n_row[str(g)] = str(len(data[data[group_col] == g]))
        rows.append(n_row)

        # Continuous variables
        for var in continuous_vars:
            if var not in data.columns:
                continue

            var_label = var.replace('_', ' ').title()
            if var in use_median:
                var_label += ', median [IQR]'
            else:
                var_label += ', mean (SD)'

            var_row = {'Variable': var_label}

            # Overall
            overall_stats = self.continuous_stats(data[var])
            if var in use_median:
                var_row['Overall'] = f"{overall_stats.median:.1f} [{overall_stats.q1:.1f}-{overall_stats.q3:.1f}]"
            else:
                var_row['Overall'] = f"{overall_stats.mean:.1f} ({overall_stats.std:.1f})"

            group_means = []
            group_stds = []
            group_data = []

            for g in groups:
                subset = data[data[group_col] == g][var].dropna()
                group_data.append(subset)
                stats_result = self.continuous_stats(subset)
                group_means.append(stats_result.mean)
                group_stds.append(stats_result.std)

                if var in use_median:
                    var_row[str(g)] = f"{stats_result.median:.1f} [{stats_result.q1:.1f}-{stats_result.q3:.1f}]"
                else:
                    var_row[str(g)] = f"{stats_result.mean:.1f} ({stats_result.std:.1f})"

            # Calculate SMD (for binary comparisons or vs first group)
            if include_smd and len(groups) >= 2:
                smd = self._calculate_smd_continuous(group_means[0], group_stds[0], len(group_data[0]),
                                                      group_means[1], group_stds[1], len(group_data[1]))
                var_row['SMD'] = f"{smd:.3f}"

            # Calculate p-value
            if len(groups) == 2:
                _, p = stats.ttest_ind(group_data[0], group_data[1])
            else:
                _, p = stats.f_oneway(*group_data)
            var_row['p-value'] = self._format_pvalue(p)

            # Missing data
            if include_missing:
                missing_pct = data[var].isna().mean() * 100
                var_row['Missing'] = f"{missing_pct:.1f}%"

            rows.append(var_row)

        # Categorical variables
        for var in categorical_vars:
            if var not in data.columns:
                continue

            categories = [c for c in data[var].dropna().unique() if pd.notna(c)]
            var_label = var.replace('_', ' ').title() + ', n (%)'

            # Variable header row with p-value
            var_row = {'Variable': var_label}
            var_row['Overall'] = ''
            for g in groups:
                var_row[str(g)] = ''

            # Chi-square p-value
            contingency = pd.crosstab(data[group_col], data[var])
            _, p, _, _ = stats.chi2_contingency(contingency)
            var_row['p-value'] = self._format_pvalue(p)

            # Missing data
            if include_missing:
                missing_pct = data[var].isna().mean() * 100
                var_row['Missing'] = f"{missing_pct:.1f}%"

            if include_smd and len(groups) >= 2:
                var_row['SMD'] = ''

            rows.append(var_row)

            # Category rows
            for cat in categories:
                cat_row = {'Variable': f"  {cat}"}

                # Overall
                total_count = (data[var] == cat).sum()
                total_n = data[var].notna().sum()
                cat_row['Overall'] = f"{total_count} ({total_count/total_n*100:.1f}%)"

                proportions = []
                counts = []

                for g in groups:
                    subset = data[data[group_col] == g][var]
                    count = (subset == cat).sum()
                    total = subset.notna().sum()
                    pct = count / total * 100 if total > 0 else 0
                    cat_row[str(g)] = f"{count} ({pct:.1f}%)"
                    proportions.append(pct / 100 if total > 0 else 0)
                    counts.append((count, total))

                # Calculate SMD for categorical (for binary comparisons)
                if include_smd and len(groups) >= 2:
                    smd = self._calculate_smd_categorical(proportions[0], proportions[1])
                    cat_row['SMD'] = f"{smd:.3f}"

                cat_row['p-value'] = ''
                if include_missing:
                    cat_row['Missing'] = ''

                rows.append(cat_row)

        # Create DataFrame
        table_df = pd.DataFrame(rows)

        # Generate HTML with styling
        html = self._generate_table1_html(table_df, groups, group_col)

        return {
            'table_data': rows,
            'table_df': table_df.to_dict(orient='records'),
            'table_html': html,
            'groups': [str(g) for g in groups],
            'group_col': group_col,
            'n_continuous': len(continuous_vars),
            'n_categorical': len(categorical_vars),
            'continuous_vars': continuous_vars,
            'categorical_vars': categorical_vars
        }

    def _calculate_smd_continuous(self, mean1: float, sd1: float, n1: int,
                                   mean2: float, sd2: float, n2: int) -> float:
        """Calculate Standardized Mean Difference (Cohen's d) for continuous variables."""
        if sd1 == 0 and sd2 == 0:
            return 0.0

        # Pooled standard deviation
        pooled_sd = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))

        if pooled_sd == 0:
            return 0.0

        return abs(mean1 - mean2) / pooled_sd

    def _calculate_smd_categorical(self, p1: float, p2: float) -> float:
        """Calculate SMD for categorical/binary variables."""
        # Use the formula: SMD = (p1 - p2) / sqrt((p1(1-p1) + p2(1-p2)) / 2)
        var1 = p1 * (1 - p1) if 0 < p1 < 1 else 0.001
        var2 = p2 * (1 - p2) if 0 < p2 < 1 else 0.001
        pooled_var = (var1 + var2) / 2

        if pooled_var <= 0:
            return 0.0

        return abs(p1 - p2) / np.sqrt(pooled_var)

    def _generate_table1_html(self, df: pd.DataFrame, groups: List, group_col: str) -> str:
        """Generate publication-quality HTML for Table 1."""
        html = """<style>
.table1 { border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 12px; }
.table1 th { background-color: #4f46e5; color: white; padding: 10px; text-align: left; font-weight: bold; }
.table1 td { padding: 8px; border-bottom: 1px solid #e5e7eb; }
.table1 tr:nth-child(even) { background-color: #f9fafb; }
.table1 tr:hover { background-color: #f3f4f6; }
.table1 .var-header { font-weight: bold; background-color: #f3f4f6; }
.table1 .indent { padding-left: 20px; }
.table1 .p-sig { color: #059669; font-weight: bold; }
.table1 .smd-high { color: #dc2626; }
.table1 .smd-moderate { color: #d97706; }
.table1 .missing-high { color: #dc2626; }
</style>
<table class="table1">
<thead>
<tr>
"""
        # Header row
        for col in df.columns:
            if col == 'Variable':
                html += f'<th style="width: 200px;">{col}</th>\n'
            elif col == 'p-value':
                html += f'<th style="width: 80px; text-align: center;">{col}</th>\n'
            elif col == 'SMD':
                html += f'<th style="width: 60px; text-align: center;">{col}</th>\n'
            elif col == 'Missing':
                html += f'<th style="width: 70px; text-align: center;">{col}</th>\n'
            else:
                # Group column - show N in header
                n = df[df['Variable'] == 'N'][col].values[0] if 'N' in df['Variable'].values else ''
                html += f'<th style="text-align: center;">{col}<br/>(n={n})</th>\n'

        html += '</tr>\n</thead>\n<tbody>\n'

        # Data rows (skip N row since it's in header)
        for _, row in df.iterrows():
            if row['Variable'] == 'N':
                continue

            # Check if this is a variable header (contains 'n (%)' or 'mean' or 'median')
            is_header = any(x in str(row['Variable']) for x in ['n (%)', 'mean', 'median', 'Mean', 'Median'])
            is_indent = row['Variable'].startswith('  ')

            row_class = 'var-header' if is_header else ''

            html += f'<tr class="{row_class}">\n'

            for col in df.columns:
                value = row[col]
                td_class = ''
                style = ''

                if col == 'Variable':
                    if is_indent:
                        td_class = 'indent'
                        value = value.strip()
                elif col == 'p-value':
                    style = 'text-align: center;'
                    try:
                        p_val = float(value.replace('<', '')) if value and value not in ['', 'nan'] else 1.0
                        if p_val < 0.05:
                            td_class = 'p-sig'
                    except:
                        pass
                elif col == 'SMD':
                    style = 'text-align: center;'
                    try:
                        smd_val = float(value) if value and value not in ['', 'nan'] else 0.0
                        if smd_val > 0.2:
                            td_class = 'smd-high' if smd_val > 0.5 else 'smd-moderate'
                    except:
                        pass
                elif col == 'Missing':
                    style = 'text-align: center;'
                    try:
                        miss_val = float(value.replace('%', '')) if value and value not in ['', 'nan'] else 0.0
                        if miss_val > 10:
                            td_class = 'missing-high'
                    except:
                        pass
                else:
                    style = 'text-align: center;'

                html += f'<td class="{td_class}" style="{style}">{value if pd.notna(value) else ""}</td>\n'

            html += '</tr>\n'

        html += '</tbody>\n</table>'

        # Add footnotes
        html += """
<div style="font-size: 11px; color: #6b7280; margin-top: 10px;">
<p>Values are presented as mean (SD), median [IQR], or n (%).</p>
<p>SMD = Standardized Mean Difference. Values >0.1 indicate imbalance; >0.2 moderate imbalance; >0.5 large imbalance.</p>
<p>P-values from t-test/ANOVA for continuous variables and chi-square for categorical variables.</p>
</div>
"""
        return html

    def summary_statistics(self, data: pd.DataFrame,
                          variables: Optional[List[str]] = None) -> pd.DataFrame:
        """Generate summary statistics for multiple variables.

        Args:
            data: DataFrame with the data
            variables: List of variables (if None, uses all numeric)

        Returns:
            DataFrame with summary statistics
        """
        if variables is None:
            variables = data.select_dtypes(include=[np.number]).columns.tolist()

        results = []
        for var in variables:
            if var in data.columns:
                stats_result = self.continuous_stats(data[var], var)
                results.append(stats_result.to_dict())

        return pd.DataFrame(results)

    def _format_pvalue(self, p: float) -> str:
        """Format p-value for display."""
        if p < 0.001:
            return "<0.001"
        elif p < 0.01:
            return f"{p:.3f}"
        else:
            return f"{p:.2f}"

    def percentiles(self, data: pd.Series,
                   percentiles: List[float] = None) -> Dict[str, float]:
        """Calculate percentiles.

        Args:
            data: Numeric data series
            percentiles: List of percentiles (0-100)

        Returns:
            Dictionary of percentiles
        """
        if percentiles is None:
            percentiles = [5, 10, 25, 50, 75, 90, 95]

        clean_data = data.dropna()
        result = {}
        for p in percentiles:
            result[f"P{p}"] = float(np.percentile(clean_data, p))
        return result

    def frequency_table(self, data: pd.Series,
                       sort_by: str = 'count') -> pd.DataFrame:
        """Create frequency table for categorical variable.

        Args:
            data: Categorical data series
            sort_by: Sort by 'count' or 'category'

        Returns:
            DataFrame with frequency table
        """
        value_counts = data.value_counts()
        total = len(data.dropna())

        result = pd.DataFrame({
            'Category': value_counts.index,
            'Count': value_counts.values,
            'Percentage': (value_counts.values / total * 100).round(2),
            'Cumulative %': ((value_counts.cumsum().values / total) * 100).round(2)
        })

        if sort_by == 'category':
            result = result.sort_values('Category')

        return result

    def cross_tabulation(self, data: pd.DataFrame,
                        row_var: str,
                        col_var: str,
                        margins: bool = True,
                        percentages: str = 'none') -> pd.DataFrame:
        """Create cross-tabulation (contingency table).

        Args:
            data: DataFrame with the data
            row_var: Row variable name
            col_var: Column variable name
            margins: Include row/column totals
            percentages: 'none', 'row', 'column', or 'total'

        Returns:
            DataFrame with cross-tabulation
        """
        crosstab = pd.crosstab(data[row_var], data[col_var], margins=margins)

        if percentages == 'row':
            # Row percentages
            return crosstab.div(crosstab.iloc[:, -1], axis=0) * 100
        elif percentages == 'column':
            # Column percentages
            return crosstab.div(crosstab.iloc[-1, :], axis=1) * 100
        elif percentages == 'total':
            # Total percentages
            return crosstab / crosstab.iloc[-1, -1] * 100

        return crosstab
