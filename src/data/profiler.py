"""Data profiling module for automatic dataset analysis."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from dataclasses import dataclass, field, asdict


@dataclass
class VariableProfile:
    """Profile for a single variable."""
    name: str
    dtype: str
    statistical_type: str  # continuous, categorical, binary, ordinal, datetime
    n_total: int
    n_missing: int
    missing_pct: float
    n_unique: int

    # For continuous variables
    mean: Optional[float] = None
    std: Optional[float] = None
    median: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    iqr: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    is_normal: Optional[bool] = None
    normality_pvalue: Optional[float] = None

    # For categorical variables
    categories: Optional[List[str]] = None
    category_counts: Optional[Dict[str, int]] = None
    mode: Optional[str] = None

    # Outliers
    n_outliers: Optional[int] = None
    outlier_pct: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class DatasetProfile:
    """Complete profile for a dataset."""
    n_rows: int
    n_columns: int
    n_continuous: int
    n_categorical: int
    n_binary: int
    n_datetime: int
    total_missing: int
    total_missing_pct: float
    complete_cases: int
    complete_cases_pct: float
    variables: List[VariableProfile] = field(default_factory=list)
    potential_id_columns: List[str] = field(default_factory=list)
    potential_outcome_columns: List[str] = field(default_factory=list)
    potential_group_columns: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['variables'] = [v.to_dict() for v in self.variables]
        return result


class DataProfiler:
    """Automatic data profiler for statistical analysis."""

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.profile: Optional[DatasetProfile] = None

    def profile_dataset(self, df: pd.DataFrame) -> DatasetProfile:
        """Generate complete profile for dataset.

        Args:
            df: DataFrame to profile

        Returns:
            DatasetProfile object
        """
        # Basic counts
        n_rows, n_columns = df.shape

        # Profile each variable
        variables = []
        n_continuous = 0
        n_categorical = 0
        n_binary = 0
        n_datetime = 0

        for col in df.columns:
            var_profile = self._profile_variable(df[col], col)
            variables.append(var_profile)

            if var_profile.statistical_type == 'continuous':
                n_continuous += 1
            elif var_profile.statistical_type == 'categorical':
                n_categorical += 1
            elif var_profile.statistical_type == 'binary':
                n_binary += 1
            elif var_profile.statistical_type == 'datetime':
                n_datetime += 1

        # Missing data summary
        total_missing = df.isnull().sum().sum()
        total_cells = n_rows * n_columns
        total_missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0

        complete_cases = df.dropna().shape[0]
        complete_cases_pct = (complete_cases / n_rows * 100) if n_rows > 0 else 0

        # Detect potential column roles
        potential_id = self._detect_id_columns(df, variables)
        potential_outcome = self._detect_outcome_columns(df, variables)
        potential_group = self._detect_group_columns(df, variables)

        # Generate warnings
        warnings = self._generate_warnings(df, variables)

        self.profile = DatasetProfile(
            n_rows=n_rows,
            n_columns=n_columns,
            n_continuous=n_continuous,
            n_categorical=n_categorical,
            n_binary=n_binary,
            n_datetime=n_datetime,
            total_missing=total_missing,
            total_missing_pct=round(total_missing_pct, 2),
            complete_cases=complete_cases,
            complete_cases_pct=round(complete_cases_pct, 2),
            variables=variables,
            potential_id_columns=potential_id,
            potential_outcome_columns=potential_outcome,
            potential_group_columns=potential_group,
            warnings=warnings
        )

        return self.profile

    def _profile_variable(self, series: pd.Series, name: str) -> VariableProfile:
        """Profile a single variable."""
        n_total = len(series)
        n_missing = series.isnull().sum()
        missing_pct = (n_missing / n_total * 100) if n_total > 0 else 0

        clean_series = series.dropna()
        n_unique = clean_series.nunique()

        # Determine statistical type
        stat_type = self._determine_statistical_type(series, n_unique)

        profile = VariableProfile(
            name=name,
            dtype=str(series.dtype),
            statistical_type=stat_type,
            n_total=n_total,
            n_missing=int(n_missing),
            missing_pct=round(missing_pct, 2),
            n_unique=n_unique
        )

        if stat_type == 'continuous':
            self._add_continuous_stats(profile, clean_series)
        elif stat_type in ['categorical', 'binary']:
            self._add_categorical_stats(profile, clean_series)

        return profile

    def _determine_statistical_type(self, series: pd.Series, n_unique: int) -> str:
        """Determine the statistical type of a variable."""
        clean_series = series.dropna()
        n_total = len(clean_series)

        if n_total == 0:
            return 'unknown'

        # Check datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime'

        # Check numeric
        if pd.api.types.is_numeric_dtype(series):
            if n_unique == 2:
                return 'binary'
            elif n_unique <= 10 and (n_unique / n_total) < 0.05:
                return 'categorical'
            else:
                return 'continuous'

        # String/object
        if n_unique == 2:
            return 'binary'
        elif n_unique <= 20:
            return 'categorical'
        else:
            return 'text'

    def _add_continuous_stats(self, profile: VariableProfile, series: pd.Series):
        """Add statistics for continuous variables."""
        if len(series) == 0:
            return

        profile.mean = float(series.mean())
        profile.std = float(series.std())
        profile.median = float(series.median())
        profile.min = float(series.min())
        profile.max = float(series.max())
        profile.q1 = float(series.quantile(0.25))
        profile.q3 = float(series.quantile(0.75))
        profile.iqr = profile.q3 - profile.q1
        profile.skewness = float(series.skew())
        profile.kurtosis = float(series.kurtosis())

        # Normality test
        if len(series) >= 3:
            try:
                if len(series) < 50:
                    stat, pvalue = stats.shapiro(series)
                else:
                    stat, pvalue = stats.normaltest(series)
                profile.is_normal = pvalue > self.significance_level
                profile.normality_pvalue = float(pvalue)
            except:
                profile.is_normal = None
                profile.normality_pvalue = None

        # Outliers (IQR method)
        if profile.iqr and profile.iqr > 0:
            lower_bound = profile.q1 - 1.5 * profile.iqr
            upper_bound = profile.q3 + 1.5 * profile.iqr
            outliers = (series < lower_bound) | (series > upper_bound)
            profile.n_outliers = int(outliers.sum())
            profile.outlier_pct = round(outliers.sum() / len(series) * 100, 2)

    def _add_categorical_stats(self, profile: VariableProfile, series: pd.Series):
        """Add statistics for categorical variables."""
        if len(series) == 0:
            return

        value_counts = series.value_counts()
        profile.categories = list(value_counts.index.astype(str))
        profile.category_counts = {str(k): int(v) for k, v in value_counts.items()}
        profile.mode = str(value_counts.index[0]) if len(value_counts) > 0 else None

    def _detect_id_columns(self, df: pd.DataFrame,
                           variables: List[VariableProfile]) -> List[str]:
        """Detect potential ID columns."""
        id_columns = []
        n_rows = len(df)

        for var in variables:
            # Check if all unique
            if var.n_unique == n_rows and var.n_missing == 0:
                id_columns.append(var.name)
            # Check common ID column names
            elif var.name.lower() in ['id', 'subject_id', 'patient_id', 'record_id',
                                       'study_id', 'participant_id', 'case_id']:
                id_columns.append(var.name)

        return id_columns

    def _detect_outcome_columns(self, df: pd.DataFrame,
                                 variables: List[VariableProfile]) -> List[str]:
        """Detect potential outcome columns."""
        outcome_hints = ['outcome', 'result', 'response', 'status', 'event',
                        'death', 'survival', 'score', 'endpoint', 'primary',
                        'secondary', 'success', 'failure', 'positive', 'negative']

        outcome_columns = []
        for var in variables:
            name_lower = var.name.lower()
            for hint in outcome_hints:
                if hint in name_lower:
                    outcome_columns.append(var.name)
                    break

        return outcome_columns

    def _detect_group_columns(self, df: pd.DataFrame,
                               variables: List[VariableProfile]) -> List[str]:
        """Detect potential grouping columns."""
        group_hints = ['group', 'arm', 'treatment', 'cohort', 'category',
                      'type', 'class', 'stage', 'grade', 'sex', 'gender',
                      'condition', 'intervention', 'control', 'placebo']

        group_columns = []
        for var in variables:
            if var.statistical_type in ['binary', 'categorical']:
                name_lower = var.name.lower()
                for hint in group_hints:
                    if hint in name_lower:
                        group_columns.append(var.name)
                        break
                # Also include binary/categorical with few categories
                if var.name not in group_columns and var.n_unique <= 5:
                    group_columns.append(var.name)

        return group_columns

    def _generate_warnings(self, df: pd.DataFrame,
                           variables: List[VariableProfile]) -> List[str]:
        """Generate warnings about data quality."""
        warnings = []

        # Small sample size
        if len(df) < 30:
            warnings.append(f"Small sample size (n={len(df)}). "
                          "Consider non-parametric tests.")

        # High missing data
        for var in variables:
            if var.missing_pct > 20:
                warnings.append(f"High missing data in '{var.name}' "
                              f"({var.missing_pct}%). Consider imputation.")

        # Non-normal distributions
        non_normal = [var.name for var in variables
                      if var.is_normal is False and var.statistical_type == 'continuous']
        if non_normal:
            warnings.append(f"Non-normal distribution detected in: "
                          f"{', '.join(non_normal[:5])}. "
                          "Consider non-parametric alternatives.")

        # High outlier percentage
        for var in variables:
            if var.outlier_pct and var.outlier_pct > 5:
                warnings.append(f"High outlier percentage in '{var.name}' "
                              f"({var.outlier_pct}%). Review data quality.")

        # Constant columns
        constant_cols = [var.name for var in variables if var.n_unique == 1]
        if constant_cols:
            warnings.append(f"Constant columns (no variation): "
                          f"{', '.join(constant_cols)}")

        return warnings

    def get_variable_by_name(self, name: str) -> Optional[VariableProfile]:
        """Get variable profile by name."""
        if self.profile is None:
            return None
        for var in self.profile.variables:
            if var.name == name:
                return var
        return None

    def get_continuous_variables(self) -> List[str]:
        """Get names of continuous variables."""
        if self.profile is None:
            return []
        return [var.name for var in self.profile.variables
                if var.statistical_type == 'continuous']

    def get_categorical_variables(self) -> List[str]:
        """Get names of categorical variables."""
        if self.profile is None:
            return []
        return [var.name for var in self.profile.variables
                if var.statistical_type in ['categorical', 'binary']]

    def get_binary_variables(self) -> List[str]:
        """Get names of binary variables."""
        if self.profile is None:
            return []
        return [var.name for var in self.profile.variables
                if var.statistical_type == 'binary']

    def generate_summary_text(self) -> str:
        """Generate human-readable summary of the dataset."""
        if self.profile is None:
            return "No data profiled yet."

        p = self.profile

        summary = f"""
Dataset Summary
===============
Total observations: {p.n_rows}
Total variables: {p.n_columns}

Variable Types:
- Continuous: {p.n_continuous}
- Categorical: {p.n_categorical}
- Binary: {p.n_binary}
- Datetime: {p.n_datetime}

Data Quality:
- Missing values: {p.total_missing} ({p.total_missing_pct}%)
- Complete cases: {p.complete_cases} ({p.complete_cases_pct}%)

Potential Column Roles:
- ID columns: {', '.join(p.potential_id_columns) or 'None detected'}
- Outcome columns: {', '.join(p.potential_outcome_columns) or 'None detected'}
- Grouping columns: {', '.join(p.potential_group_columns) or 'None detected'}
"""

        if p.warnings:
            summary += "\nWarnings:\n"
            for warning in p.warnings:
                summary += f"- {warning}\n"

        return summary.strip()
