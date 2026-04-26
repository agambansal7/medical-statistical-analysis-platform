"""Data Quality Dashboard.

Provides comprehensive data quality assessment:
- Missing data visualization (heatmap, patterns)
- Outlier flagging with suggested actions
- Distribution warnings with recommendations
- Data dictionary auto-generation
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import warnings


@dataclass
class ColumnQuality:
    """Quality assessment for a single column."""
    name: str
    dtype: str
    inferred_type: str  # continuous, categorical, binary, datetime, text, id

    # Completeness
    n_total: int
    n_missing: int
    missing_percent: float
    missing_pattern: str  # MCAR, MAR, MNAR suggestion

    # Uniqueness
    n_unique: int
    unique_percent: float
    is_id_candidate: bool

    # Distribution (for numeric)
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    is_normal: Optional[bool] = None

    # Outliers
    n_outliers: int = 0
    outlier_percent: float = 0.0
    outlier_indices: List[int] = field(default_factory=list)

    # Categories (for categorical)
    categories: Optional[List[str]] = None
    category_counts: Optional[Dict[str, int]] = None
    imbalance_ratio: Optional[float] = None

    # Warnings and recommendations
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DataQualityReport:
    """Complete data quality report."""
    timestamp: str
    n_rows: int
    n_columns: int

    # Overall quality scores
    completeness_score: float  # 0-100
    validity_score: float
    consistency_score: float
    overall_score: float

    # Column-level assessments
    columns: Dict[str, ColumnQuality] = field(default_factory=dict)

    # Missing data analysis
    missing_pattern_matrix: Optional[List[List[int]]] = None
    missing_correlations: Optional[Dict[str, Dict[str, float]]] = None

    # Data dictionary
    data_dictionary: List[Dict[str, Any]] = field(default_factory=list)

    # Recommendations
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'n_rows': self.n_rows,
            'n_columns': self.n_columns,
            'completeness_score': self.completeness_score,
            'validity_score': self.validity_score,
            'consistency_score': self.consistency_score,
            'overall_score': self.overall_score,
            'columns': {k: vars(v) for k, v in self.columns.items()},
            'critical_issues': self.critical_issues,
            'warnings': self.warnings,
            'suggestions': self.suggestions,
            'data_dictionary': self.data_dictionary
        }


class DataQualityDashboard:
    """Comprehensive data quality assessment."""

    # Thresholds
    MISSING_WARNING_THRESHOLD = 0.05
    MISSING_CRITICAL_THRESHOLD = 0.20
    OUTLIER_THRESHOLD = 0.01
    IMBALANCE_WARNING = 10  # 10:1 ratio
    CARDINALITY_ID_THRESHOLD = 0.95

    def __init__(self):
        self.report = None

    def assess(self, data: pd.DataFrame) -> DataQualityReport:
        """Run complete data quality assessment.

        Args:
            data: DataFrame to assess

        Returns:
            DataQualityReport with comprehensive assessment
        """
        n_rows, n_columns = data.shape

        # Assess each column
        columns = {}
        for col in data.columns:
            columns[col] = self._assess_column(data, col)

        # Calculate overall scores
        completeness_score = self._calculate_completeness_score(columns)
        validity_score = self._calculate_validity_score(columns)
        consistency_score = self._calculate_consistency_score(data, columns)
        overall_score = (completeness_score + validity_score + consistency_score) / 3

        # Analyze missing data patterns
        missing_matrix = self._create_missing_matrix(data)
        missing_correlations = self._analyze_missing_correlations(data)

        # Generate data dictionary
        data_dict = self._generate_data_dictionary(data, columns)

        # Compile issues and recommendations
        critical_issues, warnings, suggestions = self._compile_recommendations(columns, data)

        self.report = DataQualityReport(
            timestamp=datetime.now().isoformat(),
            n_rows=n_rows,
            n_columns=n_columns,
            completeness_score=completeness_score,
            validity_score=validity_score,
            consistency_score=consistency_score,
            overall_score=overall_score,
            columns=columns,
            missing_pattern_matrix=missing_matrix,
            missing_correlations=missing_correlations,
            data_dictionary=data_dict,
            critical_issues=critical_issues,
            warnings=warnings,
            suggestions=suggestions
        )

        return self.report

    def _assess_column(self, data: pd.DataFrame, col: str) -> ColumnQuality:
        """Assess quality of a single column."""
        series = data[col]
        n_total = len(series)
        n_missing = series.isna().sum()
        missing_pct = n_missing / n_total if n_total > 0 else 0
        n_unique = series.nunique()
        unique_pct = n_unique / n_total if n_total > 0 else 0

        # Infer type
        dtype = str(series.dtype)
        inferred_type = self._infer_type(series, n_unique, n_total)

        # Check if ID candidate
        is_id = unique_pct > self.CARDINALITY_ID_THRESHOLD and inferred_type != 'datetime'

        quality = ColumnQuality(
            name=col,
            dtype=dtype,
            inferred_type=inferred_type,
            n_total=n_total,
            n_missing=n_missing,
            missing_percent=missing_pct,
            missing_pattern=self._infer_missing_pattern(series, data),
            n_unique=n_unique,
            unique_percent=unique_pct,
            is_id_candidate=is_id
        )

        # Type-specific analysis
        if inferred_type == 'continuous':
            self._assess_numeric(series, quality)
        elif inferred_type in ['categorical', 'binary']:
            self._assess_categorical(series, quality)

        # Generate warnings and recommendations
        self._generate_column_recommendations(quality)

        return quality

    def _infer_type(self, series: pd.Series, n_unique: int, n_total: int) -> str:
        """Infer the semantic type of a column."""
        dtype = str(series.dtype)

        # Check for datetime
        if 'datetime' in dtype:
            return 'datetime'

        # Check for numeric
        if dtype in ['float64', 'float32', 'int64', 'int32']:
            if n_unique == 2:
                return 'binary'
            elif n_unique <= 10:
                return 'categorical'
            else:
                return 'continuous'

        # Check for text
        if dtype == 'object':
            if n_unique == 2:
                return 'binary'
            elif n_unique <= 20 or n_unique / n_total < 0.05:
                return 'categorical'
            elif n_unique / n_total > 0.95:
                return 'id'
            else:
                return 'text'

        return 'unknown'

    def _infer_missing_pattern(self, series: pd.Series, data: pd.DataFrame) -> str:
        """Infer the likely missing data mechanism."""
        if series.isna().sum() == 0:
            return 'complete'

        # Check if missingness is related to other variables
        missing_indicator = series.isna().astype(int)

        # Simple heuristic: check correlation with other variables
        correlations = []
        for other_col in data.columns:
            if other_col != series.name and data[other_col].dtype in ['float64', 'int64']:
                try:
                    corr = data[other_col].corr(missing_indicator)
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                except:
                    pass

        if correlations:
            max_corr = max(correlations)
            if max_corr > 0.3:
                return 'MAR (possibly)'  # Missing at Random
            elif max_corr > 0.1:
                return 'MCAR (possibly)'  # Missing Completely at Random

        return 'MCAR (possibly)'

    def _assess_numeric(self, series: pd.Series, quality: ColumnQuality):
        """Assess numeric column statistics."""
        clean = series.dropna()

        if len(clean) < 3:
            return

        quality.mean = float(clean.mean())
        quality.median = float(clean.median())
        quality.std = float(clean.std())
        quality.min_val = float(clean.min())
        quality.max_val = float(clean.max())
        quality.skewness = float(clean.skew())
        quality.kurtosis = float(clean.kurtosis())

        # Test normality
        if len(clean) >= 8:
            try:
                _, p_value = stats.shapiro(clean.sample(min(len(clean), 5000)))
                quality.is_normal = p_value > 0.05
            except:
                quality.is_normal = None

        # Detect outliers using IQR method
        q1 = clean.quantile(0.25)
        q3 = clean.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = clean[(clean < lower) | (clean > upper)]
        quality.n_outliers = len(outliers)
        quality.outlier_percent = quality.n_outliers / len(clean) if len(clean) > 0 else 0
        quality.outlier_indices = list(outliers.index[:20])  # Limit to first 20

    def _assess_categorical(self, series: pd.Series, quality: ColumnQuality):
        """Assess categorical column."""
        clean = series.dropna()
        counts = clean.value_counts()

        quality.categories = list(counts.index[:20])  # Top 20
        quality.category_counts = counts.to_dict()

        if len(counts) >= 2:
            quality.imbalance_ratio = counts.iloc[0] / counts.iloc[-1]

    def _generate_column_recommendations(self, quality: ColumnQuality):
        """Generate warnings and recommendations for a column."""
        # Missing data warnings
        if quality.missing_percent >= self.MISSING_CRITICAL_THRESHOLD:
            quality.warnings.append(f"Critical: {quality.missing_percent:.1%} missing values")
            quality.recommendations.append("Consider multiple imputation or excluding variable")
        elif quality.missing_percent >= self.MISSING_WARNING_THRESHOLD:
            quality.warnings.append(f"Warning: {quality.missing_percent:.1%} missing values")
            quality.recommendations.append("Investigate missing data pattern")

        # Outlier warnings
        if quality.outlier_percent >= self.OUTLIER_THRESHOLD:
            quality.warnings.append(f"{quality.n_outliers} potential outliers detected ({quality.outlier_percent:.1%})")
            quality.recommendations.append("Review outliers for data errors or genuine extreme values")

        # Distribution warnings
        if quality.inferred_type == 'continuous':
            if quality.skewness is not None and abs(quality.skewness) > 2:
                quality.warnings.append(f"Highly skewed distribution (skewness={quality.skewness:.2f})")
                quality.recommendations.append("Consider log transformation or use non-parametric methods")

            if quality.is_normal is False:
                quality.warnings.append("Distribution is non-normal")
                quality.recommendations.append("Use non-parametric tests or transform data")

        # Imbalance warnings
        if quality.imbalance_ratio is not None and quality.imbalance_ratio > self.IMBALANCE_WARNING:
            quality.warnings.append(f"Highly imbalanced categories (ratio {quality.imbalance_ratio:.1f}:1)")
            quality.recommendations.append("Consider resampling or using appropriate methods for imbalanced data")

        # ID candidate warning
        if quality.is_id_candidate:
            quality.warnings.append("Appears to be an identifier variable (high cardinality)")
            quality.recommendations.append("Exclude from statistical analyses")

    def _calculate_completeness_score(self, columns: Dict[str, ColumnQuality]) -> float:
        """Calculate overall completeness score."""
        if not columns:
            return 0

        completeness_rates = [1 - c.missing_percent for c in columns.values()]
        return float(np.mean(completeness_rates) * 100)

    def _calculate_validity_score(self, columns: Dict[str, ColumnQuality]) -> float:
        """Calculate validity score based on outliers and type consistency."""
        if not columns:
            return 0

        validity_scores = []
        for col in columns.values():
            score = 100
            # Penalize for outliers
            score -= min(col.outlier_percent * 100, 20)
            # Penalize for type issues
            if col.inferred_type == 'unknown':
                score -= 10
            validity_scores.append(max(score, 0))

        return float(np.mean(validity_scores))

    def _calculate_consistency_score(
        self,
        data: pd.DataFrame,
        columns: Dict[str, ColumnQuality]
    ) -> float:
        """Calculate data consistency score."""
        # Check for potential inconsistencies
        issues = 0
        checks = 0

        for col_name, col in columns.items():
            checks += 1
            # Check for mixed types in object columns
            if col.dtype == 'object':
                try:
                    pd.to_numeric(data[col_name], errors='raise')
                    issues += 0.5  # Numeric stored as string
                except:
                    pass

        if checks == 0:
            return 100

        return float(max(0, 100 - (issues / checks * 100)))

    def _create_missing_matrix(self, data: pd.DataFrame) -> List[List[int]]:
        """Create missing data pattern matrix."""
        # Create binary missing indicator matrix
        missing = data.isna().astype(int)

        # Get unique patterns
        patterns = missing.drop_duplicates()

        return patterns.values.tolist()[:100]  # Limit patterns

    def _analyze_missing_correlations(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Analyze correlations between missing indicators."""
        missing = data.isna().astype(int)

        # Only analyze columns with some missing data
        has_missing = missing.columns[missing.sum() > 0]

        if len(has_missing) < 2:
            return {}

        correlations = {}
        for col1 in has_missing:
            correlations[col1] = {}
            for col2 in has_missing:
                if col1 != col2:
                    corr = missing[col1].corr(missing[col2])
                    if not np.isnan(corr):
                        correlations[col1][col2] = float(corr)

        return correlations

    def _generate_data_dictionary(
        self,
        data: pd.DataFrame,
        columns: Dict[str, ColumnQuality]
    ) -> List[Dict[str, Any]]:
        """Auto-generate data dictionary."""
        dictionary = []

        for col_name, col in columns.items():
            entry = {
                'variable_name': col_name,
                'data_type': col.dtype,
                'inferred_type': col.inferred_type,
                'n_missing': col.n_missing,
                'missing_percent': f"{col.missing_percent:.1%}",
                'n_unique': col.n_unique,
                'description': self._generate_description(col),
            }

            # Add statistics based on type
            if col.inferred_type == 'continuous':
                entry['statistics'] = {
                    'mean': f"{col.mean:.2f}" if col.mean else None,
                    'std': f"{col.std:.2f}" if col.std else None,
                    'min': col.min_val,
                    'max': col.max_val,
                }
            elif col.inferred_type in ['categorical', 'binary']:
                entry['categories'] = col.categories[:10] if col.categories else []
                entry['value_counts'] = {
                    k: v for k, v in list(col.category_counts.items())[:10]
                } if col.category_counts else {}

            dictionary.append(entry)

        return dictionary

    def _generate_description(self, col: ColumnQuality) -> str:
        """Auto-generate variable description."""
        if col.inferred_type == 'id':
            return "Unique identifier variable"
        elif col.inferred_type == 'binary':
            if col.categories:
                return f"Binary variable ({col.categories[0]}/{col.categories[1] if len(col.categories) > 1 else 'other'})"
            return "Binary variable"
        elif col.inferred_type == 'categorical':
            return f"Categorical variable with {col.n_unique} categories"
        elif col.inferred_type == 'continuous':
            if col.mean and col.std:
                return f"Continuous variable (mean={col.mean:.1f}, SD={col.std:.1f})"
            return "Continuous numeric variable"
        elif col.inferred_type == 'datetime':
            return "Date/time variable"
        else:
            return "Variable of unknown type"

    def _compile_recommendations(
        self,
        columns: Dict[str, ColumnQuality],
        data: pd.DataFrame
    ) -> Tuple[List[str], List[str], List[str]]:
        """Compile all recommendations."""
        critical = []
        warnings = []
        suggestions = []

        # Overall data issues
        total_missing = sum(c.n_missing for c in columns.values())
        total_cells = data.size
        overall_missing = total_missing / total_cells if total_cells > 0 else 0

        if overall_missing > 0.3:
            critical.append(f"Critical: {overall_missing:.1%} of data is missing")

        # Column-level issues
        for col in columns.values():
            for w in col.warnings:
                if 'Critical' in w:
                    critical.append(f"{col.name}: {w}")
                else:
                    warnings.append(f"{col.name}: {w}")

            for r in col.recommendations:
                suggestions.append(f"{col.name}: {r}")

        # General suggestions
        if len(columns) > 50:
            suggestions.append("Consider feature selection - many variables present")

        n_rows = len(data)
        if n_rows < 100:
            warnings.append("Small sample size - results may have limited power")
        elif n_rows > 100000:
            suggestions.append("Large dataset - consider sampling for exploratory analysis")

        return critical, warnings, suggestions

    def get_missing_summary(self) -> pd.DataFrame:
        """Get summary of missing data."""
        if self.report is None:
            return pd.DataFrame()

        rows = []
        for col_name, col in self.report.columns.items():
            if col.n_missing > 0:
                rows.append({
                    'Variable': col_name,
                    'N Missing': col.n_missing,
                    '% Missing': f"{col.missing_percent:.1%}",
                    'Pattern': col.missing_pattern
                })

        return pd.DataFrame(rows).sort_values('N Missing', ascending=False)

    def get_outlier_summary(self) -> pd.DataFrame:
        """Get summary of outliers."""
        if self.report is None:
            return pd.DataFrame()

        rows = []
        for col_name, col in self.report.columns.items():
            if col.n_outliers > 0:
                rows.append({
                    'Variable': col_name,
                    'N Outliers': col.n_outliers,
                    '% Outliers': f"{col.outlier_percent:.1%}",
                    'Min': col.min_val,
                    'Max': col.max_val
                })

        return pd.DataFrame(rows).sort_values('N Outliers', ascending=False)
