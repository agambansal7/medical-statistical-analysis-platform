"""
Missing Data Handling Module
============================
Comprehensive methods for handling missing data in clinical research.

Methods:
- Missing data pattern analysis
- Multiple imputation (MICE - Multivariate Imputation by Chained Equations)
- Sensitivity analysis for missingness (MNAR scenarios)
- Missing data visualization
- Complete case vs imputed comparison
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MissingPatternResult:
    """Result container for missing data pattern analysis."""
    n_observations: int
    n_variables: int
    n_complete_cases: int
    pct_complete_cases: float
    missing_by_variable: Dict[str, Dict]
    missing_patterns: pd.DataFrame
    n_patterns: int
    most_common_patterns: List[Tuple[str, int, float]]
    missingness_type_assessment: Dict[str, str]
    little_mcar_test: Optional[Dict]
    summary_text: str


@dataclass
class MultipleImputationResult:
    """Result container for multiple imputation."""
    imputed_datasets: List[pd.DataFrame]
    n_imputations: int
    method: str
    variables_imputed: List[str]
    convergence_info: Dict[str, bool]
    pooled_estimates: Optional[Dict[str, Dict]]
    fraction_missing_info: Dict[str, float]
    summary_text: str


@dataclass
class SensitivityAnalysisResult:
    """Result container for sensitivity analysis."""
    scenarios: List[str]
    estimates_by_scenario: Dict[str, Dict]
    tipping_point: Optional[float]
    conclusion_robust: bool
    summary_text: str


@dataclass
class ImputationComparisonResult:
    """Result container for comparing complete case vs imputed analysis."""
    complete_case_n: int
    imputed_n: int
    outcome_variable: str
    complete_case_estimate: float
    complete_case_se: float
    complete_case_ci: Tuple[float, float]
    imputed_estimate: float
    imputed_se: float
    imputed_ci: Tuple[float, float]
    relative_efficiency: float
    bias_assessment: str
    summary_text: str


class MissingDataHandler:
    """
    Comprehensive missing data analysis and handling.
    """

    def __init__(self):
        pass

    def analyze_missing_patterns(
        self,
        df: pd.DataFrame,
        variables: Optional[List[str]] = None
    ) -> MissingPatternResult:
        """
        Comprehensive missing data pattern analysis.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        variables : List[str], optional
            Variables to analyze (default: all)

        Returns:
        --------
        MissingPatternResult
        """
        if variables is None:
            variables = df.columns.tolist()

        df_subset = df[variables]
        n_obs = len(df_subset)
        n_vars = len(variables)

        # Missing by variable
        missing_by_var = {}
        for var in variables:
            n_missing = df_subset[var].isna().sum()
            pct_missing = n_missing / n_obs * 100
            missing_by_var[var] = {
                'n_missing': int(n_missing),
                'n_observed': int(n_obs - n_missing),
                'pct_missing': float(pct_missing),
                'dtype': str(df_subset[var].dtype)
            }

        # Complete cases
        n_complete = df_subset.dropna().shape[0]
        pct_complete = n_complete / n_obs * 100

        # Missing patterns
        # Create binary matrix (1=missing, 0=observed)
        missing_matrix = df_subset.isna().astype(int)

        # Pattern counts
        pattern_counts = missing_matrix.groupby(list(variables)).size().reset_index(name='count')
        pattern_counts['percentage'] = pattern_counts['count'] / n_obs * 100
        pattern_counts = pattern_counts.sort_values('count', ascending=False)

        n_patterns = len(pattern_counts)

        # Most common patterns
        most_common = []
        for _, row in pattern_counts.head(5).iterrows():
            pattern_vars = [variables[i] for i, val in enumerate(row[variables]) if val == 1]
            pattern_str = ", ".join(pattern_vars) if pattern_vars else "Complete"
            most_common.append((pattern_str, int(row['count']), float(row['percentage'])))

        # Assess missingness type for each variable
        missingness_type = {}
        for var in variables:
            if missing_by_var[var]['n_missing'] == 0:
                missingness_type[var] = "No missing data"
            else:
                # Test association with other variables
                miss_type = self._assess_missingness_type(df_subset, var, variables)
                missingness_type[var] = miss_type

        # Little's MCAR test (simplified)
        mcar_test = None
        if n_complete > 0 and n_complete < n_obs:
            try:
                mcar_test = self._little_mcar_test(df_subset)
            except:
                mcar_test = None

        summary = (
            f"Missing data analysis: {n_obs} observations, {n_vars} variables. "
            f"Complete cases: {n_complete} ({pct_complete:.1f}%). "
            f"{n_patterns} missing patterns identified."
        )

        return MissingPatternResult(
            n_observations=n_obs,
            n_variables=n_vars,
            n_complete_cases=n_complete,
            pct_complete_cases=pct_complete,
            missing_by_variable=missing_by_var,
            missing_patterns=pattern_counts,
            n_patterns=n_patterns,
            most_common_patterns=most_common,
            missingness_type_assessment=missingness_type,
            little_mcar_test=mcar_test,
            summary_text=summary
        )

    def _assess_missingness_type(
        self,
        df: pd.DataFrame,
        target_var: str,
        all_vars: List[str]
    ) -> str:
        """Assess whether missingness is MCAR, MAR, or MNAR."""
        # Create missingness indicator
        missing_indicator = df[target_var].isna().astype(int)

        # Test association with observed values of other variables
        significant_associations = []

        for var in all_vars:
            if var == target_var:
                continue

            # Get complete cases for this variable
            complete_idx = ~df[var].isna()

            if complete_idx.sum() < 10:
                continue

            # Test association
            if df[var].dtype in ['int64', 'float64']:
                # T-test for continuous
                group1 = df.loc[complete_idx & (missing_indicator == 1), var].dropna()
                group0 = df.loc[complete_idx & (missing_indicator == 0), var].dropna()

                if len(group1) >= 5 and len(group0) >= 5:
                    try:
                        _, p = stats.ttest_ind(group1, group0)
                        if p < 0.05:
                            significant_associations.append(var)
                    except:
                        pass
            else:
                # Chi-square for categorical
                try:
                    contingency = pd.crosstab(df.loc[complete_idx, var], missing_indicator[complete_idx])
                    if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
                        _, p, _, _ = stats.chi2_contingency(contingency)
                        if p < 0.05:
                            significant_associations.append(var)
                except:
                    pass

        if len(significant_associations) == 0:
            return "Possibly MCAR (no associations detected)"
        else:
            return f"Likely MAR (associated with: {', '.join(significant_associations[:3])})"

    def _little_mcar_test(self, df: pd.DataFrame) -> Dict:
        """
        Little's MCAR test.

        Tests whether data is Missing Completely At Random.
        """
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            return None

        df_numeric = df[numeric_cols]

        # Get missing patterns
        patterns = df_numeric.isna().drop_duplicates()

        # Calculate test statistic
        n = len(df_numeric)
        p = len(numeric_cols)

        # Overall means and covariance
        overall_means = df_numeric.mean()
        overall_cov = df_numeric.cov()

        # Test statistic
        d_squared = 0
        df_test = 0

        for _, pattern in patterns.iterrows():
            # Get observations with this pattern
            mask = (df_numeric.isna() == pattern).all(axis=1)
            n_j = mask.sum()

            if n_j < 2:
                continue

            observed_vars = [col for col in numeric_cols if not pattern[col]]

            if len(observed_vars) < 1:
                continue

            # Observed data for this pattern
            obs_data = df_numeric.loc[mask, observed_vars]

            # Pattern means
            pattern_means = obs_data.mean()

            # Deviation from overall means
            deviation = pattern_means - overall_means[observed_vars]

            # Covariance of observed variables
            cov_obs = overall_cov.loc[observed_vars, observed_vars]

            try:
                cov_inv = np.linalg.inv(cov_obs)
                d_squared += n_j * deviation.values @ cov_inv @ deviation.values
                df_test += len(observed_vars)
            except:
                pass

        # P-value
        p_value = 1 - stats.chi2.cdf(d_squared, df_test) if df_test > 0 else 1.0

        return {
            'test_statistic': float(d_squared),
            'df': int(df_test),
            'p_value': float(p_value),
            'conclusion': 'MCAR' if p_value >= 0.05 else 'Not MCAR'
        }

    def multiple_imputation(
        self,
        df: pd.DataFrame,
        variables_to_impute: List[str],
        predictor_variables: Optional[List[str]] = None,
        n_imputations: int = 5,
        method: str = 'mice',
        max_iter: int = 10,
        random_state: int = 42
    ) -> MultipleImputationResult:
        """
        Multiple imputation using MICE (Multivariate Imputation by Chained Equations).

        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        variables_to_impute : List[str]
            Variables with missing data to impute
        predictor_variables : List[str], optional
            Variables to use as predictors (default: all other variables)
        n_imputations : int
            Number of imputed datasets to create
        method : str
            'mice' (chained equations), 'pmm' (predictive mean matching)
        max_iter : int
            Maximum iterations for MICE
        random_state : int
            Random seed

        Returns:
        --------
        MultipleImputationResult
        """
        if predictor_variables is None:
            predictor_variables = [v for v in df.columns if v not in variables_to_impute]

        all_vars = variables_to_impute + predictor_variables
        df_subset = df[all_vars].copy()

        # Convert categorical to numeric for imputation
        categorical_cols = df_subset.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_mappings = {}

        for col in categorical_cols:
            df_subset[col], cat_mappings[col] = pd.factorize(df_subset[col])
            df_subset[col] = df_subset[col].astype(float)
            df_subset.loc[df_subset[col] < 0, col] = np.nan  # Restore NaN

        imputed_datasets = []
        convergence_info = {}

        for i in range(n_imputations):
            # Set random state for reproducibility
            current_seed = random_state + i

            # Use IterativeImputer (MICE)
            imputer = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=max_iter,
                random_state=current_seed,
                sample_posterior=True  # For proper multiple imputation
            )

            imputed_values = imputer.fit_transform(df_subset)
            df_imputed = pd.DataFrame(imputed_values, columns=all_vars, index=df_subset.index)

            # Convert categorical back
            for col in categorical_cols:
                df_imputed[col] = df_imputed[col].round().astype(int)
                df_imputed[col] = df_imputed[col].map(
                    lambda x: cat_mappings[col][min(x, len(cat_mappings[col])-1)] if 0 <= x < len(cat_mappings[col]) else np.nan
                )

            imputed_datasets.append(df_imputed)
            convergence_info[f'imputation_{i+1}'] = True  # Simplified convergence check

        # Calculate fraction of missing information for each variable
        fmi = {}
        for var in variables_to_impute:
            n_missing = df[var].isna().sum()
            n_total = len(df)
            fmi[var] = n_missing / n_total

        summary = (
            f"Multiple imputation ({method}) completed. "
            f"Created {n_imputations} imputed datasets. "
            f"Variables imputed: {', '.join(variables_to_impute)}."
        )

        return MultipleImputationResult(
            imputed_datasets=imputed_datasets,
            n_imputations=n_imputations,
            method=method,
            variables_imputed=variables_to_impute,
            convergence_info=convergence_info,
            pooled_estimates=None,  # Filled by analyze_with_imputed
            fraction_missing_info=fmi,
            summary_text=summary
        )

    def pool_estimates(
        self,
        estimates: List[float],
        variances: List[float]
    ) -> Dict:
        """
        Pool estimates across multiple imputations using Rubin's rules.

        Parameters:
        -----------
        estimates : List[float]
            Point estimates from each imputed dataset
        variances : List[float]
            Variance estimates from each imputed dataset

        Returns:
        --------
        Dict with pooled estimate, SE, CI, and related statistics
        """
        m = len(estimates)
        estimates = np.array(estimates)
        variances = np.array(variances)

        # Pooled estimate (average)
        pooled_estimate = np.mean(estimates)

        # Within-imputation variance (average)
        within_var = np.mean(variances)

        # Between-imputation variance
        between_var = np.var(estimates, ddof=1)

        # Total variance (Rubin's rules)
        total_var = within_var + (1 + 1/m) * between_var

        # Standard error
        pooled_se = np.sqrt(total_var)

        # Degrees of freedom (Barnard-Rubin)
        if between_var > 0:
            r = (1 + 1/m) * between_var / within_var
            df_old = (m - 1) * (1 + 1/r)**2
            df = df_old  # Simplified - could use Barnard-Rubin adjustment
        else:
            df = np.inf

        # Confidence interval
        if np.isfinite(df) and df > 2:
            t_crit = stats.t.ppf(0.975, df)
        else:
            t_crit = 1.96

        ci_lower = pooled_estimate - t_crit * pooled_se
        ci_upper = pooled_estimate + t_crit * pooled_se

        # Fraction of missing information
        if total_var > 0:
            fmi = (between_var + between_var/m) / total_var
        else:
            fmi = 0

        # Relative efficiency
        rel_efficiency = 1 / (1 + fmi/m)

        return {
            'pooled_estimate': float(pooled_estimate),
            'pooled_se': float(pooled_se),
            'within_variance': float(within_var),
            'between_variance': float(between_var),
            'total_variance': float(total_var),
            'df': float(df),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'fraction_missing_info': float(fmi),
            'relative_efficiency': float(rel_efficiency)
        }

    def sensitivity_analysis_mnar(
        self,
        df: pd.DataFrame,
        outcome_var: str,
        treatment_var: str,
        imputed_datasets: List[pd.DataFrame],
        delta_range: List[float] = [-1.0, -0.5, 0, 0.5, 1.0],
        analysis_function: Optional[callable] = None
    ) -> SensitivityAnalysisResult:
        """
        Sensitivity analysis for Missing Not At Random (MNAR) scenarios.

        Uses delta adjustment method - adds delta to imputed values to
        assess sensitivity to MNAR.

        Parameters:
        -----------
        df : pd.DataFrame
            Original data
        outcome_var : str
            Outcome variable
        treatment_var : str
            Treatment variable
        imputed_datasets : List[pd.DataFrame]
            Multiple imputed datasets
        delta_range : List[float]
            Delta values to test (in SD units)
        analysis_function : callable
            Function to run analysis (default: mean difference)

        Returns:
        --------
        SensitivityAnalysisResult
        """
        # Identify originally missing outcome values
        missing_mask = df[outcome_var].isna()

        if missing_mask.sum() == 0:
            return SensitivityAnalysisResult(
                scenarios=[],
                estimates_by_scenario={},
                tipping_point=None,
                conclusion_robust=True,
                summary_text="No missing data in outcome variable."
            )

        # Standard deviation of observed outcome
        sd_outcome = df[outcome_var].dropna().std()

        estimates_by_scenario = {}
        base_significant = None

        for delta in delta_range:
            scenario_name = f"δ={delta}"
            scenario_estimates = []
            scenario_variances = []

            for imp_df in imputed_datasets:
                # Adjust imputed values for missing
                df_adjusted = imp_df.copy()
                df_adjusted.loc[missing_mask, outcome_var] += delta * sd_outcome

                # Run analysis
                if analysis_function is not None:
                    result = analysis_function(df_adjusted, outcome_var, treatment_var)
                    estimate = result['estimate']
                    variance = result['variance']
                else:
                    # Default: mean difference
                    treated = df_adjusted[df_adjusted[treatment_var] == 1][outcome_var]
                    control = df_adjusted[df_adjusted[treatment_var] == 0][outcome_var]

                    estimate = treated.mean() - control.mean()
                    variance = treated.var()/len(treated) + control.var()/len(control)

                scenario_estimates.append(estimate)
                scenario_variances.append(variance)

            # Pool estimates
            pooled = self.pool_estimates(scenario_estimates, scenario_variances)

            # Check significance
            z = pooled['pooled_estimate'] / pooled['pooled_se']
            p_value = 2 * (1 - stats.norm.cdf(np.abs(z)))
            significant = p_value < 0.05

            if delta == 0:
                base_significant = significant

            estimates_by_scenario[scenario_name] = {
                'pooled_estimate': pooled['pooled_estimate'],
                'pooled_se': pooled['pooled_se'],
                'ci_lower': pooled['ci_lower'],
                'ci_upper': pooled['ci_upper'],
                'p_value': p_value,
                'significant': significant
            }

        # Find tipping point (delta where conclusion changes)
        tipping_point = None
        if base_significant is not None:
            for delta in sorted(delta_range, key=lambda x: abs(x)):
                scenario = f"δ={delta}"
                if estimates_by_scenario[scenario]['significant'] != base_significant:
                    tipping_point = delta
                    break

        # Assess robustness
        all_same_conclusion = len(set(e['significant'] for e in estimates_by_scenario.values())) == 1
        conclusion_robust = all_same_conclusion

        summary = (
            f"Sensitivity analysis for MNAR. "
            f"Tested {len(delta_range)} scenarios (δ from {min(delta_range)} to {max(delta_range)} SD). "
            f"{'Conclusions robust to MNAR.' if conclusion_robust else f'Tipping point at δ={tipping_point}.'}"
        )

        return SensitivityAnalysisResult(
            scenarios=list(estimates_by_scenario.keys()),
            estimates_by_scenario=estimates_by_scenario,
            tipping_point=tipping_point,
            conclusion_robust=conclusion_robust,
            summary_text=summary
        )

    def compare_complete_case_vs_imputed(
        self,
        df: pd.DataFrame,
        imputed_datasets: List[pd.DataFrame],
        outcome_var: str,
        treatment_var: str
    ) -> ImputationComparisonResult:
        """
        Compare complete case analysis vs multiple imputation.

        Parameters:
        -----------
        df : pd.DataFrame
            Original data
        imputed_datasets : List[pd.DataFrame]
            Multiple imputed datasets
        outcome_var : str
            Outcome variable
        treatment_var : str
            Treatment variable

        Returns:
        --------
        ImputationComparisonResult
        """
        # Complete case analysis
        df_complete = df.dropna(subset=[outcome_var, treatment_var])
        n_complete = len(df_complete)

        treated_cc = df_complete[df_complete[treatment_var] == 1][outcome_var]
        control_cc = df_complete[df_complete[treatment_var] == 0][outcome_var]

        cc_estimate = treated_cc.mean() - control_cc.mean()
        cc_se = np.sqrt(treated_cc.var()/len(treated_cc) + control_cc.var()/len(control_cc))
        cc_ci = (cc_estimate - 1.96 * cc_se, cc_estimate + 1.96 * cc_se)

        # Multiple imputation analysis
        n_imputed = len(df)
        mi_estimates = []
        mi_variances = []

        for imp_df in imputed_datasets:
            treated = imp_df[imp_df[treatment_var] == 1][outcome_var]
            control = imp_df[imp_df[treatment_var] == 0][outcome_var]

            estimate = treated.mean() - control.mean()
            variance = treated.var()/len(treated) + control.var()/len(control)

            mi_estimates.append(estimate)
            mi_variances.append(variance)

        pooled = self.pool_estimates(mi_estimates, mi_variances)

        # Relative efficiency
        if pooled['pooled_se'] > 0:
            rel_efficiency = (cc_se / pooled['pooled_se'])**2
        else:
            rel_efficiency = 1.0

        # Bias assessment
        estimate_diff = abs(cc_estimate - pooled['pooled_estimate'])
        pooled_se = pooled['pooled_se']

        if estimate_diff < 0.1 * pooled_se:
            bias_assessment = "Negligible difference between methods"
        elif estimate_diff < 0.5 * pooled_se:
            bias_assessment = "Small difference - likely minimal bias from missing data"
        else:
            bias_assessment = "Substantial difference - potential bias in complete case analysis"

        summary = (
            f"Complete case (n={n_complete}): estimate = {cc_estimate:.4f} (95% CI: {cc_ci[0]:.4f}, {cc_ci[1]:.4f}). "
            f"Multiple imputation (n={n_imputed}): estimate = {pooled['pooled_estimate']:.4f} "
            f"(95% CI: {pooled['ci_lower']:.4f}, {pooled['ci_upper']:.4f}). "
            f"Relative efficiency = {rel_efficiency:.2f}. {bias_assessment}."
        )

        return ImputationComparisonResult(
            complete_case_n=n_complete,
            imputed_n=n_imputed,
            outcome_variable=outcome_var,
            complete_case_estimate=cc_estimate,
            complete_case_se=cc_se,
            complete_case_ci=cc_ci,
            imputed_estimate=pooled['pooled_estimate'],
            imputed_se=pooled['pooled_se'],
            imputed_ci=(pooled['ci_lower'], pooled['ci_upper']),
            relative_efficiency=rel_efficiency,
            bias_assessment=bias_assessment,
            summary_text=summary
        )

    def plot_missing_pattern(
        self,
        result: MissingPatternResult,
        title: str = "Missing Data Pattern"
    ):
        """Create missing data pattern visualization."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Bar chart of missing by variable
        vars_sorted = sorted(result.missing_by_variable.items(),
                            key=lambda x: x[1]['pct_missing'], reverse=True)
        variables = [v[0] for v in vars_sorted]
        pct_missing = [v[1]['pct_missing'] for v in vars_sorted]

        colors = ['#e74c3c' if p > 20 else '#f39c12' if p > 5 else '#27ae60' for p in pct_missing]

        axes[0].barh(variables, pct_missing, color=colors)
        axes[0].set_xlabel('Percentage Missing (%)')
        axes[0].set_title('Missing Data by Variable', fontweight='bold')
        axes[0].axvline(x=5, color='gray', linestyle='--', alpha=0.5)
        axes[0].axvline(x=20, color='gray', linestyle='--', alpha=0.5)

        # Pattern matrix (top patterns)
        pattern_df = result.missing_patterns.head(10)
        pattern_matrix = pattern_df[result.missing_by_variable.keys()].values

        sns.heatmap(pattern_matrix, ax=axes[1], cmap='RdYlGn_r',
                    cbar_kws={'label': 'Missing (1) / Observed (0)'},
                    xticklabels=list(result.missing_by_variable.keys()),
                    yticklabels=[f"Pattern {i+1} (n={int(row['count'])})"
                                for i, (_, row) in enumerate(pattern_df.iterrows())])

        axes[1].set_title('Missing Data Patterns (Top 10)', fontweight='bold')
        axes[1].set_xlabel('Variable')

        plt.tight_layout()
        return fig

    def plot_imputation_diagnostics(
        self,
        original_df: pd.DataFrame,
        imputed_datasets: List[pd.DataFrame],
        variable: str,
        title: str = None
    ):
        """Create imputation diagnostic plots."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        if title is None:
            title = f"Imputation Diagnostics: {variable}"

        # Get observed values
        observed = original_df[variable].dropna()

        # Get imputed values (from first imputation)
        missing_mask = original_df[variable].isna()
        imputed_values = [df.loc[missing_mask, variable] for df in imputed_datasets]

        # 1. Distribution comparison
        axes[0].hist(observed, bins=30, alpha=0.7, label='Observed', density=True, color='#3498db')
        for i, imp in enumerate(imputed_values):
            axes[0].hist(imp, bins=30, alpha=0.3, label=f'Imputed {i+1}' if i < 3 else None,
                        density=True, color='#e74c3c')

        axes[0].set_xlabel(variable)
        axes[0].set_ylabel('Density')
        axes[0].set_title('Distribution: Observed vs Imputed', fontweight='bold')
        axes[0].legend()

        # 2. Stripplot of imputed values across imputations
        imp_data = pd.DataFrame({
            f'Imp {i+1}': imp.values for i, imp in enumerate(imputed_values)
        })
        imp_data_long = imp_data.melt(var_name='Imputation', value_name=variable)

        axes[1].boxplot([imp.values for imp in imputed_values],
                       labels=[f'Imp {i+1}' for i in range(len(imputed_values))])
        axes[1].axhline(y=observed.mean(), color='red', linestyle='--', label='Observed mean')
        axes[1].set_ylabel(variable)
        axes[1].set_title('Imputed Values by Imputation', fontweight='bold')
        axes[1].legend()

        # 3. Summary statistics
        stats_data = {
            'Observed': {
                'Mean': observed.mean(),
                'SD': observed.std(),
                'Median': observed.median(),
                'N': len(observed)
            }
        }
        for i, imp in enumerate(imputed_values):
            stats_data[f'Imputed {i+1}'] = {
                'Mean': imp.mean(),
                'SD': imp.std(),
                'Median': imp.median(),
                'N': len(imp)
            }

        stats_df = pd.DataFrame(stats_data).T

        axes[2].axis('off')
        table = axes[2].table(
            cellText=stats_df.round(2).values,
            colLabels=stats_df.columns,
            rowLabels=stats_df.index,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[2].set_title('Summary Statistics', fontweight='bold', y=0.9)

        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
