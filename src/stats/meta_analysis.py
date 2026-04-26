"""
Meta-Analysis Module
====================
Comprehensive meta-analysis methods for combining results across studies.

Methods:
- Fixed effects model (Mantel-Haenszel, inverse variance)
- Random effects model (DerSimonian-Laird, REML)
- Heterogeneity assessment (I², Q-test, tau²)
- Publication bias (Funnel plot, Egger's test, trim-and-fill)
- Subgroup and meta-regression analysis
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')


@dataclass
class StudyData:
    """Container for individual study data."""
    study_id: str
    effect_size: float  # Log OR, Log RR, SMD, etc.
    se: float           # Standard error
    n_treatment: Optional[int] = None
    n_control: Optional[int] = None
    events_treatment: Optional[int] = None
    events_control: Optional[int] = None
    weight: Optional[float] = None


@dataclass
class MetaAnalysisResult:
    """Result container for meta-analysis."""
    model_type: str  # 'fixed' or 'random'
    method: str      # 'inverse_variance', 'mantel_haenszel', 'dersimonian_laird', 'reml'
    pooled_effect: float
    pooled_se: float
    pooled_ci_lower: float
    pooled_ci_upper: float
    pooled_z: float
    pooled_p_value: float
    study_weights: Dict[str, float]
    study_effects: Dict[str, float]
    n_studies: int
    total_n: int
    heterogeneity: 'HeterogeneityResult'
    prediction_interval: Optional[Tuple[float, float]]
    summary_text: str


@dataclass
class HeterogeneityResult:
    """Result container for heterogeneity assessment."""
    q_statistic: float
    q_df: int
    q_p_value: float
    i_squared: float
    i_squared_ci_lower: float
    i_squared_ci_upper: float
    tau_squared: float
    tau: float
    h_squared: float
    interpretation: str


@dataclass
class PublicationBiasResult:
    """Result container for publication bias assessment."""
    egger_intercept: float
    egger_se: float
    egger_t: float
    egger_p_value: float
    begg_z: float
    begg_p_value: float
    trim_fill_added: int
    trim_fill_effect: float
    trim_fill_ci_lower: float
    trim_fill_ci_upper: float
    fail_safe_n: int
    summary_text: str


@dataclass
class SubgroupResult:
    """Result container for subgroup analysis."""
    subgroup_name: str
    subgroup_effects: Dict[str, MetaAnalysisResult]
    between_group_q: float
    between_group_df: int
    between_group_p: float
    summary_text: str


@dataclass
class MetaRegressionResult:
    """Result container for meta-regression."""
    coefficients: Dict[str, Dict]
    r_squared: float
    residual_heterogeneity: float
    model_q: float
    model_df: int
    model_p: float
    n_studies: int
    summary_text: str


class MetaAnalysis:
    """
    Comprehensive meta-analysis implementation.
    """

    def __init__(self):
        pass

    def prepare_binary_data(
        self,
        studies: List[Dict],
        measure: str = 'OR'
    ) -> List[StudyData]:
        """
        Prepare binary outcome data for meta-analysis.

        Parameters:
        -----------
        studies : List[Dict]
            List of dictionaries with keys:
            - 'study_id': Study identifier
            - 'events_t': Events in treatment group
            - 'n_t': Total in treatment group
            - 'events_c': Events in control group
            - 'n_c': Total in control group
        measure : str
            'OR' (odds ratio), 'RR' (risk ratio), 'RD' (risk difference)

        Returns:
        --------
        List[StudyData]
        """
        study_data = []

        for s in studies:
            a = s['events_t']      # Treatment events
            b = s['n_t'] - a       # Treatment non-events
            c = s['events_c']      # Control events
            d = s['n_c'] - c       # Control non-events

            # Add continuity correction if needed
            if a == 0 or b == 0 or c == 0 or d == 0:
                a += 0.5
                b += 0.5
                c += 0.5
                d += 0.5

            if measure == 'OR':
                effect = np.log((a * d) / (b * c))
                se = np.sqrt(1/a + 1/b + 1/c + 1/d)
            elif measure == 'RR':
                effect = np.log((a / (a + b)) / (c / (c + d)))
                se = np.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
            elif measure == 'RD':
                effect = (a / (a + b)) - (c / (c + d))
                se = np.sqrt((a*b)/((a+b)**3) + (c*d)/((c+d)**3))
            else:
                raise ValueError(f"Unknown measure: {measure}")

            study_data.append(StudyData(
                study_id=s['study_id'],
                effect_size=effect,
                se=se,
                n_treatment=s['n_t'],
                n_control=s['n_c'],
                events_treatment=s['events_t'],
                events_control=s['events_c']
            ))

        return study_data

    def prepare_continuous_data(
        self,
        studies: List[Dict],
        measure: str = 'SMD'
    ) -> List[StudyData]:
        """
        Prepare continuous outcome data for meta-analysis.

        Parameters:
        -----------
        studies : List[Dict]
            List of dictionaries with keys:
            - 'study_id': Study identifier
            - 'mean_t', 'sd_t', 'n_t': Treatment group stats
            - 'mean_c', 'sd_c', 'n_c': Control group stats
        measure : str
            'SMD' (standardized mean difference/Hedges' g), 'MD' (mean difference)

        Returns:
        --------
        List[StudyData]
        """
        study_data = []

        for s in studies:
            mean_t = s['mean_t']
            sd_t = s['sd_t']
            n_t = s['n_t']
            mean_c = s['mean_c']
            sd_c = s['sd_c']
            n_c = s['n_c']

            if measure == 'SMD':
                # Hedges' g (bias-corrected SMD)
                pooled_sd = np.sqrt(((n_t - 1) * sd_t**2 + (n_c - 1) * sd_c**2) / (n_t + n_c - 2))

                # Cohen's d
                d = (mean_t - mean_c) / pooled_sd

                # Hedges' correction factor
                j = 1 - 3 / (4 * (n_t + n_c - 2) - 1)
                effect = d * j

                # SE of Hedges' g
                se = np.sqrt((n_t + n_c) / (n_t * n_c) + effect**2 / (2 * (n_t + n_c)))

            elif measure == 'MD':
                effect = mean_t - mean_c
                se = np.sqrt(sd_t**2 / n_t + sd_c**2 / n_c)
            else:
                raise ValueError(f"Unknown measure: {measure}")

            study_data.append(StudyData(
                study_id=s['study_id'],
                effect_size=effect,
                se=se,
                n_treatment=n_t,
                n_control=n_c
            ))

        return study_data

    def fixed_effects(
        self,
        studies: List[StudyData],
        method: str = 'inverse_variance'
    ) -> MetaAnalysisResult:
        """
        Fixed effects meta-analysis.

        Parameters:
        -----------
        studies : List[StudyData]
            Prepared study data
        method : str
            'inverse_variance' or 'mantel_haenszel'

        Returns:
        --------
        MetaAnalysisResult
        """
        effects = np.array([s.effect_size for s in studies])
        variances = np.array([s.se**2 for s in studies])
        weights = 1 / variances

        # Pooled effect
        pooled_effect = np.sum(weights * effects) / np.sum(weights)
        pooled_var = 1 / np.sum(weights)
        pooled_se = np.sqrt(pooled_var)

        # 95% CI
        z = stats.norm.ppf(0.975)
        ci_lower = pooled_effect - z * pooled_se
        ci_upper = pooled_effect + z * pooled_se

        # Z-test
        z_stat = pooled_effect / pooled_se
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

        # Heterogeneity
        heterogeneity = self._calculate_heterogeneity(effects, variances, weights)

        # Study weights (percentage)
        total_weight = np.sum(weights)
        study_weights = {s.study_id: float(w / total_weight * 100) for s, w in zip(studies, weights)}
        study_effects = {s.study_id: float(s.effect_size) for s in studies}

        # Total N
        total_n = sum(s.n_treatment + s.n_control for s in studies if s.n_treatment and s.n_control)

        summary = (
            f"Fixed effects meta-analysis ({method}). "
            f"k={len(studies)} studies, N={total_n}. "
            f"Pooled effect = {pooled_effect:.4f} (95% CI: {ci_lower:.4f}, {ci_upper:.4f}), p = {p_value:.4f}. "
            f"I² = {heterogeneity.i_squared:.1f}%."
        )

        return MetaAnalysisResult(
            model_type='fixed',
            method=method,
            pooled_effect=pooled_effect,
            pooled_se=pooled_se,
            pooled_ci_lower=ci_lower,
            pooled_ci_upper=ci_upper,
            pooled_z=z_stat,
            pooled_p_value=p_value,
            study_weights=study_weights,
            study_effects=study_effects,
            n_studies=len(studies),
            total_n=total_n,
            heterogeneity=heterogeneity,
            prediction_interval=None,
            summary_text=summary
        )

    def random_effects(
        self,
        studies: List[StudyData],
        method: str = 'dersimonian_laird'
    ) -> MetaAnalysisResult:
        """
        Random effects meta-analysis.

        Parameters:
        -----------
        studies : List[StudyData]
            Prepared study data
        method : str
            'dersimonian_laird' or 'reml' (restricted maximum likelihood)

        Returns:
        --------
        MetaAnalysisResult
        """
        effects = np.array([s.effect_size for s in studies])
        variances = np.array([s.se**2 for s in studies])
        k = len(studies)

        if method == 'dersimonian_laird':
            tau_squared = self._dersimonian_laird_tau(effects, variances)
        elif method == 'reml':
            tau_squared = self._reml_tau(effects, variances)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Random effects weights
        weights = 1 / (variances + tau_squared)

        # Pooled effect
        pooled_effect = np.sum(weights * effects) / np.sum(weights)
        pooled_var = 1 / np.sum(weights)
        pooled_se = np.sqrt(pooled_var)

        # 95% CI
        z = stats.norm.ppf(0.975)
        ci_lower = pooled_effect - z * pooled_se
        ci_upper = pooled_effect + z * pooled_se

        # Z-test
        z_stat = pooled_effect / pooled_se
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

        # Heterogeneity
        heterogeneity = self._calculate_heterogeneity(effects, variances, 1/variances)
        heterogeneity.tau_squared = tau_squared
        heterogeneity.tau = np.sqrt(tau_squared)

        # Prediction interval
        if k >= 3:
            t_crit = stats.t.ppf(0.975, df=k-2)
            pred_se = np.sqrt(pooled_var + tau_squared)
            pred_lower = pooled_effect - t_crit * pred_se
            pred_upper = pooled_effect + t_crit * pred_se
            prediction_interval = (pred_lower, pred_upper)
        else:
            prediction_interval = None

        # Study weights
        total_weight = np.sum(weights)
        study_weights = {s.study_id: float(w / total_weight * 100) for s, w in zip(studies, weights)}
        study_effects = {s.study_id: float(s.effect_size) for s in studies}

        # Total N
        total_n = sum(s.n_treatment + s.n_control for s in studies if s.n_treatment and s.n_control)

        summary = (
            f"Random effects meta-analysis ({method}). "
            f"k={len(studies)} studies, N={total_n}. "
            f"Pooled effect = {pooled_effect:.4f} (95% CI: {ci_lower:.4f}, {ci_upper:.4f}), p = {p_value:.4f}. "
            f"τ² = {tau_squared:.4f}, I² = {heterogeneity.i_squared:.1f}%."
        )

        return MetaAnalysisResult(
            model_type='random',
            method=method,
            pooled_effect=pooled_effect,
            pooled_se=pooled_se,
            pooled_ci_lower=ci_lower,
            pooled_ci_upper=ci_upper,
            pooled_z=z_stat,
            pooled_p_value=p_value,
            study_weights=study_weights,
            study_effects=study_effects,
            n_studies=len(studies),
            total_n=total_n,
            heterogeneity=heterogeneity,
            prediction_interval=prediction_interval,
            summary_text=summary
        )

    def _dersimonian_laird_tau(
        self,
        effects: np.ndarray,
        variances: np.ndarray
    ) -> float:
        """Calculate tau-squared using DerSimonian-Laird method."""
        weights = 1 / variances
        k = len(effects)

        # Q statistic
        weighted_mean = np.sum(weights * effects) / np.sum(weights)
        Q = np.sum(weights * (effects - weighted_mean)**2)

        # C constant
        C = np.sum(weights) - np.sum(weights**2) / np.sum(weights)

        # Tau-squared
        tau_squared = max(0, (Q - (k - 1)) / C)

        return tau_squared

    def _reml_tau(
        self,
        effects: np.ndarray,
        variances: np.ndarray
    ) -> float:
        """Calculate tau-squared using REML."""
        def neg_reml(tau_sq):
            if tau_sq < 0:
                return np.inf

            weights = 1 / (variances + tau_sq)
            weighted_mean = np.sum(weights * effects) / np.sum(weights)

            # Log-likelihood
            ll = -0.5 * np.sum(np.log(variances + tau_sq))
            ll -= 0.5 * np.sum(weights * (effects - weighted_mean)**2)
            ll -= 0.5 * np.log(np.sum(weights))

            return -ll

        # Optimize
        result = minimize_scalar(neg_reml, bounds=(0, 10), method='bounded')
        return max(0, result.x)

    def _calculate_heterogeneity(
        self,
        effects: np.ndarray,
        variances: np.ndarray,
        weights: np.ndarray
    ) -> HeterogeneityResult:
        """Calculate heterogeneity statistics."""
        k = len(effects)

        # Q statistic
        weighted_mean = np.sum(weights * effects) / np.sum(weights)
        Q = np.sum(weights * (effects - weighted_mean)**2)
        df = k - 1

        # Q p-value
        q_p_value = 1 - stats.chi2.cdf(Q, df)

        # I-squared
        if Q > df:
            i_squared = (Q - df) / Q * 100
        else:
            i_squared = 0

        # I-squared CI (using Q distribution)
        # Higgins & Thompson method
        if Q > df + 1:
            i_sq_lower = ((Q - df) / Q - 1.96 * np.sqrt(2 * df) / Q) * 100
            i_sq_upper = ((Q - df) / Q + 1.96 * np.sqrt(2 * df) / Q) * 100
        else:
            i_sq_lower = 0
            i_sq_upper = min(100, (Q - df + 1.96 * np.sqrt(2 * df)) / Q * 100) if Q > 0 else 0

        i_sq_lower = max(0, i_sq_lower)
        i_sq_upper = min(100, i_sq_upper)

        # H-squared
        h_squared = Q / df if df > 0 else 1

        # Tau-squared (DL estimate)
        C = np.sum(weights) - np.sum(weights**2) / np.sum(weights)
        tau_squared = max(0, (Q - df) / C) if C > 0 else 0
        tau = np.sqrt(tau_squared)

        # Interpretation
        if i_squared < 25:
            interpretation = "Low heterogeneity (I² < 25%)"
        elif i_squared < 50:
            interpretation = "Moderate heterogeneity (25% ≤ I² < 50%)"
        elif i_squared < 75:
            interpretation = "Substantial heterogeneity (50% ≤ I² < 75%)"
        else:
            interpretation = "Considerable heterogeneity (I² ≥ 75%)"

        return HeterogeneityResult(
            q_statistic=Q,
            q_df=df,
            q_p_value=q_p_value,
            i_squared=i_squared,
            i_squared_ci_lower=i_sq_lower,
            i_squared_ci_upper=i_sq_upper,
            tau_squared=tau_squared,
            tau=tau,
            h_squared=h_squared,
            interpretation=interpretation
        )

    def publication_bias(
        self,
        studies: List[StudyData],
        pooled_effect: float
    ) -> PublicationBiasResult:
        """
        Assess publication bias using multiple methods.

        Parameters:
        -----------
        studies : List[StudyData]
            Prepared study data
        pooled_effect : float
            Pooled effect estimate

        Returns:
        --------
        PublicationBiasResult
        """
        effects = np.array([s.effect_size for s in studies])
        ses = np.array([s.se for s in studies])
        k = len(studies)

        # Egger's test
        # Regression of effect/SE on 1/SE
        precision = 1 / ses
        std_effects = effects / ses

        # Weighted linear regression
        X = np.column_stack([np.ones(k), precision])
        model = np.linalg.lstsq(X, std_effects, rcond=None)
        coeffs = model[0]

        egger_intercept = coeffs[0]
        residuals = std_effects - X @ coeffs
        mse = np.sum(residuals**2) / (k - 2)
        var_coef = mse * np.linalg.inv(X.T @ X)
        egger_se = np.sqrt(var_coef[0, 0])
        egger_t = egger_intercept / egger_se
        egger_p = 2 * (1 - stats.t.cdf(np.abs(egger_t), df=k-2))

        # Begg's test (rank correlation)
        # Standardized effect vs variance
        std_effects_centered = (effects - pooled_effect) / ses
        ranks_effect = stats.rankdata(std_effects_centered)
        ranks_var = stats.rankdata(ses**2)

        # Kendall's tau
        tau, begg_p = stats.kendalltau(ranks_effect, ranks_var)
        begg_z = tau / np.sqrt(2 * (2 * k + 5) / (9 * k * (k - 1)))

        # Trim and fill
        trim_fill_result = self._trim_and_fill(studies, pooled_effect)

        # Fail-safe N (Rosenthal)
        z_scores = effects / ses
        sum_z = np.sum(z_scores)
        fail_safe_n = int((sum_z / 1.645)**2 - k) if sum_z > 0 else 0
        fail_safe_n = max(0, fail_safe_n)

        summary = (
            f"Publication bias assessment. "
            f"Egger's test: intercept = {egger_intercept:.3f}, p = {egger_p:.4f}. "
            f"Begg's test: z = {begg_z:.3f}, p = {begg_p:.4f}. "
            f"Trim-and-fill: {trim_fill_result['added']} studies imputed. "
            f"Fail-safe N = {fail_safe_n}."
        )

        return PublicationBiasResult(
            egger_intercept=egger_intercept,
            egger_se=egger_se,
            egger_t=egger_t,
            egger_p_value=egger_p,
            begg_z=begg_z,
            begg_p_value=begg_p,
            trim_fill_added=trim_fill_result['added'],
            trim_fill_effect=trim_fill_result['effect'],
            trim_fill_ci_lower=trim_fill_result['ci_lower'],
            trim_fill_ci_upper=trim_fill_result['ci_upper'],
            fail_safe_n=fail_safe_n,
            summary_text=summary
        )

    def _trim_and_fill(
        self,
        studies: List[StudyData],
        pooled_effect: float
    ) -> Dict:
        """Duval and Tweedie's trim-and-fill method."""
        effects = np.array([s.effect_size for s in studies])
        ses = np.array([s.se for s in studies])
        k = len(studies)

        # Center effects around pooled estimate
        centered = effects - pooled_effect

        # Rank by absolute deviation
        ranks = stats.rankdata(np.abs(centered))

        # Estimate number of missing studies (R0 estimator)
        n_positive = np.sum(centered > 0)
        n_negative = np.sum(centered < 0)

        # If asymmetric, estimate missing
        if n_positive > n_negative:
            k0 = int((4 * n_positive - k) / 2)
            side = 'left'
        else:
            k0 = int((4 * n_negative - k) / 2)
            side = 'right'

        k0 = max(0, k0)

        if k0 == 0:
            return {
                'added': 0,
                'effect': pooled_effect,
                'ci_lower': pooled_effect - 1.96 * np.mean(ses),
                'ci_upper': pooled_effect + 1.96 * np.mean(ses)
            }

        # Impute missing studies (mirror the most extreme)
        if side == 'left':
            extreme_idx = np.argsort(effects)[-k0:]
        else:
            extreme_idx = np.argsort(effects)[:k0]

        # Create mirror studies
        imputed_effects = 2 * pooled_effect - effects[extreme_idx]
        imputed_ses = ses[extreme_idx]

        # Combine
        all_effects = np.concatenate([effects, imputed_effects])
        all_ses = np.concatenate([ses, imputed_ses])
        all_weights = 1 / all_ses**2

        # New pooled estimate
        new_pooled = np.sum(all_weights * all_effects) / np.sum(all_weights)
        new_se = 1 / np.sqrt(np.sum(all_weights))

        return {
            'added': k0,
            'effect': new_pooled,
            'ci_lower': new_pooled - 1.96 * new_se,
            'ci_upper': new_pooled + 1.96 * new_se
        }

    def subgroup_analysis(
        self,
        studies: List[StudyData],
        subgroup_var: str,
        subgroups: Dict[str, List[str]],
        model_type: str = 'random'
    ) -> SubgroupResult:
        """
        Subgroup meta-analysis.

        Parameters:
        -----------
        studies : List[StudyData]
            Prepared study data with study_id matching subgroups
        subgroup_var : str
            Name of subgrouping variable
        subgroups : Dict[str, List[str]]
            Dictionary mapping subgroup names to study IDs
        model_type : str
            'fixed' or 'random'

        Returns:
        --------
        SubgroupResult
        """
        subgroup_results = {}

        for subgroup_name, study_ids in subgroups.items():
            subgroup_studies = [s for s in studies if s.study_id in study_ids]

            if len(subgroup_studies) < 2:
                continue

            if model_type == 'fixed':
                result = self.fixed_effects(subgroup_studies)
            else:
                result = self.random_effects(subgroup_studies)

            subgroup_results[subgroup_name] = result

        # Test for between-group differences
        # Q_between = Q_total - sum(Q_within)
        all_result = self.random_effects(studies) if model_type == 'random' else self.fixed_effects(studies)
        q_total = all_result.heterogeneity.q_statistic

        q_within = sum(r.heterogeneity.q_statistic for r in subgroup_results.values())

        q_between = q_total - q_within
        df_between = len(subgroup_results) - 1
        p_between = 1 - stats.chi2.cdf(q_between, df_between) if df_between > 0 else 1.0

        summary = (
            f"Subgroup analysis by {subgroup_var}. "
            f"{len(subgroup_results)} subgroups analyzed. "
            f"Between-group Q = {q_between:.2f}, df = {df_between}, p = {p_between:.4f}."
        )

        return SubgroupResult(
            subgroup_name=subgroup_var,
            subgroup_effects=subgroup_results,
            between_group_q=q_between,
            between_group_df=df_between,
            between_group_p=p_between,
            summary_text=summary
        )

    def meta_regression(
        self,
        studies: List[StudyData],
        moderators: Dict[str, List[float]],
        method: str = 'random'
    ) -> MetaRegressionResult:
        """
        Meta-regression analysis.

        Parameters:
        -----------
        studies : List[StudyData]
            Prepared study data
        moderators : Dict[str, List[float]]
            Dictionary mapping moderator names to values (must match study order)
        method : str
            'fixed' or 'random'

        Returns:
        --------
        MetaRegressionResult
        """
        effects = np.array([s.effect_size for s in studies])
        variances = np.array([s.se**2 for s in studies])
        k = len(studies)

        # Prepare design matrix
        X_list = [np.ones(k)]  # Intercept
        mod_names = ['intercept']

        for name, values in moderators.items():
            X_list.append(np.array(values))
            mod_names.append(name)

        X = np.column_stack(X_list)
        p = X.shape[1]

        # Get tau-squared for random effects
        if method == 'random':
            tau_sq = self._dersimonian_laird_tau(effects, variances)
            weights = 1 / (variances + tau_sq)
        else:
            tau_sq = 0
            weights = 1 / variances

        # Weighted least squares
        W = np.diag(weights)
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ effects

        try:
            beta = np.linalg.solve(XtWX, XtWy)
            var_beta = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            # Singular matrix
            beta = np.zeros(p)
            var_beta = np.eye(p)

        # Standard errors and p-values
        coefficients = {}
        for i, name in enumerate(mod_names):
            se = np.sqrt(var_beta[i, i])
            z = beta[i] / se if se > 0 else 0
            p_val = 2 * (1 - stats.norm.cdf(np.abs(z)))

            coefficients[name] = {
                'coefficient': float(beta[i]),
                'se': float(se),
                'z': float(z),
                'p_value': float(p_val),
                'ci_lower': float(beta[i] - 1.96 * se),
                'ci_upper': float(beta[i] + 1.96 * se)
            }

        # R-squared (proportion of heterogeneity explained)
        # Compare tau-squared before and after
        fitted = X @ beta
        residuals = effects - fitted

        if method == 'random':
            tau_sq_residual = self._dersimonian_laird_tau(residuals, variances)
            tau_sq_total = self._dersimonian_laird_tau(effects, variances)

            if tau_sq_total > 0:
                r_squared = 1 - tau_sq_residual / tau_sq_total
            else:
                r_squared = 0
        else:
            ss_residual = np.sum(weights * residuals**2)
            ss_total = np.sum(weights * (effects - np.average(effects, weights=weights))**2)
            r_squared = 1 - ss_residual / ss_total if ss_total > 0 else 0

        r_squared = max(0, r_squared)

        # Model test (Q_model)
        q_model = np.sum(weights * (fitted - np.average(fitted, weights=weights))**2)
        df_model = p - 1
        p_model = 1 - stats.chi2.cdf(q_model, df_model) if df_model > 0 else 1.0

        summary = (
            f"Meta-regression ({method} effects). "
            f"k={k} studies, {len(moderators)} moderator(s). "
            f"R² = {r_squared*100:.1f}%. "
            f"Model Q = {q_model:.2f}, p = {p_model:.4f}."
        )

        return MetaRegressionResult(
            coefficients=coefficients,
            r_squared=r_squared,
            residual_heterogeneity=tau_sq_residual if method == 'random' else 0,
            model_q=q_model,
            model_df=df_model,
            model_p=p_model,
            n_studies=k,
            summary_text=summary
        )

    def forest_plot(
        self,
        result: MetaAnalysisResult,
        studies: List[StudyData],
        title: str = "Forest Plot",
        measure_label: str = "Effect Size",
        exponentiate: bool = False
    ):
        """
        Create a forest plot.

        Parameters:
        -----------
        result : MetaAnalysisResult
            Meta-analysis results
        studies : List[StudyData]
            Study data
        title : str
            Plot title
        measure_label : str
            Label for x-axis
        exponentiate : bool
            If True, display exp(effect) (for OR, RR)

        Returns:
        --------
        matplotlib figure
        """
        import matplotlib.pyplot as plt

        k = len(studies)
        fig, ax = plt.subplots(figsize=(12, max(6, k * 0.5)))

        # Transform if needed
        if exponentiate:
            transform = np.exp
            null_value = 1
        else:
            transform = lambda x: x
            null_value = 0

        y_positions = np.arange(k + 1, 0, -1)  # Studies + overall

        # Plot individual studies
        for i, study in enumerate(studies):
            effect = transform(study.effect_size)
            ci_lower = transform(study.effect_size - 1.96 * study.se)
            ci_upper = transform(study.effect_size + 1.96 * study.se)

            y = y_positions[i]

            # Effect size and CI
            ax.plot([ci_lower, ci_upper], [y, y], 'b-', linewidth=1.5)
            ax.plot(effect, y, 'bs', markersize=8)

            # Study label
            weight = result.study_weights.get(study.study_id, 0)
            ax.text(-0.05, y, f"{study.study_id}", ha='right', va='center',
                    fontsize=10, transform=ax.get_yaxis_transform())
            ax.text(1.05, y, f"{effect:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]  ({weight:.1f}%)",
                    ha='left', va='center', fontsize=9, transform=ax.get_yaxis_transform())

        # Plot overall effect
        overall_effect = transform(result.pooled_effect)
        overall_ci_lower = transform(result.pooled_ci_lower)
        overall_ci_upper = transform(result.pooled_ci_upper)

        y_overall = 0.5
        ax.plot([overall_ci_lower, overall_ci_upper], [y_overall, y_overall], 'r-', linewidth=2)
        ax.plot(overall_effect, y_overall, 'rD', markersize=12)

        ax.text(-0.05, y_overall, "Overall", ha='right', va='center',
                fontsize=11, fontweight='bold', transform=ax.get_yaxis_transform())
        ax.text(1.05, y_overall,
                f"{overall_effect:.2f} [{overall_ci_lower:.2f}, {overall_ci_upper:.2f}]",
                ha='left', va='center', fontsize=10, fontweight='bold',
                transform=ax.get_yaxis_transform())

        # Reference line
        ax.axvline(x=null_value, color='gray', linestyle='--', linewidth=1)

        # Formatting
        ax.set_xlabel(measure_label, fontsize=12)
        ax.set_yticks([])
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add heterogeneity info
        het = result.heterogeneity
        ax.text(0.5, -0.08,
                f"Heterogeneity: I² = {het.i_squared:.1f}%, τ² = {het.tau_squared:.4f}, Q = {het.q_statistic:.2f} (p = {het.q_p_value:.4f})",
                ha='center', va='top', fontsize=10, transform=ax.transAxes)

        plt.tight_layout()
        return fig

    def funnel_plot(
        self,
        studies: List[StudyData],
        pooled_effect: float,
        title: str = "Funnel Plot"
    ):
        """
        Create a funnel plot for publication bias assessment.

        Parameters:
        -----------
        studies : List[StudyData]
            Study data
        pooled_effect : float
            Pooled effect estimate
        title : str
            Plot title

        Returns:
        --------
        matplotlib figure
        """
        import matplotlib.pyplot as plt

        effects = np.array([s.effect_size for s in studies])
        ses = np.array([s.se for s in studies])

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot studies
        ax.scatter(effects, ses, s=80, c='#3498db', edgecolors='black', linewidth=0.5)

        # Pooled effect line
        ax.axvline(x=pooled_effect, color='red', linestyle='--', linewidth=1.5, label='Pooled effect')

        # Funnel (pseudo 95% CI)
        max_se = ses.max() * 1.1
        se_range = np.linspace(0.01, max_se, 100)
        lower_bound = pooled_effect - 1.96 * se_range
        upper_bound = pooled_effect + 1.96 * se_range

        ax.fill_betweenx(se_range, lower_bound, upper_bound, alpha=0.2, color='gray')
        ax.plot(lower_bound, se_range, 'gray', linestyle='-', linewidth=1)
        ax.plot(upper_bound, se_range, 'gray', linestyle='-', linewidth=1)

        # Invert y-axis (smaller SE at top)
        ax.invert_yaxis()

        ax.set_xlabel('Effect Size', fontsize=12)
        ax.set_ylabel('Standard Error', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')

        plt.tight_layout()
        return fig
