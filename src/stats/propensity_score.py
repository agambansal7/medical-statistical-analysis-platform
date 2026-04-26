"""
Propensity Score Analysis Module
================================
Comprehensive propensity score methods for causal inference in observational studies.

Methods:
- Propensity score estimation (logistic regression, GBM)
- Propensity score matching (nearest neighbor, caliper, optimal)
- Inverse probability weighting (IPW)
- Doubly robust estimation (AIPW)
- Covariate balance assessment
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PropensityScoreResult:
    """Result container for propensity score estimation."""
    propensity_scores: np.ndarray
    treatment_variable: str
    covariates: List[str]
    model_type: str
    auc: float
    brier_score: float
    n_treated: int
    n_control: int
    summary_text: str


@dataclass
class MatchingResult:
    """Result container for propensity score matching."""
    matched_data: pd.DataFrame
    n_treated_matched: int
    n_control_matched: int
    n_unmatched: int
    matching_method: str
    caliper: Optional[float]
    balance_before: Dict[str, Dict]
    balance_after: Dict[str, Dict]
    standardized_mean_diff_before: Dict[str, float]
    standardized_mean_diff_after: Dict[str, float]
    summary_text: str


@dataclass
class IPWResult:
    """Result container for inverse probability weighting."""
    weights: np.ndarray
    ate: float  # Average Treatment Effect
    ate_se: float
    ate_ci_lower: float
    ate_ci_upper: float
    att: float  # Average Treatment Effect on Treated
    att_se: float
    att_ci_lower: float
    att_ci_upper: float
    effective_sample_size: float
    max_weight: float
    weight_truncation_applied: bool
    summary_text: str


@dataclass
class DoublyRobustResult:
    """Result container for doubly robust estimation."""
    ate: float
    ate_se: float
    ate_ci_lower: float
    ate_ci_upper: float
    ate_p_value: float
    outcome_model_r2: float
    propensity_model_auc: float
    n_observations: int
    summary_text: str


@dataclass
class BalanceAssessment:
    """Result container for covariate balance assessment."""
    variables: List[str]
    smd_before: Dict[str, float]
    smd_after: Dict[str, float]
    variance_ratio_before: Dict[str, float]
    variance_ratio_after: Dict[str, float]
    ks_statistic_before: Dict[str, float]
    ks_statistic_after: Dict[str, float]
    balance_achieved: bool
    summary_text: str


class PropensityScoreAnalysis:
    """
    Comprehensive propensity score analysis for causal inference.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def estimate_propensity_scores(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        covariates: List[str],
        method: str = 'logistic'
    ) -> PropensityScoreResult:
        """
        Estimate propensity scores using logistic regression or gradient boosting.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        treatment_col : str
            Binary treatment variable (0/1)
        covariates : List[str]
            Covariates for propensity score model
        method : str
            'logistic' or 'gbm' (gradient boosting)

        Returns:
        --------
        PropensityScoreResult
        """
        # Prepare data
        df_clean = df.dropna(subset=[treatment_col] + covariates)
        X = df_clean[covariates].values
        y = df_clean[treatment_col].values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit model
        if method == 'logistic':
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif method == 'gbm':
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, random_state=42
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        model.fit(X_scaled, y)

        # Get propensity scores
        ps = model.predict_proba(X_scaled)[:, 1]

        # Calculate AUC
        from sklearn.metrics import roc_auc_score, brier_score_loss
        auc = roc_auc_score(y, ps)
        brier = brier_score_loss(y, ps)

        n_treated = int(y.sum())
        n_control = len(y) - n_treated

        summary = (
            f"Propensity scores estimated using {method}. "
            f"N={len(y)} (Treated={n_treated}, Control={n_control}). "
            f"Model AUC={auc:.3f}, Brier Score={brier:.3f}."
        )

        return PropensityScoreResult(
            propensity_scores=ps,
            treatment_variable=treatment_col,
            covariates=covariates,
            model_type=method,
            auc=auc,
            brier_score=brier,
            n_treated=n_treated,
            n_control=n_control,
            summary_text=summary
        )

    def match(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        propensity_scores: np.ndarray,
        covariates: List[str],
        method: str = 'nearest',
        caliper: Optional[float] = 0.2,
        ratio: int = 1,
        replace: bool = False
    ) -> MatchingResult:
        """
        Perform propensity score matching.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data (must match propensity_scores length)
        treatment_col : str
            Binary treatment variable
        propensity_scores : np.ndarray
            Estimated propensity scores
        covariates : List[str]
            Covariates for balance assessment
        method : str
            'nearest' (nearest neighbor) or 'optimal'
        caliper : float
            Maximum distance for matching (in SD of PS). None for no caliper.
        ratio : int
            Number of controls to match per treated unit
        replace : bool
            Whether to allow matching with replacement

        Returns:
        --------
        MatchingResult
        """
        df_match = df.copy()
        df_match['_ps'] = propensity_scores
        df_match['_original_index'] = df_match.index

        # Separate treated and control
        treated = df_match[df_match[treatment_col] == 1].copy()
        control = df_match[df_match[treatment_col] == 0].copy()

        # Calculate caliper in PS units
        if caliper is not None:
            caliper_value = caliper * df_match['_ps'].std()
        else:
            caliper_value = np.inf

        # Balance before matching
        balance_before = self._calculate_balance(df_match, treatment_col, covariates)
        smd_before = self._calculate_smd(df_match, treatment_col, covariates)

        # Nearest neighbor matching
        matched_control_indices = []
        matched_treated_indices = []
        control_available = control.index.tolist()

        for idx in treated.index:
            if not control_available and not replace:
                break

            treated_ps = treated.loc[idx, '_ps']

            if replace:
                candidates = control
            else:
                candidates = control.loc[control_available]

            # Calculate distances
            distances = np.abs(candidates['_ps'].values - treated_ps)

            # Find matches within caliper
            valid_matches = np.where(distances <= caliper_value)[0]

            if len(valid_matches) == 0:
                continue

            # Get best matches
            n_matches = min(ratio, len(valid_matches))
            best_match_indices = valid_matches[np.argsort(distances[valid_matches])[:n_matches]]

            for match_idx in best_match_indices:
                control_idx = candidates.index[match_idx]
                matched_control_indices.append(control_idx)
                matched_treated_indices.append(idx)

                if not replace:
                    control_available.remove(control_idx)

        # Create matched dataset
        matched_treated = treated.loc[list(set(matched_treated_indices))]
        matched_control = control.loc[matched_control_indices]

        matched_data = pd.concat([matched_treated, matched_control], ignore_index=True)
        matched_data = matched_data.drop(columns=['_ps', '_original_index'])

        # Balance after matching
        balance_after = self._calculate_balance(matched_data, treatment_col, covariates)
        smd_after = self._calculate_smd(matched_data, treatment_col, covariates)

        n_unmatched = len(treated) - len(set(matched_treated_indices))

        summary = (
            f"Propensity score matching ({method}, caliper={caliper}). "
            f"Matched {len(set(matched_treated_indices))} treated to {len(matched_control_indices)} controls. "
            f"{n_unmatched} treated units unmatched. "
            f"Mean SMD reduced from {np.mean(list(smd_before.values())):.3f} to {np.mean(list(smd_after.values())):.3f}."
        )

        return MatchingResult(
            matched_data=matched_data,
            n_treated_matched=len(set(matched_treated_indices)),
            n_control_matched=len(matched_control_indices),
            n_unmatched=n_unmatched,
            matching_method=method,
            caliper=caliper,
            balance_before=balance_before,
            balance_after=balance_after,
            standardized_mean_diff_before=smd_before,
            standardized_mean_diff_after=smd_after,
            summary_text=summary
        )

    def inverse_probability_weighting(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        propensity_scores: np.ndarray,
        estimand: str = 'ATE',
        stabilized: bool = True,
        truncate_weights: Optional[Tuple[float, float]] = (0.01, 0.99)
    ) -> IPWResult:
        """
        Estimate treatment effects using inverse probability weighting.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        treatment_col : str
            Binary treatment variable
        outcome_col : str
            Outcome variable
        propensity_scores : np.ndarray
            Estimated propensity scores
        estimand : str
            'ATE' (average treatment effect) or 'ATT' (on treated)
        stabilized : bool
            Whether to use stabilized weights
        truncate_weights : Tuple[float, float]
            Percentiles for weight truncation (None for no truncation)

        Returns:
        --------
        IPWResult
        """
        df_ipw = df.copy()
        df_ipw['_ps'] = propensity_scores
        df_ipw = df_ipw.dropna(subset=[treatment_col, outcome_col, '_ps'])

        A = df_ipw[treatment_col].values  # Treatment
        Y = df_ipw[outcome_col].values    # Outcome
        ps = df_ipw['_ps'].values         # Propensity scores

        # Calculate weights
        if estimand == 'ATE':
            # ATE weights
            weights = A / ps + (1 - A) / (1 - ps)
            if stabilized:
                p_treat = A.mean()
                weights = A * p_treat / ps + (1 - A) * (1 - p_treat) / (1 - ps)
        elif estimand == 'ATT':
            # ATT weights
            weights = A + (1 - A) * ps / (1 - ps)
            if stabilized:
                p_treat = A.mean()
                weights = A + (1 - A) * ps / (1 - ps) * (1 - p_treat) / p_treat
        else:
            raise ValueError(f"Unknown estimand: {estimand}")

        # Truncate weights
        truncated = False
        if truncate_weights is not None:
            lower = np.percentile(weights, truncate_weights[0] * 100)
            upper = np.percentile(weights, truncate_weights[1] * 100)
            weights = np.clip(weights, lower, upper)
            truncated = True

        # Calculate weighted means
        weighted_y1 = np.sum(A * weights * Y) / np.sum(A * weights)
        weighted_y0 = np.sum((1 - A) * weights * Y) / np.sum((1 - A) * weights)

        ate = weighted_y1 - weighted_y0

        # Bootstrap for standard errors
        n_bootstrap = 1000
        ate_boots = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(len(Y), size=len(Y), replace=True)
            A_b, Y_b, w_b = A[idx], Y[idx], weights[idx]

            wy1 = np.sum(A_b * w_b * Y_b) / np.sum(A_b * w_b) if np.sum(A_b * w_b) > 0 else 0
            wy0 = np.sum((1 - A_b) * w_b * Y_b) / np.sum((1 - A_b) * w_b) if np.sum((1 - A_b) * w_b) > 0 else 0
            ate_boots.append(wy1 - wy0)

        ate_se = np.std(ate_boots)
        ate_ci = np.percentile(ate_boots, [2.5, 97.5])

        # ATT calculation
        att_weights = A + (1 - A) * ps / (1 - ps)
        if truncate_weights is not None:
            att_weights = np.clip(att_weights, lower, upper)

        y1_mean = np.mean(Y[A == 1])
        y0_weighted = np.sum((1 - A) * att_weights * Y) / np.sum((1 - A) * att_weights)
        att = y1_mean - y0_weighted

        # Bootstrap for ATT SE
        att_boots = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(Y), size=len(Y), replace=True)
            A_b, Y_b, ps_b = A[idx], Y[idx], ps[idx]
            att_w = A_b + (1 - A_b) * ps_b / (1 - ps_b)

            y1_m = np.mean(Y_b[A_b == 1]) if np.sum(A_b) > 0 else 0
            y0_w = np.sum((1 - A_b) * att_w * Y_b) / np.sum((1 - A_b) * att_w) if np.sum((1 - A_b) * att_w) > 0 else 0
            att_boots.append(y1_m - y0_w)

        att_se = np.std(att_boots)
        att_ci = np.percentile(att_boots, [2.5, 97.5])

        # Effective sample size
        ess = (np.sum(weights))**2 / np.sum(weights**2)

        summary = (
            f"IPW estimation (estimand={estimand}, stabilized={stabilized}). "
            f"ATE = {ate:.4f} (95% CI: {ate_ci[0]:.4f}, {ate_ci[1]:.4f}). "
            f"ATT = {att:.4f} (95% CI: {att_ci[0]:.4f}, {att_ci[1]:.4f}). "
            f"Effective sample size = {ess:.1f}. Max weight = {weights.max():.2f}."
        )

        return IPWResult(
            weights=weights,
            ate=ate,
            ate_se=ate_se,
            ate_ci_lower=ate_ci[0],
            ate_ci_upper=ate_ci[1],
            att=att,
            att_se=att_se,
            att_ci_lower=att_ci[0],
            att_ci_upper=att_ci[1],
            effective_sample_size=ess,
            max_weight=weights.max(),
            weight_truncation_applied=truncated,
            summary_text=summary
        )

    def doubly_robust_estimation(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: List[str],
        propensity_method: str = 'logistic'
    ) -> DoublyRobustResult:
        """
        Doubly robust (Augmented IPW) estimation of treatment effects.

        Combines outcome regression and propensity score weighting for
        robustness - consistent if either model is correctly specified.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        treatment_col : str
            Binary treatment variable
        outcome_col : str
            Outcome variable
        covariates : List[str]
            Covariates for both models
        propensity_method : str
            Method for propensity score estimation

        Returns:
        --------
        DoublyRobustResult
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        df_clean = df.dropna(subset=[treatment_col, outcome_col] + covariates)

        A = df_clean[treatment_col].values
        Y = df_clean[outcome_col].values
        X = df_clean[covariates].values
        X_scaled = self.scaler.fit_transform(X)

        n = len(Y)

        # Step 1: Estimate propensity scores
        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        ps_model.fit(X_scaled, A)
        ps = ps_model.predict_proba(X_scaled)[:, 1]
        ps = np.clip(ps, 0.01, 0.99)  # Trim extreme values

        from sklearn.metrics import roc_auc_score
        ps_auc = roc_auc_score(A, ps)

        # Step 2: Fit outcome models
        # Model for treated
        X_treated = X_scaled[A == 1]
        Y_treated = Y[A == 1]
        outcome_model_1 = LinearRegression()
        outcome_model_1.fit(X_treated, Y_treated)

        # Model for control
        X_control = X_scaled[A == 0]
        Y_control = Y[A == 0]
        outcome_model_0 = LinearRegression()
        outcome_model_0.fit(X_control, Y_control)

        # Predict potential outcomes for everyone
        mu1 = outcome_model_1.predict(X_scaled)  # E[Y(1)|X]
        mu0 = outcome_model_0.predict(X_scaled)  # E[Y(0)|X]

        # Calculate R-squared for outcome model
        Y_pred = A * mu1 + (1 - A) * mu0
        outcome_r2 = r2_score(Y, Y_pred)

        # Step 3: AIPW estimator
        # ψ(1) = μ1(X) + A(Y - μ1(X)) / ps
        # ψ(0) = μ0(X) + (1-A)(Y - μ0(X)) / (1-ps)

        psi_1 = mu1 + A * (Y - mu1) / ps
        psi_0 = mu0 + (1 - A) * (Y - mu0) / (1 - ps)

        # ATE = E[ψ(1) - ψ(0)]
        ate = np.mean(psi_1 - psi_0)

        # Influence function for variance estimation
        influence = psi_1 - psi_0 - ate
        ate_se = np.sqrt(np.var(influence) / n)

        # Confidence interval
        z = stats.norm.ppf(0.975)
        ate_ci_lower = ate - z * ate_se
        ate_ci_upper = ate + z * ate_se

        # P-value
        z_stat = ate / ate_se
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

        summary = (
            f"Doubly robust (AIPW) estimation. "
            f"ATE = {ate:.4f} (SE = {ate_se:.4f}, 95% CI: {ate_ci_lower:.4f} to {ate_ci_upper:.4f}, p = {p_value:.4f}). "
            f"Propensity model AUC = {ps_auc:.3f}. Outcome model R² = {outcome_r2:.3f}."
        )

        return DoublyRobustResult(
            ate=ate,
            ate_se=ate_se,
            ate_ci_lower=ate_ci_lower,
            ate_ci_upper=ate_ci_upper,
            ate_p_value=p_value,
            outcome_model_r2=outcome_r2,
            propensity_model_auc=ps_auc,
            n_observations=n,
            summary_text=summary
        )

    def assess_balance(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        covariates: List[str],
        weights: Optional[np.ndarray] = None
    ) -> BalanceAssessment:
        """
        Comprehensive covariate balance assessment.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        treatment_col : str
            Binary treatment variable
        covariates : List[str]
            Covariates to assess
        weights : np.ndarray, optional
            IPW weights (for weighted balance)

        Returns:
        --------
        BalanceAssessment
        """
        smd = {}
        var_ratio = {}
        ks_stat = {}

        treated = df[df[treatment_col] == 1]
        control = df[df[treatment_col] == 0]

        for var in covariates:
            if var not in df.columns:
                continue

            t_vals = treated[var].dropna().values
            c_vals = control[var].dropna().values

            # Standardized mean difference
            pooled_std = np.sqrt((np.var(t_vals) + np.var(c_vals)) / 2)
            if pooled_std > 0:
                smd[var] = (np.mean(t_vals) - np.mean(c_vals)) / pooled_std
            else:
                smd[var] = 0

            # Variance ratio
            if np.var(c_vals) > 0:
                var_ratio[var] = np.var(t_vals) / np.var(c_vals)
            else:
                var_ratio[var] = np.inf

            # KS statistic
            ks_stat[var] = stats.ks_2samp(t_vals, c_vals).statistic

        # Check if balance achieved (all SMD < 0.1)
        balance_achieved = all(abs(s) < 0.1 for s in smd.values())

        summary = (
            f"Balance assessment for {len(covariates)} covariates. "
            f"Mean absolute SMD = {np.mean(np.abs(list(smd.values()))):.3f}. "
            f"{'Balance achieved (all SMD < 0.1).' if balance_achieved else 'Imbalance detected.'}"
        )

        return BalanceAssessment(
            variables=covariates,
            smd_before=smd,
            smd_after={},  # Filled after matching/weighting
            variance_ratio_before=var_ratio,
            variance_ratio_after={},
            ks_statistic_before=ks_stat,
            ks_statistic_after={},
            balance_achieved=balance_achieved,
            summary_text=summary
        )

    def _calculate_balance(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        covariates: List[str]
    ) -> Dict[str, Dict]:
        """Calculate balance statistics for each covariate."""
        balance = {}
        treated = df[df[treatment_col] == 1]
        control = df[df[treatment_col] == 0]

        for var in covariates:
            if var not in df.columns:
                continue
            balance[var] = {
                'treated_mean': treated[var].mean(),
                'control_mean': control[var].mean(),
                'treated_std': treated[var].std(),
                'control_std': control[var].std()
            }

        return balance

    def _calculate_smd(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        covariates: List[str]
    ) -> Dict[str, float]:
        """Calculate standardized mean differences."""
        smd = {}
        treated = df[df[treatment_col] == 1]
        control = df[df[treatment_col] == 0]

        for var in covariates:
            if var not in df.columns:
                continue
            t_mean = treated[var].mean()
            c_mean = control[var].mean()
            pooled_std = np.sqrt((treated[var].var() + control[var].var()) / 2)

            if pooled_std > 0:
                smd[var] = (t_mean - c_mean) / pooled_std
            else:
                smd[var] = 0

        return smd

    def plot_balance(
        self,
        balance_before: Dict[str, float],
        balance_after: Dict[str, float],
        title: str = "Covariate Balance"
    ):
        """
        Create a Love plot showing covariate balance before and after matching.

        Returns matplotlib figure.
        """
        import matplotlib.pyplot as plt

        variables = list(balance_before.keys())
        smd_before = [balance_before[v] for v in variables]
        smd_after = [balance_after.get(v, 0) for v in variables]

        fig, ax = plt.subplots(figsize=(10, max(6, len(variables) * 0.4)))

        y_pos = np.arange(len(variables))

        ax.scatter(smd_before, y_pos, marker='o', s=100, label='Before Matching',
                   color='#e74c3c', alpha=0.7)
        ax.scatter(smd_after, y_pos, marker='s', s=100, label='After Matching',
                   color='#27ae60', alpha=0.7)

        # Reference lines
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.axvline(x=-0.1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=0.1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvspan(-0.1, 0.1, alpha=0.1, color='green', label='Acceptable range')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(variables)
        ax.set_xlabel('Standardized Mean Difference')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='best')

        plt.tight_layout()
        return fig
